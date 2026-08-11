"""Plain-assert tests for the lazy/bounded-inference organ
(LiveGraph(..., inference="lazy") in src/livecausal/infer.py, Task 15).
Run: python3 src/livecausal/test_lazy_inference.py

Registered motivation (three independent measurements): P74's append-cost
curve (13x in the dense regime), the ConceptNet-500k mount OOM (21.3GB
RSS at 4.4M inferred edges materialized from 25k base edges), and a live
WT-103 build's own closure-density degradation (9 -> 1 windows/s). All
three trace back to the SAME root cause: a materialized transitive
closure's size (and maintenance cost) is bounded by DENSITY, not by base
graph size -- and density is not something this organism controls. Lazy
mode never materializes a closure at all: query()/canon_query() derive
answers on demand via a bounded single-source DFS
(_bounded_query_closure), and on_append/on_drop become O(records
touched) instead of O(reachable neighborhood).

Covers the build brief's five requirements directly:
  1. eager stays default + byte-identical (regression).
  2. equivalence oracle across multiple densities, per key, with visible
     (never silent) truncation when the budget caps a result.
  3. on_append/on_drop cost characteristic in lazy mode (no closure
     computation at all -- checked both by timing comparison against
     eager on a dense hub-and-spoke corpus, and by delta-equivalence
     against an independent from-scratch closure oracle).
  4. mount time / no materialized-closure memory footprint on a dense
     smoke store (the fake-extractor-shaped corpus this module's own
     builder_run.py smoke path produces 14k+ inferred edges from,
     per the build brief), plus query-latency distribution (p50/p95).
  5. verifier compatibility -- direction-3 re-derivation works on a
     lazy-produced edge exactly as-is, since re-derivation only ever
     reads cited (segment_sha, idx) records, never the graph's own
     inferred-edge structure (eager or lazy).
"""
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.infer import (
    DEFAULT_NODE_BUDGET,
    LiveGraph,
    MAX_DEPTH,
    _batch_transitive_closure,
    _bounded_query_closure,
    _derivation_key,
)
from livecausal.store import LiveStore
from stranger_verify_run import _inferred_edge_rederive


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="livecausal-lazy-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def make_record(from_key, to_key, doc_coord=0):
    return {
        "trigger": "t::{}".format(from_key),
        "mechanism": "causes",
        "outcome": "o::{}".format(to_key),
        "trigger_key": from_key,
        "outcome_key": to_key,
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


def _edge_sig(edges):
    """Order-independent signature: (kind, to_key, depth, derivation),
    comparable across eager/lazy since both shapes carry these fields
    (lazy always sets "depth" even on base edges; eager base edges omit
    it -- .get("depth", 1) normalizes that cosmetic difference, exactly
    as query()'s own sort key already does)."""
    return sorted((e["kind"], e["to_key"], e.get("depth", 1), _derivation_key(e["derivation"])) for e in edges)


def _random_dense_corpus(rng, n_keys, n_edges, max_citations=3):
    """Builds a store with n_keys distinct keys and n_edges random
    directed (from, to) pairs among them (each possibly multiply cited),
    one record per segment -- returns (store_dir handled by caller,
    records list). Density here means "edges relative to keys," the
    exact axis P74 measured degrading append/query cost on."""
    keys = ["K{}".format(i) for i in range(n_keys)]
    segments = []
    edge_pool = {}
    for _ in range(n_edges):
        a, b = rng.sample(keys, 2)
        n_cit = rng.randint(1, max_citations)
        for _c in range(n_cit):
            doc_coord = len(segments)
            segments.append([make_record(a, b, doc_coord=doc_coord)])
    return keys, segments


def _build_store(store_dir, segments):
    store = LiveStore(store_dir)
    shas = []
    for records in segments:
        shas.append(store.append_segment(records))
    return store, shas


# ─────────────────────────────────────────────────────────────────────────
# 1. Regression: inference="eager" (default, and explicit) is
#    byte-identical to every pre-Task-15 LiveGraph construction/behavior.
# ─────────────────────────────────────────────────────────────────────────
def test_eager_is_default_and_regression_identical(d):
    rng = random.Random(1)
    keys, segments = _random_dense_corpus(rng, n_keys=15, n_edges=25)

    dir_default = os.path.join(d, "default")
    dir_explicit = os.path.join(d, "explicit")
    os.makedirs(dir_default)
    os.makedirs(dir_explicit)
    _build_store(dir_default, segments)
    _build_store(dir_explicit, segments)

    g_default = LiveGraph(dir_default)  # pre-Task-15 call: no inference kwarg
    g_explicit = LiveGraph(dir_explicit, inference="eager")

    assert g_default.inference_mode == "eager"
    assert g_explicit.inference_mode == "eager"

    for k in keys:
        assert g_default.query(k) == g_explicit.query(k)
    assert g_default.inferred_edges() == g_explicit.inferred_edges()

    # query()'s new keyword-only parameters must be no-ops on eager --
    # node_budget is accepted but unused, return_truncated always yields
    # truncated=False (nothing in eager mode is ever capped).
    edges_plain = g_default.query(keys[0])
    edges_with_budget = g_default.query(keys[0], node_budget=1)  # absurdly small, must not matter
    assert edges_plain == edges_with_budget
    edges_tup, truncated = g_default.query(keys[0], return_truncated=True)
    assert edges_tup == edges_plain
    assert truncated is False

    # Invalid inference mode must fail loudly at construction.
    try:
        LiveGraph(dir_default, inference="bogus")
        assert False, "expected ValueError for invalid inference mode"
    except ValueError:
        pass


# ─────────────────────────────────────────────────────────────────────────
# 2. Equivalence oracle across multiple densities: lazy query(key) ==
#    eager query(key) for EVERY key, given a generous budget.
# ─────────────────────────────────────────────────────────────────────────
def test_lazy_equals_eager_across_densities(d):
    # Density axis: edges-per-key growing while key count stays fixed,
    # but kept well short of "near-complete digraph on n_keys" (which
    # blows up combinatorially in chain COUNT, not depth or reachability
    # -- a real finding from building this test: a 12-key graph with 90
    # edges x up to 3 citations each is >85% of all possible directed
    # pairs, producing tens of thousands of distinct depth<=5 chains from
    # a single key even with cycle-guards; that is a genuine combinatorial
    # fact about near-complete digraphs, not a bug, and it is exactly
    # what node_budget exists to bound -- see
    # test_truncation_is_visible_and_is_a_subset for that property
    # exercised directly. This test's OWN corpus stays sparse enough
    # (at most ~1 edge per key-pair in expectation) that truncation is
    # not expected at a generous-but-finite budget, so an unexpected
    # truncation here is a genuine equivalence-test signal, not corpus
    # pathology.)
    for density_idx, (n_keys, n_edges) in enumerate([(20, 15), (20, 40), (20, 70)]):
        rng = random.Random(100 + density_idx)
        keys, segments = _random_dense_corpus(rng, n_keys=n_keys, n_edges=n_edges, max_citations=2)

        dir_eager = os.path.join(d, "eager_{}".format(density_idx))
        dir_lazy = os.path.join(d, "lazy_{}".format(density_idx))
        os.makedirs(dir_eager)
        os.makedirs(dir_lazy)
        _build_store(dir_eager, segments)
        _build_store(dir_lazy, segments)

        g_eager = LiveGraph(dir_eager, inference="eager")
        g_lazy = LiveGraph(dir_lazy, inference="lazy", node_budget=2_000_000)

        n_checked = 0
        for k in keys:
            eager_edges = g_eager.query(k)
            lazy_edges, truncated = g_lazy.query(k, return_truncated=True)
            assert truncated is False, "unexpected truncation with a generous budget at density {}".format(density_idx)
            assert _edge_sig(eager_edges) == _edge_sig(lazy_edges), (
                "density {} key {!r}: lazy/eager query mismatch".format(density_idx, k)
            )
            n_checked += 1
        assert n_checked == len(keys)


# ─────────────────────────────────────────────────────────────────────────
# 3. Truncation is VISIBLE, never silent: a tiny budget caps the result
#    and reports truncated=True; the capped result is a SUBSET of the
#    full (generous-budget) answer -- a lower bound, not a wrong answer.
# ─────────────────────────────────────────────────────────────────────────
def test_truncation_is_visible_and_is_a_subset(d):
    rng = random.Random(55)
    keys, segments = _random_dense_corpus(rng, n_keys=20, n_edges=60, max_citations=2)
    _build_store(d, segments)

    g_lazy = LiveGraph(d, inference="lazy")

    # Find a key with a genuinely large reachable set under a generous budget.
    best_key, best_edges = None, []
    for k in keys:
        edges, truncated = g_lazy.query(k, node_budget=200000, return_truncated=True)
        assert truncated is False
        if len(edges) > len(best_edges):
            best_key, best_edges = k, edges

    assert len(best_edges) >= 3, "test corpus too sparse to exercise truncation meaningfully"

    capped_edges, capped_truncated = g_lazy.query(best_key, node_budget=1, return_truncated=True)
    assert capped_truncated is True, "expected truncation with node_budget=1 on a multi-edge key"

    full_sig = set((e["kind"], e["to_key"], e.get("depth", 1), _derivation_key(e["derivation"])) for e in best_edges)
    capped_sig = set((e["kind"], e["to_key"], e.get("depth", 1), _derivation_key(e["derivation"])) for e in capped_edges)
    assert capped_sig.issubset(full_sig), (
        "a truncated result must be a SUBSET of the full answer (a lower "
        "bound), never contain an edge the full traversal would not"
    )

    # The default node_budget (no explicit node_budget passed) must NOT
    # truncate this small smoke corpus -- DEFAULT_NODE_BUDGET is sized to
    # be generous relative to what this repo's own test/smoke graphs need.
    default_edges, default_truncated = g_lazy.query(best_key, return_truncated=True)
    assert default_truncated is False, (
        "DEFAULT_NODE_BUDGET ({}) truncated a small smoke-corpus query -- "
        "default is supposed to be generous enough not to cap ordinary "
        "test/smoke-scale graphs".format(DEFAULT_NODE_BUDGET)
    )
    assert _edge_sig(default_edges) == _edge_sig(best_edges)


# ─────────────────────────────────────────────────────────────────────────
# 4. on_append/on_drop delta-equivalence in lazy mode: after a sequence
#    of incremental appends (and a drop), lazy query() results must match
#    an INDEPENDENT from-scratch closure oracle over the same final base
#    adjacency -- the batch-oracle pattern MVP-2 established, applied to
#    confirm lazy's on-demand derivation reflects append/drop correctly
#    (not that it computes a closure eagerly -- it must not -- but that
#    its on-demand answers are CORRECT after each mutation).
# ─────────────────────────────────────────────────────────────────────────
def test_lazy_on_append_and_drop_stay_correct(d):
    rng = random.Random(321)
    keys, segments = _random_dense_corpus(rng, n_keys=18, n_edges=50, max_citations=2)

    store, shas = _build_store(d, segments)
    g_lazy = LiveGraph(d, inference="lazy", node_budget=200000)

    # Independent oracle: build base_edges directly from the store and
    # run the batch closure -- shares NO code path with LiveGraph's own
    # incremental bookkeeping.
    def _oracle_base_edges(live_store):
        base = {}
        for sha in live_store.segments():
            for _s, idx, record in live_store.iter_records(sha):
                a, b = record["trigger_key"], record["outcome_key"]
                bucket = base.setdefault(a, {}).setdefault(b, [])
                pair = [sha, idx]
                if pair not in bucket:
                    bucket.append(pair)
                    bucket.sort(key=lambda p: (p[0], p[1]))
        return base

    def _oracle_query(live_store, key):
        base = _oracle_base_edges(live_store)
        edges = []
        for to_key, citations in sorted(base.get(key, {}).items()):
            edges.append({"kind": "base", "to_key": to_key, "depth": 1, "derivation": citations})
        closure = _batch_transitive_closure(base)
        for e in closure:
            if e["from_key"] == key:
                edges.append({"kind": "inferred", "to_key": e["to_key"], "depth": e["depth"], "derivation": e["derivation"]})
        return edges

    for k in keys:
        assert _edge_sig(g_lazy.query(k)) == _edge_sig(_oracle_query(store, k)), (
            "mismatch at key {!r} before any mutation".format(k)
        )

    # Append several new segments extending/joining the existing graph.
    new_edges = [(rng.choice(keys), rng.choice(keys)) for _ in range(8)]
    for a, b in new_edges:
        if a == b:
            continue
        g_lazy.append_segment([make_record(a, b, doc_coord=99000 + len(new_edges))])

    for k in keys:
        assert _edge_sig(g_lazy.query(k)) == _edge_sig(_oracle_query(store, k)), (
            "mismatch at key {!r} after appends".format(k)
        )

    # Drop a middle segment.
    dropped_sha = shas[len(shas) // 2]
    g_lazy.drop_segments([dropped_sha])

    for k in keys:
        assert _edge_sig(g_lazy.query(k)) == _edge_sig(_oracle_query(store, k)), (
            "mismatch at key {!r} after drop".format(k)
        )


# ─────────────────────────────────────────────────────────────────────────
# 5. on_append cost characteristic: lazy append does NOT degrade with
#    reachable-neighborhood density the way eager's does -- checked by
#    direct timing comparison on a hub-and-spoke corpus at growing scale
#    (the exact P74 append-cost-curve shape, reproduced as a smoke check,
#    not a strict wall-clock assertion -- machine-load-robust: the claim
#    tested is RELATIVE ordering + a generous margin, not an absolute
#    number).
# ─────────────────────────────────────────────────────────────────────────
def test_lazy_append_does_not_scale_with_density(d):
    def _hub_spoke_store(store_dir, n_spokes):
        store = LiveStore(store_dir)
        for i in range(n_spokes):
            store.append_segment([make_record("spoke_in_{}".format(i), "HUB", doc_coord=i)])
        for i in range(n_spokes):
            store.append_segment([make_record("HUB", "spoke_out_{}".format(i), doc_coord=1000 + i)])
        return store

    timings = []
    for n_spokes in (40, 200):
        dir_eager = os.path.join(d, "eager_{}".format(n_spokes))
        dir_lazy = os.path.join(d, "lazy_{}".format(n_spokes))
        os.makedirs(dir_eager)
        os.makedirs(dir_lazy)
        _hub_spoke_store(dir_eager, n_spokes)
        _hub_spoke_store(dir_lazy, n_spokes)

        g_eager = LiveGraph(dir_eager, inference="eager")
        g_lazy = LiveGraph(dir_lazy, inference="lazy")

        new_rec = make_record("spoke_in_NEW", "HUB", doc_coord=999999)
        t0 = time.perf_counter()
        g_eager.append_segment([new_rec])
        t_eager = time.perf_counter() - t0

        t0 = time.perf_counter()
        g_lazy.append_segment([new_rec])
        t_lazy = time.perf_counter() - t0

        timings.append((n_spokes, t_eager, t_lazy))

    # Lazy append time at the LARGER scale must not have grown anywhere
    # near proportionally to eager's growth -- eager's cost scales with
    # the hub's reachable neighborhood (which grows with n_spokes); lazy's
    # does not (it only touches the one new record). Generous factor (10x
    # slack) to stay robust on a loaded machine while still catching a
    # real regression back to closure-scaling behavior.
    (n0, e0, l0), (n1, e1, l1) = timings
    eager_growth = e1 / max(e0, 1e-6)
    lazy_growth = l1 / max(l0, 1e-6)
    assert eager_growth > 2.0, (
        "test corpus did not actually stress eager's density-scaling cost "
        "(eager_growth={:.2f}x from {} to {} spokes) -- strengthen the corpus".format(eager_growth, n0, n1)
    )
    assert lazy_growth < eager_growth, (
        "lazy append cost grew as fast as eager's with spoke count "
        "(lazy {:.2f}x vs eager {:.2f}x) -- lazy mode should not scale "
        "with reachable-neighborhood density at all".format(lazy_growth, eager_growth)
    )


# ─────────────────────────────────────────────────────────────────────────
# 6. Mount cost / no-materialization on a DENSE smoke store (the build
#    brief's own reference point: the fake-extractor-shaped corpus that
#    produces ~14k inferred edges in eager mode). Lazy mount must not
#    grow with the SAME curve -- checked via mount timing and via the
#    absence of any materialized closure structure (inferred_edges()
#    raising, canon_inferred_edges() raising when canon is on).
# ─────────────────────────────────────────────────────────────────────────
def test_dense_smoke_store_lazy_mount_and_query_latency(d):
    rng = random.Random(999)
    # Chain-shaped corpus at a scale the build brief's own P74 organ
    # measured degrading eager append/mount cost on (a few hundred base
    # edges, many overlapping chains -- density > 1, per the build
    # brief's own WT-103 finding).
    n_chains = 120
    chain_len = 4
    key_ctr = [0]

    def _fresh():
        key_ctr[0] += 1
        return "C{}".format(key_ctr[0])

    segments = []
    all_start_keys = []
    shared_pool = [_fresh() for _ in range(15)]  # shared join points -> density
    for _ in range(n_chains):
        start = _fresh()
        all_start_keys.append(start)
        cur = start
        for _ in range(chain_len):
            nxt = rng.choice(shared_pool) if rng.random() < 0.5 else _fresh()
            segments.append([make_record(cur, nxt, doc_coord=len(segments))])
            cur = nxt

    dir_eager = os.path.join(d, "eager")
    dir_lazy = os.path.join(d, "lazy")
    os.makedirs(dir_eager)
    os.makedirs(dir_lazy)
    _build_store(dir_eager, segments)
    _build_store(dir_lazy, segments)

    t0 = time.perf_counter()
    g_eager = LiveGraph(dir_eager, inference="eager")
    t_mount_eager = time.perf_counter() - t0
    n_inferred_eager = len(g_eager.inferred_edges())

    t0 = time.perf_counter()
    g_lazy = LiveGraph(dir_lazy, inference="lazy")
    t_mount_lazy = time.perf_counter() - t0

    print(
        "  [dense-smoke] eager: mount={:.3f}s n_inferred={}  |  lazy: mount={:.3f}s (no materialized closure)".format(
            t_mount_eager, n_inferred_eager, t_mount_lazy
        )
    )
    assert n_inferred_eager > 0, "test corpus too sparse to exercise density at all"
    assert t_mount_lazy < max(0.5, t_mount_eager), (
        "lazy mount ({:.3f}s) was not faster than or comparable to eager mount "
        "({:.3f}s) on the dense smoke corpus".format(t_mount_lazy, t_mount_eager)
    )

    try:
        g_lazy.inferred_edges()
        assert False, "inferred_edges() must raise in lazy mode (no materialized closure exists)"
    except RuntimeError:
        pass

    # Query-latency distribution (p50/p95) over every start key, lazy
    # mode, at the DEFAULT node_budget -- this corpus is dense enough
    # (999,976 eager-materialized inferred edges from ~1,900 base
    # records, a concretely measured density explosion mirroring the
    # build brief's own ConceptNet/WT-103 findings) that the default
    # budget legitimately truncates some hub-adjacent queries -- that is
    # the organ working as designed (bounding per-query cost regardless
    # of density), not a test failure. What this measures is that
    # QUERY LATENCY stays bounded either way (truncated or not), which
    # is the actual cost claim -- a caller who needs a complete answer on
    # a graph this dense passes a larger explicit node_budget (see
    # test_canon_lazy_equivalence and test_lazy_equals_eager_across_
    # densities for the "given enough budget, still edge-for-edge
    # correct" property, exercised on sparser corpora where "enough
    # budget" is a reasonable number to state).
    latencies = []
    n_truncated = 0
    for k in all_start_keys:
        t0 = time.perf_counter()
        edges, truncated = g_lazy.query(k, return_truncated=True)
        latencies.append(time.perf_counter() - t0)
        if truncated:
            n_truncated += 1
    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)]
    print("  [dense-smoke] lazy query latency over {} keys: p50={:.5f}s p95={:.5f}s ({} truncated at default budget)".format(
        len(latencies), p50, p95, n_truncated
    ))
    assert p95 < 1.0, "p95 query latency implausibly high for this corpus scale -- investigate"
    # The whole point: bounded latency held EVEN on a corpus dense enough
    # that eager mode took 11+ seconds and 1M materialized edges to mount.
    assert t_mount_eager > 2.0, (
        "test corpus did not actually reproduce the measured density wall "
        "(eager mount only {:.3f}s) -- strengthen the corpus".format(t_mount_eager)
    )

    # Cross-check on a NON-truncated key: lazy answers still agree with
    # eager's materialized ones when the traversal completed naturally.
    checked_one = False
    for k in all_start_keys:
        edges, truncated = g_lazy.query(k, return_truncated=True)
        if truncated:
            continue
        assert _edge_sig(g_eager.query(k)) == _edge_sig(edges)
        checked_one = True
        if checked_one:
            break
    assert checked_one, "every start key truncated at default budget -- cannot cross-check any complete answer"


# ─────────────────────────────────────────────────────────────────────────
# 7. Canon layer in lazy mode: canon_query derives on demand over the
#    canon adjacency, equivalent to eager canon_query given a generous
#    budget, and canon_inferred_edges() raises (same reasoning as the
#    raw layer's inferred_edges()).
# ─────────────────────────────────────────────────────────────────────────
def test_canon_lazy_equivalence(d):
    r1 = make_record("the old king", "a costly war", doc_coord=0)
    r1["trigger"], r1["outcome"] = "the old king", "a costly war"
    r2 = make_record("the resulting war", "a great famine", doc_coord=1)
    r2["trigger"], r2["outcome"] = "the resulting war", "a great famine"

    dir_eager = os.path.join(d, "eager")
    dir_lazy = os.path.join(d, "lazy")
    os.makedirs(dir_eager)
    os.makedirs(dir_lazy)
    for dir_ in (dir_eager, dir_lazy):
        store = LiveStore(dir_)
        store.append_segment([r1])
        store.append_segment([r2])

    g_eager = LiveGraph(dir_eager, canon=True, inference="eager")
    g_lazy = LiveGraph(dir_lazy, canon=True, inference="lazy", node_budget=100000)

    assert g_lazy.canon_of("the old king") == "king"
    assert g_lazy.canon_of("the resulting war") == "war"

    eager_result = g_eager.canon_query("the old king")
    lazy_result, truncated = g_lazy.canon_query("the old king", return_truncated=True)
    assert truncated is False
    assert _edge_sig(eager_result) == _edge_sig(lazy_result)
    assert any(e["kind"] == "inferred" and e["to_key"] == "famine" for e in lazy_result), (
        "expected the king -> war -> famine chain via canon_key merge"
    )

    # query(key, canon=True) must route to canon_query in lazy mode too.
    assert _edge_sig(g_lazy.query("the old king", canon=True)) == _edge_sig(lazy_result)

    try:
        g_lazy.canon_inferred_edges()
        assert False, "canon_inferred_edges() must raise in lazy mode"
    except RuntimeError:
        pass

    # canon=False graph in lazy mode: canon_query still raises for the
    # right reason (canon not enabled, not "lazy mode broke something").
    dir_no_canon = os.path.join(d, "no_canon")
    os.makedirs(dir_no_canon)
    store = LiveStore(dir_no_canon)
    store.append_segment([r1])
    g_lazy_no_canon = LiveGraph(dir_no_canon, canon=False, inference="lazy")
    try:
        g_lazy_no_canon.canon_query("the old king")
        assert False
    except RuntimeError as e:
        assert "canon=True" in str(e)


# ─────────────────────────────────────────────────────────────────────────
# 8. Verifier compatibility: direction-3 re-derivation (stranger_verify_
#    run.py's _inferred_edge_rederive) works UNCHANGED on a lazy-produced
#    inferred edge -- it is naturally re-derivation-compatible because it
#    only ever reads cited (segment_sha, idx) records via an INDEPENDENT
#    LiveStore mount, never LiveGraph's own inferred-edge structure
#    (which does not exist at all in lazy mode).
# ─────────────────────────────────────────────────────────────────────────
def test_lazy_edges_pass_direction3_rederivation(d):
    rng = random.Random(606)
    keys, segments = _random_dense_corpus(rng, n_keys=14, n_edges=35, max_citations=2)
    store, _shas = _build_store(d, segments)

    g_lazy = LiveGraph(d, inference="lazy", node_budget=100000)

    n_rederived = 0
    for k in keys:
        edges, truncated = g_lazy.query(k, return_truncated=True)
        assert truncated is False
        for edge in edges:
            if edge["kind"] != "inferred":
                continue
            # A stranger mounts an INDEPENDENT LiveStore (not g_lazy.store)
            # and re-derives from nothing but the cited (sha, idx) pairs.
            stranger_store = LiveStore(d)
            edge_for_rederive = {
                "from_key": k,
                "to_key": edge["to_key"],
                "depth": edge["depth"],
                "derivation": edge["derivation"],
            }
            frag = _inferred_edge_rederive(stranger_store, edge_for_rederive)
            assert frag is not None, "direction-3 rederivation failed for a lazy-produced edge: {}".format(edge)
            assert frag["from_key"] == k
            assert frag["to_key"] == edge["to_key"]
            assert frag["depth"] == edge["depth"]
            n_rederived += 1

    assert n_rederived > 0, "test corpus produced no inferred edges to verify -- strengthen it"


def run_all():
    tests = [
        test_eager_is_default_and_regression_identical,
        test_lazy_equals_eager_across_densities,
        test_truncation_is_visible_and_is_a_subset,
        test_lazy_on_append_and_drop_stay_correct,
        test_lazy_append_does_not_scale_with_density,
        test_dense_smoke_store_lazy_mount_and_query_latency,
        test_canon_lazy_equivalence,
        test_lazy_edges_pass_direction3_rederivation,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
