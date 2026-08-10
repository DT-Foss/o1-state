"""Plain-assert tests for the canon-layer's semi-naive delta maintenance
and persisted canon_map.jsonl (src/livecausal/infer.py, Task 12 / P75c
follow-up). Run:
    python3 src/livecausal/test_canon_delta.py

P75c (scored, real P72 artifact) found the P74-era design cost 17.8s on a
WARM mount (canon_inferred.jsonl cache hit) because every raw_key was
re-canonicalized -- re-parsed through spaCy -- on every mount regardless
of the inferred-edge cache, since only the closure step was cached, not
the raw_key -> canon_key fold itself. This file covers what Task 12 was
scoped to fix:

  1. Warm-mount timing on a smoke store: a SECOND mount (fresh LiveGraph
     object, both canon_map.jsonl and canon_inferred.jsonl now on disk)
     must be dramatically faster than the first (cold) mount -- checked
     both as a relative ratio (warm << cold, the shape the build brief
     asks for) and against an absolute bar generous enough not to be
     flaky on a loaded CI machine, but tight enough to catch a real
     regression back to full-reparse-on-warm-mount behavior.
  2. Delta equivalence against a from-scratch full fold (the batch-oracle
     pattern MVP-2/test_infer.py already established for the RAW layer,
     applied here to the CANON layer): incremental on_append/on_drop
     calls must produce the exact same canon_inferred edge set a fresh,
     from-scratch fold over the same final base adjacency would -- on a
     corpus deliberately constructed to exercise the many-to-one join
     case the build brief calls out by name (a new record folding onto
     an ALREADY-PRESENT canon_key must correctly connect to that
     canon_key's existing citations, not just its own).
  3. Regression: the full existing livecausal suite (imported indirectly
     by running each test file) is expected to stay green -- this file
     does not re-run them, run_all_livecausal_tests.py-style scripts (or
     manual sequential runs) cover that; test_canon.py's own 6 tests
     already re-verify canon=False byte-identical regression and the
     connectivity/no-mutation/stranger invariants against the NEW
     map+delta implementation (they were re-run green during this task,
     since they exercise the class through its public API only).
"""
import os
import random
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.canon import canonicalize_with_default_nlp
from livecausal.infer import LiveGraph, _batch_transitive_closure, _derivation_key
from livecausal.store import LiveStore


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="livecausal-canon-delta-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _make_record(trigger, outcome, doc_coord):
    return {
        "trigger": trigger,
        "mechanism": "causes",
        "outcome": outcome,
        "trigger_key": trigger.strip().lower(),
        "outcome_key": outcome.strip().lower(),
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


def _full_canon_fold_reference(store_dir, nlp=None):
    """Independent, from-scratch reference computation of the canon
    layer's inferred edges: mounts a fresh LiveStore (NOT the LiveGraph
    under test), folds raw base edges to canon_key adjacency by calling
    canonicalize_with_default_nlp on every distinct raw_key exactly once
    (a plain dict fold, deliberately NOT sharing any code path with
    infer.py's LiveGraph.__canon_*__ methods -- this is the oracle,
    independent implementation the batch-oracle pattern requires), then
    runs infer.py's own _batch_transitive_closure over that adjacency.
    Returns the sorted canon_inferred edge signature (comparable to
    LiveGraph.canon_inferred_edges() after the same signature transform).
    """
    store = LiveStore(store_dir)
    canon_base_edges = {}
    raw_to_canon = {}

    def _canon_of(raw_key):
        if raw_key in raw_to_canon:
            return raw_to_canon[raw_key]
        ck = canonicalize_with_default_nlp(raw_key)
        raw_to_canon[raw_key] = ck
        return ck

    for sha in store.segments():
        for _s, idx, record in store.iter_records(sha):
            from_key = record.get("trigger_key")
            to_key = record.get("outcome_key")
            if from_key is None or to_key is None:
                continue
            canon_from = _canon_of(from_key)
            canon_to = _canon_of(to_key)
            bucket = canon_base_edges.setdefault(canon_from, {}).setdefault(canon_to, [])
            pair = [sha, idx]
            if pair not in bucket:
                bucket.append(pair)
                bucket.sort(key=lambda p: (p[0], p[1]))

    edges = _batch_transitive_closure(canon_base_edges)
    return _edge_signature(edges)


def _edge_signature(edges):
    return sorted(
        (e["from_key"], e["to_key"], e["depth"], _derivation_key(e["derivation"]))
        for e in edges
    )


def _graph_canon_signature(graph):
    return _edge_signature(graph.canon_inferred_edges())


# ─────────────────────────────────────────────────────────────────────────
# Corpus generator: deliberately constructs the many-to-one join case --
# several DISTINCT raw_key surface strings across DIFFERENT segments that
# all canonicalize to the SAME canon_key, so a new segment's citation can
# only correctly chain if it joins against citations contributed by
# raw_keys appended in EARLIER, unrelated segments.
# ─────────────────────────────────────────────────────────────────────────
_DETERMINERS = ["the", "a", "the old", "the resulting", "the ensuing", "that terrible"]
_ADJECTIVES = ["", "severe ", "rising ", "sudden "]


def _many_to_one_corpus(rng, n_heads=14, chain_len=3, n_extra_branches=10):
    """Returns a list of segments (each a list of records), where a
    length-`chain_len` chain of head nouns is expressed via records whose
    raw trigger/outcome phrasing VARIES (different determiner/adjective
    each time a head noun is referenced), one record per segment -- so
    consecutive hops of the SAME canon-level chain are contributed by
    raw-string-DIFFERENT records in DIFFERENT segments, and the
    canon-level join only exists because canonicalization folds them
    together. Also adds n_extra_branches single-hop records off random
    existing heads with yet another determiner/adjective variant, to
    exercise "a new citation on an existing (canon_from, canon_to) pair"
    (not just "a new pair") and "a new citation whose canon_from already
    has other neighbors" (ancestor-side fan-out) in the delta path.
    """
    heads = ["entity{}".format(i) for i in range(n_heads)]

    def _phrase(head):
        return "{} {}".format(rng.choice(_DETERMINERS), head).replace("  ", " ") if rng.random() > 0.3 \
            else "{} {}{}".format(rng.choice(_DETERMINERS), rng.choice(_ADJECTIVES), head).replace("  ", " ")

    segments = []
    doc_coord = [0]

    def _next_coord():
        doc_coord[0] += 1
        return doc_coord[0]

    # Main chains: several independent chains over the head pool, each
    # hop its own segment with independently varied phrasing.
    idx = 0
    while idx + chain_len < len(heads):
        chain_heads = heads[idx: idx + chain_len + 1]
        for i in range(chain_len):
            rec = _make_record(_phrase(chain_heads[i]), _phrase(chain_heads[i + 1]), _next_coord())
            segments.append([rec])
        idx += chain_len + 1

    # Extra branches: single-hop records reusing existing heads with new
    # phrasing, appended AFTER the main chains -- these are the
    # "citation on an existing pair, from an old canon component"
    # append-order-sensitive cases the delta path must get right.
    for _ in range(n_extra_branches):
        a, b = rng.sample(heads, 2)
        rec = _make_record(_phrase(a), _phrase(b), _next_coord())
        segments.append([rec])

    return segments


# ─────────────────────────────────────────────────────────────────────────
# 1. Warm-mount timing: second mount dramatically faster than the first.
# ─────────────────────────────────────────────────────────────────────────
def test_warm_mount_much_faster_than_cold(d):
    rng = random.Random(42)
    segments = _many_to_one_corpus(rng, n_heads=30, chain_len=3, n_extra_branches=20)

    store = LiveStore(d)
    for records in segments:
        store.append_segment(records)

    t0 = time.perf_counter()
    g1 = LiveGraph(d, canon=True)
    t_cold = time.perf_counter() - t0
    assert g1.was_canon_rebuilt_on_mount() is True
    assert g1.was_canon_loaded_from_cache() is False

    t0 = time.perf_counter()
    g2 = LiveGraph(d, canon=True)
    t_warm = time.perf_counter() - t0
    assert g2.was_canon_rebuilt_on_mount() is False
    assert g2.was_canon_loaded_from_cache() is True

    assert g1.canon_inferred_edges() == g2.canon_inferred_edges(), (
        "warm mount must reproduce the exact same canon_inferred edge set as the cold mount"
    )

    # Relative: warm must be a small fraction of cold (the build brief's
    # own shape requirement). Generous factor (well under 1x, not
    # requiring e.g. 100x) so this does not flake on a loaded machine --
    # the absolute bar below is the one that actually enforces P75c's
    # <5s requirement.
    assert t_warm < t_cold, "warm mount was not faster than cold mount at all: {:.3f}s vs {:.3f}s".format(
        t_warm, t_cold
    )
    assert t_warm < max(1.0, t_cold * 0.5), (
        "warm mount ({:.3f}s) is not a small fraction of cold mount ({:.3f}s) -- "
        "suggests canon_map.jsonl is not being used to skip re-parsing".format(t_warm, t_cold)
    )

    # Absolute: P75c's registered consequence -- warm mount must clear
    # the 5s bar with real margin. This machine's actual warm-mount
    # numbers during development sat at ~0.02s in-process and ~1.5-1.9s
    # across a genuine fresh subprocess (dominated by spaCy's own model
    # load inside env_pin(), not by any re-parsing of probe content) --
    # 5s leaves comfortable headroom for a loaded CI machine while still
    # catching a real regression back to full-reparse-on-warm-mount.
    assert t_warm < 5.0, "warm mount took {:.3f}s -- P75c's <5s bar not met".format(t_warm)

    # A third mount (yet another fresh object) must also be warm --
    # confirms the map+cache combination is stably reused, not a
    # one-time fluke of the second mount specifically.
    t0 = time.perf_counter()
    g3 = LiveGraph(d, canon=True)
    t_warm2 = time.perf_counter() - t0
    assert g3.was_canon_loaded_from_cache() is True
    assert t_warm2 < 5.0


# ─────────────────────────────────────────────────────────────────────────
# 2. Delta equivalence against a from-scratch full fold, on a corpus
#    constructed to exercise the many-to-one canon_key join case.
# ─────────────────────────────────────────────────────────────────────────
def test_canon_delta_equals_full_fold_on_many_to_one_corpus(d):
    rng = random.Random(777)
    segments = _many_to_one_corpus(rng, n_heads=24, chain_len=3, n_extra_branches=15)

    incr_dir = os.path.join(d, "incremental")
    os.makedirs(incr_dir)

    # Incremental: append segment by segment via append_segment (which
    # calls on_append -> _canon_on_append internally), exactly the shape
    # a live builder session would use.
    graph = LiveGraph(incr_dir, canon=True)
    for records in segments:
        graph.append_segment(records)

    incr_sig = _graph_canon_signature(graph)
    assert len(incr_sig) > 0, "test is vacuous: no canon-inferred edges produced at all"

    # Batch reference: an INDEPENDENT from-scratch fold (not LiveGraph's
    # own cold-mount path, which would just be testing infer.py against
    # itself) over the exact same final store content.
    batch_sig = _full_canon_fold_reference(incr_dir)

    assert incr_sig == batch_sig, (
        "canon-layer delta result diverges from a from-scratch full fold "
        "on the many-to-one join corpus -- delta maintenance is incorrect"
    )

    # Also cross-check against a COLD-mounted fresh LiveGraph over the
    # same store (infer.py's own full-fold path, via a fresh object) --
    # a second, cheaper equivalence check that stays inside infer.py's
    # public API (the earlier check is the true independent oracle).
    fresh_dir = os.path.join(d, "fresh_cold")
    os.makedirs(fresh_dir)
    fresh_store = LiveStore(fresh_dir)
    for records in segments:
        fresh_store.append_segment(records)
    fresh_graph = LiveGraph(fresh_dir, canon=True)
    assert fresh_graph.was_canon_rebuilt_on_mount() is True
    assert _graph_canon_signature(fresh_graph) == incr_sig


# ─────────────────────────────────────────────────────────────────────────
# 3. Delta equivalence after a DROP, on the same many-to-one corpus --
#    the drop must remove exactly the citing segment's contribution,
#    with the map's stamp correctly re-written (not left stale) so a
#    SUBSEQUENT mount stays warm rather than silently forcing a re-parse.
# ─────────────────────────────────────────────────────────────────────────
def test_canon_delta_drop_equals_full_fold_without_segment(d):
    rng = random.Random(2024)
    segments = _many_to_one_corpus(rng, n_heads=20, chain_len=3, n_extra_branches=12)

    incr_dir = os.path.join(d, "incremental")
    os.makedirs(incr_dir)

    graph = LiveGraph(incr_dir, canon=True)
    shas = [graph.append_segment(records) for records in segments]

    # Drop a middle segment -- one that contributes a hop inside a main
    # chain (not just an extra branch), so the drop actually invalidates
    # at least one multi-hop canon_inferred edge.
    mid_idx = len(shas) // 3
    dropped_sha = shas[mid_idx]
    graph.drop_segments([dropped_sha])

    incr_sig = _graph_canon_signature(graph)

    # Batch reference over a store built from every segment EXCEPT the
    # dropped one.
    ref_dir = os.path.join(d, "ref_without_dropped")
    os.makedirs(ref_dir)
    ref_store = LiveStore(ref_dir)
    for i, records in enumerate(segments):
        if i == mid_idx:
            continue
        ref_store.append_segment(records)
    batch_sig = _full_canon_fold_reference(ref_dir)

    assert incr_sig == batch_sig, (
        "canon-layer drop-delta result diverges from a from-scratch full "
        "fold over the store without the dropped segment"
    )

    # The map's stamp must have been re-written after the drop (per
    # _canon_on_drop's docstring: content unchanged, stamp re-synced to
    # the new manifest) -- a subsequent mount must stay WARM, not force
    # a full re-parse because the map's stamp is stale relative to the
    # post-drop manifest.
    t0 = time.perf_counter()
    graph2 = LiveGraph(incr_dir, canon=True)
    t_warm_after_drop = time.perf_counter() - t0
    assert graph2.was_canon_loaded_from_cache() is True, (
        "mount after a drop-only session was NOT warm -- canon_map.jsonl's "
        "stamp was not correctly re-written by on_drop"
    )
    assert t_warm_after_drop < 5.0
    assert _graph_canon_signature(graph2) == incr_sig


def run_all():
    tests = [
        test_warm_mount_much_faster_than_cold,
        test_canon_delta_equals_full_fold_on_many_to_one_corpus,
        test_canon_delta_drop_equals_full_fold_without_segment,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
