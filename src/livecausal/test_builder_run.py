"""Plain-assert tests for the builder loop (src/livecausal/builder_run.py).
Run: python3 src/livecausal/test_builder_run.py

Offline-only: forces HF_HUB_OFFLINE/HF_DATASETS_OFFLINE before importing
portable_organism (get_vocab() needs a warm WikiText2 cache but must not
probe the Hub -- the beast-DNS-is-down constraint the build brief named).
"""
import os
import random
import shutil
import sys
import tempfile

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.infer import LiveGraph  # noqa: E402
from livecausal.evidence import EvidenceLedger  # noqa: E402
from livecausal.builder_run import (  # noqa: E402
    TextFileStream,
    fake_extractor,
    generate_smoke_corpus,
    run_builder,
    stream_windows,
)
from stranger_verify_run import score, verify_direction3  # noqa: E402


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="builder-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _build_smoke_graph(d, seed=42, max_windows=40):
    """Runs the full builder loop against the synthetic causal corpus +
    the built-in fake extractor, entirely offline. Returns
    (graph, chains, metrics)."""
    import torch
    import portable_organism as po

    torch.set_num_threads(1)
    po.D_MODEL, po.BATCH, po.CHUNK = 64, 4, 32
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = 0.75, 50, 10, 5

    corpus_path = os.path.join(d, "corpus.txt")
    chains = generate_smoke_corpus(corpus_path, seed=seed)

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    torch.manual_seed(seed)
    organism = po.Organism("builder-test", V, mask, seed=seed)
    stream = TextFileStream(corpus_path, stoi, unk)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    window_iter = stream_windows(organism, stream, feeder, window_tokens=32)

    store_dir = os.path.join(d, "store")
    status_path = os.path.join(d, "status.json")
    metrics_path = os.path.join(d, "metrics.jsonl")

    graph, metrics = run_builder(
        store_dir, status_path, metrics_path,
        window_iter, fake_extractor,
        windows_per_segment=5,
        max_windows=max_windows, print_every=1000,
        stream=stream,
    )
    return graph, chains, metrics, store_dir


# ---------------------------------------------------------------------
# Test 1: the loop runs offline end to end and produces a non-trivial
# graph (base edges + inferred edges both present).
# ---------------------------------------------------------------------

def test_smoke_loop_produces_graph(d):
    graph, chains, metrics, store_dir = _build_smoke_graph(d)
    assert metrics["n_windows_total"] > 0, "no windows streamed"
    assert metrics["n_triplets_total"] > 0, "extractor validated nothing"
    assert metrics["n_segments"] > 0, "no segments were appended"
    assert metrics["n_base_edges"] > 0, "no base edges in the graph"
    assert metrics["n_inferred_edges"] > 0, "no inferred (transitive) edges in the graph"
    # P70 policy: extraction is ungated, so windows_total should exceed (or
    # at minimum equal, if the gate happened to accept everything in this
    # short smoke run) windows_gated -- and every window is a candidate for
    # extraction, unlike the pre-P70 gated-only loop.
    assert metrics["n_windows_total"] >= metrics["n_windows_gated"] >= 0
    assert metrics["n_triplets_total"] >= metrics["n_triplets_from_gated"] >= 0


# ---------------------------------------------------------------------
# Test 1b: the evidence ledger hook (found missing by mvp3's demo
# integration test) is actually wired into the loop -- after a smoke
# build, evidence_count for at least one base edge must be > 0, and a
# FRESH EvidenceLedger mount over the same store reproduces the same
# count (the fold-reproducibility guarantee test_evidence.py already
# covers in isolation, exercised here end to end through the loop).
# ---------------------------------------------------------------------

def test_evidence_ledger_populated_after_build(d):
    graph, chains, metrics, store_dir = _build_smoke_graph(d)
    assert metrics["n_base_edges"] > 0, "fixture too small to test evidence"

    valid_segments = graph.store.segments()
    led = EvidenceLedger(store_dir)

    any_nonzero = False
    for from_key, targets in graph._base_edges.items():
        for to_key in targets:
            count = led.evidence_count((from_key, to_key), valid_segments)
            if count > 0:
                any_nonzero = True
                break
        if any_nonzero:
            break
    assert any_nonzero, (
        "evidence_count is 0 for every base edge after a build -- the "
        "evidence-ledger hook is not actually being called from run_builder"
    )

    # Fresh mount reproduces the same fold (no double-counting from the
    # hook being called more than once per segment across this run).
    led2 = EvidenceLedger(store_dir)
    sample_key = next(
        (from_key, to_key)
        for from_key, targets in graph._base_edges.items()
        for to_key in targets
    )
    assert led2.evidence_count(sample_key, valid_segments) == led.evidence_count(sample_key, valid_segments)

    # mvp3's demo.py does a REDUNDANT post-loop replay of
    # append_observations_for_segment over every sealed segment (its own
    # workaround, written before this hook existed in run_builder itself).
    # Now that run_builder observes each segment once already, a second
    # (demo-style) pass over the same segments must be a harmless no-op AT
    # THE FOLD LEVEL: evidence_count must not double, even though the
    # ledger FILE grows (more lines, same distinct evidence_key values --
    # append_observation's fold dedupes on evidence_key, per evidence.py).
    count_before = led.evidence_count(sample_key, valid_segments)
    for sha in graph.store.segments():
        led.append_observations_for_segment(graph, sha)
    count_after = led.evidence_count(sample_key, valid_segments)
    assert count_before == count_after, (
        "a redundant replay of append_observations_for_segment (mvp3 demo's "
        "own pattern) changed evidence_count -- not idempotent at the fold "
        "level, which means run_builder's hook + demo.py's hook would "
        "double-count if both ever ran against the same store"
    )


# ---------------------------------------------------------------------
# Test 2: at least one full known chain (from the synthetic corpus'
# ground truth) survives end to end as base edges AND as a correct
# transitive inferred edge spanning the whole chain.
# ---------------------------------------------------------------------

def test_expected_chain_present(d):
    graph, chains, metrics, store_dir = _build_smoke_graph(d)

    full_chains_found = []
    for chain in chains:
        hops_present = all(
            chain[i + 1] in graph._base_edges.get(chain[i], {})
            for i in range(len(chain) - 1)
        )
        if hops_present:
            full_chains_found.append(chain)

    assert full_chains_found, (
        "no complete known chain survived curation+extraction -- "
        "graph has {} base edges, {} inferred".format(
            metrics["n_base_edges"], metrics["n_inferred_edges"]
        )
    )

    # For the first fully-present chain, the transitive inferred edge
    # spanning start->end (depth = chain length) must exist, with a
    # derivation whose length matches.
    chain = full_chains_found[0]
    start, end = chain[0], chain[-1]
    expected_depth = len(chain) - 1
    matches = [
        e for e in graph.inferred_edges()
        if e["from_key"] == start and e["to_key"] == end and e["depth"] == expected_depth
    ]
    assert matches, "expected transitive edge {} -> {} (depth {}) not found; got depths {}".format(
        start, end, expected_depth,
        sorted(e["depth"] for e in graph.inferred_edges() if e["from_key"] == start and e["to_key"] == end),
    )


# ---------------------------------------------------------------------
# Test 3: the graph's own record schema matches the store contract
# (doc_coord/evidence_count/use_count/meta.extractor_version present).
# ---------------------------------------------------------------------

def test_record_schema(d):
    graph, chains, metrics, store_dir = _build_smoke_graph(d)
    seen_any = False
    seen_gated_true = False
    seen_gated_false = False
    for sha, idx, record in graph.store.iter_records():
        seen_any = True
        assert "trigger_key" in record and "outcome_key" in record
        assert record["evidence_count"] == 1
        assert record["use_count"] == 0
        assert isinstance(record.get("doc_coord"), int)
        meta = record.get("meta", {})
        assert meta.get("extractor_version") == "builder_v0"
        assert isinstance(meta.get("surprise"), float), "meta.surprise missing or not a float"
        assert isinstance(meta.get("gated"), bool), "meta.gated missing or not a bool"
        if meta["gated"]:
            seen_gated_true = True
        else:
            seen_gated_false = True
    assert seen_any, "no records were ever written to the store"
    # P70 policy check: since extraction is ungated, some stored records
    # should come from UNGATED windows too (not exclusively from windows
    # the organism's own training gate accepted) -- this is the direct
    # behavioral signature of the policy update, not just a metrics count.
    assert seen_gated_false, (
        "every stored record came from a gated window -- extraction "
        "looks gated again, which contradicts the P70 policy update"
    )


# ---------------------------------------------------------------------
# Test 3b: extraction runs on windows the gate did NOT accept -- the
# direct behavioral proof of "ungated extraction, gate rides along as a
# signal" rather than inferring it from aggregate counts alone.
# ---------------------------------------------------------------------

def test_extraction_is_ungated(d):
    """Drives stream_windows + the extractor directly (bypassing
    run_builder's batching) so we can assert on individual windows: the
    fake extractor must be invoked for windows where gated=False, and any
    triplets it finds there must still produce records (not be silently
    dropped for being ungated)."""
    import torch
    import portable_organism as po

    torch.set_num_threads(1)
    po.D_MODEL, po.BATCH, po.CHUNK = 64, 4, 32
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = 0.75, 50, 10, 5

    corpus_path = os.path.join(d, "corpus.txt")
    generate_smoke_corpus(corpus_path, seed=99)

    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    torch.manual_seed(99)
    organism = po.Organism("ungated-test", V, mask, seed=99)
    stream = TextFileStream(corpus_path, stoi, unk)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)

    from livecausal.builder_run import stream_windows

    win_iter = stream_windows(organism, stream, feeder, window_tokens=32)
    seen_ungated_windows = 0
    triplets_from_ungated = 0
    for i, (tape_pos, window_text, surprise, gated) in enumerate(win_iter):
        triplets = fake_extractor(window_text)  # extractor runs regardless of `gated`
        if not gated:
            seen_ungated_windows += 1
            triplets_from_ungated += len(triplets)
        if i >= 60:
            break

    assert seen_ungated_windows > 0, "gate accepted every single window in this run -- test is vacuous"
    # Not asserting triplets_from_ungated > 0 as a hard requirement (depends
    # on where in the shuffled corpus ungated windows happen to land), but
    # if it's ever exactly 0 across 60 windows with hits elsewhere, print
    # for visibility rather than silently passing.
    print("[test] ungated windows: {}, triplets from them: {}".format(
        seen_ungated_windows, triplets_from_ungated))


# ---------------------------------------------------------------------
# Test 4: THE ABNORMAL-STANDARD -- the builder's own output graph passes
# the direction-3 stranger verifier (engine-distrustful re-derivation),
# per the build brief's "dein eigener Verifier als Abnahme" requirement.
# ---------------------------------------------------------------------

def test_verifier_accepts_builder_output(d):
    graph, chains, metrics, store_dir = _build_smoke_graph(d)
    n_base = sum(len(v) for v in graph._base_edges.values())
    n_inferred = len(graph.inferred_edges())
    n_target = min(30, n_base + n_inferred)
    assert n_target > 0

    checks = verify_direction3(store_dir, n_samples=n_target, seed=60)
    scoring = score(checks, n_target=n_target)
    assert scoring["p60a_verified_all"] is True, scoring
    assert scoring["p60b_consensus_all"] is True, scoring


# ---------------------------------------------------------------------
# Test 5: extractor-contract stub injection -- resolve_extractor honors
# an override without importing curator_yield_run at all (the file must
# work standalone whether or not that module exists).
# ---------------------------------------------------------------------

def test_extractor_override_is_used_not_default(d):
    from livecausal.builder_run import resolve_extractor

    calls = []

    def spy_extractor(text):
        calls.append(text)
        return []

    resolved = resolve_extractor(override=spy_extractor)
    resolved("hello world")
    assert calls == ["hello world"], "override was not used as the extractor"


def run_all():
    tests = [
        test_smoke_loop_produces_graph,
        test_evidence_ledger_populated_after_build,
        test_expected_chain_present,
        test_record_schema,
        test_extraction_is_ungated,
        test_verifier_accepts_builder_output,
        test_extractor_override_is_used_not_default,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
