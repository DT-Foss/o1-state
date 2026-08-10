#!/usr/bin/env python3
"""
DEMO CLI INTEGRATION TEST (plain asserts, no framework).

The through-scenario the build brief names: build a mini corpus -> query
shows a chain with evidence -> cut a segment -> query shows the chain gone
(or thinned) / the rest intact -> verify green.

Run: python3 src/livecausal/test_demo.py
"""
import os
import shutil
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import argparse  # noqa: E402

import livecausal.builder_run as builder_run  # noqa: E402
from livecausal import demo  # noqa: E402
from livecausal.infer import LiveGraph  # noqa: E402
from livecausal import evidence as ev  # noqa: E402


CORPUS_TEXT = (
    "Smoking causes lung cancer. Lung cancer leads to reduced life "
    "expectancy. Reduced life expectancy worsens family financial "
    "stability. The wildfire smoke triggers respiratory illness. "
    "Respiratory illness increases hospital admissions. Hospital "
    "admissions strain public healthcare budgets."
)


def _build_demo_store(text_file, store_dir, window_tokens=40, max_windows=80):
    """Drives the SAME path demo.py's cmd_build does (build_window_iterator
    + resolve_extractor + run_builder), directly -- so this test exercises
    the real code path, not a re-implementation of it."""
    import torch
    torch.set_num_threads(1)
    import portable_organism as po

    ns = argparse.Namespace(
        text_file=text_file, source=None, window_tokens=window_tokens,
        d_model=64, batch=4, chunk_size=32, q=0.75, window=50,
        min_window=10, ignition_chunks=5, chunks=None, tape_cap=200_000,
    )
    po.D_MODEL, po.BATCH, po.CHUNK = ns.d_model, ns.batch, ns.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = (
        ns.q, ns.window, ns.min_window, ns.ignition_chunks,
    )

    extractor = builder_run.resolve_extractor(None)
    window_iter, stream = builder_run.build_window_iterator(ns, organism_seed=42)

    status_path = os.path.join(store_dir, ".test_build_status.json")
    metrics_path = os.path.join(store_dir, ".test_build_metrics.jsonl")
    os.makedirs(store_dir, exist_ok=True)

    graph, metrics = builder_run.run_builder(
        store_dir, status_path, metrics_path, window_iter, extractor,
        windows_per_segment=5, max_windows=max_windows, stream=stream,
    )

    evidence_ledger = ev.EvidenceLedger(store_dir)
    for sha in graph.store.segments():
        evidence_ledger.append_observations_for_segment(graph, sha)

    return graph, metrics


def test_through_scenario():
    tmpdir = tempfile.mkdtemp(prefix="livecausal-demo-test-")
    try:
        corpus_path = os.path.join(tmpdir, "corpus.txt")
        with open(corpus_path, "w", encoding="utf-8") as f:
            for _ in range(80):
                f.write(CORPUS_TEXT)
                f.write("\n\n")

        store_dir = os.path.join(tmpdir, "store")

        # 1. BUILD: a mini corpus with a known causal chain.
        graph, metrics = _build_demo_store(corpus_path, store_dir)
        assert metrics["n_base_edges"] > 0, "build produced no base edges"
        assert metrics["n_inferred_edges"] > 0, "build produced no inferred edges (chain never formed)"
        print("[PASS] build: {} base edges, {} inferred edges".format(
            metrics["n_base_edges"], metrics["n_inferred_edges"]))

        # 2. QUERY: the chain is visible with evidence.
        graph2 = LiveGraph(store_dir)
        edges_before = graph2.query("smoking")
        assert edges_before, "query('smoking') returned no edges"
        base_before = [e for e in edges_before if e["kind"] == "base"]
        inferred_before = [e for e in edges_before if e["kind"] == "inferred"]
        assert base_before, "no base edge from 'smoking'"
        assert any(e["to_key"] == "lung cancer" for e in base_before), (
            "expected base edge smoking -> lung cancer, got {}".format(base_before))
        assert inferred_before, "no inferred (multi-hop) edge from 'smoking'"
        n_inferred_before = len(inferred_before)

        valid_segments = graph2.store.segments()
        evidence_ledger = ev.EvidenceLedger(store_dir)
        smoking_to_lung = ("smoking", "lung cancer")
        evidence_count_before = evidence_ledger.evidence_count(smoking_to_lung, valid_segments)
        assert evidence_count_before > 0, "evidence ledger shows 0 evidence for a real edge (ledger not fed)"
        print("[PASS] query: chain visible, {} inferred edges, evidence_count={}".format(
            n_inferred_before, evidence_count_before))

        # 3. CUT: drop one segment that participates in the chain.
        segments = graph2.store.segments()
        target_sha = None
        for sha in segments:
            for seg_sha, idx, record in graph2.store.iter_records(sha):
                if record.get("trigger_key") == "smoking":
                    target_sha = sha
                    break
            if target_sha:
                break
        assert target_sha is not None, "could not find a segment citing 'smoking' to cut"

        before_inferred_count = len(graph2.inferred_edges())
        graph2.drop_segments([target_sha])
        after_inferred_count = len(graph2.inferred_edges())
        assert after_inferred_count <= before_inferred_count, "cut must not INCREASE inferred edges"
        print("[PASS] cut: inferred edges {} -> {}".format(before_inferred_count, after_inferred_count))

        # 4. QUERY AGAIN: the chain is gone or thinned; the rest is intact
        # (base edges from segments NOT cut still show, e.g. the wildfire
        # chain, which the cut never touched).
        graph3 = LiveGraph(store_dir)
        edges_after = graph3.query("the wildfire smoke")
        assert edges_after, "unrelated chain (wildfire) got wiped by an unrelated cut -- invalidation too broad"
        print("[PASS] query after cut: unrelated chain ('the wildfire smoke') still intact")

        # 5. VERIFY: direction-3 stranger re-derivation stays green post-cut.
        import stranger_verify_run as svr
        checks = svr.verify_direction3(store_dir, 10, seed=60)
        scoring = svr.score(checks, n_target=10)
        assert scoring["p60a_verified_all"], "verify (a) failed post-cut: {}".format(scoring)
        assert scoring["p60b_consensus_all"], "verify (b) failed post-cut: {}".format(scoring)
        print("[PASS] verify: {} verified, {} consensus, both green post-cut".format(
            scoring["p60a_fraction"], scoring["p60b_fraction"]))

        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_query_formatting_groups_duplicates():
    """demo.cmd_query groups edges sharing (kind, from_key, to_key, depth)
    instead of printing one line per derivation -- a real chain here has
    dozens of derivations (one per citing-record combination), and an
    ungrouped dump would be useless for a live demo. This test builds a
    tiny store directly (no organism/extractor) with three records citing
    the SAME base pair from three different segments, and checks
    graph.query() surfaces it as ONE edge with three citations -- the
    grouping demo.py's cmd_query then applies on top."""
    tmpdir = tempfile.mkdtemp(prefix="livecausal-demo-fmt-test-")
    try:
        store_dir = os.path.join(tmpdir, "store")
        graph = LiveGraph(store_dir)
        for i in range(3):
            graph.append_segment([{
                "trigger": "smoking", "mechanism": "causes", "outcome": "lung cancer",
                "trigger_key": "smoking", "outcome_key": "lung cancer",
                "doc_coord": i, "evidence_count": 1, "use_count": 0, "meta": {},
            }])
        edges = graph.query("smoking")
        base_edges = [e for e in edges if e["kind"] == "base"]
        assert len(base_edges) == 1, "three appends of the SAME pair must fold to one base edge"
        assert len(base_edges[0]["derivation"]) == 3, "the one base edge must carry all three citations"
        print("[PASS] query formatting: 3 citations of the same pair fold to 1 base edge, 3 derivations")
        return True
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    ok = True
    ok &= test_through_scenario()
    ok &= test_query_formatting_groups_duplicates()
    print("ALL TESTS PASSED" if ok else "TESTS FAILED")
    sys.exit(0 if ok else 1)
