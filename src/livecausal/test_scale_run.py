"""Plain-assert tests for the scale falsifier harness
(src/livecausal/scale_run.py). Run: python3 src/livecausal/test_scale_run.py

Offline-only, self-built local corpora -- never touches results/p72_* or
any other real P72/P73 artifact (per the build brief).
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
from livecausal.store import LiveStore, canonical_bytes  # noqa: E402
from livecausal.p71_run import chain_segments, branching_segments  # noqa: E402
from livecausal.scale_run import (  # noqa: E402
    analyze_append_curve,
    load_source_segments,
    replay_segments,
    run_scale_harness,
    sample_drops,
)


def with_tmpdir(fn):
    d = tempfile.mkdtemp(prefix="scale-test-")
    try:
        fn(d)
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _make_source_store(d, seed=1, n_segments=30):
    """A small synthetic source store (branching + chain mix, so replay
    exercises multi-derivation edges, not just a single linear chain --
    p71_run.py's own generators, reused directly rather than re-invented)."""
    rng = random.Random(seed)
    store_dir = os.path.join(d, "source_store")
    store = LiveStore(store_dir)
    segs_chain = chain_segments(rng, n_segments // 2, key_prefix="C")
    segs_branch = branching_segments(rng, n_segments // 2, key_prefix="B")
    all_segs = segs_chain + segs_branch
    rng.shuffle(all_segs)
    for records in all_segs:
        store.append_segment(records)
    return store_dir


# ---------------------------------------------------------------------
# Test 1: replay produces the SAME graph a direct build would (same
# store, same manifest order -- content-addressed segments replay
# identically regardless of which physical directory holds them).
# ---------------------------------------------------------------------

def test_replay_matches_direct_build(d):
    source_store_dir = _make_source_store(d, seed=1)

    segments = load_source_segments(source_store_dir)
    assert segments, "fixture produced no segments"

    replayed_graph, rows, replay_tmp = replay_segments(segments, count_closures=True)
    try:
        replayed_edges = sorted(
            (e["from_key"], e["to_key"], e["depth"], tuple(map(tuple, e["derivation"])))
            for e in replayed_graph.inferred_edges()
        )

        # Direct build: append the SAME segments (by content) straight
        # into a fresh graph over the ORIGINAL source store dir's own
        # content -- i.e. just mount the source store itself.
        direct_graph = LiveGraph(source_store_dir)
        direct_edges = sorted(
            (e["from_key"], e["to_key"], e["depth"], tuple(map(tuple, e["derivation"])))
            for e in direct_graph.inferred_edges()
        )

        assert len(rows) == len(segments)
        assert replayed_edges == direct_edges, "replay diverged from a direct mount of the same store"
        assert rows[-1]["cumulative_base_edges"] == sum(
            len(v) for v in direct_graph._base_edges.values()
        )
    finally:
        shutil.rmtree(replay_tmp, ignore_errors=True)


# ---------------------------------------------------------------------
# Test 2: closure-call accounting is exact -- sum of per-step deltas plus
# the one initial (empty-store) mount call equals the graph's own total.
# ---------------------------------------------------------------------

def test_closure_accounting_is_exact(d):
    source_store_dir = _make_source_store(d, seed=2, n_segments=20)
    segments = load_source_segments(source_store_dir)

    graph, rows, tmp = replay_segments(segments, count_closures=True)
    try:
        assert graph.was_rebuilt_on_mount() is True, "fresh empty store must report a rebuilt mount"
        sum_deltas = sum(r["closure_calls_delta"] for r in rows)
        assert sum_deltas + 1 == graph.closure_calls, (
            "closure_calls_delta sum ({}) + 1 initial mount != graph.closure_calls ({})".format(
                sum_deltas, graph.closure_calls
            )
        )
        # every delta must be non-negative (a closure call count can only grow)
        assert all(r["closure_calls_delta"] >= 0 for r in rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------
# Test 3: drop samples always report zero new closures (P71c's exact
# guarantee, reused on real-shaped data) and are independent -- the SAME
# segment dropped from N separate copies gives byte-identical resulting
# inferred-edge sets each time.
# ---------------------------------------------------------------------

def test_drop_samples_are_zero_closure_and_independent(d):
    source_store_dir = _make_source_store(d, seed=3, n_segments=20)
    segments = load_source_segments(source_store_dir)
    graph, rows, tmp = replay_segments(segments, count_closures=True)
    try:
        shas = [r["segment_sha"] for r in rows]
        drop_rows = sample_drops(tmp, shas, n_samples=5, seed=999)
        assert len(drop_rows) == 5
        assert all(d_["zero_new_closures"] for d_ in drop_rows), drop_rows
        assert all(d_["closure_calls_delta"] == 0 for d_ in drop_rows)

        # Determinism: dropping the SAME sha from two independent fresh
        # copies of the replayed store gives the same resulting graph.
        target_sha = shas[0]
        results = []
        for _ in range(2):
            copy_dir = tempfile.mkdtemp(prefix="scale-drop-det-")
            try:
                for name in os.listdir(tmp):
                    s = os.path.join(tmp, name)
                    dpath = os.path.join(copy_dir, name)
                    if os.path.isdir(s):
                        shutil.copytree(s, dpath)
                    else:
                        shutil.copy2(s, dpath)
                g = LiveGraph(copy_dir)
                g.drop_segments([target_sha])
                edges = sorted(
                    (e["from_key"], e["to_key"], e["depth"], tuple(map(tuple, e["derivation"])))
                    for e in g.inferred_edges()
                )
                results.append(edges)
            finally:
                shutil.rmtree(copy_dir, ignore_errors=True)
        assert results[0] == results[1], "dropping the same segment from independent copies diverged"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------
# Test 4: analyze_append_curve buckets cover every row exactly once (no
# row dropped, no row double-counted), and the yield totals sum to the
# same total as summing n_new_inferred_edges directly over all rows.
# ---------------------------------------------------------------------

def test_curve_analysis_buckets_are_exhaustive(d):
    source_store_dir = _make_source_store(d, seed=4, n_segments=27)  # not evenly divisible by 5
    segments = load_source_segments(source_store_dir)
    graph, rows, tmp = replay_segments(segments, count_closures=False)
    try:
        curve = analyze_append_curve(rows, n_buckets=5)
        n_covered = sum(b["n_appends_in_bucket"] for b in curve["buckets"])
        assert n_covered == len(rows), "bucket coverage {} != total rows {}".format(n_covered, len(rows))

        total_yield_from_buckets = sum(yb["n_new_inferred_total"] for yb in curve["yield_buckets"])
        total_yield_direct = sum(r["n_new_inferred_edges"] for r in rows)
        assert total_yield_from_buckets == total_yield_direct

        # step ranges must be contiguous and non-overlapping, covering 1..len(rows)
        seen_steps = set()
        for b in curve["buckets"]:
            lo, hi = b["step_range"]
            for s in range(lo, hi + 1):
                assert s not in seen_steps, "step {} covered by more than one bucket".format(s)
                seen_steps.add(s)
        assert seen_steps == set(r["step"] for r in rows)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_curve_analysis_handles_tiny_input(d):
    # 1 row, more buckets requested than rows -- must not crash or divide by zero.
    rows = [{
        "step": 1, "segment_sha": "x", "n_records": 1, "append_seconds": 0.001,
        "n_new_base_edges": 1, "n_new_inferred_edges": 0, "closure_calls_delta": 1,
        "cumulative_segments": 1, "cumulative_base_edges": 1, "cumulative_inferred_edges": 0,
    }]
    curve = analyze_append_curve(rows, n_buckets=5)
    assert len(curve["buckets"]) == 1
    assert curve["ratio_last_over_first"] is None  # only one bucket -- no ratio defined

    empty_curve = analyze_append_curve([], n_buckets=5)
    assert empty_curve["buckets"] == []
    assert empty_curve["ratio_last_over_first"] is None


# ---------------------------------------------------------------------
# Test 5: run_scale_harness end to end on a self-built source store --
# never touches a real P72/P73 artifact, produces the full payload shape
# the build brief's output-JSON spec names, deterministic across two
# independent calls against the SAME source store.
# ---------------------------------------------------------------------

def test_run_scale_harness_end_to_end_and_deterministic(d):
    source_store_dir = _make_source_store(d, seed=5, n_segments=24)

    result1 = run_scale_harness(source_store_dir, n_drop_samples=6, n_buckets=4, drop_seed=42)
    result2 = run_scale_harness(source_store_dir, n_drop_samples=6, n_buckets=4, drop_seed=42)

    for key in (
        "n_segments_replayed", "n_records_total", "final_n_base_edges",
        "final_n_inferred_edges", "closure_calls_total", "closure_accounting_consistent",
        "all_drops_zero_closures", "n_drop_zero_closures",
    ):
        assert result1[key] == result2[key], "{} differs across identical runs: {} vs {}".format(
            key, result1[key], result2[key]
        )

    assert result1["closure_accounting_consistent"] is True
    assert result1["all_drops_zero_closures"] is True
    assert result1["final_n_base_edges"] > 0
    assert len(result1["per_append_rows"]) == result1["n_segments_replayed"]
    assert result1["append_curve"]["buckets"]


def run_all():
    tests = [
        test_replay_matches_direct_build,
        test_closure_accounting_is_exact,
        test_drop_samples_are_zero_closure_and_independent,
        test_curve_analysis_buckets_are_exhaustive,
        test_curve_analysis_handles_tiny_input,
        test_run_scale_harness_end_to_end_and_deterministic,
    ]
    for t in tests:
        with_tmpdir(t)
        print("OK  {}".format(t.__name__))
    print("ALL {} TESTS PASSED".format(len(tests)))


if __name__ == "__main__":
    run_all()
