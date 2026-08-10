"""P71 measurement run (analysis/PREDICTIONS.md, "the live graph never
rebuilds"). Runs the three registered clauses and writes
results/livecausal_p71.json. Harness only -- does not write to
analysis/PREDICTIONS.md, does not commit.

Run: OMP_NUM_THREADS=1 nice -n 15 python3 src/livecausal/p71_run.py
"""

import hashlib
import json
import os
import random
import shutil
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.infer import LiveGraph
from livecausal.store import LiveStore, canonical_bytes

RESULTS_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "results",
    "livecausal_p71.json",
)


def make_record(from_key, to_key, doc_coord=0):
    return {
        "trigger": "trig::{}".format(from_key),
        "mechanism": "causes",
        "outcome": "outc::{}".format(to_key),
        "trigger_key": from_key,
        "outcome_key": to_key,
        "doc_coord": doc_coord,
        "evidence_count": 1,
        "use_count": 0,
        "meta": {},
    }


def chain_segments(rng, n_segments, key_prefix, records_per_segment_range=(1, 2)):
    """n_segments segments of chain-linked records (single global chain of
    keys, hopping through n_segments segments, each segment holding 1-2
    hops -- "chain-mix" matching the robustness tests used in MVP-2). Keys
    are prefixed to keep independent chain_segments() calls collision-free
    within one store.
    """
    segments = []
    counter = [0]

    def _fresh():
        counter[0] += 1
        return "{}K{}".format(key_prefix, counter[0])

    cursor = _fresh()
    for _ in range(n_segments):
        hops = rng.randint(*records_per_segment_range)
        records = []
        for i in range(hops):
            nxt = _fresh()
            records.append(make_record(cursor, nxt, doc_coord=i))
            cursor = nxt
        segments.append(records)
    return segments


def branching_segments(rng, n_segments, key_prefix):
    """A mix of linear-chain and branching (diamond-style) segments, so the
    equivalence check exercises multi-derivation edges too (per the
    robustness sweep done in MVP-2, not just single linear chains).
    """
    segments = []
    counter = [0]

    def _fresh():
        counter[0] += 1
        return "{}B{}".format(key_prefix, counter[0])

    frontier = [_fresh()]
    for _ in range(n_segments):
        src = rng.choice(frontier)
        dst = _fresh()
        segments.append([make_record(src, dst, doc_coord=0)])
        frontier.append(dst)
        if len(frontier) > 6:
            frontier = frontier[-6:]
    return segments


def canonical_inferred_bytes(graph):
    """sha256-comparable canonical serialization of a graph's sorted
    inferred-edge set: JSON-Lines over the edges, sort_keys, same
    convention as LiveStore's own canonical_bytes (reused directly).
    """
    edges = graph.inferred_edges()
    return canonical_bytes(edges)


def fresh_store_dir():
    return tempfile.mkdtemp(prefix="livecausal-p71-")


# ----------------------------------------------------------------------
# (a) Equivalence at scale
# ----------------------------------------------------------------------

def run_clause_a():
    rng = random.Random(71001)
    n_segments = 40
    # ~50 records total across 40 segments: mostly 1-hop, some 2-hop.
    segments = chain_segments(rng, n_segments, key_prefix="A", records_per_segment_range=(1, 2))
    n_records = sum(len(s) for s in segments)

    incr_dir = fresh_store_dir()
    batch_dir = fresh_store_dir()
    try:
        incr_graph = LiveGraph(incr_dir)
        for records in segments:
            sha = incr_graph.store.append_segment(records)
            incr_graph.on_append(sha)

        batch_store = LiveStore(batch_dir)
        for records in segments:
            batch_store.append_segment(records)
        batch_graph = LiveGraph(batch_dir)  # fresh mount -> forces full rebuild

        incr_bytes = canonical_inferred_bytes(incr_graph)
        batch_bytes = canonical_inferred_bytes(batch_graph)
        incr_sha = hashlib.sha256(incr_bytes).hexdigest()
        batch_sha = hashlib.sha256(batch_bytes).hexdigest()

        n_inferred_incr = len(incr_graph.inferred_edges())
        n_inferred_batch = len(batch_graph.inferred_edges())

        passed = (
            incr_sha == batch_sha
            and batch_graph.was_rebuilt_on_mount() is True
            and n_inferred_incr == n_inferred_batch
            and n_inferred_incr > 0
        )

        return {
            "n_segments": n_segments,
            "n_records": n_records,
            "incremental_sha256": incr_sha,
            "batch_sha256": batch_sha,
            "n_inferred_incremental": n_inferred_incr,
            "n_inferred_batch": n_inferred_batch,
            "batch_was_full_rebuild": batch_graph.was_rebuilt_on_mount(),
            "p71a_pass": passed,
        }
    finally:
        shutil.rmtree(incr_dir, ignore_errors=True)
        shutil.rmtree(batch_dir, ignore_errors=True)


# ----------------------------------------------------------------------
# (b) Delta scaling
# ----------------------------------------------------------------------

def _build_base_graph(store_dir, n_segments, seed):
    rng = random.Random(seed)
    segments = chain_segments(rng, n_segments, key_prefix="S", records_per_segment_range=(1, 2))
    graph = LiveGraph(store_dir)
    for records in segments:
        sha = graph.store.append_segment(records)
        graph.on_append(sha)
    return graph, segments


def _fixed_delta_records(anchor_key, delta_size=50):
    """A fixed-size, fixed-shape 50-record delta segment: one long chain
    hanging off anchor_key, deterministic regardless of graph size (same
    delta content appended in every graph-size condition)."""
    records = []
    cursor = anchor_key
    for i in range(delta_size):
        nxt = "DELTA{}".format(i)
        records.append(make_record(cursor, nxt, doc_coord=i))
        cursor = nxt
    return records


def _anchor_key_for(n_segments, seed):
    # Re-derive the first key chain_segments() would have minted, without
    # re-running the RNG against a live graph -- matches "S" prefix, K1.
    return "SK1"


def run_clause_b(n_repeats=5):
    sizes = [5, 10, 20, 40, 80]
    per_size = {}
    for n_segments in sizes:
        timings = []
        n_new_inferred_last = None
        for rep in range(n_repeats):
            store_dir = fresh_store_dir()
            try:
                seed = 71100 + n_segments * 100 + rep
                graph, _segments = _build_base_graph(store_dir, n_segments, seed)
                anchor = _anchor_key_for(n_segments, seed)
                delta_records = _fixed_delta_records(anchor, delta_size=50)
                sha = graph.store.append_segment(delta_records)

                t0 = time.perf_counter()
                new_edges = graph.on_append(sha)
                t1 = time.perf_counter()

                timings.append(t1 - t0)
                n_new_inferred_last = len(new_edges)
            finally:
                shutil.rmtree(store_dir, ignore_errors=True)

        per_size[n_segments] = {
            "wall_seconds_raw": timings,
            "wall_seconds_median": statistics.median(timings),
            "n_new_inferred_last_rep": n_new_inferred_last,
        }

    t5 = per_size[5]["wall_seconds_median"]
    t80 = per_size[80]["wall_seconds_median"]
    ratio_80_over_5 = (t80 / t5) if t5 > 0 else float("inf")
    passed = ratio_80_over_5 <= 4.0

    return {
        "sizes": sizes,
        "n_repeats": n_repeats,
        "per_size": {str(k): v for k, v in per_size.items()},
        "t5_median_seconds": t5,
        "t80_median_seconds": t80,
        "ratio_t80_over_t5": ratio_80_over_5,
        "p71b_pass": passed,
    }


# ----------------------------------------------------------------------
# (c) Truncation is a scan
# ----------------------------------------------------------------------

def run_clause_c(n_repeats=5):
    n_segments = 80

    # Reference append cost at the same graph size (80 segments), for the
    # "no more wall-time than one append at the same graph size" clause --
    # reuse clause (b)'s own 80-segment timing distribution as the
    # same-size append reference, rebuilt fresh here for a self-contained
    # measurement (independent RNG stream, same recipe).
    append_timings = []
    for rep in range(n_repeats):
        store_dir = fresh_store_dir()
        try:
            seed = 71200 + rep
            graph, _segments = _build_base_graph(store_dir, n_segments, seed)
            anchor = _anchor_key_for(n_segments, seed)
            delta_records = _fixed_delta_records(anchor, delta_size=50)
            sha = graph.store.append_segment(delta_records)
            t0 = time.perf_counter()
            graph.on_append(sha)
            t1 = time.perf_counter()
            append_timings.append(t1 - t0)
        finally:
            shutil.rmtree(store_dir, ignore_errors=True)

    # Truncation timing + closure-call instrumentation + bit-equality
    # against a batch rebuild without the dropped segment.
    drop_timings = []
    closure_calls_during_drop = []
    bit_equal_flags = []
    for rep in range(n_repeats):
        incr_dir = fresh_store_dir()
        batch_dir = fresh_store_dir()
        try:
            seed = 71300 + rep
            # _build_base_graph doesn't hand back segment shas; build here
            # directly with explicit sha tracking so we know which mid
            # segment to drop both incrementally and in the batch ref.
            rng = random.Random(seed)
            segs = chain_segments(rng, n_segments, key_prefix="S", records_per_segment_range=(1, 2))
            graph = LiveGraph(incr_dir, count_closures=True)
            shas = []
            for records in segs:
                sha = graph.store.append_segment(records)
                shas.append(sha)
                graph.on_append(sha)

            mid_idx = n_segments // 2
            dropped_sha = shas[mid_idx]

            calls_before = graph.closure_calls
            t0 = time.perf_counter()
            graph.drop_segments([dropped_sha])
            t1 = time.perf_counter()
            calls_after = graph.closure_calls

            drop_timings.append(t1 - t0)
            closure_calls_during_drop.append(calls_after - calls_before)

            batch_store = LiveStore(batch_dir)
            for i, records in enumerate(segs):
                if i == mid_idx:
                    continue
                batch_store.append_segment(records)
            batch_graph = LiveGraph(batch_dir)

            incr_bytes = canonical_inferred_bytes(graph)
            batch_bytes = canonical_inferred_bytes(batch_graph)
            bit_equal_flags.append(
                hashlib.sha256(incr_bytes).hexdigest() == hashlib.sha256(batch_bytes).hexdigest()
            )
        finally:
            shutil.rmtree(incr_dir, ignore_errors=True)
            shutil.rmtree(batch_dir, ignore_errors=True)

    append_median = statistics.median(append_timings)
    drop_median = statistics.median(drop_timings)
    zero_new_closures = all(c == 0 for c in closure_calls_during_drop)
    all_bit_equal = all(bit_equal_flags)
    no_more_than_append = drop_median <= append_median

    passed = zero_new_closures and all_bit_equal and no_more_than_append

    return {
        "n_segments": n_segments,
        "n_repeats": n_repeats,
        "append_wall_seconds_raw": append_timings,
        "append_wall_seconds_median": append_median,
        "drop_wall_seconds_raw": drop_timings,
        "drop_wall_seconds_median": drop_median,
        "drop_no_more_than_append": no_more_than_append,
        "closure_calls_during_drop_per_rep": closure_calls_during_drop,
        "zero_new_closures_on_drop": zero_new_closures,
        "bit_equal_to_batch_without_segment_per_rep": bit_equal_flags,
        "all_bit_equal_to_batch": all_bit_equal,
        "p71c_pass": passed,
    }


def main():
    started = time.time()
    result_a = run_clause_a()
    result_b = run_clause_b()
    result_c = run_clause_c()
    finished = time.time()

    out = {
        "prediction": "P71",
        "spec_ref": "analysis/PREDICTIONS.md P71",
        "engine_commit": "ca2b82a",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "clause_a_equivalence_at_scale": result_a,
        "clause_b_delta_scaling": result_b,
        "clause_c_truncation_is_scan": result_c,
        "p71a_pass": result_a["p71a_pass"],
        "p71b_pass": result_b["p71b_pass"],
        "p71c_pass": result_c["p71c_pass"],
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    print("wrote {}".format(RESULTS_PATH))
    print("p71a_pass:", result_a["p71a_pass"])
    print("p71b_pass:", result_b["p71b_pass"], "ratio_t80_over_t5:", result_b["ratio_t80_over_t5"])
    print("p71c_pass:", result_c["p71c_pass"])


if __name__ == "__main__":
    main()
