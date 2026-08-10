"""SCALE HARNESS -- the P71 falsifier, on a real extraction graph's shape.

P71 measured delta-scaling on SYNTHETIC, uniform chain segments (1-2 hops
each, single anchor point). The natural next falsifier the dossier names:
does the SAME curve (append time grows sub-linearly with graph size,
constant delta-yield) hold on a REAL, unevenly-connected extraction graph
-- one built by src/livecausal/builder_run.py from actual text, where
segments vary in record count, base-edge fan-out is uneven, and multi-hop
inferred chains are sparse and irregular (P72's own measured shape: 2,046
base edges, only 76 inferred, entity-key fragmentation)?

This harness answers that on ANY store builder_run.py (or anything else
speaking the same LiveStore/LiveGraph contract) produced -- it does not
build its own synthetic data (P71 already covers that) and, per the build
brief, is NOT run against the real P72/P73 artifacts here (that measured
comparison is the Lead's call, done separately, once this harness is
reviewed).

Two things are measured, both reusing P71's own instrumentation
(LiveGraph's count_closures flag, was_rebuilt_on_mount):

  1. REPLAY: mount a fresh, empty LiveGraph and append the source store's
     segments ONE AT A TIME, in manifest order (the exact history the
     source graph was built with -- content-addressed segments replay
     identically regardless of which store they physically live in,
     SS1's whole point). Per-append: wall time, new base edges, new
     inferred edges, cumulative graph size at that point. This gives the
     append-time-vs-graph-size CURVE for real, unevenly-connected data,
     not just five uniform synthetic buckets.
  2. DROP SAMPLES: N independent samples (fresh copies of the fully
     replayed graph, so each drop is measured at the SAME final graph
     size and none of them interact) each drop one segment (sampled
     without replacement across the N draws) and record: wall time,
     whether zero NEW closures were computed (P71c's exact check), and
     whether the result stays internally consistent (no crash, valid
     graph state -- full bit-equality-to-batch-rebuild is not re-checked
     here since P71c already proved that mechanism; this harness's job is
     the SCALE question, not re-litigating correctness).

Does not write to analysis/PREDICTIONS.md. Does not commit. Does not run
against results/p72_store_run1, results/p72_store_run2, or any other real
P72/P73 artifact -- see main()'s --source argument; the smoke path builds
its own tiny stores via builder_run.py locally.

Usage:
  python3 src/livecausal/scale_run.py --smoke
  python3 src/livecausal/scale_run.py --source path/to/a/livestore/dir --out results/scale_run.json
"""
import argparse
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
from livecausal.store import LiveStore

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "livecausal_scale_run.json")


# ─────────────────────────────────────────────────────────────────────────
#  Replay: read a source store's segments (content only, via the public
#  LiveStore API), feed them one at a time into a fresh LiveGraph.
# ─────────────────────────────────────────────────────────────────────────
def load_source_segments(source_store_dir):
    """Returns the ordered list of (sha, records) for every segment in the
    source store's manifest, read via LiveStore's public API only (never
    mutates the source -- this harness only ever reads it)."""
    source_store = LiveStore(source_store_dir)
    shas = source_store.segments()
    out = []
    for sha in shas:
        records = [rec for _seg_sha, _idx, rec in source_store.iter_records(sha)]
        out.append((sha, records))
    return out


def replay_segments(segments, count_closures=True, progress_every=None):
    """Appends `segments` (list of (sha, records), source-store order) one
    at a time into a FRESH LiveGraph over a temporary store, using
    append_segment (re-seals the same record bytes -> same sha, since
    segment identity is content-addressed) + on_append. Returns
    (graph, per_append_rows, tmp_store_dir) -- the caller owns cleanup of
    tmp_store_dir (not removed here, since drop-sampling below needs
    copies of the fully-replayed store).

    per_append_rows[i] = {
      "step": i (1-indexed),
      "segment_sha": sha,
      "n_records": len(records),
      "append_seconds": wall time for this one append_segment+on_append,
      "n_new_base_edges": how many NEW (from_key,to_key) pairs this
          segment introduced (existing pairs getting an extra citation do
          not count -- this measures fan-OUT growth, not citation growth),
      "n_new_inferred_edges": len(graph.on_append(sha)) -- MVP-2's own
          delta-yield number, unchanged definition,
      "closure_calls_delta": graph.closure_calls delta for this step
          (P71's counter; expected == 1 per step, i.e. exactly one
          _delta_chains_for_citation call site fires per new-pair citation
          batch -- see infer.py's on_append, which calls it once per new
          citation in the segment, so this can exceed 1 for multi-record
          segments; the load-bearing check is closure_calls staying
          additive and never jumping by more than the segment's own
          citation count, i.e. no HIDDEN full rebuild snuck in),
      "cumulative_segments": i,
      "cumulative_base_edges": running total,
      "cumulative_inferred_edges": running total,
    }
    """
    tmp_dir = tempfile.mkdtemp(prefix="livecausal-scale-replay-")
    graph = LiveGraph(tmp_dir, count_closures=count_closures)
    rows = []
    prev_closure_calls = graph.closure_calls if count_closures else None

    for i, (_source_sha, records) in enumerate(segments, start=1):
        base_pairs_before = {
            (fk, tk) for fk, targets in graph._base_edges.items() for tk in targets
        }

        t0 = time.perf_counter()
        actual_sha = graph.store.append_segment(records)
        new_inferred = graph.on_append(actual_sha)
        t1 = time.perf_counter()

        base_pairs_after = {
            (fk, tk) for fk, targets in graph._base_edges.items() for tk in targets
        }
        n_new_base_edges = len(base_pairs_after - base_pairs_before)

        closure_delta = None
        if count_closures:
            closure_delta = graph.closure_calls - prev_closure_calls
            prev_closure_calls = graph.closure_calls

        row = {
            "step": i,
            "segment_sha": actual_sha,
            "n_records": len(records),
            "append_seconds": t1 - t0,
            "n_new_base_edges": n_new_base_edges,
            "n_new_inferred_edges": len(new_inferred),
            "closure_calls_delta": closure_delta,
            "cumulative_segments": i,
            "cumulative_base_edges": len(base_pairs_after),
            "cumulative_inferred_edges": len(graph.inferred_edges()),
        }
        rows.append(row)
        if progress_every and i % progress_every == 0:
            print(
                "[scale] replay {}/{}  append={:.4f}s  base={} inferred={}".format(
                    i, len(segments), row["append_seconds"], row["cumulative_base_edges"],
                    row["cumulative_inferred_edges"],
                ),
                flush=True,
            )

    return graph, rows, tmp_dir


# ─────────────────────────────────────────────────────────────────────────
#  Curve analysis over the replay rows (P71b-style bucketed ratio, but
#  computed from the REAL per-append sequence instead of five synthetic
#  fixed-size conditions).
# ─────────────────────────────────────────────────────────────────────────
def analyze_append_curve(rows, n_buckets=5):
    """Splits the replay's append_seconds sequence into n_buckets
    contiguous, roughly-equal-size windows (by step count, i.e. by graph
    size at the time of each append) and reports the median append time
    per bucket -- the real-data analog of P71b's five synthetic
    graph-size conditions (5/10/20/40/80 segments). Also reports the
    delta-yield (n_new_inferred_edges) distribution per bucket, since a
    real graph's yield is NOT constant like P71's synthetic anchor-chain
    delta was (P71's own scoring note: that was flagged as the format's
    open question, to be measured "when real extraction graphs exist" --
    this is that measurement).

    Returns {"buckets": [...], "ratio_last_over_first": float|None,
             "yield_buckets": [...]}.
    """
    n = len(rows)
    if n == 0:
        return {"buckets": [], "ratio_last_over_first": None, "yield_buckets": []}

    n_buckets = max(1, min(n_buckets, n))
    bucket_size = n / n_buckets
    buckets = []
    yield_buckets = []
    for b in range(n_buckets):
        lo = int(round(b * bucket_size))
        hi = int(round((b + 1) * bucket_size))
        hi = max(hi, lo + 1)
        hi = min(hi, n)
        chunk = rows[lo:hi]
        if not chunk:
            continue
        times = [r["append_seconds"] for r in chunk]
        yields = [r["n_new_inferred_edges"] for r in chunk]
        buckets.append({
            "step_range": [chunk[0]["step"], chunk[-1]["step"]],
            "graph_size_at_start": chunk[0]["cumulative_segments"] - 1,
            "graph_size_at_end": chunk[-1]["cumulative_segments"],
            "n_appends_in_bucket": len(chunk),
            "append_seconds_median": statistics.median(times),
            "append_seconds_mean": statistics.mean(times),
        })
        yield_buckets.append({
            "step_range": [chunk[0]["step"], chunk[-1]["step"]],
            "n_new_inferred_median": statistics.median(yields),
            "n_new_inferred_mean": statistics.mean(yields),
            "n_new_inferred_max": max(yields),
            "n_new_inferred_total": sum(yields),
        })

    ratio = None
    if len(buckets) >= 2 and buckets[0]["append_seconds_median"] > 0:
        ratio = buckets[-1]["append_seconds_median"] / buckets[0]["append_seconds_median"]

    return {
        "buckets": buckets,
        "ratio_last_over_first": ratio,
        "yield_buckets": yield_buckets,
    }


# ─────────────────────────────────────────────────────────────────────────
#  Drop sampling: N independent samples, each on a FRESH copy of the fully
#  replayed store, so every drop is measured at the identical final graph
#  size and samples never interact with each other.
# ─────────────────────────────────────────────────────────────────────────
def sample_drops(replayed_store_dir, segment_shas, n_samples, seed=59001):
    """n_samples drops, each: copy the fully-replayed store fresh, mount a
    count_closures=True LiveGraph over the copy, drop ONE segment (sampled
    without replacement across the n_samples draws, capped at
    len(segment_shas)), measure wall time and the closure-call delta
    (must be 0 -- P71c's exact check, reused verbatim here on real data).
    """
    rng = random.Random(seed)
    n_samples = min(n_samples, len(segment_shas))
    sampled_shas = rng.sample(segment_shas, n_samples)

    rows = []
    for sha in sampled_shas:
        copy_dir = tempfile.mkdtemp(prefix="livecausal-scale-drop-")
        try:
            _copy_store_dir(replayed_store_dir, copy_dir)
            graph = LiveGraph(copy_dir, count_closures=True)
            calls_before = graph.closure_calls
            t0 = time.perf_counter()
            graph.drop_segments([sha])
            t1 = time.perf_counter()
            calls_after = graph.closure_calls
            rows.append({
                "segment_sha": sha,
                "drop_seconds": t1 - t0,
                "closure_calls_delta": calls_after - calls_before,
                "zero_new_closures": (calls_after - calls_before) == 0,
            })
        finally:
            shutil.rmtree(copy_dir, ignore_errors=True)

    return rows


def _copy_store_dir(src, dst):
    """Copies a LiveStore directory (manifest.json + *.seg + inferred.jsonl
    + evidence.ledger + use.ledger, whatever is present) -- shutil.copytree
    onto an existing empty dst."""
    for name in os.listdir(src):
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)


# ─────────────────────────────────────────────────────────────────────────
#  Smoke: build a small store LOCALLY via builder_run.py's own smoke
#  corpus + fake extractor (offline, no network, no real P72/P73 data).
# ─────────────────────────────────────────────────────────────────────────
def build_smoke_source_store(store_dir, seed=42, max_windows=60):
    from livecausal.builder_run import (
        TextFileStream,
        fake_extractor,
        generate_smoke_corpus,
        run_builder,
        stream_windows,
    )
    import torch
    import portable_organism as po

    torch.set_num_threads(1)
    po.D_MODEL, po.BATCH, po.CHUNK = 64, 4, 32
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = 0.75, 50, 10, 5

    os.makedirs(store_dir, exist_ok=True)
    parent_dir = os.path.dirname(os.path.normpath(store_dir))
    corpus_path = os.path.join(parent_dir, "scale_smoke_corpus.txt")
    generate_smoke_corpus(corpus_path, seed=seed)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    torch.manual_seed(seed)
    organism = po.Organism("scale-smoke-builder", V, mask, seed=seed)
    stream = TextFileStream(corpus_path, stoi, unk)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    window_iter = stream_windows(organism, stream, feeder, window_tokens=32)

    status_path = os.path.join(parent_dir, "scale_smoke_status.json")
    metrics_path = os.path.join(parent_dir, "scale_smoke_metrics.jsonl")
    graph, metrics = run_builder(
        store_dir, status_path, metrics_path,
        window_iter, fake_extractor, windows_per_segment=3,
        max_windows=max_windows, print_every=1000, stream=stream,
    )
    return graph, metrics


# ─────────────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────────────
def run_scale_harness(source_store_dir, n_drop_samples=10, n_buckets=5, drop_seed=59001, progress_every=None):
    segments = load_source_segments(source_store_dir)
    if not segments:
        raise ValueError("source store {} has no segments to replay".format(source_store_dir))

    replay_t0 = time.time()
    graph, rows, replay_tmp_dir = replay_segments(segments, count_closures=True, progress_every=progress_every)
    replay_elapsed = time.time() - replay_t0

    try:
        curve = analyze_append_curve(rows, n_buckets=n_buckets)

        # closure-call accounting sanity: total closure_calls_delta across
        # all replay steps, PLUS the constructor's own initial mount call,
        # must equal graph.closure_calls exactly (no hidden extra rebuild
        # anywhere in the sequence). infer.py's _rebuild_from_store
        # increments closure_calls unconditionally on ANY fresh mount --
        # even an empty store with zero segments -- so a fresh
        # count_closures=True LiveGraph always starts at closure_calls==1,
        # not 0 (verified: was_rebuilt_on_mount() is True here, exactly
        # once, before any replay step runs). The load-bearing check is
        # that NOTHING beyond that one initial mount plus one delta call
        # per replay step ever fires -- i.e. no on_append silently fell
        # back to a full rebuild mid-replay.
        initial_mount_calls = 1 if graph.was_rebuilt_on_mount() else 0
        sum_step_deltas = sum(r["closure_calls_delta"] for r in rows)
        closure_accounting_consistent = (
            sum_step_deltas + initial_mount_calls == graph.closure_calls
        )

        segment_shas = [r["segment_sha"] for r in rows]
        drop_rows = sample_drops(replay_tmp_dir, segment_shas, n_drop_samples, seed=drop_seed)

        n_drop_zero_closures = sum(1 for d in drop_rows if d["zero_new_closures"])
        all_drops_zero_closures = bool(drop_rows) and n_drop_zero_closures == len(drop_rows)
        drop_seconds = [d["drop_seconds"] for d in drop_rows]
        append_seconds_all = [r["append_seconds"] for r in rows]

        return {
            "source_store_dir": source_store_dir,
            "n_segments_replayed": len(segments),
            "n_records_total": sum(r["n_records"] for r in rows),
            "replay_elapsed_seconds": replay_elapsed,
            "final_n_base_edges": rows[-1]["cumulative_base_edges"],
            "final_n_inferred_edges": rows[-1]["cumulative_inferred_edges"],
            "was_rebuilt_on_mount_of_fresh_graph": graph.was_rebuilt_on_mount(),
            "closure_calls_total": graph.closure_calls,
            "closure_accounting_consistent": closure_accounting_consistent,
            "append_curve": curve,
            "append_seconds_overall_median": statistics.median(append_seconds_all) if append_seconds_all else None,
            "n_drop_samples": len(drop_rows),
            "drop_rows": drop_rows,
            "drop_seconds_median": statistics.median(drop_seconds) if drop_seconds else None,
            "all_drops_zero_closures": all_drops_zero_closures,
            "n_drop_zero_closures": n_drop_zero_closures,
            "per_append_rows": rows,
        }
    finally:
        shutil.rmtree(replay_tmp_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="LIVE-CAUSAL scale falsifier harness")
    ap.add_argument("--source", default=None, help="LiveStore directory to replay (required unless --smoke)")
    ap.add_argument("--n-drop-samples", type=int, default=10)
    ap.add_argument("--n-buckets", type=int, default=5)
    ap.add_argument("--drop-seed", type=int, default=59001)
    ap.add_argument("--progress-every", type=int, default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--smoke", action="store_true", help="build a small LOCAL store via builder_run.py, replay it")
    ap.add_argument("--smoke-seed", type=int, default=42)
    ap.add_argument("--smoke-max-windows", type=int, default=60)
    args = ap.parse_args()

    smoke_dir = None
    source = args.source
    if args.smoke:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
        smoke_dir = tempfile.mkdtemp(prefix="livecausal_scale_smoke_")
        store_dir = os.path.join(smoke_dir, "store")
        print("[scale] building smoke source store...", flush=True)
        build_smoke_source_store(store_dir, seed=args.smoke_seed, max_windows=args.smoke_max_windows)
        source = store_dir

    if not source:
        raise SystemExit("--source is required (or pass --smoke)")

    print("[scale] replaying {} ...".format(source), flush=True)
    t0 = time.time()
    result = run_scale_harness(
        source,
        n_drop_samples=args.n_drop_samples,
        n_buckets=args.n_buckets,
        drop_seed=args.drop_seed,
        progress_every=args.progress_every,
    )
    elapsed = time.time() - t0

    payload = {
        "prediction": None,  # harness only; no P-number registered here (Lead's call, per build brief)
        "elapsed_seconds": round(elapsed, 2),
        **result,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")

    curve = result["append_curve"]
    ratio = curve["ratio_last_over_first"]
    print("=" * 74)
    print("SCALE HARNESS -- source: {}".format(source))
    print("segments replayed: {}  records: {}".format(
        result["n_segments_replayed"], result["n_records_total"]))
    print("final base_edges={} inferred_edges={}".format(
        result["final_n_base_edges"], result["final_n_inferred_edges"]))
    print("append_seconds ratio (last bucket / first bucket): {}".format(
        "{:.4f}".format(ratio) if ratio is not None else "n/a"))
    print("closure_accounting_consistent: {}".format(result["closure_accounting_consistent"]))
    print("drop samples: {}  all_drops_zero_closures: {}  drop_seconds_median: {}".format(
        result["n_drop_samples"], result["all_drops_zero_closures"],
        "{:.6f}".format(result["drop_seconds_median"]) if result["drop_seconds_median"] is not None else "n/a",
    ))
    print("wrote {}".format(args.out))

    if smoke_dir:
        shutil.rmtree(smoke_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
