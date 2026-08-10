"""P75 measurement orchestrator (analysis/PREDICTIONS.md, "canonicalization
lifts the measured constraint"). Runs all four registered clauses against
a WORKING COPY of a source LiveStore -- never the source directory itself
(the DECISIONS read-only-artifact rule: gescorte Stores sind read-only,
Konsumenten mounten immer eine Kopie).

Does NOT import consult_run.py's internals -- canon-organ's Baustelle;
this orchestrator drives it as a SUBPROCESS with --canon, exactly the
"Interface: --canon zusätzlich zu den bekannten Flags" contract.

Clauses (registered P75, analysis/PREDICTIONS.md -- read-only reference,
this file does not write there):
  (a)  WRITE SIDE: LiveGraph(canon=False) inferred-edge count (must equal
       the pre-canon baseline exactly -- canon=False is untouched by
       construction, but this harness measures it directly rather than
       assuming it) vs LiveGraph(canon=True) canon_inferred_edges() count,
       plus the raw-key -> canon-key compression ratio.
  (b)  READ SIDE: consult_run.py --canon as a subprocess against the
       WORKING COPY, --source wt103, collecting coverage and
       mean_delta_real/random from its own output JSON.
  (c)  COST: cold canon mount (cache file deleted first, includes spaCy
       model load) wall time, vs warm re-mount (from the stamped cache)
       wall time.
  (d)  MOAT: N sampled canonical inferred edges re-derived from ONLY their
       cited raw records + canon.canonical_key (env_pin match enforced),
       mirroring stranger_verify_run.verify_direction3's re-derivation
       discipline (test_canon.py's own test_stranger_can_rederive_
       canonical_edge does the same thing for a single edge; this
       generalizes it to a sampled batch over any source store). After
       all canon operations: store.verify() True + every segment sha in
       the working copy unchanged from the source's own manifest.

Does not write to analysis/PREDICTIONS.md. Does not commit. Never touches
a real P72/P73/P74 artifact directly as a mutation target -- --source is
always copied first; deployment against the real registered artifact is
the Lead's own run, on the runner, per the build brief.

Usage:
  python3 src/livecausal/p75_run.py --smoke
  python3 src/livecausal/p75_run.py --source results/p72_store_run1 --out results/p75_run.json
"""
import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.infer import LiveGraph
from livecausal.store import LiveStore
from livecausal import canon as canon_mod

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONSULT_RUN = os.path.join(REPO_ROOT, "src", "livecausal", "consult_run.py")
DEFAULT_OUT = os.path.join(REPO_ROOT, "results", "p75_run.json")


# ─────────────────────────────────────────────────────────────────────────
#  Working-copy discipline: --source is NEVER a write target. Every clause
#  operates on a fresh copy; the source directory is opened only via
#  LiveStore's read-only public API (segments()/iter_records()) when this
#  file needs to compare the copy's manifest against the source's.
# ─────────────────────────────────────────────────────────────────────────
def clone_store(source_dir, dest_dir):
    """Copies a LiveStore directory (manifest.json, *.seg, any existing
    inferred.jsonl/canon_inferred.jsonl/evidence.ledger/use.ledger) into
    dest_dir. dest_dir must not exist yet (created here) -- this never
    writes into source_dir."""
    if os.path.exists(dest_dir):
        raise ValueError("clone_store: dest_dir {} already exists".format(dest_dir))
    shutil.copytree(source_dir, dest_dir)


def source_segment_shas(source_dir):
    """Reads the source's segment list via the public LiveStore API only
    (never mounts a LiveGraph over the source -- no cache files get
    written into it this way either)."""
    return LiveStore(source_dir).segments()


# ─────────────────────────────────────────────────────────────────────────
#  (a) Write side: canon=False baseline vs canon=True lift.
# ─────────────────────────────────────────────────────────────────────────
def measure_write_side(copy_dir):
    t0 = time.perf_counter()
    raw_graph = LiveGraph(copy_dir, canon=False)
    t1 = time.perf_counter()
    n_inferred_raw = len(raw_graph.inferred_edges())
    n_base_raw = sum(len(v) for v in raw_graph._base_edges.values())

    t2 = time.perf_counter()
    canon_graph = LiveGraph(copy_dir, canon=True)
    t3 = time.perf_counter()
    n_inferred_canon = len(canon_graph.canon_inferred_edges())
    n_canon_base_pairs = sum(len(v) for v in canon_graph._canon_base_edges.values())

    # Compression: how many distinct raw keys fold into how many distinct
    # canon keys, over the SAME record set (both from/to key columns).
    raw_keys = set()
    for from_key, targets in raw_graph._base_edges.items():
        raw_keys.add(from_key)
        raw_keys.update(targets.keys())
    canon_keys = set(canon_graph._raw_to_canon.values())
    n_raw_keys = len(raw_keys)
    n_canon_keys = len(canon_keys)
    compression_ratio = (n_raw_keys / n_canon_keys) if n_canon_keys else None

    lift_factor = (n_inferred_canon / n_inferred_raw) if n_inferred_raw else None

    return {
        "n_base_edges_raw": n_base_raw,
        "n_inferred_edges_raw": n_inferred_raw,
        "raw_mount_seconds": t1 - t0,
        "n_canon_base_pairs": n_canon_base_pairs,
        "n_inferred_edges_canon": n_inferred_canon,
        "canon_mount_seconds": t3 - t2,
        "n_distinct_raw_keys": n_raw_keys,
        "n_distinct_canon_keys": n_canon_keys,
        "raw_to_canon_compression_ratio": compression_ratio,
        "lift_factor_canon_over_raw": lift_factor,
        "canon_env_pin": canon_graph.canon_env_pin,
    }, raw_graph, canon_graph


# ─────────────────────────────────────────────────────────────────────────
#  (c) Cost: cold canon mount (cache deleted first, includes model load)
#  vs warm re-mount (from the stamped cache).
# ─────────────────────────────────────────────────────────────────────────
def measure_canon_cost(copy_dir):
    cache_path = os.path.join(copy_dir, "canon_inferred.jsonl")
    if os.path.exists(cache_path):
        os.remove(cache_path)

    t0 = time.perf_counter()
    graph_cold = LiveGraph(copy_dir, canon=True)
    t1 = time.perf_counter()
    cold_seconds = t1 - t0
    cold_was_rebuilt = graph_cold.was_canon_rebuilt_on_mount()

    t2 = time.perf_counter()
    graph_warm = LiveGraph(copy_dir, canon=True)
    t3 = time.perf_counter()
    warm_seconds = t3 - t2
    warm_loaded_from_cache = graph_warm.was_canon_loaded_from_cache()

    return {
        "cold_mount_seconds": cold_seconds,
        "cold_was_rebuilt_on_mount": cold_was_rebuilt,
        "warm_mount_seconds": warm_seconds,
        "warm_loaded_from_cache": warm_loaded_from_cache,
    }


# ─────────────────────────────────────────────────────────────────────────
#  (d) Moat: sampled canonical inferred edges, re-derived from ONLY the
#  cited raw records + canon.canonical_key (env_pin-pinned), mirroring
#  stranger_verify_run.verify_direction3 / test_canon.py's single-edge
#  re-derivation test, generalized to a seeded sample.
# ─────────────────────────────────────────────────────────────────────────
def _load_record(store, sha, idx):
    for seg_sha, rec_idx, record in store.iter_records(sha):
        if rec_idx == idx:
            return record
    return None


def rederive_canon_edge(stranger_store, edge, nlp):
    """Re-derives one canonical inferred edge from ONLY its cited raw
    (segment_sha, idx) records + canon.canonical_key(raw_key, nlp=nlp) --
    no access to the graph's own canon cache or in-memory adjacency.
    Returns (ok: bool, detail: dict)."""
    derivation = edge["derivation"]
    if not derivation:
        return False, {"reason": "empty derivation"}

    rederived_path = []
    for sha, idx in derivation:
        record = _load_record(stranger_store, sha, idx)
        if record is None:
            return False, {"reason": "missing record", "sha": sha, "idx": idx}
        raw_from = record.get("trigger_key")
        raw_to = record.get("outcome_key")
        if raw_from is None or raw_to is None:
            return False, {"reason": "record missing trigger_key/outcome_key", "sha": sha, "idx": idx}
        canon_from = canon_mod.canonical_key(raw_from, nlp=nlp)
        canon_to = canon_mod.canonical_key(raw_to, nlp=nlp)
        if not rederived_path:
            rederived_path.append(canon_from)
        elif rederived_path[-1] != canon_from:
            return False, {
                "reason": "chain break",
                "expected_from": rederived_path[-1],
                "got_from": canon_from,
                "at_hop_sha": sha,
                "at_hop_idx": idx,
            }
        rederived_path.append(canon_to)

    if rederived_path[0] != edge["from_key"]:
        return False, {"reason": "from_key mismatch", "expected": edge["from_key"], "got": rederived_path[0]}
    if rederived_path[-1] != edge["to_key"]:
        return False, {"reason": "to_key mismatch", "expected": edge["to_key"], "got": rederived_path[-1]}
    if len(derivation) != edge.get("depth", len(derivation)):
        return False, {"reason": "depth mismatch", "expected": edge.get("depth"), "got": len(derivation)}

    return True, {"path": rederived_path}


def measure_moat(copy_dir, canon_graph, n_samples=30, seed=59075):
    """Samples up to n_samples canonical inferred edges (seeded, without
    replacement) and re-derives each one from a FRESH LiveStore mount
    over the SAME copy_dir -- a "stranger" that never touches
    canon_graph's in-memory state, only the cited raw records + the
    pinned canonical_key function. Resolves the stranger's spaCy pipeline
    from canon_graph.canon_env_pin (env_pin match enforced: if the
    recorded pin claims model_available, the stranger loads the SAME
    default pipeline canon.py's module cache would give any caller in
    this process/environment; if not, the stranger uses nlp=None, the
    deterministic fallback -- exactly what canon_query's own resolution
    does, so "enforced" here means the stranger's resolution follows the
    identical rule the graph used, not a looser one)."""
    all_edges = canon_graph.canon_inferred_edges()
    rng = random.Random(seed)
    n_samples = min(n_samples, len(all_edges))
    sampled = rng.sample(all_edges, n_samples) if n_samples else []

    pin = canon_graph.canon_env_pin
    stranger_nlp = canon_mod._get_nlp() if (pin and pin.get("model_available")) else None

    stranger_store = LiveStore(copy_dir)
    rows = []
    for edge in sampled:
        ok, detail = rederive_canon_edge(stranger_store, edge, stranger_nlp)
        rows.append({
            "from_key": edge["from_key"],
            "to_key": edge["to_key"],
            "depth": edge.get("depth"),
            "ok": ok,
            "detail": detail if not ok else None,
        })

    n_ok = sum(1 for r in rows if r["ok"])
    return {
        "n_sampled": len(rows),
        "n_rederived_ok": n_ok,
        "all_rederived_ok": bool(rows) and n_ok == len(rows),
        "env_pin_used": pin,
        "rows": rows,
    }


def check_no_mutation(copy_dir, expected_shas):
    """After every canon operation: store.verify() True AND the copy's
    segment list is byte-identical (same shas, same order) to what the
    source had before any copy operation ran."""
    store = LiveStore(copy_dir)
    verify_ok = store.verify()
    current_shas = store.segments()
    shas_unchanged = current_shas == expected_shas
    return {
        "store_verify_true": verify_ok,
        "segment_shas_unchanged": shas_unchanged,
        "n_segments": len(current_shas),
    }


# ─────────────────────────────────────────────────────────────────────────
#  (b) Read side: consult_run.py --canon as a subprocess (canon-organ's
#  own file, never imported here).
# ─────────────────────────────────────────────────────────────────────────
def run_consult_subprocess(copy_dir, source="wt103", n_words=40000, max_gaps=40, seed=60, out_path=None, extra_args=None):
    if out_path is None:
        out_path = tempfile.mktemp(prefix="p75_consult_", suffix=".json")
    cmd = [
        sys.executable, CONSULT_RUN,
        "--store", copy_dir,
        "--max-gaps", str(max_gaps),
        "--seed", str(seed),
        "--canon",
        "--out", out_path,
    ]
    if source is not None:
        cmd += ["--source", source, "--n-words", str(n_words)]
    if extra_args:
        cmd += extra_args
    print("[p75] running: {}".format(" ".join(cmd)), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    print("[p75] consult_run exit={} elapsed={:.1f}s".format(proc.returncode, elapsed), flush=True)
    if proc.returncode != 0:
        print("STDOUT TAIL:", proc.stdout[-2000:], flush=True)
        print("STDERR TAIL:", proc.stderr[-2000:], flush=True)
        return {"exit_code": proc.returncode, "elapsed_seconds": elapsed, "error": True}, None

    with open(out_path, "r", encoding="utf-8") as f:
        consult_result = json.load(f)

    return {
        "exit_code": proc.returncode,
        "elapsed_seconds": elapsed,
        "error": False,
        "coverage": consult_result.get("coverage"),
        "n_spikes": consult_result.get("n_spikes"),
        "n_consults": consult_result.get("n_consults"),
        "mean_delta_real": consult_result.get("mean_delta_real"),
        "mean_delta_random": consult_result.get("mean_delta_random"),
        "canon_env_pin": consult_result.get("canon_env_pin"),
        "out_path": out_path,
    }, consult_result


# ─────────────────────────────────────────────────────────────────────────
#  Smoke: build a small LOCAL store via builder_run.py's own smoke recipe
#  (offline, no network, no real P72/P73/P74 artifact).
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
    corpus_path = os.path.join(parent_dir, "p75_smoke_corpus.txt")
    generate_smoke_corpus(corpus_path, seed=seed)
    vocab, stoi, unk, mask, val_ids = po.get_vocab()
    V = len(vocab)
    torch.manual_seed(seed)
    organism = po.Organism("p75-smoke-builder", V, mask, seed=seed)
    stream = TextFileStream(corpus_path, stoi, unk)
    feeder = po.ChunkFeeder(stream, po.BATCH, po.CHUNK)
    window_iter = stream_windows(organism, stream, feeder, window_tokens=32)

    status_path = os.path.join(parent_dir, "p75_smoke_status.json")
    metrics_path = os.path.join(parent_dir, "p75_smoke_metrics.jsonl")
    graph, metrics = run_builder(
        store_dir, status_path, metrics_path,
        window_iter, fake_extractor, windows_per_segment=3,
        max_windows=max_windows, print_every=1000, stream=stream,
    )
    return graph, metrics


# ─────────────────────────────────────────────────────────────────────────
#  Main orchestration
# ─────────────────────────────────────────────────────────────────────────
def run_p75(source_dir, work_root=None, n_moat_samples=30, moat_seed=59075,
            consult_source="wt103", consult_n_words=40000, consult_max_gaps=40,
            consult_seed=60, run_read_side=True):
    own_work_root = work_root is None
    if work_root is None:
        work_root = tempfile.mkdtemp(prefix="livecausal-p75-")
    copy_dir = os.path.join(work_root, "copy")

    try:
        expected_shas = source_segment_shas(source_dir)
        clone_store(source_dir, copy_dir)

        write_side, raw_graph, canon_graph = measure_write_side(copy_dir)

        cost = measure_canon_cost(copy_dir)
        # measure_canon_cost mounted two fresh graphs; canon_graph above
        # (from measure_write_side) is now stale relative to whichever
        # cache state cost-measurement left behind -- re-mount once more
        # for the moat sampling so it reads the SAME state cost's warm
        # mount just validated.
        canon_graph_for_moat = LiveGraph(copy_dir, canon=True)

        moat = measure_moat(copy_dir, canon_graph_for_moat, n_samples=n_moat_samples, seed=moat_seed)
        no_mutation = check_no_mutation(copy_dir, expected_shas)

        read_side = None
        if run_read_side:
            read_side, _consult_raw = run_consult_subprocess(
                copy_dir, source=consult_source, n_words=consult_n_words,
                max_gaps=consult_max_gaps, seed=consult_seed,
            )
            # consult_run.py may have grown use.ledger inside the copy --
            # that's expected (use.ledger is the one file allowed to grow,
            # per the build brief: "NICHT den Store mutieren außer
            # use.ledger"). Re-check no-mutation on segments specifically
            # (already segment-scoped above) plus re-verify after the
            # consult subprocess ran, so a consult-side bug that touched a
            # segment would still be caught.
            no_mutation_after_read = check_no_mutation(copy_dir, expected_shas)
        else:
            no_mutation_after_read = None

        clause_a = {
            "n_inferred_raw": write_side["n_inferred_edges_raw"],
            "n_inferred_canon": write_side["n_inferred_edges_canon"],
            "lift_factor": write_side["lift_factor_canon_over_raw"],
            "compression_ratio": write_side["raw_to_canon_compression_ratio"],
        }

        clause_b1 = None
        clause_b2 = None
        if read_side and not read_side.get("error"):
            clause_b1 = {"coverage": read_side.get("coverage")}
            clause_b2 = {
                "mean_delta_real": read_side.get("mean_delta_real"),
                "mean_delta_random": read_side.get("mean_delta_random"),
                "real_beats_random": (
                    read_side.get("mean_delta_real") is not None
                    and read_side.get("mean_delta_random") is not None
                    and read_side["mean_delta_real"] > read_side["mean_delta_random"]
                ),
            }

        clause_c = cost
        clause_d = {
            "n_sampled": moat["n_sampled"],
            "n_rederived_ok": moat["n_rederived_ok"],
            "all_rederived_ok": moat["all_rederived_ok"],
            "store_verify_true": no_mutation["store_verify_true"],
            "segment_shas_unchanged": no_mutation["segment_shas_unchanged"],
        }

        return {
            "source_dir": source_dir,
            "copy_dir": copy_dir,
            "n_source_segments": len(expected_shas),
            "write_side": write_side,
            "cost": cost,
            "moat": moat,
            "no_mutation_before_read": no_mutation,
            "no_mutation_after_read": no_mutation_after_read,
            "read_side": read_side,
            "clause_a_write_side": clause_a,
            "clause_b1_coverage": clause_b1,
            "clause_b2_real_beats_random": clause_b2,
            "clause_c_cost": clause_c,
            "clause_d_moat": clause_d,
            "env_pin": write_side["canon_env_pin"],
        }
    finally:
        if own_work_root:
            shutil.rmtree(work_root, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="P75 canonicalization measurement orchestrator")
    ap.add_argument("--source", default=None, help="source LiveStore directory (COPIED, never mutated)")
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--n-moat-samples", type=int, default=30)
    ap.add_argument("--moat-seed", type=int, default=59075)
    ap.add_argument("--consult-source", default="wt103", choices=("c4", "wt103"))
    ap.add_argument("--consult-n-words", type=int, default=40000)
    ap.add_argument("--consult-max-gaps", type=int, default=40)
    ap.add_argument("--consult-seed", type=int, default=60)
    ap.add_argument("--no-read-side", action="store_true", help="skip the consult_run.py subprocess (write-side + cost + moat only)")
    ap.add_argument("--smoke", action="store_true", help="build a small LOCAL source store, run every clause offline")
    ap.add_argument("--smoke-seed", type=int, default=42)
    ap.add_argument("--smoke-max-windows", type=int, default=60)
    args = ap.parse_args()

    work_root = tempfile.mkdtemp(prefix="livecausal-p75-")
    source = args.source
    run_read_side = not args.no_read_side
    try:
        if args.smoke:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            smoke_source_dir = os.path.join(work_root, "smoke_source")
            print("[p75] building smoke source store...", flush=True)
            build_smoke_source_store(smoke_source_dir, seed=args.smoke_seed, max_windows=args.smoke_max_windows)
            source = smoke_source_dir
            run_read_side = False  # smoke has no network; --source wt103 for consult_run needs it

        if not source:
            raise SystemExit("--source is required (or pass --smoke)")

        print("[p75] source: {}".format(source), flush=True)
        result = run_p75(
            source, work_root=work_root,
            n_moat_samples=args.n_moat_samples, moat_seed=args.moat_seed,
            consult_source=args.consult_source, consult_n_words=args.consult_n_words,
            consult_max_gaps=args.consult_max_gaps, consult_seed=args.consult_seed,
            run_read_side=run_read_side,
        )

        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True, default=str)
            f.write("\n")

        print("=" * 74)
        print("P75 -- source: {}".format(source))
        print("(a) write side: raw={} canon={} lift={}".format(
            result["clause_a_write_side"]["n_inferred_raw"],
            result["clause_a_write_side"]["n_inferred_canon"],
            result["clause_a_write_side"]["lift_factor"],
        ))
        if result["clause_b1_coverage"]:
            print("(b1) coverage: {}".format(result["clause_b1_coverage"]["coverage"]))
        if result["clause_b2_real_beats_random"]:
            print("(b2) real_beats_random: {}  (real={} random={})".format(
                result["clause_b2_real_beats_random"]["real_beats_random"],
                result["clause_b2_real_beats_random"]["mean_delta_real"],
                result["clause_b2_real_beats_random"]["mean_delta_random"],
            ))
        print("(c) cold={:.2f}s warm={:.2f}s".format(
            result["clause_c_cost"]["cold_mount_seconds"], result["clause_c_cost"]["warm_mount_seconds"]))
        print("(d) moat: {}/{} rederived, no_mutation store_verify={} shas_unchanged={}".format(
            result["clause_d_moat"]["n_rederived_ok"], result["clause_d_moat"]["n_sampled"],
            result["clause_d_moat"]["store_verify_true"], result["clause_d_moat"]["segment_shas_unchanged"],
        ))
        print("wrote {}".format(args.out))
    finally:
        shutil.rmtree(work_root, ignore_errors=True)


if __name__ == "__main__":
    main()
