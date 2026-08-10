"""P72 measurement run (analysis/PREDICTIONS.md, "the builder builds, end
to end"). Orchestrates two identical builder_run.py runs over WT-103 at
P70 cadence, a manifest-sha reproducibility comparison, and a direction-3
verifier pass over the first run's store. Harness only -- does not write
to analysis/PREDICTIONS.md, does not commit.

Does NOT import or modify src/livecausal/builder_run.py or
src/curator_yield_run.py -- drives builder_run.py as a subprocess (its own
CLI contract), exactly as a human operator would.

Run (core, venv active, network, beside P69):
  source /root/o1lab-venv/bin/activate
  OMP_NUM_THREADS=1 nice -n 12 python3 src/livecausal/p72_run.py
"""
import json
import os
import subprocess
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(REPO_ROOT, "results")
BUILDER_RUN = os.path.join(REPO_ROOT, "src", "livecausal", "builder_run.py")

# P70 cadence, exactly (curator_yield_run.iter_windows's own defaults,
# repeated explicitly here rather than relied on implicitly, since
# builder_run.py's CLI defaults are the smaller SMOKE cadence, not P70's).
P70_CADENCE = {
    "d-model": 128,
    "batch": 8,
    "chunk-size": 64,
    "q": 0.75,
    "window": 500,
    "min-window": 100,
    "ignition-chunks": 100,
}
CHUNKS = 3000
SEED = 42
WINDOW_TOKENS = 128  # matches P70's own window_tokens default


def run_builder_subprocess(store_dir, out_prefix, seed=SEED):
    cmd = [
        sys.executable, BUILDER_RUN,
        "--source", "wt103",
        "--chunks", str(CHUNKS),
        "--seed", str(seed),
        "--window-tokens", str(WINDOW_TOKENS),
        "--store-dir", store_dir,
        "--out-prefix", out_prefix,
    ]
    for flag, val in P70_CADENCE.items():
        cmd += ["--{}".format(flag), str(val)]
    print("[p72] running: {}".format(" ".join(cmd)), flush=True)
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    elapsed = time.time() - t0
    print("[p72] exit={} elapsed={:.1f}s".format(proc.returncode, elapsed), flush=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode != 0:
        print("STDERR TAIL:", proc.stderr[-4000:], flush=True)
    return proc, elapsed


def read_status(out_prefix):
    path = out_prefix + "_status.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def manifest_shas(store_dir):
    manifest_path = os.path.join(store_dir, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["segments"]


def run_verifier(store_dir, n_samples=30, seed=60):
    sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
    from stranger_verify_run import score, verify_direction3  # noqa: E402

    checks = verify_direction3(store_dir, n_samples, seed)
    scoring = score(checks, n_target=n_samples)
    n_base = sum(1 for c in checks if c["class"] == "base")
    n_inferred = sum(1 for c in checks if c["class"] == "inferred")
    return {
        "n_sampled": len(checks),
        "n_base": n_base,
        "n_inferred": n_inferred,
        "scoring": scoring,
        "checks": checks,
    }


def clause_a(status1):
    n_base = status1["n_base_edges"]
    n_inferred = status1["n_inferred_edges"]
    return {
        "n_base_edges": n_base,
        "n_inferred_edges": n_inferred,
        "p72a_base_pass": bool(n_base >= 1000),
        "p72a_inferred_pass": bool(n_inferred >= 200),
        "p72a_pass": bool(n_base >= 1000 and n_inferred >= 200),
    }


def clause_b(status1, status2, manifest1, manifest2, rebuilt_note):
    manifests_equal = manifest1 == manifest2
    return {
        "run1_manifest_segment_count": len(manifest1),
        "run2_manifest_segment_count": len(manifest2),
        "manifests_bit_equal": bool(manifests_equal),
        "rebuild_note": rebuilt_note,
        "p72b_pass": bool(manifests_equal),
    }


def clause_c(verify_result):
    scoring = verify_result["scoring"]
    return {
        "n_sampled": verify_result["n_sampled"],
        "n_base_sampled": verify_result["n_base"],
        "n_inferred_sampled": verify_result["n_inferred"],
        "p60a_fraction": scoring["p60a_fraction"],
        "p60b_fraction": scoring["p60b_fraction"],
        "p72c_verified_pass": scoring["p60a_verified_all"],
        "p72c_consensus_pass": scoring["p60b_consensus_all"],
        "p72c_pass": bool(scoring["p60a_verified_all"] and scoring["p60b_consensus_all"]),
    }


def clause_d(status1):
    n_gated = status1["n_windows_gated"]
    n_total = status1["n_windows_total"]
    frac = (n_gated / n_total) if n_total else None
    in_band = frac is not None and 0.20 <= frac <= 0.30
    return {
        "n_windows_gated": n_gated,
        "n_windows_total": n_total,
        "gated_fraction": round(frac, 4) if frac is not None else None,
        "band": [0.20, 0.30],
        "p72d_pass": bool(in_band),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    store1 = os.path.join(RESULTS_DIR, "p72_store_run1")
    store2 = os.path.join(RESULTS_DIR, "p72_store_run2")
    out1 = os.path.join(RESULTS_DIR, "p72_builder_run1")
    out2 = os.path.join(RESULTS_DIR, "p72_builder_run2")

    started = time.time()

    print("[p72] === RUN 1 ===", flush=True)
    proc1, elapsed1 = run_builder_subprocess(store1, out1, seed=SEED)
    status1 = read_status(out1)

    print("[p72] === RUN 2 ===", flush=True)
    proc2, elapsed2 = run_builder_subprocess(store2, out2, seed=SEED)
    status2 = read_status(out2)

    manifest1 = manifest_shas(store1)
    manifest2 = manifest_shas(store2)

    # was_rebuilt_on_mount is not directly recoverable from status.json (it
    # is a LiveGraph in-process attribute, not persisted) -- report what IS
    # recoverable: the structural guarantee that run_builder() constructs
    # exactly one LiveGraph per process (grep-verified against
    # builder_run.py, not re-derived here to respect the "don't touch
    # builder_run.py" constraint) means _rebuild_from_store fires at most
    # once, at the initial mount of an empty store dir -- there is no
    # second mount anywhere in the loop to trigger another one.
    rebuilt_note = (
        "run_builder() constructs exactly one LiveGraph per process "
        "(single `graph = LiveGraph(store_dir)` call site in "
        "builder_run.py, verified by inspection, not re-derived at "
        "runtime to avoid touching that file) and never re-mounts; "
        "_rebuild_from_store therefore fires at most once, at the "
        "initial mount of an empty store directory, and never again "
        "during the run -- 'zero full rebuilds' is structural, not "
        "just counted."
    )

    print("[p72] === VERIFIER (run1 store) ===", flush=True)
    verify_result = run_verifier(store1, n_samples=30, seed=60)

    a = clause_a(status1)
    b = clause_b(status1, status2, manifest1, manifest2, rebuilt_note)
    c = clause_c(verify_result)
    d = clause_d(status1)

    finished = time.time()

    out = {
        "prediction": "P72",
        "spec_ref": "analysis/PREDICTIONS.md P72",
        "host": os.uname().nodename if hasattr(os, "uname") else None,
        "cadence": P70_CADENCE,
        "chunks": CHUNKS,
        "seed": SEED,
        "window_tokens": WINDOW_TOKENS,
        "started_unix": started,
        "finished_unix": finished,
        "elapsed_seconds": finished - started,
        "run1": {
            "store_dir": store1,
            "out_prefix": out1,
            "elapsed_seconds": elapsed1,
            "exit_code": proc1.returncode,
            "status": status1,
            "env_pin": status1.get("env_pin"),
        },
        "run2": {
            "store_dir": store2,
            "out_prefix": out2,
            "elapsed_seconds": elapsed2,
            "exit_code": proc2.returncode,
            "status": status2,
            "env_pin": status2.get("env_pin"),
        },
        "clause_a_graph_grows": a,
        "clause_b_mechanics_hold": b,
        "clause_c_audit_passes": c,
        "clause_d_signal_rides_along": d,
        "p72a_pass": a["p72a_pass"],
        "p72b_pass": b["p72b_pass"],
        "p72c_pass": c["p72c_pass"],
        "p72d_pass": d["p72d_pass"],
    }

    run1_path = os.path.join(RESULTS_DIR, "p72_run1.json")
    with open(run1_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, sort_keys=True)
        f.write("\n")

    compare_path = os.path.join(RESULTS_DIR, "p72_compare.json")
    with open(compare_path, "w", encoding="utf-8") as f:
        json.dump(b, f, indent=2, sort_keys=True)
        f.write("\n")

    verify_path = os.path.join(RESULTS_DIR, "p72_verify.json")
    with open(verify_path, "w", encoding="utf-8") as f:
        json.dump(verify_result, f, indent=2, sort_keys=True)
        f.write("\n")

    print("=" * 74)
    print("wrote {}".format(run1_path))
    print("wrote {}".format(compare_path))
    print("wrote {}".format(verify_path))
    print("p72a_pass:", a["p72a_pass"], "| base_edges", a["n_base_edges"], "inferred_edges", a["n_inferred_edges"])
    print("p72b_pass:", b["p72b_pass"], "| manifests_bit_equal", b["manifests_bit_equal"])
    print("p72c_pass:", c["p72c_pass"], "| verified", c["p60a_fraction"], "consensus", c["p60b_fraction"])
    print("p72d_pass:", d["p72d_pass"], "| gated_fraction", d["gated_fraction"])


if __name__ == "__main__":
    main()
