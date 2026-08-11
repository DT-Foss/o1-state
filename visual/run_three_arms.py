#!/usr/bin/env python3
"""
RUN THREE ARMS -- Lead's design-sharpening after the Phase-1 smoke result
(model L1 0.214 was WORSE than both trivial baselines: copy-last 0.023,
mean-frame 0.164).

Three arms, same seed/source/budget, LR 1e-3 across the board:
  (A) residual=True,  gate=on   -- main arm: delta-parametrized output,
      surprise-gated training (the design the Lead is betting resolves the
      collapse, per its own reasoning above FramePredictor.__init__).
  (B) residual=True,  gate=off  -- ablation: same parametrization, but every
      chunk is trained. Isolates whether the SMALL Phase-1 gradient-step
      budget (not the gate policy itself) was the binding constraint.
  (C) residual=False, gate=on   -- control: today's (Phase-1) parametrization
      at the new LR/budget, so a Phase-1-vs-Phase-2 comparison isn't
      confounded by the LR/budget change alone.

50,000 frames per arm (~2 min each on MPS per the Phase-1 measured rate of
7.1s/3000 frames). GIFs every 5000 frames. Each arm writes its own
metrics.jsonl (with the copy-last baseline logged per chunk, per
train_visual.py) under results/visual_arm_{a,b,c}_*.

This script does not import train_visual.py's main() (argparse-driven CLI,
awkward to call twice on one process) -- it runs each arm as its own
subprocess with the same flags a human would type, then reads back the
three metrics.jsonl files to print a compact final-L1 comparison table
against the run's own copy-last and mean-frame baselines.
"""

import os
import sys
import json
import time
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

VISUAL_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(VISUAL_DIR)
RESULTS_DIR = os.path.join(REPO_ROOT, "results")

FRAMES = 50_000
SEED = 42
LR = 1e-3
GIF_EVERY = 5000
GIF_WINDOW = 80
Q = 0.75

ARMS = [
    {"name": "arm_a_residual_gated", "flags": ["--residual"]},
    {"name": "arm_b_residual_nogate", "flags": ["--residual", "--no-gate"]},
    {"name": "arm_c_direct_gated", "flags": []},
]


def run_arm(arm):
    out_prefix = os.path.join(RESULTS_DIR, f"visual_{arm['name']}")
    cmd = [
        sys.executable, "-u", os.path.join(VISUAL_DIR, "train_visual.py"),
        "--source", "synthetic", "--frames", str(FRAMES), "--q", str(Q),
        "--seed", str(SEED), "--lr", str(LR),
        "--out-prefix", out_prefix,
        "--gif-every", str(GIF_EVERY), "--gif-window", str(GIF_WINDOW),
        "--device", "mps",
    ] + arm["flags"]
    log_path = f"{out_prefix}.log"
    print(f"[run_three_arms] starting {arm['name']}: {' '.join(cmd)}", flush=True)
    t0 = time.time()
    with open(log_path, "w") as lf:
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
    wall = time.time() - t0
    if proc.returncode != 0:
        print(f"[run_three_arms] {arm['name']} FAILED (exit {proc.returncode}) "
              f"after {wall:.1f}s -- see {log_path}", flush=True)
        with open(log_path) as lf:
            print(lf.read()[-3000:], flush=True)
        raise RuntimeError(f"{arm['name']} failed")
    print(f"[run_three_arms] {arm['name']} done in {wall:.1f}s", flush=True)
    return out_prefix, wall


def summarize(out_prefix, n_tail=20):
    metrics_path = f"{out_prefix}_metrics.jsonl"
    recs = [json.loads(l) for l in open(metrics_path)]
    tail = recs[-n_tail:]
    final_l1 = sum(r["l1"] for r in tail) / len(tail)
    final_copy_last = sum(r["copy_last_l1"] for r in tail) / len(tail)
    n_bwd = sum(r["gated"] for r in recs)
    gate_rate = n_bwd / len(recs)
    return {
        "n_chunks": len(recs),
        "final_l1": round(final_l1, 4),
        "final_copy_last_l1": round(final_copy_last, 4),
        "gate_rate_overall": round(gate_rate, 4),
        "l1_first": round(recs[0]["l1"], 4),
        "l1_last": round(recs[-1]["l1"], 4),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = {}
    for arm in ARMS:
        out_prefix, wall = run_arm(arm)
        results[arm["name"]] = {"out_prefix": out_prefix, "wall_s": round(wall, 1)}

    print("\n" + "=" * 78)
    print("THREE-ARM COMPARISON (50,000 frames each, synthetic, seed=42, LR=1e-3)")
    print("=" * 78)
    header = f"{'arm':28s} {'final L1':>10s} {'copy-last L1':>14s} {'gate rate':>10s} {'wall_s':>8s}"
    print(header)
    print("-" * len(header))
    for arm in ARMS:
        name = arm["name"]
        s = summarize(results[name]["out_prefix"])
        results[name].update(s)
        beats_baseline = "BEATS copy-last" if s["final_l1"] < s["final_copy_last_l1"] else "worse than copy-last"
        print(f"{name:28s} {s['final_l1']:>10.4f} {s['final_copy_last_l1']:>14.4f} "
              f"{s['gate_rate_overall']:>10.3f} {results[name]['wall_s']:>8.1f}  [{beats_baseline}]")

    summary_path = os.path.join(RESULTS_DIR, "visual_three_arms_summary.json")
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[run_three_arms] summary written: {summary_path}")


if __name__ == "__main__":
    main()
