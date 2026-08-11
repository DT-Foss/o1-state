#!/usr/bin/env python3
"""
Winograd Schema Challenge Benchmark
====================================
Tests pronoun coreference resolution requiring commonsense reasoning.
285 problems (WSC273+) and 104 SuperGLUE validation examples.
GPT-4 achieves ~90%+ on WSC. Random baseline: 50%.
"""

import sys
import os
import time
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.winograd_solver import WinogradSolver, evaluate_wsc_superglue, evaluate_wsc273


def main():
    parser = argparse.ArgumentParser(description='Winograd Schema Challenge Benchmark')
    parser.add_argument('--flm', type=str, default=None, help='Path to FossLanguageModel pickle file')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load FLM if provided
    flm = None
    if args.flm:
        with open(args.flm, 'rb') as f:
            flm = pickle.load(f)

    solver = WinogradSolver(flm=flm)

    print("=" * 65)
    print("  WINOGRAD SCHEMA CHALLENGE — Pronoun Coreference Resolution")
    print("=" * 65)

    t0 = time.time()

    # ── SuperGLUE WSC (validation set, labeled) ──
    sg_path = os.path.join(base, 'data', 'external_benchmarks',
                           'superglue_wsc', 'WSC', 'val.jsonl')
    if os.path.exists(sg_path):
        correct, total, errors = evaluate_wsc_superglue(solver, sg_path)
        pct = correct / total * 100 if total > 0 else 0
        print(f"\n  SuperGLUE WSC (val): {correct}/{total} ({pct:.1f}%)")
        if errors:
            print(f"  Sample errors:")
            for e in errors[:5]:
                print(f"    {e}")
    else:
        print(f"\n  SuperGLUE WSC: data not found at {sg_path}")

    # ── WSC273 ──
    wsc_path = os.path.join(base, 'data', 'external_benchmarks', 'wsc273.json')
    if os.path.exists(wsc_path):
        correct273, total273, errors273 = evaluate_wsc273(solver, wsc_path)
        pct273 = correct273 / total273 * 100 if total273 > 0 else 0
        print(f"\n  WSC273: {correct273}/{total273} ({pct273:.1f}%)")
        if errors273:
            print(f"  Sample errors:")
            for e in errors273[:5]:
                print(f"    {e}")
    else:
        print(f"\n  WSC273: data not found")

    elapsed = time.time() - t0

    print(f"\n  Time: {elapsed:.1f}s")
    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4':>10}")
    print(f"  {'─' * 57}")
    if os.path.exists(sg_path):
        print(f"  {'SuperGLUE WSC accuracy':<35} {pct:>9.1f}% {'~90%':>10}")
    if os.path.exists(wsc_path):
        print(f"  {'WSC273 accuracy':<35} {pct273:>9.1f}% {'~90%':>10}")
    print(f"  {'Random baseline':<35} {'50%':>10} {'':>10}")
    print(f"  {'Majority baseline (all False)':<35} {'63%':>10} {'':>10}")
    print(f"  {'Parameters':<35} {'0':>10} {'~200B':>10}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
