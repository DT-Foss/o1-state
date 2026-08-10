#!/usr/bin/env python3
"""
PIQA Benchmark — Physical Intuition QA
========================================
Tests physical commonsense reasoning on 1838 examples.
GPT-4 achieves ~90%+. Random baseline: 50% (2 choices).
"""

import sys
import os
import time
import pickle
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.piqa_solver import PIQASolver, evaluate_piqa


def main():
    parser = argparse.ArgumentParser(description='PIQA Benchmark')
    parser.add_argument('--flm', type=str, default=None, help='Path to FossLanguageModel pickle file')
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'data', 'external_benchmarks', 'piqa_valid.jsonl')
    labels_path = os.path.join(base, 'data', 'external_benchmarks', 'piqa_valid_labels.lst')

    if not os.path.exists(data_path):
        print(f"PIQA data not found at {data_path}")
        return
    if not os.path.exists(labels_path):
        print(f"PIQA labels not found at {labels_path}")
        return

    glove_path = os.path.join(base, 'data', 'glove.6B.100d.txt')
    conceptnet_path = os.path.join(base, 'data', 'conceptnet_index.pkl')

    flm = None
    if args.flm:
        if not os.path.exists(args.flm):
            print(f"FLM pickle not found at {args.flm}")
            return
        with open(args.flm, 'rb') as f:
            flm = pickle.load(f)

    solver = PIQASolver(glove_path=glove_path, conceptnet_path=conceptnet_path, flm=flm)

    print("=" * 65)
    print("  PIQA — Physical Intuition QA")
    print("=" * 65)

    t0 = time.time()
    correct, total, errors = evaluate_piqa(solver, data_path, labels_path)
    elapsed = time.time() - t0

    pct = correct / total * 100 if total > 0 else 0

    print(f"\n  Result: {correct}/{total} ({pct:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")

    if errors:
        print(f"\n  Sample errors ({min(len(errors), 10)}):")
        for e in errors[:10]:
            print(f"    Goal: {e['goal']}")
            print(f"      Pred: {e['pred']}")
            print(f"      True: {e['correct']}")
            print()

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'PIQA accuracy':<35} {pct:>9.1f}% {'~90%':>10}")
    print(f"  {'Random baseline (2 choices)':<35} {'50%':>10} {'':>10}")
    print(f"  {'Parameters':<35} {'0':>10} {'~200B':>10}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
