#!/usr/bin/env python3
"""
ARC-Challenge Benchmark
========================
Tests science knowledge reasoning on 1172 multiple-choice questions.
GPT-4 achieves ~85-90%. Random baseline: ~25% (4 choices).
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.arc_solver import ARCSolver


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base, 'data', 'external_benchmarks', 'arc_challenge_test.jsonl')

    if not os.path.exists(data_path):
        print(f"ARC-Challenge data not found at {data_path}")
        return

    solver = ARCSolver()

    print("=" * 65)
    print("  ARC-CHALLENGE — Science Knowledge Reasoning")
    print("=" * 65)

    correct = 0
    total = 0
    errors = []
    category_stats = {}  # track by question type

    t0 = time.time()

    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            question = item['question']
            choices = item['choices']['text']
            labels = item['choices']['label']
            answer_key = item['answerKey']

            prediction = solver.solve(question, choices, labels)
            total += 1

            if prediction == answer_key:
                correct += 1
            else:
                if len(errors) < 20:
                    correct_text = choices[labels.index(answer_key)] if answer_key in labels else '?'
                    pred_text = choices[labels.index(prediction)] if prediction in labels else '?'
                    errors.append({
                        'q': question[:80],
                        'pred': f"{prediction}: {pred_text[:40]}",
                        'correct': f"{answer_key}: {correct_text[:40]}",
                    })

    elapsed = time.time() - t0
    pct = correct / total * 100 if total > 0 else 0

    print(f"\n  Result: {correct}/{total} ({pct:.1f}%)")
    print(f"  Time: {elapsed:.1f}s")

    if errors:
        print(f"\n  Sample errors ({min(len(errors), 10)}):")
        for e in errors[:10]:
            print(f"    Q: {e['q']}")
            print(f"      Pred: {e['pred']}")
            print(f"      True: {e['correct']}")
            print()

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'ARC-Challenge accuracy':<35} {pct:>9.1f}% {'~85%':>10}")
    print(f"  {'Random baseline (4 choices)':<35} {'25%':>10} {'':>10}")
    print(f"  {'Parameters':<35} {'0':>10} {'~200B':>10}")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
