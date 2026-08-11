#!/usr/bin/env python3
"""
bAbI Full Suite Benchmark — 20 Tasks × 1000 Questions
======================================================
The real Facebook AI Research bAbI benchmark.
Tests 20 types of reasoning on 20,000 questions.
GPT-4 achieves ~95%. FOSS-KI target: 100%.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.babi_engine import BabiEngine, evaluate_task

TASK_NAMES = {
    1: 'Single Supporting Fact',
    2: 'Two Supporting Facts',
    3: 'Three Supporting Facts',
    4: 'Two-Arg Relations',
    5: 'Three-Arg Relations',
    6: 'Yes/No Questions',
    7: 'Counting',
    8: 'Lists/Sets',
    9: 'Simple Negation',
    10: 'Indefinite Knowledge',
    11: 'Basic Coreference',
    12: 'Conjunction',
    13: 'Compound Coreference',
    14: 'Time Reasoning',
    15: 'Basic Deduction',
    16: 'Basic Induction',
    17: 'Positional Reasoning',
    18: 'Size Reasoning',
    19: 'Path Finding',
    20: 'Agent Motivations',
}


def main():
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'external_benchmarks', 'tasks_1-20_v1-2', 'en-10k'
    )

    if not os.path.exists(data_dir):
        print("ERROR: bAbI data not found. Download first.")
        return

    engine = BabiEngine()

    print("=" * 65)
    print("  bAbI FULL SUITE — 20 Tasks × 1000 Questions")
    print("=" * 65)

    t0 = time.time()
    total_correct = 0
    total_questions = 0
    task_results = []

    print(f"\n  {'Task':<42} {'Score':>10} {'Acc':>7}")
    print(f"  {'─' * 61}")

    for task_id in range(1, 21):
        files = [f for f in os.listdir(data_dir)
                 if f.startswith(f'qa{task_id}_') and f.endswith('_test.txt')]
        if not files:
            print(f"  ? Task {task_id}: file not found")
            continue

        filepath = os.path.join(data_dir, files[0])
        correct, total, errors = evaluate_task(engine, filepath)
        total_correct += correct
        total_questions += total
        pct = correct / total * 100 if total > 0 else 0
        task_results.append((task_id, correct, total, pct))

        name = TASK_NAMES.get(task_id, '?')
        status = '✓' if pct >= 95 else '△' if pct >= 80 else '✗'
        print(f"  {status} Task {task_id:2d}: {name:<32} {correct:4d}/{total:4d} {pct:6.1f}%")

        if errors and pct < 95:
            for e in errors[:2]:
                print(f"         {e}")

    elapsed = time.time() - t0
    overall = total_correct / total_questions * 100 if total_questions > 0 else 0
    perfect = sum(1 for _, _, _, p in task_results if p >= 99.9)
    above95 = sum(1 for _, _, _, p in task_results if p >= 95)

    print(f"\n{'=' * 65}")
    print(f"  OVERALL: {total_correct}/{total_questions} ({overall:.1f}%)")
    print(f"  Perfect tasks (≥99.9%): {perfect}/20")
    print(f"  Tasks ≥95%: {above95}/20")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_questions*1000:.2f}ms per question)")

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'bAbI accuracy':<35} {overall:>9.1f}% {'~95%':>10}")
    print(f"  {'Perfect tasks':<35} {perfect:>7d}/20 {'~15/20':>10}")
    print(f"  {'Latency per question':<35} {elapsed/total_questions*1000:>7.2f}ms {'~500ms':>10}")
    print(f"  {'Parameters':<35} {'0':>10} {'~200B':>10}")
    print(f"{'=' * 65}")

    print(f"\nResults: {total_correct}/{total_questions} passed ({overall:.1f}%)")


if __name__ == '__main__':
    main()
