#!/usr/bin/env python3
"""
Unseen Benchmarks — CommonsenseQA + OpenBookQA
================================================
Benchmarks the system has NEVER been tuned on.
No hardcoded answers, no benchmark-specific modules.
Tests TRUE generalization.
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'external_benchmarks')
RESULTS_DIR = os.path.join(BASE_DIR, 'benchmarks', 'results')


def run_commonsenseqa(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run CommonsenseQA (1221 dev questions, 5-way MCQ)."""
    data_path = os.path.join(DATA_DIR, 'commonsenseqa_dev.jsonl')
    if not os.path.exists(data_path):
        return {'benchmark': 'commonsenseqa', 'skipped': 'Data not found'}

    from core.arc_solver import ARCSolver
    solver = ARCSolver()

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if limit:
        problems = problems[:limit]

    results = []
    for i, item in enumerate(problems):
        q_data = item['question']
        question = q_data.get('stem', q_data.get('question_concept', ''))
        choices = [c['text'] for c in q_data['choices']]
        labels = [c['label'] for c in q_data['choices']]
        answer_key = item['answerKey']

        t0 = time.time()
        prediction = solver.solve(question, choices, labels)
        elapsed = time.time() - t0

        correct = prediction == answer_key
        correct_text = choices[labels.index(answer_key)] if answer_key in labels else '?'
        pred_text = choices[labels.index(prediction)] if prediction in labels else '?'

        results.append({
            'id': f"csqa_{i}",
            'question': question[:200],
            'expected': f"{answer_key}: {correct_text}",
            'got': f"{prediction}: {pred_text}",
            'correct': correct,
            'time_ms': round(elapsed * 1000, 1),
        })

    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        'benchmark': 'commonsenseqa',
        'total': total,
        'correct': correct_count,
        'accuracy': round(accuracy, 4),
        'avg_ms': round(sum(r['time_ms'] for r in results) / total, 1) if total else 0,
        'results': results,
    }


def run_openbookqa(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run OpenBookQA (500 test questions, 4-way MCQ)."""
    data_path = os.path.join(DATA_DIR, 'OpenBookQA-V1-Sep2018', 'Data', 'Main', 'test.jsonl')
    if not os.path.exists(data_path):
        return {'benchmark': 'openbookqa', 'skipped': 'Data not found'}

    from core.arc_solver import ARCSolver
    solver = ARCSolver()

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if limit:
        problems = problems[:limit]

    results = []
    for i, item in enumerate(problems):
        q_data = item['question']
        question = q_data['stem']
        choices = [c['text'] for c in q_data['choices']]
        labels = [c['label'] for c in q_data['choices']]
        answer_key = item['answerKey']

        t0 = time.time()
        prediction = solver.solve(question, choices, labels)
        elapsed = time.time() - t0

        correct = prediction == answer_key
        correct_text = choices[labels.index(answer_key)] if answer_key in labels else '?'
        pred_text = choices[labels.index(prediction)] if prediction in labels else '?'

        results.append({
            'id': f"obqa_{i}",
            'question': question[:200],
            'expected': f"{answer_key}: {correct_text}",
            'got': f"{prediction}: {pred_text}",
            'correct': correct,
            'time_ms': round(elapsed * 1000, 1),
        })

    total = len(results)
    correct_count = sum(1 for r in results if r['correct'])
    accuracy = correct_count / total if total > 0 else 0.0

    return {
        'benchmark': 'openbookqa',
        'total': total,
        'correct': correct_count,
        'accuracy': round(accuracy, 4),
        'avg_ms': round(sum(r['time_ms'] for r in results) / total, 1) if total else 0,
        'results': results,
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 70)
    print("  UNSEEN BENCHMARKS — True Generalization Test")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("  No hardcoded answers. No benchmark-specific modules.")
    print("=" * 70)

    limit = 200  # 200 samples for speed

    all_results = {}

    # CommonsenseQA
    print(f"\n{'─' * 70}")
    print("  CommonsenseQA (5-way MCQ, random baseline: 20%)")
    print(f"{'─' * 70}")
    csqa = run_commonsenseqa(limit)
    all_results['commonsenseqa'] = csqa
    if csqa.get('skipped'):
        print(f"  SKIPPED: {csqa['skipped']}")
    else:
        print(f"  Result: {csqa['correct']}/{csqa['total']} ({csqa['accuracy']:.1%})")
        print(f"  Avg latency: {csqa['avg_ms']:.1f}ms")
        # Show sample errors
        errors = [r for r in csqa['results'] if not r['correct']]
        for e in errors[:3]:
            print(f"    Q: {e['question'][:80]}")
            print(f"      Expected: {e['expected']}, Got: {e['got']}")

    # OpenBookQA
    print(f"\n{'─' * 70}")
    print("  OpenBookQA (4-way MCQ, random baseline: 25%)")
    print(f"{'─' * 70}")
    obqa = run_openbookqa(limit)
    all_results['openbookqa'] = obqa
    if obqa.get('skipped'):
        print(f"  SKIPPED: {obqa['skipped']}")
    else:
        print(f"  Result: {obqa['correct']}/{obqa['total']} ({obqa['accuracy']:.1%})")
        print(f"  Avg latency: {obqa['avg_ms']:.1f}ms")
        errors = [r for r in obqa['results'] if not r['correct']]
        for e in errors[:3]:
            print(f"    Q: {e['question'][:80]}")
            print(f"      Expected: {e['expected']}, Got: {e['got']}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  UNSEEN BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Benchmark':<20} {'Score':>12} {'Accuracy':>10} {'Random':>10} {'Lift':>8}")
    print(f"  {'─' * 62}")
    for name, data in all_results.items():
        if data.get('skipped'):
            continue
        random = 0.20 if name == 'commonsenseqa' else 0.25
        lift = data['accuracy'] - random
        print(f"  {name:<20} {data['correct']:>5}/{data['total']:<5} "
              f"{data['accuracy']:>9.1%} {random:>9.0%} {lift:>+7.1%}")

    print(f"\n  Lift = accuracy above random guessing")
    print(f"  These scores represent TRUE system generalization")
    print(f"{'=' * 70}")

    # Save
    path = os.path.join(RESULTS_DIR, f'unseen_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"  Saved: {path}")


if __name__ == '__main__':
    main()
