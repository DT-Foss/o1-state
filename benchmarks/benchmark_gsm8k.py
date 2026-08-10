#!/usr/bin/env python3
"""
GSM8K Benchmark — Grade School Math
=====================================
1319 multi-step arithmetic word problems.
Tests: number extraction, operation chaining, multi-step reasoning.
GPT-4 achieves ~92%.
"""

import sys
import os
import json
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.word_problem import WordProblemSolver


def extract_answer(answer_str: str) -> str:
    """Extract the final numeric answer from GSM8K answer format."""
    # GSM8K format: "reasoning text\n#### 42"
    if '####' in answer_str:
        return answer_str.split('####')[-1].strip()
    return answer_str.strip()


def normalize_number(s: str) -> float:
    """Normalize a number string to float for comparison."""
    s = s.strip().replace(',', '').replace('$', '').replace('%', '')
    try:
        return float(s)
    except ValueError:
        return float('nan')


def main():
    data_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'external_benchmarks', 'gsm8k_test.jsonl'
    )

    if not os.path.exists(data_path):
        print("ERROR: gsm8k_test.jsonl not found. Download first.")
        return

    # Load data
    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    solver = WordProblemSolver()

    print("=" * 65)
    print("  GSM8K BENCHMARK — Grade School Math (1319 problems)")
    print("=" * 65)

    t0 = time.time()
    correct = 0
    attempted = 0
    no_answer = 0
    total = len(problems)

    errors = []

    for i, item in enumerate(problems):
        question = item['question']
        expected_str = extract_answer(item['answer'])
        expected = normalize_number(expected_str)

        result = solver.solve(question)

        if result is None or result.get('answer') is None:
            no_answer += 1
            if len(errors) < 10:
                errors.append(f"  NO ANSWER: {question[:80]}... (expected: {expected_str})")
            continue

        attempted += 1
        got = result['answer']
        got_float = float(got) if isinstance(got, (int, float)) else normalize_number(str(got))

        if abs(got_float - expected) < 0.01:
            correct += 1
        else:
            if len(errors) < 10:
                errors.append(
                    f"  WRONG: Got {got}, Expected {expected_str}\n"
                    f"         Q: {question[:80]}..."
                )

    elapsed = time.time() - t0
    acc = correct / total * 100
    attempted_acc = correct / max(attempted, 1) * 100

    print(f"\n  Total: {total}")
    print(f"  Attempted: {attempted} ({attempted/total*100:.0f}%)")
    print(f"  No answer: {no_answer}")
    print(f"  Correct: {correct}")
    print(f"  Accuracy (of total): {acc:.1f}%")
    print(f"  Accuracy (of attempted): {attempted_acc:.1f}%")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.1f}ms per problem)")

    print(f"\n  Sample errors:")
    for e in errors[:5]:
        print(e)

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'GSM8K accuracy':<35} {acc:>9.1f}% {'~92%':>10}")
    print(f"  {'Coverage (attempted)':<35} {attempted/total*100:>8.0f}% {'100%':>10}")
    print(f"  {'Latency per problem':<35} {elapsed/total*1000:>8.1f}ms {'~500ms':>10}")

    print(f"\n{'=' * 65}")
    print(f"  BASELINE RESULTS: {correct}/{total} ({acc:.1f}%)")
    print(f"{'=' * 65}")


if __name__ == '__main__':
    main()
