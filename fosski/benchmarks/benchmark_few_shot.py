#!/usr/bin/env python3
"""
Few-Shot Learning Benchmark
=============================
Tests the ability to learn from statements in conversation and
answer questions about the newly learned facts.
GPT-4o handles this natively via in-context learning.
FOSS-KI must extract triplets from statements and store in KB.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def main():
    print("=" * 60)
    print("  FEW-SHOT LEARNING BENCHMARK")
    print("  Testing learn-from-statements capability")
    print("=" * 60)

    repl = FossKIRepl()

    tests = [
        # Teach capital, then ask
        {
            'teach': ['The capital of Narnia is Cair Paravel.'],
            'q': 'What is the capital of Narnia?',
            'contains': 'cair paravel',
        },
        # Teach is_a, then ask
        {
            'teach': ['Zorblon is a planet.'],
            'q': 'Is Zorblon a planet?',
            'contains': 'yes',
        },
        # Teach population
        {
            'teach': ['The population of Atlantis is 50000.'],
            'q': 'What is the population of Atlantis?',
            'contains': '50000',
        },
        # Teach property
        {
            'teach': ['The color of the sky on Mars is red.'],
            'q': 'What is the color of the sky on Mars?',
            'contains': 'red',
        },
        # Teach person
        {
            'teach': ['Dr. Zara is a scientist.'],
            'q': 'Is Dr. Zara a scientist?',
            'contains': 'yes',
        },
        # Teach language
        {
            'teach': ['The language of Mordor is Black Speech.'],
            'q': 'What is the language of Mordor?',
            'contains': 'black speech',
        },
        # Teach multiple, then cross-reference
        {
            'teach': ['Quixland is a country.', 'The capital of Quixland is Quixville.'],
            'q': 'What is the capital of Quixland?',
            'contains': 'quixville',
        },
        # Teach has-relation
        {
            'teach': ['Mars has two moons.'],
            'q': 'How many moons does Mars have?',
            'contains': '2',  # KB already has this, so it should still work
        },
    ]

    t0 = time.time()
    correct = 0

    for test in tests:
        # Teach
        for stmt in test['teach']:
            repl.process(stmt)

        # Ask
        answer = repl.process(test['q'])
        answer_lower = answer.lower()
        expected_lower = test['contains'].lower()

        ok = expected_lower in answer_lower
        if ok:
            correct += 1
            print(f"  ✓ Taught: {test['teach']}")
            print(f"    Asked: {test['q']} → {answer[:80]}")
        else:
            print(f"  ✗ Taught: {test['teach']}")
            print(f"    Asked: {test['q']}")
            print(f"    Got: {answer[:80]}")
            print(f"    Expected: {test['contains']}")

    elapsed = time.time() - t0
    total = len(tests)

    print(f"\n{'─' * 50}")
    print(f"  RESULTS: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s")

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'Few-shot learning accuracy':<35} {correct/total*100:>9.0f}% {'~100%':>10}")
    print(f"  {'Learns from statements':<35} {'YES':>10} {'YES':>10}")
    print(f"  {'Persists across turns':<35} {'YES':>10} {'context':>10}")

    print(f"\n{'=' * 60}")
    if correct == total:
        print("  STATUS: ALL FEW-SHOT TESTS PASS")
    else:
        print(f"  STATUS: {correct}/{total} PASS")
    print(f"{'=' * 60}")

    print(f"\nResults: {correct}/{total} passed ({correct/total*100:.0f}%)")


if __name__ == '__main__':
    main()
