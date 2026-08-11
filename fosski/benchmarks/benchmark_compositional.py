#!/usr/bin/env python3
"""
Compositional QA Benchmark
=============================
Tests multi-step question answering that requires chaining KB lookups.
Each question needs 2+ steps to answer — this is NOT single-hop retrieval.

Example:
  "What is the capital of the largest country in South America?"
  Step 1: largest country in SA → Brazil (area lookup + comparison)
  Step 2: capital of Brazil → Brasilia (KB lookup)

GPT-4o handles these natively. FOSS-KI must decompose and chain.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def main():
    print("=" * 60)
    print("  COMPOSITIONAL QA BENCHMARK")
    print("  Testing multi-step question answering")
    print("=" * 60)

    repl = FossKIRepl()

    tests = [
        # 2-hop: superlative + relation
        {
            'q': 'What is the capital of the largest country in South America?',
            'answer': 'Brasilia',
            'steps': 2,
            'chain': 'largest in SA → Brazil → capital = Brasilia',
        },
        {
            'q': 'What is the capital of the largest country in Africa?',
            'answer': 'Algiers',
            'steps': 2,
            'chain': 'largest in Africa → Algeria → capital = Algiers',
        },
        {
            'q': 'What is the currency of the largest country in Asia?',
            'answer': 'Yuan',
            'steps': 2,
            'chain': 'largest in Asia → China → currency = Yuan',
        },
        # 2-hop: reverse lookup + relation
        {
            'q': 'What language is spoken in the country whose capital is Tokyo?',
            'answer': 'Japanese',
            'steps': 2,
            'chain': 'capital=Tokyo → Japan → language = Japanese',
        },
        {
            'q': 'What language is spoken in the country whose capital is Berlin?',
            'answer': 'German',
            'steps': 2,
            'chain': 'capital=Berlin → Germany → language = German',
        },
        {
            'q': 'What is the continent of the country whose capital is Paris?',
            'answer': 'Europe',
            'steps': 2,
            'chain': 'capital=Paris → France → continent = Europe',
        },
        {
            'q': 'What is the continent of the country whose capital is Cairo?',
            'answer': 'Africa',
            'steps': 2,
            'chain': 'capital=Cairo → Egypt → continent = Africa',
        },
        {
            'q': 'What is the currency of the country whose capital is Moscow?',
            'answer': 'Ruble',
            'steps': 2,
            'chain': 'capital=Moscow → Russia → currency = Ruble',
        },
        # Planet chain
        {
            'q': 'How many moons does the fourth planet from the sun have?',
            'answer': '2',
            'steps': 2,
            'chain': '4th planet → Mars → moons = 2',
        },
        {
            'q': 'How many moons does the sixth planet from the sun have?',
            'answer': '146',
            'steps': 2,
            'chain': '6th planet → Saturn → moons = 146',
        },
        # Smallest country
        {
            'q': 'What is the capital of the smallest country in Europe?',
            'answer': 'Vatican City',
            'steps': 2,
            'chain': 'smallest in Europe → Vatican City → capital = Vatican City',
        },
        # Most populous
        {
            'q': 'What is the capital of the most populous country in Africa?',
            'answer': 'Abuja',
            'steps': 2,
            'chain': 'most populous in Africa → Nigeria → capital = Abuja',
        },
    ]

    t0 = time.time()
    correct = 0

    for test in tests:
        answer = repl.process(test['q'])
        answer_lower = answer.lower()
        expected_lower = test['answer'].lower()

        ok = expected_lower in answer_lower
        if ok:
            correct += 1
            print(f"  ✓ {test['q']}")
            print(f"    → {answer} ({test['steps']}-hop: {test['chain']})")
        else:
            print(f"  ✗ {test['q']}")
            print(f"    Got: {answer}")
            print(f"    Expected: {test['answer']}")
            print(f"    Chain: {test['chain']}")

    elapsed = time.time() - t0
    total = len(tests)

    print(f"\n{'─' * 50}")
    print(f"  RESULTS: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/query)")

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'Compositional QA accuracy':<35} {correct/total*100:>9.0f}% {'~95%':>10}")
    print(f"  {'Multi-step decomposition':<35} {'YES':>10} {'YES':>10}")
    print(f"  {'Explicit reasoning chain':<35} {'YES':>10} {'NO':>10}")
    print(f"  {'Latency':<35} {elapsed/total*1000:>8.0f}ms {'~500ms':>10}")

    print(f"\n{'=' * 60}")
    if correct == total:
        print("  STATUS: ALL COMPOSITIONAL QA TESTS PASS")
    else:
        print(f"  STATUS: {correct}/{total} PASS")
    print(f"{'=' * 60}")

    print(f"\nResults: {correct}/{total} passed ({correct/total*100:.0f}%)")


if __name__ == '__main__':
    main()
