#!/usr/bin/env python3
"""
Hard Questions Benchmark
=========================
Tests questions that require multiple reasoning strategies, trick questions,
reverse lookups, and non-standard phrasings. These are the questions that
distinguish a competent AI from a basic lookup system.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def main():
    print("=" * 60)
    print("  HARD QUESTIONS BENCHMARK")
    print("  Testing advanced reasoning and retrieval")
    print("=" * 60)

    repl = FossKIRepl()

    tests = [
        # Color mixing (causal reasoning)
        {'q': 'If I mix red and blue paint, what color do I get?', 'contains': 'purple', 'cat': 'causal'},

        # Trick questions
        {'q': 'What is heavier, a kilogram of steel or a kilogram of feathers?', 'contains': 'same', 'cat': 'trick'},

        # Reverse lookups (known_as → entity)
        {'q': 'What planet is known as the Red Planet?', 'contains': 'mars', 'cat': 'reverse'},
        {'q': 'What organ pumps blood through the body?', 'contains': 'heart', 'cat': 'reverse'},

        # Superlative with non-standard scoping
        {'q': 'What is the largest planet in our solar system?', 'contains': 'jupiter', 'cat': 'superlative'},
        {'q': 'What is the smallest planet in our solar system?', 'contains': 'mercury', 'cat': 'superlative'},

        # Reverse known_for
        {'q': 'Who was the first person to walk on the moon?', 'contains': 'armstrong', 'cat': 'reverse'},

        # Temporal with "what year"
        {'q': 'What year did World War II end?', 'contains': '1945', 'cat': 'temporal'},

        # Deepest (superlative)
        {'q': 'What is the deepest ocean?', 'contains': 'pacific', 'cat': 'superlative'},

        # Population comparison
        {'q': 'What country has the most people?', 'contains': 'china', 'cat': 'comparison'},

        # Compound subject lookup
        {'q': 'What is the boiling point of water?', 'contains': '100', 'cat': 'compound'},
        {'q': 'What is the freezing point of water?', 'contains': '0', 'cat': 'compound'},

        # "for" instead of "of"
        {'q': 'What is the chemical formula for water?', 'contains': 'h2o', 'cat': 'phrasing'},

        # Spider legs (commonsense)
        {'q': 'How many legs does a spider have?', 'contains': '8', 'cat': 'commonsense'},

        # Chemical symbol
        {'q': 'What is the chemical symbol for gold?', 'contains': 'au', 'cat': 'science'},

        # "What are..." concepts
        {'q': 'What are the primary colors?', 'contains': 'red', 'cat': 'concept'},
    ]

    t0 = time.time()
    correct = 0
    cat_results = {}

    for test in tests:
        answer = repl.process(test['q'])
        answer_lower = answer.lower()
        expected_lower = test['contains'].lower()

        ok = expected_lower in answer_lower
        cat = test['cat']
        if cat not in cat_results:
            cat_results[cat] = {'correct': 0, 'total': 0}
        cat_results[cat]['total'] += 1

        if ok:
            correct += 1
            cat_results[cat]['correct'] += 1
            print(f"  ✓ {test['q']}")
            print(f"    → {answer[:100]}")
        else:
            print(f"  ✗ {test['q']}")
            print(f"    Got: {answer[:100]}")
            print(f"    Expected to contain: {test['contains']}")

    elapsed = time.time() - t0
    total = len(tests)

    print(f"\n{'─' * 50}")
    print(f"  RESULTS: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/query)")

    print(f"\n  {'Category':<20} {'Score':>10}")
    print(f"  {'─' * 32}")
    for cat, r in sorted(cat_results.items()):
        pct = r['correct'] / r['total'] * 100
        print(f"  {cat:<20} {r['correct']}/{r['total']:>2} ({pct:.0f}%)")

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'Hard questions accuracy':<35} {correct/total*100:>9.0f}% {'~95%':>10}")
    print(f"  {'Latency':<35} {elapsed/total*1000:>8.0f}ms {'~500ms':>10}")

    print(f"\n{'=' * 60}")
    if correct == total:
        print("  STATUS: ALL HARD QUESTIONS PASS")
    else:
        print(f"  STATUS: {correct}/{total} PASS")
    print(f"{'=' * 60}")

    print(f"\nResults: {correct}/{total} passed ({correct/total*100:.0f}%)")


if __name__ == '__main__':
    main()
