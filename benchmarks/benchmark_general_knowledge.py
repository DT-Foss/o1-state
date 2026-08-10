#!/usr/bin/env python3
"""
General Knowledge Benchmark
=============================
Tests the kind of questions a normal user would ask — the bread-and-butter
that GPT-4o handles natively. These test KB retrieval, routing, and
natural language understanding.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def main():
    print("=" * 60)
    print("  GENERAL KNOWLEDGE BENCHMARK")
    print("  Testing everyday user questions")
    print("=" * 60)

    repl = FossKIRepl()

    tests = [
        # Superlative questions
        {'q': 'What is the largest ocean?', 'contains': 'pacific', 'cat': 'superlative'},
        {'q': 'What is the tallest mountain in the world?', 'contains': 'everest', 'cat': 'superlative'},
        {'q': 'What is the largest country in the world?', 'contains': 'russia', 'cat': 'superlative'},
        {'q': 'What is the smallest country in the world?', 'contains': 'vatican', 'cat': 'superlative'},
        {'q': 'What is the longest river in the world?', 'contains': 'nile', 'cat': 'superlative'},
        {'q': 'What is the largest desert in the world?', 'contains': 'desert', 'cat': 'superlative'},

        # Formula/property questions
        {'q': 'What is the chemical formula for water?', 'contains': 'h2o', 'cat': 'science'},
        {'q': 'What is the speed of light?', 'contains': '299792458', 'cat': 'science'},
        {'q': 'What is the boiling point of water?', 'contains': '100', 'cat': 'science'},
        {'q': 'What is the freezing point of water?', 'contains': '0', 'cat': 'science'},

        # "What are..." concept questions
        {'q': 'What are the three states of matter?', 'contains': 'solid', 'cat': 'concept'},
        {'q': 'What are the primary colors?', 'contains': 'red', 'cat': 'concept'},
        {'q': 'What are the cardinal directions?', 'contains': 'north', 'cat': 'concept'},
        {'q': 'What are the seasons?', 'contains': 'spring', 'cat': 'concept'},

        # Person questions
        {'q': 'Who wrote Romeo and Juliet?', 'contains': 'shakespeare', 'cat': 'person'},
        {'q': 'Who invented the telephone?', 'contains': 'bell', 'cat': 'person'},
        {'q': 'Who discovered penicillin?', 'contains': 'fleming', 'cat': 'person'},
        {'q': 'Who painted the Mona Lisa?', 'contains': 'vinci', 'cat': 'person'},

        # Country facts
        {'q': 'What is the capital of Japan?', 'contains': 'tokyo', 'cat': 'geography'},
        {'q': 'What is the capital of France?', 'contains': 'paris', 'cat': 'geography'},
        {'q': 'What is the population of China?', 'contains': '1.4', 'cat': 'geography'},
        {'q': 'What language is spoken in Brazil?', 'contains': 'portuguese', 'cat': 'geography'},
        {'q': 'What currency is used in Japan?', 'contains': 'yen', 'cat': 'geography'},
        {'q': 'What continent is Egypt in?', 'contains': 'africa', 'cat': 'geography'},

        # History
        {'q': 'When was the French Revolution?', 'contains': '1789', 'cat': 'history'},

        # Definition questions
        {'q': 'What is photosynthesis?', 'contains': 'process', 'cat': 'definition'},
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
    print(f"  {'General knowledge accuracy':<35} {correct/total*100:>9.0f}% {'~98%':>10}")
    print(f"  {'Latency':<35} {elapsed/total*1000:>8.0f}ms {'~500ms':>10}")

    print(f"\n{'=' * 60}")
    if correct == total:
        print("  STATUS: ALL GENERAL KNOWLEDGE TESTS PASS")
    else:
        print(f"  STATUS: {correct}/{total} PASS")
    print(f"{'=' * 60}")

    print(f"\nResults: {correct}/{total} passed ({correct/total*100:.0f}%)")


if __name__ == '__main__':
    main()
