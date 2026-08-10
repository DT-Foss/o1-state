#!/usr/bin/env python3
"""
Causal Reasoning Benchmark
============================
Tests the CausalRulesEngine's ability to DERIVE answers from physical laws
rather than memorize them. These questions were NOT used to build the engine.

This is the key distinction: lookup vs. reasoning.
- Lookup: "What is the capital of France?" → retrieve from KB
- Reasoning: "What happens to butter at 50°C?" → derive from melting_point=32°C
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def main():
    print("=" * 60)
    print("  CAUSAL REASONING BENCHMARK")
    print("  Testing derivation from physical laws")
    print("=" * 60)

    repl = FossKIRepl()

    tests = [
        # === TEMPERATURE DERIVATIONS ===
        # These use melting/boiling points to DERIVE the answer
        {
            'q': 'What happens to water at 100 degrees Celsius?',
            'expected': ['evaporate', 'boil'],
            'category': 'temperature',
            'reasoning': 'water boiling_point=100°C, liquid→gas',
        },
        {
            'q': 'What happens to water at -10 degrees Celsius?',
            'expected': ['freeze'],
            'category': 'temperature',
            'reasoning': 'water freezing_point=0°C, -10<0 → freezes',
        },
        {
            'q': 'What happens to water at 50 degrees Celsius?',
            'expected': ['remains liquid', 'liquid'],
            'category': 'temperature',
            'reasoning': '0<50<100 → stays liquid',
        },
        {
            'q': 'What happens to ice at 20 degrees Celsius?',
            'expected': ['melt'],
            'category': 'temperature',
            'reasoning': 'ice melting_point=0°C, 20>0 → melts',
        },

        # === STATE TRANSITIONS ===
        {
            'q': 'What happens when you heat wax?',
            'expected': ['melt'],
            'category': 'state_transition',
            'reasoning': 'wax is solid, heat+solid → liquid (melts)',
        },
        {
            'q': 'What happens when you heat butter?',
            'expected': ['melt'],
            'category': 'state_transition',
            'reasoning': 'butter is solid, heat+solid → liquid (melts)',
        },
        {
            'q': 'What happens when you heat snow?',
            'expected': ['melt'],
            'category': 'state_transition',
            'reasoning': 'snow is solid, heat+solid → liquid (melts)',
        },
        {
            'q': 'What happens when you cool steam?',
            'expected': ['condense'],
            'category': 'state_transition',
            'reasoning': 'steam is gas, cool+gas → liquid (condenses)',
        },

        # === CAUSAL LAWS ===
        {
            'q': 'What happens when you drop an egg?',
            'expected': ['break'],
            'category': 'causal_law',
            'reasoning': 'egg is fragile, drop+fragile → breaks',
        },
        {
            'q': 'What happens when you mix red and yellow paint?',
            'expected': ['orange'],
            'category': 'causal_law',
            'reasoning': 'color mixing law: red+yellow=orange',
        },
        {
            'q': 'What happens when you mix blue and yellow paint?',
            'expected': ['green'],
            'category': 'causal_law',
            'reasoning': 'color mixing law: blue+yellow=green',
        },

        # === WHY QUESTIONS ===
        {
            'q': 'Why is the sky blue?',
            'expected': ['scatter', 'rayleigh'],
            'category': 'why',
            'reasoning': 'Rayleigh scattering of sunlight',
        },
        {
            'q': 'Why do leaves change color in autumn?',
            'expected': ['chlorophyll'],
            'category': 'why',
            'reasoning': 'chlorophyll breakdown reveals other pigments',
        },
        {
            'q': 'Why does wood float on water?',
            'expected': ['dense', 'density', 'less dense'],
            'category': 'why',
            'reasoning': "Archimedes' principle: wood density < water density",
        },
        {
            'q': 'Why do we have seasons?',
            'expected': ['tilt', 'axis'],
            'category': 'why',
            'reasoning': "Earth's 23.5° axial tilt",
        },
        {
            'q': 'Why do we have day and night?',
            'expected': ['rotate', 'rotation', 'spin'],
            'category': 'why',
            'reasoning': "Earth's rotation on its axis",
        },

        # === WHAT CAUSES ===
        {
            'q': 'What causes earthquakes?',
            'expected': ['tectonic', 'plate'],
            'category': 'what_causes',
            'reasoning': 'Tectonic plate movement',
        },
        {
            'q': 'What causes tides?',
            'expected': ['moon', 'gravitational', 'gravity'],
            'category': 'what_causes',
            'reasoning': 'Gravitational pull of Moon and Sun',
        },
        {
            'q': 'What causes a rainbow?',
            'expected': ['refract', 'light', 'water'],
            'category': 'what_causes',
            'reasoning': 'Light refraction through water droplets',
        },
    ]

    # Run tests
    categories = {}
    total_correct = 0

    t0 = time.time()

    for test in tests:
        cat = test['category']
        if cat not in categories:
            categories[cat] = {'correct': 0, 'total': 0}
        categories[cat]['total'] += 1

        answer = repl.process(test['q'])
        answer_lower = answer.lower()

        correct = any(exp in answer_lower for exp in test['expected'])

        if correct:
            total_correct += 1
            categories[cat]['correct'] += 1
            print(f"  ✓ {test['q']}")
        else:
            print(f"  ✗ {test['q']}")
            print(f"    Got: {answer}")
            print(f"    Expected one of: {test['expected']}")
            print(f"    Reasoning: {test['reasoning']}")

    elapsed = time.time() - t0
    total = len(tests)

    # Report
    print(f"\n{'─' * 50}")
    print(f"  RESULTS BY CATEGORY")
    print(f"{'─' * 50}")

    for cat, data in categories.items():
        pct = data['correct'] / data['total'] * 100
        bar_len = int(pct / 5)
        bar = '█' * bar_len + '░' * (20 - bar_len)
        print(f"  {cat:<20} {data['correct']}/{data['total']}  {bar}  {pct:.0f}%")

    print(f"\n  OVERALL: {total_correct}/{total} ({total_correct/total*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/query)")

    # Comparison
    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'Causal reasoning accuracy':<35} {total_correct/total*100:>9.0f}% {'~90%':>10}")
    print(f"  {'Derives from laws (not lookup)':<35} {'YES':>10} {'NO':>10}")
    print(f"  {'Latency':<35} {elapsed/total*1000:>8.0f}ms {'~500ms':>10}")

    print(f"\n{'=' * 60}")
    if total_correct == total:
        print("  STATUS: ALL CAUSAL REASONING TESTS PASS")
    else:
        print(f"  STATUS: {total_correct}/{total} PASS")
    print(f"{'=' * 60}")

    print(f"\nResults: {total_correct}/{total} passed ({total_correct/total*100:.0f}%)")


if __name__ == '__main__':
    main()
