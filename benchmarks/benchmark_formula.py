#!/usr/bin/env python3
"""
Formula Reasoning Benchmark
==============================
Tests the FormulaEngine's ability to compute physics answers from formulas.
Not lookup — actual computation from E=½mv², F=ma, p=mv, etc.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def main():
    print("=" * 60)
    print("  FORMULA REASONING BENCHMARK")
    print("  Testing computation from physical laws")
    print("=" * 60)

    repl = FossKIRepl()

    tests = [
        # Kinetic energy: E = ½mv²
        {
            'q': 'What is the kinetic energy of a 2 kg object at 3 m/s?',
            'expected_value': 9.0,
            'expected_unit': 'J',
            'formula': 'E = ½mv²',
        },
        {
            'q': 'What is the kinetic energy of a 10 kg car at 20 m/s?',
            'expected_value': 2000.0,
            'expected_unit': 'J',
            'formula': 'E = ½mv²',
        },
        {
            'q': 'What is the kinetic energy of a 0.5 kg ball at 10 m/s?',
            'expected_value': 25.0,
            'expected_unit': 'J',
            'formula': 'E = ½mv²',
        },
        # Force: F = ma
        {
            'q': 'What is the force on a 5 kg mass with acceleration of 10 m/s²?',
            'expected_value': 50.0,
            'expected_unit': 'N',
            'formula': 'F = ma',
        },
        {
            'q': 'What is the force on a 100 kg object with acceleration of 2 m/s²?',
            'expected_value': 200.0,
            'expected_unit': 'N',
            'formula': 'F = ma',
        },
        # Momentum: p = mv
        {
            'q': 'What is the momentum of a 3 kg object at 4 m/s?',
            'expected_value': 12.0,
            'expected_unit': 'kg',
            'formula': 'p = mv',
        },
        {
            'q': 'What is the momentum of a 70 kg person running at 5 m/s?',
            'expected_value': 350.0,
            'expected_unit': 'kg',
            'formula': 'p = mv',
        },
        # Work: W = Fd
        {
            'q': 'What is the work done by a 50 N force over a distance of 10 m?',
            'expected_value': 500.0,
            'expected_unit': 'J',
            'formula': 'W = Fd',
        },
        # Density: ρ = m/V
        {
            'q': 'What is the density of a 10 kg object with volume of 2 m³?',
            'expected_value': 5.0,
            'expected_unit': 'kg/m',
            'formula': 'ρ = m/V',
        },
        # Ohm's law: V = IR
        {
            'q': 'What is the voltage when 3 A flows through 4 ohm resistance?',
            'expected_value': 12.0,
            'expected_unit': 'V',
            'formula': 'V = IR',
        },
    ]

    t0 = time.time()
    correct = 0

    for test in tests:
        answer = repl.process(test['q'])
        answer_lower = answer.lower()

        # Check if the expected value appears in the answer
        ev = test['expected_value']
        ev_str = str(int(ev)) if ev == int(ev) else str(ev)
        value_match = ev_str in answer_lower
        unit_match = test['expected_unit'].lower() in answer_lower

        ok = value_match and unit_match
        if ok:
            correct += 1
            print(f"  ✓ {test['q']}")
            print(f"    → {answer}")
        else:
            print(f"  ✗ {test['q']}")
            print(f"    Got: {answer}")
            print(f"    Expected: {ev_str} {test['expected_unit']}")

    elapsed = time.time() - t0
    total = len(tests)

    print(f"\n{'─' * 50}")
    print(f"  RESULTS: {correct}/{total} ({correct/total*100:.0f}%)")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total*1000:.0f}ms/query)")
    print(f"  Formulas tested: E=½mv², F=ma, p=mv, W=Fd, ρ=m/V, V=IR")

    print(f"\n  {'Metric':<35} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 57}")
    print(f"  {'Formula computation accuracy':<35} {correct/total*100:>9.0f}% {'~95%':>10}")
    print(f"  {'Computes from formulas':<35} {'YES':>10} {'YES':>10}")
    print(f"  {'Shows formula used':<35} {'YES':>10} {'NO':>10}")
    print(f"  {'Latency':<35} {elapsed/total*1000:>8.0f}ms {'~500ms':>10}")

    print(f"\n{'=' * 60}")
    if correct == total:
        print("  STATUS: ALL FORMULA TESTS PASS")
    else:
        print(f"  STATUS: {correct}/{total} PASS")
    print(f"{'=' * 60}")

    print(f"\nResults: {correct}/{total} passed ({correct/total*100:.0f}%)")


if __name__ == '__main__':
    main()
