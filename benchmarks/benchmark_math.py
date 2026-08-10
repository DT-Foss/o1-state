"""
Benchmark: Math Solver (A4)
==============================
Tests:
  1. Arithmetic: basic operations
  2. Linear equations
  3. Quadratic equations
  4. Systems of equations
  5. Derivatives
  6. Integrals
  7. Number theory (GCD, LCM, prime, factorize)
  8. Integration via ReasoningEngine
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.math_solver import MathSolver
from core.reasoning import ReasoningEngine


def test_arithmetic():
    """Test 1: Basic arithmetic."""
    print("Test 1: Arithmetic")
    ms = MathSolver()
    results = []

    cases = [
        ("17 * 23", "391"),
        ("100 / 4", "25"),
        ("2 ^ 10", "1024"),
        ("What is 15 + 27?", "42"),
        ("calculate 99 - 37", "62"),
        ("17 % 5", "2"),
    ]

    for query, expected in cases:
        r = ms.solve(query)
        # Compare numerically
        try:
            ok = float(r['answer']) == float(expected) if r['answer'] else False
        except (ValueError, TypeError):
            ok = str(r.get('answer', '')) == expected
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {r.get('answer')} (expected {expected})")

    return all(results)


def test_linear():
    """Test 2: Linear equations."""
    print("Test 2: Linear Equations")
    ms = MathSolver()
    results = []

    cases = [
        ("solve x + 3 = 7", "4"),
        ("solve 2x + 5 = 11", "3"),
        ("x - 5 = 10", "15"),
    ]

    for query, expected in cases:
        r = ms.solve(query)
        answer = r.get('answer', '')
        ok = expected in str(answer)
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer}")

    return all(results)


def test_quadratic():
    """Test 3: Quadratic equations."""
    print("Test 3: Quadratic Equations")
    ms = MathSolver()
    results = []

    # x^2 - 5x + 6 = 0 → x = 3 or x = 2
    r = ms.solve("solve x^2 - 5x + 6 = 0")
    answer = str(r.get('answer', ''))
    ok = '3' in answer and '2' in answer
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: x^2 - 5x + 6 = 0 → {answer}")

    # x^2 + 1 = 0 → complex roots
    r = ms.solve("solve x^2 + 0x + 1 = 0")
    answer = str(r.get('answer', ''))
    ok2 = 'complex' in answer.lower() or 'i' in answer
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: x^2 + 1 = 0 → {answer}")

    return all(results)


def test_system():
    """Test 4: Systems of equations."""
    print("Test 4: Systems of Equations")
    ms = MathSolver()
    results = []

    # 2x + 3y = 7, x - y = 1 → x=2, y=1
    r = ms.solve("2x + 3y = 7, x - y = 1")
    answer = str(r.get('answer', ''))
    ok = '2' in answer and '1' in answer
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: 2x+3y=7, x-y=1 → {answer}")

    return all(results)


def test_derivative():
    """Test 5: Polynomial derivatives."""
    print("Test 5: Derivatives")
    ms = MathSolver()
    results = []

    cases = [
        ("derivative of 3x^2 + 2x - 1", "6x + 2"),
        ("derivative of x^3", "3x^2"),
        ("derivative of 5", "0"),
    ]

    for query, expected in cases:
        r = ms.solve(query)
        answer = str(r.get('answer', ''))
        ok = answer.replace(' ', '') == expected.replace(' ', '')
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer} (expected {expected})")

    return all(results)


def test_integral():
    """Test 6: Polynomial integrals."""
    print("Test 6: Integrals")
    ms = MathSolver()
    results = []

    # integral of 6x + 2 → 3x^2 + 2x + C
    r = ms.solve("integral of 6x + 2")
    answer = str(r.get('answer', ''))
    ok = '3x^2' in answer.replace(' ', '') and '+ C' in answer
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: ∫(6x+2)dx → {answer}")

    return all(results)


def test_number_theory():
    """Test 7: Number theory operations."""
    print("Test 7: Number Theory")
    ms = MathSolver()
    results = []

    cases = [
        ("GCD of 84 and 36", "12"),
        ("LCM of 12 and 18", "36"),
        ("Is 97 prime?", "prime"),
        ("Is 100 prime?", "not prime"),
        ("factorize 84", "2 × 2 × 3 × 7"),
        ("fibonacci 10", "55"),
        ("10!", "3628800"),
    ]

    for query, expected in cases:
        r = ms.solve(query)
        answer = str(r.get('answer', ''))
        ok = expected in answer
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer}")

    return all(results)


def test_reasoning_integration():
    """Test 8: Math via ReasoningEngine (Fiber 3)."""
    print("Test 8: ReasoningEngine Integration")
    engine = ReasoningEngine()
    results = []

    # Arithmetic via reasoning
    r = engine.reason("What is 17 * 23?")
    ok = r.get('answer') and '391' in str(r['answer'])
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: 17*23 via Reasoning → {r.get('answer')}")

    # GCD via reasoning
    r = engine.reason("GCD of 48 and 18")
    ok2 = r.get('answer') and '6' in str(r['answer'])
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: GCD(48,18) via Reasoning → {r.get('answer')}")

    # Solve via existing prove path
    r = engine.reason("Solve x + 5 = 12")
    ok3 = r.get('answer') and '7' in str(r['answer'])
    results.append(ok3)
    print(f"  {'PASS' if ok3 else 'FAIL'}: x+5=12 via Reasoning → {r.get('answer')}")

    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("MATH SOLVER BENCHMARK (A4)")
    print("=" * 60)
    print()

    tests = [
        test_arithmetic,
        test_linear,
        test_quadratic,
        test_system,
        test_derivative,
        test_integral,
        test_number_theory,
        test_reasoning_integration,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            ok = test()
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            import traceback
            print(f"  ERROR: {e}")
            traceback.print_exc()
            failed += 1
        print()

    print("=" * 60)
    total = passed + failed
    print(f"RESULTS: {passed}/{total} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED")
    print("=" * 60)
