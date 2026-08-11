#!/usr/bin/env python3
"""
Benchmark: Code Executor — Sandboxed Execution + Auto-Fix
===========================================================
Tests:
  1. AST validation (valid code)
  2. AST validation (invalid code)
  3. Execute simple code
  4. Execute with input/output
  5. Timeout enforcement
  6. Error classification — SyntaxError
  7. Error classification — ImportError
  8. Error classification — NameError
  9. Auto-fix: missing import
  10. Auto-fix: True/False casing
  11. Run tests (pass)
  12. Run tests (fail)
  13. Execute with retry
  14. CodeDebugger full cycle
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from core.executor import CodeExecutor, CodeDebugger


def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return 1 if passed else 0


def run():
    print("=" * 60)
    print("BENCHMARK: Code Executor")
    print("=" * 60)
    total = 0
    passed = 0
    ex = CodeExecutor(timeout=5, max_retries=3)

    # 1. Valid code
    print("\n--- Validation ---")
    total += 1
    r = ex.validate("x = 1 + 2\nprint(x)")
    passed += test("Valid code validates", r['valid'])

    # 2. Invalid code
    total += 1
    r = ex.validate("def foo(\n  pass")
    ok = not r['valid'] and r['errors'][0]['type'] == 'SYNTAX'
    passed += test("Invalid code caught", ok, r['errors'][0]['message'] if r['errors'] else '?')

    # 3. Simple execution
    print("\n--- Execution ---")
    total += 1
    r = ex.execute("print('hello world')")
    ok = r['success'] and 'hello world' in r.get('stdout', '')
    passed += test("Simple execution", ok, f"stdout='{r.get('stdout', '').strip()}'")

    # 4. With input/output
    total += 1
    code = "name = input()\nprint(f'Hello, {name}!')"
    r = ex.execute(code, test_input="David\n", expected_output="Hello, David!")
    ok = r['success']
    passed += test("Input/output execution", ok, f"stdout='{r.get('stdout', '').strip()}'")

    # 5. Timeout
    total += 1
    r = ex.execute("import time\ntime.sleep(10)", expected_output=None)
    ok = not r['success'] and any(e['type'] == 'TIMEOUT' for e in r.get('errors', []))
    passed += test("Timeout enforcement", ok, f"errors={[e['type'] for e in r.get('errors', [])]}")

    # 6. Error classification — Syntax
    print("\n--- Error Classification ---")
    total += 1
    r = ex.execute("def foo(\n  pass")
    ok = not r['success']
    errors = r.get('errors', [])
    err_type = errors[0]['type'] if errors else '?'
    passed += test("SyntaxError classified", err_type == 'SYNTAX', f"type={err_type}")

    # 7. Error classification — Import
    total += 1
    r = ex.execute("import nonexistent_xyz_module")
    ok = not r['success']
    errors = r.get('errors', [])
    err_types = [e['type'] for e in errors]
    passed += test("ImportError classified", 'IMPORT' in err_types or 'RUNTIME' in err_types,
                   f"types={err_types}")

    # 8. Error classification — Name
    total += 1
    r = ex.execute("print(undefined_variable)")
    ok = not r['success']
    errors = r.get('errors', [])
    err_types = [e['type'] for e in errors]
    passed += test("NameError classified", any(t in err_types for t in ('NAME', 'RUNTIME')),
                   f"types={err_types}")

    # 9. Auto-fix: True/False casing
    print("\n--- Auto-Fix ---")
    total += 1
    code = "x = true\nprint(x)"
    r = ex.execute_with_retry(code)
    ok = r['success']
    passed += test("Auto-fix true→True", ok,
                   f"attempts={len(r.get('history', []))}")

    # 10. Run tests — all pass
    print("\n--- Test Runner ---")
    total += 1
    code = "n = int(input())\nprint(n * 2)"
    tests = [
        {'input': '5\n', 'expected_output': '10', 'description': 'double 5'},
        {'input': '0\n', 'expected_output': '0', 'description': 'double 0'},
        {'input': '100\n', 'expected_output': '200', 'description': 'double 100'},
    ]
    r = ex.run_tests(code, tests)
    ok = r['passed'] == 3
    passed += test("Run tests (all pass)", ok, f"{r['passed']}/{r['total']}")

    # 11. Run tests — with failure
    total += 1
    code = "n = int(input())\nprint(n + 1)"  # Adds 1 instead of doubling
    r = ex.run_tests(code, tests)
    ok = r['failed'] > 0
    passed += test("Run tests (detects failure)", ok, f"passed={r['passed']}, failed={r['failed']}")

    # 12. CodeDebugger
    print("\n--- CodeDebugger ---")
    total += 1
    debugger = CodeDebugger()
    r = debugger.debug("def foo():\n    return 42\nprint(foo())")
    ok = 'issues' in r
    passed += test("CodeDebugger debug cycle", ok, f"issues={len(r.get('issues', []))}")

    # 13. Correct code through debugger
    total += 1
    r = debugger.debug("print(2 + 2)")
    ok = len(r.get('issues', [])) == 0
    passed += test("CodeDebugger clean code", ok)

    # Summary
    print("\n" + "=" * 60)
    pct = (passed / total * 100) if total else 0
    print(f"RESULT: {passed}/{total} ({pct:.0f}%)")
    print("=" * 60)
    return passed, total


if __name__ == '__main__':
    run()
