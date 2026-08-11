"""
Benchmark: Instruction Parser (A6)
=====================================
Tests:
  1. Single instruction parsing
  2. Compound instruction splitting
  3. Knowledge + Math + Tool routing
  4. Full execution pipeline
  5. Chain-of-thought detection
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.instructor import InstructionParser, InstructionExecutor
from core.knowledge import KnowledgeStore
from core.router import VortexRouter
from core.chain import ChainOfThought
from core.math_solver import MathSolver
from core.tools import ToolExecutor


def test_single_parse():
    """Test 1: Single instruction parsing."""
    print("Test 1: Single Instruction Parsing")
    parser = InstructionParser()
    results = []

    cases = [
        ("What is the capital of France?", 'knowledge'),
        ("Calculate 17 * 23", 'math'),
        ("Read file config.py", 'tool'),
        ("Why does ice float?", 'reasoning'),
        ("Find all *.py files", 'tool'),
        ("Solve x + 3 = 7", 'math'),
        ("GCD of 48 and 18", 'math'),
    ]

    for instruction, expected_type in cases:
        actions = parser.parse(instruction)
        ok = len(actions) >= 1 and actions[0].action_type == expected_type
        results.append(ok)
        actual = actions[0].action_type if actions else 'none'
        print(f"  {'PASS' if ok else 'FAIL'}: \"{instruction[:40]}\" → {actual} (expected {expected_type})")

    return all(results)


def test_compound():
    """Test 2: Compound instruction splitting."""
    print("Test 2: Compound Instructions")
    parser = InstructionParser()
    results = []

    # Two separate actions
    actions = parser.parse("What is the capital of France and then calculate 2+2")
    ok = len(actions) == 2
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: 'X and then Y' → {len(actions)} actions")

    if len(actions) == 2:
        ok1 = actions[0].action_type == 'knowledge'
        ok2 = actions[1].action_type == 'math'
        results.append(ok1)
        results.append(ok2)
        print(f"  {'PASS' if ok1 else 'FAIL'}: Action 1 = {actions[0].action_type}")
        print(f"  {'PASS' if ok2 else 'FAIL'}: Action 2 = {actions[1].action_type}")
    else:
        results.extend([False, False])

    return all(results)


def test_routing():
    """Test 3: Correct routing to subsystems."""
    print("Test 3: Subsystem Routing")
    parser = InstructionParser()
    results = []

    routing_tests = [
        ("derivative of 3x^2 + 2x", 'math'),
        ("grep for 'TODO' in core/", 'tool'),
        ("list files in /tmp", 'tool'),
        ("tell me about Einstein", 'knowledge'),
        ("prove that 2+2=4", 'reasoning'),
        ("Is 97 prime?", 'math'),
    ]

    for instruction, expected_type in routing_tests:
        actions = parser.parse(instruction)
        actual = actions[0].action_type if actions else 'none'
        ok = actual == expected_type
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: \"{instruction[:35]}\" → {actual}")

    return all(results)


def test_execution():
    """Test 4: Full execution pipeline."""
    print("Test 4: Full Execution")

    ks = KnowledgeStore(dim=64)
    ks.store_fact('France', 'capital', 'Paris')
    ks.store_fact('Germany', 'capital', 'Berlin')
    ks.store_fact('Water', 'formula', 'H2O')

    router = VortexRouter(knowledge_store=ks)
    math = MathSolver()
    tools = ToolExecutor(confirm_dangerous=False)

    executor = InstructionExecutor(
        router=router, math_solver=math, tool_executor=tools
    )
    results = []

    # Knowledge query
    r = executor.execute("What is the capital of France?")
    ok = r['answer'] and 'Paris' in str(r['answer'])
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Knowledge → {r.get('answer')}")

    # Math query
    r = executor.execute("Calculate 17 * 23")
    ok = r['answer'] and '391' in str(r['answer'])
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Math → {r.get('answer')}")

    # Tool: read this benchmark file
    r = executor.execute(f"Read file {__file__}")
    ok = r['answer'] is not None
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Tool read → {'OK' if ok else 'FAIL'}")

    return all(results)


def test_chain_detection():
    """Test 5: Chain-of-thought queries detected."""
    print("Test 5: Chain Detection")
    parser = InstructionParser()
    results = []

    chain_queries = [
        "In which country was the physicist who discovered radium born?",
        "What is the capital of the country of Einstein?",
    ]

    for q in chain_queries:
        actions = parser.parse(q)
        ok = len(actions) >= 1 and actions[0].action_type == 'chain'
        results.append(ok)
        actual = actions[0].action_type if actions else 'none'
        print(f"  {'PASS' if ok else 'FAIL'}: \"{q[:50]}\" → {actual}")

    return all(results)


def test_explain():
    """Test 6: Human-readable explanation."""
    print("Test 6: Explanation Output")
    parser = InstructionParser()
    results = []

    explanation = parser.explain("What is the capital of France?")
    ok = 'knowledge' in explanation.lower()
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Explanation contains 'knowledge'")
    print(f"  {explanation}")

    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("INSTRUCTION PARSER BENCHMARK (A6)")
    print("=" * 60)
    print()

    tests = [
        test_single_parse,
        test_compound,
        test_routing,
        test_execution,
        test_chain_detection,
        test_explain,
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
