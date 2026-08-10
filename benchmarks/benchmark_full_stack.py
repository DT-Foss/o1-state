"""
Benchmark: Full Stack Integration Test
=========================================
Tests the complete FOSS-KI system end-to-end:
  1. Knowledge Q&A (Fiber 2)
  2. Reasoning (Fiber 3)
  3. Math (Fiber 3 + MathSolver)
  4. Chain-of-Thought (multi-hop)
  5. Tool Use
  6. Conversation Memory
  7. Instruction Parsing
  8. Anti-Hallucination across all paths
  9. Mixed-mode queries (compound)
  10. REPL simulation
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def setup_repl():
    """Create a fully loaded REPL instance."""
    repl = FossKIRepl(knowledge_dim=64)

    # Load comprehensive knowledge
    facts = [
        # Geography
        ('France', 'capital', 'Paris'),
        ('Germany', 'capital', 'Berlin'),
        ('Japan', 'capital', 'Tokyo'),
        ('Poland', 'capital', 'Warsaw'),
        ('Italy', 'capital', 'Rome'),
        ('Warsaw', 'country', 'Poland'),
        ('Berlin', 'country', 'Germany'),
        ('Paris', 'country', 'France'),
        ('Tokyo', 'country', 'Japan'),
        # Science
        ('Water', 'formula', 'H2O'),
        ('Salt', 'formula', 'NaCl'),
        ('Radium', 'discoverer', 'Marie Curie'),
        ('Penicillin', 'discoverer', 'Alexander Fleming'),
        ('DNA', 'discoverer', 'Watson and Crick'),
        # People
        ('Marie Curie', 'birthplace', 'Warsaw'),
        ('Marie Curie', 'nationality', 'Polish'),
        ('Albert Einstein', 'birthplace', 'Ulm'),
        ('Albert Einstein', 'nationality', 'German'),
        ('Alexander Fleming', 'birthplace', 'Darvel'),
        # Culture
        ('Python', 'creator', 'Guido van Rossum'),
        ('Linux', 'creator', 'Linus Torvalds'),
    ]
    for s, r, o in facts:
        repl.knowledge.store_fact(s, r, o)

    return repl


def test_knowledge_qa():
    """Test 1: Basic knowledge Q&A."""
    print("Test 1: Knowledge Q&A")
    repl = setup_repl()
    results = []

    cases = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Japan?", "Tokyo"),
        ("What is the formula of Water?", "H2O"),
        ("Who discovered Radium?", "Marie Curie"),
        ("Who created Python?", "Guido van Rossum"),
    ]

    for query, expected in cases:
        answer = repl.process(query)
        ok = expected in str(answer)
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer[:50]}")

    return all(results)


def test_reasoning():
    """Test 2: Reasoning queries."""
    print("Test 2: Reasoning")
    repl = setup_repl()
    results = []

    # Why question
    r = repl.process("Why does ice float?")
    ok = 'expands' in r.lower() or 'density' in r.lower() or 'water' in r.lower()
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Why ice floats → {r[:60]}")

    return all(results)


def test_math():
    """Test 3: Math operations."""
    print("Test 3: Math")
    repl = setup_repl()
    results = []

    cases = [
        ("Calculate 17 * 23", "391"),
        ("Is 97 prime?", "prime"),
        ("GCD of 84 and 36", "12"),
        ("derivative of 3x^2 + 2x - 1", "6x + 2"),
        ("Solve x + 5 = 12", "7"),
    ]

    for query, expected in cases:
        answer = repl.process(query)
        ok = expected in str(answer)
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer[:50]}")

    return all(results)


def test_chain():
    """Test 4: Chain-of-thought multi-hop."""
    print("Test 4: Chain-of-Thought")
    repl = setup_repl()
    # Inject chain engine
    from core.chain import ChainOfThought
    repl.chain = ChainOfThought(router=repl.router)
    repl.instructor.chain = repl.chain
    results = []

    # Multi-hop: who discovered radium → born where → which country
    r = repl.process("In which country was the physicist who discovered radium born?")
    ok = 'Poland' in r or 'Warsaw' in r
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Multi-hop → {r[:60]}")

    return all(results)


def test_tools():
    """Test 5: Tool use."""
    print("Test 5: Tool Use")
    repl = setup_repl()
    repl.tools.confirm_dangerous = False
    results = []

    # Read this file
    r = repl.process(f"Read file {__file__}")
    ok = 'Benchmark' in r or len(r) > 50
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Read file → {len(r)} chars")

    # Python eval
    r = repl.process("Calculate sqrt(144)")
    ok = '12' in r
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: sqrt(144) → {r[:30]}")

    return all(results)


def test_memory():
    """Test 6: Conversation memory."""
    print("Test 6: Conversation Memory")
    repl = setup_repl()
    results = []

    # Multiple turns
    repl.process("What is the capital of France?")
    repl.process("What is the capital of Germany?")
    repl.process("What is the formula of Water?")

    # Check memory
    stats = repl.memory.get_summary()
    ok = stats['total_turns'] >= 6  # 3 user + 3 system turns
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: {stats['total_turns']} turns recorded")

    topics = repl.memory.get_recent_topics(5)
    ok2 = len(topics) >= 2
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: {len(topics)} topics: {topics[:3]}")

    return all(results)


def test_anti_hallucination():
    """Test 7: Anti-hallucination across all paths."""
    print("Test 7: Anti-Hallucination")
    repl = setup_repl()
    results = []

    fictional = [
        "What is the capital of Narnia?",
        "Who discovered the Flux Capacitor?",
        "What is the formula of Unobtainium?",
    ]

    for query in fictional:
        answer = repl.process(query)
        # Should NOT contain a confident wrong answer
        ok = ('information' in answer.lower() or
              'don\'t' in answer.lower() or
              'unknown' in answer.lower() or
              answer == '')
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer[:50]}")

    return all(results)


def test_commands():
    """Test 8: REPL commands."""
    print("Test 8: REPL Commands")
    repl = setup_repl()
    results = []

    # /stats
    r = repl.handle_command('/stats')
    ok = 'Routing' in r
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: /stats works")

    # /knowledge
    r = repl.handle_command('/knowledge')
    ok = 'facts' in r.lower()
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: /knowledge works")

    # /tools
    r = repl.handle_command('/tools')
    ok = 'read_file' in r
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: /tools works")

    # /help
    r = repl.handle_command('/help')
    ok = 'Available' in r
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: /help works")

    return all(results)


def test_mixed_session():
    """Test 9: Mixed-mode session (knowledge + math + tools)."""
    print("Test 9: Mixed Session")
    repl = setup_repl()
    repl.tools.confirm_dangerous = False
    results = []

    session = [
        ("What is the capital of France?", lambda r: 'Paris' in r),
        ("Calculate 2 ** 10", lambda r: '1024' in r),
        ("Is 97 prime?", lambda r: 'prime' in r.lower()),
        ("What is the formula of Salt?", lambda r: 'NaCl' in r),
        ("GCD of 48 and 18", lambda r: '6' in r),
    ]

    for query, check in session:
        answer = repl.process(query)
        ok = check(answer)
        results.append(ok)
        print(f"  {'PASS' if ok else 'FAIL'}: {query} → {answer[:40]}")

    return all(results)


def test_conversation_flow():
    """Test 10: Natural conversation flow."""
    print("Test 10: Conversation Flow")
    repl = setup_repl()
    results = []

    # Conversation about France
    r1 = repl.process("What is the capital of France?")
    ok1 = 'Paris' in r1
    results.append(ok1)
    print(f"  {'PASS' if ok1 else 'FAIL'}: Q1 → {r1[:30]}")

    # Follow-up about Germany
    r2 = repl.process("What is the capital of Germany?")
    ok2 = 'Berlin' in r2
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: Q2 → {r2[:30]}")

    # Switch to math
    r3 = repl.process("Calculate 15 * 7")
    ok3 = '105' in r3
    results.append(ok3)
    print(f"  {'PASS' if ok3 else 'FAIL'}: Q3 → {r3[:30]}")

    # Back to knowledge
    r4 = repl.process("Who discovered Penicillin?")
    ok4 = 'Fleming' in r4
    results.append(ok4)
    print(f"  {'PASS' if ok4 else 'FAIL'}: Q4 → {r4[:30]}")

    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("FULL STACK INTEGRATION TEST")
    print("=" * 60)
    print()

    tests = [
        test_knowledge_qa,
        test_reasoning,
        test_math,
        test_chain,
        test_tools,
        test_memory,
        test_anti_hallucination,
        test_commands,
        test_mixed_session,
        test_conversation_flow,
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
    print(f"FULL STACK RESULTS: {passed}/{total} tests passed")
    if failed == 0:
        print("ALL TESTS PASSED — SYSTEM OPERATIONAL")
    print("=" * 60)
