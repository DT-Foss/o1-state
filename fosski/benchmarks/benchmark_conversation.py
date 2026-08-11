"""
Benchmark: Conversation Memory (A3)
=====================================
Tests:
  1. Window management: turns enter/compress correctly
  2. Fact extraction from turns
  3. Context generation (summary + recent)
  4. Topic tracking across turns
  5. Memory search
  6. Persistence to knowledge store
  7. Long conversation simulation (50 turns, window=5)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.conversation import ConversationMemory
from core.knowledge import KnowledgeStore


def test_window():
    """Test 1: Window management."""
    print("Test 1: Window Management")
    mem = ConversationMemory(window_size=3)
    results = []

    # Add 5 turns
    mem.add_turn('user', 'What is the capital of France?')
    mem.add_turn('system', 'The capital of France is Paris.',
                 {'answer': 'Paris', 'subject': 'France', 'relation': 'capital'})
    mem.add_turn('user', 'What about Germany?')
    mem.add_turn('system', 'The capital of Germany is Berlin.',
                 {'answer': 'Berlin', 'subject': 'Germany', 'relation': 'capital'})
    mem.add_turn('user', 'And Japan?')

    ok1 = len(mem.window) == 3
    results.append(ok1)
    print(f"  {'PASS' if ok1 else 'FAIL'}: Window size = {len(mem.window)} (expected 3)")

    ok2 = mem.total_turns == 5
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: Total turns = {mem.total_turns}")

    ok3 = mem.compressions >= 1
    results.append(ok3)
    print(f"  {'PASS' if ok3 else 'FAIL'}: Compressions = {mem.compressions}")

    return all(results)


def test_fact_extraction():
    """Test 2: Facts extracted from compressed turns."""
    print("Test 2: Fact Extraction")
    mem = ConversationMemory(window_size=2)
    results = []

    # Add turns that will get compressed
    mem.add_turn('system', 'The capital of France is Paris.',
                 {'answer': 'Paris', 'subject': 'France', 'relation': 'capital'})
    mem.add_turn('system', 'The formula of Water is H2O.',
                 {'answer': 'H2O', 'subject': 'Water', 'relation': 'formula'})
    # These two push the first two out
    mem.add_turn('user', 'New question 1')
    mem.add_turn('user', 'New question 2')

    ok = len(mem.summary_facts) >= 2
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: {len(mem.summary_facts)} facts extracted")

    # Check that France/capital/Paris was extracted
    has_france = any(s == 'France' and r == 'capital' and o == 'Paris'
                     for s, r, o in mem.summary_facts)
    results.append(has_france)
    print(f"  {'PASS' if has_france else 'FAIL'}: France/capital/Paris in summary")

    return all(results)


def test_context_generation():
    """Test 3: Context string generation."""
    print("Test 3: Context Generation")
    mem = ConversationMemory(window_size=2)
    results = []

    mem.add_turn('system', 'The capital of France is Paris.',
                 {'answer': 'Paris', 'subject': 'France', 'relation': 'capital'})
    mem.add_turn('user', 'What about Germany?')
    mem.add_turn('system', 'Berlin is the capital.',
                 {'answer': 'Berlin', 'subject': 'Germany', 'relation': 'capital'})

    ctx = mem.get_context()
    # Should have summary section + recent turns
    has_known = 'Known:' in ctx or 'France' in ctx
    has_recent = 'Germany' in ctx
    results.append(has_known)
    results.append(has_recent)
    print(f"  {'PASS' if has_known else 'FAIL'}: Summary contains compressed facts")
    print(f"  {'PASS' if has_recent else 'FAIL'}: Recent turns preserved")
    print(f"  Context:\n    {ctx[:200]}")

    return all(results)


def test_topic_tracking():
    """Test 4: Topic frequency tracking."""
    print("Test 4: Topic Tracking")
    mem = ConversationMemory(window_size=10)
    results = []

    mem.add_turn('user', 'Tell me about France')
    mem.add_turn('user', 'What about France and Germany?')
    mem.add_turn('user', 'More about France')

    topics = mem.get_topic_frequency()
    ok = topics.get('France', 0) == 3
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: France mentioned {topics.get('France', 0)} times (expected 3)")

    ok2 = topics.get('Germany', 0) == 1
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: Germany mentioned {topics.get('Germany', 0)} times (expected 1)")

    recent = mem.get_recent_topics(3)
    ok3 = recent[0] == 'France'
    results.append(ok3)
    print(f"  {'PASS' if ok3 else 'FAIL'}: Most recent topic = {recent[0]}")

    return all(results)


def test_memory_search():
    """Test 5: Searching conversation memory."""
    print("Test 5: Memory Search")
    mem = ConversationMemory(window_size=2)
    results = []

    # Add enough turns to force compression
    mem.add_turn('system', 'Paris is the capital of France.',
                 {'answer': 'Paris', 'subject': 'France', 'relation': 'capital'})
    mem.add_turn('system', 'H2O is the formula for water.',
                 {'answer': 'H2O', 'subject': 'Water', 'relation': 'formula'})
    mem.add_turn('user', 'Tell me about Germany')
    mem.add_turn('system', 'Berlin is the capital of Germany.')

    # Search for France (should be in summary)
    results_france = mem.search_memory('France')
    ok = len(results_france) > 0
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: Found {len(results_france)} results for 'France'")

    # Search for Germany (should be in window)
    results_germany = mem.search_memory('Germany')
    ok2 = len(results_germany) > 0
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: Found {len(results_germany)} results for 'Germany'")

    return all(results)


def test_persistence():
    """Test 6: Persist summary facts to knowledge store."""
    print("Test 6: Knowledge Persistence")
    ks = KnowledgeStore(dim=64)
    mem = ConversationMemory(window_size=2, knowledge_store=ks)
    results = []

    mem.add_turn('system', 'The capital of France is Paris.',
                 {'answer': 'Paris', 'subject': 'France', 'relation': 'capital'})
    mem.add_turn('system', 'Water formula is H2O.',
                 {'answer': 'H2O', 'subject': 'Water', 'relation': 'formula'})
    mem.add_turn('user', 'Filler 1')
    mem.add_turn('user', 'Filler 2')

    # Now persist to KS
    stored = mem.persist_to_knowledge()
    ok = stored >= 2
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}: {stored} facts persisted to KnowledgeStore")

    # Verify they're queryable
    r = ks.query('France', 'capital')
    ok2 = r['fact'] is not None and r['fact'][2] == 'Paris'
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: France/capital → {r['fact'][2] if r['fact'] else '?'}")

    return all(results)


def test_long_conversation():
    """Test 7: 50-turn conversation simulation."""
    print("Test 7: Long Conversation (50 turns)")
    mem = ConversationMemory(window_size=5)
    results = []

    countries = [
        ('France', 'Paris'), ('Germany', 'Berlin'), ('Japan', 'Tokyo'),
        ('Italy', 'Rome'), ('Spain', 'Madrid'), ('Brazil', 'Brasilia'),
        ('Canada', 'Ottawa'), ('Australia', 'Canberra'), ('Egypt', 'Cairo'),
        ('India', 'New Delhi'), ('China', 'Beijing'), ('Russia', 'Moscow'),
        ('Mexico', 'Mexico City'), ('Sweden', 'Stockholm'),
        ('Norway', 'Oslo'), ('Finland', 'Helsinki'),
    ]

    for i, (country, capital) in enumerate(countries):
        mem.add_turn('user', f'What is the capital of {country}?')
        mem.add_turn('system', f'The capital of {country} is {capital}.',
                     {'answer': capital, 'subject': country, 'relation': 'capital'})
        # Ask follow-up sometimes
        if i % 3 == 0:
            mem.add_turn('user', f'Tell me more about {country}')

    stats = mem.get_summary()
    print(f"  Total turns: {stats['total_turns']}")
    print(f"  Window: {stats['window_turns']}")
    print(f"  Summary facts: {stats['summary_facts']}")
    print(f"  Compressions: {stats['compressions']}")
    print(f"  Unique topics: {stats['unique_topics']}")

    # Window should be exactly 5
    ok1 = stats['window_turns'] == 5
    results.append(ok1)
    print(f"  {'PASS' if ok1 else 'FAIL'}: Window = {stats['window_turns']} (expected 5)")

    # Should have compressed facts
    ok2 = stats['summary_facts'] > 5
    results.append(ok2)
    print(f"  {'PASS' if ok2 else 'FAIL'}: {stats['summary_facts']} summary facts")

    # Context should not be huge
    ctx = mem.get_context()
    ok3 = len(ctx) < 5000
    results.append(ok3)
    print(f"  {'PASS' if ok3 else 'FAIL'}: Context length = {len(ctx)} chars")

    # Early topics should still be findable
    results_france = mem.search_memory('France')
    ok4 = len(results_france) > 0
    results.append(ok4)
    print(f"  {'PASS' if ok4 else 'FAIL'}: France still findable after 50 turns")

    return all(results)


if __name__ == '__main__':
    print("=" * 60)
    print("CONVERSATION MEMORY BENCHMARK (A3)")
    print("=" * 60)
    print()

    tests = [
        test_window,
        test_fact_extraction,
        test_context_generation,
        test_topic_tracking,
        test_memory_search,
        test_persistence,
        test_long_conversation,
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
