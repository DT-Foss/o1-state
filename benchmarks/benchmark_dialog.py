#!/usr/bin/env python3
"""
Multi-Turn Dialog Benchmark
=============================
Tests the DialogSystem on multi-turn conversations with:
  1. Direct fact queries
  2. Reference resolution (pronouns, ellipsis)
  3. Anti-hallucination (fictional entities)
  4. Context carryover across turns
  5. Ambiguous queries

No transformer, no NLU, no intent classifier.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.dialog import DialogSystem


def test_direct_queries(ds):
    """Test 1: Direct fact queries."""
    print(f"\n  {'━' * 60}")
    print("  [1] Direct Fact Queries")
    print(f"  {'━' * 60}")

    queries = [
        ("What is the capital of France?", "Paris"),
        ("What is the capital of Germany?", "Berlin"),
        ("What is the symbol of Gold?", "Au"),
        ("Who created Python?", "Guido van Rossum"),
        ("What is the formula of Water?", "H2O"),
        ("What is the capital of Japan?", "Tokyo"),
    ]

    correct = 0
    for question, expected in queries:
        ds.reset()
        t0 = time.time()
        result = ds.turn(question)
        t_ms = (time.time() - t0) * 1000

        ok = result['answer'] == expected if result['answer'] else False
        correct += ok
        marker = "+" if ok else "x"
        print(f"    {marker} Q: {question}")
        print(f"      A: {result['response']} ({result['confidence']}, {t_ms:.1f}ms)")

    print(f"\n    Score: {correct}/{len(queries)}")
    return correct, len(queries)


def test_reference_resolution(ds):
    """Test 2: Multi-turn reference resolution."""
    print(f"\n  {'━' * 60}")
    print("  [2] Reference Resolution (Multi-Turn)")
    print(f"  {'━' * 60}")

    conversations = [
        # Conversation 1: Ellipsis with "And X?"
        {
            'name': 'Ellipsis',
            'turns': [
                ("What is the capital of France?", "Paris"),
                ("And what about Germany?", "Berlin"),
                ("And what about Japan?", "Tokyo"),
            ]
        },
        # Conversation 2: Same relation, different entity
        {
            'name': 'Sequential',
            'turns': [
                ("What is the symbol of Gold?", "Au"),
                ("What is the symbol of Iron?", "Fe"),
            ]
        },
        # Conversation 3: Mixed query types
        {
            'name': 'Mixed',
            'turns': [
                ("What is the capital of France?", "Paris"),
                ("Who created Linux?", "Linus Torvalds"),
                ("And what about Python?", "Guido van Rossum"),
            ]
        },
    ]

    total_correct = 0
    total_queries = 0

    for conv in conversations:
        ds.reset()
        print(f"\n    Conversation: {conv['name']}")
        for question, expected in conv['turns']:
            result = ds.turn(question)
            ok = result['answer'] == expected if result['answer'] else False
            total_correct += ok
            total_queries += 1
            marker = "+" if ok else "x"
            print(f"      {marker} Q: {question}")
            print(f"        A: {result['response']} (subj={result['subject']}, rel={result['relation']})")

    print(f"\n    Score: {total_correct}/{total_queries}")
    return total_correct, total_queries


def test_anti_hallucination(ds):
    """Test 3: Reject fictional entities."""
    print(f"\n  {'━' * 60}")
    print("  [3] Anti-Hallucination (should all REJECT)")
    print(f"  {'━' * 60}")

    queries = [
        "What is the capital of Narnia?",
        "What is the capital of Mordor?",
        "Who created Atlantis?",
        "What is the formula of Unobtanium?",
        "What is the symbol of Mythrilium?",
    ]

    rejected = 0
    for question in queries:
        ds.reset()
        result = ds.turn(question)
        is_rejected = result['confidence'] in ('REJECTED', 'UNKNOWN')
        rejected += is_rejected
        marker = "+" if is_rejected else "x"
        print(f"    {marker} Q: {question}")
        print(f"      R: {result['response']} ({result['confidence']})")

    print(f"\n    Rejected: {rejected}/{len(queries)}")
    return rejected, len(queries)


def test_context_carryover(ds):
    """Test 4: Context carries over across turns."""
    print(f"\n  {'━' * 60}")
    print("  [4] Context Carryover")
    print(f"  {'━' * 60}")

    ds.reset()

    # Build up context
    r1 = ds.turn("What is the capital of France?")
    assert r1['answer'] == "Paris"
    print(f"    Turn 1: {r1['response']}")

    # Ask about a fictional entity — should reject
    r2 = ds.turn("What is the capital of Narnia?")
    rejected = r2['confidence'] in ('REJECTED', 'UNKNOWN')
    print(f"    Turn 2: {r2['response']} (rejected={rejected})")

    # Go back to a real entity — should still work
    r3 = ds.turn("And what about Germany?")
    ok = r3['answer'] == "Berlin"
    print(f"    Turn 3: {r3['response']} (correct={ok})")

    # Check turn count
    print(f"    Total turns: {ds.n_turns}")
    print(f"    Entities seen: {ds.context_entities}")

    score = (1 if r1['answer'] == "Paris" else 0) + (1 if rejected else 0) + (1 if ok else 0)
    print(f"\n    Score: {score}/3")
    return score, 3


def test_parse_robustness(ds):
    """Test 5: Various question formats."""
    print(f"\n  {'━' * 60}")
    print("  [5] Parse Robustness (various question formats)")
    print(f"  {'━' * 60}")

    queries = [
        ("What is the capital of France?", "Paris"),
        ("capital of France", "Paris"),
        ("France capital", "Paris"),
        ("France's capital?", "Paris"),
        ("What is France's capital?", "Paris"),
        ("Who created Python?", "Guido van Rossum"),
    ]

    correct = 0
    for question, expected in queries:
        ds.reset()
        result = ds.turn(question)
        ok = result['answer'] == expected if result['answer'] else False
        correct += ok
        marker = "+" if ok else "x"
        print(f"    {marker} \"{question}\" → {result['answer']} "
              f"(subj={result['subject']}, rel={result['relation']})")

    print(f"\n    Score: {correct}/{len(queries)}")
    return correct, len(queries)


def main():
    print("=" * 70)
    print("MULTI-TURN DIALOG BENCHMARK")
    print("=" * 70)

    # Setup
    ds = DialogSystem(knowledge_dim=128)
    facts = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Japan", "capital", "Tokyo"),
        ("Italy", "capital", "Rome"),
        ("Spain", "capital", "Madrid"),
        ("UK", "capital", "London"),
        ("USA", "capital", "Washington"),
        ("China", "capital", "Beijing"),
        ("Water", "formula", "H2O"),
        ("Gold", "symbol", "Au"),
        ("Iron", "symbol", "Fe"),
        ("Python", "creator", "Guido van Rossum"),
        ("Linux", "creator", "Linus Torvalds"),
        ("Einstein", "theory", "relativity"),
    ]
    ds.load_knowledge(facts)
    print(f"\n  Loaded {len(facts)} facts into knowledge store.")

    t0 = time.time()

    scores = []
    scores.append(test_direct_queries(ds))
    scores.append(test_reference_resolution(ds))
    scores.append(test_anti_hallucination(ds))
    scores.append(test_context_carryover(ds))
    scores.append(test_parse_robustness(ds))

    total_time = (time.time() - t0) * 1000

    # Summary
    total_correct = sum(s[0] for s in scores)
    total_queries = sum(s[1] for s in scores)

    print(f"\n{'=' * 70}")
    print("DIALOG BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"""
  Total: {total_correct}/{total_queries} ({total_correct/total_queries:.0%})
  Time:  {total_time:.0f}ms total

  Components used:
    - QueryParser (rule-based, no ML)
    - EntityTracker (recency-weighted reference resolution)
    - KnowledgeStore (Modern Hopfield + anti-hallucination)
    - Confidence-gated responses (REJECTED → "I don't know")

  NO transformer, NO NLU model, NO intent classifier.
  Architecture provides alignment by construction:
    can only answer what it knows, refuses what it doesn't.
""")


if __name__ == "__main__":
    main()
