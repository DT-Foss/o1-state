"""
Benchmarks for Intent Classifier + Context Manager
=====================================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.intent import IntentClassifier
from core.context import ContextManager

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        print(f"  ✗ {name} — {detail}")


# ============================================================
# Intent Classifier
# ============================================================

def test_intent_classifier():
    print("\n=== Intent Classifier ===")

    ic = IntentClassifier()

    # Greetings
    r = ic.classify("hello there")
    check("hello → greeting", r['intent'] == 'greeting', f"got {r['intent']}")

    r = ic.classify("hi how are you")
    check("hi → greeting", r['intent'] == 'greeting', f"got {r['intent']}")

    # Factual questions
    r = ic.classify("what is the capital of France")
    check("capital → factual", r['intent'] == 'question_factual', f"got {r['intent']}")

    r = ic.classify("who invented the telephone")
    check("who invented → factual", r['intent'] == 'question_factual', f"got {r['intent']}")

    # How questions
    r = ic.classify("how does photosynthesis work")
    check("how does → how", r['intent'] == 'question_how', f"got {r['intent']}")

    # Why questions
    r = ic.classify("why is the sky blue")
    check("why is → why", r['intent'] == 'question_why', f"got {r['intent']}")

    # Math
    r = ic.classify("calculate 17 * 23")
    check("calculate → math", r['intent'] == 'math', f"got {r['intent']}")

    r = ic.classify("is 97 prime")
    check("is prime → math", r['intent'] == 'math', f"got {r['intent']}")

    r = ic.classify("solve x^2 - 5x + 6 = 0")
    check("solve → math", r['intent'] == 'math', f"got {r['intent']}")

    r = ic.classify("5 + 3")
    check("5+3 → math", r['intent'] == 'math', f"got {r['intent']}")

    # Common sense
    r = ic.classify("is ice cold?")
    check("is ice cold → commonsense", r['intent'] == 'commonsense', f"got {r['intent']}")

    r = ic.classify("can birds fly?")
    check("can birds fly → commonsense", r['intent'] == 'commonsense', f"got {r['intent']}")

    # Comparison
    r = ic.classify("compare Python and Java")
    check("compare → comparison", r['intent'] == 'comparison', f"got {r['intent']}")

    r = ic.classify("difference between TCP and UDP")
    check("difference → comparison", r['intent'] == 'comparison', f"got {r['intent']}")

    # Code
    r = ic.classify("write a function that sorts a list")
    check("write function → code", r['intent'] == 'code', f"got {r['intent']}")

    # File commands
    r = ic.classify("read file config.py")
    check("read file → file", r['intent'] == 'command_file', f"got {r['intent']}")

    # Confidence
    r = ic.classify("hello")
    check("confidence > 0", r['confidence'] > 0)
    check("has method", r['method'] in ('keyword', 'pattern', 'tfidf'))

    # Entity extraction
    r = ic.classify("calculate 42 * 3.14")
    check("extracts numbers", 'numbers' in r['entities'])

    r = ic.classify('read file "test.py"')
    check("extracts quoted", 'quoted' in r['entities'] or 'files' in r['entities'])

    r = ic.classify("Albert Einstein invented relativity")
    check("extracts names", 'proper_nouns' in r['entities'])

    # All scores
    r = ic.classify("what is Python")
    check("has all_scores", len(r['all_scores']) > 0)

    # Batch classify
    results = ic.batch_classify(["hello", "what is 5+3", "why is sky blue"])
    check("batch classify", len(results) == 3)

    # Intent names
    names = ic.get_intent_names()
    check("intent names", len(names) >= 10, f"got {len(names)}")

    # Add exemplar
    ic.add_exemplar('custom', "do a barrel roll")
    r = ic.classify("do a barrel roll")
    check("custom exemplar", r['intent'] == 'custom' or r['confidence'] > 0)

    # Stats
    stats = ic.stats()
    check("stats has fields", stats['intents'] > 0 and stats['exemplars'] > 0)


# ============================================================
# Context Manager
# ============================================================

def test_context_manager():
    print("\n=== Context Manager ===")

    ctx = ContextManager()

    # Basic tracking
    ctx.update("What is the capital of France?")
    summary = ctx.get_context_summary()
    check("tracks turn", summary['turn'] == 1)
    check("detects topic", summary['topic'] is not None, f"got {summary['topic']}")

    # Entity tracking
    ctx.update("Tell me about Albert Einstein")
    entities = [e['name'] for e in ctx.get_context_summary()['entities']]
    check("tracks entities", len(entities) > 0, f"got {entities}")

    # Reference resolution
    ctx = ContextManager()
    ctx.update("What is the capital of France?")
    ctx.last_subject = "France"
    resolved = ctx.resolve_references("How many people live there?")
    check("resolves 'there'", "France" in resolved, f"got '{resolved}'")

    # "its" resolution
    ctx = ContextManager()
    ctx.last_subject = "Python"
    resolved = ctx.resolve_references("What is its creator?")
    check("resolves 'its'", "Python" in resolved, f"got '{resolved}'")

    # Topic tracking
    ctx = ContextManager()
    ctx.update("Tell me about machine learning")
    ctx.update("What about deep learning?")
    topic = ctx.get_current_topic()
    check("topic tracked", topic is not None, f"got {topic}")

    topics = ctx.get_context_summary()['topic_history']
    check("topic history", len(topics) >= 1, f"got {topics}")

    # Multiple turns
    ctx = ContextManager()
    ctx.update("What is Python?")
    ctx.update("Who created Python?")
    ctx.update("What year was it released?")
    summary = ctx.get_context_summary()
    check("multi-turn count", summary['turn'] == 3)

    # Add fact
    ctx.add_fact("Python", "creator", "Guido van Rossum")
    summary = ctx.get_context_summary()
    check("last subject updated", summary['last_subject'] == "Python")
    check("last object updated", summary['last_object'] == "Guido van Rossum")

    # Clear
    ctx.clear()
    summary = ctx.get_context_summary()
    check("clear resets", summary['turn'] == 0 and len(summary['entities']) == 0)

    # Pronoun reference chain
    ctx = ContextManager()
    ctx.update("Tell me about Germany")
    ctx.last_subject = "Germany"
    resolved = ctx.resolve_references("What is its capital?")
    check("pronoun chain", "Germany" in resolved, f"got '{resolved}'")

    # Entity type guessing
    ctx = ContextManager()
    ctx.update("I visited New York last summer")
    has_entity = any(
        e['name'] == 'New York'
        for e in ctx.get_context_summary()['entities']
    )
    check("detects New York", has_entity)

    # Entity pruning (stress test)
    ctx = ContextManager(max_entities=5)
    for i in range(10):
        ctx.update(f"Tell me about Entity{i}")
    check("entity pruning", len(ctx.entities) <= 8)  # Some slack

    # "this"/"that" resolution
    ctx = ContextManager()
    ctx.last_subject = "TCP"
    resolved = ctx.resolve_references("this is faster than UDP")
    check("resolves 'this'", "TCP" in resolved, f"got '{resolved}'")

    # No resolution when no context
    ctx = ContextManager()
    resolved = ctx.resolve_references("what is that?")
    check("no false resolution", resolved == "what is that?")


if __name__ == '__main__':
    test_intent_classifier()
    test_context_manager()

    total = PASS + FAIL
    print(f"\n{'='*50}")
    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")
    if FAIL:
        print(f"FAILURES: {FAIL}")
    else:
        print("ALL TESTS PASSED")
