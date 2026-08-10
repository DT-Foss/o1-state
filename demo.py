#!/usr/bin/env python3
"""
FOSS-KI Demo — Shows the full pipeline in action.

Run: python3 demo.py
"""
import time
import sys

sys.path.insert(0, '.')
from core.dialog import DialogSystem
from core.composer import ResponseComposer
from core.formulierer import Formulierer
from core.metacognition import MetacognitionEngine
from core.wikidata import WikidataImporter


def header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def ask(ds, question):
    t0 = time.time()
    r = ds.turn(question)
    ms = (time.time() - t0) * 1000
    print(f"  Q: {question}")
    print(f"  A: {r['response']}")
    print(f"     [{r['confidence']}, {r['source']}, {ms:.0f}ms]")
    print()
    return r


def main():
    print("FOSS-KI — Transformer-Free AI Demo")
    print("No GPU. No backprop. No transformer. CPU only.")
    print("Architecture: PPM + Hopfield + Constraint Solver")

    # ── Phase 1: Bootstrap ──
    header("Phase 1: Knowledge Bootstrap (521 facts, 120 entities)")
    ds = DialogSystem(knowledge_dim=128)
    wi = WikidataImporter(ds.knowledge)
    t0 = time.time()
    n = wi.import_countries_simple()
    print(f"  Loaded {n} facts in {time.time()-t0:.1f}s")

    # Enable full pipeline
    meta = MetacognitionEngine(ds.knowledge)
    ds.enable_metacognition(meta)
    ds.enable_composer(ResponseComposer(knowledge_store=ds.knowledge))
    ds.enable_formulierer(Formulierer())

    # ── Phase 2: Basic Q&A ──
    header("Phase 2: Question Answering")
    ask(ds, "What is the capital of France?")
    ask(ds, "Who created Python?")
    ask(ds, "Where is Japan?")

    # ── Phase 3: Multi-fact composition ──
    header("Phase 3: Multi-Fact Composition (Composer + Formulierer)")
    ask(ds, "Tell me about Switzerland")
    ask(ds, "Tell me about Alan Turing")

    # ── Phase 4: Anti-Hallucination ──
    header("Phase 4: Anti-Hallucination (rejects unknown entities)")
    ask(ds, "What is the capital of Narnia?")
    ask(ds, "Tell me about Hogwarts")

    # ── Phase 5: Multi-turn conversation ──
    header("Phase 5: Multi-Turn Dialogue (reference resolution)")
    ask(ds, "What is the capital of Germany?")
    ask(ds, "And what about France?")
    ask(ds, "And Japan?")

    # ── Phase 6: Compare ──
    header("Phase 6: Entity Comparison")
    ask(ds, "Compare France and Germany")

    # ── Phase 7: Learning from text ──
    header("Phase 7: Learning from Text (Metacognition Gap-Fill)")
    text = ("The Eiffel Tower is located in Paris. It was designed by "
            "Gustave Eiffel and completed in 1889. The tower is 330 meters tall.")
    print(f"  Feeding text: \"{text[:80]}...\"")
    n_new = meta.ingest(text)
    print(f"  Extracted {n_new} new facts")
    for s, r, o in ds.knowledge.facts[-n_new:]:
        print(f"    ({s}, {r}, {o})")
    print()
    ask(ds, "Tell me about the Eiffel Tower")

    # ── Phase 8: Self-Test ──
    header("Phase 8: Self-Test (auto-generated benchmark)")
    test_queries = []
    for s, r, o in ds.knowledge.facts[:50]:
        if r in ('capital', 'language', 'creator', 'inventor', 'discoverer'):
            test_queries.append({'subject': s, 'relation': r, 'expected': o})

    t0 = time.time()
    result = meta.selftest(test_queries)
    print(f"  {result['correct']}/{result['total']} correct "
          f"({result['accuracy']:.0%}) in {time.time()-t0:.1f}s")

    # ── Summary ──
    header("Summary")
    print(f"  Total facts: {ds.knowledge.n_facts}")
    print(f"  Conversation turns: {ds.n_turns}")
    print(f"  Components: PPM (language) + Hopfield (memory) + "
          f"Rule-based parser + Composer + Formulierer + Metacognition")
    print(f"  GPU required: None")
    print(f"  External API calls: Zero")
    print()


if __name__ == '__main__':
    main()
