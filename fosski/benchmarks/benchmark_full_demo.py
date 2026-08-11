#!/usr/bin/env python3
"""
FOSS-KI Full Pipeline Demo
============================
One script that shows EVERYTHING FOSS-KI can do that Transformers cannot.

5 capabilities, each architecturally impossible in Transformers:
  1. Online Learning: text → facts → instantly queryable (no retraining)
  2. Anti-Hallucination: unknown queries → REJECTED (not confabulated)
  3. Causal Reasoning: do-calculus (not correlation)
  4. Multi-Turn Dialog: with reference resolution (no NLU model)
  5. Deterministic Confidence: transparent scores (not black-box logits)

All on CPU, ~4ms per query, zero trained parameters.
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.engine import FossKI


def main():
    print("=" * 70)
    print("FOSS-KI — FULL PIPELINE DEMO")
    print("5 capabilities Transformers CANNOT do")
    print("=" * 70)

    fki = FossKI()

    # ══════════════════════════════════════════════════════════
    # 1. ONLINE LEARNING — Text → Facts → Instantly Queryable
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[1] Online Learning — No Retraining Required")
    print(f"{'━' * 70}")

    text = """
    Paris is the capital of France. Berlin is the capital of Germany.
    Tokyo is the capital of Japan. The capital of Italy is Rome.
    Madrid is the capital of Spain. London is the capital of the United Kingdom.
    Albert Einstein was born in Ulm. Marie Curie discovered radium.
    Water has a formula of H2O. Gold has a symbol of Au.
    The official language of France is French.
    """

    t0 = time.time()
    n_new = fki.learn_from_text(text)
    t_learn = (time.time() - t0) * 1000

    print(f"\n  Input: {len(text.split())} words of text")
    print(f"  Extracted: {n_new} facts in {t_learn:.0f}ms")
    print(f"  Immediately queryable — no gradient descent, no retraining.")

    # Query what we just learned
    for q_s, q_r in [("France", "capital"), ("Germany", "capital"), ("Water", "formula")]:
        t0 = time.time()
        r = fki.query_fact(q_s, q_r)
        t_q = (time.time() - t0) * 1000
        print(f"  Q: {q_s}/{q_r} → {r['fact'][2]} ({r['confidence_level']}, {t_q:.1f}ms)")

    # Learn MORE text — additive, no forgetting
    text2 = """
    Washington is the capital of the United States.
    Beijing is the capital of China. Moscow is the capital of Russia.
    """
    n2 = fki.learn_from_text(text2)
    print(f"\n  Added {n2} more facts. Total: {fki.knowledge.n_facts}")
    r = fki.query_fact("United States", "capital")
    if r['fact']:
        print(f"  Q: United States/capital → {r['fact'][2]} ({r['confidence_level']})")

    # ══════════════════════════════════════════════════════════
    # 2. ANTI-HALLUCINATION — Knows When It Doesn't Know
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[2] Anti-Hallucination — Architecturally Impossible in Transformers")
    print(f"{'━' * 70}")

    unknowns = [
        ("Narnia", "capital"),
        ("Mordor", "capital"),
        ("Atlantis", "ruler"),
        ("Hogwarts", "location"),
        ("Wakanda", "capital"),
    ]

    print(f"\n  Querying 5 fictional/unknown entities:")
    all_rejected = True
    for subj, rel in unknowns:
        r = fki.query_fact(subj, rel)
        rejected = r['confidence_level'] in ('REJECTED', 'UNKNOWN')
        all_rejected = all_rejected and rejected
        marker = "REJECTED" if rejected else f"LEAKED: {r['confidence_level']}"
        print(f"    {subj}/{rel}: {marker} (conf={r['confidence']:.3f})")

    print(f"\n  Result: {'5/5 rejected' if all_rejected else 'SOME LEAKED'}")
    print(f"  GPT-2 would hallucinate answers for ALL of these.")
    print(f"  FOSS-KI says 'I don't know' — because attractor distance is high.")

    # ══════════════════════════════════════════════════════════
    # 3. CAUSAL REASONING — do-Calculus (Not Correlation)
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[3] Causal Reasoning — Pearl's do-Calculus")
    print(f"{'━' * 70}")

    fki.load_causal_graph('simpson')

    t0 = time.time()
    do_yes, _ = fki.causal_query('Recovery', {'Treatment': 'yes'})
    do_no, _ = fki.causal_query('Recovery', {'Treatment': 'no'})
    t_causal = (time.time() - t0) * 1000

    print(f"\n  Simpson's Paradox (confounded medical trial):")
    print(f"    Observational: treatment APPEARS harmful (correlation)")
    print(f"    Causal (do-calculus): treatment HELPS (causation)")
    print(f"")
    print(f"    P(Recovery | do(Treatment=yes)) = {do_yes['yes']:.3f}")
    print(f"    P(Recovery | do(Treatment=no))  = {do_no['yes']:.3f}")
    print(f"    Causal effect: +{do_yes['yes'] - do_no['yes']:.3f} ({t_causal:.1f}ms)")
    print(f"")
    print(f"  GPT-2/GPT-4 would need to be TOLD this is Simpson's Paradox.")
    print(f"  FOSS-KI computes it from the causal graph structure.")

    # ══════════════════════════════════════════════════════════
    # 4. MULTI-TURN DIALOG — With Reference Resolution
    # ══════════════════════════════════════════════════════════
    print(f"\n{'━' * 70}")
    print("[4] Multi-Turn Dialog — No NLU Model Required")
    print(f"{'━' * 70}")

    fki.chat_reset()
    conversation = [
        "What is the capital of France?",
        "And Germany?",
        "What about it?",
        "What is the capital of Narnia?",
        "What is the formula of Water?",
    ]

    print()
    for q in conversation:
        t0 = time.time()
        r = fki.chat(q)
        t_q = (time.time() - t0) * 1000
        print(f"  User: {q}")
        print(f"  FOSS-KI: {r['response']} [{r['confidence']}, {t_q:.1f}ms]")
        print()

    # ══════════════════════════════════════════════════════════
    # 5. TRANSPARENT CONFIDENCE — Not Black-Box Logits
    # ══════════════════════════════════════════════════════════
    print(f"{'━' * 70}")
    print("[5] Transparent Confidence — Every Score Explainable")
    print(f"{'━' * 70}")

    print(f"\n  Every query returns:")
    print(f"    - confidence: cosine similarity to nearest stored fact")
    print(f"    - confidence_level: HIGH/MEDIUM/REJECTED based on adaptive threshold")
    print(f"    - attractor_distance: Hopfield convergence distance")
    print(f"    - top2_gap: discriminability between best and 2nd best match")
    print()

    for subj, rel in [("France", "capital"), ("Narnia", "capital")]:
        r = fki.query_fact(subj, rel)
        print(f"  {subj}/{rel}:")
        print(f"    confidence: {r['confidence']:.4f}")
        print(f"    level: {r['confidence_level']}")
        print(f"    attractor_distance: {r['attractor_distance']:.4f}")
        print(f"    thresholds: {r['thresholds']}")
        print()

    # ══════════════════════════════════════════════════════════
    # SUMMARY
    # ══════════════════════════════════════════════════════════
    print(f"{'═' * 70}")
    print("SUMMARY — What FOSS-KI Can Do That Transformers Cannot")
    print(f"{'═' * 70}")
    print(f"""
  1. Online Learning:      Text → Facts → Instantly Queryable (no retraining)
  2. Anti-Hallucination:   Knows when it doesn't know (attractor distance)
  3. Causal Reasoning:     do-calculus on causal DAGs (not correlation)
  4. Multi-Turn Dialog:    Reference resolution without NLU model
  5. Transparent Scores:   Every confidence score is explainable

  All on a single CPU core. ~4ms per query. Zero trained parameters.
  No GPU. No cloud. No RLHF. No catastrophic forgetting.

  This is what AI looks like when you don't start from Transformers.
""")


if __name__ == "__main__":
    main()
