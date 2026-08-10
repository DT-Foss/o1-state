#!/usr/bin/env python3
"""
Benchmark: Language Core — PPM Hierarchical vs Baselines
==========================================================
Tests:
1. Character-level PPM: bits/char (target: <2.0 for English)
2. Word-level PPM: perplexity comparison
3. Hierarchical model: combined prediction quality
4. Generation quality at different temperatures
5. Context length: how far back does it use?
6. Scaling: performance vs training data size
7. Online learning: adaptation to new domains
8. Comparison: PPM orders 1-12
"""

import sys
import os
import time
import math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.language import PPMModel, HierarchicalLanguageModel


# ═══════════════════════════════════════════════════════════════
# Test Corpora
# ═══════════════════════════════════════════════════════════════

def generate_english_corpus(n_sentences=500, seed=42):
    """Generate structured English text with patterns."""
    rng = np.random.RandomState(seed)

    subjects = ["the cat", "the dog", "a bird", "the fish", "a man",
                "the woman", "the child", "a student", "the teacher",
                "a doctor", "the engineer", "a scientist", "the artist",
                "a writer", "the musician"]
    verbs_intrans = ["runs", "walks", "jumps", "sleeps", "laughs",
                     "sings", "dances", "thinks", "works", "plays"]
    verbs_trans = ["sees", "finds", "takes", "makes", "gives",
                   "reads", "writes", "builds", "draws", "loves"]
    adverbs = ["quickly", "slowly", "quietly", "loudly", "happily",
               "sadly", "carefully", "eagerly", "gently", "boldly"]
    adjectives = ["big", "small", "old", "young", "bright",
                  "dark", "fast", "slow", "happy", "sad"]
    objects = ["the book", "a house", "the car", "a song", "the food",
               "a picture", "the garden", "a letter", "the bridge",
               "a machine"]
    preps = ["in the park", "at home", "near the river", "on the hill",
             "by the lake", "through the forest", "across the field",
             "under the bridge", "along the road", "behind the house",
             "with great care", "without hesitation", "during the night",
             "before the dawn", "after the storm"]
    conjunctions = ["and", "but", "so", "because", "although",
                    "while", "when", "if", "since", "unless"]

    sentences = []
    for _ in range(n_sentences):
        pattern = rng.randint(8)
        if pattern == 0:
            s = f"{rng.choice(subjects)} {rng.choice(verbs_intrans)}"
        elif pattern == 1:
            s = f"{rng.choice(subjects)} {rng.choice(verbs_intrans)} {rng.choice(adverbs)}"
        elif pattern == 2:
            s = f"{rng.choice(subjects)} {rng.choice(verbs_trans)} {rng.choice(objects)}"
        elif pattern == 3:
            s = (f"{rng.choice(subjects)} {rng.choice(adverbs)} "
                 f"{rng.choice(verbs_trans)} {rng.choice(objects)}")
        elif pattern == 4:
            s = (f"{rng.choice(subjects)} {rng.choice(verbs_intrans)} "
                 f"{rng.choice(preps)}")
        elif pattern == 5:
            s = (f"the {rng.choice(adjectives)} {rng.choice(subjects).split()[-1]} "
                 f"{rng.choice(verbs_trans)} {rng.choice(objects)} {rng.choice(preps)}")
        elif pattern == 6:
            s1 = f"{rng.choice(subjects)} {rng.choice(verbs_intrans)}"
            s2 = f"{rng.choice(subjects)} {rng.choice(verbs_trans)} {rng.choice(objects)}"
            s = f"{s1} {rng.choice(conjunctions)} {s2}"
        else:
            s = (f"{rng.choice(preps)} , {rng.choice(subjects)} "
                 f"{rng.choice(adverbs)} {rng.choice(verbs_trans)} {rng.choice(objects)}")
        sentences.append(s)

    return " . ".join(sentences) + " ."


def generate_technical_corpus(n_sentences=200, seed=42):
    """Generate technical/scientific text."""
    rng = np.random.RandomState(seed)

    subjects = ["the algorithm", "the system", "the network", "the model",
                "the function", "the matrix", "the graph", "the process"]
    verbs = ["computes", "optimizes", "converges", "transforms",
             "propagates", "minimizes", "maximizes", "estimates"]
    objects = ["the gradient", "the loss function", "the parameters",
               "the distribution", "the eigenvalues", "the topology",
               "the spectral gap", "the convergence rate"]
    qualifiers = ["efficiently", "rapidly", "asymptotically",
                  "probabilistically", "deterministically", "iteratively"]

    sentences = []
    for _ in range(n_sentences):
        s = f"{rng.choice(subjects)} {rng.choice(verbs)} {rng.choice(objects)} {rng.choice(qualifiers)}"
        sentences.append(s)

    return " . ".join(sentences) + " ."


# ═══════════════════════════════════════════════════════════════
# Main Benchmark
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("FOSS-KI LANGUAGE CORE BENCHMARK")
    print("=" * 80)

    corpus = generate_english_corpus(1000, seed=42)
    split = int(len(corpus) * 0.8)
    train_text = corpus[:split]
    test_text = corpus[split:]

    train_words = train_text.split()
    test_words = test_text.split()

    print(f"\n  Corpus: {len(corpus)} chars, {len(train_words)} train words, {len(test_words)} test words")
    print(f"  Vocabulary: {len(set(train_words))} unique words")

    # ── TEST 1: Character PPM — bits/char ─────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 1: Character-Level PPM — Bits per Character")
    print(f"{'━' * 80}")
    print("  Target: <2.0 bits/char for English (Shannon limit ~1.0)")
    print("  GPT-2: ~0.97 bits/char on WikiText")

    print(f"\n  {'Order':<8} {'Train bpc':>10} {'Test bpc':>10} {'Train time':>12} {'Context used':>14}")
    print(f"  {'─' * 58}")

    train_chars = list(train_text)
    test_chars = list(test_text[:2000])  # Limit test for speed

    for order in [1, 2, 3, 4, 6, 8, 10, 12]:
        ppm = PPMModel(max_order=order)

        t0 = time.time()
        ppm.train(train_chars)
        train_time = time.time() - t0

        # Train bpc
        train_bpc = 0
        n_train = 0
        for i in range(1, min(len(train_chars), 2000)):
            context = train_chars[:i]
            probs = ppm.predict(context)
            p = probs.get(train_chars[i], 1e-10)
            train_bpc -= math.log2(max(p, 1e-10))
            n_train += 1
        train_bpc /= max(n_train, 1)

        # Test bpc
        test_bpc = 0
        n_test = 0
        for i in range(1, len(test_chars)):
            context = test_chars[:i]
            probs = ppm.predict(context)
            p = probs.get(test_chars[i], 1e-10)
            test_bpc -= math.log2(max(p, 1e-10))
            n_test += 1
        test_bpc /= max(n_test, 1)

        # How much context is actually used
        effective_ctx = min(order, 12)

        print(f"  {order:<8} {train_bpc:>10.3f} {test_bpc:>10.3f} {train_time:>12.3f}s {effective_ctx:>14}")

    # ── TEST 2: Word-Level PPM ────────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 2: Word-Level PPM — Perplexity")
    print(f"{'━' * 80}")
    print("  T387 Bigram baseline: PPL ≈ 7.3")
    print("  Target: PPL < 7.3 (beat the bigram)")

    print(f"\n  {'Order':<8} {'Test PPL':>10} {'Vocab coverage':>16}")
    print(f"  {'─' * 38}")

    for order in [1, 2, 3, 4, 5, 6]:
        ppm_w = PPMModel(max_order=order)
        ppm_w.train(train_words)

        # Test perplexity
        total_lp = 0
        n = 0
        hits = 0
        for i in range(1, len(test_words)):
            context = test_words[:i]
            probs = ppm_w.predict(context)
            p = probs.get(test_words[i], 0)
            if p > 0:
                total_lp += math.log2(p)
                hits += 1
            else:
                total_lp += math.log2(1e-10)
            n += 1

        ppl = 2.0 ** (-total_lp / max(n, 1))
        coverage = hits / max(n, 1)
        print(f"  {order:<8} {ppl:>10.2f} {coverage:>16.1%}")

    # ── TEST 3: Hierarchical Model ────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 3: Hierarchical Language Model (Char + Word + Phrase)")
    print(f"{'━' * 80}")

    hlm = HierarchicalLanguageModel(max_orders=[10, 5, 3])

    t0 = time.time()
    hlm.train(train_text)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    # Character-level metric
    bpc = hlm.bits_per_char(test_text[:2000])
    print(f"  Bits/char (char level): {bpc:.3f}")

    # Level weights
    print(f"  Level weights: {dict(zip(hlm.level_names, hlm.weights))}")

    # Generation samples
    print(f"\n  Generation samples (seed='the cat'):")
    for temp in [0.5, 0.8, 1.0, 1.5]:
        gen = hlm.generate("the cat", n_words=15, temperature=temp,
                           rng=np.random.RandomState(42))
        print(f"    T={temp}: the cat {gen}")

    # ── TEST 4: Context Length Effect ─────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 4: Context Length — How Far Back Does PPM Look?")
    print(f"{'━' * 80}")

    ppm_deep = PPMModel(max_order=12)
    ppm_deep.train(train_chars)

    # Test: restrict context and measure bpc
    print(f"\n  {'Context limit':<15} {'bpc':>8}")
    print(f"  {'─' * 25}")

    for ctx_limit in [1, 2, 3, 5, 8, 12, 20, 50, 100, 'full']:
        total_bits = 0
        n = 0
        for i in range(1, min(len(test_chars), 1000)):
            if ctx_limit == 'full':
                context = test_chars[:i]
            else:
                context = test_chars[max(0, i - ctx_limit):i]
            probs = ppm_deep.predict(context)
            p = probs.get(test_chars[i], 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
            n += 1
        bpc = total_bits / max(n, 1)
        print(f"  {str(ctx_limit):<15} {bpc:>8.3f}")

    # ── TEST 5: Scaling with Data ─────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 5: Scaling — Performance vs Training Data Size")
    print(f"{'━' * 80}")

    print(f"\n  {'Train sentences':<18} {'Train chars':>12} {'Test bpc':>10} {'Test PPL (word)':>16}")
    print(f"  {'─' * 60}")

    for n_sent in [50, 100, 200, 500, 1000, 2000]:
        corp = generate_english_corpus(n_sent, seed=42)
        sp = int(len(corp) * 0.8)
        tr = corp[:sp]
        te = corp[sp:sp + 2000]

        # Char PPM
        ppm_c = PPMModel(max_order=8)
        ppm_c.train(list(tr))

        bpc = 0
        te_chars = list(te)
        for i in range(1, len(te_chars)):
            p = ppm_c.predict(te_chars[:i])
            bpc -= math.log2(max(p.get(te_chars[i], 1e-10), 1e-10))
        bpc /= max(len(te_chars) - 1, 1)

        # Word PPM
        ppm_w = PPMModel(max_order=4)
        tr_words = tr.split()
        te_words = te.split()
        ppm_w.train(tr_words)

        total_lp = 0
        for i in range(1, len(te_words)):
            probs = ppm_w.predict(te_words[:i])
            total_lp += math.log2(max(probs.get(te_words[i], 1e-10), 1e-10))
        ppl = 2.0 ** (-total_lp / max(len(te_words) - 1, 1))

        print(f"  {n_sent:<18} {len(tr):>12} {bpc:>10.3f} {ppl:>16.2f}")

    # ── TEST 6: Online Learning ───────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 6: Online Learning — Adaptation to New Domain")
    print(f"{'━' * 80}")
    print("  Train on English, then adapt to technical text online.")

    # Train on general English
    ppm_online = PPMModel(max_order=8)
    ppm_online.train(list(train_text))

    # Test on technical text BEFORE adaptation
    tech_text = generate_technical_corpus(100, seed=99)
    tech_chars = list(tech_text[:1000])

    bpc_before = 0
    for i in range(1, len(tech_chars)):
        p = ppm_online.predict(tech_chars[:i])
        bpc_before -= math.log2(max(p.get(tech_chars[i], 1e-10), 1e-10))
    bpc_before /= max(len(tech_chars) - 1, 1)

    # Adapt online: feed technical text character by character
    # PPM naturally updates with each observation
    ppm_adapted = PPMModel(max_order=8)
    ppm_adapted.train(list(train_text))

    tech_text2 = generate_technical_corpus(200, seed=77)
    tech_chars2 = list(tech_text2[:2000])

    # Online adaptation: train and test simultaneously
    bpc_chunks = []
    chunk_size = 200
    for start in range(0, len(tech_chars2) - chunk_size, chunk_size):
        chunk = tech_chars2[start:start + chunk_size]

        # Test on this chunk
        bpc_chunk = 0
        for i in range(1, len(chunk)):
            context = tech_chars2[:start + i]
            p = ppm_adapted.predict(context)
            bpc_chunk -= math.log2(max(p.get(chunk[i], 1e-10), 1e-10))
        bpc_chunk /= max(len(chunk) - 1, 1)
        bpc_chunks.append(bpc_chunk)

        # Update model with this chunk (online learning)
        for i in range(len(chunk)):
            ppm_adapted.update(tech_chars2[:start + i], chunk[i])

    bpc_after = bpc_chunks[-1] if bpc_chunks else bpc_before

    print(f"\n  Before adaptation (technical text): {bpc_before:.3f} bpc")
    print(f"  After online adaptation:            {bpc_after:.3f} bpc")
    print(f"  Improvement: {(bpc_before - bpc_after) / bpc_before * 100:.1f}%")

    if bpc_chunks:
        print(f"\n  Adaptation curve (bpc per chunk of {chunk_size} chars):")
        for i, bpc in enumerate(bpc_chunks):
            bar = "█" * int(max(0, 40 - bpc * 10))
            print(f"    Chunk {i}: {bpc:.3f} {bar}")

    # ── TEST 7: Generation Comparison ─────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 7: Text Generation Quality")
    print(f"{'━' * 80}")

    hlm2 = HierarchicalLanguageModel(max_orders=[10, 5, 3])
    hlm2.train(train_text)

    seeds = ["the dog", "a student", "in the park"]
    for seed in seeds:
        print(f"\n  Seed: '{seed}'")
        for temp in [0.7, 1.0]:
            gen = hlm2.generate(seed, n_words=20, temperature=temp,
                               rng=np.random.RandomState(42))
            print(f"    T={temp}: {seed} {gen}")

    # ── TEST 8: Comparison Table ──────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 8: Final Comparison")
    print(f"{'━' * 80}")

    print(f"""
  ┌──────────────────────┬────────────┬──────────────┬─────────────┐
  │ Method               │ bpc (char) │ PPL (word)   │ Online?     │
  ├──────────────────────┼────────────┼──────────────┼─────────────┤
  │ Bigram (T387)        │ ~4.0       │ ~7.3         │ No          │
  │ PPM Order 4          │ see above  │ see above    │ YES         │
  │ PPM Order 8          │ see above  │ see above    │ YES         │
  │ Hierarchical (ours)  │ see above  │ see above    │ YES         │
  ├──────────────────────┼────────────┼──────────────┼─────────────┤
  │ GPT-2 Small (ref)    │ ~0.97      │ ~29.4 (Wiki) │ No          │
  │ PPM* (literature)    │ ~1.5       │ N/A          │ YES         │
  │ CTW (literature)     │ ~1.3       │ N/A          │ YES         │
  └──────────────────────┴────────────┴──────────────┴─────────────┘

  * Literature values on real English text (Brown corpus, etc.)
  Our values are on synthetic corpus — NOT directly comparable.

  KEY FINDING:
  PPM with variable-order context is FUNDAMENTALLY different
  from T387's bigram walks. It uses the LONGEST matching context
  with escape to shorter contexts. This is how compression
  algorithms (gzip, bzip2, zstd) work — and they achieve
  excellent language modeling.
""")


if __name__ == "__main__":
    main()
