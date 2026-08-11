#!/usr/bin/env python3
"""
T90 — Bernoulli-Shift Temperature for Generation
===================================================
Tests the ×2 mod 9 perturbation on probability RANKS.

Measures:
  1. Diversity (unique n-grams) as function of temperature
  2. Coherence (PPM-score of generated text)
  3. Repetition rate
  4. Comparison with standard softmax temperature
  5. Quality of the algebraic structure vs random perturbation

Hypothesis: Bernoulli-shift produces qualitatively different
diversity because the perturbation is algebraically structured
(orbit structure {1,2,4,8,7,5},{3,6},{9}) not random.
"""

import numpy as np
import sys, os, time, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.vortex import VortexLanguageModel
from core.language import PPMModel


def load_gutenberg(path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
    for marker in ['*** START OF THE PROJECT GUTENBERG', '*** START OF THIS PROJECT GUTENBERG']:
        idx = text.find(marker)
        if idx >= 0:
            text = text[text.index('\n', idx) + 1:]
            break
    for marker in ['*** END OF THE PROJECT GUTENBERG', '*** END OF THIS PROJECT GUTENBERG']:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
            break
    return text.strip().replace('\r\n', '\n')


def ngram_diversity(text, n=3):
    """Fraction of unique n-grams in text."""
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def repetition_rate(text, window=20):
    """Fraction of characters that appear in repeated windows."""
    if len(text) < window * 2:
        return 0.0
    windows = [text[i:i+window] for i in range(len(text) - window + 1)]
    counts = Counter(windows)
    repeated = sum(c - 1 for c in counts.values() if c > 1)
    return repeated / max(len(windows), 1)


def main():
    print("=" * 70)
    print("T90 — BERNOULLI-SHIFT TEMPERATURE BENCHMARK")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    alice_path = os.path.join(data_dir, 'alice.txt')

    if not os.path.exists(alice_path):
        print(f"  ERROR: {alice_path} not found")
        return

    alice = load_gutenberg(alice_path)
    train_text = alice[:30000]
    eval_text = alice[30000:32000]

    # ── Train Vortex model ──
    print("\n[1] Training Vortex Language Model...")
    t0 = time.time()
    vlm = VortexLanguageModel(order=6, bridge_strength=0.0, temperature=0.0)
    vlm.train(train_text)
    print(f"  Trained in {time.time() - t0:.1f}s")

    # ── Also train a plain PPM for comparison ──
    print("[2] Training baseline PPM...")
    ppm = PPMModel(max_order=6)
    ppm.train(list(train_text))

    # ── Test 1: Bernoulli-Shift at different temperatures ──
    print(f"\n{'━' * 70}")
    print("[3] Bernoulli-Shift Generation Quality")
    print(f"{'━' * 70}")

    temperatures = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    seed = "Alice was beginning to get very "

    print(f"\n  {'T':>4} {'3-gram%':>8} {'5-gram%':>8} {'Rep%':>6} {'BPC':>7} {'Sample':<50}")
    print(f"  {'─' * 85}")

    for temp in temperatures:
        vlm.temperature = temp
        rng = np.random.RandomState(42)

        # Generate 5 samples, measure statistics
        all_text = ""
        for _ in range(5):
            gen = vlm.generate(list(seed), 150, rng=rng)
            all_text += gen

        d3 = ngram_diversity(all_text, 3) * 100
        d5 = ngram_diversity(all_text, 5) * 100
        rep = repetition_rate(all_text) * 100

        # Coherence: BPC of generated text scored by trained PPM
        gen_chars = list(all_text[:500])
        total_bits = 0.0
        for i in range(1, len(gen_chars)):
            ctx = gen_chars[:i]
            p = ppm.predict(ctx)
            prob = max(p.get(gen_chars[i], 1e-10), 1e-10)
            total_bits -= math.log2(prob)
        bpc = total_bits / max(len(gen_chars) - 1, 1)

        # Show first sample
        sample = vlm.generate(list(seed), 60, rng=np.random.RandomState(42))
        print(f"  {temp:>4.1f} {d3:>7.1f}% {d5:>7.1f}% {rep:>5.1f}% {bpc:>6.2f}  {sample[:50]}")

    # ── Test 2: Compare with standard softmax temperature ──
    print(f"\n{'━' * 70}")
    print("[4] Bernoulli-Shift vs Softmax Temperature")
    print(f"{'━' * 70}")

    print(f"\n  {'Method':<20} {'T':>4} {'3-gram%':>8} {'5-gram%':>8} {'BPC':>7}")
    print(f"  {'─' * 55}")

    for temp in [0.3, 0.5, 0.7]:
        # Bernoulli
        vlm.temperature = temp
        rng = np.random.RandomState(42)
        all_bernoulli = ""
        for _ in range(5):
            all_bernoulli += vlm.generate(list(seed), 150, rng=rng)

        d3_b = ngram_diversity(all_bernoulli, 3) * 100
        d5_b = ngram_diversity(all_bernoulli, 5) * 100

        gen_chars = list(all_bernoulli[:500])
        total_bits = 0.0
        for i in range(1, len(gen_chars)):
            ctx = gen_chars[:i]
            p = ppm.predict(ctx)
            prob = max(p.get(gen_chars[i], 1e-10), 1e-10)
            total_bits -= math.log2(prob)
        bpc_b = total_bits / max(len(gen_chars) - 1, 1)

        # Softmax (using raw PPM with temperature)
        rng = np.random.RandomState(42)
        all_softmax = ""
        for _ in range(5):
            gen_tokens = ppm.generate(list(seed), 150, temperature=temp, rng=rng)
            all_softmax += ''.join(gen_tokens)

        d3_s = ngram_diversity(all_softmax, 3) * 100
        d5_s = ngram_diversity(all_softmax, 5) * 100

        gen_chars = list(all_softmax[:500])
        total_bits = 0.0
        for i in range(1, len(gen_chars)):
            ctx = gen_chars[:i]
            p = ppm.predict(ctx)
            prob = max(p.get(gen_chars[i], 1e-10), 1e-10)
            total_bits -= math.log2(prob)
        bpc_s = total_bits / max(len(gen_chars) - 1, 1)

        print(f"  {'Bernoulli ×2mod9':<20} {temp:>4.1f} {d3_b:>7.1f}% {d5_b:>7.1f}% {bpc_b:>6.2f}")
        print(f"  {'Softmax':<20} {temp:>4.1f} {d3_s:>7.1f}% {d5_s:>7.1f}% {bpc_s:>6.2f}")
        print()

    # ── Test 3: Orbit structure analysis ──
    print(f"{'━' * 70}")
    print("[5] ×2 mod 9 Orbit Structure")
    print(f"{'━' * 70}")

    print("\n  Orbits of ×2 mod 9:")
    visited = set()
    orbits = []
    for start in range(1, 10):
        if start in visited:
            continue
        orbit = []
        x = start
        while x not in visited:
            visited.add(x)
            orbit.append(x)
            x = (2 * x) % 9 or 9
        orbits.append(orbit)
        print(f"    {' → '.join(map(str, orbit))} → {orbit[0]} (length {len(orbit)})")

    print(f"\n  Main orbit {{1,2,4,8,7,5}} has period 6 — visits every rank")
    print(f"  Fixed point {{9}} — rank 9 stays at 9 (bottom stays bottom)")
    print(f"  Short orbit {{3,6}} — alternates between mid-ranks")
    print(f"\n  This means T>0 redistributes mass from rank 1 → rank 2,")
    print(f"  rank 2 → rank 4, rank 4 → rank 8, etc. — structured exploration")
    print(f"  that visits alternatives in a SPECIFIC algebraic order,")
    print(f"  NOT randomly like softmax temperature.")

    # ── Test 4: Show actual probability redistribution ──
    print(f"\n{'━' * 70}")
    print("[6] Probability Redistribution at T=0.3")
    print(f"{'━' * 70}")

    vlm.temperature = 0.0
    context = list("Alice was beginning to get very tired of sitting by her sister ")
    probs_cold = vlm.predict(context)

    vlm.temperature = 0.3
    probs_warm = vlm.predict(context)

    # Top 15 symbols
    top_cold = sorted(probs_cold.items(), key=lambda x: -x[1])[:15]
    top_warm = sorted(probs_warm.items(), key=lambda x: -x[1])[:15]

    print(f"\n  {'Rank':>4} {'T=0.0':>20} {'T=0.3':>20} {'Δ':>8}")
    print(f"  {'─' * 55}")

    for i in range(min(15, len(top_cold))):
        sym_c, p_c = top_cold[i]
        sym_w, p_w = top_warm[i]
        delta = p_w - p_c
        s_c = repr(sym_c) if sym_c in (' ', '\n') else sym_c
        s_w = repr(sym_w) if sym_w in (' ', '\n') else sym_w
        print(f"  {i+1:>4}  {s_c:>3} = {p_c:.4f}        {s_w:>3} = {p_w:.4f}  {delta:>+.4f}")

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("T90 SUMMARY")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
