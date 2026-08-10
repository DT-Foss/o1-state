#!/usr/bin/env python3
"""
T90b — Fair Bernoulli vs Softmax Comparison
=============================================
Compares at EQUAL COHERENCE (same BPC), measures diversity.

The question: at the same quality level, does Bernoulli ×2 mod 9
produce qualitatively different diversity than softmax?
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
    if len(text) < n:
        return 0.0
    ngrams = [text[i:i+n] for i in range(len(text) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def word_diversity(text):
    words = text.split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def score_bpc(text, scorer):
    """Score generated text with a reference PPM."""
    chars = list(text)
    total_bits = 0.0
    for i in range(1, min(len(chars), 500)):
        ctx = chars[:i]
        p = scorer.predict(ctx)
        prob = max(p.get(chars[i], 1e-10), 1e-10)
        total_bits -= math.log2(prob)
    return total_bits / max(min(len(chars), 500) - 1, 1)


def main():
    print("=" * 70)
    print("T90b — FAIR BERNOULLI vs SOFTMAX COMPARISON")
    print("=" * 70)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    alice_path = os.path.join(data_dir, 'alice.txt')
    alice = load_gutenberg(alice_path)
    train_text = alice[:30000]

    # Train both models on same data
    print("\n[1] Training models...")
    vlm = VortexLanguageModel(order=6, bridge_strength=0.0, temperature=0.0)
    vlm.train(train_text)

    ppm = PPMModel(max_order=6)
    ppm.train(list(train_text))

    # Reference scorer (separate PPM, never used for generation)
    scorer = PPMModel(max_order=6)
    scorer.train(list(train_text))

    seed = "Alice was beginning to get very "
    n_gen = 200
    n_samples = 10

    # ── Sweep both methods, collect (BPC, diversity) pairs ──
    print("\n[2] Sweeping temperature ranges...")

    # Softmax: use raw PPM with standard temperature
    softmax_points = []
    for temp in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 2.0, 3.0]:
        texts = []
        for s in range(n_samples):
            rng = np.random.RandomState(s)
            gen = ppm.generate(list(seed), n_gen, temperature=temp, rng=rng)
            texts.append(''.join(gen))

        all_text = ' '.join(texts)
        bpc = np.mean([score_bpc(t, scorer) for t in texts])
        d3 = ngram_diversity(all_text, 3) * 100
        d5 = ngram_diversity(all_text, 5) * 100
        wd = word_diversity(all_text) * 100
        softmax_points.append((temp, bpc, d3, d5, wd, texts[0][:60]))

    # Bernoulli: use VortexLanguageModel with temporal modulation
    bernoulli_points = []
    for temp in [0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        vlm.temperature = temp
        texts = []
        for s in range(n_samples):
            rng = np.random.RandomState(s)
            gen = vlm.generate(list(seed), n_gen, rng=rng)
            texts.append(gen)

        all_text = ' '.join(texts)
        bpc = np.mean([score_bpc(t, scorer) for t in texts])
        d3 = ngram_diversity(all_text, 3) * 100
        d5 = ngram_diversity(all_text, 5) * 100
        wd = word_diversity(all_text) * 100
        bernoulli_points.append((temp, bpc, d3, d5, wd, texts[0][:60]))

    # ── Print results ──
    print(f"\n{'━' * 70}")
    print("[3] Softmax Temperature Results")
    print(f"{'━' * 70}")
    print(f"  {'T':>5} {'BPC':>6} {'3g%':>6} {'5g%':>6} {'Wd%':>6} {'Sample':<55}")
    print(f"  {'─' * 85}")
    for temp, bpc, d3, d5, wd, sample in softmax_points:
        print(f"  {temp:>5.2f} {bpc:>6.2f} {d3:>5.1f}% {d5:>5.1f}% {wd:>5.1f}% {sample}")

    print(f"\n{'━' * 70}")
    print("[4] Bernoulli ×2mod9 Temperature Results")
    print(f"{'━' * 70}")
    print(f"  {'T':>5} {'BPC':>6} {'3g%':>6} {'5g%':>6} {'Wd%':>6} {'Sample':<55}")
    print(f"  {'─' * 85}")
    for temp, bpc, d3, d5, wd, sample in bernoulli_points:
        print(f"  {temp:>5.2f} {bpc:>6.2f} {d3:>5.1f}% {d5:>5.1f}% {wd:>5.1f}% {sample}")

    # ── Find matching BPC ranges ──
    print(f"\n{'━' * 70}")
    print("[5] Equal-Coherence Comparison")
    print(f"{'━' * 70}")

    # Find BPC ranges where both methods have data
    target_bpcs = [2.0, 2.5, 3.0, 4.0]

    print(f"\n  {'Target BPC':>10} {'Method':<15} {'Actual BPC':>10} {'3g%':>6} {'5g%':>6} {'Wd%':>6}")
    print(f"  {'─' * 60}")

    for target in target_bpcs:
        # Find closest softmax point
        s_best = min(softmax_points, key=lambda x: abs(x[1] - target))
        b_best = min(bernoulli_points, key=lambda x: abs(x[1] - target))

        if abs(s_best[1] - target) < 1.5 and abs(b_best[1] - target) < 1.5:
            print(f"  {target:>10.1f} {'Softmax':<15} {s_best[1]:>10.2f} {s_best[2]:>5.1f}% {s_best[3]:>5.1f}% {s_best[4]:>5.1f}%")
            print(f"  {'':>10} {'Bernoulli':<15} {b_best[1]:>10.2f} {b_best[2]:>5.1f}% {b_best[3]:>5.1f}% {b_best[4]:>5.1f}%")

            # Which has higher diversity at similar BPC?
            d_diff = b_best[2] - s_best[2]
            if d_diff > 1:
                winner = "BERNOULLI"
            elif d_diff < -1:
                winner = "SOFTMAX"
            else:
                winner = "TIE"
            print(f"  {'':>10} {'→ Winner:':<15} {winner} (Δ3g = {d_diff:+.1f}%)")
            print()

    # ── Temporal variation analysis ──
    print(f"{'━' * 70}")
    print("[6] Temporal Temperature Variation (Bernoulli T=0.15)")
    print(f"{'━' * 70}")

    vlm.temperature = 0.15
    vlm._generation_step = 0
    context = list(seed)

    print(f"\n  {'Step':>4} {'Orbit':>6} {'T_eff':>6} {'Top1':>12} {'Top2':>12} {'Top3':>12} {'Entropy':>8}")
    print(f"  {'─' * 65}")

    for step in range(12):
        vlm._generation_step = step
        probs = vlm.predict(context)
        top = sorted(probs.items(), key=lambda x: -x[1])[:3]

        orbit_idx = step % 6
        orbit_val = [1, 2, 4, 8, 7, 5][orbit_idx]
        t_eff = 0.15 * orbit_val / 9

        # Entropy
        h = -sum(p * math.log2(max(p, 1e-10)) for p in probs.values())

        t1 = f"{repr(top[0][0]):>3}={top[0][1]:.3f}" if len(top) > 0 else ""
        t2 = f"{repr(top[1][0]):>3}={top[1][1]:.3f}" if len(top) > 1 else ""
        t3 = f"{repr(top[2][0]):>3}={top[2][1]:.3f}" if len(top) > 2 else ""

        print(f"  {step:>4} {orbit_val:>4}/9 {t_eff:>6.3f} {t1:>12} {t2:>12} {t3:>12} {h:>7.3f}")

    # ── Summary ──
    print(f"\n{'═' * 70}")
    print("T90 VERDICT")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
