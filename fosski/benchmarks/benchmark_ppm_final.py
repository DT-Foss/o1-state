#!/usr/bin/env python3
"""
PPM Final Benchmark — Full-scale test on real text.
"""

import numpy as np
import sys, os, time, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def split_bpc(train_text, eval_text, order):
    ppm = PPMModel(max_order=order)
    ppm.train(list(train_text))
    total_bits = 0.0
    context = list(train_text[-order:])
    for ch in eval_text:
        probs = ppm.predict(context[-order:])
        p = probs.get(ch, 1e-10)
        total_bits -= math.log2(max(p, 1e-10))
        context.append(ch)
    return total_bits / len(eval_text)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    pride = load_gutenberg(os.path.join(data_dir, 'pride.txt'))
    alice = load_gutenberg(os.path.join(data_dir, 'alice.txt'))
    shakes = load_gutenberg(os.path.join(data_dir, 'shakespeare.txt'))

    print("=" * 70)
    print("PPM FINAL BENCHMARK — Real English Text")
    print("=" * 70)
    print(f"  Pride & Prejudice: {len(pride):,} chars")
    print(f"  Alice:             {len(alice):,} chars")
    print(f"  Shakespeare:       {len(shakes):,} chars")

    # ── 1: Full P&P scaling ─────────────────────────────────
    print(f"\n{'━' * 70}")
    print("1: Full P&P — Training size × Order (eval = next 10K)")
    print(f"{'━' * 70}")

    print(f"\n  {'Train':>8} {'Ord3':>8} {'Ord5':>8} {'Ord7':>8} {'Ord8':>8} {'Time':>8}")
    print(f"  {'─' * 50}")

    for train_size in [50000, 100000, 200000]:
        eval_start = train_size
        eval_end = min(train_size + 10000, len(pride))
        if eval_end - eval_start < 1000:
            break
        eval_chunk = pride[eval_start:eval_end]
        results = []
        t0 = time.time()
        for order in [3, 5, 7, 8]:
            bpc = split_bpc(pride[:train_size], eval_chunk, order)
            results.append(bpc)
        elapsed = time.time() - t0
        print(f"  {train_size:>8,} {'  '.join(f'{r:>6.3f}' for r in results)} {elapsed:>7.1f}s")

    # ── 2: Online BPC on full Alice ─────────────────────────
    print(f"\n{'━' * 70}")
    print("2: Online BPC — Full Alice (144K chars, skip 2000 warmup)")
    print(f"{'━' * 70}")

    print(f"\n  {'Order':<8} {'BPC':>8} {'Time':>8}")
    print(f"  {'─' * 26}")

    for order in [5, 8]:
        t0 = time.time()
        ppm = PPMModel(max_order=order)
        chars = list(alice)
        total_bits = 0.0
        n = 0
        for i in range(1, len(chars)):
            ctx = chars[max(0, i - order):i]
            probs = ppm.predict(ctx)
            p = probs.get(chars[i], 1e-10)
            if i > 2000:
                total_bits -= math.log2(max(p, 1e-10))
                n += 1
            ppm.update(ctx, chars[i])
        bpc = total_bits / n
        elapsed = time.time() - t0
        print(f"  {order:<8} {bpc:>8.3f} {elapsed:>7.1f}s")

    # ── 3: Cross-domain adaptation ──────────────────────────
    print(f"\n{'━' * 70}")
    print("3: Cross-Domain — Train P&P 200K → Eval Alice/Shakespeare")
    print(f"{'━' * 70}")

    ppm_pp = PPMModel(max_order=5)
    ppm_pp.train(list(pride[:100000]))

    for name, text in [('P&P (in-domain)', pride[100000:110000]),
                       ('Alice', alice[:10000]),
                       ('Shakespeare', shakes[:10000])]:
        total_bits = 0.0
        context = list(pride[99995:100000])
        for ch in text:
            probs = ppm_pp.predict(context[-5:])
            p = probs.get(ch, 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
            context.append(ch)
        bpc = total_bits / len(text)
        print(f"  {name:<25} BPC = {bpc:.3f}")

    # ── 4: Cross-domain with online adaptation ──────────────
    print(f"\n{'━' * 70}")
    print("4: Cross-Domain + Online Adaptation (order=5)")
    print(f"{'━' * 70}")

    for name, text in [('Alice', alice[:10000]), ('Shakespeare', shakes[:10000])]:
        # Frozen (no update)
        ppm_f = PPMModel(max_order=5)
        ppm_f.train(list(pride[:100000]))
        bits_f = 0.0
        ctx_f = list(pride[99995:100000])
        for ch in text:
            probs = ppm_f.predict(ctx_f[-5:])
            p = probs.get(ch, 1e-10)
            bits_f -= math.log2(max(p, 1e-10))
            ctx_f.append(ch)
        bpc_f = bits_f / len(text)

        # Online (update after each char)
        ppm_o = PPMModel(max_order=5)
        ppm_o.train(list(pride[:100000]))
        bits_o = 0.0
        ctx_o = list(pride[99995:100000])
        for ch in text:
            probs = ppm_o.predict(ctx_o[-5:])
            p = probs.get(ch, 1e-10)
            bits_o -= math.log2(max(p, 1e-10))
            ppm_o.update(ctx_o[-5:], ch)
            ctx_o.append(ch)
        bpc_o = bits_o / len(text)

        print(f"  {name:<15} Frozen={bpc_f:.3f}  Online={bpc_o:.3f}  Δ={bpc_f-bpc_o:.3f}")

    # ── 5: Generation ───────────────────────────────────────
    print(f"\n{'━' * 70}")
    print("5: Generation (order=5, P&P 200K training, T=0.5)")
    print(f"{'━' * 70}")

    ppm_gen = PPMModel(max_order=5)
    ppm_gen.train(list(pride[:100000]))
    rng = np.random.RandomState(42)

    seeds = [
        "It is a truth universally ",
        "Mr. Darcy looked at her ",
        "Elizabeth could not help ",
    ]

    for seed in seeds:
        gen = ppm_gen.generate(list(seed), 200, temperature=0.5, rng=rng)
        print(f"\n  {seed}|{''.join(gen[:80])}")

    print(f"\n{'═' * 70}")
    print("SUMMARY")
    print(f"{'═' * 70}")


if __name__ == "__main__":
    main()
