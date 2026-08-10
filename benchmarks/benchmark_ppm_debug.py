#!/usr/bin/env python3
"""
PPM Debug — Why does higher order hurt?
Find root cause and fix.
"""

import numpy as np
import sys, os, time, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.language import PPMModel


def load_gutenberg(path, max_chars=None):
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
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)
    if max_chars:
        text = text[:max_chars]
    return text


def online_bpc(text, order, skip_warmup=0):
    """Standard PPM evaluation: predict then update, char by char."""
    ppm = PPMModel(max_order=order)
    chars = list(text)
    total_bits = 0.0
    n = 0

    for i in range(1, len(chars)):
        context = chars[max(0, i - order):i]  # Only pass last `order` chars as context
        probs = ppm.predict(context)
        p = probs.get(chars[i], 1e-10)
        if i > skip_warmup:
            total_bits -= math.log2(max(p, 1e-10))
            n += 1
        ppm.update(context, chars[i])

    return total_bits / max(n, 1)


def split_bpc(train_text, eval_text, order):
    """Train on train_text, evaluate on eval_text without updating."""
    ppm = PPMModel(max_order=order)
    ppm.train(list(train_text))

    total_bits = 0.0
    context = list(train_text[-(order * 2):])

    for ch in eval_text:
        ctx = context[-(order):]
        probs = ppm.predict(ctx)
        p = probs.get(ch, 1e-10)
        total_bits -= math.log2(max(p, 1e-10))
        context.append(ch)

    return total_bits / len(eval_text)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
    pride = load_gutenberg(os.path.join(data_dir, 'pride.txt'))
    alice = load_gutenberg(os.path.join(data_dir, 'alice.txt'))

    print("=" * 70)
    print("PPM DEBUG — Order Scaling & Context Handling")
    print("=" * 70)

    # ── Test A: Online BPC with warmup skip ─────────────────
    print(f"\n{'━' * 70}")
    print("A: Online BPC (Alice 20K, skip first 500 chars warmup)")
    print(f"{'━' * 70}")

    text = alice[:20000]
    print(f"\n  {'Order':<8} {'BPC (raw)':>10} {'BPC (skip500)':>14}")
    print(f"  {'─' * 34}")

    for order in [3, 5, 8, 12]:
        bpc_raw = online_bpc(text, order, skip_warmup=0)
        bpc_skip = online_bpc(text, order, skip_warmup=500)
        print(f"  {order:<8} {bpc_raw:>10.3f} {bpc_skip:>14.3f}")

    # ── Test B: Context length passed to predict ────────────
    print(f"\n{'━' * 70}")
    print("B: Context length effect (split, Pride 100K train, 5K eval)")
    print(f"{'━' * 70}")
    print("  Testing: does passing FULL history vs clipped context matter?")

    train = pride[:100000]
    evl = pride[100000:105000]

    print(f"\n  {'Order':<8} {'Clipped ctx':>12} {'Full ctx':>10}")
    print(f"  {'─' * 32}")

    for order in [3, 5, 8]:
        bpc_clip = split_bpc(train, evl, order)

        # Full context version: pass ALL preceding chars
        ppm = PPMModel(max_order=order)
        ppm.train(list(train))
        total_bits = 0.0
        all_chars = list(train + evl)
        start = len(train)
        for i in range(start, len(all_chars)):
            context = all_chars[max(0, i - order):i]  # Sliding window (PPM only uses last order chars)
            probs = ppm.predict(context)
            p = probs.get(all_chars[i], 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
        bpc_full = total_bits / len(evl)

        print(f"  {order:<8} {bpc_clip:>12.3f} {bpc_full:>10.3f}")

    # ── Test C: Large data, order scaling ───────────────────
    print(f"\n{'━' * 70}")
    print("C: Order scaling with LARGE training (Pride & Prejudice)")
    print(f"{'━' * 70}")

    print(f"\n  {'Train':>8} {'Ord3':>8} {'Ord5':>8} {'Ord8':>8} {'Ord12':>8}")
    print(f"  {'─' * 42}")

    for train_size in [20000, 50000, 100000, 200000]:
        if train_size + 5000 > len(pride):
            break
        results = []
        for order in [3, 5, 8, 12]:
            bpc = split_bpc(pride[:train_size], pride[train_size:train_size+5000], order)
            results.append(bpc)
        print(f"  {train_size:>8,} {'  '.join(f'{r:>6.3f}' for r in results)}")

    # ── Test D: Online on full Pride & Prejudice ────────────
    print(f"\n{'━' * 70}")
    print("D: Online BPC on full texts (skip 1000 warmup)")
    print(f"{'━' * 70}")

    print(f"\n  {'Text':<15} {'Chars':>8} {'Ord5':>8} {'Ord8':>8}")
    print(f"  {'─' * 41}")

    for name, text in [('Alice', alice), ('Pride 50K', pride[:50000])]:
        results = []
        for order in [5, 8]:
            t0 = time.time()
            bpc = online_bpc(text, order, skip_warmup=1000)
            elapsed = time.time() - t0
            results.append(f"{bpc:.3f}")
        print(f"  {name:<15} {len(text):>8,} {'  '.join(results)}")

    # ── Test E: Generation comparison ───────────────────────
    print(f"\n{'━' * 70}")
    print("E: Generation quality (order 8, Alice 30K)")
    print(f"{'━' * 70}")

    ppm = PPMModel(max_order=8)
    ppm.train(list(alice[:30000]))

    seed = list("Alice was beginning to get very ")
    rng = np.random.RandomState(42)

    for temp in [0.3, 0.5, 0.8, 1.0]:
        gen = ppm.generate(seed, 150, temperature=temp, rng=rng)
        print(f"\n  T={temp}: {''.join(seed[-15:])}|{''.join(gen[:80])}")

    print(f"\n{'═' * 70}")
    print("DONE")


if __name__ == "__main__":
    main()
