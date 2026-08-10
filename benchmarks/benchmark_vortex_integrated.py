#!/usr/bin/env python3
"""
Benchmark: Integrated Vortex Language Model on Real Text
==========================================================
Tests ALL 5 vortex components working together, not individually.

1. Bridge strength sweep (THE critical parameter)
2. Integrated system vs. standard PPM on real text
3. Online adaptation with fiber feedback
4. Bernoulli perturbation for generation
5. 3-adic scale weighting validation
"""

import numpy as np
import sys, os, time, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.language import PPMModel
from core.vortex import VortexLanguageModel


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


def baseline_bpc(train_text, eval_text, order, online=False):
    """Standard PPM baseline."""
    ppm = PPMModel(max_order=order)
    ppm.train(list(train_text))

    total_bits = 0.0
    context = list(train_text[-order:])
    for ch in eval_text:
        probs = ppm.predict(context)
        p = max(probs.get(ch, 1e-10), 1e-10)
        total_bits -= math.log2(p)
        if online:
            ppm.update(context, ch)
        context.append(ch)

    return total_bits / len(eval_text)


def main():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    print("=" * 70)
    print("VORTEX INTEGRATED BENCHMARK — Real English Text")
    print("=" * 70)

    # Load texts
    texts = {}
    for name, fname in [('Alice', 'alice.txt'), ('Shakespeare', 'shakespeare.txt'),
                        ('Pride', 'pride.txt')]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            texts[name] = load_gutenberg(path)
            print(f"  {name}: {len(texts[name]):,} chars")

    alice = texts.get('Alice', '')
    pride = texts.get('Pride', '')
    shakes = texts.get('Shakespeare', '')

    # ── TEST 1: Bridge Strength Sweep ───────────────────────
    print(f"\n{'━' * 70}")
    print("TEST 1: Bridge Strength Sweep (×3 coupling, P&P 50K train)")
    print(f"{'━' * 70}")
    print("  This is THE critical parameter.")
    print("  0.0 = fibers isolated, 1.0 = fibers fully coupled")

    train_text = pride[:50000]
    eval_text = pride[50000:55000]

    # Train the integrated model
    print("\n  Training Vortex model on 50K chars...")
    t0 = time.time()

    # Get baseline first
    base_bpc = baseline_bpc(train_text, eval_text, order=6)
    print(f"  Standard PPM-6 baseline: {base_bpc:.4f} bpc")

    # Sweep bridge strength
    print(f"\n  {'Bridge':>8} {'BPC':>8} {'Δ vs base':>10} {'Improvement':>12}")
    print(f"  {'─' * 42}")

    best_bridge = 0.0
    best_bpc = 999.0
    sweep_results = []

    for bs in [0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.0]:
        vortex = VortexLanguageModel(order=6, bridge_strength=bs)
        vortex.train(train_text)

        # Eval without online adapt (fair comparison to baseline)
        total_bits = 0.0
        context = list(train_text[-6:])
        for ch in eval_text:
            probs = vortex.predict(context)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)
            context.append(ch)
        bpc = total_bits / len(eval_text)

        delta = bpc - base_bpc
        improvement = (1 - bpc / base_bpc) * 100
        sweep_results.append((bs, bpc, delta, improvement))

        if bpc < best_bpc:
            best_bpc = bpc
            best_bridge = bs

        marker = " ← BEST" if bpc <= best_bpc else ""
        print(f"  {bs:>8.2f} {bpc:>8.4f} {delta:>+10.4f} {improvement:>+11.2f}%{marker}")

    print(f"\n  Optimal bridge: {best_bridge:.2f}, BPC: {best_bpc:.4f}")
    print(f"  Improvement over baseline: {(1 - best_bpc / base_bpc) * 100:+.2f}%")

    # ── TEST 2: Integrated vs Baseline on All Texts ─────────
    print(f"\n{'━' * 70}")
    print(f"TEST 2: Integrated System vs Baseline (bridge={best_bridge:.2f})")
    print(f"{'━' * 70}")

    print(f"\n  {'Text':<15} {'Baseline':>10} {'Vortex':>10} {'Δ':>8} {'Improv':>8}")
    print(f"  {'─' * 54}")

    for name, text in texts.items():
        if len(text) < 10000:
            continue

        train_size = min(50000, len(text) - 5000)
        tr = text[:train_size]
        ev = text[train_size:train_size + 5000]

        # Baseline
        b_bpc = baseline_bpc(tr, ev, order=6)

        # Vortex
        vortex = VortexLanguageModel(order=6, bridge_strength=best_bridge)
        vortex.train(tr)

        total_bits = 0.0
        ctx = list(tr[-6:])
        for ch in ev:
            probs = vortex.predict(ctx)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)
            ctx.append(ch)
        v_bpc = total_bits / len(ev)

        delta = v_bpc - b_bpc
        imp = (1 - v_bpc / b_bpc) * 100
        print(f"  {name:<15} {b_bpc:>10.4f} {v_bpc:>10.4f} {delta:>+8.4f} {imp:>+7.2f}%")

    # ── TEST 3: Online Adaptation — The Architecture ────────
    print(f"\n{'━' * 70}")
    print("TEST 3: Online Adaptation (train P&P → adapt to Shakespeare)")
    print(f"{'━' * 70}")
    print("  Every fiber updates with every observation. System learns from itself.")

    if shakes:
        train_pp = pride[:50000]
        eval_shakes = shakes[:5000]

        # Baseline: frozen PPM
        b_frozen = baseline_bpc(train_pp, eval_shakes, order=6, online=False)
        b_online = baseline_bpc(train_pp, eval_shakes, order=6, online=True)

        # Vortex: frozen
        vortex_f = VortexLanguageModel(order=6, bridge_strength=best_bridge)
        vortex_f.train(train_pp)
        total_bits = 0.0
        ctx = list(train_pp[-6:])
        for ch in eval_shakes:
            probs = vortex_f.predict(ctx)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)
            ctx.append(ch)
        v_frozen = total_bits / len(eval_shakes)

        # Vortex: online (all fibers update)
        vortex_o = VortexLanguageModel(order=6, bridge_strength=best_bridge)
        vortex_o.train(train_pp)
        total_bits = 0.0
        ctx = list(train_pp[-6:])
        for ch in eval_shakes:
            probs = vortex_o.predict(ctx)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)
            vortex_o.update(ctx, ch)  # ALL fibers update
            ctx.append(ch)
        v_online = total_bits / len(eval_shakes)

        print(f"\n  {'Mode':<35} {'BPC':>8}")
        print(f"  {'─' * 45}")
        print(f"  {'PPM-6 frozen':<35} {b_frozen:>8.4f}")
        print(f"  {'PPM-6 online':<35} {b_online:>8.4f}")
        print(f"  {'Vortex frozen':<35} {v_frozen:>8.4f}")
        print(f"  {'Vortex online (all fibers)':<35} {v_online:>8.4f}")

        gap_base = b_frozen - b_online
        gap_vortex = v_frozen - v_online
        print(f"\n  PPM online gain: {gap_base:.4f} bpc ({gap_base/b_frozen*100:.1f}%)")
        print(f"  Vortex online gain: {gap_vortex:.4f} bpc ({gap_vortex/v_frozen*100:.1f}%)")
        if gap_vortex > gap_base:
            print(f"  → Vortex adapts {gap_vortex/max(gap_base,0.001):.1f}× FASTER than baseline")

    # ── TEST 4: 3-Adic vs Equal Weighting ───────────────────
    print(f"\n{'━' * 70}")
    print("TEST 4: 3-Adic Weighting vs Equal Weighting")
    print(f"{'━' * 70}")
    print("  3-adic: [1, 1/3, 1/9] normalized. Equal: [1/3, 1/3, 1/3].")

    train_t = pride[:50000]
    eval_t = pride[50000:55000]

    for weight_name, weights in [('3-adic [1, 1/3, 1/9]', np.array([1.0, 1/3, 1/9])),
                                  ('Equal [1/3, 1/3, 1/3]', np.array([1/3, 1/3, 1/3])),
                                  ('Char-only [1, 0, 0]', np.array([1.0, 0.01, 0.01]))]:
        weights = weights / weights.sum()

        vortex = VortexLanguageModel(order=6, bridge_strength=best_bridge)
        vortex._scale_weights = weights
        vortex.train(train_t)

        total_bits = 0.0
        ctx = list(train_t[-6:])
        for ch in eval_t:
            probs = vortex.predict(ctx)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)
            ctx.append(ch)
        bpc = total_bits / len(eval_t)

        print(f"  {weight_name:<30} BPC = {bpc:.4f}")

    # ── TEST 5: Bernoulli Perturbation for Generation ───────
    print(f"\n{'━' * 70}")
    print("TEST 5: Bernoulli Perturbation — Generation Quality")
    print(f"{'━' * 70}")

    vortex_gen = VortexLanguageModel(order=6, bridge_strength=best_bridge)
    vortex_gen.train(pride[:50000])
    rng = np.random.RandomState(42)

    seed = "It is a truth universally "

    for temp in [0.0, 0.1, 0.3, 0.5]:
        vortex_gen.temperature = temp
        gen = vortex_gen.generate(list(seed), 120, rng=rng)
        print(f"\n  T={temp}: {seed}|{gen[:80]}")

    # ── TEST 6: Scaling with training data ──────────────────
    print(f"\n{'━' * 70}")
    print(f"TEST 6: Data Scaling (bridge={best_bridge:.2f})")
    print(f"{'━' * 70}")

    print(f"\n  {'Train size':>10} {'PPM-6':>8} {'Vortex':>8} {'Δ':>8}")
    print(f"  {'─' * 38}")

    for train_size in [10000, 30000, 50000, 100000]:
        if train_size + 5000 > len(pride):
            break
        tr = pride[:train_size]
        ev = pride[train_size:train_size + 5000]

        b = baseline_bpc(tr, ev, order=6)

        vortex = VortexLanguageModel(order=6, bridge_strength=best_bridge)
        vortex.train(tr)
        total_bits = 0.0
        ctx = list(tr[-6:])
        for ch in ev:
            probs = vortex.predict(ctx)
            p = max(probs.get(ch, 1e-10), 1e-10)
            total_bits -= math.log2(p)
            ctx.append(ch)
        v = total_bits / len(ev)

        print(f"  {train_size:>10,} {b:>8.4f} {v:>8.4f} {v-b:>+8.4f}")

    # ── SUMMARY ─────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print("SUMMARY")
    print(f"{'═' * 70}")
    print(f"""
  Optimal bridge strength (×3 coupling): {best_bridge:.2f}
  Best Vortex BPC: {best_bpc:.4f}
  Baseline PPM-6: {base_bpc:.4f}
  Net improvement: {(1 - best_bpc / base_bpc) * 100:+.2f}%

  Target: >5% improvement to validate fiber architecture.
  If <2%: bridge strength or fiber features need rethinking.
""")


if __name__ == "__main__":
    main()
