#!/usr/bin/env python3
"""
Benchmark: Language Core on Real English Text
================================================
Tests PPM on Gutenberg texts:
1. Alice in Wonderland (~174KB)
2. Shakespeare Sonnets (~120KB)
3. Pride and Prejudice (~772KB)

Metrics:
- Character-level bits per character (bpc)
- Word-level perplexity
- Online adaptation (train Alice → adapt Shakespeare)
- Generation quality
- PPM order scaling (4-12)
- Training data volume scaling
"""

import numpy as np
import sys, os, time, math, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.language import PPMModel, HierarchicalLanguageModel


def load_gutenberg(path, max_chars=None):
    """Load a Gutenberg text, stripping header/footer."""
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    # Strip Gutenberg header
    start_markers = ['*** START OF THE PROJECT GUTENBERG', '*** START OF THIS PROJECT GUTENBERG']
    for marker in start_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[text.index('\n', idx) + 1:]
            break

    # Strip Gutenberg footer
    end_markers = ['*** END OF THE PROJECT GUTENBERG', '*** END OF THIS PROJECT GUTENBERG',
                   'End of the Project Gutenberg', 'End of Project Gutenberg']
    for marker in end_markers:
        idx = text.find(marker)
        if idx >= 0:
            text = text[:idx]
            break

    # Clean
    text = text.strip()
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)

    if max_chars:
        text = text[:max_chars]

    return text


def main():
    print("=" * 80)
    print("LANGUAGE CORE — Real Text Benchmark")
    print("=" * 80)

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')

    # Load texts
    texts = {}
    for name, fname in [('Alice', 'alice.txt'), ('Shakespeare', 'shakespeare.txt'), ('Pride', 'pride.txt')]:
        path = os.path.join(data_dir, fname)
        if os.path.exists(path):
            texts[name] = load_gutenberg(path)
            print(f"  Loaded {name}: {len(texts[name]):,} chars, {len(texts[name].split()):,} words")
        else:
            print(f"  WARNING: {path} not found")

    if not texts:
        print("No data files found!")
        return

    # ── TEST 1a: Online BPC (standard PPM evaluation) ─────────
    print(f"\n{'━' * 80}")
    print("TEST 1a: Online BPC — predict+update (standard PPM evaluation)")
    print(f"{'━' * 80}")
    print("  Literature reference: PPM achieves ~1.5 bpc on English text")

    alice = texts.get('Alice', '')
    test_text = alice[:20000]

    print(f"\n  Text: {len(test_text):,} chars")
    print(f"\n  {'Order':<8} {'BPC':>8} {'Vocab':>8} {'Time (s)':>10}")
    print(f"  {'─' * 38}")

    for order in [3, 5, 6, 8, 10, 12]:
        t0 = time.time()
        ppm = PPMModel(max_order=order)

        total_bits = 0.0
        chars = list(test_text)

        for i in range(1, len(chars)):
            context = chars[:i]
            probs = ppm.predict(context)
            p = probs.get(chars[i], 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
            # Online: update after prediction
            ppm.update(context, chars[i])

        bpc = total_bits / (len(chars) - 1)
        elapsed = time.time() - t0
        print(f"  {order:<8} {bpc:>8.3f} {len(ppm.vocab):>8} {elapsed:>10.1f}")

    # ── TEST 1b: Split BPC (train/eval separate) ────────────────
    print(f"\n{'━' * 80}")
    print("TEST 1b: Split BPC — train 15K, eval 5K (no online update)")
    print(f"{'━' * 80}")

    train_text = test_text[:15000]
    eval_text = test_text[15000:]

    print(f"\n  Train: {len(train_text):,} chars, Eval: {len(eval_text):,} chars")
    print(f"\n  {'Order':<8} {'BPC':>8} {'Vocab':>8} {'Time (s)':>10}")
    print(f"  {'─' * 38}")

    for order in [3, 5, 6, 8, 10, 12]:
        t0 = time.time()
        ppm = PPMModel(max_order=order)
        ppm.train(list(train_text))

        # Evaluate bpc on held-out text
        total_bits = 0.0
        n = 0
        context = list(train_text[-order:])

        for ch in eval_text:
            probs = ppm.predict(context)
            p = probs.get(ch, 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
            context.append(ch)
            n += 1

        bpc = total_bits / max(n, 1)
        elapsed = time.time() - t0
        print(f"  {order:<8} {bpc:>8.3f} {len(ppm.vocab):>8} {elapsed:>10.1f}")

    # ── TEST 2: Cross-Text BPC ──────────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 2: Cross-Text BPC (order=8, 20K chars each)")
    print(f"{'━' * 80}")

    print(f"\n  {'Text':<15} {'BPC':>8} {'Words':>8} {'Unique chars':>14}")
    print(f"  {'─' * 48}")

    for name, text in texts.items():
        chunk = text[:20000]
        train_chunk = chunk[:15000]
        eval_chunk = chunk[15000:]

        ppm = PPMModel(max_order=8)
        ppm.train(list(train_chunk))

        total_bits = 0.0
        context = list(train_chunk[-8:])
        for ch in eval_chunk:
            probs = ppm.predict(context)
            p = probs.get(ch, 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
            context.append(ch)

        bpc = total_bits / max(len(eval_chunk), 1)
        unique_chars = len(set(chunk))
        print(f"  {name:<15} {bpc:>8.3f} {len(chunk.split()):>8} {unique_chars:>14}")

    # ── TEST 3: Training Data Scaling ───────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 3: Training Data Scaling (order=8, eval on Alice 15K-20K)")
    print(f"{'━' * 80}")

    eval_text_3 = alice[15000:20000]

    print(f"\n  {'Train size':<12} {'BPC':>8} {'Time (s)':>10}")
    print(f"  {'─' * 34}")

    for train_size in [1000, 3000, 5000, 10000, 15000]:
        t0 = time.time()
        ppm = PPMModel(max_order=8)
        ppm.train(list(alice[:train_size]))

        total_bits = 0.0
        context = list(alice[max(0, train_size - 8):train_size])
        for ch in eval_text_3:
            probs = ppm.predict(context)
            p = probs.get(ch, 1e-10)
            total_bits -= math.log2(max(p, 1e-10))
            context.append(ch)

        bpc = total_bits / max(len(eval_text_3), 1)
        elapsed = time.time() - t0
        print(f"  {train_size:<12,} {bpc:>8.3f} {elapsed:>10.1f}")

    # ── TEST 4: Online Adaptation ───────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 4: Online Adaptation (train Alice → evaluate Shakespeare)")
    print(f"{'━' * 80}")

    alice_train = alice[:15000]
    shakes = texts.get('Shakespeare', '')
    if not shakes:
        print("  Shakespeare text not available, skipping")
    else:
        shakes_eval = shakes[:5000]

        # Baseline: Alice-trained model on Shakespeare (no adaptation)
        ppm_frozen = PPMModel(max_order=8)
        ppm_frozen.train(list(alice_train))

        total_bits_frozen = 0.0
        context = list(alice_train[-8:])
        for ch in shakes_eval:
            probs = ppm_frozen.predict(context)
            p = probs.get(ch, 1e-10)
            total_bits_frozen -= math.log2(max(p, 1e-10))
            context.append(ch)
        bpc_frozen = total_bits_frozen / len(shakes_eval)

        # Online: Alice-trained model adapts while reading Shakespeare
        ppm_online = PPMModel(max_order=8)
        ppm_online.train(list(alice_train))

        total_bits_online = 0.0
        context = list(alice_train[-8:])
        for ch in shakes_eval:
            probs = ppm_online.predict(context)
            p = probs.get(ch, 1e-10)
            total_bits_online -= math.log2(max(p, 1e-10))
            # KEY: update model with this observation
            ppm_online.update(context, ch)
            context.append(ch)
        bpc_online = total_bits_online / len(shakes_eval)

        # Shakespeare-trained baseline
        ppm_native = PPMModel(max_order=8)
        ppm_native.train(list(shakes[:15000]))
        total_bits_native = 0.0
        context = list(shakes[14992:15000])
        for ch in shakes[15000:20000]:
            probs = ppm_native.predict(context)
            p = probs.get(ch, 1e-10)
            total_bits_native -= math.log2(max(p, 1e-10))
            context.append(ch)
        bpc_native = total_bits_native / min(5000, len(shakes) - 15000)

        print(f"\n  {'Mode':<30} {'BPC':>8}")
        print(f"  {'─' * 40}")
        print(f"  {'Alice→Shakespeare (frozen)':<30} {bpc_frozen:>8.3f}")
        print(f"  {'Alice→Shakespeare (online)':<30} {bpc_online:>8.3f}")
        print(f"  {'Shakespeare native':<30} {bpc_native:>8.3f}")
        adapt_pct = (1 - (bpc_online - bpc_native) / (bpc_frozen - bpc_native)) * 100
        print(f"\n  Adaptation: {adapt_pct:.1f}% of gap closed by online learning")

    # ── TEST 5: Word-Level Perplexity ───────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 5: Word-Level Perplexity (order=5)")
    print(f"{'━' * 80}")

    print(f"\n  {'Text':<15} {'PPL':>10} {'Vocab':>8}")
    print(f"  {'─' * 36}")

    for name, text in texts.items():
        words = text[:30000].split()
        train_words = words[:int(len(words) * 0.8)]
        eval_words = words[int(len(words) * 0.8):]

        ppm_w = PPMModel(max_order=5)
        ppm_w.train(train_words)

        total_lp = 0.0
        n = 0
        for i in range(1, len(eval_words)):
            ctx = eval_words[max(0, i - 5):i]
            probs = ppm_w.predict(ctx)
            p = probs.get(eval_words[i], 1e-10)
            total_lp += math.log2(max(p, 1e-10))
            n += 1

        ppl = 2.0 ** (-total_lp / max(n, 1))
        print(f"  {name:<15} {ppl:>10.1f} {len(ppm_w.vocab):>8}")

    # ── TEST 6: Generation Quality ──────────────────────────────
    print(f"\n{'━' * 80}")
    print("TEST 6: Text Generation (char-level, order=8)")
    print(f"{'━' * 80}")

    for name in ['Alice', 'Shakespeare']:
        text = texts.get(name, '')
        if not text:
            continue

        ppm_gen = PPMModel(max_order=8)
        ppm_gen.train(list(text[:30000]))

        rng = np.random.RandomState(42)

        for temp in [0.5, 0.8, 1.0]:
            seed = list(text[5000:5050])
            generated = ppm_gen.generate(seed, 200, temperature=temp, rng=rng)
            gen_text = ''.join(generated)
            print(f"\n  [{name} T={temp}] {''.join(seed[-20:])}|{gen_text[:80]}")

    # ── TEST 7: Large-Scale (Pride & Prejudice) ─────────────────
    print(f"\n{'━' * 80}")
    print("TEST 7: Large-Scale — Pride & Prejudice (order=8)")
    print(f"{'━' * 80}")

    pride = texts.get('Pride', '')
    if pride:
        # Use more data
        for train_size in [50000, 100000, 200000]:
            if train_size > len(pride) - 5000:
                break

            t0 = time.time()
            ppm = PPMModel(max_order=8)
            ppm.train(list(pride[:train_size]))

            eval_start = train_size
            eval_end = min(train_size + 5000, len(pride))
            eval_chunk = pride[eval_start:eval_end]

            total_bits = 0.0
            context = list(pride[train_size - 8:train_size])
            for ch in eval_chunk:
                probs = ppm.predict(context)
                p = probs.get(ch, 1e-10)
                total_bits -= math.log2(max(p, 1e-10))
                context.append(ch)

            bpc = total_bits / len(eval_chunk)
            elapsed = time.time() - t0
            print(f"  Train={train_size:>7,} chars  BPC={bpc:.3f}  Vocab={len(ppm.vocab)}  Time={elapsed:.1f}s")

    # ── SUMMARY ─────────────────────────────────────────────────
    print(f"\n{'═' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'═' * 80}")
    print("""
  REFERENCE:
  - PPM literature: ~1.5 bpc on English (Cleary & Witten 1984)
  - LSTM/RNN: ~1.2 bpc
  - Transformer (GPT-2 small): ~0.9 bpc
  - Entropy of English: ~1.0-1.3 bpc (Shannon 1951)

  KEY QUESTION: Does our PPM implementation match ~1.5 bpc on real text?
""")


if __name__ == "__main__":
    main()
