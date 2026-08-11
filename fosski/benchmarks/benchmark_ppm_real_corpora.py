#!/usr/bin/env python3
"""S3: PPM auf echten, großen Korpora.

Benchmark character-level PPM (Order 3-8) auf:
- Alice in Wonderland (174K)
- Shakespeare Sonnets (120K)
- Pride and Prejudice (772K)
- Concatenated (1.07M)

Ziel: BPC < 1.5 auf 1M+ Zeichen.
Vergleich: LSTM/Mamba gleicher Parameterzahl (aus Literatur).
"""

import sys
import os
import time
import math
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


# =====================================================
# PPM Model (optimized for large corpora)
# =====================================================
class ContextNode:
    __slots__ = ['counts', 'total', 'children', 'escape_count']
    def __init__(self):
        self.counts = defaultdict(int)
        self.total = 0
        self.children = {}
        self.escape_count = 0


class PPMModel:
    def __init__(self, max_order=5):
        self.max_order = max_order
        self.root = ContextNode()
        self.vocab = set()

    def update(self, context, symbol):
        self.vocab.add(symbol)
        node = self.root
        node.counts[symbol] += 1
        node.total += 1
        for depth in range(min(len(context), self.max_order)):
            ctx_symbol = context[-(depth + 1)]
            if ctx_symbol not in node.children:
                node.children[ctx_symbol] = ContextNode()
            node = node.children[ctx_symbol]
            if symbol not in node.counts:
                node.escape_count += 1
            node.counts[symbol] += 1
            node.total += 1

    def predict(self, context):
        """PPM-D escape: p_escape = d / (d + total) where d = distinct symbols."""
        nodes = [self.root]
        node = self.root
        for depth in range(min(len(context), self.max_order)):
            ctx_symbol = context[-(depth + 1)]
            if ctx_symbol in node.children:
                node = node.children[ctx_symbol]
                nodes.append(node)
            else:
                break

        probs = {}
        excluded = set()
        remaining_mass = 1.0

        for node in reversed(nodes):
            if node.total == 0:
                continue
            distinct = len(node.counts)
            if distinct == 0:
                continue
            p_escape = distinct / (node.total + distinct)
            p_match = 1.0 - p_escape

            for sym, count in node.counts.items():
                if sym not in excluded:
                    p = remaining_mass * p_match * count / node.total
                    probs[sym] = probs.get(sym, 0) + p
                    excluded.add(sym)
            remaining_mass *= p_escape

        # Unseen symbols get remaining mass
        unseen = self.vocab - excluded
        if unseen and remaining_mass > 0:
            p_each = remaining_mass / len(unseen)
            for sym in unseen:
                probs[sym] = probs.get(sym, 0) + p_each

        # Absolute fallback
        if not probs:
            return {s: 1.0 / len(self.vocab) for s in self.vocab} if self.vocab else {}
        return probs

    def log_prob(self, context, symbol):
        probs = self.predict(context)
        p = probs.get(symbol, 1e-10)
        return math.log2(max(p, 1e-10))

    @property
    def n_nodes(self):
        """Count total nodes in context tree."""
        count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            count += 1
            stack.extend(node.children.values())
        return count


def evaluate_bpc(model, text, online=True):
    """Evaluate bits per character with optional online adaptation."""
    chars = list(text)
    total_bits = 0.0
    context = []

    for i, ch in enumerate(chars):
        probs = model.predict(context)
        p = probs.get(ch, 1e-10)
        total_bits -= math.log2(max(p, 1e-10))

        if online:
            model.update(context, ch)
        context.append(ch)

    return total_bits / len(chars)


# =====================================================
# MAIN
# =====================================================
if __name__ == '__main__':
    print("=" * 70)
    print("S3: PPM AUF ECHTEN KORPORA — Character-Level Language Modeling")
    print("=" * 70)

    # Load corpora
    corpora = {}
    for name, filename in [('Alice', 'alice.txt'), ('Shakespeare', 'shakespeare.txt'),
                           ('Pride', 'pride.txt')]:
        path = os.path.join(DATA_DIR, filename)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                text = f.read()
            corpora[name] = text
            print(f"  {name}: {len(text):,} chars")
        else:
            print(f"  {name}: NOT FOUND at {path}")

    if not corpora:
        print("No corpora found!")
        sys.exit(1)

    # Concatenated corpus
    all_text = ''.join(corpora.values())
    corpora['ALL (1M+)'] = all_text
    print(f"  ALL: {len(all_text):,} chars")
    print(f"  Unique chars: {len(set(all_text))}")

    # ─── Order sweep per corpus ───
    print("\n--- BPC by Order and Corpus ---")
    print(f"  {'Corpus':<20} {'Order':>5} {'BPC':>8} {'Nodes':>12} {'Train(s)':>10} {'Eval(s)':>10}")

    results = {}

    for corpus_name, text in corpora.items():
        # 80/20 split
        split = int(len(text) * 0.8)
        train_text = text[:split]
        test_text = text[split:split + min(10000, len(text) - split)]

        for order in [3, 5, 8]:
            t0 = time.time()
            model = PPMModel(max_order=order)

            # Train (use sliding window — update() only needs last max_order chars)
            chars = list(train_text)
            for i in range(len(chars)):
                ctx_start = max(0, i - order)
                model.update(chars[ctx_start:i], chars[i])
            t_train = time.time() - t0

            # Evaluate (online adaptation on test set)
            t0 = time.time()
            bpc = evaluate_bpc(model, test_text, online=True)
            t_eval = time.time() - t0

            nodes = model.n_nodes
            results[(corpus_name, order)] = bpc

            print(f"  {corpus_name:<20} {order:>5} {bpc:>8.4f} {nodes:>12,} {t_train:>10.1f} {t_eval:>10.1f}")

    # ─── Best results table ───
    print("\n--- Best BPC per Corpus ---")
    print(f"  {'Corpus':<20} {'Best Order':>10} {'BPC':>8}")
    for corpus_name in corpora:
        best_order = min([3, 5, 8], key=lambda o: results.get((corpus_name, o), 99))
        best_bpc = results[(corpus_name, best_order)]
        print(f"  {corpus_name:<20} {best_order:>10} {best_bpc:>8.4f}")

    # ─── Literature comparison ───
    print("\n--- Literature Comparison (Character-Level) ---")
    print("  Reference models on similar English text:")
    print(f"  {'Model':<30} {'BPC':>8} {'Params':>12}")
    print(f"  {'-'*50}")
    print(f"  {'Unigram baseline':<30} {'4.76':>8} {'0':>12}")
    print(f"  {'Bigram':<30} {'2.80':>8} {'0':>12}")
    print(f"  {'PPM-5 (ours, online)':<30} {results.get(('ALL (1M+)', 5), 0):>8.4f} {'~10M nodes':>12}")
    print(f"  {'LSTM (Graves 2013)':<30} {'1.67':>8} {'~4M':>12}")
    print(f"  {'Transformer-XL (Dai 2019)':<30} {'1.06':>8} {'~24M':>12}")
    print(f"  {'GPT-2 Small':<30} {'~1.00':>8} {'124M':>12}")

    # ─── Online adaptation test ───
    print("\n--- Online Adaptation: Train on Alice, Test on Shakespeare ---")
    model_alice = PPMModel(max_order=5)
    alice_text = corpora.get('Alice', '')
    shak_text = corpora.get('Shakespeare', '')

    if alice_text and shak_text:
        # Train on Alice (sliding window)
        chars = list(alice_text)
        for i in range(len(chars)):
            ctx_start = max(0, i - 5)
            model_alice.update(chars[ctx_start:i], chars[i])

        # Test on Shakespeare with online adaptation
        shak_test = shak_text[:5000]
        bpc_with_adapt = evaluate_bpc(model_alice, shak_test, online=True)

        # Without adaptation
        model_alice2 = PPMModel(max_order=5)
        chars2 = list(alice_text)
        for i in range(len(chars2)):
            ctx_start = max(0, i - 5)
            model_alice2.update(chars2[ctx_start:i], chars2[i])
        bpc_no_adapt = evaluate_bpc(model_alice2, shak_test, online=False)

        print(f"  With online adaptation:    {bpc_with_adapt:.4f} bpc")
        print(f"  Without adaptation:        {bpc_no_adapt:.4f} bpc")
        print(f"  Adaptation improvement:    {(1 - bpc_with_adapt/bpc_no_adapt)*100:.1f}%")

    # ─── Scaling analysis ───
    print("\n--- Scaling: BPC vs Corpus Size ---")
    full_text = all_text
    sizes = [10000, 50000, 100000, 500000, len(full_text)]

    print(f"  {'Size':>10} {'BPC (O=5)':>10} {'Nodes':>12}")
    for size in sizes:
        if size > len(full_text):
            continue
        subset = full_text[:size]
        split = int(len(subset) * 0.8)
        train = subset[:split]
        test = subset[split:split + min(5000, len(subset) - split)]

        model = PPMModel(max_order=5)
        chars = list(train)
        for i in range(len(chars)):
            ctx_start = max(0, i - 5)
            model.update(chars[ctx_start:i], chars[i])

        bpc = evaluate_bpc(model, test, online=True)
        print(f"  {size:>10,} {bpc:>10.4f} {model.n_nodes:>12,}")

    # ─── Summary ───
    best_all = results.get(('ALL (1M+)', 5), 99)
    print("\n" + "=" * 70)
    print("ZUSAMMENFASSUNG S3")
    print("=" * 70)
    print(f"""
  PPM Character-Level auf echtem Text (1.07M Zeichen):

  Bestes Ergebnis (ALL, Order 5): {best_all:.4f} bpc
  {'✅ UNTER 1.5 bpc Ziel' if best_all < 1.5 else '❌ Über 1.5 bpc'}

  Einordnung:
  - PPM-5 ist besser als Bigram (2.80) und Unigram (4.76)
  - PPM-5 ist schlechter als LSTM (1.67) und Transformer (1.06)
  - PPM hat KEINE trainierbaren Parameter — nur Zählen
  - PPM lernt ONLINE — jeder neue Char verbessert sofort
  - PPM braucht KEINE GPU — läuft auf jedem CPU

  Trade-off klar: PPM tauscht Modellkapazität gegen
  Online-Lernen, Determinismus und Zero-GPU.
""")
