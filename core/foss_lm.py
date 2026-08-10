"""
Foss Language Model — PS-Lifted Consensus over PPM Context Trees
=================================================================
PPM data structure (exact contexts, no hash collisions) with
Foss consensus mathematics (topology-steered fusion, not escape).

Architecture:
  Layer 1: 4 PPM context trees at different max orders (3, 5, 8, 12)
           + 1 word-level Markov chain
           Each tree stores EXACT contexts via trie (no hashing).
  Layer 2: PS-Lifted Consensus fuses predictions via barbell topology
           Confident chains dominate. Topology controls information flow.
           Replaces PPM's ad-hoc escape mechanism.
  Layer 3: Z₂ momentum — directional prediction inertia
           Two momentum vectors (pos/neg) track prediction direction.

Research results used:
  T391: 26× consensus speedup on barbell topologies
  T385: PS-Lifted γ ~ m^{-0.381}
  Foss Gap Theorem: linear Cheeger improvement via Fiedler lifting
"""

import numpy as np
from collections import defaultdict
import math
import re
import ast

from .consensus import FossConsensus


class ContextNode:
    """Node in a PPM context tree. Exact trie — no hash collisions."""
    __slots__ = ['counts', 'total', 'children', 'n_distinct']

    def __init__(self):
        self.counts = defaultdict(int)
        self.total = 0
        self.children = {}
        self.n_distinct = 0  # number of distinct symbols seen


class TrieLM:
    """Single context tree language model at a fixed max order.

    Uses PPM trie structure for EXACT context matching.
    No hashing — every context is stored precisely.
    Kneser-Ney smoothing at each level.
    """

    def __init__(self, max_order):
        self.max_order = max_order
        self.root = ContextNode()
        self.vocab = set()
        self._continuation_counts = defaultdict(int)
        self._total_continuations = 0

    def update(self, context, symbol):
        """Update model with context → symbol observation."""
        self.vocab.add(symbol)

        # Update root (order 0)
        if symbol not in self.root.counts:
            self.root.n_distinct += 1
            self._continuation_counts[symbol] += 1
            self._total_continuations += 1
        self.root.counts[symbol] += 1
        self.root.total += 1

        # Update deeper orders
        node = self.root
        for depth in range(min(len(context), self.max_order)):
            ctx_sym = context[-(depth + 1)]
            if ctx_sym not in node.children:
                node.children[ctx_sym] = ContextNode()
            node = node.children[ctx_sym]
            if symbol not in node.counts:
                node.n_distinct += 1
            node.counts[symbol] += 1
            node.total += 1

    def train(self, sequence):
        """Train on a sequence. Uses sliding window (no copies)."""
        for i in range(len(sequence)):
            # Pass full sequence + index instead of slicing (avoids O(n²) copies)
            self._update_fast(sequence, i)

    def _update_fast(self, sequence, pos):
        """Update with context ending at pos. Zero-copy."""
        symbol = sequence[pos]
        self.vocab.add(symbol)

        # Update root (order 0)
        if symbol not in self.root.counts:
            self.root.n_distinct += 1
            self._continuation_counts[symbol] += 1
            self._total_continuations += 1
        self.root.counts[symbol] += 1
        self.root.total += 1

        # Update deeper orders — walk backwards from pos
        node = self.root
        max_depth = min(pos, self.max_order)
        for depth in range(max_depth):
            ctx_sym = sequence[pos - depth - 1]
            if ctx_sym not in node.children:
                node.children[ctx_sym] = ContextNode()
            node = node.children[ctx_sym]
            if symbol not in node.counts:
                node.n_distinct += 1
            node.counts[symbol] += 1
            node.total += 1

    def predict(self, context):
        """Predict using deepest matching context with Kneser-Ney smoothing.

        Unlike PPM escape, this returns the distribution from the
        DEEPEST matching context only. The PS-Lifted Consensus in
        FossLanguageModel handles multi-order fusion.
        """
        # Find deepest matching node
        node = self.root
        for depth in range(min(len(context), self.max_order)):
            ctx_sym = context[-(depth + 1)]
            if ctx_sym in node.children:
                node = node.children[ctx_sym]
            else:
                break

        if node.total == 0:
            # Fall back to continuation distribution
            if self._total_continuations > 0:
                return {s: self._continuation_counts.get(s, 1) / self._total_continuations
                        for s in self.vocab}
            if self.vocab:
                return {s: 1.0 / len(self.vocab) for s in self.vocab}
            return {}

        # Kneser-Ney smoothing
        D = 0.75
        n_nonzero = node.n_distinct
        lam = D * n_nonzero / node.total if node.total > 0 else 0

        total_cont = self._total_continuations or len(self.vocab)
        probs = {}
        for s in self.vocab:
            c = node.counts.get(s, 0)
            p_main = max(c - D, 0) / node.total
            p_cont = self._continuation_counts.get(s, 1) / total_cont
            probs[s] = p_main + lam * p_cont

        return probs

    def prune(self, min_count=2):
        """Remove nodes with total count <= min_count. Returns nodes removed."""
        removed = [0]

        def _prune_node(node):
            dead_keys = []
            for key, child in node.children.items():
                _prune_node(child)
                if child.total <= min_count and not child.children:
                    dead_keys.append(key)
            for key in dead_keys:
                del node.children[key]
                removed[0] += 1

        _prune_node(self.root)
        return removed[0]

    def count_nodes(self):
        """Count total nodes in trie."""
        count = [0]

        def _count(node):
            count[0] += 1
            for child in node.children.values():
                _count(child)

        _count(self.root)
        return count[0]


def _defaultdict_int():
    """Top-level function for pickle-safe defaultdict(defaultdict(int))."""
    return defaultdict(int)


class WordMarkovLM:
    """Word-level Markov chain."""

    def __init__(self, order=3):
        self.order = order
        self.transitions = defaultdict(_defaultdict_int)
        self.state_totals = defaultdict(int)
        self.vocab = set()

    def _ctx_key(self, words):
        return tuple(words[-self.order:])

    def train_words(self, words):
        for i in range(len(words)):
            ctx = self._ctx_key(words[max(0, i - self.order):i])
            w = words[i]
            self.vocab.add(w)
            self.transitions[ctx][w] += 1
            self.state_totals[ctx] += 1

    def predict_words(self, context_words):
        ctx = self._ctx_key(context_words)
        counts = self.transitions.get(ctx, {})
        total = self.state_totals.get(ctx, 0)
        if total == 0:
            return {}
        alpha = 0.01
        v = max(len(self.vocab), 1)
        denom = total + alpha * v
        return {w: (c + alpha) / denom for w, c in counts.items()}


class FossLanguageModel:
    """
    Foss Language Model: PS-Lifted Consensus over PPM Context Trees.

    Layer 1: 5 chains (PPM trie orders 3, 5, 8, 12 + word-level)
    Layer 2: PS-Lifted Consensus fusion (topology × confidence)
    Layer 3: Z₂ momentum (directional inertia)
    """

    N_CHAINS = 4  # 3 character + 1 word
    CHAR_ORDERS = [3, 5, 8]

    def __init__(self, momentum_decay=0.95):
        # Layer 1: PPM context trees (exact, no hashing)
        self.chains = [TrieLM(max_order=o) for o in self.CHAR_ORDERS]
        self.word_chain = WordMarkovLM(order=3)

        # Layer 2: PS-Lifted Consensus on chain topology
        # Linear chain: order3 → order5 → order8 → word
        A = np.zeros((self.N_CHAINS, self.N_CHAINS))
        A[0, 1] = A[1, 0] = 1
        A[1, 2] = A[2, 1] = 1
        A[2, 3] = A[3, 2] = 1
        self.consensus = FossConsensus(A)

        # Precompute converged consensus weights (O(1) fusion)
        mc = self.consensus.mc
        T_lift = mc.T_lift.copy()
        T_power = np.linalg.matrix_power(T_lift, 200)
        n = mc.n
        embed = np.zeros((n, 2 * n))
        for i in range(n):
            embed[i, i] = 1.0
            embed[i, n + i] = 1.0
        projected = embed @ T_power
        self._fusion_weights = np.zeros(n)
        for j in range(n):
            self._fusion_weights[j] = (projected[0, j] + projected[0, n + j]) / 2

        # Layer 3: Z₂ momentum
        self._momentum_pos = {}
        self._momentum_neg = {}
        self._momentum_decay = momentum_decay
        self._last_distribution = None

        # Unified vocab
        self.vocab = set()

    def train(self, text):
        """Train all chains on text."""
        chars = list(text)
        for chain in self.chains:
            chain.train(chars)

        words = re.findall(r'\b\w+\b', text.lower())
        self.word_chain.train_words(words)

        self.vocab = set()
        for chain in self.chains:
            self.vocab |= chain.vocab

    def _get_word_position(self, context):
        """Current position within current word."""
        partial = []
        for c in reversed(context):
            if c == ' ' or not (c.isalnum() or c == "'"):
                break
            partial.append(c)
        partial.reverse()
        return len(partial), ''.join(partial).lower()

    def predict(self, context):
        """Predict next character using full Foss pipeline."""
        if not self.vocab:
            return {}

        symbols = sorted(self.vocab)
        n_sym = len(symbols)
        sym_idx = {s: i for i, s in enumerate(symbols)}

        # ── Layer 1: Independent chain predictions ──
        chain_probs = np.zeros((self.N_CHAINS, n_sym))

        for ci, chain in enumerate(self.chains):
            probs = chain.predict(context)
            for s, p in probs.items():
                if s in sym_idx:
                    chain_probs[ci, sym_idx[s]] = p

        # Word chain: position-aware word→char bridge
        pos_in_word, partial_word = self._get_word_position(context)
        words = re.findall(r'\b\w+\b', ''.join(context[-80:]).lower())
        word_probs = self.word_chain.predict_words(words)

        if word_probs and pos_in_word >= 0:
            for word, wp in word_probs.items():
                if pos_in_word < len(word):
                    if pos_in_word == 0 or word.startswith(partial_word):
                        next_char = word[pos_in_word]
                        if next_char in sym_idx:
                            chain_probs[self.N_CHAINS - 1, sym_idx[next_char]] += wp
                        if pos_in_word == 0 and next_char.upper() in sym_idx:
                            chain_probs[self.N_CHAINS - 1, sym_idx[next_char.upper()]] += wp * 0.3
                elif pos_in_word == len(word):
                    if ' ' in sym_idx:
                        chain_probs[self.N_CHAINS - 1, sym_idx[' ']] += wp

        # Normalize all rows
        for ci in range(self.N_CHAINS):
            row_sum = chain_probs[ci].sum()
            if row_sum > 0:
                chain_probs[ci] /= row_sum
            else:
                chain_probs[ci] = 1.0 / n_sym

        # ── Layer 2: PS-Lifted Consensus ──
        # Topology weights × per-chain confidence (inverse entropy)
        max_ent = math.log(n_sym) if n_sym > 1 else 1.0
        confidence = np.zeros(self.N_CHAINS)
        for ci in range(self.N_CHAINS):
            b = chain_probs[ci]
            ent = -np.sum(b * np.log(b + 1e-10))
            confidence[ci] = max(0.01, 1.0 - ent / max_ent)
        mod_weights = self._fusion_weights * confidence
        total_w = mod_weights.sum()
        if total_w > 0:
            mod_weights /= total_w
        combined = mod_weights @ chain_probs

        # Normalize
        total = combined.sum()
        if total > 0:
            combined /= total

        # ── Layer 3: Z₂ momentum ──
        for s in self.vocab:
            if s in sym_idx:
                net_mom = self._momentum_pos.get(s, 0) - self._momentum_neg.get(s, 0)
                if abs(net_mom) > 0.001:
                    combined[sym_idx[s]] *= (1.0 + net_mom)
        total = combined.sum()
        if total > 0:
            combined /= total

        return {symbols[i]: combined[i] for i in range(n_sym)}

    def _update_momentum(self, predicted_symbol, current_dist=None):
        """Update Z₂ momentum with directional tracking."""
        for d in (self._momentum_pos, self._momentum_neg):
            for s in list(d.keys()):
                d[s] *= self._momentum_decay
                if abs(d[s]) < 0.001:
                    del d[s]

        if predicted_symbol and current_dist is not None and self._last_distribution is not None:
            for s in current_dist:
                p_now = current_dist.get(s, 0)
                p_prev = self._last_distribution.get(s, 0)
                delta = p_now - p_prev
                if abs(delta) > 0.01:
                    if delta > 0:
                        self._momentum_pos[s] = self._momentum_pos.get(s, 0) + delta * 0.5
                    else:
                        self._momentum_neg[s] = self._momentum_neg.get(s, 0) + abs(delta) * 0.5

        if current_dist is not None:
            self._last_distribution = dict(current_dist)

    def log_prob(self, context, symbol):
        """Log2 probability of symbol given context."""
        probs = self.predict(context)
        p = probs.get(symbol, 1e-10)
        return math.log2(max(p, 1e-10))

    def perplexity(self, sequence):
        """Compute perplexity on a test sequence."""
        total_log_prob = 0.0
        n = 0
        for i in range(1, len(sequence)):
            ctx = sequence[:i]
            sym = sequence[i]
            probs = self.predict(ctx)
            p = probs.get(sym, 1e-10)
            total_log_prob += math.log2(max(p, 1e-10))
            n += 1
            self._update_momentum(sym, probs)
        if n == 0:
            return float('inf')
        return 2.0 ** (-total_log_prob / n)

    def score_continuation(self, context, continuation):
        """Score how natural a continuation is (lower = more natural)."""
        full = list(context + ' ' + continuation)
        ctx_len = len(list(context)) + 1
        total_log_prob = 0.0
        n = 0
        for i in range(ctx_len, len(full)):
            ctx = full[:i]
            sym = full[i]
            probs = self.predict(ctx)
            p = probs.get(sym, 1e-10)
            total_log_prob += math.log2(max(p, 1e-10))
            n += 1
            self._update_momentum(sym, probs)
        if n == 0:
            return float('inf')
        return -total_log_prob / n

    def save(self, path):
        """Save model state."""
        import json
        data = {
            'char_orders': self.CHAR_ORDERS,
            'vocab': sorted(self.vocab),
            'momentum_decay': self._momentum_decay,
        }
        # Serialize trie structures
        def node_to_dict(node):
            d = {'c': dict(node.counts), 't': node.total, 'nd': node.n_distinct}
            if node.children:
                d['ch'] = {k: node_to_dict(v) for k, v in node.children.items()}
            return d

        data['chains'] = []
        for chain in self.chains:
            data['chains'].append({
                'order': chain.max_order,
                'tree': node_to_dict(chain.root),
            })
        data['word_chain'] = {
            'transitions': {str(k): dict(v) for k, v in self.word_chain.transitions.items()},
            'totals': {str(k): v for k, v in self.word_chain.state_totals.items()},
            'vocab': sorted(self.word_chain.vocab),
        }
        with open(path, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """Load model from file."""
        import json
        with open(path) as f:
            data = json.load(f)

        model = cls()
        model.vocab = set(data['vocab'])

        def dict_to_node(d):
            node = ContextNode()
            node.counts = defaultdict(int, d['c'])
            node.total = d['t']
            node.n_distinct = d.get('nd', len(d['c']))
            if 'ch' in d:
                node.children = {k: dict_to_node(v) for k, v in d['ch'].items()}
            return node

        for ci, chain_data in enumerate(data.get('chains', [])):
            if ci < len(model.chains):
                model.chains[ci].root = dict_to_node(chain_data['tree'])
                model.chains[ci].vocab = model.vocab.copy()

        if 'word_chain' in data:
            wc = data['word_chain']
            model.word_chain.vocab = set(wc.get('vocab', []))
            for k, v in wc['transitions'].items():
                model.word_chain.transitions[ast.literal_eval(k)] = defaultdict(int, v)
            model.word_chain.state_totals = defaultdict(int,
                {ast.literal_eval(k): v for k, v in wc['totals'].items()})

        return model

    def generate(self, context, n_tokens, temperature=1.0, rng=None):
        """Generate tokens autoregressively using PS-Lifted consensus."""
        if rng is None:
            rng = np.random.RandomState()

        generated = list(context)

        for _ in range(n_tokens):
            probs = self.predict(generated)
            if not probs:
                break

            symbols = list(probs.keys())
            weights = np.array([probs[s] for s in symbols])

            # Temperature scaling
            if temperature != 1.0:
                log_w = np.log(weights + 1e-10) / temperature
                log_w -= log_w.max()
                weights = np.exp(log_w)

            weights /= weights.sum()

            next_sym = symbols[rng.choice(len(symbols), p=weights)]
            generated.append(next_sym)

        return generated[len(context):]

    def update_context(self, chars):
        """Feed characters into model context (online learning)."""
        for i, ch in enumerate(chars):
            ctx = chars[:i]
            for chain in self.chains:
                chain.update(ctx, ch)
            self.vocab.add(ch)

    @property
    def max_order(self):
        """Compatibility with PPMModel interface."""
        return max(self.CHAR_ORDERS)
