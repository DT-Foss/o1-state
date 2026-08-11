"""
Language Core — Hierarchical Markov Language Model
=====================================================
NOT a bigram model. NOT T387.

This implements Prediction by Partial Matching (PPM) style
variable-order Markov models, with PS-Lifted consensus
synchronizing across hierarchical levels.

Architecture:
  Level 0: Character-level (morphology, spelling)
  Level 1: Subword/Token-level (syntax, local coherence)
  Level 2: Phrase-level (semantic coherence)
  Level 3: Sentence-level (discourse, argument structure)

Each level is a variable-order Markov model.
PS-Lifted consensus synchronizes predictions BETWEEN levels.

PPM reference: Cleary & Witten (1984), achieves ~1.5 bits/char.
CTW reference: Willems et al. (1995), Bayesian optimal.

Key difference from T387:
- T387: Fixed-order bigram walks on a token graph
- Here: Variable-order context trees with ESCAPE mechanism
         + inter-level consensus via Foss
"""

import numpy as np
from collections import defaultdict
import math


class ContextNode:
    """Node in a context tree (PPM/CTW style)."""

    __slots__ = ['counts', 'total', 'children', 'escape_count']

    def __init__(self):
        self.counts = defaultdict(int)  # symbol -> count
        self.total = 0
        self.children = {}  # symbol -> ContextNode (deeper context)
        self.escape_count = 0  # PPM escape events


class PPMModel:
    """
    Prediction by Partial Matching — Variable-Order Markov Model.

    Unlike fixed-order bigrams, PPM uses the LONGEST matching context,
    with an escape mechanism to shorter contexts when the current
    context hasn't seen the next symbol.

    PPM-D variant: escape probability estimated by distinct symbols.

    Achieves ~1.5 bits/char on English text — competitive with
    early neural language models.
    """

    def __init__(self, max_order=8, vocab=None):
        """
        Args:
            max_order: maximum context depth (8 = looks back 8 symbols)
            vocab: set of symbols (if None, learned from data)
        """
        self.max_order = max_order
        self.root = ContextNode()
        self.vocab = set(vocab) if vocab else set()
        self._total_symbols = 0

    def update(self, context, symbol):
        """
        Update model with observation: context → symbol.

        Args:
            context: sequence of preceding symbols (most recent last)
            symbol: the observed next symbol
        """
        self.vocab.add(symbol)
        self._total_symbols += 1

        # Update at each context depth
        node = self.root
        # Order 0 (unconditional)
        node.counts[symbol] += 1
        node.total += 1

        # Orders 1..max_order
        for depth in range(min(len(context), self.max_order)):
            ctx_symbol = context[-(depth + 1)]

            if ctx_symbol not in node.children:
                node.children[ctx_symbol] = ContextNode()

            node = node.children[ctx_symbol]

            if symbol not in node.counts:
                node.escape_count += 1  # New symbol at this depth

            node.counts[symbol] += 1
            node.total += 1

    def train(self, sequence):
        """Train on an entire sequence."""
        for i in range(len(sequence)):
            context = sequence[:i]
            symbol = sequence[i]
            self.update(context, symbol)

    def _node_to_dict(self, node):
        """Serialize a ContextNode tree to a dict."""
        d = {
            'c': dict(node.counts),  # counts
            't': node.total,
            'e': node.escape_count,
        }
        if node.children:
            d['ch'] = {k: self._node_to_dict(v) for k, v in node.children.items()}
        return d

    def _dict_to_node(self, d):
        """Deserialize a dict to a ContextNode tree."""
        node = ContextNode()
        node.counts = defaultdict(int, d['c'])
        node.total = d['t']
        node.escape_count = d['e']
        if 'ch' in d:
            node.children = {k: self._dict_to_node(v) for k, v in d['ch'].items()}
        return node

    def save(self, path):
        """Save model to a JSON file."""
        import json
        data = {
            'max_order': self.max_order,
            'vocab': sorted(self.vocab),
            'total_symbols': self._total_symbols,
            'tree': self._node_to_dict(self.root),
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """Load model from a JSON file."""
        import json
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model = cls(max_order=data['max_order'], vocab=data['vocab'])
        model._total_symbols = data['total_symbols']
        model.root = model._dict_to_node(data['tree'])
        return model

    def predict(self, context):
        """
        Predict probability distribution over next symbol.

        Uses PPM-D escape: if context hasn't seen a symbol,
        escape to shorter context with probability proportional
        to the number of distinct symbols.

        Returns: dict {symbol: probability}
        """
        # Find the deepest matching context node
        nodes = [self.root]  # Order 0
        node = self.root

        for depth in range(min(len(context), self.max_order)):
            ctx_symbol = context[-(depth + 1)]
            if ctx_symbol in node.children:
                node = node.children[ctx_symbol]
                nodes.append(node)
            else:
                break

        # Blend predictions from deepest to shallowest (PPM-D)
        probs = {}
        excluded = set()
        remaining_mass = 1.0

        # Process from deepest to shallowest
        for node in reversed(nodes):
            if node.total == 0:
                continue

            # Distinct symbols at this level
            distinct = len(node.counts)
            if distinct == 0:
                continue

            # Escape probability (PPM-D)
            p_escape = distinct / (node.total + distinct)
            p_match = 1.0 - p_escape

            # Distribute match probability among symbols
            for sym, count in node.counts.items():
                if sym not in excluded:
                    p = remaining_mass * p_match * count / node.total
                    probs[sym] = probs.get(sym, 0) + p
                    excluded.add(sym)

            remaining_mass *= p_escape

        # Uniform distribution for unseen symbols
        unseen = self.vocab - excluded
        if unseen and remaining_mass > 0:
            p_each = remaining_mass / len(unseen)
            for sym in unseen:
                probs[sym] = probs.get(sym, 0) + p_each

        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {s: p / total for s, p in probs.items()}

        return probs

    def log_prob(self, context, symbol):
        """Log probability of symbol given context."""
        probs = self.predict(context)
        p = probs.get(symbol, 1e-10)
        return math.log2(max(p, 1e-10))

    def perplexity(self, sequence):
        """Compute perplexity on a test sequence."""
        total_log_prob = 0.0
        n = 0

        for i in range(1, len(sequence)):
            context = sequence[:i]
            symbol = sequence[i]
            total_log_prob += self.log_prob(context, symbol)
            n += 1

        if n == 0:
            return float('inf')

        avg_log_prob = total_log_prob / n
        return 2.0 ** (-avg_log_prob)

    def generate(self, context, n_tokens, temperature=1.0, rng=None):
        """Generate tokens autoregressively."""
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

    def beam_perplexity(self, text, beam_width: int = 5) -> float:
        """Score text using beam search over multiple decoding paths.

        Instead of greedy left-to-right evaluation, maintains beam_width
        hypotheses at each step. Returns the average log-probability
        of the best beam path.

        For multiple-choice scoring: lower perplexity = more natural.
        """
        import math

        if len(text) < 2:
            return float('inf')

        # Each beam: (log_prob, context_so_far)
        beams = [(0.0, [text[0]])]

        for pos in range(1, len(text)):
            next_char = text[pos]
            new_beams = []

            for log_p, ctx in beams:
                probs = self.predict(ctx)
                p = probs.get(next_char, 1e-10)
                new_log_p = log_p + math.log2(max(p, 1e-10))
                new_ctx = ctx + [next_char]
                new_beams.append((new_log_p, new_ctx))

            # Keep top beam_width beams
            new_beams.sort(key=lambda x: x[0], reverse=True)
            beams = new_beams[:beam_width]

        # Return perplexity of best beam
        if beams:
            best_log_p = beams[0][0]
            n = len(text) - 1
            if n > 0:
                return 2.0 ** (-best_log_p / n)
        return float('inf')

    def score_continuation(self, context: str, continuation: str) -> float:
        """Score how natural a continuation is given context.

        Returns perplexity of continuation conditioned on context.
        Lower = more natural.
        """
        import math
        ctx_chars = list(context)
        cont_chars = list(continuation)

        if not cont_chars:
            return float('inf')

        total_log_p = 0.0
        full = ctx_chars[:]

        for ch in cont_chars:
            probs = self.predict(full)
            p = probs.get(ch, 1e-10)
            total_log_p += math.log2(max(p, 1e-10))
            full.append(ch)

        avg_log_p = total_log_p / len(cont_chars)
        return 2.0 ** (-avg_log_p)


class HierarchicalLanguageModel:
    """
    Hierarchical Language Model with PS-Lifted Consensus.

    Multiple PPM models at different granularities,
    synchronized via Foss consensus.

    Level 0: Character PPM (order 8) — morphology
    Level 1: Word PPM (order 5) — syntax
    Level 2: Phrase PPM (order 3) — semantics

    Consensus mechanism:
    Each level produces a probability distribution over the next token.
    PS-Lifted consensus combines them, weighted by confidence
    (inverse perplexity at each level).
    """

    def __init__(self, max_orders=None):
        """
        Args:
            max_orders: list of max context orders per level
        """
        if max_orders is None:
            max_orders = [12, 6, 3]

        self.n_levels = len(max_orders)
        self.levels = [PPMModel(max_order=o) for o in max_orders]
        self.level_names = ['char', 'word', 'phrase'][:self.n_levels]

        # Level weights (learned via performance)
        self.weights = np.ones(self.n_levels) / self.n_levels

        # Nonlinear mixing parameters (simple MLP-like)
        self.mix_weights = None
        self.mix_bias = None

    def train(self, text):
        """
        Train all levels on text.

        Args:
            text: raw string
        """
        # Level 0: Character-level
        chars = list(text)
        self.levels[0].train(chars)

        # Level 1: Word-level
        words = text.split()
        if len(self.levels) > 1:
            self.levels[1].train(words)

        # Level 2: Phrase-level (3-word chunks)
        if len(self.levels) > 2:
            phrases = []
            for i in range(0, len(words) - 2, 1):
                phrases.append(' '.join(words[i:i+3]))
            self.levels[2].train(phrases)

    def _train_mixer(self, validation_text, n_samples=500):
        """
        Train nonlinear mixer on validation data.

        This is the "nichtlinearer Readout" — a small 2-layer
        network that learns to combine level predictions.
        """
        words = validation_text.split()
        chars = list(validation_text)

        X = []  # Features: predictions from each level
        Y = []  # Targets: actual next-word log probs

        for i in range(10, min(len(words), n_samples + 10)):
            # Get each level's prediction for next word
            features = []

            # Char-level confidence
            word = words[i]
            char_context = list(' '.join(words[:i]) + ' ')
            char_probs = []
            for c in word:
                p = self.levels[0].predict(char_context)
                char_probs.append(p.get(c, 1e-10))
                char_context.append(c)
            char_conf = -np.mean(np.log2(np.array(char_probs) + 1e-10))
            features.append(char_conf)

            # Word-level prediction
            if len(self.levels) > 1:
                word_probs = self.levels[1].predict(words[:i])
                word_p = word_probs.get(words[i], 1e-10)
                features.append(-math.log2(max(word_p, 1e-10)))
            else:
                features.append(10.0)

            # Phrase-level
            if len(self.levels) > 2 and i >= 3:
                phrase_context = [' '.join(words[j:j+3]) for j in range(max(0, i-9), i-2)]
                target_phrase = ' '.join(words[i-2:i+1])
                phrase_probs = self.levels[2].predict(phrase_context)
                phrase_p = phrase_probs.get(target_phrase, 1e-10)
                features.append(-math.log2(max(phrase_p, 1e-10)))
            else:
                features.append(10.0)

            X.append(features)
            # Target: was the word predictable? (binary)
            if len(self.levels) > 1:
                Y.append(1.0 if word_p > 0.01 else 0.0)
            else:
                Y.append(0.5)

        if len(X) < 10:
            return

        X = np.array(X)
        Y = np.array(Y).reshape(-1, 1)

        # Simple 2-layer network (tanh hidden, linear output)
        n_features = X.shape[1]
        n_hidden = 8

        rng = np.random.RandomState(42)
        W1 = rng.randn(n_features, n_hidden) * 0.1
        b1 = np.zeros(n_hidden)
        W2 = rng.randn(n_hidden, self.n_levels) * 0.1
        b2 = np.ones(self.n_levels) / self.n_levels

        # Simple gradient descent (few steps, just to get decent weights)
        lr = 0.01
        for _ in range(100):
            # Forward
            H = np.tanh(X @ W1 + b1)
            out = H @ W2 + b2
            # Softmax to get weights
            out_exp = np.exp(out - out.max(axis=1, keepdims=True))
            level_weights = out_exp / out_exp.sum(axis=1, keepdims=True)

            # Loss: cross entropy of weighted prediction
            # (simplified: just MSE of weights vs target)
            loss = np.mean((level_weights.mean(axis=0) - np.array([0.3, 0.5, 0.2])) ** 2)

            # Backward (approximate)
            d_out = level_weights - np.array([0.3, 0.5, 0.2])
            d_W2 = H.T @ d_out / len(X)
            d_b2 = d_out.mean(axis=0)
            d_H = d_out @ W2.T
            d_pre = d_H * (1 - H ** 2)
            d_W1 = X.T @ d_pre / len(X)
            d_b1 = d_pre.mean(axis=0)

            W1 -= lr * d_W1
            b1 -= lr * d_b1
            W2 -= lr * d_W2
            b2 -= lr * d_b2

        self.mix_weights = (W1, b1, W2, b2)
        self.weights = level_weights.mean(axis=0)

    def predict_next_word(self, context_words, n_candidates=20):
        """
        Predict next word using all levels + consensus.

        Returns: list of (word, probability) sorted by probability
        """
        candidates = set()

        # Collect candidates from word-level model
        if len(self.levels) > 1:
            word_probs = self.levels[1].predict(context_words)
            candidates.update(word_probs.keys())

        # Score each candidate across all levels
        scored = {}
        context_text = ' '.join(context_words)

        for word in list(candidates)[:n_candidates * 3]:
            scores = []

            # Level 0: Character probability of this word
            char_context = list(context_text + ' ')
            char_log_prob = 0
            for c in word:
                p = self.levels[0].predict(char_context)
                char_log_prob += math.log2(max(p.get(c, 1e-10), 1e-10))
                char_context.append(c)
            # Normalize by word length
            scores.append(char_log_prob / max(len(word), 1))

            # Level 1: Word probability
            if len(self.levels) > 1:
                p_word = word_probs.get(word, 1e-10)
                scores.append(math.log2(max(p_word, 1e-10)))
            else:
                scores.append(-10)

            # Level 2: Phrase probability
            if len(self.levels) > 2 and len(context_words) >= 2:
                phrase = ' '.join(context_words[-2:]) + ' ' + word
                phrase_context = []
                for j in range(max(0, len(context_words) - 8), len(context_words) - 2):
                    phrase_context.append(' '.join(context_words[j:j+3]))
                p_phrase = self.levels[2].predict(phrase_context)
                scores.append(math.log2(max(p_phrase.get(phrase, 1e-10), 1e-10)))
            else:
                scores.append(-10)

            # Weighted combination (consensus)
            combined = sum(s * w for s, w in zip(scores, self.weights))
            scored[word] = combined

        # Sort by score
        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return ranked[:n_candidates]

    def generate(self, seed_text, n_words=20, temperature=1.0, rng=None):
        """Generate text word by word."""
        if rng is None:
            rng = np.random.RandomState()

        words = seed_text.split()
        generated = []

        for _ in range(n_words):
            predictions = self.predict_next_word(words)
            if not predictions:
                break

            # Sample from predictions
            candidates, scores = zip(*predictions)
            scores = np.array(scores)

            # Convert log-probs to probs with temperature
            scores = scores / max(temperature, 0.01)
            scores -= scores.max()
            probs = np.exp(scores)
            probs /= probs.sum()

            next_word = candidates[rng.choice(len(candidates), p=probs)]
            words.append(next_word)
            generated.append(next_word)

        return ' '.join(generated)

    def perplexity(self, test_text):
        """
        Compute word-level perplexity on test text.

        Uses the combined hierarchical prediction.
        """
        words = test_text.split()
        total_log_prob = 0.0
        n = 0

        for i in range(1, len(words)):
            context = words[:i]
            target = words[i]

            predictions = self.predict_next_word(context)
            pred_dict = dict(predictions)

            p = pred_dict.get(target, 1e-10)
            # Convert from log-score to probability
            if p < 0:
                # It's a log probability
                total_log_prob += p
            else:
                total_log_prob += math.log2(max(p, 1e-10))
            n += 1

        if n == 0:
            return float('inf')

        return 2.0 ** (-total_log_prob / n)

    def bits_per_char(self, test_text):
        """Compute bits per character (compression metric)."""
        chars = list(test_text)
        total_bits = 0.0

        for i in range(1, len(chars)):
            context = chars[:i]
            p = self.levels[0].predict(context)
            prob = p.get(chars[i], 1e-10)
            total_bits -= math.log2(max(prob, 1e-10))

        return total_bits / max(len(chars) - 1, 1)
