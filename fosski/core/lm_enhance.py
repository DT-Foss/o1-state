"""
Language Model Enhancements — Pre-Transformer Techniques
==========================================================
Three proven techniques that improve PPM prediction quality:

1. Cache Language Model (Kuhn & De Mori, 1990)
   - Dynamic unigram that adapts to current text
   - If "Kubernetes" appears, "pod"/"container" get boosted
   - Reduces cross-domain BPC by ~20%

2. Modified Kneser-Ney Smoothing (Chen & Goodman, 1998)
   - State-of-the-art smoothing until 2013
   - Uses continuation counts instead of raw frequencies
   - "San" in "San Francisco" should NOT boost "San" as a standalone word
   - 10-15% better perplexity than PPM-D escape

3. Trigger Pairs (Rosenfeld, 1996)
   - Long-range word associations beyond PPM's context window
   - "democracy" triggers "voting"/"election" even 100 words later
   - Bridges the gap between character-level PPM and discourse coherence

All three wrap around the existing PPMModel without modifying it.
"""

import math
from collections import defaultdict, Counter
from typing import Dict, List, Optional, Tuple, Set


# ============================================================
# 1. Cache Language Model
# ============================================================

class CacheLM:
    """
    Cache Language Model (Kuhn & De Mori, 1990).

    Maintains a sliding window of recently seen symbols/words.
    Symbols that appeared recently get a probability boost.

    This captures topic persistence: if "Kubernetes" appeared
    recently, related terms like "pod", "container", "namespace"
    are more likely even if the PPM hasn't trained on them.

    Usage:
        cache = CacheLM(window_size=2000)
        cache.observe('k')
        cache.observe('u')
        ...
        boost = cache.predict()  # {symbol: prob based on recency}
    """

    def __init__(self, window_size: int = 2000, decay: float = 0.95):
        """
        Args:
            window_size: how many recent symbols to remember
            decay: exponential decay factor for older symbols
        """
        self.window_size = window_size
        self.decay = decay
        self._window: List[str] = []
        self._counts: Dict[str, float] = defaultdict(float)
        self._total: float = 0.0

    def observe(self, symbol: str):
        """Record a symbol observation."""
        self._window.append(symbol)
        self._counts[symbol] += 1.0
        self._total += 1.0

        # Trim window
        if len(self._window) > self.window_size:
            old = self._window.pop(0)
            self._counts[old] -= 1.0
            self._total -= 1.0
            if self._counts[old] <= 0:
                del self._counts[old]

    def observe_sequence(self, symbols: List[str]):
        """Record a sequence of symbols."""
        for s in symbols:
            self.observe(s)

    def predict(self) -> Dict[str, float]:
        """
        Get cache-based probability distribution.

        Recency-weighted: more recent occurrences count more.
        """
        if not self._window:
            return {}

        # Recency-weighted counts
        weighted: Dict[str, float] = defaultdict(float)
        total = 0.0
        n = len(self._window)

        for i, sym in enumerate(self._window):
            # More recent = higher weight
            weight = self.decay ** (n - 1 - i)
            weighted[sym] += weight
            total += weight

        if total <= 0:
            return {}

        return {sym: w / total for sym, w in weighted.items()}

    def clear(self):
        """Clear the cache."""
        self._window.clear()
        self._counts.clear()
        self._total = 0.0


class WordCacheLM:
    """
    Word-level cache for topic persistence.

    Tracks recently seen words and their co-occurrence patterns.
    When "Kubernetes" appears, boosts all words that co-occurred
    with "Kubernetes" in the recent window.
    """

    def __init__(self, window_size: int = 200, co_window: int = 10):
        """
        Args:
            window_size: max words in cache
            co_window: co-occurrence window size for associations
        """
        self.window_size = window_size
        self.co_window = co_window
        self._words: List[str] = []
        self._cooccur: Dict[str, Counter] = defaultdict(Counter)

    def observe_word(self, word: str):
        """Record a word and update co-occurrences."""
        word_lower = word.lower()
        self._words.append(word_lower)

        # Update co-occurrence with recent words
        start = max(0, len(self._words) - self.co_window - 1)
        for i in range(start, len(self._words) - 1):
            other = self._words[i]
            if other != word_lower:
                self._cooccur[word_lower][other] += 1
                self._cooccur[other][word_lower] += 1

        # Trim
        if len(self._words) > self.window_size:
            self._words = self._words[-self.window_size:]

    def observe_text(self, text: str):
        """Record all words in a text."""
        for word in text.split():
            clean = ''.join(c for c in word.lower() if c.isalnum())
            if clean:
                self.observe_word(clean)

    def get_topic_boost(self) -> Dict[str, float]:
        """
        Get topic-based word probability boost.

        Words that co-occurred with recently seen words get boosted.
        """
        if not self._words:
            return {}

        # Count recent word frequencies
        recent = self._words[-min(50, len(self._words)):]
        recent_counts = Counter(recent)

        # Boost words associated with recent frequent words
        boost: Dict[str, float] = defaultdict(float)
        for word, count in recent_counts.most_common(20):
            if word in self._cooccur:
                for associated, co_count in self._cooccur[word].most_common(30):
                    boost[associated] += co_count * count

        # Normalize
        total = sum(boost.values())
        if total > 0:
            return {w: b / total for w, b in boost.items()}
        return {}

    def clear(self):
        self._words.clear()
        self._cooccur.clear()


# ============================================================
# 2. Modified Kneser-Ney Smoothing
# ============================================================

class KneserNeyModel:
    """
    Modified Kneser-Ney Smoothing (Chen & Goodman, 1998).

    The key insight: a word's backoff probability should be based
    on how many DIFFERENT contexts it appears in (continuation count),
    not its raw frequency.

    "San" appears very frequently, but almost always before "Francisco".
    Its continuation count is LOW → low backoff probability.
    "the" appears in many different contexts → high continuation count.

    Three discount values D1, D2, D3+ (instead of single D in original KN):
    - D1: discount for words seen exactly once
    - D2: discount for words seen exactly twice
    - D3: discount for words seen 3+ times

    This replaces PPM-D's simple escape probability with a
    mathematically superior smoothing method.
    """

    def __init__(self, max_order: int = 5):
        self.max_order = max_order

        # n-gram counts: order → (context_tuple → Counter of next words)
        self._counts: Dict[int, Dict[tuple, Counter]] = {
            i: defaultdict(Counter) for i in range(max_order + 1)
        }

        # Continuation counts for KN lower-order distributions
        # How many unique left-contexts does word w appear in?
        self._continuation: Dict[int, Counter] = {
            i: Counter() for i in range(max_order + 1)
        }

        # Total distinct n-grams at each order (for normalization)
        self._n1: Dict[int, int] = defaultdict(int)  # count of n-grams appearing once
        self._n2: Dict[int, int] = defaultdict(int)  # count appearing twice
        self._n3plus: Dict[int, int] = defaultdict(int)  # count appearing 3+

        # Discount values (computed after training)
        self._D: Dict[int, Tuple[float, float, float]] = {}

        self.vocab: Set[str] = set()
        self._trained = False

    def train(self, tokens: List[str]):
        """
        Train on a token sequence (words or characters).

        Must call finalize() after all training data is processed.
        """
        self.vocab.update(tokens)

        for i in range(len(tokens)):
            word = tokens[i]

            # Order 0 (unigram)
            self._counts[0][()][word] += 1

            # Orders 1..max_order
            for order in range(1, self.max_order + 1):
                if i < order:
                    continue
                context = tuple(tokens[i - order:i])
                prev_count = self._counts[order][context][word]
                self._counts[order][context][word] += 1

                # Track continuation counts (unique left-contexts for word)
                if prev_count == 0:
                    self._continuation[order][word] += 1

    def finalize(self):
        """
        Compute discount values from training data.

        Must be called after all train() calls and before predict().
        """
        for order in range(self.max_order + 1):
            n1 = 0
            n2 = 0
            n3plus = 0
            for context_counts in self._counts[order].values():
                for count in context_counts.values():
                    if count == 1:
                        n1 += 1
                    elif count == 2:
                        n2 += 1
                    else:
                        n3plus += 1

            self._n1[order] = n1
            self._n2[order] = n2
            self._n3plus[order] = n3plus

            # Compute Y (common factor)
            total_n12 = n1 + 2 * n2
            if total_n12 == 0:
                self._D[order] = (0.0, 0.0, 0.0)
                continue

            Y = n1 / total_n12

            # Three discount values
            D1 = 1.0 - 2.0 * Y * (n2 / max(n1, 1))
            D2 = 2.0 - 3.0 * Y * (self._n3plus.get(order, 1) / max(n2, 1))
            D3 = 3.0 - 4.0 * Y * (self._n3plus.get(order, 1) / max(self._n3plus.get(order, 1), 1))

            # Clamp to reasonable range
            D1 = max(0.0, min(D1, 0.99))
            D2 = max(D1, min(D2, 0.99))
            D3 = max(D2, min(D3, 0.99))

            self._D[order] = (D1, D2, D3)

        self._trained = True

    def predict(self, context: List[str]) -> Dict[str, float]:
        """
        Predict probability distribution over next token.

        Uses Modified Kneser-Ney interpolation from highest to lowest order.
        """
        if not self._trained:
            self.finalize()

        return self._kn_predict(context, self.max_order)

    def _kn_predict(self, context: List[str], order: int) -> Dict[str, float]:
        """Recursive Kneser-Ney prediction."""
        if order == 0:
            return self._unigram_predict()

        ctx = tuple(context[-order:]) if len(context) >= order else tuple(context)

        if ctx not in self._counts[order] or not self._counts[order][ctx]:
            # Context not seen → fall back to lower order
            return self._kn_predict(context, order - 1)

        context_counts = self._counts[order][ctx]
        total = sum(context_counts.values())

        if total == 0:
            return self._kn_predict(context, order - 1)

        D1, D2, D3 = self._D.get(order, (0.5, 0.7, 0.9))

        probs: Dict[str, float] = {}
        n1_ctx = 0  # words with count 1 in this context
        n2_ctx = 0  # words with count 2
        n3_ctx = 0  # words with count 3+

        for word, count in context_counts.items():
            if count == 1:
                discount = D1
                n1_ctx += 1
            elif count == 2:
                discount = D2
                n2_ctx += 1
            else:
                discount = D3
                n3_ctx += 1

            probs[word] = max(count - discount, 0.0) / total

        # Interpolation weight (lambda)
        gamma = (D1 * n1_ctx + D2 * n2_ctx + D3 * n3_ctx) / total

        # Lower-order distribution (KN uses continuation counts, not raw)
        lower = self._kn_continuation_predict(context, order - 1) if order > 1 else self._unigram_predict()

        # Interpolate
        for word in set(list(probs.keys()) + list(lower.keys())):
            p_high = probs.get(word, 0.0)
            p_low = lower.get(word, 0.0)
            probs[word] = p_high + gamma * p_low

        return probs

    def _kn_continuation_predict(self, context: List[str], order: int) -> Dict[str, float]:
        """
        Lower-order KN distribution using continuation counts.

        Instead of raw frequency of word w, use:
        "In how many different contexts does w appear?"
        """
        if order <= 0:
            return self._unigram_predict()

        # Continuation count for each word at this order
        total_continuation = sum(self._continuation[order].values())
        if total_continuation == 0:
            return self._unigram_predict()

        probs = {}
        for word, cont_count in self._continuation[order].items():
            probs[word] = cont_count / total_continuation

        return probs

    def _unigram_predict(self) -> Dict[str, float]:
        """Unigram distribution."""
        if () not in self._counts[0]:
            # Uniform over vocab
            if self.vocab:
                p = 1.0 / len(self.vocab)
                return {w: p for w in self.vocab}
            return {}

        counts = self._counts[0][()]
        total = sum(counts.values())
        if total == 0:
            return {}

        return {w: c / total for w, c in counts.items()}

    def log_prob(self, context: List[str], word: str) -> float:
        """Log2 probability of word given context."""
        probs = self.predict(context)
        p = probs.get(word, 1e-10)
        return math.log2(max(p, 1e-10))

    def perplexity(self, tokens: List[str]) -> float:
        """Compute perplexity on token sequence."""
        total = 0.0
        n = 0
        for i in range(1, len(tokens)):
            context = tokens[:i]
            total += self.log_prob(context, tokens[i])
            n += 1
        if n == 0:
            return float('inf')
        return 2.0 ** (-total / n)


# ============================================================
# 3. Trigger Pairs — Long-Range Word Associations
# ============================================================

class TriggerModel:
    """
    Trigger Pairs (Rosenfeld, 1996).

    Captures long-range word associations that go beyond the
    PPM context window. When "democracy" appears anywhere in
    the document, "voting", "election", "parliament" become
    more likely even 100+ words later.

    This is NOT a co-occurrence matrix. It's a CONDITIONAL model:
    P(word_b | word_a appeared in document) / P(word_b)

    Only pairs with high mutual information are kept.

    Usage:
        triggers = TriggerModel()
        triggers.train(corpus_of_documents)
        triggers.finalize(min_mi=1.0)
        boost = triggers.get_boost(document_words_so_far)
    """

    def __init__(self, max_pairs: int = 50000, max_distance: int = 500):
        """
        Args:
            max_pairs: maximum trigger pairs to keep
            max_distance: maximum word distance for co-occurrence
        """
        self.max_pairs = max_pairs
        self.max_distance = max_distance

        # Training counts
        self._pair_count: Dict[Tuple[str, str], int] = defaultdict(int)
        self._word_count: Counter = Counter()
        self._n_docs = 0

        # Final trigger pairs (after finalize)
        self._triggers: Dict[str, List[Tuple[str, float]]] = {}
        self._finalized = False

    def train_document(self, words: List[str]):
        """
        Train on one document (or paragraph).

        Counts word co-occurrences within max_distance.
        """
        self._n_docs += 1
        word_set = set()
        clean = [w.lower() for w in words if len(w) > 2]

        for i, w in enumerate(clean):
            self._word_count[w] += 1
            word_set.add(w)

            # Co-occurrences within window
            for j in range(i + 1, min(i + self.max_distance, len(clean))):
                w2 = clean[j]
                if w != w2:
                    self._pair_count[(w, w2)] += 1

    def train_corpus(self, documents: List[List[str]]):
        """Train on multiple documents."""
        for doc in documents:
            self.train_document(doc)

    def finalize(self, min_mi: float = 0.5, top_k_per_trigger: int = 20,
                 min_count: int = 3):
        """
        Compute mutual information and keep top pairs.

        MI(a→b) = log2(P(b|a_appeared) / P(b))

        Args:
            min_mi: minimum mutual information to keep a pair
            top_k_per_trigger: max triggered words per trigger
            min_count: minimum word frequency to consider
        """
        if self._n_docs == 0:
            self._finalized = True
            return

        total_words = sum(self._word_count.values())
        if total_words == 0:
            self._finalized = True
            return

        # Compute MI for each pair
        scored_pairs: List[Tuple[str, str, float]] = []

        for (w_a, w_b), co_count in self._pair_count.items():
            count_a = self._word_count[w_a]
            count_b = self._word_count[w_b]

            if count_a < min_count or count_b < min_count:
                continue

            # P(b | a appeared) ≈ co_count / count_a
            p_b_given_a = co_count / count_a
            # P(b) = count_b / total
            p_b = count_b / total_words

            if p_b <= 0:
                continue

            mi = math.log2(max(p_b_given_a / p_b, 1e-10))

            if mi >= min_mi:
                scored_pairs.append((w_a, w_b, mi))

        # Sort by MI descending
        scored_pairs.sort(key=lambda x: x[2], reverse=True)
        scored_pairs = scored_pairs[:self.max_pairs]

        # Group by trigger word
        self._triggers = defaultdict(list)
        for w_a, w_b, mi in scored_pairs:
            self._triggers[w_a].append((w_b, mi))

        # Keep top-k per trigger
        for trigger in self._triggers:
            self._triggers[trigger] = sorted(
                self._triggers[trigger], key=lambda x: x[1], reverse=True
            )[:top_k_per_trigger]

        self._finalized = True

    def get_boost(self, context_words: List[str]) -> Dict[str, float]:
        """
        Get trigger-based probability boost for given context.

        Looks at all words in context, finds their trigger pairs,
        and returns a probability boost for triggered words.
        """
        if not self._finalized or not self._triggers:
            return {}

        boost: Dict[str, float] = defaultdict(float)
        seen = set(w.lower() for w in context_words)

        for word in seen:
            if word in self._triggers:
                for triggered, mi in self._triggers[word]:
                    boost[triggered] += mi

        # Normalize to probability distribution
        total = sum(boost.values())
        if total > 0:
            return {w: b / total for w, b in boost.items()}
        return {}

    def get_triggers_for(self, word: str) -> List[Tuple[str, float]]:
        """Get trigger pairs for a specific word."""
        if not self._finalized:
            self.finalize()
        return self._triggers.get(word.lower(), [])

    @property
    def n_pairs(self) -> int:
        """Total number of trigger pairs."""
        return sum(len(v) for v in self._triggers.values())


# ============================================================
# Enhanced PPM — Combines all three improvements
# ============================================================

class EnhancedLM:
    """
    Enhanced Language Model combining PPM + Cache + KN + Triggers.

    This wraps around an existing PPM model and augments its
    predictions with:
    - Cache probabilities (topic persistence)
    - Trigger pair boosts (long-range associations)

    Can also use KneserNeyModel as a drop-in replacement for
    PPM's predict() method.

    Usage:
        from core.language import PPMModel
        ppm = PPMModel(max_order=8)
        ppm.train(list(text))

        enhanced = EnhancedLM(ppm)
        enhanced.cache.observe_sequence(list(recent_text))

        # Enhanced prediction
        probs = enhanced.predict(context)
    """

    def __init__(self, lm_model=None, kn_model: Optional[KneserNeyModel] = None,
                 cache_weight: float = 0.1,
                 trigger_weight: float = 0.05,
                 kn_weight: float = 0.0):
        """
        Args:
            lm_model: existing PPMModel instance
            kn_model: KneserNeyModel (if None, PPM is used alone)
            cache_weight: weight for cache predictions (0.0-0.3)
            trigger_weight: weight for trigger predictions (0.0-0.2)
            kn_weight: weight for KN model (0 = PPM only, 1 = KN only)
        """
        self.lm = lm_model
        self.kn = kn_model
        self.cache = CacheLM()
        self.word_cache = WordCacheLM()
        self.triggers = TriggerModel()

        self.cache_weight = cache_weight
        self.trigger_weight = trigger_weight
        self.kn_weight = kn_weight

    def predict(self, context: list) -> Dict[str, float]:
        """
        Enhanced prediction combining all models.

        The base model (PPM or KN) provides the main prediction.
        Cache and triggers provide additive boosts.
        """
        # Base prediction
        base_weight = 1.0 - self.cache_weight - self.trigger_weight

        if self.kn and self.kn_weight > 0 and self.lm:
            # Interpolate PPM and KN
            ppm_probs = self.lm.predict(context) if self.lm else {}
            kn_probs = self.kn.predict(list(context))
            base_probs = self._interpolate(
                ppm_probs, kn_probs,
                1.0 - self.kn_weight, self.kn_weight
            )
        elif self.kn and self.kn_weight >= 1.0:
            base_probs = self.kn.predict(list(context))
        elif self.lm:
            base_probs = self.lm.predict(context)
        else:
            base_probs = {}

        # Cache boost
        cache_probs = self.cache.predict() if self.cache_weight > 0 else {}

        # Trigger boost (word-level, so extract words from char context)
        trigger_probs = {}
        if self.trigger_weight > 0:
            # Convert char context to words for trigger lookup
            text = ''.join(context) if context and isinstance(context[0], str) and len(context[0]) == 1 else ' '.join(str(c) for c in context)
            words = text.split()
            trigger_probs = self.triggers.get_boost(words)

        # Combine
        combined = {}
        all_symbols = set(list(base_probs.keys()) + list(cache_probs.keys()) + list(trigger_probs.keys()))

        for sym in all_symbols:
            p = (base_weight * base_probs.get(sym, 0.0) +
                 self.cache_weight * cache_probs.get(sym, 0.0) +
                 self.trigger_weight * trigger_probs.get(sym, 0.0))
            if p > 0:
                combined[sym] = p

        # Normalize
        total = sum(combined.values())
        if total > 0:
            combined = {s: p / total for s, p in combined.items()}

        return combined

    def observe(self, symbol: str):
        """Observe a symbol (updates cache)."""
        self.cache.observe(symbol)

    def observe_text(self, text: str):
        """Observe text (updates word cache and char cache)."""
        self.word_cache.observe_text(text)
        for c in text:
            self.cache.observe(c)

    @staticmethod
    def _interpolate(dist_a: Dict[str, float], dist_b: Dict[str, float],
                     weight_a: float, weight_b: float) -> Dict[str, float]:
        """Interpolate two distributions."""
        combined = {}
        all_keys = set(list(dist_a.keys()) + list(dist_b.keys()))
        for key in all_keys:
            p = weight_a * dist_a.get(key, 0.0) + weight_b * dist_b.get(key, 0.0)
            if p > 0:
                combined[key] = p
        total = sum(combined.values())
        if total > 0:
            return {k: v / total for k, v in combined.items()}
        return combined

    def log_prob(self, context: list, symbol: str) -> float:
        """Log2 probability of symbol given context."""
        probs = self.predict(context)
        p = probs.get(symbol, 1e-10)
        return math.log2(max(p, 1e-10))

    def perplexity(self, sequence: list) -> float:
        """Compute perplexity, updating cache as we go."""
        total = 0.0
        n = 0
        for i in range(1, len(sequence)):
            context = sequence[:i]
            symbol = sequence[i]
            total += self.log_prob(context, symbol)
            self.observe(symbol)
            n += 1
        if n == 0:
            return float('inf')
        return 2.0 ** (-total / n)

    def bits_per_char(self, text: str) -> float:
        """BPC with cache enabled (dynamic adaptation)."""
        chars = list(text)
        return -sum(
            self.log_prob(chars[:i], chars[i])
            for i in range(1, len(chars))
        ) / max(len(chars) - 1, 1)
