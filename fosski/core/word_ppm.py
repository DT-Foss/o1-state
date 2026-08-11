"""
Word-Level PPM (Trigram) Language Model
========================================
Scores text by word-trigram probability.
Unlike character-level PPM (order 8 = 8 chars context),
this sees 3-word context — enough for basic sentence fluency.

Training: count word trigrams from corpus.
Scoring: average log-probability of each word given 2-word context.
"""

import re
import pickle
import math
from collections import defaultdict
from typing import Dict, Tuple


class WordPPM:
    """Word-level trigram model with Kneser-Ney-like backoff."""

    def __init__(self):
        self.trigram_counts = defaultdict(int)   # (w1, w2, w3) → count
        self.bigram_counts = defaultdict(int)    # (w1, w2) → count
        self.unigram_counts = defaultdict(int)   # w → count
        self.total_words = 0
        self.vocab_size = 0

    def train(self, corpus_path: str, max_lines: int = 0):
        """Train on a text corpus (one sentence per line or continuous text)."""
        print(f"Training word PPM from {corpus_path}...")
        with open(corpus_path) as f:
            for line_num, line in enumerate(f):
                if max_lines and line_num >= max_lines:
                    break
                words = re.findall(r'\b[a-z]+\b', line.lower())
                if len(words) < 3:
                    continue

                for w in words:
                    self.unigram_counts[w] += 1
                    self.total_words += 1

                for i in range(len(words) - 1):
                    self.bigram_counts[(words[i], words[i+1])] += 1

                for i in range(len(words) - 2):
                    self.trigram_counts[(words[i], words[i+1], words[i+2])] += 1

                if (line_num + 1) % 100000 == 0:
                    print(f"  {line_num+1} lines, {self.total_words} words, "
                          f"{len(self.trigram_counts)} trigrams")

        self.vocab_size = len(self.unigram_counts)
        print(f"  Done: {self.total_words} words, {self.vocab_size} vocab, "
              f"{len(self.trigram_counts)} trigrams, "
              f"{len(self.bigram_counts)} bigrams")

    def log_prob(self, w1: str, w2: str, w3: str) -> float:
        """Log2 probability of w3 given (w1, w2) with backoff."""
        # Trigram
        tri_key = (w1, w2, w3)
        bi_key = (w1, w2)
        bi_count = self.bigram_counts.get(bi_key, 0)

        if bi_count > 0:
            tri_count = self.trigram_counts.get(tri_key, 0)
            if tri_count > 0:
                # Trigram probability with discount
                p_tri = (tri_count - 0.75) / bi_count
                return math.log2(max(p_tri, 1e-10))

        # Backoff to bigram
        bi_key2 = (w2, w3)
        uni_count = self.unigram_counts.get(w2, 0)

        if uni_count > 0:
            bi_count2 = self.bigram_counts.get(bi_key2, 0)
            if bi_count2 > 0:
                p_bi = (bi_count2 - 0.5) / uni_count
                return math.log2(max(p_bi, 1e-10))

        # Backoff to unigram
        uni = self.unigram_counts.get(w3, 0)
        if uni > 0:
            p_uni = uni / self.total_words
            return math.log2(max(p_uni, 1e-10))

        # Unknown word
        return math.log2(1.0 / (self.total_words + self.vocab_size))

    def score_text(self, text: str) -> float:
        """Score text by average word-trigram log-probability.
        Lower (less negative) = more natural."""
        words = re.findall(r'\b[a-z]+\b', text.lower())
        if len(words) < 3:
            return -20.0  # Very uncertain for short texts

        total = 0.0
        n = 0
        for i in range(2, len(words)):
            total += self.log_prob(words[i-2], words[i-1], words[i])
            n += 1

        return total / n if n > 0 else -20.0

    def score_continuation(self, context: str, ending: str) -> float:
        """Score how natural an ending is given context.
        Uses last 2 words of context as trigram seed."""
        ctx_words = re.findall(r'\b[a-z]+\b', context.lower())
        end_words = re.findall(r'\b[a-z]+\b', ending.lower())

        if not end_words:
            return -20.0

        # Seed from context tail
        if len(ctx_words) >= 2:
            seed = ctx_words[-2:]
        elif len(ctx_words) == 1:
            seed = ['<s>', ctx_words[0]]
        else:
            seed = ['<s>', '<s>']

        all_words = seed + end_words
        total = 0.0
        n = 0
        for i in range(2, len(all_words)):
            total += self.log_prob(all_words[i-2], all_words[i-1], all_words[i])
            n += 1

        return total / n if n > 0 else -20.0

    def perplexity(self, text: str) -> float:
        """Perplexity of text. Lower = more predictable."""
        avg_log_prob = self.score_text(text)
        return 2.0 ** (-avg_log_prob)

    def save(self, path: str):
        """Save model to pickle."""
        data = {
            'trigram_counts': dict(self.trigram_counts),
            'bigram_counts': dict(self.bigram_counts),
            'unigram_counts': dict(self.unigram_counts),
            'total_words': self.total_words,
            'vocab_size': self.vocab_size,
        }
        with open(path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'WordPPM':
        """Load model from pickle."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        model = cls()
        model.trigram_counts = defaultdict(int, data['trigram_counts'])
        model.bigram_counts = defaultdict(int, data['bigram_counts'])
        model.unigram_counts = defaultdict(int, data['unigram_counts'])
        model.total_words = data['total_words']
        model.vocab_size = data['vocab_size']
        return model
