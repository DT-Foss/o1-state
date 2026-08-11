"""
Random Indexing — Kanerva (2009) Sparse Distributed Representations
====================================================================
Domain-adaptive word embeddings computed in O(n) from any text corpus.
No external data files. No gradients. Builds in seconds.

Algorithm:
  1. Assign each word a random sparse vector (RI vector):
     d-dimensional, with k nonzero entries (+1 or -1).
  2. For each word occurrence, add the RI vectors of context words
     (within window) to that word's context vector.
  3. The context vector IS the embedding.

Properties:
  - Approximates SVD/LSA (Kanerva 2009, Sahlgren 2005)
  - O(n) construction (one pass through corpus)
  - O(k) per update (k << d, sparse operations)
  - Domain-adaptive: train on WikiHow → WikiHow-specific embeddings
  - No external files needed

Parameters:
  - d: embedding dimension (1000-2000 is typical)
  - k: nonzero entries per RI vector (6-10)
  - window: context window size (2-5 words)
"""

import os
import re
import numpy as np
from collections import defaultdict
from typing import Dict, Optional


class RandomIndexing:
    """
    Random Indexing word embeddings.

    Train on any text corpus to get domain-specific embeddings.
    """

    def __init__(self, dim: int = 1000, k: int = 8, window: int = 3,
                 min_count: int = 5, seed: int = 42):
        """
        Args:
            dim: embedding dimension
            k: nonzero entries per random index vector (k/2 positive, k/2 negative)
            window: context window (words on each side)
            min_count: minimum word frequency to include
            seed: random seed for reproducibility
        """
        self.dim = dim
        self.k = k
        self.window = window
        self.min_count = min_count
        self._rng = np.random.RandomState(seed)

        self._ri_vectors: Dict[str, np.ndarray] = {}  # word → random index vector
        self._context_vectors: Dict[str, np.ndarray] = {}  # word → accumulated context
        self._word_counts: Dict[str, int] = defaultdict(int)
        self._vocab: set = set()
        self._trained = False

    def _get_ri_vector(self, word: str) -> np.ndarray:
        """Get or create the random index vector for a word."""
        if word not in self._ri_vectors:
            vec = np.zeros(self.dim)
            # Deterministic: seed from word hash
            h = hash(word) & 0x7FFFFFFF
            rng = np.random.RandomState(h)
            indices = rng.choice(self.dim, self.k, replace=False)
            signs = rng.choice([-1.0, 1.0], self.k)
            vec[indices] = signs
            self._ri_vectors[word] = vec
        return self._ri_vectors[word]

    def _tokenize(self, text: str):
        """Extract content words."""
        return [w for w in re.findall(r'\b[a-z]+\b', text.lower())
                if len(w) >= 2]

    def train(self, text: str):
        """
        Train on a text corpus. Can be called multiple times
        to train on additional text.

        Args:
            text: raw text (will be lowercased and tokenized)
        """
        words = self._tokenize(text)

        # Count word frequencies
        for w in words:
            self._word_counts[w] += 1

        # Single pass: accumulate context vectors
        for i, word in enumerate(words):
            if word not in self._context_vectors:
                self._context_vectors[word] = np.zeros(self.dim)

            # Add RI vectors of context words
            start = max(0, i - self.window)
            end = min(len(words), i + self.window + 1)
            for j in range(start, end):
                if j != i:
                    ctx_word = words[j]
                    self._context_vectors[word] += self._get_ri_vector(ctx_word)

        self._trained = True

    def train_from_file(self, path: str, max_bytes: int = 0):
        """Train from a text file, optionally limiting size."""
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            if max_bytes > 0:
                text = f.read(max_bytes)
            else:
                text = f.read()
        self.train(text)

    def build_vocab(self):
        """Finalize vocabulary (apply min_count filter)."""
        self._vocab = {w for w, c in self._word_counts.items()
                       if c >= self.min_count}

    @property
    def vocab_size(self):
        if not self._vocab:
            self.build_vocab()
        return len(self._vocab)

    def get_vector(self, word: str) -> Optional[np.ndarray]:
        """Get the embedding vector for a word."""
        if not self._vocab:
            self.build_vocab()
        word = word.lower()
        if word not in self._vocab or word not in self._context_vectors:
            return None
        vec = self._context_vectors[word]
        norm = np.linalg.norm(vec)
        if norm > 0:
            return vec / norm
        return None

    def get_vectors_batch(self, words):
        """Get vectors for multiple words. Returns (word_list, vector_matrix)."""
        result_words = []
        result_vecs = []
        for w in words:
            v = self.get_vector(w)
            if v is not None:
                result_words.append(w)
                result_vecs.append(v)
        if result_vecs:
            return result_words, np.array(result_vecs)
        return [], np.zeros((0, self.dim))

    def similarity(self, w1: str, w2: str) -> float:
        """Cosine similarity between two words."""
        v1 = self.get_vector(w1)
        v2 = self.get_vector(w2)
        if v1 is None or v2 is None:
            return 0.0
        return float(np.dot(v1, v2))

    def nearest_neighbors(self, word: str, top_k: int = 10):
        """Find nearest neighbors for a word."""
        vec = self.get_vector(word)
        if vec is None:
            return []

        if not self._vocab:
            self.build_vocab()

        results = []
        for w in self._vocab:
            if w == word:
                continue
            v = self.get_vector(w)
            if v is not None:
                sim = float(np.dot(vec, v))
                results.append((w, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def sent_vec(self, text: str) -> np.ndarray:
        """Compute sentence vector as mean of word vectors."""
        words = self._tokenize(text)
        vecs = []
        for w in words:
            v = self.get_vector(w)
            if v is not None:
                vecs.append(v)
        if vecs:
            mean = np.mean(vecs, axis=0)
            norm = np.linalg.norm(mean)
            return mean / norm if norm > 0 else mean
        return np.zeros(self.dim)

    def cosine_sent(self, text1: str, text2: str) -> float:
        """Cosine similarity between two texts."""
        v1 = self.sent_vec(text1)
        v2 = self.sent_vec(text2)
        return float(np.dot(v1, v2))

    def save(self, path: str):
        """Save trained model."""
        import pickle
        data = {
            'dim': self.dim,
            'k': self.k,
            'window': self.window,
            'min_count': self.min_count,
            'word_counts': dict(self._word_counts),
            'context_vectors': {w: v for w, v in self._context_vectors.items()
                                if w in self._vocab},
            'vocab': self._vocab,
        }
        tmp = path + '.tmp'
        with open(tmp, 'wb') as f:
            pickle.dump(data, f)
        os.rename(tmp, path)  # atomic

    @classmethod
    def load(cls, path: str) -> 'RandomIndexing':
        """Load trained model."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
        ri = cls(dim=data['dim'], k=data['k'], window=data['window'],
                 min_count=data['min_count'])
        ri._word_counts = defaultdict(int, data['word_counts'])
        ri._context_vectors = data['context_vectors']
        ri._vocab = data['vocab']
        ri._trained = True
        return ri
