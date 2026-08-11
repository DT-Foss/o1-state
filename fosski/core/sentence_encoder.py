"""
Positional Sentence Encoder — word order matters.
==================================================
Replaces naive np.mean(word_vecs) with position-aware encoding.

Three encoding modes:
1. Exponential decay: recent words weighted more (captures recency)
2. SIF (Smooth Inverse Frequency): rare words weighted more
3. Bigram-aware: concatenates consecutive word-pair diffs

All return a fixed-size vector suitable for cosine similarity.
"""

import re
import math
import numpy as np
from typing import Dict, Optional, Tuple


class SentenceEncoder:
    """Position-aware sentence encoder using GloVe vectors."""

    def __init__(self, word2idx: Dict[str, int], vecs: np.ndarray,
                 mode: str = 'sif', sif_a: float = 1e-3):
        """
        Args:
            word2idx: word → index mapping
            vecs: GloVe vectors (N, dim)
            mode: 'decay', 'sif', or 'bigram'
            sif_a: SIF smoothing parameter (lower = more weight on rare words)
        """
        self.word2idx = word2idx
        self.vecs = vecs
        self.dim = vecs.shape[1] if vecs is not None else 100
        self.mode = mode
        self.sif_a = sif_a

        # Precompute word frequencies for SIF weighting
        # Approximate: use position in GloVe file as frequency rank
        # GloVe is sorted by frequency, so index 0 = most frequent
        self._word_weight = {}
        if word2idx and mode == 'sif':
            n = len(word2idx)
            for word, idx in word2idx.items():
                # Zipf approximation: freq ~ 1/(rank+1)
                rank = idx + 1
                freq = 1.0 / rank
                # SIF weight: a / (a + freq)
                self._word_weight[word] = sif_a / (sif_a + freq)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to a fixed-size vector."""
        if self.word2idx is None:
            return np.zeros(self.dim)

        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        if not tokens:
            return np.zeros(self.dim)

        if self.mode == 'decay':
            return self._encode_decay(tokens)
        elif self.mode == 'sif':
            return self._encode_sif(tokens)
        elif self.mode == 'bigram':
            return self._encode_bigram(tokens)
        else:
            return self._encode_mean(tokens)

    def _encode_mean(self, tokens):
        """Baseline: simple average (bag-of-words)."""
        indices = [self.word2idx[w] for w in tokens if w in self.word2idx]
        if indices:
            return np.mean(self.vecs[indices], axis=0)
        return np.zeros(self.dim)

    def _encode_decay(self, tokens, decay: float = 0.85):
        """Exponential decay: last words weighted most.
        Good for continuation tasks — what comes last matters most."""
        indices = []
        weights = []
        n = len(tokens)
        for i, w in enumerate(tokens):
            if w in self.word2idx:
                indices.append(self.word2idx[w])
                # Weight increases exponentially toward end
                weights.append(decay ** (n - 1 - i))

        if not indices:
            return np.zeros(self.dim)

        weights = np.array(weights)
        weights /= weights.sum()
        return np.sum(self.vecs[indices] * weights[:, None], axis=0)

    def _encode_sif(self, tokens):
        """SIF (Smooth Inverse Frequency) weighted average.
        Rare words get higher weight — captures content over function words.
        From Arora et al. 2017 'A Simple but Tough-to-Beat Baseline'."""
        indices = []
        weights = []
        for w in tokens:
            if w in self.word2idx:
                indices.append(self.word2idx[w])
                weights.append(self._word_weight.get(w, self.sif_a))

        if not indices:
            return np.zeros(self.dim)

        weights = np.array(weights)
        weights /= weights.sum()
        return np.sum(self.vecs[indices] * weights[:, None], axis=0)

    def _encode_bigram(self, tokens):
        """Bigram-aware: average + difference between consecutive words.
        Captures local word order (not just bag-of-words)."""
        indices = [self.word2idx[w] for w in tokens if w in self.word2idx]
        if not indices:
            return np.zeros(self.dim)

        word_vecs = self.vecs[indices]
        mean_vec = np.mean(word_vecs, axis=0)

        if len(word_vecs) >= 2:
            # Average of consecutive differences captures word order
            diffs = word_vecs[1:] - word_vecs[:-1]
            diff_vec = np.mean(diffs, axis=0)
            # Concatenate and project back to dim via sum
            # (keeps same dimensionality for drop-in replacement)
            combined = mean_vec + 0.3 * diff_vec
            return combined

        return mean_vec

    def encode_continuation(self, context: str, ending: str) -> Tuple[np.ndarray, np.ndarray]:
        """Encode context and ending separately for continuation scoring.
        Context uses full encoding; ending uses decay (recent = important)."""
        ctx_vec = self.encode(context)

        # For endings, use decay encoding — last words of ending matter most
        tokens = re.findall(r'\b[a-z]+\b', ending.lower())
        if not tokens:
            end_vec = np.zeros(self.dim)
        else:
            end_vec = self._encode_decay(tokens, decay=0.9)

        return ctx_vec, end_vec

    def similarity(self, text_a: str, text_b: str) -> float:
        """Cosine similarity between two encoded texts."""
        va = self.encode(text_a)
        vb = self.encode(text_b)
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(np.dot(va, vb) / (na * nb))

    def coherence(self, text: str) -> float:
        """Semantic coherence: average pairwise similarity of content words.
        Higher = more semantically tight/focused text."""
        tokens = re.findall(r'\b[a-z]+\b', text.lower())
        indices = [self.word2idx[w] for w in tokens if w in self.word2idx]
        if len(indices) < 2:
            return 0.0

        word_vecs = self.vecs[indices]
        # Normalize
        norms = np.linalg.norm(word_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normed = word_vecs / norms

        # Average pairwise cosine
        sim_matrix = normed @ normed.T
        n = len(indices)
        # Exclude diagonal
        total = (sim_matrix.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
        return float(total)
