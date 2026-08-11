"""
Experience Pool — Online Learning ohne Gradients (D3)
======================================================
Source: doom/foss-v2/modules/experience_pool.py

Circular buffer storing (query_emb, answer, quality) from successful queries.
Before next inference: retrieve top-3 relevant experiences as priming.

This gives the pipeline a form of "memory" — it learns from its own
successful queries without any gradient computation.

Key properties:
- Max 500 entries, oldest pruned
- Deduplication via cosine similarity > 0.9
- Quality threshold: only store high-confidence results
- Retrieval: top-3 most similar past experiences
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


class ExperiencePool:
    """Circular buffer of successful query experiences for priming."""

    def __init__(self, dim: int = 512, max_size: int = 500,
                 quality_threshold: float = 0.5):
        """
        Args:
            dim: embedding dimension
            max_size: maximum stored experiences
            quality_threshold: minimum confidence to store
        """
        self.dim = dim
        self.max_size = max_size
        self.quality_threshold = quality_threshold

        self._keys = np.zeros((max_size, dim), dtype=np.float32)
        self._answers: List[Optional[str]] = [None] * max_size
        self._qualities = np.zeros(max_size, dtype=np.float32)
        self._count = 0
        self._ptr = 0

    def store(self, query_emb: np.ndarray, answer: str, quality: float):
        """Store a successful experience."""
        if quality < self.quality_threshold:
            return

        norm = np.linalg.norm(query_emb)
        if norm < 1e-8:
            return
        q_normed = query_emb / norm

        # Dedup: skip if very similar experience exists
        if self._count > 0:
            sims = self._keys[:self._count] @ q_normed
            if float(np.max(sims)) > 0.9:
                # Update quality via EMA
                idx = int(np.argmax(sims))
                self._qualities[idx] = 0.7 * self._qualities[idx] + 0.3 * quality
                return

        self._keys[self._ptr] = q_normed
        self._answers[self._ptr] = answer
        self._qualities[self._ptr] = quality
        self._ptr = (self._ptr + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)

    def retrieve(self, query_emb: np.ndarray, top_k: int = 3
                 ) -> List[Tuple[str, float]]:
        """Retrieve most relevant past experiences for priming.

        Returns: [(answer, similarity), ...] sorted by relevance
        """
        if self._count == 0:
            return []

        norm = np.linalg.norm(query_emb)
        if norm < 1e-8:
            return []

        q = query_emb / norm
        sims = self._keys[:self._count] @ q

        k = min(top_k, self._count)
        top_idx = np.argpartition(sims, -k)[-k:]
        top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

        results = []
        for i in top_idx:
            if sims[i] > 0.2 and self._answers[i] is not None:
                results.append((self._answers[i], float(sims[i])))
        return results

    @property
    def size(self) -> int:
        return self._count

    def stats(self) -> Dict[str, float]:
        if self._count == 0:
            return {'size': 0, 'avg_quality': 0.0}
        return {
            'size': self._count,
            'avg_quality': float(np.mean(self._qualities[:self._count])),
        }
