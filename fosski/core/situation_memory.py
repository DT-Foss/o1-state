"""
Situation Memory — Anti-Hallucination + Generalization (D1)
============================================================
Modern Hopfield storing (query_embedding, answer, quality_score) triplets.

Key insight from doom/foss-v2/modules/situation_memory.py:
- Attractor distance = "have I seen this before?"
- High distance = novel query → say "I don't know" instead of hallucinating
- Low distance = similar query seen → reuse cached answer with confidence

Storage: circular buffer (max_size), EMA update for quality scores.
Retrieval: softmax attention over stored query embeddings.

This is the ANTI-HALLUCINATION layer. Without it, the pipeline
confidently returns garbage for novel queries. With it, novel queries
get low confidence and "I don't know" as answer.
"""

import numpy as np
from typing import Optional, Tuple, List, Dict


class SituationMemory:
    """Circular buffer of (query_emb, answer, quality) triplets with Hopfield retrieval."""

    def __init__(self, dim: int = 512, max_size: int = 500, beta: float = 6.0,
                 novelty_threshold: float = 0.3):
        """
        Args:
            dim: embedding dimension
            max_size: maximum stored situations
            beta: softmax temperature for retrieval
            novelty_threshold: below this similarity = novel query
        """
        self.dim = dim
        self.max_size = max_size
        self.beta = beta
        self.novelty_threshold = novelty_threshold

        self._keys = np.zeros((max_size, dim), dtype=np.float32)
        self._answers: List[Optional[str]] = [None] * max_size
        self._qualities = np.zeros(max_size, dtype=np.float32)
        self._count = 0
        self._ptr = 0  # Circular buffer write pointer
        self._key_norms: Optional[np.ndarray] = None

    def store(self, query_emb: np.ndarray, answer: str, quality: float):
        """Store a new situation.

        Args:
            query_emb: embedding of the query (512d)
            answer: the answer that was given
            quality: quality score (e.g., consensus confidence, CASI score)
        """
        # Check for duplicates: if very similar query exists, update via EMA
        if self._count > 0:
            idx, sim = self._nearest(query_emb)
            if sim > 0.9:
                # EMA update: blend old quality with new
                self._qualities[idx] = 0.7 * self._qualities[idx] + 0.3 * quality
                self._answers[idx] = answer  # Update to latest answer
                return

        # Store at current pointer
        norm = np.linalg.norm(query_emb)
        if norm > 1e-8:
            self._keys[self._ptr] = query_emb / norm
        else:
            self._keys[self._ptr] = query_emb

        self._answers[self._ptr] = answer
        self._qualities[self._ptr] = quality
        self._ptr = (self._ptr + 1) % self.max_size
        self._count = min(self._count + 1, self.max_size)
        self._key_norms = None  # Invalidate cache

    def retrieve(self, query_emb: np.ndarray, top_k: int = 3
                 ) -> Tuple[Optional[str], float, bool]:
        """Retrieve most similar situation.

        Returns:
            (answer, confidence, is_novel)
            is_novel=True means query is far from all stored situations
        """
        if self._count == 0:
            return None, 0.0, True

        idx, sim = self._nearest(query_emb)

        if sim < self.novelty_threshold:
            return None, 0.0, True  # Novel query — don't hallucinate

        answer = self._answers[idx]
        quality = float(self._qualities[idx])
        confidence = sim * quality  # Blend similarity with stored quality

        return answer, confidence, False

    def _nearest(self, query_emb: np.ndarray) -> Tuple[int, float]:
        """Find nearest stored situation by cosine similarity."""
        norm = np.linalg.norm(query_emb)
        if norm < 1e-8:
            return 0, 0.0

        q = query_emb / norm
        active = self._keys[:self._count]
        sims = active @ q
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def is_novel(self, query_emb: np.ndarray) -> Tuple[bool, float]:
        """Check if a query is novel (far from all stored situations).

        Returns: (is_novel, max_similarity)
        """
        if self._count == 0:
            return True, 0.0

        _, sim = self._nearest(query_emb)
        return sim < self.novelty_threshold, sim

    @property
    def size(self) -> int:
        return self._count

    def stats(self) -> Dict[str, float]:
        if self._count == 0:
            return {'size': 0, 'avg_quality': 0.0}
        return {
            'size': self._count,
            'avg_quality': float(np.mean(self._qualities[:self._count])),
            'min_quality': float(np.min(self._qualities[:self._count])),
        }
