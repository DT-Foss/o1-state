"""
Hopfield Template Bank — Associative Memory with Qwen3 Embeddings
==================================================================
Modern Hopfield Network (exponential capacity) storing KB + ConceptNet
patterns for fast content-addressable retrieval.

Unlike the GloVe-based HopfieldSequenceMemory, this uses Qwen3 512d
projected embeddings for richer semantic representation.

Storage capacity: O(d^(d-1)) for d=512 → effectively unlimited for
our 500-2000 pattern range.

Retrieval: softmax attention over stored patterns (Modern Hopfield).
Temperature β controls sharpness (high β → exact match, low β → fuzzy).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional


class HopfieldTemplateBank:
    """Modern Hopfield Network with Qwen3 embeddings as patterns.

    Patterns are stored as (key_vec, value_info) pairs:
    - key_vec: 512d Qwen3 embedding (what to match)
    - value_info: dict with answer, relation, metadata (what to return)

    Retrieval uses softmax attention: exp(β * K^T q) / Z
    """

    def __init__(self, emb_store, beta: float = 8.0):
        self.emb_store = emb_store
        self.beta = beta
        self._keys = []       # list of 512d vectors
        self._values = []     # list of dicts: {answer, relation, subject, ...}
        self._key_matrix = None
        self._key_normed = None
        self._pattern_bias = None
        self._svd_V = None
        self._svd_mean = None

    def store(self, key_words: List[str], value: dict):
        """Store a pattern: key_words → value.

        Args:
            key_words: words that form the "query" pattern
            value: dict with at least 'answer' key
        """
        # Average embeddings of key words
        vecs = []
        for w in key_words:
            v = self.emb_store.encode(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return

        key_vec = np.mean(vecs, axis=0).astype(np.float32)
        self._keys.append(key_vec)
        self._values.append(value)
        self._key_matrix = None  # Invalidate cache

    def store_fact(self, subject: str, relation: str, obj: str):
        """Store a KB fact as a pattern.

        Key = average(subject_words + relation_words)
        Value = {answer: obj, subject, relation}
        """
        key_words = subject.lower().split() + relation.lower().replace('_', ' ').split()
        key_words = [w for w in key_words if len(w) > 1]
        if not key_words:
            return
        self.store(key_words, {
            'answer': obj,
            'subject': subject,
            'relation': relation,
        })

    def store_facts(self, facts: List[Tuple[str, str, str]]):
        """Bulk store KB facts."""
        for s, r, o in facts:
            self.store_fact(s, r, str(o))
        self._build_matrix()

    def _build_matrix(self):
        """Build key matrix for fast retrieval.

        Two optimizations:
        1. SVD compression 512d → 32d (F2: ID_needed = 2×log₂(K)+1.2 ≈ 23d
           for K=2000 patterns. 32d = safe margin. 16× noise reduction.)
        2. Sinkhorn bias removal (FB2/F7): subtract per-pattern average
           similarity to prevent attention sinks. 1 iteration = 86% effect.
        """
        if not self._keys:
            return
        self._key_matrix = np.array(self._keys, dtype=np.float32)

        # SVD compression: 512d → 32d (F2 intrinsic dimension law)
        if self._key_matrix.shape[0] >= 10:
            mean_key = self._key_matrix.mean(axis=0, keepdims=True)
            centered = self._key_matrix - mean_key
            U, S, Vt = np.linalg.svd(centered, full_matrices=False)
            k = min(32, len(S))  # 32d = safe above ID_needed ≈ 23
            self._svd_V = Vt[:k].T.astype(np.float32)  # (512, 32) projection
            self._svd_mean = mean_key[0].astype(np.float32)
            # Project keys to compressed space
            compressed = (self._key_matrix - self._svd_mean) @ self._svd_V  # (n, 32)
            norms = np.linalg.norm(compressed, axis=1, keepdims=True)
            self._key_normed = (compressed / np.clip(norms, 1e-8, None)).astype(np.float32)
        else:
            self._svd_V = None
            self._svd_mean = None
            norms = np.linalg.norm(self._key_matrix, axis=1, keepdims=True)
            self._key_normed = self._key_matrix / np.clip(norms, 1e-8, None)

        # Sinkhorn bias: mean cosine similarity per pattern to all others
        # High bias = pattern is "popular" = attention sink
        sim_matrix = self._key_normed @ self._key_normed.T
        self._pattern_bias = sim_matrix.mean(axis=1)  # (n_patterns,)

        # Auto-calibrate beta after SVD compression
        # In lower dimensions, cosine similarities are different.
        # Set beta so that a "good match" (top 5% of similarity scores)
        # gets softmax weight > 0.5
        off_diag = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
        if len(off_diag) > 10:
            p95 = float(np.percentile(off_diag, 95))
            p50 = float(np.percentile(off_diag, 50))
            gap = max(p95 - p50, 0.01)
            # beta such that exp(beta * gap) >> 1 → beta ≈ 3/gap
            self.beta = max(4.0, min(30.0, 3.0 / gap))

    def retrieve(self, query_words: List[str], top_k: int = 5,
                 beta: float = None) -> List[Tuple[dict, float]]:
        """Retrieve patterns matching query words.

        Uses Modern Hopfield dynamics: softmax(β * K^T q)

        Args:
            query_words: content words from the question
            top_k: number of results
            beta: inverse temperature (default: self.beta)

        Returns:
            [(value_dict, score), ...] sorted by score
        """
        if self._key_normed is None:
            self._build_matrix()
        if self._key_normed is None or len(self._keys) == 0:
            return []

        if beta is None:
            beta = self.beta

        # Build query vector
        vecs = []
        for w in query_words:
            v = self.emb_store.encode(w)
            if v is not None:
                vecs.append(v)
        if not vecs:
            return []

        query = np.mean(vecs, axis=0).astype(np.float32)

        # Project query through same SVD as keys
        if self._svd_V is not None:
            query = (query - self._svd_mean) @ self._svd_V  # 512d → 32d

        query_norm = np.linalg.norm(query)
        if query_norm < 1e-8:
            return []
        query /= query_norm

        # Softmax attention with Sinkhorn bias removal (FB2/F7)
        scores = beta * (self._key_normed @ query)
        if self._pattern_bias is not None:
            scores -= beta * self._pattern_bias  # Remove attention sink bias
        scores -= scores.max()  # Numerical stability
        weights = np.exp(scores)
        weights /= weights.sum()

        # Top-k
        top_idx = np.argpartition(weights, -min(top_k, len(weights)))[-top_k:]
        top_idx = top_idx[np.argsort(weights[top_idx])[::-1]]

        return [(self._values[i], float(weights[i])) for i in top_idx
                if weights[i] > 1e-6]

    def query_answer(self, question: str, top_k: int = 3) -> Optional[Tuple[str, float]]:
        """Convenience: extract content words from question, retrieve best answer.

        Returns (answer_string, confidence) or None.
        """
        STOP = frozenset('what who where when how which why is are was were the a an '
                         'of in on at to for with by from as does do did has have had'.split())

        q = question.lower()
        for ch in '?.,!;:"\'-()[]{}':
            q = q.replace(ch, ' ')
        words = [w for w in q.split() if w not in STOP and len(w) > 1]

        if not words:
            return None

        results = self.retrieve(words, top_k=top_k)
        if not results:
            return None

        best_val, best_score = results[0]
        return best_val.get('answer'), best_score

    @property
    def n_patterns(self):
        return len(self._keys)

    def stats(self) -> Dict[str, int]:
        return {
            'patterns': self.n_patterns,
            'dim': self.emb_store.dim if self.emb_store else 0,
        }


def build_hopfield_bank(emb_store, knowledge_store=None,
                        commonsense_engine=None,
                        max_patterns: int = 2000) -> HopfieldTemplateBank:
    """Build a Hopfield Template Bank from KB + ConceptNet.

    Args:
        emb_store: EmbeddingStore (Qwen3 512d)
        knowledge_store: KnowledgeStore with facts
        commonsense_engine: CommonSenseEngine with cn_lookup
        max_patterns: cap on number of patterns

    Returns:
        HopfieldTemplateBank with 500-2000 patterns
    """
    bank = HopfieldTemplateBank(emb_store, beta=8.0)
    n = 0

    # KB facts
    if knowledge_store:
        for s, r, o in knowledge_store.facts:
            if n >= max_patterns:
                break
            s_str = str(s).lower().strip()
            o_str = str(o).lower().strip()
            if not s_str or not o_str or s_str.startswith('/m/'):
                continue
            bank.store_fact(s_str, r, o_str)
            n += 1

    # ConceptNet top concepts
    if commonsense_engine and n < max_patterns:
        seen = set()
        for concept in list(commonsense_engine._by_subject.keys())[:500]:
            if n >= max_patterns:
                break
            facts = commonsense_engine.cn_lookup(concept, max_results=5)
            for rel, obj, weight in facts:
                if weight < 1.5:
                    continue
                key = (concept, rel, obj)
                if key in seen:
                    continue
                seen.add(key)
                bank.store_fact(concept, rel, obj)
                n += 1

    bank._build_matrix()
    return bank
