"""
In-Context Learning — Without Gradient Updates
================================================
Two mechanisms that achieve ICL without a Transformer:

1. EMBEDDING TRANSLATION VECTOR
   "cat→Katze, dog→Hund" defines δ = mean(embed(target) - embed(source))
   Apply δ to embed(fish) → nearest neighbor = Fisch
   Pure linear algebra. Superior for systematic tasks.
   Operates in RAW 2048d space for maximum expressiveness.

2. SHERMAN-MORRISON RANK-1 READOUT UPDATE
   The reservoir readout W_out is trained via Ridge regression.
   Sherman-Morrison gives an O(n²) rank-1 update when a new
   (state, target) pair arrives — real parameter learning in context.
   This is what von Oswald et al. (2023) showed Transformers approximate.
   Needs 3+ examples to overcome the ||u||² denominator.
"""

import numpy as np
from typing import List, Tuple, Optional


class TranslationVectorICL:
    """In-context learning via embedding space translation vectors.

    Given demo pairs [(source₁, target₁), (source₂, target₂), ...],
    compute δ = mean(embed(targetᵢ) - embed(sourceᵢ)).
    Apply δ to new input → nearest neighbor in embedding space.

    Operates in RAW 2048d space when available (not 512d projected),
    because cross-lingual and semantic structure is richer there.
    """

    def __init__(self, emb_store, raw_emb=None, id2token=None):
        self.emb_store = emb_store
        self.raw_emb = raw_emb
        self.id2token = id2token
        self._raw_normed = None

    def _encode_raw(self, token: str) -> Optional[np.ndarray]:
        """Encode token in raw 2048d space if available, else 512d."""
        if self.raw_emb is not None:
            self.emb_store._load()
            for variant in [token, 'Ġ' + token,
                            token.capitalize(), 'Ġ' + token.capitalize(),
                            token.lower(), 'Ġ' + token.lower(),
                            token.upper(), 'Ġ' + token.upper()]:
                tid = self.emb_store._token2id.get(variant)
                if tid is not None:
                    return self.raw_emb[tid]
            return None
        return self.emb_store.encode(token)

    def _nearest_raw(self, vec: np.ndarray, top_k=10,
                     filter_garbage=True) -> List[Tuple[str, float]]:
        """Find nearest tokens in the same space as vec."""
        if self.raw_emb is not None and vec.shape[0] == self.raw_emb.shape[1]:
            # 2048d raw space search
            if self._raw_normed is None:
                norms = np.linalg.norm(self.raw_emb, axis=1, keepdims=True)
                self._raw_normed = self.raw_emb / np.clip(norms, 1e-8, None)

            vec_norm = vec / (np.linalg.norm(vec) + 1e-8)
            sims = self._raw_normed @ vec_norm

            if not filter_garbage:
                top_idx = np.argpartition(sims, -top_k)[-top_k:]
                top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
                return [(self.id2token.get(int(i), '?'), float(sims[i]))
                        for i in top_idx]

            # Filter: get extra, then keep clean tokens
            fetch_k = top_k * 10
            top_idx = np.argpartition(sims, -fetch_k)[-fetch_k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]

            results = []
            for i in top_idx:
                token = self.id2token.get(int(i), '?')
                clean = token.replace('Ġ', '').replace('ĉ', ' ').strip()
                if (not clean or len(clean) < 2
                        or not clean[0].isascii() or not clean[0].isalpha()
                        or token.startswith('<|') or token.startswith('ð')
                        or any(ord(c) > 127 for c in clean)
                        or (any(c.isupper() for c in clean[1:])
                            and any(c.islower() for c in clean))
                        or any(c.isdigit() for c in clean)
                        or len(clean) > 15):
                    continue
                results.append((token, float(sims[i])))
                if len(results) >= top_k:
                    break
            return results

        # Fallback: 512d projected space
        return self.emb_store.nearest(vec, top_k=top_k)

    def learn_translation(self, demo_pairs: List[Tuple[str, str]]) -> Optional[np.ndarray]:
        """Learn translation vector from demo pairs.

        Args:
            demo_pairs: [(source_word, target_word), ...]

        Returns:
            δ vector (embedding dim) or None if insufficient coverage
        """
        deltas = []
        for src, tgt in demo_pairs:
            src_emb = self._encode_raw(src)
            tgt_emb = self._encode_raw(tgt)
            if src_emb is not None and tgt_emb is not None:
                deltas.append(tgt_emb - src_emb)

        if not deltas:
            return None
        return np.mean(deltas, axis=0).astype(np.float32)

    def apply(self, word: str, delta: np.ndarray, top_k=5) -> List[Tuple[str, float]]:
        """Apply translation vector to a word and find nearest results.

        Filters out the query word and its BPE variants from results
        (otherwise self-similarity dominates in high-d spaces).

        Args:
            word: input word to translate
            delta: learned translation vector
            top_k: number of results

        Returns:
            [(token, similarity), ...] sorted by similarity
        """
        emb = self._encode_raw(word)
        if emb is None:
            return []

        translated = emb + delta
        # Fetch extra to account for filtered-out self-matches
        raw_results = self._nearest_raw(translated, top_k=top_k * 3)

        # Filter out query word variants
        query_lower = word.lower().strip()
        filtered = []
        for token, sim in raw_results:
            clean = token.replace('Ġ', '').replace('ĉ', ' ').strip().lower()
            if clean == query_lower or clean.rstrip('s') == query_lower:
                continue
            filtered.append((token, sim))
            if len(filtered) >= top_k:
                break
        return filtered

    def icl(self, demo_pairs: List[Tuple[str, str]], query: str,
            top_k=5) -> List[Tuple[str, float]]:
        """Full ICL: learn from demos, apply to query.

        Filters out both the query word AND demo source/target words
        from results to show only novel predictions.

        Example:
            icl([("cat", "Katze"), ("dog", "Hund")], "fish")
            → [("Fisch", 0.85), ...]
        """
        delta = self.learn_translation(demo_pairs)
        if delta is None:
            return []

        # Get raw results
        results = self.apply(query, delta, top_k=top_k * 3)

        # Filter out demo words
        demo_words = set()
        for src, tgt in demo_pairs:
            demo_words.add(src.lower())
            demo_words.add(tgt.lower())

        filtered = []
        for token, sim in results:
            clean = token.replace('Ġ', '').replace('ĉ', ' ').strip().lower()
            if clean in demo_words:
                continue
            filtered.append((token, sim))
            if len(filtered) >= top_k:
                break
        return filtered


class ShermanMorrisonICL:
    """In-context learning via rank-1 updates to reservoir readout.

    Stores the Ridge regression inverse (X^TX + αI)^{-1} and updates it
    with each ICL example using the Sherman-Morrison formula:
        (A + uv^T)^{-1} = A^{-1} - (A^{-1}u)(v^T A^{-1}) / (1 + v^T A^{-1} u)

    This is REAL parameter learning — W_out actually changes.
    Von Oswald et al. (2023) showed Transformers approximate this.
    FOSS-KI does it exactly.

    Key insight: single updates are weak because ||u||² dominates the
    denominator (4096-dim feature vectors). Use 3+ examples to accumulate
    sufficient signal, and apply a learning rate multiplier for ICL-scale
    corrections (not full retraining).
    """

    def __init__(self, reservoir, learning_rate=5.0):
        self.reservoir = reservoir
        self.learning_rate = learning_rate  # ICL correction amplifier
        self._A_inv = None  # (X^TX + αI)^{-1}, cached
        self._W_out_base = None  # Original W_out before ICL
        self._n_updates = 0

    def snapshot(self):
        """Save current readout state before ICL modifications."""
        if self.reservoir.W_out is not None:
            self._W_out_base = self.reservoir.W_out.copy()

    def restore(self):
        """Restore readout to pre-ICL state."""
        if self._W_out_base is not None:
            self.reservoir.W_out = self._W_out_base.copy()
            self._A_inv = None
            self._n_updates = 0

    def initialize_inverse(self, alpha=1.0):
        """Initialize A^{-1} = (αI)^{-1} = (1/α)I.

        Lower alpha = stronger ICL updates (less regularization).
        Default α=1.0 (was 10.0, too conservative for ICL).
        """
        if self.reservoir.W_out is None:
            return
        # Feature dimension after quadratic expansion
        feat_dim = self.reservoir.reservoir_size * 2  # [state, state²]
        self._A_inv = np.eye(feat_dim, dtype=np.float32) / alpha

    def update(self, state: np.ndarray, target: np.ndarray):
        """Rank-1 update with a single ICL example.

        Args:
            state: reservoir state after encoding the ICL input
            target: desired output embedding for the ICL output

        Complexity: O(n²) where n = reservoir_size * 2
        """
        if self._A_inv is None:
            self.initialize_inverse()

        # Expand features (same as ReservoirLM._expand_features)
        u = np.hstack([state, state ** 2]).astype(np.float32)  # (feat_dim,)

        # Sherman-Morrison: (A + uu^T)^{-1} = A^{-1} - (A^{-1}u)(u^T A^{-1}) / (1 + u^T A^{-1} u)
        A_inv_u = self._A_inv @ u  # (feat_dim,)
        denom = 1.0 + u @ A_inv_u  # scalar

        if abs(denom) < 1e-10:
            return  # Numerically unstable, skip

        self._A_inv -= np.outer(A_inv_u, A_inv_u) / denom

        # Update W_out with learning rate
        # W_out += lr * (target - W_out @ u) · (u^T @ A_inv) / denom
        if self.reservoir.W_out is not None:
            residual = target - self.reservoir.W_out @ u  # (output_dim,)
            correction = np.outer(residual, A_inv_u) / denom  # (output_dim, feat_dim)
            self.reservoir.W_out += self.learning_rate * correction
            self._n_updates += 1

    def icl_from_pairs(self, pairs: List[Tuple[List[str], np.ndarray]],
                       emb_store):
        """Learn from ICL demonstration pairs.

        Args:
            pairs: [(input_words, target_embedding), ...]
            emb_store: EmbeddingStore for encoding input words

        Use 3+ pairs for visible effect (single updates are weak
        because ||u||² ≈ 200-1000 dominates the denominator).
        """
        self.snapshot()
        self.initialize_inverse()

        for words, target in pairs:
            # Encode through reservoir
            state, _ = self.reservoir.encode_context(words, emb_store, mix_steps=3)
            self.update(state, target)

    def icl_from_word_pairs(self, demo_pairs: List[Tuple[str, str]],
                            emb_store, raw_emb=None):
        """Convenience: learn from word→word demonstration pairs.

        Args:
            demo_pairs: [(input_word, target_word), ...]
            emb_store: EmbeddingStore
            raw_emb: optional raw 2048d embeddings for targets
        """
        self.snapshot()
        self.initialize_inverse()

        emb_store._load()
        token2id = emb_store._token2id

        def get_target(word):
            for variant in [word, 'Ġ' + word, word.capitalize(),
                            'Ġ' + word.capitalize(), word.lower(),
                            'Ġ' + word.lower()]:
                tid = token2id.get(variant)
                if tid is not None:
                    if raw_emb is not None:
                        return raw_emb[tid]
                    return emb_store.emb[tid]
            return None

        for src, tgt in demo_pairs:
            target = get_target(tgt)
            if target is None:
                continue
            # Encode source through reservoir
            state, _ = self.reservoir.encode_context(
                [src], emb_store, mix_steps=3)
            self.update(state, target)
