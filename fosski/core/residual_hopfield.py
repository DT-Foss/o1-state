"""
Residual Stream Hopfield — Pre-Contextualized Embeddings
=========================================================
Stores Qwen3 Layer-10 and Layer-18 hidden states as Hopfield patterns.
These are "stolen" contextualized representations — the word "Paris"
after 10 transformer layers already encodes "capital of France".

Lookup: given a query (512d projected), project to 2048d, find
nearest residual, return the pre-contextualized meaning.

This gives us transformer-depth semantic representation WITHOUT
running the transformer at inference time.
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional


class ResidualHopfield:
    """Hopfield memory over pre-extracted transformer residual states.

    Patterns: 2048d hidden states from Qwen3 layers 10 and 18.
    Query: project 512d embedding to 2048d, softmax attention.
    """

    def __init__(self, data_dir=None):
        if data_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            data_dir = os.path.join(base, 'data')

        self._data_dir = data_dir
        self._loaded = False
        self._entries = {}   # word → {'layer_10': vec, 'layer_18': vec}
        self._keys_10 = None  # (N, 2048) matrix for layer 10
        self._keys_18 = None  # (N, 2048) matrix for layer 18
        self._words = []     # word list aligned with matrices
        self._proj = None    # SVD projection matrix (2048, 512) for query projection

    def _load(self):
        if self._loaded:
            return
        self._loaded = True

        res_path = os.path.join(self._data_dir, 'qwen3_residuals.npz')
        if not os.path.exists(res_path):
            return

        data = np.load(res_path)

        # Parse: "word__layer_10" → word, layer_10
        entries = {}
        for key in data.files:
            parts = key.rsplit('__', 1)
            if len(parts) != 2:
                continue
            word, layer = parts
            if word not in entries:
                entries[word] = {}
            entries[word][layer] = data[key].astype(np.float32)

        self._entries = entries
        self._words = sorted(entries.keys())

        # Build matrices
        vecs_10, vecs_18 = [], []
        for w in self._words:
            vecs_10.append(entries[w].get('layer_10', np.zeros(2048, dtype=np.float32)))
            vecs_18.append(entries[w].get('layer_18', np.zeros(2048, dtype=np.float32)))

        if vecs_10:
            self._keys_10 = np.array(vecs_10, dtype=np.float32)
            norms = np.linalg.norm(self._keys_10, axis=1, keepdims=True)
            self._keys_10_normed = self._keys_10 / np.clip(norms, 1e-8, None)

        if vecs_18:
            self._keys_18 = np.array(vecs_18, dtype=np.float32)
            norms = np.linalg.norm(self._keys_18, axis=1, keepdims=True)
            self._keys_18_normed = self._keys_18 / np.clip(norms, 1e-8, None)

        # Load SVD projection for 512d → 2048d inverse mapping
        svd_path = os.path.join(self._data_dir, 'qwen3_svd_V512.npy')
        mean_path = os.path.join(self._data_dir, 'qwen3_svd_mean.npy')
        if os.path.exists(svd_path) and os.path.exists(mean_path):
            V = np.load(svd_path).astype(np.float32)  # (512, 2048)
            self._mean = np.load(mean_path).astype(np.float32)
            self._proj = V  # (512, 2048) — multiply 512d vec by this to get 2048d
        else:
            self._proj = None
            self._mean = None

    @property
    def available(self):
        self._load()
        return len(self._entries) > 0

    @property
    def n_patterns(self):
        self._load()
        return len(self._words)

    def project_to_2048(self, vec_512d: np.ndarray) -> np.ndarray:
        """Project 512d embedding back to approximate 2048d space.

        Uses pseudo-inverse of SVD projection: x_2048 ≈ V^T @ x_512 + mean
        """
        if self._proj is None:
            self._load()
        if self._proj is None:
            # Can't project, pad with zeros
            return np.concatenate([vec_512d, np.zeros(2048 - len(vec_512d))])

        # V is (512, 2048), so V^T is (2048, 512)
        x_2048 = (self._proj.T @ vec_512d).astype(np.float32)
        if self._mean is not None:
            x_2048 += self._mean
        return x_2048

    def retrieve(self, query_2048: np.ndarray, layer: str = 'layer_18',
                 top_k: int = 5, beta: float = 8.0) -> List[Tuple[str, float]]:
        """Retrieve nearest residual patterns.

        Args:
            query_2048: 2048d query vector
            layer: 'layer_10' or 'layer_18'
            top_k: number of results
            beta: inverse temperature

        Returns: [(word, score), ...]
        """
        self._load()
        if layer == 'layer_10':
            keys = self._keys_10_normed
        else:
            keys = self._keys_18_normed

        if keys is None or len(self._words) == 0:
            return []

        # Cosine similarity
        q_norm = query_2048 / (np.linalg.norm(query_2048) + 1e-8)
        scores = beta * (keys @ q_norm)
        scores -= scores.max()
        weights = np.exp(scores)
        weights /= weights.sum()

        top_idx = np.argpartition(weights, -min(top_k, len(weights)))[-top_k:]
        top_idx = top_idx[np.argsort(weights[top_idx])[::-1]]

        return [(self._words[i], float(weights[i])) for i in top_idx]

    def retrieve_from_512d(self, query_512d: np.ndarray, layer: str = 'layer_18',
                           top_k: int = 5) -> List[Tuple[str, float]]:
        """Convenience: project 512d query and retrieve."""
        q_2048 = self.project_to_2048(query_512d)
        return self.retrieve(q_2048, layer=layer, top_k=top_k)

    def get_contextualized(self, word: str, layer: str = 'layer_18') -> Optional[np.ndarray]:
        """Get the pre-contextualized 2048d representation of a word.

        This is the transformer's internal representation AFTER processing
        through 10 or 18 layers. Much richer than the raw embedding.
        """
        self._load()
        entry = self._entries.get(word.lower())
        if entry is None:
            return None
        return entry.get(layer)

    def enrich_prediction(self, pred_2048: np.ndarray,
                          alpha: float = 0.3) -> np.ndarray:
        """Enrich a reservoir prediction with nearest residual pattern.

        Blends the raw prediction with the nearest contextualized
        representation, giving it transformer-like depth.

        Args:
            pred_2048: reservoir output (2048d)
            alpha: blend weight (0 = pure reservoir, 1 = pure residual)

        Returns: enriched 2048d vector
        """
        self._load()
        if self._keys_18_normed is None:
            return pred_2048

        # Find nearest residual
        q_norm = pred_2048 / (np.linalg.norm(pred_2048) + 1e-8)
        sims = self._keys_18_normed @ q_norm
        best_idx = np.argmax(sims)
        best_residual = self._keys_18[best_idx]

        # Blend
        return (1 - alpha) * pred_2048 + alpha * best_residual
