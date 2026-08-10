"""
Fast Nearest Neighbor Index — Annoy + FAISS Wrapper
=====================================================
Replaces brute-force cosine similarity (O(V·d) = 48ms/query)
with approximate NN via Annoy (<5ms/query) or FAISS IVF (<1ms).

Annoy (Spotify): No OpenMP dependency, works cleanly with Anaconda/MLX.
FAISS: Faster but has OpenMP conflicts with Anaconda. Use after Anaconda removal.

Primary index for FOSS-KI: 151,936 × 2048d token embeddings.
"""

import os
import numpy as np
from typing import List, Tuple, Optional, Dict

# Try Annoy first (no OpenMP conflicts)
try:
    from annoy import AnnoyIndex
    HAS_ANNOY = True
except ImportError:
    HAS_ANNOY = False

# FAISS as fallback (may have OpenMP issues with Anaconda)
HAS_FAISS = False
try:
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import faiss
    HAS_FAISS = True
except (ImportError, OSError):
    pass


class NNIndex:
    """Fast nearest neighbor index over token embeddings.

    Uses Annoy (angular distance ≈ cosine) by default.
    Falls back to FAISS or brute-force numpy.
    """

    def __init__(self, embeddings: np.ndarray, id2token: Dict[int, str] = None,
                 mode: str = 'auto', n_trees: int = 50):
        """Build index.

        Args:
            embeddings: (V, d) float32 matrix
            id2token: optional token ID → string mapping
            mode: 'annoy', 'faiss', 'flat', or 'auto' (try annoy → faiss → flat)
            n_trees: number of trees for Annoy (more = more accurate, slower build)
        """
        self.embeddings = embeddings.astype(np.float32)
        self.id2token = id2token or {}
        self.n_vectors, self.dim = embeddings.shape
        self._backend = 'numpy'  # fallback

        # Normalize for cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.normed = (self.embeddings / np.clip(norms, 1e-8, None)).astype(np.float32)

        if mode == 'auto':
            if HAS_ANNOY:
                mode = 'annoy'
            elif HAS_FAISS:
                mode = 'faiss'
            else:
                mode = 'flat'

        if mode == 'annoy' and HAS_ANNOY:
            self._build_annoy(n_trees)
        elif mode == 'faiss' and HAS_FAISS:
            self._build_faiss()
        # else: numpy fallback

    def _build_annoy(self, n_trees: int):
        """Build or load Annoy index (angular distance ≈ cosine similarity).

        Saves to disk after first build for instant reloads.
        """
        # Check for pre-built index on disk
        cache_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                  'data', f'annoy_index_{self.n_vectors}_{self.dim}_{n_trees}.ann')
        self._annoy = AnnoyIndex(self.dim, 'angular')

        if os.path.exists(cache_path):
            self._annoy.load(cache_path)
            self._backend = 'annoy'
            return

        # Build from scratch (slow, ~60s for 151K × 2048d)
        for i in range(self.n_vectors):
            self._annoy.add_item(i, self.normed[i])
        self._annoy.build(n_trees)
        # Save for next time
        try:
            self._annoy.save(cache_path)
        except Exception:
            pass
        self._backend = 'annoy'

    def _build_faiss(self):
        """Build FAISS IVF index."""
        nlist = min(256, self.n_vectors // 100)
        quantizer = faiss.IndexFlatIP(self.dim)
        self._faiss = faiss.IndexIVFFlat(quantizer, self.dim, nlist,
                                         faiss.METRIC_INNER_PRODUCT)
        self._faiss.train(self.normed)
        self._faiss.add(self.normed)
        self._faiss.nprobe = min(16, nlist)
        self._backend = 'faiss'

    def search(self, query: np.ndarray, top_k: int = 10) -> List[Tuple[int, float]]:
        """Find top-k nearest neighbors.

        Returns: [(token_id, cosine_similarity), ...]
        """
        q = query.astype(np.float32)
        q_norm = q / (np.linalg.norm(q) + 1e-8)

        if self._backend == 'annoy':
            ids, dists = self._annoy.get_nns_by_vector(q_norm, top_k, include_distances=True)
            # Annoy angular distance → cosine similarity: cos = 1 - d²/2
            return [(int(i), 1.0 - d * d / 2.0) for i, d in zip(ids, dists)]

        elif self._backend == 'faiss':
            q_2d = q_norm.reshape(1, -1)
            scores, ids = self._faiss.search(q_2d, top_k)
            return [(int(ids[0][i]), float(scores[0][i]))
                    for i in range(top_k) if ids[0][i] >= 0]

        else:
            # Numpy fallback
            sims = self.normed @ q_norm
            top_idx = np.argpartition(sims, -top_k)[-top_k:]
            top_idx = top_idx[np.argsort(sims[top_idx])[::-1]]
            return [(int(i), float(sims[i])) for i in top_idx]

    def search_tokens(self, query: np.ndarray, top_k: int = 10,
                      filter_fn=None) -> List[Tuple[str, float]]:
        """Search and return cleaned token strings."""
        raw_results = self.search(query, top_k=top_k * 5)
        results = []
        for tid, sim in raw_results:
            token = self.id2token.get(tid, '')
            if filter_fn and not filter_fn(token):
                continue
            results.append((token, sim))
            if len(results) >= top_k:
                break
        return results

    def get_embedding(self, token_id: int) -> Optional[np.ndarray]:
        """Get original (un-normalized) embedding for a token ID."""
        if 0 <= token_id < self.n_vectors:
            return self.embeddings[token_id]
        return None

    @property
    def backend(self) -> str:
        return self._backend
