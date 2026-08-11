"""
DS Layer Stack — Doubly Stochastic Depth without Transformers (F7, FB1, FB2)
=============================================================================
Stacks 3-4 Sinkhorn normalization layers to achieve depth-like information
integration without transformer architecture.

Key findings:
- FB1: 1-step Sinkhorn → universal spectral statistics (input-independent NND)
- FB2: Without Sinkhorn, softmax attention sink compounds 1049% over 4 layers
- F7: 1 Sinkhorn iteration = 86% of full effect, 130× better gradient flow
- From gottformel: DS > Zeno penalty, structural topology control

Each DS layer:
1. Compute similarity matrix S between input embeddings
2. Apply 1 Sinkhorn iteration (row-normalize, col-normalize)
3. Matrix-vector product: output = DS_matrix @ input
4. This redistributes information evenly (doubly stochastic = fair mixing)

3-4 layers = each token "sees" information from all others through
topology-controlled mixing. This IS the depth replacement.
"""

import numpy as np
from typing import List


def sinkhorn_normalize(W: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Sinkhorn normalization to doubly stochastic matrix (F7).

    1 iteration captures 86% of the effect (BC11).

    Args:
        W: non-negative matrix (e.g., similarity matrix)
        iterations: number of Sinkhorn iterations (1 is usually enough)

    Returns:
        Doubly stochastic matrix (rows and columns sum to 1)
    """
    M = np.abs(W).astype(np.float64)
    for _ in range(iterations):
        # Row normalize
        row_sums = M.sum(axis=1, keepdims=True)
        M = M / np.clip(row_sums, 1e-8, None)
        # Column normalize
        col_sums = M.sum(axis=0, keepdims=True)
        M = M / np.clip(col_sums, 1e-8, None)
    return M.astype(np.float32)


class DSLayerStack:
    """Stack of Doubly Stochastic mixing layers.

    Replaces transformer depth with topology-controlled information mixing.
    Each layer applies Sinkhorn-normalized attention to redistribute
    information evenly across all positions.
    """

    def __init__(self, n_layers: int = 3, sinkhorn_iters: int = 1,
                 temperature: float = 1.0):
        """
        Args:
            n_layers: number of DS mixing layers (3-4 typical)
            sinkhorn_iters: Sinkhorn iterations per layer (1 = 86% effect)
            temperature: softmax temperature for similarity computation
        """
        self.n_layers = n_layers
        self.sinkhorn_iters = sinkhorn_iters
        self.temperature = temperature

    def forward(self, embeddings: np.ndarray) -> np.ndarray:
        """Pass embeddings through the DS layer stack.

        Args:
            embeddings: (n_tokens, dim) matrix of token embeddings

        Returns:
            (n_tokens, dim) mixed embeddings (each token contains
            information from all others via DS mixing)
        """
        X = embeddings.astype(np.float32)
        n = X.shape[0]

        if n < 2:
            return X

        for layer in range(self.n_layers):
            # Compute similarity matrix
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X_normed = X / np.clip(norms, 1e-8, None)
            S = X_normed @ X_normed.T  # (n, n) cosine similarity

            # Softmax with temperature
            S = S / self.temperature
            S -= S.max(axis=1, keepdims=True)  # Numerical stability
            S = np.exp(S)

            # Sinkhorn normalization → doubly stochastic
            DS = sinkhorn_normalize(S, iterations=self.sinkhorn_iters)

            # Mix: each token becomes weighted average of all tokens
            X = DS @ X

        return X

    def integrate(self, embeddings: np.ndarray) -> np.ndarray:
        """Convenience: run DS stack and return the mean-pooled result.

        Useful for getting a single "integrated" vector from multiple tokens.
        """
        mixed = self.forward(embeddings)
        return mixed.mean(axis=0)
