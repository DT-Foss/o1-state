"""
Ricci-Curvature Attention for Language
=======================================
Ported from DOOM Foss-V2 Visual Cortex (modules/visual_cortex.py).

Original: 2D Möbius grid → Ricci scalar → attention peaks on game screen.
Language: 1D Möbius chain → Ricci scalar → attention peaks in token sequence.

The curvature R = -2·e^(-2φ)·∇²φ where φ = -log(1-λ²).
High curvature = information-rich position = "pay attention here."

This is O(n) attention (n = sequence length), not O(n²) like Transformers.
No key-query-value. No softmax. Pure geometry.

The eigenvalue landscape λ is driven by token embeddings:
  - Semantically important tokens → high |λ| → high curvature
  - Stop words / filler → low |λ| → flat curvature
"""

import numpy as np


def mobius_coupling(lam, v):
    """Möbius transform: f(λ,v) = (λ+v)/(1+λv). Exact Lorentz boost."""
    return (lam + v) / (1.0 + lam * v)


def ricci_curvature_1d(lambdas):
    """Ricci scalar on 1D chain.

    R = -2·e^(-2φ)·∂²φ/∂x² where φ = -log(1-λ²).
    Zero-padded boundary (no periodic wrap — sentences have start/end).

    Args:
        lambdas: (n,) eigenvalue array, values in (-1, 1)

    Returns:
        (n,) curvature at each position
    """
    lam = np.clip(np.abs(lambdas), 0.001, 0.9999)
    phi = -np.log(1.0 - lam ** 2)

    # 1D Laplacian with zero-padding (non-periodic for language)
    n = len(phi)
    padded = np.pad(phi, 1, mode='edge')  # replicate boundary
    laplacian = padded[:-2] + padded[2:] - 2.0 * phi

    R = -2.0 * np.exp(-2.0 * phi) * laplacian
    return R


class RicciAttention:
    """O(n) attention via Ricci curvature on token embedding landscape.

    For each token sequence:
    1. Map embeddings → eigenvalue chain λ ∈ (-1, 1)
    2. Run Möbius diffusion (coupling between neighbors)
    3. Compute Ricci curvature at each position
    4. High curvature = attention peak = important token

    Returns attention weights (not softmax — geometric, unnormalized).
    """

    def __init__(self, coupling=0.3, n_diffusion_steps=3, decay=0.95):
        self.coupling = coupling
        self.n_steps = n_diffusion_steps
        self.decay = decay

    def attend(self, embeddings):
        """Compute attention weights for a sequence of embeddings.

        Args:
            embeddings: list of (dim,) embedding vectors, one per token

        Returns:
            weights: (n,) attention weight per token position
            curvature: (n,) raw curvature values
        """
        n = len(embeddings)
        if n == 0:
            return np.array([]), np.array([])
        if n == 1:
            return np.array([1.0]), np.array([1.0])

        # Map embeddings to eigenvalues λ ∈ (0, 0.9)
        # Use CONTRAST with context mean — content words stand out, stop words blend in
        emb_array = np.array(embeddings, dtype=np.float32)
        context_mean = emb_array.mean(axis=0)

        # Cosine distance from context centroid = specificity
        norms = np.linalg.norm(emb_array, axis=1)
        cm_norm = np.linalg.norm(context_mean) + 1e-8
        cos_sims = (emb_array @ context_mean) / (norms * cm_norm + 1e-8)
        specificity = 1.0 - cos_sims  # high = far from mean = specific/important

        # Normalize to [0.1, 0.85]
        s_min, s_max = specificity.min(), specificity.max()
        if s_max - s_min > 1e-8:
            lambdas = 0.1 + 0.75 * (specificity - s_min) / (s_max - s_min)
        else:
            lambdas = np.full(n, 0.4)

        # Boost: neighbor contrast (tokens different from both neighbors)
        for i in range(n):
            cos_prev = cos_next = 0.0
            if i > 0:
                cos_prev = float(np.dot(embeddings[i], embeddings[i - 1]) / (
                    norms[i] * norms[i - 1] + 1e-8))
            if i < n - 1:
                cos_next = float(np.dot(embeddings[i], embeddings[i + 1]) / (
                    norms[i] * norms[i + 1] + 1e-8))
            contrast = 1.0 - (cos_prev + cos_next) / max(1, (i > 0) + (i < n - 1))
            lambdas[i] = np.clip(lambdas[i] + contrast * 0.1, 0.05, 0.95)

        # Möbius diffusion steps
        for _ in range(self.n_steps):
            new_lam = lambdas.copy()
            for i in range(n):
                # Coupling from neighbors
                v = 0.0
                count = 0
                if i > 0:
                    v += (lambdas[i - 1] - lambdas[i]) * self.coupling
                    count += 1
                if i < n - 1:
                    v += (lambdas[i + 1] - lambdas[i]) * self.coupling
                    count += 1
                if count > 0:
                    v /= count
                new_lam[i] = mobius_coupling(lambdas[i], np.clip(v, -0.5, 0.5))
            lambdas = np.clip(new_lam, -0.9999, 0.9999)

        # Decay
        lambdas *= self.decay

        # Compute curvature
        curvature = ricci_curvature_1d(lambdas)

        # Attention weights: λ (base importance) × (1 + |curvature|) (geometric boost)
        # λ captures semantic specificity, curvature captures local contrast
        weights = lambdas * (1.0 + np.abs(curvature))
        total = weights.sum()
        if total > 0:
            weights /= total

        return weights, curvature

    def weighted_pool(self, embeddings):
        """Pool embeddings using Ricci attention weights.

        Returns a single vector that emphasizes semantically important tokens.
        O(n) complexity — no key-query-value, no softmax.
        """
        weights, _ = self.attend(embeddings)
        if len(weights) == 0:
            return np.zeros_like(embeddings[0]) if embeddings else np.zeros(1)

        result = np.zeros_like(embeddings[0], dtype=np.float64)
        for i, emb in enumerate(embeddings):
            result += weights[i] * emb
        return result.astype(np.float32)
