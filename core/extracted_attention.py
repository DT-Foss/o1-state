"""
Extracted Attention — Pretrained Q·K^T from Qwen3-1.7B
======================================================
Uses the LEARNED Q/K/V projection matrices from Qwen3-1.7B as fixed
importance-weighting for the Reservoir ESN.

What the Transformer learned in these matrices:
  - Q_proj: "What am I looking for?" — query features (16 heads × 128d)
  - K_proj: "What am I?" — key features (8 KV heads × 128d, GQA)
  - V_proj: "What is my content?" — value features (8 KV heads × 128d)
  - q_norm/k_norm: per-head normalization (QK-Norm, Qwen3 feature)

Prefers DEEP layers (10+18) where specialization happens.
Falls back to shallow layers (0-2) if deep not available.

We use these for TWO things:
  1. IMPORTANCE WEIGHTING: Q·K^T scores tell which tokens matter
     → Proper multi-head GQA with QK-Norm, NO RoPE (position-free)
     → Pooled-query attention: avg(Q) · each K / sqrt(d) → softmax
     → O(n) complexity (single pooled query, not n×n)

  2. VALUE FEATURES: V_proj transforms raw embeddings to content features
     → Used as reservoir input (richer than raw embeddings)

NO self-attention loop. NO autoregressive generation.
Just the learned linear transforms applied as feature extractors.
"""

import numpy as np
import os


def _rms_norm(x, weight, eps=1e-6):
    """RMSNorm: x * weight / sqrt(mean(x²) + eps)."""
    rms = np.sqrt(np.mean(x ** 2) + eps)
    return (x / rms) * weight


class ExtractedAttention:
    """Pretrained Qwen3-1.7B attention as fixed importance weighting.

    Multi-head GQA with QK-Norm. Position-free (no RoPE).
    Returns per-token importance weights for the Reservoir ESN.

    Prefers deep layers (10+18) for better token discrimination.
    """

    def __init__(self, data_path=None):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Try deep layers first, fall back to shallow
        self._deep_path = os.path.join(base, 'data', 'qwen3_deep_layers.npz')
        if data_path is None:
            self._shallow_path = os.path.join(base, 'data', 'qwen3_attention_layers.npz')
        else:
            self._shallow_path = data_path

        self._loaded = False
        self._using_deep = False

        self.input_norm = None
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.q_norm = None
        self.k_norm = None

        self.n_q_heads = 16
        self.n_kv_heads = 8
        self.head_dim = 128

    def _load(self):
        if self._loaded:
            return

        # Prefer deep layers (10+18) — actual specialization
        if os.path.exists(self._deep_path):
            data = np.load(self._deep_path)
            # Average layers 10 and 18 for robust representation
            layers = [10, 18]
            self.input_norm = np.mean(
                [data[f'input_norm_{i}'].astype(np.float32) for i in layers], axis=0)
            self.q_proj = np.mean(
                [data[f'q_proj_{i}'].astype(np.float32) for i in layers], axis=0)
            self.k_proj = np.mean(
                [data[f'k_proj_{i}'].astype(np.float32) for i in layers], axis=0)
            self.v_proj = np.mean(
                [data[f'v_proj_{i}'].astype(np.float32) for i in layers], axis=0)
            self.q_norm = np.mean(
                [data[f'q_norm_{i}'].astype(np.float32) for i in layers], axis=0)
            self.k_norm = np.mean(
                [data[f'k_norm_{i}'].astype(np.float32) for i in layers], axis=0)
            self._using_deep = True
            self._loaded = True

        # Fallback to shallow layers (0-2)
        elif os.path.exists(self._shallow_path):
            data = np.load(self._shallow_path)

            def avg(name):
                return np.mean([data[f'{name}_{i}'].astype(np.float32) for i in range(3)], axis=0)

            self.input_norm = avg('input_norm')
            self.q_proj = avg('q_proj')
            self.k_proj = avg('k_proj')
            self.v_proj = avg('v_proj')
            self.q_norm = avg('q_norm')
            self.k_norm = avg('k_norm')
            self._using_deep = False
            self._loaded = True

        # Random projection 1024 → 512 for reservoir
        if self._loaded:
            rng = np.random.default_rng(42)
            self._proj_512 = rng.choice(
                [-1, 0, 0, 0, 0, 1], size=(1024, 512)
            ).astype(np.float32) * np.sqrt(3.0 / 512)

    @property
    def available(self):
        self._load()
        return self.q_proj is not None

    @property
    def using_deep_layers(self):
        self._load()
        return self._using_deep

    def _compute_qk(self, raw_embeddings):
        """Compute Q and K with proper multi-head GQA and QK-Norm.

        Args:
            raw_embeddings: (n, 2048) raw Qwen3 embeddings

        Returns:
            queries: (n, n_q_heads, head_dim) normalized Q per head
            keys:    (n, n_kv_heads, head_dim) normalized K per head
        """
        n = len(raw_embeddings)
        embs = np.array(raw_embeddings, dtype=np.float32)

        # RMSNorm
        normed = np.array([_rms_norm(e, self.input_norm) for e in embs])

        # Project
        q_flat = normed @ self.q_proj.T   # (n, 2048)
        k_flat = normed @ self.k_proj.T   # (n, 1024)

        # Reshape to heads
        q = q_flat.reshape(n, self.n_q_heads, self.head_dim)   # (n, 16, 128)
        k = k_flat.reshape(n, self.n_kv_heads, self.head_dim)  # (n, 8, 128)

        # QK-Norm: per-head RMSNorm + learned scale
        for h in range(self.n_q_heads):
            rms = np.sqrt(np.mean(q[:, h, :] ** 2, axis=1, keepdims=True) + 1e-6)
            q[:, h, :] = (q[:, h, :] / rms) * self.q_norm[None, :]

        for h in range(self.n_kv_heads):
            rms = np.sqrt(np.mean(k[:, h, :] ** 2, axis=1, keepdims=True) + 1e-6)
            k[:, h, :] = (k[:, h, :] / rms) * self.k_norm[None, :]

        return q, k

    def compute_importance(self, raw_embeddings, temperature=2.0):
        """Compute per-token importance via Key-Space Contrast.

        The Transformer learned WHAT makes a good key representation.
        Content words produce distinctive keys; stop words produce generic ones.

        Method: Multi-head key contrast
          1. Project all tokens through K_proj with QK-Norm
          2. Per head: compute cosine distance from mean key
          3. Average distances across heads
          4. Softmax with temperature

        This captures the Transformer's learned notion of "informativeness"
        without needing RoPE, causal masks, or 28-layer refinement.

        Args:
            raw_embeddings: list of (2048,) raw Qwen3 embeddings
            temperature: >1 = smoother weights, <1 = sharper

        Returns:
            (n,) importance weights summing to 1
        """
        self._load()
        n = len(raw_embeddings)
        if self.q_proj is None or n == 0:
            return np.ones(max(n, 1)) / max(n, 1)
        if n == 1:
            return np.array([1.0])

        _, k = self._compute_qk(raw_embeddings)  # k: (n, 8, 128)

        # Per-head: cosine distance from mean key
        scores = np.zeros(n, dtype=np.float32)

        for h in range(self.n_kv_heads):
            k_h = k[:, h, :]  # (n, 128)

            # Normalize per token
            k_norms = np.linalg.norm(k_h, axis=1, keepdims=True)
            k_normed = k_h / np.clip(k_norms, 1e-8, None)

            # Mean key for this head
            mean_k = k_normed.mean(axis=0)
            mean_k /= np.linalg.norm(mean_k) + 1e-8

            # Cosine distance from mean = specificity
            cos_sims = k_normed @ mean_k  # (n,)
            distances = 1.0 - cos_sims    # high = specific/important

            scores += distances

        # Average across heads
        scores /= self.n_kv_heads

        # Softmax with temperature
        scores = scores / temperature
        scores = scores - scores.max()
        weights = np.exp(scores)
        weights /= weights.sum() + 1e-8

        return weights

    def extract_value_features(self, raw_embedding):
        """Extract 512d value features from a raw 2048d embedding.

        V_proj captures what the token MEANS (content features).
        These are richer than raw embeddings for semantic tasks.
        """
        self._load()
        if self.v_proj is None:
            return None
        normed = _rms_norm(raw_embedding.astype(np.float32), self.input_norm)
        value = self.v_proj @ normed  # (1024,)
        return (value @ self._proj_512).astype(np.float32)

    def attend_and_pool(self, raw_embeddings, temperature=2.0):
        """Importance-weighted pooling of value features.

        1. Compute importance via Q·K^T (multi-head GQA)
        2. Extract V features per token
        3. Weighted sum → single 512d vector

        Returns:
            pooled: (512,) importance-weighted feature vector
            weights: (n,) per-token importance
        """
        self._load()
        n = len(raw_embeddings)
        if self.v_proj is None or n == 0:
            return np.zeros(512, dtype=np.float32), np.ones(max(n, 1)) / max(n, 1)

        weights = self.compute_importance(raw_embeddings, temperature)

        # Value features with RMSNorm
        embs = np.array(raw_embeddings, dtype=np.float32)
        normed = np.array([_rms_norm(e, self.input_norm) for e in embs])
        values = normed @ self.v_proj.T  # (n, 1024)

        # Weighted pool → project to 512d
        pooled_1024 = (weights[:, None] * values).sum(axis=0)
        pooled_512 = (pooled_1024 @ self._proj_512).astype(np.float32)

        return pooled_512, weights
