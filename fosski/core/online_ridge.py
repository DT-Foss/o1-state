"""
Online Ridge Regression — Learning during Inference (D5)
=========================================================
Source: doom/foss-v2/ARCHITECTURE.md lines 596-618

Rank-1 updates to the Ridge regression readout during inference.
No backpropagation, no gradient computation. Just outer products.

Key formula:
    XtX *= decay         # exponential forgetting
    XtY *= decay
    XtX += outer(state, state)
    XtY += outer(state, target)
    W_out = solve(XtX, XtY)

Properties:
- Online: updates after every successful query
- Forgetting: exponential decay (0.999) prevents saturation
- Efficient: rank-1 update is O(d²), not O(d³)
- No gradients: pure linear algebra

Combined with Sinkhorn on W_out, this can trigger grokking —
sudden generalization after enough online updates.
"""

import numpy as np
from typing import Optional


class OnlineRidge:
    """Online Ridge regression with exponential forgetting."""

    def __init__(self, input_dim: int, output_dim: int,
                 alpha: float = 1e-4, decay: float = 0.999):
        """
        Args:
            input_dim: reservoir state dimension (e.g., 4096 for expanded)
            output_dim: output embedding dimension (e.g., 2048)
            alpha: ridge regularization
            decay: exponential forgetting factor (0.999 = slow forget)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.decay = decay

        # Running statistics
        self.XtX = alpha * np.eye(input_dim, dtype=np.float32)
        self.XtY = np.zeros((input_dim, output_dim), dtype=np.float32)
        self.W_out: Optional[np.ndarray] = None
        self.n_updates = 0

    def update(self, state: np.ndarray, target: np.ndarray):
        """Rank-1 update with a new (state, target) pair.

        Args:
            state: reservoir state (input_dim,)
            target: target embedding (output_dim,)
        """
        state = state.astype(np.float32).ravel()
        target = target.astype(np.float32).ravel()

        # Exponential forgetting
        self.XtX *= self.decay
        self.XtY *= self.decay

        # Rank-1 update
        self.XtX += np.outer(state, state)
        self.XtY += np.outer(state, target)

        self.n_updates += 1

        # Recompute W_out periodically (every 10 updates)
        if self.n_updates % 10 == 0:
            self._solve()

    def _solve(self):
        """Solve for W_out = (XtX)^{-1} XtY."""
        try:
            self.W_out = np.linalg.solve(self.XtX, self.XtY).T
        except np.linalg.LinAlgError:
            # Singular matrix — add more regularization
            reg = self.XtX + self.alpha * 10 * np.eye(self.input_dim)
            self.W_out = np.linalg.solve(reg, self.XtY).T

    def predict(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Predict output from state."""
        if self.W_out is None:
            return None
        return self.W_out @ state.astype(np.float32)

    def force_solve(self):
        """Force W_out recomputation (call after batch of updates)."""
        self._solve()

    @property
    def ready(self) -> bool:
        return self.W_out is not None and self.n_updates >= 10
