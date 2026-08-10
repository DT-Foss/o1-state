"""
Reservoir Bridge — Connect Pre-trained Readout to Pipeline
==========================================================
Lightweight ESN wrapper that uses the pre-trained W_out from
reservoir_readout.npz to convert embedding sequences into predictions.

No PS-Lifted topology needed at inference — the topology was used
during TRAINING to shape the readout weights. At inference we just
need: input → state update → W_out @ state → prediction.

The Z₂ structure (state_pos/state_neg) provides:
  - state_pos: forward-biased reservoir state
  - state_neg: backward-biased (negated input) state
  - Novelty = |pos - neg|: high = surprising input
"""

import os
import numpy as np
from typing import Optional


class ReservoirBridge:
    """Minimal ESN that uses pre-trained readout weights.

    Interface matches what FossPipeline._reservoir_retrieve() expects:
      .reset_state()
      .step(vec_512d)
      .state_pos  (2048d)
      .state_neg  (2048d)
      .predict(state_2048d) → 2048d embedding prediction
      .W_out  (not None = signal that reservoir is ready)
    """

    def __init__(self, data_dir: str = None, leak_rate: float = 0.3,
                 input_scaling: float = 0.1, spectral_radius: float = 0.95):
        self.available = False
        self.W_out = None

        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), 'data')

        readout_path = os.path.join(data_dir, 'reservoir_readout.npz')
        if not os.path.exists(readout_path):
            return

        data = np.load(readout_path)
        self.W_out = data['W_out'].astype(np.float32)  # (2048, 4096)
        self.reservoir_size = int(data['reservoir_size'])  # 2048
        self.input_dim = int(data['input_dim'])  # 512

        self.leak_rate = leak_rate
        self.n = self.reservoir_size  # 2048

        # Fixed random weights (deterministic seed for reproducibility)
        rng = np.random.RandomState(42)
        self.W_in = (rng.randn(self.n, self.input_dim) * input_scaling).astype(np.float32)

        # Sparse reservoir matrix — random sparse (10% density) scaled to spectral_radius
        # This is fast: 2048² × 0.1 = ~420K nonzeros
        density = 0.1
        W_res = rng.randn(self.n, self.n).astype(np.float32)
        mask = (rng.rand(self.n, self.n) < density).astype(np.float32)
        W_res *= mask
        # Scale to target spectral radius
        eigenvalues = np.abs(np.linalg.eigvals(W_res.astype(np.float64)))
        max_eig = np.max(eigenvalues) if len(eigenvalues) > 0 else 1.0
        if max_eig > 0:
            W_res *= spectral_radius / max_eig
        self.W_res = W_res.astype(np.float32)

        # Z₂ states
        self.state_pos = np.zeros(self.n, dtype=np.float32)
        self.state_neg = np.zeros(self.n, dtype=np.float32)

        self.available = True

    def reset_state(self):
        """Reset both Z₂ layers to zero."""
        self.state_pos = np.zeros(self.n, dtype=np.float32)
        self.state_neg = np.zeros(self.n, dtype=np.float32)

    def step(self, input_vec: np.ndarray):
        """Single reservoir step with Z₂ lifting.

        pos layer gets +input, neg layer gets -input.
        The difference (pos - neg) measures novelty/surprise.
        """
        inp = input_vec.astype(np.float32)

        # Positive layer: standard ESN update
        pre_pos = self.W_res @ self.state_pos + self.W_in @ inp
        new_pos = np.tanh(pre_pos)
        self.state_pos = ((1 - self.leak_rate) * self.state_pos
                          + self.leak_rate * new_pos)

        # Negative layer: negated input (Z₂ parity flip)
        pre_neg = self.W_res @ self.state_neg + self.W_in @ (-inp)
        new_neg = np.tanh(pre_neg)
        self.state_neg = ((1 - self.leak_rate) * self.state_neg
                          + self.leak_rate * new_neg)

    def predict(self, state: np.ndarray) -> Optional[np.ndarray]:
        """Map reservoir state to 2048d embedding prediction.

        W_out was trained via ridge regression on KB fact pairs:
          input = subject embedding → reservoir → state
          target = object embedding (the answer)

        state can be 2048d (avg of pos/neg) — we concat pos+neg for 4096d.
        """
        if self.W_out is None:
            return None

        # Build full 4096d state vector (pos concat neg)
        if state.shape[0] == self.n:
            # Single layer passed — use stored pos/neg
            full_state = np.concatenate([self.state_pos, self.state_neg])
        elif state.shape[0] == 2 * self.n:
            full_state = state
        else:
            # Pad or truncate
            full_state = np.zeros(2 * self.n, dtype=np.float32)
            full_state[:min(state.shape[0], 2 * self.n)] = state[:2 * self.n]

        # W_out @ state → 2048d prediction
        pred = self.W_out @ full_state.astype(np.float32)
        return pred
