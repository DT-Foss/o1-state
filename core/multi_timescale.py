"""
Multi-Timescale Reservoir — Fast/Medium/Slow Bands (D4)
========================================================
Source: doom/foss-v2/ARCHITECTURE.md

Three reservoir bands operating at different timescales:
- Fast  (512 nodes, leak=0.5): Token-level features, rapid response
- Medium (1024 nodes, leak=0.3): Sentence-level semantics
- Slow  (512 nodes, leak=0.1): Context accumulation, long-term memory

Z₂ between Fast and Slow bands = mismatch detector.
When fast and slow disagree strongly → novel or contradictory input.

Each band is an independent ESN with its own leak rate.
The combined state [fast; medium; slow] gives multi-resolution representation.
"""

import numpy as np
from typing import Tuple, Optional


class MultiTimescaleReservoir:
    """Three-band reservoir with different temporal dynamics."""

    def __init__(self, input_dim: int = 512,
                 fast_size: int = 512, medium_size: int = 1024, slow_size: int = 512,
                 spectral_radius: float = 0.95, seed: int = 42):
        """
        Args:
            input_dim: dimension of input embeddings
            fast_size: fast band neurons (leak=0.5)
            medium_size: medium band neurons (leak=0.3)
            slow_size: slow band neurons (leak=0.1)
            spectral_radius: target spectral radius for reservoir weights
            seed: random seed for reproducibility
        """
        self.input_dim = input_dim
        self.fast_size = fast_size
        self.medium_size = medium_size
        self.slow_size = slow_size
        self.total_size = fast_size + medium_size + slow_size

        rng = np.random.default_rng(seed)

        # Initialize bands
        self._bands = []
        for size, leak in [(fast_size, 0.5), (medium_size, 0.3), (slow_size, 0.1)]:
            W_in = (rng.standard_normal((size, input_dim)) * 0.1).astype(np.float32)
            W = rng.standard_normal((size, size)).astype(np.float32)
            # Scale to spectral radius
            eig_max = np.max(np.abs(np.linalg.eigvals(W)))
            if eig_max > 0:
                W *= spectral_radius / eig_max
            state = np.zeros(size, dtype=np.float32)
            self._bands.append({
                'W_in': W_in, 'W': W, 'state': state,
                'leak': leak, 'size': size
            })

        self.W_out = None  # Trained readout

    def reset(self):
        """Reset all band states to zero."""
        for band in self._bands:
            band['state'] = np.zeros(band['size'], dtype=np.float32)

    def step(self, x: np.ndarray) -> np.ndarray:
        """Feed input through all bands and return combined state.

        Args:
            x: input vector (input_dim,)

        Returns:
            combined state (total_size,)
        """
        states = []
        for band in self._bands:
            pre = band['W_in'] @ x + band['W'] @ band['state']
            new_state = np.tanh(pre)
            # Leaky integration
            band['state'] = (1 - band['leak']) * band['state'] + band['leak'] * new_state
            states.append(band['state'])

        return np.concatenate(states)

    @property
    def state_fast(self) -> np.ndarray:
        return self._bands[0]['state']

    @property
    def state_medium(self) -> np.ndarray:
        return self._bands[1]['state']

    @property
    def state_slow(self) -> np.ndarray:
        return self._bands[2]['state']

    def z2_mismatch(self) -> float:
        """Z₂ mismatch between fast and slow bands.

        High mismatch = fast dynamics disagree with slow context.
        This signals novelty or contradiction.

        Returns:
            mismatch score (0 = agreement, >0.5 = strong disagreement)
        """
        fast = self.state_fast
        slow = self.state_slow

        # Project to same dimension if different sizes
        min_dim = min(len(fast), len(slow))
        diff = fast[:min_dim] - slow[:min_dim]
        return float(np.mean(np.abs(diff)))

    def combined_state(self) -> np.ndarray:
        """Get the full combined state [fast; medium; slow]."""
        return np.concatenate([b['state'] for b in self._bands])

    def predict(self, state: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
        """Predict output from combined state."""
        if self.W_out is None:
            return None
        if state is None:
            state = self.combined_state()
        return self.W_out @ state
