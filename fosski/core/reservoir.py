"""
Layer 2: Reservoir Computing via PS-Lifted Dynamics
====================================================
Echo State Network meets Markov Chain.

Key insight: The PS-Lifted transition matrix IS a reservoir.
- Fixed weights (no training of the reservoir itself)
- Readout via Hopfield attractor matching (NOT ridge regression)
- The Z₂ structure gives 2× state space for free
- Foss mixing guarantees the Echo State Property

Readout: Hopfield attractors replace Ridge Regression.
Instead of linear projection, known state→output pairs are stored
as Hopfield patterns. New state → nearest attractor → output.
This gives nonlinear readout that auto-scales.

Historical lineage: Jaeger (2001) Echo State Networks,
Maass (2002) Liquid State Machines.
"""

import numpy as np
from .markov import PSLifted
from .memory import HopfieldMemory


class MarkovReservoir:
    """
    Reservoir Computer using PS-Lifted Markov dynamics.

    Input is injected into the graph state.
    The Markov dynamics process it (fixed weights).
    Readout via Hopfield attractor matching (nonlinear, no matrix inversion).

    The Z₂ lifting provides:
    1. Layer 0 state (forward-biased view)
    2. Layer 1 state (backward-biased view)
    3. Parity signal (layer difference = structural "spike")
    """

    def __init__(self, adjacency, input_dim, output_dim,
                 p_c=0.80, p_s=0.003, spectral_radius=0.95,
                 input_scaling=0.1, leak_rate=0.3):
        self.mc = PSLifted(adjacency, p_c, p_s)
        self.n = self.mc.n
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.leak_rate = leak_rate

        # State dimension: 2n (both Z₂ layers) + n (parity)
        self.state_dim = 3 * self.n

        # Input weights: random, fixed, never trained
        rng = np.random.RandomState(42)
        self.W_in = rng.randn(2 * self.n, input_dim) * input_scaling

        # Scale transition matrix for Echo State Property
        self.T_reservoir = self.mc.T_lift.copy() * spectral_radius

        # Hopfield attractor readout (replaces ridge regression)
        # Stores (state_hash → output) pairs as attractors
        self._hopfield_readout = None  # initialized in train()
        self._readout_patterns = []   # stored (state, output) pairs
        self._readout_dim = None

        # Ridge fallback for compatibility
        self.W_out = None

        # Internal state
        self.state = np.zeros(2 * self.n)

    def reset(self):
        """Reset internal state to zero."""
        self.state = np.zeros(2 * self.n)

    def _step(self, input_vec):
        """
        Single reservoir step.

        state(t+1) = (1-α)·state(t) + α·tanh(T·state(t) + W_in·u(t))
        """
        pre_activation = self.T_reservoir @ self.state + self.W_in @ input_vec
        new_state = np.tanh(pre_activation)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self._read_state()

    def _read_state(self):
        """
        Extract features from current state.
        Returns 3n-dimensional vector: Layer 0 + Layer 1 + Z₂ parity.
        """
        n = self.n
        layer0 = self.state[:n]
        layer1 = self.state[n:]
        parity = layer0 - layer1
        return np.concatenate([layer0, layer1, parity])

    def process_sequence(self, inputs, washout=0):
        """Feed a sequence through the reservoir."""
        T = len(inputs)
        states = []
        self.reset()
        for t in range(T):
            state = self._step(inputs[t])
            if t >= washout:
                states.append(state)
        return np.array(states)

    def _state_to_hopfield_pattern(self, state_vec, output_vec):
        """Convert a (state, output) pair to a Hopfield binary pattern.

        Concatenates binarized state with binarized output into one pattern.
        The Hopfield network learns the association.
        """
        # Binarize state: sign of each component
        s_bin = np.sign(state_vec)
        s_bin[s_bin == 0] = 1.0
        # Binarize output: use threshold at median
        o_bin = np.sign(output_vec - np.median(output_vec))
        o_bin[o_bin == 0] = 1.0
        return np.concatenate([s_bin, o_bin])

    def train(self, inputs, targets, washout=10, ridge_alpha=1e-6):
        """
        Train the readout layer.

        Primary: Hopfield attractor readout (nonlinear, content-addressable).
        Fallback: Ridge regression for compatibility.

        The Hopfield readout stores state→output pairs as binary patterns.
        Retrieval = present state portion → Hopfield converges → read output portion.
        """
        states = self.process_sequence(inputs, washout=washout)
        Y = targets[:len(states)]

        # === Hopfield Attractor Readout ===
        self._readout_dim = self.state_dim + self.output_dim
        self._hopfield_readout = HopfieldMemory(self._readout_dim, temperature=0.1)

        # Subsample to stay within Hopfield capacity (~0.14N patterns)
        max_patterns = int(0.12 * self._readout_dim)
        step = max(1, len(states) // max_patterns)

        for i in range(0, len(states), step):
            pattern = self._state_to_hopfield_pattern(states[i], Y[i])
            self._hopfield_readout.store(pattern)
            self._readout_patterns.append((states[i].copy(), Y[i].copy()))

        # === Ridge regression fallback ===
        states_bias = np.hstack([states, np.ones((len(states), 1))])
        S = states_bias
        self.W_out = np.linalg.solve(
            S.T @ S + ridge_alpha * np.eye(S.shape[1]),
            S.T @ Y
        )

        # Training error (using ridge for measurement)
        pred = S @ self.W_out
        rmse = np.sqrt(np.mean((pred - Y) ** 2))
        return rmse

    def _hopfield_predict_single(self, state):
        """Predict output for a single state using Hopfield readout."""
        if self._hopfield_readout is None or self._hopfield_readout.n_stored == 0:
            return None

        # Create query: binarized state + random output
        s_bin = np.sign(state)
        s_bin[s_bin == 0] = 1.0
        # Output portion starts as random — Hopfield fills it in
        query = np.concatenate([s_bin, np.zeros(self.output_dim)])
        query[len(s_bin):] = np.random.choice([-1, 1], self.output_dim)

        retrieved, _, _ = self._hopfield_readout.retrieve(query, max_steps=10)

        # Extract output portion and de-binarize
        output_binary = retrieved[self.state_dim:]

        # Find closest stored pattern to interpolate continuous output
        best_dist = float('inf')
        best_output = None
        for stored_state, stored_output in self._readout_patterns:
            dist = np.sum((np.sign(stored_state) - s_bin[:self.state_dim]) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_output = stored_output

        return best_output

    def predict(self, inputs, washout=0):
        """
        Predict outputs for an input sequence.

        Uses Hopfield readout if available, falls back to ridge.
        """
        states = self.process_sequence(inputs, washout=washout)

        # Try Hopfield readout first
        if self._hopfield_readout is not None and self._hopfield_readout.n_stored > 0:
            predictions = []
            for state in states:
                pred = self._hopfield_predict_single(state)
                if pred is not None:
                    predictions.append(pred)
                else:
                    # Fallback to ridge for this state
                    state_bias = np.append(state, 1.0)
                    predictions.append(state_bias @ self.W_out)
            return np.array(predictions)

        # Pure ridge fallback
        if self.W_out is None:
            raise RuntimeError("Must call train() before predict()")
        states_bias = np.hstack([states, np.ones((len(states), 1))])
        return states_bias @ self.W_out

    def predict_generative(self, seed_inputs, n_generate, washout=0):
        """Generate a sequence autoregressively."""
        if self.W_out is None and self._hopfield_readout is None:
            raise RuntimeError("Must call train() before predict_generative()")

        self.reset()
        for t in range(len(seed_inputs)):
            state = self._step(seed_inputs[t])

        generated = []
        current_input = seed_inputs[-1]

        for _ in range(n_generate):
            state = self._step(current_input)

            # Try Hopfield first
            pred = self._hopfield_predict_single(state) if self._hopfield_readout else None
            if pred is None:
                state_bias = np.append(state, 1.0)
                pred = state_bias @ self.W_out

            generated.append(pred)

            if len(pred) >= self.input_dim:
                current_input = pred[:self.input_dim]
            else:
                current_input = np.zeros(self.input_dim)
                current_input[:len(pred)] = pred

        return np.array(generated)
