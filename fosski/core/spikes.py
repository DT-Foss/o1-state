"""
Layer 6: Z₂ Spike Layer — Event-Driven Processing
=====================================================
The Z₂ parity of the PS-Lifted chain is a binary signal:
Layer 0 vs Layer 1. When they DISAGREE, that's a "spike."

This maps directly to Spiking Neural Networks (SNNs):
- Spike = parity flip (Layer 0 - Layer 1 ≠ 0)
- Spike timing = when the flip happens
- Spike rate = how often layers disagree

Energy advantage: Only compute when a spike occurs.
No spike = no update needed.

Historical lineage: Maass (1997) SNNs, Intel Loihi, IBM TrueNorth
"""

import numpy as np
from .markov import PSLifted


class SpikeEncoder:
    """
    Convert continuous signals to spike trains via Z₂ parity.

    Input signal → inject into PS-Lifted graph → monitor parity.
    Strong input = frequent parity flips = high spike rate.
    Weak input = rare flips = low rate.
    """

    def __init__(self, adjacency, p_c=0.80, p_s=0.003, threshold=0.01):
        """
        Args:
            adjacency: graph for the Markov chain
            threshold: minimum parity magnitude to count as spike
        """
        self.mc = PSLifted(adjacency, p_c, p_s)
        self.n = self.mc.n
        self.threshold = threshold

    def encode(self, signal, n_steps=50):
        """
        Encode a continuous signal as spike train.

        Args:
            signal: (n,) continuous values to encode

        Returns:
            spike_train: (n_steps, n) binary spike matrix
            spike_times: list of (time, neuron_id) tuples
        """
        n = self.n
        state = np.zeros(2 * n)
        state[:n] = signal

        spike_train = np.zeros((n_steps, n))
        spike_times = []

        for t in range(n_steps):
            state = self.mc.T_lift @ state

            # Parity = difference between layers
            parity = state[:n] - state[n:]

            # Spikes where parity exceeds threshold
            spikes = np.abs(parity) > self.threshold
            spike_train[t] = spikes.astype(float)

            for neuron in np.where(spikes)[0]:
                spike_times.append((t, neuron, np.sign(parity[neuron])))

        return spike_train, spike_times

    def rate_code(self, signal, n_steps=50):
        """
        Encode signal as spike rates (average spike frequency per neuron).
        """
        spike_train, _ = self.encode(signal, n_steps)
        return spike_train.mean(axis=0)


class SpikeDecoder:
    """
    Decode spike trains back to continuous values.

    Uses population rate coding: spike rate ∝ encoded value.
    """

    def __init__(self, n_neurons, output_dim):
        self.n = n_neurons
        self.output_dim = output_dim
        self.W_decode = None

    def fit(self, spike_rates, targets, ridge_alpha=1e-4):
        """
        Train decoder (linear) from spike rates to outputs.
        """
        X = np.column_stack([spike_rates, np.ones(len(spike_rates))])
        Y = targets

        self.W_decode = np.linalg.solve(
            X.T @ X + ridge_alpha * np.eye(X.shape[1]),
            X.T @ Y
        )
        pred = X @ self.W_decode
        return np.sqrt(np.mean((pred - Y) ** 2))

    def decode(self, spike_rates):
        """Decode spike rates to continuous output."""
        if self.W_decode is None:
            raise RuntimeError("Must call fit() first")
        X = np.append(spike_rates, 1.0)
        return X @ self.W_decode


class EventDrivenProcessor:
    """
    Event-driven processing: only compute when spikes occur.

    The core energy-saving mechanism. Instead of running
    the full Markov chain every step, only update nodes
    that received a spike.

    Energy savings: proportional to spike sparsity.
    If 90% of neurons are silent, save ~90% compute.
    """

    def __init__(self, adjacency, p_c=0.80, p_s=0.003):
        self.mc = PSLifted(adjacency, p_c, p_s)
        self.n = self.mc.n
        self.state = np.zeros(2 * self.n)
        self.ops_count = 0

    def inject_spike(self, neuron_id, magnitude=1.0, layer=0):
        """Inject a spike at a specific neuron."""
        if layer == 0:
            self.state[neuron_id] += magnitude
        else:
            self.state[self.n + neuron_id] += magnitude

    def step_sparse(self):
        """
        Update only neurons with nonzero state (event-driven).

        Returns: (output_spikes, n_operations)
        """
        n = self.n
        T = self.mc.T_lift
        new_state = np.zeros(2 * n)
        ops = 0

        # Only process nonzero neurons
        active = np.where(np.abs(self.state) > 1e-8)[0]

        for i in active:
            # Propagate this neuron's state to neighbors
            new_state += self.state[i] * T[i]
            ops += np.count_nonzero(T[i])

        self.state = new_state
        self.ops_count += ops

        # Detect output spikes (parity events)
        parity = self.state[:n] - self.state[n:]
        spikes = np.abs(parity) > 0.01

        return spikes, ops

    @property
    def sparsity(self):
        """Fraction of silent neurons."""
        active = np.count_nonzero(np.abs(self.state) > 1e-8)
        return 1.0 - active / (2 * self.n)
