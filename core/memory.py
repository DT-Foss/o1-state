"""
Layer 3: Hopfield Associative Memory
======================================
Energy-based pattern storage using Hopfield dynamics.

Key insight from T388: Using Foss WALKS for retrieval FAILS
(walks escape clusters). Instead, use Hopfield energy minimization
for STORAGE, and Foss consensus for COORDINATION between
multiple memory modules.

Modern Hopfield (Ramsauer 2020): attention IS Hopfield retrieval.
We use the classical form (binary patterns, Hebbian weights)
because it's gradient-free and fits the Markov framework.

Capacity: ~0.14N patterns for N neurons (classical).
Modern Hopfield: exponential capacity but needs continuous values.

Historical lineage: Hopfield (1982), Ramsauer et al. (2020)
"""

import numpy as np


class HopfieldMemory:
    """
    Classical Hopfield network with extensions.

    Stores binary patterns as energy minima.
    Retrieval = energy minimization from noisy query.

    Extensions beyond classical Hopfield:
    1. Temperature parameter for stochastic retrieval
    2. Pattern metadata (labels, timestamps)
    3. Forgetting mechanism (weighted Hebbian)
    4. Modern Hopfield energy (exponential, optional)
    """

    def __init__(self, n_neurons, temperature=0.0):
        """
        Args:
            n_neurons: number of neurons (pattern dimension)
            temperature: 0 = deterministic, >0 = stochastic (Boltzmann)
        """
        self.n = n_neurons
        self.temperature = temperature
        self.W = np.zeros((n_neurons, n_neurons))
        self.patterns = []
        self.metadata = []
        self.n_stored = 0

    def store(self, pattern, meta=None, weight=1.0):
        """
        Store a pattern using Hebbian learning.

        Args:
            pattern: binary vector {-1, +1}^n
            meta: optional metadata dict
            weight: importance weight (for weighted forgetting)
        """
        p = np.array(pattern, dtype=float)
        assert len(p) == self.n

        # Hebbian: ΔW = (1/N) * p ⊗ p
        self.W += weight * np.outer(p, p) / self.n
        np.fill_diagonal(self.W, 0)  # No self-connections

        self.patterns.append(p.copy())
        self.metadata.append(meta or {})
        self.n_stored += 1

    def store_batch(self, patterns, metas=None):
        """Store multiple patterns at once."""
        for i, p in enumerate(patterns):
            meta = metas[i] if metas else None
            self.store(p, meta)

    def retrieve(self, query, max_steps=100):
        """
        Retrieve nearest stored pattern via energy minimization.

        Asynchronous update: one neuron at a time, random order.

        Returns: (retrieved_pattern, steps, energy)
        """
        state = np.array(query, dtype=float)
        rng = np.random.RandomState()

        for step in range(max_steps):
            old_state = state.copy()

            # Async update in random order
            order = rng.permutation(self.n)
            for i in order:
                h = np.dot(self.W[i], state)

                if self.temperature > 0:
                    # Stochastic (Boltzmann) — clamp exponent to avoid overflow
                    exponent = np.clip(-2 * h / self.temperature, -500, 500)
                    prob = 1.0 / (1.0 + np.exp(exponent))
                    state[i] = 1.0 if rng.random() < prob else -1.0
                else:
                    # Deterministic
                    state[i] = 1.0 if h >= 0 else -1.0

            if np.array_equal(state, old_state):
                return state, step + 1, self.energy(state)

        return state, max_steps, self.energy(state)

    def energy(self, state):
        """Compute Hopfield energy: E = -0.5 * s^T W s"""
        return -0.5 * state @ self.W @ state

    def identify(self, retrieved):
        """Find which stored pattern is closest to retrieved state."""
        if not self.patterns:
            return -1, 0.0

        overlaps = [np.dot(retrieved, p) / self.n for p in self.patterns]
        best = np.argmax(overlaps)
        return best, overlaps[best]

    def capacity_check(self):
        """
        Check if we're near theoretical capacity (0.14N).
        Returns: (n_stored, capacity, utilization)
        """
        capacity = int(0.14 * self.n)
        utilization = self.n_stored / max(capacity, 1)
        return self.n_stored, capacity, utilization

    def forget(self, pattern_idx, decay=0.5):
        """
        Partially forget a pattern (weighted un-learning).

        Doesn't fully remove — reduces its attractor basin.
        """
        if 0 <= pattern_idx < len(self.patterns):
            p = self.patterns[pattern_idx]
            self.W -= decay * np.outer(p, p) / self.n
            np.fill_diagonal(self.W, 0)

    def clear(self):
        """Reset all memory."""
        self.W = np.zeros((self.n, self.n))
        self.patterns = []
        self.metadata = []
        self.n_stored = 0


class SparseDistributedMemory:
    """
    Kanerva's Sparse Distributed Memory (SDM).

    Inspired by HTM/Numenta: sparse binary codes, random addresses.
    Higher capacity than Hopfield, more biologically plausible.

    Address space: random binary vectors (hard locations)
    Read/Write: activate locations within Hamming distance d
    """

    def __init__(self, n_dims, n_locations=1000, access_radius=None):
        """
        Args:
            n_dims: dimension of patterns
            n_locations: number of hard locations (random addresses)
            access_radius: Hamming distance for activation (default: n/4)
        """
        self.n = n_dims
        self.n_loc = n_locations
        self.radius = access_radius or n_dims // 4

        # Random hard locations (binary)
        rng = np.random.RandomState(42)
        self.addresses = rng.choice([-1, 1], size=(n_locations, n_dims))

        # Counters at each location (accumulate patterns)
        self.counters = np.zeros((n_locations, n_dims))

    def _activated(self, address):
        """Find locations within Hamming radius of address."""
        # Hamming distance = (n - dot_product) / 2
        dots = self.addresses @ address
        distances = (self.n - dots) / 2
        return distances <= self.radius

    def write(self, address, data):
        """Write data at locations near address."""
        active = self._activated(address)
        self.counters[active] += data

    def read(self, address):
        """Read data from locations near address."""
        active = self._activated(address)
        if not np.any(active):
            return np.zeros(self.n)
        summed = self.counters[active].sum(axis=0)
        return np.sign(summed)  # Threshold to binary
