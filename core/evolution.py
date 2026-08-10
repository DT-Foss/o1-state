"""
Layer 5: Evolutionary Architecture Search
============================================
Find optimal graph topologies for the Markov reservoir.

MAP-Elites / Quality-Diversity over graph structures.
The evolved topology determines HOW the reservoir processes
information. Different topologies = different computational
properties.

What we evolve:
- Graph adjacency (which nodes connect)
- Edge weights (connection strength)
- Number of nodes (reservoir size)
- Bottleneck structure (for consensus speedup)

Fitness = task performance (prediction accuracy, convergence speed)
Diversity = graph properties (spectral gap, clustering, diameter)

Historical lineage:
- Holland (1975) Genetic Algorithms
- Stanley (2002) NEAT (NeuroEvolution)
- Mouret (2015) MAP-Elites
"""

import numpy as np
from .markov import PSLifted


class GraphGenome:
    """A graph topology encoded for evolutionary search."""

    def __init__(self, n_nodes, adjacency=None, rng=None):
        self.n = n_nodes
        if rng is None:
            rng = np.random.RandomState()
        self.rng = rng

        if adjacency is not None:
            self.A = adjacency.copy()
        else:
            # Random sparse graph
            self.A = np.zeros((n_nodes, n_nodes))
            p = 2.0 / n_nodes  # Sparse
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):
                    if rng.random() < p:
                        self.A[i, j] = self.A[j, i] = 1.0

            # Ensure connected
            self._ensure_connected()

        self.fitness = None
        self.features = None

    def _ensure_connected(self):
        """Add edges until graph is connected."""
        n = self.n
        visited = set()
        queue = [0]
        visited.add(0)
        while queue:
            node = queue.pop(0)
            for j in range(n):
                if self.A[node, j] > 0 and j not in visited:
                    visited.add(j)
                    queue.append(j)

        unvisited = set(range(n)) - visited
        while unvisited:
            u = self.rng.choice(list(visited))
            v = self.rng.choice(list(unvisited))
            self.A[u, v] = self.A[v, u] = 1.0
            visited.add(v)
            unvisited.remove(v)

    def mutate(self, rate=0.1):
        """Mutate by adding/removing edges."""
        child_A = self.A.copy()
        n = self.n

        for i in range(n):
            for j in range(i + 1, n):
                if self.rng.random() < rate:
                    child_A[i, j] = 1.0 - child_A[i, j]
                    child_A[j, i] = child_A[i, j]

        child = GraphGenome(n, child_A, self.rng)
        child._ensure_connected()
        return child

    def crossover(self, other):
        """Crossover: take upper triangle from self, lower from other."""
        child_A = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.rng.random() < 0.5:
                    child_A[i, j] = child_A[j, i] = self.A[i, j]
                else:
                    child_A[i, j] = child_A[j, i] = other.A[i, j]

        child = GraphGenome(self.n, child_A, self.rng)
        child._ensure_connected()
        return child

    def compute_features(self):
        """Compute descriptive features for MAP-Elites grid."""
        mc = PSLifted(self.A)
        degrees = self.A.sum(axis=1)
        n_edges = int(self.A.sum() / 2)

        self.features = {
            'gamma_base': mc.gamma_base,
            'gamma_lift': mc.gamma_lift,
            'ratio': mc.ratio,
            'density': n_edges / max(self.n * (self.n - 1) / 2, 1),
            'avg_degree': degrees.mean(),
            'max_degree': degrees.max(),
            'degree_std': degrees.std(),
        }
        return self.features


class TopologyEvolver:
    """
    MAP-Elites style evolutionary search over graph topologies.

    Maintains a grid of elite solutions indexed by behavioral features.
    Each cell contains the best topology found for that feature combination.
    """

    def __init__(self, n_nodes, fitness_fn, feature_bins=None):
        """
        Args:
            n_nodes: number of nodes in evolved graphs
            fitness_fn: callable(adjacency) -> float (higher = better)
            feature_bins: dict mapping feature_name -> list of bin edges
        """
        self.n_nodes = n_nodes
        self.fitness_fn = fitness_fn
        self.rng = np.random.RandomState(42)

        # Default features for MAP-Elites grid
        if feature_bins is None:
            feature_bins = {
                'density': [0, 0.1, 0.2, 0.3, 0.5, 1.0],
                'ratio': [0, 0.5, 1.0, 2.0, 5.0, 50.0],
            }
        self.feature_bins = feature_bins

        # Grid: store best genome per cell
        self.grid = {}
        self.history = []
        self.generation = 0

    def _cell_key(self, features):
        """Map features to grid cell."""
        key = []
        for fname, bins in self.feature_bins.items():
            val = features.get(fname, 0)
            bin_idx = np.searchsorted(bins, val) - 1
            bin_idx = max(0, min(bin_idx, len(bins) - 2))
            key.append(bin_idx)
        return tuple(key)

    def _evaluate(self, genome):
        """Evaluate a genome: compute fitness and features."""
        genome.compute_features()
        genome.fitness = self.fitness_fn(genome.A)
        return genome

    def initialize(self, n_initial=50):
        """Generate initial random population."""
        for _ in range(n_initial):
            genome = GraphGenome(self.n_nodes, rng=self.rng)
            self._evaluate(genome)

            key = self._cell_key(genome.features)
            if key not in self.grid or genome.fitness > self.grid[key].fitness:
                self.grid[key] = genome

            self.history.append(genome.fitness)

    def step(self, n_offspring=20, mutation_rate=0.1):
        """One generation: select parents, mutate/crossover, evaluate."""
        self.generation += 1
        elites = list(self.grid.values())

        if not elites:
            self.initialize()
            return

        for _ in range(n_offspring):
            if self.rng.random() < 0.7 or len(elites) < 2:
                # Mutation
                parent = elites[self.rng.randint(len(elites))]
                child = parent.mutate(mutation_rate)
            else:
                # Crossover
                p1, p2 = self.rng.choice(len(elites), 2, replace=False)
                child = elites[p1].crossover(elites[p2])

            self._evaluate(child)

            key = self._cell_key(child.features)
            if key not in self.grid or child.fitness > self.grid[key].fitness:
                self.grid[key] = child

            self.history.append(child.fitness)

    def evolve(self, n_generations=50, n_offspring=20, mutation_rate=0.1):
        """Run full evolution."""
        self.initialize()

        for _ in range(n_generations):
            self.step(n_offspring, mutation_rate)

        return self.best()

    def best(self):
        """Return the best genome across all cells."""
        if not self.grid:
            return None
        return max(self.grid.values(), key=lambda g: g.fitness)

    def coverage(self):
        """How many grid cells are filled."""
        total_cells = 1
        for bins in self.feature_bins.values():
            total_cells *= (len(bins) - 1)
        return len(self.grid), total_cells, len(self.grid) / max(total_cells, 1)
