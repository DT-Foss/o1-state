"""
Layer 1: PS-Lifted Markov Engine
=================================
The substrate. Everything runs on this.

Fixed transition matrix with Fiedler-oriented Z₂ lifting.
NOT learned — structurally determined by graph topology.

Proven results:
- γ_lift ~ m^{-0.381} on barbells (T385)
- 26× consensus speedup (T391)
- 20,085× metastability breaking (T201)
- γ = 2·p_r = 0.394 universal ceiling
"""

import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix


class PSLifted:
    """
    PS-Lifted Markov chain with Z₂ symmetry.

    The graph is doubled: each node i exists in Layer 0 and Layer 1.
    Layer 0 prefers forward (Fiedler-ascending) transitions.
    Layer 1 prefers backward (Fiedler-descending) transitions.
    Cross-layer transitions carry probability p_r.

    Parameters:
        p_c: forward probability (default 0.80)
        p_s: self-loop probability (default 0.003)
        p_r: cross-layer probability (1 - p_c - p_s = 0.197)
    """

    def __init__(self, adjacency, p_c=0.80, p_s=0.003):
        self.A = np.array(adjacency, dtype=float)
        self.n = self.A.shape[0]
        self.p_c = p_c
        self.p_s = p_s
        self.p_r = 1.0 - p_c - p_s

        self._build()

    def _build(self):
        """Build base and lifted transition matrices."""
        n = self.n
        A = self.A
        degrees = A.sum(axis=1)

        # Base transition matrix (standard random walk)
        self.T_base = np.zeros((n, n))
        for i in range(n):
            if degrees[i] > 0:
                self.T_base[i] = A[i] / degrees[i]
            else:
                self.T_base[i, i] = 1.0

        # Fiedler vector (second smallest eigenvector of Laplacian)
        L = np.diag(degrees) - A
        try:
            L_sparse = csr_matrix(L)
            _, vecs = eigsh(L_sparse, k=min(3, n - 1), which='SM')
            self.fiedler = vecs[:, 1]
        except Exception:
            self.fiedler = np.random.randn(n)

        # Orient edges by Fiedler value
        self.forward = {i: [] for i in range(n)}
        self.backward = {i: [] for i in range(n)}

        for i in range(n):
            for j in range(n):
                if A[i, j] > 0:
                    if self.fiedler[j] > self.fiedler[i]:
                        self.forward[i].append(j)
                    else:
                        self.backward[i].append(j)

        # Build lifted transition matrix (2n × 2n)
        p_c, p_r, p_s = self.p_c, self.p_r, self.p_s
        self.T_lift = np.zeros((2 * n, 2 * n))

        for i in range(n):
            nf = len(self.forward[i])
            nb = len(self.backward[i])

            # Layer 0: forward stays, backward crosses
            rs = 0.0
            if nf > 0:
                w = p_c / nf
                for j in self.forward[i]:
                    self.T_lift[i, j] = w
                    rs += w
            if nb > 0:
                w = p_r / nb
                for j in self.backward[i]:
                    self.T_lift[i, n + j] = w
                    rs += w
            self.T_lift[i, i] = max(p_s, 1.0 - rs)

            # Layer 1: backward stays, forward crosses
            rs = 0.0
            if nb > 0:
                w = p_c / nb
                for j in self.backward[i]:
                    self.T_lift[n + i, n + j] = w
                    rs += w
            if nf > 0:
                w = p_r / nf
                for j in self.forward[i]:
                    self.T_lift[n + i, j] = w
                    rs += w
            self.T_lift[n + i, n + i] = max(p_s, 1.0 - rs)

        # Spectral gaps
        self.gamma_base = self._spectral_gap(self.T_base)
        self.gamma_lift = self._spectral_gap(self.T_lift)

    def _spectral_gap(self, T):
        """Compute spectral gap = 1 - |λ₂|."""
        try:
            eigs = np.sort(np.abs(np.linalg.eigvals(T)))[::-1]
            return 1.0 - eigs[1] if len(eigs) > 1 else 0.0
        except Exception:
            return 0.0

    @property
    def ratio(self):
        """γ_lift / γ_base — the Foss amplification factor."""
        if self.gamma_base < 1e-10:
            return float('inf')
        return self.gamma_lift / self.gamma_base

    def propagate(self, state, steps=1, use_lifted=True):
        """
        Propagate a state vector through the chain.

        Args:
            state: (n,) vector for base, (2n,) for lifted
            steps: number of propagation steps
            use_lifted: if True, use T_lift; else T_base

        Returns:
            Propagated state, projected to n-dimensional if lifted.
        """
        T = self.T_lift if use_lifted else self.T_base
        n = self.n

        if use_lifted and len(state) == n:
            # Embed into lifted space: duplicate in both layers
            s = np.zeros(2 * n)
            s[:n] = state
            s[n:] = state
        else:
            s = state.copy()

        for _ in range(steps):
            s = s @ T

        # Project back to physical space
        if use_lifted and len(s) == 2 * n:
            return (s[:n] + s[n:]) / 2
        return s

    def consensus(self, values, tol=1e-6, max_steps=10000, use_lifted=True):
        """
        Run gossip consensus until convergence.

        Returns: (converged_values, steps)
        """
        T = self.T_lift if use_lifted else self.T_base
        n = self.n

        if use_lifted:
            v = np.zeros(2 * n)
            v[:n] = values
            v[n:] = values
        else:
            v = values.copy()

        for step in range(1, max_steps + 1):
            v_new = T @ v

            if use_lifted:
                proj_new = (v_new[:n] + v_new[n:]) / 2
                proj_old = (v[:n] + v[n:]) / 2
            else:
                proj_new = v_new[:n]
                proj_old = v[:n]

            if np.max(np.abs(proj_new - proj_old)) < tol:
                return proj_new, step

            v = v_new

        if use_lifted:
            return (v[:n] + v[n:]) / 2, max_steps
        return v[:n], max_steps

    def z2_parity(self, state_2n):
        """
        Extract Z₂ parity signal from lifted state.

        Returns the DIFFERENCE between Layer 0 and Layer 1.
        This is the "spike" — nonzero when layers disagree.
        """
        n = self.n
        return state_2n[:n] - state_2n[n:]

    def detect_topology_change(self, A_new):
        """
        Detect if graph topology changed by comparing Foss ratios.

        Returns: (changed, delta_ratio, new_ratio)
        """
        old_ratio = self.ratio
        mc_new = PSLifted(A_new, self.p_c, self.p_s)
        new_ratio = mc_new.ratio
        delta = abs(new_ratio - old_ratio)
        # Relative change > 5% = topology changed
        changed = delta / max(old_ratio, 0.01) > 0.05
        return changed, delta, new_ratio
