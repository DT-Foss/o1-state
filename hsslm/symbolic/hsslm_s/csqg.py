"""Cheeger Spectral Quality Gate (CSQG).

Spectral quality metric for generated tokens -- gates output.
No learned parameters. Purely deterministic.

Formulas (from "Linear Cheeger Improvement"):
    Gershgorin disc for row i: D_i = {z : |z - a_ii| <= R_i}
                                 where R_i = sum_{j!=i} |a_ij|

    Cheeger conductance: h(S) = |boundary(S)| / min(vol(S), vol(V\\S))

    Spectral gap: lambda_2 >= h^2 / 2   (Cheeger inequality)

    Quality score: Q = lambda_2 / lambda_max  [normalized spectral gap, in [0, 1]]

    Gershgorin alignment: G_align = 1 - max_i(R_i) / (min_i|P_ii| + eps)

    Cheeger quality gate:
        Q_Cheeger = G_align * sqrt(lambda_2 / 8)
"""

import numpy as np
from typing import Tuple, Set, Optional


EPS: float = 1e-8
TAU_GATE_RESOLVED: float = 0.15  # Derived from Ginibre constant K_G = 1.08747
KAPPA: float = 2.0               # Contraction acceleration factor
LAMBDA_DAMP: float = 0.5         # Expansion damping factor


def gershgorin_discs(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Gershgorin discs for each row.

    D_i = { z in C : |z - a_ii| <= R_i }
    where center[i] = a_ii, radius[i] = sum_{j!=i} |a_ij|.

    Args:
        matrix: Square matrix (n x n).

    Returns:
        (centers, radii) as 1-D arrays of length n.
            centers[i] = matrix[i, i]
            radii[i]   = sum_{j!=i} |matrix[i, j]|
    """
    n = matrix.shape[0]
    if n < 1:
        return np.array([]), np.array([])

    centers = np.abs(np.diag(matrix))
    radii = np.zeros(n, dtype=matrix.dtype)

    for i in range(n):
        row = np.abs(matrix[i, :])
        row[i] = 0.0  # exclude diagonal
        radii[i] = np.sum(row)

    return centers, radii


def cheeger_conductance(subset: Set[int], adjacency: np.ndarray) -> float:
    """h(S) = |boundary(S)| / min(vol(S), vol(V\S))

    Compute the Cheeger conductance of a subset S of vertices.
    The boundary consists of edges with one endpoint in S and one
    in V \ S.  vol(S) is the sum of all edge weights incident to S.

    Args:
        subset:  Set of vertex indices (subset of {0, ..., n-1}).
        adjacency:  Symmetric weight matrix W (n x n, non-negative).

    Returns:
        Conductance h(S) in [0, 1].  Returns 1.0 for edge cases.
    """
    n = adjacency.shape[0]
    if n < 2 or len(subset) == 0 or len(subset) == n:
        return 1.0

    all_vertices = set(range(n))
    complement = all_vertices - subset

    # Boundary cut: sum of weights crossing from subset to complement
    boundary_cut = 0.0
    for i in subset:
        for j in complement:
            boundary_cut += adjacency[i, j]

    # Volume of subset
    vol_S = 0.0
    for i in subset:
        vol_S += np.sum(adjacency[i, :])

    # Volume of complement
    vol_comp = 0.0
    for i in complement:
        vol_comp += np.sum(adjacency[i, :])

    min_vol = min(vol_S, vol_comp)
    if min_vol < EPS:
        return 1.0

    return float(boundary_cut / min_vol)


def spectral_gap(laplacian: np.ndarray) -> float:
    """Compute quality score Q = lambda_2 / lambda_max.

    lambda_2  = second smallest eigenvalue of the normalized Laplacian.
    lambda_max = largest eigenvalue.

    The normalized Laplacian is assumed to be symmetric positive
    semi-definite with eigenvalues in [0, 2].  Q lies in [0, 1].

    Args:
        laplacian: Normalized Laplacian matrix L_norm (n x n).

    Returns:
        Q = lambda_2 / lambda_max (float).
    """
    n = laplacian.shape[0]
    if n < 2:
        return 1.0

    # Use NumPy's real symmetric eigensolver (deterministic)
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues_sorted = np.sort(eigenvalues)

    lambda_1 = eigenvalues_sorted[0]      # smallest (should be ~0)
    lambda_2 = eigenvalues_sorted[1] if n > 1 else lambda_1
    lambda_max = eigenvalues_sorted[-1]    # largest

    if lambda_max < EPS:
        return 1.0

    # Clamp lambda_2 to [0, lambda_max]
    lambda_2 = max(0.0, min(lambda_2, lambda_max))

    return float(lambda_2 / lambda_max)


def compute_quality_gate(token_states: np.ndarray) -> float:
    """Compute spectral quality of current token state matrix.

    Steps:
        1. Build similarity graph: W_ij = |<phi_i | phi_j>|^2.
        2. Symmetrize via neighborhood filter: W_sym = min(W, W^T).
        3. Compute normalized Laplacian: L_norm = I - D^{-1/2} W_sym D^{-1/2}.
        4. Compute Gershgorin alignment score.
        5. Compute spectral gap Q = lambda_2 / lambda_max.
        6. Combine: Q_Cheeger = G_align * sqrt(lambda_2 / 8).

    High Q_Cheeger -> coherent output (pass).
    Low Q_Cheeger  -> reject and resample.

    Args:
        token_states: (n_tokens x d_state) array of token state vectors.

    Returns:
        Quality score Q_Cheeger in [0, 1].
    """
    n = token_states.shape[0]
    if n < 2:
        return 1.0  # Single token is perfectly coherent

    # 1. Similarity graph: W_ij = |<psi_i | psi_j>|^2 = |dot(z_i, z_j)|^2
    # Normalize states to unit vectors first
    norms = np.linalg.norm(token_states, axis=1, keepdims=True) + EPS
    states_unit = token_states / norms
    W = np.abs(states_unit @ states_unit.T) ** 2

    # 2. Symmetrize: g(i,j) = min(W(i,j), W(j,i))
    W_sym = np.minimum(W, W.T)

    # 3. Degree matrix and normalized Laplacian
    row_sums = np.sum(W_sym, axis=1)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(row_sums + EPS))
    L_norm = np.eye(n) - D_inv_sqrt @ W_sym @ D_inv_sqrt

    # 4. Gershgorin alignment
    diag = np.abs(np.diag(W_sym))
    radii = np.zeros(n)
    for i in range(n):
        row = np.abs(W_sym[i, :])
        row[i] = 0.0
        radii[i] = np.sum(row)

    max_R = np.max(radii)
    min_diag = np.min(diag) if n > 0 else EPS

    G_align = 1.0 - max_R / (min_diag + EPS)
    G_align = max(0.0, min(1.0, G_align))

    # 5. Spectral gap: Q = lambda_2 / lambda_max
    eigenvalues = np.linalg.eigvalsh(L_norm)
    eigenvalues_sorted = np.sort(eigenvalues)

    lambda_2 = eigenvalues_sorted[1] if n > 1 else eigenvalues_sorted[0]
    lambda_max = eigenvalues_sorted[-1]

    lambda_2 = max(0.0, min(lambda_2, 2.0))  # Clamp to [0, 2]

    # 6. Cheeger quality gate: Q_Cheeger = G_align * sqrt(lambda_2 / 8)
    Q_Cheeger = G_align * np.sqrt(lambda_2 / 8.0)
    Q_Cheeger = max(0.0, min(1.0, Q_Cheeger))

    return float(Q_Cheeger)


def quality_filter_logits(
    logits: np.ndarray,
    threshold: float = 0.5,
) -> bool:
    """Check if logits pass spectral quality gate.

    Uses Cheeger inequality to estimate conductance from the
    normalized Laplacian of the logit state graph.  Returns True
    if the quality score exceeds the threshold.

    Args:
        logits: (n x d) array of logit vectors.
        threshold: Minimum quality score to pass (default 0.5).

    Returns:
        True if quality > threshold (pass), False otherwise.
    """
    Q = compute_quality_gate(logits)
    return Q > threshold


def gate_tau(tau_current: float, Q: float) -> float:
    """Modulate tau based on Cheeger quality gate.

    If Q < tau_gate: state is in a bottleneck -> increase contraction
        tau_new = tau * (1 + kappa * (tau_gate - Q))
    Else: state flows well -> decrease contraction
        tau_new = tau * (1 - lambda * Q)

    Args:
        tau_current: Current tau value.
        Q: Quality score from compute_quality_gate().

    Returns:
        Modulated tau value clamped to [0.3, 0.95].
    """
    if Q < TAU_GATE_RESOLVED:
        tau_new = tau_current * (1.0 + KAPPA * (TAU_GATE_RESOLVED - Q))
    else:
        tau_new = tau_current * (1.0 - LAMBDA_DAMP * Q)

    return float(max(0.3, min(0.95, tau_new)))


class CSQGMonitor:
    """Stateful spectral quality monitor.

    Tracks quality scores over a sliding window and provides
    pass/fail decisions plus trend detection.
    """

    def __init__(self, window_size: int = 10, threshold: float = 0.5):
        self.window_size = window_size
        self.threshold = threshold
        self._quality_buffer: list = []

    def check(self, token_states: np.ndarray) -> Tuple[bool, float]:
        """Return (passes, quality_score).

        Computes the spectral quality of the given token states and
        compares against the configured threshold.

        Args:
            token_states: (n_tokens x d_state) array.

        Returns:
            (passes, Q) where passes is True if Q > threshold.
        """
        Q = compute_quality_gate(token_states)
        self._quality_buffer.append(Q)
        if len(self._quality_buffer) > self.window_size:
            self._quality_buffer.pop(0)
        return Q > self.threshold, Q

    def get_trend(self) -> float:
        """Return quality trend (increasing > 0, decreasing < 0, stable ~0).

        Computes a simple linear slope over the quality window using
        least-squares on indices [0, 1, ..., k-1] vs quality values.
        """
        k = len(self._quality_buffer)
        if k < 2:
            return 0.0

        x = np.arange(k, dtype=np.float64)
        y = np.array(self._quality_buffer, dtype=np.float64)

        # Linear regression slope = Cov(x, y) / Var(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        cov = np.sum((x - x_mean) * (y - y_mean))
        var = np.sum((x - x_mean) ** 2)

        if var < EPS:
            return 0.0

        return float(cov / var)

    def reset(self):
        """Clear the quality history for a new generation sequence."""
        self._quality_buffer.clear()
