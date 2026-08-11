"""
Reservoir Health Diagnostics — Effective Rank + TwoNN (F9, F10)
================================================================
Two diagnostic tools to check if the reservoir is working correctly:

1. Effective Rank (F9): Shannon entropy of singular values
   - Measures true dimensionality of reservoir states
   - Low rank = degenerate reservoir (all states similar)
   - Expected: ~860d effective for 2048d reservoir (MERA rank decay F6)

2. TwoNN Intrinsic Dimension (F10): ratio-based ID estimator
   - Measures manifold dimension of state space
   - ID < 5 = collapsed (reservoir is useless)
   - ID > 100 = noise-dominated (no learned structure)
   - Sweet spot: 20-80 (structured but complex)

Both are cheap O(n²) computations on a sample of states.
Run after training or loading to verify reservoir health.
"""

import numpy as np
from typing import Dict, Tuple


def effective_rank(sigma: np.ndarray) -> float:
    """Effective rank from singular values (F9).

    Shannon entropy of normalized singular values.

    Args:
        sigma: singular values (from np.linalg.svd)

    Returns:
        effective rank (1.0 = rank-1, d = full rank)
    """
    s = sigma / sigma.sum()
    s = s[s > 1e-12]  # Filter zeros
    entropy = -float(np.sum(s * np.log(s)))
    return float(np.exp(entropy))


def twonn_id(X: np.ndarray, n_sample: int = 500) -> float:
    """TwoNN intrinsic dimension estimator (F10).

    Based on ratio of nearest-neighbor distances.
    Facco et al. (2017): ID = 1 / mean(log(r2/r1))

    Args:
        X: (n_points, dim) data matrix (e.g., reservoir states)
        n_sample: subsample size for speed

    Returns:
        estimated intrinsic dimension
    """
    n = X.shape[0]
    if n < 10:
        return 0.0

    # Subsample if needed
    if n > n_sample:
        idx = np.random.default_rng(42).choice(n, n_sample, replace=False)
        X = X[idx]

    # Pairwise distances
    from scipy.spatial.distance import cdist
    dist = cdist(X, X)
    np.fill_diagonal(dist, np.inf)

    # First and second nearest neighbor distances
    sorted_dist = np.sort(dist, axis=1)
    r1 = sorted_dist[:, 0]
    r2 = sorted_dist[:, 1]

    # Filter degenerate points (r1 = 0)
    valid = r1 > 1e-12
    if valid.sum() < 5:
        return 0.0

    r1 = r1[valid]
    r2 = r2[valid]

    mu = r2 / r1
    return float(1.0 / np.mean(np.log(mu)))


def reservoir_health(reservoir) -> Dict[str, float]:
    """Full health check on a reservoir.

    Args:
        reservoir: ReservoirLM instance with W_out

    Returns:
        dict with diagnostic metrics
    """
    result = {}

    # Check W_out effective rank
    if reservoir.W_out is not None:
        _, sigma, _ = np.linalg.svd(reservoir.W_out, full_matrices=False)
        result['readout_effective_rank'] = effective_rank(sigma)
        result['readout_condition_number'] = float(sigma[0] / max(sigma[-1], 1e-12))

    # Check W_in effective rank
    if hasattr(reservoir, 'W') and reservoir.W is not None:
        _, sigma_w, _ = np.linalg.svd(reservoir.W, full_matrices=False)
        result['reservoir_effective_rank'] = effective_rank(sigma_w)

    # MERA prediction (F6): expected rank after 5 mixing steps
    if hasattr(reservoir, 'reservoir_size'):
        n = reservoir.reservoir_size
        mera_predicted = 2 * n * np.exp(-0.31 * 5)  # After 5 steps
        result['mera_predicted_rank'] = float(mera_predicted)

    return result


def diagnose_states(states: np.ndarray) -> Dict[str, float]:
    """Diagnose a collection of reservoir states.

    Args:
        states: (n_states, dim) matrix of reservoir states

    Returns:
        dict with effective_rank, intrinsic_dimension, variance
    """
    if states.shape[0] < 5:
        return {'error': 'too few states'}

    _, sigma, _ = np.linalg.svd(states, full_matrices=False)

    result = {
        'effective_rank': effective_rank(sigma),
        'top_sv_ratio': float(sigma[0] / max(sigma.sum(), 1e-12)),
        'state_variance': float(np.var(states)),
    }

    try:
        result['intrinsic_dimension'] = twonn_id(states)
    except Exception:
        result['intrinsic_dimension'] = -1.0

    return result
