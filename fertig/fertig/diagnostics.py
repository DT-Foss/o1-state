"""Optional numerical diagnostics with explicit validity boundaries."""

from __future__ import annotations

import numpy as np


def two_nn_intrinsic_dimension(samples: np.ndarray) -> float:
    """Estimate intrinsic dimension with the TwoNN maximum likelihood rule.

    The estimator is meaningful for sampled points in a continuous metric
    space.  It must not be applied to one-hot relation labels as a proof that a
    semantic schema is complete.  Duplicate points and near-regular lattices
    are rejected because nearest-neighbour ties make the estimate singular.
    """

    X = np.asarray(samples, dtype=float)
    if X.ndim != 2 or len(X) < 4:
        raise ValueError("TwoNN requires at least four 2-D sample rows")
    if not np.all(np.isfinite(X)):
        raise ValueError("TwoNN samples must be finite")
    delta = X[:, None, :] - X[None, :, :]
    distances = np.sqrt(np.sum(delta * delta, axis=2))
    np.fill_diagonal(distances, np.inf)
    nearest = np.partition(distances, 1, axis=1)[:, :2]
    r1 = nearest[:, 0]
    r2 = nearest[:, 1]
    valid = np.isfinite(r2) & (r1 > 0.0) & (r2 > r1)
    if np.count_nonzero(valid) < max(4, len(X) // 2):
        raise ValueError("TwoNN geometry is degenerate (duplicates or ties)")
    logs = np.log(r2[valid] / r1[valid])
    if float(np.median(logs)) < 1e-3:
        raise ValueError("TwoNN geometry is near-regular and ill-conditioned")
    estimate = float(len(logs) / np.sum(logs))
    if not np.isfinite(estimate):
        raise ValueError("TwoNN estimate is not finite")
    return estimate
