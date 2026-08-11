"""Optional numerical diagnostics with explicit validity boundaries."""

from __future__ import annotations

import numpy as np


def intrinsic_dimension(X: np.ndarray, k: int = 2) -> float:
    """TwoNN-Schaetzer (Levina-Bickel-Variante, gottformel_formulas.md):
    ID ~ (1/|S|) * Sum log(r2_i / r1_i) fuer die zwei naechsten Nachbarn.
    Saturiert die ID der Relations-Nutzung, ist das Primitiv-Schema
    vollstaendig; waechst sie, fehlen Primitive.

    LIVE-ONLY: kein Lab-Gegenpart. Anders als two_nn_intrinsic_dimension
    (die strengere MLE-Variante unten, die bei degenerierter/near-regular
    Geometrie einen ValueError wirft) faellt diese Variante bei zu wenig
    Punkten oder Nullabstaenden still auf 1.0 zurueck statt zu werfen.
    Erhalten per Lead-Entscheidung, da beim primitives/relations-Umbau
    kein Konsument diese Funktion ausserhalb ihres eigenen Tests aufruft
    (Test bleibt unveraendert unter tests/test_primitives.py)."""
    n = len(X)
    if n < 3:
        return 1.0
    from scipy.spatial.distance import pdist, squareform  # optional
    try:
        D = squareform(pdist(X, metric="euclidean"))
    except ImportError:  # pragma: no cover
        D = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.sqrt(np.sum((X[i] - X[j]) ** 2)))
                D[i, j] = D[j, i] = d
    np.fill_diagonal(D, np.inf)
    ratios = []
    for i in range(n):
        nn = np.sort(D[i])[:k]
        if nn[0] > 0 and nn[-1] > 0:
            ratios.append(np.log(nn[-1] / nn[0]))
    if not ratios:
        return 1.0
    return float(len(ratios) / (np.sum(ratios) + 1e-12))


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
