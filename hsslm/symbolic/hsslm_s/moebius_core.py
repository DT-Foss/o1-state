"""Pure Moebius-Lorentz core — zero neural components.

Implements the foundational Moebius coupling operations (F4–F8),
Moebius-SSM state transition dynamics (F17–F24), and
Sinkhorn renormalization (F24) for doubly-stochastic projection.

All formulas reference the HSSLM-S mathematical specification.
"""

import numpy as np
from typing import Tuple, Optional

# ---------------------------------------------------------------------------
# Module-level constants (no magic numbers)
# ---------------------------------------------------------------------------
C_CURVATURE: float = 1.0          # F3: hyperboloid curvature (normalised)
D_SPATIAL: int = 16               # F2: state-space spatial dimension
TAU_MAX: float = 0.95             # F35: maximum contraction before phase transition
LAMBDA_MIN: float = -0.99         # clamp: minimum coupling (anti-aligned)
LAMBDA_MAX: float = 0.99          # clamp: maximum coupling (fully aligned)
ETA_HAMILTON: float = 0.01        # F23: Hamiltonian gradient step size
MU_PENALTY: float = 0.1           # F22, F59: Gershgorin penalty coupling
GAMMA_CHAIN: float = 0.85         # F29, F52: confidence chain decay constant
BETA_INV_TEMP: float = 3.0        # F37, F38: cubic repulsion inverse temperature
T0_TEMP: float = 1.0              # F36: initial sampling temperature
SINKHORN_MAX_ITER: int = 100      # F24: maximum Sinkhorn iterations
SINKHORN_TOL: float = 1e-8        # F24: Sinkhorn convergence tolerance
EPS_PERTURB: float = 0.001        # F62: uniform coupling perturbation strength
LAMBDA2_MIN: float = 0.05         # F42, F62: minimum spectral gap
LAMBDA2_TARGET: float = 0.10      # F61: target spectral gap
D_MAX: float = np.arccosh(17.0)   # F26: maximal hyperbolic distance on H^16


# ===========================================================================
# F4 – F8: Moebius Coupling Primitives
# ===========================================================================

def moebius_couple(lam: float, v: float) -> float:
    """F4: Moebius coupling function.

    f(lambda, v) = (lambda + v) / (1 + lambda * v)

    This is the hyperbolic tangent addition formula:
    if lambda = tanh(alpha) and v = tanh(beta),
    then f(lambda, v) = tanh(alpha + beta).
    """
    denom = 1.0 + lam * v
    if abs(denom) < 1e-12:
        return np.sign(lam + v) * 0.999999
    return (lam + v) / denom


def period_function(lam: float) -> float:
    """F5: Period function (Lorentz-Poincare factor).

    g(lambda) = (1 - lambda^2)^(-1/2) = cosh(arctanh(lambda))

    Acts as a local scaling factor that diverges as |lambda| -> 1,
    representing critical coupling.
    """
    lam_clamped = np.clip(lam, -0.999999, 0.999999)
    return 1.0 / np.sqrt(1.0 - lam_clamped * lam_clamped)


def lorentz_factor(lam: float, v: float) -> float:
    """F6: Lorentz factor for state transitions.

    gamma(lambda, v) = (1 + lambda * v) / sqrt(1 - v^2)

    Governs the relativistic scaling of state updates.
    Velocity v is dimensionless, bounded as |v| < 1.
    """
    v_clamped = np.clip(v, -0.999999, 0.999999)
    return (1.0 + lam * v) / np.sqrt(1.0 - v_clamped * v_clamped)


def velocity_add(v1: float, v2: float) -> float:
    """F7: Velocity addition as state transition (Einstein velocity addition).

    v1 (+) v2 = (v1 + v2) / (1 + v1 * v2)

    For n sequential transitions:
    (+)_{k=1}^{n} v_k = tanh(sum_{k=1}^{n} arctanh(v_k))
    """
    denom = 1.0 + v1 * v2
    if abs(denom) < 1e-12:
        return np.sign(v1 + v2) * 0.999999
    return (v1 + v2) / denom


def velocity_add_n(velocities: np.ndarray) -> float:
    """F7 (vectorised): n-fold velocity addition.

    (+)_{k=1}^{n} v_k = tanh(sum_{k=1}^{n} arctanh(v_k))
    """
    v_safe = np.clip(velocities, -0.999999, 0.999999)
    return float(np.tanh(np.sum(np.arctanh(v_safe))))


def velocity_extract(
    x_t: np.ndarray, x_tp1: np.ndarray
) -> float:
    """F8: State velocity extraction.

    v^{(t)} = ||x^{sp,(t+1)} - x^{sp,(t)}||_2 / (x^{0,(t+1)} - x^{0,(t)})

    with constraint |v^{(t)}| < 1 enforced by the light-cone condition.
    """
    dx_sp = x_tp1[1:] - x_t[1:]
    dx_0 = x_tp1[0] - x_t[0]
    if abs(dx_0) < 1e-12:
        return 0.0
    v = float(np.linalg.norm(dx_sp) / dx_0)
    return float(np.clip(v, -0.999999, 0.999999))


# ===========================================================================
# Vectorised versions (NumPy arrays)
# ===========================================================================

def moebius_couple_vec(lam: np.ndarray, v: np.ndarray) -> np.ndarray:
    """F4 (vectorised): element-wise Moebius coupling.

    f(lambda, v) = (lambda + v) / (1 + lambda * v)
    """
    denom = 1.0 + lam * v
    denom = np.where(np.abs(denom) < 1e-12, np.sign(lam + v) * 1e12, denom)
    return (lam + v) / denom


def period_function_vec(lam: np.ndarray) -> np.ndarray:
    """F5 (vectorised): element-wise period function.

    g(lambda) = (1 - lambda^2)^(-1/2)
    """
    lam_clamped = np.clip(lam, -0.999999, 0.999999)
    return 1.0 / np.sqrt(1.0 - lam_clamped * lam_clamped)


def lorentz_factor_vec(lam: np.ndarray, v: np.ndarray) -> np.ndarray:
    """F6 (vectorised): element-wise Lorentz factor.

    gamma(lambda, v) = (1 + lambda * v) / sqrt(1 - v^2)
    """
    v_clamped = np.clip(v, -0.999999, 0.999999)
    return (1.0 + lam * v) / np.sqrt(1.0 - v_clamped * v_clamped)


# ===========================================================================
# F12: Hyperbolic distance on H^d
# ===========================================================================

def minkowski_inner(x: np.ndarray, y: np.ndarray) -> float:
    """Minkowski inner product: <x, y>_M = -x^0*y^0 + sum_{k=1}^d x^k*y^k."""
    return float(-x[0] * y[0] + np.dot(x[1:], y[1:]))


def hyperbolic_distance(x: np.ndarray, y: np.ndarray) -> float:
    """F12: Hyperbolic distance on H^d.

    d_H(x_i, x_j) = arccosh(-<x_i, x_j>_M)
                 = arccosh(x_i^0 * x_j^0 - x_i^{sp} . x_j^{sp})

    For normalised states on H^d with c=1.
    """
    inner = -minkowski_inner(x, y)
    inner_clamped = max(1.0, inner)
    return float(np.arccosh(inner_clamped))


def normalize_to_hyperboloid(x: np.ndarray, c: float = C_CURVATURE) -> np.ndarray:
    """F11: Project state onto the hyperboloid H^d.

    x -> c * x / sqrt(-<x, x>_M)

    Enforces <x, x>_M = -c^2 for all valid states.
    """
    mink_sq = -(-x[0] * x[0] + np.dot(x[1:], x[1:]))
    if mink_sq <= 0:
        # Degenerate case: perturb slightly
        x = x.copy()
        x[0] = np.sqrt(1.0 + np.dot(x[1:], x[1:])) + 0.01
        mink_sq = -(-x[0] * x[0] + np.dot(x[1:], x[1:]))
    norm = np.sqrt(max(mink_sq, 1e-12))
    return c * x / norm


# ===========================================================================
# F15: Fidelity tensor
# ===========================================================================

def fidelity_tensor(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """F15: Fidelity between two symbol states.

    F_{ij} = |<x_i, x_j>_M|^(-1)

    Satisfies 0 < F_{ij} <= 1 with F_{ii} = 1.
    """
    inner = abs(minkowski_inner(x_i, x_j))
    if inner < 1e-12:
        return 1.0
    return 1.0 / inner


# ===========================================================================
# F17–F20: Moebius-SSM State Transition
# ===========================================================================

def moebius_state_transition(
    state: np.ndarray,
    x_input: np.ndarray,
    A: np.ndarray,
    dt: np.ndarray,
    v_coupling: np.ndarray,
    tau_max: float = TAU_MAX,
) -> np.ndarray:
    """Core Moebius state update (F17–F20).

    Parameters
    ----------
    state : np.ndarray, shape (d_state, d_inner)
        Current state matrix.
    x_input : np.ndarray, shape (d_inner,)
        Input vector.
    A : np.ndarray, shape (d_inner, d_state)
        State transition matrix.
    dt : np.ndarray, shape (d_inner,)
        Time-step discretisation.
    v_coupling : np.ndarray, shape (d_state, d_inner)
        Coupling velocity matrix.
    tau_max : float
        Maximum allowed coupling magnitude (default 0.95).

    Algorithm
    ---------
    1. lambda = exp(A * dt)               [discretised state matrix]
    2. lambda_clamped = clamp(lambda, -tau_max, tau_max)
    3. lambda_new = f(lambda_clamped, v_coupling)  [Moebius coupling, F4]
    4. gate = g(lambda_new)               [period gating, F5]
    5. new_state = gate * lambda_new * state + 0.01 * x_input * (1 - |lambda_new|)
    """
    # Step 1: discretised state matrix  lambda = exp(A @ dt)
    lam = np.exp(A @ dt)  # shape (d_state,)

    # Step 2: clamp to safe range
    lam_clamped = np.clip(lam, -tau_max, tau_max)  # shape (d_state,)

    # Step 3: Moebius coupling with velocity matrix
    # lam_clamped has shape (d_state,), v_coupling has shape (d_state, d_inner)
    # We couple element-wise across the state dimensions
    lam_new = moebius_couple_vec(
        lam_clamped[:, np.newaxis], v_coupling
    )  # shape (d_state, d_inner)

    # Step 4: period gating  g(lambda) = (1 - lambda^2)^(-1/2)
    gate = period_function_vec(lam_new)  # shape (d_state, d_inner)

    # Step 5: state update with input injection
    abs_lam = np.abs(lam_new)
    input_term = 0.01 * x_input[np.newaxis, :] * (1.0 - abs_lam)
    new_state = gate * lam_new * state + input_term

    return new_state


# ===========================================================================
# F18: Coupling strength update
# ===========================================================================

def coupling_update(lam_t: float, v_ctx: float) -> float:
    """F18: Coupling strength update via Moebius composition.

    lambda^{(t+1)} = f(lambda^{(t)}, v_ctx)
                   = (lambda^{(t)} + v_ctx) / (1 + lambda^{(t)} * v_ctx)

    Clamped to (-0.99, 0.99) for numerical stability.
    """
    lam_new = moebius_couple(lam_t, v_ctx)
    return float(np.clip(lam_new, LAMBDA_MIN, LAMBDA_MAX))


def contextual_velocity(neighbor_velocities: np.ndarray) -> float:
    """F18: Contextual velocity as rapidity-space average.

    v_ctx = tanh( (1/|N(i)|) * sum_{j in N(i)} arctanh(v_{ij}) )

    Returns 0.0 if neighbor_velocities is empty.
    """
    if neighbor_velocities.size == 0:
        return 0.0
    v_safe = np.clip(neighbor_velocities, -0.999999, 0.999999)
    avg_rapidity = np.mean(np.arctanh(v_safe))
    return float(np.tanh(avg_rapidity))


# ===========================================================================
# F19: Pairwise transition velocity
# ===========================================================================

def pairwise_velocity(x_i: np.ndarray, e_j: np.ndarray) -> float:
    """F19: Pairwise transition velocity between symbols.

    v_{ij}^{(t)} = <x_i^{(t)}, e_j>_M / (x_i^{0,(t)} * e_j^0)
                 = (-x_i^0 + x_i^{sp} . b_j) / x_i^0

    Measures alignment of current state with primitive signature of s_j.
    """
    x0_i = x_i[0]
    e0_j = e_j[0]
    denom = x0_i * e0_j
    if abs(denom) < 1e-12:
        return 0.0
    return float(minkowski_inner(x_i, e_j) / denom)


# ===========================================================================
# F20: Neighborhood filter (symmetrised minimum)
# ===========================================================================

def neighborhood_filter(W: np.ndarray) -> np.ndarray:
    """F20: Neighborhood filter — symmetrised minimum.

    tilde{W}_{ij} = min(W_{ij}, W_{ji})

    Eliminates one-directional noise by enforcing symmetric coupling.
    """
    return np.minimum(W, W.T)


# ===========================================================================
# F21: Gershgorin alignment penalty
# ===========================================================================

def gershgorin_penalty(W: np.ndarray) -> float:
    """F21: Gershgorin alignment penalty.

    For each row i:
        R_i = sum_{j != i} |tilde{W}_{ij}|
        P_i = max(0, R_i - tilde{W}_{ii})^2

    Total penalty: P_total = sum_i P_i
    """
    n = W.shape[0]
    total = 0.0
    for i in range(n):
        row_sum = np.sum(np.abs(W[i, :])) - abs(W[i, i])
        penalty_i = max(0.0, row_sum - W[i, i]) ** 2
        total += penalty_i
    return float(total)


# ===========================================================================
# F24: Sinkhorn renormalization
# ===========================================================================

def sinkhorn_renormalize(
    W: np.ndarray,
    n_iter: int = SINKHORN_MAX_ITER,
    tol: float = SINKHORN_TOL,
) -> np.ndarray:
    """F24: Sinkhorn renormalization — project to doubly-stochastic manifold.

    W^{DS} = lim_{k -> inf} S_k,
    where S_{k+1} = C_row(C_col(S_k))

    C_row(S)_{ij} = S_{ij} / sum_k S_{ik}
    C_col(S)_{ij} = S_{ij} / sum_k S_{kj}

    Iterate for K = 100 steps or until max error < 1e-8.
    """
    S = W.copy()
    # Ensure non-negative
    S = np.maximum(S, 0.0)

    for _ in range(n_iter):
        # Row normalisation
        row_sums = S.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)
        S = S / row_sums

        # Column normalisation
        col_sums = S.sum(axis=0, keepdims=True)
        col_sums = np.where(col_sums < 1e-12, 1.0, col_sums)
        S = S / col_sums

        # Check convergence
        row_errors = np.abs(S.sum(axis=1) - 1.0)
        col_errors = np.abs(S.sum(axis=0) - 1.0)
        if np.max(row_errors) < tol and np.max(col_errors) < tol:
            break

    return S


# ===========================================================================
# F22–F23: Hamiltonian dynamics
# ===========================================================================

def hamiltonian_step(
    states: np.ndarray,
    refs: np.ndarray,
    mu: float = MU_PENALTY,
    eta: float = ETA_HAMILTON,
) -> np.ndarray:
    """F22–F23: Single Hamiltonian gradient-flow step.

    dH/dx_i = x_i - x_{ref,i} / d_H(x_i, x_{ref,i}) + mu * nabla P_i

    x_i^{(t+1)} = x_i^{(t)} - eta * dH/dx_i

    Followed by renormalisation to hyperboloid (F11).
    """
    new_states = states.copy()
    for i in range(states.shape[0]):
        x_i = states[i]
        ref_i = refs[i]

        # Gradient of hyperbolic potential
        d_h = hyperbolic_distance(x_i, ref_i)
        if d_h < 1e-12:
            d_h = 1e-12
        dH = x_i - ref_i / d_h

        # Gradient descent step
        x_new = x_i - eta * dH

        # Renormalise to hyperboloid
        new_states[i] = normalize_to_hyperboloid(x_new)

    return new_states


# ===========================================================================
# F25: Binary composition (hyperbolic addition)
# ===========================================================================

def binary_composition(x_i: np.ndarray, x_j: np.ndarray) -> np.ndarray:
    """F25: Binary composition of two symbol states.

    x_{i o j} = (1 / sqrt(1 - lambda_{ij}^2)) *
                [x_i^0 * x_j^0 - x_i^{sp} . x_j^{sp};
                 x_i^0 * x_j^{sp} + x_j^0 * x_i^{sp}]

    Hyperbolic addition — composition = adding rapidities on H^d.
    """
    d_h = hyperbolic_distance(x_i, x_j)
    lam_ij = np.tanh(d_h / D_MAX)
    scale = 1.0 / np.sqrt(max(1.0 - lam_ij ** 2, 1e-12))

    x0_composed = x_i[0] * x_j[0] - np.dot(x_i[1:], x_j[1:])
    xsp_composed = x_i[0] * x_j[1:] + x_j[0] * x_i[1:]

    x_comp = np.concatenate([[x0_composed], xsp_composed])
    return scale * x_comp


# ===========================================================================
# F26: Composition coupling coefficient
# ===========================================================================

def composition_coupling(x_i: np.ndarray, x_j: np.ndarray) -> float:
    """F26: Composition coupling coefficient.

    lambda_{ij} = tanh( d_H(x_i, x_j) / d_max )

    where d_max = arccosh(d + 1) is the maximal distance on H^d.
    """
    d_h = hyperbolic_distance(x_i, x_j)
    return float(np.tanh(d_h / D_MAX))


# ===========================================================================
# F41–F42: Spectral gap / Cheeger
# ===========================================================================

def compute_spectral_gap(W_ds: np.ndarray) -> float:
    """Compute the spectral gap lambda_2 of the random-walk Laplacian.

    L = D - W^{DS}, then lambda_2 = second-smallest eigenvalue of L.
    """
    D = np.diag(W_ds.sum(axis=1))
    L = D - W_ds
    eigs = np.linalg.eigvalsh(L)
    eigs_sorted = np.sort(eigs)
    if len(eigs_sorted) >= 2:
        return float(eigs_sorted[1])
    return 0.0


def fiedler_orientation(L: np.ndarray) -> Tuple[np.ndarray, float]:
    """F39: Fiedler eigenvector orientation.

    L @ v_2 = lambda_2 @ v_2

    Returns the Fiedler vector v_2 and eigenvalue lambda_2.
    """
    eigs, vecs = np.linalg.eigh(L)
    idx = np.argsort(eigs)
    eigs = eigs[idx]
    vecs = vecs[:, idx]
    if len(eigs) >= 2:
        return vecs[:, 1], float(eigs[1])
    return vecs[:, 0], float(eigs[0])


def spectral_gap_maintenance(L: np.ndarray) -> np.ndarray:
    """F62: Spectral gap maintenance — ensure lambda_2 >= 0.05.

    If violated, apply perturbation:
        W^{(t)} <- W^{(t)} + epsilon * J,  epsilon = 0.001
    then re-run Sinkhorn.
    """
    v2, lam2 = fiedler_orientation(L)
    if lam2 >= LAMBDA2_MIN:
        return L

    n = L.shape[0]
    # Recover W from L = D - W
    W = np.diag(np.diag(L)) - L
    # Apply uniform coupling perturbation
    J = np.ones((n, n)) / n
    W_new = W + EPS_PERTURB * J
    # Ensure non-negative
    W_new = np.maximum(W_new, 0.0)
    # Rebalance
    W_ds = sinkhorn_renormalize(W_new)
    # Rebuild Laplacian
    D_new = np.diag(W_ds.sum(axis=1))
    L_new = D_new - W_ds

    return L_new


# ===========================================================================
# F27: Path confidence for transitive inference
# ===========================================================================

def path_confidence(
    path: Tuple[int, ...],
    fidelity_matrix: np.ndarray,
    gamma: float = GAMMA_CHAIN,
) -> float:
    """F27: Confidence for an inference path.

    conf(pi) = prod_{k=1}^{L-1} F_{i_k, i_{k+1}} * gamma^{L-2}

    where F_{ij} is the fidelity tensor (F15).
    """
    if len(path) < 2:
        return 1.0
    conf = 1.0
    for k in range(len(path) - 1):
        i, j = path[k], path[k + 1]
        if i < fidelity_matrix.shape[0] and j < fidelity_matrix.shape[1]:
            conf *= fidelity_matrix[i, j]
    conf *= gamma ** (len(path) - 2)
    return float(conf)


# ===========================================================================
# F16: Berry phase for cyclic sequences
# ===========================================================================

def berry_phase(states: np.ndarray) -> float:
    """F16: Berry phase for cyclic symbol sequences.

    Phi_B(sigma) = arg( prod_{k=1}^{L} <x_{i_k}, x_{i_{k+1}}>_M ),  i_{L+1} = i_1

    Topological phase invariant under smooth deformations of state path.
    """
    n = states.shape[0]
    if n < 2:
        return 0.0
    product = 1.0 + 0.0j
    for k in range(n):
        kp1 = (k + 1) % n
        inner = minkowski_inner(states[k], states[kp1])
        product *= complex(inner, 0.0)
    return float(np.angle(product))
