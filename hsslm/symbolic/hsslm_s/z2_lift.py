"""Z2 topological lift — physical + momentum decomposition (F28).

Implements PS-Lifted Z2 doubling, quantum potential,
Foss topological index, and self-loop transitions.
"""

import numpy as np
from typing import Tuple

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
P_C: float = 0.65          # F28: PS-lifted forward probability
P_S: float = 0.003         # F28: PS-lifted self-loop probability
P_B: float = 1.0 - P_C - P_S  # PS-lifted backward probability
BETA_INV_TEMP: float = 3.0  # F37: cubic repulsion inverse temperature
GINIBRE_A: float = 1.2     # F38: Ginibre exponential constant


# ===========================================================================
# F28: Z2 decomposition
# ===========================================================================

def z2_decompose(h: np.ndarray, pc: float = P_C) -> Tuple[np.ndarray, np.ndarray]:
    """F28: Split h into physical (+) and momentum (-) components.

    h_+ = h * pc          (forward component)
    h_- = h * (1 - pc)    (backward component)

    physical = h_+ + h_-  = h
    momentum = h_+ - h_-  = h * (2*pc - 1)

    Parameters
    ----------
    h : np.ndarray
        Input state vector or matrix.
    pc : float
        Forward coupling probability (default 0.65).

    Returns
    -------
    physical : np.ndarray
        Physical component: h_+ + h_-.
    momentum : np.ndarray
        Momentum component: h_+ - h_-.
    """
    h_plus = h * pc
    h_minus = h * (1.0 - pc)
    physical = h_plus + h_minus
    momentum = h_plus - h_minus
    return physical, momentum


def z2_recombine(physical: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """Recombine Z2 components into original state.

    h_+ = (physical + momentum) / 2
    h_- = (physical - momentum) / 2
    h   = h_+ + h_- = physical

    (Identity since physical = h by construction.)
    """
    return physical.copy()


# ===========================================================================
# Quantum potential (Bohmian)
# ===========================================================================

def quantum_potential(physical: np.ndarray, momentum: np.ndarray) -> np.ndarray:
    """Compute Bohmian quantum potential from Z2 components.

    Q_i = m_i / x_i

    where m_i is the momentum component and x_i is the physical component.
    The mass parameter m = 0.1 (F48) is used as a regularisation.
    """
    m_bare = 0.1  # F48: bare mass parameter
    denom = np.abs(physical)
    denom = np.where(denom < 1e-12, 1e-12, denom)
    return m_bare * momentum / denom


# ===========================================================================
# F40, F45: Foss Topological Index
# ===========================================================================

def compute_foss_index(physical: np.ndarray, momentum: np.ndarray) -> float:
    """F40, F45: Foss Topological Index.

    F = S_ent / ln(n)

    where S_ent = -sum_i pi_i * ln(pi_i) is the Shannon entropy
    and pi are the stationary probabilities derived from the physical component.

    Target: F_oss = 0.75 +/- 0.05.
    Bounds: 0 <= F_oss <= 1.
    """
    # Derive probabilities from physical component magnitudes
    pi = np.abs(physical).flatten()
    pi_sum = np.sum(pi)
    if pi_sum < 1e-12:
        return 0.0
    pi = pi / pi_sum

    # Compute Shannon entropy
    pi_positive = pi[pi > 1e-12]
    s_ent = -np.sum(pi_positive * np.log(pi_positive))

    n = len(pi)
    if n <= 1:
        return 0.0

    foss = float(s_ent / np.log(n))
    return float(np.clip(foss, 0.0, 1.0))


def foss_index_from_distribution(pi: np.ndarray) -> float:
    """F45: Foss Topological Index from explicit probability distribution.

    F_oss = -sum_i pi_i * ln(pi_i) / ln(n)

    Target: 0.75 +/- 0.05.
    """
    pi = np.asarray(pi).flatten()
    pi_sum = np.sum(pi)
    if pi_sum < 1e-12:
        return 0.0
    pi = pi / pi_sum

    pi_pos = pi[pi > 1e-12]
    s_ent = -np.sum(pi_pos * np.log(pi_pos))

    n = len(pi)
    if n <= 1:
        return 0.0
    return float(np.clip(s_ent / np.log(n), 0.0, 1.0))


# ===========================================================================
# Self-loop transition
# ===========================================================================

def self_loop_transition(
    h: np.ndarray, h_new: np.ndarray, ps: float = P_S
) -> np.ndarray:
    """Apply self-loop transition: h_out = ps * h + (1 - ps) * h_new.

    Parameters
    ----------
    h : np.ndarray
        Current state.
    h_new : np.ndarray
        Proposed new state.
    ps : float
        Self-loop probability (default 0.003).

    Returns
    -------
    h_out : np.ndarray
        Blended state with self-loop memory.
    """
    return ps * h + (1.0 - ps) * h_new


# ===========================================================================
# F38: Ginibre kernel weighting
# ===========================================================================

def ginibre_weight(
    distances: np.ndarray, d_bar: float = 1.0
) -> np.ndarray:
    """F38: Ginibre kernel symbol-selection weight.

    s_j = d_H(x_ctx, e_j) / d_bar       (normalised spacing)
    w_j = s_j^3 * exp(-1.2 * s_j^2)

    Parameters
    ----------
    distances : np.ndarray
        Hyperbolic distances from context to each symbol.
    d_bar : float
        Mean distance for normalisation (default 1.0).

    Returns
    -------
    weights : np.ndarray
        Ginibre-kernel selection weights.
    """
    s = distances / max(d_bar, 1e-12)
    return s ** 3 * np.exp(-GINIBRE_A * s ** 2)


# ===========================================================================
# F47: Mobius strip parameterisation
# ===========================================================================

def mobius_strip_parameterise(
    u: float, v: float
) -> Tuple[float, float, float]:
    """F47: Mobius strip parameterisation for language cycles.

    (u, v) -> ((1 + v*cos(u/2))*cos(u),
                (1 + v*cos(u/2))*sin(u),
                v*sin(u/2))

    where u in [0, 2*pi), v in [-1, 1].

    Cyclic symbol sequences correspond to closed paths with odd winding
    number (non-contractible loops), ensuring semantic inversion after
    one full cycle.
    """
    x = (1.0 + v * np.cos(u / 2.0)) * np.cos(u)
    y = (1.0 + v * np.cos(u / 2.0)) * np.sin(u)
    z = v * np.sin(u / 2.0)
    return float(x), float(y), float(z)


# ===========================================================================
# F16-adjacent: Berry phase on Z2-doubled states
# ===========================================================================

def berry_phase_z2(states: np.ndarray) -> float:
    """Compute Berry phase for cyclic sequences on Z2-doubled states.

    Uses the momentum component as the phase generator:
    Phi_B = arg( prod_k <phys_k + i*mom_k, phys_{k+1} + i*mom_{k+1}> )

    This captures the topological twist introduced by the Z2 doubling.
    """
    n = states.shape[0]
    if n < 2:
        return 0.0

    # Decompose each state
    phys, mom = z2_decompose(states)

    # Complex inner product on the doubled space
    product = 1.0 + 0.0j
    for k in range(n):
        kp1 = (k + 1) % n
        z_k = complex(phys[k], mom[k]) if np.isscalar(phys[k]) else complex(np.mean(phys[k]), np.mean(mom[k]))
        z_kp1 = complex(phys[kp1], mom[kp1]) if np.isscalar(phys[kp1]) else complex(np.mean(phys[kp1]), np.mean(mom[kp1]))
        product *= z_k * np.conj(z_kp1)

    return float(np.angle(product))
