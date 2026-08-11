"""Entropy-Gradient Tau Advection (EGTA).

Self-tuning tau via entropy gradient of BvN path weights.
No learned parameters. Purely deterministic.

Formulas (from "Collapse is Contraction"):
    S(tau) = -sum_k (w_k / W) * log(w_k / W)     where W = sum_k w_k
    dF/dtau ~= (S(tau) - S(tau_prev)) / (tau - tau_prev + eps)
    tau_{t+1} = clamp(tau_t - eta * sgn(dF/dtau) * |dF/dtau|^alpha, tau_min, tau_max)
    eta = (1 - tau_t^2)^(3/2)                     [geometry-dependent step]
    alpha = 1/2                                   [fractional exponent]
"""

import numpy as np
from typing import Tuple, Optional


EPS: float = 1e-8
TAU_MIN: float = 0.3
TAU_MAX: float = 0.95
ALPHA: float = 0.5


def compute_entropy(path_weights: np.ndarray) -> float:
    """S(tau) = -sum_k (w_k / W) * log(w_k / W) -- Shannon entropy of BvN weights.

    Args:
        path_weights: 1-D array of non-negative BvN path weights.

    Returns:
        Shannon entropy (float, >= 0).
    """
    W = np.sum(path_weights) + EPS
    p = path_weights / W
    # Mask zero probabilities to avoid log(0)
    p_safe = np.where(p > EPS, p, EPS)
    S = -np.sum(p_safe * np.log(p_safe))
    return float(max(S, 0.0))


def compute_entropy_gradient(
    current_entropy: float,
    prev_entropy: float,
    current_tau: float,
    prev_tau: float,
) -> float:
    """dF/dtau ~= (S(tau) - S(tau_prev)) / (tau - tau_prev + eps)

    Finite-difference approximation of the entropy gradient with respect
    to tau.  When prev_* values are unavailable, falls back to the
    analytic first-order approximation from the paper:
        dF/dtau ~= -(1 / tau^2) * S(tau).

    Args:
        current_entropy:  S(tau_t).
        prev_entropy:     S(tau_{t-1}), or None for first step.
        current_tau:      tau_t.
        prev_tau:         tau_{t-1}, or None for first step.

    Returns:
        Scalar entropy gradient (float).
    """
    if prev_entropy is not None and prev_tau is not None:
        dS_dtau = (current_entropy - prev_entropy) / (current_tau - prev_tau + EPS)
    else:
        # Analytic approximation: dF/dtau ~= -(1/tau^2) * S
        dS_dtau = -(1.0 / (current_tau ** 2 + EPS)) * current_entropy
    return float(dS_dtau)


def compute_step_size(tau: float) -> float:
    """eta = (1 - tau^2)^(3/2) -- geometry-dependent step size.

    Derived from dT/dtau = tau / (1 - tau^2)^(3/2) on the Moebius
    temperature map T(tau) = (1 - tau^2)^(-1/2) - 1.

    Args:
        tau: Current tau value in (0, 1).

    Returns:
        Step size eta (float).
    """
    return float((1.0 - tau ** 2) ** 1.5)


def _sgn(x: float) -> float:
    """sgn(x) = +1 if x > 0, -1 if x < 0, 0 if x == 0."""
    if x > EPS:
        return 1.0
    elif x < -EPS:
        return -1.0
    return 0.0


def advect_tau(
    tau_current: float,
    entropy_gradient: float,
    tau_min: float = TAU_MIN,
    tau_max: float = TAU_MAX,
    alpha: float = ALPHA,
) -> float:
    """tau_{t+1} = clamp(tau_t - eta * sgn(dF/dtau) * |dF/dtau|^alpha, tau_min, tau_max)

    Deterministic tau advection driven by the entropy gradient.

    Args:
        tau_current:       tau_t.
        entropy_gradient:  dF/dtau (scalar).
        tau_min:           Lower bound for tau.
        tau_max:           Upper bound for tau.
        alpha:             Fractional exponent (default 1/2).

    Returns:
        New tau value (float).
    """
    eta = compute_step_size(tau_current)
    direction = _sgn(entropy_gradient)
    magnitude = abs(entropy_gradient) ** alpha
    tau_next = tau_current - eta * direction * magnitude
    tau_next = max(tau_min, min(tau_max, tau_next))
    return float(tau_next)


def egta_update(
    tau: float,
    path_weights: np.ndarray,
    prev_tau: Optional[float] = None,
    prev_entropy: Optional[float] = None,
) -> Tuple[float, float]:
    """Full EGTA update.  Returns (new_tau, new_entropy).

    Usage:
        tau = 0.65
        prev_entropy = None
        for step in range(max_tokens):
            path_weights = bvn_decompose(logits)
            tau, prev_entropy = egta_update(
                tau, path_weights,
                prev_tau=tau if step > 0 else None,
                prev_entropy=prev_entropy,
            )
            token = contraction_sample(logits, tau=tau)

    Args:
        tau:           Current tau value.
        path_weights:  BvN path weights (1-D array, non-negative).
        prev_tau:      Previous tau (None on first step).
        prev_entropy:  Previous entropy (None on first step).

    Returns:
        (new_tau, current_entropy) as floats.
    """
    S_current = compute_entropy(path_weights)
    dF_dtau = compute_entropy_gradient(S_current, prev_entropy, tau, prev_tau)
    tau_new = advect_tau(tau, dF_dtau)
    return tau_new, S_current


class EGTAScheduler:
    """Stateful EGTA scheduler for generation sequences.

    Maintains tau and entropy history so callers need not track
    prev_tau / prev_entropy manually.
    """

    def __init__(self, tau_init: float = 0.65, tau_min: float = TAU_MIN, tau_max: float = TAU_MAX):
        self.tau = tau_init
        self.tau_min = tau_min
        self.tau_max = tau_max
        self._prev_tau: Optional[float] = None
        self._prev_entropy: Optional[float] = None
        self.tau_history: list = []
        self.entropy_history: list = []

    def step(self, path_weights: np.ndarray) -> float:
        """Return tau for this generation step.

        Internally updates the stored tau and entropy so the next
        call uses the current values as ``prev_*``.

        Args:
            path_weights: BvN path weights (1-D array, non-negative).

        Returns:
            New tau value for this step.
        """
        tau_new, S_current = egta_update(
            self.tau,
            path_weights,
            prev_tau=self._prev_tau,
            prev_entropy=self._prev_entropy,
        )
        # Store current values as "previous" for next step
        self._prev_tau = self.tau
        self._prev_entropy = S_current
        self.tau = tau_new
        self.tau_history.append(tau_new)
        self.entropy_history.append(S_current)
        return tau_new

    def reset(self, tau_init: float = 0.65):
        """Reset scheduler for a new generation sequence."""
        self.tau = tau_init
        self._prev_tau = None
        self._prev_entropy = None
        self.tau_history.clear()
        self.entropy_history.clear()
