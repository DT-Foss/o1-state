"""Foss Gate — deterministic quality filter tied to the Möbius state.

The gate now validates extracted causal triplets against metrics computed
from the actual Möbius-SSM state (geometric naturalness, Foss topological
index, and structural checks) instead of token-hash statistics.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from itertools import groupby


class FossGate:
    """Deterministic quality filter for VERITAS causal triplets.

    Parameters
    ----------
    min_confidence : float
        Minimum geometric confidence for a triplet to be accepted.
    foss_index_range : tuple[float, float]
        Acceptable range for the Foss topological index computed from the
        Z2-decomposed Möbius state.
    min_component_len : int
        Minimum character length for trigger / mechanism / outcome.
    max_repetition : int
        Maximum allowed length of a run of identical words in a component.
    """

    def __init__(
        self,
        min_confidence: float = 0.10,
        foss_index_range: Tuple[float, float] = (0.60, 0.98),
        min_component_len: int = 4,
        max_repetition: int = 3,
    ):
        self.min_confidence = min_confidence
        self.foss_index_range = foss_index_range
        self.min_component_len = min_component_len
        self.max_repetition = max_repetition
        self.stats = {
            "checked": 0,
            "passed": 0,
            "rejected": 0,
            "step_failures": {f"P{i}": 0 for i in range(1, 7)},
        }

    def _record(self, passed: bool, step: str, reason: str):
        self.stats["checked"] += 1
        if passed:
            self.stats["passed"] += 1
        else:
            self.stats["rejected"] += 1
            self.stats["step_failures"][step] += 1
        return passed, reason

    def validate_triplet(
        self,
        triplet,
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Validate a CausalTriplet using real Möbius-state metrics.

        Parameters
        ----------
        triplet : CausalTriplet
            The extracted causal triplet.
        metrics : dict, optional
            Must contain keys produced by VeritasEngine:
            - 'geometric_conf' : float, geometric naturalness/confidence
            - 'foss_index'     : float, Foss topological index
            - 'physical'       : np.ndarray, physical Z2 component
            - 'momentum'       : np.ndarray, momentum Z2 component

        Returns
        -------
        (is_valid, reason) : Tuple[bool, str]
        """
        metrics = metrics or {}
        geometric_conf = float(metrics.get("geometric_conf", 0.0))
        foss_index = float(metrics.get("foss_index", -1.0))

        # P1: structural length checks
        for name, value in [
            ("trigger", triplet.trigger),
            ("mechanism", triplet.mechanism),
            ("outcome", triplet.outcome),
        ]:
            if len(value.strip()) < self.min_component_len:
                return self._record(
                    False, "P1", f"{name} too short ({len(value)} chars)"
                )

        # P2: geometric confidence threshold
        if geometric_conf < self.min_confidence:
            return self._record(
                False,
                "P2",
                f"geometric confidence {geometric_conf:.4f} < {self.min_confidence}",
            )

        # P3: Foss topological index within expected manifold range
        if not (self.foss_index_range[0] <= foss_index <= self.foss_index_range[1]):
            return self._record(
                False,
                "P3",
                f"Foss index {foss_index:.4f} outside {self.foss_index_range}",
            )

        # P4: repetition guard — reject long runs of the same token
        for component in (triplet.trigger, triplet.mechanism, triplet.outcome):
            words = component.lower().split()
            for _, group in groupby(words):
                if sum(1 for _ in group) > self.max_repetition:
                    return self._record(False, "P4", "excessive token repetition")

        # P5: sanity — mechanism must contain a verb-like token or marker
        mechanism = triplet.mechanism.lower()
        if not any(
            c.isalpha() and len(c) > 2 for c in mechanism.split()
        ):
            return self._record(False, "P5", "mechanism lacks content")

        # P6: state finiteness
        physical = metrics.get("physical")
        momentum = metrics.get("momentum")
        if physical is not None and momentum is not None:
            if not (np.all(np.isfinite(physical)) and np.all(np.isfinite(momentum))):
                return self._record(False, "P6", "non-finite Z2 state components")

        return self._record(True, "", "")

    def get_stats(self) -> dict:
        """Return gate statistics."""
        checked = self.stats["checked"]
        rejected = self.stats["rejected"]
        return {
            **self.stats,
            "rejection_rate_pct": round(rejected / checked * 100.0, 2) if checked else 0.0,
        }
