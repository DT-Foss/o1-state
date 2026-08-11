"""Direction-correct statistics for GroundZero-v1 shortcut controls."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from .certificates import MetricBound


Prediction = bool | None


@dataclass(frozen=True, slots=True)
class MatchedTwinControl:
    """One matched positive/negative shortcut probe.

    A perfectly inverted prediction still carries every answer bit, so it is
    counted as an informative leak.  Abstention or a constant prediction does
    not distinguish the twins.
    """

    positive_prediction: Prediction
    negative_prediction: Prediction

    def __post_init__(self) -> None:
        for value in (self.positive_prediction, self.negative_prediction):
            if value not in (True, False, None):
                raise TypeError("matched-twin predictions must be bool or None")

    @property
    def direct_accuracy(self) -> float:
        return (
            float(self.positive_prediction is True)
            + float(self.negative_prediction is False)
        ) / 2.0

    @property
    def answered_fraction(self) -> float:
        return (
            float(self.positive_prediction is not None)
            + float(self.negative_prediction is not None)
        ) / 2.0

    @property
    def informative_leak(self) -> bool:
        # Abstention is itself an observable output.  A selective pattern such
        # as (True, None) identifies the twin just as surely as (True, False),
        # even though it cannot satisfy the full benchmark coverage floor.
        return self.positive_prediction != self.negative_prediction


@dataclass(frozen=True, slots=True)
class ShortcutControlBound:
    """Upper-bound gate for a shortcut's twin-discrimination rate."""

    name: str
    leakage: MetricBound
    answer_coverage: MetricBound
    maximum_leakage: float
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("control name cannot be empty")
        if self.leakage.successes is None or self.answer_coverage.successes is None:
            raise TypeError("shortcut controls require binary metrics")
        if not 0.0 <= self.maximum_leakage < 1.0:
            raise ValueError("maximum_leakage must lie in [0, 1)")

    @property
    def rejected_as_grounder(self) -> bool:
        """True only when the shortcut leakage UCB is below the ceiling."""

        return self.leakage.upper_bound <= self.maximum_leakage

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "leakage": self.leakage.to_dict(),
            "answer_coverage": self.answer_coverage.to_dict(),
            "maximum_leakage": self.maximum_leakage,
            "description": self.description,
            "rejected_as_grounder": self.rejected_as_grounder,
        }


@dataclass(frozen=True, slots=True)
class ChanceShortcutControl:
    """Orientation-invariant upper-bound gate for a chance-level baseline."""

    name: str
    direct_outcomes: tuple[bool | None, ...]
    confidence: float = 0.95
    chance_level: float = 0.50
    tolerance: float = 0.25
    description: str = ""

    def __post_init__(self) -> None:
        outcomes = tuple(self.direct_outcomes)
        if not outcomes or any(value not in (True, False, None) for value in outcomes):
            raise ValueError(
                "direct_outcomes must be a non-empty bool/None sequence"
            )
        if not 0.0 <= self.chance_level < 1.0:
            raise ValueError("chance_level must lie in [0, 1)")
        if not 0.0 <= self.tolerance < 1.0 - self.chance_level:
            raise ValueError("tolerance must keep chance_level+tolerance below one")
        object.__setattr__(self, "direct_outcomes", outcomes)

    @property
    def orientation_invariant_successes(self) -> int:
        answered = tuple(value for value in self.direct_outcomes if value is not None)
        successes = sum(answered)
        return max(successes, len(answered) - successes)

    @property
    def answered(self) -> int:
        return sum(value is not None for value in self.direct_outcomes)

    @property
    def leakage(self) -> MetricBound:
        successes = self.orientation_invariant_successes
        return MetricBound.binary(
            (True,) * successes
            + (False,) * (self.answered - successes),
            confidence=self.confidence,
        )

    @property
    def coverage(self) -> MetricBound:
        return MetricBound.binary(
            (value is not None for value in self.direct_outcomes),
            confidence=self.confidence,
        )

    @property
    def maximum_leakage(self) -> float:
        return self.chance_level + self.tolerance

    @property
    def rejected_as_grounder(self) -> bool:
        return (
            self.coverage.upper_bound < 0.80
            or self.leakage.upper_bound <= self.maximum_leakage
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "leakage": self.leakage.to_dict(),
            "coverage": self.coverage.to_dict(),
            "chance_level": self.chance_level,
            "tolerance": self.tolerance,
            "maximum_leakage": self.maximum_leakage,
            "description": self.description,
            "rejected_as_grounder": self.rejected_as_grounder,
        }


def matched_twin_control_bound(
    name: str,
    outcomes: Iterable[MatchedTwinControl],
    *,
    maximum_leakage: float = 0.20,
    confidence: float = 0.95,
    description: str = "",
) -> ShortcutControlBound:
    values = tuple(outcomes)
    leakage = MetricBound.binary(
        (value.informative_leak for value in values), confidence=confidence
    )
    # Coverage is descriptive here; low coverage cannot manufacture evidence
    # for a positive capability axis, but an abstaining shortcut is correctly
    # harmless.  Leakage itself is gated in the conservative UCB direction.
    coverage = MetricBound.binary(
        (value.answered_fraction == 1.0 for value in values), confidence=confidence
    )
    return ShortcutControlBound(
        name,
        leakage,
        coverage,
        maximum_leakage,
        description,
    )


__all__ = [
    "ChanceShortcutControl",
    "MatchedTwinControl",
    "Prediction",
    "ShortcutControlBound",
    "matched_twin_control_bound",
]
