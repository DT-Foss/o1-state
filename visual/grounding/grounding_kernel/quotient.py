"""Finite interventional semantics and an executable completeness theorem.

The module makes a narrow but strong statement.  In a declared finite world,
two histories have the same operational meaning when every available
intervention induces the same distribution over public consequences.  A
predicate is groundable only if it is constant on those equivalence classes.

For Boolean predicates, a set of grounded anchors is expressively complete
exactly when their truth signatures separate all operational classes.  The
forward direction is constructive: :func:`synthesize_boolean_program` returns
a disjunctive-normal-form program for any groundable Boolean target.  The
converse is an impossibility witness, not a low benchmark score: two classes
with the same anchor signature cannot be separated by any Boolean formula over
those anchors.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from math import isclose, isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any


class OperationalSemanticsError(ValueError):
    """Base class for invalid or non-identifiable finite semantics."""


class NonTransitiveToleranceError(OperationalSemanticsError):
    """Raised when a numeric threshold does not induce an equivalence relation."""


class NonGroundablePredicateError(OperationalSemanticsError):
    """Raised when a target differs inside an operational equivalence class."""


class IncompleteAnchorBasisError(OperationalSemanticsError):
    """Raised when grounded anchors do not separate the operational quotient."""


def _require_unique_hashables(values: Sequence[Hashable], label: str) -> tuple[Hashable, ...]:
    result = tuple(values)
    if not result:
        raise OperationalSemanticsError(f"{label} cannot be empty")
    for value in result:
        try:
            hash(value)
        except TypeError as exc:
            raise TypeError(f"{label} must contain only hashable values") from exc
    if len(set(result)) != len(result):
        raise OperationalSemanticsError(f"{label} must be unique")
    return result


def _stable_key(value: Hashable) -> tuple[str, str]:
    return type(value).__qualname__, repr(value)


@dataclass(frozen=True, slots=True)
class NonIdentifiabilityWitness:
    """Two operationally identical histories that a target tries to distinguish."""

    left: Hashable
    right: Hashable
    left_value: Hashable
    right_value: Hashable


@dataclass(frozen=True, slots=True)
class PredicateGroundability:
    """Whether a declared predicate factors through the operational quotient.

    This is a necessary identifiability condition inside the declared finite
    model, not by itself evidence that a learner acquired or uses the predicate.
    """

    groundable: bool
    quotient: tuple[tuple[Hashable, ...], ...]
    witness: NonIdentifiabilityWitness | None = None

    @property
    def quotient_compatible(self) -> bool:
        """Explicit name for the scoped necessary condition."""

        return self.groundable


@dataclass(frozen=True, slots=True)
class AnchorGroundabilityFailure:
    """An anchor and the quotient pair on which it changes value."""

    anchor: Hashable
    witness: NonIdentifiabilityWitness


@dataclass(frozen=True, slots=True)
class AnchorClosureCertificate:
    """Necessary-and-sufficient finite Boolean expressivity certificate."""

    quotient: tuple[tuple[Hashable, ...], ...]
    anchors: tuple[Hashable, ...]
    signatures: tuple[tuple[bool, ...], ...]
    all_anchors_groundable: bool
    separates_quotient: bool
    indistinguishable_classes: tuple[int, int] | None
    ungroundable_anchors: tuple[AnchorGroundabilityFailure, ...] = ()

    @property
    def quotient_size(self) -> int:
        return len(self.quotient)

    @property
    def signature_class_count(self) -> int:
        return len(set(self.signatures))

    @property
    def expressible_subsets(self) -> int:
        """Number of extensional subsets expressible by Boolean anchor formulas."""

        return 1 << self.signature_class_count

    @property
    def certified_expressible_subsets(self) -> int:
        """Expressible subsets whose basis itself passes the grounding condition."""

        return self.expressible_subsets if self.all_anchors_groundable else 0

    @property
    def total_groundable_subsets(self) -> int:
        return 1 << self.quotient_size

    @property
    def complete(self) -> bool:
        return self.all_anchors_groundable and self.separates_quotient

    @property
    def theorem(self) -> str:
        if self.complete:
            return (
                "Every Boolean predicate constant on the operational quotient is "
                "constructively expressible as a disjunction of anchor minterms."
            )
        if not self.all_anchors_groundable:
            names = ", ".join(repr(item.anchor) for item in self.ungroundable_anchors)
            return (
                "The proposed basis is not grounded: anchor(s) "
                f"{names} change value inside an operational equivalence class."
            )
        return (
            "No Boolean formula over these anchors can distinguish every operational "
            "class; the reported equal-signature pair is an impossibility witness."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "quotient": [[repr(history) for history in block] for block in self.quotient],
            "anchors": [repr(anchor) for anchor in self.anchors],
            "signatures": [list(signature) for signature in self.signatures],
            "all_anchors_groundable": self.all_anchors_groundable,
            "separates_quotient": self.separates_quotient,
            "indistinguishable_classes": self.indistinguishable_classes,
            "ungroundable_anchors": [
                {
                    "anchor": repr(item.anchor),
                    "left": repr(item.witness.left),
                    "right": repr(item.witness.right),
                    "left_value": repr(item.witness.left_value),
                    "right_value": repr(item.witness.right_value),
                }
                for item in self.ungroundable_anchors
            ],
            "quotient_size": self.quotient_size,
            "signature_class_count": self.signature_class_count,
            "expressible_subsets": self.expressible_subsets,
            "certified_expressible_subsets": self.certified_expressible_subsets,
            "total_groundable_subsets": self.total_groundable_subsets,
            "complete": self.complete,
            "theorem": self.theorem,
        }


@dataclass(frozen=True, slots=True)
class Literal:
    anchor: Hashable
    positive: bool


@dataclass(frozen=True, slots=True)
class BooleanGroundingProgram:
    """Constructive DNF proof that a target is generated by grounded anchors."""

    terms: tuple[tuple[Literal, ...], ...]
    target_positive_classes: tuple[int, ...]
    certificate: AnchorClosureCertificate

    def evaluate(
        self,
        history: Hashable,
        anchors: Mapping[Hashable, Mapping[Hashable, bool]],
    ) -> bool:
        """Evaluate the synthesized program using only anchor judgments."""

        def literal_value(literal: Literal) -> bool:
            try:
                value = anchors[literal.anchor][history]
            except KeyError as exc:
                raise KeyError(
                    f"missing value for anchor {literal.anchor!r} at history {history!r}"
                ) from exc
            if not isinstance(value, bool):
                raise TypeError("Boolean grounding programs require Boolean anchor values")
            return value if literal.positive else not value

        return any(all(literal_value(literal) for literal in term) for term in self.terms)

    def to_dict(self) -> dict[str, Any]:
        return {
            "terms": [
                [
                    {"anchor": repr(literal.anchor), "positive": literal.positive}
                    for literal in term
                ]
                for term in self.terms
            ],
            "target_positive_classes": list(self.target_positive_classes),
            "certificate_complete": self.certificate.complete,
        }


class FiniteInterventionalModel:
    """A validated categorical kernel ``P(outcome | history, do(intervention))``."""

    def __init__(
        self,
        histories: Sequence[Hashable],
        interventions: Sequence[Hashable],
        distributions: Mapping[tuple[Hashable, Hashable], Mapping[Hashable, float]],
        *,
        normalization_tolerance: float = 1e-9,
    ) -> None:
        self._histories = _require_unique_hashables(histories, "histories")
        self._interventions = _require_unique_hashables(interventions, "interventions")
        if (
            isinstance(normalization_tolerance, bool)
            or not isinstance(normalization_tolerance, Real)
            or not isfinite(float(normalization_tolerance))
            or not 0.0 <= float(normalization_tolerance) <= 1e-6
        ):
            raise OperationalSemanticsError(
                "normalization_tolerance must lie in [0, 1e-6]"
            )
        normalization_tolerance = float(normalization_tolerance)
        expected = {
            (history, intervention)
            for history in self._histories
            for intervention in self._interventions
        }
        supplied = set(distributions)
        if supplied != expected:
            missing = expected - supplied
            extra = supplied - expected
            raise OperationalSemanticsError(
                f"kernel must define the full history×intervention product; "
                f"missing={len(missing)}, extra={len(extra)}"
            )
        rows: dict[tuple[Hashable, Hashable], Mapping[Hashable, float]] = {}
        for key in sorted(expected, key=lambda item: (_stable_key(item[0]), _stable_key(item[1]))):
            raw = distributions[key]
            if not raw:
                raise OperationalSemanticsError(f"empty outcome distribution at {key!r}")
            row: dict[Hashable, float] = {}
            for outcome, probability in raw.items():
                try:
                    hash(outcome)
                except TypeError as exc:
                    raise TypeError("outcomes must be hashable") from exc
                if isinstance(probability, bool) or not isinstance(probability, Real):
                    raise TypeError("probabilities must be real numeric values")
                value = float(probability)
                if not isfinite(value) or not 0.0 <= value <= 1.0:
                    raise OperationalSemanticsError(
                        f"probabilities must be finite and within [0, 1] at {key!r}"
                    )
                row[outcome] = value
            total = sum(row.values())
            if total <= 0.0:
                raise OperationalSemanticsError(
                    f"probabilities at {key!r} need positive total mass"
                )
            if not isclose(total, 1.0, rel_tol=0.0, abs_tol=normalization_tolerance):
                raise OperationalSemanticsError(
                    f"probabilities at {key!r} sum to {total!r}, expected 1"
                )
            rows[key] = MappingProxyType(
                {outcome: probability / total for outcome, probability in row.items()}
            )
        self._distributions = MappingProxyType(rows)

    @property
    def histories(self) -> tuple[Hashable, ...]:
        return self._histories

    @property
    def interventions(self) -> tuple[Hashable, ...]:
        return self._interventions

    @property
    def distributions(
        self,
    ) -> Mapping[tuple[Hashable, Hashable], Mapping[Hashable, float]]:
        return self._distributions

    def total_variation(
        self,
        left: Hashable,
        right: Hashable,
        intervention: Hashable,
    ) -> float:
        """Return total-variation distance for one controlled intervention."""

        try:
            left_row = self._distributions[(left, intervention)]
            right_row = self._distributions[(right, intervention)]
        except KeyError as exc:
            raise KeyError("unknown history or intervention") from exc
        outcomes = set(left_row) | set(right_row)
        return 0.5 * sum(
            abs(left_row.get(outcome, 0.0) - right_row.get(outcome, 0.0))
            for outcome in outcomes
        )

    def operational_distance(self, left: Hashable, right: Hashable) -> float:
        """Return ``sup_i TV(P(.|left,do(i)), P(.|right,do(i)))``."""

        return max(
            self.total_variation(left, right, intervention)
            for intervention in self._interventions
        )

    def operational_partition(
        self,
        *,
        tolerance: float = 0.0,
    ) -> tuple[tuple[Hashable, ...], ...]:
        """Return the sensorimotor quotient, rejecting non-transitive thresholds.

        Approximate pairwise closeness is not automatically an equivalence
        relation.  Silently taking connected components would falsely merge
        endpoints farther apart than ``tolerance``, so this method fails closed.
        """

        if not isfinite(tolerance) or tolerance < 0.0:
            raise OperationalSemanticsError("tolerance must be finite and non-negative")
        similar = {
            (left, right): self.operational_distance(left, right) <= tolerance
            for left in self._histories
            for right in self._histories
        }
        for left in self._histories:
            for middle in self._histories:
                if not similar[(left, middle)]:
                    continue
                for right in self._histories:
                    if similar[(middle, right)] and not similar[(left, right)]:
                        raise NonTransitiveToleranceError(
                            "the requested tolerance does not induce an equivalence relation: "
                            f"{left!r}≈{middle!r} and {middle!r}≈{right!r}, but "
                            f"{left!r}≉{right!r}"
                        )
        remaining = set(self._histories)
        blocks: list[tuple[Hashable, ...]] = []
        for representative in self._histories:
            if representative not in remaining:
                continue
            block = tuple(
                history
                for history in self._histories
                if history in remaining and similar[(representative, history)]
            )
            remaining.difference_update(block)
            blocks.append(block)
        return tuple(blocks)

    def predicate_groundability(
        self,
        predicate: Mapping[Hashable, Hashable],
        *,
        tolerance: float = 0.0,
    ) -> PredicateGroundability:
        """Check the necessary quotient-factorization condition for a predicate."""

        self._validate_total_predicate(predicate)
        quotient = self.operational_partition(tolerance=tolerance)
        for block in quotient:
            representative = block[0]
            for history in block[1:]:
                if predicate[history] != predicate[representative]:
                    return PredicateGroundability(
                        False,
                        quotient,
                        NonIdentifiabilityWitness(
                            representative,
                            history,
                            predicate[representative],
                            predicate[history],
                        ),
                    )
        return PredicateGroundability(True, quotient)

    def anchor_closure_certificate(
        self,
        anchors: Mapping[Hashable, Mapping[Hashable, bool]],
        *,
        tolerance: float = 0.0,
    ) -> AnchorClosureCertificate:
        """Certify whether Boolean anchors generate every groundable predicate."""

        quotient = self.operational_partition(tolerance=tolerance)
        names = tuple(sorted(anchors, key=_stable_key))
        failures: list[AnchorGroundabilityFailure] = []
        for name in names:
            values = anchors[name]
            self._validate_boolean_predicate(values)
            report = self.predicate_groundability(values, tolerance=tolerance)
            if not report.groundable:
                assert report.witness is not None
                failures.append(AnchorGroundabilityFailure(name, report.witness))
        signatures = tuple(
            tuple(bool(anchors[name][block[0]]) for name in names) for block in quotient
        )
        witness: tuple[int, int] | None = None
        seen: dict[tuple[bool, ...], int] = {}
        for index, signature in enumerate(signatures):
            if signature in seen and witness is None:
                witness = (seen[signature], index)
            seen.setdefault(signature, index)
        return AnchorClosureCertificate(
            quotient=quotient,
            anchors=names,
            signatures=signatures,
            all_anchors_groundable=not failures,
            separates_quotient=witness is None,
            indistinguishable_classes=witness,
            ungroundable_anchors=tuple(failures),
        )

    def synthesize_boolean_program(
        self,
        target: Mapping[Hashable, bool],
        anchors: Mapping[Hashable, Mapping[Hashable, bool]],
        *,
        tolerance: float = 0.0,
    ) -> BooleanGroundingProgram:
        """Construct a Boolean anchor program for a groundable target.

        This function is the constructive half of the finite completeness
        theorem.  It refuses both non-groundable targets and incomplete anchor
        bases rather than returning a best-effort symbolic approximation.
        """

        self._validate_boolean_predicate(target)
        target_report = self.predicate_groundability(target, tolerance=tolerance)
        if not target_report.groundable:
            witness = target_report.witness
            assert witness is not None
            raise NonGroundablePredicateError(
                "target differs on operationally indistinguishable histories: "
                f"{witness.left!r} and {witness.right!r}"
            )
        certificate = self.anchor_closure_certificate(anchors, tolerance=tolerance)
        if not certificate.complete:
            raise IncompleteAnchorBasisError(certificate.theorem)
        terms: list[tuple[Literal, ...]] = []
        positive_classes: list[int] = []
        for index, block in enumerate(certificate.quotient):
            if not target[block[0]]:
                continue
            positive_classes.append(index)
            terms.append(
                tuple(
                    Literal(anchor, value)
                    for anchor, value in zip(
                        certificate.anchors,
                        certificate.signatures[index],
                        strict=True,
                    )
                )
            )
        return BooleanGroundingProgram(tuple(terms), tuple(positive_classes), certificate)

    def _validate_total_predicate(self, predicate: Mapping[Hashable, Hashable]) -> None:
        expected = set(self._histories)
        supplied = set(predicate)
        if supplied != expected:
            raise OperationalSemanticsError(
                "predicate must define every and only declared history; "
                f"missing={len(expected - supplied)}, extra={len(supplied - expected)}"
            )
        for value in predicate.values():
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError("predicate values must be hashable") from exc

    def _validate_boolean_predicate(self, predicate: Mapping[Hashable, bool]) -> None:
        self._validate_total_predicate(predicate)
        if any(not isinstance(value, bool) for value in predicate.values()):
            raise TypeError("Boolean predicates require bool values, not truthy values")


__all__ = [
    "AnchorClosureCertificate",
    "AnchorGroundabilityFailure",
    "BooleanGroundingProgram",
    "FiniteInterventionalModel",
    "IncompleteAnchorBasisError",
    "Literal",
    "NonGroundablePredicateError",
    "NonIdentifiabilityWitness",
    "NonTransitiveToleranceError",
    "OperationalSemanticsError",
    "PredicateGroundability",
]
