"""Deterministic active acquisition of opaque operational concepts.

This module deliberately knows only learner-visible records: an opaque action,
raw sensor change, optional generic scalar feedback, and (for v0 compatibility)
an optional opaque outcome code.  A hypothesis is consequently a table of
observable consequence distributions, never a decoder for a hidden world
property.  GroundZero-v1 trials omit outcome codes entirely.

The implementation supports both exact version spaces (use deterministic
consequences) and finite Bayesian models (use categorical distributions).
Expected posterior entropy is evaluated exactly, and every executed probe is
charged to an immutable, hashable experiment ledger.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from hashlib import sha256
from math import fsum, isfinite, log2
from types import MappingProxyType
from typing import Protocol, runtime_checkable
import json

import numpy as np


_EPSILON = 1e-12
_USE_KEY = object()


class InconsistentObservationError(ValueError):
    """Raised when every hypothesis assigns zero likelihood to an observation."""


@runtime_checkable
class LearnerVisibleTransition(Protocol):
    """Structural sensorimotor subset shared by v0 and v1 transitions."""

    before: object
    action: object
    after: object


@runtime_checkable
class InterventionObserver(Protocol):
    """Execute one learner-visible intervention and return its evidence."""

    def __call__(self, payload: object) -> object: ...


@runtime_checkable
class ConsequenceEncoder(Protocol):
    """Turn returned evidence into a hashable operational consequence."""

    def __call__(self, evidence: object) -> Hashable: ...


@runtime_checkable
class HypothesisModel(Protocol):
    """Minimal protocol accepted by :class:`BayesianVersionSpace`."""

    hypothesis_id: Hashable
    prior: float

    def distribution(self, intervention_key: Hashable) -> Mapping[Hashable, float]: ...


@runtime_checkable
class AcquisitionPolicy(Protocol):
    """Policy interface used by :func:`run_acquisition`."""

    name: str

    def select(
        self,
        version_space: "BayesianVersionSpace",
        candidates: Sequence["CandidateIntervention"],
        ledger: "AcquisitionLedger",
        budget: "ProbeBudget",
    ) -> "ProbeScore | None": ...


def _require_hashable(value: object, field: str) -> Hashable:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be hashable") from exc
    return value  # type: ignore[return-value]


def _canonical(value: object) -> object:
    """Return a deterministic JSON value without interpreting opaque labels."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError("ledger values must be finite")
        return {"float": value.hex()}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, bytes):
        return {"bytes": value.hex()}
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: _stable_key(item[0]))
        return {"mapping": [[_canonical(key), _canonical(item)] for key, item in items]}
    if isinstance(value, (tuple, list)):
        return [_canonical(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "dataclass": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": [
                [field.name, _canonical(getattr(value, field.name))]
                for field in fields(value)
                if field.repr
            ],
        }
    return {
        "opaque_type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
    }


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _stable_key(value: object) -> tuple[str, bytes]:
    return (f"{type(value).__module__}.{type(value).__qualname__}", _canonical_bytes(value))


def _field(record: object, name: str, default: object = _USE_KEY) -> object:
    if isinstance(record, Mapping):
        if name in record:
            return record[name]
    elif hasattr(record, name):
        return getattr(record, name)
    if default is _USE_KEY:
        raise AttributeError(f"learner-visible evidence is missing {name!r}")
    return default


def _action_component(action: object, name: str) -> Hashable | None:
    value = _field(action, name, None)
    if value is None:
        return None
    if isinstance(value, list):
        value = tuple(value)
    return _require_hashable(value, f"action {name}")


def _pixel_array(observation: object) -> np.ndarray | None:
    pixels = _field(observation, "pixels", None)
    if pixels is None:
        if isinstance(observation, np.ndarray):
            pixels = observation
        else:
            return None
    array = np.asarray(pixels)
    if array.size == 0 or not (
        np.issubdtype(array.dtype, np.number) or array.dtype == np.bool_
    ):
        raise TypeError("pixels must be a non-empty numeric array")
    return array


@dataclass(frozen=True, slots=True)
class ObservableConsequence:
    """Token-free signature extracted solely from public action/transition data."""

    action_code: Hashable | None
    action_target: Hashable | None
    action_vector: Hashable | None
    outcome_code: Hashable | None
    pixels_changed: bool
    changed_values: int | None
    absolute_change: float | None
    change_digest: str | None
    task_feedback: bool | None = None
    scalar_feedback: float | None = None

    def __post_init__(self) -> None:
        for name in ("action_code", "action_target", "action_vector", "outcome_code"):
            value = getattr(self, name)
            if value is not None:
                _require_hashable(value, name)
        if self.changed_values is not None and self.changed_values < 0:
            raise ValueError("changed_values cannot be negative")
        if self.absolute_change is not None and (
            not isfinite(self.absolute_change) or self.absolute_change < 0.0
        ):
            raise ValueError("absolute_change must be finite and non-negative")
        if self.scalar_feedback is not None and not isfinite(self.scalar_feedback):
            raise ValueError("scalar_feedback must be finite")


def learner_visible_consequence(record: object) -> Hashable:
    """Encode a public transition/evidence record, ignoring its concept token.

    Already-hashable values that do not look like transition records pass
    through unchanged.  This makes small categorical test worlds convenient,
    while ``Transition`` and ``GroundingEvidence`` work without adapters.
    """

    if isinstance(record, ObservableConsequence):
        return record
    has_transition_shape = all(
        (name in record if isinstance(record, Mapping) else hasattr(record, name))
        for name in ("before", "action", "after")
    )
    if not has_transition_shape:
        return _require_hashable(record, "observed consequence")

    before = _field(record, "before")
    action = _field(record, "action")
    after = _field(record, "after")
    outcome_value = _field(record, "outcome_code", None)
    outcome = (
        None
        if outcome_value is None
        else _require_hashable(outcome_value, "outcome_code")
    )
    feedback_value = _field(record, "task_feedback", None)
    feedback = None if feedback_value is None else bool(feedback_value)
    scalar_feedback_value = _field(record, "scalar_feedback", None)
    scalar_feedback = None
    if scalar_feedback_value is not None:
        scalar_feedback = float(scalar_feedback_value)
        if not isfinite(scalar_feedback):
            raise ValueError("scalar_feedback must be finite")

    before_pixels = _pixel_array(before)
    after_pixels = _pixel_array(after)
    changed_values: int | None = None
    absolute_change: float | None = None
    change_digest: str | None = None
    if before_pixels is not None or after_pixels is not None:
        if before_pixels is None or after_pixels is None:
            raise ValueError("before and after must expose pixels together")
        if before_pixels.shape != after_pixels.shape:
            raise ValueError("before and after pixel shapes differ")
        difference = after_pixels.astype(np.float64) - before_pixels.astype(np.float64)
        changed_values = int(np.count_nonzero(difference))
        absolute_change = float(np.sum(np.abs(difference), dtype=np.float64))
        digest_payload = str(tuple(int(v) for v in difference.shape)).encode("ascii")
        digest_payload += np.ascontiguousarray(difference, dtype=np.float64).tobytes()
        change_digest = sha256(digest_payload).hexdigest()
        pixels_changed = changed_values > 0
    else:
        explicit_change = _field(record, "pixels_changed", None)
        if explicit_change is None:
            explicit_change = before != after
        pixels_changed = bool(explicit_change)

    return ObservableConsequence(
        action_code=_action_component(action, "code"),
        action_target=_action_component(action, "target"),
        action_vector=_action_component(action, "vector"),
        outcome_code=outcome,
        pixels_changed=pixels_changed,
        changed_values=changed_values,
        absolute_change=absolute_change,
        change_digest=change_digest,
        task_feedback=feedback,
        scalar_feedback=scalar_feedback,
    )


@dataclass(frozen=True, slots=True)
class CandidateIntervention:
    """One opaque, costed probe candidate.

    ``payload`` is excluded from equality and ledgers because it may be a live
    action object.  ``key`` is its stable learner-chosen identity.
    """

    key: Hashable
    payload: object = field(default=_USE_KEY, compare=False, repr=False)
    cost: float = 1.0
    max_uses: int = 1

    def __post_init__(self) -> None:
        _require_hashable(self.key, "intervention key")
        if self.payload is _USE_KEY:
            object.__setattr__(self, "payload", self.key)
        cost = float(self.cost)
        if not isfinite(cost) or cost <= 0.0:
            raise ValueError("intervention cost must be finite and positive")
        if isinstance(self.max_uses, bool) or not isinstance(self.max_uses, int):
            raise TypeError("max_uses must be an integer")
        if self.max_uses < 1:
            raise ValueError("max_uses must be positive")
        object.__setattr__(self, "cost", cost)

    def __hash__(self) -> int:
        return hash((self.key, self.cost, self.max_uses))


def _normalise_distribution(specification: object) -> Mapping[Hashable, float]:
    if isinstance(specification, Mapping):
        if not specification:
            raise ValueError("consequence distributions cannot be empty")
        raw: list[tuple[Hashable, float]] = []
        for outcome, probability in specification.items():
            key = _require_hashable(outcome, "consequence signature")
            value = float(probability)
            if not isfinite(value) or value < 0.0:
                raise ValueError("consequence probabilities must be finite and non-negative")
            raw.append((key, value))
        total = fsum(value for _, value in raw)
        if total <= 0.0:
            raise ValueError("a consequence distribution needs positive mass")
        values = {
            key: value / total
            for key, value in sorted(raw, key=lambda item: _stable_key(item[0]))
            if value > 0.0
        }
        return MappingProxyType(values)
    outcome = _require_hashable(specification, "consequence signature")
    return MappingProxyType({outcome: 1.0})


@dataclass(frozen=True, slots=True)
class OperationalHypothesis:
    """An opaque hypothesis identified only by predicted consequences."""

    hypothesis_id: Hashable
    consequences: Mapping[Hashable, object]
    prior: float = 1.0

    def __post_init__(self) -> None:
        if self.hypothesis_id is None:
            raise ValueError("hypothesis_id cannot be None; None is reserved for abstention")
        _require_hashable(self.hypothesis_id, "hypothesis_id")
        prior = float(self.prior)
        if not isfinite(prior) or prior <= 0.0:
            raise ValueError("hypothesis prior must be finite and positive")
        table: dict[Hashable, Mapping[Hashable, float]] = {}
        for intervention, specification in sorted(
            self.consequences.items(), key=lambda item: _stable_key(item[0])
        ):
            key = _require_hashable(intervention, "intervention key")
            table[key] = _normalise_distribution(specification)
        if not table:
            raise ValueError("a hypothesis needs at least one intervention prediction")
        object.__setattr__(self, "consequences", MappingProxyType(table))
        object.__setattr__(self, "prior", prior)

    def distribution(self, intervention_key: Hashable) -> Mapping[Hashable, float]:
        try:
            return self.consequences[intervention_key]  # type: ignore[return-value]
        except KeyError as exc:
            raise KeyError(
                f"hypothesis {self.hypothesis_id!r} has no prediction for "
                f"intervention {intervention_key!r}"
            ) from exc


def _entropy(probabilities: Sequence[float]) -> float:
    return -fsum(value * log2(value) for value in probabilities if value > 0.0)


@dataclass(frozen=True, slots=True)
class ProbeScore:
    """Exact Bayesian value of one costed intervention."""

    intervention: CandidateIntervention
    prior_entropy: float
    expected_posterior_entropy: float
    information_gain: float
    information_gain_per_cost: float
    outcome_probabilities: tuple[tuple[Hashable, float], ...]


@dataclass(frozen=True, slots=True)
class AcquisitionDecision:
    """Resolved concept or an explicit, label-free abstention."""

    token: Hashable
    status: str
    hypothesis_id: Hashable | None
    abstained: bool
    reason: str | None
    posterior: tuple[tuple[Hashable, float], ...]
    entropy: float
    equivalence_classes: tuple[tuple[Hashable, ...], ...]


@dataclass(frozen=True, slots=True)
class BayesianVersionSpace:
    """Immutable posterior over operational hypotheses for one opaque token."""

    token: Hashable
    hypotheses: tuple[HypothesisModel, ...]
    posterior: Mapping[Hashable, float] | None = None

    def __post_init__(self) -> None:
        token = _require_hashable(self.token, "opaque concept token")
        hypotheses = tuple(
            sorted(tuple(self.hypotheses), key=lambda item: _stable_key(item.hypothesis_id))
        )
        if not hypotheses:
            raise ValueError("version space needs at least one hypothesis")
        identifiers = [
            _require_hashable(item.hypothesis_id, "hypothesis_id") for item in hypotheses
        ]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("hypothesis identifiers must be unique")
        if self.posterior is None:
            raw = {item.hypothesis_id: float(item.prior) for item in hypotheses}
        else:
            if set(self.posterior) != set(identifiers):
                raise ValueError("posterior must contain every hypothesis exactly once")
            raw = {identifier: float(self.posterior[identifier]) for identifier in identifiers}
        if any(not isfinite(value) or value < 0.0 for value in raw.values()):
            raise ValueError("posterior masses must be finite and non-negative")
        total = fsum(raw.values())
        if total <= 0.0:
            raise ValueError("posterior needs positive mass")
        normalised = {identifier: raw[identifier] / total for identifier in identifiers}
        object.__setattr__(self, "token", token)
        object.__setattr__(self, "hypotheses", hypotheses)
        object.__setattr__(self, "posterior", MappingProxyType(normalised))

    @property
    def posterior_items(self) -> tuple[tuple[Hashable, float], ...]:
        posterior = self.posterior
        assert posterior is not None
        return tuple((item.hypothesis_id, posterior[item.hypothesis_id]) for item in self.hypotheses)

    @property
    def entropy(self) -> float:
        return _entropy([mass for _, mass in self.posterior_items])

    @property
    def intervention_keys(self) -> tuple[Hashable, ...]:
        first = self.hypotheses[0]
        if isinstance(first, OperationalHypothesis):
            keys = set(first.consequences)
            for hypothesis in self.hypotheses[1:]:
                if isinstance(hypothesis, OperationalHypothesis):
                    keys &= set(hypothesis.consequences)
            return tuple(sorted(keys, key=_stable_key))
        return ()

    def distribution(
        self, hypothesis_id: Hashable, intervention_key: Hashable
    ) -> Mapping[Hashable, float]:
        for hypothesis in self.hypotheses:
            if hypothesis.hypothesis_id == hypothesis_id:
                distribution = hypothesis.distribution(intervention_key)
                return _normalise_distribution(distribution)
        raise KeyError(f"unknown hypothesis {hypothesis_id!r}")

    def predictive_probability(
        self, intervention_key: Hashable, consequence: Hashable
    ) -> float:
        posterior = self.posterior
        assert posterior is not None
        return fsum(
            posterior[hypothesis.hypothesis_id]
            * float(hypothesis.distribution(intervention_key).get(consequence, 0.0))
            for hypothesis in self.hypotheses
        )

    def score(self, intervention: CandidateIntervention) -> ProbeScore:
        posterior = self.posterior
        assert posterior is not None
        distributions: dict[Hashable, Mapping[Hashable, float]] = {}
        outcomes: set[Hashable] = set()
        for hypothesis in self.hypotheses:
            distribution = _normalise_distribution(hypothesis.distribution(intervention.key))
            distributions[hypothesis.hypothesis_id] = distribution
            outcomes.update(distribution)

        outcome_probabilities: list[tuple[Hashable, float]] = []
        expected = 0.0
        for outcome in sorted(outcomes, key=_stable_key):
            marginal = fsum(
                posterior[hypothesis.hypothesis_id]
                * distributions[hypothesis.hypothesis_id].get(outcome, 0.0)
                for hypothesis in self.hypotheses
            )
            if marginal <= 0.0:
                continue
            conditional = [
                posterior[hypothesis.hypothesis_id]
                * distributions[hypothesis.hypothesis_id].get(outcome, 0.0)
                / marginal
                for hypothesis in self.hypotheses
            ]
            expected += marginal * _entropy(conditional)
            outcome_probabilities.append((outcome, marginal))
        gain = max(0.0, self.entropy - expected)
        return ProbeScore(
            intervention=intervention,
            prior_entropy=self.entropy,
            expected_posterior_entropy=expected,
            information_gain=gain,
            information_gain_per_cost=gain / intervention.cost,
            outcome_probabilities=tuple(outcome_probabilities),
        )

    def update(
        self, intervention: CandidateIntervention | Hashable, consequence: Hashable
    ) -> "BayesianVersionSpace":
        key = intervention.key if isinstance(intervention, CandidateIntervention) else intervention
        consequence = _require_hashable(consequence, "observed consequence")
        posterior = self.posterior
        assert posterior is not None
        raw = {
            hypothesis.hypothesis_id: posterior[hypothesis.hypothesis_id]
            * float(hypothesis.distribution(key).get(consequence, 0.0))
            for hypothesis in self.hypotheses
        }
        if fsum(raw.values()) <= 0.0:
            raise InconsistentObservationError(
                f"observation {consequence!r} has zero likelihood for intervention {key!r}"
            )
        return BayesianVersionSpace(self.token, self.hypotheses, raw)

    def equivalence_classes(
        self, intervention_keys: Sequence[Hashable] | None = None
    ) -> tuple[tuple[Hashable, ...], ...]:
        keys = tuple(self.intervention_keys if intervention_keys is None else intervention_keys)
        posterior = self.posterior
        assert posterior is not None
        groups: dict[bytes, list[Hashable]] = {}
        for hypothesis in self.hypotheses:
            if posterior[hypothesis.hypothesis_id] <= _EPSILON:
                continue
            operational = tuple(
                (
                    key,
                    tuple(
                        sorted(
                            hypothesis.distribution(key).items(),
                            key=lambda item: _stable_key(item[0]),
                        )
                    ),
                )
                for key in keys
            )
            groups.setdefault(_canonical_bytes(operational), []).append(hypothesis.hypothesis_id)
        classes = [tuple(sorted(group, key=_stable_key)) for group in groups.values()]
        return tuple(sorted(classes, key=lambda group: _stable_key(group[0])))

    def decision(
        self,
        candidates: Sequence[CandidateIntervention] = (),
        *,
        confidence: float = 1.0,
    ) -> AcquisitionDecision:
        if not 0.5 < confidence <= 1.0:
            raise ValueError("confidence must be in (0.5, 1]")
        posterior = self.posterior
        assert posterior is not None
        supported = [
            (identifier, mass) for identifier, mass in self.posterior_items if mass > _EPSILON
        ]
        available = tuple(candidates)
        if not available:
            available = tuple(CandidateIntervention(key) for key in self.intervention_keys)
        classes = self.equivalence_classes(tuple(candidate.key for candidate in available))
        informative = any(self.score(candidate).information_gain > _EPSILON for candidate in available)
        if len(supported) > 1 and not informative:
            return AcquisitionDecision(
                self.token,
                "AMBIGUOUS",
                None,
                True,
                "operationally-non-identifiable",
                self.posterior_items,
                self.entropy,
                classes,
            )

        ordered = sorted(supported, key=lambda item: (-item[1], _stable_key(item[0])))
        unique_best = len(ordered) == 1 or ordered[0][1] > ordered[1][1] + _EPSILON
        if ordered and unique_best and ordered[0][1] >= confidence - _EPSILON:
            return AcquisitionDecision(
                self.token,
                "RESOLVED",
                ordered[0][0],
                False,
                None,
                self.posterior_items,
                self.entropy,
                classes,
            )
        return AcquisitionDecision(
            self.token,
            "IN_PROGRESS",
            None,
            True,
            "insufficient-evidence",
            self.posterior_items,
            self.entropy,
            classes,
        )


@dataclass(frozen=True, slots=True)
class ProbeBudget:
    """Hard cap shared by active and baseline acquisition policies."""

    max_probes: int
    max_cost: float

    def __post_init__(self) -> None:
        if isinstance(self.max_probes, bool) or not isinstance(self.max_probes, int):
            raise TypeError("max_probes must be an integer")
        if self.max_probes < 0:
            raise ValueError("max_probes cannot be negative")
        cost = float(self.max_cost)
        if not isfinite(cost) or cost < 0.0:
            raise ValueError("max_cost must be finite and non-negative")
        object.__setattr__(self, "max_cost", cost)


def _design_hash(budget: ProbeBudget, candidates: Sequence[CandidateIntervention]) -> str:
    design = {
        "budget": asdict(budget),
        "candidates": [
            {"key": item.key, "cost": item.cost, "max_uses": item.max_uses}
            for item in sorted(candidates, key=lambda item: _stable_key(item.key))
        ],
    }
    return sha256(_canonical_bytes(design)).hexdigest()


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One auditable posterior transition and its exact acquisition score."""

    index: int
    policy: str
    intervention_key: Hashable
    cost: float
    observed_consequence: Hashable
    predictive_probability: float
    entropy_before: float
    expected_entropy_after: float
    information_gain: float
    information_gain_per_cost: float
    entropy_after: float
    posterior: tuple[tuple[Hashable, float], ...]
    remaining_probes: int
    remaining_cost: float
    consistent: bool = True


@dataclass(frozen=True, slots=True)
class AcquisitionLedger:
    """Immutable probe/cost account with a policy-independent design hash."""

    policy: str
    budget: ProbeBudget
    design_hash: str
    entries: tuple[LedgerEntry, ...] = ()

    @classmethod
    def create(
        cls,
        policy: str,
        budget: ProbeBudget,
        candidates: Sequence[CandidateIntervention],
    ) -> "AcquisitionLedger":
        return cls(str(policy), budget, _design_hash(budget, candidates))

    @property
    def probes_used(self) -> int:
        return len(self.entries)

    @property
    def cost_used(self) -> float:
        return fsum(entry.cost for entry in self.entries)

    @property
    def remaining_probes(self) -> int:
        return self.budget.max_probes - self.probes_used

    @property
    def remaining_cost(self) -> float:
        return max(0.0, self.budget.max_cost - self.cost_used)

    @property
    def ledger_hash(self) -> str:
        return sha256(_canonical_bytes(asdict(self))).hexdigest()

    def uses(self, intervention_key: Hashable) -> int:
        return sum(entry.intervention_key == intervention_key for entry in self.entries)

    def can_afford(self, candidate: CandidateIntervention) -> bool:
        return (
            self.remaining_probes > 0
            and candidate.cost <= self.remaining_cost + _EPSILON
            and self.uses(candidate.key) < candidate.max_uses
        )

    def append(self, entry: LedgerEntry) -> "AcquisitionLedger":
        if entry.index != len(self.entries):
            raise ValueError("ledger entry index is not consecutive")
        if entry.policy != self.policy:
            raise ValueError("ledger entry policy does not match ledger")
        if self.remaining_probes < 1 or entry.cost > self.remaining_cost + _EPSILON:
            raise ValueError("ledger entry exceeds budget")
        return AcquisitionLedger(self.policy, self.budget, self.design_hash, self.entries + (entry,))


@dataclass(frozen=True, slots=True)
class InformationGainPolicy:
    """Choose maximal exact expected entropy reduction per unit cost."""

    minimum_gain: float = _EPSILON
    name: str = "information-gain"

    def select(
        self,
        version_space: BayesianVersionSpace,
        candidates: Sequence[CandidateIntervention],
        ledger: AcquisitionLedger,
        budget: ProbeBudget,
    ) -> ProbeScore | None:
        del ledger, budget
        scores = [version_space.score(candidate) for candidate in candidates]
        scores = [score for score in scores if score.information_gain > self.minimum_gain]
        if not scores:
            return None
        return min(
            scores,
            key=lambda score: (
                -score.information_gain_per_cost,
                -score.information_gain,
                score.intervention.cost,
                _stable_key(score.intervention.key),
            ),
        )


@dataclass(frozen=True, slots=True)
class RandomPolicy:
    """Seeded random-permutation baseline without process-global RNG state."""

    seed: int = 0
    name: str = "random"

    def select(
        self,
        version_space: BayesianVersionSpace,
        candidates: Sequence[CandidateIntervention],
        ledger: AcquisitionLedger,
        budget: ProbeBudget,
    ) -> ProbeScore | None:
        del budget
        if not candidates:
            return None
        ordered = tuple(sorted(candidates, key=lambda item: _stable_key(item.key)))
        material = (self.seed, ledger.design_hash, ledger.probes_used)
        index = int.from_bytes(sha256(_canonical_bytes(material)).digest()[:8], "big") % len(
            ordered
        )
        return version_space.score(ordered[index])


@dataclass(frozen=True, slots=True)
class PassivePolicy:
    """Fixed-order, non-adaptive baseline."""

    order: tuple[Hashable, ...] = ()
    name: str = "passive"

    def __post_init__(self) -> None:
        order = tuple(_require_hashable(item, "passive intervention key") for item in self.order)
        object.__setattr__(self, "order", order)

    def select(
        self,
        version_space: BayesianVersionSpace,
        candidates: Sequence[CandidateIntervention],
        ledger: AcquisitionLedger,
        budget: ProbeBudget,
    ) -> ProbeScore | None:
        del budget
        if not candidates:
            return None
        by_key = {candidate.key: candidate for candidate in candidates}
        for key in self.order:
            candidate = by_key.get(key)
            if candidate is not None:
                return version_space.score(candidate)
        candidate = min(candidates, key=lambda item: _stable_key(item.key))
        return version_space.score(candidate)


@dataclass(frozen=True, slots=True)
class AcquisitionRun:
    """Final posterior, decision, and complete resource ledger."""

    version_space: BayesianVersionSpace
    decision: AcquisitionDecision
    ledger: AcquisitionLedger

    @property
    def probes_used(self) -> int:
        return self.ledger.probes_used

    @property
    def cost_used(self) -> float:
        return self.ledger.cost_used


def _validate_candidates(
    version_space: BayesianVersionSpace, candidates: Sequence[CandidateIntervention]
) -> tuple[CandidateIntervention, ...]:
    values = tuple(candidates)
    if not values:
        raise ValueError("at least one candidate intervention is required")
    keys = [candidate.key for candidate in values]
    if len(set(keys)) != len(keys):
        raise ValueError("candidate intervention keys must be unique")
    for hypothesis in version_space.hypotheses:
        for candidate in values:
            _normalise_distribution(hypothesis.distribution(candidate.key))
    return values


def _terminal_decision(
    state: BayesianVersionSpace,
    status: str,
    reason: str,
) -> AcquisitionDecision:
    return AcquisitionDecision(
        state.token,
        status,
        None,
        True,
        reason,
        state.posterior_items,
        state.entropy,
        state.equivalence_classes(),
    )


def run_acquisition(
    version_space: BayesianVersionSpace,
    candidates: Sequence[CandidateIntervention],
    observe: InterventionObserver,
    budget: ProbeBudget,
    *,
    policy: AcquisitionPolicy | None = None,
    encoder: ConsequenceEncoder = learner_visible_consequence,
    confidence: float = 1.0,
) -> AcquisitionRun:
    """Run one deterministic, budgeted acquisition policy to termination."""

    candidates = _validate_candidates(version_space, candidates)
    selected_policy: AcquisitionPolicy = policy or InformationGainPolicy()
    ledger = AcquisitionLedger.create(selected_policy.name, budget, candidates)
    state = version_space

    while True:
        current = state.decision(candidates, confidence=confidence)
        if current.status in {"RESOLVED", "AMBIGUOUS"}:
            return AcquisitionRun(state, current, ledger)
        eligible = tuple(candidate for candidate in candidates if ledger.can_afford(candidate))
        if not eligible:
            reason = "probe-budget-exhausted" if ledger.remaining_probes == 0 else "cost-budget-exhausted"
            return AcquisitionRun(
                state,
                _terminal_decision(state, "BUDGET_EXHAUSTED", reason),
                ledger,
            )
        score = selected_policy.select(state, eligible, ledger, budget)
        if score is None or score.intervention not in eligible:
            return AcquisitionRun(
                state,
                _terminal_decision(state, "AMBIGUOUS", "policy-found-no-informative-probe"),
                ledger,
            )

        raw_observation = observe(score.intervention.payload)
        observed = _require_hashable(encoder(raw_observation), "encoded consequence")
        predictive = state.predictive_probability(score.intervention.key, observed)
        consistent = predictive > 0.0
        next_state = state
        if consistent:
            next_state = state.update(score.intervention, observed)

        remaining_probes = ledger.remaining_probes - 1
        remaining_cost = max(0.0, ledger.remaining_cost - score.intervention.cost)
        entry = LedgerEntry(
            index=ledger.probes_used,
            policy=selected_policy.name,
            intervention_key=score.intervention.key,
            cost=score.intervention.cost,
            observed_consequence=observed,
            predictive_probability=predictive,
            entropy_before=state.entropy,
            expected_entropy_after=score.expected_posterior_entropy,
            information_gain=score.information_gain,
            information_gain_per_cost=score.information_gain_per_cost,
            entropy_after=next_state.entropy,
            posterior=next_state.posterior_items,
            remaining_probes=remaining_probes,
            remaining_cost=remaining_cost,
            consistent=consistent,
        )
        ledger = ledger.append(entry)
        if not consistent:
            return AcquisitionRun(
                state,
                _terminal_decision(
                    state, "MODEL_MISSPECIFIED", "zero-likelihood-observation"
                ),
                ledger,
            )
        state = next_state


def _observer_for_policy(
    observers: InterventionObserver | Mapping[str, InterventionObserver], policy_name: str
) -> InterventionObserver:
    if isinstance(observers, Mapping):
        try:
            return observers[policy_name]
        except KeyError as exc:
            raise KeyError(f"missing observer for policy {policy_name!r}") from exc
    return observers


def run_policy_baselines(
    version_space: BayesianVersionSpace,
    candidates: Sequence[CandidateIntervention],
    observers: InterventionObserver | Mapping[str, InterventionObserver],
    budget: ProbeBudget,
    *,
    random_seed: int = 0,
    passive_order: Sequence[Hashable] = (),
    encoder: ConsequenceEncoder = learner_visible_consequence,
    confidence: float = 1.0,
) -> Mapping[str, AcquisitionRun]:
    """Run active, seeded-random and passive policies on one declared design.

    A pure observer may be shared.  Stateful environments should be supplied
    as a mapping with one freshly reset observer per policy name.
    """

    candidates = _validate_candidates(version_space, candidates)
    policies: tuple[AcquisitionPolicy, ...] = (
        InformationGainPolicy(),
        RandomPolicy(random_seed),
        PassivePolicy(tuple(passive_order)),
    )
    runs: dict[str, AcquisitionRun] = {}
    for selected_policy in policies:
        runs[selected_policy.name] = run_acquisition(
            version_space,
            candidates,
            _observer_for_policy(observers, selected_policy.name),
            budget,
            policy=selected_policy,
            encoder=encoder,
            confidence=confidence,
        )
    design_hashes = {run.ledger.design_hash for run in runs.values()}
    if len(design_hashes) != 1:
        raise RuntimeError("baseline policies did not execute the same candidate/budget design")
    return MappingProxyType(runs)


__all__ = [
    "AcquisitionDecision",
    "AcquisitionLedger",
    "AcquisitionPolicy",
    "AcquisitionRun",
    "BayesianVersionSpace",
    "CandidateIntervention",
    "ConsequenceEncoder",
    "HypothesisModel",
    "InconsistentObservationError",
    "InformationGainPolicy",
    "InterventionObserver",
    "LearnerVisibleTransition",
    "LedgerEntry",
    "ObservableConsequence",
    "OperationalHypothesis",
    "PassivePolicy",
    "ProbeBudget",
    "ProbeScore",
    "RandomPolicy",
    "learner_visible_consequence",
    "run_acquisition",
    "run_policy_baselines",
]
