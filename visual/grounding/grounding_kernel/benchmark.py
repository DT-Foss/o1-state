"""GroundZero-v0: a deterministic, sealed operational-grounding benchmark.

The evaluator owns semantic state and codebook decoders.  Learners receive
only raw observations, opaque action/outcome codes, opaque symbol codes and
task feedback.  The benchmark never touches the network, pretrained models,
the FERTIG text graph, or the filesystem.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable
import argparse
import json

import numpy as np

from .certificates import (
    REQUIRED_GROUND_ZERO_AXES,
    CertificateAxis,
    CertificateScope,
    GroundingCertificate,
    MetricBound,
    binary_axis,
    manifest_hash,
)
from .composition import And, Atom, EvaluationResult, TruthValue, evaluate
from .contracts import (
    Action,
    ActionKind,
    Observation,
    OutcomeKind,
    PredicateKind,
    Transition,
)
from .microworld import EvaluatorHarness, WorldConfig
from .protocol import audit_agent_boundary


BENCHMARK_ID = "ground-zero-v0"
IDENTIFIABLE_PREDICATES = (
    PredicateKind.MOVABLE,
    PredicateKind.LIFTABLE,
    PredicateKind.MAGNETIC,
    PredicateKind.FITS_SLOT_A,
    PredicateKind.FITS_SLOT_B,
    PredicateKind.SWITCHABLE,
)


class BenchmarkProtocolError(RuntimeError):
    """Raised when an injected environment or learner violates the contract."""


@dataclass(frozen=True, slots=True)
class GroundingEvidence:
    """A learner-visible, task-labelled sensorimotor transition.

    ``token`` is an opaque integer.  No semantic enum, object identifier,
    world seed, codebook, oracle object or latent property crosses this type.
    """

    token: int
    before: Observation
    action: Action
    after: Observation
    outcome_code: int
    task_feedback: bool | None = None

    @classmethod
    def from_transition(
        cls,
        token: int,
        transition: Transition,
        *,
        task_feedback: bool | None = None,
    ) -> "GroundingEvidence":
        return cls(
            int(token),
            transition.before,
            transition.action,
            transition.after,
            int(transition.outcome_code),
            None if task_feedback is None else bool(task_feedback),
        )

    @property
    def transition(self) -> Transition:
        return Transition(self.before, self.action, self.after, self.outcome_code)

    def digest(self) -> str:
        return manifest_hash(
            {
                "token": self.token,
                "before": self.before.digest(),
                "action": asdict(self.action),
                "after": self.after.digest(),
                "outcome_code": self.outcome_code,
                "task_feedback": self.task_feedback,
            }
        )


@runtime_checkable
class GroundZeroLearner(Protocol):
    """Minimal injected learner API; structural duck typing is supported."""

    def fit(self, experiences: Sequence[GroundingEvidence]) -> Any: ...

    def predict_token(
        self,
        evidence_or_before: Transition | Observation,
        action: Action | None = None,
        after: Observation | None = None,
        candidates: Sequence[Hashable] | None = None,
    ) -> Hashable | None: ...


EnvironmentFactory = Callable[[int, WorldConfig], Any]
LearnerFactory = Callable[[int], Any]


@dataclass(frozen=True, slots=True)
class PairedTokenCase:
    transition: Transition
    expected: int
    remapped_expected: int


@dataclass(frozen=True, slots=True)
class TokenCase:
    transition: Transition
    expected: int


@dataclass(frozen=True, slots=True)
class CompositionCase:
    definition: And
    atom_transitions: tuple[tuple[int, Transition], ...]
    expected: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "atom_transitions", tuple(self.atom_transitions))

    @property
    def evidence(self) -> Mapping[int, Transition]:
        return MappingProxyType(dict(self.atom_transitions))


@dataclass(frozen=True, slots=True)
class AbstentionCase:
    transition: Transition
    negative_control_token: int
    invariant_verified: bool


@dataclass(frozen=True, slots=True)
class GroundZeroDataset:
    """Evaluator-owned sealed split.  Only individual public records are passed on."""

    train: tuple[GroundingEvidence, ...]
    train_remapped: tuple[GroundingEvidence, ...]
    train_intervention_shuffled: tuple[GroundingEvidence, ...]
    token_remapping: tuple[tuple[int, int], ...]
    identifiable_tokens: tuple[int, ...]
    remapped_identifiable_tokens: tuple[int, ...]
    token_cases: tuple[PairedTokenCase, ...]
    nuisance_cases: tuple[TokenCase, ...]
    intervention_cases: tuple[TokenCase, ...]
    composition_cases: tuple[CompositionCase, ...]
    abstention_cases: tuple[AbstentionCase, ...]
    public_environment_manifest: Mapping[str, Any]
    split_manifest: Mapping[str, Any]

    def __post_init__(self) -> None:
        tuple_fields = (
            "train",
            "train_remapped",
            "train_intervention_shuffled",
            "token_remapping",
            "identifiable_tokens",
            "remapped_identifiable_tokens",
            "token_cases",
            "nuisance_cases",
            "intervention_cases",
            "composition_cases",
            "abstention_cases",
        )
        for name in tuple_fields:
            object.__setattr__(self, name, tuple(getattr(self, name)))
        object.__setattr__(
            self,
            "public_environment_manifest",
            MappingProxyType(dict(self.public_environment_manifest)),
        )
        object.__setattr__(self, "split_manifest", MappingProxyType(dict(self.split_manifest)))


@dataclass(frozen=True, slots=True)
class BenchmarkThresholds:
    """Pre-registered GroundZero-v0 gates."""

    token_remapping_equivariance: float = 0.85
    nuisance_transfer: float = 0.80
    intervention_necessity: float = 0.65
    unseen_composition: float = 0.80
    honest_abstention: float = 0.85
    answer_coverage: float = 0.80
    confidence: float = 0.95

    def __post_init__(self) -> None:
        for name in REQUIRED_GROUND_ZERO_AXES:
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} threshold must lie in (0, 1]")
            object.__setattr__(self, name, value)
        if not 0.0 < self.answer_coverage <= 1.0:
            raise ValueError("answer_coverage must lie in (0, 1]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")


@dataclass(frozen=True, slots=True)
class TrialRecord:
    axis: str
    trial_id: int
    passed: bool
    answered: bool
    expected: str
    prediction: str
    evidence_hash: str
    detail: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "trial_id": self.trial_id,
            "passed": self.passed,
            "answered": self.answered,
            "expected": self.expected,
            "prediction": self.prediction,
            "evidence_hash": self.evidence_hash,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class ControlResult:
    """Negative baseline that must not satisfy the corresponding grounding gate."""

    name: str
    performance: MetricBound
    coverage: MetricBound
    grounding_threshold: float
    coverage_threshold: float
    description: str

    @property
    def grounding_gate_passed(self) -> bool:
        return (
            self.performance.lower_bound >= self.grounding_threshold
            and self.coverage.lower_bound >= self.coverage_threshold
        )

    @property
    def rejected_as_grounder(self) -> bool:
        return not self.grounding_gate_passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "performance": self.performance.to_dict(),
            "coverage": self.coverage.to_dict(),
            "grounding_threshold": self.grounding_threshold,
            "coverage_threshold": self.coverage_threshold,
            "grounding_gate_passed": self.grounding_gate_passed,
            "rejected_as_grounder": self.rejected_as_grounder,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLedger:
    benchmark_id: str
    seed: int
    episodes: int
    dataset_hash: str
    trials: tuple[TrialRecord, ...]
    controls: tuple[ControlResult, ...] = ()

    @property
    def axis_counts(self) -> Mapping[str, Mapping[str, int]]:
        counts: dict[str, dict[str, int]] = {}
        for axis in REQUIRED_GROUND_ZERO_AXES:
            records = [record for record in self.trials if record.axis == axis]
            counts[axis] = {
                "trials": len(records),
                "passed": sum(record.passed for record in records),
                "answered": sum(record.answered for record in records),
            }
        return MappingProxyType(
            {name: MappingProxyType(values) for name, values in counts.items()}
        )

    @property
    def ledger_hash(self) -> str:
        return manifest_hash(
            {
                "benchmark_id": self.benchmark_id,
                "seed": self.seed,
                "episodes": self.episodes,
                "dataset_hash": self.dataset_hash,
                "trials": [trial.to_dict() for trial in self.trials],
                "controls": [control.to_dict() for control in self.controls],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,
            "episodes": self.episodes,
            "dataset_hash": self.dataset_hash,
            "axis_counts": {name: dict(values) for name, values in self.axis_counts.items()},
            "trials": [trial.to_dict() for trial in self.trials],
            "controls": [control.to_dict() for control in self.controls],
            "ledger_hash": self.ledger_hash,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    certificate: GroundingCertificate
    ledger: EvidenceLedger

    @property
    def passed(self) -> bool:
        return self.certificate.passed and all(
            control.rejected_as_grounder for control in self.ledger.controls
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": BENCHMARK_ID,
            "passed": self.passed,
            "metric_axes_passed": self.certificate.passed,
            "negative_controls_valid": all(
                control.rejected_as_grounder for control in self.ledger.controls
            ),
            "certificate": self.certificate.to_dict(),
            "ledger": self.ledger.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class _CanonicalInterface:
    action_codes: Mapping[ActionKind, int]
    outcome_codes: Mapping[OutcomeKind, int]
    symbol_codes: Mapping[PredicateKind, int]


def _derived_seed(seed: int, namespace: str, index: int = 0) -> int:
    payload = f"{BENCHMARK_ID}|{seed}|{namespace}|{index}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:4], "big")


def _seeded_derangement(values: Sequence[int], seed: int) -> tuple[int, ...]:
    """Return a seeded single-cycle permutation with no fixed points."""

    result = [int(value) for value in values]
    if len(result) < 2:
        raise ValueError("a token remapping requires at least two values")
    rng = np.random.default_rng(np.random.SeedSequence((_derived_seed(seed, "lexicon"), 29)))
    # Sattolo's algorithm produces one cycle and therefore a derangement.
    for index in range(len(result) - 1, 0, -1):
        swap = int(rng.integers(0, index))
        result[index], result[swap] = result[swap], result[index]
    return tuple(result)


def _factory_harness(
    factory: EnvironmentFactory,
    seed: int,
    config: WorldConfig,
) -> Any:
    harness = factory(seed, config)
    if not hasattr(harness, "agent") or not hasattr(harness, "oracle"):
        raise BenchmarkProtocolError("environment_factory must return an evaluator harness")
    leaks = audit_agent_boundary(harness.agent)
    if leaks:
        raise BenchmarkProtocolError(f"agent environment leaks privileged names: {leaks!r}")
    return harness


def _canonical_interface(harness: Any, seed: int) -> _CanonicalInterface:
    action_codes: dict[ActionKind, int] = {}
    for code in harness.agent.action_codes:
        action_codes[harness.oracle.decode_action(code)] = int(code)
    symbol_codes = {
        predicate: int(harness.oracle.encode_symbol(predicate)) for predicate in PredicateKind
    }
    rng = np.random.default_rng(np.random.SeedSequence((_derived_seed(seed, "outcomes"), 17)))
    # Use NumPy's integer-domain sampling overload.  Materialising this range
    # would allocate roughly 7.2 GiB for eleven opaque outcome codes.
    opaque = rng.choice(899_999_999, size=len(OutcomeKind), replace=False)
    opaque = opaque.astype(np.int64, copy=False) + 1_100_000_000
    outcome_codes = {
        kind: int(code) for kind, code in zip(OutcomeKind, opaque.tolist(), strict=True)
    }
    return _CanonicalInterface(
        MappingProxyType(action_codes),
        MappingProxyType(outcome_codes),
        MappingProxyType(symbol_codes),
    )


def _action_spec(predicate: PredicateKind) -> tuple[ActionKind, tuple[tuple[int, int], ...]]:
    if predicate is PredicateKind.MOVABLE:
        return ActionKind.PUSH, ((1, 0), (-1, 0), (0, 1), (0, -1))
    if predicate is PredicateKind.LIFTABLE:
        return ActionKind.LIFT, ((0, 0),)
    if predicate is PredicateKind.MAGNETIC:
        return ActionKind.MAGNET, ((0, 0),)
    if predicate is PredicateKind.FITS_SLOT_A:
        return ActionKind.INSERT, ((0, -1),)
    if predicate is PredicateKind.FITS_SLOT_B:
        return ActionKind.INSERT, ((0, 1),)
    if predicate is PredicateKind.SWITCHABLE:
        return ActionKind.TOGGLE, ((0, 0),)
    raise ValueError(f"predicate is not intervention-identifiable: {predicate.value}")


_SUCCESS_OUTCOME = {
    PredicateKind.MOVABLE: OutcomeKind.MOVED,
    PredicateKind.LIFTABLE: OutcomeKind.LIFTED,
    PredicateKind.MAGNETIC: OutcomeKind.ATTRACTED,
    PredicateKind.FITS_SLOT_A: OutcomeKind.INSERTED,
    PredicateKind.FITS_SLOT_B: OutcomeKind.INSERTED,
    PredicateKind.SWITCHABLE: OutcomeKind.ACTIVATED,
}


def _normalise_transition(
    transition: Transition,
    source_oracle: Any,
    canonical: _CanonicalInterface,
) -> Transition:
    action_kind = source_oracle.decode_action(transition.action.code)
    outcome_kind = source_oracle.decode_outcome(transition.outcome_code)
    action = Action(
        canonical.action_codes[action_kind],
        transition.action.target,
        transition.action.vector,
    )
    return Transition(
        transition.before,
        action,
        transition.after,
        canonical.outcome_codes[outcome_kind],
    )


def _probe(
    factory: EnvironmentFactory,
    world_seed: int,
    config: WorldConfig,
    object_id: int,
    predicate: PredicateKind,
    canonical: _CanonicalInterface,
) -> tuple[Transition, bool, bool]:
    """Return public transition, oracle truth and whether the expected effect occurred."""

    action_kind, vectors = _action_spec(predicate)
    last: tuple[Transition, bool, bool] | None = None
    for vector in vectors:
        harness = _factory_harness(factory, world_seed, config)
        truth = bool(harness.oracle.predicate(object_id, predicate))
        raw_action_code = next(
            code
            for code in harness.agent.action_codes
            if harness.oracle.decode_action(code) is action_kind
        )
        target = harness.oracle.object_center(object_id)
        raw = harness.agent.step(Action(raw_action_code, target, vector))
        outcome = harness.oracle.decode_outcome(raw.outcome_code)
        normalised = _normalise_transition(raw, harness.oracle, canonical)
        effective = outcome is _SUCCESS_OUTCOME[predicate]
        last = (normalised, truth, effective)
        if not truth or effective:
            return last
    assert last is not None
    return last


def _world_objects(
    factory: EnvironmentFactory,
    world_seed: int,
    config: WorldConfig,
) -> tuple[tuple[int, Mapping[PredicateKind, bool]], ...]:
    harness = _factory_harness(factory, world_seed, config)
    return tuple((state.object_id, state.predicates) for state in harness.oracle.snapshot().objects)


def _collect_training(
    *,
    seed: int,
    per_predicate: int,
    factory: EnvironmentFactory,
    config: WorldConfig,
    canonical: _CanonicalInterface,
) -> tuple[GroundingEvidence, ...]:
    buckets: dict[PredicateKind, list[GroundingEvidence]] = {
        predicate: [] for predicate in IDENTIFIABLE_PREDICATES
    }
    for world_index in range(2_000):
        if all(len(bucket) >= per_predicate for bucket in buckets.values()):
            break
        world_seed = _derived_seed(seed, "train-world", world_index)
        for object_id, predicates in _world_objects(factory, world_seed, config):
            for predicate in IDENTIFIABLE_PREDICATES:
                if len(buckets[predicate]) >= per_predicate or not predicates[predicate]:
                    continue
                transition, truth, effective = _probe(
                    factory, world_seed, config, object_id, predicate, canonical
                )
                if truth and effective:
                    buckets[predicate].append(
                        GroundingEvidence.from_transition(
                            canonical.symbol_codes[predicate], transition
                        )
                    )
    missing = [predicate.value for predicate, values in buckets.items() if len(values) < per_predicate]
    if missing:
        raise BenchmarkProtocolError(f"could not generate positive training evidence: {missing}")
    # Interleave categories to prevent an order-only learner from exploiting blocks.
    records = [
        buckets[predicate][index]
        for index in range(per_predicate)
        for predicate in IDENTIFIABLE_PREDICATES
    ]
    return tuple(records)


def _shuffled_intervention_training(
    records: Sequence[GroundingEvidence],
    token_order: Sequence[int],
) -> tuple[GroundingEvidence, ...]:
    """Destroy token↔action/outcome pairing while preserving every marginal."""

    grouped: dict[int, list[GroundingEvidence]] = defaultdict(list)
    for record in records:
        grouped[record.token].append(record)
    if any(not grouped[token] for token in token_order):
        raise BenchmarkProtocolError("every identifiable token needs training evidence")
    shifted = {
        token: token_order[(index + 1) % len(token_order)]
        for index, token in enumerate(token_order)
    }
    positions = Counter()
    ablated: list[GroundingEvidence] = []
    for record in records:
        source_token = shifted[record.token]
        position = positions[source_token] % len(grouped[source_token])
        replacement = grouped[source_token][position]
        positions[source_token] += 1
        ablated.append(
            GroundingEvidence.from_transition(
                record.token,
                replacement.transition,
                task_feedback=record.task_feedback,
            )
        )
    return tuple(ablated)


def _collect_positive_cases(
    *,
    seed: int,
    count: int,
    factory: EnvironmentFactory,
    config: WorldConfig,
    canonical: _CanonicalInterface,
) -> tuple[TokenCase, ...]:
    cases: list[TokenCase] = []
    for world_index in range(4_000):
        if len(cases) >= count:
            break
        world_seed = _derived_seed(seed, "sealed-test-world", world_index)
        wanted = IDENTIFIABLE_PREDICATES[len(cases) % len(IDENTIFIABLE_PREDICATES)]
        for object_id, predicates in _world_objects(factory, world_seed, config):
            if not predicates[wanted]:
                continue
            transition, truth, effective = _probe(
                factory, world_seed, config, object_id, wanted, canonical
            )
            if truth and effective:
                cases.append(TokenCase(transition, canonical.symbol_codes[wanted]))
                break
    if len(cases) < count:
        raise BenchmarkProtocolError("could not generate enough sealed positive cases")
    return tuple(cases)


def _collect_composition_cases(
    *,
    seed: int,
    count: int,
    factory: EnvironmentFactory,
    config: WorldConfig,
    canonical: _CanonicalInterface,
) -> tuple[CompositionCase, ...]:
    pairs = (
        (PredicateKind.MOVABLE, PredicateKind.LIFTABLE),
        (PredicateKind.MAGNETIC, PredicateKind.SWITCHABLE),
        (PredicateKind.FITS_SLOT_A, PredicateKind.MOVABLE),
        (PredicateKind.FITS_SLOT_B, PredicateKind.LIFTABLE),
    )
    positive: list[CompositionCase] = []
    negative: list[CompositionCase] = []
    target_each = (count + 1) // 2
    for world_index in range(6_000):
        if len(positive) >= target_each and len(negative) >= count // 2:
            break
        world_seed = _derived_seed(seed, "composition-world", world_index)
        pair = pairs[world_index % len(pairs)]
        for object_id, predicates in _world_objects(factory, world_seed, config):
            expected = bool(predicates[pair[0]] and predicates[pair[1]])
            pool = positive if expected else negative
            limit = target_each if expected else count // 2
            if len(pool) >= limit:
                continue
            atoms: list[tuple[int, Transition]] = []
            usable = True
            for predicate in pair:
                transition, truth, effective = _probe(
                    factory, world_seed, config, object_id, predicate, canonical
                )
                if truth and not effective:
                    usable = False
                    break
                atoms.append((canonical.symbol_codes[predicate], transition))
            if not usable:
                continue
            definition = And(*(Atom(token) for token, _transition in atoms))
            pool.append(CompositionCase(definition, tuple(atoms), expected))
            break
    if len(positive) < target_each or len(negative) < count // 2:
        raise BenchmarkProtocolError("could not generate balanced composition cases")
    cases: list[CompositionCase] = []
    for index in range(max(len(positive), len(negative))):
        if index < len(positive):
            cases.append(positive[index])
        if index < len(negative):
            cases.append(negative[index])
    return tuple(cases[:count])


def _collect_abstention_cases(
    *,
    seed: int,
    count: int,
    factory: EnvironmentFactory,
    config: WorldConfig,
    canonical: _CanonicalInterface,
) -> tuple[AbstentionCase, ...]:
    check = _factory_harness(factory, _derived_seed(seed, "negative-control-proof"), config)
    invariant = bool(check.oracle.negative_control_invariant())
    if not invariant:
        raise BenchmarkProtocolError("negative-control predicate is observable in this environment")
    cases: list[AbstentionCase] = []
    predicate = PredicateKind.MOVABLE
    for world_index in range(4_000):
        if len(cases) >= count:
            break
        world_seed = _derived_seed(seed, "abstention-world", world_index)
        for object_id, predicates in _world_objects(factory, world_seed, config):
            if predicates[predicate]:
                continue
            transition, truth, _effective = _probe(
                factory, world_seed, config, object_id, predicate, canonical
            )
            if not truth:
                cases.append(
                    AbstentionCase(
                        transition,
                        canonical.symbol_codes[PredicateKind.NEGATIVE_CONTROL],
                        invariant,
                    )
                )
                break
    if len(cases) < count:
        raise BenchmarkProtocolError("could not generate enough negative-control probes")
    return tuple(cases)


def build_ground_zero_dataset(
    *,
    seed: int = 0,
    episodes: int = 96,
    environment_factory: EnvironmentFactory | None = None,
    world_config: WorldConfig | None = None,
) -> GroundZeroDataset:
    """Build a deterministic evaluator-owned dataset with sealed seed domains."""

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(episodes, bool) or not isinstance(episodes, (int, np.integer)):
        raise TypeError("episodes must be an integer")
    episodes = int(episodes)
    if episodes < 8:
        raise ValueError("episodes must be at least 8")
    selected_factory = environment_factory or EvaluatorHarness
    config = world_config or WorldConfig(object_count=8, max_steps=8)
    canonical_harness = _factory_harness(selected_factory, int(seed), config)
    canonical = _canonical_interface(canonical_harness, int(seed))
    token_order = tuple(canonical.symbol_codes[p] for p in IDENTIFIABLE_PREDICATES)
    token_values = tuple(sorted(canonical_harness.agent.symbol_codes))
    remapped_values = _seeded_derangement(token_values, int(seed))
    remapping = dict(zip(token_values, remapped_values, strict=True))
    remapped_tokens = tuple(remapping[token] for token in token_order)

    train = _collect_training(
        seed=int(seed),
        per_predicate=max(12, (episodes + len(token_order) - 1) // len(token_order)),
        factory=selected_factory,
        config=config,
        canonical=canonical,
    )
    train_remapped = tuple(
        GroundingEvidence(
            remapping[record.token],
            record.before,
            record.action,
            record.after,
            record.outcome_code,
            record.task_feedback,
        )
        for record in train
    )
    shuffled = _shuffled_intervention_training(train, token_order)
    positive_cases = _collect_positive_cases(
        seed=int(seed),
        count=episodes,
        factory=selected_factory,
        config=config,
        canonical=canonical,
    )
    paired_cases = tuple(
        PairedTokenCase(
            case.transition,
            case.expected,
            remapping[case.expected],
        )
        for case in positive_cases
    )
    composition_cases = _collect_composition_cases(
        seed=int(seed),
        count=episodes,
        factory=selected_factory,
        config=config,
        canonical=canonical,
    )
    abstention_cases = _collect_abstention_cases(
        seed=int(seed),
        count=episodes,
        factory=selected_factory,
        config=config,
        canonical=canonical,
    )

    public_manifest = {
        "agent": asdict(canonical_harness.agent.manifest),
        "world_config": asdict(config),
        "environment": type(canonical_harness.agent).__qualname__,
        "boundary_audit": list(audit_agent_boundary(canonical_harness.agent)),
    }
    split_manifest = {
        "train_namespace": "train-world",
        "test_namespaces": [
            "sealed-test-world",
            "composition-world",
            "abstention-world",
        ],
        "episodes": episodes,
        "train_evidence": [record.digest() for record in train],
        "test_evidence": [case.transition.before.digest() for case in positive_cases],
        "composition_evidence": [
            [transition.before.digest() for _token, transition in case.atom_transitions]
            for case in composition_cases
        ],
        "negative_control_invariant": True,
    }
    return GroundZeroDataset(
        train=train,
        train_remapped=train_remapped,
        train_intervention_shuffled=shuffled,
        token_remapping=tuple(sorted(remapping.items())),
        identifiable_tokens=token_order,
        remapped_identifiable_tokens=remapped_tokens,
        token_cases=paired_cases,
        nuisance_cases=positive_cases,
        intervention_cases=positive_cases,
        composition_cases=composition_cases,
        abstention_cases=abstention_cases,
        public_environment_manifest=public_manifest,
        split_manifest=split_manifest,
    )


def _transition_signature(transition: Any) -> tuple[int, tuple[int, int], int, bool]:
    before = transition.before
    after = transition.after
    return (
        int(transition.action.code),
        tuple(transition.action.vector),
        int(transition.outcome_code),
        not np.array_equal(before.pixels, after.pixels),
    )


class _ReferenceTransitionLearner:
    """Small count-based reference; validates the protocol, not visual intelligence."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = int(seed)
        self._exact: dict[int, Counter[tuple[int, tuple[int, int], int, bool]]] = {}
        self._coarse: dict[int, Counter[tuple[int, int, bool]]] = {}
        self._actions: dict[int, set[int]] = {}

    @property
    def manifest(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "name": "deterministic-counting-reference",
                "version": 1,
                "seed": self._seed,
                "uses": "opaque action/outcome/effect signatures only",
            }
        )

    def fit(self, experiences: Sequence[GroundingEvidence]) -> "_ReferenceTransitionLearner":
        exact: dict[int, Counter[tuple[int, tuple[int, int], int, bool]]] = defaultdict(Counter)
        coarse: dict[int, Counter[tuple[int, int, bool]]] = defaultdict(Counter)
        actions: dict[int, set[int]] = defaultdict(set)
        for record in experiences:
            if record.task_feedback is False:
                continue
            signature = _transition_signature(record)
            exact[int(record.token)][signature] += 1
            coarse[int(record.token)][(signature[0], signature[2], signature[3])] += 1
            actions[int(record.token)].add(signature[0])
        self._exact = dict(exact)
        self._coarse = dict(coarse)
        self._actions = dict(actions)
        return self

    def _scores(self, transition: Any, candidates: Sequence[Hashable]) -> dict[Hashable, float]:
        signature = _transition_signature(transition)
        coarse = (signature[0], signature[2], signature[3])
        scores: dict[Hashable, float] = {}
        for token in candidates:
            if not isinstance(token, (int, np.integer)):
                scores[token] = 0.0
                continue
            value = int(token)
            scores[token] = 2.0 * self._exact.get(value, Counter())[signature]
            scores[token] += self._coarse.get(value, Counter())[coarse]
        return scores

    def predict_token(
        self,
        evidence_or_before: Any,
        action: Action | None = None,
        after: Observation | None = None,
        candidates: Sequence[Hashable] | None = None,
    ) -> Hashable | None:
        if action is not None or after is not None:
            if action is None or after is None or not isinstance(evidence_or_before, Observation):
                raise TypeError("before/action/after must be supplied together")
            raise TypeError("reference learner requires an outcome-bearing transition")
        if candidates is None:
            candidates = tuple(self._exact)
        scores = self._scores(evidence_or_before, tuple(candidates))
        if not scores:
            return None
        best_score = max(scores.values())
        winners = [token for token, score in scores.items() if score == best_score and score > 0.0]
        return winners[0] if len(winners) == 1 else None

    def supports_token(self, transition: Any, token: int) -> bool | None:
        prediction = self.predict_token(transition, candidates=tuple(self._exact))
        if prediction == token:
            return True
        if prediction is not None:
            return False
        action_code = int(transition.action.code)
        if action_code in self._actions.get(int(token), set()):
            # A diagnostic action for this token was executed, but none of its
            # learned positive effect signatures occurred.
            return False
        return None


class _StaticPixelsOnlyBaseline:
    """Deliberately action-blind nearest-centroid control.

    It receives exactly the same records but discards actions, outcomes and
    post-intervention frames.  Hidden causal properties are independently
    randomized across the sealed worlds, so this control must fail.
    """

    def __init__(self, _seed: int = 0) -> None:
        self._centroids: dict[int, np.ndarray] = {}

    @staticmethod
    def _features(observation: Observation) -> np.ndarray:
        pixels = observation.pixels.astype(np.float64) / 255.0
        return np.concatenate(
            (
                pixels.mean(axis=(0, 1)),
                pixels.std(axis=(0, 1)),
                np.quantile(pixels, (0.25, 0.50, 0.75), axis=(0, 1)).reshape(-1),
            )
        )

    def fit(self, experiences: Sequence[GroundingEvidence]) -> "_StaticPixelsOnlyBaseline":
        grouped: dict[int, list[np.ndarray]] = defaultdict(list)
        for record in experiences:
            if record.task_feedback is not False:
                grouped[int(record.token)].append(self._features(record.before))
        self._centroids = {
            token: np.mean(np.stack(features), axis=0) for token, features in grouped.items()
        }
        return self

    def predict_token(
        self,
        transition: Transition,
        *,
        candidates: Sequence[int],
    ) -> int | None:
        feature = self._features(transition.before)
        distances = {
            int(token): float(np.linalg.norm(feature - self._centroids[int(token)]))
            for token in candidates
            if int(token) in self._centroids
        }
        if not distances:
            return None
        return min(distances, key=lambda token: (distances[token], token))


def _learner_manifest(learner: Any) -> Mapping[str, Any]:
    value = getattr(learner, "manifest", None)
    if callable(value):
        value = value()
    if isinstance(value, Mapping):
        return dict(value)
    return {
        "class": f"{type(learner).__module__}.{type(learner).__qualname__}",
        "declared_manifest": False,
    }


def _new_learner(factory: LearnerFactory, seed: int) -> Any:
    learner = factory(seed)
    if not callable(getattr(learner, "fit", None)):
        raise BenchmarkProtocolError("learner must provide fit(experiences)")
    if not callable(getattr(learner, "predict_token", None)):
        raise BenchmarkProtocolError("learner must provide predict_token(..., candidates=...)")
    return learner


def _predict(learner: Any, transition: Transition, candidates: Sequence[int]) -> Hashable | None:
    return learner.predict_token(transition, candidates=tuple(candidates))


def _atom_truth(
    learner: Any,
    transition: Transition,
    token: int,
    all_tokens: Sequence[int],
) -> bool | None:
    method = getattr(learner, "supports_token", None)
    if callable(method):
        value = method(transition, token)
        if value is None or isinstance(value, (bool, np.bool_)):
            return None if value is None else bool(value)
        if isinstance(value, TruthValue):
            return value.as_python()
        raise BenchmarkProtocolError("supports_token must return bool, TruthValue, or None")
    prediction = _predict(learner, transition, all_tokens)
    if prediction is None:
        return None
    return prediction == token


def _evaluate_composition(
    learner: Any,
    case: CompositionCase,
    all_tokens: Sequence[int],
) -> bool | None:
    evidence = case.evidence

    def resolver(symbol: Hashable) -> bool | None:
        token = int(symbol)
        return _atom_truth(learner, evidence[token], token, all_tokens)

    # Each atom is tested by its own diagnostic intervention.  A learner's
    # convenience evaluate_definition(evidence, ast) method intentionally
    # accepts only one transition, so the multi-evidence evaluator composes
    # the learner's public atom judgments here.
    result = evaluate(case.definition, resolver)
    if isinstance(result, EvaluationResult):
        return result.as_python()
    if isinstance(result, TruthValue):
        return result.as_python()
    if isinstance(result, (bool, np.bool_)) or result is None:
        return None if result is None else bool(result)
    if hasattr(result, "as_python"):
        return result.as_python()
    if hasattr(result, "value") and isinstance(result.value, TruthValue):
        return result.value.as_python()
    raise BenchmarkProtocolError("evaluate_definition returned an unsupported value")


def _trial_hash(transition: Transition) -> str:
    return manifest_hash(
        {
            "before": transition.before.digest(),
            "action": asdict(transition.action),
            "after": transition.after.digest(),
            "outcome": transition.outcome_code,
        }
    )


def _axis_from_records(
    name: str,
    records: Sequence[TrialRecord],
    *,
    threshold: float,
    baseline: float,
    coverage_threshold: float,
    confidence: float,
    description: str,
    coverage_is_execution: bool = False,
) -> CertificateAxis:
    return binary_axis(
        name,
        (record.passed for record in records),
        threshold=threshold,
        baseline=baseline,
        opportunities=(True for _record in records)
        if coverage_is_execution
        else (record.answered for record in records),
        coverage_threshold=coverage_threshold,
        confidence=confidence,
        description=description,
    )


def run_benchmark(
    *,
    seed: int = 0,
    episodes: int = 96,
    learner_factory: LearnerFactory | None = None,
    environment_factory: EnvironmentFactory | None = None,
    world_config: WorldConfig | None = None,
    thresholds: BenchmarkThresholds | None = None,
) -> BenchmarkResult:
    """Run GroundZero-v0 and return a deterministic certificate and ledger.

    Factories are dependency-injected.  The learner factory receives only an
    integer initialization seed.  Its instances receive tuples of
    :class:`GroundingEvidence` and individual :class:`Transition` queries;
    evaluator harnesses and oracle capabilities never cross that boundary.
    """

    selected_thresholds = thresholds or BenchmarkThresholds()
    dataset = build_ground_zero_dataset(
        seed=seed,
        episodes=episodes,
        environment_factory=environment_factory,
        world_config=world_config,
    )
    factory = learner_factory or (lambda learner_seed: _ReferenceTransitionLearner(learner_seed))
    base = _new_learner(factory, _derived_seed(seed, "learner-base"))
    remapped = _new_learner(factory, _derived_seed(seed, "learner-remapped"))
    shuffled = _new_learner(factory, _derived_seed(seed, "learner-shuffled"))
    static_pixels = _StaticPixelsOnlyBaseline(_derived_seed(seed, "static-pixels-control"))
    base.fit(dataset.train)
    remapped.fit(dataset.train_remapped)
    shuffled.fit(dataset.train_intervention_shuffled)
    static_pixels.fit(dataset.train)

    records_by_axis: dict[str, list[TrialRecord]] = {
        name: [] for name in REQUIRED_GROUND_ZERO_AXES
    }

    inverse_remapping = {new: old for old, new in dataset.token_remapping}
    for index, case in enumerate(dataset.token_cases):
        base_prediction = _predict(base, case.transition, dataset.identifiable_tokens)
        remapped_prediction = _predict(
            remapped, case.transition, dataset.remapped_identifiable_tokens
        )
        inverse_prediction = inverse_remapping.get(remapped_prediction)
        passed = (
            base_prediction == case.expected
            and remapped_prediction == case.remapped_expected
            and inverse_prediction == base_prediction
        )
        records_by_axis["token_remapping_equivariance"].append(
            TrialRecord(
                "token_remapping_equivariance",
                index,
                passed,
                base_prediction is not None and remapped_prediction is not None,
                repr((case.expected, case.remapped_expected)),
                repr((base_prediction, remapped_prediction)),
                _trial_hash(case.transition),
                (("inverse_prediction", repr(inverse_prediction)),),
            )
        )

    for index, case in enumerate(dataset.nuisance_cases):
        prediction = _predict(base, case.transition, dataset.identifiable_tokens)
        records_by_axis["nuisance_transfer"].append(
            TrialRecord(
                "nuisance_transfer",
                index,
                prediction == case.expected,
                prediction is not None,
                repr(case.expected),
                repr(prediction),
                _trial_hash(case.transition),
                (("split", "renderer-and-object holdout"),),
            )
        )

    for index, case in enumerate(dataset.intervention_cases):
        prediction = _predict(base, case.transition, dataset.identifiable_tokens)
        ablated_prediction = _predict(shuffled, case.transition, dataset.identifiable_tokens)
        static_prediction = _predict(
            static_pixels, case.transition, dataset.identifiable_tokens
        )
        passed = (
            prediction == case.expected
            and ablated_prediction != case.expected
        )
        records_by_axis["intervention_necessity"].append(
            TrialRecord(
                "intervention_necessity",
                index,
                passed,
                prediction is not None,
                repr(case.expected),
                repr(prediction),
                _trial_hash(case.transition),
                (
                    ("shuffled_prediction", repr(ablated_prediction)),
                    ("static_pixels_no_action_prediction", repr(static_prediction)),
                ),
            )
        )

    for index, case in enumerate(dataset.composition_cases):
        prediction = _evaluate_composition(base, case, dataset.identifiable_tokens)
        first_transition = case.atom_transitions[0][1]
        records_by_axis["unseen_composition"].append(
            TrialRecord(
                "unseen_composition",
                index,
                prediction is case.expected,
                prediction is not None,
                repr(case.expected),
                repr(prediction),
                _trial_hash(first_transition),
                (("direct_composite_training_examples", "0"),),
            )
        )

    for index, case in enumerate(dataset.abstention_cases):
        prediction = _predict(base, case.transition, (case.negative_control_token,))
        records_by_axis["honest_abstention"].append(
            TrialRecord(
                "honest_abstention",
                index,
                prediction is None and case.invariant_verified,
                case.invariant_verified,
                "None",
                repr(prediction),
                _trial_hash(case.transition),
                (("counterfactual_invariance", repr(case.invariant_verified)),),
            )
        )

    axes = (
        _axis_from_records(
            "token_remapping_equivariance",
            records_by_axis["token_remapping_equivariance"],
            threshold=selected_thresholds.token_remapping_equivariance,
            baseline=0.50,
            coverage_threshold=selected_thresholds.answer_coverage,
            confidence=selected_thresholds.confidence,
            description="Predictions commute with a fresh opaque token permutation.",
        ),
        _axis_from_records(
            "nuisance_transfer",
            records_by_axis["nuisance_transfer"],
            threshold=selected_thresholds.nuisance_transfer,
            baseline=1.0 / len(dataset.identifiable_tokens),
            coverage_threshold=selected_thresholds.answer_coverage,
            confidence=selected_thresholds.confidence,
            description="Grounded bindings transfer to sealed renderers, objects, and positions.",
        ),
        _axis_from_records(
            "intervention_necessity",
            records_by_axis["intervention_necessity"],
            threshold=selected_thresholds.intervention_necessity,
            baseline=1.0 / len(dataset.identifiable_tokens),
            coverage_threshold=selected_thresholds.answer_coverage,
            confidence=selected_thresholds.confidence,
            description="Correct evidence pairing succeeds while action/outcome pairing ablation fails.",
        ),
        _axis_from_records(
            "unseen_composition",
            records_by_axis["unseen_composition"],
            threshold=selected_thresholds.unseen_composition,
            baseline=0.50,
            coverage_threshold=selected_thresholds.answer_coverage,
            confidence=selected_thresholds.confidence,
            description="A typed conjunction is evaluated with zero direct composite examples.",
        ),
        _axis_from_records(
            "honest_abstention",
            records_by_axis["honest_abstention"],
            threshold=selected_thresholds.honest_abstention,
            baseline=0.50,
            coverage_threshold=selected_thresholds.answer_coverage,
            confidence=selected_thresholds.confidence,
            description="The learner abstains on a counterfactually unidentifiable predicate.",
            coverage_is_execution=True,
        ),
    )

    learner_manifest = {
        "base": _learner_manifest(base),
        "remapped": _learner_manifest(remapped),
        "intervention_shuffled": _learner_manifest(shuffled),
        "negative_control": {
            "name": "static-pixels-no-action-nearest-centroid",
            "allowed_inputs": ["before.pixels"],
        },
    }
    dataset_manifest = {
        "environment": dataset.public_environment_manifest,
        "split": dataset.split_manifest,
        "token_remapping_hash": manifest_hash(dataset.token_remapping),
        "train_hash": manifest_hash([record.digest() for record in dataset.train]),
    }
    dataset_hash = manifest_hash(dataset_manifest)
    scope = CertificateScope.from_manifests(
        benchmark_id=BENCHMARK_ID,
        environment_family="sealed-finite-causal-microworld-v0",
        sensor_contract="readonly-rgb-pixels+tick+terminal-v0",
        action_contract="opaque-action+target+motor-vector+opaque-outcome-v0",
        target_language=(
            "ostensive-transition-classification+typed-strong-kleene-conjunction-v0"
        ),
        execution_boundary=(
            "trusted-in-process-reference; serialized-subprocess-v0 is required "
            "before extending the claim to adversarial candidates"
        ),
        manifests={
            "benchmark": {
                "id": BENCHMARK_ID,
                "seed": int(seed),
                "episodes": int(episodes),
                "thresholds": asdict(selected_thresholds),
                "claim_limit": (
                    "v0 certifies transition classification and typed conjunction; "
                    "it does not certify active experiment selection or policy learning"
                ),
            },
            "environment": dataset.public_environment_manifest,
            "learner": learner_manifest,
            "lexicon": {
                "base_tokens": dataset.identifiable_tokens,
                "remapping_hash": manifest_hash(dataset.token_remapping),
            },
            "train_split": {
                "hash": manifest_hash([record.digest() for record in dataset.train]),
                "count": len(dataset.train),
            },
            "test_split": dataset.split_manifest,
        },
    )
    certificate = GroundingCertificate(scope, axes)
    static_outcomes: list[bool] = []
    static_coverage: list[bool] = []
    for case in dataset.nuisance_cases:
        value = _predict(static_pixels, case.transition, dataset.identifiable_tokens)
        static_outcomes.append(value == case.expected)
        static_coverage.append(value is not None)
    controls = (
        ControlResult(
            name="static_pixels_no_action",
            performance=MetricBound.binary(
                static_outcomes, confidence=selected_thresholds.confidence
            ),
            coverage=MetricBound.binary(
                static_coverage, confidence=selected_thresholds.confidence
            ),
            grounding_threshold=selected_thresholds.nuisance_transfer,
            coverage_threshold=selected_thresholds.answer_coverage,
            description=(
                "Uses only the pre-intervention RGB frame; it must not satisfy "
                "the grounding gate for randomized hidden causal predicates."
            ),
        ),
    )
    ledger = EvidenceLedger(
        BENCHMARK_ID,
        int(seed),
        int(episodes),
        dataset_hash,
        tuple(record for name in REQUIRED_GROUND_ZERO_AXES for record in records_by_axis[name]),
        controls,
    )
    return BenchmarkResult(certificate, ledger)


def _result_summary(result: BenchmarkResult, *, learner: str) -> dict[str, Any]:
    """Return a compact, stable CLI view without discarding the full ledger API."""

    return {
        "benchmark_id": BENCHMARK_ID,
        "learner": learner,
        "passed": result.passed,
        "score": result.certificate.score,
        "scope_hash": result.certificate.scope.scope_hash,
        "certificate_hash": result.certificate.certificate_hash,
        "axes": {
            axis.name: {
                "estimate": axis.performance.estimate,
                "lower_bound_95": axis.performance.lower_bound,
                "coverage": axis.coverage.estimate,
                "coverage_lower_bound_95": axis.coverage.lower_bound,
                "passed": axis.passed,
            }
            for axis in result.certificate.axes
        },
        "controls": {
            control.name: {
                "estimate": control.performance.estimate,
                "coverage": control.coverage.estimate,
                "rejected_as_grounder": control.rejected_as_grounder,
            }
            for control in result.ledger.controls
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reproducible GroundZero-v0 command-line experiment."""

    parser = argparse.ArgumentParser(
        prog="grounding-kernel",
        description=(
            "Run the scoped GroundZero-v0 operational symbol-grounding certificate."
        ),
    )
    parser.add_argument("--seed", type=int, default=3, help="non-negative experiment seed")
    parser.add_argument(
        "--episodes",
        type=int,
        default=24,
        help="trials per mandatory certificate axis (minimum: 8)",
    )
    parser.add_argument(
        "--learner",
        choices=("binder", "reference"),
        default="binder",
        help="learner under test; reference only self-checks the protocol",
    )
    parser.add_argument(
        "--full-json",
        action="store_true",
        help="emit the complete certificate and evidence ledger",
    )
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.episodes < 8:
        parser.error("--episodes must be at least 8")

    selected_learner_factory: LearnerFactory | None = None
    if arguments.learner == "binder":
        from .binder import SensorimotorBinder

        def binder_factory(_seed: int) -> SensorimotorBinder:
            return SensorimotorBinder()

        selected_learner_factory = binder_factory
    result = run_benchmark(
        seed=arguments.seed,
        episodes=arguments.episodes,
        learner_factory=selected_learner_factory,
    )
    document = (
        result.to_dict()
        if arguments.full_json
        else _result_summary(result, learner=arguments.learner)
    )
    print(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":") if arguments.compact else None,
            indent=None if arguments.compact else 2,
        )
    )
    return 0 if result.passed else 1


__all__ = [
    "BENCHMARK_ID",
    "AbstentionCase",
    "BenchmarkProtocolError",
    "BenchmarkResult",
    "BenchmarkThresholds",
    "CompositionCase",
    "ControlResult",
    "EnvironmentFactory",
    "EvidenceLedger",
    "GroundZeroDataset",
    "GroundZeroLearner",
    "GroundingEvidence",
    "LearnerFactory",
    "PairedTokenCase",
    "TokenCase",
    "TrialRecord",
    "build_ground_zero_dataset",
    "main",
    "run_benchmark",
]


if __name__ == "__main__":  # pragma: no cover - exercised through a CLI smoke test
    raise SystemExit(main())
