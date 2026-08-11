"""GroundZero-v1 transparent reference self-test.

This module intentionally reports claim grade 2 from ``THREAT_MODEL.md``.  It
exercises outcome-free perception, actually executed active interventions,
closed-loop language behavior and conservative shortcut controls across outer
world blocks.  It does *not* accept an arbitrary candidate and therefore does
not issue an adversarial grounding certificate.  ``v1_runner`` supplies the
persistent serialized candidate boundary for the next claim grade.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any
import argparse
import hmac
import json
import platform

import numpy as np

from .active_process import run_active_process_block
from .certificates import (
    CertificateAxis,
    CertificateScope,
    GroundingCertificate,
    MetricBound,
    manifest_hash,
)
from .episode_binder import EpisodeConceptBinder, action_target_transcript
from .process_language import run_process_language_block
from .process_quotient import process_pair_quotient
from .processworld import ProcessConfig, ProcessHarness
from .v1_adapters import (
    episode_to_query_trace,
    fresh_opaque_token,
    ostensive_record_to_support,
    support_episode_to_binder_record,
)
from .v1_controls import (
    ChanceShortcutControl,
    MatchedTwinControl,
)
from .v1_wire import public_schema_manifest


BENCHMARK_ID = "ground-zero-v1-reference-self-test"
CLAIM_GRADE = 2
DEFAULT_BLOCKS = 32
_DEVELOPMENT_EVALUATOR_SECRET = sha256(
    f"{BENCHMARK_ID}|public-development-evaluator-secret-v1".encode()
).digest()
V1_REQUIRED_AXES = (
    "active_acquisition_efficiency",
    "object_affordance_invariance",
    "process_causality_and_order",
    "sensor_context_necessity",
    "description_to_action",
    "trace_to_description",
    "factorial_composition",
    "lexicon_permutation_equivariance",
    "proof_grounded_symbolic_theft",
    "open_set_honesty",
)


@dataclass(frozen=True, slots=True)
class V1ControlSpec:
    mode: str
    channel_mask: tuple[str, ...]
    description: str

    def __post_init__(self) -> None:
        if self.mode not in {"matched", "chance", "executed", "lookup"}:
            raise ValueError("unsupported control aggregation mode")
        if not self.channel_mask or len(set(self.channel_mask)) != len(self.channel_mask):
            raise ValueError("control channel masks must be non-empty and unique")
        if not self.description:
            raise ValueError("control description cannot be empty")


def _control_specs() -> Mapping[str, V1ControlSpec]:
    values: dict[str, V1ControlSpec] = {}
    descriptions = {
        "action_only": "Opaque action code/vector/tick channels without raw sensors.",
        "action_outcome_only": "Opaque action codes with the stripped outcome channel.",
        "action_target_only": "Complete action/vector/target trajectories without raw sensors.",
        "target_only": "Complete target-coordinate trajectories without raw sensors or actions.",
    }
    masks = {
        "action_only": ("action.code", "action.vector", "tick.delta"),
        "action_outcome_only": ("action.code", "outcome.absent"),
        "action_target_only": (
            "action.code",
            "action.vector",
            "action.target.trajectory",
            "tick.delta",
        ),
        "target_only": ("action.target.trajectory", "tick.delta"),
    }
    for prefix in descriptions:
        for task in ("context", "object", "process"):
            values[f"{prefix}_{task}"] = V1ControlSpec(
                "matched",
                (*masks[prefix], f"task.{task}"),
                descriptions[prefix],
            )
    values.update(
        {
            "definition_leaf_deletion": V1ControlSpec(
                "executed",
                ("language.definition", "grounded.leaf.deleted"),
                "Deleting a directly grounded leaf must make the composite unknown.",
            ),
            "definition_leaf_swap": V1ControlSpec(
                "executed",
                ("language.definition", "grounded.leaf.swapped"),
                "Swapping a grounded leaf must change requested behavior.",
            ),
            "full_utterance_lookup": V1ControlSpec(
                "lookup",
                ("utterance.exact", "support.lookup.table"),
                "An exact support-utterance table cannot factor a held-out composition.",
            ),
            "intervention_consequence_shuffle": V1ControlSpec(
                "chance",
                ("sensor.trace", "intervention.consequence.deranged"),
                "Deranging executed consequences must destroy resolution; inversion leaks.",
            ),
            "target_rank_only": V1ControlSpec(
                "chance",
                ("initial.rgb", "candidate.demonstrated-rank", "action.scheme"),
                "A demonstrated visual rank without intervention binding is insufficient.",
            ),
            "temporal_shuffle": V1ControlSpec(
                "executed",
                ("raw.rgb", "action.sequence.permuted"),
                "Actually executed temporal permutations must not be accepted.",
            ),
        }
    )
    return MappingProxyType(values)


V1_CONTROL_SPECS = _control_specs()
REQUIRED_V1_CONTROLS = tuple(sorted(V1_CONTROL_SPECS))


@dataclass(frozen=True, slots=True)
class V1Thresholds:
    """Predeclared marginal cluster-level gates for the reference self-test."""

    performance: float = 0.80
    answer_coverage: float = 0.80
    confidence: float = 0.95
    shortcut_leakage_ceiling: float = 0.20
    chance_tolerance: float = 0.25

    def __post_init__(self) -> None:
        if not 0.5 < self.performance <= 1.0:
            raise ValueError("performance must lie in (0.5, 1]")
        if not 0.0 < self.answer_coverage <= 1.0:
            raise ValueError("answer_coverage must lie in (0, 1]")
        if not 0.0 < self.confidence < 1.0:
            raise ValueError("confidence must lie in (0, 1)")
        if not 0.0 <= self.shortcut_leakage_ceiling < 1.0:
            raise ValueError("shortcut_leakage_ceiling must lie in [0, 1)")
        if not 0.0 <= self.chance_tolerance < 0.50:
            raise ValueError("chance_tolerance must lie in [0, 0.5)")


@dataclass(frozen=True, slots=True)
class V1TrialRecord:
    axis: str
    block: int
    passed: bool
    answered: bool
    evidence_hash: str
    detail: tuple[tuple[str, str], ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "block": self.block,
            "passed": self.passed,
            "answered": self.answered,
            "evidence_hash": self.evidence_hash,
            "detail": dict(self.detail),
        }


@dataclass(frozen=True, slots=True)
class V1ControlResult:
    """Direction-correct upper-bound gate on a shortcut's leakage."""

    name: str
    leakage: MetricBound
    coverage: MetricBound
    maximum_leakage: float
    description: str
    minimum_coverage: float = 0.80

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("control name cannot be empty")
        if self.leakage.successes is None or self.coverage.successes is None:
            raise TypeError("v1 controls require binary confidence intervals")
        if not 0.0 <= self.maximum_leakage < 1.0:
            raise ValueError("maximum_leakage must lie in [0, 1)")
        if not 0.0 < self.minimum_coverage <= 1.0:
            raise ValueError("minimum_coverage must lie in (0, 1]")

    @property
    def performance(self) -> MetricBound:
        """Backward-compatible alias; the quantity is shortcut leakage."""

        return self.leakage

    @property
    def low_leakage_proven(self) -> bool:
        return self.leakage.upper_bound <= self.maximum_leakage

    @property
    def insufficient_coverage_proven(self) -> bool:
        return self.coverage.upper_bound < self.minimum_coverage

    @property
    def insufficient_as_full_grounder(self) -> bool:
        return self.low_leakage_proven or self.insufficient_coverage_proven

    @property
    def rejected_as_grounder(self) -> bool:
        """Compatibility alias for ``insufficient_as_full_grounder``."""

        return self.insufficient_as_full_grounder

    @property
    def rejection_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if self.low_leakage_proven:
            reasons.append("low-leakage-upper-bound")
        if self.insufficient_coverage_proven:
            reasons.append("insufficient-coverage-upper-bound")
        return tuple(reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "leakage": self.leakage.to_dict(),
            "coverage": self.coverage.to_dict(),
            "maximum_leakage": self.maximum_leakage,
            "minimum_coverage": self.minimum_coverage,
            "description": self.description,
            "low_leakage_proven": self.low_leakage_proven,
            "insufficient_coverage_proven": self.insufficient_coverage_proven,
            "insufficient_as_full_grounder": self.insufficient_as_full_grounder,
            "rejection_reasons": list(self.rejection_reasons),
            "rejected_as_grounder": self.rejected_as_grounder,
        }


@dataclass(frozen=True, slots=True)
class V1ControlTrial:
    """One committed outer-block observation for a registered control."""

    name: str
    block: int
    channel_mask: tuple[str, ...]
    first: bool | None
    second: bool | None
    evidence_hash: str

    def __post_init__(self) -> None:
        if self.name not in V1_CONTROL_SPECS:
            raise ValueError(f"unregistered control: {self.name}")
        if isinstance(self.block, bool) or not isinstance(self.block, int) or self.block < 0:
            raise ValueError("control block must be a non-negative integer")
        expected = V1_CONTROL_SPECS[self.name].channel_mask
        if tuple(self.channel_mask) != expected:
            raise ValueError("control trial channel mask does not match its registry")
        if self.first not in (True, False, None) or self.second not in (
            True,
            False,
            None,
        ):
            raise TypeError("control trial outcomes must be bool or None")
        if len(self.evidence_hash) != 64:
            raise ValueError("control evidence hash must be a SHA-256 hex digest")
        mode = V1_CONTROL_SPECS[self.name].mode
        if mode != "matched" and self.second is not None:
            raise ValueError("only matched controls may carry a second outcome")
        if mode == "executed" and self.first is None:
            raise ValueError("executed controls require a Boolean leakage event")
        if mode == "lookup" and self.first is False:
            raise ValueError("lookup controls encode answer=True or abstention=None")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "block": self.block,
            "channel_mask": list(self.channel_mask),
            "first": self.first,
            "second": self.second,
            "evidence_hash": self.evidence_hash,
        }


@dataclass(frozen=True, slots=True)
class V1EvidenceLedger:
    benchmark_id: str
    seed: int
    blocks: int
    dataset_hash: str
    trials: tuple[V1TrialRecord, ...]
    control_trials: tuple[V1ControlTrial, ...]
    controls: tuple[V1ControlResult, ...]

    def __post_init__(self) -> None:
        names = tuple(control.name for control in self.controls)
        if len(names) != len(set(names)):
            raise ValueError("control names must be unique")
        if set(names) != set(REQUIRED_V1_CONTROLS):
            raise ValueError("aggregate controls must exactly match the required registry")
        observed = {(trial.name, trial.block) for trial in self.control_trials}
        expected = {
            (name, block)
            for name in REQUIRED_V1_CONTROLS
            for block in range(self.blocks)
        }
        if len(observed) != len(self.control_trials) or observed != expected:
            raise ValueError("require exactly one raw control trial per name and block")

    @property
    def axis_counts(self) -> Mapping[str, Mapping[str, int]]:
        values: dict[str, Mapping[str, int]] = {}
        for axis in V1_REQUIRED_AXES:
            records = tuple(record for record in self.trials if record.axis == axis)
            values[axis] = MappingProxyType(
                {
                    "clusters": len(records),
                    "passed": sum(record.passed for record in records),
                    "answered": sum(record.answered for record in records),
                }
            )
        return MappingProxyType(values)

    @property
    def ledger_hash(self) -> str:
        return manifest_hash(
            {
                "benchmark_id": self.benchmark_id,
                "seed": self.seed,
                "blocks": self.blocks,
                "dataset_hash": self.dataset_hash,
                "trials": [record.to_dict() for record in self.trials],
                "control_trials": [trial.to_dict() for trial in self.control_trials],
                "controls": [control.to_dict() for control in self.controls],
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "benchmark_id": self.benchmark_id,
            "seed": self.seed,
            "blocks": self.blocks,
            "dataset_hash": self.dataset_hash,
            "axis_counts": {
                name: dict(values) for name, values in self.axis_counts.items()
            },
            "trials": [record.to_dict() for record in self.trials],
            "control_trials": [trial.to_dict() for trial in self.control_trials],
            "controls": [control.to_dict() for control in self.controls],
            "ledger_hash": self.ledger_hash,
        }


@dataclass(frozen=True, slots=True)
class V1BenchmarkResult:
    """Reference metric bundle; deliberately not an adversarial certificate."""

    metrics: GroundingCertificate
    ledger: V1EvidenceLedger
    preregistered_default: bool

    @property
    def certificate(self) -> GroundingCertificate:
        """Compatibility alias for older callers; see ``certificate_eligible``."""

        return self.metrics

    @property
    def passed(self) -> bool:
        return self.metrics.passed and self.controls_complete and all(
            control.rejected_as_grounder for control in self.ledger.controls
        )

    @property
    def controls_complete(self) -> bool:
        observed = {control.name for control in self.ledger.controls}
        return observed == set(REQUIRED_V1_CONTROLS) and len(
            self.ledger.control_trials
        ) == self.ledger.blocks * len(REQUIRED_V1_CONTROLS)

    @property
    def certificate_eligible(self) -> bool:
        return False

    def to_dict(self) -> dict[str, Any]:
        metric_bundle = self.metrics.to_dict()
        metric_bundle["kind"] = "noncompensatory-reference-metric-bundle"
        return {
            "kind": "reference-component-self-test",
            "claim_grade": CLAIM_GRADE,
            "benchmark_id": BENCHMARK_ID,
            "self_test_passed": self.passed,
            "passed": self.passed,
            "preregistered_default": self.preregistered_default,
            "certificate_eligible": False,
            "certified": False,
            "metric_axes_passed": self.metrics.passed,
            "shortcut_controls_insufficient_as_full_grounders": all(
                control.insufficient_as_full_grounder
                for control in self.ledger.controls
            )
            and self.controls_complete,
            "negative_controls_valid": all(
                control.insufficient_as_full_grounder
                for control in self.ledger.controls
            )
            and self.controls_complete,
            "required_controls_complete": self.controls_complete,
            "metric_bundle": metric_bundle,
            "ledger": self.ledger.to_dict(),
        }


def _derived_seed(seed: int, namespace: str, block: int) -> int:
    payload = f"{BENCHMARK_ID}|{seed}|{namespace}|{block}".encode()
    return int.from_bytes(sha256(payload).digest()[:8], "big")


def _reference_source_manifest() -> dict[str, object]:
    root = Path(__file__).resolve().parent
    package_sources = tuple(sorted(root.glob("*.py"), key=lambda path: path.name))
    source_hashes = {
        f"grounding_kernel/{path.name}": sha256(path.read_bytes()).hexdigest()
        for path in package_sources
    }
    project_manifest = root.parent / "pyproject.toml"
    source_hashes[project_manifest.name] = sha256(project_manifest.read_bytes()).hexdigest()
    return {
        "source_hashes": source_hashes,
        "model_commitment": manifest_hash(source_hashes),
        "runtime": {"python": platform.python_version(), "numpy": np.__version__},
    }


def _trial(
    axis: str,
    block: int,
    passed: bool,
    answered: bool,
    detail: Mapping[str, object],
) -> V1TrialRecord:
    rendered = tuple(sorted((str(key), repr(value)) for key, value in detail.items()))
    return V1TrialRecord(
        axis,
        block,
        bool(passed),
        bool(answered),
        manifest_hash({"axis": axis, "block": block, "detail": rendered}),
        rendered,
    )


def _control_trial(
    name: str,
    block: int,
    first: bool | None,
    second: bool | None = None,
    *,
    evidence: Mapping[str, object] | None = None,
) -> V1ControlTrial:
    if name not in V1_CONTROL_SPECS:
        raise ValueError(f"unregistered control: {name}")
    payload = {
        "name": name,
        "block": block,
        "channel_mask": V1_CONTROL_SPECS[name].channel_mask,
        "first": first,
        "second": second,
        "evidence": dict(evidence or {}),
    }
    return V1ControlTrial(
        name,
        block,
        V1_CONTROL_SPECS[name].channel_mask,
        first,
        second,
        manifest_hash(payload),
    )


def _axis(
    name: str,
    records: Sequence[V1TrialRecord],
    thresholds: V1Thresholds,
    description: str,
    *,
    coverage_is_execution: bool = False,
) -> CertificateAxis:
    return CertificateAxis(
        name=name,
        performance=MetricBound.binary_exact(
            (record.passed for record in records),
            confidence=thresholds.confidence,
        ),
        threshold=thresholds.performance,
        baseline=0.50,
        coverage=MetricBound.binary_exact(
            (True for _record in records)
            if coverage_is_execution
            else (record.answered for record in records),
            confidence=thresholds.confidence,
        ),
        coverage_threshold=thresholds.answer_coverage,
        description=description,
    )


def _active_block(
    seed: int,
    block: int,
    evaluator_secret: bytes,
) -> tuple[bool, bool, dict[str, object]]:
    report = run_active_process_block(
        seed,
        block,
        evaluator_secret=evaluator_secret,
    )
    return report.passed, report.answered, dict(report.detail)


def _binder_block(
    seed: int,
    block: int,
) -> tuple[
    dict[str, tuple[bool, bool, dict[str, object]]],
    dict[str, MatchedTwinControl],
]:
    world_seed = _derived_seed(seed, "binder-world", block) % (1 << 31)
    support_records = tuple(
        support_episode_to_binder_record(
            ostensive_record_to_support(
                record,
                turn_id=variant * 100 + index,
                remaining_cost=1_000.0,
            )
        )
        for variant in (0, 1, 2)
        for index, record in enumerate(
            ProcessHarness(
                world_seed,
                renderer_variant=variant,
                world_variant=variant,
            ).oracle.examples(include_negative_control=True)
        )
    )
    full = EpisodeConceptBinder(mode="full").fit(support_records)
    action_only = EpisodeConceptBinder(mode="action_only").fit(support_records)
    target_only = EpisodeConceptBinder(mode="target_only").fit(support_records)
    action_target_only = EpisodeConceptBinder(mode="action_target_only").fit(
        support_records
    )
    action_outcome_only = EpisodeConceptBinder(mode="action_outcome_only").fit(
        support_records
    )
    holdout_renderer = 10_000 + block
    holdout_world = 20_000 + block
    holdout = ProcessHarness(
        world_seed,
        renderer_variant=holdout_renderer,
        world_variant=holdout_world,
    )

    shelter_positive, shelter_negative = holdout.oracle.affordance_pair()
    shelter_traces = (
        episode_to_query_trace(shelter_positive.episode),
        episode_to_query_trace(shelter_negative.episode),
    )
    shelter_predictions = (
        full.supports_token(shelter_traces[0], shelter_positive.token),
        full.supports_token(shelter_traces[1], shelter_negative.token),
    )

    process_positive, process_negative = holdout.oracle.process_pair()
    process_traces = (
        episode_to_query_trace(process_positive.episode),
        episode_to_query_trace(process_negative.episode),
    )
    process_predictions = (
        full.supports_token(process_traces[0], process_positive.token),
        full.supports_token(process_traces[1], process_negative.token),
    )
    if action_target_transcript(process_traces[0]) != action_target_transcript(
        process_traces[1]
    ):
        raise RuntimeError(
            "matched process twins must expose identical complete action-target transcripts"
        )
    shuffle_harness = ProcessHarness(
        world_seed,
        renderer_variant=holdout_renderer,
        world_variant=holdout_world,
    )
    shuffle_harness.agent.reset()
    for index in (0, 2, 1, 3):
        shuffle_harness.agent.step(process_positive.episode.transitions[index].action)
    shuffled_trace = episode_to_query_trace(shuffle_harness.agent.episode())
    shuffled_prediction = full.supports_token(shuffled_trace, process_positive.token)

    context_positive, context_negative = holdout.oracle.context_pair()
    context_traces = (
        episode_to_query_trace(context_positive.episode),
        episode_to_query_trace(context_negative.episode),
    )
    context_predictions = (
        full.supports_token(context_traces[0], context_positive.token),
        full.supports_token(context_traces[1], context_negative.token),
    )
    controls: dict[str, MatchedTwinControl] = {}

    def add_control(
        name: str,
        model: EpisodeConceptBinder,
        traces: tuple[object, object],
        token: int,
    ) -> MatchedTwinControl:
        value = MatchedTwinControl(
            model.supports_token(traces[0], token),
            model.supports_token(traces[1], token),
        )
        controls[name] = value
        return value

    sensorless_context = add_control(
        "action_target_only_context",
        action_target_only,
        context_traces,
        context_positive.token,
    )
    action_context = add_control(
        "action_only_context", action_only, context_traces, context_positive.token
    )
    add_control("action_only_object", action_only, shelter_traces, shelter_positive.token)
    add_control("action_only_process", action_only, process_traces, process_positive.token)
    for prefix, model in (
        ("target_only", target_only),
        ("action_target_only", action_target_only),
        ("action_outcome_only", action_outcome_only),
    ):
        add_control(f"{prefix}_object", model, shelter_traces, shelter_positive.token)
        add_control(f"{prefix}_process", model, process_traces, process_positive.token)
        if prefix != "action_target_only":
            add_control(f"{prefix}_context", model, context_traces, context_positive.token)

    negative_left, negative_right = holdout.oracle.negative_control_pair()
    negative_traces = (
        episode_to_query_trace(negative_left.episode),
        episode_to_query_trace(negative_right.episode),
    )
    negative_predictions = (
        full.supports_token(negative_traces[0], negative_left.token),
        full.supports_token(negative_traces[1], negative_right.token),
    )
    negative_report = process_pair_quotient(
        negative_left,
        negative_right,
        commitment_nonce=f"{seed}:{block}",
    )
    fresh_token = fresh_opaque_token(
        tuple(int(token) for token in full.tokens), nonce=block
    )
    fresh_prediction = full.supports_token(process_traces[0], fresh_token)

    context_passed = (
        context_predictions == (True, False)
        and not sensorless_context.informative_leak
        and not action_context.informative_leak
        and context_positive.episode.non_sensor_transcript()
        == context_negative.episode.non_sensor_transcript()
    )
    open_passed = (
        negative_left.token in full.tokens
        and negative_predictions == (None, None)
        and fresh_prediction is None
        and not negative_report.quotient_compatible
    )
    results = {
        "object_affordance_invariance": (
            shelter_predictions == (True, False),
            all(value is not None for value in shelter_predictions),
            {
                "predictions": shelter_predictions,
                "binder": full.manifest.digest(),
                "renderer": holdout_renderer,
                "world_variant": holdout_world,
                "outcome_channel": "absent",
            },
        ),
        "process_causality_and_order": (
            process_predictions == (True, False) and shuffled_prediction is not True,
            all(value is not None for value in process_predictions),
            {
                "predictions": process_predictions,
                "shuffled_prediction": shuffled_prediction,
                "binder": full.manifest.digest(),
                "outcome_channel": "absent",
            },
        ),
        "sensor_context_necessity": (
            context_passed,
            all(value is not None for value in context_predictions),
            {
                "full": context_predictions,
                "sensorless_action_target": (
                    sensorless_context.positive_prediction,
                    sensorless_context.negative_prediction,
                ),
                "action_only": (
                    action_context.positive_prediction,
                    action_context.negative_prediction,
                ),
                "transcripts_equal": True,
            },
        ),
        "open_set_honesty": (
            open_passed,
            True,
            {
                "negative_predictions": negative_predictions,
                "fresh_prediction": fresh_prediction,
                "negative_quotient_compatible": negative_report.quotient_compatible,
                "profile_hash": negative_report.profile_hash,
                "same_domain_fresh_token": True,
            },
        ),
    }
    return results, controls


def _aggregate_control(
    name: str,
    trials: Sequence[V1ControlTrial],
    thresholds: V1Thresholds,
) -> V1ControlResult:
    """Recompute one aggregate exclusively from committed block observations."""

    spec = V1_CONTROL_SPECS[name]
    ordered = tuple(sorted(trials, key=lambda trial: trial.block))
    if len(ordered) == 0 or any(trial.name != name for trial in ordered):
        raise ValueError("control aggregation requires same-name raw trials")
    if spec.mode == "matched":
        values = tuple(MatchedTwinControl(trial.first, trial.second) for trial in ordered)
        leakage_outcomes = tuple(value.informative_leak for value in values)
        coverage_outcomes = tuple(value.answered_fraction == 1.0 for value in values)
        maximum = thresholds.shortcut_leakage_ceiling
    elif spec.mode == "chance":
        direct = tuple(trial.first for trial in ordered)
        bound = ChanceShortcutControl(
            name,
            direct,
            confidence=thresholds.confidence,
            chance_level=0.50,
            tolerance=thresholds.chance_tolerance,
            description=spec.description,
        )
        answered = bound.answered
        orientation_successes = bound.orientation_invariant_successes
        leakage_outcomes = (True,) * orientation_successes + (False,) * (
            answered - orientation_successes
        )
        coverage_outcomes = tuple(value is not None for value in direct)
        maximum = bound.maximum_leakage
    elif spec.mode == "executed":
        if any(trial.first is None for trial in ordered):
            raise ValueError("executed controls require a Boolean leakage event")
        leakage_outcomes = tuple(bool(trial.first) for trial in ordered)
        coverage_outcomes = (True,) * len(ordered)
        maximum = thresholds.shortcut_leakage_ceiling
    else:
        if any(trial.first is False for trial in ordered):
            raise ValueError("lookup controls encode an answer as True or abstention as None")
        leakage_outcomes = tuple(trial.first is not None for trial in ordered)
        coverage_outcomes = leakage_outcomes
        maximum = thresholds.shortcut_leakage_ceiling
    return V1ControlResult(
        name,
        MetricBound.binary_exact(
            leakage_outcomes,
            confidence=thresholds.confidence,
        ),
        MetricBound.binary_exact(
            coverage_outcomes,
            confidence=thresholds.confidence,
        ),
        maximum,
        spec.description,
        minimum_coverage=thresholds.answer_coverage,
    )


def _run_v1_benchmark(
    *,
    seed: int = 3,
    blocks: int = DEFAULT_BLOCKS,
    thresholds: V1Thresholds | None = None,
    evaluator_secret: bytes | None = None,
    _evaluation_master_commitment: str | None = None,
) -> V1BenchmarkResult:
    """Run the claim-grade-2 reference experiment and return metrics+ledger."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(blocks, bool) or not isinstance(blocks, int) or blocks < 8:
        raise ValueError("blocks must be an integer of at least 8")
    selected = thresholds or V1Thresholds()
    if evaluator_secret is None:
        selected_evaluator_secret = _DEVELOPMENT_EVALUATOR_SECRET
        evaluator_secret_kind = "public-development-key"
    else:
        if not isinstance(evaluator_secret, bytes) or len(evaluator_secret) < 32:
            raise ValueError("evaluator_secret must contain at least 32 private bytes")
        selected_evaluator_secret = evaluator_secret
        evaluator_secret_kind = (
            "domain-separated-master-key"
            if _evaluation_master_commitment is not None
            else "caller-supplied-target-commitment-key"
        )
    # Chronology is not inferable from numeric defaults.  Seed 3 was used
    # during development, so it must never be relabelled preregistered merely
    # because the caller selected the default arguments.  A post-freeze
    # holdout is recorded externally with a source commitment and one-time
    # seed reveal in RESULTS_V1.md.
    preregistered = False
    reference_commitment = _reference_source_manifest()
    records_by_axis: dict[str, list[V1TrialRecord]] = {
        axis: [] for axis in V1_REQUIRED_AXES
    }
    control_trials: list[V1ControlTrial] = []
    manifests: list[dict[str, object]] = []

    for block in range(blocks):
        active_passed, active_answered, active_detail = _active_block(
            seed,
            block,
            selected_evaluator_secret,
        )
        records_by_axis["active_acquisition_efficiency"].append(
            _trial(
                "active_acquisition_efficiency",
                block,
                active_passed,
                active_answered,
                active_detail,
            )
        )
        shuffle_prediction = active_detail.get("intervention_shuffle_prediction", True)
        if shuffle_prediction is not None and not isinstance(shuffle_prediction, bool):
            raise TypeError("intervention shuffle prediction must be bool or None")
        control_trials.append(
            _control_trial(
                "intervention_consequence_shuffle",
                block,
                shuffle_prediction,
                evidence={
                    key: value
                    for key, value in active_detail.items()
                    if key.startswith("intervention_shuffle_")
                },
            )
        )

        binder_results, binder_block_controls = _binder_block(seed, block)
        for axis, (passed, answered, detail) in binder_results.items():
            records_by_axis[axis].append(_trial(axis, block, passed, answered, detail))
        for name, value in binder_block_controls.items():
            control_trials.append(
                _control_trial(
                    name,
                    block,
                    value.positive_prediction,
                    value.negative_prediction,
                )
            )
        shuffled_prediction = binder_results["process_causality_and_order"][2][
            "shuffled_prediction"
        ]
        control_trials.append(
            _control_trial(
                "temporal_shuffle",
                block,
                shuffled_prediction is True,
                evidence={"prediction": shuffled_prediction},
            )
        )

        language = run_process_language_block(seed, block)
        for axis, (passed, answered, detail) in language.results.items():
            records_by_axis[axis].append(_trial(axis, block, passed, answered, detail))
        control_trials.extend(
            (
                _control_trial(
                    "full_utterance_lookup",
                    block,
                    True if language.lookup_leaked else None,
                ),
                _control_trial(
                    "target_rank_only",
                    block,
                    language.target_only_prediction,
                ),
                _control_trial(
                    "definition_leaf_deletion",
                    block,
                    language.definition_leaf_deletion_leaked,
                ),
                _control_trial(
                    "definition_leaf_swap",
                    block,
                    language.definition_leaf_swap_leaked,
                ),
            )
        )
        # A fresh language token is part of the open-set conjunction, not just
        # a descriptive control.
        existing_open = records_by_axis["open_set_honesty"][-1]
        if not language.language_fresh_unknown:
            records_by_axis["open_set_honesty"][-1] = _trial(
                "open_set_honesty",
                block,
                False,
                existing_open.answered,
                {"binder_evidence": existing_open.evidence_hash, "language_fresh": False},
            )
        manifests.append(
            {
                "block": block,
                "active": active_detail,
                "binder": {
                    axis: detail
                    for axis, (_passed, _answered, detail) in binder_results.items()
                },
                "language": {
                    axis: detail
                    for axis, (_passed, _answered, detail) in language.results.items()
                },
            }
        )

    descriptions = {
        "active_acquisition_efficiency": (
            "Hypotheses come from independent public support worlds; selected opaque "
            "interventions execute through reset/step and achieve strict paired "
            "cost-to-correct savings over exhaustive random and counterbalanced passive."
        ),
        "object_affordance_invariance": (
            "Outcome-free raw visual effects distinguish protective affordance from a "
            "matched nonprotective twin in a new renderer/world instance."
        ),
        "process_causality_and_order": (
            "A perturbation-revealed self-sustaining process transfers, while an "
            "actually executed temporal action permutation is not accepted."
        ),
        "sensor_context_necessity": (
            "Only the raw-sensor learner solves a pair with identical complete "
            "nonsensory transcripts; inversion also counts as shortcut leakage."
        ),
        "description_to_action": (
            "A wholly withheld opaque utterance visually locates and intervenes on "
            "candidates in a new world before causing the intended process."
        ),
        "trace_to_description": (
            "A feedback-free raw world trace is first recognized as an operational "
            "program, then described and parsed back to that referent."
        ),
        "factorial_composition": (
            "The held-out role×scheme pairing executes, and both actual slot-swap "
            "programs produce the expected different world consequences."
        ),
        "lexicon_permutation_equivariance": (
            "A fresh total surface-token permutation changes utterances but not the "
            "recovered closed-loop behavior."
        ),
        "proof_grounded_symbolic_theft": (
            "The definition proof materializes the executable referent itself; an "
            "unanchored dictionary cycle remains unknown."
        ),
        "open_set_honesty": (
            "Same-domain fresh tokens and an equally exposed quotient-unidentifiable "
            "concept are rejected."
        ),
    }
    axes = tuple(
        _axis(
            axis,
            records_by_axis[axis],
            selected,
            descriptions[axis],
            coverage_is_execution=axis
            in {"proof_grounded_symbolic_theft", "open_set_honesty"},
        )
        for axis in V1_REQUIRED_AXES
    )
    controls = [
        _aggregate_control(
            name,
            tuple(trial for trial in control_trials if trial.name == name),
            selected,
        )
        for name in REQUIRED_V1_CONTROLS
    ]

    dataset_manifest = {
        "seed": seed,
        "blocks": blocks,
        "cluster_unit": "outer-seed-block; component-specific worlds are nested evidence",
        "block_manifests_hash": manifest_hash(manifests),
        "wire_schema": public_schema_manifest(),
        "model_commitment": reference_commitment["model_commitment"],
        "evaluator_secret_commitment": sha256(selected_evaluator_secret).hexdigest(),
        "evaluator_secret_kind": evaluator_secret_kind,
        "evaluation_seed_mode": (
            "post-freeze-master-derived"
            if _evaluation_master_commitment is not None
            else "explicit-development-seed"
        ),
        "evaluation_master_commitment": _evaluation_master_commitment,
        "preregistered_default": False,
        "development_configuration": (
            seed == 3 and blocks == DEFAULT_BLOCKS and selected == V1Thresholds()
        ),
        "marginal_confidence_intervals": "one-sided-exact-clopper-pearson",
    }
    dataset_hash = manifest_hash(dataset_manifest)
    scope = CertificateScope.from_manifests(
        benchmark_id=BENCHMARK_ID,
        environment_family="finite-processworld-v1-reference-family",
        sensor_contract="raw-rgb-ordered-public-traces; support-only-generic-feedback",
        action_contract="opaque-action+image-target+motor-vector+public-reset-step",
        target_language="fixed-order-opaque-slots+closed-loop-two-part-programs",
        execution_boundary=(
            "transparent-in-process-multi-component-reference-self-test; the persistent "
            "JSON v1_runner exists but this benchmark does not yet route every axis "
            "through one candidate or an external OS sandbox"
        ),
        manifests={
            "benchmark": {
                "id": BENCHMARK_ID,
                "claim_grade": CLAIM_GRADE,
                "seed": seed,
                "blocks": blocks,
                "thresholds": asdict(selected),
                "required_axes": V1_REQUIRED_AXES,
                "required_controls": {
                    name: {
                        "mode": V1_CONTROL_SPECS[name].mode,
                        "channel_mask": V1_CONTROL_SPECS[name].channel_mask,
                    }
                    for name in REQUIRED_V1_CONTROLS
                },
            },
            "dataset": dataset_manifest,
            "reference_stack": {
                "episode_binder": (
                    "ordered-rgb-intervention-v3-matched-action-target-controls"
                ),
                "active": "support-induced-live-intervention-eig-v2",
                "language": "one-token-per-opaque-operational-slot-v1",
                "program_executor": "perceptual-role+diagnostic+closed-loop-scheme-v2",
                "process_config": asdict(ProcessConfig()),
                "commitment": reference_commitment,
            },
            "claim_boundary": {
                "evidence_for": (
                    "transparent finite reference components learning outcome-free "
                    "object/process distinctions and a bounded compositional motor code"
                ),
                "not_yet_a_certificate_for": (
                    "arbitrary or adversarial candidates, one frozen end-to-end model, "
                    "open-world language, social convention, sim-to-real transfer, "
                    "consciousness, or universal meaning"
                ),
            },
        },
    )
    metrics = GroundingCertificate(scope, axes, V1_REQUIRED_AXES)
    ledger = V1EvidenceLedger(
        BENCHMARK_ID,
        seed,
        blocks,
        dataset_hash,
        tuple(
            record
            for axis in V1_REQUIRED_AXES
            for record in records_by_axis[axis]
        ),
        tuple(control_trials),
        tuple(controls),
    )
    return V1BenchmarkResult(metrics, ledger, preregistered)


def run_v1_benchmark(
    *,
    seed: int = 3,
    blocks: int = DEFAULT_BLOCKS,
    thresholds: V1Thresholds | None = None,
    evaluator_secret: bytes | None = None,
) -> V1BenchmarkResult:
    """Run an explicitly non-sealed development/reference configuration."""

    return _run_v1_benchmark(
        seed=seed,
        blocks=blocks,
        thresholds=thresholds,
        evaluator_secret=evaluator_secret,
        _evaluation_master_commitment=None,
    )


def run_v1_benchmark_from_master_seed(
    evaluation_master_seed: bytes,
    *,
    blocks: int = DEFAULT_BLOCKS,
    thresholds: V1Thresholds | None = None,
) -> V1BenchmarkResult:
    """Derive every public world seed and audit key from one sealed master.

    The caller is responsible for generating this master only after the source
    and candidate commitments are frozen, withholding it until the ledger is
    terminal, and revealing it afterwards for reproduction.  Domain-separated
    HMAC derivations prevent the target-commitment key from being the world
    generator itself.
    """

    if not isinstance(evaluation_master_seed, bytes) or len(evaluation_master_seed) < 32:
        raise ValueError("evaluation_master_seed must contain at least 32 private bytes")
    seed_material = hmac.new(
        evaluation_master_seed,
        b"ground-zero-v1/evaluation-world-master",
        sha256,
    ).digest()
    target_key = hmac.new(
        evaluation_master_seed,
        b"ground-zero-v1/target-commitment-key",
        sha256,
    ).digest()
    derived_seed = int.from_bytes(seed_material[:8], "big")
    return _run_v1_benchmark(
        seed=derived_seed,
        blocks=blocks,
        thresholds=thresholds,
        evaluator_secret=target_key,
        _evaluation_master_commitment=sha256(evaluation_master_seed).hexdigest(),
    )


def _summary(result: V1BenchmarkResult) -> dict[str, Any]:
    return {
        "kind": "reference-component-self-test",
        "claim_grade": CLAIM_GRADE,
        "benchmark_id": BENCHMARK_ID,
        "self_test_passed": result.passed,
        "passed": result.passed,
        "certified": False,
        "certificate_eligible": False,
        "preregistered_default": result.preregistered_default,
        "required_controls_complete": result.controls_complete,
        "control_trial_count": len(result.ledger.control_trials),
        "inference": "one-sided-exact-clopper-pearson",
        "score": result.metrics.score,
        "scope_hash": result.metrics.scope.scope_hash,
        "metric_bundle_hash": result.metrics.certificate_hash,
        "ledger_hash": result.ledger.ledger_hash,
        "axes": {
            axis.name: {
                "estimate": axis.performance.estimate,
                "lower_bound_95": axis.performance.lower_bound,
                "coverage": axis.coverage.estimate,
                "coverage_lower_bound_95": axis.coverage.lower_bound,
                "passed": axis.passed,
            }
            for axis in result.metrics.axes
        },
        "controls": {
            control.name: {
                "leakage_estimate": control.leakage.estimate,
                "leakage_upper_bound_95": control.leakage.upper_bound,
                "maximum_leakage": control.maximum_leakage,
                "coverage_estimate": control.coverage.estimate,
                "coverage_upper_bound_95": control.coverage.upper_bound,
                "minimum_coverage": control.minimum_coverage,
                "low_leakage_proven": control.low_leakage_proven,
                "insufficient_coverage_proven": (
                    control.insufficient_coverage_proven
                ),
                "insufficient_as_full_grounder": (
                    control.insufficient_as_full_grounder
                ),
                "rejection_reasons": list(control.rejection_reasons),
                "rejected_as_grounder": control.rejected_as_grounder,
            }
            for control in result.ledger.controls
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="grounding-kernel-v1",
        description=(
            "Run the claim-grade-2 GroundZero-v1 reference self-test; this does not "
            "issue an adversarial grounding certificate."
        ),
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS)
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--shortcut-ceiling", type=float, default=0.20)
    parser.add_argument("--chance-tolerance", type=float, default=0.25)
    parser.add_argument("--full-json", action="store_true")
    parser.add_argument("--compact", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.seed < 0:
        parser.error("--seed must be non-negative")
    if arguments.blocks < 8:
        parser.error("--blocks must be at least 8")
    try:
        thresholds = V1Thresholds(
            performance=arguments.threshold,
            answer_coverage=arguments.threshold,
            shortcut_leakage_ceiling=arguments.shortcut_ceiling,
            chance_tolerance=arguments.chance_tolerance,
        )
    except ValueError as exc:
        parser.error(str(exc))
    result = run_v1_benchmark(
        seed=arguments.seed,
        blocks=arguments.blocks,
        thresholds=thresholds,
    )
    document = result.to_dict() if arguments.full_json else _summary(result)
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
    "CLAIM_GRADE",
    "DEFAULT_BLOCKS",
    "REQUIRED_V1_CONTROLS",
    "V1_CONTROL_SPECS",
    "V1_REQUIRED_AXES",
    "V1BenchmarkResult",
    "V1ControlSpec",
    "V1ControlTrial",
    "V1ControlResult",
    "V1EvidenceLedger",
    "V1Thresholds",
    "V1TrialRecord",
    "main",
    "run_v1_benchmark",
    "run_v1_benchmark_from_master_seed",
]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
