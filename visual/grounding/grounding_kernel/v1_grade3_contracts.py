"""Versioned public contracts for one persistent Grade-3 grounder.

These records are a parallel protocol, not an extension of the frozen
``grounding-session/1`` schema.  Every identifier is opaque.  No seed, oracle
label, semantic enum, likelihood table, latent object ID, or target truth is a
legal field.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol, TypeAlias, runtime_checkable

import numpy as np

from .contracts import Action
from .v1_contracts import (
    ALLOWED_ACTION_SCHEMAS,
    ALLOWED_SENSOR_SCHEMAS,
    BeliefDecision,
    DescriptionDecision,
    MAX_PUBLIC_TICK,
    PublicTrace,
    PublicTurn,
    SessionPhase,
    Utterance,
)


GRADE3_PROTOCOL_VERSION = "grounding-grade3-session/1"
MAX_GRADE3_PROBES = 64
MAX_GRADE3_OPTIONS = 4_096
MAX_GRADE3_CANDIDATES = 4_096
MAX_GRADE3_TRANSITIONS = 8_192
MAX_GRADE3_ACTION_CODES = 4_096
MAX_GRADE3_MOTOR_VECTORS = 4_096
MAX_GRADE3_INTEGER = MAX_PUBLIC_TICK


def _integer(
    value: object,
    field: str,
    *,
    minimum: int = 0,
    maximum: int = MAX_GRADE3_INTEGER,
) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if not minimum <= result <= maximum:
        raise ValueError(f"{field} must lie in [{minimum}, {maximum}]")
    return result


def _number(value: object, field: str, *, minimum: float = 0.0) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result) or result < minimum:
        raise ValueError(f"{field} must be finite and at least {minimum}")
    return result


def _probability(value: object, field: str) -> float:
    result = _number(value, field)
    if result > 1.0:
        raise ValueError(f"{field} must be at most 1")
    return result


def _pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field} must be an integer pair")
    return (
        _integer(value[0], f"{field}[0]", minimum=-MAX_GRADE3_INTEGER),
        _integer(value[1], f"{field}[1]", minimum=-MAX_GRADE3_INTEGER),
    )


def _wire_bounded_action(value: object, field: str) -> Action:
    if not isinstance(value, Action):
        raise TypeError(f"{field} must be Action")
    _integer(value.code, f"{field}.code", minimum=-MAX_GRADE3_INTEGER)
    _pair(value.target, f"{field}.target")
    _pair(value.vector, f"{field}.vector")
    return value


def _feedback_free(trace: object, field: str, *, nonempty: bool = False) -> PublicTrace:
    if not isinstance(trace, PublicTrace):
        raise TypeError(f"{field} must be PublicTrace")
    if trace.has_feedback:
        raise ValueError(f"{field} must be feedback-free")
    if nonempty and not trace.transitions:
        raise ValueError(f"{field} must be nonempty")
    for index, transition in enumerate(trace.transitions):
        _wire_bounded_action(transition.action, f"{field}.transitions[{index}].action")
    return trace


def _in_frame(action: Action, trace_shape: tuple[int, int, int]) -> bool:
    height, width, _channels = trace_shape
    x, y = action.target
    return 0 <= x < width and 0 <= y < height


@dataclass(frozen=True, slots=True)
class Grade3SessionManifest:
    protocol_version: str
    sensor_schema: str
    action_schema: str
    support_record_budget: int
    acquisition_cost_budget: float
    query_budget: int
    motor_action_cost_budget: float
    motor_reset_budget: int

    def __post_init__(self) -> None:
        if self.protocol_version != GRADE3_PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must equal {GRADE3_PROTOCOL_VERSION!r}")
        if not isinstance(self.sensor_schema, str):
            raise TypeError("sensor_schema must be a string")
        if self.sensor_schema not in ALLOWED_SENSOR_SCHEMAS:
            raise ValueError(f"sensor_schema must be one of {sorted(ALLOWED_SENSOR_SCHEMAS)!r}")
        if not isinstance(self.action_schema, str):
            raise TypeError("action_schema must be a string")
        if self.action_schema not in ALLOWED_ACTION_SCHEMAS:
            raise ValueError(f"action_schema must be one of {sorted(ALLOWED_ACTION_SCHEMAS)!r}")
        object.__setattr__(
            self,
            "support_record_budget",
            _integer(self.support_record_budget, "support_record_budget"),
        )
        object.__setattr__(
            self,
            "acquisition_cost_budget",
            _number(self.acquisition_cost_budget, "acquisition_cost_budget"),
        )
        object.__setattr__(
            self, "query_budget", _integer(self.query_budget, "query_budget", minimum=1)
        )
        object.__setattr__(
            self,
            "motor_action_cost_budget",
            _number(self.motor_action_cost_budget, "motor_action_cost_budget"),
        )
        object.__setattr__(
            self,
            "motor_reset_budget",
            _integer(self.motor_reset_budget, "motor_reset_budget"),
        )


class MotorPhase(str, Enum):
    PROBE = "probe"
    EXECUTE = "execute"


class MotorDirective(str, Enum):
    ACT = "act"
    RESET_PROBE = "reset_probe"
    RESET_EXECUTE = "reset_execute"
    COMPLETE = "complete"
    ABSTAIN = "abstain"


@dataclass(frozen=True, slots=True)
class MotorActionSpace:
    action_codes: tuple[int, ...]
    motor_vectors: tuple[tuple[int, int], ...]
    max_trace_steps: int

    def __post_init__(self) -> None:
        codes = tuple(
            _integer(value, "action_code", minimum=-MAX_GRADE3_INTEGER)
            for value in self.action_codes
        )
        vectors = tuple(_pair(value, "motor_vector") for value in self.motor_vectors)
        if not codes or len(set(codes)) != len(codes):
            raise ValueError("action_codes must be nonempty and unique")
        if len(codes) > MAX_GRADE3_ACTION_CODES:
            raise ValueError(f"action_codes cannot exceed {MAX_GRADE3_ACTION_CODES}")
        if not vectors or len(set(vectors)) != len(vectors):
            raise ValueError("motor_vectors must be nonempty and unique")
        if len(vectors) > MAX_GRADE3_MOTOR_VECTORS:
            raise ValueError(f"motor_vectors cannot exceed {MAX_GRADE3_MOTOR_VECTORS}")
        object.__setattr__(self, "action_codes", codes)
        object.__setattr__(self, "motor_vectors", vectors)
        object.__setattr__(
            self,
            "max_trace_steps",
            _integer(
                self.max_trace_steps,
                "max_trace_steps",
                minimum=1,
                maximum=MAX_GRADE3_TRANSITIONS,
            ),
        )

    def permits(self, action: Action, shape: tuple[int, int, int]) -> bool:
        return (
            isinstance(action, Action)
            and action.code in self.action_codes
            and tuple(action.vector) in self.motor_vectors
            and _in_frame(action, shape)
        )


@dataclass(frozen=True, slots=True)
class MotorQuery:
    query_id: int
    scope_id: int
    step_index: int
    utterance: Utterance
    phase: MotorPhase
    completed_probes: tuple[PublicTrace, ...]
    current_trace: PublicTrace
    action_space: MotorActionSpace
    remaining_action_cost: float
    remaining_resets: int

    def __post_init__(self) -> None:
        for field in ("query_id", "scope_id", "step_index"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        if not isinstance(self.utterance, Utterance):
            raise TypeError("utterance must be Utterance")
        object.__setattr__(self, "phase", MotorPhase(self.phase))
        if not isinstance(self.action_space, MotorActionSpace):
            raise TypeError("action_space must be MotorActionSpace")
        probes = tuple(self.completed_probes)
        if len(probes) > MAX_GRADE3_PROBES:
            raise ValueError(f"completed_probes cannot exceed {MAX_GRADE3_PROBES}")
        for index, trace in enumerate(probes):
            _feedback_free(trace, f"completed_probes[{index}]", nonempty=True)
        current = _feedback_free(self.current_trace, "current_trace")
        traces = (*probes, current)
        shapes = {trace.initial.shape for trace in traces}
        if len(shapes) != 1:
            raise ValueError("all motor traces must share one RGB shape")
        total = sum(len(trace.transitions) for trace in traces)
        if total > MAX_GRADE3_TRANSITIONS:
            raise ValueError("motor history exceeds the transition limit")
        for trace in traces:
            if len(trace.transitions) > self.action_space.max_trace_steps:
                raise ValueError("a motor trace exceeds action_space.max_trace_steps")
            for transition in trace.transitions:
                if not self.action_space.permits(transition.action, transition.before.shape):
                    raise ValueError("motor history contains an action outside action_space")
        if MotorPhase(self.phase) is MotorPhase.EXECUTE and not probes:
            raise ValueError("execution requires completed causal probes")
        object.__setattr__(self, "completed_probes", probes)
        object.__setattr__(
            self,
            "remaining_action_cost",
            _number(self.remaining_action_cost, "remaining_action_cost"),
        )
        object.__setattr__(
            self,
            "remaining_resets",
            _integer(self.remaining_resets, "remaining_resets"),
        )


@dataclass(frozen=True, slots=True)
class MotorDecision:
    directive: MotorDirective
    action: Action | None
    unknown_probability: float

    def __post_init__(self) -> None:
        directive = MotorDirective(self.directive)
        if self.action is not None:
            _wire_bounded_action(self.action, "action")
        if (directive is MotorDirective.ACT) is (self.action is None):
            raise ValueError("exactly ACT decisions must carry an action")
        unknown = _probability(self.unknown_probability, "unknown_probability")
        if directive is MotorDirective.ABSTAIN and unknown <= 0.0:
            raise ValueError("ABSTAIN requires positive unknown probability")
        if directive is not MotorDirective.ABSTAIN and unknown != 0.0:
            raise ValueError("non-ABSTAIN directives require zero unknown probability")
        object.__setattr__(self, "directive", directive)
        object.__setattr__(self, "unknown_probability", unknown)


@dataclass(frozen=True, slots=True)
class ProbeEvidence:
    probe_id: int
    trace: PublicTrace

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _integer(self.probe_id, "probe_id"))
        _feedback_free(self.trace, "trace", nonempty=True)


@dataclass(frozen=True, slots=True)
class TraceBeliefQuery:
    query_id: int
    scope_id: int
    problem_id: int
    candidates: tuple[int, ...]
    evidence: tuple[ProbeEvidence, ...]

    def __post_init__(self) -> None:
        for field in ("query_id", "scope_id", "problem_id"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        candidates = tuple(_integer(value, "candidate") for value in self.candidates)
        if not candidates or len(candidates) > MAX_GRADE3_CANDIDATES:
            raise ValueError("candidates must be nonempty and bounded")
        if len(set(candidates)) != len(candidates):
            raise ValueError("candidates must be unique")
        evidence = tuple(self.evidence)
        if not evidence or len(evidence) > MAX_GRADE3_PROBES:
            raise ValueError("evidence must be nonempty and bounded")
        if not all(isinstance(item, ProbeEvidence) for item in evidence):
            raise TypeError("evidence must contain ProbeEvidence")
        if len({item.probe_id for item in evidence}) != len(evidence):
            raise ValueError("probe IDs must be unique")
        if sum(len(item.trace.transitions) for item in evidence) > MAX_GRADE3_TRANSITIONS:
            raise ValueError("belief evidence exceeds the transition limit")
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(
            self, "evidence", tuple(sorted(evidence, key=lambda item: item.probe_id))
        )


@dataclass(frozen=True, slots=True)
class TraceDescriptionQuery:
    query_id: int
    scope_id: int
    evidence: tuple[ProbeEvidence, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "query_id", _integer(self.query_id, "query_id"))
        object.__setattr__(self, "scope_id", _integer(self.scope_id, "scope_id"))
        evidence = tuple(self.evidence)
        if not evidence or len(evidence) > MAX_GRADE3_PROBES:
            raise ValueError("description evidence must be nonempty and bounded")
        if not all(isinstance(item, ProbeEvidence) for item in evidence):
            raise TypeError("evidence must contain ProbeEvidence")
        if len({item.probe_id for item in evidence}) != len(evidence):
            raise ValueError("probe IDs must be unique")
        if sum(len(item.trace.transitions) for item in evidence) > MAX_GRADE3_TRANSITIONS:
            raise ValueError("description evidence exceeds the transition limit")
        object.__setattr__(
            self, "evidence", tuple(sorted(evidence, key=lambda item: item.probe_id))
        )


@dataclass(frozen=True, slots=True)
class OstensiveSupportRecord:
    scope_id: int
    source_id: int
    turn: PublicTurn
    trace: PublicTrace

    def __post_init__(self) -> None:
        object.__setattr__(self, "scope_id", _integer(self.scope_id, "scope_id"))
        object.__setattr__(self, "source_id", _integer(self.source_id, "source_id"))
        if not isinstance(self.turn, PublicTurn):
            raise TypeError("turn must be PublicTurn")
        if self.turn.phase is not SessionPhase.SUPPORT:
            raise ValueError("ostensive support requires a support-phase turn")
        if self.turn.utterance is None:
            raise ValueError("ostensive support requires an opaque utterance")
        _feedback_free(self.trace, "trace")
        if self.turn.observation != self.trace.initial:
            raise ValueError("support turn and trace must share their initial frame")


@dataclass(frozen=True, slots=True)
class CausalSupportRecord:
    scope_id: int
    problem_id: int
    hypothesis_id: int
    probe_id: int
    source_id: int
    trace: PublicTrace

    def __post_init__(self) -> None:
        for field in (
            "scope_id",
            "problem_id",
            "hypothesis_id",
            "probe_id",
            "source_id",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        _feedback_free(self.trace, "trace", nonempty=True)


Grade3SupportRecord: TypeAlias = OstensiveSupportRecord | CausalSupportRecord


@dataclass(frozen=True, slots=True)
class ProbeOption:
    probe_id: int
    cost: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "probe_id", _integer(self.probe_id, "probe_id"))
        object.__setattr__(self, "cost", _number(self.cost, "cost", minimum=1e-15))


@dataclass(frozen=True, slots=True)
class ProbeOffer:
    scope_id: int
    problem_id: int
    step_index: int
    options: tuple[ProbeOption, ...]
    remaining_cost: float

    def __post_init__(self) -> None:
        for field in ("scope_id", "problem_id", "step_index"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        options = tuple(self.options)
        if not options or len(options) > MAX_GRADE3_OPTIONS:
            raise ValueError("probe options must be nonempty and bounded")
        if not all(isinstance(option, ProbeOption) for option in options):
            raise TypeError("options must contain ProbeOption")
        if len({option.probe_id for option in options}) != len(options):
            raise ValueError("probe option IDs must be unique")
        object.__setattr__(self, "options", options)
        object.__setattr__(self, "remaining_cost", _number(self.remaining_cost, "remaining_cost"))


@dataclass(frozen=True, slots=True)
class ProbeDecision:
    probe_id: int | None
    unknown_probability: float

    def __post_init__(self) -> None:
        if self.probe_id is not None:
            object.__setattr__(self, "probe_id", _integer(self.probe_id, "probe_id"))
        unknown = _probability(self.unknown_probability, "unknown_probability")
        if self.probe_id is None and unknown <= 0.0:
            raise ValueError("probe abstention requires positive unknown probability")
        if self.probe_id is not None and unknown != 0.0:
            raise ValueError("a selected probe requires zero unknown probability")
        object.__setattr__(self, "unknown_probability", unknown)


@dataclass(frozen=True, slots=True)
class ProbeResult:
    scope_id: int
    problem_id: int
    probe_id: int
    trace: PublicTrace
    cost: float
    remaining_cost: float

    def __post_init__(self) -> None:
        for field in ("scope_id", "problem_id", "probe_id"):
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        _feedback_free(self.trace, "trace", nonempty=True)
        cost = _number(self.cost, "cost", minimum=1e-15)
        remaining = _number(self.remaining_cost, "remaining_cost")
        object.__setattr__(self, "cost", cost)
        object.__setattr__(self, "remaining_cost", remaining)


@runtime_checkable
class Grade3Grounder(Protocol):
    def begin(self, manifest: Grade3SessionManifest) -> None: ...

    def observe_support(self, record: Grade3SupportRecord) -> None: ...

    def choose_probe(self, offer: ProbeOffer) -> ProbeDecision: ...

    def observe_probe(self, result: ProbeResult) -> None: ...

    def freeze(self) -> None: ...

    def checkpoint_commitment(self) -> str: ...

    def motor(self, query: MotorQuery) -> MotorDecision: ...

    def trace_belief(self, query: TraceBeliefQuery) -> BeliefDecision: ...

    def describe(self, query: TraceDescriptionQuery) -> DescriptionDecision: ...


__all__ = [
    "CausalSupportRecord",
    "GRADE3_PROTOCOL_VERSION",
    "Grade3Grounder",
    "Grade3SessionManifest",
    "Grade3SupportRecord",
    "MAX_GRADE3_CANDIDATES",
    "MAX_GRADE3_ACTION_CODES",
    "MAX_GRADE3_INTEGER",
    "MAX_GRADE3_MOTOR_VECTORS",
    "MAX_GRADE3_OPTIONS",
    "MAX_GRADE3_PROBES",
    "MAX_GRADE3_TRANSITIONS",
    "MotorActionSpace",
    "MotorDecision",
    "MotorDirective",
    "MotorPhase",
    "MotorQuery",
    "OstensiveSupportRecord",
    "ProbeDecision",
    "ProbeEvidence",
    "ProbeOffer",
    "ProbeOption",
    "ProbeResult",
    "TraceBeliefQuery",
    "TraceDescriptionQuery",
]
