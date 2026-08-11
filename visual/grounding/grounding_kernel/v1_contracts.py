"""Learner-visible contracts for interactive GroundZero-v1 sessions.

These values intentionally contain no evaluator enums, latent object/event
identifiers, semantic outcome labels, seeds, or oracle handles.  V1 agents see
raw observations, opaque motor commands, opaque utterance tokens, bounded
scalar task feedback, and authoritative remaining budgets.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from .contracts import Action, Observation


PROTOCOL_VERSION = "grounding-session/1"
SENSOR_SCHEMA_RGB_U8 = "rgb-u8-v1"
ACTION_SCHEMA_OPAQUE_MOTOR = "opaque-motor-target-vector-v1"
ALLOWED_SENSOR_SCHEMAS = frozenset({SENSOR_SCHEMA_RGB_U8})
ALLOWED_ACTION_SCHEMAS = frozenset({ACTION_SCHEMA_OPAQUE_MOTOR})
MAX_UTTERANCE_TOKENS = 256
MAX_TRACE_TRANSITIONS = 4_096
MAX_BELIEF_CANDIDATES = 4_096
MAX_PUBLIC_TICK = (1 << 63) - 1


def _integer(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _finite_float(
    value: object,
    field: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise ValueError(f"{field} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{field} must be at most {maximum}")
    return result


def _probability(value: object, field: str) -> float:
    return _finite_float(value, field, minimum=0.0, maximum=1.0)


def _validate_observation(value: object, field: str) -> Observation:
    if not isinstance(value, Observation):
        raise TypeError(f"{field} must be an Observation")
    if not 0 <= value.tick <= MAX_PUBLIC_TICK:
        raise ValueError(f"{field}.tick must lie in [0, {MAX_PUBLIC_TICK}]")
    return value


class SessionPhase(str, Enum):
    """Public protocol phase; names describe permissions, not world semantics."""

    SUPPORT = "support"
    ACQUISITION = "acquisition"
    QUERY = "query"
    FROZEN_QUERY = "frozen_query"
    REVISION = "revision"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class SessionManifest:
    """Public schemas and authoritative resource limits for one session."""

    protocol_version: str
    sensor_schema: str
    action_schema: str
    support_episode_budget: int
    intervention_cost_budget: float
    query_budget: int

    def __post_init__(self) -> None:
        if self.protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must equal {PROTOCOL_VERSION!r}")
        if not isinstance(self.sensor_schema, str) or not isinstance(self.action_schema, str):
            raise TypeError("sensor_schema and action_schema must be strings")
        if self.sensor_schema not in ALLOWED_SENSOR_SCHEMAS:
            raise ValueError(
                f"sensor_schema must be one of {sorted(ALLOWED_SENSOR_SCHEMAS)!r}"
            )
        if self.action_schema not in ALLOWED_ACTION_SCHEMAS:
            raise ValueError(
                f"action_schema must be one of {sorted(ALLOWED_ACTION_SCHEMAS)!r}"
            )
        object.__setattr__(
            self,
            "support_episode_budget",
            _integer(self.support_episode_budget, "support_episode_budget", minimum=0),
        )
        object.__setattr__(
            self,
            "intervention_cost_budget",
            _finite_float(
                self.intervention_cost_budget,
                "intervention_cost_budget",
                minimum=0.0,
            ),
        )
        object.__setattr__(
            self,
            "query_budget",
            _integer(self.query_budget, "query_budget", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class Utterance:
    """A non-empty sequence of fresh opaque surface-token integers."""

    tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        tokens = tuple(_integer(token, "tokens") for token in self.tokens)
        if not tokens:
            raise ValueError("utterances cannot be empty")
        if len(tokens) > MAX_UTTERANCE_TOKENS:
            raise ValueError(f"utterances cannot exceed {MAX_UTTERANCE_TOKENS} tokens")
        object.__setattr__(self, "tokens", tokens)


def _feedback(value: object | None) -> float | None:
    if value is None:
        return None
    return _finite_float(value, "scalar_feedback", minimum=-1.0, maximum=1.0)


@dataclass(frozen=True, slots=True)
class PublicTransition:
    """Raw sensory transition with generic feedback and no semantic outcome code."""

    before: Observation
    action: Action
    after: Observation
    scalar_feedback: float | None = None

    def __post_init__(self) -> None:
        _validate_observation(self.before, "before")
        _validate_observation(self.after, "after")
        if not isinstance(self.action, Action):
            raise TypeError("action must be an Action")
        if self.after.tick != self.before.tick + 1:
            raise ValueError("public transition ticks must be consecutive")
        object.__setattr__(self, "scalar_feedback", _feedback(self.scalar_feedback))

    @property
    def pixels_changed(self) -> bool:
        return not np.array_equal(self.before.pixels, self.after.pixels)


@dataclass(frozen=True, slots=True)
class PublicTrace:
    """Continuous sequence of learner-visible v1 transitions."""

    initial: Observation
    transitions: tuple[PublicTransition, ...] = ()

    def __post_init__(self) -> None:
        _validate_observation(self.initial, "initial")
        transitions = tuple(self.transitions)
        if len(transitions) > MAX_TRACE_TRANSITIONS:
            raise ValueError(
                f"public traces cannot exceed {MAX_TRACE_TRANSITIONS} transitions"
            )
        previous = self.initial
        for transition in transitions:
            if not isinstance(transition, PublicTransition):
                raise TypeError("transitions must contain only PublicTransition values")
            if transition.before != previous:
                raise ValueError("public trace contains a discontinuous transition")
            previous = transition.after
        object.__setattr__(self, "transitions", transitions)

    @property
    def current(self) -> Observation:
        return self.transitions[-1].after if self.transitions else self.initial

    @property
    def total_feedback(self) -> float:
        return sum(
            transition.scalar_feedback or 0.0 for transition in self.transitions
        )

    @property
    def has_feedback(self) -> bool:
        return any(
            transition.scalar_feedback is not None for transition in self.transitions
        )

    def feedback_stripped(self) -> "PublicTrace":
        """Return the same sensory/action trace with every feedback value hidden."""

        return PublicTrace(
            self.initial,
            tuple(
                PublicTransition(
                    transition.before,
                    transition.action,
                    transition.after,
                    None,
                )
                for transition in self.transitions
            ),
        )

    def append(self, transition: PublicTransition) -> "PublicTrace":
        if transition.before != self.current:
            raise ValueError("transition does not continue this public trace")
        return PublicTrace(self.initial, self.transitions + (transition,))


@dataclass(frozen=True, slots=True)
class PublicTurn:
    """One phase-labelled learner input with authoritative remaining cost."""

    turn_id: int
    phase: SessionPhase
    observation: Observation
    utterance: Utterance | None = None
    ostensive_pixel_cue: tuple[int, int, int, int] | None = None
    scalar_feedback: float | None = None
    remaining_cost: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "turn_id", _integer(self.turn_id, "turn_id", minimum=0))
        object.__setattr__(self, "phase", SessionPhase(self.phase))
        _validate_observation(self.observation, "observation")
        if self.utterance is not None and not isinstance(self.utterance, Utterance):
            raise TypeError("utterance must be an Utterance or None")
        cue = self.ostensive_pixel_cue
        if cue is not None:
            if self.phase is not SessionPhase.SUPPORT:
                raise ValueError("ostensive_pixel_cue is allowed only during support")
            if not isinstance(cue, (tuple, list)) or len(cue) != 4:
                raise TypeError("ostensive_pixel_cue must be an (x0, y0, x1, y1) box")
            box = tuple(_integer(value, "ostensive_pixel_cue", minimum=0) for value in cue)
            x0, y0, x1, y1 = box
            height, width, _channels = self.observation.shape
            if not (x0 < x1 <= width and y0 < y1 <= height):
                raise ValueError("ostensive_pixel_cue must be a non-empty in-frame box")
            object.__setattr__(self, "ostensive_pixel_cue", box)
        feedback = _feedback(self.scalar_feedback)
        if self.phase in {
            SessionPhase.QUERY,
            SessionPhase.FROZEN_QUERY,
            SessionPhase.COMPLETE,
        } and feedback is not None:
            raise ValueError("scalar_feedback is forbidden during sealed query phases")
        object.__setattr__(self, "scalar_feedback", feedback)
        object.__setattr__(
            self,
            "remaining_cost",
            _finite_float(self.remaining_cost, "remaining_cost", minimum=0.0),
        )


@dataclass(frozen=True, slots=True)
class ExperimentDecision:
    action: Action | None
    unknown_probability: float

    def __post_init__(self) -> None:
        if self.action is not None and not isinstance(self.action, Action):
            raise TypeError("action must be an Action or None")
        object.__setattr__(
            self,
            "unknown_probability",
            _probability(self.unknown_probability, "unknown_probability"),
        )


@dataclass(frozen=True, slots=True)
class DescriptionDecision:
    utterance: Utterance | None
    unknown_probability: float

    def __post_init__(self) -> None:
        if self.utterance is not None and not isinstance(self.utterance, Utterance):
            raise TypeError("utterance must be an Utterance or None")
        object.__setattr__(
            self,
            "unknown_probability",
            _probability(self.unknown_probability, "unknown_probability"),
        )


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: Action | None
    unknown_probability: float

    def __post_init__(self) -> None:
        if self.action is not None and not isinstance(self.action, Action):
            raise TypeError("action must be an Action or None")
        object.__setattr__(
            self,
            "unknown_probability",
            _probability(self.unknown_probability, "unknown_probability"),
        )


@dataclass(frozen=True, slots=True)
class BeliefDecision:
    """Calibrated categorical belief with an explicit open-set mass."""

    candidate_probabilities: tuple[tuple[int, float], ...]
    unknown_probability: float

    def __post_init__(self) -> None:
        pairs: list[tuple[int, float]] = []
        for candidate, probability in self.candidate_probabilities:
            pairs.append(
                (
                    _integer(candidate, "candidate"),
                    _probability(probability, "candidate_probability"),
                )
            )
        if len(pairs) > MAX_BELIEF_CANDIDATES:
            raise ValueError(
                f"beliefs cannot exceed {MAX_BELIEF_CANDIDATES} candidates"
            )
        if len({candidate for candidate, _probability_value in pairs}) != len(pairs):
            raise ValueError("belief candidates must be unique")
        pairs.sort(key=lambda item: item[0])
        unknown = _probability(self.unknown_probability, "unknown_probability")
        total = unknown + sum(probability for _candidate, probability in pairs)
        if not isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("candidate probabilities plus unknown_probability must sum to 1")
        object.__setattr__(self, "candidate_probabilities", tuple(pairs))
        object.__setattr__(self, "unknown_probability", unknown)

    @property
    def distribution(self) -> Mapping[int, float]:
        return MappingProxyType(dict(self.candidate_probabilities))


@runtime_checkable
class InteractiveGrounder(Protocol):
    """Candidate-side behavior required by the future isolated v1 runner."""

    def begin(self, manifest: SessionManifest) -> None: ...

    def observe_support(self, turn: PublicTurn) -> None: ...

    def choose_experiment(self, turn: PublicTurn) -> ExperimentDecision: ...

    def describe(self, trace: PublicTrace) -> DescriptionDecision: ...

    def begin_goal(self, utterance: Utterance, observation: Observation) -> None: ...

    def act(self, observation: Observation) -> ActionDecision: ...

    def report_belief(self, candidates: Sequence[int]) -> BeliefDecision: ...

    def freeze(self) -> None: ...


__all__ = [
    "ACTION_SCHEMA_OPAQUE_MOTOR",
    "ALLOWED_ACTION_SCHEMAS",
    "ALLOWED_SENSOR_SCHEMAS",
    "ActionDecision",
    "BeliefDecision",
    "DescriptionDecision",
    "ExperimentDecision",
    "InteractiveGrounder",
    "MAX_BELIEF_CANDIDATES",
    "MAX_PUBLIC_TICK",
    "MAX_TRACE_TRANSITIONS",
    "MAX_UTTERANCE_TOKENS",
    "PROTOCOL_VERSION",
    "PublicTrace",
    "PublicTransition",
    "PublicTurn",
    "SessionManifest",
    "SessionPhase",
    "SENSOR_SCHEMA_RGB_U8",
    "Utterance",
]
