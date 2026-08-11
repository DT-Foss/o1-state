"""Learner-side visual target binding and observation-conditioned execution.

The policy sees raw RGB observations, opaque action schemas and an already
fitted learner-side episode binder.  It imports no ProcessWorld evaluator
types, semantic enums, object identifiers, renderer variants or seeds.

Visually identical candidates cannot be selected from a single passive frame.
Full mode therefore performs public reset/step probes, asks the binder about
the resulting outcome- and feedback-free traces, and only then executes the
scheme on the uniquely supported candidate in a freshly reset episode.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from .contracts import Action, Observation
from .v1_contracts import PublicTrace, PublicTransition


Pixel = tuple[int, int]
Vector = tuple[int, int]
PolicyMode = Literal["full", "target_only", "action_only", "no_sensor"]


def _integer(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field} must be an integer pair")
    return (_integer(value[0], field), _integer(value[1], field))


def _pixels(observation: object) -> np.ndarray:
    if not hasattr(observation, "pixels"):
        raise AttributeError("observation is missing public pixels")
    frame = np.asarray(getattr(observation, "pixels"))
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("public observations must be RGB arrays")
    if frame.dtype != np.uint8:
        raise TypeError("public observations must use uint8 pixels")
    return frame


def _foreground(frame: np.ndarray) -> np.ndarray:
    normalized = frame.astype(np.float64) / 255.0
    border = np.concatenate(
        (normalized[0], normalized[-1], normalized[:, 0], normalized[:, -1]),
        axis=0,
    )
    background = np.median(border, axis=0)
    return np.linalg.norm(normalized - background, axis=2) > 0.08


@dataclass(frozen=True, slots=True)
class _Component:
    center: tuple[float, float]
    descriptor: tuple[float, float, float, float]

    @property
    def target(self) -> Pixel:
        return (int(round(self.center[0])), int(round(self.center[1])))


def _components(observation: object) -> tuple[_Component, ...]:
    frame = _pixels(observation)
    pending = np.array(_foreground(frame), dtype=np.bool_, copy=True)
    height, width = pending.shape
    result: list[_Component] = []
    while np.any(pending):
        first = np.argwhere(pending)[0]
        stack = [(int(first[0]), int(first[1]))]
        pending[stack[0]] = False
        points: list[tuple[int, int]] = []
        while stack:
            y, x = stack.pop()
            points.append((y, x))
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and pending[ny, nx]:
                    pending[ny, nx] = False
                    stack.append((ny, nx))
        ys = np.fromiter((point[0] for point in points), dtype=np.float64)
        xs = np.fromiter((point[1] for point in points), dtype=np.float64)
        box_width = int(np.max(xs) - np.min(xs) + 1)
        box_height = int(np.max(ys) - np.min(ys) + 1)
        area = len(points)
        descriptor = (
            area / float(height * width),
            box_width / float(width),
            box_height / float(height),
            area / float(box_width * box_height),
        )
        result.append(
            _Component(
                (float(np.mean(xs)), float(np.mean(ys))),
                descriptor,
            )
        )
    return tuple(sorted(result, key=lambda component: (component.center[1], component.center[0])))


def _descriptor_distance(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    # Area and dimensions are small normalized values; fill ratio needs less
    # relative weight so it cannot dominate a shape-size mismatch.
    weights = (8.0, 3.0, 3.0, 0.35)
    return float(
        np.sqrt(
            sum(weight * (a - b) ** 2 for weight, a, b in zip(weights, left, right, strict=True))
        )
    )


def _initial_and_transitions(trace: object) -> tuple[object, tuple[object, ...]]:
    transitions = tuple(getattr(trace, "transitions"))
    if not transitions:
        raise ValueError("target demonstrations must contain at least one transition")
    initial = getattr(trace, "initial", transitions[0].before)
    return initial, transitions


@dataclass(frozen=True, slots=True)
class VisualTargetSelector:
    """Translation-tolerant target prototype learned from raw support frames."""

    component_descriptor: tuple[float, float, float, float]
    match_radius: float
    demonstrated_rank: int | None

    def __post_init__(self) -> None:
        descriptor = tuple(float(value) for value in self.component_descriptor)
        if len(descriptor) != 4 or not all(isfinite(value) for value in descriptor):
            raise ValueError("component_descriptor must contain four finite values")
        if not isfinite(self.match_radius) or self.match_radius <= 0.0:
            raise ValueError("match_radius must be positive and finite")
        if self.demonstrated_rank is not None:
            object.__setattr__(
                self,
                "demonstrated_rank",
                _integer(self.demonstrated_rank, "demonstrated_rank"),
            )
            if self.demonstrated_rank < 0:
                raise ValueError("demonstrated_rank must be non-negative")
        object.__setattr__(self, "component_descriptor", descriptor)
        object.__setattr__(self, "match_radius", float(self.match_radius))

    @classmethod
    def from_traces(
        cls,
        traces: Iterable[object],
        *,
        match_radius: float = 0.055,
    ) -> "VisualTargetSelector":
        descriptors: list[tuple[float, float, float, float]] = []
        demonstrations: list[tuple[object, Pixel]] = []
        for trace in traces:
            initial, transitions = _initial_and_transitions(trace)
            target = _pair(getattr(transitions[0].action, "target"), "action target")
            candidates = _components(initial)
            if not candidates:
                raise ValueError("support frame contains no visual components")
            selected = min(
                candidates,
                key=lambda component: (component.target[0] - target[0]) ** 2
                + (component.target[1] - target[1]) ** 2,
            )
            distance = (selected.target[0] - target[0]) ** 2 + (
                selected.target[1] - target[1]
            ) ** 2
            if distance > 8**2:
                raise ValueError("support action target does not bind a visible component")
            descriptors.append(selected.descriptor)
            demonstrations.append((initial, target))
        if not descriptors:
            raise ValueError("at least one target demonstration is required")
        prototype_array = np.mean(np.asarray(descriptors, dtype=np.float64), axis=0)
        prototype = tuple(float(value) for value in prototype_array)
        selector = cls(prototype, match_radius, None)

        ranks: list[int] = []
        for initial, target in demonstrations:
            matches = selector.candidates(initial)
            if not matches:
                raise ValueError("learned prototype rejects its own support target")
            distances = tuple(
                (candidate[0] - target[0]) ** 2 + (candidate[1] - target[1]) ** 2
                for candidate in matches
            )
            rank = int(np.argmin(np.asarray(distances)))
            if distances[rank] > 8**2:
                raise ValueError("support target is not among prototype matches")
            ranks.append(rank)
        demonstrated_rank = ranks[0] if len(set(ranks)) == 1 else None
        return cls(prototype, match_radius, demonstrated_rank)

    def candidates(self, observation: object) -> tuple[Pixel, ...]:
        matches = tuple(
            component
            for component in _components(observation)
            if _descriptor_distance(component.descriptor, self.component_descriptor)
            <= self.match_radius
        )
        return tuple(component.target for component in matches)

    def track(self, observation: object, previous: Pixel) -> Pixel:
        candidates = self.candidates(observation)
        if not candidates:
            raise LookupError("target prototype is absent from the current observation")
        return min(
            candidates,
            key=lambda target: (target[0] - previous[0]) ** 2
            + (target[1] - previous[1]) ** 2,
        )


@runtime_checkable
class PublicResetStepEnvironment(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: Action) -> object: ...


@runtime_checkable
class EpisodeMembershipModel(Protocol):
    def supports_token(self, episode: object, token: object) -> bool | None: ...


@dataclass(frozen=True, slots=True)
class CandidateEvidence:
    initial_target: Pixel
    prediction: bool | None
    trace: PublicTrace


@dataclass(frozen=True, slots=True)
class PolicyExecution:
    """Immutable public evidence and final execution, if uniquely resolved."""

    resolved: bool
    trace: PublicTrace | None
    evidence: tuple[CandidateEvidence, ...]
    mode: PolicyMode


@dataclass(frozen=True, slots=True)
class BinaryControlAudit:
    """Evaluator score that treats systematic inversion as informative leakage."""

    answered: int
    correct: int
    inverted: int
    total: int

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.answered if self.answered else 0.0

    @property
    def inverted_accuracy(self) -> float:
        return self.inverted / self.answered if self.answered else 0.0

    @property
    def informative(self) -> bool:
        return self.answered > 0 and max(self.correct, self.inverted) > self.answered / 2


def audit_binary_control(
    predictions: Sequence[bool | None],
    expected: Sequence[bool],
) -> BinaryControlAudit:
    """Score correct and complemented predictions separately from abstentions."""

    predicted = tuple(predictions)
    truth = tuple(expected)
    if len(predicted) != len(truth) or not truth:
        raise ValueError("predictions and expected must have the same non-zero length")
    if not all(value is None or isinstance(value, (bool, np.bool_)) for value in predicted):
        raise TypeError("predictions must be Boolean or None")
    if not all(isinstance(value, (bool, np.bool_)) for value in truth):
        raise TypeError("expected values must be Boolean")
    answered = sum(value is not None for value in predicted)
    correct = sum(
        value is not None and bool(value) is bool(label)
        for value, label in zip(predicted, truth, strict=True)
    )
    inverted = sum(
        value is not None and bool(value) is not bool(label)
        for value, label in zip(predicted, truth, strict=True)
    )
    return BinaryControlAudit(answered, correct, inverted, len(truth))


def _schema_steps(schema: object) -> tuple[tuple[int, Vector], ...]:
    raw_steps = tuple(getattr(schema, "steps"))
    steps = tuple(
        (_integer(code, "action code"), _pair(vector, "action vector"))
        for code, vector in raw_steps
    )
    if not steps:
        raise ValueError("action schema cannot be empty")
    return steps


class ObservationConditionedPolicy:
    """Probe, bind and execute an opaque schema on a fresh public world."""

    def __init__(
        self,
        selector: VisualTargetSelector,
        binder: EpisodeMembershipModel,
        token: object,
        action_scheme: object,
        *,
        mode: PolicyMode = "full",
    ) -> None:
        if not isinstance(selector, VisualTargetSelector):
            raise TypeError("selector must be a VisualTargetSelector")
        if not isinstance(binder, EpisodeMembershipModel):
            raise TypeError("binder must expose supports_token(episode, token)")
        try:
            hash(token)
        except TypeError as error:
            raise TypeError("token must be hashable") from error
        if mode not in ("full", "target_only", "action_only", "no_sensor"):
            raise ValueError("mode must be full, target_only, action_only or no_sensor")
        self._selector = selector
        self._binder = binder
        self._token = token
        self._steps = _schema_steps(action_scheme)
        self.mode = mode

    @property
    def selector(self) -> VisualTargetSelector:
        return self._selector

    @property
    def action_steps(self) -> tuple[tuple[int, Vector], ...]:
        return self._steps

    def _run(self, environment: PublicResetStepEnvironment, rank: int) -> PublicTrace:
        initial = environment.reset()
        candidates = self._selector.candidates(initial)
        if not 0 <= rank < len(candidates):
            raise LookupError("selected visual candidate is absent after reset")
        target = candidates[rank]
        previous = initial
        transitions: list[PublicTransition] = []
        for index, (code, vector) in enumerate(self._steps):
            if index:
                target = self._selector.track(previous, target)
            action = Action(code, target, vector)
            raw = environment.step(action)
            before = getattr(raw, "before")
            after = getattr(raw, "after")
            if before != previous:
                raise RuntimeError("environment returned a discontinuous transition")
            transitions.append(PublicTransition(before, action, after, None))
            previous = after
        return PublicTrace(initial, tuple(transitions))

    def execute(self, environment: PublicResetStepEnvironment) -> PolicyExecution:
        if not isinstance(environment, PublicResetStepEnvironment):
            raise TypeError("environment must expose reset() and step(Action)")
        if self.mode in ("action_only", "no_sensor"):
            # Opaque action identities alone provide no image-space referent.
            return PolicyExecution(False, None, (), self.mode)

        initial = environment.reset()
        candidates = self._selector.candidates(initial)
        if not candidates:
            return PolicyExecution(False, None, (), self.mode)
        if self.mode == "target_only":
            rank = self._selector.demonstrated_rank
            if rank is None or rank >= len(candidates):
                return PolicyExecution(False, None, (), self.mode)
            return PolicyExecution(True, self._run(environment, rank), (), self.mode)

        evidence: list[CandidateEvidence] = []
        for rank, target in enumerate(candidates):
            trace = self._run(environment, rank).feedback_stripped()
            prediction = self._binder.supports_token(trace, self._token)
            if prediction is not None and not isinstance(prediction, (bool, np.bool_)):
                raise TypeError("binder predictions must be Boolean or None")
            evidence.append(CandidateEvidence(target, prediction, trace))
        supported = [
            rank for rank, item in enumerate(evidence) if item.prediction is True
        ]
        if len(supported) != 1:
            return PolicyExecution(False, None, tuple(evidence), self.mode)
        final = self._run(environment, supported[0]).feedback_stripped()
        return PolicyExecution(True, final, tuple(evidence), self.mode)


__all__ = [
    "BinaryControlAudit",
    "CandidateEvidence",
    "EpisodeMembershipModel",
    "ObservationConditionedPolicy",
    "PolicyExecution",
    "PolicyMode",
    "PublicResetStepEnvironment",
    "VisualTargetSelector",
    "audit_binary_control",
]
