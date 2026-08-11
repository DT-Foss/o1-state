"""Immutable data contracts for the sealed grounding microworld.

The learner-facing records in this module deliberately contain only pixels,
opaque integers and motor coordinates.  Semantic enums and oracle records are
used by the evaluator; an :class:`~grounding_kernel.microworld.Microworld`
instance never returns them.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from types import MappingProxyType
from typing import Mapping

import numpy as np


Pixel = tuple[int, int]
Vector = tuple[int, int]


def _integer(value: object, field: str) -> int:
    """Return a strict Python integer, rejecting booleans and floats."""

    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    return int(value)


def _pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field} must be a pair of integers")
    return (_integer(value[0], f"{field}[0]"), _integer(value[1], f"{field}[1]"))


class ActionKind(str, Enum):
    """Evaluator-only meanings behind the freshly permuted motor codes."""

    PUSH = "push"
    LIFT = "lift"
    MAGNET = "magnet"
    INSERT = "insert"
    TOGGLE = "toggle"


class OutcomeKind(str, Enum):
    """Evaluator-only meanings behind opaque transition outcome codes."""

    MISS = "miss"
    NO_EFFECT = "no_effect"
    BLOCKED = "blocked"
    MOVED = "moved"
    LIFTED = "lifted"
    LOWERED = "lowered"
    ATTRACTED = "attracted"
    INSERTED = "inserted"
    MISMATCH = "mismatch"
    ACTIVATED = "activated"
    DEACTIVATED = "deactivated"


class PredicateKind(str, Enum):
    """Evaluator-only predicates represented by fresh opaque symbol codes."""

    MOVABLE = "movable"
    LIFTABLE = "liftable"
    MAGNETIC = "magnetic"
    FITS_SLOT_A = "fits_slot_a"
    FITS_SLOT_B = "fits_slot_b"
    SWITCHABLE = "switchable"
    NEGATIVE_CONTROL = "negative_control"


@dataclass(frozen=True, slots=True)
class Action:
    """A learner-issued primitive action with no semantic name.

    ``target`` is an image-space ``(x, y)`` coordinate. ``vector`` is one of
    the five finite motor vectors declared by the environment manifest.  All
    action codes share this shape so their argument arity does not disclose
    their hidden permutation.
    """

    code: int
    target: Pixel
    vector: Vector = (0, 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "code", _integer(self.code, "code"))
        object.__setattr__(self, "target", _pair(self.target, "target"))
        object.__setattr__(self, "vector", _pair(self.vector, "vector"))


@dataclass(frozen=True, slots=True, eq=False)
class Observation:
    """Raw immutable RGB frame visible to the learner."""

    pixels: np.ndarray
    tick: int
    terminal: bool = False

    def __post_init__(self) -> None:
        frame = np.asarray(self.pixels)
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("pixels must have shape (height, width, 3)")
        if frame.dtype != np.uint8:
            raise TypeError("pixels must have dtype uint8")
        # Own the memory and freeze it: callers cannot mutate past trajectory
        # records or the simulator through an array view.
        frame = np.array(frame, dtype=np.uint8, order="C", copy=True)
        frame.setflags(write=False)
        object.__setattr__(self, "pixels", frame)
        object.__setattr__(self, "tick", _integer(self.tick, "tick"))
        object.__setattr__(self, "terminal", bool(self.terminal))

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.pixels.shape  # type: ignore[return-value]

    def digest(self) -> str:
        """Stable observation digest for audit logs (not a world/seed hash)."""

        payload = self.pixels.tobytes() + self.tick.to_bytes(8, "big", signed=False)
        payload += bytes((int(self.terminal),))
        return sha256(payload).hexdigest()

    def __eq__(self, other: object) -> bool:
        return (
            isinstance(other, Observation)
            and self.tick == other.tick
            and self.terminal == other.terminal
            and np.array_equal(self.pixels, other.pixels)
        )


@dataclass(frozen=True, slots=True)
class Transition:
    """Immutable learner-visible action/outcome record."""

    before: Observation
    action: Action
    after: Observation
    outcome_code: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "outcome_code", _integer(self.outcome_code, "outcome_code"))
        if self.after.tick != self.before.tick + 1:
            raise ValueError("transition ticks must be consecutive")

    @property
    def pixels_changed(self) -> bool:
        return not np.array_equal(self.before.pixels, self.after.pixels)


@dataclass(frozen=True, slots=True)
class Trajectory:
    """Persistent, immutable trajectory value."""

    initial: Observation
    transitions: tuple[Transition, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        previous = self.initial
        for transition in transitions:
            if transition.before != previous:
                raise ValueError("trajectory contains a discontinuous transition")
            previous = transition.after
        object.__setattr__(self, "transitions", transitions)

    @property
    def current(self) -> Observation:
        return self.transitions[-1].after if self.transitions else self.initial

    def append(self, transition: Transition) -> "Trajectory":
        if transition.before != self.current:
            raise ValueError("transition does not continue this trajectory")
        return Trajectory(self.initial, self.transitions + (transition,))


@dataclass(frozen=True, slots=True)
class AgentManifest:
    """Safe public shape of an episode; contains no seed or semantic map."""

    observation_shape: tuple[int, int, int]
    action_codes: tuple[int, ...]
    symbol_codes: tuple[int, ...]
    motor_vectors: tuple[Vector, ...]
    max_steps: int

    def __post_init__(self) -> None:
        shape = tuple(_integer(v, "observation_shape") for v in self.observation_shape)
        if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
            raise ValueError("observation_shape must be positive RGB dimensions")
        action_codes = tuple(_integer(v, "action_codes") for v in self.action_codes)
        symbol_codes = tuple(_integer(v, "symbol_codes") for v in self.symbol_codes)
        if len(set(action_codes)) != len(action_codes):
            raise ValueError("action codes must be unique")
        if len(set(symbol_codes)) != len(symbol_codes):
            raise ValueError("symbol codes must be unique")
        vectors = tuple(_pair(v, "motor_vectors") for v in self.motor_vectors)
        object.__setattr__(self, "observation_shape", shape)
        object.__setattr__(self, "action_codes", action_codes)
        object.__setattr__(self, "symbol_codes", symbol_codes)
        object.__setattr__(self, "motor_vectors", vectors)
        object.__setattr__(self, "max_steps", _integer(self.max_steps, "max_steps"))


@dataclass(frozen=True, slots=True)
class OracleObjectState:
    """Privileged object state returned only by the evaluator oracle."""

    object_id: int
    position: tuple[int, int]
    predicates: Mapping[PredicateKind, bool]
    lifted: bool
    active: bool
    inserted_slot: int | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "object_id", _integer(self.object_id, "object_id"))
        object.__setattr__(self, "position", _pair(self.position, "position"))
        predicates = {PredicateKind(key): bool(value) for key, value in self.predicates.items()}
        object.__setattr__(self, "predicates", MappingProxyType(predicates))
        if self.inserted_slot is not None:
            object.__setattr__(
                self, "inserted_slot", _integer(self.inserted_slot, "inserted_slot")
            )


@dataclass(frozen=True, slots=True)
class OracleSnapshot:
    """Privileged latent snapshot, never reachable through the agent API."""

    tick: int
    objects: tuple[OracleObjectState, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "tick", _integer(self.tick, "tick"))
        object.__setattr__(self, "objects", tuple(self.objects))


@dataclass(frozen=True, slots=True)
class OpaqueCodebook:
    """Seed-derived evaluator mapping between meanings and opaque integers.

    The seed is consumed and discarded; it is never retained on this object.
    Learners receive sorted tuples of integer codes, not this mapping.
    """

    actions: Mapping[ActionKind, int]
    outcomes: Mapping[OutcomeKind, int]
    predicates: Mapping[PredicateKind, int]

    def __post_init__(self) -> None:
        actions = {ActionKind(k): _integer(v, "action code") for k, v in self.actions.items()}
        outcomes = {
            OutcomeKind(k): _integer(v, "outcome code") for k, v in self.outcomes.items()
        }
        predicates = {
            PredicateKind(k): _integer(v, "predicate code") for k, v in self.predicates.items()
        }
        if set(actions) != set(ActionKind):
            raise ValueError("action codebook must cover every action kind")
        if set(outcomes) != set(OutcomeKind):
            raise ValueError("outcome codebook must cover every outcome kind")
        if set(predicates) != set(PredicateKind):
            raise ValueError("predicate codebook must cover every predicate kind")
        all_codes = tuple(actions.values()) + tuple(outcomes.values()) + tuple(predicates.values())
        if len(set(all_codes)) != len(all_codes):
            raise ValueError("all opaque codes must be globally unique")
        object.__setattr__(self, "actions", MappingProxyType(actions))
        object.__setattr__(self, "outcomes", MappingProxyType(outcomes))
        object.__setattr__(self, "predicates", MappingProxyType(predicates))

    @classmethod
    def from_seed(cls, seed: int) -> "OpaqueCodebook":
        seed = _integer(seed, "seed")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        rng = np.random.default_rng(np.random.SeedSequence((seed, 0xC0DEB00C)))
        count = len(ActionKind) + len(OutcomeKind) + len(PredicateKind)
        # Fixed-width positive integers avoid length/frequency hints while the
        # fresh random injection destroys ordinal semantic information.
        # Passing an integer population keeps sampling O(count); constructing
        # ``arange(899_999_999)`` here would waste several gigabytes merely to
        # choose a few dozen codes.
        codes = rng.choice(899_999_999, size=count, replace=False) + 100_000_000
        values = [int(code) for code in codes]
        a_end = len(ActionKind)
        o_end = a_end + len(OutcomeKind)
        return cls(
            dict(zip(ActionKind, values[:a_end], strict=True)),
            dict(zip(OutcomeKind, values[a_end:o_end], strict=True)),
            dict(zip(PredicateKind, values[o_end:], strict=True)),
        )

    @property
    def action_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self.actions.values()))

    @property
    def outcome_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self.outcomes.values()))

    @property
    def symbol_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self.predicates.values()))

    def decode_action(self, code: int) -> ActionKind:
        value = _integer(code, "code")
        for kind, opaque in self.actions.items():
            if opaque == value:
                return kind
        raise KeyError(f"unknown action code: {value}")

    def decode_outcome(self, code: int) -> OutcomeKind:
        value = _integer(code, "code")
        for kind, opaque in self.outcomes.items():
            if opaque == value:
                return kind
        raise KeyError(f"unknown outcome code: {value}")

    def decode_predicate(self, code: int) -> PredicateKind:
        value = _integer(code, "code")
        for kind, opaque in self.predicates.items():
            if opaque == value:
                return kind
        raise KeyError(f"unknown symbol code: {value}")
