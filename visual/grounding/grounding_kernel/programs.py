"""Compile grounded operational referents into executable opaque motor programs."""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .contracts import Action, Observation
from .language import GroundedReferent
from .v1_contracts import PublicTrace, PublicTransition


def _pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{field} must be an integer pair")
    result: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise TypeError(f"{field} must be an integer pair")
        result.append(item)
    return result[0], result[1]


@dataclass(frozen=True, slots=True)
class TargetTrack:
    """Learner-visible image-space targets for a bounded motor episode."""

    targets: tuple[tuple[int, int], ...]

    def __post_init__(self) -> None:
        targets = tuple(_pair(target, "target") for target in self.targets)
        if not targets:
            raise ValueError("a target track cannot be empty")
        object.__setattr__(self, "targets", targets)

    @classmethod
    def from_episode(cls, episode: object) -> "TargetTrack":
        transitions = tuple(getattr(episode, "transitions"))
        return cls(tuple(tuple(step.action.target) for step in transitions))


@dataclass(frozen=True, slots=True)
class ActionScheme:
    """Opaque action-code/vector sequence, independent of concrete targets."""

    steps: tuple[tuple[int, tuple[int, int]], ...]

    def __post_init__(self) -> None:
        converted: list[tuple[int, tuple[int, int]]] = []
        for code, vector in self.steps:
            if isinstance(code, bool) or not isinstance(code, int):
                raise TypeError("action scheme codes must be integers")
            converted.append((code, _pair(vector, "action vector")))
        if not converted:
            raise ValueError("an action scheme cannot be empty")
        object.__setattr__(self, "steps", tuple(converted))

    @classmethod
    def from_episode(cls, episode: object) -> "ActionScheme":
        transitions = tuple(getattr(episode, "transitions"))
        return cls(
            tuple((int(step.action.code), tuple(step.action.vector)) for step in transitions)
        )


@dataclass(frozen=True, slots=True)
class ProgramSchema:
    """Opaque operational slot IDs needed to compile a two-part program."""

    target_type_id: Hashable
    scheme_type_id: Hashable

    def __post_init__(self) -> None:
        for value in (self.target_type_id, self.scheme_type_id):
            try:
                hash(value)
            except TypeError as exc:
                raise TypeError("program schema IDs must be hashable") from exc
        if self.target_type_id == self.scheme_type_id:
            raise ValueError("target and scheme operational slots must be distinct")


@dataclass(frozen=True, slots=True)
class CompiledProgram:
    actions: tuple[Action, ...]

    def __post_init__(self) -> None:
        actions = tuple(self.actions)
        if not actions or not all(isinstance(action, Action) for action in actions):
            raise ValueError("compiled programs require at least one Action")
        object.__setattr__(self, "actions", actions)


@runtime_checkable
class PublicMotorEnvironment(Protocol):
    def reset(self) -> Observation: ...

    def step(self, action: Action) -> object: ...


class GroundedProgramExecutor:
    """Use a parsed referent to produce and execute a public motor sequence."""

    def __init__(self, schema: ProgramSchema) -> None:
        if not isinstance(schema, ProgramSchema):
            raise TypeError("schema must be a ProgramSchema")
        self._schema = schema

    @property
    def schema(self) -> ProgramSchema:
        return self._schema

    def compile(self, referent: GroundedReferent) -> CompiledProgram:
        if not isinstance(referent, GroundedReferent):
            raise TypeError("compile expects a GroundedReferent")
        target_meaning = referent.meaning_for(self._schema.target_type_id)
        scheme_meaning = referent.meaning_for(self._schema.scheme_type_id)
        if target_meaning is None or scheme_meaning is None:
            raise ValueError("referent does not contain the program schema's required slots")
        if not isinstance(target_meaning.value, TargetTrack):
            raise TypeError("target operational value must be a TargetTrack")
        if not isinstance(scheme_meaning.value, ActionScheme):
            raise TypeError("scheme operational value must be an ActionScheme")
        if len(target_meaning.value.targets) < len(scheme_meaning.value.steps):
            raise ValueError("target track is shorter than the action scheme")
        actions = tuple(
            Action(code, target_meaning.value.targets[index], vector)
            for index, (code, vector) in enumerate(scheme_meaning.value.steps)
        )
        return CompiledProgram(actions)

    def execute(
        self,
        environment: PublicMotorEnvironment,
        referent: GroundedReferent,
    ) -> PublicTrace:
        """Execute without retaining any legacy outcome-code channel."""

        if not isinstance(environment, PublicMotorEnvironment):
            raise TypeError("environment must expose reset() and step(Action)")
        program = self.compile(referent)
        initial = environment.reset()
        previous = initial
        transitions: list[PublicTransition] = []
        for action in program.actions:
            raw = environment.step(action)
            before = getattr(raw, "before")
            after = getattr(raw, "after")
            if before != previous:
                raise RuntimeError("environment returned a discontinuous transition")
            transitions.append(PublicTransition(before, action, after, None))
            previous = after
        return PublicTrace(initial, tuple(transitions))


def build_program_referent(
    schema: ProgramSchema,
    target_track: TargetTrack,
    action_scheme: ActionScheme,
) -> GroundedReferent:
    """Build a language referent from two learner-visible operational records."""

    from .language import OperationalMeaning

    return GroundedReferent(
        (
            OperationalMeaning(schema.target_type_id, target_track),
            OperationalMeaning(schema.scheme_type_id, action_scheme),
        )
    )


__all__ = [
    "ActionScheme",
    "CompiledProgram",
    "GroundedProgramExecutor",
    "ProgramSchema",
    "PublicMotorEnvironment",
    "TargetTrack",
    "build_program_referent",
]
