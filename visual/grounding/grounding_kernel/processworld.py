"""Sealed finite world for operational object and process concepts.

This module deliberately puts noun-like affordances and verb-like events into
the same undifferentiated opaque concept channel.  Learners receive raw RGB
frames, opaque integer action/outcome/concept codes and scalar feedback.  They
never receive English meanings, parts of speech, object identifiers, seeds or
the evaluator codebook.

The two central constructions are:

* visually matched structures that differ only in protective affordance; and
* passively identical movers that diverge only after a perturbation because
  one process is self-sustaining and the other externally driven.

Consequently, the event analogue of ``run`` is an ordered interventional
trajectory schema, not a property of a string or a single image.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType

import numpy as np

from .contracts import Action, Observation, Transition


Vector = tuple[int, int]
Pixel = tuple[int, int]
PROCESS_MOTOR_VECTORS: tuple[Vector, ...] = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


class ProcessActionKind(str, Enum):
    """Evaluator-only meanings behind freshly sampled action codes."""

    ADVANCE = "advance"
    PERTURB = "perturb"
    ENTER = "enter"
    HAZARD = "hazard"


class ProcessOutcomeKind(str, Enum):
    """Evaluator-only meanings behind opaque outcome codes."""

    MISSED = "missed"
    ADVANCED = "advanced"
    PERTURBED = "perturbed"
    ENTERED = "entered"
    PROTECTED = "protected"
    DAMAGED = "damaged"
    APPLIED = "applied"


class ProcessConceptKind(str, Enum):
    """Evaluator-only concept meanings; no POS distinction crosses the boundary."""

    SHELTER = "shelter"
    MOVING = "moving"
    RUNNING = "running"
    CONTEXT_INSIDE = "context_inside"
    NEGATIVE_CONTROL = "negative_control"


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class ProcessConfig:
    """Evaluator-side finite geometry and interaction budget."""

    frame_size: int = 72
    max_steps: int = 32
    displacement: int = 5

    def __post_init__(self) -> None:
        for name in ("frame_size", "max_steps", "displacement"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if self.frame_size < 64:
            raise ValueError("frame_size must be at least 64")
        if self.max_steps < 4:
            raise ValueError("max_steps must be at least 4")
        if not 2 <= self.displacement <= 8:
            raise ValueError("displacement must lie in [2, 8]")


@dataclass(frozen=True, slots=True)
class ProcessManifest:
    """Complete learner-visible metadata, containing no semantic mapping."""

    observation_shape: tuple[int, int, int]
    action_codes: tuple[int, ...]
    concept_codes: tuple[int, ...]
    motor_vectors: tuple[Vector, ...]
    max_steps: int

    def __post_init__(self) -> None:
        shape = tuple(_strict_int(value, "observation_shape") for value in self.observation_shape)
        actions = tuple(_strict_int(value, "action_codes") for value in self.action_codes)
        concepts = tuple(_strict_int(value, "concept_codes") for value in self.concept_codes)
        vectors = tuple(
            (_strict_int(vector[0], "motor_vectors"), _strict_int(vector[1], "motor_vectors"))
            for vector in self.motor_vectors
        )
        if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
            raise ValueError("observation_shape must be positive RGB dimensions")
        if len(set(actions)) != len(actions) or len(set(concepts)) != len(concepts):
            raise ValueError("opaque code sets must be unique")
        object.__setattr__(self, "observation_shape", shape)
        object.__setattr__(self, "action_codes", actions)
        object.__setattr__(self, "concept_codes", concepts)
        object.__setattr__(self, "motor_vectors", vectors)
        object.__setattr__(self, "max_steps", _strict_int(self.max_steps, "max_steps"))


@dataclass(frozen=True, slots=True)
class NonSensorTranscript:
    """Projection that removes every raw sensory value and spatial target.

    Image-space targets are deliberately omitted: they are sensory bindings,
    not a nonsensory shortcut. ``pixel_change_marginal`` retains only the
    aggregate number of changed/unchanged transitions, never their location or
    temporal placement.
    """

    length: int
    steps: tuple[tuple[int, Vector, int, int, float], ...]
    pixel_change_marginal: tuple[int, int]


@dataclass(frozen=True, slots=True)
class PublicEpisode:
    """Immutable learner-visible episode with no concept token or oracle data."""

    transitions: tuple[Transition, ...]
    scalar_feedback: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        transitions = tuple(self.transitions)
        feedback = tuple(float(value) for value in self.scalar_feedback)
        if not feedback:
            feedback = (0.0,) * len(transitions)
        if len(feedback) != len(transitions):
            raise ValueError("scalar_feedback must align with transitions")
        if not all(np.isfinite(value) for value in feedback):
            raise ValueError("scalar_feedback must be finite")
        object.__setattr__(self, "transitions", transitions)
        object.__setattr__(self, "scalar_feedback", feedback)

    @property
    def observations(self) -> tuple[Observation, ...]:
        if not self.transitions:
            return ()
        return (self.transitions[0].before,) + tuple(step.after for step in self.transitions)

    def reordered(self, order: Sequence[int]) -> "PublicEpisode":
        """Return a diagnostic temporal permutation without repairing continuity."""

        indices = tuple(_strict_int(index, "order") for index in order)
        if sorted(indices) != list(range(len(self.transitions))):
            raise ValueError("order must be a permutation of every transition index")
        return PublicEpisode(
            tuple(self.transitions[index] for index in indices),
            tuple(self.scalar_feedback[index] for index in indices),
        )

    def non_sensor_transcript(self) -> NonSensorTranscript:
        changed = tuple(step.pixels_changed for step in self.transitions)
        steps = tuple(
            (
                int(step.action.code),
                tuple(step.action.vector),
                int(step.outcome_code),
                int(step.after.tick - step.before.tick),
                float(feedback),
            )
            for step, feedback in zip(self.transitions, self.scalar_feedback, strict=True)
        )
        return NonSensorTranscript(
            length=len(steps),
            steps=steps,
            pixel_change_marginal=(changed.count(False), changed.count(True)),
        )


@dataclass(frozen=True, slots=True)
class OstensiveRecord:
    """Opaque concept supervision over an otherwise token-free public episode."""

    token: int
    episode: PublicEpisode
    task_feedback: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "token", _strict_int(self.token, "token"))
        object.__setattr__(self, "task_feedback", bool(self.task_feedback))


def _component_count(mask: np.ndarray) -> int:
    """Count 4-connected components in a small Boolean mask using NumPy storage."""

    pending = np.array(mask, dtype=np.bool_, copy=True)
    count = 0
    height, width = pending.shape
    while np.any(pending):
        start = np.argwhere(pending)[0]
        stack = [(int(start[0]), int(start[1]))]
        pending[stack[0]] = False
        count += 1
        while stack:
            y, x = stack.pop()
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < height and 0 <= nx < width and pending[ny, nx]:
                    pending[ny, nx] = False
                    stack.append((ny, nx))
    return count


def operational_signature(episode: PublicEpisode) -> tuple[tuple[object, ...], ...]:
    """Extract a semantic-token-free ordered intervention signature.

    Only learner-visible action/outcome codes and translation-invariant change
    geometry are used.  Exact colours, background texture, absolute target
    positions and concept tokens are excluded, making the signature stable
    across the renderer variants generated from one world.
    """

    if not isinstance(episode, PublicEpisode):
        raise TypeError("operational_signature expects a PublicEpisode")
    signature: list[tuple[object, ...]] = []
    for transition, feedback in zip(
        episode.transitions, episode.scalar_feedback, strict=True
    ):
        changed = np.any(transition.before.pixels != transition.after.pixels, axis=2)
        count = int(np.count_nonzero(changed))
        if count:
            ys, xs = np.nonzero(changed)
            box = (int(np.max(xs) - np.min(xs) + 1), int(np.max(ys) - np.min(ys) + 1))
            components = _component_count(changed)
        else:
            box = (0, 0)
            components = 0
        signature.append(
            (
                int(transition.action.code),
                tuple(transition.action.vector),
                int(transition.outcome_code),
                round(float(feedback), 12),
                bool(count),
                count,
                box,
                components,
            )
        )
    return tuple(signature)


@dataclass(frozen=True, slots=True)
class ProcessOracleSnapshot:
    """Privileged evaluator snapshot; never returned by :class:`ProcessWorld`."""

    tick: int
    protective_structure: int
    nonprotective_structure: int
    self_sustaining_mover: int
    external_mover: int
    probe_inside: int | None
    negative_control: bool


@dataclass(frozen=True, slots=True)
class _Codebook:
    actions: Mapping[ProcessActionKind, int]
    outcomes: Mapping[ProcessOutcomeKind, int]
    concepts: Mapping[ProcessConceptKind, int]

    def __post_init__(self) -> None:
        actions = {ProcessActionKind(key): int(value) for key, value in self.actions.items()}
        outcomes = {ProcessOutcomeKind(key): int(value) for key, value in self.outcomes.items()}
        concepts = {ProcessConceptKind(key): int(value) for key, value in self.concepts.items()}
        values = tuple(actions.values()) + tuple(outcomes.values()) + tuple(concepts.values())
        if set(actions) != set(ProcessActionKind) or set(outcomes) != set(ProcessOutcomeKind):
            raise ValueError("codebook does not cover the full intervention vocabulary")
        if set(concepts) != set(ProcessConceptKind) or len(set(values)) != len(values):
            raise ValueError("codebook does not cover unique concept codes")
        object.__setattr__(self, "actions", MappingProxyType(actions))
        object.__setattr__(self, "outcomes", MappingProxyType(outcomes))
        object.__setattr__(self, "concepts", MappingProxyType(concepts))

    @classmethod
    def generate(cls, seed: int) -> "_Codebook":
        rng = np.random.default_rng(np.random.SeedSequence((seed, 0xA11CE)))
        count = len(ProcessActionKind) + len(ProcessOutcomeKind) + len(ProcessConceptKind)
        values = [
            int(value)
            for value in rng.choice(899_999_999, size=count, replace=False) + 100_000_000
        ]
        action_end = len(ProcessActionKind)
        outcome_end = action_end + len(ProcessOutcomeKind)
        return cls(
            dict(zip(ProcessActionKind, values[:action_end], strict=True)),
            dict(zip(ProcessOutcomeKind, values[action_end:outcome_end], strict=True)),
            dict(zip(ProcessConceptKind, values[outcome_end:], strict=True)),
        )

    def _decode(self, mapping: Mapping[Enum, int], code: int, kind: str) -> Enum:
        code = _strict_int(code, "code")
        for meaning, opaque in mapping.items():
            if opaque == code:
                return meaning
        raise KeyError(f"unknown {kind} code")


@dataclass(slots=True)
class _Structure:
    object_id: int
    center: Pixel
    protective: bool


@dataclass(slots=True)
class _Mover:
    object_id: int
    center: Pixel
    self_sustaining: bool
    perturbed: bool = False
    stalled: bool = False


class _ProcessEngine:
    def __init__(
        self,
        config: ProcessConfig,
        codebook: _Codebook,
        semantic_rng: np.random.Generator,
        render_rng: np.random.Generator,
        world_variant: int = 0,
    ) -> None:
        self.config = config
        self.codebook = codebook
        camera = render_rng.integers(-2, 3, size=2)
        self._camera = (int(camera[0]), int(camera[1]))
        self._background = tuple(int(value) for value in render_rng.integers(18, 58, size=3))
        self._line_color = tuple(int(value) for value in render_rng.integers(70, 112, size=3))
        self._structure_color = tuple(
            int(value) for value in render_rng.integers(110, 225, size=3)
        )
        self._mover_color = tuple(int(value) for value in render_rng.integers(85, 238, size=3))
        self._probe_color = tuple(int(value) for value in render_rng.integers(120, 245, size=3))
        self._shield_color = tuple(int(value) for value in render_rng.integers(150, 256, size=3))
        self._damage_color = tuple(int(value) for value in render_rng.integers(70, 210, size=3))
        if self._shield_color == self._probe_color:
            self._shield_color = (
                (self._shield_color[0] + 1) % 256,
                self._shield_color[1],
                self._shield_color[2],
            )
        if self._damage_color == self._probe_color:
            self._damage_color = (
                (self._damage_color[0] + 1) % 256,
                self._damage_color[1],
                self._damage_color[2],
            )
        self._roof_style = int(render_rng.integers(0, 3))
        self._texture_phase = int(render_rng.integers(0, 2))

        structure_centers = [(18, 21), (54, 21)]
        protective_side = int(semantic_rng.integers(0, 2))
        instance_base = world_variant * 1_000
        self._structures = [
            _Structure(instance_base + index, center, index == protective_side)
            for index, center in enumerate(structure_centers)
        ]
        mover_rows = [47, 58]
        self_side = int(semantic_rng.integers(0, 2))
        self._movers = [
            _Mover(instance_base + 100 + index, (12, row), index == self_side)
            for index, row in enumerate(mover_rows)
        ]
        self._negative_control = bool(semantic_rng.integers(0, 2))
        self._probe = (36, 35)
        self._probe_inside: int | None = None
        self._hazard_state = 0
        self._generic_hazard_outcome = False
        self._tick = 0
        self._initial = self._state_copy()
        self._transitions: tuple[Transition, ...] = ()

    @property
    def manifest(self) -> ProcessManifest:
        size = self.config.frame_size
        return ProcessManifest(
            observation_shape=(size, size, 3),
            action_codes=tuple(sorted(self.codebook.actions.values())),
            concept_codes=tuple(sorted(self.codebook.concepts.values())),
            motor_vectors=PROCESS_MOTOR_VECTORS,
            max_steps=self.config.max_steps,
        )

    def _state_copy(self) -> tuple[object, ...]:
        return (
            tuple(replace(structure) for structure in self._structures),
            tuple(replace(mover) for mover in self._movers),
            self._negative_control,
            self._probe,
            self._probe_inside,
            self._hazard_state,
            self._generic_hazard_outcome,
        )

    def _restore(self, state: tuple[object, ...]) -> None:
        structures, movers, negative, probe, inside, hazard, generic = state
        self._structures = [replace(value) for value in structures]  # type: ignore[arg-type]
        self._movers = [replace(value) for value in movers]  # type: ignore[arg-type]
        self._negative_control = bool(negative)
        self._probe = tuple(probe)  # type: ignore[arg-type,assignment]
        self._probe_inside = inside  # type: ignore[assignment]
        self._hazard_state = int(hazard)
        self._generic_hazard_outcome = bool(generic)

    def fork(self, *, flip_negative: bool = False) -> "_ProcessEngine":
        clone = object.__new__(_ProcessEngine)
        for name in (
            "config",
            "codebook",
            "_camera",
            "_background",
            "_line_color",
            "_structure_color",
            "_mover_color",
            "_probe_color",
            "_shield_color",
            "_damage_color",
            "_roof_style",
            "_texture_phase",
        ):
            setattr(clone, name, getattr(self, name))
        clone._structures = [replace(value) for value in self._structures]
        clone._movers = [replace(value) for value in self._movers]
        clone._negative_control = (
            not self._negative_control if flip_negative else self._negative_control
        )
        clone._probe = self._probe
        clone._probe_inside = self._probe_inside
        clone._hazard_state = self._hazard_state
        clone._generic_hazard_outcome = self._generic_hazard_outcome
        clone._tick = self._tick
        clone._initial = self._initial
        clone._transitions = self._transitions
        return clone

    def reset(self) -> Observation:
        self._restore(self._initial)
        self._tick = 0
        self._transitions = ()
        return self.observe()

    def observe(self) -> Observation:
        return Observation(
            self._render(),
            self._tick,
            self._tick >= self.config.max_steps,
        )

    def episode(self, feedback: Iterable[float] = ()) -> PublicEpisode:
        return PublicEpisode(self._transitions, tuple(feedback))

    def _decode_action(self, code: int) -> ProcessActionKind:
        result = self.codebook._decode(self.codebook.actions, code, "action")
        if not isinstance(result, ProcessActionKind):
            raise AssertionError("action decoder returned wrong enum type")
        return result

    def step(self, action: Action) -> Transition:
        if not isinstance(action, Action):
            raise TypeError("step expects an Action")
        if self._tick >= self.config.max_steps:
            raise RuntimeError("episode is terminal")
        size = self.config.frame_size
        if not (0 <= action.target[0] < size and 0 <= action.target[1] < size):
            raise ValueError("target lies outside the frame")
        if action.vector not in PROCESS_MOTOR_VECTORS:
            raise ValueError("unsupported motor vector")
        try:
            kind = self._decode_action(action.code)
        except KeyError as error:
            raise ValueError("unknown opaque action code") from error
        before = self.observe()
        outcome = self._apply(kind, action.target)
        self._tick += 1
        after = self.observe()
        transition = Transition(before, action, after, self.codebook.outcomes[outcome])
        self._transitions += (transition,)
        return transition

    def _screen(self, point: Pixel) -> Pixel:
        return (point[0] + self._camera[0], point[1] + self._camera[1])

    def _nearest_structure(self, target: Pixel) -> _Structure | None:
        matches = [
            (self._distance(target, self._screen(structure.center)), structure)
            for structure in self._structures
        ]
        distance, structure = min(matches, key=lambda item: item[0])
        return structure if distance <= 12**2 else None

    def _nearest_mover(self, target: Pixel) -> _Mover | None:
        matches = [
            (self._distance(target, self._screen(mover.center)), mover) for mover in self._movers
        ]
        distance, mover = min(matches, key=lambda item: item[0])
        return mover if distance <= 7**2 else None

    @staticmethod
    def _distance(left: Pixel, right: Pixel) -> int:
        return (left[0] - right[0]) ** 2 + (left[1] - right[1]) ** 2

    def _apply(self, kind: ProcessActionKind, target: Pixel) -> ProcessOutcomeKind:
        if kind in (ProcessActionKind.ADVANCE, ProcessActionKind.PERTURB):
            mover = self._nearest_mover(target)
            if mover is None:
                return ProcessOutcomeKind.MISSED
            if kind is ProcessActionKind.PERTURB:
                mover.perturbed = True
                return ProcessOutcomeKind.PERTURBED
            if mover.perturbed and not mover.self_sustaining:
                mover.stalled = True
            else:
                x, y = mover.center
                mover.center = (x + self.config.displacement, y)
                mover.perturbed = False
            return ProcessOutcomeKind.ADVANCED

        structure = self._nearest_structure(target)
        if structure is None:
            return ProcessOutcomeKind.MISSED
        if kind is ProcessActionKind.ENTER:
            self._probe = (structure.center[0], structure.center[1] + 4)
            self._probe_inside = structure.object_id
            self._hazard_state = 0
            return ProcessOutcomeKind.ENTERED
        if kind is ProcessActionKind.HAZARD:
            inside_target = self._probe_inside == structure.object_id
            protected = inside_target and structure.protective
            self._hazard_state = 1 if protected else -1
            if self._generic_hazard_outcome:
                return ProcessOutcomeKind.APPLIED
            return ProcessOutcomeKind.PROTECTED if protected else ProcessOutcomeKind.DAMAGED
        raise AssertionError(f"unhandled process action: {kind}")

    def structure_center(self, object_id: int) -> Pixel:
        for structure in self._structures:
            if structure.object_id == object_id:
                return self._screen(structure.center)
        raise KeyError(f"unknown structure id: {object_id}")

    def mover_center(self, object_id: int) -> Pixel:
        for mover in self._movers:
            if mover.object_id == object_id:
                return self._screen(mover.center)
        raise KeyError(f"unknown mover id: {object_id}")

    def _render(self) -> np.ndarray:
        size = self.config.frame_size
        frame = np.empty((size, size, 3), dtype=np.uint8)
        frame[...] = self._background
        yy, xx = np.indices((size, size))
        checker = ((xx + yy + self._texture_phase) % 2 == 0)[..., None]
        adjustment = np.where(checker, 3, 0).astype(np.uint8)
        frame = np.clip(frame.astype(np.int16) + adjustment, 0, 255).astype(np.uint8)
        horizon = 39 + self._camera[1]
        frame[max(0, horizon) : min(size, horizon + 1), :] = self._line_color
        for structure in self._structures:
            self._draw_structure(frame, self._screen(structure.center))
        for mover in self._movers:
            self._draw_mover(frame, self._screen(mover.center))
        self._draw_probe(frame, self._screen(self._probe))
        return frame

    def _draw_structure(self, frame: np.ndarray, center: Pixel) -> None:
        cx, cy = center
        color = self._structure_color
        # Both structures share exactly the same topology and nuisance style.
        frame[cy - 8 : cy + 9, cx - 8 : cx - 6] = color
        frame[cy - 8 : cy + 9, cx + 6 : cx + 8] = color
        if self._roof_style == 0:
            frame[cy - 9 : cy - 7, cx - 8 : cx + 8] = color
        elif self._roof_style == 1:
            for offset in range(9):
                y = cy - 10 + abs(offset - 4) // 2
                x = cx - 8 + 2 * offset
                frame[y : y + 2, x : min(frame.shape[1], x + 3)] = color
        else:
            frame[cy - 10 : cy - 8, cx - 7 : cx + 7] = color
            frame[cy - 8 : cy - 6, cx - 9 : cx + 9] = color

    def _draw_mover(self, frame: np.ndarray, center: Pixel) -> None:
        cx, cy = center
        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= 3**2
        frame[mask] = self._mover_color

    def _draw_probe(self, frame: np.ndarray, center: Pixel) -> None:
        cx, cy = center
        if self._hazard_state == 0:
            frame[cy - 1 : cy + 2, cx - 1 : cx + 2] = self._probe_color
            return

        # Protection and damage are raw visual consequences, not merely
        # opaque outcome-code differences.  Both glyphs use nine foreground
        # pixels, have the same 5x5 extent and induce the same aggregate
        # changed-pixel count; only their spatial arrangement differs.  This
        # keeps the context pair transcript-matched while making pixels
        # causally necessary.
        if self._hazard_state > 0:
            color = self._shield_color
            offsets = (
                (-2, 0),
                (-1, 0),
                (0, 0),
                (1, 0),
                (2, 0),
                (0, -2),
                (0, -1),
                (0, 1),
                (0, 2),
            )
        else:
            color = self._damage_color
            offsets = (
                (-2, -1),
                (-1, -1),
                (1, -1),
                (-1, 1),
                (1, 1),
                (2, 1),
                (-1, -2),
                (0, 0),
                (1, 2),
            )
        for dx, dy in offsets:
            frame[cy + dy, cx + dx] = color


class ProcessWorld:
    """Learner capability: raw pixels and opaque integer channels only."""

    __slots__ = (
        "__manifest_capability",
        "__episode_capability",
        "__observe_capability",
        "__reset_capability",
        "__step_capability",
    )

    def __init__(self, engine: _ProcessEngine) -> None:
        self.__manifest_capability = engine.manifest
        self.__episode_capability = engine.episode
        self.__observe_capability = engine.observe
        self.__reset_capability = engine.reset
        self.__step_capability = engine.step

    @property
    def manifest(self) -> ProcessManifest:
        return self.__manifest_capability

    @property
    def action_codes(self) -> tuple[int, ...]:
        return self.manifest.action_codes

    @property
    def concept_codes(self) -> tuple[int, ...]:
        return self.manifest.concept_codes

    def reset(self) -> Observation:
        return self.__reset_capability()

    def observe(self) -> Observation:
        return self.__observe_capability()

    def step(
        self,
        action: Action | int,
        target: Pixel | None = None,
        vector: Vector = (0, 0),
    ) -> Transition:
        if isinstance(action, Action):
            if target is not None or vector != (0, 0):
                raise TypeError("target/vector must be omitted with an Action record")
            record = action
        else:
            if target is None:
                raise TypeError("target is required with an opaque action code")
            record = Action(action, target, vector)
        return self.__step_capability(record)

    def episode(self) -> PublicEpisode:
        return self.__episode_capability()


@dataclass(frozen=True, slots=True)
class AffordanceCounterfactualSet:
    """Matched public worlds for evaluator-controlled active acquisition.

    Pattern truth and construction parameters are deliberately absent.  The
    evaluator retains the requested ordering; a learner receives only
    independent public worlds, one common image-space target and one common
    sequence of opaque diagnostic actions.
    """

    target: Pixel
    diagnostic_actions: tuple[Action, ...]
    worlds: tuple[ProcessWorld, ...]

    def __post_init__(self) -> None:
        target = (
            _strict_int(self.target[0], "target"),
            _strict_int(self.target[1], "target"),
        )
        actions = tuple(self.diagnostic_actions)
        worlds = tuple(self.worlds)
        if not actions or not all(isinstance(action, Action) for action in actions):
            raise ValueError("diagnostic_actions must contain public Action records")
        if not worlds or not all(isinstance(world, ProcessWorld) for world in worlds):
            raise ValueError("worlds must contain independent ProcessWorld capabilities")
        if any(action.target != target for action in actions):
            raise ValueError("every diagnostic action must use the common target")
        object.__setattr__(self, "target", target)
        object.__setattr__(self, "diagnostic_actions", actions)
        object.__setattr__(self, "worlds", worlds)


class _ProcessOracle:
    """Evaluator-only semantic capability."""

    __slots__ = ("_engine",)

    def __init__(self, engine: _ProcessEngine) -> None:
        self._engine = engine

    def decode_action(self, code: int) -> ProcessActionKind:
        result = self._engine.codebook._decode(self._engine.codebook.actions, code, "action")
        if not isinstance(result, ProcessActionKind):
            raise AssertionError("wrong action enum")
        return result

    def decode_outcome(self, code: int) -> ProcessOutcomeKind:
        result = self._engine.codebook._decode(self._engine.codebook.outcomes, code, "outcome")
        if not isinstance(result, ProcessOutcomeKind):
            raise AssertionError("wrong outcome enum")
        return result

    def decode_concept(self, code: int) -> ProcessConceptKind:
        result = self._engine.codebook._decode(self._engine.codebook.concepts, code, "concept")
        if not isinstance(result, ProcessConceptKind):
            raise AssertionError("wrong concept enum")
        return result

    def encode_concept(self, concept: ProcessConceptKind) -> int:
        return self._engine.codebook.concepts[ProcessConceptKind(concept)]

    def snapshot(self) -> ProcessOracleSnapshot:
        protective = next(value for value in self._engine._structures if value.protective)
        nonprotective = next(value for value in self._engine._structures if not value.protective)
        self_sustaining = next(value for value in self._engine._movers if value.self_sustaining)
        external = next(value for value in self._engine._movers if not value.self_sustaining)
        return ProcessOracleSnapshot(
            tick=self._engine._tick,
            protective_structure=protective.object_id,
            nonprotective_structure=nonprotective.object_id,
            self_sustaining_mover=self_sustaining.object_id,
            external_mover=external.object_id,
            probe_inside=self._engine._probe_inside,
            negative_control=self._engine._negative_control,
        )

    def structure_center(self, object_id: int) -> Pixel:
        return self._engine.structure_center(object_id)

    def mover_center(self, object_id: int) -> Pixel:
        return self._engine.mover_center(object_id)

    def _action(self, kind: ProcessActionKind, target: Pixel) -> Action:
        return Action(self._engine.codebook.actions[kind], target)

    def affordance_counterfactuals(
        self,
        patterns: Sequence[tuple[bool, bool]] = (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ),
        *,
        target_index: int = 0,
    ) -> AffordanceCounterfactualSet:
        """Construct same-render public worlds differing only in latent affordance.

        The returned record carries no requested pattern labels.  Callers that
        evaluate hypotheses retain ``patterns`` outside the learner boundary.
        """

        target_index = _strict_int(target_index, "target_index")
        if target_index not in (0, 1):
            raise ValueError("target_index must be 0 or 1")
        converted: list[tuple[bool, bool]] = []
        for pattern in patterns:
            if not isinstance(pattern, (tuple, list)) or len(pattern) != 2:
                raise TypeError("patterns must contain Boolean pairs")
            if not all(isinstance(value, (bool, np.bool_)) for value in pattern):
                raise TypeError("patterns must contain Boolean pairs")
            converted.append((bool(pattern[0]), bool(pattern[1])))
        if not converted:
            raise ValueError("at least one counterfactual pattern is required")

        target = self._engine.structure_center(
            self._engine._structures[target_index].object_id
        )
        actions = (
            self._action(ProcessActionKind.ENTER, target),
            self._action(ProcessActionKind.HAZARD, target),
        )
        worlds: list[ProcessWorld] = []
        for pattern in converted:
            clone = self._engine.fork()
            clone.reset()
            for structure, protective in zip(clone._structures, pattern, strict=True):
                structure.protective = protective
            clone._initial = clone._state_copy()
            clone.reset()
            worlds.append(ProcessWorld(clone))
        return AffordanceCounterfactualSet(target, actions, tuple(worlds))

    def _structure_episode(
        self,
        object_id: int,
        *,
        protective: bool | None = None,
    ) -> PublicEpisode:
        engine = self._engine.fork()
        engine.reset()
        if protective is not None:
            selected = next(
                structure for structure in engine._structures if structure.object_id == object_id
            )
            selected.protective = bool(protective)
            engine._initial = engine._state_copy()
            engine.reset()
        target = engine.structure_center(object_id)
        engine.step(Action(engine.codebook.actions[ProcessActionKind.ENTER], target))
        engine.step(Action(engine.codebook.actions[ProcessActionKind.HAZARD], target))
        return engine.episode()

    def affordance_pair(self) -> tuple[OstensiveRecord, OstensiveRecord]:
        snapshot = self.snapshot()
        token = self.encode_concept(ProcessConceptKind.SHELTER)
        # Counterfactualize one and the same visible structure.  Positive and
        # negative records therefore have byte-identical public actions and
        # initial pixels; only the hidden affordance and its raw consequence
        # differ.
        anchor = snapshot.protective_structure
        return (
            OstensiveRecord(
                token,
                self._structure_episode(anchor, protective=True),
                True,
            ),
            OstensiveRecord(
                token,
                self._structure_episode(anchor, protective=False),
                False,
            ),
        )

    def _movement_episode(
        self,
        object_id: int,
        *,
        perturb: bool,
        self_sustaining: bool | None = None,
    ) -> PublicEpisode:
        engine = self._engine.fork()
        engine.reset()
        if self_sustaining is not None:
            selected = next(
                mover for mover in engine._movers if mover.object_id == object_id
            )
            selected.self_sustaining = bool(self_sustaining)
            selected.perturbed = False
            selected.stalled = False
            engine._initial = engine._state_copy()
            engine.reset()
        advance = engine.codebook.actions[ProcessActionKind.ADVANCE]
        perturb_code = engine.codebook.actions[ProcessActionKind.PERTURB]
        initial_x, initial_y = engine.mover_center(object_id)
        # One fixed open-loop target lies within reach both before and after
        # each possible displacement.  It prevents a tracker/oracle from
        # writing the observed process outcome back into future Action.target
        # coordinates.
        target = (initial_x + engine.config.displacement, initial_y)
        engine.step(Action(advance, target))
        if perturb:
            engine.step(Action(perturb_code, target))
        engine.step(Action(advance, target))
        engine.step(Action(advance, target))
        return engine.episode()

    def passive_process_pair(self) -> tuple[PublicEpisode, PublicEpisode]:
        snapshot = self.snapshot()
        return (
            self._movement_episode(snapshot.self_sustaining_mover, perturb=False),
            self._movement_episode(snapshot.external_mover, perturb=False),
        )

    def target_role_pair(self) -> tuple[PublicEpisode, PublicEpisode]:
        """Return evaluator-curated demonstrations of the two visible actors.

        These episodes deliberately have different public targets: their only
        purpose is to ground two target-role words to distinct referents.  They
        are not positive/negative evidence for a process concept; causal
        process twins must come from :meth:`process_pair`, whose complete
        public action transcripts are exactly matched.
        """

        return self.passive_process_pair()

    def movement_pair(self) -> tuple[OstensiveRecord, OstensiveRecord]:
        token = self.encode_concept(ProcessConceptKind.MOVING)
        left, right = self.passive_process_pair()
        return (OstensiveRecord(token, left, True), OstensiveRecord(token, right, True))

    def process_pair(self) -> tuple[OstensiveRecord, OstensiveRecord]:
        snapshot = self.snapshot()
        token = self.encode_concept(ProcessConceptKind.RUNNING)
        anchor = snapshot.self_sustaining_mover
        return (
            OstensiveRecord(
                token,
                self._movement_episode(
                    anchor,
                    perturb=True,
                    self_sustaining=True,
                ),
                True,
            ),
            OstensiveRecord(
                token,
                self._movement_episode(
                    anchor,
                    perturb=True,
                    self_sustaining=False,
                ),
                False,
            ),
        )

    def context_pair(self) -> tuple[OstensiveRecord, OstensiveRecord]:
        snapshot = self.snapshot()
        token = self.encode_concept(ProcessConceptKind.CONTEXT_INSIDE)
        records: list[OstensiveRecord] = []
        structure_id = snapshot.protective_structure
        for inside in (True, False):
            engine = self._engine.fork()
            engine.reset()
            structure = next(
                value for value in engine._structures if value.object_id == structure_id
            )
            if inside:
                engine._probe = (structure.center[0], structure.center[1] + 4)
                engine._probe_inside = structure.object_id
            else:
                engine._probe = (structure.center[0], structure.center[1] + 15)
                engine._probe_inside = None
            engine._generic_hazard_outcome = True
            engine._initial = engine._state_copy()
            target = engine.structure_center(structure.object_id)
            engine.step(Action(engine.codebook.actions[ProcessActionKind.HAZARD], target))
            records.append(OstensiveRecord(token, engine.episode(), inside))
        return (records[0], records[1])

    def negative_control_pair(self) -> tuple[OstensiveRecord, OstensiveRecord]:
        snapshot = self.snapshot()
        token = self.encode_concept(ProcessConceptKind.NEGATIVE_CONTROL)
        object_id = snapshot.self_sustaining_mover
        left_engine = self._engine.fork()
        right_engine = self._engine.fork(flip_negative=True)
        left_engine.reset()
        right_engine.reset()
        # Reset restores the shared initial state, so flip again after reset.
        right_engine._negative_control = not left_engine._negative_control
        for engine in (left_engine, right_engine):
            code = engine.codebook.actions[ProcessActionKind.ADVANCE]
            engine.step(Action(code, engine.mover_center(object_id)))
            engine.step(Action(code, engine.mover_center(object_id)))
        return (
            OstensiveRecord(token, left_engine.episode(), left_engine._negative_control),
            OstensiveRecord(token, right_engine.episode(), right_engine._negative_control),
        )

    def negative_control_invariant(self) -> bool:
        left, right = self.negative_control_pair()
        return (
            left.task_feedback != right.task_feedback
            and left.episode == right.episode
            and operational_signature(left.episode) == operational_signature(right.episode)
        )

    def examples(self, *, include_negative_control: bool = False) -> tuple[OstensiveRecord, ...]:
        records = (
            *self.affordance_pair(),
            *self.movement_pair(),
            *self.process_pair(),
            *self.context_pair(),
        )
        return records + self.negative_control_pair() if include_negative_control else records

    def matches_concept(self, episode: PublicEpisode, concept: ProcessConceptKind) -> bool:
        """Evaluator predicate over an ordered public trajectory."""

        concept = ProcessConceptKind(concept)
        actions = tuple(self.decode_action(step.action.code) for step in episode.transitions)
        outcomes = tuple(self.decode_outcome(step.outcome_code) for step in episode.transitions)
        changed = tuple(step.pixels_changed for step in episode.transitions)
        if concept is ProcessConceptKind.SHELTER:
            return actions == (ProcessActionKind.ENTER, ProcessActionKind.HAZARD) and outcomes[
                -1:
            ] == (ProcessOutcomeKind.PROTECTED,)
        if concept is ProcessConceptKind.MOVING:
            return (
                len(actions) >= 2
                and all(action is ProcessActionKind.ADVANCE for action in actions)
                and sum(changed) >= 2
            )
        if concept is ProcessConceptKind.RUNNING:
            return (
                actions
                == (
                    ProcessActionKind.ADVANCE,
                    ProcessActionKind.PERTURB,
                    ProcessActionKind.ADVANCE,
                    ProcessActionKind.ADVANCE,
                )
                and changed == (True, False, True, True)
            )
        if concept is ProcessConceptKind.CONTEXT_INSIDE:
            raise ValueError("context truth requires the held-out oracle scene relation")
        if concept is ProcessConceptKind.NEGATIVE_CONTROL:
            raise ValueError("negative-control truth is interventionally unidentifiable")
        raise AssertionError(f"unhandled concept: {concept}")


class ProcessHarness:
    """Evaluator-owned pair; pass only ``agent`` or public records to learners."""

    __slots__ = ("agent", "oracle")

    def __init__(
        self,
        seed: int,
        config: ProcessConfig | None = None,
        *,
        renderer_variant: int = 0,
        world_variant: int = 0,
    ) -> None:
        seed = _strict_int(seed, "seed")
        renderer_variant = _strict_int(renderer_variant, "renderer_variant")
        world_variant = _strict_int(world_variant, "world_variant")
        if seed < 0 or renderer_variant < 0 or world_variant < 0:
            raise ValueError("seed, renderer_variant and world_variant must be non-negative")
        selected = config if config is not None else ProcessConfig()
        if not isinstance(selected, ProcessConfig):
            raise TypeError("config must be a ProcessConfig")
        codebook = _Codebook.generate(seed)
        semantic_rng = np.random.default_rng(
            np.random.SeedSequence((seed, world_variant, 0x5E1A))
        )
        render_rng = np.random.default_rng(
            np.random.SeedSequence((seed, renderer_variant, 0xBADC0DE))
        )
        engine = _ProcessEngine(
            selected,
            codebook,
            semantic_rng,
            render_rng,
            world_variant,
        )
        self.agent = ProcessWorld(engine)
        self.oracle = _ProcessOracle(engine)


def audit_process_agent(agent: ProcessWorld) -> tuple[str, ...]:
    """Detect accidental public evaluator capabilities on a learner object."""

    forbidden = {
        "seed",
        "oracle",
        "codebook",
        "objects",
        "object_ids",
        "decode_action",
        "decode_outcome",
        "decode_concept",
        "snapshot",
        "latent",
        "part_of_speech",
        "pos",
    }
    return tuple(sorted(name for name in dir(agent) if name in forbidden))
