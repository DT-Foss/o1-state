"""Deterministic raw-pixel microworld with a capability-separated oracle.

The environment is deliberately small enough to audit exhaustively.  Hidden
properties affect controlled interventions but are not rendered directly.
Fresh seeded integer permutations hide action, outcome and predicate names.
The learner-facing :class:`Microworld` contains no public seed, object ID,
latent state, decoder or oracle capability.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType

import numpy as np

from .contracts import (
    Action,
    ActionKind,
    AgentManifest,
    Observation,
    OpaqueCodebook,
    OracleObjectState,
    OracleSnapshot,
    OutcomeKind,
    Pixel,
    PredicateKind,
    Trajectory,
    Transition,
    Vector,
)


MOTOR_VECTORS: tuple[Vector, ...] = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))

# The invariant checker deliberately quantifies a finite, public probe
# alphabet: every object, every opaque primitive, and every motor vector.  It
# does not claim to enumerate arbitrary image coordinates (which are
# observationally equivalent misses except at an object centre).
INVARIANT_PROBE_VECTORS: tuple[Vector, ...] = MOTOR_VECTORS
DEFAULT_INVARIANT_DEPTH = 2
# Covers depth two for the benchmark's eight-object configuration (40_200
# prefixes) while still failing closed for unexpectedly large worlds/depths.
DEFAULT_INVARIANT_MAX_TRANSITIONS = 100_000
MAX_INVARIANT_DEPTH = 4


@dataclass(frozen=True, slots=True)
class WorldConfig:
    """Evaluator-selected finite world geometry and budget."""

    grid_size: int = 7
    cell_pixels: int = 9
    margin_pixels: int = 3
    object_count: int = 6
    max_steps: int = 64

    def __post_init__(self) -> None:
        values = {
            "grid_size": self.grid_size,
            "cell_pixels": self.cell_pixels,
            "margin_pixels": self.margin_pixels,
            "object_count": self.object_count,
            "max_steps": self.max_steps,
        }
        for field, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError(f"{field} must be an integer")
            object.__setattr__(self, field, int(value))
        if self.grid_size < 7:
            raise ValueError("grid_size must be at least 7")
        if self.cell_pixels < 7:
            raise ValueError("cell_pixels must be at least 7")
        if self.margin_pixels < 2:
            raise ValueError("margin_pixels must be at least 2")
        if self.object_count < 6:
            raise ValueError("object_count must be at least 6")
        capacity = (self.grid_size - 2) * self.grid_size
        if self.object_count > capacity:
            raise ValueError("object_count exceeds the collision-free start capacity")
        if self.max_steps < 1:
            raise ValueError("max_steps must be positive")

    @property
    def frame_size(self) -> int:
        return self.grid_size * self.cell_pixels + 2 * self.margin_pixels


@dataclass(slots=True)
class _LatentObject:
    object_id: int
    position: tuple[int, int]
    movable: bool
    liftable: bool
    magnetic: bool
    slot_profile: int
    switchable: bool
    negative_control: bool
    visual_shape: int
    visual_color: tuple[int, int, int]
    visual_texture: int
    lifted: bool = False
    active: bool = False
    inserted_slot: int | None = None

    def predicate(self, predicate: PredicateKind) -> bool:
        if predicate is PredicateKind.MOVABLE:
            return self.movable
        if predicate is PredicateKind.LIFTABLE:
            return self.liftable
        if predicate is PredicateKind.MAGNETIC:
            return self.magnetic
        if predicate is PredicateKind.FITS_SLOT_A:
            return self.slot_profile == 0
        if predicate is PredicateKind.FITS_SLOT_B:
            return self.slot_profile == 1
        if predicate is PredicateKind.SWITCHABLE:
            return self.switchable
        if predicate is PredicateKind.NEGATIVE_CONTROL:
            return self.negative_control
        raise ValueError(f"unsupported predicate: {predicate!r}")


class _Engine:
    """Privileged mutable state. It is never returned through the agent API."""

    def __init__(
        self,
        config: WorldConfig,
        codebook: OpaqueCodebook,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.codebook = codebook
        self._magnet_cell = (0, config.grid_size // 2)
        self._slot_cells = ((config.grid_size - 1, 1), (config.grid_size - 1, config.grid_size - 2))

        # Nuisance variables are sampled independently from causal properties.
        self._camera_offset = (int(rng.integers(-1, 2)), int(rng.integers(-1, 2)))
        self._background = tuple(int(v) for v in rng.integers(20, 61, size=3))
        self._grid_color = tuple(int(v) for v in rng.integers(65, 96, size=3))
        self._light_gain = int(rng.integers(82, 119)) / 100.0
        texture = rng.integers(-7, 8, size=(config.frame_size, config.frame_size, 1))
        self._background_texture = texture.astype(np.int16)

        palette = rng.integers(75, 236, size=(max(config.object_count, 8), 3))
        property_columns: list[np.ndarray] = []
        for _ in range(4):
            column = np.zeros(config.object_count, dtype=np.bool_)
            column[: config.object_count // 2] = True
            rng.shuffle(column)
            property_columns.append(column)
        slots = np.arange(config.object_count, dtype=np.int64) % 2
        rng.shuffle(slots)
        negative = np.arange(config.object_count, dtype=np.int64) % 2 == 0
        rng.shuffle(negative)

        fixtures = {self._magnet_cell, *self._slot_cells}
        candidates = [
            (x, y)
            for x in range(1, config.grid_size - 1)
            for y in range(config.grid_size)
            if (x, y) not in fixtures
        ]
        order = rng.permutation(len(candidates))[: config.object_count]
        positions = [candidates[int(index)] for index in order]
        objects: list[_LatentObject] = []
        for index, position in enumerate(positions):
            color = np.clip(palette[index].astype(np.float64) * self._light_gain, 50, 245)
            objects.append(
                _LatentObject(
                    object_id=index,
                    position=position,
                    movable=bool(property_columns[0][index]),
                    liftable=bool(property_columns[1][index]),
                    magnetic=bool(property_columns[2][index]),
                    slot_profile=int(slots[index]),
                    switchable=bool(property_columns[3][index]),
                    negative_control=bool(negative[index]),
                    visual_shape=int(rng.integers(0, 4)),
                    visual_color=tuple(int(v) for v in color),
                    visual_texture=int(rng.integers(0, 4)),
                )
            )
        self._initial_objects = tuple(replace(obj) for obj in objects)
        self._objects = objects
        self._tick = 0
        initial = self._observation()
        self._trajectory = Trajectory(initial)

    @property
    def manifest(self) -> AgentManifest:
        size = self.config.frame_size
        return AgentManifest(
            observation_shape=(size, size, 3),
            action_codes=self.codebook.action_codes,
            symbol_codes=self.codebook.symbol_codes,
            motor_vectors=MOTOR_VECTORS,
            max_steps=self.config.max_steps,
        )

    def _fork(self, *, flip_negative_control: bool = False) -> "_Engine":
        clone = object.__new__(_Engine)
        clone.config = self.config
        clone.codebook = self.codebook
        clone._magnet_cell = self._magnet_cell
        clone._slot_cells = self._slot_cells
        clone._camera_offset = self._camera_offset
        clone._background = self._background
        clone._grid_color = self._grid_color
        clone._light_gain = self._light_gain
        clone._background_texture = self._background_texture

        def copied(obj: _LatentObject) -> _LatentObject:
            return replace(
                obj,
                negative_control=(
                    not obj.negative_control if flip_negative_control else obj.negative_control
                ),
            )

        clone._initial_objects = tuple(copied(obj) for obj in self._initial_objects)
        clone._objects = [copied(obj) for obj in self._objects]
        clone._tick = self._tick
        clone._trajectory = self._trajectory
        return clone

    def reset(self) -> Observation:
        self._objects = [replace(obj) for obj in self._initial_objects]
        self._tick = 0
        initial = self._observation()
        self._trajectory = Trajectory(initial)
        return initial

    def observe(self) -> Observation:
        return self._observation()

    def trajectory(self) -> Trajectory:
        return self._trajectory

    def _grid_center(self, position: tuple[int, int], *, lifted: bool = False) -> Pixel:
        x, y = position
        offset_x, offset_y = self._camera_offset
        px = self.config.margin_pixels + x * self.config.cell_pixels + self.config.cell_pixels // 2
        py = self.config.margin_pixels + y * self.config.cell_pixels + self.config.cell_pixels // 2
        if lifted:
            py -= 2
        return (px + offset_x, py + offset_y)

    def _target_index(self, target: Pixel) -> int | None:
        tx, ty = target
        radius = max(2, self.config.cell_pixels // 2)
        candidates: list[tuple[int, int]] = []
        for index, obj in enumerate(self._objects):
            cx, cy = self._grid_center(obj.position, lifted=obj.lifted)
            distance = (tx - cx) ** 2 + (ty - cy) ** 2
            if distance <= radius**2:
                candidates.append((distance, index))
        return min(candidates)[1] if candidates else None

    def object_center(self, object_id: int) -> Pixel:
        obj = self._object_by_id(object_id)
        return self._grid_center(obj.position, lifted=obj.lifted)

    def object_at(self, target: Pixel) -> int | None:
        index = self._target_index(target)
        return None if index is None else self._objects[index].object_id

    def _object_by_id(self, object_id: int) -> _LatentObject:
        if isinstance(object_id, bool) or not isinstance(object_id, (int, np.integer)):
            raise TypeError("object_id must be an integer")
        for obj in self._objects:
            if obj.object_id == int(object_id):
                return obj
        raise KeyError(f"unknown object id: {object_id}")

    def step(self, action: Action) -> Transition:
        before = self._observation()
        outcome = self._advance(action)
        after = self._observation()
        transition = Transition(before, action, after, self.codebook.outcomes[outcome])
        self._trajectory = self._trajectory.append(transition)
        return transition

    def _advance(self, action: Action) -> OutcomeKind:
        """Run the actual transition kernel without constructing audit records.

        The oracle uses this narrow internal hook when it must explore many
        counterfactual branches.  Learner-visible ``step`` wraps the same
        kernel with immutable before/after records and trajectory accounting.
        """

        if not isinstance(action, Action):
            raise TypeError("step expects an Action record")
        if self._tick >= self.config.max_steps:
            raise RuntimeError("episode is terminal; call reset before stepping again")
        width = self.config.frame_size
        if not (0 <= action.target[0] < width and 0 <= action.target[1] < width):
            raise ValueError("target lies outside the observation frame")
        if action.vector not in MOTOR_VECTORS:
            raise ValueError(f"vector must be one of {MOTOR_VECTORS!r}")
        try:
            kind = self.codebook.decode_action(action.code)
        except KeyError as error:
            raise ValueError("unknown opaque action code") from error

        index = self._target_index(action.target)
        if index is None:
            outcome = OutcomeKind.MISS
        else:
            outcome = self._apply(kind, index, action.vector)
        self._tick += 1
        return outcome

    def _apply(self, kind: ActionKind, index: int, vector: Vector) -> OutcomeKind:
        if kind is ActionKind.PUSH:
            return self._push(index, vector)
        if kind is ActionKind.LIFT:
            return self._lift(index)
        if kind is ActionKind.MAGNET:
            return self._magnet(index)
        if kind is ActionKind.INSERT:
            return self._insert(index, vector)
        if kind is ActionKind.TOGGLE:
            return self._toggle(index)
        raise AssertionError(f"unhandled action kind: {kind}")

    def _cell_available(self, position: tuple[int, int], *, moving_index: int) -> bool:
        x, y = position
        if not (0 <= x < self.config.grid_size and 0 <= y < self.config.grid_size):
            return False
        if position in {self._magnet_cell, *self._slot_cells}:
            return False
        return all(
            other_index == moving_index or obj.position != position
            for other_index, obj in enumerate(self._objects)
        )

    def _push(self, index: int, vector: Vector) -> OutcomeKind:
        obj = self._objects[index]
        if not obj.movable or obj.inserted_slot is not None or vector == (0, 0):
            return OutcomeKind.NO_EFFECT
        destination = (obj.position[0] + vector[0], obj.position[1] + vector[1])
        if not self._cell_available(destination, moving_index=index):
            return OutcomeKind.BLOCKED
        obj.position = destination
        obj.lifted = False
        return OutcomeKind.MOVED

    def _lift(self, index: int) -> OutcomeKind:
        obj = self._objects[index]
        if not obj.liftable or obj.inserted_slot is not None:
            return OutcomeKind.NO_EFFECT
        obj.lifted = not obj.lifted
        return OutcomeKind.LIFTED if obj.lifted else OutcomeKind.LOWERED

    def _magnet(self, index: int) -> OutcomeKind:
        obj = self._objects[index]
        if not obj.magnetic or obj.inserted_slot is not None:
            return OutcomeKind.NO_EFFECT
        x, y = obj.position
        target_x, target_y = self._magnet_cell
        if (x, y) == self._magnet_cell:
            return OutcomeKind.NO_EFFECT
        dx = 0 if x == target_x else (1 if target_x > x else -1)
        # Move along one axis deterministically. This makes the intervention a
        # finite transition while avoiding an unobservable force scalar.
        dy = 0 if dx else (0 if y == target_y else (1 if target_y > y else -1))
        destination = (x + dx, y + dy)
        if destination != self._magnet_cell and not self._cell_available(
            destination, moving_index=index
        ):
            return OutcomeKind.BLOCKED
        if any(
            other_index != index and other.position == destination
            for other_index, other in enumerate(self._objects)
        ):
            return OutcomeKind.BLOCKED
        obj.position = destination
        obj.lifted = False
        return OutcomeKind.ATTRACTED

    def _insert(self, index: int, vector: Vector) -> OutcomeKind:
        obj = self._objects[index]
        if obj.inserted_slot is not None:
            return OutcomeKind.NO_EFFECT
        if vector[1] == 0:
            return OutcomeKind.NO_EFFECT
        slot = 0 if vector[1] < 0 else 1
        if obj.slot_profile != slot:
            return OutcomeKind.MISMATCH
        destination = self._slot_cells[slot]
        if any(
            other_index != index and other.position == destination
            for other_index, other in enumerate(self._objects)
        ):
            return OutcomeKind.BLOCKED
        obj.position = destination
        obj.inserted_slot = slot
        obj.lifted = False
        return OutcomeKind.INSERTED

    def _toggle(self, index: int) -> OutcomeKind:
        obj = self._objects[index]
        if not obj.switchable:
            return OutcomeKind.NO_EFFECT
        obj.active = not obj.active
        return OutcomeKind.ACTIVATED if obj.active else OutcomeKind.DEACTIVATED

    def _observation(self) -> Observation:
        return Observation(
            self._render(),
            tick=self._tick,
            terminal=self._tick >= self.config.max_steps,
        )

    def _render(self) -> np.ndarray:
        size = self.config.frame_size
        base = np.empty((size, size, 3), dtype=np.int16)
        base[...] = np.asarray(self._background, dtype=np.int16)
        base += self._background_texture
        frame = np.clip(base, 0, 255).astype(np.uint8)

        margin = self.config.margin_pixels
        cell = self.config.cell_pixels
        offset_x, offset_y = self._camera_offset
        for grid in range(self.config.grid_size + 1):
            line_x = margin + grid * cell + offset_x
            line_y = margin + grid * cell + offset_y
            if 0 <= line_x < size:
                frame[:, line_x : line_x + 1] = self._grid_color
            if 0 <= line_y < size:
                frame[line_y : line_y + 1, :] = self._grid_color

        self._draw_fixture(frame, self._magnet_cell, (184, 65, 191), mode="magnet")
        self._draw_fixture(frame, self._slot_cells[0], (219, 188, 66), mode="round")
        self._draw_fixture(frame, self._slot_cells[1], (66, 184, 219), mode="square")

        for obj in sorted(self._objects, key=lambda value: value.object_id):
            center = self._grid_center(obj.position, lifted=obj.lifted)
            if obj.lifted:
                self._draw_disc(frame, (center[0] + 2, center[1] + 3), 3, (15, 15, 18))
            if obj.active:
                self._draw_ring(frame, center, max(3, cell // 2), (248, 244, 135))
            self._draw_object(frame, center, obj)
        return frame

    def _draw_fixture(
        self,
        frame: np.ndarray,
        cell_position: tuple[int, int],
        color: tuple[int, int, int],
        *,
        mode: str,
    ) -> None:
        cx, cy = self._grid_center(cell_position)
        radius = max(2, self.config.cell_pixels // 2 - 1)
        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        if mode == "round":
            distance = (xx - cx) ** 2 + (yy - cy) ** 2
            mask = (distance <= radius**2) & (distance >= max(0, radius - 1) ** 2)
        elif mode == "square":
            outer = (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
            inner = (np.abs(xx - cx) < radius - 1) & (np.abs(yy - cy) < radius - 1)
            mask = outer & ~inner
        else:
            mask = (np.abs(xx - cx) <= 1) | (np.abs(yy - cy) <= 1)
            mask &= (np.abs(xx - cx) <= radius) & (np.abs(yy - cy) <= radius)
        frame[mask] = color

    @staticmethod
    def _draw_disc(
        frame: np.ndarray,
        center: Pixel,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        frame[(xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2] = color

    @staticmethod
    def _draw_ring(
        frame: np.ndarray,
        center: Pixel,
        radius: int,
        color: tuple[int, int, int],
    ) -> None:
        cx, cy = center
        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        distance = (xx - cx) ** 2 + (yy - cy) ** 2
        frame[(distance <= radius**2) & (distance >= max(0, radius - 1) ** 2)] = color

    def _draw_object(self, frame: np.ndarray, center: Pixel, obj: _LatentObject) -> None:
        cx, cy = center
        radius = max(2, self.config.cell_pixels // 2 - 1)
        yy, xx = np.ogrid[: frame.shape[0], : frame.shape[1]]
        dx = xx - cx
        dy = yy - cy
        if obj.visual_shape == 0:
            mask = dx**2 + dy**2 <= radius**2
        elif obj.visual_shape == 1:
            mask = (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
        elif obj.visual_shape == 2:
            mask = np.abs(dx) + np.abs(dy) <= radius + 1
        else:
            mask = ((np.abs(dx) <= 1) | (np.abs(dy) <= 1)) & (
                (np.abs(dx) <= radius) & (np.abs(dy) <= radius)
            )
        color = np.asarray(obj.visual_color, dtype=np.int16)
        checker = ((xx + yy + obj.visual_texture) % 2) * (3 + obj.visual_texture)
        textured = np.clip(color.reshape(1, 1, 3) + checker[..., None], 0, 255).astype(np.uint8)
        frame[mask] = textured[mask]


class Microworld:
    """Sealed learner capability exposing pixels and opaque codes only."""

    __slots__ = (
        "__manifest_capability",
        "__observe_capability",
        "__reset_capability",
        "__step_capability",
        "__trajectory_capability",
    )

    def __init__(self, engine: _Engine) -> None:
        # Store narrow bound capabilities instead of publishing the engine.
        self.__manifest_capability = engine.manifest
        self.__observe_capability = engine.observe
        self.__reset_capability = engine.reset
        self.__step_capability = engine.step
        self.__trajectory_capability = engine.trajectory

    @property
    def manifest(self) -> AgentManifest:
        return self.__manifest_capability

    @property
    def action_codes(self) -> tuple[int, ...]:
        return self.manifest.action_codes

    @property
    def symbol_codes(self) -> tuple[int, ...]:
        return self.manifest.symbol_codes

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
        """Apply an action record, or an opaque code plus image coordinates."""

        if isinstance(action, Action):
            if target is not None or vector != (0, 0):
                raise TypeError("target/vector must be omitted when passing an Action")
            record = action
        else:
            if target is None:
                raise TypeError("target is required when passing an opaque action code")
            record = Action(action, target, vector)
        return self.__step_capability(record)

    def trajectory(self) -> Trajectory:
        return self.__trajectory_capability()


class _Oracle:
    """Concrete evaluator capability tied to one engine."""

    __slots__ = ("_engine",)

    def __init__(self, engine: _Engine) -> None:
        self._engine = engine

    @property
    def manifest(self) -> AgentManifest:
        return self._engine.manifest

    def snapshot(self) -> OracleSnapshot:
        states: list[OracleObjectState] = []
        for obj in self._engine._objects:
            predicates = {predicate: obj.predicate(predicate) for predicate in PredicateKind}
            states.append(
                OracleObjectState(
                    object_id=obj.object_id,
                    position=obj.position,
                    predicates=predicates,
                    lifted=obj.lifted,
                    active=obj.active,
                    inserted_slot=obj.inserted_slot,
                )
            )
        return OracleSnapshot(self._engine._tick, tuple(states))

    def decode_action(self, code: int) -> ActionKind:
        return self._engine.codebook.decode_action(code)

    def decode_outcome(self, code: int) -> OutcomeKind:
        return self._engine.codebook.decode_outcome(code)

    def decode_symbol(self, code: int) -> PredicateKind:
        return self._engine.codebook.decode_predicate(code)

    def encode_symbol(self, predicate: PredicateKind) -> int:
        return self._engine.codebook.predicates[PredicateKind(predicate)]

    def predicate(self, object_id: int, predicate: int | PredicateKind) -> bool:
        if isinstance(predicate, PredicateKind):
            kind = predicate
        else:
            kind = self.decode_symbol(predicate)
        return self._engine._object_by_id(object_id).predicate(kind)

    def object_at(self, target: Pixel) -> int | None:
        return self._engine.object_at(target)

    def object_center(self, object_id: int) -> Pixel:
        return self._engine.object_center(object_id)

    def intervention_signature(self, object_id: int) -> Mapping[ActionKind, OutcomeKind]:
        """Return outcomes in a canonical, obstruction-free counterfactual.

        Each primitive starts from a fresh clone containing only the selected
        object at the centre of the grid.  Mutable state is neutralised
        (lowered, inactive and not inserted), time is reset, and fixtures and
        nuisance rendering are unchanged.  Push uses the east cardinal vector;
        insertion uses the object's compatible canonical fixture.  Consequently
        the action-level signature tracks intrinsic affordances and cannot turn
        into ``BLOCKED`` or ``MISMATCH`` merely because the live episode used an
        obstructed or incompatible probe.
        """

        source = self._engine._object_by_id(object_id)
        outcomes: dict[ActionKind, OutcomeKind] = {}
        for kind in ActionKind:
            clone = self._engine._fork()
            canonical = replace(
                source,
                position=(clone.config.grid_size // 2, clone.config.grid_size // 2),
                lifted=False,
                active=False,
                inserted_slot=None,
            )
            clone._objects = [canonical]
            clone._initial_objects = (replace(canonical),)
            clone._tick = 0
            vector: Vector = (1, 0) if kind is ActionKind.PUSH else (0, 0)
            if kind is ActionKind.INSERT:
                vector = (0, -1) if canonical.slot_profile == 0 else (0, 1)
            action = Action(clone.codebook.actions[kind], clone.object_center(object_id), vector)
            outcomes[kind] = clone._advance(action)
        return MappingProxyType(outcomes)

    def negative_control_invariant(
        self,
        actions: Iterable[Action] = (),
        *,
        depth: int = DEFAULT_INVARIANT_DEPTH,
        max_transitions: int = DEFAULT_INVARIANT_MAX_TRANSITIONS,
    ) -> bool:
        """Check a precisely bounded counterfactual non-interference claim.

        If ``actions`` is non-empty, exactly that trajectory is compared in a
        world whose negative-control bits are flipped.  Otherwise the method
        exhaustively compares *every sequence of lengths one through* ``depth``
        over this finite probe alphabet:

        * the target is the current image-space centre of each object;
        * every :class:`ActionKind` is exercised; and
        * every vector in :data:`INVARIANT_PROBE_VECTORS` is exercised.

        Outcome codes and complete after-observations (pixels, tick and terminal
        status) must agree at every prefix.  Arbitrary background coordinates
        are intentionally outside the claim; they are misses and do not select
        an object's negative-control bit in this transition system.

        The exact number of explored prefix transitions is checked against
        ``max_transitions`` before execution, and depths above
        :data:`MAX_INVARIANT_DEPTH` are rejected.  The method therefore never
        silently samples or returns ``True`` for a truncated search.
        """

        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)):
            raise TypeError("depth must be an integer")
        if isinstance(max_transitions, bool) or not isinstance(
            max_transitions, (int, np.integer)
        ):
            raise TypeError("max_transitions must be an integer")
        depth = int(depth)
        max_transitions = int(max_transitions)
        if not 0 <= depth <= MAX_INVARIANT_DEPTH:
            raise ValueError(f"depth must lie in [0, {MAX_INVARIANT_DEPTH}]")
        if max_transitions < 1:
            raise ValueError("max_transitions must be positive")

        sequence = tuple(actions)
        if any(not isinstance(action, Action) for action in sequence):
            raise TypeError("actions must contain only Action records")
        remaining_steps = self._engine.config.max_steps - self._engine._tick
        required_steps = len(sequence) if sequence else depth
        if required_steps > remaining_steps:
            raise ValueError("requested invariant path exceeds the remaining episode budget")
        if sequence and len(sequence) > max_transitions:
            raise ValueError("supplied action sequence exceeds max_transitions")

        left = self._engine._fork()
        right = self._engine._fork(flip_negative_control=True)
        if left.observe() != right.observe():
            return False
        for action in sequence:
            l_transition = left.step(action)
            r_transition = right.step(action)
            if l_transition.outcome_code != r_transition.outcome_code:
                return False
            if l_transition.after != r_transition.after:
                return False

        if sequence:
            return True

        object_ids = tuple(obj.object_id for obj in self._engine._objects)
        branch_factor = len(object_ids) * len(ActionKind) * len(INVARIANT_PROBE_VECTORS)
        explored = 0
        layer = 1
        for _ in range(depth):
            layer *= branch_factor
            explored += layer
            if explored > max_transitions:
                raise ValueError(
                    "exhaustive invariant search requires "
                    f"{explored} transitions, exceeding max_transitions={max_transitions}"
                )

        # Rendering is a pure function of the complete object state and tick.
        # Cache it because many distinct action sequences converge to the same
        # state; transition/outcome exploration itself remains sequence-exact.
        observation_cache: dict[tuple[object, ...], Observation] = {}

        def state_key(engine: _Engine) -> tuple[object, ...]:
            objects = tuple(
                (
                    obj.object_id,
                    obj.position,
                    obj.movable,
                    obj.liftable,
                    obj.magnetic,
                    obj.slot_profile,
                    obj.switchable,
                    obj.negative_control,
                    obj.visual_shape,
                    obj.visual_color,
                    obj.visual_texture,
                    obj.lifted,
                    obj.active,
                    obj.inserted_slot,
                )
                for obj in engine._objects
            )
            return (engine._tick, objects)

        def observation(engine: _Engine) -> Observation:
            key = state_key(engine)
            cached = observation_cache.get(key)
            if cached is None:
                cached = engine.observe()
                observation_cache[key] = cached
            return cached

        def visit(prefix_left: _Engine, prefix_right: _Engine, levels: int) -> bool:
            if levels == 0:
                return True
            for object_id in object_ids:
                for kind in ActionKind:
                    code = self._engine.codebook.actions[kind]
                    for vector in INVARIANT_PROBE_VECTORS:
                        probe_left = prefix_left._fork()
                        probe_right = prefix_right._fork()
                        action = Action(code, probe_left.object_center(object_id), vector)
                        left_outcome = probe_left._advance(action)
                        right_outcome = probe_right._advance(action)
                        if left_outcome is not right_outcome:
                            return False
                        if observation(probe_left) != observation(probe_right):
                            return False
                        if not visit(probe_left, probe_right, levels - 1):
                            return False
            return True

        return visit(left, right, depth)


class EvaluatorHarness:
    """Evaluator-owned bundle; pass only ``agent`` to candidate code."""

    __slots__ = ("agent", "oracle")

    def __init__(self, seed: int, config: WorldConfig | None = None) -> None:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
        seed = int(seed)
        if seed < 0:
            raise ValueError("seed must be non-negative")
        selected = config if config is not None else WorldConfig()
        if not isinstance(selected, WorldConfig):
            raise TypeError("config must be a WorldConfig")
        codebook = OpaqueCodebook.from_seed(seed)
        rng = np.random.default_rng(np.random.SeedSequence((seed, 0x51A1ED)))
        engine = _Engine(selected, codebook, rng)
        self.agent = Microworld(engine)
        self.oracle = _Oracle(engine)


def create_microworld(seed: int, config: WorldConfig | None = None) -> EvaluatorHarness:
    """Create a fresh evaluator harness without retaining the input seed."""

    return EvaluatorHarness(seed, config)
