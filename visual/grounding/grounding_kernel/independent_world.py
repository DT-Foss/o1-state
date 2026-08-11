"""An independently implemented finite sensorimotor environment.

The learner capability in this module exposes raw RGB observations and opaque
integer action, outcome, and token alphabets.  Evaluator-only meanings and
latent state live behind a separate oracle capability.  The simulator does not
import or wrap either ``microworld`` or ``processworld``.

This environment is deliberately small.  It supports finite operational tests
of causal persistence and delayed response; it does not establish reference
outside this simulator family.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from types import MappingProxyType
from typing import TypeVar

import numpy as np

from .contracts import Action, Observation, Trajectory, Transition


__all__ = [
    "INDEPENDENT_MOTOR_VECTORS",
    "IndependentAgent",
    "IndependentConfig",
    "IndependentHarness",
    "IndependentManifest",
    "IndependentOracle",
    "IndependentOracleDevice",
    "IndependentOracleSnapshot",
    "IndependentWorld",
    "audit_independent_agent",
    "create_independent_world",
    "pixel_change_pattern",
    "trace_is_continuous",
]


Pixel = tuple[int, int]
Vector = tuple[int, int]
_EnumT = TypeVar("_EnumT", bound=Enum)
INDEPENDENT_MOTOR_VECTORS: tuple[Vector, ...] = (
    (0, 0),
    (1, 0),
    (-1, 0),
    (0, 1),
    (0, -1),
)


def _strict_int(value: object, field: str, *, minimum: int | None = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


def _integer_tuple(values: Iterable[object], field: str) -> tuple[int, ...]:
    return tuple(_strict_int(value, field) for value in values)


class _ActionMeaning(str, Enum):
    """Evaluator-only meanings behind a freshly sampled motor alphabet."""

    EXCITE = "excite"
    ADVANCE = "advance"
    QUERY = "query"
    CLEAR = "clear"


class _OutcomeMeaning(str, Enum):
    """Evaluator-only meanings behind opaque transition outcomes."""

    ACCEPTED = "accepted"
    ADVANCED = "advanced"
    ACTIVE = "active"
    INACTIVE = "inactive"
    CLEARED = "cleared"
    ALREADY_CLEAR = "already_clear"
    MISSED = "missed"


class _TokenMeaning(str, Enum):
    """Evaluator-only latent roles behind opaque concept tokens."""

    RETAINER = "retainer"
    RELAY = "relay"
    DELAY_TWIN_A = "delay_twin_a"
    DELAY_TWIN_B = "delay_twin_b"


@dataclass(frozen=True, slots=True)
class IndependentConfig:
    """Evaluator configuration for the finite interaction geometry."""

    frame_size: int = 64
    max_steps: int = 8
    device_radius: int = 6

    def __post_init__(self) -> None:
        for name in ("frame_size", "max_steps", "device_radius"):
            object.__setattr__(self, name, _strict_int(getattr(self, name), name))
        if self.frame_size < 56:
            raise ValueError("frame_size must be at least 56")
        if self.max_steps < 4:
            raise ValueError("max_steps must be at least 4")
        if not 4 <= self.device_radius <= self.frame_size // 8:
            raise ValueError("device_radius is incompatible with frame_size")


@dataclass(frozen=True, slots=True)
class IndependentManifest:
    """The complete learner-visible schema, with no semantic mapping."""

    observation_shape: tuple[int, int, int]
    action_codes: tuple[int, ...]
    outcome_codes: tuple[int, ...]
    token_codes: tuple[int, ...]
    motor_vectors: tuple[Vector, ...]
    max_steps: int

    def __post_init__(self) -> None:
        shape = _integer_tuple(self.observation_shape, "observation_shape")
        actions = _integer_tuple(self.action_codes, "action_codes")
        outcomes = _integer_tuple(self.outcome_codes, "outcome_codes")
        tokens = _integer_tuple(self.token_codes, "token_codes")
        vectors = tuple(
            (
                _strict_int(vector[0], "motor_vectors"),
                _strict_int(vector[1], "motor_vectors"),
            )
            for vector in self.motor_vectors
        )
        if len(shape) != 3 or shape[2] != 3 or min(shape) <= 0:
            raise ValueError("observation_shape must be positive RGB dimensions")
        joined = actions + outcomes + tokens
        if len(set(joined)) != len(joined):
            raise ValueError("opaque action, outcome, and token codes must be globally unique")
        if not vectors or len(set(vectors)) != len(vectors):
            raise ValueError("motor_vectors must be non-empty and unique")
        object.__setattr__(self, "observation_shape", shape)
        object.__setattr__(self, "action_codes", tuple(sorted(actions)))
        object.__setattr__(self, "outcome_codes", tuple(sorted(outcomes)))
        object.__setattr__(self, "token_codes", tuple(sorted(tokens)))
        object.__setattr__(self, "motor_vectors", vectors)
        object.__setattr__(
            self,
            "max_steps",
            _strict_int(self.max_steps, "max_steps", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class IndependentOracleDevice:
    """Privileged device state returned only by the evaluator oracle."""

    latent_role: str
    token_code: int
    center: Pixel
    phase: int
    active: bool
    last_query: bool | None


@dataclass(frozen=True, slots=True)
class IndependentOracleSnapshot:
    """Privileged world snapshot; never returned through the agent API."""

    tick: int
    devices: tuple[IndependentOracleDevice, ...]


@dataclass(frozen=True, slots=True)
class _Codebook:
    actions: Mapping[_ActionMeaning, int]
    outcomes: Mapping[_OutcomeMeaning, int]
    tokens: Mapping[_TokenMeaning, int]

    def __post_init__(self) -> None:
        actions = {
            _ActionMeaning(key): _strict_int(value, "action code")
            for key, value in self.actions.items()
        }
        outcomes = {
            _OutcomeMeaning(key): _strict_int(value, "outcome code")
            for key, value in self.outcomes.items()
        }
        tokens = {
            _TokenMeaning(key): _strict_int(value, "token code")
            for key, value in self.tokens.items()
        }
        if set(actions) != set(_ActionMeaning):
            raise ValueError("action codebook must cover the complete action alphabet")
        if set(outcomes) != set(_OutcomeMeaning):
            raise ValueError("outcome codebook must cover the complete outcome alphabet")
        if set(tokens) != set(_TokenMeaning):
            raise ValueError("token codebook must cover the complete token alphabet")
        values = tuple(actions.values()) + tuple(outcomes.values()) + tuple(tokens.values())
        if len(set(values)) != len(values):
            raise ValueError("opaque codes must be globally unique")
        object.__setattr__(self, "actions", MappingProxyType(actions))
        object.__setattr__(self, "outcomes", MappingProxyType(outcomes))
        object.__setattr__(self, "tokens", MappingProxyType(tokens))

    @classmethod
    def sample(cls, seed: int, variant: int) -> _Codebook:
        rng = np.random.default_rng(np.random.SeedSequence((seed, variant, 0x1D3E_7A11)))
        count = len(_ActionMeaning) + len(_OutcomeMeaning) + len(_TokenMeaning)
        codes = [int(code) for code in rng.choice(899_999_999, count, replace=False) + 100_000_000]
        action_end = len(_ActionMeaning)
        outcome_end = action_end + len(_OutcomeMeaning)
        return cls(
            dict(zip(_ActionMeaning, codes[:action_end], strict=True)),
            dict(zip(_OutcomeMeaning, codes[action_end:outcome_end], strict=True)),
            dict(zip(_TokenMeaning, codes[outcome_end:], strict=True)),
        )

    @property
    def action_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self.actions.values()))

    @property
    def outcome_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self.outcomes.values()))

    @property
    def token_codes(self) -> tuple[int, ...]:
        return tuple(sorted(self.tokens.values()))

    def decode_action(self, code: int) -> _ActionMeaning:
        return self._decode(code, self.actions, "action")

    def decode_outcome(self, code: int) -> _OutcomeMeaning:
        return self._decode(code, self.outcomes, "outcome")

    def decode_token(self, code: int) -> _TokenMeaning:
        return self._decode(code, self.tokens, "token")

    @staticmethod
    def _decode(code: int, mapping: Mapping[_EnumT, int], label: str) -> _EnumT:
        value = _strict_int(code, f"{label} code")
        for meaning, opaque in mapping.items():
            if opaque == value:
                return meaning
        raise KeyError(f"unknown {label} code: {value}")


@dataclass(frozen=True, slots=True)
class _DeviceSpec:
    role: _TokenMeaning
    center: Pixel


@dataclass(slots=True)
class _DeviceState:
    spec: _DeviceSpec
    phase: int = 0
    active: bool = False
    last_query: bool | None = None


@dataclass(frozen=True, slots=True)
class _Renderer:
    background: tuple[int, int, int]
    outline: tuple[int, int, int]
    dormant: tuple[int, int, int]
    active: tuple[int, int, int]
    positive: tuple[int, int, int]
    negative: tuple[int, int, int]
    accent: tuple[int, int, int]
    shape: int

    @classmethod
    def sample(cls, seed: int, variant: int) -> _Renderer:
        palette = np.array(
            [
                (13, 22, 40),
                (241, 245, 249),
                (45, 212, 191),
                (251, 113, 133),
                (250, 204, 21),
                (96, 165, 250),
                (168, 85, 247),
            ],
            dtype=np.uint8,
        )
        rng = np.random.default_rng(np.random.SeedSequence((seed, variant, 0x63B1_2D09)))
        palette = palette[rng.permutation(len(palette))]
        channel_order = rng.permutation(3)
        palette = palette[:, channel_order]
        values = [tuple(int(channel) for channel in colour) for colour in palette]
        return cls(*values, shape=int(rng.integers(0, 3)))


@dataclass(frozen=True, slots=True)
class _Blueprint:
    config: IndependentConfig
    codebook: _Codebook
    devices: tuple[_DeviceSpec, ...]
    renderer: _Renderer
    manifest: IndependentManifest


def _make_blueprint(
    seed: int,
    config: IndependentConfig,
    *,
    codebook_variant: int,
    renderer_variant: int,
    world_variant: int,
) -> _Blueprint:
    seed = _strict_int(seed, "seed", minimum=0)
    codebook_variant = _strict_int(codebook_variant, "codebook_variant", minimum=0)
    renderer_variant = _strict_int(renderer_variant, "renderer_variant", minimum=0)
    world_variant = _strict_int(world_variant, "world_variant", minimum=0)
    codebook = _Codebook.sample(seed, codebook_variant)
    renderer = _Renderer.sample(seed, renderer_variant)
    low = config.frame_size // 4
    high = config.frame_size - low - 1
    centers: tuple[Pixel, ...] = ((low, low), (high, low), (low, high), (high, high))
    world_rng = np.random.default_rng(np.random.SeedSequence((seed, world_variant, 0x4B97_A6C3)))
    roles = tuple(_TokenMeaning)
    order = tuple(int(index) for index in world_rng.permutation(len(roles)))
    devices = tuple(
        _DeviceSpec(roles[index], center) for index, center in zip(order, centers, strict=True)
    )
    manifest = IndependentManifest(
        (config.frame_size, config.frame_size, 3),
        codebook.action_codes,
        codebook.outcome_codes,
        codebook.token_codes,
        INDEPENDENT_MOTOR_VECTORS,
        config.max_steps,
    )
    return _Blueprint(config, codebook, devices, renderer, manifest)


class _Engine:
    """Mutable simulator core retained behind capability objects."""

    __slots__ = ("_blueprint", "_devices", "_tick", "_trajectory")

    def __init__(self, blueprint: _Blueprint) -> None:
        self._blueprint = blueprint
        self._devices: list[_DeviceState] = []
        self._tick = 0
        self._trajectory: Trajectory
        self.reset()

    def reset(self) -> Observation:
        self._devices = [_DeviceState(spec) for spec in self._blueprint.devices]
        self._tick = 0
        initial = self.observe()
        self._trajectory = Trajectory(initial)
        return initial

    def observe(self) -> Observation:
        return Observation(
            self._render(), self._tick, self._tick >= self._blueprint.config.max_steps
        )

    def trajectory(self) -> Trajectory:
        return self._trajectory

    def step(self, action: Action) -> Transition:
        if not isinstance(action, Action):
            raise TypeError("action must be an Action")
        if self._tick >= self._blueprint.config.max_steps:
            raise RuntimeError("episode is terminal; call reset before stepping again")
        if action.vector not in self._blueprint.manifest.motor_vectors:
            raise ValueError("action vector is not declared by the manifest")
        meaning = self._blueprint.codebook.decode_action(action.code)
        before = self.observe()
        device = self._device_at(action.target)
        if device is None:
            outcome = _OutcomeMeaning.MISSED
        else:
            outcome = self._apply(device, meaning)
        self._tick += 1
        after = self.observe()
        transition = Transition(before, action, after, self._blueprint.codebook.outcomes[outcome])
        self._trajectory = self._trajectory.append(transition)
        return transition

    def snapshot(self) -> IndependentOracleSnapshot:
        codebook = self._blueprint.codebook
        devices = tuple(
            IndependentOracleDevice(
                state.spec.role.value,
                codebook.tokens[state.spec.role],
                state.spec.center,
                state.phase,
                state.active,
                state.last_query,
            )
            for state in self._devices
        )
        return IndependentOracleSnapshot(self._tick, devices)

    def _device_at(self, target: Pixel) -> _DeviceState | None:
        x, y = target
        radius = self._blueprint.config.device_radius + 1
        matches = [
            state
            for state in self._devices
            if abs(x - state.spec.center[0]) <= radius and abs(y - state.spec.center[1]) <= radius
        ]
        return matches[0] if len(matches) == 1 else None

    @staticmethod
    def _apply(device: _DeviceState, meaning: _ActionMeaning) -> _OutcomeMeaning:
        if meaning is _ActionMeaning.EXCITE:
            device.last_query = None
            device.phase = 1
            device.active = device.spec.role in {
                _TokenMeaning.RETAINER,
                _TokenMeaning.RELAY,
            }
            return _OutcomeMeaning.ACCEPTED
        if meaning is _ActionMeaning.ADVANCE:
            device.last_query = None
            if device.spec.role is _TokenMeaning.RETAINER:
                if device.phase:
                    device.phase += 1
                    device.active = True
            elif device.spec.role is _TokenMeaning.RELAY:
                if device.phase:
                    device.phase += 1
                device.active = False
            elif device.phase:
                device.phase += 1
                device.active = device.phase >= 3
            return _OutcomeMeaning.ADVANCED
        if meaning is _ActionMeaning.QUERY:
            device.last_query = device.active
            return _OutcomeMeaning.ACTIVE if device.active else _OutcomeMeaning.INACTIVE
        changed = device.phase != 0 or device.active or device.last_query is not None
        device.phase = 0
        device.active = False
        device.last_query = None
        return _OutcomeMeaning.CLEARED if changed else _OutcomeMeaning.ALREADY_CLEAR

    def _render(self) -> np.ndarray:
        config = self._blueprint.config
        renderer = self._blueprint.renderer
        frame = np.empty((config.frame_size, config.frame_size, 3), dtype=np.uint8)
        frame[...] = renderer.background
        # A renderer-specific border is a nuisance channel, not a device label.
        frame[:2, :] = renderer.accent
        frame[-2:, :] = renderer.accent
        frame[:, :2] = renderer.accent
        frame[:, -2:] = renderer.accent
        yy, xx = np.ogrid[: config.frame_size, : config.frame_size]
        radius = config.device_radius
        for state in self._devices:
            cx, cy = state.spec.center
            dx = np.abs(xx - cx)
            dy = np.abs(yy - cy)
            if renderer.shape == 0:
                outer = np.maximum(dx, dy) <= radius
                inner = np.maximum(dx, dy) <= radius - 2
            elif renderer.shape == 1:
                outer = dx + dy <= radius + 2
                inner = dx + dy <= radius - 1
            else:
                outer = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius + 1) ** 2
                inner = (xx - cx) ** 2 + (yy - cy) ** 2 <= (radius - 2) ** 2
            frame[outer] = renderer.outline
            frame[inner] = renderer.active if state.active else renderer.dormant
            if state.last_query is not None:
                marker_y = cy + radius + 2
                marker_x0 = cx - radius + 1
                marker_x1 = cx + radius
                colour = renderer.positive if state.last_query else renderer.negative
                frame[marker_y : marker_y + 2, marker_x0:marker_x1] = colour
        return frame


class IndependentAgent:
    """Narrow learner-facing capability for :class:`IndependentHarness`."""

    __slots__ = ("_engine",)

    def __init__(self, engine: _Engine) -> None:
        self._engine = engine

    @property
    def manifest(self) -> IndependentManifest:
        return self._engine._blueprint.manifest

    @property
    def action_codes(self) -> tuple[int, ...]:
        return self.manifest.action_codes

    @property
    def outcome_codes(self) -> tuple[int, ...]:
        return self.manifest.outcome_codes

    @property
    def token_codes(self) -> tuple[int, ...]:
        return self.manifest.token_codes

    def reset(self) -> Observation:
        return self._engine.reset()

    def observe(self) -> Observation:
        return self._engine.observe()

    def step(self, action: Action) -> Transition:
        return self._engine.step(action)

    def trajectory(self) -> Trajectory:
        return self._engine.trajectory()


# Environment-style alias matching the package's existing naming convention.
IndependentWorld = IndependentAgent


class IndependentOracle:
    """Evaluator-only decoder, snapshot, and controlled-probe capability."""

    __slots__ = ("_blueprint", "_live_engine")

    def __init__(self, blueprint: _Blueprint, live_engine: _Engine) -> None:
        self._blueprint = blueprint
        self._live_engine = live_engine

    @property
    def manifest(self) -> IndependentManifest:
        return self._blueprint.manifest

    def snapshot(self) -> IndependentOracleSnapshot:
        return self._live_engine.snapshot()

    def decode_action(self, code: int) -> str:
        return self._blueprint.codebook.decode_action(code).value

    def decode_outcome(self, code: int) -> str:
        return self._blueprint.codebook.decode_outcome(code).value

    def decode_token(self, code: int) -> str:
        return self._blueprint.codebook.decode_token(code).value

    def encode_action(self, meaning: str) -> int:
        return self._blueprint.codebook.actions[_ActionMeaning(meaning)]

    def encode_token(self, meaning: str) -> int:
        return self._blueprint.codebook.tokens[_TokenMeaning(meaning)]

    def device_center(self, token: int) -> Pixel:
        role = self._blueprint.codebook.decode_token(token)
        for device in self._blueprint.devices:
            if device.role is role:
                return device.center
        raise AssertionError("blueprint token has no device")

    def matched_causal_tokens(self) -> tuple[int, int]:
        """Return twins identical through excitation but separated by advance."""

        tokens = self._blueprint.codebook.tokens
        return tokens[_TokenMeaning.RETAINER], tokens[_TokenMeaning.RELAY]

    def nonidentifiable_tokens(self) -> tuple[int, int]:
        """Return latent twins with identical allowed intervention consequences."""

        tokens = self._blueprint.codebook.tokens
        return tokens[_TokenMeaning.DELAY_TWIN_A], tokens[_TokenMeaning.DELAY_TWIN_B]

    def run_probe(self, token: int, program: Sequence[int]) -> Trajectory:
        """Run one opaque action program on a fresh evaluator-side clone."""

        role = self._blueprint.codebook.decode_token(token)
        codes = _integer_tuple(program, "program")
        if not codes:
            raise ValueError("program cannot be empty")
        if len(codes) > self.manifest.max_steps:
            raise ValueError("program exceeds the episode step budget")
        engine = _Engine(self._blueprint)
        center = next(device.center for device in self._blueprint.devices if device.role is role)
        for code in codes:
            engine.step(Action(code, center, (0, 0)))
        return engine.trajectory()

    def causal_twin_traces(self) -> tuple[Trajectory, Trajectory]:
        """Return same-target counterfactual clones separated only by latent role."""

        program = tuple(
            self._blueprint.codebook.actions[meaning]
            for meaning in (
                _ActionMeaning.EXCITE,
                _ActionMeaning.ADVANCE,
                _ActionMeaning.QUERY,
            )
        )
        retainer_token, _relay_token = self.matched_causal_tokens()
        anchor = self.device_center(retainer_token)
        return (
            self._counterfactual_probe(_TokenMeaning.RETAINER, anchor, program),
            self._counterfactual_probe(_TokenMeaning.RELAY, anchor, program),
        )

    def _counterfactual_probe(
        self,
        role: _TokenMeaning,
        anchor: Pixel,
        program: Sequence[int],
    ) -> Trajectory:
        devices = tuple(
            _DeviceSpec(role if device.center == anchor else device.role, device.center)
            for device in self._blueprint.devices
        )
        counterfactual = _Blueprint(
            self._blueprint.config,
            self._blueprint.codebook,
            devices,
            self._blueprint.renderer,
            self._blueprint.manifest,
        )
        engine = _Engine(counterfactual)
        for code in program:
            engine.step(Action(code, anchor, (0, 0)))
        return engine.trajectory()

    def consequences(self, token: int, program: Sequence[int]) -> tuple[str, ...]:
        """Decode one probe for evaluator diagnostics, never for learner input."""

        trace = self.run_probe(token, program)
        return tuple(self.decode_outcome(step.outcome_code) for step in trace.transitions)


class IndependentHarness:
    """Construct separate learner and evaluator capabilities over one engine."""

    __slots__ = ("agent", "oracle")

    def __init__(
        self,
        seed: int,
        config: IndependentConfig | None = None,
        *,
        codebook_variant: int = 0,
        renderer_variant: int = 0,
        world_variant: int = 0,
    ) -> None:
        selected = config or IndependentConfig()
        if not isinstance(selected, IndependentConfig):
            raise TypeError("config must be an IndependentConfig")
        blueprint = _make_blueprint(
            seed,
            selected,
            codebook_variant=codebook_variant,
            renderer_variant=renderer_variant,
            world_variant=world_variant,
        )
        engine = _Engine(blueprint)
        self.agent = IndependentAgent(engine)
        self.oracle = IndependentOracle(blueprint, engine)


def create_independent_world(
    seed: int,
    config: IndependentConfig | None = None,
    *,
    codebook_variant: int = 0,
    renderer_variant: int = 0,
    world_variant: int = 0,
) -> IndependentHarness:
    """Return a deterministic capability-separated environment harness."""

    return IndependentHarness(
        seed,
        config,
        codebook_variant=codebook_variant,
        renderer_variant=renderer_variant,
        world_variant=world_variant,
    )


_FORBIDDEN_AGENT_NAMES = frozenset(
    {
        "seed",
        "rng",
        "oracle",
        "codebook",
        "decode_action",
        "decode_outcome",
        "decode_token",
        "encode_action",
        "encode_token",
        "snapshot",
        "latent",
        "latents",
        "devices",
        "device_center",
        "matched_causal_tokens",
        "nonidentifiable_tokens",
        "run_probe",
    }
)


def audit_independent_agent(agent: IndependentAgent) -> tuple[str, ...]:
    """Report obvious public capability leaks (not a Python security sandbox)."""

    if not isinstance(agent, IndependentAgent):
        raise TypeError("agent must be an IndependentAgent")
    public = {name for name in dir(agent) if not name.startswith("_")}
    return tuple(sorted(public & _FORBIDDEN_AGENT_NAMES))


def pixel_change_pattern(trace: Trajectory) -> tuple[bool, ...]:
    """Return a renderer-value-independent diagnostic over a raw trajectory."""

    if not isinstance(trace, Trajectory):
        raise TypeError("trace must be a Trajectory")
    return tuple(step.pixels_changed for step in trace.transitions)


def trace_is_continuous(trace: Trajectory) -> bool:
    """Explicitly check continuity for external audit code."""

    observations = (trace.initial,) + tuple(step.after for step in trace.transitions)
    return all(left.tick + 1 == right.tick for left, right in pairwise(observations))
