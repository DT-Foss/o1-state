"""Deterministic, leakage-separated Grade-3 ProcessWorld cases.

Only :class:`Grade3PublicCase` is learner-facing.  It contains raw RGB,
opaque integer symbols, generic support corrections, and feedback-free public
traces.  Evaluator truth, semantic construction choices, seeds, and executable
world factories live behind :class:`Grade3EvaluatorCase`.

The generated support is deliberately richer than a bag of labelled frames:

* two interventional predicates have positive and negative causal twins;
* two target-role words are grounded by an ostensive cue plus those twins;
* two motor words denote temporally extended public action schemes;
* three cells identify a two-by-two compositional lexicon while the fourth is
  held out;
* empty, zero-correction records define a grounded chain and a closed,
  unanchored cycle; and
* a four-hypothesis causal table is measured in independent support worlds,
  while holdout evidence is produced only by executing real public probes.

``Transition.outcome_code`` never crosses this module's public boundary.
Support correction occurs only on :class:`PublicTurn`; every trace, including
support traces, is feedback-free and has the exact shape used at query time.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from types import MappingProxyType

import numpy as np

from .certificates import manifest_hash
from .processworld import ProcessConfig, ProcessHarness, ProcessWorld
from .v1_adapters import episode_to_query_trace
from .v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    SENSOR_SCHEMA_RGB_U8,
    PublicTrace,
    PublicTurn,
    SessionPhase,
    Utterance,
)
from .v1_grade3_contracts import (
    GRADE3_PROTOCOL_VERSION,
    CausalSupportRecord,
    Grade3SessionManifest,
    Grade3SupportRecord,
    MotorActionSpace,
    OstensiveSupportRecord,
    ProbeOffer,
    ProbeOption,
)


GRADE3_CASE_VERSION = "processworld-grade3-cases/1"
_OPAQUE_LOWER = 100_000_000
_OPAQUE_SPAN = 900_000_000
_FACTORIAL_CELL_COUNT = 4
_CAUSAL_PATTERNS: tuple[tuple[bool, bool], ...] = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field} must be an integer")
    result = int(value)
    if result < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return result


class _OpaqueAllocator:
    """Collision-free deterministic allocator with no semantic payload."""

    def __init__(self, seed: int, reserved: Sequence[int]) -> None:
        self._seed = _integer(seed, "seed")
        self._occupied = {_integer(value, "reserved", minimum=1) for value in reserved}

    def take(self, namespace: str) -> int:
        if not isinstance(namespace, str) or not namespace:
            raise TypeError("namespace must be a nonempty string")
        prefix = f"{GRADE3_CASE_VERSION}|{self._seed}|{namespace}|".encode("utf-8")
        counter = 0
        while True:
            digest = sha256(prefix + counter.to_bytes(8, "big")).digest()
            candidate = _OPAQUE_LOWER + int.from_bytes(digest[:8], "big") % _OPAQUE_SPAN
            if candidate not in self._occupied:
                self._occupied.add(candidate)
                return candidate
            counter += 1


def _variant_rng(seed: int, variant: int, namespace: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence((_integer(seed, "seed"), _integer(variant, "variant"), namespace))
    )


def _permuted(values: Sequence[int], rng: np.random.Generator) -> tuple[int, ...]:
    source = tuple(values)
    order = tuple(int(index) for index in rng.permutation(len(source)))
    return tuple(source[index] for index in order)


def _cue(trace: PublicTrace, radius: int = 8) -> tuple[int, int, int, int]:
    if not trace.transitions:
        raise ValueError("an ostensive cue requires a nonempty trace")
    x, y = trace.transitions[0].action.target
    height, width, _channels = trace.initial.shape
    return (
        max(0, x - radius),
        max(0, y - radius),
        min(width, x + radius + 1),
        min(height, y + radius + 1),
    )


def _trace_commitment(trace: PublicTrace) -> str:
    return manifest_hash(
        {
            "initial": trace.initial.digest(),
            "transitions": [
                {
                    "before": transition.before.digest(),
                    "action": {
                        "code": transition.action.code,
                        "target": list(transition.action.target),
                        "vector": list(transition.action.vector),
                    },
                    "after": transition.after.digest(),
                }
                for transition in trace.transitions
            ],
        }
    )


def _operational_trace_signature(trace: PublicTrace) -> tuple[tuple[object, ...], ...]:
    """Renderer-tolerant signature using only action and RGB consequences."""

    if not isinstance(trace, PublicTrace) or trace.has_feedback or not trace.transitions:
        raise ValueError("operational signatures require nonempty feedback-free traces")
    signature: list[tuple[object, ...]] = []
    for transition in trace.transitions:
        before = np.asarray(transition.before.pixels, dtype=np.int16)
        after = np.asarray(transition.after.pixels, dtype=np.int16)
        changed = np.any(before != after, axis=2)
        count = int(np.count_nonzero(changed))
        if count:
            ys, xs = np.nonzero(changed)
            geometry = (
                count,
                int(np.max(xs) - np.min(xs) + 1),
                int(np.max(ys) - np.min(ys) + 1),
            )
        else:
            geometry = (0, 0, 0)
        signature.append(
            (
                transition.action.code,
                tuple(transition.action.vector),
                transition.after.tick - transition.before.tick,
                geometry,
            )
        )
    return tuple(signature)


def _records_commitment(
    ostensive: Sequence[OstensiveSupportRecord],
    causal: Sequence[CausalSupportRecord],
) -> str:
    return manifest_hash(
        {
            "ostensive": [
                {
                    "scope": record.scope_id,
                    "source": record.source_id,
                    "turn": record.turn.turn_id,
                    "tokens": (
                        list(record.turn.utterance.tokens)
                        if record.turn.utterance is not None
                        else None
                    ),
                    "cue": record.turn.ostensive_pixel_cue,
                    "correction": record.turn.scalar_feedback,
                    "trace": _trace_commitment(record.trace),
                }
                for record in ostensive
            ],
            "causal": [
                {
                    "scope": record.scope_id,
                    "problem": record.problem_id,
                    "hypothesis": record.hypothesis_id,
                    "probe": record.probe_id,
                    "source": record.source_id,
                    "trace": _trace_commitment(record.trace),
                }
                for record in causal
            ],
        }
    )


@dataclass(frozen=True, slots=True)
class Grade3CaseManifest:
    """Learner-safe commitment to the extracted public case."""

    version: str
    observation_shape: tuple[int, int, int]
    ostensive_record_count: int
    causal_record_count: int
    support_source_count: int
    hypothesis_count: int
    probe_count: int
    public_dataset_commitment: str

    def __post_init__(self) -> None:
        if self.version != GRADE3_CASE_VERSION:
            raise ValueError(f"version must equal {GRADE3_CASE_VERSION!r}")
        shape = tuple(
            _integer(value, "observation_shape", minimum=1) for value in self.observation_shape
        )
        if len(shape) != 3 or shape[2] != 3:
            raise ValueError("observation_shape must be an RGB shape")
        object.__setattr__(self, "observation_shape", shape)
        for field in (
            "ostensive_record_count",
            "causal_record_count",
            "support_source_count",
            "hypothesis_count",
            "probe_count",
        ):
            object.__setattr__(self, field, _integer(getattr(self, field), field, minimum=1))
        digest = self.public_dataset_commitment
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError("public_dataset_commitment must be a SHA-256 hex digest")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise ValueError("public_dataset_commitment must be a SHA-256 hex digest") from error


@dataclass(frozen=True, slots=True)
class Grade3PublicCase:
    """Complete learner-facing case; no evaluator capability is reachable."""

    case_manifest: Grade3CaseManifest
    session_manifest: Grade3SessionManifest
    action_space: MotorActionSpace
    ostensive_support: tuple[OstensiveSupportRecord, ...]
    causal_support: tuple[CausalSupportRecord, ...]
    scope_id: int
    problem_id: int
    hypothesis_candidates: tuple[int, ...]
    probe_options: tuple[ProbeOption, ...]
    heldout_instruction: Utterance
    definition_queries: tuple[Utterance, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.case_manifest, Grade3CaseManifest):
            raise TypeError("case_manifest must be Grade3CaseManifest")
        if not isinstance(self.session_manifest, Grade3SessionManifest):
            raise TypeError("session_manifest must be Grade3SessionManifest")
        if not isinstance(self.action_space, MotorActionSpace):
            raise TypeError("action_space must be MotorActionSpace")
        ostensive = tuple(self.ostensive_support)
        causal = tuple(self.causal_support)
        if not ostensive or not all(
            isinstance(value, OstensiveSupportRecord) for value in ostensive
        ):
            raise TypeError("ostensive_support must contain support records")
        if not causal or not all(isinstance(value, CausalSupportRecord) for value in causal):
            raise TypeError("causal_support must contain causal records")
        if any(record.trace.has_feedback for record in (*ostensive, *causal)):
            raise ValueError("every Grade-3 case trace must be feedback-free")
        object.__setattr__(self, "ostensive_support", ostensive)
        object.__setattr__(self, "causal_support", causal)
        object.__setattr__(self, "scope_id", _integer(self.scope_id, "scope_id"))
        object.__setattr__(self, "problem_id", _integer(self.problem_id, "problem_id"))
        candidates = tuple(
            _integer(value, "hypothesis_candidate") for value in self.hypothesis_candidates
        )
        if len(candidates) != _FACTORIAL_CELL_COUNT or len(set(candidates)) != len(candidates):
            raise ValueError("the causal case requires four unique hypotheses")
        options = tuple(self.probe_options)
        if len(options) != 2 or not all(isinstance(value, ProbeOption) for value in options):
            raise ValueError("the causal case requires two public probe options")
        if not isinstance(self.heldout_instruction, Utterance):
            raise TypeError("heldout_instruction must be Utterance")
        definitions = tuple(self.definition_queries)
        if not definitions or not all(isinstance(value, Utterance) for value in definitions):
            raise TypeError("definition_queries must contain Utterance values")
        object.__setattr__(self, "hypothesis_candidates", candidates)
        object.__setattr__(self, "probe_options", options)
        object.__setattr__(self, "definition_queries", definitions)

    @property
    def support_records(self) -> tuple[Grade3SupportRecord, ...]:
        return (*self.ostensive_support, *self.causal_support)

    def offer(
        self,
        *,
        step_index: int,
        remaining_cost: float,
        exclude: Sequence[int] = (),
    ) -> ProbeOffer:
        excluded = {_integer(value, "exclude") for value in exclude}
        options = tuple(option for option in self.probe_options if option.probe_id not in excluded)
        if not options:
            raise ValueError("an offer must retain at least one unexecuted probe")
        return ProbeOffer(
            self.scope_id,
            self.problem_id,
            _integer(step_index, "step_index"),
            options,
            remaining_cost,
        )


@dataclass(frozen=True, slots=True)
class Grade3EvaluatorFacts:
    """Privileged expectations; this record must never enter candidate RPC."""

    affordance_predicate_token: int
    process_predicate_token: int
    positive_role_token: int
    negative_role_token: int
    move_scheme_token: int
    run_scheme_token: int
    demonstrated_factorials: tuple[Utterance, ...]
    heldout_factorial: Utterance
    grounded_definition_base: int
    grounded_definition_middle: int
    grounded_definition_chain: int
    unanchored_cycle: tuple[int, int]
    move_steps: tuple[tuple[int, tuple[int, int]], ...]
    run_steps: tuple[tuple[int, tuple[int, int]], ...]
    hypothesis_patterns: tuple[tuple[int, tuple[bool, bool]], ...]
    true_hypothesis_id: int

    def __post_init__(self) -> None:
        token_fields = (
            "affordance_predicate_token",
            "process_predicate_token",
            "positive_role_token",
            "negative_role_token",
            "move_scheme_token",
            "run_scheme_token",
            "grounded_definition_base",
            "grounded_definition_middle",
            "grounded_definition_chain",
        )
        for field in token_fields:
            object.__setattr__(self, field, _integer(getattr(self, field), field))
        demonstrations = tuple(self.demonstrated_factorials)
        if len(demonstrations) != 3 or not all(
            isinstance(value, Utterance) for value in demonstrations
        ):
            raise ValueError("exactly three factorial cells must be demonstrated")
        if not isinstance(self.heldout_factorial, Utterance):
            raise TypeError("heldout_factorial must be Utterance")
        cycle = tuple(_integer(value, "unanchored_cycle") for value in self.unanchored_cycle)
        if len(cycle) != 2 or cycle[0] == cycle[1]:
            raise ValueError("unanchored_cycle must contain two different tokens")
        patterns = tuple(self.hypothesis_patterns)
        if len(patterns) != 4 or {pattern for _identifier, pattern in patterns} != set(
            _CAUSAL_PATTERNS
        ):
            raise ValueError("hypothesis_patterns must cover the binary factorial")
        if self.true_hypothesis_id not in {identifier for identifier, _pattern in patterns}:
            raise ValueError("true_hypothesis_id must name a registered hypothesis")
        object.__setattr__(self, "demonstrated_factorials", demonstrations)
        object.__setattr__(self, "unanchored_cycle", cycle)
        object.__setattr__(self, "hypothesis_patterns", patterns)


Grade3ProbeCallback = Callable[[], PublicTrace]


@dataclass(frozen=True, slots=True)
class _MotorExpectation:
    required_membership: bool
    action_steps: tuple[tuple[int, tuple[int, int]], ...]
    execution_signature: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class FreshMotorTrial:
    """Evaluator-owned motor world and independent trace scorer.

    Only ``agent`` is handed to the candidate runner.  Scoring consumes the
    evaluator-owned transcript afterward and checks three things jointly:
    the requested public action scheme, its actual RGB consequence, and the
    causal membership of the selected target established by the completed
    feedback-/outcome-free diagnostic probes.
    """

    agent: ProcessWorld
    _expectations: Mapping[tuple[int, ...], _MotorExpectation]
    _positive_diagnostic: tuple[tuple[object, ...], ...]
    _negative_diagnostic: tuple[tuple[object, ...], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.agent, ProcessWorld):
            raise TypeError("agent must be a learner-visible ProcessWorld")
        expectations = dict(self._expectations)
        if not expectations or not all(
            isinstance(value, _MotorExpectation) for value in expectations.values()
        ):
            raise TypeError("motor expectations must be nonempty and typed")
        object.__setattr__(self, "_expectations", MappingProxyType(expectations))
        if self._positive_diagnostic == self._negative_diagnostic:
            raise ValueError("causal diagnostics must distinguish membership")

    def score(
        self,
        trace: PublicTrace | None,
        utterance: Utterance,
        *,
        completed_probes: Sequence[PublicTrace],
    ) -> bool:
        """Score an executed transcript without feedback, outcome, or truth fields."""

        if trace is None or not isinstance(trace, PublicTrace) or trace.has_feedback:
            return False
        if not isinstance(utterance, Utterance):
            raise TypeError("utterance must be Utterance")
        expectation = self._expectations.get(utterance.tokens)
        if expectation is None or not trace.transitions:
            return False
        observed_steps = tuple(
            (transition.action.code, tuple(transition.action.vector))
            for transition in trace.transitions
        )
        if observed_steps != expectation.action_steps:
            return False
        try:
            if _operational_trace_signature(trace) != expectation.execution_signature:
                return False
        except ValueError:
            return False

        probes = tuple(completed_probes)
        if len(probes) < 2 or any(
            not isinstance(probe, PublicTrace) or probe.has_feedback or not probe.transitions
            for probe in probes
        ):
            return False
        execution_target = trace.transitions[0].action.target
        matching = tuple(
            probe for probe in probes if probe.transitions[0].action.target == execution_target
        )
        if len(matching) != 1:
            return False
        try:
            diagnostic = _operational_trace_signature(matching[0])
        except ValueError:
            return False
        expected_diagnostic = (
            self._positive_diagnostic
            if expectation.required_membership
            else self._negative_diagnostic
        )
        return diagnostic == expected_diagnostic

    def score_result(self, result: object) -> bool:
        """Structural convenience adapter for ``MotorEpisodeResult``."""

        return self.score(
            getattr(result, "execution_trace", None),
            getattr(result, "utterance"),
            completed_probes=tuple(getattr(result, "completed_probes")),
        )


class HeldoutProbeBank:
    """Evaluator-only, replayable real-world probe callbacks."""

    __slots__ = (
        "_config",
        "_probe_targets",
        "_renderer_variant",
        "_seed",
        "_true_pattern",
        "_world_variant",
    )

    def __init__(
        self,
        *,
        seed: int,
        config: ProcessConfig,
        renderer_variant: int,
        world_variant: int,
        probe_targets: Mapping[int, int],
        true_pattern: tuple[bool, bool],
    ) -> None:
        self._seed = _integer(seed, "seed")
        if not isinstance(config, ProcessConfig):
            raise TypeError("config must be ProcessConfig")
        self._config = config
        self._renderer_variant = _integer(renderer_variant, "renderer_variant")
        self._world_variant = _integer(world_variant, "world_variant")
        targets = {
            _integer(probe_id, "probe_id"): _integer(target, "target_index")
            for probe_id, target in probe_targets.items()
        }
        if set(targets.values()) != {0, 1} or len(targets) != 2:
            raise ValueError("probe_targets must bijectively cover target indices 0 and 1")
        pattern = tuple(bool(value) for value in true_pattern)
        if len(pattern) != 2:
            raise ValueError("true_pattern must contain two Boolean values")
        self._probe_targets = MappingProxyType(targets)
        self._true_pattern = (pattern[0], pattern[1])

    def execute(self, probe_id: int) -> PublicTrace:
        identifier = _integer(probe_id, "probe_id")
        try:
            target_index = self._probe_targets[identifier]
        except KeyError as error:
            raise KeyError("unknown opaque holdout probe") from error
        harness = ProcessHarness(
            self._seed,
            self._config,
            renderer_variant=self._renderer_variant + target_index,
            world_variant=self._world_variant + target_index,
        )
        positive, negative = harness.oracle.process_pair()
        selected = positive if self._true_pattern[target_index] else negative
        return episode_to_query_trace(selected.episode)

    @property
    def callbacks(self) -> Mapping[int, Grade3ProbeCallback]:
        return MappingProxyType(
            {probe_id: partial(self.execute, probe_id) for probe_id in self._probe_targets}
        )

    def fresh_motor_world(self) -> ProcessWorld:
        """Return a fresh learner-visible world sharing only the opaque codebook."""

        return self._fresh_motor_harness().agent

    def _fresh_motor_harness(self) -> ProcessHarness:
        return ProcessHarness(
            self._seed,
            self._config,
            renderer_variant=self._renderer_variant + 1,
            world_variant=self._world_variant + 1,
        )

    def _fresh_motor_trial(self, facts: Grade3EvaluatorFacts) -> FreshMotorTrial:
        harness = self._fresh_motor_harness()
        positive_run, negative_run = harness.oracle.process_pair()
        positive_move, negative_move = harness.oracle.target_role_pair()
        positive_run_trace = episode_to_query_trace(positive_run.episode)
        negative_run_trace = episode_to_query_trace(negative_run.episode)
        positive_move_trace = episode_to_query_trace(positive_move)
        negative_move_trace = episode_to_query_trace(negative_move)
        positive_diagnostic = _operational_trace_signature(positive_run_trace)
        negative_diagnostic = _operational_trace_signature(negative_run_trace)
        move_positive_signature = _operational_trace_signature(positive_move_trace)
        move_negative_signature = _operational_trace_signature(negative_move_trace)
        expectations = {
            (
                facts.positive_role_token,
                facts.move_scheme_token,
            ): _MotorExpectation(True, facts.move_steps, move_positive_signature),
            (
                facts.negative_role_token,
                facts.move_scheme_token,
            ): _MotorExpectation(False, facts.move_steps, move_negative_signature),
            (
                facts.positive_role_token,
                facts.run_scheme_token,
            ): _MotorExpectation(True, facts.run_steps, positive_diagnostic),
            (
                facts.negative_role_token,
                facts.run_scheme_token,
            ): _MotorExpectation(False, facts.run_steps, negative_diagnostic),
            (facts.grounded_definition_base,): _MotorExpectation(
                False, facts.move_steps, move_negative_signature
            ),
            (facts.grounded_definition_middle,): _MotorExpectation(
                False, facts.move_steps, move_negative_signature
            ),
            (facts.grounded_definition_chain,): _MotorExpectation(
                False, facts.move_steps, move_negative_signature
            ),
        }
        return FreshMotorTrial(
            harness.agent,
            expectations,
            positive_diagnostic,
            negative_diagnostic,
        )

    def heldout_description_trace(self) -> PublicTrace:
        """Evaluator trace for the withheld positive-role/run composition."""

        harness = ProcessHarness(
            self._seed,
            self._config,
            renderer_variant=self._renderer_variant + 2,
            world_variant=self._world_variant + 2,
        )
        return episode_to_query_trace(harness.oracle.process_pair()[0].episode)


@dataclass(frozen=True, slots=True)
class Grade3EvaluatorCase:
    facts: Grade3EvaluatorFacts
    probes: HeldoutProbeBank

    def __post_init__(self) -> None:
        if not isinstance(self.facts, Grade3EvaluatorFacts):
            raise TypeError("facts must be Grade3EvaluatorFacts")
        if not isinstance(self.probes, HeldoutProbeBank):
            raise TypeError("probes must be HeldoutProbeBank")

    def new_motor_trial(self) -> FreshMotorTrial:
        """Construct a fresh world plus evaluator-only operational scorer."""

        return self.probes._fresh_motor_trial(self.facts)


@dataclass(frozen=True, slots=True)
class Grade3CaseBundle:
    """Explicit public/private split for one outer Grade-3 block."""

    public: Grade3PublicCase
    evaluator: Grade3EvaluatorCase

    def __post_init__(self) -> None:
        if not isinstance(self.public, Grade3PublicCase):
            raise TypeError("public must be Grade3PublicCase")
        if not isinstance(self.evaluator, Grade3EvaluatorCase):
            raise TypeError("evaluator must be Grade3EvaluatorCase")


@dataclass(frozen=True, slots=True)
class _SurfaceAssignment:
    affordance: int
    process: int
    positive_role: int
    negative_role: int
    move: int
    run: int
    definition_base: int
    definition_middle: int
    definition_chain: int
    cycle_left: int
    cycle_right: int


def _surface_assignment(
    allocator: _OpaqueAllocator, seed: int, permutation_variant: int
) -> _SurfaceAssignment:
    pool = tuple(allocator.take(f"surface:{index}") for index in range(11))
    values = _permuted(pool, _variant_rng(seed, permutation_variant, 0x51FACE))
    return _SurfaceAssignment(*values)


def _ostensive_records(
    *,
    seed: int,
    config: ProcessConfig,
    renderer_variant: int,
    support_worlds: int,
    scope_id: int,
    source_ids: Sequence[int],
    definition_source_id: int,
    surface: _SurfaceAssignment,
) -> tuple[OstensiveSupportRecord, ...]:
    total = support_worlds * 11 + 5
    specs: list[tuple[int, Utterance, PublicTrace, tuple[int, int, int, int] | None, float]] = []
    definition_observation = None
    for source_index in range(support_worlds):
        harness = ProcessHarness(
            seed,
            config,
            renderer_variant=renderer_variant * 10_007 + source_index,
            world_variant=source_index + 1,
        )
        if definition_observation is None:
            definition_observation = harness.agent.observe()
        affordance_positive, affordance_negative = harness.oracle.affordance_pair()
        process_positive, process_negative = harness.oracle.process_pair()
        move_positive, _move_second = harness.oracle.movement_pair()
        positive_move, negative_move = harness.oracle.target_role_pair()

        affordance_positive_trace = episode_to_query_trace(affordance_positive.episode)
        affordance_negative_trace = episode_to_query_trace(affordance_negative.episode)
        process_positive_trace = episode_to_query_trace(process_positive.episode)
        process_negative_trace = episode_to_query_trace(process_negative.episode)
        move_trace = episode_to_query_trace(move_positive.episode)
        positive_move_trace = episode_to_query_trace(positive_move)
        negative_move_trace = episode_to_query_trace(negative_move)
        source_id = source_ids[source_index]
        specs.extend(
            (
                (
                    source_id,
                    Utterance((surface.affordance,)),
                    affordance_positive_trace,
                    None,
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.affordance,)),
                    affordance_negative_trace,
                    None,
                    -1.0,
                ),
                (
                    source_id,
                    Utterance((surface.process,)),
                    process_positive_trace,
                    None,
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.process,)),
                    process_negative_trace,
                    None,
                    -1.0,
                ),
                (
                    source_id,
                    Utterance((surface.positive_role,)),
                    process_positive_trace,
                    _cue(process_positive_trace),
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.negative_role,)),
                    process_negative_trace,
                    _cue(process_negative_trace),
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.move,)),
                    move_trace,
                    None,
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.run,)),
                    process_positive_trace,
                    None,
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.positive_role, surface.move)),
                    positive_move_trace,
                    None,
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.negative_role, surface.move)),
                    negative_move_trace,
                    None,
                    1.0,
                ),
                (
                    source_id,
                    Utterance((surface.negative_role, surface.run)),
                    process_negative_trace,
                    None,
                    1.0,
                ),
            )
        )

    assert definition_observation is not None
    empty = PublicTrace(definition_observation)
    specs.extend(
        (
            (
                definition_source_id,
                Utterance(
                    (
                        surface.definition_base,
                        surface.negative_role,
                        surface.move,
                    )
                ),
                empty,
                None,
                0.0,
            ),
            (
                definition_source_id,
                Utterance((surface.definition_middle, surface.definition_base)),
                empty,
                None,
                0.0,
            ),
            (
                definition_source_id,
                Utterance((surface.definition_chain, surface.definition_middle)),
                empty,
                None,
                0.0,
            ),
            (
                definition_source_id,
                Utterance((surface.cycle_left, surface.cycle_right)),
                empty,
                None,
                0.0,
            ),
            (
                definition_source_id,
                Utterance((surface.cycle_right, surface.cycle_left)),
                empty,
                None,
                0.0,
            ),
        )
    )
    if len(specs) != total:
        raise AssertionError("the ostensive case cardinality changed unexpectedly")
    records: list[OstensiveSupportRecord] = []
    for turn_id, (source_id, utterance, trace, cue, correction) in enumerate(specs):
        turn = PublicTurn(
            turn_id,
            SessionPhase.SUPPORT,
            trace.initial,
            utterance=utterance,
            ostensive_pixel_cue=cue,
            scalar_feedback=correction,
            remaining_cost=float(total - turn_id - 1),
        )
        records.append(OstensiveSupportRecord(scope_id, source_id, turn, trace))
    return tuple(records)


def _causal_records(
    *,
    seed: int,
    config: ProcessConfig,
    renderer_variant: int,
    support_worlds: int,
    scope_id: int,
    problem_id: int,
    source_ids: Sequence[int],
    hypothesis_patterns: Sequence[tuple[int, tuple[bool, bool]]],
    probe_targets: Mapping[int, int],
) -> tuple[CausalSupportRecord, ...]:
    records: list[CausalSupportRecord] = []
    for source_index in range(support_worlds):
        source_id = source_ids[source_index]
        for probe_id, target_index in probe_targets.items():
            # Each table cell is measured by executing a causal process twin
            # in an independent renderer/world instance.  The hypothesis bit
            # chooses the hidden causal disposition; only its RGB consequence
            # appears in the resulting support record.
            harness = ProcessHarness(
                seed,
                config,
                renderer_variant=(
                    renderer_variant * 10_009 + 5_000 + source_index * 2 + target_index
                ),
                world_variant=5_000 + source_index * 2 + target_index,
            )
            positive, negative = harness.oracle.process_pair()
            for hypothesis_id, pattern in hypothesis_patterns:
                selected = positive if pattern[target_index] else negative
                records.append(
                    CausalSupportRecord(
                        scope_id,
                        problem_id,
                        hypothesis_id,
                        probe_id,
                        source_id,
                        episode_to_query_trace(selected.episode),
                    )
                )
    return tuple(records)


def build_grade3_case(
    seed: int,
    *,
    renderer_variant: int = 0,
    permutation_variant: int = 0,
    support_worlds: int = 3,
    config: ProcessConfig | None = None,
) -> Grade3CaseBundle:
    """Extract one replayable Grade-3 case with a strict public/private split."""

    seed = _integer(seed, "seed")
    renderer_variant = _integer(renderer_variant, "renderer_variant")
    permutation_variant = _integer(permutation_variant, "permutation_variant")
    support_worlds = _integer(support_worlds, "support_worlds", minimum=2)
    selected_config = ProcessConfig() if config is None else config
    if not isinstance(selected_config, ProcessConfig):
        raise TypeError("config must be ProcessConfig or None")

    anchor = ProcessHarness(seed, selected_config)
    process_manifest = anchor.agent.manifest
    allocator = _OpaqueAllocator(
        seed,
        (*process_manifest.action_codes, *process_manifest.concept_codes),
    )
    surface = _surface_assignment(allocator, seed, permutation_variant)
    scope_id = allocator.take("scope")
    problem_id = allocator.take("problem")

    hypothesis_pool = tuple(allocator.take(f"hypothesis:{index}") for index in range(4))
    hypothesis_ids = _permuted(
        hypothesis_pool,
        _variant_rng(seed, permutation_variant, 0xC0A541),
    )
    hypothesis_patterns = tuple(zip(hypothesis_ids, _CAUSAL_PATTERNS, strict=True))

    probe_pool = tuple(allocator.take(f"probe:{index}") for index in range(2))
    probe_ids = _permuted(
        probe_pool,
        _variant_rng(seed, permutation_variant, 0xB0BE),
    )
    probe_targets = MappingProxyType({probe_ids[0]: 0, probe_ids[1]: 1})

    ostensive_source_ids = tuple(
        allocator.take(f"ostensive-source:{index}") for index in range(support_worlds)
    )
    definition_source_id = allocator.take("definition-source")
    causal_source_ids = tuple(
        allocator.take(f"causal-source:{index}") for index in range(support_worlds)
    )
    ostensive = _ostensive_records(
        seed=seed,
        config=selected_config,
        renderer_variant=renderer_variant,
        support_worlds=support_worlds,
        scope_id=scope_id,
        source_ids=ostensive_source_ids,
        definition_source_id=definition_source_id,
        surface=surface,
    )
    causal = _causal_records(
        seed=seed,
        config=selected_config,
        renderer_variant=renderer_variant,
        support_worlds=support_worlds,
        scope_id=scope_id,
        problem_id=problem_id,
        source_ids=causal_source_ids,
        hypothesis_patterns=hypothesis_patterns,
        probe_targets=probe_targets,
    )

    # Order is a nuisance variable.  Turn IDs remain authoritative and opaque
    # record contents remain unchanged under this deterministic presentation shuffle.
    record_rng = _variant_rng(seed, permutation_variant, 0x0D3D)
    ostensive = tuple(ostensive[int(index)] for index in record_rng.permutation(len(ostensive)))
    causal = tuple(causal[int(index)] for index in record_rng.permutation(len(causal)))

    probe_cost = 4.0
    probe_options = tuple(ProbeOption(probe_id, probe_cost) for probe_id in probe_ids)
    action_space = MotorActionSpace(
        process_manifest.action_codes,
        process_manifest.motor_vectors,
        process_manifest.max_steps,
    )
    session_manifest = Grade3SessionManifest(
        GRADE3_PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        len(ostensive) + len(causal),
        probe_cost * len(probe_options),
        16,
        64.0,
        20,
    )
    case_manifest = Grade3CaseManifest(
        GRADE3_CASE_VERSION,
        process_manifest.observation_shape,
        len(ostensive),
        len(causal),
        support_worlds,
        len(hypothesis_ids),
        len(probe_ids),
        _records_commitment(ostensive, causal),
    )

    heldout = Utterance((surface.positive_role, surface.run))
    definition_queries = (
        Utterance((surface.definition_base,)),
        Utterance((surface.definition_middle,)),
        Utterance((surface.definition_chain,)),
        Utterance((surface.cycle_left,)),
        Utterance((surface.cycle_right,)),
    )
    public = Grade3PublicCase(
        case_manifest,
        session_manifest,
        action_space,
        ostensive,
        causal,
        scope_id,
        problem_id,
        hypothesis_ids,
        probe_options,
        heldout,
        definition_queries,
    )

    reference = ProcessHarness(
        seed,
        selected_config,
        renderer_variant=renderer_variant * 10_007,
        world_variant=1,
    )
    move_trace = episode_to_query_trace(reference.oracle.movement_pair()[0].episode)
    run_trace = episode_to_query_trace(reference.oracle.process_pair()[0].episode)
    move_steps = tuple(
        (step.action.code, tuple(step.action.vector)) for step in move_trace.transitions
    )
    run_steps = tuple(
        (step.action.code, tuple(step.action.vector)) for step in run_trace.transitions
    )
    truth_rng = _variant_rng(seed, permutation_variant, 0x7A17)
    true_pattern = _CAUSAL_PATTERNS[int(truth_rng.integers(0, len(_CAUSAL_PATTERNS)))]
    true_hypothesis = next(
        identifier for identifier, pattern in hypothesis_patterns if pattern == true_pattern
    )
    facts = Grade3EvaluatorFacts(
        surface.affordance,
        surface.process,
        surface.positive_role,
        surface.negative_role,
        surface.move,
        surface.run,
        (
            Utterance((surface.positive_role, surface.move)),
            Utterance((surface.negative_role, surface.move)),
            Utterance((surface.negative_role, surface.run)),
        ),
        heldout,
        surface.definition_base,
        surface.definition_middle,
        surface.definition_chain,
        (surface.cycle_left, surface.cycle_right),
        move_steps,
        run_steps,
        hypothesis_patterns,
        true_hypothesis,
    )
    heldout_renderer = renderer_variant * 10_013 + 900_000
    heldout_world = 900_000 + permutation_variant
    probes = HeldoutProbeBank(
        seed=seed,
        config=selected_config,
        renderer_variant=heldout_renderer,
        world_variant=heldout_world,
        probe_targets=probe_targets,
        true_pattern=true_pattern,
    )
    return Grade3CaseBundle(public, Grade3EvaluatorCase(facts, probes))


__all__ = [
    "GRADE3_CASE_VERSION",
    "FreshMotorTrial",
    "Grade3CaseBundle",
    "Grade3CaseManifest",
    "Grade3EvaluatorCase",
    "Grade3EvaluatorFacts",
    "Grade3ProbeCallback",
    "Grade3PublicCase",
    "HeldoutProbeBank",
    "build_grade3_case",
]
