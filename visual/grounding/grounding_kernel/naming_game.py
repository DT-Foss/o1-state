"""Deterministic multi-agent naming games for :mod:`grounding_kernel.social`.

Private vocabularies and round plans are evaluator capabilities.  The learner
view receives only staged public prompts, accepts one option (or abstention),
and releases feedback after that action.  Python object separation is an API
boundary rather than an operating-system sandbox; untrusted learners must use
the repository's process/serialization isolation in a sealed evaluation.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from threading import RLock
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from .social import (
    FEATURE_NAMESPACE,
    RELATION_NAMESPACE,
    ROLE_NAMESPACE,
    ArgumentUtterance,
    ContextCandidate,
    Exchange,
    ListenerAction,
    NamingPrompt,
    ReferentialContext,
    ReferentialFeedback,
    RelationalReferent,
    SocialAgentEnvironment,
    StructuredUtterance,
    Transcript,
    joint_attention_for,
    make_target_commitment,
)


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a strict integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def deterministic_nonce(seed: int | bytes, round_id: int) -> bytes:
    """Return a reproducible 256-bit evaluator nonce for deterministic suites."""

    round_id = _integer(round_id, "round_id")
    if isinstance(seed, bool) or not isinstance(seed, (int, bytes)):
        raise TypeError("seed must be a strict integer or bytes")
    seed_bytes = seed if isinstance(seed, bytes) else str(seed).encode("ascii")
    return sha256(
        b"fertig.social.deterministic-nonce.v1\x00"
        + seed_bytes
        + b"\x00"
        + str(round_id).encode("ascii")
    ).digest()


@dataclass(frozen=True, slots=True, order=True)
class Primitive:
    """Evaluator-only atomic meaning behind one private opaque token."""

    namespace: int
    key: tuple[int, ...]

    def __post_init__(self) -> None:
        namespace = _integer(self.namespace, "namespace")
        if namespace not in {RELATION_NAMESPACE, ROLE_NAMESPACE, FEATURE_NAMESPACE}:
            raise ValueError("unknown primitive namespace")
        key = tuple(_integer(value, "primitive key") for value in self.key)
        expected = 2 if namespace == FEATURE_NAMESPACE else 1
        if len(key) != expected:
            raise ValueError(f"namespace {namespace} requires a key of length {expected}")
        object.__setattr__(self, "namespace", namespace)
        object.__setattr__(self, "key", key)


@dataclass(frozen=True, slots=True)
class VocabularyBinding:
    """Evaluator-only primitive/token binding."""

    primitive: Primitive
    token: int

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, Primitive):
            raise TypeError("primitive must be a Primitive")
        object.__setattr__(self, "token", _integer(self.token, "token"))


def primitives_for(frames: Iterable[RelationalReferent]) -> tuple[Primitive, ...]:
    """Extract the finite compositional alphabet required by a frame family."""

    values: set[Primitive] = set()
    frame_values = tuple(frames)
    if not frame_values:
        raise ValueError("at least one frame is required")
    for frame in frame_values:
        if not isinstance(frame, RelationalReferent):
            raise TypeError("frames must be RelationalReferent records")
        values.add(Primitive(RELATION_NAMESPACE, (frame.relation_code,)))
        for argument in frame.arguments:
            values.add(Primitive(ROLE_NAMESPACE, (argument.role_code,)))
            values.update(
                Primitive(
                    FEATURE_NAMESPACE,
                    (sample.channel_code, sample.value_code),
                )
                for sample in argument.referent.samples
            )
    return tuple(sorted(values))


def _permutation_rank(seed: bytes, agent_id: int, token: int) -> bytes:
    return sha256(
        b"fertig.social.private-vocabulary.v1\x00"
        + seed
        + b"\x00"
        + str(agent_id).encode("ascii")
        + b"\x00"
        + str(token).encode("ascii")
    ).digest()


@dataclass(frozen=True, slots=True)
class PrivateVocabulary:
    """Evaluator-only bijection from operational primitives to integer tokens."""

    agent_id: int
    bindings: tuple[VocabularyBinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "agent_id", _integer(self.agent_id, "agent_id"))
        bindings = tuple(self.bindings)
        if not bindings or any(not isinstance(binding, VocabularyBinding) for binding in bindings):
            raise TypeError("bindings must be a non-empty sequence of VocabularyBinding records")
        primitives = tuple(binding.primitive for binding in bindings)
        tokens = tuple(binding.token for binding in bindings)
        if len(set(primitives)) != len(primitives):
            raise ValueError("private vocabulary primitives must be unique")
        if len(set(tokens)) != len(tokens):
            raise ValueError("private vocabulary tokens must be unique")
        object.__setattr__(
            self,
            "bindings",
            tuple(sorted(bindings, key=lambda binding: binding.primitive)),
        )

    @classmethod
    def permuted(
        cls,
        agent_id: int,
        frames: Iterable[RelationalReferent],
        *,
        seed: int | bytes,
        token_codes: Sequence[int] | None = None,
    ) -> "PrivateVocabulary":
        """Build an agent-salted deterministic permutation over one token pool."""

        agent_id = _integer(agent_id, "agent_id")
        if isinstance(seed, bool) or not isinstance(seed, (int, bytes)):
            raise TypeError("seed must be a strict integer or bytes")
        seed_bytes = seed if isinstance(seed, bytes) else str(seed).encode("ascii")
        primitives = primitives_for(frames)
        if token_codes is None:
            tokens = tuple(range(100_000, 100_000 + len(primitives)))
        else:
            tokens = tuple(_integer(token, "token code") for token in token_codes)
        if len(tokens) != len(primitives) or len(set(tokens)) != len(tokens):
            raise ValueError("token_codes must contain one unique token per primitive")
        permuted_tokens = tuple(
            sorted(tokens, key=lambda token: (_permutation_rank(seed_bytes, agent_id, token), token))
        )
        return cls(
            agent_id,
            tuple(
                VocabularyBinding(primitive, token)
                for primitive, token in zip(primitives, permuted_tokens, strict=True)
            ),
        )

    @property
    def primitive_to_token(self) -> Mapping[Primitive, int]:
        return MappingProxyType(
            {binding.primitive: binding.token for binding in self.bindings}
        )

    @property
    def token_to_primitive(self) -> Mapping[int, Primitive]:
        return MappingProxyType(
            {binding.token: binding.primitive for binding in self.bindings}
        )

    @property
    def fingerprint(self) -> str:
        payload = b"|".join(
            (
                f"{binding.primitive.namespace}:"
                f"{','.join(map(str, binding.primitive.key))}:"
                f"{binding.token}"
            ).encode("ascii")
            for binding in self.bindings
        )
        return sha256(b"fertig.social.vocabulary.v1\x00" + payload).hexdigest()

    def token(self, namespace: int, key: tuple[int, ...]) -> int:
        primitive = Primitive(namespace, key)
        try:
            return self.primitive_to_token[primitive]
        except KeyError as error:
            raise KeyError("private vocabulary does not cover this primitive") from error

    def encode(self, frame: RelationalReferent) -> StructuredUtterance:
        """Compose an utterance without storing any whole-frame phrase."""

        if not isinstance(frame, RelationalReferent):
            raise TypeError("frame must be a RelationalReferent")
        relation_token = self.token(RELATION_NAMESPACE, (frame.relation_code,))
        arguments = tuple(
            ArgumentUtterance(
                self.token(ROLE_NAMESPACE, (argument.role_code,)),
                tuple(
                    self.token(
                        FEATURE_NAMESPACE,
                        (sample.channel_code, sample.value_code),
                    )
                    for sample in argument.referent.samples
                ),
            )
            for argument in frame.arguments
        )
        return StructuredUtterance(relation_token, arguments)


def make_private_vocabularies(
    agent_ids: Iterable[int],
    frames: Iterable[RelationalReferent],
    *,
    seed: int | bytes,
    token_codes: Sequence[int] | None = None,
) -> tuple[PrivateVocabulary, ...]:
    """Create independently agent-salted vocabularies over a shared alphabet."""

    agents = tuple(_integer(agent, "agent_id") for agent in agent_ids)
    if not agents or len(set(agents)) != len(agents):
        raise ValueError("agent_ids must be non-empty and unique")
    frame_values = tuple(frames)
    return tuple(
        PrivateVocabulary.permuted(
            agent,
            frame_values,
            seed=seed,
            token_codes=token_codes,
        )
        for agent in agents
    )


def substitute_tokens(
    utterance: StructuredUtterance,
    substitutions: Iterable[tuple[int, int]],
) -> StructuredUtterance:
    """Apply deterministic token noise without consulting primitive meanings."""

    pairs = tuple(
        (_integer(source, "substitution source"), _integer(target, "substitution target"))
        for source, target in substitutions
    )
    if len({source for source, _ in pairs}) != len(pairs):
        raise ValueError("substitution sources must be unique")
    mapping = dict(pairs)
    return StructuredUtterance(
        mapping.get(utterance.relation_token, utterance.relation_token),
        tuple(
            ArgumentUtterance(
                mapping.get(argument.role_token, argument.role_token),
                tuple(mapping.get(token, token) for token in argument.referent_tokens),
            )
            for argument in utterance.arguments
        ),
    )


@dataclass(frozen=True, slots=True)
class RoundPlan:
    """Evaluator-only target, speaker behavior, and commitment nonce."""

    round_id: int
    speaker_id: int
    listener_id: int
    context: ReferentialContext
    target_option_id: int
    nonce: bytes
    spoken_option_id: int | None = None
    utterance_override: StructuredUtterance | None = None
    attention_event_code: int = 70_001
    ostensive_action_code: int = 70_003

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", _integer(self.round_id, "round_id"))
        object.__setattr__(self, "speaker_id", _integer(self.speaker_id, "speaker_id"))
        object.__setattr__(self, "listener_id", _integer(self.listener_id, "listener_id"))
        if self.speaker_id == self.listener_id:
            raise ValueError("speaker and listener must be different agents")
        if not isinstance(self.context, ReferentialContext):
            raise TypeError("context must be a ReferentialContext")
        target = _integer(self.target_option_id, "target_option_id")
        if self.context.candidate(target) is None:
            raise ValueError("target_option_id must occur in the context")
        object.__setattr__(self, "target_option_id", target)
        if not isinstance(self.nonce, bytes) or len(self.nonce) < 16:
            raise ValueError("nonce must contain at least 16 bytes")
        if self.spoken_option_id is not None:
            spoken = _integer(self.spoken_option_id, "spoken_option_id")
            if self.context.candidate(spoken) is None:
                raise ValueError("spoken_option_id must occur in the context")
            object.__setattr__(self, "spoken_option_id", spoken)
        if self.utterance_override is not None and not isinstance(
            self.utterance_override,
            StructuredUtterance,
        ):
            raise TypeError("utterance_override must be a StructuredUtterance")
        object.__setattr__(
            self,
            "attention_event_code",
            _integer(self.attention_event_code, "attention_event_code"),
        )
        object.__setattr__(
            self,
            "ostensive_action_code",
            _integer(self.ostensive_action_code, "ostensive_action_code"),
        )

    @property
    def deceptive(self) -> bool:
        return self.spoken_option_id not in (None, self.target_option_id)


class RoundStateError(RuntimeError):
    """A prompt/action call violated the staged round protocol."""


class _GameCore:
    def __init__(
        self,
        vocabularies: Iterable[PrivateVocabulary],
        plans: Iterable[RoundPlan],
    ) -> None:
        vocabulary_values = tuple(vocabularies)
        plan_values = tuple(plans)
        if not vocabulary_values or any(
            not isinstance(vocabulary, PrivateVocabulary)
            for vocabulary in vocabulary_values
        ):
            raise TypeError("vocabularies must contain PrivateVocabulary records")
        if not plan_values or any(not isinstance(plan, RoundPlan) for plan in plan_values):
            raise TypeError("plans must contain RoundPlan records")
        round_ids = tuple(plan.round_id for plan in plan_values)
        if round_ids != tuple(sorted(round_ids)) or len(set(round_ids)) != len(round_ids):
            raise ValueError("round plans must have unique increasing round IDs")
        vocabulary_agents = tuple(vocabulary.agent_id for vocabulary in vocabulary_values)
        if len(set(vocabulary_agents)) != len(vocabulary_agents):
            raise ValueError("there must be exactly one vocabulary per agent")
        self.vocabularies = {
            vocabulary.agent_id: vocabulary for vocabulary in vocabulary_values
        }
        participants = {
            participant
            for plan in plan_values
            for participant in (plan.speaker_id, plan.listener_id)
        }
        missing = participants - self.vocabularies.keys()
        if missing:
            raise ValueError(f"agents without private vocabularies: {sorted(missing)}")
        for plan in plan_values:
            spoken_option = (
                plan.target_option_id
                if plan.spoken_option_id is None
                else plan.spoken_option_id
            )
            spoken_candidate = plan.context.candidate(spoken_option)
            assert spoken_candidate is not None
            if plan.utterance_override is None:
                self.vocabularies[plan.speaker_id].encode(spoken_candidate.referent)
        self.plans = plan_values
        self.index = 0
        self.active: NamingPrompt | None = None
        self.transcript = Transcript()
        self.lock = RLock()

    @property
    def complete(self) -> bool:
        return self.index == len(self.plans) and self.active is None

    def current_plan(self) -> RoundPlan:
        if self.index >= len(self.plans):
            raise StopIteration("naming game is complete")
        return self.plans[self.index]

    def open_round(self) -> NamingPrompt:
        with self.lock:
            if self.active is not None:
                raise RoundStateError("the active prompt requires a listener action")
            plan = self.current_plan()
            spoken_option = (
                plan.target_option_id
                if plan.spoken_option_id is None
                else plan.spoken_option_id
            )
            spoken_candidate = plan.context.candidate(spoken_option)
            assert spoken_candidate is not None
            utterance = plan.utterance_override or self.vocabularies[
                plan.speaker_id
            ].encode(spoken_candidate.referent)
            commitment = make_target_commitment(
                round_id=plan.round_id,
                speaker_id=plan.speaker_id,
                listener_id=plan.listener_id,
                context=plan.context,
                utterance=utterance,
                target_option_id=plan.target_option_id,
                nonce=plan.nonce,
            )
            self.active = NamingPrompt(
                plan.round_id,
                plan.speaker_id,
                plan.listener_id,
                plan.context,
                utterance,
                commitment,
            )
            return self.active

    def submit(self, option_id: int | None) -> ReferentialFeedback:
        with self.lock:
            if self.active is None:
                raise RoundStateError("next_prompt must be called before submit")
            plan = self.current_plan()
            if option_id is not None:
                option_id = _integer(option_id, "option_id")
                if plan.context.candidate(option_id) is None:
                    raise ValueError("option_id must occur in the active context")
            action = ListenerAction(plan.round_id, plan.listener_id, option_id)
            target = plan.context.candidate(plan.target_option_id)
            assert target is not None
            feedback = ReferentialFeedback(
                plan.round_id,
                plan.target_option_id,
                option_id == plan.target_option_id,
                plan.nonce,
                joint_attention_for(
                    target.referent,
                    (plan.speaker_id, plan.listener_id),
                    event_code=plan.attention_event_code,
                    action_code=plan.ostensive_action_code,
                    initial_tick=plan.round_id * 10_000,
                ),
            )
            exchange = Exchange(self.active, action, feedback)
            self.transcript = self.transcript.append(exchange)
            self.index += 1
            self.active = None
            return feedback


class AgentNamingGame(SocialAgentEnvironment):
    """Narrow learner capability for one staged multi-agent game."""

    __slots__ = ("__core",)

    def __init__(self, core: _GameCore) -> None:
        self.__core = core

    @property
    def transcript(self) -> Transcript:
        return self.__core.transcript

    @property
    def complete(self) -> bool:
        return self.__core.complete

    @property
    def pending_prompt(self) -> NamingPrompt | None:
        return self.__core.active

    def next_prompt(self) -> NamingPrompt:
        return self.__core.open_round()

    def submit(self, option_id: int | None) -> ReferentialFeedback:
        return self.__core.submit(option_id)


@runtime_checkable
class SocialEvaluatorOracle(Protocol):
    """Privileged evaluator capability; never pass it to a learner."""

    @property
    def transcript(self) -> Transcript: ...

    @property
    def complete(self) -> bool: ...

    def private_vocabulary(self, agent_id: int) -> PrivateVocabulary: ...

    def plan(self, round_id: int) -> RoundPlan: ...

    def verify(self) -> bool: ...


class NamingGameOracle(SocialEvaluatorOracle):
    """Evaluator view of private vocabularies, targets, and completed evidence."""

    __slots__ = ("__core",)

    def __init__(self, core: _GameCore) -> None:
        self.__core = core

    @property
    def transcript(self) -> Transcript:
        return self.__core.transcript

    @property
    def complete(self) -> bool:
        return self.__core.complete

    @property
    def active_target_option_id(self) -> int | None:
        if self.__core.active is None:
            return None
        return self.__core.current_plan().target_option_id

    @property
    def vocabulary_fingerprints(self) -> Mapping[int, str]:
        return MappingProxyType(
            {
                agent: vocabulary.fingerprint
                for agent, vocabulary in sorted(self.__core.vocabularies.items())
            }
        )

    def private_vocabulary(self, agent_id: int) -> PrivateVocabulary:
        agent_id = _integer(agent_id, "agent_id")
        try:
            return self.__core.vocabularies[agent_id]
        except KeyError as error:
            raise KeyError("unknown agent") from error

    def plan(self, round_id: int) -> RoundPlan:
        round_id = _integer(round_id, "round_id")
        try:
            return next(plan for plan in self.__core.plans if plan.round_id == round_id)
        except StopIteration as error:
            raise KeyError("unknown round") from error

    def verify(self) -> bool:
        return self.__core.transcript.verify()


def create_naming_game(
    vocabularies: Iterable[PrivateVocabulary],
    plans: Iterable[RoundPlan],
) -> tuple[AgentNamingGame, NamingGameOracle]:
    """Return capability-separated learner and evaluator views of one game."""

    core = _GameCore(vocabularies, plans)
    return AgentNamingGame(core), NamingGameOracle(core)


def context(
    options: Iterable[tuple[int, RelationalReferent]],
) -> ReferentialContext:
    """Compact constructor for deterministic experiments."""

    return ReferentialContext(
        tuple(ContextCandidate(option_id, referent) for option_id, referent in options)
    )


__all__ = [
    "AgentNamingGame",
    "NamingGameOracle",
    "Primitive",
    "PrivateVocabulary",
    "RoundPlan",
    "RoundStateError",
    "SocialEvaluatorOracle",
    "VocabularyBinding",
    "context",
    "create_naming_game",
    "deterministic_nonce",
    "make_private_vocabularies",
    "primitives_for",
    "substitute_tokens",
]
