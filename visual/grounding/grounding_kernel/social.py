"""Auditable social grounding over opaque operational referents.

This module deliberately has no text encoder, token spelling feature, semantic
ontology, or pretrained component.  Every learner-visible atom is a strict
integer.  A convention learner can therefore use only equality, recurrence,
public perceptual/interventional traces, and post-action referential feedback.

The protocol separates a round into an immutable :class:`NamingPrompt`, a
listener :class:`ListenerAction`, and :class:`ReferentialFeedback`.  The target,
success bit, commitment nonce, and joint-attention trace occur only in the last
record.  :class:`Exchange` validates the ordering and the target commitment,
while :class:`Transcript` rebuilds and validates a hash-chain ledger.

The reference :class:`ConventionLearner` is intentionally conservative.  It
learns speaker-specific atomic counters for relation, role, and operational
feature tokens.  It resolves a token only after a configurable evidence floor
and dominance margin, composes unseen frames from those atoms, and abstains on
unknown, ambiguous, duplicated, or unreliable readings.  This handles bounded
noise or occasional deception, but—as no feedback-only system can—cannot
identify a consistently deceptive majority without an external trust anchor.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from hashlib import sha256
import json
from types import MappingProxyType
from typing import Protocol, TypeAlias, runtime_checkable


OpaqueCode: TypeAlias = int
PrimitiveKey: TypeAlias = tuple[int, ...]

RELATION_NAMESPACE = 0
ROLE_NAMESPACE = 1
FEATURE_NAMESPACE = 2

_HEX_DIGEST_LENGTH = 64
_MIN_NONCE_BYTES = 16
_LEDGER_GENESIS = sha256(b"fertig.social.ledger.v1").hexdigest()


def _integer(value: object, field_name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be a strict integer")
    if value < minimum:
        raise ValueError(f"{field_name} must be at least {minimum}")
    return value


def _optional_integer(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _integer(value, field_name)


def _codes(values: Iterable[object], field_name: str) -> tuple[int, ...]:
    return tuple(_integer(value, field_name) for value in values)


def _digest(payload: object, *, domain: str) -> str:
    encoded = json.dumps(
        {"domain": domain, "payload": payload},
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _valid_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != _HEX_DIGEST_LENGTH:
        raise ValueError(f"{field_name} must be a 64-character lowercase hex digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field_name} must be a 64-character lowercase hex digest")
    return value


@dataclass(frozen=True, slots=True, order=True)
class OpaqueSample:
    """One learner-visible channel/value pair with no semantic name."""

    channel_code: OpaqueCode
    value_code: OpaqueCode

    def __post_init__(self) -> None:
        object.__setattr__(self, "channel_code", _integer(self.channel_code, "channel_code"))
        object.__setattr__(self, "value_code", _integer(self.value_code, "value_code"))


@dataclass(frozen=True, slots=True)
class OperationalReferent:
    """An ordered operational trace with unique opaque channel codes.

    Order is learner-visible sensor/trace structure.  It is intentionally not
    reconstructed by sorting the opaque channel numbers: numeric sorting would
    cease to be equivariant under a fresh channel-code permutation.
    """

    samples: tuple[OpaqueSample, ...]

    def __post_init__(self) -> None:
        samples = tuple(self.samples)
        if not samples:
            raise ValueError("an operational referent requires at least one sample")
        if any(not isinstance(sample, OpaqueSample) for sample in samples):
            raise TypeError("referent samples must be OpaqueSample records")
        channels = tuple(sample.channel_code for sample in samples)
        if len(set(channels)) != len(channels):
            raise ValueError("an operational referent has at most one value per channel")
        object.__setattr__(self, "samples", samples)


@dataclass(frozen=True, slots=True)
class RoleArgument:
    """An opaque structural role bound to an operational referent."""

    role_code: OpaqueCode
    referent: OperationalReferent = field(compare=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_code", _integer(self.role_code, "role_code"))
        if not isinstance(self.referent, OperationalReferent):
            raise TypeError("referent must be an OperationalReferent")


@dataclass(frozen=True, slots=True)
class RelationalReferent:
    """A compositional relation with an ordered public argument structure.

    Role codes are unique but never numerically sorted; the sequence is the
    grammar's structural order and survives arbitrary role-code permutations.
    """

    relation_code: OpaqueCode
    arguments: tuple[RoleArgument, ...]

    def __post_init__(self) -> None:
        relation_code = _integer(self.relation_code, "relation_code")
        arguments = tuple(self.arguments)
        if not arguments:
            raise ValueError("a relational referent requires at least one role argument")
        if any(not isinstance(argument, RoleArgument) for argument in arguments):
            raise TypeError("arguments must be RoleArgument records")
        roles = tuple(argument.role_code for argument in arguments)
        if len(set(roles)) != len(roles):
            raise ValueError("role codes must be unique within a relational referent")
        object.__setattr__(self, "relation_code", relation_code)
        object.__setattr__(self, "arguments", arguments)

    def argument(self, role_code: int) -> RoleArgument | None:
        role_code = _integer(role_code, "role_code")
        return next(
            (argument for argument in self.arguments if argument.role_code == role_code),
            None,
        )


@dataclass(frozen=True, slots=True)
class PublicTraceEvent:
    """One public, integer-only attention or intervention observation."""

    tick: int
    agent_id: OpaqueCode
    event_code: OpaqueCode
    role_code: OpaqueCode
    referent: OperationalReferent
    action_code: OpaqueCode | None = None
    outcome_code: OpaqueCode | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tick", _integer(self.tick, "tick"))
        object.__setattr__(self, "agent_id", _integer(self.agent_id, "agent_id"))
        object.__setattr__(self, "event_code", _integer(self.event_code, "event_code"))
        object.__setattr__(self, "role_code", _integer(self.role_code, "role_code"))
        if not isinstance(self.referent, OperationalReferent):
            raise TypeError("referent must be an OperationalReferent")
        object.__setattr__(
            self,
            "action_code",
            _optional_integer(self.action_code, "action_code"),
        )
        object.__setattr__(
            self,
            "outcome_code",
            _optional_integer(self.outcome_code, "outcome_code"),
        )


@dataclass(frozen=True, slots=True)
class JointAttentionTrace:
    """Public ostension attested independently by all listed participants."""

    participants: tuple[OpaqueCode, ...]
    events: tuple[PublicTraceEvent, ...]

    def __post_init__(self) -> None:
        participants = _codes(self.participants, "participant")
        events = tuple(self.events)
        if len(participants) < 2 or len(set(participants)) != len(participants):
            raise ValueError("joint attention requires at least two unique participants")
        if not events or any(not isinstance(event, PublicTraceEvent) for event in events):
            raise TypeError("events must be a non-empty sequence of PublicTraceEvent records")
        if any(event.agent_id not in participants for event in events):
            raise ValueError("every trace event must belong to a declared participant")
        if tuple(event.tick for event in events) != tuple(
            sorted(event.tick for event in events)
        ):
            raise ValueError("joint-attention events must be ordered by nondecreasing tick")
        object.__setattr__(self, "participants", participants)
        object.__setattr__(self, "events", events)

    def attests(self, target: RelationalReferent) -> bool:
        """Return whether every participant co-attended every target argument.

        A relation is operationally attested when at least one public event has
        its opaque relation code as the intervention outcome.  Neither this
        method nor the learner decodes what that code means.
        """

        if not isinstance(target, RelationalReferent):
            raise TypeError("target must be a RelationalReferent")
        for argument in target.arguments:
            for participant in self.participants:
                if not any(
                    event.agent_id == participant
                    and event.role_code == argument.role_code
                    and event.referent == argument.referent
                    for event in self.events
                ):
                    return False
        return any(event.outcome_code == target.relation_code for event in self.events)


def joint_attention_for(
    target: RelationalReferent,
    participants: Iterable[int],
    *,
    event_code: int,
    action_code: int,
    initial_tick: int = 0,
) -> JointAttentionTrace:
    """Construct a canonical public ostensive trace for a grounded frame."""

    participant_codes = _codes(participants, "participant")
    if len(participant_codes) < 2 or len(set(participant_codes)) != len(participant_codes):
        raise ValueError("joint attention requires at least two unique participants")
    event_code = _integer(event_code, "event_code")
    action_code = _integer(action_code, "action_code")
    tick = _integer(initial_tick, "initial_tick")
    events: list[PublicTraceEvent] = []
    for argument in target.arguments:
        for participant in participant_codes:
            events.append(
                PublicTraceEvent(
                    tick,
                    participant,
                    event_code,
                    argument.role_code,
                    argument.referent,
                    action_code,
                    target.relation_code,
                )
            )
            tick += 1
    return JointAttentionTrace(participant_codes, tuple(events))


@dataclass(frozen=True, slots=True)
class ContextCandidate:
    """One action option and its learner-visible operational description."""

    option_id: OpaqueCode
    referent: RelationalReferent
    public_trace: tuple[PublicTraceEvent, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "option_id", _integer(self.option_id, "option_id"))
        if not isinstance(self.referent, RelationalReferent):
            raise TypeError("referent must be a RelationalReferent")
        trace = tuple(self.public_trace)
        if any(not isinstance(event, PublicTraceEvent) for event in trace):
            raise TypeError("public_trace must contain PublicTraceEvent records")
        object.__setattr__(self, "public_trace", trace)


@dataclass(frozen=True, slots=True)
class ReferentialContext:
    """An ordered action set; duplicate meanings remain explicitly ambiguous."""

    candidates: tuple[ContextCandidate, ...]

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if len(candidates) < 2:
            raise ValueError("a naming-game context requires at least two candidates")
        if any(not isinstance(candidate, ContextCandidate) for candidate in candidates):
            raise TypeError("candidates must be ContextCandidate records")
        option_ids = tuple(candidate.option_id for candidate in candidates)
        if len(set(option_ids)) != len(option_ids):
            raise ValueError("context option IDs must be unique")
        object.__setattr__(self, "candidates", candidates)

    @property
    def option_ids(self) -> tuple[int, ...]:
        return tuple(candidate.option_id for candidate in self.candidates)

    def candidate(self, option_id: int) -> ContextCandidate | None:
        option_id = _integer(option_id, "option_id")
        return next(
            (candidate for candidate in self.candidates if candidate.option_id == option_id),
            None,
        )

    def ordinal(self, option_id: int) -> int | None:
        option_id = _integer(option_id, "option_id")
        return next(
            (index for index, candidate in enumerate(self.candidates) if candidate.option_id == option_id),
            None,
        )


@dataclass(frozen=True, slots=True)
class ArgumentUtterance:
    """Opaque tokens for one role and its operational referent."""

    role_token: OpaqueCode
    referent_tokens: tuple[OpaqueCode, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "role_token", _integer(self.role_token, "role_token"))
        tokens = _codes(self.referent_tokens, "referent_token")
        if not tokens:
            raise ValueError("an argument utterance requires at least one referent token")
        object.__setattr__(self, "referent_tokens", tokens)


@dataclass(frozen=True, slots=True)
class StructuredUtterance:
    """A relation token followed by compositional role/referent phrases."""

    relation_token: OpaqueCode
    arguments: tuple[ArgumentUtterance, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relation_token",
            _integer(self.relation_token, "relation_token"),
        )
        arguments = tuple(self.arguments)
        if not arguments or any(not isinstance(arg, ArgumentUtterance) for arg in arguments):
            raise TypeError("arguments must be a non-empty sequence of ArgumentUtterance records")
        object.__setattr__(self, "arguments", arguments)

    @property
    def surface(self) -> tuple[int, ...]:
        """Return an untyped full-utterance key for explicit shortcut controls."""

        flattened: list[int] = [self.relation_token, len(self.arguments)]
        for argument in self.arguments:
            flattened.extend((argument.role_token, len(argument.referent_tokens)))
            flattened.extend(argument.referent_tokens)
        return tuple(flattened)


def _wire_sample(sample: OpaqueSample) -> list[int]:
    return [sample.channel_code, sample.value_code]


def _wire_operational(referent: OperationalReferent) -> list[list[int]]:
    return [_wire_sample(sample) for sample in referent.samples]


def _wire_frame(referent: RelationalReferent) -> dict[str, object]:
    return {
        "relation": referent.relation_code,
        "arguments": [
            {
                "role": argument.role_code,
                "referent": _wire_operational(argument.referent),
            }
            for argument in referent.arguments
        ],
    }


def _wire_event(event: PublicTraceEvent) -> dict[str, object]:
    return {
        "tick": event.tick,
        "agent": event.agent_id,
        "event": event.event_code,
        "role": event.role_code,
        "referent": _wire_operational(event.referent),
        "action": event.action_code,
        "outcome": event.outcome_code,
    }


def _wire_context(context: ReferentialContext) -> list[dict[str, object]]:
    return [
        {
            "option": candidate.option_id,
            "referent": _wire_frame(candidate.referent),
            "trace": [_wire_event(event) for event in candidate.public_trace],
        }
        for candidate in context.candidates
    ]


def _wire_utterance(utterance: StructuredUtterance) -> dict[str, object]:
    return {
        "relation_token": utterance.relation_token,
        "arguments": [
            {
                "role_token": argument.role_token,
                "referent_tokens": list(argument.referent_tokens),
            }
            for argument in utterance.arguments
        ],
    }


def make_target_commitment(
    *,
    round_id: int,
    speaker_id: int,
    listener_id: int,
    context: ReferentialContext,
    utterance: StructuredUtterance,
    target_option_id: int,
    nonce: bytes,
) -> str:
    """Commit to a target and complete public prompt before listener action."""

    round_id = _integer(round_id, "round_id")
    speaker_id = _integer(speaker_id, "speaker_id")
    listener_id = _integer(listener_id, "listener_id")
    target_option_id = _integer(target_option_id, "target_option_id")
    if context.candidate(target_option_id) is None:
        raise ValueError("target_option_id must occur in the context")
    if not isinstance(nonce, bytes) or len(nonce) < _MIN_NONCE_BYTES:
        raise ValueError(f"commitment nonce must contain at least {_MIN_NONCE_BYTES} bytes")
    return _digest(
        {
            "round": round_id,
            "speaker": speaker_id,
            "listener": listener_id,
            "context": _wire_context(context),
            "utterance": _wire_utterance(utterance),
            "target": target_option_id,
            "nonce": nonce.hex(),
        },
        domain="fertig.social.target-commitment.v1",
    )


@dataclass(frozen=True, slots=True)
class NamingPrompt:
    """The complete pre-action learner view; it contains no target or feedback."""

    round_id: int
    speaker_id: OpaqueCode
    listener_id: OpaqueCode
    context: ReferentialContext
    utterance: StructuredUtterance
    target_commitment: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", _integer(self.round_id, "round_id"))
        object.__setattr__(self, "speaker_id", _integer(self.speaker_id, "speaker_id"))
        object.__setattr__(self, "listener_id", _integer(self.listener_id, "listener_id"))
        if self.speaker_id == self.listener_id:
            raise ValueError("speaker and listener must be distinct agents")
        if not isinstance(self.context, ReferentialContext):
            raise TypeError("context must be a ReferentialContext")
        if not isinstance(self.utterance, StructuredUtterance):
            raise TypeError("utterance must be a StructuredUtterance")
        object.__setattr__(
            self,
            "target_commitment",
            _valid_digest(self.target_commitment, "target_commitment"),
        )


@dataclass(frozen=True, slots=True)
class ListenerAction:
    """A context option, or ``None`` for an explicit abstention."""

    round_id: int
    listener_id: OpaqueCode
    option_id: OpaqueCode | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", _integer(self.round_id, "round_id"))
        object.__setattr__(self, "listener_id", _integer(self.listener_id, "listener_id"))
        object.__setattr__(self, "option_id", _optional_integer(self.option_id, "option_id"))


@dataclass(frozen=True, slots=True)
class ReferentialFeedback:
    """Post-action success, target reveal, nonce reveal, and public ostension."""

    round_id: int
    target_option_id: OpaqueCode
    success: bool
    nonce: bytes
    joint_attention: JointAttentionTrace

    def __post_init__(self) -> None:
        object.__setattr__(self, "round_id", _integer(self.round_id, "round_id"))
        object.__setattr__(
            self,
            "target_option_id",
            _integer(self.target_option_id, "target_option_id"),
        )
        if not isinstance(self.success, bool):
            raise TypeError("success must be boolean")
        if not isinstance(self.nonce, bytes) or len(self.nonce) < _MIN_NONCE_BYTES:
            raise ValueError(f"feedback nonce must contain at least {_MIN_NONCE_BYTES} bytes")
        if not isinstance(self.joint_attention, JointAttentionTrace):
            raise TypeError("joint_attention must be a JointAttentionTrace")


@dataclass(frozen=True, slots=True)
class Exchange:
    """One complete, commitment-checked prompt/action/feedback exchange."""

    prompt: NamingPrompt
    action: ListenerAction
    feedback: ReferentialFeedback

    def __post_init__(self) -> None:
        if not isinstance(self.prompt, NamingPrompt):
            raise TypeError("prompt must be a NamingPrompt")
        if not isinstance(self.action, ListenerAction):
            raise TypeError("action must be a ListenerAction")
        if not isinstance(self.feedback, ReferentialFeedback):
            raise TypeError("feedback must be ReferentialFeedback")
        if len({self.prompt.round_id, self.action.round_id, self.feedback.round_id}) != 1:
            raise ValueError("prompt, action, and feedback round IDs must agree")
        if self.action.listener_id != self.prompt.listener_id:
            raise ValueError("only the designated listener may act")
        if self.action.option_id is not None and self.prompt.context.candidate(
            self.action.option_id
        ) is None:
            raise ValueError("listener action must select a context option or abstain")
        target = self.prompt.context.candidate(self.feedback.target_option_id)
        if target is None:
            raise ValueError("feedback target must occur in the prompt context")
        expected_success = self.action.option_id == self.feedback.target_option_id
        if self.feedback.success is not expected_success:
            raise ValueError("feedback success does not match the committed action and target")
        expected_commitment = make_target_commitment(
            round_id=self.prompt.round_id,
            speaker_id=self.prompt.speaker_id,
            listener_id=self.prompt.listener_id,
            context=self.prompt.context,
            utterance=self.prompt.utterance,
            target_option_id=self.feedback.target_option_id,
            nonce=self.feedback.nonce,
        )
        if self.prompt.target_commitment != expected_commitment:
            raise ValueError("target reveal does not open the pre-action commitment")
        required_participants = {self.prompt.speaker_id, self.prompt.listener_id}
        if not required_participants.issubset(self.feedback.joint_attention.participants):
            raise ValueError("joint-attention trace must include speaker and listener")
        if not self.feedback.joint_attention.attests(target.referent):
            raise ValueError("joint-attention trace does not attest the revealed target")


def _wire_prompt(prompt: NamingPrompt) -> dict[str, object]:
    return {
        "round": prompt.round_id,
        "speaker": prompt.speaker_id,
        "listener": prompt.listener_id,
        "context": _wire_context(prompt.context),
        "utterance": _wire_utterance(prompt.utterance),
        "commitment": prompt.target_commitment,
    }


def _wire_action(action: ListenerAction) -> dict[str, object]:
    return {
        "round": action.round_id,
        "listener": action.listener_id,
        "option": action.option_id,
    }


def _wire_feedback(feedback: ReferentialFeedback) -> dict[str, object]:
    return {
        "round": feedback.round_id,
        "target": feedback.target_option_id,
        "success": feedback.success,
        "nonce": feedback.nonce.hex(),
        "joint_attention": {
            "participants": list(feedback.joint_attention.participants),
            "events": [_wire_event(event) for event in feedback.joint_attention.events],
        },
    }


@dataclass(frozen=True, slots=True)
class HashChainRecord:
    """One domain-separated hash-chain commitment to a transcript event."""

    index: int
    event_type: str
    payload_digest: str
    previous_digest: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _integer(self.index, "index"))
        if self.event_type not in {"prompt", "action", "feedback"}:
            raise ValueError("event_type must be prompt, action, or feedback")
        for field_name in ("payload_digest", "previous_digest", "digest"):
            object.__setattr__(self, field_name, _valid_digest(getattr(self, field_name), field_name))


def _record_digest(
    index: int,
    event_type: str,
    payload_digest: str,
    previous_digest: str,
) -> str:
    return _digest(
        {
            "index": index,
            "event_type": event_type,
            "payload_digest": payload_digest,
            "previous_digest": previous_digest,
        },
        domain="fertig.social.ledger-record.v1",
    )


@dataclass(frozen=True, slots=True)
class HashChainLedger:
    """Persistent hash-chain whose head commits to all preceding events."""

    records: tuple[HashChainRecord, ...] = ()

    def __post_init__(self) -> None:
        records = tuple(self.records)
        if any(not isinstance(record, HashChainRecord) for record in records):
            raise TypeError("records must be HashChainRecord values")
        object.__setattr__(self, "records", records)
        if not self.verify():
            raise ValueError("invalid hash-chain ledger")

    @property
    def head(self) -> str:
        return self.records[-1].digest if self.records else _LEDGER_GENESIS

    def append(self, event_type: str, payload: object) -> "HashChainLedger":
        if event_type not in {"prompt", "action", "feedback"}:
            raise ValueError("event_type must be prompt, action, or feedback")
        payload_digest = _digest(payload, domain=f"fertig.social.{event_type}.v1")
        index = len(self.records)
        digest = _record_digest(index, event_type, payload_digest, self.head)
        record = HashChainRecord(index, event_type, payload_digest, self.head, digest)
        return HashChainLedger(self.records + (record,))

    def verify(self) -> bool:
        previous = _LEDGER_GENESIS
        for expected_index, record in enumerate(self.records):
            if record.index != expected_index or record.previous_digest != previous:
                return False
            if record.digest != _record_digest(
                record.index,
                record.event_type,
                record.payload_digest,
                record.previous_digest,
            ):
                return False
            previous = record.digest
        return True


def _extend_ledger(ledger: HashChainLedger, exchange: Exchange) -> HashChainLedger:
    return (
        ledger.append("prompt", _wire_prompt(exchange.prompt))
        .append("action", _wire_action(exchange.action))
        .append("feedback", _wire_feedback(exchange.feedback))
    )


@dataclass(frozen=True, slots=True)
class Transcript:
    """Immutable completed exchanges plus a deterministically rebuilt ledger."""

    exchanges: tuple[Exchange, ...] = ()
    ledger: HashChainLedger = field(default_factory=HashChainLedger)

    def __post_init__(self) -> None:
        exchanges = tuple(self.exchanges)
        if any(not isinstance(exchange, Exchange) for exchange in exchanges):
            raise TypeError("exchanges must be Exchange records")
        round_ids = tuple(exchange.prompt.round_id for exchange in exchanges)
        if round_ids != tuple(sorted(round_ids)) or len(set(round_ids)) != len(round_ids):
            raise ValueError("transcript round IDs must be unique and increasing")
        expected = HashChainLedger()
        for exchange in exchanges:
            expected = _extend_ledger(expected, exchange)
        if self.ledger != expected:
            raise ValueError("ledger does not commit to exactly these transcript exchanges")
        object.__setattr__(self, "exchanges", exchanges)

    @property
    def head(self) -> str:
        return self.ledger.head

    def append(self, exchange: Exchange) -> "Transcript":
        if not isinstance(exchange, Exchange):
            raise TypeError("exchange must be an Exchange")
        if self.exchanges and exchange.prompt.round_id <= self.exchanges[-1].prompt.round_id:
            raise ValueError("appended round IDs must increase")
        return Transcript(
            self.exchanges + (exchange,),
            _extend_ledger(self.ledger, exchange),
        )

    def prefix(self, length: int) -> "Transcript":
        length = _integer(length, "length")
        if length > len(self.exchanges):
            raise ValueError("prefix length exceeds the transcript")
        result = Transcript()
        for exchange in self.exchanges[:length]:
            result = result.append(exchange)
        return result

    def verify(self) -> bool:
        try:
            Transcript(self.exchanges, self.ledger)
        except (TypeError, ValueError):
            return False
        return True


class DecisionStatus(str, Enum):
    """Epistemic status of a listener decision."""

    RESOLVED = "resolved"
    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"
    UNRELIABLE = "unreliable"


@dataclass(frozen=True, slots=True)
class BindingDecision:
    namespace: int
    token: int
    status: DecisionStatus
    primitive: PrimitiveKey | None
    candidates: tuple[PrimitiveKey, ...]
    observations: int
    confidence: float

    @property
    def resolved(self) -> bool:
        return self.status is DecisionStatus.RESOLVED and self.primitive is not None


@dataclass(frozen=True, slots=True)
class ReferentialDecision:
    status: DecisionStatus
    option_id: int | None
    candidates: tuple[int, ...]
    confidence: float
    evidence_count: int
    reason: str

    @property
    def resolved(self) -> bool:
        return self.status is DecisionStatus.RESOLVED and self.option_id is not None

    @property
    def abstained(self) -> bool:
        return self.option_id is None


@dataclass(frozen=True, slots=True)
class SpeakerAssessment:
    speaker_id: int
    completed_rounds: int
    usable_rounds: int
    consistency: float
    conflicting_bindings: int


class ConventionLearner:
    """Speaker-specific, equality-only induction of a compositional convention."""

    def __init__(
        self,
        agent_id: int,
        *,
        minimum_evidence: int = 2,
        dominance: float = 0.75,
        minimum_margin: int = 1,
        minimum_speaker_consistency: float = 0.60,
    ) -> None:
        self.agent_id = _integer(agent_id, "agent_id")
        self.minimum_evidence = _integer(minimum_evidence, "minimum_evidence", minimum=1)
        if not 0.5 < dominance <= 1.0:
            raise ValueError("dominance must lie in (0.5, 1]")
        self.dominance = float(dominance)
        self.minimum_margin = _integer(minimum_margin, "minimum_margin", minimum=1)
        if not 0.0 <= minimum_speaker_consistency <= 1.0:
            raise ValueError("minimum_speaker_consistency must lie in [0, 1]")
        self.minimum_speaker_consistency = float(minimum_speaker_consistency)
        self._counts: defaultdict[
            tuple[int, int, int], Counter[PrimitiveKey]
        ] = defaultdict(Counter)
        self._completed: Counter[int] = Counter()
        self._usable: Counter[int] = Counter()
        self._seen: set[str] = set()

    def _reset(self) -> None:
        self._counts.clear()
        self._completed.clear()
        self._usable.clear()
        self._seen.clear()

    @staticmethod
    def _exchange_id(exchange: Exchange) -> str:
        return _digest(
            {
                "prompt": _wire_prompt(exchange.prompt),
                "action": _wire_action(exchange.action),
                "feedback": _wire_feedback(exchange.feedback),
            },
            domain="fertig.social.exchange-evidence.v1",
        )

    @staticmethod
    def _aligned_bindings(
        utterance: StructuredUtterance,
        target: RelationalReferent,
    ) -> tuple[tuple[int, int, PrimitiveKey], ...] | None:
        if len(utterance.arguments) != len(target.arguments):
            return None
        result: list[tuple[int, int, PrimitiveKey]] = [
            (RELATION_NAMESPACE, utterance.relation_token, (target.relation_code,))
        ]
        for phrase, argument in zip(utterance.arguments, target.arguments, strict=True):
            if len(phrase.referent_tokens) != len(argument.referent.samples):
                return None
            result.append((ROLE_NAMESPACE, phrase.role_token, (argument.role_code,)))
            result.extend(
                (
                    FEATURE_NAMESPACE,
                    token,
                    (sample.channel_code, sample.value_code),
                )
                for token, sample in zip(
                    phrase.referent_tokens,
                    argument.referent.samples,
                    strict=True,
                )
            )
        return tuple(result)

    def observe(self, exchange: Exchange) -> "ConventionLearner":
        """Learn only from a complete, validated post-action exchange."""

        if not isinstance(exchange, Exchange):
            raise TypeError("observe expects a completed Exchange")
        evidence_id = self._exchange_id(exchange)
        if evidence_id in self._seen:
            return self
        self._seen.add(evidence_id)
        speaker = exchange.prompt.speaker_id
        self._completed[speaker] += 1
        target_candidate = exchange.prompt.context.candidate(exchange.feedback.target_option_id)
        assert target_candidate is not None
        bindings = self._aligned_bindings(exchange.prompt.utterance, target_candidate.referent)
        if bindings is None:
            return self
        self._usable[speaker] += 1
        for namespace, token, primitive in bindings:
            self._counts[(speaker, namespace, token)][primitive] += 1
        return self

    def fit(self, transcript: Transcript) -> "ConventionLearner":
        if not isinstance(transcript, Transcript):
            raise TypeError("fit expects an immutable Transcript")
        self._reset()
        for exchange in transcript.exchanges:
            self.observe(exchange)
        return self

    @classmethod
    def from_transcript(
        cls,
        agent_id: int,
        transcript: Transcript,
        **kwargs: object,
    ) -> "ConventionLearner":
        """Induce a newcomer solely by replaying the public transcript."""

        return cls(agent_id, **kwargs).fit(transcript)

    def binding(
        self,
        speaker_id: int,
        namespace: int,
        token: int,
    ) -> BindingDecision:
        speaker_id = _integer(speaker_id, "speaker_id")
        namespace = _integer(namespace, "namespace")
        token = _integer(token, "token")
        counter = self._counts.get((speaker_id, namespace, token), Counter())
        observations = sum(counter.values())
        ordered = tuple(
            primitive
            for primitive, _ in sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        )
        if observations < self.minimum_evidence:
            return BindingDecision(
                namespace,
                token,
                DecisionStatus.UNKNOWN,
                None,
                ordered,
                observations,
                0.0 if observations == 0 else counter[ordered[0]] / observations,
            )
        top = counter[ordered[0]]
        second = counter[ordered[1]] if len(ordered) > 1 else 0
        confidence = top / observations
        if top - second < self.minimum_margin or confidence < self.dominance:
            return BindingDecision(
                namespace,
                token,
                DecisionStatus.AMBIGUOUS,
                None,
                ordered,
                observations,
                confidence,
            )
        return BindingDecision(
            namespace,
            token,
            DecisionStatus.RESOLVED,
            ordered[0],
            ordered,
            observations,
            confidence,
        )

    def speaker_assessment(self, speaker_id: int) -> SpeakerAssessment:
        speaker_id = _integer(speaker_id, "speaker_id")
        counters = [
            counter
            for (candidate_speaker, _, _), counter in self._counts.items()
            if candidate_speaker == speaker_id
        ]
        evidence = sum(sum(counter.values()) for counter in counters)
        consistent = sum(max(counter.values()) for counter in counters if counter)
        conflicts = sum(1 for counter in counters if len(counter) > 1)
        shape_rate = (
            self._usable[speaker_id] / self._completed[speaker_id]
            if self._completed[speaker_id]
            else 0.0
        )
        token_consistency = consistent / evidence if evidence else 0.0
        return SpeakerAssessment(
            speaker_id,
            self._completed[speaker_id],
            self._usable[speaker_id],
            shape_rate * token_consistency,
            conflicts,
        )

    def _decode(self, prompt: NamingPrompt) -> tuple[
        RelationalReferent | None,
        tuple[BindingDecision, ...],
        DecisionStatus,
    ]:
        decisions: list[BindingDecision] = [
            self.binding(
                prompt.speaker_id,
                RELATION_NAMESPACE,
                prompt.utterance.relation_token,
            )
        ]
        for argument in prompt.utterance.arguments:
            decisions.append(
                self.binding(prompt.speaker_id, ROLE_NAMESPACE, argument.role_token)
            )
            decisions.extend(
                self.binding(prompt.speaker_id, FEATURE_NAMESPACE, token)
                for token in argument.referent_tokens
            )
        decision_tuple = tuple(decisions)
        if any(decision.status is DecisionStatus.AMBIGUOUS for decision in decision_tuple):
            return None, decision_tuple, DecisionStatus.AMBIGUOUS
        if any(not decision.resolved for decision in decision_tuple):
            return None, decision_tuple, DecisionStatus.UNKNOWN

        cursor = 1
        relation = decisions[0].primitive
        assert relation is not None and len(relation) == 1
        arguments: list[RoleArgument] = []
        for phrase in prompt.utterance.arguments:
            role = decisions[cursor].primitive
            cursor += 1
            assert role is not None and len(role) == 1
            samples: list[OpaqueSample] = []
            for _ in phrase.referent_tokens:
                feature = decisions[cursor].primitive
                cursor += 1
                assert feature is not None and len(feature) == 2
                samples.append(OpaqueSample(feature[0], feature[1]))
            try:
                arguments.append(RoleArgument(role[0], OperationalReferent(tuple(samples))))
            except ValueError:
                return None, decision_tuple, DecisionStatus.AMBIGUOUS
        try:
            referent = RelationalReferent(relation[0], tuple(arguments))
        except ValueError:
            return None, decision_tuple, DecisionStatus.AMBIGUOUS
        return referent, decision_tuple, DecisionStatus.RESOLVED

    def choose(self, prompt: NamingPrompt) -> ReferentialDecision:
        """Select exactly one grounded option or return an explicit abstention."""

        if not isinstance(prompt, NamingPrompt):
            raise TypeError("choose expects a NamingPrompt")
        assessment = self.speaker_assessment(prompt.speaker_id)
        referent, bindings, status = self._decode(prompt)
        evidence_count = min((binding.observations for binding in bindings), default=0)
        binding_confidence = min((binding.confidence for binding in bindings), default=0.0)
        confidence = min(binding_confidence, assessment.consistency)
        if assessment.completed_rounds and (
            assessment.consistency < self.minimum_speaker_consistency
        ):
            return ReferentialDecision(
                DecisionStatus.UNRELIABLE,
                None,
                (),
                confidence,
                evidence_count,
                "speaker evidence is below the consistency floor",
            )
        if referent is None:
            reason = (
                "multiple primitive bindings remain viable"
                if status is DecisionStatus.AMBIGUOUS
                else "at least one primitive lacks sufficient post-action evidence"
            )
            return ReferentialDecision(status, None, (), confidence, evidence_count, reason)
        candidates = tuple(
            candidate.option_id
            for candidate in prompt.context.candidates
            if candidate.referent == referent
        )
        if not candidates:
            return ReferentialDecision(
                DecisionStatus.UNKNOWN,
                None,
                (),
                confidence,
                evidence_count,
                "decoded frame is absent from the current operational context",
            )
        if len(candidates) > 1:
            return ReferentialDecision(
                DecisionStatus.AMBIGUOUS,
                None,
                candidates,
                confidence,
                evidence_count,
                "multiple context options instantiate the decoded frame",
            )
        return ReferentialDecision(
            DecisionStatus.RESOLVED,
            candidates[0],
            candidates,
            confidence,
            evidence_count,
            "unique composition of speaker-specific grounded atoms",
        )

    @property
    def observed_speakers(self) -> tuple[int, ...]:
        return tuple(sorted(self._completed))

    @property
    def evidence_counts(self) -> Mapping[
        tuple[int, int, int], Mapping[PrimitiveKey, int]
    ]:
        snapshot = {
            key: MappingProxyType(dict(sorted(counter.items())))
            for key, counter in sorted(self._counts.items())
        }
        return MappingProxyType(snapshot)


class ConventionPopulation:
    """Public-transcript induction for admission, broadcast, and turnover."""

    def __init__(self) -> None:
        self._learners: dict[int, ConventionLearner] = {}

    @property
    def agent_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._learners))

    def admit(
        self,
        agent_id: int,
        transcript: Transcript = Transcript(),
        **learner_options: object,
    ) -> ConventionLearner:
        agent_id = _integer(agent_id, "agent_id")
        if agent_id in self._learners:
            raise ValueError("agent is already present")
        learner = ConventionLearner.from_transcript(
            agent_id,
            transcript,
            **learner_options,
        )
        self._learners[agent_id] = learner
        return learner

    def retire(self, agent_id: int) -> ConventionLearner:
        agent_id = _integer(agent_id, "agent_id")
        try:
            return self._learners.pop(agent_id)
        except KeyError as error:
            raise KeyError("agent is not present") from error

    def learner(self, agent_id: int) -> ConventionLearner:
        agent_id = _integer(agent_id, "agent_id")
        try:
            return self._learners[agent_id]
        except KeyError as error:
            raise KeyError("agent is not present") from error

    def broadcast(self, exchange: Exchange) -> None:
        for learner in self._learners.values():
            learner.observe(exchange)


@dataclass(frozen=True, slots=True)
class CodePermutation:
    """A finite closed bijection; unlisted opaque codes remain unchanged."""

    pairs: tuple[tuple[int, int], ...] = ()

    def __post_init__(self) -> None:
        pairs = tuple(
            (_integer(source, "permutation source"), _integer(target, "permutation target"))
            for source, target in self.pairs
        )
        sources = tuple(source for source, _ in pairs)
        targets = tuple(target for _, target in pairs)
        if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError("permutation sources and targets must each be unique")
        if set(sources) != set(targets):
            raise ValueError("a finite permutation must map a code set onto itself")
        object.__setattr__(self, "pairs", tuple(sorted(pairs)))

    @classmethod
    def cycle(cls, values: Sequence[int]) -> "CodePermutation":
        codes = _codes(values, "permutation code")
        if len(codes) < 2 or len(set(codes)) != len(codes):
            raise ValueError("a cycle requires at least two unique codes")
        return cls(tuple(zip(codes, codes[1:] + codes[:1], strict=True)))

    def apply(self, code: int) -> int:
        code = _integer(code, "code")
        return dict(self.pairs).get(code, code)

    def inverse(self) -> "CodePermutation":
        return CodePermutation(tuple((target, source) for source, target in self.pairs))


@dataclass(frozen=True, slots=True)
class SocialRenaming:
    """Independent permutations for every learner-visible integer namespace."""

    agents: CodePermutation = CodePermutation()
    options: CodePermutation = CodePermutation()
    tokens: CodePermutation = CodePermutation()
    relations: CodePermutation = CodePermutation()
    roles: CodePermutation = CodePermutation()
    channels: CodePermutation = CodePermutation()
    values: CodePermutation = CodePermutation()
    events: CodePermutation = CodePermutation()
    actions: CodePermutation = CodePermutation()


def _rename_operational(
    referent: OperationalReferent,
    renaming: SocialRenaming,
) -> OperationalReferent:
    return OperationalReferent(
        tuple(
            OpaqueSample(
                renaming.channels.apply(sample.channel_code),
                renaming.values.apply(sample.value_code),
            )
            for sample in referent.samples
        )
    )


def rename_relational(
    referent: RelationalReferent,
    renaming: SocialRenaming,
) -> RelationalReferent:
    return RelationalReferent(
        renaming.relations.apply(referent.relation_code),
        tuple(
            RoleArgument(
                renaming.roles.apply(argument.role_code),
                _rename_operational(argument.referent, renaming),
            )
            for argument in referent.arguments
        ),
    )


def _rename_event(event: PublicTraceEvent, renaming: SocialRenaming) -> PublicTraceEvent:
    return PublicTraceEvent(
        event.tick,
        renaming.agents.apply(event.agent_id),
        renaming.events.apply(event.event_code),
        renaming.roles.apply(event.role_code),
        _rename_operational(event.referent, renaming),
        None if event.action_code is None else renaming.actions.apply(event.action_code),
        (
            None
            if event.outcome_code is None
            else renaming.relations.apply(event.outcome_code)
        ),
    )


def rename_context(
    context: ReferentialContext,
    renaming: SocialRenaming,
) -> ReferentialContext:
    return ReferentialContext(
        tuple(
            ContextCandidate(
                renaming.options.apply(candidate.option_id),
                rename_relational(candidate.referent, renaming),
                tuple(_rename_event(event, renaming) for event in candidate.public_trace),
            )
            for candidate in context.candidates
        )
    )


def rename_utterance(
    utterance: StructuredUtterance,
    renaming: SocialRenaming,
) -> StructuredUtterance:
    return StructuredUtterance(
        renaming.tokens.apply(utterance.relation_token),
        tuple(
            ArgumentUtterance(
                renaming.tokens.apply(argument.role_token),
                tuple(renaming.tokens.apply(token) for token in argument.referent_tokens),
            )
            for argument in utterance.arguments
        ),
    )


def rename_exchange(exchange: Exchange, renaming: SocialRenaming) -> Exchange:
    """Rename a complete exchange and recompute its target commitment."""

    context = rename_context(exchange.prompt.context, renaming)
    utterance = rename_utterance(exchange.prompt.utterance, renaming)
    target = renaming.options.apply(exchange.feedback.target_option_id)
    speaker = renaming.agents.apply(exchange.prompt.speaker_id)
    listener = renaming.agents.apply(exchange.prompt.listener_id)
    prompt = NamingPrompt(
        exchange.prompt.round_id,
        speaker,
        listener,
        context,
        utterance,
        make_target_commitment(
            round_id=exchange.prompt.round_id,
            speaker_id=speaker,
            listener_id=listener,
            context=context,
            utterance=utterance,
            target_option_id=target,
            nonce=exchange.feedback.nonce,
        ),
    )
    action = ListenerAction(
        exchange.action.round_id,
        listener,
        (
            None
            if exchange.action.option_id is None
            else renaming.options.apply(exchange.action.option_id)
        ),
    )
    trace = JointAttentionTrace(
        tuple(renaming.agents.apply(agent) for agent in exchange.feedback.joint_attention.participants),
        tuple(_rename_event(event, renaming) for event in exchange.feedback.joint_attention.events),
    )
    feedback = ReferentialFeedback(
        exchange.feedback.round_id,
        target,
        exchange.feedback.success,
        exchange.feedback.nonce,
        trace,
    )
    return Exchange(prompt, action, feedback)


def rename_transcript(transcript: Transcript, renaming: SocialRenaming) -> Transcript:
    result = Transcript()
    for exchange in transcript.exchanges:
        result = result.append(rename_exchange(exchange, renaming))
    return result


@dataclass(frozen=True, slots=True)
class EquivarianceResult:
    original: ReferentialDecision
    renamed: ReferentialDecision
    expected_renamed_option: int | None
    passed: bool


def check_equivariance(
    original: ReferentialDecision,
    renamed: ReferentialDecision,
    renaming: SocialRenaming,
) -> EquivarianceResult:
    expected = (
        None if original.option_id is None else renaming.options.apply(original.option_id)
    )
    passed = original.status is renamed.status and renamed.option_id == expected
    if original.abstained:
        passed = passed and renamed.abstained
    return EquivarianceResult(original, renamed, expected, passed)


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """Decision-only swap that preserves surface form and option positions."""

    factual_prompt: NamingPrompt
    factual_target: int
    contrast_prompt: NamingPrompt
    contrast_target: int


def swap_candidate_referents(
    prompt: NamingPrompt,
    factual_target: int,
    distractor: int,
) -> CounterfactualPair:
    """Swap two candidates' operational content while preserving all surfaces.

    The contrast commitment is intentionally a domain-separated placeholder:
    it has no hidden target/nonce and therefore cannot form an ``Exchange``.
    Counterfactual prompts are for decision controls only.
    """

    factual_target = _integer(factual_target, "factual_target")
    distractor = _integer(distractor, "distractor")
    if factual_target == distractor:
        raise ValueError("counterfactual swap requires two different options")
    first = prompt.context.candidate(factual_target)
    second = prompt.context.candidate(distractor)
    if first is None or second is None:
        raise ValueError("both swapped options must occur in the context")
    candidates = tuple(
        ContextCandidate(candidate.option_id, second.referent, second.public_trace)
        if candidate.option_id == factual_target
        else ContextCandidate(candidate.option_id, first.referent, first.public_trace)
        if candidate.option_id == distractor
        else candidate
        for candidate in prompt.context.candidates
    )
    context = ReferentialContext(candidates)
    placeholder = _digest(
        {
            "round": prompt.round_id,
            "context": _wire_context(context),
            "utterance": _wire_utterance(prompt.utterance),
        },
        domain="fertig.social.uncommitted-counterfactual.v1",
    )
    contrast = NamingPrompt(
        prompt.round_id,
        prompt.speaker_id,
        prompt.listener_id,
        context,
        prompt.utterance,
        placeholder,
    )
    return CounterfactualPair(prompt, factual_target, contrast, distractor)


@dataclass(frozen=True, slots=True)
class CounterfactualResult:
    factual: ReferentialDecision
    contrast: ReferentialDecision
    passed: bool


def evaluate_counterfactual(
    predictor: Callable[[NamingPrompt], ReferentialDecision],
    pair: CounterfactualPair,
) -> CounterfactualResult:
    factual = predictor(pair.factual_prompt)
    contrast = predictor(pair.contrast_prompt)
    return CounterfactualResult(
        factual,
        contrast,
        factual.option_id == pair.factual_target
        and contrast.option_id == pair.contrast_target,
    )


@dataclass(frozen=True, slots=True)
class ControlScore:
    answered: int
    correct: int
    total: int

    @property
    def coverage(self) -> float:
        return self.answered / self.total if self.total else 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.answered if self.answered else 0.0


def score_predictor(
    predictor: Callable[[NamingPrompt], ReferentialDecision],
    cases: Iterable[tuple[NamingPrompt, int]],
) -> ControlScore:
    values = tuple(cases)
    answered = 0
    correct = 0
    for prompt, target in values:
        target = _integer(target, "target")
        decision = predictor(prompt)
        answered += int(decision.option_id is not None)
        correct += int(decision.option_id == target)
    return ControlScore(answered, correct, len(values))


class PositionShortcut:
    """Explicit negative control using only target ordinal and context size."""

    def __init__(self) -> None:
        self._counts: defaultdict[int, Counter[int]] = defaultdict(Counter)

    def fit(self, transcript: Transcript) -> "PositionShortcut":
        self._counts.clear()
        for exchange in transcript.exchanges:
            ordinal = exchange.prompt.context.ordinal(exchange.feedback.target_option_id)
            assert ordinal is not None
            self._counts[len(exchange.prompt.context.candidates)][ordinal] += 1
        return self

    def choose(self, prompt: NamingPrompt) -> ReferentialDecision:
        counts = self._counts[len(prompt.context.candidates)]
        if not counts:
            return ReferentialDecision(
                DecisionStatus.UNKNOWN,
                None,
                (),
                0.0,
                0,
                "position shortcut has no matching context size",
            )
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            return ReferentialDecision(
                DecisionStatus.AMBIGUOUS,
                None,
                (),
                0.0,
                sum(counts.values()),
                "position shortcut has a tied ordinal",
            )
        ordinal, count = ordered[0]
        option_id = prompt.context.candidates[ordinal].option_id
        return ReferentialDecision(
            DecisionStatus.RESOLVED,
            option_id,
            (option_id,),
            count / sum(counts.values()),
            sum(counts.values()),
            "position-only negative control",
        )


class SurfaceLookupShortcut:
    """Explicit negative control memorizing full utterance-to-target ordinal."""

    def __init__(self) -> None:
        self._counts: defaultdict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def fit(self, transcript: Transcript) -> "SurfaceLookupShortcut":
        self._counts.clear()
        for exchange in transcript.exchanges:
            ordinal = exchange.prompt.context.ordinal(exchange.feedback.target_option_id)
            assert ordinal is not None
            self._counts[exchange.prompt.utterance.surface][ordinal] += 1
        return self

    def choose(self, prompt: NamingPrompt) -> ReferentialDecision:
        counts = self._counts[prompt.utterance.surface]
        if not counts:
            return ReferentialDecision(
                DecisionStatus.UNKNOWN,
                None,
                (),
                0.0,
                0,
                "full utterance was not observed",
            )
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
            return ReferentialDecision(
                DecisionStatus.AMBIGUOUS,
                None,
                (),
                0.0,
                sum(counts.values()),
                "surface lookup has tied target ordinals",
            )
        ordinal, count = ordered[0]
        if ordinal >= len(prompt.context.candidates):
            return ReferentialDecision(
                DecisionStatus.UNKNOWN,
                None,
                (),
                0.0,
                sum(counts.values()),
                "memorized ordinal is invalid in this context",
            )
        option_id = prompt.context.candidates[ordinal].option_id
        return ReferentialDecision(
            DecisionStatus.RESOLVED,
            option_id,
            (option_id,),
            count / sum(counts.values()),
            sum(counts.values()),
            "full-utterance lookup negative control",
        )


@runtime_checkable
class SocialAgentEnvironment(Protocol):
    """Learner-facing staged protocol; no oracle or private vocabulary."""

    @property
    def transcript(self) -> Transcript: ...

    @property
    def complete(self) -> bool: ...

    def next_prompt(self) -> NamingPrompt: ...

    def submit(self, option_id: int | None) -> ReferentialFeedback: ...


_FORBIDDEN_AGENT_NAMES = frozenset(
    {
        "codebook",
        "codebooks",
        "nonce",
        "oracle",
        "plan",
        "plans",
        "private_vocabulary",
        "private_vocabularies",
        "seed",
        "target",
        "target_option_id",
        "vocabulary",
        "vocabularies",
    }
)


def audit_social_boundary(environment: SocialAgentEnvironment) -> tuple[str, ...]:
    """Find forbidden *public* capabilities on a learner-facing game object."""

    public = {name for name in dir(environment) if not name.startswith("_")}
    return tuple(sorted(public & _FORBIDDEN_AGENT_NAMES))


__all__ = [
    "ArgumentUtterance",
    "BindingDecision",
    "CodePermutation",
    "ContextCandidate",
    "ControlScore",
    "ConventionLearner",
    "ConventionPopulation",
    "CounterfactualPair",
    "CounterfactualResult",
    "DecisionStatus",
    "EquivarianceResult",
    "Exchange",
    "FEATURE_NAMESPACE",
    "HashChainLedger",
    "HashChainRecord",
    "JointAttentionTrace",
    "ListenerAction",
    "NamingPrompt",
    "OpaqueCode",
    "OpaqueSample",
    "OperationalReferent",
    "PositionShortcut",
    "PrimitiveKey",
    "PublicTraceEvent",
    "RELATION_NAMESPACE",
    "ROLE_NAMESPACE",
    "ReferentialContext",
    "ReferentialDecision",
    "ReferentialFeedback",
    "RelationalReferent",
    "RoleArgument",
    "SocialAgentEnvironment",
    "SocialRenaming",
    "SpeakerAssessment",
    "StructuredUtterance",
    "SurfaceLookupShortcut",
    "Transcript",
    "audit_social_boundary",
    "check_equivariance",
    "evaluate_counterfactual",
    "joint_attention_for",
    "make_target_commitment",
    "rename_context",
    "rename_exchange",
    "rename_relational",
    "rename_transcript",
    "rename_utterance",
    "score_predictor",
    "swap_candidate_referents",
]
