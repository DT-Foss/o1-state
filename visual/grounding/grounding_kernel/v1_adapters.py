"""Evaluator-side adapters from ProcessWorld episodes to the v1 public SDK.

The central conversion deliberately drops ``Transition.outcome_code``.  Core
v1 trials therefore expose only raw before/action/after records and generic
scalar feedback, even though ProcessWorld retains opaque outcome codes for v0
compatibility and diagnostic ablations.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite
from typing import Iterable

from .processworld import OstensiveRecord, PublicEpisode
from .v1_contracts import (
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionPhase,
    Utterance,
)


@dataclass(frozen=True, slots=True)
class SupportEpisode:
    """One support turn plus its outcome-code-free public trace."""

    turn: PublicTurn
    trace: PublicTrace

    def __post_init__(self) -> None:
        if not isinstance(self.turn, PublicTurn) or not isinstance(self.trace, PublicTrace):
            raise TypeError("support episodes require a PublicTurn and PublicTrace")
        if self.turn.phase is not SessionPhase.SUPPORT:
            raise ValueError("support episode turns must use the support phase")
        if self.turn.observation != self.trace.initial:
            raise ValueError("support turn and trace must share the initial observation")


@dataclass(frozen=True, slots=True)
class BinderSupportRecord:
    """Learner-visible token supervision over an outcome- and feedback-free trace.

    ``support_feedback`` is a generic signed correction, not evaluator truth.
    Query calls receive only ``episode`` and ``token`` and therefore cannot
    inspect this training-only value.
    """

    token: int
    episode: PublicTrace
    support_feedback: float

    def __post_init__(self) -> None:
        if isinstance(self.token, bool) or not isinstance(self.token, int):
            raise TypeError("token must be an integer")
        if not isinstance(self.episode, PublicTrace):
            raise TypeError("episode must be a PublicTrace")
        if self.episode.has_feedback:
            raise ValueError("binder support traces must carry no query-visible feedback")
        if isinstance(self.support_feedback, bool) or not isinstance(
            self.support_feedback, (int, float)
        ):
            raise TypeError("support_feedback must be a finite signed number")
        feedback = float(self.support_feedback)
        if not isfinite(feedback) or not -1.0 <= feedback <= 1.0:
            raise ValueError("support_feedback must be finite and lie in [-1, 1]")
        if feedback == 0.0:
            raise ValueError("support_feedback must be non-zero supervision")
        object.__setattr__(self, "support_feedback", feedback)


def episode_to_public_trace(
    episode: PublicEpisode,
    *,
    strip_feedback: bool = False,
) -> PublicTrace:
    """Convert an episode while dropping every semantic/opaque outcome code."""

    if not isinstance(episode, PublicEpisode):
        raise TypeError("episode must be a ProcessWorld PublicEpisode")
    if not episode.transitions:
        raise ValueError("an empty ProcessWorld episode has no public initial observation")
    transitions = tuple(
        PublicTransition(
            transition.before,
            transition.action,
            transition.after,
            None if strip_feedback else feedback,
        )
        for transition, feedback in zip(
            episode.transitions,
            episode.scalar_feedback,
            strict=True,
        )
    )
    return PublicTrace(transitions[0].before, transitions)


def episode_to_query_trace(episode: PublicEpisode) -> PublicTrace:
    """Create a sealed outcome- and feedback-free binder query."""

    return episode_to_public_trace(episode, strip_feedback=True)


def ostensive_record_to_support(
    record: OstensiveRecord,
    *,
    turn_id: int,
    remaining_cost: float,
    ostensive_pixel_cue: tuple[int, int, int, int] | None = None,
) -> SupportEpisode:
    """Create a support input using corrective feedback only during support."""

    if not isinstance(record, OstensiveRecord):
        raise TypeError("record must be an OstensiveRecord")
    # Corrective supervision belongs only to the support turn.  The causal
    # trace itself has the exact same public shape used later at query time.
    trace = episode_to_query_trace(record.episode)
    turn = PublicTurn(
        turn_id=turn_id,
        phase=SessionPhase.SUPPORT,
        observation=trace.initial,
        utterance=Utterance((record.token,)),
        ostensive_pixel_cue=ostensive_pixel_cue,
        scalar_feedback=1.0 if record.task_feedback else -1.0,
        remaining_cost=remaining_cost,
    )
    return SupportEpisode(turn, trace)


def support_episode_to_binder_record(support: SupportEpisode) -> BinderSupportRecord:
    """Normalize a v1 support exchange for :class:`EpisodeConceptBinder`."""

    if not isinstance(support, SupportEpisode):
        raise TypeError("support must be a SupportEpisode")
    utterance = support.turn.utterance
    if utterance is None or len(utterance.tokens) != 1:
        raise ValueError("binder support requires exactly one opaque token")
    feedback = support.turn.scalar_feedback
    if feedback is None or feedback == 0.0:
        raise ValueError("binder support requires non-zero generic feedback")
    return BinderSupportRecord(
        token=utterance.tokens[0],
        episode=support.trace.feedback_stripped(),
        support_feedback=feedback,
    )


def fresh_opaque_token(existing_tokens: Iterable[int], *, nonce: int = 0) -> int:
    """Derive an unused token in the same decimal code domain.

    This evaluator utility hashes only already-public opaque values plus a
    nonce.  It neither receives nor reconstructs a seed or semantic codebook.
    """

    values: list[int] = []
    for value in existing_tokens:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("existing_tokens must contain integers")
        if value <= 0:
            raise ValueError("existing_tokens must be positive")
        values.append(value)
    if not values:
        raise ValueError("at least one existing token is required")
    if isinstance(nonce, bool) or not isinstance(nonce, int):
        raise TypeError("nonce must be an integer")
    if nonce < 0:
        raise ValueError("nonce must be non-negative")
    widths = {len(str(value)) for value in values}
    if len(widths) != 1:
        raise ValueError("existing tokens must share one decimal code domain")
    width = widths.pop()
    lower = 1 if width == 1 else 10 ** (width - 1)
    upper = 10**width
    occupied = set(values)
    payload = repr((tuple(sorted(occupied)), nonce)).encode("ascii")
    for counter in range(len(occupied) + 1):
        digest = sha256(payload + counter.to_bytes(8, "big")).digest()
        candidate = lower + int.from_bytes(digest[:8], "big") % (upper - lower)
        if candidate not in occupied:
            return candidate
    raise RuntimeError("opaque token domain is exhausted")


__all__ = [
    "BinderSupportRecord",
    "SupportEpisode",
    "episode_to_public_trace",
    "episode_to_query_trace",
    "fresh_opaque_token",
    "ostensive_record_to_support",
    "support_episode_to_binder_record",
]
