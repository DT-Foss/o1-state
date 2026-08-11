"""Evaluator-side adapters from ProcessWorld episodes to the v1 public SDK.

The central conversion deliberately drops ``Transition.outcome_code``.  Core
v1 trials therefore expose only raw before/action/after records and generic
scalar feedback, even though ProcessWorld retains opaque outcome codes for v0
compatibility and diagnostic ablations.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    trace = episode_to_public_trace(record.episode)
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


__all__ = [
    "SupportEpisode",
    "episode_to_public_trace",
    "ostensive_record_to_support",
]
