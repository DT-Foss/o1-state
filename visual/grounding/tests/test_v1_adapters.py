from __future__ import annotations

from grounding_kernel.processworld import ProcessHarness
from grounding_kernel.v1_adapters import (
    episode_to_public_trace,
    ostensive_record_to_support,
)
from grounding_kernel.v1_contracts import SessionPhase


def test_process_episode_conversion_drops_outcome_codes_but_preserves_raw_causality() -> None:
    record = ProcessHarness(44).oracle.process_pair()[0]
    trace = episode_to_public_trace(record.episode)

    assert len(trace.transitions) == len(record.episode.transitions)
    for original, converted in zip(
        record.episode.transitions,
        trace.transitions,
        strict=True,
    ):
        assert converted.before == original.before
        assert converted.action == original.action
        assert converted.after == original.after
        assert not hasattr(converted, "outcome_code")


def test_sealed_trace_strips_feedback_while_support_keeps_only_generic_correction() -> None:
    positive, negative = ProcessHarness(55).oracle.affordance_pair()
    support_positive = ostensive_record_to_support(
        positive,
        turn_id=0,
        remaining_cost=4.0,
    )
    support_negative = ostensive_record_to_support(
        negative,
        turn_id=1,
        remaining_cost=4.0,
    )

    assert support_positive.turn.phase is SessionPhase.SUPPORT
    assert support_positive.turn.scalar_feedback == 1.0
    assert support_negative.turn.scalar_feedback == -1.0
    assert support_positive.turn.utterance is not None
    assert support_positive.turn.utterance.tokens == (positive.token,)

    query_trace = episode_to_public_trace(positive.episode, strip_feedback=True)
    assert not query_trace.has_feedback
