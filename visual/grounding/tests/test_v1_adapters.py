from __future__ import annotations

from grounding_kernel.processworld import ProcessHarness
from grounding_kernel.v1_adapters import (
    BinderSupportRecord,
    episode_to_query_trace,
    episode_to_public_trace,
    fresh_opaque_token,
    ostensive_record_to_support,
    support_episode_to_binder_record,
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
    assert not support_positive.trace.has_feedback

    query_trace = episode_to_query_trace(positive.episode)
    assert not query_trace.has_feedback


def test_support_adapter_separates_generic_training_correction_from_query_trace() -> None:
    record = ProcessHarness(65).oracle.process_pair()[0]
    support = ostensive_record_to_support(record, turn_id=3, remaining_cost=2.0)
    normalized = support_episode_to_binder_record(support)

    assert isinstance(normalized, BinderSupportRecord)
    assert normalized.token == record.token
    assert normalized.support_feedback == 1.0
    assert normalized.episode == episode_to_query_trace(record.episode)
    assert not normalized.episode.has_feedback
    assert set(BinderSupportRecord.__dataclass_fields__) == {
        "token",
        "episode",
        "support_feedback",
    }


def test_fresh_token_stays_in_public_code_domain_without_seed_or_codebook() -> None:
    manifest = ProcessHarness(910).agent.manifest
    fresh = fresh_opaque_token(manifest.concept_codes, nonce=7)
    replay = fresh_opaque_token(reversed(manifest.concept_codes), nonce=7)

    assert fresh == replay
    assert fresh not in manifest.concept_codes
    assert len(str(fresh)) == len(str(manifest.concept_codes[0])) == 9
