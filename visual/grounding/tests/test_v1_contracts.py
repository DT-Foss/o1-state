from __future__ import annotations

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    PROTOCOL_VERSION,
    SENSOR_SCHEMA_RGB_U8,
    ActionDecision,
    BeliefDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
    Utterance,
)


def _observation(tick: int, value: int = 0) -> Observation:
    return Observation(np.full((5, 6, 3), value, dtype=np.uint8), tick)


def test_public_trace_has_raw_consequences_but_no_semantic_outcome() -> None:
    before = _observation(0)
    after = _observation(1, 20)
    transition = PublicTransition(before, Action(919, (2, 3), (1, 0)), after, 0.25)
    trace = PublicTrace(before, (transition,))

    assert trace.current == after
    assert trace.total_feedback == pytest.approx(0.25)
    assert transition.pixels_changed
    assert not hasattr(transition, "outcome_code")
    assert not hasattr(transition, "object_id")


def test_turn_cues_are_support_only_and_budgets_are_authoritative() -> None:
    observation = _observation(0)
    turn = PublicTurn(
        7,
        SessionPhase.SUPPORT,
        observation,
        Utterance((101, 202)),
        (1, 1, 4, 4),
        1.0,
        3.5,
    )
    assert turn.remaining_cost == 3.5

    with pytest.raises(ValueError, match="support"):
        PublicTurn(
            8,
            SessionPhase.QUERY,
            observation,
            ostensive_pixel_cue=(1, 1, 4, 4),
        )


def test_manifest_and_utterances_fail_closed_on_invalid_limits() -> None:
    manifest = SessionManifest(
        PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        12,
        9,
        20,
    )
    assert manifest.intervention_cost_budget == 9.0

    with pytest.raises(ValueError, match="protocol_version"):
        SessionManifest(
            "future", SENSOR_SCHEMA_RGB_U8, ACTION_SCHEMA_OPAQUE_MOTOR, 1, 1, 1
        )
    with pytest.raises(ValueError, match="sensor_schema"):
        SessionManifest(
            PROTOCOL_VERSION,
            "code 9 means shelter",
            ACTION_SCHEMA_OPAQUE_MOTOR,
            1,
            1,
            1,
        )
    with pytest.raises(ValueError, match="empty"):
        Utterance(())
    with pytest.raises(TypeError, match="integer"):
        Utterance((True,))


def test_beliefs_include_normalized_unknown_mass_and_unique_candidates() -> None:
    belief = BeliefDecision(((20, 0.2), (10, 0.3)), 0.5)
    assert belief.candidate_probabilities == ((10, 0.3), (20, 0.2))
    assert dict(belief.distribution) == {10: 0.3, 20: 0.2}

    with pytest.raises(ValueError, match="sum"):
        BeliefDecision(((10, 0.6),), 0.5)
    with pytest.raises(ValueError, match="unique"):
        BeliefDecision(((10, 0.2), (10, 0.3)), 0.5)
    with pytest.raises(ValueError, match="at most"):
        ActionDecision(None, 1.1)


def test_trace_continuity_and_feedback_bounds_are_enforced() -> None:
    before = _observation(0)
    after = _observation(1)
    transition = PublicTransition(before, Action(5, (0, 0)), after)
    with pytest.raises(ValueError, match="discontinuous"):
        PublicTrace(_observation(0, 1), (transition,))
    with pytest.raises(ValueError, match="at most"):
        PublicTransition(before, Action(5, (0, 0)), after, 1.01)


def test_feedback_is_stripped_or_forbidden_for_sealed_queries() -> None:
    before = _observation(0)
    after = _observation(1)
    trace = PublicTrace(
        before,
        (PublicTransition(before, Action(5, (0, 0)), after, 0.75),),
    )
    assert trace.has_feedback
    assert not trace.feedback_stripped().has_feedback

    with pytest.raises(ValueError, match="forbidden"):
        PublicTurn(
            1,
            SessionPhase.FROZEN_QUERY,
            before,
            scalar_feedback=1.0,
        )


def test_negative_observation_ticks_are_not_valid_v1_records() -> None:
    negative = Observation(np.zeros((5, 6, 3), dtype=np.uint8), -1)
    with pytest.raises(ValueError, match="tick"):
        PublicTurn(0, SessionPhase.SUPPORT, negative)
    with pytest.raises(ValueError, match="tick"):
        PublicTrace(negative)
