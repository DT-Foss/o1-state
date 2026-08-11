from __future__ import annotations

from hashlib import sha256

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    PROTOCOL_VERSION,
    SENSOR_SCHEMA_RGB_U8,
    ActionDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
    Utterance,
)
from grounding_kernel.v1_session import (
    EvaluationSession,
    SessionBudgetError,
    SessionStateError,
)


def _digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def _observation(tick: int, value: int = 0) -> Observation:
    return Observation(np.full((4, 4, 3), value, dtype=np.uint8), tick)


def _transition() -> PublicTransition:
    return PublicTransition(
        _observation(0),
        Action(505, (1, 1), (1, 0)),
        _observation(1, 12),
        0.25,
    )


def _manifest() -> SessionManifest:
    return SessionManifest(
        PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        1,
        2.0,
        1,
    )


def test_full_commit_support_acquire_freeze_query_sequence_is_auditable() -> None:
    session = EvaluationSession(_manifest(), _digest("model"))
    session.commit_codebook(_digest("secret-codebook"))
    transition = _transition()
    trace = PublicTrace(transition.before, (transition,))
    turn = PublicTurn(
        0,
        SessionPhase.SUPPORT,
        transition.before,
        Utterance((9001,)),
        (0, 0, 2, 2),
        remaining_cost=2.0,
    )
    session.record_support(turn, trace)
    session.begin_acquisition()
    session.record_experiment(transition, 1.25)
    session.freeze(_digest("checkpoint"))
    session.record_query(Utterance((9001,)), ActionDecision(None, 1.0))
    ledger = session.complete()

    assert ledger.phase is SessionPhase.COMPLETE
    assert ledger.support_episodes_used == 1
    assert ledger.intervention_cost_used == pytest.approx(1.25)
    assert ledger.queries_used == 1
    assert len(ledger.events) == 7
    assert len(ledger.ledger_hash) == 64
    assert session.commitments["model"] == _digest("model")
    assert session.commitments["codebook"] == _digest("secret-codebook")


def test_codebook_must_follow_model_commit_and_precede_support() -> None:
    session = EvaluationSession(_manifest(), _digest("model"))
    transition = _transition()
    turn = PublicTurn(
        0,
        SessionPhase.SUPPORT,
        transition.before,
        remaining_cost=2.0,
    )
    with pytest.raises(SessionStateError, match="codebook"):
        session.record_support(turn, PublicTrace(transition.before, (transition,)))
    with pytest.raises(ValueError, match="distinct"):
        session.commit_codebook(_digest("model"))


def test_all_three_budgets_fail_closed() -> None:
    session = EvaluationSession(_manifest(), _digest("model"))
    session.commit_codebook(_digest("codebook"))
    transition = _transition()
    trace = PublicTrace(transition.before, (transition,))
    turn = PublicTurn(0, SessionPhase.SUPPORT, transition.before, remaining_cost=2.0)
    session.record_support(turn, trace)
    with pytest.raises(SessionBudgetError, match="support"):
        session.record_support(
            PublicTurn(1, SessionPhase.SUPPORT, transition.before, remaining_cost=2.0),
            trace,
        )
    session.begin_acquisition()
    with pytest.raises(SessionBudgetError, match="cost"):
        session.record_experiment(transition, 2.01)
    session.record_experiment(transition, 2.0)
    session.freeze(_digest("checkpoint"))
    session.record_query(Utterance((1,)), ActionDecision(None, 1.0))
    with pytest.raises(SessionBudgetError, match="query"):
        session.record_query(Utterance((1,)), ActionDecision(None, 1.0))


def test_phase_and_remaining_cost_forgery_are_rejected() -> None:
    session = EvaluationSession(_manifest(), _digest("model"))
    session.commit_codebook(_digest("codebook"))
    transition = _transition()
    forged = PublicTurn(
        0,
        SessionPhase.SUPPORT,
        transition.before,
        remaining_cost=999.0,
    )
    with pytest.raises(SessionStateError, match="authoritative"):
        session.record_support(forged, PublicTrace(transition.before))
    with pytest.raises(SessionStateError, match="phase"):
        session.freeze(_digest("checkpoint"))


def test_ledger_is_deterministic_and_wire_allowlist_applies_to_queries() -> None:
    def execute() -> str:
        session = EvaluationSession(_manifest(), _digest("model"))
        session.commit_codebook(_digest("codebook"))
        session.begin_acquisition()
        session.freeze(_digest("checkpoint"))
        session.record_query(Utterance((4,)), ActionDecision(None, 1.0))
        return session.complete().ledger_hash

    assert execute() == execute()

    session = EvaluationSession(_manifest(), _digest("model"))
    session.commit_codebook(_digest("codebook"))
    session.begin_acquisition()
    session.freeze(_digest("checkpoint"))
    with pytest.raises(TypeError, match="unsupported"):
        session.record_query(object(), ActionDecision(None, 1.0))


def test_query_router_rejects_feedback_leaks_and_nondecision_responses() -> None:
    session = EvaluationSession(_manifest(), _digest("model"))
    session.commit_codebook(_digest("codebook"))
    session.begin_acquisition()
    session.freeze(_digest("checkpoint"))
    transition = _transition()
    trace = PublicTrace(transition.before, (transition,))

    with pytest.raises(SessionStateError, match="feedback-stripped"):
        session.record_query(trace, ActionDecision(None, 1.0))
    with pytest.raises(SessionStateError, match="decision"):
        session.record_query(trace.feedback_stripped(), Utterance((9,)))

    session.record_query(trace.feedback_stripped(), ActionDecision(None, 1.0))
    assert session.ledger.queries_used == 1
