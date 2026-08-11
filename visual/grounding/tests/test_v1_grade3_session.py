from dataclasses import FrozenInstanceError, replace

import pytest

from grounding_kernel.v1_grade3_contracts import (
    GRADE3_PROTOCOL_VERSION,
    Grade3SessionManifest,
)
from grounding_kernel.v1_grade3_session import (
    Grade3EvaluationSession,
    Grade3Resource,
    Grade3SessionBudgetError,
    Grade3SessionPhase,
    Grade3SessionStateError,
)


def manifest() -> Grade3SessionManifest:
    return Grade3SessionManifest(
        GRADE3_PROTOCOL_VERSION,
        "rgb-u8-v1",
        "opaque-motor-target-vector-v1",
        2,
        3.5,
        2,
        4.5,
        3,
    )


def session() -> Grade3EvaluationSession:
    value = Grade3EvaluationSession(manifest(), "11" * 32, "22" * 32)
    value.commit_codebook("33" * 32)
    return value


def test_separate_reservations_are_precharged_and_failures_remain_consumed() -> None:
    value = session()
    support = value.reserve(Grade3Resource.SUPPORT_RECORD, 1, "support", "44" * 32)
    assert value.remaining_support_records == 1
    assert value.ledger.support_records_reserved == 1
    event = value.consume(support, status="failed_consumed", result_hash="55" * 32)
    assert event.status == "failed_consumed"
    assert value.remaining_support_records == 1
    assert value.ledger.support_records_used == 1
    assert value.ledger.support_records_reserved == 0

    value.begin_acquisition()
    acquisition = value.reserve(Grade3Resource.ACQUISITION_COST, 1.25, "probe", "66" * 32)
    assert value.remaining_acquisition_cost == pytest.approx(2.25)
    value.consume(acquisition, status="completed", result_hash="77" * 32)
    assert value.ledger.acquisition_cost_used == pytest.approx(1.25)

    value.freeze("88" * 32)
    query = value.reserve_query(9, "describe", "99" * 32)
    action = value.reserve(Grade3Resource.MOTOR_ACTION_COST, 2.5, "motor_action", "aa" * 32)
    reset = value.reserve(Grade3Resource.MOTOR_RESET, 1, "motor_reset", "bb" * 32)
    assert value.remaining_queries == 1
    assert value.remaining_motor_action_cost == pytest.approx(2.0)
    assert value.remaining_motor_resets == 2
    value.consume(action, status="failed_consumed", result_hash="cc" * 32)
    value.consume(reset, status="completed", result_hash="dd" * 32)
    value.consume(query, status="abstained", result_hash="ee" * 32)

    ledger = value.ledger
    assert ledger.sealed_queries_used == 1
    assert ledger.motor_action_cost_used == pytest.approx(2.5)
    assert ledger.motor_resets_used == 1
    assert ledger.sealed_query_ids == (9,)
    assert ledger.chain_valid
    assert len(ledger.ledger_hash) == 64


def test_phase_gates_active_reservations_and_query_ids_fail_closed() -> None:
    value = session()
    pending = value.reserve(Grade3Resource.SUPPORT_RECORD, 1, "support", "44" * 32)
    with pytest.raises(Grade3SessionStateError, match="reservations"):
        value.begin_acquisition()
    value.consume(pending, status="completed", result_hash=None)
    value.begin_acquisition()
    with pytest.raises(Grade3SessionStateError, match="phase"):
        value.reserve(Grade3Resource.SUPPORT_RECORD, 1, "late", "55" * 32)
    value.freeze("66" * 32)

    first = value.reserve_query(7, "belief", "77" * 32)
    value.consume(first, status="failed_consumed", result_hash="88" * 32)
    with pytest.raises(Grade3SessionStateError, match="reused"):
        value.reserve_query(7, "belief", "99" * 32)
    second = value.reserve_query(8, "belief", "aa" * 32)
    value.consume(second, status="completed", result_hash="bb" * 32)
    with pytest.raises(Grade3SessionBudgetError, match="budget"):
        value.reserve_query(10, "belief", "cc" * 32)


def test_ledger_is_immutable_and_detects_event_tampering() -> None:
    value = session()
    value.freeze("44" * 32)
    ledger = value.complete()
    assert ledger.phase is Grade3SessionPhase.COMPLETE
    assert ledger.chain_valid
    with pytest.raises(FrozenInstanceError):
        ledger.phase = Grade3SessionPhase.SUPPORT  # type: ignore[misc]

    events = list(ledger.events)
    events[-1] = replace(events[-1], status="failed")
    tampered = replace(ledger, events=tuple(events))
    assert not tampered.chain_valid
    assert tampered.ledger_hash != ledger.ledger_hash


def test_commitments_and_integral_resources_are_strict() -> None:
    value = Grade3EvaluationSession(manifest(), "11" * 32, "22" * 32)
    with pytest.raises(ValueError, match="independent"):
        value.commit_codebook("11" * 32)
    value.commit_codebook("33" * 32)
    with pytest.raises(Grade3SessionStateError, match="already"):
        value.commit_codebook("44" * 32)
    with pytest.raises(ValueError, match="integral"):
        value.reserve(Grade3Resource.SUPPORT_RECORD, 0.5, "support", "55" * 32)
    with pytest.raises(ValueError, match="positive"):
        value.reserve(Grade3Resource.SUPPORT_RECORD, 0, "support", "66" * 32)
