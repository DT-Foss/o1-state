from __future__ import annotations

from dataclasses import fields

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    SENSOR_SCHEMA_RGB_U8,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionPhase,
    Utterance,
)
from grounding_kernel.v1_grade3_contracts import (
    CausalSupportRecord,
    GRADE3_PROTOCOL_VERSION,
    Grade3SessionManifest,
    MAX_GRADE3_ACTION_CODES,
    MAX_GRADE3_INTEGER,
    MAX_GRADE3_TRANSITIONS,
    MotorActionSpace,
    MotorDecision,
    MotorDirective,
    MotorPhase,
    MotorQuery,
    OstensiveSupportRecord,
    ProbeDecision,
    ProbeEvidence,
    ProbeOffer,
    ProbeOption,
    ProbeResult,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)


def _observation(tick: int, value: int = 0, *, width: int = 5) -> Observation:
    return Observation(np.full((4, width, 3), value, dtype=np.uint8), tick)


def _trace(*, feedback: float | None = None, width: int = 5) -> PublicTrace:
    before = _observation(0, width=width)
    after = _observation(1, 31, width=width)
    transition = PublicTransition(
        before,
        Action(701, (2, 1), (1, 0)),
        after,
        feedback,
    )
    return PublicTrace(before, (transition,))


def _manifest() -> Grade3SessionManifest:
    return Grade3SessionManifest(
        GRADE3_PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        30,
        8.5,
        12,
        25.0,
        6,
    )


def test_grade3_contract_fields_are_an_exact_opaque_allowlist() -> None:
    exact = {
        Grade3SessionManifest: (
            "protocol_version",
            "sensor_schema",
            "action_schema",
            "support_record_budget",
            "acquisition_cost_budget",
            "query_budget",
            "motor_action_cost_budget",
            "motor_reset_budget",
        ),
        MotorActionSpace: ("action_codes", "motor_vectors", "max_trace_steps"),
        MotorQuery: (
            "query_id",
            "scope_id",
            "step_index",
            "utterance",
            "phase",
            "completed_probes",
            "current_trace",
            "action_space",
            "remaining_action_cost",
            "remaining_resets",
        ),
        MotorDecision: ("directive", "action", "unknown_probability"),
        ProbeEvidence: ("probe_id", "trace"),
        TraceBeliefQuery: (
            "query_id",
            "scope_id",
            "problem_id",
            "candidates",
            "evidence",
        ),
        TraceDescriptionQuery: ("query_id", "scope_id", "evidence"),
        OstensiveSupportRecord: ("scope_id", "source_id", "turn", "trace"),
        CausalSupportRecord: (
            "scope_id",
            "problem_id",
            "hypothesis_id",
            "probe_id",
            "source_id",
            "trace",
        ),
        ProbeOption: ("probe_id", "cost"),
        ProbeOffer: (
            "scope_id",
            "problem_id",
            "step_index",
            "options",
            "remaining_cost",
        ),
        ProbeDecision: ("probe_id", "unknown_probability"),
        ProbeResult: (
            "scope_id",
            "problem_id",
            "probe_id",
            "trace",
            "cost",
            "remaining_cost",
        ),
    }
    forbidden = {
        "seed",
        "world_variant",
        "semantic_label",
        "target_truth",
        "oracle_handle",
        "likelihood_table",
        "outcome_code",
    }
    for record_type, expected in exact.items():
        actual = tuple(field.name for field in fields(record_type))
        assert actual == expected
        assert forbidden.isdisjoint(actual)


def test_manifest_is_separate_schema_strict_and_finite() -> None:
    manifest = _manifest()
    assert manifest.protocol_version == "grounding-grade3-session/1"
    assert manifest.acquisition_cost_budget == 8.5

    with pytest.raises(ValueError, match="protocol_version"):
        Grade3SessionManifest(
            "grounding-session/1",
            SENSOR_SCHEMA_RGB_U8,
            ACTION_SCHEMA_OPAQUE_MOTOR,
            1,
            1,
            1,
            1,
            1,
        )
    with pytest.raises(ValueError, match="sensor_schema"):
        Grade3SessionManifest(
            GRADE3_PROTOCOL_VERSION,
            "semantic-object-array",
            ACTION_SCHEMA_OPAQUE_MOTOR,
            1,
            1,
            1,
            1,
            1,
        )
    with pytest.raises(TypeError, match="integer"):
        Grade3SessionManifest(
            GRADE3_PROTOCOL_VERSION,
            SENSOR_SCHEMA_RGB_U8,
            ACTION_SCHEMA_OPAQUE_MOTOR,
            True,
            1,
            1,
            1,
            1,
        )
    with pytest.raises(ValueError, match="finite"):
        Grade3SessionManifest(
            GRADE3_PROTOCOL_VERSION,
            SENSOR_SCHEMA_RGB_U8,
            ACTION_SCHEMA_OPAQUE_MOTOR,
            1,
            float("inf"),
            1,
            1,
            1,
        )


def test_motor_action_space_is_finite_unique_and_checks_full_actions() -> None:
    space = MotorActionSpace((701, 702), ((1, 0), (0, 0)), 8)
    assert space.permits(Action(701, (4, 3), (1, 0)), (4, 5, 3))
    assert not space.permits(Action(999, (4, 3), (1, 0)), (4, 5, 3))
    assert not space.permits(Action(701, (5, 3), (1, 0)), (4, 5, 3))
    assert not space.permits(Action(701, (4, 3), (-1, 0)), (4, 5, 3))

    with pytest.raises(ValueError, match="unique"):
        MotorActionSpace((701, 701), ((1, 0),), 8)
    with pytest.raises(ValueError, match="unique"):
        MotorActionSpace((701,), ((1, 0), (1, 0)), 8)
    with pytest.raises(ValueError, match="cannot exceed"):
        MotorActionSpace(tuple(range(MAX_GRADE3_ACTION_CODES + 1)), ((0, 0),), 8)
    with pytest.raises(ValueError, match="must lie"):
        MotorActionSpace((701,), ((0, 0),), MAX_GRADE3_TRANSITIONS + 1)


def test_motor_query_carries_complete_feedback_free_reset_history() -> None:
    trace = _trace()
    space = MotorActionSpace((701,), ((1, 0),), 8)
    query = MotorQuery(
        9,
        4,
        2,
        Utterance((101, 202)),
        MotorPhase.EXECUTE,
        (trace,),
        PublicTrace(_observation(0)),
        space,
        4.0,
        1,
    )
    assert query.completed_probes == (trace,)
    assert query.phase is MotorPhase.EXECUTE

    with pytest.raises(ValueError, match="requires completed"):
        MotorQuery(
            9,
            4,
            0,
            Utterance((101,)),
            MotorPhase.EXECUTE,
            (),
            PublicTrace(_observation(0)),
            space,
            4.0,
            1,
        )
    with pytest.raises(ValueError, match="feedback-free"):
        MotorQuery(
            9,
            4,
            0,
            Utterance((101,)),
            MotorPhase.PROBE,
            (),
            _trace(feedback=0.5),
            space,
            4.0,
            1,
        )
    with pytest.raises(ValueError, match="RGB shape"):
        MotorQuery(
            9,
            4,
            1,
            Utterance((101,)),
            MotorPhase.PROBE,
            (trace,),
            PublicTrace(_observation(0, width=6)),
            space,
            4.0,
            1,
        )
    with pytest.raises(ValueError, match="outside action_space"):
        MotorQuery(
            9,
            4,
            1,
            Utterance((101,)),
            MotorPhase.PROBE,
            (_trace(),),
            PublicTrace(_observation(0)),
            MotorActionSpace((999,), ((1, 0),), 8),
            4.0,
            1,
        )


def test_motor_and_probe_decisions_make_abstention_unambiguous() -> None:
    action = Action(701, (2, 1), (1, 0))
    assert MotorDecision(MotorDirective.ACT, action, 0.0).action == action
    assert MotorDecision(MotorDirective.ABSTAIN, None, 0.7).unknown_probability == 0.7
    assert ProbeDecision(8, 0.0).probe_id == 8
    assert ProbeDecision(None, 1.0).probe_id is None

    with pytest.raises(ValueError, match="exactly ACT"):
        MotorDecision(MotorDirective.ACT, None, 0.0)
    with pytest.raises(ValueError, match="exactly ACT"):
        MotorDecision(MotorDirective.COMPLETE, action, 0.0)
    with pytest.raises(ValueError, match="positive"):
        MotorDecision(MotorDirective.ABSTAIN, None, 0.0)
    with pytest.raises(ValueError, match="zero"):
        ProbeDecision(8, 0.1)
    with pytest.raises(ValueError, match="positive"):
        ProbeDecision(None, 0.0)


def test_support_and_query_evidence_are_raw_bounded_and_canonicalized() -> None:
    trace = _trace()
    turn = PublicTurn(
        2,
        SessionPhase.SUPPORT,
        trace.initial,
        Utterance((101,)),
        scalar_feedback=1.0,
    )
    support = OstensiveSupportRecord(3, 17, turn, trace)
    assert support.turn.scalar_feedback == 1.0
    assert not support.trace.has_feedback

    causal = CausalSupportRecord(3, 4, 5, 6, 17, trace)
    assert causal.hypothesis_id == 5
    with pytest.raises(ValueError, match="nonempty"):
        CausalSupportRecord(3, 4, 5, 6, 17, PublicTrace(trace.initial))
    with pytest.raises(ValueError, match="feedback-free"):
        CausalSupportRecord(3, 4, 5, 6, 17, _trace(feedback=-0.5))

    with pytest.raises(ValueError, match="support-phase"):
        OstensiveSupportRecord(
            3,
            17,
            PublicTurn(2, SessionPhase.QUERY, trace.initial, Utterance((101,))),
            trace,
        )
    with pytest.raises(ValueError, match="utterance"):
        OstensiveSupportRecord(
            3,
            17,
            PublicTurn(2, SessionPhase.SUPPORT, trace.initial),
            trace,
        )

    first = ProbeEvidence(20, trace)
    second = ProbeEvidence(10, trace)
    belief = TraceBeliefQuery(1, 3, 4, (99, 88), (first, second))
    description = TraceDescriptionQuery(2, 3, (first, second))
    assert tuple(item.probe_id for item in belief.evidence) == (10, 20)
    assert tuple(item.probe_id for item in description.evidence) == (10, 20)
    with pytest.raises(ValueError, match="unique"):
        TraceBeliefQuery(1, 3, 4, (99, 99), (first,))
    with pytest.raises(ValueError, match="unique"):
        TraceDescriptionQuery(1, 3, (first, first))
    with pytest.raises(ValueError, match="feedback-free"):
        ProbeEvidence(1, _trace(feedback=0.1))
    with pytest.raises(ValueError, match="must lie"):
        ProbeEvidence(MAX_GRADE3_INTEGER + 1, trace)


def test_probe_offers_and_results_are_costed_and_feedback_free() -> None:
    offer = ProbeOffer(3, 4, 0, (ProbeOption(8, 0.5), ProbeOption(9, 1.5)), 1.0)
    assert tuple(option.probe_id for option in offer.options) == (8, 9)
    result = ProbeResult(3, 4, 8, _trace(), 0.5, 0.5)
    assert result.remaining_cost == 0.5

    with pytest.raises(ValueError, match="unique"):
        ProbeOffer(3, 4, 0, (ProbeOption(8, 0.5), ProbeOption(8, 1.0)), 1.0)
    with pytest.raises(ValueError, match="at least"):
        ProbeOption(8, 0.0)
    with pytest.raises(ValueError, match="feedback-free"):
        ProbeResult(3, 4, 8, _trace(feedback=0.2), 0.5, 0.5)
