from __future__ import annotations

from hashlib import sha256
import json

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    SENSOR_SCHEMA_RGB_U8,
    BeliefDecision,
    DescriptionDecision,
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
from grounding_kernel.v1_grade3_wire import (
    decode_grade3_message,
    encode_grade3_message,
    grade3_schema_manifest,
)
from grounding_kernel.v1_wire import (
    MAX_JSON_NODES,
    WireProtocolError,
    public_schema_manifest,
)


def _observation(tick: int, value: int = 0) -> Observation:
    return Observation(np.full((4, 5, 3), value, dtype=np.uint8), tick)


def _trace() -> PublicTrace:
    before = _observation(0)
    after = _observation(1, 41)
    transition = PublicTransition(
        before,
        Action(701, (2, 1), (1, 0)),
        after,
    )
    return PublicTrace(before, (transition,))


def _values() -> tuple[object, ...]:
    trace = _trace()
    evidence = ProbeEvidence(19, trace)
    action_space = MotorActionSpace((701,), ((1, 0),), 8)
    turn = PublicTurn(
        2,
        SessionPhase.SUPPORT,
        trace.initial,
        Utterance((101,)),
        (1, 0, 4, 3),
        1.0,
        3.0,
    )
    return (
        Grade3SessionManifest(
            GRADE3_PROTOCOL_VERSION,
            SENSOR_SCHEMA_RGB_U8,
            ACTION_SCHEMA_OPAQUE_MOTOR,
            30,
            8.5,
            12,
            25.0,
            6,
        ),
        OstensiveSupportRecord(3, 17, turn, trace),
        CausalSupportRecord(3, 4, 5, 19, 17, trace),
        ProbeOffer(3, 4, 0, (ProbeOption(19, 0.5), ProbeOption(23, 1.0)), 2.0),
        ProbeDecision(19, 0.0),
        ProbeResult(3, 4, 19, trace, 0.5, 1.5),
        MotorQuery(
            9,
            3,
            2,
            Utterance((101, 202)),
            MotorPhase.EXECUTE,
            (trace,),
            PublicTrace(_observation(0)),
            action_space,
            4.0,
            1,
        ),
        MotorDecision(
            MotorDirective.ACT,
            Action(701, (2, 1), (1, 0)),
            0.0,
        ),
        TraceBeliefQuery(10, 3, 4, (5, 6), (evidence,)),
        TraceDescriptionQuery(11, 3, (evidence,)),
        BeliefDecision(((5, 0.75),), 0.25),
        DescriptionDecision(Utterance((101, 202)), 0.0),
    )


@pytest.mark.parametrize("value", _values(), ids=lambda value: type(value).__name__)
def test_every_grade3_top_level_type_has_a_canonical_roundtrip(value: object) -> None:
    encoded = encode_grade3_message(value)
    assert encoded == encode_grade3_message(value)
    assert decode_grade3_message(encoded) == value
    envelope = json.loads(encoded)
    assert envelope["protocol"] == GRADE3_PROTOCOL_VERSION
    assert list(envelope) == ["payload", "protocol", "type"]


def test_schema_manifest_exactly_matches_the_roundtripped_type_allowlist() -> None:
    manifest = grade3_schema_manifest()
    actual = sorted(json.loads(encode_grade3_message(value))["type"] for value in _values())
    assert actual == manifest["top_level_types"]
    assert manifest["protocol"] == "grounding-grade3-session/1"
    assert manifest["transport"] == "canonical-json-utf8"
    assert "scalar_feedback_in_trace" in manifest["forbidden"]
    assert "semantic_label" in manifest["forbidden"]


def test_nested_traces_are_raw_and_never_disclose_outcome_codes() -> None:
    envelope = json.loads(encode_grade3_message(CausalSupportRecord(3, 4, 5, 19, 17, _trace())))
    transition = envelope["payload"]["trace"]["transitions"][0]
    assert transition["scalar_feedback"] is None
    assert "outcome_code" not in transition
    assert "semantic_label" not in json.dumps(envelope)

    transition["scalar_feedback"] = 0.25
    with pytest.raises(WireProtocolError, match="feedback-free"):
        decode_grade3_message(json.dumps(envelope).encode())

    transition["scalar_feedback"] = None
    transition["outcome_code"] = 999
    with pytest.raises(WireProtocolError, match="extra"):
        decode_grade3_message(json.dumps(envelope).encode())


def test_extra_missing_unknown_wrong_protocol_and_nonfinite_fail_closed() -> None:
    envelope = json.loads(encode_grade3_message(ProbeDecision(19, 0.0)))
    envelope["payload"]["target_truth"] = 19
    with pytest.raises(WireProtocolError, match="extra"):
        decode_grade3_message(json.dumps(envelope).encode())

    envelope = json.loads(encode_grade3_message(ProbeDecision(19, 0.0)))
    del envelope["payload"]["probe_id"]
    with pytest.raises(WireProtocolError, match="missing"):
        decode_grade3_message(json.dumps(envelope).encode())

    envelope = {
        "protocol": GRADE3_PROTOCOL_VERSION,
        "type": "pickle",
        "payload": {},
    }
    with pytest.raises(WireProtocolError, match="unsupported"):
        decode_grade3_message(json.dumps(envelope).encode())

    envelope["protocol"] = "grounding-session/1"
    with pytest.raises(WireProtocolError, match="protocol version"):
        decode_grade3_message(json.dumps(envelope).encode())

    raw = (
        b'{"payload":{"probe_id":null,"unknown_probability":NaN},'
        b'"protocol":"grounding-grade3-session/1","type":"probe_decision"}'
    )
    with pytest.raises(WireProtocolError, match="non-finite"):
        decode_grade3_message(raw)


def test_duplicate_keys_boolean_smuggling_and_huge_integers_are_rejected() -> None:
    duplicate = (
        b'{"payload":{"probe_id":19,"unknown_probability":0.0},'
        b'"protocol":"grounding-grade3-session/1",'
        b'"type":"probe_decision","type":"motor_decision"}'
    )
    with pytest.raises(WireProtocolError, match="duplicate"):
        decode_grade3_message(duplicate)

    query = next(value for value in _values() if isinstance(value, MotorQuery))
    nested_duplicate = encode_grade3_message(query).replace(
        b'"max_trace_steps":8',
        b'"max_trace_steps":8,"max_trace_steps":8',
        1,
    )
    with pytest.raises(WireProtocolError, match="duplicate"):
        decode_grade3_message(nested_duplicate)

    envelope = json.loads(encode_grade3_message(ProbeDecision(19, 0.0)))
    envelope["payload"]["probe_id"] = True
    with pytest.raises(WireProtocolError, match="integer"):
        decode_grade3_message(json.dumps(envelope).encode())

    envelope["payload"]["probe_id"] = 10**100
    with pytest.raises(WireProtocolError, match="wire range"):
        decode_grade3_message(json.dumps(envelope).encode())


def test_depth_node_and_byte_limits_are_enforced_before_semantic_decode() -> None:
    deep = b"[" * 40 + b"0" + b"]" * 40
    with pytest.raises(WireProtocolError, match="nesting"):
        decode_grade3_message(deep)

    huge_nodes = json.dumps(
        {
            "protocol": GRADE3_PROTOCOL_VERSION,
            "type": "unknown",
            "payload": [0] * MAX_JSON_NODES,
        },
        separators=(",", ":"),
    ).encode()
    with pytest.raises(WireProtocolError, match="JSON-node"):
        decode_grade3_message(huge_nodes)

    with pytest.raises(WireProtocolError, match="max_message_bytes"):
        encode_grade3_message(ProbeDecision(19, 0.0), max_message_bytes=5)
    encoded = encode_grade3_message(ProbeDecision(19, 0.0))
    with pytest.raises(WireProtocolError, match="max_message_bytes"):
        decode_grade3_message(encoded, max_message_bytes=5)
    with pytest.raises(TypeError, match="bytes"):
        decode_grade3_message("not bytes")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="integer"):
        encode_grade3_message(ProbeDecision(19, 0.0), max_message_bytes=True)
    with pytest.raises(ValueError, match="positive"):
        decode_grade3_message(encoded, max_message_bytes=0)
    with pytest.raises(WireProtocolError, match="unbalanced"):
        decode_grade3_message(b'{"payload":{}')


def test_legacy_v1_schema_commitment_is_unchanged_by_grade3_protocol() -> None:
    canonical = json.dumps(
        public_schema_manifest(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    assert sha256(canonical).hexdigest() == (
        "be96b697fc255deddad8b4d036f70c445b7f5c4483451c31d699d3591dd10627"
    )
    assert public_schema_manifest()["protocol"] == "grounding-session/1"
    assert GRADE3_PROTOCOL_VERSION != public_schema_manifest()["protocol"]
