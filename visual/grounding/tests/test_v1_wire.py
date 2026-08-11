from __future__ import annotations

import json

import numpy as np
import pytest

from grounding_kernel.contracts import Action, Observation
from grounding_kernel.v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    PROTOCOL_VERSION,
    SENSOR_SCHEMA_RGB_U8,
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    ExperimentDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
    Utterance,
)
from grounding_kernel.v1_wire import (
    WireProtocolError,
    decode_message,
    encode_message,
    public_schema_manifest,
)


def _observation(tick: int, value: int = 0) -> Observation:
    return Observation(np.full((3, 4, 3), value, dtype=np.uint8), tick)


def _values() -> tuple[object, ...]:
    before = _observation(0)
    after = _observation(1, 33)
    action = Action(5_000_001, (2, 1), (1, 0))
    transition = PublicTransition(before, action, after, -0.25)
    return (
        SessionManifest(
            PROTOCOL_VERSION,
            SENSOR_SCHEMA_RGB_U8,
            ACTION_SCHEMA_OPAQUE_MOTOR,
            4,
            6.5,
            8,
        ),
        Utterance((101, 303)),
        transition,
        PublicTrace(before, (transition,)),
        PublicTurn(
            0,
            SessionPhase.SUPPORT,
            before,
            Utterance((101,)),
            (0, 0, 2, 2),
            None,
            6.5,
        ),
        ExperimentDecision(action, 0.1),
        DescriptionDecision(Utterance((303,)), 0.2),
        ActionDecision(None, 1.0),
        BeliefDecision(((101, 0.7),), 0.3),
    )


@pytest.mark.parametrize("value", _values(), ids=lambda value: type(value).__name__)
def test_every_allowlisted_value_roundtrips(value: object) -> None:
    assert decode_message(encode_message(value)) == value


def test_wire_is_canonical_and_schema_manifest_is_explicit() -> None:
    value = Utterance((7, 9))
    assert encode_message(value) == encode_message(value)
    envelope = json.loads(encode_message(value))
    assert list(envelope) == ["payload", "protocol", "type"]
    manifest = public_schema_manifest()
    assert manifest["protocol"] == PROTOCOL_VERSION
    assert "evaluator-enum" in manifest["forbidden"]


def test_extra_missing_unknown_and_nonfinite_payloads_fail_closed() -> None:
    envelope = json.loads(encode_message(Utterance((7,))))
    envelope["payload"]["semantic_name"] = "house"
    with pytest.raises(WireProtocolError, match="extra"):
        decode_message(json.dumps(envelope).encode())

    envelope = json.loads(encode_message(Utterance((7,))))
    del envelope["payload"]["tokens"]
    with pytest.raises(WireProtocolError, match="missing"):
        decode_message(json.dumps(envelope).encode())

    envelope = {"protocol": PROTOCOL_VERSION, "type": "pickle", "payload": {}}
    with pytest.raises(WireProtocolError, match="unsupported"):
        decode_message(json.dumps(envelope).encode())

    raw = (
        b'{"payload":{"action":null,"unknown_probability":NaN},'
        b'"protocol":"grounding-session/1","type":"action_decision"}'
    )
    with pytest.raises(WireProtocolError, match="non-finite"):
        decode_message(raw)

    duplicate = (
        b'{"payload":{"tokens":[7]},"protocol":"grounding-session/1",'
        b'"type":"utterance","type":"belief_decision"}'
    )
    with pytest.raises(WireProtocolError, match="duplicate"):
        decode_message(duplicate)


def test_bad_pixels_type_and_oversize_messages_are_rejected() -> None:
    encoded = encode_message(PublicTrace(_observation(0)))
    envelope = json.loads(encoded)
    envelope["payload"]["initial"]["shape"] = [999, 999, 3]
    with pytest.raises(WireProtocolError, match="length"):
        decode_message(json.dumps(envelope).encode())

    with pytest.raises(TypeError, match="unsupported"):
        encode_message(object())
    with pytest.raises(WireProtocolError, match="max_message_bytes"):
        encode_message(Utterance((1,)), max_message_bytes=5)
    with pytest.raises(TypeError, match="bytes"):
        decode_message("not bytes")  # type: ignore[arg-type]


def test_action_keys_are_exact_and_boolean_integer_smuggling_is_rejected() -> None:
    envelope = json.loads(encode_message(ActionDecision(Action(9, (1, 1)), 0.0)))
    envelope["payload"]["action"]["object_id"] = 4
    with pytest.raises(WireProtocolError, match="extra"):
        decode_message(json.dumps(envelope).encode())

    envelope = json.loads(encode_message(Utterance((7,))))
    envelope["payload"]["tokens"] = [True]
    with pytest.raises(WireProtocolError, match="integer"):
        decode_message(json.dumps(envelope).encode())


def test_deep_json_huge_integers_and_negative_ticks_are_wire_errors() -> None:
    deep = b"[" * 40 + b"0" + b"]" * 40
    with pytest.raises(WireProtocolError, match="nesting"):
        decode_message(deep)

    envelope = json.loads(encode_message(Utterance((7,))))
    envelope["payload"]["tokens"] = [10**100]
    with pytest.raises(WireProtocolError, match="integer range"):
        decode_message(json.dumps(envelope).encode())

    trace = json.loads(encode_message(PublicTrace(_observation(0))))
    trace["payload"]["initial"]["tick"] = -1
    with pytest.raises(WireProtocolError, match="integer range"):
        decode_message(json.dumps(trace).encode())
