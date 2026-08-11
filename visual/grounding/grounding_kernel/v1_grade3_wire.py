"""Strict canonical JSON wire for ``grounding-grade3-session/1``."""

from __future__ import annotations

from dataclasses import asdict
from math import isfinite
from typing import Any
import json

from .contracts import Action
from .v1_contracts import (
    PROTOCOL_VERSION,
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    PublicTrace,
    PublicTurn,
    Utterance,
)
from .v1_grade3_contracts import (
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
from .v1_wire import (
    MAX_JSON_DEPTH,
    MAX_JSON_NODES,
    MAX_MESSAGE_BYTES,
    MAX_PIXEL_BYTES,
    MAX_WIRE_INTEGER,
    WireProtocolError,
    decode_message,
    encode_message,
)


def _exact(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise WireProtocolError(f"{label} must be a JSON object")
    actual = set(value)
    if actual != keys:
        raise WireProtocolError(
            f"{label} keys must be exactly {sorted(keys)!r}; "
            f"missing={sorted(keys - actual)!r}, extra={sorted(actual - keys)!r}"
        )
    return value


def _integer(value: object, label: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireProtocolError(f"{label} must be an integer")
    if not minimum <= value <= MAX_WIRE_INTEGER:
        raise WireProtocolError(f"{label} lies outside the wire range")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise WireProtocolError(f"{label} must be finite")
    return result


def _walk(value: object) -> int:
    """Validate and count a JSON tree without unbounded recursion."""

    stack: list[tuple[object, int]] = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        if depth > MAX_JSON_DEPTH:
            raise WireProtocolError("JSON nesting exceeds the protocol limit")
        nodes += 1
        if nodes > MAX_JSON_NODES:
            raise WireProtocolError("message exceeds the JSON-node limit")
        if isinstance(current, dict):
            if any(not isinstance(key, str) for key in current):
                raise WireProtocolError("JSON keys must be strings")
            stack.extend((item, depth + 1) for item in current.values())
        elif isinstance(current, list):
            stack.extend((item, depth + 1) for item in current)
        elif current is None or isinstance(current, (str, bool, int)):
            continue
        elif isinstance(current, float) and isfinite(current):
            continue
        else:
            raise WireProtocolError(f"unsupported JSON value {type(current).__name__}")
    return nodes


def _reject_constant(value: str) -> None:
    raise WireProtocolError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WireProtocolError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _preflight_json(data: bytes) -> None:
    """Reject excessive structural depth before invoking the JSON parser."""

    depth = 0
    in_string = False
    escaped = False
    for byte in data:
        if in_string:
            if escaped:
                escaped = False
            elif byte == 0x5C:  # backslash
                escaped = True
            elif byte == 0x22:  # quote
                in_string = False
            continue
        if byte == 0x22:
            in_string = True
        elif byte in (0x7B, 0x5B):  # { [
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise WireProtocolError("JSON nesting exceeds the protocol limit")
        elif byte in (0x7D, 0x5D):  # } ]
            depth -= 1
            if depth < 0:
                raise WireProtocolError("wire message has unbalanced JSON delimiters")
    if depth != 0:
        raise WireProtocolError("wire message has unbalanced JSON delimiters")


def _message_limit(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("max_message_bytes must be an integer")
    if value < 1:
        raise ValueError("max_message_bytes must be positive")
    return value


def _base_payload(value: object, expected_type: str) -> dict[str, Any]:
    envelope = json.loads(encode_message(value).decode("utf-8"))
    if envelope["type"] != expected_type:
        raise TypeError(f"expected legacy wire type {expected_type}")
    return envelope["payload"]


def _decode_base(payload: object, message_type: str) -> object:
    envelope = {
        "protocol": PROTOCOL_VERSION,
        "type": message_type,
        "payload": payload,
    }
    return decode_message(
        json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )


def _trace(trace: PublicTrace) -> dict[str, Any]:
    return _base_payload(trace, "public_trace")


def _decode_trace(value: object) -> PublicTrace:
    result = _decode_base(value, "public_trace")
    if not isinstance(result, PublicTrace):  # pragma: no cover
        raise WireProtocolError("decoded value is not PublicTrace")
    return result


def _utterance(value: Utterance) -> dict[str, Any]:
    return _base_payload(value, "utterance")


def _decode_utterance(value: object) -> Utterance:
    result = _decode_base(value, "utterance")
    if not isinstance(result, Utterance):  # pragma: no cover
        raise WireProtocolError("decoded value is not Utterance")
    return result


def _turn(value: PublicTurn) -> dict[str, Any]:
    return _base_payload(value, "public_turn")


def _decode_turn(value: object) -> PublicTurn:
    result = _decode_base(value, "public_turn")
    if not isinstance(result, PublicTurn):  # pragma: no cover
        raise WireProtocolError("decoded value is not PublicTurn")
    return result


def _action(value: Action | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return _base_payload(ActionDecision(value, 0.0), "action_decision")["action"]


def _decode_action(value: object) -> Action | None:
    if value is None:
        return None
    result = _decode_base(
        {"action": value, "unknown_probability": 0.0},
        "action_decision",
    )
    if not isinstance(result, ActionDecision):  # pragma: no cover
        raise WireProtocolError("decoded value is not ActionDecision")
    return result.action


def _action_space(value: MotorActionSpace) -> dict[str, Any]:
    return {
        "action_codes": list(value.action_codes),
        "motor_vectors": [list(vector) for vector in value.motor_vectors],
        "max_trace_steps": value.max_trace_steps,
    }


def _decode_action_space(value: object) -> MotorActionSpace:
    obj = _exact(
        value,
        {"action_codes", "motor_vectors", "max_trace_steps"},
        "action_space",
    )
    codes = obj["action_codes"]
    vectors = obj["motor_vectors"]
    if not isinstance(codes, list) or not isinstance(vectors, list):
        raise WireProtocolError("action space codes/vectors must be lists")
    converted_vectors: list[tuple[int, int]] = []
    for vector in vectors:
        if not isinstance(vector, list) or len(vector) != 2:
            raise WireProtocolError("motor vectors must be two-integer lists")
        converted_vectors.append(
            (
                _integer(vector[0], "motor_vector", minimum=-MAX_WIRE_INTEGER),
                _integer(vector[1], "motor_vector", minimum=-MAX_WIRE_INTEGER),
            )
        )
    return MotorActionSpace(
        tuple(_integer(code, "action_code", minimum=-MAX_WIRE_INTEGER) for code in codes),
        tuple(converted_vectors),
        _integer(obj["max_trace_steps"], "max_trace_steps", minimum=1),
    )


def _evidence(value: ProbeEvidence) -> dict[str, Any]:
    return {"probe_id": value.probe_id, "trace": _trace(value.trace)}


def _decode_evidence(value: object) -> ProbeEvidence:
    obj = _exact(value, {"probe_id", "trace"}, "probe_evidence")
    return ProbeEvidence(_integer(obj["probe_id"], "probe_id"), _decode_trace(obj["trace"]))


def _payload(value: object) -> tuple[str, dict[str, Any]]:
    if isinstance(value, Grade3SessionManifest):
        return "grade3_manifest", asdict(value)
    if isinstance(value, OstensiveSupportRecord):
        return "ostensive_support", {
            "scope_id": value.scope_id,
            "source_id": value.source_id,
            "turn": _turn(value.turn),
            "trace": _trace(value.trace),
        }
    if isinstance(value, CausalSupportRecord):
        return "causal_support", {
            "scope_id": value.scope_id,
            "problem_id": value.problem_id,
            "hypothesis_id": value.hypothesis_id,
            "probe_id": value.probe_id,
            "source_id": value.source_id,
            "trace": _trace(value.trace),
        }
    if isinstance(value, ProbeOffer):
        return "probe_offer", {
            "scope_id": value.scope_id,
            "problem_id": value.problem_id,
            "step_index": value.step_index,
            "options": [asdict(option) for option in value.options],
            "remaining_cost": value.remaining_cost,
        }
    if isinstance(value, ProbeDecision):
        return "probe_decision", asdict(value)
    if isinstance(value, ProbeResult):
        return "probe_result", {
            "scope_id": value.scope_id,
            "problem_id": value.problem_id,
            "probe_id": value.probe_id,
            "trace": _trace(value.trace),
            "cost": value.cost,
            "remaining_cost": value.remaining_cost,
        }
    if isinstance(value, MotorQuery):
        return "motor_query", {
            "query_id": value.query_id,
            "scope_id": value.scope_id,
            "step_index": value.step_index,
            "utterance": _utterance(value.utterance),
            "phase": value.phase.value,
            "completed_probes": [_trace(trace) for trace in value.completed_probes],
            "current_trace": _trace(value.current_trace),
            "action_space": _action_space(value.action_space),
            "remaining_action_cost": value.remaining_action_cost,
            "remaining_resets": value.remaining_resets,
        }
    if isinstance(value, MotorDecision):
        return "motor_decision", {
            "directive": value.directive.value,
            "action": _action(value.action),
            "unknown_probability": value.unknown_probability,
        }
    if isinstance(value, TraceBeliefQuery):
        return "trace_belief_query", {
            "query_id": value.query_id,
            "scope_id": value.scope_id,
            "problem_id": value.problem_id,
            "candidates": list(value.candidates),
            "evidence": [_evidence(item) for item in value.evidence],
        }
    if isinstance(value, TraceDescriptionQuery):
        return "trace_description_query", {
            "query_id": value.query_id,
            "scope_id": value.scope_id,
            "evidence": [_evidence(item) for item in value.evidence],
        }
    if isinstance(value, BeliefDecision):
        return "belief_decision", _base_payload(value, "belief_decision")
    if isinstance(value, DescriptionDecision):
        return "description_decision", _base_payload(value, "description_decision")
    raise TypeError(f"unsupported Grade-3 wire value: {type(value).__name__}")


def encode_grade3_message(value: object, *, max_message_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    limit = _message_limit(max_message_bytes)
    message_type, payload = _payload(value)
    envelope = {
        "protocol": GRADE3_PROTOCOL_VERSION,
        "type": message_type,
        "payload": payload,
    }
    _walk(envelope)
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > limit:
        raise WireProtocolError("encoded message exceeds max_message_bytes")
    return encoded


def _parse(data: bytes, max_message_bytes: int) -> dict[str, Any]:
    if not isinstance(data, bytes):
        raise TypeError("Grade-3 wire messages must be bytes")
    limit = _message_limit(max_message_bytes)
    if len(data) > limit:
        raise WireProtocolError("Grade-3 wire message exceeds max_message_bytes")
    _preflight_json(data)
    try:
        decoded = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicates,
        )
    except (
        UnicodeDecodeError,
        json.JSONDecodeError,
        ValueError,
        RecursionError,
        OverflowError,
    ) as exc:
        if isinstance(exc, WireProtocolError):
            raise
        raise WireProtocolError("message is not strict UTF-8 JSON") from exc
    _walk(decoded)
    envelope = _exact(decoded, {"protocol", "type", "payload"}, "envelope")
    if envelope["protocol"] != GRADE3_PROTOCOL_VERSION:
        raise WireProtocolError("unsupported Grade-3 protocol version")
    if not isinstance(envelope["type"], str):
        raise WireProtocolError("envelope.type must be a string")
    return envelope


def decode_grade3_message(data: bytes, *, max_message_bytes: int = MAX_MESSAGE_BYTES) -> object:
    envelope = _parse(data, max_message_bytes)
    kind = envelope["type"]
    payload = envelope["payload"]
    try:
        if kind == "grade3_manifest":
            obj = _exact(
                payload,
                {
                    "protocol_version",
                    "sensor_schema",
                    "action_schema",
                    "support_record_budget",
                    "acquisition_cost_budget",
                    "query_budget",
                    "motor_action_cost_budget",
                    "motor_reset_budget",
                },
                kind,
            )
            return Grade3SessionManifest(
                protocol_version=obj["protocol_version"],
                sensor_schema=obj["sensor_schema"],
                action_schema=obj["action_schema"],
                support_record_budget=_integer(
                    obj["support_record_budget"], "support_record_budget"
                ),
                acquisition_cost_budget=_number(
                    obj["acquisition_cost_budget"], "acquisition_cost_budget"
                ),
                query_budget=_integer(obj["query_budget"], "query_budget", minimum=1),
                motor_action_cost_budget=_number(
                    obj["motor_action_cost_budget"], "motor_action_cost_budget"
                ),
                motor_reset_budget=_integer(obj["motor_reset_budget"], "motor_reset_budget"),
            )
        if kind == "ostensive_support":
            obj = _exact(payload, {"scope_id", "source_id", "turn", "trace"}, kind)
            return OstensiveSupportRecord(
                _integer(obj["scope_id"], "scope_id"),
                _integer(obj["source_id"], "source_id"),
                _decode_turn(obj["turn"]),
                _decode_trace(obj["trace"]),
            )
        if kind == "causal_support":
            obj = _exact(
                payload,
                {
                    "scope_id",
                    "problem_id",
                    "hypothesis_id",
                    "probe_id",
                    "source_id",
                    "trace",
                },
                kind,
            )
            return CausalSupportRecord(
                *(
                    _integer(obj[field], field)
                    for field in (
                        "scope_id",
                        "problem_id",
                        "hypothesis_id",
                        "probe_id",
                        "source_id",
                    )
                ),
                _decode_trace(obj["trace"]),
            )
        if kind == "probe_offer":
            obj = _exact(
                payload,
                {
                    "scope_id",
                    "problem_id",
                    "step_index",
                    "options",
                    "remaining_cost",
                },
                kind,
            )
            if not isinstance(obj["options"], list):
                raise WireProtocolError("options must be a list")
            options = []
            for value in obj["options"]:
                option = _exact(value, {"probe_id", "cost"}, "probe_option")
                options.append(
                    ProbeOption(
                        _integer(option["probe_id"], "probe_id"),
                        _number(option["cost"], "cost"),
                    )
                )
            return ProbeOffer(
                _integer(obj["scope_id"], "scope_id"),
                _integer(obj["problem_id"], "problem_id"),
                _integer(obj["step_index"], "step_index"),
                tuple(options),
                _number(obj["remaining_cost"], "remaining_cost"),
            )
        if kind == "probe_decision":
            obj = _exact(payload, {"probe_id", "unknown_probability"}, kind)
            return ProbeDecision(
                None if obj["probe_id"] is None else _integer(obj["probe_id"], "probe_id"),
                _number(obj["unknown_probability"], "unknown_probability"),
            )
        if kind == "probe_result":
            obj = _exact(
                payload,
                {
                    "scope_id",
                    "problem_id",
                    "probe_id",
                    "trace",
                    "cost",
                    "remaining_cost",
                },
                kind,
            )
            return ProbeResult(
                _integer(obj["scope_id"], "scope_id"),
                _integer(obj["problem_id"], "problem_id"),
                _integer(obj["probe_id"], "probe_id"),
                _decode_trace(obj["trace"]),
                _number(obj["cost"], "cost"),
                _number(obj["remaining_cost"], "remaining_cost"),
            )
        if kind == "motor_query":
            obj = _exact(
                payload,
                {
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
                },
                kind,
            )
            if not isinstance(obj["completed_probes"], list):
                raise WireProtocolError("completed_probes must be a list")
            return MotorQuery(
                _integer(obj["query_id"], "query_id"),
                _integer(obj["scope_id"], "scope_id"),
                _integer(obj["step_index"], "step_index"),
                _decode_utterance(obj["utterance"]),
                MotorPhase(obj["phase"]),
                tuple(_decode_trace(trace) for trace in obj["completed_probes"]),
                _decode_trace(obj["current_trace"]),
                _decode_action_space(obj["action_space"]),
                _number(obj["remaining_action_cost"], "remaining_action_cost"),
                _integer(obj["remaining_resets"], "remaining_resets"),
            )
        if kind == "motor_decision":
            obj = _exact(payload, {"directive", "action", "unknown_probability"}, kind)
            return MotorDecision(
                MotorDirective(obj["directive"]),
                _decode_action(obj["action"]),
                _number(obj["unknown_probability"], "unknown_probability"),
            )
        if kind == "trace_belief_query":
            obj = _exact(
                payload,
                {"query_id", "scope_id", "problem_id", "candidates", "evidence"},
                kind,
            )
            if not isinstance(obj["candidates"], list) or not isinstance(obj["evidence"], list):
                raise WireProtocolError("candidates/evidence must be lists")
            return TraceBeliefQuery(
                _integer(obj["query_id"], "query_id"),
                _integer(obj["scope_id"], "scope_id"),
                _integer(obj["problem_id"], "problem_id"),
                tuple(_integer(value, "candidate") for value in obj["candidates"]),
                tuple(_decode_evidence(value) for value in obj["evidence"]),
            )
        if kind == "trace_description_query":
            obj = _exact(
                payload,
                {"query_id", "scope_id", "evidence"},
                kind,
            )
            if not isinstance(obj["evidence"], list):
                raise WireProtocolError("evidence must be a list")
            return TraceDescriptionQuery(
                _integer(obj["query_id"], "query_id"),
                _integer(obj["scope_id"], "scope_id"),
                tuple(_decode_evidence(value) for value in obj["evidence"]),
            )
        if kind in {"belief_decision", "description_decision"}:
            return _decode_base(payload, kind)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, WireProtocolError):
            raise
        raise WireProtocolError(f"invalid {kind} payload: {exc}") from exc
    raise WireProtocolError(f"unsupported Grade-3 message type: {kind!r}")


def grade3_schema_manifest() -> dict[str, Any]:
    return {
        "protocol": GRADE3_PROTOCOL_VERSION,
        "top_level_types": [
            "belief_decision",
            "causal_support",
            "description_decision",
            "grade3_manifest",
            "motor_decision",
            "motor_query",
            "ostensive_support",
            "probe_decision",
            "probe_offer",
            "probe_result",
            "trace_belief_query",
            "trace_description_query",
        ],
        "transport": "canonical-json-utf8",
        "max_message_bytes": MAX_MESSAGE_BYTES,
        "max_pixel_bytes": MAX_PIXEL_BYTES,
        "max_json_depth": MAX_JSON_DEPTH,
        "max_json_nodes": MAX_JSON_NODES,
        "max_wire_integer": MAX_WIRE_INTEGER,
        "forbidden": [
            "pickle",
            "callable",
            "seed",
            "world_variant",
            "semantic_label",
            "target_truth",
            "oracle_handle",
            "likelihood_table",
            "scalar_feedback_in_trace",
        ],
    }


__all__ = [
    "decode_grade3_message",
    "encode_grade3_message",
    "grade3_schema_manifest",
]
