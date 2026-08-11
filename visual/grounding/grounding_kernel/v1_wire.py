"""Strict, size-bounded JSON wire format for GroundZero-v1 public records."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from math import isfinite, prod
from typing import Any
import base64
import binascii
import json

import numpy as np

from .contracts import Action, Observation
from .v1_contracts import (
    MAX_PUBLIC_TICK,
    PROTOCOL_VERSION,
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


MAX_MESSAGE_BYTES = 16 * 1024 * 1024
MAX_PIXEL_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 32
MAX_JSON_NODES = 100_000
MAX_WIRE_INTEGER = (1 << 63) - 1


class WireProtocolError(ValueError):
    """A v1 message is malformed, oversized, unsupported, or ambiguous."""


def _exact(mapping: object, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(mapping, dict):
        raise WireProtocolError(f"{label} must be a JSON object")
    actual = set(mapping)
    if actual != keys:
        raise WireProtocolError(
            f"{label} keys must be exactly {sorted(keys)!r}; "
            f"missing={sorted(keys - actual)!r}, extra={sorted(actual - keys)!r}"
        )
    return mapping


def _strict_int(
    value: object,
    label: str,
    *,
    minimum: int = -MAX_WIRE_INTEGER,
    maximum: int = MAX_WIRE_INTEGER,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise WireProtocolError(f"{label} must be an integer")
    if not minimum <= value <= maximum:
        raise WireProtocolError(f"{label} lies outside the wire integer range")
    return value


def _strict_float(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WireProtocolError(f"{label} must be a finite number")
    result = float(value)
    if not isfinite(result):
        raise WireProtocolError(f"{label} must be finite")
    return result


def _walk_limits(value: Any, *, depth: int = 0) -> int:
    if depth > MAX_JSON_DEPTH:
        raise WireProtocolError("JSON nesting exceeds the protocol limit")
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise WireProtocolError("JSON object keys must be strings")
        return 1 + sum(_walk_limits(item, depth=depth + 1) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_walk_limits(item, depth=depth + 1) for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return 1
    if isinstance(value, float) and isfinite(value):
        return 1
    raise WireProtocolError(f"unsupported JSON value {type(value).__name__}")


def _observe(value: Observation) -> dict[str, Any]:
    if not 0 <= value.tick <= MAX_PUBLIC_TICK:
        raise WireProtocolError("observation.tick lies outside the public range")
    raw = value.pixels.tobytes(order="C")
    if len(raw) > MAX_PIXEL_BYTES:
        raise WireProtocolError("observation exceeds the pixel-byte limit")
    return {
        "shape": list(value.shape),
        "pixels_b64": base64.b64encode(raw).decode("ascii"),
        "tick": value.tick,
        "terminal": value.terminal,
    }


def _decode_observation(value: object) -> Observation:
    obj = _exact(value, {"shape", "pixels_b64", "tick", "terminal"}, "observation")
    shape_value = obj["shape"]
    if not isinstance(shape_value, list) or len(shape_value) != 3:
        raise WireProtocolError("observation.shape must be a three-integer list")
    shape = tuple(_strict_int(item, "observation.shape") for item in shape_value)
    if min(shape) <= 0 or shape[2] != 3:
        raise WireProtocolError("observation.shape must be positive RGB dimensions")
    encoded = obj["pixels_b64"]
    if not isinstance(encoded, str):
        raise WireProtocolError("observation.pixels_b64 must be a string")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise WireProtocolError("observation.pixels_b64 is not canonical base64") from exc
    expected = prod(shape)
    if expected > MAX_PIXEL_BYTES or len(raw) != expected:
        raise WireProtocolError("observation pixel payload length does not match shape")
    terminal = obj["terminal"]
    if not isinstance(terminal, bool):
        raise WireProtocolError("observation.terminal must be boolean")
    pixels = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    return Observation(
        pixels,
        _strict_int(
            obj["tick"],
            "observation.tick",
            minimum=0,
            maximum=MAX_PUBLIC_TICK,
        ),
        terminal,
    )


def _action(value: Action | None) -> dict[str, Any] | None:
    if value is None:
        return None
    return {"code": value.code, "target": list(value.target), "vector": list(value.vector)}


def _decode_action(value: object, *, optional: bool = False) -> Action | None:
    if value is None and optional:
        return None
    obj = _exact(value, {"code", "target", "vector"}, "action")
    target = obj["target"]
    vector = obj["vector"]
    if not isinstance(target, list) or len(target) != 2:
        raise WireProtocolError("action.target must be a two-integer list")
    if not isinstance(vector, list) or len(vector) != 2:
        raise WireProtocolError("action.vector must be a two-integer list")
    return Action(
        _strict_int(obj["code"], "action.code"),
        tuple(_strict_int(item, "action.target") for item in target),
        tuple(_strict_int(item, "action.vector") for item in vector),
    )


def _utterance(value: Utterance | None) -> dict[str, Any] | None:
    return None if value is None else {"tokens": list(value.tokens)}


def _decode_utterance(value: object, *, optional: bool = False) -> Utterance | None:
    if value is None and optional:
        return None
    obj = _exact(value, {"tokens"}, "utterance")
    tokens = obj["tokens"]
    if not isinstance(tokens, list):
        raise WireProtocolError("utterance.tokens must be a list")
    return Utterance(tuple(_strict_int(token, "utterance.token") for token in tokens))


def _transition(value: PublicTransition) -> dict[str, Any]:
    return {
        "before": _observe(value.before),
        "action": _action(value.action),
        "after": _observe(value.after),
        "scalar_feedback": value.scalar_feedback,
    }


def _decode_transition(value: object) -> PublicTransition:
    obj = _exact(
        value,
        {"before", "action", "after", "scalar_feedback"},
        "public_transition",
    )
    action = _decode_action(obj["action"])
    assert action is not None
    feedback = obj["scalar_feedback"]
    return PublicTransition(
        _decode_observation(obj["before"]),
        action,
        _decode_observation(obj["after"]),
        None if feedback is None else _strict_float(feedback, "scalar_feedback"),
    )


def _trace(value: PublicTrace) -> dict[str, Any]:
    return {
        "initial": _observe(value.initial),
        "transitions": [_transition(item) for item in value.transitions],
    }


def _decode_trace(value: object) -> PublicTrace:
    obj = _exact(value, {"initial", "transitions"}, "public_trace")
    transitions = obj["transitions"]
    if not isinstance(transitions, list):
        raise WireProtocolError("public_trace.transitions must be a list")
    return PublicTrace(
        _decode_observation(obj["initial"]),
        tuple(_decode_transition(item) for item in transitions),
    )


def _payload(value: object) -> tuple[str, dict[str, Any]]:
    if isinstance(value, SessionManifest):
        return "session_manifest", asdict(value)
    if isinstance(value, Utterance):
        payload = _utterance(value)
        assert payload is not None
        return "utterance", payload
    if isinstance(value, PublicTransition):
        return "public_transition", _transition(value)
    if isinstance(value, PublicTrace):
        return "public_trace", _trace(value)
    if isinstance(value, PublicTurn):
        return "public_turn", {
            "turn_id": value.turn_id,
            "phase": value.phase.value,
            "observation": _observe(value.observation),
            "utterance": _utterance(value.utterance),
            "ostensive_pixel_cue": (
                None if value.ostensive_pixel_cue is None else list(value.ostensive_pixel_cue)
            ),
            "scalar_feedback": value.scalar_feedback,
            "remaining_cost": value.remaining_cost,
        }
    if isinstance(value, ExperimentDecision):
        return "experiment_decision", {
            "action": _action(value.action),
            "unknown_probability": value.unknown_probability,
        }
    if isinstance(value, DescriptionDecision):
        return "description_decision", {
            "utterance": _utterance(value.utterance),
            "unknown_probability": value.unknown_probability,
        }
    if isinstance(value, ActionDecision):
        return "action_decision", {
            "action": _action(value.action),
            "unknown_probability": value.unknown_probability,
        }
    if isinstance(value, BeliefDecision):
        return "belief_decision", {
            "candidate_probabilities": [list(item) for item in value.candidate_probabilities],
            "unknown_probability": value.unknown_probability,
        }
    raise TypeError(f"unsupported v1 wire value: {type(value).__name__}")


def encode_message(value: object, *, max_message_bytes: int = MAX_MESSAGE_BYTES) -> bytes:
    """Encode one allowlisted public value into canonical UTF-8 JSON."""

    message_type, payload = _payload(value)
    envelope = {"protocol": PROTOCOL_VERSION, "type": message_type, "payload": payload}
    if _walk_limits(envelope) > MAX_JSON_NODES:
        raise WireProtocolError("message exceeds the JSON-node limit")
    encoded = json.dumps(
        envelope,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > max_message_bytes:
        raise WireProtocolError("encoded message exceeds max_message_bytes")
    return encoded


def _reject_constant(value: str) -> None:
    raise WireProtocolError(f"non-finite JSON constant is forbidden: {value}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise WireProtocolError(f"duplicate JSON key is forbidden: {key!r}")
        result[key] = value
    return result


def _preflight_json(data: bytes) -> None:
    """Bound structural nesting before the recursive stdlib parser runs."""

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


def decode_message(data: bytes, *, max_message_bytes: int = MAX_MESSAGE_BYTES) -> object:
    """Decode one exact-schema v1 JSON message; unknown types fail closed."""

    if not isinstance(data, bytes):
        raise TypeError("wire messages must be bytes")
    if len(data) > max_message_bytes:
        raise WireProtocolError("wire message exceeds max_message_bytes")
    _preflight_json(data)
    try:
        decoded = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_constant,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError, OverflowError) as exc:
        if isinstance(exc, WireProtocolError):
            raise
        raise WireProtocolError("wire message is not strict UTF-8 JSON") from exc
    try:
        if _walk_limits(decoded) > MAX_JSON_NODES:
            raise WireProtocolError("message exceeds the JSON-node limit")
    except (RecursionError, OverflowError) as exc:
        raise WireProtocolError("wire message exceeds structural limits") from exc
    envelope = _exact(decoded, {"protocol", "type", "payload"}, "envelope")
    if envelope["protocol"] != PROTOCOL_VERSION:
        raise WireProtocolError("unsupported protocol version")
    message_type = envelope["type"]
    if not isinstance(message_type, str):
        raise WireProtocolError("envelope.type must be a string")
    payload = envelope["payload"]

    try:
        if message_type == "session_manifest":
            obj = _exact(
                payload,
                {
                    "protocol_version",
                    "sensor_schema",
                    "action_schema",
                    "support_episode_budget",
                    "intervention_cost_budget",
                    "query_budget",
                },
                "session_manifest",
            )
            return SessionManifest(
                protocol_version=obj["protocol_version"],
                sensor_schema=obj["sensor_schema"],
                action_schema=obj["action_schema"],
                support_episode_budget=_strict_int(
                    obj["support_episode_budget"], "support_episode_budget"
                ),
                intervention_cost_budget=_strict_float(
                    obj["intervention_cost_budget"], "intervention_cost_budget"
                ),
                query_budget=_strict_int(obj["query_budget"], "query_budget"),
            )
        if message_type == "utterance":
            result = _decode_utterance(payload)
            assert result is not None
            return result
        if message_type == "public_transition":
            return _decode_transition(payload)
        if message_type == "public_trace":
            return _decode_trace(payload)
        if message_type == "public_turn":
            obj = _exact(
                payload,
                {
                    "turn_id",
                    "phase",
                    "observation",
                    "utterance",
                    "ostensive_pixel_cue",
                    "scalar_feedback",
                    "remaining_cost",
                },
                "public_turn",
            )
            cue = obj["ostensive_pixel_cue"]
            if cue is not None:
                if not isinstance(cue, list) or len(cue) != 4:
                    raise WireProtocolError("ostensive_pixel_cue must be a four-integer list")
                cue = tuple(_strict_int(item, "ostensive_pixel_cue") for item in cue)
            feedback = obj["scalar_feedback"]
            return PublicTurn(
                _strict_int(obj["turn_id"], "turn_id"),
                SessionPhase(obj["phase"]),
                _decode_observation(obj["observation"]),
                _decode_utterance(obj["utterance"], optional=True),
                cue,
                None if feedback is None else _strict_float(feedback, "scalar_feedback"),
                _strict_float(obj["remaining_cost"], "remaining_cost"),
            )
        if message_type in {"experiment_decision", "action_decision"}:
            obj = _exact(payload, {"action", "unknown_probability"}, message_type)
            arguments = (
                _decode_action(obj["action"], optional=True),
                _strict_float(obj["unknown_probability"], "unknown_probability"),
            )
            return (
                ExperimentDecision(*arguments)
                if message_type == "experiment_decision"
                else ActionDecision(*arguments)
            )
        if message_type == "description_decision":
            obj = _exact(
                payload,
                {"utterance", "unknown_probability"},
                "description_decision",
            )
            return DescriptionDecision(
                _decode_utterance(obj["utterance"], optional=True),
                _strict_float(obj["unknown_probability"], "unknown_probability"),
            )
        if message_type == "belief_decision":
            obj = _exact(
                payload,
                {"candidate_probabilities", "unknown_probability"},
                "belief_decision",
            )
            pairs = obj["candidate_probabilities"]
            if not isinstance(pairs, list):
                raise WireProtocolError("candidate_probabilities must be a list")
            converted: list[tuple[int, float]] = []
            for pair in pairs:
                if not isinstance(pair, list) or len(pair) != 2:
                    raise WireProtocolError("candidate probability rows must have length two")
                converted.append(
                    (
                        _strict_int(pair[0], "candidate"),
                        _strict_float(pair[1], "candidate_probability"),
                    )
                )
            return BeliefDecision(
                tuple(converted),
                _strict_float(obj["unknown_probability"], "unknown_probability"),
            )
    except (TypeError, ValueError) as exc:
        if isinstance(exc, WireProtocolError):
            raise
        raise WireProtocolError(f"invalid {message_type} payload: {exc}") from exc
    raise WireProtocolError(f"unsupported message type: {message_type!r}")


def public_schema_manifest() -> dict[str, Any]:
    """Machine-readable exact allowlist for certificate manifests."""

    return {
        "protocol": PROTOCOL_VERSION,
        "top_level_types": [
            "action_decision",
            "belief_decision",
            "description_decision",
            "experiment_decision",
            "public_trace",
            "public_transition",
            "public_turn",
            "session_manifest",
            "utterance",
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
            "evaluator-enum",
            "latent-id",
            "oracle-handle",
        ],
    }


__all__ = [
    "MAX_JSON_DEPTH",
    "MAX_JSON_NODES",
    "MAX_MESSAGE_BYTES",
    "MAX_PIXEL_BYTES",
    "MAX_WIRE_INTEGER",
    "WireProtocolError",
    "decode_message",
    "encode_message",
    "public_schema_manifest",
]
