"""Process/capability isolation for executing an untrusted learner.

This module deliberately claims less than an operating-system sandbox.  The
learner runs in a fresh ``multiprocessing`` ``spawn`` process and never receives
the evaluator's :class:`Microworld`, oracle, engine, or a bound method owned by
any of them.  Its only environment capability is :class:`LearnerClient`, a
bytes/JSON RPC proxy.  A learner can still use any filesystem, network, or OS
capability granted to that process; deployments needing those restrictions
must add an OS sandbox around this boundary.

Only explicit JSON envelopes and serialized ``Action``, ``Observation``,
``Transition``, ``Trajectory`` and ``AgentManifest`` values cross the pipe.
In particular, the evaluator never calls ``Connection.recv`` on child data, so
no child-controlled pickle is deserialized in the privileged parent.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib import import_module
from math import isfinite
from multiprocessing import get_context
from multiprocessing.connection import Connection
from types import TracebackType
from typing import Any, TypeAlias
import base64
import json
import time
import traceback

import numpy as np

from .contracts import Action, AgentManifest, Observation, Trajectory, Transition
from .protocol import AgentEnvironment


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

RPC_VERSION = 1
DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_REQUESTS = 10_000
DEFAULT_MAX_MESSAGE_BYTES = 16 * 1024 * 1024
_SHUTDOWN_GRACE_SECONDS = 0.25
_RPC_COMMANDS = frozenset(
    {"manifest", "observe", "reset", "step", "trajectory", "complete", "failed"}
)
_ENVIRONMENT_COMMANDS = frozenset({"manifest", "observe", "reset", "step", "trajectory"})


class IsolationError(RuntimeError):
    """Base error raised by the process isolation boundary."""


class LearnerProtocolError(IsolationError):
    """The child violated the bounded RPC protocol."""


class LearnerExecutionError(IsolationError):
    """The learner callable raised an exception in the child process."""

    def __init__(self, remote_type: str, message: str, remote_traceback: str) -> None:
        detail = f"{remote_type}: {message}" if message else remote_type
        super().__init__(f"learner failed: {detail}")
        self.remote_type = remote_type
        self.remote_message = message
        self.remote_traceback = remote_traceback


class LearnerTimeoutError(TimeoutError):
    """The learner did not finish within its evaluator-selected deadline."""


class RemoteEnvironmentError(RuntimeError):
    """A valid learner RPC reached the environment but the operation failed."""

    def __init__(self, remote_type: str, message: str) -> None:
        detail = f"{remote_type}: {message}" if message else remote_type
        super().__init__(f"environment rejected RPC: {detail}")
        self.remote_type = remote_type
        self.remote_message = message


@dataclass(frozen=True, slots=True)
class LearnerRunResult:
    """JSON-safe learner result plus auditable process/RPC metadata."""

    value: JSONValue
    request_count: int
    exitcode: int


def _strict_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise LearnerProtocolError(f"{field} must be an integer")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], field: str) -> None:
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise LearnerProtocolError(f"{field} keys differ; missing={missing}, extra={extra}")


def _json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LearnerProtocolError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> None:
    raise LearnerProtocolError(f"non-finite JSON constant is forbidden: {value}")


def _validate_json_value(value: object, *, depth: int = 0) -> JSONValue:
    if depth > 64:
        raise LearnerProtocolError("JSON value exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise LearnerProtocolError("JSON numbers must be finite")
        return value
    if isinstance(value, list):
        return [_validate_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise LearnerProtocolError("JSON object keys must be strings")
        return {
            key: _validate_json_value(item, depth=depth + 1) for key, item in value.items()
        }
    raise LearnerProtocolError(f"value is not JSON-serializable: {type(value).__name__}")


def _decode_json(payload: bytes) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8", errors="strict")
        value = json.loads(
            text,
            object_pairs_hook=_json_object,
            parse_constant=_reject_json_constant,
        )
    except LearnerProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as error:
        raise LearnerProtocolError("message is not strict UTF-8 JSON") from error
    validated = _validate_json_value(value)
    if not isinstance(validated, dict):
        raise LearnerProtocolError("RPC envelope must be a JSON object")
    return validated


def _encode_json(value: Mapping[str, object], max_bytes: int) -> bytes:
    validated = _validate_json_value(dict(value))
    encoded = json.dumps(
        validated,
        allow_nan=False,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if len(encoded) > max_bytes:
        raise LearnerProtocolError(
            f"RPC message is {len(encoded)} bytes; limit is {max_bytes} bytes"
        )
    return encoded


def _send_json(connection: Connection, value: Mapping[str, object], max_bytes: int) -> None:
    connection.send_bytes(_encode_json(value, max_bytes))


def _receive_json(connection: Connection, max_bytes: int) -> dict[str, Any]:
    try:
        payload = connection.recv_bytes(maxlength=max_bytes)
    except OSError as error:
        raise LearnerProtocolError("RPC frame exceeded the byte limit") from error
    return _decode_json(payload)


def _encode_action(action: Action) -> dict[str, JSONValue]:
    return {
        "type": "action",
        "code": action.code,
        "target": [action.target[0], action.target[1]],
        "vector": [action.vector[0], action.vector[1]],
    }


def _decode_pair(value: object, field: str) -> tuple[int, int]:
    if not isinstance(value, list) or len(value) != 2:
        raise LearnerProtocolError(f"{field} must be a two-item JSON array")
    return (_strict_int(value[0], f"{field}[0]"), _strict_int(value[1], f"{field}[1]"))


def _decode_action(value: object) -> Action:
    if not isinstance(value, dict):
        raise LearnerProtocolError("action record must be an object")
    _exact_keys(value, {"type", "code", "target", "vector"}, "action")
    if value["type"] != "action":
        raise LearnerProtocolError("record type must be 'action'")
    try:
        return Action(
            _strict_int(value["code"], "action.code"),
            _decode_pair(value["target"], "action.target"),
            _decode_pair(value["vector"], "action.vector"),
        )
    except (TypeError, ValueError) as error:
        raise LearnerProtocolError(str(error)) from error


def _encode_observation(observation: Observation) -> dict[str, JSONValue]:
    return {
        "type": "observation",
        "pixels": base64.b64encode(observation.pixels.tobytes(order="C")).decode("ascii"),
        "shape": list(observation.shape),
        "tick": observation.tick,
        "terminal": observation.terminal,
    }


def _decode_observation(value: object) -> Observation:
    if not isinstance(value, dict):
        raise LearnerProtocolError("observation record must be an object")
    _exact_keys(value, {"type", "pixels", "shape", "tick", "terminal"}, "observation")
    if value["type"] != "observation":
        raise LearnerProtocolError("record type must be 'observation'")
    shape_value = value["shape"]
    if not isinstance(shape_value, list) or len(shape_value) != 3:
        raise LearnerProtocolError("observation.shape must contain three dimensions")
    shape = tuple(_strict_int(item, "observation.shape") for item in shape_value)
    if min(shape) <= 0 or shape[2] != 3:
        raise LearnerProtocolError("observation.shape must be positive RGB dimensions")
    pixels_value = value["pixels"]
    if not isinstance(pixels_value, str):
        raise LearnerProtocolError("observation.pixels must be base64 text")
    try:
        raw = base64.b64decode(pixels_value, validate=True)
    except (ValueError, base64.binascii.Error) as error:
        raise LearnerProtocolError("observation.pixels is not canonical base64") from error
    expected = shape[0] * shape[1] * shape[2]
    if len(raw) != expected:
        raise LearnerProtocolError("observation pixel byte count does not match shape")
    terminal = value["terminal"]
    if not isinstance(terminal, bool):
        raise LearnerProtocolError("observation.terminal must be boolean")
    frame = np.frombuffer(raw, dtype=np.uint8).reshape(shape)
    try:
        return Observation(frame, _strict_int(value["tick"], "observation.tick"), terminal)
    except (TypeError, ValueError) as error:
        raise LearnerProtocolError(str(error)) from error


def _encode_transition(transition: Transition) -> dict[str, JSONValue]:
    return {
        "type": "transition",
        "before": _encode_observation(transition.before),
        "action": _encode_action(transition.action),
        "after": _encode_observation(transition.after),
        "outcome_code": transition.outcome_code,
    }


def _decode_transition(value: object) -> Transition:
    if not isinstance(value, dict):
        raise LearnerProtocolError("transition record must be an object")
    _exact_keys(
        value,
        {"type", "before", "action", "after", "outcome_code"},
        "transition",
    )
    if value["type"] != "transition":
        raise LearnerProtocolError("record type must be 'transition'")
    try:
        return Transition(
            _decode_observation(value["before"]),
            _decode_action(value["action"]),
            _decode_observation(value["after"]),
            _strict_int(value["outcome_code"], "transition.outcome_code"),
        )
    except (TypeError, ValueError) as error:
        raise LearnerProtocolError(str(error)) from error


def _encode_manifest(manifest: AgentManifest) -> dict[str, JSONValue]:
    return {
        "type": "manifest",
        "observation_shape": list(manifest.observation_shape),
        "action_codes": list(manifest.action_codes),
        "symbol_codes": list(manifest.symbol_codes),
        "motor_vectors": [list(vector) for vector in manifest.motor_vectors],
        "max_steps": manifest.max_steps,
    }


def _decode_int_list(value: object, field: str) -> tuple[int, ...]:
    if not isinstance(value, list):
        raise LearnerProtocolError(f"{field} must be a JSON array")
    return tuple(_strict_int(item, field) for item in value)


def _decode_manifest(value: object) -> AgentManifest:
    if not isinstance(value, dict):
        raise LearnerProtocolError("manifest record must be an object")
    _exact_keys(
        value,
        {
            "type",
            "observation_shape",
            "action_codes",
            "symbol_codes",
            "motor_vectors",
            "max_steps",
        },
        "manifest",
    )
    if value["type"] != "manifest":
        raise LearnerProtocolError("record type must be 'manifest'")
    vectors = value["motor_vectors"]
    if not isinstance(vectors, list):
        raise LearnerProtocolError("manifest.motor_vectors must be a JSON array")
    try:
        return AgentManifest(
            _decode_int_list(value["observation_shape"], "manifest.observation_shape"),
            _decode_int_list(value["action_codes"], "manifest.action_codes"),
            _decode_int_list(value["symbol_codes"], "manifest.symbol_codes"),
            tuple(_decode_pair(vector, "manifest.motor_vectors") for vector in vectors),
            _strict_int(value["max_steps"], "manifest.max_steps"),
        )
    except (TypeError, ValueError) as error:
        raise LearnerProtocolError(str(error)) from error


def _encode_trajectory(trajectory: Trajectory) -> dict[str, JSONValue]:
    return {
        "type": "trajectory",
        "initial": _encode_observation(trajectory.initial),
        "transitions": [_encode_transition(item) for item in trajectory.transitions],
    }


def _decode_trajectory(value: object) -> Trajectory:
    if not isinstance(value, dict):
        raise LearnerProtocolError("trajectory record must be an object")
    _exact_keys(value, {"type", "initial", "transitions"}, "trajectory")
    if value["type"] != "trajectory":
        raise LearnerProtocolError("record type must be 'trajectory'")
    transitions = value["transitions"]
    if not isinstance(transitions, list):
        raise LearnerProtocolError("trajectory.transitions must be a JSON array")
    try:
        return Trajectory(
            _decode_observation(value["initial"]),
            tuple(_decode_transition(item) for item in transitions),
        )
    except (TypeError, ValueError) as error:
        raise LearnerProtocolError(str(error)) from error


def _request_envelope(request_id: int, command: str, payload: JSONValue) -> dict[str, object]:
    return {"version": RPC_VERSION, "id": request_id, "command": command, "payload": payload}


def _parse_request(value: Mapping[str, object]) -> tuple[int, str, dict[str, JSONValue]]:
    _exact_keys(value, {"version", "id", "command", "payload"}, "request")
    if _strict_int(value["version"], "request.version") != RPC_VERSION:
        raise LearnerProtocolError("unsupported RPC version")
    request_id = _strict_int(value["id"], "request.id")
    if request_id < 0:
        raise LearnerProtocolError("request.id must be non-negative")
    command = value["command"]
    if not isinstance(command, str) or command not in _RPC_COMMANDS:
        raise LearnerProtocolError("request command is not allowlisted")
    payload = value["payload"]
    if not isinstance(payload, dict):
        raise LearnerProtocolError("request.payload must be an object")
    return request_id, command, payload


def _success_response(request_id: int, record: JSONValue = None) -> dict[str, object]:
    return {"version": RPC_VERSION, "id": request_id, "ok": True, "record": record}


def _error_response(request_id: int, error: Exception) -> dict[str, object]:
    return {
        "version": RPC_VERSION,
        "id": request_id,
        "ok": False,
        "error": {"type": type(error).__name__, "message": str(error)},
    }


class LearnerClient:
    """Child-only proxy implementing the serialized learner environment API.

    Bound methods on this object have ``LearnerClient`` as ``.__self__``.  Its
    only retained capability is a pipe endpoint speaking the strict allowlist
    above; no object reference in this process points at the evaluator engine.
    """

    __slots__ = (
        "__connection",
        "__manifest",
        "__max_message_bytes",
        "__request_id",
        "__finished",
    )

    def __init__(self, connection: Connection, max_message_bytes: int) -> None:
        self.__connection = connection
        self.__max_message_bytes = max_message_bytes
        self.__request_id = 0
        self.__finished = False
        self.__manifest = _decode_manifest(self.__request("manifest", {}))

    @property
    def manifest(self) -> AgentManifest:
        return self.__manifest

    @property
    def action_codes(self) -> tuple[int, ...]:
        return self.manifest.action_codes

    @property
    def symbol_codes(self) -> tuple[int, ...]:
        return self.manifest.symbol_codes

    def reset(self) -> Observation:
        return _decode_observation(self.__request("reset", {}))

    def observe(self) -> Observation:
        return _decode_observation(self.__request("observe", {}))

    def step(
        self,
        action: Action | int,
        target: tuple[int, int] | None = None,
        vector: tuple[int, int] = (0, 0),
    ) -> Transition:
        if isinstance(action, Action):
            if target is not None or vector != (0, 0):
                raise TypeError("target/vector must be omitted when passing an Action")
            record = action
        else:
            if target is None:
                raise TypeError("target is required when passing an opaque action code")
            record = Action(action, target, vector)
        return _decode_transition(self.__request("step", {"action": _encode_action(record)}))

    def trajectory(self) -> Trajectory:
        return _decode_trajectory(self.__request("trajectory", {}))

    def __request(self, command: str, payload: JSONValue) -> JSONValue:
        if self.__finished:
            raise RuntimeError("learner RPC session is already finished")
        request_id = self.__request_id
        self.__request_id += 1
        _send_json(
            self.__connection,
            _request_envelope(request_id, command, payload),
            self.__max_message_bytes,
        )
        response = _receive_json(self.__connection, self.__max_message_bytes)
        if response.get("ok") is True:
            _exact_keys(response, {"version", "id", "ok", "record"}, "success response")
        else:
            _exact_keys(response, {"version", "id", "ok", "error"}, "error response")
        version = _strict_int(response["version"], "response.version")
        response_id = _strict_int(response["id"], "response.id")
        if version != RPC_VERSION or response_id != request_id:
            raise LearnerProtocolError("RPC response does not match request")
        if response["ok"] is True:
            return _validate_json_value(response["record"])
        if response["ok"] is not False:
            raise LearnerProtocolError("response.ok must be boolean")
        error = response["error"]
        if not isinstance(error, dict):
            raise LearnerProtocolError("response.error must be an object")
        _exact_keys(error, {"type", "message"}, "response.error")
        remote_type = error["type"]
        message = error["message"]
        if not isinstance(remote_type, str) or not isinstance(message, str):
            raise LearnerProtocolError("response error fields must be strings")
        raise RemoteEnvironmentError(remote_type, message)

    def _finish(self, value: JSONValue) -> None:
        checked = _validate_json_value(value)
        self.__request("complete", {"result": checked})
        self.__finished = True

    def _fail(self, error: BaseException) -> None:
        details = {
            "type": type(error).__name__,
            "message": str(error)[:4_096],
            "traceback": "".join(traceback.format_exception(error))[-16_384:],
        }
        try:
            self.__request("failed", details)
        finally:
            self.__finished = True


def _resolve_entrypoint(reference: str) -> Callable[[LearnerClient], JSONValue]:
    _resolve_entrypoint_syntax(reference)
    module_name, attribute_path = reference.split(":", 1)
    candidate: object = import_module(module_name)
    for part in attribute_path.split("."):
        candidate = getattr(candidate, part)
    if not callable(candidate):
        raise TypeError("learner entrypoint is not callable")
    return candidate  # type: ignore[return-value]


def _learner_child_main(
    connection: Connection,
    entrypoint: str,
    max_message_bytes: int,
) -> None:
    client: LearnerClient | None = None
    try:
        learner = _resolve_entrypoint(entrypoint)
        client = LearnerClient(connection, max_message_bytes)
        client._finish(learner(client))
    except BaseException as error:
        try:
            if client is not None:
                client._fail(error)
            else:
                _send_json(
                    connection,
                    _request_envelope(
                        0,
                        "failed",
                        {
                            "type": type(error).__name__,
                            "message": str(error)[:4_096],
                            "traceback": "".join(traceback.format_exception(error))[-16_384:],
                        },
                    ),
                    max_message_bytes,
                )
                _receive_json(connection, max_message_bytes)
        except BaseException:
            pass
    finally:
        connection.close()


class LearnerProcess:
    """Evaluator-side owner of one spawn process and one bounded RPC session."""

    __slots__ = (
        "_environment",
        "_entrypoint",
        "_timeout",
        "_max_requests",
        "_max_message_bytes",
        "_process",
        "_connection",
        "_started_at",
        "_closed",
        "_ran",
        "_exitcode",
    )

    def __init__(
        self,
        environment: AgentEnvironment,
        entrypoint: str,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
    ) -> None:
        if not isinstance(environment, AgentEnvironment):
            raise TypeError("environment must satisfy AgentEnvironment")
        _resolve_entrypoint_syntax(entrypoint)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be numeric")
        if not isfinite(float(timeout)) or timeout <= 0:
            raise ValueError("timeout must be finite and positive")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int):
            raise TypeError("max_requests must be an integer")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        if isinstance(max_message_bytes, bool) or not isinstance(max_message_bytes, int):
            raise TypeError("max_message_bytes must be an integer")
        if max_message_bytes < 1_024:
            raise ValueError("max_message_bytes must be at least 1024")
        self._environment = environment
        self._entrypoint = entrypoint
        self._timeout = float(timeout)
        self._max_requests = max_requests
        self._max_message_bytes = max_message_bytes
        self._process = None
        self._connection: Connection | None = None
        self._started_at: float | None = None
        self._closed = False
        self._ran = False
        self._exitcode: int | None = None

    def __enter__(self) -> "LearnerProcess":
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        self.close()

    @property
    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("learner process is closed")
        if self._process is not None:
            return
        context = get_context("spawn")
        parent_connection, child_connection = context.Pipe(duplex=True)
        process = context.Process(
            target=_learner_child_main,
            args=(child_connection, self._entrypoint, self._max_message_bytes),
            name="grounding-kernel-learner",
            daemon=True,
        )
        self._connection = parent_connection
        self._process = process
        self._started_at = time.monotonic()
        try:
            process.start()
        except BaseException:
            parent_connection.close()
            child_connection.close()
            self._connection = None
            self._process = None
            self._started_at = None
            raise
        finally:
            child_connection.close()

    def run(self) -> LearnerRunResult:
        if self._ran:
            raise RuntimeError("learner process can only be run once")
        self._ran = True
        self.start()
        assert self._connection is not None
        assert self._process is not None
        assert self._started_at is not None
        deadline = self._started_at + self._timeout
        request_count = 0
        result: JSONValue = None
        learner_error: LearnerExecutionError | None = None
        completed = False
        try:
            while not completed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise LearnerTimeoutError(
                        f"learner exceeded timeout of {self._timeout:.3f} seconds"
                    )
                if not self._connection.poll(min(remaining, 0.05)):
                    if not self._process.is_alive():
                        raise LearnerExecutionError(
                            "ChildProcessExit",
                            f"learner exited with code {self._process.exitcode}",
                            "",
                        )
                    continue
                try:
                    request = _receive_json(self._connection, self._max_message_bytes)
                except EOFError as error:
                    raise LearnerExecutionError(
                        "ChildProcessExit", "learner closed RPC without a result", ""
                    ) from error
                request_id, command, payload = _parse_request(request)
                if request_id != request_count:
                    raise LearnerProtocolError(
                        f"request.id must be the next sequence number ({request_count})"
                    )
                request_count += 1
                if request_count > self._max_requests:
                    raise LearnerProtocolError(
                        f"learner exceeded request limit of {self._max_requests}"
                    )

                if command in _ENVIRONMENT_COMMANDS:
                    response = self._dispatch_environment(request_id, command, payload)
                    _send_json(self._connection, response, self._max_message_bytes)
                    continue
                if command == "complete":
                    _exact_keys(payload, {"result"}, "complete payload")
                    result = _validate_json_value(payload["result"])
                    _send_json(
                        self._connection,
                        _success_response(request_id),
                        self._max_message_bytes,
                    )
                    completed = True
                    continue
                _exact_keys(payload, {"type", "message", "traceback"}, "failed payload")
                remote_type = payload["type"]
                message = payload["message"]
                remote_traceback = payload["traceback"]
                if not all(isinstance(item, str) for item in (remote_type, message, remote_traceback)):
                    raise LearnerProtocolError("failed payload fields must be strings")
                learner_error = LearnerExecutionError(remote_type, message, remote_traceback)
                _send_json(
                    self._connection,
                    _success_response(request_id),
                    self._max_message_bytes,
                )
                completed = True

            self._join_or_stop(deadline)
            if learner_error is not None:
                raise learner_error
            exitcode = self._exitcode
            if exitcode != 0:
                raise LearnerExecutionError(
                    "ChildProcessExit", f"learner exited with code {exitcode}", ""
                )
            return LearnerRunResult(result, request_count, exitcode)
        finally:
            self.close()

    def _dispatch_environment(
        self,
        request_id: int,
        command: str,
        payload: Mapping[str, JSONValue],
    ) -> dict[str, object]:
        try:
            if command == "manifest":
                _exact_keys(payload, set(), "manifest payload")
                record: JSONValue = _encode_manifest(self._environment.manifest)
            elif command == "observe":
                _exact_keys(payload, set(), "observe payload")
                record = _encode_observation(self._environment.observe())
            elif command == "reset":
                _exact_keys(payload, set(), "reset payload")
                record = _encode_observation(self._environment.reset())
            elif command == "step":
                _exact_keys(payload, {"action"}, "step payload")
                record = _encode_transition(self._environment.step(_decode_action(payload["action"])))
            elif command == "trajectory":
                _exact_keys(payload, set(), "trajectory payload")
                record = _encode_trajectory(self._environment.trajectory())
            else:  # pragma: no cover - caller already checks the allowlist
                raise LearnerProtocolError("environment command is not allowlisted")
            return _success_response(request_id, record)
        except LearnerProtocolError:
            raise
        except Exception as error:
            return _error_response(request_id, error)

    def _join_or_stop(self, deadline: float) -> None:
        assert self._process is not None
        remaining = max(0.0, deadline - time.monotonic())
        self._process.join(min(_SHUTDOWN_GRACE_SECONDS, remaining))
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(_SHUTDOWN_GRACE_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(_SHUTDOWN_GRACE_SECONDS)
        self._exitcode = self._process.exitcode

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._process is not None:
            self._process.join(_SHUTDOWN_GRACE_SECONDS)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(_SHUTDOWN_GRACE_SECONDS)
            if self._process.is_alive():
                self._process.kill()
                self._process.join(_SHUTDOWN_GRACE_SECONDS)
            self._exitcode = self._process.exitcode


def _resolve_entrypoint_syntax(reference: object) -> None:
    if not isinstance(reference, str):
        raise TypeError("learner entrypoint must be a string")
    if reference.count(":") != 1:
        raise ValueError("learner entrypoint must have form 'module:callable'")
    module_name, attribute_path = reference.split(":", 1)
    if not module_name or not attribute_path:
        raise ValueError("learner entrypoint must have form 'module:callable'")
    if any(not part.isidentifier() for part in module_name.split(".")):
        raise ValueError("learner module must be an absolute dotted identifier")
    if any(not part.isidentifier() or part == "<locals>" for part in attribute_path.split(".")):
        raise ValueError("learner callable must be a dotted identifier")


def run_isolated_learner(
    environment: AgentEnvironment,
    entrypoint: str,
    *,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_requests: int = DEFAULT_MAX_REQUESTS,
    max_message_bytes: int = DEFAULT_MAX_MESSAGE_BYTES,
) -> LearnerRunResult:
    """Run one importable learner callable behind the spawn/JSON boundary."""

    with LearnerProcess(
        environment,
        entrypoint,
        timeout=timeout,
        max_requests=max_requests,
        max_message_bytes=max_message_bytes,
    ) as process:
        return process.run()
