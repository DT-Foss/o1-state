"""Persistent process boundary for the parallel Grade-3 candidate protocol.

This is the trusted-reference boundary.  It recursively commits the candidate
package and the complete Python SDK before the codebook is constructed, then
uses one spawned process and checks the honest-reference checkpoint around
every sealed query.  It is intentionally not the Grade-4 VM sandbox.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, invalidate_caches
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any
import json
import math
import os
import stat
import sys
import time
import traceback

from .certificates import manifest_hash
from .v1_contracts import BeliefDecision, DescriptionDecision
from .v1_grade3_contracts import (
    CausalSupportRecord,
    Grade3Grounder,
    Grade3SessionManifest,
    MotorDecision,
    MotorQuery,
    OstensiveSupportRecord,
    ProbeDecision,
    ProbeOffer,
    ProbeResult,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)
from .v1_grade3_wire import decode_grade3_message, encode_grade3_message
from .v1_isolation import (
    CandidateExecutionError,
    CandidateIsolationError,
    CandidateProtocolError,
    CandidateTimeoutError,
)
from .v1_wire import MAX_JSON_DEPTH, MAX_JSON_NODES, MAX_MESSAGE_BYTES


GRADE3_ISOLATION_PROTOCOL = "grounding-grade3-candidate-rpc/1"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_REQUESTS = 50_000
MAX_PACKAGE_FILES = 4_096
MAX_PACKAGE_BYTES = 256 * 1024 * 1024
_JOIN_GRACE_SECONDS = 0.25
_QUERY_COMMANDS = frozenset({"motor", "trace_belief", "describe"})
_COMMANDS = frozenset(
    {
        "begin",
        "observe_support",
        "choose_probe",
        "observe_probe",
        "freeze",
        "motor",
        "trace_belief",
        "describe",
        "checkpoint",
        "close",
    }
)


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise CandidateProtocolError(f"{field} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise CandidateProtocolError(
            f"{field} must be a lowercase SHA-256 digest"
        ) from exc
    return value


def _stable_read(path: Path) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CandidateProtocolError(f"artifact is not a regular file: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    def identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
        return (
            value.st_dev,
            value.st_ino,
            value.st_size,
            value.st_mtime_ns,
            value.st_ctime_ns,
        )
    if identity(before) != identity(after):
        raise CandidateProtocolError(f"artifact changed while read: {path}")
    result = b"".join(chunks)
    if len(result) != before.st_size:
        raise CandidateProtocolError(f"artifact changed while read: {path}")
    return result


def _package_files(root: Path, *, python_only: bool = False) -> tuple[Path, ...]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("artifact root must be a real directory")
    values: list[Path] = []
    for path in sorted(root.rglob("*"), key=lambda item: item.as_posix()):
        relative = path.relative_to(root)
        if "__pycache__" in relative.parts or path.suffix in {".pyc", ".pyo"}:
            continue
        metadata = path.lstat()
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"symlinks are forbidden in artifact roots: {relative}")
        if stat.S_ISDIR(metadata.st_mode):
            continue
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"special files are forbidden in artifact roots: {relative}")
        if python_only and path.suffix != ".py":
            continue
        values.append(path)
    if not values or len(values) > MAX_PACKAGE_FILES:
        raise ValueError("artifact file count is empty or exceeds the limit")
    if sum(path.stat().st_size for path in values) > MAX_PACKAGE_BYTES:
        raise ValueError("artifact bytes exceed the package limit")
    return tuple(values)


def _entrypoint_parts(entrypoint: object) -> tuple[str, str]:
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise ValueError("entrypoint must have form package.module:factory")
    module, factory = entrypoint.split(":", 1)
    if not module or any(not part.isidentifier() for part in module.split(".")):
        raise ValueError("entrypoint module is invalid")
    if not factory or any(not part.isidentifier() for part in factory.split(".")):
        raise ValueError("entrypoint factory is invalid")
    return module, factory


@dataclass(frozen=True, slots=True)
class Grade3ArtifactCommitment:
    entrypoint: str
    package_name: str
    package_root: str
    candidate_files: tuple[tuple[str, str], ...]
    sdk_root: str
    sdk_files: tuple[tuple[str, str], ...]
    sdk_commitment: str
    digest: str

    def __post_init__(self) -> None:
        module, _factory = _entrypoint_parts(self.entrypoint)
        if module.split(".")[0] != self.package_name:
            raise CandidateProtocolError("entrypoint lies outside the candidate package")
        if not self.candidate_files or not self.sdk_files:
            raise CandidateProtocolError("candidate and SDK manifests must be nonempty")
        for files, label in (
            (self.candidate_files, "candidate"),
            (self.sdk_files, "sdk"),
        ):
            paths = [path for path, _value in files]
            if paths != sorted(paths) or len(paths) != len(set(paths)):
                raise CandidateProtocolError(f"{label} manifest paths are not canonical")
            for path, value in files:
                if Path(path).is_absolute() or ".." in Path(path).parts:
                    raise CandidateProtocolError(f"{label} path escapes its root")
                _digest(value, f"{label} file digest")
        _digest(self.sdk_commitment, "sdk_commitment")
        expected = manifest_hash(
            {
                "version": GRADE3_ISOLATION_PROTOCOL,
                "entrypoint": self.entrypoint,
                "package_name": self.package_name,
                "candidate_files": list(self.candidate_files),
                "sdk_files": list(self.sdk_files),
                "sdk_commitment": self.sdk_commitment,
            }
        )
        if self.digest != expected:
            raise CandidateProtocolError("Grade-3 artifact commitment is inconsistent")


def commit_grade3_candidate(
    package_root: str | os.PathLike[str],
    entrypoint: str,
    *,
    sdk_root: str | os.PathLike[str] | None = None,
) -> Grade3ArtifactCommitment:
    """Commit package bytes lexically; never import candidate code on the host."""

    root = Path(package_root).resolve(strict=True)
    module, _factory = _entrypoint_parts(entrypoint)
    if module.split(".")[0] != root.name:
        raise ValueError("entrypoint must begin with the artifact package directory name")
    module_path = root.joinpath(*module.split(".")[1:]).with_suffix(".py")
    package_path = root.joinpath(*module.split(".")[1:], "__init__.py")
    if not module_path.is_file() and not package_path.is_file():
        raise ValueError("entrypoint module source is absent from the package root")
    sdk = (
        Path(sdk_root).resolve(strict=True)
        if sdk_root is not None
        else Path(__file__).resolve().parent
    )
    candidate_paths = _package_files(root)
    sdk_paths = _package_files(sdk, python_only=True)

    def rows(paths: tuple[Path, ...], base: Path) -> tuple[tuple[str, str], ...]:
        return tuple(
            (path.relative_to(base).as_posix(), sha256(_stable_read(path)).hexdigest())
            for path in paths
        )

    candidate_files = rows(candidate_paths, root)
    sdk_files = rows(sdk_paths, sdk)
    sdk_commitment = manifest_hash({"sdk_files": list(sdk_files)})
    digest = manifest_hash(
        {
            "version": GRADE3_ISOLATION_PROTOCOL,
            "entrypoint": entrypoint,
            "package_name": root.name,
            "candidate_files": list(candidate_files),
            "sdk_files": list(sdk_files),
            "sdk_commitment": sdk_commitment,
        }
    )
    return Grade3ArtifactCommitment(
        entrypoint,
        root.name,
        str(root),
        candidate_files,
        str(sdk),
        sdk_files,
        sdk_commitment,
        digest,
    )


def verify_grade3_artifact(commitment: Grade3ArtifactCommitment) -> str:
    if not isinstance(commitment, Grade3ArtifactCommitment):
        raise TypeError("commitment must be Grade3ArtifactCommitment")
    for root_text, rows in (
        (commitment.package_root, commitment.candidate_files),
        (commitment.sdk_root, commitment.sdk_files),
    ):
        root = Path(root_text).resolve(strict=True)
        for relative, expected in rows:
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise CandidateProtocolError(f"committed artifact disappeared: {relative}")
            if sha256(_stable_read(path)).hexdigest() != expected:
                raise CandidateProtocolError(f"committed artifact changed: {relative}")
    return commitment.digest


def _stage_candidate(
    commitment: Grade3ArtifactCommitment,
) -> tuple[TemporaryDirectory[str], Path]:
    directory = TemporaryDirectory(prefix="grounding-grade3-candidate-")
    stage = Path(directory.name)
    package = stage / commitment.package_name
    try:
        for relative, expected in commitment.candidate_files:
            source = Path(commitment.package_root) / relative
            data = _stable_read(source)
            if sha256(data).hexdigest() != expected:
                raise CandidateProtocolError(f"candidate changed before staging: {relative}")
            destination = package / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
            try:
                remaining = memoryview(data)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written <= 0:  # pragma: no cover - defensive OS boundary
                        raise OSError("short write while staging candidate artifact")
                    remaining = remaining[written:]
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(destination, 0o400)
        for path in sorted(
            (item for item in stage.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(path, 0o500)
        os.chmod(stage, 0o500)
        return directory, stage
    except BaseException:
        os.chmod(stage, 0o700)
        directory.cleanup()
        raise


def _verify_stage(commitment: Grade3ArtifactCommitment, stage: Path) -> None:
    for relative, expected in commitment.candidate_files:
        path = stage / commitment.package_name / relative
        data = _stable_read(path)
        if sha256(data).hexdigest() != expected:
            raise CandidateProtocolError(f"staged candidate failed rehash: {relative}")


def _load_candidate(commitment: Grade3ArtifactCommitment, stage: Path) -> Grade3Grounder:
    module_name, attribute = _entrypoint_parts(commitment.entrypoint)
    sys.path.insert(0, str(stage))
    for name in tuple(sys.modules):
        if name == commitment.package_name or name.startswith(
            f"{commitment.package_name}."
        ):
            sys.modules.pop(name, None)
    invalidate_caches()
    module: object = import_module(module_name)
    factory: object = module
    for part in attribute.split("."):
        factory = getattr(factory, part)
    if not callable(factory):
        raise CandidateProtocolError("candidate factory is not callable")
    candidate = factory()
    if not isinstance(candidate, Grade3Grounder):
        raise CandidateProtocolError("candidate does not implement Grade3Grounder")
    allowed = {
        (stage / commitment.package_name / relative).resolve()
        for relative, _digest_value in commitment.candidate_files
        if relative.endswith(".py")
    }
    for name, loaded in tuple(sys.modules.items()):
        if name != commitment.package_name and not name.startswith(
            f"{commitment.package_name}."
        ):
            continue
        source = getattr(loaded, "__file__", None)
        if source is None or Path(source).resolve() not in allowed:
            raise CandidateProtocolError("candidate loaded a module outside its manifest")
    return candidate


def _checkpoint(candidate: object) -> str:
    method = getattr(candidate, "checkpoint_commitment", None)
    if not callable(method):
        raise CandidateProtocolError("candidate lacks checkpoint_commitment")
    return _digest(method(), "checkpoint")


def _json(value: object, max_bytes: int) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(encoded) > max_bytes:
        raise CandidateProtocolError("Grade-3 RPC message exceeds the byte limit")
    return encoded


def _rpc_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateProtocolError(f"duplicate Grade-3 RPC key: {key!r}")
        result[key] = value
    return result


def _rpc_constant(value: str) -> None:
    raise CandidateProtocolError(f"non-finite Grade-3 RPC constant: {value}")


def _rpc_nodes(value: object, depth: int = 0) -> int:
    if depth > MAX_JSON_DEPTH:
        raise CandidateProtocolError("Grade-3 RPC nesting exceeds the limit")
    if isinstance(value, dict):
        return 1 + sum(_rpc_nodes(item, depth + 1) for item in value.values())
    if isinstance(value, list):
        return 1 + sum(_rpc_nodes(item, depth + 1) for item in value)
    if isinstance(value, float) and not math.isfinite(value):
        raise CandidateProtocolError("Grade-3 RPC numbers must be finite")
    if value is None or isinstance(value, (str, bool, int, float)):
        return 1
    raise CandidateProtocolError("Grade-3 RPC contains an unsupported JSON value")


def _decode_json(value: bytes, max_bytes: int) -> dict[str, Any]:
    if len(value) > max_bytes:
        raise CandidateProtocolError("Grade-3 RPC message exceeds the byte limit")
    try:
        decoded = json.loads(
            value.decode("ascii"),
            object_pairs_hook=_rpc_object,
            parse_constant=_rpc_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CandidateProtocolError("Grade-3 RPC is not strict ASCII JSON") from exc
    if _rpc_nodes(decoded) > MAX_JSON_NODES:
        raise CandidateProtocolError("Grade-3 RPC exceeds the JSON-node limit")
    if not isinstance(decoded, dict):
        raise CandidateProtocolError("Grade-3 RPC envelope must be an object")
    return decoded


def _wire(value: object) -> dict[str, Any]:
    return json.loads(encode_grade3_message(value).decode("utf-8"))


def _unwire(value: object) -> object:
    return decode_grade3_message(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def _child_main(
    connection: Connection,
    commitment: Grade3ArtifactCommitment,
    stage_text: str,
    max_message_bytes: int,
    max_requests: int,
) -> None:
    began = False
    frozen: str | None = None
    requests = 0
    try:
        verify_grade3_artifact(commitment)
        stage = Path(stage_text).resolve(strict=True)
        _verify_stage(commitment, stage)
        candidate = _load_candidate(commitment, stage)
        while True:
            raw = connection.recv_bytes(maxlength=max_message_bytes)
            envelope = _decode_json(raw, max_message_bytes)
            if set(envelope) != {"protocol", "id", "command", "payload"}:
                raise CandidateProtocolError("Grade-3 RPC keys are not exact")
            if envelope["protocol"] != GRADE3_ISOLATION_PROTOCOL:
                raise CandidateProtocolError("wrong Grade-3 RPC version")
            request_id = envelope["id"]
            command = envelope["command"]
            if isinstance(request_id, bool) or request_id != requests:
                raise CandidateProtocolError("request IDs must be consecutive")
            if command not in _COMMANDS:
                raise CandidateProtocolError("command is not allowlisted")
            requests += 1
            if requests > max_requests:
                raise CandidateProtocolError("candidate request budget exhausted")
            payload = envelope["payload"]
            if command == "close":
                if payload is not None:
                    raise CandidateProtocolError("close payload must be null")
                result: object = None
                connection.send_bytes(
                    _json(
                        {
                            "protocol": GRADE3_ISOLATION_PROTOCOL,
                            "id": request_id,
                            "ok": True,
                            "result": result,
                            "error": None,
                        },
                        max_message_bytes,
                    )
                )
                break
            if command == "begin":
                if began or frozen is not None:
                    raise CandidateProtocolError("begin occurs exactly once first")
                manifest = _unwire(payload)
                if not isinstance(manifest, Grade3SessionManifest):
                    raise CandidateProtocolError("begin requires Grade3SessionManifest")
                candidate.begin(manifest)
                began = True
                result = None
            elif not began:
                raise CandidateProtocolError("begin must precede every command")
            elif command == "observe_support":
                if frozen is not None:
                    raise CandidateProtocolError("support is pre-freeze")
                record = _unwire(payload)
                if not isinstance(record, (OstensiveSupportRecord, CausalSupportRecord)):
                    raise CandidateProtocolError("observe_support requires a support record")
                candidate.observe_support(record)
                result = None
            elif command == "choose_probe":
                if frozen is not None:
                    raise CandidateProtocolError("probe choice is pre-freeze")
                offer = _unwire(payload)
                if not isinstance(offer, ProbeOffer):
                    raise CandidateProtocolError("choose_probe requires ProbeOffer")
                decision = candidate.choose_probe(offer)
                if not isinstance(decision, ProbeDecision):
                    raise CandidateProtocolError("choose_probe returned the wrong type")
                if decision.probe_id is not None and decision.probe_id not in {
                    option.probe_id for option in offer.options
                }:
                    raise CandidateProtocolError("candidate selected an unoffered probe")
                result = _wire(decision)
            elif command == "observe_probe":
                if frozen is not None:
                    raise CandidateProtocolError("probe observation is pre-freeze")
                probe = _unwire(payload)
                if not isinstance(probe, ProbeResult):
                    raise CandidateProtocolError("observe_probe requires ProbeResult")
                candidate.observe_probe(probe)
                result = None
            elif command == "freeze":
                if frozen is not None or payload is not None:
                    raise CandidateProtocolError("freeze occurs exactly once")
                candidate.freeze()
                frozen = _checkpoint(candidate)
                result = frozen
            elif command == "checkpoint":
                if frozen is None or payload is not None:
                    raise CandidateProtocolError("checkpoint requires freeze")
                result = _checkpoint(candidate)
                if result != frozen:
                    raise CandidateProtocolError("candidate mutated after freeze")
            elif command in _QUERY_COMMANDS:
                if frozen is None:
                    raise CandidateProtocolError("sealed query requires freeze")
                if _checkpoint(candidate) != frozen:
                    raise CandidateProtocolError("candidate mutated before query")
                query = _unwire(payload)
                if command == "motor":
                    if not isinstance(query, MotorQuery):
                        raise CandidateProtocolError("motor requires MotorQuery")
                    response = candidate.motor(query)
                    expected = MotorDecision
                elif command == "trace_belief":
                    if not isinstance(query, TraceBeliefQuery):
                        raise CandidateProtocolError(
                            "trace_belief requires TraceBeliefQuery"
                        )
                    response = candidate.trace_belief(query)
                    expected = BeliefDecision
                    if isinstance(response, BeliefDecision) and not {
                        item for item, _probability in response.candidate_probabilities
                    }.issubset(query.candidates):
                        raise CandidateProtocolError("belief returned an unrequested ID")
                else:
                    if not isinstance(query, TraceDescriptionQuery):
                        raise CandidateProtocolError(
                            "describe requires TraceDescriptionQuery"
                        )
                    response = candidate.describe(query)
                    expected = DescriptionDecision
                if not isinstance(response, expected):
                    raise CandidateProtocolError(f"{command} returned the wrong type")
                if _checkpoint(candidate) != frozen:
                    raise CandidateProtocolError("candidate mutated during sealed query")
                result = _wire(response)
            else:  # pragma: no cover
                raise CandidateProtocolError("unsupported command")
            connection.send_bytes(
                _json(
                    {
                        "protocol": GRADE3_ISOLATION_PROTOCOL,
                        "id": request_id,
                        "ok": True,
                        "result": result,
                        "error": None,
                    },
                    max_message_bytes,
                )
            )
    except BaseException as exc:
        error = {
            "type": f"{type(exc).__module__}.{type(exc).__qualname__}",
            "message": str(exc),
            "traceback": "".join(traceback.format_exception(exc))[-8_192:],
        }
        try:
            connection.send_bytes(
                _json(
                    {
                        "protocol": GRADE3_ISOLATION_PROTOCOL,
                        "id": max(0, requests - 1),
                        "ok": False,
                        "result": None,
                        "error": error,
                    },
                    max_message_bytes,
                )
            )
        except BaseException:
            pass
    finally:
        connection.close()


@dataclass(frozen=True, slots=True)
class FrozenGrade3Candidate:
    artifact_commitment: str
    sdk_commitment: str
    checkpoint_commitment: str
    request_count: int


class IsolatedGrade3Grounder:
    """Parent proxy for one recursively committed persistent candidate."""

    def __init__(
        self,
        commitment: Grade3ArtifactCommitment,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if not isinstance(commitment, Grade3ArtifactCommitment):
            raise TypeError("commitment must be Grade3ArtifactCommitment")
        if timeout <= 0.0 or max_requests < 1:
            raise ValueError("timeout and max_requests must be positive")
        self.commitment = commitment
        self.timeout = float(timeout)
        self.max_requests = int(max_requests)
        self.max_message_bytes = int(max_message_bytes)
        self._process: Any = None
        self._connection: Connection | None = None
        self._request_id = 0
        self._closed = False
        self._stage: TemporaryDirectory[str] | None = None
        self._frozen: FrozenGrade3Candidate | None = None

    @property
    def request_count(self) -> int:
        return self._request_id

    @property
    def frozen(self) -> FrozenGrade3Candidate | None:
        return self._frozen

    def start(self) -> "IsolatedGrade3Grounder":
        if self._closed or self._process is not None:
            raise CandidateIsolationError("candidate proxy cannot be started")
        verify_grade3_artifact(self.commitment)
        directory, stage = _stage_candidate(self.commitment)
        parent, child = get_context("spawn").Pipe(duplex=True)
        process = get_context("spawn").Process(
            target=_child_main,
            args=(
                child,
                self.commitment,
                str(stage),
                self.max_message_bytes,
                self.max_requests,
            ),
            daemon=True,
        )
        try:
            process.start()
            child.close()
        except BaseException:
            parent.close()
            child.close()
            os.chmod(stage, 0o700)
            directory.cleanup()
            raise
        self._process = process
        self._connection = parent
        self._stage = directory
        return self

    def __enter__(self) -> "IsolatedGrade3Grounder":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        self.close()

    def _call(self, command: str, payload: object | None) -> object:
        if self._closed or self._process is None or self._connection is None:
            raise CandidateIsolationError("candidate process is not running")
        if self._request_id >= self.max_requests:
            raise CandidateProtocolError("candidate request budget exhausted")
        request_id = self._request_id
        self._request_id += 1
        wire_payload = None if payload is None else _wire(payload)
        self._connection.send_bytes(
            _json(
                {
                    "protocol": GRADE3_ISOLATION_PROTOCOL,
                    "id": request_id,
                    "command": command,
                    "payload": wire_payload,
                },
                self.max_message_bytes,
            )
        )
        deadline = time.monotonic() + self.timeout
        if not self._connection.poll(max(0.0, deadline - time.monotonic())):
            self._stop()
            raise CandidateTimeoutError(f"candidate timed out during {command}")
        response = _decode_json(
            self._connection.recv_bytes(maxlength=self.max_message_bytes),
            self.max_message_bytes,
        )
        if set(response) != {"protocol", "id", "ok", "result", "error"}:
            raise CandidateProtocolError("Grade-3 response keys are not exact")
        if (
            response["protocol"] != GRADE3_ISOLATION_PROTOCOL
            or response["id"] != request_id
        ):
            raise CandidateProtocolError("Grade-3 response does not match request")
        if response["ok"] is True and response["error"] is None:
            return response["result"]
        error = response["error"]
        if not isinstance(error, dict):
            raise CandidateProtocolError("candidate returned a malformed error")
        raise CandidateExecutionError(
            str(error.get("type")),
            str(error.get("message")),
            str(error.get("traceback")),
        )

    def begin(self, manifest: Grade3SessionManifest) -> None:
        self._call("begin", manifest)

    def observe_support(
        self, record: OstensiveSupportRecord | CausalSupportRecord
    ) -> None:
        self._call("observe_support", record)

    def choose_probe(self, offer: ProbeOffer) -> ProbeDecision:
        result = _unwire(self._call("choose_probe", offer))
        if not isinstance(result, ProbeDecision):
            raise CandidateProtocolError("choose_probe returned the wrong type")
        return result

    def observe_probe(self, result: ProbeResult) -> None:
        self._call("observe_probe", result)

    def freeze(self) -> FrozenGrade3Candidate:
        checkpoint = _digest(self._call("freeze", None), "checkpoint")
        frozen = FrozenGrade3Candidate(
            self.commitment.digest,
            self.commitment.sdk_commitment,
            checkpoint,
            self.request_count,
        )
        self._frozen = frozen
        return frozen

    def assert_frozen(self) -> str:
        checkpoint = _digest(self._call("checkpoint", None), "checkpoint")
        if self._frozen is None or checkpoint != self._frozen.checkpoint_commitment:
            raise CandidateProtocolError("checkpoint differs from frozen record")
        return checkpoint

    def motor(self, query: MotorQuery) -> MotorDecision:
        result = _unwire(self._call("motor", query))
        if not isinstance(result, MotorDecision):
            raise CandidateProtocolError("motor returned the wrong type")
        return result

    def trace_belief(self, query: TraceBeliefQuery) -> BeliefDecision:
        result = _unwire(self._call("trace_belief", query))
        if not isinstance(result, BeliefDecision):
            raise CandidateProtocolError("trace_belief returned the wrong type")
        return result

    def describe(self, query: TraceDescriptionQuery) -> DescriptionDecision:
        result = _unwire(self._call("describe", query))
        if not isinstance(result, DescriptionDecision):
            raise CandidateProtocolError("describe returned the wrong type")
        return result

    def _stop(self) -> None:
        if self._process is None:
            return
        self._process.join(_JOIN_GRACE_SECONDS)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(_JOIN_GRACE_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(_JOIN_GRACE_SECONDS)

    def close(self) -> None:
        if self._closed:
            return
        if self._process is not None and self._process.is_alive():
            try:
                self._call("close", None)
            except (CandidateIsolationError, CandidateTimeoutError, OSError, EOFError):
                pass
        self._stop()
        if self._connection is not None:
            self._connection.close()
        if self._stage is not None:
            root = Path(self._stage.name)
            if root.exists():
                for path in root.rglob("*"):
                    try:
                        os.chmod(path, 0o700 if path.is_dir() else 0o600)
                    except OSError:
                        pass
                os.chmod(root, 0o700)
            self._stage.cleanup()
        self._closed = True


__all__ = [
    "DEFAULT_MAX_REQUESTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FrozenGrade3Candidate",
    "GRADE3_ISOLATION_PROTOCOL",
    "Grade3ArtifactCommitment",
    "IsolatedGrade3Grounder",
    "commit_grade3_candidate",
    "verify_grade3_artifact",
]
