"""Persistent JSON-only process boundary for GroundZero-v1 candidates.

The evaluator commits the candidate artifact before constructing a secret
codebook, then talks to one persistent candidate instance for support,
acquisition, freeze and every sealed query.  Only values accepted by
``v1_wire`` cross the pipe; evaluator environments and oracle capabilities do
not.

This is a capability/serialization boundary, not an operating-system sandbox.
Python code in the child still has whatever filesystem and network authority
the host grants it.  Consequently this module can certify protocol isolation
for trusted reference candidates, while adversarial claims additionally need
an external OS sandbox.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib import import_module, invalidate_caches, util as importlib_util
from math import isfinite
from multiprocessing import get_context
from multiprocessing.connection import Connection
from pathlib import Path
from tempfile import TemporaryDirectory
from types import TracebackType
from typing import Any, TypeAlias
import base64
import json
import os
import stat
import sys
import time
import traceback

from .contracts import Observation
from .v1_contracts import (
    MAX_BELIEF_CANDIDATES,
    ActionDecision,
    BeliefDecision,
    DescriptionDecision,
    ExperimentDecision,
    InteractiveGrounder,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionManifest,
    SessionPhase,
    Utterance,
)
from .v1_wire import MAX_MESSAGE_BYTES, decode_message, encode_message


JSONScalar: TypeAlias = None | bool | int | float | str
JSONValue: TypeAlias = JSONScalar | list["JSONValue"] | dict[str, "JSONValue"]

ISOLATION_PROTOCOL = "grounding-candidate-rpc/1"
_ARTIFACT_MANIFEST_VERSION = "grounding-candidate-artifact/2"
DEFAULT_TIMEOUT_SECONDS = 20.0
DEFAULT_MAX_REQUESTS = 20_000
_JOIN_GRACE_SECONDS = 0.25
_COMMANDS = frozenset(
    {
        "begin",
        "observe_support",
        "choose_experiment",
        "observe_experiment",
        "describe",
        "begin_goal",
        "act",
        "report_belief",
        "freeze",
        "checkpoint",
        "close",
    }
)
_QUERY_COMMANDS = frozenset({"describe", "begin_goal", "act", "report_belief"})


class CandidateIsolationError(RuntimeError):
    """Base error raised by the v1 candidate boundary."""


class CandidateProtocolError(CandidateIsolationError):
    """A child or caller violated the exact bounded RPC protocol."""


class CandidateExecutionError(CandidateIsolationError):
    """The candidate raised inside its isolated process."""

    def __init__(self, remote_type: str, message: str, remote_traceback: str) -> None:
        detail = f"{remote_type}: {message}" if message else remote_type
        super().__init__(f"candidate failed: {detail}")
        self.remote_type = remote_type
        self.remote_message = message
        self.remote_traceback = remote_traceback


class CandidateTimeoutError(TimeoutError):
    """A persistent candidate failed to answer before the deadline."""


@dataclass(frozen=True, slots=True)
class CandidateArtifactCommitment:
    """Pre-codebook commitment to an importable entrypoint and its artifacts."""

    entrypoint: str
    source_path: str
    artifact_paths: tuple[str, ...]
    module_root: str
    file_digests: tuple[tuple[str, str], ...]
    digest: str

    def __post_init__(self) -> None:
        module_name, _attribute = _validate_entrypoint(self.entrypoint)
        source = _absolute_path(self.source_path, "source_path")
        module_root = _absolute_path(self.module_root, "module_root")
        artifacts = tuple(
            _absolute_path(path, "artifact_paths") for path in self.artifact_paths
        )
        if source in artifacts or len(set(artifacts)) != len(artifacts):
            raise CandidateProtocolError(
                "candidate source and artifact paths must be unique"
            )
        if artifacts != tuple(sorted(artifacts)):
            raise CandidateProtocolError("candidate artifact paths must be sorted")
        try:
            source.relative_to(module_root)
        except ValueError as exc:
            raise CandidateProtocolError(
                "candidate source_path must lie below module_root"
            ) from exc
        expected_source = _module_source_relative_path(module_name, source)
        if source.relative_to(module_root) != expected_source:
            raise CandidateProtocolError(
                "candidate module_root does not match the entrypoint source layout"
            )
        expected_paths = (str(source), *(str(path) for path in artifacts))
        digests = tuple(self.file_digests)
        if tuple(path for path, _digest in digests) != expected_paths:
            raise CandidateProtocolError(
                "file_digests must cover source then every sorted artifact exactly once"
            )
        for _path, value in digests:
            _validate_digest(value, "file digest")
        expected_digest = _artifact_digest(
            self.entrypoint,
            str(source),
            tuple(str(path) for path in artifacts),
            str(module_root),
            digests,
        )
        if _validate_digest(self.digest, "digest") != expected_digest:
            raise CandidateProtocolError(
                "candidate artifact digest does not match its file manifest"
            )
        object.__setattr__(self, "source_path", str(source))
        object.__setattr__(self, "artifact_paths", tuple(str(path) for path in artifacts))
        object.__setattr__(self, "module_root", str(module_root))
        object.__setattr__(self, "file_digests", digests)


@dataclass(frozen=True, slots=True)
class _StagedCandidateArtifact:
    entrypoint: str
    artifact_commitment: str
    module_root: str
    source_path: str
    files: tuple[tuple[str, str, str], ...]


def _absolute_path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise CandidateProtocolError(f"{field} must be a non-empty absolute path")
    path = Path(value)
    if not path.is_absolute() or str(path) != str(path.resolve(strict=False)):
        raise CandidateProtocolError(f"{field} must be a canonical absolute path")
    return path


@dataclass(frozen=True, slots=True)
class FrozenCandidate:
    """Record one process and its candidate-declared checkpoint digest.

    Grade 3 assumes the reference candidate reports complete state.  Only an
    evaluator-owned VM/process snapshot can attest adversarial full state.
    """

    artifact_commitment: str
    checkpoint_commitment: str
    requests_at_freeze: int


def _validate_digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise CandidateProtocolError(f"{field} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise CandidateProtocolError(
            f"{field} must be a lowercase SHA-256 digest"
        ) from exc
    return value


def _validate_entrypoint(reference: object) -> tuple[str, str]:
    if not isinstance(reference, str) or reference.count(":") != 1:
        raise ValueError("candidate entrypoint must have form 'module:callable'")
    module_name, attribute_path = reference.split(":", 1)
    if not module_name or any(not part.isidentifier() for part in module_name.split(".")):
        raise ValueError("candidate module must be an absolute dotted identifier")
    if not attribute_path or any(
        not part.isidentifier() or part == "<locals>"
        for part in attribute_path.split(".")
    ):
        raise ValueError("candidate callable must be a dotted identifier")
    return module_name, attribute_path


def _resolve_entrypoint(reference: str) -> object:
    module_name, attribute_path = _validate_entrypoint(reference)
    value: object = import_module(module_name)
    for part in attribute_path.split("."):
        value = getattr(value, part)
    if not callable(value):
        raise TypeError("candidate entrypoint must resolve to a callable factory")
    return value


def _module_source_relative_path(module_name: str, source: Path) -> Path:
    parts = module_name.split(".")
    if source.name == "__init__.py":
        return Path(*parts) / "__init__.py"
    if source.suffix != ".py":
        raise CandidateProtocolError("candidate entrypoint must resolve to Python source")
    return Path(*parts[:-1], f"{parts[-1]}.py")


def _module_layout(module_name: str, source: Path) -> tuple[Path, Path]:
    relative = _module_source_relative_path(module_name, source)
    source_parts = source.parts
    relative_parts = relative.parts
    if len(source_parts) <= len(relative_parts) or (
        source_parts[-len(relative_parts) :] != relative_parts
    ):
        raise CandidateProtocolError(
            "candidate source path does not match its dotted module name"
        )
    root = Path(*source_parts[: -len(relative_parts)])
    return root, relative


def _artifact_digest(
    entrypoint: str,
    source_path: str,
    artifact_paths: tuple[str, ...],
    module_root: str,
    file_digests: tuple[tuple[str, str], ...],
) -> str:
    payload = {
        "version": _ARTIFACT_MANIFEST_VERSION,
        "entrypoint": entrypoint,
        "source_path": source_path,
        "artifact_paths": list(artifact_paths),
        "module_root": module_root,
        "files": [
            {"path": path, "sha256": digest} for path, digest in file_digests
        ],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return sha256(encoded).hexdigest()


def _read_stable_file(path: Path, label: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise CandidateProtocolError(f"candidate {label} cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise CandidateProtocolError(
                f"candidate {label} is not a regular file: {path}"
            )
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after:
        raise CandidateProtocolError(f"candidate {label} changed while it was read: {path}")
    result = b"".join(chunks)
    if len(result) != before.st_size:
        raise CandidateProtocolError(f"candidate {label} changed while it was read: {path}")
    return result


def _current_source(entrypoint: str) -> Path:
    module_name, _attribute = _validate_entrypoint(entrypoint)
    invalidate_caches()
    spec = importlib_util.find_spec(module_name)
    if spec is None or spec.origin is None or spec.origin in {"built-in", "frozen"}:
        raise CandidateProtocolError("candidate module must have a readable source file")
    try:
        return Path(spec.origin).resolve(strict=True)
    except OSError as exc:
        raise CandidateProtocolError("candidate source file is no longer available") from exc


def _verified_candidate_bytes(
    commitment: CandidateArtifactCommitment,
    *,
    verify_import_resolution: bool,
) -> dict[str, bytes]:
    if verify_import_resolution and _current_source(commitment.entrypoint) != Path(
        commitment.source_path
    ):
        raise CandidateProtocolError(
            "candidate entrypoint no longer resolves to the committed source"
        )
    expected = dict(commitment.file_digests)
    verified: dict[str, bytes] = {}
    for path, expected_digest in commitment.file_digests:
        data = _read_stable_file(Path(path), "artifact")
        if sha256(data).hexdigest() != expected_digest:
            raise CandidateProtocolError(f"candidate artifact changed after commitment: {path}")
        verified[path] = data
    if set(verified) != set(expected):  # pragma: no cover - dataclass invariant
        raise CandidateProtocolError("candidate artifact manifest is incomplete")
    digest = _artifact_digest(
        commitment.entrypoint,
        commitment.source_path,
        commitment.artifact_paths,
        commitment.module_root,
        commitment.file_digests,
    )
    if digest != commitment.digest:
        raise CandidateProtocolError("candidate artifact manifest commitment changed")
    return verified


def verify_candidate_artifact(commitment: CandidateArtifactCommitment) -> str:
    """Rehash every committed live byte and return the unchanged commitment."""

    if not isinstance(commitment, CandidateArtifactCommitment):
        raise TypeError("commitment must be CandidateArtifactCommitment")
    _verified_candidate_bytes(commitment, verify_import_resolution=True)
    return commitment.digest


def _staged_relative_path(original: Path, module_root: Path) -> Path:
    try:
        return original.relative_to(module_root)
    except ValueError:
        path_hash = sha256(str(original).encode("utf-8")).hexdigest()
        return Path("_external_artifacts") / path_hash


def _stage_candidate_artifact(
    commitment: CandidateArtifactCommitment,
    verified: dict[str, bytes],
) -> tuple[TemporaryDirectory[str], _StagedCandidateArtifact]:
    directory = TemporaryDirectory(prefix="grounding-candidate-")
    staging_root = Path(directory.name).resolve(strict=True)
    original_root = Path(commitment.module_root)
    staged_files: list[tuple[str, str, str]] = []
    try:
        for original_text, expected_digest in commitment.file_digests:
            original = Path(original_text)
            relative = _staged_relative_path(original, original_root)
            destination = staging_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            data = verified[original_text]
            # Exclusive creation makes the staged manifest one file -> one byte string.
            descriptor = os.open(
                destination,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
                0o400,
            )
            try:
                view = memoryview(data)
                written = 0
                while written < len(view):
                    written += os.write(descriptor, view[written:])
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            os.chmod(destination, 0o400)
            staged_files.append((original_text, str(destination), expected_digest))
        for path in sorted(
            (item for item in staging_root.rglob("*") if item.is_dir()),
            key=lambda item: len(item.parts),
            reverse=True,
        ):
            os.chmod(path, 0o500)
        os.chmod(staging_root, 0o500)
        source_index = tuple(path for path, _digest in commitment.file_digests).index(
            commitment.source_path
        )
        staged = _StagedCandidateArtifact(
            entrypoint=commitment.entrypoint,
            artifact_commitment=commitment.digest,
            module_root=str(staging_root),
            source_path=staged_files[source_index][1],
            files=tuple(staged_files),
        )
        return directory, staged
    except BaseException:
        os.chmod(staging_root, 0o700)
        directory.cleanup()
        raise


def _verify_staged_candidate(
    commitment: CandidateArtifactCommitment,
    staged: _StagedCandidateArtifact,
) -> None:
    if (
        staged.entrypoint != commitment.entrypoint
        or staged.artifact_commitment != commitment.digest
    ):
        raise CandidateProtocolError("staged candidate manifest does not match commitment")
    staging_root = _absolute_path(staged.module_root, "staged module_root")
    root_metadata = os.stat(staging_root, follow_symlinks=False)
    if not stat.S_ISDIR(root_metadata.st_mode) or root_metadata.st_mode & 0o222:
        raise CandidateProtocolError("staged candidate root must be read-only")
    original_root = Path(commitment.module_root)
    expected_files: list[tuple[str, str, str]] = []
    for original, digest in commitment.file_digests:
        destination = staging_root / _staged_relative_path(Path(original), original_root)
        expected_files.append((original, str(destination), digest))
    if staged.files != tuple(expected_files):
        raise CandidateProtocolError("staged candidate file map is not canonical")
    expected_source = dict(
        (original, destination) for original, destination, _digest in staged.files
    )[commitment.source_path]
    if staged.source_path != expected_source:
        raise CandidateProtocolError("staged candidate source does not match commitment")
    directories = {staging_root}
    for _original, destination, _digest in staged.files:
        current = Path(destination).parent
        while current != staging_root:
            directories.add(current)
            current = current.parent
    for directory in directories:
        metadata = os.stat(directory, follow_symlinks=False)
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o222:
            raise CandidateProtocolError("staged candidate directories must be read-only")
    for _original, destination, expected_digest in staged.files:
        path = _absolute_path(destination, "staged artifact path")
        try:
            path.relative_to(staging_root)
        except ValueError as exc:
            raise CandidateProtocolError("staged artifact escaped its declared root") from exc
        data = _read_stable_file(path, "staged artifact")
        metadata = os.stat(path, follow_symlinks=False)
        if metadata.st_mode & 0o222:
            raise CandidateProtocolError("staged candidate files must be read-only")
        if sha256(data).hexdigest() != expected_digest:
            raise CandidateProtocolError("staged candidate bytes failed child rehash")


def _resolve_staged_entrypoint(
    commitment: CandidateArtifactCommitment,
    staged: _StagedCandidateArtifact,
) -> object:
    module_name, _attribute = _validate_entrypoint(commitment.entrypoint)
    original_root = Path(commitment.module_root)
    staging_root = Path(staged.module_root)
    retained_paths: list[str] = []
    for entry in sys.path:
        try:
            if Path(entry or os.getcwd()).resolve(strict=False) == original_root:
                continue
        except OSError:
            pass
        retained_paths.append(entry)
    sys.path[:] = [str(staging_root), *retained_paths]
    staged_modules = {
        ".".join(module_name.split(".")[:prefix_length])
        for prefix_length in range(1, len(module_name.split(".")) + 1)
    }
    for original, _destination, _digest in staged.files:
        try:
            relative = Path(original).relative_to(original_root)
        except ValueError:
            continue
        if relative.suffix != ".py":
            continue
        parts = list(relative.with_suffix("").parts)
        if parts and parts[-1] == "__init__":
            parts.pop()
        if parts and all(part.isidentifier() for part in parts):
            staged_modules.add(".".join(parts))
    for staged_module in sorted(staged_modules, key=lambda item: item.count("."), reverse=True):
        sys.modules.pop(staged_module, None)
    os.chdir(staging_root)
    os.environ["GROUNDING_CANDIDATE_ARTIFACT_MANIFEST"] = json.dumps(
        {
            "version": _ARTIFACT_MANIFEST_VERSION,
            "commitment": commitment.digest,
            "files": [
                {"original": original, "staged": destination, "sha256": digest}
                for original, destination, digest in staged.files
            ],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    invalidate_caches()
    factory = _resolve_entrypoint(commitment.entrypoint)
    loaded = sys.modules.get(module_name)
    loaded_path = getattr(loaded, "__file__", None)
    if loaded_path is None or Path(loaded_path).resolve(strict=True) != Path(
        staged.source_path
    ):
        raise CandidateProtocolError("candidate was not imported from committed staged bytes")
    return factory


def commit_candidate_artifact(
    entrypoint: str,
    artifact_paths: Iterable[str | os.PathLike[str]] = (),
) -> CandidateArtifactCommitment:
    """Hash candidate source and declared weight/config artifacts.

    This function is evaluator-side and must be called before the codebook or
    sealed worlds are sampled.  Artifact order is canonicalized and duplicate
    paths are rejected so the commitment cannot be reinterpreted later.
    """

    module_name, _attribute = _validate_entrypoint(entrypoint)
    try:
        source = _current_source(entrypoint)
    except CandidateProtocolError as exc:
        raise ValueError(str(exc)) from exc
    module_root, _relative_source = _module_layout(module_name, source)
    declared = tuple(Path(value).resolve(strict=True) for value in artifact_paths)
    if len(set(declared)) != len(declared):
        raise ValueError("candidate artifact paths must be unique")
    if source in declared:
        raise ValueError("candidate source and artifact paths must be unique")
    package_initializers = tuple(
        module_root.joinpath(*module_name.split(".")[:prefix], "__init__.py")
        for prefix in range(1, len(module_name.split(".")))
    )
    auto_declared = tuple(path for path in package_initializers if path.is_file())
    ordered = tuple(sorted(set(declared).union(auto_declared), key=lambda item: str(item)))
    paths = (source, *ordered)
    try:
        contents = {
            str(path): _read_stable_file(path, "source" if path == source else "artifact")
            for path in paths
        }
    except CandidateProtocolError as exc:
        raise ValueError(str(exc)) from exc
    file_digests = tuple(
        (str(path), sha256(contents[str(path)]).hexdigest()) for path in paths
    )
    artifact_paths_text = tuple(str(path) for path in ordered)
    digest = _artifact_digest(
        entrypoint,
        str(source),
        artifact_paths_text,
        str(module_root),
        file_digests,
    )
    return CandidateArtifactCommitment(
        entrypoint=entrypoint,
        source_path=str(source),
        artifact_paths=artifact_paths_text,
        module_root=str(module_root),
        file_digests=file_digests,
        digest=digest,
    )


def _json_value(value: object, *, depth: int = 0) -> JSONValue:
    if depth > 32:
        raise CandidateProtocolError("RPC JSON nesting exceeds 32 levels")
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        if not -(1 << 63) <= value <= (1 << 63) - 1:
            raise CandidateProtocolError("RPC integer exceeds signed 64-bit range")
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise CandidateProtocolError("RPC numbers must be finite")
        return value
    if isinstance(value, (tuple, list)):
        return [_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise CandidateProtocolError("RPC object keys must be strings")
        return {
            key: _json_value(item, depth=depth + 1)
            for key, item in sorted(value.items())
        }
    raise CandidateProtocolError(
        f"unsupported RPC value: {type(value).__name__}"
    )


def _encode_json(value: dict[str, object], max_bytes: int) -> bytes:
    payload = json.dumps(
        _json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    if len(payload) > max_bytes:
        raise CandidateProtocolError("RPC frame exceeds max_message_bytes")
    return payload


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise CandidateProtocolError(f"duplicate RPC key: {key!r}")
        result[key] = value
    return result


def _decode_json(payload: bytes, max_bytes: int) -> dict[str, Any]:
    if len(payload) > max_bytes:
        raise CandidateProtocolError("RPC frame exceeds max_message_bytes")
    try:
        value = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                CandidateProtocolError(f"forbidden JSON constant: {constant}")
            ),
        )
    except CandidateProtocolError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise CandidateProtocolError("RPC frame is not strict UTF-8 JSON") from exc
    checked = _json_value(value)
    if not isinstance(checked, dict):
        raise CandidateProtocolError("RPC envelope must be an object")
    return checked


def _exact(value: object, keys: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        actual = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise CandidateProtocolError(
            f"{label} keys must be exactly {sorted(keys)!r}; got {actual!r}"
        )
    return value


def _wire(value: object) -> str:
    return base64.b64encode(encode_message(value)).decode("ascii")


def _unwire(value: object) -> object:
    if not isinstance(value, str):
        raise CandidateProtocolError("wire item must be base64 text")
    try:
        raw = base64.b64decode(value, validate=True)
    except ValueError as exc:
        raise CandidateProtocolError("wire item is not canonical base64") from exc
    try:
        return decode_message(raw)
    except (TypeError, ValueError) as exc:
        raise CandidateProtocolError(f"invalid nested v1 record: {exc}") from exc


def _request(request_id: int, command: str, arguments: list[JSONValue]) -> dict[str, object]:
    if command not in _COMMANDS:
        raise CandidateProtocolError(f"RPC command is not allowlisted: {command!r}")
    return {
        "protocol": ISOLATION_PROTOCOL,
        "id": request_id,
        "command": command,
        "arguments": arguments,
    }


def _response(
    request_id: int,
    *,
    result: JSONValue | None = None,
    error: BaseException | None = None,
) -> dict[str, object]:
    if error is None:
        return {
            "protocol": ISOLATION_PROTOCOL,
            "id": request_id,
            "ok": True,
            "result": result,
            "error": None,
        }
    return {
        "protocol": ISOLATION_PROTOCOL,
        "id": request_id,
        "ok": False,
        "result": None,
        "error": {
            "type": type(error).__name__,
            "message": str(error)[:4_096],
            "traceback": traceback.format_exc(limit=12)[-16_384:],
        },
    }


def _candidate_commitment(candidate: object) -> str:
    method = getattr(candidate, "checkpoint_commitment", None)
    if not callable(method):
        raise CandidateProtocolError(
            "candidate must implement checkpoint_commitment() -> sha256"
        )
    return _validate_digest(method(), "checkpoint commitment")


def _child_main(
    connection: Connection,
    commitment: CandidateArtifactCommitment,
    staged: _StagedCandidateArtifact,
    max_message_bytes: int,
    max_requests: int,
) -> None:
    candidate: object | None = None
    began = False
    frozen: str | None = None
    requests = 0
    try:
        # The child independently authenticates the exact snapshot it will
        # import.  No candidate byte executes before this verification.
        _verify_staged_candidate(commitment, staged)
        factory = _resolve_staged_entrypoint(commitment, staged)
        candidate = factory()  # type: ignore[operator]
        if not isinstance(candidate, InteractiveGrounder):
            raise TypeError("candidate does not implement InteractiveGrounder")
        while True:
            try:
                payload = connection.recv_bytes(maxlength=max_message_bytes)
            except (EOFError, OSError):
                break
            try:
                envelope = _exact(
                    _decode_json(payload, max_message_bytes),
                    {"protocol", "id", "command", "arguments"},
                    "request",
                )
                if envelope["protocol"] != ISOLATION_PROTOCOL:
                    raise CandidateProtocolError("unsupported candidate RPC version")
                request_id = envelope["id"]
                command = envelope["command"]
                arguments = envelope["arguments"]
                if isinstance(request_id, bool) or not isinstance(request_id, int):
                    raise CandidateProtocolError("request id must be an integer")
                if request_id != requests:
                    raise CandidateProtocolError("request ids must be consecutive from zero")
                if not isinstance(command, str) or command not in _COMMANDS:
                    raise CandidateProtocolError("command is not allowlisted")
                if not isinstance(arguments, list):
                    raise CandidateProtocolError("arguments must be a list")
                requests += 1
                if requests > max_requests:
                    raise CandidateProtocolError("candidate request budget exhausted")
                if command == "close":
                    if arguments:
                        raise CandidateProtocolError("close takes no arguments")
                    connection.send_bytes(
                        _encode_json(_response(request_id, result=None), max_message_bytes)
                    )
                    break
                if command == "begin":
                    if began or frozen is not None or len(arguments) != 1:
                        raise CandidateProtocolError("begin must occur exactly once first")
                    manifest = _unwire(arguments[0])
                    if not isinstance(manifest, SessionManifest):
                        raise CandidateProtocolError("begin requires SessionManifest")
                    candidate.begin(manifest)
                    began = True
                    result: JSONValue | None = None
                elif not began:
                    raise CandidateProtocolError("begin must precede every other command")
                elif command == "observe_support":
                    if frozen is not None or len(arguments) != 2:
                        raise CandidateProtocolError(
                            "observe_support is pre-freeze and takes turn+trace"
                        )
                    turn, trace = (_unwire(item) for item in arguments)
                    if not isinstance(turn, PublicTurn) or not isinstance(trace, PublicTrace):
                        raise CandidateProtocolError(
                            "observe_support requires PublicTurn and PublicTrace"
                        )
                    candidate.observe_support(turn, trace)
                    result = None
                elif command == "choose_experiment":
                    if frozen is not None or len(arguments) != 1:
                        raise CandidateProtocolError(
                            "choose_experiment is pre-freeze and takes one turn"
                        )
                    turn = _unwire(arguments[0])
                    if not isinstance(turn, PublicTurn):
                        raise CandidateProtocolError("choose_experiment requires PublicTurn")
                    if turn.phase is not SessionPhase.ACQUISITION:
                        raise CandidateProtocolError(
                            "choose_experiment requires an acquisition turn"
                        )
                    if turn.scalar_feedback is not None:
                        raise CandidateProtocolError(
                            "acquisition turns cannot contain scalar feedback"
                        )
                    decision = candidate.choose_experiment(turn)
                    if not isinstance(decision, ExperimentDecision):
                        raise CandidateProtocolError(
                            "choose_experiment must return ExperimentDecision"
                        )
                    result = _wire(decision)
                elif command == "observe_experiment":
                    if frozen is not None or len(arguments) != 2:
                        raise CandidateProtocolError(
                            "observe_experiment is pre-freeze and takes turn+transition"
                        )
                    turn = _unwire(arguments[0])
                    transition = _unwire(arguments[1])
                    if not isinstance(turn, PublicTurn) or not isinstance(
                        transition, PublicTransition
                    ):
                        raise CandidateProtocolError(
                            "observe_experiment requires PublicTurn and PublicTransition"
                        )
                    if turn.phase is not SessionPhase.ACQUISITION:
                        raise CandidateProtocolError(
                            "observe_experiment requires an acquisition turn"
                        )
                    if turn.scalar_feedback is not None:
                        raise CandidateProtocolError(
                            "acquisition turns cannot contain scalar feedback"
                        )
                    if transition.scalar_feedback is not None:
                        raise CandidateProtocolError(
                            "acquisition transitions cannot contain scalar feedback"
                        )
                    if transition.before != turn.observation:
                        raise CandidateProtocolError(
                            "experiment transition.before must equal turn.observation"
                        )
                    candidate.observe_experiment(turn, transition)
                    result = None
                elif command == "freeze":
                    if frozen is not None or arguments:
                        raise CandidateProtocolError("freeze occurs exactly once")
                    candidate.freeze()
                    frozen = _candidate_commitment(candidate)
                    result = frozen
                elif command == "checkpoint":
                    if frozen is None or arguments:
                        raise CandidateProtocolError("checkpoint requires a frozen candidate")
                    current = _candidate_commitment(candidate)
                    if current != frozen:
                        raise CandidateProtocolError("candidate mutated after freeze")
                    result = current
                elif command in _QUERY_COMMANDS:
                    if frozen is None:
                        raise CandidateProtocolError(f"{command} requires freeze")
                    before = _candidate_commitment(candidate)
                    if before != frozen:
                        raise CandidateProtocolError("candidate mutated after freeze")
                    if command == "describe":
                        if len(arguments) != 1:
                            raise CandidateProtocolError("describe takes one trace")
                        trace = _unwire(arguments[0])
                        if not isinstance(trace, PublicTrace) or trace.has_feedback:
                            raise CandidateProtocolError(
                                "describe requires a feedback-free PublicTrace"
                            )
                        decision = candidate.describe(trace)
                        expected = DescriptionDecision
                        result = _wire(decision)
                    elif command == "begin_goal":
                        if len(arguments) != 2:
                            raise CandidateProtocolError(
                                "begin_goal takes utterance and observation trace"
                            )
                        utterance = _unwire(arguments[0])
                        observation_trace = _unwire(arguments[1])
                        if not isinstance(utterance, Utterance) or not isinstance(
                            observation_trace, PublicTrace
                        ) or observation_trace.transitions:
                            raise CandidateProtocolError(
                                "begin_goal requires Utterance and empty PublicTrace"
                            )
                        candidate.begin_goal(utterance, observation_trace.initial)
                        expected = None
                        result = None
                    elif command == "act":
                        if len(arguments) != 1:
                            raise CandidateProtocolError("act takes one observation trace")
                        observation_trace = _unwire(arguments[0])
                        if not isinstance(observation_trace, PublicTrace) or (
                            observation_trace.transitions
                        ):
                            raise CandidateProtocolError(
                                "act requires an empty PublicTrace observation wrapper"
                            )
                        decision = candidate.act(observation_trace.initial)
                        expected = ActionDecision
                        result = _wire(decision)
                    else:
                        if len(arguments) != 1 or not isinstance(arguments[0], list):
                            raise CandidateProtocolError(
                                "report_belief takes one integer candidate list"
                            )
                        candidates = arguments[0]
                        if any(isinstance(item, bool) or not isinstance(item, int) for item in candidates):
                            raise CandidateProtocolError("belief candidates must be integers")
                        if len(candidates) > MAX_BELIEF_CANDIDATES:
                            raise CandidateProtocolError(
                                "belief candidate list exceeds the protocol limit"
                            )
                        if len(set(candidates)) != len(candidates):
                            raise CandidateProtocolError(
                                "belief candidates must be unique"
                            )
                        decision = candidate.report_belief(tuple(candidates))
                        if isinstance(decision, BeliefDecision) and not {
                            candidate_id
                            for candidate_id, _probability in (
                                decision.candidate_probabilities
                            )
                        }.issubset(candidates):
                            raise CandidateProtocolError(
                                "belief decision contains an unrequested candidate"
                            )
                        expected = BeliefDecision
                        result = _wire(decision)
                    if expected is not None and not isinstance(decision, expected):
                        raise CandidateProtocolError(
                            f"{command} must return {expected.__name__}"
                        )
                    after = _candidate_commitment(candidate)
                    if after != frozen:
                        raise CandidateProtocolError("candidate mutated during sealed query")
                else:  # pragma: no cover - allowlist and branches are exhaustive
                    raise CandidateProtocolError("unsupported command")
                response = _response(request_id, result=result)
            except BaseException as exc:  # return a bounded remote error and terminate
                response = _response(requests - 1 if requests else 0, error=exc)
                connection.send_bytes(_encode_json(response, max_message_bytes))
                break
            connection.send_bytes(_encode_json(response, max_message_bytes))
    except BaseException as exc:
        try:
            connection.send_bytes(
                _encode_json(_response(0, error=exc), max_message_bytes)
            )
        except BaseException:
            pass
    finally:
        connection.close()


class IsolatedGrounder:
    """Parent-side proxy for one persistent, precommitted candidate instance."""

    def __init__(
        self,
        commitment: CandidateArtifactCommitment,
        *,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_requests: int = DEFAULT_MAX_REQUESTS,
        max_message_bytes: int = MAX_MESSAGE_BYTES,
    ) -> None:
        if not isinstance(commitment, CandidateArtifactCommitment):
            raise TypeError("commitment must be CandidateArtifactCommitment")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be numeric")
        if not isfinite(float(timeout)) or timeout <= 0.0:
            raise ValueError("timeout must be finite and positive")
        if isinstance(max_requests, bool) or not isinstance(max_requests, int):
            raise TypeError("max_requests must be an integer")
        if max_requests < 1:
            raise ValueError("max_requests must be positive")
        if isinstance(max_message_bytes, bool) or not isinstance(max_message_bytes, int):
            raise TypeError("max_message_bytes must be an integer")
        if not 1_024 <= max_message_bytes <= MAX_MESSAGE_BYTES:
            raise ValueError(
                f"max_message_bytes must lie in [1024, {MAX_MESSAGE_BYTES}]"
            )
        self.commitment = commitment
        self.timeout = float(timeout)
        self.max_requests = max_requests
        self.max_message_bytes = max_message_bytes
        self._process: Any = None
        self._connection: Connection | None = None
        self._request_id = 0
        self._closed = False
        self._frozen: FrozenCandidate | None = None
        self._stage_directory: TemporaryDirectory[str] | None = None

    @property
    def frozen(self) -> FrozenCandidate | None:
        return self._frozen

    @property
    def request_count(self) -> int:
        return self._request_id

    def start(self) -> "IsolatedGrounder":
        if self._closed:
            raise CandidateIsolationError("candidate proxy is closed")
        if self._process is not None:
            raise CandidateIsolationError("candidate process is already started")
        # Rehash live files, then execute only a private snapshot built from
        # those same verified bytes.  Subsequent live-path mutation cannot
        # change what the spawned process imports.
        verified = _verified_candidate_bytes(
            self.commitment,
            verify_import_resolution=True,
        )
        directory, staged = _stage_candidate_artifact(self.commitment, verified)
        parent: Connection | None = None
        child: Connection | None = None
        process: Any = None
        try:
            _verify_staged_candidate(self.commitment, staged)
            context = get_context("spawn")
            parent, child = context.Pipe(duplex=True)
            process = context.Process(
                target=_child_main,
                args=(
                    child,
                    self.commitment,
                    staged,
                    self.max_message_bytes,
                    self.max_requests,
                ),
                daemon=True,
            )
            process.start()
            child.close()
        except BaseException:
            if child is not None:
                try:
                    child.close()
                except OSError:
                    pass
            if parent is not None:
                try:
                    parent.close()
                except OSError:
                    pass
            if process is not None and process.pid is not None:
                process.join(_JOIN_GRACE_SECONDS)
                if process.is_alive():
                    process.terminate()
                    process.join(_JOIN_GRACE_SECONDS)
                if process.is_alive():
                    process.kill()
                    process.join(_JOIN_GRACE_SECONDS)
            self._cleanup_stage(directory)
            raise
        self._stage_directory = directory
        self._process = process
        self._connection = parent
        return self

    def __enter__(self) -> "IsolatedGrounder":
        return self.start()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback_value: TracebackType | None,
    ) -> None:
        self.close()

    def _call(self, command: str, arguments: list[JSONValue]) -> JSONValue:
        if self._closed:
            raise CandidateIsolationError("candidate proxy is closed")
        if self._process is None or self._connection is None:
            raise CandidateIsolationError("candidate process has not been started")
        if self._request_id >= self.max_requests:
            raise CandidateProtocolError("candidate request budget exhausted")
        request_id = self._request_id
        self._request_id += 1
        payload = _encode_json(
            _request(request_id, command, arguments), self.max_message_bytes
        )
        try:
            self._connection.send_bytes(payload)
        except (BrokenPipeError, EOFError, OSError) as exc:
            raise CandidateIsolationError("candidate pipe closed during send") from exc
        deadline = time.monotonic() + self.timeout
        while not self._connection.poll(max(0.0, deadline - time.monotonic())):
            self._stop()
            raise CandidateTimeoutError(
                f"candidate did not answer {command!r} within {self.timeout:g}s"
            )
        try:
            raw = self._connection.recv_bytes(maxlength=self.max_message_bytes)
        except (EOFError, OSError) as exc:
            raise CandidateIsolationError("candidate pipe closed during receive") from exc
        response = _exact(
            _decode_json(raw, self.max_message_bytes),
            {"protocol", "id", "ok", "result", "error"},
            "response",
        )
        if response["protocol"] != ISOLATION_PROTOCOL or response["id"] != request_id:
            raise CandidateProtocolError("candidate response does not match request")
        if response["ok"] is True and response["error"] is None:
            return response["result"]
        error = _exact(response["error"], {"type", "message", "traceback"}, "error")
        raise CandidateExecutionError(
            str(error["type"]), str(error["message"]), str(error["traceback"])
        )

    @staticmethod
    def _decision(value: JSONValue, expected: type[Any]) -> Any:
        result = _unwire(value)
        if not isinstance(result, expected):
            raise CandidateProtocolError(
                f"candidate returned {type(result).__name__}; expected {expected.__name__}"
            )
        return result

    def begin(self, manifest: SessionManifest) -> None:
        self._call("begin", [_wire(manifest)])

    def observe_support(self, turn: PublicTurn, trace: PublicTrace) -> None:
        self._call("observe_support", [_wire(turn), _wire(trace)])

    def choose_experiment(self, turn: PublicTurn) -> ExperimentDecision:
        if not isinstance(turn, PublicTurn):
            raise TypeError("turn must be PublicTurn")
        if turn.phase is not SessionPhase.ACQUISITION:
            raise CandidateProtocolError("choose_experiment requires an acquisition turn")
        if turn.scalar_feedback is not None:
            raise CandidateProtocolError(
                "acquisition turns cannot contain scalar feedback"
            )
        return self._decision(
            self._call("choose_experiment", [_wire(turn)]), ExperimentDecision
        )

    def observe_experiment(
        self,
        turn: PublicTurn,
        transition: PublicTransition,
    ) -> None:
        if not isinstance(turn, PublicTurn):
            raise TypeError("turn must be PublicTurn")
        if not isinstance(transition, PublicTransition):
            raise TypeError("transition must be PublicTransition")
        if turn.phase is not SessionPhase.ACQUISITION:
            raise CandidateProtocolError("observe_experiment requires an acquisition turn")
        if turn.scalar_feedback is not None:
            raise CandidateProtocolError(
                "acquisition turns cannot contain scalar feedback"
            )
        if transition.scalar_feedback is not None:
            raise CandidateProtocolError(
                "acquisition transitions cannot contain scalar feedback"
            )
        if transition.before != turn.observation:
            raise CandidateProtocolError(
                "experiment transition.before must equal turn.observation"
            )
        self._call("observe_experiment", [_wire(turn), _wire(transition)])

    def freeze(self) -> FrozenCandidate:
        value = self._call("freeze", [])
        checkpoint = _validate_digest(value, "checkpoint commitment")
        frozen = FrozenCandidate(
            self.commitment.digest,
            checkpoint,
            self.request_count,
        )
        self._frozen = frozen
        return frozen

    def assert_frozen(self) -> str:
        value = self._call("checkpoint", [])
        checkpoint = _validate_digest(value, "checkpoint commitment")
        if self._frozen is None or checkpoint != self._frozen.checkpoint_commitment:
            raise CandidateProtocolError("candidate checkpoint differs from frozen record")
        return checkpoint

    def describe(self, trace: PublicTrace) -> DescriptionDecision:
        return self._decision(
            self._call("describe", [_wire(trace.feedback_stripped())]),
            DescriptionDecision,
        )

    def begin_goal(self, utterance: Utterance, observation: Observation) -> None:
        self._call("begin_goal", [_wire(utterance), _wire(PublicTrace(observation))])

    def act(self, observation: Observation) -> ActionDecision:
        return self._decision(
            self._call("act", [_wire(PublicTrace(observation))]), ActionDecision
        )

    def report_belief(self, candidates: Sequence[int]) -> BeliefDecision:
        values = list(candidates)
        if any(isinstance(item, bool) or not isinstance(item, int) for item in values):
            raise TypeError("belief candidates must be integers")
        if len(values) > MAX_BELIEF_CANDIDATES:
            raise CandidateProtocolError(
                "belief candidate list exceeds the protocol limit"
            )
        if len(set(values)) != len(values):
            raise CandidateProtocolError("belief candidates must be unique")
        decision = self._decision(
            self._call("report_belief", [values]), BeliefDecision
        )
        if not {
            candidate_id
            for candidate_id, _probability in decision.candidate_probabilities
        }.issubset(values):
            raise CandidateProtocolError(
                "belief decision contains an unrequested candidate"
            )
        return decision

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

    @staticmethod
    def _cleanup_stage(directory: TemporaryDirectory[str]) -> None:
        root = Path(directory.name)
        if root.exists():
            for path in root.rglob("*"):
                try:
                    if path.is_symlink():
                        continue
                    os.chmod(path, 0o700 if path.is_dir() else 0o600)
                except OSError:
                    pass
            try:
                os.chmod(root, 0o700)
            except OSError:
                pass
        directory.cleanup()

    def close(self) -> None:
        if self._closed:
            return
        if self._connection is not None and self._process is not None:
            try:
                if self._process.is_alive() and self._request_id < self.max_requests:
                    self._call("close", [])
            except (CandidateIsolationError, CandidateTimeoutError):
                pass
        self._closed = True
        if self._connection is not None:
            self._connection.close()
            self._connection = None
        self._stop()
        if self._stage_directory is not None:
            self._cleanup_stage(self._stage_directory)
            self._stage_directory = None


__all__ = [
    "CandidateArtifactCommitment",
    "CandidateExecutionError",
    "CandidateIsolationError",
    "CandidateProtocolError",
    "CandidateTimeoutError",
    "DEFAULT_MAX_REQUESTS",
    "DEFAULT_TIMEOUT_SECONDS",
    "FrozenCandidate",
    "ISOLATION_PROTOCOL",
    "IsolatedGrounder",
    "commit_candidate_artifact",
    "verify_candidate_artifact",
]
