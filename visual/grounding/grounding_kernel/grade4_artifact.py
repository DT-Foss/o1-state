"""Opaque, evaluator-owned ingestion for untrusted Grade-4 submissions.

This module is deliberately lexical: submission bytes are never imported,
executed, or passed to a general-purpose object deserializer.  Supported
containers are ZIP and uncompressed TAR.  Extraction is manual and publishes
only a fully written, read-only tree by an atomic same-filesystem rename.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import BinaryIO, Literal
import json
import os
import secrets
import shutil
import stat
import tarfile
import tempfile
import unicodedata
import zipfile


ARTIFACT_SCHEMA_VERSION = "grounding-grade4-opaque-artifact/1"
FILE_MODE = 0o444
DIRECTORY_MODE = 0o555
_EMPTY_DIGEST = sha256(b"").hexdigest()
_COPY_CHUNK = 1024 * 1024
_ZIP_METHODS = frozenset({zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED})
_HARD_MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
_HARD_MAX_FILES = 4_096
_HARD_MAX_DIRECTORIES = 4_096
_HARD_MAX_FILE_BYTES = 64 * 1024 * 1024
_HARD_MAX_TOTAL_FILE_BYTES = 256 * 1024 * 1024
_HARD_MAX_COMPRESSION_RATIO = 100.0
_HARD_MAX_PATH_BYTES = 1_024
_HARD_MAX_PATH_DEPTH = 32


class Grade4ArtifactError(RuntimeError):
    """Base class for fail-closed artifact errors."""


class ArtifactRejected(Grade4ArtifactError):
    """The untrusted archive does not satisfy the ingestion policy."""


class ArtifactVerificationError(Grade4ArtifactError):
    """A previously sealed tree no longer matches its manifest."""


@dataclass(frozen=True, slots=True, order=True)
class Sha256Digest:
    """A strongly typed, canonical SHA-256 digest."""

    value: str

    def __post_init__(self) -> None:
        if len(self.value) != 64 or self.value != self.value.lower():
            raise ValueError("SHA-256 digest must contain 64 lowercase hexadecimal digits")
        try:
            bytes.fromhex(self.value)
        except ValueError as exc:
            raise ValueError("SHA-256 digest must be hexadecimal") from exc

    @classmethod
    def of(cls, value: bytes) -> Sha256Digest:
        return cls(sha256(value).hexdigest())

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, slots=True)
class ArchivePolicy:
    """Resource and namespace limits applied before publication."""

    max_archive_bytes: int = _HARD_MAX_ARCHIVE_BYTES
    max_files: int = _HARD_MAX_FILES
    max_directories: int = _HARD_MAX_DIRECTORIES
    max_file_bytes: int = _HARD_MAX_FILE_BYTES
    max_total_file_bytes: int = _HARD_MAX_TOTAL_FILE_BYTES
    max_compression_ratio: float = _HARD_MAX_COMPRESSION_RATIO
    max_path_bytes: int = _HARD_MAX_PATH_BYTES
    max_path_depth: int = _HARD_MAX_PATH_DEPTH

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_archive_bytes,
            self.max_files,
            self.max_directories,
            self.max_file_bytes,
            self.max_total_file_bytes,
            self.max_path_bytes,
            self.max_path_depth,
        )
        if any(isinstance(value, bool) or value <= 0 for value in integer_limits):
            raise ValueError("artifact policy integer limits must be positive")
        if not isinstance(self.max_compression_ratio, (int, float)) or isinstance(
            self.max_compression_ratio, bool
        ):
            raise ValueError("max_compression_ratio must be numeric")
        if not (1.0 <= float(self.max_compression_ratio) < float("inf")):
            raise ValueError("max_compression_ratio must be finite and at least one")
        hard_limits = (
            (self.max_archive_bytes, _HARD_MAX_ARCHIVE_BYTES),
            (self.max_files, _HARD_MAX_FILES),
            (self.max_directories, _HARD_MAX_DIRECTORIES),
            (self.max_file_bytes, _HARD_MAX_FILE_BYTES),
            (self.max_total_file_bytes, _HARD_MAX_TOTAL_FILE_BYTES),
            (self.max_path_bytes, _HARD_MAX_PATH_BYTES),
            (self.max_path_depth, _HARD_MAX_PATH_DEPTH),
        )
        if any(value > maximum for value, maximum in hard_limits):
            raise ValueError("artifact policy may tighten but not exceed hard safety ceilings")
        if float(self.max_compression_ratio) > _HARD_MAX_COMPRESSION_RATIO:
            raise ValueError("compression-ratio policy exceeds the hard safety ceiling")

    def canonical_payload(self) -> dict[str, int | float]:
        return {
            "max_archive_bytes": self.max_archive_bytes,
            "max_compression_ratio": float(self.max_compression_ratio),
            "max_directories": self.max_directories,
            "max_file_bytes": self.max_file_bytes,
            "max_files": self.max_files,
            "max_path_bytes": self.max_path_bytes,
            "max_path_depth": self.max_path_depth,
            "max_total_file_bytes": self.max_total_file_bytes,
        }


DEFAULT_ARCHIVE_POLICY = ArchivePolicy()


@dataclass(frozen=True, slots=True, order=True)
class ArtifactEntry:
    """One canonical node in the published tree."""

    path: str
    kind: Literal["directory", "file"]
    size: int
    mode: int
    content_digest: Sha256Digest

    def __post_init__(self) -> None:
        canonical = _canonical_path(self.path, DEFAULT_ARCHIVE_POLICY)
        if canonical != self.path:
            raise ValueError("artifact entry path is not canonical")
        if self.kind not in {"directory", "file"}:
            raise ValueError("artifact entry kind is invalid")
        if isinstance(self.size, bool) or self.size < 0:
            raise ValueError("artifact entry size must be nonnegative")
        if self.kind == "directory":
            if self.size != 0 or self.mode != DIRECTORY_MODE:
                raise ValueError("directory entries require canonical size and mode")
            if self.content_digest.value != _EMPTY_DIGEST:
                raise ValueError("directory entries require the empty-content digest")
        elif self.mode != FILE_MODE:
            raise ValueError("file entries require canonical read-only mode")


def _policy_digest(policy: ArchivePolicy) -> Sha256Digest:
    payload = json.dumps(
        policy.canonical_payload(), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return Sha256Digest.of(b"grade4-policy\0" + payload)


def _tree_digest(entries: tuple[ArtifactEntry, ...]) -> Sha256Digest:
    digest = sha256(b"grounding-grade4-tree/1\0")
    for entry in entries:
        path = entry.path.encode("utf-8")
        digest.update(len(path).to_bytes(8, "big"))
        digest.update(path)
        digest.update(b"D" if entry.kind == "directory" else b"F")
        digest.update(entry.size.to_bytes(8, "big"))
        digest.update(entry.mode.to_bytes(4, "big"))
        digest.update(bytes.fromhex(entry.content_digest.value))
    return Sha256Digest(digest.hexdigest())


def _manifest_digest(
    archive_format: str,
    archive_digest: Sha256Digest,
    policy_digest: Sha256Digest,
    entries: tuple[ArtifactEntry, ...],
    total_file_bytes: int,
    tree_digest: Sha256Digest,
) -> Sha256Digest:
    payload = {
        "archive_digest": archive_digest.value,
        "archive_format": archive_format,
        "entries": [
            {
                "content_digest": entry.content_digest.value,
                "kind": entry.kind,
                "mode": entry.mode,
                "path": entry.path,
                "size": entry.size,
            }
            for entry in entries
        ],
        "policy_digest": policy_digest.value,
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "total_file_bytes": total_file_bytes,
        "tree_digest": tree_digest.value,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return Sha256Digest.of(b"grounding-grade4-manifest/1\0" + encoded)


@dataclass(frozen=True, slots=True)
class Grade4ArtifactManifest:
    """Location-independent evidence for every archive byte and tree node."""

    archive_format: Literal["tar", "zip"]
    archive_digest: Sha256Digest
    policy_digest: Sha256Digest
    entries: tuple[ArtifactEntry, ...]
    total_file_bytes: int
    tree_digest: Sha256Digest
    digest: Sha256Digest
    schema_version: str = ARTIFACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ARTIFACT_SCHEMA_VERSION:
            raise ValueError("artifact manifest schema version is unsupported")
        if self.archive_format not in {"tar", "zip"}:
            raise ValueError("artifact manifest archive format is unsupported")
        if not self.entries:
            raise ValueError("artifact manifest must be nonempty")
        if tuple(sorted(self.entries, key=lambda entry: entry.path)) != self.entries:
            raise ValueError("artifact manifest entries must be path-sorted")
        paths = [entry.path for entry in self.entries]
        if len(paths) != len(set(paths)):
            raise ValueError("artifact manifest paths must be unique")
        _validate_manifest_namespace(self.entries)
        expected_total = sum(entry.size for entry in self.entries if entry.kind == "file")
        if self.total_file_bytes != expected_total:
            raise ValueError("artifact manifest total size is inconsistent")
        expected_tree = _tree_digest(self.entries)
        if self.tree_digest != expected_tree:
            raise ValueError("artifact manifest tree digest is inconsistent")
        expected = _manifest_digest(
            self.archive_format,
            self.archive_digest,
            self.policy_digest,
            self.entries,
            self.total_file_bytes,
            self.tree_digest,
        )
        if self.digest != expected:
            raise ValueError("artifact manifest digest is inconsistent")


@dataclass(frozen=True, slots=True)
class SealedArtifact:
    """A published read-only tree plus its immutable manifest."""

    root: Path
    manifest: Grade4ArtifactManifest

    def verify(self) -> None:
        verify_sealed_artifact(self)


@dataclass(frozen=True, slots=True)
class _Member:
    path: str
    kind: Literal["directory", "file"]
    size: int
    compressed_size: int
    ordinal: int


def _canonical_path(value: str, policy: ArchivePolicy) -> str:
    if not isinstance(value, str) or not value:
        raise ArtifactRejected("archive member path must be nonempty text")
    if "\x00" in value:
        raise ArtifactRejected("archive member path contains NUL")
    if "\\" in value:
        raise ArtifactRejected("archive member path contains a backslash")
    if value.startswith("/") or PurePosixPath(value).is_absolute():
        raise ArtifactRejected("archive member path is absolute")
    if PureWindowsPath(value).drive:
        raise ArtifactRejected("archive member path contains a drive prefix")
    if value.endswith("/") or "//" in value:
        raise ArtifactRejected("archive member path is not canonical")
    parts = value.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ArtifactRejected("archive member path traverses or contains dot segments")
    if len(parts) > policy.max_path_depth:
        raise ArtifactRejected("archive member path exceeds the depth limit")
    if unicodedata.normalize("NFC", value) != value:
        raise ArtifactRejected("archive member path is not Unicode NFC")
    if any(
        any(unicodedata.category(character).startswith("C") for character in part) for part in parts
    ):
        raise ArtifactRejected("archive member path contains control characters")
    encoded = value.encode("utf-8")
    if len(encoded) > policy.max_path_bytes:
        raise ArtifactRejected("archive member path exceeds the byte limit")
    if any(len(part.encode("utf-8")) > 255 for part in parts):
        raise ArtifactRejected("archive member path component exceeds 255 bytes")
    return value


def _collision_key(path: str) -> str:
    return unicodedata.normalize("NFC", path).casefold()


def _validate_namespace(members: list[_Member], policy: ArchivePolicy) -> tuple[str, ...]:
    leaves: dict[str, str] = {}
    spellings: dict[str, str] = {}
    directories: set[str] = set()
    for member in members:
        if member.path in leaves:
            raise ArtifactRejected(f"duplicate archive member path: {member.path}")
        leaves[member.path] = member.kind
        parts = member.path.split("/")
        for index in range(1, len(parts) + 1):
            prefix = "/".join(parts[:index])
            key = _collision_key(prefix)
            prior = spellings.get(key)
            if prior is not None and prior != prefix:
                raise ArtifactRejected(
                    f"Unicode NFC/casefold path collision: {prior!r} and {prefix!r}"
                )
            spellings[key] = prefix
            if index < len(parts):
                directories.add(prefix)
    for path, kind in leaves.items():
        if kind == "file" and path in directories:
            raise ArtifactRejected(f"file shadows a required directory: {path}")
        parent = PurePosixPath(path).parent
        while str(parent) != ".":
            parent_path = parent.as_posix()
            if leaves.get(parent_path) == "file":
                raise ArtifactRejected(f"file shadows a required directory: {parent_path}")
            directories.add(parent_path)
            parent = parent.parent
    directories.update(path for path, kind in leaves.items() if kind == "directory")
    if len(directories) > policy.max_directories:
        raise ArtifactRejected("archive directory count exceeds the policy")
    return tuple(sorted(directories))


def _validate_manifest_namespace(entries: tuple[ArtifactEntry, ...]) -> None:
    by_path = {entry.path: entry.kind for entry in entries}
    collision: dict[str, str] = {}
    for entry in entries:
        key = _collision_key(entry.path)
        prior = collision.get(key)
        if prior is not None and prior != entry.path:
            raise ValueError("artifact manifest contains a casefold collision")
        collision[key] = entry.path
        parent = PurePosixPath(entry.path).parent
        while str(parent) != ".":
            if by_path.get(parent.as_posix()) != "directory":
                raise ValueError("artifact manifest omits a parent directory")
            parent = parent.parent


def _stable_read(path: Path, max_bytes: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactRejected(f"archive path cannot be opened safely: {path}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactRejected("archive source must be a regular file")
        if before.st_size > max_bytes:
            raise ArtifactRejected("archive source exceeds the byte limit")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(_COPY_CHUNK, max_bytes - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > max_bytes:
                raise ArtifactRejected("archive source exceeds the byte limit")
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or total != before.st_size:
        raise ArtifactRejected("archive source changed while it was read")
    return b"".join(chunks)


def _source_bytes(
    source: bytes | bytearray | memoryview | str | os.PathLike[str], policy: ArchivePolicy
) -> bytes:
    if isinstance(source, bytes):
        result = source
    elif isinstance(source, (bytearray, memoryview)):
        result = bytes(source)
    elif isinstance(source, (str, os.PathLike)):
        result = _stable_read(Path(source), policy.max_archive_bytes)
    else:
        raise TypeError("archive source must be bytes-like or a filesystem path")
    if not result:
        raise ArtifactRejected("archive source is empty")
    if len(result) > policy.max_archive_bytes:
        raise ArtifactRejected("archive source exceeds the byte limit")
    return result


def _zip_members(data: bytes, policy: ArchivePolicy) -> tuple[list[_Member], tuple[str, ...]]:
    if not data.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        raise ArtifactRejected("ZIP archive has a noncanonical preamble")
    _preflight_zip_directory(data, policy)
    try:
        with zipfile.ZipFile(BytesIO(data), mode="r") as archive:
            infos = archive.infolist()
    except (OSError, ValueError, zipfile.BadZipFile, NotImplementedError) as exc:
        raise ArtifactRejected("ZIP archive metadata is invalid") from exc
    members: list[_Member] = []
    total_size = 0
    total_compressed = 0
    file_count = 0
    for ordinal, info in enumerate(infos):
        original = info.orig_filename
        if "\x00" in original or original != info.filename:
            raise ArtifactRejected("ZIP member path contains NUL")
        if info.flag_bits & 0x1:
            raise ArtifactRejected("encrypted ZIP members are forbidden")
        is_directory = info.is_dir()
        raw_path = original[:-1] if is_directory else original
        path = _canonical_path(raw_path, policy)
        unix_mode = (info.external_attr >> 16) & 0xFFFF
        file_type = stat.S_IFMT(unix_mode)
        if file_type == stat.S_IFLNK:
            raise ArtifactRejected("ZIP symbolic links are forbidden")
        if is_directory:
            if file_type not in {0, stat.S_IFDIR} or info.file_size != 0:
                raise ArtifactRejected("ZIP directory metadata is inconsistent")
            kind: Literal["directory", "file"] = "directory"
        else:
            if file_type not in {0, stat.S_IFREG}:
                raise ArtifactRejected("ZIP special files are forbidden")
            kind = "file"
            file_count += 1
            if file_count > policy.max_files:
                raise ArtifactRejected("archive file count exceeds the policy")
            if info.compress_type not in _ZIP_METHODS:
                raise ArtifactRejected("ZIP compression method is not allowed")
            if info.file_size > policy.max_file_bytes:
                raise ArtifactRejected("archive member exceeds the file-size limit")
            if info.compress_size == 0 and info.file_size > 0:
                raise ArtifactRejected("ZIP member has an invalid compressed size")
            ratio = info.file_size / max(info.compress_size, 1)
            if ratio > policy.max_compression_ratio:
                raise ArtifactRejected("ZIP member exceeds the compression-ratio limit")
            total_size += info.file_size
            total_compressed += info.compress_size
            if total_size > policy.max_total_file_bytes:
                raise ArtifactRejected("archive exceeds the total file-size limit")
        members.append(_Member(path, kind, info.file_size, info.compress_size, ordinal))
    if file_count == 0:
        raise ArtifactRejected("archive must contain at least one regular file")
    if total_size / max(total_compressed, 1) > policy.max_compression_ratio:
        raise ArtifactRejected("ZIP archive exceeds the aggregate compression-ratio limit")
    directories = _validate_namespace(members, policy)
    return members, directories


def _preflight_zip_directory(data: bytes, policy: ArchivePolicy) -> None:
    minimum_eocd = 22
    search_start = max(0, len(data) - (minimum_eocd + 65_535))
    offset = data.rfind(b"PK\x05\x06", search_start)
    if offset < 0 or offset + minimum_eocd > len(data):
        raise ArtifactRejected("ZIP end-of-central-directory record is missing")
    disk = int.from_bytes(data[offset + 4 : offset + 6], "little")
    directory_disk = int.from_bytes(data[offset + 6 : offset + 8], "little")
    entries_on_disk = int.from_bytes(data[offset + 8 : offset + 10], "little")
    entries_total = int.from_bytes(data[offset + 10 : offset + 12], "little")
    directory_size = int.from_bytes(data[offset + 12 : offset + 16], "little")
    directory_offset = int.from_bytes(data[offset + 16 : offset + 20], "little")
    comment_size = int.from_bytes(data[offset + 20 : offset + 22], "little")
    if offset + minimum_eocd + comment_size != len(data):
        raise ArtifactRejected("ZIP has trailing or malformed opaque data")
    if disk != 0 or directory_disk != 0 or entries_on_disk != entries_total:
        raise ArtifactRejected("multi-disk ZIP archives are forbidden")
    if entries_total == 0xFFFF or directory_size == 0xFFFFFFFF or directory_offset == 0xFFFFFFFF:
        raise ArtifactRejected("ZIP64 archives are outside the Grade-4 ingestion profile")
    if entries_total > policy.max_files + policy.max_directories:
        raise ArtifactRejected("ZIP central-directory entry count exceeds the policy")
    if directory_offset + directory_size > offset:
        raise ArtifactRejected("ZIP central-directory bounds are inconsistent")


def _tar_number(field: bytes) -> int:
    if not field:
        raise ArtifactRejected("TAR numeric field is empty")
    if field[0] & 0x80:
        if field[0] & 0x40:
            raise ArtifactRejected("negative TAR sizes are forbidden")
        return int.from_bytes(bytes((field[0] & 0x7F,)) + field[1:], "big")
    value = field.rstrip(b"\x00 ").lstrip(b" ")
    if not value:
        return 0
    if any(character not in b"01234567" for character in value):
        raise ArtifactRejected("TAR numeric field is not canonical octal")
    return int(value, 8)


def _reject_hidden_nul(field: bytes, label: str) -> None:
    position = field.find(b"\x00")
    if position >= 0 and any(field[position + 1 :]):
        raise ArtifactRejected(f"TAR {label} contains a hidden NUL suffix")


def _validate_raw_tar_paths(data: bytes, policy: ArchivePolicy) -> None:
    """Reject path bytes that tarfile would silently truncate at NUL."""

    if len(data) % tarfile.BLOCKSIZE != 0:
        raise ArtifactRejected("TAR archive is not block-aligned")
    offset = 0
    saw_header = False
    header_count = 0
    max_headers = policy.max_files + policy.max_directories + 128
    while offset + tarfile.BLOCKSIZE <= len(data):
        header = data[offset : offset + tarfile.BLOCKSIZE]
        if not any(header):
            if any(data[offset:]):
                raise ArtifactRejected("TAR has nonzero data after its end marker")
            break
        saw_header = True
        header_count += 1
        if header_count > max_headers:
            raise ArtifactRejected("TAR header count exceeds the policy")
        _reject_hidden_nul(header[0:100], "name")
        _reject_hidden_nul(header[345:500], "prefix")
        _reject_hidden_nul(header[157:257], "link name")
        size = _tar_number(header[124:136])
        payload_start = offset + tarfile.BLOCKSIZE
        payload_end = payload_start + size
        if payload_end > len(data):
            raise ArtifactRejected("TAR member payload exceeds the archive")
        type_flag = header[156:157]
        if type_flag in {b"x", b"g"} and b"\x00" in data[payload_start:payload_end]:
            raise ArtifactRejected("TAR PAX metadata contains NUL")
        if type_flag in {b"L", b"K"}:
            payload = data[payload_start:payload_end]
            position = payload.find(b"\x00")
            if position >= 0 and any(payload[position + 1 :]):
                raise ArtifactRejected("TAR GNU path metadata contains a hidden NUL suffix")
        blocks = (size + tarfile.BLOCKSIZE - 1) // tarfile.BLOCKSIZE
        offset = payload_start + blocks * tarfile.BLOCKSIZE
    if not saw_header:
        raise ArtifactRejected("TAR archive contains no headers")


def _tar_members(data: bytes, policy: ArchivePolicy) -> tuple[list[_Member], tuple[str, ...]]:
    if data.startswith((b"\x1f\x8b", b"BZh", b"\xfd7zXZ\x00")):
        raise ArtifactRejected("compressed TAR is forbidden; submit ZIP or uncompressed TAR")
    _validate_raw_tar_paths(data, policy)
    try:
        with tarfile.open(fileobj=BytesIO(data), mode="r:") as archive:
            infos = archive.getmembers()
    except (OSError, ValueError, tarfile.TarError) as exc:
        raise ArtifactRejected("TAR archive metadata is invalid") from exc
    members: list[_Member] = []
    total_size = 0
    file_count = 0
    for ordinal, info in enumerate(infos):
        raw_path = info.name[:-1] if info.isdir() and info.name.endswith("/") else info.name
        path = _canonical_path(raw_path, policy)
        if info.issym() or info.islnk():
            raise ArtifactRejected("TAR symbolic and hard links are forbidden")
        if info.issparse():
            raise ArtifactRejected("TAR sparse files are forbidden")
        if info.isdir():
            if info.size != 0:
                raise ArtifactRejected("TAR directory metadata is inconsistent")
            kind: Literal["directory", "file"] = "directory"
            size = 0
        elif info.isreg():
            kind = "file"
            size = info.size
            file_count += 1
            if file_count > policy.max_files:
                raise ArtifactRejected("archive file count exceeds the policy")
            if size > policy.max_file_bytes:
                raise ArtifactRejected("archive member exceeds the file-size limit")
            total_size += size
            if total_size > policy.max_total_file_bytes:
                raise ArtifactRejected("archive exceeds the total file-size limit")
        else:
            raise ArtifactRejected("TAR special files are forbidden")
        members.append(_Member(path, kind, size, size, ordinal))
    if file_count == 0:
        raise ArtifactRejected("archive must contain at least one regular file")
    directories = _validate_namespace(members, policy)
    return members, directories


def _inspect_archive(
    data: bytes, policy: ArchivePolicy
) -> tuple[Literal["tar", "zip"], list[_Member], tuple[str, ...]]:
    if data.startswith((b"PK\x03\x04", b"PK\x05\x06")):
        members, directories = _zip_members(data, policy)
        return "zip", members, directories
    members, directories = _tar_members(data, policy)
    return "tar", members, directories


def _write_all(descriptor: int, chunk: bytes) -> None:
    remaining = memoryview(chunk)
    while remaining:
        written = os.write(descriptor, remaining)
        if written <= 0:
            raise ArtifactRejected("short write while materializing artifact")
        remaining = remaining[written:]


def _materialize_file(source: BinaryIO, target: Path, expected_size: int) -> Sha256Digest:
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    try:
        descriptor = os.open(target, flags, 0o600)
    except OSError as exc:
        raise ArtifactRejected(f"cannot create canonical artifact member: {target.name}") from exc
    digest = sha256()
    total = 0
    try:
        while True:
            chunk = source.read(min(_COPY_CHUNK, expected_size - total + 1))
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise ArtifactRejected("archive member expands beyond its declared size")
            digest.update(chunk)
            _write_all(descriptor, chunk)
        if total != expected_size:
            raise ArtifactRejected("archive member is shorter than its declared size")
        os.fsync(descriptor)
        os.fchmod(descriptor, FILE_MODE)
    finally:
        os.close(descriptor)
    return Sha256Digest(digest.hexdigest())


def _extract(
    data: bytes,
    archive_format: Literal["tar", "zip"],
    members: list[_Member],
    directories: tuple[str, ...],
    root: Path,
) -> tuple[ArtifactEntry, ...]:
    for directory in sorted(directories, key=lambda value: (value.count("/"), value)):
        (root / directory).mkdir(mode=0o700)
    entries_by_path: dict[str, ArtifactEntry] = {
        directory: ArtifactEntry(
            directory, "directory", 0, DIRECTORY_MODE, Sha256Digest(_EMPTY_DIGEST)
        )
        for directory in directories
    }
    try:
        if archive_format == "zip":
            with zipfile.ZipFile(BytesIO(data), mode="r") as archive:
                infos = archive.infolist()
                for member in members:
                    if member.kind == "directory":
                        continue
                    with archive.open(infos[member.ordinal], mode="r") as source:
                        digest = _materialize_file(source, root / member.path, member.size)
                    entries_by_path[member.path] = ArtifactEntry(
                        member.path, "file", member.size, FILE_MODE, digest
                    )
        else:
            with tarfile.open(fileobj=BytesIO(data), mode="r:") as archive:
                infos = archive.getmembers()
                for member in members:
                    if member.kind == "directory":
                        continue
                    source = archive.extractfile(infos[member.ordinal])
                    if source is None:
                        raise ArtifactRejected("TAR regular file has no readable payload")
                    with source:
                        digest = _materialize_file(source, root / member.path, member.size)
                    entries_by_path[member.path] = ArtifactEntry(
                        member.path, "file", member.size, FILE_MODE, digest
                    )
    except (OSError, EOFError, RuntimeError, zipfile.BadZipFile, tarfile.TarError) as exc:
        if isinstance(exc, ArtifactRejected):
            raise
        raise ArtifactRejected("archive payload failed strict extraction") from exc
    for directory in sorted(directories, key=lambda value: (-value.count("/"), value)):
        path = root / directory
        os.chmod(path, DIRECTORY_MODE, follow_symlinks=False)
        _fsync_directory(path)
    os.chmod(root, DIRECTORY_MODE, follow_symlinks=False)
    _fsync_directory(root)
    return tuple(sorted(entries_by_path.values(), key=lambda entry: entry.path))


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _make_tree_removable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, _files in os.walk(root, topdown=True, followlinks=False):
        try:
            os.chmod(current, 0o700, follow_symlinks=False)
        except OSError:
            pass
        for directory in directories:
            path = Path(current, directory)
            if not path.is_symlink():
                try:
                    os.chmod(path, 0o700, follow_symlinks=False)
                except OSError:
                    pass


def ingest_opaque_submission(
    source: bytes | bytearray | memoryview | str | os.PathLike[str],
    destination_parent: str | os.PathLike[str],
    *,
    policy: ArchivePolicy = DEFAULT_ARCHIVE_POLICY,
) -> SealedArtifact:
    """Validate and atomically publish an opaque submission without importing it."""

    if not isinstance(policy, ArchivePolicy):
        raise TypeError("policy must be an ArchivePolicy")
    data = _source_bytes(source, policy)
    archive_digest = Sha256Digest.of(data)
    archive_format, members, directories = _inspect_archive(data, policy)
    raw_parent = Path(destination_parent)
    try:
        parent_metadata = raw_parent.lstat()
    except OSError as exc:
        raise ArtifactRejected("destination parent must already exist") from exc
    if stat.S_ISLNK(parent_metadata.st_mode) or not stat.S_ISDIR(parent_metadata.st_mode):
        raise ArtifactRejected("destination parent must be a real directory")
    parent = raw_parent.resolve(strict=True)
    temporary = Path(tempfile.mkdtemp(prefix=".grade4-ingest-", dir=parent))
    published: Path | None = None
    did_publish = False
    try:
        entries = _extract(data, archive_format, members, directories, temporary)
        total = sum(entry.size for entry in entries if entry.kind == "file")
        policy_digest = _policy_digest(policy)
        tree_digest = _tree_digest(entries)
        manifest_digest = _manifest_digest(
            archive_format,
            archive_digest,
            policy_digest,
            entries,
            total,
            tree_digest,
        )
        manifest = Grade4ArtifactManifest(
            archive_format,
            archive_digest,
            policy_digest,
            entries,
            total,
            tree_digest,
            manifest_digest,
        )
        for _attempt in range(16):
            candidate = parent / (f"artifact-{manifest.digest.value[:16]}-{secrets.token_hex(8)}")
            if not os.path.lexists(candidate):
                published = candidate
                break
        if published is None:
            raise ArtifactRejected("could not allocate a fresh artifact directory")
        os.rename(temporary, published)
        did_publish = True
        _fsync_directory(parent)
        artifact = SealedArtifact(published, manifest)
        verify_sealed_artifact(artifact)
        return artifact
    except BaseException:
        cleanup = published if did_publish and published is not None else temporary
        _make_tree_removable(cleanup)
        shutil.rmtree(cleanup, ignore_errors=True)
        raise


def _stable_file_digest(path: Path, expected_size: int) -> Sha256Digest:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ArtifactVerificationError(f"sealed file cannot be opened safely: {path}") from exc
    digest = sha256()
    total = 0
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise ArtifactVerificationError("sealed file type or link count changed")
        if stat.S_IMODE(before.st_mode) != FILE_MODE or before.st_size != expected_size:
            raise ArtifactVerificationError("sealed file mode or size changed")
        while True:
            chunk = os.read(descriptor, _COPY_CHUNK)
            if not chunk:
                break
            total += len(chunk)
            if total > expected_size:
                raise ArtifactVerificationError("sealed file grew during verification")
            digest.update(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    identity_before = (
        before.st_dev,
        before.st_ino,
        before.st_mode,
        before.st_nlink,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    identity_after = (
        after.st_dev,
        after.st_ino,
        after.st_mode,
        after.st_nlink,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if identity_before != identity_after or total != expected_size:
        raise ArtifactVerificationError("sealed file changed during verification")
    return Sha256Digest(digest.hexdigest())


def verify_sealed_tree(root: str | os.PathLike[str], manifest: Grade4ArtifactManifest) -> None:
    """Re-hash a sealed tree and fail on any content, path, type, or mode drift."""

    if not isinstance(manifest, Grade4ArtifactManifest):
        raise TypeError("manifest must be a Grade4ArtifactManifest")
    path = Path(root)
    try:
        before_root = path.lstat()
    except OSError as exc:
        raise ArtifactVerificationError("sealed artifact root is missing") from exc
    if not stat.S_ISDIR(before_root.st_mode) or stat.S_ISLNK(before_root.st_mode):
        raise ArtifactVerificationError("sealed artifact root is not a real directory")
    if stat.S_IMODE(before_root.st_mode) != DIRECTORY_MODE:
        raise ArtifactVerificationError("sealed artifact root mode changed")
    actual: dict[str, tuple[str, int]] = {}
    for current, directory_names, file_names in os.walk(path, topdown=True, followlinks=False):
        directory_names.sort()
        file_names.sort()
        current_path = Path(current)
        for name in directory_names:
            child = current_path / name
            metadata = child.lstat()
            relative = child.relative_to(path).as_posix()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
                raise ArtifactVerificationError(f"sealed directory type changed: {relative}")
            if stat.S_IMODE(metadata.st_mode) != DIRECTORY_MODE:
                raise ArtifactVerificationError(f"sealed directory mode changed: {relative}")
            actual[relative] = ("directory", 0)
        for name in file_names:
            child = current_path / name
            metadata = child.lstat()
            relative = child.relative_to(path).as_posix()
            if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
                raise ArtifactVerificationError(f"sealed file type changed: {relative}")
            actual[relative] = ("file", metadata.st_size)
    expected = {entry.path: (entry.kind, entry.size) for entry in manifest.entries}
    if actual != expected:
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ArtifactVerificationError(
            f"sealed artifact namespace changed (missing={missing}, extra={extra})"
        )
    observed_entries: list[ArtifactEntry] = []
    for entry in manifest.entries:
        node = path / entry.path
        if entry.kind == "directory":
            observed_entries.append(entry)
            continue
        content_digest = _stable_file_digest(node, entry.size)
        if content_digest != entry.content_digest:
            raise ArtifactVerificationError(f"sealed file content changed: {entry.path}")
        observed_entries.append(
            ArtifactEntry(entry.path, "file", entry.size, FILE_MODE, content_digest)
        )
    if _tree_digest(tuple(observed_entries)) != manifest.tree_digest:
        raise ArtifactVerificationError("sealed tree digest changed")
    after_root = path.lstat()
    root_identity_before = (
        before_root.st_dev,
        before_root.st_ino,
        before_root.st_mode,
        before_root.st_mtime_ns,
        before_root.st_ctime_ns,
    )
    root_identity_after = (
        after_root.st_dev,
        after_root.st_ino,
        after_root.st_mode,
        after_root.st_mtime_ns,
        after_root.st_ctime_ns,
    )
    if root_identity_before != root_identity_after:
        raise ArtifactVerificationError("sealed artifact root changed during verification")


def verify_sealed_artifact(artifact: SealedArtifact) -> None:
    if not isinstance(artifact, SealedArtifact):
        raise TypeError("artifact must be a SealedArtifact")
    verify_sealed_tree(artifact.root, artifact.manifest)


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "ArchivePolicy",
    "ArtifactEntry",
    "ArtifactRejected",
    "ArtifactVerificationError",
    "DEFAULT_ARCHIVE_POLICY",
    "DIRECTORY_MODE",
    "FILE_MODE",
    "Grade4ArtifactError",
    "Grade4ArtifactManifest",
    "SealedArtifact",
    "Sha256Digest",
    "ingest_opaque_submission",
    "verify_sealed_artifact",
    "verify_sealed_tree",
]
