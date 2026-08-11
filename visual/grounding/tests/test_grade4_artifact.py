from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
import os
import shutil
import stat
import tarfile
import warnings
import zipfile

import pytest

from grounding_kernel.grade4_artifact import (
    DIRECTORY_MODE,
    FILE_MODE,
    ArchivePolicy,
    ArtifactRejected,
    ArtifactVerificationError,
    SealedArtifact,
    ingest_opaque_submission,
    verify_sealed_artifact,
)


def _zip(
    entries: list[tuple[str, bytes]],
    *,
    compression: int = zipfile.ZIP_STORED,
) -> bytes:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=compression) as archive:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            for name, payload in entries:
                archive.writestr(name, payload)
    return buffer.getvalue()


def _zip_with_mode(name: str, payload: bytes, mode: int) -> bytes:
    buffer = BytesIO()
    info = zipfile.ZipInfo(name)
    info.create_system = 3
    info.external_attr = mode << 16
    with zipfile.ZipFile(buffer, mode="w") as archive:
        archive.writestr(info, payload)
    return buffer.getvalue()


def _tar(entries: list[tarfile.TarInfo], payloads: dict[str, bytes] | None = None) -> bytes:
    payloads = payloads or {}
    buffer = BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for info in entries:
            payload = payloads.get(info.name)
            archive.addfile(info, BytesIO(payload) if payload is not None else None)
    return buffer.getvalue()


def _regular_tar_info(name: str, payload: bytes) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    info.mode = 0o777
    return info


def _make_removable(root: Path) -> None:
    if not root.exists() or root.is_symlink():
        return
    for current, directories, _files in os.walk(root, topdown=True, followlinks=False):
        os.chmod(current, 0o700)
        for directory in directories:
            path = Path(current, directory)
            if not path.is_symlink():
                os.chmod(path, 0o700)
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture
def ingest(tmp_path: Path) -> Iterator[Callable[..., SealedArtifact]]:
    roots: list[Path] = []

    def run(source: object, *, policy: ArchivePolicy | None = None) -> SealedArtifact:
        kwargs = {} if policy is None else {"policy": policy}
        artifact = ingest_opaque_submission(source, tmp_path, **kwargs)  # type: ignore[arg-type]
        roots.append(artifact.root)
        return artifact

    yield run
    for root in roots:
        _make_removable(root)


def test_opaque_package_is_sealed_without_executing_init(
    tmp_path: Path, ingest: Callable[..., SealedArtifact]
) -> None:
    marker = tmp_path / "EXECUTED"
    source = (
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('candidate executed')\n"
    ).encode()
    archive = _zip(
        [
            ("candidate/", b""),
            ("candidate/__init__.py", source),
            ("candidate/model.bin", b"\x00\x01opaque\xff"),
        ]
    )

    artifact = ingest(archive)

    assert not marker.exists()
    assert (artifact.root / "candidate/__init__.py").read_bytes() == source
    assert (artifact.root / "candidate/model.bin").read_bytes() == b"\x00\x01opaque\xff"
    assert stat.S_IMODE(artifact.root.stat().st_mode) == DIRECTORY_MODE
    assert stat.S_IMODE((artifact.root / "candidate").stat().st_mode) == DIRECTORY_MODE
    assert stat.S_IMODE((artifact.root / "candidate/__init__.py").stat().st_mode) == FILE_MODE
    artifact.verify()


def test_path_source_is_stably_read_and_location_does_not_affect_manifest(
    tmp_path: Path, ingest: Callable[..., SealedArtifact]
) -> None:
    archive = _zip([("submission.py", b"VALUE = 7\n")])
    source = tmp_path / "submission.zip"
    source.write_bytes(archive)

    first = ingest(source)
    second = ingest(archive)

    assert first.root != second.root
    assert first.manifest == second.manifest
    assert first.manifest.archive_digest.value != first.manifest.tree_digest.value
    assert first.manifest.digest.value != first.manifest.archive_digest.value


def test_manifest_is_frozen_and_binds_archive_metadata_and_virtual_paths(
    ingest: Callable[..., SealedArtifact],
) -> None:
    plain = _zip([("a.py", b"same")])
    commented_buffer = BytesIO()
    with zipfile.ZipFile(commented_buffer, mode="w") as archive:
        archive.writestr("a.py", b"same")
        archive.comment = b"opaque metadata"
    commented = commented_buffer.getvalue()
    renamed = _zip([("b.py", b"same")])

    first = ingest(plain)
    second = ingest(commented)
    third = ingest(renamed)

    assert first.manifest.tree_digest == second.manifest.tree_digest
    assert first.manifest.archive_digest != second.manifest.archive_digest
    assert first.manifest.digest != second.manifest.digest
    assert first.manifest.tree_digest != third.manifest.tree_digest
    with pytest.raises(FrozenInstanceError):
        first.manifest.total_file_bytes = 99  # type: ignore[misc]


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        "a/../../escape.py",
        "./dot.py",
        "a//double.py",
        "a\\windows.py",
        "C:/drive.py",
        "control\x01.py",
    ],
)
def test_rejects_noncanonical_and_traversing_zip_paths(tmp_path: Path, name: str) -> None:
    with pytest.raises(ArtifactRejected):
        ingest_opaque_submission(_zip([(name, b"x")]), tmp_path)
    assert not (tmp_path.parent / "escape.py").exists()


def test_rejects_embedded_nul_in_raw_zip_name(tmp_path: Path) -> None:
    archive = _zip([("badXname.py", b"x")])
    poisoned = archive.replace(b"badXname.py", b"bad\x00name.py")
    assert poisoned != archive

    with pytest.raises(ArtifactRejected, match="NUL"):
        ingest_opaque_submission(poisoned, tmp_path)


def test_rejects_embedded_nul_hidden_by_tar_parser(tmp_path: Path) -> None:
    payload = b"x"
    info = _regular_tar_info("bad\x00hidden", payload)

    with pytest.raises(ArtifactRejected, match="hidden NUL"):
        ingest_opaque_submission(_tar([info], {info.name: payload}), tmp_path)


def test_rejects_duplicate_exact_paths(tmp_path: Path) -> None:
    archive = _zip([("same.py", b"first"), ("same.py", b"second")])

    with pytest.raises(ArtifactRejected, match="duplicate"):
        ingest_opaque_submission(archive, tmp_path)


@pytest.mark.parametrize(
    "entries",
    [
        [("A.py", b"a"), ("a.py", b"b")],
        [("é.py", b"a"), ("e\u0301.py", b"b")],
        [("Root/a.py", b"a"), ("root/b.py", b"b")],
    ],
)
def test_rejects_unicode_nfc_and_casefold_collisions(
    tmp_path: Path, entries: list[tuple[str, bytes]]
) -> None:
    with pytest.raises(ArtifactRejected):
        ingest_opaque_submission(_zip(entries), tmp_path)


def test_rejects_zip_symlink(tmp_path: Path) -> None:
    archive = _zip_with_mode("link", b"target", stat.S_IFLNK | 0o777)

    with pytest.raises(ArtifactRejected, match="symbolic"):
        ingest_opaque_submission(archive, tmp_path)


def test_rejects_tar_symbolic_and_hard_links(tmp_path: Path) -> None:
    payload = b"safe"
    regular = _regular_tar_info("target", payload)
    symbolic = tarfile.TarInfo("symbolic")
    symbolic.type = tarfile.SYMTYPE
    symbolic.linkname = "target"
    hard = tarfile.TarInfo("hard")
    hard.type = tarfile.LNKTYPE
    hard.linkname = "target"

    for link in (symbolic, hard):
        archive = _tar([regular, link], {"target": payload})
        with pytest.raises(ArtifactRejected, match="links"):
            ingest_opaque_submission(archive, tmp_path)


@pytest.mark.parametrize("member_type", [tarfile.FIFOTYPE, tarfile.CHRTYPE, tarfile.BLKTYPE])
def test_rejects_tar_special_files(tmp_path: Path, member_type: bytes) -> None:
    special = tarfile.TarInfo("device")
    special.type = member_type

    with pytest.raises(ArtifactRejected, match="special"):
        ingest_opaque_submission(_tar([special]), tmp_path)


def _set_zip_encryption_bits(archive: bytes) -> bytes:
    result = bytearray(archive)
    local = result.find(b"PK\x03\x04")
    central = result.find(b"PK\x01\x02")
    assert local >= 0 and central >= 0
    local_flags = int.from_bytes(result[local + 6 : local + 8], "little") | 1
    central_flags = int.from_bytes(result[central + 8 : central + 10], "little") | 1
    result[local + 6 : local + 8] = local_flags.to_bytes(2, "little")
    result[central + 8 : central + 10] = central_flags.to_bytes(2, "little")
    return bytes(result)


def test_rejects_encrypted_zip_metadata(tmp_path: Path) -> None:
    encrypted = _set_zip_encryption_bits(_zip([("secret.py", b"opaque")]))

    with pytest.raises(ArtifactRejected, match="encrypted"):
        ingest_opaque_submission(encrypted, tmp_path)


def test_rejects_size_count_and_compression_bombs(tmp_path: Path) -> None:
    too_many = ArchivePolicy(max_files=1)
    with pytest.raises(ArtifactRejected, match="file count"):
        ingest_opaque_submission(_zip([("a", b"a"), ("b", b"b")]), tmp_path, policy=too_many)

    too_large = ArchivePolicy(max_file_bytes=3, max_total_file_bytes=3)
    with pytest.raises(ArtifactRejected, match="file-size"):
        ingest_opaque_submission(_zip([("large", b"1234")]), tmp_path, policy=too_large)

    ratio_limited = ArchivePolicy(max_compression_ratio=2.0)
    bomb = _zip([("zeros", b"\x00" * 100_000)], compression=zipfile.ZIP_DEFLATED)
    with pytest.raises(ArtifactRejected, match="compression-ratio"):
        ingest_opaque_submission(bomb, tmp_path, policy=ratio_limited)


def test_rejects_compressed_tar_before_decompression(tmp_path: Path) -> None:
    buffer = BytesIO()
    payload = b"x"
    info = _regular_tar_info("file", payload)
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        archive.addfile(info, BytesIO(payload))

    with pytest.raises(ArtifactRejected, match="compressed TAR"):
        ingest_opaque_submission(buffer.getvalue(), tmp_path)


def test_rejects_archive_source_symlink(tmp_path: Path) -> None:
    real = tmp_path / "real.zip"
    real.write_bytes(_zip([("file", b"x")]))
    alias = tmp_path / "alias.zip"
    alias.symlink_to(real)

    with pytest.raises(ArtifactRejected, match="opened safely"):
        ingest_opaque_submission(alias, tmp_path)


def test_corrupt_payload_is_never_partially_published(tmp_path: Path) -> None:
    archive = _zip([("file", b"UNIQUE_PAYLOAD")])
    corrupted = archive.replace(b"UNIQUE_PAYLOAD", b"UNIQUE_PAYLOAE", 1)
    assert corrupted != archive

    with pytest.raises(ArtifactRejected, match="payload"):
        ingest_opaque_submission(corrupted, tmp_path)

    assert list(tmp_path.iterdir()) == []


def test_tar_regular_files_are_manually_extracted_and_normalized(
    ingest: Callable[..., SealedArtifact],
) -> None:
    payload = b"#!/bin/sh\nexit 99\n"
    directory = tarfile.TarInfo("pkg")
    directory.type = tarfile.DIRTYPE
    directory.mode = 0o777
    regular = _regular_tar_info("pkg/run.sh", payload)

    artifact = ingest(_tar([directory, regular], {"pkg/run.sh": payload}))

    assert artifact.manifest.archive_format == "tar"
    assert (artifact.root / "pkg/run.sh").read_bytes() == payload
    assert stat.S_IMODE((artifact.root / "pkg/run.sh").stat().st_mode) == FILE_MODE
    verify_sealed_artifact(artifact)


def test_verification_fails_on_content_mutation(
    ingest: Callable[..., SealedArtifact],
) -> None:
    artifact = ingest(_zip([("file", b"original")]))
    target = artifact.root / "file"
    target.chmod(0o644)
    target.write_bytes(b"mutated!")
    target.chmod(FILE_MODE)

    with pytest.raises(ArtifactVerificationError, match="content"):
        artifact.verify()


def test_verification_fails_on_missing_and_extra_nodes(
    ingest: Callable[..., SealedArtifact],
) -> None:
    missing = ingest(_zip([("file", b"x")]))
    missing.root.chmod(0o755)
    (missing.root / "file").unlink()
    missing.root.chmod(DIRECTORY_MODE)
    with pytest.raises(ArtifactVerificationError, match="namespace"):
        missing.verify()

    extra = ingest(_zip([("file", b"x")]))
    extra.root.chmod(0o755)
    (extra.root / "extra").write_bytes(b"intruder")
    extra.root.chmod(DIRECTORY_MODE)
    with pytest.raises(ArtifactVerificationError, match="namespace"):
        extra.verify()


def test_verification_fails_on_file_directory_and_root_mode_drift(
    ingest: Callable[..., SealedArtifact],
) -> None:
    file_drift = ingest(_zip([("pkg/file", b"x")]))
    (file_drift.root / "pkg/file").chmod(0o644)
    with pytest.raises(ArtifactVerificationError, match="mode"):
        file_drift.verify()

    directory_drift = ingest(_zip([("pkg/file", b"x")]))
    (directory_drift.root / "pkg").chmod(0o755)
    with pytest.raises(ArtifactVerificationError, match="directory mode"):
        directory_drift.verify()

    root_drift = ingest(_zip([("file", b"x")]))
    root_drift.root.chmod(0o755)
    with pytest.raises(ArtifactVerificationError, match="root mode"):
        root_drift.verify()


def test_verification_rejects_post_seal_hardlink(
    tmp_path: Path, ingest: Callable[..., SealedArtifact]
) -> None:
    artifact = ingest(_zip([("file", b"x")]))
    outside = tmp_path / "outside-hardlink"
    os.link(artifact.root / "file", outside)

    with pytest.raises(ArtifactVerificationError, match="link count"):
        artifact.verify()
    outside.unlink()
