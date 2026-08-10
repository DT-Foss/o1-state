"""LIVE-CAUSAL segment store.

Implements the append-only, sealed, content-addressed segment layer
described in analysis/LIVE_CAUSAL_SPEC.md (SS1, SS3).

A segment is a sealed batch of triplet records. Its canonical serialization
is JSON-Lines over the records (sort_keys=True, ensure_ascii=False, one
record per line, \\n-terminated); sha256 over those exact bytes names the
segment file (`<sha256>.seg`). The file itself carries one header line
({"version": 2, "count": N, "sha256": hex}) before the record lines, so a
segment is self-describing without touching the manifest.

The manifest (`manifest.json`) is the ordered list of segment hashes plus
free-form meta, written atomically (tmp file + os.replace).
"""

import hashlib
import json
import os
import tempfile

MANIFEST_VERSION = 2
SEGMENT_VERSION = 2
MANIFEST_NAME = "manifest.json"


def _canonical_line(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"


def canonical_bytes(records):
    """Canonical serialization of a list of records: JSON-Lines, sorted keys,
    non-ASCII kept literal, one record per line, newline-terminated.
    """
    return "".join(_canonical_line(r) for r in records).encode("utf-8")


def segment_sha(records):
    """sha256 hex digest over the canonical bytes of records."""
    return hashlib.sha256(canonical_bytes(records)).hexdigest()


def _atomic_write(path, data_bytes):
    directory = os.path.dirname(path) or "."
    fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


class LiveStore:
    """Append-only, content-addressed segment store.

    dir_path holds one manifest.json plus one <sha256>.seg file per segment.
    """

    def __init__(self, dir_path):
        self.dir_path = dir_path
        self.mount()

    def mount(self):
        """Idempotent: ensure the store directory and manifest exist."""
        os.makedirs(self.dir_path, exist_ok=True)
        manifest_path = self._manifest_path()
        if not os.path.exists(manifest_path):
            self._write_manifest({"version": MANIFEST_VERSION, "segments": [], "meta": {}})

    def _manifest_path(self):
        return os.path.join(self.dir_path, MANIFEST_NAME)

    def _segment_path(self, sha):
        return os.path.join(self.dir_path, "{}.seg".format(sha))

    def _read_manifest(self):
        with open(self._manifest_path(), "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_manifest(self, manifest):
        data = json.dumps(manifest, sort_keys=True, ensure_ascii=False, indent=2).encode("utf-8")
        _atomic_write(self._manifest_path(), data)

    def append_segment(self, records):
        """Seal records into a new segment file and append its hash to the
        manifest. Returns the segment's sha256 hex digest.

        Idempotent on identical content: re-appending records that hash to
        an existing segment leaves the manifest order untouched (no
        duplicate entry) but still verifies the segment file is intact.
        """
        records = list(records)
        body = canonical_bytes(records)
        sha = hashlib.sha256(body).hexdigest()

        seg_path = self._segment_path(sha)
        if not os.path.exists(seg_path):
            header = _canonical_line(
                {"version": SEGMENT_VERSION, "count": len(records), "sha256": sha}
            ).encode("utf-8")
            _atomic_write(seg_path, header + body)

        manifest = self._read_manifest()
        if sha not in manifest["segments"]:
            manifest["segments"].append(sha)
            self._write_manifest(manifest)

        return sha

    def drop_segments(self, shas):
        """Remove segments (by sha) from the manifest and delete their files.
        Unknown shas are ignored.
        """
        shas = set(shas)
        manifest = self._read_manifest()
        manifest["segments"] = [s for s in manifest["segments"] if s not in shas]
        self._write_manifest(manifest)
        for sha in shas:
            seg_path = self._segment_path(sha)
            if os.path.exists(seg_path):
                os.remove(seg_path)

    def segments(self):
        """Ordered list of segment sha256 hashes, per the manifest."""
        return list(self._read_manifest()["segments"])

    def _load_segment_records(self, sha):
        seg_path = self._segment_path(sha)
        with open(seg_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if not lines:
            raise ValueError("empty segment file: {}".format(seg_path))
        records = []
        for line in lines[1:]:
            line = line.rstrip("\n")
            if line == "":
                continue
            records.append(json.loads(line))
        return records

    def iter_records(self, sha=None):
        """Yield (sha, idx, record) for all segments in manifest order, or
        for a single segment when sha is given.
        """
        shas = [sha] if sha is not None else self.segments()
        for s in shas:
            records = self._load_segment_records(s)
            for idx, record in enumerate(records):
                yield (s, idx, record)

    def verify(self, sha=None):
        """Recompute sha256 over the canonical bytes of a segment's records
        and compare against the file name (and its own header claim).
        Verifies all segments in the manifest when sha is None.
        """
        shas = [sha] if sha is not None else self.segments()
        for s in shas:
            seg_path = self._segment_path(s)
            if not os.path.exists(seg_path):
                return False
            try:
                records = self._load_segment_records(s)
            except (ValueError, json.JSONDecodeError):
                return False
            recomputed = segment_sha(records)
            if recomputed != s:
                return False
            with open(seg_path, "r", encoding="utf-8") as f:
                header = json.loads(f.readline())
            if header.get("sha256") != s or header.get("count") != len(records):
                return False
        return True
