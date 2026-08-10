"""LIVE-CAUSAL evidence calculus.

Implements analysis/EVIDENCE_CALCULUS_DRAFT.md (Lead-reviewed, accepted
2026-08-10), the three binding recommendations:

  1. Evidence counting = distinct source-document identity extracted from
     doc_coord (fallback: a W-window distance heuristic when doc_coord is
     a flat int with no document-id prefix; fallback-of-fallback: segment
     provenance when neither applies), stored as a separate append-only
     EVIDENCE LEDGER, never mutating sealed segments. evidence_count is a
     pure fold over the ledger; dead lines (citing a dropped segment) are
     filtered at read time, exactly like inferred-edge invalidation (SS2).
  2. Conflicts always coexist in storage. dominance() is a read-time ratio
     over ledger-derived evidence counts -- never a write-time deletion.
     Inference (src/livecausal/infer.py) propagates through contested
     edges unconditionally and tags inherited contested status; this
     module computes `contested` at read time from the ledger, infer.py
     only carries the flag through derivations (see contested_for_derivation
     below, the one hook infer.py calls).
  3. use_count gets the same append-only-ledger treatment (separate file,
     logical sequence numbers, never wall-clock time), kept as a distinct
     number from evidence_count.

Neither ledger lives inside store.py or infer.py's data model -- both are
independent append-only logs, keyed by (edge_key, ...), that cite base
records by (segment_sha, idx) the same way inferred-edge derivations do.
A stranger can recompute evidence_count/use_count/dominance/contested from
the ledgers + the store alone (P60 pattern, extended per the draft's
closing paragraph).

edge_key: this module uses infer.py's actual graph-edge identity,
(from_key, to_key) -- i.e. (trigger_key, outcome_key) -- NOT the draft's
full (trigger_key, mechanism, outcome_key) triple. This is a deliberate,
flagged interpretation (see the build report): infer.py's base-edge
adjacency is keyed purely on (from_key, to_key) with no mechanism
dimension (_record_base_edge in infer.py extracts only trigger_key/
outcome_key), so aligning edge_key with that identity is what makes this
module attach to infer.py's edges "minimal-invasively," per the build
brief, without inventing a second edge-identity axis infer.py does not
have. A useful side effect: two records citing the same (from_key,
to_key) pair with DIFFERENT mechanisms collide on the same edge_key and
are exactly what the draft's SS2 "contradiction" case describes (two
claims about the same node pair, incompatible content) -- conflict
detection falls out of the existing key rather than needing a new one.
"""

import hashlib
import json
import os
import tempfile

EVIDENCE_LEDGER_NAME = "evidence.ledger"
USE_LEDGER_NAME = "use.ledger"
LEDGER_VERSION = 1

# SS2.2's proposed defaults, explicitly flagged in the draft as knobs, not
# asserted-correct constants -- kept here as named module constants so a
# caller (or a future measured registration) can override without editing
# read-time call sites throughout.
DEFAULT_DOMINANCE_RATIO = 2.0
DEFAULT_DOMINANCE_FLOOR = 2

# Default doc-distance window (SS1a Option A / the Lead's W review note),
# used ONLY as the fallback when doc_coord is a flat int with no
# resolvable document-id prefix (see _evidence_key_for_record). Written
# into every evidence ledger's header so the fold stays reproducible from
# the file alone even if a future caller changes this module constant.
DEFAULT_DOC_WINDOW_W = 50


def _canonical_line(record):
    # Mirrors store.py's private _canonical_line (JSON-Lines, sorted keys,
    # non-ASCII kept literal) without importing store's private helper.
    return json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n"


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


def canonical_bytes(lines):
    """Canonical serialization of a list of ledger line-dicts: JSON-Lines,
    sorted keys, non-ASCII kept literal, newline-terminated. Same
    discipline as store.py's canonical_bytes, applied to ledger lines."""
    return "".join(_canonical_line(r) for r in lines).encode("utf-8")


def ledger_sha256(lines):
    return hashlib.sha256(canonical_bytes(lines)).hexdigest()


def edge_key_to_list(edge_key):
    """edge_key is (from_key, to_key); ledger lines store it as a 2-list
    (JSON has no tuples) -- this is the canonical on-disk shape."""
    return [edge_key[0], edge_key[1]]


def edge_key_from_list(lst):
    return (lst[0], lst[1])


# ─────────────────────────────────────────────────────────────────────────
#  Document identity / evidence-key extraction from doc_coord (SS1a).
# ─────────────────────────────────────────────────────────────────────────
def _evidence_key_for_record(record, sha, idx, window_w):
    """Returns (evidence_key, method) for one record, per SS1a's
    recommendation C with B as the documented fallback, extended with the
    W-window heuristic for the flat-int doc_coord case actually in use
    today (see module docstring: doc_coord is currently always a flat int
    in this codebase's live-causal records, not yet the structured
    (lane_seed, episode, frame, offset) list form SS1a's Option C
    describes for pixel-world coordinates).

    Resolution order:
      1. doc_coord is a list/tuple of length >= 2: the document identity is
         everything but the last (finest-resolution) component -- the
         structured-coordinate case SS1a Option C names directly.
      2. doc_coord is an int (or a length-1 list): no document-id prefix
         exists to extract. Bucket it into a window of width window_w
         (evidence_key = doc_coord // window_w) -- this is SS1a Option A,
         used here ONLY as the documented fallback the Lead's review note
         requires: window_w is written into the ledger header, never a
         silent constant.
      3. doc_coord missing/unusable: SS1a Option B, segment provenance --
         evidence_key = the citing segment's sha (at most one evidence
         unit per segment for this edge, regardless of how many times the
         edge recurs inside it).
    """
    coord = record.get("doc_coord")
    if isinstance(coord, (list, tuple)) and len(coord) >= 2:
        return (tuple(coord[:-1]), "doc_prefix")
    if isinstance(coord, (int, float)):
        return (int(coord) // window_w, "window")
    if isinstance(coord, (list, tuple)) and len(coord) == 1:
        return (int(coord[0]) // window_w, "window")
    return (sha, "segment")


# ─────────────────────────────────────────────────────────────────────────
#  Evidence ledger
# ─────────────────────────────────────────────────────────────────────────
class EvidenceLedger:
    """Append-only observation log: one line per (edge, independent
    source) sighting. Lives beside manifest.json in the store directory.

    Line shape: {"edge_key": [from_key, to_key], "evidence_key": ...,
                 "segment": sha, "idx": i}
    Header (line 0): {"version": 1, "kind": "evidence", "w": window_w}

    evidence_count is a pure fold: distinct evidence_key values seen for
    an edge_key, restricted to lines whose segment is still present in
    the store (valid_segments) -- dead lines are filtered at read time,
    never rewritten (SS1b, mirrors infer.py's on_drop invalidation
    philosophy without touching infer.py's own state).
    """

    def __init__(self, store_dir, window_w=DEFAULT_DOC_WINDOW_W):
        self.store_dir = store_dir
        self.path = os.path.join(store_dir, EVIDENCE_LEDGER_NAME)
        self.window_w = window_w
        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.path):
            # Header already committed to disk; window_w for THIS ledger
            # is whatever its header says (self-describing, per the Lead's
            # review note), not necessarily the constructor argument --
            # re-read it so a stale in-memory default never wins.
            with open(self.path, "r", encoding="utf-8") as f:
                header_line = f.readline()
            if header_line:
                header = json.loads(header_line)
                self.window_w = header.get("w", self.window_w)
            return
        header = {"version": LEDGER_VERSION, "kind": "evidence", "w": self.window_w}
        _atomic_write(self.path, _canonical_line(header).encode("utf-8"))

    def append_observation(self, edge_key, record, sha, idx):
        """Records one sighting of edge_key from the given (sha, idx)
        base record. Idempotent in effect (not in file size): appending
        the identical (edge_key, evidence_key, segment, idx) line twice
        just adds a duplicate line, which the fold below already
        deduplicates via `set` over evidence_key -- so double-observation
        (e.g. a defensive re-run) cannot inflate evidence_count."""
        evidence_key, method = _evidence_key_for_record(record, sha, idx, self.window_w)
        line = {
            "edge_key": edge_key_to_list(edge_key),
            "evidence_key": list(evidence_key) if isinstance(evidence_key, tuple) else evidence_key,
            "segment": sha,
            "idx": idx,
            "method": method,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canonical_line(line))
        return line

    def append_observations_for_segment(self, graph, sha):
        """Convenience: walk every record in segment `sha` (via the
        LiveGraph's store) and append one evidence-ledger observation per
        record that encodes a base edge. Mirrors infer.py's on_append
        shape (iterate a segment's records once) without touching
        infer.py itself."""
        appended = []
        for _seg_sha, idx, record in graph.store.iter_records(sha):
            from_key = record.get("trigger_key")
            to_key = record.get("outcome_key")
            if from_key is None or to_key is None:
                continue
            edge_key = (from_key, to_key)
            appended.append(self.append_observation(edge_key, record, sha, idx))
        return appended

    def _iter_lines(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:  # skip header
            line = line.rstrip("\n")
            if line == "":
                continue
            yield json.loads(line)

    def evidence_count(self, edge_key, valid_segments):
        """Distinct evidence_key values observed for edge_key, counting
        only lines whose segment is in valid_segments (a set/iterable of
        segment shas still present in the store's manifest -- the caller
        supplies this, typically store.segments(), so this ledger never
        needs to mount a LiveStore itself)."""
        valid = set(valid_segments)
        target = edge_key_to_list(edge_key)
        seen = set()
        for line in self._iter_lines():
            if line["edge_key"] != target:
                continue
            if line["segment"] not in valid:
                continue
            ek = line["evidence_key"]
            seen.add(ek if not isinstance(ek, list) else tuple(ek))
        return len(seen)

    def all_edge_keys(self, valid_segments):
        """All distinct edge_keys with at least one valid (non-dropped)
        observation -- used by dominance() to enumerate contested pairs."""
        valid = set(valid_segments)
        keys = set()
        for line in self._iter_lines():
            if line["segment"] not in valid:
                continue
            keys.add(tuple(line["edge_key"]))
        return keys


# ─────────────────────────────────────────────────────────────────────────
#  Use ledger
# ─────────────────────────────────────────────────────────────────────────
class UseLedger:
    """Append-only log of successful consults against an edge. Separate
    file from EvidenceLedger (read-load isolation, per SS3's recommendation
    -- use events are expected far higher-frequency once the organism
    consults the graph live).

    Line shape: {"edge_key": [from_key, to_key], "seq": n,
                 "segment": sha, "idx": i}
    Header (line 0): {"version": 1, "kind": "use"}

    seq is a LOGICAL sequence number the caller owns and supplies --
    never wall-clock time (SS1's segment-header discipline extended here,
    per SS3). segment/idx cite the base record the consult resolved
    against, so a use-ledger line can be invalidated exactly like an
    evidence-ledger line when its segment is dropped.
    """

    def __init__(self, store_dir):
        self.store_dir = store_dir
        self.path = os.path.join(store_dir, USE_LEDGER_NAME)
        self._ensure_header()

    def _ensure_header(self):
        if os.path.exists(self.path):
            return
        header = {"version": LEDGER_VERSION, "kind": "use"}
        _atomic_write(self.path, _canonical_line(header).encode("utf-8"))

    def append_use(self, edge_key, seq, sha, idx):
        line = {
            "edge_key": edge_key_to_list(edge_key),
            "seq": int(seq),
            "segment": sha,
            "idx": idx,
        }
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(_canonical_line(line))
        return line

    def _iter_lines(self):
        if not os.path.exists(self.path):
            return
        with open(self.path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[1:]:
            line = line.rstrip("\n")
            if line == "":
                continue
            yield json.loads(line)

    def use_count(self, edge_key, valid_segments):
        valid = set(valid_segments)
        target = edge_key_to_list(edge_key)
        n = 0
        for line in self._iter_lines():
            if line["edge_key"] != target:
                continue
            if line["segment"] not in valid:
                continue
            n += 1
        return n

    def max_seq(self):
        """Highest seq ever appended (across ALL lines, valid or not --
        seq is the caller's own monotonic counter, unaffected by segment
        drops); 0 if the ledger is empty. Lets a caller resume numbering
        without re-scanning for a max externally."""
        m = 0
        for line in self._iter_lines():
            if line["seq"] > m:
                m = line["seq"]
        return m


# ─────────────────────────────────────────────────────────────────────────
#  Dominance (SS2): read-time ratio over ledger-derived evidence counts.
#  Never writes anything -- pure function of the evidence ledger.
# ─────────────────────────────────────────────────────────────────────────
def dominance(evidence_ledger, edge_key_a, edge_key_b, valid_segments):
    """Returns a dict describing the read-time dominance relationship
    between two edge_keys sharing the same node pair with conflicting
    mechanisms (SS2's contradiction case): {"count_a", "count_b", "ratio"}
    where ratio = max(count)/min(count) (None if the loser has 0 evidence
    -- an undefined ratio, not infinity, so callers don't need an isinf
    check). This function computes the RATIO only; it does not decide a
    winner -- that is is_dominant()'s job, parameterized by the knobs
    below (never resolved here into a boolean the store would "own")."""
    count_a = evidence_ledger.evidence_count(edge_key_a, valid_segments)
    count_b = evidence_ledger.evidence_count(edge_key_b, valid_segments)
    hi, lo = max(count_a, count_b), min(count_a, count_b)
    ratio = (hi / lo) if lo > 0 else None
    return {
        "edge_key_a": edge_key_a,
        "edge_key_b": edge_key_b,
        "count_a": count_a,
        "count_b": count_b,
        "ratio": ratio,
    }


def is_dominant(dom, ratio_threshold=DEFAULT_DOMINANCE_RATIO, floor=DEFAULT_DOMINANCE_FLOOR):
    """Applies the SS2.2 default policy (both explicitly flagged knobs, not
    asserted-correct constants -- pass your own ratio_threshold/floor to
    override) to a dominance() result: the winning side needs
    count >= floor AND count >= ratio_threshold * loser's count (the
    floor kills the degenerate "1 vs 0" false-dominance case the draft
    names). Returns the winning edge_key, or None if neither side
    dominates (contested / no decidable winner)."""
    count_a, count_b = dom["count_a"], dom["count_b"]
    hi_key, hi, lo = (
        (dom["edge_key_a"], count_a, count_b)
        if count_a >= count_b
        else (dom["edge_key_b"], count_b, count_a)
    )
    if hi < floor:
        return None
    if lo == 0:
        # Only "dominant" if the floor is cleared on the winning side AND
        # there IS a genuine conflict on record (lo == 0 with nothing on
        # the other side isn't really a contest at all -- see contested()
        # below, which requires both sides to have >=1 evidence line to
        # call something contested in the first place).
        return hi_key
    if hi >= ratio_threshold * lo:
        return hi_key
    return None


def contested(evidence_ledger, edge_key_a, edge_key_b, valid_segments):
    """True iff BOTH sides of a conflicting node-pair have at least one
    valid evidence observation (SS2: coexistence in storage, contested
    only means "both sides are real, decide later"). A one-sided edge
    with zero counter-evidence is not contested -- it just IS the graph's
    only claim for that node pair."""
    count_a = evidence_ledger.evidence_count(edge_key_a, valid_segments)
    count_b = evidence_ledger.evidence_count(edge_key_b, valid_segments)
    return count_a > 0 and count_b > 0


def contested_for_derivation(base_contested_lookup, derivation, edge_keys_by_hop):
    """SS2.3's inference-propagation rule, exposed as the ONE hook
    infer.py needs (not implemented inside infer.py itself, per the build
    brief's "minimal-invasive" instruction): an inferred edge is contested
    iff ANY base edge in its derivation chain is contested.

    base_contested_lookup: callable(edge_key) -> bool, typically a closure
        over an EvidenceLedger + a fixed valid_segments snapshot and a
        caller-supplied conflict-pairing (this module does not invent
        which edge_key is "the conflicting one" for another -- that is
        SS2's node-pair-with-incompatible-mechanism relationship, which
        requires mechanism data this module's edge_key intentionally
        collapses away; see the build report's flagged edge_key
        interpretation. A caller that HAS mechanism data can build this
        lookup; this function stays a pure boolean-OR over whatever the
        caller decides is contested.)
    derivation: list of [segment_sha, idx] pairs (an inferred edge's
        derivation, exactly infer.py's shape).
    edge_keys_by_hop: list of (from_key, to_key) edge_keys, one per hop in
        `derivation`, in the same order -- the caller (infer.py, or a
        wrapper around it) already knows each hop's base edge_key from the
        chain it walked to build the derivation; this function does not
        re-derive it, keeping inference's own re-derivation logic (SS2's
        "re-derivable by a stranger from the cited base edges alone")
        entirely inside infer.py, untouched.
    """
    return any(base_contested_lookup(ek) for ek in edge_keys_by_hop)
