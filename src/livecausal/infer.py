"""LIVE-CAUSAL delta-inference engine.

Implements SS2 of analysis/LIVE_CAUSAL_SPEC.md: incremental (semi-naive)
transitive closure over the base trigger_key -> outcome_key edges produced
by LiveStore records, never a full rebuild on append.

v1 mirror / flag (read per the build brief):
    vendor/fabel/language/hsslm_s/inference.py and
    vendor/fabel/dotcausal_package/src/dotcausal/inference.py both run a
    3-pass engine: Pass 1 exact keyword/adjacency chaining (transitive,
    depth-capped at 5 hops -- hsslm's `max_depth = 5` in
    `pass1_exact_chains`, dotcausal's DEFAULT_INFERENCE_RULES joins), Pass 2
    semantic direction propagation (classifies each mechanism string as
    positive/negative/neutral via `detect_mechanism_direction` and combines
    signs along the chain), Pass 3 Jaro-Winkler fuzzy entity matching.

    MIRRORED here: Pass 1's transitive rule, exactly --
        A -> B, B -> C  =>  A -> C
    walked over the trigger_key/outcome_key adjacency, depth-capped at 5
    (v1's `max_depth`), cycle-free (v1 skips a neighbor already in
    `current_path`; here a key may not repeat within one derivation chain).

    FLAGGED, not implemented (v0 scope): direction propagation (v1 Pass 2)
    and fuzzy matching (v1 Pass 3). Both are text-classification /
    string-similarity passes over mechanism/entity strings, not graph joins
    over the key adjacency this schema exposes (SPEC's inferred-edge schema
    is {from_key, to_key, depth, derivation} -- no direction or similarity
    field). Folding either in would (a) require classifying `mechanism`
    text per record, which the delta-closure/derivation-invalidation
    contract here says nothing about, and (b) make an inferred edge's
    identity depend on string-similarity thresholds computed over the
    *current* full key set -- breaking the "new base edge -> exactly the
    reachable delta" locality semi-naive evaluation depends on. Left for a
    v1 extension request if the team wants it; the transitive rule alone is
    what SS2 names explicitly ("hsslm's 5-hop closure").

Determinism: every public method that returns an edge collection returns a
list sorted by a total order over its fields (never dict/set iteration
order). Derivation lists are stored in chain order (root-to-leaf), which is
already deterministic per chain; the caller-visible sort key for an
inferred edge is (from_key, to_key, depth, derivation).
"""

import json
import os
import tempfile

from .store import LiveStore
from . import canon as canon_mod

INFERRED_NAME = "inferred.jsonl"
CANON_INFERRED_NAME = "canon_inferred.jsonl"
CANON_MAP_NAME = "canon_map.jsonl"
MAX_DEPTH = 5  # v1's max_depth (hsslm pass1_exact_chains), mirrored exactly.


def _canonical_line(record):
    # Mirrors store.py's private _canonical_line (JSON-Lines, sorted keys,
    # non-ASCII kept literal) without importing store's private helper --
    # store.py is not to be touched or leaned on beyond its public API.
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


def _derivation_key(derivation):
    # derivation is a list of [sha, idx] pairs (JSON-shaped); tuple-ify for
    # a stable, comparable sort key.
    return tuple((sha, idx) for sha, idx in derivation)


def _edge_sort_key(edge):
    return (
        edge["from_key"],
        edge["to_key"],
        edge["depth"],
        _derivation_key(edge["derivation"]),
    )


def _record_base_edge(record):
    """Extract the (from_key, to_key) base edge a record encodes, or None
    if the record is missing either key (malformed records are skipped by
    the graph, not fatal -- the store layer owns record validity).
    """
    from_key = record.get("trigger_key")
    to_key = record.get("outcome_key")
    if from_key is None or to_key is None:
        return None
    return (from_key, to_key)


class LiveGraph:
    """Base adjacency + delta-maintained transitive-inferred edges over a
    LiveStore's records.

    Base edges: keyed by (from_key, to_key) -> sorted list of (sha, idx)
    citing records (a base edge can be produced by more than one record).

    Inferred edges: transitive chains of length depth in [2, MAX_DEPTH],
    each carrying its full derivation (the ordered list of [sha, idx]
    base-edge citations that produced it, one per hop). Distinct chains
    between the same (from_key, to_key) with distinct derivations are
    distinct inferred edges -- this is what makes drop-invalidation exact
    (SS2: "exactly the inferred edges whose derivation cites it").
    """

    def __init__(self, store_dir, count_closures=False, canon=False, nlp=None):
        self.store_dir = store_dir
        self.store = LiveStore(store_dir)
        # base_edges[from_key][to_key] = sorted list of [sha, idx]
        self._base_edges = {}
        # inferred_by_from[from_key] = list of inferred-edge dicts
        self._inferred_by_from = {}
        # All inferred edges, for persistence / full listing.
        self._inferred_all = []
        # Reverse adjacency for ancestor walks: base_rev[to_key] = set(from_key)
        self._base_rev = {}
        self._rebuilt_on_mount = False
        self._loaded_from_cache = False
        # Optional instrumentation (P71c): counts calls into the two
        # closure-computing routines (_batch_transitive_closure via full
        # rebuild, _delta_chains_for_citation via on_append). Off by
        # default; when off, closure_calls stays None and behavior is
        # byte-identical to before this flag existed. on_drop() never
        # calls either routine, so this counter is what makes "zero new
        # closures on drop" a checkable fact rather than an assertion.
        self.closure_calls = 0 if count_closures else None

        # ------------------------------------------------------------
        # Canonicalization layer (opt-in, P74 organ; canon.py). OFF by
        # default -- canon=False is the entire pre-P74 code path, touched
        # nowhere below this block, so default construction stays
        # byte-identical to every LiveGraph built before this flag
        # existed (the regression guarantee test_canon.py checks).
        #
        # `nlp` is an explicit, injectable spaCy pipeline (or None) passed
        # straight through to canon.canonical_key -- lets a caller pin a
        # specific loaded pipeline (tests) or force the no-spaCy fallback
        # path (nlp=False, see _resolve_nlp below) without touching
        # canon.py's own module-level cache. Default None means "resolve
        # canon.py's module-cached default pipeline, load it once."
        # ------------------------------------------------------------
        self.canon_enabled = bool(canon)
        self._canon_nlp_arg = nlp
        self._canon_nlp = None  # resolved lazily in _resolve_nlp()
        self.canon_env_pin = None
        # canon_base_edges[canon_key][canon_key] = sorted list of [sha, idx]
        # (same citation shape as _base_edges -- citations still point at
        # RAW records; only the from/to key strings are canonical).
        self._canon_base_edges = {}
        self._canon_base_rev = {}
        self._canon_inferred_all = []
        self._canon_inferred_by_from = {}
        # raw_key -> canon_key memo (persisted to canon_map.jsonl, see the
        # canon-layer section below) -- every raw_key is parsed through
        # canon_mod.canonical_key AT MOST ONCE per process lifetime,
        # whether via a warm map load or a genuine cache miss.
        self._raw_to_canon = {}
        self._canon_rebuilt_on_mount = False
        self._canon_loaded_from_cache = False

        self._mount()

    # ------------------------------------------------------------------
    # Mount / cache
    # ------------------------------------------------------------------

    def _cache_path(self):
        return os.path.join(self.store_dir, INFERRED_NAME)

    def _mount(self):
        manifest_segments = self.store.segments()
        cache_path = self._cache_path()
        if os.path.exists(cache_path):
            cached = self._try_load_cache(cache_path, manifest_segments)
            if cached:
                self._loaded_from_cache = True
                self._mount_canon(manifest_segments)
                return
        # No cache, or manifest stamp mismatch -> full rebuild from the store.
        self._rebuild_from_store()
        self._rebuilt_on_mount = True
        self._write_cache()
        self._mount_canon(manifest_segments)

    def _try_load_cache(self, cache_path, manifest_segments):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return False
        if not lines:
            return False
        try:
            header = json.loads(lines[0])
        except (ValueError, json.JSONDecodeError):
            return False
        if header.get("segments") != manifest_segments:
            return False

        inferred = []
        for line in lines[1:]:
            line = line.rstrip("\n")
            if line == "":
                continue
            try:
                inferred.append(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                return False

        # Cache is valid for the current manifest stamp: rebuild base
        # adjacency from the store (cheap, needed for query()/on_append
        # ancestor-descendant walks) and adopt the cached inferred edges.
        self._rebuild_base_edges_only()
        self._inferred_all = inferred
        self._reindex_inferred()
        return True

    def _rebuild_base_edges_only(self):
        self._base_edges = {}
        self._base_rev = {}
        for sha, idx, record in self.store.iter_records():
            edge = _record_base_edge(record)
            if edge is None:
                continue
            from_key, to_key = edge
            self._add_base_citation(from_key, to_key, sha, idx)

    def _rebuild_from_store(self):
        """Full rebuild: base adjacency + inferred edges, from scratch."""
        self._base_edges = {}
        self._base_rev = {}
        self._inferred_all = []
        self._inferred_by_from = {}

        # First pass: collect all base edges (with their citing records).
        for sha, idx, record in self.store.iter_records():
            edge = _record_base_edge(record)
            if edge is None:
                continue
            from_key, to_key = edge
            self._add_base_citation(from_key, to_key, sha, idx)

        # Second pass: derive all transitive chains from scratch. This is
        # the batch (non-incremental) closure, used only here and as the
        # equivalence oracle in tests -- on_append never calls this.
        if self.closure_calls is not None:
            self.closure_calls += 1
        all_inferred = _batch_transitive_closure(self._base_edges)
        self._inferred_all = all_inferred
        self._reindex_inferred()

    def _reindex_inferred(self):
        self._inferred_by_from = {}
        for edge in self._inferred_all:
            self._inferred_by_from.setdefault(edge["from_key"], []).append(edge)
        for key in self._inferred_by_from:
            self._inferred_by_from[key].sort(key=_edge_sort_key)
        self._inferred_all.sort(key=_edge_sort_key)

    def _add_base_citation(self, from_key, to_key, sha, idx):
        citations = self._base_edges.setdefault(from_key, {}).setdefault(to_key, [])
        pair = [sha, idx]
        if pair not in citations:
            citations.append(pair)
            citations.sort(key=lambda p: (p[0], p[1]))
        self._base_rev.setdefault(to_key, set()).add(from_key)

    def _write_cache(self):
        header = _canonical_line({"segments": self.store.segments()})
        body = "".join(_canonical_line(e) for e in self._inferred_all)
        data = (header + body).encode("utf-8")
        _atomic_write(self._cache_path(), data)

    # ------------------------------------------------------------------
    # Canonicalization layer (P74 organ, opt-in via canon=True).
    #
    # P75c (registered, scored 2026-08-10): the P74-era full-fold-per-
    # mount design cost 17.8s on a warm mount even with
    # canon_inferred.jsonl cache-hit (warm_loaded_from_cache=True) --
    # every raw_key was re-canonicalized (re-parsed through spaCy) on
    # EVERY mount regardless of the inferred-edge cache, because only the
    # closure step was cached, not the raw_key -> canon_key fold itself.
    # That is now a measured, registered cost regression the build brief
    # names as PFLICHT (mandatory) before builder integration -- fixed
    # here with two changes:
    #
    # 1. PERSISTED CANON MAP (canon_map.jsonl, this section below): every
    #    raw_key -> canon_key pair this graph has ever computed is
    #    appended to a manifest-and-env_pin-stamped file, mirroring
    #    canon_inferred.jsonl's own cache-validity discipline exactly
    #    (same two-field stamp: segments() AND canon_env_pin). A warm
    #    mount with a valid map stamp loads the map directly and calls
    #    canonical_key() ZERO times -- pure dict deserialization.
    #
    # 2. SEMI-NAIVE CANON DELTA (on_append/on_drop below, replacing the
    #    P74-era _refresh_canon full fold): on_append canonicalizes ONLY
    #    the raw_keys that are genuinely NEW to the map (a map hit costs
    #    a dict lookup, not a spaCy parse), then folds the resulting new
    #    CANONICAL base-edge citations through the exact same
    #    _delta_chains_for_citation machinery the raw layer's on_append
    #    already uses -- just called with canon_key adjacency
    #    (_canon_base_edges/_canon_base_rev) instead of raw adjacency.
    #
    #    The build brief's own concern -- "a new record folding onto an
    #    ALREADY-PRESENT canon_key can newly connect two previously-
    #    disjoint canon components whose citations were appended
    #    arbitrarily long ago" -- turns out to need NO special-casing:
    #    _delta_chains_for_citation was already written generically
    #    enough for this. It never assumes from_key/to_key are new to the
    #    adjacency, only that the CITATION (a specific (sha, idx) pair on
    #    a specific (from_key, to_key) edge) is new -- it walks
    #    ancestors(from_key) and descendants(to_key) fresh, against
    #    whatever the CURRENT adjacency contains, every time it is
    #    called. Feeding it a canonical (canon_from, canon_to, sha, idx)
    #    citation is therefore already correct for the many-to-one case:
    #    if canon_from already had five other ancestors from raw_keys
    #    appended segments ago, the ancestor walk finds all five, exactly
    #    as it would for a raw citation joining an old raw component. The
    #    ONLY new requirement is emitting one canonical citation per NEW
    #    raw base-edge citation (not per NEW canon_key pair) -- verified
    #    directly by test_canon_delta.py's batch-oracle equivalence test
    #    (delta result == a from-scratch full fold, on a corpus
    #    constructed to exercise exactly this many-to-one join case).
    # ------------------------------------------------------------------

    def _canon_cache_path(self):
        return os.path.join(self.store_dir, CANON_INFERRED_NAME)

    def _canon_map_path(self):
        return os.path.join(self.store_dir, CANON_MAP_NAME)

    def _resolve_nlp(self):
        """Resolves the spaCy pipeline used for this graph's canon calls,
        once, cached on self. `nlp=False` (explicit, not None) at
        construction forces the no-spaCy fallback path regardless of what
        is installed -- useful for tests that need the fallback
        deterministically without uninstalling anything. `nlp=None`
        (the default) resolves canon.py's own module-cached default
        pipeline (loaded at most once per process). Any other value
        (an already-loaded spaCy Language object) is used as-is."""
        if self._canon_nlp_arg is False:
            return None
        if self._canon_nlp_arg is not None:
            return self._canon_nlp_arg
        return canon_mod._get_nlp()

    def _mount_canon(self, manifest_segments):
        if not self.canon_enabled:
            return
        self._canon_nlp = self._resolve_nlp()
        self.canon_env_pin = canon_mod.env_pin()

        # Two INDEPENDENT caches, each stamped the same way (segments +
        # canon_env_pin) but validated separately: the map cache (raw_key
        # -> canon_key, the expensive-to-recompute spaCy-parse artifact)
        # and the inferred-edge cache (the closure artifact, cheap to
        # recompute FROM a valid map, expensive to recompute from
        # scratch). A map hit + closure miss still avoids every spaCy
        # call; a map miss forces the same full parse pass P74 always
        # paid, regardless of the closure cache's state.
        map_hit = self._try_load_canon_map(manifest_segments)
        if not map_hit:
            self._raw_to_canon = {}

        cache_path = self._canon_cache_path()
        closure_hit = False
        if os.path.exists(cache_path):
            closure_hit = self._try_load_canon_cache(cache_path, manifest_segments)

        if map_hit and closure_hit:
            self._canon_loaded_from_cache = True
            # Base adjacency still needs folding from the (now-cached) map
            # -- pure dict work, no canonicalization calls at all.
            self._canon_fold_base_edges_from_map()
            return

        # Fold the base adjacency (canonicalizing any raw_key the map
        # didn't already have -- on a full map hit this is zero calls; on
        # a cold/partial map this canonicalizes exactly the keys missing
        # from the map, same as before, never more).
        self._canon_fold_base_edges_from_map()

        if closure_hit:
            # Map was rebuilt/extended (or was already complete) but the
            # closure cache was stale relative to the CURRENT manifest --
            # this combination should not arise in practice (closure
            # cache and map cache share the same stamp), but if it does,
            # prefer correctness: the closure must be recomputed too,
            # since the map's completeness does not guarantee the base
            # adjacency the closure was computed over still matches.
            closure_hit = False

        if not closure_hit:
            if self.closure_calls is not None:
                self.closure_calls += 1
            self._canon_inferred_all = _batch_transitive_closure(self._canon_base_edges)
            self._reindex_canon_inferred()
            self._canon_rebuilt_on_mount = True

        self._write_canon_map()
        self._write_canon_cache()

    def _try_load_canon_map(self, manifest_segments):
        """Loads canon_map.jsonl into self._raw_to_canon if its stamp
        (segments + canon_env_pin) matches the current mount. Returns
        True on a valid load (self._raw_to_canon populated, ZERO
        canonical_key calls made), False otherwise (self._raw_to_canon
        left untouched -- caller resets it to {} on a miss)."""
        path = self._canon_map_path()
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return False
        if not lines:
            return False
        try:
            header = json.loads(lines[0])
        except (ValueError, json.JSONDecodeError):
            return False
        if header.get("segments") != manifest_segments:
            return False
        if header.get("canon_env_pin") != self.canon_env_pin:
            return False

        mapping = {}
        for line in lines[1:]:
            line = line.rstrip("\n")
            if line == "":
                continue
            try:
                entry = json.loads(line)
            except (ValueError, json.JSONDecodeError):
                return False
            mapping[entry["raw"]] = entry["canon"]
        self._raw_to_canon = mapping
        return True

    def _write_canon_map(self):
        """Persists self._raw_to_canon (the full raw_key -> canon_key
        mapping accumulated so far) to canon_map.jsonl, stamped with the
        current manifest segments + canon_env_pin -- same atomic-write,
        same header-then-body JSON-Lines discipline every other cache in
        this module uses. Called after every fold/delta that may have
        added new entries (mount, on_append); on_drop does NOT call this
        (see on_drop's own docstring: the map is a pure function's memo,
        never invalidated by a drop)."""
        header = _canonical_line({
            "segments": self.store.segments(),
            "canon_env_pin": self.canon_env_pin,
        })
        body = "".join(
            _canonical_line({"raw": raw, "canon": ck})
            for raw, ck in sorted(self._raw_to_canon.items())
        )
        data = (header + body).encode("utf-8")
        _atomic_write(self._canon_map_path(), data)

    def _canon_of_memo(self, raw_key):
        """The one canonicalization call site for the whole class: a memo
        lookup against self._raw_to_canon, calling canon_mod.canonical_key
        (a real spaCy parse, or the fallback) ONLY on a genuine miss, and
        recording the result in the memo immediately so no raw_key is
        ever parsed twice within one process's lifetime -- this is what
        makes on_append's delta cost track "how many NEW raw_keys arrived"
        rather than "how many raw_keys exist in the whole store"."""
        cached = self._raw_to_canon.get(raw_key)
        if cached is not None:
            return cached
        ck = canon_mod.canonical_key(raw_key, nlp=self._canon_nlp)
        self._raw_to_canon[raw_key] = ck
        return ck

    def _canon_fold_base_edges_from_map(self):
        """Rebuilds canon_base_edges/canon_base_rev from the RAW base
        adjacency already in memory (self._base_edges), canonicalizing
        each distinct raw_key via the memo (_canon_of_memo) -- a map hit
        costs a dict lookup per raw_key, never a parse. This is still an
        O(all base edges) pass over the in-memory adjacency (dict
        iteration, no I/O, no spaCy) -- the P75c cost that is now
        eliminated is the PARSING, not this bookkeeping fold, which was
        never the expensive part (see P75c's own finding: warm mount cost
        17.8s while this fold's own dict work is sub-millisecond -- the
        entire cost was re-parsing every raw_key through spaCy on every
        mount, even with a valid closure cache)."""
        self._canon_base_edges = {}
        self._canon_base_rev = {}

        for from_key, targets in self._base_edges.items():
            canon_from = self._canon_of_memo(from_key)
            for to_key, citations in targets.items():
                canon_to = self._canon_of_memo(to_key)
                bucket = self._canon_base_edges.setdefault(canon_from, {}).setdefault(canon_to, [])
                for pair in citations:
                    if pair not in bucket:
                        bucket.append(list(pair))
                bucket.sort(key=lambda p: (p[0], p[1]))
                self._canon_base_rev.setdefault(canon_to, set()).add(canon_from)

    def _reindex_canon_inferred(self):
        self._canon_inferred_by_from = {}
        for edge in self._canon_inferred_all:
            self._canon_inferred_by_from.setdefault(edge["from_key"], []).append(edge)
        for key in self._canon_inferred_by_from:
            self._canon_inferred_by_from[key].sort(key=_edge_sort_key)
        self._canon_inferred_all.sort(key=_edge_sort_key)

    def _try_load_canon_cache(self, cache_path, manifest_segments):
        """Loads canon_inferred.jsonl (the closure/transitive-edge cache)
        into self._canon_inferred_all if its stamp matches. Returns True
        on a valid load, False otherwise. This is the closure artifact
        ONLY -- it says nothing about whether the raw_key -> canon_key
        map is warm; see _try_load_canon_map for that, and _mount_canon
        for how the two independent cache hits combine."""
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
        except OSError:
            return False
        if not lines:
            return False
        try:
            header = json.loads(lines[0])
        except (ValueError, json.JSONDecodeError):
            return False
        if header.get("segments") != manifest_segments:
            return False
        if header.get("canon_env_pin") != self.canon_env_pin:
            # The P70 lesson, applied here: a cache built under a
            # different spaCy availability/version or a different
            # CANON_VERSION is not a valid cache for THIS process's
            # canonicalization function, even if the manifest stamp
            # matches exactly. Mismatch -> full rebuild, never adopted.
            return False

        inferred = []
        for line in lines[1:]:
            line = line.rstrip("\n")
            if line == "":
                continue
            try:
                inferred.append(json.loads(line))
            except (ValueError, json.JSONDecodeError):
                return False

        self._canon_inferred_all = inferred
        self._reindex_canon_inferred()
        return True

    def _write_canon_cache(self):
        header = _canonical_line({
            "segments": self.store.segments(),
            "canon_env_pin": self.canon_env_pin,
        })
        body = "".join(_canonical_line(e) for e in self._canon_inferred_all)
        data = (header + body).encode("utf-8")
        _atomic_write(self._canon_cache_path(), data)

    def _canon_on_append(self, sha, raw_new_citations):
        """Semi-naive canon-layer maintenance for a newly appended segment.

        raw_new_citations: the SAME (from_key, to_key, sha, idx,
        was_new_pair) tuples the raw layer's own on_append just computed
        for this segment (passed in rather than recomputed -- one pass
        over the segment's records is enough for both layers).

        For each raw citation, canonicalize its endpoints via the memo
        (a spaCy parse ONLY for a raw_key genuinely new to this graph's
        map -- see _canon_of_memo), fold the resulting canonical citation
        into _canon_base_edges/_canon_base_rev, and run the EXACT same
        _delta_chains_for_citation the raw layer uses, over the canonical
        adjacency instead of the raw one. See this class's canon-layer
        docstring block above for why this is already correct for the
        many-to-one join case without any special-casing."""
        if not self.canon_enabled:
            return

        canon_new_citations = []  # (canon_from, canon_to, sha, idx)
        for from_key, to_key, csha, cidx, _was_new_pair in raw_new_citations:
            canon_from = self._canon_of_memo(from_key)
            canon_to = self._canon_of_memo(to_key)
            bucket = self._canon_base_edges.setdefault(canon_from, {}).setdefault(canon_to, [])
            pair = [csha, cidx]
            if pair not in bucket:
                bucket.append(pair)
                bucket.sort(key=lambda p: (p[0], p[1]))
            self._canon_base_rev.setdefault(canon_to, set()).add(canon_from)
            canon_new_citations.append((canon_from, canon_to, csha, cidx))

        new_edges = []
        seen = set()
        for canon_from, canon_to, csha, cidx in canon_new_citations:
            hop = [csha, cidx]
            if self.closure_calls is not None:
                self.closure_calls += 1
            chains = _delta_chains_for_citation(
                self._canon_base_edges, self._canon_base_rev, canon_from, canon_to, hop, MAX_DEPTH
            )
            for edge in chains:
                dedup_key = (edge["from_key"], edge["to_key"], _derivation_key(edge["derivation"]))
                if dedup_key in seen:
                    continue
                if any(
                    e["from_key"] == edge["from_key"]
                    and e["to_key"] == edge["to_key"]
                    and _derivation_key(e["derivation"]) == _derivation_key(edge["derivation"])
                    for e in self._canon_inferred_by_from.get(edge["from_key"], [])
                ):
                    seen.add(dedup_key)
                    continue
                seen.add(dedup_key)
                new_edges.append(edge)

        for edge in new_edges:
            self._canon_inferred_all.append(edge)
            self._canon_inferred_by_from.setdefault(edge["from_key"], []).append(edge)

        self._reindex_canon_inferred()
        self._write_canon_map()
        self._write_canon_cache()
        return new_edges

    def _canon_on_drop(self, drop_set):
        """Semi-naive canon-layer maintenance for dropped segments:
        exactly mirrors the raw layer's on_drop (filter citations from
        canon_base_edges/canon_base_rev, filter inferred edges whose
        derivation cites a dropped segment) -- No re-inference of the
        surviving canon graph, same SS2 guarantee the raw layer gives.

        The canon MAP's CONTENT (raw_key -> canon_key) is never pruned by
        a drop: it is a memo of a PURE FUNCTION
        (canonical_key(raw_key, env_pin) -> canon_key) that does not
        depend on which segments are currently present -- a raw_key's
        canon_key does not change because the record that first
        introduced it was dropped, and a stale extra entry for a since-
        dropped raw_key is harmless (never read unless that exact
        raw_key is canonicalized again, in which case the memoized
        answer is still correct -- so a drop-then-re-append of the same
        raw_key costs zero re-parses). The map's STAMP (its header's
        `segments` field) DOES still need rewriting after a drop, though
        -- the manifest changed, so the on-disk stamp must track it or
        the next mount would see a stamp mismatch and wrongly treat an
        otherwise-still-correct map as invalid, forcing needless
        re-parses of every raw_key on the next cold mount. Rewriting the
        stamp is a cheap re-serialization of the SAME in-memory dict,
        not a re-parse of anything."""
        if not self.canon_enabled:
            return

        new_canon_base_edges = {}
        new_canon_base_rev = {}
        for from_key, targets in self._canon_base_edges.items():
            for to_key, citations in targets.items():
                kept = [p for p in citations if p[0] not in drop_set]
                if kept:
                    new_canon_base_edges.setdefault(from_key, {})[to_key] = kept
                    new_canon_base_rev.setdefault(to_key, set()).add(from_key)
        self._canon_base_edges = new_canon_base_edges
        self._canon_base_rev = new_canon_base_rev

        kept_inferred = [
            e
            for e in self._canon_inferred_all
            if not any(hop_sha in drop_set for hop_sha, _hop_idx in e["derivation"])
        ]
        self._canon_inferred_all = kept_inferred
        self._reindex_canon_inferred()
        # Persistence (_write_canon_cache/_write_canon_map) is deliberately
        # NOT done here: on_drop() runs BEFORE drop_segments() removes the
        # segment from the store's manifest (see drop_segments's own
        # docstring), so self.store.segments() at this point would still
        # include the about-to-be-dropped segment -- writing the map/cache
        # stamp now would stamp it with a manifest state that is about to
        # become stale, defeating the whole point of keeping the stamp in
        # sync. drop_segments() calls _write_canon_cache/_write_canon_map
        # itself, AFTER the store's manifest is actually updated.

    def canon_query(self, raw_key):
        """query()'s canonical counterpart: canonicalizes `raw_key` itself
        (the READ side of P73's coverage number -- a query for "the old
        king" must find edges filed under "king of france"'s canon_key
        too) and returns base + inferred edges over the canonical
        adjacency, in the SAME dict shape query() returns. Every returned
        edge's `derivation` still cites RAW (segment_sha, idx) coordinates
        -- canonicalization never invents a citation, it only changes
        which from/to key STRING joins which citations together. A
        stranger can therefore re-derive any canon edge from nothing but
        the cited raw records plus this module's pinned canonical_key
        function (see test_canon.py's verifier-compatibility test).

        Raises RuntimeError if this graph was not constructed with
        canon=True -- calling the canonical query path on a canon=False
        graph is a caller bug, not a silent empty result."""
        if not self.canon_enabled:
            raise RuntimeError(
                "canon_query() requires LiveGraph(..., canon=True); "
                "this graph was mounted with canon=False."
            )
        canon_key = self._canon_of_memo(raw_key)
        out = []
        for to_key, citations in sorted(self._canon_base_edges.get(canon_key, {}).items()):
            out.append({
                "kind": "base",
                "from_key": canon_key,
                "to_key": to_key,
                "derivation": [list(p) for p in citations],
            })
        for edge in self._canon_inferred_by_from.get(canon_key, []):
            out.append({
                "kind": "inferred",
                "from_key": edge["from_key"],
                "to_key": edge["to_key"],
                "depth": edge["depth"],
                "derivation": [list(p) for p in edge["derivation"]],
            })
        out.sort(key=lambda e: (e["kind"], e["to_key"], e.get("depth", 1), _derivation_key(e["derivation"])))
        return out

    def canon_inferred_edges(self):
        """All canonical inferred edges, sorted -- canon-layer counterpart
        to inferred_edges(). Empty list if canon_enabled is False."""
        return [dict(e, derivation=[list(p) for p in e["derivation"]]) for e in self._canon_inferred_all]

    def canon_of(self, raw_key):
        """The canon_key a given raw_key maps to under this graph's
        current fold. Convenience for tests/debugging; raises
        RuntimeError under the same condition canon_query() does. Uses
        the same map memo every other canon call site uses (a raw_key
        already in the map costs a dict lookup, not a re-parse)."""
        if not self.canon_enabled:
            raise RuntimeError(
                "canon_of() requires LiveGraph(..., canon=True); "
                "this graph was mounted with canon=False."
            )
        return self._canon_of_memo(raw_key)

    def was_canon_rebuilt_on_mount(self):
        return self._canon_rebuilt_on_mount

    def was_canon_loaded_from_cache(self):
        return self._canon_loaded_from_cache

    # ------------------------------------------------------------------
    # Introspection helpers (test-facing, per the build brief's
    # "Zähler/Flag testbar machen" requirement for cache validity).
    # ------------------------------------------------------------------

    def was_rebuilt_on_mount(self):
        """True if mount() had to do a full rebuild (no cache, or a stale
        manifest stamp). False if the on-disk cache was adopted as-is.
        """
        return self._rebuilt_on_mount

    def was_loaded_from_cache(self):
        return self._loaded_from_cache

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, key, canon=False):
        """Base + inferred outgoing edges of `key`, sorted deterministically.

        Returns a list of dicts:
          base:     {"kind": "base", "from_key", "to_key", "derivation": [[sha, idx], ...]}
          inferred: {"kind": "inferred", "from_key", "to_key", "depth", "derivation": [...]}

        canon=False (default): EXACT pre-P74 behavior, byte-identical to
        every caller written before this parameter existed -- `key` is
        matched against the raw base/inferred adjacency only.

        canon=True: the READ side of the P73 coverage measurement. `key`
        is itself canonicalized (canon_mod.canonical_key) before lookup,
        against the canonical adjacency built by LiveGraph(canon=True) --
        so a query for "the old king" can find edges filed under "king of
        france"'s canon_key. Requires this graph to have been constructed
        with canon=True (raises RuntimeError otherwise, via canon_query
        below -- a canon=True query against a canon=False graph is a
        caller bug, not a silent empty result). Every returned edge's
        derivation still cites raw (segment_sha, idx) coordinates.
        """
        if canon:
            return self.canon_query(key)
        out = []
        for to_key, citations in sorted(self._base_edges.get(key, {}).items()):
            out.append(
                {
                    "kind": "base",
                    "from_key": key,
                    "to_key": to_key,
                    "derivation": [list(p) for p in citations],
                }
            )
        for edge in self._inferred_by_from.get(key, []):
            out.append(
                {
                    "kind": "inferred",
                    "from_key": edge["from_key"],
                    "to_key": edge["to_key"],
                    "depth": edge["depth"],
                    "derivation": [list(p) for p in edge["derivation"]],
                }
            )
        out.sort(key=lambda e: (e["kind"], e["to_key"], e.get("depth", 1), _derivation_key(e["derivation"])))
        return out

    def inferred_edges(self):
        """All inferred edges, sorted. Read-only snapshot for tests/inspection."""
        return [dict(e, derivation=[list(p) for p in e["derivation"]]) for e in self._inferred_all]

    def base_edge_citations(self, from_key, to_key):
        return [list(p) for p in self._base_edges.get(from_key, {}).get(to_key, [])]

    def edge_keys_for_derivation(self, derivation):
        """The one hook the evidence calculus needs from infer.py
        (analysis/EVIDENCE_CALCULUS_DRAFT.md SS2.3): for an inferred
        edge's derivation ([[sha, idx], ...], root-to-leaf hop order),
        return the list of (from_key, to_key) base edge_keys, one per
        hop, in the same order. Reads each cited record fresh via the
        store (not the inferred-edge cache) -- the same re-derivation
        discipline on_append/query already use, so this stays "re-
        derivable by a stranger from the cited base edges alone." infer.py
        does not know what "contested" means; a caller (src/livecausal/
        evidence.py's contested_for_derivation) combines this list with
        its own per-edge contested lookup. This is the ONLY evidence-
        calculus-shaped addition to this file -- everything else about
        evidence/use/dominance lives in evidence.py.
        """
        keys = []
        for sha, idx in derivation:
            record = None
            for seg_sha, rec_idx, rec in self.store.iter_records(sha):
                if rec_idx == idx:
                    record = rec
                    break
            if record is None:
                raise ValueError("derivation cites unknown record ({}, {})".format(sha, idx))
            keys.append((record["trigger_key"], record["outcome_key"]))
        return keys

    # ------------------------------------------------------------------
    # Append: semi-naive delta closure
    # ------------------------------------------------------------------

    def on_append(self, sha):
        """Incorporate a newly appended segment (by sha) into the graph.

        For every new base-edge citation (A -> B via this segment's
        records), generate exactly the transitive chains that use this
        citation as one of their hops: ancestors(A) x {this citation} x
        descendants(B), depth-capped at MAX_DEPTH, cycle-free. This is the
        semi-naive delta rule from SS2: cost scales with the delta's
        neighborhood, not the graph's size. No full rebuild.
        """
        new_citations = []  # list of (from_key, to_key, sha, idx)
        for _seg_sha, idx, record in self.store.iter_records(sha):
            edge = _record_base_edge(record)
            if edge is None:
                continue
            from_key, to_key = edge
            was_new_pair = to_key not in self._base_edges.get(from_key, {})
            self._add_base_citation(from_key, to_key, sha, idx)
            new_citations.append((from_key, to_key, sha, idx, was_new_pair))

        new_edges = []
        seen = set()  # (from_key, to_key, derivation-key) already added this call
        for from_key, to_key, csha, cidx, _was_new_pair in new_citations:
            hop = [csha, cidx]
            if self.closure_calls is not None:
                self.closure_calls += 1
            chains = _delta_chains_for_citation(
                self._base_edges, self._base_rev, from_key, to_key, hop, MAX_DEPTH
            )
            for edge in chains:
                dedup_key = (edge["from_key"], edge["to_key"], _derivation_key(edge["derivation"]))
                if dedup_key in seen:
                    continue
                # Also guard against an identical inferred edge already
                # present from a prior on_append/rebuild (idempotent
                # re-append of the same segment content is a store no-op,
                # but on_append could in principle be called more than
                # once for defensive callers).
                if any(
                    e["from_key"] == edge["from_key"]
                    and e["to_key"] == edge["to_key"]
                    and _derivation_key(e["derivation"]) == _derivation_key(edge["derivation"])
                    for e in self._inferred_by_from.get(edge["from_key"], [])
                ):
                    seen.add(dedup_key)
                    continue
                seen.add(dedup_key)
                new_edges.append(edge)

        for edge in new_edges:
            self._inferred_all.append(edge)
            self._inferred_by_from.setdefault(edge["from_key"], []).append(edge)

        self._reindex_inferred()
        self._write_cache()
        # Canon-layer delta: reuses the SAME new_citations this method
        # just computed (one segment-record pass serves both layers) --
        # see _canon_on_append's docstring for why this is a correct
        # semi-naive rule even for the many-to-one canon_key join case.
        self._canon_on_append(sha, new_citations)
        return new_edges

    # ------------------------------------------------------------------
    # Drop: derivation invalidation
    # ------------------------------------------------------------------

    def on_drop(self, shas):
        """Remove base edges contributed by the dropped segments and
        exactly the inferred edges whose derivation cites one of them.
        No re-inference of the surviving graph (SS2: "No re-inference of
        the surviving graph").
        """
        drop_set = set(shas)

        # Rebuild base_edges/base_rev by filtering out citations from
        # dropped segments. Citations, not just presence, must be pruned:
        # a base pair can be citation-supported by more than one segment.
        new_base_edges = {}
        new_base_rev = {}
        for from_key, targets in self._base_edges.items():
            for to_key, citations in targets.items():
                kept = [p for p in citations if p[0] not in drop_set]
                if kept:
                    new_base_edges.setdefault(from_key, {})[to_key] = kept
                    new_base_rev.setdefault(to_key, set()).add(from_key)
        self._base_edges = new_base_edges
        self._base_rev = new_base_rev

        kept_inferred = [
            e
            for e in self._inferred_all
            if not any(hop_sha in drop_set for hop_sha, _hop_idx in e["derivation"])
        ]
        self._inferred_all = kept_inferred
        self._reindex_inferred()
        self._write_cache()
        self._canon_on_drop(drop_set)

    # ------------------------------------------------------------------
    # Drop segments through the store (convenience: mirrors store API)
    # ------------------------------------------------------------------

    def drop_segments(self, shas):
        """Drop segments from the underlying store AND invalidate the
        graph accordingly. Order matters: graph invalidation reads
        citations already collected in-memory, so this does not need the
        segment files to still exist on disk.

        Canon-layer persistence (_write_canon_cache/_write_canon_map) is
        deliberately deferred to AFTER self.store.drop_segments(shas)
        below, not done inside on_drop()/_canon_on_drop() -- writing the
        map/cache stamp before the store's manifest is actually updated
        would stamp it with the ABOUT-TO-BE-STALE segment list (the
        dropped segment still present), so a subsequent mount would see
        a stamp mismatch against the store's REAL post-drop manifest and
        wrongly force a full re-parse of every raw_key -- exactly the
        P75c cost this task exists to eliminate. Caught directly by
        test_canon_delta.py's drop-then-remount warm-mount check.
        """
        shas = list(shas)
        self.on_drop(shas)
        self.store.drop_segments(shas)
        if self.canon_enabled:
            self._write_canon_cache()
            self._write_canon_map()

    def append_segment(self, records):
        """Append records to the underlying store AND fold them into the
        graph via on_append. Convenience wrapper mirroring the store API.
        """
        sha = self.store.append_segment(records)
        self.on_append(sha)
        return sha


# ----------------------------------------------------------------------
# Batch (non-incremental) transitive closure -- the equivalence oracle.
# Used by _rebuild_from_store only; on_append never calls this.
# ----------------------------------------------------------------------


def _batch_transitive_closure(base_edges):
    """All transitive chains of depth 2..MAX_DEPTH over base_edges (dict of
    from_key -> {to_key: [[sha, idx], ...]}), cycle-free, sorted.

    A "chain" here is a specific sequence of base-edge citations (one hop
    per edge in the path); if a base pair (from_key, to_key) has multiple
    citing records, each citing record generates its own chain instance
    (distinct derivation), matching what the delta path does one hop at a
    time (on_append picks one new citation, but base_edges accumulate all
    of them, so a full rebuild must also fan out over all citations per
    hop to be equivalent).
    """
    edges = []

    def _dfs(path_keys, path_derivation, depth):
        if depth >= MAX_DEPTH:
            return
        current = path_keys[-1]
        for to_key, citations in sorted(base_edges.get(current, {}).items()):
            if to_key in path_keys:
                continue  # cycle guard
            for citation in citations:
                new_derivation = path_derivation + [list(citation)]
                new_path = path_keys + [to_key]
                new_depth = depth + 1
                if new_depth >= 2:
                    edges.append(
                        {
                            "from_key": new_path[0],
                            "to_key": new_path[-1],
                            "depth": new_depth,
                            "derivation": new_derivation,
                        }
                    )
                _dfs(new_path, new_derivation, new_depth)

    for start_key in sorted(base_edges.keys()):
        _dfs([start_key], [], 0)

    edges.sort(key=_edge_sort_key)
    return edges


# ----------------------------------------------------------------------
# Semi-naive delta: chains using one specific new citation as a hop.
# ----------------------------------------------------------------------


def _delta_chains_for_citation(base_edges, base_rev, from_key, to_key, hop, max_depth):
    """All transitive chains of depth 2..max_depth that use the base-edge
    citation (from_key -[hop]-> to_key) as exactly one of their hops.

    This is ancestors(from_key) x {hop} x descendants(to_key), assembled
    as: for every prefix ending at from_key (possibly empty, i.e.
    from_key itself as chain start) and every suffix starting at to_key
    (possibly empty, i.e. to_key itself as chain end), join
    prefix + hop + suffix, provided the combined length is within
    [2, max_depth] and the combined key path has no repeats (cycle-free).
    """
    prefixes = _walk_ancestors(base_rev, base_edges, from_key, max_depth)
    suffixes = _walk_descendants(base_edges, to_key, max_depth)

    results = []
    for prefix_keys, prefix_derivation in prefixes:
        remaining_after_prefix = max_depth - len(prefix_derivation) - 1  # hops left for suffix
        if remaining_after_prefix < 0:
            continue
        for suffix_keys, suffix_derivation in suffixes:
            if len(suffix_derivation) > remaining_after_prefix:
                continue
            # prefix_keys ends at from_key; the hop itself contributes
            # to_key (suffix_keys[0]); suffix_keys[1:] contributes any
            # further descent beyond to_key.
            full_keys = prefix_keys + suffix_keys
            if len(set(full_keys)) != len(full_keys):
                continue  # cycle guard over the assembled path
            derivation = prefix_derivation + [list(hop)] + suffix_derivation
            depth = len(derivation)
            if depth < 2 or depth > max_depth:
                continue
            results.append(
                {
                    "from_key": full_keys[0],
                    "to_key": full_keys[-1],
                    "depth": depth,
                    "derivation": derivation,
                }
            )
    results.sort(key=_edge_sort_key)
    return results


def _walk_ancestors(base_rev, base_edges, key, max_depth):
    """All (path_keys, path_derivation) ending at `key`, path_keys[-1] ==
    key, path_derivation is the ordered list of citations for path_keys[0]
    -> ... -> key. Includes the zero-length path (just [key], []).
    Depth-capped so that len(path_derivation) <= max_depth - 1 (room left
    for at least the mandatory hop).
    """
    results = [([key], [])]

    def _dfs(path_keys, path_derivation):
        if len(path_derivation) >= max_depth - 1:
            return
        current = path_keys[0]
        for parent in sorted(base_rev.get(current, set())):
            if parent in path_keys:
                continue
            citations = base_edges.get(parent, {}).get(current, [])
            for citation in citations:
                new_path = [parent] + path_keys
                new_derivation = [list(citation)] + path_derivation
                results.append((new_path, new_derivation))
                _dfs(new_path, new_derivation)

    _dfs([key], [])
    return results


def _walk_descendants(base_edges, key, max_depth):
    """All (path_keys, path_derivation) starting at `key`, path_keys[0] ==
    key. Includes the zero-length path (just [key], []). Depth-capped so
    that len(path_derivation) <= max_depth - 1.
    """
    results = [([key], [])]

    def _dfs(path_keys, path_derivation):
        if len(path_derivation) >= max_depth - 1:
            return
        current = path_keys[-1]
        for to_key, citations in sorted(base_edges.get(current, {}).items()):
            if to_key in path_keys:
                continue
            for citation in citations:
                new_path = path_keys + [to_key]
                new_derivation = path_derivation + [list(citation)]
                results.append((new_path, new_derivation))
                _dfs(new_path, new_derivation)

    _dfs([key], [])
    return results


# ----------------------------------------------------------------------
# Bench harness (harness only -- no measurement started here; the lead
# registers P71 and runs this).
# ----------------------------------------------------------------------


def bench_append(n_base_segments, delta_size, seed=0):
    """Measure wall-clock time of one on_append call against a graph that
    already holds n_base_segments segments of chain-shaped records, then
    appends one more segment of delta_size new chain-linked records.

    Returns a dict: {"n_base_segments", "delta_size", "n_base_edges_before",
    "n_inferred_before", "append_seconds", "n_new_inferred"}.

    Harness only: does not interpret results, does not write to
    analysis/PREDICTIONS.md. The caller decides what to do with the number.
    """
    import random
    import shutil
    import tempfile
    import time

    rng = random.Random(seed)
    tmpdir = tempfile.mkdtemp(prefix="livecausal-bench-")
    try:
        graph = LiveGraph(tmpdir)
        key_counter = [0]

        def _fresh_key():
            key_counter[0] += 1
            return "K{}".format(key_counter[0])

        # Seed n_base_segments segments, each a short chain of a few
        # random-length hops, to build up base-graph + inferred-edge mass.
        for _ in range(n_base_segments):
            chain_len = rng.randint(2, 4)
            keys = [_fresh_key() for _ in range(chain_len + 1)]
            records = []
            for i in range(chain_len):
                records.append(
                    {
                        "trigger": keys[i],
                        "mechanism": "bench",
                        "outcome": keys[i + 1],
                        "trigger_key": keys[i],
                        "outcome_key": keys[i + 1],
                        "doc_coord": i,
                        "evidence_count": 1,
                        "use_count": 0,
                        "meta": {},
                    }
                )
            graph.append_segment(records)

        n_base_edges_before = sum(len(v) for v in graph._base_edges.values())
        n_inferred_before = len(graph._inferred_all)

        # Build the delta segment: chained onto an existing key so the
        # ancestor/descendant walk has something to fan out over.
        anchor_key = "K1" if key_counter[0] >= 1 else _fresh_key()
        delta_keys = [anchor_key] + [_fresh_key() for _ in range(delta_size)]
        delta_records = []
        for i in range(delta_size):
            delta_records.append(
                {
                    "trigger": delta_keys[i],
                    "mechanism": "bench-delta",
                    "outcome": delta_keys[i + 1],
                    "trigger_key": delta_keys[i],
                    "outcome_key": delta_keys[i + 1],
                    "doc_coord": i,
                    "evidence_count": 1,
                    "use_count": 0,
                    "meta": {},
                }
            )
        sha = graph.store.append_segment(delta_records)

        t0 = time.perf_counter()
        new_edges = graph.on_append(sha)
        t1 = time.perf_counter()

        return {
            "n_base_segments": n_base_segments,
            "delta_size": delta_size,
            "n_base_edges_before": n_base_edges_before,
            "n_inferred_before": n_inferred_before,
            "append_seconds": t1 - t0,
            "n_new_inferred": len(new_edges),
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
