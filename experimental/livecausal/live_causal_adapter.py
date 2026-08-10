"""
LiveCausalAdapter -- serves core.knowledge.KnowledgeStore.query()'s exact
return shape from a livecausal_bridge.LiveGraph store, so repl.py's Foss
Pipeline / Instructor / _direct_kb_lookup callers can be pointed at a
LIVE-CAUSAL segment store instead of the original in-memory KnowledgeStore
WITHOUT changing a single call site -- only the constructor differs.

Not a KnowledgeStore subclass: KnowledgeStore carries Hopfield/encoder
machinery this adapter deliberately does not reimplement (Tier 3 fuzzy
matching stays OUT of scope for this MVP -- see confidence_level mapping
below). This is a narrow, explicit adapter over the two methods repl.py's
fast paths actually call: query(subject=..., relation=...) and
find_by_entity(entity). Any caller reaching for Hopfield-only
KnowledgeStore behavior (fuzzy Tier 3, .facts iteration, .store_fact) is
out of this adapter's contract and will hit AttributeError -- deliberate,
not an oversight, so a caller that needs those either stays on the
original KnowledgeStore or this adapter grows that method explicitly
later, never silently.

================================================================
CONFIDENCE DERIVATION -- documented honestly, not asserted as principled
================================================================
KnowledgeStore.query() returns a real cosine-similarity confidence in
[0, 1] for Tier 3 fuzzy matches, and a hardcoded 1.0 for Tier 1/2 EXACT
dict matches (core/knowledge.py:546, :572, :580 -- an exact (subject,
relation) or (subject) dict hit is just confidence=1.0, no computation).

LiveGraph has no equivalent notion of "confidence" at all -- query() and
canon_query() return base/inferred EDGES with derivations (citations),
not a scalar score. The only per-edge signal evidence.py exposes is
evidence_count: the number of DISTINCT independent sources (evidence_key
folds) that assert the same (trigger_key, outcome_key) edge.

This adapter's mapping (a real, working decision -- not a placeholder):

  base-kind edge (direct trigger_key -> outcome_key hit, depth-0/exact):
    confidence_level = 'HIGH'
    confidence        = min(1.0, 0.5 + 0.1 * evidence_count)
    -- i.e. a single citing record (evidence_count=1, the seed value every
       converted knowledge_full.json triplet carries, since each triplet
       is its own independent evidence_key -- see convert_knowledge_full.py's
       doc_coord rationale) yields confidence=0.6: clearly above any
       reasonable IDK_THRESHOLD (repl.py's is 0.25), clearly BELOW
       KnowledgeStore's exact-match 1.0, because a single-record base
       edge is real but has exactly one witness -- treating it as
       maximally confident would misrepresent a graph edge as more
       certain than a Dict-Index exact hit that also passed FOSS-KI's own
       Hopfield anti-hallucination checks upstream. Each additional
       independent evidence_key nudges confidence up by 0.1, capped at
       1.0 (10 independent sources -> maximal confidence). This is a
       DELIBERATE, documented linear scale, not a principled probability
       -- there is no ground truth yet for what evidence_count=5 vs.
       evidence_count=1 SHOULD mean epistemically; 0.1/source is a
       starting knob (mirrors evidence.py's own DEFAULT_DOMINANCE_RATIO
       being a named, overridable constant, not a derived one).

  inferred-kind edge (transitive chain, depth >= 2):
    confidence_level = 'MEDIUM'
    confidence        = min(0.85, 0.4 + 0.05 * evidence_count) / depth_penalty
    where depth_penalty = 1.0 + 0.1 * (depth - 2)
    -- inferred edges are capped below base edges' ceiling (0.85 vs 1.0)
       because they are a DERIVED claim (A->B->C chained into A->C), not
       a directly-cited fact; longer chains (higher depth) are penalized
       further since each additional hop is one more opportunity for the
       chain's semantics to have drifted (infer.py's own docstring flags
       that direction/mechanism-compatibility propagation is NOT
       implemented yet -- Pass 2/3 of the v1 mirror are explicitly out of
       scope, so a depth-4 inferred edge here carries LESS interpretive
       guarantee than a depth-2 one, and the confidence score says so).

  no edge found at all (empty query() result for both directions tried):
    confidence_level = 'REJECTED'
    confidence        = 0.0
    -- this is the anti-hallucination contract this adapter exists to
       preserve: an entity/relation the graph has never seen returns
       REJECTED, exactly like KnowledgeStore's Tier-3 REJECTED path, so
       repl.py's existing "Pipeline: SKIPPED — unknown to KnowledgeStore"
       guard (the P1 anti-hallucination fix) keeps working unmodified
       against a LiveCausalAdapter-backed self.knowledge.

  attractor_distance: always 0.0 for base/REJECTED (there is no Hopfield
  attractor in this adapter's model at all), and a synthetic
  0.1 * (depth - 1) for inferred edges -- reusing "distance" as "how far
  this is from a direct citation," which is what attractor_distance means
  operationally in the one place repl.py actually reads it (nowhere in
  the current fast-path code, as of this fix -- checked directly; it is
  read from the dict but not branched on in repl.py today, so this field
  is populated for API-shape completeness, not because a caller depends
  on its value yet).
================================================================
"""
import os
import sys

sys.path.insert(0, os.path.expanduser("~"))
# NOTE: LiveGraph is not exported by livecausal_bridge/__init__.py (which
# only re-exports LiveStore, canonical_bytes, segment_sha) -- imported
# from the infer submodule directly rather than editing the bridge
# package's __init__.py, which is upstream-owned code this adapter
# should not modify.
from livecausal_bridge.infer import LiveGraph  # noqa: E402
from livecausal_bridge.evidence import EvidenceLedger  # noqa: E402


def _normalize(s):
    if s is None:
        return s
    return s.strip().lower()


class LiveCausalAdapter:
    """Drop-in replacement for the subset of KnowledgeStore's read
    interface repl.py's fast paths call, backed by a LIVE-CAUSAL LiveGraph
    segment store instead of an in-memory fact list.

    Construction takes a store_dir (the LiveStore/LiveGraph directory a
    converter like convert_knowledge_full.py has already populated) --
    NOT a knowledge_full.json path. Conversion is a separate, explicit
    step (see convert_knowledge_full.py), never implicit inside this
    adapter's constructor, so "which store_dir is this adapter reading"
    is always an answerable, explicit question for a caller.
    """

    def __init__(self, store_dir, canon=False, nlp=None):
        self.store_dir = store_dir
        self.graph = LiveGraph(store_dir, canon=canon, nlp=nlp)
        self.evidence = EvidenceLedger(store_dir)
        self._ensure_evidence_backfilled()
        # repl.py-compatibility surface: many of repl.py's fast-path
        # solvers (_direct_kb_lookup, _solve_reasoning, _solve_compositional,
        # and ~80 more call sites as of this fix -- checked directly via
        # grep, not assumed) iterate `self.knowledge.facts` as a flat
        # (subject, relation, object) list, bypassing query()/find_by_entity
        # entirely. KnowledgeStore.facts is an in-memory Python list this
        # adapter has no equivalent of (LiveGraph stores segment-sealed
        # records, not a flat fact list) -- .facts below MATERIALIZES one,
        # once, at construction time, by walking every record in the store.
        # This is the adapter's one real scope compromise: it is a SNAPSHOT
        # (does not see records appended after construction) and an O(all
        # records) cost paid once up front rather than never -- honest
        # trade-off for an MVP that needs repl.py's existing fast-path
        # solvers to work at all, not a hidden performance cliff. For the
        # 4,855-record knowledge_full.json conversion this is sub-second;
        # it would NOT scale unchanged to a store with millions of records.
        self.dim = 128
        self.encoder = None
        self._encoder_type = 'live_causal_adapter_no_encoder'
        self._facts_cache = None

    @property
    def facts(self):
        if self._facts_cache is None:
            seen = set()
            out = []
            for sha in self.graph.store.segments():
                for _seg_sha, _idx, record in self.graph.store.iter_records(sha):
                    triple = (record.get('trigger'), record.get('mechanism'), record.get('outcome'))
                    if None in triple:
                        continue
                    if triple in seen:
                        continue
                    seen.add(triple)
                    out.append(triple)
            self._facts_cache = out
        return self._facts_cache

    def store_fact(self, subject, relation, obj):
        """Write path: converts one (s, r, o) fact into a livecausal_bridge
        record and appends it as its own one-record segment. Present so
        repl.py's learn-from-statement path (`_learn_from_statement`,
        which calls self.knowledge.store_fact for declarative user input)
        does not crash under the adapter -- NOT wired into the read-side
        facts cache automatically (self._facts_cache is NOT invalidated
        here) since repl.py's fast-path solvers reading .facts are not
        expected to observe a just-learned fact mid-session for this MVP;
        a caller needing that can del adapter._facts_cache to force a
        re-materialization on next access."""
        record = {
            "trigger": subject,
            "mechanism": relation,
            "outcome": obj,
            "trigger_key": _normalize(subject),
            "outcome_key": _normalize(obj),
            "doc_coord": "live_session:runtime",
            "evidence_count": 1,
            "use_count": 0,
            "meta": {"source": "repl_runtime_store_fact"},
        }
        sha = self.graph.append_segment([record])
        self.evidence.append_observations_for_segment(self.graph, sha)
        return sha

    def store_facts(self, facts):
        for s, r, o in facts:
            self.store_fact(s, r, o)

    def _ensure_evidence_backfilled(self):
        """One-time (idempotent) backfill: if the evidence ledger has no
        observations yet for segments already in the store (e.g. a store
        the converter populated before this adapter's first run, or
        before evidence.py's ledger existed at all), record one
        observation per base-edge-bearing record so evidence_count() has
        something to fold over. Safe to call every construction --
        EvidenceLedger.append_observation's dedup is by evidence_key
        content, so re-backfilling an already-observed segment just adds
        harmless duplicate lines the fold already collapses (documented
        in evidence.py's own append_observation docstring)."""
        valid_segments = set(self.graph.store.segments())
        observed_segments = set()
        # Cheap check: does the ledger already have ANY line citing each
        # segment? If a store has many segments this is O(ledger size)
        # once, not per-segment -- acceptable for an MVP backfill pass.
        for line in self.evidence._iter_lines():
            observed_segments.add(line["segment"])
        missing = valid_segments - observed_segments
        for sha in sorted(missing):
            self.evidence.append_observations_for_segment(self.graph, sha)

    # ------------------------------------------------------------------
    # KnowledgeStore-compatible interface
    # ------------------------------------------------------------------

    def query(self, subject=None, relation=None, max_tier=3):
        """KnowledgeStore.query()-compatible: same return dict shape
        (fact, confidence, confidence_level, attractor_distance,
        input_similarity, top2_gap, steps, thresholds). `max_tier` is
        accepted for signature compatibility but not consulted -- this
        adapter has no tiered fallback structure of its own (see module
        docstring: Hopfield Tier 3 fuzzy matching is out of scope)."""
        empty = {
            'fact': None,
            'confidence': 0.0,
            'confidence_level': 'REJECTED',
            'attractor_distance': 0.0,
            'input_similarity': 0.0,
            'top2_gap': 0.0,
            'steps': 0,
            'thresholds': (0.7, 0.4),
        }
        if not subject:
            return empty
        key = _normalize(subject)
        edges = self.graph.query(key)
        if relation:
            # Filter to edges whose citing record's mechanism matches the
            # requested relation. Reads the cited record fresh via the
            # store, same re-derivation discipline infer.py itself uses
            # (edge_keys_for_derivation) rather than trusting a cached
            # mechanism string anywhere.
            rel_lower = relation.lower().strip()
            edges = [e for e in edges if self._edge_mechanism_matches(e, rel_lower)]

        if not edges:
            return empty

        # Prefer a base (exact, depth-0) edge over an inferred one; among
        # base edges, prefer the one with the highest evidence_count.
        base_edges = [e for e in edges if e['kind'] == 'base']
        pick_pool = base_edges if base_edges else edges
        best = max(pick_pool, key=lambda e: self._evidence_count_for(e))

        return self._edge_to_query_result(best, key)

    def find_by_entity(self, entity):
        """KnowledgeStore.find_by_entity()-compatible: all (s, r, o)
        triples where `entity` appears as trigger_key (outgoing edges).
        Unlike KnowledgeStore, this adapter does not also scan for
        `entity` as an OBJECT (LiveGraph's base_rev index supports it,
        but repl.py's only caller of find_by_entity -- the anti-
        hallucination pipeline guard added in the Phase 1 fix -- only
        needs a truthy/falsy "is this entity known at all" signal, which
        outgoing-edge presence already answers; extending to reverse
        lookup is a one-line addition to this method if a future caller
        needs it, not attempted here to keep this MVP's surface honest
        about what it actually implements)."""
        key = _normalize(entity)
        edges = self.graph.query(key)
        results = []
        for e in edges:
            record = self._first_citing_record(e)
            if record is None:
                continue
            results.append((record['trigger'], record['mechanism'], record['outcome']))
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _first_citing_record(self, edge):
        derivation = edge['derivation']
        if not derivation:
            return None
        sha, idx = derivation[0]
        for _seg_sha, rec_idx, rec in self.graph.store.iter_records(sha):
            if rec_idx == idx:
                return rec
        return None

    def _edge_mechanism_matches(self, edge, rel_lower):
        for sha, idx in edge['derivation']:
            for _seg_sha, rec_idx, rec in self.graph.store.iter_records(sha):
                if rec_idx == idx and rec.get('mechanism', '').lower().strip() == rel_lower:
                    return True
        return False

    def _evidence_count_for(self, edge):
        edge_key = (edge['from_key'], edge['to_key'])
        valid_segments = self.graph.store.segments()
        return self.evidence.evidence_count(edge_key, valid_segments)

    def _edge_to_query_result(self, edge, queried_key):
        record = self._first_citing_record(edge)
        if record is None:
            return {
                'fact': None,
                'confidence': 0.0,
                'confidence_level': 'REJECTED',
                'attractor_distance': 0.0,
                'input_similarity': 0.0,
                'top2_gap': 0.0,
                'steps': 0,
                'thresholds': (0.7, 0.4),
            }
        fact = (record['trigger'], record['mechanism'], record['outcome'])
        ec = self._evidence_count_for(edge)

        if edge['kind'] == 'base':
            confidence = min(1.0, 0.5 + 0.1 * ec)
            confidence_level = 'HIGH'
            attractor_distance = 0.0
        else:
            depth = edge.get('depth', 2)
            depth_penalty = 1.0 + 0.1 * (depth - 2)
            confidence = min(0.85, 0.4 + 0.05 * ec) / depth_penalty
            confidence_level = 'MEDIUM'
            attractor_distance = 0.1 * (depth - 1)

        return {
            'fact': fact,
            'confidence': confidence,
            'confidence_level': confidence_level,
            'attractor_distance': attractor_distance,
            'input_similarity': confidence,
            'top2_gap': 1.0,
            'steps': 0,
            'thresholds': (0.7, 0.4),
            # Extra, adapter-specific field beyond KnowledgeStore's contract
            # -- documented as an addition, not a silent extension callers
            # must know about to use this adapter for the compatible subset.
            'evidence_count': ec,
            'edge_kind': edge['kind'],
        }

    # ------------------------------------------------------------------
    # Passthrough to the underlying store (for the demo script's cut/
    # append choreography -- not part of the KnowledgeStore-compat surface)
    # ------------------------------------------------------------------

    def segments(self):
        return self.graph.store.segments()

    def append_segment(self, records):
        return self.graph.append_segment(records)

    def drop_segments(self, shas):
        return self.graph.drop_segments(shas)
