"""
Bridge: FERTIG's pipeline speaking directly over o1-state's LiveStore/
LiveGraph (src/livecausal), instead of a static .causal file.

DESIGN DECISION (per lead's request in the integration task): option (b),
a direct walk over LiveGraph.query(), not option (a) (export to .causal).

Why (b): LiveGraph.query(key) already returns exactly the shape FERTIG's
walk_chain needs -- {"to_key", "derivation": [[sha, idx], ...]} per outgoing
edge, with citations attached. This is BETTER than pipeline.load_graph's
static Dict[int,Dict[int,float]] adjacency: the store is LIVE (grows as the
builder appends, shrinks as segments are dropped/cut), and every hop the
walk takes already carries its (sha, idx) receipt -- exactly what the lead
asked to see in the output. Building this in "reasonable time" turned out
to be true: LiveGraph.query() plus one record lookup per hop (for the
mechanism word) is the entire adapter, no new inference logic needed.

Why NOT (a): a one-shot .causal export would need to be re-run every time
the store changes (stale unless re-triggered), and throws away the live
citation trail -- FERTIG's own .causal format has no (sha, idx) provenance
field, so an export would have to invent one, duplicating what LiveGraph
already tracks natively. (a) remains available as a fallback if a future
caller needs a portable static snapshot (e.g. to hand FERTIG's graph to a
process that can't import src/livecausal), documented here but not built.

Trade-off actually paid: LiveGraph's key space is (trigger_key, outcome_key)
STRING pairs (raw record keys, e.g. "tar buildup"), not FERTIG's own
integer-indexed vocab (Dict[str,int] stoi + Dict[int,Dict[int,float]] adj).
This adapter does NOT rebuild FERTIG's int-indexed structures -- it walks
LiveGraph's string-keyed adjacency directly and feeds
fertig.sampler.contraction_sample the same way pipeline.walk_chain does,
just working in string-key space instead of int space (contraction_sample
only needs a logits vector over CANDIDATE next-hops, which here are the
outgoing edges query() returns -- there is no need to materialize the
FULL vocab as an int array just to walk one store).

Mechanism words are NOT in LiveGraph.query()'s edge dicts (the base-edge
dict is from_key/to_key/derivation only) -- they live in the RECORD each
citation points at. This adapter resolves the mechanism for a hop by
looking up the record at derivation[0] = [sha, idx] via LiveStore's
segment reader. Cheap: one JSON-lines read per hop, not per query.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
# NOTE: src/livecausal/infer.py uses a relative import (`from .store import
# LiveStore`), so it must be imported as part of the `livecausal` package
# (`from livecausal.infer import LiveGraph`) with only `src/` on sys.path --
# adding `src/livecausal` itself to sys.path breaks that relative import.

from . import sampler  # noqa: E402


def _mechanism_for_citation(store, sha: str, idx: int) -> str:
    """Resolve the mechanism word for one (sha, idx) citation by reading
    the record it points at. store: src.livecausal.store.LiveStore."""
    try:
        records = store._load_segment_records(sha)  # noqa: SLF001 -- read-only, no public accessor for a single record by idx
        return records[idx].get("mechanism", "leads to")
    except (KeyError, IndexError, FileNotFoundError):
        return "leads to"


def live_walk_chain(
    graph,               # src.livecausal.infer.LiveGraph instance
    store,                # src.livecausal.store.LiveStore instance (same store_dir as graph)
    start_key: str,
    n: int = 8,
    tau: float = 0.3,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Walk LiveGraph.query() outgoing edges from start_key, contraction-
    sampled exactly like pipeline.walk_chain, but over the LIVE store's
    string-keyed adjacency instead of a static .causal file's int-indexed
    adjacency.

    Returns a list of hop dicts (in walk order):
        {"from_key", "to_key", "mechanism", "citation": {"sha", "idx"}}

    Honest dead ends: if query(cur) returns no outgoing edges, the walk
    stops (same "ehrliche Sackgasse" contract as pipeline.walk_chain --
    no fabricated continuation).
    """
    hops: List[Dict] = []
    cur = start_key
    rng = np.random.default_rng(seed)

    for _ in range(n):
        # query() returns BOTH base (single real edge, one citation) and
        # inferred (multi-hop transitive shortcut, derivation length > 1)
        # edges. A WALK must take base edges only -- an inferred edge is
        # already a summary of several hops the walk should traverse one
        # at a time (pipeline.walk_chain's static adjacency has no
        # inferred edges at all, so this filter is what makes a live walk
        # behave the same way: one real edge per step, not a shortcut).
        # Verified live: query("the playstation") returns a base edge to
        # "beginning" (1 citation) AND an inferred edge straight to
        # "the sign" (depth=2, 2 citations) -- taking the inferred edge
        # would have skipped the actual intermediate hop and its receipt.
        edges = [e for e in graph.query(cur) if e["kind"] == "base"]
        if not edges:
            break  # honest dead end -- LiveGraph has no further BASE edge

        # Build a logits vector over the candidate next-hops (by to_key),
        # weighted by confidence-proxy = number of citing records (a base
        # edge with more supporting records is more "confident" -- the
        # LiveStore's own evidence_count concept, read off the derivation
        # length here since query() already exposes it per edge).
        to_keys = [e["to_key"] for e in edges]
        weights = np.array([max(1, len(e["derivation"])) for e in edges], dtype=float)
        logits = np.log(weights + 1e-9)

        # contraction_sample expects a logits array indexed 0..len-1 and
        # returns an index into it -- reuse it exactly as pipeline.walk_chain
        # does, just over this hop's local candidate set instead of the
        # full vocab (no int-vocab needed for a string-keyed live walk).
        idx_choice = sampler.contraction_sample(logits, tau=tau, top_k=min(10, len(logits)))
        chosen_edge = edges[int(idx_choice)]
        to_key = chosen_edge["to_key"]

        citation_sha, citation_idx = chosen_edge["derivation"][0]
        mechanism = _mechanism_for_citation(store, citation_sha, citation_idx)

        hops.append({
            "from_key": cur,
            "to_key": to_key,
            "mechanism": mechanism,
            "citation": {"sha": citation_sha, "idx": citation_idx},
        })
        cur = to_key

    return hops


def live_canon_walk_chain(
    graph,               # src.livecausal.infer.LiveGraph instance, MUST be built with canon=True
    store,                # src.livecausal.store.LiveStore instance (same store_dir as graph)
    start_key: str,
    n: int = 8,
    tau: float = 0.3,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Canon-walk variant of live_walk_chain: steps over graph.canon_query()
    instead of graph.query() -- the SAME base-only principle applies (one
    real hop per step, no inferred shortcuts), but the adjacency joins
    canonicalized keys, not raw exact strings.

    Why this matters (measured on results/p72_store_local): the raw
    string-keyed adjacency has only 40/1944 from_keys that are ALSO a
    to_key somewhere else (2.1% chainable) -- most walks dead-end after one
    hop because "the sign" and "The Sign," and "signs" never join as the
    SAME node in exact-string space. The canon adjacency folds
    morphological/casing variants onto one canon_key BEFORE building
    edges, so the same underlying concept mentioned in different records
    joins up: measured chainable overlap jumps from 40 to 444 (11x) on the
    same store. _canon_base_edges never invents a hop -- it only changes
    which from/to key STRING two already-real citations get filed under
    (canon.canonical_key is a pinned, deterministic normalization: casing/
    lemma folding, not semantic guessing), so "canon joins more, but never
    fabricates an edge" holds structurally, the same honesty guarantee as
    the raw walk.

    Requires graph.canon_enabled (i.e. constructed as LiveGraph(store_dir,
    canon=True)) -- raises the same RuntimeError as canon_query() itself
    otherwise, no silent fallback to the raw walk.

    Returns the SAME hop-dict shape as live_walk_chain:
        {"from_key", "to_key", "mechanism", "citation": {"sha", "idx"}}
    but from_key/to_key are now CANON keys (not the raw record strings).
    The mechanism word still comes from the RAW cited record (canon_query's
    derivation always cites raw (sha, idx) coordinates, canonicalization
    never touches citations -- see canon_query's own docstring) via the
    same _mechanism_for_citation lookup live_walk_chain uses.
    """
    if not graph.canon_enabled:
        raise RuntimeError(
            "live_canon_walk_chain requires LiveGraph(..., canon=True); "
            "this graph was mounted with canon=False."
        )

    hops: List[Dict] = []
    # canon_query() canonicalizes its input itself (idempotent on an
    # already-canonical key, verified: canonical_key(canonical_key(x)) ==
    # canonical_key(x) for the fallback normalizer this repo pins), so a
    # raw start_key works on hop 0 and the already-canonical to_key from
    # hop i works unchanged as the query key for hop i+1.
    cur = start_key

    for _ in range(n):
        edges = [e for e in graph.canon_query(cur) if e["kind"] == "base"]
        if not edges:
            break  # honest dead end -- no further BASE edge in canon space either

        weights = np.array([max(1, len(e["derivation"])) for e in edges], dtype=float)
        logits = np.log(weights + 1e-9)
        idx_choice = sampler.contraction_sample(logits, tau=tau, top_k=min(10, len(logits)))
        chosen_edge = edges[int(idx_choice)]
        to_key = chosen_edge["to_key"]

        citation_sha, citation_idx = chosen_edge["derivation"][0]
        mechanism = _mechanism_for_citation(store, citation_sha, citation_idx)

        hops.append({
            "from_key": chosen_edge["from_key"],
            "to_key": to_key,
            "mechanism": mechanism,
            "citation": {"sha": citation_sha, "idx": citation_idx},
        })
        cur = to_key

    return hops


def verbalize_live_walk(hops: List[Dict]) -> str:
    """Verbalize a live_walk_chain() output using the SAME opener/polarity
    logic as pipeline.py's speech verbalizer, but reading mechanism/
    citation off the live hop dicts instead of the static mech table.
    Citations are appended per clause as [sha:idx8] -- the receipt travels
    with the sentence, per the lead's requirement."""
    from .pipeline import _CAUSE_OPENERS, _POS_VERBS, _NEG_VERBS, _CONTRAST

    if not hops:
        return ""

    parts = []
    for i, hop in enumerate(hops):
        subj = hop["from_key"].capitalize() if i == 0 else hop["from_key"]
        mech = hop["mechanism"]
        obj = hop["to_key"]
        cite = f"[{hop['citation']['sha'][:8]}:{hop['citation']['idx']}]"

        verb_word = mech.split()[-1] if mech.split() else mech
        if verb_word in _NEG_VERBS or verb_word in _POS_VERBS:
            clause = f"{subj} {mech} {obj}"
        else:
            clause = f"{subj} {mech} {obj}" if mech else f"{subj} leads to {obj}"

        if i > 0:
            opener = _CONTRAST[i % len(_CONTRAST)] if i % 2 else _CAUSE_OPENERS[i % len(_CAUSE_OPENERS)]
            clause = f"{opener} {clause[0].lower() + clause[1:]}"

        parts.append(f"{clause} {cite}")

    return ". ".join(parts) + "."
