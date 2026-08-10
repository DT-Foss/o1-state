#!/usr/bin/env python3
"""
LIVE-CAUSAL DEMO CLI — the three-command investor demo.

A thin wrapper over the finished pieces (builder_run, LiveGraph, the
evidence ledgers, the direction-3 stranger verifier). No new logic here:
every command below calls into an already-built, already-tested module and
formats its output as a plain, aligned-column report.

Commands:
  build   --text-file FILE --store DIR
              runs builder_run's offline text-file path (the real fabel
              extractor via curator_yield_run.extract_validated), prints a
              summary table when done.
  query   --store DIR --key KEY
              LiveGraph.query(key) + evidence-ledger folds per edge
              (evidence_count / use_count / contested), plus a human-
              readable derivation-chain listing with segment citations.
  verify  --store DIR [--n 10]
              runs src/stranger_verify_run.py's direction-3 engine-
              distrustful re-derivation check, prints the check table +
              verdict line.
  cut     --store DIR --segment SHA
              LiveGraph.drop_segments([SHA]) (which calls on_drop, then
              the store's own drop) and prints exactly what got
              invalidated — the live property, made visible.

Usage:
  python3 -m livecausal.demo build --text-file korpus.txt --store /tmp/demo_store
  python3 -m livecausal.demo query --store /tmp/demo_store --key "smoking"
  python3 -m livecausal.demo verify --store /tmp/demo_store --n 10
  python3 -m livecausal.demo cut --store /tmp/demo_store --segment <sha256>
"""
import argparse
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(REPO_ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from livecausal.infer import LiveGraph  # noqa: E402
from livecausal import evidence as ev  # noqa: E402
import livecausal.builder_run as builder_run  # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
#  small formatting helpers (plain text, aligned columns, no color/emoji)
# ─────────────────────────────────────────────────────────────────────────
def _rule(width=74):
    print("-" * width)


def _kv_table(pairs):
    """pairs: list of (label, value). Prints a two-column table, labels
    left-aligned and padded to the longest label."""
    if not pairs:
        return
    w = max(len(str(k)) for k, _ in pairs)
    for k, v in pairs:
        print("  {:<{w}} : {}".format(k, v, w=w))


# ─────────────────────────────────────────────────────────────────────────
#  build
# ─────────────────────────────────────────────────────────────────────────
def cmd_build(args):
    """Thin wrapper over builder_run.build_window_iterator / resolve_extractor
    / run_builder — the exact same offline text-file path builder_run.main()
    drives, just with a summary table printed at the end instead of
    builder_run's own one-line final print."""
    import torch  # noqa: E402
    torch.set_num_threads(1)
    import portable_organism as po  # noqa: E402

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    # A Namespace matching what build_window_iterator/run_builder expect
    # (mirrors builder_run.main()'s argparse defaults) -- this command is
    # the --text-file / offline path only, per the demo brief ("build a
    # mini corpus"); the online --source path stays builder_run.py's own
    # CLI surface, unchanged.
    ns = argparse.Namespace(
        text_file=args.text_file,
        source=None,
        window_tokens=args.window_tokens,
        d_model=args.d_model,
        batch=args.batch,
        chunk_size=args.chunk_size,
        q=args.q,
        window=args.window,
        min_window=args.min_window,
        ignition_chunks=args.ignition_chunks,
        chunks=None,
        tape_cap=args.tape_cap,
    )

    po.D_MODEL, po.BATCH, po.CHUNK = ns.d_model, ns.batch, ns.chunk_size
    po.GATE_Q, po.GATE_WINDOW, po.MIN_WINDOW, po.IGNITION_CHUNKS = (
        ns.q, ns.window, ns.min_window, ns.ignition_chunks,
    )

    extractor = builder_run.resolve_extractor(None)  # real fabel bridge, per the brief
    window_iter, stream = builder_run.build_window_iterator(ns, organism_seed=args.seed)

    status_path = os.path.join(args.store, ".demo_build_status.json")
    metrics_path = os.path.join(args.store, ".demo_build_metrics.jsonl")
    os.makedirs(args.store, exist_ok=True)

    t0 = time.time()
    graph, metrics = builder_run.run_builder(
        args.store,
        status_path,
        metrics_path,
        window_iter,
        extractor,
        args.windows_per_segment,
        max_windows=args.max_windows,
        max_seconds=args.max_seconds,
        print_every=args.print_every,
        stream=stream,
    )
    wall = time.time() - t0

    # FLAGGED (per the build brief: "fehlt dir ein kleiner Getter, flagge
    # ihn"): builder_run.run_builder never calls
    # EvidenceLedger.append_observations_for_segment (MVP-7's own intended
    # hook, per evidence.py's docstring: "Mirrors infer.py's on_append
    # shape... without touching infer.py itself") -- the evidence/use
    # ledgers exist but nothing in the builder loop feeds them, so
    # evidence_count would read 0 for every edge after a fresh build. Not
    # fixed inside builder_run.py (forbidden: "bestehende Module ändern
    # (nur importieren)") -- this wrapper closes the gap itself, once,
    # after the loop finishes, by replaying every sealed segment through
    # the ledger's own documented entry point. Idempotent (re-running
    # `build` against the same store re-observes the same segments, which
    # the ledger's evidence_count fold already dedupes by evidence_key).
    evidence_ledger = ev.EvidenceLedger(args.store)
    for sha in graph.store.segments():
        evidence_ledger.append_observations_for_segment(graph, sha)

    print()
    print("=" * 74)
    print("BUILD SUMMARY — {}".format(args.store))
    _rule()
    n_tokens = metrics["n_windows_total"] * args.window_tokens
    _kv_table([
        ("tokens streamed (approx, windows x window_tokens)", "{:,}".format(n_tokens)),
        ("windows (gated / total)", "{:,} / {:,}".format(
            metrics["n_windows_gated"], metrics["n_windows_total"])),
        ("validated triplets (from gated / total)", "{:,} / {:,}".format(
            metrics["n_triplets_from_gated"], metrics["n_triplets_total"])),
        ("segments sealed", "{:,}".format(metrics["n_segments"])),
        ("base edges", "{:,}".format(metrics["n_base_edges"])),
        ("inferred edges", "{:,}".format(metrics["n_inferred_edges"])),
        ("wall clock", "{:.2f}s".format(wall)),
    ])
    _rule()
    print("store: {}".format(args.store))


# ─────────────────────────────────────────────────────────────────────────
#  query
# ─────────────────────────────────────────────────────────────────────────
def _conflict_pairs_for_edge(graph, from_key, to_key):
    """A base pair (from_key, to_key) can be cited by records with DIFFERENT
    mechanism strings -- evidence.py's module docstring names this exact
    case as its 'contradiction' shape, since edge_key collapses to
    (from_key, to_key) with no mechanism dimension. This is a query-time
    read only (no new identity invented): group the pair's citing records
    by mechanism and report the distinct mechanism strings seen, so a
    'contested' read has something concrete to point at."""
    mechanisms = set()
    for sha, idx in graph.base_edge_citations(from_key, to_key):
        for seg_sha, rec_idx, record in graph.store.iter_records(sha):
            if rec_idx == idx:
                mech = record.get("mechanism")
                if mech:
                    mechanisms.add(mech)
                break
    return mechanisms


def _format_derivation(graph, derivation):
    """A -> B -> C style human-readable chain for an inferred edge's
    derivation, with each hop's segment citation and evidence_sentence
    (if the record carries one -- curator_yield_run.extract_validated's
    triplets do)."""
    if not derivation:
        return "(no hops)"
    hops = []
    keys = graph.edge_keys_for_derivation(derivation)
    for (from_key, to_key), (sha, idx) in zip(keys, derivation):
        hops.append((from_key, to_key, sha, idx))
    chain_keys = [hops[0][0]] + [h[1] for h in hops]
    chain_str = " -> ".join(chain_keys)
    lines = ["    {}".format(chain_str)]
    for from_key, to_key, sha, idx in hops:
        cite = "        [{}...:{}] {} -> {}".format(sha[:12], idx, from_key, to_key)
        record = None
        for seg_sha, rec_idx, rec in graph.store.iter_records(sha):
            if rec_idx == idx:
                record = rec
                break
        if record is not None:
            evsent = record.get("meta", {}).get("evidence_sentence") or record.get("evidence_sentence")
            quote = record.get("trigger", "") + " " + record.get("mechanism", "") + " " + record.get("outcome", "")
            cite += "  ({})".format((evsent or quote).strip())
        lines.append(cite)
    return "\n".join(lines)


def cmd_query(args):
    graph = LiveGraph(args.store)
    edges = graph.query(args.key)

    valid_segments = graph.store.segments()
    evidence_ledger = ev.EvidenceLedger(args.store)
    use_ledger = ev.UseLedger(args.store)

    print("=" * 74)
    print("QUERY — key: {!r}  (store: {})".format(args.key, args.store))
    _rule()

    if not edges:
        print("  (no outgoing edges for this key)")
        return

    # Group by (kind, from_key, to_key, depth): the same (from_key, to_key)
    # pair, especially at inferred depth >= 2, typically has MANY distinct
    # derivations (one per citing-record combination along the chain) --
    # each is a real, independently re-derivable edge (MVP-2's identity),
    # but listing all of them is a dump, not a demo. Show one representative
    # chain per group plus a derivation count; the full set is still exactly
    # what graph.query() returns, nothing hidden, just not all printed.
    groups = {}
    order = []
    for edge in edges:
        gkey = (edge["kind"], edge["from_key"], edge["to_key"], edge.get("depth"))
        if gkey not in groups:
            groups[gkey] = []
            order.append(gkey)
        groups[gkey].append(edge)

    for gkey in order:
        kind, from_key, to_key, depth = gkey
        group_edges = groups[gkey]
        representative = group_edges[0]

        edge_key = (from_key, to_key)
        evidence_count = evidence_ledger.evidence_count(edge_key, valid_segments)
        use_count = use_ledger.use_count(edge_key, valid_segments)

        mechanisms = _conflict_pairs_for_edge(graph, from_key, to_key)
        contested_flag = len(mechanisms) > 1  # >1 distinct mechanism on this pair -> a real conflict

        header = "  [{}] {} -> {}".format(kind.upper(), from_key, to_key)
        if kind == "inferred":
            header += "  (depth={})".format(depth)
        if len(group_edges) > 1:
            header += "  [{} derivations, showing 1]".format(len(group_edges))
        print(header)
        _kv_table([
            ("evidence_count", evidence_count),
            ("use_count", use_count),
            ("contested", "{}{}".format(
                contested_flag,
                "  (mechanisms: {})".format(", ".join(sorted(mechanisms))) if contested_flag else "")),
        ])
        if kind == "inferred":
            print(_format_derivation(graph, representative["derivation"]))
        else:
            for sha, idx in representative["derivation"][:3]:
                print("    [{}...:{}]".format(sha[:12], idx))
            if len(representative["derivation"]) > 3:
                print("    ... and {} more citation(s)".format(len(representative["derivation"]) - 3))
        print()


# ─────────────────────────────────────────────────────────────────────────
#  verify
# ─────────────────────────────────────────────────────────────────────────
def cmd_verify(args):
    import stranger_verify_run as svr  # noqa: E402

    manifest_path = os.path.join(args.store, "manifest.json")
    if not os.path.isdir(args.store) or not os.path.exists(manifest_path):
        print("VERIFY — no LiveStore at {} (no manifest.json)".format(args.store))
        sys.exit(1)

    checks = svr.verify_direction3(args.store, args.n, args.seed,
                                    with_corpus_replay=args.with_corpus_replay)
    n_target = max(10, args.n) if args.n >= 10 else args.n
    scoring = svr.score(checks, n_target)

    print()
    print("=" * 74)
    print("VERIFY (direction 3, engine-distrustful stranger re-derivation) — {}".format(args.store))
    _rule()
    print("  {:<8} {:<7} {:<40} {:<10} {:<10}".format("class", "found", "edge", "verify", "consensus"))
    for c in checks:
        if c["class"] == "base":
            edge_str = "{} -> {}".format(c["edge"]["from_key"], c["edge"]["to_key"])
        else:
            edge_str = "{} -> {} (depth={})".format(
                c["edge"]["from_key"], c["edge"]["to_key"], c["edge"]["depth"])
        print("  {:<8} {:<7} {:<40} {:<10} {:<10}".format(
            c["class"], "-", edge_str[:40],
            "OK" if c["found"] else "FAIL",
            "OK" if c["consensus"] else "FAIL"))
    _rule()
    _kv_table([
        ("sampled", scoring["n_sampled"]),
        ("verified", "{} (pass={})".format(scoring["p60a_fraction"], scoring["p60a_verified_all"])),
        ("consensus", "{} (pass={})".format(scoring["p60b_fraction"], scoring["p60b_consensus_all"])),
    ])
    _rule()
    verdict = "VERDICT: PASS" if (scoring["p60a_verified_all"] and scoring["p60b_consensus_all"]) else "VERDICT: FAIL"
    print(verdict)


# ─────────────────────────────────────────────────────────────────────────
#  cut
# ─────────────────────────────────────────────────────────────────────────
def cmd_cut(args):
    graph = LiveGraph(args.store)

    before_base = sum(len(v) for v in graph._base_edges.values())
    before_inferred = set(
        (e["from_key"], e["to_key"], e["depth"], tuple(tuple(p) for p in e["derivation"]))
        for e in graph.inferred_edges()
    )

    graph.drop_segments([args.segment])

    after_base = sum(len(v) for v in graph._base_edges.values())
    after_inferred = set(
        (e["from_key"], e["to_key"], e["depth"], tuple(tuple(p) for p in e["derivation"]))
        for e in graph.inferred_edges()
    )

    removed_inferred = before_inferred - after_inferred

    print("=" * 74)
    print("CUT — dropped segment {} from {}".format(args.segment, args.store))
    _rule()
    _kv_table([
        ("base edges", "{} -> {}".format(before_base, after_base)),
        ("inferred edges", "{} -> {}".format(len(before_inferred), len(after_inferred))),
        ("inferred edges invalidated", len(removed_inferred)),
    ])
    if removed_inferred:
        print()
        print("  invalidated inferred edges (derivation cited the dropped segment):")
        for from_key, to_key, depth, derivation in sorted(removed_inferred):
            print("    {} -> {} (depth={})".format(from_key, to_key, depth))
    _rule()
    print("no re-inference of the surviving graph — this is the live property: "
          "cut removes exactly what it cites, nothing else.")


# ─────────────────────────────────────────────────────────────────────────
#  argv
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="LIVE-CAUSAL demo CLI")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_build = sub.add_parser("build", help="stream a corpus, extract, store, infer")
    p_build.add_argument("--text-file", required=True)
    p_build.add_argument("--store", required=True)
    p_build.add_argument("--window-tokens", type=int, default=32)
    p_build.add_argument("--windows-per-segment", type=int, default=5)
    p_build.add_argument("--d-model", type=int, default=64)
    p_build.add_argument("--batch", type=int, default=4)
    p_build.add_argument("--chunk-size", type=int, default=32)
    p_build.add_argument("--seed", type=int, default=42)
    p_build.add_argument("--q", type=float, default=0.75)
    p_build.add_argument("--window", type=int, default=50)
    p_build.add_argument("--min-window", type=int, default=10)
    p_build.add_argument("--ignition-chunks", type=int, default=5)
    p_build.add_argument("--max-windows", type=int, default=None)
    p_build.add_argument("--max-seconds", type=float, default=None)
    p_build.add_argument("--tape-cap", type=int, default=200_000)
    p_build.add_argument("--print-every", type=int, default=10)
    p_build.set_defaults(func=cmd_build)

    p_query = sub.add_parser("query", help="query a key: base+inferred edges, evidence, derivations")
    p_query.add_argument("--store", required=True)
    p_query.add_argument("--key", required=True)
    p_query.set_defaults(func=cmd_query)

    p_verify = sub.add_parser("verify", help="direction-3 stranger re-derivation check")
    p_verify.add_argument("--store", required=True)
    p_verify.add_argument("--n", type=int, default=10)
    p_verify.add_argument("--seed", type=int, default=60)
    p_verify.add_argument("--with-corpus-replay", action="store_true")
    p_verify.set_defaults(func=cmd_verify)

    p_cut = sub.add_parser("cut", help="drop a segment, show what gets invalidated")
    p_cut.add_argument("--store", required=True)
    p_cut.add_argument("--segment", required=True)
    p_cut.set_defaults(func=cmd_cut)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
