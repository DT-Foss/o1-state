"""P83 runner: the lazy organ on the real dense artifact.

Clause (a): lazy vs eager mount wall on the same store copy (cold, no cache).
Clause (b): 50 seeded keys — untruncated lazy answers equal eager exactly,
truncated ones are strict subsets with the flag up.
Clause (c): closure_calls == 0 on the lazy graph across mount + queries +
10 seeded drops; post-drop re-queries cite no dropped sha.

Run on a COPY of the store only (read-only-artifact rule).
"""
import argparse
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livecausal.infer import LiveGraph


def edge_sig(e):
    return (e["kind"], e["from_key"], e["to_key"], e.get("depth", 1),
            json.dumps(e["derivation"], sort_keys=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="COPY of the store, never the artifact")
    ap.add_argument("--n-keys", type=int, default=50)
    ap.add_argument("--n-drops", type=int, default=10)
    ap.add_argument("--node-budget", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=83)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cache = os.path.join(args.store, "inferred.jsonl")
    assert not os.path.exists(cache), "cold-mount contract: remove inferred.jsonl from the copy first"

    # --- clause (a): mounts. FS cache warmed for BOTH arms first (else the
    # second mount reads warm and the comparison measures the disk, not the
    # closure); "cold" in the register means no inferred.jsonl, not cold FS. ---
    for root, _dirs, files in os.walk(args.store):
        for fn in files:
            with open(os.path.join(root, fn), "rb") as fh:
                fh.read()

    t0 = time.perf_counter()
    lazy = LiveGraph(args.store, inference="lazy", node_budget=args.node_budget,
                     count_closures=True)
    lazy_mount_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    eager = LiveGraph(args.store, inference="eager")
    eager_mount_s = time.perf_counter() - t0

    keys = sorted(lazy._base_edges.keys())
    sample = rng.sample(keys, min(args.n_keys, len(keys)))

    # --- clause (b): agreement on the sample ---
    per_key = []
    n_exact = n_subset = n_truncated = n_fail = 0
    for k in sample:
        t0 = time.perf_counter()
        lazy_edges, truncated = lazy.query(k, return_truncated=True)
        q_s = time.perf_counter() - t0
        eager_edges = eager.query(k)
        ls = {edge_sig(e) for e in lazy_edges}
        es = {edge_sig(e) for e in eager_edges}
        if truncated:
            ok = ls < es  # strict subset: something was genuinely cut
            n_truncated += 1
            n_subset += int(ok)
        else:
            ok = ls == es
            n_exact += int(ok)
        n_fail += int(not ok)
        per_key.append({"key": k, "truncated": bool(truncated), "ok": bool(ok),
                        "n_lazy": len(ls), "n_eager": len(es),
                        "lazy_query_s": round(q_s, 6)})

    # --- clause (c): drops on the lazy graph, citations must vanish ---
    shas = lazy.store.segments()
    dropped = rng.sample(shas, min(args.n_drops, len(shas)))
    lazy.drop_segments(dropped)
    dropped_set = set(dropped)
    stale = []
    recheck_keys = [k for k in sample if k in lazy._base_edges][:20]
    for k in recheck_keys:
        edges, _tr = lazy.query(k, return_truncated=True)
        for e in edges:
            for sha, _idx in e["derivation"]:
                if sha in dropped_set:
                    stale.append({"key": k, "sha": sha})

    result = {
        "registered": "P83",
        "store": args.store,
        "seed": args.seed,
        "node_budget": args.node_budget,
        "clause_a": {
            "lazy_mount_s": round(lazy_mount_s, 4),
            "eager_mount_s": round(eager_mount_s, 4),
            "speedup": round(eager_mount_s / lazy_mount_s, 2),
            "bar_speedup_ge_5": eager_mount_s / lazy_mount_s >= 5.0,
            "bar_lazy_le_5s": lazy_mount_s <= 5.0,
        },
        "clause_b": {
            "n_keys": len(sample),
            "n_exact_equal": n_exact,
            "n_truncated": n_truncated,
            "n_truncated_strict_subset": n_subset,
            "n_fail": n_fail,
            "bar_all_agree": n_fail == 0,
            "median_lazy_query_s": round(sorted(p["lazy_query_s"] for p in per_key)[len(per_key) // 2], 6),
        },
        "clause_c": {
            "closure_calls_lazy": lazy.closure_calls,
            "bar_zero_closures": lazy.closure_calls == 0,
            "n_dropped": len(dropped),
            "n_recheck_keys": len(recheck_keys),
            "stale_citations": stale,
            "bar_no_stale": len(stale) == 0,
        },
        "per_key": per_key,
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    for c in ("clause_a", "clause_b", "clause_c"):
        view = {k: v for k, v in result[c].items() if k != "stale_citations" or v}
        print(c, json.dumps(view))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
