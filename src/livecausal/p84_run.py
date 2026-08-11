"""P84 runner: the lazy organ on the store class that broke eager (ConceptNet full).

Lazy mode only — this store's eager materialization ran out of memory at 21GB,
so there is no eager arm here by design.

Clause (a): lazy mount peak RSS <= 8GB and wall <= 300s.
Clause (b): 50 uniform + 20 hub-targeted keys at node_budget=5000; at least one
hub truncates with the flag; every truncated answer well-formed; closure_calls 0.
Clause (c): base-edge subset of the lazy answer equals a direct raw-record scan
for 20 keys; a second lazy mount reproduces all sampled answers identically.

Run on a COPY of the store only (read-only-artifact rule).
"""
import argparse
import json
import os
import random
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from livecausal.infer import LiveGraph, _record_base_edge
except ModuleNotFoundError:
    # Flat bridge deployment (e.g. ~/livecausal_bridge with __init__.py):
    # the directory this script sits in IS the package.
    import importlib
    _here = os.path.dirname(os.path.abspath(__file__))
    _infer = importlib.import_module(os.path.basename(_here) + ".infer")
    LiveGraph = _infer.LiveGraph
    _record_base_edge = _infer._record_base_edge


def _rss_gb():
    ru = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KB, macOS bytes.
    return ru / 1e6 if sys.platform.startswith("linux") else ru / 1e9


def edge_sig(e):
    return (e["kind"], e["from_key"], e["to_key"], e.get("depth", 1),
            json.dumps(e["derivation"], sort_keys=True))


def well_formed(edges):
    need = {"kind", "from_key", "to_key", "derivation"}
    return all(need <= set(e) for e in edges)


def query_all(graph, keys, budget):
    out = {}
    for k in keys:
        t0 = time.perf_counter()
        edges, truncated = graph.query(k, node_budget=budget, return_truncated=True)
        out[k] = {"edges": edges, "truncated": bool(truncated),
                  "wall_s": round(time.perf_counter() - t0, 6)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--store", required=True, help="COPY of the store, never the artifact")
    ap.add_argument("--node-budget", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=84)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # --- clause (a): the mount that used to be impossible ---
    t0 = time.perf_counter()
    g = LiveGraph(args.store, inference="lazy", node_budget=args.node_budget,
                  count_closures=True)
    mount_s = time.perf_counter() - t0
    mount_rss = _rss_gb()

    keys = sorted(g._base_edges.keys())
    degrees = {k: len(g._base_edges[k]) for k in keys}
    deg_sorted = sorted(degrees.values())
    n = len(deg_sorted)
    degree_stats = {
        "n_keys": n,
        "max": deg_sorted[-1],
        "p99": deg_sorted[int(n * 0.99)],
        "median": deg_sorted[n // 2],
    }

    uniform = rng.sample(keys, min(50, len(keys)))
    hubs = [k for k, _d in sorted(degrees.items(), key=lambda kv: (-kv[1], kv[0]))[:20]]

    # --- clause (b): truncation on real hubs ---
    answers = query_all(g, uniform + hubs, args.node_budget)
    n_trunc_uniform = sum(answers[k]["truncated"] for k in uniform)
    n_trunc_hub = sum(answers[k]["truncated"] for k in hubs)
    all_wf = all(well_formed(a["edges"]) for a in answers.values())

    # --- clause (c1): base-edge ground truth from a raw-record scan ---
    check_keys = set(uniform[:20])
    expected = {k: {} for k in check_keys}
    for sha, idx, record in g.store.iter_records():
        edge = _record_base_edge(record)
        if edge is None:
            continue
        from_key, to_key = edge
        if from_key in check_keys:
            cites = expected[from_key].setdefault(to_key, [])
            if [sha, idx] not in cites:
                cites.append([sha, idx])
    base_mismatches = []
    for k in sorted(check_keys):
        lazy_base = {e["to_key"]: e["derivation"] for e in answers[k]["edges"]
                     if e["kind"] == "base"}
        exp = {t: sorted(c) for t, c in expected[k].items()}
        got = {t: sorted(d) for t, d in lazy_base.items()}
        if exp != got:
            base_mismatches.append({"key": k, "expected_n": len(exp), "got_n": len(got)})

    closure_calls_final = g.closure_calls

    # --- clause (c2): determinism across a second lazy mount ---
    g2 = LiveGraph(args.store, inference="lazy", node_budget=args.node_budget,
                   count_closures=True)
    answers2 = query_all(g2, uniform + hubs, args.node_budget)
    det_mismatches = [k for k in answers
                      if [edge_sig(e) for e in answers[k]["edges"]]
                      != [edge_sig(e) for e in answers2[k]["edges"]]
                      or answers[k]["truncated"] != answers2[k]["truncated"]]

    result = {
        "registered": "P84",
        "store": args.store,
        "seed": args.seed,
        "node_budget": args.node_budget,
        "clause_a": {
            "lazy_mount_s": round(mount_s, 3),
            "peak_rss_gb_after_mount": round(mount_rss, 3),
            "peak_rss_gb_final": round(_rss_gb(), 3),
            "bar_rss_le_8gb": mount_rss <= 8.0,
            "bar_wall_le_300s": mount_s <= 300.0,
        },
        "degree_stats": degree_stats,
        "clause_b": {
            "n_uniform": len(uniform),
            "n_hub": len(hubs),
            "n_truncated_uniform": n_trunc_uniform,
            "n_truncated_hub": n_trunc_hub,
            "bar_hub_truncates": n_trunc_hub >= 1,
            "all_well_formed": all_wf,
            "closure_calls": closure_calls_final,
            "bar_zero_closures": closure_calls_final == 0,
            "hub_degrees": [degrees[k] for k in hubs],
            "median_query_s": round(sorted(a["wall_s"] for a in answers.values())[len(answers) // 2], 6),
        },
        "clause_c": {
            "n_ground_truth_keys": len(check_keys),
            "base_mismatches": base_mismatches,
            "bar_base_exact": len(base_mismatches) == 0,
            "n_determinism_keys": len(answers),
            "determinism_mismatches": det_mismatches,
            "bar_deterministic": len(det_mismatches) == 0,
            "second_mount_closure_calls": g2.closure_calls,
        },
        "per_key": {k: {"n_edges": len(a["edges"]), "truncated": a["truncated"],
                        "wall_s": a["wall_s"], "degree": degrees.get(k, 0)}
                    for k, a in answers.items()},
    }
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    for c in ("clause_a", "degree_stats", "clause_b", "clause_c"):
        view = dict(result[c])
        view.pop("hub_degrees", None)
        print(c, json.dumps(view))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
