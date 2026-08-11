"""P76 scoring: density-normalized yield law from the WT-103 full build telemetry.

Clause (a): per-fifth r_b = sum(n_new_inferred) / sum(density_pre * n_new_base),
bar max/min <= 3 over fifths 2-5; contrast: raw per-base yield spread >= 5x.
Clause (b): descriptive cost curve, median append_seconds per fifth with density.
"""
import json
import statistics


METRICS = "/Users/bhkmie/Documents/Forschung/O1_juli/results/wt103_full_metrics.jsonl"
OUT = "/Users/bhkmie/Documents/Forschung/O1_juli/results/livecausal_p76_yield.json"

rows = []
with open(METRICS) as f:
    for line in f:
        r = json.loads(line)
        if r.get("type") == "append":
            rows.append(r)

n = len(rows)
fifth = n // 5
buckets = [rows[i * fifth: (i + 1) * fifth if i < 4 else n] for i in range(5)]

per_fifth = []
prev_base, prev_inf = 0, 0
idx = 0
for b_i, bucket in enumerate(buckets):
    num, den, raw_base, raw_inf = 0.0, 0.0, 0, 0
    costs = []
    for r in bucket:
        if idx == 0:
            density_pre = 0.0
        else:
            pb = rows[idx - 1]["n_base_edges"]
            pi = rows[idx - 1]["n_inferred_edges"]
            density_pre = (pi / pb) if pb > 0 else 0.0
        n_new_base = r["n_base_edges"] - (rows[idx - 1]["n_base_edges"] if idx else 0)
        n_new_inf = r["n_inferred_edges"] - (rows[idx - 1]["n_inferred_edges"] if idx else 0)
        num += n_new_inf
        den += density_pre * n_new_base
        raw_base += n_new_base
        raw_inf += n_new_inf
        costs.append(r["append_seconds"])
        idx += 1
    last = bucket[-1]
    per_fifth.append({
        "fifth": b_i + 1,
        "n_appends": len(bucket),
        "sum_new_inferred": raw_inf,
        "sum_new_base": raw_base,
        "sum_density_x_base": round(den, 3),
        "r_b": round(num / den, 4) if den > 0 else None,
        "raw_yield_per_base": round(raw_inf / raw_base, 5) if raw_base else None,
        "median_append_s": round(statistics.median(costs), 5),
        "density_at_end": round(last["n_inferred_edges"] / last["n_base_edges"], 4),
        "base_edges_at_end": last["n_base_edges"],
        "inferred_edges_at_end": last["n_inferred_edges"],
    })

r_vals = [pf["r_b"] for pf in per_fifth[1:]]
raw_vals = [pf["raw_yield_per_base"] for pf in per_fifth[1:]]
r_spread = max(r_vals) / min(r_vals)
raw_spread = max(raw_vals) / min(raw_vals)

result = {
    "registered": "P76",
    "n_appends": n,
    "final_base": rows[-1]["n_base_edges"],
    "final_inferred": rows[-1]["n_inferred_edges"],
    "final_density": round(rows[-1]["n_inferred_edges"] / rows[-1]["n_base_edges"], 4),
    "per_fifth": per_fifth,
    "clause_a": {
        "r_b_fifths_2_5": r_vals,
        "r_spread_max_over_min": round(r_spread, 3),
        "bar_r_spread_le_3": r_spread <= 3.0,
        "raw_yield_fifths_2_5": raw_vals,
        "raw_spread_max_over_min": round(raw_spread, 3),
        "contrast_raw_spread_ge_5": raw_spread >= 5.0,
    },
    "clause_b_cost_medians_s": [
        {"fifth": pf["fifth"], "median_append_s": pf["median_append_s"],
         "density_at_end": pf["density_at_end"]} for pf in per_fifth
    ],
}

with open(OUT, "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result["clause_a"], indent=2))
print("cost:", [(c["fifth"], c["median_append_s"], c["density_at_end"]) for c in result["clause_b_cost_medians_s"]])
print("wrote", OUT)
