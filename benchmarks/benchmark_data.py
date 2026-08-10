#!/usr/bin/env python3
"""Benchmark: Data Processing — CSV, JSON, Statistics"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test(name, passed, detail=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}" + (f" — {detail}" if detail else ""))
    return 1 if passed else 0


def run():
    print("=" * 60)
    print("BENCHMARK: Data Processing")
    print("=" * 60)
    total = 0
    passed = 0

    from core.data import (
        DataTable, mean, median, std, percentile, mode,
        pearson_correlation, describe, flatten_json, json_query,
    )

    # === Statistics ===
    print("\n--- Statistics ---")
    total += 1
    ok = mean([1, 2, 3, 4, 5]) == 3.0
    passed += test("Mean", ok)

    total += 1
    ok = median([1, 2, 3, 4, 5]) == 3.0 and median([1, 2, 3, 4]) == 2.5
    passed += test("Median", ok)

    total += 1
    s = std([2, 4, 4, 4, 5, 5, 7, 9])
    ok = 2.0 < s < 2.2  # Sample std (n-1) = 2.138
    passed += test("Standard deviation", ok, f"std={s:.3f}")

    total += 1
    p = percentile([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50)
    ok = 5.0 <= p <= 5.5
    passed += test("Percentile (50th)", ok, f"p50={p}")

    total += 1
    ok = mode([1, 2, 2, 3, 3, 3]) == 3
    passed += test("Mode", ok)

    total += 1
    r = pearson_correlation([1, 2, 3, 4, 5], [2, 4, 6, 8, 10])
    ok = r > 0.99
    passed += test("Pearson correlation (perfect)", ok, f"r={r:.4f}")

    total += 1
    r = pearson_correlation([1, 2, 3, 4, 5], [5, 4, 3, 2, 1])
    ok = r < -0.99
    passed += test("Pearson correlation (inverse)", ok, f"r={r:.4f}")

    total += 1
    d = describe([1, 2, 3, 4, 5])
    ok = d['count'] == 5 and d['mean'] == 3.0 and d['min'] == 1
    passed += test("Describe", ok, f"mean={d['mean']}, std={d['std']}")

    # === CSV ===
    print("\n--- CSV Processing ---")
    csv_data = """name,age,score
Alice,25,85.5
Bob,30,92.0
Charlie,22,78.3
Diana,28,95.1
Eve,35,88.7"""

    total += 1
    dt = DataTable.from_csv(csv_data)
    ok = len(dt) == 5 and len(dt.columns) == 3
    passed += test("CSV parsing", ok, f"{len(dt)} rows, cols={dt.columns}")

    total += 1
    ok = dt.rows[0]['age'] == 25 and isinstance(dt.rows[0]['score'], float)
    passed += test("Type inference", ok, f"age={dt.rows[0]['age']} ({type(dt.rows[0]['age']).__name__})")

    total += 1
    ages = dt.column_numeric('age')
    ok = mean(ages) == 28.0
    passed += test("Column extraction", ok, f"mean_age={mean(ages)}")

    total += 1
    filtered = dt.filter(lambda r: r['score'] > 90)
    ok = len(filtered) == 2  # Bob (92) and Diana (95.1)
    passed += test("Filter (score > 90)", ok, f"{len(filtered)} rows")

    total += 1
    sorted_dt = dt.sort_by('score', reverse=True)
    ok = sorted_dt.rows[0]['name'] == 'Diana'
    passed += test("Sort by score desc", ok, f"top={sorted_dt.rows[0]['name']}")

    total += 1
    agg = dt.aggregate('score', 'mean')
    ok = 87 < agg < 89
    passed += test("Aggregate mean(score)", ok, f"mean={agg:.1f}")

    total += 1
    dt2 = dt.add_column('grade', lambda r: 'A' if r['score'] >= 90 else 'B')
    ok = dt2.rows[1]['grade'] == 'A'  # Bob has 92
    passed += test("Add computed column", ok)

    total += 1
    vc = dt.value_counts('name')
    ok = len(vc) == 5  # All unique
    passed += test("Value counts", ok, f"{len(vc)} unique names")

    total += 1
    csv_out = dt.to_csv()
    ok = 'name,age,score' in csv_out and 'Alice' in csv_out
    passed += test("CSV export", ok, f"{len(csv_out)} chars")

    # === JSON ===
    print("\n--- JSON Processing ---")
    total += 1
    json_data = '[{"x": 1, "y": "a"}, {"x": 2, "y": "b"}, {"x": 3, "y": "c"}]'
    dt_j = DataTable.from_json(json_data)
    ok = len(dt_j) == 3
    passed += test("JSON array parsing", ok, f"{len(dt_j)} rows")

    total += 1
    nested = {"user": {"name": "Alice", "address": {"city": "Oslo"}}, "scores": [1, 2, 3]}
    flat = flatten_json(nested)
    ok = flat.get('user.name') == 'Alice' and flat.get('user.address.city') == 'Oslo'
    passed += test("JSON flattening", ok, f"keys={list(flat.keys())}")

    total += 1
    result = json_query(nested, 'user.address.city')
    ok = result == 'Oslo'
    passed += test("JSON query (dot path)", ok, f"result={result}")

    total += 1
    result = json_query(nested, 'scores[1]')
    ok = result == 2
    passed += test("JSON query (array index)", ok, f"result={result}")

    # === Advanced ===
    print("\n--- Advanced Operations ---")
    total += 1
    r = dt.correlation('age', 'score')
    ok = isinstance(r, float)
    passed += test("Correlation (age vs score)", ok, f"r={r:.3f}")

    total += 1
    grouped = dt.group_by('name')
    ok = len(grouped) == 5
    passed += test("Group by", ok, f"{len(grouped)} groups")

    total += 1
    summary = dt.summary()
    ok = 'DataTable' in summary and 'age' in summary
    passed += test("Table summary", ok)

    total += 1
    table_str = dt.print_table()
    ok = 'Alice' in table_str and '---' in table_str
    passed += test("ASCII table formatting", ok)

    total += 1
    deduped = DataTable.from_csv("a,b\n1,2\n1,2\n3,4").deduplicate()
    ok = len(deduped) == 2
    passed += test("Deduplication", ok, f"{len(deduped)} rows after dedup")

    # Summary
    print("\n" + "=" * 60)
    pct = (passed / total * 100) if total else 0
    print(f"RESULT: {passed}/{total} ({pct:.0f}%)")
    print("=" * 60)
    return passed, total


if __name__ == '__main__':
    run()
