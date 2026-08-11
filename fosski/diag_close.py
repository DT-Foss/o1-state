"""Find problems where our answer is wrong but close — off by a factor or small delta."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

off_by_factor = []  # pred = gold * N for some small integer
off_by_delta = []   # pred = gold ± small_number

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    if str(pred) == gold or pred == 'NULL':
        continue
    try:
        g = float(gold)
        p = float(pred)
    except:
        continue
    if g == 0:
        continue

    ratio = p / g
    for factor in [2, 3, 0.5, 0.1, 10, 4, 5]:
        if abs(ratio - factor) < 0.01:
            off_by_factor.append((i, q[:100], gold, pred, factor))
            break

    # Close answers (within 10% or off by 1-2)
    if abs(p - g) <= 2 and abs(p - g) > 0:
        off_by_delta.append((i, q[:100], gold, pred, p-g))

print(f"Off by factor: {len(off_by_factor)}")
by_factor = {}
for idx, q, g, p, f in off_by_factor:
    by_factor.setdefault(f, []).append((idx, q, g, p))
for f in sorted(by_factor.keys()):
    items = by_factor[f]
    print(f"\n  Factor {f}: {len(items)}")
    for idx, q, g, p in items[:5]:
        print(f"    #{idx} G={g} P={p} | {q}")

print(f"\n\nOff by small delta (<=2): {len(off_by_delta)}")
for idx, q, g, p, d in off_by_delta[:20]:
    print(f"  #{idx} G={g} P={p} (d={d:+.1f}) | {q}")
