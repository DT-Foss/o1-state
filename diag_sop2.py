"""Show SOP correct cases to understand what to preserve."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

right = []
wrong = []

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []
    is_sop = any('SOP:' in s for s in steps)
    if not is_sop:
        continue
    if str(pred) == gold:
        right.append((i, q, gold, str(pred), steps))
    else:
        wrong.append((i, q, gold, str(pred), steps))

print(f"=== CORRECT SOP ({len(right)}) ===")
for idx, q, gold, pred, steps in right:
    sop_step = [s for s in steps if 'SOP:' in s][0]
    print(f"#{idx}: GOLD={gold} | {sop_step}")
    print(f"  Q: {q[:200]}")
    print()

# Categorize wrong by question type
print(f"\n=== WRONG SOP CATEGORIES ({len(wrong)}) ===")
comparison = []
multiperiod = []
unit_rate = []
other = []
for idx, q, gold, pred, steps in wrong:
    ql = q.lower()
    if re.search(r'how much (?:more|less|cheaper|expensive)', ql) or 'difference' in ql:
        comparison.append((idx, q[:80], gold, pred))
    elif re.search(r'per (?:week|month|day|year|hour|minute)', ql) and re.search(r'(?:week|month|day|year|hour|minute)s?[\s,.]', ql):
        multiperiod.append((idx, q[:80], gold, pred))
    elif re.search(r'\d+\s+\w+\s+cost\s+\$\d+', ql):
        unit_rate.append((idx, q[:80], gold, pred))
    else:
        other.append((idx, q[:80], gold, pred))

print(f"Comparison (more/less/diff): {len(comparison)}")
for idx, q, g, p in comparison:
    print(f"  #{idx}: G={g} P={p} | {q}")
print(f"\nMulti-period: {len(multiperiod)}")
for idx, q, g, p in multiperiod:
    print(f"  #{idx}: G={g} P={p} | {q}")
print(f"\nUnit-rate mismatch: {len(unit_rate)}")
for idx, q, g, p in unit_rate:
    print(f"  #{idx}: G={g} P={p} | {q}")
print(f"\nOther: {len(other)}")
for idx, q, g, p in other:
    print(f"  #{idx}: G={g} P={p} | {q}")
