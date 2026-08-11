import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()
pct_fails = []

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold_raw = item['answer'].split('####')[-1].strip().replace(',','')
        if '%' not in q:
            continue
        result = solver.solve(q)
        pred_raw = result.get('answer','NULL') if result else 'NULL'
        if str(pred_raw) != gold_raw:
            pct_fails.append((q[:150], gold_raw, str(pred_raw)))

# Subcategorize
pct_discount = 0  # "N% off/discount"
pct_increase = 0  # "increased by N%"
pct_of_total = 0  # "N% of total"
pct_tip_tax = 0   # "N% tip/tax"
pct_other = 0

for q, g, p in pct_fails:
    ql = q.lower()
    if 'discount' in ql or 'off' in ql or 'sale' in ql or 'cheaper' in ql:
        pct_discount += 1
    elif 'increas' in ql or 'raise' in ql or 'rose' in ql:
        pct_increase += 1
    elif 'tip' in ql or 'tax' in ql or 'fee' in ql or 'surcharge' in ql or 'interest' in ql:
        pct_tip_tax += 1
    elif re.search(r'\d+%\s+(?:of|are|is|were)', ql):
        pct_of_total += 1
    else:
        pct_other += 1

print(f"Total pct failures: {len(pct_fails)}")
print(f"  Discount/off: {pct_discount}")
print(f"  Increase/raise: {pct_increase}")
print(f"  Tip/tax/fee/interest: {pct_tip_tax}")
print(f"  % of total: {pct_of_total}")
print(f"  Other: {pct_other}")

print("\n=== Increase/raise (first 10) ===")
ct = 0
for q, g, p in pct_fails:
    ql = q.lower()
    if 'increas' in ql or 'raise' in ql or 'rose' in ql:
        ct += 1
        if ct <= 10:
            print(f"  GOLD={g} PRED={p} | {q}")

print("\n=== Tip/tax/fee (first 10) ===")
ct = 0
for q, g, p in pct_fails:
    ql = q.lower()
    if 'tip' in ql or 'tax' in ql or 'fee' in ql or 'surcharge' in ql or 'interest' in ql:
        ct += 1
        if ct <= 10:
            print(f"  GOLD={g} PRED={p} | {q}")

print("\n=== Discount (first 10) ===")
ct = 0
for q, g, p in pct_fails:
    ql = q.lower()
    if 'discount' in ql or 'off' in ql or 'sale' in ql or 'cheaper' in ql:
        ct += 1
        if ct <= 10:
            print(f"  GOLD={g} PRED={p} | {q}")
