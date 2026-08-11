"""Find the EASIEST problems we get wrong — 2 sentences, simple operation."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

easy_fails = []

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
    if len(sents) > 3:
        continue
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    if str(pred) == gold:
        continue

    gold_f = float(gold)
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]

    # Check simple operations
    for a in nums:
        for b in nums:
            if a == b:
                continue
            for op_name, result_val in [
                (f'{a}*{b}', a*b),
                (f'{a}+{b}', a+b),
                (f'{a}-{b}', a-b),
                (f'{a}/{b}', a/b if b else 0),
            ]:
                if abs(result_val - gold_f) < 0.01 and result_val > 0:
                    easy_fails.append((i, q, gold, str(pred), op_name, len(sents)))
                    break
            else:
                continue
            break
        else:
            continue
        break

print(f"Easy fails (<=3 sentences, simple 2-num operation): {len(easy_fails)}")
# Group by operation type
by_op = {}
for idx, q, gold, pred, op, ns in easy_fails:
    op_type = op.split('(')[0] if '(' in op else (
        '*' if '*' in op else '+' if '+' in op else '-' if '-' in op else '/')
    if op_type not in by_op:
        by_op[op_type] = []
    by_op[op_type].append((idx, q, gold, pred, op, ns))

for op_type in sorted(by_op.keys()):
    items = by_op[op_type]
    print(f"\n=== {op_type} ({len(items)}) ===")
    for idx, q, gold, pred, op, ns in items[:10]:
        print(f"  #{idx} [{ns}s] G={gold} P={pred} OP={op} | {q[:120]}")
