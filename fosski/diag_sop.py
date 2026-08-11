"""Diagnose SOP false positives in two_step solver."""
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

    # Check if SOP pattern fired
    is_sop = any('SOP:' in s for s in steps)
    if not is_sop:
        continue

    if str(pred) == gold:
        right.append((i, q, gold, str(pred), steps))
    else:
        wrong.append((i, q, gold, str(pred), steps))

print(f"SOP: {len(right)} right, {len(wrong)} wrong ({100*len(right)/(len(right)+len(wrong)):.1f}%)")
print(f"\n=== WRONG ({len(wrong)}) ===")
for idx, q, gold, pred, steps in wrong[:30]:
    sop_step = [s for s in steps if 'SOP:' in s][0]
    print(f"#{idx}: GOLD={gold} PRED={pred} | {sop_step}")
    print(f"  Q: {q[:150]}")
    print()
