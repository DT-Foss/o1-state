"""Analyze sequential path failures to find fixable patterns."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

seq_fails = []
seq_correct = []

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []

    if not any('Start:' in s for s in steps):
        continue
    if any('SOP:' in s or 'take ' in s.lower() or 'entity_chain' in s for s in steps):
        continue

    if str(pred) == gold:
        seq_correct.append((i, q[:100], gold, str(pred), steps))
    else:
        seq_fails.append((i, q[:100], gold, str(pred), steps))

print(f"Sequential: {len(seq_correct)} correct, {len(seq_fails)} wrong")

# Find common step patterns in wrong answers
print(f"\n=== WRONG (first 30) ===")
for idx, q, gold, pred, steps in seq_fails[:30]:
    step_summary = ' → '.join(steps[:5])
    print(f"#{idx} G={gold} P={pred} | {step_summary}")
    print(f"  {q}")
    print()
