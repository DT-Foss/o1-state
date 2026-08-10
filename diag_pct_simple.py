"""Find 2-sentence percentage problems that should be easy."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

# Also check what paths the 19 correct pct problems use
correct_paths = []

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    if '%' not in q and 'percent' not in q.lower():
        continue
    sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []

    if str(pred) == gold:
        correct_paths.append((i, len(sents), q[:120], steps))
    elif len(sents) <= 3:
        # Simple percentage fails
        gold_f = float(gold)
        nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]
        pcts = [float(x) for x in re.findall(r'(\d+\.?\d*)\s*%', q)]

        # What's the relationship?
        op = '?'
        for base in nums:
            for pct in pcts:
                if abs(base * pct / 100 - gold_f) < 0.01:
                    op = f'{base}*{pct}%={gold_f}'
                if abs(base * (1 - pct/100) - gold_f) < 0.01:
                    op = f'{base}*(1-{pct}%)={gold_f}'
                if abs(base * (1 + pct/100) - gold_f) < 0.01:
                    op = f'{base}*(1+{pct}%)={gold_f}'
                if abs(base - pct/100 * base - gold_f) < 0.01:
                    op = f'{base}-{pct}%*{base}={gold_f}'

        print(f"#{i} [{len(sents)}s] G={gold} P={pred} OP={op}")
        print(f"  {q[:200]}")
        print()

print(f"\n=== CORRECT PCT ({len(correct_paths)}) ===")
for idx, ns, q, steps in correct_paths:
    print(f"#{idx} [{ns}s] | {' | '.join(steps[:3])}")
    print(f"  {q}")
    print()
