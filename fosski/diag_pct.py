"""Analyze failing percentage problems to find fixable patterns."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver, extract_numbers

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

pct_fails = []
pct_correct = 0

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    if '%' not in q and 'percent' not in q.lower():
        continue
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    if str(pred) == gold:
        pct_correct += 1
    else:
        pct_fails.append((i, q, gold, str(pred)))

print(f"Percentage problems: {pct_correct} correct, {len(pct_fails)} wrong")

# Sub-categorize by sentence count and pattern
for idx, q, gold, pred in pct_fails[:50]:
    sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
    nums = re.findall(r'\d+\.?\d*', q)
    pcts = re.findall(r'(\d+\.?\d*)\s*%', q)
    gold_f = float(gold)

    # Try to figure out what operation gives gold
    all_nums = [float(n) for n in nums if float(n) > 0]
    op = '?'
    for a in all_nums:
        for b in all_nums:
            if a == b:
                continue
            # pct of base
            if abs(a * b / 100 - gold_f) < 0.01:
                op = f'{a}*{b}/100'
                break
            # base - pct*base/100
            if abs(a - a * b / 100 - gold_f) < 0.01:
                op = f'{a}-{a}*{b}/100'
                break
            if abs(b - b * a / 100 - gold_f) < 0.01:
                op = f'{b}-{b}*{a}/100'
                break
            # base + pct*base/100
            if abs(a + a * b / 100 - gold_f) < 0.01:
                op = f'{a}+{a}*{b}/100'
                break
            if abs(b + b * a / 100 - gold_f) < 0.01:
                op = f'{b}+{b}*{a}/100'
                break
        if op != '?':
            break

    print(f"#{idx} [{len(sents)}s] G={gold} P={pred} %={','.join(pcts)} OP={op}")
    print(f"  {q[:150]}")
    print()
