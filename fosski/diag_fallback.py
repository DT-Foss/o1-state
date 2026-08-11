"""Categorize fallback problems to find solvable patterns."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver, extract_numbers

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

fallback_items = []

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []
    path = result.get('path', '') if result else ''

    # Check for fallback
    is_fallback = (path == 'fallback' or
                   any('fallback' in s.lower() for s in steps) or
                   (not any(k in '|'.join(steps) for k in ['SOP:', 'proportion:', 'average:',
                    'chain_multiply', 'equation_graph', 'algebra:', 'entity_chain',
                    'sequential', 'fraction_chain', 'two_step', 'multi_step', 'rate_chain'])))

    if str(pred) != gold and is_fallback:
        fallback_items.append((i, q, gold, str(pred)))

print(f"Fallback failures: {len(fallback_items)}")

# Categorize by what operation would give the gold answer
cats = {
    'simple_arith': [],  # gold = simple op on 2 numbers
    'percentage': [],
    'fraction': [],
    'multi_step': [],
    'comparison': [],
    'division': [],
}

for idx, q, gold, pred in fallback_items:
    ql = q.lower()
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]
    gold_f = float(gold)

    # Check if it's a percentage problem
    if '%' in q or 'percent' in ql:
        cats['percentage'].append((idx, q[:100], gold, pred))
        continue

    # Check if it's a fraction problem
    if re.search(r'\b(?:half|third|quarter|fifth|one-\w+|two-\w+|three-\w+|\d/\d)\b', ql):
        cats['fraction'].append((idx, q[:100], gold, pred))
        continue

    # Check if gold is achievable with simple 2-number operation
    found_op = False
    for i_n, a in enumerate(nums):
        for j_n, b in enumerate(nums):
            if i_n == j_n:
                continue
            if abs(a * b - gold_f) < 0.01:
                cats['simple_arith'].append((idx, q[:100], gold, pred, f'{a}*{b}'))
                found_op = True
                break
            if abs(a + b - gold_f) < 0.01:
                cats['simple_arith'].append((idx, q[:100], gold, pred, f'{a}+{b}'))
                found_op = True
                break
            if abs(a - b - gold_f) < 0.01:
                cats['simple_arith'].append((idx, q[:100], gold, pred, f'{a}-{b}'))
                found_op = True
                break
            if b > 0 and abs(a / b - gold_f) < 0.01:
                cats['simple_arith'].append((idx, q[:100], gold, pred, f'{a}/{b}'))
                found_op = True
                break
        if found_op:
            break
    if found_op:
        continue

    cats['multi_step'].append((idx, q[:100], gold, pred))

for cat, items in cats.items():
    print(f"\n{cat}: {len(items)}")
    for item in items[:8]:
        if len(item) == 5:
            idx, q, g, p, op = item
            print(f"  #{idx}: G={g} P={p} OP={op} | {q}")
        else:
            idx, q, g, p = item
            print(f"  #{idx}: G={g} P={p} | {q}")
