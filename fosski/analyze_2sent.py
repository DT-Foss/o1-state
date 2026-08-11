import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver, extract_numbers

solver = WordProblemSolver()
fails_2 = []

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold = item['answer'].split('####')[-1].strip().replace(',','')
        sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
        if len(sents) != 2:
            continue
        result = solver.solve(q)
        pred = result.get('answer','NULL') if result else 'NULL'
        if str(pred) != gold:
            fails_2.append((q, gold, str(pred)))

print(f"2-sentence failures: {len(fails_2)}")
# Try to categorize by operation needed
for q, gold, pred in fails_2[:40]:
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]
    gold_f = float(gold)
    # Check if answer is achievable
    ops = []
    for i, a in enumerate(nums):
        for j, b in enumerate(nums):
            if i == j:
                continue
            if abs(a * b - gold_f) < 0.01:
                ops.append(f'{a}×{b}')
            if abs(a + b - gold_f) < 0.01:
                ops.append(f'{a}+{b}')
            if abs(a - b - gold_f) < 0.01:
                ops.append(f'{a}-{b}')
            if b > 0 and abs(a / b - gold_f) < 0.01:
                ops.append(f'{a}/{b}')
    op_str = ', '.join(ops[:3]) if ops else 'complex'
    print(f"GOLD={gold} PRED={pred} OP=[{op_str}] | {q[:100]}")
