import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()
by_length = {}

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold = item['answer'].split('####')[-1].strip().replace(',','')
        sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
        n = len(sents)
        result = solver.solve(q)
        pred = result.get('answer','NULL') if result else 'NULL'
        correct = str(pred) == gold
        if n not in by_length:
            by_length[n] = {'correct': 0, 'total': 0}
        by_length[n]['total'] += 1
        if correct:
            by_length[n]['correct'] += 1

print("Sentences | Correct/Total | Accuracy")
for n in sorted(by_length.keys()):
    d = by_length[n]
    pct = 100 * d['correct'] / d['total'] if d['total'] > 0 else 0
    print(f"    {n:3d}    | {d['correct']:4d}/{d['total']:4d}   | {pct:.1f}%")
