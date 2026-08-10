import json, sys
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()
fails = []
with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold = item['answer'].split('####')[-1].strip().replace(',','')
        result = solver.solve(q)
        pred = result.get('answer','NULL') if result else 'NULL'
        if str(pred) != gold:
            fails.append((q[:150], gold, str(pred)))

for q, gold, pred in fails[:60]:
    print(f'GOLD={gold} PRED={pred} | {q}')
print(f'\nTotal failures: {len(fails)}')
