import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

# Short (2-3 sentence) percentage problems that we're failing
with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold = item['answer'].split('####')[-1].strip().replace(',','')
        if '%' not in q:
            continue
        sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
        if len(sents) > 3:
            continue
        result = solver.solve(q)
        pred = result.get('answer','NULL') if result else 'NULL'
        if str(pred) == gold:
            continue
        print(f"GOLD={gold} PRED={pred} | {q[:150]}")
