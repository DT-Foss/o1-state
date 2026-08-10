"""Find problems with no steps — degenerate solver path."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []
    correct = str(pred) == gold

    if not steps and pred != 'NULL':
        print(f"#{i} G={gold} P={pred} {'OK' if correct else 'WRONG'}")
        print(f"  Q: {q[:150]}")
        print(f"  result: {result}")
        print()
