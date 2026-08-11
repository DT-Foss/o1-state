"""Find problems where we return NULL — these need any solver to fire."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

null_results = []

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    if str(pred) == 'NULL':
        sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
        null_results.append((i, q, gold, len(sents)))

print(f"NULL results: {len(null_results)}")
# Show by sentence count
by_len = {}
for idx, q, gold, ns in null_results:
    if ns not in by_len:
        by_len[ns] = []
    by_len[ns].append((idx, q, gold))

for ns in sorted(by_len.keys()):
    items = by_len[ns]
    print(f"\n{ns} sentences: {len(items)} NULL")
    for idx, q, gold in items[:5]:
        print(f"  #{idx} G={gold} | {q[:120]}")
