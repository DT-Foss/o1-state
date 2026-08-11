"""Get accuracy by solver path to find where to focus."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

path_stats = {}

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []
    correct = str(pred) == gold

    # Determine path from steps
    path = 'none'
    for s in steps:
        if 'SOP:' in s: path = 'sop'; break
        elif 'proportion:' in s.lower(): path = 'proportion'; break
        elif 'average:' in s.lower(): path = 'average'; break
        elif 'chain_multiply' in s: path = 'chain_multiply'; break
        elif 'equation_graph' in s: path = 'equation_graph'; break
        elif 'algebra' in s.lower() and 'Algebra' in s: path = 'algebra'; break
        elif 'entity_chain' in s: path = 'entity_chain'; break
        elif 'fraction_chain' in s or 'take ' in s.lower(): path = 'fraction_chain'; break
    if path == 'none':
        if any('rate' in s.lower() for s in steps): path = 'rate'
        elif any('Start:' in s for s in steps): path = 'sequential'
        elif any('×' in s for s in steps): path = 'two_step_other'
        elif steps: path = 'other'
        else: path = 'no_steps'

    if path not in path_stats:
        path_stats[path] = {'correct': 0, 'total': 0}
    path_stats[path]['total'] += 1
    if correct:
        path_stats[path]['correct'] += 1

print(f"{'Path':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10} {'Wrong':>8} {'Gain@50%':>10}")
print("-" * 70)
total_correct = 0
total_count = 0
for path in sorted(path_stats.keys(), key=lambda k: -path_stats[k]['total']):
    d = path_stats[path]
    wrong = d['total'] - d['correct']
    pct = 100 * d['correct'] / d['total'] if d['total'] > 0 else 0
    gain = max(0, int(d['total'] * 0.5) - d['correct'])
    total_correct += d['correct']
    total_count += d['total']
    print(f"{path:<20} {d['correct']:>8} {d['total']:>8} {pct:>9.1f}% {wrong:>8} {gain:>10}")
print(f"\n{'TOTAL':<20} {total_correct:>8} {total_count:>8}")
