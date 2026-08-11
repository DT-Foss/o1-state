"""Trace which specific two_step pattern fires for each problem."""
import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    lines = f.readlines()

pattern_stats = {}

for i, line in enumerate(lines):
    item = json.loads(line)
    q = item['question']
    gold = item['answer'].split('####')[-1].strip().replace(',','')
    result = solver.solve(q)
    pred = result.get('answer','NULL') if result else 'NULL'
    steps = result.get('steps', []) if result else []
    correct = str(pred) == gold

    # Identify pattern from steps
    pat = 'unknown'
    for s in steps:
        if 'SOP' in s: pat = 'SOP'; break
        elif 'entity_chain' in s or 'starts with' in s.lower(): pat = 'entity_chain'; break
        elif 'take ' in s and 'remaining' in s: pat = 'fraction_chain'; break
        elif 'proportion' in s.lower(): pat = 'proportion'; break
        elif 'average' in s.lower() or 'Mean' in s: pat = 'average'; break
        elif 'chain_multiply' in s: pat = 'chain_multiply'; break
        elif '/ (1 -' in s or '/ (1 +' in s: pat = 'reverse_pct'; break
        elif 'Start:' in s: pat = 'sequential'; break
        elif 'equation_graph' in s: pat = 'equation_graph'; break
        elif 'Algebra' in s: pat = 'algebra'; break
        elif '× (1 -' in s or '× (1 +' in s: pat = 'pct_complement'; break
        elif '% fee' in s or '% fee' in s.lower(): pat = 'pct_fee'; break
        elif '%' in s: pat = 'pct_other'; break
    if pat == 'unknown':
        if not steps:
            pat = 'no_steps'
        elif any('×' in s for s in steps):
            pat = 'two_step_mult'
        elif any('/' in s for s in steps):
            pat = 'two_step_div'
        elif any('+' in s for s in steps):
            pat = 'two_step_add'
        elif any('-' in s for s in steps):
            pat = 'two_step_sub'

    if pat not in pattern_stats:
        pattern_stats[pat] = {'correct': 0, 'total': 0}
    pattern_stats[pat]['total'] += 1
    if correct:
        pattern_stats[pat]['correct'] += 1

print(f"{'Pattern':<20} {'Right':>6} {'Total':>6} {'Accuracy':>9} {'Wrong':>6}")
print("-" * 50)
for pat in sorted(pattern_stats.keys(), key=lambda k: -pattern_stats[k]['total']):
    d = pattern_stats[pat]
    wrong = d['total'] - d['correct']
    pct = 100 * d['correct'] / d['total'] if d['total'] > 0 else 0
    print(f"{pat:<20} {d['correct']:>6} {d['total']:>6} {pct:>8.1f}% {wrong:>6}")
