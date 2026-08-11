import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()
close_fails = []
factor2_fails = []

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold_raw = item['answer'].split('####')[-1].strip().replace(',','')
        gold = float(gold_raw) if '.' in gold_raw else int(gold_raw)
        result = solver.solve(q)
        pred_raw = result.get('answer','NULL') if result else 'NULL'
        if str(pred_raw) == gold_raw:
            continue
        try:
            pred = float(pred_raw)
        except:
            continue
        if gold == 0:
            continue
        ratio = pred / gold
        # Within 20%
        if 0.8 <= ratio <= 1.2 and pred != gold:
            close_fails.append((q[:120], gold, pred, ratio))
        # Factor of 2 or 1/2
        if 1.9 <= ratio <= 2.1 or 0.45 <= ratio <= 0.55:
            factor2_fails.append((q[:120], gold, pred, ratio))

print(f"=== Close (within 20%), {len(close_fails)} total ===")
for q, g, p, r in close_fails[:20]:
    print(f"  GOLD={g} PRED={p} ratio={r:.3f} | {q}")

print(f"\n=== Factor of 2 off, {len(factor2_fails)} total ===")
for q, g, p, r in factor2_fails[:20]:
    print(f"  GOLD={g} PRED={p} ratio={r:.3f} | {q}")
