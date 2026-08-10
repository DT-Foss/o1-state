import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()
fails = []
with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold_raw = item['answer'].split('####')[-1].strip().replace(',','')
        gold = float(gold_raw) if '.' in gold_raw else int(gold_raw)
        result = solver.solve(q)
        pred = result.get('answer','NULL') if result else 'NULL'
        if str(pred) != gold_raw:
            fails.append((q, gold, pred))

# Find questions where answer is a simple product of two numbers in the question
simple_product = 0
simple_sum = 0
null_results = 0
close_off = 0  # within 2x of correct

for q, gold, pred in fails:
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]
    if pred == 'NULL':
        null_results += 1

    # Check if gold is product of any two nums
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i != j and abs(nums[i] * nums[j] - gold) < 0.01:
                simple_product += 1
                break
        else:
            continue
        break

print(f"Total failures: {len(fails)}")
print(f"NULL results: {null_results}")
print(f"Simple product of 2 nums: {simple_product}")

# Show problems where we return NULL
print("\n=== NULL RESULTS (first 20) ===")
ct = 0
for q, gold, pred in fails:
    if str(pred) == 'NULL':
        ct += 1
        if ct <= 20:
            print(f"GOLD={gold} | {q[:150]}")
print(f"\n=== Close-off (pred within 10% of gold, first 20) ===")
ct = 0
for q, gold, pred in fails:
    try:
        p = float(pred)
        if gold != 0 and 0.9 <= p/gold <= 1.1 and p != gold:
            ct += 1
            if ct <= 20:
                print(f"GOLD={gold} PRED={pred} | {q[:150]}")
    except:
        pass

# Show the most common wrong answer patterns
print("\n=== Short problems (1-2 sentences) first 20 ===")
ct = 0
for q, gold, pred in fails:
    sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
    if len(sents) <= 2:
        ct += 1
        if ct <= 20:
            print(f"GOLD={gold} PRED={pred} | {q[:150]}")
print(f"Total short failures: {ct}")
