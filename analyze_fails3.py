import json, sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver, extract_numbers

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
            fails.append((q, gold, pred, result))

# Find cases where the answer is achievable by a simple sequence of operations on numbers in the question
# Specifically: look at 2-sentence problems where answer = n1 * n2 / n3
print("=== 2-sentence problems where answer = simple ops of 2-3 numbers ===")
ct = 0
for q, gold, pred, _ in fails:
    sents = [s.strip() for s in re.split(r'[.!?]+', q) if s.strip()]
    if len(sents) > 3:
        continue
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]
    if len(nums) < 2 or len(nums) > 5:
        continue
    # Check if gold = any simple operation on 2-3 of these nums
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            # a * b
            if abs(nums[i] * nums[j] - gold) < 0.01:
                ct += 1
                if ct <= 15:
                    print(f"  {nums[i]}*{nums[j]}={gold} PRED={pred} | {q[:120]}")
                break
            # a * b / c
            for k in range(len(nums)):
                if k == i or k == j:
                    continue
                if nums[k] != 0 and abs(nums[i] * nums[j] / nums[k] - gold) < 0.01:
                    ct += 1
                    if ct <= 15:
                        print(f"  {nums[i]}*{nums[j]}/{nums[k]}={gold} PRED={pred} | {q[:120]}")
                    break
        else:
            continue
        break

print(f"\nTotal simple ops failures (short problems): {ct}")

# Find percentage problems
print("\n=== Percentage problems ===")
ct2 = 0
for q, gold, pred, _ in fails:
    if '%' in q:
        ct2 += 1
        if ct2 <= 15:
            print(f"  GOLD={gold} PRED={pred} | {q[:120]}")
print(f"Total: {ct2}")

# Find problems where gold = product of all numbers
print("\n=== Product of all numbers ===")
ct3 = 0
for q, gold, pred, _ in fails:
    nums = [float(x) for x in re.findall(r'\d+\.?\d*', q) if float(x) > 0]
    if len(nums) >= 2 and len(nums) <= 4:
        prod = 1
        for n in nums:
            prod *= n
        if abs(prod - gold) < 0.01:
            ct3 += 1
            if ct3 <= 10:
                print(f"  {'*'.join(str(int(n)) for n in nums)}={gold} PRED={pred} | {q[:120]}")
print(f"Total: {ct3}")

# Find "dozen" problems (÷12)
print("\n=== Dozen/unit conversion ===")
ct4 = 0
for q, gold, pred, _ in fails:
    ql = q.lower()
    if 'dozen' in ql or 'feet' in ql or 'inch' in ql or 'pound' in ql:
        ct4 += 1
        if ct4 <= 10:
            print(f"  GOLD={gold} PRED={pred} | {q[:120]}")
print(f"Total: {ct4}")

print(f"\nOverall failures: {len(fails)}")
