import json, sys
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

# Find problems where chain_multiply returns wrong answer but would be correct without it
wrong_from_chain = []
right_from_chain = []

with open('data/external_benchmarks/gsm8k_test.jsonl') as f:
    for line in f:
        item = json.loads(line)
        q = item['question']
        gold = item['answer'].split('####')[-1].strip().replace(',','')

        sents = solver._split_sentences(q)
        question = ''
        ctx = []
        for s in sents:
            if '?' in s:
                question = s
            else:
                ctx.append(s)
        if not question and ctx:
            import re
            last = ctx[-1]
            if re.search(r'\b(?:calculate|find|determine)\b', last, re.I):
                question = ctx.pop()
        q_facts = solver._extract_question_facts(question)
        if q_facts:
            ctx.insert(0, q_facts)
        q_nums = solver._extract_question_numbers(question)
        if q_nums:
            ctx.append(q_nums)

        chain_r = solver._try_chain_multiply(ctx, question, q)
        if chain_r is not None:
            if str(chain_r['answer']) == gold:
                right_from_chain.append((q[:100], gold, chain_r['answer']))
            else:
                wrong_from_chain.append((q[:100], gold, chain_r['answer']))

print(f"Chain multiply: {len(right_from_chain)} correct, {len(wrong_from_chain)} wrong")
print("\n=== WRONG from chain multiply (first 20) ===")
for q, g, p in wrong_from_chain[:20]:
    print(f"  GOLD={g} PRED={p} | {q}")
print("\n=== RIGHT from chain multiply (first 10) ===")
for q, g, p in right_from_chain[:10]:
    print(f"  GOLD={g} PRED={p} | {q}")
