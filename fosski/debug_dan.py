import sys
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

q = "Dan plants 3 rose bushes. Each rose bush has 25 roses. Each rose has 8 thorns. How many thorns are there total?"
sents = solver._split_sentences(q)
question = [s for s in sents if '?' in s][0]
ctx = [s for s in sents if '?' not in s]
q_facts = solver._extract_question_facts(question)
if q_facts:
    ctx.insert(0, q_facts)

print("Context:", ctx)
print("Question:", question)

# Test each solver
r1 = solver._try_equation_graph(ctx, question, q)
print(f"eq_graph: {r1}")
r2 = solver._try_algebra(ctx, question, q)
print(f"algebra: {r2}")
r3 = solver._try_entity_chain(ctx, question)
print(f"entity: {r3}")
r4 = solver._try_sequential_ops(ctx, question, q)
print(f"seq_ops: {r4}")
r5 = solver._try_fraction_chain(ctx, question, q)
print(f"frac: {r5}")
r6 = solver._try_two_step(ctx, question, q)
print(f"two_step: {r6}")
r7 = solver._try_multi_step(ctx, question, q)
print(f"multi_step: {r7}")
r8 = solver._try_chain_multiply(ctx, question, q)
print(f"chain_mult: {r8}")
