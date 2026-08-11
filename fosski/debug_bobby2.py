import sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()
q = "Brian's friend Bobby has 5 fewer than 3 times as many video games as Brian does.  If Brian has 20 video games but lost 5 right before the comparison was made, how many does Bobby have?"

sents = solver._split_sentences(q)
question = ''
context_sents = []
for s in sents:
    if '?' in s:
        question = s
    else:
        context_sents.append(s)
q_facts = solver._extract_question_facts(question)
if q_facts:
    context_sents.insert(0, q_facts)

print("Context:", context_sents)
print("Question:", question)

# Test equation graph
r = solver._try_equation_graph(context_sents, question, q)
print(f"\nEquation graph: {r}")

# Test entity chain
r2 = solver._try_entity_chain(context_sents, question)
print(f"Entity chain: {r2}")

# Test multi-step
r3 = solver._try_multi_step(context_sents, question, q)
print(f"Multi-step: {r3}")
