import sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

# Bobby
q = "Brian's friend Bobby has 5 fewer than 3 times as many video games as Brian does.  If Brian has 20 video games but lost 5 right before the comparison was made, how many does Bobby have?"
sents = solver._split_sentences(q)
print("Bobby sentences:")
for i, s in enumerate(sents):
    print(f"  {i}: {s}")

# Check what solve returns
r = solver.solve(q)
print(f"\nResult: {r}")

# Check if MS12 pattern matches
all_lower = ' '.join(sents).lower()
m = re.search(r'(\d+)\s+(?:fewer|less|more)\s+than\s+(\d+)\s+times', all_lower)
print(f"MS12 match: {m}")
if m:
    print(f"  offset={m.group(1)}, mult={m.group(2)}")
