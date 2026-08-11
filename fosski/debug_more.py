import sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

# Cobra/mamba — needs entity × count sum
q1 = "A cobra, which has 70 spots, has twice as many spots as a mamba. If there are 40 cobras and 60 mambas in a snake park, what is half the number of spots in the snake park?"
sents = solver._split_sentences(q1)
print("Cobra sentences:")
for i, s in enumerate(sents):
    print(f"  {i}: {s}")

# Jill — multi-rate sum
q2 = "Jill gets paid $20 per hour to teach and $30 to be a cheerleading coach. If she works 50 weeks a year, 35 hours a week as a teacher and 15 hours a week as a cheerleading coach, how much does she make a year?"
sents2 = solver._split_sentences(q2)
print("\nJill sentences:")
for i, s in enumerate(sents2):
    print(f"  {i}: {s}")

# Test what _try_multi_step returns for Jill
question2 = [s for s in sents2 if '?' in s][0]
ctx2 = [s for s in sents2 if '?' not in s]
q_facts = solver._extract_question_facts(question2)
if q_facts:
    ctx2.insert(0, q_facts)
print(f"\nJill context: {ctx2}")
print(f"Jill question: {question2}")
r2 = solver._try_multi_step(ctx2, question2, q2)
print(f"Multi-step: {r2}")

# Claire — eggs to dozens
q3 = "Claire makes a 3 egg omelet every morning for breakfast.  How many dozens of eggs will she eat in 4 weeks?"
r3 = solver.solve(q3)
print(f"\nClaire: {r3}")
# 3 eggs/day × 28 days = 84 eggs / 12 = 7 dozens

# Candidate
q4 = "Two candidates are running for class representative at Sarai's school. If the winner got 3/4 of the votes and the total number of students who voted is 80, calculate the number of votes the loser got."
r4 = solver.solve(q4)
print(f"Candidate: {r4}")
# 80 × (1 - 3/4) = 80 × 1/4 = 20
