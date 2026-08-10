import sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

q1 = "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. How many students enrolled in hip-hop dance?"
sents = solver._split_sentences(q1)
print("Dance sentences:")
for i, s in enumerate(sents):
    print(f"  {i}: {s}")

q2 = "Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the rest had the letter D. How many of the cards had the letter D on them?"
sents2 = solver._split_sentences(q2)
print("\nCarly sentences:")
for i, s in enumerate(sents2):
    print(f"  {i}: {s}")
