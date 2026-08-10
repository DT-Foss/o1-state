import sys, re
sys.path.insert(0, '.')
from core.word_problem import WordProblemSolver

solver = WordProblemSolver()

# Dance class
q1 = "In a dance class of 20 students, 20% enrolled in contemporary dance, 25% of the remaining enrolled in jazz dance, and the rest enrolled in hip-hop dance. How many students enrolled in hip-hop dance?"
r1 = solver.solve(q1)
print(f"Dance: {r1}")

# Carly
q2 = "Carly had 80 cards, 2/5 of the cards had the letter A on them, 1/2 of the remaining had the letter B, 5/8 of the rest had the letter C on them, and the rest had the letter D. How many of the cards had the letter D on them?"
r2 = solver.solve(q2)
print(f"Carly: {r2}")

# Grade 5
q3 = "Out of the 200 Grade 5 students, 2/5 are boys and 2/3 of the girls are in the girl scout. How many girls are not in the girl scout?"
r3 = solver.solve(q3)
print(f"Grade5: {r3}")
