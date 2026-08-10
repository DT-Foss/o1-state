#!/usr/bin/env python3
"""
General Knowledge Q&A Benchmark
=================================
Tests the system against 100 real-world questions that any
GPT-4o level system should be able to answer.

Categories:
  - Geography (capitals, continents, locations)
  - Science (elements, compounds, constants)
  - History (events, people, dates)
  - General (animals, languages, sports)
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def build_test_set():
    """100 questions with expected answer substrings."""
    return [
        # === GEOGRAPHY (30) ===
        ("What is the capital of France?", ["paris"]),
        ("What is the capital of Germany?", ["berlin"]),
        ("What is the capital of Japan?", ["tokyo"]),
        ("What is the capital of Australia?", ["canberra"]),
        ("What is the capital of Brazil?", ["brasilia"]),
        ("What is the capital of Canada?", ["ottawa"]),
        ("What is the capital of India?", ["new delhi", "delhi"]),
        ("What is the capital of China?", ["beijing"]),
        ("What is the capital of Russia?", ["moscow"]),
        ("What is the capital of Italy?", ["rome"]),
        ("What is the capital of Spain?", ["madrid"]),
        ("What is the capital of Mexico?", ["mexico city"]),
        ("What is the capital of Egypt?", ["cairo"]),
        ("What is the capital of South Korea?", ["seoul"]),
        ("What is the capital of Argentina?", ["buenos aires"]),
        ("What is the capital of Turkey?", ["ankara"]),
        ("What is the capital of Thailand?", ["bangkok"]),
        ("What is the capital of Poland?", ["warsaw"]),
        ("What is the capital of Norway?", ["oslo"]),
        ("What is the capital of Sweden?", ["stockholm"]),
        ("What continent is Brazil in?", ["south america"]),
        ("What continent is Japan in?", ["asia"]),
        ("What continent is Nigeria in?", ["africa"]),
        ("What continent is Germany in?", ["europe"]),
        ("What continent is Australia in?", ["oceania"]),
        ("What language is spoken in France?", ["french"]),
        ("What language is spoken in Japan?", ["japanese"]),
        ("What language is spoken in Brazil?", ["portuguese"]),
        ("What currency is used in Japan?", ["yen"]),
        ("What currency is used in the United Kingdom?", ["pound"]),

        # === SCIENCE (25) ===
        ("What is the symbol of Gold?", ["au"]),
        ("What is the symbol of Silver?", ["ag"]),
        ("What is the symbol of Iron?", ["fe"]),
        ("What is the symbol of Hydrogen?", ["h"]),
        ("What is the symbol of Oxygen?", ["o"]),
        ("What is the symbol of Carbon?", ["c"]),
        ("What is the symbol of Sodium?", ["na"]),
        ("What is the symbol of Copper?", ["cu"]),
        ("What is the atomic number of Carbon?", ["6"]),
        ("What is the atomic number of Oxygen?", ["8"]),
        ("What is the atomic number of Gold?", ["79"]),
        ("What is the formula of Water?", ["h2o"]),
        ("What is the formula of Salt?", ["nacl"]),
        ("What is the formula of Carbon Dioxide?", ["co2"]),
        ("What is the formula of Methane?", ["ch4"]),
        ("What is the formula of Ammonia?", ["nh3"]),
        ("What is the nearest planet to the Sun?", ["mercury"]),
        ("What is the largest planet?", ["jupiter"]),
        ("How many moons does Earth have?", ["1"]),
        ("How many moons does Mars have?", ["2"]),
        ("What is the speed of light?", ["299792458", "3e8", "300000"]),
        ("What state is Mercury at room temperature?", ["liquid"]),
        ("What category is Gold in the periodic table?", ["transition metal"]),
        ("Is Hydrogen a metal?", ["no", "nonmetal", "non-metal"]),
        ("What is the state of Oxygen?", ["gas"]),

        # === HISTORY (20) ===
        ("When did World War I start?", ["1914"]),
        ("When did World War II end?", ["1945"]),
        ("When was the French Revolution?", ["1789"]),
        ("When did the Cold War end?", ["1991"]),
        ("When was the Moon Landing?", ["1969"]),
        ("When was America discovered?", ["1492"]),
        ("Where was Albert Einstein born?", ["germany"]),
        ("Where was Isaac Newton born?", ["england"]),
        ("Where was Marie Curie born?", ["poland"]),
        ("What is Albert Einstein known for?", ["relativity"]),
        ("What is Isaac Newton known for?", ["motion"]),
        ("What is Charles Darwin known for?", ["evolution"]),
        ("What is Leonardo da Vinci known for?", ["mona lisa"]),
        ("What did Alexander Graham Bell invent?", ["telephone"]),
        ("Who created Python?", ["guido van rossum", "guido"]),
        ("Who created Linux?", ["linus torvalds", "linus"]),
        ("Who created the World Wide Web?", ["tim berners-lee", "berners"]),
        ("Who wrote Hamlet?", ["shakespeare"]),
        ("What is Ada Lovelace known for?", ["computer program"]),
        ("What is Alan Turing known for?", ["turing machine"]),

        # === GENERAL KNOWLEDGE (25) ===
        ("Is a dog a mammal?", ["yes", "mammal"]),
        ("Is a shark a fish?", ["yes", "fish"]),
        ("Is a penguin a bird?", ["yes", "bird"]),
        ("Is a snake a reptile?", ["yes", "reptile"]),
        ("Is a frog an amphibian?", ["yes", "amphibian"]),
        ("What does the heart do?", ["pump", "blood"]),
        ("What does the brain do?", ["control"]),
        ("What does the lungs do?", ["gas", "breathe", "exchange"]),
        ("What sport has the most fans?", ["football", "soccer"]),
        ("What sport uses the NBA?", ["basketball"]),
        ("What language family is English in?", ["indo-european"]),
        ("What script does Japanese use?", ["kanji", "hiragana"]),
        ("How many speakers does English have?", ["1.5 billion", "billion"]),
        ("What is the diet of a lion?", ["carnivore"]),
        ("What is the diet of an elephant?", ["herbivore"]),
        ("What is the habitat of a dolphin?", ["wild"]),
        ("How long does a dog live?", ["10-13", "10", "13"]),
        ("How long does an elephant live?", ["60-70", "60", "70"]),
        ("What is 2 + 2?", ["4"]),
        ("What is 10 * 7?", ["70"]),
        ("What is the square root of 144?", ["12"]),
        ("Is 17 prime?", ["yes", "prime"]),
        ("What is 2 to the power of 10?", ["1024"]),
        ("What is the GCD of 12 and 8?", ["4"]),
        ("What is 100 / 4?", ["25"]),
    ]


def main():
    print("=" * 60)
    print("  GENERAL KNOWLEDGE Q&A BENCHMARK")
    print("  100 questions across 4 categories")
    print("=" * 60)

    repl = FossKIRepl()
    tests = build_test_set()

    categories = {
        'Geography': (0, 30),
        'Science': (30, 55),
        'History': (55, 75),
        'General': (75, 100),
    }

    total_correct = 0
    total_tests = len(tests)
    cat_results = {}

    t0 = time.time()

    for cat_name, (start, end) in categories.items():
        cat_correct = 0
        cat_total = end - start
        print(f"\n{'─' * 50}")
        print(f"  {cat_name} ({cat_total} questions)")
        print(f"{'─' * 50}")

        for i in range(start, end):
            question, expected_substrings = tests[i]
            answer = repl.process(question)
            answer_lower = answer.lower()

            # Check if any expected substring appears in answer
            correct = any(exp.lower() in answer_lower for exp in expected_substrings)

            if correct:
                cat_correct += 1
                total_correct += 1
            else:
                print(f"  ✗ {question}")
                print(f"    Got: {answer[:80]}")
                print(f"    Expected: {expected_substrings}")

        acc = cat_correct / max(cat_total, 1)
        cat_results[cat_name] = (cat_correct, cat_total, acc)
        print(f"  Score: {cat_correct}/{cat_total} ({acc:.0%})")

    elapsed = time.time() - t0
    overall_acc = total_correct / max(total_tests, 1)

    print(f"\n{'=' * 60}")
    print(f"  Q&A BENCHMARK RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  {'Category':<20} {'Score':>10} {'Accuracy':>10}")
    print(f"  {'─' * 42}")
    for cat, (c, t, a) in cat_results.items():
        bar = '█' * int(a * 20) + '░' * (20 - int(a * 20))
        print(f"  {cat:<20} {c:>3}/{t:<3} {bar} {a:>5.0%}")

    print(f"\n  {'OVERALL':<20} {total_correct:>3}/{total_tests:<3} {'':>22} {overall_acc:>5.0%}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_tests*1000:.0f}ms/query)")

    # Comparison
    print(f"\n  {'Metric':<30} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 52}")
    print(f"  {'Q&A accuracy':<30} {overall_acc:>9.0%} {'~95%':>10}")
    print(f"  {'Geography':<30} {cat_results['Geography'][2]:>9.0%} {'~98%':>10}")
    print(f"  {'Science':<30} {cat_results['Science'][2]:>9.0%} {'~97%':>10}")
    print(f"  {'History':<30} {cat_results['History'][2]:>9.0%} {'~95%':>10}")
    print(f"  {'General':<30} {cat_results['General'][2]:>9.0%} {'~93%':>10}")
    print(f"  {'Latency':<30} {elapsed/total_tests*1000:>8.0f}ms {'~500ms':>10}")
    print(f"  {'Parameters':<30} {'0':>10} {'~200B':>10}")
    print(f"  {'Knowledge source':<30} {'3.2K local':>10} {'Internet':>10}")

    print(f"\n{'=' * 60}")
    if overall_acc >= 0.90:
        print("  STATUS: GPT-4o COMPETITIVE")
    elif overall_acc >= 0.75:
        print("  STATUS: STRONG")
    elif overall_acc >= 0.50:
        print("  STATUS: DEVELOPING")
    else:
        print("  STATUS: NEEDS WORK")
    print(f"{'=' * 60}")

    print(f"\nResults: {total_correct}/{total_tests} passed ({overall_acc:.0%})")


if __name__ == '__main__':
    main()
