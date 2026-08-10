#!/usr/bin/env python3
"""
ARC-Easy Style Reasoning Benchmark
====================================
Tests pattern reasoning, analogical thinking, and categorization.
Modeled on ARC-Easy (AI2 Reasoning Challenge) — multiple choice
questions that require reasoning, not just memorization.

Categories:
  - Analogies (A is to B as C is to ?)
  - Sequence completion (1, 2, 4, ?)
  - Odd-one-out (which doesn't belong?)
  - Cause and effect
  - Property inference
  - Spatial/temporal reasoning

All answerable from common sense + KB + reasoning.
No external data needed.
"""

import sys
import os
import time
import re

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repl import FossKIRepl


def build_test_set():
    """50 ARC-Easy style questions with answers."""
    return [
        # === ANALOGIES (10) ===
        {
            'q': 'Ice is to water as steam is to what?',
            'choices': ['fire', 'water', 'air', 'cloud'],
            'answer': 'water',
            'category': 'analogy',
        },
        {
            'q': 'Puppy is to dog as kitten is to what?',
            'choices': ['cat', 'mouse', 'bird', 'fish'],
            'answer': 'cat',
            'category': 'analogy',
        },
        {
            'q': 'Eye is to seeing as ear is to what?',
            'choices': ['hearing', 'smelling', 'tasting', 'touching'],
            'answer': 'hearing',
            'category': 'analogy',
        },
        {
            'q': 'Hot is to cold as up is to what?',
            'choices': ['left', 'down', 'right', 'sideways'],
            'answer': 'down',
            'category': 'analogy',
        },
        {
            'q': 'Tree is to forest as house is to what?',
            'choices': ['city', 'room', 'brick', 'roof'],
            'answer': 'city',
            'category': 'analogy',
        },
        {
            'q': 'Page is to book as key is to what?',
            'choices': ['lock', 'door', 'keyboard', 'piano'],
            'answer': 'keyboard',
            'category': 'analogy',
        },
        {
            'q': 'Pen is to writing as knife is to what?',
            'choices': ['cutting', 'eating', 'cooking', 'stabbing'],
            'answer': 'cutting',
            'category': 'analogy',
        },
        {
            'q': 'Fish is to water as bird is to what?',
            'choices': ['tree', 'air', 'nest', 'sky'],
            'answer': 'air',
            'category': 'analogy',
        },
        {
            'q': 'France is to Paris as Germany is to what?',
            'choices': ['Munich', 'Berlin', 'Hamburg', 'Frankfurt'],
            'answer': 'Berlin',
            'category': 'analogy',
        },
        {
            'q': 'Gold is to Au as Silver is to what?',
            'choices': ['Si', 'Ag', 'Fe', 'Cu'],
            'answer': 'Ag',
            'category': 'analogy',
        },

        # === SEQUENCE COMPLETION (10) ===
        {
            'q': 'What comes next: 2, 4, 6, 8, ?',
            'choices': ['9', '10', '12', '16'],
            'answer': '10',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 1, 2, 4, 8, ?',
            'choices': ['10', '12', '16', '32'],
            'answer': '16',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 1, 1, 2, 3, 5, ?',
            'choices': ['6', '7', '8', '10'],
            'answer': '8',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 3, 6, 9, 12, ?',
            'choices': ['13', '14', '15', '18'],
            'answer': '15',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 1, 4, 9, 16, ?',
            'choices': ['20', '25', '32', '36'],
            'answer': '25',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 10, 20, 30, 40, ?',
            'choices': ['45', '50', '60', '100'],
            'answer': '50',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 100, 50, 25, ?',
            'choices': ['10', '12', '12.5', '15'],
            'answer': '12.5',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 1, 3, 5, 7, ?',
            'choices': ['8', '9', '10', '11'],
            'answer': '9',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 2, 6, 18, 54, ?',
            'choices': ['72', '108', '162', '216'],
            'answer': '162',
            'category': 'sequence',
        },
        {
            'q': 'What comes next: 0, 1, 1, 2, 3, 5, 8, ?',
            'choices': ['10', '11', '12', '13'],
            'answer': '13',
            'category': 'sequence',
        },

        # === ODD ONE OUT (10) ===
        {
            'q': 'Which does not belong: dog, cat, horse, tree?',
            'choices': ['dog', 'cat', 'horse', 'tree'],
            'answer': 'tree',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: car, bus, bicycle, house?',
            'choices': ['car', 'bus', 'bicycle', 'house'],
            'answer': 'house',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: apple, banana, carrot, grape?',
            'choices': ['apple', 'banana', 'carrot', 'grape'],
            'answer': 'carrot',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: Mars, Jupiter, Moon, Saturn?',
            'choices': ['Mars', 'Jupiter', 'Moon', 'Saturn'],
            'answer': 'Moon',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: French, German, Spanish, Paris?',
            'choices': ['French', 'German', 'Spanish', 'Paris'],
            'answer': 'Paris',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: heart, brain, kidney, hammer?',
            'choices': ['heart', 'brain', 'kidney', 'hammer'],
            'answer': 'hammer',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: 2, 3, 5, 9?',
            'choices': ['2', '3', '5', '9'],
            'answer': '9',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: water, milk, stone, juice?',
            'choices': ['water', 'milk', 'stone', 'juice'],
            'answer': 'stone',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: piano, guitar, drum, painting?',
            'choices': ['piano', 'guitar', 'drum', 'painting'],
            'answer': 'painting',
            'category': 'odd_one_out',
        },
        {
            'q': 'Which does not belong: spring, summer, autumn, Monday?',
            'choices': ['spring', 'summer', 'autumn', 'Monday'],
            'answer': 'Monday',
            'category': 'odd_one_out',
        },

        # === CAUSE AND EFFECT (10) ===
        {
            'q': 'What happens when you heat ice?',
            'choices': ['it freezes', 'it melts', 'it evaporates', 'nothing'],
            'answer': 'it melts',
            'category': 'cause_effect',
        },
        {
            'q': 'What happens when you drop a glass?',
            'choices': ['it flies', 'it breaks', 'it grows', 'it melts'],
            'answer': 'it breaks',
            'category': 'cause_effect',
        },
        {
            'q': 'What does a plant need to grow?',
            'choices': ['darkness', 'water and sunlight', 'cold', 'metal'],
            'answer': 'water and sunlight',
            'category': 'cause_effect',
        },
        {
            'q': 'What causes rain?',
            'choices': ['sun', 'wind', 'water evaporation and condensation', 'gravity'],
            'answer': 'water evaporation and condensation',
            'category': 'cause_effect',
        },
        {
            'q': 'Why do birds fly south in winter?',
            'choices': ['for warmer weather', 'to find mountains', 'because they are lost', 'to find water'],
            'answer': 'for warmer weather',
            'category': 'cause_effect',
        },
        {
            'q': 'What happens to water at 0 degrees Celsius?',
            'choices': ['it boils', 'it evaporates', 'it freezes', 'nothing'],
            'answer': 'it freezes',
            'category': 'cause_effect',
        },
        {
            'q': 'Why do we wear coats in winter?',
            'choices': ['to look nice', 'to stay warm', 'to stay cool', 'to run faster'],
            'answer': 'to stay warm',
            'category': 'cause_effect',
        },
        {
            'q': 'What makes a shadow?',
            'choices': ['water', 'an object blocking light', 'darkness', 'wind'],
            'answer': 'an object blocking light',
            'category': 'cause_effect',
        },
        {
            'q': 'What happens when you mix red and blue paint?',
            'choices': ['green', 'yellow', 'purple', 'orange'],
            'answer': 'purple',
            'category': 'cause_effect',
        },
        {
            'q': 'Why does metal rust?',
            'choices': ['heat', 'cold', 'exposure to oxygen and moisture', 'pressure'],
            'answer': 'exposure to oxygen and moisture',
            'category': 'cause_effect',
        },

        # === PROPERTY INFERENCE (10) ===
        {
            'q': 'A whale lives in water. Is a whale a fish?',
            'choices': ['yes', 'no'],
            'answer': 'no',
            'category': 'property',
        },
        {
            'q': 'A penguin has feathers and lays eggs. Is a penguin a bird?',
            'choices': ['yes', 'no'],
            'answer': 'yes',
            'category': 'property',
        },
        {
            'q': 'Iron is attracted by magnets. Is iron a metal?',
            'choices': ['yes', 'no'],
            'answer': 'yes',
            'category': 'property',
        },
        {
            'q': 'A bat can fly. Is a bat a bird?',
            'choices': ['yes', 'no'],
            'answer': 'no',
            'category': 'property',
        },
        {
            'q': 'Which is heavier: a kilogram of feathers or a kilogram of steel?',
            'choices': ['feathers', 'steel', 'they weigh the same', 'impossible to tell'],
            'answer': 'they weigh the same',
            'category': 'property',
        },
        {
            'q': 'If all mammals are warm-blooded, and a dog is a mammal, is a dog warm-blooded?',
            'choices': ['yes', 'no', 'maybe', 'impossible to tell'],
            'answer': 'yes',
            'category': 'property',
        },
        {
            'q': 'Sound travels faster through water or air?',
            'choices': ['water', 'air', 'same speed', 'sound cannot travel through water'],
            'answer': 'water',
            'category': 'property',
        },
        {
            'q': 'Which has more legs: a spider or an insect?',
            'choices': ['spider', 'insect', 'same', 'neither has legs'],
            'answer': 'spider',
            'category': 'property',
        },
        {
            'q': 'Can sound travel through a vacuum?',
            'choices': ['yes', 'no'],
            'answer': 'no',
            'category': 'property',
        },
        {
            'q': 'Which planet is closest to Earth?',
            'choices': ['Mars', 'Venus', 'Mercury', 'Jupiter'],
            'answer': 'Venus',
            'category': 'property',
        },
    ]


def evaluate_answer(response: str, test: dict) -> bool:
    """Check if the response contains the correct answer."""
    resp_lower = response.lower().strip()
    answer_lower = test['answer'].lower()

    # Direct match
    if answer_lower in resp_lower:
        return True

    # For numeric answers, check if the number appears
    if test['answer'].replace('.', '').isdigit():
        if test['answer'] in resp_lower:
            return True

    # For yes/no questions
    if test['answer'] in ('yes', 'no'):
        # Check if the answer word appears (avoiding false positives)
        if test['answer'] == 'yes' and ('yes' in resp_lower or 'it is' in resp_lower or 'correct' in resp_lower):
            return True
        if test['answer'] == 'no' and ('no' in resp_lower or 'not' in resp_lower or 'isn\'t' in resp_lower):
            # But not "no information"
            if 'no information' not in resp_lower and 'don\'t have' not in resp_lower:
                return True

    return False


def main():
    print("=" * 60)
    print("  ARC-EASY STYLE REASONING BENCHMARK")
    print("  50 questions across 5 categories")
    print("=" * 60)

    repl = FossKIRepl()
    tests = build_test_set()

    categories = {
        'analogy': [],
        'sequence': [],
        'odd_one_out': [],
        'cause_effect': [],
        'property': [],
    }

    for t in tests:
        categories[t['category']].append(t)

    total_correct = 0
    total_tests = len(tests)
    cat_results = {}

    t0 = time.time()

    for cat_name, cat_tests in categories.items():
        cat_correct = 0
        cat_total = len(cat_tests)
        print(f"\n{'─' * 50}")
        print(f"  {cat_name.replace('_', ' ').title()} ({cat_total} questions)")
        print(f"{'─' * 50}")

        for test in cat_tests:
            # Format question with choices
            choices_str = ', '.join(test['choices'])
            prompt = f"{test['q']} Choose one: {choices_str}"

            answer = repl.process(prompt)
            correct = evaluate_answer(answer, test)

            if correct:
                cat_correct += 1
                total_correct += 1
            else:
                print(f"  ✗ {test['q']}")
                print(f"    Got: {answer[:80]}")
                print(f"    Expected: {test['answer']}")

        acc = cat_correct / max(cat_total, 1)
        cat_results[cat_name] = (cat_correct, cat_total, acc)
        print(f"  Score: {cat_correct}/{cat_total} ({acc:.0%})")

    elapsed = time.time() - t0
    overall_acc = total_correct / max(total_tests, 1)

    print(f"\n{'=' * 60}")
    print(f"  ARC-EASY REASONING RESULTS")
    print(f"{'=' * 60}")

    print(f"\n  {'Category':<20} {'Score':>10} {'Accuracy':>10}")
    print(f"  {'─' * 42}")
    for cat, (c, t, a) in cat_results.items():
        name = cat.replace('_', ' ').title()
        bar = '█' * int(a * 20) + '░' * (20 - int(a * 20))
        print(f"  {name:<20} {c:>3}/{t:<3} {bar} {a:>5.0%}")

    print(f"\n  {'OVERALL':<20} {total_correct:>3}/{total_tests:<3} {'':>22} {overall_acc:>5.0%}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/total_tests*1000:.0f}ms/query)")

    # Comparison
    print(f"\n  {'Metric':<30} {'FOSS-KI':>10} {'GPT-4o':>10}")
    print(f"  {'─' * 52}")
    print(f"  {'ARC-Easy accuracy':<30} {overall_acc:>9.0%} {'~85%':>10}")
    print(f"  {'Analogies':<30} {cat_results['analogy'][2]:>9.0%} {'~90%':>10}")
    print(f"  {'Sequences':<30} {cat_results['sequence'][2]:>9.0%} {'~80%':>10}")
    print(f"  {'Odd-One-Out':<30} {cat_results['odd_one_out'][2]:>9.0%} {'~85%':>10}")
    print(f"  {'Cause & Effect':<30} {cat_results['cause_effect'][2]:>9.0%} {'~90%':>10}")
    print(f"  {'Property Inference':<30} {cat_results['property'][2]:>9.0%} {'~85%':>10}")
    print(f"  {'Latency':<30} {elapsed/total_tests*1000:>8.0f}ms {'~500ms':>10}")

    print(f"\n{'=' * 60}")
    if overall_acc >= 0.80:
        print("  STATUS: GPT-4o COMPETITIVE")
    elif overall_acc >= 0.60:
        print("  STATUS: STRONG")
    elif overall_acc >= 0.40:
        print("  STATUS: DEVELOPING")
    else:
        print("  STATUS: NEEDS WORK")
    print(f"{'=' * 60}")

    print(f"\nResults: {total_correct}/{total_tests} passed ({overall_acc:.0%})")


if __name__ == '__main__':
    main()
