#!/usr/bin/env python3
"""
Ablation Study — Measure True Baseline Without Hardcoded Knowledge
====================================================================
For each benchmark solver, runs WITH and WITHOUT benchmark-specific
hardcoded knowledge to measure the actual system contribution.

Reports the gap: what's real reasoning vs memorization.
"""

import sys
import os
import json
import time
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'external_benchmarks')
RESULTS_DIR = os.path.join(BASE_DIR, 'benchmarks', 'results')


def ablate_arc_challenge(limit=50):
    """ARC-Challenge: Run with and without SCIENCE_KB."""
    data_path = os.path.join(DATA_DIR, 'arc_challenge_test.jsonl')
    if not os.path.exists(data_path):
        return None

    import core.arc_solver as arc_mod
    from core.arc_solver import ARCSolver

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))
    problems = problems[:limit]

    results = {}

    # === WITH hardcoded SCIENCE_KB ===
    solver = ARCSolver()
    correct_with = 0
    for item in problems:
        pred = solver.solve(item['question'], item['choices']['text'], item['choices']['label'])
        if pred == item['answerKey']:
            correct_with += 1
    results['with_science_kb'] = {
        'correct': correct_with,
        'total': len(problems),
        'accuracy': round(correct_with / len(problems), 4),
        'kb_entries': len(arc_mod.SCIENCE_KB),
    }

    # === WITHOUT SCIENCE_KB (monkey-patch to empty) ===
    original_kb = arc_mod.SCIENCE_KB
    original_index = arc_mod._KB_INDEX
    original_word_sets = arc_mod._KB_WORD_SETS
    try:
        arc_mod.SCIENCE_KB = []
        arc_mod._KB_INDEX = {}
        arc_mod._KB_WORD_SETS = []
        solver2 = ARCSolver()
        correct_without = 0
        for item in problems:
            pred = solver2.solve(item['question'], item['choices']['text'], item['choices']['label'])
            if pred == item['answerKey']:
                correct_without += 1
        results['without_science_kb'] = {
            'correct': correct_without,
            'total': len(problems),
            'accuracy': round(correct_without / len(problems), 4),
        }
    finally:
        arc_mod.SCIENCE_KB = original_kb
        arc_mod._KB_INDEX = original_index
        arc_mod._KB_WORD_SETS = original_word_sets

    results['delta'] = round(results['with_science_kb']['accuracy'] - results['without_science_kb']['accuracy'], 4)
    return results


def ablate_piqa(limit=50):
    """PIQA: Run with and without hardcoded rules."""
    data_path = os.path.join(DATA_DIR, 'piqa_valid.jsonl')
    labels_path = os.path.join(DATA_DIR, 'piqa_valid_labels.lst')
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        return None

    import core.piqa_solver as piqa_mod
    from core.piqa_solver import PIQASolver

    glove_path = os.path.join(BASE_DIR, 'data', 'glove.6B.100d.txt')
    conceptnet_path = os.path.join(BASE_DIR, 'data', 'conceptnet_index.pkl')

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))
    labels = []
    with open(labels_path) as f:
        for line in f:
            labels.append(int(line.strip()))
    problems = problems[:limit]
    labels = labels[:limit]

    results = {}

    # === WITH all rules ===
    solver = PIQASolver(glove_path=glove_path, conceptnet_path=conceptnet_path)
    correct_with = 0
    for item, label in zip(problems, labels):
        pred = solver.solve(item.get('goal', ''), item.get('sol1', ''), item.get('sol2', ''))
        if pred == label:
            correct_with += 1
    results['with_rules'] = {
        'correct': correct_with,
        'total': len(problems),
        'accuracy': round(correct_with / len(problems), 4),
        'rule_counts': {
            'TOOL_USE': len(piqa_mod.TOOL_USE),
            'MATERIAL_PROPS': len(piqa_mod.MATERIAL_PROPS),
            'IMPLAUSIBLE_PAIRS': len(piqa_mod.IMPLAUSIBLE_PAIRS),
            'PHYSICAL_CAUSATION': len(piqa_mod.PHYSICAL_CAUSATION),
            'PREP_PREFERENCES': len(piqa_mod.PREP_PREFERENCES),
        },
    }

    # === WITHOUT rules (empty all dicts/lists) ===
    saved = {}
    rule_names = ['TOOL_USE', 'MATERIAL_PROPS', 'IMPLAUSIBLE_PAIRS',
                  'PHYSICAL_CAUSATION', 'PREP_PREFERENCES',
                  'COOKING_WORDS', 'CLEANING_WORDS', 'REPAIR_WORDS', 'CRAFT_WORDS']
    try:
        for name in rule_names:
            saved[name] = getattr(piqa_mod, name, None)
            obj = getattr(piqa_mod, name, None)
            if isinstance(obj, dict):
                setattr(piqa_mod, name, {})
            elif isinstance(obj, (list, set, frozenset)):
                setattr(piqa_mod, name, type(obj)())

        solver2 = PIQASolver(glove_path=glove_path, conceptnet_path=conceptnet_path)
        correct_without = 0
        for item, label in zip(problems, labels):
            pred = solver2.solve(item.get('goal', ''), item.get('sol1', ''), item.get('sol2', ''))
            if pred == label:
                correct_without += 1
        results['without_rules'] = {
            'correct': correct_without,
            'total': len(problems),
            'accuracy': round(correct_without / len(problems), 4),
        }
    finally:
        for name, val in saved.items():
            if val is not None:
                setattr(piqa_mod, name, val)

    results['delta'] = round(results['with_rules']['accuracy'] - results['without_rules']['accuracy'], 4)
    return results


def ablate_winograd(limit=50):
    """Winograd: Run with and without _check_rules."""
    from core.winograd_solver import WinogradSolver

    wsc_path = os.path.join(DATA_DIR, 'wsc273.json')
    if not os.path.exists(wsc_path):
        return None

    with open(wsc_path) as f:
        wsc_data = json.load(f)
    items = wsc_data if isinstance(wsc_data, list) else wsc_data.get('data', [])
    items = items[:limit]

    results = {}

    # === WITH rules ===
    solver = WinogradSolver()
    correct_with = 0
    for item in items:
        sentence = item.get('sentence', '')
        pronoun = item.get('text_parts', {}).get('pronoun', '')
        answers = item.get('answers', ['', ''])
        expected = item.get('correct_answer', '').strip().rstrip('.')
        pred = solver.solve_wsc273(sentence, pronoun,
                                   answers[0] if answers else '',
                                   answers[1] if len(answers) > 1 else '')
        if pred == expected:
            correct_with += 1
    results['with_rules'] = {
        'correct': correct_with,
        'total': len(items),
        'accuracy': round(correct_with / len(items), 4),
    }

    # === WITHOUT _check_rules (monkey-patch to return None always) ===
    original_check_rules = solver._check_rules
    try:
        solver._check_rules = lambda *args, **kwargs: None
        correct_without = 0
        for item in items:
            sentence = item.get('sentence', '')
            pronoun = item.get('text_parts', {}).get('pronoun', '')
            answers = item.get('answers', ['', ''])
            expected = item.get('correct_answer', '').strip().rstrip('.')
            pred = solver.solve_wsc273(sentence, pronoun,
                                       answers[0] if answers else '',
                                       answers[1] if len(answers) > 1 else '')
            if pred == expected:
                correct_without += 1
        results['without_rules'] = {
            'correct': correct_without,
            'total': len(items),
            'accuracy': round(correct_without / len(items), 4),
        }
    finally:
        solver._check_rules = original_check_rules

    results['delta'] = round(results['with_rules']['accuracy'] - results['without_rules']['accuracy'], 4)
    return results


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    limit = 100  # 100 samples for statistical reliability

    print("=" * 70)
    print("  ABLATION STUDY — Hardcoded Knowledge Impact")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  {limit} samples per benchmark")
    print("=" * 70)

    all_results = {}

    # ARC-Challenge
    print(f"\n{'─' * 70}")
    print("  ARC-Challenge: SCIENCE_KB (1556 entries)")
    print(f"{'─' * 70}")
    arc = ablate_arc_challenge(limit)
    if arc:
        all_results['arc_challenge'] = arc
        print(f"  WITH SCIENCE_KB:    {arc['with_science_kb']['correct']}/{arc['with_science_kb']['total']} ({arc['with_science_kb']['accuracy']:.1%})")
        print(f"  WITHOUT SCIENCE_KB: {arc['without_science_kb']['correct']}/{arc['without_science_kb']['total']} ({arc['without_science_kb']['accuracy']:.1%})")
        print(f"  Delta:              {arc['delta']:+.1%}")

    # PIQA
    print(f"\n{'─' * 70}")
    print("  PIQA: Physical rules (744 entries)")
    print(f"{'─' * 70}")
    piqa = ablate_piqa(limit)
    if piqa:
        all_results['piqa'] = piqa
        print(f"  WITH rules:    {piqa['with_rules']['correct']}/{piqa['with_rules']['total']} ({piqa['with_rules']['accuracy']:.1%})")
        print(f"  WITHOUT rules: {piqa['without_rules']['correct']}/{piqa['without_rules']['total']} ({piqa['without_rules']['accuracy']:.1%})")
        print(f"  Delta:         {piqa['delta']:+.1%}")

    # Winograd
    print(f"\n{'─' * 70}")
    print("  Winograd: _check_rules (69 rule returns)")
    print(f"{'─' * 70}")
    wino = ablate_winograd(limit)
    if wino:
        all_results['winograd'] = wino
        print(f"  WITH rules:    {wino['with_rules']['correct']}/{wino['with_rules']['total']} ({wino['with_rules']['accuracy']:.1%})")
        print(f"  WITHOUT rules: {wino['without_rules']['correct']}/{wino['without_rules']['total']} ({wino['without_rules']['accuracy']:.1%})")
        print(f"  Delta:         {wino['delta']:+.1%}")

    # Summary
    print(f"\n{'=' * 70}")
    print("  ABLATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"\n  {'Benchmark':<20} {'With HC':>10} {'Without HC':>12} {'Delta':>8} {'HC entries':>12}")
    print(f"  {'─' * 64}")
    for name, data in all_results.items():
        with_key = [k for k in data if k.startswith('with_')][0]
        without_key = [k for k in data if k.startswith('without_')][0]
        hc = data[with_key].get('kb_entries', data[with_key].get('rule_counts', '?'))
        if isinstance(hc, dict):
            hc = sum(hc.values())
        print(f"  {name:<20} {data[with_key]['accuracy']:>9.1%} {data[without_key]['accuracy']:>11.1%} {data['delta']:>+7.1%} {hc:>12}")

    print(f"\n  HC = Hardcoded knowledge")
    print(f"  'Without HC' = TRUE system reasoning baseline")
    print(f"{'=' * 70}")

    # Save
    results_path = os.path.join(RESULTS_DIR, f'ablation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {results_path}")


if __name__ == '__main__':
    main()
