#!/usr/bin/env python3
"""
Regression Suite — Unified Benchmark Runner
=============================================
Runs ALL benchmarks, records per-question results to JSON,
compares against stored baseline, flags regressions.

Usage:
    python3 benchmarks/regression_suite.py                  # Run all
    python3 benchmarks/regression_suite.py --only arc_easy  # Run one
    python3 benchmarks/regression_suite.py --save-baseline  # Save current as baseline
    python3 benchmarks/regression_suite.py --compare        # Compare to baseline
    python3 benchmarks/regression_suite.py --quick          # 50 samples per benchmark
"""

import sys
import os
import json
import time
import argparse
from datetime import datetime
from typing import Dict, Any, List, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'external_benchmarks')
RESULTS_DIR = os.path.join(BASE_DIR, 'benchmarks', 'results')
BASELINE_FILE = os.path.join(RESULTS_DIR, 'baseline.json')


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)


# ─── ARC-Easy (50 custom questions via REPL) ─────────────────────────

def run_arc_easy(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run ARC-Easy style benchmark (50 custom questions)."""
    from benchmarks.benchmark_arc_easy import build_test_set, evaluate_answer
    from repl import FossKIRepl

    tests = build_test_set()
    if limit:
        tests = tests[:limit]

    repl = FossKIRepl()
    results = []

    for i, test in enumerate(tests):
        choices_str = ', '.join(test['choices'])
        prompt = f"{test['q']} Choose one: {choices_str}"

        t0 = time.time()
        answer = repl.process(prompt)
        elapsed = time.time() - t0

        correct = evaluate_answer(answer, test)
        results.append({
            'id': f"arc_easy_{i}",
            'question': test['q'],
            'expected': test['answer'],
            'got': answer[:200],
            'correct': correct,
            'category': test['category'],
            'time_ms': round(elapsed * 1000, 1),
        })

    return _summarize('arc_easy', results)


# ─── ARC-Challenge (real dataset, 1172 questions) ────────────────────

def run_arc_challenge(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run ARC-Challenge benchmark on real dataset."""
    data_path = os.path.join(DATA_DIR, 'arc_challenge_test.jsonl')
    if not os.path.exists(data_path):
        return _skip('arc_challenge', 'Data file not found')

    from core.arc_solver import ARCSolver
    solver = ARCSolver()

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if limit:
        problems = problems[:limit]

    results = []
    for i, item in enumerate(problems):
        question = item['question']
        choices = item['choices']['text']
        labels = item['choices']['label']
        answer_key = item['answerKey']

        t0 = time.time()
        prediction = solver.solve(question, choices, labels)
        elapsed = time.time() - t0

        correct = prediction == answer_key
        correct_text = choices[labels.index(answer_key)] if answer_key in labels else '?'
        pred_text = choices[labels.index(prediction)] if prediction in labels else '?'

        results.append({
            'id': f"arc_challenge_{i}",
            'question': question[:200],
            'expected': f"{answer_key}: {correct_text}",
            'got': f"{prediction}: {pred_text}",
            'correct': correct,
            'time_ms': round(elapsed * 1000, 1),
        })

    return _summarize('arc_challenge', results)


# ─── HellaSwag (10042 examples) ──────────────────────────────────────

def run_hellaswag(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run HellaSwag benchmark."""
    data_path = os.path.join(DATA_DIR, 'hellaswag_val.jsonl')
    if not os.path.exists(data_path):
        return _skip('hellaswag', 'Data file not found')

    from core.hellaswag_solver import HellaSwagSolver

    glove_path = os.path.join(BASE_DIR, 'data', 'glove.6B.100d.txt')
    conceptnet_path = os.path.join(BASE_DIR, 'data', 'conceptnet_index.pkl')
    pmi_path = os.path.join(BASE_DIR, 'data', 'pmi_index.pkl')

    solver = HellaSwagSolver(
        glove_path=glove_path,
        conceptnet_path=conceptnet_path,
        pmi_path=pmi_path,
    )

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if limit:
        problems = problems[:limit]

    results = []
    for i, item in enumerate(problems):
        ctx = item.get('ctx', '')
        endings = item.get('endings', [])
        label = int(item.get('label', 0))

        t0 = time.time()
        pred = solver.solve(ctx, endings)
        elapsed = time.time() - t0

        correct = pred == label
        results.append({
            'id': f"hellaswag_{i}",
            'question': ctx[:150],
            'expected': label,
            'got': pred,
            'correct': correct,
            'time_ms': round(elapsed * 1000, 1),
        })

    return _summarize('hellaswag', results)


# ─── GSM8k (1319 math problems) ──────────────────────────────────────

def run_gsm8k(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run GSM8k math benchmark."""
    data_path = os.path.join(DATA_DIR, 'gsm8k_test.jsonl')
    if not os.path.exists(data_path):
        return _skip('gsm8k', 'Data file not found')

    from core.word_problem import WordProblemSolver

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    if limit:
        problems = problems[:limit]

    solver = WordProblemSolver()
    results = []

    for i, item in enumerate(problems):
        question = item['question']
        answer_str = item['answer']
        if '####' in answer_str:
            expected_str = answer_str.split('####')[-1].strip()
        else:
            expected_str = answer_str.strip()

        expected_str_clean = expected_str.replace(',', '').replace('$', '').replace('%', '')
        try:
            expected = float(expected_str_clean)
        except ValueError:
            expected = float('nan')

        t0 = time.time()
        result = solver.solve(question)
        elapsed = time.time() - t0

        got = None
        correct = False
        if result and result.get('answer') is not None:
            got = result['answer']
            try:
                got_f = float(got) if isinstance(got, (int, float)) else float(str(got).replace(',', '').replace('$', '').replace('%', ''))
                correct = abs(got_f - expected) < 0.01
            except (ValueError, TypeError):
                correct = False

        results.append({
            'id': f"gsm8k_{i}",
            'question': question[:200],
            'expected': expected_str,
            'got': str(got) if got is not None else 'NO_ANSWER',
            'correct': correct,
            'time_ms': round(elapsed * 1000, 1),
        })

    return _summarize('gsm8k', results)


# ─── PIQA (1838 examples) ────────────────────────────────────────────

def run_piqa(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run PIQA physical intuition benchmark."""
    data_path = os.path.join(DATA_DIR, 'piqa_valid.jsonl')
    labels_path = os.path.join(DATA_DIR, 'piqa_valid_labels.lst')
    if not os.path.exists(data_path) or not os.path.exists(labels_path):
        return _skip('piqa', 'Data files not found')

    from core.piqa_solver import PIQASolver

    glove_path = os.path.join(BASE_DIR, 'data', 'glove.6B.100d.txt')
    conceptnet_path = os.path.join(BASE_DIR, 'data', 'conceptnet_index.pkl')

    solver = PIQASolver(glove_path=glove_path, conceptnet_path=conceptnet_path)

    problems = []
    with open(data_path) as f:
        for line in f:
            problems.append(json.loads(line))

    labels = []
    with open(labels_path) as f:
        for line in f:
            labels.append(int(line.strip()))

    if limit:
        problems = problems[:limit]
        labels = labels[:limit]

    results = []
    for i, (item, label) in enumerate(zip(problems, labels)):
        goal = item.get('goal', '')
        sol1 = item.get('sol1', '')
        sol2 = item.get('sol2', '')

        t0 = time.time()
        pred = solver.solve(goal, sol1, sol2)
        elapsed = time.time() - t0

        correct = pred == label
        results.append({
            'id': f"piqa_{i}",
            'question': goal[:200],
            'expected': label,
            'got': pred,
            'correct': correct,
            'time_ms': round(elapsed * 1000, 1),
        })

    return _summarize('piqa', results)


# ─── Winograd (WSC273 + SuperGLUE) ───────────────────────────────────

def run_winograd(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run Winograd Schema Challenge."""
    from core.winograd_solver import WinogradSolver

    solver = WinogradSolver()
    results = []

    # WSC273
    wsc_path = os.path.join(DATA_DIR, 'wsc273.json')
    if os.path.exists(wsc_path):
        with open(wsc_path) as f:
            wsc_data = json.load(f)

        items = wsc_data if isinstance(wsc_data, list) else wsc_data.get('data', [])
        if limit:
            items = items[:limit]

        for i, item in enumerate(items):
            sentence = item.get('sentence', '')
            pronoun = item.get('text_parts', {}).get('pronoun', '')
            answers = item.get('answers', ['', ''])
            expected = item.get('correct_answer', '').strip().rstrip('.')

            t0 = time.time()
            pred = solver.solve_wsc273(sentence, pronoun,
                                       answers[0] if len(answers) > 0 else '',
                                       answers[1] if len(answers) > 1 else '')
            elapsed = time.time() - t0

            correct = pred == expected
            results.append({
                'id': f"winograd_{i}",
                'question': sentence[:200],
                'expected': expected,
                'got': pred,
                'correct': correct,
                'time_ms': round(elapsed * 1000, 1),
            })

    # SuperGLUE WSC val
    sg_path = os.path.join(DATA_DIR, 'superglue_wsc', 'WSC', 'val.jsonl')
    if os.path.exists(sg_path):
        sg_items = []
        with open(sg_path) as f:
            for line in f:
                sg_items.append(json.loads(line))

        if limit:
            sg_items = sg_items[:limit]

        for i, item in enumerate(sg_items):
            text = item.get('text', '')
            target = item.get('target', {})
            label = item.get('label', 0)

            t0 = time.time()
            pred = solver.solve_superglue(
                text,
                target.get('span1_text', ''),
                target.get('span2_text', ''),
                target.get('span1_index', 0),
                target.get('span2_index', 0),
            )
            elapsed = time.time() - t0

            correct = pred == (label == 1 if isinstance(label, int) else label)
            results.append({
                'id': f"winograd_sg_{i}",
                'question': text[:200],
                'expected': label,
                'got': pred,
                'correct': correct,
                'time_ms': round(elapsed * 1000, 1),
            })

    if not results:
        return _skip('winograd', 'No data files found')

    return _summarize('winograd', results)


# ─── bAbI (self-generated — marked as synthetic) ─────────────────────

def run_babi(limit: Optional[int] = None) -> Dict[str, Any]:
    """Run bAbI benchmark (currently self-generated — marked synthetic)."""
    from benchmarks.benchmark_babi import (
        task1_single_supporting_fact,
        task2_two_supporting_facts,
        task3_three_supporting_facts,
    )
    from core.knowledge import KnowledgeStore

    all_tasks = [
        ('task1', task1_single_supporting_fact()),
        ('task2', task2_two_supporting_facts()),
        ('task3', task3_three_supporting_facts()),
    ]

    results = []
    for task_name, stories in all_tasks:
        if limit and len(results) >= limit:
            break
        for j, story in enumerate(stories):
            if limit and len(results) >= limit:
                break
            # Load facts into temp KB
            temp_kb = KnowledgeStore()
            for s, r, o in story['facts']:
                temp_kb.store_fact(s, r, o)

            t0 = time.time()
            answer = None
            if 'query' in story:
                # Direct query: (subject, relation)
                subj, rel = story['query']
                for idx in temp_kb._entity_subject_index.get(subj.lower(), []):
                    _s, fr, fo = temp_kb.facts[idx]
                    if fr.lower() == rel.lower():
                        answer = fo
            else:
                # No direct query — skip (Task #108 will use real bAbI)
                continue

            elapsed = time.time() - t0
            expected = story['answer']
            correct = answer is not None and answer.lower() == expected.lower()

            results.append({
                'id': f"babi_{task_name}_{j}",
                'question': story['question'],
                'expected': expected,
                'got': answer or 'NO_ANSWER',
                'correct': correct,
                'synthetic': True,
                'time_ms': round(elapsed * 1000, 1),
            })

    return _summarize('babi_synthetic', results)


# ─── Helpers ──────────────────────────────────────────────────────────

def _summarize(name: str, results: List[Dict]) -> Dict[str, Any]:
    """Build summary from per-question results."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    accuracy = correct / total if total > 0 else 0.0
    avg_ms = sum(r['time_ms'] for r in results) / total if total > 0 else 0.0

    return {
        'benchmark': name,
        'total': total,
        'correct': correct,
        'accuracy': round(accuracy, 4),
        'avg_ms': round(avg_ms, 1),
        'results': results,
    }


def _skip(name: str, reason: str) -> Dict[str, Any]:
    """Return skip result."""
    return {
        'benchmark': name,
        'total': 0,
        'correct': 0,
        'accuracy': 0.0,
        'avg_ms': 0.0,
        'results': [],
        'skipped': reason,
    }


def compare_to_baseline(current: Dict, baseline: Dict) -> List[str]:
    """Compare current results to baseline, return list of regressions."""
    regressions = []

    for bench_name, bench_data in current.items():
        if bench_name == '_meta':
            continue
        base = baseline.get(bench_name)
        if not base:
            continue

        # Aggregate regression check
        cur_acc = bench_data['accuracy']
        base_acc = base['accuracy']
        if cur_acc < base_acc - 0.005:  # >0.5% drop = regression
            regressions.append(
                f"REGRESSION {bench_name}: {base_acc:.1%} → {cur_acc:.1%} "
                f"(Δ{(cur_acc - base_acc)*100:+.1f}pp)"
            )

        # Per-question regression check
        base_results = {r['id']: r['correct'] for r in base.get('results', [])}
        for r in bench_data.get('results', []):
            was_correct = base_results.get(r['id'])
            if was_correct is True and r['correct'] is False:
                regressions.append(
                    f"  ITEM REGRESSION {r['id']}: was correct, now wrong. "
                    f"Q: {r['question'][:80]}"
                )

    return regressions


# ─── Main ─────────────────────────────────────────────────────────────

BENCHMARKS = {
    'arc_easy': run_arc_easy,
    'arc_challenge': run_arc_challenge,
    'hellaswag': run_hellaswag,
    'gsm8k': run_gsm8k,
    'piqa': run_piqa,
    'winograd': run_winograd,
    'babi': run_babi,
}


def main():
    parser = argparse.ArgumentParser(description='FOSS-KI Regression Suite')
    parser.add_argument('--only', type=str, help='Run only this benchmark')
    parser.add_argument('--save-baseline', action='store_true', help='Save as baseline')
    parser.add_argument('--compare', action='store_true', help='Compare to baseline')
    parser.add_argument('--quick', action='store_true', help='50 samples per benchmark')
    parser.add_argument('--limit', type=int, default=None, help='Max items per benchmark')
    args = parser.parse_args()

    ensure_dirs()
    limit = args.limit or (50 if args.quick else None)

    benchmarks_to_run = {args.only: BENCHMARKS[args.only]} if args.only else BENCHMARKS

    print("=" * 70)
    print("  FOSS-KI REGRESSION SUITE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if limit:
        print(f"  Limit: {limit} items per benchmark")
    print("=" * 70)

    all_results = {}
    total_t0 = time.time()

    for name, runner in benchmarks_to_run.items():
        print(f"\n{'─' * 70}")
        print(f"  Running: {name}")
        print(f"{'─' * 70}")

        try:
            result = runner(limit=limit)
            all_results[name] = result

            if result.get('skipped'):
                print(f"  SKIPPED: {result['skipped']}")
            else:
                print(f"  Result: {result['correct']}/{result['total']} "
                      f"({result['accuracy']:.1%}) "
                      f"[{result['avg_ms']:.0f}ms/item]")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = _skip(name, str(e))

    total_elapsed = time.time() - total_t0

    # Add metadata
    all_results['_meta'] = {
        'timestamp': datetime.now().isoformat(),
        'total_time_s': round(total_elapsed, 1),
        'limit': limit,
    }

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_path = os.path.join(RESULTS_DIR, f'run_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Results saved: {results_path}")

    # Save baseline if requested
    if args.save_baseline:
        with open(BASELINE_FILE, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        print(f"  Baseline saved: {BASELINE_FILE}")

    # Compare to baseline
    if args.compare or os.path.exists(BASELINE_FILE):
        if os.path.exists(BASELINE_FILE):
            with open(BASELINE_FILE) as f:
                baseline = json.load(f)
            regressions = compare_to_baseline(all_results, baseline)

            if regressions:
                print(f"\n{'!' * 70}")
                print(f"  ⚠ {len(regressions)} REGRESSIONS DETECTED")
                print(f"{'!' * 70}")
                for reg in regressions[:20]:
                    print(f"  {reg}")
            else:
                print(f"\n  ✓ No regressions vs baseline")

    # Summary table
    print(f"\n{'=' * 70}")
    print(f"  {'Benchmark':<20} {'Score':>12} {'Accuracy':>10} {'Avg ms':>10}")
    print(f"  {'─' * 54}")
    total_correct = 0
    total_items = 0
    for name, data in sorted(all_results.items()):
        if name == '_meta':
            continue
        if data.get('skipped'):
            print(f"  {name:<20} {'SKIPPED':>12} {'':>10} {'':>10}")
        else:
            print(f"  {name:<20} {data['correct']:>5}/{data['total']:<5} "
                  f"{data['accuracy']:>9.1%} {data['avg_ms']:>9.1f}")
            total_correct += data['correct']
            total_items += data['total']

    if total_items > 0:
        overall = total_correct / total_items
        print(f"  {'─' * 54}")
        print(f"  {'TOTAL':<20} {total_correct:>5}/{total_items:<5} "
              f"{overall:>9.1%}")

    print(f"\n  Total time: {total_elapsed:.1f}s")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
