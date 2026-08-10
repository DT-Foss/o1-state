#!/usr/bin/env python3
"""
Self-Test Run — The System Tests Itself
=========================================
Runs all benchmarks through the Self-Loop:
  1. Predict what each test should do
  2. Execute it
  3. Compare prediction vs reality
  4. Attribute failures
  5. Store patterns (F7)
  6. Track quality (F9)
  7. Report what to improve

This is the first autonomous self-improvement cycle.
"""

import sys
import os
import time
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.self_loop import SelfLoop
from core.epistemic import EpistemicState
from core.abstract_interpreter import AbstractInterpreter


def run_benchmark(path: str) -> dict:
    """Run a benchmark file and capture results."""
    t0 = time.time()
    try:
        result = subprocess.run(
            ['python3', path],
            capture_output=True, text=True, timeout=300,
        )
        elapsed = time.time() - t0

        # Parse pass/fail from output
        stdout = result.stdout
        passed = 0
        total = 0
        import re

        # Strategy 1: Look for summary lines with results/score
        for line in stdout.split('\n'):
            ll = line.lower().strip()
            # "Results: X/Y passed", "RESULT: X/Y (100%)", "Score: X/Y"
            if any(kw in ll for kw in ['result', 'score:', 'total:', 'summary']):
                m = re.search(r'(\d+)/(\d+)', line)
                if m:
                    p, t = int(m.group(1)), int(m.group(2))
                    # Take the largest total (final summary line)
                    if t > total:
                        passed = p
                        total = t

        # Strategy 2: Count PASS/FAIL lines
        if total == 0:
            pass_count = 0
            fail_count = 0
            for line in stdout.split('\n'):
                stripped = line.strip()
                if any(stripped.startswith(p) for p in
                       ['PASS:', '[PASS]', '  PASS:', '  [PASS]']):
                    pass_count += 1
                elif any(stripped.startswith(p) for p in
                         ['FAIL:', '[FAIL]', '  FAIL:', '  [FAIL]']):
                    fail_count += 1
            if pass_count + fail_count > 0:
                passed = pass_count
                total = pass_count + fail_count

        return {
            'path': path,
            'success': result.returncode == 0,
            'passed': passed,
            'total': total,
            'elapsed': round(elapsed, 2),
            'stdout': stdout,
            'stderr': result.stderr,
            'returncode': result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            'path': path,
            'success': False,
            'passed': 0,
            'total': 0,
            'elapsed': 300,
            'stdout': '',
            'stderr': 'TIMEOUT',
            'returncode': -1,
        }
    except Exception as e:
        return {
            'path': path,
            'success': False,
            'passed': 0,
            'total': 0,
            'elapsed': 0,
            'stdout': '',
            'stderr': str(e),
            'returncode': -1,
        }


def main():
    print("=" * 60)
    print("  FOSS-KI SELF-TEST RUN")
    print("  The system tests itself.")
    print("=" * 60)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    bench_dir = os.path.join(base, 'benchmarks')

    # Find all benchmark files
    benchmarks = sorted([
        os.path.join(bench_dir, f)
        for f in os.listdir(bench_dir)
        if f.startswith('benchmark_') and f.endswith('.py')
    ])

    print(f"\nFound {len(benchmarks)} benchmark files.\n")

    # Initialize self-loop components
    sl = SelfLoop()
    ai = AbstractInterpreter()

    total_passed = 0
    total_tests = 0
    total_benchmarks = 0
    failed_benchmarks = []
    results = []
    predictions_correct = 0
    predictions_total = 0

    t0_all = time.time()

    for bench_path in benchmarks:
        bench_name = os.path.basename(bench_path).replace('benchmark_', '').replace('.py', '')
        print(f"{'─' * 50}")

        # PREDICT: Will this benchmark pass?
        comp = sl.epistemic.competence_map()
        if bench_name in comp:
            predicted_pass = comp[bench_name]['accuracy'] > 0.5
            prediction_conf = 'HIGH' if comp[bench_name]['accuracy'] > 0.8 else 'MEDIUM'
        else:
            # First run — predict success (optimistic)
            predicted_pass = True
            prediction_conf = 'LOW'

        print(f"Running: {bench_name} (predicted: {'PASS' if predicted_pass else 'FAIL'})")

        result = run_benchmark(bench_path)

        # COMPARE: Was prediction correct?
        actual_pass = result['success']
        if predicted_pass == actual_pass:
            predictions_correct += 1
        predictions_total += 1
        results.append(result)

        if result['success'] and result['total'] > 0:
            total_passed += result['passed']
            total_tests += result['total']
            total_benchmarks += 1
            status = f"✓ {result['passed']}/{result['total']}"

            # Record in epistemic state
            sl.epistemic.record_query(
                bench_name, 'benchmark',
                confidence='HIGH',
                answered=True,
                correct=(result['passed'] == result['total']),
                source='self_test'
            )

            # Track in evaluator
            sl.evaluator.evaluate(
                query=bench_name,
                response=f"{result['passed']}/{result['total']} passed",
                expected=f"{result['total']}/{result['total']} passed",
                domain=bench_name,
                latency_ms=result['elapsed'] * 1000,
                confidence=result['passed'] / max(result['total'], 1),
                was_correct=(result['passed'] == result['total']),
            )
        elif result['success']:
            total_benchmarks += 1
            status = "✓ (no counts)"
            sl.epistemic.record_query(
                bench_name, 'benchmark',
                confidence='MEDIUM', answered=True, correct=True,
                source='self_test'
            )
        else:
            failed_benchmarks.append(bench_name)
            status = f"✗ FAILED"

            # Analyze failure
            if result['stderr']:
                error_lines = result['stderr'].strip().split('\n')
                last_error = error_lines[-1] if error_lines else 'unknown'
                print(f"  Error: {last_error[:100]}")

                # Store pattern
                error_type = last_error.split(':')[0].strip()
                sl.memory.store_error_fix(
                    error_type,
                    last_error[:100],
                    f'Benchmark {bench_name} failed',
                    code_context=bench_name
                )

            sl.epistemic.record_query(
                bench_name, 'benchmark',
                confidence='LOW', answered=True, correct=False,
                source='self_test'
            )

        print(f"  {status} ({result['elapsed']}s)")

    elapsed_all = time.time() - t0_all

    # === REPORT ===
    print(f"\n{'=' * 60}")
    print("  SELF-TEST REPORT")
    print(f"{'=' * 60}")
    print(f"\nBenchmarks: {total_benchmarks}/{len(benchmarks)} passed")
    print(f"Tests: {total_passed}/{total_tests} passed "
          f"({total_passed/max(total_tests,1)*100:.0f}%)")
    print(f"Time: {elapsed_all:.1f}s")

    if failed_benchmarks:
        print(f"\nFAILED benchmarks ({len(failed_benchmarks)}):")
        for fb in failed_benchmarks:
            print(f"  ✗ {fb}")

    # Epistemic report
    print(f"\n{'─' * 50}")
    print("EPISTEMIC STATE:")
    comp = sl.epistemic.competence_map()
    for domain, info in sorted(comp.items(), key=lambda x: x[1]['accuracy'], reverse=True):
        bar = '█' * int(info['accuracy'] * 20) + '░' * (20 - int(info['accuracy'] * 20))
        print(f"  {domain:30s} {bar} {info['accuracy']:5.0%}")

    # Learning priorities
    priorities = sl.epistemic.learning_priorities(5)
    if priorities:
        print(f"\nLEARNING PRIORITIES:")
        for p in priorities:
            print(f"  → {p['concept']} (score: {p['priority_score']})")

    # Pattern memory
    mem_stats = sl.memory.get_statistics()
    print(f"\nPATTERN MEMORY: {mem_stats['total_patterns']} patterns stored")

    # Diagnostic
    diag = sl.run_diagnostic()
    bench_pred_acc = predictions_correct / max(predictions_total, 1)
    print(f"\nBENCHMARK PREDICTION: {predictions_correct}/{predictions_total} "
          f"({bench_pred_acc:.0%})")
    print(f"LOOP PREDICTION ACCURACY: {diag['prediction_accuracy']:.0%}")

    # Weakest domain
    weakest = diag.get('weakest_domain')
    if weakest:
        print(f"WEAKEST DOMAIN: {weakest}")

    print(f"\n{'=' * 60}")
    if total_passed == total_tests and not failed_benchmarks:
        print("  ALL SYSTEMS NOMINAL")
    else:
        print(f"  {len(failed_benchmarks)} BENCHMARKS NEED ATTENTION")
    print(f"{'=' * 60}")

    return {
        'passed': total_passed,
        'total': total_tests,
        'benchmarks_ok': total_benchmarks,
        'benchmarks_total': len(benchmarks),
        'failed': failed_benchmarks,
        'elapsed': elapsed_all,
        'prediction_accuracy': bench_pred_acc,
    }


if __name__ == '__main__':
    result = main()
    sys.exit(0 if not result['failed'] else 1)
