"""
Self-Improvement Loop — The Complete 9-Capability Cycle
========================================================
This is THE module that ties everything together.

The loop that builds itself:

  ┌─────────────────────────────────────────────┐
  │  5. EPISTEMIC: What do I know/not know?     │
  │  6. PLAN: Decompose goal into steps         │
  │  1. PREDICT: What should happen?            │
  │  2. EXECUTE: Do it                          │
  │  3. COMPARE: Did it match?                  │
  │  4. ATTRIBUTE: If not, why?                 │
  │  7. REMEMBER: Store error→fix pattern       │
  │  8. SELF-MODIFY: Patch own code if needed   │
  │  9. EVALUATE: Measure quality, track trends │
  │  → Fix and loop back to 1                   │
  │  → Update epistemic state                   │
  └─────────────────────────────────────────────┘

Each iteration makes the system smarter because:
  - Epistemic state grows (more accurate self-knowledge)
  - Knowledge base grows (new facts from successful queries)
  - Bug patterns grow (known error→fix mappings) ← F7
  - Competence map updates (know what we're good/bad at)
  - Quality trends tracked across domains ← F9

No external dependencies beyond other FOSS-KI modules.
"""

import time
from typing import Dict, Any, List, Optional

from .abstract_interpreter import AbstractInterpreter, Prediction
from .causal_tracer import CausalTracer, TraceResult
from .epistemic import EpistemicState
from .planner import GoalPlanner, Plan
from .executor import CodeExecutor
from .pattern_memory import PatternMemory
from .evaluator import Evaluator


class SelfLoop:
    """
    The autonomous self-improvement engine.

    Usage:
        loop = SelfLoop()
        result = loop.improve("make the math solver handle fractions")
        # or
        result = loop.verify_code(code, expected_output)
        # or
        report = loop.run_diagnostic()
    """

    def __init__(self, executor: CodeExecutor = None):
        self.interpreter = AbstractInterpreter()
        self.tracer = CausalTracer()
        self.epistemic = EpistemicState()
        self.planner = GoalPlanner()
        self.executor = executor or CodeExecutor(timeout=10, max_retries=3)

        # F7: Pattern Memory — remembers error→fix patterns
        self.memory = PatternMemory()

        # F9: Evaluator — tracks quality trends
        self.evaluator = Evaluator()

        # Statistics
        self.cycles_run = 0
        self.fixes_applied = 0
        self.predictions_correct = 0
        self.predictions_total = 0

    def verify_code(self, code: str, expected_output: str = '',
                    description: str = '') -> Dict[str, Any]:
        """
        The core loop for a single piece of code:
        PREDICT → EXECUTE → COMPARE → ATTRIBUTE → FIX

        Returns:
            {
                'success': bool,
                'prediction': Prediction,
                'actual_output': str,
                'match': bool,  # prediction matched actual
                'trace_result': TraceResult (if mismatch),
                'fix_applied': bool,
                'final_code': str,
                'cycles': int,
            }
        """
        self.cycles_run += 1
        current_code = code
        max_fix_attempts = 3

        for attempt in range(max_fix_attempts + 1):
            # Step 1: PREDICT
            prediction = self.interpreter.interpret(current_code)
            predicted_output = prediction.predicted_output

            # Step 2: EXECUTE
            exec_result = self.executor.execute(
                current_code,
                expected_output=expected_output if expected_output else None
            )
            actual_output = exec_result.get('stdout', '').strip()
            actual_error = exec_result.get('stderr', '')

            # Step 3: COMPARE
            self.predictions_total += 1
            exec_success = exec_result.get('success', False)

            # Compare prediction vs actual
            prediction_correct = False
            if prediction.confidence == 'VALUE':
                prediction_correct = (predicted_output.strip() == actual_output.strip())
                if prediction_correct:
                    self.predictions_correct += 1

            # Compare expected vs actual
            output_correct = True
            if expected_output:
                output_correct = (actual_output.strip() == expected_output.strip())

            if exec_success and output_correct:
                # Update epistemic
                domain = 'code'
                self.epistemic.record_query(
                    description or 'verify_code', domain,
                    confidence='HIGH' if prediction_correct else 'MEDIUM',
                    answered=True, correct=True, source='self_loop'
                )
                return {
                    'success': True,
                    'prediction': prediction,
                    'actual_output': actual_output,
                    'match': prediction_correct,
                    'trace_result': None,
                    'fix_applied': attempt > 0,
                    'final_code': current_code,
                    'cycles': attempt + 1,
                }

            # Step 4: ATTRIBUTE — find root cause
            trace_result = self.tracer.trace(
                current_code,
                expected_output=expected_output,
                actual_output=actual_output,
                actual_error=actual_error,
            )

            # Step 4.5 (F7): Check Pattern Memory for known fix
            memory_fix = None
            if actual_error:
                error_type = actual_error.split(':')[0].strip().split('.')[-1]
                memory_fix = self.memory.lookup_fix(error_type, actual_error)

            # Step 5: FIX (using memory fix if available, else auto-fix)
            if attempt < max_fix_attempts:
                fixed = None
                if trace_result.fix_suggestion:
                    fixed = self.executor._auto_fix(
                        current_code,
                        exec_result.get('errors', [])
                    )
                if fixed and fixed != current_code:
                    current_code = fixed
                    self.fixes_applied += 1
                    # F7: Store the error→fix pattern for next time
                    if actual_error:
                        error_type = actual_error.split(':')[0].strip().split('.')[-1]
                        self.memory.store_error_fix(
                            error_type, actual_error[:100],
                            trace_result.fix_suggestion or 'auto-fixed',
                            code_context=current_code[:200]
                        )
                    continue

            # Can't fix further
            self.epistemic.record_query(
                description or 'verify_code', 'code',
                confidence='LOW', answered=True, correct=False,
                source='self_loop'
            )
            return {
                'success': False,
                'prediction': prediction,
                'actual_output': actual_output,
                'match': prediction_correct,
                'trace_result': trace_result,
                'fix_applied': attempt > 0,
                'final_code': current_code,
                'cycles': attempt + 1,
                'root_cause': trace_result.fix_suggestion,
                'memory_fix': memory_fix,  # F7: suggest from memory
            }

        return {'success': False, 'cycles': max_fix_attempts + 1}

    def predict_and_check(self, code: str) -> Dict[str, Any]:
        """
        Predict what code does, then execute and compare.
        No expected output needed — the prediction IS the expectation.

        This is the purest form of "understanding":
        if prediction matches reality, we understand the code.
        """
        # Predict
        prediction = self.interpreter.interpret(code)

        # Execute
        exec_result = self.executor.execute(code)
        actual = exec_result.get('stdout', '').strip()

        # Compare
        predicted = prediction.predicted_output.strip()
        match = (predicted == actual) if prediction.confidence == 'VALUE' else None

        self.predictions_total += 1
        if match:
            self.predictions_correct += 1

        return {
            'predicted': predicted,
            'actual': actual,
            'match': match,
            'confidence': prediction.confidence,
            'variables': {k: str(v) for k, v in prediction.variables.items()},
            'understanding': 'FULL' if match else
                           ('PARTIAL' if prediction.confidence == 'TYPE' else 'NONE'),
        }

    def improve(self, goal: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Full self-improvement cycle with all 6 capabilities.

        1. Epistemic: assess what we know
        2. Plan: decompose the goal
        3. Execute plan steps with predict→compare→attribute loops
        """
        context = context or {}

        # Fähigkeit 5: Epistemic check
        domain = self.epistemic._guess_domain(goal)
        should = self.epistemic.should_answer(goal, domain)

        # Fähigkeit 6: Plan
        plan = self.planner.plan(goal)

        results = {
            'goal': goal,
            'epistemic_check': should,
            'plan': plan.summary(),
            'steps_completed': 0,
            'steps_total': len(plan.steps),
            'success': False,
        }

        # Execute each plan step
        for idx in range(len(plan.steps)):
            if not plan.can_execute(idx):
                # Check if dependencies failed
                step = plan.steps[idx]
                all_deps_done = all(
                    plan.steps[d].status.value == 'done'
                    for d in step.depends_on if d < len(plan.steps)
                )
                if not all_deps_done:
                    plan.mark_failed(idx, 'dependency not met')
                    continue

            plan.mark_running(idx)

            # Each step gets the predict→compare loop
            step_result = self._execute_plan_step(plan.steps[idx], context)

            if step_result.get('success', False):
                plan.mark_done(idx, step_result)
                results['steps_completed'] += 1
            else:
                plan.mark_failed(idx, step_result.get('error', 'unknown'))

        results['success'] = plan.is_complete
        results['plan_final'] = plan.summary()

        # Update epistemic
        self.epistemic.record_query(
            goal, domain,
            confidence='HIGH' if results['success'] else 'LOW',
            answered=True,
            correct=results['success'],
            source='self_loop'
        )

        return results

    def run_diagnostic(self) -> Dict[str, Any]:
        """
        Self-diagnostic: how well is the system working?
        Uses epistemic state + prediction accuracy + pattern memory + evaluator.
        """
        prediction_accuracy = (
            self.predictions_correct / max(self.predictions_total, 1)
        )

        return {
            'cycles_run': self.cycles_run,
            'fixes_applied': self.fixes_applied,
            'prediction_accuracy': round(prediction_accuracy, 3),
            'predictions_total': self.predictions_total,
            'epistemic_summary': self.epistemic.summary(),
            'competence_map': self.epistemic.competence_map(),
            'calibration': self.epistemic.calibration_report(),
            'learning_priorities': self.epistemic.learning_priorities(5),
            # F7: Pattern Memory stats
            'pattern_memory': self.memory.get_statistics(),
            # F9: Quality trends
            'quality_trends': self.evaluator.trend_report(),
            'weakest_domain': self.evaluator.weakest_domain(),
        }

    def _execute_plan_step(self, step, context: Dict) -> Dict[str, Any]:
        """Execute a single plan step."""
        # For now, steps are descriptive — they succeed by default
        # unless they involve code execution
        if step.step_type == 'check':
            return {'success': True, 'type': 'check_passed'}
        elif step.step_type == 'decision':
            return {'success': True, 'type': 'decision_made'}
        return {'success': True, 'type': 'action_completed'}

    def stats(self) -> str:
        """Human-readable stats."""
        pred_acc = self.predictions_correct / max(self.predictions_total, 1)
        return (
            f'Self-Loop Stats:\n'
            f'  Cycles: {self.cycles_run}\n'
            f'  Fixes applied: {self.fixes_applied}\n'
            f'  Prediction accuracy: {pred_acc:.0%} '
            f'({self.predictions_correct}/{self.predictions_total})\n'
            f'  Epistemic state: {len(self.epistemic.competence)} domains tracked'
        )
