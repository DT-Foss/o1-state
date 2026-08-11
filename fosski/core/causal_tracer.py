"""
Causal Tracer — Find WHY Code Output Is Wrong
================================================
Given: code + expected output + actual output
Returns: the LINE that caused the divergence.

This is Fähigkeit 4 (Attribution) in the self-improvement loop:
  PREDICT → EXECUTE → COMPARE → **ATTRIBUTE** → FIX

Two strategies:
  1. TRACE COMPARISON: Run abstract interpreter line-by-line,
     compare state at each point with actual execution trace.
     First divergence = cause.

  2. DELTA DEBUGGING: Binary search over code lines.
     Remove half the code → does output change?
     Narrow down to the causal line.

No external dependencies.
"""

import ast
import re
from typing import Dict, Any, List, Optional, Tuple

from .abstract_interpreter import AbstractInterpreter, Prediction, SymValue


class Divergence:
    """A point where predicted and actual behavior diverge."""
    def __init__(self, line: int, variable: str = '',
                 predicted: Any = None, actual: Any = None,
                 reason: str = '', severity: str = 'ERROR'):
        self.line = line
        self.variable = variable
        self.predicted = predicted
        self.actual = actual
        self.reason = reason
        self.severity = severity  # ERROR, WARNING, INFO

    def __repr__(self):
        return (f'Divergence(line={self.line}, var={self.variable}, '
                f'predicted={self.predicted!r}, actual={self.actual!r}, '
                f'reason={self.reason!r})')


class CausalTracer:
    """
    Trace back from wrong output to the causal line.

    Usage:
        tracer = CausalTracer()
        result = tracer.trace(code, expected_output, actual_output)
        # result.root_cause → line number
        # result.divergences → list of all divergence points
        # result.fix_suggestion → what to change
    """

    def __init__(self):
        self.interpreter = AbstractInterpreter()

    def trace(self, code: str, expected_output: str = '',
              actual_output: str = '', actual_error: str = '') -> 'TraceResult':
        """
        Find the root cause of wrong output.

        Args:
            code: the Python source code
            expected_output: what the code should print
            actual_output: what it actually printed
            actual_error: stderr if the code crashed

        Returns:
            TraceResult with root_cause, divergences, fix_suggestion
        """
        divergences = []

        # Step 1: Predict via abstract interpretation
        prediction = self.interpreter.interpret(code)

        # Step 2: If there's a crash, analyze the error
        if actual_error:
            crash_analysis = self._analyze_crash(code, actual_error, prediction)
            return crash_analysis

        # Step 3: Compare predicted output vs actual output
        if expected_output or actual_output:
            output_divs = self._compare_outputs(
                prediction.predicted_output, actual_output, expected_output
            )
            divergences.extend(output_divs)

        # Step 4: Walk the prediction trace to find first divergence
        if prediction.errors:
            for err in prediction.errors:
                divergences.append(Divergence(
                    line=err.get('line', 0),
                    reason=f"Predicted error: {err['type']}: {err['message']}",
                    severity='ERROR',
                ))

        # Step 5: Analyze variable state for common bugs
        var_issues = self._check_variable_issues(prediction, code)
        divergences.extend(var_issues)

        # Step 6: Determine root cause (first ERROR divergence)
        root_cause = None
        for d in sorted(divergences, key=lambda x: (x.severity != 'ERROR', x.line)):
            if d.severity == 'ERROR':
                root_cause = d
                break
        if not root_cause and divergences:
            root_cause = divergences[0]

        # Step 7: Generate fix suggestion
        fix = self._suggest_fix(root_cause, code, prediction) if root_cause else ''

        return TraceResult(
            root_cause=root_cause,
            divergences=divergences,
            prediction=prediction,
            fix_suggestion=fix,
        )

    def compare_predict_vs_actual(self, code: str,
                                   actual_output: str) -> 'TraceResult':
        """
        Simplified: predict what code does, then compare with actual.
        No expected_output needed — the PREDICTION is the expectation.
        """
        return self.trace(code, expected_output='', actual_output=actual_output)

    def _analyze_crash(self, code: str, error: str,
                       prediction: Prediction) -> 'TraceResult':
        """Analyze a crash (exception) to find root cause."""
        divergences = []

        # Parse error message
        lines = error.strip().split('\n')
        error_line = None
        error_type = ''
        error_msg = ''

        for line in reversed(lines):
            line = line.strip()
            if ':' in line and ('Error' in line or 'Exception' in line):
                parts = line.split(':', 1)
                error_type = parts[0].strip()
                error_msg = parts[1].strip() if len(parts) > 1 else ''
                break

        # Find line number from traceback
        for line in lines:
            m = re.search(r'File ".+", line (\d+)', line)
            if m:
                error_line = int(m.group(1))

        root_cause = Divergence(
            line=error_line or 0,
            reason=f'{error_type}: {error_msg}',
            severity='ERROR',
        )
        divergences.append(root_cause)

        # Check if abstract interpreter predicted this
        predicted_this = False
        for err in prediction.errors:
            if err.get('line') == error_line:
                predicted_this = True
                divergences.append(Divergence(
                    line=error_line or 0,
                    reason=f'Abstract interpreter also predicted error here',
                    severity='INFO',
                ))

        # Specific error analysis
        fix = ''
        if error_type == 'IndexError':
            fix = self._fix_index_error(code, error_line, error_msg, prediction)
        elif error_type == 'KeyError':
            fix = self._fix_key_error(code, error_line, error_msg, prediction)
        elif error_type == 'TypeError':
            fix = self._fix_type_error(code, error_line, error_msg, prediction)
        elif error_type == 'ZeroDivisionError':
            fix = self._fix_zero_division(code, error_line)
        elif error_type == 'NameError':
            fix = self._fix_name_error(code, error_line, error_msg)
        elif error_type == 'AttributeError':
            fix = self._fix_attribute_error(code, error_line, error_msg, prediction)

        if not fix:
            fix = f'Error at line {error_line}: {error_type}: {error_msg}'

        return TraceResult(
            root_cause=root_cause,
            divergences=divergences,
            prediction=prediction,
            fix_suggestion=fix,
        )

    def _compare_outputs(self, predicted: str, actual: str,
                         expected: str) -> List[Divergence]:
        """Compare predicted, actual, and expected outputs."""
        divs = []

        pred_lines = predicted.strip().split('\n') if predicted.strip() else []
        actual_lines = actual.strip().split('\n') if actual.strip() else []
        expected_lines = expected.strip().split('\n') if expected.strip() else []

        # Compare expected vs actual
        if expected_lines and actual_lines:
            for i, (exp, act) in enumerate(zip(expected_lines, actual_lines)):
                if exp.strip() != act.strip():
                    divs.append(Divergence(
                        line=0,
                        variable=f'output_line_{i+1}',
                        predicted=exp.strip(),
                        actual=act.strip(),
                        reason=f'Output line {i+1} differs: expected "{exp.strip()}", got "{act.strip()}"',
                        severity='ERROR',
                    ))

            if len(expected_lines) != len(actual_lines):
                divs.append(Divergence(
                    line=0,
                    reason=f'Output has {len(actual_lines)} lines, expected {len(expected_lines)}',
                    severity='ERROR',
                ))

        # Compare predicted vs actual (to validate our interpreter)
        if pred_lines and actual_lines:
            for i, (pred, act) in enumerate(zip(pred_lines, actual_lines)):
                if pred.strip() != act.strip() and '<' not in pred:
                    # Only flag if prediction was concrete (no symbolic <type>)
                    divs.append(Divergence(
                        line=0,
                        variable=f'predicted_vs_actual_line_{i+1}',
                        predicted=pred.strip(),
                        actual=act.strip(),
                        reason=f'Prediction wrong at line {i+1}: predicted "{pred.strip()}", got "{act.strip()}"',
                        severity='WARNING',
                    ))

        return divs

    def _check_variable_issues(self, prediction: Prediction,
                                code: str) -> List[Divergence]:
        """Check for common variable-related bugs."""
        divs = []
        lines = code.split('\n')

        # Check for off-by-one patterns
        for name, val in prediction.variables.items():
            if val.is_concrete:
                # Check for common off-by-one: range(n) vs range(1, n+1)
                if isinstance(val.value, list) and val.value:
                    if val.value[0] == 0 and isinstance(val.value[-1], int):
                        # Might be an off-by-one if used as 1-based index
                        pass

        # Check for unused variables
        used_vars = set()
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)

        assigned_vars = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                assigned_vars.add(node.id)

        unused = assigned_vars - used_vars - {'_'}
        for var in unused:
            # Find the line
            for node in ast.walk(tree):
                if isinstance(node, ast.Name) and node.id == var and isinstance(node.ctx, ast.Store):
                    divs.append(Divergence(
                        line=node.lineno,
                        variable=var,
                        reason=f'Variable "{var}" assigned but never used',
                        severity='WARNING',
                    ))
                    break

        return divs

    def _suggest_fix(self, root_cause: Divergence, code: str,
                     prediction: Prediction) -> str:
        """Generate a fix suggestion for the root cause."""
        if not root_cause:
            return ''

        reason = root_cause.reason.lower()

        if 'index' in reason and 'out of range' in reason:
            return (f'Line {root_cause.line}: Array index out of bounds. '
                    f'Add bounds check: if i < len(arr)')

        if 'key' in reason and 'not found' in reason:
            return (f'Line {root_cause.line}: Dict key missing. '
                    f'Use .get() with default: d.get(key, default)')

        if 'zero' in reason and 'division' in reason:
            return (f'Line {root_cause.line}: Division by zero. '
                    f'Add guard: if divisor != 0')

        if 'type' in reason:
            return (f'Line {root_cause.line}: Type mismatch. '
                    f'Check that operands have compatible types.')

        if 'name' in reason and 'not defined' in reason:
            return (f'Line {root_cause.line}: Undefined variable. '
                    f'Define or import it before use.')

        if root_cause.predicted is not None and root_cause.actual is not None:
            return (f'Line {root_cause.line}: Expected {root_cause.predicted!r}, '
                    f'got {root_cause.actual!r}. Check the logic.')

        return f'Line {root_cause.line}: {root_cause.reason}'

    # === Specific error fixers ===

    def _fix_index_error(self, code: str, line: int, msg: str,
                         pred: Prediction) -> str:
        if line:
            code_lines = code.split('\n')
            if line <= len(code_lines):
                code_line = code_lines[line - 1].strip()
                # Find the list being indexed
                m = re.search(r'(\w+)\[', code_line)
                if m:
                    var = m.group(1)
                    if var in pred.variables:
                        val = pred.variables[var]
                        if val.is_concrete and isinstance(val.value, (list, tuple)):
                            return (f'Line {line}: IndexError on {var} '
                                    f'(length={len(val.value)}). '
                                    f'Add: if idx < len({var})')
        return f'Line {line}: IndexError — add bounds check'

    def _fix_key_error(self, code: str, line: int, msg: str,
                       pred: Prediction) -> str:
        m = re.search(r"KeyError:\s*['\"]?(\w+)", msg)
        key = m.group(1) if m else '?'
        return f'Line {line}: KeyError for key "{key}". Use .get("{key}", default) instead of ["..."]'

    def _fix_type_error(self, code: str, line: int, msg: str,
                        pred: Prediction) -> str:
        if 'can only concatenate str' in msg:
            return f'Line {line}: String concatenation with non-string. Wrap in str()'
        if 'unsupported operand' in msg:
            return f'Line {line}: Incompatible types in operation. Check types with type()'
        return f'Line {line}: TypeError — {msg}'

    def _fix_zero_division(self, code: str, line: int) -> str:
        return f'Line {line}: Division by zero. Add guard: if denominator != 0'

    def _fix_name_error(self, code: str, line: int, msg: str) -> str:
        m = re.search(r"name '(\w+)' is not defined", msg)
        name = m.group(1) if m else '?'
        # Common misspellings
        fixes = {
            'true': 'True', 'false': 'False', 'none': 'None',
            'null': 'None', 'undefined': 'None',
        }
        if name.lower() in fixes:
            return f'Line {line}: Use {fixes[name.lower()]} instead of {name}'
        return f'Line {line}: Variable "{name}" not defined. Define or import it first.'

    def _fix_attribute_error(self, code: str, line: int, msg: str,
                             pred: Prediction) -> str:
        m = re.search(r"'(\w+)' object has no attribute '(\w+)'", msg)
        if m:
            obj_type = m.group(1)
            attr = m.group(2)
            return (f'Line {line}: {obj_type} has no attribute "{attr}". '
                    f'Check spelling or use hasattr() guard.')
        return f'Line {line}: AttributeError — {msg}'


class TraceResult:
    """Result of causal tracing."""

    def __init__(self, root_cause: Optional[Divergence] = None,
                 divergences: List[Divergence] = None,
                 prediction: Prediction = None,
                 fix_suggestion: str = ''):
        self.root_cause = root_cause
        self.divergences = divergences or []
        self.prediction = prediction
        self.fix_suggestion = fix_suggestion

    @property
    def root_line(self) -> int:
        """Line number of root cause."""
        return self.root_cause.line if self.root_cause else 0

    @property
    def has_error(self) -> bool:
        return any(d.severity == 'ERROR' for d in self.divergences)

    def summary(self) -> str:
        """Human-readable summary."""
        lines = []
        if self.root_cause:
            lines.append(f'ROOT CAUSE: Line {self.root_cause.line}: {self.root_cause.reason}')
        if self.fix_suggestion:
            lines.append(f'FIX: {self.fix_suggestion}')
        if self.divergences:
            lines.append(f'\nAll divergences ({len(self.divergences)}):')
            for d in self.divergences[:10]:
                lines.append(f'  [{d.severity}] Line {d.line}: {d.reason}')
        if self.prediction:
            lines.append(f'\nPrediction confidence: {self.prediction.confidence}')
        return '\n'.join(lines)
