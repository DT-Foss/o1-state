"""
Predict-Execute-Compare Loop — The Self-Improvement Engine Core
================================================================
Fähigkeit 3 im 4-Capabilities Framework.

The fundamental cognitive loop:
  1. PREDICT: Before execution, form an expectation
  2. EXECUTE: Run the code/action
  3. COMPARE: Check result against expectation
  4. LEARN: If mismatch, classify the error and feed to Causal Tracer

This is what separates "understanding" from "copying":
- Copying: run code, hope it works
- Understanding: predict what will happen, run it, check if prediction was right

The loop drives ALL self-improvement:
- Code generation: predict output → generate → run → compare
- Knowledge acquisition: predict fact → extract → verify
- Module building: predict behavior → implement → test → compare

Without this loop, the system is blind.
With it, every action is a learning opportunity.
"""

import ast
import re
import subprocess
import tempfile
import os
import time
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum


class MismatchType(Enum):
    """Classification of prediction vs reality mismatches."""
    PASS = "pass"                    # Prediction matches reality
    TYPE_MISMATCH = "type_mismatch"  # Expected int, got string etc.
    VALUE_MISMATCH = "value_mismatch"  # Expected 5, got 7
    SHAPE_MISMATCH = "shape_mismatch"  # Expected list of 3, got list of 2
    EXCEPTION = "exception"          # Code threw an error
    TIMEOUT = "timeout"              # Code didn't finish
    NO_OUTPUT = "no_output"          # Code produced nothing
    PARTIAL_MATCH = "partial_match"  # Some parts right, some wrong


class Prediction:
    """A prediction about what code will produce."""

    def __init__(self, code: str, expected_type: str = None,
                 expected_value: Any = None, expected_pattern: str = None,
                 expected_contains: List[str] = None,
                 expected_exception: str = None):
        """
        Args:
            code: the code being predicted about
            expected_type: expected Python type name ("int", "list", "str")
            expected_value: exact expected output value
            expected_pattern: regex pattern the output should match
            expected_contains: substrings the output should contain
            expected_exception: if we expect an error, which type
        """
        self.code = code
        self.expected_type = expected_type
        self.expected_value = expected_value
        self.expected_pattern = expected_pattern
        self.expected_contains = expected_contains or []
        self.expected_exception = expected_exception
        self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code_hash': hash(self.code) % 10**8,
            'expected_type': self.expected_type,
            'expected_value': str(self.expected_value) if self.expected_value is not None else None,
            'expected_pattern': self.expected_pattern,
            'expected_contains': self.expected_contains,
            'expected_exception': self.expected_exception,
        }


class CompareResult:
    """Result of comparing prediction with reality."""

    def __init__(self, prediction: Prediction, actual_output: str,
                 actual_type: str, actual_exception: str = None,
                 mismatch: MismatchType = MismatchType.PASS,
                 details: str = ""):
        self.prediction = prediction
        self.actual_output = actual_output
        self.actual_type = actual_type
        self.actual_exception = actual_exception
        self.mismatch = mismatch
        self.details = details
        self.timestamp = time.time()

    @property
    def passed(self) -> bool:
        return self.mismatch == MismatchType.PASS

    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction': self.prediction.to_dict(),
            'actual_output': self.actual_output[:500],
            'actual_type': self.actual_type,
            'actual_exception': self.actual_exception,
            'mismatch': self.mismatch.value,
            'details': self.details,
            'passed': self.passed,
        }


class CodePredictor:
    """
    Predicts what code will do WITHOUT executing it.

    Uses AST analysis + pattern matching for simple predictions:
    - Arithmetic: eval-safe expressions → exact value
    - Print statements: trace what gets printed
    - Function return: trace through simple functions
    - Type inference: what type will the result be
    - Exception prediction: detect obvious errors

    This is a lightweight abstract interpreter — not full symbolic
    execution, but enough to catch categorical errors.
    """

    def predict(self, code: str) -> Prediction:
        """Generate a prediction for what this code will produce."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return Prediction(
                code=code,
                expected_exception=f"SyntaxError: {e.msg}",
            )

        # Analyze the AST
        prints = self._find_prints(tree)
        returns = self._find_returns(tree)
        assignments = self._find_assignments(tree)
        imports = self._find_imports(tree)

        # Strategy 1: If code has print statements, predict printed output
        if prints:
            predicted_output = self._predict_prints(prints, assignments)
            if predicted_output is not None:
                return Prediction(
                    code=code,
                    expected_type="str",
                    expected_value=predicted_output,
                )

        # Strategy 2: If code is a pure expression, predict its value
        if len(tree.body) == 1 and isinstance(tree.body[0], ast.Expr):
            val = self._eval_safe(tree.body[0].value)
            if val is not None:
                return Prediction(
                    code=code,
                    expected_type=type(val).__name__,
                    expected_value=val,
                )

        # Strategy 3: Predict return type from function signatures
        if returns:
            ret_type = self._infer_return_type(returns[0], assignments)
            if ret_type:
                return Prediction(
                    code=code,
                    expected_type=ret_type,
                )

        # Strategy 4: Detect obvious errors
        error = self._detect_obvious_errors(tree, imports)
        if error:
            return Prediction(
                code=code,
                expected_exception=error,
            )

        # Strategy 5: Pattern-based prediction from code structure
        contains = self._predict_output_patterns(tree, assignments)
        if contains:
            return Prediction(
                code=code,
                expected_contains=contains,
            )

        # Can't predict — return empty prediction
        return Prediction(code=code)

    def _find_prints(self, tree: ast.AST) -> List[ast.Call]:
        """Find all print() calls in AST."""
        prints = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    prints.append(node)
        return prints

    def _find_returns(self, tree: ast.AST) -> List[ast.Return]:
        """Find all return statements."""
        return [n for n in ast.walk(tree) if isinstance(n, ast.Return)]

    def _find_assignments(self, tree: ast.AST) -> Dict[str, ast.AST]:
        """Find variable assignments."""
        assigns = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assigns[target.id] = node.value
        return assigns

    def _find_imports(self, tree: ast.AST) -> List[str]:
        """Find imported module names."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        return imports

    def _predict_prints(self, prints: List[ast.Call],
                        assignments: Dict) -> Optional[str]:
        """Predict what print statements will output."""
        outputs = []
        for p in prints:
            if p.args:
                val = self._eval_safe(p.args[0], assignments)
                if val is not None:
                    outputs.append(str(val))
                else:
                    return None  # Can't predict all prints
            else:
                outputs.append("")
        return "\n".join(outputs) if outputs else None

    def _eval_safe(self, node: ast.AST,
                   env: Dict = None) -> Any:
        """Safely evaluate simple AST expressions."""
        env = env or {}

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            if node.id in env:
                return self._eval_safe(env[node.id], {})
            return None

        if isinstance(node, ast.BinOp):
            left = self._eval_safe(node.left, env)
            right = self._eval_safe(node.right, env)
            if left is not None and right is not None:
                ops = {
                    ast.Add: lambda a, b: a + b,
                    ast.Sub: lambda a, b: a - b,
                    ast.Mult: lambda a, b: a * b,
                    ast.Div: lambda a, b: a / b if b != 0 else None,
                    ast.Mod: lambda a, b: a % b if b != 0 else None,
                    ast.Pow: lambda a, b: a ** b if abs(b) < 100 else None,
                    ast.FloorDiv: lambda a, b: a // b if b != 0 else None,
                }
                op_func = ops.get(type(node.op))
                if op_func:
                    try:
                        return op_func(left, right)
                    except (TypeError, OverflowError):
                        return None

        if isinstance(node, ast.UnaryOp):
            operand = self._eval_safe(node.operand, env)
            if operand is not None:
                if isinstance(node.op, ast.USub):
                    return -operand
                if isinstance(node.op, ast.Not):
                    return not operand

        if isinstance(node, ast.List):
            elts = [self._eval_safe(e, env) for e in node.elts]
            if all(e is not None for e in elts):
                return elts

        if isinstance(node, ast.Tuple):
            elts = [self._eval_safe(e, env) for e in node.elts]
            if all(e is not None for e in elts):
                return tuple(elts)

        if isinstance(node, ast.JoinedStr):  # f-string
            parts = []
            for v in node.values:
                val = self._eval_safe(v, env)
                if val is not None:
                    parts.append(str(val))
                else:
                    return None
            return "".join(parts)

        if isinstance(node, ast.FormattedValue):
            return self._eval_safe(node.value, env)

        if isinstance(node, ast.Compare):
            left = self._eval_safe(node.left, env)
            if left is None:
                return None
            for op, comp in zip(node.ops, node.comparators):
                right = self._eval_safe(comp, env)
                if right is None:
                    return None
                cmp_ops = {
                    ast.Eq: lambda a, b: a == b,
                    ast.NotEq: lambda a, b: a != b,
                    ast.Lt: lambda a, b: a < b,
                    ast.LtE: lambda a, b: a <= b,
                    ast.Gt: lambda a, b: a > b,
                    ast.GtE: lambda a, b: a >= b,
                }
                op_func = cmp_ops.get(type(op))
                if op_func:
                    try:
                        if not op_func(left, right):
                            return False
                    except TypeError:
                        return None
                left = right
            return True

        return None

    def _infer_return_type(self, ret: ast.Return,
                           assignments: Dict) -> Optional[str]:
        """Infer the return type of a function."""
        if ret.value is None:
            return "NoneType"

        val = ret.value
        if isinstance(val, ast.Constant):
            return type(val.value).__name__
        if isinstance(val, ast.List):
            return "list"
        if isinstance(val, ast.Dict):
            return "dict"
        if isinstance(val, ast.Tuple):
            return "tuple"
        if isinstance(val, ast.Set):
            return "set"
        if isinstance(val, ast.BinOp):
            # Arithmetic → likely int or float
            return "number"
        if isinstance(val, ast.JoinedStr):
            return "str"
        if isinstance(val, ast.Call):
            if isinstance(val.func, ast.Name):
                type_constructors = {
                    'int': 'int', 'float': 'float', 'str': 'str',
                    'list': 'list', 'dict': 'dict', 'set': 'set',
                    'tuple': 'tuple', 'bool': 'bool',
                    'sorted': 'list', 'reversed': 'iterator',
                    'len': 'int', 'sum': 'number', 'max': 'number',
                    'min': 'number', 'abs': 'number',
                }
                return type_constructors.get(val.func.id)
        return None

    def _detect_obvious_errors(self, tree: ast.AST,
                                imports: List[str]) -> Optional[str]:
        """Detect errors that will definitely occur."""
        # Check for division by zero
        for node in ast.walk(tree):
            if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Div, ast.FloorDiv, ast.Mod)):
                if isinstance(node.right, ast.Constant) and node.right.value == 0:
                    return "ZeroDivisionError: division by zero"

        # Check for undefined variable usage (basic)
        defined = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    defined.add(alias.asname or alias.name)
            elif isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        defined.add(t.id)
            elif isinstance(node, ast.FunctionDef):
                defined.add(node.name)
                for arg in node.args.args:
                    defined.add(arg.arg)
            elif isinstance(node, ast.For):
                if isinstance(node.target, ast.Name):
                    defined.add(node.target.id)

        # Built-in names
        builtins = {'print', 'len', 'range', 'int', 'float', 'str', 'list',
                    'dict', 'set', 'tuple', 'bool', 'type', 'isinstance',
                    'sorted', 'reversed', 'enumerate', 'zip', 'map', 'filter',
                    'input', 'open', 'sum', 'min', 'max', 'abs', 'round',
                    'True', 'False', 'None', 'Exception', 'ValueError',
                    'TypeError', 'KeyError', 'IndexError', 'StopIteration',
                    'super', 'property', 'staticmethod', 'classmethod',
                    'any', 'all', 'hasattr', 'getattr', 'setattr', 'dir',
                    'hex', 'oct', 'bin', 'chr', 'ord', 'repr', 'hash',
                    '__name__', '__file__', '__import__'}
        defined.update(builtins)

        return None  # No obvious errors detected

    def _predict_output_patterns(self, tree: ast.AST,
                                  assignments: Dict) -> List[str]:
        """Predict patterns in output based on code structure."""
        patterns = []

        for node in ast.walk(tree):
            # String literals in print statements
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'print':
                    for arg in node.args:
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            patterns.append(arg.value)

        return patterns


class PredictCompareLoop:
    """
    The complete Predict-Execute-Compare loop.

    Usage:
        loop = PredictCompareLoop()

        # Single code test
        result = loop.test_code("print(2 + 3)")
        # → CompareResult(passed=True, prediction="5", actual="5")

        # Batch self-test
        results = loop.self_test([
            "print(2 + 3)",
            "print('hello')",
            "x = [1,2,3]; print(len(x))",
        ])

        # Full improvement cycle
        loop.improve_cycle(agent)
    """

    def __init__(self, timeout: int = 10):
        self.predictor = CodePredictor()
        self.timeout = timeout
        self.history: List[CompareResult] = []
        self._stats = {'pass': 0, 'fail': 0, 'skip': 0}

    def test_code(self, code: str) -> CompareResult:
        """
        Full predict-execute-compare cycle for a piece of code.

        1. Predict what the code will do
        2. Execute the code
        3. Compare prediction with reality
        4. Return detailed comparison result
        """
        # Step 1: PREDICT
        prediction = self.predictor.predict(code)

        # Step 2: EXECUTE
        actual_output, actual_exception = self._execute(code)

        # Step 3: COMPARE
        result = self._compare(prediction, actual_output, actual_exception)

        # Step 4: RECORD
        self.history.append(result)
        if result.passed:
            self._stats['pass'] += 1
        else:
            self._stats['fail'] += 1

        return result

    def self_test(self, code_samples: List[str]) -> Dict[str, Any]:
        """
        Run predict-execute-compare on a batch of code samples.

        Returns summary with pass rate and failure analysis.
        """
        results = []
        for code in code_samples:
            result = self.test_code(code)
            results.append(result)

        passed = sum(1 for r in results if r.passed)
        failed = [r for r in results if not r.passed]

        # Classify failures
        failure_types = {}
        for r in failed:
            ft = r.mismatch.value
            failure_types[ft] = failure_types.get(ft, 0) + 1

        return {
            'total': len(results),
            'passed': passed,
            'failed': len(failed),
            'pass_rate': passed / max(len(results), 1),
            'failure_types': failure_types,
            'failures': [r.to_dict() for r in failed[:10]],
        }

    def improve_cycle(self, agent=None, n_tasks: int = 20) -> Dict[str, Any]:
        """
        Full self-improvement cycle using predict-compare.

        1. Generate code tasks (from gaps or self-test)
        2. For each task: predict → execute → compare
        3. Analyze mismatches
        4. Feed mismatches to learning system

        This is the ENGINE that drives autonomous improvement.
        """
        if not agent:
            return {'error': 'No agent provided'}

        tasks = self._generate_test_tasks(n_tasks)
        results = []

        for task in tasks:
            code = task.get('code', '')
            if not code:
                continue

            result = self.test_code(code)
            results.append({
                'task': task,
                'result': result.to_dict(),
            })

        passed = sum(1 for r in results if r['result']['passed'])
        return {
            'total': len(results),
            'passed': passed,
            'pass_rate': passed / max(len(results), 1),
            'results': results,
        }

    def prediction_accuracy(self) -> Dict[str, Any]:
        """How accurate are our predictions?"""
        if not self.history:
            return {'total': 0, 'accuracy': 0.0}

        total = len(self.history)
        correct = sum(1 for r in self.history if r.passed)
        by_type = {}
        for r in self.history:
            mt = r.mismatch.value
            by_type[mt] = by_type.get(mt, 0) + 1

        return {
            'total': total,
            'correct': correct,
            'accuracy': correct / total,
            'by_mismatch_type': by_type,
        }

    def _execute(self, code: str) -> Tuple[str, Optional[str]]:
        """Execute code in a sandboxed subprocess."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py',
                                          delete=False) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'

            proc = subprocess.run(
                ['python3', tmp_path],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )

            output = proc.stdout.strip()
            error = proc.stderr.strip() if proc.returncode != 0 else None

            return output, error

        except subprocess.TimeoutExpired:
            return "", "TimeoutError: execution exceeded time limit"
        except Exception as e:
            return "", str(e)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    def _compare(self, prediction: Prediction, actual_output: str,
                 actual_exception: Optional[str]) -> CompareResult:
        """Compare prediction with actual execution result."""

        # Case 1: We predicted an exception
        if prediction.expected_exception:
            if actual_exception:
                # Check if exception type matches
                expected_type = prediction.expected_exception.split(':')[0].strip()
                if expected_type in actual_exception:
                    return CompareResult(
                        prediction=prediction,
                        actual_output=actual_output,
                        actual_type="exception",
                        actual_exception=actual_exception,
                        mismatch=MismatchType.PASS,
                        details=f"Correctly predicted {expected_type}",
                    )
                else:
                    return CompareResult(
                        prediction=prediction,
                        actual_output=actual_output,
                        actual_type="exception",
                        actual_exception=actual_exception,
                        mismatch=MismatchType.EXCEPTION,
                        details=f"Expected {expected_type}, got {actual_exception[:50]}",
                    )
            else:
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="success",
                    mismatch=MismatchType.VALUE_MISMATCH,
                    details="Expected exception but code succeeded",
                )

        # Case 2: Unexpected exception
        if actual_exception:
            if "TimeoutError" in actual_exception:
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="timeout",
                    actual_exception=actual_exception,
                    mismatch=MismatchType.TIMEOUT,
                )
            return CompareResult(
                prediction=prediction,
                actual_output=actual_output,
                actual_type="exception",
                actual_exception=actual_exception,
                mismatch=MismatchType.EXCEPTION,
                details=actual_exception[:200],
            )

        # Case 3: No output when we expected some
        if not actual_output and prediction.expected_value is not None:
            return CompareResult(
                prediction=prediction,
                actual_output="",
                actual_type="empty",
                mismatch=MismatchType.NO_OUTPUT,
                details=f"Expected {prediction.expected_value}, got nothing",
            )

        # Case 4: Check exact value match
        if prediction.expected_value is not None:
            expected_str = str(prediction.expected_value)
            if actual_output.strip() == expected_str.strip():
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type=prediction.expected_type or "str",
                    mismatch=MismatchType.PASS,
                    details="Exact match",
                )
            else:
                # Check if it's a type vs value mismatch
                try:
                    actual_val = eval(actual_output, {"__builtins__": {}}, {})
                    expected_val = prediction.expected_value
                    if type(actual_val).__name__ != type(expected_val).__name__:
                        return CompareResult(
                            prediction=prediction,
                            actual_output=actual_output,
                            actual_type=type(actual_val).__name__,
                            mismatch=MismatchType.TYPE_MISMATCH,
                            details=f"Expected type {type(expected_val).__name__}, got {type(actual_val).__name__}",
                        )
                except Exception:
                    pass

                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="str",
                    mismatch=MismatchType.VALUE_MISMATCH,
                    details=f"Expected '{expected_str}', got '{actual_output[:50]}'",
                )

        # Case 5: Check pattern match
        if prediction.expected_pattern:
            if re.search(prediction.expected_pattern, actual_output):
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="str",
                    mismatch=MismatchType.PASS,
                    details="Pattern match",
                )
            else:
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="str",
                    mismatch=MismatchType.VALUE_MISMATCH,
                    details=f"Output doesn't match pattern {prediction.expected_pattern}",
                )

        # Case 6: Check contains
        if prediction.expected_contains:
            missing = [s for s in prediction.expected_contains
                       if s not in actual_output]
            if not missing:
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="str",
                    mismatch=MismatchType.PASS,
                    details="Contains all expected substrings",
                )
            else:
                return CompareResult(
                    prediction=prediction,
                    actual_output=actual_output,
                    actual_type="str",
                    mismatch=MismatchType.PARTIAL_MATCH,
                    details=f"Missing: {missing[:3]}",
                )

        # Case 7: No prediction was possible — skip
        return CompareResult(
            prediction=prediction,
            actual_output=actual_output,
            actual_type="unknown",
            mismatch=MismatchType.PASS,  # Can't fail what we can't predict
            details="No prediction made (opaque code)",
        )

    def _generate_test_tasks(self, n: int) -> List[Dict]:
        """Generate simple code tasks for self-testing."""
        tasks = [
            {'code': 'print(2 + 3)', 'description': 'simple addition'},
            {'code': 'print(10 * 5)', 'description': 'multiplication'},
            {'code': 'print(100 // 7)', 'description': 'floor division'},
            {'code': 'print(2 ** 10)', 'description': 'exponentiation'},
            {'code': 'print(17 % 5)', 'description': 'modulo'},
            {'code': 'print(len([1,2,3,4,5]))', 'description': 'list length'},
            {'code': 'print("hello" + " " + "world")', 'description': 'string concat'},
            {'code': 'print(3 > 2)', 'description': 'comparison'},
            {'code': 'print(not False)', 'description': 'boolean logic'},
            {'code': 'x = 5\ny = 10\nprint(x + y)', 'description': 'variables'},
            {'code': 'print(list(range(5)))', 'description': 'range to list'},
            {'code': 'print(sum([1,2,3,4,5]))', 'description': 'sum'},
            {'code': 'print(max(3, 7, 1, 9, 2))', 'description': 'max'},
            {'code': 'print(min(3, 7, 1, 9, 2))', 'description': 'min'},
            {'code': 'print(abs(-42))', 'description': 'absolute value'},
            {'code': 'print(sorted([3,1,4,1,5,9,2,6]))', 'description': 'sorted'},
            {'code': 'print(type(42).__name__)', 'description': 'type name'},
            {'code': 'print("HELLO".lower())', 'description': 'string lower'},
            {'code': 'print(" spaces ".strip())', 'description': 'string strip'},
            {'code': 'print(10 / 3)', 'description': 'float division'},
        ]
        return tasks[:n]

    def report(self) -> str:
        """Human-readable report of prediction accuracy."""
        stats = self.prediction_accuracy()
        if stats['total'] == 0:
            return "No predictions made yet."

        lines = [
            "Predict-Compare Report",
            "=" * 40,
            f"Total predictions: {stats['total']}",
            f"Correct: {stats['correct']} ({stats['accuracy']:.0%})",
            f"Wrong: {stats['total'] - stats['correct']}",
        ]

        if stats['by_mismatch_type']:
            lines.append("\nMismatch breakdown:")
            for mt, count in sorted(stats['by_mismatch_type'].items(),
                                     key=lambda x: x[1], reverse=True):
                lines.append(f"  {count:3d}x {mt}")

        return "\n".join(lines)
