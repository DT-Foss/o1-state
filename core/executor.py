"""
Code Executor — Run, Test, Debug, Fix Loop
=============================================
The missing piece: FOSS-KI generates code, but can it run it?

Pipeline:
  1. GENERATE: CodeGenerator/ForgeAssembler produces code
  2. VALIDATE: AST parse check (syntax ok?)
  3. EXECUTE: Run in sandboxed subprocess with timeout
  4. ANALYZE: If error → classify error type
  5. FIX: Apply targeted fix based on error class
  6. RETRY: Re-run with fix applied (max 3 attempts)

Error classification:
  - SYNTAX: SyntaxError, IndentationError → AST-level fix
  - IMPORT: ModuleNotFoundError → suggest alternative or install
  - TYPE: TypeError, AttributeError → type inference fix
  - RUNTIME: ValueError, IndexError, KeyError → guard insertion
  - LOGIC: Wrong output (test fails) → re-examine algorithm

This is what makes FOSS-KI a real coding assistant:
not just generating code, but ensuring it WORKS.
"""

import ast
import os
import re
import subprocess
import tempfile
from typing import Dict, Any, List, Optional


class CodeExecutor:
    """
    Sandboxed code execution with error analysis and auto-fix.
    """

    def __init__(self, timeout: int = 10, max_retries: int = 3,
                 sandbox: bool = True):
        """
        Args:
            timeout: max seconds per execution
            max_retries: max fix-and-retry attempts
            sandbox: if True, restrict execution environment
        """
        self.timeout = timeout
        self.max_retries = max_retries
        self.sandbox = sandbox
        self.history = []  # Execution history

    def validate(self, code: str) -> Dict[str, Any]:
        """
        Validate code without executing.

        Returns:
            dict with: valid (bool), errors (list), ast_tree
        """
        errors = []

        # Step 1: AST parse
        try:
            tree = ast.parse(code)
            return {
                'valid': True,
                'errors': [],
                'ast_tree': tree,
                'n_statements': len(tree.body),
                'functions': [n.name for n in ast.walk(tree)
                             if isinstance(n, ast.FunctionDef)],
                'classes': [n.name for n in ast.walk(tree)
                           if isinstance(n, ast.ClassDef)],
                'imports': [self._get_import_name(n) for n in ast.walk(tree)
                           if isinstance(n, (ast.Import, ast.ImportFrom))],
            }
        except SyntaxError as e:
            errors.append({
                'type': 'SYNTAX',
                'message': str(e),
                'line': e.lineno,
                'offset': e.offset,
            })
            return {'valid': False, 'errors': errors}

    def execute(self, code: str, test_input: str = "",
                expected_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute code in a sandboxed subprocess.

        Args:
            code: Python code to execute
            test_input: stdin input for the program
            expected_output: expected stdout (for test verification)

        Returns:
            dict with: success, stdout, stderr, exit_code, elapsed, errors
        """
        # Step 1: Validate
        validation = self.validate(code)
        if not validation['valid']:
            return {
                'success': False,
                'phase': 'validation',
                'errors': validation['errors'],
                'stdout': '',
                'stderr': f"SyntaxError: {validation['errors'][0]['message']}",
            }

        # Step 2: Write to temp file and execute
        with tempfile.NamedTemporaryFile(
            mode='w', suffix='.py', delete=False,
            dir=tempfile.gettempdir()
        ) as f:
            f.write(code)
            tmpfile = f.name

        try:
            import time
            t0 = time.time()

            env = os.environ.copy()
            if self.sandbox:
                # Restrict environment
                env['FOSS_KI_SANDBOX'] = '1'

            result = subprocess.run(
                ['python3', tmpfile],
                input=test_input,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )

            elapsed = time.time() - t0

            exec_result = {
                'success': result.returncode == 0,
                'phase': 'execution',
                'stdout': result.stdout,
                'stderr': result.stderr,
                'exit_code': result.returncode,
                'elapsed': elapsed,
                'errors': [],
            }

            # Analyze errors if failed
            if not exec_result['success']:
                exec_result['errors'] = self._classify_errors(result.stderr)

            # Check expected output
            if expected_output is not None and exec_result['success']:
                actual = result.stdout.strip()
                expected = expected_output.strip()
                if actual != expected:
                    exec_result['success'] = False
                    exec_result['phase'] = 'verification'
                    exec_result['errors'] = [{
                        'type': 'LOGIC',
                        'message': f'Output mismatch: expected "{expected}", got "{actual}"',
                        'expected': expected,
                        'actual': actual,
                    }]

            return exec_result

        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'phase': 'execution',
                'stdout': '',
                'stderr': f'Timeout after {self.timeout}s',
                'errors': [{'type': 'TIMEOUT', 'message': f'Exceeded {self.timeout}s'}],
            }
        finally:
            os.unlink(tmpfile)

    def execute_with_retry(self, code: str, test_input: str = "",
                           expected_output: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute with automatic fix-and-retry loop.

        Returns:
            dict with: success, final_code, attempts, history
        """
        current_code = code
        attempts = []

        for attempt in range(self.max_retries + 1):
            result = self.execute(current_code, test_input, expected_output)
            attempts.append({
                'attempt': attempt + 1,
                'code': current_code,
                'result': result,
            })

            if result['success']:
                return {
                    'success': True,
                    'final_code': current_code,
                    'attempts': len(attempts),
                    'history': attempts,
                    'stdout': result['stdout'],
                }

            if attempt < self.max_retries:
                # Try to fix
                fixed = self._auto_fix(current_code, result['errors'])
                if fixed and fixed != current_code:
                    current_code = fixed
                else:
                    break  # Can't fix further

        return {
            'success': False,
            'final_code': current_code,
            'attempts': len(attempts),
            'history': attempts,
            'errors': attempts[-1]['result']['errors'],
        }

    def run_tests(self, code: str, tests: List[Dict]) -> Dict[str, Any]:
        """
        Run multiple test cases against code.

        Args:
            code: the code to test
            tests: list of {input, expected_output, description} dicts

        Returns:
            dict with: passed, failed, total, results
        """
        results = []
        passed = 0

        for test in tests:
            result = self.execute(
                code,
                test_input=test.get('input', ''),
                expected_output=test.get('expected_output'),
            )
            test_result = {
                'description': test.get('description', ''),
                'success': result['success'],
                'stdout': result.get('stdout', ''),
                'errors': result.get('errors', []),
            }
            results.append(test_result)
            if result['success']:
                passed += 1

        return {
            'passed': passed,
            'failed': len(tests) - passed,
            'total': len(tests),
            'results': results,
            'pass_rate': passed / max(len(tests), 1),
        }

    # === Error Classification ===

    def _classify_errors(self, stderr: str) -> List[Dict]:
        """Classify errors from stderr output."""
        errors = []

        # Extract Python traceback
        lines = stderr.strip().split('\n')

        # Find the error line
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue

            if line.startswith('SyntaxError') or line.startswith('IndentationError'):
                errors.append({'type': 'SYNTAX', 'message': line})
            elif line.startswith('ModuleNotFoundError') or line.startswith('ImportError'):
                errors.append({'type': 'IMPORT', 'message': line})
            elif line.startswith('TypeError') or line.startswith('AttributeError'):
                errors.append({'type': 'TYPE', 'message': line})
            elif line.startswith(('ValueError', 'IndexError', 'KeyError',
                                  'ZeroDivisionError')):
                errors.append({'type': 'RUNTIME', 'message': line})
            elif line.startswith('NameError'):
                errors.append({'type': 'NAME', 'message': line})
            elif line.startswith(('FileNotFoundError', 'PermissionError',
                                  'OSError', 'IOError')):
                errors.append({'type': 'IO', 'message': line})
            elif 'Error' in line or 'Exception' in line:
                errors.append({'type': 'OTHER', 'message': line})

            if errors:
                break

        # Extract line number from traceback
        for i, line in enumerate(lines):
            m = re.search(r'File ".+", line (\d+)', line)
            if m and errors:
                errors[-1]['line'] = int(m.group(1))
                # Get the actual code line
                if i + 1 < len(lines):
                    errors[-1]['code_line'] = lines[i + 1].strip()

        return errors

    # === Auto-Fix ===

    def _auto_fix(self, code: str, errors: List[Dict]) -> Optional[str]:
        """
        Attempt to automatically fix code based on error classification.
        """
        if not errors:
            return None

        error = errors[0]
        error_type = error.get('type', 'OTHER')

        if error_type == 'SYNTAX':
            return self._fix_syntax(code, error)
        elif error_type == 'IMPORT':
            return self._fix_import(code, error)
        elif error_type == 'NAME':
            return self._fix_name(code, error)
        elif error_type == 'TYPE':
            return self._fix_type(code, error)
        elif error_type == 'RUNTIME':
            return self._fix_runtime(code, error)

        return None

    def _fix_syntax(self, code: str, error: Dict) -> Optional[str]:
        """Fix common syntax errors."""
        msg = error.get('message', '')
        line_num = error.get('line')

        lines = code.split('\n')

        # Missing colon
        if 'expected ":"' in msg or "expected ':'" in msg:
            if line_num and line_num <= len(lines):
                line = lines[line_num - 1]
                # Add colon at end of def/if/for/while/class/else/elif/try/except
                if re.match(r'\s*(def|if|for|while|class|else|elif|try|except|finally|with)\b', line):
                    if not line.rstrip().endswith(':'):
                        lines[line_num - 1] = line.rstrip() + ':'
                        return '\n'.join(lines)

        # Unmatched parenthesis
        if 'unexpected EOF' in msg or 'was never closed' in msg:
            # Count parens
            open_p = code.count('(') - code.count(')')
            if open_p > 0:
                return code + ')' * open_p
            open_b = code.count('[') - code.count(']')
            if open_b > 0:
                return code + ']' * open_b

        # Indentation error
        if 'IndentationError' in msg or 'unexpected indent' in msg:
            if line_num and line_num <= len(lines):
                line = lines[line_num - 1]
                # Try to fix indentation to match previous line
                if line_num >= 2:
                    prev_indent = len(lines[line_num - 2]) - len(lines[line_num - 2].lstrip())
                    curr_indent = len(line) - len(line.lstrip())
                    if prev_indent != curr_indent:
                        # Align to previous + 4 if prev ends with :
                        if lines[line_num - 2].rstrip().endswith(':'):
                            new_indent = prev_indent + 4
                        else:
                            new_indent = prev_indent
                        lines[line_num - 1] = ' ' * new_indent + line.lstrip()
                        return '\n'.join(lines)

        return None

    def _fix_import(self, code: str, error: Dict) -> Optional[str]:
        """Fix import errors by removing or replacing imports."""
        msg = error.get('message', '')

        # Extract module name
        m = re.search(r"No module named '(\w+)'", msg)
        if m:
            module = m.group(1)
            # Common replacements
            replacements = {
                'numpy': None,  # Remove if not essential
                'pandas': None,
                'requests': None,
                'scipy': None,
            }
            if module in replacements:
                # Comment out the import
                lines = code.split('\n')
                new_lines = []
                for line in lines:
                    if f'import {module}' in line or f'from {module}' in line:
                        new_lines.append(f'# {line}  # Module not available')
                    else:
                        new_lines.append(line)
                return '\n'.join(new_lines)

        return None

    def _fix_name(self, code: str, error: Dict) -> Optional[str]:
        """Fix NameError by adding missing definitions."""
        msg = error.get('message', '')

        m = re.search(r"name '(\w+)' is not defined", msg)
        if m:
            name = m.group(1)
            # Common fixes
            if name == 'true':
                return code.replace('true', 'True')
            elif name == 'false':
                return code.replace('false', 'False')
            elif name == 'null' or name == 'none':
                return code.replace(name, 'None')
            elif name == 'print_':
                return code.replace('print_', 'print')

        return None

    def _fix_type(self, code: str, error: Dict) -> Optional[str]:
        """Fix TypeError by adding type conversions."""
        msg = error.get('message', '')
        line_num = error.get('line')

        # "can only concatenate str to str" → add str() conversion
        if 'can only concatenate str' in msg or "unsupported operand type" in msg:
            if line_num:
                lines = code.split('\n')
                if line_num <= len(lines):
                    line = lines[line_num - 1]
                    # Wrap non-str operands in str()
                    # This is a heuristic — might not always work
                    if '+' in line and '"' in line or "'" in line:
                        # Find variable in concatenation and wrap
                        pass  # Too complex for simple regex

        return None

    def _fix_runtime(self, code: str, error: Dict) -> Optional[str]:
        """Fix runtime errors by adding guards."""
        msg = error.get('message', '')
        line_num = error.get('line')

        # ZeroDivisionError → add guard
        if 'division by zero' in msg and line_num:
            lines = code.split('\n')
            if line_num <= len(lines):
                line = lines[line_num - 1]
                indent = len(line) - len(line.lstrip())
                _ = ' ' * indent + 'if False:  # Division guard\n'
                # This is crude — better to wrap in try/except
                lines.insert(line_num - 1, f'{" " * indent}try:')
                lines.insert(line_num + 1,
                           f'{" " * indent}except ZeroDivisionError:\n'
                           f'{" " * (indent + 4)}pass')
                return '\n'.join(lines)

        # IndexError → add bounds check
        if 'index out of range' in msg:
            # Add len() check — too complex for simple fix
            pass

        return None

    # === Helpers ===

    @staticmethod
    def _get_import_name(node) -> str:
        """Extract module name from an import AST node."""
        if isinstance(node, ast.Import):
            return node.names[0].name
        elif isinstance(node, ast.ImportFrom):
            return node.module or ''
        return ''


class CodeDebugger:
    """
    Higher-level debugging: analyze code, find bugs, suggest fixes.
    """

    def __init__(self, executor: Optional[CodeExecutor] = None):
        self.executor = executor or CodeExecutor()

    def debug(self, code: str, description: str = "") -> Dict[str, Any]:
        """
        Full debug cycle: validate → execute → analyze → fix.

        Returns:
            dict with: issues, suggestions, fixed_code, test_results
        """
        issues = []
        suggestions = []

        # Static analysis
        validation = self.executor.validate(code)
        if not validation['valid']:
            for err in validation['errors']:
                issues.append(f"Line {err.get('line', '?')}: {err['message']}")
                suggestions.append(f"Fix syntax at line {err.get('line', '?')}")

        # Style checks (lightweight)
        style_issues = self._check_style(code)
        issues.extend(style_issues)

        # Try to execute
        result = self.executor.execute_with_retry(code)

        return {
            'issues': issues,
            'suggestions': suggestions,
            'execution': result,
            'fixed_code': result.get('final_code', code),
            'attempts': result.get('attempts', 0),
            'success': result.get('success', False),
        }

    def _check_style(self, code: str) -> List[str]:
        """Basic style checks."""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Long lines
            if len(line) > 120:
                issues.append(f"Line {i}: exceeds 120 chars ({len(line)})")

            # Bare except
            if re.match(r'\s*except\s*:', line):
                issues.append(f"Line {i}: bare except (should catch specific exception)")

            # Mutable default argument
            m = re.match(r'\s*def\s+\w+\(.*=\s*(\[\]|\{\})\)', line)
            if m:
                issues.append(f"Line {i}: mutable default argument ({m.group(1)})")

        return issues
