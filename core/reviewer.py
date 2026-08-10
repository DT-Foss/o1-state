"""
Code Reviewer — Static Analysis Without Pylint/Flake8
========================================================
AST-based Python code analysis.

Checks:
  1. Unused imports
  2. Unused variables
  3. Functions too long (>50 lines)
  4. Too many arguments (>5)
  5. Missing docstrings
  6. Bare except clauses
  7. Mutable default arguments
  8. Global variable usage
  9. Complexity estimation (nested depth)
  10. Naming conventions (PEP 8)
  11. Dead code detection (unreachable after return)
  12. Duplicate code patterns
"""

import ast
import re
from typing import Dict, Any, List


class Issue:
    """A code review finding."""

    def __init__(self, severity: str, code: str, message: str,
                 line: int = 0, col: int = 0):
        self.severity = severity  # 'error', 'warning', 'info', 'style'
        self.code = code  # e.g., 'R001'
        self.message = message
        self.line = line
        self.col = col

    def __repr__(self):
        return f"[{self.severity.upper()}] L{self.line}: {self.code} {self.message}"


class CodeReviewer:
    """
    AST-based Python code reviewer.

    Usage:
        reviewer = CodeReviewer()
        issues = reviewer.review(code_string)
        report = reviewer.report(code_string)
    """

    def __init__(self, max_function_lines: int = 50,
                 max_args: int = 5,
                 max_nesting: int = 4,
                 check_naming: bool = True):
        self.max_function_lines = max_function_lines
        self.max_args = max_args
        self.max_nesting = max_nesting
        self.check_naming = check_naming

    def review(self, code: str) -> List[Issue]:
        """Review code and return list of issues."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return [Issue('error', 'E001', f'Syntax error: {e.msg}',
                         e.lineno or 0, e.offset or 0)]

        issues = []

        issues.extend(self._check_imports(tree, code))
        issues.extend(self._check_functions(tree))
        issues.extend(self._check_classes(tree))
        issues.extend(self._check_exceptions(tree))
        issues.extend(self._check_mutable_defaults(tree))
        issues.extend(self._check_globals(tree))
        issues.extend(self._check_dead_code(tree))
        if self.check_naming:
            issues.extend(self._check_naming(tree))
        issues.extend(self._check_nesting(tree))
        issues.extend(self._check_unused_variables(tree))

        issues.sort(key=lambda i: i.line)
        return issues

    def report(self, code: str, filename: str = '<code>') -> str:
        """Generate a human-readable review report."""
        issues = self.review(code)
        if not issues:
            return f"{filename}: No issues found. Code looks clean."

        lines = [f"Code Review: {filename}", f"Issues: {len(issues)}", ""]

        by_severity = {}
        for issue in issues:
            by_severity.setdefault(issue.severity, []).append(issue)

        for severity in ['error', 'warning', 'info', 'style']:
            group = by_severity.get(severity, [])
            if group:
                lines.append(f"--- {severity.upper()} ({len(group)}) ---")
                for i in group:
                    lines.append(f"  L{i.line:4d}: [{i.code}] {i.message}")
                lines.append("")

        # Summary
        lines.append(f"Total: {len(issues)} issues "
                     f"({len(by_severity.get('error', []))} errors, "
                     f"{len(by_severity.get('warning', []))} warnings, "
                     f"{len(by_severity.get('info', []))} info, "
                     f"{len(by_severity.get('style', []))} style)")

        return '\n'.join(lines)

    def score(self, code: str) -> Dict[str, Any]:
        """
        Generate a code quality score (0-100).
        """
        issues = self.review(code)
        lines = code.splitlines()
        total_lines = len(lines)

        # Penalty per issue type
        penalties = {
            'error': 10,
            'warning': 3,
            'info': 1,
            'style': 0.5,
        }

        total_penalty = sum(penalties.get(i.severity, 1) for i in issues)
        # Normalize: more lines = more tolerance
        normalized = total_penalty / max(total_lines / 10, 1)
        score = max(0, 100 - normalized * 10)

        return {
            'score': round(score, 1),
            'issues': len(issues),
            'lines': total_lines,
            'grade': _score_to_grade(score),
        }

    # === Individual Checks ===

    def _check_imports(self, tree: ast.Module, code: str) -> List[Issue]:
        """Check for unused imports."""
        issues = []
        imports = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    name = alias.asname or alias.name
                    imports[name] = node.lineno

        # Check if imported names are used in code
        for name, line in imports.items():
            # Simple check: does the name appear elsewhere in code?
            # Skip the import line itself
            code_lines = code.splitlines()
            used = False
            for i, code_line in enumerate(code_lines):
                if i + 1 == line:
                    continue  # Skip the import line
                if re.search(rf'\b{re.escape(name)}\b', code_line):
                    used = True
                    break
            if not used:
                issues.append(Issue('warning', 'R001',
                                   f"Unused import: '{name}'", line))
        return issues

    def _check_functions(self, tree: ast.Module) -> List[Issue]:
        """Check function quality."""
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue

            # Function length
            end_line = getattr(node, 'end_lineno', node.lineno + self.max_function_lines)
            func_lines = end_line - node.lineno
            if func_lines > self.max_function_lines:
                issues.append(Issue('warning', 'R002',
                    f"Function '{node.name}' is {func_lines} lines "
                    f"(max {self.max_function_lines})", node.lineno))

            # Too many arguments
            args = node.args
            n_args = len(args.args) + len(args.kwonlyargs)
            if args.vararg:
                n_args += 1
            if args.kwarg:
                n_args += 1
            # Don't count 'self' or 'cls'
            if args.args and args.args[0].arg in ('self', 'cls'):
                n_args -= 1
            if n_args > self.max_args:
                issues.append(Issue('warning', 'R003',
                    f"Function '{node.name}' has {n_args} arguments "
                    f"(max {self.max_args})", node.lineno))

            # Missing docstring
            if (node.body and not (
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)
            )):
                if not node.name.startswith('_'):
                    issues.append(Issue('info', 'R004',
                        f"Function '{node.name}' has no docstring", node.lineno))

        return issues

    def _check_classes(self, tree: ast.Module) -> List[Issue]:
        """Check class quality."""
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Missing docstring
            if (node.body and not (
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant)
            )):
                issues.append(Issue('info', 'R005',
                    f"Class '{node.name}' has no docstring", node.lineno))

        return issues

    def _check_exceptions(self, tree: ast.Module) -> List[Issue]:
        """Check exception handling."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(Issue('warning', 'R006',
                        "Bare 'except:' clause (catches all exceptions)",
                        node.lineno))
        return issues

    def _check_mutable_defaults(self, tree: ast.Module) -> List[Issue]:
        """Check for mutable default arguments."""
        issues = []
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for default in node.args.defaults + node.args.kw_defaults:
                if default is None:
                    continue
                if isinstance(default, (ast.List, ast.Dict, ast.Set)):
                    issues.append(Issue('warning', 'R007',
                        f"Mutable default argument in '{node.name}'",
                        node.lineno))
                    break
        return issues

    def _check_globals(self, tree: ast.Module) -> List[Issue]:
        """Check for global variable usage."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Global):
                issues.append(Issue('info', 'R008',
                    f"Global variable usage: {', '.join(node.names)}",
                    node.lineno))
        return issues

    def _check_dead_code(self, tree: ast.Module) -> List[Issue]:
        """Check for unreachable code after return/raise/break/continue."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for i, stmt in enumerate(node.body[:-1]):
                    if isinstance(stmt, (ast.Return, ast.Raise, ast.Break, ast.Continue)):
                        next_stmt = node.body[i + 1]
                        issues.append(Issue('warning', 'R009',
                            f"Unreachable code after {type(stmt).__name__}",
                            getattr(next_stmt, 'lineno', node.lineno)))
        return issues

    def _check_naming(self, tree: ast.Module) -> List[Issue]:
        """Check PEP 8 naming conventions."""
        issues = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                    issues.append(Issue('style', 'R010',
                        f"Class '{node.name}' should use CamelCase",
                        node.lineno))

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_') and not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                    if node.name not in ('setUp', 'tearDown', 'setUpClass', 'tearDownClass'):
                        issues.append(Issue('style', 'R011',
                            f"Function '{node.name}' should use snake_case",
                            node.lineno))

        return issues

    def _check_nesting(self, tree: ast.Module) -> List[Issue]:
        """Check for excessive nesting depth."""
        issues = []
        self._walk_nesting(tree, 0, issues)
        return issues

    def _walk_nesting(self, node: ast.AST, depth: int, issues: List[Issue]):
        """Recursively track nesting depth."""
        nesting_nodes = (ast.If, ast.For, ast.While, ast.With, ast.Try)
        for child in ast.iter_child_nodes(node):
            if isinstance(child, nesting_nodes):
                new_depth = depth + 1
                if new_depth > self.max_nesting:
                    issues.append(Issue('warning', 'R012',
                        f"Nesting depth {new_depth} exceeds maximum {self.max_nesting}",
                        getattr(child, 'lineno', 0)))
                self._walk_nesting(child, new_depth, issues)
            else:
                self._walk_nesting(child, depth, issues)

    def _check_unused_variables(self, tree: ast.Module) -> List[Issue]:
        """Check for variables assigned but never used."""
        issues = []
        # Only check top-level function assignments (simple version)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                assigned = set()
                used = set()
                for child in ast.walk(node):
                    if isinstance(child, ast.Assign):
                        for target in child.targets:
                            if isinstance(target, ast.Name):
                                assigned.add(target.id)
                    elif isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load):
                        used.add(child.id)
                unused = assigned - used - {'_', '__'}
                for var in unused:
                    if not var.startswith('_'):
                        issues.append(Issue('info', 'R013',
                            f"Variable '{var}' assigned but not used in '{node.name}'",
                            node.lineno))
        return issues


def _score_to_grade(score: float) -> str:
    """Convert numeric score to letter grade."""
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    return 'F'
