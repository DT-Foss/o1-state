"""
Test Generator — Automatic Test Scaffolding
=============================================
Generates test skeletons from Python source code.
Analyzes function signatures, docstrings, and return types
to create meaningful test cases.

Usage:
    gen = TestGenerator()
    tests = gen.generate_from_file("core/math_solver.py")
    print(tests)
"""

import ast
import os
from typing import Dict, List, Union


class FunctionInfo:
    """Extracted info about a function."""

    def __init__(self, name: str, args: List[str], defaults: List[str],
                 return_type: str, docstring: str, is_method: bool,
                 class_name: str, decorators: List[str]):
        self.name = name
        self.args = args
        self.defaults = defaults
        self.return_type = return_type
        self.docstring = docstring
        self.is_method = is_method
        self.class_name = class_name
        self.decorators = decorators


class TestGenerator:
    """Generate test scaffolds from Python source."""

    def __init__(self, style: str = 'function'):
        """
        Args:
            style: 'function' for standalone tests, 'class' for unittest-style
        """
        self.style = style

    def analyze_file(self, filepath: str) -> List[FunctionInfo]:
        """Extract all functions and methods from a file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        return self.analyze_source(source)

    def analyze_source(self, source: str) -> List[FunctionInfo]:
        """Extract function info from source code."""
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        functions: List[FunctionInfo] = []

        # Only iterate over top-level statements (not ast.walk which recurses)
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        info = self._extract_function(item, class_name=node.name)
                        if not info.name.startswith('_') or info.name == '__init__':
                            functions.append(info)

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info = self._extract_function(node)
                if not info.name.startswith('_'):
                    functions.append(info)

        return functions

    def _extract_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
                          class_name: str = '') -> FunctionInfo:
        """Extract info from a function AST node."""
        name = node.name

        # Args (skip 'self' for methods)
        args = []
        defaults = []
        for arg in node.args.args:
            arg_name = arg.arg
            if arg_name == 'self':
                continue
            args.append(arg_name)

        # Defaults
        for default in node.args.defaults:
            if isinstance(default, ast.Constant):
                defaults.append(repr(default.value))
            else:
                defaults.append('...')

        # Return type annotation
        return_type = ''
        if node.returns:
            return_type = ast.dump(node.returns)

        # Docstring
        docstring = ''
        if (node.body and isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value

        # Decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(dec.id)
            elif isinstance(dec, ast.Attribute):
                decorators.append(dec.attr)

        is_method = bool(class_name)

        return FunctionInfo(
            name=name, args=args, defaults=defaults,
            return_type=return_type, docstring=docstring,
            is_method=is_method, class_name=class_name,
            decorators=decorators,
        )

    def generate_from_file(self, filepath: str) -> str:
        """Generate test code for a Python file."""
        functions = self.analyze_file(filepath)
        module_name = os.path.splitext(os.path.basename(filepath))[0]
        module_path = filepath.replace('/', '.').replace('.py', '')

        if self.style == 'class':
            return self._generate_class_tests(functions, module_name, module_path)
        return self._generate_function_tests(functions, module_name, module_path)

    def _generate_function_tests(self, functions: List[FunctionInfo],
                                  module_name: str, module_path: str) -> str:
        """Generate standalone test functions."""
        lines = [
            f'"""Auto-generated tests for {module_name}."""',
            '',
            'import sys',
            'import os',
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
            '',
        ]

        # Collect imports
        classes = set()
        top_functions = set()
        for f in functions:
            if f.class_name:
                classes.add(f.class_name)
            else:
                top_functions.add(f.name)

        if classes or top_functions:
            imports = list(classes) + list(top_functions)
            lines.append(f'from {module_path} import {", ".join(sorted(imports))}')
            lines.append('')

        # Global pass/fail counter
        lines.extend([
            'PASS = 0',
            'FAIL = 0',
            '',
            '',
            'def check(name, condition, detail=""):',
            '    global PASS, FAIL',
            '    if condition:',
            '        PASS += 1',
            '        print(f"  ✓ {name}")',
            '    else:',
            '        FAIL += 1',
            '        print(f"  ✗ {name} — {detail}")',
            '',
        ])

        # Group by class
        by_class: Dict[str, List[FunctionInfo]] = {}
        for f in functions:
            key = f.class_name or '__module__'
            if key not in by_class:
                by_class[key] = []
            by_class[key].append(f)

        for class_name, class_functions in by_class.items():
            if class_name == '__module__':
                # Top-level functions
                lines.append('')
                lines.append(f'def test_{module_name}_functions():')
                lines.append(f'    print("\\n=== {module_name} functions ===")')
                for f in class_functions:
                    lines.extend(self._generate_test_body(f, indent=4))
            else:
                lines.append('')
                lines.append(f'def test_{class_name.lower()}():')
                lines.append(f'    print("\\n=== {class_name} ===")')

                # Instantiation
                init_func = next(
                    (f for f in class_functions if f.name == '__init__'), None
                )
                if init_func:
                    args_str = self._generate_sample_args(init_func)
                    lines.append(f'    obj = {class_name}({args_str})')
                    lines.append(f'    check("{class_name} creates", obj is not None)')
                else:
                    lines.append(f'    obj = {class_name}()')
                    lines.append(f'    check("{class_name} creates", obj is not None)')

                lines.append('')

                for f in class_functions:
                    if f.name == '__init__':
                        continue
                    lines.extend(self._generate_test_body(f, indent=4, obj_name='obj'))

        # Main
        lines.extend([
            '',
            '',
            "if __name__ == '__main__':",
        ])

        for class_name in by_class:
            if class_name == '__module__':
                lines.append(f'    test_{module_name}_functions()')
            else:
                lines.append(f'    test_{class_name.lower()}()')

        lines.extend([
            '',
            '    total = PASS + FAIL',
            '    print(f"\\n{\'=\'*50}")',
            '    print(f"Results: {PASS}/{total} passed ({PASS/total*100:.0f}%)")',
            '    if FAIL:',
            '        print(f"FAILURES: {FAIL}")',
            '    else:',
            '        print("ALL TESTS PASSED")',
        ])

        return '\n'.join(lines)

    def _generate_test_body(self, func: FunctionInfo, indent: int = 4,
                            obj_name: str = '') -> List[str]:
        """Generate test assertions for a function."""
        pad = ' ' * indent
        lines = []
        name = func.name
        call_prefix = f'{obj_name}.' if obj_name else ''

        # Skip property decorators
        if 'property' in func.decorators:
            lines.append(f'{pad}# Property: {name}')
            lines.append(f'{pad}check("{name} accessible", hasattr({obj_name}, "{name}"))')
            return lines

        args_str = self._generate_sample_args(func)
        call = f'{call_prefix}{name}({args_str})'

        lines.append(f'{pad}# {name}')
        lines.append(f'{pad}result = {call}')
        lines.append(f'{pad}check("{name} returns", result is not None)')

        # Type-specific checks based on return type hints or docstring
        if 'Dict' in func.return_type or 'dict' in func.return_type:
            lines.append(f'{pad}check("{name} returns dict", isinstance(result, dict))')
        elif 'List' in func.return_type or 'list' in func.return_type:
            lines.append(f'{pad}check("{name} returns list", isinstance(result, (list, tuple)))')
        elif 'str' in func.return_type:
            lines.append(f'{pad}check("{name} returns str", isinstance(result, str))')
        elif 'bool' in func.return_type:
            lines.append(f'{pad}check("{name} returns bool", isinstance(result, bool))')
        elif 'int' in func.return_type or 'float' in func.return_type:
            lines.append(f'{pad}check("{name} returns number", isinstance(result, (int, float)))')

        lines.append('')
        return lines

    def _generate_sample_args(self, func: FunctionInfo) -> str:
        """Generate sample arguments for a function call."""
        if not func.args:
            return ''

        # Use defaults where available
        n_args = len(func.args)
        n_defaults = len(func.defaults)

        parts = []
        for i, arg in enumerate(func.args):
            # Check if this arg has a default
            default_idx = i - (n_args - n_defaults)
            if default_idx >= 0:
                # Has default, can skip
                continue

            # Generate sample value based on name heuristics
            sample = self._guess_sample_value(arg)
            parts.append(sample)

        return ', '.join(parts)

    def _guess_sample_value(self, arg_name: str) -> str:
        """Guess a sample value for an argument based on its name."""
        name = arg_name.lower()

        if name in ('text', 'source', 'code', 'content', 'message', 'query',
                     'question', 'input', 'prompt'):
            return '"test"'
        if name in ('path', 'filepath', 'file_path', 'filename', 'dir', 'directory'):
            return '"."'
        if name in ('n', 'count', 'size', 'limit', 'max', 'min', 'num',
                     'top_k', 'top_n', 'k'):
            return '5'
        if name in ('dim', 'dimension', 'dimensions', 'width', 'height'):
            return '64'
        if name in ('enabled', 'verbose', 'debug', 'force', 'recursive'):
            return 'True'
        if name in ('threshold', 'weight', 'alpha', 'beta', 'rate',
                     'temperature', 'epsilon'):
            return '0.5'
        if name in ('items', 'data', 'values', 'elements', 'facts'):
            return '[]'
        if name in ('config', 'options', 'params', 'kwargs', 'settings'):
            return '{}'
        if name in ('name', 'label', 'title', 'key', 'tag'):
            return '"test"'

        return '"test"'

    def _generate_class_tests(self, functions: List[FunctionInfo],
                               module_name: str, module_path: str) -> str:
        """Generate unittest-style test classes."""
        lines = [
            f'"""Auto-generated tests for {module_name}."""',
            '',
            'import unittest',
            'import sys',
            'import os',
            "sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))",
            '',
        ]

        classes = set(f.class_name for f in functions if f.class_name)
        top_funcs = [f.name for f in functions if not f.class_name]

        imports = list(classes) + top_funcs
        if imports:
            lines.append(f'from {module_path} import {", ".join(sorted(imports))}')
            lines.append('')

        # Group by class
        by_class: Dict[str, List[FunctionInfo]] = {}
        for f in functions:
            key = f.class_name or '__module__'
            if key not in by_class:
                by_class[key] = []
            by_class[key].append(f)

        for class_name, class_functions in by_class.items():
            test_class = f'Test{class_name}' if class_name != '__module__' else f'Test{module_name.title()}'
            lines.append(f'class {test_class}(unittest.TestCase):')

            if class_name != '__module__':
                lines.append(f'    def setUp(self):')
                init_func = next(
                    (f for f in class_functions if f.name == '__init__'), None
                )
                if init_func:
                    args = self._generate_sample_args(init_func)
                    lines.append(f'        self.obj = {class_name}({args})')
                else:
                    lines.append(f'        self.obj = {class_name}()')
                lines.append('')

            for f in class_functions:
                if f.name == '__init__':
                    continue
                test_name = f'test_{f.name}'
                lines.append(f'    def {test_name}(self):')
                if class_name != '__module__':
                    args = self._generate_sample_args(f)
                    lines.append(f'        result = self.obj.{f.name}({args})')
                else:
                    args = self._generate_sample_args(f)
                    lines.append(f'        result = {f.name}({args})')
                lines.append(f'        self.assertIsNotNone(result)')
                lines.append('')

        lines.extend([
            '',
            "if __name__ == '__main__':",
            '    unittest.main()',
        ])

        return '\n'.join(lines)
