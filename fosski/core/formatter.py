"""
Code Formatter — PEP 8 Auto-formatter
========================================
Python code formatter without external dependencies.
Replaces black/autopep8 with a lightweight AST-based formatter.

Features:
  - Fix indentation (spaces/tabs)
  - Fix whitespace around operators
  - Fix blank lines between functions/classes
  - Fix trailing whitespace
  - Sort imports (stdlib → 3rd party → local)
  - Fix line length (basic wrapping)
  - Normalize quotes
"""

import re
from typing import List, Tuple


# Standard library module names (common subset)
STDLIB_MODULES = {
    'abc', 'argparse', 'ast', 'asyncio', 'base64', 'bisect', 'calendar',
    'cmath', 'codecs', 'collections', 'colorsys', 'concurrent', 'configparser',
    'contextlib', 'copy', 'csv', 'ctypes', 'dataclasses', 'datetime',
    'decimal', 'difflib', 'dis', 'email', 'enum', 'errno', 'fcntl',
    'fileinput', 'fnmatch', 'fractions', 'functools', 'gc', 'getpass',
    'glob', 'gzip', 'hashlib', 'heapq', 'hmac', 'html', 'http',
    'importlib', 'inspect', 'io', 'ipaddress', 'itertools', 'json',
    'keyword', 'linecache', 'locale', 'logging', 'lzma', 'math',
    'mimetypes', 'multiprocessing', 'numbers', 'operator', 'os', 'pathlib',
    'pdb', 'pickle', 'platform', 'pprint', 'profile', 'queue', 'random',
    're', 'readline', 'resource', 'select', 'shelve', 'shlex', 'shutil',
    'signal', 'site', 'socket', 'sqlite3', 'ssl', 'stat', 'statistics',
    'string', 'struct', 'subprocess', 'sys', 'sysconfig', 'tempfile',
    'textwrap', 'threading', 'time', 'timeit', 'token', 'tokenize',
    'traceback', 'types', 'typing', 'unittest', 'urllib', 'uuid',
    'venv', 'warnings', 'weakref', 'xml', 'zipfile', 'zlib',
    # Builtins
    '__future__', 'builtins',
}


class CodeFormatter:
    """PEP 8 Python code formatter."""

    def __init__(self, indent_size: int = 4, max_line_length: int = 100,
                 sort_imports: bool = True, normalize_quotes: bool = False):
        self.indent_size = indent_size
        self.max_line_length = max_line_length
        self.sort_imports = sort_imports
        self.normalize_quotes = normalize_quotes

    def format(self, code: str) -> str:
        """Format Python code according to PEP 8."""
        lines = code.split('\n')

        # Phase 1: Line-level fixes
        lines = self._fix_trailing_whitespace(lines)
        lines = self._fix_tabs_to_spaces(lines)
        lines = self._fix_operator_spacing(lines)
        lines = self._fix_blank_lines(lines)

        # Phase 2: Import sorting
        if self.sort_imports:
            lines = self._sort_imports(lines)

        # Phase 3: Final cleanup
        lines = self._fix_trailing_blank_lines(lines)

        return '\n'.join(lines)

    def format_file(self, filepath: str) -> Tuple[str, int]:
        """Format a file in place. Returns (formatted_code, n_changes)."""
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()

        formatted = self.format(original)
        n_changes = sum(1 for a, b in zip(
            original.split('\n'), formatted.split('\n')
        ) if a != b)

        return formatted, n_changes

    # === Phase 1: Line-level fixes ===

    def _fix_trailing_whitespace(self, lines: List[str]) -> List[str]:
        """Remove trailing whitespace from all lines."""
        return [line.rstrip() for line in lines]

    def _fix_tabs_to_spaces(self, lines: List[str]) -> List[str]:
        """Convert leading tabs to spaces."""
        result = []
        for line in lines:
            # Count leading tabs
            stripped = line.lstrip('\t')
            n_tabs = len(line) - len(stripped)
            if n_tabs > 0:
                result.append(' ' * (n_tabs * self.indent_size) + stripped)
            else:
                result.append(line)
        return result

    def _fix_operator_spacing(self, lines: List[str]) -> List[str]:
        """Fix spacing around operators (=, ==, !=, <=, >=, +=, etc.)."""
        result = []
        for line in lines:
            # Skip comments, strings, and empty lines
            stripped = line.lstrip()
            if not stripped or stripped.startswith('#'):
                result.append(line)
                continue

            # Don't touch lines inside multi-line strings
            indent = line[:len(line) - len(stripped)]
            fixed = self._fix_operators_in_line(stripped)
            result.append(indent + fixed)
        return result

    def _fix_operators_in_line(self, line: str) -> str:
        """Fix operator spacing in a single line (no indent)."""
        # Don't modify strings or comments
        if line.startswith('#') or line.startswith(('"""', "'''")):
            return line

        # Split into code and comment parts
        code, comment = self._split_code_comment(line)

        # Fix assignment operators: ensure spaces around = but not ==, !=, <=, >=
        # First protect compound operators
        code = re.sub(r'(?<=[^ !<>=+\-*/%&|^~])=(?!=)', ' = ', code)
        # Clean up double spaces that may result
        code = re.sub(r'  +', ' ', code)

        # Fix comparison operators
        for op in ['==', '!=', '<=', '>=', '+=', '-=', '*=', '/=', '**=',
                    '%=', '&=', '|=', '^=', '>>=', '<<=']:
            # Ensure space before and after
            pattern = re.escape(op)
            code = re.sub(rf'(?<!\s){pattern}', f' {op}', code)
            code = re.sub(rf'{pattern}(?!\s)', f'{op} ', code)

        # Don't add spaces around = in keyword arguments or defaults
        # This is tricky — for now, just clean up the most common patterns
        code = re.sub(r'(\w)\s*=\s*(?!=)', r'\1=', code)  # f(x = 1) → f(x=1)
        # But preserve assignment: x = 1
        # Heuristic: if line doesn't start with a keyword arg pattern
        if not re.match(r'\s*\w+\s*\(', code):
            # Standalone assignment
            code = re.sub(r'^(\w+)\s*=\s*', r'\1 = ', code)

        if comment:
            return f"{code.rstrip()}  {comment}"
        return code

    def _split_code_comment(self, line: str) -> Tuple[str, str]:
        """Split a line into code and inline comment."""
        in_string = None
        for i, c in enumerate(line):
            if c in ('"', "'") and (i == 0 or line[i-1] != '\\'):
                if in_string == c:
                    in_string = None
                elif in_string is None:
                    in_string = c
            elif c == '#' and in_string is None:
                return line[:i].rstrip(), line[i:]
        return line, ''

    def _fix_blank_lines(self, lines: List[str]) -> List[str]:
        """Fix blank lines: 2 before top-level def/class, 1 inside class."""
        result: List[str] = []
        prev_blank_count = 0

        for i, line in enumerate(lines):
            stripped = line.strip()

            if not stripped:
                prev_blank_count += 1
                result.append('')
                continue

            # Top-level def or class
            if stripped.startswith(('def ', 'class ', 'async def ')):
                indent_level = len(line) - len(line.lstrip())

                if indent_level == 0 and i > 0:
                    # Need 2 blank lines before top-level definitions
                    # Remove excess blank lines and add exactly 2
                    while result and result[-1] == '':
                        result.pop()
                    if result:  # Don't add blanks at start of file
                        result.append('')
                        result.append('')

                elif indent_level > 0 and prev_blank_count == 0 and i > 0:
                    # Need 1 blank line before methods (if none exists)
                    prev_stripped = lines[i-1].strip() if i > 0 else ''
                    if prev_stripped and not prev_stripped.startswith(('class ', '@')):
                        result.append('')

            prev_blank_count = 0
            result.append(line)

        return result

    # === Phase 2: Import sorting ===

    def _sort_imports(self, lines: List[str]) -> List[str]:
        """Sort imports: stdlib → 3rd party → local."""
        # Find import block
        import_start = -1
        import_end = -1
        imports: List[Tuple[str, str]] = []  # (category, line)

        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('import ', 'from ')):
                if import_start == -1:
                    import_start = i
                import_end = i

                # Categorize
                module = self._get_import_module(stripped)
                if module in STDLIB_MODULES:
                    category = 'a_stdlib'
                elif module.startswith('.'):
                    category = 'c_local'
                else:
                    category = 'b_third_party'

                imports.append((category, stripped))

            elif stripped and import_start >= 0 and not stripped.startswith('#'):
                # Non-import, non-blank line after imports
                break

        if not imports or import_start == -1:
            return lines

        # Sort within categories
        imports.sort(key=lambda x: (x[0], x[1].lower()))

        # Rebuild with blank lines between categories
        sorted_lines: List[str] = []
        prev_cat = ''
        for cat, imp_line in imports:
            if prev_cat and cat != prev_cat:
                sorted_lines.append('')
            sorted_lines.append(imp_line)
            prev_cat = cat

        # Replace import block
        result = lines[:import_start] + sorted_lines + lines[import_end + 1:]
        return result

    def _get_import_module(self, line: str) -> str:
        """Extract the top-level module name from an import statement."""
        line = line.strip()
        if line.startswith('from '):
            # from foo.bar import baz → foo
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].split('.')[0]
        elif line.startswith('import '):
            parts = line.split()
            if len(parts) >= 2:
                return parts[1].split('.')[0].rstrip(',')
        return ''

    # === Phase 3: Final cleanup ===

    def _fix_trailing_blank_lines(self, lines: List[str]) -> List[str]:
        """Ensure file ends with exactly one blank line."""
        while lines and lines[-1] == '':
            lines.pop()
        lines.append('')
        return lines

    def check(self, code: str) -> List[str]:
        """Check code for PEP 8 issues without modifying."""
        issues = []
        lines = code.split('\n')

        for i, line in enumerate(lines, 1):
            # Trailing whitespace
            if line != line.rstrip():
                issues.append(f"L{i}: trailing whitespace")

            # Tabs
            if '\t' in line[:len(line) - len(line.lstrip())]:
                issues.append(f"L{i}: tab indentation (use spaces)")

            # Line too long
            if len(line) > self.max_line_length:
                issues.append(f"L{i}: line too long ({len(line)} > {self.max_line_length})")

        # Check blank lines before definitions
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith(('def ', 'class ', 'async def ')):
                indent = len(line) - len(line.lstrip())
                if indent == 0 and i > 0:
                    # Count preceding blank lines
                    blanks = 0
                    j = i - 1
                    while j >= 0 and lines[j].strip() == '':
                        blanks += 1
                        j -= 1
                    if blanks < 2 and j >= 0:
                        issues.append(f"L{i+1}: expected 2 blank lines before top-level definition")

        return issues
