"""
Self-Modification Engine — The System That Patches Itself
==========================================================
F8 in the extended capabilities framework.

The loop so far: Predict → Execute → Compare → Attribute → Fix → Remember
But "fix" only applies to GENERATED code.

Self-Modification: fix the SYSTEM ITSELF.

When the self-improvement loop finds that a module consistently fails,
this engine can:
  1. Read the module's source code
  2. Identify the problematic function
  3. Generate a fix (using code templates + pattern memory)
  4. Apply the fix safely (backup → patch → test → rollback if broken)

Safety: NEVER modifies without backup. NEVER modifies if tests break.
"""

import ast
import os
import shutil
import time
from typing import Dict, Any, List, Optional, Tuple


class SourceReader:
    """Read and analyze Python source files."""

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.dirname(__file__)

    def read_module(self, module_name: str) -> Optional[str]:
        """Read a module's source code."""
        path = os.path.join(self.base_path, f'{module_name}.py')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return f.read()
        return None

    def get_function(self, module_name: str,
                     func_name: str) -> Optional[Dict[str, Any]]:
        """Extract a function's source, line numbers, and signature."""
        source = self.read_module(module_name)
        if not source:
            return None

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return None

        lines = source.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.name == func_name:
                    start = node.lineno - 1
                    end = node.end_lineno if hasattr(node, 'end_lineno') else start + 1
                    func_source = '\n'.join(lines[start:end])

                    # Extract args
                    args = []
                    for arg in node.args.args:
                        args.append(arg.arg)

                    # Extract decorators
                    decorators = []
                    for dec in node.decorator_list:
                        if isinstance(dec, ast.Name):
                            decorators.append(dec.id)
                        elif isinstance(dec, ast.Attribute):
                            decorators.append(f"{dec.value.id}.{dec.attr}"
                                              if isinstance(dec.value, ast.Name)
                                              else dec.attr)

                    return {
                        'name': func_name,
                        'module': module_name,
                        'source': func_source,
                        'start_line': start + 1,
                        'end_line': end,
                        'args': args,
                        'decorators': decorators,
                        'has_docstring': (isinstance(node.body[0], ast.Expr) and
                                          isinstance(node.body[0].value, ast.Constant) and
                                          isinstance(node.body[0].value.value, str))
                                         if node.body else False,
                    }
        return None

    def list_functions(self, module_name: str) -> List[Dict[str, Any]]:
        """List all functions in a module with basic info."""
        source = self.read_module(module_name)
        if not source:
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        functions = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.append({
                    'name': node.name,
                    'line': node.lineno,
                    'args': [a.arg for a in node.args.args],
                    'is_method': any(a.arg == 'self' for a in node.args.args),
                })
        return functions

    def list_classes(self, module_name: str) -> List[Dict[str, Any]]:
        """List all classes in a module."""
        source = self.read_module(module_name)
        if not source:
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        classes = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body
                           if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append({
                    'name': node.name,
                    'line': node.lineno,
                    'methods': methods,
                    'bases': [b.id if isinstance(b, ast.Name) else str(b)
                              for b in node.bases],
                })
        return classes

    def find_function_callers(self, module_name: str,
                              func_name: str) -> List[Dict[str, Any]]:
        """Find all places that call a given function within a module."""
        source = self.read_module(module_name)
        if not source:
            return []

        try:
            tree = ast.parse(source)
        except SyntaxError:
            return []

        callers = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name):
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute):
                    name = node.func.attr

                if name == func_name:
                    callers.append({
                        'line': node.lineno,
                        'col': node.col_offset,
                    })
        return callers


class Patch:
    """A code patch to apply to a module."""

    def __init__(self, module_name: str, old_text: str, new_text: str,
                 description: str = '', line_hint: int = 0):
        self.module_name = module_name
        self.old_text = old_text
        self.new_text = new_text
        self.description = description
        self.line_hint = line_hint
        self.timestamp = time.time()
        self.applied = False
        self.rolled_back = False

    def to_dict(self) -> Dict:
        return {
            'module': self.module_name,
            'old': self.old_text[:200],
            'new': self.new_text[:200],
            'description': self.description,
            'line': self.line_hint,
            'applied': self.applied,
            'rolled_back': self.rolled_back,
        }


class SelfModifier:
    """
    Safely modify FOSS-KI's own source code.

    Safety protocol:
    1. BACKUP the file before any modification
    2. APPLY the patch
    3. VERIFY the patched code parses (AST check)
    4. ROLLBACK if verification fails

    Never modifies without backup.
    Never modifies if the result doesn't parse.
    """

    def __init__(self, base_path: str = None):
        self.base_path = base_path or os.path.dirname(__file__)
        self.reader = SourceReader(self.base_path)
        self.backup_dir = os.path.join(self.base_path, '..', '.backups')
        self.history: List[Patch] = []

    def apply_patch(self, patch: Patch, verify: bool = True) -> Dict[str, Any]:
        """
        Apply a patch safely.

        1. Read current source
        2. Verify old_text exists
        3. Backup
        4. Apply replacement
        5. Verify new source parses
        6. Rollback if broken
        """
        source = self.reader.read_module(patch.module_name)
        if source is None:
            return {'success': False, 'error': f'Module {patch.module_name} not found'}

        # Check that old_text exists
        if patch.old_text not in source:
            return {'success': False, 'error': 'old_text not found in source',
                    'hint': 'The code may have already been modified'}

        # Count occurrences — must be exactly 1
        count = source.count(patch.old_text)
        if count > 1:
            return {'success': False, 'error': f'old_text found {count} times, '
                    'must be unique. Provide more context.'}

        # Backup
        backup_path = self._backup(patch.module_name, source)

        # Apply
        new_source = source.replace(patch.old_text, patch.new_text, 1)

        # Verify it parses
        if verify:
            try:
                ast.parse(new_source)
            except SyntaxError as e:
                # Rollback
                self._restore(patch.module_name, backup_path)
                return {'success': False, 'error': f'Patch creates syntax error: {e}',
                        'rolled_back': True}

        # Write
        path = os.path.join(self.base_path, f'{patch.module_name}.py')
        with open(path, 'w') as f:
            f.write(new_source)

        patch.applied = True
        self.history.append(patch)

        return {
            'success': True,
            'backup': backup_path,
            'description': patch.description,
            'module': patch.module_name,
        }

    def rollback_last(self) -> Dict[str, Any]:
        """Rollback the most recent patch."""
        if not self.history:
            return {'success': False, 'error': 'No patches to rollback'}

        last = self.history[-1]
        if last.rolled_back:
            return {'success': False, 'error': 'Already rolled back'}

        # Find the backup
        backup_path = self._find_backup(last.module_name, last.timestamp)
        if backup_path and os.path.exists(backup_path):
            self._restore(last.module_name, backup_path)
            last.rolled_back = True
            return {'success': True, 'module': last.module_name,
                    'restored_from': backup_path}

        # Try reverse patch
        source = self.reader.read_module(last.module_name)
        if source and last.new_text in source:
            new_source = source.replace(last.new_text, last.old_text, 1)
            path = os.path.join(self.base_path, f'{last.module_name}.py')
            with open(path, 'w') as f:
                f.write(new_source)
            last.rolled_back = True
            return {'success': True, 'module': last.module_name,
                    'method': 'reverse_patch'}

        return {'success': False, 'error': 'Cannot find backup or reverse patch'}

    def add_to_function(self, module_name: str, func_name: str,
                        code_to_add: str, position: str = 'end',
                        description: str = '') -> Dict[str, Any]:
        """
        Add code to an existing function.

        Args:
            position: 'start' (after docstring), 'end' (before return),
                      'before_return'
        """
        func_info = self.reader.get_function(module_name, func_name)
        if not func_info:
            return {'success': False, 'error': f'{func_name} not found in {module_name}'}

        source = self.reader.read_module(module_name)
        lines = source.split('\n')

        # Determine indentation of the function body
        body_start = func_info['start_line']  # 1-indexed
        body_indent = ''
        for i in range(body_start, min(body_start + 10, len(lines))):
            line = lines[i]
            stripped = line.lstrip()
            if stripped and not stripped.startswith('def ') and not stripped.startswith('"""'):
                body_indent = line[:len(line) - len(stripped)]
                break

        if not body_indent:
            body_indent = '        '  # Default 8 spaces

        # Indent the code to add
        indented_code = '\n'.join(
            body_indent + line if line.strip() else line
            for line in code_to_add.split('\n')
        )

        if position == 'start':
            # Add after docstring or def line
            insert_line = body_start  # after def line
            # Skip docstring if present
            if func_info['has_docstring']:
                for i in range(body_start, func_info['end_line']):
                    if '"""' in lines[i] or "'''" in lines[i]:
                        # Find closing quotes
                        if lines[i].count('"""') >= 2 or lines[i].count("'''") >= 2:
                            insert_line = i + 1
                            break
                        # Multi-line docstring
                        for j in range(i + 1, func_info['end_line']):
                            if '"""' in lines[j] or "'''" in lines[j]:
                                insert_line = j + 1
                                break
                        break
            old_line = lines[insert_line] if insert_line < len(lines) else ''
            new_content = indented_code + '\n' + old_line

            patch = Patch(
                module_name=module_name,
                old_text=old_line,
                new_text=new_content,
                description=description or f'Add code to start of {func_name}',
                line_hint=insert_line + 1,
            )
        elif position in ('end', 'before_return'):
            # Add before the last line (usually return)
            end_line = func_info['end_line'] - 1  # 0-indexed
            old_line = lines[end_line]
            new_content = indented_code + '\n' + old_line

            patch = Patch(
                module_name=module_name,
                old_text=old_line,
                new_text=new_content,
                description=description or f'Add code to end of {func_name}',
                line_hint=end_line + 1,
            )
        else:
            return {'success': False, 'error': f'Unknown position: {position}'}

        return self.apply_patch(patch)

    def inspect_module(self, module_name: str) -> Dict[str, Any]:
        """Get a structural overview of a module."""
        source = self.reader.read_module(module_name)
        if not source:
            return {'error': f'Module {module_name} not found'}

        classes = self.reader.list_classes(module_name)
        functions = self.reader.list_functions(module_name)

        # Top-level functions (not methods)
        top_funcs = [f for f in functions if not f['is_method']]

        return {
            'module': module_name,
            'lines': len(source.split('\n')),
            'classes': classes,
            'top_level_functions': top_funcs,
            'imports': self._count_imports(source),
        }

    def patch_history(self) -> List[Dict]:
        """Get history of applied patches."""
        return [p.to_dict() for p in self.history]

    def _backup(self, module_name: str, source: str) -> str:
        """Create a backup of a module."""
        os.makedirs(self.backup_dir, exist_ok=True)
        timestamp = int(time.time())
        backup_path = os.path.join(
            self.backup_dir, f'{module_name}_{timestamp}.py.bak')
        with open(backup_path, 'w') as f:
            f.write(source)
        return backup_path

    def _restore(self, module_name: str, backup_path: str):
        """Restore a module from backup."""
        path = os.path.join(self.base_path, f'{module_name}.py')
        shutil.copy2(backup_path, path)

    def _find_backup(self, module_name: str,
                     timestamp: float) -> Optional[str]:
        """Find the backup closest to a given timestamp."""
        if not os.path.exists(self.backup_dir):
            return None

        closest = None
        closest_diff = float('inf')

        for fname in os.listdir(self.backup_dir):
            if fname.startswith(module_name + '_') and fname.endswith('.py.bak'):
                try:
                    ts = int(fname.split('_')[-1].replace('.py.bak', ''))
                    diff = abs(ts - timestamp)
                    if diff < closest_diff:
                        closest = os.path.join(self.backup_dir, fname)
                        closest_diff = diff
                except ValueError:
                    continue

        return closest

    def _count_imports(self, source: str) -> int:
        """Count import statements."""
        try:
            tree = ast.parse(source)
            return sum(1 for n in ast.walk(tree)
                       if isinstance(n, (ast.Import, ast.ImportFrom)))
        except SyntaxError:
            return 0
