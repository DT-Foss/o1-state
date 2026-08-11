"""
Abstract Interpreter — Predict Code Output Without Executing
==============================================================
The missing piece for UNDERSTANDING vs. COPYING.

Understanding = having an internal model that PREDICTS what code does.
This module walks Python AST and simulates execution symbolically.

Levels of abstraction:
  1. TYPE: "this function returns a list" (cheapest, always possible)
  2. SHAPE: "this function returns a list of 3 integers" (medium)
  3. VALUE: "this function returns [3, 7, 9]" (expensive, not always possible)

For the self-improvement loop:
  PREDICT → EXECUTE → COMPARE → if mismatch → ATTRIBUTE (find causal line)

No external dependencies.
"""

import ast
import operator
from typing import Dict, Any, List, Optional, Tuple, Union


# Symbolic value types
class SymType:
    """Represents the abstract TYPE of a value."""
    def __init__(self, name: str, element_type: str = '', length: int = -1):
        self.name = name              # 'int', 'str', 'list', 'dict', etc.
        self.element_type = element_type  # for containers: element type
        self.length = length          # for containers: known length (-1 = unknown)

    def __repr__(self):
        if self.element_type:
            len_str = f', len={self.length}' if self.length >= 0 else ''
            return f'{self.name}[{self.element_type}{len_str}]'
        return self.name

    def __eq__(self, other):
        if not isinstance(other, SymType):
            return False
        return self.name == other.name


class SymValue:
    """A value that can be concrete or symbolic."""
    def __init__(self, value=None, sym_type: SymType = None, symbolic: bool = False):
        self.value = value
        self.type = sym_type or self._infer_type(value)
        self.symbolic = symbolic  # True = we only know the type, not the value

    def _infer_type(self, value) -> SymType:
        if value is None:
            return SymType('NoneType')
        if isinstance(value, bool):
            return SymType('bool')
        if isinstance(value, int):
            return SymType('int')
        if isinstance(value, float):
            return SymType('float')
        if isinstance(value, str):
            return SymType('str')
        if isinstance(value, list):
            elem = type(value[0]).__name__ if value else 'unknown'
            return SymType('list', elem, len(value))
        if isinstance(value, dict):
            return SymType('dict', length=len(value))
        if isinstance(value, tuple):
            return SymType('tuple', length=len(value))
        if isinstance(value, set):
            return SymType('set', length=len(value))
        return SymType(type(value).__name__)

    @property
    def is_concrete(self) -> bool:
        return not self.symbolic

    def __repr__(self):
        if self.symbolic:
            return f'<{self.type}>'
        return f'{self.value!r}:{self.type}'


class AbstractInterpreter:
    """
    Walk Python AST and predict what code does without executing it.

    Usage:
        ai = AbstractInterpreter()
        prediction = ai.interpret(code_string)
        # prediction.return_type, prediction.return_value, prediction.side_effects
        # prediction.variables — state of all variables at end
    """

    # Safe operators for constant folding
    SAFE_OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.Eq: operator.eq,
        ast.NotEq: operator.ne,
        ast.Lt: operator.lt,
        ast.LtE: operator.le,
        ast.Gt: operator.gt,
        ast.GtE: operator.ge,
        ast.And: lambda a, b: a and b,
        ast.Or: lambda a, b: a or b,
    }

    MAX_LOOP_ITERATIONS = 1000  # Safety limit

    def __init__(self):
        self.env: Dict[str, SymValue] = {}  # Variable environment
        self.output: List[str] = []          # Captured print() calls
        self.return_value: Optional[SymValue] = None
        self.side_effects: List[str] = []    # file writes, network calls, etc.
        self.trace: List[Dict[str, Any]] = []  # Execution trace per line
        self.errors: List[Dict[str, Any]] = []
        self._call_depth = 0
        self._functions: Dict[str, ast.FunctionDef] = {}

    def interpret(self, code: str) -> 'Prediction':
        """
        Interpret code and return a prediction.

        Returns:
            Prediction object with return_type, return_value, output,
            variables, side_effects, errors, trace
        """
        self.env = {}
        self.output = []
        self.return_value = None
        self.side_effects = []
        self.trace = []
        self.errors = []
        self._functions = {}

        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return Prediction(
                errors=[{'type': 'SYNTAX', 'message': str(e), 'line': e.lineno}],
                confidence='NONE',
            )

        # First pass: collect function definitions
        for node in tree.body:
            if isinstance(node, ast.FunctionDef):
                self._functions[node.name] = node

        # Second pass: execute top-level statements
        for node in tree.body:
            try:
                self._exec_node(node)
            except _ReturnSignal:
                break
            except _BreakSignal:
                break
            except Exception as e:
                self.errors.append({
                    'type': 'INTERPRET_ERROR',
                    'message': str(e),
                    'line': getattr(node, 'lineno', None),
                })

        # Determine confidence
        has_symbolic = any(v.symbolic for v in self.env.values())
        n_errors = len(self.errors)
        if n_errors > 0:
            confidence = 'LOW'
        elif has_symbolic:
            confidence = 'TYPE'  # We know types but not all values
        else:
            confidence = 'VALUE'  # We know exact values

        return Prediction(
            return_value=self.return_value,
            output=list(self.output),
            variables={k: v for k, v in self.env.items() if not k.startswith('_')},
            side_effects=self.side_effects,
            errors=self.errors,
            trace=self.trace,
            confidence=confidence,
            functions=list(self._functions.keys()),
        )

    def _exec_node(self, node: ast.AST):
        """Execute a single AST node."""
        line = getattr(node, 'lineno', None)

        if isinstance(node, ast.Assign):
            val = self._eval_expr(node.value)
            for target in node.targets:
                self._assign(target, val)
            self._trace(line, 'assign', {self._target_name(t): val for t in node.targets})

        elif isinstance(node, ast.AugAssign):
            current = self._eval_expr(node.target)
            val = self._eval_expr(node.value)
            op_type = type(node.op)
            if op_type in self.SAFE_OPS and current.is_concrete and val.is_concrete:
                try:
                    result = self.SAFE_OPS[op_type](current.value, val.value)
                    self._assign(node.target, SymValue(result))
                except Exception:
                    self._assign(node.target, SymValue(sym_type=current.type, symbolic=True))
            else:
                self._assign(node.target, SymValue(sym_type=current.type, symbolic=True))
            self._trace(line, 'aug_assign', {self._target_name(node.target): self.env.get(self._target_name(node.target))})

        elif isinstance(node, ast.Expr):
            val = self._eval_expr(node.value)
            self._trace(line, 'expr', val)

        elif isinstance(node, ast.If):
            cond = self._eval_expr(node.test)
            if cond.is_concrete:
                if cond.value:
                    for stmt in node.body:
                        self._exec_node(stmt)
                else:
                    for stmt in node.orelse:
                        self._exec_node(stmt)
            else:
                # Can't determine branch — mark affected variables as symbolic
                self._trace(line, 'if_symbolic', 'condition unknown')

        elif isinstance(node, ast.For):
            self._exec_for(node)

        elif isinstance(node, ast.While):
            self._exec_while(node)

        elif isinstance(node, ast.Return):
            if node.value:
                self.return_value = self._eval_expr(node.value)
            else:
                self.return_value = SymValue(None)
            raise _ReturnSignal()

        elif isinstance(node, ast.FunctionDef):
            self._functions[node.name] = node
            self.env[node.name] = SymValue(
                sym_type=SymType('function'), symbolic=True
            )

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name
                self.env[name] = SymValue(sym_type=SymType('module'), symbolic=True)
                self.side_effects.append(f'import {alias.name}')

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                self.env[name] = SymValue(sym_type=SymType('unknown'), symbolic=True)
                self.side_effects.append(f'from {node.module} import {alias.name}')

        elif isinstance(node, ast.Pass):
            pass

        elif isinstance(node, ast.Break):
            raise _BreakSignal()

        elif isinstance(node, ast.Continue):
            raise _ContinueSignal()

    def _exec_for(self, node: ast.For):
        """Execute a for loop with bounded iteration."""
        iterable = self._eval_expr(node.iter)

        if iterable.is_concrete and isinstance(iterable.value, (list, tuple, range, set)):
            items = list(iterable.value)
            count = 0
            for item in items:
                if count >= self.MAX_LOOP_ITERATIONS:
                    self.errors.append({
                        'type': 'LOOP_LIMIT',
                        'message': f'Loop exceeded {self.MAX_LOOP_ITERATIONS} iterations',
                        'line': node.lineno,
                    })
                    break
                self._assign(node.target, SymValue(item))
                try:
                    for stmt in node.body:
                        self._exec_node(stmt)
                except _BreakSignal:
                    break
                except _ContinueSignal:
                    pass
                count += 1
        else:
            # Can't iterate symbolically — mark loop variable as symbolic
            target_name = self._target_name(node.target)
            if iterable.type.element_type:
                self.env[target_name] = SymValue(
                    sym_type=SymType(iterable.type.element_type), symbolic=True
                )
            else:
                self.env[target_name] = SymValue(sym_type=SymType('unknown'), symbolic=True)
            self._trace(node.lineno, 'for_symbolic', f'iterating over {iterable.type}')

    def _exec_while(self, node: ast.While):
        """Execute a while loop with bounded iteration."""
        count = 0
        while count < self.MAX_LOOP_ITERATIONS:
            cond = self._eval_expr(node.test)
            if cond.is_concrete:
                if not cond.value:
                    break
            else:
                # Can't determine loop termination
                self._trace(node.lineno, 'while_symbolic', 'condition unknown')
                break
            try:
                for stmt in node.body:
                    self._exec_node(stmt)
            except _BreakSignal:
                break
            except _ContinueSignal:
                pass
            count += 1

        if count >= self.MAX_LOOP_ITERATIONS:
            self.errors.append({
                'type': 'LOOP_LIMIT',
                'message': f'While loop exceeded {self.MAX_LOOP_ITERATIONS} iterations',
                'line': node.lineno,
            })

    def _eval_expr(self, node: ast.AST) -> SymValue:
        """Evaluate an expression, returning concrete or symbolic value."""
        if isinstance(node, ast.Constant):
            return SymValue(node.value)

        elif isinstance(node, ast.Name):
            name = node.id
            if name in self.env:
                return self.env[name]
            # Builtins
            builtins_map = {'True': True, 'False': False, 'None': None}
            if name in builtins_map:
                return SymValue(builtins_map[name])
            return SymValue(sym_type=SymType('unknown'), symbolic=True)

        elif isinstance(node, ast.BinOp):
            left = self._eval_expr(node.left)
            right = self._eval_expr(node.right)
            op_type = type(node.op)

            if left.is_concrete and right.is_concrete and op_type in self.SAFE_OPS:
                try:
                    result = self.SAFE_OPS[op_type](left.value, right.value)
                    return SymValue(result)
                except Exception:
                    return SymValue(sym_type=left.type, symbolic=True)

            # Type inference for symbolic
            if isinstance(node.op, ast.Add):
                if left.type.name == 'str' or right.type.name == 'str':
                    return SymValue(sym_type=SymType('str'), symbolic=True)
                if left.type.name == 'list' or right.type.name == 'list':
                    return SymValue(sym_type=SymType('list'), symbolic=True)
            if isinstance(node.op, ast.Div):
                return SymValue(sym_type=SymType('float'), symbolic=True)
            return SymValue(sym_type=left.type, symbolic=True)

        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_expr(node.operand)
            if operand.is_concrete:
                if isinstance(node.op, ast.USub):
                    return SymValue(-operand.value)
                elif isinstance(node.op, ast.Not):
                    return SymValue(not operand.value)
                elif isinstance(node.op, ast.UAdd):
                    return SymValue(+operand.value)
            return SymValue(sym_type=operand.type, symbolic=True)

        elif isinstance(node, ast.BoolOp):
            values = [self._eval_expr(v) for v in node.values]
            if all(v.is_concrete for v in values):
                if isinstance(node.op, ast.And):
                    result = all(v.value for v in values)
                else:
                    result = any(v.value for v in values)
                return SymValue(result)
            return SymValue(sym_type=SymType('bool'), symbolic=True)

        elif isinstance(node, ast.Compare):
            left = self._eval_expr(node.left)
            if len(node.ops) == 1 and len(node.comparators) == 1:
                right = self._eval_expr(node.comparators[0])
                op_type = type(node.ops[0])
                if left.is_concrete and right.is_concrete and op_type in self.SAFE_OPS:
                    try:
                        result = self.SAFE_OPS[op_type](left.value, right.value)
                        return SymValue(result)
                    except Exception:
                        pass
            return SymValue(sym_type=SymType('bool'), symbolic=True)

        elif isinstance(node, ast.List):
            elements = [self._eval_expr(e) for e in node.elts]
            if all(e.is_concrete for e in elements):
                return SymValue([e.value for e in elements])
            elem_type = elements[0].type.name if elements else 'unknown'
            return SymValue(sym_type=SymType('list', elem_type, len(elements)), symbolic=True)

        elif isinstance(node, ast.Tuple):
            elements = [self._eval_expr(e) for e in node.elts]
            if all(e.is_concrete for e in elements):
                return SymValue(tuple(e.value for e in elements))
            return SymValue(sym_type=SymType('tuple', length=len(elements)), symbolic=True)

        elif isinstance(node, ast.Dict):
            keys = [self._eval_expr(k) if k else SymValue(sym_type=SymType('unknown'), symbolic=True) for k in node.keys]
            values = [self._eval_expr(v) for v in node.values]
            if all(k.is_concrete for k in keys) and all(v.is_concrete for v in values):
                return SymValue({k.value: v.value for k, v in zip(keys, values)})
            return SymValue(sym_type=SymType('dict', length=len(keys)), symbolic=True)

        elif isinstance(node, ast.Set):
            elements = [self._eval_expr(e) for e in node.elts]
            if all(e.is_concrete for e in elements):
                return SymValue({e.value for e in elements})
            return SymValue(sym_type=SymType('set', length=len(elements)), symbolic=True)

        elif isinstance(node, ast.Subscript):
            val = self._eval_expr(node.value)
            if isinstance(node.slice, ast.Constant) and val.is_concrete:
                try:
                    return SymValue(val.value[node.slice.value])
                except (IndexError, KeyError, TypeError):
                    self.errors.append({
                        'type': 'INDEX_ERROR',
                        'message': f'Index {node.slice.value} out of range',
                        'line': node.lineno,
                    })
            return SymValue(sym_type=SymType('unknown'), symbolic=True)

        elif isinstance(node, ast.Call):
            return self._eval_call(node)

        elif isinstance(node, ast.Attribute):
            val = self._eval_expr(node.value)
            # Common attribute lookups
            attr = node.attr
            if val.is_concrete and isinstance(val.value, str):
                if attr in ('upper', 'lower', 'strip', 'title'):
                    return SymValue(sym_type=SymType('function'), symbolic=True)
            return SymValue(sym_type=SymType('unknown'), symbolic=True)

        elif isinstance(node, ast.IfExp):
            cond = self._eval_expr(node.test)
            if cond.is_concrete:
                if cond.value:
                    return self._eval_expr(node.body)
                else:
                    return self._eval_expr(node.orelse)
            return SymValue(sym_type=SymType('unknown'), symbolic=True)

        elif isinstance(node, ast.ListComp):
            return self._eval_listcomp(node)

        elif isinstance(node, ast.JoinedStr):
            # f-string
            return SymValue(sym_type=SymType('str'), symbolic=True)

        elif isinstance(node, ast.FormattedValue):
            return SymValue(sym_type=SymType('str'), symbolic=True)

        return SymValue(sym_type=SymType('unknown'), symbolic=True)

    def _eval_call(self, node: ast.Call) -> SymValue:
        """Evaluate a function call."""
        func_name = ''
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr

        args = [self._eval_expr(a) for a in node.args]

        # Built-in functions we can simulate
        if func_name == 'print':
            parts = []
            for a in args:
                if a.is_concrete:
                    parts.append(str(a.value))
                else:
                    parts.append(f'<{a.type}>')
            self.output.append(' '.join(parts))
            return SymValue(None)

        elif func_name == 'len':
            if args and args[0].is_concrete:
                return SymValue(len(args[0].value))
            if args and args[0].type.length >= 0:
                return SymValue(args[0].type.length)
            return SymValue(sym_type=SymType('int'), symbolic=True)

        elif func_name == 'range':
            if all(a.is_concrete for a in args):
                vals = [a.value for a in args]
                return SymValue(list(range(*vals)))
            return SymValue(sym_type=SymType('list', 'int'), symbolic=True)

        elif func_name == 'int':
            if args and args[0].is_concrete:
                try:
                    return SymValue(int(args[0].value))
                except (ValueError, TypeError):
                    pass
            return SymValue(sym_type=SymType('int'), symbolic=True)

        elif func_name == 'float':
            if args and args[0].is_concrete:
                try:
                    return SymValue(float(args[0].value))
                except (ValueError, TypeError):
                    pass
            return SymValue(sym_type=SymType('float'), symbolic=True)

        elif func_name == 'str':
            if args and args[0].is_concrete:
                return SymValue(str(args[0].value))
            return SymValue(sym_type=SymType('str'), symbolic=True)

        elif func_name == 'list':
            if args and args[0].is_concrete:
                return SymValue(list(args[0].value))
            return SymValue(sym_type=SymType('list'), symbolic=True)

        elif func_name == 'sorted':
            if args and args[0].is_concrete:
                try:
                    return SymValue(sorted(args[0].value))
                except TypeError:
                    pass
            return SymValue(sym_type=SymType('list'), symbolic=True)

        elif func_name == 'reversed':
            if args and args[0].is_concrete:
                return SymValue(list(reversed(args[0].value)))
            return SymValue(sym_type=SymType('list'), symbolic=True)

        elif func_name == 'sum':
            if args and args[0].is_concrete:
                try:
                    return SymValue(sum(args[0].value))
                except TypeError:
                    pass
            return SymValue(sym_type=SymType('int'), symbolic=True)

        elif func_name == 'min':
            if all(a.is_concrete for a in args):
                return SymValue(min(a.value for a in args))
            return SymValue(sym_type=SymType('int'), symbolic=True)

        elif func_name == 'max':
            if all(a.is_concrete for a in args):
                return SymValue(max(a.value for a in args))
            return SymValue(sym_type=SymType('int'), symbolic=True)

        elif func_name == 'abs':
            if args and args[0].is_concrete:
                return SymValue(abs(args[0].value))
            return SymValue(sym_type=SymType('int'), symbolic=True)

        elif func_name == 'type':
            if args:
                return SymValue(sym_type=SymType('type'), symbolic=True)

        elif func_name == 'isinstance':
            return SymValue(sym_type=SymType('bool'), symbolic=True)

        elif func_name == 'enumerate':
            return SymValue(sym_type=SymType('list', 'tuple'), symbolic=True)

        elif func_name == 'zip':
            return SymValue(sym_type=SymType('list', 'tuple'), symbolic=True)

        elif func_name == 'map':
            return SymValue(sym_type=SymType('list'), symbolic=True)

        elif func_name == 'filter':
            return SymValue(sym_type=SymType('list'), symbolic=True)

        elif func_name in ('input', 'open'):
            self.side_effects.append(f'call to {func_name}()')
            return SymValue(sym_type=SymType('str'), symbolic=True)

        # User-defined functions
        if func_name in self._functions:
            return self._call_user_function(func_name, args)

        # Method calls on known types
        if isinstance(node.func, ast.Attribute):
            obj = self._eval_expr(node.func.value)
            return self._eval_method_call(obj, node.func.attr, args)

        return SymValue(sym_type=SymType('unknown'), symbolic=True)

    def _call_user_function(self, name: str, args: List[SymValue]) -> SymValue:
        """Call a user-defined function."""
        if self._call_depth >= 10:
            return SymValue(sym_type=SymType('unknown'), symbolic=True)

        func_node = self._functions[name]
        self._call_depth += 1

        # Save environment
        saved_env = dict(self.env)
        saved_return = self.return_value

        # Bind arguments
        for i, param in enumerate(func_node.args.args):
            param_name = param.arg
            if i < len(args):
                self.env[param_name] = args[i]
            elif func_node.args.defaults:
                # Default values
                default_idx = i - (len(func_node.args.args) - len(func_node.args.defaults))
                if default_idx >= 0:
                    self.env[param_name] = self._eval_expr(func_node.args.defaults[default_idx])
                else:
                    self.env[param_name] = SymValue(sym_type=SymType('unknown'), symbolic=True)

        # Execute function body
        self.return_value = None
        try:
            for stmt in func_node.body:
                self._exec_node(stmt)
        except _ReturnSignal:
            pass

        result = self.return_value or SymValue(None)

        # Restore environment
        self.env = saved_env
        self.return_value = saved_return
        self._call_depth -= 1

        return result

    def _eval_method_call(self, obj: SymValue, method: str, args: List[SymValue]) -> SymValue:
        """Evaluate a method call on a known object."""
        if obj.is_concrete:
            # String methods
            if isinstance(obj.value, str):
                if method == 'upper':
                    return SymValue(obj.value.upper())
                elif method == 'lower':
                    return SymValue(obj.value.lower())
                elif method == 'strip':
                    return SymValue(obj.value.strip())
                elif method == 'split':
                    sep = args[0].value if args and args[0].is_concrete else None
                    return SymValue(obj.value.split(sep))
                elif method == 'join':
                    if args and args[0].is_concrete:
                        return SymValue(obj.value.join(args[0].value))
                elif method == 'replace':
                    if len(args) >= 2 and all(a.is_concrete for a in args[:2]):
                        return SymValue(obj.value.replace(args[0].value, args[1].value))
                elif method == 'startswith':
                    if args and args[0].is_concrete:
                        return SymValue(obj.value.startswith(args[0].value))
                elif method == 'endswith':
                    if args and args[0].is_concrete:
                        return SymValue(obj.value.endswith(args[0].value))
                elif method == 'format':
                    return SymValue(sym_type=SymType('str'), symbolic=True)

            # List methods
            elif isinstance(obj.value, list):
                if method == 'append':
                    if args and args[0].is_concrete:
                        obj.value.append(args[0].value)
                    return SymValue(None)
                elif method == 'extend':
                    if args and args[0].is_concrete:
                        obj.value.extend(args[0].value)
                    return SymValue(None)
                elif method == 'pop':
                    if obj.value:
                        idx = args[0].value if args and args[0].is_concrete else -1
                        try:
                            return SymValue(obj.value.pop(idx))
                        except IndexError:
                            pass
                    return SymValue(sym_type=SymType('unknown'), symbolic=True)
                elif method == 'sort':
                    obj.value.sort()
                    return SymValue(None)
                elif method == 'reverse':
                    obj.value.reverse()
                    return SymValue(None)
                elif method == 'index':
                    if args and args[0].is_concrete:
                        try:
                            return SymValue(obj.value.index(args[0].value))
                        except ValueError:
                            pass
                elif method == 'count':
                    if args and args[0].is_concrete:
                        return SymValue(obj.value.count(args[0].value))

            # Dict methods
            elif isinstance(obj.value, dict):
                if method == 'keys':
                    return SymValue(list(obj.value.keys()))
                elif method == 'values':
                    return SymValue(list(obj.value.values()))
                elif method == 'items':
                    return SymValue(list(obj.value.items()))
                elif method == 'get':
                    if args and args[0].is_concrete:
                        default = args[1].value if len(args) > 1 and args[1].is_concrete else None
                        return SymValue(obj.value.get(args[0].value, default))

        # Type-based inference for symbolic objects
        if obj.type.name == 'str':
            if method in ('upper', 'lower', 'strip', 'title', 'replace', 'format'):
                return SymValue(sym_type=SymType('str'), symbolic=True)
            if method == 'split':
                return SymValue(sym_type=SymType('list', 'str'), symbolic=True)
            if method in ('startswith', 'endswith', 'isdigit', 'isalpha'):
                return SymValue(sym_type=SymType('bool'), symbolic=True)
            if method in ('find', 'index', 'count'):
                return SymValue(sym_type=SymType('int'), symbolic=True)
        elif obj.type.name == 'list':
            if method in ('append', 'extend', 'sort', 'reverse', 'clear', 'insert'):
                return SymValue(None)
            if method in ('index', 'count'):
                return SymValue(sym_type=SymType('int'), symbolic=True)
            if method == 'copy':
                return SymValue(sym_type=SymType('list'), symbolic=True)
            if method == 'pop':
                return SymValue(sym_type=SymType('unknown'), symbolic=True)
        elif obj.type.name == 'dict':
            if method in ('keys', 'values', 'items'):
                return SymValue(sym_type=SymType('list'), symbolic=True)
            if method == 'get':
                return SymValue(sym_type=SymType('unknown'), symbolic=True)

        return SymValue(sym_type=SymType('unknown'), symbolic=True)

    def _eval_listcomp(self, node: ast.ListComp) -> SymValue:
        """Evaluate a list comprehension."""
        if len(node.generators) != 1:
            return SymValue(sym_type=SymType('list'), symbolic=True)

        gen = node.generators[0]
        iterable = self._eval_expr(gen.iter)

        if not iterable.is_concrete:
            return SymValue(sym_type=SymType('list'), symbolic=True)

        try:
            items = list(iterable.value)
        except TypeError:
            return SymValue(sym_type=SymType('list'), symbolic=True)

        results = []
        target_name = self._target_name(gen.target)

        for item in items[:self.MAX_LOOP_ITERATIONS]:
            self.env[target_name] = SymValue(item)

            # Check conditions
            passes = True
            for cond in gen.ifs:
                c = self._eval_expr(cond)
                if c.is_concrete and not c.value:
                    passes = False
                    break
                elif c.symbolic:
                    return SymValue(sym_type=SymType('list'), symbolic=True)

            if passes:
                val = self._eval_expr(node.elt)
                if val.is_concrete:
                    results.append(val.value)
                else:
                    return SymValue(sym_type=SymType('list'), symbolic=True)

        return SymValue(results)

    def _assign(self, target: ast.AST, value: SymValue):
        """Assign a value to a target."""
        if isinstance(target, ast.Name):
            self.env[target.id] = value
        elif isinstance(target, ast.Tuple) or isinstance(target, ast.List):
            if value.is_concrete and isinstance(value.value, (tuple, list)):
                for t, v in zip(target.elts, value.value):
                    self._assign(t, SymValue(v))
            else:
                for t in target.elts:
                    self._assign(t, SymValue(sym_type=SymType('unknown'), symbolic=True))
        elif isinstance(target, ast.Subscript):
            # a[i] = val — modify in place if possible
            obj = self._eval_expr(target.value)
            if obj.is_concrete and isinstance(target.slice, ast.Constant):
                try:
                    obj.value[target.slice.value] = value.value if value.is_concrete else None
                except (IndexError, KeyError, TypeError):
                    pass

    def _target_name(self, target: ast.AST) -> str:
        """Get the name from an assignment target."""
        if isinstance(target, ast.Name):
            return target.id
        return '?'

    def _trace(self, line: int, kind: str, data: Any):
        """Record execution trace entry."""
        self.trace.append({'line': line, 'kind': kind, 'data': str(data)})


class Prediction:
    """Result of abstract interpretation."""

    def __init__(self, return_value: SymValue = None, output: List[str] = None,
                 variables: Dict[str, SymValue] = None,
                 side_effects: List[str] = None,
                 errors: List[Dict] = None,
                 trace: List[Dict] = None,
                 confidence: str = 'NONE',
                 functions: List[str] = None):
        self.return_value = return_value
        self.output = output or []
        self.variables = variables or {}
        self.side_effects = side_effects or []
        self.errors = errors or []
        self.trace = trace or []
        self.confidence = confidence  # NONE, LOW, TYPE, VALUE
        self.functions = functions or []

    @property
    def predicted_output(self) -> str:
        """What we predict the program will print."""
        return '\n'.join(self.output)

    @property
    def return_type(self) -> str:
        """Predicted return type."""
        if self.return_value:
            return str(self.return_value.type)
        return 'None'

    def summary(self) -> str:
        """Human-readable prediction summary."""
        lines = [f'Confidence: {self.confidence}']
        if self.output:
            lines.append(f'Predicted output:')
            for o in self.output:
                lines.append(f'  {o}')
        if self.return_value:
            lines.append(f'Return: {self.return_value}')
        if self.variables:
            lines.append('Variables:')
            for k, v in self.variables.items():
                lines.append(f'  {k} = {v}')
        if self.errors:
            lines.append('Predicted errors:')
            for e in self.errors:
                lines.append(f'  {e["type"]}: {e["message"]}')
        if self.side_effects:
            lines.append(f'Side effects: {", ".join(self.side_effects)}')
        return '\n'.join(lines)


# Signal exceptions for control flow
class _ReturnSignal(Exception):
    pass

class _BreakSignal(Exception):
    pass

class _ContinueSignal(Exception):
    pass
