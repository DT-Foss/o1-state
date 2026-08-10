"""
Tool-Use Framework — External Action Execution
================================================
Gives the FOSS-KI system the ability to execute actions
beyond Q&A: file operations, shell commands, web search,
calculations, and custom tools.

Architecture:
  1. Tool Registry: register callable tools with descriptions
  2. Intent Parser: detect when a query needs a tool
  3. Argument Extractor: pull args from natural language
  4. Executor: run the tool and return structured results
  5. Safety: whitelist/sandbox, confirmation for dangerous ops

This is what turns FOSS-KI from a knowledge system into an agent.
"""

import os
import re
import subprocess
import json
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path


class Tool:
    """A registered tool that the system can invoke."""

    def __init__(self, name: str, description: str, fn: Callable,
                 args: Optional[List[Dict]] = None,
                 dangerous: bool = False, category: str = 'general'):
        """
        Args:
            name: tool identifier (e.g., 'read_file', 'shell')
            description: natural language description for intent matching
            fn: the callable to execute
            args: list of {name, type, description, required} dicts
            dangerous: if True, requires confirmation before execution
            category: tool category for grouping
        """
        self.name = name
        self.description = description
        self.fn = fn
        self.args = args or []
        self.dangerous = dangerous
        self.category = category

    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the tool with given arguments."""
        try:
            result = self.fn(**kwargs)
            return {
                'success': True,
                'result': result,
                'tool': self.name,
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'tool': self.name,
            }


class ToolRegistry:
    """
    Registry of available tools with intent matching.
    """

    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_builtins()

    def register(self, tool: Tool):
        """Register a tool."""
        self.tools[tool.name] = tool

    def get(self, name: str) -> Optional[Tool]:
        return self.tools.get(name)

    def list_tools(self) -> List[Dict]:
        """List all registered tools."""
        return [
            {
                'name': t.name,
                'description': t.description,
                'category': t.category,
                'dangerous': t.dangerous,
                'args': t.args,
            }
            for t in self.tools.values()
        ]

    def match_intent(self, query: str) -> Optional[str]:
        """
        Match a query to the best tool using keyword matching.

        Returns tool name or None.
        """
        q = query.lower()

        # Explicit tool invocation: "use tool X" or "run X"
        m = re.search(r'(?:use\s+tool|run|execute)\s+(\w+)', q)
        if m:
            name = m.group(1)
            if name in self.tools:
                return name

        # Keyword matching against tool descriptions
        best_score = 0
        best_tool = None

        for name, tool in self.tools.items():
            score = 0
            desc_words = set(tool.description.lower().split())
            query_words = set(q.split())

            # Word overlap
            overlap = desc_words & query_words
            score = len(overlap)

            # Boost for category keywords
            if tool.category == 'file' and any(w in q for w in ('file', 'read', 'write', 'open', 'save', 'create')):
                score += 3
            if tool.category == 'shell' and any(w in q for w in ('run', 'execute', 'command', 'shell', 'terminal')):
                score += 3
            if tool.category == 'math' and any(w in q for w in ('calculate', 'compute', 'solve', 'math')):
                score += 3
            if tool.name == 'grep' and 'grep' in q:
                score += 5  # Explicit "grep" keyword → strong match
            if tool.category == 'search' and any(w in q for w in ('search', 'find', 'look')):
                score += 3
            if tool.category == 'web' and any(w in q for w in ('web', 'url', 'fetch', 'download', 'http')):
                score += 3

            if score > best_score:
                best_score = score
                best_tool = name

        return best_tool if best_score >= 2 else None

    def _register_builtins(self):
        """Register built-in tools."""

        # File reading
        self.register(Tool(
            name='read_file',
            description='Read contents of a file from disk',
            fn=_tool_read_file,
            args=[{'name': 'path', 'type': 'str', 'description': 'file path', 'required': True}],
            category='file',
        ))

        # File writing
        self.register(Tool(
            name='write_file',
            description='Write content to a file on disk',
            fn=_tool_write_file,
            args=[
                {'name': 'path', 'type': 'str', 'description': 'file path', 'required': True},
                {'name': 'content', 'type': 'str', 'description': 'file content', 'required': True},
            ],
            dangerous=True,
            category='file',
        ))

        # List directory
        self.register(Tool(
            name='list_dir',
            description='List files and directories in a path',
            fn=_tool_list_dir,
            args=[{'name': 'path', 'type': 'str', 'description': 'directory path', 'required': False}],
            category='file',
        ))

        # Shell command
        self.register(Tool(
            name='shell',
            description='Run a shell command and return output',
            fn=_tool_shell,
            args=[{'name': 'command', 'type': 'str', 'description': 'shell command', 'required': True}],
            dangerous=True,
            category='shell',
        ))

        # Search files
        self.register(Tool(
            name='search_files',
            description='Search for files matching a pattern using glob',
            fn=_tool_search_files,
            args=[
                {'name': 'pattern', 'type': 'str', 'description': 'glob pattern', 'required': True},
                {'name': 'path', 'type': 'str', 'description': 'search root', 'required': False},
            ],
            category='search',
        ))

        # Grep
        self.register(Tool(
            name='grep',
            description='Search file contents for a pattern',
            fn=_tool_grep,
            args=[
                {'name': 'pattern', 'type': 'str', 'description': 'regex pattern', 'required': True},
                {'name': 'path', 'type': 'str', 'description': 'file or directory', 'required': False},
            ],
            category='search',
        ))

        # JSON parse
        self.register(Tool(
            name='parse_json',
            description='Parse and query JSON data',
            fn=_tool_parse_json,
            args=[
                {'name': 'data', 'type': 'str', 'description': 'JSON string', 'required': True},
                {'name': 'path', 'type': 'str', 'description': 'JSON path like .key.subkey', 'required': False},
            ],
            category='data',
        ))

        # Python eval (sandboxed)
        self.register(Tool(
            name='python_eval',
            description='Evaluate a Python expression safely',
            fn=_tool_python_eval,
            args=[{'name': 'expression', 'type': 'str', 'description': 'Python expression', 'required': True}],
            category='math',
        ))


class ToolExecutor:
    """
    Orchestrates tool invocation from natural language queries.

    Detects tool intent, extracts arguments, executes, returns results.
    """

    def __init__(self, registry: Optional[ToolRegistry] = None,
                 confirm_dangerous: bool = True):
        self.registry = registry or ToolRegistry()
        self.confirm_dangerous = confirm_dangerous
        self.history = []  # Execution history

    def should_use_tool(self, query: str) -> Optional[str]:
        """Check if a query should invoke a tool."""
        return self.registry.match_intent(query)

    def execute(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with explicit arguments."""
        tool = self.registry.get(tool_name)
        if not tool:
            return {'success': False, 'error': f'Unknown tool: {tool_name}'}

        result = tool.execute(**kwargs)
        self.history.append({
            'tool': tool_name,
            'args': kwargs,
            'result': result,
        })
        return result

    def execute_from_query(self, query: str) -> Dict[str, Any]:
        """
        Parse a query, detect tool and args, execute.

        Returns:
            dict with success, result/error, tool, args_extracted
        """
        tool_name = self.should_use_tool(query)
        if not tool_name:
            return {'success': False, 'error': 'No matching tool found'}

        tool = self.registry.get(tool_name)
        args = self._extract_args(query, tool)

        if tool.dangerous and self.confirm_dangerous:
            return {
                'success': False,
                'needs_confirmation': True,
                'tool': tool_name,
                'args': args,
                'message': f"Tool '{tool_name}' is dangerous. Confirm execution?",
            }

        result = tool.execute(**args)
        self.history.append({
            'tool': tool_name,
            'args': args,
            'result': result,
        })
        return result

    def _extract_args(self, query: str, tool: Tool) -> Dict[str, str]:
        """Extract arguments from natural language for a tool."""
        args = {}

        # File path extraction
        path_match = re.search(r'["\']([^"\']+)["\']', query)
        if not path_match:
            path_match = re.search(r'(?:file|path|in)\s+(\S+)', query)
        if not path_match:
            # Try to find a path-like string
            path_match = re.search(r'(/[\w./\-]+|\.[\w./\-]+)', query)

        if path_match:
            for arg_def in tool.args:
                if arg_def['name'] in ('path', 'file'):
                    args[arg_def['name']] = path_match.group(1)
                    break

        # Command extraction (everything after "run" or "execute")
        cmd_match = re.search(r'(?:run|execute|command)\s+["\']?(.+?)["\']?$', query, re.I)
        if cmd_match:
            args['command'] = cmd_match.group(1).strip()

        # Pattern extraction
        pat_match = re.search(r'(?:pattern|for|matching)\s+["\']?(.+?)["\']?(?:\s+in|\s*$)', query, re.I)
        if pat_match:
            args['pattern'] = pat_match.group(1).strip()

        # Content extraction (everything in quotes after "write")
        content_match = re.search(r'(?:write|content)\s+["\'](.+?)["\']', query, re.I)
        if content_match:
            args['content'] = content_match.group(1)

        # Expression extraction
        expr_match = re.search(r'(?:calculate|compute|evaluate|eval)\s+(.+)', query, re.I)
        if expr_match:
            args['expression'] = expr_match.group(1).strip()

        return args


# === Built-in Tool Implementations ===

def _tool_read_file(path: str) -> str:
    """Read file contents."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if p.stat().st_size > 1_000_000:
        # Read first 10K chars for large files
        with open(p, 'r', encoding='utf-8', errors='replace') as f:
            return f.read(10000) + "\n... (truncated, file is large)"
    with open(p, 'r', encoding='utf-8', errors='replace') as f:
        return f.read()


def _tool_write_file(path: str, content: str) -> str:
    """Write content to file."""
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, 'w', encoding='utf-8') as f:
        f.write(content)
    return f"Written {len(content)} chars to {path}"


def _tool_list_dir(path: str = '.') -> List[str]:
    """List directory contents."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {path}")
    entries = []
    for entry in sorted(p.iterdir()):
        suffix = '/' if entry.is_dir() else ''
        entries.append(f"{entry.name}{suffix}")
    return entries


def _tool_shell(command: str, timeout: int = 30) -> str:
    """Run shell command (sandboxed with timeout)."""
    # Block obviously dangerous commands
    dangerous = ['rm -rf /', 'mkfs', 'dd if=/dev/zero', ':(){', 'fork bomb']
    for d in dangerous:
        if d in command:
            raise PermissionError(f"Blocked dangerous command: {command}")

    result = subprocess.run(
        command, shell=True, capture_output=True, text=True,
        timeout=timeout
    )
    output = result.stdout
    if result.stderr:
        output += f"\nSTDERR: {result.stderr}"
    if result.returncode != 0:
        output += f"\n(exit code: {result.returncode})"
    return output.strip()


def _tool_search_files(pattern: str, path: str = '.') -> List[str]:
    """Search for files matching glob pattern."""
    p = Path(path).expanduser()
    matches = list(p.glob(pattern))
    return [str(m) for m in matches[:100]]  # Limit to 100 results


def _tool_grep(pattern: str, path: str = '.') -> List[Dict]:
    """Search file contents for a regex pattern."""
    p = Path(path).expanduser()
    results = []

    if p.is_file():
        files = [p]
    else:
        files = list(p.rglob('*.py'))[:50]  # Limit to 50 Python files

    for f in files:
        try:
            with open(f, 'r', encoding='utf-8', errors='replace') as fh:
                for i, line in enumerate(fh, 1):
                    if re.search(pattern, line):
                        results.append({
                            'file': str(f),
                            'line': i,
                            'text': line.strip(),
                        })
                        if len(results) >= 50:
                            return results
        except (OSError, UnicodeDecodeError):
            continue

    return results


def _tool_parse_json(data: str, path: str = None) -> Any:
    """Parse JSON and optionally extract a path."""
    parsed = json.loads(data)
    if path:
        for key in path.strip('.').split('.'):
            if isinstance(parsed, dict):
                parsed = parsed.get(key)
            elif isinstance(parsed, list) and key.isdigit():
                parsed = parsed[int(key)]
            else:
                return None
    return parsed


def _tool_python_eval(expression: str) -> str:
    """Safely evaluate a Python expression."""
    import math as _math
    safe_globals = {
        '__builtins__': {},
        'abs': abs, 'round': round, 'min': min, 'max': max,
        'sum': sum, 'len': len, 'sorted': sorted, 'range': range,
        'int': int, 'float': float, 'str': str, 'list': list,
        'math': _math, 'pi': _math.pi, 'e': _math.e,
        'sqrt': _math.sqrt, 'log': _math.log, 'sin': _math.sin,
        'cos': _math.cos, 'tan': _math.tan,
    }
    result = eval(expression, safe_globals, {})
    return str(result)
