"""
Tool Plugin Loader — Auto-Discovery for FOSS-KI Tools
=======================================================
Scans the tools/ directory, loads all tool modules, and provides
a registry for the Agent to dispatch tool calls.

Each tool module must have:
  - description: str
  - keywords: list[str]
  - patterns: list[str] (regex patterns)
  - execute(args: dict) -> dict

Self-extension: new tools can be added by dropping .py files in tools/.
The system reloads on demand — no restart needed.
"""

import os
import re
import importlib
import importlib.util
from typing import Dict, Any, List, Optional, Tuple


class ToolRegistry:
    """
    Auto-discovers and manages tool plugins.

    Drop a .py file in tools/ → it's available to the Agent.
    """

    def __init__(self, tools_dir=None):
        if tools_dir is None:
            tools_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                'tools'
            )
        self.tools_dir = tools_dir
        self.tools = {}  # name → module
        self._load_all()

    def _load_all(self):
        """Scan tools/ and load all tool modules."""
        if not os.path.isdir(self.tools_dir):
            return

        for filename in os.listdir(self.tools_dir):
            if filename.endswith('.py') and not filename.startswith('_'):
                name = filename[:-3]
                self._load_tool(name)

    def _load_tool(self, name: str):
        """Load a single tool module."""
        path = os.path.join(self.tools_dir, f"{name}.py")
        if not os.path.exists(path):
            return

        try:
            spec = importlib.util.spec_from_file_location(f"tools.{name}", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            # Validate required attributes
            if not hasattr(module, 'execute') or not callable(module.execute):
                return
            if not hasattr(module, 'description'):
                return

            self.tools[name] = module
        except Exception:
            pass  # Skip broken tools silently

    def reload(self):
        """Reload all tools (for runtime extension)."""
        self.tools.clear()
        self._load_all()

    def match(self, text: str) -> Optional[Tuple[str, Dict[str, str]]]:
        """
        Match user input against tool patterns.

        Returns: (tool_name, extracted_args) or None
        """
        text_lower = text.lower().strip()

        best_match = None
        best_priority = -1

        for name, module in self.tools.items():
            # Check keywords
            keywords = getattr(module, 'keywords', [])
            keyword_score = sum(1 for kw in keywords if kw in text_lower)

            if keyword_score == 0:
                continue

            # Try patterns to extract args
            patterns = getattr(module, 'patterns', [])
            args = {}

            for pattern in patterns:
                m = re.search(pattern, text, re.I)
                if m:
                    # Use first capture group as the main argument
                    groups = m.groups()
                    if groups:
                        # Map to common arg names based on tool type
                        if 'path' in pattern or 'file' in name:
                            args['path'] = groups[0]
                        elif 'command' in pattern or name == 'shell':
                            args['command'] = groups[0]
                            if len(groups) > 1:
                                args['command'] += ' ' + groups[1]
                        else:
                            args['input'] = groups[0]
                    keyword_score += 2  # Pattern match bonus
                    break

            if keyword_score > best_priority:
                best_priority = keyword_score
                best_match = (name, args)

        return best_match

    def execute(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool by name."""
        if name not in self.tools:
            return {'success': False, 'response': f"Unknown tool: {name}"}

        try:
            return self.tools[name].execute(args)
        except Exception as e:
            return {'success': False, 'response': f"Tool {name} error: {e}"}

    def list_tools(self) -> List[Dict[str, str]]:
        """List all available tools."""
        result = []
        for name, module in sorted(self.tools.items()):
            result.append({
                'name': name,
                'description': getattr(module, 'description', ''),
                'keywords': getattr(module, 'keywords', []),
            })
        return result

    def describe(self) -> str:
        """Human-readable tool listing."""
        tools = self.list_tools()
        if not tools:
            return "No tools available."
        lines = ["Available tools:"]
        for t in tools:
            lines.append(f"  - {t['name']}: {t['description']}")
        return '\n'.join(lines)
