"""
Plugin System — Dynamic Module Loading
=========================================
Allows extending FOSS-KI with external plugins.

Plugins are Python files in the plugins/ directory with a
register() function that receives the REPL instance.

Example plugin (plugins/hello.py):

    def register(repl):
        def hello_command(arg):
            return f"Hello, {arg or 'World'}!"

        repl.register_command('/hello', hello_command, "Say hello")
"""

import os
import importlib
import importlib.util
from typing import Dict, Any, List, Optional, Callable, Tuple


class PluginInfo:
    """Metadata about a loaded plugin."""

    def __init__(self, name: str, path: str):
        self.name = name
        self.path = path
        self.module: Any = None
        self.version = '0.0.0'
        self.description = ''
        self.author = ''
        self.loaded = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'path': self.path,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'loaded': self.loaded,
        }


class PluginManager:
    """Manages plugin discovery, loading, and lifecycle."""

    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.plugin_dirs = plugin_dirs or []
        self.plugins: Dict[str, PluginInfo] = {}
        self.hooks: Dict[str, List[Callable]] = {}
        self._commands: Dict[str, Tuple[Callable, str]] = {}

    def add_plugin_dir(self, path: str):
        """Add a directory to search for plugins."""
        path = os.path.abspath(path)
        if path not in self.plugin_dirs:
            self.plugin_dirs.append(path)

    def discover(self) -> List[str]:
        """Discover all plugins in plugin directories."""
        discovered = []
        for plugin_dir in self.plugin_dirs:
            if not os.path.isdir(plugin_dir):
                continue

            for filename in sorted(os.listdir(plugin_dir)):
                if filename.endswith('.py') and not filename.startswith('_'):
                    name = filename[:-3]
                    path = os.path.join(plugin_dir, filename)
                    if name not in self.plugins:
                        self.plugins[name] = PluginInfo(name, path)
                        discovered.append(name)

        return discovered

    def load(self, name: str, context: Any = None) -> bool:
        """Load a specific plugin by name."""
        info = self.plugins.get(name)
        if not info:
            return False

        if info.loaded:
            return True

        try:
            spec = importlib.util.spec_from_file_location(
                f"foss_ki_plugin_{name}", info.path
            )
            if spec is None or spec.loader is None:
                return False

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            info.module = module
            info.version = getattr(module, '__version__', '0.0.0')
            info.description = getattr(module, '__description__', '')
            info.author = getattr(module, '__author__', '')

            # Call register() if available
            if hasattr(module, 'register') and context is not None:
                module.register(context)

            info.loaded = True
            return True

        except Exception:
            info.loaded = False
            return False

    def load_all(self, context: Any = None) -> Dict[str, bool]:
        """Load all discovered plugins."""
        results = {}
        for name in self.plugins:
            results[name] = self.load(name, context)
        return results

    def unload(self, name: str) -> bool:
        """Unload a plugin."""
        info = self.plugins.get(name)
        if not info or not info.loaded:
            return False

        # Call unregister() if available
        if info.module and hasattr(info.module, 'unregister'):
            try:
                info.module.unregister()
            except Exception:
                pass

        info.loaded = False
        info.module = None
        return True

    def reload(self, name: str, context: Any = None) -> bool:
        """Reload a plugin."""
        self.unload(name)
        return self.load(name, context)

    # === Hook system ===

    def register_hook(self, event: str, callback: Callable):
        """Register a callback for an event."""
        if event not in self.hooks:
            self.hooks[event] = []
        self.hooks[event].append(callback)

    def emit(self, event: str, **kwargs) -> List[Any]:
        """Emit an event and collect results from all hooks."""
        results = []
        for callback in self.hooks.get(event, []):
            try:
                result = callback(**kwargs)
                results.append(result)
            except Exception:
                pass
        return results

    # === Command registration ===

    def register_command(self, name: str, handler: Callable, description: str = ''):
        """Register a plugin command."""
        self._commands[name] = (handler, description)

    def get_command(self, name: str) -> Optional[Tuple[Callable, str]]:
        """Get a registered command handler."""
        return self._commands.get(name)

    def list_commands(self) -> List[Dict[str, str]]:
        """List all plugin commands."""
        return [
            {'name': name, 'description': desc}
            for name, (_, desc) in sorted(self._commands.items())
        ]

    # === Info ===

    def list_plugins(self) -> List[Dict[str, Any]]:
        """List all plugins with their status."""
        return [info.to_dict() for info in self.plugins.values()]

    def get_plugin(self, name: str) -> Optional[Dict[str, Any]]:
        """Get info about a specific plugin."""
        info = self.plugins.get(name)
        return info.to_dict() if info else None

    def stats(self) -> Dict[str, int]:
        """Get plugin system statistics."""
        total = len(self.plugins)
        loaded = sum(1 for p in self.plugins.values() if p.loaded)
        return {
            'total': total,
            'loaded': loaded,
            'hooks': sum(len(h) for h in self.hooks.values()),
            'commands': len(self._commands),
        }
