"""
Config System — Persistent Configuration for FOSS-KI
======================================================
TOML-style config files without external dependencies.

Features:
  - Load/save config from files
  - Layered config: defaults → system → user → runtime
  - Type-safe access with defaults
  - Hot-reload on file change
  - Config validation
"""

import os
import json
import re
from typing import Dict, Any, Optional, List
from pathlib import Path


# === Default Configuration ===
DEFAULTS = {
    'system': {
        'name': 'FOSS-KI',
        'version': '0.3.0',
        'debug': False,
        'trace': False,
    },
    'knowledge': {
        'dim': 128,
        'max_facts': 100000,
        'anti_hallucination': True,
        'confidence_threshold': 0.5,
    },
    'ppm': {
        'order': 5,
        'smoothing': 'escape',
        'model_path': '',
    },
    'language': {
        'default': 'en',
        'supported': ['en', 'de'],
    },
    'web': {
        'enabled': True,
        'timeout': 15,
        'cache_ttl': 300,
        'rate_limit': 1.0,
        'max_response_size': 1000000,
        'fallback_on_no_answer': True,
    },
    'tools': {
        'confirm_dangerous': True,
        'enabled_categories': ['file', 'search', 'math', 'web', 'data'],
    },
    'conversation': {
        'window_size': 20,
        'compress_after': 10,
        'persist_to_knowledge': True,
    },
    'execution': {
        'timeout': 10,
        'max_retries': 3,
        'sandbox': True,
    },
    'embeddings': {
        'dim': 64,
        'max_vocab': 5000,
    },
    'scheduler': {
        'max_concurrent': 4,
        'default_retry_delay': 1.0,
    },
    'watcher': {
        'poll_interval': 2.0,
        'debounce': 0.5,
        'recursive': True,
    },
}


class Config:
    """
    Layered configuration system.

    Priority: runtime > user > system > defaults

    Usage:
        cfg = Config()
        cfg.load('~/.foss-ki/config.json')
        dim = cfg.get('knowledge.dim', 128)
        cfg.set('web.timeout', 30)
        cfg.save()
    """

    def __init__(self, config_path: Optional[str] = None):
        self._data: Dict[str, Any] = {}
        self._runtime: Dict[str, Any] = {}
        self._path = config_path
        self._load_defaults()

    def _load_defaults(self):
        """Load default configuration."""
        self._data = _deep_copy(DEFAULTS)

    def load(self, path: str) -> bool:
        """Load configuration from a JSON file."""
        p = Path(path).expanduser()
        if not p.exists():
            return False

        try:
            with open(p, 'r') as f:
                user_config = json.load(f)
            _deep_merge(self._data, user_config)
            self._path = str(p)
            return True
        except (json.JSONDecodeError, OSError):
            return False

    def save(self, path: Optional[str] = None) -> bool:
        """Save current configuration to file."""
        p = Path(path or self._path or '~/.foss-ki/config.json').expanduser()
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, 'w') as f:
                json.dump(self._data, f, indent=2)
            return True
        except OSError:
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a config value by dotted key path.

        Args:
            key: e.g., 'knowledge.dim', 'web.timeout'
            default: fallback value

        Returns config value or default.
        """
        # Check runtime overrides first
        if key in self._runtime:
            return self._runtime[key]

        # Navigate nested dict
        parts = key.split('.')
        current = self._data
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return default
        return current

    def set(self, key: str, value: Any, runtime_only: bool = False):
        """
        Set a config value.

        Args:
            key: dotted key path
            value: new value
            runtime_only: if True, don't persist to file
        """
        if runtime_only:
            self._runtime[key] = value
            return

        parts = key.split('.')
        current = self._data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get an entire config section."""
        return dict(self._data.get(section, {}))

    def to_dict(self) -> Dict[str, Any]:
        """Get full config as dict."""
        return _deep_copy(self._data)

    def validate(self) -> List[str]:
        """Validate configuration. Returns list of errors (empty = valid)."""
        errors = []

        dim = self.get('knowledge.dim')
        if not isinstance(dim, int) or dim < 8:
            errors.append(f"knowledge.dim must be int >= 8, got {dim}")

        timeout = self.get('web.timeout')
        if not isinstance(timeout, (int, float)) or timeout < 1:
            errors.append(f"web.timeout must be >= 1, got {timeout}")

        order = self.get('ppm.order')
        if not isinstance(order, int) or order < 1 or order > 10:
            errors.append(f"ppm.order must be 1-10, got {order}")

        return errors

    def __repr__(self):
        return f"Config({len(self._data)} sections)"


# === Helpers ===

def _deep_copy(d: Dict) -> Dict:
    """Deep copy a nested dict."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _deep_merge(base: Dict, override: Dict):
    """Recursively merge override into base."""
    for key, value in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(value, dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
