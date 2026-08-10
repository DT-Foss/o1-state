"""
File Watcher — Reactive Filesystem Monitoring
===============================================
Watch directories for changes and trigger actions.

Capabilities:
  - Directory monitoring (polling-based, no inotify dependency)
  - Pattern matching (glob-style)
  - Action dispatch on create/modify/delete
  - Debouncing (ignore rapid-fire saves)
  - Change history tracking

This gives FOSS-KI awareness of its environment.
When a file changes, the system can react: re-read knowledge,
re-train models, update caches, trigger pipelines.
"""

import os
import time
import fnmatch
import hashlib
import threading
from typing import Dict, Any, List, Optional, Callable, Set
from pathlib import Path
from enum import Enum


class ChangeType(Enum):
    CREATED = 'created'
    MODIFIED = 'modified'
    DELETED = 'deleted'


class FileEvent:
    """A detected file change."""

    def __init__(self, path: str, change_type: ChangeType,
                 size: int = 0, mtime: float = 0):
        self.path = path
        self.change_type = change_type
        self.size = size
        self.mtime = mtime
        self.timestamp = time.time()

    def __repr__(self):
        return f"FileEvent({self.change_type.value}: {self.path})"


class FileWatcher:
    """
    Poll-based file watcher with pattern matching and debouncing.

    Usage:
        watcher = FileWatcher('/path/to/watch')
        watcher.add_pattern('*.py', on_python_change)
        watcher.add_pattern('*.json', on_json_change)
        watcher.start()  # Background thread
    """

    def __init__(self, watch_dir: str, poll_interval: float = 2.0,
                 debounce: float = 0.5, recursive: bool = True,
                 ignore_patterns: Optional[List[str]] = None):
        """
        Args:
            watch_dir: directory to monitor
            poll_interval: seconds between polls
            debounce: ignore changes within this window
            recursive: watch subdirectories
            ignore_patterns: glob patterns to ignore
        """
        self.watch_dir = Path(watch_dir).resolve()
        self.poll_interval = poll_interval
        self.debounce = debounce
        self.recursive = recursive
        self.ignore_patterns = ignore_patterns or [
            '__pycache__', '*.pyc', '.git', '.DS_Store',
            '*.swp', '*.swo', '*~', '.idea', '.vscode',
        ]

        self._snapshot: Dict[str, Dict] = {}  # path → {mtime, size, hash}
        self._handlers: List[Dict] = []  # {pattern, callback, change_types}
        self._history: List[FileEvent] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._last_events: Dict[str, float] = {}  # path → last_event_time

    def add_pattern(self, pattern: str, callback: Callable,
                    change_types: Optional[List[ChangeType]] = None):
        """
        Register a callback for files matching a glob pattern.

        Args:
            pattern: glob pattern (e.g., '*.py', 'data/*.json')
            callback: fn(event: FileEvent) to call
            change_types: which change types to react to (default: all)
        """
        self._handlers.append({
            'pattern': pattern,
            'callback': callback,
            'change_types': change_types or list(ChangeType),
        })

    def add_callback(self, callback: Callable,
                     change_types: Optional[List[ChangeType]] = None):
        """Register a callback for ALL file changes."""
        self.add_pattern('*', callback, change_types)

    def scan(self) -> List[FileEvent]:
        """
        Scan for changes since last scan.
        Returns list of FileEvents.
        """
        events = []
        current_files: Dict[str, Dict] = {}
        now = time.time()

        # Walk directory
        if self.recursive:
            for root, dirs, files in os.walk(self.watch_dir):
                # Filter ignored directories
                dirs[:] = [d for d in dirs if not self._should_ignore(d)]
                for f in files:
                    if self._should_ignore(f):
                        continue
                    path = os.path.join(root, f)
                    info = self._file_info(path)
                    if info:
                        current_files[path] = info
        else:
            for f in os.listdir(self.watch_dir):
                if self._should_ignore(f):
                    continue
                path = os.path.join(str(self.watch_dir), f)
                if os.path.isfile(path):
                    info = self._file_info(path)
                    if info:
                        current_files[path] = info

        # Detect changes
        # New files
        for path, info in current_files.items():
            if path not in self._snapshot:
                event = FileEvent(path, ChangeType.CREATED, info['size'], info['mtime'])
                if self._should_fire(path, now):
                    events.append(event)

        # Modified files
        for path, info in current_files.items():
            if path in self._snapshot:
                old = self._snapshot[path]
                if info['mtime'] != old['mtime'] or info['size'] != old['size']:
                    event = FileEvent(path, ChangeType.MODIFIED, info['size'], info['mtime'])
                    if self._should_fire(path, now):
                        events.append(event)

        # Deleted files
        for path in self._snapshot:
            if path not in current_files:
                event = FileEvent(path, ChangeType.DELETED)
                if self._should_fire(path, now):
                    events.append(event)

        # Update snapshot
        self._snapshot = current_files

        # Dispatch events
        for event in events:
            self._dispatch(event)
            self._history.append(event)

        # Trim history
        if len(self._history) > 1000:
            self._history = self._history[-500:]

        return events

    def start(self):
        """Start background monitoring thread."""
        if self._running:
            return
        self._running = True
        # Take initial snapshot without firing events
        self._take_snapshot()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def get_history(self, last_n: int = 20) -> List[Dict]:
        """Get recent change history."""
        return [
            {
                'path': e.path,
                'type': e.change_type.value,
                'time': e.timestamp,
                'size': e.size,
            }
            for e in self._history[-last_n:]
        ]

    def get_stats(self) -> Dict:
        """Get watcher statistics."""
        return {
            'watch_dir': str(self.watch_dir),
            'tracked_files': len(self._snapshot),
            'total_events': len(self._history),
            'handlers': len(self._handlers),
            'running': self._running,
        }

    # === Internal ===

    def _take_snapshot(self):
        """Take initial snapshot without firing events."""
        if self.recursive:
            for root, dirs, files in os.walk(self.watch_dir):
                dirs[:] = [d for d in dirs if not self._should_ignore(d)]
                for f in files:
                    if self._should_ignore(f):
                        continue
                    path = os.path.join(root, f)
                    info = self._file_info(path)
                    if info:
                        self._snapshot[path] = info
        else:
            for f in os.listdir(self.watch_dir):
                if self._should_ignore(f):
                    continue
                path = os.path.join(str(self.watch_dir), f)
                if os.path.isfile(path):
                    info = self._file_info(path)
                    if info:
                        self._snapshot[path] = info

    def _poll_loop(self):
        """Background polling loop."""
        while self._running:
            try:
                self.scan()
            except Exception:
                pass
            time.sleep(self.poll_interval)

    def _dispatch(self, event: FileEvent):
        """Dispatch event to matching handlers."""
        filename = os.path.basename(event.path)
        rel_path = os.path.relpath(event.path, str(self.watch_dir))

        for handler in self._handlers:
            if event.change_type not in handler['change_types']:
                continue
            pattern = handler['pattern']
            if fnmatch.fnmatch(filename, pattern) or fnmatch.fnmatch(rel_path, pattern):
                try:
                    handler['callback'](event)
                except Exception:
                    pass  # Don't crash on handler errors

    def _should_ignore(self, name: str) -> bool:
        """Check if a file/dir should be ignored."""
        return any(fnmatch.fnmatch(name, p) for p in self.ignore_patterns)

    def _should_fire(self, path: str, now: float) -> bool:
        """Debounce check."""
        last = self._last_events.get(path, 0)
        if now - last < self.debounce:
            return False
        self._last_events[path] = now
        return True

    @staticmethod
    def _file_info(path: str) -> Optional[Dict]:
        """Get file metadata."""
        try:
            stat = os.stat(path)
            return {
                'mtime': stat.st_mtime,
                'size': stat.st_size,
            }
        except (OSError, PermissionError):
            return None
