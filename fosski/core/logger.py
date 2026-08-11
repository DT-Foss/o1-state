"""
Structured Logger — JSON + Text Logging with Rotation
=======================================================
No external dependencies. Replaces Python's logging module
with structured output, rotation, and query support.

Features:
  1. JSON and text output formats
  2. Log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
  3. File rotation by size
  4. Context fields (user, session, module)
  5. Log querying (filter by level, time, pattern)
  6. Performance timing decorator
"""

import os
import json
import time
import functools
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable


# Log levels
DEBUG = 10
INFO = 20
WARNING = 30
ERROR = 40
CRITICAL = 50

LEVEL_NAMES = {
    DEBUG: 'DEBUG',
    INFO: 'INFO',
    WARNING: 'WARNING',
    ERROR: 'ERROR',
    CRITICAL: 'CRITICAL',
}

NAME_TO_LEVEL = {v: k for k, v in LEVEL_NAMES.items()}


class LogEntry:
    """A single log entry."""

    __slots__ = ('timestamp', 'level', 'message', 'fields', 'module', 'elapsed')

    def __init__(self, level: int, message: str,
                 fields: Optional[Dict[str, Any]] = None,
                 module: str = '', elapsed: float = 0.0):
        self.timestamp = datetime.now().isoformat()
        self.level = level
        self.message = message
        self.fields = fields or {}
        self.module = module
        self.elapsed = elapsed

    def to_dict(self) -> Dict[str, Any]:
        d = {
            'ts': self.timestamp,
            'level': LEVEL_NAMES.get(self.level, 'UNKNOWN'),
            'msg': self.message,
        }
        if self.module:
            d['module'] = self.module
        if self.elapsed:
            d['elapsed_ms'] = round(self.elapsed * 1000, 2)
        if self.fields:
            d.update(self.fields)
        return d

    def to_text(self) -> str:
        ts = self.timestamp[11:23]  # HH:MM:SS.mmm
        lvl = LEVEL_NAMES.get(self.level, '?')[:4]
        mod = f"[{self.module}] " if self.module else ""
        elapsed = f" ({self.elapsed*1000:.1f}ms)" if self.elapsed else ""
        fields = ""
        if self.fields:
            fields = " " + " ".join(f"{k}={v}" for k, v in self.fields.items())
        return f"{ts} {lvl:4s} {mod}{self.message}{elapsed}{fields}"

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class Logger:
    """
    Structured logger with file output, rotation, and querying.

    Usage:
        log = Logger('myapp', level=INFO)
        log.info("Server started", port=8080)
        log.error("Connection failed", host="db.local", retry=3)

        # With module context
        db_log = log.child('database')
        db_log.info("Query executed", rows=42)

        # File output with rotation
        log = Logger('app', file='app.log', max_bytes=1_000_000)

        # Timing decorator
        @log.timed
        def slow_function():
            time.sleep(0.1)
    """

    def __init__(self, name: str = 'foss-ki',
                 level: int = INFO,
                 file: Optional[str] = None,
                 max_bytes: int = 10_000_000,
                 backup_count: int = 3,
                 json_format: bool = False,
                 parent: Optional['Logger'] = None):
        self.name = name
        self.level = level
        self.file = file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.json_format = json_format
        self._parent = parent
        self._context: Dict[str, Any] = {}
        self._buffer: List[LogEntry] = []
        self._max_buffer = 10000
        self._handlers: List[Callable[[LogEntry], None]] = []

    def child(self, module: str) -> 'Logger':
        """Create a child logger with module context."""
        child = Logger(
            name=self.name,
            level=self.level,
            file=self.file,
            max_bytes=self.max_bytes,
            backup_count=self.backup_count,
            json_format=self.json_format,
            parent=self,
        )
        child._context = {**self._context, 'module': module}
        return child

    def add_handler(self, handler: Callable[[LogEntry], None]):
        """Add a custom log handler."""
        self._handlers.append(handler)

    def with_context(self, **kwargs) -> 'Logger':
        """Return logger with additional context fields."""
        child = Logger(
            name=self.name, level=self.level, file=self.file,
            max_bytes=self.max_bytes, backup_count=self.backup_count,
            json_format=self.json_format, parent=self._parent,
        )
        child._context = {**self._context, **kwargs}
        child._buffer = self._buffer
        child._handlers = self._handlers
        return child

    def debug(self, msg: str, **kwargs):
        self._log(DEBUG, msg, kwargs)

    def info(self, msg: str, **kwargs):
        self._log(INFO, msg, kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(WARNING, msg, kwargs)

    def error(self, msg: str, **kwargs):
        self._log(ERROR, msg, kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(CRITICAL, msg, kwargs)

    def _log(self, level: int, msg: str, fields: Dict[str, Any]):
        if level < self.level:
            return

        merged = {**self._context, **fields}
        module = merged.pop('module', '')
        entry = LogEntry(level, msg, merged, module)

        # Buffer
        self._buffer.append(entry)
        if len(self._buffer) > self._max_buffer:
            self._buffer = self._buffer[-self._max_buffer:]

        # Format
        line = entry.to_json() if self.json_format else entry.to_text()

        # Console output (only if no parent — avoid duplicates)
        if not self._parent:
            print(line)

        # File output
        if self.file:
            self._write_to_file(line)

        # Custom handlers
        for handler in self._handlers:
            try:
                handler(entry)
            except Exception:
                pass

        # Propagate to parent
        if self._parent:
            self._parent._buffer.append(entry)

    def _write_to_file(self, line: str):
        """Write to file with rotation."""
        try:
            if os.path.exists(self.file):
                size = os.path.getsize(self.file)
                if size >= self.max_bytes:
                    self._rotate()
            with open(self.file, 'a', encoding='utf-8') as f:
                f.write(line + '\n')
        except OSError:
            pass

    def _rotate(self):
        """Rotate log files."""
        for i in range(self.backup_count - 1, 0, -1):
            src = f"{self.file}.{i}"
            dst = f"{self.file}.{i+1}"
            if os.path.exists(src):
                try:
                    os.rename(src, dst)
                except OSError:
                    pass
        try:
            os.rename(self.file, f"{self.file}.1")
        except OSError:
            pass

    def timed(self, fn: Callable) -> Callable:
        """Decorator to log function execution time."""
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - start
                entry = LogEntry(INFO, f"{fn.__name__} completed",
                                module=self._context.get('module', ''),
                                elapsed=elapsed)
                self._buffer.append(entry)
                if not self._parent:
                    line = entry.to_json() if self.json_format else entry.to_text()
                    print(line)
                return result
            except Exception as e:
                elapsed = time.perf_counter() - start
                self.error(f"{fn.__name__} failed: {e}", elapsed_ms=round(elapsed*1000, 2))
                raise
        return wrapper

    # === Query ===

    def query(self, level: Optional[int] = None,
              pattern: Optional[str] = None,
              module: Optional[str] = None,
              last_n: int = 100) -> List[LogEntry]:
        """Query buffered log entries."""
        results = self._buffer[-last_n:]
        if level is not None:
            results = [e for e in results if e.level >= level]
        if pattern:
            import re
            pat = re.compile(pattern, re.IGNORECASE)
            results = [e for e in results if pat.search(e.message)]
        if module:
            results = [e for e in results if e.module == module]
        return results

    def stats(self) -> Dict[str, int]:
        """Get log level statistics."""
        counts: Dict[str, int] = {}
        for entry in self._buffer:
            name = LEVEL_NAMES.get(entry.level, 'UNKNOWN')
            counts[name] = counts.get(name, 0) + 1
        return counts

    def tail(self, n: int = 10) -> List[str]:
        """Get last N log lines as text."""
        entries = self._buffer[-n:]
        return [e.to_text() for e in entries]

    def clear(self):
        """Clear the log buffer."""
        self._buffer.clear()
