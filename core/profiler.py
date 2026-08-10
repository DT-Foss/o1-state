"""
Profiler — Performance Measurement for FOSS-KI
=================================================
Zero-dependency profiler for measuring:
  - Function execution time
  - Memory usage (RSS)
  - Call frequency
  - Hot path identification
  - Per-component latency breakdown

Usage:
    profiler = Profiler()
    with profiler.measure("knowledge_lookup"):
        result = knowledge.query("test")
    profiler.report()
"""

import time
import sys
from typing import Dict, Any, List, Optional, Callable
from collections import defaultdict
from contextlib import contextmanager
from functools import wraps


def get_memory_mb() -> float:
    """Get current process RSS in MB (macOS/Linux)."""
    try:
        import resource
        # ru_maxrss is in bytes on macOS, KB on Linux
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform == 'darwin':
            return usage / (1024 * 1024)
        return usage / 1024
    except ImportError:
        return 0.0


class TimingRecord:
    """Accumulated timing statistics for a named operation."""

    __slots__ = ('name', 'total_time', 'call_count', 'min_time',
                 'max_time', 'last_time')

    def __init__(self, name: str):
        self.name = name
        self.total_time = 0.0
        self.call_count = 0
        self.min_time = float('inf')
        self.max_time = 0.0
        self.last_time = 0.0

    def record(self, elapsed: float):
        self.total_time += elapsed
        self.call_count += 1
        self.last_time = elapsed
        if elapsed < self.min_time:
            self.min_time = elapsed
        if elapsed > self.max_time:
            self.max_time = elapsed

    @property
    def avg_time(self) -> float:
        if self.call_count == 0:
            return 0.0
        return self.total_time / self.call_count

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'calls': self.call_count,
            'total_ms': round(self.total_time * 1000, 3),
            'avg_ms': round(self.avg_time * 1000, 3),
            'min_ms': round(self.min_time * 1000, 3) if self.min_time < float('inf') else 0,
            'max_ms': round(self.max_time * 1000, 3),
            'last_ms': round(self.last_time * 1000, 3),
        }


class Profiler:
    """Performance profiler for FOSS-KI components."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        self.timings: Dict[str, TimingRecord] = {}
        self.start_time = time.time()
        self.start_memory = get_memory_mb()
        self._stack: List[str] = []  # For nested measurements

    @contextmanager
    def measure(self, name: str):
        """Context manager to measure a named operation."""
        if not self.enabled:
            yield
            return

        self._stack.append(name)
        t0 = time.time()
        try:
            yield
        finally:
            elapsed = time.time() - t0
            self._stack.pop()

            # Use full path for nested measurements
            full_name = '.'.join(self._stack + [name]) if self._stack else name

            if full_name not in self.timings:
                self.timings[full_name] = TimingRecord(full_name)
            self.timings[full_name].record(elapsed)

    def wrap(self, name: Optional[str] = None):
        """Decorator to profile a function."""
        def decorator(func: Callable) -> Callable:
            label = name or f"{func.__module__}.{func.__qualname__}"

            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.measure(label):
                    return func(*args, **kwargs)
            return wrapper
        return decorator

    def record(self, name: str, elapsed: float):
        """Manually record a timing."""
        if not self.enabled:
            return
        if name not in self.timings:
            self.timings[name] = TimingRecord(name)
        self.timings[name].record(elapsed)

    def get_stats(self, name: str) -> Optional[Dict[str, Any]]:
        """Get stats for a named operation."""
        rec = self.timings.get(name)
        return rec.to_dict() if rec else None

    def get_all_stats(self) -> List[Dict[str, Any]]:
        """Get all timing stats sorted by total time."""
        stats = [rec.to_dict() for rec in self.timings.values()]
        stats.sort(key=lambda x: x['total_ms'], reverse=True)
        return stats

    def get_hot_paths(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """Get the N slowest operations."""
        return self.get_all_stats()[:top_n]

    def report(self, top_n: int = 20) -> str:
        """Generate a human-readable profiling report."""
        lines = []
        elapsed = time.time() - self.start_time
        current_mem = get_memory_mb()

        lines.append("=" * 70)
        lines.append("FOSS-KI Performance Report")
        lines.append("=" * 70)
        lines.append(f"  Uptime: {elapsed:.1f}s")
        lines.append(f"  Memory: {current_mem:.1f} MB (start: {self.start_memory:.1f} MB)")
        lines.append(f"  Operations tracked: {len(self.timings)}")
        lines.append("")

        stats = self.get_all_stats()[:top_n]
        if not stats:
            lines.append("  No operations recorded.")
            return '\n'.join(lines)

        # Header
        lines.append(f"  {'Operation':<35} {'Calls':>6} {'Total':>10} {'Avg':>8} {'Max':>8}")
        lines.append(f"  {'-'*35} {'-'*6} {'-'*10} {'-'*8} {'-'*8}")

        for s in stats:
            name = s['name'][:35]
            lines.append(
                f"  {name:<35} {s['calls']:>6} {s['total_ms']:>9.1f}ms "
                f"{s['avg_ms']:>7.2f}ms {s['max_ms']:>7.2f}ms"
            )

        # Latency breakdown by component
        components: Dict[str, float] = defaultdict(float)
        for s in self.get_all_stats():
            component = s['name'].split('.')[0]
            components[component] += s['total_ms']

        if len(components) > 1:
            lines.append("")
            lines.append("  Component Breakdown:")
            total_ms = sum(components.values())
            for comp, ms in sorted(components.items(), key=lambda x: x[1], reverse=True):
                pct = (ms / total_ms * 100) if total_ms > 0 else 0
                bar = '#' * int(pct / 2)
                lines.append(f"    {comp:<25} {ms:>9.1f}ms ({pct:>5.1f}%) {bar}")

        return '\n'.join(lines)

    def reset(self):
        """Reset all timing data."""
        self.timings.clear()
        self.start_time = time.time()
        self.start_memory = get_memory_mb()

    def merge(self, other: 'Profiler'):
        """Merge another profiler's data into this one."""
        for name, rec in other.timings.items():
            if name not in self.timings:
                self.timings[name] = TimingRecord(name)
            t = self.timings[name]
            t.total_time += rec.total_time
            t.call_count += rec.call_count
            if rec.min_time < t.min_time:
                t.min_time = rec.min_time
            if rec.max_time > t.max_time:
                t.max_time = rec.max_time
            t.last_time = rec.last_time


class Benchmark:
    """Run benchmarks and report results."""

    def __init__(self):
        self.results: List[Dict[str, Any]] = []

    def run(self, name: str, func: Callable, iterations: int = 100,
            warmup: int = 5) -> Dict[str, Any]:
        """Run a benchmark function multiple times."""
        # Warmup
        for _ in range(warmup):
            func()

        # Measure
        times = []
        for _ in range(iterations):
            t0 = time.time()
            func()
            times.append(time.time() - t0)

        times.sort()
        result = {
            'name': name,
            'iterations': iterations,
            'total_ms': round(sum(times) * 1000, 3),
            'avg_ms': round(sum(times) / len(times) * 1000, 3),
            'median_ms': round(times[len(times) // 2] * 1000, 3),
            'min_ms': round(times[0] * 1000, 3),
            'max_ms': round(times[-1] * 1000, 3),
            'p95_ms': round(times[int(len(times) * 0.95)] * 1000, 3),
            'p99_ms': round(times[int(len(times) * 0.99)] * 1000, 3),
            'ops_per_sec': round(len(times) / sum(times)) if sum(times) > 0 else 0,
        }

        self.results.append(result)
        return result

    def report(self) -> str:
        """Generate benchmark report."""
        lines = ["Benchmark Results:", ""]
        lines.append(f"  {'Name':<30} {'Avg':>8} {'P95':>8} {'P99':>8} {'Ops/s':>8}")
        lines.append(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

        for r in self.results:
            lines.append(
                f"  {r['name']:<30} {r['avg_ms']:>7.2f}ms "
                f"{r['p95_ms']:>7.2f}ms {r['p99_ms']:>7.2f}ms "
                f"{r['ops_per_sec']:>7}"
            )

        return '\n'.join(lines)


# Singleton global profiler
_global_profiler = Profiler(enabled=False)


def get_global_profiler() -> Profiler:
    """Get the global profiler instance."""
    return _global_profiler


def enable_profiling():
    """Enable the global profiler."""
    _global_profiler.enabled = True


def disable_profiling():
    """Disable the global profiler."""
    _global_profiler.enabled = False
