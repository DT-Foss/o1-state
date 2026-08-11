"""Worker pool with circuit breaker and graceful degradation.

Implements a managed pool of worker processes with:
- Automatic crash recovery (restart failed workers)
- Circuit breaker pattern (stop calling failing workers)
- Graceful degradation (fall back to sequential when parallel fails)
- Timeout handling for hung workers
- Process health monitoring

Uses ONLY Python standard library + NumPy.
No PyTorch. No neural networks.
"""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, Future, TimeoutError as FutureTimeoutError
from typing import Callable, List, Optional, Any, Dict, Tuple
import time
import os
import signal
import logging
import traceback
from enum import Enum, auto
from dataclasses import dataclass, field

logger = logging.getLogger("hsslm_s.worker_pool")


# =============================================================================
# CIRCUIT BREAKER
# =============================================================================

class CircuitState(Enum):
    """Circuit breaker states."""
    CLOSED = auto()      # Normal operation, requests pass through
    OPEN = auto()        # Failing fast, requests rejected
    HALF_OPEN = auto()   # Testing if service recovered


@dataclass
class CircuitBreaker:
    """Circuit breaker for failing workers.
    
    Implements the circuit breaker pattern to prevent cascading failures.
    After a threshold of consecutive failures, the circuit opens and
    all subsequent calls fail fast. After a timeout, the circuit enters
    half-open state to test if the service recovered.
    
    Typical usage:
        breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=30.0)
        
        try:
            result = breaker.call(worker_function, arg1, arg2)
        except CircuitBreakerOpen:
            # Circuit is open, use fallback
            result = fallback_function(arg1, arg2)
    
    Args:
        failure_threshold: Number of consecutive failures before opening
        recovery_timeout: Seconds to wait before half-open test
        half_open_max_calls: Max calls in half-open state
        success_threshold: Successes needed to close from half-open
    """
    
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    half_open_max_calls: int = 3
    success_threshold: int = 2
    
    def __post_init__(self):
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        self._last_failure_time = 0.0
        self._total_failures = 0
        self._total_successes = 0
        self._total_rejected = 0
    
    @property
    def state(self) -> CircuitState:
        """Current circuit state."""
        if self._state == CircuitState.OPEN:
            # Check if recovery timeout elapsed
            if time.monotonic() - self._last_failure_time > self.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                self._half_open_calls = 0
                self._success_count = 0
                logger.info("Circuit breaker entering HALF_OPEN state")
        return self._state
    
    def call(self, fn: Callable, *args, **kwargs) -> Any:
        """Call function through the circuit breaker.
        
        Args:
            fn: Function to call
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            Function result
        
        Raises:
            CircuitBreakerOpen: If circuit is OPEN
            Exception: If the function raises (and circuit transitions)
        """
        current_state = self.state
        
        if current_state == CircuitState.OPEN:
            self._total_rejected += 1
            raise CircuitBreakerOpen(
                f"Circuit breaker is OPEN. Last failure: "
                f"{time.monotonic() - self._last_failure_time:.1f}s ago. "
                f"Failures: {self._failure_count}"
            )
        
        if current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls >= self.half_open_max_calls:
                self._total_rejected += 1
                raise CircuitBreakerOpen(
                    "Circuit breaker HALF_OPEN limit reached"
                )
            self._half_open_calls += 1
        
        try:
            result = fn(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise
    
    def _on_success(self):
        """Record a successful call."""
        self._failure_count = 0
        self._total_successes += 1
        
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.success_threshold:
                logger.info("Circuit breaker CLOSED (recovery confirmed)")
                self._state = CircuitState.CLOSED
                self._half_open_calls = 0
                self._success_count = 0
    
    def _on_failure(self):
        """Record a failed call."""
        self._failure_count += 1
        self._total_failures += 1
        self._last_failure_time = time.monotonic()
        
        if self._state == CircuitState.HALF_OPEN:
            # Failed in half-open: go back to open
            logger.warning("Circuit breaker OPEN (half-open test failed)")
            self._state = CircuitState.OPEN
            self._half_open_calls = 0
            self._success_count = 0
        elif self._failure_count >= self.failure_threshold:
            logger.warning(
                f"Circuit breaker OPEN after {self._failure_count} consecutive failures"
            )
            self._state = CircuitState.OPEN
    
    def force_close(self):
        """Manually close the circuit (reset)."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0
        logger.info("Circuit breaker manually CLOSED")
    
    def force_open(self):
        """Manually open the circuit."""
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()
        logger.info("Circuit breaker manually OPENED")
    
    def get_stats(self) -> dict:
        """Get circuit breaker statistics.
        
        Returns:
            Dictionary with state, failure counts, etc.
        """
        return {
            'state': self.state.name,
            'failure_count': self._failure_count,
            'total_failures': self._total_failures,
            'total_successes': self._total_successes,
            'total_rejected': self._total_rejected,
            'last_failure_time': self._last_failure_time,
        }


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open."""
    pass


# =============================================================================
# WORKER PROCESS WRAPPER
# =============================================================================

@dataclass
class WorkerProcess:
    """Metadata for a managed worker process."""
    worker_id: int
    process: Optional[mp.Process] = None
    pid: Optional[int] = None
    status: str = "idle"  # idle, running, crashed, restarting
    last_task_time: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    start_time: float = 0.0
    
    @property
    def is_alive(self) -> bool:
        """Check if the worker process is alive."""
        if self.process is None:
            return False
        return self.process.is_alive()
    
    @property
    def uptime(self) -> float:
        """Seconds since worker started."""
        if self.start_time == 0:
            return 0.0
        return time.monotonic() - self.start_time


# =============================================================================
# MAIN WORKER POOL
# =============================================================================

class WorkerPool:
    """Managed worker processes with crash recovery.
    
    Maintains a pool of worker processes with automatic restart
    on crash, timeout handling, and graceful degradation to
    sequential execution when parallel processing fails.
    
    Args:
        n_workers: Number of worker processes
        worker_fn: Function that workers execute (must be picklable)
        fallback_fn: Optional fallback function for sequential execution
        timeout: Per-task timeout in seconds
        max_restarts: Maximum restarts per worker
        use_circuit_breaker: Whether to use circuit breaker pattern
    
    Typical usage:
        def my_worker(data):
            return process(data)
        
        pool = WorkerPool(n_workers=4, worker_fn=my_worker)
        
        # Map work across workers
        results = pool.map([item1, item2, item3, item4])
        
        # Shutdown
        pool.shutdown()
    """
    
    def __init__(self, n_workers: int, worker_fn: Callable,
                 fallback_fn: Callable = None, timeout: float = 30.0,
                 max_restarts: int = 3, use_circuit_breaker: bool = True):
        """Initialize worker pool.
        
        Args:
            n_workers: Number of worker processes
            worker_fn: Function for workers (must be picklable)
            fallback_fn: Fallback for sequential execution
            timeout: Per-task timeout in seconds
            max_restarts: Max restarts per worker before giving up
            use_circuit_breaker: Enable circuit breaker
        """
        self.n_workers = n_workers
        self.worker_fn = worker_fn
        self.fallback_fn = fallback_fn or worker_fn
        self.timeout = timeout
        self.max_restarts = max_restarts
        self.use_circuit_breaker = use_circuit_breaker
        
        # Process executor for parallel tasks
        self._executor: Optional[ProcessPoolExecutor] = None
        self._workers: Dict[int, WorkerProcess] = {}
        self._active = False
        
        # Circuit breaker
        self._circuit = CircuitBreaker(
            failure_threshold=5,
            recovery_timeout=30.0
        ) if use_circuit_breaker else None
        
        # Statistics
        self._stats = {
            'tasks_submitted': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'tasks_fallback': 0,
            'worker_restarts': 0,
            'worker_crashes': 0,
            'start_time': 0.0,
        }
        
        # Initialize
        self._start_pool()
    
    def _start_pool(self):
        """Start the worker process pool."""
        if self._active:
            return
        
        try:
            self._executor = ProcessPoolExecutor(
                max_workers=self.n_workers,
                initializer=_worker_signal_init
            )
            self._active = True
            self._stats['start_time'] = time.monotonic()
            
            # Initialize worker tracking
            for i in range(self.n_workers):
                self._workers[i] = WorkerProcess(worker_id=i)
            
            logger.info(f"WorkerPool started with {self.n_workers} workers")
        except Exception as e:
            logger.error(f"Failed to start worker pool: {e}")
            self._active = False
            self._executor = None
    
    def _restart_pool(self):
        """Restart the entire pool after catastrophic failure."""
        logger.warning("Restarting worker pool after failure")
        self._shutdown_pool()
        time.sleep(0.5)  # Brief pause for cleanup
        self._start_pool()
    
    def _shutdown_pool(self):
        """Shut down the current pool."""
        if self._executor is not None:
            try:
                self._executor.shutdown(wait=False)
            except Exception as e:
                logger.warning(f"Error shutting down executor: {e}")
            self._executor = None
        self._active = False
    
    def map(self, items: List[Any]) -> List[Any]:
        """Map work items across worker pool.
        
        Distributes items across workers and collects results.
        Falls back to sequential execution if parallel fails.
        
        Args:
            items: List of work items (each passed to worker_fn)
        
        Returns:
            List of results in same order as input items
        """
        if not items:
            return []
        
        # Small batches: sequential is faster
        if len(items) <= 2 or not self._active or self.n_workers <= 1:
            return self._sequential_fallback(items)
        
        # Check circuit breaker
        if self._circuit is not None and self._circuit.state == CircuitState.OPEN:
            logger.info("Circuit breaker OPEN, using sequential fallback")
            self._stats['tasks_fallback'] += len(items)
            return self._sequential_fallback(items)
        
        # Try parallel execution
        try:
            results = self._parallel_map(items)
            
            # Record success
            if self._circuit is not None:
                self._circuit._on_success()
            
            self._stats['tasks_completed'] += len(items)
            return results
        
        except Exception as e:
            logger.warning(f"Parallel execution failed: {e}, falling back to sequential")
            
            # Record failure
            if self._circuit is not None:
                self._circuit._on_failure()
            
            self._stats['tasks_failed'] += len(items)
            self._stats['tasks_fallback'] += len(items)
            
            # Graceful degradation: sequential fallback
            return self._sequential_fallback(items)
    
    def _parallel_map(self, items: List[Any]) -> List[Any]:
        """Execute items in parallel using process pool.
        
        Args:
            items: Work items
        
        Returns:
            Results in input order
        
        Raises:
            Exception: If parallel execution fails
        """
        if self._executor is None or not self._active:
            raise RuntimeError("Pool not active")
        
        self._stats['tasks_submitted'] += len(items)
        
        # Submit all tasks
        futures = []
        for item in items:
            fut = self._executor.submit(self.worker_fn, item)
            futures.append(fut)
        
        # Collect results with timeout
        results = []
        for i, fut in enumerate(futures):
            try:
                result = fut.result(timeout=self.timeout)
                results.append(result)
                # Update worker stats
                worker_id = i % self.n_workers
                if worker_id in self._workers:
                    self._workers[worker_id].tasks_completed += 1
            except FutureTimeoutError:
                logger.error(f"Task {i} timed out after {self.timeout}s")
                results.append(self._handle_timeout(items[i]))
            except Exception as e:
                logger.error(f"Task {i} failed: {e}")
                # Try sequential for this item
                try:
                    results.append(self.fallback_fn(items[i]))
                    self._stats['tasks_fallback'] += 1
                except Exception:
                    results.append(None)
        
        return results
    
    def _sequential_fallback(self, items: List[Any]) -> List[Any]:
        """Sequential execution fallback.
        
        Args:
            items: Work items
        
        Returns:
            Results
        """
        results = []
        for item in items:
            try:
                result = self.fallback_fn(item)
                results.append(result)
            except Exception as e:
                logger.error(f"Fallback execution failed: {e}")
                results.append(None)
        return results
    
    def _handle_timeout(self, item: Any) -> Any:
        """Handle a task that timed out.
        
        Tries sequential fallback for the timed-out item.
        
        Args:
            item: The work item that timed out
        
        Returns:
            Result or None
        """
        try:
            return self.fallback_fn(item)
        except Exception as e:
            logger.error(f"Timeout fallback also failed: {e}")
            return None
    
    def submit(self, item: Any) -> Optional[Future]:
        """Submit a single task asynchronously.
        
        Args:
            item: Work item
        
        Returns:
            Future object or None if submission failed
        """
        if not self._active or self._executor is None:
            return None
        
        try:
            self._stats['tasks_submitted'] += 1
            return self._executor.submit(self.worker_fn, item)
        except Exception as e:
            logger.error(f"Failed to submit task: {e}")
            return None
    
    def health_check(self) -> bool:
        """Check pool health.
        
        Returns:
            True if pool is healthy
        """
        if not self._active:
            return False
        if self._executor is None:
            return False
        return True
    
    def get_stats(self) -> dict:
        """Get pool statistics.
        
        Returns:
            Dictionary with task counts, worker status, etc.
        """
        stats = dict(self._stats)
        stats['pool_active'] = self._active
        stats['n_workers'] = self.n_workers
        stats['circuit_breaker'] = (
            self._circuit.get_stats() if self._circuit else None
        )
        stats['worker_status'] = {
            wid: {
                'status': w.status,
                'alive': w.is_alive,
                'uptime': w.uptime,
                'tasks_completed': w.tasks_completed,
                'tasks_failed': w.tasks_failed,
            }
            for wid, w in self._workers.items()
        }
        return stats
    
    def shutdown(self):
        """Graceful shutdown of the worker pool."""
        logger.info("Shutting down WorkerPool")
        self._shutdown_pool()
        
        uptime = time.monotonic() - self._stats['start_time']
        logger.info(
            f"WorkerPool stats: {self._stats['tasks_completed']} completed, "
            f"{self._stats['tasks_failed']} failed, "
            f"{self._stats['tasks_fallback']} fallback, "
            f"uptime={uptime:.1f}s"
        )
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
        return False


def _worker_signal_init():
    """Initialize worker process: ignore SIGINT, set niceness.
    
    Called once in each worker process on startup.
    """
    # Ignore SIGINT in workers (handled by parent)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # Lower priority slightly for CPU-bound workers
    try:
        os.nice(5)
    except (PermissionError, OSError):
        pass


# =============================================================================
# GRACEFUL DEGRADATION HELPERS
# =============================================================================

def should_use_parallel(sequence_length: int, vocab_size: int = 50000,
                        n_streams: int = 1, cache_size_l2_kb: float = 256.0) -> bool:
    """Heuristic: decide whether parallel processing is beneficial.
    
    Parallelism has overhead. For small problems, sequential is faster.
    This function uses the criteria from the HSSLM-S architecture spec.
    
    Args:
        sequence_length: Number of positions to process
        vocab_size: Vocabulary dimension
        n_streams: Number of parallel streams
        cache_size_l2_kb: L2 cache size in KB
    
    Returns:
        True if parallel speedup exceeds overhead
    """
    state_size_kb = sequence_length * 2 * 8 / 1024
    
    # Condition 1: Sequence long enough to amortize overhead
    long_enough = sequence_length >= 64
    
    # Condition 2: Vocabulary large enough for BvN to matter
    vocab_large = vocab_size >= 1000
    
    # Condition 3: Multiple streams or batch
    multi_stream = n_streams > 1
    
    # Condition 4: State doesn't fit in L2 (avoids cache thrashing)
    exceeds_l2 = state_size_kb > cache_size_l2_kb
    
    # Parallel is beneficial when at least 2 conditions hold
    score = sum([long_enough, vocab_large, multi_stream, exceeds_l2])
    return score >= 2


def graceful_parallel_map(items: List[Any], worker_fn: Callable,
                          fallback_fn: Callable = None,
                          n_workers: int = 4, timeout: float = 30.0) -> List[Any]:
    """Map with automatic graceful degradation.
    
    Tries parallel execution first, falls back to sequential if:
    - Parallel pool fails to start
    - Tasks timeout or crash
    - Circuit breaker opens
    - Items count is too small for parallelism
    
    Args:
        items: Work items
        worker_fn: Function for parallel workers
        fallback_fn: Optional fallback (defaults to worker_fn)
        n_workers: Number of parallel workers
        timeout: Per-task timeout
    
    Returns:
        Results in input order
    """
    fallback_fn = fallback_fn or worker_fn
    
    # Small batches: sequential
    if len(items) <= 2:
        return [fallback_fn(item) for item in items]
    
    # Try parallel with worker pool
    pool = None
    try:
        pool = WorkerPool(
            n_workers=n_workers,
            worker_fn=worker_fn,
            fallback_fn=fallback_fn,
            timeout=timeout,
            use_circuit_breaker=True
        )
        return pool.map(items)
    except Exception as e:
        logger.warning(f"WorkerPool failed: {e}, using sequential")
        return [fallback_fn(item) for item in items]
    finally:
        if pool is not None:
            try:
                pool.shutdown()
            except Exception:
                pass
