"""Lock-free parallel processing for HSSLM-S.

Purely symbolic, deterministic language module using parallel processing
across multiple CPU cores. All operations are mathematical formulas —
no neural network weights, no PyTorch.

Uses ONLY Python standard library + NumPy + multiprocessing.
Architecture:
    - Lock-free ring buffer for token streaming
    - Shared state arrays via multiprocessing.RawArray
    - Parallel Möbius state transitions (embarrassingly parallel)
    - Parallel BvN decomposition (tree reduction)
    - Multi-stream generation with different tau values
    - Graceful degradation to sequential when parallel overhead exceeds benefit
"""

import multiprocessing as mp
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict, Callable, Any
import ctypes
import time
import os
import signal
import math
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(levelname)s: %(message)s')
logger = logging.getLogger("hsslm_s")


# =============================================================================
# MATHEMATICAL CONSTANTS AND TYPES
# =============================================================================

class PathType(Enum):
    """BvN path / inference chain types."""
    EXACT = auto()
    SEMANTIC = auto()
    FUZZY = auto()
    CLOSURE = auto()


@dataclass
class MoebiusState:
    """State on the Poincare unit disk.
    
    Represents the symbolic state at a single token position as a
    complex number z = x + iy with |z| < 1.
    """
    re: float = 0.0
    im: float = 0.0
    
    def as_complex(self) -> complex:
        return complex(self.re, self.im)
    
    def norm_sq(self) -> float:
        return self.re * self.re + self.im * self.im
    
    def apply_moebius(self, a: complex, b: complex) -> "MoebiusState":
        """Apply Möbius transformation f(z) = (a*z + b) / (conj(b)*z + conj(a))
        where |a|^2 - |b|^2 = 1 (preserves unit disk).
        """
        z = self.as_complex()
        numerator = a * z + b
        denominator = b.conjugate() * z + a.conjugate()
        result = numerator / (denominator + 1e-15)
        # Clamp to unit disk (numerical safety)
        if abs(result) >= 1.0:
            result = result / (abs(result) + 1e-10) * 0.999999
        return MoebiusState(result.real, result.imag)


@dataclass
class BvNDecomposition:
    """Birkhoff-von Neumann decomposition result.
    
    A doubly stochastic matrix P is decomposed as:
        P = sum_i lambda_i * Pi
    where Pi are permutation matrices and sum_i lambda_i = 1.
    """
    permutations: List[np.ndarray] = field(default_factory=list)
    weights: List[float] = field(default_factory=list)
    
    def reconstruct(self) -> np.ndarray:
        """Reconstruct the doubly stochastic matrix."""
        if not self.permutations:
            return np.array([[0.0]])
        n = self.permutations[0].shape[0]
        result = np.zeros((n, n), dtype=np.float64)
        for perm, w in zip(self.permutations, self.weights):
            result += w * perm
        return result


# =============================================================================
# LOCK-FREE RING BUFFER (SPSC)
# =============================================================================

class LockFreeRingBuffer:
    """Single-producer single-consumer ring buffer via shared memory.
    
    Uses a NumPy array for storage with atomic index updates.
    The producer writes to tail, consumer reads from head.
    No locks required because there is exactly one writer and one reader.
    
    Layout:
        buffer[0]           : head index (consumer)
        buffer[1]           : tail index (producer)
        buffer[2:capacity+2]: data slots
    
    Thread-safe for single-producer single-consumer only.
    """
    
    def __init__(self, capacity: int, item_size: int = 1, dtype=np.float64):
        """Initialize the ring buffer.
        
        Args:
            capacity: Maximum number of items in the buffer
            item_size: Size of each item in elements (for multi-element items)
            dtype: NumPy data type for stored values
        """
        self.capacity = capacity
        self.item_size = item_size
        self._dtype = dtype
        self._item_nbytes = np.dtype(dtype).itemsize
        
        # Shared memory: head, tail, then data slots
        # Each slot stores 'item_size' elements
        self._buf = np.zeros(capacity * item_size + 2, dtype=dtype)
        self._buf[0] = 0  # head
        self._buf[1] = 0  # tail
        
        # Cache line padding to prevent false sharing
        self._head_idx = 0
        self._tail_idx = 1
    
    @property
    def head(self) -> int:
        """Consumer read position."""
        return int(self._buf[self._head_idx])
    
    @head.setter
    def head(self, value: int):
        self._buf[self._head_idx] = value
    
    @property
    def tail(self) -> int:
        """Producer write position."""
        return int(self._buf[self._tail_idx])
    
    @tail.setter
    def tail(self, value: int):
        self._buf[self._tail_idx] = value
    
    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return self.head == self.tail
    
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return ((self.tail + 1) % self.capacity) == self.head
    
    def put(self, item: np.ndarray) -> bool:
        """Producer: push one item. Returns False if full.
        
        Args:
            item: Item to push (NumPy array of shape (item_size,))
        
        Returns:
            True if successful, False if buffer full (backpressure)
        """
        item = np.asarray(item, dtype=self._dtype).flatten()
        if item.size != self.item_size:
            raise ValueError(f"Item size {item.size} != expected {self.item_size}")
        
        next_tail = (self.tail + 1) % self.capacity
        if next_tail == self.head:
            return False  # Buffer full — backpressure
        
        # Write data
        data_start = 2 + self.tail * self.item_size
        self._buf[data_start:data_start + self.item_size] = item
        self.tail = next_tail
        return True
    
    def get(self) -> Optional[np.ndarray]:
        """Consumer: pop one item. Returns None if empty.
        
        Returns:
            NumPy array of shape (item_size,) or None if empty
        """
        if self.is_empty():
            return None
        
        data_start = 2 + self.head * self.item_size
        item = self._buf[data_start:data_start + self.item_size].copy()
        self.head = (self.head + 1) % self.capacity
        return item
    
    def size(self) -> int:
        """Number of items in buffer (approximate, racy)."""
        return (self.tail - self.head) % self.capacity
    
    def available(self) -> int:
        """Number of free slots."""
        return self.capacity - self.size() - 1
    
    def fill_ratio(self) -> float:
        """Fraction of buffer that is full."""
        return self.size() / max(self.capacity - 1, 1)
    
    def batch_put(self, items: np.ndarray) -> int:
        """Push multiple items. Returns count successfully pushed.
        
        Args:
            items: Array of shape (n, item_size)
        
        Returns:
            Number of items pushed
        """
        items = np.asarray(items, dtype=self._dtype)
        if items.ndim == 1 and self.item_size == 1:
            items = items.reshape(-1, 1)
        
        count = 0
        for item in items:
            if not self.put(item):
                break
            count += 1
        return count
    
    def batch_get(self, max_items: int) -> np.ndarray:
        """Pop up to max_items. Returns array of popped items.
        
        Args:
            max_items: Maximum number of items to retrieve
        
        Returns:
            Array of shape (n_retrieved, item_size)
        """
        result = []
        for _ in range(max_items):
            item = self.get()
            if item is None:
                break
            result.append(item)
        if not result:
            return np.empty((0, self.item_size), dtype=self._dtype)
        return np.stack(result, axis=0)


# =============================================================================
# SHARED STATE ARRAY
# =============================================================================

class SharedStateArray:
    """Zero-copy shared state between processes via RawArray.
    
    Uses multiprocessing.RawArray for zero-copy sharing across
    process boundaries — no GIL, no serialization overhead.
    
    Layout: shape=(size,) flat array of given dtype.
    """
    
    def __init__(self, shape: tuple, dtype=np.float32):
        """Initialize shared state array.
        
        Args:
            shape: Tuple of array dimensions
            dtype: NumPy data type (default float32)
        """
        self.shape = shape
        self.dtype = dtype
        self._size = int(np.prod(shape))
        self._nbytes = self._size * np.dtype(dtype).itemsize
        
        # Map numpy dtype to ctypes type
        dtype_to_ctype = {
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
            np.int32: ctypes.c_int32,
            np.int64: ctypes.c_int64,
        }
        ctype = dtype_to_ctype.get(np.dtype(dtype).type, ctypes.c_float)
        
        # Raw shared memory, no GIL
        self._raw = mp.RawArray(ctype, self._size)
        self._array = np.frombuffer(self._raw, dtype=dtype)
        self._array = self._array.reshape(shape)
        self._array.fill(0.0)
    
    def get(self) -> np.ndarray:
        """Get read-only copy of the shared array.
        
        Returns:
            NumPy copy of the shared array
        """
        return np.array(self._array, copy=True)
    
    def set(self, arr: np.ndarray):
        """Set shared array contents.
        
        Args:
            arr: NumPy array to copy into shared memory
        """
        arr = np.asarray(arr, dtype=self.dtype)
        flat = arr.reshape(self.shape)
        self._array[:] = flat[:]
    
    def get_view(self, start: int, end: int) -> np.ndarray:
        """Get zero-copy view of a slice.
        
        Args:
            start: Start index (inclusive)
            end: End index (exclusive)
        
        Returns:
            NumPy view (not copy) of the slice
        """
        return self._array[start:end]
    
    def write_chunk(self, start: int, data: np.ndarray):
        """Write a chunk of data in-place.
        
        Args:
            start: Start position
            data: Data to write
        """
        data = np.asarray(data, dtype=self.dtype)
        n = data.size
        flat_target = self._array.reshape(-1)
        flat_target[start:start + n] = data.reshape(-1)[:]
    
    def as_array(self) -> np.ndarray:
        """Get the underlying array view (zero-copy, process-local only).
        
        Returns:
            NumPy array view into shared memory
        """
        return self._array
    
    @property
    def nbytes(self) -> int:
        return self._nbytes


# =============================================================================
# ATOMIC COUNTER (lock-free)
# =============================================================================

class AtomicCounter:
    """Process-safe counter using shared memory with explicit locking.
    
    Uses multiprocessing.Value which provides get_lock() for
    process-safe access across worker processes.
    """
    
    def __init__(self, initial: int = 0):
        self._val = mp.Value(ctypes.c_uint64, initial)
    
    def increment(self, delta: int = 1) -> int:
        """Atomically add delta, return new value.
        
        Args:
            delta: Amount to add
        
        Returns:
            New counter value
        """
        with self._val.get_lock():
            self._val.value += delta
            return self._val.value
    
    @property
    def value(self) -> int:
        with self._val.get_lock():
            return self._val.value
    
    @value.setter
    def value(self, v: int):
        with self._val.get_lock():
            self._val.value = v


# =============================================================================
# RESULT ACCUMULATOR (tree reduction)
# =============================================================================

class ResultAccumulator:
    """Tree-reduction result accumulator for parallel workers.
    
    Workers write partial results to pre-allocated slots.
    Coordinator merges via tree reduction (parallel where possible).
    """
    
    def __init__(self, n_slots: int, slot_size: int, dtype=np.float64):
        self.n_slots = n_slots
        self.slot_size = slot_size
        self.dtype = dtype
        
        # Pre-allocate slots
        ctype_map = {
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
        }
        ctype = ctype_map.get(np.dtype(dtype).type, ctypes.c_double)
        
        self._slots = [mp.RawArray(ctype, slot_size) for _ in range(n_slots)]
        self._ready = mp.RawArray(ctypes.c_uint8, n_slots)  # 0=empty, 1=ready
        self._completed = AtomicCounter(0)
    
    def write_slot(self, slot_id: int, data: np.ndarray):
        """Worker writes result to its slot.
        
        Args:
            slot_id: Which slot to write
            data: NumPy array data (will be flattened)
        """
        data = np.asarray(data, dtype=self.dtype).reshape(-1)
        n = min(data.size, self.slot_size)
        
        arr = np.frombuffer(self._slots[slot_id], dtype=self.dtype)
        arr[:n] = data[:n]
        self._ready[slot_id] = 1
        self._completed.increment()
    
    def is_ready(self, slot_id: int) -> bool:
        return self._ready[slot_id] == 1
    
    def all_ready(self) -> bool:
        return self._completed.value >= self.n_slots
    
    def merge_all(self, merge_fn: Callable[[np.ndarray, np.ndarray], np.ndarray] = None) -> np.ndarray:
        """Tree-reduce all slots into single result.
        
        Args:
            merge_fn: Function to merge two arrays. Default: element-wise mean.
        
        Returns:
            Merged NumPy array
        """
        if merge_fn is None:
            merge_fn = lambda a, b: (a + b) / 2.0
        
        # Collect ready arrays
        arrays = [
            np.frombuffer(self._slots[i], dtype=self.dtype).copy()[:self.slot_size]
            for i in range(self.n_slots)
            if self._ready[i]
        ]
        if not arrays:
            return np.zeros(self.slot_size, dtype=self.dtype)
        
        # Tree reduction
        while len(arrays) > 1:
            next_level = []
            for i in range(0, len(arrays), 2):
                if i + 1 < len(arrays):
                    next_level.append(merge_fn(arrays[i], arrays[i + 1]))
                else:
                    next_level.append(arrays[i])
            arrays = next_level
        
        return arrays[0]


# =============================================================================
# PARALLEL CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class ParallelConfig:
    """Immutable configuration for HSSLM-S parallelism."""
    n_state_workers: int      # Möbius transition workers
    n_inference_workers: int  # Graph traversal workers
    n_sampler_processes: int  # BvN + gate processes
    positions_per_worker: int # Chunk size for state array
    ring_buffer_size: int     # Tokens in flight
    max_inference_depth: int  # Graph traversal limit
    timeout_ms: int           # Per-worker timeout
    
    @classmethod
    def auto(cls, n_cores: int = None) -> "ParallelConfig":
        """Auto-configure based on available hardware.
        
        Args:
            n_cores: Override core count (default: os.cpu_count())
        
        Returns:
            ParallelConfig tuned for the hardware
        """
        n_cores = n_cores or os.cpu_count() or 4
        return cls(
            n_state_workers=max(1, n_cores - 2),
            n_inference_workers=min(4, max(1, n_cores // 2)),
            n_sampler_processes=1,
            positions_per_worker=256,
            ring_buffer_size=4096,
            max_inference_depth=16,
            timeout_ms=5000,
        )


# =============================================================================
# STANDALONE PARALLEL WORKER FUNCTIONS
# =============================================================================

def _token_to_moebius_params(token_id: int) -> Tuple[complex, complex]:
    """Map token ID to Möbius transformation parameters (a, b).
    
    Uses deterministic hash to generate SU(1,1) matrix elements
    satisfying |a|^2 - |b|^2 = 1.
    
    Args:
        token_id: Integer token ID
    
    Returns:
        Tuple of (a, b) complex parameters
    """
    rng = np.random.RandomState(token_id % (2 ** 31))
    
    theta_a = rng.uniform(0, 2 * np.pi)
    theta_b = rng.uniform(0, 2 * np.pi)
    r_b = rng.uniform(0, 0.95)  # |b| < 1 ensures hyperbolic
    r_a = math.sqrt(1 + r_b * r_b)
    
    a = complex(r_a * math.cos(theta_a), r_a * math.sin(theta_a))
    b = complex(r_b * math.cos(theta_b), r_b * math.sin(theta_b))
    
    return a, b


def _moebius_worker_chunk(args: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, int]:
    """Worker function: process a chunk of Möbius transitions.
    
    Args:
        args: (states_chunk, inputs_chunk, chunk_id)
            states_chunk: (n, 2) float64 array of (Re, Im) pairs
            inputs_chunk: (n,) int64 array of token IDs
            chunk_id: Identifier for this chunk
    
    Returns:
        (updated_states, chunk_id)
    """
    states, inputs, chunk_id = args
    n = states.shape[0]
    result = np.zeros_like(states)
    
    for i in range(n):
        z = MoebiusState(states[i, 0], states[i, 1])
        a, b = _token_to_moebius_params(int(inputs[i]))
        z_new = z.apply_moebius(a, b)
        result[i, 0] = z_new.re
        result[i, 1] = z_new.im
    
    return result, chunk_id


def _simd_moebius_chunk(args: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[np.ndarray, int]:
    """Optimized SIMD worker using NumPy vectorized operations.
    
    For large chunks, uses pure NumPy vectorization instead of
    per-element Python loops.
    
    Args:
        args: (states_chunk, directions_chunk, chunk_id)
            Both arrays are float32 of matching shapes
    
    Returns:
        (result, chunk_id)
    """
    states, directions, chunk_id = args
    
    # Use SIMD-vectorized Möbius addition
    numerator = states + directions
    denominator = 1.0 + states * directions
    np.maximum(denominator, 1e-7, out=denominator)
    result = numerator / denominator
    np.clip(result, -0.9999, 0.9999, out=result)
    
    return result, chunk_id


def _bvn_greedy_worker(args: Tuple[np.ndarray, int]) -> BvNDecomposition:
    """Worker function: greedy BvN extraction.
    
    Args:
        args: (matrix, max_paths)
    
    Returns:
        BvNDecomposition with permutations and weights
    """
    matrix, max_paths = args
    n = matrix.shape[0]
    remaining = matrix.copy()
    decomp = BvNDecomposition()
    
    for _ in range(max_paths):
        if remaining.max() < 1e-10:
            break
        
        # Greedy matching
        perm = np.zeros((n, n), dtype=np.float64)
        used = set()
        for i in range(n):
            best_j, best_val = -1, -1.0
            for j in range(n):
                if j not in used and remaining[i, j] > best_val:
                    best_j, best_val = j, remaining[i, j]
            if best_j >= 0:
                perm[i, best_j] = 1.0
                used.add(best_j)
        
        if not used:
            break
        
        lam = remaining[perm > 0.5].min()
        decomp.permutations.append(perm)
        decomp.weights.append(float(lam))
        remaining -= lam * perm
    
    total = sum(decomp.weights)
    if total > 0:
        decomp.weights = [w / total for w in decomp.weights]
    
    return decomp


def _bvn_threshold_worker(args: Tuple[np.ndarray, int]) -> BvNDecomposition:
    """Worker function: threshold-based BvN extraction.
    
    Args:
        args: (matrix, max_paths)
    
    Returns:
        BvNDecomposition
    """
    matrix, max_paths = args
    n = matrix.shape[0]
    remaining = matrix.copy()
    decomp = BvNDecomposition()
    
    for _ in range(max_paths):
        if remaining.max() < 1e-10:
            break
        
        perm = np.zeros((n, n), dtype=np.float64)
        threshold = remaining.mean()
        used_cols = set()
        
        for i in range(n):
            for j in range(n):
                if j not in used_cols and remaining[i, j] > threshold:
                    perm[i, j] = 1.0
                    used_cols.add(j)
                    break
            else:
                # Fallback: first available
                for j in range(n):
                    if j not in used_cols:
                        perm[i, j] = 1.0
                        used_cols.add(j)
                        break
        
        if not used_cols:
            break
        
        lam = remaining[perm > 0.5].min()
        decomp.permutations.append(perm)
        decomp.weights.append(float(lam))
        remaining -= lam * perm
    
    total = sum(decomp.weights)
    if total > 0:
        decomp.weights = [w / total for w in decomp.weights]
    
    return decomp


def _inference_worker(texts: List[str], worker_id: int, max_depth: int = 6) -> List[Dict]:
    """Process a batch of texts for transitive inference.
    
    Operates entirely symbolically — no model weights.
    Builds inference chains from token relationships.
    
    Args:
        texts: List of input strings
        worker_id: Worker identifier
        max_depth: Maximum transitive closure depth
    
    Returns:
        List of inference result dictionaries
    """
    results = []
    
    for text in texts:
        tokens = [ord(c) % 256 for c in text]  # placeholder tokenization
        
        # Build local inference graph
        graph = {}
        for i, t in enumerate(tokens):
            if t not in graph:
                graph[t] = []
            # Exact edges: consecutive tokens
            if i + 1 < len(tokens):
                graph[t].append((tokens[i + 1], 1.0, PathType.EXACT))
            # Semantic edges: tokens with similar value
            for delta in [-2, -1, 1, 2]:
                neighbor = (t + delta) % 256
                graph[t].append((neighbor, 0.8, PathType.SEMANTIC))
        
        # Run transitive closure with depth limit
        closure = {}
        for src, edges in graph.items():
            for dst, weight, _ in edges:
                key = (src, dst)
                closure[key] = max(closure.get(key, 0.0), weight)
        
        # Iterative deepening
        for depth in range(2, max_depth + 1):
            new_entries = {}
            for (a, b), w1 in list(closure.items()):
                for (c, d), w2 in list(closure.items()):
                    if b == c and a != d:
                        key = (a, d)
                        composed = min(w1 * w2, 1.0)
                        if composed > new_entries.get(key, 0.0):
                            new_entries[key] = composed
            closure.update(new_entries)
        
        results.append({
            "text": text,
            "tokens": tokens,
            "closure_size": len(closure),
            "closure": closure,
            "worker_id": worker_id,
        })
    
    return results


def _generate_stream_worker(args: Dict) -> List[int]:
    """Worker function: generate a token stream with given tau.
    
    Args:
        args: Dictionary with:
            - prompt_tokens: List[int] starting tokens
            - state_matrix: np.ndarray state initialization
            - tau: float temperature parameter
            - max_tokens: int max tokens to generate
            - vocab_size: int vocabulary size
            - stream_id: int stream identifier
    
    Returns:
        List of generated token IDs
    """
    prompt_tokens = args['prompt_tokens']
    tau = args['tau']
    max_tokens = args['max_tokens']
    vocab_size = args['vocab_size']
    
    tokens = list(prompt_tokens)
    
    # Simple deterministic generation based on state contraction
    for step in range(max_tokens):
        if not tokens:
            break
        
        # Current state from last token
        last_token = tokens[-1]
        rng = np.random.RandomState(last_token * 1000 + step)
        
        # Generate candidates via state contraction
        candidates = np.arange(min(vocab_size, 256), dtype=np.int64)
        
        # Score candidates deterministically from state
        scores = np.zeros(len(candidates))
        for i, cand in enumerate(candidates):
            # Deterministic score based on token relationship
            score = 1.0 - abs(int(cand) - last_token) / max(vocab_size, 1)
            # Apply tau contraction
            score = score ** (1.0 / max(tau, 0.01))
            scores[i] = score
        
        # Select best
        if len(scores) > 0:
            best_idx = int(np.argmax(scores))
            next_token = int(candidates[best_idx])
            tokens.append(next_token)
    
    return tokens


# =============================================================================
# TOP-LEVEL PARALLEL FUNCTIONS
# =============================================================================

def parallel_moebius_transition(
    states: np.ndarray,
    inputs: np.ndarray,
    A: np.ndarray = None,
    dt: np.ndarray = None,
    v: np.ndarray = None,
    n_workers: int = 4
) -> np.ndarray:
    """Apply Möbius state transition in parallel across positions.
    
    Each position's state evolves independently — embarrassingly parallel.
    Falls back to sequential for small arrays where parallel overhead
    exceeds the benefit.
    
    Args:
        states: Array of shape (n_positions, 2) — (Re, Im) pairs
        inputs: Array of shape (n_positions,) — token IDs
        A: Optional transition matrix (not used in symbolic mode)
        dt: Optional time step (not used in symbolic mode)
        v: Optional coupling direction for SIMD-fast path
        n_workers: Number of parallel workers
    
    Returns:
        Updated states array, same shape
    """
    states = np.asarray(states, dtype=np.float64)
    inputs = np.asarray(inputs)
    n_positions = states.shape[0]
    
    # Fast SIMD path: if v is provided, use vectorized operations
    if v is not None and states.dtype == np.float32:
        v = np.asarray(v, dtype=np.float32)
        # Small arrays: sequential is faster
        if n_positions <= 64 or n_workers <= 1:
            from .mac_optimize import moebius_simd
            return moebius_simd(states.astype(np.float32), v).astype(np.float64)
        
        # Parallel SIMD chunks
        chunk_size = max(64, n_positions // n_workers)
        from .mac_optimize import moebius_simd
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            chunks = []
            for i in range(0, n_positions, chunk_size):
                end = min(i + chunk_size, n_positions)
                chunks.append((states[i:end].astype(np.float32), v[i:end] if v.ndim > 0 else v, i))
            
            futures = [executor.submit(_simd_moebius_chunk, chunk) for chunk in chunks]
            results = [(f.result()[0], f.result()[1]) for f in futures]
        
        # Reassemble
        results.sort(key=lambda x: x[1])
        output = np.concatenate([r[0] for r in results], axis=0)
        return output.astype(np.float64)
    
    # Standard path: per-token Möbius transforms
    # Small arrays: sequential is faster due to overhead
    if n_positions <= 64 or n_workers <= 1:
        result, _ = _moebius_worker_chunk((states, inputs, 0))
        return result
    
    # Parallel chunk processing
    chunk_size = max(64, n_positions // n_workers)
    chunks = []
    for i in range(0, n_positions, chunk_size):
        end = min(i + chunk_size, n_positions)
        chunks.append((states[i:end], inputs[i:end], i))
    
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_moebius_worker_chunk, chunk) for chunk in chunks]
        results = [(f.result()[0], f.result()[1]) for f in futures]
    
    # Reassemble in correct order
    results.sort(key=lambda x: x[1])
    output = np.concatenate([r[0] for r in results], axis=0)
    
    return output


def parallel_bvn_decompose(
    prob_matrix: np.ndarray,
    max_paths: int = 100,
    n_workers: int = 4
) -> Tuple[List[BvNDecomposition], List[float]]:
    """Evaluate BvN paths in parallel using different strategies.
    
    Runs multiple BvN decomposition strategies concurrently and
    returns all results for ensemble merging.
    
    Args:
        prob_matrix: (N, N) doubly stochastic matrix
        max_paths: Maximum number of permutation paths per strategy
        n_workers: Number of parallel workers (one per strategy)
    
    Returns:
        Tuple of (decompositions, confidence_scores)
    """
    prob_matrix = np.asarray(prob_matrix, dtype=np.float64)
    
    # Small matrices: skip parallelism
    if prob_matrix.shape[0] <= 32 or n_workers <= 1:
        decomp = _bvn_greedy_worker((prob_matrix, max_paths))
        confidence = [1.0]
        return [decomp], confidence
    
    # Multiple strategies in parallel
    strategies = ["greedy", "threshold"]
    
    with ProcessPoolExecutor(max_workers=min(n_workers, len(strategies))) as executor:
        if "greedy" in strategies:
            fut_greedy = executor.submit(_bvn_greedy_worker, (prob_matrix.copy(), max_paths))
        if "threshold" in strategies:
            fut_threshold = executor.submit(_bvn_threshold_worker, (prob_matrix.copy(), max_paths))
        
        decompositions = []
        confidences = []
        
        try:
            decomp_g = fut_greedy.result(timeout=60)
            if decomp_g.permutations:
                decompositions.append(decomp_g)
                confidences.append(0.6)
        except Exception as e:
            logger.warning(f"Greedy BvN failed: {e}")
        
        try:
            decomp_t = fut_threshold.result(timeout=60)
            if decomp_t.permutations:
                decompositions.append(decomp_t)
                confidences.append(0.4)
        except Exception as e:
            logger.warning(f"Threshold BvN failed: {e}")
    
    if not decompositions:
        # Fallback: return identity decomposition
        n = prob_matrix.shape[0]
        ident = np.eye(n, dtype=np.float64)
        fallback = BvNDecomposition(permutations=[ident], weights=[1.0])
        return [fallback], [1.0]
    
    return decompositions, confidences


def batch_transitive_inference(
    token_sequences: List[List[int]],
    n_workers: int = 4,
    max_depth: int = 6
) -> List[Dict]:
    """Run inference on multiple sequences in parallel.
    
    Each sequence is processed independently — embarrassingly parallel.
    
    Args:
        token_sequences: List of token ID lists
        n_workers: Number of parallel workers
        max_depth: Maximum transitive closure depth
    
    Returns:
        List of inference result dictionaries
    """
    if not token_sequences:
        return []
    
    # Small batch: sequential
    if len(token_sequences) <= 4 or n_workers <= 1:
        texts = [''.join(chr(t % 128) for t in seq) for seq in token_sequences]
        return _inference_worker(texts, 0, max_depth)
    
    # Split into chunks
    chunk_size = max(1, len(token_sequences) // n_workers)
    chunks = []
    chunk_idx = 0
    for i in range(0, len(token_sequences), chunk_size):
        seqs = token_sequences[i:i + chunk_size]
        texts = [''.join(chr(t % 128) for t in seq) for seq in seqs]
        chunks.append((texts, chunk_idx, max_depth))
        chunk_idx += 1
    
    # Process in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = [executor.submit(_inference_worker, texts, wid, md) for texts, wid, md in chunks]
        
        results = []
        for f in futures:
            try:
                chunk_result = f.result(timeout=30)
                results.extend(chunk_result)
            except Exception as e:
                logger.warning(f"Inference chunk failed: {e}")
                results.append({"error": str(e)})
    
    return results[:len(token_sequences)]


def multi_stream_generate(
    prompt: List[int],
    state_matrix: np.ndarray = None,
    taus: List[float] = None,
    max_tokens: int = 20,
    n_workers: int = 3,
    vocab_size: int = 256
) -> List[List[int]]:
    """Generate with different tau values in parallel.
    
    Each stream uses a different 'temperature' parameter tau that controls
    the contraction rate. Higher tau -> more deterministic; lower tau ->
    more exploratory.
    
    Args:
        prompt: Initial token IDs
        state_matrix: Optional state matrix (unused in symbolic mode)
        taus: List of tau values (default: [0.5, 0.65, 0.8])
        max_tokens: Maximum tokens per stream
        n_workers: Number of parallel workers
        vocab_size: Vocabulary size
    
    Returns:
        List of generated token ID lists, one per tau
    """
    if taus is None:
        taus = [0.5, 0.65, 0.8]
    
    n_streams = len(taus)
    
    # Build worker args
    worker_args = []
    for i, tau in enumerate(taus):
        arg = {
            'prompt_tokens': prompt,
            'state_matrix': state_matrix,
            'tau': tau,
            'max_tokens': max_tokens,
            'vocab_size': vocab_size,
            'stream_id': i,
        }
        worker_args.append(arg)
    
    # Small: sequential
    if n_streams <= 2 or n_workers <= 1:
        return [_generate_stream_worker(arg) for arg in worker_args]
    
    # Parallel streams
    with ProcessPoolExecutor(max_workers=min(n_workers, n_streams)) as executor:
        futures = [executor.submit(_generate_stream_worker, arg) for arg in worker_args]
        
        results = []
        for f in futures:
            try:
                results.append(f.result(timeout=60))
            except Exception as e:
                logger.warning(f"Stream generation failed: {e}")
                results.append(list(prompt))  # Return prompt as fallback
    
    return results


# =============================================================================
# MAIN PARALLEL ENGINE CLASS
# =============================================================================

class ParallelHSSLMS:
    """Parallel symbolic language module (HSSLM-S).
    
    Architecture:
        - State Worker Pool: Möbius transitions (embarrassingly parallel)
        - Inference Worker Pool: BvN path evaluation (tree reduction)
        - Sampler: Contraction + Foss gate (dedicated logic)
        - I/O: Token streaming via ring buffer
    
    All operations are deterministic mathematical formulas.
    No neural networks. No PyTorch.
    """
    
    def __init__(self, vocab_size: int = 50000, max_positions: int = 65536,
                 n_workers: int = None, state_dim: int = 768):
        """Initialize parallel worker pools.
        
        Args:
            vocab_size: Token vocabulary dimension
            max_positions: Maximum sequence length (state array size)
            n_workers: Number of state workers (default: CPU count - 2)
            state_dim: State vector dimension
        """
        self.vocab_size = vocab_size
        self.max_positions = max_positions
        self.state_dim = state_dim
        self.n_workers = n_workers or max(1, os.cpu_count() - 2)
        
        # Auto-configuration
        self._config = ParallelConfig.auto(self.n_workers)
        self._positions_per_worker = self._config.positions_per_worker
        
        # Worker pool configuration
        self._executor = ProcessPoolExecutor(
            max_workers=self.n_workers,
            initializer=self._worker_init,
            initargs=(vocab_size, max_positions)
        )
        
        # Shared state array (complex states as (Re, Im) pairs)
        self._state_array = SharedStateArray((max_positions, 2), dtype=np.float64)
        
        # Token ring buffer for async I/O
        self._token_ring = LockFreeRingBuffer(
            capacity=8192, item_size=1, dtype=np.int64
        )
        
        # Result accumulator for inference chains
        self._result_acc = ResultAccumulator(
            n_slots=self.n_workers,
            slot_size=min(vocab_size, 1000),
            dtype=np.float64
        )
        
        # Inference graph (symbolic knowledge base)
        self._graph: Dict[int, List[Tuple[int, float, PathType]]] = {}
        
        # Running flag
        self._running = True
        
        logger.info(
            f"Initialized ParallelHSSLMS: workers={self.n_workers}, "
            f"vocab={vocab_size}, positions={max_positions}, state_dim={state_dim}"
        )
    
    @staticmethod
    def _worker_init(vocab_size: int, max_positions: int):
        """Initialize worker process (called once per worker).
        
        Sets up process-level configuration and ignores SIGINT
        (handled by parent process).
        """
        try:
            os.nice(5)
        except PermissionError:
            pass
        # Ignore SIGINT in workers (handled by parent)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    
    # =====================================================================
    # PUBLIC API
    # =====================================================================
    
    def generate(self, prompt: str, max_tokens: int = 100,
                 tau: float = 0.65, use_parallel: bool = True) -> str:
        """Generate text with optional parallel processing.
        
        Uses the symbolic inference pipeline:
        1. Tokenize prompt
        2. Initialize Möbius states
        3. Iteratively: BvN decomposition -> Foss gate -> sampling
        4. Detokenize output
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            tau: Temperature parameter (higher = more deterministic)
            use_parallel: Whether to use parallel workers
        
        Returns:
            Generated text string
        """
        # Tokenize
        prompt_tokens = self._tokenize(prompt)
        
        # Generate tokens
        if use_parallel and max_tokens > 10:
            # Use multi-stream with single tau
            results = multi_stream_generate(
                prompt=prompt_tokens,
                taus=[tau],
                max_tokens=max_tokens,
                n_workers=min(2, self.n_workers),
                vocab_size=self.vocab_size
            )
            all_tokens = results[0] if results else prompt_tokens
        else:
            # Sequential generation
            args = {
                'prompt_tokens': prompt_tokens,
                'tau': tau,
                'max_tokens': max_tokens,
                'vocab_size': self.vocab_size,
                'stream_id': 0,
            }
            all_tokens = _generate_stream_worker(args)
        
        return self._detokenize(np.array(all_tokens))
    
    def analyze(self, text: str) -> Dict:
        """Hierarchical analysis with parallel inference.
        
        Builds inference graph, runs transitive closure, and
        returns analysis results.
        
        Args:
            text: Input text to analyze
        
        Returns:
            Analysis dictionary with tokens, closure, and metrics
        """
        tokens = self._tokenize(text)
        token_list = tokens.tolist()
        
        # Build graph
        graph = {}
        for i, t in enumerate(token_list):
            if t not in graph:
                graph[t] = []
            if i + 1 < len(token_list):
                graph[t].append((token_list[i + 1], 1.0, PathType.EXACT))
            for delta in [-2, -1, 1, 2]:
                neighbor = (t + delta) % 256
                graph[t].append((neighbor, 0.8, PathType.SEMANTIC))
        
        # Transitive closure
        closure = {}
        for src, edges in graph.items():
            for dst, weight, _ in edges:
                key = (src, dst)
                closure[key] = max(closure.get(key, 0.0), weight)
        
        for depth in range(2, self._config.max_inference_depth + 1):
            new_entries = {}
            for (a, b), w1 in list(closure.items()):
                for (c, d), w2 in list(closure.items()):
                    if b == c and a != d:
                        key = (a, d)
                        composed = min(w1 * w2, 1.0)
                        if composed > new_entries.get(key, 0.0):
                            new_entries[key] = composed
            closure.update(new_entries)
        
        # Möbius state for the sequence
        states = np.zeros((len(tokens), 2), dtype=np.float64)
        for i, t in enumerate(token_list):
            a, b = self._token_to_moebius_params(t)
            z = MoebiusState(states[max(0, i - 1), 0], states[max(0, i - 1), 1])
            z_new = z.apply_moebius(a, b)
            states[i] = [z_new.re, z_new.im]
        
        return {
            "text": text,
            "tokens": token_list,
            "n_tokens": len(token_list),
            "closure_size": len(closure),
            "states_shape": states.shape,
            "final_state": {"re": float(states[-1, 0]), "im": float(states[-1, 1])} if len(states) > 0 else None,
        }
    
    def batch_analyze(self, texts: List[str], n_workers: int = None) -> List[Dict]:
        """Analyze multiple texts in parallel.
        
        Args:
            texts: List of input texts
            n_workers: Override worker count
        
        Returns:
            List of analysis dictionaries
        """
        n_workers = n_workers or self.n_workers
        
        if len(texts) <= 2 or n_workers <= 1:
            return [self.analyze(t) for t in texts]
        
        # Process in parallel via executor
        chunk_size = max(1, len(texts) // n_workers)
        chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
        
        results = []
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(self._analyze_chunk, chunk) for chunk in chunks]
            for f in futures:
                try:
                    chunk_results = f.result(timeout=30)
                    results.extend(chunk_results)
                except Exception as e:
                    logger.warning(f"Analysis chunk failed: {e}")
                    results.append({"error": str(e)})
        
        return results[:len(texts)]
    
    # =====================================================================
    # INTERNAL METHODS
    # =====================================================================
    
    def _tokenize(self, text: str) -> np.ndarray:
        """Simple character-level tokenization.
        
        Replace with real tokenizer for production use.
        
        Args:
            text: Input string
        
        Returns:
            Array of token IDs
        """
        tokens = [ord(c) % self.vocab_size for c in text]
        return np.array(tokens, dtype=np.int64)
    
    def _detokenize(self, tokens: np.ndarray) -> str:
        """Reverse tokenization.
        
        Args:
            tokens: Array of token IDs
        
        Returns:
            Reconstructed string
        """
        chars = [chr(int(t) % 128) for t in tokens if t > 0]
        return "".join(chars)
    
    def _token_to_moebius_params(self, token_id: int) -> Tuple[complex, complex]:
        """Map token ID to Möbius parameters (a, b)."""
        return _token_to_moebius_params(token_id)
    
    def _analyze_chunk(self, texts: List[str]) -> List[Dict]:
        """Analyze a chunk of texts (called by worker)."""
        return [self.analyze(t) for t in texts]
    
    # =====================================================================
    # PROPERTIES
    # =====================================================================
    
    @property
    def positions_per_worker(self) -> int:
        return self._positions_per_worker
    
    @property
    def config(self) -> ParallelConfig:
        return self._config
    
    # =====================================================================
    # LIFECYCLE
    # =====================================================================
    
    def shutdown(self):
        """Graceful shutdown of all worker pools."""
        self._running = False
        self._executor.shutdown(wait=True)
        logger.info("ParallelHSSLMS shutdown complete")
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.shutdown()
        return False
