"""
HSSLM Parallel Inference - Base Infrastructure

Shared memory management, lock-free state handling, and common utilities
for all parallel strategies on Apple Silicon M4.

Key components:
    - SharedModelWeights: Read-only shared weights across processes (mmap)
    - LockFreeStateManager: Per-request state slices without locks
    - GenerationRequest/Result: IPC message types
    - WorkerProcess: Base class for all worker processes
    - cleanup_shared_memory: Resource cleanup utility
"""

from __future__ import annotations

import os
import sys
import time
import atexit
import signal
import weakref
import logging
import tempfile
import threading
import traceback
import multiprocessing as mp
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Generic, List, Optional,
    Set, Tuple, TypeVar, Union, Protocol,
)
from concurrent.futures import Future, InvalidStateError

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("hsslm.parallel")

# ---------------------------------------------------------------------------
# Device detection (Apple Silicon optimised)
# ---------------------------------------------------------------------------

def get_optimal_device() -> torch.device:
    """Return the best available device for M4 Mac."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

DEFAULT_DTYPE = torch.float16  # fp16 for M4 efficiency
STATE_DTYPE = torch.float32    # states in fp32 for stability
MAX_SEQ_LEN = 2048
VOCAB_SIZE = 16384
D_MODEL = 256
D_STATE = 16
D_INNER = 512  # D_MODEL * 2
N_LAYERS = 6

# Memory layout: per-request state buffer
# Each layer state: (B, D_INNER, D_STATE) = (1, 512, 16) = 32 KiB
# 6 layers: 192 KiB per request in fp32 -> 384 KiB
STATE_BYTES_PER_REQUEST = N_LAYERS * D_INNER * D_STATE * 4  # fp32

# ---------------------------------------------------------------------------
# Message types for IPC
# ---------------------------------------------------------------------------

class RequestType(Enum):
    GENERATE = auto()
    GENERATE_STREAM = auto()
    ANALYZE = auto()
    SHUTDOWN = auto()


@dataclass
class GenerationRequest:
    """A generation request sent to a worker."""
    request_id: int
    request_type: RequestType
    prompt_token_ids: List[int]
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    eos_token_id: Optional[int] = None
    # Optional: pre-computed prefix states (for shared prefix)
    prefix_states: Optional[List[torch.Tensor]] = None
    # For multi-stream: which stream index
    stream_index: int = 0


@dataclass
class GenerationResult:
    """Result returned from a worker."""
    request_id: int
    token_ids: List[int]
    finished: bool = True
    # Per-token metadata
    token_logprobs: Optional[List[float]] = None
    # Final states for stateful continuation
    final_states: Optional[List[torch.Tensor]] = None
    # Hierarchical analysis (if requested)
    hierarchy_info: Optional[Dict[str, Any]] = None
    # Error info
    error: Optional[str] = None


@dataclass
class StreamingToken:
    """Single token in a streaming response."""
    request_id: int
    token_id: int
    is_last: bool = False
    logprob: Optional[float] = None


# ---------------------------------------------------------------------------
# Shared Model Weights (read-only, mmap-backed)
# ---------------------------------------------------------------------------

class SharedModelWeights:
    """Manages model weights shared read-only across processes.

    On Apple Silicon, this uses file-backed mmap so multiple processes
    share the same physical memory pages (copy-on-write).

    Usage:
        weights = SharedModelWeights(model)
        # In child process:
        state_dict = weights.load()
    """

    _instances: Dict[str, SharedModelWeights] = {}
    _lock = threading.Lock()

    def __init__(self, model: nn.Module, cache_dir: Optional[str] = None) -> None:
        self.model = model
        self.cache_dir = cache_dir or tempfile.gettempdir()
        self.weight_files: Dict[str, str] = {}
        self.metadata: Dict[str, Any] = {}
        self._saved = False
        self._finalizer = None

        # Build unique key from model hash
        self.instance_id = f"hsslm_{id(model)}_{os.getpid()}"

    def save(self) -> "SharedModelWeights":
        """Save model weights to shared memory files."""
        if self._saved:
            return self

        os.makedirs(self.cache_dir, exist_ok=True)
        state_dict = self.model.state_dict()

        for name, param in state_dict.items():
            # Save each tensor to a file that can be mmap'd
            filepath = os.path.join(
                self.cache_dir, f"{self.instance_id}_{name.replace('.', '_')}.pt"
            )
            # Save in a format that allows memory mapping
            torch.save(param.contiguous().cpu(), filepath)
            self.weight_files[name] = filepath

        self.metadata = {
            "param_names": list(state_dict.keys()),
            "shapes": {k: list(v.shape) for k, v in state_dict.items()},
            "dtypes": {k: str(v.dtype) for k, v in state_dict.items()},
        }
        self._saved = True

        # Register cleanup
        self._finalizer = weakref.finalize(
            self, self._cleanup_files, self.weight_files.copy()
        )

        logger.info(
            "SharedModelWeights: saved %d tensors to %s",
            len(self.weight_files), self.cache_dir,
        )
        return self

    def load(self, device: Optional[torch.device] = None) -> Dict[str, torch.Tensor]:
        """Load weights into current process (shared memory)."""
        if not self._saved and self.model is not None:
            self.save()

        device = device or get_optimal_device()
        state_dict: Dict[str, torch.Tensor] = {}

        for name, filepath in self.weight_files.items():
            # Load with mmap for shared memory
            tensor = torch.load(
                filepath,
                map_location="cpu",
                mmap=True,  # Critical: enables shared memory
            )
            state_dict[name] = tensor.to(device)

        return state_dict

    @staticmethod
    def _cleanup_files(files: Dict[str, str]) -> None:
        """Remove temporary weight files on cleanup."""
        for filepath in files.values():
            try:
                if os.path.exists(filepath):
                    os.remove(filepath)
            except OSError:
                pass

    def cleanup(self) -> None:
        """Explicit cleanup of shared memory files."""
        self._cleanup_files(self.weight_files)
        self.weight_files.clear()
        self._saved = False


# ---------------------------------------------------------------------------
# Lock-Free State Manager
# ---------------------------------------------------------------------------

class LockFreeStateManager:
    """Manages per-request SSM states without locks.

    Each request gets a unique state slice. States are pre-allocated in
    contiguous memory for cache efficiency. No locks needed because each
    request only touches its own slice.

    Memory layout for N concurrent requests:
        [Request 0: Layer 0 State][Layer 1 State]...[Layer 5 State]
        [Request 1: Layer 0 State][Layer 1 State]...[Layer 5 State]
        ...
        [Request N-1: Layer 0 State]...[Layer 5 State]

    Each layer state: (B, D_INNER, D_STATE) typically (1, 512, 16)
    """

    def __init__(
        self,
        max_concurrent: int = 64,
        n_layers: int = N_LAYERS,
        d_inner: int = D_INNER,
        d_state: int = D_STATE,
        device: Optional[torch.device] = None,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.n_layers = n_layers
        self.d_inner = d_inner
        self.d_state = d_state
        self.device = device or get_optimal_device()

        # Pre-allocate contiguous state buffer: (max_concurrent, n_layers, d_inner, d_state)
        self._state_buffer = torch.zeros(
            max_concurrent, n_layers, d_inner, d_state,
            dtype=STATE_DTYPE, device=self.device,
        )

        # Track which slots are free using an atomic bitset
        # 0 = free, 1 = occupied
        self._slot_occupied = torch.zeros(max_concurrent, dtype=torch.bool, device="cpu")
        self._slot_lock = threading.Lock()  # Only for slot allocation, not state access

        # Request ID -> slot index mapping
        self._request_slots: Dict[int, int] = {}
        self._next_request_id = 0

    def allocate_slot(self, request_id: Optional[int] = None) -> Tuple[int, int]:
        """Allocate a free state slot. Returns (request_id, slot_index).

        Thread-safe slot allocation. Once allocated, the caller has
        exclusive access to that slot - no locks needed for state ops.
        """
        with self._slot_lock:
            # Find first free slot
            free_mask = ~self._slot_occupied
            if not free_mask.any():
                raise RuntimeError(
                    f"All {self.max_concurrent} state slots occupied"
                )
            slot_idx = free_mask.nonzero(as_tuple=True)[0][0].item()
            self._slot_occupied[slot_idx] = True

            if request_id is None:
                request_id = self._next_request_id
                self._next_request_id += 1
            self._request_slots[request_id] = slot_idx

        return request_id, slot_idx

    def get_states(self, request_id: int) -> List[torch.Tensor]:
        """Get the state list for a request (lock-free after allocation).

        Returns a list of tensors that are views into the shared buffer.
        The caller has exclusive access - no synchronization needed.
        """
        slot_idx = self._request_slots.get(request_id)
        if slot_idx is None:
            raise KeyError(f"Request {request_id} not found")

        # Return views into the buffer - each is (n_layers, d_inner, d_state)
        states = [
            self._state_buffer[slot_idx, layer_idx]
            for layer_idx in range(self.n_layers)
        ]
        return states

    def set_states(
        self, request_id: int, states: List[torch.Tensor]
    ) -> None:
        """Set states for a request (lock-free, caller has exclusive access)."""
        slot_idx = self._request_slots[request_id]
        for layer_idx, state in enumerate(states):
            self._state_buffer[slot_idx, layer_idx].copy_(state)

    def reset_states(self, request_id: int) -> None:
        """Zero out states for a request."""
        slot_idx = self._request_slots[request_id]
        self._state_buffer[slot_idx].zero_()

    def release_slot(self, request_id: int) -> None:
        """Release a state slot back to the pool."""
        with self._slot_lock:
            slot_idx = self._request_slots.pop(request_id, None)
            if slot_idx is not None:
                self._state_buffer[slot_idx].zero_()
                self._slot_occupied[slot_idx] = False

    def get_memory_usage(self) -> Dict[str, Any]:
        """Report current memory usage."""
        with self._slot_lock:
            used = self._slot_occupied.sum().item()
        return {
            "total_slots": self.max_concurrent,
            "used_slots": used,
            "free_slots": self.max_concurrent - used,
            "buffer_size_mb": (
                self._state_buffer.numel() * self._state_buffer.element_size()
                / (1024 * 1024)
            ),
            "bytes_per_request": STATE_BYTES_PER_REQUEST,
        }


# ---------------------------------------------------------------------------
# Foss Gate Quality Filter
# ---------------------------------------------------------------------------

class FossGate:
    """Quality filter for generated tokens.

    Implements the Foss Gate scoring mechanism that evaluates token
    candidates across multiple quality dimensions:
        - Confidence: softmax probability mass
        - Diversity: entropy of the distribution
        - Coherence: consistency with previous tokens

    Used by multi-stream generation to pick the best output.
    """

    def __init__(
        self,
        confidence_weight: float = 0.4,
        diversity_weight: float = 0.3,
        coherence_weight: float = 0.3,
    ) -> None:
        self.w_conf = confidence_weight
        self.w_div = diversity_weight
        self.w_coh = coherence_weight

    def score_token(
        self,
        logits: torch.Tensor,
        token_id: int,
        prev_token_id: Optional[int] = None,
    ) -> float:
        """Score a single token choice.

        Args:
            logits: (vocab_size,) logits for this position.
            token_id: The chosen token ID.
            prev_token_id: Previous token for coherence.

        Returns:
            Quality score in [0, 1]. Higher is better.
        """
        probs = torch.softmax(logits, dim=-1)

        # Confidence: probability of chosen token
        confidence = probs[token_id].item()

        # Diversity: entropy of the distribution (normalized)
        entropy = -(probs * torch.log(probs + 1e-10)).sum()
        max_entropy = torch.log(torch.tensor(probs.size(0), dtype=torch.float32))
        diversity = (entropy / max_entropy).item()

        # Coherence: how well does this token fit the context
        # (simplified: high-confidence but not too peaked)
        coherence = confidence * (1.0 - abs(diversity - 0.5) * 2)

        score = (
            self.w_conf * confidence
            + self.w_div * diversity
            + self.w_coh * coherence
        )
        return min(max(score, 0.0), 1.0)

    def select_best_stream(
        self,
        stream_results: List[Tuple[int, List[int], List[float]]],
    ) -> Tuple[int, List[int]]:
        """Select the best stream from multi-stream generation.

        Args:
            stream_results: List of (stream_idx, token_ids, scores).

        Returns:
            (best_stream_idx, best_token_ids).
        """
        best_idx = 0
        best_score = -1.0

        for stream_idx, token_ids, scores in stream_results:
            # Average score across tokens, weighted by recency
            if scores:
                weights = torch.softmax(
                    torch.arange(len(scores), dtype=torch.float32) * 0.1,
                    dim=0,
                )
                avg_score = sum(s * w.item() for s, w in zip(scores, weights))
            else:
                avg_score = 0.0

            if avg_score > best_score:
                best_score = avg_score
                best_idx = stream_idx

        return best_idx, stream_results[best_idx][1]


# ---------------------------------------------------------------------------
# Birkhoff-von Neumann Decomposition (for P-6)
# ---------------------------------------------------------------------------

def birkhoff_von_neumann_decompose(
    matrix: torch.Tensor, max_paths: int = 8
) -> List[Tuple[torch.Tensor, float]]:
    """Decompose a doubly stochastic matrix into permutation matrices.

    Args:
        matrix: (N, N) doubly stochastic matrix.
        max_paths: Maximum number of permutation paths.

    Returns:
        List of (permutation_matrix, weight) tuples.
    """
    N = matrix.shape[0]
    remaining = matrix.clone()
    paths: List[Tuple[torch.Tensor, float]] = []

    for _ in range(max_paths):
        # Find a permutation using Hungarian-like greedy approach
        perm = torch.zeros_like(remaining)
        row_remaining = remaining.clone()

        for col in range(N):
            row_vals = row_remaining[:, col]
            if row_vals.sum() > 0:
                row_idx = row_vals.argmax()
                perm[row_idx, col] = 1.0
                row_remaining[row_idx, :] = 0

        # Compute weight = minimum entry in the permutation
        weight = (remaining * perm).sum()
        if weight < 1e-6:
            break

        paths.append((perm, weight.item()))
        remaining = remaining - weight * perm
        remaining = remaining.clamp(min=0.0)

        if remaining.sum() < 1e-6:
            break

    return paths


# ---------------------------------------------------------------------------
# Moebius Confidence Merge (for P-3)
# ---------------------------------------------------------------------------

def moebius_confidence_merge(
    scores: List[torch.Tensor],
    weights: Optional[List[float]] = None,
) -> torch.Tensor:
    """Merge confidence scores from multiple passes using Moebius transform.

    Args:
        scores: List of (vocab_size,) score tensors from each pass.
        weights: Optional weights for each pass.

    Returns:
        (vocab_size,) merged scores.
    """
    if weights is None:
        weights = [1.0 / len(scores)] * len(scores)

    # Normalize weights
    total = sum(weights)
    weights = [w / total for w in weights]

    # Convert to "belief" space via logit transform
    beliefs = []
    for score, weight in zip(scores, weights):
        # Clamp to avoid infinities
        score = torch.clamp(score, 1e-6, 1 - 1e-6)
        belief = torch.log(score / (1 - score)) * weight
        beliefs.append(belief)

    # Sum beliefs (Moebius addition in logit space)
    merged_belief = sum(beliefs)

    # Convert back to probability
    merged = torch.sigmoid(merged_belief)
    merged = merged / merged.sum()  # Renormalize

    return merged


# ---------------------------------------------------------------------------
# Graceful shutdown handler
# ---------------------------------------------------------------------------

class ShutdownManager:
    """Manages graceful shutdown across processes and threads."""

    def __init__(self) -> None:
        self._shutdown_event = threading.Event()
        self._child_processes: Set[mp.Process] = set()
        self._threads: Set[threading.Thread] = set()
        self._lock = threading.Lock()

        # Install signal handlers
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, self._signal_handler)

    def _signal_handler(self, signum, frame) -> None:
        """Handle shutdown signals gracefully."""
        logger.info("Received signal %d, initiating graceful shutdown...", signum)
        self.shutdown()

    def register_process(self, proc: mp.Process) -> None:
        with self._lock:
            self._child_processes.add(proc)

    def register_thread(self, thread: threading.Thread) -> None:
        with self._lock:
            self._threads.add(thread)

    def is_shutdown(self) -> bool:
        return self._shutdown_event.is_set()

    def shutdown(self, timeout: float = 5.0) -> None:
        """Gracefully shut down all workers."""
        self._shutdown_event.set()

        # Terminate child processes
        with self._lock:
            procs = list(self._child_processes)

        for proc in procs:
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=timeout / len(procs) if procs else 1.0)
                if proc.is_alive():
                    proc.kill()
                    proc.join(timeout=1.0)

        logger.info("Shutdown complete")


# ---------------------------------------------------------------------------
# Utility: Torch multiprocessing start method
# ---------------------------------------------------------------------------

def setup_multiprocessing() -> None:
    """Configure torch multiprocessing for macOS."""
    # On macOS, 'spawn' is the default and most reliable
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass  # Already set

    # Enable MPS if available
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


# ---------------------------------------------------------------------------
# Model loader utility for workers
# ---------------------------------------------------------------------------

def load_model_for_worker(
    weight_files: Optional[Dict[str, str]] = None,
    model_class: Optional[type] = None,
    config: Optional[Dict[str, Any]] = None,
    device: Optional[torch.device] = None,
) -> nn.Module:
    """Load model in a worker process.

    Args:
        weight_files: Dict of name -> filepath for mmap loading.
        model_class: The model class to instantiate.
        config: Model configuration dict.
        device: Target device.

    Returns:
        Loaded model on target device.
    """
    device = device or get_optimal_device()

    # Import here to avoid issues in spawn context
    try:
        from hsslm.model import HSSLM
    except ImportError:
        # Fallback: add parent to path
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from hsslm.model import HSSLM

    model_class = model_class or HSSLM
    config = config or {}

    model = model_class(config).to(device)
    model.eval()

    if weight_files:
        state_dict = {}
        for name, filepath in weight_files.items():
            tensor = torch.load(filepath, map_location="cpu", mmap=True)
            state_dict[name] = tensor.to(device)
        model.load_state_dict(state_dict, strict=False)

    return model


# ---------------------------------------------------------------------------
# Performance monitoring
# ---------------------------------------------------------------------------

class PerformanceMonitor:
    """Track inference performance metrics."""

    def __init__(self, window_size: int = 100) -> None:
        self.window_size = window_size
        self._latencies: List[float] = []
        self._throughputs: List[float] = []
        self._token_counts: List[int] = []
        self._lock = threading.Lock()
        self._start_time: Optional[float] = None

    def record_request(
        self, n_tokens: int, latency_sec: float
    ) -> None:
        """Record metrics for a completed request."""
        with self._lock:
            self._latencies.append(latency_sec)
            self._token_counts.append(n_tokens)
            if latency_sec > 0:
                self._throughputs.append(n_tokens / latency_sec)

            # Keep window
            if len(self._latencies) > self.window_size:
                self._latencies = self._latencies[-self.window_size:]
                self._throughputs = self._throughputs[-self.window_size:]
                self._token_counts = self._token_counts[-self.window_size:]

    def get_stats(self) -> Dict[str, float]:
        """Get current performance statistics."""
        with self._lock:
            if not self._latencies:
                return {"requests": 0, "avg_latency_ms": 0, "avg_tokens_per_sec": 0}

            import statistics
            return {
                "requests": len(self._latencies),
                "avg_latency_ms": statistics.mean(self._latencies) * 1000,
                "p50_latency_ms": (
                    statistics.median(self._latencies) * 1000
                    if self._latencies else 0
                ),
                "p99_latency_ms": (
                    sorted(self._latencies)[int(len(self._latencies) * 0.99)] * 1000
                    if len(self._latencies) >= 100 else
                    max(self._latencies) * 1000
                ),
                "avg_tokens_per_sec": statistics.mean(self._throughputs),
                "total_tokens": sum(self._token_counts),
            }

    def reset(self) -> None:
        with self._lock:
            self._latencies.clear()
            self._throughputs.clear()
            self._token_counts.clear()


__all__ = [
    "get_optimal_device",
    "SharedModelWeights",
    "LockFreeStateManager",
    "FossGate",
    "GenerationRequest",
    "GenerationResult",
    "StreamingToken",
    "RequestType",
    "ShutdownManager",
    "PerformanceMonitor",
    "birkhoff_von_neumann_decompose",
    "moebius_confidence_merge",
    "setup_multiprocessing",
    "load_model_for_worker",
    "logger",
]
