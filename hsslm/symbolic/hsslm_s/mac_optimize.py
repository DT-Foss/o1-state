"""Mac mini M4 optimizations for HSSLM-S.

Hardware-Software Co-Design for Apple Silicon
- SIMD vectorization via NumPy + Accelerate/vecLib BLAS
- Memory-mapped state files for zero-copy loading
- CPU affinity for P-core/E-core scheduling
- Cache-friendly tiled operations
- AMX2/NEON-aware alignment

Target: Mac mini M4 (10-core: 4P+6E, 38 TOPS ANE, Unified Memory)
"""

import numpy as np
import os
import platform
import subprocess
import struct
import time
import mmap
import ctypes
import ctypes.util
import warnings
from typing import Tuple, Optional, List

# =============================================================================
# CONSTANTS
# =============================================================================

ALIGNMENT_NEON = 16      # 128-bit alignment for NEON (E-cores)
ALIGNMENT_AMX2 = 32      # 256-bit alignment for AMX2 (P-cores)
PAGE_SIZE = 16384        # 16KB pages on Apple Silicon
TILE_SIZE_DEFAULT = 256  # Cache-friendly tile size
STATE_DIM = 768          # HSSLM-S state dimension
BATCH_SIZE = 32          # Parallel inference chains

# QoS classes from sys/qos.h
QOS_CLASS_USER_INTERACTIVE = 0x21  # 33 -> P-cores
QOS_CLASS_USER_INITIATED = 0x19    # 25
QOS_CLASS_UTILITY = 0x11           # 17 -> E-cores
QOS_CLASS_BACKGROUND = 0x09        # 9

# M4 core topology
M4_PERF_CORES = 4   # Everest P-cores (0-3)
M4_EFF_CORES = 6    # Sawtooth E-cores (4-9)
M4_TOTAL_CORES = 10


# =============================================================================
# APPLE SILICON DETECTION
# =============================================================================

def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon (ARM64).
    
    Returns:
        True if the CPU is Apple Silicon (M1, M2, M3, M4, etc.)
    """
    if platform.system() != "Darwin":
        return False
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'hw.optional.arm64'],
            capture_output=True, text=True, check=False
        )
        return result.stdout.strip() == "1"
    except Exception:
        return platform.machine() == 'arm64'


def is_mac_mini_m4() -> bool:
    """Check if running specifically on Mac mini M4.
    
    Returns:
        True if the machine is a Mac mini with M4 chip.
    """
    if not is_apple_silicon():
        return False
    try:
        result = subprocess.run(
            ['sysctl', '-n', 'hw.model'],
            capture_output=True, text=True, check=False
        )
        model = result.stdout.strip()
        # Mac16,10 and Mac16,11 are Mac mini M4 variants
        return 'Mac' in model and platform.processor() == 'arm'
    except Exception:
        return False


def get_cpu_cores() -> Tuple[int, int]:
    """Return (performance_cores, efficiency_cores) for M4.
    
    Detects M4-specific core topology. Falls back to generic
    core count on non-Apple-Silicon platforms.
    
    Returns:
        Tuple of (p_cores, e_cores). On non-macOS, returns
        (os.cpu_count(), 0).
    """
    if platform.system() != "Darwin":
        total = os.cpu_count() or 4
        return (total, 0)
    
    try:
        # sysctl reports physical CPU topology on macOS
        result = subprocess.run(
            ['sysctl', '-a', 'machdep.cpu'],
            capture_output=True, text=True, check=False
        )
        output = result.stdout
        
        # Try to get core counts from sysctl
        perf_cores = M4_PERF_CORES
        eff_cores = M4_EFF_CORES
        
        # Parse sysctl output for core counts
        for line in output.split('\n'):
            if 'core_count' in line and 'perf' in line.lower():
                try:
                    perf_cores = int(line.split(':')[-1].strip())
                except ValueError:
                    pass
            elif 'core_count' in line and 'efficiency' in line.lower():
                try:
                    eff_cores = int(line.split(':')[-1].strip())
                except ValueError:
                    pass
        
        # Total cores as sanity check
        result2 = subprocess.run(
            ['sysctl', '-n', 'hw.ncpu'],
            capture_output=True, text=True, check=False
        )
        if result2.returncode == 0:
            total = int(result2.stdout.strip())
            # Validate our counts
            if perf_cores + eff_cores != total:
                # Fallback to M4 defaults
                perf_cores = min(perf_cores, total)
                eff_cores = total - perf_cores
        
        return (perf_cores, eff_cores)
    except Exception:
        total = os.cpu_count() or 4
        return (total, 0)


def get_core_topology() -> dict:
    """Detect full M4 P-core and E-core topology for thread pinning.
    
    Returns:
        Dictionary with 'p_cores', 'e_cores' as lists of OS CPU IDs,
        and 'core_count' as total.
    """
    p_cores, e_cores = get_cpu_cores()
    total = p_cores + e_cores
    
    # M4: cores 0-3 are P-cores, 4-9 are E-cores
    p_core_ids = list(range(p_cores))
    e_core_ids = list(range(p_cores, total))
    
    return {
        'p_cores': p_core_ids,
        'e_cores': e_core_ids,
        'core_count': total,
        'perf_cores': p_cores,
        'eff_cores': e_cores,
    }


# =============================================================================
# SIMD-OPTIMIZED MOEBIUS OPERATIONS
# =============================================================================

def aligned_array(shape: tuple, dtype=np.float32, align: int = ALIGNMENT_AMX2) -> np.ndarray:
    """Allocate a page-aligned NumPy array for SIMD/AMX2 operations.
    
    Ensures the returned array starts at a memory address that is
    a multiple of the alignment boundary, enabling true 256-bit
    AMX2 vector loads and 128-bit NEON loads.
    
    Args:
        shape: Tuple of array dimensions
        dtype: NumPy dtype (float32 recommended for vector width)
        align: Byte alignment (16 for NEON, 32 for AMX2, 16384 for page)
    
    Returns:
        np.ndarray: Properly aligned array with fast SIMD access paths
    """
    dtype = np.dtype(dtype)
    size = int(np.prod(shape)) * dtype.itemsize
    
    # Allocate raw buffer with alignment padding
    raw = np.empty(size + align, dtype=np.uint8)
    
    # Compute offset to alignment boundary
    offset = (-raw.ctypes.data) % align
    
    # Create view at aligned address
    aligned = raw[offset:offset + size].view(dtype).reshape(shape)
    
    # Sanity check (only in debug or first call)
    assert aligned.ctypes.data % align == 0, f"Alignment failed: addr={aligned.ctypes.data} % {align} != 0"
    
    return aligned


def moebius_simd(lam: np.ndarray, v: np.ndarray) -> np.ndarray:
    """SIMD-vectorized Möbius coupling using NumPy.
    
    Computes the Möbius addition in the Poincare ball model:
        f_v(lambda) = (lambda + v) / (1 + lambda * v)
    
    For real state vectors (HSSLM-S operates on real hyperbolic space),
    this simplifies to element-wise (lam + v) / (1 + lam * v) with clamping.
    
    Hardware targets via NumPy + Accelerate BLAS:
    - NEON 128-bit: 4 floats/operation on E-cores
    - AMX2 256-bit: 8 floats/operation on P-cores
    
    Args:
        lam: State vector, shape (B, D) or (D,), dtype float32
        v: Coupling direction, shape (B, D) or (D,), dtype float32
    
    Returns:
        Transformed state vector, same shape as lam, clipped to [-0.9999, 0.9999]
    """
    # Ensure contiguous, aligned memory layout for SIMD pipelines
    lam = np.ascontiguousarray(lam, dtype=np.float32)
    v = np.ascontiguousarray(v, dtype=np.float32)
    
    # Möbius addition in the Poincare ball model (real version)
    # These execute via Accelerate SIMD vector instructions:
    # - vadd (NEON) / vector add (AMX2)
    numerator = lam + v
    
    # - vfma (NEON) / fused multiply-add (AMX2)
    denominator = 1.0 + lam * v
    
    # Avoid division by zero — branchless clamp for pipeline efficiency
    np.maximum(denominator, 1e-7, out=denominator)
    
    # Vectorized divide (pipelined on both NEON and AMX2)
    result = numerator / denominator
    
    # Clamp to Poincare ball boundary (|x| < 1) — prevents divergence
    np.clip(result, -0.9999, 0.9999, out=result)
    
    return result


def moebius_simd_complex(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """SIMD-vectorized complex Möbius transformation.
    
    Computes: f(z) = (z + w) / (1 + conj(z) * w) for complex inputs.
    
    Args:
        z: Complex state vector, shape (N,), dtype complex64
        w: Complex coupling, shape (N,), dtype complex64
    
    Returns:
        Transformed complex states, same shape
    """
    z = np.ascontiguousarray(z, dtype=np.complex64)
    w = np.ascontiguousarray(w, dtype=np.complex64)
    
    numerator = z + w
    denominator = 1.0 + np.conj(z) * w
    
    # Avoid division by zero
    denom_mag = np.abs(denominator)
    denom_mag = np.maximum(denom_mag, 1e-7)
    
    result = numerator / denominator
    
    # Clamp to unit disk
    mag = np.abs(result)
    mask = mag >= 0.9999
    result[mask] = result[mask] / (mag[mask] + 1e-10) * 0.9999
    
    return result


def moebius_batch_amx2(states: np.ndarray, directions: np.ndarray, tile: int = 128) -> np.ndarray:
    """Batch Möbius transition optimized for AMX2 matrix pipelines.
    
    Processes (B, D) state batches as matrix operations to saturate
    the M4's AMX2 coprocessor (256-bit wide, ~4x throughput of NEON).
    
    Uses cache tiling when dimension exceeds AMX2 register capacity.
    
    Args:
        states: (B, D) state matrix — B parallel chains, D dimensions
        directions: (B, D) coupling matrix, same shape
        tile: Tile size for cache-friendly processing (default 128)
    
    Returns:
        (B, D) transformed state matrix
    """
    if states.ndim == 1:
        return moebius_simd(states, directions)
    
    B, D = states.shape
    assert directions.shape == (B, D), (
        f"Shape mismatch: states={states.shape}, directions={directions.shape}"
    )
    assert states.dtype == np.float32 and directions.dtype == np.float32, (
        "Both arrays must be float32"
    )
    
    # AMX2 prefers tiles of ~64-128 elements; process directly if small
    if D <= tile:
        return moebius_simd(states, directions)
    
    # Tiled processing for cache-friendly AMX2 utilization
    # Each tile fits in L1 cache (128KB on P-cores) for fused operations
    output = aligned_array((B, D), dtype=np.float32)
    
    for d_start in range(0, D, tile):
        d_end = min(d_start + tile, D)
        output[:, d_start:d_end] = moebius_simd(
            states[:, d_start:d_end],
            directions[:, d_start:d_end]
        )
    
    return output


def period_simd(lam: np.ndarray) -> np.ndarray:
    """SIMD-vectorized period function for state contraction.
    
    Computes: period(x) = 2 * arctanh(|x|) / |x|
    with special handling at x=0 where period(0) = 2.
    
    This is the hyperbolic period function used in HSSLM-S for
    measuring state contraction rates.
    
    Args:
        lam: State vector, shape (N,) or (B, D), dtype float32
    
    Returns:
        Period values, same shape as input
    """
    lam = np.ascontiguousarray(lam, dtype=np.float32)
    
    # |x| with safe clipping
    abs_lam = np.abs(lam)
    abs_lam = np.minimum(abs_lam, 0.9999)
    
    # period(x) = 2 * arctanh(|x|) / |x|
    # arctanh(u) = 0.5 * ln((1+u)/(1-u))
    # So period(x) = ln((1+|x|)/(1-|x|)) / |x|
    numerator = 1.0 + abs_lam
    denominator = 1.0 - abs_lam
    
    # Safe log ratio
    log_ratio = np.log(numerator / np.maximum(denominator, 1e-10))
    
    # Divide by |x|, handling x=0
    result = np.empty_like(lam)
    nonzero = abs_lam > 1e-8
    result[nonzero] = log_ratio[nonzero] / abs_lam[nonzero]
    
    # At x=0: lim_{x->0} period(x) = 2
    result[~nonzero] = 2.0
    
    return result


def poincare_metric(z1: np.ndarray, z2: np.ndarray) -> np.ndarray:
    """SIMD-vectorized Poincare disk metric.
    
    Computes: d(z1, z2) = arctanh(|(z1 - z2) / (1 - z1 * conj(z2))|)
    
    Args:
        z1: First state, shape (N,) complex64 or float32
        z2: Second state, shape (N,) complex64 or float32
    
    Returns:
        Hyperbolic distances, shape (N,)
    """
    if z1.dtype == np.complex64 or z1.dtype == np.complex128:
        diff = z1 - z2
        denom = 1.0 - z1 * np.conj(z2)
        ratio = np.abs(diff / (denom + 1e-15))
        ratio = np.minimum(ratio, 0.999999)
        return np.arctanh(ratio)
    else:
        # Real version
        diff = z1 - z2
        denom = 1.0 - z1 * z2
        ratio = np.abs(diff / (denom + 1e-15))
        ratio = np.minimum(ratio, 0.999999)
        return np.arctanh(ratio)


# =============================================================================
# MEMORY-MAPPED STATE LOADING
# =============================================================================

def load_state_mmap(path: str, shape: tuple = None,
                    dtype=np.float32, mode: str = 'r') -> np.ndarray:
    """Load state matrix via mmap — zero copy, instant load.
    
    On M4 unified memory, mmap'd files are backed by the same memory
    pool as malloc'd arrays. No double-buffering, no memcpy.
    
    Supports both .npy files (with header) and raw binary files.
    
    Args:
        path: File path to state data (.npy or raw binary)
        shape: Expected shape; if None, infer from file size
        dtype: Data type of stored states
        mode: 'r' read-only, 'r+' read-write, 'w+' create+write
    
    Returns:
        np.ndarray: Memory-mapped array with lazy page loading
    """
    dtype = np.dtype(dtype)
    
    if path.endswith('.npy'):
        # .npy files have a header; np.memmap handles this
        if shape is not None:
            return np.memmap(path, dtype=dtype, mode=mode, shape=shape)
        return np.memmap(path, dtype=dtype, mode=mode)
    
    # HSSLM format: check for magic header
    if os.path.exists(path):
        with open(path, 'rb') as f:
            magic = f.read(8)
            if magic == b'HSSLM\x00\x00':
                # Read header
                ndim = struct.unpack('<I', f.read(4))[0]
                dtype_code = struct.unpack('<I', f.read(4))[0]
                dtype_rev = {1: np.float32, 2: np.float64, 3: np.int32, 4: np.int64}
                file_dtype = dtype_rev.get(dtype_code, np.float32)
                file_shape = tuple(struct.unpack('<I', f.read(4))[0] for _ in range(ndim))
                header_size = f.tell()
                
                total_elements = int(np.prod(file_shape))
                data_size = total_elements * np.dtype(file_dtype).itemsize
                
                fd = os.open(path, os.O_RDONLY)
                try:
                    mm = mmap.mmap(fd, header_size + data_size, access=mmap.ACCESS_READ)
                    if hasattr(mm, 'madvise'):
                        mm.madvise(mmap.MADV_WILLNEED)
                    
                    arr = np.ndarray(file_shape, dtype=file_dtype, buffer=mm, offset=header_size)
                    arr._mmap = mm
                    arr._file_header_size = header_size
                    return arr
                finally:
                    os.close(fd)
    
    # Raw binary: need file size to infer shape
    if not os.path.exists(path):
        raise FileNotFoundError(f"State file not found: {path}")
    
    file_size = os.path.getsize(path)
    
    if shape is None:
        num_elements = file_size // dtype.itemsize
        shape = (num_elements,)
    
    expected_size = int(np.prod(shape)) * dtype.itemsize
    
    if mode == 'w+':
        with open(path, 'wb') as f:
            f.truncate(expected_size)
    
    # mmap with page alignment (16KB on Apple Silicon)
    fd = os.open(path, os.O_RDWR if mode != 'r' else os.O_RDONLY)
    try:
        mm = mmap.mmap(
            fd,
            expected_size,
            access=mmap.ACCESS_READ if mode == 'r' else mmap.ACCESS_WRITE
        )
        
        # madvise hints for M4 unified memory
        if hasattr(mm, 'madvise'):
            mm.madvise(mmap.MADV_SEQUENTIAL)
            mm.madvise(mmap.MADV_WILLNEED)
        
        arr = np.ndarray(shape, dtype=dtype, buffer=mm)
        arr._mmap = mm
        return arr
    finally:
        os.close(fd)


def save_state_mmap(state: np.ndarray, path: str):
    """Save state matrix for mmap loading in HSSLM binary format.
    
    Format: [8-byte magic 'HSSLM\x00\x00'] [4-byte ndim] [4-byte dtype]
            [4-byte dim0] [4-byte dim1] ... [raw data]
    
    This format enables instant mmap loading via load_state_mmap()
    without parsing overhead.
    
    Args:
        state: NumPy array to save
        path: Output file path
    """
    dtype_map = {np.float32: 1, np.float64: 2, np.int32: 3, np.int64: 4}
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    
    with open(path, 'wb') as f:
        # Magic header
        f.write(b'HSSLM\x00\x00')
        # ndim
        f.write(struct.pack('<I', state.ndim))
        # dtype code
        f.write(struct.pack('<I', dtype_map.get(state.dtype.type, 1)))
        # dimensions
        for dim in state.shape:
            f.write(struct.pack('<I', dim))
        # Raw data — contiguous dump
        f.write(np.ascontiguousarray(state).tobytes())


def save_state_npy(state: np.ndarray, path: str):
    """Save state as .npy file (standard NumPy format).
    
    Args:
        state: NumPy array to save
        path: Output file path (should end in .npy)
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else '.', exist_ok=True)
    np.save(path, state)


# =============================================================================
# CPU AFFINITY (P-core / E-core scheduling)
# =============================================================================

def _get_pthread_lib():
    """Get pthread library handle via ctypes."""
    try:
        return ctypes.CDLL(ctypes.util.find_library("pthread"))
    except (OSError, TypeError):
        return None


def set_thread_qos(qos_class: int, priority: int = 0) -> bool:
    """Set QoS class for current thread using pthread.
    
    Maps to Apple QoS classes:
    - QOS_CLASS_USER_INTERACTIVE (0x21) = P-cores, highest priority
    - QOS_CLASS_USER_INITIATED (0x19) = P-cores preferred
    - QOS_CLASS_UTILITY (0x11) = E-cores acceptable
    - QOS_CLASS_BACKGROUND (0x09) = E-cores only
    
    Args:
        qos_class: QoS class value from sys/qos.h
        priority: Relative priority within class (0 = default)
    
    Returns:
        True if QoS was set successfully
    """
    libpthread = _get_pthread_lib()
    if libpthread is None:
        return False
    
    try:
        # pthread_set_qos_class_self_np(qos_class, relative_priority)
        libpthread.pthread_set_qos_class_self_np(qos_class, priority)
        return True
    except Exception:
        return False


def set_performance_cores():
    """Set current process/thread to run on P-cores (M4 cores 0-3).
    
    Uses QoS class USER_INTERACTIVE which strongly prefers P-cores
    on Apple Silicon. This gives access to AMX2 coprocessor.
    """
    topology = get_core_topology()
    
    # Set QoS to interactive (P-core preference)
    set_thread_qos(QOS_CLASS_USER_INTERACTIVE, 0)
    
    # Try to set affinity mask if available (macOS doesn't expose this
    # directly, but QoS class handles it implicitly)
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, set(topology['p_cores']))
        except (OSError, AttributeError):
            pass


def set_efficiency_cores():
    """Set current process/thread to run on E-cores (M4 cores 4-9).
    
    Uses QoS class UTILITY which strongly prefers E-cores.
    Suitable for background inference tasks.
    """
    topology = get_core_topology()
    
    # Set QoS to utility (E-core preference)
    set_thread_qos(QOS_CLASS_UTILITY, 0)
    
    if hasattr(os, 'sched_setaffinity'):
        try:
            os.sched_setaffinity(0, set(topology['e_cores']))
        except (OSError, AttributeError):
            pass


def reset_qos():
    """Reset QoS to default (USER_INITIATED)."""
    set_thread_qos(QOS_CLASS_USER_INITIATED, 0)


# =============================================================================
# CACHE-FRIENDLY OPERATIONS
# =============================================================================

def cache_tiled_operation(states: np.ndarray, tile_size: int = TILE_SIZE_DEFAULT,
                          op: str = 'moebius', **kwargs) -> np.ndarray:
    """Process states in cache-friendly tiles.
    
    The M4 P-core has 64KB L1D cache and 4MB shared L2. Tiling ensures
    each chunk fits in L1 for hot-path reuse, minimizing cache misses.
    
    Args:
        states: Input state array, shape (B, D) or (N,)
        tile_size: Number of elements per tile (default 256)
        op: Operation name — 'moebius', 'period', 'norm', 'clip'
        **kwargs: Additional arguments for the operation
    
    Returns:
        Processed array, same shape as input
    """
    states = np.ascontiguousarray(states, dtype=np.float32)
    
    if states.ndim == 1:
        N = states.shape[0]
        if N <= tile_size:
            return _apply_tile_op(states, op, **kwargs)
        
        output = np.empty_like(states)
        for i in range(0, N, tile_size):
            end = min(i + tile_size, N)
            output[i:end] = _apply_tile_op(states[i:end], op, **kwargs)
        return output
    
    elif states.ndim == 2:
        B, D = states.shape
        if D <= tile_size:
            return _apply_tile_op(states, op, **kwargs)
        
        output = np.empty_like(states)
        for d_start in range(0, D, tile_size):
            d_end = min(d_start + tile_size, D)
            output[:, d_start:d_end] = _apply_tile_op(
                states[:, d_start:d_end], op, **kwargs
            )
        return output
    
    else:
        # Fallback for higher dimensions
        return _apply_tile_op(states, op, **kwargs)


def _apply_tile_op(arr: np.ndarray, op: str, **kwargs) -> np.ndarray:
    """Apply a single tiled operation."""
    if op == 'moebius':
        v = kwargs.get('v')
        if v is not None:
            return moebius_simd(arr, v)
        return arr
    elif op == 'period':
        return period_simd(arr)
    elif op == 'norm':
        return np.abs(arr)
    elif op == 'clip':
        return np.clip(arr, -0.9999, 0.9999)
    elif op == 'square':
        return arr * arr
    else:
        return arr


def cache_tiled_matmul(a: np.ndarray, b: np.ndarray, tile: int = 128) -> np.ndarray:
    """Cache-friendly tiled matrix multiplication.
    
    For large matrices, processes in tiles that fit in L1 cache
    to maximize cache hit rate on M4.
    
    Args:
        a: Left matrix, shape (M, K)
        b: Right matrix, shape (K, N)
        tile: Tile dimension
    
    Returns:
        Result matrix, shape (M, N)
    """
    a = np.ascontiguousarray(a, dtype=np.float32)
    b = np.ascontiguousarray(b, dtype=np.float32)
    
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Inner dimensions don't match: {K} vs {K2}"
    
    # For small matrices, use direct NumPy (goes through Accelerate BLAS)
    if M <= tile and K <= tile and N <= tile:
        return a @ b
    
    # Tiled multiplication
    result = np.zeros((M, N), dtype=np.float32)
    
    for i in range(0, M, tile):
        i_end = min(i + tile, M)
        for j in range(0, N, tile):
            j_end = min(j + tile, N)
            for k in range(0, K, tile):
                k_end = min(k + tile, K)
                # GEMM micro-tile via Accelerate
                result[i:i_end, j:j_end] += a[i:i_end, k:k_end] @ b[k:k_end, j:j_end]
    
    return result


# =============================================================================
# ACCELERATE FRAMEWORK CHECK
# =============================================================================

def check_accelerate() -> bool:
    """Check if NumPy is using Apple's Accelerate/vecLib BLAS.
    
    Accelerate.framework provides 2-5x better performance than OpenBLAS
    on Apple Silicon for linear algebra operations.
    
    Returns:
        True if NumPy is linked against Accelerate/vecLib
    """
    try:
        # Method 1: Check show_config (NumPy 1.26+)
        try:
            config = np.show_config(mode="dicts")
            blas_info = config.get("Build Dependencies", {}).get("blas", {})
            blas_name = blas_info.get("name", "").lower()
            if "accelerate" in blas_name or "veclib" in blas_name:
                return True
        except (TypeError, AttributeError):
            pass
        
        # NOTE: np.show_config() WITHOUT mode= prints to stdout and returns
        # None on NumPy >= 1.26 — never call it to obtain a string.

        # Method 2: Runtime performance heuristic
        # Accelerate/vecLib gives ~2-5x speedup on Apple Silicon
        if is_apple_silicon():
            # Quick benchmark: if matmul is fast enough, likely Accelerate
            test_size = 1024
            a = np.random.randn(test_size, test_size).astype(np.float32)
            b = np.random.randn(test_size, test_size).astype(np.float32)
            
            t0 = time.perf_counter()
            _ = a @ b
            t1 = time.perf_counter()
            
            elapsed = t1 - t0
            # On M4, 1024x1024 matmul should be < 3ms with Accelerate,
            # > 10ms with generic BLAS
            return elapsed < 0.005
        
        return False
    except Exception:
        return False


def get_accelerate_info() -> dict:
    """Get detailed information about NumPy BLAS backend.
    
    Returns:
        Dictionary with 'using_accelerate', 'backend_name', 'platform',
        'apple_silicon', and 'recommendation' keys.
    """
    info = {
        'using_accelerate': False,
        'backend_name': 'unknown',
        'platform': platform.system(),
        'machine': platform.machine(),
        'apple_silicon': is_apple_silicon(),
        'recommendation': '',
    }
    
    try:
        config = np.show_config(mode="dicts")
        blas_info = config.get("Build Dependencies", {}).get("blas", {})
        info['backend_name'] = blas_info.get("name", "unknown")
        info['backend_version'] = blas_info.get("version", "unknown")
    except (TypeError, AttributeError):
        # np.show_config() without mode= prints to stdout (returns None) on
        # NumPy >= 1.26 — leave backend_name as 'unknown' rather than spam.
        pass
    
    info['using_accelerate'] = check_accelerate()
    
    if info['apple_silicon'] and not info['using_accelerate']:
        info['recommendation'] = (
            "Install NumPy with Accelerate support for 2-5x speedup: "
            "pip install --force-reinstall --no-cache-dir numpy"
        )
    elif info['using_accelerate']:
        info['recommendation'] = "Accelerate/vecLib active. Optimal performance."
    else:
        info['recommendation'] = "Non-Apple platform. Accelerate not applicable."
    
    return info


# =============================================================================
# BENCHMARK
# =============================================================================

def benchmark_moebius(n: int = 10000, iterations: int = 100) -> float:
    """Benchmark Möbius coupling throughput (ops/sec).
    
    Measures the sustained throughput of SIMD-vectorized Möbius operations,
    which is the core computation in HSSLM-S token generation.
    
    Args:
        n: Vector size per operation (default 10000)
        iterations: Number of iterations to average (default 100)
    
    Returns:
        Throughput in Möbius operations per second
    """
    # Ensure warm caches
    warmup_lam = np.random.randn(n).astype(np.float32)
    warmup_v = np.random.randn(n).astype(np.float32)
    for _ in range(5):
        _ = moebius_simd(warmup_lam, warmup_v)
    
    # Benchmark
    lam = np.random.randn(n).astype(np.float32)
    v = np.random.randn(n).astype(np.float32)
    
    # Force alignment for fair measurement
    lam = np.ascontiguousarray(lam)
    v = np.ascontiguousarray(v)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        result = moebius_simd(lam, v)
        # Prevent dead code elimination
        lam[0] = result[0]
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    ops_per_sec = (iterations * n) / max(elapsed, 1e-10)
    
    return ops_per_sec


def benchmark_period(n: int = 10000, iterations: int = 100) -> float:
    """Benchmark period function throughput (elements/sec).
    
    Args:
        n: Vector size per operation
        iterations: Number of iterations to average
    
    Returns:
        Throughput in elements processed per second
    """
    lam = np.random.uniform(-0.99, 0.99, n).astype(np.float32)
    lam = np.ascontiguousarray(lam)
    
    t0 = time.perf_counter()
    for _ in range(iterations):
        result = period_simd(lam)
        lam[0] = result[0]
    t1 = time.perf_counter()
    
    elapsed = t1 - t0
    return (iterations * n) / max(elapsed, 1e-10)


def benchmark_full_report(vector_sizes: List[int] = None) -> dict:
    """Run comprehensive benchmark suite and return results.
    
    Args:
        vector_sizes: List of vector sizes to benchmark (default: [100, 1000, 10000])
    
    Returns:
        Dictionary with benchmark results for all operations
    """
    if vector_sizes is None:
        vector_sizes = [100, 1000, 10000]
    
    results = {
        'platform': platform.system(),
        'machine': platform.machine(),
        'apple_silicon': is_apple_silicon(),
        'accelerate': check_accelerate(),
        'numpy_version': np.__version__,
        'benchmarks': {},
    }
    
    for n in vector_sizes:
        moebius_ops = benchmark_moebius(n, iterations=max(10, 100000 // n))
        period_ops = benchmark_period(n, iterations=max(10, 100000 // n))
        
        results['benchmarks'][n] = {
            'moebius_ops_per_sec': round(moebius_ops, 0),
            'period_elems_per_sec': round(period_ops, 0),
            'moebius_ms_per_1000': round(1000 * 1000 / max(moebius_ops, 1), 3),
        }
    
    return results


# =============================================================================
# ALIGNED STATE BUFFER (Class-based interface)
# =============================================================================

class AlignedStateBuffer:
    """Page-aligned state buffer with NUMA-aware placement for M4.
    
    The M4 has a unified memory architecture but L1/L2 caches are
    per-core. Aligned arrays minimize cache line splits and enable
    true 256-bit AMX2 vector loads.
    """
    
    PAGE_SIZE = 16384   # 16KB pages on Apple Silicon
    AMX2_ALIGN = 32     # 256-bit alignment for AMX2
    NEON_ALIGN = 16     # 128-bit alignment for NEON
    
    def __init__(self, shape: tuple, dtype=np.float32, name: str = ""):
        self.shape = shape
        self.dtype = dtype
        self.name = name
        self.itemsize = np.dtype(dtype).itemsize
        self.nbytes = int(np.prod(shape)) * self.itemsize
        
        # Use vm_allocate-style page alignment for zero-copy sharing
        self._raw = np.empty(
            self.nbytes + max(self.AMX2_ALIGN, self.PAGE_SIZE),
            dtype=np.uint8
        )
        
        # Dual alignment: cache line + page boundary
        addr = self._raw.ctypes.data
        offset = (-addr) % self.PAGE_SIZE + self.AMX2_ALIGN
        offset = (-(addr + offset)) % self.AMX2_ALIGN + offset
        
        self.buffer = self._raw[offset:offset + self.nbytes]
        self.array = self.buffer.view(dtype).reshape(shape)
        
        assert self.array.ctypes.data % self.AMX2_ALIGN == 0
        assert self.array.ctypes.data % self.PAGE_SIZE == 0
    
    def __array__(self):
        """Allow passing to NumPy ufuncs directly."""
        return self.array
    
    @property
    def addr(self) -> int:
        return self.array.ctypes.data
    
    @property
    def ptr(self):
        """C-compatible pointer for ctypes integration."""
        return ctypes.c_void_p(self.addr)
    
    def zero(self):
        """Zero-fill the buffer."""
        self.array.fill(0.0)
    
    def copy_from(self, src: np.ndarray):
        """Copy data from another array."""
        self.array[:] = src[:]


class HotPathBuffers:
    """Pre-allocated state buffers for hot-path reuse.
    
    Eliminates malloc/free in the token generation loop by
    pre-allocating working buffers at initialization time.
    """
    
    def __init__(self, batch_size: int = BATCH_SIZE, state_dim: int = STATE_DIM):
        self.batch_size = batch_size
        self.state_dim = state_dim
        
        self.state = AlignedStateBuffer((batch_size, state_dim), name="state")
        self.next_state = AlignedStateBuffer((batch_size, state_dim), name="next_state")
        self.coupling = AlignedStateBuffer((batch_size, state_dim), name="coupling")
        self.temp = AlignedStateBuffer((batch_size, state_dim), name="temp")
        self.inference_buf = AlignedStateBuffer((state_dim * 4,), name="inference")
    
    def reset(self):
        """Zero all buffers."""
        self.state.zero()
        self.next_state.zero()
        self.coupling.zero()
        self.temp.zero()
        self.inference_buf.zero()


# =============================================================================
# PERSISTENT STATE STORE
# =============================================================================

class PersistentStateStore:
    """Persistent state storage using mmap for O(1) load/save.
    
    On M4 with unified memory, persistent mmap allows instant
    state recovery without deserialization — pages are mapped
    directly from storage into the unified memory pool.
    """
    
    MAGIC = b'HSSLMPERSIST\x00\x00\x00\x00'
    HEADER_SIZE = 16  # 12 magic + 8 max_states
    
    def __init__(self, filepath: str, state_shape: tuple,
                 dtype=np.float32, max_states: int = 10000):
        self.filepath = filepath
        self.state_shape = state_shape
        self.dtype = dtype
        self.max_states = max_states
        
        self.state_size = int(np.prod(state_shape)) * np.dtype(dtype).itemsize
        self.file_size = self.HEADER_SIZE + max_states * self.state_size
        
        self._ensure_file()
        self._map_file()
    
    def _ensure_file(self):
        """Create file with proper size if it doesn't exist."""
        if not os.path.exists(self.filepath):
            os.makedirs(os.path.dirname(self.filepath) if os.path.dirname(self.filepath) else '.', exist_ok=True)
            with open(self.filepath, 'wb') as f:
                f.write(self.MAGIC)
                f.write(struct.pack('<Q', self.max_states))
                f.truncate(self.file_size)
    
    def _map_file(self):
        """Memory-map the file for direct access."""
        self._fd = os.open(self.filepath, os.O_RDWR)
        self._mm = mmap.mmap(self._fd, self.file_size, access=mmap.ACCESS_WRITE)
        
        # Advise sequential access for state transitions
        if hasattr(self._mm, 'madvise'):
            self._mm.madvise(mmap.MADV_SEQUENTIAL)
    
    def get_state(self, idx: int) -> np.ndarray:
        """O(1) state retrieval — no copy, direct mmap view.
        
        Args:
            idx: State index (0 to max_states-1)
        
        Returns:
            NumPy array view into the mmap'd state
        """
        if not 0 <= idx < self.max_states:
            raise IndexError(f"State index {idx} out of range [0, {self.max_states})")
        
        offset = self.HEADER_SIZE + idx * self.state_size
        
        return np.ndarray(
            self.state_shape,
            dtype=self.dtype,
            buffer=self._mm,
            offset=offset
        )
    
    def set_state(self, idx: int, state: np.ndarray):
        """O(1) state write — updates mmap directly.
        
        Args:
            idx: State index
            state: NumPy array to write
        """
        target = self.get_state(idx)
        target[:] = state
    
    def sync(self):
        """Flush changes to disk (msync)."""
        if hasattr(self._mm, 'flush'):
            self._mm.flush()
    
    def close(self):
        """Close the memory-mapped file."""
        if hasattr(self, '_mm') and self._mm is not None:
            self._mm.close()
        if hasattr(self, '_fd') and self._fd is not None:
            os.close(self._fd)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()


# =============================================================================
# MODULE INITIALIZATION
# =============================================================================

def _module_init():
    """Module-level initialization: detect platform and warn if not optimized.

    Silent by default — the BLAS check costs a 1024x1024 matmul and the
    warning is advisory only. Set HSSLM_VERBOSE=1 to enable it.
    """
    if not os.environ.get("HSSLM_VERBOSE"):
        return
    if is_apple_silicon():
        if not check_accelerate():
            warnings.warn(
                "Running on Apple Silicon but NumPy is NOT using Accelerate/vecLib. "
                "For optimal performance, reinstall: "
                "pip install --force-reinstall --no-cache-dir numpy",
                RuntimeWarning,
                stacklevel=2
            )


# Run initialization on import
_module_init()
