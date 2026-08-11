"""HSSLM-S: Hybrid Symbolic-Sublinguistic Mathematical Language Model - Symbolic.

Purely symbolic, deterministic language module with:
- Mac M4 optimized SIMD operations
- Lock-free parallel processing
- Backpressure-controlled pipelines
- Resilient worker pools with circuit breakers

No PyTorch. No neural networks. Only Python standard library + NumPy.
"""

__version__ = "1.0.0"
__author__ = "HSSLM-S Team"

# Module availability
MODULES = {
    'mac_optimize': None,
    'parallel_engine': None,
    'backpressure': None,
    'worker_pool': None,
}

# Lazy imports to avoid circular dependencies and handle optional modules
def _import_modules():
    """Import all submodules."""
    try:
        from . import mac_optimize
        MODULES['mac_optimize'] = mac_optimize
    except ImportError as e:
        import warnings
        warnings.warn(f"mac_optimize not available: {e}", ImportWarning)
    
    try:
        from . import parallel_engine
        MODULES['parallel_engine'] = parallel_engine
    except ImportError as e:
        import warnings
        warnings.warn(f"parallel_engine not available: {e}", ImportWarning)
    
    try:
        from . import backpressure
        MODULES['backpressure'] = backpressure
    except ImportError as e:
        import warnings
        warnings.warn(f"backpressure not available: {e}", ImportWarning)
    
    try:
        from . import worker_pool
        MODULES['worker_pool'] = worker_pool
    except ImportError as e:
        import warnings
        warnings.warn(f"worker_pool not available: {e}", ImportWarning)


# Import on first access
_import_modules()

# Convenience re-exports (only if modules loaded)
if MODULES['mac_optimize'] is not None:
    from .mac_optimize import (
        is_apple_silicon,
        get_cpu_cores,
        moebius_simd,
        period_simd,
        load_state_mmap,
        save_state_mmap,
        set_performance_cores,
        set_efficiency_cores,
        cache_tiled_operation,
        check_accelerate,
        benchmark_moebius,
        AlignedStateBuffer,
        HotPathBuffers,
    )

if MODULES['parallel_engine'] is not None:
    from .parallel_engine import (
        LockFreeRingBuffer,
        SharedStateArray,
        parallel_moebius_transition,
        parallel_bvn_decompose,
        batch_transitive_inference,
        multi_stream_generate,
        ParallelHSSLMS,
        AtomicCounter,
        ResultAccumulator,
        ParallelConfig,
        MoebiusState,
        BvNDecomposition,
        PathType,
    )

if MODULES['backpressure'] is not None:
    from .backpressure import (
        BackpressureController,
        AdaptiveBackpressure,
        BackpressureMetrics,
        BackpressureState,
    )

if MODULES['worker_pool'] is not None:
    from .worker_pool import (
        WorkerPool,
        CircuitBreaker,
        CircuitBreakerOpen,
        should_use_parallel,
        graceful_parallel_map,
    )

__all__ = [
    # mac_optimize
    'is_apple_silicon',
    'get_cpu_cores',
    'moebius_simd',
    'period_simd',
    'load_state_mmap',
    'save_state_mmap',
    'set_performance_cores',
    'set_efficiency_cores',
    'cache_tiled_operation',
    'check_accelerate',
    'benchmark_moebius',
    'AlignedStateBuffer',
    'HotPathBuffers',
    # parallel_engine
    'LockFreeRingBuffer',
    'SharedStateArray',
    'parallel_moebius_transition',
    'parallel_bvn_decompose',
    'batch_transitive_inference',
    'multi_stream_generate',
    'ParallelHSSLMS',
    'AtomicCounter',
    'ResultAccumulator',
    'ParallelConfig',
    'MoebiusState',
    'BvNDecomposition',
    'PathType',
    # backpressure
    'BackpressureController',
    'AdaptiveBackpressure',
    'BackpressureMetrics',
    'BackpressureState',
    # worker_pool
    'WorkerPool',
    'CircuitBreaker',
    'CircuitBreakerOpen',
    'should_use_parallel',
    'graceful_parallel_map',
]
