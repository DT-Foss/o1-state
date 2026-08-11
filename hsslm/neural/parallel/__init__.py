"""
HSSLM Parallel Inference Architecture

Complete parallel processing system for HSSLM-C inference on Apple Silicon (M4).

Seven parallel strategies:
    P-1: Async Inference Engine        - Process-pool with futures
    P-2: Multi-Stream Generation       - Parallel sampling at different temperatures
    P-3: Parallel Deterministic        - 3-pass inference in parallel
    P-4: Producer-Consumer Pipeline    - Stage-separated processing
    P-5: Batched Speculative Decoding  - Draft-verify pipeline
    P-6: Parallel BvN Path Evaluation  - Multi-path permutation scoring
    P-7: Shared-State Concurrent Serve - Multi-user shared weights

Usage:
    from hsslm.parallel import AsyncInferenceEngine, MultiStreamGenerator
    engine = AsyncInferenceEngine(model_path, num_workers=4)
    async for token in engine.generate("Hello world", max_tokens=100):
        print(token)
"""

__version__ = "1.0.0"
