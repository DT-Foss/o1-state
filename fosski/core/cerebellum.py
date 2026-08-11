"""
Cerebellum Reasoning Loop — Iterative Refinement (D2)
======================================================
Source: doom/foss-v2/modules/cerebellum.py

Predict → Compare (CASI) → Correct → Re-inject → Repeat (3-5×)

The reservoir's Echo State Property means re-injection with corrected
input produces DIFFERENT trajectories each time — this is "thinking"
without transformer layers.

Key insight: each iteration uses CASI to measure whether the answer
improved. If CASI increases → correction helped. If CASI decreases
or plateaus → stop iterating.

This replaces the transformer's depth (28 layers of sequential
processing) with iterative refinement (3-5 passes through the
same reservoir with corrected input).
"""

import numpy as np
from typing import Optional, Tuple, Callable
from .casi_gate import casi_1d


class CerebellumLoop:
    """Iterative refinement loop using reservoir re-injection."""

    def __init__(self, max_iterations: int = 5, min_improvement: float = 0.1):
        """
        Args:
            max_iterations: maximum refinement passes
            min_improvement: minimum CASI improvement to continue
        """
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement

    def refine(self, initial_answer: str, initial_confidence: float,
               query_fn: Callable, content_words: list,
               emb_store=None) -> Tuple[str, float, int]:
        """Run the cerebellum refinement loop.

        Args:
            initial_answer: first-pass answer from pipeline
            initial_confidence: first-pass confidence
            query_fn: function(content_words) → (answer, confidence)
                     Called with modified content_words each iteration
            content_words: original content words
            emb_store: embedding store for similarity computation

        Returns:
            (best_answer, best_confidence, n_iterations)
        """
        best_answer = initial_answer
        best_confidence = initial_confidence
        prev_casi = 0.0
        confidences = [initial_confidence]

        for i in range(self.max_iterations):
            # Re-inject: add previous answer as context word
            augmented_words = content_words + [best_answer.lower().split()[0]]

            # Run pipeline again with augmented input
            try:
                new_answer, new_confidence = query_fn(augmented_words)
            except Exception:
                break

            if new_answer is None:
                break

            confidences.append(new_confidence)

            # CASI check: is confidence stream structured?
            if len(confidences) >= 4:
                casi = casi_1d(np.array(confidences), n_permutations=10)
                improvement = casi - prev_casi
                prev_casi = casi

                if i > 0 and improvement < self.min_improvement:
                    break  # No improvement → stop

            # Keep best
            if new_confidence > best_confidence:
                best_answer = new_answer
                best_confidence = new_confidence

        return best_answer, best_confidence, len(confidences) - 1
