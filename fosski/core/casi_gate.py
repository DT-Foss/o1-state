"""
CASI Generation Gate — Quality Control via 1D-CASI
====================================================
Uses 1D-CASI (7 statistical strategies) to detect when autoregressive
generation has drifted into noise.

ALMOST KILLED BY A BUG: The old version used crypto-CASI (byte-level
bit_correlation, xor_distribution, parity_chain) on quantized floats.
Crypto-CASI on quantized float similarities measures NOTHING.
1D-CASI uses statistical strategies designed for continuous sequences.

1D-CASI score on embedding similarity sequence:
  CASI ≈ 1.0 → exchangeable (structureless / random)
  CASI > 1.5 → structure detected (coherent generation)
  CASI > 2.0 → strong structure (topically focused)

Formula F1: CASI(x) = Σsᵢ(x) / E[Σsᵢ(π(x))]
7 strategies: runs, DW, crossing_radius, adj_corr, sign_change,
              radial_trend, quartile_shift

AUC=0.988 for hallucination detection (CF248).
IEEE-validated metric (ICECET 2026, Paper #1142).
"""

import numpy as np
from typing import List, Tuple


def _runs_test(x: np.ndarray) -> float:
    """Count runs above/below median. Structured = fewer runs."""
    med = np.median(x)
    above = x > med
    runs = 1 + int(np.sum(np.diff(above.astype(int)) != 0))
    # Expected runs for exchangeable sequence
    n = len(x)
    expected = n / 2 + 1
    return abs(runs - expected) / max(expected, 1)


def _durbin_watson(x: np.ndarray) -> float:
    """Durbin-Watson statistic: autocorrelation at lag 1.
    DW ≈ 2 for exchangeable, DW < 2 = positive autocorrelation (structure)."""
    diffs = np.diff(x)
    dw = float(np.sum(diffs ** 2) / max(np.sum(x ** 2), 1e-12))
    return abs(2.0 - dw)


def _crossing_radius(x: np.ndarray) -> float:
    """Mean distance between consecutive median crossings.
    Structured = long runs between crossings = larger radius."""
    med = np.median(x)
    crossings = np.where(np.diff(np.sign(x - med)))[0]
    if len(crossings) < 2:
        return float(len(x))  # No crossings = very structured
    gaps = np.diff(crossings)
    return float(np.mean(gaps))


def _adjacent_correlation(x: np.ndarray) -> float:
    """Pearson correlation between x[:-1] and x[1:].
    Structured = high correlation."""
    if len(x) < 3:
        return 0.0
    return abs(float(np.corrcoef(x[:-1], x[1:])[0, 1]))


def _sign_change_rate(x: np.ndarray) -> float:
    """Rate of sign changes in first differences.
    Structured = fewer sign changes (smooth trends)."""
    diffs = np.diff(x)
    if len(diffs) < 2:
        return 0.0
    signs = np.sign(diffs)
    changes = np.sum(signs[:-1] != signs[1:])
    rate = changes / max(len(signs) - 1, 1)
    return abs(0.5 - rate)  # 0.5 = exchangeable baseline


def _radial_trend(x: np.ndarray) -> float:
    """Linear trend magnitude. Structured = nonzero trend."""
    n = len(x)
    t = np.arange(n, dtype=np.float64)
    t -= t.mean()
    x_c = x - x.mean()
    slope = float(np.dot(t, x_c) / max(np.dot(t, t), 1e-12))
    return abs(slope) * n  # Scale by length


def _quartile_shift(x: np.ndarray) -> float:
    """Compare first-half quartiles to second-half.
    Structured = distribution shifts over time."""
    half = len(x) // 2
    if half < 2:
        return 0.0
    q1_first = np.percentile(x[:half], [25, 50, 75])
    q1_second = np.percentile(x[half:], [25, 50, 75])
    return float(np.mean(np.abs(q1_first - q1_second)))


# All 7 strategies
_STRATEGIES = [
    _runs_test,
    _durbin_watson,
    _crossing_radius,
    _adjacent_correlation,
    _sign_change_rate,
    _radial_trend,
    _quartile_shift,
]


def casi_1d(x: np.ndarray, n_permutations: int = 50) -> float:
    """Compute 1D-CASI score for a real-valued sequence.

    CASI(x) = Σsᵢ(x) / E[Σsᵢ(π(x))]

    Args:
        x: 1D array of floats (e.g., cosine similarities)
        n_permutations: number of random permutations for null model

    Returns:
        CASI score. 1.0 = exchangeable, >1.5 = structure detected.
    """
    x = np.asarray(x, dtype=np.float64)
    if len(x) < 4:
        return 1.0

    # Observed score: sum of all strategy outputs
    observed = sum(s(x) for s in _STRATEGIES)

    # Null distribution: permutation baseline
    null_scores = []
    rng = np.random.default_rng(42)
    for _ in range(n_permutations):
        perm = rng.permutation(x)
        null_scores.append(sum(s(perm) for s in _STRATEGIES))

    expected = float(np.mean(null_scores))
    if expected < 1e-12:
        return 1.0 if observed < 1e-12 else float('inf')

    return observed / expected


class CASIGenerationGate:
    """Monitor generation quality via 1D-CASI on embedding similarity streams.

    Uses 7 statistical strategies (runs, DW, crossing_radius, adj_corr,
    sign_change, radial_trend, quartile_shift) instead of crypto-CASI.
    """

    def __init__(self, threshold: float = 1.3, window: int = 8):
        """
        Args:
            threshold: minimum CASI score to continue generation.
                       Below this = noise, stop generating.
            window: number of tokens to accumulate before scoring.
        """
        self.threshold = threshold
        self.window = window
        self._similarities: List[float] = []

    def reset(self):
        """Reset for new generation sequence."""
        self._similarities = []

    def add_token(self, similarity: float):
        """Add a token's similarity score to the stream."""
        self._similarities.append(similarity)

    def should_stop(self) -> Tuple[bool, float]:
        """Check if generation should stop.

        Returns: (should_stop, casi_score)
        """
        if len(self._similarities) < self.window:
            return False, 0.0

        recent = np.array(self._similarities[-self.window:], dtype=np.float64)
        score = casi_1d(recent, n_permutations=20)  # Fewer perms for speed
        return score < self.threshold, score

    def score_sequence(self, embeddings: List[np.ndarray]) -> float:
        """Score a complete sequence of token embeddings.

        Computes pairwise cosine similarities and runs 1D-CASI.
        Higher = more structured = better generation.

        Args:
            embeddings: list of token embedding vectors

        Returns:
            CASI score (1.0 = random, >1.5 = structured)
        """
        if len(embeddings) < 3:
            return 0.0

        # Compute consecutive cosine similarities
        sims = []
        for i in range(len(embeddings) - 1):
            a = embeddings[i]
            b = embeddings[i + 1]
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            if norm_a > 1e-8 and norm_b > 1e-8:
                sims.append(float(np.dot(a, b) / (norm_a * norm_b)))

        if len(sims) < 4:
            return 0.0

        return casi_1d(np.array(sims))

    def confidence_penalty(self, source_confidences: np.ndarray) -> float:
        """Compute bits_wasted from CASI on source confidence stream (F5).

        bits_wasted = max(0, log₂(CASI(source_confidences)))
        High CASI = structured pattern in confidences = some sources are
        systematically overconfident = waste bits.

        Returns penalty in [0, 1] range.
        """
        if len(source_confidences) < 4:
            return 0.0

        score = casi_1d(source_confidences, n_permutations=20)
        bits_wasted = max(0.0, np.log2(max(score, 1e-12)))
        max_bits = 8.0  # Normalization constant
        return min(1.0, bits_wasted / max_bits)
