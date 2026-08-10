"""
Evaluator — Multi-Dimensional Quality Assessment
==================================================
F9 in the extended capabilities framework.

pass/fail is not enough. The system needs to know:
  - HOW GOOD was the answer? (not just right/wrong)
  - Is it getting BETTER over time?
  - WHERE is it weakest?

Evaluation dimensions:
  1. Correctness: Did it get the right answer?
  2. Completeness: Did it answer the full question?
  3. Naturalness: Does it sound like a human?
  4. Speed: How fast?
  5. Confidence calibration: Does it KNOW when it's right/wrong?

All metrics are 0-1 scale. Combined into a single quality score.
"""

import time
import re
import json
import os
from typing import Dict, Any, List, Optional
from collections import defaultdict


class QualityScore:
    """Multi-dimensional quality assessment of a single response."""

    def __init__(self, query: str, response: str, domain: str = 'general'):
        self.query = query
        self.response = response
        self.domain = domain
        self.timestamp = time.time()

        # Dimensions (0-1)
        self.correctness = 0.0
        self.completeness = 0.0
        self.naturalness = 0.0
        self.speed = 0.0
        self.calibration = 0.0

        # Combined
        self.overall = 0.0

    def compute(self, expected: str = '', latency_ms: float = 0,
                confidence: float = 0.5, was_correct: bool = None) -> float:
        """
        Compute all quality dimensions.

        Args:
            expected: expected answer (for correctness check)
            latency_ms: response time in milliseconds
            confidence: system's stated confidence (0-1)
            was_correct: ground truth (None if unknown)
        """
        self.correctness = self._score_correctness(expected, was_correct)
        self.completeness = self._score_completeness()
        self.naturalness = self._score_naturalness()
        self.speed = self._score_speed(latency_ms)
        self.calibration = self._score_calibration(confidence, was_correct)

        # Weighted combination (correctness matters most)
        weights = {
            'correctness': 0.40,
            'completeness': 0.20,
            'naturalness': 0.15,
            'speed': 0.10,
            'calibration': 0.15,
        }
        self.overall = (
            weights['correctness'] * self.correctness +
            weights['completeness'] * self.completeness +
            weights['naturalness'] * self.naturalness +
            weights['speed'] * self.speed +
            weights['calibration'] * self.calibration
        )
        return self.overall

    def _score_correctness(self, expected: str,
                            was_correct: bool = None) -> float:
        """Score correctness (0-1)."""
        if was_correct is not None:
            return 1.0 if was_correct else 0.0

        if not expected:
            # No ground truth — use heuristics
            if not self.response or self.response.strip() == '':
                return 0.0
            if "don't know" in self.response.lower() or \
               "don't understand" in self.response.lower():
                return 0.3  # At least honest
            return 0.5  # Unknown

        # Compare with expected
        resp_lower = self.response.lower().strip()
        exp_lower = expected.lower().strip()

        if resp_lower == exp_lower:
            return 1.0
        if exp_lower in resp_lower:
            return 0.9  # Contains the answer
        # Word overlap
        resp_words = set(resp_lower.split())
        exp_words = set(exp_lower.split())
        if exp_words:
            overlap = len(resp_words & exp_words) / len(exp_words)
            return min(overlap, 1.0)
        return 0.0

    def _score_completeness(self) -> float:
        """Score completeness: is the response substantial enough?"""
        if not self.response:
            return 0.0

        words = self.response.split()
        n_words = len(words)

        # Too short = incomplete
        if n_words < 3:
            return 0.3
        if n_words < 10:
            return 0.6

        # Check for common incompleteness markers
        incomplete_markers = [
            "i don't know", "i'm not sure", "i cannot",
            "no information", "not available",
        ]
        for marker in incomplete_markers:
            if marker in self.response.lower():
                return 0.4

        # Sentences? More complete if multiple sentences.
        sentences = [s.strip() for s in re.split(r'[.!?]', self.response)
                     if s.strip()]
        if len(sentences) >= 2:
            return 0.9

        return 0.7

    def _score_naturalness(self) -> float:
        """Score naturalness: does it sound like a human?"""
        if not self.response:
            return 0.0

        score = 0.5  # baseline

        # Positive: starts with capital, ends with period
        if self.response[0].isupper():
            score += 0.1
        if self.response.rstrip().endswith(('.', '!', '?')):
            score += 0.1

        # Negative: template-like patterns
        robot_patterns = [
            r'^The \w+ of .+ is .+\.$',  # "The capital of X is Y."
            r'^I don\'t have (?:enough )?information',
            r'^Error:',
            r'^None$',
        ]
        for pattern in robot_patterns:
            if re.match(pattern, self.response):
                score -= 0.2

        # Positive: varied sentence structure
        sentences = re.split(r'[.!?]', self.response)
        if len(sentences) >= 2:
            lengths = [len(s.split()) for s in sentences if s.strip()]
            if lengths and max(lengths) - min(lengths) > 3:
                score += 0.1  # Varied length = more natural

        # Positive: uses common conversational words
        conversational = ['also', 'however', 'for example', 'because',
                          'which', 'although', 'indeed']
        for word in conversational:
            if word in self.response.lower():
                score += 0.05
                break

        return max(0.0, min(1.0, score))

    def _score_speed(self, latency_ms: float) -> float:
        """Score speed: faster is better, with diminishing returns."""
        if latency_ms <= 0:
            return 0.5  # Unknown

        if latency_ms < 50:
            return 1.0   # Instant
        if latency_ms < 200:
            return 0.9   # Fast
        if latency_ms < 500:
            return 0.8   # Good
        if latency_ms < 1000:
            return 0.6   # OK
        if latency_ms < 3000:
            return 0.4   # Slow
        return 0.2        # Very slow

    def _score_calibration(self, confidence: float,
                            was_correct: bool = None) -> float:
        """
        Score confidence calibration.

        Perfect calibration: high confidence → correct, low confidence → wrong.
        Worst: high confidence but wrong, or low confidence but right.
        """
        if was_correct is None:
            return 0.5  # Can't assess

        if was_correct:
            # Correct answer: higher confidence = better calibration
            return confidence
        else:
            # Wrong answer: lower confidence = better calibration
            return 1.0 - confidence

    def to_dict(self) -> Dict:
        return {
            'query': self.query[:100],
            'domain': self.domain,
            'correctness': round(self.correctness, 3),
            'completeness': round(self.completeness, 3),
            'naturalness': round(self.naturalness, 3),
            'speed': round(self.speed, 3),
            'calibration': round(self.calibration, 3),
            'overall': round(self.overall, 3),
            'timestamp': self.timestamp,
        }


class Evaluator:
    """
    Tracks quality over time across domains.

    Usage:
        ev = Evaluator()
        score = ev.evaluate("What is the capital of France?",
                            "Paris is the capital of France.",
                            expected="Paris", domain="geography")
        report = ev.trend_report()
    """

    def __init__(self, storage_path: str = None):
        self.storage_path = storage_path or os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            'data', 'evaluations.json'
        )
        self.scores: List[QualityScore] = []
        self.by_domain: Dict[str, List[QualityScore]] = defaultdict(list)
        self._load()

    def evaluate(self, query: str, response: str, expected: str = '',
                 domain: str = 'general', latency_ms: float = 0,
                 confidence: float = 0.5,
                 was_correct: bool = None) -> QualityScore:
        """Evaluate a single response."""
        qs = QualityScore(query, response, domain)
        qs.compute(expected, latency_ms, confidence, was_correct)
        self.scores.append(qs)
        self.by_domain[domain].append(qs)
        self._save()
        return qs

    def domain_report(self, domain: str = None) -> Dict[str, Any]:
        """Get quality report for a domain (or all domains)."""
        if domain:
            scores = self.by_domain.get(domain, [])
            return self._summarize(scores, domain)

        # All domains
        report = {}
        for d, scores in self.by_domain.items():
            report[d] = self._summarize(scores, d)

        # Overall
        report['_overall'] = self._summarize(self.scores, 'all')
        return report

    def trend_report(self, window: int = 20) -> Dict[str, Any]:
        """
        Show quality trends: is the system improving?

        Compares recent window vs earlier scores.
        """
        if len(self.scores) < window * 2:
            return {
                'enough_data': False,
                'total_evaluations': len(self.scores),
                'needed': window * 2,
            }

        recent = self.scores[-window:]
        earlier = self.scores[-window*2:-window]

        recent_avg = sum(s.overall for s in recent) / len(recent)
        earlier_avg = sum(s.overall for s in earlier) / len(earlier)

        improvement = recent_avg - earlier_avg
        pct_change = improvement / max(earlier_avg, 0.01)

        # Per-dimension trends
        dimensions = ['correctness', 'completeness', 'naturalness',
                       'speed', 'calibration']
        dim_trends = {}
        for dim in dimensions:
            r = sum(getattr(s, dim) for s in recent) / len(recent)
            e = sum(getattr(s, dim) for s in earlier) / len(earlier)
            dim_trends[dim] = {
                'recent': round(r, 3),
                'earlier': round(e, 3),
                'change': round(r - e, 3),
                'improving': r > e,
            }

        return {
            'enough_data': True,
            'total_evaluations': len(self.scores),
            'recent_avg': round(recent_avg, 3),
            'earlier_avg': round(earlier_avg, 3),
            'improvement': round(improvement, 3),
            'pct_change': round(pct_change * 100, 1),
            'improving': improvement > 0,
            'dimensions': dim_trends,
        }

    def weakest_domain(self) -> Optional[str]:
        """Find the domain with lowest quality scores."""
        if not self.by_domain:
            return None

        worst = None
        worst_score = 2.0  # > max possible

        for domain, scores in self.by_domain.items():
            if len(scores) >= 3:  # Need minimum data
                avg = sum(s.overall for s in scores[-10:]) / len(scores[-10:])
                if avg < worst_score:
                    worst_score = avg
                    worst = domain

        return worst

    def _summarize(self, scores: List[QualityScore],
                   label: str) -> Dict[str, Any]:
        """Summarize a list of quality scores."""
        if not scores:
            return {'label': label, 'count': 0}

        n = len(scores)
        return {
            'label': label,
            'count': n,
            'overall': round(sum(s.overall for s in scores) / n, 3),
            'correctness': round(sum(s.correctness for s in scores) / n, 3),
            'completeness': round(sum(s.completeness for s in scores) / n, 3),
            'naturalness': round(sum(s.naturalness for s in scores) / n, 3),
            'speed': round(sum(s.speed for s in scores) / n, 3),
            'calibration': round(sum(s.calibration for s in scores) / n, 3),
        }

    def _load(self):
        """Load evaluation history."""
        if os.path.exists(self.storage_path):
            try:
                with open(self.storage_path, 'r') as f:
                    data = json.load(f)
                # We only store summaries, not full QualityScore objects
                # (they're too large to persist fully)
            except (json.JSONDecodeError, KeyError):
                pass

    def _save(self):
        """Save evaluation summary to disk."""
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        # Save domain summaries (not individual scores — too large)
        data = {
            'version': 1,
            'updated': time.time(),
            'total_evaluations': len(self.scores),
            'domains': {d: self._summarize(scores, d)
                        for d, scores in self.by_domain.items()},
        }
        try:
            with open(self.storage_path, 'w') as f:
                json.dump(data, f, indent=2)
        except OSError:
            pass
