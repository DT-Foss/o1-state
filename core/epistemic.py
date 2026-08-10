"""
Epistemic Awareness — Know What You Know (and What You Don't)
===============================================================
Claude's 4 Fähigkeiten sind: Predict, Execute, Compare, Attribute.
Was Claude ÜBERSEHEN hat: Fähigkeit 5 — EPISTEMIC AWARENESS.

A system that doesn't know what it doesn't know will:
  - Hallucinate answers instead of saying "I don't know"
  - Not prioritize learning the RIGHT gaps
  - Waste effort re-learning things it already knows

This module tracks:
  1. KNOWLEDGE CONFIDENCE: How certain is each fact? (source, age, frequency)
  2. COMPETENCE MAP: What question types can we answer well? (accuracy per domain)
  3. CALIBRATION: Are our confidence scores accurate? (predicted vs actual accuracy)
  4. KNOWLEDGE FRONTIER: What's the boundary of our knowledge?

No external dependencies.
"""

import time
import json
import os
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict


class EpistemicState:
    """
    Track what the system knows, how well it knows it,
    and where the boundaries are.
    """

    def __init__(self, log_path: str = ''):
        # Competence tracking: domain → {correct, total, last_update}
        self.competence: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {'correct': 0, 'total': 0, 'last_update': 0}
        )

        # Confidence calibration: binned predicted confidence → actual accuracy
        self.calibration_bins: Dict[str, Dict[str, int]] = {
            'HIGH': {'correct': 0, 'total': 0},
            'MEDIUM': {'correct': 0, 'total': 0},
            'LOW': {'correct': 0, 'total': 0},
        }

        # Knowledge frontier: concepts we've been asked about but couldn't answer
        self.frontier: Dict[str, Dict[str, Any]] = {}

        # Source trust: how reliable is each knowledge source
        self.source_trust: Dict[str, float] = {
            'bootstrap': 0.95,
            'commonsense': 0.90,
            'cbr': 0.70,
            'web': 0.50,
            'user': 0.80,
        }

        # Query history for calibration
        self._history: List[Dict] = []
        self._max_history = 1000

        self._log_path = log_path

    def record_query(self, query: str, domain: str, confidence: str,
                     answered: bool, correct: Optional[bool] = None,
                     source: str = ''):
        """
        Record a query result for competence tracking.

        Args:
            query: the user's question
            domain: category (factual, math, commonsense, code, etc.)
            confidence: 'HIGH', 'MEDIUM', 'LOW'
            answered: did we produce an answer?
            correct: was the answer correct? (None if unknown)
            source: where the answer came from
        """
        entry = {
            'query': query,
            'domain': domain,
            'confidence': confidence,
            'answered': answered,
            'correct': correct,
            'source': source,
            'timestamp': time.time(),
        }
        self._history.append(entry)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

        # Update competence
        comp = self.competence[domain]
        comp['total'] += 1
        if correct is True:
            comp['correct'] += 1
        comp['last_update'] = time.time()

        # Update calibration
        if confidence in self.calibration_bins and correct is not None:
            self.calibration_bins[confidence]['total'] += 1
            if correct:
                self.calibration_bins[confidence]['correct'] += 1

        # Track frontier (unanswered questions)
        if not answered:
            key = self._extract_concept(query)
            if key:
                if key not in self.frontier:
                    self.frontier[key] = {
                        'first_asked': time.time(),
                        'times_asked': 0,
                        'domain': domain,
                    }
                self.frontier[key]['times_asked'] += 1
                self.frontier[key]['last_asked'] = time.time()

    def should_answer(self, query: str, domain: str) -> Dict[str, Any]:
        """
        Should the system attempt to answer this query,
        or should it say "I don't know"?

        Returns:
            {
                'should_answer': bool,
                'confidence': float,  # 0.0 to 1.0
                'reason': str,
                'domain_accuracy': float,
            }
        """
        comp = self.competence.get(domain, {'correct': 0, 'total': 0})
        total = comp['total']
        correct = comp['correct']

        if total == 0:
            # No history for this domain — answer cautiously
            return {
                'should_answer': True,
                'confidence': 0.5,
                'reason': 'no history for domain',
                'domain_accuracy': 0.0,
            }

        accuracy = correct / total

        # If we're bad at this domain, say so
        if total >= 10 and accuracy < 0.3:
            return {
                'should_answer': False,
                'confidence': accuracy,
                'reason': f'poor accuracy in {domain}: {accuracy:.0%} ({correct}/{total})',
                'domain_accuracy': accuracy,
            }

        # Check if this specific concept is on the frontier
        concept = self._extract_concept(query)
        if concept and concept in self.frontier:
            frontier_info = self.frontier[concept]
            if frontier_info['times_asked'] >= 3:
                return {
                    'should_answer': False,
                    'confidence': 0.1,
                    'reason': f'"{concept}" asked {frontier_info["times_asked"]}× without answer',
                    'domain_accuracy': accuracy,
                }

        return {
            'should_answer': True,
            'confidence': min(accuracy + 0.2, 1.0),
            'reason': f'{domain} accuracy: {accuracy:.0%}',
            'domain_accuracy': accuracy,
        }

    def knowledge_confidence(self, concept: str, relation: str = '',
                             source: str = '') -> float:
        """
        How confident are we in a specific piece of knowledge?

        Returns 0.0 to 1.0.
        """
        base = self.source_trust.get(source, 0.5)

        # Boost if corroborated from multiple sources
        # (would need fact provenance tracking — simplified here)

        # Reduce confidence for frontier concepts
        if concept.lower() in self.frontier:
            base *= 0.5

        return min(base, 1.0)

    def competence_map(self) -> Dict[str, Dict[str, Any]]:
        """
        Return a map of domain → accuracy.
        This IS the system's self-knowledge.
        """
        result = {}
        for domain, comp in self.competence.items():
            total = comp['total']
            correct = comp['correct']
            accuracy = correct / total if total > 0 else 0.0
            result[domain] = {
                'accuracy': round(accuracy, 3),
                'total_queries': total,
                'correct': correct,
                'grade': self._grade(accuracy, total),
            }
        return result

    def calibration_report(self) -> Dict[str, Any]:
        """
        Are our confidence levels accurate?

        Perfect calibration: HIGH-confidence answers are 90%+ correct,
        MEDIUM ~60-80%, LOW ~30-50%.
        """
        report = {}
        for level, counts in self.calibration_bins.items():
            total = counts['total']
            correct = counts['correct']
            accuracy = correct / total if total > 0 else 0.0
            expected = {'HIGH': 0.9, 'MEDIUM': 0.7, 'LOW': 0.4}
            gap = accuracy - expected.get(level, 0.5)
            report[level] = {
                'accuracy': round(accuracy, 3),
                'total': total,
                'expected': expected.get(level, 0.5),
                'calibration_gap': round(gap, 3),
                'status': 'calibrated' if abs(gap) < 0.15 else
                         ('overconfident' if gap < -0.15 else 'underconfident'),
            }
        return report

    def learning_priorities(self, top_n: int = 10) -> List[Dict[str, Any]]:
        """
        What should the system learn next?
        Ranked by: frequency of failed queries × recency.
        """
        priorities = []

        # From frontier (things we couldn't answer)
        for concept, info in self.frontier.items():
            recency = 1.0 / (1.0 + (time.time() - info.get('last_asked', 0)) / 3600)
            score = info['times_asked'] * recency
            priorities.append({
                'concept': concept,
                'domain': info['domain'],
                'times_asked': info['times_asked'],
                'priority_score': round(score, 2),
                'type': 'frontier',
            })

        # From weak domains
        for domain, comp in self.competence.items():
            total = comp['total']
            correct = comp['correct']
            if total >= 5:
                accuracy = correct / total
                if accuracy < 0.7:
                    priorities.append({
                        'concept': domain,
                        'domain': domain,
                        'times_asked': total,
                        'priority_score': round((1 - accuracy) * total, 2),
                        'type': 'weak_domain',
                    })

        priorities.sort(key=lambda x: x['priority_score'], reverse=True)
        return priorities[:top_n]

    def i_dont_know(self, query: str) -> str:
        """
        Generate an honest "I don't know" response with self-awareness.
        Better than hallucinating.
        """
        concept = self._extract_concept(query)

        if concept and concept in self.frontier:
            info = self.frontier[concept]
            return (f"I don't have reliable information about {concept}. "
                    f"This has been asked {info['times_asked']} times before "
                    f"and I haven't been able to answer it yet.")

        # Check domain competence
        domain = self._guess_domain(query)
        comp = self.competence.get(domain, {'correct': 0, 'total': 0})
        if comp['total'] > 0:
            accuracy = comp['correct'] / comp['total']
            if accuracy < 0.5:
                return (f"I'm not confident in my ability to answer {domain} "
                        f"questions (accuracy: {accuracy:.0%}). "
                        f"I'd rather not guess.")

        return "I don't have enough information to answer that reliably."

    def summary(self) -> str:
        """Human-readable epistemic summary."""
        lines = ['Epistemic State Summary', '=' * 40]

        # Competence
        comp = self.competence_map()
        if comp:
            lines.append('\nCompetence by Domain:')
            for domain, info in sorted(comp.items(), key=lambda x: x[1]['accuracy'], reverse=True):
                lines.append(f'  {domain}: {info["accuracy"]:.0%} '
                           f'({info["correct"]}/{info["total_queries"]}) '
                           f'[{info["grade"]}]')

        # Calibration
        cal = self.calibration_report()
        lines.append('\nCalibration:')
        for level, info in cal.items():
            if info['total'] > 0:
                lines.append(f'  {level}: {info["accuracy"]:.0%} '
                           f'(expected {info["expected"]:.0%}) '
                           f'[{info["status"]}]')

        # Frontier
        if self.frontier:
            lines.append(f'\nKnowledge Frontier ({len(self.frontier)} concepts):')
            top = sorted(self.frontier.items(),
                        key=lambda x: x[1]['times_asked'], reverse=True)[:5]
            for concept, info in top:
                lines.append(f'  {concept}: asked {info["times_asked"]}×')

        # History
        lines.append(f'\nTotal queries tracked: {len(self._history)}')

        return '\n'.join(lines)

    def save(self, path: str = ''):
        """Save epistemic state to JSON."""
        path = path or self._log_path
        if not path:
            return
        data = {
            'competence': dict(self.competence),
            'calibration': self.calibration_bins,
            'frontier': self.frontier,
            'source_trust': self.source_trust,
            'history_size': len(self._history),
        }
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, path: str = ''):
        """Load epistemic state from JSON."""
        path = path or self._log_path
        if not path or not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        for k, v in data.get('competence', {}).items():
            self.competence[k] = v
        self.calibration_bins = data.get('calibration', self.calibration_bins)
        self.frontier = data.get('frontier', {})
        self.source_trust = data.get('source_trust', self.source_trust)

    # === Private helpers ===

    def _extract_concept(self, query: str) -> str:
        """Extract the main concept from a query."""
        q = query.lower().strip().rstrip('?')
        # Remove question words
        for prefix in ['what is ', 'who is ', 'where is ', 'tell me about ',
                       'explain ', 'how does ', 'why does ', 'why is ',
                       'what are ', 'how do ', 'can ']:
            if q.startswith(prefix):
                return q[len(prefix):].strip()
        # Take last meaningful word
        words = [w for w in q.split() if len(w) > 3]
        return words[-1] if words else ''

    def _guess_domain(self, query: str) -> str:
        """Guess the domain of a query."""
        q = query.lower()
        if any(w in q for w in ['calculate', 'solve', 'prime', 'gcd', 'factor']):
            return 'math'
        if any(w in q for w in ['code', 'function', 'class', 'program', 'debug']):
            return 'code'
        if any(w in q for w in ['capital', 'country', 'population', 'language']):
            return 'factual'
        if any(w in q for w in ['is', 'can', 'does', 'do']):
            return 'commonsense'
        if any(w in q for w in ['why', 'how', 'explain']):
            return 'reasoning'
        return 'general'

    @staticmethod
    def _grade(accuracy: float, n: int) -> str:
        """Grade competence level."""
        if n < 5:
            return 'insufficient data'
        if accuracy >= 0.9:
            return 'excellent'
        if accuracy >= 0.7:
            return 'good'
        if accuracy >= 0.5:
            return 'fair'
        return 'poor'
