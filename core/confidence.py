"""
Unified Confidence System for FOSS-KI
=======================================
Every answer source emits a ConfidenceScore with comparable semantics.
Enables: early-exit, parallel evaluation, calibrated IDK threshold.

Sources:
    knowledge   — KnowledgeStore (dict-index, Hopfield)
    commonsense — CommonSenseEngine (property/taxonomy/ConceptNet)
    reasoning   — ReasoningEngine (why/how/if/math)
    multi_hop   — MultiHopReasoner (syllogism, hypothetical, causal chain)
    instructor  — InstructionExecutor → VortexRouter
    composer    — Composer (entity biography)
    solver      — Structured solvers (sequence, analogy, odd-one-out, cause/effect)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class AnswerSource(Enum):
    KNOWLEDGE = 'knowledge'
    COMMONSENSE = 'commonsense'
    REASONING = 'reasoning'
    MULTI_HOP = 'multi_hop'
    INSTRUCTOR = 'instructor'
    COMPOSER = 'composer'
    SOLVER = 'solver'
    CBR = 'cbr'
    WEB = 'web'
    UNKNOWN = 'unknown'


@dataclass
class ConfidenceScore:
    """Unified confidence score emitted by every answer source."""

    source: AnswerSource
    raw: float              # Source-native score (0-1 for most, varies by source)
    calibrated: float       # Normalized to [0, 1] — comparable across sources
    answer: str             # The actual answer text
    method: str = ''        # How the answer was derived (e.g. 'dict_exact', 'hopfield_fuzzy')
    metadata: dict = field(default_factory=dict)  # Source-specific extras

    @property
    def is_confident(self) -> bool:
        """Above the global IDK threshold."""
        return self.calibrated >= IDK_THRESHOLD

    @property
    def level(self) -> str:
        """Human-readable confidence level."""
        if self.calibrated >= 0.8:
            return 'HIGH'
        elif self.calibrated >= 0.5:
            return 'MEDIUM'
        elif self.calibrated >= 0.2:
            return 'LOW'
        return 'NONE'

    def __lt__(self, other):
        return self.calibrated < other.calibrated

    def __repr__(self):
        return (f"ConfidenceScore({self.source.value}, "
                f"cal={self.calibrated:.2f}, raw={self.raw:.2f}, "
                f"method={self.method!r})")


# Global IDK threshold — if best score across all engines is below this, say "I don't know"
IDK_THRESHOLD = 0.25


def calibrate_knowledge(result: dict) -> ConfidenceScore:
    """
    Calibrate a KnowledgeStore.query() result to a ConfidenceScore.

    Dict-exact (Tier 1/2) → 1.0
    Hopfield HIGH → 0.85
    Hopfield MEDIUM → 0.55
    Hopfield REJECTED → 0.1
    """
    fact = result.get('fact')
    conf = result.get('confidence', 0.0)
    level = result.get('confidence_level', 'UNKNOWN')
    top2_gap = result.get('top2_gap', 0.0)

    answer = ''
    if fact:
        answer = str(fact[2]) if len(fact) >= 3 else str(fact)

    # Tier 1/2: exact dict match
    if level == 'HIGH' and conf >= 0.99:
        cal = 1.0
        method = 'dict_exact'
    elif level == 'HIGH':
        # Hopfield HIGH — boost by top2_gap (discrimination confidence)
        cal = 0.75 + min(top2_gap, 0.25) * 0.4
        method = 'hopfield_high'
    elif level == 'MEDIUM':
        cal = 0.45 + min(top2_gap, 0.2) * 0.5
        method = 'hopfield_medium'
    elif level == 'REJECTED':
        cal = 0.1
        method = 'hopfield_rejected'
    else:
        cal = 0.0
        method = 'unknown'

    return ConfidenceScore(
        source=AnswerSource.KNOWLEDGE,
        raw=conf,
        calibrated=min(cal, 1.0),
        answer=answer,
        method=method,
        metadata={
            'fact': fact,
            'attractor_distance': result.get('attractor_distance'),
            'top2_gap': top2_gap,
        },
    )


def calibrate_commonsense(result: dict) -> ConfidenceScore:
    """
    Calibrate a CommonSenseEngine.query() result.

    found=True with answer → 0.7 (strong CS knowledge)
    found=True with properties only → 0.5
    found=False → 0.0
    """
    found = result.get('found', False)
    answer_text = ''

    if not found:
        return ConfidenceScore(
            source=AnswerSource.COMMONSENSE,
            raw=0.0, calibrated=0.0,
            answer='', method='no_match',
        )

    answer_val = result.get('answer')
    props = result.get('properties')
    relation = result.get('relation', '')

    if answer_val is not None:
        answer_text = str(answer_val)
        # Direct answer (yes/no, specific value) — high confidence
        cal = 0.75
        method = f'cs_direct_{relation}' if relation else 'cs_direct'
    elif props:
        answer_text = str(props)
        cal = 0.5
        method = 'cs_properties'
    else:
        cal = 0.3
        method = 'cs_found_no_answer'

    # ConceptNet assertions have weight — use it if available
    weight = result.get('weight', 1.0)
    if weight < 0.5:
        cal *= 0.7  # Low-weight ConceptNet assertion → reduce confidence

    # Low-confidence flag from ConceptNet weight filtering
    if result.get('low_confidence'):
        cal *= 0.5

    return ConfidenceScore(
        source=AnswerSource.COMMONSENSE,
        raw=weight if weight != 1.0 else cal,
        calibrated=min(cal, 1.0),
        answer=answer_text,
        method=method,
        metadata={'relation': relation, 'concept': result.get('concept')},
    )


def calibrate_reasoning(result: dict) -> ConfidenceScore:
    """
    Calibrate a ReasoningEngine.reason() result.

    confidence_level mapping:
        HIGH → 0.85
        MEDIUM → 0.6
        LOW → 0.35
        NONE → 0.0
    """
    answer_text = str(result.get('answer', ''))
    level = result.get('confidence_level', 'NONE')
    rtype = result.get('reasoning_type', 'unknown')

    cal_map = {'HIGH': 0.85, 'MEDIUM': 0.6, 'LOW': 0.35, 'NONE': 0.0}
    cal = cal_map.get(level, 0.0)

    return ConfidenceScore(
        source=AnswerSource.REASONING,
        raw=cal,
        calibrated=cal,
        answer=answer_text,
        method=rtype,
        metadata={'fiber': result.get('fiber'), 'escalated': result.get('escalated')},
    )


def calibrate_multi_hop(result: dict) -> ConfidenceScore:
    """
    Calibrate a MultiHopReasoner.reason() result.

    answered=True → base 0.65, reduced by hop count (more hops = less reliable)
    answered=False → 0.0
    """
    if not result.get('answered'):
        return ConfidenceScore(
            source=AnswerSource.MULTI_HOP,
            raw=0.0, calibrated=0.0,
            answer='', method='no_chain',
        )

    hops = result.get('hops', 1)
    method = result.get('method', 'unknown')
    answer_text = str(result.get('answer', ''))

    # More hops = less reliable: 1 hop=0.7, 2 hops=0.6, 3+=0.5
    cal = max(0.5, 0.75 - 0.05 * hops)

    return ConfidenceScore(
        source=AnswerSource.MULTI_HOP,
        raw=cal,
        calibrated=cal,
        answer=answer_text,
        method=method,
        metadata={'hops': hops, 'chain': result.get('chain')},
    )


def calibrate_instructor(result: dict) -> ConfidenceScore:
    """
    Calibrate an InstructionExecutor.execute() result.

    Router answers come with confidence from knowledge query.
    We treat instructor as a fallback — base 0.5, adjusted by answer presence.
    """
    answer_text = str(result.get('answer', ''))
    if not answer_text or answer_text == 'None':
        return ConfidenceScore(
            source=AnswerSource.INSTRUCTOR,
            raw=0.0, calibrated=0.0,
            answer='', method='no_answer',
        )

    # Instructor wraps router — moderate confidence
    cal = 0.5
    return ConfidenceScore(
        source=AnswerSource.INSTRUCTOR,
        raw=0.5,
        calibrated=cal,
        answer=answer_text,
        method='instructor_route',
        metadata={'trace': result.get('trace', [])},
    )


def calibrate_solver(answer: str, solver_type: str) -> ConfidenceScore:
    """
    Calibrate structured solver output (sequence, analogy, cause/effect, etc).

    Structured solvers are deterministic pattern-matched — high confidence when they fire.
    """
    if not answer:
        return ConfidenceScore(
            source=AnswerSource.SOLVER,
            raw=0.0, calibrated=0.0,
            answer='', method=solver_type,
        )

    # Structured solvers are reliable when they match
    cal = 0.9
    return ConfidenceScore(
        source=AnswerSource.SOLVER,
        raw=1.0,
        calibrated=cal,
        answer=str(answer),
        method=solver_type,
    )


def calibrate_cbr(answer: str) -> ConfidenceScore:
    """Calibrate Case-Based Reasoning answer. Low confidence — last resort."""
    if not answer:
        return ConfidenceScore(
            source=AnswerSource.CBR,
            raw=0.0, calibrated=0.0,
            answer='', method='cbr',
        )
    return ConfidenceScore(
        source=AnswerSource.CBR,
        raw=0.3,
        calibrated=0.3,
        answer=str(answer),
        method='cbr',
    )


def calibrate_web(result: dict) -> ConfidenceScore:
    """Calibrate web search result."""
    if not result.get('success') or not result.get('answer'):
        return ConfidenceScore(
            source=AnswerSource.WEB,
            raw=0.0, calibrated=0.0,
            answer='', method='web_fail',
        )
    return ConfidenceScore(
        source=AnswerSource.WEB,
        raw=0.6,
        calibrated=0.6,
        answer=str(result['answer']),
        method='web_search',
        metadata={'source': result.get('source', 'Web')},
    )


def best_answer(scores: List[ConfidenceScore],
                question: str = '') -> Optional[ConfidenceScore]:
    """
    Select the best answer from a list of ConfidenceScores.
    Applies answer-type guard if question is provided.

    Returns None if no score meets the IDK threshold.
    """
    valid = [s for s in scores if s.answer and s.calibrated > 0]
    if not valid:
        return None

    # Apply answer-type guard: penalize type mismatches
    if question:
        expected = classify_question_type(question)
        if expected:
            for s in valid:
                if not answer_matches_type(s.answer, expected):
                    # Penalize but don't zero — could still be best option
                    s.calibrated *= 0.4

    best = max(valid, key=lambda s: s.calibrated)
    if best.calibrated < IDK_THRESHOLD:
        return None
    return best


# ============================================================
# Answer-Type Guard
# ============================================================

class QuestionType:
    WHO = 'who'          # expects entity/person name
    WHEN = 'when'        # expects temporal expression
    WHERE = 'where'      # expects location
    HOW_MANY = 'how_many'  # expects numeric
    YES_NO = 'yes_no'    # expects yes/no/boolean


def classify_question_type(question: str) -> Optional[str]:
    """Classify what answer type a question expects."""
    q = question.lower().strip()

    if q.startswith('who ') or q.startswith('whom '):
        return QuestionType.WHO
    if q.startswith('when ') or 'what year' in q or 'what date' in q:
        return QuestionType.WHEN
    if q.startswith('where '):
        return QuestionType.WHERE
    if q.startswith('how many') or q.startswith('how much'):
        return QuestionType.HOW_MANY
    if (q.startswith(('is ', 'are ', 'was ', 'were ', 'do ', 'does ', 'did ',
                       'can ', 'could ', 'will ', 'would ', 'should ', 'has ', 'have '))
            and q.endswith('?')):
        return QuestionType.YES_NO

    return None  # unclassified — no type constraint


import re as _re

def answer_matches_type(answer: str, expected: str) -> bool:
    """Check if an answer matches the expected question type."""
    a = answer.strip()
    a_lower = a.lower()

    if expected == QuestionType.WHO:
        # Should contain a proper noun (capitalized word) or person-like token
        if _re.search(r'[A-Z][a-z]+', a):
            return True
        # Known person-related words
        if any(w in a_lower for w in ['he', 'she', 'they', 'person', 'author',
                                       'scientist', 'president', 'king', 'queen']):
            return True
        return False

    if expected == QuestionType.WHEN:
        # Should contain a year, date, or temporal word
        if _re.search(r'\b\d{4}\b', a):  # year
            return True
        if _re.search(r'\b\d{1,2}[/.-]\d{1,2}', a):  # date
            return True
        if any(w in a_lower for w in ['century', 'year', 'ago', 'bc', 'ad',
                                       'january', 'february', 'march', 'april',
                                       'may', 'june', 'july', 'august',
                                       'september', 'october', 'november', 'december',
                                       'monday', 'tuesday', 'wednesday', 'thursday',
                                       'friday', 'saturday', 'sunday',
                                       'morning', 'evening', 'night', 'noon',
                                       'spring', 'summer', 'fall', 'winter',
                                       'ancient', 'medieval', 'modern']):
            return True
        return False

    if expected == QuestionType.WHERE:
        # Should contain a place name (capitalized) or location word
        if _re.search(r'[A-Z][a-z]+', a):
            return True
        if any(w in a_lower for w in ['city', 'country', 'state', 'ocean',
                                       'river', 'mountain', 'continent',
                                       'north', 'south', 'east', 'west',
                                       'in the', 'located']):
            return True
        return False

    if expected == QuestionType.HOW_MANY:
        # Should contain a number
        if _re.search(r'\d+', a):
            return True
        # Number words
        if any(w in a_lower for w in ['zero', 'one', 'two', 'three', 'four',
                                       'five', 'six', 'seven', 'eight', 'nine',
                                       'ten', 'hundred', 'thousand', 'million',
                                       'billion', 'none', 'no ']):
            return True
        return False

    if expected == QuestionType.YES_NO:
        # Should start with yes/no or contain boolean indicator
        if a_lower.startswith(('yes', 'no', 'true', 'false')):
            return True
        return False

    return True  # unknown type — don't penalize
