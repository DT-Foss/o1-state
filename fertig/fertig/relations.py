"""Schema-driven relation extraction with conservative quality gates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

from . import primitives

Triplet = tuple[str, str, str, float]

_LEADING = {
    "a", "an", "the", "this", "that", "these", "those", "some",
    "any", "each", "every", "its", "their", "our", "your", "my",
}
_PRONOUNS = {
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "who", "which", "there", "here",
}
_BAD_TERMINALS = {
    "is", "are", "was", "were", "be", "been", "being", "can", "could",
    "may", "might", "will", "would", "shall", "should", "has", "have",
    "had", "do", "does", "did",
}
_NEGATION = re.compile(r"\b(?:not|no|never|without|rarely|seldom|neither|nor)\b", re.I)


def split_sentences(text: str) -> list[str]:
    """Split prose without requiring a space after punctuation."""

    return [s.strip() for s in re.split(r"(?<=[.!?])(?:\s+|$)|[\r\n]+", text)
            if s.strip()]


def is_negated(sentence: str) -> bool:
    """Conservative guard: reject a sentence containing explicit negation."""

    return bool(_NEGATION.search(sentence))


def normalize_entity(value: str, *, max_words: int = 12) -> Optional[str]:
    """Normalize an entity phrase while preserving years and other numbers."""

    words = re.findall(r"[a-z0-9]+(?:'[a-z]+)?", str(value).casefold())
    while words and words[0] in _LEADING:
        words.pop(0)
    while words and words[-1] in {"and", "or", "but"}:
        words.pop()
    if not words or len(words) > max_words:
        return None
    if len(words) == 1 and words[0] in _PRONOUNS:
        return None
    if words[-1] in _BAD_TERMINALS:
        return None
    if not any(len(word) >= 2 or word.isdigit() for word in words):
        return None
    return " ".join(words)


@dataclass(frozen=True)
class ExtractionStats:
    sentences: int
    emitted: int
    negated: int
    rejected_entities: int


def extract_relations(
    text: str,
    *,
    confidence_scale: float = 1.0,
    max_triplets: int = 40,
    families: Optional[Iterable[primitives.RelationFamily | str]] = None,
    with_stats: bool = False,
) -> list[Triplet] | tuple[list[Triplet], ExtractionStats]:
    """Extract canonical relations from text.

    Every regex is owned by the canonical relation registry and must expose
    named ``subject`` and ``object`` groups.  Unknown relations cannot be
    emitted by this function.
    """

    if confidence_scale < 0.0:
        raise ValueError("confidence_scale must be non-negative")
    allowed = None
    if families is not None:
        allowed = {primitives.RelationFamily(f) for f in families}

    best: dict[tuple[str, str, str], float] = {}
    sentences = split_sentences(text)
    negated = rejected = 0
    for sentence in sentences:
        if is_negated(sentence):
            negated += 1
            continue
        for spec in primitives.RELATIONS.values():
            if allowed is not None and spec.family not in allowed:
                continue
            for pattern in spec.patterns:
                match = re.search(pattern.regex, sentence, flags=re.IGNORECASE)
                if match is None:
                    continue
                subject = normalize_entity(match.group("subject"))
                obj = normalize_entity(match.group("object"))
                if subject is None or obj is None or subject == obj:
                    rejected += 1
                    break
                confidence = pattern.confidence
                if confidence is None:
                    confidence = spec.confidence
                confidence = min(1.0, max(0.0, confidence * confidence_scale))
                key = (subject, spec.name, obj)
                best[key] = max(best.get(key, 0.0), confidence)
                break

    rows = [(a, relation, b, confidence)
            for (a, relation, b), confidence in best.items()]
    rows.sort(key=lambda row: (-row[3], row[0], row[1], row[2]))
    rows = rows[:max(0, max_triplets)]
    if not with_stats:
        return rows
    return rows, ExtractionStats(
        sentences=len(sentences), emitted=len(rows), negated=negated,
        rejected_entities=rejected,
    )


def canonicalize_triplets(
    triplets: Iterable[tuple[str, str, str, float]],
    *,
    canonical_only: bool = False,
    min_confidence: float = 0.0,
) -> tuple[list[Triplet], list[Triplet]]:
    """Normalize external triplets and quarantine rejected/unknown rows."""

    accepted: dict[tuple[str, str, str], float] = {}
    rejected: list[Triplet] = []
    for raw_a, raw_relation, raw_b, raw_confidence in triplets:
        a = normalize_entity(raw_a)
        b = normalize_entity(raw_b)
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            confidence = -1.0
        canonical = primitives.canonicalize_mechanism(raw_relation)
        normalized = canonical or primitives.normalize_mechanism(raw_relation)
        raw_row: Triplet = (
            str(raw_a), str(raw_relation), str(raw_b), max(confidence, 0.0)
        )
        if (a is None or b is None or a == b or normalized is None
                or confidence < min_confidence
                or confidence > 1.0
                or (canonical_only and canonical is None)):
            rejected.append(raw_row)
            continue
        key = (a, normalized, b)
        accepted[key] = max(accepted.get(key, 0.0), confidence)
    rows = [(a, relation, b, confidence)
            for (a, relation, b), confidence in accepted.items()]
    rows.sort(key=lambda row: (row[0], row[1], row[2]))
    return rows, rejected
