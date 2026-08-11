"""Canonical semantic relations used by extraction, graphs, and QA.

The registry is deliberately small and explicit.  Surface forms map into a
canonical relation; an unknown mechanism remains unknown and is never silently
coerced into a known relation.  This makes schema coverage measurable and keeps
the graph honest.

The mathematical material in ``Formeln`` informs control and evaluation, not
word meaning.  In particular, digital roots are not semantic features: the
vortexmath notes themselves report that mod-9 position classes are unsuitable
for language and degrade PPM performance.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Iterable, Mapping, Optional


class RelationFamily(str, Enum):
    DEFINITION = "definition"
    PROPERTY = "property"
    PART_WHOLE = "part_whole"
    MATERIAL = "material"
    LOCATION = "location"
    FUNCTION = "function"
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    COMPARATIVE = "comparative"
    ASSOCIATION = "association"


@dataclass(frozen=True)
class ExtractionPattern:
    """Regex with named ``subject`` and ``object`` capture groups."""

    regex: str
    confidence: Optional[float] = None

    def __post_init__(self) -> None:
        compiled = re.compile(self.regex, flags=re.IGNORECASE)
        names = compiled.groupindex
        if "subject" not in names or "object" not in names:
            raise ValueError(
                "relation patterns require named subject and object groups"
            )


@dataclass(frozen=True)
class RelationSpec:
    name: str
    family: RelationFamily
    confidence: float
    aliases: tuple[str, ...] = ()
    question_markers: tuple[str, ...] = ()
    patterns: tuple[ExtractionPattern, ...] = ()
    inverse: Optional[str] = None

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"invalid canonical relation: {self.name!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"invalid confidence for {self.name!r}")


_SUBJECT = r"(?P<subject>[a-z0-9][a-z0-9'()\-/ ]{0,119}?)"
_OBJECT = r"(?P<object>[a-z0-9][a-z0-9'()\-/ ]{0,119}?)"
_END = r"\s*(?:[,;.]|\b(?:that|which|who)\b|$)"


def _p(middle: str, *, confidence: Optional[float] = None) -> ExtractionPattern:
    return ExtractionPattern(
        rf"^\s*{_SUBJECT}\s+{middle}\s+{_OBJECT}{_END}", confidence
    )


def _comparative(name: str, adjective: str, inverse: str) -> RelationSpec:
    return RelationSpec(
        name=name,
        family=RelationFamily.COMPARATIVE,
        confidence=0.50,
        aliases=(adjective,),
        question_markers=(adjective.replace("_than", ""), "compared with",
                          "different from", "difference between"),
        patterns=(_p(rf"(?:is|are|was|were)\s+{adjective.replace('_than', '')}\s+than"),),
        inverse=inverse,
    )


_SPECS = (
    RelationSpec(
        "is_a", RelationFamily.DEFINITION, 0.50,
        aliases=("instance_of", "type_of", "kind_of"),
        question_markers=("what is", "what are", "type of", "kind of"),
        patterns=(
            _p(r"(?:is|are)\s+(?:a|an)\s+(?:type|kind|form|species)\s+of", confidence=0.55),
            _p(r"(?:is|are|was|were)\s+(?:a|an)"),
        ),
    ),
    RelationSpec(
        "defined_as", RelationFamily.DEFINITION, 0.45,
        aliases=("described_as", "means"),
        question_markers=("defined as", "definition of", "meaning of"),
        patterns=(_p(r"(?:is|are)\s+defined\s+as"),),
    ),
    RelationSpec(
        "has_property", RelationFamily.PROPERTY, 0.36,
        aliases=("property", "has_trait"),
        question_markers=("property of", "characteristic of", "what is it like"),
        patterns=(
            _p(
                r"(?:is|are)\s+(?="
                r"(?:small|large|big|tall|short|old|new|fast|slow|hot|cold|"
                r"light|heavy|strong|weak|solid|liquid|gaseous|transparent|"
                r"opaque|flexible|rigid|elastic|brittle|soluble|insoluble|"
                r"flammable|toxic|visible|invisible)\b)"
            ),
        ),
    ),
    RelationSpec(
        "part_of", RelationFamily.PART_WHOLE, 0.50,
        aliases=("member_of",),
        question_markers=("part of", "component of", "member of"),
        patterns=(_p(r"(?:is|are)\s+(?:a\s+)?part\s+of"),),
        inverse="contains",
    ),
    RelationSpec(
        "contains", RelationFamily.PART_WHOLE, 0.45,
        aliases=("includes", "comprises", "has_part"),
        question_markers=("contains", "includes", "comprises", "made up of"),
        patterns=(_p(r"(?:contains|contain|includes|include|comprises|comprise)"),),
        inverse="part_of",
    ),
    RelationSpec(
        "made_of", RelationFamily.MATERIAL, 0.55,
        aliases=("built_from",),
        question_markers=("made of", "made from", "material"),
        patterns=(
            _p(r"(?:is|are|was|were)\s+made\s+(?:of|from)"),
            _p(r"(?:is|are|was|were)\s+built\s+from"),
        ),
    ),
    RelationSpec(
        "consists_of", RelationFamily.MATERIAL, 0.50,
        aliases=("composed_of",),
        question_markers=("consists of", "composed of"),
        patterns=(
            _p(r"(?:consists|consist)\s+of"),
            _p(r"(?:is|are)\s+composed\s+of"),
        ),
    ),
    RelationSpec(
        "located_in", RelationFamily.LOCATION, 0.48,
        aliases=("found_in", "present_in", "situated_in"),
        question_markers=("located in", "found in", "where is", "where are"),
        patterns=(
            _p(r"(?:is|are|was|were)\s+(?:located|situated|found|present)\s+in"),
        ),
    ),
    RelationSpec(
        "used_for", RelationFamily.FUNCTION, 0.50,
        aliases=("purpose",),
        question_markers=("used for", "purpose of", "function of"),
        patterns=(_p(r"(?:is|are|was|were)\s+used\s+for"),),
    ),
    RelationSpec(
        "capable_of", RelationFamily.FUNCTION, 0.42,
        aliases=("used_to", "can"),
        question_markers=("used to", "capable of", "able to"),
        patterns=(
            _p(r"(?:is|are|was|were)\s+(?:used|designed|able)\s+to"),
            _p(r"can"),
        ),
    ),
    RelationSpec(
        "responsible_for", RelationFamily.FUNCTION, 0.50,
        aliases=("controls", "regulates"),
        question_markers=("responsible for", "role of", "controls", "regulates"),
        patterns=(
            _p(r"(?:is|are)\s+responsible\s+for"),
            _p(r"(?:controls|control|regulates|regulate)"),
        ),
    ),
    RelationSpec(
        "produces", RelationFamily.FUNCTION, 0.45,
        aliases=("creates", "generates", "releases"),
        question_markers=("produces", "creates", "generates", "releases"),
        patterns=(_p(r"(?:produces|produce|creates|create|generates|generate|releases|release)"),),
    ),
    RelationSpec(
        "causes", RelationFamily.CAUSAL, 0.40,
        aliases=("cause", "leads_to", "lead_to", "results_in", "result_in", "triggers", "induces"),
        question_markers=("causes", "cause of", "leads to", "results in",
                          "because of", "effect", "affect"),
        patterns=(_p(r"(?:causes|cause|leads\s+to|lead\s+to|results\s+in|result\s+in|triggers|trigger|induces|induce)"),),
    ),
    RelationSpec(
        "prevents", RelationFamily.CAUSAL, 0.40,
        aliases=("prevent", "blocks", "protects_against"),
        question_markers=("prevents", "prevent", "blocks", "protects against"),
        patterns=(_p(r"(?:prevents|prevent|blocks|block|protects\s+against)"),),
    ),
    RelationSpec(
        "reduces", RelationFamily.CAUSAL, 0.40,
        aliases=("reduce", "lowers", "decreases"),
        question_markers=("reduces", "reduce", "lowers", "decreases"),
        patterns=(_p(r"(?:reduces|reduce|lowers|lower|decreases|decrease)"),),
    ),
    RelationSpec(
        "increases", RelationFamily.CAUSAL, 0.40,
        aliases=("increase", "raises", "boosts"),
        question_markers=("increases", "increase", "raises", "more likely"),
        patterns=(_p(r"(?:increases|increase|raises|raise|boosts|boost)"),),
    ),
    RelationSpec(
        "improves", RelationFamily.CAUSAL, 0.40,
        aliases=("improve", "enhances", "strengthens", "promotes"),
        question_markers=("improves", "improve", "enhances", "better"),
        patterns=(_p(r"(?:improves|improve|enhances|enhance|strengthens|strengthen|promotes|promote)"),),
    ),
    RelationSpec(
        "related_to", RelationFamily.ASSOCIATION, 0.34,
        aliases=("associated_with", "linked_to", "linked_with", "correlated_with", "related"),
        question_markers=("related to", "associated with", "linked to", "correlated with"),
        patterns=(_p(r"(?:is|are)\s+(?:associated|linked|correlated|related)\s+(?:with|to)"),),
    ),
    RelationSpec(
        "created_in", RelationFamily.TEMPORAL, 0.60,
        aliases=("developed_in", "invented_in", "founded_in", "discovered_in", "released_in", "established_in"),
        question_markers=("created in", "developed in", "invented in", "founded in", "which year", "what year", "oldest", "newest", "earliest", "latest"),
        patterns=(
            ExtractionPattern(
                rf"^\s*{_SUBJECT}\s+(?:was|were)\s+(?:developed|invented|founded|discovered|released|introduced|established|created|built)\s+(?:in|during)\s+(?P<object>(?:1[0-9]{{3}}|20[0-9]{{2}}))(?=\s*(?:[,;.]|\bby\b|$))"
            ),
            ExtractionPattern(
                rf"^\s*{_SUBJECT}\s+dates?(?:\s+back)?\s+to\s+(?P<object>(?:1[0-9]{{3}}|20[0-9]{{2}}))(?=\s*(?:[,;.]|$))",
                confidence=0.55,
            ),
        ),
    ),
    _comparative("smaller_than", "smaller_than", "larger_than"),
    _comparative("larger_than", "larger_than", "smaller_than"),
    _comparative("taller_than", "taller_than", "shorter_than"),
    _comparative("shorter_than", "shorter_than", "taller_than"),
    _comparative("older_than", "older_than", "younger_than"),
    _comparative("younger_than", "younger_than", "older_than"),
    _comparative("faster_than", "faster_than", "slower_than"),
    _comparative("slower_than", "slower_than", "faster_than"),
    _comparative("hotter_than", "hotter_than", "colder_than"),
    _comparative("colder_than", "colder_than", "hotter_than"),
    _comparative("lighter_than", "lighter_than", "heavier_than"),
    _comparative("heavier_than", "heavier_than", "lighter_than"),
    _comparative("stronger_than", "stronger_than", "weaker_than"),
    _comparative("weaker_than", "weaker_than", "stronger_than"),
)


RELATIONS: Mapping[str, RelationSpec] = MappingProxyType(
    {spec.name: spec for spec in _SPECS}
)


def _normal_form(value: str) -> str:
    return "_".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


_ALIASES: dict[str, str] = {}
for _spec in _SPECS:
    for _alias in (_spec.name, *_spec.aliases):
        _key = _normal_form(_alias)
        previous = _ALIASES.setdefault(_key, _spec.name)
        if previous != _spec.name:
            raise ValueError(f"ambiguous relation alias {_alias!r}")


def canonicalize_mechanism(mechanism: str) -> Optional[str]:
    """Return the canonical relation or ``None`` for an unknown mechanism."""

    key = _normal_form(mechanism)
    prefixes = (
        "indirectly_", "directly_", "significantly_", "strongly_",
        "weakly_", "highly_", "positively_", "negatively_", "closely_",
        "mainly_", "primarily_", "partly_", "largely_",
    )
    for prefix in prefixes:
        if key.startswith(prefix):
            key = key[len(prefix):]
            break
    return _ALIASES.get(key)


def normalize_mechanism(mechanism: str, *, preserve_unknown: bool = True) -> Optional[str]:
    """Normalize known relations without fabricating meaning for unknown ones."""

    canonical = canonicalize_mechanism(mechanism)
    if canonical is not None:
        return canonical
    return (_normal_form(mechanism) or None) if preserve_unknown else None


def is_canonical(mechanism: str) -> bool:
    return mechanism in RELATIONS


def relations_for_family(family: RelationFamily | str) -> frozenset[str]:
    family = RelationFamily(family)
    return frozenset(s.name for s in _SPECS if s.family == family)


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", text.casefold()))


def _contains_phrase(haystack: tuple[str, ...], needle: str) -> bool:
    wanted = _tokens(needle)
    if not wanted or len(wanted) > len(haystack):
        return False
    return any(haystack[i:i + len(wanted)] == wanted
               for i in range(len(haystack) - len(wanted) + 1))


def question_primitives(question: str) -> frozenset[str]:
    """Canonical relations explicitly requested by a question.

    Matching is token/phrase based; substring accidents such as ``for`` in
    ``formula`` cannot activate a relation.
    """

    tokens = _tokens(question)
    found = {
        spec.name
        for spec in _SPECS
        if any(_contains_phrase(tokens, marker)
               for marker in spec.question_markers)
    }
    return frozenset(found)


@dataclass(frozen=True)
class SchemaCoverage:
    total: int
    canonical: int
    unknown: Mapping[str, int]
    entropy_bits: float
    effective_relations: float

    @property
    def ratio(self) -> float:
        return self.canonical / self.total if self.total else 1.0


def schema_coverage(mechanisms: Iterable[str]) -> SchemaCoverage:
    """Measure canonical coverage and relation diversity.

    Entropy/effective relation count are appropriate for categorical relation
    usage.  Intrinsic dimension is not: one-hot relation labels do not form a
    meaningful continuous geometry.
    """

    known: dict[str, int] = {}
    unknown: dict[str, int] = {}
    total = 0
    for raw in mechanisms:
        total += 1
        relation = canonicalize_mechanism(raw)
        if relation is None:
            key = _normal_form(raw) or "<empty>"
            unknown[key] = unknown.get(key, 0) + 1
        else:
            known[relation] = known.get(relation, 0) + 1
    canonical = sum(known.values())
    entropy = 0.0
    if canonical:
        for count in known.values():
            p = count / canonical
            entropy -= p * math.log2(p)
    return SchemaCoverage(
        total=total,
        canonical=canonical,
        unknown=MappingProxyType(dict(sorted(unknown.items()))),
        entropy_bits=entropy,
        effective_relations=2.0 ** entropy,
    )
