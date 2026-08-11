"""Deterministic language induction over opaque operational referents.

The learner in this module never receives a part-of-speech tag, a semantic
name, or an evaluator codebook.  A demonstration contains only an opaque token
sequence and an optional learner-visible :class:`GroundedReferent`.  Each item
in that referent has an opaque ``type_id`` identifying an operational slot and
an opaque value observed in that slot.

The implementation is intentionally conservative.  It searches for a unique
bijection between token positions and operational slots which makes every
token-to-value association consistent.  If no such order exists, or if more
than one order fits, it abstains.  It stores atomic bindings and an induced
order template rather than a table of complete utterances, permitting held-out
factorial combinations in both interpretation and description.

Dictionary definitions are grounded separately through
:func:`grounding_kernel.composition.least_fixed_point`.  Only meanings and
tokens directly anchored by paired demonstrations seed that closure; a closed
dictionary cycle therefore remains unknown.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Hashable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from itertools import permutations, product
import json
from types import MappingProxyType
from typing import Any, TypeAlias

from .composition import ClosureResult, Expression, least_fixed_point


OpaqueToken: TypeAlias = Hashable
TokenSequence: TypeAlias = tuple[OpaqueToken, ...]


def _stable_key(value: object) -> tuple[str, str]:
    return (type(value).__qualname__, repr(value))


def _meaning_key(value: "OperationalMeaning") -> tuple[tuple[str, str], tuple[str, str]]:
    return (_stable_key(value.type_id), _stable_key(value.value))


def _sequence_key(values: Sequence[Hashable]) -> tuple[tuple[str, str], ...]:
    return tuple(_stable_key(value) for value in values)


def _require_hashable(value: object, field: str) -> Hashable:
    try:
        hash(value)
    except TypeError as exc:
        raise TypeError(f"{field} must be hashable") from exc
    return value  # type: ignore[return-value]


def _canonical_text(value: object) -> str:
    """Return a stable-enough audit rendering without interpreting values."""

    if isinstance(value, OperationalMeaning):
        return (
            "meaning("
            + _canonical_text(value.type_id)
            + ","
            + _canonical_text(value.value)
            + ")"
        )
    if isinstance(value, GroundedReferent):
        return "referent(" + ",".join(_canonical_text(item) for item in value.meanings) + ")"
    if isinstance(value, Mapping):
        ordered = sorted(value, key=_stable_key)
        return "{" + ",".join(
            _canonical_text(key) + ":" + _canonical_text(value[key]) for key in ordered
        ) + "}"
    if isinstance(value, (tuple, list)):
        return "[" + ",".join(_canonical_text(item) for item in value) + "]"
    if isinstance(value, (set, frozenset)):
        return "{" + ",".join(
            _canonical_text(item) for item in sorted(value, key=_stable_key)
        ) + "}"
    if isinstance(value, bytes):
        return "bytes:" + value.hex()
    if value is None or isinstance(value, (str, int, float, bool)):
        return f"{type(value).__qualname__}:{value!r}"
    shape = getattr(value, "shape", None)
    dtype = getattr(value, "dtype", None)
    tobytes = getattr(value, "tobytes", None)
    if shape is not None and dtype is not None and callable(tobytes):
        digest = sha256(tobytes()).hexdigest()
        return f"array:{tuple(shape)!r}:{dtype!s}:{digest}"
    return f"{type(value).__qualname__}:{value!r}"


class Resolution(str, Enum):
    """Epistemic state of a language query; these are not semantic labels."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    UNKNOWN = "unknown"


UNKNOWN = Resolution.UNKNOWN


@dataclass(frozen=True, slots=True)
class OperationalMeaning:
    """One typed, learner-visible operational value.

    ``type_id`` names an opaque sensor/action/world slot, not a linguistic
    category.  Equality includes both the slot and its observed value.
    """

    type_id: Hashable
    value: Hashable

    def __post_init__(self) -> None:
        _require_hashable(self.type_id, "type_id")
        _require_hashable(self.value, "value")


@dataclass(frozen=True, slots=True)
class GroundedReferent:
    """An unordered typed collection of learner-visible operational values."""

    meanings: tuple[OperationalMeaning, ...]

    def __post_init__(self) -> None:
        meanings = tuple(self.meanings)
        if not meanings:
            raise ValueError("a grounded referent requires at least one operational meaning")
        if not all(isinstance(meaning, OperationalMeaning) for meaning in meanings):
            raise TypeError("referent items must be OperationalMeaning records")
        type_ids = [meaning.type_id for meaning in meanings]
        if len(set(type_ids)) != len(type_ids):
            raise ValueError("a referent may contain only one value per type_id")
        object.__setattr__(self, "meanings", tuple(sorted(meanings, key=_meaning_key)))

    @property
    def schema(self) -> tuple[Hashable, ...]:
        """Return the opaque operational slot IDs in deterministic order."""

        return tuple(meaning.type_id for meaning in self.meanings)

    def meaning_for(self, type_id: Hashable) -> OperationalMeaning | None:
        for meaning in self.meanings:
            if meaning.type_id == type_id:
                return meaning
        return None


@dataclass(frozen=True, slots=True)
class Demonstration:
    """An opaque utterance paired with learner-visible grounding evidence.

    ``referent=None`` explicitly represents a no-sensor negative control.  The
    optional ``evidence`` object is used only to fingerprint proofs; it is never
    inspected for semantic fields.
    """

    tokens: TokenSequence
    referent: GroundedReferent | None
    evidence: object | None = None

    def __post_init__(self) -> None:
        tokens = tuple(self.tokens)
        if not tokens:
            raise ValueError("demonstration token sequences must be non-empty")
        for token in tokens:
            _require_hashable(token, "opaque token")
        if self.referent is not None and not isinstance(self.referent, GroundedReferent):
            raise TypeError("referent must be GroundedReferent or None")
        object.__setattr__(self, "tokens", tokens)


@dataclass(frozen=True, slots=True)
class LanguageProof:
    """Small deterministic derivation tree for one language decision."""

    rule: str
    conclusion: str
    evidence: tuple[str, ...] = ()
    premises: tuple["LanguageProof", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "conclusion": self.conclusion,
            "evidence": list(self.evidence),
            "premises": [premise.to_dict() for premise in self.premises],
        }


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    """One immutable induction event in the evidence ledger."""

    rule: str
    conclusion: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "conclusion": self.conclusion,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class LanguageLedger:
    """Deterministically serialized evidence ledger."""

    entries: tuple[LedgerEntry, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [entry.to_dict() for entry in self.entries], "digest": self.digest}

    @property
    def digest(self) -> str:
        payload = json.dumps(
            [entry.to_dict() for entry in self.entries],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class OrderTemplate:
    """Learned token-position to opaque-operational-slot bijection."""

    schema: tuple[Hashable, ...]
    position_types: tuple[Hashable, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LexicalBinding:
    """Directly demonstrated token-to-operational-meaning binding."""

    token: Hashable
    meaning: OperationalMeaning
    positions: tuple[int, ...]
    evidence: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MeaningResult:
    status: Resolution
    meaning: OperationalMeaning | None
    candidates: tuple[OperationalMeaning, ...]
    proof: LanguageProof

    @property
    def resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


@dataclass(frozen=True, slots=True)
class InterpretationResult:
    status: Resolution
    referent: GroundedReferent | None
    candidates: tuple[GroundedReferent, ...]
    proof: LanguageProof

    @property
    def resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


@dataclass(frozen=True, slots=True)
class DescriptionResult:
    status: Resolution
    utterance: TokenSequence | None
    candidates: tuple[TokenSequence, ...]
    proof: LanguageProof

    @property
    def resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


@dataclass(frozen=True, slots=True)
class DefinitionResult:
    """Groundedness decision for a dictionary-supplied typed expression."""

    symbol: Hashable
    status: Resolution
    expression: Expression | None
    confidence: float
    closure: ClosureResult
    proof: LanguageProof

    @property
    def resolved(self) -> bool:
        return self.status is Resolution.RESOLVED


@dataclass(frozen=True, slots=True)
class _PreparedDemo:
    evidence_id: str
    demo: Demonstration


_SCOPE_CONCLUSION = (
    "type_id values are opaque operational slots, not part-of-speech names or semantic labels"
)


class GroundedLanguageLearner:
    """Induce a finite compositional code from grounded demonstrations.

    The supported grammar is deliberately exact and inspectable: one token per
    operational meaning and a fixed order for each learned referent schema.
    Multiple schemas and directly evidenced synonyms are supported.  There is
    no smoothing, token spelling feature, pretrained vocabulary, or full-
    utterance retrieval path.
    """

    def __init__(self, definitions: Mapping[Hashable, Expression] | None = None) -> None:
        self._definitions: dict[Hashable, Expression] = {}
        self._templates: dict[tuple[Hashable, ...], OrderTemplate] = {}
        self._token_meanings: dict[Hashable, set[OperationalMeaning]] = {}
        self._binding_evidence: dict[tuple[Hashable, OperationalMeaning], set[str]] = {}
        self._binding_positions: dict[tuple[Hashable, OperationalMeaning], set[int]] = {}
        self._positional: dict[
            tuple[tuple[Hashable, ...], int, Hashable], OperationalMeaning
        ] = {}
        self._positional_evidence: dict[
            tuple[tuple[Hashable, ...], int, Hashable], set[str]
        ] = {}
        self._positional_reverse: dict[
            tuple[tuple[Hashable, ...], int, OperationalMeaning], set[Hashable]
        ] = {}
        self._ledger_entries: list[LedgerEntry] = [LedgerEntry("scope", _SCOPE_CONCLUSION)]
        if definitions:
            self.add_definitions(definitions)

    @staticmethod
    def _prepare(demonstrations: Iterable[Demonstration]) -> tuple[_PreparedDemo, ...]:
        raw = tuple(demonstrations)
        if not all(isinstance(demo, Demonstration) for demo in raw):
            raise TypeError("fit expects Demonstration records")
        keyed: list[tuple[str, Demonstration]] = []
        for demo in raw:
            payload = _canonical_text((demo.tokens, demo.referent, demo.evidence)).encode("utf-8")
            keyed.append((sha256(payload).hexdigest(), demo))
        keyed.sort(key=lambda item: (item[0], _sequence_key(item[1].tokens)))
        occurrences: defaultdict[str, int] = defaultdict(int)
        prepared: list[_PreparedDemo] = []
        for digest, demo in keyed:
            occurrence = occurrences[digest]
            occurrences[digest] += 1
            prepared.append(_PreparedDemo(f"demo:{digest}:{occurrence}", demo))
        return tuple(prepared)

    def _reset_induction(self) -> None:
        self._templates.clear()
        self._token_meanings.clear()
        self._binding_evidence.clear()
        self._binding_positions.clear()
        self._positional.clear()
        self._positional_evidence.clear()
        self._positional_reverse.clear()
        self._ledger_entries = [LedgerEntry("scope", _SCOPE_CONCLUSION)]
        for symbol in sorted(self._definitions, key=_stable_key):
            self._ledger_entries.append(
                LedgerEntry("definition-registered", f"registered definition for {symbol!r}")
            )

    def fit(self, demonstrations: Iterable[Demonstration]) -> "GroundedLanguageLearner":
        """Induce uniquely identifiable order templates and atomic bindings."""

        prepared = self._prepare(demonstrations)
        self._reset_induction()
        groups: defaultdict[tuple[Hashable, ...], list[_PreparedDemo]] = defaultdict(list)

        for item in prepared:
            demo = item.demo
            if demo.referent is None:
                self._ledger_entries.append(
                    LedgerEntry(
                        "ignored-no-sensor",
                        f"no operational referent for {_canonical_text(demo.tokens)}",
                        (item.evidence_id,),
                    )
                )
                continue
            if len(demo.tokens) != len(demo.referent.meanings):
                self._ledger_entries.append(
                    LedgerEntry(
                        "rejected-arity",
                        "simple-order induction requires one token per operational slot",
                        (item.evidence_id,),
                    )
                )
                continue
            groups[demo.referent.schema].append(item)

        for schema in sorted(groups, key=lambda value: tuple(_stable_key(item) for item in value)):
            self._induce_group(schema, tuple(groups[schema]))

        self._record_bindings()
        return self

    def _induce_group(
        self,
        schema: tuple[Hashable, ...],
        examples: tuple[_PreparedDemo, ...],
    ) -> None:
        valid_orders: list[tuple[Hashable, ...]] = []
        for position_types in permutations(schema):
            observed: defaultdict[tuple[int, Hashable], set[OperationalMeaning]] = defaultdict(set)
            for item in examples:
                referent = item.demo.referent
                assert referent is not None
                for position, (token, type_id) in enumerate(
                    zip(item.demo.tokens, position_types, strict=True)
                ):
                    meaning = referent.meaning_for(type_id)
                    assert meaning is not None
                    observed[(position, token)].add(meaning)
            if all(len(meanings) == 1 for meanings in observed.values()):
                valid_orders.append(position_types)

        evidence_ids = tuple(sorted(item.evidence_id for item in examples))
        schema_text = _canonical_text(schema)
        if not valid_orders:
            self._ledger_entries.append(
                LedgerEntry(
                    "rejected-inconsistent-pairs",
                    f"no consistent position order for opaque schema {schema_text}",
                    evidence_ids,
                )
            )
            return
        if len(valid_orders) != 1:
            rendered = tuple(_canonical_text(order) for order in valid_orders)
            self._ledger_entries.append(
                LedgerEntry(
                    "ambiguous-order",
                    f"{len(valid_orders)} orders fit opaque schema {schema_text}: {rendered!r}",
                    evidence_ids,
                )
            )
            return

        position_types = valid_orders[0]
        template = OrderTemplate(schema, position_types, evidence_ids)
        self._templates[schema] = template
        self._ledger_entries.append(
            LedgerEntry(
                "induced-order",
                f"positions map to opaque type_ids {_canonical_text(position_types)}",
                evidence_ids,
            )
        )

        for item in examples:
            referent = item.demo.referent
            assert referent is not None
            for position, (token, type_id) in enumerate(
                zip(item.demo.tokens, position_types, strict=True)
            ):
                meaning = referent.meaning_for(type_id)
                assert meaning is not None
                positional_key = (schema, position, token)
                existing = self._positional.get(positional_key)
                if existing is not None and existing != meaning:
                    raise AssertionError("validated template produced an inconsistent binding")
                self._positional[positional_key] = meaning
                self._positional_evidence.setdefault(positional_key, set()).add(item.evidence_id)
                self._positional_reverse.setdefault((schema, position, meaning), set()).add(token)
                self._token_meanings.setdefault(token, set()).add(meaning)
                self._binding_evidence.setdefault((token, meaning), set()).add(item.evidence_id)
                self._binding_positions.setdefault((token, meaning), set()).add(position)

    def _record_bindings(self) -> None:
        binding_keys = sorted(
            self._binding_evidence,
            key=lambda item: (_stable_key(item[0]), _meaning_key(item[1])),
        )
        for token, meaning in binding_keys:
            evidence = tuple(sorted(self._binding_evidence[(token, meaning)]))
            self._ledger_entries.append(
                LedgerEntry(
                    "direct-lexical-binding",
                    f"{token!r} -> {_canonical_text(meaning)}",
                    evidence,
                )
            )

        by_meaning: defaultdict[OperationalMeaning, list[Hashable]] = defaultdict(list)
        for token, meanings in self._token_meanings.items():
            if len(meanings) == 1:
                by_meaning[next(iter(meanings))].append(token)
        for meaning in sorted(by_meaning, key=_meaning_key):
            tokens = tuple(sorted(by_meaning[meaning], key=_stable_key))
            if len(tokens) < 2:
                continue
            evidence = tuple(
                sorted(
                    evidence_id
                    for token in tokens
                    for evidence_id in self._binding_evidence[(token, meaning)]
                )
            )
            self._ledger_entries.append(
                LedgerEntry(
                    "shared-grounding-synonyms",
                    f"directly co-grounded tokens {_canonical_text(tokens)} share "
                    f"{_canonical_text(meaning)}",
                    evidence,
                )
            )

    @property
    def ledger(self) -> LanguageLedger:
        return LanguageLedger(tuple(self._ledger_entries))

    @property
    def order_templates(self) -> tuple[OrderTemplate, ...]:
        return tuple(
            self._templates[schema]
            for schema in sorted(
                self._templates,
                key=lambda value: tuple(_stable_key(item) for item in value),
            )
        )

    @property
    def bindings(self) -> tuple[LexicalBinding, ...]:
        keys = sorted(
            self._binding_evidence,
            key=lambda item: (_stable_key(item[0]), _meaning_key(item[1])),
        )
        return tuple(
            LexicalBinding(
                token,
                meaning,
                tuple(sorted(self._binding_positions[(token, meaning)])),
                tuple(sorted(self._binding_evidence[(token, meaning)])),
            )
            for token, meaning in keys
        )

    @property
    def lexicon(self) -> Mapping[Hashable, tuple[OperationalMeaning, ...]]:
        values = {
            token: tuple(sorted(meanings, key=_meaning_key))
            for token, meanings in sorted(self._token_meanings.items(), key=lambda item: _stable_key(item[0]))
        }
        return MappingProxyType(values)

    def meaning(self, token: Hashable) -> MeaningResult:
        _require_hashable(token, "opaque token")
        candidates = tuple(sorted(self._token_meanings.get(token, ()), key=_meaning_key))
        premises = tuple(
            LanguageProof(
                "paired-demonstration",
                f"{token!r} -> {_canonical_text(candidate)}",
                tuple(sorted(self._binding_evidence[(token, candidate)])),
            )
            for candidate in candidates
        )
        if not candidates:
            proof = LanguageProof("no-direct-binding", f"meaning of {token!r} is UNKNOWN")
            return MeaningResult(Resolution.UNKNOWN, None, (), proof)
        if len(candidates) > 1:
            proof = LanguageProof(
                "multiple-direct-bindings",
                f"meaning of {token!r} is ambiguous",
                premises=premises,
            )
            return MeaningResult(Resolution.AMBIGUOUS, None, candidates, proof)
        proof = LanguageProof(
            "unique-direct-binding",
            f"meaning of {token!r} is {_canonical_text(candidates[0])}",
            premises=premises,
        )
        return MeaningResult(Resolution.RESOLVED, candidates[0], candidates, proof)

    def synonyms(self, token: Hashable) -> tuple[Hashable, ...]:
        """Return other tokens sharing one directly demonstrated meaning."""

        decision = self.meaning(token)
        if not decision.resolved or decision.meaning is None:
            return ()
        result = []
        for candidate, meanings in self._token_meanings.items():
            if candidate != token and meanings == {decision.meaning}:
                result.append(candidate)
        return tuple(sorted(result, key=_stable_key))

    @staticmethod
    def _tokens(tokens: Sequence[Hashable]) -> TokenSequence:
        values = tuple(tokens)
        if not values:
            raise ValueError("token sequences must be non-empty")
        for token in values:
            _require_hashable(token, "opaque token")
        return values

    def interpret(self, tokens: Sequence[Hashable]) -> InterpretationResult:
        """Compose a grounded referent from atomic bindings and learned order."""

        utterance = self._tokens(tokens)
        candidate_proofs: dict[GroundedReferent, LanguageProof] = {}
        for schema, template in sorted(
            self._templates.items(),
            key=lambda item: tuple(_stable_key(value) for value in item[0]),
        ):
            if len(template.position_types) != len(utterance):
                continue
            meanings: list[OperationalMeaning] = []
            premises: list[LanguageProof] = []
            for position, token in enumerate(utterance):
                key = (schema, position, token)
                meaning = self._positional.get(key)
                if meaning is None:
                    break
                meanings.append(meaning)
                premises.append(
                    LanguageProof(
                        "positioned-lexical-binding",
                        f"position {position}: {token!r} -> {_canonical_text(meaning)}",
                        tuple(sorted(self._positional_evidence[key])),
                    )
                )
            else:
                referent = GroundedReferent(tuple(meanings))
                candidate_proofs.setdefault(
                    referent,
                    LanguageProof(
                        "compose-induced-order",
                        f"{_canonical_text(utterance)} -> {_canonical_text(referent)}",
                        template.evidence,
                        tuple(premises),
                    ),
                )

        candidates = tuple(sorted(candidate_proofs, key=lambda item: _meaning_sequence_key(item.meanings)))
        if not candidates:
            proof = LanguageProof(
                "no-grounded-parse",
                f"{_canonical_text(utterance)} is UNKNOWN",
            )
            return InterpretationResult(Resolution.UNKNOWN, None, (), proof)
        if len(candidates) > 1:
            proof = LanguageProof(
                "multiple-grounded-parses",
                f"{_canonical_text(utterance)} has {len(candidates)} grounded parses",
                premises=tuple(candidate_proofs[candidate] for candidate in candidates),
            )
            return InterpretationResult(Resolution.AMBIGUOUS, None, candidates, proof)
        return InterpretationResult(
            Resolution.RESOLVED,
            candidates[0],
            candidates,
            candidate_proofs[candidates[0]],
        )

    def ground_action(self, tokens: Sequence[Hashable]) -> InterpretationResult:
        """Alias spelling out the instruction-to-operational-frame direction."""

        return self.interpret(tokens)

    interpret_instruction = ground_action

    def describe(self, referent: GroundedReferent) -> DescriptionResult:
        """Compose surface tokens for a new grounded referent."""

        if not isinstance(referent, GroundedReferent):
            raise TypeError("describe expects a GroundedReferent")
        template = self._templates.get(referent.schema)
        if template is None:
            proof = LanguageProof(
                "no-order-template",
                f"no grounded description for {_canonical_text(referent)}",
            )
            return DescriptionResult(Resolution.UNKNOWN, None, (), proof)

        choices: list[tuple[Hashable, ...]] = []
        for position, type_id in enumerate(template.position_types):
            meaning = referent.meaning_for(type_id)
            assert meaning is not None
            tokens = self._positional_reverse.get((referent.schema, position, meaning), set())
            if not tokens:
                proof = LanguageProof(
                    "missing-reverse-binding",
                    f"no token directly denotes {_canonical_text(meaning)} at position {position}",
                )
                return DescriptionResult(Resolution.UNKNOWN, None, (), proof)
            choices.append(tuple(sorted(tokens, key=_stable_key)))

        candidates = tuple(sorted(set(product(*choices)), key=_sequence_key))
        utterance = candidates[0]
        premises = tuple(
            LanguageProof(
                "reverse-lexical-binding",
                f"{_canonical_text(referent.meaning_for(type_id))} -> {utterance[position]!r}",
                tuple(
                    sorted(
                        self._positional_evidence[
                            (referent.schema, position, utterance[position])
                        ]
                    )
                ),
            )
            for position, type_id in enumerate(template.position_types)
        )
        proof = LanguageProof(
            "realize-induced-order",
            f"{_canonical_text(referent)} -> {_canonical_text(utterance)}",
            template.evidence,
            premises,
        )
        # Surface synonyms are alternative correct realizations, not competing
        # operational interpretations, so they do not make the result ambiguous.
        return DescriptionResult(Resolution.RESOLVED, utterance, candidates, proof)

    describe_fact = describe

    def round_trip(self, referent: GroundedReferent) -> bool:
        description = self.describe(referent)
        if not description.resolved or description.utterance is None:
            return False
        interpretation = self.interpret(description.utterance)
        return interpretation.resolved and interpretation.referent == referent

    def _anchor_confidences(self) -> dict[Hashable, float]:
        anchors: dict[Hashable, float] = {}
        for token, meanings in self._token_meanings.items():
            if len(meanings) == 1:
                anchors[token] = 1.0
        for meanings in self._token_meanings.values():
            for meaning in meanings:
                anchors[meaning] = 1.0
        return anchors

    @property
    def definitions(self) -> Mapping[Hashable, Expression]:
        return MappingProxyType(dict(self._definitions))

    @property
    def composition_anchors(self) -> Mapping[Hashable, float]:
        return MappingProxyType(self._anchor_confidences())

    def add_definitions(
        self, definitions: Mapping[Hashable, Expression]
    ) -> "GroundedLanguageLearner":
        """Register typed dictionary expressions without treating them as evidence."""

        staged = dict(self._definitions)
        changed: list[Hashable] = []
        for symbol, expression in definitions.items():
            _require_hashable(symbol, "definition symbol")
            if staged.get(symbol) != expression:
                changed.append(symbol)
            staged[symbol] = expression
        # This validates expression types/sorts while preserving unresolved
        # cycles as zero-confidence closure members.
        least_fixed_point(staged, self._anchor_confidences())
        self._definitions = staged
        for symbol in sorted(changed, key=_stable_key):
            self._ledger_entries.append(
                LedgerEntry("definition-registered", f"registered definition for {symbol!r}")
            )
        return self

    def definition_closure(self) -> ClosureResult:
        """Run the composition module's cycle-safe least grounding closure."""

        return least_fixed_point(self._definitions, self._anchor_confidences())

    grounding_closure = definition_closure

    def resolve_definition(self, symbol: Hashable) -> DefinitionResult:
        """Return a definition only when its complete dependency graph is grounded."""

        _require_hashable(symbol, "definition symbol")
        closure = self.definition_closure()
        confidence = closure.confidence(symbol)
        expression = self._definitions.get(symbol)
        if expression is None:
            proof = LanguageProof("missing-definition", f"definition of {symbol!r} is UNKNOWN")
            return DefinitionResult(
                symbol,
                Resolution.UNKNOWN,
                None,
                0.0,
                closure,
                proof,
            )
        grounding_proof = closure.proof(symbol)
        if confidence <= closure.tolerance or grounding_proof is None:
            cycles = tuple(
                cycle for cycle in closure.unresolved_cycles if symbol in cycle
            )
            evidence = (f"unresolved_cycles={cycles!r}",) if cycles else ()
            proof = LanguageProof(
                "ungrounded-definition",
                f"definition of {symbol!r} is UNKNOWN",
                evidence,
            )
            return DefinitionResult(
                symbol,
                Resolution.UNKNOWN,
                None,
                confidence,
                closure,
                proof,
            )
        proof_payload = json.dumps(
            grounding_proof.to_dict(), sort_keys=True, separators=(",", ":")
        )
        proof = LanguageProof(
            "least-fixed-point-symbolic-theft",
            f"definition of {symbol!r} is grounded",
            (proof_payload,),
        )
        return DefinitionResult(
            symbol,
            Resolution.RESOLVED,
            expression,
            confidence,
            closure,
            proof,
        )

    symbolic_theft = resolve_definition


def _meaning_sequence_key(
    meanings: Sequence[OperationalMeaning],
) -> tuple[tuple[tuple[str, str], tuple[str, str]], ...]:
    return tuple(_meaning_key(meaning) for meaning in meanings)


LanguageLearner = GroundedLanguageLearner


__all__ = [
    "UNKNOWN",
    "DefinitionResult",
    "Demonstration",
    "DescriptionResult",
    "GroundedLanguageLearner",
    "GroundedReferent",
    "InterpretationResult",
    "LanguageLearner",
    "LanguageLedger",
    "LanguageProof",
    "LedgerEntry",
    "LexicalBinding",
    "MeaningResult",
    "OpaqueToken",
    "OperationalMeaning",
    "OrderTemplate",
    "Resolution",
    "TokenSequence",
]
