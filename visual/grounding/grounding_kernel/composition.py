"""Typed symbolic composition on top of independently grounded atoms.

This module deliberately does not infer meanings from dictionary text.  It
only (a) evaluates definitions whose leaves can already be resolved against
sensorimotor evidence and (b) computes the least grounding closure of a
definition graph.  Starting the latter at zero is important: a closed cycle
such as ``a := b; b := a`` never manufactures its own grounding.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterable, Mapping
from dataclasses import dataclass
from enum import Enum, IntEnum
from types import MappingProxyType
from typing import Any, TypeAlias


def _stable_key(value: Hashable) -> tuple[str, str]:
    return (type(value).__qualname__, repr(value))


class Sort(str, Enum):
    """Small type universe for executable definitions."""

    BOOLEAN = "boolean"
    ENTITY = "entity"
    ACTION = "action"
    OBSERVATION = "observation"
    ANY = "any"


class TruthValue(IntEnum):
    """Strong-Kleene truth values used for honest abstention."""

    UNKNOWN = -1
    FALSE = 0
    TRUE = 1

    @property
    def resolved(self) -> bool:
        return self is not TruthValue.UNKNOWN

    def as_python(self) -> bool | None:
        if self is TruthValue.UNKNOWN:
            return None
        return self is TruthValue.TRUE


@dataclass(frozen=True, slots=True)
class Atom:
    """A grounded predicate leaf or a typed literal term."""

    symbol: Hashable
    sort: Sort = Sort.BOOLEAN

    def __post_init__(self) -> None:
        try:
            hash(self.symbol)
        except TypeError as exc:
            raise TypeError("atom symbols must be hashable") from exc
        object.__setattr__(self, "sort", Sort(self.sort))


def entity(symbol: Hashable) -> Atom:
    """Return an entity-sorted atom for use as a relation argument."""

    return Atom(symbol, Sort.ENTITY)


@dataclass(frozen=True, slots=True, init=False)
class And:
    """Boolean conjunction with an ergonomic variadic constructor."""

    terms: tuple["Expression", ...]
    sort: Sort = Sort.BOOLEAN

    def __init__(self, *terms: "Expression") -> None:
        if len(terms) == 1 and isinstance(terms[0], (tuple, list)):
            terms = tuple(terms[0])
        values = tuple(terms)
        if not values:
            raise ValueError("And requires at least one term")
        _require_boolean(values, "And")
        object.__setattr__(self, "terms", values)
        object.__setattr__(self, "sort", Sort.BOOLEAN)


@dataclass(frozen=True, slots=True, init=False)
class Or:
    """Boolean disjunction with an ergonomic variadic constructor."""

    terms: tuple["Expression", ...]
    sort: Sort = Sort.BOOLEAN

    def __init__(self, *terms: "Expression") -> None:
        if len(terms) == 1 and isinstance(terms[0], (tuple, list)):
            terms = tuple(terms[0])
        values = tuple(terms)
        if not values:
            raise ValueError("Or requires at least one term")
        _require_boolean(values, "Or")
        object.__setattr__(self, "terms", values)
        object.__setattr__(self, "sort", Sort.BOOLEAN)


@dataclass(frozen=True, slots=True)
class Not:
    """Boolean negation.  Grounding a negation still requires its operand."""

    term: "Expression"
    sort: Sort = Sort.BOOLEAN

    def __post_init__(self) -> None:
        _require_boolean((self.term,), "Not")


@dataclass(frozen=True, slots=True)
class Relation:
    """A simple typed binary relation application.

    Relation operands are terms, not truth-valued predicates.  Use
    :func:`entity` for entity literals, for example
    ``Relation(code, entity("x"), entity("y"))``.
    """

    symbol: Hashable
    left: Atom
    right: Atom
    domain: Sort = Sort.ENTITY
    codomain: Sort = Sort.ENTITY
    sort: Sort = Sort.BOOLEAN

    def __post_init__(self) -> None:
        try:
            hash(self.symbol)
        except TypeError as exc:
            raise TypeError("relation symbols must be hashable") from exc
        domain = Sort(self.domain)
        codomain = Sort(self.codomain)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "codomain", codomain)
        if not isinstance(self.left, Atom) or not isinstance(self.right, Atom):
            raise TypeError("relation operands must be Atom terms")
        if not _sort_matches(self.left.sort, domain):
            raise TypeError(
                f"left operand has sort {self.left.sort.value}, expected {domain.value}"
            )
        if not _sort_matches(self.right.sort, codomain):
            raise TypeError(
                f"right operand has sort {self.right.sort.value}, expected {codomain.value}"
            )


Expression: TypeAlias = Atom | And | Or | Not | Relation
DefinitionAST: TypeAlias = Expression


def _sort_matches(actual: Sort, expected: Sort) -> bool:
    return actual is Sort.ANY or expected is Sort.ANY or actual is expected


def _require_boolean(terms: tuple[Expression, ...], owner: str) -> None:
    for term in terms:
        if not isinstance(term, (Atom, And, Or, Not, Relation)):
            raise TypeError(f"{owner} terms must be definition expressions")
        if not _sort_matches(term.sort, Sort.BOOLEAN):
            raise TypeError(f"{owner} requires boolean terms, got {term.sort.value}")


@dataclass(frozen=True, slots=True)
class Derivation:
    """Auditable truth derivation returned by :func:`evaluate`."""

    rule: str
    conclusion: str
    value: TruthValue
    premises: tuple["Derivation", ...] = ()
    evidence: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rule": self.rule,
            "conclusion": self.conclusion,
            "value": self.value.name.lower(),
            "evidence": self.evidence,
            "premises": [premise.to_dict() for premise in self.premises],
        }


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    value: TruthValue
    derivation: Derivation

    @property
    def resolved(self) -> bool:
        return self.value.resolved

    def as_python(self) -> bool | None:
        return self.value.as_python()


ResolverValue: TypeAlias = TruthValue | bool | None | tuple[TruthValue | bool | None, str]
AtomResolver: TypeAlias = Callable[[Hashable], ResolverValue]
RelationResolver: TypeAlias = Callable[[Hashable, Hashable, Hashable], ResolverValue]


def _resolver_value(value: ResolverValue) -> tuple[TruthValue, str | None]:
    evidence: str | None = None
    if isinstance(value, tuple):
        if len(value) != 2:
            raise TypeError("resolver tuples must be (truth, evidence)")
        value, evidence_value = value
        evidence = str(evidence_value)
    if value is None:
        return TruthValue.UNKNOWN, evidence
    if isinstance(value, TruthValue):
        return value, evidence
    if isinstance(value, (bool,)):
        return TruthValue.TRUE if value else TruthValue.FALSE, evidence
    raise TypeError("resolvers must return bool, TruthValue, None, or (value, evidence)")


def _render(expression: Expression) -> str:
    if isinstance(expression, Atom):
        return repr(expression.symbol)
    if isinstance(expression, Not):
        return f"not({_render(expression.term)})"
    if isinstance(expression, (And, Or)):
        operator = " and " if isinstance(expression, And) else " or "
        return "(" + operator.join(_render(term) for term in expression.terms) + ")"
    return (
        f"{expression.symbol!r}({_render(expression.left)}, "
        f"{_render(expression.right)})"
    )


def evaluate(
    expression: Expression,
    atom_resolver: AtomResolver,
    relation_resolver: RelationResolver | None = None,
) -> EvaluationResult:
    """Evaluate a definition using three-valued logic and return its proof.

    ``None`` from a resolver means "not established", never false.  This is
    the composition layer's abstention path and prevents missing sensorimotor
    evidence from silently turning into a negative fact.
    """

    if isinstance(expression, Atom):
        if expression.sort is not Sort.BOOLEAN:
            raise TypeError("only boolean atoms can be evaluated as definitions")
        value, evidence = _resolver_value(atom_resolver(expression.symbol))
        proof = Derivation("grounded-atom", _render(expression), value, evidence=evidence)
        return EvaluationResult(value, proof)

    if isinstance(expression, Relation):
        if relation_resolver is None:
            value, evidence = TruthValue.UNKNOWN, "no relation resolver"
        else:
            value, evidence = _resolver_value(
                relation_resolver(
                    expression.symbol,
                    expression.left.symbol,
                    expression.right.symbol,
                )
            )
        proof = Derivation("grounded-relation", _render(expression), value, evidence=evidence)
        return EvaluationResult(value, proof)

    if isinstance(expression, Not):
        child = evaluate(expression.term, atom_resolver, relation_resolver)
        if child.value is TruthValue.UNKNOWN:
            value = TruthValue.UNKNOWN
        elif child.value is TruthValue.TRUE:
            value = TruthValue.FALSE
        else:
            value = TruthValue.TRUE
        proof = Derivation("not-elimination", _render(expression), value, (child.derivation,))
        return EvaluationResult(value, proof)

    if isinstance(expression, (And, Or)):
        children = tuple(
            evaluate(term, atom_resolver, relation_resolver) for term in expression.terms
        )
        values = tuple(child.value for child in children)
        if isinstance(expression, And):
            if TruthValue.FALSE in values:
                value = TruthValue.FALSE
            elif all(value is TruthValue.TRUE for value in values):
                value = TruthValue.TRUE
            else:
                value = TruthValue.UNKNOWN
            rule = "and-introduction"
        else:
            if TruthValue.TRUE in values:
                value = TruthValue.TRUE
            elif all(value is TruthValue.FALSE for value in values):
                value = TruthValue.FALSE
            else:
                value = TruthValue.UNKNOWN
            rule = "or-introduction"
        proof = Derivation(
            rule,
            _render(expression),
            value,
            tuple(child.derivation for child in children),
        )
        return EvaluationResult(value, proof)

    raise TypeError(f"unsupported definition expression: {type(expression).__name__}")


@dataclass(frozen=True, slots=True)
class GroundingDerivation:
    """A confidence-carrying derivation in the definition closure."""

    conclusion: Hashable
    confidence: float
    rule: str
    iteration: int
    premises: tuple["GroundingDerivation", ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "conclusion": repr(self.conclusion),
            "confidence": self.confidence,
            "rule": self.rule,
            "iteration": self.iteration,
            "premises": [premise.to_dict() for premise in self.premises],
        }


@dataclass(frozen=True, slots=True)
class ClosureResult:
    """Least-fixed-point grounding result, including unresolved cycles."""

    confidences: Mapping[Hashable, float]
    derivations: Mapping[Hashable, GroundingDerivation]
    iterations: int
    unresolved_cycles: tuple[tuple[Hashable, ...], ...]
    tolerance: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "confidences", MappingProxyType(dict(self.confidences)))
        object.__setattr__(self, "derivations", MappingProxyType(dict(self.derivations)))

    @property
    def grounded(self) -> frozenset[Hashable]:
        return frozenset(
            symbol for symbol, confidence in self.confidences.items() if confidence > self.tolerance
        )

    @property
    def unresolved(self) -> frozenset[Hashable]:
        return frozenset(
            symbol
            for symbol, confidence in self.confidences.items()
            if confidence <= self.tolerance
        )

    def confidence(self, symbol: Hashable) -> float:
        return float(self.confidences.get(symbol, 0.0))

    def proof(self, symbol: Hashable) -> GroundingDerivation | None:
        return self.derivations.get(symbol)

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.confidences, key=_stable_key)
        return {
            "iterations": self.iterations,
            "tolerance": self.tolerance,
            "confidences": {repr(key): self.confidences[key] for key in ordered},
            "grounded": [repr(key) for key in sorted(self.grounded, key=_stable_key)],
            "unresolved": [repr(key) for key in sorted(self.unresolved, key=_stable_key)],
            "unresolved_cycles": [
                [repr(key) for key in cycle] for cycle in self.unresolved_cycles
            ],
            "derivations": {
                repr(key): self.derivations[key].to_dict()
                for key in sorted(self.derivations, key=_stable_key)
            },
        }


def _anchor_confidences(
    anchors: Mapping[Hashable, float | bool] | Iterable[Hashable],
) -> dict[Hashable, float]:
    if isinstance(anchors, Mapping):
        pairs = anchors.items()
    else:
        pairs = ((symbol, 1.0) for symbol in anchors)
    result: dict[Hashable, float] = {}
    for symbol, raw_confidence in pairs:
        try:
            hash(symbol)
        except TypeError as exc:
            raise TypeError("anchor symbols must be hashable") from exc
        confidence = float(raw_confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("anchor confidences must lie in [0, 1]")
        result[symbol] = max(result.get(symbol, 0.0), confidence)
    return result


def dependencies(expression: Expression) -> frozenset[Hashable]:
    """Return every grounded symbol needed to interpret ``expression``."""

    if isinstance(expression, Atom):
        return frozenset((expression.symbol,))
    if isinstance(expression, Relation):
        return frozenset(
            (expression.symbol, expression.left.symbol, expression.right.symbol)
        )
    if isinstance(expression, Not):
        return dependencies(expression.term)
    if isinstance(expression, (And, Or)):
        result: set[Hashable] = set()
        for term in expression.terms:
            result.update(dependencies(term))
        return frozenset(result)
    raise TypeError(f"unsupported definition expression: {type(expression).__name__}")


def _support(
    expression: Expression,
    confidence: Mapping[Hashable, float],
    proofs: Mapping[Hashable, GroundingDerivation],
    iteration: int,
) -> tuple[float, GroundingDerivation | None]:
    if isinstance(expression, Atom):
        value = confidence.get(expression.symbol, 0.0)
        proof = proofs.get(expression.symbol)
        if proof is None and value > 0.0:
            proof = GroundingDerivation(expression.symbol, value, "available-atom", iteration)
        return value, proof

    if isinstance(expression, Relation):
        symbols = (expression.symbol, expression.left.symbol, expression.right.symbol)
        values = tuple(confidence.get(symbol, 0.0) for symbol in symbols)
        value = min(values)
        if value <= 0.0:
            return 0.0, None
        premises = tuple(
            proofs.get(symbol)
            or GroundingDerivation(symbol, confidence[symbol], "available-term", iteration)
            for symbol in symbols
        )
        return value, GroundingDerivation(
            _render(expression), value, "typed-relation", iteration, premises
        )

    if isinstance(expression, Not):
        value, proof = _support(expression.term, confidence, proofs, iteration)
        if value <= 0.0 or proof is None:
            return 0.0, None
        return value, GroundingDerivation(
            _render(expression), value, "grounded-negation", iteration, (proof,)
        )

    if isinstance(expression, (And, Or)):
        supported = tuple(
            _support(term, confidence, proofs, iteration) for term in expression.terms
        )
        if isinstance(expression, And):
            value = min(item[0] for item in supported)
            if value <= 0.0 or any(item[1] is None for item in supported):
                return 0.0, None
            premises = tuple(item[1] for item in supported if item[1] is not None)
            rule = "grounded-conjunction"
        else:
            best_index = max(
                range(len(supported)),
                key=lambda index: (supported[index][0], -index),
            )
            value, best_proof = supported[best_index]
            if value <= 0.0 or best_proof is None:
                return 0.0, None
            premises = (best_proof,)
            rule = "grounded-disjunction"
        return value, GroundingDerivation(
            _render(expression), value, rule, iteration, premises
        )

    raise TypeError(f"unsupported definition expression: {type(expression).__name__}")


def _unresolved_cycles(
    definitions: Mapping[Hashable, Expression],
    confidence: Mapping[Hashable, float],
    tolerance: float,
) -> tuple[tuple[Hashable, ...], ...]:
    unresolved = {
        symbol for symbol in definitions if confidence.get(symbol, 0.0) <= tolerance
    }
    graph = {
        symbol: tuple(
            sorted(dependencies(definitions[symbol]) & unresolved, key=_stable_key)
        )
        for symbol in unresolved
    }
    index = 0
    stack: list[Hashable] = []
    on_stack: set[Hashable] = set()
    indices: dict[Hashable, int] = {}
    lowlinks: dict[Hashable, int] = {}
    components: list[tuple[Hashable, ...]] = []

    def visit(symbol: Hashable) -> None:
        nonlocal index
        indices[symbol] = index
        lowlinks[symbol] = index
        index += 1
        stack.append(symbol)
        on_stack.add(symbol)
        for dependency in graph[symbol]:
            if dependency not in indices:
                visit(dependency)
                lowlinks[symbol] = min(lowlinks[symbol], lowlinks[dependency])
            elif dependency in on_stack:
                lowlinks[symbol] = min(lowlinks[symbol], indices[dependency])
        if lowlinks[symbol] != indices[symbol]:
            return
        component: list[Hashable] = []
        while True:
            member = stack.pop()
            on_stack.remove(member)
            component.append(member)
            if member == symbol:
                break
        ordered = tuple(sorted(component, key=_stable_key))
        if len(ordered) > 1 or (len(ordered) == 1 and ordered[0] in graph[ordered[0]]):
            components.append(ordered)

    for symbol in sorted(unresolved, key=_stable_key):
        if symbol not in indices:
            visit(symbol)
    return tuple(sorted(components, key=lambda cycle: tuple(_stable_key(v) for v in cycle)))


def least_fixed_point(
    definitions: Mapping[Hashable, Expression],
    anchors: Mapping[Hashable, float | bool] | Iterable[Hashable],
    *,
    tolerance: float = 1e-12,
    max_iterations: int | None = None,
) -> ClosureResult:
    """Compute the cycle-safe least grounding closure of definitions.

    The lattice starts with direct sensorimotor anchors and zero everywhere
    else.  Logical operators combine *grounding confidence*, not truth: a
    negated predicate, for example, is interpretable only when the positive
    predicate itself is grounded.  Values monotonically increase, so a
    dictionary-only cycle remains exactly zero unless an external anchor or a
    grounded alternative enters it.
    """

    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    definition_map = dict(definitions)
    for symbol, expression in definition_map.items():
        try:
            hash(symbol)
        except TypeError as exc:
            raise TypeError("definition symbols must be hashable") from exc
        if not isinstance(expression, (Atom, And, Or, Not, Relation)):
            raise TypeError("definition values must be typed expressions")
        if expression.sort is not Sort.BOOLEAN:
            raise TypeError("definitions must have boolean output sort")

    direct = _anchor_confidences(anchors)
    all_symbols = set(definition_map) | set(direct)
    confidence = {symbol: direct.get(symbol, 0.0) for symbol in all_symbols}
    proofs: dict[Hashable, GroundingDerivation] = {
        symbol: GroundingDerivation(symbol, value, "sensorimotor-anchor", 0)
        for symbol, value in direct.items()
        if value > tolerance
    }
    limit = max_iterations if max_iterations is not None else max(1, len(definition_map) + 1)
    if limit < 1:
        raise ValueError("max_iterations must be positive")

    completed_iterations = 0
    for iteration in range(1, limit + 1):
        completed_iterations = iteration
        updates: dict[Hashable, tuple[float, GroundingDerivation]] = {}
        for symbol in sorted(definition_map, key=_stable_key):
            value, support_proof = _support(
                definition_map[symbol], confidence, proofs, iteration
            )
            if value > confidence.get(symbol, 0.0) + tolerance and support_proof is not None:
                proof = GroundingDerivation(
                    symbol, value, "definition", iteration, (support_proof,)
                )
                updates[symbol] = (value, proof)
        if not updates:
            break
        for symbol, (value, proof) in updates.items():
            confidence[symbol] = value
            proofs[symbol] = proof
    else:
        # A finite min/max definition graph normally stabilises after at most
        # |definitions| waves.  Raising avoids returning a non-fixed result if
        # a caller deliberately supplies a smaller limit.
        raise RuntimeError("grounding closure did not converge within max_iterations")

    cycles = _unresolved_cycles(definition_map, confidence, tolerance)
    ordered_confidence = {
        symbol: confidence[symbol] for symbol in sorted(confidence, key=_stable_key)
    }
    ordered_proofs = {symbol: proofs[symbol] for symbol in sorted(proofs, key=_stable_key)}
    return ClosureResult(
        ordered_confidence,
        ordered_proofs,
        completed_iterations,
        cycles,
        tolerance,
    )


grounding_closure = least_fixed_point

