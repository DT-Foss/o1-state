from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest

from grounding_kernel.active import (
    BayesianVersionSpace,
    CandidateIntervention,
    InformationGainPolicy,
    ObservableConsequence,
    OperationalHypothesis,
    ProbeBudget,
    learner_visible_consequence,
    run_acquisition,
    run_policy_baselines,
)
from grounding_kernel.benchmark import GroundingEvidence
from grounding_kernel.contracts import Action, Observation, Transition
from grounding_kernel.v1_contracts import PublicTransition


def _seeded_hypotheses(count: int = 8) -> tuple[OperationalHypothesis, ...]:
    consequences: list[OperationalHypothesis] = []
    for index in range(count):
        table: dict[str, str] = {"z-diagnostic": f"signature-{index}"}
        table.update(
            {
                f"distractor-{candidate}": (
                    "match" if candidate == index else "other"
                )
                for candidate in range(count - 1)
            }
        )
        consequences.append(OperationalHypothesis(f"hypothesis-{index}", table))
    return tuple(consequences)


def _candidates(count: int = 8) -> tuple[CandidateIntervention, ...]:
    return (
        *(CandidateIntervention(f"distractor-{index}") for index in range(count - 1)),
        CandidateIntervention("z-diagnostic"),
    )


def _world(hypothesis: OperationalHypothesis):
    def observe(intervention_key: object) -> object:
        distribution = hypothesis.distribution(intervention_key)  # type: ignore[arg-type]
        assert len(distribution) == 1
        return next(iter(distribution))

    return observe


def test_expected_information_gain_beats_or_matches_seeded_random_probes() -> None:
    hypotheses = _seeded_hypotheses()
    candidates = _candidates()
    budget = ProbeBudget(max_probes=len(candidates), max_cost=float(len(candidates)))
    active_counts: list[int] = []
    random_counts: list[int] = []

    for seed in range(8):
        space = BayesianVersionSpace(token=90_000 + seed, hypotheses=hypotheses)
        runs = run_policy_baselines(
            space,
            candidates,
            _world(hypotheses[seed]),
            budget,
            random_seed=seed,
        )
        active = runs["information-gain"]
        random = runs["random"]
        active_counts.append(active.probes_used)
        random_counts.append(random.probes_used)

        assert active.decision.status == "RESOLVED"
        assert random.decision.status == "RESOLVED"
        assert active.probes_used <= random.probes_used
        assert active.ledger.entries[0].intervention_key == "z-diagnostic"
        assert len({run.ledger.design_hash for run in runs.values()}) == 1
        assert all(run.ledger.budget == budget for run in runs.values())

    assert sum(active_counts) < sum(random_counts)


def test_exact_entropy_per_cost_and_bayesian_update() -> None:
    hypotheses = tuple(
        OperationalHypothesis(
            f"h{index}",
            {
                "cheap": "left" if index < 2 else "right",
                "expensive": f"unique-{index}",
            },
        )
        for index in range(4)
    )
    space = BayesianVersionSpace("opaque-token", hypotheses)
    cheap = CandidateIntervention("cheap", cost=1.0)
    expensive = CandidateIntervention("expensive", cost=10.0)

    assert space.score(cheap).information_gain == pytest.approx(1.0)
    assert space.score(expensive).information_gain == pytest.approx(2.0)
    choice = InformationGainPolicy().select(
        space,
        (expensive, cheap),
        # The policy does not mutate or infer budgets from this empty ledger.
        run_acquisition(
            space,
            (cheap,),
            lambda _payload: "left",
            ProbeBudget(0, 0.0),
        ).ledger,
        ProbeBudget(1, 10.0),
    )
    assert choice is not None and choice.intervention.key == "cheap"

    updated = space.update(cheap, "left")
    assert dict(updated.posterior_items) == {"h0": 0.5, "h1": 0.5, "h2": 0.0, "h3": 0.0}
    assert updated.entropy == pytest.approx(1.0)


def test_identical_hypotheses_stay_ambiguous_without_inventing_a_label() -> None:
    hypotheses = (
        OperationalHypothesis("candidate-a", {"probe": "same"}, prior=9.0),
        OperationalHypothesis("candidate-b", {"probe": "same"}, prior=1.0),
    )
    space = BayesianVersionSpace(123_456, hypotheses)
    candidate = CandidateIntervention("probe")

    decision = space.decision((candidate,), confidence=0.6)
    run = run_acquisition(
        space,
        (candidate,),
        lambda _payload: "same",
        ProbeBudget(5, 5.0),
        confidence=0.6,
    )

    assert decision.status == "AMBIGUOUS"
    assert decision.abstained
    assert decision.hypothesis_id is None
    assert decision.reason == "operationally-non-identifiable"
    assert decision.equivalence_classes == (("candidate-a", "candidate-b"),)
    assert run.decision == decision
    assert run.probes_used == 0


def test_token_renaming_is_equivariant_for_probe_choice_posterior_and_ledger() -> None:
    hypotheses = _seeded_hypotheses(4)
    candidates = _candidates(4)
    budget = ProbeBudget(4, 4.0)
    first = run_acquisition(
        BayesianVersionSpace(101, hypotheses),
        tuple(reversed(candidates)),
        _world(hypotheses[2]),
        budget,
    )
    renamed = run_acquisition(
        BayesianVersionSpace("fresh-opaque-name", hypotheses),
        candidates,
        _world(hypotheses[2]),
        budget,
    )

    assert first.decision.token == 101
    assert renamed.decision.token == "fresh-opaque-name"
    assert first.decision.hypothesis_id == renamed.decision.hypothesis_id == "hypothesis-2"
    assert first.version_space.posterior_items == renamed.version_space.posterior_items
    assert first.ledger.entries == renamed.ledger.entries
    assert first.ledger.design_hash == renamed.ledger.design_hash
    assert first.ledger.ledger_hash == renamed.ledger.ledger_hash


def _observation(x: int, tick: int) -> Observation:
    pixels = np.zeros((5, 5, 3), dtype=np.uint8)
    pixels[2, x] = (20, 100, 240)
    return Observation(pixels, tick)


def test_current_transition_and_grounding_evidence_are_token_free_inputs() -> None:
    transition = Transition(
        _observation(1, 0),
        Action(7_001, (1, 2), (1, 0)),
        _observation(2, 1),
        8_009,
    )
    first = GroundingEvidence.from_transition(111, transition, task_feedback=True)
    renamed = GroundingEvidence.from_transition(999_999, transition, task_feedback=True)

    transition_signature = learner_visible_consequence(transition)
    first_signature = learner_visible_consequence(first)
    renamed_signature = learner_visible_consequence(renamed)

    assert isinstance(first_signature, ObservableConsequence)
    assert first_signature == renamed_signature
    assert transition_signature != first_signature  # task feedback is public evidence
    assert first_signature.action_code == transition.action.code
    assert first_signature.outcome_code == transition.outcome_code
    assert first_signature.pixels_changed
    assert first_signature.changed_values == 6


def test_v1_transition_needs_no_semantic_outcome_code() -> None:
    transition = PublicTransition(
        _observation(1, 0),
        Action(7_001, (1, 2), (1, 0)),
        _observation(2, 1),
        scalar_feedback=0.25,
    )

    signature = learner_visible_consequence(transition)

    assert isinstance(signature, ObservableConsequence)
    assert signature.outcome_code is None
    assert signature.scalar_feedback == pytest.approx(0.25)
    assert signature.pixels_changed


class _GuardedEvidence:
    def __init__(self, transition: Transition) -> None:
        self.before = transition.before
        self.action = transition.action
        self.after = transition.after
        self.outcome_code = transition.outcome_code

    def __getattr__(self, name: str) -> object:
        if name in {"task_feedback", "scalar_feedback"}:
            return None
        raise AssertionError(f"unexpected field access: {name}")


def test_active_module_uses_no_privileged_types_or_fields() -> None:
    source_path = Path(__file__).parents[1] / "grounding_kernel" / "active.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert imported_names.isdisjoint(
        {"Oracle", "PredicateKind", "ActionKind", "OutcomeKind", "EvaluatorHarness"}
    )

    transition = Transition(
        _observation(1, 0),
        Action(77, (1, 2), (1, 0)),
        _observation(2, 1),
        88,
    )
    assert isinstance(learner_visible_consequence(_GuardedEvidence(transition)), ObservableConsequence)


def test_ledger_is_deterministic_cost_accounted_and_records_tie_breaking() -> None:
    hypotheses = (
        OperationalHypothesis("h0", {"a": "x", "b": "x"}),
        OperationalHypothesis("h1", {"a": "y", "b": "y"}),
    )
    candidates = (
        CandidateIntervention("b", cost=0.75),
        CandidateIntervention("a", cost=0.75),
    )
    budget = ProbeBudget(2, 1.5)

    def execute() -> object:
        return run_acquisition(
            BayesianVersionSpace("token", hypotheses),
            candidates,
            lambda _payload: "x",
            budget,
        )

    first = execute()
    second = execute()

    assert first == second
    assert first.ledger.ledger_hash == second.ledger.ledger_hash
    assert first.ledger.entries[0].intervention_key == "a"
    assert first.ledger.probes_used == 1
    assert first.ledger.cost_used == pytest.approx(0.75)
    assert first.ledger.remaining_probes == 1
    assert first.ledger.remaining_cost == pytest.approx(0.75)
    assert first.ledger.entries[0].remaining_cost == pytest.approx(0.75)


def test_zero_likelihood_observation_is_ledgered_and_abstained() -> None:
    hypotheses = (
        OperationalHypothesis("h0", {"probe": "x"}),
        OperationalHypothesis("h1", {"probe": "y"}),
    )
    run = run_acquisition(
        BayesianVersionSpace("token", hypotheses),
        (CandidateIntervention("probe"),),
        lambda _payload: "outside-model",
        ProbeBudget(1, 1.0),
    )

    assert run.decision.status == "MODEL_MISSPECIFIED"
    assert run.decision.hypothesis_id is None
    assert run.decision.abstained
    assert len(run.ledger.entries) == 1
    assert run.ledger.entries[0].consistent is False
    assert run.ledger.entries[0].predictive_probability == 0.0
