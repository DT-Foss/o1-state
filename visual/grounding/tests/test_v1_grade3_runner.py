from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from grounding_kernel.certificates import manifest_hash
from grounding_kernel.contracts import Action, Observation, Transition
from grounding_kernel.v1_contracts import (
    BeliefDecision,
    DescriptionDecision,
    PublicTrace,
    PublicTransition,
    PublicTurn,
    SessionPhase,
    Utterance,
)
from grounding_kernel.v1_grade3_contracts import (
    GRADE3_PROTOCOL_VERSION,
    Grade3SessionManifest,
    MotorActionSpace,
    MotorDecision,
    MotorDirective,
    OstensiveSupportRecord,
    ProbeDecision,
    ProbeEvidence,
    ProbeOffer,
    ProbeOption,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)
from grounding_kernel.v1_grade3_isolation import (
    GRADE3_ISOLATION_PROTOCOL,
    FrozenGrade3Candidate,
    Grade3ArtifactCommitment,
    IsolatedGrade3Grounder as RealIsolatedGrade3Grounder,
    commit_grade3_candidate,
)
from grounding_kernel.v1_grade3_runner import Grade3EvaluationRunner
from grounding_kernel.v1_grade3_session import (
    Grade3SessionBudgetError,
    Grade3SessionStateError,
)


def observation(tick: int, marker: int = 0, *, terminal: bool = False) -> Observation:
    pixels = np.zeros((12, 12, 3), dtype=np.uint8)
    pixels[2:4, 2:4] = marker
    return Observation(pixels, tick, terminal)


ACTION = Action(7, (2, 2), (1, 0))
ACTION_SPACE = MotorActionSpace((7,), ((1, 0),), 4)


def one_step_trace(marker: int = 1) -> PublicTrace:
    before = observation(0, marker)
    after = observation(1, marker + 1)
    return PublicTrace(before, (PublicTransition(before, ACTION, after, None),))


def manifest(
    *,
    acquisition: float = 4.0,
    queries: int = 4,
    actions: float = 8.0,
    resets: int = 5,
) -> Grade3SessionManifest:
    return Grade3SessionManifest(
        GRADE3_PROTOCOL_VERSION,
        "rgb-u8-v1",
        "opaque-motor-target-vector-v1",
        3,
        acquisition,
        queries,
        actions,
        resets,
    )


def artifact() -> Grade3ArtifactCommitment:
    candidate_files = (("__init__.py", "11" * 32),)
    sdk_files = (("contracts.py", "22" * 32),)
    sdk = manifest_hash({"sdk_files": list(sdk_files)})
    digest = manifest_hash(
        {
            "version": GRADE3_ISOLATION_PROTOCOL,
            "entrypoint": "candidate.module:build",
            "package_name": "candidate",
            "candidate_files": list(candidate_files),
            "sdk_files": list(sdk_files),
            "sdk_commitment": sdk,
        }
    )
    return Grade3ArtifactCommitment(
        "candidate.module:build",
        "candidate",
        "/nonexistent/candidate",
        candidate_files,
        "/nonexistent/sdk",
        sdk_files,
        sdk,
        digest,
    )


class FakeIsolatedGrade3Grounder:
    latest: FakeIsolatedGrade3Grounder | None = None
    motor_policy: Callable[[object], MotorDecision] | None = None
    fail_belief = False

    def __init__(self, commitment: Grade3ArtifactCommitment, *, timeout: float) -> None:
        self.commitment = commitment
        self.timeout = timeout
        self.started = False
        self.begun = False
        self.closed = False
        self.support_records: list[object] = []
        self.probe_results: list[object] = []
        self.motor_queries: list[object] = []
        self.motor_index = 0
        self.frozen: FrozenGrade3Candidate | None = None
        type(self).latest = self

    def start(self) -> FakeIsolatedGrade3Grounder:
        self.started = True
        return self

    def begin(self, _manifest: Grade3SessionManifest) -> None:
        assert self.started
        self.begun = True

    def observe_support(self, record: object) -> None:
        self.support_records.append(record)

    def choose_probe(self, offer: ProbeOffer) -> ProbeDecision:
        return ProbeDecision(offer.options[0].probe_id, 0.0)

    def observe_probe(self, result: object) -> None:
        self.probe_results.append(result)

    def freeze(self) -> FrozenGrade3Candidate:
        self.frozen = FrozenGrade3Candidate(
            self.commitment.digest,
            self.commitment.sdk_commitment,
            "44" * 32,
            7,
        )
        return self.frozen

    def motor(self, query: object) -> MotorDecision:
        self.motor_queries.append(query)
        policy = type(self).motor_policy
        if policy is not None:
            return policy(query)
        decisions = (
            MotorDecision(MotorDirective.ACT, ACTION, 0.0),
            MotorDecision(MotorDirective.RESET_EXECUTE, None, 0.0),
            MotorDecision(MotorDirective.ACT, ACTION, 0.0),
            MotorDecision(MotorDirective.COMPLETE, None, 0.0),
        )
        decision = decisions[self.motor_index]
        self.motor_index += 1
        return decision

    def trace_belief(self, _query: TraceBeliefQuery) -> BeliefDecision:
        if type(self).fail_belief:
            raise RuntimeError("belief exploded")
        return BeliefDecision(((5, 1.0),), 0.0)

    def describe(self, _query: TraceDescriptionQuery) -> DescriptionDecision:
        return DescriptionDecision(Utterance((101, 202)), 0.0)

    def assert_frozen(self) -> str:
        assert self.frozen is not None
        return self.frozen.checkpoint_commitment

    def close(self) -> None:
        self.closed = True


@pytest.fixture(autouse=True)
def fake_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeIsolatedGrade3Grounder.latest = None
    FakeIsolatedGrade3Grounder.motor_policy = None
    FakeIsolatedGrade3Grounder.fail_belief = False
    monkeypatch.setattr(
        "grounding_kernel.v1_grade3_runner.IsolatedGrade3Grounder",
        FakeIsolatedGrade3Grounder,
    )


def started_runner(**manifest_kwargs: object) -> Grade3EvaluationRunner:
    return Grade3EvaluationRunner(
        artifact(),
        manifest(**manifest_kwargs),  # type: ignore[arg-type]
        lambda: "33" * 32,
    ).start()


def support_record() -> OstensiveSupportRecord:
    frame = observation(0)
    turn = PublicTurn(
        0,
        SessionPhase.SUPPORT,
        frame,
        Utterance((101,)),
        (1, 1, 5, 5),
        1.0,
        4.0,
    )
    return OstensiveSupportRecord(1, 2, turn, PublicTrace(frame))


def frozen_runner(**manifest_kwargs: object) -> Grade3EvaluationRunner:
    runner = started_runner(**manifest_kwargs)
    runner.support(support_record())
    runner.begin_acquisition()
    runner.freeze()
    return runner


def test_persistent_lifecycle_probe_motor_and_sealed_queries() -> None:
    runner = started_runner()
    fake = FakeIsolatedGrade3Grounder.latest
    assert fake is not None and fake.begun
    runner.support(support_record())
    assert len(fake.support_records) == 1
    runner.begin_acquisition()
    offer = ProbeOffer(
        1,
        8,
        0,
        (ProbeOption(90, 1.5), ProbeOption(91, 2.0)),
        runner.remaining_acquisition_cost,
    )
    executed = runner.acquire_probe(offer, lambda probe_id: one_step_trace(probe_id % 5))
    assert executed.result is not None
    assert executed.result.remaining_cost == pytest.approx(2.5)
    assert runner.audit.acquisition_cost_used == pytest.approx(1.5)
    runner.freeze()

    state: dict[str, Observation] = {}

    def reset() -> Observation:
        state["current"] = observation(0, 3)
        return state["current"]

    def step(action: Action) -> Transition:
        before = state["current"]
        after = observation(before.tick + 1, 4)
        state["current"] = after
        return Transition(before, action, after, 999)

    result = runner.run_motor_episode(
        1,
        1,
        Utterance((101, 202)),
        ACTION_SPACE,
        reset,
        step,
    )
    assert result.completed and not result.abstained
    assert len(result.completed_probes) == 1
    assert result.execution_trace is not None
    assert len(result.execution_trace.transitions) == 1
    assert len(result.transcript) == 4
    assert result.resets_consumed == 2
    assert result.action_cost_consumed == pytest.approx(2.0)
    assert all(
        transition.scalar_feedback is None
        for trace in (*result.completed_probes, result.execution_trace)
        for transition in trace.transitions
    )

    evidence = (ProbeEvidence(90, one_step_trace()),)
    belief = runner.trace_belief(TraceBeliefQuery(2, 1, 8, (5, 6), evidence))
    description = runner.describe(TraceDescriptionQuery(3, 1, evidence))
    assert belief.candidate_probabilities == ((5, 1.0),)
    assert description.utterance == Utterance((101, 202))
    ledger = runner.complete()
    assert ledger.sealed_queries_used == 3
    assert ledger.motor_action_cost_used == pytest.approx(2.0)
    assert ledger.motor_resets_used == 2
    assert ledger.chain_valid
    assert fake is FakeIsolatedGrade3Grounder.latest
    runner.close()


def test_motor_world_failure_consumes_query_action_and_initial_reset() -> None:
    runner = frozen_runner()
    calls = 0

    def reset() -> Observation:
        return observation(0)

    def mismatched_step(_action: Action) -> PublicTransition:
        nonlocal calls
        calls += 1
        before = observation(0)
        wrong = Action(7, (3, 3), (1, 0))
        return PublicTransition(before, wrong, observation(1), None)

    with pytest.raises(Grade3SessionStateError, match="different action"):
        runner.run_motor_episode(4, 1, Utterance((1, 2)), ACTION_SPACE, reset, mismatched_step)
    assert calls == 1
    assert runner.audit.sealed_queries_used == 1
    assert runner.audit.motor_action_cost_used == pytest.approx(1.0)
    assert runner.audit.motor_resets_used == 1
    assert runner.terminal_failure is not None
    with pytest.raises(Grade3SessionStateError, match="open"):
        runner.describe(TraceDescriptionQuery(5, 1, (ProbeEvidence(1, one_step_trace()),)))


def test_action_budget_is_reserved_before_step_callback() -> None:
    runner = frozen_runner(actions=0.5)
    called = False

    def step(_action: Action) -> Transition:
        nonlocal called
        called = True
        raise AssertionError("must not execute")

    with pytest.raises(Grade3SessionBudgetError, match="budget"):
        runner.run_motor_episode(
            1,
            1,
            Utterance((1, 2)),
            ACTION_SPACE,
            lambda: observation(0),
            step,
        )
    assert not called
    assert runner.audit.sealed_queries_used == 1
    assert runner.audit.motor_action_cost_used == 0.0
    assert runner.audit.motor_resets_used == 1


def test_invalid_reset_phase_is_terminal_and_failed_query_is_not_refunded() -> None:
    runner = frozen_runner()

    def invalid_policy(_query: object) -> MotorDecision:
        # An empty current probe may not be archived.
        return MotorDecision(MotorDirective.RESET_EXECUTE, None, 0.0)

    FakeIsolatedGrade3Grounder.motor_policy = invalid_policy
    reset_calls = 0

    def reset() -> Observation:
        nonlocal reset_calls
        reset_calls += 1
        return observation(0)

    with pytest.raises(Grade3SessionStateError, match="empty causal probe"):
        runner.run_motor_episode(
            1,
            1,
            Utterance((1, 2)),
            ACTION_SPACE,
            reset,
            lambda _action: None,  # type: ignore[arg-type]
        )
    assert reset_calls == 1
    assert runner.audit.sealed_queries_used == 1
    assert runner.audit.motor_resets_used == 1


def test_failed_causal_probe_and_failed_belief_are_consumed() -> None:
    runner = started_runner()
    runner.begin_acquisition()
    offer = ProbeOffer(
        1,
        2,
        0,
        (ProbeOption(9, 2.0),),
        runner.remaining_acquisition_cost,
    )
    with pytest.raises(RuntimeError, match="world failed"):
        runner.acquire_probe(
            offer,
            lambda _probe: (_ for _ in ()).throw(RuntimeError("world failed")),
        )
    assert runner.audit.acquisition_cost_used == pytest.approx(2.0)

    other = frozen_runner()
    FakeIsolatedGrade3Grounder.fail_belief = True
    query = TraceBeliefQuery(1, 1, 2, (5,), (ProbeEvidence(9, one_step_trace()),))
    with pytest.raises(RuntimeError, match="belief exploded"):
        other.trace_belief(query)
    assert other.audit.sealed_queries_used == 1
    assert other.terminal_failure is not None


def test_abstention_terminates_one_logical_query_without_world_action() -> None:
    runner = frozen_runner()
    FakeIsolatedGrade3Grounder.motor_policy = lambda _query: MotorDecision(
        MotorDirective.ABSTAIN, None, 0.75
    )
    step_called = False

    def step(_action: Action) -> Transition:
        nonlocal step_called
        step_called = True
        raise AssertionError("abstention must not step")

    result = runner.run_motor_episode(
        1,
        1,
        Utterance((404,)),
        ACTION_SPACE,
        lambda: observation(0),
        step,
    )
    assert result.abstained and not result.completed
    assert result.unknown_probability == pytest.approx(0.75)
    assert not step_called
    assert runner.audit.sealed_queries_used == 1
    assert runner.audit.motor_action_cost_used == 0.0
    assert runner.audit.motor_resets_used == 1


def test_real_process_runner_closes_probe_motor_belief_and_description_seams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from grounding_kernel.v1_grade3_cases import build_grade3_case

    root = Path(__file__).resolve().parents[1]
    commitment = commit_grade3_candidate(
        root / "grounding_reference_candidate",
        "grounding_reference_candidate.candidate:build",
    )
    # The case is intentionally materialized only after the candidate/SDK
    # bytes have been committed.
    bundle = build_grade3_case(81723, support_worlds=2)
    public = bundle.public
    monkeypatch.setattr(
        "grounding_kernel.v1_grade3_runner.IsolatedGrade3Grounder",
        RealIsolatedGrade3Grounder,
    )
    runner = Grade3EvaluationRunner(
        commitment,
        public.session_manifest,
        lambda: public.case_manifest.public_dataset_commitment,
    ).start()
    try:
        for record in public.support_records:
            runner.support(record)
        runner.begin_acquisition()
        used: list[int] = []
        evidence: list[ProbeEvidence] = []
        for step_index in range(2):
            offer = public.offer(
                step_index=step_index,
                remaining_cost=runner.remaining_acquisition_cost,
                exclude=used,
            )
            executed = runner.acquire_probe(offer, bundle.evaluator.probes.execute)
            assert executed.result is not None
            used.append(executed.result.probe_id)
            evidence.append(ProbeEvidence(executed.result.probe_id, executed.result.trace))
        runner.freeze()

        world = bundle.evaluator.probes.fresh_motor_world()
        motor = runner.run_motor_episode(
            1,
            public.scope_id,
            public.heldout_instruction,
            public.action_space,
            world.reset,
            world.step,
        )
        assert motor.completed
        assert len(motor.completed_probes) == 2
        assert motor.execution_trace is not None
        assert len(motor.execution_trace.transitions) == 4

        belief = runner.trace_belief(
            TraceBeliefQuery(
                2,
                public.scope_id,
                public.problem_id,
                public.hypothesis_candidates,
                tuple(evidence),
            )
        )
        assert belief.candidate_probabilities == ((bundle.evaluator.facts.true_hypothesis_id, 1.0),)
        description_trace = bundle.evaluator.probes.heldout_description_trace()
        description = runner.describe(
            TraceDescriptionQuery(
                3,
                public.scope_id,
                (ProbeEvidence(public.probe_options[0].probe_id, description_trace),),
            )
        )
        assert description.utterance == public.heldout_instruction
        ledger = runner.complete()
        assert ledger.chain_valid
        assert ledger.support_records_used == len(public.support_records)
        assert ledger.acquisition_cost_used == pytest.approx(8.0)
        assert ledger.sealed_queries_used == 3
        assert ledger.motor_resets_used == 3
        assert ledger.motor_action_cost_used == pytest.approx(12.0)
    finally:
        runner.close()
