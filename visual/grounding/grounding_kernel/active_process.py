"""Leakage-free active-acquisition experiment over ProcessWorld twins.

Evaluator code creates matched counterfactual worlds, but the learner-facing
model is fitted exclusively from public pixel consequences measured in
independent support worlds.  Holdout interventions are real opaque actions
executed through ``ProcessWorld.reset/step`` and charged one unit per action by
the authoritative v1 session ledger.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from hmac import new as new_hmac
from itertools import permutations
from math import fsum, isclose
from secrets import token_bytes
from statistics import mean
from types import MappingProxyType

from .active import (
    AcquisitionCommitments,
    AcquisitionRun,
    BayesianVersionSpace,
    CandidateIntervention,
    InformationGainPolicy,
    OperationalSupportRecord,
    PassivePolicy,
    ProbeBudget,
    SensoryTraceConsequence,
    hypotheses_from_support,
    run_acquisition,
    sensory_trace_consequence,
)
from .certificates import manifest_hash
from .contracts import Action, Observation
from .process_quotient import process_pair_quotient
from .processworld import ProcessHarness, ProcessWorld
from .v1_adapters import episode_to_public_trace
from .v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    PROTOCOL_VERSION,
    SENSOR_SCHEMA_RGB_U8,
    PublicTrace,
    PublicTransition,
    SessionManifest,
)
from .v1_session import EvaluationSession, SessionAuditLedger


_PATTERNS = ((False, False), (False, True), (True, False), (True, True))
_SUPPORT_SOURCES = 2
_ACTION_COST = 1.0
# Generated outside the public seed namespace.  It stays evaluator-side and is
# used only to derive per-block target nonces; policies never receive it or a
# target commitment during acquisition.
_EVALUATOR_TARGET_KEY = token_bytes(32)


def _derived_int(seed: int, namespace: str, block: int, *, modulus: int) -> int:
    payload = f"ground-zero-v1-active|{seed}|{namespace}|{block}".encode("utf-8")
    return int.from_bytes(sha256(payload).digest()[:8], "big") % modulus


def _opaque_token(seed: int, namespace: str, block: int) -> int:
    return 100_000_000 + _derived_int(seed, namespace, block, modulus=899_999_999)


def _opaque_key(seed: int, namespace: str, block: int) -> str:
    payload = f"ground-zero-v1-active|{seed}|{namespace}|{block}".encode("utf-8")
    return sha256(payload).hexdigest()


def _private_target_nonce(
    seed: int,
    block: int,
    evaluator_secret: bytes | None,
) -> str:
    key = _EVALUATOR_TARGET_KEY if evaluator_secret is None else evaluator_secret
    if not isinstance(key, bytes) or len(key) < 32:
        raise ValueError("evaluator_secret must contain at least 32 private bytes")
    context = f"ground-zero-v1-active-target|{seed}|{block}".encode("utf-8")
    return new_hmac(key, context, sha256).hexdigest()


@dataclass(frozen=True, slots=True)
class OpaqueProbePlan:
    """Opaque action template with an observation-relative public target cue."""

    key: str
    action_steps: tuple[tuple[int, tuple[int, int]], ...]
    target_fraction: tuple[float, float]

    def __post_init__(self) -> None:
        if not self.key:
            raise ValueError("probe plan key cannot be empty")
        steps = tuple((int(code), tuple(vector)) for code, vector in self.action_steps)
        if not steps or any(len(vector) != 2 for _code, vector in steps):
            raise ValueError("probe plans need opaque code/vector steps")
        fractions = tuple(float(value) for value in self.target_fraction)
        if len(fractions) != 2 or any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("target fractions must lie in [0, 1]")
        object.__setattr__(self, "action_steps", steps)
        object.__setattr__(self, "target_fraction", fractions)

    @classmethod
    def from_public_actions(
        cls,
        key: str,
        actions: Sequence[Action],
        observation: Observation,
    ) -> "OpaqueProbePlan":
        actions = tuple(actions)
        if not actions:
            raise ValueError("a public diagnostic must contain actions")
        if any(action.target != actions[0].target for action in actions):
            raise ValueError("diagnostic actions must share one public target")
        height, width = observation.shape[:2]
        denominator_x = max(width - 1, 1)
        denominator_y = max(height - 1, 1)
        target = actions[0].target
        return cls(
            key,
            tuple((action.code, tuple(action.vector)) for action in actions),
            (target[0] / denominator_x, target[1] / denominator_y),
        )

    @property
    def action_count(self) -> int:
        return len(self.action_steps)

    def target(self, observation: Observation) -> tuple[int, int]:
        height, width = observation.shape[:2]
        return (
            round(self.target_fraction[0] * max(width - 1, 1)),
            round(self.target_fraction[1] * max(height - 1, 1)),
        )

    def actions(self, observation: Observation) -> tuple[Action, ...]:
        target = self.target(observation)
        return tuple(Action(code, target, vector) for code, vector in self.action_steps)

    def manifest(self) -> dict[str, object]:
        return {
            "key": self.key,
            "steps": [
                {"opaque_code": code, "vector": list(vector)}
                for code, vector in self.action_steps
            ],
            "target_fraction": list(self.target_fraction),
            "cost_per_action": _ACTION_COST,
        }


@dataclass(frozen=True, slots=True)
class ObservedPolicyRun:
    """One policy result paired with its authoritative evaluator session."""

    acquisition: AcquisitionRun
    session: SessionAuditLedger
    correct: bool
    censored_cost: float
    censored_probes: float


@dataclass(frozen=True, slots=True)
class InterventionShuffleControl:
    """Executed intervention/consequence derangement with redacted evidence."""

    run: ObservedPolicyRun
    permutation_commitment: str
    evidence_hash: str

    def __post_init__(self) -> None:
        for name in ("permutation_commitment", "evidence_hash"):
            value = getattr(self, name)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or value.lower() != value
            ):
                raise ValueError(f"{name} must be a lowercase SHA-256 digest")
            try:
                bytes.fromhex(value)
            except ValueError as exc:
                raise ValueError(
                    f"{name} must be a lowercase SHA-256 digest"
                ) from exc

    @property
    def prediction(self) -> bool | None:
        """True=correct, False=resolved inversion, None=honest non-resolution."""

        if self.run.acquisition.decision.status != "RESOLVED":
            return None
        return self.run.correct


@dataclass(frozen=True, slots=True)
class ActiveProcessBlockReport:
    """Auditable active trial plus paired exact-order baseline effects."""

    passed: bool
    answered: bool
    active: ObservedPolicyRun
    random_orders: tuple[ObservedPolicyRun, ...]
    passive_orders: tuple[ObservedPolicyRun, ...]
    intervention_shuffle: InterventionShuffleControl
    ambiguous_status: str
    ambiguous_hypothesis_id: Hashable | None
    commitments: AcquisitionCommitments
    support_worlds: int
    hypothesis_count: int
    candidate_count: int
    true_signature_commitment: str
    negative_quotient_profile: str

    @property
    def random_cost(self) -> float:
        return mean(run.censored_cost for run in self.random_orders)

    @property
    def passive_cost(self) -> float:
        return mean(run.censored_cost for run in self.passive_orders)

    @property
    def random_probes(self) -> float:
        return mean(run.censored_probes for run in self.random_orders)

    @property
    def passive_probes(self) -> float:
        return mean(run.censored_probes for run in self.passive_orders)

    @property
    def paired_cost_saving(self) -> float:
        return min(self.random_cost, self.passive_cost) - self.active.censored_cost

    @property
    def paired_probe_saving(self) -> float:
        return min(self.random_probes, self.passive_probes) - self.active.censored_probes

    @property
    def detail(self) -> Mapping[str, object]:
        random_correct = mean(float(run.correct) for run in self.random_orders)
        passive_correct = mean(float(run.correct) for run in self.passive_orders)
        return MappingProxyType(
            {
                "active_probes": self.active.acquisition.probes_used,
                "active_cost": self.active.acquisition.cost_used,
                "random_expected_probes": self.random_probes,
                "random_expected_cost": self.random_cost,
                "passive_counterbalanced_probes": self.passive_probes,
                "passive_counterbalanced_cost": self.passive_cost,
                "paired_probe_saving": self.paired_probe_saving,
                "paired_cost_saving": self.paired_cost_saving,
                "strict_saving": self.paired_cost_saving > 0.0
                and self.paired_probe_saving > 0.0,
                "active_correct": self.active.correct,
                "random_correct_rate": random_correct,
                "passive_correct_rate": passive_correct,
                "random_orders_enumerated": len(self.random_orders),
                "passive_orders_counterbalanced": len(self.passive_orders),
                "baseline_censoring": "full-budget-on-unresolved-or-incorrect",
                "probe_budget": self.active.acquisition.ledger.budget.max_probes,
                "cost_budget": self.active.acquisition.ledger.budget.max_cost,
                "hypothesis_count": self.hypothesis_count,
                "candidate_count": self.candidate_count,
                "support_worlds": self.support_worlds,
                "ambiguous_status": self.ambiguous_status,
                "ambiguous_hypothesis_id": self.ambiguous_hypothesis_id,
                "problem_commitment": self.commitments.problem,
                "policy_commitment": self.commitments.policy,
                "target_commitment": self.commitments.target,
                "experiment_hash": self.active.acquisition.ledger.experiment_hash,
                "active_ledger_hash": self.active.acquisition.ledger.ledger_hash,
                "active_session_ledger_hash": self.active.session.ledger_hash,
                "random_ledger_hashes": tuple(
                    run.acquisition.ledger.ledger_hash for run in self.random_orders
                ),
                "random_session_ledger_hashes": tuple(
                    run.session.ledger_hash for run in self.random_orders
                ),
                "passive_ledger_hashes": tuple(
                    run.acquisition.ledger.ledger_hash for run in self.passive_orders
                ),
                "passive_session_ledger_hashes": tuple(
                    run.session.ledger_hash for run in self.passive_orders
                ),
                "intervention_shuffle_prediction": self.intervention_shuffle.prediction,
                "intervention_shuffle_status": (
                    self.intervention_shuffle.run.acquisition.decision.status
                ),
                "intervention_shuffle_permutation_commitment": (
                    self.intervention_shuffle.permutation_commitment
                ),
                "intervention_shuffle_evidence_hash": (
                    self.intervention_shuffle.evidence_hash
                ),
                "intervention_shuffle_ledger_hash": (
                    self.intervention_shuffle.run.acquisition.ledger.ledger_hash
                ),
                "intervention_shuffle_session_ledger_hash": (
                    self.intervention_shuffle.run.session.ledger_hash
                ),
                "true_signature_commitment": self.true_signature_commitment,
                "negative_quotient_profile": self.negative_quotient_profile,
            }
        )


def _execute_plan(
    world: ProcessWorld,
    plan: OpaqueProbePlan,
    session: EvaluationSession | None = None,
) -> SensoryTraceConsequence:
    initial = world.reset()
    public: list[PublicTransition] = []
    for action in plan.actions(initial):
        transition = world.step(action)
        visible = PublicTransition(
            transition.before,
            transition.action,
            transition.after,
            None,
        )
        if session is not None:
            session.record_experiment(visible, _ACTION_COST)
        public.append(visible)
    return sensory_trace_consequence(PublicTrace(initial, tuple(public)))


def _plan_from_counterfactual(
    key: str,
    counterfactuals: object,
) -> OpaqueProbePlan:
    worlds = tuple(getattr(counterfactuals, "worlds"))
    actions = tuple(getattr(counterfactuals, "diagnostic_actions"))
    target = tuple(getattr(counterfactuals, "target"))
    if len(worlds) != len(_PATTERNS):
        raise RuntimeError("counterfactual family must preserve every declared pattern")
    initial = worlds[0].reset()
    if any(world.reset() != initial for world in worlds[1:]):
        raise RuntimeError("counterfactual worlds must have identical public initial sensors")
    if any(action.target != target for action in actions):
        raise RuntimeError("counterfactual diagnostic actions must share the public target")
    return OpaqueProbePlan.from_public_actions(key, actions, initial)


def _decoy_plan(
    key: str,
    diagnostic: OpaqueProbePlan,
) -> OpaqueProbePlan:
    # The upper image centre is publicly in-bounds and outside both structures.
    first_code, first_vector = diagnostic.action_steps[0]
    steps = tuple((first_code, first_vector) for _step in diagnostic.action_steps)
    return OpaqueProbePlan(key, steps, (0.5, 0.05))


def _intervention_derangement(
    seed: int,
    block: int,
    plans: Sequence[OpaqueProbePlan],
) -> Mapping[str, str]:
    keys = tuple(plan.key for plan in plans)
    if len(keys) < 2 or len(set(keys)) != len(keys):
        raise ValueError("shuffle control needs at least two unique interventions")
    action_counts = {plan.action_count for plan in plans}
    if len(action_counts) != 1:
        raise ValueError("shuffle control requires equal actual action costs")
    shift = 1 + _derived_int(
        seed,
        "intervention-consequence-derangement",
        block,
        modulus=len(keys) - 1,
    )
    mapping = {
        key: keys[(index + shift) % len(keys)] for index, key in enumerate(keys)
    }
    if set(mapping) != set(mapping.values()) or any(
        requested == executed for requested, executed in mapping.items()
    ):
        raise RuntimeError("shuffle mapping must be a fixpoint-free permutation")
    return MappingProxyType(mapping)


def _counterfactual_sets(
    harness: ProcessHarness,
) -> tuple[object, object]:
    left = harness.oracle.affordance_counterfactuals(_PATTERNS, target_index=0)
    right = harness.oracle.affordance_counterfactuals(_PATTERNS, target_index=1)
    if tuple(getattr(left, "target")) == tuple(getattr(right, "target")):
        raise RuntimeError("the two public target cues must be distinct")
    return left, right


def _candidate_plans(
    seed: int,
    block: int,
    support_harness: ProcessHarness,
) -> tuple[OpaqueProbePlan, ...]:
    left_set, right_set = _counterfactual_sets(support_harness)
    left = _plan_from_counterfactual(_opaque_key(seed, "policy-0", block), left_set)
    right = _plan_from_counterfactual(_opaque_key(seed, "policy-1", block), right_set)
    if left.action_steps != right.action_steps:
        raise RuntimeError("matched targets must use identical opaque action sequences")
    decoy_count = 1 + _derived_int(seed, "decoy-count", block, modulus=2)
    decoys = tuple(
        _decoy_plan(_opaque_key(seed, f"decoy-{index}", block), left)
        for index in range(decoy_count)
    )
    return (left, right, *decoys)


def _table_digest(table: Mapping[str, SensoryTraceConsequence]) -> str:
    return manifest_hash(
        {
            "consequences": [
                {"intervention": key, "sensory_digest": table[key].digest}
                for key in sorted(table)
            ]
        }
    )


def _support_records(
    seed: int,
    block: int,
    world_seed: int,
    renderer_variant: int,
    plans: Sequence[OpaqueProbePlan],
) -> tuple[
    tuple[OperationalSupportRecord, ...],
    tuple[str, ...],
    Mapping[str, Mapping[str, SensoryTraceConsequence]],
]:
    table_by_source_and_pattern: list[tuple[str, tuple[Mapping[str, SensoryTraceConsequence], ...]]] = []
    for source_index in range(_SUPPORT_SOURCES):
        world_variant = 1_000 + block * 10 + source_index
        harness = ProcessHarness(
            world_seed,
            renderer_variant=renderer_variant,
            world_variant=world_variant,
        )
        left_set, right_set = _counterfactual_sets(harness)
        sets = (left_set, right_set)
        source_id = _opaque_key(seed, f"support-world-{source_index}", block)
        pattern_tables: list[Mapping[str, SensoryTraceConsequence]] = []
        for pattern_index in range(len(_PATTERNS)):
            rows: dict[str, SensoryTraceConsequence] = {}
            for plan_index, plan in enumerate(plans):
                selected_set = sets[plan_index] if plan_index < 2 else left_set
                world = tuple(getattr(selected_set, "worlds"))[pattern_index]
                rows[plan.key] = _execute_plan(world, plan)
            pattern_tables.append(MappingProxyType(rows))
        table_by_source_and_pattern.append((source_id, tuple(pattern_tables)))

    reference_tables = table_by_source_and_pattern[0][1]
    table_digests = tuple(_table_digest(table) for table in reference_tables)
    if len(set(table_digests)) != len(_PATTERNS):
        raise RuntimeError("support interventions must distinguish four operational hypotheses")
    hypothesis_ids = tuple(
        sha256(f"blind-signature|{seed}|{block}|{digest}".encode("utf-8")).hexdigest()
        for digest in table_digests
    )
    records: list[OperationalSupportRecord] = []
    for source_id, tables in table_by_source_and_pattern:
        if tuple(_table_digest(table) for table in tables) != table_digests:
            raise RuntimeError("independent support worlds disagree on operational signatures")
        for hypothesis_id, table in zip(hypothesis_ids, tables, strict=True):
            for intervention, consequence in table.items():
                records.append(
                    OperationalSupportRecord(
                        hypothesis_id,
                        intervention,
                        consequence,
                        source_id,
                    )
                )
    table_by_hypothesis = MappingProxyType(
        {
            hypothesis_id: reference_tables[index]
            for index, hypothesis_id in enumerate(hypothesis_ids)
        }
    )
    return tuple(records), hypothesis_ids, table_by_hypothesis


def _design_commitments(
    records: Sequence[OperationalSupportRecord],
    plans: Sequence[OpaqueProbePlan],
) -> AcquisitionCommitments:
    problem = manifest_hash(
        {
            "support": [
                {
                    "hypothesis": str(record.hypothesis_id),
                    "intervention": str(record.intervention_key),
                    "sensory_consequence": getattr(record.consequence, "digest", repr(record.consequence)),
                    "source": str(record.source_id),
                }
                for record in records
            ],
            "minimum_independent_sources": _SUPPORT_SOURCES,
        }
    )
    policy = manifest_hash({"policies": [plan.manifest() for plan in plans]})
    return AcquisitionCommitments(problem, policy)


def _target_commitment(
    token: int,
    true_hypothesis: str,
    nonce: str,
) -> str:
    return manifest_hash(
        {
            "nonce": nonce,
            "opaque_token": token,
            "positive_operational_signature": true_hypothesis,
        }
    )


def _session(
    budget: ProbeBudget,
    commitments: AcquisitionCommitments,
    codebook_commitment: str,
) -> EvaluationSession:
    manifest = SessionManifest(
        PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        support_episode_budget=0,
        intervention_cost_budget=budget.max_cost,
        query_budget=1,
    )
    session = EvaluationSession(manifest, commitments.digest)
    session.commit_codebook(codebook_commitment)
    session.begin_acquisition()
    return session


def _observer_and_session(
    *,
    world_seed: int,
    renderer_variant: int,
    world_variant: int,
    pattern_index: int,
    plans: Sequence[OpaqueProbePlan],
    budget: ProbeBudget,
    commitments: AcquisitionCommitments,
    codebook_commitment: str,
    consequence_derangement: Mapping[str, str] | None = None,
):
    harness = ProcessHarness(
        world_seed,
        renderer_variant=renderer_variant,
        world_variant=world_variant,
    )
    left_set, right_set = _counterfactual_sets(harness)
    worlds_by_key: dict[str, ProcessWorld] = {}
    for index, plan in enumerate(plans):
        selected_set = (left_set, right_set)[index] if index < 2 else left_set
        worlds_by_key[plan.key] = tuple(getattr(selected_set, "worlds"))[pattern_index]
    plans_by_key = {plan.key: plan for plan in plans}
    execution_keys = (
        {key: key for key in plans_by_key}
        if consequence_derangement is None
        else dict(consequence_derangement)
    )
    if set(execution_keys) != set(plans_by_key) or set(execution_keys.values()) != set(
        plans_by_key
    ):
        raise ValueError("consequence derangement must permute every intervention")
    if any(
        plans_by_key[requested].action_count
        != plans_by_key[executed].action_count
        for requested, executed in execution_keys.items()
    ):
        raise ValueError("deranged probes must preserve actual action cost")
    session = _session(budget, commitments, codebook_commitment)

    def observe(payload: object) -> SensoryTraceConsequence:
        if not isinstance(payload, OpaqueProbePlan):
            raise TypeError("ProcessWorld active probes require an OpaqueProbePlan")
        execution_key = execution_keys[payload.key]
        execution_plan = plans_by_key[execution_key]
        return _execute_plan(worlds_by_key[execution_key], execution_plan, session)

    return observe, session


def _finish_run(
    acquisition: AcquisitionRun,
    session: EvaluationSession,
    true_hypothesis: str,
    target_commitment: str,
    budget: ProbeBudget,
) -> ObservedPolicyRun:
    correct = (
        acquisition.decision.status == "RESOLVED"
        and acquisition.decision.hypothesis_id == true_hypothesis
    )
    if not isclose(
        acquisition.cost_used,
        session.ledger.intervention_cost_used,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise RuntimeError("active and authoritative session costs disagree")
    acquisition = acquisition.bind_target_audit(target_commitment)
    checkpoint = manifest_hash(
        {
            "posterior": [list(item) for item in acquisition.version_space.posterior_items],
            "active_ledger": acquisition.ledger.ledger_hash,
        }
    )
    session.freeze(checkpoint)
    audit = session.complete()
    return ObservedPolicyRun(
        acquisition,
        audit,
        correct,
        acquisition.cost_used if correct else budget.max_cost,
        float(acquisition.probes_used if correct else budget.max_probes),
    )


def _run_order(
    *,
    state: BayesianVersionSpace,
    candidates: Sequence[CandidateIntervention],
    plans: Sequence[OpaqueProbePlan],
    order: Sequence[str] | None,
    policy_name: str,
    world_seed: int,
    renderer_variant: int,
    world_variant: int,
    pattern_index: int,
    budget: ProbeBudget,
    commitments: AcquisitionCommitments,
    target_commitment: str,
    codebook_commitment: str,
    true_hypothesis: str,
    consequence_derangement: Mapping[str, str] | None = None,
) -> ObservedPolicyRun:
    observe, session = _observer_and_session(
        world_seed=world_seed,
        renderer_variant=renderer_variant,
        world_variant=world_variant,
        pattern_index=pattern_index,
        plans=plans,
        budget=budget,
        commitments=commitments,
        codebook_commitment=codebook_commitment,
        consequence_derangement=consequence_derangement,
    )
    policy = (
        InformationGainPolicy(name=policy_name)
        if order is None
        else PassivePolicy(tuple(order), name=policy_name)
    )
    acquisition = run_acquisition(
        state,
        candidates,
        observe,
        budget,
        policy=policy,
        commitments=commitments,
    )
    return _finish_run(
        acquisition,
        session,
        true_hypothesis,
        target_commitment,
        budget,
    )


def _shuffle_control(
    run: ObservedPolicyRun,
    derangement: Mapping[str, str],
) -> InterventionShuffleControl:
    permutation_commitment = manifest_hash(
        {
            "kind": "executed-intervention-consequence-derangement-v1",
            "mapping": [
                {"requested": requested, "executed": derangement[requested]}
                for requested in sorted(derangement)
            ],
        }
    )
    prediction = (
        run.correct if run.acquisition.decision.status == "RESOLVED" else None
    )
    evidence_hash = manifest_hash(
        {
            "permutation_commitment": permutation_commitment,
            "prediction": prediction,
            "decision_status": run.acquisition.decision.status,
            "acquisition_ledger": run.acquisition.ledger.ledger_hash,
            "session_ledger": run.session.ledger_hash,
        }
    )
    return InterventionShuffleControl(run, permutation_commitment, evidence_hash)


def _ambiguous_negative_trial(
    seed: int,
    block: int,
    world_seed: int,
    renderer_variant: int,
) -> tuple[str, Hashable | None, str]:
    support: list[OperationalSupportRecord] = []
    negative_ids = (
        _opaque_key(seed, "negative-hypothesis-0", block),
        _opaque_key(seed, "negative-hypothesis-1", block),
    )
    intervention = _opaque_key(seed, "negative-intervention", block)
    for source_index in range(_SUPPORT_SOURCES):
        harness = ProcessHarness(
            world_seed,
            renderer_variant=renderer_variant,
            world_variant=30_000 + block * 10 + source_index,
        )
        pair = harness.oracle.negative_control_pair()
        consequences = tuple(
            sensory_trace_consequence(
                episode_to_public_trace(record.episode, strip_feedback=True)
            )
            for record in pair
        )
        if consequences[0] != consequences[1]:
            raise RuntimeError("negative counterfactuals must be publicly identical")
        source_id = _opaque_key(seed, f"negative-source-{source_index}", block)
        support.extend(
            OperationalSupportRecord(identifier, intervention, consequences[index], source_id)
            for index, identifier in enumerate(negative_ids)
        )
    hypotheses = hypotheses_from_support(support, minimum_sources=_SUPPORT_SOURCES)
    state = BayesianVersionSpace(
        _opaque_token(seed, "negative-token", block),
        hypotheses,
    )
    decision = state.decision((CandidateIntervention(intervention, cost=2.0),))
    holdout = ProcessHarness(
        world_seed,
        renderer_variant=renderer_variant,
        world_variant=40_000 + block,
    )
    quotient = process_pair_quotient(
        *holdout.oracle.negative_control_pair(),
        commitment_nonce=_opaque_key(seed, "negative-target-nonce", block),
    )
    if quotient.groundable:
        raise RuntimeError("negative control must fail quotient factorization")
    return decision.status, decision.hypothesis_id, quotient.profile_hash


def run_active_process_block(
    seed: int,
    block: int,
    *,
    evaluator_secret: bytes | None = None,
) -> ActiveProcessBlockReport:
    """Run one committed, paired active ProcessWorld block."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if isinstance(block, bool) or not isinstance(block, int) or block < 0:
        raise ValueError("block must be a non-negative integer")
    world_seed = _derived_int(seed, "shared-codebook", block, modulus=1 << 31)
    renderer_variant = _derived_int(seed, "shared-renderer", block, modulus=1 << 20)
    support_harness = ProcessHarness(
        world_seed,
        renderer_variant=renderer_variant,
        world_variant=500 + block,
    )
    plans = _candidate_plans(seed, block, support_harness)
    records, hypothesis_ids, _support_tables = _support_records(
        seed,
        block,
        world_seed,
        renderer_variant,
        plans,
    )
    hypotheses = hypotheses_from_support(records, minimum_sources=_SUPPORT_SOURCES)
    if len(hypotheses) < 4:
        raise RuntimeError("active ProcessWorld blocks require at least four hypotheses")

    token = _opaque_token(seed, "active-token", block)
    target_selector = _derived_int(seed, "sealed-positive-signature", block, modulus=len(_PATTERNS))
    true_hypothesis = hypothesis_ids[target_selector]
    commitments = _design_commitments(records, plans)
    target_commitment = _target_commitment(
        token,
        true_hypothesis,
        _private_target_nonce(seed, block, evaluator_secret),
    )
    state = BayesianVersionSpace(token, hypotheses)
    candidates = tuple(
        CandidateIntervention(
            plan.key,
            payload=plan,
            cost=float(plan.action_count),
        )
        for plan in plans
    )
    budget = ProbeBudget(
        max_probes=len(candidates),
        max_cost=fsum(candidate.cost for candidate in candidates),
    )
    codebook_commitment = manifest_hash(
        {
            "nonce": _opaque_key(seed, "codebook-nonce", block),
            "opaque_action_codes": list(support_harness.agent.action_codes),
        }
    )
    holdout_variant = 20_000 + block

    active = _run_order(
        state=state,
        candidates=candidates,
        plans=plans,
        order=None,
        policy_name="information-gain",
        world_seed=world_seed,
        renderer_variant=renderer_variant,
        world_variant=holdout_variant,
        pattern_index=target_selector,
        budget=budget,
        commitments=commitments,
        target_commitment=target_commitment,
        codebook_commitment=codebook_commitment,
        true_hypothesis=true_hypothesis,
    )

    keys = tuple(candidate.key for candidate in candidates)
    random_orders = tuple(
        _run_order(
            state=state,
            candidates=candidates,
            plans=plans,
            order=order,
            policy_name="random-exact-order",
            world_seed=world_seed,
            renderer_variant=renderer_variant,
            world_variant=holdout_variant,
            pattern_index=target_selector,
            budget=budget,
            commitments=commitments,
            target_commitment=target_commitment,
            codebook_commitment=codebook_commitment,
            true_hypothesis=true_hypothesis,
        )
        for order in permutations(keys)
    )
    passive_orders = tuple(
        _run_order(
            state=state,
            candidates=candidates,
            plans=plans,
            order=order,
            policy_name="passive-counterbalanced",
            world_seed=world_seed,
            renderer_variant=renderer_variant,
            world_variant=holdout_variant,
            pattern_index=target_selector,
            budget=budget,
            commitments=commitments,
            target_commitment=target_commitment,
            codebook_commitment=codebook_commitment,
            true_hypothesis=true_hypothesis,
        )
        for order in (keys, tuple(reversed(keys)))
    )
    shuffle_derangement = _intervention_derangement(seed, block, plans)
    shuffled_run = _run_order(
        state=state,
        candidates=candidates,
        plans=plans,
        order=None,
        policy_name="intervention-consequence-shuffle",
        world_seed=world_seed,
        renderer_variant=renderer_variant,
        world_variant=holdout_variant,
        pattern_index=target_selector,
        budget=budget,
        commitments=commitments,
        target_commitment=target_commitment,
        codebook_commitment=codebook_commitment,
        true_hypothesis=true_hypothesis,
        consequence_derangement=shuffle_derangement,
    )
    intervention_shuffle = _shuffle_control(shuffled_run, shuffle_derangement)
    ambiguous_status, ambiguous_id, negative_profile = _ambiguous_negative_trial(
        seed,
        block,
        world_seed,
        renderer_variant,
    )
    experiment_hashes = {
        run.acquisition.ledger.experiment_hash
        for run in (active, *random_orders, *passive_orders, shuffled_run)
    }
    sessions_match_cost = all(
        isclose(
            run.acquisition.cost_used,
            run.session.intervention_cost_used,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        for run in (active, *random_orders, *passive_orders, shuffled_run)
    )
    audit_commitments = commitments.bind_target(target_commitment)
    provisional = ActiveProcessBlockReport(
        passed=False,
        answered=active.correct,
        active=active,
        random_orders=random_orders,
        passive_orders=passive_orders,
        intervention_shuffle=intervention_shuffle,
        ambiguous_status=ambiguous_status,
        ambiguous_hypothesis_id=ambiguous_id,
        commitments=audit_commitments,
        support_worlds=_SUPPORT_SOURCES * len(_PATTERNS),
        hypothesis_count=len(hypotheses),
        candidate_count=len(candidates),
        true_signature_commitment=target_commitment,
        negative_quotient_profile=negative_profile,
    )
    passed = (
        active.correct
        and all(run.correct for run in random_orders)
        and all(run.correct for run in passive_orders)
        and intervention_shuffle.prediction is not True
        and provisional.paired_cost_saving > 0.0
        and provisional.paired_probe_saving > 0.0
        and ambiguous_status == "AMBIGUOUS"
        and ambiguous_id is None
        and len(experiment_hashes) == 1
        and sessions_match_cost
    )
    return ActiveProcessBlockReport(
        passed=passed,
        answered=active.correct,
        active=active,
        random_orders=random_orders,
        passive_orders=passive_orders,
        intervention_shuffle=intervention_shuffle,
        ambiguous_status=ambiguous_status,
        ambiguous_hypothesis_id=ambiguous_id,
        commitments=audit_commitments,
        support_worlds=provisional.support_worlds,
        hypothesis_count=provisional.hypothesis_count,
        candidate_count=provisional.candidate_count,
        true_signature_commitment=provisional.true_signature_commitment,
        negative_quotient_profile=negative_profile,
    )


__all__ = [
    "ActiveProcessBlockReport",
    "InterventionShuffleControl",
    "ObservedPolicyRun",
    "OpaqueProbePlan",
    "run_active_process_block",
]
