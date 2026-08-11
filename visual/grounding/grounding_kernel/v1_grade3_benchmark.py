"""Reproducible honest-reference diagnostic for the Grade-3 protocol.

This benchmark is deliberately noncompensatory: every positive operational
axis and every negative control must pass.  It is an honest-reference process
diagnostic, not a universal solution claim and not an adversarial sandbox
certificate.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from collections.abc import Sequence

import numpy as np

from .certificates import manifest_hash
from .contracts import Action, Observation
from .v1_contracts import (
    ACTION_SCHEMA_OPAQUE_MOTOR,
    SENSOR_SCHEMA_RGB_U8,
    PublicTrace,
    PublicTransition,
    Utterance,
)
from .v1_grade3_cases import Grade3CaseBundle, build_grade3_case
from .v1_grade3_contracts import (
    GRADE3_PROTOCOL_VERSION,
    Grade3SessionManifest,
    ProbeEvidence,
    TraceBeliefQuery,
    TraceDescriptionQuery,
)
from .v1_grade3_isolation import Grade3ArtifactCommitment, commit_grade3_candidate
from .v1_grade3_runner import Grade3EvaluationRunner, MotorEpisodeResult


GRADE3_BENCHMARK_PROTOCOL = "grounding-grade3-honest-diagnostic/1"
GRADE3_REFERENCE_ENTRYPOINT = "grounding_reference_candidate.candidate:build"
HONEST_GRADE3_CLAIM = (
    "Honest-reference Grade-3 operational diagnostic; not a universal symbol-"
    "grounding solution, not an adversarial certificate, and not Grade-4 isolation."
)

GRADE3_POSITIVE_AXES = (
    "artifact_before_dataset_materialization",
    "single_persistent_checkpoint",
    "active_causal_belief",
    "heldout_factorial_action",
    "heldout_reverse_description",
    "grounded_definition_base",
    "grounded_definition_middle",
    "grounded_definition_chain",
)

GRADE3_NEGATIVE_CONTROLS = (
    "probe_association_shuffle_safe",
    "temporal_corruption_rejected",
    "static_sensor_not_operational",
    "fresh_symbol_abstention",
    "unannounced_remap_abstention",
    "ungrounded_cycle_abstention",
    "feedback_outcome_free_trajectory_channel",
)


def _integer(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    if value < minimum:
        raise ValueError(f"{field} must be at least {minimum}")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or value.lower() != value:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest") from exc
    return value


@dataclass(frozen=True, slots=True)
class Grade3BenchmarkCheck:
    name: str
    passed: bool
    evidence_hash: str
    note: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("check name must be a nonempty string")
        if not isinstance(self.passed, bool):
            raise TypeError("check passed must be bool")
        _digest(self.evidence_hash, "evidence_hash")
        if not isinstance(self.note, str) or not self.note:
            raise TypeError("check note must be a nonempty string")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "passed": self.passed,
            "evidence_hash": self.evidence_hash,
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class Grade3BenchmarkReport:
    protocol_version: str
    seed: int
    renderer_variant: int
    permutation_variant: int
    support_worlds: int
    artifact_commitment: str
    sdk_commitment: str
    checkpoint_commitment: str
    dataset_commitment: str
    manifest_commitment: str
    schema_commitment: str
    ledger_commitment: str
    benchmark_definition_commitment: str
    candidate_request_count: int
    support_records_consumed: int
    acquisition_cost_consumed: float
    sealed_queries_consumed: int
    motor_action_cost_consumed: float
    motor_resets_consumed: int
    axes: tuple[Grade3BenchmarkCheck, ...]
    controls: tuple[Grade3BenchmarkCheck, ...]
    claim_scope: str = HONEST_GRADE3_CLAIM

    def __post_init__(self) -> None:
        if self.protocol_version != GRADE3_BENCHMARK_PROTOCOL:
            raise ValueError(f"protocol_version must equal {GRADE3_BENCHMARK_PROTOCOL!r}")
        for field in (
            "seed",
            "renderer_variant",
            "permutation_variant",
            "support_worlds",
            "candidate_request_count",
            "support_records_consumed",
            "sealed_queries_consumed",
            "motor_resets_consumed",
        ):
            minimum = 2 if field == "support_worlds" else 0
            object.__setattr__(self, field, _integer(getattr(self, field), field, minimum=minimum))
        for field in (
            "artifact_commitment",
            "sdk_commitment",
            "checkpoint_commitment",
            "dataset_commitment",
            "manifest_commitment",
            "schema_commitment",
            "ledger_commitment",
            "benchmark_definition_commitment",
        ):
            _digest(getattr(self, field), field)
        for field in ("acquisition_cost_consumed", "motor_action_cost_consumed"):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{field} must be a nonnegative finite number")
            numeric = float(value)
            if not np.isfinite(numeric) or numeric < 0.0:
                raise ValueError(f"{field} must be a nonnegative finite number")
            object.__setattr__(self, field, numeric)
        axes = tuple(self.axes)
        controls = tuple(self.controls)
        if not all(isinstance(item, Grade3BenchmarkCheck) for item in (*axes, *controls)):
            raise TypeError("axes and controls must contain Grade3BenchmarkCheck")
        if tuple(item.name for item in axes) != GRADE3_POSITIVE_AXES:
            raise ValueError("positive axes do not match the benchmark definition")
        if tuple(item.name for item in controls) != GRADE3_NEGATIVE_CONTROLS:
            raise ValueError("negative controls do not match the benchmark definition")
        if self.claim_scope != HONEST_GRADE3_CLAIM:
            raise ValueError("claim_scope cannot widen the honest Grade-3 claim")
        object.__setattr__(self, "axes", axes)
        object.__setattr__(self, "controls", controls)

    @property
    def passed(self) -> bool:
        """Logical conjunction: no axis can compensate for another failure."""

        return all(check.passed for check in (*self.axes, *self.controls))

    @property
    def noncompensatory(self) -> bool:
        return True

    @property
    def universal_certificate(self) -> bool:
        return False

    @property
    def adversarial_certificate(self) -> bool:
        return False

    def _material(self) -> dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "seed": self.seed,
            "renderer_variant": self.renderer_variant,
            "permutation_variant": self.permutation_variant,
            "support_worlds": self.support_worlds,
            "artifact_commitment": self.artifact_commitment,
            "sdk_commitment": self.sdk_commitment,
            "checkpoint_commitment": self.checkpoint_commitment,
            "dataset_commitment": self.dataset_commitment,
            "manifest_commitment": self.manifest_commitment,
            "schema_commitment": self.schema_commitment,
            "ledger_commitment": self.ledger_commitment,
            "benchmark_definition_commitment": self.benchmark_definition_commitment,
            "candidate_request_count": self.candidate_request_count,
            "support_records_consumed": self.support_records_consumed,
            "acquisition_cost_consumed": self.acquisition_cost_consumed,
            "sealed_queries_consumed": self.sealed_queries_consumed,
            "motor_action_cost_consumed": self.motor_action_cost_consumed,
            "motor_resets_consumed": self.motor_resets_consumed,
            "axes": [check.to_dict() for check in self.axes],
            "controls": [check.to_dict() for check in self.controls],
            "claim_scope": self.claim_scope,
        }

    @property
    def report_hash(self) -> str:
        return manifest_hash(self._material())

    def to_dict(self) -> dict[str, object]:
        return {
            **self._material(),
            "passed": self.passed,
            "noncompensatory": self.noncompensatory,
            "universal_certificate": self.universal_certificate,
            "adversarial_certificate": self.adversarial_certificate,
            "report_hash": self.report_hash,
        }


def grade3_benchmark_manifest(support_worlds: int = 3) -> Grade3SessionManifest:
    """Return the codebook-independent resource shape used before case creation."""

    worlds = _integer(support_worlds, "support_worlds", minimum=2)
    # Each world contributes eleven ostensive records and eight causal table
    # cells; five zero-feedback definitions are shared across worlds.
    support_records = 19 * worlds + 5
    return Grade3SessionManifest(
        GRADE3_PROTOCOL_VERSION,
        SENSOR_SCHEMA_RGB_U8,
        ACTION_SCHEMA_OPAQUE_MOTOR,
        support_records,
        8.0,
        16,
        64.0,
        20,
    )


def _check(name: str, passed: bool, evidence: object, note: str) -> Grade3BenchmarkCheck:
    return Grade3BenchmarkCheck(name, bool(passed), manifest_hash(evidence), note)


def _temporally_reversed(trace: PublicTrace) -> PublicTrace:
    if not isinstance(trace, PublicTrace) or not trace.transitions:
        raise ValueError("temporal corruption requires a nonempty PublicTrace")
    original_frames = (trace.initial, *(item.after for item in trace.transitions))
    frames = tuple(
        Observation(frame.pixels, index, False)
        for index, frame in enumerate(reversed(original_frames))
    )
    actions = tuple(item.action for item in reversed(trace.transitions))
    transitions = tuple(
        PublicTransition(frames[index], action, frames[index + 1], None)
        for index, action in enumerate(actions)
    )
    return PublicTrace(frames[0], transitions)


def _occupied_tokens(bundle: Grade3CaseBundle) -> set[int]:
    public = bundle.public
    values = {
        *public.action_space.action_codes,
        *public.hypothesis_candidates,
        *(option.probe_id for option in public.probe_options),
        *public.heldout_instruction.tokens,
        *(token for query in public.definition_queries for token in query.tokens),
    }
    for record in public.ostensive_support:
        if record.turn.utterance is not None:
            values.update(record.turn.utterance.tokens)
    return values


class _StaticMotorWorld:
    """Evaluator control exposing shape-correct but information-free RGB frames."""

    def __init__(self, shape: tuple[int, int, int]) -> None:
        self._shape = shape
        self._current = Observation(np.zeros(shape, dtype=np.uint8), 0, False)
        self.steps = 0

    def reset(self) -> Observation:
        self._current = Observation(np.zeros(self._shape, dtype=np.uint8), 0, False)
        self.steps = 0
        return self._current

    def step(self, action: Action) -> PublicTransition:
        before = self._current
        self.steps += 1
        self._current = Observation(np.zeros(self._shape, dtype=np.uint8), self.steps, False)
        return PublicTransition(before, action, self._current, None)


def _all_traces_feedback_and_outcome_free(traces: tuple[PublicTrace, ...]) -> bool:
    return all(
        not trace.has_feedback
        and all(
            transition.scalar_feedback is None and not hasattr(transition, "outcome_code")
            for transition in trace.transitions
        )
        for trace in traces
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def run_grade3_benchmark(
    seed: int = 1,
    *,
    renderer_variant: int = 0,
    permutation_variant: int = 0,
    support_worlds: int = 3,
    package_root: str | Path | None = None,
    sdk_root: str | Path | None = None,
    entrypoint: str = GRADE3_REFERENCE_ENTRYPOINT,
    timeout: float = 20.0,
) -> Grade3BenchmarkReport:
    """Execute one deterministic, single-process honest Grade-3 diagnostic."""

    selected_seed = _integer(seed, "seed")
    selected_renderer = _integer(renderer_variant, "renderer_variant")
    selected_permutation = _integer(permutation_variant, "permutation_variant")
    selected_worlds = _integer(support_worlds, "support_worlds", minimum=2)
    project = _default_project_root()
    candidate_root = (
        project / "grounding_reference_candidate" if package_root is None else Path(package_root)
    )
    selected_sdk = project / "grounding_kernel" if sdk_root is None else Path(sdk_root)

    # Literal lifecycle boundary: candidate package plus the full SDK are
    # committed before any case, codebook, surface token, or trace exists.
    artifact: Grade3ArtifactCommitment = commit_grade3_candidate(
        candidate_root, entrypoint, sdk_root=selected_sdk
    )
    generic_manifest = grade3_benchmark_manifest(selected_worlds)
    bundle_box: list[Grade3CaseBundle] = []
    runner_box: list[Grade3EvaluationRunner] = []
    materialization_request_counts: list[int] = []

    def materialize_codebook() -> str:
        if len(runner_box) != 1 or bundle_box:
            raise RuntimeError("deferred case factory must run exactly once")
        proxy = runner_box[0].candidate
        if proxy.request_count != 1 or proxy.frozen is not None:
            raise RuntimeError("case materialization must follow child begin and precede freeze")
        materialization_request_counts.append(proxy.request_count)
        bundle = build_grade3_case(
            selected_seed,
            renderer_variant=selected_renderer,
            permutation_variant=selected_permutation,
            support_worlds=selected_worlds,
        )
        if bundle.public.session_manifest != generic_manifest:
            raise RuntimeError("deferred case does not match the generic manifest")
        bundle_box.append(bundle)
        return bundle.public.case_manifest.public_dataset_commitment

    runner = Grade3EvaluationRunner(
        artifact,
        generic_manifest,
        materialize_codebook,
        timeout=timeout,
    )
    runner_box.append(runner)

    with runner:
        if len(bundle_box) != 1:
            raise RuntimeError("deferred case factory did not materialize one case")
        bundle = bundle_box[0]
        public = bundle.public
        facts = bundle.evaluator.facts
        for record in public.support_records:
            runner.support(record)
        runner.begin_acquisition()
        used_probe_ids: list[int] = []
        acquired_evidence: list[ProbeEvidence] = []
        for step_index in range(2):
            executed = runner.acquire_probe(
                public.offer(
                    step_index=step_index,
                    remaining_cost=runner.remaining_acquisition_cost,
                    exclude=used_probe_ids,
                ),
                bundle.evaluator.probes.execute,
            )
            if executed.result is None:
                raise RuntimeError("reference candidate abstained from a causal probe")
            used_probe_ids.append(executed.result.probe_id)
            acquired_evidence.append(ProbeEvidence(executed.result.probe_id, executed.result.trace))
        frozen = runner.freeze()

        true_belief = runner.trace_belief(
            TraceBeliefQuery(
                1,
                public.scope_id,
                public.problem_id,
                public.hypothesis_candidates,
                tuple(acquired_evidence),
            )
        )
        swapped_evidence = (
            ProbeEvidence(acquired_evidence[0].probe_id, acquired_evidence[1].trace),
            ProbeEvidence(acquired_evidence[1].probe_id, acquired_evidence[0].trace),
        )
        swapped_belief = runner.trace_belief(
            TraceBeliefQuery(
                2,
                public.scope_id,
                public.problem_id,
                public.hypothesis_candidates,
                swapped_evidence,
            )
        )

        heldout_trace = bundle.evaluator.probes.heldout_description_trace()
        description = runner.describe(
            TraceDescriptionQuery(
                3,
                public.scope_id,
                (ProbeEvidence(public.probe_options[0].probe_id, heldout_trace),),
            )
        )
        corrupted_trace = _temporally_reversed(heldout_trace)
        corrupted_description = runner.describe(
            TraceDescriptionQuery(
                4,
                public.scope_id,
                (ProbeEvidence(public.probe_options[0].probe_id, corrupted_trace),),
            )
        )

        instructions = (
            public.heldout_instruction,
            public.definition_queries[0],
            public.definition_queries[1],
            public.definition_queries[2],
        )
        motor_results: list[MotorEpisodeResult] = []
        motor_scores: list[bool] = []
        for query_id, instruction in enumerate(instructions, start=10):
            trial = bundle.evaluator.new_motor_trial()
            result = runner.run_motor_episode(
                query_id,
                public.scope_id,
                instruction,
                public.action_space,
                trial.agent.reset,
                trial.agent.step,
            )
            motor_results.append(result)
            motor_scores.append(trial.score_result(result))

        execution_trace = motor_results[0].execution_trace
        if execution_trace is None:
            raise RuntimeError("heldout motor execution produced no public trace")
        executed_description = runner.describe(
            TraceDescriptionQuery(
                5,
                public.scope_id,
                (ProbeEvidence(public.probe_options[0].probe_id, execution_trace),),
            )
        )

        static_trial = bundle.evaluator.new_motor_trial()
        static_world = _StaticMotorWorld(public.case_manifest.observation_shape)
        static_result = runner.run_motor_episode(
            20,
            public.scope_id,
            public.heldout_instruction,
            public.action_space,
            static_world.reset,
            static_world.step,
        )
        static_score = static_trial.score_result(static_result)

        occupied = _occupied_tokens(bundle)
        fresh_base = max(occupied, default=0) + 1_000_003
        fresh_instruction = Utterance((fresh_base,))
        remapped_instruction = Utterance((fresh_base + 1, fresh_base + 2))
        fresh_trial = bundle.evaluator.new_motor_trial()
        fresh_result = runner.run_motor_episode(
            21,
            public.scope_id,
            fresh_instruction,
            public.action_space,
            fresh_trial.agent.reset,
            fresh_trial.agent.step,
        )
        remapped_trial = bundle.evaluator.new_motor_trial()
        remapped_result = runner.run_motor_episode(
            22,
            public.scope_id,
            remapped_instruction,
            public.action_space,
            remapped_trial.agent.reset,
            remapped_trial.agent.step,
        )
        cycle_trials = []
        cycle_results = []
        for query_id, instruction in enumerate(public.definition_queries[3:], start=23):
            cycle_trial = bundle.evaluator.new_motor_trial()
            cycle_result = runner.run_motor_episode(
                query_id,
                public.scope_id,
                instruction,
                public.action_space,
                cycle_trial.agent.reset,
                cycle_trial.agent.step,
            )
            cycle_trials.append(cycle_trial)
            cycle_results.append(cycle_result)

        checkpoint_after_queries = runner.candidate.assert_frozen()
        all_public_traces = (
            *(record.trace for record in public.support_records),
            *(item.trace for item in acquired_evidence),
            heldout_trace,
            corrupted_trace,
            *(
                trace
                for result in (
                    *motor_results,
                    static_result,
                    fresh_result,
                    remapped_result,
                    *cycle_results,
                )
                for trace in (
                    *result.completed_probes,
                    *((result.execution_trace,) if result.execution_trace is not None else ()),
                )
            ),
        )

        heldout_is_fourth = (
            len(facts.demonstrated_factorials) == 3
            and public.heldout_instruction == facts.heldout_factorial
            and public.heldout_instruction not in facts.demonstrated_factorials
        )
        true_pattern = dict(facts.hypothesis_patterns)[facts.true_hypothesis_id]
        swapped_unknown = (
            not swapped_belief.candidate_probabilities and swapped_belief.unknown_probability == 1.0
        )
        swapped_equivalent = true_pattern[0] is true_pattern[1] and swapped_belief == true_belief
        axes = (
            _check(
                "artifact_before_dataset_materialization",
                materialization_request_counts == [1]
                and runner.session.commitments["artifact"] == artifact.digest
                and runner.session.commitments["codebook"]
                == public.case_manifest.public_dataset_commitment,
                {
                    "artifact": artifact.digest,
                    "materialization_request_counts": materialization_request_counts,
                    "dataset": public.case_manifest.public_dataset_commitment,
                },
                "Artifact/SDK commitment preceded the sole post-begin case factory call.",
            ),
            _check(
                "single_persistent_checkpoint",
                checkpoint_after_queries == frozen.checkpoint_commitment,
                {
                    "artifact": frozen.artifact_commitment,
                    "sdk": frozen.sdk_commitment,
                    "checkpoint": frozen.checkpoint_commitment,
                    "requests": runner.candidate.request_count,
                },
                "Every support, probe, and query used one persistent frozen process.",
            ),
            _check(
                "active_causal_belief",
                true_belief.candidate_probabilities == ((facts.true_hypothesis_id, 1.0),)
                and true_belief.unknown_probability == 0.0,
                true_belief,
                "Two evaluator-executed probes identify the opaque causal hypothesis.",
            ),
            _check(
                "heldout_factorial_action",
                heldout_is_fourth and motor_results[0].completed and motor_scores[0],
                {
                    "result": motor_results[0],
                    "score": motor_scores[0],
                    "heldout_is_fourth": heldout_is_fourth,
                },
                "The unshown fourth role×action cell is executed and externally scored.",
            ),
            _check(
                "heldout_reverse_description",
                description.utterance == public.heldout_instruction
                and description.unknown_probability == 0.0
                and executed_description.utterance == public.heldout_instruction
                and executed_description.unknown_probability == 0.0,
                {
                    "independent_trace": description,
                    "candidate_execution": executed_description,
                },
                "Both an independent trace and the candidate's own execution map back to the withheld utterance.",
            ),
            *(
                _check(
                    name,
                    result.completed and score,
                    {"result": result, "score": score},
                    "The grounded zero-feedback definition resolves and executes externally.",
                )
                for name, result, score in zip(
                    GRADE3_POSITIVE_AXES[5:8],
                    motor_results[1:],
                    motor_scores[1:],
                    strict=True,
                )
            ),
        )
        controls = (
            _check(
                "probe_association_shuffle_safe",
                swapped_unknown or swapped_equivalent,
                {
                    "belief": swapped_belief,
                    "true_pattern": true_pattern,
                    "unknown": swapped_unknown,
                    "operationally_equivalent": swapped_equivalent,
                },
                "A consequential probe-ID shuffle is unknown; an observationally identical swap remains equivariant.",
            ),
            _check(
                "temporal_corruption_rejected",
                corrupted_description.utterance != public.heldout_instruction
                and corrupted_description.unknown_probability > 0.0,
                {
                    "corrupted_trace": corrupted_trace,
                    "decision": corrupted_description,
                },
                "A reversed temporal/action sequence is not named as the heldout process.",
            ),
            _check(
                "static_sensor_not_operational",
                not static_result.completed and not static_score and static_world.steps == 0,
                {
                    "result": static_result,
                    "score": static_score,
                    "world_steps": static_world.steps,
                },
                "Shape-correct static pixels neither complete nor pass the external scorer.",
            ),
            _check(
                "fresh_symbol_abstention",
                fresh_result.abstained
                and not fresh_result.completed
                and not fresh_trial.score_result(fresh_result),
                fresh_result,
                "A fresh ungrounded surface symbol triggers explicit abstention.",
            ),
            _check(
                "unannounced_remap_abstention",
                remapped_result.abstained
                and not remapped_result.completed
                and not remapped_trial.score_result(remapped_result),
                remapped_result,
                "An unannounced fresh remapping is rejected rather than guessed.",
            ),
            _check(
                "ungrounded_cycle_abstention",
                len(cycle_results) == 2
                and all(
                    result.abstained and not result.completed and not trial.score_result(result)
                    for trial, result in zip(cycle_trials, cycle_results, strict=True)
                ),
                cycle_results,
                "Both symbols in a mutually recursive definition without causal anchor remain unknown.",
            ),
            _check(
                "feedback_outcome_free_trajectory_channel",
                _all_traces_feedback_and_outcome_free(all_public_traces),
                {
                    "trace_hashes": [manifest_hash(trace) for trace in all_public_traces],
                    "trace_count": len(all_public_traces),
                    "support_corrections_are_turn_only": all(
                        transition.scalar_feedback is None
                        for record in public.ostensive_support
                        for transition in record.trace.transitions
                    ),
                },
                "Trajectory evidence omits reward and evaluator outcome codes; ostensive corrections remain support-turn-only.",
            ),
        )

        ledger = runner.complete()
        request_count = runner.candidate.request_count
        definition_commitment = manifest_hash(
            {
                "protocol": GRADE3_BENCHMARK_PROTOCOL,
                "manifest": asdict(generic_manifest),
                "positive_axes": GRADE3_POSITIVE_AXES,
                "negative_controls": GRADE3_NEGATIVE_CONTROLS,
                "claim_scope": HONEST_GRADE3_CLAIM,
            }
        )
        report = Grade3BenchmarkReport(
            GRADE3_BENCHMARK_PROTOCOL,
            selected_seed,
            selected_renderer,
            selected_permutation,
            selected_worlds,
            artifact.digest,
            artifact.sdk_commitment,
            frozen.checkpoint_commitment,
            public.case_manifest.public_dataset_commitment,
            ledger.manifest_hash,
            ledger.wire_schema_hash,
            ledger.ledger_hash,
            definition_commitment,
            request_count,
            ledger.support_records_used,
            ledger.acquisition_cost_used,
            ledger.sealed_queries_used,
            ledger.motor_action_cost_used,
            ledger.motor_resets_used,
            axes,
            controls,
        )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    """Run the diagnostic and emit its canonical JSON-friendly report."""

    parser = argparse.ArgumentParser(
        description="Run the honest-reference Grade-3 grounding diagnostic."
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--renderer-variant", type=int, default=0)
    parser.add_argument("--permutation-variant", type=int, default=0)
    parser.add_argument("--support-worlds", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=20.0)
    arguments = parser.parse_args(None if argv is None else list(argv))
    report = run_grade3_benchmark(
        arguments.seed,
        renderer_variant=arguments.renderer_variant,
        permutation_variant=arguments.permutation_variant,
        support_worlds=arguments.support_worlds,
        timeout=arguments.timeout,
    )
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
    return 0 if report.passed else 1


__all__ = [
    "GRADE3_BENCHMARK_PROTOCOL",
    "GRADE3_NEGATIVE_CONTROLS",
    "GRADE3_POSITIVE_AXES",
    "GRADE3_REFERENCE_ENTRYPOINT",
    "HONEST_GRADE3_CLAIM",
    "Grade3BenchmarkCheck",
    "Grade3BenchmarkReport",
    "grade3_benchmark_manifest",
    "main",
    "run_grade3_benchmark",
]


if __name__ == "__main__":  # pragma: no cover - exercised through ``python -m``
    raise SystemExit(main())
