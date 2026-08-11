from __future__ import annotations

from grounding_kernel.process_language import (
    build_process_language_case,
    run_process_language_block,
)


def test_process_language_uses_public_groundings_not_coordinate_macros() -> None:
    case = build_process_language_case(3, 0)
    target = case.referents[case.held_out]
    role = target.meaning_for(case.schema.target_type_id)

    assert role is not None
    assert not hasattr(role.value, "targets")
    assert role.value.binder_digest == role.value.binder.manifest.digest()
    assert case.held_out not in {demo.tokens for demo in case.training}


def test_all_language_axes_execute_and_raw_trace_description_roundtrips() -> None:
    report = run_process_language_block(3, 0)

    assert set(report.results) == {
        "description_to_action",
        "trace_to_description",
        "factorial_composition",
        "lexicon_permutation_equivariance",
        "proof_grounded_symbolic_theft",
    }
    assert all(passed and answered for passed, answered, _detail in report.results.values())
    assert report.language_fresh_unknown
    assert not report.lookup_leaked
    assert report.target_only_prediction is None
    assert not report.definition_leaf_deletion_leaked
    assert not report.definition_leaf_swap_leaked
    trace_detail = report.results["trace_to_description"][2]
    assert trace_detail["independent_source_trace"]
    assert trace_detail["recognized_expected_referent"]
    assert trace_detail["fresh_world_reexecution"]


def test_definition_leaf_deletion_and_swap_are_explicit_kill_controls() -> None:
    reports = tuple(run_process_language_block(29, block) for block in range(3))

    assert all(not report.definition_leaf_deletion_leaked for report in reports)
    assert all(not report.definition_leaf_swap_leaked for report in reports)
    for report in reports:
        detail = report.results["proof_grounded_symbolic_theft"][2]
        assert detail["deleted_leaf_unknown"]
        assert detail["swapped_leaf_changed_behavior"]


def test_reference_behavior_transfers_over_multiple_unseen_world_variants() -> None:
    reports = tuple(run_process_language_block(17, block) for block in range(3))

    assert all(
        all(passed for passed, _answered, _detail in report.results.values())
        for report in reports
    )
