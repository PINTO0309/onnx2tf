from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_affine_concat_split_recovery_orchestration,
)
from onnx2tf.tflite_builder.passes.terminal_affine_concat_split_recovery_orchestration import (
    TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS,
    TerminalAffineConcatSplitRecoveryContext,
    build_terminal_affine_concat_split_recovery_invocations,
    run_terminal_affine_concat_split_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_AFFINE_CONCAT_SPLIT = "_run_terminal_affine_concat_split_recovery_sequence"


def _lowerer_and_helper() -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_AFFINE_CONCAT_SPLIT
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> TerminalAffineConcatSplitRecoveryContext:
    model_ir = ModelIR("terminal_affine_concat_split_recovery_test")
    return TerminalAffineConcatSplitRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: terminal_affine_concat_split_recovery_orchestration.RecoveryInvocation,
    context: TerminalAffineConcatSplitRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_terminal_affine_concat_split_recovery_is_straight_line() -> None:
    _, helper = _lowerer_and_helper()
    control_flow_nodes = (
        ast.AsyncFor,
        ast.AsyncWith,
        ast.For,
        ast.If,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
    )

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 4
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert not any(isinstance(node, control_flow_nodes) for node in ast.walk(helper))
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )

    called_names = {
        node.func.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    loaded_data_names = {
        node.id
        for node in ast.walk(helper)
        if isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id not in called_names
    }
    assert loaded_data_names == {"terminal_affine_concat_split_recovery_context"}


def test_terminal_affine_concat_split_preserves_all_call_contracts() -> None:
    context = _context()
    invocations = build_terminal_affine_concat_split_recovery_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        "_optimize_fold_mul_add_mul_affine_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_mul_add_const_prepost_nhwc_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_concat_mul_add_transpose_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains": (
            ("model_ir",),
            {},
        ),
        "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_unary_split_concat_single_post_nchw": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_split_channelwise_tail_to_single_post_nchw": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_sanitize_probable_nhwc_axis_sensitive_ops": (("model_ir",), {}),
    }


def test_terminal_affine_concat_split_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_terminal_affine_concat_split_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
            for node in ast.walk(statement)
        )
    ]

    assert len(invocation_indexes) == 2
    observed: list[tuple[str, str]] = []
    for position, index in enumerate(invocation_indexes):
        if position == 0:
            invocation = lowerer.body[index]
            assert isinstance(invocation, ast.Assign)
            assert len(invocation.targets) == 1
            assert isinstance(invocation.targets[0], ast.Name)
            assert invocation.targets[0].id == "pre_terminal_affine_results"
            recovery_count = lowerer.body[index - 1]
            assert isinstance(recovery_count, ast.Assign)
            assert len(recovery_count.targets) == 1
            assert isinstance(recovery_count.targets[0], ast.Name)
            assert recovery_count.targets[0].id == "pre_terminal_affine_tensor_count"
            previous = lowerer.body[index - 2]
            recovery_summary = lowerer.body[index + 1]
            assert isinstance(recovery_summary, ast.Assign)
            assert len(recovery_summary.targets) == 1
            assert isinstance(recovery_summary.targets[0], ast.Name)
            assert recovery_summary.targets[0].id == "_pre_terminal_affine_stats"
            assert isinstance(recovery_summary.value, ast.Call)
            assert isinstance(recovery_summary.value.func, ast.Name)
            assert recovery_summary.value.func.id == (
                "summarize_terminal_affine_concat_split_mutations"
            )
            pre_add_count = lowerer.body[index + 2]
            assert isinstance(pre_add_count, ast.Assign)
            assert len(pre_add_count.targets) == 1
            assert isinstance(pre_add_count.targets[0], ast.Name)
            assert pre_add_count.targets[0].id == (
                "pre_terminal_pre_add_tensor_count"
            )
            following = lowerer.body[index + 3]
        else:
            invocation = lowerer.body[index]
            assert isinstance(invocation, ast.Assign)
            assert len(invocation.targets) == 1
            assert isinstance(invocation.targets[0], ast.Name)
            assert invocation.targets[0].id == "terminal_affine_results"
            previous = lowerer.body[index - 2]
            following = lowerer.body[index + 2]
        assert isinstance(previous, (ast.Expr, ast.Assign))
        if position == 1:
            assert isinstance(previous, ast.Assign)
            assert len(previous.targets) == 1
            assert isinstance(previous.targets[0], ast.Name)
            assert previous.targets[0].id == (
                "_pre_terminal_affine_slice_pad_concat_stats"
            )
        assert isinstance(previous.value, ast.Call)
        assert isinstance(previous.value.func, ast.Name)
        assert isinstance(following, (ast.Expr, ast.Assign))
        if position == 0:
            assert isinstance(following, ast.Assign)
            assert len(following.targets) == 1
            assert isinstance(following.targets[0], ast.Name)
            assert following.targets[0].id == "_pre_terminal_pre_add_stats"
            assert isinstance(following.value, ast.Dict)
            following_call = following.value.values[0]
        else:
            following_call = following.value
        assert isinstance(following_call, ast.Call)
        assert isinstance(following_call.func, ast.Name)
        observed.append((previous.value.func.id, following_call.func.id))

    assert observed == [
        (
            "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
            "_optimize_transpose_pre_add_nhwc_chains",
        ),
        (
            "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
            "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
        ),
    ]


def test_terminal_affine_concat_split_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_terminal_affine_concat_split_recovery"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "terminal_affine_concat_split_recovery_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "terminal_affine_concat_split_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_terminal_affine_concat_split_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    probe_steps = build_terminal_affine_concat_split_recovery_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(
                terminal_affine_concat_split_recovery_orchestration
            ).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            terminal_affine_concat_split_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_terminal_affine_concat_split_recovery(context)

    assert events == list(TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS)


def test_terminal_affine_returns_and_summarizes_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    result_keys = (
        ("optimized_fold_mul_add_mul_affine_chains",),
        ("optimized_transpose_mul_add_const_prepost_nhwc_chains",),
        ("optimized_concat_mul_add_transpose_nhwc_bridge_chains",),
        ("optimized_concat_mul_add_transpose_add_nhwc_bridge_chains",),
        ("optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",),
        ("optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains",),
        ("optimized_singleton_gate_conv_concat_nhwc_bridge_blocks",),
        ("optimized_transpose_unary_split_concat_single_post_nchw",),
        ("optimized_transpose_split_channelwise_tail_to_single_post_nchw",),
        ("optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw",),
        (
            "sanitized_probable_nhwc_axis_sensitive_ops",
            "inserted_probable_nhwc_terminal_transposes",
        ),
    )
    value = 1
    expected_results = []
    expected_summary: dict[str, int] = {}
    for keys in result_keys:
        result: dict[str, int] = {}
        for key in keys:
            result[key] = value
            expected_summary[key] = value
            value += 1
        expected_results.append(result)
    expected_tuple = tuple(expected_results)

    def return_results(invocations, *, expected_pass_ids, phase_name):
        assert tuple(
            invocation.pass_id for invocation in invocations
        ) == TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS
        assert (
            tuple(expected_pass_ids)
            == TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS
        )
        assert phase_name == "terminal affine/concat/split recovery"
        return expected_tuple

    monkeypatch.setattr(
        terminal_affine_concat_split_recovery_orchestration,
        "run_recovery_invocations",
        return_results,
    )

    results = run_terminal_affine_concat_split_recovery(context)
    summarize = getattr(
        terminal_affine_concat_split_recovery_orchestration,
        "summarize_terminal_affine_concat_split_mutations",
    )
    summary = summarize(results, pruned_unused_tensors=13)

    assert results == expected_tuple
    assert summary == {**expected_summary, "pruned_unused_tensors": 13}
    with pytest.raises(
        ValueError,
        match=r"terminal affine mutation summary expected 11 pass results",
    ):
        summarize((), pruned_unused_tensors=0)

    _, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)


def test_lowerer_captures_second_terminal_affine_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    target_names = (
        "terminal_affine_tensor_count",
        "terminal_affine_results",
        "_terminal_affine_stats",
    )
    assignment_indices: dict[str, int] = {}
    assignments: dict[str, ast.expr] = {}
    for index, statement in enumerate(lowerer.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in target_names:
            assignment_indices[target.id] = index
            assignments[target.id] = statement.value

    first_index = min(assignment_indices.values())
    assert assignment_indices == {
        target_names[0]: first_index,
        target_names[1]: first_index + 1,
        target_names[2]: first_index + 2,
    }
    result_call = assignments[target_names[1]]
    assert isinstance(result_call, ast.Call)
    assert isinstance(result_call.func, ast.Name)
    assert result_call.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
    summary_call = assignments[target_names[2]]
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == (
        "summarize_terminal_affine_concat_split_mutations"
    )
    assert {keyword.arg for keyword in summary_call.keywords} == {
        "pruned_unused_tensors"
    }

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_pre_terminal_affine_slice_pad_concat_stats"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    following = lowerer.body[first_index + 3]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_terminal_slice_pad_concat_stats"

    recovery_statements = [
        statement
        for statement in lowerer.body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
            for node in ast.walk(statement)
        )
    ]
    assert len(recovery_statements) == 2
    assert isinstance(recovery_statements[0], ast.Assign)
    assert len(recovery_statements[0].targets) == 1
    assert isinstance(recovery_statements[0].targets[0], ast.Name)
    assert recovery_statements[0].targets[0].id == "pre_terminal_affine_results"
    assert recovery_statements[1] is lowerer.body[first_index + 1]


def test_lowerer_captures_first_terminal_affine_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    target_names = (
        "pre_terminal_affine_tensor_count",
        "pre_terminal_affine_results",
        "_pre_terminal_affine_stats",
    )
    assignment_indices: dict[str, int] = {}
    assignments: dict[str, ast.expr] = {}
    for index, statement in enumerate(lowerer.body):
        if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
            continue
        target = statement.targets[0]
        if isinstance(target, ast.Name) and target.id in target_names:
            assignment_indices[target.id] = index
            assignments[target.id] = statement.value

    first_index = min(assignment_indices.values())
    assert assignment_indices == {
        target_names[0]: first_index,
        target_names[1]: first_index + 1,
        target_names[2]: first_index + 2,
    }
    tensor_count = assignments[target_names[0]]
    assert isinstance(tensor_count, ast.Call)
    assert isinstance(tensor_count.func, ast.Name)
    assert tensor_count.func.id == "len"
    result_call = assignments[target_names[1]]
    assert isinstance(result_call, ast.Call)
    assert isinstance(result_call.func, ast.Name)
    assert result_call.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
    assert result_call.args == []
    assert result_call.keywords == []
    summary_call = assignments[target_names[2]]
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == (
        "summarize_terminal_affine_concat_split_mutations"
    )
    assert len(summary_call.args) == 1
    assert isinstance(summary_call.args[0], ast.Name)
    assert summary_call.args[0].id == "pre_terminal_affine_results"
    assert {keyword.arg for keyword in summary_call.keywords} == {
        "pruned_unused_tensors"
    }

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_pre_terminal_affine_instancenorm_dualstats_stats"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    following = lowerer.body[first_index + 3]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "pre_terminal_pre_add_tensor_count"

    recovery_statements = [
        statement
        for statement in lowerer.body
        if any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == TERMINAL_AFFINE_CONCAT_SPLIT
            for node in ast.walk(statement)
        )
    ]
    assert len(recovery_statements) == 2
    assert recovery_statements[0] is lowerer.body[first_index + 1]
    assert isinstance(recovery_statements[1], ast.Assign)
    assert len(recovery_statements[1].targets) == 1
    assert isinstance(recovery_statements[1].targets[0], ast.Name)
    assert recovery_statements[1].targets[0].id == "terminal_affine_results"


def test_pre_terminal_affine_dualstats_captures_complete_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    affine_count_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "pre_terminal_affine_tensor_count"
    )
    invocation = lowerer.body[affine_count_index - 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_pre_terminal_affine_instancenorm_dualstats_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"

    previous = lowerer.body[affine_count_index - 2]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_pre_terminal_affine_instancenorm_residual_mul_concat_stats"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
    )
    following = lowerer.body[affine_count_index]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "pre_terminal_affine_tensor_count"

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 3
    assert isinstance(direct_statements[0], ast.Assign)
    assert isinstance(direct_statements[0].targets[0], ast.Name)
    assert direct_statements[0].targets[0].id == (
        "_terminal_instancenorm_dualstats_stats"
    )
    assert isinstance(direct_statements[1], ast.Assign)
    assert isinstance(direct_statements[1].targets[0], ast.Name)
    assert direct_statements[1].targets[0].id == (
        "_very_late_instancenorm_dualstats_stats"
    )
    assert direct_statements[2] is invocation


def test_pre_terminal_affine_residual_mul_concat_captures_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    dualstats_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_pre_terminal_affine_instancenorm_dualstats_stats"
    )
    invocation = lowerer.body[dualstats_index - 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_pre_terminal_affine_instancenorm_residual_mul_concat_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"

    previous = lowerer.body[dualstats_index - 2]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == (
        "_pre_terminal_affine_instancenorm_post_bias_stats"
    )
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    following = lowerer.body[dualstats_index]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_pre_terminal_affine_instancenorm_dualstats_stats"
    )

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 3
    assert isinstance(direct_statements[0], ast.Assign)
    assert isinstance(direct_statements[0].targets[0], ast.Name)
    assert direct_statements[0].targets[0].id == (
        "_terminal_instancenorm_residual_mul_concat_stats"
    )
    assert isinstance(direct_statements[1], ast.Assign)
    assert isinstance(direct_statements[1].targets[0], ast.Name)
    assert direct_statements[1].targets[0].id == (
        "_very_late_instancenorm_residual_mul_concat_stats"
    )
    assert direct_statements[2] is invocation


def test_pre_terminal_affine_post_bias_captures_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    residual_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_pre_terminal_affine_instancenorm_residual_mul_concat_stats"
    )
    invocation = lowerer.body[residual_index - 1]
    assert isinstance(invocation, ast.Assign)
    assert len(invocation.targets) == 1
    assert isinstance(invocation.targets[0], ast.Name)
    assert invocation.targets[0].id == (
        "_pre_terminal_affine_instancenorm_post_bias_stats"
    )
    assert isinstance(invocation.value, ast.Call)
    assert isinstance(invocation.value.func, ast.Name)
    assert invocation.value.func.id == (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    assert len(invocation.value.args) == 1
    assert isinstance(invocation.value.args[0], ast.Name)
    assert invocation.value.args[0].id == "model_ir"
    assert len(invocation.value.keywords) == 1
    layout_keyword = invocation.value.keywords[0]
    assert layout_keyword.arg == "layout_state"
    assert isinstance(layout_keyword.value, ast.Attribute)
    assert isinstance(layout_keyword.value.value, ast.Name)
    assert layout_keyword.value.value.id == "session"
    assert layout_keyword.value.attr == "layout_state"

    previous = lowerer.body[residual_index - 2]
    assert isinstance(previous, ast.If)
    following = lowerer.body[residual_index]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_pre_terminal_affine_instancenorm_residual_mul_concat_stats"
    )

    direct_statements = [
        statement
        for statement in lowerer.body
        if isinstance(statement, (ast.Expr, ast.Assign))
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id
            == "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
            for node in ast.walk(statement)
        )
    ]
    assert len(direct_statements) == 4
    assert isinstance(direct_statements[0], ast.Assign)
    assert isinstance(direct_statements[0].targets[0], ast.Name)
    assert direct_statements[0].targets[0].id == (
        "_terminal_instancenorm_post_bias_stats"
    )
    assert isinstance(direct_statements[1], ast.Assign)
    assert isinstance(direct_statements[1].targets[0], ast.Name)
    assert direct_statements[1].targets[0].id == (
        "_very_late_instancenorm_post_bias_stats"
    )
    assert direct_statements[2] is invocation
    assert isinstance(direct_statements[3], ast.Assign)
    assert len(direct_statements[3].targets) == 1
    assert isinstance(direct_statements[3].targets[0], ast.Name)
    assert direct_statements[3].targets[0].id == (
        "_absolute_final_instancenorm_post_bias_stats"
    )


def test_terminal_affine_concat_split_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "terminal_affine_concat_split_recovery_orchestration.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
            for alias in node.names
        )
        for node in tree.body
    )
