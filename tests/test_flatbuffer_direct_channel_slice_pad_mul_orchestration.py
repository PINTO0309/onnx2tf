from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import channel_slice_pad_mul_orchestration
from onnx2tf.tflite_builder.passes.channel_slice_pad_mul_orchestration import (
    CHANNEL_SLICE_PAD_MUL_PASS_IDS,
    ChannelSlicePadMulContext,
    build_channel_slice_pad_mul_invocations,
    run_channel_slice_pad_mul,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
CHANNEL_SLICE_PAD_MUL = "_run_channel_slice_pad_mul_layout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == CHANNEL_SLICE_PAD_MUL
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> ChannelSlicePadMulContext:
    model_ir = ModelIR("channel_slice_pad_mul_test")
    return ChannelSlicePadMulContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: channel_slice_pad_mul_orchestration.RecoveryInvocation,
    context: ChannelSlicePadMulContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_channel_slice_pad_mul_is_a_straight_line_delegate() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 1
    assert not any(
        isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
                ast.If,
                ast.Match,
                ast.Try,
                ast.While,
                ast.With,
            ),
        )
        for node in ast.walk(helper)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )


def test_channel_slice_pad_mul_preserves_both_cleanup_contracts() -> None:
    context = _context()
    invocations = build_channel_slice_pad_mul_invocations(context)

    assert tuple(step.pass_id for step in invocations) == CHANNEL_SLICE_PAD_MUL_PASS_IDS
    expected_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {pass_id: expected_contract for pass_id in CHANNEL_SLICE_PAD_MUL_PASS_IDS}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_channel_slice_pad_mul_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_channel_slice_pad_mul_preserves_direct_and_callback_invocations() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHANNEL_SLICE_PAD_MUL
    ]
    assert len(direct_invocations) == 1
    assert direct_invocations[0].args == []
    assert direct_invocations[0].keywords == []

    terminal_context = next(
        statement.value
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "terminal_slice_concat_recovery_context"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Call)
    )
    callback_keyword = next(
        keyword
        for keyword in terminal_context.keywords
        if keyword.arg == "channel_slice_pad_mul_cluster"
    )
    assert isinstance(callback_keyword.value, ast.Name)
    assert callback_keyword.value.id == CHANNEL_SLICE_PAD_MUL


def test_channel_slice_pad_mul_preserves_direct_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "channel_slice_pad_mul_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CHANNEL_SLICE_PAD_MUL
    )

    pre_add_stats = lowerer.body[invocation_index - 1]
    pre_add_count = lowerer.body[invocation_index - 2]
    summary = lowerer.body[invocation_index + 1]
    following = lowerer.body[invocation_index + 2]
    assert isinstance(pre_add_count, ast.Assign)
    assert len(pre_add_count.targets) == 1
    assert isinstance(pre_add_count.targets[0], ast.Name)
    assert pre_add_count.targets[0].id == "pre_terminal_pre_add_tensor_count"
    assert isinstance(pre_add_stats, ast.Assign)
    assert len(pre_add_stats.targets) == 1
    assert isinstance(pre_add_stats.targets[0], ast.Name)
    assert pre_add_stats.targets[0].id == "_pre_terminal_pre_add_stats"
    assert isinstance(pre_add_stats.value, ast.Dict)
    pre_add_owner = pre_add_stats.value.values[0]
    assert isinstance(pre_add_owner, ast.Call)
    assert isinstance(pre_add_owner.func, ast.Name)
    assert pre_add_owner.func.id == "_optimize_transpose_pre_add_nhwc_chains"
    assert isinstance(summary, ast.Assign)
    assert len(summary.targets) == 1
    assert isinstance(summary.targets[0], ast.Name)
    assert summary.targets[0].id == "_pre_terminal_channel_slice_pad_mul_stats"
    assert isinstance(summary.value, ast.Call)
    assert isinstance(summary.value.func, ast.Name)
    assert summary.value.func.id == "summarize_channel_slice_pad_mul_mutations"
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_pre_terminal_affine_post_add_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert (
        following.value.func.id
        == "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )


def test_channel_slice_pad_mul_preserves_stable_callback_boundary() -> None:
    assert TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[0] == CHANNEL_SLICE_PAD_MUL
    assert TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[1] == (
        "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )


def test_channel_slice_pad_mul_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_channel_slice_pad_mul"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "channel_slice_pad_mul_context",
    )
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "channel_slice_pad_mul_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_channel_slice_pad_mul_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in CHANNEL_SLICE_PAD_MUL_PASS_IDS:
        monkeypatch.setattr(
            channel_slice_pad_mul_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_channel_slice_pad_mul(context)

    assert events == list(CHANNEL_SLICE_PAD_MUL_PASS_IDS)


def test_channel_slice_pad_mul_children_have_fixed_mutation_schemas() -> None:
    context = _context()

    results = tuple(
        invocation.run()
        for invocation in build_channel_slice_pad_mul_invocations(context)
    )

    assert results == (
        {
            "optimized_transpose_channel_slice_dual_add_bridges_strict": 0,
            "optimized_transpose_slice_muladd_conv_mergeadd_strict": 0,
            "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": 0,
        },
        {
            "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": 0,
        },
    )


def test_channel_slice_pad_mul_returns_and_summarizes_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    channel_slice_result = {
        "optimized_transpose_channel_slice_dual_add_bridges_strict": 2,
        "optimized_transpose_slice_muladd_conv_mergeadd_strict": 3,
        "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": 4,
    }
    pad_mul_result = {
        "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": 5,
    }
    expected_results = (channel_slice_result, pad_mul_result)

    def return_results(invocations, *, expected_pass_ids, phase_name):
        assert tuple(
            invocation.pass_id for invocation in invocations
        ) == CHANNEL_SLICE_PAD_MUL_PASS_IDS
        assert tuple(expected_pass_ids) == CHANNEL_SLICE_PAD_MUL_PASS_IDS
        assert phase_name == "channel-slice/pad-mul"
        return expected_results

    monkeypatch.setattr(
        channel_slice_pad_mul_orchestration,
        "run_recovery_invocations",
        return_results,
    )

    results = run_channel_slice_pad_mul(context)
    assert results == expected_results
    summarize = getattr(
        channel_slice_pad_mul_orchestration,
        "summarize_channel_slice_pad_mul_mutations",
    )
    assert summarize(results) == {
        **channel_slice_result,
        **pad_mul_result,
    }
    with pytest.raises(
        ValueError,
        match=r"channel-slice/pad-Mul mutation summary expected 2 pass results",
    ):
        summarize(())


def test_lowerer_captures_channel_slice_pad_mul_mutation_evidence() -> None:
    lowerer, helper = _lowerer_and_helper()
    target_names = (
        "channel_slice_pad_mul_results",
        "_pre_terminal_channel_slice_pad_mul_stats",
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
    }
    result_call = assignments[target_names[0]]
    assert isinstance(result_call, ast.Call)
    assert isinstance(result_call.func, ast.Name)
    assert result_call.func.id == CHANNEL_SLICE_PAD_MUL
    assert result_call.args == []
    assert result_call.keywords == []
    summary_call = assignments[target_names[1]]
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == "summarize_channel_slice_pad_mul_mutations"
    assert len(summary_call.args) == 1
    assert isinstance(summary_call.args[0], ast.Name)
    assert summary_call.args[0].id == "channel_slice_pad_mul_results"

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_pre_terminal_pre_add_stats"
    assert isinstance(previous.value, ast.Dict)
    owner_call = previous.value.values[0]
    assert isinstance(owner_call, ast.Call)
    assert isinstance(owner_call.func, ast.Name)
    assert owner_call.func.id == "_optimize_transpose_pre_add_nhwc_chains"
    tensor_count = lowerer.body[first_index - 2]
    assert isinstance(tensor_count, ast.Assign)
    assert len(tensor_count.targets) == 1
    assert isinstance(tensor_count.targets[0], ast.Name)
    assert tensor_count.targets[0].id == "pre_terminal_pre_add_tensor_count"
    following = lowerer.body[first_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_pre_terminal_affine_post_add_stats"

    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)
    assert isinstance(helper.body[0].value.func, ast.Name)
    assert helper.body[0].value.func.id == "run_channel_slice_pad_mul"


def test_pre_terminal_pre_add_captures_zero_rewrite_pruning_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    target_names = (
        "pre_terminal_pre_add_tensor_count",
        "_pre_terminal_pre_add_stats",
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
    }
    tensor_count = assignments[target_names[0]]
    assert isinstance(tensor_count, ast.Call)
    assert isinstance(tensor_count.func, ast.Name)
    assert tensor_count.func.id == "len"
    assert len(tensor_count.args) == 1
    assert isinstance(tensor_count.args[0], ast.Attribute)
    assert isinstance(tensor_count.args[0].value, ast.Name)
    assert tensor_count.args[0].value.id == "model_ir"
    assert tensor_count.args[0].attr == "tensors"

    summary = assignments[target_names[1]]
    assert isinstance(summary, ast.Dict)
    assert len(summary.keys) == 2
    assert summary.keys[0] is None
    owner_call = summary.values[0]
    assert isinstance(owner_call, ast.Call)
    assert isinstance(owner_call.func, ast.Name)
    assert owner_call.func.id == "_optimize_transpose_pre_add_nhwc_chains"
    assert len(owner_call.args) == 1
    assert isinstance(owner_call.args[0], ast.Name)
    assert owner_call.args[0].id == "model_ir"
    assert len(owner_call.keywords) == 1
    assert owner_call.keywords[0].arg == "layout_state"
    prune_key = summary.keys[1]
    assert isinstance(prune_key, ast.Constant)
    assert prune_key.value == "pruned_unused_tensors"
    prune_call = summary.values[1]
    assert isinstance(prune_call, ast.Call)
    assert isinstance(prune_call.func, ast.Name)
    assert prune_call.func.id == "max"

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.Expr)
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_run_terminal_affine_concat_split_recovery_sequence"
    )
    following = lowerer.body[first_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "channel_slice_pad_mul_results"

    production_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_optimize_transpose_pre_add_nhwc_chains"
    ]
    assert len(production_calls) == 1


def test_channel_slice_pad_mul_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "channel_slice_pad_mul_orchestration.py"
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
