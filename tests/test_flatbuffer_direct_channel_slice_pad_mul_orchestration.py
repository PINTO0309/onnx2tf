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
CHANNEL_SLICE_PAD_MUL_SUMMARY = "run_channel_slice_pad_mul_summary"
PRE_TERMINAL_PRE_ADD = "run_pre_terminal_pre_add_cleanup"
PRE_TERMINAL_CLEANUP_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_cleanup_orchestration.py"
)
PRE_TERMINAL_CLEANUP = "run_pre_terminal_cleanup"
PRE_TERMINAL_CLEANUP_RESULT = "_pre_terminal_cleanup_results"
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_affine_slice_spp_orchestration.py"
)
OUTER_OWNER = "run_pre_terminal_affine_slice_spp_cleanup"
OUTER_RESULT = "_pre_terminal_affine_slice_spp_results"


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


def _pre_terminal_cleanup_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(PRE_TERMINAL_CLEANUP_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == PRE_TERMINAL_CLEANUP
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _outer_calls() -> list[ast.Call]:
    tree = ast.parse(OUTER_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == OUTER_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == PRE_TERMINAL_CLEANUP
    ]


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
    assert direct_invocations == []
    summary_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == CHANNEL_SLICE_PAD_MUL_SUMMARY
    ]
    assert summary_invocations == []
    composite_invocations = _pre_terminal_cleanup_calls(
        CHANNEL_SLICE_PAD_MUL_SUMMARY
    )
    assert len(composite_invocations) == 1
    assert [ast.unparse(arg) for arg in composite_invocations[0].args] == [
        "context"
    ]
    assert composite_invocations[0].keywords == []

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
        and statement.targets[0].id == OUTER_RESULT
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == OUTER_OWNER
    )

    predecessor = lowerer.body[invocation_index - 1]
    summary = lowerer.body[invocation_index]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(predecessor, ast.If)
    assert isinstance(summary, ast.Assign)
    assert isinstance(summary.value, ast.Call)
    assert [ast.unparse(arg) for arg in summary.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert summary.value.keywords == []
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == (
        "_terminal_qkv_activation_layout_shape_results"
    )
    assert len(_outer_calls()) == 1
    assert len(_pre_terminal_cleanup_calls(PRE_TERMINAL_PRE_ADD)) == 1
    assert len(_pre_terminal_cleanup_calls(CHANNEL_SLICE_PAD_MUL_SUMMARY)) == 1
    assert len(
        _pre_terminal_cleanup_calls("run_pre_terminal_affine_tail_cleanup")
    ) == 1


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
    _, helper = _lowerer_and_helper()
    summary_calls = _pre_terminal_cleanup_calls(
        CHANNEL_SLICE_PAD_MUL_SUMMARY
    )
    assert len(summary_calls) == 1
    assert [ast.unparse(argument) for argument in summary_calls[0].args] == [
        "context"
    ]
    assert summary_calls[0].keywords == []

    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)
    assert isinstance(helper.body[0].value.func, ast.Name)
    assert helper.body[0].value.func.id == "run_channel_slice_pad_mul"


def test_pre_terminal_pre_add_uses_prune_aware_owner() -> None:
    lowerer, _ = _lowerer_and_helper()
    owner_calls = _pre_terminal_cleanup_calls(PRE_TERMINAL_PRE_ADD)
    assert len(owner_calls) == 1
    assert [ast.unparse(argument) for argument in owner_calls[0].args] == [
        "context"
    ]
    assert owner_calls[0].keywords == []
    assert any(
        isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == OUTER_RESULT
        for statement in lowerer.body
    )

    assert not any(
        isinstance(node, ast.Name)
        and node.id == "pre_terminal_pre_add_tensor_count"
        for node in ast.walk(lowerer)
    )


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
