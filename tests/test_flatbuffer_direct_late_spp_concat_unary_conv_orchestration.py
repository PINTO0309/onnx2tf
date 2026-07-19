from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_spp_concat_unary_conv_orchestration,
)
from onnx2tf.tflite_builder.passes.late_spp_concat_unary_conv_orchestration import (
    LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS,
    LateSPPConcatUnaryConvContext,
    build_late_spp_concat_unary_conv_invocations,
    run_late_spp_concat_unary_conv,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LATE_SPP_CONCAT_UNARY_CONV = "_run_late_spp_concat_unary_conv_pass_pair"


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
        if isinstance(node, ast.FunctionDef) and node.name == LATE_SPP_CONCAT_UNARY_CONV
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _context() -> LateSPPConcatUnaryConvContext:
    model_ir = ModelIR("late_spp_concat_unary_conv_test")
    return LateSPPConcatUnaryConvContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: late_spp_concat_unary_conv_orchestration.RecoveryInvocation,
    context: LateSPPConcatUnaryConvContext,
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


def test_late_spp_concat_unary_conv_is_a_straight_line_delegate() -> None:
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


def test_late_spp_concat_unary_conv_preserves_both_cleanup_contracts() -> None:
    context = _context()
    invocations = build_late_spp_concat_unary_conv_invocations(context)

    assert (
        tuple(step.pass_id for step in invocations)
        == LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS
    )
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
    } == {pass_id: expected_contract for pass_id in LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_late_spp_concat_unary_conv_invocations(context)[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


def test_late_spp_concat_unary_conv_invocation_remains_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LATE_SPP_CONCAT_UNARY_CONV
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_late_spp_concat_unary_conv_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "late_spp_results"
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LATE_SPP_CONCAT_UNARY_CONV
    )

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 2]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_terminal_slice_pad_concat_stats"
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_late_pre_qkv_shape_extract_stats"
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert (
        following.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )


def test_late_spp_concat_unary_conv_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    statement = helper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_late_spp_concat_unary_conv"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "late_spp_concat_unary_conv_context",
    )
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "late_spp_concat_unary_conv_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_late_spp_concat_unary_conv_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS:
        monkeypatch.setattr(
            late_spp_concat_unary_conv_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_late_spp_concat_unary_conv(context)

    assert events == list(LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS)


def test_late_spp_concat_unary_conv_returns_and_summarizes_mutations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    spp_result = {
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains": 2,
    }
    concat_result = {
        "optimized_transpose_concat_unary_fanout_conv_nhwc_chains": 3,
    }
    expected_results = (spp_result, concat_result)

    def return_results(invocations, *, expected_pass_ids, phase_name):
        assert tuple(
            invocation.pass_id for invocation in invocations
        ) == LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS
        assert tuple(expected_pass_ids) == LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS
        assert phase_name == "late SPP/concat-unary-conv"
        return expected_results

    monkeypatch.setattr(
        late_spp_concat_unary_conv_orchestration,
        "run_recovery_invocations",
        return_results,
    )

    results = run_late_spp_concat_unary_conv(context)
    summarize = getattr(
        late_spp_concat_unary_conv_orchestration,
        "summarize_late_spp_concat_unary_conv_mutations",
    )
    summary = summarize(results)

    assert results == expected_results
    assert summary == {**spp_result, **concat_result}
    with pytest.raises(
        ValueError,
        match=r"late SPP mutation summary expected 2 pass results",
    ):
        summarize(())

    _, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    assert isinstance(helper.body[0], ast.Return)
    assert isinstance(helper.body[0].value, ast.Call)


def test_lowerer_captures_late_spp_mutation_evidence() -> None:
    lowerer, _ = _lowerer_and_helper()
    target_names = ("late_spp_results", "_late_spp_stats")
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
    assert result_call.func.id == LATE_SPP_CONCAT_UNARY_CONV
    summary_call = assignments[target_names[1]]
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == (
        "summarize_late_spp_concat_unary_conv_mutations"
    )

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.Assign)
    assert len(previous.targets) == 1
    assert isinstance(previous.targets[0], ast.Name)
    assert previous.targets[0].id == "_terminal_slice_pad_concat_stats"
    assert isinstance(previous.value, ast.Call)
    assert isinstance(previous.value.func, ast.Name)
    assert previous.value.func.id == (
        "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    following = lowerer.body[first_index + 2]
    assert isinstance(following, ast.Assign)
    assert len(following.targets) == 1
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "_late_pre_qkv_shape_extract_stats"


def test_late_spp_concat_unary_conv_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "late_spp_concat_unary_conv_orchestration.py"
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
