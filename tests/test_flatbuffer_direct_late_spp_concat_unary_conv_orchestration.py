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
LATE_SPP_CONCAT_UNARY_CONV_SUMMARY = (
    "run_late_spp_concat_unary_conv_summary"
)
TERMINAL_AFFINE_SLICE_SPP_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_affine_slice_spp_orchestration.py"
)
TERMINAL_AFFINE_SLICE_SPP = "run_terminal_affine_slice_spp_cleanup"
TERMINAL_AFFINE_SLICE_SPP_RESULT = "_terminal_affine_slice_spp_results"
OUTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_affine_slice_spp_orchestration.py"
)
OUTER_OWNER = "run_pre_terminal_affine_slice_spp_cleanup"
OUTER_RESULT = "_pre_terminal_affine_slice_spp_results"
LOWERER_OWNER = "run_terminal_affine_qkv_layout_shape_cleanup"
LOWERER_RESULT = "_terminal_affine_qkv_layout_shape_results"


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


def _terminal_affine_slice_spp_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(
        TERMINAL_AFFINE_SLICE_SPP_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_AFFINE_SLICE_SPP
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
        and node.func.id == TERMINAL_AFFINE_SLICE_SPP
    ]


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

    assert invocations == []
    summary_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LATE_SPP_CONCAT_UNARY_CONV_SUMMARY
    ]
    assert summary_invocations == []
    owner_calls = _terminal_affine_slice_spp_calls(
        LATE_SPP_CONCAT_UNARY_CONV_SUMMARY
    )
    assert len(owner_calls) == 1
    assert [ast.unparse(arg) for arg in owner_calls[0].args] == ["context"]
    assert owner_calls[0].keywords == []


def test_late_spp_concat_unary_conv_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_RESULT
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LOWERER_OWNER
    )

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    assert isinstance(previous, ast.If)
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following).startswith(
        "session.record_phase_result("
        "'shape_reconciliation.terminal.expand_squeeze'"
    )
    assert len(_outer_calls()) == 1


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
    summary = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_RESULT
    )
    first_index = lowerer.body.index(summary)
    summary_call = summary.value
    assert isinstance(summary_call, ast.Call)
    assert isinstance(summary_call.func, ast.Name)
    assert summary_call.func.id == LOWERER_OWNER
    assert [ast.unparse(arg) for arg in summary_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in summary_call.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}

    previous = lowerer.body[first_index - 1]
    assert isinstance(previous, ast.If)
    following = lowerer.body[first_index + 1]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following).startswith(
        "session.record_phase_result("
        "'shape_reconciliation.terminal.expand_squeeze'"
    )
    assert len(_outer_calls()) == 1
    owner_calls = _terminal_affine_slice_spp_calls(
        LATE_SPP_CONCAT_UNARY_CONV_SUMMARY
    )
    assert len(owner_calls) == 1
    assert [ast.unparse(arg) for arg in owner_calls[0].args] == ["context"]


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
