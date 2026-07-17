from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
VERY_LATE = "_run_very_late_gather_constant_normalization_pass_cluster"
CONSTANT_FOLD_CAST = "_run_constant_fold_cast_cleanup_pass_cluster"
EFFECTIVE_OWNER_IDS = (
    "run_transpose_gather_axis_cleanup",
    *CONSTANT_FOLD_CAST_PASS_IDS,
    "run_normalization_pad_layout_cleanup",
)


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
        if isinstance(node, ast.FunctionDef) and node.name == VERY_LATE
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected expression: {ast.dump(node)}")


def _direct_calls(helper: ast.FunctionDef) -> list[ast.Call]:
    calls = [
        statement.value
        for statement in helper.body
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]
    return calls


def test_very_late_signature_and_shared_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 22
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
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

    scope_calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
    ]
    assert len(scope_calls) == 1
    assert tuple(_expression_path(arg) for arg in scope_calls[0].args) == ("model_ir",)
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in scope_calls[0].keywords
    } == {"layout_state": "session.layout_state"}


def test_very_late_preserves_all_phase_call_contracts() -> None:
    _, helper = _lowerer_and_helper()
    calls = _direct_calls(helper)

    assert tuple(call.func.id for call in calls) == (
        "run_transpose_gather_axis_cleanup",
        CONSTANT_FOLD_CAST,
        "run_normalization_pad_layout_cleanup",
    )
    assert tuple(_expression_path(arg) for arg in calls[0].args) == ("model_ir",)
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in calls[0].keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    assert calls[1].args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in calls[1].keywords
    } == {"state_scope": "state_scope"}
    assert tuple(_expression_path(arg) for arg in calls[2].args) == ("model_ir",)
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in calls[2].keywords
    } == {
        "include_instance": False,
        "include_flatten": True,
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }


def test_very_late_preserves_four_effective_owner_steps() -> None:
    _, helper = _lowerer_and_helper()
    calls = _direct_calls(helper)
    direct_ids = tuple(call.func.id for call in calls)
    effective_ids = (
        direct_ids[0],
        *CONSTANT_FOLD_CAST_PASS_IDS,
        direct_ids[2],
    )

    assert CONSTANT_FOLD_CAST_PASS_IDS == (
        "run_constant_input_fold_cleanup",
        "run_redundant_cast_cleanup",
    )
    assert effective_ids == EFFECTIVE_OWNER_IDS


def test_very_late_preserves_sole_terminal_invocation_and_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_indexes = [
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == VERY_LATE
    ]

    assert len(invocation_indexes) == 1
    invocation_index = invocation_indexes[0]
    invocation = lowerer.body[invocation_index]
    assert isinstance(invocation, ast.Expr)
    assert isinstance(invocation.value, ast.Call)
    assert invocation.value.args == []
    assert invocation.value.keywords == []

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    for boundary in (previous, following):
        assert isinstance(boundary, ast.Expr)
        assert isinstance(boundary.value, ast.Call)
        assert isinstance(boundary.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
    )
    assert following.value.func.id == "_resolve_dynamic_reshape_shapes"
