from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _apply_safe_transpose_reduction_lite,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_apply_safe_transpose_reduction_lite"
RESULT_TARGET = "_no_layout_safe_transpose_reduction_stats"
SUCCESSOR_TARGET = "_no_layout_fallback_affine_prepost_stats"


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _fallback_branch() -> ast.If:
    lowerer = _lowerer()
    parent = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(
            _call_name(child)
            == "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
            for child in statement.body
        )
        and len(statement.orelse) == 1
    )
    fallback = parent.orelse[0]
    assert isinstance(fallback, ast.If)
    assert ast.unparse(fallback.test) == (
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    return fallback


def test_safe_transpose_reduction_zero_schema_is_explicit() -> None:
    assert _apply_safe_transpose_reduction_lite(ModelIR("stable_schema")) == {
        "safe_transpose_reduction_lite_applied": 0,
        "safe_transpose_reduction_lite_reduced": 0,
        "safe_transpose_reduction_lite_unbound_after": 0,
    }


def test_safe_transpose_reduction_raw_boundary_is_explicit() -> None:
    fallback = _fallback_branch()
    assert len(fallback.body) == 2
    invocation = fallback.body[0]
    assert isinstance(invocation, ast.Expr)
    assert _call_name(invocation) == OWNER
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _single_target(fallback.body[1]) == SUCCESSOR_TARGET


@pytest.mark.xfail(
    strict=True,
    reason="the no-layout safe-transpose result is still discarded",
)
def test_safe_transpose_reduction_result_is_retained_for_observation() -> None:
    fallback = _fallback_branch()
    invocation = fallback.body[0]
    assert _single_target(invocation) == RESULT_TARGET
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert call.keywords == []
    assert _single_target(fallback.body[1]) == SUCCESSOR_TARGET

    lowerer = _lowerer()
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
