from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_tencoder_add_expand_transpose_conv_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "conv1d_tencoder_layout.py"
)
OWNER = "_optimize_tencoder_add_expand_transpose_conv_nhwc_chains"
RESULT_TARGET = "_late_conv1d_tencoder_stats"
PREDECESSOR_TARGET = "_late_conv1d_instancenorm_unary_stats"
SUCCESSOR = "_optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains"
RESULT_SCHEMA = {"optimized_tencoder_add_expand_transpose_conv_nhwc_chains": 0}


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


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


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _direct_location() -> tuple[ast.FunctionDef, int]:
    lowerer = _lowerer()
    return lowerer, next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == OWNER
    )


def test_late_conv1d_tencoder_schema_wrapper_and_cleanup_are_explicit() -> None:
    wrapper = _functions(LOWERER_PATH)[OWNER]
    assert [argument.arg for argument in wrapper.args.args] == ["model_ir"]
    assert [argument.arg for argument in wrapper.args.kwonlyargs] == [
        "graph_index",
        "layout_state",
    ]
    assert [ast.unparse(value) for value in wrapper.args.kw_defaults] == [
        "None",
        "None",
    ]
    statement = wrapper.body[0]
    assert isinstance(statement, ast.Return)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == f"{OWNER}_pass"
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
    }

    owner_function = _functions(OWNER_PATH)[OWNER]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_prune_unused_tensors"
        for node in ast.walk(owner_function)
    ) == 2
    assert _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(
        ModelIR("late_conv1d_tencoder_schema")
    ) == RESULT_SCHEMA


def test_late_conv1d_tencoder_direct_boundary_is_explicit() -> None:
    lowerer, index = _direct_location()
    invocation = lowerer.body[index]
    assert isinstance(invocation, ast.Expr)
    call = _statement_call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"layout_state": "session.layout_state"}
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _call_name(lowerer.body[index + 1]) == SUCCESSOR
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(lowerer)
    ) == 1


@pytest.mark.xfail(
    strict=True,
    reason="late Conv1D tencoder result is discarded",
)
def test_late_conv1d_tencoder_result_is_retained_for_observation() -> None:
    lowerer, index = _direct_location()
    assert _single_target(lowerer.body[index]) == RESULT_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id == RESULT_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
