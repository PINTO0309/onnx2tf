from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains,
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains,
    _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "conv1d_unary_layout.py"
)
OWNERS = (
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains",
)
RESULT_TARGETS = (
    "_late_conv1d_squeeze_unary_stats",
    "_late_conv1d_rank4_unary_stats",
    "_late_conv1d_unary_fanout_stats",
)
RESULT_SCHEMAS = (
    {"optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_chains": 0},
    {
        "optimized_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains": 0,
    },
    {
        "optimized_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains": 0,
    },
)
COMPOSITE_TARGET = "_late_conv1d_decoder_layout_results"
COMPOSITE_OWNER = "run_late_conv1d_decoder_layout_cleanup"


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


def test_late_conv1d_unary_schemas_wrappers_and_cleanup_are_explicit() -> None:
    lowerer_functions = _functions(LOWERER_PATH)
    owner_functions = _functions(OWNER_PATH)
    callbacks = (
        _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains,
        _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains,
        _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains,
    )
    for owner, callback, schema in zip(OWNERS, callbacks, RESULT_SCHEMAS):
        wrapper = lowerer_functions[owner]
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
        assert call.func.id == f"{owner}_pass"
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
        } == {
            "graph_index": "graph_index",
            "layout_state": "layout_state",
        }

        owner_function = owner_functions[owner]
        assert sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(owner_function)
        ) == 2
        assert callback(ModelIR(f"{owner}_schema")) == schema


def test_late_conv1d_unary_direct_calls_move_to_composite() -> None:
    lowerer = _lowerer()
    for owner in OWNERS:
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner
            for node in ast.walk(lowerer)
        )
    composite = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    assert _call_name(composite) == COMPOSITE_OWNER


def test_late_conv1d_unary_old_result_locals_are_removed() -> None:
    lowerer = _lowerer()
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
