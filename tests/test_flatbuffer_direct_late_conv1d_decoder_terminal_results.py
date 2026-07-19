from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input,
    _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains,
    _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATHS = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "conv1d_batchmatmul_layout.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "decoder_deconv_layout.py",
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_squeeze_mean_layout.py",
)
OWNERS = (
    "_optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains",
    "_optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input",
    "_optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains",
)
RESULT_TARGETS = (
    "_late_conv1d_batchmatmul_stats",
    "_late_decoder_deconv_stats",
    "_late_terminal_squeeze_mean_stats",
)
RESULT_SCHEMAS = (
    {"optimized_transpose_squeeze_unary_batchmatmul_nhwc_chains": 0},
    {
        "optimized_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input": 0,
    },
    {"optimized_transpose_squeeze_mean_squeeze_terminal_nhwc_chains": 0},
)
PREDECESSOR_TARGET = "_late_conv1d_tencoder_stats"
SUCCESSOR_TARGET = "_very_late_pad_layout_stats"


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


def _contains_call(node: ast.AST, name: str) -> bool:
    return any(
        isinstance(child, ast.Call)
        and (
            (isinstance(child.func, ast.Name) and child.func.id == name)
            or (isinstance(child.func, ast.Attribute) and child.func.attr == name)
        )
        for child in ast.walk(node)
    )


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _direct_locations() -> tuple[ast.FunctionDef, tuple[int, ...]]:
    lowerer = _lowerer()
    return lowerer, tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _call_name(statement) == owner
        )
        for owner in OWNERS
    )


def test_late_tail_schemas_wrappers_and_positive_cleanup_are_explicit() -> None:
    lowerer_functions = _functions(LOWERER_PATH)
    callbacks = (
        _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains,
        _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input,
        _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains,
    )
    for owner, owner_path, callback, schema in zip(
        OWNERS,
        OWNER_PATHS,
        callbacks,
        RESULT_SCHEMAS,
    ):
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

        owner_function = _functions(owner_path)[owner]
        cleanup_guards = [
            node
            for node in ast.walk(owner_function)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "rewritten"
            and _contains_call(node, "_prune_unused_tensors")
            and _contains_call(node, "sync_from_model_ir")
        ]
        assert len(cleanup_guards) == 1
        assert callback(ModelIR(f"{owner}_schema")) == schema


def test_late_tail_direct_chain_is_explicit() -> None:
    lowerer, indices = _direct_locations()
    assert indices == tuple(range(indices[0], indices[0] + len(OWNERS)))
    for index, owner, target in zip(indices, OWNERS, RESULT_TARGETS):
        invocation = lowerer.body[index]
        assert _single_target(invocation) == target
        call = _statement_call(invocation)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
        assert sum(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner
            for node in ast.walk(lowerer)
        ) == 1
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET


def test_late_tail_results_are_retained_for_observation() -> None:
    lowerer, indices = _direct_locations()
    assert tuple(_single_target(lowerer.body[index]) for index in indices) == (
        RESULT_TARGETS
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
