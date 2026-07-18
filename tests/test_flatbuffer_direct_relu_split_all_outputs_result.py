from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "split_all_outputs_layout.py"
)
RELU_SPLIT_ALL = "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains"
OWNER_NAME = "optimize_transpose_relu_split_all_outputs_to_nhwc_chains"
SPLIT_CONV_CONCAT = (
    "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    if not isinstance(statement.value, ast.Call):
        return None
    return statement.value


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


def test_relu_split_all_outputs_schema_and_positive_cleanup_are_explicit() -> None:
    lowerer_wrapper = _functions(LOWERER_PATH)[RELU_SPLIT_ALL]
    assert len(lowerer_wrapper.body) == 1
    wrapper_return = lowerer_wrapper.body[0]
    assert isinstance(wrapper_return, ast.Return)
    assert isinstance(wrapper_return.value, ast.Call)
    assert isinstance(wrapper_return.value.func, ast.Name)
    assert wrapper_return.value.func.id == f"{RELU_SPLIT_ALL}_pass"
    assert [ast.unparse(argument) for argument in wrapper_return.value.args] == [
        "model_ir",
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in wrapper_return.value.keywords
    } == {
        "graph_index": "graph_index",
        "layout_state": "layout_state",
        "max_rewrites": "max_rewrites",
        "candidate": "candidate",
    }

    owner_tree = ast.parse(OWNER_PATH.read_text(encoding="utf-8"))
    stats_key = next(
        statement
        for statement in owner_tree.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement) == "_STATS_KEY"
    )
    assert isinstance(stats_key.value, ast.Constant)
    assert stats_key.value.value == (
        "optimized_transpose_relu_split_all_outputs_to_nhwc_chains"
    )

    owner = _functions(OWNER_PATH)[OWNER_NAME]
    cleanup_guards = [
        statement
        for statement in owner.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_prune_unused_tensors"
            for node in ast.walk(statement)
        )
    ]
    assert len(cleanup_guards) == 1
    cleanup_guard = cleanup_guards[0]
    assert ast.unparse(cleanup_guard.test) == "rewritten > 0"
    assert [
        node.func.id
        for node in ast.walk(cleanup_guard)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    ] == ["_prune_unused_tensors"]
    assert not any(
        isinstance(statement, ast.Expr)
        and _call_name(statement) == "_prune_unused_tensors"
        for statement in owner.body
    )


@pytest.mark.xfail(
    strict=True,
    reason="the two direct ReLU/Split all-outputs results are not retained yet",
)
def test_lowerer_retains_both_relu_split_all_outputs_results() -> None:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    direct_results = [
        statement
        for statement in lowerer.body
        if _call_name(statement) == RELU_SPLIT_ALL
    ]
    assert len(direct_results) == 2
    assert [_single_target(statement) for statement in direct_results] == [
        "_post_sinet_relu_split_all_outputs_stats",
        "_terminal_relu_split_all_outputs_stats",
    ]
    for statement in direct_results:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}

    first_index = lowerer.body.index(direct_results[0])
    first_previous = lowerer.body[first_index - 1]
    first_following = lowerer.body[first_index + 1]
    assert _single_target(first_previous) == "_post_sinet_qkv_attention_results"
    assert _call_name(first_previous) == "_run_qkv_attention_layout_pass_cluster"
    assert _call_name(first_following) == SPLIT_CONV_CONCAT

    second_index = lowerer.body.index(direct_results[1])
    second_previous = lowerer.body[second_index - 1]
    second_following = lowerer.body[second_index + 1]
    assert _call_name(second_previous) == "_optimize_transpose_pre_concat_nhwc_chains"
    assert _call_name(second_following) == SPLIT_CONV_CONCAT
