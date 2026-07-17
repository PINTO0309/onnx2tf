from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
QKV_ATTENTION = "_run_qkv_attention_layout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == QKV_ATTENTION
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    if isinstance(node, ast.Constant):
        return node.value
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def _call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    return statement.value.func.id


def test_qkv_attention_signature_scope_and_guards_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 29
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_prefix",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]
    assert helper.args.vararg is None
    assert helper.args.kwarg is None

    conditionals = [
        statement for statement in helper.body if isinstance(statement, ast.If)
    ]
    assert [_expression_path(statement.test) for statement in conditionals] == [
        "include_layout_transpose",
        "include_prefix",
    ]
    assert not any(
        isinstance(
            node,
            (
                ast.AsyncFor,
                ast.AsyncWith,
                ast.For,
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


def test_qkv_attention_preserves_all_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = [
        "run_layout_transpose_cleanup",
        "run_qkv_attention_prefix_cleanup",
        "run_qkv_attention_bridge_cleanup",
    ]
    cleanup_calls = sorted(
        [
            node
            for node in ast.walk(helper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in cleanup_names
        ],
        key=lambda call: call.lineno,
    )

    assert [call.func.id for call in cleanup_calls] == cleanup_names
    for call in cleanup_calls:
        assert tuple(_expression_path(arg) for arg in call.args) == ("model_ir",)
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        } == {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        }


def test_qkv_attention_preserves_both_invocation_forms() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == QKV_ATTENTION
    ]

    assert len(invocations) == 3
    default_invocations = [call for call in invocations if call.keywords == []]
    assert len(default_invocations) == 2
    assert all(call.args == [] for call in default_invocations)
    late_invocation = next(call for call in invocations if call.keywords)
    assert late_invocation.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in late_invocation.keywords
    } == {
        "include_layout_transpose": "optimize_layout_transpose_chains",
        "include_prefix": False,
    }


def test_qkv_attention_preserves_both_default_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    layout_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == QKV_ATTENTION
            for node in ast.walk(statement)
        )
    )
    nested_index = next(
        index
        for index, statement in enumerate(layout_block.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
    )
    assert (
        _call_name(layout_block.body[nested_index - 1])
        == "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )
    assert (
        _call_name(layout_block.body[nested_index + 1])
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )

    top_level_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
        and statement.value.keywords == []
    )
    assert (
        _call_name(lowerer.body[top_level_index - 1])
        == "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )
    assert (
        _call_name(lowerer.body[top_level_index + 1])
        == "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains"
    )


def test_qkv_attention_preserves_late_bridge_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    late_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == QKV_ATTENTION
        and statement.value.keywords
    )

    assert (
        _call_name(lowerer.body[late_index - 1])
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
    assert (
        _call_name(lowerer.body[late_index + 1])
        == "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw"
    )
