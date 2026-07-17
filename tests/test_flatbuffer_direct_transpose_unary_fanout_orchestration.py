from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    ATTENTION_GATE_QDQ_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TRANSPOSE_UNARY_FANOUT = "_run_transpose_unary_fanout_layout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == TRANSPOSE_UNARY_FANOUT
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


def test_transpose_unary_fanout_signature_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 35
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == [
        "include_layout_transpose",
        "include_unary_passthrough",
    ]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [
        False,
        True,
    ]
    assert helper.args.vararg is None
    assert helper.args.kwarg is None

    conditional_tests = [node.test for node in helper.body if isinstance(node, ast.If)]
    assert [_expression_path(test) for test in conditional_tests] == [
        "include_layout_transpose",
        "include_unary_passthrough",
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


def test_transpose_unary_fanout_preserves_conditional_call_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = [
        "run_layout_transpose_cleanup",
        "run_transpose_unary_passthrough_cleanup",
        "run_transpose_unary_fanout_bridge_cleanup",
        "run_transpose_unary_binary_fanout_bridge_cleanup",
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


def test_transpose_unary_fanout_preserves_both_invocation_variants() -> None:
    lowerer, _ = _lowerer_and_helper()
    direct_invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TRANSPOSE_UNARY_FANOUT
    ]

    assert len(direct_invocations) == 1
    assert direct_invocations[0].args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in direct_invocations[0].keywords
    } == {
        "include_layout_transpose": True,
        "include_unary_passthrough": False,
    }

    attention_context = next(
        statement.value
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "attention_recovery_context"
            for target in statement.targets
        )
        and isinstance(statement.value, ast.Call)
    )
    callback_keyword = next(
        keyword
        for keyword in attention_context.keywords
        if keyword.arg == "transpose_unary_fanout_cluster"
    )
    assert isinstance(callback_keyword.value, ast.Name)
    assert callback_keyword.value.id == TRANSPOSE_UNARY_FANOUT


def test_transpose_unary_fanout_preserves_direct_and_callback_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    parent = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == TRANSPOSE_UNARY_FANOUT
            for node in ast.walk(statement)
        )
    )
    direct_index = next(
        index
        for index, statement in enumerate(parent.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TRANSPOSE_UNARY_FANOUT
    )
    previous = parent.body[direct_index - 1]
    following = parent.body[direct_index + 1]
    for boundary in (previous, following):
        assert isinstance(boundary, ast.Expr)
        assert isinstance(boundary.value, ast.Call)
        assert isinstance(boundary.value.func, ast.Name)
    assert previous.value.func.id == "_run_layout_attention_quantized_recovery_suffix"
    assert following.value.func.id == "_run_safe_binary_bridge_recovery_sequence"

    callback_index = ATTENTION_GATE_QDQ_PASS_IDS.index(TRANSPOSE_UNARY_FANOUT)
    assert ATTENTION_GATE_QDQ_PASS_IDS[callback_index - 1] == (
        "_optimize_transposeconv_output_channel1_terminal_transpose_chains"
    )
    assert ATTENTION_GATE_QDQ_PASS_IDS[callback_index + 1] == (
        "_optimize_transpose_dequant_relu_quantize_bridges"
    )
