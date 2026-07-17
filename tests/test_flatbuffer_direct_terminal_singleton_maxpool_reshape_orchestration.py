from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_SINGLETON_MAXPOOL_RESHAPE = "_run_terminal_singleton_maxpool_reshape_pass_pair"


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
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_SINGLETON_MAXPOOL_RESHAPE
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


def test_terminal_singleton_maxpool_reshape_is_straight_line_scoped() -> None:
    _, helper = _lowerer_and_helper()
    control_flow_nodes = (
        ast.AsyncFor,
        ast.AsyncWith,
        ast.For,
        ast.If,
        ast.Match,
        ast.Try,
        ast.While,
        ast.With,
    )

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 19
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 3
    assert not any(isinstance(node, control_flow_nodes) for node in ast.walk(helper))

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

    assignment = helper.body[0]
    assert isinstance(assignment, ast.Assign)
    assert len(assignment.targets) == 1
    assert isinstance(assignment.targets[0], ast.Name)
    assert assignment.targets[0].id == "state_scope"


def test_terminal_singleton_maxpool_reshape_preserves_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_calls = [
        statement.value
        for statement in helper.body[1:]
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]

    assert [call.func.id for call in cleanup_calls] == [
        "run_singleton_maxpool_layout_cleanup",
        "run_consecutive_reshape_cleanup",
    ]
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


def test_terminal_singleton_maxpool_reshape_invocation_is_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_SINGLETON_MAXPOOL_RESHAPE
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_terminal_singleton_maxpool_reshape_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_SINGLETON_MAXPOOL_RESHAPE
    )

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.If)
    assert isinstance(previous.test, ast.Name)
    assert previous.test.id == "optimize_layout_transpose_chains"
    previous_call = previous.body[0]
    assert isinstance(previous_call, ast.Expr)
    assert isinstance(previous_call.value, ast.Call)
    assert isinstance(previous_call.value.func, ast.Name)
    assert (
        previous_call.value.func.id
        == "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
    )

    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.If)
    assert isinstance(following.test, ast.Name)
    assert following.test.id == "optimize_layout_transpose_chains"
    following_call = following.body[0]
    assert isinstance(following_call, ast.Expr)
    assert isinstance(following_call.value, ast.Call)
    assert isinstance(following_call.value.func, ast.Name)
    assert (
        following_call.value.func.id
        == "_optimize_convpool_output_transpose_nhwc_passthrough_chains"
    )
