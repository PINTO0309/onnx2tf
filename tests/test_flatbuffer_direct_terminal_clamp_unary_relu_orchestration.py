from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_CLAMP_UNARY_RELU = "_run_terminal_clamp_unary_relu_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == TERMINAL_CLAMP_UNARY_RELU
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


def test_terminal_clamp_unary_relu_is_a_straight_line_scoped_cluster() -> None:
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
    assert helper.end_lineno - helper.lineno + 1 == 26
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 4
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


def test_terminal_clamp_unary_relu_preserves_all_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_calls = [
        statement.value
        for statement in helper.body[1:]
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
    ]

    assert [call.func.id for call in cleanup_calls] == [
        "run_clamp_cleanup",
        "run_transpose_unary_passthrough_cleanup",
        "run_maximum_zero_relu_cleanup",
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


def test_terminal_clamp_unary_relu_invocation_remains_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_CLAMP_UNARY_RELU
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_terminal_clamp_unary_relu_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_CLAMP_UNARY_RELU
    )

    previous = lowerer.body[invocation_index - 1]
    assert isinstance(previous, ast.If)
    assert isinstance(previous.test, ast.Name)
    assert previous.test.id == "optimize_layout_transpose_chains"
    previous_call = previous.body[-1]
    assert isinstance(previous_call, ast.Expr)
    assert isinstance(previous_call.value, ast.Call)
    assert isinstance(previous_call.value.func, ast.Name)
    assert previous_call.value.func.id == "_run_singleton_reshape_layout_pass_cluster"
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in previous_call.value.keywords
    } == {
        "include_layout_transpose": True,
        "include_multi_branch_gate": True,
    }

    following = lowerer.body[invocation_index + 1]
    assert isinstance(following, ast.Expr)
    assert isinstance(following.value, ast.Call)
    assert isinstance(following.value.func, ast.Name)
    assert following.value.func.id == "_run_sinet_terminal_layout_recovery_sequence"
