from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
TERMINAL_BOUNDARY = "_run_terminal_boundary_layout_pass_cluster"
TERMINAL_BOUNDARY_OWNER_IDS = (
    "run_dual_mul_concat_layout_cleanup",
    "run_boundary_input_layout_cleanup",
    "run_pad_layout_cleanup",
    "run_layout_transpose_cleanup",
    "run_transpose_gather_channel_fanout_cleanup",
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
        if isinstance(node, ast.FunctionDef) and node.name == TERMINAL_BOUNDARY
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


def test_terminal_boundary_signature_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert helper.args.kwonlyargs == []
    assert helper.args.defaults == []
    assert helper.args.kw_defaults == []
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


def test_terminal_boundary_preserves_all_owner_contracts() -> None:
    _, helper = _lowerer_and_helper()
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in TERMINAL_BOUNDARY_OWNER_IDS
    ]
    calls.sort(key=lambda call: call.lineno)

    assert tuple(call.func.id for call in calls) == TERMINAL_BOUNDARY_OWNER_IDS
    expected_contract = {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for call in calls:
        assert tuple(_expression_path(arg) for arg in call.args) == ("model_ir",)
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        } == expected_contract


def test_terminal_boundary_has_one_argument_free_production_call() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == TERMINAL_BOUNDARY
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_terminal_boundary_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == TERMINAL_BOUNDARY
    )

    previous_boundary = lowerer.body[invocation_index - 1]
    assert isinstance(previous_boundary, ast.Expr)
    assert isinstance(previous_boundary.value, ast.Call)
    assert isinstance(previous_boundary.value.func, ast.Name)
    assert previous_boundary.value.func.id == (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )

    next_boundary = lowerer.body[invocation_index + 1]
    assert isinstance(next_boundary, ast.If)
    assert isinstance(next_boundary.test, ast.Name)
    assert next_boundary.test.id == "optimize_layout_transpose_chains"
