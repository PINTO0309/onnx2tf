from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SE_FC_GATHER = "_run_se_fc_gather_channel_fanout_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == SE_FC_GATHER
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


def _direct_call_name(statement: ast.stmt) -> str:
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    return statement.value.func.id


def _direct_invocation_index(statements: list[ast.stmt]) -> int:
    return next(
        index
        for index, statement in enumerate(statements)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == SE_FC_GATHER
    )


def test_se_fc_gather_signature_and_target_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 20
    assert [argument.arg for argument in helper.args.args] == [
        "target_model_ir",
        "target_layout_state",
    ]
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.defaults == []
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
    assert tuple(_expression_path(arg) for arg in scope_calls[0].args) == (
        "target_model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in scope_calls[0].keywords
    } == {"layout_state": "target_layout_state"}


def test_se_fc_gather_preserves_both_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = (
        "run_se_fc_layout_cleanup",
        "run_transpose_gather_channel_fanout_cleanup",
    )
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in cleanup_names
    ]
    calls.sort(key=lambda call: call.lineno)

    assert tuple(call.func.id for call in calls) == cleanup_names
    expected_contract = {
        "layout_state": "target_layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for call in calls:
        assert tuple(_expression_path(arg) for arg in call.args) == ("target_model_ir",)
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        } == expected_contract


def test_se_fc_gather_preserves_both_target_forms() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SE_FC_GATHER
    ]
    invocations.sort(key=lambda call: call.lineno)

    assert [
        tuple(_expression_path(argument) for argument in invocation.args)
        for invocation in invocations
    ] == [
        ("fallback_ir", None),
        ("model_ir", "session.layout_state"),
    ]
    assert all(invocation.keywords == [] for invocation in invocations)


def test_se_fc_gather_preserves_fallback_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    fallback_block = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(candidate, ast.Expr)
            and isinstance(candidate.value, ast.Call)
            and isinstance(candidate.value.func, ast.Name)
            and candidate.value.func.id == SE_FC_GATHER
            for candidate in statement.body
        )
    )
    invocation_index = _direct_invocation_index(fallback_block.body)

    assert _direct_call_name(fallback_block.body[invocation_index - 1]) == (
        "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    assert _direct_call_name(fallback_block.body[invocation_index + 1]) == (
        "_reconcile_static_tensor_shapes"
    )


def test_se_fc_gather_preserves_main_model_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = _direct_invocation_index(lowerer.body)

    assert _direct_call_name(lowerer.body[invocation_index - 1]) == (
        "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    assert _direct_call_name(lowerer.body[invocation_index + 1]) == (
        "_reconcile_static_tensor_shapes"
    )
