from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
CONSTANT_FOLD_CAST = "_run_constant_fold_cast_cleanup_pass_cluster"
VERY_LATE_PARENT = "_run_very_late_gather_constant_normalization_pass_cluster"
LATE_LAYOUT_PARENT = "_run_late_layout_mean_spp_gather_constant_cast_pass_cluster"


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
        if isinstance(node, ast.FunctionDef) and node.name == CONSTANT_FOLD_CAST
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


def _parent(lowerer: ast.FunctionDef, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _direct_invocation_index(parent: ast.FunctionDef) -> int:
    return next(
        index
        for index, statement in enumerate(parent.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == CONSTANT_FOLD_CAST
    )


def test_constant_fold_cast_signature_and_scope_fallback_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 21
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == ["state_scope"]
    assert [_expression_path(value) for value in helper.args.kw_defaults] == [None]
    assert helper.args.vararg is None
    assert helper.args.kwarg is None

    conditionals = [node for node in ast.walk(helper) if isinstance(node, ast.If)]
    assert len(conditionals) == 1
    conditional = conditionals[0]
    assert isinstance(conditional.test, ast.Compare)
    assert _expression_path(conditional.test.left) == "state_scope"
    assert [type(operator) for operator in conditional.test.ops] == [ast.Is]
    assert [_expression_path(value) for value in conditional.test.comparators] == [None]
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


def test_constant_fold_cast_preserves_both_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = (
        "run_constant_input_fold_cleanup",
        "run_redundant_cast_cleanup",
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


def test_constant_fold_cast_has_two_external_scope_production_calls() -> None:
    lowerer, _ = _lowerer_and_helper()
    owners: list[tuple[str, ast.Call]] = []
    for parent in lowerer.body:
        if not isinstance(parent, ast.FunctionDef):
            continue
        for statement in parent.body:
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Call)
                and isinstance(statement.value.func, ast.Name)
                and statement.value.func.id == CONSTANT_FOLD_CAST
            ):
                continue
            owners.append((parent.name, statement.value))

    assert [name for name, _ in owners] == [VERY_LATE_PARENT, LATE_LAYOUT_PARENT]
    for _, invocation in owners:
        assert invocation.args == []
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in invocation.keywords
        } == {"state_scope": "state_scope"}


def test_constant_fold_cast_preserves_very_late_parent_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    parent = _parent(lowerer, VERY_LATE_PARENT)
    invocation_index = _direct_invocation_index(parent)

    assert _direct_call_name(parent.body[invocation_index - 1]) == (
        "run_transpose_gather_axis_cleanup"
    )
    assert _direct_call_name(parent.body[invocation_index + 1]) == (
        "run_normalization_pad_layout_cleanup"
    )


def test_constant_fold_cast_preserves_late_layout_parent_boundary() -> None:
    lowerer, _ = _lowerer_and_helper()
    parent = _parent(lowerer, LATE_LAYOUT_PARENT)
    invocation_index = _direct_invocation_index(parent)

    assert invocation_index == len(parent.body) - 1
    assert _direct_call_name(parent.body[invocation_index - 1]) == (
        "run_transpose_gather_axis_cleanup"
    )
