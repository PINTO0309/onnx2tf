from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LATE_LAYOUT = "_run_late_layout_mean_spp_gather_constant_cast_pass_cluster"
CONSTANT_FOLD_CAST = "_run_constant_fold_cast_cleanup_pass_cluster"
DIRECT_OWNER_IDS = (
    "run_layout_transpose_cleanup",
    "run_mean_mul_add_conv_layout_cleanup",
    "run_spp_layout_cleanup",
    "run_transpose_gather_axis_cleanup",
)
REQUIRED_OWNER_IDS = (
    *DIRECT_OWNER_IDS[1:],
    *CONSTANT_FOLD_CAST_PASS_IDS,
)
FULL_OWNER_IDS = (
    DIRECT_OWNER_IDS[0],
    *REQUIRED_OWNER_IDS,
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
        if isinstance(node, ast.FunctionDef) and node.name == LATE_LAYOUT
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


def _ordered_owner_calls(helper: ast.FunctionDef) -> list[ast.Call]:
    calls = [
        node
        for node in ast.walk(helper)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {*DIRECT_OWNER_IDS, CONSTANT_FOLD_CAST}
    ]
    return sorted(calls, key=lambda call: call.lineno)


def test_late_layout_signature_and_scope_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.posonlyargs == []
    assert helper.args.args == []
    assert [argument.arg for argument in helper.args.kwonlyargs] == [
        "include_layout_transpose"
    ]
    assert helper.args.kw_defaults == [None]
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
    assert tuple(_expression_path(argument) for argument in scope_calls[0].args) == (
        "model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in scope_calls[0].keywords
    } == {"layout_state": "session.layout_state"}


def test_late_layout_preserves_flattened_owner_order_and_contracts() -> None:
    _, helper = _lowerer_and_helper()
    calls = _ordered_owner_calls(helper)

    assert tuple(call.func.id for call in calls) == (
        *DIRECT_OWNER_IDS,
        CONSTANT_FOLD_CAST,
    )
    shared_contract = {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    for call in calls[:-1]:
        assert tuple(_expression_path(argument) for argument in call.args) == (
            "model_ir",
        )
        assert {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        } == shared_contract

    child_call = calls[-1]
    assert child_call.args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in child_call.keywords
    } == {"state_scope": "state_scope"}
    assert REQUIRED_OWNER_IDS == (
        "run_mean_mul_add_conv_layout_cleanup",
        "run_spp_layout_cleanup",
        "run_transpose_gather_axis_cleanup",
        "run_constant_input_fold_cleanup",
        "run_redundant_cast_cleanup",
    )
    assert FULL_OWNER_IDS == (
        "run_layout_transpose_cleanup",
        *REQUIRED_OWNER_IDS,
    )


def test_late_layout_optional_owner_has_one_exact_guard() -> None:
    _, helper = _lowerer_and_helper()
    conditionals = [
        statement for statement in helper.body if isinstance(statement, ast.If)
    ]

    assert len(conditionals) == 1
    conditional = conditionals[0]
    assert isinstance(conditional.test, ast.Name)
    assert conditional.test.id == "include_layout_transpose"
    assert conditional.orelse == []
    assert len(conditional.body) == 1
    statement = conditional.body[0]
    assert isinstance(statement, ast.Expr)
    assert isinstance(statement.value, ast.Call)
    assert isinstance(statement.value.func, ast.Name)
    assert statement.value.func.id == DIRECT_OWNER_IDS[0]
    assert all(
        call not in [node for node in ast.walk(conditional)]
        for call in _ordered_owner_calls(helper)[1:]
    )


def test_late_layout_has_one_required_policy_production_call() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LATE_LAYOUT
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocations[0].keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}


def test_late_layout_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LATE_LAYOUT
    )

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    for boundary in (previous, following):
        assert isinstance(boundary, ast.Expr)
        assert isinstance(boundary.value, ast.Call)
        assert isinstance(boundary.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
    assert following.value.func.id == "_replace_expand_dims_and_squeeze_with_reshape"
