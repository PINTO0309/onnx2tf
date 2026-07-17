from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LATE_SPP_CONCAT_UNARY_CONV = "_run_late_spp_concat_unary_conv_pass_pair"


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
        if isinstance(node, ast.FunctionDef) and node.name == LATE_SPP_CONCAT_UNARY_CONV
    )
    return lowerer, helper


def _expression_path(node: ast.expr) -> Any:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return f"{_expression_path(node.value)}.{node.attr}"
    raise AssertionError(f"unexpected call expression: {ast.dump(node)}")


def test_late_spp_concat_unary_conv_is_a_straight_line_scoped_pair() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 17
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
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


def test_late_spp_concat_unary_conv_preserves_both_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = [
        "run_spp_layout_cleanup",
        "run_concat_unary_conv_layout_cleanup",
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


def test_late_spp_concat_unary_conv_invocation_remains_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LATE_SPP_CONCAT_UNARY_CONV
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert invocations[0].keywords == []


def test_late_spp_concat_unary_conv_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LATE_SPP_CONCAT_UNARY_CONV
    )

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    for boundary in (previous, following):
        assert isinstance(boundary, ast.Expr)
        assert isinstance(boundary.value, ast.Call)
        assert isinstance(boundary.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains"
    )
    assert (
        following.value.func.id
        == "_optimize_transpose_shape_extract_nhwc_to_nchw_chains"
    )
