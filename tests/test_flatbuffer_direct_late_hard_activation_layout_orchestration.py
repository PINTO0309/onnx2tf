from __future__ import annotations

import ast
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
LATE_HARD_ACTIVATION_LAYOUT = "_run_late_hard_activation_layout_pass_pair"


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
        and node.name == LATE_HARD_ACTIVATION_LAYOUT
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


def test_late_hard_activation_layout_signature_scope_and_guard_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 25
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == ["include_layout_transpose"]
    assert helper.args.kw_defaults == [None]
    assert helper.args.vararg is None
    assert helper.args.kwarg is None

    conditional = next(
        statement for statement in helper.body if isinstance(statement, ast.If)
    )
    assert _expression_path(conditional.test) == "include_layout_transpose"
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


def test_late_hard_activation_layout_preserves_both_cleanup_contracts() -> None:
    _, helper = _lowerer_and_helper()
    cleanup_names = [
        "run_hard_activation_passthrough_cleanup",
        "run_layout_transpose_cleanup",
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

    hard_activation_call, layout_transpose_call = cleanup_calls
    assert tuple(_expression_path(arg) for arg in hard_activation_call.args) == (
        "model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in hard_activation_call.keywords
    } == {
        "include_hardswish": False,
        "include_hardsigmoid": True,
        "include_hardsigmoid_mul": True,
        "reverse_hardsigmoid_order": True,
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }
    assert tuple(_expression_path(arg) for arg in layout_transpose_call.args) == (
        "model_ir",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in layout_transpose_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
        "state_scope": "state_scope",
    }

    conditional = next(
        statement for statement in helper.body if isinstance(statement, ast.If)
    )
    assert layout_transpose_call in list(ast.walk(conditional))
    assert hard_activation_call not in list(ast.walk(conditional))


def test_late_hard_activation_layout_preserves_required_invocation_option() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == LATE_HARD_ACTIVATION_LAYOUT
    ]

    assert len(invocations) == 1
    assert invocations[0].args == []
    assert {
        str(keyword.arg): _expression_path(keyword.value)
        for keyword in invocations[0].keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}


def test_late_hard_activation_layout_preserves_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocation_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.Expr)
        and isinstance(statement.value, ast.Call)
        and isinstance(statement.value.func, ast.Name)
        and statement.value.func.id == LATE_HARD_ACTIVATION_LAYOUT
    )

    previous = lowerer.body[invocation_index - 1]
    following = lowerer.body[invocation_index + 1]
    for boundary in (previous, following):
        assert isinstance(boundary, ast.Expr)
        assert isinstance(boundary.value, ast.Call)
        assert isinstance(boundary.value.func, ast.Name)
    assert (
        previous.value.func.id
        == "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains"
    )
    assert following.value.func.id == "_optimize_transpose_pre_concat_nhwc_chains"
