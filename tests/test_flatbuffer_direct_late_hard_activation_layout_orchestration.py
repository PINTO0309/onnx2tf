from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import late_hard_activation_layout_orchestration
from onnx2tf.tflite_builder.passes.late_hard_activation_layout_orchestration import (
    LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS,
    LateHardActivationLayoutContext,
    active_late_hard_activation_layout_pass_ids,
    build_late_hard_activation_layout_invocations,
    run_late_hard_activation_layout,
)


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


def _context() -> LateHardActivationLayoutContext:
    model_ir = ModelIR("late_hard_activation_layout_test")
    return LateHardActivationLayoutContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: late_hard_activation_layout_orchestration.RecoveryInvocation,
    context: LateHardActivationLayoutContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        if value is context.diagnostics:
            return "session.diagnostics"
        if isinstance(value, ModelIRPassStateScope):
            return "state_scope"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_late_hard_activation_layout_signature_and_delegate_are_explicit() -> None:
    _, helper = _lowerer_and_helper()

    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert [arg.arg for arg in helper.args.kwonlyargs] == ["include_layout_transpose"]
    assert helper.args.kw_defaults == [None]
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert len(helper.body) == 1
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
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "ModelIRPassStateScope"
        for node in ast.walk(helper)
    )

    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_late_hard_activation_layout"
    assert tuple(_expression_path(arg) for arg in call.args) == (
        "late_hard_activation_layout_context",
    )
    assert {
        str(keyword.arg): _expression_path(keyword.value) for keyword in call.keywords
    } == {"include_layout_transpose": "include_layout_transpose"}


@pytest.mark.parametrize(
    ("include_layout_transpose", "expected_ids"),
    [
        (False, LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[:1]),
        (True, LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS),
    ],
)
def test_late_hard_activation_layout_preserves_both_cleanup_forms(
    include_layout_transpose: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    invocations = build_late_hard_activation_layout_invocations(
        context,
        include_layout_transpose=include_layout_transpose,
    )

    assert (
        active_late_hard_activation_layout_pass_ids(
            include_layout_transpose=include_layout_transpose,
        )
        == expected_ids
    )
    assert tuple(step.pass_id for step in invocations) == expected_ids

    expected_hard_activation_contract = (
        ("model_ir",),
        {
            "include_hardswish": False,
            "include_hardsigmoid": True,
            "include_hardsigmoid_mul": True,
            "reverse_hardsigmoid_order": True,
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    expected_layout_contract = (
        ("model_ir",),
        {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": "state_scope",
        },
    )
    expected_contracts = {
        LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[0]: expected_hard_activation_contract,
        LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[1]: expected_layout_contract,
    }
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {pass_id: expected_contracts[pass_id] for pass_id in expected_ids}

    scopes = [dict(step.keyword_args)["state_scope"] for step in invocations]
    assert all(scope is scopes[0] for scope in scopes)
    assert isinstance(scopes[0], ModelIRPassStateScope)
    assert scopes[0].model_ir is context.model_ir
    assert scopes[0].layout_state is context.layout_state
    rebuilt_scope = dict(
        build_late_hard_activation_layout_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
        )[0].keyword_args
    )["state_scope"]
    assert rebuilt_scope is not scopes[0]


@pytest.mark.parametrize(
    ("include_layout_transpose", "expected_ids"),
    [
        (False, LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[:1]),
        (True, LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS),
    ],
)
def test_late_hard_activation_layout_runner_preserves_both_instrumented_orders(
    monkeypatch: pytest.MonkeyPatch,
    include_layout_transpose: bool,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for pass_id in expected_ids:
        monkeypatch.setattr(
            late_hard_activation_layout_orchestration,
            pass_id,
            recorder(pass_id),
        )

    run_late_hard_activation_layout(
        context,
        include_layout_transpose=include_layout_transpose,
    )

    assert events == list(expected_ids)


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


def test_late_hard_activation_layout_context_is_explicit() -> None:
    lowerer, _ = _lowerer_and_helper()
    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name)
            and target.id == "late_hard_activation_layout_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_late_hard_activation_layout_module_does_not_import_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "late_hard_activation_layout_orchestration.py"
    )
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    assert not any(
        isinstance(node, ast.ImportFrom)
        and node.module == "onnx2tf.tflite_builder.lower_from_onnx2tf"
        for node in tree.body
    )
    assert not any(
        isinstance(node, ast.Import)
        and any(
            alias.name == "onnx2tf.tflite_builder.lower_from_onnx2tf"
            for alias in node.names
        )
        for node in tree.body
    )
