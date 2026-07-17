from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
from pathlib import Path
from typing import Any, Callable

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration import (
    QLINEAR_MEAN_CONCAT_PASS_IDS,
    QLinearRecoveryContext,
    build_qlinear_mean_concat_invocations,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS,
    SINetTerminalLayoutRecoveryContext,
    build_sinet_terminal_layout_recovery_invocations,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PASSES_ROOT = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"


def _expression_path(expression: ast.expr) -> object:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return f"{_expression_path(expression.value)}.{expression.attr}"
    return type(expression).__name__


def _callback(name: str) -> Callable[..., Any]:
    def callback(*args: Any, **kwargs: Any) -> tuple[str, tuple[Any, ...], dict]:
        return name, args, kwargs

    return callback


def test_qlinear_context_is_shared_and_frozen() -> None:
    model_ir = ModelIR("remaining_qlinear_context")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []

    assert QLinearRecoveryContext is ModelIRPassContext
    assert is_dataclass(QLinearRecoveryContext)
    assert tuple(field.name for field in fields(QLinearRecoveryContext)) == (
        "model_ir",
        "layout_state",
        "diagnostics",
    )
    context = QLinearRecoveryContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    assert context.model_ir is model_ir
    assert context.layout_state is layout_state
    assert context.diagnostics is diagnostics
    with pytest.raises(FrozenInstanceError):
        context.model_ir = ModelIR("replacement")


def test_qlinear_builder_accepts_the_common_context_without_observing_extra_state() -> (
    None
):
    model_ir = ModelIR("remaining_qlinear_substitute")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics = [{"sentinel": "qlinear"}]
    context = ModelIRPassContext(model_ir, layout_state, diagnostics)

    invocations = build_qlinear_mean_concat_invocations(context)

    assert tuple(item.pass_id for item in invocations) == QLINEAR_MEAN_CONCAT_PASS_IDS
    assert all(item.args == (model_ir,) for item in invocations)
    assert all(item.keyword_args == () for item in invocations)
    assert all(
        value is not layout_state and value is not diagnostics
        for item in invocations
        for value in item.args
    )


def test_sinet_terminal_context_and_callback_contract_are_frozen() -> None:
    model_ir = ModelIR("remaining_sinet_terminal_context")
    layout_state = LayoutState.from_model_ir(model_ir)
    callback = _callback("preadd_resize")

    assert is_dataclass(SINetTerminalLayoutRecoveryContext)
    assert tuple(
        field.name for field in fields(SINetTerminalLayoutRecoveryContext)
    ) == ("pass_context", "preadd_resize_recovery")
    context = SINetTerminalLayoutRecoveryContext(
        pass_context=ModelIRPassContext(model_ir, layout_state, []),
        preadd_resize_recovery=callback,
    )
    assert context.pass_context.model_ir is model_ir
    assert context.pass_context.layout_state is layout_state
    assert context.preadd_resize_recovery is callback

    invocations = build_sinet_terminal_layout_recovery_invocations(context)
    assert (
        tuple(item.pass_id for item in invocations)
        == SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS
    )
    assert invocations[0].args == (model_ir,)
    assert invocations[0].keyword_args == (("layout_state", layout_state),)
    assert invocations[1].callback is callback
    assert invocations[1].args == ()
    assert invocations[1].keyword_args == ()
    assert invocations[2].args == (model_ir,)
    assert invocations[2].keyword_args == ()


def test_lowerer_remaining_context_wiring_is_explicit() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    context_names = {"SINetTerminalLayoutRecoveryContext"}
    calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in context_names
    ]

    assert len(calls) == 1
    contracts = {
        call.func.id: {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        }
        for call in calls
    }
    assert contracts == {
        "SINetTerminalLayoutRecoveryContext": {
            "pass_context": "session.model_ir_pass_context",
            "preadd_resize_recovery": "_run_sinet_preadd_resize_recovery_sequence",
        },
    }
    assert all(call.args == [] for call in calls)

    qlinear_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "qlinear_recovery_context"
    )
    assert _expression_path(qlinear_assignment.value) == "shared_model_ir_pass_context"
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "QLinearRecoveryContext"
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "qlinear_recovery_orchestration",
        "sinet_terminal_layout_recovery_orchestration",
    ),
)
def test_remaining_context_modules_are_diagnostics_free_and_lowerer_independent(
    module_name: str,
) -> None:
    tree = ast.parse((PASSES_ROOT / f"{module_name}.py").read_text(encoding="utf-8"))
    imported_modules = {
        str(node.module)
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "diagnostics" not in attributes
    assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
