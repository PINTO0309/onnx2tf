from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
from importlib import import_module
from pathlib import Path
from typing import Any, Callable

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PASSES_ROOT = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
MODEL_LAYOUT_CONTEXT_TYPES = (
    (
        "sinet_preadd_resize_recovery_orchestration",
        "SINetPreaddResizeRecoveryContext",
    ),
    ("quantized_recovery_orchestration", "QuantizedRecoveryContext"),
    (
        "terminal_affine_concat_split_recovery_orchestration",
        "TerminalAffineConcatSplitRecoveryContext",
    ),
)
BUILDER_CONTRACTS = (
    (
        "sinet_preadd_resize_recovery_orchestration",
        "build_sinet_preadd_resize_recovery_invocations",
        6,
        frozenset({2, 3, 4, 5}),
        frozenset(),
    ),
    (
        "quantized_recovery_orchestration",
        "build_safe_binary_recovery_invocations",
        1,
        frozenset({0}),
        frozenset(),
    ),
    (
        "quantized_recovery_orchestration",
        "build_quantized_activation_binary_invocations",
        6,
        frozenset({0, 1, 2, 3}),
        frozenset({5}),
    ),
    (
        "terminal_affine_concat_split_recovery_orchestration",
        "build_terminal_affine_concat_split_recovery_invocations",
        11,
        frozenset({0, 1, 6, 7, 8, 9}),
        frozenset(),
    ),
)


def _expression_path(expression: ast.expr) -> object:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return f"{_expression_path(expression.value)}.{expression.attr}"
    return type(expression).__name__


@pytest.mark.parametrize(("module_name", "context_name"), MODEL_LAYOUT_CONTEXT_TYPES)
def test_model_layout_contexts_share_one_frozen_identity_contract(
    module_name: str,
    context_name: str,
) -> None:
    context_type = getattr(
        import_module(f"onnx2tf.tflite_builder.passes.{module_name}"),
        context_name,
    )
    model_ir = ModelIR(f"model_layout_context_{module_name}")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []

    assert context_type is ModelIRPassContext
    assert is_dataclass(context_type)
    assert tuple(field.name for field in fields(context_type)) == (
        "model_ir",
        "layout_state",
        "diagnostics",
    )
    context = context_type(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    assert context.model_ir is model_ir
    assert context.layout_state is layout_state
    assert context.diagnostics is diagnostics
    with pytest.raises(FrozenInstanceError):
        context.model_ir = ModelIR("replacement")


def test_lowerer_model_layout_context_wiring_is_explicit() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    context_names = {context_name for _, context_name in MODEL_LAYOUT_CONTEXT_TYPES}
    assignments = {
        target.id: _expression_path(statement.value)
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
    }
    context_variables = {
        "QuantizedRecoveryContext": "quantized_recovery_context",
        "SINetPreaddResizeRecoveryContext": "sinet_preadd_resize_recovery_context",
        "TerminalAffineConcatSplitRecoveryContext": (
            "terminal_affine_concat_split_recovery_context"
        ),
    }

    assert set(context_variables) == context_names
    assert {
        variable_name: assignments[variable_name]
        for variable_name in context_variables.values()
    } == {
        variable_name: "shared_model_ir_pass_context"
        for variable_name in context_variables.values()
    }
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in context_names
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize(
    (
        "module_name",
        "builder_name",
        "expected_count",
        "layout_indices",
        "context_indices",
    ),
    BUILDER_CONTRACTS,
)
def test_model_ir_pass_context_is_a_diagnostics_inert_behavioral_substitute(
    module_name: str,
    builder_name: str,
    expected_count: int,
    layout_indices: frozenset[int],
    context_indices: frozenset[int],
) -> None:
    module = import_module(f"onnx2tf.tflite_builder.passes.{module_name}")
    builder: Callable[[Any], tuple[Any, ...]] = getattr(module, builder_name)
    model_ir = ModelIR(f"model_layout_substitute_{builder_name}")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics = [{"sentinel": builder_name}]
    context = ModelIRPassContext(model_ir, layout_state, diagnostics)

    invocations = builder(context)

    assert len(invocations) == expected_count
    for index, invocation in enumerate(invocations):
        expected_args = (context,) if index in context_indices else (model_ir,)
        expected_keywords = (
            (("layout_state", layout_state),) if index in layout_indices else ()
        )
        assert invocation.args == expected_args
        assert invocation.keyword_args == expected_keywords
        assert all(value is not diagnostics for value in invocation.args)
        assert all(value is not diagnostics for _, value in invocation.keyword_args)


def test_historical_partial_contexts_use_their_expected_common_shapes() -> None:
    qlinear_module = import_module(
        "onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration"
    )
    sinet_terminal_module = import_module(
        "onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration"
    )

    assert qlinear_module.QLinearRecoveryContext is ModelIRPassContext
    assert tuple(
        field.name for field in fields(qlinear_module.QLinearRecoveryContext)
    ) == ("model_ir", "layout_state", "diagnostics")
    assert tuple(
        field.name
        for field in fields(sinet_terminal_module.SINetTerminalLayoutRecoveryContext)
    ) == ("pass_context", "preadd_resize_recovery")


@pytest.mark.parametrize(("module_name", "_"), MODEL_LAYOUT_CONTEXT_TYPES)
def test_model_layout_context_modules_are_diagnostics_free_and_lowerer_independent(
    module_name: str,
    _: str,
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
