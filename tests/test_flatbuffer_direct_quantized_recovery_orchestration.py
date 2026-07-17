from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import quantized_recovery_orchestration
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    QUANTIZED_ACTIVATION_BINARY_PASS_IDS,
    SAFE_BINARY_RECOVERY_PASS_IDS,
    QuantizedRecoveryContext,
    build_quantized_activation_binary_invocations,
    build_safe_binary_recovery_invocations,
    run_quantized_activation_binary_recovery,
    run_safe_binary_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
SAFE_BINARY = "_run_safe_binary_bridge_recovery_sequence"
QUANTIZED_ACTIVATION_BINARY = (
    "_run_quantized_activation_binary_bridge_recovery_sequence"
)


def _lowerer_and_helper(helper_name: str) -> tuple[ast.FunctionDef, ast.FunctionDef]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    helper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
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


def _context() -> QuantizedRecoveryContext:
    model_ir = ModelIR("quantized_recovery_orchestration_test")
    return QuantizedRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: quantized_recovery_orchestration.RecoveryInvocation,
    context: QuantizedRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context:
            return "context"
        if value is context.model_ir:
            return "model_ir"
        if value is context.layout_state:
            return "session.layout_state"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_quantized_recovery_sequences_are_straight_line_closures() -> None:
    expected_lines = {
        SAFE_BINARY: 2,
        QUANTIZED_ACTIVATION_BINARY: 4,
    }
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
    for helper_name, line_count in expected_lines.items():
        _, helper = _lowerer_and_helper(helper_name)

        assert helper.end_lineno is not None
        assert helper.end_lineno - helper.lineno + 1 == line_count
        assert helper.args.args == []
        assert helper.args.posonlyargs == []
        assert helper.args.kwonlyargs == []
        assert helper.args.vararg is None
        assert helper.args.kwarg is None
        assert not any(
            isinstance(node, control_flow_nodes) for node in ast.walk(helper)
        )
        assert not any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "ModelIRPassStateScope"
            for node in ast.walk(helper)
        )

        called_names = {
            node.func.id
            for node in ast.walk(helper)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        loaded_data_names = {
            node.id
            for node in ast.walk(helper)
            if isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id not in called_names
        }
        assert loaded_data_names == {"quantized_recovery_context"}


def test_safe_binary_recovery_preserves_exact_arguments() -> None:
    context = _context()
    invocations = build_safe_binary_recovery_invocations(context)

    assert tuple(step.pass_id for step in invocations) == SAFE_BINARY_RECOVERY_PASS_IDS
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {
        "_run_safe_binary_bridge_recovery_pass": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
    }


def test_quantized_activation_binary_preserves_exact_order_and_arguments() -> None:
    context = _context()
    invocations = build_quantized_activation_binary_invocations(context)
    contracts = {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    }

    assert (
        tuple(step.pass_id for step in invocations)
        == QUANTIZED_ACTIVATION_BINARY_PASS_IDS
    )
    assert contracts == {
        "_optimize_dequant_hardsigmoid_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_maxpool_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_softmax_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_optimize_dequant_logistic_quantize_chains": (
            ("model_ir",),
            {"layout_state": "session.layout_state"},
        ),
        "_canonicalize_softmax_transpose_chains": (("model_ir",), {}),
        SAFE_BINARY: (("context",), {}),
    }


def test_quantized_recovery_invocation_boundaries_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)
    expected_counts = {
        SAFE_BINARY: 3,
        QUANTIZED_ACTIVATION_BINARY: 2,
    }

    for helper_name, expected_count in expected_counts.items():
        invocations = [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == helper_name
        ]
        orchestrated_count = 0
        if helper_name == SAFE_BINARY:
            orchestrated_count = QUANTIZED_ACTIVATION_BINARY_PASS_IDS.count(helper_name)
        assert len(invocations) + orchestrated_count == expected_count
        assert all(call.args == [] for call in invocations)
        assert all(call.keywords == [] for call in invocations)


def test_quantized_recovery_context_and_wrappers_are_explicit() -> None:
    lowerer, safe_helper = _lowerer_and_helper(SAFE_BINARY)
    _, quantized_helper = _lowerer_and_helper(QUANTIZED_ACTIVATION_BINARY)
    expected_wrappers = {
        SAFE_BINARY: (safe_helper, "run_safe_binary_recovery"),
        QUANTIZED_ACTIVATION_BINARY: (
            quantized_helper,
            "run_quantized_activation_binary_recovery",
        ),
    }
    for helper_name, (helper, runner_name) in expected_wrappers.items():
        assert len(helper.body) == 1
        statement = helper.body[0]
        assert isinstance(statement, ast.Expr)
        call = statement.value
        assert isinstance(call, ast.Call)
        assert isinstance(call.func, ast.Name)
        assert call.func.id == runner_name
        assert len(call.args) == 1
        assert isinstance(call.args[0], ast.Name)
        assert call.args[0].id == "quantized_recovery_context"
        assert call.keywords == []
        assert helper.name == helper_name

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "quantized_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


@pytest.mark.parametrize(
    ("build_invocations", "run_phase", "expected_ids"),
    [
        (
            build_safe_binary_recovery_invocations,
            run_safe_binary_recovery,
            SAFE_BINARY_RECOVERY_PASS_IDS,
        ),
        (
            build_quantized_activation_binary_invocations,
            run_quantized_activation_binary_recovery,
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS,
        ),
    ],
)
def test_quantized_recovery_runners_preserve_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
    build_invocations: Any,
    run_phase: Any,
    expected_ids: tuple[str, ...],
) -> None:
    context = _context()
    probe_steps = build_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(quantized_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            quantized_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_phase(context)

    assert events == list(expected_ids)


def test_quantized_recovery_module_does_not_import_the_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "quantized_recovery_orchestration.py"
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
