from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import qlinear_recovery_orchestration
from onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration import (
    QLINEAR_MEAN_CONCAT_PASS_IDS,
    QLinearRecoveryContext,
    build_qlinear_mean_concat_invocations,
    run_qlinear_mean_concat_recovery,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
QLINEAR_MEAN_CONCAT = "_run_qlinear_mean_concat_recovery_sequence"


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
        if isinstance(node, ast.FunctionDef) and node.name == QLINEAR_MEAN_CONCAT
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


def _context() -> QLinearRecoveryContext:
    model_ir = ModelIR("qlinear_recovery_test")
    return QLinearRecoveryContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def _normalize_new_contract(
    invocation: qlinear_recovery_orchestration.RecoveryInvocation,
    context: QLinearRecoveryContext,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    def normalize(value: Any) -> Any:
        if value is context.model_ir:
            return "model_ir"
        return value

    return (
        tuple(normalize(value) for value in invocation.args),
        {key: normalize(value) for key, value in invocation.keyword_args},
    )


def test_qlinear_recovery_sequence_is_a_straight_line_closure() -> None:
    _, helper = _lowerer_and_helper()
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

    assert helper.end_lineno is not None
    assert helper.end_lineno - helper.lineno + 1 == 2
    assert helper.args.args == []
    assert helper.args.posonlyargs == []
    assert helper.args.kwonlyargs == []
    assert helper.args.vararg is None
    assert helper.args.kwarg is None
    assert not any(isinstance(node, control_flow_nodes) for node in ast.walk(helper))
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
    assert loaded_data_names == {"qlinear_recovery_context"}


def test_qlinear_recovery_preserves_exact_order_and_arguments() -> None:
    context = _context()
    invocations = build_qlinear_mean_concat_invocations(context)

    assert tuple(step.pass_id for step in invocations) == QLINEAR_MEAN_CONCAT_PASS_IDS
    assert {
        step.pass_id: _normalize_new_contract(step, context) for step in invocations
    } == {pass_id: (("model_ir",), {}) for pass_id in QLINEAR_MEAN_CONCAT_PASS_IDS}


def test_qlinear_recovery_invocations_remain_zero_argument() -> None:
    lowerer, _ = _lowerer_and_helper()
    invocations = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == QLINEAR_MEAN_CONCAT
    ]

    assert len(invocations) == 2
    assert all(call.args == [] for call in invocations)
    assert all(call.keywords == [] for call in invocations)


def test_qlinear_recovery_preserves_both_outer_boundaries() -> None:
    lowerer, _ = _lowerer_and_helper()
    boundaries: list[tuple[str, str]] = []
    for statement in lowerer.body:
        if not isinstance(statement, ast.If):
            continue
        for index, candidate in enumerate(statement.body):
            if not (
                isinstance(candidate, ast.Expr)
                and isinstance(candidate.value, ast.Call)
                and isinstance(candidate.value.func, ast.Name)
                and candidate.value.func.id == QLINEAR_MEAN_CONCAT
            ):
                continue
            previous = statement.body[index - 1]
            following = statement.body[index + 1]
            assert isinstance(previous, ast.Expr)
            assert isinstance(previous.value, ast.Call)
            assert isinstance(previous.value.func, ast.Name)
            assert isinstance(following, ast.Expr)
            assert isinstance(following.value, ast.Call)
            assert isinstance(following.value.func, ast.Name)
            boundaries.append((previous.value.func.id, following.value.func.id))

    assert boundaries == [
        (
            "_optimize_transpose_dequantize_mean_quantize_bridges",
            "_run_layout_reshape_attention_recovery_prefix",
        ),
        (
            "_set_post_progress_desc",
            "_run_layout_recovery_prefix_pass_sequence",
        ),
    ]


def test_qlinear_recovery_context_and_wrapper_are_explicit() -> None:
    lowerer, helper = _lowerer_and_helper()
    assert len(helper.body) == 1
    statement = helper.body[0]
    assert isinstance(statement, ast.Expr)
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == "run_qlinear_mean_concat_recovery"
    assert len(call.args) == 1
    assert isinstance(call.args[0], ast.Name)
    assert call.args[0].id == "qlinear_recovery_context"
    assert call.keywords == []

    context_assignment = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "qlinear_recovery_context"
            for target in statement.targets
        )
    )
    assert isinstance(context_assignment.value, ast.Name)
    assert context_assignment.value.id == "shared_model_ir_pass_context"


def test_qlinear_recovery_runner_preserves_instrumented_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = _context()
    probe_steps = build_qlinear_mean_concat_invocations(context)
    events: list[str] = []

    def recorder(pass_id: str):
        def record(*args: Any, **kwargs: Any) -> None:
            events.append(pass_id)

        return record

    for step in probe_steps:
        module_name = next(
            name
            for name, value in vars(qlinear_recovery_orchestration).items()
            if value is step.callback
        )
        monkeypatch.setattr(
            qlinear_recovery_orchestration,
            module_name,
            recorder(step.pass_id),
        )

    run_qlinear_mean_concat_recovery(context)

    assert events == list(QLINEAR_MEAN_CONCAT_PASS_IDS)


def test_qlinear_recovery_module_does_not_import_the_lowerer() -> None:
    module_path = (
        REPO_ROOT
        / "onnx2tf"
        / "tflite_builder"
        / "passes"
        / "qlinear_recovery_orchestration.py"
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
