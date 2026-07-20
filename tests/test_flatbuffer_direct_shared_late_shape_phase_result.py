from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RESULT_TARGET = "_shared_late_static_shape_stats"
DECISION_TARGET = "_shared_late_requires_reconciliation"
PHASE_ID = "shape_reconciliation.primary.shared_late"
EVIDENCE_TARGETS = (
    "shared_boundary_signature_stats",
    "shared_hardswish_stats",
    "shared_squeeze_stats",
    "shared_conv_transpose_stats",
    "shared_binary_adapter_stats",
    "shared_singleton_adapter_stats",
    "shared_singleton_channel_stats",
    "shared_duplicate_fanout_stats",
    "shared_consecutive_reshape_stats",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _shared_late_guard(lowerer: ast.FunctionDef) -> tuple[int, ast.If]:
    guard_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == DECISION_TARGET
    )
    guard = lowerer.body[guard_index]
    assert isinstance(guard, ast.If)
    return guard_index, guard


def test_shared_late_reconciliation_is_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    guard_index, guard = _shared_late_guard(lowerer)

    assert ast.unparse(guard.test) == DECISION_TARGET
    assert guard.orelse == []
    assert len(guard.body) == 1
    record = guard.body[0]
    assert isinstance(record, ast.Expr)
    assert ast.unparse(record) == (
        "session.record_phase_result("
        f"'{PHASE_ID}', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )

    decision = lowerer.body[guard_index - 1]
    assert _single_target(decision) == DECISION_TARGET
    assert ast.unparse(decision.value) == (
        "run_shared_late_reconciliation_cleanup("
        "shared_model_ir_pass_context)"
    )
    successor = lowerer.body[guard_index + 1]
    assert _single_target(successor) == (
        "_late_binary_repair_requires_reconciliation"
    )
    assert ast.unparse(successor.value) == (
        "run_late_binary_repair_cleanup(shared_model_ir_pass_context)"
    )

    assert not any(
        isinstance(node, ast.Name)
        and node.id in (RESULT_TARGET, "shared_late_tensor_count", *EVIDENCE_TARGETS)
        for node in ast.walk(lowerer)
    )


def test_shared_late_reconciliation_local_is_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
