from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RESULT_TARGET = "_shared_late_static_shape_stats"
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
        and "shared_late_tensor_count" in ast.unparse(statement.test)
    )
    guard = lowerer.body[guard_index]
    assert isinstance(guard, ast.If)
    return guard_index, guard


def test_shared_late_reconciliation_is_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    guard_index, guard = _shared_late_guard(lowerer)

    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count("
        "shared_boundary_signature_stats, shared_hardswish_stats, "
        "shared_squeeze_stats, shared_conv_transpose_stats, "
        "shared_binary_adapter_stats, shared_singleton_adapter_stats, "
        "shared_singleton_channel_stats, shared_duplicate_fanout_stats, "
        "shared_consecutive_reshape_stats) or "
        "len(model_ir.tensors) < shared_late_tensor_count"
    )
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

    evidence_assignments = lowerer.body[guard_index - 6 : guard_index]
    assert tuple(
        node.id
        for statement in evidence_assignments
        for node in ast.walk(statement.targets[0])
        if isinstance(node, ast.Name)
    ) == EVIDENCE_TARGETS
    tensor_boundary = lowerer.body[guard_index - 7]
    assert _single_target(tensor_boundary) == "shared_late_tensor_count"
    assert ast.unparse(tensor_boundary.value) == "len(model_ir.tensors)"
    successor = lowerer.body[guard_index + 1]
    assert _single_target(successor) == "late_binary_repair_tensor_count"
    assert ast.unparse(successor.value) == "len(model_ir.tensors)"

    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )


def test_shared_late_reconciliation_local_is_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
