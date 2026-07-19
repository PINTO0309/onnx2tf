from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_binary_repair_orchestration.py"
)
OWNER = "run_late_binary_repair_cleanup"
RESULT_TARGET = "_late_binary_repair_requires_reconciliation"
PHASE_ID = "shape_reconciliation.primary.late_binary_repair"
PREDECESSOR_GUARD = "_shared_late_requires_reconciliation"
SUCCESSOR_GUARD = (
    "optimize_layout_transpose_chains or "
    "apply_safe_transpose_reduction_lite_on_no_layout_opt"
)
TENSOR_COUNT_TARGET = "late_binary_repair_tensor_count"
EVIDENCE_TARGETS = (
    "late_signature_stats",
    "late_binary_adapter_stats",
    "late_singleton_adapter_stats",
)
PASS_IDS = (
    "sanitize_static_shape_signature_consistency",
    "run_indexed_binary_layout_adapter_cleanup",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _repair_guard(lowerer: ast.FunctionDef) -> tuple[int, ast.If]:
    index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and TENSOR_COUNT_TARGET in ast.unparse(statement.test)
    )
    guard = lowerer.body[index]
    assert isinstance(guard, ast.If)
    return index, guard


def test_late_binary_repair_evidence_and_guard_are_fixed() -> None:
    lowerer = _lowerer()
    guard_index, guard = _repair_guard(lowerer)
    assert _single_target(lowerer.body[guard_index - 3]) == TENSOR_COUNT_TARGET
    assert ast.unparse(lowerer.body[guard_index - 3].value) == (
        "len(model_ir.tensors)"
    )
    assert _single_target(lowerer.body[guard_index - 2]) == EVIDENCE_TARGETS[0]
    adapter_targets = lowerer.body[guard_index - 1].targets[0]
    assert isinstance(adapter_targets, ast.Tuple)
    assert tuple(
        target.id
        for target in adapter_targets.elts
        if isinstance(target, ast.Name)
    ) == EVIDENCE_TARGETS[1:]
    guard_source = ast.unparse(guard.test)
    for counter in (
        "sanitized_static_shape_signature_consistency",
        "inserted_rank4_binary_layout_fix_transpose",
        "repaired_rank4_binary_singleton_broadcast_layout_mismatch",
    ):
        assert counter in guard_source
    assert (
        "len(model_ir.tensors) < late_binary_repair_tensor_count"
        in guard_source
    )
    assert guard.orelse == []
    assert len(guard.body) == 1
    assert ast.unparse(guard.body[0]) == (
        "session.record_phase_result("
        f"'{PHASE_ID}', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )
    successor = lowerer.body[guard_index + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == SUCCESSOR_GUARD


@pytest.mark.xfail(
    strict=True,
    reason="late-binary repair decision lacks a focused pass-module owner",
)
def test_late_binary_repair_uses_one_boolean_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in sorted(
            (
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in PASS_IDS
            ),
            key=lambda node: node.lineno,
        )
    ]
    assert owner_calls == list(PASS_IDS)

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_binary_repair_cleanup(shared_model_ir_pass_context)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    guard = lowerer.body[index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == RESULT_TARGET
    assert len(guard.body) == 1
    assert ast.unparse(guard.body[0]) == (
        "session.record_phase_result("
        f"'{PHASE_ID}', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )
    successor = lowerer.body[index + 2]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == SUCCESSOR_GUARD
    assert not any(
        isinstance(node, ast.Name)
        and node.id in (TENSOR_COUNT_TARGET, *EVIDENCE_TARGETS)
        for node in ast.walk(lowerer)
    )
