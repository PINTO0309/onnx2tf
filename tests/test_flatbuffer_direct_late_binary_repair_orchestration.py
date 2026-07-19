from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import late_binary_repair_orchestration
from onnx2tf.tflite_builder.passes.late_binary_repair_orchestration import (
    LATE_BINARY_REPAIR_PASS_IDS,
)


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
SUCCESSOR_TARGET = (
    "_late_binary_layout_recovery_requires_reconciliation"
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


def test_late_binary_repair_boolean_keeps_reconciliation_in_lowerer() -> None:
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
    assert guard.orelse == []
    assert len(guard.body) == 1
    assert ast.unparse(guard.body[0]) == (
        "session.record_phase_result("
        f"'{PHASE_ID}', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )
    successor = lowerer.body[index + 2]
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in (TENSOR_COUNT_TARGET, *EVIDENCE_TARGETS)
        for node in ast.walk(lowerer)
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
    assert _single_target(successor) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in (TENSOR_COUNT_TARGET, *EVIDENCE_TARGETS)
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("trigger_index", range(-1, 4))
def test_late_binary_repair_owner_preserves_named_counters_and_prune_trigger(
    monkeypatch: pytest.MonkeyPatch,
    trigger_index: int,
) -> None:
    model_ir = ModelIR("late_binary_repair_owner")
    model_ir.tensors["prune_probe"] = TensorIR(
        name="prune_probe",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object]] = []

    def _signature_callback(candidate: ModelIR) -> dict[str, int]:
        observed.append((PASS_IDS[0], candidate))
        return {
            "sanitized_static_shape_signature_consistency": int(
                trigger_index == 0
            )
        }

    def _adapter_callback(
        candidate: ModelIR,
    ) -> tuple[dict[str, int], dict[str, int]]:
        observed.append((PASS_IDS[1], candidate))
        if trigger_index == 3:
            assert candidate.tensors.pop("prune_probe", None) is not None
        return (
            {
                "inserted_rank4_binary_layout_fix_transpose": int(
                    trigger_index == 1
                )
            },
            {
                "repaired_rank4_binary_singleton_broadcast_layout_mismatch": int(
                    trigger_index == 2
                )
            },
        )

    monkeypatch.setattr(
        late_binary_repair_orchestration,
        PASS_IDS[0],
        _signature_callback,
    )
    monkeypatch.setattr(
        late_binary_repair_orchestration,
        PASS_IDS[1],
        _adapter_callback,
    )

    assert (
        late_binary_repair_orchestration.run_late_binary_repair_cleanup(
            context
        )
        is (trigger_index >= 0)
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert LATE_BINARY_REPAIR_PASS_IDS == (
        "_sanitize_static_shape_signature_consistency",
        "run_indexed_binary_layout_adapter_cleanup",
    )
