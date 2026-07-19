from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.passes import (
    shared_late_reconciliation_orchestration,
)
from onnx2tf.tflite_builder.passes.shared_late_reconciliation_orchestration import (
    SHARED_LATE_RECONCILIATION_PASS_IDS,
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
    / "shared_late_reconciliation_orchestration.py"
)
OWNER = "run_shared_late_reconciliation_cleanup"
RESULT_TARGET = "_shared_late_requires_reconciliation"
PREDECESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_broadcast"
PHASE_ID = "shape_reconciliation.primary.shared_late"
SUCCESSOR_TARGET = "late_binary_repair_tensor_count"
TENSOR_COUNT_TARGET = "shared_late_tensor_count"
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
PASS_IDS = (
    "realign_dynamic_boundary_shape_signature_map",
    "sanitize_hardswish_tensor_shapes",
    "sanitize_squeeze_axes_with_static_input_shapes",
    "sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv",
    "run_indexed_binary_layout_adapter_cleanup",
    "run_singleton_consecutive_reshape",
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


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_shared_late_cleanup_boolean_keeps_reconciliation_in_lowerer() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_shared_late_reconciliation_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
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
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in (TENSOR_COUNT_TARGET, *EVIDENCE_TARGETS)
        for node in ast.walk(lowerer)
    )


def test_shared_late_cleanup_uses_one_boolean_owner() -> None:
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
        "run_shared_late_reconciliation_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _phase_id(lowerer.body[index - 1]) == PREDECESSOR_PHASE_ID
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
    assert _single_target(lowerer.body[index + 2]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in (TENSOR_COUNT_TARGET, *EVIDENCE_TARGETS)
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize("trigger_index", range(-1, 10))
def test_shared_late_owner_preserves_all_nine_evidence_and_prune_triggers(
    monkeypatch: pytest.MonkeyPatch,
    trigger_index: int,
) -> None:
    model_ir = ModelIR("shared_late_reconciliation_owner")
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

    def _direct_callback(pass_id: str, evidence_index: int):
        def _run(candidate: ModelIR) -> dict[str, int]:
            observed.append((pass_id, candidate))
            if evidence_index == 3 and trigger_index == 9:
                assert candidate.tensors.pop("prune_probe", None) is not None
            return {"changed": int(trigger_index == evidence_index)}

        return _run

    for evidence_index, pass_id in enumerate(PASS_IDS[:4]):
        monkeypatch.setattr(
            shared_late_reconciliation_orchestration,
            pass_id,
            _direct_callback(pass_id, evidence_index),
        )

    def _adapter_callback(
        candidate: ModelIR,
    ) -> tuple[dict[str, int], dict[str, int]]:
        observed.append((PASS_IDS[4], candidate))
        return (
            {"changed": int(trigger_index == 4)},
            {"changed": int(trigger_index == 5)},
        )

    def _reshape_callback(
        candidate: ModelIRPassContext,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int]]:
        observed.append((PASS_IDS[5], candidate))
        return (
            {"changed": int(trigger_index == 6)},
            {"changed": int(trigger_index == 7)},
            {"changed": int(trigger_index == 8)},
        )

    monkeypatch.setattr(
        shared_late_reconciliation_orchestration,
        PASS_IDS[4],
        _adapter_callback,
    )
    monkeypatch.setattr(
        shared_late_reconciliation_orchestration,
        PASS_IDS[5],
        _reshape_callback,
    )

    assert (
        shared_late_reconciliation_orchestration.run_shared_late_reconciliation_cleanup(
            context
        )
        is (trigger_index >= 0)
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed[:5])
    assert observed[5][1] is context
    assert SHARED_LATE_RECONCILIATION_PASS_IDS == (
        "_realign_dynamic_boundary_shape_signature_map",
        "_sanitize_hardswish_tensor_shapes",
        "_sanitize_squeeze_axes_with_static_input_shapes",
        "_sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv",
        "run_indexed_binary_layout_adapter_cleanup",
        "_run_singleton_consecutive_reshape_pass_cluster",
    )
