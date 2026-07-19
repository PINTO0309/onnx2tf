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


def _shared_late_guard(lowerer: ast.FunctionDef) -> tuple[int, ast.If]:
    index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and TENSOR_COUNT_TARGET in ast.unparse(statement.test)
    )
    guard = lowerer.body[index]
    assert isinstance(guard, ast.If)
    return index, guard


def test_shared_late_cleanup_evidence_and_guard_are_fixed() -> None:
    lowerer = _lowerer()
    guard_index, guard = _shared_late_guard(lowerer)
    evidence_assignments = lowerer.body[guard_index - 6 : guard_index]
    assert tuple(
        node.id
        for statement in evidence_assignments
        for node in ast.walk(statement.targets[0])
        if isinstance(node, ast.Name)
    ) == EVIDENCE_TARGETS
    tensor_count = lowerer.body[guard_index - 7]
    assert _single_target(tensor_count) == TENSOR_COUNT_TARGET
    assert ast.unparse(tensor_count.value) == "len(model_ir.tensors)"
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
    assert ast.unparse(guard.body[0]) == (
        "session.record_phase_result("
        f"'{PHASE_ID}', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )
    assert _single_target(lowerer.body[guard_index + 1]) == SUCCESSOR_TARGET


@pytest.mark.xfail(
    strict=True,
    reason="shared-late cleanup decision lacks a focused pass-module owner",
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
