from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    optional_late_binary_layout_recovery_orchestration,
)
from onnx2tf.tflite_builder.passes.optional_late_binary_layout_recovery_orchestration import (
    OPTIONAL_LATE_BINARY_LAYOUT_RECOVERY_PASS_IDS,
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
    / "optional_late_binary_layout_recovery_orchestration.py"
)
OWNER = "run_optional_late_binary_layout_recovery_cleanup"
RESULT_TARGET = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
OLD_RESULT_TARGET = "late_binary_layout_recovery_stats"
PREDECESSOR_GUARD = "_late_binary_repair_requires_reconciliation"
SUCCESSOR_TARGET = (
    "_pre_terminal_affine_slice_spp_results"
)
PHASE_ID = "shape_reconciliation.primary.late_binary_layout_recovery"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr):
        return None
    call = statement.value
    if (
        not isinstance(call, ast.Call)
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_optional_late_binary_layout_recovery_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    assignment_index = lowerer.body.index(assignment)
    assert isinstance(assignment, ast.Assign)
    assert ast.unparse(assignment.value) == (
        "run_optional_late_binary_layout_recovery_cleanup("
        "shared_model_ir_pass_context, "
        "enabled=optimize_layout_transpose_chains or "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )

    guard = lowerer.body[assignment_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == RESULT_TARGET
    assert guard.orelse == []
    assert len(guard.body) == 1
    assert _phase_id(guard.body[0]) == PHASE_ID
    assert ast.unparse(guard.body[0]) == (
        "session.record_phase_result("
        "'shape_reconciliation.primary.late_binary_layout_recovery', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )

    predecessor = lowerer.body[assignment_index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[assignment_index + 2]) == (
        SUCCESSOR_TARGET
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == OLD_RESULT_TARGET
        for node in ast.walk(lowerer)
    )


def test_optional_late_binary_layout_recovery_uses_boolean_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_late_binary_layout_recovery"
    ]
    assert len(owner_calls) == 1
    assert ast.unparse(owner_calls[0].args[0]) == "context.model_ir"

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    assignment_index = lowerer.body.index(assignment)
    assert isinstance(assignment, ast.Assign)
    assert ast.unparse(assignment.value) == (
        "run_optional_late_binary_layout_recovery_cleanup("
        "shared_model_ir_pass_context, "
        "enabled=optimize_layout_transpose_chains or "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )

    predecessor = lowerer.body[assignment_index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    guard = lowerer.body[assignment_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == RESULT_TARGET
    assert guard.orelse == []
    assert len(guard.body) == 1
    assert _phase_id(guard.body[0]) == PHASE_ID
    assert _single_target(lowerer.body[assignment_index + 2]) == (
        SUCCESSOR_TARGET
    )
    assert not any(
        isinstance(node, ast.Name) and node.id == OLD_RESULT_TARGET
        for node in ast.walk(lowerer)
    )


@pytest.mark.parametrize(
    (
        "enabled",
        "include_layout_transpose",
        "mutation_count",
        "expected_invocations",
        "expected_result",
    ),
    (
        (False, True, 1, 0, False),
        (True, False, 0, 1, False),
        (True, False, 1, 1, True),
        (True, True, 1, 1, True),
    ),
)
def test_optional_late_binary_layout_recovery_owner_preserves_decision(
    monkeypatch: pytest.MonkeyPatch,
    enabled: bool,
    include_layout_transpose: bool,
    mutation_count: int,
    expected_invocations: int,
    expected_result: bool,
) -> None:
    model_ir = ModelIR("optional_late_binary_layout_recovery")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[object, bool, object, object]] = []

    def _recovery_callback(
        candidate: ModelIR,
        *,
        include_layout_transpose: bool,
        layout_state: object,
        diagnostics: object,
    ) -> dict[str, int]:
        observed.append(
            (
                candidate,
                include_layout_transpose,
                layout_state,
                diagnostics,
            )
        )
        return {"mutation": mutation_count}

    monkeypatch.setattr(
        optional_late_binary_layout_recovery_orchestration,
        "run_late_binary_layout_recovery",
        _recovery_callback,
    )

    assert (
        optional_late_binary_layout_recovery_orchestration.run_optional_late_binary_layout_recovery_cleanup(
            context,
            enabled=enabled,
            include_layout_transpose=include_layout_transpose,
        )
        is expected_result
    )
    assert len(observed) == expected_invocations
    if observed:
        candidate, include_layout, layout_state, diagnostics = observed[0]
        assert candidate is context.model_ir
        assert include_layout is include_layout_transpose
        assert layout_state is context.layout_state
        assert diagnostics is context.diagnostics
    assert OPTIONAL_LATE_BINARY_LAYOUT_RECOVERY_PASS_IDS == (
        "run_late_binary_layout_recovery",
    )
