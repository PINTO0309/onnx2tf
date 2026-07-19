from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_reconcile_static_tensor_shapes"
EXPECTED_RESULT_TARGETS = (
    "_late_binary_repair_static_shape_stats",
    "_late_binary_layout_recovery_static_shape_stats",
)
EXPECTED_PHASE_IDS = (
    "shape_reconciliation.primary.late_binary_repair",
    "shape_reconciliation.primary.late_binary_layout_recovery",
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


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _phase_result_owner(statement: ast.stmt) -> ast.Call | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
        or not isinstance(call.args[1], ast.Call)
    ):
        return None
    return call.args[1]


def test_two_late_binary_reconciliations_are_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(lowerer):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    assignments = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Assign)
            and _single_target(node) in EXPECTED_RESULT_TARGETS
        ),
        key=lambda node: node.lineno,
    )

    assert tuple(_single_target(node) for node in assignments) == (
        EXPECTED_RESULT_TARGETS
    )
    for assignment in assignments:
        call = _statement_call(assignment)
        assert call is not None
        assert isinstance(call.func, ast.Name)
        assert call.func.id == OWNER
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"include_mutation_count": "True"}
        assert isinstance(parents[assignment], ast.If)

    repair_guard = parents[assignments[0]]
    assert isinstance(repair_guard, ast.If)
    assert "len(model_ir.tensors) < late_binary_repair_tensor_count" in (
        ast.unparse(repair_guard.test)
    )
    recovery_guard = parents[assignments[1]]
    assert isinstance(recovery_guard, ast.If)
    assert ast.unparse(recovery_guard.test) == (
        "_stats_have_positive_count(late_binary_layout_recovery_stats)"
    )
    outer_recovery_guard = parents[recovery_guard]
    assert isinstance(outer_recovery_guard, ast.If)
    assert ast.unparse(outer_recovery_guard.test) == (
        "optimize_layout_transpose_chains or "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="late binary reconciliation results have not moved to phase records",
)
def test_two_late_binary_reconciliations_use_phase_results() -> None:
    lowerer = _lowerer()
    records = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr)
            and (owner := _phase_result_owner(node)) is not None
            and isinstance(owner.func, ast.Name)
            and owner.func.id == OWNER
            and ast.literal_eval(_statement_call(node).args[0])
            in EXPECTED_PHASE_IDS
        ),
        key=lambda node: node.lineno,
    )

    assert tuple(
        ast.literal_eval(_statement_call(node).args[0]) for node in records
    ) == EXPECTED_PHASE_IDS
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
