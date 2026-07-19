from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
RESULT_TARGET = "_post_split_fallback_static_shape_stats"
PHASE_ID = "shape_reconciliation.primary.post_split_fallback"


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


def _split_index(lowerer: ast.FunctionDef) -> int:
    return next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == "split_fallback_stats"
    )


def test_post_split_fallback_reconciliation_is_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    split_index = _split_index(lowerer)

    split = lowerer.body[split_index]
    assert isinstance(split, ast.Assign)
    assert ast.unparse(split.value) == (
        "_replace_unsupported_split_with_slice("
        "model_ir, layout_state=session.layout_state)"
    )

    default = lowerer.body[split_index + 1]
    assert _single_target(default) == RESULT_TARGET
    assert isinstance(default, ast.Assign)
    assert ast.literal_eval(default.value) == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    guard = lowerer.body[split_index + 2]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(split_fallback_stats.get("
        "'replaced_unsupported_split_with_slice', 0)) > 0"
    )
    assert guard.orelse == []
    assert len(guard.body) == 1
    assignment = guard.body[0]
    assert _single_target(assignment) == RESULT_TARGET
    assert isinstance(assignment, ast.Assign)
    assert ast.unparse(assignment.value) == (
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True)"
    )

    successor = lowerer.body[split_index + 3]
    assert _single_target(successor) == "unbound_inputs"
    assert isinstance(successor, ast.Assign)
    assert ast.unparse(successor.value) == (
        "_find_unbound_nonconstant_operator_inputs(model_ir)"
    )

    result_names = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Name) and node.id == RESULT_TARGET
    ]
    assert len(result_names) == 2
    assert all(isinstance(node.ctx, ast.Store) for node in result_names)


@pytest.mark.xfail(
    strict=True,
    reason="post-split fallback result has not moved to a phase record",
)
def test_post_split_fallback_reconciliation_uses_phase_result_store() -> None:
    lowerer = _lowerer()
    split_index = _split_index(lowerer)

    guard = lowerer.body[split_index + 1]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 1
    record = guard.body[0]
    assert isinstance(record, ast.Expr)
    assert ast.unparse(record) == (
        "session.record_phase_result("
        f"'{PHASE_ID}', "
        "_reconcile_static_tensor_shapes(model_ir, "
        "include_mutation_count=True))"
    )
    assert _single_target(lowerer.body[split_index + 2]) == "unbound_inputs"
    assert not any(
        isinstance(node, ast.Name) and node.id == RESULT_TARGET
        for node in ast.walk(lowerer)
    )
