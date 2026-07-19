from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_reconcile_static_tensor_shapes"
EXPECTED_RESULT_TARGETS = (
    "_final_sinet_late_residual_static_shape_stats",
    "_final_sinet_preadd_fanout_static_shape_stats",
    "_final_sinet_dual_resize_static_shape_stats",
    "_final_sinet_shared_post_static_shape_stats",
    "_final_sinet_deep_skip_static_shape_stats",
    "_final_sinet_concat_resize_static_shape_stats",
)
EXPECTED_PHASE_IDS = (
    "shape_reconciliation.primary.final_sinet_late_residual",
    "shape_reconciliation.primary.final_sinet_preadd_fanout",
    "shape_reconciliation.primary.final_sinet_dual_resize",
    "shape_reconciliation.primary.final_sinet_shared_post",
    "shape_reconciliation.primary.final_sinet_deep_skip",
    "shape_reconciliation.primary.final_sinet_concat_resize",
)
EXPECTED_REPAIR_TARGETS = (
    "final_sinet_late_residual_stats",
    "final_sinet_preadd_fanout_stats",
    "final_sinet_dual_resize_stats",
    "final_sinet_shared_post_stats",
    "final_sinet_deep_skip_stats",
    "final_sinet_concat_resize_stats",
)
ZERO_RESULT = {
    "reconciled_static_tensor_shapes": 0,
    "reconciled_static_shape_mutations": 0,
}


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


def test_six_final_sinet_reconciliations_form_one_ordered_family() -> None:
    lowerer = _lowerer()
    repair_assignments = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Assign)
            and _single_target(node) in EXPECTED_REPAIR_TARGETS
        ),
        key=lambda node: node.lineno,
    )
    assignments = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Assign)
            and _single_target(node) in EXPECTED_RESULT_TARGETS
        ),
        key=lambda node: node.lineno,
    )

    assert tuple(_single_target(node) for node in repair_assignments) == (
        EXPECTED_REPAIR_TARGETS
    )
    assert len(assignments) == 12
    defaults = assignments[::2]
    observations = assignments[1::2]
    assert tuple(_single_target(node) for node in defaults) == (
        EXPECTED_RESULT_TARGETS
    )
    assert tuple(_single_target(node) for node in observations) == (
        EXPECTED_RESULT_TARGETS
    )
    assert all(
        isinstance(node.value, ast.Dict)
        and ast.literal_eval(node.value) == ZERO_RESULT
        for node in defaults
    )
    for repair, observation in zip(repair_assignments, observations):
        assert repair.lineno < observation.lineno
        call = _statement_call(observation)
        assert call is not None
        assert isinstance(call.func, ast.Name)
        assert call.func.id == OWNER
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"include_mutation_count": "True"}
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="final SiNet reconciliations have not moved to phase results",
)
def test_six_final_sinet_reconciliations_use_phase_results() -> None:
    lowerer = _lowerer()
    records = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr)
            and (owner := _phase_result_owner(node)) is not None
            and isinstance(owner.func, ast.Name)
            and owner.func.id == OWNER
            and owner.args
            and ast.unparse(owner.args[0]) == "model_ir"
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
