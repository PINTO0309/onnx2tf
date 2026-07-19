from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_reconcile_static_tensor_shapes"
EXPECTED_RESULT_TARGETS = (
    "_final_convinteger_static_shape_stats",
    "_final_instancenorm_static_shape_stats",
    "_final_broadcast_static_shape_stats",
)
EXPECTED_PHASE_IDS = (
    "shape_reconciliation.primary.final_convinteger",
    "shape_reconciliation.primary.final_instancenorm",
    "shape_reconciliation.primary.final_broadcast",
)
EXPECTED_REFRESH_PHASE_IDS = (
    "topology_layout.primary.final_convinteger",
    "topology_layout.primary.final_instancenorm",
    "topology_layout.primary.final_broadcast",
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


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
        or not isinstance(call.args[0], ast.Constant)
        or not isinstance(call.args[0].value, str)
    ):
        return None
    return call.args[0].value


def _phase_result_owner(statement: ast.stmt) -> ast.Call | None:
    call = _statement_call(statement)
    if _phase_id(statement) is None or not isinstance(call.args[1], ast.Call):
        return None
    return call.args[1]


def test_three_final_layout_refresh_reconciliations_are_homogeneous() -> None:
    lowerer = _lowerer()
    assignments = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Assign)
            and _single_target(node) in EXPECTED_RESULT_TARGETS
        ),
        key=lambda node: node.lineno,
    )

    assert len(assignments) == 6
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
    for observation in observations:
        call = _statement_call(observation)
        assert call is not None
        assert isinstance(call.func, ast.Name)
        assert call.func.id == OWNER
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"include_mutation_count": "True"}

    refresh_records = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr)
            and _phase_id(node) in EXPECTED_REFRESH_PHASE_IDS
        ),
        key=lambda node: node.lineno,
    )
    assert tuple(_phase_id(node) for node in refresh_records) == (
        EXPECTED_REFRESH_PHASE_IDS
    )
    assert all(
        observation.lineno < refresh.lineno
        for observation, refresh in zip(observations, refresh_records)
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


@pytest.mark.xfail(
    strict=True,
    reason="final layout-refresh reconciliations have not moved to phase results",
)
def test_three_final_layout_refresh_reconciliations_use_phase_results() -> None:
    lowerer = _lowerer()
    records = sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr) and _phase_id(node) in EXPECTED_PHASE_IDS
        ),
        key=lambda node: node.lineno,
    )

    assert tuple(_phase_id(node) for node in records) == EXPECTED_PHASE_IDS
    for record in records:
        owner = _phase_result_owner(record)
        assert owner is not None
        assert isinstance(owner.func, ast.Name)
        assert owner.func.id == OWNER
        assert [ast.unparse(argument) for argument in owner.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in owner.keywords
        } == {"include_mutation_count": "True"}
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
