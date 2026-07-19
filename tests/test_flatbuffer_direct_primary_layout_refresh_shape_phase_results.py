from __future__ import annotations

import ast
from pathlib import Path

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


def _records_for(
    lowerer: ast.FunctionDef,
    phase_ids: tuple[str, ...],
) -> list[ast.Expr]:
    return sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr) and _phase_id(node) in phase_ids
        ),
        key=lambda node: node.lineno,
    )


def test_three_final_layout_refresh_reconciliations_use_phase_results() -> None:
    lowerer = _lowerer()
    records = _records_for(lowerer, EXPECTED_PHASE_IDS)

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

    refresh_records = _records_for(lowerer, EXPECTED_REFRESH_PHASE_IDS)
    assert tuple(_phase_id(node) for node in refresh_records) == (
        EXPECTED_REFRESH_PHASE_IDS
    )
    assert all(
        record.lineno < refresh.lineno
        for record, refresh in zip(records, refresh_records)
    )


def test_three_final_layout_refresh_reconciliation_defaults_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
