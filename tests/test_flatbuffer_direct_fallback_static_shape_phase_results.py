from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "_reconcile_static_tensor_shapes"
EXPECTED_RESULT_TARGETS = (
    "_fallback_broadcast_static_shape_stats",
    "_fallback_se_fc_gather_static_shape_stats",
    "_fallback_placeholder_matmul_static_shape_stats",
    "_fallback_conv_input_static_shape_stats",
    "_fallback_mixed_concat_static_shape_stats",
    "_fallback_concat_axis_static_shape_stats",
    "_fallback_binary_layout_static_shape_stats",
)
EXPECTED_PHASE_IDS = (
    "shape_reconciliation.fallback.broadcast",
    "shape_reconciliation.fallback.se_fc_gather",
    "shape_reconciliation.fallback.placeholder_matmul",
    "shape_reconciliation.fallback.conv_input",
    "shape_reconciliation.fallback.mixed_concat",
    "shape_reconciliation.fallback.concat_axis",
    "shape_reconciliation.fallback.binary_layout",
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


def _fallback_static_shape_records(
    lowerer: ast.FunctionDef,
) -> list[ast.Expr]:
    return sorted(
        (
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr)
            and (owner := _phase_result_owner(node)) is not None
            and isinstance(owner.func, ast.Name)
            and owner.func.id == OWNER
            and owner.args
            and ast.unparse(owner.args[0]) == "fallback_ir"
        ),
        key=lambda node: node.lineno,
    )


def test_seven_fallback_static_shape_phase_records_are_homogeneous() -> None:
    records = _fallback_static_shape_records(_lowerer())

    assert len(records) == 7
    assert tuple(
        ast.literal_eval(_statement_call(node).args[0]) for node in records
    ) == EXPECTED_PHASE_IDS
    for record in records:
        owner = _phase_result_owner(record)
        assert owner is not None
        assert [ast.unparse(argument) for argument in owner.args] == ["fallback_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in owner.keywords
        } == {"include_mutation_count": "True"}


def test_seven_fallback_static_shape_locals_and_defaults_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
