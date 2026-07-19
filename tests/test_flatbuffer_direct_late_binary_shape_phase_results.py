from __future__ import annotations

import ast
from pathlib import Path

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


def _parents(lowerer: ast.FunctionDef) -> dict[ast.AST, ast.AST]:
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(lowerer):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _phase_records(lowerer: ast.FunctionDef) -> list[ast.Expr]:
    return sorted(
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


def test_two_late_binary_reconciliations_are_guarded_and_unconsumed() -> None:
    lowerer = _lowerer()
    parents = _parents(lowerer)
    records = _phase_records(lowerer)

    assert tuple(
        ast.literal_eval(_statement_call(node).args[0]) for node in records
    ) == EXPECTED_PHASE_IDS
    for record in records:
        owner = _phase_result_owner(record)
        assert owner is not None
        assert [ast.unparse(argument) for argument in owner.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in owner.keywords
        } == {"include_mutation_count": "True"}
        assert isinstance(parents[record], ast.If)

    repair_guard = parents[records[0]]
    assert isinstance(repair_guard, ast.If)
    assert ast.unparse(repair_guard.test) == (
        "_late_binary_repair_requires_reconciliation"
    )
    recovery_guard = parents[records[1]]
    assert isinstance(recovery_guard, ast.If)
    assert ast.unparse(recovery_guard.test) == (
        "_late_binary_layout_recovery_requires_reconciliation"
    )
    recovery_decision = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id
        == "_late_binary_layout_recovery_requires_reconciliation"
    )
    assert ast.unparse(recovery_decision.value) == (
        "run_optional_late_binary_layout_recovery_cleanup("
        "shared_model_ir_pass_context, "
        "enabled=optimize_layout_transpose_chains or "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    assert not any(
        isinstance(node, ast.Name)
        and node.id in EXPECTED_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_two_late_binary_reconciliation_locals_are_removed() -> None:
    lowerer = _lowerer()

    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
