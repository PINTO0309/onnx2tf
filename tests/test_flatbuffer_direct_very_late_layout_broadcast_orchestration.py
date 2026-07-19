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
    / "very_late_layout_broadcast_orchestration.py"
)
OWNER = "run_very_late_layout_broadcast_cleanup"
RESULT_TARGET = "_very_late_layout_broadcast_results"
PREDECESSOR_TARGET = "_very_late_singleton_consecutive_reshape_results"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_broadcast"
OLD_RESULT_TARGETS = (
    "_very_late_layout_transpose_cleanup_stats",
    "_very_late_broadcast_repair_stats",
)
PASS_IDS = (
    "run_layout_transpose_cleanup",
    "repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
)
LOWERER_PASS_IDS = (
    "run_layout_transpose_cleanup",
    "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
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


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


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


def test_very_late_layout_broadcast_boundary_is_ordered_and_unconsumed() -> None:
    lowerer = _lowerer()
    guard_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == "optimize_layout_transpose_chains"
        and any(
            _single_target(child) == OLD_RESULT_TARGETS[0]
            for child in statement.body
        )
    )
    guard = lowerer.body[guard_index]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 1
    layout_statement = guard.body[0]
    assert _single_target(layout_statement) == OLD_RESULT_TARGETS[0]
    assert _call_name(layout_statement) == LOWERER_PASS_IDS[0]
    layout_call = _statement_call(layout_statement)
    assert layout_call is not None
    assert [ast.unparse(argument) for argument in layout_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in layout_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    assert _single_target(lowerer.body[guard_index - 1]) == PREDECESSOR_TARGET
    broadcast_statement = lowerer.body[guard_index + 1]
    assert _single_target(broadcast_statement) == OLD_RESULT_TARGETS[1]
    assert _call_name(broadcast_statement) == LOWERER_PASS_IDS[1]
    broadcast_call = _statement_call(broadcast_statement)
    assert broadcast_call is not None
    assert [ast.unparse(argument) for argument in broadcast_call.args] == [
        "model_ir"
    ]
    assert broadcast_call.keywords == []
    assert _phase_id(lowerer.body[guard_index + 2]) == SUCCESSOR_PHASE_ID

    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    reason="very-late layout/broadcast boundary lacks a composite owner",
)
def test_very_late_layout_broadcast_boundary_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PASS_IDS
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
        "run_very_late_layout_broadcast_cleanup("
        "shared_model_ir_pass_context, "
        "include_layout_transpose=optimize_layout_transpose_chains)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
