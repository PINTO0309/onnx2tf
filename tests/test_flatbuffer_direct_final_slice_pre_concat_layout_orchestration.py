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
    / "final_slice_pre_concat_layout_orchestration.py"
)
OWNER = "run_final_slice_pre_concat_layout_cleanup"
RESULT_TARGET = "_final_slice_pre_concat_layout_results"
PREDECESSOR_TARGET = "_final_slice_concat_recovery_results"
SUCCESSOR_TARGET = "_terminal_concat_bridge_layout_results"
OLD_RESULT_TARGETS = (
    "_final_slice_prepost_passthrough_stats",
    "_final_pre_concat_stats",
)
PASS_IDS = (
    "optimize_transpose_slice_prepost_nhwc_passthrough_chains",
    "optimize_transpose_pre_concat_nhwc_chains",
)
LOWERER_PASS_IDS = (
    "_optimize_transpose_slice_prepost_nhwc_passthrough_chains",
    "_optimize_transpose_pre_concat_nhwc_chains",
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


def test_final_slice_pre_concat_pair_is_adjacent_and_unconsumed() -> None:
    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) in OLD_RESULT_TARGETS
    ]
    assert tuple(_single_target(statement) for statement in assignments) == (
        OLD_RESULT_TARGETS
    )
    indices = [lowerer.body.index(statement) for statement in assignments]
    assert indices == [indices[0], indices[0] + 1]
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[1] + 1]) == SUCCESSOR_TARGET
    assert tuple(_call_name(statement) for statement in assignments) == (
        LOWERER_PASS_IDS
    )

    first_call = _statement_call(assignments[0])
    second_call = _statement_call(assignments[1])
    assert first_call is not None
    assert second_call is not None
    assert [ast.unparse(argument) for argument in first_call.args] == [
        "model_ir"
    ]
    assert first_call.keywords == []
    assert [ast.unparse(argument) for argument in second_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in second_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }
    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    reason="final Slice/pre-Concat pair has not moved to one composite owner",
)
def test_final_slice_pre_concat_pair_uses_one_composite_owner() -> None:
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
        "run_final_slice_pre_concat_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
