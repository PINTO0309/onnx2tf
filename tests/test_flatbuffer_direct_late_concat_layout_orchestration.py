from __future__ import annotations

import ast
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_concat_layout_orchestration.py"
)
OWNER = "run_late_concat_layout_cleanup"
RESULT_TARGET = "_late_concat_layout_results"
SCOPE_TARGET = "late_concat_layout_state_scope"
PREDECESSOR_TARGET = "_late_cost_volume_conv_affine_stats"
OLD_RESULT_TARGETS = (
    "_late_concat_axis3_const_layout_stats",
    "_late_concat_dequant_quantize_layout_stats",
    "_late_concat_layernorm_layout_stats",
    "_late_concat_transpose_layout_stats",
)
PASS_IDS = (
    "run_axis3_const_concat_layout_cleanup",
    "run_dequant_concat_quantize_layout_cleanup",
    "run_layernorm_statistics_layout_cleanup",
    "run_layout_transpose_cleanup",
)


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


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def test_late_concat_layout_cluster_is_shared_ordered_and_unconsumed() -> None:
    lowerer = _lowerer()
    scope_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == SCOPE_TARGET
    )
    assert _single_target(lowerer.body[scope_index - 1]) == PREDECESSOR_TARGET

    for offset, (target, owner) in enumerate(
        zip(OLD_RESULT_TARGETS, PASS_IDS, strict=True),
        start=1,
    ):
        statement = lowerer.body[scope_index + offset]
        assert _single_target(statement) == target
        assert _call_name(statement) == owner
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
            "state_scope": SCOPE_TARGET,
        }

    successor = lowerer.body[scope_index + len(PASS_IDS) + 1]
    assert isinstance(successor, ast.If)
    assert ast.unparse(successor.test) == "optimize_layout_transpose_chains"
    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )
    assert sum(
        isinstance(node, ast.Name)
        and node.id == SCOPE_TARGET
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    ) == 4


@pytest.mark.xfail(
    strict=True,
    reason="late concat cluster has not moved to one composite owner",
)
def test_late_concat_layout_cluster_uses_one_composite_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {"ModelIRPassStateScope", *PASS_IDS}
    ]
    assert owner_calls == ["ModelIRPassStateScope", *PASS_IDS]

    lowerer = _lowerer()
    assignments = [
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    ]
    assert len(assignments) == 1
    assignment = assignments[0]
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_late_concat_layout_cleanup(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert isinstance(lowerer.body[index + 1], ast.If)
    assert not any(
        isinstance(node, ast.Name)
        and node.id in {SCOPE_TARGET, *OLD_RESULT_TARGETS}
        for node in ast.walk(lowerer)
    )
