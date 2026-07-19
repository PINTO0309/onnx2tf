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
    / "pre_terminal_instancenorm_layout_orchestration.py"
)
OWNER = "run_pre_terminal_instancenorm_layout_cleanup"
RESULT_TARGET = "_pre_terminal_instancenorm_layout_results"
PREDECESSOR_GUARD = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
SUCCESSOR_TARGET = "pre_terminal_affine_tensor_count"
OLD_RESULT_TARGETS = (
    "_pre_terminal_affine_instancenorm_post_bias_stats",
    "_pre_terminal_affine_instancenorm_residual_mul_concat_stats",
    "_pre_terminal_affine_instancenorm_dualstats_stats",
)
PASS_IDS = (
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
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


def _call_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign):
        return None
    call = statement.value
    if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def test_pre_terminal_instancenorm_layout_boundary_is_fixed() -> None:
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
    assert indices == list(range(indices[0], indices[0] + len(assignments)))
    assert tuple(_call_name(statement) for statement in assignments) == PASS_IDS
    for statement in assignments:
        assert isinstance(statement, ast.Assign)
        assert isinstance(statement.value, ast.Call)
        assert [
            ast.unparse(argument) for argument in statement.value.args
        ] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in statement.value.keywords
        } == {"layout_state": "session.layout_state"}

    predecessor = lowerer.body[indices[0] - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    for target in OLD_RESULT_TARGETS:
        assert sum(
            isinstance(node, ast.Name) and node.id == target
            for node in ast.walk(lowerer)
        ) == 1


@pytest.mark.xfail(
    strict=True,
    reason="pre-terminal InstanceNorm layout cluster lacks one owner",
)
def test_pre_terminal_instancenorm_layout_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in sorted(
            (
                node
                for node in ast.walk(owner)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and f"_{node.func.id}" in PASS_IDS
            ),
            key=lambda node: node.lineno,
        )
    ]
    assert tuple(f"_{name}" for name in owner_calls) == PASS_IDS

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert ast.unparse(assignment.value) == (
        "run_pre_terminal_instancenorm_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
