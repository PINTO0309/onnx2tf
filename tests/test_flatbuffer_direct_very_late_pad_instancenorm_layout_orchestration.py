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
    / "very_late_pad_instancenorm_layout_orchestration.py"
)
OWNER = "run_very_late_pad_instancenorm_layout_cleanup"
RESULT_TARGET = "_very_late_pad_instancenorm_layout_results"
PREDECESSOR_TARGET = "_late_conv1d_decoder_layout_results"
SUCCESSOR_TARGET = "_very_late_singleton_consecutive_reshape_results"
OLD_RESULT_TARGETS = (
    "_very_late_pad_layout_stats",
    "_very_late_instancenorm_post_bias_stats",
    "_very_late_instancenorm_residual_mul_concat_stats",
    "_very_late_instancenorm_dualstats_stats",
)
PASS_IDS = (
    "run_pad_layout_cleanup",
    "optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
)
LOWERER_PASS_IDS = (
    "run_pad_layout_cleanup",
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
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


def test_very_late_pad_instancenorm_cluster_is_ordered_and_unconsumed() -> None:
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
    assert indices == list(range(indices[0], indices[0] + len(indices)))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    assert tuple(_call_name(statement) for statement in assignments) == (
        LOWERER_PASS_IDS
    )

    first_call = _statement_call(assignments[0])
    assert first_call is not None
    assert [ast.unparse(argument) for argument in first_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in first_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }
    for statement in assignments[1:]:
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            "model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
    for target in OLD_RESULT_TARGETS:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )


@pytest.mark.xfail(
    strict=True,
    reason="very-late Pad/InstanceNorm cluster lacks a composite owner",
)
def test_very_late_pad_instancenorm_cluster_uses_one_composite_owner() -> None:
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
        "run_very_late_pad_instancenorm_layout_cleanup("
        "shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
