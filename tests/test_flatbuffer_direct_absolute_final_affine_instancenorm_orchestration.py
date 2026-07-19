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
    / "absolute_final_affine_instancenorm_orchestration.py"
)
SUMMARY_OWNER = "run_absolute_final_affine_instancenorm_cleanup"
SUMMARY_TARGET = "_absolute_final_affine_instancenorm_results"
OLD_TARGETS = (
    "_absolute_final_affine_post_add_stats",
    "_absolute_final_instancenorm_post_bias_stats",
)
AFFINE_WRAPPER = "_optimize_transpose_mul_posttranspose_add_nhwc_chains"
AFFINE_OWNER = "optimize_transpose_mul_posttranspose_add_nhwc_chains"
INSTANCENORM_WRAPPER = (
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
)
INSTANCENORM_OWNER = (
    "optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
)
PREDECESSOR_TARGET = "_absolute_final_boundary_signature_results"
SUCCESSOR_TARGET = "_absolute_final_normalization_attention_results"


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


def _target_index(lowerer: ast.FunctionDef, target_name: str) -> int:
    return next(
        index
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == target_name
    )


def test_absolute_final_affine_instancenorm_raw_pair_and_boundaries_are_fixed() -> (
    None
):
    lowerer = _lowerer()
    affine_index = _target_index(lowerer, OLD_TARGETS[0])
    assert _target_index(lowerer, OLD_TARGETS[1]) == affine_index + 1
    assert _single_target(lowerer.body[affine_index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[affine_index + 2]) == SUCCESSOR_TARGET

    affine = lowerer.body[affine_index]
    instancenorm = lowerer.body[affine_index + 1]
    assert isinstance(affine, ast.Assign)
    assert ast.unparse(affine.value) == (
        f"{AFFINE_WRAPPER}(model_ir, layout_state=session.layout_state)"
    )
    assert isinstance(instancenorm, ast.Assign)
    assert ast.unparse(instancenorm.value) == (
        f"{INSTANCENORM_WRAPPER}(model_ir, "
        "layout_state=session.layout_state)"
    )


@pytest.mark.xfail(
    strict=True,
    reason="absolute-final affine/InstanceNorm pair lacks one context owner",
)
def test_absolute_final_affine_instancenorm_uses_one_ordered_context_owner() -> (
    None
):
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    calls = [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {AFFINE_OWNER, INSTANCENORM_OWNER}
    ]
    calls.sort(key=lambda node: (node.lineno, node.col_offset))
    assert [call.func.id for call in calls] == [
        AFFINE_OWNER,
        INSTANCENORM_OWNER,
    ]
    for call in calls:
        assert [ast.unparse(argument) for argument in call.args] == [
            "context.model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "context.layout_state"}

    lowerer = _lowerer()
    summary_index = _target_index(lowerer, SUMMARY_TARGET)
    summary = lowerer.body[summary_index]
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(shared_model_ir_pass_context)"
    )
    assert _single_target(lowerer.body[summary_index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[summary_index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_TARGETS
        for node in ast.walk(lowerer)
    )

    lowerer_functions = _functions(LOWERER_PATH)
    assert AFFINE_WRAPPER in lowerer_functions
    assert INSTANCENORM_WRAPPER in lowerer_functions
