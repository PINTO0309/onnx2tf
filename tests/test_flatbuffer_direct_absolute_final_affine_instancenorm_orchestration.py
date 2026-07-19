from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    absolute_final_affine_instancenorm_orchestration,
)


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


def test_absolute_final_affine_instancenorm_summary_and_boundaries_are_fixed() -> (
    None
):
    lowerer = _lowerer()
    summary_index = _target_index(lowerer, SUMMARY_TARGET)
    assert _single_target(lowerer.body[summary_index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[summary_index + 1]) == SUCCESSOR_TARGET

    summary = lowerer.body[summary_index]
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{SUMMARY_OWNER}(shared_model_ir_pass_context)"
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_TARGETS
        for node in ast.walk(lowerer)
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


def test_absolute_final_affine_instancenorm_preserves_context_order_and_schemas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("absolute_final_affine_instancenorm")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict[str, object]] = []
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    results = (
        {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 2},
        {
            "optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains": 3
        },
    )
    events: list[tuple[str, ModelIR, dict[str, object]]] = []

    def _run_affine(candidate: ModelIR, **kwargs: object) -> dict[str, int]:
        events.append((AFFINE_OWNER, candidate, dict(kwargs)))
        return dict(results[0])

    def _run_instancenorm(
        candidate: ModelIR,
        **kwargs: object,
    ) -> dict[str, int]:
        events.append((INSTANCENORM_OWNER, candidate, dict(kwargs)))
        return dict(results[1])

    monkeypatch.setattr(
        absolute_final_affine_instancenorm_orchestration,
        AFFINE_OWNER,
        _run_affine,
    )
    monkeypatch.setattr(
        absolute_final_affine_instancenorm_orchestration,
        INSTANCENORM_OWNER,
        _run_instancenorm,
    )

    owner_module = absolute_final_affine_instancenorm_orchestration
    runner = owner_module.run_absolute_final_affine_instancenorm_cleanup
    assert runner(context) == results
    assert events == [
        (
            AFFINE_OWNER,
            model_ir,
            {"layout_state": layout_state},
        ),
        (
            INSTANCENORM_OWNER,
            model_ir,
            {"layout_state": layout_state},
        ),
    ]
