from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import late_spp_concat_unary_conv_orchestration

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_spp_concat_unary_conv_orchestration.py"
)
RAW_WRAPPER = "_run_late_spp_concat_unary_conv_pass_pair"
RAW_OWNER = "run_late_spp_concat_unary_conv"
SUMMARY_OWNER = "run_late_spp_concat_unary_conv_summary"
SUMMARY_FUNCTION = "summarize_late_spp_concat_unary_conv_mutations"
RAW_TARGET = "late_spp_results"
SUMMARY_TARGET = "_late_spp_stats"
PREDECESSOR_TARGET = "_terminal_slice_pad_concat_stats"
SUCCESSOR_TARGET = "_terminal_qkv_shape_attention_results"
COMPOSITE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_affine_slice_spp_orchestration.py"
)
COMPOSITE_OWNER = "run_terminal_affine_slice_spp_cleanup"
COMPOSITE_TARGET = "_terminal_affine_slice_spp_results"
OUTER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_affine_slice_spp_orchestration.py"
)
OUTER_OWNER = "run_pre_terminal_affine_slice_spp_cleanup"
OUTER_TARGET = "_pre_terminal_affine_slice_spp_results"


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


def _composite_summary_calls() -> list[ast.Call]:
    owner = _functions(COMPOSITE_PATH)[COMPOSITE_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SUMMARY_OWNER
    ]


def _outer_calls() -> list[ast.Call]:
    owner = _functions(OUTER_PATH)[OUTER_OWNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == COMPOSITE_OWNER
    ]


def test_late_spp_summary_boundary_is_fixed() -> None:
    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == OUTER_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{OUTER_OWNER}(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert len(_outer_calls()) == 1
    calls = _composite_summary_calls()
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert not any(
        isinstance(node, ast.Name) and node.id == RAW_TARGET
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id == SUMMARY_TARGET
        for node in ast.walk(lowerer)
    )

    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}(late_spp_concat_unary_conv_context)"
    )


def test_late_spp_uses_one_direct_summary_owner() -> None:
    owner = _functions(OWNER_PATH)[SUMMARY_OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in {RAW_OWNER, SUMMARY_FUNCTION}
    ]
    assert owner_calls.count(RAW_OWNER) == 1
    assert owner_calls.count(SUMMARY_FUNCTION) == 1

    lowerer = _lowerer()
    summary = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == OUTER_TARGET
    )
    index = lowerer.body.index(summary)
    assert isinstance(summary, ast.Assign)
    assert ast.unparse(summary.value) == (
        f"{OUTER_OWNER}(shared_model_ir_pass_context)"
    )
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert len(_outer_calls()) == 1
    assert len(_composite_summary_calls()) == 1
    assert not any(
        isinstance(node, ast.Name) and node.id == RAW_TARGET
        for node in ast.walk(lowerer)
    )

    wrapper = next(
        node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef) and node.name == RAW_WRAPPER
    )
    assert ast.unparse(wrapper.body[0]) == (
        f"return {RAW_OWNER}(late_spp_concat_unary_conv_context)"
    )


def test_late_spp_summary_owner_preserves_context_and_schema(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_spp_summary")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    raw_results = (
        {
            "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains": 2,
        },
        {
            "optimized_transpose_concat_unary_fanout_conv_nhwc_chains": 3,
        },
    )
    observed: list[object] = []

    def _run(candidate: ModelIRPassContext) -> tuple[dict[str, int], ...]:
        observed.append(candidate)
        return raw_results

    monkeypatch.setattr(
        late_spp_concat_unary_conv_orchestration,
        RAW_OWNER,
        _run,
    )

    assert late_spp_concat_unary_conv_orchestration.run_late_spp_concat_unary_conv_summary(
        context
    ) == {
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains": 2,
        "optimized_transpose_concat_unary_fanout_conv_nhwc_chains": 3,
    }
    assert observed == [context]
