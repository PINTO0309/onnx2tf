from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    pre_terminal_cleanup_orchestration,
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
    / "pre_terminal_cleanup_orchestration.py"
)
OWNER = "run_pre_terminal_cleanup"
CHILD_OWNERS = (
    "run_pre_terminal_instancenorm_layout_cleanup",
    "run_terminal_affine_concat_split_recovery_summary",
    "run_pre_terminal_pre_add_cleanup",
    "run_channel_slice_pad_mul_summary",
    "run_pre_terminal_affine_tail_cleanup",
)
RESULT_TARGETS = (
    "_pre_terminal_instancenorm_layout_results",
    "_pre_terminal_affine_stats",
    "_pre_terminal_pre_add_stats",
    "_pre_terminal_channel_slice_pad_mul_stats",
    "_pre_terminal_affine_tail_results",
)
COMPOSITE_TARGET = "_pre_terminal_cleanup_results"
SUCCESSOR_TARGET = "_terminal_affine_slice_spp_results"


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def test_pre_terminal_cleanup_current_boundary_and_schemas() -> None:
    lowerer = _lowerer()
    index, invocation = next(
        (index, statement)
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == COMPOSITE_TARGET
    )
    assert _call_name(invocation) == OWNER
    call = _call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert (
        ast.unparse(lowerer.body[index - 1].test)
        == "_late_binary_layout_recovery_requires_reconciliation"
    )
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("pre_terminal_cleanup_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = pre_terminal_cleanup_orchestration.run_pre_terminal_cleanup(
        context
    )
    assert tuple(type(result) for result in results) == (
        tuple,
        dict,
        dict,
        dict,
        tuple,
    )
    assert tuple(len(result) for result in results) == (3, 13, 2, 4, 2)
    assert results[0] == (
        {"optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains": 0},
        {
            "optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains": 0
        },
        {
            "optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains": 0
        },
    )
    assert results[2] == {
        "optimized_transpose_pre_add_nhwc_chains": 0,
        "pruned_unused_tensors": 0,
    }
    assert results[3] == {
        "optimized_transpose_channel_slice_dual_add_bridges_strict": 0,
        "optimized_transpose_slice_muladd_conv_mergeadd_strict": 0,
        "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": 0,
        "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": 0,
    }
    assert results[4] == (
        {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0},
        {
            "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 0
        },
    )


def test_pre_terminal_cleanup_has_one_context_owner() -> None:
    owner = _functions(OWNER_PATH)[OWNER]
    calls = sorted(
        (
            node
            for node in ast.walk(owner)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in CHILD_OWNERS
        ),
        key=lambda node: (node.lineno, node.col_offset),
    )
    assert [call.func.id for call in calls] == list(CHILD_OWNERS)
    assert all(
        [ast.unparse(argument) for argument in call.args] == ["context"]
        and call.keywords == []
        for call in calls
    )

    lowerer = _lowerer()
    index, invocation = next(
        (index, statement)
        for index, statement in enumerate(lowerer.body)
        if _single_target(statement) == COMPOSITE_TARGET
    )
    assert _call_name(invocation) == OWNER
    call = _call(invocation)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    assert isinstance(lowerer.body[index - 1], ast.If)
    assert (
        ast.unparse(lowerer.body[index - 1].test)
        == "_late_binary_layout_recovery_requires_reconciliation"
    )
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_cleanup_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("pre_terminal_cleanup_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    results = tuple({"stage": index} for index in range(len(CHILD_OWNERS)))
    calls: list[tuple[str, ModelIRPassContext]] = []

    def callback(index: int):
        def run(active_context: ModelIRPassContext) -> dict[str, int]:
            calls.append((CHILD_OWNERS[index], active_context))
            return results[index]

        return run

    for index, name in enumerate(CHILD_OWNERS):
        monkeypatch.setattr(
            pre_terminal_cleanup_orchestration,
            name,
            callback(index),
        )

    actual = pre_terminal_cleanup_orchestration.run_pre_terminal_cleanup(
        context
    )

    assert actual == results
    assert all(actual[index] is results[index] for index in range(len(results)))
    assert calls == [(name, context) for name in CHILD_OWNERS]
