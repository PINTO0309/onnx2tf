from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    pre_terminal_affine_slice_spp_orchestration,
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
    / "pre_terminal_affine_slice_spp_orchestration.py"
)
OWNER = "run_pre_terminal_affine_slice_spp_cleanup"
CHILD_OWNERS = (
    "run_pre_terminal_cleanup",
    "run_terminal_affine_slice_spp_cleanup",
)
RESULT_TARGETS = (
    "_pre_terminal_cleanup_results",
    "_terminal_affine_slice_spp_results",
)
COMPOSITE_TARGET = "_pre_terminal_affine_slice_spp_results"
PREDECESSOR_GUARD = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
SUCCESSOR_TARGET = "_terminal_qkv_shape_attention_results"
EXPECTED_SCHEMAS = (
    (
        (
            {
                "optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains": 0
            },
            {
                "optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains": 0
            },
            {
                "optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains": 0
            },
        ),
        {
            "optimized_fold_mul_add_mul_affine_chains": 0,
            "optimized_transpose_mul_add_const_prepost_nhwc_chains": 0,
            "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 0,
            "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 0,
            "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 0,
            "optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains": 0,
            "optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0,
            "optimized_transpose_unary_split_concat_single_post_nchw": 0,
            "optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0,
            "optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0,
            "sanitized_probable_nhwc_axis_sensitive_ops": 0,
            "inserted_probable_nhwc_terminal_transposes": 0,
            "pruned_unused_tensors": 0,
        },
        {
            "optimized_transpose_pre_add_nhwc_chains": 0,
            "pruned_unused_tensors": 0,
        },
        {
            "optimized_transpose_channel_slice_dual_add_bridges_strict": 0,
            "optimized_transpose_slice_muladd_conv_mergeadd_strict": 0,
            "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": 0,
            "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": 0,
        },
        (
            {"optimized_transpose_mul_posttranspose_add_nhwc_chains": 0},
            {
                "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 0
            },
        ),
    ),
    (
        {
            "optimized_fold_mul_add_mul_affine_chains": 0,
            "optimized_transpose_mul_add_const_prepost_nhwc_chains": 0,
            "optimized_concat_mul_add_transpose_nhwc_bridge_chains": 0,
            "optimized_concat_mul_add_transpose_add_nhwc_bridge_chains": 0,
            "optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains": 0,
            "optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains": 0,
            "optimized_singleton_gate_conv_concat_nhwc_bridge_blocks": 0,
            "optimized_transpose_unary_split_concat_single_post_nchw": 0,
            "optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0,
            "optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0,
            "sanitized_probable_nhwc_axis_sensitive_ops": 0,
            "inserted_probable_nhwc_terminal_transposes": 0,
            "pruned_unused_tensors": 0,
        },
        {
            "optimized_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains": 0
        },
        {
            "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains": 0,
            "optimized_transpose_concat_unary_fanout_conv_nhwc_chains": 0,
        },
    ),
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


def _call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def test_pre_terminal_affine_slice_spp_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("pre_terminal_affine_slice_spp_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        pre_terminal_affine_slice_spp_orchestration.run_pre_terminal_affine_slice_spp_cleanup(
            context
        )
        == EXPECTED_SCHEMAS
    )


def test_pre_terminal_affine_slice_spp_has_one_context_owner() -> None:
    assert OWNER_PATH.exists()
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
    for call in calls:
        assert [ast.unparse(argument) for argument in call.args] == [
            "context"
        ]
        assert call.keywords == []

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_pre_terminal_affine_slice_spp_runtime_order_and_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("pre_terminal_affine_slice_spp_runtime")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected_results = (
        tuple({f"pre_terminal_{index}": index} for index in range(5)),
        tuple({f"affine_slice_spp_{index}": index} for index in range(3)),
    )
    observed: list[tuple[str, ModelIRPassContext]] = []

    def _callback(index: int):
        def _run(active_context: ModelIRPassContext) -> tuple[dict[str, int], ...]:
            observed.append((CHILD_OWNERS[index], active_context))
            return expected_results[index]

        return _run

    for index, child_owner in enumerate(CHILD_OWNERS):
        monkeypatch.setattr(
            pre_terminal_affine_slice_spp_orchestration,
            child_owner,
            _callback(index),
        )

    actual = (
        pre_terminal_affine_slice_spp_orchestration.run_pre_terminal_affine_slice_spp_cleanup(
            context
        )
    )
    assert actual == expected_results
    assert actual[0] is expected_results[0]
    assert actual[1] is expected_results[1]
    assert observed == [
        (CHILD_OWNERS[0], context),
        (CHILD_OWNERS[1], context),
    ]
