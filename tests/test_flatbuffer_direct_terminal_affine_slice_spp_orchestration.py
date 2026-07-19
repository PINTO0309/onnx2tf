from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.late_spp_concat_unary_conv_orchestration import (
    run_late_spp_concat_unary_conv_summary,
)
from onnx2tf.tflite_builder.passes.stridedslice_pad_concat_bridge_layout import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.terminal_affine_concat_split_recovery_orchestration import (
    run_terminal_affine_concat_split_recovery_summary,
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
    / "terminal_affine_slice_spp_orchestration.py"
)
OWNER = "run_terminal_affine_slice_spp_cleanup"
CHILD_OWNERS = (
    "run_terminal_affine_concat_split_recovery_summary",
    "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
    "run_late_spp_concat_unary_conv_summary",
)
RESULT_TARGETS = (
    "_terminal_affine_stats",
    "_terminal_slice_pad_concat_stats",
    "_late_spp_stats",
)
COMPOSITE_TARGET = "_terminal_affine_slice_spp_results"
PREDECESSOR_TARGET = "_pre_terminal_cleanup_results"
SUCCESSOR_TARGET = "_late_pre_qkv_shape_extract_stats"
EXPECTED_SCHEMAS = (
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
        "optimized_transpose_stridedslice_pad_concat_mul_add_"
        "posttranspose_nhwc_chains": 0,
    },
    {
        "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains": 0,
        "optimized_transpose_concat_unary_fanout_conv_nhwc_chains": 0,
    },
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


def test_terminal_affine_slice_spp_current_boundary_and_schema() -> None:
    lowerer = _lowerer()
    indices = tuple(
        next(
            index
            for index, statement in enumerate(lowerer.body)
            if _single_target(statement) == target
        )
        for target in RESULT_TARGETS
    )
    assert indices == tuple(range(indices[0], indices[0] + len(indices)))
    assert _single_target(lowerer.body[indices[0] - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[indices[-1] + 1]) == SUCCESSOR_TARGET
    assert tuple(_call_name(lowerer.body[index]) for index in indices) == (
        CHILD_OWNERS
    )

    calls = tuple(_call(lowerer.body[index]) for index in indices)
    assert all(call is not None for call in calls)
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "terminal_affine_concat_split_recovery_context"
    ]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "model_ir"
    ]
    assert calls[1].keywords == []
    assert [ast.unparse(argument) for argument in calls[2].args] == [
        "late_spp_concat_unary_conv_context"
    ]
    assert calls[2].keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )

    model_ir = ModelIR("terminal_affine_slice_spp_schema")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    assert (
        run_terminal_affine_concat_split_recovery_summary(context),
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        ),
        run_late_spp_concat_unary_conv_summary(context),
    ) == EXPECTED_SCHEMAS


def test_terminal_affine_and_spp_context_aliases_remain_shared() -> None:
    lowerer = _lowerer()
    aliases = {
        _single_target(statement): ast.unparse(statement.value)
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and _single_target(statement)
        in {
            "terminal_affine_concat_split_recovery_context",
            "late_spp_concat_unary_conv_context",
        }
    }
    assert aliases == {
        "terminal_affine_concat_split_recovery_context": (
            "shared_model_ir_pass_context"
        ),
        "late_spp_concat_unary_conv_context": "shared_model_ir_pass_context",
    }


@pytest.mark.xfail(
    strict=True,
    reason="terminal affine/Slice-SPP tail has no shared-context owner",
)
def test_terminal_affine_slice_spp_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert [ast.unparse(argument) for argument in calls[1].args] == [
        "context.model_ir"
    ]
    assert [ast.unparse(argument) for argument in calls[2].args] == ["context"]
    assert all(call.keywords == [] for call in calls)

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
    assert _single_target(lowerer.body[index - 1]) == PREDECESSOR_TARGET
    assert _single_target(lowerer.body[index + 1]) == SUCCESSOR_TARGET
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
