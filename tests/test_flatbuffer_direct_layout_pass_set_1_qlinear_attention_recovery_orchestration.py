from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.boundary_batchmatmul_unary_orchestration import (
    run_boundary_batchmatmul_unary,
)
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    run_channel_shuffle_gather,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LayoutRecoveryContext,
    run_layout_reshape_attention_recovery_prefix,
)
from onnx2tf.tflite_builder.passes.pre_concat_nhwc_layout import (
    optimize_transpose_pre_concat_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration import (
    run_qlinear_mean_concat_recovery,
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
    / "layout_pass_set_1_qlinear_attention_recovery_orchestration.py"
)
OWNER = "run_layout_pass_set_1_qlinear_attention_recovery"
CHILD_OWNERS = (
    "run_qlinear_mean_concat_recovery",
    "run_layout_reshape_attention_recovery_prefix",
)
CURRENT_CHILD_OWNERS = (
    "_run_qlinear_mean_concat_recovery_sequence",
    "_run_layout_reshape_attention_recovery_prefix",
)
RESULT_TARGETS = (
    "_layout_pass_set_1_qlinear_mean_concat_results",
    "_layout_pass_set_1_final_attention_recovery_results",
)
COMPOSITE_TARGET = "_layout_pass_set_1_qlinear_attention_recovery_results"
PREDECESSOR_PHASE_ID = "cleanup.layout_pass_set_1.dequant_mean_quantize"
SUCCESSOR_PHASE_ID = "cleanup.layout_pass_set_1.instancenorm_prepost"
GUARD = "optimize_layout_transpose_chains"

QLINEAR_SCHEMA = (
    ("optimized_transpose_mean_hardsigmoid_muladd_chains",),
    ("optimized_nhwc_prefix_qlinear_silu_chains",),
    ("propagated_qlinear_concat_conv_nhwc_chains",),
    ("bypassed_concat_pre_quantize_dequantize",),
    ("optimized_transpose_mean_maxpool_concat_conv_chains",),
)
LAYOUT_PREFIX_SCHEMA = (
    (
        "removed_transpose_quantize_dequantize_bridges",
        "rewritten_add_qdq_residual_transpose_bridges",
        "rewritten_mixed_add_qdq_residual_transpose_bridges",
    ),
    (
        ("rewritten_boundary_input_transpose_batchmatmul_chains",),
        (
            "rewritten_leading_input_transpose_passthrough_chains",
            "rewritten_asin_transpose_passthrough_chains",
            "rewritten_erf_transpose_passthrough_chains",
        ),
    ),
    ("optimized_transpose_elementwise_roundtrip_nchw_nhwc_chains",),
    ("optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",),
    (
        "rewritten_hardswish_transpose_passthrough_chains",
        "rewritten_hardsigmoid_transpose_passthrough_chains",
        "rewritten_hardsigmoid_mul_transpose_passthrough_chains",
    ),
    ("rewritten_swish_transpose_passthrough_chains",),
    ("rewritten_gelu_tanh_transpose_passthrough_chains",),
    ("optimized_center_size_offset_terminal_transpose_chains",),
    (
        "rewritten_leakyrelu_transpose_passthrough_chains",
        "fused_pseudo_leakyrelu_chains",
    ),
    ("rewritten_prelu_transpose_passthrough_chains",),
    ("optimized_transpose_elementwise_concat_conv_nhwc_groups",),
    ("optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains",),
    ("optimized_transpose_pre_concat_nhwc_chains",),
    ("optimized_transpose_pre_concat_ndhwc_chains",),
    ("optimized_transpose_stridedslice_pre_concat_nhwc_chains",),
    (
        "optimized_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    ),
    ("optimized_transpose_input_chains_pre_concat_to_single_post_adapter",),
    ("optimized_transpose_slice_logistic_concat_reshape_tail_nhwc_chains",),
    (
        ("optimized_shufflenet_transpose_shuffle_chains",),
        ("optimized_shufflenet_reshape_transpose_shuffle_nhwc_chains",),
        (
            "optimized_nchw_channel_shuffle_reshape_transpose_reshape_to_gather",
        ),
        ("optimized_transpose_gather_transpose_axis_remap_nhwc_chains",),
    ),
)
ATTENTION_SCHEMA = (
    LAYOUT_PREFIX_SCHEMA,
    ("optimized_transpose_pre_add_nhwc_chains",),
    ("optimized_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains",),
    ("optimized_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains",),
    ("optimized_transpose_reshape_transpose_to_expanddims_nhwc_chains",),
    ("optimized_transpose_reshape_transpose_to_flatten_hw_nhwc_chains",),
    ("optimized_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains",),
    (
        "optimized_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains",
    ),
    (
        "optimized_attention_gather_transpose_reshape_cleanup_pattern_a",
        "optimized_attention_gather_transpose_reshape_cleanup_pattern_b",
    ),
    ("optimized_gather_axis0_singleton_to_reshape_input_chains",),
    ("optimized_attention_preproj_reshape_to_batchmatmul_ranklift_chains",),
    ("optimized_window_partition_reshape_transpose_to_space_to_depth_chains",),
    ("optimized_window_reverse_reshape_transpose_to_depth_to_space_chains",),
    ("optimized_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains",),
    (
        "optimized_squeeze_reshape_identity_chains",
        "optimized_squeeze_unary_reshape_passthrough_chains",
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


def _phase_id(statement: ast.stmt) -> str | None:
    call = _call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or ast.unparse(call.func) != "session.record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _guard_body() -> list[ast.stmt]:
    lowerer = _lowerer()
    guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test) == GUARD
        and any(
            _single_target(candidate) in RESULT_TARGETS
            for candidate in statement.body
        )
    )
    assert guard.orelse == []
    return guard.body


def _schema(value: Any) -> Any:
    if isinstance(value, dict):
        return tuple(value)
    if isinstance(value, tuple):
        return tuple(_schema(item) for item in value)
    raise AssertionError(f"unexpected result type: {type(value)!r}")


def _context() -> LayoutRecoveryContext:
    model_ir = ModelIR("layout_pass_set_1_qlinear_attention_schema")
    pass_context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    return LayoutRecoveryContext(
        pass_context=pass_context,
        boundary_batchmatmul_unary_cluster=(
            lambda: run_boundary_batchmatmul_unary(pass_context)
        ),
        pre_concat_cleanup=optimize_transpose_pre_concat_nhwc_chains,
        channel_shuffle_gather_cluster=(
            lambda: run_channel_shuffle_gather(pass_context)
        ),
    )


def test_layout_pass_set_1_qlinear_attention_current_contract() -> None:
    body = _guard_body()
    assignments = [
        statement
        for statement in body
        if _single_target(statement) in RESULT_TARGETS
    ]
    assert [_single_target(statement) for statement in assignments] == list(
        RESULT_TARGETS
    )
    assert [_call_name(statement) for statement in assignments] == list(
        CURRENT_CHILD_OWNERS
    )
    indices = [body.index(statement) for statement in assignments]
    assert indices[1] == indices[0] + 1
    assert _phase_id(body[indices[0] - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(body[indices[-1] + 1]) == SUCCESSOR_PHASE_ID
    assert all(_call(statement).args == [] for statement in assignments)
    assert all(_call(statement).keywords == [] for statement in assignments)
    assert not any(
        isinstance(node, ast.Name)
        and isinstance(node.ctx, ast.Load)
        and node.id in RESULT_TARGETS
        for statement in body
        for node in ast.walk(statement)
    )

    lowerer = _lowerer()
    context_assignments = {
        _single_target(statement): statement
        for statement in lowerer.body
        if _single_target(statement)
        in {
            "shared_model_ir_pass_context",
            "qlinear_recovery_context",
            "layout_recovery_context",
        }
    }
    assert ast.unparse(
        context_assignments["shared_model_ir_pass_context"].value
    ) == "session.model_ir_pass_context"
    assert ast.unparse(
        context_assignments["qlinear_recovery_context"].value
    ) == "shared_model_ir_pass_context"
    layout_assignment = context_assignments["layout_recovery_context"]
    assert isinstance(layout_assignment.value, ast.Call)
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in layout_assignment.value.keywords
    } == {
        "pass_context": "session.model_ir_pass_context",
        "boundary_batchmatmul_unary_cluster": (
            "_run_boundary_batchmatmul_unary_layout_pass_cluster"
        ),
        "pre_concat_cleanup": "_optimize_transpose_pre_concat_nhwc_chains",
        "channel_shuffle_gather_cluster": (
            "_run_channel_shuffle_gather_layout_pass_cluster"
        ),
    }

    context = _context()
    results = (
        run_qlinear_mean_concat_recovery(context.pass_context),
        run_layout_reshape_attention_recovery_prefix(context),
    )
    assert _schema(results) == (QLINEAR_SCHEMA, ATTENTION_SCHEMA)

    direct_counts = {
        child_owner: sum(
            1
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == child_owner
        )
        for child_owner in CURRENT_CHILD_OWNERS
    }
    assert direct_counts == {
        CURRENT_CHILD_OWNERS[0]: 2,
        CURRENT_CHILD_OWNERS[1]: 3,
    }


@pytest.mark.xfail(
    strict=True,
    reason="layout-pass-set-1 QLinear/attention owner is not implemented",
)
def test_layout_pass_set_1_qlinear_attention_has_one_context_owner() -> None:
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
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context.pass_context"
    ]
    assert calls[0].keywords == []
    assert [ast.unparse(argument) for argument in calls[1].args] == ["context"]
    assert calls[1].keywords == []

    body = _guard_body()
    assignment = next(
        statement
        for statement in body
        if _single_target(statement) == COMPOSITE_TARGET
    )
    index = body.index(assignment)
    assert _call_name(assignment) == OWNER
    call = _call(assignment)
    assert call is not None
    assert [ast.unparse(argument) for argument in call.args] == [
        "layout_recovery_context"
    ]
    assert call.keywords == []
    assert _phase_id(body[index - 1]) == PREDECESSOR_PHASE_ID
    assert _phase_id(body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in RESULT_TARGETS
        for node in ast.walk(_lowerer())
    )

    lowerer_functions = {
        node.name: node
        for node in _lowerer().body
        if isinstance(node, ast.FunctionDef)
    }
    assert all(name in lowerer_functions for name in CURRENT_CHILD_OWNERS)
