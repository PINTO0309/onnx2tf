from __future__ import annotations

import ast
from pathlib import Path

import pytest
from onnx import helper

from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.ir import ModelIR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
EXPECTED_RESULT_TARGETS = (
    "_layout_pass_set_1_layout_transpose_cleanup_stats",
    "_layout_pass_set_1_initial_affine_chain_fold_stats",
    "_layout_pass_set_1_affine_prepost_stats",
    "_layout_pass_set_1_pre_unary_affine_fanout_stats",
    "_layout_pass_set_1_mean_affine_prepost_stats",
    "_layout_pass_set_1_quantized_prelu_stats",
    "_layout_pass_set_1_dequant_transposeconv_quantize_stats",
    "_layout_pass_set_1_quantized_reshape_stats",
    "_layout_pass_set_1_transpose_binary_bridge_stats",
    "_layout_pass_set_1_duplicate_fanout_stats",
    "_layout_pass_set_1_post_binary_affine_chain_fold_stats",
    "_layout_pass_set_1_dequant_mean_quantize_stats",
    "_layout_pass_set_1_instancenorm_prepost_stats",
    "_layout_pass_set_1_squeeze_reshape_identity_stats",
    "_core_cleanup_pseudo_leakyrelu_stats",
    "_core_cleanup_yolo_decode_stats",
    "_core_cleanup_consecutive_mul_stats",
    "_core_cleanup_terminal_dequant_stats",
    "_core_cleanup_terminal_qdq_stats",
    "_core_cleanup_conv_affine_stats",
    "_core_cleanup_conv_activation_stats",
    "_core_cleanup_dynamic_reshape_stats",
    "_core_cleanup_squeeze_reshape_identity_stats",
    "_core_cleanup_prune_reconcile_stats",
    "_layout_pass_set_2_dequant_transposeconv_quantize_stats",
    "_layout_opt_elementwise_concat_conv_stats",
    "_layout_opt_spp_stats",
    "_layout_opt_pre_concat_stats",
    "_layout_opt_ndhwc_concat_stats",
    "_layout_opt_stridedslice_pre_concat_stats",
    "_layout_opt_split_mixed_pre_concat_stats",
    "_layout_opt_concat_input_adapter_stats",
    "_layout_opt_slice_logistic_concat_tail_stats",
    "_layout_opt_sa_pa_mirrorpad_stats",
    "_layout_pass_set_2_squeeze_reshape_identity_stats",
    "_layout_pass_set_2_prune_reconcile_stats",
    "_terminal_cleanup_terminal_dequant_stats",
    "_terminal_cleanup_terminal_qdq_stats",
    "_terminal_cleanup_conv_affine_stats",
    "_terminal_cleanup_conv_activation_stats",
    "_terminal_pre_argmax_stats",
    "_terminal_transpose_gather_channel_fanout_stats",
    "_terminal_softmax_transpose_stats",
    "_terminal_boundary_input_normalization_stats",
    "_terminal_boundary_input_channel_slice_stats",
    "_terminal_internal_channel_slice_stats",
    "_terminal_channel_slice_muladd_bridge_stats",
    "_terminal_boundary_stridedslice_qdq_concat_stats",
    "_terminal_swish_residual_concat_closure_stats",
    "_terminal_dequant_logistic_mul_quantize_bridge_stats",
    "_terminal_swish_qdq_island_stats",
    "_terminal_instancenorm_post_bias_stats",
    "_terminal_normalization_pad_stats",
    "_terminal_instancenorm_residual_add_stats",
    "_terminal_instancenorm_residual_mul_concat_stats",
    "_terminal_instancenorm_dualstats_stats",
    "_terminal_batchmatmul_affine_input_stats",
    "_terminal_batchmatmul_reshape_se_stats",
    "_terminal_batchmatmul_adj_flags_stats",
    "_terminal_qkv_split_conv_concat_bridge_stats",
    "_terminal_sinet_hardswish_se_stats",
    "_terminal_dequant_hardsigmoid_bridge_stats",
    "_post_terminal_indexed_shape_convergence_stats",
    "_very_late_residual_affine_prelu_stats",
    "_very_late_residual_affine_fanout_stats",
    "_very_late_prune_reconcile_stats",
    "_post_cleanup_csp_attention_stats",
    "_post_cleanup_sa_pa_mirrorpad_stats",
    "_no_layout_safe_transpose_reduction_stats",
    "_very_late_broadcast_static_shape_stats",
    "_shared_late_static_shape_stats",
    "_late_binary_repair_static_shape_stats",
    "_late_binary_layout_recovery_static_shape_stats",
    "_terminal_expand_squeeze_static_shape_stats",
    "_very_late_static_shape_stats",
    "_post_split_fallback_static_shape_stats",
    "_fallback_norm_static_shape_stats",
    "_fallback_dynamic_rank1_topology_layout_stats",
    "_fallback_broadcast_static_shape_stats",
    "_fallback_broadcast_topology_layout_stats",
    "_fallback_se_fc_gather_static_shape_stats",
    "_fallback_placeholder_matmul_static_shape_stats",
    "_fallback_post_placeholder_topology_stats",
    "_fallback_conv_input_static_shape_stats",
    "_fallback_mixed_concat_static_shape_stats",
    "_fallback_concat_axis_static_shape_stats",
    "_fallback_binary_layout_static_shape_stats",
    "_fallback_post_layout_repair_topology_stats",
    "_fallback_high_rank_bmm_static_shape_stats",
    "_fallback_topology_layout_validation_stats",
    "_primary_post_lowering_topology_stats",
    "_no_layout_post_reduction_topology_stats",
    "_absolute_final_topology_layout_stats",
    "_final_convinteger_static_shape_stats",
    "_final_convinteger_topology_layout_stats",
    "_final_instancenorm_static_shape_stats",
    "_final_instancenorm_topology_layout_stats",
    "_final_broadcast_static_shape_stats",
    "_final_broadcast_topology_layout_stats",
    "_final_mixed_singleton_concat_static_shape_stats",
    "_final_placeholder_binary_static_shape_stats",
    "_final_placeholder_topology_stats",
    "_final_se_fc_gather_static_shape_stats",
    "_final_prelu_static_shape_stats",
    "_final_consecutive_reshape_static_shape_stats",
    "_final_sinet_late_residual_static_shape_stats",
    "_final_sinet_preadd_fanout_static_shape_stats",
    "_final_sinet_dual_resize_static_shape_stats",
    "_final_sinet_shared_post_static_shape_stats",
    "_final_sinet_deep_skip_static_shape_stats",
    "_final_sinet_concat_resize_static_shape_stats",
    "_final_high_rank_bmm_static_shape_stats",
    "_final_pad_layout_static_shape_stats",
    "_final_conv_input_static_shape_stats",
    "_final_mixed_concat_static_shape_stats",
    "_final_concat_axis_static_shape_stats",
    "_final_binary_layout_static_shape_stats",
    "_terminal_topology_layout_validation_stats",
)
EXPECTED_OWNERS = (
    "run_layout_transpose_cleanup",
    "_optimize_fold_mul_add_mul_affine_chains",
    "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
    "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
    "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains",
    "run_quantized_prelu_cleanup",
    "_optimize_dequant_transposeconv_quantize_chains",
    "run_quantized_reshape_cleanup",
    "_optimize_transpose_binary_bridges",
    "run_duplicate_fanout_cleanup",
    "_optimize_fold_mul_add_mul_affine_chains",
    "_optimize_transpose_dequantize_mean_quantize_bridges",
    "_optimize_transpose_instancenorm_prepost_nhwc_chains",
    "run_squeeze_reshape_identity_cleanup",
    "_optimize_fuse_pseudo_leakyrelu_chains",
    "_optimize_yolo_decode_mul_square_anchor_chains",
    "run_consecutive_mul_constants_cleanup",
    "_sanitize_terminal_transpose_before_dequantize",
    "run_terminal_quantize_dequantize_cleanup",
    "_optimize_fold_conv_mul_add_affine_chains",
    "_optimize_fuse_conv_activation_chains",
    "_resolve_dynamic_reshape_shapes",
    "run_squeeze_reshape_identity_cleanup",
    "run_indexed_prune_reconcile_cleanup",
    "_optimize_dequant_transposeconv_quantize_chains",
    "_optimize_transpose_elementwise_concat_conv_nhwc_groups",
    "run_spp_layout_cleanup",
    "_optimize_transpose_pre_concat_nhwc_chains",
    "run_ndhwc_concat_layout_cleanup",
    "_optimize_transpose_stridedslice_pre_concat_nhwc_chains",
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains",
    "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",
    "run_squeeze_reshape_identity_cleanup",
    "run_indexed_prune_reconcile_cleanup",
    "_sanitize_terminal_transpose_before_dequantize",
    "run_terminal_quantize_dequantize_cleanup",
    "_optimize_fold_conv_mul_add_affine_chains",
    "_optimize_fuse_conv_activation_chains",
    "_optimize_transpose_pre_argmax_nhwc_terminal_chains",
    "run_transpose_gather_channel_fanout_cleanup",
    "_optimize_terminal_softmax_transpose_after_nhwc_propagation",
    "run_boundary_input_normalization_cleanup",
    "_optimize_boundary_input_transpose_channel_slice_blocks",
    "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains",
    "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains",
    "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks",
    "_optimize_transpose_swish_residual_concat_closure_nhwc_chains",
    "_optimize_transpose_dequant_logistic_mul_quantize_bridges",
    "_optimize_transpose_swish_qdq_nhwc_islands",
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "run_normalization_pad_layout_cleanup",
    "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
    "_optimize_batchmatmul_affine_transpose_input_chains",
    "_optimize_batchmatmul_reshape_se_nhwc_chains",
    "_optimize_batchmatmul_transpose_input_to_adj_flags",
    "_optimize_split_conv_concat_transpose_bridge_to_single_post_nchw",
    "_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains",
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
    "_run_indexed_shape_convergence_cleanup",
    "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
    "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
    "run_indexed_prune_reconcile_cleanup",
    "_optimize_transpose_csp_attention_nhwc_chains",
    "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",
    "_apply_safe_transpose_reduction_lite",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "run_static_shape_topology_reconciliation",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_topologically_sort_operators",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_topologically_sort_operators",
    "run_static_shape_topology_reconciliation",
    "run_topology_layout_validation",
    "_topologically_sort_operators",
    "_topologically_sort_operators",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "run_topology_layout_refresh",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_topologically_sort_operators",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "_reconcile_static_tensor_shapes",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_static_shape_topology_reconciliation",
    "run_topology_layout_validation",
)
EXPECTED_MODEL_ARGUMENTS = (
    *("model_ir",) * 76,
    *("fallback_ir",) * 14,
    *("model_ir",) * 28,
)
EXPECTED_PHASE_IDS = (
    "cleanup.layout_pass_set_1.layout_transpose",
    "cleanup.layout_pass_set_1.initial_affine_chain_fold",
    "cleanup.layout_pass_set_1.affine_prepost",
    "cleanup.layout_pass_set_1.pre_unary_affine_fanout",
    "cleanup.layout_pass_set_1.mean_affine_prepost",
    "cleanup.layout_pass_set_1.quantized_prelu",
    "cleanup.layout_pass_set_1.dequant_transposeconv_quantize",
    "cleanup.layout_pass_set_1.quantized_reshape",
    "cleanup.layout_pass_set_1.transpose_binary_bridge",
    "cleanup.layout_pass_set_1.duplicate_fanout",
    "cleanup.layout_pass_set_1.post_binary_affine_chain_fold",
    "cleanup.layout_pass_set_1.dequant_mean_quantize",
    "cleanup.layout_pass_set_1.instancenorm_prepost",
    "cleanup.layout_pass_set_1.squeeze_reshape_identity",
    "cleanup.core.pseudo_leakyrelu",
    "cleanup.core.yolo_decode",
    "cleanup.core.consecutive_mul",
    "cleanup.core.terminal_dequant",
    "cleanup.core.terminal_qdq",
    "cleanup.core.conv_affine",
    "cleanup.core.conv_activation",
    "shape_resolution.core.dynamic_reshape",
    "cleanup.core.squeeze_reshape_identity",
    "cleanup.core.prune_reconcile",
    "cleanup.layout_pass_set_2.dequant_transposeconv_quantize",
    "cleanup.layout_pass_set_2.elementwise_concat_conv",
    "cleanup.layout_pass_set_2.spp",
    "cleanup.layout_pass_set_2.pre_concat",
    "cleanup.layout_pass_set_2.ndhwc_concat",
    "cleanup.layout_pass_set_2.stridedslice_pre_concat",
    "cleanup.layout_pass_set_2.split_mixed_pre_concat",
    "cleanup.layout_pass_set_2.concat_input_adapter",
    "cleanup.layout_pass_set_2.slice_logistic_concat_tail",
    "cleanup.layout_pass_set_2.sa_pa_mirrorpad",
    "cleanup.layout_pass_set_2.squeeze_reshape_identity",
    "cleanup.layout_pass_set_2.prune_reconcile",
    "cleanup.terminal.dequant",
    "cleanup.terminal.qdq",
    "cleanup.terminal.conv_affine",
    "cleanup.terminal.conv_activation",
    "cleanup.terminal.pre_argmax",
    "cleanup.terminal.transpose_gather_channel_fanout",
    "cleanup.terminal.softmax_transpose",
    "cleanup.terminal.boundary_input_normalization",
    "cleanup.terminal.boundary_input_channel_slice",
    "cleanup.terminal.internal_channel_slice",
    "cleanup.terminal.channel_slice_muladd_bridge",
    "cleanup.terminal.boundary_stridedslice_qdq_concat",
    "cleanup.terminal.swish_residual_concat_closure",
    "cleanup.terminal.dequant_logistic_mul_quantize_bridge",
    "cleanup.terminal.swish_qdq_island",
    "cleanup.terminal.instancenorm_post_bias",
    "cleanup.terminal.normalization_pad",
    "cleanup.terminal.instancenorm_residual_add",
    "cleanup.terminal.instancenorm_residual_mul_concat",
    "cleanup.terminal.instancenorm_dualstats",
    "cleanup.terminal.batchmatmul_affine_input",
    "cleanup.terminal.batchmatmul_reshape_se",
    "cleanup.terminal.batchmatmul_adj_flags",
    "cleanup.terminal.qkv_split_conv_concat_bridge",
    "cleanup.terminal.sinet_hardswish_se",
    "cleanup.terminal.dequant_hardsigmoid_bridge",
    "shape_topology.terminal.indexed_convergence",
    "cleanup.very_late.residual_affine_prelu",
    "cleanup.very_late.residual_affine_fanout",
    "cleanup.very_late.prune_reconcile",
    "cleanup.post_cleanup.csp_attention",
    "cleanup.post_cleanup.sa_pa_mirrorpad",
    "layout.no_layout.safe_transpose_reduction",
    "shape_reconciliation.primary.very_late_broadcast",
    "shape_reconciliation.primary.shared_late",
    "shape_reconciliation.primary.late_binary_repair",
    "shape_reconciliation.primary.late_binary_layout_recovery",
    "shape_reconciliation.terminal.expand_squeeze",
    "shape_reconciliation.primary.very_late_final",
    "shape_reconciliation.primary.post_split_fallback",
    "shape_topology.fallback.norm",
    "topology_layout.fallback.post_dynamic_rank1",
    "shape_reconciliation.fallback.broadcast",
    "topology_layout.fallback.broadcast",
    "shape_reconciliation.fallback.se_fc_gather",
    "shape_reconciliation.fallback.placeholder_matmul",
    "topology.fallback.post_placeholder",
    "shape_reconciliation.fallback.conv_input",
    "shape_reconciliation.fallback.mixed_concat",
    "shape_reconciliation.fallback.concat_axis",
    "shape_reconciliation.fallback.binary_layout",
    "topology.fallback.post_layout_repair",
    "shape_topology.fallback.high_rank_batch_matmul",
    "layout_validation.fallback.terminal",
    "topology.primary.post_lowering",
    "topology.primary.no_layout_post_reduction",
    "topology_layout.primary.absolute_final",
    "shape_reconciliation.primary.final_convinteger",
    "topology_layout.primary.final_convinteger",
    "shape_reconciliation.primary.final_instancenorm",
    "topology_layout.primary.final_instancenorm",
    "shape_reconciliation.primary.final_broadcast",
    "topology_layout.primary.final_broadcast",
    "shape_reconciliation.primary.final_mixed_singleton_concat",
    "shape_reconciliation.primary.final_placeholder_binary",
    "topology.primary.final_placeholder",
    "shape_reconciliation.primary.final_se_fc_gather",
    "shape_reconciliation.primary.final_prelu",
    "shape_reconciliation.primary.final_consecutive_reshape",
    "shape_reconciliation.primary.final_sinet_late_residual",
    "shape_reconciliation.primary.final_sinet_preadd_fanout",
    "shape_reconciliation.primary.final_sinet_dual_resize",
    "shape_reconciliation.primary.final_sinet_shared_post",
    "shape_reconciliation.primary.final_sinet_deep_skip",
    "shape_reconciliation.primary.final_sinet_concat_resize",
    "shape_topology.primary.final_high_rank_batch_matmul",
    "shape_topology.primary.final_pad_layout",
    "shape_topology.primary.final_conv_input",
    "shape_topology.primary.final_mixed_concat",
    "shape_topology.primary.final_concat_axis",
    "shape_topology.primary.final_binary_layout",
    "layout_validation.primary.terminal",
)


def _lowerer() -> ast.FunctionDef:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _is_phase_result_record(statement: ast.stmt) -> bool:
    call = _statement_call(statement)
    return bool(
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
    )


def _session() -> ConversionSession:
    graph = helper.make_graph([], "phase_results", [], [])
    return ConversionSession(
        onnx_model=helper.make_model(graph),
        model_ir=ModelIR("phase_results"),
        shape_map={},
        dtype_map={},
        constants={},
    )


def test_one_hundred_eighteen_observations_use_the_bounded_session_store() -> None:
    lowerer = _lowerer()
    records = sorted(
        [
            node
            for node in ast.walk(lowerer)
            if isinstance(node, ast.Expr) and _is_phase_result_record(node)
        ],
        key=lambda node: node.lineno,
    )

    assert len(records) == 118
    assert tuple(
        ast.literal_eval(_statement_call(node).args[0]) for node in records
    ) == EXPECTED_PHASE_IDS
    nested_calls = tuple(_statement_call(node).args[1] for node in records)
    assert all(isinstance(call, ast.Call) for call in nested_calls)
    assert tuple(
        call.func.id
        for call in nested_calls
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name)
    ) == EXPECTED_OWNERS
    assert tuple(
        ast.unparse(call.args[0])
        for call in nested_calls
        if isinstance(call, ast.Call)
    ) == EXPECTED_MODEL_ARGUMENTS
    assert not any(
        isinstance(node, ast.Name) and node.id in EXPECTED_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_phase_result_store_is_bounded_integer_only_and_snapshot_isolated() -> None:
    session = _session()
    source = {"changed": 1, "cycle_detected": 0}

    assert session.phase_results_snapshot() == {}
    session.record_phase_result("topology.primary", source)
    source["changed"] = 99
    assert session.phase_results_snapshot() == {
        "topology.primary": {"changed": 1, "cycle_detected": 0}
    }

    snapshot = session.phase_results_snapshot()
    snapshot["topology.primary"]["changed"] = 77
    assert session.phase_results_snapshot()["topology.primary"]["changed"] == 1

    with pytest.raises(ValueError):
        session.record_phase_result("", {"changed": 1})
    with pytest.raises(ValueError):
        session.record_phase_result(
            "too_many_counters",
            {f"counter_{index}": index for index in range(33)},
        )
    with pytest.raises(TypeError):
        session.record_phase_result("non_integer", {"details": []})

    for index in range(1, 128):
        session.record_phase_result(f"phase_{index}", {"changed": index})
    with pytest.raises(ValueError):
        session.record_phase_result("phase_128", {"changed": 128})
