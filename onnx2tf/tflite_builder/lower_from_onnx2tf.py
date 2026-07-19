from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.utils.onnx_graph_repair import (
    repair_missing_torchvision_nms_guard_captures,
    repair_missing_torchvision_paste_masks_loop_captures,
)
from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.node import NodeView as _NodeWrap
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
    _clone_quantization,
    _permute_shape,
    _read_transpose_perm,
)
from onnx2tf.tflite_builder.core.progress import (
    ProgressSpinner as _ProgressSpinner,
    create_progress_bar as _create_progress_bar,
)
from onnx2tf.tflite_builder.core.onnx_analysis import (
    _align_boundary_signature_to_current_shape,
    _build_onnx_boundary_shape_signature_map,
    _collect_constant_arrays,  # noqa: F401 - compatibility re-export
    _collect_dynamic_boundary_tensor_names,
    _dtype_from_onnx_elem_type,  # noqa: F401 - compatibility re-export
    _extract_tensor_info,
    _graph_has_missing_rank_info,  # noqa: F401 - compatibility re-export
    _infer_missing_tensor_ranks_with_axis_constraints,  # noqa: F401 - compatibility re-export
    _infer_shapes_with_fallback,
    _node_attr_int,  # noqa: F401 - compatibility re-export
    _node_attr_ints,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.core.shape_readiness import (
    reconcile_shape_sensitive_inputs_on_demand,
)
from onnx2tf.tflite_builder.dispatcher import dispatch_node
from onnx2tf.tflite_builder.op_registry import (
    NodeValidationError,
    get_custom_op_candidate_ops,  # noqa: F401 - compatibility re-export
    get_supported_onnx_ops,  # noqa: F401 - compatibility re-export
    resolve_node_dispatch,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
)
from onnx2tf.tflite_builder.op_families.constant import lower_constant_node
from onnx2tf.tflite_builder.passes.precision import (
    _restore_precision_sensitive_reciprocal_divisions,
    _rewrite_constant_divisors_to_multiplicative_reciprocals,
)
from onnx2tf.tflite_builder.passes.recurrent_alias import (
    repair_orphan_recurrent_step_tensors,
)
from onnx2tf.tflite_builder.passes.unbound_input_layout import (
    find_unbound_nonconstant_operator_inputs,
    repair_unbound_nonconstant_inputs_with_layout_transpose,
)
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather as _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather_pass,
    _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains as _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains_pass,
    _optimize_shufflenet_transpose_shuffle_chains as _optimize_shufflenet_transpose_shuffle_chains_pass,
    _repair_nchw_channel_shuffle_concat_gathers as _repair_nchw_channel_shuffle_concat_gathers_pass,
    run_stale_nchw_channel_shuffle_repair,
)
from onnx2tf.tflite_builder.passes.mean_layout import (
    _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains as _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains_pass,
    _optimize_transpose_mean_prepost_nhwc_passthrough_chains as _optimize_transpose_mean_prepost_nhwc_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.layernorm_layout import (
    _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains as _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains_pass,
    _optimize_transpose_layernorm_stats_nhwc_propagation_chains as _optimize_transpose_layernorm_stats_nhwc_propagation_chains_pass,
    run_layernorm_statistics_layout_cleanup,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.passes.instance_normalization_layout import (
    _repair_decomposed_instance_normalization_layouts as _repair_decomposed_instance_normalization_layouts_pass,
)
from onnx2tf.tflite_builder.passes.instance_norm_prepost_layout import (
    _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains as _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains_pass,
    _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains as _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains_pass,
    _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains as _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains_pass,
    _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains as _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.instance_norm_post_bias_layout import (
    optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains as _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.instance_norm_residual_add_layout import (
    optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains as _optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.instance_norm_residual_mul_concat_layout import (
    optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains as _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.instance_norm_dual_stats_layout import (
    optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains as _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.affine_chain_fold import (
    optimize_fold_mul_add_mul_affine_chains as _optimize_fold_mul_add_mul_affine_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv_mul_affine_fold_compat import (
    optimize_fold_conv_mul_add_affine_chains as _optimize_fold_conv_mul_add_affine_chains_pass,
)
from onnx2tf.tflite_builder.passes.activation_fusion import (
    optimize_fuse_activation_chains as _optimize_fuse_activation_chains_pass,
)
from onnx2tf.tflite_builder.passes.dynamic_reshape_resolution import (
    _resolve_reshape_new_shape_from_static_input as _resolve_reshape_new_shape_from_static_input_pass,
    resolve_dynamic_reshape_shapes as _resolve_dynamic_reshape_shapes_pass,
)
from onnx2tf.tflite_builder.passes import (
    static_shape_reconciliation as _static_shape_reconciliation_pass,
)
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    run_static_shape_topology_reconciliation,
)
from onnx2tf.tflite_builder.passes.hardswish_shape_sanitization import (
    sanitize_hardswish_tensor_shapes as _sanitize_hardswish_tensor_shapes_pass,
)
from onnx2tf.tflite_builder.passes.squeeze_shape_sanitization import (
    sanitize_squeeze_axes_with_static_input_shapes as _sanitize_squeeze_axes_with_static_input_shapes_pass,
)
from onnx2tf.tflite_builder.passes.static_shape_signature_sanitization import (
    realign_dynamic_boundary_shape_signature_map as _realign_dynamic_boundary_shape_signature_map_pass,
    sanitize_static_shape_signature_consistency as _sanitize_static_shape_signature_consistency_pass,
)
from onnx2tf.tflite_builder.passes.expand_squeeze_reshape import (
    replace_expand_dims_and_squeeze_with_reshape as _replace_expand_dims_and_squeeze_with_reshape_pass,
)
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains as _optimize_transpose_mul_add_const_prepost_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.affine_post_add_layout import (
    optimize_transpose_mul_posttranspose_add_nhwc_chains as _optimize_transpose_mul_posttranspose_add_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_bridge_layout import (
    _optimize_concat_mul_add_transpose_nhwc_bridge_chains as _optimize_concat_mul_add_transpose_nhwc_bridge_chains_pass,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_transpose_add_bridge_layout import (
    _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains as _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains_pass,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_add_mean_reshape_layout import (
    _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains as _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains_pass,
)
from onnx2tf.tflite_builder.passes.concat_tree_mul_add_bridge_layout import (
    _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains as _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains_pass,
)
from onnx2tf.tflite_builder.passes.stridedslice_pad_concat_bridge_layout import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains as _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.reshape_transpose_collapse_layout import (
    _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains as _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains_pass,
)
from onnx2tf.tflite_builder.passes.attention_gather_cleanup_layout import (
    _optimize_attention_gather_transpose_reshape_cleanup_chains as _optimize_attention_gather_transpose_reshape_cleanup_chains_pass,
)
from onnx2tf.tflite_builder.passes.attention_preproj_ranklift_layout import (
    _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains as _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains_pass,
)
from onnx2tf.tflite_builder.passes.elementwise_roundtrip_nchw_nhwc_layout import (
    _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains as _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LayoutRecoveryContext,
    run_layout_recovery_prefix,
    run_layout_reshape_attention_recovery_prefix,
)
from onnx2tf.tflite_builder.passes.late_binary_layout_recovery import (
    run_late_binary_layout_recovery,
)
from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
    run_attention_gate_qdq_recovery,
    run_preadd_mean_attention_recovery,
)
from onnx2tf.tflite_builder.passes.quantized_recovery_orchestration import (
    run_quantized_activation_binary_recovery,
    run_safe_binary_recovery,
)
from onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration import (
    run_qlinear_mean_concat_recovery,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LayoutAttentionQuantizedSuffixContext,
    run_layout_attention_quantized_suffix,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
    run_terminal_slice_concat_recovery,
)
from onnx2tf.tflite_builder.passes.terminal_affine_concat_split_recovery_orchestration import (
    run_terminal_affine_concat_split_recovery,
    summarize_terminal_affine_concat_split_mutations,
)
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    run_sinet_preadd_resize_recovery,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
    run_sinet_terminal_layout_recovery,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_unary_relu_orchestration import (
    run_terminal_clamp_unary_relu,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_maxpool_reshape_orchestration import (
    run_terminal_singleton_maxpool_reshape,
)
from onnx2tf.tflite_builder.passes.late_dequant_unary_fanout_orchestration import (
    run_late_dequant_unary_fanout,
)
from onnx2tf.tflite_builder.passes.transpose_unary_fanout_orchestration import (
    run_transpose_unary_fanout,
)
from onnx2tf.tflite_builder.passes.late_spp_concat_unary_conv_orchestration import (
    run_late_spp_concat_unary_conv,
    summarize_late_spp_concat_unary_conv_mutations,
)
from onnx2tf.tflite_builder.passes.boundary_batchmatmul_unary_orchestration import (
    run_boundary_batchmatmul_unary,
)
from onnx2tf.tflite_builder.passes.channel_slice_pad_mul_orchestration import (
    run_channel_slice_pad_mul,
    summarize_channel_slice_pad_mul_mutations,
)
from onnx2tf.tflite_builder.passes.late_hard_activation_layout_orchestration import (
    run_late_hard_activation_layout,
    summarize_late_hard_activation_layout_mutations,
)
from onnx2tf.tflite_builder.passes.absolute_final_normalization_attention_orchestration import (
    run_absolute_final_normalization_attention,
)
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    run_qkv_attention,
    summarize_qkv_attention_mutations,
)
from onnx2tf.tflite_builder.passes.duplicate_quantized_prelu_orchestration import (
    run_duplicate_quantized_prelu,
)
from onnx2tf.tflite_builder.passes.very_late_gather_constant_normalization_orchestration import (
    run_very_late_gather_constant_normalization,
    summarize_very_late_gather_constant_normalization_mutations,
)
from onnx2tf.tflite_builder.passes.se_fc_gather_channel_fanout_orchestration import (
    run_se_fc_gather_channel_fanout,
)
from onnx2tf.tflite_builder.passes.terminal_boundary_layout_orchestration import (
    run_terminal_boundary_layout,
)
from onnx2tf.tflite_builder.passes.late_layout_mean_spp_gather_constant_cast_orchestration import (
    run_late_layout_mean_spp_gather_constant_cast,
    summarize_late_layout_mean_spp_gather_constant_cast_mutations,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    run_singleton_consecutive_reshape,
)
from onnx2tf.tflite_builder.passes.gate_layout_orchestration import (
    run_elementwise_gate_layout_cleanup,  # noqa: F401 - compatibility re-export
    run_gate_layout,
    run_late_ndhwc_cost_volume_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_concat_layout_orchestration import (
    run_late_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_reshape_layout_orchestration import (
    run_late_reshape_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_attention_layout_orchestration import (
    run_late_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_window_layout_orchestration import (
    run_late_window_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.final_boundary_channel_layout_orchestration import (
    run_final_boundary_channel_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    run_channel_shuffle_gather,
)
from onnx2tf.tflite_builder.passes.mean_attention_orchestration import (
    run_mean_attention,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_orchestration import (
    run_singleton_reshape,
)
from onnx2tf.tflite_builder.passes.binary_bridge_layout import (
    optimize_transpose_binary_bridges as _optimize_transpose_binary_bridges_pass,
    optimize_transpose_binary_asymmetric_fanout_bridges as _optimize_transpose_binary_asymmetric_fanout_bridges_pass,
    optimize_transpose_binary_full_post_fanout_bridges as _optimize_transpose_binary_full_post_fanout_bridges_pass,
    optimize_transpose_binary_mixed_fanout_bridges_safe as _optimize_transpose_binary_mixed_fanout_bridges_safe_pass,
    optimize_transpose_binary_single_post_bridges_safe as _optimize_transpose_binary_single_post_bridges_safe_pass,
    optimize_transpose_binary_symmetric_legacy_only_bridges_safe as _optimize_transpose_binary_symmetric_legacy_only_bridges_safe_pass,
)
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    repair_rank4_channelwise_broadcast_constants_to_runtime_layout as _repair_rank4_channelwise_broadcast_constants_to_runtime_layout_pass,
    repair_rank4_binary_layout_mismatch_with_transpose_adapter as _repair_rank4_binary_layout_mismatch_with_transpose_adapter_pass,
    repair_rank4_binary_singleton_broadcast_layout_mismatch as _repair_rank4_binary_singleton_broadcast_layout_mismatch_pass,
    run_indexed_binary_layout_adapter_cleanup,
)
from onnx2tf.tflite_builder.passes.conv_output_passthrough_layout import (
    optimize_transposeconv_output_channel1_terminal_transpose_chains as _optimize_transposeconv_output_channel1_terminal_transpose_chains_pass,
    optimize_transposeconv_output_nhwc_passthrough_chains as _optimize_transposeconv_output_nhwc_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.convpool_output_passthrough_compat import (
    optimize_convpool_output_transpose_nhwc_passthrough_chains as _optimize_convpool_output_transpose_nhwc_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.split_channelwise_layout import (
    optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw as _optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw_pass,
    optimize_transpose_split_channelwise_tail_to_single_post_nchw as _optimize_transpose_split_channelwise_tail_to_single_post_nchw_pass,
    optimize_transpose_unary_split_concat_single_post_nchw as _optimize_transpose_unary_split_concat_single_post_nchw_pass,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains as _optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains_pass,
    optimize_transpose_relu_split_all_outputs_to_nhwc_chains as _optimize_transpose_relu_split_all_outputs_to_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.split_conv_concat_bridge_layout import (
    optimize_split_conv_concat_transpose_bridge_to_single_post_nchw as _optimize_split_conv_concat_transpose_bridge_to_single_post_nchw_pass,
)
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    optimize_gelu_tanh_transpose_passthrough_chains as _optimize_gelu_tanh_transpose_passthrough_chains_pass,
    optimize_swish_transpose_passthrough_chains as _optimize_swish_transpose_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.center_size_offset_layout import (
    optimize_center_size_offset_terminal_transpose_chains as _optimize_center_size_offset_terminal_transpose_chains_pass,
)
from onnx2tf.tflite_builder.passes.leakyrelu_passthrough_layout import (
    optimize_leakyrelu_transpose_passthrough_chains as _optimize_leakyrelu_transpose_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.prelu_passthrough_layout import (
    optimize_prelu_transpose_passthrough_chains as _optimize_prelu_transpose_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.elementwise_concat_layout import (
    optimize_transpose_elementwise_concat_conv_nhwc_groups as _optimize_transpose_elementwise_concat_conv_nhwc_groups_pass,
)
from onnx2tf.tflite_builder.passes.stridedslice_concat_layout import (
    optimize_transpose_stridedslice_pre_concat_nhwc_chains as _optimize_transpose_stridedslice_pre_concat_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.split_mixed_concat_layout import (
    optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains as _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.concat_input_adapter_layout import (
    optimize_transpose_input_chains_pre_concat_to_single_post_adapter as _optimize_transpose_input_chains_pre_concat_to_single_post_adapter_pass,
)
from onnx2tf.tflite_builder.passes.pre_add_direct_unary_layout import (
    optimize_transpose_pre_add_direct_unary_nhwc_chains as _optimize_transpose_pre_add_direct_unary_nhwc_chains_pass,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.passes.pre_add_layout import (
    optimize_transpose_pre_add_nhwc_chains as _optimize_transpose_pre_add_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.dual_pre_add_layout import (
    optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains as _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.terminal_affine_fc_layout import (
    optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains as _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.terminal_prelu_bmm_layout import (
    optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains as _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.terminal_affine_prelu_layout import (
    optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains as _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains_pass,
)
from onnx2tf.tflite_builder.passes.mean_affine_prepost_layout import (
    optimize_transpose_mean_mul_add_const_prepost_nhwc_chains as _optimize_transpose_mean_mul_add_const_prepost_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.mean_hardsigmoid_muladd_layout import (
    optimize_transpose_mean_hardsigmoid_muladd_chains as _optimize_transpose_mean_hardsigmoid_muladd_chains_pass,
)
from onnx2tf.tflite_builder.passes.batchmatmul_affine_input_layout import (
    optimize_batchmatmul_affine_transpose_input_chains as _optimize_batchmatmul_affine_transpose_input_chains_pass,
)
from onnx2tf.tflite_builder.passes.batchmatmul_se_layout import (
    optimize_batchmatmul_reshape_se_nhwc_chains as _optimize_batchmatmul_reshape_se_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.batchmatmul_adjoint_layout import (
    optimize_batchmatmul_transpose_input_to_adj_flags as _optimize_batchmatmul_transpose_input_to_adj_flags_pass,
)
from onnx2tf.tflite_builder.passes.probable_nhwc_axis_sanitizer import (
    sanitize_probable_nhwc_axis_sensitive_ops as _sanitize_probable_nhwc_axis_sensitive_ops_pass,
)
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains as _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains_pass,
)
from onnx2tf.tflite_builder.passes.residual_affine_prelu_layout import (
    optimize_transpose_pre_add_mul_add_prelu_nhwc_chains as _optimize_transpose_pre_add_mul_add_prelu_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.residual_affine_fanout_layout import (
    optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains as _optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.pre_unary_affine_fanout_layout import (
    optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains as _optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.pre_add_mulconst_reshape_suffix_compat_layout import (
    optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat as _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat_pass,
)
from onnx2tf.tflite_builder.passes.pre_unary_reshape_suffix_compat_layout import (
    optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat as _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat_pass,
)
from onnx2tf.tflite_builder.passes.pre_unary_squeeze_suffix_compat_layout import (
    optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat as _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat_pass,
)
from onnx2tf.tflite_builder.passes.expanddims_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat as _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat_pass,
)
from onnx2tf.tflite_builder.passes.flatten_hw_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat as _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat_pass,
)
from onnx2tf.tflite_builder.passes.attention_qkv_reshape_compat_layout import (
    optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat as _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat_pass,
)
from onnx2tf.tflite_builder.passes.slice_logistic_concat_reshape_tail_layout import (
    optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains as _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    optimize_sinet_late_residual_pre_add_mul_add_prelu_chains as _optimize_sinet_late_residual_pre_add_mul_add_prelu_chains_pass,
    optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains as _optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains_pass,
    optimize_sinet_shuffle_residual_transpose_chains as _optimize_sinet_shuffle_residual_transpose_chains_pass,
)
from onnx2tf.tflite_builder.passes.singleton_gate_layout import (
    optimize_singleton_gate_conv_concat_nhwc_bridge_blocks as _optimize_singleton_gate_conv_concat_nhwc_bridge_blocks_pass,
)
from onnx2tf.tflite_builder.passes.sinet_deep_skip_layout import (
    optimize_sinet_deep_skip_concat_resize_affine_tail_chains as _optimize_sinet_deep_skip_concat_resize_affine_tail_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_dual_resize_layout import (
    optimize_sinet_deep_skip_dual_resize_affine_transpose_chains as _optimize_sinet_deep_skip_dual_resize_affine_transpose_chains_pass,
    optimize_sinet_dual_resize_affine_transpose_chains as _optimize_sinet_dual_resize_affine_transpose_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_preadd_fanout_layout import (
    optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains as _optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_shared_post_layout import (
    optimize_sinet_shared_post_prelu_transpose_fanout_chains as _optimize_sinet_shared_post_prelu_transpose_fanout_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_concat_resize_layout import (
    optimize_sinet_concat_resize_affine_transpose_chains as _optimize_sinet_concat_resize_affine_transpose_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_tail_concat_layout import (
    optimize_sinet_concat_resize_affine_tail_concat_transpose_chains as _optimize_sinet_concat_resize_affine_tail_concat_transpose_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_softmax_mask_layout import (
    optimize_sinet_softmax_mask_residual_nhwc_tail_chains as _optimize_sinet_softmax_mask_residual_nhwc_tail_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_mix_attention_layout import (
    optimize_sinet_mix_attention_double_logistic_nhwc_chains as _optimize_sinet_mix_attention_double_logistic_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.sinet_sa_pa_mirrorpad_layout import (
    optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains as _optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains_pass,
)
from onnx2tf.tflite_builder.passes.terminal_mean_layout import (
    _optimize_transpose_pre_unary_mean_terminal_nhwc_chains as _optimize_transpose_pre_unary_mean_terminal_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.se_layout import (
    _optimize_transpose_se_conv_mul_prepost_nhwc_chains as _optimize_transpose_se_conv_mul_prepost_nhwc_chains_pass,
    _optimize_transpose_se_fc_mul_prepost_nhwc_chains as _optimize_transpose_se_fc_mul_prepost_nhwc_chains_pass,
    run_se_fc_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.elementwise_gate_layout import (
    _optimize_transpose_logistic_muladd_prepost_nhwc_chains as _optimize_transpose_logistic_muladd_prepost_nhwc_chains_pass,
    _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains as _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains_pass,
    _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains as _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains_pass,
    _optimize_transpose_weighted_add_swish_prepost_nhwc_chains as _optimize_transpose_weighted_add_swish_prepost_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.multi_branch_gate_layout import (
    _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains as _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.dual_postconv_gate_layout import (
    _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains as _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains_pass,
    _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains as _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.ndhwc_gate_layout import (
    _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains as _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains_pass,
    _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains as _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains_pass,
    run_ndhwc_gate_layout_cleanup,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.passes.cost_volume_scatter_layout import (
    _optimize_transpose_cost_volume_scatter_ndhwc_chains as _optimize_transpose_cost_volume_scatter_ndhwc_chains_pass,
    run_cost_volume_scatter_layout_cleanup,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.passes.add_concat_suffix_layout import (
    _optimize_transpose_add_concat_const_suffix_nhwc_chains as _optimize_transpose_add_concat_const_suffix_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.dual_mul_concat_layout import (
    _optimize_transpose_dual_mul_concat_prepost_nhwc_chains as _optimize_transpose_dual_mul_concat_prepost_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.axis3_const_concat_layout import (
    _optimize_transpose_axis3_const_concat_bridge_nhwc_chains as _optimize_transpose_axis3_const_concat_bridge_nhwc_chains_pass,
    run_axis3_const_concat_layout_cleanup,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.passes.concat_global_pool_layout import (
    _repair_nchw_concat_global_pool_conv_axes as _repair_nchw_concat_global_pool_conv_axes_pass,
)
from onnx2tf.tflite_builder.passes.concat_transpose_conv_layout import (
    _repair_nchw_concat_transpose_conv_axes as _repair_nchw_concat_transpose_conv_axes_pass,
)
from onnx2tf.tflite_builder.passes.mixed_singleton_concat_layout import (
    _repair_mixed_singleton_nchw_inputs_for_nhwc_concat as _repair_mixed_singleton_nchw_inputs_for_nhwc_concat_pass,
)
from onnx2tf.tflite_builder.passes.window_partition_layout import (
    _optimize_window_partition_reshape_transpose_to_space_to_depth_chains as _optimize_window_partition_reshape_transpose_to_space_to_depth_chains_pass,
    _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains as _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains as _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains_pass,
    _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains as _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains_pass,
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains as _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv1d_instance_norm_layout import (
    _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains as _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv1d_tencoder_layout import (
    _optimize_tencoder_add_expand_transpose_conv_nhwc_chains as _optimize_tencoder_add_expand_transpose_conv_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains as _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.decoder_deconv_layout import (
    _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input as _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input_pass,
)
from onnx2tf.tflite_builder.passes.terminal_squeeze_mean_layout import (
    _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains as _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.dequant_concat_quantize_layout import (
    _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains as _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains_pass,
    run_dequant_concat_quantize_layout_cleanup,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.passes.qlinear_concat_conv_compat import (
    optimize_nhwc_propagation_qlinear_concat_conv as _optimize_nhwc_propagation_qlinear_concat_conv_pass,
)
from onnx2tf.tflite_builder.passes.qlinear_silu_prefix_layout import (
    _optimize_nhwc_prefix_qlinear_silu_chains as _optimize_nhwc_prefix_qlinear_silu_chains_pass,
)
from onnx2tf.tflite_builder.passes.mean_maxpool_concat_layout import (
    _optimize_transpose_mean_maxpool_concat_conv_chains as _optimize_transpose_mean_maxpool_concat_conv_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv_input_adapter_repair import (
    _repair_singleton_nhwc_conv_input_reshapes as _repair_singleton_nhwc_conv_input_reshapes_pass,
    _repair_stale_nchw_to_nhwc_conv_input_transposes as _repair_stale_nchw_to_nhwc_conv_input_transposes_pass,
    _run_indexed_conv_input_adapter_repairs as _run_indexed_conv_input_adapter_repairs_pass,
)
from onnx2tf.tflite_builder.passes.mixed_concat_input_repair import (
    _repair_mixed_nhwc_inputs_for_nchw_concat as _repair_mixed_nhwc_inputs_for_nchw_concat_pass,
)
from onnx2tf.tflite_builder.passes.stale_binary_adapter_repair import (
    _repair_stale_nchw_to_nhwc_channelwise_binary_transposes as _repair_stale_nchw_to_nhwc_channelwise_binary_transposes_pass,
)
from onnx2tf.tflite_builder.passes.concat_unary_conv_layout import (
    _optimize_transpose_concat_unary_fanout_conv_nhwc_chains as _optimize_transpose_concat_unary_fanout_conv_nhwc_chains_pass,
    run_concat_unary_conv_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.spp_layout import (
    _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains as _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains_pass,
    run_spp_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.ndhwc_concat_layout import (
    _optimize_transpose_pre_concat_ndhwc_chains as _optimize_transpose_pre_concat_ndhwc_chains_pass,
    run_ndhwc_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_layout import (
    run_nhwc_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_legacy_layout import (
    optimize_transpose_pre_concat_nhwc_chains_legacy as _optimize_transpose_pre_concat_nhwc_chains_legacy_pass,
)
from onnx2tf.tflite_builder.passes.slice_prepost_layout import (
    optimize_transpose_slice_prepost_nhwc_passthrough_chains as _optimize_transpose_slice_prepost_nhwc_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.shape_extract_layout import (
    optimize_transpose_shape_extract_nhwc_to_nchw_chains as _optimize_transpose_shape_extract_nhwc_to_nchw_chains_pass,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_quantized_layout import (
    run_nhwc_concat_quantized_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    _is_identity_perm,  # noqa: F401 - compatibility re-export
    _is_inverse_perm,  # noqa: F401 - compatibility re-export
    _optimize_layout_transpose_chains as _optimize_layout_transpose_chains_pass,
    _optimize_trailing_output_transpose_passthrough_chains as _optimize_trailing_output_transpose_passthrough_chains_pass,
    _optimize_transpose_gather_transpose_axis_remap_nhwc_chains as _optimize_transpose_gather_transpose_axis_remap_nhwc_chains_pass,
    _optimize_transpose_gather_transpose_nhwc_channel_chains as _optimize_transpose_gather_transpose_nhwc_channel_chains_pass,
    _optimize_transpose_unary_binary_full_post_fanout_bridges as _optimize_transpose_unary_binary_full_post_fanout_bridges_pass,
    _optimize_transpose_unary_fanout_inverse_post_bridges as _optimize_transpose_unary_fanout_inverse_post_bridges_pass,
    _optimize_transpose_unary_passthrough_chains as _optimize_transpose_unary_passthrough_chains_pass,
    run_layout_transpose_cleanup,
    run_trailing_output_transpose_cleanup,  # noqa: F401 - compatibility re-export
    run_transpose_gather_channel_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import (
    _optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains as _optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains_pass,
    _optimize_transpose_instancenorm_pad_prepost_nhwc_chains as _optimize_transpose_instancenorm_pad_prepost_nhwc_chains_pass,
    _optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains as _optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains_pass,
    _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains as _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains_pass,
    _optimize_transpose_pad_prepost_nhwc_chains as _optimize_transpose_pad_prepost_nhwc_chains_pass,
    _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains as _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains_pass,
    repair_channel_last_inputs_for_channel_first_pad,
    run_normalization_pad_layout_cleanup,
    run_pad_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.quantized_layout import (
    repair_channel_last_convinteger_input_transposes,
)
from onnx2tf.tflite_builder.passes.quantized_activation import (
    optimize_transpose_dequant_hardsigmoid_quantize_bridges,
    optimize_transpose_dequant_mul_add_prelu_quantize_bridges,
    optimize_transpose_dequant_relu_quantize_bridges,
)
from onnx2tf.tflite_builder.passes.quantized_gate import (
    optimize_transpose_dequant_logistic_mul_quantize_bridges,
)
from onnx2tf.tflite_builder.passes.quantized_swish_layout import (
    optimize_transpose_swish_qdq_nhwc_islands as _optimize_transpose_swish_qdq_nhwc_islands_pass,
    optimize_transpose_swish_residual_concat_closure_nhwc_chains as _optimize_transpose_swish_residual_concat_closure_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.conv_input_layout import (
    sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv as _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv_pass,
)
from onnx2tf.tflite_builder.passes.quantized_prelu import (
    _optimize_dequant_prelu_depthwise_quantize_chains as _optimize_dequant_prelu_depthwise_quantize_chains_pass,
    _optimize_dequant_prelu_quantize_chains as _optimize_dequant_prelu_quantize_chains_pass,
    _optimize_transpose_dequant_prelu_quantize_bridges as _optimize_transpose_dequant_prelu_quantize_bridges_pass,
    _optimize_transpose_dequant_prelu_transpose_bridges as _optimize_transpose_dequant_prelu_transpose_bridges_pass,
    run_quantized_prelu_cleanup,
)
from onnx2tf.tflite_builder.passes.quantized_reshape import (
    _optimize_dequant_reshape_quantize_chains as _optimize_dequant_reshape_quantize_chains_pass,
    run_quantized_reshape_cleanup,
)
from onnx2tf.tflite_builder.passes.high_rank_binary import (
    coalesce_static_high_rank_binary_operators,
)
from onnx2tf.tflite_builder.passes.high_rank_matmul import (
    _compress_static_high_rank_batch_matmul as _compress_static_high_rank_batch_matmul_pass,
)
from onnx2tf.tflite_builder.passes.dynamic_reshape import (
    restore_placeholder_matmul_flattened_inputs as _restore_placeholder_matmul_flattened_inputs_pass,
    rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs as _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs_pass,
)
from onnx2tf.tflite_builder.passes.gather_reshape_cleanup import (
    _optimize_gather_axis0_singleton_to_reshape_input_chains as _optimize_gather_axis0_singleton_to_reshape_input_chains_pass,
)
from onnx2tf.tflite_builder.passes.terminal_softmax_layout import (
    _optimize_terminal_softmax_transpose_after_nhwc_propagation as _optimize_terminal_softmax_transpose_after_nhwc_propagation_pass,
)
from onnx2tf.tflite_builder.passes.softmax_transpose_canonicalization import (
    _canonicalize_softmax_transpose_chains as _canonicalize_softmax_transpose_chains_pass,
)
from onnx2tf.tflite_builder.passes.terminal_argmax_layout import (
    _optimize_transpose_pre_argmax_nhwc_terminal_chains as _optimize_transpose_pre_argmax_nhwc_terminal_chains_pass,
)
from onnx2tf.tflite_builder.passes.quantized_pool import (
    _optimize_dequant_maxpool_quantize_chains as _optimize_dequant_maxpool_quantize_chains_pass,
)
from onnx2tf.tflite_builder.passes.quantized_logistic import (
    _optimize_dequant_logistic_quantize_chains as _optimize_dequant_logistic_quantize_chains_pass,
)
from onnx2tf.tflite_builder.passes.quantized_softmax import (
    _optimize_dequant_softmax_quantize_chains as _optimize_dequant_softmax_quantize_chains_pass,
)
from onnx2tf.tflite_builder.passes.quantized_hardsigmoid import (
    _optimize_dequant_hardsigmoid_quantize_chains as _optimize_dequant_hardsigmoid_quantize_chains_pass,
)
from onnx2tf.tflite_builder.passes.quantized_transpose_conv import (
    _optimize_dequant_transposeconv_quantize_chains as _optimize_dequant_transposeconv_quantize_chains_pass,
)
from onnx2tf.tflite_builder.passes.split_fallback import (
    replace_unsupported_split_with_slice as _replace_unsupported_split_with_slice_pass,
)
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    _optimize_consecutive_reshape_passthrough_chains as _optimize_consecutive_reshape_passthrough_chains_pass,
    _optimize_fold_consecutive_mul_constants_chains as _optimize_fold_consecutive_mul_constants_chains_pass,
    _optimize_fuse_pseudo_leakyrelu_chains as _optimize_fuse_pseudo_leakyrelu_chains_pass,
    _optimize_maximum_with_zero_input2_to_relu as _optimize_maximum_with_zero_input2_to_relu_pass,
    _optimize_squeeze_unary_reshape_passthrough_chains as _optimize_squeeze_unary_reshape_passthrough_chains_pass,
    _optimize_squeeze_reshape_identity_chains as _optimize_squeeze_reshape_identity_chains_pass,
    _optimize_maximum_minimum_relu0to1_chains as _optimize_maximum_minimum_relu0to1_chains_pass,
    _optimize_duplicate_reshape_fanout as _optimize_duplicate_reshape_fanout_pass,
    _optimize_duplicate_transpose_fanout as _optimize_duplicate_transpose_fanout_pass,
    prune_dead_operators as _prune_dead_operators_pass,
    run_consecutive_reshape_cleanup,
    run_consecutive_mul_constants_cleanup,
    run_duplicate_fanout_cleanup,
    run_squeeze_reshape_identity_cleanup,
)
from onnx2tf.tflite_builder.passes.prune_reconcile import (
    run_indexed_prune_reconcile_cleanup,
)
from onnx2tf.tflite_builder.passes.topology_layout_refresh import (
    run_topology_layout_refresh,
)
from onnx2tf.tflite_builder.passes.topology_layout_validation import (
    run_topology_layout_validation,
)
from onnx2tf.tflite_builder.passes.singleton_maxpool_layout import (
    _optimize_singleton_layout_reshape_maxpool_binary_cast_chains as _optimize_singleton_layout_reshape_maxpool_binary_cast_chains_pass,
    _optimize_singleton_nms_maxpool_nhwc_chains as _optimize_singleton_nms_maxpool_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    _optimize_singleton_channel_layout_transpose_to_reshape as _optimize_singleton_channel_layout_transpose_to_reshape_pass,
    _optimize_consecutive_inverse_singleton_layout_reshapes as _optimize_consecutive_inverse_singleton_layout_reshapes_pass,
    _optimize_flatten_concat_expanddims_to_nhwc_concat as _optimize_flatten_concat_expanddims_to_nhwc_concat_pass,
    _optimize_singleton_layout_reshape_unary_passthrough_chains as _optimize_singleton_layout_reshape_unary_passthrough_chains_pass,
    _optimize_singleton_reshape_concat_post_transpose_nhwc_chains as _optimize_singleton_reshape_concat_post_transpose_nhwc_chains_pass,
    _optimize_singleton_spatial_nhwc_transpose_reshape_flatten as _optimize_singleton_spatial_nhwc_transpose_reshape_flatten_pass,
)
from onnx2tf.tflite_builder.passes.cast_cleanup import (
    _optimize_redundant_int32_to_int64_passthrough_cast_chains as _optimize_redundant_int32_to_int64_passthrough_cast_chains_pass,
    _optimize_redundant_int64_to_int32_cast_chains as _optimize_redundant_int64_to_int32_cast_chains_pass,
)
from onnx2tf.tflite_builder.passes.quantization_cleanup import (
    _optimize_concat_pre_quantize_dequantize as _optimize_concat_pre_quantize_dequantize_pass,
    _optimize_terminal_quantize_dequantize as _optimize_terminal_quantize_dequantize_pass,
    _optimize_transpose_dequantize_mean_quantize_bridges as _optimize_transpose_dequantize_mean_quantize_bridges_pass,
    _quantized_tensors_share_exact_grid as _quantized_tensors_share_exact_grid_pass,
    _sanitize_terminal_transpose_before_dequantize as _sanitize_terminal_transpose_before_dequantize_pass,
    run_terminal_quantize_dequantize_cleanup,
)
from onnx2tf.tflite_builder.passes.transpose_qdq_bridge_layout import (
    optimize_transpose_quant_dequant_bridges as _optimize_transpose_quant_dequant_bridges_pass,
)
from onnx2tf.tflite_builder.passes.attention_layout import (
    _optimize_transpose_csp_attention_nhwc_chains as _optimize_transpose_csp_attention_nhwc_chains_pass,
    _optimize_transpose_conv_attention_nhwc_propagation_chains as _optimize_transpose_conv_attention_nhwc_propagation_chains_pass,
    _optimize_attention_qkv_gather_reshape_transpose_hoist_chains as _optimize_attention_qkv_gather_reshape_transpose_hoist_chains_pass,
    _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains as _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains_pass,
    _optimize_attention_qkv_shared_pretranspose_slice_nchw_chains as _optimize_attention_qkv_shared_pretranspose_slice_nchw_chains_pass,
    _optimize_attention_qkv_slice_replace_gather_reshape_chains as _optimize_attention_qkv_slice_replace_gather_reshape_chains_pass,
    _optimize_attention_qkv_slice_to_split_chains as _optimize_attention_qkv_slice_to_split_chains_pass,
    _optimize_attention_split_post_reshape_collapse_chains as _optimize_attention_split_post_reshape_collapse_chains_pass,
    _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains as _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains_pass,
    run_mixed_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    _optimize_asin_transpose_passthrough_chains as _optimize_asin_transpose_passthrough_chains_pass,
    _optimize_erf_transpose_passthrough_chains as _optimize_erf_transpose_passthrough_chains_pass,
    _optimize_hardsigmoid_transpose_passthrough_chains as _optimize_hardsigmoid_transpose_passthrough_chains_pass,
    _optimize_hardsigmoid_mul_transpose_passthrough_chains as _optimize_hardsigmoid_mul_transpose_passthrough_chains_pass,
    _optimize_hardswish_transpose_passthrough_chains as _optimize_hardswish_transpose_passthrough_chains_pass,
    _optimize_leading_input_transpose_passthrough_chains as _optimize_leading_input_transpose_passthrough_chains_pass,
)
from onnx2tf.tflite_builder.passes.hardswish_se_layout import (
    optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains as _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains_pass,
)
from onnx2tf.tflite_builder.passes.boundary_input_layout import (
    _optimize_boundary_input_layout_transposes as _optimize_boundary_input_layout_transposes_pass,
)
from onnx2tf.tflite_builder.passes.boundary_input_chains import (
    _optimize_boundary_input_transpose_batchmatmul_chains as _optimize_boundary_input_transpose_batchmatmul_chains_pass,
    _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains as _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains_pass,
    run_boundary_input_normalization_cleanup,
)
from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    _optimize_boundary_input_transpose_channel_slice_blocks as _optimize_boundary_input_transpose_channel_slice_blocks_pass,
    _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks as _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks_pass,
    _optimize_internal_transpose_channel_slice_nhwc_propagation_chains as _optimize_internal_transpose_channel_slice_nhwc_propagation_chains_pass,
    _optimize_transpose_channel_slice_dual_add_bridges_strict as _optimize_transpose_channel_slice_dual_add_bridges_strict_pass,
    _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains as _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains_pass,
    _optimize_transpose_slice_muladd_conv_mergeadd_strict as _optimize_transpose_slice_muladd_conv_mergeadd_strict_pass,
    _optimize_transpose_slice_muladd_mergeadd_posttranspose_strict as _optimize_transpose_slice_muladd_mergeadd_posttranspose_strict_pass,
)
from onnx2tf.tflite_builder.passes.constant_fold import (
    _optimize_mul_square_anchor_constant_chains as _optimize_yolo_decode_mul_square_anchor_chains_pass,
    _optimize_constant_binary_elementwise_chains,  # noqa: F401 - compatibility re-export
    _optimize_constant_input_cast_chains,  # noqa: F401 - compatibility re-export
    _optimize_constant_input_pad_chains,  # noqa: F401 - compatibility re-export
    _optimize_constant_input_pool_chains,  # noqa: F401 - compatibility re-export
    _optimize_constant_input_scatter_nd_chains,  # noqa: F401 - compatibility re-export
)
from onnx2tf.tflite_builder.reporting import (
    build_op_coverage_report as _build_op_coverage_report,
    build_tensor_correspondence_report as _build_tensor_correspondence_report,
    write_op_coverage_report as _write_op_coverage_report,
    write_tensor_correspondence_report as _write_tensor_correspondence_report,
)






def _get_protected_boundary_tensor_names(model_ir: ModelIR) -> set[str]:
    raw_names = model_ir.metadata.get(
        "protected_boundary_tensor_names",
        [],
    )
    if not isinstance(raw_names, list):
        return set()
    return {
        str(name)
        for name in raw_names
        if str(name).strip() != ""
    }


def _append_model_outputs_preserving_order(
    model_ir: ModelIR,
    output_names: List[str],
) -> None:
    if len(output_names) <= 0:
        return
    merged_outputs = [
        str(name)
        for name in list(model_ir.outputs) + list(output_names)
        if str(name) != ""
    ]
    model_ir.outputs = list(dict.fromkeys(merged_outputs))


def _prune_dead_operators(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _prune_dead_operators_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _find_unbound_nonconstant_operator_inputs(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> List[Dict[str, Any]]:
    """
    Detect operator inputs that are neither:
    - produced by another operator,
    - model inputs, nor
    - constant tensors with embedded data.

    Such tensors become unexpected runtime-fed inputs in TFLite and can trigger
    errors like "Input tensor N lacks data".
    """
    return find_unbound_nonconstant_operator_inputs(
        model_ir,
        graph_index=graph_index,
    )


def _repair_orphan_recurrent_step_tensors(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """
    Repair orphan recurrent step aliases left behind by late graph-output rewrites.

    Some recurrent lowerings materialize both:
    - an internal step tensor such as `*_h_step_9`, and
    - a user-visible graph output such as `hidden_out`
      from the same RESHAPE(input, `*_h_step_shape_9`).

    Late cleanup can collapse the producing RESHAPE to the graph output name
    while leaving internal CONCAT users still pointing at the orphaned
    `*_h_step_*` tensor. That later surfaces as `Input tensor N lacks data` in
    TFLite because the orphan tensor has no producer and no constant buffer.
    """
    repaired = repair_orphan_recurrent_step_tensors(
        model_ir,
        graph_index=graph_index,
    )
    return {"repaired_orphan_recurrent_step_tensors": int(repaired)}


def _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """
    Repair a strict subset of unbound dynamic inputs by inserting a layout transpose.

    Current targets (conservative):
    - Unbound tensor `t` is rank-4 NCHW and used as data input of:
      - RESHAPE input[0], or
      - SHAPE input[0], or
      - SPLIT input[1].
    - There exists an earlier produced rank-4 NHWC tensor `s` with
      shape [N,H,W,C] where `t` has shape [N,C,H,W].
    - Insert `TRANSPOSE(s, [0,3,1,2]) -> t` just before the consumer.
    - Unbound tensor `t` is used by MUL fanout (input[0]) and a nearest
      upstream ADD already materializes the corresponding NHWC tensor under a
      renamed `_input_nhwc` output. Insert `TRANSPOSE(s, [0,3,1,2]) -> t`
      to reconnect dropped alias names.
    """
    result = repair_unbound_nonconstant_inputs_with_layout_transpose(
        model_ir,
        graph_index=graph_index,
    )
    if result.repaired > 0:
        _reconcile_static_tensor_shapes(
            model_ir,
            graph_index=result.graph_index,
        )
    return {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": int(
            result.repaired
        )
    }

def _count_ops_by_type(model_ir: ModelIR, op_type: str) -> int:
    target = str(op_type)
    return int(sum(1 for op in model_ir.operators if str(op.op_type) == target))


def _apply_safe_transpose_reduction_lite(model_ir: ModelIR) -> Dict[str, int]:
    """
    Run a conservative transpose-reduction bundle for fallback/non-aggressive paths.

    Safety contract:
    - Apply only a curated set of historically safe transpose/layout passes.
    - Roll back all changes if unbound runtime inputs are introduced.
    """
    before_transpose = _count_ops_by_type(model_ir, "TRANSPOSE")
    if int(before_transpose) <= 0:
        return {
            "safe_transpose_reduction_lite_applied": 0,
            "safe_transpose_reduction_lite_reduced": 0,
            "safe_transpose_reduction_lite_unbound_after": 0,
        }

    snapshot = copy.deepcopy(model_ir)
    pass_sequence = [
        _optimize_transpose_quant_dequant_bridges,
        _optimize_duplicate_transpose_fanout,
        _optimize_singleton_channel_layout_transpose_to_reshape,
        _optimize_transpose_unary_fanout_inverse_post_bridges,
        _optimize_transpose_unary_passthrough_chains,
        _optimize_transpose_elementwise_concat_conv_nhwc_groups,
        _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains,
        _optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
        _optimize_transpose_pre_add_nhwc_chains,
        _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains,
        _optimize_transpose_se_fc_mul_prepost_nhwc_chains,
        _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
        _optimize_transposeconv_output_nhwc_passthrough_chains,
    ]

    applied = 0
    for optimize_pass in pass_sequence:
        optimize_pass(model_ir)
        applied += 1

    _prune_dead_operators(model_ir)
    _reconcile_static_tensor_shapes(model_ir)

    unbound_after = _find_unbound_nonconstant_operator_inputs(model_ir)
    if len(unbound_after) > 0:
        model_ir.tensors = snapshot.tensors
        model_ir.operators = snapshot.operators
        model_ir.inputs = snapshot.inputs
        model_ir.outputs = snapshot.outputs
        model_ir.subgraphs = snapshot.subgraphs
        model_ir.metadata = snapshot.metadata
        return {
            "safe_transpose_reduction_lite_applied": 0,
            "safe_transpose_reduction_lite_reduced": 0,
            "safe_transpose_reduction_lite_unbound_after": int(len(unbound_after)),
        }

    after_transpose = _count_ops_by_type(model_ir, "TRANSPOSE")
    reduced = int(before_transpose - after_transpose)
    if reduced <= 0:
        model_ir.tensors = snapshot.tensors
        model_ir.operators = snapshot.operators
        model_ir.inputs = snapshot.inputs
        model_ir.outputs = snapshot.outputs
        model_ir.subgraphs = snapshot.subgraphs
        model_ir.metadata = snapshot.metadata
        return {
            "safe_transpose_reduction_lite_applied": 0,
            "safe_transpose_reduction_lite_reduced": 0,
            "safe_transpose_reduction_lite_unbound_after": 0,
        }

    return {
        "safe_transpose_reduction_lite_applied": int(applied),
        "safe_transpose_reduction_lite_reduced": int(reduced),
        "safe_transpose_reduction_lite_unbound_after": 0,
    }
































def _resolve_reshape_new_shape_from_static_input(
    new_shape: List[int],
    input_signature: Optional[List[int]],
    allow_zero: Optional[bool] = None,
) -> Optional[List[int]]:
    return _resolve_reshape_new_shape_from_static_input_pass(
        new_shape,
        input_signature,
        allow_zero,
    )


def _resolve_dynamic_reshape_shapes(
    model_ir: ModelIR,
    prefer_runtime_inferable_from_onnx_raw: bool = False,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return _resolve_dynamic_reshape_shapes_pass(
        model_ir,
        prefer_runtime_inferable_from_onnx_raw,
        graph_index=graph_index,
    )


def _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _sanitize_hardswish_tensor_shapes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return _sanitize_hardswish_tensor_shapes_pass(
        model_ir,
        graph_index=graph_index,
    )


def _sanitize_squeeze_axes_with_static_input_shapes(model_ir: ModelIR) -> Dict[str, int]:
    return _sanitize_squeeze_axes_with_static_input_shapes_pass(model_ir)


def _replace_expand_dims_and_squeeze_with_reshape(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _replace_expand_dims_and_squeeze_with_reshape_pass(
        model_ir,
        layout_state=layout_state,
    )


def _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Compatibility wrapper for the dedicated Conv-input layout owner."""

    return _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv_pass(
        model_ir,
        graph_index=graph_index,
    )


def _repair_rank4_binary_layout_mismatch_with_transpose_adapter(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _repair_rank4_binary_layout_mismatch_with_transpose_adapter_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _repair_rank4_binary_singleton_broadcast_layout_mismatch(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _repair_rank4_binary_singleton_broadcast_layout_mismatch_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _sanitize_static_shape_signature_consistency(model_ir: ModelIR) -> Dict[str, int]:
    return _sanitize_static_shape_signature_consistency_pass(model_ir)


def _realign_dynamic_boundary_shape_signature_map(model_ir: ModelIR) -> Dict[str, int]:
    return _realign_dynamic_boundary_shape_signature_map_pass(model_ir)


def _replace_unsupported_split_with_slice(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _replace_unsupported_split_with_slice_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _infer_slice_output_shape_and_resolved_params(
    input_shape: Optional[List[int]],
    begin_vals: Optional[List[int]],
    size_vals: Optional[List[int]],
) -> Tuple[Optional[List[int]], Optional[List[int]], Optional[List[int]]]:
    return (
        _static_shape_reconciliation_pass
        ._infer_slice_output_shape_and_resolved_params(
            input_shape,
            begin_vals,
            size_vals,
        )
    )


def _infer_slice_output_signature(
    *,
    input_shape: Optional[List[int]],
    input_signature: Optional[List[int]],
    begin_vals: Optional[List[int]],
    size_vals: Optional[List[int]],
) -> Optional[List[int]]:
    return _static_shape_reconciliation_pass._infer_slice_output_signature(
        input_shape=input_shape,
        input_signature=input_signature,
        begin_vals=begin_vals,
        size_vals=size_vals,
    )


def _infer_batch_matmul_output_shape_and_signature(
    shape_a: Optional[List[int]],
    shape_b: Optional[List[int]],
    signature_a: Optional[List[int]],
    signature_b: Optional[List[int]],
    adj_x: bool,
    adj_y: bool,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    return (
        _static_shape_reconciliation_pass
        ._infer_batch_matmul_output_shape_and_signature(
            shape_a,
            shape_b,
            signature_a,
            signature_b,
            adj_x,
            adj_y,
        )
    )


def _infer_rank4_signature_from_input(
    *,
    input_signature: Optional[List[int]],
    output_shape: Optional[List[int]],
    existing_output_signature: Optional[List[int]] = None,
    propagate_channel: bool = False,
) -> Optional[List[int]]:
    return _static_shape_reconciliation_pass._infer_rank4_signature_from_input(
        input_signature=input_signature,
        output_shape=output_shape,
        existing_output_signature=existing_output_signature,
        propagate_channel=propagate_channel,
    )


def _normalize_reduce_axes_for_rank(
    axes: Optional[List[int]],
    rank: int,
) -> Optional[List[int]]:
    return _static_shape_reconciliation_pass._normalize_reduce_axes_for_rank(
        axes,
        rank,
    )


def _infer_reduce_output_shape_and_signature(
    *,
    input_shape: Optional[List[int]],
    input_signature: Optional[List[int]],
    axes: Optional[List[int]],
    keep_dims: bool,
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    return (
        _static_shape_reconciliation_pass
        ._infer_reduce_output_shape_and_signature(
            input_shape=input_shape,
            input_signature=input_signature,
            axes=axes,
            keep_dims=keep_dims,
        )
    )


def _parse_axes_option(raw_axes: Any) -> List[int]:
    return _static_shape_reconciliation_pass._parse_axes_option(raw_axes)


def _infer_squeeze_output_shape_and_signature(
    *,
    input_shape: Optional[List[int]],
    input_signature: Optional[List[int]],
    squeeze_axes: Optional[List[int]],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    return (
        _static_shape_reconciliation_pass
        ._infer_squeeze_output_shape_and_signature(
            input_shape=input_shape,
            input_signature=input_signature,
            squeeze_axes=squeeze_axes,
        )
    )


def _infer_conv_out_dim(
    in_size: int,
    kernel_size: int,
    stride: int,
    dilation: int,
    padding: str,
) -> Optional[int]:
    return _static_shape_reconciliation_pass._infer_conv_out_dim(
        in_size,
        kernel_size,
        stride,
        dilation,
        padding,
    )


def _reconcile_static_tensor_shapes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    include_mutation_count: bool = False,
) -> Dict[str, int]:
    return _static_shape_reconciliation_pass.reconcile_static_tensor_shapes(
        model_ir,
        graph_index=graph_index,
        include_mutation_count=include_mutation_count,
    )


def _run_indexed_shape_convergence_cleanup(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    prune_stats = _prune_dead_operators(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    first_reconcile_stats = _reconcile_static_tensor_shapes(
        model_ir,
        graph_index=graph_index,
    )
    reshape_stats = _resolve_dynamic_reshape_shapes(
        model_ir,
        graph_index=graph_index,
    )
    final_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if _stats_have_positive_count(
        prune_stats,
        first_reconcile_stats,
        reshape_stats,
    ):
        final_reconcile_stats = _reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    return {
        "removed_dead_operators": int(
            prune_stats.get("removed_dead_operators", 0)
        ),
        "resolved_dynamic_reshape_shapes": int(
            reshape_stats.get("resolved_dynamic_reshape_shapes", 0)
        ),
        "reconciled_static_tensor_shapes": int(
            first_reconcile_stats.get("reconciled_static_tensor_shapes", 0)
            + final_reconcile_stats.get("reconciled_static_tensor_shapes", 0)
        ),
    }


def _run_indexed_final_shape_activation_convergence(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """
    Run the terminal metadata/fusion convergence with one graph index.

    Shape sanitation and reconciliation do not change graph topology. The
    final activation fusion is the only structural mutation and updates the
    supplied index differentially before the last reconciliation.
    """

    graph_index = ModelIRGraphIndex(model_ir)
    convergence_stats = _run_indexed_shape_convergence_cleanup(
        model_ir,
        layout_state=layout_state,
        graph_index=graph_index,
    )
    hardswish_stats = _sanitize_hardswish_tensor_shapes(
        model_ir,
        graph_index=graph_index,
    )
    first_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if _stats_have_positive_count(
        convergence_stats,
        hardswish_stats,
    ):
        first_reconcile_stats = _reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    reshape_stats = _resolve_dynamic_reshape_shapes(
        model_ir,
        graph_index=graph_index,
    )
    second_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if _stats_have_positive_count(
        first_reconcile_stats,
        reshape_stats,
    ):
        second_reconcile_stats = _reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    fusion_tensor_count = len(model_ir.tensors)
    fusion_stats = _optimize_fuse_conv_activation_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    final_reconcile_stats = {"reconciled_static_tensor_shapes": 0}
    if (
        _stats_have_positive_count(
            second_reconcile_stats,
            fusion_stats,
        )
        or len(model_ir.tensors) < fusion_tensor_count
    ):
        final_reconcile_stats = _reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
    return {
        **convergence_stats,
        "sanitized_hardswish_tensor_shapes": int(
            hardswish_stats.get("sanitized_hardswish_tensor_shapes", 0)
        ),
        "resolved_dynamic_reshape_shapes": int(
            convergence_stats.get("resolved_dynamic_reshape_shapes", 0)
            + reshape_stats.get("resolved_dynamic_reshape_shapes", 0)
        ),
        "reconciled_static_tensor_shapes": int(
            convergence_stats.get("reconciled_static_tensor_shapes", 0)
            + first_reconcile_stats.get("reconciled_static_tensor_shapes", 0)
            + second_reconcile_stats.get("reconciled_static_tensor_shapes", 0)
            + final_reconcile_stats.get("reconciled_static_tensor_shapes", 0)
        ),
        **fusion_stats,
    }


def _restore_placeholder_matmul_flattened_inputs(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _restore_placeholder_matmul_flattened_inputs_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_quant_dequant_bridges(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_quant_dequant_bridges_pass(model_ir)


def _optimize_duplicate_transpose_fanout(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_duplicate_transpose_fanout_pass(model_ir)


def _optimize_duplicate_reshape_fanout(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_duplicate_reshape_fanout_pass(model_ir)


def _optimize_transpose_dequant_prelu_quantize_bridges(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_dequant_prelu_quantize_bridges_pass(model_ir)


def _optimize_transpose_dequant_relu_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """
    Fold transpose wrappers around DEQUANTIZE->(RELU|RELU6)->QUANTIZE chains.

    Target pattern:
      Xq --Transpose(P)--> Aq --DEQUANTIZE--> A --(RELU|RELU6)--> B --QUANTIZE--> Bq --Transpose(inv(P))--> Yq

    Rewritten:
      Xq --DEQUANTIZE--> A --(RELU|RELU6)--> B --QUANTIZE--> Yq

    Safety conditions:
    - Chain is linear (single consumer at each bridge tensor)
    - Pre/Post transpose permutations are exact inverses
    - Quantized tensors use per-tensor quantization only
    """
    return optimize_transpose_dequant_relu_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

def _optimize_transpose_dequant_hardsigmoid_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return optimize_transpose_dequant_hardsigmoid_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )

def _optimize_transpose_dequant_mul_add_prelu_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return optimize_transpose_dequant_mul_add_prelu_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )


def _optimize_transpose_dequant_prelu_transpose_bridges(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_dequant_prelu_transpose_bridges_pass(model_ir)


def _optimize_transpose_dequant_logistic_mul_quantize_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return optimize_transpose_dequant_logistic_mul_quantize_bridges(
        model_ir,
        graph_index=graph_index,
    )


def _optimize_transpose_swish_qdq_nhwc_islands(
    model_ir: ModelIR,
    *,
    min_spatial_stage: int = 160,
    require_concat_closure: bool = False,
) -> Dict[str, int]:
    return _optimize_transpose_swish_qdq_nhwc_islands_pass(
        model_ir,
        min_spatial_stage=int(min_spatial_stage),
        require_concat_closure=bool(require_concat_closure),
    )


def _optimize_transpose_swish_residual_concat_closure_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_swish_residual_concat_closure_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_binary_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed transpose-binary bridge owner."""

    return _optimize_transpose_binary_bridges_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_binary_asymmetric_fanout_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed asymmetric fan-out bridge owner."""

    return _optimize_transpose_binary_asymmetric_fanout_bridges_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_binary_full_post_fanout_bridges(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed full-post fan-out bridge owner."""

    return _optimize_transpose_binary_full_post_fanout_bridges_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_binary_single_post_bridges_safe(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed single-post bridge owner."""

    return _optimize_transpose_binary_single_post_bridges_safe_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_binary_mixed_fanout_bridges_safe(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed mixed fan-out bridge owner."""

    return _optimize_transpose_binary_mixed_fanout_bridges_safe_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_binary_symmetric_legacy_only_bridges_safe(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed legacy-only bridge owner."""

    return _optimize_transpose_binary_symmetric_legacy_only_bridges_safe_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_maximum_minimum_relu0to1_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_maximum_minimum_relu0to1_chains_pass(model_ir)


def _optimize_maximum_with_zero_input2_to_relu(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_maximum_with_zero_input2_to_relu_pass(model_ir)


def _optimize_fuse_pseudo_leakyrelu_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_fuse_pseudo_leakyrelu_chains_pass(model_ir)


def _optimize_yolo_decode_mul_square_anchor_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_yolo_decode_mul_square_anchor_chains_pass(model_ir)


def _optimize_fold_consecutive_mul_constants_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_fold_consecutive_mul_constants_chains_pass(model_ir)
def _optimize_leading_input_transpose_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_leading_input_transpose_passthrough_chains_pass(model_ir)


def _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_boundary_input_transpose_mul_sum_reshape_nhwc_chains_pass(model_ir)


def _optimize_boundary_input_transpose_batchmatmul_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_boundary_input_transpose_batchmatmul_chains_pass(model_ir)


def _optimize_asin_transpose_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_asin_transpose_passthrough_chains_pass(model_ir)


def _optimize_erf_transpose_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_erf_transpose_passthrough_chains_pass(model_ir)


def _optimize_hardswish_transpose_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_hardswish_transpose_passthrough_chains_pass(model_ir)


def _optimize_hardsigmoid_mul_transpose_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_hardsigmoid_mul_transpose_passthrough_chains_pass(model_ir)


def _optimize_hardsigmoid_transpose_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_hardsigmoid_transpose_passthrough_chains_pass(model_ir)


def _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pad_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_unary_pad_prepost_to_single_adapter_nhwc_chains_pass(model_ir)


def _optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_norm_subgraph_pad_prepost_nhwc_chains_pass(model_ir)


def _optimize_swish_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_swish_transpose_passthrough_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_gelu_tanh_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_gelu_tanh_transpose_passthrough_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_center_size_offset_terminal_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_center_size_offset_terminal_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_leakyrelu_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_leakyrelu_transpose_passthrough_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_prelu_transpose_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_prelu_transpose_passthrough_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_elementwise_concat_conv_nhwc_groups(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_elementwise_concat_conv_nhwc_groups_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_pre_concat_nhwc_chains_legacy(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_nhwc_chains_legacy_pass(model_ir)


def _optimize_transpose_pre_concat_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Any = None,
    diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, int]:
    """Run indexed bounded families before the remaining legacy families."""

    indexed_stats = run_nhwc_concat_layout_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    quantized_indexed_stats = run_nhwc_concat_quantized_layout_cleanup(
        model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    legacy_stats = _optimize_transpose_pre_concat_nhwc_chains_legacy(model_ir)
    indexed_stats_keys = (
        "optimized_transpose_pre_concat_nhwc_direct_chains",
        "optimized_transpose_pre_concat_nhwc_unary_chains",
        "optimized_transpose_pre_concat_nhwc_pad_chains",
        "optimized_transpose_pre_concat_nhwc_dequantize_chains",
        "optimized_transpose_pre_concat_nhwc_prelu_chains",
        "optimized_transpose_pre_concat_nhwc_softmax_chains",
        "optimized_transpose_pre_concat_nhwc_swish_chains",
        "optimized_transpose_pre_concat_nhwc_slice_chains",
        "optimized_transpose_pre_concat_nhwc_split_chains",
        "optimized_transpose_pre_concat_nhwc_add_chains",
        "optimized_transpose_pre_concat_nhwc_leaky_chains",
    )
    quantized_indexed_stats_keys = (
        "optimized_transpose_pre_concat_nhwc_quantized_direct_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_unary_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_pad_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_unary_pad_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_all_pad_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_swish_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_dequantize_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_prelu_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_softmax_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_leaky_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_slice_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_split_chains",
        "optimized_transpose_pre_concat_nhwc_quantized_add_chains",
    )
    optimized = sum(
        int(indexed_stats.get(stats_key, 0))
        for stats_key in indexed_stats_keys
    )
    optimized += sum(
        int(quantized_indexed_stats.get(stats_key, 0))
        for stats_key in quantized_indexed_stats_keys
    )
    optimized += int(
        legacy_stats.get("optimized_transpose_pre_concat_nhwc_chains", 0)
    )
    return {
        "optimized_transpose_pre_concat_nhwc_chains": int(optimized)
    }


def _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pre_dequant_concat_quantize_post_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_concat_unary_fanout_conv_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_resize_add_concat_affine_conv_spp_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_pre_concat_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_pre_concat_ndhwc_chains_pass(model_ir)


def _optimize_transpose_slice_prepost_nhwc_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_slice_prepost_nhwc_passthrough_chains_pass(
        model_ir
    )


def _optimize_transpose_shape_extract_nhwc_to_nchw_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_shape_extract_nhwc_to_nchw_chains_pass(
        model_ir
    )


def _optimize_transpose_stridedslice_pre_concat_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Any = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_stridedslice_pre_concat_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_input_chains_pre_concat_to_single_post_adapter_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_shufflenet_transpose_shuffle_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_shufflenet_transpose_shuffle_chains_pass(model_ir)

def _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_shufflenet_reshape_transpose_shuffle_nhwc_chains_pass(
        model_ir
    )

def _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_nchw_channel_shuffle_reshape_transpose_reshape_to_gather_pass(
        model_ir
    )

def _repair_nchw_channel_shuffle_concat_gathers(model_ir: ModelIR) -> Dict[str, int]:
    return _repair_nchw_channel_shuffle_concat_gathers_pass(model_ir)

def _repair_nchw_concat_transpose_conv_axes(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _repair_nchw_concat_transpose_conv_axes_pass(
        model_ir,
        layout_state=layout_state,
    )


def _repair_nchw_concat_global_pool_conv_axes(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _repair_nchw_concat_global_pool_conv_axes_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_gather_transpose_axis_remap_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_gather_transpose_axis_remap_nhwc_chains_pass(model_ir)


def _optimize_transpose_pre_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_add_nhwc_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_dual_pre_add_to_single_post_adapter_nhwc_chains_pass(
        model_ir
    )


def _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_terminal_transpose_mul_add_reshape_fc_nhwc_chains_pass(
        model_ir
    )


def _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_terminal_transpose_prelu_reshape_batchmatmul_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pre_add_mul_add_prelu_nhwc_chains_pass(model_ir)


def _optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains_pass(
        model_ir
    )

def _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return (
        _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains_pass(
            model_ir
        )
    )


def _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_attention_gather_transpose_reshape_cleanup_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_gather_transpose_reshape_cleanup_chains_pass(
        model_ir
    )


def _optimize_gather_axis0_singleton_to_reshape_input_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_gather_axis0_singleton_to_reshape_input_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains_pass(
        model_ir
    )


def _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_window_partition_reshape_transpose_to_space_to_depth_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_tencoder_add_expand_transpose_conv_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_squeeze_unary_reshape_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_squeeze_unary_reshape_passthrough_chains_pass(model_ir)
def _optimize_squeeze_reshape_identity_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_squeeze_reshape_identity_chains_pass(model_ir)


def _optimize_transpose_instancenorm_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Dispatch decomposed InstanceNormalization pre/post tails in graph order."""
    max_total_rewrites = 32
    rewritten = 0
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    owners = (
        (
            _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains_pass,
            "optimized_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains",
        ),
        (
            _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains_pass,
            "optimized_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains",
        ),
        (
            _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains_pass,
            "optimized_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains",
        ),
        (
            _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains_pass,
            "optimized_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains",
        ),
    )
    while rewritten < max_total_rewrites:
        changed = False
        for pre_index in list(active_index.operator_indices("TRANSPOSE")):
            if pre_index < 0 or pre_index >= len(model_ir.operators):
                continue
            pre = model_ir.operators[int(pre_index)]
            if _read_transpose_perm(model_ir, pre) != [0, 3, 1, 2]:
                continue
            for owner, stats_key in owners:
                stats = owner(
                    model_ir,
                    graph_index=active_index,
                    layout_state=layout_state,
                    max_rewrites=1,
                    candidate=pre,
                )
                if int(stats.get(stats_key, 0)) <= 0:
                    continue
                rewritten += 1
                changed = True
                break
            if changed:
                break
        if not changed:
            break
    return {
        "optimized_transpose_instancenorm_prepost_nhwc_chains": int(rewritten)
    }
def _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed decomposed-InstanceNorm post-bias owner."""

    return (
        _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains_pass(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
            max_rewrites=max_rewrites,
            candidate=candidate,
        )
    )


def _optimize_transpose_instancenorm_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_instancenorm_pad_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_flatten_globalnorm_pad_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed InstanceNorm residual-ADD adapter owner."""

    return (
        _optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains_pass(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
            max_rewrites=max_rewrites,
            candidate=candidate,
        )
    )


def _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed InstanceNorm residual-MUL/CONCAT owner."""

    return (
        _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains_pass(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
            max_rewrites=max_rewrites,
            candidate=candidate,
        )
    )


def _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed dual-stat InstanceNorm/residual owner."""

    return (
        _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains_pass(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
            max_rewrites=max_rewrites,
            candidate=candidate,
        )
    )


def _optimize_fold_mul_add_mul_affine_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed floating-point affine-chain fold owner."""

    return _optimize_fold_mul_add_mul_affine_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_mul_add_const_prepost_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed affine pre/post layout owner."""

    return _optimize_transpose_mul_add_const_prepost_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_pad_mul_posttranspose_add_nhwc_chains_pass(model_ir)


def _optimize_transpose_mul_posttranspose_add_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed affine post-ADD layout owner."""

    return _optimize_transpose_mul_posttranspose_add_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_shuffle_residual_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet Shuffle residual layout owner."""

    return _optimize_sinet_shuffle_residual_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet Shuffle post-MUL layout owner."""

    return _optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet late-residual layout owner."""

    return _optimize_sinet_late_residual_pre_add_mul_add_prelu_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )

def _optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet deep-skip layout owner."""

    return _optimize_sinet_deep_skip_concat_resize_affine_tail_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )

def _optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet pre-ADD fan-out layout owner."""

    return _optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )

def _optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed sibling-adapter dual-Resize owner."""

    return _optimize_sinet_deep_skip_dual_resize_affine_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_shared_post_prelu_transpose_fanout_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet shared-post fan-out layout owner."""

    return _optimize_sinet_shared_post_prelu_transpose_fanout_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )

def _optimize_sinet_concat_resize_affine_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet Concat/Resize affine layout owner."""

    return _optimize_sinet_concat_resize_affine_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )

def _optimize_sinet_dual_resize_affine_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed direct-adapter dual-Resize owner."""

    return _optimize_sinet_dual_resize_affine_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_concat_resize_affine_tail_concat_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet two-Concat affine tail owner."""

    return _optimize_sinet_concat_resize_affine_tail_concat_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_softmax_mask_residual_nhwc_tail_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet Softmax-mask residual owner."""

    return _optimize_sinet_softmax_mask_residual_nhwc_tail_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains_pass(
        model_ir
    )


def _optimize_transpose_mean_mul_add_const_prepost_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_mean_mul_add_const_prepost_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_mean_prepost_nhwc_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_mean_prepost_nhwc_passthrough_chains_pass(model_ir)


def _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_mean_mul_reshape_add_conv_nhwc_chains_pass(model_ir)


def _optimize_transpose_layernorm_stats_nhwc_propagation_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_layernorm_stats_nhwc_propagation_chains_pass(model_ir)


def _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_layernorm_stats_via_existing_post_transpose_nhwc_chains_pass(model_ir)


def _optimize_transpose_pre_unary_mean_terminal_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_pre_unary_mean_terminal_nhwc_chains_pass(model_ir)


def _optimize_transpose_se_conv_mul_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_se_conv_mul_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_se_fc_mul_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_se_fc_mul_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_sum_logistic_muladd_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_weighted_add_swish_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_weighted_add_swish_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_nested_weighted_add_swish_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_logistic_muladd_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_logistic_muladd_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_osnet_multi_gate_muladd_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_logistic_sub_muladd_dual_postconv_nhwc_chains_pass(model_ir)


def _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_logistic_sub_mul_postadd_nhwc_chains_pass(model_ir)


def _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_3d_leaky_logistic_muladd_ndhwc_chains_pass(model_ir)


def _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains_pass(model_ir)


def _optimize_transpose_cost_volume_scatter_ndhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_cost_volume_scatter_ndhwc_chains_pass(model_ir)


def _optimize_transpose_conv_attention_nhwc_propagation_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_conv_attention_nhwc_propagation_chains_pass(model_ir)


def _optimize_batchmatmul_affine_transpose_input_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_batchmatmul_affine_transpose_input_chains_pass(model_ir)


def _optimize_batchmatmul_reshape_se_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_batchmatmul_reshape_se_nhwc_chains_pass(model_ir)


def _optimize_batchmatmul_transpose_input_to_adj_flags(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_batchmatmul_transpose_input_to_adj_flags_pass(model_ir)


def _optimize_attention_qkv_gather_reshape_transpose_hoist_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_qkv_gather_reshape_transpose_hoist_chains_pass(model_ir)


def _optimize_attention_qkv_slice_replace_gather_reshape_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_qkv_slice_replace_gather_reshape_chains_pass(model_ir)


def _optimize_attention_qkv_slice_to_split_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_qkv_slice_to_split_chains_pass(model_ir)


def _optimize_attention_split_post_reshape_collapse_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_split_post_reshape_collapse_chains_pass(model_ir)


def _optimize_attention_qkv_shared_pretranspose_slice_nchw_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_qkv_shared_pretranspose_slice_nchw_chains_pass(model_ir)


def _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_attention_qkv_weighted_sum_bridge_to_nhwc_chains_pass(model_ir)


def _optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_split_conv_concat_transpose_bridge_to_single_post_nchw_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_relu_split_all_outputs_to_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return (
        _optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains_pass(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
            max_rewrites=max_rewrites,
            candidate=candidate,
        )
    )


def _optimize_transpose_csp_attention_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_csp_attention_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet SA/PA MirrorPad layout owner."""

    return _optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_sinet_mix_attention_double_logistic_nhwc_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    """Dispatch the indexed SiNet double-Logistic mix-attention owner."""

    return _optimize_sinet_mix_attention_double_logistic_nhwc_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_mixed_mean_reducemax_concat_mirrorpad_nhwc_chains_pass(model_ir)


def _optimize_transpose_add_concat_const_suffix_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_add_concat_const_suffix_nhwc_chains_pass(model_ir)


def _optimize_transpose_dual_mul_concat_prepost_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_dual_mul_concat_prepost_nhwc_chains_pass(model_ir)


def _optimize_transpose_axis3_const_concat_bridge_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_axis3_const_concat_bridge_nhwc_chains_pass(model_ir)


def _optimize_transposeconv_output_nhwc_passthrough_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transposeconv_output_nhwc_passthrough_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transposeconv_output_channel1_terminal_transpose_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: Optional[int] = None,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transposeconv_output_channel1_terminal_transpose_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_singleton_channel_layout_transpose_to_reshape(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_singleton_channel_layout_transpose_to_reshape_pass(model_ir)


def _optimize_singleton_layout_reshape_unary_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_singleton_layout_reshape_unary_passthrough_chains_pass(model_ir)


def _optimize_consecutive_inverse_singleton_layout_reshapes(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_consecutive_inverse_singleton_layout_reshapes_pass(model_ir)


def _optimize_singleton_layout_reshape_maxpool_binary_cast_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_singleton_layout_reshape_maxpool_binary_cast_chains_pass(model_ir)


def _optimize_redundant_int64_to_int32_cast_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_redundant_int64_to_int32_cast_chains_pass(model_ir)
def _optimize_redundant_int32_to_int64_passthrough_cast_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_redundant_int32_to_int64_passthrough_cast_chains_pass(
        model_ir
    )
def _optimize_singleton_nms_maxpool_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_singleton_nms_maxpool_nhwc_chains_pass(model_ir)


def _optimize_consecutive_reshape_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_consecutive_reshape_passthrough_chains_pass(model_ir)


def _optimize_flatten_concat_expanddims_to_nhwc_concat(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_flatten_concat_expanddims_to_nhwc_concat_pass(model_ir)


def _optimize_singleton_spatial_nhwc_transpose_reshape_flatten(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_singleton_spatial_nhwc_transpose_reshape_flatten_pass(model_ir)


def _optimize_singleton_reshape_concat_post_transpose_nhwc_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_singleton_reshape_concat_post_transpose_nhwc_chains_pass(model_ir)


def _optimize_singleton_gate_conv_concat_nhwc_bridge_blocks(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_singleton_gate_conv_concat_nhwc_bridge_blocks_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _repair_mixed_singleton_nchw_inputs_for_nhwc_concat_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_unary_split_concat_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_unary_split_concat_single_post_nchw_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_split_channelwise_tail_to_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_split_channelwise_tail_to_single_post_nchw_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
    max_rewrites: int = 32,
    candidate: Optional[OperatorIR] = None,
) -> Dict[str, int]:
    return _optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=max_rewrites,
        candidate=candidate,
    )


def _sanitize_probable_nhwc_axis_sensitive_ops(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _sanitize_probable_nhwc_axis_sensitive_ops_pass(model_ir)


def _optimize_transpose_unary_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_unary_passthrough_chains_pass(model_ir)


def _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains_pass(
        model_ir
    )


def _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains_pass(
        model_ir
    )


def _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return (
        _repair_rank4_channelwise_broadcast_constants_to_runtime_layout_pass(
            model_ir,
            graph_index=graph_index,
        )
    )


def _repair_decomposed_instance_normalization_layouts(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _repair_decomposed_instance_normalization_layouts_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_convpool_output_transpose_nhwc_passthrough_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_convpool_output_transpose_nhwc_passthrough_chains_pass(
        model_ir
    )


def _optimize_transpose_unary_fanout_inverse_post_bridges(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_unary_fanout_inverse_post_bridges_pass(model_ir)


def _optimize_transpose_unary_binary_full_post_fanout_bridges(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_unary_binary_full_post_fanout_bridges_pass(model_ir)

def _optimize_trailing_output_transpose_passthrough_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_trailing_output_transpose_passthrough_chains_pass(model_ir)

def _optimize_dequant_prelu_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_dequant_prelu_quantize_chains_pass(model_ir)


def _optimize_dequant_prelu_depthwise_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_dequant_prelu_depthwise_quantize_chains_pass(model_ir)


def _optimize_dequant_transposeconv_quantize_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_dequant_transposeconv_quantize_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_dequant_reshape_quantize_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_dequant_reshape_quantize_chains_pass(model_ir)


def _optimize_dequant_hardsigmoid_quantize_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_dequant_hardsigmoid_quantize_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_dequant_maxpool_quantize_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_dequant_maxpool_quantize_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_dequant_softmax_quantize_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_dequant_softmax_quantize_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_dequant_logistic_quantize_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_dequant_logistic_quantize_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _quantized_tensors_share_exact_grid(
    model_ir: ModelIR,
    lhs_name: str,
    rhs_name: str,
) -> bool:
    return _quantized_tensors_share_exact_grid_pass(
        model_ir,
        lhs_name,
        rhs_name,
    )
def _optimize_terminal_quantize_dequantize(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_terminal_quantize_dequantize_pass(model_ir)
def _optimize_fuse_conv_activation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_fuse_activation_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_fold_conv_mul_add_affine_chains(
    model_ir: ModelIR,
    *,
    enable_conv_add_only_fold: bool = True,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_fold_conv_mul_add_affine_chains_pass(
        model_ir,
        enable_conv_add_only_fold=enable_conv_add_only_fold,
        layout_state=layout_state,
    )


def _sanitize_terminal_transpose_before_dequantize(model_ir: ModelIR) -> Dict[str, int]:
    return _sanitize_terminal_transpose_before_dequantize_pass(model_ir)


def _optimize_concat_pre_quantize_dequantize(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_concat_pre_quantize_dequantize_pass(model_ir)


def _optimize_transpose_dequantize_mean_quantize_bridges(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_dequantize_mean_quantize_bridges_pass(model_ir)


def _optimize_transpose_mean_hardsigmoid_muladd_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_transpose_mean_hardsigmoid_muladd_chains_pass(model_ir)


def _optimize_nhwc_propagation_qlinear_concat_conv(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_nhwc_propagation_qlinear_concat_conv_pass(model_ir)


def _repair_singleton_nhwc_conv_input_reshapes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return _repair_singleton_nhwc_conv_input_reshapes_pass(
        model_ir,
        graph_index=graph_index,
    )


def _repair_stale_nchw_to_nhwc_conv_input_transposes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return _repair_stale_nchw_to_nhwc_conv_input_transposes_pass(
        model_ir,
        graph_index=graph_index,
    )


def _run_indexed_conv_input_adapter_repairs(model_ir: ModelIR) -> Dict[str, int]:
    return _run_indexed_conv_input_adapter_repairs_pass(model_ir)


def _repair_mixed_nhwc_inputs_for_nchw_concat(model_ir: ModelIR) -> Dict[str, int]:
    return _repair_mixed_nhwc_inputs_for_nchw_concat_pass(model_ir)


def _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    return _repair_stale_nchw_to_nhwc_channelwise_binary_transposes_pass(
        model_ir,
        graph_index=graph_index,
    )


def _run_indexed_binary_layout_convergence(model_ir: ModelIR) -> Dict[str, int]:
    """Run up to three terminal binary-layout convergence rounds with one index."""

    graph_index = ModelIRGraphIndex(model_ir)
    repaired_constants = 0
    removed_transposes = 0
    reconciled_shapes = 0
    for _ in range(3):
        broadcast_stats = (
            _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
                model_ir,
                graph_index=graph_index,
            )
        )
        transpose_stats = (
            _repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
                model_ir,
                graph_index=graph_index,
            )
        )
        reconcile_stats = _reconcile_static_tensor_shapes(
            model_ir,
            graph_index=graph_index,
        )
        repaired_constants += int(
            broadcast_stats.get(
                "repaired_rank4_channelwise_broadcast_constants",
                0,
            )
        )
        removed_transposes += int(
            transpose_stats.get(
                "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes",
                0,
            )
        )
        reconciled_shapes += int(
            reconcile_stats.get("reconciled_static_tensor_shapes", 0)
        )
        if not _stats_have_positive_count(
            broadcast_stats,
            transpose_stats,
            reconcile_stats,
        ):
            break
    return {
        "repaired_rank4_channelwise_broadcast_constants": int(
            repaired_constants
        ),
        "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes": int(
            removed_transposes
        ),
        "reconciled_static_tensor_shapes": int(reconciled_shapes),
    }


def _optimize_nhwc_prefix_qlinear_silu_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_nhwc_prefix_qlinear_silu_chains_pass(model_ir)


def _optimize_transpose_mean_maxpool_concat_conv_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_mean_maxpool_concat_conv_chains_pass(model_ir)


def _canonicalize_softmax_transpose_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _canonicalize_softmax_transpose_chains_pass(model_ir)


def _optimize_terminal_softmax_transpose_after_nhwc_propagation(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_terminal_softmax_transpose_after_nhwc_propagation_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_pre_argmax_nhwc_terminal_chains(
    model_ir: ModelIR,
    *,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_pre_argmax_nhwc_terminal_chains_pass(
        model_ir,
        layout_state=layout_state,
    )


def _optimize_transpose_gather_transpose_nhwc_channel_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_gather_transpose_nhwc_channel_chains_pass(model_ir)

def _optimize_layout_transpose_chains(model_ir: ModelIR) -> Dict[str, int]:
    return _optimize_layout_transpose_chains_pass(model_ir)


def _optimize_boundary_input_transpose_channel_slice_blocks(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_boundary_input_transpose_channel_slice_blocks_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_internal_transpose_channel_slice_nhwc_propagation_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_internal_transpose_channel_slice_nhwc_propagation_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_transpose_channel_slice_dual_add_bridges_strict(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_channel_slice_dual_add_bridges_strict_pass(model_ir)


def _optimize_transpose_slice_muladd_conv_mergeadd_strict(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_slice_muladd_conv_mergeadd_strict_pass(model_ir)


def _optimize_transpose_slice_muladd_mergeadd_posttranspose_strict(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_transpose_slice_muladd_mergeadd_posttranspose_strict_pass(model_ir)



def _optimize_concat_mul_add_transpose_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_concat_mul_add_transpose_nhwc_bridge_chains_pass(model_ir)



def _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains_pass(
        model_ir
    )


def _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return (
        _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains_pass(
            model_ir
        )
    )


def _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains_pass(
        model_ir
    )


def _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains_pass(
            model_ir
        )
    )


def _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _optimize_boundary_input_layout_transposes(
    model_ir: ModelIR,
) -> Dict[str, int]:
    return _optimize_boundary_input_layout_transposes_pass(model_ir)






def build_op_coverage_report(
    *,
    onnx_graph: onnx.ModelProto,
    output_file_name: str,
    opset_min: int = 13,
    opset_max: int = 18,
    conversion_error: Optional[str] = None,
    allow_custom_ops: bool = False,
    custom_op_allowlist: Optional[List[str]] = None,
    disable_group_convolution: bool = False,
    preprocess_report: Optional[Dict[str, Any]] = None,
    output_nms_with_argmax: bool = False,
    switch_nms_version: str = "v4",
) -> Dict[str, Any]:
    return _build_op_coverage_report(
        onnx_graph=onnx_graph,
        output_file_name=output_file_name,
        opset_min=opset_min,
        opset_max=opset_max,
        conversion_error=conversion_error,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
        disable_group_convolution=disable_group_convolution,
        preprocess_report=preprocess_report,
        output_nms_with_argmax=output_nms_with_argmax,
        switch_nms_version=switch_nms_version,
    )


def write_op_coverage_report(
    *,
    report: Dict[str, Any],
    output_report_path: str,
) -> str:
    return _write_op_coverage_report(
        report=report,
        output_report_path=output_report_path,
    )








def build_tensor_correspondence_report(
    *,
    onnx_graph: onnx.ModelProto,
    model_ir: ModelIR,
) -> Dict[str, Any]:
    return _build_tensor_correspondence_report(
        onnx_graph=onnx_graph,
        model_ir=model_ir,
    )


def write_tensor_correspondence_report(
    *,
    report: Dict[str, Any],
    output_report_path: str,
) -> str:
    return _write_tensor_correspondence_report(
        report=report,
        output_report_path=output_report_path,
    )


def _compress_static_high_rank_batch_matmul(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    return _compress_static_high_rank_batch_matmul_pass(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )


def _stats_have_positive_count(*stats: Dict[str, int]) -> bool:
    """Return whether pure mutation-count dictionaries report a change."""

    return any(
        int(value) > 0
        for result in stats
        for value in result.values()
    )


def lower_onnx_to_ir(
    onnx_graph: onnx.ModelProto,
    output_file_name: str,
    allow_custom_ops: bool = False,
    custom_op_allowlist: Optional[List[str]] = None,
    optimize_layout_transpose_chains: bool = True,
    transpose_inputs_to_nhwc: bool = False,
    keep_ncw_or_nchw_or_ncdhw_input_names: Optional[List[str]] = None,
    keep_nwc_or_nhwc_or_ndhwc_input_names: Optional[List[str]] = None,
    keep_shape_absolutely_input_names: Optional[List[str]] = None,
    disable_group_convolution: bool = False,
    output_nms_with_argmax: bool = False,
    switch_nms_version: str = "v4",
    mvn_epsilon: float = 1e-10,
    show_progress: bool = False,
    apply_safe_transpose_reduction_lite_on_no_layout_opt: bool = False,
    disable_suppression_flextranspose: bool = False,
    number_of_dimensions_after_flextranspose_compression: int = 6,
    disable_suppression_flexstridedslice: bool = False,
    number_of_dimensions_after_flexstridedslice_compression: int = 5,
    optimization_for_gpu_delegate: bool = False,
    replace_argmax_to_reducemax_and_indices_is_int64: bool = False,
    replace_argmax_to_reducemax_and_indices_is_float32: bool = False,
    replace_argmax_to_fused_argmax_and_indices_is_int64: bool = False,
    replace_argmax_to_fused_argmax_and_indices_is_float32: bool = False,
    fused_argmax_scale_ratio: float = 0.5,
    replace_to_pseudo_operators: Optional[List[str]] = None,
    protected_boundary_tensor_names: Optional[List[str]] = None,
    _internal_pass_diagnostics: Optional[List[Dict[str, Any]]] = None,
) -> ModelIR:
    repair_missing_torchvision_nms_guard_captures(onnx_graph)
    repair_missing_torchvision_paste_masks_loop_captures(onnx_graph)
    onnx_graph = _infer_shapes_with_fallback(onnx_graph)

    shape_map, dtype_map = _extract_tensor_info(onnx_graph)
    dynamic_boundary_tensors = _collect_dynamic_boundary_tensor_names(onnx_graph)
    onnx_boundary_signature_map = _build_onnx_boundary_shape_signature_map(
        onnx_graph=onnx_graph,
        shape_map=shape_map,
    )
    constants: Dict[str, np.ndarray] = {}
    for ini in onnx_graph.graph.initializer:
        constants[ini.name] = np.asarray(numpy_helper.to_array(ini))
    # Producer/consumer information is built once by ConversionSession below.
    graph_output_names = [str(o.name) for o in onnx_graph.graph.output]

    model_ir = ModelIR(name=output_file_name)
    model_ir.metadata["tensor_lineage_events"] = []
    model_ir.metadata["onnx_dynamic_input_tensor_names"] = list(
        dynamic_boundary_tensors["inputs"]
    )
    model_ir.metadata["onnx_dynamic_output_tensor_names"] = list(
        dynamic_boundary_tensors["outputs"]
    )
    model_ir.metadata["onnx_boundary_shape_signature_map"] = dict(
        onnx_boundary_signature_map
    )
    model_ir.metadata["original_graph_output_names"] = [
        str(output.name)
        for output in onnx_graph.graph.output
        if str(output.name) != ""
    ]
    model_ir.metadata["protected_boundary_tensor_names"] = list(
        dict.fromkeys(
            [
                str(name)
                for name in (protected_boundary_tensor_names or [])
                if str(name).strip() != ""
            ]
        )
    )
    session = ConversionSession(
        onnx_model=onnx_graph,
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
    )

    def _finalize_model_ir(result: ModelIR) -> ModelIR:
        if _internal_pass_diagnostics is not None:
            _internal_pass_diagnostics.extend(copy.deepcopy(session.diagnostics))
        return result
    ctx = LoweringContext(
        model_ir=model_ir,
        shape_map=shape_map,
        dtype_map=dtype_map,
        constants=constants,
        onnx_model=onnx_graph,
        allow_custom_ops=allow_custom_ops,
        custom_op_allowlist=custom_op_allowlist,
        disable_group_convolution=disable_group_convolution,
        tensor_consumer_count=session.tensor_consumer_count,
        graph_output_names=graph_output_names,
        output_nms_with_argmax=output_nms_with_argmax,
        switch_nms_version=switch_nms_version,
        mvn_epsilon=mvn_epsilon,
        disable_suppression_flextranspose=disable_suppression_flextranspose,
        number_of_dimensions_after_flextranspose_compression=number_of_dimensions_after_flextranspose_compression,
        disable_suppression_flexstridedslice=disable_suppression_flexstridedslice,
        number_of_dimensions_after_flexstridedslice_compression=number_of_dimensions_after_flexstridedslice_compression,
        optimization_for_gpu_delegate=optimization_for_gpu_delegate,
        replace_argmax_to_reducemax_and_indices_is_int64=replace_argmax_to_reducemax_and_indices_is_int64,
        replace_argmax_to_reducemax_and_indices_is_float32=replace_argmax_to_reducemax_and_indices_is_float32,
        replace_argmax_to_fused_argmax_and_indices_is_int64=replace_argmax_to_fused_argmax_and_indices_is_int64,
        replace_argmax_to_fused_argmax_and_indices_is_float32=replace_argmax_to_fused_argmax_and_indices_is_float32,
        fused_argmax_scale_ratio=fused_argmax_scale_ratio,
        replace_to_pseudo_operators=replace_to_pseudo_operators,
        session=session,
    )

    keep_ncw_input_names = {
        str(v) for v in (keep_ncw_or_nchw_or_ncdhw_input_names or [])
    }
    keep_nwc_input_names = {
        str(v) for v in (keep_nwc_or_nhwc_or_ndhwc_input_names or [])
    }
    keep_shape_abs_input_names = {
        str(v) for v in (keep_shape_absolutely_input_names or [])
    }
    input_name_remap: Dict[str, str] = {}

    # Inputs
    initializer_names = {ini.name for ini in onnx_graph.graph.initializer}
    for raw_graph_input in onnx_graph.graph.input:
        # protobuf's generic repeated-field stub can expose the iterator item
        # as ``type[In]`` to Pylance instead of the concrete message type.
        graph_input = cast(onnx.ValueInfoProto, raw_graph_input)
        if graph_input.name in initializer_names:
            continue
        input_name = str(graph_input.name)
        ctx.ensure_tensor(input_name)
        input_tensor = model_ir.tensors[input_name]
        model_ir.inputs.append(input_name)

        if not transpose_inputs_to_nhwc:
            continue

        input_rank = len(list(input_tensor.shape))
        if input_rank not in [3, 4, 5]:
            continue

        direct_input_consumers = session.graph_index.consumers_of(input_name)
        direct_consumer_op_types = {
            str(getattr(consumer, "op_type", ""))
            for consumer in direct_input_consumers
        }
        recurrent_op_types = {"LSTM", "GRU", "RNN"}
        recurrent_layout_neutral_op_types = recurrent_op_types | {
            "Cast",
            "Identity",
            "Shape",
            "Size",
        }
        preserve_recurrent_sequence_layout = bool(
            direct_consumer_op_types & recurrent_op_types
        ) and direct_consumer_op_types.issubset(recurrent_layout_neutral_op_types)
        keep_shape_abs = (
            input_name in keep_shape_abs_input_names
            or preserve_recurrent_sequence_layout
        )
        keep_ncw = input_name in keep_ncw_input_names
        keep_nwc = (
            input_name in keep_nwc_input_names
            and input_name not in keep_ncw_input_names
        )
        if keep_shape_abs or keep_ncw or keep_nwc:
            continue

        original_shape = list(input_tensor.shape)
        original_signature = (
            list(input_tensor.shape_signature)
            if input_tensor.shape_signature is not None
            else list(input_tensor.shape)
        )
        if input_rank == 3:
            perm_internal_to_external = [0, 2, 1]
            perm_external_to_internal = [0, 2, 1]
        elif input_rank == 4:
            perm_internal_to_external = [0, 2, 3, 1]
            perm_external_to_internal = [0, 3, 1, 2]
        else:
            perm_internal_to_external = [0, 2, 3, 4, 1]
            perm_external_to_internal = [0, 4, 1, 2, 3]

        external_shape = _permute_shape(original_shape, perm_internal_to_external)
        external_signature = _permute_shape(
            original_signature, perm_internal_to_external
        )
        if external_shape is None or external_signature is None:
            continue

        input_tensor.shape = list(external_shape)
        input_tensor.shape_signature = list(external_signature)

        internal_input_name = ctx.add_intermediate_tensor(
            f"{input_name}_onnx_ncx_internal",
            dtype=str(input_tensor.dtype),
            shape=original_shape,
        )
        internal_tensor = model_ir.tensors[internal_input_name]
        internal_tensor.shape_signature = list(original_signature)
        internal_tensor.quantization = _clone_quantization(input_tensor.quantization)

        perm_name = ctx.add_const_tensor(
            f"{internal_input_name}_perm",
            np.asarray(perm_external_to_internal, dtype=np.int32),
        )
        ctx.add_operator(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[input_name, perm_name],
                outputs=[internal_input_name],
            )
        )
        input_name_remap[input_name] = internal_input_name
        # Keep ONNX consumer-count semantics for remapped synthetic input names.
        # Several builders use this count to safely elide inverse transpose pairs;
        # without this handoff, synthetic names appear to have zero consumers and
        # can incorrectly drop the boundary transpose required by other branches.
        ctx.tensor_consumer_count[internal_input_name] = int(
            ctx.tensor_consumer_count.get(input_name, 0)
        )

    # Initializers as tensors
    for name, value in constants.items():
        if name not in model_ir.tensors:
            ctx.add_const_tensor(name, value)

    # Nodes
    graph_nodes = list(onnx_graph.graph.node)
    progress_bar = _create_progress_bar(
        total=len(graph_nodes),
        desc="flatbuffer_direct lowering",
        enabled=bool(show_progress),
    )
    lowering_progress_spinner = _ProgressSpinner(progress_bar)
    try:
        lowering_progress_spinner.start()
        for node in graph_nodes:
            try:
                if node.op_type == "Constant":
                    lower_constant_node(node=node, ctx=ctx)
                    continue

                wrapped = _NodeWrap(
                    node,
                    input_name_remap=input_name_remap,
                    shape_map=shape_map,
                    dtype_map=dtype_map,
                )
                reconcile_shape_sensitive_inputs_on_demand(
                    node=wrapped,
                    ctx=ctx,
                )
                try:
                    dispatch_node(wrapped, ctx)
                except NodeValidationError as ve:
                    raise NotImplementedError(
                        f"flatbuffer_direct validation failed: "
                        f"op={ve.node_op} node={ve.node_name} "
                        f"reason_code={ve.reason_code} message={ve.message}"
                    ) from ve
            finally:
                if progress_bar is not None:
                    progress_bar.update(1)
    finally:
        if progress_bar is not None:
            lowering_progress_spinner.stop()
            progress_bar.close()

    post_progress_total = 6 if optimize_layout_transpose_chains else 4
    post_progress_step = 0
    post_progress_bar = _create_progress_bar(
        total=post_progress_total,
        desc="flatbuffer_direct post-lowering",
        enabled=bool(show_progress),
    )
    post_progress_spinner = _ProgressSpinner(post_progress_bar)

    def _set_post_progress_desc(stage_label: str) -> None:
        if post_progress_bar is None:
            return
        post_progress_spinner.start()
        post_progress_bar.set_description_str(
            f"flatbuffer_direct post-lowering [{post_progress_step + 1}/{post_progress_total}] {stage_label}"
        )

    def _advance_post_progress() -> None:
        nonlocal post_progress_step
        if post_progress_bar is None:
            return
        post_progress_spinner.stop()
        post_progress_bar.update(1)
        post_progress_step = int(post_progress_step + 1)

    def _run_mean_attention_layout_pass_cluster(
        *,
        include_layernorm: bool = False,
        include_conv_attention: bool = True,
    ) -> Tuple[Dict[str, int], ...]:
        return run_mean_attention(
            mean_attention_context,
            include_layernorm=include_layernorm,
            include_conv_attention=include_conv_attention,
        )

    def _run_qkv_attention_layout_pass_cluster(
        *,
        include_layout_transpose: bool = False,
        include_prefix: bool = True,
    ) -> Tuple[Dict[str, int], ...]:
        return run_qkv_attention(
            qkv_attention_context,
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        )

    def _run_duplicate_quantized_prelu_pass_cluster(
        *,
        include_transpose: bool,
    ) -> Tuple[Dict[str, int], ...]:
        return run_duplicate_quantized_prelu(
            duplicate_quantized_prelu_context,
            include_transpose=include_transpose,
        )

    def _run_very_late_gather_constant_normalization_pass_cluster() -> Tuple[
        Dict[str, int], ...
    ]:
        return run_very_late_gather_constant_normalization(
            very_late_gather_constant_normalization_context
        )

    def _run_se_fc_gather_channel_fanout_pass_cluster(
        target_model_ir: ModelIR,
        target_layout_state: LayoutState | None,
    ) -> Tuple[Dict[str, int], Dict[str, int]]:
        return run_se_fc_gather_channel_fanout(
            ModelIRPassContext(
                model_ir=target_model_ir,
                layout_state=target_layout_state,
                diagnostics=session.diagnostics,
            )
        )

    def _run_terminal_boundary_layout_pass_cluster() -> Tuple[Dict[str, int], ...]:
        return run_terminal_boundary_layout(terminal_boundary_layout_context)

    def _run_gate_layout_pass_cluster(
        *,
        include_mixed_attention: bool = True,
    ) -> Tuple[Dict[str, int], ...]:
        return run_gate_layout(
            gate_layout_context,
            include_mixed_attention=include_mixed_attention,
        )

    def _run_channel_shuffle_gather_layout_pass_cluster(
        *,
        include_two_way_shuffle: bool = True,
        include_nhwc_shuffle: bool = True,
        include_post_gather_cleanup: bool = False,
    ) -> Tuple[Dict[str, int], ...]:
        return run_channel_shuffle_gather(
            channel_shuffle_gather_context,
            include_two_way_shuffle=include_two_way_shuffle,
            include_nhwc_shuffle=include_nhwc_shuffle,
            include_post_gather_cleanup=include_post_gather_cleanup,
        )

    def _run_transpose_unary_fanout_layout_pass_cluster(
        *,
        include_layout_transpose: bool = False,
        include_unary_passthrough: bool = True,
    ) -> Tuple[Dict[str, int], ...]:
        return run_transpose_unary_fanout(
            transpose_unary_fanout_context,
            include_layout_transpose=include_layout_transpose,
            include_unary_passthrough=include_unary_passthrough,
        )

    def _run_late_dequant_unary_fanout_pass_cluster() -> Tuple[
        Dict[str, int], ...
    ]:
        return run_late_dequant_unary_fanout(
            late_dequant_unary_fanout_context
        )

    def _run_terminal_singleton_maxpool_reshape_pass_pair() -> Tuple[
        Dict[str, int], ...
    ]:
        return run_terminal_singleton_maxpool_reshape(
            terminal_singleton_maxpool_reshape_context
        )

    def _run_terminal_clamp_unary_relu_pass_cluster() -> Tuple[Dict[str, int], ...]:
        return run_terminal_clamp_unary_relu(
            terminal_clamp_unary_relu_context
        )

    def _run_late_layout_mean_spp_gather_constant_cast_pass_cluster(
        *,
        include_layout_transpose: bool,
    ) -> Tuple[Dict[str, int], ...]:
        return run_late_layout_mean_spp_gather_constant_cast(
            late_layout_mean_spp_gather_constant_cast_context,
            include_layout_transpose=include_layout_transpose,
        )

    def _run_late_spp_concat_unary_conv_pass_pair() -> Tuple[Dict[str, int], ...]:
        return run_late_spp_concat_unary_conv(
            late_spp_concat_unary_conv_context
        )

    def _run_late_hard_activation_layout_pass_pair(
        *,
        include_layout_transpose: bool,
    ) -> Tuple[Dict[str, int], ...]:
        return run_late_hard_activation_layout(
            late_hard_activation_layout_context,
            include_layout_transpose=include_layout_transpose,
        )

    def _run_absolute_final_normalization_attention_pass_pair() -> Tuple[Dict[str, int], ...]:
        return run_absolute_final_normalization_attention(
            absolute_final_normalization_attention_context
        )

    def _run_boundary_batchmatmul_unary_layout_pass_cluster() -> Tuple[
        Dict[str, int], ...
    ]:
        return run_boundary_batchmatmul_unary(
            boundary_batchmatmul_unary_context
        )

    def _run_channel_slice_pad_mul_layout_pass_cluster() -> Tuple[Dict[str, int], ...]:
        return run_channel_slice_pad_mul(
            channel_slice_pad_mul_context
        )

    def _run_singleton_reshape_layout_pass_cluster(
        *,
        include_layout_transpose: bool = False,
        include_duplicate_fanout: bool = False,
        include_multi_branch_gate: bool = False,
        include_spatial_concat_post_transpose: bool = True,
    ) -> Tuple[Dict[str, int], ...]:
        return run_singleton_reshape(
            singleton_reshape_context,
            include_layout_transpose=include_layout_transpose,
            include_duplicate_fanout=include_duplicate_fanout,
            include_multi_branch_gate=include_multi_branch_gate,
            include_spatial_concat_post_transpose=(
                include_spatial_concat_post_transpose
            ),
        )

    def _run_singleton_consecutive_reshape_pass_cluster(
        target_model_ir: ModelIR,
        target_layout_state: LayoutState | None,
    ) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
        return run_singleton_consecutive_reshape(
            ModelIRPassContext(
                model_ir=target_model_ir,
                layout_state=target_layout_state,
                diagnostics=session.diagnostics,
            )
        )

    shared_model_ir_pass_context = session.model_ir_pass_context
    boundary_batchmatmul_unary_context = shared_model_ir_pass_context
    channel_slice_pad_mul_context = shared_model_ir_pass_context
    late_hard_activation_layout_context = shared_model_ir_pass_context
    absolute_final_normalization_attention_context = shared_model_ir_pass_context
    qkv_attention_context = shared_model_ir_pass_context
    duplicate_quantized_prelu_context = shared_model_ir_pass_context
    very_late_gather_constant_normalization_context = shared_model_ir_pass_context
    terminal_boundary_layout_context = shared_model_ir_pass_context
    gate_layout_context = shared_model_ir_pass_context
    channel_shuffle_gather_context = shared_model_ir_pass_context
    mean_attention_context = shared_model_ir_pass_context
    singleton_reshape_context = shared_model_ir_pass_context
    late_layout_mean_spp_gather_constant_cast_context = shared_model_ir_pass_context
    layout_recovery_context = LayoutRecoveryContext(
        pass_context=session.model_ir_pass_context,
        boundary_batchmatmul_unary_cluster=(
            _run_boundary_batchmatmul_unary_layout_pass_cluster
        ),
        pre_concat_cleanup=_optimize_transpose_pre_concat_nhwc_chains,
        channel_shuffle_gather_cluster=(
            _run_channel_shuffle_gather_layout_pass_cluster
        ),
    )
    attention_recovery_context = AttentionRecoveryContext(
        pass_context=session.model_ir_pass_context,
        mean_attention_cluster=_run_mean_attention_layout_pass_cluster,
        gate_layout_cluster=_run_gate_layout_pass_cluster,
        transpose_unary_fanout_cluster=(
            _run_transpose_unary_fanout_layout_pass_cluster
        ),
    )
    quantized_recovery_context = shared_model_ir_pass_context
    qlinear_recovery_context = shared_model_ir_pass_context
    terminal_slice_concat_recovery_context = TerminalSliceConcatRecoveryContext(
        pass_context=session.model_ir_pass_context,
        channel_slice_pad_mul_cluster=(
            _run_channel_slice_pad_mul_layout_pass_cluster
        ),
    )
    terminal_affine_concat_split_recovery_context = shared_model_ir_pass_context
    terminal_clamp_unary_relu_context = shared_model_ir_pass_context
    terminal_singleton_maxpool_reshape_context = shared_model_ir_pass_context
    late_dequant_unary_fanout_context = shared_model_ir_pass_context
    transpose_unary_fanout_context = shared_model_ir_pass_context
    late_spp_concat_unary_conv_context = shared_model_ir_pass_context
    sinet_preadd_resize_recovery_context = shared_model_ir_pass_context

    def _run_layout_recovery_prefix_pass_sequence() -> Tuple[Any, ...]:
        return run_layout_recovery_prefix(layout_recovery_context)

    def _run_layout_reshape_attention_recovery_prefix() -> Tuple[Any, ...]:
        return run_layout_reshape_attention_recovery_prefix(
            layout_recovery_context
        )

    def _run_preadd_mean_attention_recovery_sequence() -> Tuple[Any, ...]:
        return run_preadd_mean_attention_recovery(attention_recovery_context)

    def _run_attention_gate_qdq_recovery_sequence() -> Tuple[Any, ...]:
        return run_attention_gate_qdq_recovery(attention_recovery_context)

    def _run_safe_binary_bridge_recovery_sequence() -> Tuple[Any, ...]:
        return run_safe_binary_recovery(quantized_recovery_context)

    def _run_quantized_activation_binary_bridge_recovery_sequence(
    ) -> Tuple[Any, ...]:
        return run_quantized_activation_binary_recovery(
            quantized_recovery_context
        )

    def _run_qlinear_mean_concat_recovery_sequence() -> Tuple[Any, ...]:
        return run_qlinear_mean_concat_recovery(qlinear_recovery_context)

    layout_attention_quantized_suffix_context = (
        LayoutAttentionQuantizedSuffixContext(
            pass_context=session.model_ir_pass_context,
            mean_attention_cluster=_run_mean_attention_layout_pass_cluster,
            attention_gate_qdq_recovery=(
                _run_attention_gate_qdq_recovery_sequence
            ),
            duplicate_quantized_prelu_cluster=(
                _run_duplicate_quantized_prelu_pass_cluster
            ),
        )
    )

    def _run_layout_attention_quantized_recovery_suffix(
        *,
        include_duplicate_transpose: bool,
    ) -> Tuple[Any, ...]:
        return run_layout_attention_quantized_suffix(
            layout_attention_quantized_suffix_context,
            include_duplicate_transpose=include_duplicate_transpose,
        )

    def _run_terminal_slice_concat_layout_recovery_sequence() -> Tuple[Any, ...]:
        return run_terminal_slice_concat_recovery(
            terminal_slice_concat_recovery_context
        )

    def _run_terminal_affine_concat_split_recovery_sequence() -> "Tuple[Dict[str, int], ...]":
        return run_terminal_affine_concat_split_recovery(
            terminal_affine_concat_split_recovery_context
        )

    def _run_sinet_preadd_resize_recovery_sequence() -> Tuple[Dict[str, int], ...]:
        return run_sinet_preadd_resize_recovery(
            sinet_preadd_resize_recovery_context
        )

    sinet_terminal_layout_recovery_context = SINetTerminalLayoutRecoveryContext(
        pass_context=session.model_ir_pass_context,
        preadd_resize_recovery=_run_sinet_preadd_resize_recovery_sequence,
    )

    def _run_sinet_terminal_layout_recovery_sequence() -> Tuple[Any, ...]:
        return run_sinet_terminal_layout_recovery(
            sinet_terminal_layout_recovery_context
        )

    _set_post_progress_desc("outputs")

    # Outputs
    for graph_output in onnx_graph.graph.output:
        ctx.ensure_tensor(graph_output.name)
        _append_model_outputs_preserving_order(
            model_ir,
            [str(graph_output.name)],
        )
    protected_boundary_outputs = [
        str(name)
        for name in _get_protected_boundary_tensor_names(model_ir)
        if str(name) in model_ir.tensors
    ]
    _append_model_outputs_preserving_order(
        model_ir,
        protected_boundary_outputs,
    )
    dynamic_boundary_signature_map: Dict[str, List[int]] = {}
    onnx_boundary_signature_map = model_ir.metadata.get(
        "onnx_boundary_shape_signature_map",
        {},
    )
    if not isinstance(onnx_boundary_signature_map, dict):
        onnx_boundary_signature_map = {}
    dynamic_boundary_name_set = set(
        list(model_ir.metadata.get("onnx_dynamic_input_tensor_names", []))
        + list(model_ir.metadata.get("onnx_dynamic_output_tensor_names", []))
    )
    for boundary_name in list(model_ir.inputs) + list(model_ir.outputs):
        if str(boundary_name) not in dynamic_boundary_name_set:
            continue
        boundary_tensor = model_ir.tensors.get(str(boundary_name), None)
        if boundary_tensor is None or boundary_tensor.shape_signature is None:
            continue
        snapshot_signature = [int(v) for v in list(boundary_tensor.shape_signature)]
        hinted_signature = onnx_boundary_signature_map.get(str(boundary_name), None)
        hinted_signature_aligned = _align_boundary_signature_to_current_shape(
            boundary_signature=(
                [int(v) for v in list(hinted_signature)]
                if isinstance(hinted_signature, list)
                else None
            ),
            current_shape=[int(v) for v in list(boundary_tensor.shape)],
        )
        if hinted_signature_aligned is not None:
            if len(hinted_signature_aligned) == len(snapshot_signature):
                for axis in range(len(snapshot_signature)):
                    if int(hinted_signature_aligned[axis]) > 0:
                        snapshot_signature[axis] = int(hinted_signature_aligned[axis])
        if any(int(v) < 0 for v in snapshot_signature):
            dynamic_boundary_signature_map[str(boundary_name)] = snapshot_signature
    model_ir.metadata["dynamic_boundary_shape_signature_map"] = (
        dynamic_boundary_signature_map
    )
    _advance_post_progress()

    if optimize_layout_transpose_chains:
        _set_post_progress_desc("layout optimize pass-set 1")
        has_qdq_ops = any(
            str(op.op_type) in {"QUANTIZE", "DEQUANTIZE"}
            for op in model_ir.operators
        )
        # NOTE:
        # Binary/fanout transpose rewrites can invalidate tensor shape/layout
        # assumptions in multi-branch int8 graphs. Keep them enabled on
        # non-QDQ graphs where the pass is effective and stable.
        enable_transpose_binary_bridge_optimizations = not has_qdq_ops
        # Duplicate transpose fanout dedup is safe on pure float graphs and
        # removes redundant NHWC/NCHW adapters in multi-branch heads.
        enable_duplicate_transpose_fanout_optimizations = not has_qdq_ops

        session.record_phase_result(
            "cleanup.layout_pass_set_1.layout_transpose",
            run_layout_transpose_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        _layout_pass_set_1_initial_attention_recovery_results = (
            _run_layout_reshape_attention_recovery_prefix()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.initial_affine_chain_fold",
            _optimize_fold_mul_add_mul_affine_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.affine_prepost",
            _optimize_transpose_mul_add_const_prepost_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.pre_unary_affine_fanout",
            _optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains(
                model_ir
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.mean_affine_prepost",
            _optimize_transpose_mean_mul_add_const_prepost_nhwc_chains(model_ir),
        )
        _layout_pass_set_1_mean_attention_results = (
            _run_mean_attention_layout_pass_cluster(include_layernorm=True)
        )
        _layout_pass_set_1_attention_gate_qdq_results = (
            _run_attention_gate_qdq_recovery_sequence()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.quantized_prelu",
            run_quantized_prelu_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.dequant_transposeconv_quantize",
            _optimize_dequant_transposeconv_quantize_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.quantized_reshape",
            run_quantized_reshape_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        _layout_pass_set_1_quantized_activation_binary_results = (
            _run_quantized_activation_binary_bridge_recovery_sequence()
        )
        if enable_transpose_binary_bridge_optimizations:
            session.record_phase_result(
                "cleanup.layout_pass_set_1.transpose_binary_bridge",
                _optimize_transpose_binary_bridges(
                    model_ir,
                    layout_state=session.layout_state,
                ),
            )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.duplicate_fanout",
            run_duplicate_fanout_cleanup(
                model_ir,
                include_transpose=enable_duplicate_transpose_fanout_optimizations,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        # Binary bridge rewrites can introduce new transpose-(q|dq)-transpose patterns.
        _layout_pass_set_1_post_binary_attention_recovery_results = (
            _run_layout_reshape_attention_recovery_prefix()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.post_binary_affine_chain_fold",
            _optimize_fold_mul_add_mul_affine_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        _layout_pass_set_1_attention_quantized_suffix_results = (
            _run_layout_attention_quantized_recovery_suffix(
                include_duplicate_transpose=enable_duplicate_transpose_fanout_optimizations,
            )
        )
        _layout_pass_set_1_safe_binary_results = (
            _run_safe_binary_bridge_recovery_sequence()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.dequant_mean_quantize",
            _optimize_transpose_dequantize_mean_quantize_bridges(model_ir),
        )
        _layout_pass_set_1_qlinear_mean_concat_results = (
            _run_qlinear_mean_concat_recovery_sequence()
        )
        _layout_pass_set_1_final_attention_recovery_results = (
            _run_layout_reshape_attention_recovery_prefix()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.instancenorm_prepost",
            _optimize_transpose_instancenorm_prepost_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_1.squeeze_reshape_identity",
            run_squeeze_reshape_identity_cleanup(
                model_ir,
                include_unary_passthrough=True,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        _layout_pass_set_1_final_attention_quantized_suffix_results = (
            _run_layout_attention_quantized_recovery_suffix(
                include_duplicate_transpose=enable_duplicate_transpose_fanout_optimizations,
            )
        )
        _layout_pass_set_1_transpose_unary_fanout_results = (
            _run_transpose_unary_fanout_layout_pass_cluster(
                include_layout_transpose=True,
                include_unary_passthrough=False,
            )
        )
        _layout_pass_set_1_final_safe_binary_results = (
            _run_safe_binary_bridge_recovery_sequence()
        )
        _advance_post_progress()
    _set_post_progress_desc("core cleanup passes")
    session.record_phase_result(
        "cleanup.core.pseudo_leakyrelu",
        _optimize_fuse_pseudo_leakyrelu_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.core.yolo_decode",
        _optimize_yolo_decode_mul_square_anchor_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.core.consecutive_mul",
        run_consecutive_mul_constants_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.core.terminal_dequant",
        _sanitize_terminal_transpose_before_dequantize(model_ir),
    )
    session.record_phase_result(
        "cleanup.core.terminal_qdq",
        run_terminal_quantize_dequantize_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.core.conv_affine",
        _optimize_fold_conv_mul_add_affine_chains(
            model_ir,
            enable_conv_add_only_fold=True,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.core.conv_activation",
        _optimize_fuse_conv_activation_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "shape_resolution.core.dynamic_reshape",
        _resolve_dynamic_reshape_shapes(model_ir),
    )
    session.record_phase_result(
        "cleanup.core.squeeze_reshape_identity",
        run_squeeze_reshape_identity_cleanup(
            model_ir,
            include_unary_passthrough=True,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.core.prune_reconcile",
        run_indexed_prune_reconcile_cleanup(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    _advance_post_progress()
    if optimize_layout_transpose_chains:
        _set_post_progress_desc("layout recovery pass-set 2")
        # Final recovery sweep:
        # some transpose-binary patterns become shape-safe only after static
        # metadata reconciliation, so run bridge passes once more.
        _layout_pass_set_2_qlinear_mean_concat_results = (
            _run_qlinear_mean_concat_recovery_sequence()
        )
        _layout_pass_set_2_layout_recovery_prefix_results = (
            _run_layout_recovery_prefix_pass_sequence()
        )
        _layout_pass_set_2_preadd_mean_attention_results = (
            _run_preadd_mean_attention_recovery_sequence()
        )
        _layout_pass_set_2_attention_gate_qdq_results = (
            _run_attention_gate_qdq_recovery_sequence()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.dequant_transposeconv_quantize",
            _optimize_dequant_transposeconv_quantize_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        _layout_pass_set_2_quantized_activation_binary_results = (
            _run_quantized_activation_binary_bridge_recovery_sequence()
        )
        # Binary bridge recovery can recreate pre/post transpose wrappers around CONCAT.
        session.record_phase_result(
            "cleanup.layout_pass_set_2.elementwise_concat_conv",
            _optimize_transpose_elementwise_concat_conv_nhwc_groups(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.spp",
            run_spp_layout_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.pre_concat",
            _optimize_transpose_pre_concat_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.ndhwc_concat",
            run_ndhwc_concat_layout_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.stridedslice_pre_concat",
            _optimize_transpose_stridedslice_pre_concat_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.split_mixed_pre_concat",
            _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.concat_input_adapter",
            _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.slice_logistic_concat_tail",
            _optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        _layout_opt_channel_shuffle_gather_results = (
            _run_channel_shuffle_gather_layout_pass_cluster(
                include_post_gather_cleanup=True,
            )
        )
        _layout_opt_preadd_mean_attention_results = (
            _run_preadd_mean_attention_recovery_sequence()
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.sa_pa_mirrorpad",
            _optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        _layout_opt_gate_layout_results = _run_gate_layout_pass_cluster(
            include_mixed_attention=False
        )
        for _ in range(2):
            normalization_graph_index = ModelIRGraphIndex(model_ir)
            rewritten_instnorm = int(
                _optimize_transpose_instancenorm_prepost_nhwc_chains(
                    model_ir,
                    graph_index=normalization_graph_index,
                    layout_state=session.layout_state,
                ).get(
                    "optimized_transpose_instancenorm_prepost_nhwc_chains", 0
                )
            )
            rewritten_instnorm_posttranspose_bias = int(
                _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
                    model_ir,
                    graph_index=normalization_graph_index,
                    layout_state=session.layout_state,
                ).get(
                    "optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains", 0
                )
            )
            normalization_pad_stats = run_normalization_pad_layout_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            )
            rewritten_instnorm_pad = int(
                normalization_pad_stats.get(
                    "optimized_transpose_instancenorm_pad_prepost_nhwc_chains", 0
                )
            )
            rewritten_flat_globalnorm_pad = int(
                normalization_pad_stats.get(
                    "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains", 0
                )
            )
            residual_graph_index = ModelIRGraphIndex(model_ir)
            rewritten_instnorm_residual = int(
                _optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
                    model_ir,
                    graph_index=residual_graph_index,
                    layout_state=session.layout_state,
                ).get(
                    "optimized_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains", 0
                )
            )
            rewritten_instnorm_residual_tail = int(
                _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
                    model_ir,
                    graph_index=residual_graph_index,
                    layout_state=session.layout_state,
                ).get(
                    "optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains", 0
                )
            )
            rewritten_instnorm_dualstats_residual = int(
                _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
                    model_ir,
                    graph_index=residual_graph_index,
                    layout_state=session.layout_state,
                ).get(
                    "optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains", 0
                )
            )
            if (
                rewritten_instnorm
                + rewritten_instnorm_posttranspose_bias
                + rewritten_instnorm_pad
                + rewritten_flat_globalnorm_pad
                + rewritten_instnorm_residual
                + rewritten_instnorm_residual_tail
                + rewritten_instnorm_dualstats_residual
                <= 0
            ):
                break
        session.record_phase_result(
            "cleanup.layout_pass_set_2.squeeze_reshape_identity",
            run_squeeze_reshape_identity_cleanup(
                model_ir,
                include_unary_passthrough=True,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            ),
        )
        session.record_phase_result(
            "cleanup.layout_pass_set_2.prune_reconcile",
            run_indexed_prune_reconcile_cleanup(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        _advance_post_progress()
    _set_post_progress_desc("terminal cleanup passes")
    # Recovery sweeps above can re-introduce terminal TRANSPOSE->DEQUANTIZE.
    # Run terminal sanitizers once more at the very end.
    session.record_phase_result(
        "cleanup.terminal.dequant",
        _sanitize_terminal_transpose_before_dequantize(model_ir),
    )
    session.record_phase_result(
        "cleanup.terminal.qdq",
        run_terminal_quantize_dequantize_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.conv_affine",
        _optimize_fold_conv_mul_add_affine_chains(
            model_ir,
            enable_conv_add_only_fold=True,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.conv_activation",
        _optimize_fuse_conv_activation_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.pre_argmax",
        _optimize_transpose_pre_argmax_nhwc_terminal_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.transpose_gather_channel_fanout",
        run_transpose_gather_channel_fanout_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.softmax_transpose",
        _optimize_terminal_softmax_transpose_after_nhwc_propagation(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.boundary_input_normalization",
        run_boundary_input_normalization_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.boundary_input_channel_slice",
        _optimize_boundary_input_transpose_channel_slice_blocks(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.internal_channel_slice",
        _optimize_internal_transpose_channel_slice_nhwc_propagation_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.channel_slice_muladd_bridge",
        _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    _terminal_slice_concat_recovery_results = (
        _run_terminal_slice_concat_layout_recovery_sequence()
    )
    session.record_phase_result(
        "cleanup.terminal.boundary_stridedslice_qdq_concat",
        _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.swish_residual_concat_closure",
        _optimize_transpose_swish_residual_concat_closure_nhwc_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.terminal.dequant_logistic_mul_quantize_bridge",
        _optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir),
    )
    session.record_phase_result(
        "cleanup.terminal.swish_qdq_island",
        _optimize_transpose_swish_qdq_nhwc_islands(model_ir),
    )
    # Late recovery passes can recreate Conv->InstNorm(NCHW)->Pad wrappers.
    session.record_phase_result(
        "cleanup.terminal.instancenorm_post_bias",
        _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.normalization_pad",
        run_normalization_pad_layout_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.instancenorm_residual_add",
        _optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.instancenorm_residual_mul_concat",
        _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.instancenorm_dualstats",
        _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    _terminal_boundary_layout_results = (
        _run_terminal_boundary_layout_pass_cluster()
    )
    if optimize_layout_transpose_chains:
        # Boundary/layout recovery can still recreate NCHW wrappers around MEAN.
        # Run dedicated NHWC passthrough once more in the terminal stage.
        _terminal_mean_attention_results = (
            _run_mean_attention_layout_pass_cluster(
                include_conv_attention=False,
            )
        )
        session.record_phase_result(
            "cleanup.terminal.batchmatmul_affine_input",
            _optimize_batchmatmul_affine_transpose_input_chains(model_ir),
        )
        session.record_phase_result(
            "cleanup.terminal.batchmatmul_reshape_se",
            _optimize_batchmatmul_reshape_se_nhwc_chains(model_ir),
        )
        session.record_phase_result(
            "cleanup.terminal.batchmatmul_adj_flags",
            _optimize_batchmatmul_transpose_input_to_adj_flags(model_ir),
        )
        _terminal_qkv_attention_results = (
            _run_qkv_attention_layout_pass_cluster()
        )
        session.record_phase_result(
            "cleanup.terminal.qkv_split_conv_concat_bridge",
            _optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
                model_ir,
                layout_state=session.layout_state,
            ),
        )
        # Run the multi-branch gate rewrite at terminal stage so earlier
        # generic passes do not re-wrap rewritten NHWC tensors.
        _terminal_singleton_reshape_results = (
            _run_singleton_reshape_layout_pass_cluster(
                include_layout_transpose=True,
                include_multi_branch_gate=True,
            )
        )
    _terminal_clamp_unary_relu_results = (
        _run_terminal_clamp_unary_relu_pass_cluster()
    )
    _terminal_sinet_layout_recovery_results = (
        _run_sinet_terminal_layout_recovery_sequence()
    )
    session.record_phase_result(
        "cleanup.terminal.sinet_hardswish_se",
        _optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            model_ir
        ),
    )
    session.record_phase_result(
        "cleanup.terminal.dequant_hardsigmoid_bridge",
        _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir),
    )
    # Terminal MUL/ADD/PRELU rewriting can recreate NCHW bridge wrappers.
    _terminal_sinet_preadd_resize_results = (
        _run_sinet_preadd_resize_recovery_sequence()
    )
    # Apply singleton transpose->reshape rewrite regardless of layout-opt mode.
    # This is required for fallback relowering (optimize_layout_transpose_chains=False)
    # where channelwise [1,C,1,1] -> [1,1,1,C] adapters can remain as TRANSPOSE.
    _post_terminal_singleton_reshape_results = (
        _run_singleton_reshape_layout_pass_cluster(
            include_duplicate_fanout=True,
            include_spatial_concat_post_transpose=False,
        )
    )
    session.record_phase_result(
        "shape_topology.terminal.indexed_convergence",
        _run_indexed_shape_convergence_cleanup(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    # Very late shape reconciliation can expose strict shuffle-residual patterns.
    # Re-run terminal transpose reducers once at absolute end.
    _very_late_sinet_layout_recovery_results = (
        _run_sinet_terminal_layout_recovery_sequence()
    )
    _very_late_sinet_preadd_resize_results = (
        _run_sinet_preadd_resize_recovery_sequence()
    )
    # Final cleanup for residual transpose bridges introduced in late SiNet blocks.
    session.record_phase_result(
        "cleanup.very_late.residual_affine_prelu",
        _optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.very_late.residual_affine_fanout",
        _optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.very_late.prune_reconcile",
        run_indexed_prune_reconcile_cleanup(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    # Dead-op pruning can unblock strict locality guards in this recovery.
    # Its pre-Add/PRELU rewrites can recreate SiNet dual-resize adapters.
    _post_cleanup_sinet_preadd_resize_results = (
        _run_sinet_preadd_resize_recovery_sequence()
    )
    # Late SiNet rewrites can recreate SA/PA MIRROR_PAD NCHW<->NHWC bridges.
    session.record_phase_result(
        "cleanup.post_cleanup.csp_attention",
        _optimize_transpose_csp_attention_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.post_cleanup.sa_pa_mirrorpad",
        _optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.post_sinet.batchmatmul_affine_input",
        _optimize_batchmatmul_affine_transpose_input_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.post_sinet.batchmatmul_reshape_se",
        _optimize_batchmatmul_reshape_se_nhwc_chains(model_ir),
    )
    session.record_phase_result(
        "cleanup.post_sinet.batchmatmul_adj_flags",
        _optimize_batchmatmul_transpose_input_to_adj_flags(model_ir),
    )
    _post_sinet_qkv_attention_results = (
        _run_qkv_attention_layout_pass_cluster()
    )
    session.record_phase_result(
        "cleanup.post_sinet.relu_split_all_outputs",
        _optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.post_sinet.relu_split_conv_concat",
        _optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.post_sinet.split_conv_concat_bridge",
        _optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.post_sinet.mix_attention",
        _optimize_sinet_mix_attention_double_logistic_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
    )
    session.record_phase_result(
        "cleanup.post_sinet.mixed_attention_layout",
        run_mixed_attention_layout_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        ),
    )
    session.record_phase_result(
        "cleanup.post_sinet.dequant_hardsigmoid_bridge",
        _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir),
    )
    session.record_phase_result(
        "cleanup.late.ndhwc_cost_volume",
        run_late_ndhwc_cost_volume_layout_cleanup(
            shared_model_ir_pass_context,
        ),
    )
    _late_cost_volume_conv_affine_stats = (
        _optimize_fold_conv_mul_add_affine_chains(
            model_ir,
            enable_conv_add_only_fold=True,
            layout_state=session.layout_state,
        )
    )
    _late_concat_layout_results = run_late_concat_layout_cleanup(
        shared_model_ir_pass_context,
    )
    if optimize_layout_transpose_chains:
        _late_concat_elementwise_fanout_stats = (
            _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
                model_ir
            )
        )
    _late_reshape_layout_results = run_late_reshape_layout_cleanup(
        shared_model_ir_pass_context,
    )
    _late_channel_shuffle_gather_results = (
        _run_channel_shuffle_gather_layout_pass_cluster(
            include_two_way_shuffle=False,
            include_nhwc_shuffle=False,
        )
    )
    _late_attention_layout_results = run_late_attention_layout_cleanup(
        shared_model_ir_pass_context,
    )
    _late_window_layout_results = run_late_window_layout_cleanup(
        shared_model_ir_pass_context,
    )
    # Late transpose/layout rewrites can invalidate previously resolved
    # RESHAPE constants. Re-resolve once at absolute end.
    _late_final_shape_activation_convergence_stats = (
        _run_indexed_final_shape_activation_convergence(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Final boundary cleanup after all late transpose/layout rewrites.
    _final_boundary_channel_layout_results = (
        run_final_boundary_channel_layout_cleanup(
            shared_model_ir_pass_context,
        )
    )
    _final_slice_concat_recovery_results = (
        _run_terminal_slice_concat_layout_recovery_sequence()
    )
    _final_slice_prepost_passthrough_stats = (
        _optimize_transpose_slice_prepost_nhwc_passthrough_chains(model_ir)
    )
    # Keep pre-concat NHWC relayout at terminal stage as late strict rewrites
    # can recreate CONCAT(axis=1)+post-transpose wrappers.
    _final_pre_concat_stats = (
        _optimize_transpose_pre_concat_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
    )
    _terminal_relu_split_all_outputs_stats = (
        _optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _terminal_relu_split_conv_concat_stats = (
        _optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _terminal_split_mixed_pre_concat_stats = (
        _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _terminal_concat_input_adapter_stats = (
        _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _terminal_concat_unary_conv_stats = (
        run_concat_unary_conv_layout_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
    )
    _terminal_shape_extract_stats = (
        _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)
    )
    if optimize_layout_transpose_chains:
        _terminal_elementwise_fanout_stats = (
            _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
                model_ir
            )
        )
    _terminal_singleton_maxpool_reshape_results = (
        _run_terminal_singleton_maxpool_reshape_pass_pair()
    )
    if optimize_layout_transpose_chains:
        _terminal_convpool_output_passthrough_stats = (
            _optimize_convpool_output_transpose_nhwc_passthrough_chains(
                model_ir
            )
        )
    elif apply_safe_transpose_reduction_lite_on_no_layout_opt:
        session.record_phase_result(
            "layout.no_layout.safe_transpose_reduction",
            _apply_safe_transpose_reduction_lite(model_ir),
        )
        # Keep strict, const-only NHWC<->NCHW affine bridge folding enabled
        # in no-layout fallback so simple TRANSPOSE->MUL->ADD->TRANSPOSE
        # chains are still reduced.
        _no_layout_fallback_affine_prepost_stats = (
            _optimize_transpose_mul_add_const_prepost_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            )
        )
    _late_dequant_hardsigmoid_bridge_stats = (
        _optimize_transpose_dequant_hardsigmoid_quantize_bridges(model_ir)
    )
    _late_dequant_unary_fanout_results = (
        _run_late_dequant_unary_fanout_pass_cluster()
    )
    # No-layout fallback relowering can still keep strict
    # TRANSPOSE->LOGISTIC->MUL->TRANSPOSE swish wrappers (e.g. MobileViT stem).
    _late_swish_transpose_passthrough_stats = (
        _optimize_swish_transpose_passthrough_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Conv1D shim patterns can retain:
    # TRANSPOSE -> SQUEEZE -> UNARY -> EXPAND_DIMS -> TRANSPOSE.
    # Fold them to a single rank-4 UNARY op.
    _late_conv1d_squeeze_unary_stats = (
        _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Variant with intermediate rank-4 transpose:
    # TRANSPOSE -> UNARY -> TRANSPOSE -> RESHAPE -> EXPAND_DIMS -> TRANSPOSE.
    _late_conv1d_rank4_unary_stats = (
        _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Fanout variant:
    # Keep NCHW side branch and bypass only the NHWC wrapper branch.
    _late_conv1d_unary_fanout_stats = (
        _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # InstanceNorm(flat) bridge variant:
    # T -> SQUEEZE -> RESHAPE -> IN -> RESHAPE -> UNARY -> EXPAND -> T
    # can be rewritten in NHWC by swapping C/W at reshape boundary.
    _late_conv1d_instancenorm_unary_stats = (
        _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # tencoder residual-gated branch variant:
    # dual (post-conv T->SQUEEZE) branches merge via ADD then EXPAND->T before next CONV.
    _late_conv1d_tencoder_stats = (
        _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Conv1D tail patterns may still contain:
    # TRANSPOSE -> SQUEEZE -> UNARY* -> BATCH_MATMUL.
    # Rewire to NHWC + adjX matmul when mathematically equivalent.
    _late_conv1d_batchmatmul_stats = (
        _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Decoder linear->deconv tails can keep:
    # BATCH_MATMUL -> ADD -> EXPAND_DIMS -> TRANSPOSE -> TRANSPOSE_CONV.
    # Transpose BATCH_MATMUL/ADD layout so TRANSPOSE before deconv is removed.
    _late_decoder_deconv_stats = (
        _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Decoder terminal tails can keep:
    # TRANSPOSE -> SQUEEZE -> MEAN -> SQUEEZE.
    # Remap squeeze/mean axes in NHWC and drop the transpose.
    _late_terminal_squeeze_mean_stats = (
        _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Very late transpose/layout rewrites can recreate PAD-adjacent NCHW wrappers.
    _very_late_pad_layout_stats = run_pad_layout_cleanup(
        model_ir,
        layout_state=session.layout_state,
        diagnostics=session.diagnostics,
    )
    _very_late_instancenorm_post_bias_stats = (
        _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _very_late_instancenorm_residual_mul_concat_stats = (
        _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _very_late_instancenorm_dualstats_stats = (
        _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Norm-subgraph fallback rewrites can introduce channelwise
    # [1,C,1,1]->[1,1,1,C] adapters as TRANSPOSE in no-layout mode.
    # Re-canonicalize them to RESHAPE at the very end.
    _very_late_singleton_consecutive_reshape_results = (
        _run_singleton_consecutive_reshape_pass_cluster(
            model_ir,
            session.layout_state,
        )
    )
    if optimize_layout_transpose_chains:
        _very_late_layout_transpose_cleanup_stats = (
            run_layout_transpose_cleanup(
                model_ir,
                layout_state=session.layout_state,
                diagnostics=session.diagnostics,
            )
        )
    _very_late_broadcast_repair_stats = (
        _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
            model_ir
        )
    )
    session.record_phase_result(
        "shape_reconciliation.primary.very_late_broadcast",
        _reconcile_static_tensor_shapes(
            model_ir,
            include_mutation_count=True,
        ),
    )
    shared_late_tensor_count = len(model_ir.tensors)
    shared_boundary_signature_stats = (
        _realign_dynamic_boundary_shape_signature_map(model_ir)
    )
    # Keep final serialized metadata consistent for tools that render
    # shape_signature (e.g. Netron): HARD_SWISH is shape-preserving.
    shared_hardswish_stats = _sanitize_hardswish_tensor_shapes(model_ir)
    # Final guardrail for runtime validity: ensure SQUEEZE axes target
    # singleton dimensions after all late layout/shape rewrites.
    shared_squeeze_stats = _sanitize_squeeze_axes_with_static_input_shapes(
        model_ir
    )
    shared_conv_transpose_stats = (
        _sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv(model_ir)
    )
    shared_binary_adapter_stats, shared_singleton_adapter_stats = (
        run_indexed_binary_layout_adapter_cleanup(model_ir)
    )
    (
        shared_singleton_channel_stats,
        shared_duplicate_fanout_stats,
        shared_consecutive_reshape_stats,
    ) = _run_singleton_consecutive_reshape_pass_cluster(
        model_ir,
        session.layout_state,
    )
    if _stats_have_positive_count(
        shared_boundary_signature_stats,
        shared_hardswish_stats,
        shared_squeeze_stats,
        shared_conv_transpose_stats,
        shared_binary_adapter_stats,
        shared_singleton_adapter_stats,
        shared_singleton_channel_stats,
        shared_duplicate_fanout_stats,
        shared_consecutive_reshape_stats,
    ) or len(model_ir.tensors) < shared_late_tensor_count:
        session.record_phase_result(
            "shape_reconciliation.primary.shared_late",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    late_binary_repair_tensor_count = len(model_ir.tensors)
    late_signature_stats = _sanitize_static_shape_signature_consistency(
        model_ir
    )
    late_binary_adapter_stats, late_singleton_adapter_stats = (
        run_indexed_binary_layout_adapter_cleanup(model_ir)
    )
    if (
        int(
            late_signature_stats.get(
                "sanitized_static_shape_signature_consistency",
                0,
            )
        )
        + int(
            late_binary_adapter_stats.get(
                "inserted_rank4_binary_layout_fix_transpose",
                0,
            )
        )
        + int(
            late_singleton_adapter_stats.get(
                "repaired_rank4_binary_singleton_broadcast_layout_mismatch",
                0,
            )
        )
        > 0
        or len(model_ir.tensors) < late_binary_repair_tensor_count
    ):
        session.record_phase_result(
            "shape_reconciliation.primary.late_binary_repair",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    if optimize_layout_transpose_chains or apply_safe_transpose_reduction_lite_on_no_layout_opt:
        late_binary_layout_recovery_stats = run_late_binary_layout_recovery(
            model_ir,
            include_layout_transpose=optimize_layout_transpose_chains,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
        if _stats_have_positive_count(late_binary_layout_recovery_stats):
            session.record_phase_result(
                "shape_reconciliation.primary.late_binary_layout_recovery",
                _reconcile_static_tensor_shapes(
                    model_ir,
                    include_mutation_count=True,
                ),
            )
    # Keep this at absolute end of optimization pipeline: several late
    # shape/layout repair passes can recreate the exact tail pattern.
    _pre_terminal_affine_instancenorm_post_bias_stats = (
        _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _pre_terminal_affine_instancenorm_residual_mul_concat_stats = (
        _optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _pre_terminal_affine_instancenorm_dualstats_stats = (
        _optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    # Late bridge rewrites above can recreate strict
    # TRANSPOSE->MUL(const)->ADD(const)->TRANSPOSE fragments.
    pre_terminal_affine_tensor_count = len(model_ir.tensors)
    pre_terminal_affine_results = (
        _run_terminal_affine_concat_split_recovery_sequence()
    )
    _pre_terminal_affine_stats = summarize_terminal_affine_concat_split_mutations(
        pre_terminal_affine_results,
        pruned_unused_tensors=max(
            0,
            int(pre_terminal_affine_tensor_count - len(model_ir.tensors)),
        ),
    )
    pre_terminal_pre_add_tensor_count = len(model_ir.tensors)
    _pre_terminal_pre_add_stats = {
        **_optimize_transpose_pre_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        ),
        "pruned_unused_tensors": max(
            0,
            int(pre_terminal_pre_add_tensor_count - len(model_ir.tensors)),
        ),
    }
    channel_slice_pad_mul_results = _run_channel_slice_pad_mul_layout_pass_cluster()
    _pre_terminal_channel_slice_pad_mul_stats = (
        summarize_channel_slice_pad_mul_mutations(channel_slice_pad_mul_results)
    )
    _pre_terminal_affine_post_add_stats = (
        _optimize_transpose_mul_posttranspose_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _pre_terminal_affine_slice_pad_concat_stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )
    # Strict slice/merge cleanup above can recreate simple affine bridge tails:
    # TRANSPOSE->MUL(const)->TRANSPOSE->ADD(const).
    # Keep this after pre_add/slice/pad strict rewrites: those passes can
    # recreate CONCAT->MUL->TRANSPOSE->ADD NHWC bridge tails.
    terminal_affine_tensor_count = len(model_ir.tensors)
    terminal_affine_results = _run_terminal_affine_concat_split_recovery_sequence()
    _terminal_affine_stats = summarize_terminal_affine_concat_split_mutations(
        terminal_affine_results,
        pruned_unused_tensors=max(
            0,
            int(terminal_affine_tensor_count - len(model_ir.tensors)),
        ),
    )
    _terminal_slice_pad_concat_stats = (
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            model_ir
        )
    )
    late_spp_results = _run_late_spp_concat_unary_conv_pass_pair()
    _late_spp_stats = summarize_late_spp_concat_unary_conv_mutations(
        late_spp_results
    )
    _late_pre_qkv_shape_extract_stats = (
        _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)
    )
    # Keep QKV bridge reductions at the terminal stage: some late strict
    # transpose/add/slice rewrites above can recreate this exact motif.
    late_qkv_tensor_count = len(model_ir.tensors)
    late_qkv_results = _run_qkv_attention_layout_pass_cluster(
        include_layout_transpose=optimize_layout_transpose_chains,
        include_prefix=False,
    )
    _late_qkv_stats = summarize_qkv_attention_mutations(
        late_qkv_results,
        include_layout_transpose=optimize_layout_transpose_chains,
        include_prefix=False,
        pruned_unused_tensors=max(
            0,
            int(late_qkv_tensor_count - len(model_ir.tensors)),
        ),
    )
    _terminal_split_conv_concat_bridge_stats = (
        _optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    terminal_hardswish_se_tensor_count = len(model_ir.tensors)
    _terminal_hardswish_se_stats = {
        **_optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            model_ir
        ),
        "pruned_unused_tensors": max(
            0,
            int(terminal_hardswish_se_tensor_count - len(model_ir.tensors)),
        ),
    }
    # Late affine/fusion cleanups can recreate
    # TRANSPOSE->(ADD/MUL hard-sigmoid-like)->MUL->TRANSPOSE wrappers.
    # Run strict hard-sigmoid transpose passthrough once more at terminal stage.
    late_hard_activation_tensor_count = len(model_ir.tensors)
    late_hard_activation_results = _run_late_hard_activation_layout_pass_pair(
        include_layout_transpose=optimize_layout_transpose_chains,
    )
    _late_hard_activation_stats = summarize_late_hard_activation_layout_mutations(
        late_hard_activation_results,
        include_layout_transpose=optimize_layout_transpose_chains,
        pruned_unused_tensors=max(
            0,
            int(late_hard_activation_tensor_count - len(model_ir.tensors)),
        ),
    )
    # Absolute-end cleanup: late bridge rewrites can recreate strict
    # pre/post CONCAT transpose wrappers and SHAPE-extract transposes.
    _absolute_final_pre_concat_stats = (
        _optimize_transpose_pre_concat_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
    )
    _late_pre_layout_cluster_shape_extract_stats = (
        _optimize_transpose_shape_extract_nhwc_to_nchw_chains(model_ir)
    )
    late_layout_cluster_tensor_count = len(model_ir.tensors)
    late_layout_cluster_results = (
        _run_late_layout_mean_spp_gather_constant_cast_pass_cluster(
            include_layout_transpose=optimize_layout_transpose_chains,
        )
    )
    _late_layout_cluster_stats = (
        summarize_late_layout_mean_spp_gather_constant_cast_mutations(
            late_layout_cluster_results,
            include_layout_transpose=optimize_layout_transpose_chains,
            pruned_unused_tensors=max(
                0,
                int(late_layout_cluster_tensor_count - len(model_ir.tensors)),
            ),
        )
    )
    _terminal_expand_squeeze_stats = _replace_expand_dims_and_squeeze_with_reshape(
        model_ir,
        layout_state=session.layout_state,
    )
    session.record_phase_result(
        "shape_reconciliation.terminal.expand_squeeze",
        _reconcile_static_tensor_shapes(
            model_ir,
            include_mutation_count=True,
        ),
    )
    _advance_post_progress()

    _late_orphan_recurrent_repair_stats = (
        _repair_orphan_recurrent_step_tensors(model_ir)
    )
    _late_unbound_input_repair_stats = (
        _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(model_ir)
    )
    # The late unbound-input repair can inject strict
    # NHWC->NCHW->NHWC MUL/ADD wrappers (repair_perm tensors).
    # Fold them again before final shape/topology reconciliation.
    _very_late_affine_post_add_stats = (
        _optimize_transpose_mul_posttranspose_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    very_late_normalization_tensor_count = len(model_ir.tensors)
    very_late_normalization_results = (
        _run_very_late_gather_constant_normalization_pass_cluster()
    )
    _very_late_normalization_stats = (
        summarize_very_late_gather_constant_normalization_mutations(
            very_late_normalization_results,
            pruned_unused_tensors=max(
                0,
                int(very_late_normalization_tensor_count - len(model_ir.tensors)),
            ),
        )
    )
    # Very late terminal bridge/transpose rewrites above can still stale out
    # RESHAPE constant inputs. Re-resolve once immediately before final sort.
    _very_late_dynamic_reshape_stats = _resolve_dynamic_reshape_shapes(
        model_ir,
        prefer_runtime_inferable_from_onnx_raw=True,
    )
    very_late_conv_input_tensor_count = len(model_ir.tensors)
    _very_late_conv_input_stats = {
        **_run_indexed_conv_input_adapter_repairs(model_ir),
        "pruned_unused_tensors": max(
            0,
            int(very_late_conv_input_tensor_count - len(model_ir.tensors)),
        ),
    }
    _very_late_stale_channel_shuffle_stats = (
        run_stale_nchw_channel_shuffle_repair(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
    )
    _very_late_concat_transpose_conv_axis_stats = (
        _repair_nchw_concat_transpose_conv_axes(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _very_late_concat_global_pool_conv_axis_stats = (
        _repair_nchw_concat_global_pool_conv_axes(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _very_late_dynamic_rank1_reshape_stats = (
        _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    session.record_phase_result(
        "shape_reconciliation.primary.very_late_final",
        _reconcile_static_tensor_shapes(
            model_ir,
            include_mutation_count=True,
        ),
    )
    split_fallback_stats = _replace_unsupported_split_with_slice(
        model_ir,
        layout_state=session.layout_state,
    )
    if int(split_fallback_stats.get("replaced_unsupported_split_with_slice", 0)) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.post_split_fallback",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )

    # Safety fallback:
    # Some aggressive transpose/layout rewrites can leave dangling dynamic inputs
    # (tensor is consumed but has no producer and no embedded constant data),
    # which later appears as unexpected runtime-fed input in TFLite.
    # When detected, rebuild with transpose-chain optimization disabled to
    # preserve graph connectivity.
    unbound_inputs = _find_unbound_nonconstant_operator_inputs(model_ir)
    if optimize_layout_transpose_chains and len(unbound_inputs) > 0:
        _set_post_progress_desc("fallback relowering")
        _advance_post_progress()
        if post_progress_bar is not None:
            post_progress_bar.close()
        fallback_ir = lower_onnx_to_ir(
            onnx_graph=onnx_graph,
            output_file_name=output_file_name,
            allow_custom_ops=allow_custom_ops,
            custom_op_allowlist=custom_op_allowlist,
            optimize_layout_transpose_chains=False,
            transpose_inputs_to_nhwc=transpose_inputs_to_nhwc,
            keep_ncw_or_nchw_or_ncdhw_input_names=keep_ncw_or_nchw_or_ncdhw_input_names,
            keep_nwc_or_nhwc_or_ndhwc_input_names=keep_nwc_or_nhwc_or_ndhwc_input_names,
            keep_shape_absolutely_input_names=keep_shape_absolutely_input_names,
            disable_group_convolution=disable_group_convolution,
            output_nms_with_argmax=output_nms_with_argmax,
            switch_nms_version=switch_nms_version,
            show_progress=show_progress,
            apply_safe_transpose_reduction_lite_on_no_layout_opt=False,
            number_of_dimensions_after_flextranspose_compression=number_of_dimensions_after_flextranspose_compression,
            number_of_dimensions_after_flexstridedslice_compression=number_of_dimensions_after_flexstridedslice_compression,
            protected_boundary_tensor_names=protected_boundary_tensor_names,
        )
        fallback_norm_tensor_count = len(fallback_ir.tensors)
        fallback_norm_stats = {
            **run_pad_layout_cleanup(
                fallback_ir,
                include_pad=False,
                include_unary=False,
                include_norm=True,
                diagnostics=session.diagnostics,
            ),
            "pruned_unused_tensors": max(
                0,
                fallback_norm_tensor_count - len(fallback_ir.tensors),
            ),
        }
        if int(fallback_norm_stats.get("optimized_transpose_norm_subgraph_pad_prepost_nhwc_chains", 0)) > 0:
            (
                _fallback_binary_adapter_stats,
                _fallback_singleton_adapter_stats,
            ) = run_indexed_binary_layout_adapter_cleanup(fallback_ir)
            _fallback_singleton_consecutive_reshape_results = (
                _run_singleton_consecutive_reshape_pass_cluster(
                    fallback_ir,
                    None,
                )
            )
            session.record_phase_result(
                "shape_topology.fallback.norm",
                run_static_shape_topology_reconciliation(fallback_ir),
            )
        _fallback_dynamic_rank1_stats = (
            _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(fallback_ir)
        )
        session.record_phase_result(
            "topology_layout.fallback.post_dynamic_rank1",
            run_topology_layout_refresh(fallback_ir),
        )
        fallback_broadcast_repair_stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
            fallback_ir
        )
        if int(fallback_broadcast_repair_stats.get("repaired_rank4_channelwise_broadcast_constants", 0)) > 0:
            session.record_phase_result(
                "shape_reconciliation.fallback.broadcast",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
            session.record_phase_result(
                "topology_layout.fallback.broadcast",
                run_topology_layout_refresh(fallback_ir),
            )
        fallback_se_fc_gather_tensor_count = len(fallback_ir.tensors)
        fallback_sinet_shuffle_stats = (
            _optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
                fallback_ir,
                layout_state=None,
            )
        )
        fallback_se_fc_stats, fallback_gather_stats = (
            _run_se_fc_gather_channel_fanout_pass_cluster(
                fallback_ir,
                None,
            )
        )
        if (
            int(
                fallback_sinet_shuffle_stats.get(
                    "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains",
                    0,
                )
            )
            + int(
                fallback_se_fc_stats.get(
                    "optimized_transpose_se_fc_mul_prepost_nhwc_chains",
                    0,
                )
            )
            + int(
                fallback_gather_stats.get(
                    "optimized_transpose_gather_transpose_nhwc_channel_chains",
                    0,
                )
            )
            > 0
            or len(fallback_ir.tensors) < fallback_se_fc_gather_tensor_count
        ):
            session.record_phase_result(
                "shape_reconciliation.fallback.se_fc_gather",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
        fallback_placeholder_matmul_stats = (
            _restore_placeholder_matmul_flattened_inputs(fallback_ir)
        )
        if int(
            fallback_placeholder_matmul_stats.get(
                "restored_placeholder_matmul_flattened_inputs",
                0,
            )
        ) > 0:
            session.record_phase_result(
                "shape_reconciliation.fallback.placeholder_matmul",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
        session.record_phase_result(
            "topology.fallback.post_placeholder",
            _topologically_sort_operators(fallback_ir),
        )
        _fallback_precision_div_rewrite_stats = (
            _rewrite_constant_divisors_to_multiplicative_reciprocals(fallback_ir)
        )
        _fallback_precision_consecutive_mul_stats = (
            run_consecutive_mul_constants_cleanup(
                fallback_ir,
                diagnostics=session.diagnostics,
            )
        )
        _fallback_precision_div_restore_stats = (
            _restore_precision_sensitive_reciprocal_divisions(fallback_ir)
        )
        _fallback_unbound_repair_stats = (
            _repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
                fallback_ir
            )
        )
        fallback_conv_input_tensor_count = len(fallback_ir.tensors)
        fallback_conv_input_stats = {
            **_run_indexed_conv_input_adapter_repairs(fallback_ir),
            "pruned_unused_tensors": max(
                0,
                fallback_conv_input_tensor_count - len(fallback_ir.tensors),
            ),
        }
        if int(
            fallback_conv_input_stats.get(
                "repaired_stale_nchw_to_nhwc_conv_input_transposes",
                0,
            )
        ) > 0:
            session.record_phase_result(
                "shape_reconciliation.fallback.conv_input",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
        fallback_concat_layout_stats = (
            _repair_mixed_nhwc_inputs_for_nchw_concat(fallback_ir)
        )
        if int(
            fallback_concat_layout_stats.get(
                "repaired_mixed_nhwc_inputs_for_nchw_concat",
                0,
            )
        ) > 0:
            session.record_phase_result(
                "shape_reconciliation.fallback.mixed_concat",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
        fallback_concat_axis_stats = _repair_nchw_concat_transpose_conv_axes(
            fallback_ir
        )
        if int(
            fallback_concat_axis_stats.get(
                "repaired_nchw_concat_transpose_conv_axes",
                0,
            )
        ) > 0:
            session.record_phase_result(
                "shape_reconciliation.fallback.concat_axis",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
        fallback_binary_layout_tensor_count = len(fallback_ir.tensors)
        fallback_binary_layout_stats = {
            **_repair_stale_nchw_to_nhwc_channelwise_binary_transposes(
                fallback_ir
            ),
            "pruned_unused_tensors": max(
                0,
                fallback_binary_layout_tensor_count - len(fallback_ir.tensors),
            ),
        }
        if int(
            fallback_binary_layout_stats.get(
                "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes",
                0,
            )
        ) > 0:
            session.record_phase_result(
                "shape_reconciliation.fallback.binary_layout",
                _reconcile_static_tensor_shapes(
                    fallback_ir,
                    include_mutation_count=True,
                ),
            )
        session.record_phase_result(
            "topology.fallback.post_layout_repair",
            _topologically_sort_operators(fallback_ir),
        )
        fallback_ir.metadata["layout_optimize_fallback"] = {
            "reason": "dangling_dynamic_inputs_detected",
            "count": int(len(unbound_inputs)),
            "samples": [dict(v) for v in list(unbound_inputs[:8])],
        }
        fallback_high_rank_bmm_stats = _compress_static_high_rank_batch_matmul(
            fallback_ir
        )
        if int(
            fallback_high_rank_bmm_stats.get(
                "compressed_static_high_rank_batch_matmul",
                0,
            )
        ) > 0:
            session.record_phase_result(
                "shape_topology.fallback.high_rank_batch_matmul",
                run_static_shape_topology_reconciliation(fallback_ir),
            )
        _fallback_binary_layout_convergence_stats = (
            _run_indexed_binary_layout_convergence(fallback_ir)
        )
        session.record_phase_result(
            "layout_validation.fallback.terminal",
            run_topology_layout_validation(fallback_ir),
        )
        return _finalize_model_ir(fallback_ir)

    _final_precision_div_rewrite_stats = (
        _rewrite_constant_divisors_to_multiplicative_reciprocals(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _final_precision_consecutive_mul_stats = (
        run_consecutive_mul_constants_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
    )
    _final_precision_div_restore_stats = (
        _restore_precision_sensitive_reciprocal_divisions(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _set_post_progress_desc("topological sort")
    session.record_phase_result(
        "topology.primary.post_lowering",
        _topologically_sort_operators(model_ir),
    )
    if apply_safe_transpose_reduction_lite_on_no_layout_opt:
        # In no-layout fallback path, some strict MUL/ADD affine bridges become
        # reducible only after final topological normalization.
        _no_layout_final_se_fc_stats = run_se_fc_layout_cleanup(
            model_ir,
            layout_state=session.layout_state,
            diagnostics=session.diagnostics,
        )
        _no_layout_final_affine_prepost_stats = (
            _optimize_transpose_mul_add_const_prepost_nhwc_chains(
                model_ir,
                layout_state=session.layout_state,
            )
        )
        session.record_phase_result(
            "topology.primary.no_layout_post_reduction",
            _topologically_sort_operators(model_ir),
        )
    # Final boundary-signature restore:
    # late static-shape reconciliations may overwrite graph-boundary dynamic
    # contracts (e.g. NMS selected_indices leading axis).
    _absolute_final_boundary_signature_stats = (
        _realign_dynamic_boundary_shape_signature_map(model_ir)
    )
    _absolute_final_static_signature_stats = (
        _sanitize_static_shape_signature_consistency(model_ir)
    )
    # Absolute-final guard: topological sort + signature sanitize can expose
    # one more strict TRANSPOSE->MUL(const)->TRANSPOSE->ADD(const) fragment.
    _absolute_final_affine_post_add_stats = (
        _optimize_transpose_mul_posttranspose_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _absolute_final_instancenorm_post_bias_stats = (
        _optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _absolute_final_normalization_attention_results = (
        _run_absolute_final_normalization_attention_pass_pair()
    )
    _absolute_final_dynamic_rank1_stats = (
        _rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    session.record_phase_result(
        "topology_layout.primary.absolute_final",
        run_topology_layout_refresh(model_ir),
    )
    final_convinteger_layout_stats = repair_channel_last_convinteger_input_transposes(
        model_ir,
        layout_state=session.layout_state,
    )
    if int(
        final_convinteger_layout_stats.get(
            "repaired_channel_last_convinteger_input_transposes",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_convinteger",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
        session.record_phase_result(
            "topology_layout.primary.final_convinteger",
            run_topology_layout_refresh(model_ir),
        )
    final_instancenorm_repair_stats = _repair_decomposed_instance_normalization_layouts(
        model_ir,
        layout_state=session.layout_state,
    )
    if int(final_instancenorm_repair_stats.get("repaired_decomposed_instance_normalization_layouts", 0)) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_instancenorm",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
        session.record_phase_result(
            "topology_layout.primary.final_instancenorm",
            run_topology_layout_refresh(model_ir),
        )
    final_broadcast_repair_stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(model_ir)
    if int(final_broadcast_repair_stats.get("repaired_rank4_channelwise_broadcast_constants", 0)) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_broadcast",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
        session.record_phase_result(
            "topology_layout.primary.final_broadcast",
            run_topology_layout_refresh(model_ir),
        )
    final_mixed_singleton_concat_stats = (
        _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_mixed_singleton_concat_stats.get(
            "repaired_mixed_singleton_nchw_inputs_for_nhwc_concat",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_mixed_singleton_concat",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    final_placeholder_matmul_stats = (
        _restore_placeholder_matmul_flattened_inputs(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    _final_placeholder_matmul_static_shape_stats = {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }
    if int(
        final_placeholder_matmul_stats.get(
            "restored_placeholder_matmul_flattened_inputs",
            0,
        )
    ) > 0:
        _final_placeholder_matmul_static_shape_stats = (
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            )
        )
        final_placeholder_reconcile_stats = {
            "reconciled_static_tensor_shapes": int(
                _final_placeholder_matmul_static_shape_stats.get(
                    "reconciled_static_tensor_shapes",
                    0,
                )
            )
        }
        final_placeholder_binary_tensor_count = len(model_ir.tensors)
        (
            final_placeholder_exact_binary_stats,
            final_placeholder_singleton_binary_stats,
        ) = run_indexed_binary_layout_adapter_cleanup(model_ir)
        if _stats_have_positive_count(
            final_placeholder_reconcile_stats,
            final_placeholder_exact_binary_stats,
            final_placeholder_singleton_binary_stats,
        ) or len(model_ir.tensors) < final_placeholder_binary_tensor_count:
            session.record_phase_result(
                "shape_reconciliation.primary.final_placeholder_binary",
                _reconcile_static_tensor_shapes(
                    model_ir,
                    include_mutation_count=True,
                ),
            )
        session.record_phase_result(
            "topology.primary.final_placeholder",
            _topologically_sort_operators(model_ir),
        )
    # Absolute-final SiNet/SE cleanup:
    # late broadcast/layout repairs can recreate SE gate and channel-shuffle
    # NHWC<->NCHW wrappers after the earlier dedicated passes have run.
    final_se_fc_gather_tensor_count = len(model_ir.tensors)
    final_sinet_shuffle_stats = (
        _optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    final_se_fc_stats, final_gather_stats = (
        _run_se_fc_gather_channel_fanout_pass_cluster(
            model_ir,
            session.layout_state,
        )
    )
    if (
        int(
            final_sinet_shuffle_stats.get(
                "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains",
                0,
            )
        )
        + int(
            final_se_fc_stats.get(
                "optimized_transpose_se_fc_mul_prepost_nhwc_chains",
                0,
            )
        )
        + int(
            final_gather_stats.get(
                "optimized_transpose_gather_transpose_nhwc_channel_chains",
                0,
            )
        )
        > 0
        or len(model_ir.tensors) < final_se_fc_gather_tensor_count
    ):
        session.record_phase_result(
            "shape_reconciliation.primary.final_se_fc_gather",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    # Absolute-final PRELU cleanup:
    # late layout/broadcast/singleton repairs can still recreate strict
    # TRANSPOSE->PRELU->inverse-TRANSPOSE wrappers (e.g. SiNet entry blocks).
    final_prelu_tensor_count = len(model_ir.tensors)
    final_prelu_stats = _optimize_prelu_transpose_passthrough_chains(
        model_ir,
        layout_state=session.layout_state,
    )
    if (
        int(
            final_prelu_stats.get(
                "rewritten_prelu_transpose_passthrough_chains",
                0,
            )
        )
        > 0
        or len(model_ir.tensors) < final_prelu_tensor_count
    ):
        session.record_phase_result(
            "shape_reconciliation.primary.final_prelu",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    # Absolute-final reshape cleanup:
    # very late repair/reconciliation passes above can still recreate trivial
    # singleton-growth RESHAPE chains (e.g. 2D->3D->4D Conv1D input shims).
    final_consecutive_reshape_stats = run_consecutive_reshape_cleanup(
        model_ir,
        layout_state=session.layout_state,
        diagnostics=session.diagnostics,
    )
    if (
        int(
            final_consecutive_reshape_stats.get(
                "removed_noop_reshape_chains",
                0,
            )
        )
        + int(
            final_consecutive_reshape_stats.get(
                "rewritten_consecutive_reshape_passthrough_chains",
                0,
            )
        )
        + int(
            final_consecutive_reshape_stats.get(
                "rewritten_fanout_bypass_reshape_passthrough_chains",
                0,
            )
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_consecutive_reshape",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    # Keep this after the final shape reconciliation: earlier than this,
    # SiNet-specific residual branches are not yet in their terminal form and
    # the strict matcher can fire on upstream blocks instead.
    final_sinet_late_residual_stats = (
        _optimize_sinet_late_residual_pre_add_mul_add_prelu_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_sinet_late_residual_stats.get(
            "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_sinet_late_residual",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    final_sinet_preadd_fanout_stats = (
        _optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_sinet_preadd_fanout_stats.get(
            "optimized_sinet_deep_skip_pre_add_concat_prelu_fanout_chains",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_sinet_preadd_fanout",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    final_sinet_dual_resize_stats = (
        _optimize_sinet_deep_skip_dual_resize_affine_transpose_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_sinet_dual_resize_stats.get(
            "optimized_sinet_deep_skip_dual_resize_affine_transpose_chains",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_sinet_dual_resize",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    final_sinet_shared_post_stats = (
        _optimize_sinet_shared_post_prelu_transpose_fanout_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_sinet_shared_post_stats.get(
            "optimized_sinet_shared_post_prelu_transpose_fanout_chains",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_sinet_shared_post",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    final_sinet_deep_skip_stats = (
        _optimize_sinet_deep_skip_concat_resize_affine_tail_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_sinet_deep_skip_stats.get(
            "optimized_sinet_deep_skip_concat_resize_affine_tail_chains",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_sinet_deep_skip",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    # Keep this after all late SiNet residual/deep-skip folds: those passes can
    # still recreate the mid-stage concat+resize affine NHWC/NCHW bridge.
    final_sinet_concat_resize_stats = (
        _optimize_sinet_concat_resize_affine_transpose_chains(
            model_ir,
            layout_state=session.layout_state,
        )
    )
    if int(
        final_sinet_concat_resize_stats.get(
            "optimized_sinet_concat_resize_affine_transpose_chains",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_reconciliation.primary.final_sinet_concat_resize",
            _reconcile_static_tensor_shapes(
                model_ir,
                include_mutation_count=True,
            ),
        )
    final_high_rank_bmm_stats = _compress_static_high_rank_batch_matmul(
        model_ir,
        layout_state=session.layout_state,
    )
    if int(
        final_high_rank_bmm_stats.get(
            "compressed_static_high_rank_batch_matmul",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_topology.primary.final_high_rank_batch_matmul",
            run_static_shape_topology_reconciliation(model_ir),
        )
    final_pad_layout_stats = repair_channel_last_inputs_for_channel_first_pad(
        model_ir,
        layout_state=session.layout_state,
    )
    if int(
        final_pad_layout_stats.get(
            "repaired_channel_last_inputs_for_channel_first_pad",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_topology.primary.final_pad_layout",
            run_static_shape_topology_reconciliation(model_ir),
        )
    final_conv_input_tensor_count = len(model_ir.tensors)
    final_conv_input_stats = {
        **_repair_stale_nchw_to_nhwc_conv_input_transposes(model_ir),
        "pruned_unused_tensors": max(
            0,
            final_conv_input_tensor_count - len(model_ir.tensors),
        ),
    }
    if int(
        final_conv_input_stats.get(
            "repaired_stale_nchw_to_nhwc_conv_input_transposes",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_topology.primary.final_conv_input",
            run_static_shape_topology_reconciliation(model_ir),
        )
    final_concat_layout_stats = _repair_mixed_nhwc_inputs_for_nchw_concat(
        model_ir
    )
    if int(
        final_concat_layout_stats.get(
            "repaired_mixed_nhwc_inputs_for_nchw_concat",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_topology.primary.final_mixed_concat",
            run_static_shape_topology_reconciliation(model_ir),
        )
    final_concat_axis_stats = _repair_nchw_concat_transpose_conv_axes(model_ir)
    if int(
        final_concat_axis_stats.get(
            "repaired_nchw_concat_transpose_conv_axes",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_topology.primary.final_concat_axis",
            run_static_shape_topology_reconciliation(model_ir),
        )
    final_binary_layout_tensor_count = len(model_ir.tensors)
    final_binary_layout_stats = {
        **_repair_stale_nchw_to_nhwc_channelwise_binary_transposes(model_ir),
        "pruned_unused_tensors": max(
            0,
            final_binary_layout_tensor_count - len(model_ir.tensors),
        ),
    }
    if int(
        final_binary_layout_stats.get(
            "repaired_stale_nchw_to_nhwc_channelwise_binary_transposes",
            0,
        )
    ) > 0:
        session.record_phase_result(
            "shape_topology.primary.final_binary_layout",
            run_static_shape_topology_reconciliation(model_ir),
        )
    _advance_post_progress()
    if post_progress_bar is not None:
        post_progress_spinner.stop()
        post_progress_bar.close()

    # Terminal layout cleanup can expose stale direct-NCHW fallback bridges
    # that were not identifiable before the final annotations were available.
    _final_binary_layout_convergence_stats = (
        _run_indexed_binary_layout_convergence(model_ir)
    )
    _final_high_rank_binary_stats = coalesce_static_high_rank_binary_operators(
        model_ir,
        layout_state=session.layout_state,
    )
    _final_dynamic_boundary_signature_stats = (
        _realign_dynamic_boundary_shape_signature_map(model_ir)
    )
    session.record_phase_result(
        "layout_validation.primary.terminal",
        run_topology_layout_validation(model_ir),
    )
    return _finalize_model_ir(model_ir)
