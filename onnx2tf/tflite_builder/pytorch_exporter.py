from __future__ import annotations

import ast
import contextlib
import copy
import functools
import json
import math
import multiprocessing
import os
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
import zipfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import onnx
import torch

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder._pytorch_exporter_native_codegen_common import (
    _NativeCodegenBindings,
    _NativeCodegenState,
    _NativeModelFileWriterContext,
)
from onnx2tf.tflite_builder._pytorch_exporter_native_codegen_pipeline import (
    execute_native_codegen_pipeline,
)
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    channel_first_logical_layout,
    channel_last_logical_layout,
    infer_model_ir_logical_layouts,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_rank,
    normalize_logical_layout,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.pytorch_capabilities import (
    _DIRECT_CODEGEN_SUPPORTED_OP_TYPES,
    _can_emit_direct_module_call_for_codegen,
    _ensure_direct_codegen_supported,
    _ensure_native_export_supported_ops,
    _ensure_no_custom_ops,
    _ensure_supported_ops,
    _is_direct_codegen_unsupported_error,
    _is_channel_last_layout_for_codegen,
    _supports_runtime_wrapper_model_ir,
    get_supported_pytorch_kernel_op_types,
)
from onnx2tf.tflite_builder.pytorch_binary_policy import (
    _all_consumers_are_channel_first_binary_ops_for_codegen,
    _binary_operand_expr_for_codegen,
    _binary_output_target_shape_literal_for_codegen,
    _binary_requires_runtime_alignment_for_codegen,
    _binary_runtime_shape_passthrough_operand_for_codegen,
    _can_emit_channel_first_binary_op_for_codegen,
    _can_omit_materialized_channel_last_alias_recursive_for_codegen,
    _channel_first_binary_input_expr_for_codegen,
    _preferred_binary_alignment_anchor_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_concat_policy import (
    _can_fold_channel_last_alias_slice_consumer_for_codegen,
    _can_keep_channel_first_slice_output_for_codegen,
    _channel_first_concat_input_expr_for_codegen,
    _is_valid_concat_axis_for_channel_first_shapes_for_codegen,
    _resolve_concat_axis_for_channel_first_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_channel_first_policy import (
    _can_emit_channel_first_shape_preserving_unary_op_for_codegen,
    _can_resolve_channel_first_expr_statically_for_codegen,
    _channel_first_passthrough_input_expr_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_constant_policy import (
    _axis_expr_from_input_for_codegen,
    _constant_pad_pairs_for_codegen,
    _int_scalar_literal_expr_for_codegen,
    _is_constant_tensor_name_for_codegen,
    _pad_literal_expr_for_codegen,
    _reshape_shape_tensor_uses_runtime_dims_for_codegen,
    _scalar_literal_expr_for_codegen,
    _shape_tensor_constant_is_non_zero_int_vector_for_codegen,
    _static_mirror_pad_expr_for_codegen,
    _static_int_tensor_values_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_constant_alias_policy import (
    _binary_trailing_axis_constant_buffer_alias_shape_for_codegen,
    _channel_first_rank4_constant_buffer_alias_shape_for_codegen,
    _constant_permute_for_broadcast_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_export_errors import (
    ModelIRPyTorchExportError,
    NativePyTorchGenerationTimeoutError,
)
from onnx2tf.tflite_builder.pytorch_expression_policy import (
    _channel_first_constant_expr_for_buffer_attr_for_codegen,
    _derived_local_var_name_for_codegen,
    _permuted_constant_expr_for_tensor_name_for_codegen,
    _tensor_dtype_name_for_codegen,
    _tensor_expr_for_channel_first_bridge_for_codegen,
    _tensor_expr_for_codegen,
    _transposed_constant_expr_for_tensor_name_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_fast_precanonicalize_policy import (
    _FastPrecanonicalizeRepairContext,
    _build_fast_precanonicalize_repair_context,
    _convert_nchw_pad_to_nhwc_pad_values,
    _convert_nhwc_pad_to_nchw_pad_values,
    _fast_precanonicalize_expr_identifiers,
    _fast_precanonicalize_has_channel_last_spatial_consumer,
    _fast_precanonicalize_infer_consumer_layout,
    _fast_precanonicalize_is_cf_like,
    _fast_precanonicalize_is_nhwc_like,
    _fast_precanonicalize_preferred_channel_count,
    _fast_precanonicalize_resolve_alias,
    _has_immediate_rank4_permute_source,
    _infer_unique_channel_count_from_rank4_shape,
    _repair_binary_alignment_layout,
    _repair_cf_pool_target_shape,
    _repair_cf_resize_target_shape,
    _repair_cf_reduce_max_axis,
    _repair_cf_softmax_axis,
    _repair_concat_axis_from_input_layouts,
    _repair_dynamic_cf_binary_anchor_at,
    _repair_dynamic_cf_binary_anchor_shapes,
    _repair_cf_gather_slice_at,
    _repair_nhwc_average_pool_binary_bridge,
    _repair_nhwc_buffer_binary_alignment_at,
    _repair_split_axis_from_consumers,
    _repair_singleton_reshape_cf_binary_at,
    _repair_terminal_classifier_tail_layout,
    _propagate_cf_prelu_output,
    _restore_channel_last_spatial_pool_chains,
)
from onnx2tf.tflite_builder.pytorch_fusion_policy import (
    _match_affine_layer_norm_for_codegen,
    _match_if_axis0_tensor_mux_slice_for_codegen,
    _match_swish_activation_pattern_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_emitters import (
    _DIRECT_CODEGEN_BINARY_FUNCTIONS,
    _DIRECT_CODEGEN_MODULE_OP_TYPES,
    _DIRECT_CODEGEN_UNARY_EXPRESSIONS,
    _activation_lines_for_codegen,
    _emit_maybe_aligned_expr_for_codegen,
    _emit_module_output_expr_for_codegen,
    _emit_native_binary_op_for_codegen,
    _emit_native_concat_op_for_codegen,
    _emit_native_direct_module_op_for_codegen,
    _emit_native_shape_transform_misc_op_for_codegen,
    _emit_native_transpose_op_for_codegen,
    _emit_native_unary_op_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _add_synthetic_tensor_to_model_ir,
    _broadcast_shapes_relaxed,
    _extract_statement_assignments,
    _extract_statement_loads,
    _is_all_ones_shape,
    _remap_axis_values_through_permutation,
    _remap_mask_bits_through_permutation,
    _shape_can_broadcast_to_target_relaxed,
    _shape_lists_equal,
    _shape_lists_equal_relaxed,
    _shape_literal,
)
from onnx2tf.tflite_builder.pytorch_codegen_stages import (
    _build_forward_stage_methods,
    _build_named_encoder_methods_composite,
)
from onnx2tf.tflite_builder.pytorch_codegen_values import (
    _conv_block_activation_config,
    _conv_block_activation_config_from_fused_name,
    _is_small_inline_constant_tensor,
    _python_literal_for_constant_tensor,
    _torch_dtype_literal,
)
from onnx2tf.tflite_builder.pytorch_shape_policy import (
    _conv2d_input_pre_permute_for_codegen,
    _conv2d_output_spatial_shape_for_codegen,
    _conv2d_same_pad_padding_arg_for_codegen,
    _conv3d_output_spatial_shape_for_codegen,
    _conv3d_transpose_output_spatial_shape_for_codegen,
    _fast_precanonicalize_rank4_layout_hint,
    _infer_batch_matmul_shape_for_codegen,
    _infer_conv2d_ctor_params_for_codegen,
    _infer_conv2d_layout_candidate_for_codegen,
    _infer_conv3d_ctor_params_for_codegen,
    _infer_conv3d_transpose_ctor_params_for_codegen,
    _infer_reduction_shape_for_codegen,
    _matmul_broadcast_shape_for_codegen,
    _normalize_cf_rank4_shape,
    _normalize_nhwc_rank4_shape,
    _reshape_preserves_channel_last_sequence_for_codegen,
    _reshape_special_layout_plan,
    _should_skip_align_for_shape_preserving_unary_for_codegen,
    _topk_codegen_layout_bridge_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_shape_expression_policy import (
    _reconstruct_shape_list_expr_for_codegen,
    _reconstruct_shape_scalar_expr_for_codegen,
    _shape_tensor_length_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_naming import (
    _build_buffer_attr_name_map,
    _build_tensor_var_name_map,
    _canonical_codegen_name_for_codegen,
    _direct_codegen_module_attr_base,
    _make_tensor_storage_name_map,
    _make_unique_identifier,
    _next_unique_attr_name_for_codegen,
    _sanitize_python_identifier,
    _shorten_generated_python_identifier,
)
from onnx2tf.tflite_builder.pytorch_nms_policy import (
    _is_identity_nms_postprocess_gather_for_codegen,
    _range_only_feeds_identity_nms_postprocess_gathers_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_package_sources import (
    _build_native_runtime_source,
    _patch_generated_runtime_pool2d_channel_last_recovery,
    _write_generated_package_common_files,
    _write_wrapper_model_file,
)
from onnx2tf.tflite_builder.pytorch_package_selection import (
    _has_tflite_import_preferred_control_or_recurrent_ops,
    _should_prefer_saved_model_backed_package,
    _should_prefer_tflite_backed_package,
)
from onnx2tf.tflite_builder.pytorch_indexing_codegen import (
    _direct_dynamic_gather_expr,
    _direct_gather_expr,
    _direct_gather_reshape_expr,
    _direct_slice_expr,
    _direct_strided_slice_expr,
    _direct_symbolic_strided_slice_expr,
    _is_suffix_flatten_gather_reshape,
    _reshape_is_plain_singleton_axis_drop,
    _should_elide_crd_to_dcr_gather_for_depth_to_space,
)
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _base_target_shape_values_for_model_ir,
    _channel_first_shape_for_tensor_for_codegen,
    _channel_first_shape_values_for_model_ir,
    _expected_channel_dim_for_tensor_for_codegen,
    _gather_input_pre_permute_for_codegen,
    _infer_effective_rank4_runtime_layout_for_codegen,
    _is_sequential_single_input_graph_for_codegen,
    _native_codegen_cache_bucket_for_model_ir,
    _producer_op_for_model_ir,
    _rank4_channel_first_shape_for_tensor_for_codegen,
    _resize_target_shape_literal_for_model_ir,
    _resolve_channel_first_named_tensor_shape_for_codegen,
    _target_shape_literal_for_model_ir,
    _target_shape_values_for_model_ir,
    _tensor_shape_list_for_model_ir,
)
from onnx2tf.tflite_builder.pytorch_source_graph_rewrites import (
    _bridge_boundary_metadata_gather_nd_inputs,
    _infer_gather_nd_shape_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_state_dict_support import (
    _build_native_generated_state_dict,
)
from onnx2tf.tflite_builder.pytorch_string_normalizer_exporter import (
    _extract_string_normalizer_config_from_onnx_graph,
    export_pytorch_package_from_string_normalizer_onnx,
)
from onnx2tf.tflite_builder.pytorch_runtime_wrapper_exporter import (
    _export_runtime_wrapper_package_from_model_ir,
)
from onnx2tf.tflite_builder.pytorch_reshape_policy import (
    _reshape_codegen_is_plain_data_only_for_codegen,
    _reshape_prefers_feature_last_for_adjx_batch_matmul_for_codegen,
    _static_sequence_length_for_model_ir,
    _tensor_exact_static_shape_list_for_model_ir,
)
from onnx2tf.tflite_builder.pytorch_reduction_policy import (
    _channel_first_reduction_plan_for_codegen,
    _direct_mean_reduction_expr_for_codegen,
    _normalized_constant_reduction_axes_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_recurrent_codegen_policy import (
    _require_constant_array_from_model_ir,
    _sequence_lstm_bias_array_for_model_ir,
)
from onnx2tf.tflite_builder.pytorch_source_parser import (
    _SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN,
    _any_line_matches,
    _count_lines_matching,
    _extract_prefixed_call_exprs,
    _model_source_lines,
    _normalize_permute_dims_expr,
    _parse_aligned_binary_assign_with_shape,
    _parse_align_binary_inputs_to_anchor_assign_with_shape,
    _parse_align_tensor_target_shape_expr,
    _parse_align_tensor_target_shape_assign,
    _parse_apply_concat_inputs_axis_and_shape,
    _parse_apply_pool2d_assign_with_shape,
    _parse_apply_pool2d_input_and_channel_last,
    _parse_apply_pool2d_input_expr,
    _parse_apply_pool2d_input_channel_last_and_is_max,
    _parse_apply_resize_assign,
    _parse_apply_resize_input_size_shape_and_channel_last,
    _parse_apply_resize_input_and_channel_last,
    _parse_apply_softmax_input_axis_and_shape,
    _parse_apply_softmax_input_and_axis,
    _parse_apply_softmax_assign,
    _parse_binary_add_args,
    _parse_binary_mul_args,
    _parse_channel_last_gather_slice_assign,
    _parse_constant_pad_assign,
    _parse_copy_call_expr,
    _parse_dynamic_apply_pool2d_assign,
    _parse_dynamic_binary_add_align_assign,
    _parse_int_list_literal,
    _parse_local_response_norm_assign,
    _parse_rank4_shape_expr,
    _parse_rank4_shape_literal,
    _parse_reduce_max_assign,
    _parse_simple_assignment_line,
    _parse_simple_assignment_line_cached,
    _parse_simple_return_identifier,
    _parse_static_binary_add_align_assign,
    _parse_tensor_split_assign,
    _parse_torch_permute_assign,
    _parse_torch_cat_inputs_and_dim,
    _resolve_nhwc_to_nchw_bridge_source,
    _parse_local_response_norm_input_expr,
    _split_top_level_csv_exprs,
    _strip_outer_parentheses,
)
from onnx2tf.tflite_builder.pytorch_source_rewrites import (
    _collapse_redundant_torch_permute_chains,
    _fold_boundary_transpose_pad_conv_bridges,
    _fold_channel_first_gap_conv_bridges,
    _fold_channel_first_hardsigmoid_gate_conv_bridges,
    _fold_channel_last_affine_conv_bridges,
    _fold_channel_last_prelu_bridges,
    _fold_rank4_reshape_permute_conv_bridges,
    _inline_trivial_public_layout_bridge_aliases,
    _prune_dead_forward_lines,
    _repair_channel_last_gap_conv_inputs,
    _repair_exported_program_direct_conv_cf_add_targets,
    _rewrite_channel_first_gap_outputs_to_explicit_channel_last,
    _rewrite_channel_first_se_scale_binary_bridges,
    _rewrite_channel_last_binary_bridge_chains,
    _rewrite_channel_last_gap_means_to_reduce_mean,
)
from onnx2tf.tflite_builder.pytorch_artifact_exporters import (
    _export_dynamo_onnx_from_generated_package,
    _export_exported_program_from_generated_package,
    export_pytorch_package_from_saved_model_artifact,
    export_pytorch_package_from_tflite_artifact,
    export_torchscript_from_generated_package,
)
from onnx2tf.tflite_builder.pytorch_exported_program_archive import (
    _fold_inverse_permute_round_trips_in_exported_program_archive,
    _strip_stack_traces_from_exported_program_archive,
)
from onnx2tf.tflite_builder.pytorch_export_support import (
    _build_metadata_payload,
    _build_pytorch_export_example_inputs,
    _build_torchscript_example_inputs,
    _generated_package_non_native_skip_reason,
    _generated_package_torch_export_skip_reason,
    _load_generated_package_export_metadata,
    _metadata_has_dynamic_public_inputs,
    _remove_generated_package_artifact_if_exists,
    _run_generated_package_export_child,
    _sanitize_torchscript_file_stem,
    _write_generated_package_export_metadata,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _perm_cl_to_cf,
    _permute_shape,
    _preferred_reshape_target_values,
    _is_layout_only_transpose_by_shape,
    _clone_tensor,
    _read_transpose_perm,
    _read_onnx_squeeze_axes,
    _read_onnx_unsqueeze_axes,
    _inverse_axis_permutation,
    _pad_output_matches_pre_permuted_input,
    _should_emit_channel_last_space_to_depth,
    _should_emit_channel_last_depth_to_space,
    _can_emit_direct_torch_reshape_shape,
    _is_channel_last_factorized_reshape,
    _is_channel_last_factorized_rank3_sequence_reshape,
    _is_channel_last_factorized_rank3_sequence_reshape_by_shape,
    _has_channel_last_factorized_rank3_sequence_consumer,
    _tensor_name_suggests_channel_last_layout_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_layout_bridge_policy import (
    _fold_single_consumer_public_input_bridge_for_codegen,
    _has_channel_last_consumer_hint_for_same_shape_transpose_for_codegen,
    _is_batchless_rank3_public_output_transpose_for_codegen,
    _match_single_consumer_layout_bridge_transpose_for_codegen,
)
from onnx2tf.tflite_builder.pytorch_onnx_utils import (
    _clear_onnx_graph_and_node_metadata_in_place,
    _onnx_node_maps,
    _onnx_node_attr,
    _onnx_set_node_attr,
    _onnx_replace_all_node_inputs,
    _onnx_remove_nodes_by_name,
    _onnx_repair_inferred_shapes_in_place,
    _onnx_get_initializer_index,
    _onnx_set_initializer_array,
    _onnx_make_unique_initializer_name,
    _onnx_get_initializer_array,
    _onnx_get_initializer_scalar,
    _onnx_evaluate_constant_scatter_nd,
    _onnx_evaluate_constant_reshape,
    _onnx_evaluate_constant_binary_elementwise,
    _onnx_get_value_info_shape,
    _onnx_get_tensor_elem_type,
    _onnx_resolve_static_shape,
    _onnx_restore_missing_internal_pad_value_info_shapes_in_place,
    _onnx_fold_constant_scatter_nd_in_place,
    _onnx_fold_constant_reshape_in_place,
    _onnx_fold_constant_binary_elementwise_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_layout_passes import (
    _onnx_resolve_rank4_shape,
    _onnx_convert_pads_nhwc_to_nchw,
    _onnx_rewrite_slice_axis_to_nchw_in_place,
    _onnx_fold_relu_layout_bridges_in_place,
    _onnx_fold_reducesum_sigmoid_layout_bridges_in_place,
    _onnx_fold_mul_reducesum_sigmoid_layout_bridges_in_place,
    _onnx_fold_inverse_transpose_pairs_in_place,
    _onnx_fold_pad_layout_bridges_in_place,
    _onnx_fold_residual_add_layout_bridges_in_place,
    _onnx_remove_passthrough_identity_nodes_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_optimizer import (
    _optimize_dynamo_exported_onnx_in_place,
)
from onnx2tf.tflite_builder.pytorch_onnx_artifact_support import (
    _merge_reference_public_boundary_metadata,
    _sanitize_dynamo_exported_onnx_metadata,
)
from onnx2tf.tflite_builder.passes.pytorch_compat import (
    _restore_same_average_pool_exclude_pad_correction_for_native_runtime,
)
from onnx2tf.tflite_builder.passes.pytorch_layout_validation import (
    _collect_feature_last_sequence_tensor_names,
)
from onnx2tf.tflite_builder.passes.pytorch_normalization import (
    _collect_model_op_types,
    normalize_model_ir_for_pytorch_channel_first,
    prepare_model_ir_for_native_pytorch,
)
from onnx2tf.tflite_builder.passes.pytorch_recurrent import (
    _can_direct_codegen_sequence_lstm_op,
    _can_direct_codegen_sequence_rnn_op,
    _sequence_lstm_index_spec,
    _sequence_lstm_input_name,
)
from onnx2tf.tflite_builder.tflite_importer import (
    import_model_ir_from_tflite,
)


def _prepare_native_codegen_state(
    context: _NativeModelFileWriterContext,
) -> _NativeCodegenState:
    state = _NativeCodegenState(context=context)
    cache_bucket = _native_codegen_cache_bucket_for_model_ir(model_ir=context.model_ir)
    cache_bucket["graph_index"] = context.graph_index
    state.used_local_var_names = set(context.tensor_var_names.values())
    state.public_input_names = {str(name) for name in list(context.model_ir.inputs)}
    state.public_layout_bridge_tensor_names = {
        str(name)
        for name in list(context.model_ir.metadata.get("public_layout_bridge_tensor_names", []))
        if str(name) != ""
    }
    return state


def _build_native_codegen_bindings(
    state: _NativeCodegenState,
) -> _NativeCodegenBindings:
    _ = state
    return _NativeCodegenBindings(
        module_globals=dict(globals()),
        canonicalize_generated_model_source_fn=_canonicalize_generated_model_source_for_raw_export_with_fast_path,
    )


def _build_native_constant_aliases(
    state: _NativeCodegenState,
    bindings: _NativeCodegenBindings,
) -> None:
    _ = state
    _ = bindings


def _emit_native_forward_lines(
    state: _NativeCodegenState,
    bindings: _NativeCodegenBindings,
) -> None:
    execute_native_codegen_pipeline(state, bindings)


def _finalize_native_codegen(
    state: _NativeCodegenState,
    bindings: _NativeCodegenBindings,
) -> List[Tuple[str, str]]:
    _ = bindings
    return [] if state.load_specs_result is None else list(state.load_specs_result)


def export_dynamo_onnx_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    native_package_generation_timeout_sec: Optional[int] = 0,
    raise_on_failure: bool = True,
) -> Optional[str]:
    return _export_dynamo_onnx_from_generated_package(
        package_dir=package_dir,
        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
        shape_hints=shape_hints,
        test_data_nhwc_path=test_data_nhwc_path,
        native_package_generation_timeout_sec=native_package_generation_timeout_sec,
        raise_on_failure=raise_on_failure,
        temporarily_rewrite_generated_model_source_for_exported_program_fn=(
            _temporarily_rewrite_generated_model_source_for_exported_program
        ),
        reapply_post_export_final_model_repairs_fn=(
            _reapply_post_export_final_model_repairs
        ),
    )


def export_exported_program_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    native_package_generation_timeout_sec: Optional[int] = 0,
    raise_on_failure: bool = True,
) -> Optional[str]:
    return _export_exported_program_from_generated_package(
        package_dir=package_dir,
        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
        shape_hints=shape_hints,
        test_data_nhwc_path=test_data_nhwc_path,
        native_package_generation_timeout_sec=native_package_generation_timeout_sec,
        raise_on_failure=raise_on_failure,
        temporarily_rewrite_generated_model_source_for_exported_program_fn=(
            _temporarily_rewrite_generated_model_source_for_exported_program
        ),
        reapply_post_export_final_model_repairs_fn=(
            _reapply_post_export_final_model_repairs
        ),
    )


def _canonicalize_generated_model_source_for_raw_export(
    package_path: Path,
    model_ir: ModelIR | None = None,
) -> None:
    def _convert_nhwc_pad_to_nchw_pad_for_source(pad_values: list[int]) -> list[int] | None:
        if len(pad_values) % 2 != 0 or len(pad_values) > 8:
            return None
        nhwc_inner_to_outer = ["C", "W", "H", "N"]
        nchw_inner_to_outer = ["W", "H", "C", "N"]
        semantic_pairs = {name: [0, 0] for name in nhwc_inner_to_outer}
        pair_count = len(pad_values) // 2
        for idx in range(pair_count):
            semantic_pairs[nhwc_inner_to_outer[idx]] = [
                int(pad_values[idx * 2]),
                int(pad_values[idx * 2 + 1]),
            ]
        output_pairs = [semantic_pairs[name] for name in nchw_inner_to_outer]
        last_nonzero = -1
        for idx, pair in enumerate(output_pairs):
            if pair != [0, 0]:
                last_nonzero = idx
        if last_nonzero < 0:
            return []
        flattened = []
        for pair in output_pairs[: last_nonzero + 1]:
            flattened.extend(pair)
        return flattened

    model_path = package_path / "model.py"
    if not model_path.exists():
        return
    lines = model_path.read_text(encoding="utf-8").splitlines()
    suppress_import_line = "import logging"
    suppress_apply_line = "logging.getLogger('torch.onnx._internal.exporter._registration').setLevel(logging.ERROR)"
    if suppress_apply_line not in lines:
        torch_import_index = next(
            (i for i, line in enumerate(lines) if line.strip() == "import torch"),
            None,
        )
        if torch_import_index is not None:
            if suppress_import_line not in lines:
                lines.insert(torch_import_index, suppress_import_line)
                torch_import_index += 1
            insert_index = torch_import_index + 1
            while insert_index < len(lines) and lines[insert_index].strip() == "":
                insert_index += 1
            lines.insert(insert_index, "")
            lines.insert(insert_index, suppress_apply_line)
    lines = _fold_channel_first_gap_conv_bridges(lines)
    lines = _repair_channel_last_gap_conv_inputs(lines)
    lines = _fold_channel_first_hardsigmoid_gate_conv_bridges(lines)
    lines = _rewrite_channel_first_se_scale_binary_bridges(lines)
    lines = _rewrite_channel_first_gap_outputs_to_explicit_channel_last(lines)
    lines = _rewrite_channel_last_gap_means_to_reduce_mean(lines)
    lines = _repair_channel_last_gap_conv_inputs(lines)
    model_ir_shape_map: Dict[str, List[int]] = {}
    model_ir_cf_names: Set[str] = set()
    tensor_name_by_var_name: Dict[str, str] = {}
    public_output_name_by_var_name: Dict[str, str] = {}
    if model_ir is not None:
        tensor_var_name_map = _build_tensor_var_name_map(model_ir)
        tensor_name_by_var_name = {
            str(var_name): str(tensor_name)
            for tensor_name, var_name in tensor_var_name_map.items()
        }
        public_output_name_by_var_name = {
            str(var_name): str(tensor_name)
            for tensor_name, var_name in tensor_var_name_map.items()
            if str(tensor_name) in {str(name) for name in list(model_ir.outputs)}
        }
        for tensor_name, tensor in model_ir.tensors.items():
            shape_values = list(getattr(tensor, "shape", []) or [])
            if len(shape_values) > 0 and all(int(v) > 0 for v in shape_values):
                model_ir_shape_map[str(tensor_name)] = [int(v) for v in shape_values]
            if is_channel_first_logical_layout(normalize_logical_layout(tensor.logical_layout)):
                model_ir_cf_names.add(str(tensor_name))

    resolved_tensor_name_cache: Dict[str, str] = {}
    model_ir_exact_shape_cache: Dict[str, List[int] | None] = {}
    model_ir_channel_first_cache: Dict[str, bool] = {}
    recent_rank4_shape_cache: Dict[Tuple[str, int], List[int] | None] = {}
    recent_rank4_shape_by_name_cache: Dict[Tuple[str, int], Tuple[int, List[int] | None]] = {}
    known_cf_name_cache: Dict[Tuple[str, int, int, int, int, int], bool] = {}
    expr_identifier_tokens_cache: Dict[str, Set[str]] = {}
    simple_identifier_expr_cache: Dict[str, bool] = {}
    regex_match_cache: Dict[Tuple[str, str], re.Match[str] | None] = {}

    function_start_by_index: List[int] = [-1] * len(lines)
    current_function_start = -1
    for idx, line in enumerate(lines):
        if line.startswith("    def "):
            current_function_start = idx
        function_start_by_index[idx] = current_function_start

    function_end_by_index: List[int] = [len(lines)] * len(lines)
    next_function_start = len(lines)
    for idx in range(len(lines) - 1, -1, -1):
        function_end_by_index[idx] = next_function_start
        if lines[idx].startswith("    def "):
            next_function_start = idx

    def _resolve_model_ir_tensor_name(name: str) -> str:
        normalized_name = str(name)
        cached = resolved_tensor_name_cache.get(normalized_name, None)
        if cached is not None:
            return cached
        resolved = tensor_name_by_var_name.get(normalized_name, None)
        if resolved is None and normalized_name.endswith("_cf"):
            resolved = tensor_name_by_var_name.get(normalized_name[:-3], None)
        if resolved is None:
            resolved = normalized_name
        resolved = str(resolved)
        resolved_tensor_name_cache[normalized_name] = resolved
        return resolved

    def _model_ir_exact_shape(name: str) -> List[int] | None:
        normalized_name = str(name)
        if normalized_name in model_ir_exact_shape_cache:
            return model_ir_exact_shape_cache[normalized_name]
        resolved = model_ir_shape_map.get(_resolve_model_ir_tensor_name(normalized_name), None)
        model_ir_exact_shape_cache[normalized_name] = resolved
        return resolved

    def _model_ir_is_channel_first(name: str) -> bool:
        normalized_name = str(name)
        cached = model_ir_channel_first_cache.get(normalized_name, None)
        if cached is not None:
            return cached
        resolved = _resolve_model_ir_tensor_name(normalized_name) in model_ir_cf_names
        model_ir_channel_first_cache[normalized_name] = resolved
        return resolved

    def _expr_identifier_tokens(expr: str) -> Set[str]:
        cached = expr_identifier_tokens_cache.get(expr, None)
        if cached is not None:
            return cached
        tokens = {
            token
            for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", str(expr))
            if token not in {"torch", "self", "True", "False"}
        }
        expr_identifier_tokens_cache[expr] = tokens
        return tokens

    def _is_simple_identifier_expr(expr: str) -> bool:
        cached = simple_identifier_expr_cache.get(expr, None)
        if cached is not None:
            return cached
        normalized_expr = _strip_outer_parentheses(str(expr).strip())
        result = (
            normalized_expr != ""
            and normalized_expr.replace("_", "").isalnum()
            and " " not in normalized_expr
        )
        simple_identifier_expr_cache[expr] = result
        return result

    def _expr_references_known_cf_identifier(expr: str, singleton_names: Set[str]) -> bool:
        expr_tokens = _expr_identifier_tokens(expr)
        return bool(expr_tokens.intersection(singleton_names | cf_aliases))

    def _cached_regex_match(
        cache_name: str,
        regex: re.Pattern[str],
        text: str,
    ) -> re.Match[str] | None:
        cache_key = (cache_name, text)
        if cache_key in regex_match_cache:
            return regex_match_cache[cache_key]
        match = regex.match(text)
        regex_match_cache[cache_key] = match
        return match

    def _eventual_public_output_exact_shape(
        source_name: str,
        start_index: int,
    ) -> List[int] | None:
        direct_output_name = public_output_name_by_var_name.get(str(source_name), None)
        if direct_output_name is not None:
            return _model_ir_exact_shape(direct_output_name)
        function_end = _function_end_index(start_index)
        reachable_names: Set[str] = {str(source_name)}
        for future_index in range(start_index + 1, function_end):
            alias_assign = _parse_simple_assignment_line(lines[future_index])
            if (
                alias_assign is not None
                and _is_simple_identifier_expr(alias_assign[2])
                and _strip_outer_parentheses(str(alias_assign[2]).strip()) in reachable_names
            ):
                alias_name = str(alias_assign[1])
                reachable_names.add(alias_name)
                output_name = public_output_name_by_var_name.get(alias_name, None)
                if output_name is not None:
                    return _model_ir_exact_shape(output_name)
            return_value = _parse_simple_return_identifier(lines[future_index])
            if return_value is not None:
                output_name = public_output_name_by_var_name.get(
                    str(return_value),
                    None,
                )
                if output_name is not None and str(return_value) in reachable_names:
                    return _model_ir_exact_shape(output_name)
        return None

    def _parse_simple_return_identifier(current_line: str) -> str | None:
        stripped = str(current_line).strip()
        direct = re.fullmatch(r"return\s+([A-Za-z0-9_]+)", stripped)
        if direct is not None:
            return str(direct.group(1))
        parenthesized = re.fullmatch(r"return\s+\(\s*([A-Za-z0-9_]+)\s*\)", stripped)
        if parenthesized is not None:
            return str(parenthesized.group(1))
        return None

    register_buffer_re = re.compile(
        r"^(?P<indent>\s*)self\.register_buffer\('(?P<name>[A-Za-z0-9_]+)', torch\.zeros\(\[(?P<shape>[0-9, ]+)\], dtype=torch\.(?P<dtype>[A-Za-z0-9_]+)\), persistent=(?P<persistent>True|False)\)$"
    )
    self_const_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*=\s*\(*\s*self\.(?P<attr>[A-Za-z0-9_]+)\s*\)*$"
    )
    raw_const_pair_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<pair>[A-Za-z0-9_]+)"
        r"(?::\s*(?:(?:typing\.)?Tuple\[[^\]]+\]|tuple\[[^\]]+\]))?"
        r"\s*=\s*\(?\s*\(\s*\(?\s*(?P<rhs0>[A-Za-z0-9_\.]+)\s*\)?\s*,\s*\(?\s*(?P<rhs1>[A-Za-z0-9_\.]+)\s*\)?\s*\)\s*\)?$"
    )
    raw_tuple_const_alias_re = re.compile(
        r"^(?P<indent>\s*)\(?\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)?\s*=\s*\(?\s*\(?\s*(?P<rhs0>[A-Za-z0-9_\.]+)\s*\)?\s*,\s*\(?\s*(?P<rhs1>[A-Za-z0-9_\.]+)\s*\)?\s*\)?$"
    )
    raw_tuple_const_unpack_re = re.compile(
        r"^(?P<indent>\s*)\(?\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)?\s*=\s*\(*\s*(?P<pair>[A-Za-z0-9_]+)\s*\)*$"
    )
    raw_generic_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)"
        r"(?::\s*(?:torch\.Tensor|(?:typing\.)?Tuple\[[^\]]+\]|tuple\[[^\]]+\]))?"
        r"\s*=\s*\(*\s*(?P<rhs>[A-Za-z0-9_]+)\s*\)*$"
    )
    transposed_const_use_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<expr>.*torch\.matmul\(.+, (?P<temp>[A-Za-z0-9_]+)\.transpose\(-1, -2\)\).*)$"
    )
    scalar_as_tensor_re = re.compile(
        r"torch\.as_tensor\((?P<value>[-+]?(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?), dtype=torch\.[A-Za-z0-9_]+, device=_module_device\(self\)\)"
    )
    scalar_first_binary_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.(?P<op>add|mul)\("
        r"(?P<scalar>[-+]?(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?), "
        r"(?P<rhs>.+)\)$"
    )
    scalar_first_inline_binary_re = re.compile(
        r"torch\.(?P<op>add|mul)\("
        r"(?P<scalar>[-+]?(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?), "
        r"(?P<rhs>[A-Za-z0-9_\.]+)\)"
    )
    tensor_literal_as_tensor_re = re.compile(
        r"torch\.as_tensor\((?P<literal>\[.+?\]), dtype=torch\.(?P<dtype>[A-Za-z0-9_]+), device=_module_device\(self\)\)"
    )
    minimum_scalar_tensor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<prefix>.*)torch\.minimum\((?P<tensor>[A-Za-z0-9_]+), "
        r"torch\.as_tensor\((?P<value>[-+]?(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?), dtype=torch\.[A-Za-z0-9_]+, device=_module_device\(self\)\)\)(?P<suffix>.*)$"
    )
    maximum_scalar_tensor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<prefix>.*)torch\.maximum\((?P<tensor>[A-Za-z0-9_]+), "
        r"torch\.as_tensor\((?P<value>[-+]?(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?), dtype=torch\.[A-Za-z0-9_]+, device=_module_device\(self\)\)\)(?P<suffix>.*)$"
    )
    assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<alias>(?:[A-Za-z0-9_]+_public_layout_bridge|in_public_layout_bridge))\s*=\s*_torch_permute\((?P<input>[A-Za-z0-9_]+), \[0, 2, 3, 1\]\)$"
    )
    trivial_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<alias>(?:[A-Za-z0-9_]+_public_layout_bridge|in_public_layout_bridge))\s*=\s*(?P<input>[A-Za-z0-9_]+)$"
    )
    generic_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>[A-Za-z0-9_]+)$"
    )
    return_value_re = re.compile(
        r"^(?P<indent>\s*)return (?P<value>[A-Za-z0-9_]+)$"
    )
    rank4_reshape_consumer_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<input>[A-Za-z0-9_]+), _resolve_reshape_shape\(\[(?P<shape>[0-9,\- ]+)\], (?P=input), allow_zero=False\)\)$"
    )
    rank3_reshape_from_rank4_source_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<src>[A-Za-z0-9_]+), "
        r"(?:_resolve_reshape_shape\(\[(?P<resolved_shape>[0-9,\- ]+)\], (?P=src), allow_zero=False\)|\[(?P<shape>[0-9,\- ]+)\])\)$"
    )
    generic_expr_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+)$"
    )
    generic_module_call_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>[A-Za-z0-9_]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    split_re = re.compile(
        r"^(?P<indent>\s*)(?P<outputs>[A-Za-z0-9_, ]+)\s*=\s*list\(torch\.tensor_split\((?P<alias>[A-Za-z0-9_]+), (?P<sections>\d+), dim=_normalize_dim\(3, (?P=alias)\.ndim\)\)\)$"
    )
    generic_split_re = re.compile(
        r"^(?P<indent>\s*)(?P<outputs>[A-Za-z0-9_, ]+)\s*=\s*list\(torch\.tensor_split\((?P<input>[A-Za-z0-9_]+), (?P<sections>\d+), dim=_normalize_dim\((?P<axis>-?\d+), (?P=input)\.ndim\)\)\)$"
    )
    concat_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_concat\(\[(?P<inputs>[A-Za-z0-9_, ]+)\], axis=3, target_shape=\[(?P<shape>[0-9, ]+)\], fused='NONE'\)$"
    )
    generic_apply_concat_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_concat\(\[(?P<inputs>[A-Za-z0-9_, ]+)\], axis=(?P<axis>-?\d+), target_shape=\[(?P<shape>[0-9, ]+)\], fused='(?P<fused>[^']+)'\)$"
    )
    channel_last_gather_slice_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\[:, :, :, \[(?P<indices>[0-9,\s-]+)\]\]$"
    )
    pad_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.pad\(_align_tensor_to_target_shape\((?P<input>[A-Za-z0-9_]+), \[(?P<shape>[0-9, ]+)\]\), \[(?P<pad>[0-9, ]+)\], mode='constant', value=(?P<value>[-+0-9.eE]+)\)$"
    )
    rank3_const_pad_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.pad\((?P<input>[A-Za-z0-9_]+), \[(?P<pad0>-?\d+), (?P<pad1>-?\d+)\], mode='constant', value=(?P<value>[-+0-9.eE]+)\)$"
    )
    rank4_const_pad_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.pad\((?P<input>[A-Za-z0-9_]+), \[(?P<pad0>-?\d+), (?P<pad1>-?\d+), (?P<pad2>-?\d+), (?P<pad3>-?\d+)\], mode='constant', value=(?P<value>[-+0-9.eE]+)\)$"
    )
    rank4_const_pad6_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.pad\((?P<input>[A-Za-z0-9_]+), \[0, 0, (?P<pad0>-?\d+), (?P<pad1>-?\d+), (?P<pad2>-?\d+), (?P<pad3>-?\d+)\], mode='constant', value=(?P<value>[-+0-9.eE]+)\)$"
    )
    aligned_rank4_const_pad6_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.pad\(_align_tensor_to_target_shape\((?P<input>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)\]\), \[0, 0, (?P<pad0>-?\d+), (?P<pad1>-?\d+), (?P<pad2>-?\d+), (?P<pad3>-?\d+)\], mode='constant', value=(?P<value>[-+0-9.eE]+)\)$"
    )
    apply_pool2d_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_pool2d\((?:input=)?(?P<input>[A-Za-z0-9_]+), (?P<rest>.+), target_shape=[\[\(](?P<shape>[0-9, ]+)[\]\)], is_max_pool=(?P<is_max>True|False), channel_last=(?P<channel_last>True|False)\)$"
    )
    local_response_norm_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*F\.local_response_norm\((?P<input>[A-Za-z0-9_]+), size=(?P<size>\d+), alpha=(?P<alpha>[-+0-9.eE]+), beta=(?P<beta>[-+0-9.eE]+), k=(?P<k>[-+0-9.eE]+)\)$"
    )
    cf_nhwc_materialize_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\((?P<src>[A-Za-z0-9_]+)\.permute\(0, 2, 3, 1\)\.contiguous\(\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)\]\)$"
    )
    binary_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    binary_cf_consumer_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs(?:_to_anchor)?\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    singleton_cf_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\((?P<expr>.+), \[(?P<n>\d+), 1, (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    cf_concat_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.cat\(\[(?P<inputs>[A-Za-z0-9_, ]+)\], dim=1\)$"
    )
    generic_torch_cat_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.cat\(\[(?P<inputs>[A-Za-z0-9_, ]+)\], dim=(?P<axis>-?\d+)\)$"
    )
    binary_same_permute_cf_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.(?P<op>add|sub)\("
        r"(?P<a>[A-Za-z0-9_]+)\.permute\(0, 2, 1\)\.contiguous\(\), "
        r"(?P<b>[A-Za-z0-9_]+)\.permute\(0, 2, 1\)\.contiguous\(\)\)$"
    )
    pool2d_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_pool2d\((?:input=)?(?P<input>[A-Za-z0-9_]+), (?P<rest>.+), target_shape=[\[\(](?P<shape>[0-9, ]+)[\]\)], is_max_pool=(?P<is_max>True|False), channel_last=False\)$"
    )
    binary_anchor_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+), \[(?P<n>\d+), 1, (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    same_shape_singleton_reshape_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<expr>.+), \[(?P<n>\d+), 1, (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    unary_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<expr>torch\.(?:clamp|relu|neg|sigmoid|exp)\(.+\))$"
    )
    binary_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<expr>torch\.(?:mul|add|sub|div|minimum|maximum)\(.+\))$"
    )
    simple_binary_expr_re = re.compile(
        r"^torch\.(?P<op>add|sub|mul|div|minimum|maximum)\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\)$"
    )
    transpose_conv_input_bridge_re = re.compile(
        r"^(?P<indent>\s*)(?P<alias>[A-Za-z0-9_]+)\s*=\s*_torch_permute\((?P<src>[A-Za-z0-9_]+), \[0, 2, 3, 1\]\)$"
    )
    transpose_conv_apply_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_module_transpose_conv2d\((?P<input>[A-Za-z0-9_]+), (?P<prefix>.+), target_shape=\[(?P<target>[0-9, ]+)\], fallback_shape=\[(?P<fallback>[0-9, ]+)\], target_logical_layout='NHWC', fused='(?P<fused>[^']+)'\)$"
    )
    transpose_conv_crop_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<src>[A-Za-z0-9_]+)\[0:1, 0:1, (?P<start>\d+):(?P<end>\d+), 0:(?P<width>\d+)\]$"
    )
    transpose_conv_output_permute_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_torch_permute\((?P<src>[A-Za-z0-9_]+), \[0, 3, 1, 2\]\)$"
    )
    self_permute_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P=lhs)\.permute\(0, 3, 1, 2\)\.contiguous\(\)$"
    )
    transpose_conv_bias_fix_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<expr>self\.[A-Za-z0-9_]+)\.permute\(0, 2, 3, 1\)\.contiguous\(\), \[(?P<n>\d+), 1, (?P<w>\d+), 1\]\)$"
    )
    transpose_conv_bias_add_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.add\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    simple_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<src>[A-Za-z0-9_]+)$"
    )
    generic_aligned_tensor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\((?P<expr>.+), \[(?P<shape>[0-9, ]+)\]\)$"
    )
    permute_contiguous_cf_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<src>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)$"
    )
    singleton_const_anchor_fix_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\("
        r"(?P<input>[A-Za-z0-9_]+), torch\.reshape\(self\.(?P<const_attr>[A-Za-z0-9_]+), \[1, 1, 1, 1\]\), \[1, 1, 1, 1\]\)$"
    )
    recent_cf_singleton_shape_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*.*\[(?P<n>\d+), (?P<c>\d+), 1, 1\]\)?$"
    )
    reshape_from_inverse_permute_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<src>[A-Za-z0-9_]+)\.permute\(0, 2, 3, 1\)\.contiguous\(\), \[(?P<shape>[0-9, ]+)\]\)$"
    )
    binary_anchor_align_nhwc_singleton_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), 1\]\)$"
    )
    binary_anchor_align_rank4_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)\]\)$"
    )
    aligned_nhwc_singleton_binary_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\((?P<expr>torch\.(?:add|sub|mul|div|minimum|maximum)\(.+\)), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), 1\]\)$"
    )
    aligned_nhwc_rank4_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\((?P<expr>.+), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)\]\)$"
    )
    permuted_cf_module_input_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<module>self\.[A-Za-z0-9_]+)\((?P<src>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)$"
    )
    output_back_permute_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_torch_permute\((?P<src>[A-Za-z0-9_]+), \[0, 3, 1, 2\]\)$"
    )
    def _resolve_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> str | None:
            parts = _split_top_level_csv_exprs(args_expr)
            input_expr: str | None = None
            perm_expr: str | None = None
            if len(parts) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
            ):
                input_expr, perm_expr = parts[0].strip(), parts[1].strip()
            else:
                kwargs: Dict[str, str] = {}
                for part in parts:
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                        continue
                    key, value = part.split("=", 1)
                    kwargs[key.strip()] = value.strip()
                input_expr = kwargs.get("input", kwargs.get("x"))
                perm_expr = kwargs.get("perm", kwargs.get("dims"))
            if input_expr is None or perm_expr is None:
                return None
            if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
                return None
            if _normalize_permute_dims_expr(perm_expr) != "0,2,3,1":
                return None
            return input_expr

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                return _parse_permute_like_args(stripped[len(prefix) : -1])
        method_match = re.fullmatch(r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped)
        if method_match is not None and _normalize_permute_dims_expr(str(method_match.group("dims"))) == "0,2,3,1":
            return str(method_match.group("input"))
        return None

    def _parse_transpose_conv_input_bridge_assign(
        current_line: str,
    ) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        source_expr = _resolve_nchw_to_nhwc_bridge_source(rhs)
        if source_expr is None:
            return None
        return indent, lhs, source_expr

    def _parse_public_layout_bridge_assign(
        current_line: str,
    ) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        if re.fullmatch(r"(?:[A-Za-z0-9_]+_public_layout_bridge|in_public_layout_bridge)", lhs) is None:
            return None
        if re.fullmatch(r"[A-Za-z0-9_]+", rhs) is not None:
            return indent, lhs, rhs
        source_expr = _resolve_nchw_to_nhwc_bridge_source(rhs)
        if source_expr is None:
            return None
        return indent, lhs, source_expr

    def _parse_align_binary_inputs_to_anchor_assign(
        current_line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(current_line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_a: str | None = None
        input_b: str | None = None
        shape_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
            shape_expr = parts[2].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("a"))
            input_b = kwargs.get("other", kwargs.get("b"))
            shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
        if input_a is None or input_b is None or shape_expr is None:
            return None
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if rank4_shape is None:
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            input_a,
            input_b,
            [int(dim) for dim in rank4_shape],
        )

    def _parse_align_binary_inputs_assign(
        current_line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(current_line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_a: str | None = None
        input_b: str | None = None
        shape_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
            shape_expr = parts[2].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("lhs"))
            input_b = kwargs.get("other", kwargs.get("rhs"))
            shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
        rank4_shape = _parse_rank4_shape_literal(shape_expr) if shape_expr is not None else None
        if input_a is None or input_b is None or rank4_shape is None:
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            input_a,
            input_b,
            [int(dim) for dim in rank4_shape],
        )

    def _parse_singleton_const_reshape_attr(expr: str) -> str | None:
        stripped = str(expr).strip()
        reshape_match = re.fullmatch(r"torch\.reshape\((?P<args>.+)\)", stripped)
        if reshape_match is None:
            return None
        parts = _split_top_level_csv_exprs(str(reshape_match.group("args")))
        input_expr: str | None = None
        shape_expr: str | None = None
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_expr = parts[0].strip()
            shape_expr = parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            shape_expr = kwargs.get("shape")
        if input_expr is None or shape_expr is None:
            return None
        attr_match = re.fullmatch(r"\(*\s*self\.(?P<attr>[A-Za-z0-9_]+)\s*\)*", input_expr)
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if attr_match is None or rank4_shape is None:
            return None
        if [int(dim) for dim in rank4_shape] != [1, 1, 1, 1]:
            return None
        return str(attr_match.group("attr"))

    def _parse_singleton_const_anchor_fix_assign(
        current_line: str,
    ) -> Tuple[str, str, str, str, str] | None:
        parsed = _parse_align_binary_inputs_to_anchor_assign(current_line)
        if parsed is None:
            return None
        indent, lhs0, lhs1, input_a, input_b, rank4_shape = parsed
        if rank4_shape != [1, 1, 1, 1]:
            return None
        const_attr_a = _parse_singleton_const_reshape_attr(input_a)
        const_attr_b = _parse_singleton_const_reshape_attr(input_b)
        if const_attr_a is not None and const_attr_b is None:
            return indent, lhs0, lhs1, input_b, const_attr_a
        if const_attr_b is not None and const_attr_a is None:
            return indent, lhs0, lhs1, input_a, const_attr_b
        return None

    def _parse_permuted_cf_module_input_assign(
        current_line: str,
    ) -> Tuple[str, str, str, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        call_match = re.fullmatch(r"(?P<module>self\.[A-Za-z0-9_]+)\((?P<args>.+)\)", rhs.strip())
        if call_match is None:
            return None
        args = _split_top_level_csv_exprs(str(call_match.group("args")))
        if len(args) != 1:
            return None
        source_expr = _resolve_nhwc_to_nchw_bridge_source(args[0])
        if source_expr is None:
            return None
        return indent, lhs, str(call_match.group("module")), source_expr

    def _parse_transpose_conv_bias_fix_assign(
        current_line: str,
    ) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        call_match = re.fullmatch(r"torch\.reshape\((?P<args>.+)\)", rhs.strip())
        if call_match is None:
            return None
        parts = _split_top_level_csv_exprs(str(call_match.group("args")))
        input_expr: str | None = None
        shape_expr: str | None = None
        if len(parts) == 2 and all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts):
            input_expr, shape_expr = parts[0].strip(), parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            shape_expr = kwargs.get("shape")
        if input_expr is None or shape_expr is None:
            return None
        shape = _parse_rank4_shape_literal(shape_expr)
        if shape is None:
            return None
        stripped = input_expr.strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> str | None:
            parts = _split_top_level_csv_exprs(args_expr)
            source_expr: str | None = None
            perm_expr: str | None = None
            if len(parts) == 2 and all(re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts):
                source_expr, perm_expr = parts[0].strip(), parts[1].strip()
            else:
                kwargs: Dict[str, str] = {}
                for part in parts:
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                        continue
                    key, value = part.split("=", 1)
                    kwargs[key.strip()] = value.strip()
                source_expr = kwargs.get("input", kwargs.get("x"))
                perm_expr = kwargs.get("perm", kwargs.get("dims"))
            if source_expr is None or perm_expr is None:
                return None
            if re.fullmatch(r"[A-Za-z0-9_\.]+", source_expr) is None:
                return None
            if _normalize_permute_dims_expr(perm_expr) != "0,2,3,1":
                return None
            return source_expr

        bridge_expr: str | None = None
        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                bridge_expr = _parse_permute_like_args(stripped[len(prefix) : -1])
                break
        if bridge_expr is None:
            method_match = re.fullmatch(r"(?P<input>[A-Za-z0-9_\.]+)\.permute\((?P<dims>.+)\)", stripped)
            if method_match is not None and _normalize_permute_dims_expr(str(method_match.group("dims"))) == "0,2,3,1":
                bridge_expr = str(method_match.group("input"))
        if bridge_expr is None:
            return None
        return indent, lhs, bridge_expr, list(shape)

    def _parse_transpose_conv_bias_add_assign(
        current_line: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        expr, shape_expr = align_parts
        shape = _parse_rank4_shape_literal(shape_expr)
        if shape is None:
            return None
        add_parts = _parse_binary_add_args(expr)
        if add_parts is None:
            return None
        return indent, lhs, add_parts[0], add_parts[1], list(shape)

    def _parse_output_back_permute_assign(
        current_line: str,
    ) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args_expr: str) -> str | None:
            parts = _split_top_level_csv_exprs(args_expr)
            input_expr: str | None = None
            perm_expr: str | None = None
            if len(parts) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
            ):
                input_expr, perm_expr = parts[0].strip(), parts[1].strip()
            else:
                kwargs: Dict[str, str] = {}
                for part in parts:
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                        continue
                    key, value = part.split("=", 1)
                    kwargs[key.strip()] = value.strip()
                input_expr = kwargs.get("input", kwargs.get("x"))
                perm_expr = kwargs.get("perm", kwargs.get("dims"))
            if input_expr is None or perm_expr is None or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
                return None
            if _normalize_permute_dims_expr(perm_expr) != "0,3,1,2":
                return None
            return input_expr

        for prefix in ("_torch_permute(", "torch.permute("):
            if stripped.startswith(prefix) and stripped.endswith(")"):
                source_name = _parse_permute_like_args(stripped[len(prefix) : -1])
                if source_name is not None:
                    return indent, lhs, source_name
        method_match = re.fullmatch(r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped)
        if method_match is None:
            return None
        if _normalize_permute_dims_expr(str(method_match.group("dims"))) != "0,3,1,2":
            return None
        return indent, lhs, str(method_match.group("input"))
    apply_softmax_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_softmax\((?:input=)?(?P<input>[A-Za-z0-9_]+), axis=(?P<axis>-?\d+), beta=(?P<beta>[-0-9.eE]+), target_shape=[\[\(](?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)[\]\)]\)$"
    )
    reduce_max_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_reduce_max\((?P<input>[A-Za-z0-9_]+), _normalize_axes\(\[(?P<axis>-?\d+)\], (?P=input)\.ndim\), (?P<keepdims>True|False)\)$"
    )
    argmax_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.argmax\((?P<input>[A-Za-z0-9_]+), dim=_normalize_dim\((?P<axis>-?\d+), (?P=input)\.ndim\), keepdim=(?P<keepdim>True|False)\)\.to\(dtype=torch\.int64\)$"
    )
    def _parse_argmax_assign(
        current_line: str,
    ) -> Tuple[str, str, str, int, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        suffix = ".to(dtype=torch.int64)"
        prefix = "torch.argmax("
        if not stripped.endswith(suffix):
            return None
        argmax_expr = stripped[: -len(suffix)].strip()
        if not argmax_expr.startswith(prefix) or not argmax_expr.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(argmax_expr[len(prefix) : -1])
        input_expr: str | None = None
        axis_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    axis_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            positional_index += 1
        if (
            input_expr is None
            or axis_expr is None
            or keepdim_expr is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
            or re.fullmatch(r"True|False", keepdim_expr) is None
        ):
            return None
        axis_match = re.fullmatch(
            r"_normalize_dim\(\s*(?P<axis>-?\d+)\s*,\s*[A-Za-z0-9_]+\.ndim\s*\)",
            axis_expr,
        )
        if axis_match is None:
            return None
        return indent, lhs, input_expr, int(axis_match.group("axis")), keepdim_expr
    rank4_mean_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.mean\((?P<input>[A-Za-z0-9_]+), dim=\[(?P<dim0>-?\d+), (?P<dim1>-?\d+)\], keepdim=(?P<keepdim>True|False)\)$"
    )
    rank4_reshape_consumer_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*torch\.reshape\((?P<input>[A-Za-z0-9_]+), "
        r"(?:_resolve_reshape_shape\(\[(?P<resolved_shape>[0-9,\- ]+)\], (?P=input), allow_zero=False\)|\[(?P<shape>[0-9,\- ]+)\])\)$"
    )
    generic_reshape_consumer_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*torch\.reshape\((?P<input>[A-Za-z0-9_]+), "
        r"(?:_resolve_reshape_shape\(\[(?P<resolved_shape>[0-9,\- ]+)\], (?P=input), allow_zero=False\)|\[(?P<shape>[0-9,\- ]+)\])\)$"
    )
    reduce_sum_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_reduce_sum\((?P<input>[A-Za-z0-9_]+), _normalize_axes\(\[(?P<axis>-?\d+)\], (?P=input)\.ndim\), (?P<keepdims>True|False)\)$"
    )
    def _parse_rank4_reshape_consumer_assign(
        current_line: str,
    ) -> Tuple[str, str, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        call_match = re.fullmatch(r"torch\.reshape\((?P<args>.+)\)", rhs.strip())
        if call_match is None:
            return None
        parts = _split_top_level_csv_exprs(str(call_match.group("args")))
        input_expr: str | None = None
        shape_expr: str | None = None
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_expr, shape_expr = parts[0].strip(), parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            shape_expr = kwargs.get("shape")
        if input_expr is None or shape_expr is None:
            return None
        if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
            return None
        if (
            re.fullmatch(
                rf"_resolve_reshape_shape\(\[[0-9,\- ]+\], {re.escape(input_expr)}, allow_zero=False\)",
                shape_expr,
            )
            is None
            and _parse_rank4_shape_literal(shape_expr) is None
        ):
            return None
        return indent, lhs, input_expr

    def _parse_rank3_reshape_from_rank4_source_assign(
        current_line: str,
    ) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        call_match = re.fullmatch(r"torch\.reshape\((?P<args>.+)\)", rhs.strip())
        if call_match is None:
            return None
        parts = _split_top_level_csv_exprs(str(call_match.group("args")))
        input_expr: str | None = None
        shape_expr: str | None = None
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_expr, shape_expr = parts[0].strip(), parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            shape_expr = kwargs.get("shape")
        if input_expr is None or shape_expr is None:
            return None
        if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
            return None
        resolved_shape_match = re.fullmatch(
            rf"_resolve_reshape_shape\(\[(?P<shape>[0-9,\- ]+)\], {re.escape(input_expr)}, allow_zero=False\)",
            shape_expr,
        )
        if resolved_shape_match is not None:
            shape_values = [
                int(value.strip())
                for value in str(resolved_shape_match.group("shape")).split(",")
                if value.strip()
            ]
        else:
            shape_match = re.fullmatch(r"[\[\(](?P<shape>[0-9,\- ]+)[\]\)]", shape_expr.strip())
            if shape_match is None:
                return None
            shape_values = [
                int(value.strip())
                for value in str(shape_match.group("shape")).split(",")
                if value.strip()
            ]
        if len(shape_values) != 3:
            return None
        return indent, lhs, input_expr, shape_values

    def _parse_rank4_mean_assign(
        current_line: str,
    ) -> Tuple[str, str, str, int, int, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.mean("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                dim_expr = part.strip()
            elif positional_index == 2:
                keepdim_expr = part.strip()
            positional_index += 1
        if (
            input_expr is None
            or dim_expr is None
            or keepdim_expr is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
            or re.fullmatch(r"True|False", keepdim_expr) is None
        ):
            return None
        dim_match = re.fullmatch(r"[\[\(]\s*(?P<dim0>-?\d+)\s*,\s*(?P<dim1>-?\d+)\s*[\]\)]", dim_expr)
        if dim_match is None:
            return None
        return indent, lhs, input_expr, int(dim_match.group("dim0")), int(dim_match.group("dim1")), keepdim_expr
    def _parse_reduce_sum_assign(
        current_line: str,
    ) -> Tuple[str, str, str, int, str] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "_reduce_sum("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        axes_expr: str | None = None
        keepdims_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key in {"axes", "axis", "dim"}:
                    axes_expr = value
                elif key == "keepdims":
                    keepdims_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                axes_expr = part.strip()
            elif positional_index == 2:
                keepdims_expr = part.strip()
            positional_index += 1
        if (
            input_expr is None
            or axes_expr is None
            or keepdims_expr is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
            or re.fullmatch(r"True|False", keepdims_expr) is None
        ):
            return None
        axis_match = re.fullmatch(
            r"_normalize_axes\(\s*[\[\(]\s*(?P<axis>-?\d+)\s*,?\s*[\]\)]\s*,\s*[A-Za-z0-9_]+\.ndim\s*\)",
            axes_expr,
        )
        if axis_match is None:
            return None
        return indent, lhs, input_expr, int(axis_match.group("axis")), keepdims_expr
    sub_from_one_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.sub\(1\.0, (?P<input>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    rank3_resize_input_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<src>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    rank3_matmul_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.matmul\((?P<x>[A-Za-z0-9_]+), (?P<y>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    rank4_singleton_reshape_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<src>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), 1\]\)$"
    )
    rank4_singleton_matmul_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.matmul\((?P<x>[A-Za-z0-9_]+), (?P<y>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), 1\]\)$"
    )
    apply_resize_nhwc_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_resize\((?:input=)?(?P<input>[A-Za-z0-9_]+), (?:size=)?[\[\(](?P<out_h>\d+), (?P<out_w>\d+)[\]\)], method='(?P<method>[^']+)', target_shape=[\[\(](?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)[\]\)], align_corners=(?P<align>True|False), half_pixel_centers=(?P<hpc>True|False), channel_last=True\)$"
    )
    apply_resize_cf_bad_target_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_resize\((?:input=)?(?P<input>[A-Za-z0-9_]+), (?:size=)?[\[\(](?P<out_h>\d+), (?P<out_w>\d+)[\]\)], method='(?P<method>[^']+)', target_shape=[\[\(](?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)[\]\)](?P<rest>.*), channel_last=False\)$"
    )
    apply_resize_cf_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_resize\((?:input=)?(?P<input>[A-Za-z0-9_]+), (?:size=)?[\[\(](?P<out_h>\d+), (?P<out_w>\d+)[\]\)], method='(?P<method>[^']+)', target_shape=[\[\(](?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)[\]\)], align_corners=(?P<align>True|False), half_pixel_centers=(?P<hpc>True|False), channel_last=False\)$"
    )
    changed = False
    cf_pad_aliases: set[str] = set()
    cf_aliases: set[str] = set()
    forced_cf_aliases: set[str] = set()
    singleton_cf_seeds: set[str] = set()
    cf_materialized_alias_sources: Dict[str, str] = {}
    generic_alias_sources: Dict[str, str] = {}
    conv_block_out_channels: Dict[str, int] = {}
    module_output_producers: Dict[str, str] = {}

    conv_block_decl_re = re.compile(r"^\s*self\.(?P<module>[A-Za-z0-9_]+) = _Conv2dBlock\($")
    in_channels_re = re.compile(r"^\s*in_channels=(?P<channels>\d+),$")
    out_channels_re = re.compile(r"^\s*out_channels=(?P<channels>\d+),$")
    module_output_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>[A-Za-z0-9_]+)\("
    )
    cf_name_token_re = re.compile(r"(?:^|_)cf(?:_|$)")
    cf_out_token_re = re.compile(r"(?:^|_)out_cf(?:_|$)")
    cf_bn_const_expr_re = re.compile(
        r"torch\.(?P<op>mul|add)\((?P<input>[A-Za-z0-9_]+), self\.(?P<const_attr>[A-Za-z0-9_]+)\)"
    )
    cf_permute_source_re = re.compile(
        r"(?P<src>[A-Za-z0-9_]+)\.permute\(0, 2, 3, 1\)\.contiguous\(\)"
    )
    def _parse_cf_nhwc_materialize_assign(
        current_line: str,
    ) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        shape = _parse_rank4_shape_literal(shape_expr)
        if shape is None:
            return None

        def _parse_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
            stripped = str(expr).strip()
            if stripped.endswith(".contiguous()"):
                stripped = stripped[: -len(".contiguous()")].strip()

            def _parse_permute_like_args(args_expr: str) -> str | None:
                parts = _split_top_level_csv_exprs(args_expr)
                input_name: str | None = None
                perm_expr: str | None = None
                if len(parts) == 2 and all(
                    re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
                ):
                    input_name, perm_expr = parts[0].strip(), parts[1].strip()
                else:
                    kwargs: Dict[str, str] = {}
                    for part in parts:
                        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                            continue
                        key, value = part.split("=", 1)
                        kwargs[key.strip()] = value.strip()
                    input_name = kwargs.get("input", kwargs.get("x"))
                    perm_expr = kwargs.get("perm", kwargs.get("dims"))
                if input_name is None or perm_expr is None or re.fullmatch(r"[A-Za-z0-9_]+", input_name) is None:
                    return None
                if _normalize_permute_dims_expr(perm_expr) != "0,2,3,1":
                    return None
                return input_name

            for prefix in ("_torch_permute(", "torch.permute("):
                if stripped.startswith(prefix) and stripped.endswith(")"):
                    return _parse_permute_like_args(stripped[len(prefix) : -1])
            method_match = re.fullmatch(r"(?P<input>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)", stripped)
            if method_match is None:
                return None
            if _normalize_permute_dims_expr(str(method_match.group("dims"))) != "0,2,3,1":
                return None
            return str(method_match.group("input"))

        source = _parse_nchw_to_nhwc_bridge_source(input_expr)
        if source is None:
            return None
        return indent, lhs, source, list(shape)
    for index, line in enumerate(lines):
        conv_block_decl_match = conv_block_decl_re.match(line)
        if conv_block_decl_match is not None:
            module_name = str(conv_block_decl_match.group("module"))
            for lookahead in range(index + 1, min(len(lines), index + 12)):
                out_channels_match = out_channels_re.match(lines[lookahead])
                if out_channels_match is not None:
                    conv_block_out_channels[module_name] = int(out_channels_match.group("channels"))
                    break
        module_output_assign_match = module_output_assign_re.match(line)
        if module_output_assign_match is not None:
            module_output_producers[str(module_output_assign_match.group("lhs"))] = str(
                module_output_assign_match.group("module")
            )

    def _is_known_cf_name(name: str, singleton_names: set[str]) -> bool:
        cache_key = (
            str(name),
            len(singleton_names),
            len(cf_aliases),
            len(forced_cf_aliases),
            len(cf_materialized_alias_sources),
            len(generic_alias_sources),
        )
        cached = known_cf_name_cache.get(cache_key, None)
        if cached is not None:
            return cached
        resolved_name = name
        visited_aliases: set[str] = set()
        while resolved_name not in visited_aliases:
            visited_aliases.add(resolved_name)
            next_name = (
                cf_materialized_alias_sources.get(resolved_name)
                or generic_alias_sources.get(resolved_name)
            )
            if next_name is None:
                break
            resolved_name = next_name
        if resolved_name in forced_cf_aliases:
            return True
        resolved_tensor = (
            model_ir.tensors.get(_resolve_model_ir_tensor_name(resolved_name), None)
            if model_ir is not None
            else None
        )
        if resolved_tensor is not None:
            resolved_layout = normalize_logical_layout(resolved_tensor.logical_layout)
            if is_channel_first_logical_layout(resolved_layout):
                known_cf_name_cache[cache_key] = True
                return True
            if is_channel_last_logical_layout(resolved_layout):
                known_cf_name_cache[cache_key] = False
                return False
        cf_name_token_match = cf_name_token_re.search(resolved_name) is not None
        cf_out_token_match = cf_out_token_re.search(resolved_name) is not None
        result = (
            resolved_name in cf_aliases
            or resolved_name in singleton_names
            or _model_ir_is_channel_first(resolved_name)
            or resolved_name.endswith("_cf")
            or resolved_name.endswith("_out_cf")
            or cf_name_token_match
            or cf_out_token_match
        )
        known_cf_name_cache[cache_key] = result
        return result

    def _is_name_available_in_function(name: str, line_index: int) -> bool:
        function_start = -1
        for candidate in range(line_index, -1, -1):
            if lines[candidate].startswith("    def "):
                function_start = candidate
                break
        if function_start < 0:
            return False
        if re.search(rf"\b{re.escape(name)}\b", lines[function_start]) is not None:
            return True
        assign_re = re.compile(rf"^\s*{re.escape(name)}\s*=")
        for candidate in range(function_start + 1, line_index):
            if assign_re.match(lines[candidate]) is not None:
                return True
        return False

    def _declares_channel_last_name(name: str) -> bool:
        if name in forced_cf_aliases:
            return False
        resolved_name = _resolve_model_ir_tensor_name(name)
        tensor = (
            model_ir.tensors.get(resolved_name, None)
            if model_ir is not None
            else None
        )
        if tensor is not None:
            layout = normalize_logical_layout(tensor.logical_layout)
            if is_channel_last_logical_layout(layout):
                return True
            if is_channel_first_logical_layout(layout):
                return False
        return "_nhwc" in name

    def _infer_cf_channel_count(name: str) -> int | None:
        producer_module = module_output_producers.get(name, None)
        if producer_module is not None:
            out_channels = conv_block_out_channels.get(producer_module, None)
            if out_channels is not None:
                return int(out_channels)
        if model_ir is None:
            return None
        exact_shape = _tensor_exact_static_shape_list_for_model_ir(
            model_ir=model_ir,
            tensor_name=name,
        )
        if exact_shape is None or len(exact_shape) != 4:
            return None
        if name.endswith("_nhwc_cf"):
            return int(exact_shape[3])
        if _is_known_cf_name(name, singleton_cf_seeds):
            return int(exact_shape[1])
        return None

    def _resolve_codegen_alias_source(name: str) -> str:
        resolved_name = str(name)
        visited_aliases: set[str] = set()
        while resolved_name not in visited_aliases:
            visited_aliases.add(resolved_name)
            next_name = (
                cf_materialized_alias_sources.get(resolved_name)
                or generic_alias_sources.get(resolved_name)
            )
            if next_name is None:
                break
            resolved_name = next_name
        return resolved_name

    def _function_end_index(line_index: int) -> int:
        if line_index < 0 or line_index >= len(function_end_by_index):
            return len(lines)
        return function_end_by_index[line_index]

    def _has_rank4_reshape_consumer(name: str, line_index: int) -> bool:
        function_end = _function_end_index(line_index)
        for future_line in lines[line_index + 1 : function_end]:
            reshape_assign = _parse_rank4_reshape_consumer_assign(future_line)
            if reshape_assign is not None and str(reshape_assign[2]) == name:
                return True
        return False

    def _parse_raw_local_response_norm_input(current_line: str) -> str | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        _, _, rhs = assign
        prefix = "F.local_response_norm("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
            input_expr = parts[0].strip()
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            if key.strip() == "input":
                input_expr = value.strip()
        if input_expr is None or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
            return None
        return input_expr

    def _parse_raw_apply_softmax_assign(
        current_line: str,
    ) -> Tuple[str, str, str, int, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        softmax_args = _parse_apply_softmax_input_axis_and_shape(rhs)
        if softmax_args is None:
            return None
        input_expr, axis_value, rank4_shape = softmax_args
        if rank4_shape is None or re.fullmatch(r"[A-Za-z0-9_]+", input_expr.strip()) is None:
            return None
        beta_expr: str | None = None
        stripped = rhs.strip()
        prefix = "_apply_softmax("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            if key.strip() == "beta":
                beta_expr = value.strip()
                break
        if beta_expr is None or re.fullmatch(r"[-+0-9.eE]+", beta_expr) is None:
            return None
        return (
            indent,
            lhs,
            input_expr.strip(),
            axis_value,
            beta_expr,
            [int(v) for v in list(rank4_shape)],
        )

    def _parse_raw_sub_from_one_align_assign(
        current_line: str,
    ) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        binary_args = _parse_binary_sub_args(input_expr)
        shape = _parse_rank4_shape_literal(shape_expr)
        if binary_args is None or shape is None or len(shape) != 4:
            return None
        a_expr, b_expr = binary_args
        if str(a_expr).strip() != "1.0" or re.fullmatch(r"[A-Za-z0-9_]+", str(b_expr).strip()) is None:
            return None
        return indent, lhs, str(b_expr).strip(), [int(v) for v in shape]

    def _parse_raw_rank4_singleton_reshape_assign(
        current_line: str,
    ) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(current_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        call_match = re.fullmatch(r"torch\.reshape\((?P<args>.+)\)", rhs.strip())
        if call_match is None:
            return None
        parts = _split_top_level_csv_exprs(str(call_match.group("args")))
        input_expr: str | None = None
        shape_expr: str | None = None
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_expr, shape_expr = parts[0].strip(), parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_expr = kwargs.get("input")
            shape_expr = kwargs.get("shape")
        shape = _parse_rank4_shape_literal(shape_expr) if shape_expr is not None else None
        if (
            input_expr is None
            or shape is None
            or len(shape) != 4
            or int(shape[3]) != 1
            or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
        ):
            return None
        return indent, lhs, input_expr, [int(v) for v in shape]

    def _has_nearby_local_response_norm_consumer(name: str, line_index: int) -> bool:
        function_end = _function_end_index(line_index)
        for future_index in range(line_index + 1, min(function_end, line_index + 4)):
            future_line = lines[future_index]
            if future_line.strip() == "":
                continue
            lrn_assign = _parse_raw_local_response_norm_input(future_line)
            return lrn_assign is not None and str(lrn_assign) == name
        return False

    def _has_nearby_channel_last_spatial_consumer(
        name: str,
        line_index: int,
        visited_names: Set[str] | None = None,
    ) -> bool:
        if visited_names is None:
            visited_names = set()
        if name in visited_names:
            return False
        visited_names.add(name)
        function_end = _function_end_index(line_index)
        for future_index in range(line_index + 1, min(function_end, line_index + 10)):
            future_line = lines[future_index]
            if future_line.strip() == "":
                continue
            future_slice_assign = _parse_channel_last_gather_slice_assign(future_line)
            if future_slice_assign is not None and str(future_slice_assign[1]) == name:
                return True
            alias_assign = _parse_simple_assignment_line(future_line)
            if (
                alias_assign is not None
                and _is_simple_identifier_expr(alias_assign[2])
                and _strip_outer_parentheses(str(alias_assign[2]).strip()) == name
            ):
                alias_name = str(alias_assign[1])
                if alias_name.startswith("_space_to_depth_x_") or alias_name.startswith("_depth_to_space_x_"):
                    return True
            future_pool_assign = _parse_apply_pool2d_assign_with_shape(lines[future_index])
            if (
                future_pool_assign is not None
                and str(future_pool_assign[2]) == name
                and bool(future_pool_assign[6])
            ):
                return True
        return False

    def _find_stage_boundary_cat_consumer(name: str, line_index: int) -> bool:
        stage_return_index = None
        for lookahead_index in range(line_index + 1, min(line_index + 8, len(lines))):
            if lines[lookahead_index].strip() == "":
                continue
            stage_return_value = _parse_simple_return_identifier(lines[lookahead_index])
            if stage_return_value is not None and str(stage_return_value) == name:
                stage_return_index = lookahead_index
            break
        if stage_return_index is None:
            return None
        stage_signature_index = None
        for lookahead_index in range(stage_return_index + 1, min(stage_return_index + 6, len(lines))):
            if lines[lookahead_index].startswith("    def "):
                stage_signature_index = lookahead_index
                break
        if (
            stage_signature_index is None
            or re.search(rf"\b{re.escape(name)}\b", lines[stage_signature_index]) is None
        ):
            return None
        for lookahead_index in range(stage_signature_index + 1, min(stage_signature_index + 8, len(lines))):
            if lines[lookahead_index].startswith("    def "):
                break
            if lines[lookahead_index].strip() == "":
                continue
            stage_cat_assign = _parse_simple_assignment_line(lines[lookahead_index])
            parsed_stage_cat = (
                _parse_torch_cat_inputs_and_dim(stage_cat_assign[2])
                if stage_cat_assign is not None
                else None
            )
            if (
                parsed_stage_cat is not None
                and parsed_stage_cat[1] == 1
                and name in {
                    input_name.strip()
                    for input_name in parsed_stage_cat[0]
                    if input_name.strip()
                }
            ):
                return True
        return False

    def _find_same_function_cat_consumer(name: str, line_index: int) -> bool:
        function_end = _function_end_index(line_index)
        for lookahead_index in range(line_index + 1, min(line_index + 5, function_end)):
            if lines[lookahead_index].strip() == "":
                continue
            same_function_cat_assign = _parse_simple_assignment_line(lines[lookahead_index])
            parsed_same_function_cat = (
                _parse_torch_cat_inputs_and_dim(same_function_cat_assign[2])
                if same_function_cat_assign is not None
                else None
            )
            if (
                parsed_same_function_cat is not None
                and parsed_same_function_cat[1] == 1
                and name in {
                    input_name.strip()
                    for input_name in parsed_same_function_cat[0]
                    if input_name.strip()
                }
            ):
                return True
        return False

    def _find_recent_rank4_shape(name: str, line_index: int) -> Optional[List[int]]:
        cache_key = (str(name), int(line_index))
        cached_shape = recent_rank4_shape_cache.get(cache_key, None)
        if cache_key in recent_rank4_shape_cache:
            return None if cached_shape is None else list(cached_shape)
        function_start = function_start_by_index[line_index] if 0 <= line_index < len(function_start_by_index) else -1
        name_cache_key = (str(name), function_start)
        cached_name_shape = recent_rank4_shape_by_name_cache.get(name_cache_key, None)
        if cached_name_shape is not None:
            resolved_at, resolved_shape = cached_name_shape
            if resolved_at < 0 or resolved_at < line_index:
                recent_rank4_shape_cache[cache_key] = None if resolved_shape is None else list(resolved_shape)
                return None if resolved_shape is None else list(resolved_shape)
        exact_shape = _model_ir_exact_shape(name)
        if exact_shape is not None and len(exact_shape) == 4:
            resolved_exact_shape = [int(v) for v in exact_shape]
            recent_rank4_shape_by_name_cache[name_cache_key] = (-1, resolved_exact_shape)
            recent_rank4_shape_cache[cache_key] = resolved_exact_shape
            return list(resolved_exact_shape)
        resolved_name = name
        visited_names: set[str] = set()
        for candidate in range(line_index - 1, function_start, -1):
            assign_line = lines[candidate]
            generic_alias_assign = _parse_simple_assignment_line(assign_line)
            if (
                generic_alias_assign is not None
                and str(generic_alias_assign[1]) == resolved_name
                and _is_simple_identifier_expr(generic_alias_assign[2])
            ):
                rhs = _strip_outer_parentheses(str(generic_alias_assign[2]).strip())
                if rhs in visited_names:
                    break
                visited_names.add(rhs)
                resolved_name = rhs
                exact_shape = _model_ir_exact_shape(resolved_name)
                if exact_shape is not None and len(exact_shape) == 4:
                    resolved_exact_shape = [int(v) for v in exact_shape]
                    recent_rank4_shape_by_name_cache[name_cache_key] = (-1, resolved_exact_shape)
                    recent_rank4_shape_cache[cache_key] = resolved_exact_shape
                    return list(resolved_exact_shape)
                continue
            aligned_match = aligned_nhwc_rank4_re.match(assign_line)
            if aligned_match is not None and str(aligned_match.group("lhs")) == resolved_name:
                resolved_shape = [
                    int(aligned_match.group("n")),
                    int(aligned_match.group("h")),
                    int(aligned_match.group("w")),
                    int(aligned_match.group("c")),
                ]
                recent_rank4_shape_by_name_cache[name_cache_key] = (candidate, resolved_shape)
                recent_rank4_shape_cache[cache_key] = resolved_shape
                return list(resolved_shape)
            resize_match = apply_resize_cf_re.match(assign_line)
            if resize_match is not None and str(resize_match.group("lhs")) == resolved_name:
                resolved_shape = [
                    int(resize_match.group("n")),
                    int(resize_match.group("c")),
                    int(resize_match.group("out_h")),
                    int(resize_match.group("out_w")),
                ]
                recent_rank4_shape_by_name_cache[name_cache_key] = (candidate, resolved_shape)
                recent_rank4_shape_cache[cache_key] = resolved_shape
                return list(resolved_shape)
            rank4_singleton_reshape_match = rank4_singleton_reshape_re.match(assign_line)
            if (
                rank4_singleton_reshape_match is not None
                and str(rank4_singleton_reshape_match.group("lhs")) == resolved_name
            ):
                resolved_shape = [
                    int(rank4_singleton_reshape_match.group("n")),
                    1,
                    int(rank4_singleton_reshape_match.group("h")),
                    int(rank4_singleton_reshape_match.group("w")),
                ]
                recent_rank4_shape_by_name_cache[name_cache_key] = (candidate, resolved_shape)
                recent_rank4_shape_cache[cache_key] = resolved_shape
                return list(resolved_shape)
            rank4_singleton_matmul_match = rank4_singleton_matmul_re.match(assign_line)
            if (
                rank4_singleton_matmul_match is not None
                and str(rank4_singleton_matmul_match.group("lhs")) == resolved_name
            ):
                resolved_shape = [
                    int(rank4_singleton_matmul_match.group("n")),
                    1,
                    int(rank4_singleton_matmul_match.group("h")),
                    int(rank4_singleton_matmul_match.group("w")),
                ]
                recent_rank4_shape_by_name_cache[name_cache_key] = (candidate, resolved_shape)
                recent_rank4_shape_cache[cache_key] = resolved_shape
                return list(resolved_shape)
        recent_rank4_shape_by_name_cache[name_cache_key] = (line_index, None)
        recent_rank4_shape_cache[cache_key] = None
        return None

    def _find_recent_singleton_cf_channel_count(line_index: int) -> int | None:
        function_start = function_start_by_index[line_index] if 0 <= line_index < len(function_start_by_index) else -1
        for candidate in range(line_index - 1, function_start, -1):
            match = recent_cf_singleton_shape_re.search(lines[candidate])
            if match is None:
                continue
            channel_count = int(match.group("c"))
            if channel_count > 1:
                return channel_count
        return None

    buffer_specs: Dict[str, Tuple[int, List[int], str, bool]] = {}
    const_temp_assignments: Dict[str, Tuple[int, str, str, str]] = {}
    transposed_const_alias_specs: Dict[str, Tuple[str, List[int], str]] = {}
    inline_const_buffer_specs: Dict[str, Tuple[str, str]] = {}
    inline_const_buffer_cache: Dict[Tuple[str, str], str] = {}
    raw_pidnet_const_alias_sources: Dict[str, str] = {}
    raw_pidnet_const_pair_alias_sources: Dict[str, Tuple[str, str]] = {}

    def _buffer_channel_count(const_attr: str) -> int | None:
        buffer_spec = buffer_specs.get(str(const_attr), None)
        if buffer_spec is None:
            return None
        _, shape_values, _, _ = buffer_spec
        non_singleton_dims = [int(value) for value in shape_values if int(value) != 1]
        if len(non_singleton_dims) == 1:
            return int(non_singleton_dims[0])
        if len(shape_values) == 4:
            if int(shape_values[1]) != 1 and int(shape_values[3]) == 1:
                return int(shape_values[1])
            if int(shape_values[3]) != 1 and int(shape_values[1]) == 1:
                return int(shape_values[3])
        return None

    def _resolve_raw_pidnet_const_expr(const_expr: str) -> Tuple[str, str] | None:
        resolved_expr = str(const_expr)
        visited_exprs: set[str] = set()
        while resolved_expr not in visited_exprs:
            visited_exprs.add(resolved_expr)
            aliased_attr = raw_pidnet_const_alias_sources.get(resolved_expr, None)
            if aliased_attr is None:
                break
            resolved_expr = f"self.{aliased_attr}"
        if not resolved_expr.startswith("self."):
            return None
        return str(const_expr), resolved_expr[len("self.") :]

    def _parse_raw_pidnet_align_binary_out_assign(
        line: str,
        op: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        binary_match = re.fullmatch(rf"torch\.{op}\((?P<args>.+)\)", input_expr.strip())
        if binary_match is None:
            return None
        binary_args = (
            _parse_binary_mul_args(str(binary_match.group("args")))
            if op == "mul"
            else _parse_binary_add_args(str(binary_match.group("args")))
        )
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if binary_args is None or rank4_shape is None:
            return None
        return indent, lhs, str(binary_args[0]), str(binary_args[1]), list(rank4_shape)

    def _parse_raw_pidnet_reshape_binary_out_assign(
        line: str,
        op: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.reshape("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        shape_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "shape":
                    shape_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                shape_expr = part.strip()
            positional_index += 1
        if input_expr is None or shape_expr is None:
            return None
        binary_match = re.fullmatch(rf"torch\.{op}\((?P<args>.+)\)", input_expr.strip())
        if binary_match is None:
            return None
        binary_args = (
            _parse_binary_mul_args(str(binary_match.group("args")))
            if op == "mul"
            else _parse_binary_add_args(str(binary_match.group("args")))
        )
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if binary_args is None or rank4_shape is None:
            return None
        return indent, lhs, str(binary_args[0]), str(binary_args[1]), list(rank4_shape)

    def _parse_raw_pidnet_scale4_mul_reshape_assign(
        line: str,
    ) -> Tuple[str, str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.reshape("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        shape_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "shape":
                    shape_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                shape_expr = part.strip()
            positional_index += 1
        rank4_shape = _parse_rank4_shape_literal(str(shape_expr))
        if input_expr is None or rank4_shape is None:
            return None
        if not (
            int(rank4_shape[0]) == 1
            and int(rank4_shape[1]) == 1
            and int(rank4_shape[3]) == 1
            and int(rank4_shape[2]) > 1
        ):
            return None
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", str(input_expr).strip())
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        input_a = str(mul_args[0]).strip()
        input_b = str(mul_args[1]).strip()
        resolved_a = _resolve_raw_pidnet_const_expr(input_a)
        resolved_b = _resolve_raw_pidnet_const_expr(input_b)
        if resolved_a is not None and resolved_b is None:
            return indent, lhs, input_b, resolved_a[0], int(rank4_shape[2])
        if resolved_b is not None and resolved_a is None:
            return indent, lhs, input_a, resolved_b[0], int(rank4_shape[2])
        return None

    def _parse_raw_pidnet_scale4_mul_reshape_variant_assign(
        line: str,
    ) -> Tuple[str, str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.reshape("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        shape_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "shape":
                    shape_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                shape_expr = part.strip()
            positional_index += 1
        rank4_shape = _parse_rank4_shape_literal(str(shape_expr))
        if input_expr is None or rank4_shape is None:
            return None
        if not (
            int(rank4_shape[0]) == 1
            and int(rank4_shape[1]) == 1
            and int(rank4_shape[2]) == 1
            and int(rank4_shape[3]) > 1
        ):
            return None
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", str(input_expr).strip())
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        input_a = str(mul_args[0]).strip()
        input_b = str(mul_args[1]).strip()

        def _parse_inner_const_reshape(expr: str) -> str | None:
            inner_stripped = str(expr).strip()
            inner_prefix = "torch.reshape("
            if not inner_stripped.startswith(inner_prefix) or not inner_stripped.endswith(")"):
                return None
            inner_parts = _split_top_level_csv_exprs(inner_stripped[len(inner_prefix) : -1])
            inner_input_expr: str | None = None
            for inner_index, inner_part in enumerate(inner_parts):
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", inner_part) is not None:
                    key, value = inner_part.split("=", 1)
                    if key.strip() == "input":
                        inner_input_expr = value.strip()
                    continue
                if inner_index == 0:
                    inner_input_expr = inner_part.strip()
            if inner_input_expr is None:
                return None
            resolved_inner = _resolve_raw_pidnet_const_expr(inner_input_expr)
            if resolved_inner is None:
                return None
            return resolved_inner[0]

        const_expr_a = _parse_inner_const_reshape(input_a)
        const_expr_b = _parse_inner_const_reshape(input_b)
        if const_expr_a is not None and const_expr_b is None:
            return indent, lhs, input_b, const_expr_a, int(rank4_shape[3])
        if const_expr_b is not None and const_expr_a is None:
            return indent, lhs, input_a, const_expr_b, int(rank4_shape[3])
        return None

    def _parse_raw_pidnet_scale4_mul_reshape_variant_reversed_assign(
        line: str,
    ) -> Tuple[str, str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.reshape("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        shape_expr: str | None = None
        positional_index = 0
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "shape":
                    shape_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                shape_expr = part.strip()
            positional_index += 1
        rank4_shape = _parse_rank4_shape_literal(str(shape_expr))
        if input_expr is None or rank4_shape is None:
            return None
        if not (
            int(rank4_shape[0]) == 1
            and int(rank4_shape[1]) == 1
            and int(rank4_shape[2]) == 1
            and int(rank4_shape[3]) > 1
        ):
            return None
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", str(input_expr).strip())
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        input_a = str(mul_args[0]).strip()
        input_b = str(mul_args[1]).strip()

        def _parse_inner_const_reshape(expr: str) -> str | None:
            inner_stripped = str(expr).strip()
            inner_prefix = "torch.reshape("
            if not inner_stripped.startswith(inner_prefix) or not inner_stripped.endswith(")"):
                return None
            inner_parts = _split_top_level_csv_exprs(inner_stripped[len(inner_prefix) : -1])
            inner_input_expr: str | None = None
            for inner_index, inner_part in enumerate(inner_parts):
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", inner_part) is not None:
                    key, value = inner_part.split("=", 1)
                    if key.strip() == "input":
                        inner_input_expr = value.strip()
                    continue
                if inner_index == 0:
                    inner_input_expr = inner_part.strip()
            if inner_input_expr is None:
                return None
            resolved_inner = _resolve_raw_pidnet_const_expr(inner_input_expr)
            if resolved_inner is None:
                return None
            return resolved_inner[0]

        const_expr_a = _parse_inner_const_reshape(input_a)
        const_expr_b = _parse_inner_const_reshape(input_b)
        if const_expr_a is not None and const_expr_b is None:
            return indent, lhs, input_b, const_expr_a, int(rank4_shape[3])
        if const_expr_b is not None and const_expr_a is None:
            return indent, lhs, input_a, const_expr_b, int(rank4_shape[3])
        return None

    def _parse_raw_pidnet_scale4_direct_mul_assign(
        line: str,
    ) -> Tuple[str, str, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", rhs.strip())
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        input_a = str(mul_args[0]).strip()
        input_b = str(mul_args[1]).strip()
        resolved_a = _resolve_raw_pidnet_const_expr(input_a)
        resolved_b = _resolve_raw_pidnet_const_expr(input_b)
        if resolved_a is not None and resolved_b is None:
            return indent, lhs, input_b, resolved_a[0]
        if resolved_b is not None and resolved_a is None:
            return indent, lhs, input_a, resolved_b[0]
        return None

    def _parse_raw_pidnet_scale3_mul_align_assign(
        line: str,
    ) -> Tuple[str, str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, shape_expr = align_parts
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if rank4_shape is None:
            return None
        if not (
            int(rank4_shape[0]) == 1
            and int(rank4_shape[2]) == 1
            and int(rank4_shape[1]) > 1
            and int(rank4_shape[3]) > 1
        ):
            return None
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", input_expr.strip())
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        input_a = str(mul_args[0]).strip()
        input_b = str(mul_args[1]).strip()
        resolved_a = _resolve_raw_pidnet_const_expr(input_a)
        resolved_b = _resolve_raw_pidnet_const_expr(input_b)
        if resolved_a is not None and resolved_b is None:
            return indent, lhs, input_b, resolved_a[0], int(rank4_shape[1])
        if resolved_b is not None and resolved_a is None:
            return indent, lhs, input_a, resolved_b[0], int(rank4_shape[1])
        return None

    def _parse_raw_pidnet_scale3_add_anchor_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, int] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_a: str | None = None
        input_b: str | None = None
        shape_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
            shape_expr = parts[2].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("a"))
            input_b = kwargs.get("other", kwargs.get("b"))
            shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
        if input_a is None or input_b is None or shape_expr is None:
            return None
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if rank4_shape is None:
            return None
        if not (
            int(rank4_shape[0]) == 1
            and int(rank4_shape[2]) == 1
            and int(rank4_shape[1]) > 1
            and int(rank4_shape[3]) > 1
        ):
            return None
        resolved_a = _resolve_raw_pidnet_const_expr(input_a)
        resolved_b = _resolve_raw_pidnet_const_expr(input_b)
        if resolved_a is not None and resolved_b is None:
            return (
                str(assign_match.group("indent")),
                str(assign_match.group("lhs0")),
                str(assign_match.group("lhs1")),
                input_b,
                resolved_a[0],
                int(rank4_shape[1]),
            )
        if resolved_b is not None and resolved_a is None:
            return (
                str(assign_match.group("indent")),
                str(assign_match.group("lhs0")),
                str(assign_match.group("lhs1")),
                input_a,
                resolved_b[0],
                int(rank4_shape[1]),
            )
        return None

    def _parse_raw_pidnet_scale3_const_anchor_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, int] | None:
        parsed = _parse_raw_pidnet_scale3_add_anchor_assign(line)
        if parsed is None:
            return None
        indent, lhs0, lhs1, input_name, const_expr, channel_count = parsed
        if _resolve_raw_pidnet_const_expr(const_expr) is None and not str(const_expr).startswith("self."):
            return None
        return indent, lhs0, lhs1, input_name, const_expr, channel_count

    def _parse_raw_pidnet_scale4_add_anchor_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, int] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_a: str | None = None
        input_b: str | None = None
        shape_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
            shape_expr = parts[2].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("a"))
            input_b = kwargs.get("other", kwargs.get("b"))
            shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
        if input_a is None or input_b is None or shape_expr is None:
            return None
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if rank4_shape is None:
            return None
        if not (
            int(rank4_shape[0]) == 1
            and int(rank4_shape[1]) == 1
            and int(rank4_shape[2]) == 1
            and int(rank4_shape[3]) > 1
        ):
            return None
        resolved_a = _resolve_raw_pidnet_const_expr(input_a)
        resolved_b = _resolve_raw_pidnet_const_expr(input_b)
        if resolved_a is not None and resolved_b is None:
            return (
                str(assign_match.group("indent")),
                str(assign_match.group("lhs0")),
                str(assign_match.group("lhs1")),
                input_b,
                resolved_a[0],
                int(rank4_shape[3]),
            )
        if resolved_b is not None and resolved_a is None:
            return (
                str(assign_match.group("indent")),
                str(assign_match.group("lhs0")),
                str(assign_match.group("lhs1")),
                input_a,
                resolved_b[0],
                int(rank4_shape[3]),
            )
        return None

    inside_nms_method = False
    for index, line in enumerate(lines):
        stripped_line = line.strip()
        if line.startswith("    def "):
            inside_nms_method = stripped_line.startswith("def _run_nms_")
        binary_same_permute_match = binary_same_permute_cf_re.match(line)
        next_binary_same_permute_match = (
            binary_same_permute_cf_re.match(lines[index + 1]) if index + 1 < len(lines) else None
        )
        next_cat_assign = (
            _parse_simple_assignment_line(lines[index + 2]) if index + 2 < len(lines) else None
        )
        parsed_next_cat = (
            _parse_torch_cat_inputs_and_dim(next_cat_assign[2])
            if next_cat_assign is not None
            else None
        )
        if (
            binary_same_permute_match is not None
            and next_binary_same_permute_match is not None
            and parsed_next_cat is not None
            and str(binary_same_permute_match.group("a")) == str(next_binary_same_permute_match.group("a"))
            and str(binary_same_permute_match.group("b")) == str(next_binary_same_permute_match.group("b"))
            and {str(binary_same_permute_match.group("op")), str(next_binary_same_permute_match.group("op"))}
            == {"add", "sub"}
            and parsed_next_cat[1] in {1, 2}
        ):
            cat_inputs = [
                token.strip()
                for token in parsed_next_cat[0]
                if token.strip()
            ]
            first_lhs = str(binary_same_permute_match.group("lhs"))
            second_lhs = str(next_binary_same_permute_match.group("lhs"))
            if cat_inputs == [first_lhs, second_lhs]:
                indent = str(binary_same_permute_match.group("indent"))
                a = str(binary_same_permute_match.group("a"))
                b = str(binary_same_permute_match.group("b"))
                cat_lhs = str(next_cat_assign[1] if next_cat_assign is not None else next_cat_match.group("lhs"))
                cat_axis = parsed_next_cat[1] if parsed_next_cat is not None else int(next_cat_match.group("axis"))
                lines[index] = f"{indent}{first_lhs} = torch.{binary_same_permute_match.group('op')}({a}, {b})"
                lines[index + 1] = (
                    f"{next_binary_same_permute_match.group('indent')}{second_lhs} = "
                    f"torch.{next_binary_same_permute_match.group('op')}({a}, {b})"
                )
                if cat_axis == 1:
                    lines[index + 2] = (
                        f"{(next_cat_assign[0] if next_cat_assign is not None else next_cat_match.group('indent'))}{cat_lhs} = "
                        f"torch.cat([{first_lhs}, {second_lhs}], dim=1).permute(0, 2, 1).contiguous()"
                    )
                else:
                    lines[index + 2] = (
                        f"{(next_cat_assign[0] if next_cat_assign is not None else next_cat_match.group('indent'))}{cat_lhs} = "
                        f"torch.cat([{first_lhs}, {second_lhs}], dim=2)"
                    )
                cf_aliases.add(cat_lhs)
                changed = True
                line = lines[index]
        minimum_scalar_match = minimum_scalar_tensor_re.match(line)
        if minimum_scalar_match is not None:
            lines[index] = (
                f"{minimum_scalar_match.group('indent')}{minimum_scalar_match.group('lhs')} = "
                f"{minimum_scalar_match.group('prefix')}torch.clamp({minimum_scalar_match.group('tensor')}, "
                f"max={minimum_scalar_match.group('value')}){minimum_scalar_match.group('suffix')}"
            )
            changed = True
            line = lines[index]
        maximum_scalar_match = maximum_scalar_tensor_re.match(line)
        if maximum_scalar_match is not None:
            lines[index] = (
                f"{maximum_scalar_match.group('indent')}{maximum_scalar_match.group('lhs')} = "
                f"{maximum_scalar_match.group('prefix')}torch.clamp({maximum_scalar_match.group('tensor')}, "
                f"min={maximum_scalar_match.group('value')}){maximum_scalar_match.group('suffix')}"
            )
            changed = True
            line = lines[index]
        scalar_first_binary_match = scalar_first_binary_re.match(line)
        if scalar_first_binary_match is not None:
            rewritten_line = (
                f"{scalar_first_binary_match.group('indent')}{scalar_first_binary_match.group('lhs')} = "
                f"torch.{scalar_first_binary_match.group('op')}("
                f"{scalar_first_binary_match.group('rhs')}, {scalar_first_binary_match.group('scalar')})"
            )
            if rewritten_line != line:
                lines[index] = rewritten_line
                changed = True
                line = rewritten_line
        scalar_first_inline_replaced = scalar_first_inline_binary_re.sub(
            r"torch.\g<op>(\g<rhs>, \g<scalar>)",
            line,
        )
        if scalar_first_inline_replaced != line:
            lines[index] = scalar_first_inline_replaced
            changed = True
            line = scalar_first_inline_replaced
        scalar_as_tensor_replaced = line
        if not inside_nms_method and "_apply_non_max_suppression_v4(" not in line:
            scalar_as_tensor_replaced = scalar_as_tensor_re.sub(r"\g<value>", line)
        if scalar_as_tensor_replaced != line:
            lines[index] = scalar_as_tensor_replaced
            changed = True
            line = scalar_as_tensor_replaced
        parsed_pidnet_spp_scale3_anchor = _parse_raw_pidnet_scale3_const_anchor_assign(line)
        if parsed_pidnet_spp_scale3_anchor is not None:
            (
                pidnet_spp_scale3_anchor_indent,
                pidnet_spp_scale3_anchor_lhs0,
                pidnet_spp_scale3_anchor_lhs1,
                pidnet_spp_scale3_anchor_input,
                pidnet_spp_scale3_anchor_const_expr,
                pidnet_spp_scale3_anchor_c,
            ) = parsed_pidnet_spp_scale3_anchor
            lines[index] = (
                f"{pidnet_spp_scale3_anchor_indent}{pidnet_spp_scale3_anchor_lhs0}, "
                f"{pidnet_spp_scale3_anchor_lhs1} = _align_binary_inputs_to_anchor("
                f"{pidnet_spp_scale3_anchor_input}, "
                f"torch.reshape({pidnet_spp_scale3_anchor_const_expr}, [1, {pidnet_spp_scale3_anchor_c}, 1, 1]), "
                f"[1, {pidnet_spp_scale3_anchor_c}, 1, 1])"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale3_mul = _parse_raw_pidnet_scale3_mul_align_assign(line)
        if parsed_pidnet_spp_scale3_mul is not None:
            pidnet_spp_scale3_mul_indent, pidnet_spp_scale3_mul_lhs, pidnet_spp_scale3_mul_input, pidnet_spp_scale3_mul_const_expr, pidnet_spp_scale3_mul_c = parsed_pidnet_spp_scale3_mul
            lines[index] = (
                f"{pidnet_spp_scale3_mul_indent}{pidnet_spp_scale3_mul_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale3_mul_input}, "
                f"{pidnet_spp_scale3_mul_const_expr}), [1, {pidnet_spp_scale3_mul_c}, 1, 1])"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale3_add = _parse_raw_pidnet_scale3_add_anchor_assign(line)
        if parsed_pidnet_spp_scale3_add is not None:
            (
                pidnet_spp_scale3_add_indent,
                pidnet_spp_scale3_add_lhs0,
                pidnet_spp_scale3_add_lhs1,
                pidnet_spp_scale3_add_input,
                pidnet_spp_scale3_add_const_expr,
                pidnet_spp_scale3_add_c,
            ) = parsed_pidnet_spp_scale3_add
            lines[index] = (
                f"{pidnet_spp_scale3_add_indent}{pidnet_spp_scale3_add_lhs0}, "
                f"{pidnet_spp_scale3_add_lhs1} = _align_binary_inputs_to_anchor("
                f"{pidnet_spp_scale3_add_input}, "
                f"torch.reshape({pidnet_spp_scale3_add_const_expr}, [1, {pidnet_spp_scale3_add_c}, 1, 1]), "
                f"[1, {pidnet_spp_scale3_add_c}, 1, 1])"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale3_mul_out = _parse_raw_pidnet_align_binary_out_assign(line, "mul")
        if (
            parsed_pidnet_spp_scale3_mul_out is not None
            and parsed_pidnet_spp_scale3_mul_out[4][0] == 1
            and parsed_pidnet_spp_scale3_mul_out[4][2] == 1
        ):
            pidnet_spp_scale3_mul_out_indent, pidnet_spp_scale3_mul_out_lhs, pidnet_spp_scale3_mul_out_a, pidnet_spp_scale3_mul_out_b, pidnet_spp_scale3_mul_out_shape = parsed_pidnet_spp_scale3_mul_out
            lines[index] = (
                f"{pidnet_spp_scale3_mul_out_indent}{pidnet_spp_scale3_mul_out_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale3_mul_out_a}, "
                f"{pidnet_spp_scale3_mul_out_b}), [1, {pidnet_spp_scale3_mul_out_shape[1]}, 1, 1])"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale4_direct_mul = _parse_raw_pidnet_scale4_direct_mul_assign(line)
        if parsed_pidnet_spp_scale4_direct_mul is not None:
            (
                pidnet_spp_scale4_mul_indent,
                pidnet_spp_scale4_mul_lhs,
                pidnet_spp_scale4_mul_input,
                pidnet_spp_scale4_mul_const_expr,
            ) = parsed_pidnet_spp_scale4_direct_mul
            const_attr_name = (
                str(pidnet_spp_scale4_mul_const_expr)[len("self.") :]
                if str(pidnet_spp_scale4_mul_const_expr).startswith("self.")
                else str(pidnet_spp_scale4_mul_const_expr)
            )
            preferred_channels = _infer_cf_channel_count(str(pidnet_spp_scale4_mul_input))
            if preferred_channels is None:
                recent_shape = _find_recent_rank4_shape(str(pidnet_spp_scale4_mul_input), index)
                if recent_shape is not None and len(recent_shape) == 4:
                    preferred_channels = int(recent_shape[1])
            if preferred_channels is None:
                preferred_channels = _buffer_channel_count(const_attr_name)
            if preferred_channels is None:
                continue
            channel_count = int(preferred_channels)
            lines[index] = (
                f"{pidnet_spp_scale4_mul_indent}{pidnet_spp_scale4_mul_lhs} = "
                f"torch.mul({pidnet_spp_scale4_mul_input}, "
                f"torch.reshape({pidnet_spp_scale4_mul_const_expr}, [1, {channel_count}, 1, 1]))"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale4_mul_reshape = _parse_raw_pidnet_scale4_mul_reshape_assign(line)
        if parsed_pidnet_spp_scale4_mul_reshape is not None:
            (
                pidnet_spp_scale4_mul_reshape_indent,
                pidnet_spp_scale4_mul_reshape_lhs,
                pidnet_spp_scale4_mul_reshape_input,
                pidnet_spp_scale4_mul_reshape_const_expr,
                pidnet_spp_scale4_mul_reshape_c,
            ) = parsed_pidnet_spp_scale4_mul_reshape
            lines[index] = (
                f"{pidnet_spp_scale4_mul_reshape_indent}{pidnet_spp_scale4_mul_reshape_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale4_mul_reshape_input}, "
                f"torch.reshape({pidnet_spp_scale4_mul_reshape_const_expr}, [1, {pidnet_spp_scale4_mul_reshape_c}, 1, 1])), "
                f"[1, {pidnet_spp_scale4_mul_reshape_c}, 1, 1])"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale4_mul_reshape_variant = _parse_raw_pidnet_scale4_mul_reshape_variant_assign(line)
        if parsed_pidnet_spp_scale4_mul_reshape_variant is not None:
            (
                pidnet_spp_scale4_mul_reshape_variant_indent,
                pidnet_spp_scale4_mul_reshape_variant_lhs,
                pidnet_spp_scale4_mul_reshape_variant_input,
                pidnet_spp_scale4_mul_reshape_variant_const_expr,
                pidnet_spp_scale4_mul_reshape_variant_c,
            ) = parsed_pidnet_spp_scale4_mul_reshape_variant
            lines[index] = (
                f"{pidnet_spp_scale4_mul_reshape_variant_indent}{pidnet_spp_scale4_mul_reshape_variant_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale4_mul_reshape_variant_input}, "
                f"torch.reshape({pidnet_spp_scale4_mul_reshape_variant_const_expr}, [1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])), "
                f"[1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])"
            )
            changed = True
            line = lines[index]
        else:
            parsed_pidnet_spp_scale4_mul_reshape_variant_reversed = (
                _parse_raw_pidnet_scale4_mul_reshape_variant_reversed_assign(line)
            )
            if parsed_pidnet_spp_scale4_mul_reshape_variant_reversed is not None:
                (
                    pidnet_spp_scale4_mul_reshape_variant_indent,
                    pidnet_spp_scale4_mul_reshape_variant_lhs,
                    pidnet_spp_scale4_mul_reshape_variant_input,
                    pidnet_spp_scale4_mul_reshape_variant_const_expr,
                    pidnet_spp_scale4_mul_reshape_variant_c,
                ) = parsed_pidnet_spp_scale4_mul_reshape_variant_reversed
                lines[index] = (
                    f"{pidnet_spp_scale4_mul_reshape_variant_indent}{pidnet_spp_scale4_mul_reshape_variant_lhs} = "
                    f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale4_mul_reshape_variant_input}, "
                    f"torch.reshape({pidnet_spp_scale4_mul_reshape_variant_const_expr}, [1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])), "
                    f"[1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])"
                )
                changed = True
                line = lines[index]
        parsed_pidnet_spp_scale4_add = _parse_raw_pidnet_scale4_add_anchor_assign(line)
        if parsed_pidnet_spp_scale4_add is not None:
            (
                pidnet_spp_scale4_add_indent,
                pidnet_spp_scale4_add_lhs0,
                pidnet_spp_scale4_add_lhs1,
                pidnet_spp_scale4_add_input,
                pidnet_spp_scale4_add_const_expr,
                pidnet_spp_scale4_add_c,
            ) = parsed_pidnet_spp_scale4_add
            lines[index] = (
                f"{pidnet_spp_scale4_add_indent}{pidnet_spp_scale4_add_lhs0}, "
                f"{pidnet_spp_scale4_add_lhs1} = _align_binary_inputs_to_anchor("
                f"{pidnet_spp_scale4_add_input}, "
                f"torch.reshape({pidnet_spp_scale4_add_const_expr}, [1, {pidnet_spp_scale4_add_c}, 1, 1]), "
                f"[1, {pidnet_spp_scale4_add_c}, 1, 1])"
            )
            changed = True
            line = lines[index]
            continue
        parsed_pidnet_pag4_mul2_out = _parse_raw_pidnet_align_binary_out_assign(line, "mul")
        if (
            parsed_pidnet_pag4_mul2_out is not None
            and parsed_pidnet_pag4_mul2_out[4][0] == 1
            and parsed_pidnet_pag4_mul2_out[4][2] > 1
            and parsed_pidnet_pag4_mul2_out[4][3] > 1
        ):
            pidnet_pag4_mul2_out_indent, pidnet_pag4_mul2_out_lhs, pidnet_pag4_mul2_out_a, pidnet_pag4_mul2_out_b, pidnet_pag4_mul2_out_shape = parsed_pidnet_pag4_mul2_out
            lines[index] = (
                f"{pidnet_pag4_mul2_out_indent}{pidnet_pag4_mul2_out_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_pag4_mul2_out_a}, "
                f"{pidnet_pag4_mul2_out_b}), "
                f"[1, {pidnet_pag4_mul2_out_shape[1]}, {pidnet_pag4_mul2_out_shape[2]}, {pidnet_pag4_mul2_out_shape[3]}])"
            )
            cf_aliases.add(str(pidnet_pag4_mul2_out_lhs))
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale3_add_out = _parse_raw_pidnet_align_binary_out_assign(line, "add")
        if (
            parsed_pidnet_spp_scale3_add_out is not None
            and parsed_pidnet_spp_scale3_add_out[4][0] == 1
            and parsed_pidnet_spp_scale3_add_out[4][2] == 1
            and parsed_pidnet_spp_scale3_add_out[4][1] == parsed_pidnet_spp_scale3_add_out[4][3]
        ):
            pidnet_spp_scale3_add_out_indent, pidnet_spp_scale3_add_out_lhs, pidnet_spp_scale3_add_out_a, pidnet_spp_scale3_add_out_b, pidnet_spp_scale3_add_out_shape = parsed_pidnet_spp_scale3_add_out
            lines[index] = (
                f"{pidnet_spp_scale3_add_out_indent}{pidnet_spp_scale3_add_out_lhs} = "
                f"_align_tensor_to_target_shape(torch.add({pidnet_spp_scale3_add_out_a}, "
                f"{pidnet_spp_scale3_add_out_b}), [1, {pidnet_spp_scale3_add_out_shape[1]}, 1, 1])"
            )
            changed = True
            line = lines[index]
        parsed_pidnet_spp_scale4_add_out = _parse_raw_pidnet_reshape_binary_out_assign(line, "add")
        if (
            parsed_pidnet_spp_scale4_add_out is not None
            and parsed_pidnet_spp_scale4_add_out[4][0] == 1
            and parsed_pidnet_spp_scale4_add_out[4][1] == 1
            and parsed_pidnet_spp_scale4_add_out[4][2] == 1
        ):
            pidnet_spp_scale4_add_out_indent, pidnet_spp_scale4_add_out_lhs, pidnet_spp_scale4_add_out_a, pidnet_spp_scale4_add_out_b, pidnet_spp_scale4_add_out_shape = parsed_pidnet_spp_scale4_add_out
            lines[index] = (
                f"{pidnet_spp_scale4_add_out_indent}{pidnet_spp_scale4_add_out_lhs} = "
                f"_align_tensor_to_target_shape(torch.add({pidnet_spp_scale4_add_out_a}, "
                f"{pidnet_spp_scale4_add_out_b}), [1, {pidnet_spp_scale4_add_out_shape[3]}, 1, 1])"
            )
            changed = True
            line = lines[index]
        if not inside_nms_method and "_apply_non_max_suppression_v4(" not in line:
            def _replace_inline_tensor_literal(match: re.Match[str]) -> str:
                literal = str(match.group("literal"))
                dtype = str(match.group("dtype"))
                cache_key = (literal, dtype)
                alias_attr = inline_const_buffer_cache.get(cache_key, "")
                if alias_attr == "":
                    alias_attr = f"const_inline_literal_{len(inline_const_buffer_cache)}"
                    inline_const_buffer_cache[cache_key] = alias_attr
                    inline_const_buffer_specs[alias_attr] = (literal, dtype)
                return f"self.{alias_attr}"

            tensor_literal_replaced = tensor_literal_as_tensor_re.sub(_replace_inline_tensor_literal, line)
            if tensor_literal_replaced != line:
                lines[index] = tensor_literal_replaced
                changed = True
                line = tensor_literal_replaced
        register_buffer_match = register_buffer_re.match(line)
        if register_buffer_match is not None:
            shape_values = [
                int(value.strip())
                for value in str(register_buffer_match.group("shape")).split(",")
                if value.strip()
            ]
            buffer_specs[str(register_buffer_match.group("name"))] = (
                int(index),
                shape_values,
                str(register_buffer_match.group("dtype")),
                str(register_buffer_match.group("persistent")) == "True",
            )
            continue
        self_const_alias_match = self_const_alias_re.match(line)
        if self_const_alias_match is not None:
            raw_pidnet_const_alias_sources[str(self_const_alias_match.group("lhs"))] = str(
                self_const_alias_match.group("attr")
            )
            const_temp_assignments[str(self_const_alias_match.group("lhs"))] = (
                int(index),
                str(self_const_alias_match.group("attr")),
                str(self_const_alias_match.group("indent")),
                str(self_const_alias_match.group("lhs")),
            )
            continue
        raw_const_pair_alias_match = raw_const_pair_alias_re.match(line)
        if raw_const_pair_alias_match is not None:
            resolved_attrs: List[str] = []
            for rhs_group in ("rhs0", "rhs1"):
                rhs_name = str(raw_const_pair_alias_match.group(rhs_group))
                if rhs_name.startswith("self."):
                    resolved_attrs.append(rhs_name[len("self.") :])
                    continue
                aliased_attr = raw_pidnet_const_alias_sources.get(rhs_name, None)
                if aliased_attr is None:
                    resolved_attrs = []
                    break
                resolved_attrs.append(aliased_attr)
            if len(resolved_attrs) == 2:
                raw_pidnet_const_pair_alias_sources[str(raw_const_pair_alias_match.group("pair"))] = (
                    resolved_attrs[0],
                    resolved_attrs[1],
                )
            continue
        raw_tuple_const_alias_match = raw_tuple_const_alias_re.match(line)
        if raw_tuple_const_alias_match is not None:
            for lhs_group, rhs_group in (("lhs0", "rhs0"), ("lhs1", "rhs1")):
                rhs_name = str(raw_tuple_const_alias_match.group(rhs_group))
                if rhs_name.startswith("self."):
                    raw_pidnet_const_alias_sources[str(raw_tuple_const_alias_match.group(lhs_group))] = rhs_name[len("self.") :]
                    continue
                aliased_attr = raw_pidnet_const_alias_sources.get(rhs_name, None)
                if aliased_attr is not None:
                    raw_pidnet_const_alias_sources[str(raw_tuple_const_alias_match.group(lhs_group))] = aliased_attr
            continue
        raw_tuple_const_unpack_match = raw_tuple_const_unpack_re.match(line)
        if raw_tuple_const_unpack_match is not None:
            pair_sources = raw_pidnet_const_pair_alias_sources.get(
                str(raw_tuple_const_unpack_match.group("pair")),
                None,
            )
            if pair_sources is not None:
                raw_pidnet_const_alias_sources[str(raw_tuple_const_unpack_match.group("lhs0"))] = str(
                    pair_sources[0]
                )
                raw_pidnet_const_alias_sources[str(raw_tuple_const_unpack_match.group("lhs1"))] = str(
                    pair_sources[1]
                )
            continue
        raw_generic_alias_assign = _parse_simple_assignment_line(line)
        if (
            raw_generic_alias_assign is not None
            and _is_simple_identifier_expr(raw_generic_alias_assign[2])
        ):
            raw_generic_alias_lhs = str(raw_generic_alias_assign[1])
            rhs_name = _strip_outer_parentheses(str(raw_generic_alias_assign[2]).strip())
            aliased_attr = raw_pidnet_const_alias_sources.get(rhs_name, None)
            if aliased_attr is not None:
                raw_pidnet_const_alias_sources[raw_generic_alias_lhs] = aliased_attr
            aliased_pair = raw_pidnet_const_pair_alias_sources.get(rhs_name, None)
            if aliased_pair is not None:
                raw_pidnet_const_pair_alias_sources[raw_generic_alias_lhs] = aliased_pair
        transposed_const_use_match = transposed_const_use_re.match(line)
        if transposed_const_use_match is None:
            continue
        temp_name = str(transposed_const_use_match.group("temp"))
        temp_assignment = const_temp_assignments.get(temp_name, None)
        if temp_assignment is None:
            continue
        temp_index, source_attr, temp_indent, temp_lhs = temp_assignment
        buffer_spec = buffer_specs.get(source_attr, None)
        if buffer_spec is None:
            continue
        _, source_shape, source_dtype, _ = buffer_spec
        if len(source_shape) < 2:
            continue
        alias_attr = f"{source_attr}_transposed"
        alias_shape = list(source_shape[:-2]) + [int(source_shape[-1]), int(source_shape[-2])]
        transposed_const_alias_specs[alias_attr] = (source_attr, alias_shape, source_dtype)
        lines[temp_index] = f"{temp_indent}{temp_lhs} = self.{alias_attr}"
        lines[index] = line.replace(f"{temp_name}.transpose(-1, -2)", temp_name)
        changed = True
    for index, line in enumerate(lines):
        aligned_nhwc_match = aligned_nhwc_rank4_re.match(line)
        if aligned_nhwc_match is None:
            continue
        lhs = str(aligned_nhwc_match.group("lhs"))
        expr = str(aligned_nhwc_match.group("expr"))
        if "_cf" not in expr:
            continue
        future_cf_spatial_consumer = False
        for lookahead in range(index + 1, min(len(lines), index + 80)):
            lookahead_pool_assign = _parse_apply_pool2d_assign_with_shape(lines[lookahead])
            if (
                lookahead_pool_assign is not None
                and str(lookahead_pool_assign[2]) == lhs
                and bool(lookahead_pool_assign[6])
            ):
                future_cf_spatial_consumer = True
                break
            lookahead_mean_assign = _parse_rank4_mean_assign(lines[lookahead])
            if (
                lookahead_mean_assign is not None
                and str(lookahead_mean_assign[2]) == lhs
                and int(lookahead_mean_assign[3]) == 1
                and int(lookahead_mean_assign[4]) == 2
            ):
                future_cf_spatial_consumer = True
                break
        if not future_cf_spatial_consumer:
            continue
        indent = str(aligned_nhwc_match.group("indent"))
        n = int(aligned_nhwc_match.group("n"))
        h = int(aligned_nhwc_match.group("h"))
        w = int(aligned_nhwc_match.group("w"))
        c = int(aligned_nhwc_match.group("c"))
        rewritten_line = (
            f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
        )
        alias_added = lhs not in cf_aliases
        lines[index] = rewritten_line
        cf_aliases.add(lhs)
        if line != rewritten_line or alias_added:
            changed = True
    for index in range(len(lines)):
        cf_nhwc_materialize_match = cf_nhwc_materialize_re.match(lines[index])
        cf_nhwc_materialize_assign = _parse_cf_nhwc_materialize_assign(lines[index])
        if cf_nhwc_materialize_match is not None or cf_nhwc_materialize_assign is not None:
            alias = str(cf_nhwc_materialize_match.group("lhs")) if cf_nhwc_materialize_match is not None else str(cf_nhwc_materialize_assign[1])
            source = str(cf_nhwc_materialize_match.group("src")) if cf_nhwc_materialize_match is not None else str(cf_nhwc_materialize_assign[2])
            if cf_nhwc_materialize_match is not None:
                n = int(cf_nhwc_materialize_match.group("n"))
                h = int(cf_nhwc_materialize_match.group("h"))
                w = int(cf_nhwc_materialize_match.group("w"))
                c = int(cf_nhwc_materialize_match.group("c"))
            else:
                n, h, w, c = [int(v) for v in list(cf_nhwc_materialize_assign[3])]
            alias_tensor = (
                model_ir.tensors.get(_resolve_model_ir_tensor_name(alias), None)
                if model_ir is not None
                else None
            )
            source_tensor = (
                model_ir.tensors.get(_resolve_model_ir_tensor_name(source), None)
                if model_ir is not None
                else None
            )
            alias_layout = (
                normalize_logical_layout(alias_tensor.logical_layout)
                if alias_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            source_layout = (
                normalize_logical_layout(source_tensor.logical_layout)
                if source_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            future_pool_consumer = any(
                (
                    lookahead_pool_assign is not None
                    and str(lookahead_pool_assign[2]) == alias
                )
                for lookahead in range(index + 1, min(index + 5, len(lines)))
                for lookahead_pool_assign in [
                    _parse_apply_pool2d_assign_with_shape(lines[lookahead])
                ]
            )
            future_reshape_consumer = any(
                (
                    (
                        lookahead_rank4_reshape_assign is not None
                        and str(lookahead_rank4_reshape_assign[2]) == alias
                    )
                    or (
                        lookahead_rank3_reshape_assign is not None
                        and str(lookahead_rank3_reshape_assign[2]) == alias
                    )
                )
                for lookahead in range(index + 1, min(index + 5, len(lines)))
                for lookahead_rank4_reshape_assign, lookahead_rank3_reshape_assign in [(
                    _parse_rank4_reshape_consumer_assign(lines[lookahead]),
                    _parse_rank3_reshape_from_rank4_source_assign(lines[lookahead]),
                )]
            )
            future_binary_align_consumer = any(
                (
                    (
                        lookahead_binary_match is not None
                        and str(lookahead_binary_match.group("a")) == alias
                    )
                    or (
                        lookahead_binary_assign is not None
                        and str(lookahead_binary_assign[3]) == alias
                    )
                )
                for lookahead in range(index + 1, min(index + 5, len(lines)))
                for lookahead_binary_match, lookahead_binary_assign in [(
                    binary_align_re.match(lines[lookahead]),
                    _parse_align_binary_inputs_assign(lines[lookahead]),
                )]
            )
            if (
                alias_tensor is not None
                and source_tensor is not None
                and is_channel_last_logical_layout(alias_layout)
                and is_channel_last_logical_layout(source_layout)
                and list(alias_tensor.shape) == [n, h, w, c]
                and not future_pool_consumer
                and not future_reshape_consumer
                and not future_binary_align_consumer
            ):
                indent = (
                    str(cf_nhwc_materialize_match.group("indent"))
                    if cf_nhwc_materialize_match is not None
                    else str(cf_nhwc_materialize_assign[0])
                )
                lines[index] = f"{indent}{alias} = {source}"
                changed = True
                continue
            for lookahead in range(index + 1, min(index + 5, len(lines))):
                binary_align_match = binary_align_re.match(lines[lookahead])
                binary_align_assign = _parse_align_binary_inputs_assign(lines[lookahead])
                if binary_align_match is None and binary_align_assign is None:
                    continue
                binary_input_a = (
                    str(binary_align_match.group("a"))
                    if binary_align_match is not None
                    else str(binary_align_assign[3])
                )
                if binary_input_a != alias:
                    continue
                target_shape = (
                    [
                        int(binary_align_match.group("n")),
                        int(binary_align_match.group("c")),
                        int(binary_align_match.group("h")),
                        int(binary_align_match.group("w")),
                    ]
                    if binary_align_match is not None
                    else [int(v) for v in list(binary_align_assign[5])]
                )
                binary_input_b = (
                    str(binary_align_match.group("b"))
                    if binary_align_match is not None
                    else str(binary_align_assign[4])
                )
                if (
                    target_shape == [n, c, h, w]
                    and "_cf" in binary_input_b
                ):
                    indent = (
                        str(cf_nhwc_materialize_match.group("indent"))
                        if cf_nhwc_materialize_match is not None
                        else str(cf_nhwc_materialize_assign[0])
                    )
                    lines[index] = f"{indent}{alias} = {source}"
                    cf_aliases.add(alias)
                    changed = True
                    break
        cf_concat_match = cf_concat_re.match(lines[index])
        if cf_concat_match is not None:
            cf_aliases.add(str(cf_concat_match.group("lhs")))
        binary_align_match = binary_align_re.match(lines[index])
        if binary_align_match is not None:
            cf_aliases.add(str(binary_align_match.group("lhs0")))
            cf_aliases.add(str(binary_align_match.group("lhs1")))
        generic_alias_assign = _parse_simple_assignment_line(lines[index])
        if (
            generic_alias_assign is not None
            and _is_simple_identifier_expr(generic_alias_assign[2])
        ):
            lhs = str(generic_alias_assign[1])
            rhs = _strip_outer_parentheses(str(generic_alias_assign[2]).strip())
            next_reshape_assign = None
            next_rank3_reshape_assign = None
            for lookahead_index in range(index + 1, min(index + 5, len(lines))):
                candidate_reshape_assign = _parse_rank4_reshape_consumer_assign(lines[lookahead_index])
                if candidate_reshape_assign is not None and str(candidate_reshape_assign[2]) == lhs:
                    next_reshape_assign = candidate_reshape_assign
                    break
                candidate_rank3_reshape_assign = _parse_rank3_reshape_from_rank4_source_assign(
                    lines[lookahead_index]
                )
                if (
                    candidate_rank3_reshape_assign is not None
                    and str(candidate_rank3_reshape_assign[2]) == lhs
                ):
                    next_rank3_reshape_assign = candidate_rank3_reshape_assign
                    break
            lhs_exact_shape = _model_ir_exact_shape(lhs)
            rhs_exact_shape = _model_ir_exact_shape(rhs)
            lhs_tensor = (
                model_ir.tensors.get(_resolve_model_ir_tensor_name(lhs), None)
                if model_ir is not None
                else None
            )
            rhs_tensor = (
                model_ir.tensors.get(_resolve_model_ir_tensor_name(rhs), None)
                if model_ir is not None
                else None
            )
            lhs_layout = (
                normalize_logical_layout(lhs_tensor.logical_layout)
                if lhs_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            rhs_layout = (
                normalize_logical_layout(rhs_tensor.logical_layout)
                if rhs_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            nhwc_target_shape = lhs_exact_shape
            if nhwc_target_shape is None and rhs_exact_shape is not None and len(rhs_exact_shape) == 4:
                nhwc_target_shape = [
                    int(rhs_exact_shape[0]),
                    int(rhs_exact_shape[2]),
                    int(rhs_exact_shape[3]),
                    int(rhs_exact_shape[1]),
                ]
            if (
                next_reshape_assign is not None
                and lhs_exact_shape is not None
                and len(lhs_exact_shape) == 4
                and is_channel_last_logical_layout(lhs_layout)
                and (
                    _is_known_cf_name(rhs, set())
                    or is_channel_first_logical_layout(rhs_layout)
                )
            ):
                indent = str(generic_alias_assign[0])
                lines[index] = (
                    f"{indent}{lhs} = _align_tensor_to_target_shape("
                    f"{rhs}.permute(0, 2, 3, 1).contiguous(), {lhs_exact_shape})"
                )
                changed = True
                cf_materialized_alias_sources[lhs] = rhs
                generic_alias_sources.pop(lhs, None)
                line = lines[index]
            elif (
                "_nhwc" in lhs
                and _is_known_cf_name(rhs, set())
                and next_reshape_assign is not None
            ):
                indent = str(generic_alias_assign[0])
                lines[index] = f"{indent}{lhs} = {rhs}.permute(0, 2, 3, 1).contiguous()"
                changed = True
                cf_materialized_alias_sources[lhs] = rhs
                generic_alias_sources.pop(lhs, None)
                line = lines[index]
            elif (
                "_nhwc" in lhs
                and _is_known_cf_name(rhs, set())
                and next_rank3_reshape_assign is not None
            ):
                indent = str(generic_alias_assign[0])
                if nhwc_target_shape is not None and len(nhwc_target_shape) == 4:
                    lines[index] = (
                        f"{indent}{lhs} = _align_tensor_to_target_shape("
                        f"{rhs}.permute(0, 2, 3, 1).contiguous(), {nhwc_target_shape})"
                    )
                else:
                    lines[index] = f"{indent}{lhs} = {rhs}.permute(0, 2, 3, 1).contiguous()"
                changed = True
                cf_materialized_alias_sources[lhs] = rhs
                generic_alias_sources.pop(lhs, None)
                line = lines[index]
            elif (
                next_rank3_reshape_assign is not None
                and _is_known_cf_name(rhs, set())
                and rhs_exact_shape is not None
                and len(rhs_exact_shape) == 4
            ):
                rank3_target_shape = [int(value) for value in next_rank3_reshape_assign[3]]
                if (
                    nhwc_target_shape is not None
                    and len(nhwc_target_shape) == 4
                    and len(rank3_target_shape) == 3
                    and int(rank3_target_shape[0]) == int(nhwc_target_shape[0])
                    and int(rank3_target_shape[2]) == int(nhwc_target_shape[3])
                    and int(rank3_target_shape[1])
                    == int(np.prod(nhwc_target_shape[1:3], dtype=np.int64))
                ):
                    indent = str(generic_alias_assign[0])
                    lines[index] = (
                        f"{indent}{lhs} = _align_tensor_to_target_shape("
                        f"{rhs}.permute(0, 2, 3, 1).contiguous(), {nhwc_target_shape})"
                    )
                    changed = True
                    cf_materialized_alias_sources[lhs] = rhs
                    generic_alias_sources.pop(lhs, None)
                    line = lines[index]
            else:
                generic_alias_sources[lhs] = rhs
                if _is_known_cf_name(rhs, set()):
                    cf_aliases.add(lhs)
        rank3_reshape_from_rank4_source_assign = _parse_rank3_reshape_from_rank4_source_assign(
            lines[index]
        )
        if rank3_reshape_from_rank4_source_assign is not None:
            lhs = str(rank3_reshape_from_rank4_source_assign[1])
            src = str(rank3_reshape_from_rank4_source_assign[2])
            target_shape = [int(value) for value in rank3_reshape_from_rank4_source_assign[3]]
            src_exact_shape = _model_ir_exact_shape(src)
            lhs_exact_shape = _model_ir_exact_shape(lhs)
            resolved_src_alias = _resolve_codegen_alias_source(src)
            resolved_src_alias_tensor_name = _resolve_model_ir_tensor_name(resolved_src_alias)
            src_tensor = (
                model_ir.tensors.get(_resolve_model_ir_tensor_name(src), None)
                if model_ir is not None
                else None
            )
            resolved_src_name = _resolve_model_ir_tensor_name(src)
            src_layout = (
                normalize_logical_layout(src_tensor.logical_layout)
                if src_tensor is not None
                else LOGICAL_LAYOUT_UNKNOWN
            )
            src_runtime_is_cf = _is_known_cf_name(src, singleton_cf_seeds) or (
                cf_materialized_alias_sources.get(src) is not None
            )
            src_declares_channel_last = (
                src_tensor is not None
                and is_channel_last_logical_layout(src_layout)
            )
            src_nhwc_cf_channel_count = _infer_cf_channel_count(resolved_src_alias)
            src_nhwc_cf_flatten_shape_matches = (
                src.endswith("_nhwc_cf")
                or resolved_src_alias.endswith("_nhwc_cf")
                or resolved_src_alias_tensor_name.endswith("_nhwc_cf")
            )
            src_cf_flatten_to_nwc_matches = (
                src_exact_shape is not None
                and len(src_exact_shape) == 4
                and len(target_shape) == 3
                and src_runtime_is_cf
                and int(target_shape[0]) == int(src_exact_shape[0])
                and int(target_shape[2]) == int(src_exact_shape[1])
                and int(target_shape[1]) == int(src_exact_shape[2] * src_exact_shape[3])
            )
            if src_cf_flatten_to_nwc_matches:
                indent = str(rank3_reshape_from_rank4_source_assign[0])
                lines[index] = (
                    f"{indent}{lhs} = torch.reshape("
                    f"{src}.permute(0, 2, 3, 1).contiguous(), "
                    f"{target_shape})"
                )
                changed = True
                line = lines[index]
                continue
            if (
                src_exact_shape is not None
                and len(src_exact_shape) == 4
                and len(target_shape) == 3
                and src_nhwc_cf_channel_count is not None
                and src_nhwc_cf_flatten_shape_matches
                and int(target_shape[0]) == int(src_exact_shape[0])
                and int(target_shape[1]) == int(src_nhwc_cf_channel_count)
                and int(target_shape[2]) == int(src_exact_shape[2] * src_exact_shape[3])
            ):
                indent = str(rank3_reshape_from_rank4_source_assign[0])
                lines[index] = (
                    f"{indent}{lhs} = torch.reshape("
                    f"{src}.permute(0, 3, 1, 2).contiguous(), "
                    f"{target_shape})"
                )
                changed = True
                line = lines[index]
                continue
            if (
                src_exact_shape is not None
                and lhs_exact_shape is not None
                and len(src_exact_shape) == 4
                and len(lhs_exact_shape) == 3
                and src_tensor is not None
                and (
                    (
                        is_channel_first_logical_layout(src_layout)
                        and _tensor_name_suggests_channel_last_layout_for_codegen(resolved_src_name)
                    )
                    or (src_runtime_is_cf and src_declares_channel_last)
                )
            ):
                if target_shape == lhs_exact_shape:
                    indent = str(rank3_reshape_from_rank4_source_assign[0])
                    lines[index] = (
                        f"{indent}{lhs} = torch.reshape("
                        f"_align_tensor_to_target_shape({src}.permute(0, 2, 3, 1).contiguous(), {src_exact_shape}), "
                        f"{target_shape})"
                    )
                    changed = True
                    line = lines[index]
        singleton_cf_align_match = singleton_cf_align_re.match(lines[index])
        if singleton_cf_align_match is not None:
            expr = str(singleton_cf_align_match.group("expr"))
            lhs = str(singleton_cf_align_match.group("lhs"))
            singleton_cf_seeds.add(lhs)
            cf_aliases.add(lhs)
            if "_torch_permute(" not in expr and ".permute(" not in expr and ("_nhwc" in expr or "torch." in expr):
                indent = str(singleton_cf_align_match.group("indent"))
                n = int(singleton_cf_align_match.group("n"))
                h = int(singleton_cf_align_match.group("h"))
                w = int(singleton_cf_align_match.group("w"))
                lines[index] = f"{indent}{lhs} = torch.reshape({expr}, [{n}, 1, {h}, {w}])"
                changed = True
        pad_align_match = pad_align_re.match(lines[index])
        if pad_align_match is not None:
            input_name = str(pad_align_match.group("input"))
            if "_nhwc" not in input_name:
                try:
                    pad_values = [int(value.strip()) for value in str(pad_align_match.group("pad")).split(",")]
                except Exception:
                    pad_values = []
                nchw_pad = _convert_nhwc_pad_to_nchw_pad_for_source(pad_values)
                if nchw_pad is not None:
                    indent = str(pad_align_match.group("indent"))
                    lhs = str(pad_align_match.group("lhs"))
                    value = str(pad_align_match.group("value"))
                    pad_text = ", ".join(str(v) for v in nchw_pad)
                    lines[index] = (
                        f"{indent}{lhs} = F.pad({input_name}, [{pad_text}], mode='constant', value={value})"
                    )
                    cf_pad_aliases.add(lhs)
                    changed = True
        pool2d_match = pool2d_re.match(lines[index])
        if pool2d_match is not None:
            lhs = str(pool2d_match.group("lhs"))
            input_name = str(pool2d_match.group("input"))
            rest = str(pool2d_match.group("rest"))
            exact_shape = _model_ir_exact_shape(lhs)
            if (
                "stride_h=1" in rest
                and "stride_w=1" in rest
                and "padding='SAME'" in rest
                and _is_known_cf_name(input_name, singleton_cf_seeds)
            ):
                indent = str(pool2d_match.group("indent"))
                is_max_pool = str(pool2d_match.group("is_max")) == "True"
                target_shape_literal = (
                    repr(exact_shape)
                    if (
                        not is_max_pool
                        and exact_shape is not None
                        and len(exact_shape) == 4
                    )
                    else f"_tensor_shape_list({input_name})"
                )
                lines[index] = (
                    f"{indent}{lhs} = _apply_pool2d("
                    f"{input_name}, {rest}, "
                    f"target_shape={target_shape_literal}, is_max_pool={pool2d_match.group('is_max')}, channel_last=False)"
                )
                changed = True
            elif exact_shape is not None and len(exact_shape) == 4:
                indent = str(pool2d_match.group("indent"))
                lines[index] = (
                    f"{indent}{lhs} = _apply_pool2d("
                    f"{input_name}, {pool2d_match.group('rest')}, "
                    f"target_shape={repr(exact_shape)}, is_max_pool={pool2d_match.group('is_max')}, channel_last=False)"
                )
                changed = True
            if _is_known_cf_name(input_name, singleton_cf_seeds) or _model_ir_is_channel_first(lhs):
                cf_aliases.add(lhs)
        generic_pool2d_assign = _parse_apply_pool2d_assign_with_shape(lines[index])
        if generic_pool2d_assign is not None:
            input_name = str(generic_pool2d_assign[2])
            lhs = str(generic_pool2d_assign[1])
            if (
                bool(generic_pool2d_assign[6])
                and not _declares_channel_last_name(lhs)
                and not _has_nearby_local_response_norm_consumer(lhs, index)
                and not _has_nearby_channel_last_spatial_consumer(lhs, index)
                and (
                    (
                        _is_known_cf_name(input_name, singleton_cf_seeds)
                        and not _tensor_name_suggests_channel_last_layout_for_codegen(input_name)
                    )
                    or input_name in cf_pad_aliases
                )
            ):
                indent = str(generic_pool2d_assign[0])
                rest = str(generic_pool2d_assign[3])
                shape_values = [int(v) for v in list(generic_pool2d_assign[4])]
                exact_shape = _model_ir_exact_shape(lhs)
                target_shape_literal = (
                    repr(exact_shape)
                    if exact_shape is not None and len(exact_shape) == 4
                    else (
                        repr(shape_values)
                        if (
                            len(shape_values) == 4
                            and int(shape_values[1]) != int(shape_values[2])
                            and int(shape_values[2]) == int(shape_values[3])
                        )
                        else (
                            repr([shape_values[0], shape_values[3], shape_values[1], shape_values[2]])
                            if len(shape_values) == 4
                            else repr(shape_values)
                        )
                    )
                )
                is_max = repr(bool(generic_pool2d_assign[5]))
                lines[index] = (
                    f"{indent}{lhs} = _apply_pool2d({input_name}, {rest}, "
                    f"target_shape={target_shape_literal}, is_max_pool={is_max}, channel_last=False)"
                )
                cf_aliases.add(lhs)
                changed = True
        mean_assign = _parse_rank4_mean_assign(lines[index])
        if mean_assign is not None:
            input_name = str(mean_assign[2])
            if (
                _is_known_cf_name(input_name, singleton_cf_seeds)
                and int(mean_assign[3]) == 1
                and int(mean_assign[4]) == 2
            ):
                indent = str(mean_assign[0])
                lhs = str(mean_assign[1])
                keepdim = str(mean_assign[5])
                lines[index] = (
                    f"{indent}{lhs} = torch.mean({input_name}, dim=[2, 3], keepdim={keepdim})"
                )
                singleton_cf_seeds.add(lhs)
                changed = True
        concat_args = _parse_simple_assignment_line(lines[index])
        parsed_apply_concat = (
            _parse_apply_concat_inputs_axis_and_shape(concat_args[2])
            if concat_args is not None
            else None
        )
        if concat_args is not None and parsed_apply_concat is not None:
            input_names = (
                [name.strip() for name in parsed_apply_concat[0] if name.strip()]
            )
            next_channel_last_gather_slice_assign = (
                _parse_channel_last_gather_slice_assign(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            if len(input_names) >= 2 and all(
                ("_cf" in input_name) or (input_name in cf_pad_aliases) or (input_name in cf_aliases)
                for input_name in input_names
            ) and not (
                next_channel_last_gather_slice_assign is not None
                and str(next_channel_last_gather_slice_assign[1]) == concat_args[1]
            ):
                indent = concat_args[0]
                lhs = concat_args[1]
                lines[index] = f"{indent}{lhs} = torch.cat([{', '.join(input_names)}], dim=1)"
                cf_aliases.add(lhs)
                changed = True
        parsed_torch_cat = (
            _parse_torch_cat_inputs_and_dim(concat_args[2])
            if concat_args is not None
            else None
        )
        if concat_args is not None and parsed_torch_cat is not None:
            input_names = (
                [name.strip() for name in parsed_torch_cat[0] if name.strip()]
            )
            normalized_inputs = [
                source_name if source_name != name and _is_name_available_in_function(source_name, index) else name
                for name in input_names
                for source_name in [cf_materialized_alias_sources.get(name, name)]
            ]
            lhs = concat_args[1]
            axis = parsed_torch_cat[1]
            next_channel_last_gather_slice_assign = (
                _parse_channel_last_gather_slice_assign(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            if model_ir is not None and lhs in model_ir.outputs and axis != 1:
                continue
            if (
                len(normalized_inputs) >= 2
                and all(_is_known_cf_name(name, singleton_cf_seeds) for name in normalized_inputs)
                and (
                    axis != 1
                    or normalized_inputs != input_names
                )
                and not (
                    next_channel_last_gather_slice_assign is not None
                    and str(next_channel_last_gather_slice_assign[1]) == lhs
                )
                and (_is_known_cf_name(lhs, singleton_cf_seeds) or _model_ir_is_channel_first(lhs) or lhs.endswith("_cf"))
            ):
                indent = str(concat_args[0])
                lines[index] = f"{indent}{lhs} = torch.cat([{', '.join(normalized_inputs)}], dim=1)"
                cf_aliases.add(lhs)
                changed = True
    for index in range(len(lines) - 1):
        simple_assign = _parse_simple_assignment_line(lines[index])
        public_layout_bridge_assign = _parse_public_layout_bridge_assign(lines[index])
        if simple_assign is None and public_layout_bridge_assign is None:
            continue
        split_assign = _parse_tensor_split_assign(lines[index + 1])
        if split_assign is None:
            continue
        simple_alias_assign: Tuple[str, str, str] | None = None
        if public_layout_bridge_assign is None and simple_assign is not None:
            simple_rhs = _strip_outer_parentheses(str(simple_assign[2]).strip())
            if (
                re.fullmatch(r"(?:[A-Za-z0-9_]+_public_layout_bridge|in_public_layout_bridge)", str(simple_assign[1]))
                is not None
                and re.fullmatch(r"[A-Za-z0-9_]+", simple_rhs) is not None
            ):
                simple_alias_assign = (str(simple_assign[0]), str(simple_assign[1]), simple_rhs)
        resolved_assign = public_layout_bridge_assign if public_layout_bridge_assign is not None else simple_alias_assign
        if resolved_assign is None:
            continue
        resolved_alias = str(resolved_assign[1])
        split_alias = str(split_assign[2]) if split_assign is not None else ""
        if split_alias != resolved_alias:
            continue
        indent = str(resolved_assign[0])
        alias = resolved_alias
        input_name = str(resolved_assign[2])
        outputs = ", ".join(split_assign[1])
        sections = int(split_assign[3])
        if sections <= 1:
            continue
        lines[index] = f"{indent}{alias} = {input_name}"
        lines[index + 1] = (
            f"{indent}{outputs} = list(torch.tensor_split("
            f"{alias}, {sections}, dim=_normalize_dim(1, {alias}.ndim)))"
        )
        changed = True
    for index, line in enumerate(lines):
        split_assign = _parse_tensor_split_assign(line)
        if split_assign is None:
            continue
        alias = str(split_assign[2])
        if not re.fullmatch(r"(?:[A-Za-z0-9_]+_public_layout_bridge|in_public_layout_bridge)", alias):
            continue
        indent = str(split_assign[0])
        outputs = ", ".join(split_assign[1])
        sections = int(split_assign[3])
        lines[index] = (
            f"{indent}{outputs} = list(torch.tensor_split("
            f"{alias}, {sections}, dim=_normalize_dim(1, {alias}.ndim)))"
        )
        changed = True
    singleton_cf_vars: set[str] = set(singleton_cf_seeds)
    index = 0
    while index < len(lines):
        cf_concat_match = cf_concat_re.match(lines[index])
        cf_materialize_match = (
            cf_nhwc_materialize_re.match(lines[index + 1])
            if index + 1 < len(lines)
            else None
        )
        split_match = split_re.match(lines[index + 2]) if index + 2 < len(lines) else None
        if (
            cf_concat_match is not None
            and cf_materialize_match is not None
            and split_match is not None
            and str(cf_materialize_match.group("src")) == str(cf_concat_match.group("lhs"))
            and str(split_match.group("alias")) == str(cf_materialize_match.group("lhs"))
        ):
            indent = str(cf_materialize_match.group("indent"))
            alias = str(cf_materialize_match.group("lhs"))
            sections = int(split_match.group("sections"))
            outputs = str(split_match.group("outputs"))
            lines[index + 1] = f"{indent}{alias} = {cf_concat_match.group('lhs')}"
            lines[index + 2] = (
                f"{indent}{outputs} = list(torch.tensor_split("
                f"{alias}, {sections}, dim=_normalize_dim(1, {alias}.ndim)))"
            )
            for output_name in [token.strip() for token in outputs.split(",") if token.strip()]:
                singleton_cf_vars.add(output_name)
            cf_aliases.add(alias)
            changed = True
            index += 3
            continue
        split_assign = _parse_tensor_split_assign(lines[index])
        if split_assign is not None and int(split_assign[3]) > 1:
            if int(split_assign[4]) == 1:
                for output_name in [token.strip() for token in split_assign[1] if token.strip()]:
                    singleton_cf_vars.add(output_name)
            elif (
                int(split_assign[4]) == 3
                and _is_known_cf_name(str(split_assign[2]), singleton_cf_vars)
            ):
                indent = str(split_assign[0])
                outputs = ", ".join(split_assign[1])
                input_name = str(split_assign[2])
                sections = int(split_assign[3])
                lines[index] = (
                    f"{indent}{outputs} = list(torch.tensor_split("
                    f"{input_name}, {sections}, dim=_normalize_dim(1, {input_name}.ndim)))"
                )
                for output_name in [token.strip() for token in outputs.split(",") if token.strip()]:
                    singleton_cf_vars.add(output_name)
                changed = True
        reshape_match = same_shape_singleton_reshape_re.match(lines[index])
        unary_assign_match = unary_assign_re.match(lines[index])
        if unary_assign_match is not None:
            lhs = str(unary_assign_match.group("lhs"))
            indent = str(unary_assign_match.group("indent"))
            expr = str(unary_assign_match.group("expr"))
            unary_passthrough_match = re.match(
                r"torch\.(?:relu|neg|sigmoid|exp|clamp)\((?P<arg0>[A-Za-z0-9_]+).*\)$",
                expr,
            )
            if (
                unary_passthrough_match is not None
                and str(unary_passthrough_match.group("arg0")) in singleton_cf_vars
            ):
                singleton_cf_vars.add(lhs)
            if lhs not in singleton_cf_vars:
                for lookahead in range(index + 1, min(index + 6, len(lines))):
                    later_reshape_match = same_shape_singleton_reshape_re.match(lines[lookahead])
                    if later_reshape_match is None:
                        continue
                    later_expr = str(later_reshape_match.group("expr"))
                    if re.search(rf"\b{re.escape(lhs)}\b", later_expr) is None:
                        continue
                    n = int(later_reshape_match.group("n"))
                    h = int(later_reshape_match.group("h"))
                    w = int(later_reshape_match.group("w"))
                    lines[index] = (
                        f"{indent}{lhs} = torch.reshape({expr}, [{n}, 1, {h}, {w}])"
                    )
                    singleton_cf_vars.add(lhs)
                    changed = True
                    break
        binary_assign_match = binary_assign_re.match(lines[index])
        if binary_assign_match is not None:
            lhs = str(binary_assign_match.group("lhs"))
            expr = str(binary_assign_match.group("expr"))
            if any(
                re.search(rf"\b{re.escape(name)}\b", expr) is not None
                for name in sorted(singleton_cf_vars)
            ):
                singleton_cf_vars.add(lhs)
        anchor_match_current = binary_anchor_align_re.match(lines[index])
        if anchor_match_current is not None:
            lhs0 = str(anchor_match_current.group("lhs0"))
            lhs1 = str(anchor_match_current.group("lhs1"))
            singleton_cf_vars.add(lhs0)
            singleton_cf_vars.add(lhs1)
            cf_aliases.add(lhs0)
            cf_aliases.add(lhs1)
        if reshape_match is not None:
            expr = str(reshape_match.group("expr"))
            indent = str(reshape_match.group("indent"))
            lhs = str(reshape_match.group("lhs"))
            n = int(reshape_match.group("n"))
            h = int(reshape_match.group("h"))
            w = int(reshape_match.group("w"))
            singleton_cf_vars.add(lhs)
            next_binary_assign_match = (
                binary_assign_re.match(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            if (
                next_binary_assign_match is not None
                and h == 1
                and w > 1
            ):
                next_binary_expr_match = simple_binary_expr_re.match(
                    str(next_binary_assign_match.group("expr"))
                )
                if next_binary_expr_match is not None:
                    arg_a = str(next_binary_expr_match.group("a"))
                    arg_b = str(next_binary_expr_match.group("b"))
                    other_arg = arg_b if arg_a == lhs else arg_a if arg_b == lhs else None
                    if (
                        other_arg is not None
                        and (
                            _is_known_cf_name(other_arg, singleton_cf_vars)
                            or other_arg in cf_aliases
                            or other_arg.endswith("_cf")
                            or other_arg.endswith("_out_cf")
                        )
                    ):
                        lines[index] = (
                            f"{indent}{lhs} = torch.reshape({expr}, [{n}, {w}, 1, 1])"
                        )
                        cf_aliases.add(lhs)
                        changed = True
                        index += 1
                        continue
            if expr in singleton_cf_vars:
                lines[index] = f"{indent}{lhs} = {expr}"
                singleton_cf_vars.add(lhs)
                changed = True
                index += 1
                continue
            previous_line = lines[index - 1] if index > 0 else ""
            anchor_match = binary_anchor_align_re.match(previous_line)
            if (
                anchor_match is not None
                and (
                    expr == f"torch.sub({anchor_match.group('lhs1')}, {anchor_match.group('lhs0')})"
                    or expr == f"torch.sub({anchor_match.group('lhs0')}, {anchor_match.group('lhs1')})"
                    or expr == f"torch.add({anchor_match.group('lhs1')}, {anchor_match.group('lhs0')})"
                    or expr == f"torch.add({anchor_match.group('lhs0')}, {anchor_match.group('lhs1')})"
                    or expr == f"torch.mul({anchor_match.group('lhs1')}, {anchor_match.group('lhs0')})"
                    or expr == f"torch.mul({anchor_match.group('lhs0')}, {anchor_match.group('lhs1')})"
                    or expr == f"torch.div({anchor_match.group('lhs1')}, {anchor_match.group('lhs0')})"
                    or expr == f"torch.div({anchor_match.group('lhs0')}, {anchor_match.group('lhs1')})"
                )
            ):
                lines[index] = f"{indent}{lhs} = {expr}"
                singleton_cf_vars.add(lhs)
                changed = True
                index += 1
                continue
            unary_match = re.match(
                r"torch\.(?:mul|add|sub|div)\((?P<arg0>[A-Za-z0-9_]+), (?P<arg1>.+)\)$",
                expr,
            )
            arg1_uses_singleton_cf = False
            if unary_match is not None and len(singleton_cf_vars) > 0:
                arg1_uses_singleton_cf = (
                    re.search(
                        rf"\b(?:{'|'.join(re.escape(name) for name in sorted(singleton_cf_vars))})\b",
                        str(unary_match.group("arg1")),
                    )
                    is not None
                )
            if unary_match is not None and (
                str(unary_match.group("arg0")) in singleton_cf_vars
                or arg1_uses_singleton_cf
            ):
                lines[index] = f"{indent}{lhs} = {expr}"
                singleton_cf_vars.add(lhs)
                changed = True
                index += 1
                continue
            unary_passthrough_match = re.match(
                r"torch\.(?:relu|neg|sigmoid|exp|clamp)\((?P<arg0>[A-Za-z0-9_]+).*\)$",
                expr,
            )
            if (
                unary_passthrough_match is not None
                and str(unary_passthrough_match.group("arg0")) in singleton_cf_vars
            ):
                lines[index] = f"{indent}{lhs} = {expr}"
                singleton_cf_vars.add(lhs)
                changed = True
        binary_anchor_nhwc_match = binary_anchor_align_nhwc_singleton_re.match(lines[index])
        parsed_binary_anchor_nhwc = _parse_align_binary_inputs_to_anchor_assign(lines[index])
        if (
            parsed_binary_anchor_nhwc is not None
            and parsed_binary_anchor_nhwc[5][0] == 1
            and parsed_binary_anchor_nhwc[5][3] == 1
            and re.fullmatch(r"[A-Za-z0-9_]+", parsed_binary_anchor_nhwc[3]) is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", parsed_binary_anchor_nhwc[4]) is not None
        ):
            _, lhs0, lhs1, input_a, input_b, anchor_shape = parsed_binary_anchor_nhwc
            input_a_is_cf = _is_known_cf_name(input_a, singleton_cf_vars)
            input_b_is_cf = _is_known_cf_name(input_b, singleton_cf_vars)
            neither_input_is_explicit_nhwc = (
                not input_a.endswith("_nhwc")
                and not input_a.endswith("_nhwc_cf")
                and not input_b.endswith("_nhwc")
                and not input_b.endswith("_nhwc_cf")
            )
            if (
                (input_a_is_cf and input_b_is_cf)
                or (
                    neither_input_is_explicit_nhwc
                    and (input_a_is_cf or input_b_is_cf)
                )
            ):
                indent = parsed_binary_anchor_nhwc[0]
                n, h, w = int(anchor_shape[0]), int(anchor_shape[1]), int(anchor_shape[2])
                lines[index] = (
                    f"{indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor("
                    f"{input_a}, {input_b}, [{n}, 1, {h}, {w}])"
                )
                singleton_cf_vars.add(lhs0)
                singleton_cf_vars.add(lhs1)
                changed = True
        binary_anchor_rank4_match = binary_anchor_align_rank4_re.match(lines[index])
        parsed_binary_anchor_rank4 = _parse_align_binary_inputs_to_anchor_assign(lines[index])
        if (
            parsed_binary_anchor_rank4 is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", parsed_binary_anchor_rank4[3]) is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", parsed_binary_anchor_rank4[4]) is not None
        ):
            lhs0 = parsed_binary_anchor_rank4[1]
            lhs1 = parsed_binary_anchor_rank4[2]
            input_a = parsed_binary_anchor_rank4[3]
            input_b = parsed_binary_anchor_rank4[4]
            if (
                _is_known_cf_name(input_a, singleton_cf_vars)
                and _is_known_cf_name(input_b, singleton_cf_vars)
            ):
                cf_aliases.add(lhs0)
                cf_aliases.add(lhs1)
                changed = True
        aligned_nhwc_singleton_binary_match = aligned_nhwc_singleton_binary_re.match(lines[index])
        if aligned_nhwc_singleton_binary_match is not None:
            expr = str(aligned_nhwc_singleton_binary_match.group("expr"))
            if _expr_references_known_cf_identifier(expr, singleton_cf_vars):
                indent = str(aligned_nhwc_singleton_binary_match.group("indent"))
                lhs = str(aligned_nhwc_singleton_binary_match.group("lhs"))
                n = int(aligned_nhwc_singleton_binary_match.group("n"))
                h = int(aligned_nhwc_singleton_binary_match.group("h"))
                w = int(aligned_nhwc_singleton_binary_match.group("w"))
                lines[index] = (
                    f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, 1, {h}, {w}])"
                )
                singleton_cf_vars.add(lhs)
                changed = True
        index += 1
    index = 0
    while index + 7 < len(lines):
        rank3_input_match = rank3_resize_input_re.match(lines[index])
        tmp_x0_match = generic_expr_assign_re.match(lines[index + 1])
        tmp_y0_assign = _parse_simple_assignment_line(lines[index + 2])
        rank3_matmul_match = rank3_matmul_re.match(lines[index + 3])
        rank4_reshape_match = rank4_singleton_reshape_re.match(lines[index + 4])
        tmp_x1_match = generic_expr_assign_re.match(lines[index + 5])
        tmp_y1_assign = _parse_simple_assignment_line(lines[index + 6])
        rank4_matmul_match = rank4_singleton_matmul_re.match(lines[index + 7])
        if (
            rank3_input_match is None
            or tmp_x0_match is None
            or tmp_y0_assign is None
            or not _is_simple_identifier_expr(tmp_y0_assign[2])
            or rank3_matmul_match is None
            or rank4_reshape_match is None
            or tmp_x1_match is None
            or tmp_y1_assign is None
            or not _is_simple_identifier_expr(tmp_y1_assign[2])
            or rank4_matmul_match is None
        ):
            index += 1
            continue
        tmp_y0_rhs = _strip_outer_parentheses(str(tmp_y0_assign[2]).strip())
        tmp_y1_rhs = _strip_outer_parentheses(str(tmp_y1_assign[2]).strip())
        rank3_input_lhs = str(rank3_input_match.group("lhs"))
        rank3_input_src = str(rank3_input_match.group("src"))
        if (
            tmp_y0_rhs != rank3_input_lhs
            or str(rank3_matmul_match.group("x")) != str(tmp_x0_match.group("lhs"))
            or str(rank3_matmul_match.group("y")) != str(tmp_y0_assign[1])
            or str(rank4_reshape_match.group("src")) != str(rank3_matmul_match.group("lhs"))
            or tmp_y1_rhs != str(rank4_reshape_match.group("lhs"))
            or str(rank4_matmul_match.group("x")) != str(tmp_x1_match.group("lhs"))
            or str(rank4_matmul_match.group("y")) != str(tmp_y1_assign[1])
        ):
            index += 1
            continue
        indent = str(rank3_input_match.group("indent"))
        rank3_input_n = int(rank3_input_match.group("n"))
        rank3_matmul_h = int(rank3_matmul_match.group("h"))
        rank3_matmul_w = int(rank3_matmul_match.group("w"))
        rank4_matmul_w = int(rank4_matmul_match.group("w"))
        rank3_matmul_lhs = str(rank3_matmul_match.group("lhs"))
        rank4_reshape_lhs = str(rank4_reshape_match.group("lhs"))
        rank4_matmul_lhs = str(rank4_matmul_match.group("lhs"))
        tmp_x1_name = str(tmp_x1_match.group("lhs"))
        lines[index] = f"{indent}{rank3_input_lhs} = {rank3_input_src}"
        lines[index + 3] = (
            f"{indent}{rank3_matmul_lhs} = _align_tensor_to_target_shape("
            f"torch.matmul({tmp_x0_match.group('lhs')}, {tmp_y0_assign[1]}), "
            f"[{rank3_input_n}, 1, {rank3_matmul_h}, {rank3_matmul_w}])"
        )
        lines[index + 4] = f"{indent}{rank4_reshape_lhs} = {rank3_matmul_lhs}"
        lines[index + 7] = (
            f"{indent}{rank4_matmul_lhs} = _align_tensor_to_target_shape("
            f"torch.matmul({tmp_y1_assign[1]}, {tmp_x1_name}.transpose(-1, -2)), "
            f"[{rank3_input_n}, 1, {rank3_matmul_h}, {rank4_matmul_w}])"
        )
        singleton_cf_vars.add(rank4_matmul_lhs)
        cf_aliases.add(rank4_matmul_lhs)
        changed = True
        index += 8
    def _pidnet_forced_resize_target(
        lhs_name: str,
        input_name: str,
        current_target_shape: Sequence[int],
        out_h: int,
        out_w: int,
    ) -> list[int] | None:
        if not (
            _is_known_cf_name(input_name, singleton_cf_vars)
            or input_name.endswith("_cf")
            or input_name.endswith("_out_cf")
            or lhs_name.endswith("_cf")
            or lhs_name.endswith("_out_cf")
        ):
            return None
        preferred_channel_count = _infer_cf_channel_count(input_name)
        shape_values = [int(v) for v in list(current_target_shape)]
        if (
            len(shape_values) == 4
            and int(shape_values[1]) == int(out_h)
            and int(shape_values[2]) == int(out_w)
        ):
            preferred_channel_count = int(shape_values[3])
        if preferred_channel_count is None:
            lhs_exact_shape = _model_ir_exact_shape(lhs_name)
            if lhs_exact_shape is not None and len(lhs_exact_shape) == 4:
                normalized_exact = _normalize_cf_rank4_shape(
                    lhs_exact_shape,
                    out_hw=(out_h, out_w),
                )
                preferred_channel_count = int(normalized_exact[1])
        normalized_shape = _normalize_cf_rank4_shape(
            current_target_shape,
            preferred_channel_count=preferred_channel_count,
            out_hw=(out_h, out_w),
        )
        if normalized_shape == [int(v) for v in list(current_target_shape)]:
            return None
        return normalized_shape

    def _parse_resize_assign_line(
        line: str,
    ) -> Tuple[str, str, str, int, int, List[int], str, bool, bool, bool] | None:
        assignment = _parse_simple_assignment_line(line)
        if assignment is None:
            return None
        indent, lhs, rhs = assignment
        stripped_rhs = rhs.strip()
        if not stripped_rhs.startswith("_apply_resize("):
            return None
        resize_args = _parse_apply_resize_input_size_shape_and_channel_last(stripped_rhs)
        if resize_args is None:
            return None
        input_name, size_value, shape_value, channel_last = resize_args
        if size_value is None or shape_value is None:
            return None
        method_value: str | None = None
        align_value: bool | None = None
        hpc_value: bool | None = None
        parts = _split_top_level_csv_exprs(stripped_rhs[len("_apply_resize(") : -1])
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "method":
                method_match = re.fullmatch(r"'(?P<method>[^']+)'", value)
                if method_match is not None:
                    method_value = str(method_match.group("method"))
            elif key == "align_corners" and value in {"True", "False"}:
                align_value = value == "True"
            elif key == "half_pixel_centers" and value in {"True", "False"}:
                hpc_value = value == "True"
        if method_value is None or align_value is None or hpc_value is None:
            return None
        return (
            indent,
            lhs,
            input_name,
            int(size_value[0]),
            int(size_value[1]),
            [int(shape_value[0]), int(shape_value[1]), int(shape_value[2]), int(shape_value[3])],
            method_value,
            align_value,
            hpc_value,
            channel_last,
        )

    for index, line in enumerate(lines):
        parsed_resize = _parse_resize_assign_line(line)
        if parsed_resize is None or parsed_resize[9]:
            continue
        indent, lhs, input_name, out_h, out_w, target_shape, method, align_corners, half_pixel_centers, _ = parsed_resize
        if not _is_known_cf_name(input_name, singleton_cf_vars):
            continue
        normalized_target_shape = _pidnet_forced_resize_target(
            lhs_name=lhs,
            input_name=input_name,
            current_target_shape=target_shape,
            out_h=out_h,
            out_w=out_w,
        )
        if normalized_target_shape is None:
            continue
        lines[index] = (
            f"{indent}{lhs} = _apply_resize("
            f"{input_name}, [{out_h}, {out_w}], method='{method}', "
            f"target_shape={normalized_target_shape}"
            f", align_corners={align_corners}, half_pixel_centers={half_pixel_centers}, channel_last=False)"
        )
        cf_aliases.add(lhs)
        changed = True
    for index, line in enumerate(lines):
        parsed_resize = _parse_resize_assign_line(line)
        if parsed_resize is None or not parsed_resize[9]:
            continue
        indent, lhs, input_name, out_h, out_w, target_shape, method, align_corners, half_pixel_centers, _ = parsed_resize
        n, h, w, c = target_shape
        alias_match = (
            re.fullmatch(
                r"(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<src>[A-Za-z0-9_]+)",
                lines[index + 1],
            )
            if index + 1 < len(lines)
            else None
        )
        argmax_index = index + 1
        if alias_match is not None and str(alias_match.group("src")) == lhs:
            argmax_index = index + 2
        argmax_assign = None
        if argmax_index < len(lines):
            argmax_assign = _parse_argmax_assign(lines[argmax_index])
        if (
            (_is_known_cf_name(input_name, singleton_cf_vars) or input_name.endswith("_nhwc_cf"))
            and argmax_assign is not None
            and int(argmax_assign[3]) == 3
            and (
                str(argmax_assign[2]) == lhs
                or (
                    alias_match is not None
                    and str(alias_match.group("src")) == lhs
                    and str(argmax_assign[2]) == str(alias_match.group("lhs"))
                )
            )
        ):
            channel_count = _infer_cf_channel_count(input_name)
            if channel_count is None:
                channel_count = c
            lines[index] = (
                f"{indent}{lhs} = _apply_resize("
                f"{input_name}, [{out_h}, {out_w}], method='{method}', "
                f"target_shape=[{n}, {channel_count}, {out_h}, {out_w}], "
                f"align_corners={align_corners}, "
                f"half_pixel_centers={half_pixel_centers}, channel_last=False)"
            )
            if alias_match is not None and str(alias_match.group("src")) == lhs:
                alias_lhs = str(alias_match.group("lhs"))
                lines[index + 1] = f"{alias_match.group('indent')}{alias_lhs} = {lhs}"
                cf_aliases.add(alias_lhs)
            argmax_indent = str(argmax_assign[0])
            argmax_lhs = str(argmax_assign[1])
            argmax_input = str(argmax_assign[2])
            argmax_keepdim = str(argmax_assign[4])
            lines[argmax_index] = (
                f"{argmax_indent}{argmax_lhs} = "
                f"torch.argmax({argmax_input}, "
                f"dim=_normalize_dim(1, {argmax_input}.ndim), "
                f"keepdim={argmax_keepdim}).to(dtype=torch.int64)"
            )
            cf_aliases.add(lhs)
            changed = True
            continue
        forced_pidnet_target = _pidnet_forced_resize_target(
            lhs,
            input_name,
            [n, h, w, c],
            out_h,
            out_w,
        )
        if forced_pidnet_target is not None:
            lines[index] = (
                f"{indent}{lhs} = _apply_resize("
                f"{input_name}, [{out_h}, {out_w}], method='{method}', "
                f"target_shape={forced_pidnet_target}, "
                f"align_corners={align_corners}, "
                f"half_pixel_centers={half_pixel_centers}, channel_last=False)"
            )
            cf_aliases.add(lhs)
            changed = True
            continue
        lhs_exact_shape = _model_ir_exact_shape(lhs)
        lhs_tensor = (
            model_ir.tensors.get(_resolve_model_ir_tensor_name(lhs), None)
            if model_ir is not None
            else None
        )
        lhs_layout = (
            normalize_logical_layout(lhs_tensor.logical_layout)
            if lhs_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        if (
            lhs_exact_shape is not None
            and len(lhs_exact_shape) == 4
            and is_channel_last_logical_layout(lhs_layout)
            and [n, h, w, c] != [int(v) for v in lhs_exact_shape]
        ):
            n, h, w, c = [int(v) for v in lhs_exact_shape]
            lines[index] = (
                f"{indent}{lhs} = _apply_resize("
                f"{input_name}, [{out_h}, {out_w}], method='{method}', "
                f"target_shape=[{n}, {h}, {w}, {c}], "
                f"align_corners={align_corners}, "
                f"half_pixel_centers={half_pixel_centers}, channel_last=True)"
            )
            changed = True
            line = lines[index]
        function_end = _function_end_index(index)
        alias_consumer_name = lhs
        alias_line_index = None
        if alias_match is not None and str(alias_match.group("src")) == lhs:
            alias_consumer_name = str(alias_match.group("lhs"))
            alias_line_index = index + 1
        saw_future_use = False
        only_binary_cf_consumers = True
        consumer_shapes: set[Tuple[int, int, int, int]] = set()
        tracked_names = {lhs, alias_consumer_name}
        for future_index in range(index + 1, function_end):
            if alias_line_index is not None and future_index == alias_line_index:
                continue
            future_line = lines[future_index]
            if (
                re.search(rf"\b{re.escape(lhs)}\b", future_line) is None
                and re.search(rf"\b{re.escape(alias_consumer_name)}\b", future_line) is None
            ):
                continue
            saw_future_use = True
            binary_cf_consumer_match = binary_cf_consumer_re.match(future_line)
            if (
                binary_cf_consumer_match is None
                or not tracked_names.intersection(
                    {
                        str(binary_cf_consumer_match.group("a")),
                        str(binary_cf_consumer_match.group("b")),
                    }
                )
            ):
                only_binary_cf_consumers = False
                break
            consumer_shapes.add(
                (
                    int(binary_cf_consumer_match.group("n")),
                    int(binary_cf_consumer_match.group("c")),
                    int(binary_cf_consumer_match.group("h")),
                    int(binary_cf_consumer_match.group("w")),
                )
            )
        if saw_future_use:
            if only_binary_cf_consumers:
                if len(consumer_shapes) == 1:
                    _, target_c, target_h, target_w = next(iter(consumer_shapes))
                else:
                    only_binary_cf_consumers = False
            if only_binary_cf_consumers:
                current_target_shape = [int(v) for v in [n, h, w, c]]
                if (
                    int(current_target_shape[1]) == int(out_h)
                    and int(current_target_shape[2]) == int(out_w)
                ):
                    target_shape_text = (
                        f"[_tensor_shape_list({input_name})[0], "
                        f"_tensor_shape_list({input_name})[1], {out_h}, {out_w}]"
                    )
                else:
                    target_shape_text = f"[{n}, {target_c}, {target_h}, {target_w}]"
                lines[index] = (
                    f"{indent}{lhs} = _apply_resize("
                    f"{input_name}, [{out_h}, {out_w}], method='{method}', "
                    f"target_shape={target_shape_text}, "
                    f"align_corners={align_corners}, "
                    f"half_pixel_centers={half_pixel_centers}, channel_last=False)"
                )
                cf_aliases.add(lhs)
                if alias_consumer_name != lhs:
                    cf_aliases.add(alias_consumer_name)
                if target_c == 1:
                    singleton_cf_vars.add(lhs)
                changed = True
                continue
        if not _is_known_cf_name(input_name, singleton_cf_vars):
            continue
        lines[index] = (
            f"{indent}{lhs} = _apply_resize("
            f"{input_name}, [{out_h}, {out_w}], method='{method}', "
            f"target_shape=[{n}, {c}, {out_h}, {out_w}], "
            f"align_corners={align_corners}, "
            f"half_pixel_centers={half_pixel_centers}, channel_last=False)"
        )
        cf_aliases.add(lhs)
        if c == 1:
            singleton_cf_vars.add(lhs)
        changed = True
    for index in range(len(lines) - 1):
        aligned_resize_input_match = aligned_nhwc_rank4_re.match(lines[index])
        if aligned_resize_input_match is None:
            continue
        resize_cf_match = (
            apply_resize_cf_re.match(lines[index + 1])
            or apply_resize_cf_bad_target_re.match(lines[index + 1])
            or apply_resize_nhwc_re.match(lines[index + 1])
        )
        if (
            resize_cf_match is None
            or str(resize_cf_match.group("input")) != str(aligned_resize_input_match.group("lhs"))
            or (
                not _find_same_function_cat_consumer(str(resize_cf_match.group("lhs")), index + 1)
                and not _find_stage_boundary_cat_consumer(str(resize_cf_match.group("lhs")), index + 1)
            )
        ):
            continue
        indent = str(aligned_resize_input_match.group("indent"))
        lhs = str(aligned_resize_input_match.group("lhs"))
        expr = str(aligned_resize_input_match.group("expr"))
        n = int(aligned_resize_input_match.group("n"))
        h = int(aligned_resize_input_match.group("h"))
        w = int(aligned_resize_input_match.group("w"))
        c = int(aligned_resize_input_match.group("c"))
        out_h = int(resize_cf_match.group("out_h"))
        out_w = int(resize_cf_match.group("out_w"))
        input_exact_shape = _model_ir_exact_shape(lhs)
        output_exact_shape = _model_ir_exact_shape(str(resize_cf_match.group("lhs")))
        resolved_input_name = _resolve_model_ir_tensor_name(lhs)
        if (
            input_exact_shape is not None
            and output_exact_shape is not None
            and len(input_exact_shape) == 4
            and len(output_exact_shape) == 4
            and _tensor_name_suggests_channel_last_layout_for_codegen(resolved_input_name)
        ):
            n = int(input_exact_shape[0])
            h = int(input_exact_shape[1])
            w = int(input_exact_shape[2])
            c = int(input_exact_shape[3])
            out_h = int(output_exact_shape[1])
            out_w = int(output_exact_shape[2])
        lines[index] = (
            f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
        )
        lines[index + 1] = (
            f"{resize_cf_match.group('indent')}{resize_cf_match.group('lhs')} = _apply_resize("
            f"{lhs}, [{out_h}, {out_w}], method='{resize_cf_match.group('method')}', "
            f"target_shape=[{n}, {c}, {out_h}, {out_w}], "
            f"align_corners={resize_cf_match.group('align')}, "
            f"half_pixel_centers={resize_cf_match.group('hpc')}, channel_last=False)"
        )
        cf_aliases.add(lhs)
        cf_aliases.add(str(resize_cf_match.group("lhs")))
        changed = True
    index = 0
    while index < len(lines):
        current_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        binary_anchor_rank4_match = (
            _cached_regex_match("binary_anchor_align_rank4_re", binary_anchor_align_rank4_re, current_line)
            if "_align_binary_inputs_to_anchor(" in current_line
            else None
        )
        parsed_binary_anchor_rank4 = (
            _parse_align_binary_inputs_to_anchor_assign(current_line)
            if "_align_binary_inputs_to_anchor(" in current_line
            else None
        )
        nhwc_mul_match = (
            _cached_regex_match("aligned_nhwc_rank4_re", aligned_nhwc_rank4_re, next_line)
            if "_align_tensor_to_target_shape(" in next_line
            else None
        )
        reduce_sum_match = None
        reduce_sum_index = -1
        if binary_anchor_rank4_match is not None and nhwc_mul_match is not None:
            mul_lhs = str(nhwc_mul_match.group("lhs"))
            for candidate_index in range(index + 2, min(index + 7, len(lines))):
                candidate_line = lines[candidate_index]
                candidate_match = (
                    _cached_regex_match("reduce_sum_re", reduce_sum_re, candidate_line)
                    if "_reduce_sum(" in candidate_line
                    else None
                )
                if (
                    candidate_match is not None
                    and str(candidate_match.group("input")) == mul_lhs
                    and int(candidate_match.group("axis")) == 3
                    and str(candidate_match.group("keepdims")) == "True"
                ):
                    reduce_sum_match = candidate_match
                    reduce_sum_index = candidate_index
                    break
        if (
            parsed_binary_anchor_rank4 is not None
            and nhwc_mul_match is not None
            and reduce_sum_match is not None
            and str(nhwc_mul_match.group("expr"))
            == f"torch.mul({parsed_binary_anchor_rank4[2]}, {parsed_binary_anchor_rank4[1]})"
        ):
            indent, lhs0, lhs1, a, b, anchor_shape = parsed_binary_anchor_rank4
            n, h, w, c = anchor_shape
            mul_lhs = str(nhwc_mul_match.group("lhs"))
            reduce_lhs = str(reduce_sum_match.group("lhs"))
            lines[index] = (
                f"{indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor({a}, {b}, [{n}, {c}, {h}, {w}])"
            )
            lines[index + 1] = (
                f"{nhwc_mul_match.group('indent')}{mul_lhs} = _align_tensor_to_target_shape("
                f"torch.mul({lhs1}, {lhs0}), [{n}, {c}, {h}, {w}])"
            )
            lines[reduce_sum_index] = (
                f"{reduce_sum_match.group('indent')}{reduce_lhs} = _reduce_sum("
                f"{mul_lhs}, _normalize_axes([1], {mul_lhs}.ndim), True)"
            )
            cf_aliases.add(mul_lhs)
            singleton_cf_vars.add(reduce_lhs)
            changed = True
            index = reduce_sum_index + 1
            continue
        if (
            binary_anchor_rank4_match is not None
            and nhwc_mul_match is not None
            and reduce_sum_match is not None
            and str(nhwc_mul_match.group("expr")) == f"torch.mul({binary_anchor_rank4_match.group('lhs1')}, {binary_anchor_rank4_match.group('lhs0')})"
        ):
            indent = str(binary_anchor_rank4_match.group("indent"))
            lhs0 = str(binary_anchor_rank4_match.group("lhs0"))
            lhs1 = str(binary_anchor_rank4_match.group("lhs1"))
            a = str(binary_anchor_rank4_match.group("a"))
            b = str(binary_anchor_rank4_match.group("b"))
            n = int(binary_anchor_rank4_match.group("n"))
            h = int(binary_anchor_rank4_match.group("h"))
            w = int(binary_anchor_rank4_match.group("w"))
            c = int(binary_anchor_rank4_match.group("c"))
            mul_lhs = str(nhwc_mul_match.group("lhs"))
            reduce_lhs = str(reduce_sum_match.group("lhs"))
            lines[index] = (
                f"{indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor({a}, {b}, [{n}, {c}, {h}, {w}])"
            )
            lines[index + 1] = (
                f"{nhwc_mul_match.group('indent')}{mul_lhs} = _align_tensor_to_target_shape("
                f"torch.mul({lhs1}, {lhs0}), [{n}, {c}, {h}, {w}])"
            )
            lines[reduce_sum_index] = (
                f"{reduce_sum_match.group('indent')}{reduce_lhs} = _reduce_sum("
                f"{mul_lhs}, _normalize_axes([1], {mul_lhs}.ndim), True)"
            )
            cf_aliases.add(mul_lhs)
            singleton_cf_vars.add(reduce_lhs)
            changed = True
            index = reduce_sum_index + 1
            continue
        resize_bad_target_match = apply_resize_cf_bad_target_re.match(lines[index])
        if resize_bad_target_match is not None:
            input_name = str(resize_bad_target_match.group("input"))
            lhs = str(resize_bad_target_match.group("lhs"))
            function_end = _function_end_index(index)
            alias_match = (
                re.fullmatch(
                    r"(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = (?P<src>[A-Za-z0-9_]+)",
                    lines[index + 1],
                )
                if index + 1 < function_end
                else None
            )
            alias_consumer_name = lhs
            alias_line_index = None
            if alias_match is not None and str(alias_match.group("src")) == lhs:
                alias_consumer_name = str(alias_match.group("lhs"))
                alias_line_index = index + 1
            saw_future_use = False
            only_binary_cf_consumers = True
            consumer_shapes: set[Tuple[int, int, int, int]] = set()
            tracked_names = {lhs, alias_consumer_name}
            for future_index in range(index + 1, function_end):
                if alias_line_index is not None and future_index == alias_line_index:
                    continue
                future_line = lines[future_index]
                if (
                    re.search(rf"\b{re.escape(lhs)}\b", future_line) is None
                    and re.search(rf"\b{re.escape(alias_consumer_name)}\b", future_line) is None
                ):
                    continue
                saw_future_use = True
                binary_cf_consumer_match = binary_cf_consumer_re.match(future_line)
                if (
                    binary_cf_consumer_match is None
                    or not tracked_names.intersection(
                        {
                            str(binary_cf_consumer_match.group("a")),
                            str(binary_cf_consumer_match.group("b")),
                        }
                    )
                ):
                    only_binary_cf_consumers = False
                    break
                consumer_shapes.add(
                    (
                        int(binary_cf_consumer_match.group("n")),
                        int(binary_cf_consumer_match.group("c")),
                        int(binary_cf_consumer_match.group("h")),
                        int(binary_cf_consumer_match.group("w")),
                    )
                )
            if saw_future_use and only_binary_cf_consumers and consumer_shapes == {
                (
                    int(resize_bad_target_match.group("n")),
                    int(resize_bad_target_match.group("h")),
                    int(resize_bad_target_match.group("w")),
                    int(resize_bad_target_match.group("c")),
                )
            }:
                index += 1
                continue
            if not (
                _is_known_cf_name(input_name, singleton_cf_seeds)
                or input_name.endswith("_nhwc_cf")
            ):
                indent = str(resize_bad_target_match.group("indent"))
                out_h = int(resize_bad_target_match.group("out_h"))
                out_w = int(resize_bad_target_match.group("out_w"))
                n = int(resize_bad_target_match.group("n"))
                h = int(resize_bad_target_match.group("h"))
                w = int(resize_bad_target_match.group("w"))
                c = int(resize_bad_target_match.group("c"))
                lines[index] = (
                    f"{indent}{lhs} = _apply_resize("
                    f"{input_name}, [{out_h}, {out_w}], method='{resize_bad_target_match.group('method')}', "
                    f"target_shape=[{n}, {h}, {w}, {c}]"
                    f"{resize_bad_target_match.group('rest')}, channel_last=True)"
                )
                changed = True
        current_line = lines[index]
        rank3_const_pad_match = (
            _cached_regex_match("rank3_const_pad_re", rank3_const_pad_re, current_line)
            if "F.pad(" in current_line and "mode='constant'" in current_line
            else None
        )
        if rank3_const_pad_match is not None:
            lhs = str(rank3_const_pad_match.group("lhs"))
            for future_index in range(index + 1, min(len(lines), index + 4)):
                future_assign = _parse_simple_assignment_line(lines[future_index])
                future_concat_args = (
                    _parse_apply_concat_inputs_axis_and_shape(future_assign[2])
                    if future_assign is not None
                    else None
                )
                if future_concat_args is None:
                    continue
                input_names = [name.strip() for name in future_concat_args[0] if name.strip()]
                if lhs not in input_names:
                    continue
                target_shape_values = future_concat_args[2] or []
                if future_concat_args[1] != 2 or len(target_shape_values) != 3:
                    continue
                indent = str(rank3_const_pad_match.group("indent"))
                input_name = str(rank3_const_pad_match.group("input"))
                pad0 = int(rank3_const_pad_match.group("pad0"))
                pad1 = int(rank3_const_pad_match.group("pad1"))
                value = str(rank3_const_pad_match.group("value"))
                lines[index] = (
                    f"{indent}{lhs} = F.pad({input_name}, [0, 0, {pad0}, {pad1}], mode='constant', value={value})"
                )
                changed = True
                break
        current_line = lines[index]
        rank4_const_pad_match = (
            _cached_regex_match("rank4_const_pad_re", rank4_const_pad_re, current_line)
            if "F.pad(" in current_line and "mode='constant'" in current_line
            else None
        )
        if rank4_const_pad_match is not None:
            lhs = str(rank4_const_pad_match.group("lhs"))
            for future_index in range(index + 1, min(len(lines), index + 4)):
                future_pool_match = _parse_apply_pool2d_assign_with_shape(lines[future_index])
                if (
                    future_pool_match is None
                    or str(future_pool_match[2]) != lhs
                    or not bool(future_pool_match[6])
                ):
                    continue
                target_shape_values = [int(value) for value in list(future_pool_match[4])]
                if len(target_shape_values) != 4:
                    continue
                indent = str(rank4_const_pad_match.group("indent"))
                input_name = str(rank4_const_pad_match.group("input"))
                pad0 = int(rank4_const_pad_match.group("pad0"))
                pad1 = int(rank4_const_pad_match.group("pad1"))
                pad2 = int(rank4_const_pad_match.group("pad2"))
                pad3 = int(rank4_const_pad_match.group("pad3"))
                value = str(rank4_const_pad_match.group("value"))
                lines[index] = (
                    f"{indent}{lhs} = F.pad("
                    f"{input_name}, [0, 0, {pad0}, {pad1}, {pad2}, {pad3}], "
                    f"mode='constant', value={value})"
                )
                changed = True
                break
        current_line = lines[index]
        rank4_const_pad6_match = (
            _cached_regex_match("rank4_const_pad6_re", rank4_const_pad6_re, current_line)
            if "F.pad(" in current_line and "mode='constant'" in current_line
            else None
        )
        if rank4_const_pad6_match is not None:
            input_name = str(rank4_const_pad6_match.group("input"))
            input_name_looks_channel_last = _tensor_name_suggests_channel_last_layout_for_codegen(input_name)
            next_line = lines[index + 1] if index + 1 < len(lines) else ""
            next_pool2d_match = _parse_apply_pool2d_assign_with_shape(next_line)
            if (
                next_pool2d_match is not None
                and str(next_pool2d_match[2]) == str(rank4_const_pad6_match.group("lhs"))
                and bool(next_pool2d_match[6])
                and not input_name_looks_channel_last
            ):
                target_shape_values = [int(value) for value in list(next_pool2d_match[4])]
                if (
                    len(target_shape_values) == 4
                    and target_shape_values[1] > target_shape_values[2]
                    and target_shape_values[1] > target_shape_values[3]
                ):
                    indent = str(rank4_const_pad6_match.group("indent"))
                    lhs = str(rank4_const_pad6_match.group("lhs"))
                    pad0 = int(rank4_const_pad6_match.group("pad0"))
                    pad1 = int(rank4_const_pad6_match.group("pad1"))
                    pad2 = int(rank4_const_pad6_match.group("pad2"))
                    pad3 = int(rank4_const_pad6_match.group("pad3"))
                    value = str(rank4_const_pad6_match.group("value"))
                    lines[index] = (
                        f"{indent}{lhs} = F.pad("
                        f"{input_name}, [{pad0}, {pad1}, {pad2}, {pad3}], "
                        f"mode='constant', value={value})"
                    )
                    cf_pad_aliases.add(lhs)
                    exact_pool_shape = _model_ir_exact_shape(str(next_pool2d_match[1]))
                    target_shape_literal = (
                        repr(exact_pool_shape)
                        if exact_pool_shape is not None and len(exact_pool_shape) == 4
                        else (
                            repr(target_shape_values)
                            if (
                                len(target_shape_values) == 4
                                and int(target_shape_values[1]) != int(target_shape_values[2])
                                and int(target_shape_values[2]) == int(target_shape_values[3])
                            )
                            else repr(target_shape_values)
                        )
                    )
                    next_pool_indent, next_pool_lhs, _, next_pool_rest, _, next_pool_is_max, _ = next_pool2d_match
                    lines[index + 1] = (
                        f"{next_pool_indent}{next_pool_lhs} = _apply_pool2d("
                        f"{lhs}, {next_pool_rest}, "
                        f"target_shape={target_shape_literal}, "
                        f"is_max_pool={next_pool_is_max}, channel_last=False)"
                    )
                    cf_aliases.add(str(next_pool_lhs))
                    changed = True
                    continue
            if _is_known_cf_name(input_name, singleton_cf_seeds) and not input_name_looks_channel_last:
                indent = str(rank4_const_pad6_match.group("indent"))
                lhs = str(rank4_const_pad6_match.group("lhs"))
                pad0 = int(rank4_const_pad6_match.group("pad0"))
                pad1 = int(rank4_const_pad6_match.group("pad1"))
                pad2 = int(rank4_const_pad6_match.group("pad2"))
                pad3 = int(rank4_const_pad6_match.group("pad3"))
                value = str(rank4_const_pad6_match.group("value"))
                lines[index] = (
                    f"{indent}{lhs} = F.pad("
                    f"{input_name}, [{pad0}, {pad1}, {pad2}, {pad3}], "
                    f"mode='constant', value={value})"
                )
                cf_pad_aliases.add(lhs)
                changed = True
        current_line = lines[index]
        aligned_rank4_const_pad6_match = (
            _cached_regex_match(
                "aligned_rank4_const_pad6_re",
                aligned_rank4_const_pad6_re,
                current_line,
            )
            if "F.pad(" in current_line and "_align_tensor_to_target_shape(" in current_line
            else None
        )
        if aligned_rank4_const_pad6_match is not None:
            input_name = str(aligned_rank4_const_pad6_match.group("input"))
            if (
                _is_known_cf_name(input_name, singleton_cf_seeds)
                and "_nhwc" not in input_name
                and not _tensor_name_suggests_channel_last_layout_for_codegen(input_name)
            ):
                indent = str(aligned_rank4_const_pad6_match.group("indent"))
                lhs = str(aligned_rank4_const_pad6_match.group("lhs"))
                pad0 = int(aligned_rank4_const_pad6_match.group("pad0"))
                pad1 = int(aligned_rank4_const_pad6_match.group("pad1"))
                pad2 = int(aligned_rank4_const_pad6_match.group("pad2"))
                pad3 = int(aligned_rank4_const_pad6_match.group("pad3"))
                value = str(aligned_rank4_const_pad6_match.group("value"))
                lines[index] = (
                    f"{indent}{lhs} = F.pad("
                    f"{input_name}, [{pad0}, {pad1}, {pad2}, {pad3}], "
                    f"mode='constant', value={value})"
                )
                cf_pad_aliases.add(lhs)
                next_line = lines[index + 1] if index + 1 < len(lines) else ""
                next_pool2d_match = (
                    _cached_regex_match("apply_pool2d_re", apply_pool2d_re, next_line)
                    if "_apply_pool2d(" in next_line
                    else None
                )
                if (
                    next_pool2d_match is not None
                    and str(next_pool2d_match.group("input")) == lhs
                    and str(next_pool2d_match.group("channel_last")) == "True"
                ):
                    target_shape_values = [
                        int(value.strip())
                        for value in str(next_pool2d_match.group("shape")).split(",")
                        if value.strip()
                    ]
                    if len(target_shape_values) == 4:
                        exact_pool_shape = _model_ir_exact_shape(str(next_pool2d_match.group("lhs")))
                        cf_target_shape = (
                            [int(v) for v in list(exact_pool_shape)]
                            if exact_pool_shape is not None and len(exact_pool_shape) == 4
                            else (
                                [int(v) for v in list(target_shape_values)]
                                if (
                                    int(target_shape_values[1]) != int(target_shape_values[2])
                                    and int(target_shape_values[2]) == int(target_shape_values[3])
                                )
                                else [
                                    int(target_shape_values[0]),
                                    int(target_shape_values[3]),
                                    int(target_shape_values[1]),
                                    int(target_shape_values[2]),
                                ]
                            )
                        )
                        lines[index + 1] = (
                            f"{next_pool2d_match.group('indent')}{next_pool2d_match.group('lhs')} = _apply_pool2d("
                            f"{lhs}, {next_pool2d_match.group('rest')}, "
                            f"target_shape={repr(cf_target_shape)}, "
                            f"is_max_pool={next_pool2d_match.group('is_max')}, channel_last=False)"
                        )
                        cf_aliases.add(str(next_pool2d_match.group("lhs")))
                changed = True
        current_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        concat_match = (
            _cached_regex_match("concat_re", concat_re, current_line)
            if "_apply_concat(" in current_line
            else None
        )
        concat_assign = _parse_simple_assignment_line(current_line)
        parsed_concat_args = (
            _parse_apply_concat_inputs_axis_and_shape(concat_assign[2])
            if concat_assign is not None and "_apply_concat(" in concat_assign[2]
            else None
        )
        next_split_match = (
            _cached_regex_match("generic_split_re", generic_split_re, next_line)
            if "torch.tensor_split(" in next_line
            else None
        )
        next_module_call_match = (
            _cached_regex_match("generic_module_call_re", generic_module_call_re, next_line)
            if " = self." in next_line and "(" in next_line
            else None
        )
        next_channel_last_gather_slice_assign = (
            _parse_channel_last_gather_slice_assign(next_line)
            if "[:, :, :, [" in next_line
            else None
        )
        gather_slice_matches: List[Tuple[int, Tuple[str, str, str]]] = []
        if concat_match is not None or parsed_concat_args is not None:
            concat_lhs = str(concat_assign[1] if concat_assign is not None else concat_match.group("lhs"))
            gather_slice_lookahead = index + 1
            while gather_slice_lookahead < len(lines):
                gather_slice_assign = _parse_channel_last_gather_slice_assign(lines[gather_slice_lookahead])
                if gather_slice_assign is None or str(gather_slice_assign[1]) != concat_lhs:
                    break
                gather_slice_matches.append((gather_slice_lookahead, gather_slice_assign))
                gather_slice_lookahead += 1
        current_line = lines[index]
        aligned_rank4_seed_match = (
            _cached_regex_match("aligned_nhwc_rank4_re", aligned_nhwc_rank4_re, current_line)
            if "_align_tensor_to_target_shape(" in current_line
            else None
        )
        if aligned_rank4_seed_match is not None:
            lhs = str(aligned_rank4_seed_match.group("lhs"))
            expr = str(aligned_rank4_seed_match.group("expr"))
            simple_source_expr = _is_simple_identifier_expr(expr)
            if simple_source_expr and _is_known_cf_name(expr, singleton_cf_vars):
                cf_materialized_alias_sources[lhs] = expr
                if simple_source_expr and _is_known_cf_name(expr, singleton_cf_vars):
                    function_end = _function_end_index(index)
                    saw_future_use = False
                    saw_non_return_future_use = False
                    for future_line in lines[index + 1 : function_end]:
                        if lhs not in _expr_identifier_tokens(future_line):
                            continue
                        saw_future_use = True
                        if not future_line.lstrip().startswith("return "):
                            saw_non_return_future_use = True
                            break
                    if not saw_future_use or not saw_non_return_future_use:
                        indent = str(aligned_rank4_seed_match.group("indent"))
                        lines[index] = f"{indent}{lhs} = {expr}"
                        cf_aliases.add(lhs)
                        if int(aligned_rank4_seed_match.group("c")) == 1:
                            singleton_cf_vars.add(lhs)
                        changed = True
                        aligned_rank4_seed_match = (
                            _cached_regex_match("aligned_nhwc_rank4_re", aligned_nhwc_rank4_re, lines[index])
                            if "_align_tensor_to_target_shape(" in lines[index]
                            else None
                        )
                        if aligned_rank4_seed_match is None:
                            continue
                if (
                    "_nhwc" not in lhs
                    and (
                        _expr_references_known_cf_identifier(expr, singleton_cf_vars)
                        or "_cf" in expr
                    )
                ):
                    target_last_dim = int(aligned_rank4_seed_match.group("c"))
                    target_mid0 = int(aligned_rank4_seed_match.group("h"))
                    target_mid1 = int(aligned_rank4_seed_match.group("w"))
                    if target_last_dim == 1 or target_last_dim < target_mid0 or target_last_dim < target_mid1:
                        cf_aliases.add(lhs)
                        if target_mid0 == 1:
                            singleton_cf_vars.add(lhs)
        if concat_match is not None or parsed_concat_args is not None:
            input_names = (
                [name.strip() for name in parsed_concat_args[0] if name.strip()]
                if parsed_concat_args is not None
                else [name.strip() for name in str(concat_match.group("inputs")).split(",") if name.strip()]
            )
            normalized_input_names = [
                source_name if source_name != name and _is_name_available_in_function(source_name, index) else name
                for name in input_names
                for source_name in [cf_materialized_alias_sources.get(name, name)]
            ]
            lhs = str(concat_assign[1] if concat_assign is not None else concat_match.group("lhs"))
            target_shape_values = (
                list(parsed_concat_args[2])
                if parsed_concat_args is not None and parsed_concat_args[2] is not None
                else [
                    int(value.strip())
                    for value in str(concat_match.group("shape")).split(",")
                    if value.strip()
                ]
            )
            target_looks_cf = (
                len(target_shape_values) == 4
                and target_shape_values[1] > target_shape_values[2]
                and target_shape_values[1] > target_shape_values[3]
            )
            should_rewrite_concat = (
                len(normalized_input_names) >= 2
                and all(_is_known_cf_name(name, singleton_cf_vars) for name in normalized_input_names)
            )
            if (
                not should_rewrite_concat
                and next_split_match is not None
                and len(target_shape_values) == 4
                and int(next_split_match.group("axis")) == 3
                and int(next_split_match.group("sections")) == target_shape_values[-1]
                and all(not _declares_channel_last_name(input_name) for input_name in normalized_input_names)
            ):
                should_rewrite_concat = True
            if (
                not should_rewrite_concat
                and len(target_shape_values) == 4
                and target_shape_values[1] < target_shape_values[2]
                and target_shape_values[1] < target_shape_values[3]
                and any(
                    _is_known_cf_name(name, singleton_cf_vars) or _declares_channel_last_name(name)
                    for name in normalized_input_names
                )
            ):
                should_rewrite_concat = True
            if (
                not should_rewrite_concat
                and target_looks_cf
                and next_module_call_match is not None
                and str(next_module_call_match.group("input")) == lhs
                and all(not _declares_channel_last_name(input_name) for input_name in normalized_input_names)
                and any(_is_known_cf_name(name, singleton_cf_vars) for name in normalized_input_names)
            ):
                should_rewrite_concat = True
            if (
                should_rewrite_concat
                and next_channel_last_gather_slice_assign is not None
                and str(next_channel_last_gather_slice_assign[1]) == lhs
            ):
                function_end = _function_end_index(index)
                supported_gather_slice_chain = len(gather_slice_matches) > 0
                for gather_slice_index, gather_slice_assign in gather_slice_matches:
                    gather_slice_lhs = str(gather_slice_assign[0])
                    gather_slice_supported = False
                    for future_index in range(gather_slice_index + 1, min(function_end, gather_slice_index + 10)):
                        permuted_use_match = permuted_cf_module_input_re.match(lines[future_index])
                        permuted_use_assign = _parse_permuted_cf_module_input_assign(lines[future_index])
                        if (
                            (
                                permuted_use_match is not None
                                or permuted_use_assign is not None
                            )
                            and (
                                str(permuted_use_match.group("src"))
                                if permuted_use_match is not None
                                else str(permuted_use_assign[3])
                            ) == gather_slice_lhs
                        ):
                            gather_slice_supported = True
                            break
                        future_assign = _parse_simple_assignment_line(lines[future_index])
                        future_concat_args = (
                            _parse_apply_concat_inputs_axis_and_shape(future_assign[2])
                            if future_assign is not None
                            else None
                        )
                        if future_concat_args is not None:
                            future_concat_inputs = {
                                name.strip() for name in future_concat_args[0] if name.strip()
                            }
                            if (
                                gather_slice_lhs in future_concat_inputs
                                and future_concat_args[1] == 3
                            ):
                                gather_slice_supported = True
                                break
                        future_legacy_assign = _parse_simple_assignment_line(lines[future_index])
                        future_legacy_concat_match = concat_re.match(lines[future_index])
                        future_legacy_concat_args = (
                            _parse_apply_concat_inputs_axis_and_shape(future_legacy_assign[2])
                            if future_legacy_assign is not None and "_apply_concat(" in future_legacy_assign[2]
                            else None
                        )
                        if future_legacy_concat_match is not None or future_legacy_concat_args is not None:
                            future_concat_inputs = {
                                name.strip()
                                for name in (
                                    future_legacy_concat_args[0]
                                    if future_legacy_concat_args is not None
                                    else str(future_legacy_concat_match.group("inputs")).split(",")
                                )
                                if name.strip()
                            }
                            if gather_slice_lhs in future_concat_inputs:
                                gather_slice_supported = True
                                break
                    if not gather_slice_supported:
                        supported_gather_slice_chain = False
                        break
                if not supported_gather_slice_chain:
                    should_rewrite_concat = False
            if should_rewrite_concat:
                indent = str(concat_assign[0] if concat_assign is not None else concat_match.group("indent"))
                lines[index] = f"{indent}{lhs} = torch.cat([{', '.join(normalized_input_names)}], dim=1)"
                cf_aliases.add(lhs)
                forced_cf_aliases.add(lhs)
                for gather_slice_index, gather_slice_assign in gather_slice_matches:
                    gather_slice_lhs = str(gather_slice_assign[0])
                    gather_slice_indent = re.match(r"^\s*", lines[gather_slice_index]).group(0)
                    gather_slice_indices = str(gather_slice_assign[2])
                    lines[gather_slice_index] = (
                        f"{gather_slice_indent}{gather_slice_lhs} = "
                        f"{lhs}[:, [{gather_slice_indices}], :, :]"
                    )
                    cf_aliases.add(gather_slice_lhs)
                    forced_cf_aliases.add(gather_slice_lhs)
                    gather_index_values = [
                        token.strip()
                        for token in gather_slice_indices.split(",")
                        if token.strip()
                    ]
                    if len(gather_index_values) == 1:
                        singleton_cf_vars.add(gather_slice_lhs)
                if next_split_match is not None and str(next_split_match.group("input")) == lhs and int(next_split_match.group("axis")) == 3:
                    lines[index + 1] = (
                        f"{next_split_match.group('indent')}{next_split_match.group('outputs')} = list(torch.tensor_split("
                        f"{lhs}, {next_split_match.group('sections')}, dim=_normalize_dim(1, {lhs}.ndim)))"
                    )
                    for output_name in [token.strip() for token in str(next_split_match.group("outputs")).split(",") if token.strip()]:
                        singleton_cf_vars.add(output_name)
                changed = True
            elif (
                next_module_call_match is not None
                and str(next_module_call_match.group("input")) == lhs
                and len(target_shape_values) == 4
                and target_looks_cf
                and any(_declares_channel_last_name(input_name) for input_name in normalized_input_names)
            ):
                indent = str(concat_assign[0] if concat_assign is not None else concat_match.group("indent"))
                n, c, h, w = target_shape_values
                lines[index] = (
                    f"{indent}{lhs} = _apply_concat([{', '.join(normalized_input_names)}], axis=3, "
                    f"target_shape=[{n}, {h}, {w}, {c}], fused='NONE')"
                )
                lines[index + 1] = (
                    f"{next_module_call_match.group('indent')}{next_module_call_match.group('lhs')} = "
                    f"self.{next_module_call_match.group('module')}(_torch_permute({lhs}, [0, 3, 1, 2]))"
                )
                changed = True
        current_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        aligned_then_split_match = (
            _cached_regex_match("aligned_nhwc_rank4_re", aligned_nhwc_rank4_re, current_line)
            if "_align_tensor_to_target_shape(" in current_line
            else None
        )
        next_split_match = (
            _cached_regex_match("generic_split_re", generic_split_re, next_line)
            if "torch.tensor_split(" in next_line
            else None
        )
        if (
            aligned_then_split_match is not None
            and next_split_match is not None
            and str(next_split_match.group("input")) == str(aligned_then_split_match.group("lhs"))
            and int(next_split_match.group("axis")) == 3
        ):
            expr = str(aligned_then_split_match.group("expr"))
            lhs = str(aligned_then_split_match.group("lhs"))
            sections = int(next_split_match.group("sections"))
            d1 = int(aligned_then_split_match.group("h"))
            d2 = int(aligned_then_split_match.group("w"))
            d3 = int(aligned_then_split_match.group("c"))
            expr_uses_cf = (
                _expr_references_known_cf_identifier(expr, singleton_cf_vars)
                or "_cf" in expr
                or _is_known_cf_name(lhs, singleton_cf_vars)
            )
            if (sections == d1 and sections != d3 and "_nhwc" not in lhs) or expr_uses_cf:
                if sections == d1 and sections != d3 and "_nhwc" not in lhs:
                    lines[index + 1] = (
                        f"{next_split_match.group('indent')}{next_split_match.group('outputs')} = list(torch.tensor_split("
                        f"{lhs}, {sections}, dim=_normalize_dim(1, {lhs}.ndim)))"
                    )
                    cf_aliases.add(lhs)
                    for output_name in [token.strip() for token in str(next_split_match.group("outputs")).split(",") if token.strip()]:
                        singleton_cf_vars.add(output_name)
                    changed = True
                elif sections == d3:
                    indent = str(aligned_then_split_match.group("indent"))
                    n = int(aligned_then_split_match.group("n"))
                    lines[index] = (
                        f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {d3}, {d1}, {d2}])"
                    )
                    lines[index + 1] = (
                        f"{next_split_match.group('indent')}{next_split_match.group('outputs')} = list(torch.tensor_split("
                        f"{lhs}, {sections}, dim=_normalize_dim(1, {lhs}.ndim)))"
                    )
                    cf_aliases.add(lhs)
                    for output_name in [token.strip() for token in str(next_split_match.group("outputs")).split(",") if token.strip()]:
                        singleton_cf_vars.add(output_name)
                    changed = True
        current_line = lines[index]
        next_line = lines[index + 1] if index + 1 < len(lines) else ""
        aligned_cf_resize_match = (
            _cached_regex_match("aligned_nhwc_rank4_re", aligned_nhwc_rank4_re, current_line)
            if "_align_tensor_to_target_shape(" in current_line
            else None
        )
        resize_match = (
            _cached_regex_match("apply_resize_nhwc_re", apply_resize_nhwc_re, next_line)
            if "_apply_resize(" in next_line
            else None
        )
        resize_cf_match = (
            _cached_regex_match("apply_resize_cf_re", apply_resize_cf_re, next_line)
            if "_apply_resize(" in next_line
            else None
        )
        resize_pair_match = resize_match if resize_match is not None else resize_cf_match
        lookahead_line = lines[index + 2] if index + 2 < len(lines) else ""
        lookahead_assign = _parse_simple_assignment_line(lookahead_line)
        concat_after_resize_match = (
            _cached_regex_match("concat_re", concat_re, lookahead_line)
            if "_apply_concat(" in lookahead_line
            else None
        )
        parsed_concat_after_resize = (
            _parse_apply_concat_inputs_axis_and_shape(lookahead_assign[2])
            if lookahead_assign is not None and "_apply_concat(" in lookahead_assign[2]
            else None
        )
        generic_cat_after_resize_match = (
            _cached_regex_match("generic_torch_cat_re", generic_torch_cat_re, lookahead_line)
            if "torch.cat(" in lookahead_line
            else None
        )
        parsed_torch_cat_after_resize = (
            _parse_torch_cat_inputs_and_dim(lookahead_assign[2])
            if lookahead_assign is not None and "torch.cat(" in lookahead_assign[2]
            else None
        )
        if (
            aligned_cf_resize_match is not None
            and resize_pair_match is not None
            and str(resize_pair_match.group("input")) == str(aligned_cf_resize_match.group("lhs"))
        ):
            expr = str(aligned_cf_resize_match.group("expr"))
            resize_lhs = str(resize_pair_match.group("lhs"))
            function_end = _function_end_index(index + 1)
            resize_binary_cf_shapes: set[Tuple[int, int, int, int]] = set()
            saw_resize_future_use = False
            resize_only_binary_cf_consumers = True
            for future_line in lines[index + 2 : function_end]:
                if re.search(rf"\b{re.escape(resize_lhs)}\b", future_line) is None:
                    continue
                saw_resize_future_use = True
                resize_binary_cf_consumer_match = binary_cf_consumer_re.match(future_line)
                if (
                    resize_binary_cf_consumer_match is None
                    or resize_lhs not in {
                        str(resize_binary_cf_consumer_match.group("a")),
                        str(resize_binary_cf_consumer_match.group("b")),
                    }
                ):
                    resize_only_binary_cf_consumers = False
                    break
                resize_binary_cf_shapes.add(
                    (
                        int(resize_binary_cf_consumer_match.group("n")),
                        int(resize_binary_cf_consumer_match.group("c")),
                        int(resize_binary_cf_consumer_match.group("h")),
                        int(resize_binary_cf_consumer_match.group("w")),
                    )
                )
            simple_binary_expr_match = simple_binary_expr_re.match(expr)
            if (
                simple_binary_expr_match is not None
                and _is_known_cf_name(str(simple_binary_expr_match.group("a")), singleton_cf_vars)
                and _is_known_cf_name(str(simple_binary_expr_match.group("b")), singleton_cf_vars)
                and saw_resize_future_use
                and resize_only_binary_cf_consumers
                and resize_binary_cf_shapes == {
                    (
                        int(aligned_cf_resize_match.group("n")),
                        int(aligned_cf_resize_match.group("c")),
                        int(resize_pair_match.group("out_h")),
                        int(resize_pair_match.group("out_w")),
                    )
                }
            ):
                indent = str(aligned_cf_resize_match.group("indent"))
                lhs = str(aligned_cf_resize_match.group("lhs"))
                n = int(aligned_cf_resize_match.group("n"))
                h = int(aligned_cf_resize_match.group("h"))
                w = int(aligned_cf_resize_match.group("w"))
                c = int(aligned_cf_resize_match.group("c"))
                out_h = int(resize_pair_match.group("out_h"))
                out_w = int(resize_pair_match.group("out_w"))
                lines[index] = (
                    f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
                )
                lines[index + 1] = (
                    f"{resize_pair_match.group('indent')}{resize_lhs} = _apply_resize("
                    f"{lhs}, [{out_h}, {out_w}], method='{resize_pair_match.group('method')}', "
                    f"target_shape=[{n}, {c}, {out_h}, {out_w}], "
                    f"align_corners={resize_pair_match.group('align')}, "
                    f"half_pixel_centers={resize_pair_match.group('hpc')}, channel_last=False)"
                )
                cf_aliases.add(lhs)
                cf_aliases.add(resize_lhs)
                changed = True
                index += 1
                continue
            if (
                "torch.mul(" in expr
                and (
                    _expr_references_known_cf_identifier(expr, singleton_cf_vars)
                    or (
                        simple_binary_expr_match is not None
                        and (
                            _is_known_cf_name(str(simple_binary_expr_match.group("a")), singleton_cf_vars)
                            or _is_known_cf_name(str(simple_binary_expr_match.group("b")), singleton_cf_vars)
                        )
                    )
                )
            ):
                indent = str(aligned_cf_resize_match.group("indent"))
                lhs = str(aligned_cf_resize_match.group("lhs"))
                n = int(aligned_cf_resize_match.group("n"))
                h = int(aligned_cf_resize_match.group("h"))
                w = int(aligned_cf_resize_match.group("w"))
                c = int(aligned_cf_resize_match.group("c"))
                out_h = int(resize_pair_match.group("out_h"))
                out_w = int(resize_pair_match.group("out_w"))
                lines[index] = (
                    f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
                )
                lines[index + 1] = (
                    f"{resize_pair_match.group('indent')}{resize_lhs} = _apply_resize("
                    f"{lhs}, [{out_h}, {out_w}], method='{resize_pair_match.group('method')}', "
                    f"target_shape=[{n}, {c}, {out_h}, {out_w}], "
                    f"align_corners={resize_pair_match.group('align')}, "
                    f"half_pixel_centers={resize_pair_match.group('hpc')}, channel_last=False)"
                )
                cf_aliases.add(lhs)
                cf_aliases.add(resize_lhs)
                if (
                    parsed_concat_after_resize is not None
                    and resize_lhs in {name.strip() for name in parsed_concat_after_resize[0] if name.strip()}
                ):
                    concat_inputs = [name.strip() for name in parsed_concat_after_resize[0] if name.strip()]
                    lines[index + 2] = (
                        f"{(lookahead_assign[0] if lookahead_assign is not None else concat_after_resize_match.group('indent'))}"
                        f"{(lookahead_assign[1] if lookahead_assign is not None else concat_after_resize_match.group('lhs'))} = "
                        f"torch.cat([{', '.join(concat_inputs)}], dim=1)"
                    )
                    cf_aliases.add(str(lookahead_assign[1] if lookahead_assign is not None else concat_after_resize_match.group("lhs")))
                elif (
                    parsed_torch_cat_after_resize is not None
                    and parsed_torch_cat_after_resize[1] != 1
                    and resize_lhs in {
                        name.strip() for name in parsed_torch_cat_after_resize[0] if name.strip()
                    }
                ):
                    cat_inputs = [
                        name.strip() for name in parsed_torch_cat_after_resize[0] if name.strip()
                    ]
                    lines[index + 2] = (
                        f"{(lookahead_assign[0] if lookahead_assign is not None else generic_cat_after_resize_match.group('indent'))}"
                        f"{(lookahead_assign[1] if lookahead_assign is not None else generic_cat_after_resize_match.group('lhs'))} = "
                        f"torch.cat([{', '.join(cat_inputs)}], dim=1)"
                    )
                    cf_aliases.add(str(lookahead_assign[1] if lookahead_assign is not None else generic_cat_after_resize_match.group("lhs")))
                else:
                    stage_return_index = None
                    for lookahead_index in range(index + 2, min(index + 8, len(lines))):
                        if lines[lookahead_index].strip() == "":
                            continue
                        stage_return_value = _parse_simple_return_identifier(lines[lookahead_index])
                        if stage_return_value is not None and str(stage_return_value) == resize_lhs:
                            stage_return_index = lookahead_index
                        break
                    if stage_return_index is not None:
                        stage_signature_index = None
                        for lookahead_index in range(stage_return_index + 1, min(stage_return_index + 6, len(lines))):
                            if lines[lookahead_index].startswith("    def "):
                                stage_signature_index = lookahead_index
                                break
                        if (
                            stage_signature_index is not None
                            and re.search(rf"\b{re.escape(resize_lhs)}\b", lines[stage_signature_index]) is not None
                        ):
                            for lookahead_index in range(stage_signature_index + 1, min(stage_signature_index + 8, len(lines))):
                                if lines[lookahead_index].startswith("    def "):
                                    break
                                if lines[lookahead_index].strip() == "":
                                    continue
                                stage_cat_assign = _parse_simple_assignment_line(lines[lookahead_index])
                                parsed_stage_cat = (
                                    _parse_torch_cat_inputs_and_dim(stage_cat_assign[2])
                                    if stage_cat_assign is not None
                                    else None
                                )
                                if (
                                    parsed_stage_cat is not None
                                    and parsed_stage_cat[1] in {1, 3}
                                    and resize_lhs in {
                                        name.strip()
                                        for name in parsed_stage_cat[0]
                                        if name.strip()
                                    }
                                ):
                                    cf_aliases.add(resize_lhs)
                                    cf_aliases.add(str(stage_cat_assign[1]))
                                    break
                changed = True
        current_line = lines[index]
        aligned_nhwc_rank4_match = (
            _cached_regex_match("aligned_nhwc_rank4_re", aligned_nhwc_rank4_re, current_line)
            if "_align_tensor_to_target_shape(" in current_line
            else None
        )
        if aligned_nhwc_rank4_match is not None:
            lhs = str(aligned_nhwc_rank4_match.group("lhs"))
            expr = str(aligned_nhwc_rank4_match.group("expr"))
            n = int(aligned_nhwc_rank4_match.group("n"))
            h = int(aligned_nhwc_rank4_match.group("h"))
            w = int(aligned_nhwc_rank4_match.group("w"))
            c = int(aligned_nhwc_rank4_match.group("c"))
            next_argmax_assign = _parse_argmax_assign(lines[index + 1]) if index + 1 < len(lines) else None
            cf_bn_const_expr_match = (
                cf_bn_const_expr_re.fullmatch(expr)
                if "self." in expr and ("torch.mul(" in expr or "torch.add(" in expr)
                else None
            )
            cf_permute_source_match = (
                cf_permute_source_re.fullmatch(expr)
                if ".permute(0, 2, 3, 1).contiguous()" in expr
                else None
            )
            if (
                cf_permute_source_match is not None
                and next_argmax_assign is not None
                and str(next_argmax_assign[2]) == lhs
                and int(next_argmax_assign[3]) == 3
            ):
                src = str(cf_permute_source_match.group("src"))
                indent = str(aligned_nhwc_rank4_match.group("indent"))
                argmax_indent = str(next_argmax_assign[0])
                argmax_lhs = str(next_argmax_assign[1])
                argmax_keepdim = str(next_argmax_assign[4])
                lines[index] = f"{indent}{lhs} = {src}"
                lines[index + 1] = (
                    f"{argmax_indent}{argmax_lhs} = "
                    f"torch.argmax({lhs}, dim=_normalize_dim(1, {lhs}.ndim), "
                    f"keepdim={argmax_keepdim}).to(dtype=torch.int64)"
                )
                cf_aliases.add(lhs)
                changed = True
                index += 2
                continue
            if cf_bn_const_expr_match is not None:
                source_name = str(cf_bn_const_expr_match.group("input"))
                const_attr = str(cf_bn_const_expr_match.group("const_attr"))
                source_is_cf = (
                    _is_known_cf_name(source_name, singleton_cf_vars)
                    or source_name in cf_aliases
                    or source_name.endswith("_cf")
                    or source_name.endswith("_out_cf")
                )
                if source_is_cf and (
                    "BatchNormalization" in const_attr
                    or "batch_normalization" in const_attr
                ):
                    resolved_source_shape = _find_recent_rank4_shape(source_name, index)
                    if resolved_source_shape is None and model_ir is not None:
                        resolved_source_shape = _tensor_exact_static_shape_list_for_model_ir(
                            model_ir=model_ir,
                            tensor_name=source_name,
                        )
                    target_shape = [n, h, w, c]
                    should_rewrite_to_cf = False
                    buffer_spec = buffer_specs.get(const_attr, None)
                    buffer_channel_count = None
                    if buffer_spec is not None:
                        _, source_shape, _, _ = buffer_spec
                        non_singleton_dims = [int(v) for v in source_shape if int(v) != 1]
                        if len(non_singleton_dims) == 1:
                            buffer_channel_count = int(non_singleton_dims[0])
                    if resolved_source_shape is not None and len(resolved_source_shape) == 4:
                        resolved_source_shape = [int(v) for v in resolved_source_shape]
                        if target_shape == resolved_source_shape:
                            cf_aliases.add(lhs)
                            if resolved_source_shape[1] == 1:
                                singleton_cf_vars.add(lhs)
                            index += 1
                            continue
                        should_rewrite_to_cf = target_shape == [
                            resolved_source_shape[0],
                            resolved_source_shape[2],
                            resolved_source_shape[3],
                            resolved_source_shape[1],
                        ]
                    elif buffer_channel_count is not None:
                        if target_shape == [n, buffer_channel_count, h, w]:
                            cf_aliases.add(lhs)
                            if buffer_channel_count == 1:
                                singleton_cf_vars.add(lhs)
                            index += 1
                            continue
                        should_rewrite_to_cf = target_shape == [n, h, w, buffer_channel_count]
                    if not should_rewrite_to_cf:
                        index += 1
                        continue
                    indent = str(aligned_nhwc_rank4_match.group("indent"))
                    lines[index] = (
                        f"{indent}{lhs} = _align_tensor_to_target_shape("
                        f"torch.{cf_bn_const_expr_match.group('op')}("
                        f"{source_name}, torch.reshape(self.{const_attr}, [1, {c}, 1, 1])), "
                        f"[{n}, {c}, {h}, {w}])"
                    )
                    cf_aliases.add(lhs)
                    if c == 1:
                        singleton_cf_vars.add(lhs)
                    changed = True
                    continue
            simple_aligned_binary_expr_match = (
                simple_binary_expr_re.match(expr)
                if "torch." in expr and "(" in expr and ")" in expr
                else None
            )
            if simple_aligned_binary_expr_match is not None:
                arg_a = str(simple_aligned_binary_expr_match.group("a"))
                arg_b = str(simple_aligned_binary_expr_match.group("b"))
                arg_a_is_cf = (
                    _is_known_cf_name(arg_a, singleton_cf_vars)
                    or arg_a in cf_aliases
                    or arg_a.endswith("_cf")
                    or arg_a.endswith("_out_cf")
                    or (arg_a.endswith("_in") and not arg_a.endswith("_in_nhwc"))
                )
                arg_b_is_cf = (
                    _is_known_cf_name(arg_b, singleton_cf_vars)
                    or arg_b in cf_aliases
                    or arg_b.endswith("_cf")
                    or arg_b.endswith("_out_cf")
                    or (arg_b.endswith("_in") and not arg_b.endswith("_in_nhwc"))
                )
                if arg_a_is_cf and arg_b_is_cf:
                    resolved_arg_a_shape = _find_recent_rank4_shape(arg_a, index)
                    resolved_arg_b_shape = _find_recent_rank4_shape(arg_b, index)
                    if resolved_arg_a_shape is None and model_ir is not None:
                        resolved_arg_a_shape = _tensor_exact_static_shape_list_for_model_ir(
                            model_ir=model_ir,
                            tensor_name=arg_a,
                        )
                    if resolved_arg_b_shape is None and model_ir is not None:
                        resolved_arg_b_shape = _tensor_exact_static_shape_list_for_model_ir(
                            model_ir=model_ir,
                            tensor_name=arg_b,
                        )
                    target_shape = [n, h, w, c]
                    common_shape = None
                    if (
                        resolved_arg_a_shape is not None
                        and resolved_arg_b_shape is not None
                        and len(resolved_arg_a_shape) == 4
                        and len(resolved_arg_b_shape) == 4
                        and [int(v) for v in resolved_arg_a_shape] == [int(v) for v in resolved_arg_b_shape]
                    ):
                        common_shape = [int(v) for v in resolved_arg_a_shape]
                    elif resolved_arg_a_shape is not None and len(resolved_arg_a_shape) == 4:
                        common_shape = [int(v) for v in resolved_arg_a_shape]
                    elif resolved_arg_b_shape is not None and len(resolved_arg_b_shape) == 4:
                        common_shape = [int(v) for v in resolved_arg_b_shape]
                    if common_shape is not None:
                        if target_shape == common_shape:
                            cf_aliases.add(lhs)
                            if common_shape[1] == 1:
                                singleton_cf_vars.add(lhs)
                            index += 1
                            continue
                        if target_shape == [
                            common_shape[0],
                            common_shape[2],
                            common_shape[3],
                            common_shape[1],
                        ]:
                            indent = str(aligned_nhwc_rank4_match.group("indent"))
                            lines[index] = (
                                f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, "
                                f"[{common_shape[0]}, {common_shape[1]}, {common_shape[2]}, {common_shape[3]}])"
                            )
                            cf_aliases.add(lhs)
                            if common_shape[1] == 1:
                                singleton_cf_vars.add(lhs)
                            changed = True
                            continue
                    next_aligned_match = (
                        aligned_nhwc_rank4_re.match(lines[index + 1])
                        if index + 1 < len(lines)
                        else None
                    )
                    next_expr_match = (
                        re.fullmatch(
                            r"torch\.(?P<op>mul|add)\((?P<input>[A-Za-z0-9_]+), self\.(?P<const_attr>[A-Za-z0-9_]+)\)",
                            str(next_aligned_match.group("expr")),
                        )
                        if next_aligned_match is not None
                        else None
                    )
                    if (
                        common_shape is None
                        and next_expr_match is not None
                        and str(next_expr_match.group("input")) == lhs
                    ):
                        next_buffer_spec = buffer_specs.get(str(next_expr_match.group("const_attr")), None)
                        next_channel_count = None
                        if next_buffer_spec is not None:
                            _, next_source_shape, _, _ = next_buffer_spec
                            next_non_singleton_dims = [
                                int(v) for v in next_source_shape if int(v) != 1
                            ]
                            if len(next_non_singleton_dims) == 1:
                                next_channel_count = int(next_non_singleton_dims[0])
                        if next_channel_count is not None and next_channel_count == c:
                            indent = str(aligned_nhwc_rank4_match.group("indent"))
                            lines[index] = (
                                f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, "
                                f"[{n}, {c}, {h}, {w}])"
                            )
                            cf_aliases.add(lhs)
                            if c == 1:
                                singleton_cf_vars.add(lhs)
                            changed = True
                            continue
            if _expr_references_known_cf_identifier(expr, singleton_cf_vars) or "_cf" in expr:
                future_cf_spatial_consumer = False
                for lookahead in range(index + 1, min(len(lines), index + 80)):
                    lookahead_pool_assign = _parse_apply_pool2d_assign_with_shape(lines[lookahead])
                    if (
                        lookahead_pool_assign is not None
                        and str(lookahead_pool_assign[2]) == lhs
                        and bool(lookahead_pool_assign[6])
                    ):
                        future_cf_spatial_consumer = True
                        break
                    lookahead_mean_assign = _parse_rank4_mean_assign(lines[lookahead])
                    if (
                        lookahead_mean_assign is not None
                        and str(lookahead_mean_assign[2]) == lhs
                        and int(lookahead_mean_assign[3]) == 1
                        and int(lookahead_mean_assign[4]) == 2
                    ):
                        future_cf_spatial_consumer = True
                        break
                if future_cf_spatial_consumer:
                    indent = str(aligned_nhwc_rank4_match.group("indent"))
                    rewritten_line = (
                        f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
                    )
                    alias_added = lhs not in cf_aliases
                    singleton_added = c == 1 and lhs not in singleton_cf_vars
                    lines[index] = rewritten_line
                    cf_aliases.add(lhs)
                    if c == 1:
                        singleton_cf_vars.add(lhs)
                    if current_line != rewritten_line or alias_added or singleton_added:
                        changed = True
                    index += 1
                    continue
                next_line = lines[index + 1] if index + 1 < len(lines) else ""
                next_permute_assign = (
                    _parse_output_back_permute_assign(next_line)
                    if next_line
                    else None
                )
                if next_permute_assign is not None and str(next_permute_assign[2]) == lhs:
                    indent = str(aligned_nhwc_rank4_match.group("indent"))
                    rewritten_line = (
                        f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
                    )
                    alias_added = lhs not in cf_aliases
                    singleton_added = c == 1 and lhs not in singleton_cf_vars
                    lines[index] = rewritten_line
                    cf_aliases.add(lhs)
                    if c == 1:
                        singleton_cf_vars.add(lhs)
                    if current_line != rewritten_line or alias_added or singleton_added:
                        changed = True
                else:
                    next_line = lines[index + 1] if index + 1 < len(lines) else ""
                    next_relu_same_lhs = next_line.strip() == f"{lhs} = torch.relu({lhs})"
                    permuted_conv_use_count = 0
                    lookahead_start = index + 2 if next_relu_same_lhs else index + 1
                    for lookahead in range(lookahead_start, min(index + 7, len(lines))):
                        permuted_use_match = permuted_cf_module_input_re.match(lines[lookahead])
                        permuted_use_assign = _parse_permuted_cf_module_input_assign(lines[lookahead])
                        if (
                            (
                                permuted_use_match is not None
                                or permuted_use_assign is not None
                            )
                            and (
                                str(permuted_use_match.group("src"))
                                if permuted_use_match is not None
                                else str(permuted_use_assign[3])
                            ) == lhs
                        ):
                            permuted_conv_use_count += 1
                    if next_relu_same_lhs and permuted_conv_use_count >= 1:
                        indent = str(aligned_nhwc_rank4_match.group("indent"))
                        lines[index] = (
                            f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
                        )
                        cf_aliases.add(lhs)
                        changed = True
                        if c == 1:
                            singleton_cf_vars.add(lhs)
            next_reduce_sum_assign = _parse_reduce_sum_assign(lines[index + 1]) if index + 1 < len(lines) else None
            if (
                next_reduce_sum_assign is not None
                and str(next_reduce_sum_assign[2]) == lhs
                and int(next_reduce_sum_assign[3]) == 3
                and str(next_reduce_sum_assign[4]) == "True"
                and (
                    _expr_references_known_cf_identifier(expr, singleton_cf_vars)
                    or "_cf" in expr
                )
            ):
                indent = str(aligned_nhwc_rank4_match.group("indent"))
                reduce_indent = str(next_reduce_sum_assign[0])
                reduce_lhs = str(next_reduce_sum_assign[1])
                lines[index] = (
                    f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{n}, {c}, {h}, {w}])"
                )
                lines[index + 1] = (
                    f"{reduce_indent}{reduce_lhs} = _reduce_sum("
                    f"{lhs}, _normalize_axes([1], {lhs}.ndim), True)"
                )
                cf_aliases.add(lhs)
                singleton_cf_vars.add(reduce_lhs)
                changed = True
        permuted_cf_module_input_match = permuted_cf_module_input_re.match(lines[index])
        permuted_cf_module_input_assign = _parse_permuted_cf_module_input_assign(lines[index])
        if permuted_cf_module_input_match is not None or permuted_cf_module_input_assign is not None:
            src = (
                str(permuted_cf_module_input_match.group("src"))
                if permuted_cf_module_input_match is not None
                else str(permuted_cf_module_input_assign[3])
            )
            recent_rank4_shape = _find_recent_rank4_shape(src, index)
            if _is_known_cf_name(src, singleton_cf_vars) or (
                recent_rank4_shape is not None
                and len(recent_rank4_shape) == 4
                and int(recent_rank4_shape[1]) == 1
            ):
                indent = (
                    str(permuted_cf_module_input_match.group("indent"))
                    if permuted_cf_module_input_match is not None
                    else str(permuted_cf_module_input_assign[0])
                )
                lhs = (
                    str(permuted_cf_module_input_match.group("lhs"))
                    if permuted_cf_module_input_match is not None
                    else str(permuted_cf_module_input_assign[1])
                )
                module = (
                    str(permuted_cf_module_input_match.group("module"))
                    if permuted_cf_module_input_match is not None
                    else str(permuted_cf_module_input_assign[2])
                )
                lines[index] = f"{indent}{lhs} = {module}({src})"
                if (
                    recent_rank4_shape is not None
                    and len(recent_rank4_shape) == 4
                    and int(recent_rank4_shape[1]) == 1
                ):
                    singleton_cf_vars.add(src)
                changed = True
        output_back_permute_assign = _parse_output_back_permute_assign(lines[index])
        if output_back_permute_assign is not None:
            src = str(output_back_permute_assign[2])
            if _is_known_cf_name(src, singleton_cf_vars):
                indent = str(output_back_permute_assign[0])
                lhs = str(output_back_permute_assign[1])
                lines[index] = f"{indent}{lhs} = {src}"
                cf_aliases.add(lhs)
                changed = True
        direct_gather_slice_assign = _parse_channel_last_gather_slice_assign(lines[index])
        if direct_gather_slice_assign is not None:
            input_name = str(direct_gather_slice_assign[1])
            if (
                _is_known_cf_name(input_name, singleton_cf_vars)
                or input_name in cf_aliases
                or input_name.endswith("_cf")
                or input_name.endswith("_out_cf")
            ):
                indent = re.match(r"^\s*", lines[index]).group(0)
                lhs = str(direct_gather_slice_assign[0])
                indices = str(direct_gather_slice_assign[2])
                lines[index] = f"{indent}{lhs} = {input_name}[:, [{indices}], :, :]"
                cf_aliases.add(lhs)
                if len([token for token in indices.split(",") if token.strip()]) == 1:
                    singleton_cf_vars.add(lhs)
                changed = True
                index += 1
                continue
        softmax_assign = _parse_raw_apply_softmax_assign(lines[index])
        next_output_permute_assign = _parse_output_back_permute_assign(lines[index + 1]) if index + 1 < len(lines) else None
        if softmax_assign is not None:
            input_name = str(softmax_assign[2])
            resolved_input_shape = _find_recent_rank4_shape(input_name, index)
            if resolved_input_shape is None and model_ir is not None:
                resolved_input_shape = _tensor_exact_static_shape_list_for_model_ir(
                    model_ir=model_ir,
                    tensor_name=input_name,
                )
            target_shape_values = [int(v) for v in softmax_assign[5]]
            if (
                (
                    _is_known_cf_name(input_name, singleton_cf_vars)
                    or (
                        resolved_input_shape is not None
                        and len(resolved_input_shape) == 4
                        and not input_name.endswith("_nhwc")
                        and not input_name.endswith("_out_nhwc")
                    )
                )
                and (input_name.endswith("_cf") or input_name.endswith("_out_cf"))
                and int(softmax_assign[3]) == 3
                and resolved_input_shape is not None
                and len(resolved_input_shape) == 4
                and [int(v) for v in resolved_input_shape] == target_shape_values
            ):
                indent = str(softmax_assign[0])
                lhs = str(softmax_assign[1])
                lines[index] = (
                    f"{indent}{lhs} = _apply_softmax("
                    f"{input_name}, axis=1, beta={softmax_assign[4]}, "
                    f"target_shape=[{target_shape_values[0]}, {target_shape_values[1]}, "
                    f"{target_shape_values[2]}, {target_shape_values[3]}])"
                )
                cf_aliases.add(lhs)
                changed = True
                softmax_assign = _parse_raw_apply_softmax_assign(lines[index])
            elif (
                (
                    _is_known_cf_name(input_name, singleton_cf_vars)
                    or input_name in cf_aliases
                    or input_name.endswith("_cf")
                    or input_name.endswith("_out_cf")
                )
                and int(softmax_assign[3]) == 3
                and resolved_input_shape is not None
                and len(resolved_input_shape) == 4
                and [int(v) for v in resolved_input_shape] == [
                    target_shape_values[0],
                    target_shape_values[3],
                    target_shape_values[1],
                    target_shape_values[2],
                ]
            ):
                indent = str(softmax_assign[0])
                lhs = str(softmax_assign[1])
                n, h, w, c = target_shape_values
                lines[index] = (
                    f"{indent}{lhs} = _apply_softmax("
                    f"{input_name}, axis=1, beta={softmax_assign[4]}, "
                    f"target_shape=[{n}, {c}, {h}, {w}])"
                )
                cf_aliases.add(lhs)
                changed = True
                next_reduce_max_assign = (
                    _parse_reduce_max_assign(lines[index + 1])
                    if index + 1 < len(lines)
                    else None
                )
                next_sub_assign = (
                    _parse_raw_sub_from_one_align_assign(lines[index + 2])
                    if index + 2 < len(lines)
                    else None
                )
                next_reshape_assign = (
                    _parse_raw_rank4_singleton_reshape_assign(lines[index + 3])
                    if index + 3 < len(lines)
                    else None
                )
                if (
                    next_reduce_max_assign is not None
                    and str(next_reduce_max_assign[2]) == lhs
                    and int(next_reduce_max_assign[3]) == 3
                    and str(next_reduce_max_assign[4]) == "False"
                ):
                    reduce_lhs = str(next_reduce_max_assign[1])
                    lines[index + 1] = (
                        f"{next_reduce_max_assign[0]}{reduce_lhs} = _reduce_max("
                        f"{lhs}, _normalize_axes([1], {lhs}.ndim), False)"
                    )
                    if (
                        next_sub_assign is not None
                        and str(next_sub_assign[2]) == reduce_lhs
                        and [int(next_sub_assign[3][0]), int(next_sub_assign[3][2]), int(next_sub_assign[3][3])] == [n, h, w]
                    ):
                        sub_lhs = str(next_sub_assign[1])
                        lines[index + 2] = (
                            f"{next_sub_assign[0]}{sub_lhs} = _align_tensor_to_target_shape("
                            f"torch.sub(1.0, {reduce_lhs}), [{n}, 1, {h}, {w}])"
                        )
                        singleton_cf_vars.add(sub_lhs)
                        if (
                            next_reshape_assign is not None
                            and str(next_reshape_assign[2]) == sub_lhs
                            and [int(next_reshape_assign[3][0]), int(next_reshape_assign[3][1]), int(next_reshape_assign[3][2])] == [n, h, w]
                        ):
                            reshape_lhs = str(next_reshape_assign[1])
                            lines[index + 3] = (
                                f"{next_reshape_assign[0]}{reshape_lhs} = {sub_lhs}"
                            )
                            singleton_cf_vars.add(reshape_lhs)
                softmax_assign = _parse_raw_apply_softmax_assign(lines[index])
        if (
            softmax_assign is not None
            and next_output_permute_assign is not None
            and str(next_output_permute_assign[2]) == str(softmax_assign[1])
        ):
            input_name = str(softmax_assign[2])
            resolved_input_name = input_name
            visited_aliases: set[str] = set()
            while resolved_input_name not in visited_aliases:
                visited_aliases.add(resolved_input_name)
                next_name = (
                    cf_materialized_alias_sources.get(resolved_input_name)
                    or generic_alias_sources.get(resolved_input_name)
                )
                if next_name is None:
                    break
                resolved_input_name = next_name
            axis = int(softmax_assign[3])
            n, h, w, c = [int(v) for v in softmax_assign[5]]
            if (
                (_is_known_cf_name(input_name, singleton_cf_vars) or resolved_input_name.endswith("_nhwc_cf"))
                and axis == 3
                and c <= h
                and c <= w
            ):
                indent = str(softmax_assign[0])
                next_lhs = str(next_output_permute_assign[1])
                lines[index] = (
                    f"{indent}{next_lhs} = _apply_softmax("
                    f"{input_name}, axis=1, beta={softmax_assign[4]}, target_shape=[{n}, {c}, {h}, {w}])"
                )
                cf_aliases.add(next_lhs)
                lines[index + 1] = ""
                changed = True
        cf_nhwc_materialize_match = cf_nhwc_materialize_re.match(lines[index])
        cf_nhwc_materialize_assign = _parse_cf_nhwc_materialize_assign(lines[index])
        if cf_nhwc_materialize_match is not None or cf_nhwc_materialize_assign is not None:
            source = str(cf_nhwc_materialize_match.group("src")) if cf_nhwc_materialize_match is not None else str(cf_nhwc_materialize_assign[2])
            alias = str(cf_nhwc_materialize_match.group("lhs")) if cf_nhwc_materialize_match is not None else str(cf_nhwc_materialize_assign[1])
            if _is_known_cf_name(source, singleton_cf_vars):
                cf_materialized_alias_sources[alias] = source
                function_end = _function_end_index(index)
                immediate_uses = "\n".join(lines[index + 1:index + 4])
                future_use_count = 0
                alias_consumed_by_rank3_reshape = False
                alias_consumed_by_binary_align = False
                future_uses_are_safe = True
                for future_line in lines[index + 1 : function_end]:
                    if re.search(rf"\b{re.escape(alias)}\b", future_line) is None:
                        continue
                    future_use_count += 1
                    if re.search(rf"\btorch\.reshape\({re.escape(alias)}, ", future_line) is not None:
                        alias_consumed_by_rank3_reshape = True
                        future_uses_are_safe = False
                        break
                    future_rank3_assign = _parse_rank3_reshape_from_rank4_source_assign(future_line)
                    if future_rank3_assign is not None and str(future_rank3_assign[2]) == alias:
                        alias_consumed_by_rank3_reshape = True
                        future_uses_are_safe = False
                        break
                    future_binary_align_assign = (
                        _parse_align_binary_inputs_assign(future_line)
                        if "_align_binary_inputs(" in future_line
                        else None
                    )
                    if (
                        "_align_binary_inputs(" in future_line
                        and (
                            future_binary_align_assign is None
                            or alias in {
                                str(future_binary_align_assign[3]),
                                str(future_binary_align_assign[4]),
                            }
                        )
                    ):
                        alias_consumed_by_binary_align = True
                        future_uses_are_safe = False
                        break
                    if not (
                        future_line.lstrip().startswith("return ")
                        or "_apply_concat(" in future_line
                        or "_torch_permute(" in future_line
                        or "torch.mul(" in future_line
                        or "_align_binary_inputs_to_anchor(" in future_line
                    ):
                        future_uses_are_safe = False
                        break
                if (
                    not alias_consumed_by_rank3_reshape
                    and not alias_consumed_by_binary_align
                    and (
                        future_use_count == 0
                        or future_uses_are_safe
                        or (
                            f"{alias}" in immediate_uses
                            and (
                                "_apply_concat(" in immediate_uses
                                or "_torch_permute(" in immediate_uses
                                or "torch.mul(" in immediate_uses
                                or "_align_binary_inputs_to_anchor(" in immediate_uses
                            )
                        )
                    )
                ):
                    indent = (
                        str(cf_nhwc_materialize_match.group("indent"))
                        if cf_nhwc_materialize_match is not None
                        else str(cf_nhwc_materialize_assign[0])
                    )
                    lines[index] = f"{indent}{alias} = {source}"
                    cf_aliases.add(alias)
                    if (
                        int(cf_nhwc_materialize_match.group("c"))
                        if cf_nhwc_materialize_match is not None
                        else int(cf_nhwc_materialize_assign[3][3])
                    ) == 1:
                        singleton_cf_vars.add(alias)
                    changed = True
        index += 1
    index = 0
    while index + 4 < len(lines):
        resize10_match = apply_resize_nhwc_re.match(lines[index])
        resize11_match = apply_resize_nhwc_re.match(lines[index + 1])
        resize12_match = apply_resize_nhwc_re.match(lines[index + 2])
        concat_match = concat_re.match(lines[index + 3])
        conv_match = permuted_cf_module_input_re.match(lines[index + 4])
        conv_assign = _parse_permuted_cf_module_input_assign(lines[index + 4])
        concat_assign = _parse_simple_assignment_line(lines[index + 3])
        parsed_concat_args = (
            _parse_apply_concat_inputs_axis_and_shape(concat_assign[2])
            if concat_assign is not None
            else None
        )
        concat_inputs = (
            [input_name.strip() for input_name in parsed_concat_args[0] if input_name.strip()]
            if parsed_concat_args is not None
            else []
        )
        if (
            resize10_match is None
            or resize11_match is None
            or resize12_match is None
            or (concat_match is None and parsed_concat_args is None)
            or (conv_match is None and conv_assign is None)
            or (
                str(conv_match.group("module")) if conv_match is not None else str(conv_assign[2])
            ) != "self.conv_block_71"
            or len(concat_inputs) != 4
            or concat_inputs[1] != str(resize10_match.group("lhs"))
            or concat_inputs[2] != str(resize11_match.group("lhs"))
            or concat_inputs[3] != str(resize12_match.group("lhs"))
            or (
                str(conv_match.group("src")) if conv_match is not None else str(conv_assign[3])
            ) != (concat_assign[1] if concat_assign is not None else str(concat_match.group("lhs")))
        ):
            index += 1
            continue
        resize10_shape = [
            int(resize10_match.group("n")),
            int(resize10_match.group("c")),
            int(resize10_match.group("out_h")),
            int(resize10_match.group("out_w")),
        ]
        resize11_shape = [
            int(resize11_match.group("n")),
            int(resize11_match.group("c")),
            int(resize11_match.group("out_h")),
            int(resize11_match.group("out_w")),
        ]
        resize12_shape = [
            int(resize12_match.group("n")),
            int(resize12_match.group("c")),
            int(resize12_match.group("out_h")),
            int(resize12_match.group("out_w")),
        ]
        lines[index] = (
            f"{resize10_match.group('indent')}{resize10_match.group('lhs')} = _apply_resize("
            f"{resize10_match.group('input')}, [{resize10_match.group('out_h')}, {resize10_match.group('out_w')}], "
            f"method='{resize10_match.group('method')}', target_shape={repr(resize10_shape)}, align_corners={resize10_match.group('align')}, "
            f"half_pixel_centers={resize10_match.group('hpc')}, channel_last=False)"
        )
        lines[index + 1] = (
            f"{resize11_match.group('indent')}{resize11_match.group('lhs')} = _apply_resize("
            f"{resize11_match.group('input')}, [{resize11_match.group('out_h')}, {resize11_match.group('out_w')}], "
            f"method='{resize11_match.group('method')}', target_shape={repr(resize11_shape)}, align_corners={resize11_match.group('align')}, "
            f"half_pixel_centers={resize11_match.group('hpc')}, channel_last=False)"
        )
        lines[index + 2] = (
            f"{resize12_match.group('indent')}{resize12_match.group('lhs')} = _apply_resize("
            f"{resize12_match.group('input')}, [{resize12_match.group('out_h')}, {resize12_match.group('out_w')}], "
            f"method='{resize12_match.group('method')}', target_shape={repr(resize12_shape)}, align_corners={resize12_match.group('align')}, "
            f"half_pixel_centers={resize12_match.group('hpc')}, channel_last=False)"
        )
        lines[index + 3] = (
            f"{(concat_assign[0] if concat_assign is not None else concat_match.group('indent'))}"
            f"{(concat_assign[1] if concat_assign is not None else concat_match.group('lhs'))} = "
            f"torch.cat([{', '.join(concat_inputs)}], dim=1)"
        )
        lines[index + 4] = (
            f"{(conv_match.group('indent') if conv_match is not None else conv_assign[0])}"
            f"{(conv_match.group('lhs') if conv_match is not None else conv_assign[1])} = "
            f"{(conv_match.group('module') if conv_match is not None else conv_assign[2])}("
            f"{(concat_assign[1] if concat_assign is not None else concat_match.group('lhs'))})"
        )
        cf_aliases.update(
            {
                str(resize10_match.group("lhs")),
                str(resize11_match.group("lhs")),
                str(resize12_match.group("lhs")),
                str(concat_assign[1] if concat_assign is not None else concat_match.group("lhs")),
            }
        )
        changed = True
        index += 5
    for index, line in enumerate(lines):
        assign_args = _parse_simple_assignment_line(line)
        parsed_torch_cat_args = (
            _parse_torch_cat_inputs_and_dim(assign_args[2])
            if assign_args is not None
            else None
        )
        if parsed_torch_cat_args is None:
            continue
        input_names = [name.strip() for name in parsed_torch_cat_args[0] if name.strip()]
        normalized_inputs = [
            source_name if source_name != name and _is_name_available_in_function(source_name, index) else name
            for name in input_names
            for source_name in [cf_materialized_alias_sources.get(name, name)]
        ]
        if normalized_inputs == input_names:
            continue
        lhs = assign_args[1]
        axis = parsed_torch_cat_args[1]
        if axis != 1:
            continue
        if not all(_is_known_cf_name(name, singleton_cf_vars) for name in normalized_inputs):
            continue
        lines[index] = (
            f"{assign_args[0]}"
            f"{lhs} = torch.cat([{', '.join(normalized_inputs)}], dim=1)"
        )
        cf_aliases.add(lhs)
        changed = True
    for index, line in enumerate(lines):
        split_assign = _parse_tensor_split_assign(line)
        if split_assign is None or int(split_assign[4]) != 3:
            continue
        input_name = str(split_assign[2])
        sections = int(split_assign[3])
        previous_match = aligned_nhwc_rank4_re.match(lines[index - 1]) if index > 0 else None
        should_rewrite_split = _is_known_cf_name(input_name, singleton_cf_vars)
        if (
            not should_rewrite_split
            and previous_match is not None
            and str(previous_match.group("lhs")) == input_name
            and "_nhwc" not in input_name
            and int(previous_match.group("h")) == sections
            and int(previous_match.group("c")) != sections
        ):
            should_rewrite_split = True
            cf_aliases.add(input_name)
        if not should_rewrite_split:
            continue
        outputs = ", ".join(split_assign[1])
        lines[index] = (
            f"{split_assign[0]}{outputs} = list(torch.tensor_split("
            f"{input_name}, {sections}, dim=_normalize_dim(1, {input_name}.ndim)))"
        )
        for output_name in [token.strip() for token in outputs.split(",") if token.strip()]:
            singleton_cf_vars.add(output_name)
        changed = True
    index = 0
    while index + 3 < len(lines):
        input_bridge_match = transpose_conv_input_bridge_re.match(lines[index])
        input_bridge_assign = _parse_transpose_conv_input_bridge_assign(lines[index])
        apply_match = transpose_conv_apply_re.match(lines[index + 1])
        crop_match = transpose_conv_crop_re.match(lines[index + 2])
        output_permute_match = transpose_conv_output_permute_re.match(lines[index + 3])
        output_permute_assign = _parse_output_back_permute_assign(lines[index + 3])
        if (
            (input_bridge_match is None and input_bridge_assign is None)
            or apply_match is None
            or crop_match is None
            or (output_permute_match is None and output_permute_assign is None)
            or str(apply_match.group("input")) != (
                str(input_bridge_match.group("alias"))
                if input_bridge_match is not None
                else str(input_bridge_assign[1])
            )
            or str(crop_match.group("src")) != str(apply_match.group("lhs"))
            or (
                str(output_permute_match.group("src"))
                if output_permute_match is not None
                else str(output_permute_assign[2])
            ) != str(crop_match.group("lhs"))
        ):
            index += 1
            continue
        try:
            target_values = [int(value.strip()) for value in str(apply_match.group("target")).split(",")]
            fallback_values = [int(value.strip()) for value in str(apply_match.group("fallback")).split(",")]
        except Exception:
            index += 1
            continue
        if (
            len(target_values) != 4
            or len(fallback_values) != 4
            or target_values[0] != fallback_values[0]
            or target_values[-1] != 1
            or fallback_values[1] != 1
            or target_values[1] != fallback_values[2]
            or target_values[2] != fallback_values[3]
        ):
            index += 1
            continue
        indent = str(apply_match.group("indent"))
        input_alias = (
            str(input_bridge_match.group("alias"))
            if input_bridge_match is not None
            else str(input_bridge_assign[1])
        )
        input_source = (
            str(input_bridge_match.group("src"))
            if input_bridge_match is not None
            else str(input_bridge_assign[2])
        )
        apply_lhs = str(apply_match.group("lhs"))
        cropped_lhs = str(crop_match.group("lhs"))
        permuted_lhs = (
            str(output_permute_match.group("lhs"))
            if output_permute_match is not None
            else str(output_permute_assign[1])
        )
        fallback_text = ", ".join(str(v) for v in fallback_values)
        lines[index] = f"{indent}{input_alias} = {input_source}"
        lines[index + 1] = (
            f"{indent}{apply_lhs} = _apply_module_transpose_conv2d("
            f"{input_alias}, {apply_match.group('prefix')}, "
            f"target_shape=[{fallback_text}], fallback_shape=[{fallback_text}], "
            f"target_logical_layout='NCHW', fused='{apply_match.group('fused')}')"
        )
        lines[index + 3] = f"{indent}{permuted_lhs} = {cropped_lhs}"
        changed = True
        index += 4
    index = 0
    while index + 1 < len(lines):
        permute_cf_match = permute_contiguous_cf_re.match(lines[index])
        reshape_inverse_match = reshape_from_inverse_permute_re.match(lines[index + 1])
        if (
            permute_cf_match is None
            or reshape_inverse_match is None
            or str(reshape_inverse_match.group("src")) != str(permute_cf_match.group("lhs"))
        ):
            index += 1
            continue
        indent = str(reshape_inverse_match.group("indent"))
        src = str(permute_cf_match.group("src"))
        lhs = str(reshape_inverse_match.group("lhs"))
        shape = str(reshape_inverse_match.group("shape"))
        lines[index + 1] = f"{indent}{lhs} = torch.reshape({src}, [{shape}])"
        changed = True
        index += 2
    index = 0
    while index + 2 < len(lines):
        alias_match = simple_alias_re.match(lines[index])
        self_permute_match = self_permute_assign_re.match(lines[index + 1])
        input_bridge_match = transpose_conv_input_bridge_re.match(lines[index + 2])
        input_bridge_assign = _parse_transpose_conv_input_bridge_assign(lines[index + 2])
        if (
            alias_match is None
            or self_permute_match is None
            or (input_bridge_match is None and input_bridge_assign is None)
            or str(self_permute_match.group("lhs")) != str(alias_match.group("lhs"))
            or (
                str(input_bridge_match.group("src"))
                if input_bridge_match is not None
                else str(input_bridge_assign[2])
            ) != str(alias_match.group("lhs"))
        ):
            index += 1
            continue
        src = str(alias_match.group("src"))
        lhs = str(alias_match.group("lhs"))
        src_exact_shape = _model_ir_exact_shape(src)
        lhs_exact_shape = _model_ir_exact_shape(lhs)
        if (
            src_exact_shape is None
            or lhs_exact_shape is None
            or len(src_exact_shape) != 3
            or len(lhs_exact_shape) != 4
            or int(np.prod(src_exact_shape, dtype=np.int64)) != int(np.prod(lhs_exact_shape, dtype=np.int64))
            or int(lhs_exact_shape[0]) != int(src_exact_shape[0])
            or int(lhs_exact_shape[1]) != int(src_exact_shape[1])
            or not (
                (int(lhs_exact_shape[2]) == 1 and int(lhs_exact_shape[3]) == int(src_exact_shape[2]))
                or (int(lhs_exact_shape[3]) == 1 and int(lhs_exact_shape[2]) == int(src_exact_shape[2]))
            )
        ):
            index += 1
            continue
        indent = str(alias_match.group("indent"))
        shape_text = ", ".join(str(v) for v in lhs_exact_shape)
        lines[index] = f"{indent}{lhs} = torch.reshape({src}, [{shape_text}])"
        lines[index + 1] = f"{indent}{lhs} = {lhs}"
        cf_aliases.add(lhs)
        changed = True
        index += 3
    index = 0
    while index + 5 < len(lines):
        self_permute_match = self_permute_assign_re.match(lines[index])
        input_bridge_match = transpose_conv_input_bridge_re.match(lines[index + 1])
        input_bridge_assign = _parse_transpose_conv_input_bridge_assign(lines[index + 1])
        apply_match = transpose_conv_apply_re.match(lines[index + 2])
        crop_match = transpose_conv_crop_re.match(lines[index + 3])
        bias_fix_match = transpose_conv_bias_fix_re.match(lines[index + 4])
        bias_fix_assign = _parse_transpose_conv_bias_fix_assign(lines[index + 4])
        bias_add_match = transpose_conv_bias_add_re.match(lines[index + 5])
        bias_add_assign = _parse_transpose_conv_bias_add_assign(lines[index + 5])
        if (
            self_permute_match is None
            or (input_bridge_match is None and input_bridge_assign is None)
            or apply_match is None
            or crop_match is None
            or (bias_fix_match is None and bias_fix_assign is None)
            or (bias_add_match is None and bias_add_assign is None)
            or (
                str(input_bridge_match.group("src"))
                if input_bridge_match is not None
                else str(input_bridge_assign[2])
            ) != str(self_permute_match.group("lhs"))
            or str(apply_match.group("input")) != (
                str(input_bridge_match.group("alias"))
                if input_bridge_match is not None
                else str(input_bridge_assign[1])
            )
            or str(crop_match.group("src")) != str(apply_match.group("lhs"))
        ):
            index += 1
            continue
        try:
            target_values = [int(value.strip()) for value in str(apply_match.group("target")).split(",")]
            fallback_values = [int(value.strip()) for value in str(apply_match.group("fallback")).split(",")]
        except Exception:
            index += 1
            continue
        if (
            len(target_values) != 4
            or len(fallback_values) != 4
            or target_values[0] != fallback_values[0]
            or target_values[-1] != 1
            or fallback_values[1] != 1
            or target_values[1] != fallback_values[2]
            or target_values[2] != fallback_values[3]
            or (
                str(bias_add_match.group("a"))
                if bias_add_match is not None
                else str(bias_add_assign[2])
            ) != str(crop_match.group("lhs"))
            or (
                str(bias_add_match.group("b"))
                if bias_add_match is not None
                else str(bias_add_assign[3])
            ) != (
                str(bias_fix_match.group("lhs"))
                if bias_fix_match is not None
                else str(bias_fix_assign[1])
            )
        ):
            index += 1
            continue
        indent = str(apply_match.group("indent"))
        input_source = str(self_permute_match.group("lhs"))
        input_alias = (
            str(input_bridge_match.group("alias"))
            if input_bridge_match is not None
            else str(input_bridge_assign[1])
        )
        apply_lhs = str(apply_match.group("lhs"))
        bias_expr = (
            str(bias_fix_match.group("expr"))
            if bias_fix_match is not None
            else str(bias_fix_assign[2])
        )
        bias_lhs = (
            str(bias_fix_match.group("lhs"))
            if bias_fix_match is not None
            else str(bias_fix_assign[1])
        )
        bias_add_lhs = (
            str(bias_add_match.group("lhs"))
            if bias_add_match is not None
            else str(bias_add_assign[1])
        )
        fallback_text = ", ".join(str(v) for v in fallback_values)
        cropped_h = int(crop_match.group("end")) - int(crop_match.group("start"))
        cropped_w = int(crop_match.group("width"))
        lines[index] = f"{indent}{input_source} = {input_source}"
        lines[index + 1] = f"{indent}{input_alias} = {input_source}"
        lines[index + 2] = (
            f"{indent}{apply_lhs} = _apply_module_transpose_conv2d("
            f"{input_alias}, {apply_match.group('prefix')}, "
            f"target_shape=[{fallback_text}], fallback_shape=[{fallback_text}], "
            f"target_logical_layout='NCHW', fused='{apply_match.group('fused')}')"
        )
        lines[index + 4] = f"{indent}{bias_lhs} = {bias_expr}"
        lines[index + 5] = (
            f"{indent}{bias_add_lhs} = _align_tensor_to_target_shape("
            f"torch.add({(bias_add_match.group('a') if bias_add_match is not None else bias_add_assign[2])}, {bias_lhs}), "
            f"[{fallback_values[0]}, 1, {cropped_h}, {cropped_w}])"
        )
        changed = True
        index += 6
    index = 0
    while index + 1 < len(lines):
        aligned_tensor_match = generic_aligned_tensor_re.match(lines[index])
        output_back_permute_assign = _parse_output_back_permute_assign(lines[index + 1])
        if (
            aligned_tensor_match is None
            or output_back_permute_assign is None
            or str(output_back_permute_assign[2]) != str(aligned_tensor_match.group("lhs"))
        ):
            index += 1
            continue
        lhs = str(aligned_tensor_match.group("lhs"))
        dst = str(output_back_permute_assign[1])
        expr = str(aligned_tensor_match.group("expr"))
        dst_exact_shape = _model_ir_exact_shape(dst)
        if (
            dst_exact_shape is None
            or len(dst_exact_shape) != 4
            or "_cf" not in expr
            or not lhs.endswith("_nhwc")
        ):
            index += 1
            continue
        indent = str(aligned_tensor_match.group("indent"))
        dst_shape_text = ", ".join(str(v) for v in dst_exact_shape)
        lines[index] = (
            f"{indent}{lhs} = _align_tensor_to_target_shape({expr}, [{dst_shape_text}])"
        )
        lines[index + 1] = f"{indent}{dst} = {lhs}"
        changed = True
        index += 2
    return_tuple_re = re.compile(r"^(?P<indent>\s*)return \((?P<values>[A-Za-z0-9_, ]+)\)$")
    return_single_re = re.compile(r"^(?P<indent>\s*)return (?P<value>[A-Za-z0-9_]+)$")
    for index, line in enumerate(lines):
        return_tuple_match = return_tuple_re.match(line)
        if return_tuple_match is not None:
            values = [value.strip() for value in str(return_tuple_match.group("values")).split(",") if value.strip()]
            rewritten_values = [cf_materialized_alias_sources.get(value, value) for value in values]
            if rewritten_values != values:
                lines[index] = f"{return_tuple_match.group('indent')}return ({', '.join(rewritten_values)})"
                changed = True
            continue
        return_single_match = return_single_re.match(line)
        if return_single_match is not None:
            value = str(return_single_match.group("value"))
            rewritten_value = cf_materialized_alias_sources.get(value, value)
            if rewritten_value != value:
                lines[index] = f"{return_single_match.group('indent')}return {rewritten_value}"
                changed = True
    if len(transposed_const_alias_specs) > 0 or len(inline_const_buffer_specs) > 0:
        if not any("from typing import" in line and "Mapping" in line for line in lines):
            for idx, line in enumerate(lines):
                if line.startswith("from typing import "):
                    if "Mapping" not in line:
                        lines[idx] = line.replace("from typing import ", "from typing import Mapping, ")
                        changed = True
                    break
        existing_alias_names = set(buffer_specs.keys())
        alias_init_lines: List[str] = []
        for alias_attr, (literal, source_dtype) in sorted(inline_const_buffer_specs.items()):
            if alias_attr in existing_alias_names:
                continue
            alias_init_lines.append(
                f"        self.register_buffer('{alias_attr}', torch.tensor({literal}, dtype=torch.{source_dtype}), persistent=False)"
            )
            existing_alias_names.add(alias_attr)
        for alias_attr, (_, alias_shape, source_dtype) in sorted(transposed_const_alias_specs.items()):
            if alias_attr in existing_alias_names:
                continue
            alias_shape_text = ", ".join(str(v) for v in alias_shape)
            alias_init_lines.append(
                f"        self.register_buffer('{alias_attr}', torch.zeros([{alias_shape_text}], dtype=torch.{source_dtype}), persistent=False)"
            )
            existing_alias_names.add(alias_attr)
        init_constants_start = next((idx for idx, line in enumerate(lines) if line.startswith("    def _init_constants(self) -> None:")), None)
        if init_constants_start is not None:
            insert_after = init_constants_start + 1
            while insert_after < len(lines) and lines[insert_after].startswith("        self.register_buffer("):
                insert_after += 1
            if len(alias_init_lines) > 0:
                lines[insert_after:insert_after] = alias_init_lines
                changed = True
        elif len(alias_init_lines) > 0:
            insert_after = next(
                (
                    idx + 1
                    for idx, line in enumerate(lines)
                    if "self._onnx2tf_torch_export_mode = False" in line
                ),
                None,
            )
            if insert_after is None:
                insert_after = next(
                    (
                        idx + 1
                        for idx, line in enumerate(lines)
                        if line.startswith("        self.input_names = ")
                    ),
                    None,
                )
            if insert_after is not None:
                lines[insert_after:insert_after] = alias_init_lines
                changed = True
        if len(transposed_const_alias_specs) > 0:
            refresh_method_name = "_refresh_transposed_constant_buffers"
            has_refresh_method = any(line.startswith(f"    def {refresh_method_name}(self) -> None:") for line in lines)
            if not has_refresh_method:
                insert_index = next((idx for idx, line in enumerate(lines) if line.startswith("    def _forward_stage_0(")), None)
                if insert_index is not None:
                    refresh_method_lines = [
                        f"    def {refresh_method_name}(self) -> None:",
                        "        with torch.no_grad():",
                    ]
                    for alias_attr, (source_attr, _, _) in sorted(transposed_const_alias_specs.items()):
                        refresh_method_lines.append(
                            f"            self.{alias_attr}.copy_(self.{source_attr}.transpose(-1, -2))"
                        )
                    refresh_method_lines.append("")
                    lines[insert_index:insert_index] = refresh_method_lines
                    changed = True
            refresh_call_line = f"        self.{refresh_method_name}()"
            if not any(line == refresh_call_line for line in lines):
                eval_index = next((idx for idx, line in enumerate(lines) if line.strip() == "if eval_mode:"), None)
                if eval_index is not None:
                    lines.insert(eval_index, refresh_call_line)
                    changed = True
            has_load_state_dict_override = any(line.startswith("    def load_state_dict(") for line in lines)
            if not has_load_state_dict_override:
                insert_index = next((idx for idx, line in enumerate(lines) if line.startswith("    def _forward_stage_0(")), None)
                if insert_index is not None:
                    load_state_dict_lines = [
                        "    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):",
                        "        result = super().load_state_dict(state_dict, strict=strict, assign=assign)",
                        f"        self.{refresh_method_name}()",
                        "        return result",
                        "",
                    ]
                    lines[insert_index:insert_index] = load_state_dict_lines
                    changed = True
    for index, line in enumerate(lines):
        generic_alias_assign = _parse_simple_assignment_line(line)
        if (
            generic_alias_assign is None
            or not _is_simple_identifier_expr(generic_alias_assign[2])
        ):
            continue
        lhs = str(generic_alias_assign[1])
        rhs = _strip_outer_parentheses(str(generic_alias_assign[2]).strip())
        reshape_consumer_found = False
        for lookahead_index in range(index + 1, min(index + 6, len(lines))):
            candidate_reshape_assign = _parse_rank4_reshape_consumer_assign(lines[lookahead_index])
            if candidate_reshape_assign is not None and str(candidate_reshape_assign[2]) == lhs:
                reshape_consumer_found = True
                break
        if not reshape_consumer_found:
            continue
        lhs_exact_shape = _model_ir_exact_shape(lhs)
        lhs_tensor = (
            model_ir.tensors.get(_resolve_model_ir_tensor_name(lhs), None)
            if model_ir is not None
            else None
        )
        lhs_layout = (
            normalize_logical_layout(lhs_tensor.logical_layout)
            if lhs_tensor is not None
            else LOGICAL_LAYOUT_UNKNOWN
        )
        if (
            lhs_exact_shape is not None
            and len(lhs_exact_shape) == 4
            and is_channel_last_logical_layout(lhs_layout)
            and (_is_known_cf_name(rhs, set()) or rhs.endswith("_cf") or rhs.endswith("_nhwc_cf"))
        ):
            indent = str(generic_alias_assign[0])
            rewritten_line = (
                f"{indent}{lhs} = _align_tensor_to_target_shape("
                f"{rhs}.permute(0, 2, 3, 1).contiguous(), {lhs_exact_shape})"
            )
            if rewritten_line != line:
                lines[index] = rewritten_line
                changed = True
            continue
        if "_nhwc" not in lhs or not (_is_known_cf_name(rhs, set()) or rhs.endswith("_nhwc_cf")):
            continue
        rhs_exact_shape = _model_ir_exact_shape(rhs)
        if rhs_exact_shape is None or len(rhs_exact_shape) != 4:
            rhs_exact_shape = None
        indent = str(generic_alias_assign[0])
        rewritten_line = f"{indent}{lhs} = {rhs}.permute(0, 2, 3, 1).contiguous()"
        if rewritten_line != line:
            lines[index] = rewritten_line
            changed = True
    if any("_tensor_shape_list(" in line for line in lines):
        runtime_import_line = "    _tensor_shape_list,"
        has_runtime_import = any(line.strip() == "_tensor_shape_list," for line in lines)
        if not has_runtime_import:
            runtime_import_block_start = next(
                (idx for idx, line in enumerate(lines) if line.strip() == "from .runtime import ("),
                None,
            )
            runtime_import_block_end = next(
                (
                    idx for idx, line in enumerate(lines)
                    if runtime_import_block_start is not None and idx > runtime_import_block_start and line.strip() == ")"
                ),
                None,
            )
            if runtime_import_block_end is not None:
                lines.insert(runtime_import_block_end, runtime_import_line)
                changed = True
    orphan_rank3_permute_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_torch_permute\((?P<src>[A-Za-z0-9_]+), \[0, 2, 1\]\)$"
    )
    square_permuted_const_alias_re = re.compile(
        r"^\s*self\.(?P<alias>[A-Za-z0-9_]+)\.copy_\("
        r"self\.(?P<source>[A-Za-z0-9_]+)\.permute\(\*\(0, 1, 3, 2\)\)\.contiguous\(\)\)$"
    )
    square_permuted_add_use_re = re.compile(
        r"(?P<prefix>torch\.add\([^,\n]+,\s*)self\.(?P<alias>[A-Za-z0-9_]+)(?P<suffix>\)\s*)"
    )
    square_target_shape_re = re.compile(
        r"(?:target_shape=)?\[(?P<a>-?\d+),\s*(?P<b>-?\d+),\s*(?P<h>-?\d+),\s*(?P<w>-?\d+)\]"
    )
    square_alias_sources: Dict[str, str] = {}
    for line in lines:
        square_permuted_const_alias_match = square_permuted_const_alias_re.match(line)
        if square_permuted_const_alias_match is None:
            continue
        square_alias_sources[str(square_permuted_const_alias_match.group("alias"))] = str(
            square_permuted_const_alias_match.group("source")
        )
    if len(square_alias_sources) > 0:
        for index, line in enumerate(lines):
            target_shape_match = square_target_shape_re.search(line)
            if target_shape_match is None:
                continue
            if int(target_shape_match.group("h")) != int(target_shape_match.group("w")):
                continue
            square_add_use_match = square_permuted_add_use_re.search(line)
            if square_add_use_match is None:
                continue
            alias_name = str(square_add_use_match.group("alias"))
            source_name = square_alias_sources.get(alias_name, "")
            if source_name == "":
                continue
            rewritten_line = (
                line[: square_add_use_match.start()]
                + str(square_add_use_match.group("prefix"))
                + f"self.{source_name}"
                + str(square_add_use_match.group("suffix"))
                + line[square_add_use_match.end() :]
            )
            if rewritten_line != line:
                lines[index] = rewritten_line
                changed = True
    for index in range(len(lines) - 1):
        aligned_resize_input_match = aligned_nhwc_rank4_re.match(lines[index])
        if aligned_resize_input_match is None:
            continue
        resize_cf_match = _parse_resize_assign_line(lines[index + 1])
        if (
            resize_cf_match is None
            or str(resize_cf_match[2]) != str(aligned_resize_input_match.group("lhs"))
            or (
                not _find_same_function_cat_consumer(str(resize_cf_match[1]), index + 1)
                and not _find_stage_boundary_cat_consumer(str(resize_cf_match[1]), index + 1)
            )
        ):
            continue
        input_exact_shape = _model_ir_exact_shape(str(aligned_resize_input_match.group("lhs")))
        output_exact_shape = _model_ir_exact_shape(str(resize_cf_match[1]))
        resolved_input_name = _resolve_model_ir_tensor_name(str(aligned_resize_input_match.group("lhs")))
        out_h = int(resize_cf_match[3])
        out_w = int(resize_cf_match[4])
        if (
            input_exact_shape is None
            or output_exact_shape is None
            or len(input_exact_shape) != 4
            or len(output_exact_shape) != 4
            or not _tensor_name_suggests_channel_last_layout_for_codegen(resolved_input_name)
        ):
            simple_expr_match = simple_binary_expr_re.match(str(aligned_resize_input_match.group("expr")))
            fallback_cf_shape = None
            if simple_expr_match is not None:
                fallback_cf_shape = _find_recent_rank4_shape(str(simple_expr_match.group("a")), index)
                if fallback_cf_shape is None:
                    fallback_cf_shape = _find_recent_rank4_shape(str(simple_expr_match.group("b")), index)
            if fallback_cf_shape is None or len(fallback_cf_shape) != 4:
                continue
            n = int(fallback_cf_shape[0])
            c = int(fallback_cf_shape[1])
            h = int(fallback_cf_shape[2])
            w = int(fallback_cf_shape[3])
        else:
            n = int(input_exact_shape[0])
            h = int(input_exact_shape[1])
            w = int(input_exact_shape[2])
            c = int(input_exact_shape[3])
            out_h = int(output_exact_shape[1])
            out_w = int(output_exact_shape[2])
        lines[index] = (
            f"{aligned_resize_input_match.group('indent')}{aligned_resize_input_match.group('lhs')} = "
            f"_align_tensor_to_target_shape({aligned_resize_input_match.group('expr')}, [{n}, {c}, {h}, {w}])"
        )
        lines[index + 1] = (
            f"{resize_cf_match[0]}{resize_cf_match[1]} = _apply_resize("
            f"{aligned_resize_input_match.group('lhs')}, [{out_h}, {out_w}], method='{resize_cf_match[6]}', "
            f"target_shape=[{n}, {c}, {out_h}, {out_w}], "
            f"align_corners={resize_cf_match[7]}, "
            f"half_pixel_centers={resize_cf_match[8]}, channel_last=False)"
        )
        changed = True
    expected_public_output_cf_target_perm = _perm_cl_to_cf(4)
    for index, line in enumerate(lines):
        if expected_public_output_cf_target_perm is None:
            continue
        resize_public_output_match = _parse_resize_assign_line(line)
        if resize_public_output_match is None:
            continue
        exact_public_output_shape = _eventual_public_output_exact_shape(
            str(resize_public_output_match[1]),
            index,
        )
        if (
            exact_public_output_shape is None
            or len(exact_public_output_shape) != 4
        ):
            continue
        current_target_shape = list(resize_public_output_match[5])
        if current_target_shape == [int(v) for v in exact_public_output_shape]:
            continue
        if _permute_shape(exact_public_output_shape, expected_public_output_cf_target_perm) != current_target_shape:
            continue
        lines[index] = (
            f"{resize_public_output_match[0]}{resize_public_output_match[1]} = _apply_resize("
            f"{resize_public_output_match[2]}, [{resize_public_output_match[3]}, {resize_public_output_match[4]}], "
            f"method='{resize_public_output_match[6]}', "
            f"target_shape={repr([int(v) for v in exact_public_output_shape])}, "
            f"align_corners={resize_public_output_match[7]}, "
            f"half_pixel_centers={resize_public_output_match[8]}, channel_last=False)"
        )
        changed = True
    pidnet_forced_resize_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_resize\("
        r"(?:input=)?(?P<input>[A-Za-z0-9_]+), (?:size=)?[\[\(](?P<out_h>\d+), (?P<out_w>\d+)[\]\)], method='(?P<method>[^']+)', "
        r"target_shape=[\[\(](?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)[\]\)], "
        r"align_corners=(?P<align>True|False), half_pixel_centers=(?P<hpc>True|False), channel_last=(?:True|False)\)$"
    )
    for index, line in enumerate(lines):
        pidnet_forced_resize_match = pidnet_forced_resize_re.match(line)
        if pidnet_forced_resize_match is None:
            continue
        lhs = str(pidnet_forced_resize_match.group("lhs"))
        target_shape = _pidnet_forced_resize_target(
            lhs,
            str(pidnet_forced_resize_match.group("input")),
            [
                int(pidnet_forced_resize_match.group("n")),
                int(pidnet_forced_resize_match.group("h")),
                int(pidnet_forced_resize_match.group("w")),
                int(pidnet_forced_resize_match.group("c")),
            ],
            int(pidnet_forced_resize_match.group("out_h")),
            int(pidnet_forced_resize_match.group("out_w")),
        )
        if target_shape is None:
            continue
        lines[index] = (
            f"{pidnet_forced_resize_match.group('indent')}{lhs} = _apply_resize("
            f"{pidnet_forced_resize_match.group('input')}, "
            f"[{pidnet_forced_resize_match.group('out_h')}, {pidnet_forced_resize_match.group('out_w')}], "
            f"method='{pidnet_forced_resize_match.group('method')}', "
            f"target_shape={target_shape}, "
            f"align_corners={pidnet_forced_resize_match.group('align')}, "
            f"half_pixel_centers={pidnet_forced_resize_match.group('hpc')}, channel_last=False)"
        )
        changed = True
    softmax_cf_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_softmax\((?:input=)?(?P<input>[A-Za-z0-9_]+), "
        r"axis=1, beta=(?P<beta>[-0-9.eE]+), target_shape=[\[\(](?P<n>\d+), (?P<c>\d+), (?P<h>\d+), (?P<w>\d+)[\]\)]\)$"
    )
    for index, line in enumerate(lines):
        resize_match = _parse_resize_assign_line(line)
        if resize_match is None:
            continue
        resize_lhs = str(resize_match[1])
        consumer_name = resize_lhs
        consumer_index = index + 1
        alias_match = (
            re.fullmatch(
                r"(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<src>[A-Za-z0-9_]+)",
                lines[index + 1],
            )
            if index + 1 < len(lines)
            else None
        )
        if alias_match is not None and str(alias_match.group("src")) == resize_lhs:
            consumer_name = str(alias_match.group("lhs"))
            consumer_index = index + 2
        binary_consumer_match = (
            binary_cf_consumer_re.match(lines[consumer_index])
            if consumer_index < len(lines)
            else None
        )
        if (
            binary_consumer_match is not None
            and consumer_name in {
                str(binary_consumer_match.group("a")),
                str(binary_consumer_match.group("b")),
            }
        ):
            resize_target_shape = [int(v) for v in resize_match[5]]
            if (
                len(resize_target_shape) == 4
                and int(resize_target_shape[1]) == int(resize_match[3])
                and int(resize_target_shape[2]) == int(resize_match[4])
            ):
                target_shape_text = (
                    f"[_tensor_shape_list({resize_match[2]})[0], "
                    f"_tensor_shape_list({resize_match[2]})[1], "
                    f"{resize_match[3]}, {resize_match[4]}]"
                )
            else:
                target_shape_text = (
                    f"[{int(binary_consumer_match.group('n'))}, "
                    f"{int(binary_consumer_match.group('c'))}, "
                    f"{int(binary_consumer_match.group('h'))}, "
                    f"{int(binary_consumer_match.group('w'))}]"
                )
            lines[index] = (
                f"{resize_match[0]}{resize_lhs} = _apply_resize("
                f"{resize_match[2]}, [{resize_match[3]}, {resize_match[4]}], "
                f"method='{resize_match[6]}', "
                f"target_shape={target_shape_text}, "
                f"align_corners={resize_match[7]}, "
                f"half_pixel_centers={resize_match[8]}, channel_last=False)"
            )
            cf_aliases.add(resize_lhs)
            if consumer_name != resize_lhs:
                cf_aliases.add(consumer_name)
            changed = True
            continue
        softmax_cf_match = (
            softmax_cf_re.match(lines[consumer_index])
            if consumer_index < len(lines)
            else None
        )
        if softmax_cf_match is not None and str(softmax_cf_match.group("input")) == consumer_name:
            n = int(softmax_cf_match.group("n"))
            c = int(softmax_cf_match.group("c"))
            h = int(softmax_cf_match.group("h"))
            w = int(softmax_cf_match.group("w"))
            lines[index] = (
                f"{resize_match[0]}{resize_lhs} = _apply_resize("
                f"{resize_match[2]}, [{resize_match[3]}, {resize_match[4]}], "
                f"method='{resize_match[6]}', "
                f"target_shape=[{n}, {c}, {h}, {w}], "
                f"align_corners={resize_match[7]}, "
                f"half_pixel_centers={resize_match[8]}, channel_last=False)"
            )
            cf_aliases.add(resize_lhs)
            if consumer_name != resize_lhs:
                cf_aliases.add(consumer_name)
            changed = True
    for index, line in enumerate(lines):
        resize_argmax_match = _parse_resize_assign_line(line)
        if resize_argmax_match is None:
            continue
        input_name = str(resize_argmax_match[2])
        if not (_is_known_cf_name(input_name, singleton_cf_vars) or input_name.endswith("_nhwc_cf")):
            continue
        alias_match = (
            re.fullmatch(
                r"(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = (?P<src>[A-Za-z0-9_]+)",
                lines[index + 1],
            )
            if index + 1 < len(lines)
            else None
        )
        argmax_index = index + 1
        if alias_match is not None and str(alias_match.group("src")) == str(resize_argmax_match[1]):
            argmax_index = index + 2
        if argmax_index >= len(lines):
            continue
        argmax_assign = _parse_argmax_assign(lines[argmax_index])
        if (
            argmax_assign is None
            or (
                str(argmax_assign[2]) != str(resize_argmax_match[1])
                and (
                    alias_match is None
                    or str(argmax_assign[2]) != str(alias_match.group("lhs"))
                )
            )
        ):
            continue
        channel_count = _infer_cf_channel_count(input_name)
        if channel_count is None:
            channel_count = int(
                _normalize_cf_rank4_shape(
                    resize_argmax_match[5],
                    out_hw=(int(resize_argmax_match[3]), int(resize_argmax_match[4])),
                )[1]
            )
        lines[index] = (
            f"{resize_argmax_match[0]}{resize_argmax_match[1]} = _apply_resize("
            f"{input_name}, [{resize_argmax_match[3]}, {resize_argmax_match[4]}], "
            f"method='{resize_argmax_match[6]}', "
            f"target_shape=[{resize_argmax_match[5][0]}, {channel_count}, "
            f"{resize_argmax_match[3]}, {resize_argmax_match[4]}], "
            f"align_corners={resize_argmax_match[7]}, "
            f"half_pixel_centers={resize_argmax_match[8]}, channel_last=False)"
        )
        lines[argmax_index] = (
            f"{argmax_assign[0]}{argmax_assign[1]} = "
            f"torch.argmax({argmax_assign[2]}, "
            f"dim=_normalize_dim(1, {argmax_assign[2]}.ndim), "
            f"keepdim={argmax_assign[4]}).to(dtype=torch.int64)"
        )
        changed = True
    for index, line in enumerate(lines):
        resize_binary_match = _parse_resize_assign_line(line)
        if resize_binary_match is None or resize_binary_match[9]:
            continue
        target_shape = [int(v) for v in resize_binary_match[5]]
        if not (
            len(target_shape) == 4
            and int(target_shape[1]) == int(resize_binary_match[3])
            and int(target_shape[2]) == int(resize_binary_match[4])
        ):
            continue
        alias_match = (
            re.fullmatch(
                r"(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<src>[A-Za-z0-9_]+)",
                lines[index + 1],
            )
            if index + 1 < len(lines)
            else None
        )
        consumer_name = str(resize_binary_match[1])
        consumer_index = index + 1
        if alias_match is not None and str(alias_match.group("src")) == consumer_name:
            consumer_name = str(alias_match.group("lhs"))
            consumer_index = index + 2
        binary_consumer_match = (
            binary_cf_consumer_re.match(lines[consumer_index])
            if consumer_index < len(lines)
            else None
        )
        if (
            binary_consumer_match is None
            or consumer_name not in {
                str(binary_consumer_match.group("a")),
                str(binary_consumer_match.group("b")),
            }
        ):
            continue
        lines[index] = (
            f"{resize_binary_match[0]}{resize_binary_match[1]} = _apply_resize("
            f"{resize_binary_match[2]}, [{resize_binary_match[3]}, {resize_binary_match[4]}], "
            f"method='{resize_binary_match[6]}', "
            f"target_shape=[_tensor_shape_list({resize_binary_match[2]})[0], "
            f"_tensor_shape_list({resize_binary_match[2]})[1], "
            f"{resize_binary_match[3]}, {resize_binary_match[4]}], "
            f"align_corners={resize_binary_match[7]}, "
            f"half_pixel_centers={resize_binary_match[8]}, channel_last=False)"
        )
        cf_aliases.add(str(resize_binary_match[1]))
        if consumer_name != str(resize_binary_match[1]):
            cf_aliases.add(consumer_name)
        changed = True
    pidnet_forced_scale4_mul_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\(torch\.mul\((?P<input>[A-Za-z0-9_]+), "
        r"self\.(?P<const_attr>[A-Za-z0-9_]+)\), \[1, 1, (?P<c>\d+), 1\]\)$"
    )
    pidnet_forced_scale4_add_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\("
        r"(?P<input>[A-Za-z0-9_]+), self\.(?P<const_attr>[A-Za-z0-9_]+), \[1, 1, 1, (?P<c>\d+)\]\)$"
    )
    for index, line in enumerate(lines):
        pidnet_forced_scale4_mul_match = pidnet_forced_scale4_mul_re.match(line)
        if pidnet_forced_scale4_mul_match is not None:
            lines[index] = (
                f"{pidnet_forced_scale4_mul_match.group('indent')}{pidnet_forced_scale4_mul_match.group('lhs')} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_forced_scale4_mul_match.group('input')}, "
                f"torch.reshape(self.{pidnet_forced_scale4_mul_match.group('const_attr')}, [1, {pidnet_forced_scale4_mul_match.group('c')}, 1, 1])), [1, {pidnet_forced_scale4_mul_match.group('c')}, 1, 1])"
            )
            changed = True
            continue
        pidnet_forced_scale4_add_match = pidnet_forced_scale4_add_re.match(line)
        if pidnet_forced_scale4_add_match is not None:
            lines[index] = (
                f"{pidnet_forced_scale4_add_match.group('indent')}{pidnet_forced_scale4_add_match.group('lhs0')}, "
                f"{pidnet_forced_scale4_add_match.group('lhs1')} = _align_binary_inputs_to_anchor("
                f"{pidnet_forced_scale4_add_match.group('input')}, "
                f"torch.reshape(self.{pidnet_forced_scale4_add_match.group('const_attr')}, [1, {pidnet_forced_scale4_add_match.group('c')}, 1, 1]), "
                f"[1, {pidnet_forced_scale4_add_match.group('c')}, 1, 1])"
            )
            changed = True
    function_def_re = re.compile(
        r"^\s*def\s+[A-Za-z0-9_]+\((?P<params>[^\)]*)\):$"
    )
    current_function_assigned: set[str] = set()
    current_function_defined: set[str] = set()
    for index, line in enumerate(lines):
        if line.startswith("    def "):
            current_function_assigned = set()
            current_function_defined = set()
            function_def_match = function_def_re.match(line)
            if function_def_match is not None:
                raw_params = str(function_def_match.group("params"))
                for raw_param in raw_params.split(","):
                    param = str(raw_param).strip()
                    if param == "":
                        continue
                    param_name = param.split(":", 1)[0].split("=", 1)[0].strip()
                    if param_name != "":
                        current_function_defined.add(param_name)
            continue
        orphan_rank3_permute_match = orphan_rank3_permute_re.match(line)
        if orphan_rank3_permute_match is not None:
            src = str(orphan_rank3_permute_match.group("src"))
            alias_source = generic_alias_sources.get(src, "")
            if (
                src not in current_function_defined
                and alias_source != ""
            ):
                lhs = str(orphan_rank3_permute_match.group("lhs"))
                indent = str(orphan_rank3_permute_match.group("indent"))
                if _is_known_cf_name(alias_source, singleton_cf_vars):
                    lines[index] = f"{indent}{lhs} = {alias_source}"
                else:
                    lines[index] = (
                        f"{indent}{lhs} = _torch_permute({alias_source}, [0, 2, 1])"
                    )
                changed = True
        generic_assign_match = re.match(r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=", lines[index])
        if generic_assign_match is not None:
            assigned_name = str(generic_assign_match.group("lhs"))
            current_function_assigned.add(assigned_name)
            current_function_defined.add(assigned_name)
    for index, line in enumerate(lines):
        parsed_pidnet_spp_scale4_mul_reshape_variant = _parse_raw_pidnet_scale4_mul_reshape_variant_assign(line)
        if parsed_pidnet_spp_scale4_mul_reshape_variant is not None:
            (
                pidnet_spp_scale4_mul_reshape_variant_indent,
                pidnet_spp_scale4_mul_reshape_variant_lhs,
                pidnet_spp_scale4_mul_reshape_variant_input,
                pidnet_spp_scale4_mul_reshape_variant_const_expr,
                pidnet_spp_scale4_mul_reshape_variant_c,
            ) = parsed_pidnet_spp_scale4_mul_reshape_variant
            lines[index] = (
                f"{pidnet_spp_scale4_mul_reshape_variant_indent}{pidnet_spp_scale4_mul_reshape_variant_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale4_mul_reshape_variant_input}, "
                f"torch.reshape({pidnet_spp_scale4_mul_reshape_variant_const_expr}, [1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])), "
                f"[1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])"
            )
            changed = True
            continue
        parsed_pidnet_spp_scale4_mul_reshape_variant_reversed = (
            _parse_raw_pidnet_scale4_mul_reshape_variant_reversed_assign(line)
        )
        if parsed_pidnet_spp_scale4_mul_reshape_variant_reversed is not None:
            (
                pidnet_spp_scale4_mul_reshape_variant_indent,
                pidnet_spp_scale4_mul_reshape_variant_lhs,
                pidnet_spp_scale4_mul_reshape_variant_input,
                pidnet_spp_scale4_mul_reshape_variant_const_expr,
                pidnet_spp_scale4_mul_reshape_variant_c,
            ) = parsed_pidnet_spp_scale4_mul_reshape_variant_reversed
            lines[index] = (
                f"{pidnet_spp_scale4_mul_reshape_variant_indent}{pidnet_spp_scale4_mul_reshape_variant_lhs} = "
                f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale4_mul_reshape_variant_input}, "
                f"torch.reshape({pidnet_spp_scale4_mul_reshape_variant_const_expr}, [1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])), "
                f"[1, {pidnet_spp_scale4_mul_reshape_variant_c}, 1, 1])"
            )
            changed = True
            continue
        continue
    for index, line in enumerate(lines):
        parsed_singleton_const_anchor_fix = _parse_singleton_const_anchor_fix_assign(line)
        if parsed_singleton_const_anchor_fix is None:
            continue
        channel_count = _find_recent_singleton_cf_channel_count(index)
        if channel_count is None:
            continue
        singleton_indent, singleton_lhs0, singleton_lhs1, singleton_input, singleton_const_attr = parsed_singleton_const_anchor_fix
        lines[index] = (
            f"{singleton_indent}{singleton_lhs0}, "
            f"{singleton_lhs1} = _align_binary_inputs_to_anchor("
            f"{singleton_input}, "
            f"torch.reshape(self.{singleton_const_attr}, [1, {channel_count}, 1, 1]), "
            f"[1, {channel_count}, 1, 1])"
        )
        changed = True
    for index, line in enumerate(lines):
        parsed_pidnet_spp_scale3_mul_out = _parse_raw_pidnet_align_binary_out_assign(line, "mul")
        if (
            parsed_pidnet_spp_scale3_mul_out is None
            or parsed_pidnet_spp_scale3_mul_out[4][0] != 1
            or parsed_pidnet_spp_scale3_mul_out[4][2] != 1
        ):
            continue
        if int(parsed_pidnet_spp_scale3_mul_out[4][3]) == 1:
            continue
        (
            pidnet_spp_scale3_mul_out_indent,
            pidnet_spp_scale3_mul_out_lhs,
            pidnet_spp_scale3_mul_out_a,
            pidnet_spp_scale3_mul_out_b,
            pidnet_spp_scale3_mul_out_shape,
        ) = parsed_pidnet_spp_scale3_mul_out
        lines[index] = (
            f"{pidnet_spp_scale3_mul_out_indent}{pidnet_spp_scale3_mul_out_lhs} = "
            f"_align_tensor_to_target_shape(torch.mul({pidnet_spp_scale3_mul_out_a}, "
            f"{pidnet_spp_scale3_mul_out_b}), [1, {pidnet_spp_scale3_mul_out_shape[1]}, 1, 1])"
        )
        changed = True
    for index in range(len(lines) - 1):
        generic_alias_assign = _parse_simple_assignment_line(lines[index])
        rank3_reshape_assign = _parse_rank3_reshape_from_rank4_source_assign(lines[index + 1])
        if (
            generic_alias_assign is None
            or not _is_simple_identifier_expr(generic_alias_assign[2])
            or rank3_reshape_assign is None
            or str(rank3_reshape_assign[2]) != str(generic_alias_assign[1])
        ):
            continue
        lhs = str(generic_alias_assign[1])
        rhs = _strip_outer_parentheses(str(generic_alias_assign[2]).strip())
        if "_nhwc" not in lhs or not _is_known_cf_name(rhs, singleton_cf_vars):
            continue
        lhs_exact_shape = _model_ir_exact_shape(lhs)
        rhs_exact_shape = _model_ir_exact_shape(rhs)
        nhwc_target_shape = lhs_exact_shape
        if nhwc_target_shape is None and rhs_exact_shape is not None and len(rhs_exact_shape) == 4:
            nhwc_target_shape = [
                int(rhs_exact_shape[0]),
                int(rhs_exact_shape[2]),
                int(rhs_exact_shape[3]),
                int(rhs_exact_shape[1]),
            ]
        if nhwc_target_shape is None or len(nhwc_target_shape) != 4:
            continue
        rank3_target_shape = (
            [
                int(value.strip())
                for value in str(
                    rank3_reshape_match.group("resolved_shape")
                    or rank3_reshape_match.group("shape")
                ).split(",")
                if value.strip()
            ]
            if rank3_reshape_match is not None
            else [int(value) for value in rank3_reshape_assign[3]]
        )
        if (
            len(rank3_target_shape) != 3
            or int(rank3_target_shape[0]) != int(nhwc_target_shape[0])
            or int(rank3_target_shape[2]) != int(nhwc_target_shape[3])
            or int(rank3_target_shape[1]) != int(np.prod(nhwc_target_shape[1:3], dtype=np.int64))
        ):
            continue
        indent = str(generic_alias_assign[0])
        lines[index] = (
            f"{indent}{lhs} = _align_tensor_to_target_shape("
            f"{rhs}.permute(0, 2, 3, 1).contiguous(), {nhwc_target_shape})"
        )
        changed = True
    finalized_lines = _fold_channel_first_hardsigmoid_gate_conv_bridges(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True
    finalized_lines = _rewrite_channel_first_se_scale_binary_bridges(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True
    finalized_lines = _repair_channel_last_gap_conv_inputs(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True
    finalized_lines = _fold_channel_first_hardsigmoid_gate_conv_bridges(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True
    finalized_lines = _rewrite_channel_first_se_scale_binary_bridges(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True
    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")








def _apply_fast_precanonicalize_repairs(package_path: Path) -> None:
    model_path = package_path / "model.py"
    if not model_path.exists():
        return
    lines = model_path.read_text(encoding="utf-8").splitlines()
    repair_context = _build_fast_precanonicalize_repair_context(lines)
    changed = False
    cf_like_names: set[str] = set(repair_context.cf_like_names)
    nhwc_like_names: set[str] = set(repair_context.nhwc_like_names)
    binary_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<expr>torch\.(?:mul|add|sub|div|minimum|maximum)\(.+\))$"
    )
    simple_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>[A-Za-z0-9_]+)$"
    )
    rank3_reshape_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<input>[A-Za-z0-9_]+), "
        r"(?:_resolve_reshape_shape\(\[(?P<resolved_shape>[0-9,\- ]+)\], (?P=input), allow_zero=False\)|\[(?P<shape>[0-9,\- ]+)\])\)$"
    )
    channel_last_prelu_consumer_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.prelu_[0-9]+\((?P<input>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)\.permute\(0, 2, 3, 1\)\.contiguous\(\)$"
    )
    permuted_conv_input_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)$"
    )
    depth_to_space_nhwc_gather_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\[:, \[(?P<indices>[0-9,\s-]+)\], :, :\]$"
    )
    aligned_bn_const_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.(?P<op>mul|add)\((?P<input>[A-Za-z0-9_]+), self\.(?P<const_attr>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)\]\)$"
    )
    aligned_bn_const_reshaped_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.(?P<op>mul|add)\((?P<input>[A-Za-z0-9_]+), torch\.reshape\(self\.(?P<const_attr>[A-Za-z0-9_]+), \[1, (?P<reshape_c>\d+), 1, 1\]\)\), \[(?P<n>\d+), (?P<c0>\d+), (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    aligned_binary_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\(torch\.(?P<op>mul|add|sub|div|minimum|maximum)\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), (?P<c>\d+)\]\)$"
    )
    aligned_rank4_any_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\((?P<expr>.+), \[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    aligned_scalar_binary_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\("
        r"torch\.(?P<op>mul|add|sub|div)\((?P<input>[A-Za-z0-9_]+), "
        r"(?P<scalar>[-+]?(?:[0-9]*\.[0-9]+|[0-9]+(?:\.[0-9]*)?)(?:[eE][-+]?\d+)?)\), "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    for index, line in enumerate(lines[:-1]):
        simple_alias_match = simple_alias_re.match(line)
        if simple_alias_match is not None:
            next_rank3_reshape_match = (
                rank3_reshape_re.match(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            next_channel_last_prelu_consumer_match = (
                channel_last_prelu_consumer_re.match(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            next_permuted_conv_input_match = (
                permuted_conv_input_re.match(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            reshape_target_dims: List[int] = []
            reshape_target_text = ""
            reshape_feature_dim: int | None = None
            rhs_name = str(simple_alias_match.group("rhs"))
            rhs_producer_module = repair_context.module_output_producers.get(rhs_name)
            rhs_producer_channels = (
                repair_context.conv_block_out_channels.get(rhs_producer_module)
                if rhs_producer_module is not None
                else None
            )
            if next_rank3_reshape_match is not None:
                reshape_target_text = str(
                    next_rank3_reshape_match.group("resolved_shape")
                    or next_rank3_reshape_match.group("shape")
                    or ""
                )
                try:
                    reshape_target_dims = [
                        int(token.strip())
                        for token in reshape_target_text.split(",")
                        if token.strip() != ""
                    ]
                except Exception:
                    reshape_target_dims = []
                if len(reshape_target_dims) == 3 and all(
                    int(dim) > 0 for dim in reshape_target_dims
                ):
                    reshape_feature_dim = int(reshape_target_dims[-1])
            if (
                next_rank3_reshape_match is not None
                and str(next_rank3_reshape_match.group("input")) == str(simple_alias_match.group("lhs"))
                and (
                    rhs_name in cf_like_names
                    or rhs_name.endswith("_cf")
                    or rhs_name.endswith("_out_cf")
                )
                and (
                    "_nhwc" in str(simple_alias_match.group("lhs"))
                    or (
                        rhs_producer_channels is not None
                        and reshape_feature_dim is not None
                        and int(rhs_producer_channels) == int(reshape_feature_dim)
                        and not str(simple_alias_match.group("lhs")).endswith("_cf")
                    )
                    or (
                        reshape_feature_dim is not None
                        and len(reshape_target_dims) == 3
                        and int(reshape_target_dims[1]) > int(reshape_target_dims[2])
                        and not str(simple_alias_match.group("lhs")).endswith("_cf")
                    )
                )
            ):
                lines[index] = (
                    f"{simple_alias_match.group('indent')}{simple_alias_match.group('lhs')} = "
                    f"{simple_alias_match.group('rhs')}.permute(0, 2, 3, 1).contiguous()"
                )
                nhwc_like_names.add(str(simple_alias_match.group("lhs")))
                changed = True
                line = lines[index]
                simple_alias_match = None
            elif (
                "_nhwc" in str(simple_alias_match.group("lhs"))
                and (
                    rhs_name in cf_like_names
                    or rhs_name.endswith("_cf")
                    or rhs_name.endswith("_out_cf")
                )
                and (
                    (
                        next_channel_last_prelu_consumer_match is not None
                        and str(next_channel_last_prelu_consumer_match.group("input")) == str(simple_alias_match.group("lhs"))
                    )
                    or (
                        next_permuted_conv_input_match is not None
                        and str(next_permuted_conv_input_match.group("input")) == str(simple_alias_match.group("lhs"))
                    )
                )
            ):
                lhs_name = str(simple_alias_match.group("lhs"))
                lhs_exact_shape = repair_context.static_shapes.get(lhs_name)
                if lhs_exact_shape is not None and len(lhs_exact_shape) == 4:
                    lines[index] = (
                        f"{simple_alias_match.group('indent')}{lhs_name} = "
                        f"_align_tensor_to_target_shape({simple_alias_match.group('rhs')}.permute(0, 2, 3, 1).contiguous(), {lhs_exact_shape})"
                    )
                else:
                    lines[index] = (
                        f"{simple_alias_match.group('indent')}{lhs_name} = "
                        f"{simple_alias_match.group('rhs')}.permute(0, 2, 3, 1).contiguous()"
                    )
                nhwc_like_names.add(lhs_name)
                changed = True
                line = lines[index]
                simple_alias_match = None
        if simple_alias_match is not None:
            rhs_name = str(simple_alias_match.group("rhs"))
            if (
                rhs_name in cf_like_names
                or rhs_name.endswith("_cf")
                or rhs_name.endswith("_out_cf")
            ):
                cf_like_names.add(str(simple_alias_match.group("lhs")))
            if rhs_name in nhwc_like_names or "_nhwc" in rhs_name:
                nhwc_like_names.add(str(simple_alias_match.group("lhs")))
        rewritten_split_line, split_cf_outputs = _repair_split_axis_from_consumers(
            line,
            index,
            lines,
            cf_like_names,
            nhwc_like_names,
            repair_context,
        )
        if rewritten_split_line is not None:
            lines[index] = rewritten_split_line
            cf_like_names.update(split_cf_outputs)
            changed = True
            line = rewritten_split_line
        aligned_scalar_binary_match = aligned_scalar_binary_re.match(line)
        if aligned_scalar_binary_match is not None and index > 0:
            prev_aligned_rank4_match = aligned_rank4_any_re.match(lines[index - 1])
            next_aligned_rank4_match = (
                aligned_rank4_any_re.match(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            next_apply_softmax_assign = (
                _parse_apply_softmax_assign(lines[index + 1])
                if index + 1 < len(lines)
                else None
            )
            if (
                prev_aligned_rank4_match is not None
                and str(prev_aligned_rank4_match.group("lhs")) == str(aligned_scalar_binary_match.group("input"))
            ):
                prev_shape = [
                    int(prev_aligned_rank4_match.group("n")),
                    int(prev_aligned_rank4_match.group("d1")),
                    int(prev_aligned_rank4_match.group("d2")),
                    int(prev_aligned_rank4_match.group("d3")),
                ]
                current_shape = [
                    int(aligned_scalar_binary_match.group("n")),
                    int(aligned_scalar_binary_match.group("d1")),
                    int(aligned_scalar_binary_match.group("d2")),
                    int(aligned_scalar_binary_match.group("d3")),
                ]
                consumer_shape = None
                current_lhs = str(aligned_scalar_binary_match.group("lhs"))
                if (
                    next_aligned_rank4_match is not None
                    and re.search(
                        rf"\b{re.escape(current_lhs)}\b",
                        str(next_aligned_rank4_match.group("expr")),
                    )
                    is not None
                ):
                    consumer_shape = [
                        int(next_aligned_rank4_match.group("n")),
                        int(next_aligned_rank4_match.group("d1")),
                        int(next_aligned_rank4_match.group("d2")),
                        int(next_aligned_rank4_match.group("d3")),
                    ]
                elif (
                    next_apply_softmax_assign is not None
                    and str(next_apply_softmax_assign[2]) == current_lhs
                ):
                    consumer_shape = [
                        int(value) for value in next_apply_softmax_assign[5]
                    ]
                if (
                    consumer_shape == prev_shape
                    and current_shape != prev_shape
                    and current_shape == [prev_shape[0], prev_shape[2], prev_shape[1], prev_shape[3]]
                ):
                    lines[index] = (
                        f"{aligned_scalar_binary_match.group('indent')}{current_lhs} = "
                        f"_align_tensor_to_target_shape(torch.{aligned_scalar_binary_match.group('op')}("
                        f"{aligned_scalar_binary_match.group('input')}, {aligned_scalar_binary_match.group('scalar')}), "
                        f"[{prev_shape[0]}, {prev_shape[1]}, {prev_shape[2]}, {prev_shape[3]}])"
                    )
                    changed = True
        if _repair_dynamic_cf_binary_anchor_at(
            index,
            lines,
            cf_like_names,
            repair_context,
        ):
            changed = True
            line = lines[index]
        aligned_binary_match = aligned_binary_re.match(line)
        aligned_binary_assign = _parse_aligned_binary_assign_with_shape(line)
        if aligned_binary_match is not None or aligned_binary_assign is not None:
            rewritten_binary_line, binary_lhs = _repair_binary_alignment_layout(
                line,
                index,
                lines,
                cf_like_names,
                nhwc_like_names,
                repair_context,
            )
            if rewritten_binary_line is not None:
                lines[index] = rewritten_binary_line
                if binary_lhs is not None:
                    cf_like_names.add(binary_lhs)
                    binary_shape_match = re.search(r"\[(?P<shape>[0-9, ]+)\]\)$", rewritten_binary_line)
                    if binary_shape_match is not None:
                        repair_context.static_shapes[binary_lhs] = _parse_int_list_literal(
                            str(binary_shape_match.group("shape"))
                        )
                changed = True
                line = rewritten_binary_line
                aligned_binary_match = None
                aligned_binary_assign = None
        if aligned_binary_match is not None:
            arg_a = str(aligned_binary_match.group("a"))
            arg_b = str(aligned_binary_match.group("b"))
            next_aligned_bn_match = aligned_bn_const_re.match(lines[index + 1])
            next_return_value = _parse_simple_return_identifier(lines[index + 1])
            next_resize_match = _parse_apply_resize_assign(lines[index + 1])
            current_shape_hint = _fast_precanonicalize_rank4_layout_hint(
                [
                    int(aligned_binary_match.group("n")),
                    int(aligned_binary_match.group("h")),
                    int(aligned_binary_match.group("w")),
                    int(aligned_binary_match.group("c")),
                ],
                preferred_channel_count=_fast_precanonicalize_preferred_channel_count(
                    arg_a,
                    cf_like_names,
                    nhwc_like_names,
                    repair_context,
                    shape_hint=[
                        int(aligned_binary_match.group("n")),
                        int(aligned_binary_match.group("h")),
                        int(aligned_binary_match.group("w")),
                        int(aligned_binary_match.group("c")),
                    ],
                ),
            )
            operands_are_cf_like = (
                (
                    arg_a in cf_like_names
                    or arg_a.endswith("_cf")
                    or arg_a.endswith("_out_cf")
                    or (arg_a.endswith("_in") and not arg_a.endswith("_in_nhwc"))
                )
                and (
                    arg_b in cf_like_names
                    or arg_b.endswith("_cf")
                    or arg_b.endswith("_out_cf")
                    or (arg_b.endswith("_in") and not arg_b.endswith("_in_nhwc"))
                )
            )
            if (
                current_shape_hint != "cf"
                and
                operands_are_cf_like
                and (
                    (
                        next_aligned_bn_match is not None
                        and str(next_aligned_bn_match.group("input")) == str(aligned_binary_match.group("lhs"))
                        and int(next_aligned_bn_match.group("c")) == int(aligned_binary_match.group("c"))
                    )
                    or (
                        next_return_value is not None
                        and str(next_return_value) == str(aligned_binary_match.group("lhs"))
                    )
                    or (
                        next_resize_match is not None
                        and str(next_resize_match[2]) == str(aligned_binary_match.group("lhs"))
                        and not bool(next_resize_match[9])
                        and int(next_resize_match[6][3]) == int(aligned_binary_match.group("c"))
                    )
                )
            ):
                lines[index] = (
                    f"{aligned_binary_match.group('indent')}{aligned_binary_match.group('lhs')} = "
                    f"_align_tensor_to_target_shape(torch.{aligned_binary_match.group('op')}("
                    f"{arg_a}, {arg_b}), "
                    f"[{aligned_binary_match.group('n')}, {aligned_binary_match.group('c')}, "
                    f"{aligned_binary_match.group('h')}, {aligned_binary_match.group('w')}])"
                )
                cf_like_names.add(str(aligned_binary_match.group("lhs")))
                changed = True
        apply_resize_match = _parse_apply_resize_assign(line)
        if apply_resize_match is not None:
            rewritten_resize_line, resize_lhs = _repair_cf_resize_target_shape(
                line,
                index,
                lines,
                cf_like_names,
                nhwc_like_names,
                repair_context,
            )
            if rewritten_resize_line is not None:
                lines[index] = rewritten_resize_line
                if resize_lhs is not None:
                    cf_like_names.add(resize_lhs)
                    resize_shape_match = re.search(r"target_shape=\[(?P<shape>[0-9, ]+)\]", rewritten_resize_line)
                    if resize_shape_match is not None:
                        repair_context.static_shapes[resize_lhs] = _parse_int_list_literal(
                            str(resize_shape_match.group("shape"))
                        )
                changed = True
                line = rewritten_resize_line
                apply_resize_match = _parse_apply_resize_assign(line)
        if apply_resize_match is not None:
            resize_indent, resize_lhs_name, input_name, out_h, out_w, resize_method, current_shape, resize_align_corners, resize_half_pixel_centers, _ = apply_resize_match
            next_aligned_bn_match = None
            if index + 1 < len(lines):
                next_aligned_bn_match = aligned_bn_const_re.match(lines[index + 1])
                if next_aligned_bn_match is None:
                    next_aligned_bn_match = aligned_bn_const_reshaped_re.match(lines[index + 1])
            next_bn_channel_count = None
            if (
                next_aligned_bn_match is not None
                and str(next_aligned_bn_match.group("input")) == resize_lhs_name
            ):
                next_bn_channel_count = repair_context.const_channel_counts.get(
                    str(next_aligned_bn_match.group("const_attr")),
                    None,
                )
            if (
                _fast_precanonicalize_rank4_layout_hint(
                    current_shape,
                    preferred_channel_count=_fast_precanonicalize_preferred_channel_count(
                        input_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=current_shape,
                    ),
                ) != "cf"
                and (
                    input_name in cf_like_names
                    or input_name.endswith("_cf")
                    or input_name.endswith("_out_cf")
                )
            ):
                preferred_channel_count = next_bn_channel_count
                if preferred_channel_count is None:
                    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                        resize_lhs_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=current_shape,
                    )
                if preferred_channel_count is None:
                    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                        input_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=current_shape,
                    )
                normalized_shape = _normalize_cf_rank4_shape(
                    current_shape,
                    preferred_channel_count=preferred_channel_count,
                    out_hw=(out_h, out_w),
                )
                lines[index] = (
                    f"{resize_indent}{resize_lhs_name} = _apply_resize("
                    f"{input_name}, [{out_h}, {out_w}], "
                    f"method='{resize_method}', "
                    f"target_shape={repr(normalized_shape)}, "
                    f"align_corners={resize_align_corners}, "
                    f"half_pixel_centers={resize_half_pixel_centers}, channel_last=False)"
                )
                cf_like_names.add(resize_lhs_name)
                changed = True
        apply_pool2d_assign = _parse_apply_pool2d_assign_with_shape(line)
        if apply_pool2d_assign is not None:
            repaired_nhwc_avg_pool_bridge, nhwc_bridge_names = _repair_nhwc_average_pool_binary_bridge(
                index,
                lines,
                cf_like_names,
                nhwc_like_names,
                repair_context,
            )
            if repaired_nhwc_avg_pool_bridge:
                nhwc_like_names.update(nhwc_bridge_names)
                cf_like_names.difference_update(nhwc_bridge_names)
                normalized_bridge_shape = _normalize_nhwc_rank4_shape(
                    apply_pool2d_assign[4],
                    preferred_channel_count=_fast_precanonicalize_preferred_channel_count(
                        str(apply_pool2d_assign[1]),
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=apply_pool2d_assign[4],
                    ),
                )
                for updated_name in nhwc_bridge_names:
                    repair_context.static_shapes[updated_name] = list(normalized_bridge_shape)
                changed = True
                line = lines[index]
                apply_pool2d_assign = _parse_apply_pool2d_assign_with_shape(line)
        if apply_pool2d_assign is not None:
            pool_indent, pool_lhs_name, pool_input_name, pool_rest, pool_shape, pool_is_max, pool_channel_last = apply_pool2d_assign
            if (
                not pool_channel_last
                and pool_is_max
                and _fast_precanonicalize_has_channel_last_spatial_consumer(
                    pool_lhs_name,
                    index,
                    lines,
                    repair_context,
                )
            ):
                lines[index] = (
                    f"{pool_indent}{pool_lhs_name} = _apply_pool2d("
                    f"{pool_input_name}, {pool_rest}, "
                    f"target_shape={repr(pool_shape)}, "
                    f"is_max_pool={pool_is_max}, channel_last=True)"
                )
                nhwc_like_names.add(pool_lhs_name)
                changed = True
                line = lines[index]
        apply_pool2d_assign = _parse_apply_pool2d_assign_with_shape(line)
        if apply_pool2d_assign is not None:
            pool_indent, pool_lhs_name, pool_input_name, pool_rest, pool_shape, pool_is_max, pool_channel_last = apply_pool2d_assign
            input_is_immediate_nhwc_bridge = _has_immediate_rank4_permute_source(
                lines,
                index,
                pool_input_name,
                [0, 2, 3, 1],
            )
            input_is_immediate_cf_bridge = _has_immediate_rank4_permute_source(
                lines,
                index,
                pool_input_name,
                [0, 3, 1, 2],
            )
            if (
                not pool_channel_last
                and (
                    input_is_immediate_nhwc_bridge
                    or (
                        not input_is_immediate_cf_bridge
                        and _fast_precanonicalize_is_nhwc_like(
                            pool_input_name,
                            nhwc_like_names,
                            repair_context,
                        )
                        and not _fast_precanonicalize_is_cf_like(
                            pool_input_name,
                            cf_like_names,
                            repair_context,
                        )
                    )
                )
            ):
                preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                    pool_lhs_name,
                    cf_like_names,
                    nhwc_like_names,
                    repair_context,
                    shape_hint=pool_shape,
                )
                if preferred_channel_count is None:
                    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                        pool_input_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=pool_shape,
                    )
                normalized_shape = _normalize_nhwc_rank4_shape(
                    pool_shape,
                    preferred_channel_count=preferred_channel_count,
                )
                lines[index] = (
                    f"{pool_indent}{pool_lhs_name} = _apply_pool2d("
                    f"{pool_input_name}, {pool_rest}, "
                    f"target_shape={repr(normalized_shape)}, "
                    f"is_max_pool={pool_is_max}, channel_last=True)"
                )
                nhwc_like_names.add(pool_lhs_name)
                changed = True
                line = lines[index]
                apply_pool2d_assign = _parse_apply_pool2d_assign_with_shape(line)
        if apply_pool2d_assign is not None:
            rewritten_pool_line, pool_lhs = _repair_cf_pool_target_shape(
                line,
                index,
                lines,
                cf_like_names,
                nhwc_like_names,
                repair_context,
            )
            if rewritten_pool_line is not None:
                lines[index] = rewritten_pool_line
                if pool_lhs is not None:
                    cf_like_names.add(pool_lhs)
                    pool_shape_match = re.search(r"target_shape=\[(?P<shape>[0-9, ]+)\]", rewritten_pool_line)
                    if pool_shape_match is not None:
                        repair_context.static_shapes[pool_lhs] = _parse_int_list_literal(
                            str(pool_shape_match.group("shape"))
                        )
                changed = True
                line = rewritten_pool_line
                apply_pool2d_assign = _parse_apply_pool2d_assign_with_shape(line)
        if apply_pool2d_assign is not None:
            pool_indent, pool_lhs_name, input_name, pool_rest, pool_shape, pool_is_max, pool_channel_last = apply_pool2d_assign
            prev_const_pad_assign = (
                _parse_constant_pad_assign(lines[index - 1])
                if index > 0
                else None
            )
            next_nonempty_line = ""
            for lookahead in range(index + 1, min(len(lines), index + 4)):
                if lines[lookahead].strip() == "":
                    continue
                next_nonempty_line = str(lines[lookahead])
                break
            next_lrn_assign = _parse_local_response_norm_assign(next_nonempty_line)
            if (
                _fast_precanonicalize_rank4_layout_hint(
                    pool_shape,
                    preferred_channel_count=_fast_precanonicalize_preferred_channel_count(
                        input_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=pool_shape,
                    ),
                ) != "cf"
                and
                pool_channel_last
                and pool_is_max
                and prev_const_pad_assign is not None
                and str(prev_const_pad_assign[1]) == input_name
                and (
                    str(prev_const_pad_assign[2]) in cf_like_names
                    or str(prev_const_pad_assign[2]).endswith("_cf")
                    or str(prev_const_pad_assign[2]).endswith("_out_cf")
                )
            ):
                pad_values = [int(value) for value in prev_const_pad_assign[3]]
                immediate_permuted_conv_consumers = 0
                for lookahead in range(index + 1, min(len(lines), index + 4)):
                    permuted_conv_match = permuted_conv_input_re.match(lines[lookahead])
                    if (
                        permuted_conv_match is not None
                        and str(permuted_conv_match.group("input")) == pool_lhs_name
                    ):
                        immediate_permuted_conv_consumers += 1
                if pad_values == [1, 1, 1, 1] and immediate_permuted_conv_consumers > 0:
                    current_shape = pool_shape
                    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                        pool_lhs_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=current_shape,
                    )
                    if preferred_channel_count is None:
                        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
                            input_name,
                            cf_like_names,
                            nhwc_like_names,
                            repair_context,
                            shape_hint=current_shape,
                        )
                    normalized_shape = _normalize_cf_rank4_shape(
                        current_shape,
                        preferred_channel_count=preferred_channel_count,
                    )
                    lines[index] = (
                        f"{pool_indent}{pool_lhs_name} = _apply_pool2d("
                        f"{input_name}, {pool_rest}, "
                        f"target_shape={repr(normalized_shape)}, "
                        f"is_max_pool={pool_is_max}, channel_last=False)"
                    )
                    cf_like_names.add(pool_lhs_name)
                    changed = True
                    continue
            if (
                not pool_channel_last
                and (
                    input_name in cf_like_names
                    or input_name.endswith("_cf")
                    or input_name.endswith("_out_cf")
                )
                and (
                    next_lrn_assign is not None and str(next_lrn_assign[2]) == pool_lhs_name
                )
            ):
                input_channel_count = repair_context.conv_block_out_channels.get(
                    repair_context.module_output_producers.get(input_name, ""),
                    None,
                )
                preferred_channel_count = (
                    int(input_channel_count)
                    if input_channel_count is not None
                    else _fast_precanonicalize_preferred_channel_count(
                        pool_lhs_name,
                        cf_like_names,
                        nhwc_like_names,
                        repair_context,
                        shape_hint=pool_shape,
                    )
                )
                if preferred_channel_count is None:
                    preferred_channel_count = max(
                        pool_shape[1],
                        pool_shape[2],
                        pool_shape[3],
                    )
                normalized_shape = _normalize_cf_rank4_shape(
                    pool_shape,
                    preferred_channel_count=int(preferred_channel_count),
                )
                lines[index] = (
                    f"{pool_indent}{pool_lhs_name} = _apply_pool2d("
                    f"{input_name}, {pool_rest}, "
                    f"target_shape={repr(normalized_shape)}, "
                    f"is_max_pool={pool_is_max}, channel_last=False)"
                )
                cf_like_names.add(pool_lhs_name)
                changed = True
        dynamic_apply_pool2d_assign = _parse_dynamic_apply_pool2d_assign(line)
        if dynamic_apply_pool2d_assign is not None:
            dynamic_pool_indent, lhs, input_name, dynamic_pool_rest, dynamic_pool_shape_input, dynamic_pool_is_max = dynamic_apply_pool2d_assign
            input_is_immediate_nhwc_bridge = _has_immediate_rank4_permute_source(
                lines,
                index,
                input_name,
                [0, 2, 3, 1],
            )
            input_is_immediate_cf_bridge = _has_immediate_rank4_permute_source(
                lines,
                index,
                input_name,
                [0, 3, 1, 2],
            )
            if (
                dynamic_pool_shape_input == input_name
                and (
                    input_is_immediate_nhwc_bridge
                    or (
                        not input_is_immediate_cf_bridge
                        and _fast_precanonicalize_is_nhwc_like(
                            input_name,
                            nhwc_like_names,
                            repair_context,
                        )
                        and not _fast_precanonicalize_is_cf_like(
                            input_name,
                            cf_like_names,
                            repair_context,
                        )
                    )
                )
            ):
                lines[index] = (
                    f"{dynamic_pool_indent}{lhs} = _apply_pool2d("
                    f"{input_name}, {dynamic_pool_rest}, "
                    f"target_shape=_tensor_shape_list({dynamic_pool_shape_input}), "
                    f"is_max_pool={dynamic_pool_is_max}, channel_last=True)"
                )
                nhwc_like_names.add(lhs)
                changed = True
                continue
            if (
                not dynamic_pool_is_max
                and (
                    input_name in cf_like_names
                    or input_name.endswith("_cf")
                    or input_name.endswith("_out_cf")
                )
                and dynamic_pool_shape_input == input_name
            ):
                repaired_target_shape: list[int] | None = None
                for lookahead in range(index + 1, min(len(lines), index + 4)):
                    aligned_rank4_match = aligned_rank4_any_re.match(lines[lookahead])
                    if (
                        aligned_rank4_match is not None
                        and re.search(rf"\b{re.escape(lhs)}\b", str(aligned_rank4_match.group("expr"))) is not None
                    ):
                        repaired_target_shape = [
                            int(aligned_rank4_match.group("n")),
                            int(aligned_rank4_match.group("d1")),
                            int(aligned_rank4_match.group("d2")),
                            int(aligned_rank4_match.group("d3")),
                        ]
                        break
                    binary_consumer_match = binary_assign_re.match(lines[lookahead])
                    if (
                        binary_consumer_match is None
                        or re.search(rf"\b{re.escape(lhs)}\b", str(binary_consumer_match.group("expr"))) is None
                        or lookahead + 1 >= len(lines)
                    ):
                        continue
                    aligned_rank4_match = aligned_rank4_any_re.match(lines[lookahead + 1])
                    if (
                        aligned_rank4_match is not None
                        and re.search(
                            rf"\b{re.escape(str(binary_consumer_match.group('lhs')))}\b",
                            str(aligned_rank4_match.group("expr")),
                        ) is not None
                    ):
                        repaired_target_shape = [
                            int(aligned_rank4_match.group("n")),
                            int(aligned_rank4_match.group("d1")),
                            int(aligned_rank4_match.group("d2")),
                            int(aligned_rank4_match.group("d3")),
                        ]
                        break
                if repaired_target_shape is not None:
                    lines[index] = (
                        f"{dynamic_pool_indent}{lhs} = _apply_pool2d("
                        f"{input_name}, {dynamic_pool_rest}, "
                        f"target_shape={repr(repaired_target_shape)}, "
                        f"is_max_pool={dynamic_pool_is_max}, channel_last=False)"
                    )
                    cf_like_names.add(lhs)
                changed = True
        rewritten_concat_line, concat_lhs = _repair_concat_axis_from_input_layouts(
            line,
            cf_like_names,
            repair_context,
        )
        if rewritten_concat_line is not None:
            lines[index] = rewritten_concat_line
            if concat_lhs is not None:
                cf_like_names.add(concat_lhs)
            changed = True
            line = rewritten_concat_line
        aligned_bn_const_match = aligned_bn_const_re.match(line)
        if aligned_bn_const_match is not None:
            input_name = str(aligned_bn_const_match.group("input"))
            const_attr = str(aligned_bn_const_match.group("const_attr"))
            channel_count = repair_context.const_channel_counts.get(const_attr)
            if (
                (
                    input_name in cf_like_names
                    or input_name.endswith("_cf")
                    or input_name.endswith("_out_cf")
                )
                and (
                    "BatchNormalization" in const_attr
                    or "batch_normalization" in const_attr
                )
                and channel_count is not None
                and int(aligned_bn_const_match.group("c")) == channel_count
            ):
                lines[index] = (
                    f"{aligned_bn_const_match.group('indent')}{aligned_bn_const_match.group('lhs')} = "
                    f"_align_tensor_to_target_shape(torch.{aligned_bn_const_match.group('op')}("
                    f"{input_name}, torch.reshape(self.{const_attr}, [1, {aligned_bn_const_match.group('c')}, 1, 1])), "
                    f"[{aligned_bn_const_match.group('n')}, {aligned_bn_const_match.group('c')}, "
                    f"{aligned_bn_const_match.group('h')}, {aligned_bn_const_match.group('w')}])"
                )
                cf_like_names.add(str(aligned_bn_const_match.group("lhs")))
                changed = True
        aligned_bn_const_reshaped_match = aligned_bn_const_reshaped_re.match(line)
        if aligned_bn_const_reshaped_match is not None:
            input_name = str(aligned_bn_const_reshaped_match.group("input"))
            const_attr = str(aligned_bn_const_reshaped_match.group("const_attr"))
            reshape_channel_count = int(aligned_bn_const_reshaped_match.group("reshape_c"))
            if (
                (
                    input_name in cf_like_names
                    or input_name.endswith("_cf")
                    or input_name.endswith("_out_cf")
                )
                and (
                    "BatchNormalization" in const_attr
                    or "batch_normalization" in const_attr
                )
            ):
                normalized_shape = _normalize_cf_rank4_shape(
                    [
                        int(aligned_bn_const_reshaped_match.group("n")),
                        int(aligned_bn_const_reshaped_match.group("c0")),
                        int(aligned_bn_const_reshaped_match.group("h")),
                        int(aligned_bn_const_reshaped_match.group("w")),
                    ],
                    preferred_channel_count=reshape_channel_count,
                )
                lines[index] = (
                    f"{aligned_bn_const_reshaped_match.group('indent')}{aligned_bn_const_reshaped_match.group('lhs')} = "
                    f"_align_tensor_to_target_shape(torch.{aligned_bn_const_reshaped_match.group('op')}("
                    f"{input_name}, torch.reshape(self.{const_attr}, [1, {reshape_channel_count}, 1, 1])), "
                    f"{repr(normalized_shape)})"
                )
                cf_like_names.add(str(aligned_bn_const_reshaped_match.group("lhs")))
                changed = True
        local_response_norm_assign = _parse_local_response_norm_assign(line)
        if local_response_norm_assign is not None:
            input_name = str(local_response_norm_assign[2])
            if (
                input_name in cf_like_names
                or input_name.endswith("_cf")
                or input_name.endswith("_out_cf")
            ):
                lhs_name = str(local_response_norm_assign[1])
                cf_like_names.add(lhs_name)
                static_input_shape = repair_context.static_shapes.get(input_name, None)
                if static_input_shape is not None and len(static_input_shape) == 4:
                    repair_context.static_shapes[lhs_name] = [int(v) for v in list(static_input_shape)]
                nhwc_like_names.discard(lhs_name)
        rewritten_softmax_line, softmax_lhs = _repair_cf_softmax_axis(
            line,
            cf_like_names,
        )
        if rewritten_softmax_line is not None:
            lines[index] = rewritten_softmax_line
            if softmax_lhs is not None:
                cf_like_names.add(softmax_lhs)
            changed = True
        rewritten_reduce_max_line, reduce_max_lhs = _repair_cf_reduce_max_axis(
            line,
            cf_like_names,
        )
        if rewritten_reduce_max_line is not None:
            lines[index] = rewritten_reduce_max_line
            if reduce_max_lhs is not None:
                cf_like_names.add(reduce_max_lhs)
            changed = True
        rewritten_terminal_line, terminal_lhs = _repair_terminal_classifier_tail_layout(
            line,
            cf_like_names,
            repair_context,
        )
        if rewritten_terminal_line is not None:
            lines[index] = rewritten_terminal_line
            if terminal_lhs is not None:
                cf_like_names.add(terminal_lhs)
            changed = True
        if _repair_nhwc_buffer_binary_alignment_at(
            index,
            lines,
            repair_context,
        ):
            changed = True
        _propagate_cf_prelu_output(line, cf_like_names)
        depth_to_space_nhwc_gather_match = depth_to_space_nhwc_gather_re.match(line)
        if depth_to_space_nhwc_gather_match is not None:
            input_name = str(depth_to_space_nhwc_gather_match.group("input"))
            lhs_name = str(depth_to_space_nhwc_gather_match.group("lhs"))
            next_line = str(lines[index + 1]) if index + 1 < len(lines) else ""
            gathered_indices = [
                int(token.strip())
                for token in str(depth_to_space_nhwc_gather_match.group("indices")).split(",")
                if token.strip() != ""
            ]
            input_is_structural_cf = _fast_precanonicalize_is_cf_like(
                input_name,
                cf_like_names,
                repair_context,
            )
            if not input_is_structural_cf:
                for back in range(max(0, index - 4), index):
                    parsed_cat = _parse_torch_cat_inputs_and_dim(
                        _parse_simple_assignment_line(lines[back])[2]
                    ) if _parse_simple_assignment_line(lines[back]) is not None else None
                    assigned_lhs = (
                        _parse_simple_assignment_line(lines[back])[1]
                        if _parse_simple_assignment_line(lines[back]) is not None
                        else None
                    )
                    if assigned_lhs != input_name or parsed_cat is None or int(parsed_cat[1]) != 1:
                        continue
                    cat_inputs = [name.strip() for name in parsed_cat[0] if name.strip()]
                    if cat_inputs and all(
                        name in cf_like_names
                        or name.endswith("_cf")
                        or name.endswith("_out_cf")
                        for name in cat_inputs
                    ):
                        input_is_structural_cf = True
                        break
            resolved_input_name = _fast_precanonicalize_resolve_alias(
                input_name,
                repair_context.aliases,
            )
            input_shape = repair_context.static_shapes.get(input_name)
            if input_shape is None:
                input_shape = repair_context.static_shapes.get(resolved_input_name)
            if (
                input_is_structural_cf
            ):
                if input_shape is not None and len(input_shape) == 4:
                    repair_context.static_shapes[lhs_name] = [
                        int(input_shape[0]),
                        int(len(gathered_indices)),
                        int(input_shape[2]),
                        int(input_shape[3]),
                    ]
                cf_like_names.add(lhs_name)
                nhwc_like_names.discard(lhs_name)
                next_permuted_conv_match = (
                    permuted_conv_input_re.match(next_line)
                    if index + 1 < len(lines)
                    else None
                )
                if (
                    next_permuted_conv_match is not None
                    and str(next_permuted_conv_match.group("input")) == lhs_name
                ):
                    conv_module = str(next_permuted_conv_match.group("module"))
                    conv_in_channels = repair_context.conv_block_out_channels.get(conv_module)
                    if conv_in_channels is None:
                        conv_in_channels = repair_context.conv_block_in_channels.get(conv_module)
                    if conv_in_channels is None or int(conv_in_channels) == int(len(gathered_indices)):
                        lines[index + 1] = (
                            f"{next_permuted_conv_match.group('indent')}{next_permuted_conv_match.group('lhs')} = "
                            f"self.{conv_module}({lhs_name})"
                        )
                        cf_like_names.add(str(next_permuted_conv_match.group("lhs")))
                        changed = True
            is_depth_to_space_reorder = (
                _fast_precanonicalize_is_nhwc_like(
                    input_name,
                    nhwc_like_names,
                    repair_context,
                )
                and not input_is_structural_cf
                and (
                    "depth_to" in lhs_name.lower()
                    or "depthtospace" in lhs_name.lower()
                    or (
                        f"= {lhs_name}" in next_line
                        and "_depth_to_space_" in next_line
                    )
                )
            )
            if is_depth_to_space_reorder:
                lines[index] = (
                    f"{depth_to_space_nhwc_gather_match.group('indent')}"
                    f"{lhs_name} = {input_name}[:, :, :, "
                    f"[{depth_to_space_nhwc_gather_match.group('indices')}]]"
                )
                changed = True
        if _repair_cf_gather_slice_at(index, lines, cf_like_names):
            changed = True
        if _repair_singleton_reshape_cf_binary_at(
            index,
            lines,
            cf_like_names,
        ):
            changed = True

    if _repair_dynamic_cf_binary_anchor_shapes(
        lines,
        cf_like_names,
        repair_context,
    ):
        changed = True
    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _apply_pidnet_fast_precanonicalize_repairs(model_path)
    _apply_humanseg_fast_precanonicalize_repairs(model_path)
    _apply_structural_fast_precanonicalize_repairs(model_path)
    _apply_dynamic_score_sampling_stage_precanonicalize_repairs(model_path)
    _apply_shadowformer_fast_precanonicalize_repairs(model_path)
    _restore_channel_last_spatial_pool_chains(model_path)
    _apply_structural_fast_precanonicalize_repairs(model_path)


def _apply_fast_precanonicalize_repairs_until_stable(
    package_path: Path,
    *,
    max_passes: int = 4,
) -> None:
    model_path = package_path / "model.py"
    if not model_path.exists():
        return
    previous_source = model_path.read_text(encoding="utf-8")
    for _ in range(max(1, int(max_passes))):
        _apply_fast_precanonicalize_repairs(package_path)
        current_source = model_path.read_text(encoding="utf-8")
        if current_source == previous_source:
            break
        previous_source = current_source


def _apply_pidnet_fast_precanonicalize_repairs(model_path: Path) -> None:
    if not model_path.exists():
        return
    lines = model_path.read_text(encoding="utf-8").splitlines()
    if not _has_pidnet_skip_signature(lines):
        return
    changed = False
    pidnet_cf_add_sources: Set[str] = set()
    pidnet_cf_alias_sources: Set[str] = set()
    pidnet_cf_binary_sources: Set[str] = set()
    pidnet_cf_binary_tuple_sources: Set[str] = set()
    pidnet_cf_mul_sources: Set[str] = set()
    pidnet_cf_reduce_sum_sources: Set[str] = set()
    pidnet_cf_mean_sources: Set[str] = set()
    pidnet_cf_pad_sources: Set[str] = set()
    pidnet_cf_pool_sources: Set[str] = set()
    pidnet_const_alias_sources: Set[str] = set()
    pidnet_tuple_alias_pairs: Dict[str, tuple[str, str]] = {}

    def _pidnet_rank4_preferred_channel_count(
        shape: Sequence[int],
        *,
        out_hw: tuple[int, int] | None = None,
        prefer_last_if_nhwc: bool = False,
    ) -> int:
        if out_hw is not None:
            out_h, out_w = int(out_hw[0]), int(out_hw[1])
            spatial_candidates = [int(dim) for dim in shape[1:] if int(dim) in {out_h, out_w}]
            if len(spatial_candidates) >= 2:
                remaining = [int(dim) for dim in shape[1:] if int(dim) not in {out_h, out_w}]
                if len(remaining) == 1:
                    return remaining[0]
        if prefer_last_if_nhwc:
            return int(shape[3])
        return max(int(shape[1]), int(shape[2]), int(shape[3]))

    pidnet_binary_anchor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\("
        r"\(*\s*(?P<a>[A-Za-z0-9_]+)\s*\)*, \(*\s*(?P<b>[A-Za-z0-9_]+)\s*\)*, "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    pidnet_binary_anchor_tuple_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)(?P<ann>:\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*_align_binary_inputs_to_anchor\("
        r"\(*\s*(?P<a>[A-Za-z0-9_]+)\s*\)*, \(*\s*(?P<b>[A-Za-z0-9_]+)\s*\)*, "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    pidnet_mul_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\("
        r"torch\.mul\((?P<args>.+)\), "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    pidnet_bn_mul_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_align_tensor_to_target_shape\("
        r"torch\.mul\((?P<args>.+)\), "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    pidnet_scale3_anchor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\("
        r"\(*\s*(?P<input>[A-Za-z0-9_]+)\s*\)*, \(*\s*(?P<const_expr>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\)*, "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    pidnet_scale3_anchor_reversed_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs_to_anchor\("
        r"\(*\s*(?P<const_expr>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\)*, \(*\s*(?P<input>[A-Za-z0-9_]+)\s*\)*, "
        r"\[(?P<n>\d+), (?P<d1>\d+), (?P<d2>\d+), (?P<d3>\d+)\]\)$"
    )
    pidnet_permute_conv_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<src_expr>.+)\)$"
    )
    pidnet_plain_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*\(*(?P<input>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*$"
    )
    pidnet_tuple_alias_re = re.compile(
        r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*,\s*"
        r"(?P<lhs1>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*\)*\s*=\s*"
        r"\(*\s*\(*(?P<input0>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*,\s*\(*(?P<input1>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*\)*$"
    )
    pidnet_tuple_pair_alias_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*"
        r"\(*\s*\(*(?P<input0>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*,\s*\(*(?P<input1>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*\)*$"
    )
    pidnet_tuple_unpack_alias_re = re.compile(
        r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*,\s*"
        r"(?P<lhs1>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*\)*\s*=\s*\(*\s*\(*(?P<input>[A-Za-z0-9_]+)\)*\s*\)*$"
    )

    def _strip_pidnet_outer_parentheses(expr: str) -> str:
        stripped = str(expr).strip()
        while stripped.startswith("(") and stripped.endswith(")"):
            inner = stripped[1:-1].strip()
            if inner == "":
                break
            stripped = inner
        return stripped

    def _parse_pidnet_cf_add_assign(line: str) -> tuple[str, str, str, str, list[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, target_shape_expr = align_parts
        add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", input_expr.strip())
        rank4_shape = _parse_rank4_shape_literal(target_shape_expr)
        if add_match is None or rank4_shape is None:
            return None
        add_args = _parse_binary_add_args(str(add_match.group("args")))
        if add_args is None:
            return None
        return (
            indent,
            lhs,
            str(add_args[0]),
            str(add_args[1]),
            list(rank4_shape),
        )

    def _parse_pidnet_cf_resize_assign(
        line: str,
    ) -> tuple[str, str, str, tuple[int, int], list[int], str, bool, bool] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "_apply_resize("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        size_expr: str | None = None
        method_expr: str | None = None
        target_shape_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        channel_last_expr: str | None = None
        if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
            input_expr = parts[0].strip()
        if len(parts) >= 2 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[1]) is None:
            size_expr = parts[1].strip()
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "input":
                input_expr = value
            elif key == "size":
                size_expr = value
            elif key == "method":
                method_expr = value
            elif key == "target_shape":
                target_shape_expr = value
            elif key == "align_corners":
                align_expr = value
            elif key == "half_pixel_centers":
                hpc_expr = value
            elif key == "channel_last":
                channel_last_expr = value
        if (
            input_expr is None
            or size_expr is None
            or method_expr is None
            or target_shape_expr is None
            or align_expr not in {"True", "False"}
            or hpc_expr not in {"True", "False"}
            or channel_last_expr != "False"
        ):
            return None
        size_match = re.fullmatch(r"[\[\(]\s*(?P<h>\d+)\s*,\s*(?P<w>\d+)\s*[\]\)]", size_expr)
        shape_value = _parse_rank4_shape_literal(target_shape_expr)
        method_match = re.fullmatch(r"'(?P<method>[^']+)'", method_expr)
        if size_match is None or shape_value is None or method_match is None:
            return None
        return (
            indent,
            lhs,
            input_expr,
            (int(size_match.group("h")), int(size_match.group("w"))),
            list(shape_value),
            str(method_match.group("method")),
            align_expr == "True",
            hpc_expr == "True",
        )

    def _resolve_pidnet_nchw_to_nhwc_bridge_source(expr: str) -> str | None:
        stripped = str(expr).strip()
        if stripped.endswith(".contiguous()"):
            stripped = stripped[: -len(".contiguous()")].strip()

        def _parse_permute_like_args(args: str) -> str | None:
            parts = _split_top_level_csv_exprs(str(args))
            if len(parts) == 2 and "=" not in parts[0] and "=" not in parts[1]:
                source_expr = parts[0].strip()
                dims_expr = _normalize_permute_dims_expr(parts[1])
                if re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is not None and dims_expr == "0,2,3,1":
                    return source_expr
                return None
            kwargs: Dict[str, str] = {}
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            source_expr = kwargs.get("input", kwargs.get("x"))
            dims_expr = kwargs.get("dims", kwargs.get("perm"))
            if (
                source_expr is None
                or dims_expr is None
                or re.fullmatch(r"[A-Za-z0-9_]+", source_expr) is None
                or _normalize_permute_dims_expr(dims_expr) != "0,2,3,1"
            ):
                return None
            return source_expr

        functional_match = re.match(r"^torch\.permute\((?P<args>.+)\)$", stripped)
        if functional_match is not None:
            return _parse_permute_like_args(str(functional_match.group("args")))

        helper_match = re.match(r"^_torch_permute\((?P<args>.+)\)$", stripped)
        if helper_match is not None:
            return _parse_permute_like_args(str(helper_match.group("args")))

        method_match = re.match(r"^(?P<src>[A-Za-z0-9_]+)\.permute\((?P<dims>.+)\)$", stripped)
        if method_match is not None and _normalize_permute_dims_expr(str(method_match.group("dims"))) == "0,2,3,1":
            return str(method_match.group("src"))
        return None

    def _parse_pidnet_cf_alias_assign(line: str) -> tuple[str, str, str, list[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        src_expr, target_shape_expr = align_parts
        source_name = _resolve_pidnet_nchw_to_nhwc_bridge_source(src_expr)
        rank4_shape = _parse_rank4_shape_literal(target_shape_expr)
        if source_name is None or rank4_shape is None:
            return None
        return (
            indent,
            lhs,
            source_name,
            list(rank4_shape),
        )

    def _parse_pidnet_mul_align_assign(line: str) -> tuple[str, str, str, str, list[int]] | None:
        match = pidnet_mul_align_re.match(line)
        if match is None:
            return None
        mul_args = _parse_binary_mul_args(str(match.group("args")))
        if mul_args is None:
            return None
        return (
            str(match.group("indent")),
            str(match.group("lhs")),
            _strip_pidnet_outer_parentheses(str(mul_args[0])),
            _strip_pidnet_outer_parentheses(str(mul_args[1])),
            [
                int(match.group("n")),
                int(match.group("d1")),
                int(match.group("d2")),
                int(match.group("d3")),
            ],
        )

    def _parse_pidnet_bn_mul_assign(line: str) -> tuple[str, str, str, str, list[int]] | None:
        match = pidnet_bn_mul_re.match(line)
        if match is None:
            return None
        mul_args = _parse_binary_mul_args(str(match.group("args")))
        if mul_args is None:
            return None
        input_a = _strip_pidnet_outer_parentheses(str(mul_args[0]))
        input_b = _strip_pidnet_outer_parentheses(str(mul_args[1]))
        if _pidnet_is_const_like_expr(input_a) and not _pidnet_is_const_like_expr(input_b):
            input_a, input_b = input_b, input_a
        if _pidnet_is_const_like_expr(input_a) or not _pidnet_is_const_like_expr(input_b):
            return None
        return (
            str(match.group("indent")),
            str(match.group("lhs")),
            input_a,
            input_b,
            [
                int(match.group("n")),
                int(match.group("d1")),
                int(match.group("d2")),
                int(match.group("d3")),
            ],
        )

    def _parse_pidnet_bn_add_anchor_assign(
        line: str,
    ) -> tuple[str, str, str, str, str, list[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)(?P<lhs>\(*\s*[A-Za-z0-9_]+\s*,\s*[A-Za-z0-9_]+\s*\)?)\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        indent = str(assign_match.group("indent"))
        lhs = str(assign_match.group("lhs"))
        rhs = str(assign_match.group("rhs")).strip()
        unpack_match = re.fullmatch(
            r"\(*\s*(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*\)*",
            lhs,
        )
        if unpack_match is None:
            return None
        prefix = "_align_binary_inputs_to_anchor("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        if len(parts) == 3 and all("=" not in part for part in parts):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
            shape_expr = parts[2].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("a"))
            input_b = kwargs.get("other", kwargs.get("b"))
            shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
            if input_a is None or input_b is None or shape_expr is None:
                return None
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if rank4_shape is None:
            return None
        const_like_a = _pidnet_is_const_like_expr(str(input_a))
        const_like_b = _pidnet_is_const_like_expr(str(input_b))
        cf_like_a = _pidnet_is_cf_like_name(str(input_a))
        cf_like_b = _pidnet_is_cf_like_name(str(input_b))
        if const_like_a and not const_like_b:
            input_a, input_b = str(input_b), str(input_a)
            cf_like_a, cf_like_b = cf_like_b, cf_like_a
            const_like_a, const_like_b = const_like_b, const_like_a
        elif cf_like_b and not cf_like_a:
            input_a, input_b = str(input_b), str(input_a)
            cf_like_a, cf_like_b = cf_like_b, cf_like_a
            const_like_a, const_like_b = const_like_b, const_like_a
        if not (cf_like_a and const_like_b):
            return None
        return (
            indent,
            str(unpack_match.group("lhs0")),
            str(unpack_match.group("lhs1")),
            str(input_a),
            str(input_b),
            list(rank4_shape),
        )

    def _parse_pidnet_reduce_sum_assign(line: str) -> tuple[str, str, str, int] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "_reduce_sum("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        axes_expr: str | None = None
        keepdims_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key in {"axes", "axis", "dim"}:
                    axes_expr = value
                elif key == "keepdims":
                    keepdims_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                axes_expr = part.strip()
            elif positional_index == 2:
                keepdims_expr = part.strip()
            positional_index += 1
        if input_expr is None or axes_expr is None or keepdims_expr != "True":
            return None
        axis_match = re.fullmatch(
            r"_normalize_axes\([\[\(](?P<axis>\d+)(?:,\s*)?[\]\)],\s*"
            r"[A-Za-z0-9_]+\.ndim\)",
            axes_expr,
        )
        if axis_match is None:
            return None
        return indent, lhs, input_expr, int(axis_match.group("axis"))

    def _parse_pidnet_sigmoid_reshape_assign(
        line: str,
    ) -> tuple[str, str, str, list[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.reshape("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        shape_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "shape":
                    shape_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                shape_expr = part.strip()
            positional_index += 1
        if input_expr is None or shape_expr is None:
            return None
        input_match = re.fullmatch(r"torch\.sigmoid\((?P<input>[A-Za-z0-9_]+)\)", input_expr)
        rank4_shape = _parse_rank4_shape_literal(shape_expr)
        if input_match is None or rank4_shape is None:
            return None
        return indent, lhs, str(input_match.group("input")), list(rank4_shape)

    def _parse_pidnet_cf_mean_assign(line: str) -> tuple[str, str, str, tuple[int, int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "torch.mean("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                dim_expr = part.strip()
            elif positional_index == 2:
                keepdim_expr = part.strip()
            positional_index += 1
        if input_expr is None or dim_expr is None or keepdim_expr != "True":
            return None
        dim_match = re.fullmatch(r"[\[\(](?P<axis0>\d+), (?P<axis1>\d+)[\]\)]", dim_expr)
        if dim_match is None:
            return None
        return indent, lhs, input_expr, (int(dim_match.group("axis0")), int(dim_match.group("axis1")))

    def _parse_pidnet_cf_pool_assign(
        line: str,
    ) -> tuple[str, str, str, int, int, int, int, str, list[int], bool] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "_apply_pool2d("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        kwargs: Dict[str, str] = {}
        if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
            kwargs["input"] = parts[0].strip()
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            kwargs[key.strip()] = value.strip()
        input_expr = kwargs.get("input")
        padding_expr = kwargs.get("padding")
        channel_last_expr = kwargs.get("channel_last")
        is_max_expr = kwargs.get("is_max_pool")
        shape_expr = kwargs.get("target_shape")
        if (
            input_expr is None
            or padding_expr is None
            or channel_last_expr != "True"
            or is_max_expr not in {"True", "False"}
            or shape_expr is None
        ):
            return None
        shape_value = _parse_rank4_shape_literal(shape_expr)
        if shape_value is None:
            return None
        try:
            fh = int(kwargs["filter_height"])
            fw = int(kwargs["filter_width"])
            sh = int(kwargs["stride_h"])
            sw = int(kwargs["stride_w"])
        except (KeyError, ValueError):
            return None
        padding_match = re.fullmatch(r"'(?P<padding>[^']+)'", padding_expr)
        if padding_match is None:
            return None
        return (
            indent,
            lhs,
            input_expr,
            fh,
            fw,
            sh,
            sw,
            str(padding_match.group("padding")),
            list(shape_value),
            is_max_expr == "True",
        )

    def _parse_pidnet_permute_conv_assign(line: str) -> tuple[str, str, str, str] | None:
        match = pidnet_permute_conv_re.match(line)
        if match is None:
            return None
        source_name = _resolve_nhwc_to_nchw_bridge_source(str(match.group("src_expr")))
        if source_name is None:
            return None
        return (
            str(match.group("indent")),
            str(match.group("lhs")),
            str(match.group("module")),
            source_name,
        )

    def _parse_pidnet_cf_pad_assign(
        line: str,
    ) -> tuple[str, str, str, list[int], list[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "F.pad("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        pad_expr: str | None = None
        mode_expr: str | None = None
        value_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "pad":
                    pad_expr = value
                elif key == "mode":
                    mode_expr = value
                elif key == "value":
                    value_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                pad_expr = part.strip()
            elif positional_index == 2:
                mode_expr = part.strip()
            elif positional_index == 3:
                value_expr = part.strip()
            positional_index += 1
        if mode_expr != "'constant'" or value_expr != "0.0" or input_expr is None or pad_expr is None:
            return None
        align_parts = _parse_align_tensor_target_shape_expr(input_expr)
        pad_match = re.fullmatch(
            r"[\[\(]\s*0\s*,\s*0\s*,\s*(?P<pad_top>\d+)\s*,\s*(?P<pad_bottom>\d+)\s*,\s*(?P<pad_left>\d+)\s*,\s*(?P<pad_right>\d+)\s*[\]\)]",
            pad_expr,
        )
        if align_parts is None or pad_match is None:
            return None
        input_name, target_shape_expr = align_parts
        rank4_shape = _parse_rank4_shape_literal(target_shape_expr)
        if rank4_shape is None:
            return None
        return (
            indent,
            lhs,
            input_name,
            list(rank4_shape),
            [
                int(pad_match.group("pad_top")),
                int(pad_match.group("pad_bottom")),
                int(pad_match.group("pad_left")),
                int(pad_match.group("pad_right")),
            ],
        )

    def _parse_pidnet_plain_cf_pad_assign(
        line: str,
    ) -> tuple[str, str, str, list[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        prefix = "F.pad("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        pad_expr: str | None = None
        mode_expr: str | None = None
        value_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "pad":
                    pad_expr = value
                elif key == "mode":
                    mode_expr = value
                elif key == "value":
                    value_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                pad_expr = part.strip()
            elif positional_index == 2:
                mode_expr = part.strip()
            elif positional_index == 3:
                value_expr = part.strip()
            positional_index += 1
        if (
            input_expr is None
            or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
            or pad_expr is None
            or mode_expr != "'constant'"
            or value_expr != "0.0"
        ):
            return None
        pad_match = re.fullmatch(
            r"[\[\(]\s*(?P<p0>\d+)\s*,\s*(?P<p1>\d+)\s*,\s*(?P<p2>\d+)\s*,\s*(?P<p3>\d+)\s*[\]\)]",
            pad_expr,
        )
        if pad_match is None:
            return None
        pad_values = [int(pad_match.group(f"p{i}")) for i in range(4)]
        if len(set(pad_values)) != 1 or int(pad_values[0]) <= 0:
            return None
        return indent, lhs, input_expr, pad_values

    def _pidnet_is_cf_like_name(name: str) -> bool:
        if name.endswith("_cf") or name.endswith("_out_cf"):
            return True
        return name in (
            pidnet_cf_add_sources
            | pidnet_cf_alias_sources
            | pidnet_cf_binary_sources
            | pidnet_cf_mul_sources
            | pidnet_cf_reduce_sum_sources
            | pidnet_cf_mean_sources
            | pidnet_cf_pad_sources
            | pidnet_cf_pool_sources
        )

    def _pidnet_is_const_like_expr(expr: str) -> bool:
        return str(expr).startswith("self.") or str(expr) in pidnet_const_alias_sources

    def _pidnet_propagate_alias(lhs_name: str, input_name: str) -> None:
        if input_name in pidnet_tuple_alias_pairs:
            pidnet_tuple_alias_pairs[lhs_name] = pidnet_tuple_alias_pairs[input_name]
        if input_name in pidnet_cf_binary_tuple_sources:
            pidnet_cf_binary_tuple_sources.add(lhs_name)
        if input_name.startswith("self.") or input_name in pidnet_const_alias_sources:
            pidnet_const_alias_sources.add(lhs_name)
        if _pidnet_is_cf_like_name(input_name):
            pidnet_cf_alias_sources.add(lhs_name)
        if input_name in pidnet_cf_binary_sources:
            pidnet_cf_binary_sources.add(lhs_name)
        if input_name in pidnet_cf_mul_sources:
            pidnet_cf_mul_sources.add(lhs_name)
        if input_name in pidnet_cf_reduce_sum_sources:
            pidnet_cf_reduce_sum_sources.add(lhs_name)
        if input_name in pidnet_cf_mean_sources:
            pidnet_cf_mean_sources.add(lhs_name)
        if input_name in pidnet_cf_pad_sources:
            pidnet_cf_pad_sources.add(lhs_name)
        if input_name in pidnet_cf_pool_sources:
            pidnet_cf_pool_sources.add(lhs_name)

    def _pidnet_has_cf_layout_consumer(name: str, *, start_index: int) -> bool:
        tracked_names: Set[str] = {str(name)}
        for lookahead_index in range(start_index + 1, len(lines)):
            future_line = str(lines[lookahead_index])
            tuple_pair_alias_match = pidnet_tuple_pair_alias_re.match(future_line)
            if tuple_pair_alias_match is not None:
                input0_name = str(tuple_pair_alias_match.group("input0"))
                input1_name = str(tuple_pair_alias_match.group("input1"))
                if input0_name in tracked_names or input1_name in tracked_names:
                    tracked_names.add(str(tuple_pair_alias_match.group("lhs")))
                continue
            tuple_unpack_alias_match = pidnet_tuple_unpack_alias_re.match(future_line)
            if (
                tuple_unpack_alias_match is not None
                and str(tuple_unpack_alias_match.group("input")) in tracked_names
            ):
                tracked_names.add(str(tuple_unpack_alias_match.group("lhs0")))
                tracked_names.add(str(tuple_unpack_alias_match.group("lhs1")))
                continue
            tuple_alias_match = pidnet_tuple_alias_re.match(future_line)
            if tuple_alias_match is not None:
                tuple_pairs = (
                    (str(tuple_alias_match.group("lhs0")), str(tuple_alias_match.group("input0"))),
                    (str(tuple_alias_match.group("lhs1")), str(tuple_alias_match.group("input1"))),
                )
                for lhs_name, input_name in tuple_pairs:
                    if input_name in tracked_names:
                        tracked_names.add(lhs_name)
                continue
            plain_alias_match = pidnet_plain_alias_re.match(future_line)
            if plain_alias_match is not None and str(plain_alias_match.group("input")) in tracked_names:
                tracked_names.add(str(plain_alias_match.group("lhs")))
                continue
            parsed_pidnet_cf_alias = _parse_pidnet_cf_alias_assign(future_line)
            if parsed_pidnet_cf_alias is not None and str(parsed_pidnet_cf_alias[2]) in tracked_names:
                tracked_names.add(str(parsed_pidnet_cf_alias[1]))
                continue
            parsed_pidnet_permute_conv = _parse_pidnet_permute_conv_assign(future_line)
            if (
                parsed_pidnet_permute_conv is not None
                and str(parsed_pidnet_permute_conv[3]) in tracked_names
            ):
                return True
            parsed_pidnet_cf_mean = _parse_pidnet_cf_mean_assign(future_line)
            if (
                parsed_pidnet_cf_mean is not None
                and str(parsed_pidnet_cf_mean[2]) in tracked_names
                and parsed_pidnet_cf_mean[3] in {
                    (1, 2),
                    (2, 3),
                }
            ):
                return True
            parsed_pidnet_cf_pad = _parse_pidnet_cf_pad_assign(future_line)
            if (
                parsed_pidnet_cf_pad is not None
                and str(parsed_pidnet_cf_pad[2]) in tracked_names
            ):
                return True
            parsed_pidnet_cf_pool = _parse_pidnet_cf_pool_assign(future_line)
            if (
                parsed_pidnet_cf_pool is not None
                and str(parsed_pidnet_cf_pool[2]) in tracked_names
            ):
                return True
            binary_anchor_match = pidnet_binary_anchor_re.match(future_line)
            if binary_anchor_match is not None and (
                str(binary_anchor_match.group("a")) in tracked_names
                or str(binary_anchor_match.group("b")) in tracked_names
            ):
                return True
            binary_anchor_tuple_match = pidnet_binary_anchor_tuple_re.match(future_line)
            if binary_anchor_tuple_match is not None and (
                str(binary_anchor_tuple_match.group("a")) in tracked_names
                or str(binary_anchor_tuple_match.group("b")) in tracked_names
            ):
                return True
            scale3_anchor_match = pidnet_scale3_anchor_re.match(future_line)
            scale3_anchor_reversed_match = pidnet_scale3_anchor_reversed_re.match(future_line)
            if (
                (scale3_anchor_match is not None and str(scale3_anchor_match.group("input")) in tracked_names)
                or (
                    scale3_anchor_reversed_match is not None
                    and str(scale3_anchor_reversed_match.group("input")) in tracked_names
                )
            ):
                return True
            parsed_pidnet_bn_mul = _parse_pidnet_bn_mul_assign(future_line)
            if (
                parsed_pidnet_bn_mul is not None
                and str(parsed_pidnet_bn_mul[2]) in tracked_names
            ):
                return True
            parsed_pidnet_bn_add_anchor = _parse_pidnet_bn_add_anchor_assign(future_line)
            if (
                parsed_pidnet_bn_add_anchor is not None
                and str(parsed_pidnet_bn_add_anchor[3]) in tracked_names
            ):
                return True
            parsed_pidnet_mul_align = _parse_pidnet_mul_align_assign(future_line)
            if parsed_pidnet_mul_align is not None and (
                str(parsed_pidnet_mul_align[2]) in tracked_names
                or str(parsed_pidnet_mul_align[3]) in tracked_names
            ):
                return True
            parsed_pidnet_reduce_sum = _parse_pidnet_reduce_sum_assign(future_line)
            if (
                parsed_pidnet_reduce_sum is not None
                and str(parsed_pidnet_reduce_sum[2]) in tracked_names
            ):
                return True
        return False

    exact_line_rewrites = {
    }

    for index, line in enumerate(lines):
        pidnet_tuple_pair_alias_match = pidnet_tuple_pair_alias_re.match(line)
        if pidnet_tuple_pair_alias_match is not None:
            lhs_name = str(pidnet_tuple_pair_alias_match.group("lhs"))
            input0_name = str(pidnet_tuple_pair_alias_match.group("input0"))
            input1_name = str(pidnet_tuple_pair_alias_match.group("input1"))
            pidnet_tuple_alias_pairs[lhs_name] = (input0_name, input1_name)
            continue
        pidnet_tuple_unpack_alias_match = pidnet_tuple_unpack_alias_re.match(line)
        if pidnet_tuple_unpack_alias_match is not None:
            input_name = str(pidnet_tuple_unpack_alias_match.group("input"))
            if input_name in pidnet_tuple_alias_pairs:
                input0_name, input1_name = pidnet_tuple_alias_pairs[input_name]
                _pidnet_propagate_alias(str(pidnet_tuple_unpack_alias_match.group("lhs0")), input0_name)
                _pidnet_propagate_alias(str(pidnet_tuple_unpack_alias_match.group("lhs1")), input1_name)
                continue
            if input_name in pidnet_cf_binary_tuple_sources:
                pidnet_cf_binary_sources.update(
                    {
                        str(pidnet_tuple_unpack_alias_match.group("lhs0")),
                        str(pidnet_tuple_unpack_alias_match.group("lhs1")),
                    }
                )
                continue
        pidnet_tuple_alias_match = pidnet_tuple_alias_re.match(line)
        if pidnet_tuple_alias_match is not None:
            tuple_pairs = (
                (str(pidnet_tuple_alias_match.group("lhs0")), str(pidnet_tuple_alias_match.group("input0"))),
                (str(pidnet_tuple_alias_match.group("lhs1")), str(pidnet_tuple_alias_match.group("input1"))),
            )
            for lhs_name, input_name in tuple_pairs:
                _pidnet_propagate_alias(lhs_name, input_name)
            continue
        pidnet_plain_alias_match = pidnet_plain_alias_re.match(line)
        if pidnet_plain_alias_match is not None:
            input_name = str(pidnet_plain_alias_match.group("input"))
            lhs_name = str(pidnet_plain_alias_match.group("lhs"))
            _pidnet_propagate_alias(lhs_name, input_name)
            continue
        parsed_pidnet_cf_resize = _parse_pidnet_cf_resize_assign(line)
        if parsed_pidnet_cf_resize is not None:
            indent, lhs_name, input_name, out_hw, current_shape, method_name, align_corners, half_pixel_centers = parsed_pidnet_cf_resize
            if (
                _pidnet_is_cf_like_name(input_name)
                or _pidnet_is_cf_like_name(lhs_name)
                or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index)
            ):
                normalized_shape = _normalize_cf_rank4_shape(
                    current_shape,
                    preferred_channel_count=_pidnet_rank4_preferred_channel_count(
                        current_shape,
                        out_hw=out_hw,
                        prefer_last_if_nhwc=True,
                    ),
                    out_hw=out_hw,
                )
                if normalized_shape != current_shape:
                    lines[index] = (
                        f"{indent}{lhs_name} = _apply_resize("
                        f"{input_name}, [{out_hw[0]}, {out_hw[1]}], "
                        f"method='{method_name}', target_shape={normalized_shape}, "
                        f"align_corners={align_corners}, "
                        f"half_pixel_centers={half_pixel_centers}, channel_last=False)"
                    )
                    changed = True
                continue
        parsed_pidnet_cf_alias = _parse_pidnet_cf_alias_assign(line)
        if parsed_pidnet_cf_alias is not None:
            alias_indent, lhs_name, input_name, current_shape = parsed_pidnet_cf_alias
            normalized_shape = _normalize_cf_rank4_shape(
                current_shape,
                preferred_channel_count=_pidnet_rank4_preferred_channel_count(
                    current_shape,
                    prefer_last_if_nhwc=True,
                ),
            )
            if (
                (_pidnet_is_cf_like_name(input_name) or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index))
                and normalized_shape != current_shape
            ):
                lines[index] = (
                    f"{alias_indent}{lhs_name} = {input_name}"
                )
                changed = True
                pidnet_cf_alias_sources.add(lhs_name)
                continue
        parsed_pidnet_cf_add = _parse_pidnet_cf_add_assign(line)
        if parsed_pidnet_cf_add is not None:
            indent, lhs_name, input_a, input_b, current_shape = parsed_pidnet_cf_add
            if all(
                _pidnet_is_cf_like_name(input_name)
                for input_name in (input_a, input_b)
            ) or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index):
                preferred_channel_count = max(int(current_shape[1]), int(current_shape[2]), int(current_shape[3]))
                normalized_shape = _normalize_cf_rank4_shape(
                    current_shape,
                    preferred_channel_count=preferred_channel_count,
                )
                if normalized_shape != current_shape:
                    lines[index] = (
                        f"{indent}{lhs_name} = "
                        f"_align_tensor_to_target_shape(torch.add({input_a}, {input_b}), {normalized_shape})"
                    )
                    changed = True
                pidnet_cf_add_sources.add(lhs_name)
                continue
        parsed_pidnet_cf_pad = _parse_pidnet_cf_pad_assign(line)
        if parsed_pidnet_cf_pad is not None:
            pad_indent, lhs_name, input_name, current_shape, pad_values = parsed_pidnet_cf_pad
            normalized_shape = _normalize_cf_rank4_shape(
                current_shape,
                preferred_channel_count=_pidnet_rank4_preferred_channel_count(current_shape),
            )
            if (
                (
                    _pidnet_is_cf_like_name(input_name)
                    or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index)
                )
                and normalized_shape != current_shape
            ):
                lines[index] = (
                    f"{pad_indent}{lhs_name} = "
                    f"F.pad({input_name}, "
                    f"[{pad_values[0]}, {pad_values[1]}, {pad_values[2]}, {pad_values[3]}], "
                    f"mode='constant', value=0.0)"
                )
                changed = True
                pidnet_cf_pad_sources.add(lhs_name)
                continue
        parsed_pidnet_plain_cf_pad = _parse_pidnet_plain_cf_pad_assign(line)
        if parsed_pidnet_plain_cf_pad is not None:
            _, lhs_name, input_name, _ = parsed_pidnet_plain_cf_pad
            if (
                _pidnet_is_cf_like_name(input_name)
                or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index)
            ):
                pidnet_cf_pad_sources.add(lhs_name)
                continue
        pidnet_binary_anchor_match = pidnet_binary_anchor_re.match(line)
        if pidnet_binary_anchor_match is not None:
            input_a = str(pidnet_binary_anchor_match.group("a"))
            input_b = str(pidnet_binary_anchor_match.group("b"))
            lhs0_name = str(pidnet_binary_anchor_match.group("lhs0"))
            lhs1_name = str(pidnet_binary_anchor_match.group("lhs1"))
            if any(_pidnet_is_cf_like_name(input_name) for input_name in (input_a, input_b)):
                scale3_anchor_match = pidnet_scale3_anchor_re.match(line)
                scale3_anchor_reversed_match = pidnet_scale3_anchor_reversed_re.match(line)
                parsed_pidnet_bn_add_anchor = _parse_pidnet_bn_add_anchor_assign(line)
                if (
                    (
                        scale3_anchor_match is not None
                        and _pidnet_is_const_like_expr(str(scale3_anchor_match.group("const_expr")))
                    ) or (
                        scale3_anchor_reversed_match is not None
                        and _pidnet_is_const_like_expr(str(scale3_anchor_reversed_match.group("const_expr")))
                    )
                ) or parsed_pidnet_bn_add_anchor is not None:
                    pass
                else:
                    current_shape = [
                        int(pidnet_binary_anchor_match.group("n")),
                        int(pidnet_binary_anchor_match.group("d1")),
                        int(pidnet_binary_anchor_match.group("d2")),
                        int(pidnet_binary_anchor_match.group("d3")),
                    ]
                    preferred_channel_count = _pidnet_rank4_preferred_channel_count(
                        current_shape,
                        prefer_last_if_nhwc=any(
                            input_name in pidnet_cf_alias_sources for input_name in (input_a, input_b)
                        ),
                    )
                    normalized_shape = _normalize_cf_rank4_shape(
                        current_shape,
                        preferred_channel_count=preferred_channel_count,
                    )
                    if normalized_shape != current_shape:
                        lines[index] = (
                            f"{pidnet_binary_anchor_match.group('indent')}{pidnet_binary_anchor_match.group('lhs0')}, "
                            f"{pidnet_binary_anchor_match.group('lhs1')} = _align_binary_inputs_to_anchor("
                            f"{input_a}, {input_b}, {normalized_shape})"
                        )
                        changed = True
                    pidnet_cf_binary_sources.update({lhs0_name, lhs1_name})
                    continue
        pidnet_binary_anchor_tuple_match = pidnet_binary_anchor_tuple_re.match(line)
        if pidnet_binary_anchor_tuple_match is not None:
            input_a = str(pidnet_binary_anchor_tuple_match.group("a"))
            input_b = str(pidnet_binary_anchor_tuple_match.group("b"))
            lhs_name = str(pidnet_binary_anchor_tuple_match.group("lhs"))
            if any(_pidnet_is_cf_like_name(input_name) for input_name in (input_a, input_b)):
                current_shape = [
                    int(pidnet_binary_anchor_tuple_match.group("n")),
                    int(pidnet_binary_anchor_tuple_match.group("d1")),
                    int(pidnet_binary_anchor_tuple_match.group("d2")),
                    int(pidnet_binary_anchor_tuple_match.group("d3")),
                ]
                if (
                    _pidnet_is_cf_like_name(input_a) and _pidnet_is_const_like_expr(input_b)
                ) or (
                    _pidnet_is_cf_like_name(input_b) and _pidnet_is_const_like_expr(input_a)
                ):
                    cf_input_name = input_a if _pidnet_is_cf_like_name(input_a) else input_b
                    const_expr = input_b if cf_input_name == input_a else input_a
                    if int(current_shape[2]) == 1:
                        preferred_channel_count = _pidnet_rank4_preferred_channel_count(current_shape)
                        normalized_shape = [int(current_shape[0]), int(preferred_channel_count), 1, 1]
                    else:
                        normalized_shape = _normalize_cf_rank4_shape(
                            current_shape,
                            preferred_channel_count=_pidnet_rank4_preferred_channel_count(current_shape),
                        )
                    ann = (pidnet_binary_anchor_tuple_match.group("ann") or "").rstrip()
                    lines[index] = (
                        f"{pidnet_binary_anchor_tuple_match.group('indent')}{lhs_name}{ann} = _align_binary_inputs_to_anchor("
                        f"{cf_input_name}, torch.reshape({const_expr}, {normalized_shape}), {normalized_shape})"
                    )
                    changed = True
                else:
                    preferred_channel_count = _pidnet_rank4_preferred_channel_count(
                        current_shape,
                        prefer_last_if_nhwc=any(
                            input_name in pidnet_cf_alias_sources for input_name in (input_a, input_b)
                        ),
                    )
                    normalized_shape = _normalize_cf_rank4_shape(
                        current_shape,
                        preferred_channel_count=preferred_channel_count,
                    )
                    if normalized_shape != current_shape:
                        ann = (pidnet_binary_anchor_tuple_match.group("ann") or "").rstrip()
                        lines[index] = (
                            f"{pidnet_binary_anchor_tuple_match.group('indent')}{lhs_name}{ann} = _align_binary_inputs_to_anchor("
                            f"{input_a}, {input_b}, {normalized_shape})"
                        )
                        changed = True
                pidnet_cf_binary_tuple_sources.add(lhs_name)
                continue
        parsed_pidnet_cf_pool = _parse_pidnet_cf_pool_assign(line)
        if parsed_pidnet_cf_pool is not None:
            (
                pool_indent,
                lhs_name,
                input_name,
                fh,
                fw,
                sh,
                sw,
                padding,
                target_shape,
                is_max_pool,
            ) = parsed_pidnet_cf_pool
            if (
                not _has_immediate_rank4_permute_source(
                    lines,
                    index,
                    input_name,
                    [0, 2, 3, 1],
                )
                and (
                    _pidnet_is_cf_like_name(input_name)
                    or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index)
                )
            ):
                lines[index] = (
                    f"{pool_indent}{lhs_name} = _apply_pool2d("
                    f"{input_name}, filter_height={fh}, "
                    f"filter_width={fw}, stride_h={sh}, stride_w={sw}, padding='{padding}', "
                    f"target_shape={target_shape}, is_max_pool={is_max_pool}, channel_last=False)"
                )
                changed = True
                pidnet_cf_pool_sources.add(lhs_name)
                continue
        parsed_pidnet_mul_align = _parse_pidnet_mul_align_assign(line)
        if parsed_pidnet_mul_align is not None:
            indent, lhs_name, input_a, input_b, current_shape = parsed_pidnet_mul_align
            if (
                _pidnet_is_cf_like_name(input_a)
                or _pidnet_is_cf_like_name(input_b)
            ):
                parsed_pidnet_bn_mul = _parse_pidnet_bn_mul_assign(line)
                if parsed_pidnet_bn_mul is not None:
                    pass
                else:
                    normalized_shape = _normalize_cf_rank4_shape(current_shape)
                    if normalized_shape == current_shape:
                        normalized_shape = _normalize_cf_rank4_shape(
                            current_shape,
                            preferred_channel_count=_pidnet_rank4_preferred_channel_count(
                                current_shape,
                                prefer_last_if_nhwc=True,
                            ),
                        )
                    if normalized_shape != current_shape:
                        lines[index] = (
                            f"{indent}{lhs_name} = "
                            f"_align_tensor_to_target_shape(torch.mul({input_a}, {input_b}), {normalized_shape})"
                        )
                        changed = True
                    pidnet_cf_mul_sources.add(lhs_name)
                    continue
        parsed_pidnet_reduce_sum = _parse_pidnet_reduce_sum_assign(line)
        if (
            parsed_pidnet_reduce_sum is not None
            and str(parsed_pidnet_reduce_sum[2]) in pidnet_cf_mul_sources
            and int(parsed_pidnet_reduce_sum[3]) == 3
        ):
            reduce_indent, reduce_lhs, reduce_input, _ = parsed_pidnet_reduce_sum
            lines[index] = (
                f"{reduce_indent}{reduce_lhs} = "
                f"_reduce_sum({reduce_input}, "
                f"_normalize_axes([1], {reduce_input}.ndim), True)"
            )
            changed = True
            pidnet_cf_reduce_sum_sources.add(str(reduce_lhs))
            continue
        parsed_pidnet_sigmoid_reshape = _parse_pidnet_sigmoid_reshape_assign(line)
        if (
            parsed_pidnet_sigmoid_reshape is not None
            and str(parsed_pidnet_sigmoid_reshape[2]) in pidnet_cf_reduce_sum_sources
        ):
            reshape_indent, reshape_lhs, reshape_input, reshape_shape = parsed_pidnet_sigmoid_reshape
            lines[index] = (
                f"{reshape_indent}{reshape_lhs} = "
                f"_align_tensor_to_target_shape(torch.sigmoid({reshape_input}), "
                f"{reshape_shape})"
            )
            changed = True
            continue
        pidnet_scale3_anchor_match = pidnet_scale3_anchor_re.match(line)
        pidnet_scale3_anchor_reversed_match = pidnet_scale3_anchor_reversed_re.match(line)
        if (
            pidnet_scale3_anchor_reversed_match is not None
            and _pidnet_is_const_like_expr(str(pidnet_scale3_anchor_reversed_match.group("const_expr")))
        ):
            pidnet_scale3_anchor_match = pidnet_scale3_anchor_reversed_match
        if pidnet_scale3_anchor_match is not None:
            input_name = str(pidnet_scale3_anchor_match.group("input"))
            const_expr = str(pidnet_scale3_anchor_match.group("const_expr"))
            current_shape = [
                int(pidnet_scale3_anchor_match.group("n")),
                int(pidnet_scale3_anchor_match.group("d1")),
                int(pidnet_scale3_anchor_match.group("d2")),
                int(pidnet_scale3_anchor_match.group("d3")),
            ]
            if int(current_shape[2]) == 1:
                preferred_channel_count = _pidnet_rank4_preferred_channel_count(current_shape)
                normalized_shape = [int(current_shape[0]), int(preferred_channel_count), 1, 1]
            else:
                normalized_shape = _normalize_cf_rank4_shape(
                    current_shape,
                    preferred_channel_count=_pidnet_rank4_preferred_channel_count(current_shape),
                )
            if _pidnet_is_cf_like_name(input_name) and _pidnet_is_const_like_expr(const_expr):
                lines[index] = (
                    f"{pidnet_scale3_anchor_match.group('indent')}{pidnet_scale3_anchor_match.group('lhs0')}, "
                    f"{pidnet_scale3_anchor_match.group('lhs1')} = _align_binary_inputs_to_anchor("
                    f"{input_name}, torch.reshape({const_expr}, {normalized_shape}), "
                    f"{normalized_shape})"
                )
                changed = True
                continue
        parsed_pidnet_cf_mean = _parse_pidnet_cf_mean_assign(line)
        if parsed_pidnet_cf_mean is not None:
            mean_indent, lhs_name, input_name, mean_axes = parsed_pidnet_cf_mean
            if (
                _pidnet_is_cf_like_name(input_name)
                or _pidnet_has_cf_layout_consumer(lhs_name, start_index=index)
            ) and mean_axes == (1, 2):
                lines[index] = (
                    f"{mean_indent}{lhs_name} = "
                    f"torch.mean({input_name}, dim=[2, 3], keepdim=True)"
                )
                changed = True
                pidnet_cf_mean_sources.add(lhs_name)
                continue
        parsed_pidnet_bn_mul = _parse_pidnet_bn_mul_assign(line)
        if parsed_pidnet_bn_mul is not None:
            indent, lhs_name, input_name, const_expr, current_shape = parsed_pidnet_bn_mul
            normalized_shape = _normalize_cf_rank4_shape(
                current_shape,
                preferred_channel_count=_pidnet_rank4_preferred_channel_count(current_shape),
            )
            if (
                input_name in pidnet_cf_mean_sources or _pidnet_is_cf_like_name(input_name)
            ) and _pidnet_is_const_like_expr(const_expr):
                lines[index] = (
                    f"{indent}{lhs_name} = "
                    f"_align_tensor_to_target_shape(torch.mul({input_name}, "
                    f"torch.reshape({const_expr}, {normalized_shape})), "
                    f"{normalized_shape})"
                )
                changed = True
                pidnet_cf_mul_sources.add(lhs_name)
                continue
        parsed_pidnet_bn_add_anchor = _parse_pidnet_bn_add_anchor_assign(line)
        if parsed_pidnet_bn_add_anchor is not None:
            (
                anchor_indent,
                lhs0_name,
                lhs1_name,
                input_name,
                const_expr,
                current_shape,
            ) = parsed_pidnet_bn_add_anchor
            normalized_shape = _normalize_cf_rank4_shape(
                current_shape,
                preferred_channel_count=_pidnet_rank4_preferred_channel_count(
                    current_shape,
                    prefer_last_if_nhwc=any(
                        name in pidnet_cf_alias_sources for name in (input_name, const_expr)
                    ),
                ),
            )
            if _pidnet_is_cf_like_name(input_name) and (
                _pidnet_is_const_like_expr(const_expr)
                or not _pidnet_is_cf_like_name(const_expr)
            ):
                lines[index] = (
                    f"{anchor_indent}{lhs0_name}, "
                    f"{lhs1_name} = _align_binary_inputs_to_anchor("
                    f"{input_name}, torch.reshape({const_expr}, {normalized_shape}), "
                    f"{normalized_shape})"
                )
                changed = True
                continue
        replacement = exact_line_rewrites.get(line, None)
        if replacement is not None:
            lines[index] = replacement
            changed = True
            continue
        parsed_pidnet_permute_conv = _parse_pidnet_permute_conv_assign(line)
        if (
            parsed_pidnet_permute_conv is not None
            and _pidnet_is_cf_like_name(str(parsed_pidnet_permute_conv[3]))
        ):
            permute_indent, permute_lhs, permute_module, permute_input = parsed_pidnet_permute_conv
            lines[index] = (
                f"{permute_indent}{permute_lhs} = "
                f"self.{permute_module}({permute_input})"
            )
            changed = True

    finalized_lines = _fold_channel_first_hardsigmoid_gate_conv_bridges(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True
    finalized_lines = _rewrite_channel_first_se_scale_binary_bridges(lines)
    if finalized_lines != lines:
        lines = finalized_lines
        changed = True

    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_humanseg_fast_precanonicalize_repairs(model_path: Path) -> None:
    if not model_path.exists():
        return
    model_source = model_path.read_text(encoding="utf-8")
    lines = model_source.splitlines()
    if not _has_humanseg_fast_repair_signature(lines):
        return
    changed = False
    conv71_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_\d+)\((?P<src_expr>.+)\)$"
    )

    def _parse_resize_assign(
        line: str,
    ) -> Tuple[str, str, str, Tuple[int, int], List[int], str, bool, bool, bool] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "_apply_resize("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        size_expr: str | None = None
        method_expr: str | None = None
        target_shape_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        channel_last_expr: str | None = None
        if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
            input_expr = parts[0].strip()
        if len(parts) >= 2 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[1]) is None:
            size_expr = parts[1].strip()
        if len(parts) >= 3 and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[2]) is None:
            method_expr = parts[2].strip()
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "input":
                input_expr = value
            elif key == "size":
                size_expr = value
            elif key == "method":
                method_expr = value
            elif key == "target_shape":
                target_shape_expr = value
            elif key == "align_corners":
                align_expr = value
            elif key == "half_pixel_centers":
                hpc_expr = value
            elif key == "channel_last":
                channel_last_expr = value
        if (
            input_expr is None
            or size_expr is None
            or method_expr is None
            or target_shape_expr is None
            or align_expr not in {"True", "False"}
            or hpc_expr not in {"True", "False"}
            or channel_last_expr not in {"True", "False"}
        ):
            return None
        size_match = re.fullmatch(r"[\[\(]\s*(?P<h>\d+)\s*,\s*(?P<w>\d+)\s*[\]\)]", size_expr)
        raw_shape = _parse_rank4_shape_literal(target_shape_expr)
        method_match = re.fullmatch(r"'(?P<method>[^']+)'", method_expr)
        if size_match is None or raw_shape is None or method_match is None:
            return None
        return (
            indent,
            lhs,
            input_expr,
            (int(size_match.group("h")), int(size_match.group("w"))),
            list(raw_shape),
            str(method_match.group("method")),
            align_expr == "True",
            hpc_expr == "True",
            channel_last_expr == "True",
        )

    def _parse_align_add_assign(line: str) -> Tuple[str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        align_args = _parse_align_tensor_target_shape_expr(rhs)
        if align_args is None:
            return None
        aligned_expr, shape_expr = align_args
        add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", aligned_expr)
        shape = _parse_rank4_shape_literal(shape_expr)
        if add_match is None or shape is None:
            return None
        add_args = _parse_binary_add_args(str(add_match.group("args")))
        if add_args is None:
            return None
        return indent, lhs, add_args[0], add_args[1], list(shape)

    def _parse_binary_anchor_assign(
        line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+)\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        args = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        if len(args) != 3:
            return None
        shape = _parse_rank4_shape_literal(args[2].strip())
        if shape is None:
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            args[0].strip(),
            args[1].strip(),
            list(shape),
        )

    def _normalized_resize_shape_from_parsed(
        parsed_resize: Tuple[str, str, str, Tuple[int, int], List[int], str, bool, bool, bool],
    ) -> List[int]:
        _, _, _, size_value, raw_shape, _, _, _, channel_last = parsed_resize
        preferred_channel_count: int | None = None
        if len(raw_shape) == 4:
            if channel_last:
                preferred_channel_count = int(raw_shape[3])
            else:
                preferred_channel_count = int(raw_shape[1])
        return _normalize_cf_rank4_shape(
            raw_shape,
            preferred_channel_count=preferred_channel_count,
            out_hw=size_value,
        )

    index = 0
    while index + 4 < len(lines):
        line0 = lines[index]
        line1 = lines[index + 1]
        line2 = lines[index + 2]
        line3 = lines[index + 3]
        line4 = lines[index + 4]
        resize0 = _parse_resize_assign(line0)
        resize1 = _parse_resize_assign(line1)
        resize2 = _parse_resize_assign(line2)
        conv_match = conv71_assign_re.match(line4)
        concat_assign = _parse_simple_assignment_line(line3)
        parsed_concat_args = (
            _parse_apply_concat_inputs_axis_and_shape(concat_assign[2])
            if concat_assign is not None and "_apply_concat(" in concat_assign[2]
            else None
        )
        if (
            resize0 is None
            or resize1 is None
            or resize2 is None
            or parsed_concat_args is None
            or conv_match is None
        ):
            index += 1
            continue
        concat_inputs = (
            [input_name.strip() for input_name in parsed_concat_args[0] if input_name.strip()]
        )
        conv_src_expr = str(conv_match.group("src_expr")).strip()
        conv_bridge_src = _resolve_nhwc_to_nchw_bridge_source(conv_src_expr)
        conv_direct_src = conv_src_expr if re.fullmatch(r"[A-Za-z0-9_]+", conv_src_expr) is not None else None
        resolved_conv_src = conv_bridge_src if conv_bridge_src is not None else conv_direct_src
        if (
            len(concat_inputs) != 4
            or concat_inputs[1] != resize0[1]
            or concat_inputs[2] != resize1[1]
            or concat_inputs[3] != resize2[1]
            or resolved_conv_src != str(concat_assign[1])
        ):
            index += 1
            continue
        indent0 = line0[: len(line0) - len(line0.lstrip())]
        indent1 = line1[: len(line1) - len(line1.lstrip())]
        indent2 = line2[: len(line2) - len(line2.lstrip())]
        indent4 = line4[: len(line4) - len(line4.lstrip())]
        resize0_shape = _normalized_resize_shape_from_parsed(resize0)
        resize1_shape = _normalized_resize_shape_from_parsed(resize1)
        resize2_shape = _normalized_resize_shape_from_parsed(resize2)
        lines[index] = (
            f"{indent0}{resize0[1]} = _apply_resize("
            f"{resize0[2]}, [{resize0[3][0]}, {resize0[3][1]}], "
            f"method='{resize0[5]}', target_shape={repr(resize0_shape)}, "
            f"align_corners={resize0[6]}, half_pixel_centers={resize0[7]}, channel_last=False)"
        )
        lines[index + 1] = (
            f"{indent1}{resize1[1]} = _apply_resize("
            f"{resize1[2]}, [{resize1[3][0]}, {resize1[3][1]}], "
            f"method='{resize1[5]}', target_shape={repr(resize1_shape)}, "
            f"align_corners={resize1[6]}, half_pixel_centers={resize1[7]}, channel_last=False)"
        )
        lines[index + 2] = (
            f"{indent2}{resize2[1]} = _apply_resize("
            f"{resize2[2]}, [{resize2[3][0]}, {resize2[3][1]}], "
            f"method='{resize2[5]}', target_shape={repr(resize2_shape)}, "
            f"align_corners={resize2[6]}, half_pixel_centers={resize2[7]}, channel_last=False)"
        )
        lines[index + 3] = (
            f"{concat_assign[0]}{concat_assign[1]} = "
            f"torch.cat([{', '.join(concat_inputs)}], dim=1)"
        )
        lines[index + 4] = (
            f"{indent4}{conv_match.group('lhs')} = self.{conv_match.group('module')}("
            f"{concat_assign[1]})"
        )
        changed = True
        index += 5

    align_add_matches_by_lhs: Dict[str, Tuple[int, Tuple[str, str, str, str, List[int]]]] = {}
    for line_index, line in enumerate(lines):
        align_add_match = _parse_align_add_assign(line)
        if align_add_match is not None:
            align_add_matches_by_lhs[str(align_add_match[1])] = (line_index, align_add_match)

    for index, line in enumerate(lines):
        resize_match = _parse_resize_assign(line)
        if resize_match is not None:
            align_add_entry = align_add_matches_by_lhs.get(str(resize_match[2]))
            if align_add_entry is not None:
                align_add_index, align_add_match = align_add_entry
                input_shape = list(align_add_match[4])
                output_shape = list(resize_match[4])
                preferred_channel_count = None
                if len(output_shape) == 4:
                    if resize_match[8]:
                        preferred_channel_count = int(output_shape[3])
                    else:
                        preferred_channel_count = int(output_shape[1])
                normalized_output_shape = _normalize_cf_rank4_shape(
                    output_shape,
                    preferred_channel_count=preferred_channel_count,
                    out_hw=resize_match[3],
                )
                normalized_input_shape = _normalize_cf_rank4_shape(
                    input_shape,
                    preferred_channel_count=normalized_output_shape[1] if len(normalized_output_shape) == 4 else preferred_channel_count,
                )
                rewritten_resize_line = (
                    f"{resize_match[0]}{resize_match[1]} = _apply_resize("
                    f"{resize_match[2]}, [{resize_match[3][0]}, {resize_match[3][1]}], "
                    f"method='{resize_match[5]}', target_shape={repr(normalized_output_shape)}, "
                    f"align_corners={resize_match[6]}, half_pixel_centers={resize_match[7]}, channel_last=False)"
                )
                if rewritten_resize_line != lines[index]:
                    lines[index] = rewritten_resize_line
                    changed = True
                rewritten_align_add_line = (
                    f"{align_add_match[0]}{align_add_match[1]} = _align_tensor_to_target_shape("
                    f"torch.add({align_add_match[2]}, {align_add_match[3]}), {repr(normalized_input_shape)})"
                )
                if rewritten_align_add_line != lines[align_add_index]:
                    lines[align_add_index] = rewritten_align_add_line
                    changed = True
                add_operands = {str(align_add_match[2]), str(align_add_match[3])}
                for back in range(max(0, align_add_index - 4), align_add_index):
                    binary_anchor_match = _parse_binary_anchor_assign(lines[back])
                    if binary_anchor_match is None:
                        continue
                    binary_outputs = {str(binary_anchor_match[1]), str(binary_anchor_match[2])}
                    if binary_outputs != add_operands:
                        continue
                    rewritten_binary_line = (
                        f"{binary_anchor_match[0]}{binary_anchor_match[1]}, {binary_anchor_match[2]} = "
                        f"_align_binary_inputs_to_anchor({binary_anchor_match[3]}, {binary_anchor_match[4]}, {repr(normalized_input_shape)})"
                    )
                    if rewritten_binary_line != lines[back]:
                        lines[back] = rewritten_binary_line
                        changed = True
                    break
        concat_assign = _parse_simple_assignment_line(line)
        parsed_concat_args = (
            _parse_apply_concat_inputs_axis_and_shape(concat_assign[2])
            if concat_assign is not None and "_apply_concat(" in concat_assign[2]
            else None
        )
        if parsed_concat_args is not None and index + 1 < len(lines):
            conv71_permute_match = conv71_assign_re.match(lines[index + 1])
            conv71_src_expr = (
                str(conv71_permute_match.group("src_expr")).strip()
                if conv71_permute_match is not None
                else None
            )
            conv71_bridge_src = (
                _resolve_nhwc_to_nchw_bridge_source(conv71_src_expr)
                if conv71_src_expr is not None
                else None
            )
            if (
                conv71_permute_match is not None
                and conv71_bridge_src == str(concat_assign[1])
            ):
                concat_inputs = (
                    [input_name.strip() for input_name in parsed_concat_args[0] if input_name.strip()]
                )
                rewritten_concat_line = (
                    f"{concat_assign[0]}{concat_assign[1]} = "
                    f"torch.cat([{', '.join(concat_inputs)}], dim=1)"
                )
                rewritten_conv_line = (
                    f"{conv71_permute_match.group('indent')}{conv71_permute_match.group('lhs')} = "
                    f"self.{conv71_permute_match.group('module')}({concat_assign[1]})"
                )
                if rewritten_concat_line != lines[index]:
                    lines[index] = rewritten_concat_line
                    changed = True
                if rewritten_conv_line != lines[index + 1]:
                    lines[index + 1] = rewritten_conv_line
                    changed = True
    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _has_dynamic_score_sampling_stage_signature(lines: Sequence[str]) -> bool:
    stage7_reshape_shape_pattern = r"(?:\[\s*-1\s*,\s*1\s*\]|\(\s*-1\s*,\s*1\s*\))"
    stage7_def_re = re.compile(
        r"^\s*def (?P<helper_name>[A-Za-z0-9_]+)\(self, .+\)(?: -> (?:tuple|Tuple|typing\.Tuple)\[torch\.Tensor,\s*torch\.Tensor\])?:$"
    )
    forward_unpack_re = re.compile(
        r"^\s*\(?(?P<descriptors>[A-Za-z0-9_]+), (?P<score>[A-Za-z0-9_]+)\)? = self\.(?P<helper_name>[A-Za-z0-9_]+)\(.+\)$"
    )
    forward_packed_re = re.compile(
        r"^\s*(?P<packed>[A-Za-z0-9_]+)(?:\s*:\s*(?:tuple|Tuple|typing\.Tuple)\[torch\.Tensor,\s*torch\.Tensor\])?\s*=\s*self\.(?P<helper_name>[A-Za-z0-9_]+)\(.+\)$"
    )
    forward_index_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\s*=\s*(?P<packed>[A-Za-z0-9_]+)\[(?P<index>[01])\]$"
    )
    gather_expr_pattern = r"(?:[A-Za-z0-9_]+|torch\.(?:gather|index_select|take_along_dim)\(.+\)|[A-Za-z0-9_]+\.(?:gather|index_select|take_along_dim)\(.+\)|_reshape_gather_output\(.+\)|_align_tensor_to_target_shape\(.+\))"
    gather_reshape_res = [
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = torch\.reshape\({gather_expr_pattern}, _resolve_reshape_shape\({stage7_reshape_shape_pattern}, {gather_expr_pattern}, allow_zero=False\)\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = torch\.reshape\(\s*input\s*=\s*{gather_expr_pattern}\s*,\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {gather_expr_pattern}, allow_zero=False\)\s*\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = torch\.reshape\(\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {gather_expr_pattern}, allow_zero=False\)\s*,\s*input\s*=\s*{gather_expr_pattern}\s*\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = {gather_expr_pattern}\.reshape\(_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {gather_expr_pattern}, allow_zero=False\)\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = torch\.reshape\({gather_expr_pattern}, {stage7_reshape_shape_pattern}\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = torch\.reshape\(\s*input\s*=\s*{gather_expr_pattern}\s*,\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = torch\.reshape\(\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*,\s*input\s*=\s*{gather_expr_pattern}\s*\)$"
        ),
        re.compile(
            rf"^\s*[A-Za-z0-9_]+ = {gather_expr_pattern}\.reshape\({stage7_reshape_shape_pattern}\)$"
        ),
    ]
    singleton_anchor_assign_re = re.compile(
        r"^\s*(?P<lhs0>[A-Za-z0-9_]+), (?P<lhs1>[A-Za-z0-9_]+) = _align_binary_inputs_to_anchor\((?P<args>.+)\)$"
    )
    anchor_pair_assign_re = re.compile(
        r"^\s*(?P<pair>[A-Za-z0-9_]+)(?:\s*:\s*(?:tuple|Tuple|typing\.Tuple)\[[^\]]+\])?\s*=\s*_align_binary_inputs_to_anchor\((?P<args>.+)\)$"
    )
    stage7_return_re = re.compile(
        r"^\s*return\s+\(?(?P<descriptors>[A-Za-z0-9_]+), (?P<score>[A-Za-z0-9_]+)\)?$"
    )
    stage7_permute_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = _torch_permute\((?P<args>.+)\)$"
    )
    stage7_shape_tensor_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = _shape_tensor\((?P<args>.+)\)$"
    )
    stage7_gather_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.gather\((?P<args>.+)\)$"
    )
    stage7_method_source_expr_pattern = r"(?:[A-Za-z0-9_]+|_align_tensor_to_target_shape\(.+\)|_reshape_gather_output\(.+\))"
    stage7_gather_method_assign_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>{stage7_method_source_expr_pattern})\.gather\((?P<args>.+)\)$"
    )
    stage7_index_select_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.index_select\((?P<args>.+)\)$"
    )
    stage7_index_select_method_assign_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>{stage7_method_source_expr_pattern})\.index_select\((?P<args>.+)\)$"
    )
    stage7_take_along_dim_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.take_along_dim\((?P<args>.+)\)$"
    )
    stage7_take_along_dim_method_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\.take_along_dim\((?P<args>.+)\)$"
    )
    stage7_gather_expr_pattern = r"(?:[A-Za-z0-9_]+|torch\.(?:gather|index_select|take_along_dim)\(.+\)|[A-Za-z0-9_]+\.(?:gather|index_select|take_along_dim)\(.+\)|_reshape_gather_output\(.+\)|_align_tensor_to_target_shape\(.+\))"
    stage7_gather_reshape_res = [
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\((?P<input>{stage7_gather_expr_pattern}), _resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*,\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\s*,\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = (?P<input>{stage7_gather_expr_pattern})\.reshape\(_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\((?P<input>{stage7_gather_expr_pattern}), {stage7_reshape_shape_pattern}\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*,\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*,\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = (?P<input>{stage7_gather_expr_pattern})\.reshape\({stage7_reshape_shape_pattern}\)$"
        ),
    ]
    stage7_score_squeeze_res = [
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.squeeze\((?P<input>.+?)\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.squeeze\(\s*input\s*=\s*(?P<input>.+?)\s*\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = (?P<input>.+?)\.squeeze\(\)$"
        ),
    ]
    stage7_score_mul_res = [
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.mul\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+)\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.mul\(\s*input\s*=\s*(?P<input0>[A-Za-z0-9_]+)\s*,\s*other\s*=\s*(?P<input1>[A-Za-z0-9_]+)\s*\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.mul\(\s*other\s*=\s*(?P<input1>[A-Za-z0-9_]+)\s*,\s*input\s*=\s*(?P<input0>[A-Za-z0-9_]+)\s*\)$"
        ),
    ]
    stage7_permute_dims_patterns = {
        (1, 0): r"(?:\[\s*1\s*,\s*0\s*\]|\(\s*1\s*,\s*0\s*\))",
    }
    def _split_stage7_top_level_args(args: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in args:
            if char == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(char)
            if char in "([{":
                depth += 1
            elif char in ")]}" and depth > 0:
                depth -= 1
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    stage7_anchor_shape_alias_sources: Dict[str, str] = {}

    def _strip_stage7_outer_parentheses(expr: str) -> str:
        stripped_expr = str(expr).strip()
        while stripped_expr.startswith("(") and stripped_expr.endswith(")"):
            depth = 0
            balanced = True
            for index, char in enumerate(stripped_expr):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(stripped_expr) - 1:
                        balanced = False
                        break
                if depth < 0:
                    balanced = False
                    break
            if not balanced or depth != 0:
                break
            inner_expr = stripped_expr[1:-1].strip()
            if not (
                re.fullmatch(r"[A-Za-z0-9_]+", inner_expr) is not None
                or (inner_expr.startswith("[") and inner_expr.endswith("]"))
                or (inner_expr.startswith("(") and inner_expr.endswith(")"))
            ):
                break
            stripped_expr = inner_expr
        return stripped_expr

    def _resolve_stage7_anchor_target_shape(target_expr: str) -> str:
        resolved_target = _strip_stage7_outer_parentheses(str(target_expr).strip())
        seen_targets: Set[str] = set()
        while (
            resolved_target in stage7_anchor_shape_alias_sources
            and resolved_target not in seen_targets
        ):
            seen_targets.add(resolved_target)
            resolved_target = _strip_stage7_outer_parentheses(
                stage7_anchor_shape_alias_sources[resolved_target]
            )
        return resolved_target

    def _parse_stage7_anchor_shape_alias(line: str) -> tuple[str, str] | None:
        shape_alias_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?:\s*:\s*[^=]+)?\s*=\s*(?P<rhs>.+?)\s*$",
            line,
        )
        if shape_alias_match is None:
            return None
        rhs_value = _strip_stage7_outer_parentheses(str(shape_alias_match.group("rhs")))
        if (
            re.fullmatch(r"(?:\[|\()\s*1\s*,\s*1\s*,\s*1\s*,\s*1\s*(?:\]|\))", rhs_value)
            is None
            and rhs_value not in stage7_anchor_shape_alias_sources
        ):
            return None
        return str(shape_alias_match.group("lhs")), rhs_value

    def _parse_stage7_singleton_anchor_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 3:
            rs_value = parts[0].strip()
            tr_value = parts[1].strip()
            target_value = _resolve_stage7_anchor_target_shape(parts[2].strip())
            normalized_tr_value = _strip_stage7_outer_parentheses(tr_value)
            if (
                re.fullmatch(r"(?:\[|\()\s*1\s*,\s*1\s*,\s*1\s*,\s*1\s*(?:\]|\))", target_value)
                is not None
                and (
                    re.fullmatch(r"[A-Za-z0-9_]+", normalized_tr_value) is not None
                    or (
                        normalized_tr_value.startswith("_align_tensor_to_target_shape(")
                        and normalized_tr_value.endswith(")")
                    )
                )
            ):
                return rs_value, tr_value
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        rs_value = keyword_values.get("input0")
        tr_value = keyword_values.get("input1")
        target_value = (
            _resolve_stage7_anchor_target_shape(keyword_values["target_shape"])
            if "target_shape" in keyword_values
            else None
        )
        if (
            rs_value is None
            or tr_value is None
            or target_value is None
            or (
                re.fullmatch(r"[A-Za-z0-9_]+", _strip_stage7_outer_parentheses(tr_value)) is None
                and not (
                    _strip_stage7_outer_parentheses(tr_value).startswith("_align_tensor_to_target_shape(")
                    and _strip_stage7_outer_parentheses(tr_value).endswith(")")
                )
            )
            or re.fullmatch(r"(?:\[|\()\s*1\s*,\s*1\s*,\s*1\s*,\s*1\s*(?:\]|\))", target_value) is None
        ):
            return None
        return rs_value, tr_value

    def _has_stage7_inline_branch_reshape(expr: str) -> bool:
        branch_expr_pattern = r"(?:[A-Za-z0-9_]+|torch\.(?:gather|index_select|take_along_dim)\(.+\)|[A-Za-z0-9_]+\.(?:gather|index_select|take_along_dim)\(.+\)|_reshape_gather_output\(.+\)|_align_tensor_to_target_shape\(.+\))"
        inline_patterns = [
            re.compile(
                rf"^torch\.reshape\((?P<input>{branch_expr_pattern}), _resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*,\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\s*\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\s*,\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*\)$"
            ),
            re.compile(
                rf"^{branch_expr_pattern}\.reshape\(_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\)$"
            ),
            re.compile(
                rf"^torch\.reshape\({branch_expr_pattern}, {stage7_reshape_shape_pattern}\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*input\s*=\s*{branch_expr_pattern}\s*,\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*,\s*input\s*=\s*{branch_expr_pattern}\s*\)$"
            ),
            re.compile(
                rf"^{branch_expr_pattern}\.reshape\({stage7_reshape_shape_pattern}\)$"
            ),
        ]
        expr = str(expr).strip()
        return any(pattern.match(expr) is not None for pattern in inline_patterns)

    def _parse_stage7_mul_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            return parts[0].strip(), parts[1].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input0 = keyword_values.get("input")
        input1 = keyword_values.get("other")
        if input0 is None or input1 is None:
            return None
        return input0, input1

    def _parse_stage7_method_mul_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 1 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            return input_name, parts[0].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        other_name = keyword_values.get("other")
        if other_name is None:
            return None
        return input_name, other_name

    def _parse_stage7_method_gather_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            dim_value = parts[0].strip()
            indices_name = parts[1].strip()
            if dim_value == "0" and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None:
                return input_name, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        indices_name = keyword_values.get("index")
        dim_value = keyword_values.get("dim")
        if (
            indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_name, indices_name

    def _parse_stage7_method_index_select_args(input_name: str, args: str) -> tuple[str, str] | None:
        return _parse_stage7_method_gather_args(input_name, args)

    def _parse_stage7_take_along_dim_args(args: str) -> tuple[str, str] | None:
        keyword_input_match = re.search(
            r"(?:^|,\s*)input\s*=\s*(?P<input>[A-Za-z0-9_]+)(?=,|$)",
            args,
        )
        keyword_index_match = re.search(
            r"(?:^|,\s*)indices\s*=\s*(?P<indices>[A-Za-z0-9_]+)(?=,|$)",
            args,
        )
        keyword_dim_match = re.search(
            r"(?:^|,\s*)dim\s*=\s*0(?=,|$)",
            args,
        )
        if (
            keyword_input_match is not None
            and keyword_index_match is not None
            and keyword_dim_match is not None
        ):
            return (
                str(keyword_input_match.group("input")),
                str(keyword_index_match.group("indices")),
            )
        positional_patterns = [
            re.compile(
                r"^\s*(?P<input>[A-Za-z0-9_]+)\s*,\s*(?P<indices>[A-Za-z0-9_]+)\s*,\s*0\s*$"
            ),
            re.compile(
                r"^\s*(?P<input>[A-Za-z0-9_]+)\s*,\s*(?P<indices>[A-Za-z0-9_]+)\s*,\s*dim\s*=\s*0\s*$"
            ),
        ]
        positional_match = next(
            (
                match
                for pattern in positional_patterns
                if (match := pattern.match(args)) is not None
            ),
            None,
        )
        if positional_match is None:
            return None
        return (
            str(positional_match.group("input")),
            str(positional_match.group("indices")),
        )

    def _parse_stage7_method_take_along_dim_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            indices_name = parts[0].strip()
            dim_value = parts[1].strip()
            if dim_value == "0" and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None:
                return input_name, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        indices_name = keyword_values.get("indices")
        dim_value = keyword_values.get("dim")
        if (
            indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_name, indices_name

    has_stage7_def = False
    stage7_param_names: list[str] = []
    stage7_param_names_by_helper: Dict[str, list[str]] = {}
    stage7_helper_defs: Set[str] = set()
    has_forward_call = False
    stage7_called_helpers: Set[str] = set()
    packed_forward_names: Set[str] = set()
    packed_forward_indices: Dict[str, Set[str]] = {}
    packed_forward_helpers: Dict[str, str] = {}
    gather_reshape_count = 0
    singleton_anchor_count = 0
    stage7_shape_tensor_count = 0
    has_descriptor_permute = False
    has_stage7_return = False
    for line in lines:
        current_line = str(line)
        parsed_shape_alias = _parse_stage7_anchor_shape_alias(current_line)
        if parsed_shape_alias is not None:
            stage7_anchor_shape_alias_sources[parsed_shape_alias[0]] = parsed_shape_alias[1]
        stage7_def_match = stage7_def_re.match(current_line)
        if stage7_def_match is not None:
            helper_name = str(stage7_def_match.group("helper_name"))
            if helper_name == "forward":
                continue
            has_stage7_def = True
            stage7_param_names = re.findall(r"([A-Za-z0-9_]+): torch\.Tensor", current_line)
            stage7_param_names_by_helper[helper_name] = list(stage7_param_names)
            stage7_helper_defs.add(helper_name)
            continue
        forward_unpack_match = forward_unpack_re.match(current_line)
        if forward_unpack_match is not None:
            has_forward_call = True
            stage7_called_helpers.add(str(forward_unpack_match.group("helper_name")))
            continue
        packed_match = forward_packed_re.match(current_line)
        if packed_match is not None:
            packed_name = str(packed_match.group("packed"))
            packed_forward_names.add(packed_name)
            packed_forward_helpers[packed_name] = str(packed_match.group("helper_name"))
            continue
        forward_index_match = forward_index_re.match(current_line)
        if forward_index_match is not None:
            packed_name = str(forward_index_match.group("packed"))
            if packed_name in packed_forward_names:
                packed_forward_indices.setdefault(packed_name, set()).add(
                    str(forward_index_match.group("index"))
                )
                if packed_forward_indices[packed_name] == {"0", "1"}:
                    has_forward_call = True
                    helper_name = packed_forward_helpers.get(packed_name, None)
                    if helper_name is not None:
                        stage7_called_helpers.add(helper_name)
            continue
        if any(pattern.match(current_line) is not None for pattern in gather_reshape_res):
            gather_reshape_count += 1
            continue
        singleton_anchor_match = singleton_anchor_assign_re.match(current_line)
        if singleton_anchor_match is not None:
            anchor_args = _parse_stage7_singleton_anchor_args(str(singleton_anchor_match.group("args")))
            if anchor_args is not None:
                singleton_anchor_count += 1
                if _has_stage7_inline_branch_reshape(anchor_args[0]):
                    gather_reshape_count += 1
                continue
        anchor_pair_match = anchor_pair_assign_re.match(current_line)
        if anchor_pair_match is not None:
            anchor_args = _parse_stage7_singleton_anchor_args(str(anchor_pair_match.group("args")))
            if anchor_args is not None:
                singleton_anchor_count += 1
                if _has_stage7_inline_branch_reshape(anchor_args[0]):
                    gather_reshape_count += 1
                continue
        stage7_take_along_dim_assign_match = stage7_take_along_dim_assign_re.match(current_line)
        stage7_gather_method_assign_match = stage7_gather_method_assign_re.match(current_line)
        stage7_index_select_method_assign_match = stage7_index_select_method_assign_re.match(current_line)
        stage7_take_along_dim_method_assign_match = stage7_take_along_dim_method_assign_re.match(current_line)
        if stage7_take_along_dim_assign_match is not None:
            if _parse_stage7_take_along_dim_args(str(stage7_take_along_dim_assign_match.group("args"))) is not None:
                continue
        if stage7_gather_method_assign_match is not None:
            if _parse_stage7_method_gather_args(
                str(stage7_gather_method_assign_match.group("input")),
                str(stage7_gather_method_assign_match.group("args")),
            ) is not None:
                continue
        if stage7_index_select_method_assign_match is not None:
            if _parse_stage7_method_index_select_args(
                str(stage7_index_select_method_assign_match.group("input")),
                str(stage7_index_select_method_assign_match.group("args")),
            ) is not None:
                continue
        if stage7_take_along_dim_method_assign_match is not None:
            if _parse_stage7_method_take_along_dim_args(
                str(stage7_take_along_dim_method_assign_match.group("input")),
                str(stage7_take_along_dim_method_assign_match.group("args")),
            ) is not None:
                continue
        if stage7_shape_tensor_assign_re.match(current_line) is not None:
            stage7_shape_tensor_count += 1
            continue
        stage7_permute_match = stage7_permute_assign_re.match(current_line)
        if stage7_permute_match is not None:
            permute_args = str(stage7_permute_match.group("args"))
            descriptor_patterns = stage7_permute_dims_patterns[(1, 0)]
            if re.match(
                rf"^(?P<input>.+?),\s*(?:perm\s*=\s*|dims\s*=\s*)?{descriptor_patterns}$",
                permute_args,
            ) is not None:
                has_descriptor_permute = True
                continue
        stage7_mul_assign_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.mul\((?P<args>.+)\)$",
            current_line,
        )
        stage7_mul_method_assign_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>.+?)\.mul\((?P<args>.+)\)$",
            current_line,
        )
        if stage7_mul_assign_match is not None or stage7_mul_method_assign_match is not None:
            parsed_mul_args = (
                _parse_stage7_mul_args(str(stage7_mul_assign_match.group("args")))
                if stage7_mul_assign_match is not None
                else _parse_stage7_method_mul_args(
                    str(stage7_mul_method_assign_match.group("input")),
                    str(stage7_mul_method_assign_match.group("args")),
                )
            )
            if (
                parsed_mul_args is not None
                and (
                    _has_stage7_inline_branch_reshape(parsed_mul_args[0])
                    or _has_stage7_inline_branch_reshape(parsed_mul_args[1])
                )
            ):
                gather_reshape_count += 1
                continue
        if stage7_return_re.match(current_line) is not None:
            has_stage7_return = True
    fallback_branch_count = min(
        stage7_shape_tensor_count,
        max(0, len(stage7_param_names) - 2),
    )
    candidate_stage7_helpers = stage7_helper_defs & stage7_called_helpers
    if candidate_stage7_helpers:
        stage7_param_names = max(
            (stage7_param_names_by_helper.get(helper_name, []) for helper_name in candidate_stage7_helpers),
            key=len,
            default=stage7_param_names,
        )
        fallback_branch_count = min(
            stage7_shape_tensor_count,
            max(0, len(stage7_param_names) - 2),
        )
    structural_fallback_ready = has_descriptor_permute and fallback_branch_count >= 1
    return (
        has_stage7_def
        and has_forward_call
        and bool(candidate_stage7_helpers)
        and (gather_reshape_count >= 1 or structural_fallback_ready)
        and (singleton_anchor_count >= 1 or structural_fallback_ready)
        and has_stage7_return
    )

def _infer_structural_rank4_channel_count(
    tensor_name: str,
    shape: Sequence[int],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> int | None:
    preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
        tensor_name,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
        shape_hint=shape,
    )
    if preferred_channel_count is not None:
        return int(preferred_channel_count)
    return _infer_unique_channel_count_from_rank4_shape(shape)


def _rewrite_structural_mixed_layout_binary_anchor_and_add(
    lines: List[str],
    index: int,
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    direct_conv_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    relu_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    parsed_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[index])
    if parsed_anchor is None:
        return False
    indent, lhs0, lhs1, input_a, input_b, current_shape = parsed_anchor
    input_a_is_cf = _fast_precanonicalize_is_cf_like(input_a, dynamic_cf_like_names, context)
    input_b_is_cf = _fast_precanonicalize_is_cf_like(input_b, dynamic_cf_like_names, context)
    input_a_is_nhwc = _fast_precanonicalize_is_nhwc_like(input_a, dynamic_nhwc_like_names, context)
    input_b_is_nhwc = _fast_precanonicalize_is_nhwc_like(input_b, dynamic_nhwc_like_names, context)
    if not (
        (input_a_is_cf and input_b_is_nhwc)
        or (input_b_is_cf and input_a_is_nhwc)
    ):
        return False
    cf_name = input_a if input_a_is_cf else input_b
    preferred_channel_count = _infer_structural_rank4_channel_count(
        cf_name,
        current_shape,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    )
    if preferred_channel_count is None:
        return False
    add_lhs: str | None = None
    parsed_dynamic_add = None
    parsed_static_add = None
    if index + 1 < len(lines):
        parsed_dynamic_add = _parse_dynamic_binary_add_align_assign(lines[index + 1])
        if (
            parsed_dynamic_add is not None
            and {parsed_dynamic_add[2], parsed_dynamic_add[3]} == {lhs0, lhs1}
        ):
            add_lhs = str(parsed_dynamic_add[1])
        else:
            parsed_dynamic_add = None
            parsed_static_add = _parse_static_binary_add_align_assign(lines[index + 1])
            if (
                parsed_static_add is not None
                and {parsed_static_add[2], parsed_static_add[3]} == {lhs0, lhs1}
            ):
                add_lhs = str(parsed_static_add[1])
            else:
                parsed_static_add = None
    if add_lhs is not None:
        def _collect_local_aliases(seed_name: str, start_line: int) -> Set[str]:
            alias_names: Set[str] = {str(seed_name)}
            for lookahead in range(start_line, min(len(lines), start_line + 5)):
                relu_match = relu_assign_re.match(lines[lookahead])
                if relu_match is not None and str(relu_match.group("input")) in alias_names:
                    alias_names.add(str(relu_match.group("lhs")))
                    continue
                alias_assign = _parse_simple_assignment_line(lines[lookahead])
                rhs_expr = str(alias_assign[2]).strip() if alias_assign is not None else None
                if (
                    alias_assign is not None
                    and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr or "") is not None
                    and rhs_expr in alias_names
                ):
                    alias_names.add(str(alias_assign[1]))
            return alias_names

        def _parse_channel_last_rank4_gap_mean_input(line: str) -> str | None:
            assign = _parse_simple_assignment_line(line)
            if assign is None:
                return None
            _, _, rhs = assign
            stripped_rhs = str(rhs).strip()
            mean_match = re.fullmatch(r"torch\.mean\((?P<args>.+)\)", stripped_rhs)
            if mean_match is not None:
                parts = _split_top_level_csv_exprs(str(mean_match.group("args")))
                input_expr: str | None = None
                dim_expr: str | None = None
                keepdim_expr: str | None = None
                positional_index = 0
                for part in parts:
                    if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "input":
                            input_expr = value
                        elif key == "dim":
                            dim_expr = value
                        elif key == "keepdim":
                            keepdim_expr = value
                        continue
                    if positional_index == 0:
                        input_expr = part.strip()
                    elif positional_index == 1:
                        dim_expr = part.strip()
                    positional_index += 1
                if input_expr is None or dim_expr is None:
                    return None
                try:
                    dim_value = ast.literal_eval(dim_expr)
                except Exception:
                    return None
                if not isinstance(dim_value, (list, tuple)):
                    return None
                if [int(v) for v in list(dim_value)] != [1, 2]:
                    return None
                if keepdim_expr is not None and keepdim_expr != "True":
                    return None
                if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
                    return None
                return str(input_expr)
            prefix = "_reduce_mean("
            if not stripped_rhs.startswith(prefix) or not stripped_rhs.endswith(")"):
                return None
            parts = _split_top_level_csv_exprs(stripped_rhs[len(prefix) : -1])
            input_expr: str | None = None
            axes_expr: str | None = None
            keepdims_expr: str | None = None
            positional_index = 0
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "input":
                        input_expr = value
                    elif key == "axes":
                        axes_expr = value
                    elif key == "keepdims":
                        keepdims_expr = value
                    continue
                if positional_index == 0:
                    input_expr = part.strip()
                elif positional_index == 1:
                    axes_expr = part.strip()
                elif positional_index == 2:
                    keepdims_expr = part.strip()
                positional_index += 1
            if input_expr is None or axes_expr is None:
                return None
            axes_match = re.fullmatch(
                r"_normalize_axes\(\[(?P<a0>-?\d+),\s*(?P<a1>-?\d+)\],\s*.+\.ndim\)",
                axes_expr,
            )
            if axes_match is None:
                return None
            if [int(axes_match.group("a0")), int(axes_match.group("a1"))] != [1, 2]:
                return None
            if keepdims_expr is not None and keepdims_expr != "True":
                return None
            if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
                return None
            return str(input_expr)

        def _resolve_binary_add_lhs(anchor_index: int, lhs_a: str, lhs_b: str) -> str | None:
            if anchor_index + 1 >= len(lines):
                return None
            downstream_dynamic_add = _parse_dynamic_binary_add_align_assign(lines[anchor_index + 1])
            if (
                downstream_dynamic_add is not None
                and {downstream_dynamic_add[2], downstream_dynamic_add[3]} == {lhs_a, lhs_b}
            ):
                return str(downstream_dynamic_add[1])
            downstream_static_add = _parse_static_binary_add_align_assign(lines[anchor_index + 1])
            if (
                downstream_static_add is not None
                and {downstream_static_add[2], downstream_static_add[3]} == {lhs_a, lhs_b}
            ):
                return str(downstream_static_add[1])
            return None

        def _has_channel_first_consumer(seed_name: str, search_start: int) -> bool:
            pending: List[Tuple[str, int, int]] = [(str(seed_name), int(search_start), 0)]
            seen: Set[Tuple[str, int]] = set()
            while pending:
                current_name, current_start, depth = pending.pop(0)
                alias_names = _collect_local_aliases(current_name, current_start)
                for lookahead in range(current_start, len(lines)):
                    direct_conv_match = direct_conv_assign_re.match(lines[lookahead])
                    if (
                        direct_conv_match is not None
                        and str(direct_conv_match.group("input")) in alias_names
                    ):
                        consumer_lhs = str(direct_conv_match.group("lhs"))
                        candidate_module = str(direct_conv_match.group("module"))
                        candidate_channels = context.conv_block_in_channels.get(candidate_module, None)
                        consumer_is_cf = _fast_precanonicalize_is_cf_like(
                            consumer_lhs,
                            dynamic_cf_like_names,
                            context,
                        )
                        consumer_is_nhwc = _fast_precanonicalize_is_nhwc_like(
                            consumer_lhs,
                            dynamic_nhwc_like_names,
                            context,
                        )
                        if (
                            candidate_channels is not None
                            and int(candidate_channels) == int(preferred_channel_count)
                            and consumer_is_cf
                            and not consumer_is_nhwc
                        ):
                            return True
                if depth >= 1:
                    continue
                for candidate_index in range(current_start, min(len(lines), current_start + 24)):
                    downstream_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[candidate_index])
                    if downstream_anchor is None:
                        continue
                    if current_name not in {str(downstream_anchor[3]), str(downstream_anchor[4])}:
                        continue
                    if int(preferred_channel_count) not in list(downstream_anchor[5])[1:]:
                        continue
                    downstream_add_lhs = _resolve_binary_add_lhs(
                        candidate_index,
                        str(downstream_anchor[1]),
                        str(downstream_anchor[2]),
                    )
                    if downstream_add_lhs is None:
                        continue
                    key = (downstream_add_lhs, depth + 1)
                    if key in seen:
                        continue
                    seen.add(key)
                    pending.append((downstream_add_lhs, candidate_index + 1, depth + 1))
            return False

        def _has_explicit_channel_last_gap_consumer(seed_name: str, search_start: int) -> bool:
            pending: List[Tuple[str, int, int]] = [(str(seed_name), int(search_start), 0)]
            seen: Set[Tuple[str, int]] = set()
            while pending:
                current_name, current_start, depth = pending.pop(0)
                alias_names = _collect_local_aliases(current_name, current_start)
                for lookahead in range(current_start, len(lines)):
                    gap_input_name = _parse_channel_last_rank4_gap_mean_input(lines[lookahead])
                    if gap_input_name is not None and gap_input_name in alias_names:
                        return True
                if depth >= 1:
                    continue
                for candidate_index in range(current_start, min(len(lines), current_start + 24)):
                    downstream_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[candidate_index])
                    if downstream_anchor is None:
                        continue
                    if current_name not in {str(downstream_anchor[3]), str(downstream_anchor[4])}:
                        continue
                    downstream_add_lhs = _resolve_binary_add_lhs(
                        candidate_index,
                        str(downstream_anchor[1]),
                        str(downstream_anchor[2]),
                    )
                    if downstream_add_lhs is None:
                        continue
                    key = (downstream_add_lhs, depth + 1)
                    if key in seen:
                        continue
                    seen.add(key)
                    pending.append((downstream_add_lhs, candidate_index + 1, depth + 1))
            return False

        alias_names = _collect_local_aliases(add_lhs, index + 2)
        has_explicit_channel_last_gap_consumer = _has_explicit_channel_last_gap_consumer(
            add_lhs,
            index + 2,
        )
        if (
            _has_channel_first_consumer(add_lhs, index + 2)
            and not has_explicit_channel_last_gap_consumer
        ):
            normalized_cf_shape = _normalize_cf_rank4_shape(
                current_shape,
                preferred_channel_count=preferred_channel_count,
            )
            changed = False
            rewritten_anchor = (
                f"{indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor("
                f"{input_a}, {input_b}, {repr(normalized_cf_shape)})"
            )
            if rewritten_anchor != lines[index]:
                lines[index] = rewritten_anchor
                dynamic_cf_like_names.update({lhs0, lhs1})
                dynamic_nhwc_like_names.difference_update({lhs0, lhs1})
                context.static_shapes[lhs0] = list(normalized_cf_shape)
                context.static_shapes[lhs1] = list(normalized_cf_shape)
                changed = True
            if parsed_dynamic_add is not None:
                rewritten_add = (
                    f"{parsed_dynamic_add[0]}{parsed_dynamic_add[1]} = _align_tensor_to_target_shape("
                    f"torch.add({parsed_dynamic_add[2]}, {parsed_dynamic_add[3]}), {repr(normalized_cf_shape)})"
                )
                if rewritten_add != lines[index + 1]:
                    lines[index + 1] = rewritten_add
                    dynamic_cf_like_names.update(alias_names)
                    dynamic_nhwc_like_names.difference_update(alias_names)
                    for alias_name in alias_names:
                        context.static_shapes[alias_name] = list(normalized_cf_shape)
                    changed = True
            elif parsed_static_add is not None:
                rewritten_add = (
                    f"{parsed_static_add[0]}{parsed_static_add[1]} = _align_tensor_to_target_shape("
                    f"torch.add({parsed_static_add[2]}, {parsed_static_add[3]}), {repr(normalized_cf_shape)})"
                )
                if rewritten_add != lines[index + 1]:
                    lines[index + 1] = rewritten_add
                    dynamic_cf_like_names.update(alias_names)
                    dynamic_nhwc_like_names.difference_update(alias_names)
                    for alias_name in alias_names:
                        context.static_shapes[alias_name] = list(normalized_cf_shape)
                    changed = True
            return changed
    normalized_shape = _normalize_nhwc_rank4_shape(
        current_shape,
        preferred_channel_count=preferred_channel_count,
    )
    changed = False
    rewritten_anchor = (
        f"{indent}{lhs0}, {lhs1} = _align_binary_inputs_to_anchor("
        f"{input_a}, {input_b}, {repr(normalized_shape)})"
    )
    if rewritten_anchor != lines[index]:
        lines[index] = rewritten_anchor
        dynamic_nhwc_like_names.update({lhs0, lhs1})
        changed = True
    if index + 1 < len(lines):
        parsed_add = _parse_dynamic_binary_add_align_assign(lines[index + 1])
        if parsed_add is not None and {parsed_add[2], parsed_add[3]} == {lhs0, lhs1}:
            rewritten_add = (
                f"{parsed_add[0]}{parsed_add[1]} = _align_tensor_to_target_shape("
                f"torch.add({parsed_add[2]}, {parsed_add[3]}), {repr(normalized_shape)})"
            )
            if rewritten_add != lines[index + 1]:
                lines[index + 1] = rewritten_add
                dynamic_nhwc_like_names.add(parsed_add[1])
                changed = True
        else:
            parsed_static_add = _parse_static_binary_add_align_assign(lines[index + 1])
            if parsed_static_add is not None and {parsed_static_add[2], parsed_static_add[3]} == {lhs0, lhs1}:
                rewritten_static_add = (
                    f"{parsed_static_add[0]}{parsed_static_add[1]} = _align_tensor_to_target_shape("
                    f"torch.add({parsed_static_add[2]}, {parsed_static_add[3]}), {repr(normalized_shape)})"
                )
                if rewritten_static_add != lines[index + 1]:
                    lines[index + 1] = rewritten_static_add
                    dynamic_nhwc_like_names.add(parsed_static_add[1])
                    changed = True
    return changed


def _rewrite_structural_plain_mixed_layout_attention_adds(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    direct_conv_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    rank5_reshape_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\("
        r"(?P<input>[A-Za-z0-9_]+), "
        r"\[(?P<n>\d+), (?P<c>\d+), 1, (?P<h>\d+), (?P<w>\d+)\]\)$"
    )

    def _resolve_assigned_permute_layout(name: str) -> str | None:
        for candidate_line in lines:
            assign = _parse_simple_assignment_line(candidate_line)
            if assign is None or str(assign[1]) != str(name):
                continue
            stripped = str(assign[2]).strip()
            if stripped.endswith(".contiguous()"):
                stripped = stripped[: -len(".contiguous()")].strip()
            for prefix in ("_torch_permute(", "torch.permute("):
                if not stripped.startswith(prefix) or not stripped.endswith(")"):
                    continue
                parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
                perm_expr: str | None = None
                if len(parts) == 2 and "=" not in parts[1]:
                    perm_expr = parts[1].strip()
                else:
                    for part in parts:
                        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                            continue
                        key, value = part.split("=", 1)
                        if key.strip() in {"perm", "dims"}:
                            perm_expr = value.strip()
                            break
                if perm_expr is None:
                    continue
                try:
                    perm_value = ast.literal_eval(perm_expr)
                except Exception:
                    continue
                if not isinstance(perm_value, (list, tuple)):
                    continue
                perm = [int(v) for v in list(perm_value)]
                if perm == [0, 2, 3, 1]:
                    return "nhwc"
                if perm == [0, 3, 1, 2]:
                    return "cf"
        return None

    def _resolve_runtime_layout(name: str) -> str | None:
        resolved = _fast_precanonicalize_resolve_alias(str(name), context.aliases)
        for candidate_line in lines:
            conv_assign = direct_conv_assign_re.match(candidate_line)
            if conv_assign is None or str(conv_assign.group("lhs")) != resolved:
                continue
            conv_input = str(conv_assign.group("input"))
            input_is_cf = _fast_precanonicalize_is_cf_like(
                conv_input,
                dynamic_cf_like_names,
                context,
            )
            input_is_nhwc = _fast_precanonicalize_is_nhwc_like(
                conv_input,
                dynamic_nhwc_like_names,
                context,
            )
            if input_is_nhwc and not input_is_cf:
                return "nhwc"
            if input_is_cf and not input_is_nhwc:
                return "cf"
            inferred_input_layout = _resolve_assigned_permute_layout(conv_input)
            if inferred_input_layout is not None:
                return inferred_input_layout
        resolved_is_cf = _fast_precanonicalize_is_cf_like(
            resolved,
            dynamic_cf_like_names,
            context,
        )
        resolved_is_nhwc = _fast_precanonicalize_is_nhwc_like(
            resolved,
            dynamic_nhwc_like_names,
            context,
        )
        if resolved_is_cf and not resolved_is_nhwc:
            return "cf"
        if resolved_is_nhwc and not resolved_is_cf:
            return "nhwc"
        return None

    for index, line in enumerate(lines):
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            continue
        indent, lhs, rhs = assign
        add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", str(rhs).strip())
        if add_match is None:
            continue
        add_args = _parse_binary_add_args(str(add_match.group("args")))
        if add_args is None:
            continue
        input_a = str(add_args[0]).strip()
        input_b = str(add_args[1]).strip()
        input_a_layout = _resolve_runtime_layout(input_a)
        input_b_layout = _resolve_runtime_layout(input_b)
        if {input_a_layout, input_b_layout} != {"cf", "nhwc"}:
            continue
        cf_input = input_a if input_a_layout == "cf" else input_b
        nhwc_input = input_a if input_a_layout == "nhwc" else input_b
        cf_channel_count = _fast_precanonicalize_preferred_channel_count(
            cf_input,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        )
        if cf_channel_count is None or int(cf_channel_count) <= 1:
            continue
        nhwc_producer = context.module_output_producers.get(
            _fast_precanonicalize_resolve_alias(nhwc_input, context.aliases),
            None,
        )
        nhwc_out_channels = (
            context.conv_block_out_channels.get(nhwc_producer, None)
            if nhwc_producer is not None
            else None
        )
        alias_names: Set[str] = {str(lhs)}
        target_cf_shape: List[int] | None = None
        for lookahead in range(index + 1, min(len(lines), index + 6)):
            alias_assign = _parse_simple_assignment_line(lines[lookahead])
            rhs_expr = str(alias_assign[2]).strip() if alias_assign is not None else ""
            if (
                alias_assign is not None
                and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr) is not None
                and rhs_expr in alias_names
            ):
                alias_names.add(str(alias_assign[1]))
            reshape_match = rank5_reshape_re.match(lines[lookahead])
            if (
                reshape_match is not None
                and str(reshape_match.group("input")) in alias_names
                and int(reshape_match.group("c")) == int(cf_channel_count)
                and int(reshape_match.group("h")) > 1
                and int(reshape_match.group("w")) > 1
            ):
                target_cf_shape = [
                    int(reshape_match.group("n")),
                    int(reshape_match.group("c")),
                    int(reshape_match.group("h")),
                    int(reshape_match.group("w")),
                ]
                break
        if target_cf_shape is None:
            continue
        if (
            nhwc_out_channels is not None
            and int(nhwc_out_channels) not in {1, int(target_cf_shape[1])}
        ):
            continue
        rewritten_input_a = (
            f"{input_a}.permute(0, 3, 1, 2).contiguous()"
            if input_a == nhwc_input
            else input_a
        )
        rewritten_input_b = (
            f"{input_b}.permute(0, 3, 1, 2).contiguous()"
            if input_b == nhwc_input
            else input_b
        )
        rewritten_line = f"{indent}{lhs} = torch.add({rewritten_input_a}, {rewritten_input_b})"
        if lines[index] == rewritten_line:
            continue
        lines[index] = rewritten_line
        dynamic_cf_like_names.update(alias_names)
        dynamic_nhwc_like_names.difference_update(alias_names)
        for alias_name in alias_names:
            context.static_shapes[str(alias_name)] = list(target_cf_shape)
        changed = True
    return changed


def _repair_plain_mixed_layout_attention_adds(
    lines: Sequence[str],
) -> List[str]:
    rewritten = [str(line) for line in lines]
    context = _build_fast_precanonicalize_repair_context(rewritten)
    dynamic_cf_like_names: Set[str] = set(context.cf_like_names)
    dynamic_nhwc_like_names: Set[str] = set(context.nhwc_like_names)
    _rewrite_structural_plain_mixed_layout_attention_adds(
        rewritten,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    )
    return rewritten


def _repair_nhwc_named_binary_add_align_outputs(
    lines: Sequence[str],
) -> List[str]:
    rewritten = [str(line) for line in lines]
    context = _build_fast_precanonicalize_repair_context(rewritten)
    dynamic_nhwc_like_names: Set[str] = set(context.nhwc_like_names)
    dynamic_cf_like_names: Set[str] = set(context.cf_like_names)
    for index, line in enumerate(rewritten):
        parsed_add = _parse_static_binary_add_align_assign(line)
        if parsed_add is None:
            continue
        indent, lhs, input_a, input_b, current_shape = parsed_add
        if (
            len(current_shape) != 4
            or ("_nhwc" not in lhs and "_to_nhwc" not in lhs)
        ):
            continue
        anchor_inputs: Tuple[str, str] | None = None
        preferred_channel_count: int | None = None
        for anchor_back in range(max(0, index - 4), index):
            parsed_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(
                rewritten[anchor_back]
            )
            if (
                parsed_anchor is None
                or {str(parsed_anchor[1]), str(parsed_anchor[2])} != {str(input_a), str(input_b)}
            ):
                continue
            anchor_inputs = (str(parsed_anchor[3]), str(parsed_anchor[4]))
            for candidate_name in anchor_inputs:
                candidate_channel_count = _fast_precanonicalize_preferred_channel_count(
                    candidate_name,
                    dynamic_cf_like_names,
                    dynamic_nhwc_like_names,
                    context,
                )
                if candidate_channel_count is not None:
                    preferred_channel_count = int(candidate_channel_count)
                    break
            break
        normalized_shape = _normalize_nhwc_rank4_shape(
            current_shape,
            preferred_channel_count=preferred_channel_count,
        )
        if normalized_shape == current_shape:
            continue
        has_nhwc_evidence = any(
            _fast_precanonicalize_is_nhwc_like(name, dynamic_nhwc_like_names, context)
            or "_nhwc" in _fast_precanonicalize_resolve_alias(name, context.aliases)
            for name in [input_a, input_b]
        )
        has_cf_evidence = any(
            _fast_precanonicalize_is_cf_like(name, dynamic_cf_like_names, context)
            for name in [input_a, input_b]
        )
        if anchor_inputs is not None:
            has_nhwc_evidence = has_nhwc_evidence or any(
                _fast_precanonicalize_is_nhwc_like(name, dynamic_nhwc_like_names, context)
                or "_nhwc" in _fast_precanonicalize_resolve_alias(name, context.aliases)
                for name in anchor_inputs
            )
            has_cf_evidence = has_cf_evidence or any(
                _fast_precanonicalize_is_cf_like(name, dynamic_cf_like_names, context)
                or _fast_precanonicalize_preferred_channel_count(
                    name,
                    dynamic_cf_like_names,
                    dynamic_nhwc_like_names,
                    context,
                ) is not None
                for name in anchor_inputs
            )
        if not has_nhwc_evidence or not has_cf_evidence:
            continue
        rewritten[index] = (
            f"{indent}{lhs} = _align_tensor_to_target_shape("
            f"torch.add({input_a}, {input_b}), {repr(normalized_shape)})"
        )
        context.static_shapes[str(lhs)] = list(normalized_shape)
        dynamic_nhwc_like_names.add(str(lhs))
        for anchor_back in range(max(0, index - 4), index):
            parsed_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(
                rewritten[anchor_back]
            )
            if (
                parsed_anchor is None
                or {str(parsed_anchor[1]), str(parsed_anchor[2])} != {str(input_a), str(input_b)}
            ):
                continue
            rewritten[anchor_back] = (
                f"{parsed_anchor[0]}{parsed_anchor[1]}, {parsed_anchor[2]} = "
                f"_align_binary_inputs_to_anchor({parsed_anchor[3]}, {parsed_anchor[4]}, "
                f"{repr(normalized_shape)})"
            )
            context.static_shapes[str(parsed_anchor[1])] = list(normalized_shape)
            context.static_shapes[str(parsed_anchor[2])] = list(normalized_shape)
            dynamic_nhwc_like_names.update({str(parsed_anchor[1]), str(parsed_anchor[2])})
            break
    return rewritten


def _repair_cf_consumed_permute_aliases(
    lines: Sequence[str],
) -> List[str]:
    rewritten = [str(line) for line in lines]
    context = _build_fast_precanonicalize_repair_context(rewritten)
    dynamic_cf_like_names: Set[str] = set(context.cf_like_names)
    dynamic_nhwc_like_names: Set[str] = set(context.nhwc_like_names)
    rank5_reshape_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\("
        r"(?P<input>[A-Za-z0-9_]+), \[(?P<n>\d+), (?P<c>\d+), 1, (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    direct_conv_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    for index, line in enumerate(rewritten):
        permute_assign = _parse_torch_permute_assign(line)
        if (
            permute_assign is None
            or permute_assign[3] != [0, 3, 1, 2]
        ):
            continue
        indent, lhs, source_name, _ = permute_assign
        if "_nhwc" in str(source_name) or "_to_nhwc" in str(source_name):
            continue
        source_shape = context.static_shapes.get(str(source_name), None)
        preferred_channel_count = _fast_precanonicalize_preferred_channel_count(
            source_name,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        )
        source_layout_hint = (
            _fast_precanonicalize_rank4_layout_hint(source_shape)
            if source_shape is not None and len(source_shape) == 4
            else None
        )
        source_is_cf_like = _fast_precanonicalize_is_cf_like(
            source_name,
            dynamic_cf_like_names,
            context,
        )
        if (
            not source_is_cf_like
            and source_layout_hint != "cf"
            and not (source_shape is not None and len(source_shape) == 4)
        ):
            continue
        if (
            preferred_channel_count is None
            and source_shape is not None
            and len(source_shape) == 4
            and (source_layout_hint == "cf" or int(source_shape[1]) > 1)
        ):
            preferred_channel_count = int(source_shape[1])
        has_cf_consumer = False
        has_nhwc_consumer = False
        for consumer_index in context.consumers.get(str(lhs), []):
            if consumer_index <= index:
                continue
            consumer_line = rewritten[consumer_index]
            conv_assign = direct_conv_assign_re.match(consumer_line)
            if conv_assign is not None and str(conv_assign.group("input")) == str(lhs):
                has_cf_consumer = True
                continue
            reshape_match = rank5_reshape_re.match(consumer_line)
            if (
                reshape_match is not None
                and str(reshape_match.group("input")) == str(lhs)
                and preferred_channel_count is not None
                and int(reshape_match.group("c")) == int(preferred_channel_count)
            ):
                has_cf_consumer = True
                continue
            assign = _parse_simple_assignment_line(consumer_line)
            rhs_expr = str(assign[2]).strip() if assign is not None else ""
            if assign is not None and "torch.add(" in rhs_expr:
                add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", rhs_expr)
                add_args = _parse_binary_add_args(str(add_match.group("args"))) if add_match is not None else None
                if add_args is not None and str(lhs) in {str(add_args[0]), str(add_args[1])}:
                    has_cf_consumer = True
                    continue
            if assign is not None and "torch.mul(" in rhs_expr:
                mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", rhs_expr)
                mul_args = _parse_binary_mul_args(str(mul_match.group("args"))) if mul_match is not None else None
                if mul_args is not None and str(lhs) in {str(mul_args[0]), str(mul_args[1])}:
                    has_cf_consumer = True
                    continue
            if "_apply_concat(" in rhs_expr:
                concat_args = _parse_apply_concat_inputs_axis_and_shape(rhs_expr)
                if (
                    concat_args is not None
                    and str(lhs) in [name.strip() for name in concat_args[0]]
                    and concat_args[1] == 3
                ):
                    has_nhwc_consumer = True
        if not has_cf_consumer or has_nhwc_consumer:
            continue
        rewritten[index] = f"{indent}{lhs} = {source_name}"
        dynamic_cf_like_names.add(str(lhs))
        dynamic_nhwc_like_names.discard(str(lhs))
        if source_shape is not None:
            context.static_shapes[str(lhs)] = [int(v) for v in list(source_shape)]
    return rewritten


def _rewrite_structural_channel_last_pool_pad_pairs(
    lines: List[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    for index in range(len(lines) - 1):
        pad_assign = _parse_constant_pad_assign(lines[index])
        pool_assign = _parse_apply_pool2d_assign_with_shape(lines[index + 1])
        if (
            pad_assign is None
            or pool_assign is None
            or pad_assign[1] != pool_assign[2]
            or not pool_assign[6]
        ):
            continue
        aligned_pad_input = _parse_align_tensor_target_shape_expr(pad_assign[2])
        aligned_pad_source = (
            aligned_pad_input[0].strip()
            if aligned_pad_input is not None
            else None
        )
        if not (
            _fast_precanonicalize_is_nhwc_like(pad_assign[2], dynamic_nhwc_like_names, context)
            or (
                aligned_pad_source is not None
                and _fast_precanonicalize_is_nhwc_like(
                    aligned_pad_source,
                    dynamic_nhwc_like_names,
                    context,
                )
            )
            or "_nhwc" in str(pad_assign[2])
            or (aligned_pad_source is not None and "_nhwc" in str(aligned_pad_source))
            or "_nhwc" in str(pool_assign[1])
        ):
            continue
        if len(pad_assign[3]) == 6 and [int(v) for v in list(pad_assign[3][:2])] == [0, 0]:
            continue
        rewritten_pad_values = _convert_nchw_pad_to_nhwc_pad_values(pad_assign[3])
        if rewritten_pad_values is None or rewritten_pad_values == pad_assign[3]:
            continue
        rewritten_pad_line = (
            f"{pad_assign[0]}{pad_assign[1]} = "
            f"F.pad({pad_assign[2]}, {repr(rewritten_pad_values)}, mode='constant', value={pad_assign[4]})"
        )
        if rewritten_pad_line != lines[index]:
            lines[index] = rewritten_pad_line
            changed = True
    return changed


def _rewrite_structural_nhwc_image_tail_bridges(
    lines: List[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    image_tail_reshape_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.reshape\((?P<input>[A-Za-z0-9_]+), "
        r"(?:\[int\(v\) for v in \[(?P<h0>\d+), (?P<w0>\d+), (?P<c0>\d+)\]\]|\[(?P<h1>\d+), (?P<w1>\d+), (?P<c1>\d+)\])\)$"
    )

    def _split_binary_args(expr: str) -> Tuple[str, str] | None:
        depth = 0
        current: List[str] = []
        parts: List[str] = []
        for char in expr:
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            current.append(char)
        if current:
            parts.append("".join(current).strip())
        if len(parts) != 2:
            return None
        return (parts[0], parts[1])

    def _split_call_args(expr: str) -> List[str]:
        depth = 0
        current: List[str] = []
        parts: List[str] = []
        for char in expr:
            if char == "," and depth == 0:
                parts.append("".join(current).strip())
                current = []
                continue
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth = max(0, depth - 1)
            current.append(char)
        if current:
            parts.append("".join(current).strip())
        return parts

    def _parse_static_rank4_binary_align_assign(
        src_line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(src_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        aligned_expr = _parse_align_tensor_target_shape_expr(rhs)
        if aligned_expr is None:
            return None
        input_expr, shape_expr = aligned_expr
        target_shape = _parse_rank4_shape_literal(shape_expr)
        if target_shape is None:
            return None
        binary_match = re.fullmatch(
            r"torch\.(?P<op>add|sub|mul|div|maximum|minimum)\((?P<args>.+)\)",
            input_expr.strip(),
        )
        if binary_match is None:
            return None
        binary_args = _split_binary_args(str(binary_match.group("args")))
        if binary_args is None:
            return None
        return (
            indent,
            lhs,
            str(binary_match.group("op")),
            str(binary_args[0]).strip(),
            str(binary_args[1]).strip(),
            list(target_shape),
        )

    def _parse_image_tail_binary_add_assign(
        src_line: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        parsed_add = _parse_static_binary_add_align_assign(src_line)
        if parsed_add is not None:
            indent, lhs, input_a, input_b, current_shape = parsed_add
            return (indent, lhs, input_a, input_b, current_shape)
        assign = _parse_simple_assignment_line(src_line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        aligned_expr = _parse_align_tensor_target_shape_expr(rhs)
        if aligned_expr is None:
            return None
        input_expr, shape_expr = aligned_expr
        current_shape = _parse_rank4_shape_literal(shape_expr)
        if current_shape is None:
            return None
        binary_match = re.fullmatch(
            r"torch\.add\(\*_align_binary_inputs_to_anchor\((?P<args>.+)\)\)",
            input_expr.strip(),
        )
        if binary_match is None:
            return None
        call_args = _split_call_args(str(binary_match.group("args")))
        if len(call_args) != 3:
            return None
        return (
            indent,
            lhs,
            str(call_args[0]).strip(),
            str(call_args[1]).strip(),
            list(current_shape),
        )

    def _rewrite_upstream_binary_producer(
        source_name: str,
        max_index: int,
        output_hwc_shape: List[int],
    ) -> bool:
        local_changed = False
        producer_index: int | None = None
        producer_binary_assign = None
        for producer_back in range(max(0, max_index - 6), max_index):
            candidate_assign = _parse_static_rank4_binary_align_assign(lines[producer_back])
            if candidate_assign is not None and str(candidate_assign[1]) == source_name:
                producer_index = producer_back
                producer_binary_assign = candidate_assign
                break
        if producer_index is None or producer_binary_assign is None:
            return False
        (
            producer_indent,
            producer_lhs,
            producer_op,
            producer_input_a,
            producer_input_b,
            producer_shape,
        ) = producer_binary_assign
        if len(producer_shape) != 4:
            return False
        rewritten_producer_binary = (
            f"{producer_indent}{producer_lhs} = _align_tensor_to_target_shape("
            f"torch.{producer_op}({producer_input_a}, {producer_input_b}), {repr(output_hwc_shape)})"
        )
        if lines[producer_index] != rewritten_producer_binary:
            lines[producer_index] = rewritten_producer_binary
            dynamic_nhwc_like_names.add(producer_lhs)
            context.static_shapes[producer_lhs] = list(output_hwc_shape)
            local_changed = True
        for producer_anchor_back in range(max(0, producer_index - 6), producer_index):
            producer_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(
                lines[producer_anchor_back]
            )
            if (
                producer_anchor_assign is None
                or {str(producer_anchor_assign[1]), str(producer_anchor_assign[2])}
                != {producer_input_a, producer_input_b}
            ):
                continue
            rewritten_producer_anchor = (
                f"{producer_anchor_assign[0]}{producer_anchor_assign[1]}, {producer_anchor_assign[2]} = "
                f"_align_binary_inputs_to_anchor({producer_anchor_assign[3]}, {producer_anchor_assign[4]}, "
                f"{repr(output_hwc_shape)})"
            )
            if lines[producer_anchor_back] != rewritten_producer_anchor:
                lines[producer_anchor_back] = rewritten_producer_anchor
                context.static_shapes[str(producer_anchor_assign[1])] = list(output_hwc_shape)
                context.static_shapes[str(producer_anchor_assign[2])] = list(output_hwc_shape)
                local_changed = True
            break
        return local_changed
    for index, line in enumerate(lines):
        parsed_add = _parse_image_tail_binary_add_assign(line)
        if parsed_add is None:
            continue
        indent, lhs, input_a, input_b, current_shape = parsed_add
        if len(current_shape) != 4 or int(current_shape[0]) != 1:
            continue
        gathered_name: str | None = None
        channel_count: int | None = None
        output_hwc_shape: List[int] | None = None
        for lookahead in range(index + 1, min(len(lines), index + 12)):
            gather_assign = _parse_simple_assignment_line(lines[lookahead])
            if (
                gather_assign is not None
                and str(gather_assign[2]).strip() == f"{lhs}[[0], :, :, :]"
            ):
                gathered_name = str(gather_assign[1])
                continue
            if gathered_name is None:
                continue
            reshape_assign = _parse_simple_assignment_line(lines[lookahead])
            if reshape_assign is not None:
                reshape_rhs = str(reshape_assign[2]).strip()
                reshape_match = re.match(
                    rf"^torch\.reshape\({re.escape(gathered_name)}, _resolve_reshape_shape\(\[-1, (?P<c>\d+)\], {re.escape(gathered_name)}, allow_zero=False\)\)$",
                    reshape_rhs,
                )
                if reshape_match is not None:
                    channel_count = int(reshape_match.group("c"))
                    continue
            image_tail_match = image_tail_reshape_re.match(lines[lookahead])
            if image_tail_match is not None and channel_count is not None:
                h = image_tail_match.group("h0") or image_tail_match.group("h1")
                w = image_tail_match.group("w0") or image_tail_match.group("w1")
                c = image_tail_match.group("c0") or image_tail_match.group("c1")
                if int(c) != int(channel_count):
                    continue
                output_hwc_shape = [1, int(h), int(w), int(c)]
                break
        if output_hwc_shape is None:
            continue
        rewritten_add = (
            f"{indent}{lhs} = _align_tensor_to_target_shape("
            f"torch.add(*_align_binary_inputs_to_anchor({input_a}, {input_b}, {repr(output_hwc_shape)})), "
            f"{repr(output_hwc_shape)})"
        )
        if lines[index] != rewritten_add:
            lines[index] = rewritten_add
            dynamic_nhwc_like_names.add(lhs)
            context.static_shapes[lhs] = list(output_hwc_shape)
            changed = True
        for source_name in [input_a, input_b]:
            if _rewrite_upstream_binary_producer(
                source_name=source_name,
                max_index=index,
                output_hwc_shape=output_hwc_shape,
            ):
                changed = True
        tracked_names = {input_a, input_b}
        for back in range(max(0, index - 4), index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is not None
                and {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])} == tracked_names
            ):
                rewritten_anchor = (
                    f"{binary_anchor_assign[0]}{binary_anchor_assign[1]}, {binary_anchor_assign[2]} = "
                    f"_align_binary_inputs_to_anchor({binary_anchor_assign[3]}, {binary_anchor_assign[4]}, "
                    f"{repr(output_hwc_shape)})"
                )
                if lines[back] != rewritten_anchor:
                    lines[back] = rewritten_anchor
                    context.static_shapes[str(binary_anchor_assign[1])] = list(output_hwc_shape)
                    context.static_shapes[str(binary_anchor_assign[2])] = list(output_hwc_shape)
                    changed = True
                for source_name in [str(binary_anchor_assign[3]), str(binary_anchor_assign[4])]:
                    if _rewrite_upstream_binary_producer(
                        source_name=source_name,
                        max_index=back,
                        output_hwc_shape=output_hwc_shape,
                    ):
                        changed = True
                break
    return changed


def _rewrite_structural_channel_first_depth_to_space_public_bridges(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    assignments_by_lhs: Dict[str, List[int]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            assignments_by_lhs.setdefault(str(assign[1]), []).append(idx)
    channel_first_gather_slice_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\[:, \[(?P<indices>[0-9,\s-]+)\], :, :\]$"
    )
    channel_last_gather_slice_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\[:, :, :, \[(?P<indices>[0-9,\s-]+)\]\]$"
    )
    manual_depth_to_space_bridge_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*"
        r"(?P<input>[A-Za-z0-9_]+)\.reshape\("
        r"(?P<n>[A-Za-z0-9_]+), (?P<h>[A-Za-z0-9_]+), (?P<w>[A-Za-z0-9_]+), "
        r"(?P<block_h>\d+), (?P<block_w>\d+), (?P<c>[A-Za-z0-9_]+) // (?P<div>\d+)\)"
        r"\.permute\(0, 1, 3, 2, 4, 5\)\.reshape\("
        r"(?P=n), (?P=h) \* (?P=block_h), (?P=w) \* (?P=block_w), (?P=c) // (?P=div)\)$"
    )

    def _find_assignment_before(name: str, upper_bound: int) -> int | None:
        for candidate_index in reversed(assignments_by_lhs.get(str(name), [])):
            if candidate_index < upper_bound:
                return candidate_index
        return None

    def _is_channel_first_depth_to_space_source(name: str, upper_bound: int) -> bool:
        current_name = str(name)
        current_upper_bound = int(upper_bound)
        visited: Set[str] = set()
        for _ in range(12):
            if current_name in visited:
                return False
            visited.add(current_name)
            if _fast_precanonicalize_is_cf_like(
                current_name,
                dynamic_cf_like_names,
                context,
            ):
                return True
            assign_index = _find_assignment_before(current_name, current_upper_bound)
            if assign_index is None:
                return False
            src_line = str(lines[assign_index])
            cf_gather_match = channel_first_gather_slice_re.match(src_line)
            if (
                cf_gather_match is not None
                and str(cf_gather_match.group("lhs")) == current_name
            ):
                current_name = str(cf_gather_match.group("input"))
                current_upper_bound = assign_index
                continue
            cl_gather_match = channel_last_gather_slice_re.match(src_line)
            if (
                cl_gather_match is not None
                and str(cl_gather_match.group("lhs")) == current_name
            ):
                return False
            assign = _parse_simple_assignment_line(src_line)
            rhs_expr = str(assign[2]).strip() if assign is not None else None
            if (
                assign is not None
                and str(assign[1]) == current_name
                and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr or "") is not None
                and rhs_expr != current_name
            ):
                current_name = str(rhs_expr)
                current_upper_bound = assign_index
                continue
            return False
        return False

    def _find_upstream_channel_first_depth_to_space_reorder(
        name: str,
        upper_bound: int,
    ) -> Tuple[int, str, str, List[int], List[str]] | None:
        current_name = str(name)
        current_upper_bound = int(upper_bound)
        alias_chain: List[str] = []
        visited: Set[str] = set()
        for _ in range(12):
            if current_name in visited:
                return None
            visited.add(current_name)
            assign_index = _find_assignment_before(current_name, current_upper_bound)
            if assign_index is None:
                return None
            assign_line = str(lines[assign_index])
            cf_gather_match = channel_first_gather_slice_re.match(assign_line)
            if (
                cf_gather_match is not None
                and str(cf_gather_match.group("lhs")) == current_name
            ):
                indices = [
                    int(token.strip())
                    for token in str(cf_gather_match.group("indices")).split(",")
                    if token.strip() != ""
                ]
                return (
                    assign_index,
                    current_name,
                    str(cf_gather_match.group("input")),
                    indices,
                    alias_chain,
                )
            cl_gather_match = channel_last_gather_slice_re.match(assign_line)
            if (
                cl_gather_match is not None
                and str(cl_gather_match.group("lhs")) == current_name
            ):
                return None
            assign = _parse_simple_assignment_line(assign_line)
            rhs_expr = str(assign[2]).strip() if assign is not None else None
            if (
                assign is not None
                and str(assign[1]) == current_name
                and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr or "") is not None
                and rhs_expr != current_name
            ):
                alias_chain.append(current_name)
                current_name = str(rhs_expr)
                current_upper_bound = assign_index
                continue
            return None
        return None

    def _try_elide_channel_first_depth_to_space_reorder(
        name: str,
        upper_bound: int,
        divisor: int,
    ) -> str | None:
        reorder = _find_upstream_channel_first_depth_to_space_reorder(
            name,
            upper_bound,
        )
        if reorder is None:
            return None
        assign_index, reordered_name, source_name, indices, alias_chain = reorder
        if not _is_channel_first_depth_to_space_source(source_name, assign_index):
            return None
        source_shape = (
            context.static_shapes.get(source_name)
            or context.static_shapes.get(
                _fast_precanonicalize_resolve_alias(source_name, context.aliases)
            )
        )
        if len(indices) == 0:
            return None
        total_channels = int(len(indices))
        if total_channels % int(divisor) != 0:
            return None
        if sorted(indices) != list(range(total_channels)):
            return None
        base_channels = total_channels // int(divisor)
        expected_indices = [
            int(channel_index) * int(divisor) + int(offset_index)
            for offset_index in range(int(divisor))
            for channel_index in range(int(base_channels))
        ]
        if indices != expected_indices:
            return None
        normalized_source_shape: List[int] | None = None
        if source_shape is not None and len(source_shape) == 4:
            if int(source_shape[1]) == total_channels:
                normalized_source_shape = [int(v) for v in list(source_shape)]
            elif int(source_shape[3]) == total_channels:
                normalized_source_shape = [
                    int(source_shape[0]),
                    int(source_shape[3]),
                    int(source_shape[1]),
                    int(source_shape[2]),
                ]
        indent = re.match(r"^\s*", str(lines[assign_index])).group(0)
        lines[assign_index] = (
            f"{indent}{reordered_name} = {source_name}"
        )
        all_alias_names = [str(reordered_name), *[str(alias) for alias in alias_chain]]
        for alias_name in all_alias_names:
            context.aliases[str(alias_name)] = str(source_name)
            if normalized_source_shape is not None:
                context.static_shapes[str(alias_name)] = [
                    int(v) for v in list(normalized_source_shape)
                ]
            dynamic_cf_like_names.add(str(alias_name))
        return str(source_name)

    for index, line in enumerate(lines):
        bridge_match = manual_depth_to_space_bridge_re.match(str(line))
        if bridge_match is None:
            continue
        input_name = str(bridge_match.group("input"))
        if not _is_channel_first_depth_to_space_source(input_name, index):
            continue
        block_h = int(bridge_match.group("block_h"))
        block_w = int(bridge_match.group("block_w"))
        divisor = int(bridge_match.group("div"))
        if block_h <= 1 or block_h != block_w or divisor != block_h * block_w:
            continue
        elided_input_name = _try_elide_channel_first_depth_to_space_reorder(
            input_name,
            index,
            divisor,
        )
        if elided_input_name is not None:
            input_name = str(elided_input_name)
        else:
            input_name = str(
                context.aliases.get(
                    input_name,
                    _fast_precanonicalize_resolve_alias(input_name, context.aliases),
                )
            )
        lines[index] = (
            f"{bridge_match.group('indent')}{bridge_match.group('lhs')} = "
            f"_torch_permute(F.pixel_shuffle({input_name}, {block_h}), [0, 2, 3, 1])"
        )
        dynamic_nhwc_like_names.add(str(bridge_match.group("lhs")))
        resolved_input_name = _fast_precanonicalize_resolve_alias(
            input_name,
            context.aliases,
        )
        input_shape = (
            context.static_shapes.get(input_name)
            or context.static_shapes.get(resolved_input_name)
        )
        if input_shape is not None and len(input_shape) == 4:
            output_shape = [
                int(input_shape[0]),
                int(input_shape[2]) * block_h,
                int(input_shape[3]) * block_w,
                int(input_shape[1]) // divisor,
            ]
            context.static_shapes[str(bridge_match.group("lhs"))] = output_shape
        changed = True
    return changed


def _rewrite_structural_cf_anchor_binary_add_targets(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False

    def _resolve_preferred_cf_anchor_input(
        anchor_input_a: str,
        anchor_input_b: str,
        shape_hint: Sequence[int],
    ) -> Tuple[str, int] | None:
        candidates: List[Tuple[str, int]] = []
        for candidate_name in [anchor_input_a, anchor_input_b]:
            candidate_is_cf = _fast_precanonicalize_is_cf_like(
                candidate_name,
                dynamic_cf_like_names,
                context,
            )
            candidate_is_nhwc = _fast_precanonicalize_is_nhwc_like(
                candidate_name,
                dynamic_nhwc_like_names,
                context,
            )
            if not candidate_is_cf or candidate_is_nhwc:
                continue
            candidate_channel_count = _fast_precanonicalize_preferred_channel_count(
                candidate_name,
                dynamic_cf_like_names,
                dynamic_nhwc_like_names,
                context,
                shape_hint=shape_hint,
            )
            if candidate_channel_count is None:
                candidate_shape = context.static_shapes.get(candidate_name)
                if candidate_shape is not None and len(candidate_shape) == 4:
                    candidate_channel_count = int(candidate_shape[1])
            if candidate_channel_count is None:
                continue
            candidates.append((str(candidate_name), int(candidate_channel_count)))
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            for candidate_name, candidate_channel_count in candidates:
                if candidate_name.endswith("_cf") or candidate_name.endswith("_out_cf"):
                    return (candidate_name, int(candidate_channel_count))
            return candidates[0]
        return None

    def _cf_dynamic_target_shape_expr(cf_anchor_input: str, preferred_channel_count: int) -> str:
        return (
            "["
            f"int({cf_anchor_input}.shape[0]), "
            f"{int(preferred_channel_count)}, "
            f"int({cf_anchor_input}.shape[2]), "
            f"int({cf_anchor_input}.shape[3])"
            "]"
        )

    for index, line in enumerate(lines):
        parsed_add = _parse_static_binary_add_align_assign(line)
        if parsed_add is None:
            continue
        indent, lhs, input_a, input_b, current_shape = parsed_add
        if len(current_shape) != 4:
            continue
        matching_binary_anchor = None
        for back in range(max(0, index - 4), index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is None
                or {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])}
                != {input_a, input_b}
            ):
                continue
            matching_binary_anchor = (back, binary_anchor_assign)
            break
        if matching_binary_anchor is None:
            continue
        anchor_index, binary_anchor_assign = matching_binary_anchor
        cf_anchor = _resolve_preferred_cf_anchor_input(
            str(binary_anchor_assign[3]),
            str(binary_anchor_assign[4]),
            current_shape,
        )
        if cf_anchor is None:
            continue
        cf_anchor_input, preferred_channel_count = cf_anchor
        if int(current_shape[1]) == int(preferred_channel_count):
            continue
        dynamic_target_shape = _cf_dynamic_target_shape_expr(
            cf_anchor_input,
            int(preferred_channel_count),
        )
        rewritten_anchor = (
            f"{binary_anchor_assign[0]}{binary_anchor_assign[1]}, {binary_anchor_assign[2]} = "
            f"_align_binary_inputs_to_anchor({binary_anchor_assign[3]}, {binary_anchor_assign[4]}, "
            f"{dynamic_target_shape})"
        )
        if lines[anchor_index] != rewritten_anchor:
            lines[anchor_index] = rewritten_anchor
            changed = True
        rewritten_add = (
            f"{indent}{lhs} = _align_tensor_to_target_shape("
            f"torch.add({input_a}, {input_b}), {dynamic_target_shape})"
        )
        if lines[index] != rewritten_add:
            lines[index] = rewritten_add
            changed = True
        dynamic_cf_like_names.update(
            {
                lhs,
                str(binary_anchor_assign[1]),
                str(binary_anchor_assign[2]),
            }
        )
        dynamic_nhwc_like_names.difference_update(
            {
                lhs,
                str(binary_anchor_assign[1]),
                str(binary_anchor_assign[2]),
            }
        )
        cf_anchor_shape = context.static_shapes.get(cf_anchor_input)
        if cf_anchor_shape is not None and len(cf_anchor_shape) == 4:
            normalized_cf_shape = [int(v) for v in list(cf_anchor_shape)]
            context.static_shapes[lhs] = list(normalized_cf_shape)
            context.static_shapes[str(binary_anchor_assign[1])] = list(normalized_cf_shape)
            context.static_shapes[str(binary_anchor_assign[2])] = list(normalized_cf_shape)
    return changed


def _rewrite_structural_channel_first_rank4_flatten_to_nwc(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    dynamic_cf_shape_re = re.compile(
        r"[\[\(]\s*int\((?P<ref>[A-Za-z0-9_]+)\.shape\[0\]\)\s*,\s*(?P<c>\d+)\s*,\s*"
        r"int\((?P=ref)\.shape\[2\]\)\s*,\s*int\((?P=ref)\.shape\[3\]\)\s*[\]\)]"
    )

    def _parse_rank4_flatten_assign(
        line: str,
    ) -> Tuple[str, str, str, str, int, str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "torch.reshape("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        if len(parts) != 2:
            return None
        input_expr = parts[0].strip()
        shape_expr = parts[1].strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None:
            return None
        resolved_match = re.fullmatch(
            r"_resolve_reshape_shape\(\[(?P<n>-?\d+), -1, (?P<c>\d+)\], (?P<input>[A-Za-z0-9_]+), allow_zero=False\)",
            shape_expr,
        )
        if resolved_match is not None and str(resolved_match.group("input")) == input_expr:
            return (
                indent,
                lhs,
                input_expr,
                str(resolved_match.group("n")),
                int(resolved_match.group("c")),
                shape_expr,
                "nwc",
            )
        try:
            literal_shape_value = ast.literal_eval(shape_expr)
        except Exception:
            return None
        if not isinstance(literal_shape_value, (list, tuple)) or len(literal_shape_value) != 3:
            return None
        literal_shape = [int(v) for v in list(literal_shape_value)]
        if int(literal_shape[1]) <= 0 or int(literal_shape[2]) <= 0:
            return None
        return (
            indent,
            lhs,
            input_expr,
            str(int(literal_shape[0])),
            int(literal_shape[1]),
            shape_expr,
            "nchw_flatten",
        )

    def _resolve_static_rank4_shape(
        name: str,
        seen: Set[str] | None = None,
    ) -> Tuple[List[int], bool] | None:
        if seen is None:
            seen = set()
        if name in seen:
            return None
        seen.add(name)
        cached = context.static_shapes.get(name)
        if cached is not None and len(cached) == 4:
            return [int(v) for v in list(cached)], True
        producer_module = context.module_output_producers.get(name, None)
        if producer_module is not None:
            out_channels = context.conv_block_out_channels.get(producer_module, None)
            if out_channels is not None:
                return [1, int(out_channels), -1, -1], False
        for src_line in lines:
            assign = _parse_simple_assignment_line(src_line)
            if assign is None or str(assign[1]) != name:
                continue
            rhs_expr = str(assign[2]).strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr) is not None:
                aliased_shape = _resolve_static_rank4_shape(rhs_expr, seen=seen)
                if aliased_shape is not None:
                    return aliased_shape
            aligned_expr = _parse_align_tensor_target_shape_expr(str(assign[2]))
            if aligned_expr is None:
                continue
            shape_expr = str(aligned_expr[1]).strip()
            shape = _parse_rank4_shape_literal(shape_expr)
            if shape is not None:
                return [int(v) for v in list(shape)], True
            dynamic_cf_match = dynamic_cf_shape_re.fullmatch(shape_expr)
            if dynamic_cf_match is not None:
                ref_name = str(dynamic_cf_match.group("ref"))
                ref_shape = context.static_shapes.get(ref_name)
                if ref_shape is not None and len(ref_shape) == 4:
                    return (
                        [
                            int(ref_shape[0]),
                            int(dynamic_cf_match.group("c")),
                            int(ref_shape[2]),
                            int(ref_shape[3]),
                        ],
                        True,
                    )
                return [1, int(dynamic_cf_match.group("c")), -1, -1], False
        return None

    def _resolve_alias_source(name: str, seen: Set[str] | None = None) -> str:
        if seen is None:
            seen = set()
        if name in seen:
            return name
        seen.add(name)
        for src_line in lines:
            assign = _parse_simple_assignment_line(src_line)
            if assign is None or str(assign[1]) != name:
                continue
            rhs_expr = str(assign[2]).strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr) is not None:
                return _resolve_alias_source(rhs_expr, seen=seen)
        return name

    for index, line in enumerate(lines):
        parsed_flatten = _parse_rank4_flatten_assign(line)
        if parsed_flatten is None:
            continue
        indent, lhs_name, input_name, batch_dim_expr, channel_count, shape_expr, flatten_kind = parsed_flatten
        resolved_input_shape = _resolve_static_rank4_shape(input_name)
        if resolved_input_shape is None:
            continue
        input_shape, has_exact_spatial_shape = resolved_input_shape
        if int(input_shape[1]) != int(channel_count):
            continue
        input_is_cf = bool(int(input_shape[1]) == int(channel_count) and int(input_shape[3]) != int(channel_count))
        input_is_nhwc = bool(int(input_shape[3]) == int(channel_count))
        rewritten: str | None = None
        if flatten_kind == "nwc":
            if not input_is_cf or input_is_nhwc:
                continue
            rewritten = (
                f"{indent}{lhs_name} = torch.reshape("
                f"{input_name}.permute(0, 2, 3, 1).contiguous(), "
                f"{shape_expr})"
            )
        else:
            resolved_source_name = _resolve_alias_source(input_name)
            source_is_nhwc_like = (
                _fast_precanonicalize_is_nhwc_like(input_name, dynamic_nhwc_like_names, context)
                or _fast_precanonicalize_is_nhwc_like(resolved_source_name, dynamic_nhwc_like_names, context)
                or "_nhwc" in resolved_source_name
            )
            if not source_is_nhwc_like:
                continue
            rewritten = (
                f"{indent}{lhs_name} = torch.reshape("
                f"{input_name}.permute(0, 3, 1, 2).contiguous(), "
                f"{shape_expr})"
            )
        if lines[index] != rewritten:
            lines[index] = rewritten
            changed = True
        if (
            flatten_kind == "nwc"
            and has_exact_spatial_shape
            and int(input_shape[2]) > 0
            and int(input_shape[3]) > 0
        ):
            batch_dim = int(input_shape[0]) if len(input_shape) > 0 else 1
            height = int(input_shape[2])
            width = int(input_shape[3])
            context.static_shapes[str(lhs_name)] = [batch_dim, height * width, channel_count]
        if flatten_kind == "nwc":
            dynamic_nhwc_like_names.add(str(lhs_name))
            dynamic_cf_like_names.discard(str(lhs_name))
        else:
            dynamic_cf_like_names.add(str(lhs_name))
            dynamic_nhwc_like_names.discard(str(lhs_name))
    return changed


def _rewrite_structural_direct_conv_cf_add_targets(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    helper_def_re = re.compile(
        r"^(?P<indent>\s*)def (?P<name>[A-Za-z0-9_]+)\("
    )
    helper_direct_call_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<helper>[A-Za-z0-9_]+)\(.+\)$"
    )
    helper_unpack_call_re = re.compile(
        r"^(?P<indent>\s*)\(?\s*(?P<lhses>[A-Za-z0-9_, ]+)\s*\)?\s*=\s*self\.(?P<helper>[A-Za-z0-9_]+)\(.+\)$"
    )
    relu_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    direct_conv_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    def _is_cf_like(name: str) -> bool:
        return _fast_precanonicalize_is_cf_like(
            name,
            dynamic_cf_like_names,
            context,
        )

    def _is_nhwc_like(name: str) -> bool:
        return _fast_precanonicalize_is_nhwc_like(
            name,
            dynamic_nhwc_like_names,
            context,
        )

    def _has_mixed_layout_add_evidence(
        add_index: int,
        input_a: str,
        input_b: str,
    ) -> bool:
        if (_is_cf_like(input_a) and _is_nhwc_like(input_b)) or (_is_cf_like(input_b) and _is_nhwc_like(input_a)):
            return True
        for back in range(max(0, add_index - 4), add_index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is None
                or {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])}
                != {input_a, input_b}
            ):
                continue
            anchor_input_a = str(binary_anchor_assign[3])
            anchor_input_b = str(binary_anchor_assign[4])
            if (_is_cf_like(anchor_input_a) and _is_nhwc_like(anchor_input_b)) or (
                _is_cf_like(anchor_input_b) and _is_nhwc_like(anchor_input_a)
            ):
                return True
        return False

    def _find_enclosing_helper(index: int) -> Tuple[str, int, int] | None:
        def_index: int | None = None
        def_name: str | None = None
        def_indent = 0
        for candidate_index in range(index, -1, -1):
            helper_def_match = helper_def_re.match(lines[candidate_index])
            if helper_def_match is None:
                continue
            def_index = candidate_index
            def_name = str(helper_def_match.group("name"))
            def_indent = len(str(helper_def_match.group("indent")))
            break
        if def_index is None or def_name is None:
            return None
        end_index = len(lines)
        for candidate_index in range(def_index + 1, len(lines)):
            next_def_match = helper_def_re.match(lines[candidate_index])
            if next_def_match is None:
                continue
            next_indent = len(str(next_def_match.group("indent")))
            if next_indent <= def_indent:
                end_index = candidate_index
                break
        return def_name, def_index, end_index

    def _find_returned_output_indexes(
        helper_name: str,
        helper_index: int,
        helper_end_index: int,
        alias_names: Set[str],
    ) -> List[int]:
        returned_indexes: List[int] = []
        for candidate_index in range(helper_index + 1, helper_end_index):
            stripped = str(lines[candidate_index]).strip()
            if not stripped.startswith("return "):
                continue
            return_expr = stripped[len("return ") :].strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", return_expr) is not None:
                if return_expr in alias_names:
                    return [0]
                continue
            parts = _split_top_level_csv_exprs(_strip_outer_parentheses(return_expr))
            for part_index, part in enumerate(parts):
                if part.strip() in alias_names:
                    returned_indexes.append(part_index)
            if returned_indexes:
                return returned_indexes
        return []

    def _resolve_stage_boundary_consumer(
        add_index: int,
        alias_names: Set[str],
        preferred_channel_count: int,
    ) -> str | None:
        enclosing_helper = _find_enclosing_helper(add_index)
        if enclosing_helper is None:
            return None
        helper_name, helper_index, helper_end_index = enclosing_helper
        returned_output_indexes = _find_returned_output_indexes(
            helper_name,
            helper_index,
            helper_end_index,
            alias_names,
        )
        if not returned_output_indexes:
            return None
        for candidate_index in range(helper_end_index, len(lines)):
            direct_call_match = helper_direct_call_re.match(lines[candidate_index])
            if (
                direct_call_match is not None
                and str(direct_call_match.group("helper")) == helper_name
                and 0 in returned_output_indexes
            ):
                caller_output = str(direct_call_match.group("lhs"))
                for candidate_module in context.module_input_consumers.get(caller_output, []):
                    candidate_channels = context.conv_block_in_channels.get(candidate_module, None)
                    if candidate_channels is not None and int(candidate_channels) == int(preferred_channel_count):
                        return str(candidate_module)
            unpack_call_match = helper_unpack_call_re.match(lines[candidate_index])
            if unpack_call_match is None or str(unpack_call_match.group("helper")) != helper_name:
                continue
            lhs_values = [value.strip() for value in str(unpack_call_match.group("lhses")).split(",") if value.strip()]
            for returned_index in returned_output_indexes:
                if returned_index >= len(lhs_values):
                    continue
                caller_output = lhs_values[returned_index]
                for candidate_module in context.module_input_consumers.get(caller_output, []):
                    candidate_channels = context.conv_block_in_channels.get(candidate_module, None)
                    if candidate_channels is not None and int(candidate_channels) == int(preferred_channel_count):
                        return str(candidate_module)
        return None

    def _find_matching_binary_anchor(
        add_index: int,
        input_a: str,
        input_b: str,
    ) -> Tuple[str, str, str, str, List[int]] | None:
        for back in range(max(0, add_index - 4), add_index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is None
                or {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])}
                != {input_a, input_b}
            ):
                continue
            return (
                str(binary_anchor_assign[0]),
                str(binary_anchor_assign[3]),
                str(binary_anchor_assign[4]),
                str(back),
                [int(v) for v in list(binary_anchor_assign[5])],
            )
        return None

    def _resolve_preferred_cf_anchor_input(
        anchor_input_a: str,
        anchor_input_b: str,
        preferred_channel_count: int,
    ) -> str | None:
        candidates: List[str] = []
        for candidate_name in [anchor_input_a, anchor_input_b]:
            candidate_is_cf = _fast_precanonicalize_is_cf_like(
                candidate_name,
                dynamic_cf_like_names,
                context,
            )
            candidate_is_nhwc = _fast_precanonicalize_is_nhwc_like(
                candidate_name,
                dynamic_nhwc_like_names,
                context,
            )
            if not candidate_is_cf or candidate_is_nhwc:
                continue
            candidate_channel_count = _fast_precanonicalize_preferred_channel_count(
                candidate_name,
                dynamic_cf_like_names,
                dynamic_nhwc_like_names,
                context,
            )
            if (
                candidate_channel_count is not None
                and int(candidate_channel_count) != int(preferred_channel_count)
            ):
                continue
            candidates.append(str(candidate_name))
        if len(candidates) == 1:
            return str(candidates[0])
        if len(candidates) > 1:
            for candidate_name in candidates:
                if candidate_name.endswith("_cf") or candidate_name.endswith("_out_cf"):
                    return str(candidate_name)
            return str(candidates[0])
        return None

    def _cf_dynamic_target_shape_expr(cf_anchor_input: str, preferred_channel_count: int) -> str:
        return (
            "["
            f"int({cf_anchor_input}.shape[0]), "
            f"{int(preferred_channel_count)}, "
            f"int({cf_anchor_input}.shape[2]), "
            f"int({cf_anchor_input}.shape[3])"
            "]"
        )

    for index, line in enumerate(lines):
        parsed_add = _parse_static_binary_add_align_assign(line)
        if parsed_add is None:
            continue
        indent, lhs, input_a, input_b, current_shape = parsed_add
        if len(current_shape) != 4:
            continue
        preferred_channel_count = _infer_structural_rank4_channel_count(
            lhs,
            current_shape,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        )
        if preferred_channel_count is None:
            continue
        if int(current_shape[1]) == int(preferred_channel_count):
            continue
        normalized_shape = _normalize_cf_rank4_shape(
            current_shape,
            preferred_channel_count=int(preferred_channel_count),
        )
        if normalized_shape == current_shape:
            continue
        alias_names: Set[str] = {lhs}
        consumer_module: str | None = None
        for lookahead in range(index + 1, min(len(lines), index + 6)):
            direct_conv_match = direct_conv_assign_re.match(lines[lookahead])
            if (
                direct_conv_match is not None
                and str(direct_conv_match.group("input")) in alias_names
            ):
                consumer_module = str(direct_conv_match.group("module"))
                break
            relu_match = relu_assign_re.match(lines[lookahead])
            if (
                relu_match is not None
                and str(relu_match.group("input")) in alias_names
            ):
                alias_names.add(str(relu_match.group("lhs")))
                continue
            alias_assign = _parse_simple_assignment_line(lines[lookahead])
            if alias_assign is None:
                continue
            rhs_expr = str(alias_assign[2]).strip()
            if (
                re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr) is not None
                and rhs_expr in alias_names
            ):
                alias_names.add(str(alias_assign[1]))
        if consumer_module is None:
            for alias_name in alias_names:
                for candidate_module in context.module_input_consumers.get(alias_name, []):
                    candidate_channels = context.conv_block_in_channels.get(candidate_module, None)
                    if candidate_channels is None or int(candidate_channels) != int(preferred_channel_count):
                        continue
                    consumer_module = str(candidate_module)
                    break
                if consumer_module is not None:
                    break
        if consumer_module is None:
            consumer_module = _resolve_stage_boundary_consumer(
                index,
                alias_names,
                int(preferred_channel_count),
            )
        if consumer_module is None:
            continue
        consumer_channels = context.conv_block_in_channels.get(consumer_module, None)
        if consumer_channels is None or int(consumer_channels) != int(preferred_channel_count):
            continue
        matching_binary_anchor = _find_matching_binary_anchor(index, input_a, input_b)
        cf_anchor_input: str | None = None
        if matching_binary_anchor is not None:
            cf_anchor_input = _resolve_preferred_cf_anchor_input(
                matching_binary_anchor[1],
                matching_binary_anchor[2],
                int(preferred_channel_count),
            )
        has_mixed_layout_evidence = _has_mixed_layout_add_evidence(index, input_a, input_b)
        if not has_mixed_layout_evidence and cf_anchor_input is None:
            continue
        rewritten_target_shape = repr(normalized_shape)
        if cf_anchor_input is not None:
            rewritten_target_shape = _cf_dynamic_target_shape_expr(
                cf_anchor_input,
                int(preferred_channel_count),
            )
        lines[index] = (
            f"{indent}{lhs} = _align_tensor_to_target_shape("
            f"torch.add({input_a}, {input_b}), {rewritten_target_shape})"
        )
        dynamic_cf_like_names.update(alias_names)
        dynamic_nhwc_like_names.discard(lhs)
        if cf_anchor_input is not None:
            cf_anchor_shape = context.static_shapes.get(cf_anchor_input)
            if cf_anchor_shape is not None and len(cf_anchor_shape) == 4:
                context.static_shapes[lhs] = [int(v) for v in list(cf_anchor_shape)]
        else:
            context.static_shapes[lhs] = list(normalized_shape)
        changed = True
        if matching_binary_anchor is not None:
            anchor_indent, anchor_input_a, anchor_input_b, anchor_back, _ = matching_binary_anchor
            anchor_back_index = int(anchor_back)
            anchor_lhs0 = input_a
            anchor_lhs1 = input_b
            if cf_anchor_input is not None:
                lines[anchor_back_index] = (
                    f"{anchor_indent}{anchor_lhs0}, {anchor_lhs1} = "
                    f"_align_binary_inputs_to_anchor({anchor_input_a}, {anchor_input_b}, "
                    f"{_cf_dynamic_target_shape_expr(cf_anchor_input, int(preferred_channel_count))})"
                )
                cf_anchor_shape = context.static_shapes.get(cf_anchor_input)
                if cf_anchor_shape is not None and len(cf_anchor_shape) == 4:
                    context.static_shapes[str(anchor_lhs0)] = [int(v) for v in list(cf_anchor_shape)]
                    context.static_shapes[str(anchor_lhs1)] = [int(v) for v in list(cf_anchor_shape)]
            else:
                lines[anchor_back_index] = (
                    f"{anchor_indent}{anchor_lhs0}, {anchor_lhs1} = "
                    f"_align_binary_inputs_to_anchor({anchor_input_a}, {anchor_input_b}, "
                    f"{repr(normalized_shape)})"
                )
                context.static_shapes[str(anchor_lhs0)] = list(normalized_shape)
                context.static_shapes[str(anchor_lhs1)] = list(normalized_shape)
            dynamic_cf_like_names.update({str(anchor_lhs0), str(anchor_lhs1)})
    return changed


def _rewrite_structural_redundant_nhwc_to_cf_conv_bridges(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    dynamic_cf_shape_re = re.compile(
        r"[\[\(]\s*int\((?P<ref>[A-Za-z0-9_]+)\.shape\[0\]\)\s*,\s*(?P<c>\d+)\s*,\s*"
        r"int\((?P=ref)\.shape\[2\]\)\s*,\s*int\((?P=ref)\.shape\[3\]\)\s*[\]\)]"
    )
    permuted_conv_input_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\("
        r"(?P<input>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\)\)$"
    )
    cf_channel_gather_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\[:, \[(?P<indices>[0-9,\s-]+)\], :, :\]$"
    )

    def _parse_rank4_gap_mean_assign(
        line: str,
    ) -> Tuple[str, str, str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        stripped = rhs.strip()
        prefix = "torch.mean("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                dim_expr = part.strip()
            elif positional_index == 2:
                keepdim_expr = part.strip()
            positional_index += 1
        if (
            input_expr is None
            or dim_expr is None
            or keepdim_expr not in {None, "True"}
            or re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
        ):
            return None
        try:
            dim_value = ast.literal_eval(dim_expr)
        except Exception:
            return None
        if not isinstance(dim_value, (list, tuple)):
            return None
        dim_list = [int(v) for v in list(dim_value)]
        if dim_list not in ([1, 2], [2, 3]):
            return None
        return indent, lhs, input_expr, dim_list

    def _has_cf_rank4_target_shape(name: str, expected_channels: int | None) -> bool:
        cached_shape = context.static_shapes.get(name, None)
        if cached_shape is not None and len(cached_shape) == 4:
            return expected_channels is None or int(cached_shape[1]) == int(expected_channels)
        for candidate_line in lines:
            assign = _parse_simple_assignment_line(candidate_line)
            if assign is None or str(assign[1]) != name:
                continue
            aligned_expr = _parse_align_tensor_target_shape_expr(str(assign[2]))
            if aligned_expr is None:
                continue
            shape_expr = str(aligned_expr[1]).strip()
            static_shape = _parse_rank4_shape_literal(shape_expr)
            if static_shape is not None:
                return expected_channels is None or int(static_shape[1]) == int(expected_channels)
            dynamic_cf_match = dynamic_cf_shape_re.fullmatch(shape_expr)
            if dynamic_cf_match is not None:
                return (
                    expected_channels is None
                    or int(dynamic_cf_match.group("c")) == int(expected_channels)
                )
        return False

    def _parse_binary_align_with_shape(
        line: str,
    ) -> Tuple[str, str, List[int]] | None:
        parsed_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(line)
        if parsed_anchor is not None:
            return str(parsed_anchor[3]), str(parsed_anchor[4]), [int(v) for v in list(parsed_anchor[5])]
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        stripped = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs("
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        if len(parts) != 3:
            return None
        if (
            re.fullmatch(r"[A-Za-z0-9_]+", parts[0].strip()) is None
            or re.fullmatch(r"[A-Za-z0-9_]+", parts[1].strip()) is None
        ):
            return None
        target_shape = _parse_rank4_shape_literal(parts[2].strip())
        if target_shape is None:
            return None
        return parts[0].strip(), parts[1].strip(), [int(v) for v in list(target_shape)]

    def _parse_relaxed_align_binary_inputs_to_anchor_assign_with_shape(
        line: str,
    ) -> Tuple[str, str, str, str, str, List[int]] | None:
        assign_match = re.match(
            r"^(?P<indent>\s*)\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?::\s*torch\.Tensor)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        if len(parts) != 3:
            return None
        target_shape = _parse_rank4_shape_literal(parts[2].strip())
        if target_shape is None:
            return None
        return (
            str(assign_match.group("indent")),
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            parts[0].strip(),
            parts[1].strip(),
            [int(v) for v in list(target_shape)],
        )

    def _parse_binary_input_pair(expr: str) -> Tuple[str, str, str] | None:
        stripped = str(expr).strip()
        for op_prefix in ("torch.mul", "torch.add"):
            prefix = f"{op_prefix}("
            if not stripped.startswith(prefix) or not stripped.endswith(")"):
                continue
            parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
            input_expr: str | None = None
            other_expr: str | None = None
            positional_index = 0
            for part in parts:
                keyword_match = re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part)
                if keyword_match is not None:
                    key, value = part.split("=", 1)
                    if key.strip() == "input":
                        input_expr = value.strip()
                    elif key.strip() == "other":
                        other_expr = value.strip()
                    continue
                if positional_index == 0:
                    input_expr = part.strip()
                elif positional_index == 1:
                    other_expr = part.strip()
                positional_index += 1
            if input_expr is None or other_expr is None:
                return None
            return op_prefix, input_expr, other_expr
        return None

    def _is_simple_tensor_name(expr: str) -> bool:
        return re.fullmatch(r"[A-Za-z0-9_]+", str(expr).strip()) is not None

    def _resolve_local_response_norm_source(name: str) -> str | None:
        for candidate_line in lines:
            assign = _parse_simple_assignment_line(candidate_line)
            if assign is None or str(assign[1]) != name:
                continue
            resolved_input = _parse_local_response_norm_input_expr(str(assign[2]).strip())
            if resolved_input is not None:
                return str(resolved_input)
        return None

    def _resolve_layout_preserving_unary_source(name: str, upper_bound: int) -> str | None:
        unary_patterns = [
            re.compile(r"^self\.prelu_[0-9]+\((?P<input>[A-Za-z0-9_]+)\)$"),
            re.compile(r"^torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"),
            re.compile(r"^torch\.sigmoid\((?P<input>[A-Za-z0-9_]+)\)$"),
            re.compile(r"^torch\.tanh\((?P<input>[A-Za-z0-9_]+)\)$"),
        ]
        for back in range(max(0, upper_bound - 16), upper_bound):
            assign = _parse_simple_assignment_line(lines[back])
            if assign is None or str(assign[1]) != name:
                continue
            rhs_expr = str(assign[2]).strip()
            for unary_pattern in unary_patterns:
                unary_match = unary_pattern.fullmatch(rhs_expr)
                if unary_match is not None:
                    return str(unary_match.group("input"))
        return None

    def _resolve_channel_first_pool_source(name: str, upper_bound: int) -> str | None:
        for back in range(max(0, upper_bound - 16), upper_bound):
            pool_assign = _parse_apply_pool2d_assign_with_shape(lines[back])
            if pool_assign is None or str(pool_assign[1]) != name or bool(pool_assign[6]):
                continue
            return str(pool_assign[2])
        return None

    def _resolve_binary_anchor_source(name: str, upper_bound: int) -> str | None:
        for back in range(max(0, upper_bound - 16), upper_bound):
            anchor_assign = _parse_relaxed_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if anchor_assign is None or name not in {str(anchor_assign[1]), str(anchor_assign[2])}:
                continue
            for candidate_expr in (str(anchor_assign[3]), str(anchor_assign[4])):
                if _is_simple_tensor_name(candidate_expr):
                    return candidate_expr
        return None

    def _resolve_layout_preserving_binary_sources(
        name: str,
        upper_bound: int,
    ) -> Tuple[str, str] | None:
        for back in range(max(0, upper_bound - 16), upper_bound):
            assign = _parse_simple_assignment_line(lines[back])
            if assign is None or str(assign[1]) != name:
                continue
            rhs_expr = str(assign[2]).strip()
            aligned_expr = _parse_align_tensor_target_shape_expr(rhs_expr)
            if aligned_expr is not None:
                rhs_expr = str(aligned_expr[0]).strip()
            parsed_binary_pair = _parse_binary_input_pair(rhs_expr)
            if parsed_binary_pair is None:
                continue
            return str(parsed_binary_pair[1]), str(parsed_binary_pair[2])
        return None

    def _rewrite_channel_first_pool_binary_chain_targets(
        name: str,
        expected_channels: int | None,
        *,
        start_index: int,
    ) -> bool:
        if expected_channels is None:
            return False
        assign_index: int | None = None
        assign_expr: str | None = None
        output_shape: List[int] | None = None
        for back in range(max(0, start_index - 16), start_index):
            assign = _parse_simple_assignment_line(lines[back])
            if assign is None or str(assign[1]) != name:
                continue
            aligned_expr = _parse_align_tensor_target_shape_expr(str(assign[2]).strip())
            if aligned_expr is None:
                continue
            parsed_shape = _parse_rank4_shape_literal(str(aligned_expr[1]).strip())
            if parsed_shape is None:
                continue
            assign_index = back
            assign_expr = str(aligned_expr[0]).strip()
            output_shape = [int(v) for v in list(parsed_shape)]
            break
        if assign_index is None or assign_expr is None or output_shape is None:
            return False
        parsed_binary_pair = _parse_binary_input_pair(assign_expr)
        if parsed_binary_pair is None:
            return False
        _, binary_input0, binary_input1 = parsed_binary_pair
        if int(output_shape[3]) != int(expected_channels):
            return False
        anchor_index: int | None = None
        anchor_assign: Tuple[str, str, str, str, str, List[int]] | None = None
        binary_inputs = {str(binary_input0), str(binary_input1)}
        for back in range(max(0, assign_index - 8), assign_index):
            parsed_anchor = _parse_relaxed_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if parsed_anchor is None:
                continue
            if {str(parsed_anchor[1]), str(parsed_anchor[2])} != binary_inputs:
                continue
            anchor_index = back
            anchor_assign = parsed_anchor
            break
        if anchor_index is None or anchor_assign is None:
            return False
        pool_index: int | None = None
        pool_assign: Tuple[str, str, str, str, List[int], bool, bool] | None = None
        for candidate_name in (str(anchor_assign[3]), str(anchor_assign[4])):
            if not _is_simple_tensor_name(candidate_name):
                continue
            for back in range(max(0, anchor_index - 8), anchor_index):
                parsed_pool = _parse_apply_pool2d_assign_with_shape(lines[back])
                if (
                    parsed_pool is None
                    or str(parsed_pool[1]) != candidate_name
                    or bool(parsed_pool[6])
                ):
                    continue
                if not _is_structurally_cf_rank4_source(
                    str(parsed_pool[2]),
                    expected_channels,
                    start_index=back,
                    visited={name},
                ):
                    continue
                pool_index = back
                pool_assign = parsed_pool
                break
            if pool_index is not None:
                break
        if pool_index is None or pool_assign is None:
            return False
        cf_shape = [int(output_shape[0]), int(expected_channels), int(output_shape[1]), int(output_shape[2])]
        lines[pool_index] = (
            f"{pool_assign[0]}{pool_assign[1]} = _apply_pool2d("
            f"{pool_assign[2]}, {pool_assign[3]}, target_shape={repr(cf_shape)}, "
            f"is_max_pool={repr(bool(pool_assign[5]))}, channel_last=False)"
        )
        lines[anchor_index] = (
            f"{anchor_assign[0]}{anchor_assign[1]}, {anchor_assign[2]} = "
            f"_align_binary_inputs_to_anchor({anchor_assign[3]}, {anchor_assign[4]}, {repr(cf_shape)})"
        )
        lines[assign_index] = (
            f"{_parse_simple_assignment_line(lines[assign_index])[0]}{name} = "
            f"_align_tensor_to_target_shape({assign_expr}, {repr(cf_shape)})"
        )
        context.static_shapes[str(pool_assign[1])] = list(cf_shape)
        context.static_shapes[str(anchor_assign[1])] = list(cf_shape)
        context.static_shapes[str(anchor_assign[2])] = list(cf_shape)
        context.static_shapes[str(name)] = list(cf_shape)
        dynamic_cf_like_names.update(
            {
                str(pool_assign[1]),
                str(anchor_assign[1]),
                str(anchor_assign[2]),
                str(name),
            }
        )
        dynamic_nhwc_like_names.discard(str(pool_assign[1]))
        dynamic_nhwc_like_names.discard(str(anchor_assign[1]))
        dynamic_nhwc_like_names.discard(str(anchor_assign[2]))
        dynamic_nhwc_like_names.discard(str(name))
        return True

    def _has_cf_binary_consumer(
        name: str,
        expected_channels: int | None,
        *,
        start_index: int,
    ) -> bool:
        if expected_channels is None:
            return False
        for lookahead in range(start_index + 1, min(len(lines), start_index + 8)):
            parsed_binary_align = _parse_binary_align_with_shape(lines[lookahead])
            if parsed_binary_align is None:
                continue
            input_a, input_b, target_shape = parsed_binary_align
            if name not in {input_a, input_b}:
                continue
            if len(target_shape) == 4 and int(target_shape[1]) == int(expected_channels):
                return True
        return False

    def _is_structurally_cf_rank4_source(
        name: str,
        expected_channels: int | None,
        *,
        start_index: int,
        visited: Set[str] | None = None,
    ) -> bool:
        if visited is None:
            visited = set()
        if name in visited:
            return False
        visited.add(name)
        has_cf_binary_consumer = _has_cf_binary_consumer(
            name,
            expected_channels,
            start_index=start_index,
        )
        for back in range(max(0, start_index - 32), start_index):
            gather_assign = cf_channel_gather_re.match(lines[back])
            if gather_assign is None or str(gather_assign.group("lhs")) != name:
                continue
            gathered_indices = [
                int(token.strip())
                for token in str(gather_assign.group("indices")).split(",")
                if token.strip() != ""
            ]
            if expected_channels is not None and int(len(gathered_indices)) != int(expected_channels):
                continue
            if _is_structurally_cf_rank4_source(
                str(gather_assign.group("input")),
                None,
                start_index=back,
                visited=visited,
            ):
                return True
        for back in range(max(0, start_index - 32), start_index):
            assign = _parse_simple_assignment_line(lines[back])
            if assign is None or str(assign[1]) != name:
                continue
            cat_args = _parse_torch_cat_inputs_and_dim(str(assign[2]))
            if cat_args is None or int(cat_args[1]) != 1:
                continue
            cat_inputs = [input_name.strip() for input_name in cat_args[0] if input_name.strip()]
            if cat_inputs and all(
                _is_structurally_cf_rank4_source(
                    input_name,
                    None,
                    start_index=back,
                    visited=visited,
                )
                for input_name in cat_inputs
            ):
                return True
        pool_source = _resolve_channel_first_pool_source(name, start_index)
        if pool_source is not None and pool_source != name:
            return _is_structurally_cf_rank4_source(
                pool_source,
                expected_channels,
                start_index=start_index,
                visited=visited,
            )
        anchor_source = _resolve_binary_anchor_source(name, start_index)
        if anchor_source is not None and anchor_source != name:
            return _is_structurally_cf_rank4_source(
                anchor_source,
                expected_channels,
                start_index=start_index,
                visited=visited,
            )
        binary_sources = _resolve_layout_preserving_binary_sources(name, start_index)
        if binary_sources is not None:
            lhs_source, rhs_source = binary_sources
            lhs_cf = (
                _is_simple_tensor_name(lhs_source)
                and _is_structurally_cf_rank4_source(
                    lhs_source,
                    expected_channels,
                    start_index=start_index,
                    visited=set(visited),
                )
            )
            rhs_cf = (
                _is_simple_tensor_name(rhs_source)
                and _is_structurally_cf_rank4_source(
                    rhs_source,
                    expected_channels,
                    start_index=start_index,
                    visited=set(visited),
                )
            )
            if lhs_cf and rhs_cf:
                return True
        if (
            _fast_precanonicalize_is_nhwc_like(name, dynamic_nhwc_like_names, context)
            and not has_cf_binary_consumer
        ):
            return False
        static_shape = context.static_shapes.get(name, None)
        if static_shape is not None and len(static_shape) == 4:
            if expected_channels is not None and int(static_shape[1]) != int(expected_channels):
                return False
            if expected_channels is None or int(static_shape[1]) == int(expected_channels):
                return True
        if _has_cf_rank4_target_shape(name, expected_channels):
            return True
        lrn_source = _resolve_local_response_norm_source(name)
        if lrn_source is not None and lrn_source != name:
            return _has_cf_rank4_target_shape(lrn_source, expected_channels)
        unary_source = _resolve_layout_preserving_unary_source(name, start_index)
        if unary_source is not None and unary_source != name:
            return _is_structurally_cf_rank4_source(
                unary_source,
                expected_channels,
                start_index=start_index,
                visited=visited,
            )
        if not _fast_precanonicalize_is_cf_like(name, dynamic_cf_like_names, context):
            return has_cf_binary_consumer
        preferred_channels = _fast_precanonicalize_preferred_channel_count(
            name,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        )
        if expected_channels is None or preferred_channels == int(expected_channels):
            return True
        return has_cf_binary_consumer

    def _has_explicit_cf_rank4_evidence(
        name: str,
        expected_channels: int | None,
        *,
        start_index: int,
    ) -> bool:
        static_shape = context.static_shapes.get(name, None)
        if static_shape is not None and len(static_shape) == 4:
            if expected_channels is None or int(static_shape[1]) == int(expected_channels):
                return True
        if _has_cf_rank4_target_shape(name, expected_channels):
            return True
        lrn_source = _resolve_local_response_norm_source(name)
        if lrn_source is not None and lrn_source != name:
            return _has_explicit_cf_rank4_evidence(
                lrn_source,
                expected_channels,
                start_index=start_index,
            )
        return _is_structurally_cf_rank4_source(
            name,
            expected_channels,
            start_index=start_index,
        )

    for index, line in enumerate(lines):
        permuted_conv_match = permuted_conv_input_re.match(line)
        if permuted_conv_match is None:
            continue
        input_name = str(permuted_conv_match.group("input"))
        conv_module = str(permuted_conv_match.group("module"))
        conv_in_channels = context.conv_block_in_channels.get(conv_module, None)
        mean_assign_index: int | None = None
        parsed_gap_mean: Tuple[str, str, str, List[int]] | None = None
        _rewrite_channel_first_pool_binary_chain_targets(
            input_name,
            conv_in_channels,
            start_index=index,
        )
        for back in range(max(0, index - 4), index):
            candidate_gap_mean = _parse_rank4_gap_mean_assign(lines[back])
            if candidate_gap_mean is None or str(candidate_gap_mean[1]) != input_name:
                continue
            mean_assign_index = back
            parsed_gap_mean = candidate_gap_mean
            break
        if (
            parsed_gap_mean is not None
            and parsed_gap_mean[3] == [1, 2]
            and _is_structurally_cf_rank4_source(
                str(parsed_gap_mean[2]),
                conv_in_channels,
                start_index=index,
            )
        ):
            lines[mean_assign_index] = (
                f"{parsed_gap_mean[0]}{parsed_gap_mean[1]} = "
                f"torch.mean({parsed_gap_mean[2]}, dim=[2, 3], keepdim=True)"
            )
            lines[index] = (
                f"{permuted_conv_match.group('indent')}{permuted_conv_match.group('lhs')} = "
                f"self.{conv_module}({input_name})"
            )
            if conv_in_channels is not None:
                context.static_shapes[input_name] = [1, int(conv_in_channels), 1, 1]
            dynamic_cf_like_names.add(str(input_name))
            dynamic_cf_like_names.add(str(permuted_conv_match.group("lhs")))
            dynamic_nhwc_like_names.discard(str(input_name))
            changed = True
            continue
        input_has_explicit_cf_evidence = _has_explicit_cf_rank4_evidence(
            input_name,
            conv_in_channels,
            start_index=index,
        )
        if not input_has_explicit_cf_evidence:
            continue
        if (
            _fast_precanonicalize_is_nhwc_like(input_name, dynamic_nhwc_like_names, context)
            and not input_has_explicit_cf_evidence
        ):
            continue
        input_shape = context.static_shapes.get(input_name, None)
        if (
            conv_in_channels is not None
            and input_shape is not None
            and len(input_shape) == 4
            and int(input_shape[1]) != int(conv_in_channels)
        ):
            continue
        lines[index] = (
            f"{permuted_conv_match.group('indent')}{permuted_conv_match.group('lhs')} = "
            f"self.{conv_module}({input_name})"
        )
        dynamic_cf_like_names.add(str(input_name))
        dynamic_cf_like_names.add(str(permuted_conv_match.group("lhs")))
        changed = True
    return changed


def _rewrite_structural_concat_conv_cf_branch_targets(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    direct_conv_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    relu_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    def _parse_resize_assign(
        line: str,
    ) -> Tuple[str, str, str, Tuple[int, int], List[int], str, bool, bool, bool] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        parsed = _parse_apply_resize_input_size_shape_and_channel_last(rhs)
        if parsed is None:
            return None
        input_name, size_value, shape_value, channel_last = parsed
        if size_value is None or shape_value is None:
            return None
        stripped = rhs.strip()
        if not stripped.startswith("_apply_resize(") or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len("_apply_resize(") : -1])
        method_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "method":
                method_expr = value
            elif key == "align_corners":
                align_expr = value
            elif key == "half_pixel_centers":
                hpc_expr = value
        if (
            method_expr is None
            or align_expr not in {"True", "False"}
            or hpc_expr not in {"True", "False"}
            or not (method_expr.startswith("'") and method_expr.endswith("'"))
        ):
            return None
        return (
            indent,
            lhs,
            input_name,
            (int(size_value[0]), int(size_value[1])),
            [int(v) for v in list(shape_value)],
            method_expr[1:-1],
            align_expr == "True",
            hpc_expr == "True",
            channel_last,
        )

    def _is_cf_like(name: str) -> bool:
        return _fast_precanonicalize_is_cf_like(
            name,
            dynamic_cf_like_names,
            context,
        )

    def _is_nhwc_like(name: str) -> bool:
        return _fast_precanonicalize_is_nhwc_like(
            name,
            dynamic_nhwc_like_names,
            context,
        )

    def _has_mixed_layout_add_evidence(
        add_index: int,
        input_a: str,
        input_b: str,
    ) -> bool:
        if (_is_cf_like(input_a) and _is_nhwc_like(input_b)) or (_is_cf_like(input_b) and _is_nhwc_like(input_a)):
            return True
        for back in range(max(0, add_index - 4), add_index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is None
                or {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])}
                != {input_a, input_b}
            ):
                continue
            anchor_input_a = str(binary_anchor_assign[3])
            anchor_input_b = str(binary_anchor_assign[4])
            if (_is_cf_like(anchor_input_a) and _is_nhwc_like(anchor_input_b)) or (
                _is_cf_like(anchor_input_b) and _is_nhwc_like(anchor_input_a)
            ):
                return True
        return False

    assignments_by_lhs: Dict[str, List[int]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            assignments_by_lhs.setdefault(str(assign[1]), []).append(idx)

    def _find_assignment_before(name: str, upper_bound: int) -> int | None:
        for candidate_index in reversed(assignments_by_lhs.get(name, [])):
            if candidate_index < upper_bound:
                return candidate_index
        return None

    def _rewrite_rank4_add_target(
        add_index: int,
        normalized_shape: List[int],
        alias_names: Set[str],
    ) -> bool:
        parsed_add = _parse_static_binary_add_align_assign(lines[add_index])
        if parsed_add is None:
            return False
        indent, lhs, input_a, input_b, current_shape = parsed_add
        unique_channel_count = _infer_unique_channel_count_from_rank4_shape(current_shape)
        if unique_channel_count is not None and int(unique_channel_count) == int(current_shape[1]):
            return False
        if current_shape == normalized_shape:
            return False
        lines[add_index] = (
            f"{indent}{lhs} = _align_tensor_to_target_shape("
            f"torch.add({input_a}, {input_b}), {repr(normalized_shape)})"
        )
        dynamic_cf_like_names.update(alias_names | {lhs})
        dynamic_nhwc_like_names.difference_update(alias_names | {lhs})
        for alias_name in alias_names | {lhs}:
            context.static_shapes[str(alias_name)] = list(normalized_shape)
        for back in range(max(0, add_index - 4), add_index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is None
                or {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])}
                != {input_a, input_b}
            ):
                continue
            lines[back] = (
                f"{binary_anchor_assign[0]}{binary_anchor_assign[1]}, {binary_anchor_assign[2]} = "
                f"_align_binary_inputs_to_anchor({binary_anchor_assign[3]}, {binary_anchor_assign[4]}, "
                f"{repr(normalized_shape)})"
            )
            context.static_shapes[str(binary_anchor_assign[1])] = list(normalized_shape)
            context.static_shapes[str(binary_anchor_assign[2])] = list(normalized_shape)
            dynamic_cf_like_names.update({str(binary_anchor_assign[1]), str(binary_anchor_assign[2])})
            dynamic_nhwc_like_names.difference_update({str(binary_anchor_assign[1]), str(binary_anchor_assign[2])})
            break
        return True

    def _resolve_branch(
        name: str,
        upper_bound: int,
    ) -> Tuple[str, int, object, Set[str]] | None:
        alias_names: Set[str] = {str(name)}
        current_name = str(name)
        current_upper_bound = int(upper_bound)
        for _ in range(12):
            assign_index = _find_assignment_before(current_name, current_upper_bound)
            if assign_index is None:
                return None
            current_line = lines[assign_index]
            resize_assign = _parse_resize_assign(current_line)
            if resize_assign is not None and str(resize_assign[1]) == current_name:
                alias_names.add(str(resize_assign[1]))
                return ("resize", assign_index, resize_assign, alias_names)
            static_add_assign = _parse_static_binary_add_align_assign(current_line)
            if (
                static_add_assign is not None
                and str(static_add_assign[1]) == current_name
                and _has_mixed_layout_add_evidence(
                    assign_index,
                    str(static_add_assign[2]),
                    str(static_add_assign[3]),
                )
            ):
                alias_names.add(str(static_add_assign[1]))
                return ("add", assign_index, static_add_assign, alias_names)
            relu_assign = relu_assign_re.match(current_line)
            if (
                relu_assign is not None
                and str(relu_assign.group("lhs")) == current_name
            ):
                alias_names.add(current_name)
                current_name = str(relu_assign.group("input"))
                current_upper_bound = assign_index
                continue
            simple_assign = _parse_simple_assignment_line(current_line)
            rhs_expr = str(simple_assign[2]).strip() if simple_assign is not None else None
            if (
                simple_assign is not None
                and str(simple_assign[1]) == current_name
                and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr or "") is not None
                and rhs_expr != current_name
            ):
                alias_names.add(current_name)
                current_name = str(rhs_expr)
                current_upper_bound = assign_index
                continue
            return None
        return None

    def _preferred_channel_count_for_branch(
        branch_name: str,
        shape: List[int],
    ) -> int | None:
        if len(shape) == 4:
            if _is_cf_like(branch_name):
                return int(shape[1])
            if _is_nhwc_like(branch_name):
                return int(shape[3])
            unique_channel_count = _infer_unique_channel_count_from_rank4_shape(shape)
            if unique_channel_count is not None:
                return int(unique_channel_count)
        return _infer_structural_rank4_channel_count(
            branch_name,
            shape,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        )

    for index, line in enumerate(lines):
        assign = _parse_simple_assignment_line(line)
        torch_cat_args = _parse_torch_cat_inputs_and_dim(assign[2]) if assign is not None else None
        if (
            assign is None
            or torch_cat_args is None
            or torch_cat_args[1] != 1
        ):
            continue
        cat_inputs = [name.strip() for name in torch_cat_args[0] if name.strip()]
        if len(cat_inputs) < 2:
            continue
        consumer_module: str | None = None
        for lookahead in range(index + 1, min(len(lines), index + 5)):
            direct_conv_match = direct_conv_assign_re.match(lines[lookahead])
            if (
                direct_conv_match is not None
                and str(direct_conv_match.group("input")) == str(assign[1])
            ):
                consumer_module = str(direct_conv_match.group("module"))
                break
        if consumer_module is None:
            for candidate_module in context.module_input_consumers.get(str(assign[1]), []):
                if candidate_module in context.conv_block_in_channels:
                    consumer_module = str(candidate_module)
                    break
        if consumer_module is None:
            continue
        conv_in_channels = context.conv_block_in_channels.get(consumer_module, None)
        if conv_in_channels is None:
            continue
        branch_specs: List[Tuple[str, int, object, Set[str], List[int], int]] = []
        valid = True
        for cat_input in cat_inputs:
            resolved_branch = _resolve_branch(cat_input, index + 1)
            if resolved_branch is None:
                valid = False
                break
            branch_kind, branch_index, branch_parsed, branch_alias_names = resolved_branch
            branch_shape = (
                list(branch_parsed[4])
                if branch_kind == "resize"
                else list(branch_parsed[4])
            )
            if len(branch_shape) != 4:
                valid = False
                break
            preferred_channel_count = _preferred_channel_count_for_branch(cat_input, branch_shape)
            if preferred_channel_count is None:
                valid = False
                break
            branch_specs.append(
                (
                    branch_kind,
                    branch_index,
                    branch_parsed,
                    set(branch_alias_names),
                    list(branch_shape),
                    int(preferred_channel_count),
                )
            )
        if not valid:
            continue
        if sum(spec[5] for spec in branch_specs) != int(conv_in_channels):
            continue
        branch_changed = False
        for branch_kind, branch_index, branch_parsed, branch_alias_names, branch_shape, preferred_channel_count in branch_specs:
            normalized_branch_shape = _normalize_cf_rank4_shape(
                branch_shape,
                preferred_channel_count=int(preferred_channel_count),
                out_hw=branch_parsed[3] if branch_kind == "resize" else None,
            )
            if branch_kind == "resize":
                resize_indent, resize_lhs, resize_input, resize_size, _, resize_method, resize_align, resize_hpc, _ = branch_parsed
                rewritten_resize = (
                    f"{resize_indent}{resize_lhs} = _apply_resize("
                    f"{resize_input}, [{resize_size[0]}, {resize_size[1]}], "
                    f"method='{resize_method}', target_shape={repr(normalized_branch_shape)}, "
                    f"align_corners={resize_align}, half_pixel_centers={resize_hpc}, channel_last=False)"
                )
                if rewritten_resize != lines[branch_index]:
                    lines[branch_index] = rewritten_resize
                    branch_changed = True
                dynamic_cf_like_names.update(branch_alias_names | {resize_lhs})
                dynamic_nhwc_like_names.difference_update(branch_alias_names | {resize_lhs})
                for alias_name in branch_alias_names | {resize_lhs}:
                    context.static_shapes[str(alias_name)] = list(normalized_branch_shape)
                resize_input_branch = _resolve_branch(str(resize_input), branch_index + 1)
                if resize_input_branch is not None and resize_input_branch[0] == "add":
                    input_branch_shape = list(resize_input_branch[2][4])
                    normalized_input_shape = _normalize_cf_rank4_shape(
                        input_branch_shape,
                        preferred_channel_count=int(preferred_channel_count),
                    )
                    if _rewrite_rank4_add_target(
                        int(resize_input_branch[1]),
                        normalized_input_shape,
                        set(resize_input_branch[3]),
                    ):
                        branch_changed = True
            else:
                if _rewrite_rank4_add_target(
                    branch_index,
                    normalized_branch_shape,
                    branch_alias_names,
                ):
                    branch_changed = True
        if branch_changed:
            changed = True
    return changed


def _rewrite_structural_channel_last_slice_concat_targets(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    assignments_by_lhs: Dict[str, List[int]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            assignments_by_lhs.setdefault(str(assign[1]), []).append(idx)

    def _find_assignment_before(name: str, upper_bound: int) -> int | None:
        for candidate_index in reversed(assignments_by_lhs.get(str(name), [])):
            if candidate_index < upper_bound:
                return candidate_index
        return None

    def _parse_rank4_direct_slice_assign(
        line: str,
    ) -> Tuple[str, str, List[str]] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        slice_match = re.fullmatch(
            r"(?P<input>[A-Za-z0-9_]+)\[(?P<dims>.+)\]",
            str(rhs).strip(),
        )
        if slice_match is None:
            return None
        dims = _split_top_level_csv_exprs(str(slice_match.group("dims")))
        if len(dims) != 4:
            return None
        return str(lhs), str(slice_match.group("input")), [str(dim).strip() for dim in dims]

    def _parse_slice_numeric_upper_bound(
        dim_expr: str,
    ) -> int | None:
        stripped = str(dim_expr).strip()
        if stripped == ":":
            return None
        for pattern in [
            r"^:(?P<end>\d+)$",
            r"^0:(?P<end>\d+)$",
            r"^:(?P<end>\d+):1$",
            r"^0:(?P<end>\d+):1$",
        ]:
            match = re.fullmatch(pattern, stripped)
            if match is not None:
                return int(match.group("end"))
        return None

    def _resolve_direct_slice_source(
        name: str,
        upper_bound: int,
    ) -> Tuple[str, List[str]] | None:
        current_name = str(name)
        current_upper_bound = int(upper_bound)
        visited: Set[str] = set()
        for _ in range(12):
            if current_name in visited:
                return None
            visited.add(current_name)
            assign_index = _find_assignment_before(current_name, current_upper_bound)
            if assign_index is None:
                return None
            parsed_slice = _parse_rank4_direct_slice_assign(str(lines[assign_index]))
            if parsed_slice is not None and parsed_slice[0] == current_name:
                return str(parsed_slice[1]), list(parsed_slice[2])
            simple_assign = _parse_simple_assignment_line(str(lines[assign_index]))
            rhs_expr = str(simple_assign[2]).strip() if simple_assign is not None else None
            if (
                simple_assign is not None
                and str(simple_assign[1]) == current_name
                and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr or "") is not None
                and rhs_expr != current_name
            ):
                current_name = str(rhs_expr)
                current_upper_bound = assign_index
                continue
            return None
        return None

    for index, line in enumerate(lines):
        assign = _parse_simple_assignment_line(line)
        concat_args = _parse_apply_concat_inputs_axis_and_shape(assign[2]) if assign is not None else None
        if assign is None or concat_args is None:
            continue
        concat_inputs, concat_axis, target_shape = concat_args
        if target_shape is None or len(target_shape) != 4 or concat_axis not in {1, 2}:
            continue
        preferred_channel_count = _infer_unique_channel_count_from_rank4_shape(target_shape)
        if preferred_channel_count is None or int(preferred_channel_count) != int(target_shape[1]):
            continue
        if int(target_shape[3]) == int(preferred_channel_count):
            continue
        slice_sources: List[str] = []
        valid = True
        for concat_input in concat_inputs:
            resolved_slice = _resolve_direct_slice_source(str(concat_input).strip(), index + 1)
            if resolved_slice is None:
                valid = False
                break
            slice_source, slice_dims = resolved_slice
            last_dim_upper = _parse_slice_numeric_upper_bound(slice_dims[3])
            if last_dim_upper is None:
                source_shape = (
                    context.static_shapes.get(slice_source)
                    or context.static_shapes.get(
                        _fast_precanonicalize_resolve_alias(slice_source, context.aliases)
                    )
                )
                if source_shape is None or len(source_shape) != 4 or int(source_shape[3]) != int(preferred_channel_count):
                    valid = False
                    break
            elif int(last_dim_upper) != int(preferred_channel_count):
                valid = False
                break
            source_assign_index = _find_assignment_before(slice_source, index + 1)
            if not (
                _fast_precanonicalize_is_nhwc_like(
                    slice_source,
                    dynamic_nhwc_like_names,
                    context,
                )
                or _fast_precanonicalize_has_channel_last_spatial_consumer(
                    slice_source,
                    source_assign_index if source_assign_index is not None else -1,
                    lines,
                    context,
                )
            ):
                valid = False
                break
            slice_sources.append(str(slice_source))
        if not valid or len(set(slice_sources)) != 1:
            continue
        normalized_shape = _normalize_nhwc_rank4_shape(
            list(target_shape),
            preferred_channel_count=int(preferred_channel_count),
        )
        rewritten = (
            f"{assign[0]}{assign[1]} = _apply_concat([{', '.join(str(name).strip() for name in concat_inputs)}], "
            f"axis={concat_axis}, target_shape={repr(normalized_shape)}, fused='NONE')"
        )
        if rewritten == lines[index]:
            continue
        lines[index] = rewritten
        dynamic_nhwc_like_names.add(str(assign[1]))
        dynamic_cf_like_names.discard(str(assign[1]))
        context.static_shapes[str(assign[1])] = list(normalized_shape)
        changed = True
    return changed


def _rewrite_structural_concat_resize_input_cf_add_targets(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    direct_conv_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    relu_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.relu\((?P<input>[A-Za-z0-9_]+)\)$"
    )
    assignments_by_lhs: Dict[str, List[int]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            assignments_by_lhs.setdefault(str(assign[1]), []).append(idx)

    def _find_assignment_before(name: str, upper_bound: int) -> int | None:
        for candidate_index in reversed(assignments_by_lhs.get(name, [])):
            if candidate_index < upper_bound:
                return candidate_index
        return None

    def _parse_resize_assign(
        line: str,
    ) -> Tuple[str, str, str, Tuple[int, int], List[int], str, bool, bool, bool] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        parsed = _parse_apply_resize_input_size_shape_and_channel_last(rhs)
        if parsed is None:
            return None
        input_name, size_value, shape_value, channel_last = parsed
        if size_value is None or shape_value is None:
            return None
        stripped = rhs.strip()
        if not stripped.startswith("_apply_resize(") or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len("_apply_resize(") : -1])
        method_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            if key.strip() == "method":
                method_expr = value.strip()
            elif key.strip() == "align_corners":
                align_expr = value.strip()
            elif key.strip() == "half_pixel_centers":
                hpc_expr = value.strip()
        if (
            method_expr is None
            or align_expr not in {"True", "False"}
            or hpc_expr not in {"True", "False"}
            or not (method_expr.startswith("'") and method_expr.endswith("'"))
        ):
            return None
        return (
            indent,
            lhs,
            input_name,
            (int(size_value[0]), int(size_value[1])),
            [int(v) for v in list(shape_value)],
            method_expr[1:-1],
            align_expr == "True",
            hpc_expr == "True",
            channel_last,
        )

    def _resolve_resize_assign(name: str, upper_bound: int) -> Tuple[int, Tuple[str, str, str, Tuple[int, int], List[int], str, bool, bool, bool]] | None:
        current_name = str(name)
        current_upper_bound = int(upper_bound)
        for _ in range(8):
            assign_index = _find_assignment_before(current_name, current_upper_bound)
            if assign_index is None:
                return None
            resize_assign = _parse_resize_assign(lines[assign_index])
            if resize_assign is not None and str(resize_assign[1]) == current_name:
                return assign_index, resize_assign
            simple_assign = _parse_simple_assignment_line(lines[assign_index])
            rhs_expr = str(simple_assign[2]).strip() if simple_assign is not None else ""
            if (
                simple_assign is not None
                and re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr) is not None
                and str(simple_assign[1]) == current_name
                and rhs_expr != current_name
            ):
                current_name = rhs_expr
                current_upper_bound = assign_index
                continue
            return None
        return None

    def _resolve_add_assign(name: str, upper_bound: int) -> Tuple[int, Tuple[str, str, str, str, List[int]]] | None:
        current_name = str(name)
        current_upper_bound = int(upper_bound)
        for _ in range(8):
            assign_index = _find_assignment_before(current_name, current_upper_bound)
            if assign_index is None:
                return None
            parsed_add = _parse_static_binary_add_align_assign(lines[assign_index])
            if parsed_add is not None and str(parsed_add[1]) == current_name:
                return assign_index, parsed_add
            relu_assign = relu_assign_re.match(lines[assign_index])
            if relu_assign is not None and str(relu_assign.group("lhs")) == current_name:
                current_name = str(relu_assign.group("input"))
                current_upper_bound = assign_index
                continue
            return None
        return None

    def _rewrite_add_target_to_cf(
        add_index: int,
        parsed_add: Tuple[str, str, str, str, List[int]],
        *,
        preferred_channel_count: int,
    ) -> bool:
        nonlocal changed
        current_shape = list(parsed_add[4])
        unique_channel_count = _infer_unique_channel_count_from_rank4_shape(current_shape)
        if unique_channel_count is not None and int(unique_channel_count) == int(current_shape[1]):
            return False
        normalized_shape = _normalize_cf_rank4_shape(
            current_shape,
            preferred_channel_count=int(preferred_channel_count),
        )
        if normalized_shape == current_shape:
            return False
        lines[add_index] = (
            f"{parsed_add[0]}{parsed_add[1]} = _align_tensor_to_target_shape("
            f"torch.add({parsed_add[2]}, {parsed_add[3]}), {repr(normalized_shape)})"
        )
        context.static_shapes[str(parsed_add[1])] = list(normalized_shape)
        dynamic_cf_like_names.add(str(parsed_add[1]))
        dynamic_nhwc_like_names.discard(str(parsed_add[1]))
        changed = True
        for back in range(max(0, add_index - 4), add_index):
            binary_anchor_assign = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[back])
            if (
                binary_anchor_assign is None
                or {str(binary_anchor_assign[1]), str(binary_anchor_assign[2])}
                != {str(parsed_add[2]), str(parsed_add[3])}
            ):
                continue
            lines[back] = (
                f"{binary_anchor_assign[0]}{binary_anchor_assign[1]}, {binary_anchor_assign[2]} = "
                f"_align_binary_inputs_to_anchor({binary_anchor_assign[3]}, {binary_anchor_assign[4]}, {repr(normalized_shape)})"
            )
            context.static_shapes[str(binary_anchor_assign[1])] = list(normalized_shape)
            context.static_shapes[str(binary_anchor_assign[2])] = list(normalized_shape)
            dynamic_cf_like_names.update({str(binary_anchor_assign[1]), str(binary_anchor_assign[2])})
            dynamic_nhwc_like_names.difference_update({str(binary_anchor_assign[1]), str(binary_anchor_assign[2])})
            break
        return True

    for index, line in enumerate(lines):
        assign = _parse_simple_assignment_line(line)
        torch_cat_args = _parse_torch_cat_inputs_and_dim(assign[2]) if assign is not None else None
        if assign is None or torch_cat_args is None or torch_cat_args[1] != 1:
            continue
        consumer_module: str | None = None
        for lookahead in range(index + 1, min(len(lines), index + 5)):
            direct_conv_match = direct_conv_assign_re.match(lines[lookahead])
            if direct_conv_match is not None and str(direct_conv_match.group("input")) == str(assign[1]):
                consumer_module = str(direct_conv_match.group("module"))
                break
        if consumer_module is None:
            for candidate_module in context.module_input_consumers.get(str(assign[1]), []):
                if candidate_module in context.conv_block_in_channels:
                    consumer_module = str(candidate_module)
                    break
        if consumer_module is None:
            continue
        conv_in_channels = context.conv_block_in_channels.get(consumer_module)
        cat_inputs = [name.strip() for name in torch_cat_args[0] if name.strip()]
        inferred_branch_channels: List[int] = []
        inferred_branch_channels_by_input: Dict[str, int] = {}
        valid_for_direct_add_branches = conv_in_channels is not None
        for cat_input in cat_inputs:
            resolved_resize = _resolve_resize_assign(cat_input, index + 1)
            if resolved_resize is not None:
                resize_shape = list(resolved_resize[1][4])
                preferred_channel_count = (
                    int(resize_shape[3])
                    if resolved_resize[1][8]
                    else int(resize_shape[1])
                )
                inferred_branch_channels.append(preferred_channel_count)
                inferred_branch_channels_by_input[cat_input] = int(preferred_channel_count)
                continue
            if conv_in_channels is None:
                continue
            resolved_add = _resolve_add_assign(cat_input, index + 1)
            if resolved_add is None:
                valid_for_direct_add_branches = False
                break
            add_shape = list(resolved_add[1][4])
            preferred_channel_count = _infer_structural_rank4_channel_count(
                cat_input,
                add_shape,
                dynamic_cf_like_names,
                dynamic_nhwc_like_names,
                context,
            )
            if preferred_channel_count is None:
                valid_for_direct_add_branches = False
                break
            inferred_branch_channels.append(int(preferred_channel_count))
            inferred_branch_channels_by_input[cat_input] = int(preferred_channel_count)
        for cat_input in cat_inputs:
            resolved_resize = _resolve_resize_assign(cat_input, index + 1)
            if resolved_resize is None:
                continue
            resize_index, resize_assign = resolved_resize
            if resize_assign[8]:
                continue
            resolved_add = _resolve_add_assign(str(resize_assign[2]), resize_index + 1)
            if resolved_add is None:
                continue
            _rewrite_add_target_to_cf(
                resolved_add[0],
                resolved_add[1],
                preferred_channel_count=int(resize_assign[4][1]),
            )
        if (
            conv_in_channels is None
            or not valid_for_direct_add_branches
            or sum(inferred_branch_channels) != int(conv_in_channels)
        ):
            continue
        for cat_input in cat_inputs:
            resolved_resize = _resolve_resize_assign(cat_input, index + 1)
            if resolved_resize is not None:
                continue
            preferred_channel_count = inferred_branch_channels_by_input.get(cat_input)
            if preferred_channel_count is None:
                continue
            resolved_add = _resolve_add_assign(cat_input, index + 1)
            if resolved_add is None:
                continue
            _rewrite_add_target_to_cf(
                resolved_add[0],
                resolved_add[1],
                preferred_channel_count=preferred_channel_count,
            )
    return changed


def _rewrite_structural_direct_cf_resize_target_shapes_for_cat_consumers(
    lines: List[str],
    dynamic_cf_like_names: Set[str],
    dynamic_nhwc_like_names: Set[str],
    context: _FastPrecanonicalizeRepairContext,
) -> bool:
    changed = False
    assignments_by_lhs: Dict[str, List[int]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            assignments_by_lhs.setdefault(str(assign[1]), []).append(idx)

    def _find_assignment_before(name: str, upper_bound: int) -> int | None:
        for candidate_index in reversed(assignments_by_lhs.get(str(name), [])):
            if candidate_index < upper_bound:
                return candidate_index
        return None

    def _parse_resize_assign(
        line: str,
    ) -> Tuple[str, str, str, Tuple[int, int], List[int], str, str | None, str | None, bool] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        indent, lhs, rhs = assign
        parsed = _parse_apply_resize_input_size_shape_and_channel_last(rhs)
        if parsed is None:
            return None
        input_name, size_value, shape_value, channel_last = parsed
        if size_value is None or shape_value is None:
            return None
        stripped = rhs.strip()
        if not stripped.startswith("_apply_resize(") or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len("_apply_resize(") : -1])
        method_expr: str | None = None
        align_expr: str | None = None
        hpc_expr: str | None = None
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            if key.strip() == "method":
                method_expr = value.strip()
            elif key.strip() == "align_corners":
                align_expr = value.strip()
            elif key.strip() == "half_pixel_centers":
                hpc_expr = value.strip()
        if method_expr is None or not (method_expr.startswith("'") and method_expr.endswith("'")):
            return None
        return (
            indent,
            lhs,
            input_name,
            (int(size_value[0]), int(size_value[1])),
            [int(v) for v in list(shape_value)],
            method_expr[1:-1],
            align_expr,
            hpc_expr,
            channel_last,
        )

    for index, line in enumerate(lines):
        resize_assign = _parse_resize_assign(line)
        if resize_assign is None:
            continue
        indent, lhs, input_name, resize_size, target_shape, method, align_corners_expr, half_pixel_centers_expr, channel_last = resize_assign
        if channel_last or len(target_shape) != 4:
            continue
        if int(target_shape[1]) != int(resize_size[0]) or int(target_shape[2]) != int(resize_size[1]):
            continue
        if int(target_shape[3]) in {int(resize_size[0]), int(resize_size[1])}:
            continue
        cat_consumer_found = False
        for lookahead in range(index + 1, min(len(lines), index + 5)):
            consumer_assign = _parse_simple_assignment_line(lines[lookahead])
            consumer_cat = _parse_torch_cat_inputs_and_dim(consumer_assign[2]) if consumer_assign is not None else None
            if (
                consumer_assign is None
                or consumer_cat is None
                or consumer_cat[1] != 1
                or lhs not in [name.strip() for name in consumer_cat[0] if name.strip()]
            ):
                continue
            cat_consumer_found = True
            break
        if not cat_consumer_found:
            continue
        preferred_channel_count = _infer_structural_rank4_channel_count(
            lhs,
            target_shape,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        )
        if preferred_channel_count is None:
            preferred_channel_count = int(target_shape[3])
        normalized_shape = _normalize_cf_rank4_shape(
            target_shape,
            preferred_channel_count=int(preferred_channel_count),
            out_hw=resize_size,
        )
        if normalized_shape == target_shape:
            continue
        lines[index] = (
            f"{indent}{lhs} = _apply_resize("
            f"{input_name}, [{resize_size[0]}, {resize_size[1]}], "
            f"method='{method}', target_shape={repr(normalized_shape)}"
            f"{', align_corners=' + align_corners_expr if align_corners_expr is not None else ''}"
            f"{', half_pixel_centers=' + half_pixel_centers_expr if half_pixel_centers_expr is not None else ''}, "
            f"channel_last=False)"
        )
        dynamic_cf_like_names.add(lhs)
        dynamic_nhwc_like_names.discard(lhs)
        context.static_shapes[str(lhs)] = list(normalized_shape)
        assign_index = _find_assignment_before(input_name, index)
        if assign_index is not None and _fast_precanonicalize_is_cf_like(input_name, dynamic_cf_like_names, context):
            dynamic_cf_like_names.add(input_name)
        changed = True
    return changed


def _rewrite_structural_rank3_feature_tail_split_axes(
    lines: List[str],
) -> bool:
    changed = False
    assignments_by_lhs: Dict[str, List[int]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            assignments_by_lhs.setdefault(str(assign[1]), []).append(idx)

    def _find_assignment_before(name: str, upper_bound: int) -> int | None:
        for candidate_index in reversed(assignments_by_lhs.get(str(name), [])):
            if candidate_index < upper_bound:
                return candidate_index
        return None

    def _parse_rank3_concat_assign(line: str) -> Tuple[str, List[int]] | None:
        assign = _parse_simple_assignment_line(line)
        concat_args = _parse_apply_concat_inputs_axis_and_shape(assign[2]) if assign is not None else None
        if assign is None or concat_args is None or concat_args[2] is None:
            return None
        target_shape = list(concat_args[2])
        if len(target_shape) != 3:
            return None
        return str(assign[1]), target_shape

    for index, line in enumerate(lines):
        split_assign = _parse_tensor_split_assign(line)
        if split_assign is None:
            continue
        indent, outputs, input_name, sections, axis = split_assign
        if sections <= 1 or axis == 2:
            continue
        concat_shape: List[int] | None = None
        concat_index = _find_assignment_before(input_name, index + 1)
        if concat_index is not None:
            parsed_concat = _parse_rank3_concat_assign(lines[concat_index])
            if parsed_concat is not None and parsed_concat[0] == input_name:
                concat_shape = list(parsed_concat[1])
        if concat_shape is None or concat_shape[2] % sections != 0:
            continue
        expected_shape = [int(concat_shape[0]), int(concat_shape[1]), int(concat_shape[2] // sections)]
        matched_outputs: Set[str] = set()
        for lookahead in range(index + 1, min(len(lines), index + 12)):
            shape_match = re.search(r"\[(?P<shape>[0-9,\s]+)\]", str(lines[lookahead]))
            if shape_match is None:
                continue
            candidate_shape = _parse_int_list_literal(str(shape_match.group("shape")))
            if candidate_shape != expected_shape:
                continue
            for output_name in outputs:
                if re.search(rf"\b{re.escape(output_name)}\b", str(lines[lookahead])) is not None:
                    matched_outputs.add(str(output_name))
        if len(matched_outputs) != len(outputs):
            continue
        lines[index] = (
            f"{indent}{', '.join(outputs)} = list(torch.tensor_split("
            f"{input_name}, {sections}, dim=_normalize_dim(2, {input_name}.ndim)))"
        )
        changed = True
    return changed


def _apply_structural_fast_precanonicalize_repairs(model_path: Path) -> None:
    if not model_path.exists():
        return
    lines = model_path.read_text(encoding="utf-8").splitlines()
    context = _build_fast_precanonicalize_repair_context(lines)
    dynamic_cf_like_names: Set[str] = set(context.cf_like_names)
    dynamic_nhwc_like_names: Set[str] = set(context.nhwc_like_names)
    changed = False
    direct_conv_assign_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*self\.(?P<module>conv_block_[0-9]+)\((?P<input>[A-Za-z0-9_]+)\)$"
    )

    simple_assignments_by_lhs: Dict[str, int] = {}
    direct_conv_assignments_by_lhs: Dict[str, Tuple[int, str, str]] = {}
    for idx, current_line in enumerate(lines):
        assign = _parse_simple_assignment_line(current_line)
        if assign is not None:
            simple_assignments_by_lhs[str(assign[1])] = idx
        direct_conv_match = direct_conv_assign_re.match(current_line)
        if direct_conv_match is not None:
            direct_conv_assignments_by_lhs[str(direct_conv_match.group("lhs"))] = (
                idx,
                str(direct_conv_match.group("module")),
                str(direct_conv_match.group("input")),
            )

    for index, line in enumerate(lines):
        simple_alias = _parse_simple_assignment_line(line)
        if (
            simple_alias is not None
            and index + 2 < len(lines)
            and re.fullmatch(r"[A-Za-z0-9_]+", simple_alias[2].strip()) is not None
            and _fast_precanonicalize_is_cf_like(simple_alias[2].strip(), dynamic_cf_like_names, context)
        ):
            pad_assign = _parse_constant_pad_assign(lines[index + 1])
            pool_assign = _parse_apply_pool2d_assign_with_shape(lines[index + 2])
            if pad_assign is not None and pool_assign is not None and pool_assign[5] and pool_assign[6]:
                aligned_pad_input = _parse_align_tensor_target_shape_expr(pad_assign[2])
                if (
                    aligned_pad_input is not None
                    and aligned_pad_input[0].strip() == simple_alias[1]
                    and pad_assign[1] == pool_assign[2]
                    and "_nhwc" in str(pool_assign[1])
                ):
                    nchw_pad = _convert_nhwc_pad_to_nchw_pad_values(pad_assign[3])
                    preferred_channel_count = _infer_structural_rank4_channel_count(
                        pool_assign[1],
                        pool_assign[4],
                        dynamic_cf_like_names,
                        dynamic_nhwc_like_names,
                        context,
                    )
                    if nchw_pad is not None and preferred_channel_count is not None:
                        lines[index + 1] = (
                            f"{pad_assign[0]}{pad_assign[1]} = "
                            f"F.pad({simple_alias[2].strip()}, {repr(nchw_pad)}, mode='constant', value={pad_assign[4]})"
                        )
                        lines[index + 2] = (
                            f"{pool_assign[0]}{pool_assign[1]} = _apply_pool2d("
                            f"{pool_assign[2]}, {pool_assign[3]}, "
                            f"target_shape={repr(_normalize_cf_rank4_shape(pool_assign[4], preferred_channel_count=preferred_channel_count))}, "
                            f"is_max_pool={pool_assign[5]}, channel_last=False)"
                        )
                        dynamic_cf_like_names.add(pool_assign[1])
                        changed = True
                        continue

        if _rewrite_structural_mixed_layout_binary_anchor_and_add(
            lines,
            index,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        ):
            changed = True
            continue

        assign = _parse_simple_assignment_line(line)
        torch_cat_args = _parse_torch_cat_inputs_and_dim(assign[2]) if assign is not None else None
        if (
            assign is not None
            and torch_cat_args is not None
            and torch_cat_args[1] == 1
        ):
            cat_inputs = [name.strip() for name in torch_cat_args[0] if name.strip()]
            if cat_inputs and all(
                _fast_precanonicalize_is_cf_like(input_name, dynamic_cf_like_names, context)
                for input_name in cat_inputs
            ):
                cat_changed = False
                for lookahead in range(index + 1, min(len(lines), index + 5)):
                    direct_conv_match = direct_conv_assign_re.match(lines[lookahead])
                    if direct_conv_match is None or str(direct_conv_match.group("input")) != assign[1]:
                        continue
                    conv_module = str(direct_conv_match.group("module"))
                    conv_lhs = str(direct_conv_match.group("lhs"))
                    conv_in_channels = context.conv_block_in_channels.get(conv_module)
                    conv_out_channels = context.conv_block_out_channels.get(conv_module)
                    if conv_in_channels is None or conv_out_channels is None:
                        continue
                    for anchor_index in range(lookahead + 1, min(len(lines), lookahead + 5)):
                        parsed_anchor = _parse_align_binary_inputs_to_anchor_assign_with_shape(lines[anchor_index])
                        if parsed_anchor is None or conv_lhs not in {parsed_anchor[3], parsed_anchor[4]}:
                            continue
                        normalized_anchor_shape = _normalize_nhwc_rank4_shape(
                            parsed_anchor[5],
                            preferred_channel_count=int(conv_out_channels),
                        )
                        lines[index] = (
                            f"{assign[0]}{assign[1]} = _apply_concat([{', '.join(cat_inputs)}], "
                            f"axis=3, target_shape={[normalized_anchor_shape[0], normalized_anchor_shape[1], normalized_anchor_shape[2], int(conv_in_channels)]}, "
                            "fused='NONE')"
                        )
                        dynamic_nhwc_like_names.add(assign[1])
                        if _rewrite_structural_mixed_layout_binary_anchor_and_add(
                            lines,
                            anchor_index,
                            dynamic_cf_like_names,
                            dynamic_nhwc_like_names,
                            context,
                        ):
                            pass
                        cat_changed = True
                        changed = True
                        break
                    if cat_changed:
                        break
                if cat_changed:
                    continue
                for lookahead in range(index + 1, min(len(lines), index + 4)):
                    permute_assign = _parse_torch_permute_assign(lines[lookahead])
                    if (
                        permute_assign is None
                        or permute_assign[3] != [0, 3, 1, 2]
                        or str(permute_assign[2]) != assign[1]
                    ):
                        continue
                    for cat_input in cat_inputs:
                        input_index = simple_assignments_by_lhs.get(cat_input)
                        input_assign = _parse_simple_assignment_line(lines[input_index]) if input_index is not None else None
                        softmax_args = _parse_apply_softmax_input_axis_and_shape(input_assign[2]) if input_assign is not None else None
                        if softmax_args is not None and softmax_args[2] is not None:
                            preferred_channel_count = _infer_structural_rank4_channel_count(
                                cat_input,
                                list(softmax_args[2]),
                                dynamic_cf_like_names,
                                dynamic_nhwc_like_names,
                                context,
                            )
                            if preferred_channel_count is not None:
                                normalized_softmax_shape = _normalize_cf_rank4_shape(
                                    list(softmax_args[2]),
                                    preferred_channel_count=preferred_channel_count,
                                )
                                rewritten_softmax = (
                                    f"{input_assign[0]}{input_assign[1]} = _apply_softmax("
                                    f"{softmax_args[0]}, axis=1, beta=1.0, target_shape={repr(normalized_softmax_shape)})"
                                )
                                if lines[input_index] != rewritten_softmax:
                                    lines[input_index] = rewritten_softmax
                                    dynamic_cf_like_names.add(cat_input)
                                    changed = True
                    rewritten_output = f"{permute_assign[0]}{permute_assign[1]} = {assign[1]}"
                    if lines[lookahead] != rewritten_output:
                        lines[lookahead] = rewritten_output
                        dynamic_cf_like_names.add(str(permute_assign[1]))
                        changed = True
                    break
            public_bridge_inputs_ok = bool(cat_inputs)
            if public_bridge_inputs_ok:
                for cat_input in cat_inputs:
                    if _fast_precanonicalize_is_cf_like(cat_input, dynamic_cf_like_names, context):
                        continue
                    input_index = simple_assignments_by_lhs.get(cat_input)
                    input_assign = _parse_simple_assignment_line(lines[input_index]) if input_index is not None else None
                    softmax_args = _parse_apply_softmax_input_axis_and_shape(input_assign[2]) if input_assign is not None else None
                    if softmax_args is None or softmax_args[2] is None:
                        public_bridge_inputs_ok = False
                        break
            if public_bridge_inputs_ok:
                for lookahead in range(index + 1, min(len(lines), index + 4)):
                    permute_assign = _parse_torch_permute_assign(lines[lookahead])
                    if (
                        permute_assign is None
                        or permute_assign[3] != [0, 3, 1, 2]
                        or str(permute_assign[2]) != assign[1]
                    ):
                        continue
                    for cat_input in cat_inputs:
                        input_index = simple_assignments_by_lhs.get(cat_input)
                        input_assign = _parse_simple_assignment_line(lines[input_index]) if input_index is not None else None
                        softmax_args = _parse_apply_softmax_input_axis_and_shape(input_assign[2]) if input_assign is not None else None
                        if softmax_args is not None and softmax_args[2] is not None:
                            preferred_channel_count = _infer_structural_rank4_channel_count(
                                cat_input,
                                list(softmax_args[2]),
                                dynamic_cf_like_names,
                                dynamic_nhwc_like_names,
                                context,
                            )
                            if preferred_channel_count is not None:
                                normalized_softmax_shape = _normalize_cf_rank4_shape(
                                    list(softmax_args[2]),
                                    preferred_channel_count=preferred_channel_count,
                                )
                                rewritten_softmax = (
                                    f"{input_assign[0]}{input_assign[1]} = _apply_softmax("
                                    f"{softmax_args[0]}, axis=1, beta=1.0, target_shape={repr(normalized_softmax_shape)})"
                                )
                                if lines[input_index] != rewritten_softmax:
                                    lines[input_index] = rewritten_softmax
                                    dynamic_cf_like_names.add(cat_input)
                                    changed = True
                    rewritten_output = f"{permute_assign[0]}{permute_assign[1]} = {assign[1]}"
                    if lines[lookahead] != rewritten_output:
                        lines[lookahead] = rewritten_output
                        dynamic_cf_like_names.add(str(permute_assign[1]))
                        changed = True
                    break
    if _rewrite_structural_channel_last_slice_concat_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_concat_conv_cf_branch_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_concat_resize_input_cf_add_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_direct_cf_resize_target_shapes_for_cat_consumers(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_channel_first_rank4_flatten_to_nwc(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_rank3_feature_tail_split_axes(lines):
        changed = True
    if _rewrite_structural_nhwc_image_tail_bridges(
        lines,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_cf_anchor_binary_add_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_direct_conv_cf_add_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_redundant_nhwc_to_cf_conv_bridges(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_channel_last_pool_pad_pairs(
        lines,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    rewritten_lines = _repair_channel_last_gap_conv_inputs(lines)
    if rewritten_lines != lines:
        lines = rewritten_lines
        changed = True
    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_structural_final_model_repairs(model_path: Path) -> None:
    if not model_path.exists():
        return
    lines = model_path.read_text(encoding="utf-8").splitlines()
    context = _build_fast_precanonicalize_repair_context(lines)
    dynamic_cf_like_names: Set[str] = set(context.cf_like_names)
    dynamic_nhwc_like_names: Set[str] = set(context.nhwc_like_names)
    changed = False
    for index, line in enumerate(lines):
        if _rewrite_structural_mixed_layout_binary_anchor_and_add(
            lines,
            index,
            dynamic_cf_like_names,
            dynamic_nhwc_like_names,
            context,
        ):
            changed = True
            continue
        assign = _parse_simple_assignment_line(line)
        softmax_args = _parse_apply_softmax_input_axis_and_shape(assign[2]) if assign is not None else None
        if (
            assign is not None
            and softmax_args is not None
            and softmax_args[2] is not None
            and _fast_precanonicalize_is_nhwc_like(softmax_args[0], dynamic_nhwc_like_names, context)
        ):
            for lookahead in range(index + 1, min(len(lines), index + 4)):
                cat_assign = _parse_simple_assignment_line(lines[lookahead])
                cat_args = _parse_torch_cat_inputs_and_dim(cat_assign[2]) if cat_assign is not None else None
                if cat_args is None or cat_args[1] != 1 or assign[1] not in [name.strip() for name in cat_args[0]]:
                    continue
                bridge_line_assign = (
                    _parse_simple_assignment_line(lines[lookahead + 1])
                    if lookahead + 1 < len(lines)
                    else None
                )
                bridge_permute = (
                    _parse_torch_permute_assign(lines[lookahead + 1])
                    if lookahead + 1 < len(lines)
                    else None
                )
                bridge_lhs: str | None = None
                if (
                    bridge_permute is not None
                    and bridge_permute[3] == [0, 3, 1, 2]
                    and str(bridge_permute[2]) == str(cat_assign[1])
                ):
                    bridge_lhs = str(bridge_permute[1])
                elif (
                    bridge_line_assign is not None
                    and str(bridge_line_assign[2]).strip() == str(cat_assign[1])
                ):
                    bridge_lhs = str(bridge_line_assign[1])
                if bridge_lhs is None:
                    continue
                preferred_channel_count = _infer_structural_rank4_channel_count(
                    assign[1],
                    list(softmax_args[2]),
                    dynamic_cf_like_names,
                    dynamic_nhwc_like_names,
                    context,
                )
                if preferred_channel_count is None:
                    continue
                normalized_shape = _normalize_cf_rank4_shape(
                    list(softmax_args[2]),
                    preferred_channel_count=preferred_channel_count,
                )
                rewritten_softmax = (
                    f"{assign[0]}{assign[1]} = _apply_softmax("
                    f"{softmax_args[0]}, axis=1, beta=1.0, target_shape={repr(normalized_shape)})"
                )
                if lines[index] != rewritten_softmax:
                    lines[index] = rewritten_softmax
                    dynamic_cf_like_names.add(assign[1])
                    changed = True
                rewritten_bridge = f"{bridge_line_assign[0]}{bridge_lhs} = {cat_assign[1]}"
                if lines[lookahead + 1] != rewritten_bridge:
                    lines[lookahead + 1] = rewritten_bridge
                    dynamic_cf_like_names.add(str(bridge_lhs))
                    changed = True
                break
    if _rewrite_structural_plain_mixed_layout_attention_adds(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_channel_last_slice_concat_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_concat_conv_cf_branch_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_concat_resize_input_cf_add_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_direct_cf_resize_target_shapes_for_cat_consumers(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_channel_first_rank4_flatten_to_nwc(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_nhwc_image_tail_bridges(
        lines,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_channel_first_depth_to_space_public_bridges(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_rank3_feature_tail_split_axes(lines):
        changed = True
    if _rewrite_structural_concat_resize_input_cf_add_targets(
        lines,
        dynamic_cf_like_names,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if _rewrite_structural_channel_last_pool_pad_pairs(
        lines,
        dynamic_nhwc_like_names,
        context,
    ):
        changed = True
    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _reapply_post_export_final_model_repairs(package_path: Path) -> None:
    model_path = Path(package_path) / "model.py"
    if model_path.exists():
        _apply_structural_final_model_repairs(model_path)


def _apply_dynamic_score_sampling_stage_precanonicalize_repairs(model_path: Path) -> None:
    if not model_path.exists():
        return
    model_source = model_path.read_text(encoding="utf-8")
    lines = model_source.splitlines()
    depth_to_space_nhwc_gather_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>[A-Za-z0-9_]+)\[:, \[(?P<indices>[0-9,\s-]+)\], :, :\]$"
    )
    changed = False

    local_alias_sources: Dict[str, str] = {}
    for line in lines:
        assign = _parse_simple_assignment_line(str(line))
        if assign is None:
            continue
        rhs_expr = str(assign[2]).strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", rhs_expr) is None:
            continue
        local_alias_sources[str(assign[1])] = rhs_expr

    def _resolve_local_alias(name: str) -> str:
        resolved = str(name)
        seen: Set[str] = set()
        while resolved not in seen and resolved in local_alias_sources:
            seen.add(resolved)
            resolved = str(local_alias_sources[resolved])
        return resolved

    def _is_structural_cf_name(name: str) -> bool:
        resolved = _resolve_local_alias(str(name))
        return resolved.endswith("_cf") or resolved.endswith("_out_cf")

    def _is_structural_nhwc_name(name: str) -> bool:
        resolved = _resolve_local_alias(str(name))
        return "_nhwc" in resolved

    index = 0
    while index < len(lines):
        current_line = str(lines[index])
        next_line = str(lines[index + 1]) if index + 1 < len(lines) else None
        depth_to_space_gather_match = depth_to_space_nhwc_gather_re.match(current_line)
        if depth_to_space_gather_match is not None:
            lhs_name = str(depth_to_space_gather_match.group("lhs"))
            input_name = str(depth_to_space_gather_match.group("input"))
            next_assigns_depth_to_space = (
                next_line is not None
                and re.match(rf"^\s*[A-Za-z0-9_]+\s*=\s*{re.escape(lhs_name)}$", next_line) is not None
                and "_depth_to_space_" in next_line
            )
            if (
                _is_structural_nhwc_name(input_name)
                and not _is_structural_cf_name(input_name)
                and (
                    "depth_to" in lhs_name.lower()
                    or "depthtospace" in lhs_name.lower()
                    or next_assigns_depth_to_space
                )
            ):
                lines[index] = (
                    f"{depth_to_space_gather_match.group('indent')}{lhs_name} = "
                    f"{input_name}[:, :, :, [{depth_to_space_gather_match.group('indices')}]]"
                )
                changed = True
        index += 1
    if not _has_dynamic_score_sampling_stage_signature(lines):
        if changed:
            model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    stage7_reshape_shape_pattern = r"(?:\[\s*-1\s*,\s*1\s*\]|\(\s*-1\s*,\s*1\s*\))"

    rank4_eq_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.eq\((?P<a>[A-Za-z0-9_]+), (?P<b>[A-Za-z0-9_]+)\), \[(?P<n>\d+), (?P<h>\d+), (?P<w>\d+), 1\]\)$"
    )
    nhwc_to_nchw_singleton_reshape_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\((?P<input>[A-Za-z0-9_]+)\.permute\(0, 3, 1, 2\)\.contiguous\(\), \[(?P<n>\d+), 1, (?P<h>\d+), (?P<w>\d+)\]\)$"
    )
    singleton_rank4_wrapper_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\((?P<expr>torch\.(?:lt|add|sub|logical_and|mul)\(.+\)), \[1, 1, 1, 1\]\)$"
    )
    singleton_mul_const_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.mul\((?P<input>[A-Za-z0-9_]+), (?P<const>self\.const_inline_literal_0)\), \[1, 2\]\)$"
    )
    self_square_align_mul_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.mul\((?P<input>[A-Za-z0-9_]+), (?P=input)\), \[(?P<n>\d+), 1\]\)$"
    )
    singleton_binary_anchor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+), (?P<lhs1>[A-Za-z0-9_]+) = _align_binary_inputs_to_anchor\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+), \[1, 1\]\)$"
    )
    singleton_mul_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.mul\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+)\), \[1, 1\]\)$"
    )
    column_binary_anchor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs0>[A-Za-z0-9_]+), (?P<lhs1>[A-Za-z0-9_]+) = _align_binary_inputs_to_anchor\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+), \[(?P<n>\d+), 1\]\)$"
    )
    column_div_align_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.div\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+)\), \[(?P<n>\d+), 1\]\)$"
    )
    div_floor_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = _align_tensor_to_target_shape\(torch\.div\((?P<input>[A-Za-z0-9_]+), (?P<divisor>\d+)\.0\), \[1\]\)$"
    )
    cast_float_re = re.compile(
        r"^(?P<indent>\s*)(?P<lhs>[A-Za-z0-9_]+) = (?P<input>[A-Za-z0-9_]+)\.to\(dtype=torch\.float32\)$"
    )
    forward_unpack_re = re.compile(
        r"^\s*\(?(?P<descriptors>[A-Za-z0-9_]+), (?P<score>[A-Za-z0-9_]+)\)? = self\.(?P<helper_name>[A-Za-z0-9_]+)\(.+\)$"
    )
    forward_packed_re = re.compile(
        r"^\s*(?P<packed>[A-Za-z0-9_]+)(?:\s*:\s*(?:tuple|Tuple|typing\.Tuple)\[torch\.Tensor,\s*torch\.Tensor\])?\s*=\s*self\.(?P<helper_name>[A-Za-z0-9_]+)\(.+\)$"
    )
    forward_index_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\s*=\s*(?P<packed>[A-Za-z0-9_]+)\[(?P<index>[01])\]$"
    )
    stage7_return_re = re.compile(
        r"^\s*return\s+\(?(?P<descriptors>[A-Za-z0-9_]+), (?P<score>[A-Za-z0-9_]+)\)?$"
    )
    stage7_permute_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = _torch_permute\((?P<args>.+)\)$"
    )
    stage7_gather_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.gather\((?P<args>.+)\)$"
    )
    stage7_method_source_expr_pattern = r"(?:[A-Za-z0-9_]+|_align_tensor_to_target_shape\(.+\)|_reshape_gather_output\(.+\))"
    stage7_gather_method_assign_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>{stage7_method_source_expr_pattern})\.gather\((?P<args>.+)\)$"
    )
    stage7_index_select_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.index_select\((?P<args>.+)\)$"
    )
    stage7_index_select_method_assign_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>{stage7_method_source_expr_pattern})\.index_select\((?P<args>.+)\)$"
    )
    stage7_take_along_dim_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.take_along_dim\((?P<args>.+)\)$"
    )
    stage7_take_along_dim_method_assign_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>{stage7_method_source_expr_pattern})\.take_along_dim\((?P<args>.+)\)$"
    )
    stage7_gather_expr_pattern = rf"(?:[A-Za-z0-9_]+|torch\.(?:gather|index_select|take_along_dim)\(.+\)|{stage7_method_source_expr_pattern}\.(?:gather|index_select|take_along_dim)\(.+\)|_reshape_gather_output\(.+\))"
    stage7_gather_reshape_res = [
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\((?P<input>{stage7_gather_expr_pattern}), _resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*,\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{stage7_gather_expr_pattern}), allow_zero=False\)\s*,\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\((?P<input>{stage7_gather_expr_pattern}), {stage7_reshape_shape_pattern}\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*,\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.reshape\(\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*,\s*input\s*=\s*(?P<input>{stage7_gather_expr_pattern})\s*\)$"
        ),
        re.compile(
            rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = (?P<input>{stage7_gather_expr_pattern})\.reshape\({stage7_reshape_shape_pattern}\)$"
        ),
    ]
    stage7_score_squeeze_res = [
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.squeeze\((?P<input>[A-Za-z0-9_]+)\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.squeeze\(\s*input\s*=\s*(?P<input>[A-Za-z0-9_]+)\s*\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = (?P<input>.+?)\.squeeze\(\)$"
        ),
    ]
    stage7_score_mul_res = [
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.mul\((?P<input0>[A-Za-z0-9_]+), (?P<input1>[A-Za-z0-9_]+)\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.mul\(\s*input\s*=\s*(?P<input0>[A-Za-z0-9_]+)\s*,\s*other\s*=\s*(?P<input1>[A-Za-z0-9_]+)\s*\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.mul\(\s*other\s*=\s*(?P<input1>[A-Za-z0-9_]+)\s*,\s*input\s*=\s*(?P<input0>[A-Za-z0-9_]+)\s*\)$"
        ),
    ]
    stage7_permute_dims_patterns = {
        (1, 0): r"(?:\[\s*1\s*,\s*0\s*\]|\(\s*1\s*,\s*0\s*\))",
        (0, 1, 3, 2): r"(?:\[\s*0\s*,\s*1\s*,\s*3\s*,\s*2\s*\]|\(\s*0\s*,\s*1\s*,\s*3\s*,\s*2\s*\))",
    }
    stage7_add_res = [
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.add\((?P<input0>[A-Za-z0-9_\[\]]+), (?P<input1>[A-Za-z0-9_\[\]]+)\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.add\(\s*input\s*=\s*(?P<input0>[A-Za-z0-9_\[\]]+)\s*,\s*other\s*=\s*(?P<input1>[A-Za-z0-9_\[\]]+)\s*\)$"
        ),
        re.compile(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+) = torch\.add\(\s*other\s*=\s*(?P<input1>[A-Za-z0-9_\[\]]+)\s*,\s*input\s*=\s*(?P<input0>[A-Za-z0-9_\[\]]+)\s*\)$"
        ),
    ]
    stage7_add_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.add\((?P<args>.+)\)$"
    )
    stage7_add_method_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>.+?)\.add\((?P<args>.+)\)$"
    )
    stage7_shape_tensor_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = _shape_tensor\((?P<args>.+)\)$"
    )
    stage7_singleton_anchor_assign_re = re.compile(
        r"^\s*(?P<lhs0>[A-Za-z0-9_]+), (?P<lhs1>[A-Za-z0-9_]+) = _align_binary_inputs_to_anchor\((?P<args>.+)\)$"
    )
    stage7_anchor_pair_assign_re = re.compile(
        r"^\s*(?P<pair>[A-Za-z0-9_]+)(?:\s*:\s*(?:tuple|Tuple|typing\.Tuple)\[[^\]]+\])?\s*=\s*_align_binary_inputs_to_anchor\((?P<args>.+)\)$"
    )
    stage7_mul_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*torch\.mul\((?P<args>.+)\)$"
    )
    stage7_mul_method_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<input>.+?)\.mul\((?P<args>.+)\)$"
    )
    stage7_pair_alias_re = re.compile(
        r"^\s*(?P<pair>[A-Za-z0-9_]+)(?:\s*:\s*(?:tuple|Tuple|typing\.Tuple)\[[^\]]+\])?\s*=\s*\(*\s*\(?(?P<rhs0>[A-Za-z0-9_]+)\)?\s*,\s*\(?(?P<rhs1>[A-Za-z0-9_]+)\)?\s*\)*$"
    )
    stage7_pair_rebind_re = re.compile(
        r"^\s*(?P<pair>[A-Za-z0-9_]+)(?:\s*:\s*(?:tuple|Tuple|typing\.Tuple)\[[^\]]+\])?\s*=\s*\(*\s*(?P<source_pair>[A-Za-z0-9_]+)\s*\)*$"
    )
    stage7_pair_unpack_re = re.compile(
        r"^\s*\(?(?P<lhs0>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\)?\s*=\s*\(*\s*(?P<pair>[A-Za-z0-9_]+)\s*\)*$"
    )
    stage7_tuple_alias_re = re.compile(
        r"^\s*\(?(?P<lhs0>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\s*,\s*(?P<lhs1>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\)?\s*=\s*\(?(?P<rhs0>[A-Za-z0-9_]+)\)?\s*,\s*\(?(?P<rhs1>[A-Za-z0-9_]+)\)?$"
    )
    stage7_plain_alias_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?:\s*:\s*torch\.Tensor)?\s*=\s*\(?(?P<rhs>[A-Za-z0-9_]+)\)?$"
    )

    index = 0
    while index < len(lines):
        line = lines[index]
        current_line = str(line)
        next_line = str(lines[index + 1]) if index + 1 < len(lines) else None
        cast_float_match = cast_float_re.match(current_line)
        if cast_float_match is not None and next_line is not None:
            div_floor_match = div_floor_re.match(next_line)
            if div_floor_match is not None and div_floor_match.group("input") == cast_float_match.group("lhs"):
                lines[index] = (
                    f"{cast_float_match.group('indent')}{cast_float_match.group('lhs')} = "
                    f"{cast_float_match.group('input')}.to(dtype=torch.int64)"
                )
                lines[index + 1] = (
                    f"{div_floor_match.group('indent')}{div_floor_match.group('lhs')} = "
                    f"_align_tensor_to_target_shape(torch.div({div_floor_match.group('input')}, "
                    f"{div_floor_match.group('divisor')}, rounding_mode='floor'), [1])"
                )
                changed = True
                index += 2
                continue
        singleton_mul_const_match = singleton_mul_const_align_re.match(current_line)
        if singleton_mul_const_match is not None:
            lines[index] = (
                f"{singleton_mul_const_match.group('indent')}{singleton_mul_const_match.group('lhs')} = "
                f"torch.div({singleton_mul_const_match.group('input')}, self.const_inline_literal_1)"
            )
            changed = True
            index += 1
            continue
        self_square_match = self_square_align_mul_re.match(current_line)
        if self_square_match is not None:
            lines[index] = (
                f"{self_square_match.group('indent')}{self_square_match.group('lhs')} = "
                f"torch.mul({self_square_match.group('input')}, {self_square_match.group('input')})"
            )
            changed = True
            index += 1
            continue
        singleton_binary_anchor_match = singleton_binary_anchor_re.match(current_line)
        if singleton_binary_anchor_match is not None and next_line is not None:
            singleton_mul_align_match = singleton_mul_align_re.match(next_line)
            if (
                singleton_mul_align_match is not None
                and singleton_mul_align_match.group("input0") == singleton_binary_anchor_match.group("lhs1")
                and singleton_mul_align_match.group("input1") == singleton_binary_anchor_match.group("lhs0")
            ):
                lines[index] = (
                    f"{singleton_binary_anchor_match.group('indent')}{singleton_binary_anchor_match.group('lhs0')}, "
                    f"{singleton_binary_anchor_match.group('lhs1')} = "
                    f"{singleton_binary_anchor_match.group('input1')}, {singleton_binary_anchor_match.group('input0')}"
                )
                lines[index + 1] = (
                    f"{singleton_mul_align_match.group('indent')}{singleton_mul_align_match.group('lhs')} = "
                    f"torch.mul({singleton_mul_align_match.group('input0')}, {singleton_mul_align_match.group('input1')})"
                )
                changed = True
                index += 2
                continue
        column_binary_anchor_match = column_binary_anchor_re.match(current_line)
        if column_binary_anchor_match is not None and next_line is not None:
            column_div_align_match = column_div_align_re.match(next_line)
            if (
                column_div_align_match is not None
                and column_div_align_match.group("input0") == column_binary_anchor_match.group("lhs0")
                and column_div_align_match.group("input1") == column_binary_anchor_match.group("lhs1")
                and column_div_align_match.group("n") == column_binary_anchor_match.group("n")
            ):
                lines[index] = (
                    f"{column_binary_anchor_match.group('indent')}{column_binary_anchor_match.group('lhs0')}, "
                    f"{column_binary_anchor_match.group('lhs1')} = "
                    f"{column_binary_anchor_match.group('input0')}, {column_binary_anchor_match.group('input1')}"
                )
                lines[index + 1] = (
                    f"{column_div_align_match.group('indent')}{column_div_align_match.group('lhs')} = "
                    f"torch.div({column_div_align_match.group('input0')}, {column_div_align_match.group('input1')})"
                )
                changed = True
                index += 2
                continue
        rank4_eq_align_match = rank4_eq_align_re.match(current_line)
        if rank4_eq_align_match is not None:
            lines[index] = (
                f"{rank4_eq_align_match.group('indent')}{rank4_eq_align_match.group('lhs')} = "
                f"torch.eq({rank4_eq_align_match.group('a')}, {rank4_eq_align_match.group('b')})"
            )
            changed = True
            index += 1
            continue
        nhwc_to_nchw_match = nhwc_to_nchw_singleton_reshape_re.match(current_line)
        if nhwc_to_nchw_match is not None:
            lines[index] = (
                f"{nhwc_to_nchw_match.group('indent')}{nhwc_to_nchw_match.group('lhs')} = "
                f"torch.reshape({nhwc_to_nchw_match.group('input')}, "
                f"[{nhwc_to_nchw_match.group('n')}, 1, {nhwc_to_nchw_match.group('h')}, {nhwc_to_nchw_match.group('w')}])"
            )
            changed = True
            index += 1
            continue
        singleton_wrapper_match = singleton_rank4_wrapper_re.match(current_line)
        if singleton_wrapper_match is not None:
            lines[index] = (
                f"{singleton_wrapper_match.group('indent')}{singleton_wrapper_match.group('lhs')} = "
                f"{singleton_wrapper_match.group('expr')}"
            )
            changed = True
            index += 1
            continue
        depth_to_space_gather_match = depth_to_space_nhwc_gather_re.match(current_line)
        if depth_to_space_gather_match is not None:
            lhs_name = str(depth_to_space_gather_match.group("lhs"))
            input_name = str(depth_to_space_gather_match.group("input"))
            next_assigns_depth_to_space = (
                next_line is not None
                and re.match(rf"^\s*[A-Za-z0-9_]+\s*=\s*{re.escape(lhs_name)}$", next_line) is not None
                and "_depth_to_space_" in next_line
            )
            if (
                _is_structural_nhwc_name(input_name)
                and not _is_structural_cf_name(input_name)
                and (
                    "depth_to" in lhs_name.lower()
                    or "depthtospace" in lhs_name.lower()
                    or next_assigns_depth_to_space
                )
            ):
                lines[index] = (
                    f"{depth_to_space_gather_match.group('indent')}{lhs_name} = "
                    f"{input_name}[:, :, :, [{depth_to_space_gather_match.group('indices')}]]"
                )
                changed = True
                index += 1
                continue
        index += 1

    stage7_helper_name: str | None = None
    stage7_def_index: int | None = None
    forward_call_index: int | None = None
    packed_forward_calls: Dict[str, Tuple[int, str]] = {}
    packed_forward_indices: Dict[str, Set[str]] = {}
    for index, line in enumerate(lines):
        forward_unpack_match = forward_unpack_re.match(line)
        if forward_unpack_match is not None:
            candidate_helper_name = str(forward_unpack_match.group("helper_name"))
            if candidate_helper_name == "forward":
                continue
            candidate_def_index = next(
                (
                    def_index
                    for def_index, candidate_line in enumerate(lines)
                    if re.match(rf"^\s*def {re.escape(candidate_helper_name)}\(", str(candidate_line)) is not None
                ),
                None,
            )
            if candidate_def_index is None:
                continue
            stage7_helper_name = candidate_helper_name
            stage7_def_index = candidate_def_index
            forward_call_index = index
            break
        forward_packed_match = forward_packed_re.match(line)
        if forward_packed_match is not None:
            packed_forward_calls[str(forward_packed_match.group("packed"))] = (
                index,
                str(forward_packed_match.group("helper_name")),
            )
            continue
        forward_index_match = forward_index_re.match(line)
        if forward_index_match is None:
            continue
        packed_name = str(forward_index_match.group("packed"))
        if packed_name not in packed_forward_calls:
            continue
        packed_forward_indices.setdefault(packed_name, set()).add(
            str(forward_index_match.group("index"))
        )
        if packed_forward_indices[packed_name] != {"0", "1"}:
            continue
        packed_call_index, candidate_helper_name = packed_forward_calls[packed_name]
        if candidate_helper_name == "forward":
            continue
        candidate_def_index = next(
            (
                def_index
                for def_index, candidate_line in enumerate(lines)
                if re.match(rf"^\s*def {re.escape(candidate_helper_name)}\(", str(candidate_line)) is not None
            ),
            None,
        )
        if candidate_def_index is None:
            continue
        stage7_helper_name = candidate_helper_name
        stage7_def_index = candidate_def_index
        forward_call_index = packed_call_index
        break
    if stage7_helper_name is None or stage7_def_index is None or forward_call_index is None:
        if changed:
            model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    stage7_end_index = next(
        (index for index in range(stage7_def_index + 1, len(lines)) if lines[index].startswith("    def ")),
        len(lines),
    )
    stage7_shape_matches: list[tuple[int, dict[str, str | None]]] = []
    stage7_singleton_anchor_matches: list[tuple[int, dict[str, str]]] = []
    stage7_descriptor_matches: list[tuple[int, dict[str, str]]] = []
    stage7_gather_reshape_matches: list[tuple[int, re.Match[str]]] = []
    stage7_gather_reshape_pending: list[tuple[int, str]] = []
    stage7_gather_matches: list[tuple[int, dict[str, str]]] = []
    stage7_reduce_permute_matches: list[tuple[int, dict[str, str]]] = []
    stage7_add_matches: list[tuple[int, dict[str, str]]] = []
    stage7_mul_matches: list[tuple[int, dict[str, str]]] = []
    stage7_return_match: re.Match[str] | None = None
    stage7_return_index: int | None = None
    block_end_index = None
    stage7_alias_sources: Dict[str, str] = {}
    stage7_pair_alias_sources: Dict[str, tuple[str, str]] = {}
    stage7_passthrough_sources: Dict[str, str] = {}

    def _split_stage7_top_level_args(args: str) -> list[str]:
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in args:
            if char == "," and depth == 0:
                part = "".join(current).strip()
                if part:
                    parts.append(part)
                current = []
                continue
            current.append(char)
            if char in "([{":
                depth += 1
            elif char in ")]}" and depth > 0:
                depth -= 1
        tail = "".join(current).strip()
        if tail:
            parts.append(tail)
        return parts

    stage7_anchor_shape_alias_sources: Dict[str, str] = {}

    def _strip_stage7_outer_parentheses(expr: str) -> str:
        stripped_expr = str(expr).strip()
        while stripped_expr.startswith("(") and stripped_expr.endswith(")"):
            depth = 0
            balanced = True
            for index, char in enumerate(stripped_expr):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                    if depth == 0 and index != len(stripped_expr) - 1:
                        balanced = False
                        break
                if depth < 0:
                    balanced = False
                    break
            if not balanced or depth != 0:
                break
            inner_expr = stripped_expr[1:-1].strip()
            if not (
                re.fullmatch(r"[A-Za-z0-9_]+", inner_expr) is not None
                or (inner_expr.startswith("[") and inner_expr.endswith("]"))
                or (inner_expr.startswith("(") and inner_expr.endswith(")"))
            ):
                break
            stripped_expr = inner_expr
        return stripped_expr

    def _resolve_stage7_anchor_target_shape(target_expr: str) -> str:
        resolved_target = _strip_stage7_outer_parentheses(str(target_expr).strip())
        seen_targets: Set[str] = set()
        while (
            resolved_target in stage7_anchor_shape_alias_sources
            and resolved_target not in seen_targets
        ):
            seen_targets.add(resolved_target)
            resolved_target = _strip_stage7_outer_parentheses(
                stage7_anchor_shape_alias_sources[resolved_target]
            )
        return resolved_target

    def _parse_stage7_anchor_shape_alias(line: str) -> tuple[str, str] | None:
        shape_alias_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?:\s*:\s*[^=]+)?\s*=\s*(?P<rhs>.+?)\s*$",
            line,
        )
        if shape_alias_match is None:
            return None
        rhs_value = _strip_stage7_outer_parentheses(str(shape_alias_match.group("rhs")))
        if (
            re.fullmatch(r"(?:\[|\()\s*1\s*,\s*1\s*,\s*1\s*,\s*1\s*(?:\]|\))", rhs_value)
            is None
            and rhs_value not in stage7_anchor_shape_alias_sources
        ):
            return None
        return str(shape_alias_match.group("lhs")), rhs_value

    def _parse_stage7_align_passthrough(line: str) -> tuple[str, str] | None:
        passthrough_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+?)\s*$",
            line,
        )
        if passthrough_match is None:
            return None
        rhs = str(passthrough_match.group("rhs")).strip()
        passthrough_prefix = "_align_tensor_to_target_shape("
        if not rhs.startswith(passthrough_prefix):
            return None
        depth = 1
        closing_index = None
        for index, char in enumerate(rhs[len(passthrough_prefix):], start=len(passthrough_prefix)):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    closing_index = index
                    break
        if closing_index is None or closing_index != len(rhs) - 1:
            return None
        passthrough_args = _split_stage7_top_level_args(
            rhs[len(passthrough_prefix):closing_index]
        )
        if not passthrough_args:
            return None
        return str(passthrough_match.group("lhs")), passthrough_args[0].strip()

    def _parse_stage7_gather_passthrough(line: str) -> tuple[str, str] | None:
        passthrough_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+?)\s*$",
            line,
        )
        if passthrough_match is None:
            return None
        rhs = str(passthrough_match.group("rhs")).strip()
        passthrough_prefix = "_reshape_gather_output("
        if not rhs.startswith(passthrough_prefix):
            return None
        depth = 1
        closing_index = None
        for index, char in enumerate(rhs[len(passthrough_prefix):], start=len(passthrough_prefix)):
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    closing_index = index
                    break
        if closing_index is None or closing_index != len(rhs) - 1:
            return None
        passthrough_args = _split_stage7_top_level_args(
            rhs[len(passthrough_prefix):closing_index]
        )
        if not passthrough_args:
            return None
        return str(passthrough_match.group("lhs")), passthrough_args[0].strip()

    def _parse_stage7_singleton_anchor_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 3:
            rs_value = parts[0].strip()
            tr_value = parts[1].strip()
            target_value = _resolve_stage7_anchor_target_shape(parts[2].strip())
            normalized_tr_value = _strip_stage7_outer_parentheses(tr_value)
            if (
                re.fullmatch(r"(?:\[|\()\s*1\s*,\s*1\s*,\s*1\s*,\s*1\s*(?:\]|\))", target_value)
                is not None
                and (
                    re.fullmatch(r"[A-Za-z0-9_]+", normalized_tr_value) is not None
                    or (
                        normalized_tr_value.startswith("_align_tensor_to_target_shape(")
                        and normalized_tr_value.endswith(")")
                    )
                )
            ):
                return rs_value, tr_value
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        rs_value = keyword_values.get("input0")
        tr_value = keyword_values.get("input1")
        target_value = (
            _resolve_stage7_anchor_target_shape(keyword_values["target_shape"])
            if "target_shape" in keyword_values
            else None
        )
        if (
            rs_value is None
            or tr_value is None
            or target_value is None
            or (
                re.fullmatch(r"[A-Za-z0-9_]+", _strip_stage7_outer_parentheses(tr_value)) is None
                and not (
                    _strip_stage7_outer_parentheses(tr_value).startswith("_align_tensor_to_target_shape(")
                    and _strip_stage7_outer_parentheses(tr_value).endswith(")")
                )
            )
            or re.fullmatch(r"(?:\[|\()\s*1\s*,\s*1\s*,\s*1\s*,\s*1\s*(?:\]|\))", target_value) is None
        ):
            return None
        return rs_value, tr_value

    def _parse_stage7_gather_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            input_expr = parts[0].strip()
            dim_value = parts[1].strip()
            indices_name = parts[2].strip()
            if dim_value == "0" and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None:
                return input_expr, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input_expr = keyword_values.get("input")
        if input_expr is None and parts and "=" not in parts[0]:
            input_expr = parts[0].strip()
        indices_name = keyword_values.get("index")
        dim_value = keyword_values.get("dim")
        if (
            input_expr is None
            or indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_expr, indices_name

    def _parse_stage7_mul_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            return parts[0].strip(), parts[1].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input0 = keyword_values.get("input")
        input1 = keyword_values.get("other")
        if input0 is None or input1 is None:
            return None
        return input0, input1

    def _parse_stage7_method_mul_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 1 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            return input_name, parts[0].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        other_name = keyword_values.get("other")
        if other_name is None:
            return None
        return input_name, other_name

    def _parse_stage7_index_select_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            input_expr = parts[0].strip()
            dim_value = parts[1].strip()
            indices_name = parts[2].strip()
            if dim_value == "0" and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None:
                return input_expr, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input_expr = next(
            (keyword_values[key] for key in ("input", "source") if key in keyword_values),
            None,
        )
        if input_expr is None and parts and "=" not in parts[0]:
            input_expr = parts[0].strip()
        indices_name = next(
            (keyword_values[key] for key in ("index", "indices") if key in keyword_values),
            None,
        )
        dim_value = keyword_values.get("dim")
        if (
            input_expr is None
            or indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_expr, indices_name

    def _parse_stage7_method_gather_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            dim_value = parts[0].strip()
            indices_name = parts[1].strip()
            if dim_value == "0" and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None:
                return input_name, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        indices_name = keyword_values.get("index")
        dim_value = keyword_values.get("dim")
        if (
            indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_name, indices_name

    def _parse_stage7_method_index_select_args(input_name: str, args: str) -> tuple[str, str] | None:
        return _parse_stage7_method_gather_args(input_name, args)

    def _parse_stage7_take_along_dim_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            input_name = parts[0].strip()
            indices_name = parts[1].strip()
            dim_value = parts[2].strip()
            if (
                input_name
                and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None
                and dim_value == "0"
            ):
                return input_name, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input_name = keyword_values.get("input")
        if input_name is None and parts and "=" not in parts[0]:
            input_name = parts[0].strip()
        indices_name = keyword_values.get("indices")
        dim_value = keyword_values.get("dim")
        if (
            input_name is None
            or indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_name, indices_name

    def _parse_stage7_method_take_along_dim_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            indices_name = parts[0].strip()
            dim_value = parts[1].strip()
            if dim_value == "0" and re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is not None:
                return input_name, indices_name
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        indices_name = keyword_values.get("indices")
        dim_value = keyword_values.get("dim")
        if (
            indices_name is None
            or dim_value != "0"
            or re.fullmatch(r"[A-Za-z0-9_]+", indices_name) is None
        ):
            return None
        return input_name, indices_name

    def _parse_stage7_shape_tensor_args(args: str) -> tuple[str, str | None] | None:
        parts = _split_stage7_top_level_args(args)
        input_name: str | None = None
        dtype_value: str | None = None
        device_source: str | None = None
        if parts and re.fullmatch(r"[A-Za-z0-9_]+", parts[0].strip()) is not None:
            input_name = parts[0].strip()
            parts = parts[1:]
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "input":
                input_name = value
            elif key == "dtype":
                dtype_value = value
            elif key == "device":
                device_match = re.fullmatch(r"(?P<source>[A-Za-z0-9_]+)\.device", value)
                if device_match is not None:
                    device_source = str(device_match.group("source"))
        if input_name is None or re.fullmatch(r"[A-Za-z0-9_]+", input_name) is None:
            return None
        if dtype_value != "torch.int32":
            return None
        return input_name, device_source

    def _parse_stage7_permute_input(args: str, expected_dims: tuple[int, ...]) -> str | None:
        dims_pattern = stage7_permute_dims_patterns.get(expected_dims)
        if dims_pattern is None:
            return None
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and re.fullmatch(dims_pattern, parts[1].strip()) is not None:
            return parts[0].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input_value = next(
            (
                keyword_values[key]
                for key in ("input", "x", "tensor")
                if key in keyword_values
            ),
            None,
        )
        dims_value = next(
            (
                keyword_values[key]
                for key in ("perm", "dims", "axes")
                if key in keyword_values
            ),
            None,
        )
        if input_value is None or dims_value is None or re.fullmatch(dims_pattern, dims_value) is None:
            return None
        return input_value

    def _unwrap_stage7_passthrough_expr(expr: str) -> str:
        unwrapped_expr = _resolve_stage7_alias(str(expr).strip())

        def _extract_exact_stage7_passthrough_args(expr_text: str) -> tuple[str, list[str]] | None:
            for passthrough_prefix in ("_align_tensor_to_target_shape(", "_reshape_gather_output("):
                if not expr_text.startswith(passthrough_prefix):
                    continue
                depth = 1
                closing_index = None
                for index, char in enumerate(expr_text[len(passthrough_prefix):], start=len(passthrough_prefix)):
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth == 0:
                            closing_index = index
                            break
                if closing_index is None or closing_index != len(expr_text) - 1:
                    return None
                passthrough_args = _split_stage7_top_level_args(
                    expr_text[len(passthrough_prefix):closing_index]
                )
                if not passthrough_args:
                    return None
                return passthrough_prefix, passthrough_args
            return None

        while True:
            if unwrapped_expr in stage7_passthrough_sources:
                unwrapped_expr = _resolve_stage7_alias(stage7_passthrough_sources[unwrapped_expr])
                continue
            stripped_expr = unwrapped_expr.strip()
            while stripped_expr.startswith("(") and stripped_expr.endswith(")"):
                depth = 0
                balanced = True
                for index, char in enumerate(stripped_expr):
                    if char == "(":
                        depth += 1
                    elif char == ")":
                        depth -= 1
                        if depth < 0 or (depth == 0 and index != len(stripped_expr) - 1):
                            balanced = False
                            break
                if not balanced or depth != 0:
                    break
                stripped_expr = stripped_expr[1:-1].strip()
            passthrough_parse = _extract_exact_stage7_passthrough_args(stripped_expr)
            if passthrough_parse is None:
                return stripped_expr
            _, passthrough_args = passthrough_parse
            unwrapped_expr = _resolve_stage7_alias(passthrough_args[0].strip())

    def _resolve_stage7_pair_index_expr(expr: str) -> str:
        stripped_expr = _strip_stage7_outer_parentheses(str(expr).strip())
        pair_index_match = re.fullmatch(r"(?P<pair>[A-Za-z0-9_]+)\[(?P<index>[01])\]", stripped_expr)
        if pair_index_match is None:
            return stripped_expr
        pair_name = str(pair_index_match.group("pair"))
        if pair_name not in stage7_pair_alias_sources:
            return stripped_expr
        pair_index = int(str(pair_index_match.group("index")))
        return str(stage7_pair_alias_sources[pair_name][pair_index])

    def _resolve_stage7_semantic_expr(expr: str) -> str:
        return _resolve_stage7_pair_index_expr(_unwrap_stage7_passthrough_expr(expr))

    def _parse_stage7_add_args(args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 2 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            return parts[0].strip(), parts[1].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        input0 = keyword_values.get("input")
        input1 = keyword_values.get("other")
        if input0 is None or input1 is None:
            return None
        return input0, input1

    def _parse_stage7_method_add_args(input_name: str, args: str) -> tuple[str, str] | None:
        parts = _split_stage7_top_level_args(args)
        if len(parts) == 1 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
            for part in parts
        ):
            return input_name, parts[0].strip()
        keyword_values: Dict[str, str] = {}
        for part in parts:
            if "=" not in part:
                continue
            key, value = part.split("=", 1)
            keyword_values[key.strip()] = value.strip()
        other_name = keyword_values.get("other")
        if other_name is None:
            return None
        return input_name, other_name

    def _parse_stage7_inline_add_sources(expr: str) -> tuple[str, str] | None:
        expr = _resolve_stage7_pair_index_expr(_unwrap_stage7_passthrough_expr(expr))
        inline_add_match = stage7_add_assign_re.match(f"tmp = {expr}")
        if inline_add_match is not None:
            parsed_add_args = _parse_stage7_add_args(str(inline_add_match.group("args")))
            if parsed_add_args is not None:
                return (
                    _resolve_stage7_semantic_expr(parsed_add_args[0]),
                    _resolve_stage7_semantic_expr(parsed_add_args[1]),
                )
        direct_match = re.match(
            r"^torch\.add\((?P<input0>[A-Za-z0-9_\[\]]+), (?P<input1>[A-Za-z0-9_\[\]]+)\)$",
            expr,
        )
        if direct_match is not None:
            return (
                _resolve_stage7_semantic_expr(str(direct_match.group("input0"))),
                _resolve_stage7_semantic_expr(str(direct_match.group("input1"))),
            )
        keyword_match = re.match(
            r"^torch\.add\(\s*input\s*=\s*(?P<input0>[A-Za-z0-9_\[\]]+)\s*,\s*other\s*=\s*(?P<input1>[A-Za-z0-9_\[\]]+)\s*\)$",
            expr,
        )
        if keyword_match is not None:
            return (
                _resolve_stage7_semantic_expr(str(keyword_match.group("input0"))),
                _resolve_stage7_semantic_expr(str(keyword_match.group("input1"))),
            )
        reordered_keyword_match = re.match(
            r"^torch\.add\(\s*other\s*=\s*(?P<input1>[A-Za-z0-9_\[\]]+)\s*,\s*input\s*=\s*(?P<input0>[A-Za-z0-9_\[\]]+)\s*\)$",
            expr,
        )
        if reordered_keyword_match is not None:
            return (
                _resolve_stage7_semantic_expr(str(reordered_keyword_match.group("input0"))),
                _resolve_stage7_semantic_expr(str(reordered_keyword_match.group("input1"))),
            )
        method_match = re.match(
            r"^(?P<input0>.+?)\.add\((?P<args>.+)\)$",
            expr,
        )
        if method_match is not None:
            parsed_method_add_args = _parse_stage7_method_add_args(
                str(method_match.group("input0")),
                str(method_match.group("args")),
            )
            if parsed_method_add_args is not None:
                return (
                    _resolve_stage7_semantic_expr(parsed_method_add_args[0]),
                    _resolve_stage7_semantic_expr(parsed_method_add_args[1]),
                )
        return None

    if stage7_def_index is not None:
        for scan_index in range(stage7_def_index + 1, stage7_end_index):
            parsed_shape_alias = _parse_stage7_anchor_shape_alias(lines[scan_index])
            if parsed_shape_alias is not None:
                stage7_anchor_shape_alias_sources[parsed_shape_alias[0]] = parsed_shape_alias[1]
            parsed_passthrough_alias = _parse_stage7_align_passthrough(lines[scan_index])
            if parsed_passthrough_alias is not None:
                stage7_passthrough_sources[parsed_passthrough_alias[0]] = parsed_passthrough_alias[1]
            parsed_gather_passthrough_alias = _parse_stage7_gather_passthrough(lines[scan_index])
            if parsed_gather_passthrough_alias is not None:
                stage7_passthrough_sources[parsed_gather_passthrough_alias[0]] = parsed_gather_passthrough_alias[1]
            anchor_pair_match = stage7_anchor_pair_assign_re.match(lines[scan_index])
            if anchor_pair_match is not None:
                anchor_pair_args = _parse_stage7_singleton_anchor_args(str(anchor_pair_match.group("args")))
                if anchor_pair_args is not None:
                    stage7_pair_alias_sources[str(anchor_pair_match.group("pair"))] = (
                        anchor_pair_args[0],
                        anchor_pair_args[1],
                    )
                    stage7_singleton_anchor_matches.append(
                        (
                            scan_index,
                            {
                                "pair": str(anchor_pair_match.group("pair")),
                                "rs": anchor_pair_args[0],
                                "tr": anchor_pair_args[1],
                            },
                        )
                    )
            pair_alias_match = stage7_pair_alias_re.match(lines[scan_index])
            if pair_alias_match is not None:
                stage7_pair_alias_sources[str(pair_alias_match.group("pair"))] = (
                    str(pair_alias_match.group("rhs0")),
                    str(pair_alias_match.group("rhs1")),
                )
            pair_rebind_match = stage7_pair_rebind_re.match(lines[scan_index])
            if pair_rebind_match is not None:
                source_pair_name = str(pair_rebind_match.group("source_pair"))
                if source_pair_name in stage7_pair_alias_sources:
                    stage7_pair_alias_sources[str(pair_rebind_match.group("pair"))] = (
                        stage7_pair_alias_sources[source_pair_name][0],
                        stage7_pair_alias_sources[source_pair_name][1],
                    )
            pair_unpack_match = stage7_pair_unpack_re.match(lines[scan_index])
            if pair_unpack_match is not None:
                pair_name = str(pair_unpack_match.group("pair"))
                if pair_name in stage7_pair_alias_sources:
                    rhs0, rhs1 = stage7_pair_alias_sources[pair_name]
                    stage7_alias_sources[str(pair_unpack_match.group("lhs0"))] = rhs0
                    stage7_alias_sources[str(pair_unpack_match.group("lhs1"))] = rhs1
            tuple_alias_match = stage7_tuple_alias_re.match(lines[scan_index])
            if tuple_alias_match is not None:
                stage7_alias_sources[str(tuple_alias_match.group("lhs0"))] = str(
                    tuple_alias_match.group("rhs0")
                )
                stage7_alias_sources[str(tuple_alias_match.group("lhs1"))] = str(
                    tuple_alias_match.group("rhs1")
                )
            alias_match = stage7_plain_alias_re.match(lines[scan_index])
            if alias_match is not None:
                stage7_alias_sources[str(alias_match.group("lhs"))] = str(alias_match.group("rhs"))
            shape_match = stage7_shape_tensor_assign_re.match(lines[scan_index])
            if shape_match is not None:
                parsed_shape_args = _parse_stage7_shape_tensor_args(str(shape_match.group("args")))
                if parsed_shape_args is not None:
                    stage7_shape_matches.append(
                        (
                            scan_index,
                            {
                                "lhs": str(shape_match.group("lhs")),
                                "input": parsed_shape_args[0],
                                "device_source": parsed_shape_args[1],
                            },
                        )
                    )
            permute_match = stage7_permute_assign_re.match(lines[scan_index])
            if permute_match is not None:
                permute_args = str(permute_match.group("args"))
                descriptor_input = _parse_stage7_permute_input(permute_args, (1, 0))
                if descriptor_input is not None:
                    stage7_descriptor_matches.append(
                        (
                            scan_index,
                            {
                                "lhs": str(permute_match.group("lhs")),
                                "input": descriptor_input,
                            },
                        )
                    )
                reduce_input = _parse_stage7_permute_input(permute_args, (0, 1, 3, 2))
                if reduce_input is not None:
                    stage7_reduce_permute_matches.append(
                        (
                            scan_index,
                            {
                                "lhs": str(permute_match.group("lhs")),
                                "input": reduce_input,
                            },
                        )
                    )
            gather_match = stage7_gather_assign_re.match(lines[scan_index])
            gather_method_match = stage7_gather_method_assign_re.match(lines[scan_index])
            index_select_match = stage7_index_select_assign_re.match(lines[scan_index])
            index_select_method_match = stage7_index_select_method_assign_re.match(lines[scan_index])
            take_along_dim_match = stage7_take_along_dim_assign_re.match(lines[scan_index])
            take_along_dim_method_match = stage7_take_along_dim_method_assign_re.match(lines[scan_index])
            if (
                gather_match is not None
                or gather_method_match is not None
                or index_select_match is not None
                or index_select_method_match is not None
                or take_along_dim_match is not None
                or take_along_dim_method_match is not None
            ):
                parsed_assign_match = (
                    gather_match
                    if gather_match is not None
                    else gather_method_match
                    if gather_method_match is not None
                    else index_select_match
                    if index_select_match is not None
                    else index_select_method_match
                    if index_select_method_match is not None
                    else take_along_dim_match
                    if take_along_dim_match is not None
                    else take_along_dim_method_match
                )
                gather_args = (
                    _parse_stage7_gather_args(str(parsed_assign_match.group("args")))
                    if gather_match is not None
                    else (
                        _parse_stage7_method_gather_args(
                            str(parsed_assign_match.group("input")),
                            str(parsed_assign_match.group("args")),
                        )
                        if gather_method_match is not None
                        else (
                        _parse_stage7_index_select_args(str(parsed_assign_match.group("args")))
                        if index_select_match is not None
                        else (
                            _parse_stage7_method_index_select_args(
                                str(parsed_assign_match.group("input")),
                                str(parsed_assign_match.group("args")),
                            )
                            if index_select_method_match is not None
                            else (
                                _parse_stage7_take_along_dim_args(str(parsed_assign_match.group("args")))
                                if take_along_dim_match is not None
                                else _parse_stage7_method_take_along_dim_args(
                                    str(parsed_assign_match.group("input")),
                                    str(parsed_assign_match.group("args")),
                                )
                            )
                        )
                        )
                    )
                )
                if gather_args is not None:
                    stage7_gather_matches.append(
                        (
                            scan_index,
                            {
                                "lhs": str(parsed_assign_match.group("lhs")),
                                "input": gather_args[0],
                                "indices": gather_args[1],
                            },
                        )
                    )
            gather_reshape_match = next(
                (
                    match
                    for pattern in stage7_gather_reshape_res
                    if (match := pattern.match(lines[scan_index])) is not None
                ),
                None,
            )
            if gather_reshape_match is not None:
                stage7_gather_reshape_matches.append((scan_index, gather_reshape_match))
            elif ".reshape(" in str(lines[scan_index]) or "torch.reshape(" in str(lines[scan_index]):
                stage7_gather_reshape_pending.append((scan_index, str(lines[scan_index])))
            add_assign_match = stage7_add_assign_re.match(lines[scan_index])
            add_method_assign_match = stage7_add_method_assign_re.match(lines[scan_index])
            if add_assign_match is not None or add_method_assign_match is not None:
                parsed_add_args = (
                    _parse_stage7_add_args(str(add_assign_match.group("args")))
                    if add_assign_match is not None
                    else _parse_stage7_method_add_args(
                        str(add_method_assign_match.group("input")),
                        str(add_method_assign_match.group("args")),
                    )
                )
                if parsed_add_args is not None:
                    stage7_add_matches.append(
                        (
                            scan_index,
                            {
                                "lhs": str(
                                    add_assign_match.group("lhs")
                                    if add_assign_match is not None
                                    else add_method_assign_match.group("lhs")
                                ),
                                "input0": parsed_add_args[0],
                                "input1": parsed_add_args[1],
                            },
                        )
                    )
            mul_assign_match = stage7_mul_assign_re.match(lines[scan_index])
            mul_method_assign_match = stage7_mul_method_assign_re.match(lines[scan_index])
            if mul_assign_match is not None or mul_method_assign_match is not None:
                parsed_mul_args = (
                    _parse_stage7_mul_args(str(mul_assign_match.group("args")))
                    if mul_assign_match is not None
                    else _parse_stage7_method_mul_args(
                        str(mul_method_assign_match.group("input")),
                        str(mul_method_assign_match.group("args")),
                    )
                )
                if parsed_mul_args is not None:
                    stage7_mul_matches.append(
                        (
                            scan_index,
                            {
                                "lhs": str(
                                    mul_assign_match.group("lhs")
                                    if mul_assign_match is not None
                                    else mul_method_assign_match.group("lhs")
                                ),
                                "input0": parsed_mul_args[0],
                                "input1": parsed_mul_args[1],
                            },
                        )
                    )
            singleton_anchor_match = stage7_singleton_anchor_assign_re.match(lines[scan_index])
            if singleton_anchor_match is not None:
                singleton_anchor_args = _parse_stage7_singleton_anchor_args(
                    str(singleton_anchor_match.group("args"))
                )
                if singleton_anchor_args is not None:
                    stage7_alias_sources[str(singleton_anchor_match.group("lhs0"))] = str(
                        singleton_anchor_args[0]
                    )
                    stage7_alias_sources[str(singleton_anchor_match.group("lhs1"))] = str(
                        singleton_anchor_args[1]
                    )
                    stage7_singleton_anchor_matches.append(
                        (
                            scan_index,
                            {
                                "lhs0": str(singleton_anchor_match.group("lhs0")),
                                "lhs1": str(singleton_anchor_match.group("lhs1")),
                                "rs": singleton_anchor_args[0],
                                "tr": singleton_anchor_args[1],
                            },
                        )
                    )
            return_match = stage7_return_re.match(lines[scan_index])
            if return_match is not None:
                stage7_return_match = return_match
                stage7_return_index = scan_index
    score_tail_input_expr: str | None = None

    def _resolve_stage7_alias(name: str) -> str:
        current = str(name)
        seen: Set[str] = set()
        while current in stage7_alias_sources and current not in seen:
            seen.add(current)
            current = str(stage7_alias_sources[current])
        return current

    def _resolve_stage7_gather_roles(input_name: str) -> tuple[str | None, str | None]:
        resolved_input_name = _unwrap_stage7_passthrough_expr(str(input_name))
        if resolved_input_name in stage7_gather_sources:
            gather_root_name = stage7_gather_input_roots.get(resolved_input_name)
            if gather_root_name is None:
                gather_root_name = next(
                    (
                        _unwrap_stage7_passthrough_expr(str(match["input"]))
                        for _, match in stage7_gather_matches
                        if str(match["lhs"]) == resolved_input_name
                    ),
                    None,
                )
            return (
                stage7_gather_sources[resolved_input_name],
                gather_root_name,
            )
        if resolved_input_name.startswith("torch.gather(") and resolved_input_name.endswith(")"):
            inline_gather_args = _parse_stage7_gather_args(
                resolved_input_name[len("torch.gather("):-1]
            )
            if inline_gather_args is not None:
                return (
                    _resolve_stage7_alias(inline_gather_args[1]),
                    _unwrap_stage7_passthrough_expr(inline_gather_args[0]),
                )
        if resolved_input_name.startswith("torch.index_select(") and resolved_input_name.endswith(")"):
            inline_index_select_args = _parse_stage7_index_select_args(
                resolved_input_name[len("torch.index_select("):-1]
            )
            if inline_index_select_args is not None:
                return (
                    _resolve_stage7_alias(inline_index_select_args[1]),
                    _unwrap_stage7_passthrough_expr(inline_index_select_args[0]),
                )
        if resolved_input_name.startswith("torch.take_along_dim(") and resolved_input_name.endswith(")"):
            inline_take_along_dim_args = _parse_stage7_take_along_dim_args(
                resolved_input_name[len("torch.take_along_dim("):-1]
            )
            if inline_take_along_dim_args is not None:
                return (
                    _resolve_stage7_alias(inline_take_along_dim_args[1]),
                    _unwrap_stage7_passthrough_expr(inline_take_along_dim_args[0]),
                )
        method_gather_match = re.fullmatch(rf"(?P<input>{stage7_method_source_expr_pattern})\.gather\((?P<args>.+)\)", resolved_input_name)
        if method_gather_match is not None:
            inline_gather_args = _parse_stage7_method_gather_args(
                str(method_gather_match.group("input")),
                str(method_gather_match.group("args")),
            )
            if inline_gather_args is not None:
                return (
                    _resolve_stage7_alias(inline_gather_args[1]),
                    _unwrap_stage7_passthrough_expr(inline_gather_args[0]),
                )
        method_index_select_match = re.fullmatch(rf"(?P<input>{stage7_method_source_expr_pattern})\.index_select\((?P<args>.+)\)", resolved_input_name)
        if method_index_select_match is not None:
            inline_index_select_args = _parse_stage7_method_index_select_args(
                str(method_index_select_match.group("input")),
                str(method_index_select_match.group("args")),
            )
            if inline_index_select_args is not None:
                return (
                    _resolve_stage7_alias(inline_index_select_args[1]),
                    _unwrap_stage7_passthrough_expr(inline_index_select_args[0]),
                )
        method_take_along_dim_match = re.fullmatch(rf"(?P<input>{stage7_method_source_expr_pattern})\.take_along_dim\((?P<args>.+)\)", resolved_input_name)
        if method_take_along_dim_match is not None:
            inline_take_along_dim_args = _parse_stage7_method_take_along_dim_args(
                str(method_take_along_dim_match.group("input")),
                str(method_take_along_dim_match.group("args")),
            )
            if inline_take_along_dim_args is not None:
                return (
                    _resolve_stage7_alias(inline_take_along_dim_args[1]),
                    _unwrap_stage7_passthrough_expr(inline_take_along_dim_args[0]),
                )
        wrapped_method_match = re.fullmatch(
            r"(?P<input>.+?)\.(?P<method>gather|index_select|take_along_dim)\((?P<args>.+)\)",
            resolved_input_name,
        )
        if wrapped_method_match is not None:
            wrapped_input_name = str(wrapped_method_match.group("input")).strip()
            wrapped_method_name = str(wrapped_method_match.group("method"))
            wrapped_method_args = str(wrapped_method_match.group("args"))
            parsed_wrapped_method_args = (
                _parse_stage7_method_gather_args(wrapped_input_name, wrapped_method_args)
                if wrapped_method_name == "gather"
                else (
                    _parse_stage7_method_index_select_args(wrapped_input_name, wrapped_method_args)
                    if wrapped_method_name == "index_select"
                    else _parse_stage7_method_take_along_dim_args(wrapped_input_name, wrapped_method_args)
                )
            )
            if parsed_wrapped_method_args is not None:
                return (
                    _resolve_stage7_alias(parsed_wrapped_method_args[1]),
                    _unwrap_stage7_passthrough_expr(parsed_wrapped_method_args[0]),
                )
        return None, None

    def _parse_stage7_inline_branch_input(expr: str) -> str | None:
        branch_expr_pattern = rf"(?:[A-Za-z0-9_]+|torch\.(?:gather|index_select|take_along_dim)\(.+\)|{stage7_method_source_expr_pattern}\.(?:gather|index_select|take_along_dim)\(.+\)|_reshape_gather_output\(.+\)|_align_tensor_to_target_shape\(.+\))"
        inline_patterns = [
            re.compile(
                rf"^torch\.reshape\((?P<input>{branch_expr_pattern}), _resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*,\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\s*\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\s*,\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*\)$"
            ),
            re.compile(
                rf"^(?P<input>{branch_expr_pattern})\.reshape\(_resolve_reshape_shape\({stage7_reshape_shape_pattern}, {branch_expr_pattern}, allow_zero=False\)\)$"
            ),
            re.compile(
                rf"^torch\.reshape\((?P<input>{branch_expr_pattern}), {stage7_reshape_shape_pattern}\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*,\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*shape\s*=\s*{stage7_reshape_shape_pattern}\s*,\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*\)$"
            ),
            re.compile(
                rf"^(?P<input>{branch_expr_pattern})\.reshape\({stage7_reshape_shape_pattern}\)$"
            ),
        ]
        expr = _resolve_stage7_alias(str(expr).strip())
        matched_inline = next(
            (
                match
                for pattern in inline_patterns
                if (match := pattern.match(expr)) is not None
            ),
            None,
        )
        if matched_inline is None:
            return None
        return str(matched_inline.group("input"))

    def _parse_stage7_inline_branch_candidates(expr: str) -> list[str]:
        resolved_expr = _resolve_stage7_alias(str(expr).strip())
        candidate_names: list[str] = []
        def _append_candidate(candidate_name: str | None) -> None:
            if candidate_name is None:
                return
            stripped_candidate = str(candidate_name).strip()
            if stripped_candidate and stripped_candidate not in candidate_names:
                candidate_names.append(stripped_candidate)
            unwrapped_candidate = _unwrap_stage7_passthrough_expr(stripped_candidate)
            if unwrapped_candidate and unwrapped_candidate not in candidate_names:
                candidate_names.append(unwrapped_candidate)

        inline_branch_input = _parse_stage7_inline_branch_input(resolved_expr)
        _append_candidate(inline_branch_input)
        branch_expr_pattern = rf"(?:[A-Za-z0-9_]+|torch\.(?:gather|index_select|take_along_dim)\(.+\)|{stage7_method_source_expr_pattern}\.(?:gather|index_select|take_along_dim)\(.+\)|_reshape_gather_output\(.+\)|_align_tensor_to_target_shape\(.+\))"
        branch_patterns = [
            re.compile(
                rf"^torch\.reshape\((?P<input>{branch_expr_pattern}), _resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{branch_expr_pattern}), allow_zero=False\)\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*,\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{branch_expr_pattern}), allow_zero=False\)\s*\)$"
            ),
            re.compile(
                rf"^torch\.reshape\(\s*shape\s*=\s*_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{branch_expr_pattern}), allow_zero=False\)\s*,\s*input\s*=\s*(?P<input>{branch_expr_pattern})\s*\)$"
            ),
            re.compile(
                rf"^(?P<input>{branch_expr_pattern})\.reshape\(_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>{branch_expr_pattern}), allow_zero=False\)\)$"
            ),
        ]
        matched_branch = next(
            (
                match
                for pattern in branch_patterns
                if (match := pattern.match(resolved_expr)) is not None
            ),
            None,
        )
        if matched_branch is not None:
            _append_candidate(str(matched_branch.group("input")).strip())
            _append_candidate(str(matched_branch.group("shape_input")).strip())
        return candidate_names

    def _parse_stage7_inline_branch_roles(expr: str) -> tuple[str | None, str | None]:
        for candidate_name in _parse_stage7_inline_branch_candidates(expr):
            resolved_roles = _resolve_stage7_gather_roles(candidate_name)
            if resolved_roles[0] is not None:
                return resolved_roles
        return (None, None)

    class _Stage7ParsedReshapeMatch:
        def __init__(self, lhs: str, input_expr: str) -> None:
            self._groups = {
                "lhs": str(lhs),
                "input": str(input_expr),
            }

        def group(self, name: str) -> str:
            return str(self._groups[name])

    def _parse_stage7_gather_reshape_pending(line: str) -> _Stage7ParsedReshapeMatch | None:
        line_match = re.match(r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+?)\s*$", str(line))
        if line_match is None:
            return None
        rhs = str(line_match.group("rhs")).strip()
        input_expr: str | None = None
        shape_expr: str | None = None
        if rhs.startswith("torch.reshape(") and rhs.endswith(")"):
            reshape_args = _split_stage7_top_level_args(rhs[len("torch.reshape("):-1])
            if len(reshape_args) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
                for part in reshape_args
            ):
                input_expr = reshape_args[0].strip()
                shape_expr = reshape_args[1].strip()
            else:
                keyword_values: Dict[str, str] = {}
                for part in reshape_args:
                    if "=" not in part:
                        continue
                    key, value = part.split("=", 1)
                    keyword_values[key.strip()] = value.strip()
                input_expr = keyword_values.get("input")
                shape_expr = keyword_values.get("shape")
        else:
            method_match = re.match(r"^(?P<input>.+?)\.reshape\((?P<shape>.+)\)$", rhs)
            if method_match is not None:
                input_expr = str(method_match.group("input")).strip()
                shape_expr = str(method_match.group("shape")).strip()
        if input_expr is None or shape_expr is None:
            return None
        if re.fullmatch(stage7_reshape_shape_pattern, shape_expr) is not None:
            if _resolve_stage7_gather_roles(input_expr)[0] is None:
                return None
            return _Stage7ParsedReshapeMatch(str(line_match.group("lhs")), input_expr)
        resolved_shape_match = re.match(
            rf"^_resolve_reshape_shape\({stage7_reshape_shape_pattern}, (?P<shape_input>.+), allow_zero=False\)$",
            shape_expr,
        )
        if resolved_shape_match is None:
            return None
        shape_input_expr = str(resolved_shape_match.group("shape_input")).strip()
        if (
            _resolve_stage7_gather_roles(input_expr)[0] is None
            and _resolve_stage7_gather_roles(shape_input_expr)[0] is None
        ):
            return None
        return _Stage7ParsedReshapeMatch(str(line_match.group("lhs")), input_expr)

    if stage7_gather_reshape_pending:
        for scan_index, pending_line in stage7_gather_reshape_pending:
            parsed_pending_match = _parse_stage7_gather_reshape_pending(pending_line)
            if parsed_pending_match is not None:
                stage7_gather_reshape_matches.append((scan_index, parsed_pending_match))

    def _parse_stage7_score_tail_input(line: str) -> str | None:
        current_line = str(line).strip()
        target_lhs = str(resolved_stage7_return_score_name)
        prefix = f"{target_lhs} = "
        if not current_line.startswith(prefix):
            return None
        rhs = current_line[len(prefix):].strip()
        if rhs.startswith("torch.reshape(") and rhs.endswith(")"):
            reshape_args = _split_stage7_top_level_args(rhs[len("torch.reshape("):-1])
            if len(reshape_args) == 2 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
                for part in reshape_args
            ):
                return reshape_args[0].strip()
            keyword_values: Dict[str, str] = {}
            for part in reshape_args:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                keyword_values[key.strip()] = value.strip()
            return keyword_values.get("input")
        method_match = re.match(r"^(?P<input>.+?)\.reshape\((?P<args>.+)\)$", rhs)
        if method_match is not None:
            return str(method_match.group("input")).strip()
        return None

    def _parse_stage7_score_squeeze_input(line: str, expected_lhs: str) -> str | None:
        current_line = str(line).strip()
        prefix = f"{expected_lhs} = "
        if not current_line.startswith(prefix):
            return None
        rhs = current_line[len(prefix):].strip()
        if rhs.startswith("torch.squeeze(") and rhs.endswith(")"):
            squeeze_args = _split_stage7_top_level_args(rhs[len("torch.squeeze("):-1])
            if len(squeeze_args) == 1 and all(
                re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None
                for part in squeeze_args
            ):
                return squeeze_args[0].strip()
            keyword_values: Dict[str, str] = {}
            for part in squeeze_args:
                if "=" not in part:
                    continue
                key, value = part.split("=", 1)
                keyword_values[key.strip()] = value.strip()
            return keyword_values.get("input")
        method_match = re.match(r"^(?P<input>.+?)\.squeeze\(\)$", rhs)
        if method_match is not None:
            return str(method_match.group("input")).strip()
        return None

    stage7_return_descriptor_name = (
        str(stage7_return_match.group("descriptors")) if stage7_return_match is not None else None
    )
    stage7_return_score_name = (
        str(stage7_return_match.group("score")) if stage7_return_match is not None else None
    )
    resolved_stage7_return_descriptor_name = (
        _resolve_stage7_alias(stage7_return_descriptor_name)
        if stage7_return_descriptor_name is not None
        else None
    )
    resolved_stage7_return_score_name = (
        _resolve_stage7_alias(stage7_return_score_name)
        if stage7_return_score_name is not None
        else None
    )
    if stage7_return_match is not None and stage7_return_index is not None:
        for scan_index in range(stage7_return_index - 1, stage7_def_index, -1):
            parsed_score_tail_input = _parse_stage7_score_tail_input(lines[scan_index])
            if parsed_score_tail_input is not None:
                score_tail_input_expr = parsed_score_tail_input
                block_end_index = scan_index
                break

    def _resolve_stage7_anchor_sources(match: dict[str, str]) -> tuple[str, str]:
        return (
            _unwrap_stage7_passthrough_expr(str(match["rs"])),
            _unwrap_stage7_passthrough_expr(str(match["tr"])),
        )

    def _stage7_anchor_output_names(match: dict[str, str]) -> tuple[str, str] | None:
        lhs0 = match.get("lhs0")
        lhs1 = match.get("lhs1")
        if lhs0 is None or lhs1 is None:
            return None
        return str(lhs0), str(lhs1)

    stage7_gather_sources = {
        str(match["lhs"]): _resolve_stage7_alias(str(match["indices"]))
        for _, match in stage7_gather_matches
    }
    stage7_gather_input_roots = {
        str(match["lhs"]): _unwrap_stage7_passthrough_expr(str(match["input"]))
        for _, match in stage7_gather_matches
    }
    stage7_rs_sources = {
        str(match.group("lhs")): _resolve_stage7_gather_roles(str(match.group("input")))[0]
        for _, match in stage7_gather_reshape_matches
    }
    stage7_branch_rs_names = {
        str(match.group("lhs"))
        for _, match in stage7_gather_reshape_matches
    }

    def _resolve_stage7_branch_add_name(rs_expr: str) -> str | None:
        resolved_rs_expr = _unwrap_stage7_passthrough_expr(str(rs_expr))
        mapped_add_name = stage7_rs_sources.get(resolved_rs_expr)
        if mapped_add_name is not None:
            return mapped_add_name
        inline_add_name, _ = _parse_stage7_inline_branch_roles(str(rs_expr))
        if inline_add_name is not None:
            return str(inline_add_name)
        gather_input_expr = next(
            (
                str(match.group("input"))
                for _, match in stage7_gather_reshape_matches
                if str(match.group("lhs")) == resolved_rs_expr
            ),
            None,
        )
        if gather_input_expr is None:
            return None
        add_name, _ = _resolve_stage7_gather_roles(gather_input_expr)
        return add_name

    def _derive_stage7_branch_mul_pair(match: dict[str, str]) -> tuple[str, str] | None:
        resolved_input0 = _resolve_stage7_alias(str(match["input0"]))
        resolved_input1 = _resolve_stage7_alias(str(match["input1"]))
        semantic_input0 = _unwrap_stage7_passthrough_expr(str(match["input0"]))
        semantic_input1 = _unwrap_stage7_passthrough_expr(str(match["input1"]))
        if (
            semantic_input0 in stage7_branch_rs_names
            and semantic_input1 in stage7_param_names
        ):
            return stage7_rs_sources[semantic_input0], semantic_input1
        if (
            semantic_input1 in stage7_branch_rs_names
            and semantic_input0 in stage7_param_names
        ):
            return stage7_rs_sources[semantic_input1], semantic_input0
        inline_roles0 = _parse_stage7_inline_branch_roles(str(match["input0"]))
        if inline_roles0[0] is not None and semantic_input1 in stage7_param_names:
            return str(inline_roles0[0]), semantic_input1
        inline_roles1 = _parse_stage7_inline_branch_roles(str(match["input1"]))
        if inline_roles1[0] is not None and semantic_input0 in stage7_param_names:
            return str(inline_roles1[0]), semantic_input0
        direct_candidate_add0 = next(
            (
                stage7_gather_sources[candidate_name]
                for candidate_name in _parse_stage7_inline_branch_candidates(str(match["input0"]))
                if candidate_name in stage7_gather_sources
            ),
            None,
        )
        if direct_candidate_add0 is not None and semantic_input1 in stage7_param_names:
            return str(direct_candidate_add0), semantic_input1
        direct_candidate_add1 = next(
            (
                stage7_gather_sources[candidate_name]
                for candidate_name in _parse_stage7_inline_branch_candidates(str(match["input1"]))
                if candidate_name in stage7_gather_sources
            ),
            None,
        )
        if direct_candidate_add1 is not None and semantic_input0 in stage7_param_names:
            return str(direct_candidate_add1), semantic_input0
        inline_branch_input0 = _parse_stage7_inline_branch_input(str(match["input0"]))
        if inline_branch_input0 is not None and semantic_input1 in stage7_param_names:
            add_name0, _ = _resolve_stage7_gather_roles(inline_branch_input0)
            if add_name0 is not None:
                return str(add_name0), semantic_input1
        inline_branch_input1 = _parse_stage7_inline_branch_input(str(match["input1"]))
        if inline_branch_input1 is not None and semantic_input0 in stage7_param_names:
            add_name1, _ = _resolve_stage7_gather_roles(inline_branch_input1)
            if add_name1 is not None:
                return str(add_name1), semantic_input0
        return None

    detected_add_names = [
        _resolve_stage7_alias(str(match["input"]))
        for _, match in stage7_shape_matches
    ]
    stage7_shape_matches_by_index = {
        scan_index: match
        for scan_index, match in stage7_shape_matches
    }
    structural_fallback_add_names: list[str] = []
    if stage7_shape_matches:
        next_shape_index = min(stage7_shape_matches_by_index)
        while next_shape_index in stage7_shape_matches_by_index:
            fallback_add_name = _resolve_stage7_alias(
                str(stage7_shape_matches_by_index[next_shape_index]["input"])
            )
            if fallback_add_name not in structural_fallback_add_names:
                structural_fallback_add_names.append(fallback_add_name)
            next_shape_index += 1
    stage7_param_names = re.findall(r"([A-Za-z0-9_]+): torch\.Tensor", lines[stage7_def_index])
    detected_tr_names = [
        _unwrap_stage7_passthrough_expr(str(match["tr"]))
        for _, match in stage7_singleton_anchor_matches
        if (
            _resolve_stage7_branch_add_name(str(match["rs"])) is not None
        )
    ]
    stage7_branch_mul_pairs = {
        branch_pair[0]: branch_pair[1]
        for _, match in stage7_mul_matches
        for branch_pair in [_derive_stage7_branch_mul_pair(match)]
        if branch_pair is not None
    }
    detected_tr_names.extend(
        tr_name
        for _, tr_name in stage7_branch_mul_pairs.items()
        if tr_name not in detected_tr_names
    )
    covered_stage7_branch_count = max(
        len(stage7_gather_reshape_matches),
        len(stage7_branch_mul_pairs),
    )
    if len(detected_tr_names) < len(stage7_gather_reshape_matches):
        fallback_detected_tr_names = [
            _unwrap_stage7_passthrough_expr(str(match["tr"]))
            for _, match in stage7_singleton_anchor_matches[: len(stage7_gather_reshape_matches)]
        ]
        detected_tr_names = (
            detected_tr_names
            + [
                tr_name
                for tr_name in fallback_detected_tr_names
                if tr_name not in detected_tr_names
            ]
        )[: len(stage7_gather_reshape_matches)]
    descriptor_input_name = next(
        (
            str(match["input"])
            for _, match in stage7_descriptor_matches
            if (
                resolved_stage7_return_descriptor_name is not None
                and str(match["lhs"]) == resolved_stage7_return_descriptor_name
            )
        ),
        None,
    )
    provisional_branch_count = min(
        len(detected_add_names),
        len(detected_tr_names),
        covered_stage7_branch_count,
    )
    descriptor_param_index = (
        stage7_param_names.index(descriptor_input_name)
        if descriptor_input_name in stage7_param_names
        else None
    )
    pre_descriptor_params = (
        stage7_param_names[:descriptor_param_index]
        if descriptor_param_index is not None
        else stage7_param_names
    )
    post_descriptor_params = (
        stage7_param_names[descriptor_param_index + 1 :]
        if descriptor_param_index is not None
        else stage7_param_names
    )
    branch_tr_param_names = set(
        post_descriptor_params[:-1]
        if len(post_descriptor_params) >= 2
        else post_descriptor_params
    )
    if branch_tr_param_names:
        stage7_branch_mul_pairs = {
            add_name: tr_name
            for add_name, tr_name in stage7_branch_mul_pairs.items()
            if tr_name in branch_tr_param_names
        }
        detected_tr_names = [
            tr_name
            for tr_name in detected_tr_names
            if tr_name in branch_tr_param_names
        ]
        detected_tr_names.extend(
            tr_name
            for _, tr_name in stage7_branch_mul_pairs.items()
            if tr_name not in detected_tr_names
        )
    if (
        not detected_tr_names
        and not stage7_gather_reshape_matches
        and not stage7_branch_mul_pairs
        and (structural_fallback_add_names or detected_add_names)
        and len(post_descriptor_params) >= len((structural_fallback_add_names or detected_add_names)) + 1
    ):
        detected_add_names = structural_fallback_add_names or detected_add_names
        detected_tr_names = list(post_descriptor_params[: len(detected_add_names)])
        covered_stage7_branch_count = max(covered_stage7_branch_count, len(detected_add_names))
    structural_fallback_tr_names = (
        list(post_descriptor_params[: len(structural_fallback_add_names)])
        if structural_fallback_add_names
        else []
    )
    ordered_add_names = [
        param_name
        for param_name in pre_descriptor_params
        if param_name in detected_add_names
    ]
    if provisional_branch_count >= 1 and len(ordered_add_names) > provisional_branch_count:
        ordered_add_names = ordered_add_names[-provisional_branch_count:]
    ordered_tr_names = [
        param_name
        for param_name in post_descriptor_params
        if param_name in detected_tr_names
    ]
    if provisional_branch_count >= 1 and len(ordered_tr_names) > provisional_branch_count:
        ordered_tr_names = ordered_tr_names[:provisional_branch_count]
    add_names = ordered_add_names or detected_add_names
    tr_names = ordered_tr_names or detected_tr_names
    gather_input_root_order = [
        root_name
        for _, match in stage7_gather_matches
        for root_name in [_unwrap_stage7_passthrough_expr(str(match["input"]))]
        if root_name is not None
    ]
    gather_input_root_order.extend(
        [
        root_name
        for _, match in stage7_gather_reshape_matches
        for _, root_name in [_resolve_stage7_gather_roles(str(match.group("input")))]
        if root_name is not None
        ]
    )
    gather_input_root_counts: Dict[str, int] = {}
    for _, match in stage7_gather_matches:
        gather_root_name = _unwrap_stage7_passthrough_expr(str(match["input"]))
        if gather_root_name is not None:
            gather_input_root_counts[gather_root_name] = gather_input_root_counts.get(gather_root_name, 0) + 1
    for _, match in stage7_gather_reshape_matches:
        _, gather_root_name = _resolve_stage7_gather_roles(str(match.group("input")))
        if gather_root_name is not None:
            gather_input_root_counts[gather_root_name] = gather_input_root_counts.get(gather_root_name, 0) + 1
    for _, match in stage7_singleton_anchor_matches:
        _, inline_gather_root_name = _parse_stage7_inline_branch_roles(str(match["rs"]))
        if inline_gather_root_name is not None:
            gather_input_root_order.append(inline_gather_root_name)
            gather_input_root_counts[inline_gather_root_name] = (
                gather_input_root_counts.get(inline_gather_root_name, 0) + 1
            )
    for _, match in stage7_mul_matches:
        for inline_gather_root_name in [
            _parse_stage7_inline_branch_roles(str(match["input0"]))[1],
            _parse_stage7_inline_branch_roles(str(match["input1"]))[1],
        ]:
            if inline_gather_root_name is not None:
                gather_input_root_order.append(inline_gather_root_name)
                gather_input_root_counts[inline_gather_root_name] = (
                    gather_input_root_counts.get(inline_gather_root_name, 0) + 1
                )

    def _rank_stage7_gather_roots(root_names: Set[str]) -> list[str]:
        return sorted(
            root_names,
            key=lambda root_name: (
                -gather_input_root_counts.get(root_name, 0),
                0
                if (
                    root_name not in stage7_param_names
                    and root_name != descriptor_input_name
                    and root_name not in detected_add_names
                    and root_name not in detected_tr_names
                )
                else 1,
                gather_input_root_order.index(root_name),
            ),
        )

    def _parse_stage7_forward_call_arg_names(line: str) -> list[str]:
        stage7_call_match = re.search(
            rf"self\.{re.escape(stage7_helper_name)}\((?P<args>.*)\)\s*$",
            line,
        )
        if stage7_call_match is None:
            return []
        arg_names: list[str] = []
        for raw_arg in _split_stage7_top_level_args(str(stage7_call_match.group("args"))):
            arg_expr = raw_arg.strip()
            if "=" in arg_expr:
                _, arg_expr = arg_expr.split("=", 1)
                arg_expr = arg_expr.strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", arg_expr) is not None:
                arg_names.append(arg_expr)
        return arg_names

    def _infer_stage7_score_map_name_from_forward_scope() -> str | None:
        if forward_call_index is None:
            return None
        forward_call_line = lines[forward_call_index]
        call_arg_names = set(_parse_stage7_forward_call_arg_names(forward_call_line))
        forward_unpack_match = forward_unpack_re.match(forward_call_line)
        descriptor_lhs_name = (
            str(forward_unpack_match.group("descriptors"))
            if forward_unpack_match is not None
            else None
        )
        score_lhs_name = (
            str(forward_unpack_match.group("score"))
            if forward_unpack_match is not None
            else None
        )
        forward_end_index = next(
            (
                index
                for index in range(forward_call_index + 1, len(lines))
                if lines[index].startswith("    def ")
            ),
            len(lines),
        )
        if descriptor_lhs_name is not None and score_lhs_name is not None:
            for scan_index in range(forward_call_index + 1, forward_end_index):
                return_match = re.match(r"^\s*return\s+\(?(?P<values>.+?)\)?\s*$", lines[scan_index])
                if return_match is None:
                    continue
                return_values = [
                    value
                    for value in _split_stage7_top_level_args(str(return_match.group("values")))
                    if re.fullmatch(r"[A-Za-z0-9_]+", value) is not None
                ]
                extra_return_values = [
                    value
                    for value in return_values
                    if value not in {descriptor_lhs_name, score_lhs_name}
                ]
                if extra_return_values:
                    return extra_return_values[-1]
        forward_def_index = next(
            (
                index
                for index in range(forward_call_index, -1, -1)
                if lines[index].startswith("    def forward(")
            ),
            None,
        )
        if forward_def_index is None:
            return None
        forward_param_names = re.findall(
            r"([A-Za-z0-9_]+): torch\.Tensor",
            lines[forward_def_index],
        )
        fallback_param_names = [
            param_name
            for param_name in forward_param_names
            if param_name not in call_arg_names
            and param_name not in {descriptor_lhs_name, score_lhs_name}
        ]
        return fallback_param_names[0] if fallback_param_names else None

    gather_input_roots = set(gather_input_root_order)
    preferred_stage7_gather_root = next(iter(_rank_stage7_gather_roots(gather_input_roots)), None)
    inferred_stage7_score_map_name = _infer_stage7_score_map_name_from_forward_scope()
    stage7_score_map_name = preferred_stage7_gather_root or inferred_stage7_score_map_name or "scores_map"
    raw_stage7_branch_pairs = [
        (add_name, tr_name)
        for _, match in stage7_gather_reshape_matches
        for add_name in [stage7_rs_sources.get(str(match.group("lhs")))]
        if add_name is not None
        for tr_name in [
            next(
                (
                    resolved_tr
                    for _, anchor_match in stage7_singleton_anchor_matches
                    if _unwrap_stage7_passthrough_expr(str(anchor_match["rs"])) == str(match.group("lhs"))
                    for resolved_tr in [_unwrap_stage7_passthrough_expr(str(anchor_match["tr"]))]
                ),
                stage7_branch_mul_pairs.get(add_name),
            )
        ]
        if tr_name is not None
    ]
    raw_stage7_branch_pairs.extend(
        (add_name, _unwrap_stage7_passthrough_expr(str(match["tr"])))
        for _, match in stage7_singleton_anchor_matches
        for add_name in [_resolve_stage7_branch_add_name(str(match["rs"]))]
        if add_name is not None
        and (add_name, _unwrap_stage7_passthrough_expr(str(match["tr"]))) not in raw_stage7_branch_pairs
    )
    raw_stage7_branch_pairs.extend(
        (add_name, _unwrap_stage7_passthrough_expr(str(match["tr"])))
        for _, match in stage7_singleton_anchor_matches
        for add_name, _ in [_parse_stage7_inline_branch_roles(str(match["rs"]))]
        if add_name is not None
    )
    raw_stage7_branch_pairs.extend(
        (add_name, tr_name)
        for add_name, tr_name in stage7_branch_mul_pairs.items()
        if (add_name, tr_name) not in raw_stage7_branch_pairs
    )
    if (
        not raw_stage7_branch_pairs
        and len(stage7_singleton_anchor_matches) <= 1
        and structural_fallback_add_names
        and len(structural_fallback_tr_names) >= len(structural_fallback_add_names)
    ):
        raw_stage7_branch_pairs.extend(
            (add_name, tr_name)
            for add_name, tr_name in zip(structural_fallback_add_names, structural_fallback_tr_names)
        )
    elif (
        not raw_stage7_branch_pairs
        and not stage7_gather_reshape_matches
        and add_names
        and tr_names
    ):
        raw_stage7_branch_pairs.extend(
            (add_name, tr_name)
            for add_name, tr_name in zip(add_names, tr_names)
        )
    if not raw_stage7_branch_pairs and stage7_singleton_anchor_matches:
        raw_stage7_branch_pairs.extend(
            (add_name, _unwrap_stage7_passthrough_expr(str(match["tr"])))
            for _, match in stage7_singleton_anchor_matches
            for add_name in [
                stage7_rs_sources.get(_unwrap_stage7_passthrough_expr(str(match["rs"])))
            ]
            if add_name is not None
        )
    deduped_stage7_branch_pairs: list[tuple[str, str]] = []
    for branch_pair in raw_stage7_branch_pairs:
        if branch_pair not in deduped_stage7_branch_pairs:
            deduped_stage7_branch_pairs.append(branch_pair)
    raw_stage7_branch_pairs = deduped_stage7_branch_pairs
    topology_ordered_add_names: list[str] = []
    topology_ordered_tr_names: list[str] = []
    for add_name, tr_name in raw_stage7_branch_pairs:
        if add_name not in topology_ordered_add_names:
            topology_ordered_add_names.append(add_name)
        if tr_name not in topology_ordered_tr_names:
            topology_ordered_tr_names.append(tr_name)
    if topology_ordered_add_names:
        add_names = topology_ordered_add_names
    if topology_ordered_tr_names:
        tr_names = topology_ordered_tr_names
    stage7_branch_pairs = raw_stage7_branch_pairs
    stage7_add_sources = {
        str(match["lhs"]): (
            _resolve_stage7_semantic_expr(str(match["input0"])),
            _resolve_stage7_semantic_expr(str(match["input1"])),
        )
        for _, match in stage7_add_matches
    }
    stage7_mul_sources = {
        str(match["lhs"]): (
            _unwrap_stage7_passthrough_expr(str(match["input0"])),
            _unwrap_stage7_passthrough_expr(str(match["input1"])),
        )
        for _, match in stage7_mul_matches
    }
    stage7_reduce_permute_sources = {
        str(match["lhs"]): _resolve_stage7_alias(str(match["input"]))
        for _, match in stage7_reduce_permute_matches
    }
    if (
        stage7_def_index is None
        or forward_call_index is None
        or block_end_index is None
        or stage7_return_match is None
        or block_end_index <= stage7_def_index
    ):
        if changed:
            model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    score_lhs = (
        resolved_stage7_return_score_name
        if resolved_stage7_return_score_name is not None
        else str(stage7_return_match.group("score"))
    )
    score_cast_name = next(
        (
            resolved_name
            for _, match in reversed(stage7_singleton_anchor_matches)
            for resolved_name in _resolve_stage7_anchor_sources(match)
            if resolved_name in stage7_param_names
            and resolved_name not in add_names
            and resolved_name not in tr_names
            and resolved_name != descriptor_input_name
            and resolved_name != stage7_score_map_name
        ),
        stage7_param_names[-1] if stage7_param_names else "wadkd_cast5_out0",
    )
    if score_cast_name == stage7_score_map_name and len(stage7_param_names) >= 2:
        score_cast_name = stage7_param_names[-2]
    if stage7_return_index is not None:
        final_anchor_param_from_text = next(
            (
                resolved_name
                for scan_index in range(stage7_return_index - 1, stage7_def_index, -1)
                for raw_match in [
                    stage7_singleton_anchor_assign_re.match(lines[scan_index]),
                    stage7_anchor_pair_assign_re.match(lines[scan_index]),
                ]
                if raw_match is not None
                for parsed_sources in [_parse_stage7_singleton_anchor_args(str(raw_match.group("args")))]
                if parsed_sources is not None
                for resolved_name in (
                    _unwrap_stage7_passthrough_expr(parsed_sources[0]),
                    _unwrap_stage7_passthrough_expr(parsed_sources[1]),
                )
                if resolved_name in stage7_param_names
                and resolved_name not in add_names
                and resolved_name not in tr_names
                and resolved_name != descriptor_input_name
                and resolved_name != stage7_score_map_name
            ),
            None,
        )
        if final_anchor_param_from_text is not None:
            score_cast_name = final_anchor_param_from_text
    raw_score_mul_input_set: Set[str] = set()
    if score_tail_input_expr is not None and block_end_index is not None:
        score_tail_input_name = _unwrap_stage7_passthrough_expr(str(score_tail_input_expr))
        score_mul_lhs_name = next(
            (
                parsed_input_name
                for scan_index in range(block_end_index - 1, stage7_def_index, -1)
                for parsed_input_name in [
                    _parse_stage7_score_squeeze_input(
                        lines[scan_index],
                        score_tail_input_name,
                    )
                ]
                if parsed_input_name is not None
            ),
            score_tail_input_name,
        )
        score_mul_lhs_name = _unwrap_stage7_passthrough_expr(score_mul_lhs_name)
        if score_mul_lhs_name is not None:
            score_mul_inputs = next(
                (
                    parsed_mul_args
                    for scan_index in range(block_end_index - 1, stage7_def_index, -1)
                    for raw_mul_match, parsed_mul_args in [
                        (
                            stage7_mul_assign_re.match(lines[scan_index]),
                            None,
                        ),
                        (
                            stage7_mul_method_assign_re.match(lines[scan_index]),
                            None,
                        ),
                    ]
                    if raw_mul_match is not None
                    and str(raw_mul_match.group("lhs")) == score_mul_lhs_name
                    for parsed_mul_args in [
                        _parse_stage7_mul_args(str(raw_mul_match.group("args")))
                        if raw_mul_match.re is stage7_mul_assign_re
                        else _parse_stage7_method_mul_args(
                            str(raw_mul_match.group("input")),
                            str(raw_mul_match.group("args")),
                        )
                    ]
                    if parsed_mul_args is not None
                ),
                None,
            )
            if score_mul_inputs is not None:
                raw_score_mul_input_set = {score_mul_inputs[0], score_mul_inputs[1]}
                score_mul_input_set = {
                    _resolve_stage7_semantic_expr(score_mul_inputs[0]),
                    _resolve_stage7_semantic_expr(score_mul_inputs[1]),
                }
                score_cast_candidate = next(
                    (
                        resolved_name
                        for _, match in reversed(stage7_singleton_anchor_matches)
                        if (
                            _stage7_anchor_output_names(match) is not None
                            and set(_stage7_anchor_output_names(match) or ()) == raw_score_mul_input_set
                        )
                        or set(_resolve_stage7_anchor_sources(match)) == score_mul_input_set
                        for resolved_name in _resolve_stage7_anchor_sources(match)
                        if resolved_name in stage7_param_names
                        and resolved_name not in add_names
                        and resolved_name not in tr_names
                        and resolved_name != descriptor_input_name
                    ),
                    None,
                )
                if score_cast_candidate is not None:
                    score_cast_name = score_cast_candidate
    if raw_score_mul_input_set:
        final_score_anchor_match = next(
            (
                match
                for _, match in reversed(stage7_singleton_anchor_matches)
                for output_names in [_stage7_anchor_output_names(match)]
                if output_names is not None and set(output_names) == raw_score_mul_input_set
            ),
            None,
        )
        if final_score_anchor_match is not None:
            explicit_score_cast_candidate = next(
                (
                    resolved_name
                    for resolved_name in _resolve_stage7_anchor_sources(final_score_anchor_match)
                    if resolved_name in stage7_param_names
                    and resolved_name not in add_names
                    and resolved_name not in tr_names
                    and resolved_name != descriptor_input_name
                ),
                None,
            )
            if explicit_score_cast_candidate is not None:
                score_cast_name = explicit_score_cast_candidate
    final_stage7_aggregate_root = None
    if score_tail_input_expr is not None and block_end_index is not None:
        score_tail_input_name = _unwrap_stage7_passthrough_expr(str(score_tail_input_expr))
        score_mul_lhs_name = next(
            (
                parsed_input_name
                for scan_index in range(block_end_index - 1, stage7_def_index, -1)
                for parsed_input_name in [
                    _parse_stage7_score_squeeze_input(
                        lines[scan_index],
                        score_tail_input_name,
                    )
                ]
                if parsed_input_name is not None
            ),
            score_tail_input_name,
        )
        score_mul_lhs_name = _unwrap_stage7_passthrough_expr(score_mul_lhs_name)
        if score_mul_lhs_name is not None:
            score_mul_inputs = next(
                (
                    parsed_mul_args
                    for scan_index in range(block_end_index - 1, stage7_def_index, -1)
                    for raw_mul_match, parsed_mul_args in [
                        (
                            stage7_mul_assign_re.match(lines[scan_index]),
                            None,
                        ),
                        (
                            stage7_mul_method_assign_re.match(lines[scan_index]),
                            None,
                        ),
                    ]
                    if raw_mul_match is not None
                    and str(raw_mul_match.group("lhs")) == score_mul_lhs_name
                    for parsed_mul_args in [
                        _parse_stage7_mul_args(str(raw_mul_match.group("args")))
                        if raw_mul_match.re is stage7_mul_assign_re
                        else _parse_stage7_method_mul_args(
                            str(raw_mul_match.group("input")),
                            str(raw_mul_match.group("args")),
                        )
                    ]
                    if parsed_mul_args is not None
                ),
                None,
            )
            if score_mul_inputs is not None:
                resolved_final_inputs = (
                    _resolve_stage7_semantic_expr(score_mul_inputs[0]),
                    _resolve_stage7_semantic_expr(score_mul_inputs[1]),
                )
                final_stage7_aggregate_root = next(
                    (
                        input_name
                        for input_name in resolved_final_inputs
                        if input_name != score_cast_name
                    ),
                    None,
                )
                if final_stage7_aggregate_root in stage7_reduce_permute_sources:
                    final_stage7_aggregate_root = stage7_reduce_permute_sources[final_stage7_aggregate_root]
    if final_stage7_aggregate_root is None and score_cast_name is not None:
        final_stage7_aggregate_root = next(
            (
                candidate_name
                for _, match in reversed(stage7_singleton_anchor_matches)
                for candidate_name in _resolve_stage7_anchor_sources(match)
                if candidate_name != score_cast_name
                and score_cast_name in _resolve_stage7_anchor_sources(match)
            ),
            None,
        )
        if final_stage7_aggregate_root in stage7_reduce_permute_sources:
            final_stage7_aggregate_root = stage7_reduce_permute_sources[final_stage7_aggregate_root]
    if final_stage7_aggregate_root is not None:
        final_anchor_score_cast_candidate = next(
            (
                resolved_name
                for _, match in reversed(stage7_singleton_anchor_matches)
                for resolved_sources in [_resolve_stage7_anchor_sources(match)]
                if (
                    final_stage7_aggregate_root in resolved_sources
                    or any(
                        source_name in stage7_reduce_permute_sources
                        and stage7_reduce_permute_sources[source_name] == final_stage7_aggregate_root
                        for source_name in resolved_sources
                    )
                )
                for resolved_name in resolved_sources
                if resolved_name in stage7_param_names
                and resolved_name not in add_names
                and resolved_name not in tr_names
                and resolved_name != descriptor_input_name
                and resolved_name != final_stage7_aggregate_root
            ),
            None,
        )
        if final_anchor_score_cast_candidate is not None:
            score_cast_name = final_anchor_score_cast_candidate

    stage7_mul_branch_pairs_by_lhs = {
        str(match["lhs"]): branch_pair
        for _, match in stage7_mul_matches
        for branch_pair in [_derive_stage7_branch_mul_pair(match)]
        if branch_pair is not None
    }

    def _collect_stage7_mul_leaves(root_name: str | None) -> Set[str]:
        if root_name is None:
            return set()
        stack = [str(root_name)]
        seen: Set[str] = set()
        result: Set[str] = set()
        while stack:
            current_name = stack.pop()
            resolved_current_name = _resolve_stage7_alias(current_name)
            if resolved_current_name in seen:
                continue
            seen.add(resolved_current_name)
            passthrough_current_name = _unwrap_stage7_passthrough_expr(resolved_current_name)
            if passthrough_current_name != resolved_current_name:
                stack.append(passthrough_current_name)
                continue
            if resolved_current_name in stage7_add_sources:
                stack.extend(stage7_add_sources[resolved_current_name])
                continue
            inline_add_sources = _parse_stage7_inline_add_sources(resolved_current_name)
            if inline_add_sources is not None:
                stack.extend(inline_add_sources)
                continue
            if resolved_current_name in stage7_mul_sources:
                result.add(resolved_current_name)
        return result

    def _collect_stage7_branch_pair_leaves(root_name: str | None) -> list[tuple[str, str]]:
        if root_name is None:
            return []
        stack = [str(root_name)]
        seen: Set[str] = set()
        result: list[tuple[str, str]] = []
        seen_pairs: Set[tuple[str, str]] = set()
        while stack:
            current_name = stack.pop()
            resolved_current_name = _resolve_stage7_alias(current_name)
            if resolved_current_name in seen:
                continue
            seen.add(resolved_current_name)
            passthrough_current_name = _unwrap_stage7_passthrough_expr(resolved_current_name)
            if passthrough_current_name != resolved_current_name:
                stack.append(passthrough_current_name)
                continue
            if resolved_current_name in stage7_add_sources:
                stack.extend(stage7_add_sources[resolved_current_name])
                continue
            inline_add_sources = _parse_stage7_inline_add_sources(resolved_current_name)
            if inline_add_sources is not None:
                stack.extend(inline_add_sources)
                continue
            branch_pair = stage7_mul_branch_pairs_by_lhs.get(resolved_current_name)
            if branch_pair is None and resolved_current_name in stage7_branch_rs_names:
                mapped_add_name = stage7_rs_sources.get(resolved_current_name)
                if mapped_add_name is not None:
                    mapped_tr_name = next(
                        (
                            resolved_tr_name
                            for _, anchor_match in stage7_singleton_anchor_matches
                            if _unwrap_stage7_passthrough_expr(str(anchor_match["rs"])) == resolved_current_name
                            for resolved_tr_name in [_unwrap_stage7_passthrough_expr(str(anchor_match["tr"]))]
                        ),
                        stage7_branch_mul_pairs.get(mapped_add_name),
                    )
                    if mapped_tr_name is not None:
                        branch_pair = (mapped_add_name, mapped_tr_name)
            if branch_pair is not None and branch_pair not in seen_pairs:
                seen_pairs.add(branch_pair)
                result.append(branch_pair)
        return result

    final_stage7_mul_leaves = _collect_stage7_mul_leaves(final_stage7_aggregate_root)
    if final_stage7_aggregate_root is not None and not final_stage7_mul_leaves:
        unwrapped_final_stage7_aggregate_root = _unwrap_stage7_passthrough_expr(
            final_stage7_aggregate_root
        )
        if unwrapped_final_stage7_aggregate_root != final_stage7_aggregate_root:
            final_stage7_mul_leaves = _collect_stage7_mul_leaves(
                unwrapped_final_stage7_aggregate_root
            )
    final_stage7_branch_pair_leaves = _collect_stage7_branch_pair_leaves(final_stage7_aggregate_root)
    if final_stage7_aggregate_root is not None and not final_stage7_branch_pair_leaves:
        unwrapped_final_stage7_aggregate_root = _unwrap_stage7_passthrough_expr(
            final_stage7_aggregate_root
        )
        if unwrapped_final_stage7_aggregate_root != final_stage7_aggregate_root:
            final_stage7_branch_pair_leaves = _collect_stage7_branch_pair_leaves(
                unwrapped_final_stage7_aggregate_root
            )

    def _stage7_branch_pair_input_set(add_name: str, tr_name: str) -> Set[str] | None:
        rs_name = next(
            (
                rs_name
                for rs_name, mapped_add_name in stage7_rs_sources.items()
                if mapped_add_name == add_name
            ),
            None,
        )
        if rs_name is not None:
            return {_resolve_stage7_alias(rs_name), _unwrap_stage7_passthrough_expr(tr_name)}
        direct_mul_input_set = next(
            (
                {
                    _unwrap_stage7_passthrough_expr(str(match["input0"])),
                    _unwrap_stage7_passthrough_expr(str(match["input1"])),
                }
                for _, match in stage7_mul_matches
                for branch_pair in [(
                    stage7_branch_mul_pairs.get(add_name),
                    _unwrap_stage7_passthrough_expr(tr_name),
                )]
                if branch_pair[0] == branch_pair[1]
                and (
                    _parse_stage7_inline_branch_roles(str(match["input0"]))[0] == add_name
                    or _parse_stage7_inline_branch_roles(str(match["input1"]))[0] == add_name
                )
            ),
            None,
        )
        if direct_mul_input_set is not None:
            return direct_mul_input_set
        return next(
            (
                {
                    _unwrap_stage7_passthrough_expr(str(match["rs"])),
                    _unwrap_stage7_passthrough_expr(str(match["tr"])),
                }
                for _, match in stage7_singleton_anchor_matches
                for inline_add_name, _ in [_parse_stage7_inline_branch_roles(str(match["rs"]))]
                if inline_add_name == add_name
                and _unwrap_stage7_passthrough_expr(str(match["tr"])) == _unwrap_stage7_passthrough_expr(tr_name)
            ),
            None,
        )

    if final_stage7_mul_leaves:
        constrained_branch_pairs = []
        for add_name, tr_name in raw_stage7_branch_pairs:
            branch_input_set = _stage7_branch_pair_input_set(add_name, tr_name)
            if branch_input_set is None:
                continue
            if any(set(stage7_mul_sources[mul_name]) == branch_input_set for mul_name in final_stage7_mul_leaves):
                constrained_branch_pairs.append((add_name, tr_name))
        if len(constrained_branch_pairs) < len(final_stage7_mul_leaves):
            constrained_branch_pairs = [
                branch_pair
                for _, match in stage7_mul_matches
                for branch_pair in [stage7_mul_branch_pairs_by_lhs.get(str(match["lhs"]))]
                if branch_pair is not None
                and str(match["lhs"]) in final_stage7_mul_leaves
            ]
        if raw_stage7_branch_pairs:
            constrained_branch_pairs = [
                branch_pair
                for branch_pair in constrained_branch_pairs
                if branch_pair in raw_stage7_branch_pairs
            ]
        if constrained_branch_pairs:
            stage7_branch_pairs = constrained_branch_pairs
    elif final_stage7_branch_pair_leaves:
        constrained_branch_pairs = [
            branch_pair
            for branch_pair in raw_stage7_branch_pairs
            if branch_pair in final_stage7_branch_pair_leaves
        ]
        if not constrained_branch_pairs:
            constrained_branch_pairs = [
                branch_pair
                for branch_pair in final_stage7_branch_pair_leaves
                if branch_pair in raw_stage7_branch_pairs
            ]
        if constrained_branch_pairs:
            stage7_branch_pairs = constrained_branch_pairs

    def _stage7_branch_mul_match(add_name: str, tr_name: str) -> dict[str, str] | None:
        return next(
            (
                match
                for _, match in stage7_mul_matches
                for branch_pair in [_derive_stage7_branch_mul_pair(match)]
                if branch_pair == (add_name, tr_name)
            ),
            None,
        )

    def _stage7_branch_pair_gather_root(add_name: str, tr_name: str) -> str | None:
        rs_name = next(
            (
                rs_name
                for rs_name, mapped_add_name in stage7_rs_sources.items()
                if mapped_add_name == add_name
            ),
            None,
        )
        if rs_name is not None:
            gather_input_name = next(
                (
                    str(match.group("input"))
                    for _, match in stage7_gather_reshape_matches
                    if str(match.group("lhs")) == rs_name
                ),
                rs_name,
            )
            return _resolve_stage7_gather_roles(gather_input_name)[1]
        direct_mul_match = _stage7_branch_mul_match(add_name, tr_name)
        direct_mul_gather_root = next(
            (
                inline_roles[1]
                for input_name in (
                    str(direct_mul_match["input0"]),
                    str(direct_mul_match["input1"]),
                )
                for inline_roles in [_parse_stage7_inline_branch_roles(input_name)]
                if inline_roles[0] == add_name and inline_roles[1] is not None
            ),
            None,
        ) if direct_mul_match is not None else None
        if direct_mul_gather_root is None and direct_mul_match is not None:
            direct_mul_gather_root = next(
                (
                    _resolve_stage7_gather_roles(resolved_input_name)[1]
                    for input_name in (
                        str(direct_mul_match["input0"]),
                        str(direct_mul_match["input1"]),
                    )
                    for resolved_input_name in [_resolve_stage7_alias(input_name)]
                    if resolved_input_name in stage7_branch_rs_names
                ),
                None,
            )
        if direct_mul_gather_root is not None:
            return direct_mul_gather_root
        return next(
            (
                inline_gather_root_name
                for _, match in stage7_singleton_anchor_matches
                if _unwrap_stage7_passthrough_expr(str(match["tr"])) == _unwrap_stage7_passthrough_expr(tr_name)
                for inline_add_name, inline_gather_root_name in [
                    _parse_stage7_inline_branch_roles(str(match["rs"]))
                ]
                if inline_add_name == add_name and inline_gather_root_name is not None
            ),
            None,
        )

    if stage7_branch_pairs:
        branch_pair_gather_roots = {
            gather_root_name
            for add_name, tr_name in stage7_branch_pairs
            for gather_root_name in [_stage7_branch_pair_gather_root(add_name, tr_name)]
            if gather_root_name is not None
        }
        if branch_pair_gather_roots:
            preferred_stage7_gather_root = next(
                iter(_rank_stage7_gather_roots(branch_pair_gather_roots)),
                preferred_stage7_gather_root,
            )
            stage7_branch_pairs = [
                (add_name, tr_name)
                for add_name, tr_name in stage7_branch_pairs
                for branch_gather_root_name in [_stage7_branch_pair_gather_root(add_name, tr_name)]
                if branch_gather_root_name is None
                or (
                    preferred_stage7_gather_root is None
                    or branch_gather_root_name == preferred_stage7_gather_root
                )
            ]
    stage7_score_map_name = preferred_stage7_gather_root or inferred_stage7_score_map_name or "scores_map"
    if f"{stage7_score_map_name}: torch.Tensor" not in lines[stage7_def_index]:
        if "->" in lines[stage7_def_index]:
            lines[stage7_def_index] = re.sub(
                r"\)\s*->",
                f", {stage7_score_map_name}: torch.Tensor) ->",
                lines[stage7_def_index],
                count=1,
            )
        else:
            lines[stage7_def_index] = re.sub(
                r"\)\s*:$",
                f", {stage7_score_map_name}: torch.Tensor):",
                lines[stage7_def_index],
                count=1,
            )
        changed = True
    forward_call_line = lines[forward_call_index]
    score_map_forward_arg = f"{stage7_score_map_name}={stage7_score_map_name}"
    if (
        f"{stage7_score_map_name})" not in forward_call_line
        and score_map_forward_arg not in forward_call_line
    ):
        stage7_call_match = re.search(
            rf"self\.{re.escape(stage7_helper_name)}\((?P<args>.*)\)\s*$",
            forward_call_line,
        )
        stage7_call_args = str(stage7_call_match.group("args")) if stage7_call_match is not None else ""
        has_keyword_call_args = "=" in stage7_call_args
        appended_forward_arg = (
            score_map_forward_arg
            if has_keyword_call_args
            else stage7_score_map_name
        )
        lines[forward_call_index] = re.sub(
            r"\)$",
            f", {appended_forward_arg})",
            forward_call_line,
            count=1,
        )
        changed = True
    if stage7_branch_pairs:
        add_names = [add_name for add_name, _ in stage7_branch_pairs]
        tr_names = [tr_name for _, tr_name in stage7_branch_pairs]
    branch_count = min(len(add_names), len(tr_names))
    relevant_add_names = set(add_names[:branch_count])
    relevant_shape_matches = [
        (scan_index, match)
        for scan_index, match in stage7_shape_matches
        if _resolve_stage7_alias(str(match["input"])) in relevant_add_names
    ]
    relevant_gather_matches = [
        (scan_index, match)
        for scan_index, match in stage7_gather_matches
        if _resolve_stage7_alias(str(match["indices"])) in relevant_add_names
        and (
            preferred_stage7_gather_root is None
            or stage7_gather_input_roots.get(str(match["lhs"])) == preferred_stage7_gather_root
        )
    ]
    relevant_rs_names = {
        str(match.group("lhs"))
        for _, match in stage7_gather_reshape_matches
        if stage7_rs_sources.get(str(match.group("lhs"))) in relevant_add_names
        and (
            preferred_stage7_gather_root is None
            or _resolve_stage7_gather_roles(str(match.group("input")))[1] == preferred_stage7_gather_root
        )
    }
    relevant_gather_reshape_matches = [
        (scan_index, match)
        for scan_index, match in stage7_gather_reshape_matches
        if str(match.group("lhs")) in relevant_rs_names
    ]
    relevant_mul_matches = [
        (scan_index, match)
        for scan_index, match in stage7_mul_matches
        for branch_pair in [_derive_stage7_branch_mul_pair(match)]
        if branch_pair is not None
        and branch_pair[0] in relevant_add_names
        and branch_pair[1] in set(tr_names[:branch_count])
        and (
            preferred_stage7_gather_root is None
            or _parse_stage7_inline_branch_roles(str(match["input0"]))[1] == preferred_stage7_gather_root
            or _parse_stage7_inline_branch_roles(str(match["input1"]))[1] == preferred_stage7_gather_root
            or (
                _resolve_stage7_alias(str(match["input0"])) in relevant_rs_names
                or _resolve_stage7_alias(str(match["input1"])) in relevant_rs_names
            )
        )
    ]
    relevant_anchor_matches = [
        (scan_index, match)
        for scan_index, match in stage7_singleton_anchor_matches
        if (
            _unwrap_stage7_passthrough_expr(str(match["rs"])) in relevant_rs_names
            or _resolve_stage7_branch_add_name(str(match["rs"])) in relevant_add_names
        )
        and _unwrap_stage7_passthrough_expr(str(match["tr"])) in set(tr_names[:branch_count])
        and (
            preferred_stage7_gather_root is None
            or _parse_stage7_inline_branch_roles(str(match["rs"]))[1] == preferred_stage7_gather_root
            or _stage7_branch_pair_gather_root(
                str(_resolve_stage7_branch_add_name(str(match["rs"]))),
                _unwrap_stage7_passthrough_expr(str(match["tr"])),
            ) == preferred_stage7_gather_root
        )
    ]
    relevant_start_indices = [
        scan_index
        for scan_index, _ in (
            relevant_shape_matches
            + relevant_gather_matches
            + relevant_gather_reshape_matches
            + relevant_mul_matches
            + relevant_anchor_matches
        )
    ]
    if not relevant_start_indices and stage7_branch_pairs:
        relevant_start_indices = [
            scan_index
            for scan_index, match in stage7_mul_matches
            for branch_pair in [_derive_stage7_branch_mul_pair(match)]
            if branch_pair in set(stage7_branch_pairs)
        ]
    block_start_index = min(relevant_start_indices) if relevant_start_indices else None
    if (
        block_start_index is None
        or block_end_index < block_start_index
        or branch_count < 1
    ):
        if changed:
            model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return
    add_names = add_names[:branch_count]
    tr_names = tr_names[:branch_count]
    temp_prefix = re.sub(r"_+", "_", re.sub(r"[^A-Za-z0-9_]+", "_", score_lhs)).strip("_")
    if not temp_prefix:
        temp_prefix = "stage7"
    if temp_prefix[0].isdigit():
        temp_prefix = f"stage7_{temp_prefix}"
    temp_prefix = f"{temp_prefix}_stage7"

    flatten_name = f"{temp_prefix}_flatten_out0"
    shape_names = [f"{temp_prefix}_shape{index}_out0" for index in range(branch_count)]
    shape_prefix_name = f"{temp_prefix}_shape_prefix_out0"
    gather_negative_names = [f"{temp_prefix}_gather{index}_is_negative" for index in range(branch_count)]
    gather_wrapped_names = [f"{temp_prefix}_gather{index}_wrapped_runtime" for index in range(branch_count)]
    gather_index_names = [f"{temp_prefix}_gather{index}_indices" for index in range(branch_count)]
    gather_names = [f"{temp_prefix}_gather{index}_out0" for index in range(branch_count)]
    rs_names = [f"{temp_prefix}_rs{index}_out0" for index in range(branch_count)]
    concat_names = [f"{temp_prefix}_concat{index}_out0" for index in range(branch_count)]
    tr_stage_names = [f"{temp_prefix}_tr{index}_out0" for index in range(branch_count)]
    rs_shape_dim0_names = [f"{temp_prefix}_rs_shape_dim0_{index}" for index in range(branch_count)]
    rs_in_shape_names = [f"{temp_prefix}_rs_in_shape_{index}" for index in range(branch_count)]
    rs_in_dim0_names = [f"{temp_prefix}_rs_in_dim0_{index}" for index in range(branch_count)]
    rs_shape_dim0_is_zero_names = [f"{temp_prefix}_rs_shape_dim0_is_zero_{index}" for index in range(branch_count)]
    rs_shape_dim0_fixed_names = [f"{temp_prefix}_rs_shape_dim0_fixed_{index}" for index in range(branch_count)]
    rs_shape_tail_names = [f"{temp_prefix}_rs_shape_tail_{index}" for index in range(branch_count)]
    rs_shape_fixed_names = [f"{temp_prefix}_rs_shape_fixed_{index}" for index in range(branch_count)]
    rs_fixed_names = [f"{temp_prefix}_rs_fixed_{index}" for index in range(branch_count)]
    mul_names = [f"{temp_prefix}_mul{index}_out0" for index in range(branch_count)]
    add_accum_names = [f"{temp_prefix}_add{index}_out0" for index in range(max(0, branch_count - 1))]
    tr14_name = f"{temp_prefix}_tr14_out0"
    mul44_name = f"{temp_prefix}_mul44_out0"
    squeeze_name = f"{temp_prefix}_squeeze_out0"

    indent = "        "
    replacement_block = [
        f"{indent}{flatten_name} = torch.reshape({stage7_score_map_name}, _resolve_reshape_shape([-1, 1], {stage7_score_map_name}, allow_zero=False))",
    ]
    for branch_index in range(branch_count):
        replacement_block.extend(
            [
                f"{indent}{shape_names[branch_index]} = _shape_tensor({add_names[branch_index]}, dtype=torch.int32, device={add_names[branch_index]}.device)",
            ]
        )
    replacement_block.append(
        f"{indent}{shape_prefix_name} = torch.ones([1], dtype=torch.int32, device={shape_names[0]}.device)"
    )
    for branch_index in range(branch_count):
        replacement_block.extend(
            [
                f"{indent}{gather_negative_names[branch_index]} = _align_tensor_to_target_shape(torch.lt({add_names[branch_index]}, 0), _tensor_shape_list({add_names[branch_index]}))",
                f"{indent}{gather_wrapped_names[branch_index]} = _align_tensor_to_target_shape(torch.add({add_names[branch_index]}, _tensor_shape_list({flatten_name})[0]), _tensor_shape_list({add_names[branch_index]}))",
                f"{indent}{gather_index_names[branch_index]} = torch.where({gather_negative_names[branch_index]}, {gather_wrapped_names[branch_index]}, {add_names[branch_index]})",
                f"{indent}{gather_names[branch_index]} = _reshape_gather_output(torch.index_select({flatten_name}, 0, {gather_index_names[branch_index]}.to(dtype=torch.int64).reshape(-1)), {flatten_name}, _shape_tensor({gather_index_names[branch_index]}, dtype=torch.int64, device={gather_index_names[branch_index]}.device), axis=0)",
                f"{indent}{rs_names[branch_index]} = torch.reshape({gather_names[branch_index]}, _resolve_reshape_shape([-1, 1], {gather_names[branch_index]}, allow_zero=False))",
                f"{indent}{concat_names[branch_index]} = _apply_concat([{shape_prefix_name}, {shape_names[branch_index]}], axis=0, target_shape=[4], fused='NONE')",
                f"{indent}{tr_stage_names[branch_index]} = _torch_permute({rs_names[branch_index]}, [1, 0])",
                f"{indent}{rs_shape_dim0_names[branch_index]} = {concat_names[branch_index]}.reshape(-1)[:1]",
                f"{indent}{rs_in_shape_names[branch_index]} = _shape_tensor({tr_stage_names[branch_index]}, dtype=torch.int32, device={tr_stage_names[branch_index]}.device)",
                f"{indent}{rs_in_dim0_names[branch_index]} = {rs_in_shape_names[branch_index]}.reshape(-1)[:1]",
                f"{indent}{rs_shape_dim0_is_zero_names[branch_index]} = _align_tensor_to_target_shape(torch.eq({rs_shape_dim0_names[branch_index]}, 0), [1])",
                f"{indent}{rs_shape_dim0_fixed_names[branch_index]} = torch.where({rs_shape_dim0_is_zero_names[branch_index]}, {rs_in_dim0_names[branch_index]}, {rs_shape_dim0_names[branch_index]})",
                f"{indent}{rs_shape_tail_names[branch_index]} = {concat_names[branch_index]}.reshape(-1)[1:]",
                f"{indent}{rs_shape_fixed_names[branch_index]} = _apply_concat([{rs_shape_dim0_fixed_names[branch_index]}, {rs_shape_tail_names[branch_index]}], axis=0, target_shape=[4], fused='NONE')",
                f"{indent}{rs_fixed_names[branch_index]} = torch.reshape({tr_stage_names[branch_index]}, _shape_list(_resolve_reshape_shape_tensor({rs_shape_fixed_names[branch_index]}, {tr_stage_names[branch_index]}, allow_zero=False)))",
                f"{indent}{mul_names[branch_index]} = torch.mul({rs_fixed_names[branch_index]}, {tr_names[branch_index]})",
            ]
        )
    aggregate_name = mul_names[0]
    for branch_index in range(1, branch_count):
        aggregate_name = add_accum_names[branch_index - 1]
        replacement_block.append(
            f"{indent}{aggregate_name} = torch.add({mul_names[0] if branch_index == 1 else add_accum_names[branch_index - 2]}, {mul_names[branch_index]})"
        )
    replacement_block.extend(
        [
            f"{indent}{tr14_name} = _torch_permute({aggregate_name}, [0, 1, 3, 2])",
            f"{indent}{mul44_name} = torch.mul({tr14_name}, {score_cast_name})",
            f"{indent}{squeeze_name} = torch.squeeze({mul44_name})",
            f"{indent}{score_lhs} = torch.reshape({squeeze_name}, _resolve_reshape_shape([-1, 1], {squeeze_name}, allow_zero=False))",
        ]
    )
    lines[block_start_index : block_end_index + 1] = replacement_block
    changed = True

    if changed:
        model_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_SHADOWFORMER_PERMUTE_0213_ARGS_PATTERN = (
    r"(?:"
    r"\*\(\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\)"
    r"|\*\[\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\]"
    r"|0\s*,\s*2\s*,\s*1\s*,\s*3"
    r"|\(\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\)"
    r"|\[\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\]"
    r"|dims\s*=\s*\(\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\)"
    r"|dims\s*=\s*\[\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\]"
    r")"
)
_SHADOWFORMER_FUNCTIONAL_PERMUTE_0213_ARGS_PATTERN = (
    r"(?:"
    r"\(\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\)"
    r"|\[\s*0\s*,\s*2\s*,\s*1\s*,\s*3\s*\]"
    r")"
)
_SHADOWFORMER_COPY_PERMUTE_SRC_RE = re.compile(
    rf"^\(?(?:.+\.permute\({_SHADOWFORMER_PERMUTE_0213_ARGS_PATTERN}\)|torch\.permute\((?:input\s*=\s*)?.+?,\s*(?:dims\s*=\s*)?{_SHADOWFORMER_FUNCTIONAL_PERMUTE_0213_ARGS_PATTERN}\))(?:\.contiguous\([^)]*\))?\)?$"
)
_SHADOWFORMER_METHOD_COPY_PERMUTE_SRC_RE = re.compile(
    rf"^\(?(?P<src>.+?)\.permute\({_SHADOWFORMER_PERMUTE_0213_ARGS_PATTERN}\)(?:\.contiguous\([^)]*\))?\)?$"
)
_SHADOWFORMER_FUNCTIONAL_COPY_PERMUTE_SRC_RE = re.compile(
    rf"^\(?torch\.permute\((?:input\s*=\s*)?(?P<src>.+?),\s*(?:dims\s*=\s*)?{_SHADOWFORMER_FUNCTIONAL_PERMUTE_0213_ARGS_PATTERN}\)(?:\.contiguous\([^)]*\))?\)?$"
)
_SHADOWFORMER_REGISTER_BUFFER_RE = re.compile(
    r"^\s*self\.register_buffer\((?P<name_kw>name\s*=\s*)?(?P<quote>['\"])(?P<buffer>[A-Za-z0-9_]+)(?P=quote),\s*"
    r"(?P<tensor_kw>tensor\s*=\s*)?torch\.zeros\((?P<zeros_prefix>(?:[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^,()]+(?:\([^)]*\))?\s*,\s*)*)(?:size\s*=\s*)?(?:\[|\()\s*1\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)\s*(?:\]|\))"
    r"(?P<zeros_kwargs>(?:,\s*[A-Za-z_][A-Za-z0-9_]*\s*=\s*[^,()]+(?:\([^)]*\))?)*)"
    r"(?P<zeros_trailing_comma>\s*,?)\)"
    r"(?:,\s*persistent\s*=\s*(?P<persistent>True|False))?"
    r"(?P<register_trailing_comma>\s*,?)\)$"
)
def _collect_shadowformer_local_aliases(lines: Sequence[str]) -> Dict[str, str]:
    alias_re = re.compile(
        r"^\s*(?P<alias>[A-Za-z0-9_]+)(?:\s*:\s*[^=]+)?\s*=\s*\(?\s*(?P<source>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\)?\s*$"
    )
    pair_alias_re = re.compile(
        r"^\s*\(?\s*(?P<lhs_alias>[A-Za-z0-9_]+)\s*,\s*(?P<rhs_alias>[A-Za-z0-9_]+)\s*\)?\s*=\s*\(?\s*\(?\s*(?P<lhs_source>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\)?\s*,\s*\(?\s*(?P<rhs_source>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\)?\s*\)?\s*\)?\s*$"
    )
    aliases: Dict[str, str] = {}
    for line in lines:
        current_line = str(line)
        pair_alias_match = pair_alias_re.match(current_line)
        if pair_alias_match is not None:
            lhs_source_name = str(pair_alias_match.group("lhs_source"))
            rhs_source_name = str(pair_alias_match.group("rhs_source"))
            aliases[str(pair_alias_match.group("lhs_alias"))] = aliases.get(lhs_source_name, lhs_source_name)
            aliases[str(pair_alias_match.group("rhs_alias"))] = aliases.get(rhs_source_name, rhs_source_name)
            continue
        alias_match = alias_re.match(current_line)
        if alias_match is None:
            continue
        alias_name = str(alias_match.group("alias"))
        source_name = str(alias_match.group("source"))
        aliases[alias_name] = aliases.get(source_name, source_name)
    resolved_aliases: Dict[str, str] = {}
    for alias_name in aliases:
        resolved_name = alias_name
        seen_names: Set[str] = set()
        while resolved_name in aliases and resolved_name not in seen_names:
            seen_names.add(resolved_name)
            next_name = aliases[resolved_name]
            if next_name == resolved_name:
                break
            resolved_name = next_name
        resolved_aliases[alias_name] = resolved_name
    return resolved_aliases


def _collect_shadowformer_buffer_aliases(lines: Sequence[str]) -> Dict[str, str]:
    buffer_aliases: Dict[str, str] = {}
    for alias_name, source_name in _collect_shadowformer_local_aliases(lines).items():
        if source_name.startswith("self."):
            buffer_aliases[alias_name] = source_name[len("self.") :]
    return buffer_aliases


def _apply_shadowformer_fast_precanonicalize_repairs(model_path: Path) -> None:
    if not model_path.exists():
        return
    model_source = model_path.read_text(encoding="utf-8")
    model_lines = model_source.splitlines()
    (
        registered_shapes,
        buffer_shapes,
        copied_buffers,
        copied_shapes,
        aligned_shapes,
        buffer_aligned_buffers,
        buffer_aligned_shapes,
    ) = _collect_shadowformer_fast_repair_facts(model_lines)
    if not _has_shadowformer_fast_repair_signature(model_lines):
        return

    changed = False
    lines = model_lines
    local_aliases = _collect_shadowformer_local_aliases(lines)
    buffer_aliases = _collect_shadowformer_buffer_aliases(lines)
    known_shadowformer_shapes = _collect_shadowformer_supported_shapes(
        registered_shapes,
        copied_shapes,
        aligned_shapes,
        buffer_aligned_shapes,
    )
    supported_buffers = {
        buffer_name
        for buffer_name, buffer_shape in buffer_shapes.items()
        if (
            buffer_shape in known_shadowformer_shapes
            and (
                buffer_name in copied_buffers
                or buffer_name in buffer_aligned_buffers
            )
        )
    }
    binary_shape_re = re.compile(
        r"^(?P<indent>\s*(?P<out_lhs>[A-Za-z0-9_]+),\s*(?P<out_rhs>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs(?:_to_anchor)?\(\(?\s*[A-Za-z0-9_\.]+\s*\)?,\s*\(?\s*[A-Za-z0-9_\.]+\s*\)?,\s*(?:\[|\())"
        rf"(?P<batch>{_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN})\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)(?P<suffix>(?:\]|\))\))$"
    )
    mul_align_shape_re = re.compile(
        r"^(?P<indent>\s*[A-Za-z0-9_]+\s*=\s*_align_tensor_to_target_shape\(torch\.mul\(\(?\s*(?P<mul_lhs>[A-Za-z0-9_]+)\s*\)?,\s*\(?\s*(?P<mul_rhs>[A-Za-z0-9_]+)\s*\)?\),\s*(?:\[|\())"
        rf"(?P<batch>{_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN})\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)(?P<suffix>(?:\]|\))\))$"
    )
    binary_output_pairs: Set[frozenset[str]] = set()
    for line in lines:
        binary_match = binary_shape_re.match(str(line))
        if binary_match is None:
            continue
        dims = [
            int(binary_match.group("d1")),
            int(binary_match.group("d2")),
            int(binary_match.group("d3")),
        ]
        inferred_shape = _infer_shadowformer_shape_from_dims(dims, known_shadowformer_shapes)
        if inferred_shape is None:
            continue
        binary_output_pairs.add(
            frozenset(
                {
                    str(binary_match.group("out_lhs")),
                    str(binary_match.group("out_rhs")),
                }
            )
        )

    for index, line in enumerate(lines):
        current_line = str(line)
        register_match = _SHADOWFORMER_REGISTER_BUFFER_RE.match(current_line)
        if register_match is not None:
            if register_match.group("buffer") not in supported_buffers:
                continue
            dims = (
                int(register_match.group("d1")),
                int(register_match.group("d2")),
                int(register_match.group("d3")),
            )
            if dims in known_shadowformer_shapes:
                shape = dims
            else:
                shape = (dims[1], dims[0], dims[2])
            if shape not in known_shadowformer_shapes:
                continue
            zeros_expr = f"torch.zeros([1, {shape[0]}, {shape[1]}, {shape[2]}]"
            zeros_prefix = str(register_match.group("zeros_prefix") or "").strip()
            if zeros_prefix:
                zeros_prefix = zeros_prefix.rstrip(", ")
                if zeros_prefix:
                    zeros_expr += f", {zeros_prefix}"
            zeros_kwargs = str(register_match.group("zeros_kwargs") or "")
            if zeros_kwargs:
                zeros_expr += zeros_kwargs
            zeros_expr += ")"
            quote = register_match.group("quote")
            tensor_kw = str(register_match.group("tensor_kw") or "")
            name_kw = str(register_match.group("name_kw") or "")
            rewritten = f"        self.register_buffer({name_kw}{quote}{register_match.group('buffer')}{quote}, {tensor_kw}{zeros_expr}"
            if register_match.group("persistent") is not None:
                rewritten += f", persistent={register_match.group('persistent')}"
            rewritten += ")"
            if rewritten != current_line:
                lines[index] = rewritten
                changed = True
            continue
        copy_call = _parse_copy_call_expr(current_line)
        if copy_call is not None:
            indent, target_expr, copy_buffer_name, src_expr, copy_kwargs = copy_call
            resolved_buffer_name = buffer_aliases.get(copy_buffer_name, copy_buffer_name)
            if resolved_buffer_name not in supported_buffers:
                continue
            normalized_src_expr = _extract_shadowformer_copy_permute_source_expr(src_expr)
            if normalized_src_expr is None:
                continue
            normalized_target_expr = _strip_outer_parentheses(target_expr)
            rewritten = f"{indent}{normalized_target_expr}.copy_({normalized_src_expr}{copy_kwargs})"
            if rewritten != current_line:
                lines[index] = rewritten
                changed = True
            continue
        binary_match = binary_shape_re.match(current_line)
        if binary_match is not None:
            dims = [
                int(binary_match.group("d1")),
                int(binary_match.group("d2")),
                int(binary_match.group("d3")),
            ]
            inferred_shape = _infer_shadowformer_shape_from_dims(dims, known_shadowformer_shapes)
            if inferred_shape is None:
                continue
            heads, height, width = inferred_shape
            rewritten = (
                f"{binary_match.group('indent')}{str(binary_match.group('batch')).strip()}, "
                f"{heads}, {height}, {width}{binary_match.group('suffix')}"
            )
            if rewritten != current_line:
                lines[index] = rewritten
                changed = True
            continue
        mul_match = mul_align_shape_re.match(current_line)
        if mul_match is not None:
            resolved_mul_lhs = local_aliases.get(str(mul_match.group("mul_lhs")), str(mul_match.group("mul_lhs")))
            resolved_mul_rhs = local_aliases.get(str(mul_match.group("mul_rhs")), str(mul_match.group("mul_rhs")))
            if frozenset({resolved_mul_lhs, resolved_mul_rhs}) not in binary_output_pairs:
                continue
            dims = [
                int(mul_match.group("d1")),
                int(mul_match.group("d2")),
                int(mul_match.group("d3")),
            ]
            inferred_shape = _infer_shadowformer_shape_from_dims(dims, known_shadowformer_shapes)
            if inferred_shape is not None:
                heads, height, width = inferred_shape
                rewritten = (
                    f"{mul_match.group('indent')}{str(mul_match.group('batch')).strip()}, "
                    f"{heads}, {height}, {width}{mul_match.group('suffix')}"
                )
                if rewritten != current_line:
                    lines[index] = rewritten
                    changed = True
                continue
    model_source = "\n".join(lines)

    if changed:
        model_path.write_text(
            model_source + ("\n" if not model_source.endswith("\n") else ""),
            encoding="utf-8",
        )


def _extract_shadowformer_copy_permute_source_expr(src_expr: str) -> str | None:
    stripped = _strip_outer_parentheses(str(src_expr).strip())
    method_match = _SHADOWFORMER_METHOD_COPY_PERMUTE_SRC_RE.match(stripped)
    if method_match is not None:
        return _strip_outer_parentheses(str(method_match.group("src")))
    functional_match = _SHADOWFORMER_FUNCTIONAL_COPY_PERMUTE_SRC_RE.match(stripped)
    if functional_match is not None:
        return _strip_outer_parentheses(str(functional_match.group("src")))
    return None


def _is_channel_last_resize_like_expr(expr: str, known_resize_outputs: Set[str]) -> bool:
    stripped = str(expr).strip()
    if stripped in known_resize_outputs:
        return True
    if re.match(
        r"^_apply_resize\((?:input=)?[A-Za-z0-9_]+, (?:size=)?[\[\(]\d+, \d+[\]\)], method='[^']+', "
        r"target_shape=[\[\(]1, \d+, \d+, \d+[\]\)], align_corners=(?:True|False), "
        r"half_pixel_centers=(?:True|False), channel_last=True\)$",
        stripped,
    ) is not None:
        return True
    for candidate in re.findall(r"\b[A-Za-z0-9_]+\b", stripped):
        if candidate.endswith("_resize_out_nhwc") or candidate.endswith("_upup_resize_out_nhwc"):
            return True
    return False


def _has_mixed_layout_decoder_merge_signature(lines: Sequence[str]) -> bool:
    resize_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*_apply_resize\((?:input=)?[A-Za-z0-9_]+, (?:size=)?[\[\(]\d+, \d+[\]\)], method='[^']+', "
        r"target_shape=[\[\(]1, \d+, \d+, \d+[\]\)], align_corners=(?:True|False), "
        r"half_pixel_centers=(?:True|False), channel_last=True\)$"
    )
    apply_concat_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*_apply_concat\((?:[\[\(](?P<inputs>.+)[\]\)]|inputs=[\[\(](?P<inputs_kw>.+)[\]\)]), axis=(?:1|3), "
        r"target_shape=[\[\(][0-9, ]+[\]\)], fused='[^']+'\)$"
    )
    resize_like_inputs: Set[str] = set()
    for line in lines:
        current_line = str(line)
        resize_match = resize_assign_re.match(current_line)
        if resize_match is not None:
            resize_like_inputs.add(str(resize_match.group("lhs")))
            continue
        alias_assign = _parse_simple_assignment_line(current_line)
        alias_rhs = (
            _strip_outer_parentheses(str(alias_assign[2]).strip())
            if alias_assign is not None
            else ""
        )
        if (
            alias_assign is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", alias_rhs) is not None
            and alias_rhs in resize_like_inputs
        ):
            resize_like_inputs.add(str(alias_assign[1]))
            continue
        for candidate in re.findall(r"\b[A-Za-z0-9_]+\b", current_line):
            if candidate.endswith("_resize_out_nhwc") or candidate.endswith("_upup_resize_out_nhwc"):
                resize_like_inputs.add(candidate)

    for line in lines:
        current_line = str(line)
        inputs: List[str] | None = None
        concat_match = apply_concat_re.match(current_line)
        if concat_match is not None:
            inputs_expr = concat_match.group("inputs")
            if inputs_expr is None:
                inputs_expr = concat_match.group("inputs_kw")
            if inputs_expr is not None:
                inputs = _split_top_level_csv_exprs(str(inputs_expr))
        elif "=" in current_line:
            _, rhs = current_line.split("=", 1)
            torch_cat_args = _parse_torch_cat_inputs_and_dim(rhs.strip())
            if torch_cat_args is not None and torch_cat_args[1] == 1:
                inputs = torch_cat_args[0]
        if inputs is None:
            continue
        if len(inputs) < 2:
            continue
        resize_inputs = [
            input_name
            for input_name in inputs
            if _is_channel_last_resize_like_expr(input_name, resize_like_inputs)
        ]
        if resize_inputs and len(resize_inputs) < len(inputs):
            return True
    return False


def _has_public_nhwc_output_bridge_signature(lines: Sequence[str]) -> bool:
    return_single_re = re.compile(
        r"^\s*return (?P<value>[A-Za-z0-9_]+)$"
    )

    returned_values: Set[str] = set()
    alias_assignments: Dict[str, str] = {}
    permute_outputs: Set[str] = set()
    for line in lines:
        current_line = str(line)
        stripped_line = current_line.lstrip()
        return_match = return_single_re.match(current_line)
        if return_match is not None:
            returned_values.add(str(return_match.group("value")))
            continue
        if stripped_line.startswith("return ") and _resolve_nhwc_to_nchw_bridge_source(stripped_line[len("return ") :]) is not None:
            return True
        alias_assign = _parse_simple_assignment_line(current_line)
        alias_rhs = (
            _strip_outer_parentheses(str(alias_assign[2]).strip())
            if alias_assign is not None
            else ""
        )
        if (
            alias_assign is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", alias_rhs) is not None
        ):
            alias_assignments[str(alias_assign[1])] = alias_rhs
            continue
        if "=" in current_line:
            lhs, rhs = current_line.split("=", 1)
            if re.fullmatch(r"\s*[A-Za-z0-9_]+\s*", lhs) is not None:
                if _resolve_nhwc_to_nchw_bridge_source(rhs.strip()) is not None:
                    permute_outputs.add(lhs.strip())
                    continue
    if not returned_values:
        return False
    pending_values = set(returned_values)
    seen_values: Set[str] = set()
    while pending_values:
        current_value = pending_values.pop()
        if current_value in seen_values:
            continue
        seen_values.add(current_value)
        if current_value in permute_outputs:
            return True
        aliased_value = alias_assignments.get(current_value, None)
        if aliased_value is not None:
            pending_values.add(aliased_value)
            continue
        return True
    return bool(returned_values)


def _has_single_mixed_layout_decoder_merge_public_output_signature(lines: Sequence[str]) -> bool:
    resize_assign_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*_apply_resize\((?:input=)?[A-Za-z0-9_]+, (?:size=)?[\[\(]\d+, \d+[\]\)], method='[^']+', "
        r"target_shape=[\[\(]1, \d+, \d+, \d+[\]\)], align_corners=(?:True|False), "
        r"half_pixel_centers=(?:True|False), channel_last=True\)$"
    )
    apply_concat_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*_apply_concat\((?:[\[\(](?P<inputs>.+)[\]\)]|inputs=[\[\(](?P<inputs_kw>.+)[\]\)]), axis=(?:1|3), "
        r"target_shape=[\[\(][0-9, ]+[\]\)], fused='[^']+'\)$"
    )
    resize_like_outputs: Set[str] = set()
    for line in lines:
        current_line = str(line)
        resize_match = resize_assign_re.match(current_line)
        if resize_match is not None:
            lhs = current_line.split("=", 1)[0].strip()
            if lhs != "":
                resize_like_outputs.add(lhs)
            continue
        alias_assign = _parse_simple_assignment_line(current_line)
        alias_rhs = (
            _strip_outer_parentheses(str(alias_assign[2]).strip())
            if alias_assign is not None
            else ""
        )
        if (
            alias_assign is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", alias_rhs) is not None
            and alias_rhs in resize_like_outputs
        ):
            resize_like_outputs.add(str(alias_assign[1]))
            continue
        for candidate in re.findall(r"\b[A-Za-z0-9_]+\b", current_line):
            if candidate.endswith("_resize_out_nhwc") or candidate.endswith("_upup_resize_out_nhwc"):
                resize_like_outputs.add(candidate)

    decoder_merge_count = 0
    for line in lines:
        current_line = str(line)
        inputs: List[str] | None = None
        concat_match = apply_concat_re.match(current_line)
        if concat_match is not None:
            inputs_expr = concat_match.group("inputs")
            if inputs_expr is None:
                inputs_expr = concat_match.group("inputs_kw")
            if inputs_expr is not None:
                inputs = _split_top_level_csv_exprs(str(inputs_expr))
        elif "=" in current_line:
            _, rhs = current_line.split("=", 1)
            torch_cat_args = _parse_torch_cat_inputs_and_dim(rhs.strip())
            if torch_cat_args is not None and torch_cat_args[1] == 1:
                inputs = torch_cat_args[0]
        if inputs is None:
            continue
        if len(inputs) != 2:
            continue
        resize_inputs = [
            input_name
            for input_name in inputs
            if _is_channel_last_resize_like_expr(input_name, resize_like_outputs)
        ]
        if len(resize_inputs) != 1:
            continue
        decoder_merge_count += 1
    return decoder_merge_count == 1 and _has_public_nhwc_output_bridge_signature(lines)


def _has_mixed_layout_decoder_merge_public_output_signature(lines: Sequence[str]) -> bool:
    return (
        _has_mixed_layout_decoder_merge_signature(lines)
        and _has_public_nhwc_output_bridge_signature(lines)
    )


def _has_nanodet_head_signature(lines: Sequence[str]) -> bool:
    apply_concat_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = _apply_concat\((?:[\[\(](?P<inputs>.+)[\]\)]|inputs=[\[\(](?P<inputs_kw>.+)[\]\)]), axis=1, "
        r"target_shape=[\[\(]1, \d+, \d+[\]\)], fused='NONE'\)$"
    )
    return_apply_concat_re = re.compile(
        r"^\s*return _apply_concat\((?:[\[\(](?P<inputs>.+)[\]\)]|inputs=[\[\(](?P<inputs_kw>.+)[\]\)]), axis=1, "
        r"target_shape=[\[\(]1, \d+, \d+[\]\)], fused='NONE'\)$"
    )

    for line in lines:
        current_line = str(line)
        inputs: List[str] | None = None
        concat_match = apply_concat_re.match(current_line)
        if concat_match is not None:
            inputs_expr = concat_match.group("inputs")
            if inputs_expr is None:
                inputs_expr = concat_match.group("inputs_kw")
            if inputs_expr is not None:
                inputs = _split_top_level_csv_exprs(str(inputs_expr))
        else:
            concat_match = return_apply_concat_re.match(current_line)
            if concat_match is not None:
                inputs_expr = concat_match.group("inputs")
                if inputs_expr is None:
                    inputs_expr = concat_match.group("inputs_kw")
                if inputs_expr is not None:
                    inputs = _split_top_level_csv_exprs(str(inputs_expr))
        if inputs is None and "=" in current_line:
            _, rhs = current_line.split("=", 1)
            torch_cat_args = _parse_torch_cat_inputs_and_dim(rhs.strip())
            if torch_cat_args is not None and torch_cat_args[1] == 1:
                inputs = torch_cat_args[0]
        if inputs is None and current_line.lstrip().startswith("return "):
            torch_cat_args = _parse_torch_cat_inputs_and_dim(current_line.lstrip()[len("return ") :].strip())
            if torch_cat_args is not None and torch_cat_args[1] == 1:
                inputs = torch_cat_args[0]
        if inputs is None:
            continue
        if len(inputs) == 4:
            return True
    return False


def _has_nanodet_backbone_pool_signature(lines: Sequence[str]) -> bool:
    aligned_pad_input_re = (
        r"_align_tensor_to_target_shape\((?:target_shape=[\[\(][0-9, ]+[\]\)], input=|input=)?[A-Za-z0-9_]+"
        r"(?:, [\[\(][0-9, ]+[\]\)]|, target_shape=[\[\(][0-9, ]+[\]\)])?\)"
    )
    padded_assign_re = re.compile(
        rf"^\s*(?P<lhs>[A-Za-z0-9_]+) = F\.pad\((?:input=)?(?:{aligned_pad_input_re}|[A-Za-z0-9_]+), "
        r"(?:pad=)?(?:\[(?:0, 0, 1, 1, 1, 1|1, 1, 1, 1)\]|\((?:0, 0, 1, 1, 1, 1|1, 1, 1, 1)\)), mode='constant', value=-3\.4028234663852886e\+38\)$"
    )
    inline_padded_re = re.compile(
        rf"^F\.pad\((?:input=)?(?:{aligned_pad_input_re}|[A-Za-z0-9_]+), "
        r"(?:pad=)?(?:\[(?:0, 0, 1, 1, 1, 1|1, 1, 1, 1)\]|\((?:0, 0, 1, 1, 1, 1|1, 1, 1, 1)\)), mode='constant', value=-3\.4028234663852886e\+38\)$"
    )

    padded_outputs: Set[str] = set()
    for line in lines:
        padded_match = padded_assign_re.match(str(line))
        if padded_match is not None:
            padded_outputs.add(str(padded_match.group("lhs")))

    if padded_outputs:
        return True

    for line in lines:
        current_line = str(line)
        if "=" not in current_line:
            continue
        _, rhs = current_line.split("=", 1)
        pool_args = _parse_apply_pool2d_input_channel_last_and_is_max(rhs.strip())
        if pool_args is None or pool_args[2] is not True:
            continue
        pool_input = pool_args[0]
        if pool_input in padded_outputs:
            return True
        if inline_padded_re.match(pool_input) is not None:
            return True
    return False


def _has_nanodet_resize_signature(lines: Sequence[str]) -> bool:
    for line in lines:
        current_line = str(line)
        assign = _parse_simple_assignment_line(current_line)
        expr_candidates = (
            [assign[2], *_extract_prefixed_call_exprs(assign[2], "_apply_resize(")]
            if assign is not None
            else _extract_prefixed_call_exprs(current_line.strip(), "_apply_resize(")
        )
        for expr in expr_candidates:
            resize_args = _parse_apply_resize_input_size_shape_and_channel_last(expr)
            if resize_args is None:
                continue
            _, size_value, shape_value, channel_last = resize_args
            if not channel_last or size_value is None or shape_value is None:
                continue
            if shape_value[0] != "1":
                continue
            return True
    return False


def _has_nhwc_stem_pool_bridge_signature(lines: Sequence[str]) -> bool:
    stem_conv_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = self\.conv_block_\d+\((?P<src_expr>.+)\)$"
    )
    padded_aligned_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = F\.pad\((?:input=)?_align_tensor_to_target_shape\((?P<input>[A-Za-z0-9_]+), "
        r"[\[\(]1, \d+, \d+, \d+[\]\)]\), (?:pad=)?[\[\(]0, 0, 1, 1, 1, 1[\]\)], mode='constant', value=-3\.4028234663852886e\+38\)$"
    )
    padded_direct_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+) = F\.pad\((?:input=)?(?P<input>[A-Za-z0-9_]+), "
        r"(?:pad=)?[\[\(]0, 0, 1, 1, 1, 1[\]\)], mode='constant', value=-3\.4028234663852886e\+38\)$"
    )
    inline_padded_aligned_re = re.compile(
        r"^F\.pad\((?:input=)?_align_tensor_to_target_shape\((?P<input>[A-Za-z0-9_]+)(?:\.permute\([^)]*\)\.contiguous\(\))?, "
        r"[\[\(]1, \d+, \d+, \d+[\]\)]\), (?:pad=)?[\[\(]0, 0, 1, 1, 1, 1[\]\)], mode='constant', value=-3\.4028234663852886e\+38\)$"
    )
    inline_padded_direct_re = re.compile(
        r"^F\.pad\((?:input=)?(?P<input>[A-Za-z0-9_]+), "
        r"(?:pad=)?[\[\(]0, 0, 1, 1, 1, 1[\]\)], mode='constant', value=-3\.4028234663852886e\+38\)$"
    )

    stem_outputs: Set[str] = set()
    nhwc_bridge_outputs: Set[str] = set()
    padded_outputs: Set[str] = set()

    for line in lines:
        current_line = str(line)
        stem_match = stem_conv_re.match(current_line)
        if stem_match is not None:
            src_expr = str(stem_match.group("src_expr")).strip()
            if re.fullmatch(r"[A-Za-z0-9_]+", src_expr) is not None or _resolve_nhwc_to_nchw_bridge_source(src_expr) is not None:
                stem_outputs.add(str(stem_match.group("lhs")))
                continue
        align_assign_match = re.match(r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+)$", current_line)
        if align_assign_match is not None:
            align_args = _parse_align_tensor_target_shape_expr(str(align_assign_match.group("rhs")))
            if align_args is not None:
                align_input, align_target_shape = align_args
                align_input = re.sub(r"\.permute\([^)]*\)\.contiguous\(\)$", "", align_input).strip()
                if (
                    align_input in stem_outputs
                    and re.fullmatch(r"[\[\(]\s*1\s*,\s*\d+\s*,\s*\d+\s*,\s*\d+\s*[\]\)]", align_target_shape) is not None
                ):
                    nhwc_bridge_outputs.add(str(align_assign_match.group("lhs")))
                    continue
        padded_match = padded_aligned_re.match(current_line)
        if padded_match is None:
            padded_match = padded_direct_re.match(current_line)
        if padded_match is not None:
            if str(padded_match.group("input")) not in nhwc_bridge_outputs:
                continue
            padded_outputs.add(str(padded_match.group("lhs")))
            continue
        assign_match = re.match(r"^\s*[A-Za-z0-9_]+\s*=\s*(?P<rhs>.+)$", current_line)
        if assign_match is not None:
            rhs = str(assign_match.group("rhs")).strip()
            pool_input = _parse_apply_pool2d_input_expr(rhs)
        else:
            rhs = ""
            pool_input = None
        pool_args = _parse_apply_pool2d_input_channel_last_and_is_max(rhs) if assign_match is not None else None
        if (
            pool_input is not None
            and pool_args is not None
            and pool_args[1]
            and pool_args[2] is True
        ):
            if pool_input in padded_outputs:
                return bool(stem_outputs) and bool(nhwc_bridge_outputs)
            inline_padded_match = inline_padded_aligned_re.match(pool_input)
            if inline_padded_match is None:
                inline_padded_match = inline_padded_direct_re.match(pool_input)
            if inline_padded_match is None:
                continue
            padded_input = str(inline_padded_match.group("input"))
            if padded_input in nhwc_bridge_outputs or padded_input in stem_outputs:
                return bool(stem_outputs)
    return False


def _has_pooled_token_attention_signature(lines: Sequence[str]) -> bool:
    self_const_alias_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*\(*\s*self\.(?P<const_attr>[A-Za-z0-9_]+)\s*\)*$"
    )
    generic_alias_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*\(*\s*(?P<input>[A-Za-z0-9_]+)\s*\)*$"
    )

    has_cf_pool = False
    nhwc_attention_tokens: Set[int] = set()
    attention_token_pairs: Set[int] = set()
    local_alias_sources: Dict[str, str] = {}
    const_alias_sources: Dict[str, str] = {}

    def _resolve_local_alias(expr: str) -> str:
        resolved = str(expr).strip()
        seen: Set[str] = set()
        while resolved not in seen:
            seen.add(resolved)
            aliased = local_alias_sources.get(resolved, None)
            if aliased is None or aliased == resolved:
                break
            resolved = aliased
        return resolved

    def _resolve_const_expr(expr: str) -> str | None:
        resolved = _resolve_local_alias(str(expr).strip())
        while True:
            if resolved.startswith("self."):
                return resolved[len("self.") :]
            aliased = const_alias_sources.get(resolved, None)
            if aliased is None or aliased == resolved:
                return None
            resolved = aliased

    def _parse_rank4_shape(shape_expr: str) -> tuple[int, int, int, int] | None:
        shape_match = re.fullmatch(
            r"[\[\(]\s*(?P<n>\d+)\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)\s*[\]\)]",
            str(shape_expr).strip(),
        )
        if shape_match is None:
            return None
        return (
            int(shape_match.group("n")),
            int(shape_match.group("d1")),
            int(shape_match.group("d2")),
            int(shape_match.group("d3")),
        )

    for line in lines:
        current_line = str(line)
        assign_match = re.match(r"^\s*[A-Za-z0-9_]+\s*=\s*(?P<rhs>.+)$", current_line)
        if assign_match is not None:
            rhs = str(assign_match.group("rhs")).strip()
            pool_args = _parse_apply_pool2d_input_channel_last_and_is_max(rhs)
            if pool_args is not None and not pool_args[1]:
                has_cf_pool = True
                continue
            align_args = _parse_align_tensor_target_shape_expr(rhs)
            if align_args is not None:
                aligned_expr, _ = align_args
                if re.fullmatch(r"torch\.mul\((?P<args>.+)\)", aligned_expr) is not None:
                    has_cf_pool = True
                    continue
                aligned_pool_args = _parse_apply_pool2d_input_channel_last_and_is_max(aligned_expr)
                if aligned_pool_args is not None and not aligned_pool_args[1]:
                    has_cf_pool = True
                    continue
        self_const_alias_match = self_const_alias_re.match(current_line)
        if self_const_alias_match is not None:
            const_alias_sources[str(self_const_alias_match.group("lhs"))] = (
                f"self.{self_const_alias_match.group('const_attr')}"
            )
            continue
        generic_alias_assign = _parse_simple_assignment_line(current_line)
        alias_rhs = (
            _strip_outer_parentheses(str(generic_alias_assign[2]).strip())
            if generic_alias_assign is not None
            else ""
        )
        if (
            generic_alias_assign is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", alias_rhs) is not None
        ):
            input_name = alias_rhs
            if not input_name.startswith("self."):
                local_alias_sources[str(generic_alias_assign[1])] = _resolve_local_alias(input_name)
            aliased_const = const_alias_sources.get(input_name, None)
            if aliased_const is not None:
                const_alias_sources[str(generic_alias_assign[1])] = aliased_const
            continue
        align_assign = _parse_align_tensor_target_shape_assign(current_line)
        if align_assign is None:
            continue
        aligned_expr, shape_expr = align_assign
        shape = _parse_rank4_shape(shape_expr)
        if shape is None:
            continue
        add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", aligned_expr)
        if add_match is None:
            continue
        add_args = _parse_binary_add_args(str(add_match.group("args")))
        if add_args is None:
            continue
        lhs_expr, rhs_expr = add_args
        resolved_lhs_const = _resolve_const_expr(lhs_expr)
        resolved_rhs_const = _resolve_const_expr(rhs_expr)
        _, dim1, dim2, dim3 = shape
        if resolved_lhs_const is not None or resolved_rhs_const is not None:
            if resolved_lhs_const is None and resolved_rhs_const is not None:
                heads, token0, token1 = dim1, dim2, dim3
            elif resolved_rhs_const is None and resolved_lhs_const is not None:
                heads, token0, token1 = dim1, dim2, dim3
            else:
                continue
            if heads >= 2 and token0 == token1 and token0 >= 16:
                attention_token_pairs.add(token0)
            continue
        height, width, channels = dim1, dim2, dim3
        if height == width and height >= 4 and channels > height:
            nhwc_attention_tokens.add(height * width)

    return has_cf_pool and bool(nhwc_attention_tokens & attention_token_pairs)


def _has_humanseg_fast_repair_signature(lines: Sequence[str]) -> bool:
    conv_assign_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*self\.conv_block_\d+\((?P<src_expr>.+)\)$"
    )
    for index in range(len(lines) - 4):
        assign0 = _parse_simple_assignment_line(str(lines[index]))
        assign1 = _parse_simple_assignment_line(str(lines[index + 1]))
        assign2 = _parse_simple_assignment_line(str(lines[index + 2]))
        concat_assign = _parse_simple_assignment_line(str(lines[index + 3]))
        line3 = str(lines[index + 3])
        line4 = str(lines[index + 4])
        conv_match = conv_assign_re.match(line4)
        resize0 = _parse_apply_resize_input_size_shape_and_channel_last(assign0[2]) if assign0 is not None else None
        resize1 = _parse_apply_resize_input_size_shape_and_channel_last(assign1[2]) if assign1 is not None else None
        resize2 = _parse_apply_resize_input_size_shape_and_channel_last(assign2[2]) if assign2 is not None else None
        concat_args = _parse_apply_concat_inputs_axis_and_shape(concat_assign[2]) if concat_assign is not None else None
        concat_lhs = concat_assign[1] if concat_assign is not None and concat_args is not None else None
        conv_src_expr = str(conv_match.group("src_expr")).strip() if conv_match is not None else None
        conv_bridge_src = _resolve_nhwc_to_nchw_bridge_source(conv_src_expr) if conv_src_expr is not None else None
        if (
            resize0 is not None
            and resize1 is not None
            and resize2 is not None
            and concat_args is not None
            and concat_args[1] == 3
            and (
                (
                    conv_match is not None
                    and conv_src_expr == concat_lhs
                )
                or (
                    conv_bridge_src is not None
                    and conv_bridge_src == concat_lhs
                )
            )
        ):
            return True
    return False


def _has_humanseg_skip_signature(lines: Sequence[str]) -> bool:
    conv_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = self\.conv_block_\d+\((?P<src_expr>.+)\)$"
    )

    align_add_lhs: Set[str] = set()
    resize_lhs_by_input: Dict[str, Set[str]] = {}
    torch_cat_inputs_by_lhs: Dict[str, List[str]] = {}
    conv_inputs: Set[str] = set()

    for line in lines:
        current_line = str(line)
        assign_match = re.match(r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+)$", current_line)
        if assign_match is not None:
            align_args = _parse_align_tensor_target_shape_expr(str(assign_match.group("rhs")).strip())
            if align_args is not None:
                align_expr, _ = align_args
                add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", align_expr)
                if add_match is not None and _parse_binary_add_args(str(add_match.group("args"))) is not None:
                    align_add_lhs.add(str(assign_match.group("lhs")))
                    continue
        if assign_match is not None:
            resize_args = _parse_apply_resize_input_size_shape_and_channel_last(str(assign_match.group("rhs")).strip())
        else:
            resize_args = None
        if resize_args is not None and not resize_args[3]:
            resize_lhs_by_input.setdefault(str(resize_args[0]), set()).add(
                str(assign_match.group("lhs"))
            )
            continue
        if assign_match is not None:
            concat_args = _parse_apply_concat_inputs_axis_and_shape(str(assign_match.group("rhs")).strip())
            if concat_args is not None and concat_args[1] == 3:
                torch_cat_inputs_by_lhs[str(assign_match.group("lhs"))] = [
                    input_name.strip() for input_name in concat_args[0] if input_name.strip()
                ]
                continue
            torch_cat_args = _parse_torch_cat_inputs_and_dim(str(assign_match.group("rhs")).strip())
            if torch_cat_args is not None and torch_cat_args[1] == 1:
                torch_cat_inputs_by_lhs[str(assign_match.group("lhs"))] = torch_cat_args[0]
                continue
        conv_match = conv_re.match(current_line)
        if conv_match is not None:
            src_expr = str(conv_match.group("src_expr")).strip()
            bridge_source = _resolve_nhwc_to_nchw_bridge_source(src_expr)
            if bridge_source is not None:
                conv_inputs.add(bridge_source)
            elif re.fullmatch(r"[A-Za-z0-9_]+", src_expr) is not None:
                conv_inputs.add(src_expr)

    if not align_add_lhs or not torch_cat_inputs_by_lhs or not conv_inputs:
        return False
    resized_from_align_add: Set[str] = set()
    all_resized_lhs: Set[str] = set()
    for resize_set in resize_lhs_by_input.values():
        all_resized_lhs.update(resize_set)
    for align_lhs in align_add_lhs:
        resized_from_align_add.update(resize_lhs_by_input.get(align_lhs, set()))
    if not all_resized_lhs:
        return False
    for cat_lhs, cat_inputs in torch_cat_inputs_by_lhs.items():
        if cat_lhs not in conv_inputs:
            continue
        resize_inputs = [input_name for input_name in cat_inputs if input_name in all_resized_lhs]
        traced_resize_inputs = [input_name for input_name in cat_inputs if input_name in resized_from_align_add]
        if len(cat_inputs) == 4 and len(resize_inputs) >= 1 and (traced_resize_inputs or align_add_lhs):
            return True
    return False


def _has_multiscale_detection_reshape_concat_tail_signature(lines: Sequence[str]) -> bool:
    permuted_conv_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*self\.conv_block_\d+\((?P<src_expr>.+)\)$"
    )
    direct_conv_re = re.compile(
        r"^\s*[A-Za-z0-9_]+\s*=\s*self\.conv_block_\d+\((?P<input>[A-Za-z0-9_]+)\)\s*$"
    )
    align_add_lhs: Set[str] = set()
    for line in lines:
        align_assign = _parse_align_tensor_target_shape_assign(str(line))
        if align_assign is None:
            continue
        aligned_expr, shape_expr = align_assign
        add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", aligned_expr)
        if add_match is None:
            continue
        if _parse_binary_add_args(str(add_match.group("args"))) is None:
            continue
        shape = _parse_rank4_shape_literal(shape_expr)
        if shape is None:
            continue
        if len(shape) == 4 and max(int(shape[1]), int(shape[3])) >= 16 and "=" in str(line):
            align_add_lhs.add(str(line).split("=", 1)[0].strip())
    if not align_add_lhs:
        return False
    has_axis1_concat = False
    has_axis2_boxes_concat = False
    has_rank3_scores_softmax = False
    for line in lines:
        current_line = str(line)
        assignment = _parse_simple_assignment_line(current_line)
        if assignment is None:
            continue
        _, _, rhs = assignment
        softmax_args = _parse_apply_softmax_input_axis_and_shape(rhs)
        if (
            softmax_args is not None
            and softmax_args[2] is not None
            and len(softmax_args[2]) == 3
            and int(softmax_args[1]) == 2
            and int(softmax_args[2][2]) == 2
        ):
            has_rank3_scores_softmax = True
        apply_concat_args = _parse_apply_concat_inputs_axis_and_shape(rhs)
        if apply_concat_args is not None:
            concat_inputs, concat_axis, concat_shape = apply_concat_args
            if len(concat_inputs) >= 2 and concat_shape is not None and len(concat_shape) == 3 and concat_shape[0] == 1:
                if concat_axis == 1 and concat_shape[2] == 2:
                    has_axis1_concat = True
                elif concat_axis == 2 and concat_shape[2] == 4:
                    has_axis2_boxes_concat = True
        torch_cat_args = _parse_torch_cat_inputs_and_dim(rhs)
        if torch_cat_args is not None and len(torch_cat_args[0]) >= 2:
            if torch_cat_args[1] == 1:
                has_axis1_concat = True
            elif torch_cat_args[1] == 2:
                has_axis2_boxes_concat = True
    if not has_axis1_concat and not has_rank3_scores_softmax:
        return False
    if not has_axis2_boxes_concat:
        return False
    for line in lines:
        match = permuted_conv_re.match(str(line))
        if match is None:
            direct_match = direct_conv_re.match(str(line))
            if direct_match is not None and str(direct_match.group("input")) in align_add_lhs:
                return True
            continue
        src_expr = str(match.group("src_expr")).strip()
        source_name = _resolve_nhwc_to_nchw_bridge_source(src_expr)
        if source_name in align_add_lhs:
            return True
        if re.fullmatch(r"[A-Za-z0-9_]+", src_expr) is not None and src_expr in align_add_lhs:
            return True
    return False


def _has_version_rfb_skip_signature(lines: Sequence[str]) -> bool:
    return _has_multiscale_detection_reshape_concat_tail_signature(lines)


def _has_iat_llie_skip_signature(lines: Sequence[str]) -> bool:
    permuted_conv_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = self\.conv_block_(?P<module>\d+)\((?P<src_expr>.+)\)$"
    )
    alias_assign_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*(?P<input>[A-Za-z0-9_]+)\s*$"
    )
    return_single_re = re.compile(r"^\s*return\s+(?P<value>[A-Za-z0-9_]+)\s*$")

    seen_modules: Set[str] = set()
    seen_rank4_rgb = False
    seen_rank4_channel64 = False
    permute_bridge_values: Set[str] = set()
    returned_values: Set[str] = set()
    alias_assignments: Dict[str, str] = {}
    for line in lines:
        current_line = str(line)
        conv_match = permuted_conv_re.match(current_line)
        if conv_match is not None:
            if _resolve_nhwc_to_nchw_bridge_source(str(conv_match.group("src_expr"))) is None:
                continue
            seen_modules.add(str(conv_match.group("module")))
            continue
        align_assign = _parse_align_tensor_target_shape_assign(current_line)
        if align_assign is not None:
            aligned_expr, shape_expr = align_assign
            add_match = re.fullmatch(r"torch\.add\((?P<args>.+)\)", aligned_expr)
            if add_match is None:
                continue
            if _parse_binary_add_args(str(add_match.group("args"))) is None:
                continue
            shape = _parse_rank4_shape_literal(shape_expr)
            if shape is None:
                continue
            if shape[3] == 64:
                seen_rank4_channel64 = True
            if shape[3] == 3:
                seen_rank4_rgb = True
            continue
        alias_assign_match = alias_assign_re.match(current_line)
        if alias_assign_match is not None:
            alias_assignments[str(alias_assign_match.group("lhs"))] = str(alias_assign_match.group("input"))
            continue
        assign_match = re.match(r"^\s*(?P<lhs>[A-Za-z0-9_]+)\s*=\s*(?P<rhs>.+)$", current_line)
        if assign_match is not None and _resolve_nhwc_to_nchw_bridge_source(str(assign_match.group("rhs")).strip()) is not None:
            permute_bridge_values.add(str(assign_match.group("lhs")))
            continue
        stripped_line = current_line.lstrip()
        if stripped_line.startswith("return "):
            return_expr = stripped_line[len("return ") :].strip()
            if _resolve_nhwc_to_nchw_bridge_source(return_expr) is not None:
                return (
                    len(seen_modules) >= 2
                    and seen_rank4_channel64
                    and seen_rank4_rgb
                    and True
                )
            for returned_expr in _split_top_level_csv_exprs(return_expr):
                if _resolve_nhwc_to_nchw_bridge_source(returned_expr) is not None:
                    return (
                        len(seen_modules) >= 2
                        and seen_rank4_channel64
                        and seen_rank4_rgb
                        and True
                    )
        return_single_match = return_single_re.match(current_line)
        if return_single_match is not None:
            returned_values.add(str(return_single_match.group("value")))

    has_output_bridge = False
    if permute_bridge_values:
        has_output_bridge = True
    else:
        pending_values = set(returned_values)
        seen_values: Set[str] = set()
        while pending_values:
            current_value = pending_values.pop()
            if current_value in seen_values:
                continue
            seen_values.add(current_value)
            if current_value in permute_bridge_values:
                has_output_bridge = True
                break
            aliased_value = alias_assignments.get(current_value, None)
            if aliased_value is not None:
                pending_values.add(aliased_value)
    return (
        len(seen_modules) >= 2
        and seen_rank4_channel64
        and seen_rank4_rgb
        and has_output_bridge
    )


def _has_pidnet_skip_signature(lines: Sequence[str]) -> bool:
    padded_inputs: Set[str] = set()
    mean_inputs: Set[str] = set()
    scale4_mul_inputs: Set[str] = set()
    scale4_mul_outputs: Set[str] = set()
    scale4_add_inputs: Set[str] = set()
    scale3_pool_outputs: Set[str] = set()
    scale3_anchor_inputs: Set[str] = set()
    pag_mul_outputs: Set[str] = set()
    pag_reduce_sum_inputs: Set[str] = set()
    local_alias_sources: Dict[str, str] = {}
    const_alias_sources: Dict[str, str] = {}
    const_pair_alias_sources: Dict[str, Tuple[str, str]] = {}

    def _resolve_local_alias(expr: str) -> str:
        resolved = str(expr).strip()
        seen: Set[str] = set()
        while resolved not in seen:
            seen.add(resolved)
            aliased = local_alias_sources.get(resolved, None)
            if aliased is None or aliased == resolved:
                break
            resolved = aliased
        return resolved

    def _resolve_const_expr(expr: str) -> str | None:
        resolved = _resolve_local_alias(str(expr).strip())
        while True:
            if resolved.startswith("self."):
                return resolved[len("self.") :]
            aliased = const_alias_sources.get(resolved, None)
            if aliased is None or aliased == resolved:
                return None
            resolved = aliased

    def _parse_pidnet_skip_plain_alias(line: str) -> tuple[str, str] | None:
        assign_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        lhs = str(assign_match.group("lhs"))
        rhs = str(assign_match.group("rhs")).strip()
        rhs_match = re.fullmatch(
            r"\(*\s*(?P<input>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\s*\)*",
            rhs,
        )
        if rhs_match is None:
            return None
        return lhs, str(rhs_match.group("input"))

    def _parse_pidnet_skip_const_pair_alias(
        line: str,
    ) -> tuple[str, str, str] | None:
        assign_match = re.match(
            r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        lhs = str(assign_match.group("lhs"))
        rhs = str(assign_match.group("rhs")).strip()
        rhs_match = re.fullmatch(
            r"\(*\s*\(*(?P<input0>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*,\s*\(*(?P<input1>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*\)*",
            rhs,
        )
        if rhs_match is None:
            return None
        return lhs, str(rhs_match.group("input0")), str(rhs_match.group("input1"))

    def _parse_pidnet_skip_tuple_const_alias(
        line: str,
    ) -> tuple[str, str, str, str] | None:
        assign_match = re.match(
            r"^\s*\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*,\s*"
            r"(?P<lhs1>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        rhs_match = re.fullmatch(
            r"\(*\s*\(*(?P<input0>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*,\s*\(*(?P<input1>self\.[A-Za-z0-9_]+|[A-Za-z0-9_]+)\)*\s*\)*",
            rhs,
        )
        if rhs_match is None:
            return None
        return (
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            str(rhs_match.group("input0")),
            str(rhs_match.group("input1")),
        )

    def _parse_pidnet_skip_tuple_const_unpack(
        line: str,
    ) -> tuple[str, str, str] | None:
        assign_match = re.match(
            r"^\s*\(*\s*(?P<lhs0>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*,\s*"
            r"(?P<lhs1>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs_match = re.fullmatch(r"\(*\s*(?P<input>[A-Za-z0-9_]+)\s*\)*", str(assign_match.group("rhs")).strip())
        if rhs_match is None:
            return None
        return (
            str(assign_match.group("lhs0")),
            str(assign_match.group("lhs1")),
            str(rhs_match.group("input")),
        )

    def _parse_pidnet_skip_scale4_const_reshape(expr: str) -> str | None:
        stripped = str(expr).strip()
        input_expr: str | None = None
        shape_expr: str | None = None

        prefix = "torch.reshape("
        if stripped.startswith(prefix) and stripped.endswith(")"):
            parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
            positional_index = 0
            for part in parts:
                if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                    key, value = part.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    if key == "input":
                        input_expr = value
                    elif key == "shape":
                        shape_expr = value
                    continue
                if positional_index == 0:
                    input_expr = part.strip()
                elif positional_index == 1:
                    shape_expr = part.strip()
                positional_index += 1
        else:
            method_match = re.fullmatch(
                r"(?P<input>[A-Za-z0-9_]+)\.(?:reshape|view)\((?P<shape>.+)\)",
                stripped,
            )
            if method_match is None:
                return None
            input_expr = str(method_match.group("input"))
            shape_expr = str(method_match.group("shape")).strip()
        if input_expr is None or shape_expr is None:
            return None
        rank4_shape = _parse_rank4_shape_expr(shape_expr)
        if rank4_shape is None:
            return None
        shape_values = [int(v) for v in list(rank4_shape)]
        non_singleton = [int(v) for v in shape_values if int(v) > 1]
        if len(non_singleton) != 1 or int(shape_values[0]) != 1:
            return None
        return _resolve_const_expr(input_expr)

    def _parse_pidnet_skip_pad_input(line: str) -> str | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, _, rhs = assign
        prefix = "F.pad("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        pad_expr: str | None = None
        mode_expr: str | None = None
        value_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "pad":
                    pad_expr = value
                elif key == "mode":
                    mode_expr = value
                elif key == "value":
                    value_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                pad_expr = part.strip()
            elif positional_index == 2:
                mode_expr = part.strip()
            elif positional_index == 3:
                value_expr = part.strip()
            positional_index += 1
        if input_expr is None or pad_expr is None or mode_expr != "'constant'" or value_expr != "0.0":
            return None
        pad_match = re.fullmatch(
            r"[\[\(]\s*(?P<p0>\d+)\s*,\s*(?P<p1>\d+)\s*,\s*(?P<p2>\d+)\s*,\s*(?P<p3>\d+)\s*[\]\)]",
            pad_expr,
        )
        if pad_match is None:
            return None
        pad_values = [int(pad_match.group(f"p{i}")) for i in range(4)]
        if len(set(pad_values)) != 1 or int(pad_values[0]) <= 0:
            return None
        return _resolve_local_alias(input_expr)

    def _parse_pidnet_skip_mean_input(line: str) -> str | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, _, rhs = assign
        prefix = "torch.mean("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        input_expr: str | None = None
        dim_expr: str | None = None
        keepdim_expr: str | None = None
        positional_index = 0
        for part in parts:
            if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                key, value = part.split("=", 1)
                key = key.strip()
                value = value.strip()
                if key == "input":
                    input_expr = value
                elif key == "dim":
                    dim_expr = value
                elif key == "keepdim":
                    keepdim_expr = value
                continue
            if positional_index == 0:
                input_expr = part.strip()
            elif positional_index == 1:
                dim_expr = part.strip()
            elif positional_index == 2:
                keepdim_expr = part.strip()
            positional_index += 1
        if input_expr is None or dim_expr is None or keepdim_expr != "True":
            return None
        try:
            dim_value = ast.literal_eval(str(dim_expr).strip())
        except Exception:
            return None
        if not isinstance(dim_value, (list, tuple)):
            return None
        normalized_axes = [int(axis) for axis in list(dim_value)]
        if normalized_axes not in ([1, 2], [2, 3]):
            return None
        return _resolve_local_alias(input_expr)

    def _parse_pidnet_skip_scale4_mul_input(line: str) -> tuple[str, str] | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        stripped = rhs.strip()
        target_shape: list[int] | None = None
        align_parts = _parse_align_tensor_target_shape_expr(stripped)
        if align_parts is not None:
            stripped = str(align_parts[0]).strip()
            target_shape = _parse_rank4_shape_literal(str(align_parts[1]))
        else:
            reshape_prefix = "torch.reshape("
            if stripped.startswith(reshape_prefix) and stripped.endswith(")"):
                reshape_parts = _split_top_level_csv_exprs(stripped[len(reshape_prefix) : -1])
                reshape_input_expr: str | None = None
                reshape_shape_expr: str | None = None
                positional_index = 0
                for part in reshape_parts:
                    if "=" in part and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is not None:
                        key, value = part.split("=", 1)
                        key = key.strip()
                        value = value.strip()
                        if key == "input":
                            reshape_input_expr = value
                        elif key == "shape":
                            reshape_shape_expr = value
                        continue
                    if positional_index == 0:
                        reshape_input_expr = part.strip()
                    elif positional_index == 1:
                        reshape_shape_expr = part.strip()
                    positional_index += 1
                if reshape_input_expr is not None and reshape_shape_expr is not None:
                    stripped = str(reshape_input_expr).strip()
                    target_shape = _parse_rank4_shape_literal(str(reshape_shape_expr))
        if target_shape is not None:
            non_singleton_dims = [int(value) for value in target_shape if int(value) > 1]
            if int(target_shape[0]) != 1 or len(non_singleton_dims) != 1:
                return None
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", stripped)
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        input_a = str(mul_args[0]).strip()
        input_b = str(mul_args[1]).strip()
        const_attr_a = _parse_pidnet_skip_scale4_const_reshape(input_a)
        if const_attr_a is None:
            const_attr_a = _resolve_const_expr(input_a)
        const_attr_b = _parse_pidnet_skip_scale4_const_reshape(input_b)
        if const_attr_b is None:
            const_attr_b = _resolve_const_expr(input_b)
        if const_attr_a is not None and const_attr_b is None:
            return lhs, _resolve_local_alias(input_b)
        if const_attr_b is not None and const_attr_a is None:
            return lhs, _resolve_local_alias(input_a)
        return None

    def _parse_pidnet_skip_scale4_add_input(line: str) -> str | None:
        assign_match = re.match(
            r"^\s*\(*\s*[A-Za-z0-9_]+\s*,\s*[A-Za-z0-9_]+\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_a: str | None = None
        input_b: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("a"))
            input_b = kwargs.get("other", kwargs.get("b"))
        if input_a is None or input_b is None:
            return None
        target_shape: list[int] | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            target_shape = _parse_rank4_shape_literal(parts[2].strip())
        else:
            target_shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
            if target_shape_expr is not None:
                target_shape = _parse_rank4_shape_literal(target_shape_expr)
        if target_shape is not None:
            non_singleton_dims = [int(value) for value in target_shape if int(value) > 1]
            if int(target_shape[0]) != 1 or len(non_singleton_dims) != 1:
                return None
        const_attr_a = _parse_pidnet_skip_scale4_const_reshape(input_a)
        if const_attr_a is None:
            const_attr_a = _resolve_const_expr(input_a)
        const_attr_b = _parse_pidnet_skip_scale4_const_reshape(input_b)
        if const_attr_b is None:
            const_attr_b = _resolve_const_expr(input_b)
        if const_attr_a is not None and const_attr_b is None:
            return _resolve_local_alias(input_b)
        if const_attr_b is not None and const_attr_a is None:
            return _resolve_local_alias(input_a)
        return None

    def _parse_pidnet_skip_scale3_pool_output(line: str) -> str | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        lhs, rhs = str(assign[1]), str(assign[2]).strip()
        prefix = "_apply_pool2d("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        kwargs: Dict[str, str] = {}
        if parts and re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", parts[0]) is None:
            kwargs["input"] = parts[0].strip()
        for part in parts:
            if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                continue
            key, value = part.split("=", 1)
            kwargs[key.strip()] = value.strip()
        target_shape = _parse_rank4_shape_literal(kwargs.get("target_shape", ""))
        if (
            target_shape is None
            or kwargs.get("channel_last") != "True"
            or int(target_shape[0]) != 1
            or int(target_shape[2]) != 1
            or int(target_shape[1]) <= 1
            or int(target_shape[3]) <= 0
        ):
            return None
        return lhs

    def _parse_pidnet_skip_scale3_anchor_input(line: str) -> str | None:
        assign_match = re.match(
            r"^\s*\(*\s*[A-Za-z0-9_]+\s*,\s*[A-Za-z0-9_]+\s*\)*\s*=\s*(?P<rhs>.+)$",
            str(line),
        )
        if assign_match is None:
            return None
        rhs = str(assign_match.group("rhs")).strip()
        prefix = "_align_binary_inputs_to_anchor("
        if not rhs.startswith(prefix) or not rhs.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(rhs[len(prefix) : -1])
        input_a: str | None = None
        input_b: str | None = None
        shape_expr: str | None = None
        if len(parts) == 3 and all(
            re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None for part in parts
        ):
            input_a = parts[0].strip()
            input_b = parts[1].strip()
            shape_expr = parts[2].strip()
        else:
            kwargs: Dict[str, str] = {}
            for part in parts:
                if re.match(r"^[A-Za-z_][A-Za-z0-9_]*\s*=", part) is None:
                    continue
                key, value = part.split("=", 1)
                kwargs[key.strip()] = value.strip()
            input_a = kwargs.get("input", kwargs.get("a"))
            input_b = kwargs.get("other", kwargs.get("b"))
            shape_expr = kwargs.get("target_shape", kwargs.get("shape"))
        target_shape = _parse_rank4_shape_literal(shape_expr or "")
        if (
            input_a is None
            or input_b is None
            or target_shape is None
            or int(target_shape[0]) != 1
            or int(target_shape[2]) != 1
            or int(target_shape[1]) <= 1
            or int(target_shape[3]) <= 1
        ):
            return None
        const_attr_a = _parse_pidnet_skip_scale4_const_reshape(input_a)
        if const_attr_a is None:
            const_attr_a = _resolve_const_expr(input_a)
        const_attr_b = _parse_pidnet_skip_scale4_const_reshape(input_b)
        if const_attr_b is None:
            const_attr_b = _resolve_const_expr(input_b)
        if const_attr_a is not None and const_attr_b is None:
            return _resolve_local_alias(input_b)
        if const_attr_b is not None and const_attr_a is None:
            return _resolve_local_alias(input_a)
        return None

    def _parse_pidnet_skip_pag_mul_output(line: str) -> str | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, lhs, rhs = assign
        align_parts = _parse_align_tensor_target_shape_expr(rhs)
        if align_parts is None:
            return None
        input_expr, _ = align_parts
        mul_match = re.fullmatch(r"torch\.mul\((?P<args>.+)\)", input_expr.strip())
        if mul_match is None:
            return None
        mul_args = _parse_binary_mul_args(str(mul_match.group("args")))
        if mul_args is None:
            return None
        lhs_root = _resolve_local_alias(str(mul_args[0]))
        rhs_root = _resolve_local_alias(str(mul_args[1]))
        if lhs_root == rhs_root:
            return None
        return lhs

    def _parse_pidnet_skip_pag_reduce_sum_input(line: str) -> str | None:
        assign = _parse_simple_assignment_line(line)
        if assign is None:
            return None
        _, _, rhs = assign
        prefix = "_reduce_sum("
        stripped = rhs.strip()
        if not stripped.startswith(prefix) or not stripped.endswith(")"):
            return None
        parts = _split_top_level_csv_exprs(stripped[len(prefix) : -1])
        if len(parts) < 3:
            return None
        input_expr = parts[0].strip()
        axes_expr = parts[1].strip()
        keepdims_expr = parts[2].strip()
        if (
            re.fullmatch(r"[A-Za-z0-9_]+", input_expr) is None
            or re.fullmatch(r"_normalize_axes\(\[3\],\s*[A-Za-z0-9_]+\.ndim\)", axes_expr) is None
            or keepdims_expr != "True"
        ):
            return None
        return input_expr

    for line in lines:
        current_line = str(line)
        parsed_pad_input = _parse_pidnet_skip_pad_input(current_line)
        if parsed_pad_input is not None:
            padded_inputs.add(str(parsed_pad_input))
            continue
        parsed_mean_input = _parse_pidnet_skip_mean_input(current_line)
        if parsed_mean_input is not None:
            mean_inputs.add(str(parsed_mean_input))
            continue
        parsed_plain_alias = _parse_pidnet_skip_plain_alias(current_line)
        if parsed_plain_alias is not None and str(parsed_plain_alias[1]).startswith("self."):
            const_alias_sources[str(parsed_plain_alias[0])] = str(parsed_plain_alias[1])
            continue
        if parsed_plain_alias is not None:
            aliased_input = str(parsed_plain_alias[1])
            if not aliased_input.startswith("self."):
                local_alias_sources[str(parsed_plain_alias[0])] = _resolve_local_alias(aliased_input)
            aliased_attr = const_alias_sources.get(aliased_input, None)
            if aliased_attr is not None:
                const_alias_sources[str(parsed_plain_alias[0])] = aliased_attr
            aliased_pair = const_pair_alias_sources.get(aliased_input, None)
            if aliased_pair is not None:
                const_pair_alias_sources[str(parsed_plain_alias[0])] = aliased_pair
            continue
        parsed_tuple_const_alias = _parse_pidnet_skip_tuple_const_alias(current_line)
        if parsed_tuple_const_alias is not None:
            tuple_pairs = (
                (str(parsed_tuple_const_alias[0]), str(parsed_tuple_const_alias[2])),
                (str(parsed_tuple_const_alias[1]), str(parsed_tuple_const_alias[3])),
            )
            for lhs_name, rhs_name in tuple_pairs:
                resolved_const = _resolve_const_expr(rhs_name)
                if resolved_const is not None:
                    const_alias_sources[lhs_name] = f"self.{resolved_const}"
            continue
        parsed_const_pair_alias = _parse_pidnet_skip_const_pair_alias(current_line)
        if parsed_const_pair_alias is not None:
            input0_name = str(parsed_const_pair_alias[1])
            input1_name = str(parsed_const_pair_alias[2])
            resolved0 = _resolve_const_expr(input0_name)
            resolved1 = _resolve_const_expr(input1_name)
            if resolved0 is not None and resolved1 is not None:
                const_pair_alias_sources[str(parsed_const_pair_alias[0])] = (
                    f"self.{resolved0}",
                    f"self.{resolved1}",
                )
            continue
        parsed_tuple_const_unpack = _parse_pidnet_skip_tuple_const_unpack(current_line)
        if parsed_tuple_const_unpack is not None:
            pair_sources = const_pair_alias_sources.get(str(parsed_tuple_const_unpack[2]), None)
            if pair_sources is not None:
                const_alias_sources[str(parsed_tuple_const_unpack[0])] = str(pair_sources[0])
                const_alias_sources[str(parsed_tuple_const_unpack[1])] = str(pair_sources[1])
            continue
        parsed_scale4_mul_input = _parse_pidnet_skip_scale4_mul_input(current_line)
        if parsed_scale4_mul_input is not None:
            scale4_mul_outputs.add(str(parsed_scale4_mul_input[0]))
            scale4_mul_inputs.add(str(parsed_scale4_mul_input[1]))
            continue
        parsed_scale4_add_input = _parse_pidnet_skip_scale4_add_input(current_line)
        if parsed_scale4_add_input is not None:
            scale4_add_inputs.add(str(parsed_scale4_add_input))
            continue
        parsed_scale3_pool_output = _parse_pidnet_skip_scale3_pool_output(current_line)
        if parsed_scale3_pool_output is not None:
            scale3_pool_outputs.add(str(parsed_scale3_pool_output))
            continue
        parsed_scale3_anchor_input = _parse_pidnet_skip_scale3_anchor_input(current_line)
        if parsed_scale3_anchor_input is not None:
            scale3_anchor_inputs.add(str(parsed_scale3_anchor_input))
            continue
        parsed_pag_mul_output = _parse_pidnet_skip_pag_mul_output(current_line)
        if parsed_pag_mul_output is not None:
            pag_mul_outputs.add(str(parsed_pag_mul_output))
            continue
        parsed_pag_reduce_sum_input = _parse_pidnet_skip_pag_reduce_sum_input(current_line)
        if parsed_pag_reduce_sum_input is not None:
            pag_reduce_sum_inputs.add(str(parsed_pag_reduce_sum_input))
            continue
    has_scale4_singleton_bn_chain = bool(scale4_mul_outputs & scale4_add_inputs)
    has_scale3_pool_anchor_context = bool(scale3_pool_outputs & scale3_anchor_inputs)
    has_spp_pool_context = bool(padded_inputs & mean_inputs)
    has_pag_gate_context = bool(pag_mul_outputs & pag_reduce_sum_inputs)
    return bool(
        has_pag_gate_context
        or has_scale3_pool_anchor_context
        or has_spp_pool_context
        or (has_scale4_singleton_bn_chain and (has_spp_pool_context or has_scale3_pool_anchor_context))
    )


def _is_interleaved_channel_pair_indices(indices: Sequence[int]) -> bool:
    if len(indices) < 4 or len(indices) % 2 != 0:
        return False
    expected_base = int(indices[0])
    pair_offset: int | None = None
    for index in range(0, len(indices), 2):
        base_index = int(indices[index])
        paired_index = int(indices[index + 1])
        if base_index != expected_base:
            return False
        current_offset = paired_index - base_index
        if current_offset <= 1:
            return False
        if pair_offset is None:
            pair_offset = current_offset
        elif current_offset != pair_offset:
            return False
        expected_base += 1
    return True


def _has_sinet_skip_signature(lines: Sequence[str]) -> bool:
    gather_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = [A-Za-z0-9_]+\[:, \[(?P<indices>[0-9,\s-]+)\], :, :\]$"
    )
    reshape_bn_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = torch\.reshape\([A-Za-z0-9_]+, [\[\(]1, (?P<channels>\d+), 1, 1[\]\)]\)$"
    )
    reshape_bn_method_re = re.compile(
        r"^\s*[A-Za-z0-9_]+ = [A-Za-z0-9_]+\.(?:reshape|view)\([\[\(]1, (?P<channels>\d+), 1, 1[\]\)]\)$"
    )
    mask_align_re = re.compile(
        r"^\s*[A-Za-z0-9_]+, [A-Za-z0-9_]+ = _align_binary_inputs(?:_to_anchor)?\([A-Za-z0-9_]+, (?P<const_expr>.+), [\[\(]1, 2, (?P<h>\d+), (?P<w>\d+)[\]\)]\)$"
    )

    has_interleaved_cf_gather = False
    has_bn_reshape = False
    has_mask_align = False
    local_alias_sources: Dict[str, str] = {}
    const_alias_sources: Dict[str, str] = {}

    generic_alias_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*\(*\s*(?P<input>[A-Za-z0-9_]+)\s*\)*$"
    )
    self_const_alias_re = re.compile(
        r"^\s*(?P<lhs>[A-Za-z0-9_]+)(?::\s*[A-Za-z0-9_\.\[\], ]+)?\s*=\s*\(*\s*self\.(?P<const_attr>[A-Za-z0-9_]+)\s*\)*$"
    )

    def _resolve_local_alias(expr: str) -> str:
        resolved = str(expr).strip()
        seen: Set[str] = set()
        while resolved not in seen:
            seen.add(resolved)
            aliased = local_alias_sources.get(resolved, None)
            if aliased is None or aliased == resolved:
                break
            resolved = aliased
        return resolved

    def _strip_outer_parentheses(expr: str) -> str:
        stripped = str(expr).strip()
        while stripped.startswith("(") and stripped.endswith(")"):
            inner = stripped[1:-1].strip()
            if inner == "":
                break
            depth = 0
            valid = True
            for char in inner:
                if char == "(":
                    depth += 1
                elif char == ")":
                    if depth == 0:
                        valid = False
                        break
                    depth -= 1
            if not valid or depth != 0:
                break
            stripped = inner
        return stripped

    def _resolve_mask_const_expr(expr: str) -> str | None:
        resolved = _strip_outer_parentheses(str(expr))
        permute_suffix = ".permute(0, 3, 1, 2).contiguous()"
        if resolved.endswith(permute_suffix):
            resolved = _strip_outer_parentheses(resolved[: -len(permute_suffix)])
        resolved = _resolve_local_alias(resolved)
        if resolved.startswith("self."):
            return resolved[len("self.") :]
        aliased_const = const_alias_sources.get(resolved, None)
        if aliased_const is None:
            return None
        if aliased_const.startswith("self."):
            return aliased_const[len("self.") :]
        return None

    for line in lines:
        current_line = str(line)
        self_const_alias_match = self_const_alias_re.match(current_line)
        if self_const_alias_match is not None:
            const_alias_sources[str(self_const_alias_match.group("lhs"))] = (
                f"self.{self_const_alias_match.group('const_attr')}"
            )
            continue
        generic_alias_assign = _parse_simple_assignment_line(current_line)
        alias_rhs = (
            _strip_outer_parentheses(str(generic_alias_assign[2]).strip())
            if generic_alias_assign is not None
            else ""
        )
        if (
            generic_alias_assign is not None
            and re.fullmatch(r"[A-Za-z0-9_]+", alias_rhs) is not None
        ):
            input_name = alias_rhs
            if not input_name.startswith("self."):
                local_alias_sources[str(generic_alias_assign[1])] = _resolve_local_alias(input_name)
            aliased_const = const_alias_sources.get(input_name, None)
            if aliased_const is not None:
                const_alias_sources[str(generic_alias_assign[1])] = aliased_const
            continue
        gather_match = gather_re.match(current_line)
        if gather_match is not None:
            raw_indices = [
                int(raw_index.strip())
                for raw_index in str(gather_match.group("indices")).split(",")
                if raw_index.strip()
            ]
            if _is_interleaved_channel_pair_indices(raw_indices):
                has_interleaved_cf_gather = True
                continue
        reshape_match = reshape_bn_re.match(current_line)
        if reshape_match is None:
            reshape_match = reshape_bn_method_re.match(current_line)
        if reshape_match is not None and int(reshape_match.group("channels")) >= 32:
            has_bn_reshape = True
            continue
        mask_align_match = mask_align_re.match(current_line)
        if (
            mask_align_match is not None
            and _resolve_mask_const_expr(str(mask_align_match.group("const_expr"))) is not None
            and int(mask_align_match.group("h")) > 0
            and int(mask_align_match.group("w")) > 0
        ):
            has_mask_align = True
    return has_interleaved_cf_gather and has_bn_reshape and has_mask_align


def _has_shadowformer_fast_repair_signature(lines: Sequence[str]) -> bool:
    (
        registered_shapes,
        _buffer_shapes,
        _copied_buffers,
        copied_shapes,
        aligned_shapes,
        _buffer_aligned_buffers,
        buffer_aligned_shapes,
    ) = _collect_shadowformer_fast_repair_facts(lines)
    return len(_collect_shadowformer_supported_shapes(registered_shapes, copied_shapes, aligned_shapes, buffer_aligned_shapes)) >= 1


def _has_shadowformer_avoid_model_ir_signature(lines: Sequence[str]) -> bool:
    (
        registered_shapes,
        _buffer_shapes,
        _copied_buffers,
        copied_shapes,
        aligned_shapes,
        _buffer_aligned_buffers,
        buffer_aligned_shapes,
    ) = _collect_shadowformer_fast_repair_facts(lines)
    supported_shapes = _collect_shadowformer_supported_shapes(
        registered_shapes,
        copied_shapes,
        aligned_shapes,
        buffer_aligned_shapes,
    )
    if buffer_aligned_shapes:
        semantic_supported_shapes = set(buffer_aligned_shapes)
    elif registered_shapes:
        semantic_supported_shapes = set(copied_shapes) if not aligned_shapes else set()
    else:
        semantic_supported_shapes = set(supported_shapes)
    has_cf_pool = False
    for line in lines:
        assign = _parse_simple_assignment_line(str(line))
        expr = assign[2] if assign is not None else str(line).strip()
        pool_args = _parse_apply_pool2d_input_and_channel_last(expr)
        if pool_args is not None and not pool_args[1]:
            has_cf_pool = True
    softmax_shapes = _collect_shadowformer_softmax_shapes(lines)
    if not has_cf_pool:
        return False
    repeated_windows: Dict[Tuple[int, int], int] = {}
    for _, heads, height, width in softmax_shapes:
        if semantic_supported_shapes and (heads, height, width) not in semantic_supported_shapes:
            continue
        if registered_shapes and not semantic_supported_shapes:
            continue
        repeated_windows[(height, width)] = repeated_windows.get((height, width), 0) + 1
    return any(count >= 2 for count in repeated_windows.values())


def _collect_shadowformer_fast_repair_facts(
    lines: Sequence[str],
) -> Tuple[
    Set[Tuple[int, int, int]],
    Dict[str, Tuple[int, int, int]],
    Set[str],
    Set[Tuple[int, int, int]],
    Set[Tuple[int, int, int]],
    Set[str],
    Set[Tuple[int, int, int]],
]:
    binary_shape_re = re.compile(
        rf"^\s*(?P<out_lhs>[A-Za-z0-9_]+),\s*(?P<out_rhs>[A-Za-z0-9_]+)\s*=\s*_align_binary_inputs(?:_to_anchor)?\(\(?\s*(?P<lhs>[A-Za-z0-9_\.]+)\s*\)?,\s*\(?\s*(?P<rhs>[A-Za-z0-9_\.]+)\s*\)?,\s*(?:\[|\()(?P<batch>{_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN})\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)(?:\]|\))\)$"
    )
    mul_align_shape_re = re.compile(
        rf"^\s*[A-Za-z0-9_]+\s*=\s*_align_tensor_to_target_shape\(torch\.mul\(\(?\s*(?P<mul_lhs>[A-Za-z0-9_]+)\s*\)?,\s*\(?\s*(?P<mul_rhs>[A-Za-z0-9_]+)\s*\)?\),\s*(?:\[|\()(?P<batch>{_SHADOWFORMER_TARGET_BATCH_EXPR_PATTERN})\s*,\s*(?P<d1>\d+)\s*,\s*(?P<d2>\d+)\s*,\s*(?P<d3>\d+)(?:\]|\))\)$"
    )

    raw_buffer_dims: Dict[str, Tuple[int, int, int]] = {}
    local_aliases = _collect_shadowformer_local_aliases(lines)
    buffer_aliases = _collect_shadowformer_buffer_aliases(lines)
    plain_copy_buffers: Set[str] = set()
    permuted_copy_buffers: Set[str] = set()

    for line in lines:
        current_line = str(line)
        register_match = _SHADOWFORMER_REGISTER_BUFFER_RE.match(current_line)
        if register_match is not None:
            raw_buffer_dims[register_match.group("buffer")] = (
                int(register_match.group("d1")),
                int(register_match.group("d2")),
                int(register_match.group("d3")),
            )
            continue
        copy_call = _parse_copy_call_expr(current_line)
        if copy_call is not None:
            _, _, copy_buffer_name, src_expr, _ = copy_call
            resolved_buffer_name = buffer_aliases.get(copy_buffer_name, copy_buffer_name)
            if resolved_buffer_name not in raw_buffer_dims:
                continue
            if _extract_shadowformer_copy_permute_source_expr(src_expr) is not None:
                permuted_copy_buffers.add(resolved_buffer_name)
            else:
                plain_copy_buffers.add(resolved_buffer_name)
            continue

    registered_shapes: Set[Tuple[int, int, int]] = set()
    buffer_shapes: Dict[str, Tuple[int, int, int]] = {}
    copied_buffers: Set[str] = set()
    copied_shapes: Set[Tuple[int, int, int]] = set()
    aligned_shapes: Set[Tuple[int, int, int]] = set()
    buffer_aligned_buffers: Set[str] = set()
    buffer_aligned_shapes: Set[Tuple[int, int, int]] = set()
    binary_output_pairs: Set[frozenset[str]] = set()

    for buffer_name, dims in raw_buffer_dims.items():
        if buffer_name in plain_copy_buffers and buffer_name not in permuted_copy_buffers:
            shape = dims
        else:
            shape = (dims[1], dims[0], dims[2])
        registered_shapes.add(shape)
        buffer_shapes[buffer_name] = shape
        if buffer_name in plain_copy_buffers or buffer_name in permuted_copy_buffers:
            copied_buffers.add(buffer_name)
            copied_shapes.add(shape)

    for line in lines:
        current_line = str(line)
        binary_match = binary_shape_re.match(current_line)
        if binary_match is not None:
            dims = [int(binary_match.group("d1")), int(binary_match.group("d2")), int(binary_match.group("d3"))]
            inferred_shape = _infer_shadowformer_shape_from_dims(dims, registered_shapes)
            if inferred_shape is not None:
                binary_output_pairs.add(
                    frozenset(
                        {
                            str(binary_match.group("out_lhs")),
                            str(binary_match.group("out_rhs")),
                        }
                    )
                )
                aligned_shapes.add(inferred_shape)
                for operand_name in (str(binary_match.group("lhs")), str(binary_match.group("rhs"))):
                    if operand_name.startswith("self."):
                        buffer_name = operand_name[len("self.") :]
                    else:
                        buffer_name = buffer_aliases.get(operand_name)
                    if buffer_name is None:
                        continue
                    buffer_shape = buffer_shapes.get(buffer_name)
                    if buffer_shape == inferred_shape:
                        buffer_aligned_buffers.add(buffer_name)
                        buffer_aligned_shapes.add(inferred_shape)
            continue
        mul_match = mul_align_shape_re.match(current_line)
        if mul_match is not None:
            resolved_mul_lhs = local_aliases.get(str(mul_match.group("mul_lhs")), str(mul_match.group("mul_lhs")))
            resolved_mul_rhs = local_aliases.get(str(mul_match.group("mul_rhs")), str(mul_match.group("mul_rhs")))
            if frozenset({resolved_mul_lhs, resolved_mul_rhs}) not in binary_output_pairs:
                continue
            dims = [int(mul_match.group("d1")), int(mul_match.group("d2")), int(mul_match.group("d3"))]
            inferred_shape = _infer_shadowformer_shape_from_dims(dims, registered_shapes)
            if inferred_shape is not None:
                aligned_shapes.add(inferred_shape)

    return (
        registered_shapes,
        buffer_shapes,
        copied_buffers,
        copied_shapes,
        aligned_shapes,
        buffer_aligned_buffers,
        buffer_aligned_shapes,
    )


def _collect_shadowformer_softmax_shapes(lines: Sequence[str]) -> List[Tuple[str, int, int, int]]:
    softmax_shapes: List[Tuple[str, int, int, int]] = []
    for line in lines:
        assign = _parse_simple_assignment_line(str(line))
        expr = assign[2] if assign is not None else str(line).strip()
        softmax_args = _parse_apply_softmax_input_axis_and_shape(expr)
        if softmax_args is None:
            continue
        _, axis_value, rank4_shape = softmax_args
        if axis_value not in {3, -1} or rank4_shape is None:
            continue
        softmax_shapes.append(
            (
                    str(rank4_shape[0]).strip(),
                    int(rank4_shape[1]),
                    int(rank4_shape[2]),
                    int(rank4_shape[3]),
            )
        )
    return softmax_shapes


def _collect_shadowformer_supported_shapes(
    registered_shapes: Set[Tuple[int, int, int]],
    copied_shapes: Set[Tuple[int, int, int]],
    aligned_shapes: Set[Tuple[int, int, int]],
    buffer_aligned_shapes: Set[Tuple[int, int, int]],
) -> Set[Tuple[int, int, int]]:
    supported_shapes: Set[Tuple[int, int, int]] = set()
    for known_shape in aligned_shapes:
        if known_shape in copied_shapes or known_shape in buffer_aligned_shapes:
            supported_shapes.add(known_shape)
    return supported_shapes


def _infer_shadowformer_shape_from_dims(
    dims: Sequence[int],
    known_shapes: Set[Tuple[int, int, int]],
) -> Tuple[int, int, int] | None:
    for known_shape in known_shapes:
        if tuple(dims) == known_shape:
            return known_shape
    for known_shape in known_shapes:
        heads, height, width = known_shape
        if tuple(dims) in (
            (height, heads, width),
            (height, width, heads),
        ):
            return known_shape
    for known_shape in known_shapes:
        if sorted(dims) == sorted(known_shape):
            return known_shape
    return None


def _should_skip_expensive_raw_canonicalize_for_native_package(package_path: Path) -> bool:
    model_path = package_path / "model.py"
    if not model_path.exists():
        return False
    try:
        model_source = model_path.read_text(encoding="utf-8")
    except Exception:
        return False
    model_lines = _model_source_lines(model_source)
    # SiNet's generated package is already repaired by the fast pre-canonicalize
    # pass, while the generic raw canonicalizer can take minutes on this pattern.
    if _has_sinet_skip_signature(model_lines):
        return True
    if _has_pidnet_skip_signature(model_lines):
        return True
    if _has_humanseg_skip_signature(model_lines):
        return True
    if _has_version_rfb_skip_signature(model_lines):
        return True
    if _has_iat_llie_skip_signature(model_lines):
        return True
    if _has_single_mixed_layout_decoder_merge_public_output_signature(model_lines):
        return True
    if _has_shadowformer_avoid_model_ir_signature(model_lines):
        return True
    if _has_shadowformer_fast_repair_signature(model_lines):
        return True
    return False


def _should_avoid_model_ir_in_raw_canonicalize_for_native_package(package_path: Path) -> bool:
    model_path = package_path / "model.py"
    if not model_path.exists():
        return False
    model_source = model_path.read_text(encoding="utf-8")
    model_lines = _model_source_lines(model_source)
    if _has_version_rfb_skip_signature(model_lines):
        return True
    if _has_iat_llie_skip_signature(model_lines):
        return True
    if _has_mixed_layout_decoder_merge_public_output_signature(model_lines):
        return True
    if _has_nhwc_stem_pool_bridge_signature(model_lines):
        return True
    if _has_nanodet_backbone_pool_signature(model_lines) and _has_nanodet_resize_signature(model_lines) and _has_nanodet_head_signature(model_lines):
        return True
    if _has_pooled_token_attention_signature(model_lines):
        return True
    if _has_shadowformer_avoid_model_ir_signature(model_lines):
        return True
    return False


def _canonicalize_generated_model_source_for_raw_export_with_fast_path(
    package_path: Path,
    model_ir: ModelIR | None = None,
) -> None:
    _apply_fast_precanonicalize_repairs_until_stable(package_path)
    if _should_skip_expensive_raw_canonicalize_for_native_package(package_path):
        return
    canonicalize_model_ir = (
        None
        if _should_avoid_model_ir_in_raw_canonicalize_for_native_package(package_path)
        else model_ir
    )
    _canonicalize_generated_model_source_for_raw_export(
        package_path,
        model_ir=canonicalize_model_ir,
    )
    _apply_fast_precanonicalize_repairs_until_stable(package_path)
    _apply_structural_final_model_repairs(package_path / "model.py")


def _rewrite_generated_model_source_for_exported_program(
    package_path: Path,
    model_ir: ModelIR | None = None,
) -> None:
    _canonicalize_generated_model_source_for_raw_export_with_fast_path(
        package_path,
        model_ir=model_ir,
    )
    model_path = package_path / "model.py"
    if not model_path.exists():
        return
    original_lines = model_path.read_text(encoding="utf-8").splitlines()
    rewritten_lines = _repair_exported_program_channel_last_pool_targets(original_lines)
    rewritten_lines = _repair_exported_program_direct_conv_cf_add_targets(rewritten_lines)
    rewritten_lines = _fold_channel_first_hardsigmoid_gate_conv_bridges(rewritten_lines)
    rewritten_lines = _rewrite_channel_first_se_scale_binary_bridges(rewritten_lines)
    rewritten_lines = _repair_channel_last_gap_conv_inputs(rewritten_lines)
    if rewritten_lines != original_lines:
        model_path.write_text("\n".join(rewritten_lines) + "\n", encoding="utf-8")


@contextlib.contextmanager
def _temporarily_rewrite_generated_model_source_for_exported_program(
    package_path: Path,
    model_ir: ModelIR | None = None,
):
    model_path = package_path / "model.py"
    if not model_path.exists():
        yield
        return
    original_source = model_path.read_text(encoding="utf-8")
    try:
        _rewrite_generated_model_source_for_exported_program(
            package_path,
            model_ir=model_ir,
        )
        _apply_fast_precanonicalize_repairs_until_stable(package_path)
        from ._pytorch_exporter_native_codegen_pipeline import (
            _rewrite_native_generated_model_postprocesses,
        )

        _rewrite_native_generated_model_postprocesses(model_path)
        _reapply_post_export_final_model_repairs(package_path)
        yield
    finally:
        current_source = model_path.read_text(encoding="utf-8")
        if current_source != original_source:
            model_path.write_text(original_source, encoding="utf-8")


def _repair_exported_program_channel_last_pool_targets(lines: List[str]) -> List[str]:
    rewritten = list(lines)
    changed = True
    while changed:
        changed = False
        repair_context = _build_fast_precanonicalize_repair_context(rewritten)
        for index, line in enumerate(rewritten):
            apply_pool2d_match = _parse_apply_pool2d_assign_with_shape(line)
            if apply_pool2d_match is None:
                continue
            pool_indent, lhs_name, pool_input_name, pool_rest, current_shape, pool_is_max, pool_channel_last = apply_pool2d_match
            if not _fast_precanonicalize_has_channel_last_spatial_consumer(
                lhs_name,
                index,
                rewritten,
                repair_context,
            ):
                continue
            if (
                pool_channel_last
                and _fast_precanonicalize_rank4_layout_hint(current_shape) == "nhwc"
            ):
                continue
            normalized_shape = _normalize_nhwc_rank4_shape(current_shape)
            rewritten_line = (
                f"{pool_indent}{lhs_name} = _apply_pool2d("
                f"{pool_input_name}, {pool_rest}, "
                f"target_shape={repr(normalized_shape)}, "
                f"is_max_pool={pool_is_max}, channel_last=True)"
            )
            if rewritten_line == line:
                continue
            rewritten[index] = rewritten_line
            changed = True
    return rewritten
























def _write_native_model_file(
    output_folder_path: str,
    *,
    model_ir: ModelIR,
    metadata: Dict[str, Any],
    tensor_storage_name_map: Dict[str, str],
) -> List[Tuple[str, str]]:
    package_dir = Path(output_folder_path)
    _ensure_direct_codegen_supported(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    load_specs = _write_native_model_file_impl(
        _NativeModelFileWriterContext(
            output_folder_path,
            model_ir,
            metadata,
            tensor_storage_name_map,
            package_dir,
            _collect_feature_last_sequence_tensor_names(
                model_ir,
                graph_index=graph_index,
            ),
            _build_tensor_var_name_map(model_ir),
            graph_index,
        )
    )
    _patch_generated_runtime_pool2d_channel_last_recovery(package_dir)
    return load_specs


def _write_native_model_file_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_impl(context)


def _write_native_model_file_codegen_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_impl(context)


def _write_native_model_file_codegen_core_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_body_impl(context)


def _write_native_model_file_codegen_core_body_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_body_main_impl(context)


def _write_native_model_file_codegen_core_body_main_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    return _write_native_model_file_codegen_core_body_main_inner_impl(context)


def _write_native_model_file_codegen_core_body_main_inner_impl(
    context: _NativeModelFileWriterContext,
) -> List[Tuple[str, str]]:
    state = _prepare_native_codegen_state(context)
    bindings = _build_native_codegen_bindings(state)
    _build_native_constant_aliases(state, bindings)
    _emit_native_forward_lines(state, bindings)
    return _finalize_native_codegen(state, bindings)


def _try_export_native_package_from_tflite_import(
    *,
    output_folder_path: str,
    fallback_tflite_path: str,
    reference_model_ir: Optional[ModelIR] = None,
    reference_onnx_graph: Optional[Any] = None,
) -> Optional[str]:
    if not os.path.exists(str(fallback_tflite_path)):
        return None
    imported_model_ir = import_model_ir_from_tflite(
        tflite_file_path=str(fallback_tflite_path),
    )
    _merge_reference_public_boundary_metadata(
        imported_model_ir=imported_model_ir,
        reference_model_ir=reference_model_ir,
        reference_onnx_graph=reference_onnx_graph,
    )
    return export_pytorch_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=output_folder_path,
        fallback_tflite_path=None,
        fallback_onnx_graph=None,
        fallback_saved_model_path=None,
        fallback_tflite_has_custom_ops=False,
    )


def _try_export_runtime_wrapper_package_from_tflite_import(
    *,
    output_folder_path: str,
    fallback_tflite_path: str,
) -> Optional[str]:
    if not os.path.exists(str(fallback_tflite_path)):
        return None
    imported_model_ir = import_model_ir_from_tflite(
        tflite_file_path=str(fallback_tflite_path),
    )
    if not _supports_runtime_wrapper_model_ir(imported_model_ir):
        return None
    return _export_runtime_wrapper_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=output_folder_path,
    )


def _write_native_package_generation_timeout_metadata(
    *,
    output_folder_path: str,
    timeout_sec: int,
) -> None:
    os.makedirs(output_folder_path, exist_ok=True)
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    metadata: Dict[str, Any] = {}
    if os.path.exists(metadata_path):
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, dict):
                metadata = dict(loaded)
        except Exception:
            metadata = {}
    metadata["native_package_generation"] = {
        "status": "timed_out",
        "timed_out": True,
        "timeout_sec": int(timeout_sec),
        "error": "native_pytorch_generation_timeout",
        "skipped_reason": "timed_out_recursion_explosion",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _export_pytorch_package_from_model_ir_timeout_worker(
    result_queue: Any,
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    fallback_tflite_path: Optional[str],
    fallback_onnx_graph: Optional[Any],
    fallback_saved_model_path: Optional[str],
    fallback_saved_model_factory: Optional[Callable[[], Optional[str]]],
    fallback_tflite_has_custom_ops: bool,
) -> None:
    try:
        result = _export_pytorch_package_from_model_ir_impl(
            model_ir=model_ir,
            output_folder_path=output_folder_path,
            fallback_tflite_path=fallback_tflite_path,
            fallback_onnx_graph=fallback_onnx_graph,
            fallback_saved_model_path=fallback_saved_model_path,
            fallback_saved_model_factory=fallback_saved_model_factory,
            fallback_tflite_has_custom_ops=fallback_tflite_has_custom_ops,
        )
        result_queue.put(
            {
                "ok": True,
                "result": str(result),
            }
        )
    except BaseException as ex:
        result_queue.put(
            {
                "ok": False,
                "error_type": type(ex).__name__,
                "error": str(ex),
                "traceback": traceback.format_exc(),
            }
        )


def _export_pytorch_package_from_model_ir_with_timeout(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    fallback_tflite_path: Optional[str] = None,
    fallback_onnx_graph: Optional[Any] = None,
    fallback_saved_model_path: Optional[str] = None,
    fallback_saved_model_factory: Optional[Callable[[], Optional[str]]] = None,
    fallback_tflite_has_custom_ops: bool = False,
    native_package_generation_timeout_sec: int,
) -> str:
    try:
        ctx = multiprocessing.get_context("fork")
    except Exception:
        return _export_pytorch_package_from_model_ir_impl(
            model_ir=model_ir,
            output_folder_path=output_folder_path,
            fallback_tflite_path=fallback_tflite_path,
            fallback_onnx_graph=fallback_onnx_graph,
            fallback_saved_model_path=fallback_saved_model_path,
            fallback_saved_model_factory=fallback_saved_model_factory,
            fallback_tflite_has_custom_ops=fallback_tflite_has_custom_ops,
        )

    result_queue = ctx.Queue(maxsize=1)
    process = ctx.Process(
        target=_export_pytorch_package_from_model_ir_timeout_worker,
        kwargs={
            "result_queue": result_queue,
            "model_ir": model_ir,
            "output_folder_path": output_folder_path,
            "fallback_tflite_path": fallback_tflite_path,
            "fallback_onnx_graph": fallback_onnx_graph,
            "fallback_saved_model_path": fallback_saved_model_path,
            "fallback_saved_model_factory": fallback_saved_model_factory,
            "fallback_tflite_has_custom_ops": bool(fallback_tflite_has_custom_ops),
        },
    )
    process.start()
    process.join(timeout=float(native_package_generation_timeout_sec))
    if process.is_alive():
        process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            try:
                process.kill()
            except Exception:
                pass
            process.join(timeout=1.0)
        shutil.rmtree(output_folder_path, ignore_errors=True)
        _write_native_package_generation_timeout_metadata(
            output_folder_path=output_folder_path,
            timeout_sec=int(native_package_generation_timeout_sec),
        )
        raise NativePyTorchGenerationTimeoutError(
            "Native PyTorch package generation timed out after "
            f"{int(native_package_generation_timeout_sec)}s. "
            "Treating this as recursion explosion."
        )

    message: Optional[Dict[str, Any]] = None
    try:
        message = result_queue.get(timeout=1.0)
    except Exception:
        message = None
    finally:
        try:
            result_queue.close()
        except Exception:
            pass

    if not isinstance(message, dict):
        raise ModelIRPyTorchExportError(
            "Native PyTorch export worker exited without returning a result."
        )
    if bool(message.get("ok", False)):
        return str(message.get("result", output_folder_path))
    error_type = str(message.get("error_type", "RuntimeError"))
    error_text = str(message.get("error", ""))
    traceback_text = str(message.get("traceback", "")).strip()
    raise ModelIRPyTorchExportError(
        f"Native PyTorch export worker failed. type={error_type} "
        f"error={error_text}"
        + (f"\n{traceback_text}" if traceback_text != "" else "")
    )


def _export_pytorch_package_from_model_ir_impl(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    fallback_tflite_path: Optional[str] = None,
    fallback_onnx_graph: Optional[Any] = None,
    fallback_saved_model_path: Optional[str] = None,
    fallback_saved_model_factory: Optional[Callable[[], Optional[str]]] = None,
    fallback_tflite_has_custom_ops: bool = False,
) -> str:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "PyTorch export requires `torch` to be installed."
        ) from ex

    resolved_fallback_saved_model_path = (
        str(fallback_saved_model_path)
        if fallback_saved_model_path is not None and str(fallback_saved_model_path).strip() != ""
        else None
    )

    def _get_fallback_saved_model_path() -> Optional[str]:
        nonlocal resolved_fallback_saved_model_path
        if resolved_fallback_saved_model_path is not None:
            return resolved_fallback_saved_model_path
        if fallback_saved_model_factory is None:
            return None
        try:
            generated_path = fallback_saved_model_factory()
        except Exception:
            return None
        if generated_path is None or str(generated_path).strip() == "":
            return None
        resolved_fallback_saved_model_path = str(generated_path)
        return resolved_fallback_saved_model_path

    if (
        fallback_tflite_path is not None
        and str(fallback_tflite_path).strip() != ""
        and not bool(fallback_tflite_has_custom_ops)
        and _has_tflite_import_preferred_control_or_recurrent_ops(model_ir)
    ):
        try:
            imported_native_package_path = _try_export_native_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
                reference_model_ir=model_ir,
                reference_onnx_graph=fallback_onnx_graph,
            )
            if imported_native_package_path is not None:
                return imported_native_package_path
        except Exception:
            pass

    try:
        normalized: Optional[ModelIR] = None
        native_prep_error: Optional[Exception] = None

        try:
            needs_same_avg_pool_restore = any(
                str(op.op_type) == "AVERAGE_POOL_2D"
                and str(op.options.get("padding", "")).upper() == "SAME"
                for op in model_ir.operators
            )
            native_model_ir = copy.deepcopy(model_ir) if needs_same_avg_pool_restore else model_ir
            if needs_same_avg_pool_restore:
                _restore_same_average_pool_exclude_pad_correction_for_native_runtime(native_model_ir)
            normalized = prepare_model_ir_for_native_pytorch(native_model_ir)
            _ensure_native_export_supported_ops(normalized)
        except Exception as ex:
            normalized = None
            native_prep_error = ex
        if (
            normalized is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and not bool(fallback_tflite_has_custom_ops)
        ):
            try:
                imported_native_package_path = _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
                if imported_native_package_path is not None:
                    return imported_native_package_path
            except Exception:
                pass
        fallback_saved_model_path_for_export = _get_fallback_saved_model_path() if normalized is None else resolved_fallback_saved_model_path
        if (
            normalized is None
            and fallback_saved_model_path_for_export is not None
            and _should_prefer_saved_model_backed_package(model_ir)
        ):
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path_for_export),
            )
        if (
            normalized is None
            and fallback_saved_model_path is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and not bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            return export_pytorch_package_from_tflite_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                tflite_file_path=str(fallback_tflite_path),
            )
        if (
            normalized is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            runtime_wrapper_path = _try_export_runtime_wrapper_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
            )
            if runtime_wrapper_path is not None:
                return runtime_wrapper_path

        if normalized is None:
            if native_prep_error is not None:
                raise native_prep_error
            raise ModelIRPyTorchExportError(
                "Native PyTorch export preparation failed for an unknown reason."
            )
        tensor_storage_name_map = _make_tensor_storage_name_map(normalized)

        os.makedirs(output_folder_path, exist_ok=True)
        metadata = _build_metadata_payload(normalized)
        metadata["execution_backend"] = "native"
        metadata["tensor_storage_names"] = dict(tensor_storage_name_map)
        native_load_specs: Optional[List[Tuple[str, str]]] = None
        try:
            native_load_specs = _write_native_model_file(
                output_folder_path,
                model_ir=normalized,
                metadata=metadata,
                tensor_storage_name_map=tensor_storage_name_map,
            )
            package_dir = Path(output_folder_path)
            _canonicalize_generated_model_source_for_raw_export_with_fast_path(
                package_dir,
                model_ir=normalized,
            )
        except ModelIRPyTorchExportError as ex:
            if not _is_direct_codegen_unsupported_error(ex):
                raise
            # Keep torch-kernel-backed packages native when runtime kernels
            # support the graph, even if direct Python codegen does not yet.
            _write_generated_package_common_files(output_folder_path)
            _write_wrapper_model_file(output_folder_path)
            metadata["execution_backend"] = "runtime_wrapper"
        metadata_path = os.path.join(output_folder_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        _apply_fast_precanonicalize_repairs_until_stable(Path(output_folder_path))
        if native_load_specs is not None:
            from ._pytorch_exporter_native_codegen_pipeline import (
                _rewrite_native_generated_model_postprocesses,
            )

            final_model_path = Path(output_folder_path) / "model.py"
            _rewrite_native_generated_model_postprocesses(final_model_path)
            _reapply_post_export_final_model_repairs(Path(output_folder_path))
        if native_load_specs is not None:
            state_dict = _build_native_generated_state_dict(
                package_path=output_folder_path,
                model_ir=normalized,
                load_specs=native_load_specs,
            )
        else:
            state_dict = {}
            for tensor_name, tensor in normalized.tensors.items():
                if not isinstance(tensor.data, np.ndarray):
                    continue
                dtype_name = str(tensor.dtype).upper()
                if dtype_name not in {"BOOL", "INT8", "INT16", "INT32", "INT64", "UINT8", "FLOAT16", "FLOAT32", "FLOAT64"}:
                    raise ModelIRPyTorchExportError(
                        f"Unsupported tensor dtype for PyTorch export: tensor={tensor_name} dtype={tensor.dtype}"
                    )
                storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
                state_dict[storage_name] = torch.as_tensor(np.asarray(tensor.data))
        torch.save(state_dict, os.path.join(output_folder_path, "state_dict.pth"))
        return str(output_folder_path)
    except Exception:
        string_config = _extract_string_normalizer_config_from_onnx_graph(
            fallback_onnx_graph,
        )
        if string_config is not None:
            return export_pytorch_package_from_string_normalizer_onnx(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                onnx_graph=fallback_onnx_graph,
            )
        fallback_saved_model_path_for_export = _get_fallback_saved_model_path()
        if fallback_saved_model_path_for_export is not None:
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path_for_export),
            )
        if fallback_tflite_path is None or str(fallback_tflite_path).strip() == "":
            raise
        try:
            runtime_wrapper_path = _try_export_runtime_wrapper_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
            )
            if runtime_wrapper_path is not None:
                return runtime_wrapper_path
        except Exception:
            pass
        if not bool(fallback_tflite_has_custom_ops):
            try:
                imported_native_package_path = _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
                if imported_native_package_path is not None:
                    return imported_native_package_path
            except Exception:
                pass
        return export_pytorch_package_from_tflite_artifact(
            model_ir=model_ir,
            output_folder_path=output_folder_path,
            tflite_file_path=str(fallback_tflite_path),
        )


def export_pytorch_package_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    fallback_tflite_path: Optional[str] = None,
    fallback_onnx_graph: Optional[Any] = None,
    fallback_saved_model_path: Optional[str] = None,
    fallback_saved_model_factory: Optional[Callable[[], Optional[str]]] = None,
    fallback_tflite_has_custom_ops: bool = False,
    native_package_generation_timeout_sec: Optional[int] = 0,
) -> str:
    timeout_sec = int(native_package_generation_timeout_sec or 0)
    if timeout_sec <= 0:
        return _export_pytorch_package_from_model_ir_impl(
            model_ir=model_ir,
            output_folder_path=output_folder_path,
            fallback_tflite_path=fallback_tflite_path,
            fallback_onnx_graph=fallback_onnx_graph,
            fallback_saved_model_path=fallback_saved_model_path,
            fallback_saved_model_factory=fallback_saved_model_factory,
            fallback_tflite_has_custom_ops=fallback_tflite_has_custom_ops,
        )
    return _export_pytorch_package_from_model_ir_with_timeout(
        model_ir=model_ir,
        output_folder_path=output_folder_path,
        fallback_tflite_path=fallback_tflite_path,
        fallback_onnx_graph=fallback_onnx_graph,
        fallback_saved_model_path=fallback_saved_model_path,
        fallback_saved_model_factory=fallback_saved_model_factory,
        fallback_tflite_has_custom_ops=fallback_tflite_has_custom_ops,
        native_package_generation_timeout_sec=timeout_sec,
    )


def debug_export_native_codegen_intermediates_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    stop_after: str = "canonicalize",
) -> Dict[str, str]:
    stop_after_normalized = str(stop_after).strip().lower()
    if stop_after_normalized not in {"write", "canonicalize"}:
        raise ValueError(
            "stop_after must be either 'write' or 'canonicalize'."
        )

    normalized = prepare_model_ir_for_native_pytorch(model_ir)
    _ensure_native_export_supported_ops(normalized)

    tensor_storage_name_map = _make_tensor_storage_name_map(normalized)
    os.makedirs(output_folder_path, exist_ok=True)
    metadata = _build_metadata_payload(normalized)
    metadata["execution_backend"] = "native"
    metadata["tensor_storage_names"] = dict(tensor_storage_name_map)

    package_dir = Path(output_folder_path)
    _write_native_model_file(
        output_folder_path,
        model_ir=normalized,
        metadata=metadata,
        tensor_storage_name_map=tensor_storage_name_map,
    )

    model_path = package_dir / "model.py"
    pre_canonicalize_path = package_dir / "model_pre_canonicalize.py"
    shutil.copyfile(model_path, pre_canonicalize_path)
    artifacts = {
        "package_path": str(package_dir),
        "model_path": str(model_path),
        "model_pre_canonicalize_path": str(pre_canonicalize_path),
    }
    if stop_after_normalized == "write":
        return artifacts

    _canonicalize_generated_model_source_for_raw_export_with_fast_path(
        package_dir,
        model_ir=normalized,
    )
    artifacts["model_post_canonicalize_path"] = str(model_path)
    return artifacts
