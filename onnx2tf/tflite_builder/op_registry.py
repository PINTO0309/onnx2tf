from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import onnx

from onnx2tf.tflite_builder.core.op_contracts import (
    DispatchEntry,
    DispatchResolution,
    NodeValidationError,
    ValidationSpec,
    get_original_node_inputs as _get_original_node_inputs,
    is_integer_dtype as _is_integer_dtype,
    is_unknown_rank_placeholder_tensor as _is_unknown_rank_placeholder_tensor,
    normalize_axis_for_rank as _normalize_axis_for_rank,
    require_const_input as _require_const_input,
    tensor_shape_with_signature as _tensor_shape_with_signature,
    validate_attrs as _validate_attrs,
    validate_counts as _validate_counts,
    validate_rank_constraints as _validate_rank_constraints,
)
from onnx2tf.tflite_builder.op_families.normalization import (
    _validate_batch_norm,
    _validate_depth_to_space,
    _validate_instance_norm,
    _validate_layer_norm,
    _validate_mean_variance_normalization,
    _validate_space_to_depth,
)
from onnx2tf.tflite_builder.op_families.quantization import (
    _validate_conv_integer,
    _validate_dynamic_quantize_linear,
    _validate_qgemm,
    _validate_qlinear_average_pool,
    _validate_qlinear_binary,
    _validate_qlinear_concat,
    _validate_qlinear_conv,
    _validate_qlinear_global_average_pool,
    _validate_qlinear_leaky_relu,
    _validate_qlinear_matmul,
    _validate_qlinear_sigmoid,
    _validate_qlinear_softmax,
    _validate_quantize_dequantize_linear,
)
from onnx2tf.tflite_builder.op_families.reduction import (
    _extract_axes,
    _normalize_axes_for_rank,
    _validate_cumprod,
    _validate_cumsum,
    _validate_reduce,
    _validate_squeeze,
    _validate_unsqueeze,
)
from onnx2tf.tflite_builder.op_families.selection import (
    _validate_argmax,
    _validate_argmin,
    _validate_gather,
    _validate_gather_elements,
    _validate_gather_nd,
    _validate_hardmax,
    _validate_non_max_suppression,
    _validate_nonzero,
    _validate_topk,
)
from onnx2tf.tflite_builder.op_families.recurrent import (
    _validate_gru,
    _validate_lstm,
    _validate_rnn,
)
from onnx2tf.tflite_builder.op_families.shape import (
    _validate_range,
)

from onnx2tf.tflite_builder.op_builders import (
    build_abs_op,
    build_acos_op,
    build_acosh_op,
    build_affine_grid_op,
    build_argmin_op,
    build_argmax_op,
    build_asin_op,
    build_asinh_op,
    build_atan_op,
    build_atanh_op,
    build_batch_normalization_op,
    build_instance_normalization_op,
    build_attention_op,
    build_binary_op,
    build_bitshift_op,
    build_bitwise_not_op,
    build_cast_op,
    build_castlike_op,
    build_center_crop_pad_op,
    build_celu_op,
    build_clip_op,
    build_col2im_op,
    build_compress_op,
    build_concat_op,
    build_constant_of_shape_op,
    build_conv2d_or_depthwise_op,
    build_conv_transpose_op,
    build_deform_conv_op,
    build_fused_conv_op,
    build_dropout_op,
    build_cosh_op,
    build_custom_passthrough_op,
    build_conv_integer_op,
    build_dequantize_linear_op,
    build_depth_to_space_op,
    build_dft_op,
    build_dynamic_quantize_linear_op,
    build_div_op,
    build_det_op,
    build_einsum_op,
    build_erf_op,
    build_eyelike_op,
    build_expand_op,
    build_flatten_op,
    build_grid_sample_op,
    build_hamming_window_op,
    build_hann_window_op,
    build_mel_weight_matrix_op,
    build_fused_matmul_op,
    build_fully_connected_from_gemm_or_matmul,
    build_gru_op,
    build_group_normalization_op,
    build_is_inf_op,
    build_is_nan_op,
    build_leaky_relu_op,
    build_multi_head_attention_op,
    build_matmul_op,
    build_gather_op,
    build_gather_nd_op,
    build_gather_elements_op,
    build_hardmax_op,
    build_roi_align_op,
    build_scatter_elements_op,
    build_scatter_nd_op,
    build_tensor_scatter_op,
    build_unique_op,
    build_non_max_suppression_op,
    build_hardsigmoid_op,
    build_global_average_pool_op,
    build_global_lp_pool_op,
    build_global_max_pool_op,
    build_negative_log_likelihood_loss_op,
    build_logsoftmax_op,
    build_max_op,
    build_min_op,
    build_inverse_op,
    build_if_op,
    build_loop_op,
    build_mish_op,
    build_nonzero_op,
    build_optional_has_element_op,
    build_qgemm_op,
    build_identity_op,
    build_lstm_op,
    build_pad_op,
    build_mod_op,
    build_one_hot_op,
    build_topk_op,
    build_l2_normalization_op,
    build_layer_normalization_op,
    build_mean_variance_normalization_op,
    build_lrn_op,
    build_logistic_op,
    build_mean_op,
    build_matmul_integer_op,
    build_lp_pool_op,
    build_max_roi_pool_op,
    build_max_unpool_op,
    build_pool2d_op,
    build_pow_op,
    build_prelu_op,
    build_qlinear_add_op,
    build_qlinear_average_pool_op,
    build_qlinear_concat_op,
    build_qlinear_conv_op,
    build_qlinear_global_average_pool_op,
    build_qlinear_leaky_relu_op,
    build_qlinear_matmul_op,
    build_qlinear_mul_op,
    build_qlinear_sigmoid_op,
    build_qlinear_softmax_op,
    build_quantize_linear_op,
    build_random_normal_op,
    build_random_normal_like_op,
    build_random_uniform_op,
    build_random_uniform_like_op,
    build_range_op,
    build_bernoulli_op,
    build_blackman_window_op,
    build_cumprod_op,
    build_cumsum_op,
    build_reduce_log_sum_exp_op,
    build_reduce_log_sum_op,
    build_reduce_l1_op,
    build_reduce_l2_op,
    build_reduce_op,
    build_reduce_sum_square_op,
    build_softmax_cross_entropy_loss_op,
    build_reciprocal_op,
    build_reverse_sequence_op,
    build_resize_op,
    build_reshape_op,
    build_rnn_op,
    build_rotary_embedding_op,
    build_selu_op,
    build_shape_op,
    build_size_op,
    build_sinh_op,
    build_slice_op,
    build_split_op,
    build_space_to_depth_op,
    build_stft_op,
    build_string_normalizer_op,
    build_squeeze_op,
    build_tile_op,
    build_softmax_op,
    build_softplus_op,
    build_softsign_op,
    build_shrink_op,
    build_sum_op,
    build_tan_op,
    build_thresholded_relu_op,
    build_transpose_op,
    build_trilu_op,
    build_unary_op,
    build_unsqueeze_op,
    build_where_op,
    is_supported_if_axis0_add_branch_pattern,
    is_supported_if_generic_branch_mux_pattern,
    is_supported_if_nested_reducemin_add_branch_pattern,
    is_supported_if_nms_guard_pattern,
    is_supported_if_sequenceconstruct_add_branch_pattern,
    is_supported_loop_static_unroll_pattern,
    is_supported_loop_while_pattern,
)
from onnx2tf.tflite_builder.ir import (
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)


_CUSTOM_OP_CANDIDATES = {
    "If",
    "Loop",
    "Scan",
    "SequenceConstruct",
    "SequenceAt",
    "SequenceInsert",
    "SequenceErase",
    "SequenceLength",
    "GridSample",
    "RoiAlign",
    "DeformConv",
    "Einsum",
    "DynamicQuantizeLinear",
    "ScatterElements",
    "Unique",
    "TopK",
    "NonMaxSuppression",
    "LSTM",
    "QLinearConv",
    "LogSoftmax",
}








def _get_main_onnx_opset(ctx: Any) -> Optional[int]:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None:
        return None
    for opset in getattr(onnx_model, "opset_import", []):
        domain = str(getattr(opset, "domain", ""))
        if domain in {"", "ai.onnx"}:
            try:
                return int(opset.version)
            except Exception:
                return None
    return None


def _resolve_softmax_axis(node: Any, ctx: Any, rank: int) -> int:
    if "axis" in node.attrs:
        axis = int(node.attrs["axis"])
    else:
        opset = _get_main_onnx_opset(ctx)
        axis = -1 if opset is not None and int(opset) >= 13 else 1
    if axis < 0:
        axis += int(rank)
    return int(axis)


def _is_tensor_consumed_or_graph_output(ctx: Any, tensor_name: str) -> bool:
    normalized_name = str(tensor_name)
    if normalized_name == "":
        return False
    graph_outputs = getattr(ctx, "graph_output_names", set())
    if normalized_name in graph_outputs:
        return True
    consumer_count = int(getattr(ctx, "tensor_consumer_count", {}).get(normalized_name, 0))
    return consumer_count > 0


def _validate_softmax(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    rank_hint = getattr(ctx, "shape_map", {}).get(input_name, None)
    if isinstance(rank_hint, list):
        rank = max(int(rank), int(len(rank_hint)))
    if rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"Softmax requires rank >= 1. shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    axis = _resolve_softmax_axis(node=node, ctx=ctx, rank=rank)
    if axis < 0 or axis >= int(rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Softmax axis is out of range. axis={axis} rank={rank} shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_reshape(node: Any, ctx: Any) -> None:
    shape_name = node.inputs[1].name
    shape_const = ctx.get_constant_array(shape_name)
    if shape_const is not None:
        return

    shape_dtype = str(ctx.get_tensor_dtype(shape_name)).upper()
    if shape_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_input_type",
            message=(
                "Reshape dynamic shape input must be INT32 or INT64 for flatbuffer_direct. "
                f"dtype={shape_dtype} tensor={shape_name}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    shape_tensor_shape = [int(v) for v in ctx.get_tensor_shape(shape_name)]
    if len(shape_tensor_shape) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Reshape dynamic shape input must be rank-1 for flatbuffer_direct. "
                f"shape={shape_tensor_shape} tensor={shape_name}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_slice(node: Any, ctx: Any) -> None:
    input_count = len(node.inputs)
    if input_count not in {1, 3, 4, 5}:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=(
                "Slice supports legacy attr form (input_count=1) or opset>=10 form "
                f"(input_count=3..5). input_count={input_count}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    dynamic_start_name = ""
    if (
        len(node.inputs) > 1
        and str(node.inputs[1].name) != ""
        and ctx.get_constant_array(node.inputs[1].name) is None
        and "starts" not in node.attrs
    ):
        dynamic_start_name = str(node.inputs[1].name)
    starts_values: List[int] = []
    if dynamic_start_name == "":
        starts_values = _extract_slice_indices(
            node=node,
            ctx=ctx,
            input_index=1,
            attr_name="starts",
            label="slice starts",
        )

    dynamic_end_name = ""
    if (
        len(node.inputs) > 2
        and str(node.inputs[2].name) != ""
        and ctx.get_constant_array(node.inputs[2].name) is None
        and "ends" not in node.attrs
    ):
        dynamic_end_name = str(node.inputs[2].name)
    ends_values: List[int] = []
    if dynamic_end_name == "":
        ends_values = _extract_slice_indices(
            node=node,
            ctx=ctx,
            input_index=2,
            attr_name="ends",
            label="slice ends",
        )

    rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    default_axis_len = int(
        len(starts_values)
        if len(starts_values) > 0
        else (len(ends_values) if len(ends_values) > 0 else 1)
    )
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=3,
        attr_name="axes",
        default_if_missing=[int(v) for v in range(default_axis_len)],
    )
    if len(node.inputs) >= 5 and str(node.inputs[4].name) != "":
        steps_arr = _require_const_input(node, ctx, 4, "slice steps")
        steps = [int(v) for v in np.asarray(steps_arr).reshape(-1).tolist()]
    elif "steps" in node.attrs:
        attr_steps = node.attrs.get("steps")
        if isinstance(attr_steps, (list, tuple, np.ndarray)):
            steps = [int(v) for v in np.asarray(attr_steps).reshape(-1).tolist()]
        elif attr_steps is None:
            steps = [1 for _ in range(len(axes))]
        else:
            steps = [int(attr_steps)]
    else:
        steps = [1 for _ in range(len(axes))]

    if len(steps) != len(axes):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/steps length mismatch. "
                f"axes_len={len(axes)} steps_len={len(steps)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(step) == 0 for step in steps):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Slice step must not be 0. steps={steps}",
            node_name=node.name,
            node_op=node.op,
        )

    normalization_rank = int(rank)
    if (
        normalization_rank <= int(axes[0])
        and dynamic_start_name == ""
        and dynamic_end_name == ""
        and len(starts_values) == 1
        and len(ends_values) == 1
        and len(axes) == 1
        and len(steps) == 1
        and int(steps[0]) == -1
        and int(starts_values[0]) == -1
        and int(ends_values[0]) <= -int(np.iinfo(np.int32).max)
        and int(axes[0]) >= 0
    ):
        normalization_rank = int(axes[0]) + 1
    normalized_axes = _normalize_axes_for_rank(axes=axes, rank=normalization_rank, node=node)
    if dynamic_start_name == "" and len(normalized_axes) != len(starts_values):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/axes length mismatch. "
                f"starts_len={len(starts_values)} axes_len={len(normalized_axes)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if (
        dynamic_start_name == ""
        and dynamic_end_name == ""
        and len(starts_values) != len(ends_values)
    ):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/ends length mismatch. "
                f"starts_len={len(starts_values)} ends_len={len(ends_values)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if dynamic_start_name != "" or (dynamic_end_name != "" and len(normalized_axes) >= 1):
        steps_positive = all(int(step) > 0 for step in steps)
        starts_non_negative = (
            all(int(v) >= 0 for v in starts_values)
            if dynamic_start_name == ""
            else True
        )
        is_supported_dynamic_slice = (
            len(normalized_axes) >= 1
            and len(steps) == len(normalized_axes)
            and steps_positive
            and (
                dynamic_start_name != ""
                or len(starts_values) == len(normalized_axes)
            )
            and (
                dynamic_start_name != ""
                or starts_non_negative
            )
            and (
                dynamic_end_name != ""
                or len(ends_values) == len(normalized_axes)
            )
        )
        if not is_supported_dynamic_slice:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Slice dynamic starts/ends lowering supports positive-step slicing "
                    "with rank-1 dynamic vectors matching axes. "
                    f"rank={rank} starts={starts_values} ends={ends_values} "
                    f"axes={normalized_axes} steps={steps}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if dynamic_start_name != "":
            dynamic_start_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_start_name)]
            dynamic_start_len = int(dynamic_start_shape[0]) if len(dynamic_start_shape) == 1 else -1
            expected_len = len(normalized_axes)
            if not (
                len(dynamic_start_shape) == 1
                and (dynamic_start_len <= 0 or dynamic_start_len == expected_len)
            ):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Slice dynamic starts must be rank-1 with length matching axes "
                        "(or unknown length) "
                        "for builtin lowering. "
                        f"shape={dynamic_start_shape} expected_len={expected_len}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        if dynamic_end_name != "":
            dynamic_end_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_end_name)]
            dynamic_end_len = int(dynamic_end_shape[0]) if len(dynamic_end_shape) == 1 else -1
            expected_len = len(normalized_axes)
            if not (
                len(dynamic_end_shape) == 1
                and (dynamic_end_len <= 0 or dynamic_end_len == expected_len)
            ):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Slice dynamic ends must be rank-1 with length matching axes "
                        "(or unknown length) "
                        "for builtin lowering. "
                        f"shape={dynamic_end_shape} expected_len={expected_len}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        return

    if dynamic_end_name != "":
        dynamic_end_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_end_name)]
        dynamic_end_len = int(dynamic_end_shape[0]) if len(dynamic_end_shape) == 1 else -1
        dynamic_end_len_ok = (
            len(dynamic_end_shape) == 1
            and (dynamic_end_len <= 0 or dynamic_end_len == len(starts_values))
        )
        axes_are_prefix = normalized_axes == [int(v) for v in range(len(normalized_axes))]
        starts_non_negative = all(int(v) >= 0 for v in starts_values)
        steps_positive = all(int(v) > 0 for v in steps)
        is_supported_dynamic_end = (
            rank >= 1
            and len(starts_values) >= 1
            and len(starts_values) == len(normalized_axes)
            and len(starts_values) == len(steps)
            and len(starts_values) <= rank
            and dynamic_end_len_ok
            and (axes_are_prefix or len(normalized_axes) == 1)
            and starts_non_negative
            and steps_positive
        )
        if not is_supported_dynamic_end:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Slice dynamic-end lowering supports prefix-axis slicing "
                    "or single-axis slicing (start>=0, step>0). "
                    f"rank={rank} dynamic_end_shape={dynamic_end_shape} "
                    f"starts={starts_values} axes={normalized_axes} steps={steps}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if any(int(step) < 0 for step in steps):
        is_supported_full_reverse = (
            dynamic_end_name == ""
            and len(starts_values) == 1
            and len(ends_values) == 1
            and len(normalized_axes) == 1
            and len(steps) == 1
            and int(steps[0]) == -1
            and int(starts_values[0]) == -1
            and int(ends_values[0]) <= -int(np.iinfo(np.int32).max)
        )
        if not is_supported_full_reverse:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Slice negative steps are not supported except full-axis reverse "
                    "(start=-1,end<=-int32_max,step=-1). "
                    f"starts={starts_values} ends={ends_values} "
                    f"axes={normalized_axes} steps={steps}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_split(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
    raw_shape = None
    if hasattr(ctx, "shape_map"):
        raw_shape = ctx.shape_map.get(str(input_name), None)
    input_rank_unknown = bool(
        len(input_shape) == 1
        and _is_unknown_rank_placeholder_tensor(ctx, input_name)
        and not (isinstance(raw_shape, (list, tuple)) and len(list(raw_shape)) > 0)
    )

    axis_raw = int(node.attrs.get("axis", 0))
    axis = int(axis_raw)
    if axis < 0:
        if input_rank_unknown:
            output_rank_candidates: list[int] = []
            for output in node.outputs:
                output_name = str(output.name)
                if output_name == "":
                    continue
                output_shape = ctx.get_tensor_shape(output_name)
                output_raw_shape = None
                if hasattr(ctx, "shape_map"):
                    output_raw_shape = ctx.shape_map.get(str(output_name), None)
                output_rank_unknown = bool(
                    len(output_shape) == 1
                    and _is_unknown_rank_placeholder_tensor(ctx, output_name)
                    and not (
                        isinstance(output_raw_shape, (list, tuple))
                        and len(list(output_raw_shape)) > 0
                    )
                )
                if not output_rank_unknown:
                    output_rank_candidates.append(len(output_shape))
            if len(output_rank_candidates) > 0:
                rank = int(max(output_rank_candidates))
                axis += rank
        else:
            axis += rank
    if (not input_rank_unknown) and (axis < 0 or axis >= rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Split axis out of range. axis={axis} rank={rank}",
            node_name=node.name,
            node_op=node.op,
        )

    split_sizes: list[int] | None = None
    if len(node.inputs) >= 2:
        split_arr = _require_const_input(node, ctx, 1, "split sizes")
        split_sizes = [int(v) for v in np.asarray(split_arr).reshape(-1).tolist()]
    elif "split" in node.attrs:
        split_attr = node.attrs.get("split")
        if isinstance(split_attr, (list, tuple, np.ndarray)):
            split_sizes = [int(v) for v in np.asarray(split_attr).reshape(-1).tolist()]
        elif split_attr is not None:
            split_sizes = [int(split_attr)]

    output_count = len(node.outputs)
    if split_sizes is not None and len(split_sizes) != output_count:
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Split split size count must match outputs. "
                f"split_len={len(split_sizes)} outputs={output_count}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    axis_dim = int(input_shape[axis]) if axis < len(input_shape) else -1
    # Some quantized models carry incomplete shape metadata in direct lowering.
    # When explicit split sizes are present, trust them even if inferred axis
    # dimension disagrees with metadata.
    if split_sizes is None:
        if axis_dim <= 0:
            raise NodeValidationError(
                reason_code="unsupported_shape_inference",
                message=(
                    "Split without explicit split sizes requires known axis dimension."
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if axis_dim % output_count != 0:
            raise NodeValidationError(
                reason_code="invalid_input_shape",
                message=(
                    f"Split without explicit sizes requires divisible axis. "
                    f"axis_dim={axis_dim} outputs={output_count}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_transpose(node: Any, ctx: Any) -> None:
    if len(node.inputs) >= 2:
        _require_const_input(node, ctx, 1, "transpose permutation")
        return
    if "perm" in node.attrs:
        perm_attr = node.attrs.get("perm")
        if isinstance(perm_attr, (list, tuple)):
            _ = [int(v) for v in perm_attr]
        elif perm_attr is not None:
            _ = int(perm_attr)
        return
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Transpose input rank must be > 0. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_conv(node: Any, ctx: Any) -> None:
    def _has_unresolved_raw_shape(tensor_name: str) -> bool:
        if not hasattr(ctx, "shape_map") or not isinstance(ctx.shape_map, dict):
            return False
        raw_shape = ctx.shape_map.get(str(tensor_name), None)
        if raw_shape is None:
            return True
        if not isinstance(raw_shape, (list, tuple)):
            return True
        if len(list(raw_shape)) == 0:
            return True
        for dim in list(raw_shape):
            if not isinstance(dim, (int, np.integer)) or int(dim) <= 0:
                return True
        return False

    weights = _require_const_input(node, ctx, 1, "conv weights")
    if weights.ndim not in [3, 4, 5]:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"Conv weight rank must be 3, 4, or 5. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    input_is_unknown_placeholder = _is_unknown_rank_placeholder_tensor(ctx, node.inputs[0].name)
    output_is_unknown_placeholder = _is_unknown_rank_placeholder_tensor(ctx, node.outputs[0].name)
    input_has_unresolved_raw_shape = _has_unresolved_raw_shape(node.inputs[0].name)
    output_has_unresolved_raw_shape = _has_unresolved_raw_shape(node.outputs[0].name)
    if int(weights.ndim) == 4:
        if (
            (len(input_shape) != 4 and not input_is_unknown_placeholder and not input_has_unresolved_raw_shape)
            or (len(output_shape) != 4 and not output_is_unknown_placeholder and not output_has_unresolved_raw_shape)
        ):
            raise NodeValidationError(
                reason_code="unsupported_tensor_rank",
                message=f"Conv2D input/output rank must be 4. input_shape={input_shape} output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
    elif int(weights.ndim) == 3:
        if (
            (len(input_shape) != 3 and not input_is_unknown_placeholder and not input_has_unresolved_raw_shape)
            or (len(output_shape) != 3 and not output_is_unknown_placeholder and not output_has_unresolved_raw_shape)
        ):
            raise NodeValidationError(
                reason_code="unsupported_tensor_rank",
                message=f"Conv1D input/output rank must be 3. input_shape={input_shape} output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
    else:
        if (
            (len(input_shape) != 5 and not input_is_unknown_placeholder and not input_has_unresolved_raw_shape)
            or (len(output_shape) != 5 and not output_is_unknown_placeholder and not output_has_unresolved_raw_shape)
        ):
            raise NodeValidationError(
                reason_code="unsupported_tensor_rank",
                message=f"Conv3D input/output rank must be 5. input_shape={input_shape} output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Conv group must be > 0. group={group}",
            node_name=node.name,
            node_op=node.op,
        )
    in_channels = int(input_shape[1]) if len(input_shape) > 1 else -1
    out_channels = int(weights.shape[0])
    weight_in_channels_per_group = int(weights.shape[1])
    if in_channels <= 0:
        inferred_channels = int(weight_in_channels_per_group) * int(group)
        if inferred_channels > 0:
            in_channels = int(inferred_channels)
    is_depthwise = (
        group > 1
        and weight_in_channels_per_group == 1
        and (out_channels % group) == 0
    )
    if int(weights.ndim) == 5 and group != 1:
        if in_channels <= 0 or in_channels % group != 0:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv3D requires input channels divisible by group. "
                    f"group={group} in_channels={in_channels}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if weight_in_channels_per_group != (in_channels // group):
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv3D weight shape is inconsistent with input channels/group. "
                    f"group={group} in_channels={in_channels} "
                    f"weight_in_channels_per_group={weight_in_channels_per_group}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if out_channels % group != 0:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv3D requires output channels divisible by group. "
                    f"group={group} out_channels={out_channels}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return
    if group != 1 and not is_depthwise:
        input_rank_supported = (
            len(input_shape) == 4
            or input_is_unknown_placeholder
            or input_has_unresolved_raw_shape
        )
        output_rank_supported = (
            len(output_shape) == 4
            or output_is_unknown_placeholder
            or output_has_unresolved_raw_shape
        )
        if (
            int(weights.ndim) != 4
            or not input_rank_supported
            or not output_rank_supported
        ):
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv currently supports rank-4 only. "
                    f"group={group} input_shape={input_shape} output_shape={output_shape} "
                    f"weight_shape={list(weights.shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if in_channels <= 0 or in_channels % group != 0:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv requires input channels divisible by group. "
                    f"group={group} in_channels={in_channels}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if weight_in_channels_per_group != (in_channels // group):
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv weight shape is inconsistent with input channels/group. "
                    f"group={group} in_channels={in_channels} "
                    f"weight_in_channels_per_group={weight_in_channels_per_group}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if out_channels % group != 0:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "Grouped Conv requires output channels divisible by group. "
                    f"group={group} out_channels={out_channels}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return


def _validate_fused_conv(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    input_is_unknown_placeholder = _is_unknown_rank_placeholder_tensor(ctx, node.inputs[0].name)
    output_is_unknown_placeholder = _is_unknown_rank_placeholder_tensor(ctx, node.outputs[0].name)
    if not input_is_unknown_placeholder and not output_is_unknown_placeholder:
        _validate_conv(node, ctx)
    else:
        weights = _require_const_input(node, ctx, 1, "conv weights")
        if weights.ndim not in [3, 4]:
            raise NodeValidationError(
                reason_code="unsupported_weight_rank",
                message=f"FusedConv weight rank must be 3 or 4. weight_shape={list(weights.shape)}",
                node_name=node.name,
                node_op=node.op,
            )
        if int(weights.ndim) == 4:
            if not input_is_unknown_placeholder and len(input_shape) != 4:
                raise NodeValidationError(
                    reason_code="unsupported_tensor_rank",
                    message=(
                        "FusedConv2D input rank must be 4 (or unknown placeholder rank=1). "
                        f"input_shape={input_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if not output_is_unknown_placeholder and len(output_shape) != 4:
                raise NodeValidationError(
                    reason_code="unsupported_tensor_rank",
                    message=(
                        "FusedConv2D output rank must be 4 (or unknown placeholder rank=1). "
                        f"output_shape={output_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        else:
            if not input_is_unknown_placeholder and len(input_shape) != 3:
                raise NodeValidationError(
                    reason_code="unsupported_tensor_rank",
                    message=(
                        "FusedConv1D input rank must be 3 (or unknown placeholder rank=1). "
                        f"input_shape={input_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if not output_is_unknown_placeholder and len(output_shape) != 3:
                raise NodeValidationError(
                    reason_code="unsupported_tensor_rank",
                    message=(
                        "FusedConv1D output rank must be 3 (or unknown placeholder rank=1). "
                        f"output_shape={output_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        group = int(node.attrs.get("group", 1))
        if group <= 0:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"FusedConv group must be > 0. group={group}",
                node_name=node.name,
                node_op=node.op,
            )

    activation_raw = node.attrs.get("activation", "Relu")
    if isinstance(activation_raw, (bytes, bytearray)):
        activation = activation_raw.decode("utf-8")
    else:
        activation = str(activation_raw)
    activation_key = str(activation).lower()
    supported = {"relu", "tanh", "sigmoid", "leakyrelu", "clip", "hardsigmoid"}
    if activation_key not in supported:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "FusedConv activation must be one of "
                "[Relu, Tanh, Sigmoid, LeakyRelu, Clip, HardSigmoid]. "
                f"activation={activation}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    params_attr = node.attrs.get("activation_params", [])
    if params_attr is None:
        params: list[Any] = []
    elif isinstance(params_attr, np.ndarray):
        params = list(np.asarray(params_attr).reshape(-1))
    elif isinstance(params_attr, (list, tuple)):
        params = []
        for item in params_attr:
            if isinstance(item, np.ndarray):
                params.extend(list(np.asarray(item).reshape(-1)))
            elif isinstance(item, (list, tuple)):
                params.extend(list(np.asarray(item).reshape(-1)))
            else:
                params.append(item)
    else:
        params = [params_attr]

    def _to_optional_float(value: Any) -> float | None:
        if value is None:
            return None
        arr = np.asarray(value)
        if int(arr.size) == 0:
            return None
        try:
            return float(arr.reshape(-1)[0])
        except Exception:
            return None

    if activation_key == "leakyrelu":
        if len(params) > 0 and _to_optional_float(params[0]) is None:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv LeakyRelu alpha must be scalar-convertible when provided. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    elif activation_key == "clip":
        if len(params) == 0:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv Clip requires activation_params with at least one bound. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        min_value = _to_optional_float(params[0]) if len(params) >= 1 else None
        max_value = _to_optional_float(params[1]) if len(params) >= 2 else None
        if len(params) >= 1 and params[0] is not None and min_value is None:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv Clip minimum must be scalar-convertible when provided. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if len(params) >= 2 and params[1] is not None and max_value is None:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv Clip maximum must be scalar-convertible when provided. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if min_value is not None and max_value is not None and float(min_value) > float(max_value):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv Clip minimum must be <= maximum. "
                    f"min={min_value} max={max_value}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if min_value is None and max_value is None:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv Clip requires at least one concrete bound. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    elif activation_key == "hardsigmoid":
        if len(params) < 2:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv HardSigmoid requires activation_params [alpha, beta]. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        alpha = _to_optional_float(params[0])
        beta = _to_optional_float(params[1])
        if alpha is None or beta is None:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "FusedConv HardSigmoid alpha/beta must be scalar-convertible. "
                    f"activation_params={params_attr}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _normalize_col2im_pair_attr(
    values: Any,
    *,
    default: int,
    node: Any,
    label: str,
) -> list[int]:
    vals = [int(v) for v in list(values)] if values is not None else []
    if len(vals) == 0:
        return [int(default), int(default)]
    if len(vals) == 1:
        return [int(vals[0]), int(vals[0])]
    if len(vals) == 2:
        return [int(vals[0]), int(vals[1])]
    raise NodeValidationError(
        reason_code="unsupported_attribute_value",
        message=f"Col2Im {label} must have length 1 or 2. {label}={vals}",
        node_name=node.name,
        node_op=node.op,
    )


def _normalize_col2im_pads_attr(values: Any, *, node: Any) -> list[int]:
    pads = [int(v) for v in list(values)] if values is not None else []
    if len(pads) == 0:
        return [0, 0, 0, 0]
    if len(pads) == 2:
        return [int(pads[0]), int(pads[1]), int(pads[0]), int(pads[1])]
    if len(pads) == 4:
        return [int(pads[0]), int(pads[1]), int(pads[2]), int(pads[3])]
    raise NodeValidationError(
        reason_code="unsupported_attribute_value",
        message=f"Col2Im pads must have length 2 or 4. pads={pads}",
        node_name=node.name,
        node_op=node.op,
    )


def _validate_col2im(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_dtype",
            message=(
                "Col2Im currently supports FLOAT16/FLOAT32 input/output in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
    if len(input_shape) != 3 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "Col2Im expects input rank=3 and output rank=4 in flatbuffer_direct. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in input_shape + output_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Col2Im requires static positive input/output shapes in flatbuffer_direct. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    image_shape = _require_const_input(node, ctx, 1, "Col2Im image_shape")
    block_shape = _require_const_input(node, ctx, 2, "Col2Im block_shape")
    image_shape_values = np.asarray(image_shape).reshape(-1)
    block_shape_values = np.asarray(block_shape).reshape(-1)
    if int(image_shape_values.size) != 2 or int(block_shape_values.size) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Col2Im image_shape/block_shape must each contain exactly 2 elements. "
                f"image_shape={list(image_shape_values.shape)} block_shape={list(block_shape_values.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    h_img = int(image_shape_values[0])
    w_img = int(image_shape_values[1])
    k_h = int(block_shape_values[0])
    k_w = int(block_shape_values[1])
    if min(h_img, w_img, k_h, k_w) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Col2Im image_shape/block_shape values must be > 0. "
                f"image_shape={[h_img, w_img]} block_shape={[k_h, k_w]}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    strides = _normalize_col2im_pair_attr(
        node.attrs.get("strides", [1, 1]),
        default=1,
        node=node,
        label="strides",
    )
    dilations = _normalize_col2im_pair_attr(
        node.attrs.get("dilations", [1, 1]),
        default=1,
        node=node,
        label="dilations",
    )
    pads = _normalize_col2im_pads_attr(node.attrs.get("pads", [0, 0, 0, 0]), node=node)
    if any(int(v) < 0 for v in list(strides) + list(dilations) + list(pads)):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Col2Im strides/dilations/pads must be non-negative. "
                f"strides={strides} dilations={dilations} pads={pads}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in strides + dilations):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Col2Im strides/dilations must be > 0. strides={strides} dilations={dilations}",
            node_name=node.name,
            node_op=node.op,
        )

    pad_top, pad_left, pad_bottom, pad_right = [int(v) for v in pads]
    dilation_h, dilation_w = [int(v) for v in dilations]
    stride_h, stride_w = [int(v) for v in strides]
    eff_k_h = (int(k_h) - 1) * int(dilation_h) + 1
    eff_k_w = (int(k_w) - 1) * int(dilation_w) + 1
    h_pad = int(h_img) + int(pad_top) + int(pad_bottom)
    w_pad = int(w_img) + int(pad_left) + int(pad_right)
    out_h = int((int(h_pad) - int(eff_k_h)) // int(stride_h) + 1)
    out_w = int((int(w_pad) - int(eff_k_w)) // int(stride_w) + 1)
    if out_h <= 0 or out_w <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Col2Im folded spatial shape must be positive. "
                f"out_h={out_h} out_w={out_w}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    n, d1, d2 = [int(v) for v in input_shape]
    out_n, out_c, out_h_out, out_w_out = [int(v) for v in output_shape]
    if int(out_n) != int(n) or int(out_h_out) != int(h_img) or int(out_w_out) != int(w_img):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Col2Im output shape does not match input/image_shape. "
                f"input_shape={input_shape} output_shape={output_shape} image_shape={[h_img, w_img]}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    k_prod = int(k_h) * int(k_w)
    out_hw = int(out_h) * int(out_w)
    canonical_valid = bool(int(d1) % int(k_prod) == 0 and int(d2) == int(out_hw))
    swapped_valid = bool(int(d2) % int(k_prod) == 0 and int(d1) == int(out_hw))
    if canonical_valid and int(d1) // int(k_prod) != int(out_c):
        canonical_valid = False
    if swapped_valid and int(d2) // int(k_prod) != int(out_c):
        swapped_valid = False
    if not canonical_valid and not swapped_valid:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Col2Im input layout must resolve to [N,C*K,L] or [N,L,C*K] with expected output C/H/W. "
                f"input_shape={input_shape} output_shape={output_shape} k_prod={k_prod} out_hw={out_hw}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_global_average_pool(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if len(input_shape) < 3 and not _is_unknown_rank_placeholder_tensor(ctx, node.inputs[0].name):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GlobalAveragePool input rank must be >=3. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_global_max_pool(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if len(input_shape) < 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GlobalMaxPool input rank must be >=3. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_global_lp_pool(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if len(input_shape) < 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GlobalLpPool input rank must be >=3. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "GlobalLpPool input/output dtype must be FLOAT16/FLOAT32. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    p = float(node.attrs.get("p", 2.0))
    if not np.isfinite(p) or p <= 0.0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GlobalLpPool p must be finite and > 0. p={p}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_max_unpool(node: Any, ctx: Any) -> None:
    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    indices_shape = _tensor_shape_with_signature(ctx, node.inputs[1].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) != 4 or len(indices_shape) != 4 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MaxUnpool supports rank-4 input/indices/output only in flatbuffer_direct. "
                f"input_shape={input_shape} indices_shape={indices_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(a) != int(b) for a, b in zip(input_shape, indices_shape)):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "MaxUnpool input and indices shapes must match in flatbuffer_direct. "
                f"input_shape={input_shape} indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not all(int(v) > 0 for v in input_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "MaxUnpool input shape must be static positive in flatbuffer_direct. "
                f"input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not all(int(v) > 0 for v in output_shape):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "MaxUnpool output shape must be static positive in flatbuffer_direct. "
                f"output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(output_shape[0]) != int(input_shape[0]) or int(output_shape[1]) != int(input_shape[1]):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "MaxUnpool output batch/channel dimensions must match input in flatbuffer_direct. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    indices_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if not _is_numeric_dtype(input_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"MaxUnpool input dtype must be numeric. input_dtype={input_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_integer_dtype(indices_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"MaxUnpool indices dtype must be integer. indices_dtype={indices_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if str(output_dtype) != str(input_dtype):
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "MaxUnpool output dtype must match input dtype in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    kernel_shape = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
    if len(kernel_shape) != 2 or any(int(v) <= 0 for v in kernel_shape):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"MaxUnpool kernel_shape must be length-2 positive. kernel_shape={kernel_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(strides) == 0:
        strides = [1, 1]
    elif len(strides) == 1:
        strides = [int(strides[0]), int(strides[0])]
    if len(strides) != 2 or any(int(v) <= 0 for v in strides):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"MaxUnpool strides must be length-2 positive. strides={strides}",
            node_name=node.name,
            node_op=node.op,
        )
    pads = [int(v) for v in list(node.attrs.get("pads", []))]
    if len(pads) not in {0, 4}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"MaxUnpool pads must be empty or length-4. pads={pads}",
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) != 0 for v in pads):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"MaxUnpool builtin path currently supports zero pads only. pads={pads}",
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) > 2 and str(node.inputs[2].name) != "":
        output_shape_const = ctx.get_constant_array(node.inputs[2].name)
        if output_shape_const is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="MaxUnpool output_shape input must be constant in flatbuffer_direct.",
                node_name=node.name,
                node_op=node.op,
            )
        output_shape_values = [int(v) for v in np.asarray(output_shape_const).reshape(-1).tolist()]
        if len(output_shape_values) != 4 or not all(int(v) > 0 for v in output_shape_values):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "MaxUnpool output_shape input must contain 4 positive values in flatbuffer_direct. "
                    f"output_shape_values={output_shape_values}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if any(int(a) != int(b) for a, b in zip(output_shape_values, output_shape)):
            raise NodeValidationError(
                reason_code="invalid_output_shape",
                message=(
                    "MaxUnpool output_shape input must match graph output shape in flatbuffer_direct. "
                    f"output_shape_values={output_shape_values} output_shape={output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_max_roi_pool(node: Any, ctx: Any) -> None:
    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MaxRoiPool builtin path currently supports rank-4 input/output only. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in input_shape + output_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "MaxRoiPool builtin path requires static positive input/output shapes. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "MaxRoiPool input/output dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(output_shape[1]) != int(input_shape[1]):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "MaxRoiPool output channels must match input channels. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    pooled_shape = [int(v) for v in list(node.attrs.get("pooled_shape", []))]
    if len(pooled_shape) != 2 or any(int(v) <= 0 for v in pooled_shape):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"MaxRoiPool pooled_shape must be length-2 positive. pooled_shape={pooled_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(output_shape[2]) != int(pooled_shape[0]) or int(output_shape[3]) != int(pooled_shape[1]):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "MaxRoiPool output pooled dims must match pooled_shape. "
                f"output_shape={output_shape} pooled_shape={pooled_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    rois = ctx.get_constant_array(node.inputs[1].name)
    if rois is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="MaxRoiPool builtin path currently requires constant rois input.",
            node_name=node.name,
            node_op=node.op,
        )
    rois_arr = np.asarray(rois, dtype=np.float32).reshape(-1, 5)
    if int(rois_arr.shape[0]) != int(output_shape[0]):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "MaxRoiPool output batch dimension must equal number of constant rois. "
                f"rois_shape={list(rois_arr.shape)} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    batch_size = int(input_shape[0])
    for roi in rois_arr:
        batch_idx = int(roi[0])
        if batch_idx < 0 or batch_idx >= batch_size:
            raise NodeValidationError(
                reason_code="unsupported_input_value",
                message=(
                    "MaxRoiPool roi batch index is out of range for builtin path. "
                    f"batch_idx={batch_idx} batch_size={batch_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_conv_transpose(node: Any, ctx: Any) -> None:
    weights = _require_const_input(node, ctx, 1, "convtranspose weights")
    if weights.ndim not in [3, 4, 5]:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"ConvTranspose weight rank must be 3, 4, or 5. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if int(weights.ndim) == 4 and len(input_shape) not in [1, 4]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ConvTranspose2D input rank must be 4 (or unknown placeholder rank=1). input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.ndim) == 3 and len(input_shape) not in [1, 3]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ConvTranspose1D input rank must be 3 (or unknown placeholder rank=1). input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.ndim) == 5 and len(input_shape) not in [1, 5]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ConvTranspose3D input rank must be 5 (or unknown placeholder rank=1). input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    group = int(node.attrs.get("group", 1))
    if group <= 0:
        raise NodeValidationError(
            reason_code="unsupported_grouped_convolution",
            message=f"ConvTranspose group must be positive. group={group}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.ndim) == 4 and int(weights.shape[0]) % int(group) != 0:
        raise NodeValidationError(
            reason_code="unsupported_grouped_convolution",
            message=(
                "ConvTranspose2D weights/input channels must be divisible by group. "
                f"weight_shape={list(weights.shape)} group={group}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.ndim) == 3:
        dilations = [int(v) for v in list(node.attrs.get("dilations", [1]))]
        output_padding = [int(v) for v in list(node.attrs.get("output_padding", []))]
        strides = [int(v) for v in list(node.attrs.get("strides", [1]))]
        if len(strides) == 0:
            strides = [1]
        if len(strides) == 2:
            strides = [int(strides[1])]
        elif len(strides) != 1:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose1D strides must have length 1. strides={strides}",
                node_name=node.name,
                node_op=node.op,
            )
        if dilations not in [[1], [1, 1]]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose1D dilations must be [1]. dilations={dilations}",
                node_name=node.name,
                node_op=node.op,
            )
        if len(output_padding) == 0:
            output_padding = [0]
        elif len(output_padding) == 2:
            output_padding = [int(output_padding[1])]
        elif len(output_padding) != 1:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose1D output_padding must have length 1. output_padding={output_padding}",
                node_name=node.name,
                node_op=node.op,
            )
        if output_padding[0] < 0 or output_padding[0] >= int(strides[0]):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "ConvTranspose1D output_padding must satisfy "
                    f"0 <= output_padding < stride. output_padding={output_padding} strides={strides}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    elif int(weights.ndim) == 4:
        dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
        output_padding = [int(v) for v in list(node.attrs.get("output_padding", []))]
        strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
        if len(strides) == 0:
            strides = [1, 1]
        elif len(strides) == 1:
            strides = [int(strides[0]), int(strides[0])]
        elif len(strides) != 2:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose strides must have length 2. strides={strides}",
                node_name=node.name,
                node_op=node.op,
            )
        if len(dilations) == 0:
            dilations = [1, 1]
        elif len(dilations) == 1:
            dilations = [int(dilations[0]), int(dilations[0])]
        elif len(dilations) != 2:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose dilations must have length 2. dilations={dilations}",
                node_name=node.name,
                node_op=node.op,
            )
        if any(int(v) <= 0 for v in dilations):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose dilations must be positive. dilations={dilations}",
                node_name=node.name,
                node_op=node.op,
            )
        if len(output_padding) == 0:
            output_padding = [0, 0]
        elif len(output_padding) == 1:
            output_padding = [int(output_padding[0]), int(output_padding[0])]
        elif len(output_padding) != 2:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose output_padding must have length 2. output_padding={output_padding}",
                node_name=node.name,
                node_op=node.op,
            )
        if any(v < 0 for v in output_padding):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose output_padding must be non-negative. output_padding={output_padding}",
                node_name=node.name,
                node_op=node.op,
            )
        if any(int(v) >= int(s) for v, s in zip(output_padding, strides)):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "ConvTranspose output_padding must satisfy "
                    f"0 <= output_padding < stride. output_padding={output_padding} strides={strides}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    else:
        if int(group) != 1:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=f"ConvTranspose3D currently supports group=1 only. group={group}",
                node_name=node.name,
                node_op=node.op,
            )
        dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1, 1]))]
        output_padding = [int(v) for v in list(node.attrs.get("output_padding", []))]
        strides = [int(v) for v in list(node.attrs.get("strides", [1, 1, 1]))]
        if len(strides) == 0:
            strides = [1, 1, 1]
        elif len(strides) == 1:
            strides = [int(strides[0]), int(strides[0]), int(strides[0])]
        elif len(strides) != 3:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose3D strides must have length 3. strides={strides}",
                node_name=node.name,
                node_op=node.op,
            )
        if dilations != [1, 1, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose3D dilations must be [1,1,1]. dilations={dilations}",
                node_name=node.name,
                node_op=node.op,
            )
        if len(output_padding) == 0:
            output_padding = [0, 0, 0]
        elif len(output_padding) == 1:
            output_padding = [int(output_padding[0]), int(output_padding[0]), int(output_padding[0])]
        elif len(output_padding) != 3:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose3D output_padding must have length 3. output_padding={output_padding}",
                node_name=node.name,
                node_op=node.op,
            )
        if any(v < 0 for v in output_padding):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose3D output_padding must be non-negative. output_padding={output_padding}",
                node_name=node.name,
                node_op=node.op,
            )
        if any(int(v) >= int(s) for v, s in zip(output_padding, strides)):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "ConvTranspose3D output_padding must satisfy "
                    f"0 <= output_padding < stride. output_padding={output_padding} strides={strides}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    if len(node.inputs) >= 3:
        _require_const_input(node, ctx, 2, "convtranspose bias")


def _validate_deform_conv(node: Any, ctx: Any) -> None:
    input_name = str(node.inputs[0].name)
    weight_name = str(node.inputs[1].name)
    offset_name = str(node.inputs[2].name)
    bias_name = str(node.inputs[3].name) if len(node.inputs) >= 4 else ""
    mask_name = str(node.inputs[4].name) if len(node.inputs) >= 5 else ""
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    offset_shape = _tensor_shape_with_signature(ctx, offset_name)
    if len(input_shape) != 4 or len(output_shape) != 4 or len(offset_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "DeformConv supports rank-4 input/output/offset only in flatbuffer_direct. "
                f"input_shape={input_shape} offset_shape={offset_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    offset_dtype = str(ctx.get_tensor_dtype(offset_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"DeformConv input dtype must be FLOAT16/FLOAT32. input_dtype={input_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if offset_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"DeformConv offset dtype must be FLOAT16/FLOAT32. offset_dtype={offset_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"DeformConv output dtype must be FLOAT16/FLOAT32. output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    weights = ctx.get_constant_array(weight_name)
    if weights is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message=f"DeformConv weights must be constant. weight={weight_name}",
            node_name=node.name,
            node_op=node.op,
        )
    weights = np.asarray(weights)
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"DeformConv weights must be rank-4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )

    group = int(node.attrs.get("group", 1))
    offset_group = int(node.attrs.get("offset_group", 1))
    if group <= 0 or offset_group <= 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"DeformConv group and offset_group must be positive. group={group} offset_group={offset_group}",
            node_name=node.name,
            node_op=node.op,
        )
    if group != 1 or offset_group != 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "DeformConv builtin lowering is currently limited to group=1 and offset_group=1 for LiteRT runtime safety. "
                f"group={group} offset_group={offset_group}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    if len(dilations) != 2 or len(strides) != 2 or len(pads) != 4:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "DeformConv dilations/strides/pads must have lengths 2/2/4. "
                f"dilations={dilations} strides={strides} pads={pads}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    kernel_shape = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
    if len(kernel_shape) == 0:
        kernel_shape = [int(weights.shape[2]), int(weights.shape[3])]
    if len(kernel_shape) != 2 or any(int(v) <= 0 for v in kernel_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"DeformConv kernel shape must resolve to 2 positive dims. kernel_shape={kernel_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.shape[2]) != int(kernel_shape[0]) or int(weights.shape[3]) != int(kernel_shape[1]):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv kernel_shape attribute must match weights. "
                f"kernel_shape={kernel_shape} weight_shape={list(weights.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    in_channels = int(input_shape[1])
    in_h = int(input_shape[2])
    in_w = int(input_shape[3])
    out_channels = int(output_shape[1])
    out_h = int(output_shape[2])
    out_w = int(output_shape[3])
    if any(int(v) <= 0 for v in [in_channels, in_h, in_w, out_channels, out_h, out_w]):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv currently requires static positive channel/spatial dims. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if in_channels % group != 0 or in_channels % offset_group != 0 or out_channels % group != 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv channels must be divisible by group/offset_group. "
                f"in_channels={in_channels} out_channels={out_channels} group={group} offset_group={offset_group}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.shape[0]) != out_channels:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv output channels must match weights. "
                f"out_channels={out_channels} weight_shape={list(weights.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(weights.shape[1]) != int(in_channels // group):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv weights/input channels are inconsistent with group. "
                f"weight_shape={list(weights.shape)} in_channels={in_channels} group={group}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    expected_offset_channels = int(2 * offset_group * kernel_shape[0] * kernel_shape[1])
    if int(offset_shape[1]) != expected_offset_channels:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv offset channels are inconsistent with offset_group/kernel_shape. "
                f"offset_shape={offset_shape} expected_offset_channels={expected_offset_channels}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(offset_shape[2]) != out_h or int(offset_shape[3]) != out_w:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DeformConv offset spatial dims must match output spatial dims. "
                f"offset_shape={offset_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if bias_name != "":
        bias = ctx.get_constant_array(bias_name)
        if bias is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="DeformConv bias must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        bias = np.asarray(bias).reshape(-1)
        if int(bias.size) != out_channels:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "DeformConv bias size must match output channels. "
                    f"bias_size={int(bias.size)} out_channels={out_channels}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if mask_name != "":
        mask_shape = _tensor_shape_with_signature(ctx, mask_name)
        if len(mask_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=f"DeformConv mask must be rank-4 when provided. mask_shape={mask_shape}",
                node_name=node.name,
                node_op=node.op,
            )
        mask_dtype = str(ctx.get_tensor_dtype(mask_name)).upper()
        if mask_dtype not in {"FLOAT16", "FLOAT32"}:
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=f"DeformConv mask dtype must be FLOAT16/FLOAT32. mask_dtype={mask_dtype}",
                node_name=node.name,
                node_op=node.op,
            )
        expected_mask_channels = int(offset_group * kernel_shape[0] * kernel_shape[1])
        if int(mask_shape[1]) != expected_mask_channels or int(mask_shape[2]) != out_h or int(mask_shape[3]) != out_w:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "DeformConv mask shape must match output spatial dims and offset_group*kernel area. "
                    f"mask_shape={mask_shape} expected_mask_channels={expected_mask_channels} output_shape={output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_pool(node: Any, ctx: Any) -> None:
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if node.op == "MaxPool":
        if ceil_mode not in [0, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"MaxPool ceil_mode must be 0 or 1. got={ceil_mode}",
                node_name=node.name,
                node_op=node.op,
            )
        # 2-output MaxPool (values + argmax indices) is supported only for a
        # restricted shape-safe form in flatbuffer_direct lowering.
        if len(node.outputs) == 2:
            input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
            input_rank = len(input_shape)
            storage_order = int(node.attrs.get("storage_order", 0))
            kernel = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
            if input_rank == 3:
                strides = [int(v) for v in list(node.attrs.get("strides", [1]))]
                dilations = [int(v) for v in list(node.attrs.get("dilations", [1]))]
            else:
                strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
                dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
            auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
            pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
            if storage_order != 0:
                raise NodeValidationError(
                    reason_code="unsupported_attribute_value",
                    message=(
                        "MaxPool with indices requires storage_order=0 in "
                        f"flatbuffer_direct. got={storage_order}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if input_rank == 3:
                if len(pads) == 0:
                    pads = [0, 0]
                elif len(pads) == 1:
                    pads = [int(pads[0]), int(pads[0])]
                elif len(pads) == 2:
                    pads = [int(pads[0]), int(pads[1])]
                elif len(pads) == 4:
                    pads = [int(pads[1]), int(pads[3])]
                else:
                    raise NodeValidationError(
                        reason_code="unsupported_attribute_value",
                        message=f"MaxPool1D with indices pads must have length 0/1/2/4. got pads={pads}",
                        node_name=node.name,
                        node_op=node.op,
                    )
                valid_1d_pairs = (
                    (kernel == [1] and strides == [1])
                    or (kernel == [2] and strides == [2])
                )
                if not valid_1d_pairs:
                    raise NodeValidationError(
                        reason_code="unsupported_attribute_value",
                        message=(
                            "MaxPool1D with indices currently supports only "
                            "kernel/strides [1]/[1] or [2]/[2]. "
                            f"got kernel_shape={kernel} strides={strides}"
                        ),
                        node_name=node.name,
                        node_op=node.op,
                    )
                if dilations != [1]:
                    raise NodeValidationError(
                        reason_code="unsupported_attribute_value",
                        message=(
                            "MaxPool1D with indices currently supports only "
                            f"dilations=[1]. got dilations={dilations}"
                        ),
                        node_name=node.name,
                        node_op=node.op,
                    )
                if kernel == [1] and any(int(v) != 0 for v in pads):
                    raise NodeValidationError(
                        reason_code="unsupported_attribute_value",
                        message=(
                            "MaxPool1D with indices kernel=[1] currently supports zero pads only. "
                            f"got pads={pads}"
                        ),
                        node_name=node.name,
                        node_op=node.op,
                    )
            else:
                if len(pads) < 4:
                    pads = [0, 0, 0, 0]
                if kernel != [2, 2] or strides != [2, 2]:
                    raise NodeValidationError(
                        reason_code="unsupported_attribute_value",
                        message=(
                            "MaxPool with indices currently supports only "
                            "kernel_shape=[2,2], strides=[2,2]. "
                            f"got kernel_shape={kernel} strides={strides}"
                        ),
                        node_name=node.name,
                        node_op=node.op,
                    )
                if dilations != [1, 1]:
                    raise NodeValidationError(
                        reason_code="unsupported_attribute_value",
                        message=(
                            "MaxPool with indices currently supports only "
                            f"dilations=[1,1]. got dilations={dilations}"
                        ),
                        node_name=node.name,
                        node_op=node.op,
                    )
            if ceil_mode != 0:
                raise NodeValidationError(
                    reason_code="unsupported_attribute_value",
                    message=(
                        "MaxPool with indices currently supports only "
                        f"ceil_mode=0. got={ceil_mode}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if auto_pad not in {"NOTSET", "VALID"}:
                raise NodeValidationError(
                    reason_code="unsupported_attribute_value",
                    message=(
                        "MaxPool with indices currently supports auto_pad "
                        f"NOTSET/VALID only. got={auto_pad}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if input_rank != 3 and any(int(v) != 0 for v in pads):
                raise NodeValidationError(
                    reason_code="unsupported_attribute_value",
                    message=(
                        "MaxPool with indices currently supports zero pads only. "
                        f"got pads={pads}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
    else:
        if ceil_mode not in [0, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"AveragePool ceil_mode must be 0 or 1. got={ceil_mode}",
                node_name=node.name,
                node_op=node.op,
            )
        count_include_pad = int(node.attrs.get("count_include_pad", 0))
        if count_include_pad not in [0, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"AveragePool count_include_pad must be 0 or 1. got={count_include_pad}",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_lp_pool(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"LpPool supports rank-4 input/output only. input_shape={input_shape} output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "LpPool input/output dtype must be FLOAT16/FLOAT32. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if len(kernel) != 2 or len(strides) != 2 or len(dilations) != 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "LpPool requires 2D kernel/strides/dilations. "
                f"kernel_shape={kernel} strides={strides} dilations={dilations}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if dilations != [1, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LpPool currently supports dilations=[1,1] only. dilations={dilations}",
            node_name=node.name,
            node_op=node.op,
        )
    p = float(node.attrs.get("p", 2.0))
    if not np.isfinite(p) or p <= 0.0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LpPool p must be finite and > 0. p={p}",
            node_name=node.name,
            node_op=node.op,
        )
    count_include_pad = int(node.attrs.get("count_include_pad", 0))
    raw_pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
    zero_pad = all(int(v) == 0 for v in raw_pads)
    if count_include_pad not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LpPool count_include_pad must be 0 or 1. count_include_pad={count_include_pad}",
            node_name=node.name,
            node_op=node.op,
        )
    if count_include_pad != 1 and not zero_pad:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "LpPool builtin decomposition requires count_include_pad=1 when pads are non-zero. "
                f"count_include_pad={count_include_pad} pads={raw_pads}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_fc(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if node.op == "Gemm":
        if input_rank != 2:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=f"Gemm input rank must be 2. rank={input_rank}",
                node_name=node.name,
                node_op=node.op,
            )
        weight_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        if len(weight_shape) != 2:
            raise NodeValidationError(
                reason_code="unsupported_weight_rank",
                message=(
                    "Gemm weight rank must be 2. "
                    f"weight_shape={weight_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        trans_a = int(node.attrs.get("transA", 0))
        trans_b = int(node.attrs.get("transB", 0))
        if trans_a not in [0, 1] or trans_b not in [0, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Gemm transA/transB must be 0 or 1 in builtin lowering. "
                    f"transA={trans_a} transB={trans_b}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return
    else:
        if input_rank < 2:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=f"{node.op} input rank must be >= 2. rank={input_rank}",
                node_name=node.name,
                node_op=node.op,
            )
    weights = _require_const_input(node, ctx, 1, "fully_connected weights")
    if weights.ndim != 2:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"FullyConnected weight rank must be 2. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_matmul(node: Any, ctx: Any) -> None:
    a_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    b_rank = len(ctx.get_tensor_shape(node.inputs[1].name))
    is_standard_matmul = a_rank >= 2 and b_rank >= 2
    is_vector_rhs_matmul = a_rank >= 2 and b_rank == 1
    is_vector_lhs_matmul = a_rank == 1 and b_rank >= 2
    is_vector_dot = a_rank == 1 and b_rank == 1
    is_scalar_multiply = a_rank == 0 or b_rank == 0
    if not (
        is_standard_matmul
        or is_vector_rhs_matmul
        or is_vector_lhs_matmul
        or is_vector_dot
        or is_scalar_multiply
    ):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MatMul input ranks must be (a_rank>=2,b_rank>=2) "
                "or vector-rhs form (a_rank>=2,b_rank=1) "
                "or vector-lhs form (a_rank=1,b_rank>=2) "
                "or vector-dot form (a_rank=1,b_rank=1) "
                "or scalar multiply form (a_rank=0 or b_rank=0). "
                f"a_rank={a_rank} b_rank={b_rank}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_multi_head_attention(node: Any, ctx: Any) -> None:
    num_heads = int(node.attrs.get("num_heads", 0))
    if num_heads <= 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"MultiHeadAttention num_heads must be > 0. num_heads={num_heads}",
            node_name=node.name,
            node_op=node.op,
        )

    unidirectional = int(node.attrs.get("unidirectional", 0))
    if unidirectional != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "MultiHeadAttention builtin lowering currently supports unidirectional=0 only. "
                f"unidirectional={unidirectional}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    query_name = node.inputs[0].name
    key_name = node.inputs[1].name
    value_name = node.inputs[2].name
    query_shape = [int(v) for v in ctx.get_tensor_shape(query_name)]
    key_shape = [int(v) for v in ctx.get_tensor_shape(key_name)]
    value_shape = [int(v) for v in ctx.get_tensor_shape(value_name)]
    if len(query_shape) != 3 or len(key_shape) != 3 or len(value_shape) != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MultiHeadAttention builtin lowering currently supports rank-3 query/key/value only. "
                f"query_shape={query_shape} key_shape={key_shape} value_shape={value_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    query_dtype = str(ctx.get_tensor_dtype(query_name)).upper()
    key_dtype = str(ctx.get_tensor_dtype(key_name)).upper()
    value_dtype = str(ctx.get_tensor_dtype(value_name)).upper()
    if len({query_dtype, key_dtype, value_dtype}) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "MultiHeadAttention builtin lowering requires query/key/value dtypes to match. "
                f"query_dtype={query_dtype} key_dtype={key_dtype} value_dtype={value_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if query_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "MultiHeadAttention builtin lowering supports FLOAT16/FLOAT32 only. "
                f"dtype={query_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    query_hidden = int(query_shape[2])
    key_hidden = int(key_shape[2])
    value_hidden = int(value_shape[2])
    if query_hidden <= 0 or key_hidden <= 0 or value_hidden <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "MultiHeadAttention builtin lowering currently requires static positive hidden sizes. "
                f"query_shape={query_shape} key_shape={key_shape} value_shape={value_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if query_hidden % num_heads != 0 or key_hidden % num_heads != 0 or value_hidden % num_heads != 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "MultiHeadAttention hidden sizes must be divisible by num_heads. "
                f"num_heads={num_heads} query_hidden={query_hidden} "
                f"key_hidden={key_hidden} value_hidden={value_hidden}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(query_hidden // num_heads) != int(key_hidden // num_heads):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "MultiHeadAttention query/key head dimensions must match. "
                f"query_head_dim={int(query_hidden // num_heads)} key_head_dim={int(key_hidden // num_heads)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_attention(node: Any, ctx: Any) -> None:
    original_inputs = [str(v.name) for v in node.inputs]
    unsupported_optional_inputs = [
        str(original_inputs[idx])
        for idx in range(3, len(original_inputs))
        if str(original_inputs[idx]) != ""
    ]
    if len(unsupported_optional_inputs) > 0:
        raise NodeValidationError(
            reason_code="unsupported_input_count",
            message=(
                "Attention builtin lowering currently supports 3-input form only "
                f"(query,key,value). unsupported_optional_inputs={unsupported_optional_inputs}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(node.outputs) != 1:
        raise NodeValidationError(
            reason_code="unsupported_output_count",
            message=f"Attention builtin lowering supports 1 output only. outputs={len(node.outputs)}",
            node_name=node.name,
            node_op=node.op,
        )
    q_num_heads = int(node.attrs.get("q_num_heads", node.attrs.get("num_heads", 0)))
    kv_num_heads = int(node.attrs.get("kv_num_heads", q_num_heads))
    if q_num_heads <= 0 or kv_num_heads <= 0 or q_num_heads != kv_num_heads:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Attention builtin lowering requires q_num_heads and kv_num_heads to be equal positive integers. "
                f"q_num_heads={q_num_heads} kv_num_heads={kv_num_heads}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(node.attrs.get("is_causal", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Attention builtin lowering currently supports is_causal=0 only. is_causal={int(node.attrs.get('is_causal', 0))}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(node.attrs.get("qk_matmul_output_mode", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Attention builtin lowering currently supports qk_matmul_output_mode=0 only. "
                f"qk_matmul_output_mode={int(node.attrs.get('qk_matmul_output_mode', 0))}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    softcap = float(node.attrs.get("softcap", 0.0))
    if not np.isfinite(softcap) or abs(softcap) > 1e-12:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Attention builtin lowering currently supports softcap=0 only. softcap={softcap}",
            node_name=node.name,
            node_op=node.op,
        )

    proxy_node = type("AttentionProxyNode", (), {})()
    proxy_node.name = str(node.name)
    proxy_node.op = "MultiHeadAttention"
    proxy_node.inputs = node.inputs
    proxy_node.outputs = node.outputs
    proxy_node.attrs = dict(node.attrs)
    proxy_node.attrs["num_heads"] = int(q_num_heads)
    proxy_node.attrs["unidirectional"] = 0
    _validate_multi_head_attention(proxy_node, ctx)


def _validate_det(node: Any, ctx: Any) -> None:
    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Det input rank must be >= 2. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "Det input/output dtype must be FLOAT16/FLOAT32. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    rows = int(input_shape[-2])
    cols = int(input_shape[-1])
    if rows != cols or rows not in {2, 3}:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Det builtin lowering currently supports static square 2x2/3x3 matrices only. "
                f"input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_output_rank = int(len(input_shape) - 2)
    if len(output_shape) not in {expected_output_rank, 1 if expected_output_rank == 0 else expected_output_rank}:
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=f"Det output rank must match input rank-2. input_shape={input_shape} output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_fused_matmul(node: Any, ctx: Any) -> None:
    _validate_matmul(node, ctx)

    a_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    b_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    y_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    supported_dtypes = {"FLOAT16", "FLOAT32"}
    if a_dtype not in supported_dtypes or b_dtype not in supported_dtypes or y_dtype not in supported_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_dtype",
            message=(
                "FusedMatMul currently supports FLOAT16/FLOAT32 tensors only in flatbuffer_direct. "
                f"a_dtype={a_dtype} b_dtype={b_dtype} y_dtype={y_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a not in [0, 1] or trans_b not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "FusedMatMul transA/transB must be 0 or 1. "
                f"transA={trans_a} transB={trans_b}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    alpha = float(node.attrs.get("alpha", 1.0))
    if not np.isfinite(alpha):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"FusedMatMul alpha must be finite. alpha={alpha}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_shape(node: Any, ctx: Any) -> None:
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"Shape output dtype must be INT32 or INT64. got={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    try:
        start = int(node.attrs.get("start", 0))
        end = int(node.attrs.get("end", input_rank))
    except Exception as ex:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Shape start/end attributes must be integer values when provided. "
                f"start={node.attrs.get('start', None)} end={node.attrs.get('end', None)}"
            ),
            node_name=node.name,
            node_op=node.op,
        ) from ex
    if not np.isfinite(float(start)) or not np.isfinite(float(end)):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Shape start/end must be finite integers. start={start} end={end}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_constant_of_shape(node: Any, ctx: Any) -> None:
    shape_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    supported_shape_dtypes = {
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }
    if shape_dtype not in supported_shape_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ConstantOfShape input shape tensor must be integer dtype. "
                f"shape_dtype={shape_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    shape_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if shape_rank != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ConstantOfShape shape input rank must be 1. got={shape_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    supported_output_dtypes = {
        "BOOL",
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
        "FLOAT16",
        "FLOAT32",
        "FLOAT64",
    }
    if output_dtype not in supported_output_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "ConstantOfShape output dtype is not supported in flatbuffer_direct builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    value_attr = node.attrs.get("value", None)
    if value_attr is None:
        return
    if hasattr(value_attr, "values"):
        value_arr = np.asarray(getattr(value_attr, "values"))
    else:
        value_arr = np.asarray(value_attr)
    if int(value_arr.size) > 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "ConstantOfShape value attribute must be scalar or single-element tensor. "
                f"value_shape={list(value_arr.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_matmul_integer(node: Any, ctx: Any) -> None:
    a_shape = ctx.get_tensor_shape(node.inputs[0].name)
    b_shape = ctx.get_tensor_shape(node.inputs[1].name)
    a_rank = len(a_shape)
    b_rank = len(b_shape)
    # Unknown-rank placeholders can appear as rank=1 in partially inferred graphs.
    if a_rank < 2 and a_rank != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MatMulInteger requires A rank >= 2 (or rank=1 unknown placeholder) "
                f"in flatbuffer_direct. a_shape={a_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if b_rank < 2 and b_rank != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MatMulInteger requires B rank >= 2 (or rank=1 unknown placeholder) "
                f"in flatbuffer_direct. b_shape={b_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    supported_input_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32"}
    a_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    b_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if a_dtype not in supported_input_dtypes or b_dtype not in supported_input_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "MatMulInteger input dtypes must be integer tensor types. "
                f"a_dtype={a_dtype} b_dtype={b_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"MatMulInteger output dtype must be INT32 or INT64. got={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    a_row_dim = int(a_shape[-2]) if len(a_shape) >= 2 else -1
    b_col_dim = int(b_shape[-1]) if len(b_shape) >= 2 else -1
    for idx, label, expected_dim in [
        (2, "a_zero_point", a_row_dim),
        (3, "b_zero_point", b_col_dim),
    ]:
        if idx >= len(node.inputs):
            continue
        zp_shape = ctx.get_tensor_shape(node.inputs[idx].name)
        if len(zp_shape) > 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    f"MatMulInteger {label} must be scalar or 1D tensor. "
                    f"shape={zp_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if len(zp_shape) == 1 and int(zp_shape[0]) > 1 and int(expected_dim) > 1 and int(zp_shape[0]) != int(expected_dim):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    f"MatMulInteger {label} length mismatch. "
                    f"shape={zp_shape} expected={expected_dim}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if label == "a_zero_point" and len(zp_shape) == 1 and int(zp_shape[0]) > 1 and len(a_shape) != 2:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "MatMulInteger vector a_zero_point currently supports rank-2 A only. "
                    f"a_shape={a_shape} a_zero_shape={zp_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_reciprocal(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    supported = {"FLOAT16", "FLOAT32"}
    if input_dtype not in supported or output_dtype not in supported:
        raise NodeValidationError(
            reason_code="unsupported_dtype",
            message=(
                "Reciprocal currently supports FLOAT16/FLOAT32 input and output in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_inverse(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    supported = {"FLOAT16", "FLOAT32"}
    if input_dtype not in supported or output_dtype not in supported:
        raise NodeValidationError(
            reason_code="unsupported_dtype",
            message=(
                "Inverse currently supports FLOAT16/FLOAT32 input and output in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_name = node.inputs[0].name
    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    if len(input_shape) < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "Inverse requires input rank >= 2 in flatbuffer_direct. "
                f"input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    raw_shape = None
    if hasattr(ctx, "shape_map") and isinstance(ctx.shape_map, dict):
        raw_shape = ctx.shape_map.get(input_name, None)
    if raw_shape is not None:
        try:
            raw_shape = [int(v) for v in list(raw_shape)]
        except Exception:
            raw_shape = None
    if raw_shape is None or len(raw_shape) < 2:
        raw_shape = [int(v) for v in input_shape]

    row_dim = int(raw_shape[-2])
    col_dim = int(raw_shape[-1])
    row_known = row_dim > 0
    col_known = col_dim > 0

    if row_known and not col_known:
        col_dim = int(row_dim)
        col_known = True
    elif col_known and not row_known:
        row_dim = int(col_dim)
        row_known = True

    if not row_known or not col_known:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Inverse requires resolvable matrix last dimensions in flatbuffer_direct. "
                f"input_shape={input_shape} raw_input_shape={raw_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if row_dim != col_dim:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Inverse requires square matrix last dimensions in flatbuffer_direct. "
                f"input_shape={input_shape} raw_input_shape={raw_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if row_dim < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Inverse builtin lowering requires matrix dims >= 1 in flatbuffer_direct. "
                f"input_shape={input_shape} raw_input_shape={raw_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if row_dim > 16:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Inverse builtin lowering currently supports square matrix dims up to 16x16 "
                f"in flatbuffer_direct. input_shape={input_shape} raw_input_shape={raw_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_pow(node: Any, ctx: Any) -> None:
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "Pow currently supports FLOAT16/FLOAT32 output in flatbuffer_direct. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_onehot(node: Any, ctx: Any) -> None:
    depth = _require_const_input(node, ctx, 1, "OneHot depth")
    depth_arr = np.asarray(depth).reshape(-1)
    if int(depth_arr.size) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"OneHot depth must be scalar. depth_shape={list(np.asarray(depth).shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    depth_value = int(depth_arr[0])
    if depth_value <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message=f"OneHot depth must be > 0. depth={depth_value}",
            node_name=node.name,
            node_op=node.op,
        )

    values = _require_const_input(node, ctx, 2, "OneHot values")
    values_arr = np.asarray(values).reshape(-1)
    if int(values_arr.size) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "OneHot values must contain exactly two elements [off_value, on_value]. "
                f"values_shape={list(np.asarray(values).shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axis = int(node.attrs.get("axis", -1))
    if axis < 0:
        axis += int(input_rank + 1)
    if axis < 0 or axis > int(input_rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"OneHot axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )



def _extract_slice_indices(
    *,
    node: Any,
    ctx: Any,
    input_index: int,
    attr_name: str,
    label: str,
) -> List[int]:
    values: Optional[List[int]] = None
    if len(node.inputs) > input_index and str(node.inputs[input_index].name) != "":
        arr = _require_const_input(node, ctx, input_index, label)
        values = [int(v) for v in np.asarray(arr).reshape(-1).tolist()]
    elif attr_name in node.attrs:
        attr_val = node.attrs.get(attr_name)
        if isinstance(attr_val, (list, tuple, np.ndarray)):
            values = [int(v) for v in np.asarray(attr_val).reshape(-1).tolist()]
        elif attr_val is None:
            values = []
        else:
            values = [int(attr_val)]
    if values is None:
        raise NodeValidationError(
            reason_code="missing_required_attribute",
            message=(
                f"{label} must be provided as constant input[{input_index}] "
                f"or attribute '{attr_name}'."
            ),
            node_name=node.name,
            node_op=node.op,
        )
    return [int(v) for v in values]
































def _validate_cast(node: Any, _ctx: Any) -> None:
    to_value = node.attrs.get("to", None)
    if to_value is None:
        return
    try:
        _ = int(to_value)
    except Exception as ex:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Cast 'to' attribute must be integer enum. got={to_value}",
            node_name=node.name,
            node_op=node.op,
        ) from ex


def _validate_expand(node: Any, _ctx: Any) -> None:
    # Expand is lowered via multiply-by-ones.
    # Dynamic shape-input cases build ones via FILL at runtime.
    return


def _validate_tile(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    multiples_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    if len(multiples_shape) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "Tile multiples input must be rank-1 in flatbuffer_direct. "
                f"multiples_shape={multiples_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if (
        len(input_shape) > 0
        and input_shape != [1]
        and len(multiples_shape) == 1
        and int(multiples_shape[0]) > 0
        and int(multiples_shape[0]) != len(input_shape)
    ):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Tile multiples length must match input rank when statically known. "
                f"input_shape={input_shape} multiples_shape={multiples_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    multiples_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if not _is_integer_dtype(multiples_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "Tile multiples dtype must be integer in flatbuffer_direct. "
                f"multiples_dtype={multiples_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    multiples_arr = ctx.get_constant_array(node.inputs[1].name)
    if multiples_arr is not None:
        if np.any(np.asarray(multiples_arr).reshape(-1) < 0):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message="Tile multiples must be non-negative.",
                node_name=node.name,
                node_op=node.op,
            )


def _is_numeric_dtype(dtype: str) -> bool:
    dt = str(dtype).upper()
    return _is_integer_dtype(dt) or dt in {"FLOAT16", "FLOAT32"}


def _normalize_scatter_nd_work_dtype(dtype: str) -> str:
    dt = str(dtype).upper()
    if dt == "INT64":
        return "INT32"
    if dt == "UINT64":
        return "UINT32"
    return dt


def _validate_scatter_nd(node: Any, ctx: Any) -> None:
    reduction = str(node.attrs.get("reduction", "none")).lower()
    if reduction != "none":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "ScatterND reduction attribute supports 'none' only in flatbuffer_direct. "
                f"reduction={reduction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    data_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    indices_shape = _tensor_shape_with_signature(ctx, node.inputs[1].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(data_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ScatterND data rank must be >= 1. data_shape={data_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(indices_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ScatterND indices rank must be >= 1. indices_shape={indices_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if (
        len(output_shape) == len(data_shape)
        and output_shape != [1]
        and data_shape != [1]
    ):
        for out_dim, data_dim in zip(output_shape, data_shape):
            if int(out_dim) > 0 and int(data_dim) > 0 and int(out_dim) != int(data_dim):
                raise NodeValidationError(
                    reason_code="invalid_output_shape",
                    message=(
                        "ScatterND output shape must match data shape. "
                        f"data_shape={data_shape} output_shape={output_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

    data_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    updates_dtype = str(ctx.get_tensor_dtype(node.inputs[2].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if not _is_numeric_dtype(data_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterND data dtype must be numeric (int/float) in flatbuffer_direct. "
                f"data_dtype={data_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_numeric_dtype(updates_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterND updates dtype must be numeric (int/float) in flatbuffer_direct. "
                f"updates_dtype={updates_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if (
        output_dtype != data_dtype
        and _normalize_scatter_nd_work_dtype(output_dtype) != str(data_dtype).upper()
    ):
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "ScatterND output dtype must match the ScatterND work dtype in flatbuffer_direct. "
                f"data_dtype={data_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    indices_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if not _is_integer_dtype(indices_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterND indices dtype must be integer in flatbuffer_direct. "
                f"indices_dtype={indices_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    k_dim = int(indices_shape[-1]) if len(indices_shape) > 0 else -1
    if k_dim <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "ScatterND requires static positive indices last dimension in flatbuffer_direct. "
                f"indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(k_dim) > int(len(data_shape)):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "ScatterND indices last dimension must be <= data rank. "
                f"indices_last_dim={k_dim} data_rank={len(data_shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_unique(node: Any, ctx: Any) -> None:
    def _is_output_consumed_or_exposed(tensor_name: str) -> bool:
        if str(tensor_name) == "":
            return False
        consumer_count = 0
        if hasattr(ctx, "tensor_consumer_count") and isinstance(ctx.tensor_consumer_count, dict):
            consumer_count = int(ctx.tensor_consumer_count.get(str(tensor_name), 0))
        if int(consumer_count) > 0:
            return True
        graph_outputs = getattr(ctx, "graph_output_names", set())
        if isinstance(graph_outputs, set):
            return str(tensor_name) in graph_outputs
        if isinstance(graph_outputs, list):
            return str(tensor_name) in set([str(v) for v in graph_outputs])
        return False

    input_name = node.inputs[0].name
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()

    if not _is_integer_dtype(input_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "Unique lowering currently supports integer input dtype only in flatbuffer_direct. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_integer_dtype(output_dtype):
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "Unique output[0] dtype must be integer in flatbuffer_direct builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    sorted_attr = int(node.attrs.get("sorted", 1))
    if int(sorted_attr) not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Unique sorted attribute must be 0 or 1. sorted={sorted_attr}",
            node_name=node.name,
            node_op=node.op,
        )

    axis_attr = node.attrs.get("axis", None)
    if axis_attr is not None:
        rank = int(len(input_shape))
        axis = int(axis_attr)
        if axis < 0:
            axis += rank
        if axis < 0 or axis >= rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"Unique axis out of range. axis={axis_attr} rank={rank}",
                node_name=node.name,
                node_op=node.op,
            )
        if int(axis) != 0:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"Unique builtin lowering supports axis=0 only when axis is specified. axis={axis}",
                node_name=node.name,
                node_op=node.op,
            )
        if int(rank) != 2:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Unique axis=0 builtin lowering requires rank-2 input. "
                    f"input_shape={input_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if int(input_shape[1]) <= 0:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Unique axis=0 builtin lowering requires static positive second dimension. "
                    f"input_shape={input_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    for output_index in range(1, len(node.outputs)):
        output_name = str(node.outputs[output_index].name)
        if _is_output_consumed_or_exposed(output_name):
            raise NodeValidationError(
                reason_code="unsupported_output_count",
                message=(
                    "Unique builtin lowering currently supports output[0] only; "
                    f"output[{output_index}] is consumed or graph-exposed. output_name={output_name}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_scatter_elements(node: Any, ctx: Any) -> None:
    reduction = str(node.attrs.get("reduction", "none")).lower()
    if reduction not in {"none", "add"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "ScatterElements reduction attribute supports 'none' and 'add' only in flatbuffer_direct. "
                f"reduction={reduction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    data_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    indices_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    updates_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[2].name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
    if len(data_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ScatterElements data rank must be >= 1. data_shape={data_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(indices_shape) != len(data_shape):
        if len(indices_shape) < len(data_shape):
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "ScatterElements requires indices rank >= data rank in flatbuffer_direct. "
                    f"data_shape={data_shape} indices_shape={indices_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    if len(updates_shape) != len(data_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "ScatterElements requires updates rank equal to data rank in flatbuffer_direct. "
                f"data_shape={data_shape} updates_shape={updates_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(indices_shape) > len(updates_shape):
        if not all(int(v) > 0 for v in indices_shape) or not all(int(v) > 0 for v in updates_shape):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "ScatterElements with indices rank > updates rank requires static positive "
                    "indices/updates dimensions in flatbuffer_direct. "
                    f"indices_shape={indices_shape} updates_shape={updates_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        prefix_shape = [int(v) for v in indices_shape[: len(indices_shape) - len(updates_shape)]]
        if not all(int(v) > 0 for v in prefix_shape):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "ScatterElements with indices rank > updates rank requires static positive "
                    "leading indices dimensions in flatbuffer_direct. "
                    f"indices_shape={indices_shape} updates_shape={updates_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        trailing_indices_shape = indices_shape[-len(updates_shape):]
        for idx_dim, upd_dim in zip(trailing_indices_shape, updates_shape):
            if int(idx_dim) > 0 and int(upd_dim) > 0 and int(idx_dim) != int(upd_dim):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "ScatterElements with indices rank > updates rank requires trailing "
                        "indices dimensions to match updates dimensions in flatbuffer_direct. "
                        f"indices_shape={indices_shape} updates_shape={updates_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
    if len(output_shape) == len(data_shape) and output_shape != [1] and data_shape != [1]:
        for out_dim, data_dim in zip(output_shape, data_shape):
            if int(out_dim) > 0 and int(data_dim) > 0 and int(out_dim) != int(data_dim):
                raise NodeValidationError(
                    reason_code="invalid_output_shape",
                    message=(
                        "ScatterElements output shape must match data shape. "
                        f"data_shape={data_shape} output_shape={output_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += len(data_shape)
    if axis < 0 or axis >= len(data_shape):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "ScatterElements axis is out of range in flatbuffer_direct. "
                f"axis={axis} rank={len(data_shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    data_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    updates_dtype = str(ctx.get_tensor_dtype(node.inputs[2].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    indices_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if not _is_numeric_dtype(data_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterElements data dtype must be numeric (int/float) in flatbuffer_direct. "
                f"data_dtype={data_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_numeric_dtype(updates_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterElements updates dtype must be numeric (int/float) in flatbuffer_direct. "
                f"updates_dtype={updates_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype != data_dtype:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "ScatterElements output dtype must match data dtype in flatbuffer_direct. "
                f"data_dtype={data_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_integer_dtype(indices_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterElements indices dtype must be integer in flatbuffer_direct. "
                f"indices_dtype={indices_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_tensor_scatter(node: Any, ctx: Any) -> None:
    data_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    updates_shape = _tensor_shape_with_signature(ctx, node.inputs[1].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    rank = int(len(data_shape))
    if int(rank) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"TensorScatter data rank must be >= 1. data_shape={data_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(updates_shape) != int(rank):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "TensorScatter updates rank must match data rank in flatbuffer_direct. "
                f"data_shape={data_shape} updates_shape={updates_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(output_shape) != int(rank):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "TensorScatter output rank must match data rank in flatbuffer_direct. "
                f"data_shape={data_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not all(int(v) > 0 for v in updates_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "TensorScatter builtin lowering requires static positive updates shape. "
                f"updates_shape={updates_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    for out_dim, data_dim in zip(output_shape, data_shape):
        if int(out_dim) > 0 and int(data_dim) > 0 and int(out_dim) != int(data_dim):
            raise NodeValidationError(
                reason_code="invalid_output_shape",
                message=(
                    "TensorScatter output shape must match data shape in flatbuffer_direct. "
                    f"data_shape={data_shape} output_shape={output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    axis = int(node.attrs.get("axis", -2))
    if axis < 0:
        axis += int(rank)
    if axis < 0 or axis >= int(rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TensorScatter axis is out of range. axis={node.attrs.get('axis', -2)} rank={rank}",
            node_name=node.name,
            node_op=node.op,
        )

    mode = str(node.attrs.get("mode", "linear")).lower()
    if mode not in {"linear", "circular"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TensorScatter mode must be 'linear' or 'circular'. mode={mode}",
            node_name=node.name,
            node_op=node.op,
        )
    if mode == "circular" and int(data_shape[axis]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "TensorScatter circular mode requires static positive axis dimension in flatbuffer_direct. "
                f"data_shape={data_shape} axis={axis}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    data_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    updates_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if not _is_numeric_dtype(data_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"TensorScatter data dtype must be numeric. data_dtype={data_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_numeric_dtype(updates_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"TensorScatter updates dtype must be numeric. updates_dtype={updates_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if str(output_dtype) != str(data_dtype):
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "TensorScatter output dtype must match data dtype in flatbuffer_direct. "
                f"data_dtype={data_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) > 2 and str(node.inputs[2].name) != "":
        write_indices_shape = _tensor_shape_with_signature(ctx, node.inputs[2].name)
        write_indices_dtype = str(ctx.get_tensor_dtype(node.inputs[2].name)).upper()
        if len(write_indices_shape) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "TensorScatter write_indices must be rank-1 in flatbuffer_direct. "
                    f"write_indices_shape={write_indices_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if int(write_indices_shape[0]) > 0 and int(updates_shape[0]) > 0 and int(write_indices_shape[0]) < int(updates_shape[0]):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "TensorScatter write_indices length must be >= updates batch dimension in flatbuffer_direct. "
                    f"write_indices_shape={write_indices_shape} updates_shape={updates_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if not _is_integer_dtype(write_indices_dtype):
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=(
                    "TensorScatter write_indices dtype must be integer in flatbuffer_direct. "
                    f"write_indices_dtype={write_indices_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if int(data_shape[axis]) > 0 and int(updates_shape[axis]) > 0:
        if len(node.inputs) <= 2 or str(node.inputs[2].name) == "":
            if int(updates_shape[axis]) > int(data_shape[axis]):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "TensorScatter without write_indices requires updates axis dim <= data axis dim "
                        "in flatbuffer_direct. "
                        f"data_shape={data_shape} updates_shape={updates_shape} axis={axis}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        elif mode == "linear":
            write_indices_const = ctx.get_constant_array(node.inputs[2].name)
            if write_indices_const is not None:
                write_indices_arr = np.asarray(write_indices_const, dtype=np.int64).reshape(-1)
                if write_indices_arr.size > 0:
                    if int(np.min(write_indices_arr)) < 0:
                        raise NodeValidationError(
                            reason_code="unsupported_input_value",
                            message=(
                                "TensorScatter linear mode requires non-negative constant write_indices in "
                                "flatbuffer_direct. "
                                f"write_indices={write_indices_arr.tolist()}"
                            ),
                            node_name=node.name,
                            node_op=node.op,
                        )
                    if int(np.max(write_indices_arr)) + int(updates_shape[axis]) > int(data_shape[axis]):
                        raise NodeValidationError(
                            reason_code="unsupported_input_value",
                            message=(
                                "TensorScatter linear mode constant write_indices exceed data axis range in "
                                "flatbuffer_direct. "
                                f"data_shape={data_shape} updates_shape={updates_shape} axis={axis} "
                                f"write_indices={write_indices_arr.tolist()}"
                            ),
                            node_name=node.name,
                            node_op=node.op,
                        )


def _validate_mel_weight_matrix(node: Any, ctx: Any) -> None:
    scalar_values: list[float] = []
    for input_tensor in node.inputs[:5]:
        scalar_arr = ctx.get_constant_array(input_tensor.name)
        if scalar_arr is None:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message="MelWeightMatrix requires all five inputs to be constant scalars in flatbuffer_direct.",
                node_name=node.name,
                node_op=node.op,
            )
        arr = np.asarray(scalar_arr)
        if int(arr.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "MelWeightMatrix requires scalar-like inputs (shape [] or [1]) in flatbuffer_direct. "
                    f"input={input_tensor.name} shape={list(arr.shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        scalar_values.append(float(arr.reshape(-1)[0]))

    num_mel_bins = int(scalar_values[0])
    dft_length = int(scalar_values[1])
    sample_rate = float(scalar_values[2])
    lower_edge_hertz = float(scalar_values[3])
    upper_edge_hertz = float(scalar_values[4])
    if int(num_mel_bins) <= 0 or int(dft_length) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message=(
                "MelWeightMatrix requires positive num_mel_bins and dft_length in flatbuffer_direct. "
                f"num_mel_bins={num_mel_bins} dft_length={dft_length}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not (float(sample_rate) > 0.0):
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message=f"MelWeightMatrix sample_rate must be positive. sample_rate={sample_rate}",
            node_name=node.name,
            node_op=node.op,
        )
    if not (0.0 <= float(lower_edge_hertz) < float(upper_edge_hertz) <= float(sample_rate) / 2.0):
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message=(
                "MelWeightMatrix requires 0 <= lower_edge_hertz < upper_edge_hertz <= sample_rate/2 "
                "in flatbuffer_direct. "
                f"lower_edge_hertz={lower_edge_hertz} upper_edge_hertz={upper_edge_hertz} sample_rate={sample_rate}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "MelWeightMatrix output dtype must be FLOAT16 or FLOAT32 in flatbuffer_direct. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    expected_shape = [int(dft_length // 2 + 1), int(num_mel_bins)]
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(output_shape) != 2:
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "MelWeightMatrix output must be rank-2 in flatbuffer_direct. "
                f"output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    for out_dim, exp_dim in zip(output_shape, expected_shape):
        if int(out_dim) > 0 and int(out_dim) != int(exp_dim):
            raise NodeValidationError(
                reason_code="invalid_output_shape",
                message=(
                    "MelWeightMatrix output shape mismatch in flatbuffer_direct. "
                    f"output_shape={output_shape} expected={expected_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_loss_common(node: Any, ctx: Any, *, allow_second_output: bool) -> None:
    input_name = node.inputs[0].name
    target_name = node.inputs[1].name
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    target_shape = _tensor_shape_with_signature(ctx, target_name)
    input_rank = int(len(input_shape))
    if int(input_rank) < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"{node.op} input rank must be >= 2 in flatbuffer_direct. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    class_dim = int(input_shape[1])
    if int(class_dim) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"{node.op} requires static positive class dimension at axis 1. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    expected_target_shape = [int(v) for idx, v in enumerate(input_shape) if int(idx) != 1]
    if len(target_shape) != len(expected_target_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                f"{node.op} target rank must equal input rank - 1 in flatbuffer_direct. "
                f"input_shape={input_shape} target_shape={target_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    for tgt_dim, exp_dim in zip(target_shape, expected_target_shape):
        if int(tgt_dim) > 0 and int(exp_dim) > 0 and int(tgt_dim) != int(exp_dim):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    f"{node.op} target shape mismatch in flatbuffer_direct. "
                    f"input_shape={input_shape} target_shape={target_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    target_dtype = str(ctx.get_tensor_dtype(target_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"{node.op} input dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. input_dtype={input_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"{node.op} output dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_integer_dtype(target_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"{node.op} target dtype must be integer in flatbuffer_direct. target_dtype={target_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) > 2 and str(node.inputs[2].name) != "":
        weight_shape = _tensor_shape_with_signature(ctx, node.inputs[2].name)
        weight_dtype = str(ctx.get_tensor_dtype(node.inputs[2].name)).upper()
        if len(weight_shape) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=f"{node.op} weight input must be rank-1 in flatbuffer_direct. weight_shape={weight_shape}",
                node_name=node.name,
                node_op=node.op,
            )
        if int(weight_shape[0]) > 0 and int(weight_shape[0]) != int(class_dim):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    f"{node.op} weight length must match class dimension in flatbuffer_direct. "
                    f"weight_shape={weight_shape} input_shape={input_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if weight_dtype not in {"FLOAT16", "FLOAT32"}:
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=f"{node.op} weight dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. weight_dtype={weight_dtype}",
                node_name=node.name,
                node_op=node.op,
            )

    reduction = str(node.attrs.get("reduction", "mean")).lower()
    if reduction not in {"none", "sum", "mean"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"{node.op} reduction must be one of none/sum/mean. reduction={reduction}",
            node_name=node.name,
            node_op=node.op,
        )

    output0_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if reduction == "none":
        if len(output0_shape) != len(target_shape):
            raise NodeValidationError(
                reason_code="invalid_output_shape",
                message=(
                    f"{node.op} output[0] rank must match target rank when reduction=none. "
                    f"output_shape={output0_shape} target_shape={target_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    elif len(output0_shape) not in {0, 1} or (len(output0_shape) == 1 and int(output0_shape[0]) not in {-1, 1}):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                f"{node.op} output[0] must be scalar-like when reduction={reduction}. "
                f"output_shape={output0_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if not allow_second_output and len(node.outputs) > 1:
        for output_index in range(1, len(node.outputs)):
            output_name = str(node.outputs[output_index].name)
            if output_name != "":
                raise NodeValidationError(
                    reason_code="unsupported_output_count",
                    message=f"{node.op} builtin lowering supports output[0] only in flatbuffer_direct.",
                    node_name=node.name,
                    node_op=node.op,
                )

    if allow_second_output and len(node.outputs) > 1 and str(node.outputs[1].name) != "":
        output1_shape = _tensor_shape_with_signature(ctx, node.outputs[1].name)
        output1_dtype = str(ctx.get_tensor_dtype(node.outputs[1].name)).upper()
        if len(output1_shape) != len(input_shape):
            raise NodeValidationError(
                reason_code="invalid_output_shape",
                message=(
                    f"{node.op} output[1] rank must match input rank in flatbuffer_direct. "
                    f"input_shape={input_shape} output1_shape={output1_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if output1_dtype not in {"FLOAT16", "FLOAT32"}:
            raise NodeValidationError(
                reason_code="unsupported_output_dtype",
                message=f"{node.op} output[1] dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. output1_dtype={output1_dtype}",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_negative_log_likelihood_loss(node: Any, ctx: Any) -> None:
    _validate_loss_common(node, ctx, allow_second_output=False)


def _validate_softmax_cross_entropy_loss(node: Any, ctx: Any) -> None:
    _validate_loss_common(node, ctx, allow_second_output=True)


def _validate_mod(node: Any, _ctx: Any) -> None:
    fmod = int(node.attrs.get("fmod", 0))
    if fmod != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Mod with fmod=1 is not supported by FLOOR_MOD lowering. fmod={fmod}",
            node_name=node.name,
            node_op=node.op,
        )




def _validate_float_unary(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "This op currently supports FLOAT16/FLOAT32 input/output in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_float_to_bool_unary(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype != "BOOL":
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "This op currently supports FLOAT16/FLOAT32 input and BOOL output in "
                "flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_mean(node: Any, ctx: Any) -> None:
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"Mean output must be FLOAT16/FLOAT32. output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, input_obj in enumerate(node.inputs):
        input_dtype = str(ctx.get_tensor_dtype(input_obj.name)).upper()
        if input_dtype not in {"FLOAT16", "FLOAT32"}:
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=(
                    "Mean builtin lowering currently supports FLOAT16/FLOAT32 inputs only. "
                    f"input_index={idx} input_dtype={input_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_float_reduce(node: Any, ctx: Any) -> None:
    _validate_reduce(node, ctx)
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "This reduce op currently supports FLOAT16/FLOAT32 input/output in "
                "flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_where(node: Any, ctx: Any) -> None:
    condition_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if condition_dtype == "BOOL":
        return
    if _is_integer_dtype(condition_dtype) or condition_dtype in {"FLOAT16", "FLOAT32"}:
        return
    raise NodeValidationError(
        reason_code="unsupported_input_dtype",
        message=(
            "Where condition dtype must be BOOL or numeric in flatbuffer_direct. "
            f"condition_dtype={condition_dtype}"
        ),
        node_name=node.name,
        node_op=node.op,
    )


def _validate_random_normal_like(node: Any, ctx: Any) -> None:
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype in {
        "FLOAT16",
        "FLOAT32",
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }:
        return
    raise NodeValidationError(
        reason_code="unsupported_output_type",
        message=(
            "RandomNormalLike output dtype is not supported in flatbuffer_direct. "
            f"dtype={output_dtype}"
        ),
        node_name=node.name,
        node_op=node.op,
    )


def _validate_random_float_output(node: Any, ctx: Any) -> None:
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_type",
            message=(
                "Random builtin lowering currently supports FLOAT16/FLOAT32 output only. "
                f"dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_random_normal(node: Any, ctx: Any) -> None:
    _validate_random_float_output(node, ctx)
    shape_attr = node.attrs.get("shape", None)
    if not isinstance(shape_attr, (list, tuple, np.ndarray)) or len(list(shape_attr)) == 0:
        raise NodeValidationError(
            reason_code="missing_required_attribute",
            message="RandomNormal requires non-empty `shape` attribute in flatbuffer_direct.",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_random_uniform(node: Any, ctx: Any) -> None:
    _validate_random_normal(node, ctx)


def _validate_random_uniform_like(node: Any, ctx: Any) -> None:
    _validate_random_float_output(node, ctx)


def _validate_bernoulli(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "Bernoulli builtin lowering currently supports FLOAT16/FLOAT32 inputs only. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in {
        "BOOL",
        "FLOAT16",
        "FLOAT32",
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }:
        raise NodeValidationError(
            reason_code="unsupported_output_type",
            message=f"Bernoulli output dtype is not supported in flatbuffer_direct. dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_window_op(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if len(input_shape) != 1 or int(input_shape[0]) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Window ops require scalar-like rank-1 length-1 input in flatbuffer_direct. "
                f"input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_reverse_sequence(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    seq_lengths_name = node.inputs[1].name
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    seq_shape = [int(v) for v in ctx.get_tensor_shape(seq_lengths_name)]
    input_rank = int(len(input_shape))
    if input_rank < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ReverseSequence input rank must be >= 2. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(seq_shape) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"ReverseSequence seq_lengths must be rank-1. seq_shape={seq_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    seq_dtype = str(ctx.get_tensor_dtype(seq_lengths_name)).upper()
    if not _is_integer_dtype(seq_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"ReverseSequence seq_lengths must be integer tensor. dtype={seq_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    batch_axis = int(node.attrs.get("batch_axis", 1))
    time_axis = int(node.attrs.get("time_axis", 0))
    if batch_axis < 0:
        batch_axis += input_rank
    if time_axis < 0:
        time_axis += input_rank
    if batch_axis < 0 or batch_axis >= input_rank or time_axis < 0 or time_axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "ReverseSequence batch_axis/time_axis must be in range. "
                f"batch_axis={batch_axis} time_axis={time_axis} rank={input_rank}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if batch_axis == time_axis:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message="ReverseSequence batch_axis and time_axis must differ.",
            node_name=node.name,
            node_op=node.op,
        )
    batch_dim = int(input_shape[batch_axis])
    if batch_dim > 0 and len(seq_shape) == 1 and int(seq_shape[0]) > 0 and int(seq_shape[0]) != batch_dim:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "ReverseSequence seq_lengths length must match batch dimension when static. "
                f"seq_shape={seq_shape} batch_dim={batch_dim}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_rotary_embedding(node: Any, ctx: Any) -> None:
    if len(node.inputs) > 3 and str(node.inputs[3].name) != "":
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message="RotaryEmbedding builtin path currently does not support position_ids input.",
            node_name=node.name,
            node_op=node.op,
        )
    if bool(node.attrs.get("interleaved", 0)):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message="RotaryEmbedding builtin path currently supports interleaved=0 only.",
            node_name=node.name,
            node_op=node.op,
        )

    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    cos_shape = _tensor_shape_with_signature(ctx, node.inputs[1].name)
    sin_shape = _tensor_shape_with_signature(ctx, node.inputs[2].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "RotaryEmbedding builtin path currently supports rank-4 input/output only. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if input_shape != output_shape:
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "RotaryEmbedding output shape must match input shape in flatbuffer_direct. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(cos_shape) != 2 or len(sin_shape) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "RotaryEmbedding cos/sin inputs must be rank-2 in flatbuffer_direct. "
                f"cos_shape={cos_shape} sin_shape={sin_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in input_shape + cos_shape + sin_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RotaryEmbedding builtin path requires static positive input/cos/sin shapes. "
                f"input_shape={input_shape} cos_shape={cos_shape} sin_shape={sin_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    cos_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    sin_dtype = str(ctx.get_tensor_dtype(node.inputs[2].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    for label, dtype in [
        ("input", input_dtype),
        ("cos", cos_dtype),
        ("sin", sin_dtype),
        ("output", output_dtype),
    ]:
        if dtype not in {"FLOAT16", "FLOAT32"}:
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=f"RotaryEmbedding {label} dtype must be FLOAT16/FLOAT32. dtype={dtype}",
                node_name=node.name,
                node_op=node.op,
            )
    if str(output_dtype) != str(input_dtype):
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "RotaryEmbedding output dtype must match input dtype in flatbuffer_direct. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    seq_len = int(input_shape[2])
    head_size = int(input_shape[3])
    rotary_dim = int(node.attrs.get("rotary_embedding_dim", 0))
    if rotary_dim == 0:
        rotary_dim = int(head_size)
    if rotary_dim <= 0 or rotary_dim > int(head_size) or int(rotary_dim % 2) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RotaryEmbedding rotary_embedding_dim must be positive, even, and <= head_size. "
                f"rotary_embedding_dim={rotary_dim} head_size={head_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_cache_shape = [int(seq_len), int(rotary_dim // 2)]
    if cos_shape != expected_cache_shape or sin_shape != expected_cache_shape:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RotaryEmbedding cos/sin shapes must match [seq_len, rotary_embedding_dim/2] in flatbuffer_direct. "
                f"cos_shape={cos_shape} sin_shape={sin_shape} expected={expected_cache_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_dft(node: Any, ctx: Any) -> None:
    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) < 2 or len(output_shape) < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"DFT input/output rank must be >= 2. input_shape={input_shape} output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in input_shape + output_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DFT builtin path requires static positive input/output shapes. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(input_shape[-1]) != 1 or int(output_shape[-1]) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "DFT builtin path currently supports real input [...,N,1] and complex-pair output [...,K,2] only. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if bool(int(node.attrs.get("inverse", 0))) or not bool(int(node.attrs.get("onesided", 0))):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "DFT builtin path currently supports onesided=1 and inverse=0 only. "
                f"onesided={node.attrs.get('onesided', 0)} inverse={node.attrs.get('inverse', 0)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    axis_value = int(node.attrs.get("axis", -2))
    if len(node.inputs) > 2 and str(node.inputs[2].name) != "":
        axis_const = ctx.get_constant_array(node.inputs[2].name)
        if axis_const is None or int(np.asarray(axis_const).size) != 1:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="DFT axis input must be a constant scalar in flatbuffer_direct.",
                node_name=node.name,
                node_op=node.op,
            )
        axis_value = int(np.asarray(axis_const).reshape(-1)[0])
    if axis_value < 0:
        axis_value += len(input_shape)
    if int(axis_value) != int(len(input_shape) - 2):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "DFT builtin path currently supports transform axis at rank-2 only. "
                f"axis={axis_value} input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    dft_length = int(input_shape[-2])
    if len(node.inputs) > 1 and str(node.inputs[1].name) != "":
        dft_const = ctx.get_constant_array(node.inputs[1].name)
        if dft_const is None or int(np.asarray(dft_const).size) != 1:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="DFT dft_length input must be a constant scalar in flatbuffer_direct.",
                node_name=node.name,
                node_op=node.op,
            )
        dft_length = int(np.asarray(dft_const).reshape(-1)[0])
    expected_output_shape = [int(v) for v in input_shape[:-2]] + [int(dft_length // 2 + 1), 2]
    if output_shape != expected_output_shape:
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "DFT output shape mismatch for constrained builtin path. "
                f"output_shape={output_shape} expected={expected_output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"DFT input/output dtype must be FLOAT16/FLOAT32. input_dtype={input_dtype} output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_stft(node: Any, ctx: Any) -> None:
    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) != 2 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "STFT builtin path currently supports rank-2 input and rank-4 output only. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in input_shape + output_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "STFT builtin path requires static positive input/output shapes. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not bool(int(node.attrs.get("onesided", 1))):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message="STFT builtin path currently supports onesided=1 only.",
            node_name=node.name,
            node_op=node.op,
        )
    frame_step_arr = ctx.get_constant_array(node.inputs[1].name)
    window_arr = ctx.get_constant_array(node.inputs[2].name)
    frame_length_arr = ctx.get_constant_array(node.inputs[3].name)
    if frame_step_arr is None or frame_length_arr is None or window_arr is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="STFT frame_step/window/frame_length inputs must be constant in flatbuffer_direct.",
            node_name=node.name,
            node_op=node.op,
        )
    if int(np.asarray(frame_step_arr).size) != 1 or int(np.asarray(frame_length_arr).size) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message="STFT frame_step and frame_length must be scalar-like in flatbuffer_direct.",
            node_name=node.name,
            node_op=node.op,
        )
    frame_step = int(np.asarray(frame_step_arr).reshape(-1)[0])
    frame_length = int(np.asarray(frame_length_arr).reshape(-1)[0])
    if frame_step <= 0 or frame_length <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message=f"STFT frame_step/frame_length must be positive. frame_step={frame_step} frame_length={frame_length}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(np.asarray(window_arr).size) != int(frame_length):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "STFT window length must equal frame_length in flatbuffer_direct. "
                f"window_shape={list(np.asarray(window_arr).shape)} frame_length={frame_length}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    signal_length = int(input_shape[1])
    if signal_length < frame_length:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "STFT builtin path requires signal_length >= frame_length. "
                f"signal_length={signal_length} frame_length={frame_length}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_output_shape = [
        int(input_shape[0]),
        int((signal_length - frame_length) // frame_step + 1),
        int(frame_length // 2 + 1),
        2,
    ]
    if output_shape != expected_output_shape:
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "STFT output shape mismatch for constrained builtin path. "
                f"output_shape={output_shape} expected={expected_output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"STFT input/output dtype must be FLOAT16/FLOAT32. input_dtype={input_dtype} output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_compress(node: Any, ctx: Any) -> None:
    data_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    condition_shape = _tensor_shape_with_signature(ctx, node.inputs[1].name)
    if len(data_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Compress data rank must be >= 1. data_shape={data_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(condition_shape) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Compress condition must be rank-1. condition_shape={condition_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    condition_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if condition_dtype != "BOOL" and not _is_integer_dtype(condition_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"Compress condition must be BOOL or integer tensor. dtype={condition_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    data_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if data_dtype != output_dtype:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"Compress output dtype must match data dtype. data_dtype={data_dtype} output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    axis_attr = node.attrs.get("axis", None)
    if axis_attr is None:
        if len(output_shape) != 1:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=f"Compress without axis must produce rank-1 output. output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
    else:
        axis = int(axis_attr)
        if axis < -len(data_shape) or axis >= len(data_shape):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"Compress axis out of range. axis={axis} data_shape={data_shape}",
                node_name=node.name,
                node_op=node.op,
            )
        normalized_axis = int(axis if axis >= 0 else axis + len(data_shape))
        if len(output_shape) != len(data_shape):
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=f"Compress with axis must preserve rank. data_shape={data_shape} output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
        axis_dim = int(data_shape[normalized_axis])
        condition_len = int(condition_shape[0])
        if axis_dim > 0 and condition_len > 0 and axis_dim != condition_len:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Compress condition length must match selected axis dimension when both are static. "
                    f"axis={normalized_axis} data_shape={data_shape} condition_shape={condition_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_affine_grid(node: Any, ctx: Any) -> None:
    theta_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    theta_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if theta_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "AffineGrid input/output dtype must be FLOAT16/FLOAT32. "
                f"theta_dtype={theta_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(theta_shape) != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"AffineGrid theta rank must be 3. theta_shape={theta_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    size_arr = ctx.get_constant_array(node.inputs[1].name)
    if size_arr is None:
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message="AffineGrid requires constant size input in flatbuffer_direct.",
            node_name=node.name,
            node_op=node.op,
        )
    size_values = [int(v) for v in np.asarray(size_arr).reshape(-1).tolist()]
    if len(size_values) not in {4, 5}:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"AffineGrid size input must have length 4 or 5. size={size_values}",
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in size_values):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"AffineGrid size input must be static positive. size={size_values}",
            node_name=node.name,
            node_op=node.op,
        )
    expected_theta_tail = [2, 3] if len(size_values) == 4 else [3, 4]
    if [int(theta_shape[1]), int(theta_shape[2])] != expected_theta_tail:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "AffineGrid theta shape must match dimensionality. "
                f"theta_shape={theta_shape} expected_tail={expected_theta_tail}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(theta_shape[0]) > 0 and int(theta_shape[0]) != int(size_values[0]):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "AffineGrid theta batch must match size batch when static. "
                f"theta_shape={theta_shape} size={size_values}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_output = (
        [int(size_values[0]), int(size_values[2]), int(size_values[3]), 2]
        if len(size_values) == 4
        else [int(size_values[0]), int(size_values[2]), int(size_values[3]), int(size_values[4]), 3]
    )
    if len(output_shape) != len(expected_output):
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=f"AffineGrid output rank mismatch. output_shape={output_shape} expected={expected_output}",
            node_name=node.name,
            node_op=node.op,
        )
    for actual, expected in zip(output_shape, expected_output):
        if int(actual) > 0 and int(actual) != int(expected):
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=f"AffineGrid output shape mismatch. output_shape={output_shape} expected={expected_output}",
                node_name=node.name,
                node_op=node.op,
            )
    align_corners = int(node.attrs.get("align_corners", 0))
    if align_corners not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"AffineGrid align_corners must be 0 or 1. align_corners={align_corners}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_center_crop_pad(node: Any, ctx: Any) -> None:
    input_shape = _tensor_shape_with_signature(ctx, node.inputs[0].name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if len(input_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"CenterCropPad input rank must be >= 1. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if input_dtype != output_dtype:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"CenterCropPad output dtype must match input dtype. input_dtype={input_dtype} output_dtype={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    if input_dtype == "STRING":
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message="CenterCropPad string dtype is unsupported in flatbuffer_direct.",
            node_name=node.name,
            node_op=node.op,
        )
    target_shape_arr = ctx.get_constant_array(node.inputs[1].name)
    if target_shape_arr is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="CenterCropPad requires constant target shape input in flatbuffer_direct.",
            node_name=node.name,
            node_op=node.op,
        )
    target_values = [int(v) for v in np.asarray(target_shape_arr).reshape(-1).tolist()]
    if any(int(v) <= 0 for v in target_values):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"CenterCropPad target shape values must be static positive. target_shape={target_values}",
            node_name=node.name,
            node_op=node.op,
        )
    axes_attr = node.attrs.get("axes", None)
    if axes_attr is None:
        axes = [int(v) for v in range(len(input_shape))]
    elif isinstance(axes_attr, np.ndarray):
        axes = [int(v) for v in np.asarray(axes_attr).reshape(-1).tolist()]
    elif isinstance(axes_attr, (list, tuple)):
        axes = [int(v) for v in axes_attr]
    else:
        axes = [int(axes_attr)]
    normalized_axes = _normalize_axes_for_rank(axes=axes, rank=len(input_shape), node=node)
    if len(target_values) != len(normalized_axes):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "CenterCropPad target shape length must match axes length. "
                f"target_shape={target_values} axes={normalized_axes}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(output_shape) != len(input_shape):
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=f"CenterCropPad output rank must match input rank. input_shape={input_shape} output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    expected_output = [int(v) for v in input_shape]
    for axis, size_value in zip(normalized_axes, target_values):
        expected_output[int(axis)] = int(size_value)
    for actual, expected in zip(output_shape, expected_output):
        if int(actual) > 0 and int(actual) != int(expected):
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=f"CenterCropPad output shape mismatch. output_shape={output_shape} expected={expected_output}",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_group_normalization(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    scale = _require_const_input(node, ctx, 1, "GroupNormalization scale")
    bias = _require_const_input(node, ctx, 2, "GroupNormalization bias")
    input_shape = _tensor_shape_with_signature(ctx, input_name)
    output_shape = _tensor_shape_with_signature(ctx, node.outputs[0].name)
    if len(input_shape) < 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GroupNormalization input rank must be >= 3. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if input_shape != output_shape:
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=f"GroupNormalization output shape must match input shape. input={input_shape} output={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"} or output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "GroupNormalization input/output dtype must be FLOAT16/FLOAT32. "
                f"input_dtype={input_dtype} output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    scale_size = int(np.asarray(scale).reshape(-1).size)
    bias_size = int(np.asarray(bias).reshape(-1).size)
    if scale_size != bias_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GroupNormalization scale/bias must have the same length for builtin lowering. "
                f"scale_shape={list(np.asarray(scale).shape)} bias_shape={list(np.asarray(bias).shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    op_name = str(node.op)
    preferred_channel_axes = [1, len(input_shape) - 1] if op_name == "GroupNormalization" else [len(input_shape) - 1, 1]
    channel_axis = None
    for axis in preferred_channel_axes:
        if 0 <= int(axis) < len(input_shape) and int(input_shape[int(axis)]) == scale_size:
            channel_axis = int(axis)
            break
    if channel_axis is None:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GroupNormalization builtin lowering requires the scale/bias length to match either C-first or C-last. "
                f"input_shape={input_shape} scale_shape={list(np.asarray(scale).shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    channels = int(input_shape[channel_axis])
    spatial_dims = [int(v) for idx, v in enumerate(input_shape[1:]) if int(idx) + 1 != channel_axis]
    if channels <= 0 or any(int(v) <= 0 for v in spatial_dims):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GroupNormalization builtin lowering requires static positive channel/spatial dims. "
                f"input_shape={input_shape} channel_axis={channel_axis}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    num_groups = int(node.attrs.get("num_groups", node.attrs.get("groups", 1)))
    if num_groups <= 0 or channels % num_groups != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GroupNormalization num_groups must be > 0 and divide channels. "
                f"num_groups={num_groups} channels={channels}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if scale_size != channels or bias_size != channels:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GroupNormalization scale/bias must be length=C constants for builtin lowering. "
                f"scale_shape={list(np.asarray(scale).shape)} bias_shape={list(np.asarray(bias).shape)} channels={channels}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    stash_type = int(node.attrs.get("stash_type", 1))
    if stash_type not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GroupNormalization stash_type must be 0 or 1. stash_type={stash_type}",
            node_name=node.name,
            node_op=node.op,
        )
    activation = int(node.attrs.get("activation", 0))
    if activation not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GroupNormalization activation must be 0 or 1. activation={activation}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_bitwise_not(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if input_dtype == "BOOL" or _is_integer_dtype(input_dtype):
        return
    raise NodeValidationError(
        reason_code="unsupported_input_dtype",
        message=f"BitwiseNot supports BOOL/integer input only. input_dtype={input_dtype}",
        node_name=node.name,
        node_op=node.op,
    )


def _validate_bitwise_bool_binary(node: Any, ctx: Any) -> None:
    lhs_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    rhs_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if lhs_dtype != "BOOL" or rhs_dtype != "BOOL":
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "BitwiseAnd/BitwiseOr are currently supported for BOOL tensors only "
                "in flatbuffer_direct. "
                f"lhs_dtype={lhs_dtype} rhs_dtype={rhs_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_bitwise_xor(node: Any, ctx: Any) -> None:
    lhs_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    rhs_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if lhs_dtype != rhs_dtype:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "BitwiseXor requires matching input dtypes in flatbuffer_direct. "
                f"lhs_dtype={lhs_dtype} rhs_dtype={rhs_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if lhs_dtype == "BOOL" or _is_integer_dtype(lhs_dtype):
        return
    raise NodeValidationError(
        reason_code="unsupported_input_dtype",
        message=f"BitwiseXor supports BOOL/integer only. dtype={lhs_dtype}",
        node_name=node.name,
        node_op=node.op,
    )


def _validate_bitshift(node: Any, ctx: Any) -> None:
    lhs_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if not _is_integer_dtype(lhs_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"BitShift lhs must be integer tensor. lhs_dtype={lhs_dtype}",
            node_name=node.name,
            node_op=node.op,
        )
    direction = str(node.attrs.get("direction", "RIGHT")).upper()
    if direction not in {"LEFT", "RIGHT"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"BitShift direction must be LEFT or RIGHT. direction={direction}",
            node_name=node.name,
            node_op=node.op,
        )
    if direction == "LEFT":
        _require_const_input(node, ctx, 1, "BitShift LEFT shift")
    rhs_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if not _is_integer_dtype(rhs_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"BitShift rhs must be integer tensor. rhs_dtype={rhs_dtype}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_eyelike(node: Any, ctx: Any) -> None:
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
    if len(output_shape) != 2:
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=f"EyeLike output rank must be 2 in flatbuffer_direct. output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in output_shape):
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=(
                "EyeLike requires fully static positive output shape in flatbuffer_direct. "
                f"output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_trilu(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    if len(input_shape) < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Trilu input rank must be >= 2. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(input_shape[-2]) <= 0 or int(input_shape[-1]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Trilu requires static positive matrix dimensions in flatbuffer_direct. "
                f"input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(node.inputs) >= 2:
        _require_const_input(node, ctx, 1, "Trilu k")


def _validate_l2_norm(node: Any, ctx: Any) -> None:
    p = float(node.attrs.get("p", 2.0))
    if abs(p - 2.0) > 1e-6:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LpNormalization p must be 2. got={p}",
            node_name=node.name,
            node_op=node.op,
        )
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axis = int(node.attrs.get("axis", -1))
    if axis < 0:
        axis += input_rank
    if axis != input_rank - 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LpNormalization axis must be last dim. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_lrn(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if len(input_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"LRN supports rank-4 input only in flatbuffer_direct. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    size = int(node.attrs.get("size", 0))
    if size <= 0 or size % 2 == 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LRN size must be a positive odd integer. size={size}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_einsum(node: Any, ctx: Any) -> None:
    equation = str(node.attrs.get("equation", "")).replace(" ", "")
    if equation == "":
        raise NodeValidationError(
            reason_code="missing_required_attribute",
            message="Einsum requires equation attribute.",
            node_name=node.name,
            node_op=node.op,
        )
    if "..." in equation:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Einsum ellipsis is not supported for builtin lowering. equation={equation}",
            node_name=node.name,
            node_op=node.op,
        )
    try:
        input_expr, out = equation.split("->", 1)
        input_terms = [str(v) for v in input_expr.split(",") if str(v) != ""]
    except Exception as ex:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Einsum equation format is invalid. equation={equation}",
            node_name=node.name,
            node_op=node.op,
        ) from ex
    if len(input_terms) not in {2, 3}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Einsum builtin lowering currently supports 2-input or selected 3-input equations only. "
                f"equation={equation}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    lhs = input_terms[0]
    rhs = input_terms[1]

    def _validate_generic_two_input_einsum() -> bool:
        if len(input_terms) != 2:
            return False
        if out == "":
            return False
        if len(lhs) == 2 and len(rhs) == 2 and len(out) == 2:
            # Keep the existing rank-2 matmul/fc validation path for pure matmul-style equations.
            is_matmul_style = (
                lhs[1] == rhs[0]
                and out[0] == lhs[0]
                and out[1] == rhs[1]
            )
            if is_matmul_style:
                return False
        if len(set(lhs)) != len(lhs) or len(set(rhs)) != len(rhs) or len(set(out)) != len(out):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Einsum generic builtin lowering does not support repeated labels "
                    f"within one term. equation={equation}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        lhs_set = set(lhs)
        rhs_set = set(rhs)
        out_set = set(out)
        if not out_set.issubset(lhs_set.union(rhs_set)):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Einsum output labels must be present in inputs for builtin lowering. "
                    f"equation={equation}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != len(lhs) or len(rhs_shape) != len(rhs) or len(out_shape) != len(out):
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum generic builtin lowering requires rank to match equation term length. "
                    f"equation={equation} lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        label_dims: Dict[str, int] = {}

        def _record_dim(label: str, dim: int, kind: str) -> None:
            if int(dim) <= 0:
                return
            prev = label_dims.get(label, None)
            if prev is not None and int(prev) != int(dim):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum label dimension mismatch for generic builtin lowering. "
                        f"equation={equation} label={label} prev={prev} current={dim} source={kind}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            label_dims[label] = int(dim)

        for axis, label in enumerate(lhs):
            _record_dim(str(label), int(lhs_shape[axis]), "lhs")
        for axis, label in enumerate(rhs):
            _record_dim(str(label), int(rhs_shape[axis]), "rhs")
        for axis, label in enumerate(out):
            out_dim = int(out_shape[axis])
            known_dim = label_dims.get(str(label), None)
            if known_dim is not None and out_dim > 0 and int(known_dim) != int(out_dim):
                raise NodeValidationError(
                    reason_code="unsupported_output_shape",
                    message=(
                        "Einsum output label dimension mismatch for generic builtin lowering. "
                        f"equation={equation} label={label} expected={known_dim} got={out_dim}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        return True

    # Specialized builtin lowering:
    #   nlhd,nhdv,nlh->nlhv
    # using TRANSPOSE+BATCH_MATMUL+TRANSPOSE+EXPAND_DIMS+MUL.
    if (
        len(input_terms) == 3
        and len(lhs) == 4
        and len(rhs) == 4
        and len(input_terms[2]) == 3
        and len(out) == 4
    ):
        scale = input_terms[2]
        if (
            lhs[0] == rhs[0] == scale[0] == out[0]
            and lhs[1] == scale[1] == out[1]
            and lhs[2] == rhs[1] == scale[2] == out[2]
            and lhs[3] == rhs[2]
            and rhs[3] == out[3]
            and lhs[3] not in out
        ):
            lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
            rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
            scale_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[2].name)]
            out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
            if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(scale_shape) != 3 or len(out_shape) != 4:
                raise NodeValidationError(
                    reason_code="unsupported_input_rank",
                    message=(
                        "Einsum equation nlhd,nhdv,nlh->nlhv requires lhs/rhs rank-4, scale rank-3, output rank-4. "
                        f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} scale_shape={scale_shape} out_shape={out_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

            def _known_dim(shape: List[int], axis: int) -> Optional[int]:
                dim = int(shape[axis])
                return dim if dim > 0 else None

            lhs_n = _known_dim(lhs_shape, 0)
            rhs_n = _known_dim(rhs_shape, 0)
            scale_n = _known_dim(scale_shape, 0)
            out_n = _known_dim(out_shape, 0)
            if lhs_n is not None and rhs_n is not None and lhs_n != rhs_n:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum batch dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_n={lhs_n} rhs_n={rhs_n}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if lhs_n is not None and scale_n is not None and lhs_n != scale_n:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum batch dimension mismatch between lhs and scale for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_n={lhs_n} scale_n={scale_n}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if lhs_n is not None and out_n is not None and lhs_n != out_n:
                raise NodeValidationError(
                    reason_code="unsupported_output_shape",
                    message=(
                        "Einsum output batch dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_n={lhs_n} out_n={out_n}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

            lhs_l = _known_dim(lhs_shape, 1)
            scale_l = _known_dim(scale_shape, 1)
            out_l = _known_dim(out_shape, 1)
            if lhs_l is not None and scale_l is not None and lhs_l != scale_l:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum sequence-length mismatch between lhs and scale for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_l={lhs_l} scale_l={scale_l}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if lhs_l is not None and out_l is not None and lhs_l != out_l:
                raise NodeValidationError(
                    reason_code="unsupported_output_shape",
                    message=(
                        "Einsum output l dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_l={lhs_l} out_l={out_l}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

            lhs_h = _known_dim(lhs_shape, 2)
            rhs_h = _known_dim(rhs_shape, 1)
            scale_h = _known_dim(scale_shape, 2)
            out_h = _known_dim(out_shape, 2)
            if lhs_h is not None and rhs_h is not None and lhs_h != rhs_h:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum head dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_h={lhs_h} rhs_h={rhs_h}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if lhs_h is not None and scale_h is not None and lhs_h != scale_h:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum head dimension mismatch between lhs and scale for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_h={lhs_h} scale_h={scale_h}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if lhs_h is not None and out_h is not None and lhs_h != out_h:
                raise NodeValidationError(
                    reason_code="unsupported_output_shape",
                    message=(
                        "Einsum output h dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_h={lhs_h} out_h={out_h}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

            lhs_d = _known_dim(lhs_shape, 3)
            rhs_d = _known_dim(rhs_shape, 2)
            if lhs_d is not None and rhs_d is not None and lhs_d != rhs_d:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Einsum contraction dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"lhs_d={lhs_d} rhs_d={rhs_d}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

            rhs_v = _known_dim(rhs_shape, 3)
            out_v = _known_dim(out_shape, 3)
            if rhs_v is not None and out_v is not None and rhs_v != out_v:
                raise NodeValidationError(
                    reason_code="unsupported_output_shape",
                    message=(
                        "Einsum output v dimension mismatch for equation nlhd,nhdv,nlh->nlhv. "
                        f"rhs_v={rhs_v} out_v={out_v}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            return

        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Unsupported 3-input Einsum equation for builtin lowering. "
                f"equation={equation}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    # Specialized builtin lowering:
    #   abgd,gf->abdf
    # using TRANSPOSE+RESHAPE+BATCH_MATMUL+RESHAPE.
    #
    # Specialized builtin lowering:
    #   abik,abjk->abij
    # using BATCH_MATMUL(adjY=True).
    if (
        len(lhs) == 4
        and len(rhs) == 4
        and len(out) == 4
        and lhs[0] == rhs[0] == out[0]
        and lhs[1] == rhs[1] == out[1]
        and lhs[2] == out[2]
        and rhs[2] == out[3]
        and lhs[3] == rhs[3]
        and lhs[3] not in out
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation abik,abjk->abij requires lhs/rhs/output rank-4. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    # Specialized builtin lowering:
    #   abij,abjd->abid
    # using BATCH_MATMUL.
    if (
        len(lhs) == 4
        and len(rhs) == 4
        and len(out) == 4
        and lhs[0] == rhs[0] == out[0]
        and lhs[1] == rhs[1] == out[1]
        and lhs[2] == out[2]
        and lhs[3] == rhs[2]
        and rhs[3] == out[3]
        and lhs[3] not in out
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation abij,abjd->abid requires lhs/rhs/output rank-4. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    # Specialized builtin lowering:
    #   abji,abjd->abid
    # using TRANSPOSE+BATCH_MATMUL.
    if (
        len(lhs) == 4
        and len(rhs) == 4
        and len(out) == 4
        and lhs[0] == rhs[0] == out[0]
        and lhs[1] == rhs[1] == out[1]
        and lhs[2] == rhs[2]
        and lhs[3] == out[2]
        and rhs[3] == out[3]
        and lhs[2] not in out
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation abji,abjd->abid requires lhs/rhs/output rank-4. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    # Specialized builtin lowering:
    #   amk,ank->amn
    # using BATCH_MATMUL(adjY=True).
    if (
        len(lhs) == 3
        and len(rhs) == 3
        and len(out) == 3
        and lhs[0] == rhs[0] == out[0]
        and lhs[1] == out[1]
        and rhs[1] == out[2]
        and lhs[2] == rhs[2]
        and lhs[2] not in out
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 3 or len(rhs_shape) != 3 or len(out_shape) != 3:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation amk,ank->amn requires lhs/rhs/output rank-3. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    # Specialized builtin lowering:
    #   bchw,bnc->bnhw
    # using TRANSPOSE+RESHAPE+TRANSPOSE+BATCH_MATMUL+RESHAPE+TRANSPOSE.
    if (
        len(lhs) == 4
        and len(rhs) == 3
        and len(out) == 4
        and lhs[0] == rhs[0] == out[0]
        and lhs[1] == rhs[2]
        and rhs[1] == out[1]
        and lhs[2] == out[2]
        and lhs[3] == out[3]
        and lhs[1] not in out
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 4 or len(rhs_shape) != 3 or len(out_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation bchw,bnc->bnhw requires lhs rank-4, rhs rank-3, output rank-4. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        def _known_dim(shape: List[int], axis: int) -> Optional[int]:
            dim = int(shape[axis])
            return dim if dim > 0 else None

        lhs_b = _known_dim(lhs_shape, 0)
        rhs_b = _known_dim(rhs_shape, 0)
        out_b = _known_dim(out_shape, 0)
        if lhs_b is not None and rhs_b is not None and lhs_b != rhs_b:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Einsum batch dimension mismatch for equation bchw,bnc->bnhw. "
                    f"lhs_b={lhs_b} rhs_b={rhs_b}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if lhs_b is not None and out_b is not None and lhs_b != out_b:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output batch dimension mismatch for equation bchw,bnc->bnhw. "
                    f"lhs_b={lhs_b} out_b={out_b}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_c = _known_dim(lhs_shape, 1)
        rhs_c = _known_dim(rhs_shape, 2)
        if lhs_c is not None and rhs_c is not None and lhs_c != rhs_c:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Einsum contraction dimension mismatch for equation bchw,bnc->bnhw. "
                    f"lhs_c={lhs_c} rhs_c={rhs_c}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        rhs_n = _known_dim(rhs_shape, 1)
        out_n = _known_dim(out_shape, 1)
        if rhs_n is not None and out_n is not None and rhs_n != out_n:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output n dimension mismatch for equation bchw,bnc->bnhw. "
                    f"rhs_n={rhs_n} out_n={out_n}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_h = _known_dim(lhs_shape, 2)
        out_h = _known_dim(out_shape, 2)
        if lhs_h is not None and out_h is not None and lhs_h != out_h:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output h dimension mismatch for equation bchw,bnc->bnhw. "
                    f"lhs_h={lhs_h} out_h={out_h}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_w = _known_dim(lhs_shape, 3)
        out_w = _known_dim(out_shape, 3)
        if lhs_w is not None and out_w is not None and lhs_w != out_w:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output w dimension mismatch for equation bchw,bnc->bnhw. "
                    f"lhs_w={lhs_w} out_w={out_w}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    if (
        len(lhs) == 4
        and len(rhs) == 2
        and len(out) == 4
        and lhs[2] == rhs[0]
        and out[0] == lhs[0]
        and out[1] == lhs[1]
        and out[2] == lhs[3]
        and out[3] == rhs[1]
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 4 or len(rhs_shape) != 2 or len(out_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation abgd,gf->abdf requires lhs rank-4, rhs rank-2, output rank-4. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        lhs_g = int(lhs_shape[2])
        rhs_g = int(rhs_shape[0])
        if lhs_g > 0 and rhs_g > 0 and lhs_g != rhs_g:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Einsum contraction dimension mismatch for equation abgd,gf->abdf. "
                    f"lhs_g={lhs_g} rhs_g={rhs_g}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    # Specialized builtin lowering:
    #   aijk,aijh->ajkh
    # using TRANSPOSE+TRANSPOSE+BATCH_MATMUL.
    if (
        len(lhs) == 4
        and len(rhs) == 4
        and len(out) == 4
        and lhs[0] == rhs[0]
        and lhs[1] == rhs[1]
        and lhs[2] == rhs[2]
        and out[0] == lhs[0]
        and out[1] == lhs[2]
        and out[2] == lhs[3]
        and out[3] == rhs[3]
        and lhs[1] not in out
    ):
        lhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
        rhs_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
        out_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        if len(lhs_shape) != 4 or len(rhs_shape) != 4 or len(out_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Einsum equation aijk,aijh->ajkh requires lhs/rhs/output rank-4. "
                    f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        def _known_dim(shape: List[int], axis: int) -> Optional[int]:
            dim = int(shape[axis])
            return dim if dim > 0 else None

        lhs_a = _known_dim(lhs_shape, 0)
        rhs_a = _known_dim(rhs_shape, 0)
        out_a = _known_dim(out_shape, 0)
        if lhs_a is not None and rhs_a is not None and lhs_a != rhs_a:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Einsum batch dimension mismatch for equation aijk,aijh->ajkh. "
                    f"lhs_a={lhs_a} rhs_a={rhs_a}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if lhs_a is not None and out_a is not None and lhs_a != out_a:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output batch dimension mismatch for equation aijk,aijh->ajkh. "
                    f"lhs_a={lhs_a} out_a={out_a}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_i = _known_dim(lhs_shape, 1)
        rhs_i = _known_dim(rhs_shape, 1)
        if lhs_i is not None and rhs_i is not None and lhs_i != rhs_i:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Einsum contraction dimension mismatch for equation aijk,aijh->ajkh. "
                    f"lhs_i={lhs_i} rhs_i={rhs_i}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_j = _known_dim(lhs_shape, 2)
        rhs_j = _known_dim(rhs_shape, 2)
        out_j = _known_dim(out_shape, 1)
        if lhs_j is not None and rhs_j is not None and lhs_j != rhs_j:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Einsum shared-j dimension mismatch for equation aijk,aijh->ajkh. "
                    f"lhs_j={lhs_j} rhs_j={rhs_j}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if lhs_j is not None and out_j is not None and lhs_j != out_j:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output j dimension mismatch for equation aijk,aijh->ajkh. "
                    f"lhs_j={lhs_j} out_j={out_j}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        lhs_k = _known_dim(lhs_shape, 3)
        out_k = _known_dim(out_shape, 2)
        if lhs_k is not None and out_k is not None and lhs_k != out_k:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output k dimension mismatch for equation aijk,aijh->ajkh. "
                    f"lhs_k={lhs_k} out_k={out_k}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

        rhs_h = _known_dim(rhs_shape, 3)
        out_h = _known_dim(out_shape, 3)
        if rhs_h is not None and out_h is not None and rhs_h != out_h:
            raise NodeValidationError(
                reason_code="unsupported_output_shape",
                message=(
                    "Einsum output h dimension mismatch for equation aijk,aijh->ajkh. "
                    f"rhs_h={rhs_h} out_h={out_h}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        return

    if _validate_generic_two_input_einsum():
        return

    if len(lhs) != 2 or len(rhs) != 2 or len(out) != 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Einsum builtin lowering currently supports rank-2 matmul-style equations only. "
                f"equation={equation}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    is_matmul_style = (
        lhs[1] == rhs[0]
        and out[0] == lhs[0]
        and out[1] == rhs[1]
    )
    if not is_matmul_style:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "Einsum equation is not matmul-style for builtin lowering. "
                f"equation={equation}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    rhs_name = node.inputs[1].name
    if ctx.get_constant_array(rhs_name) is not None:
        _validate_fc(node, ctx)
    else:
        _validate_matmul(node, ctx)














def _validate_flatten(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Flatten input rank must be >= 1. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )




























def _validate_resize(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    input_is_unknown_placeholder = _is_unknown_rank_placeholder_tensor(ctx, node.inputs[0].name)
    output_is_unknown_placeholder = _is_unknown_rank_placeholder_tensor(ctx, node.outputs[0].name)
    input_rank = len(input_shape)
    output_rank = len(output_shape)
    if input_rank not in {3, 4, 5} and not input_is_unknown_placeholder:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Resize supports rank-3/4/5 input. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if output_rank not in {3, 4, 5} and not output_is_unknown_placeholder:
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=f"Resize supports rank-3/4/5 output. output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if (
        not input_is_unknown_placeholder
        and not output_is_unknown_placeholder
        and input_rank != output_rank
    ):
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=(
                "Resize requires matching input/output rank in flatbuffer_direct. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    mode = str(node.attrs.get("mode", "nearest")).lower()
    if mode not in ["nearest", "linear", "cubic"]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Resize mode must be nearest, linear, or cubic. got={mode}",
            node_name=node.name,
            node_op=node.op,
        )
    if input_rank == 5:
        if mode == "cubic":
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message="Resize rank-5 supports nearest/linear modes only in flatbuffer_direct.",
                node_name=node.name,
                node_op=node.op,
            )
        if any(int(v) <= 0 for v in list(input_shape) + list(output_shape)):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Resize rank-5 requires static positive input/output shapes in "
                    f"flatbuffer_direct. input_shape={input_shape} output_shape={output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    default_ctm = "asymmetric" if str(node.op) == "Upsample" else "half_pixel"
    ctm = str(node.attrs.get("coordinate_transformation_mode", default_ctm)).lower()
    if mode == "nearest":
        if ctm not in ["asymmetric", "half_pixel"]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Resize(nearest) supports coordinate_transformation_mode "
                    f"asymmetric/half_pixel only. got={ctm}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        nearest_mode = str(node.attrs.get("nearest_mode", "round_prefer_floor")).lower()
        supported_nearest_modes = {"floor", "round_prefer_floor"}
        if ctm == "half_pixel":
            # TFLite half_pixel_centers computes
            # floor((out + 0.5) * input_size / output_size), which is ONNX
            # half_pixel followed by round_prefer_ceil.
            supported_nearest_modes.add("round_prefer_ceil")
        if nearest_mode not in supported_nearest_modes:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Resize nearest_mode is not representable by the selected "
                    f"TFLite coordinate mode. ctm={ctm} got={nearest_mode}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    else:
        if ctm not in ["half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners"]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Resize(linear/cubic) supports coordinate_transformation_mode "
                    f"half_pixel/pytorch_half_pixel/asymmetric/align_corners only. got={ctm}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    has_const_param = False
    has_dynamic_sizes_param = False

    def _validate_dynamic_resize_sizes_input(tensor_name: str) -> None:
        sizes_shape = ctx.get_tensor_shape(tensor_name)
        if sizes_shape != [1] and len(sizes_shape) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    "Resize dynamic sizes input must be rank-1. "
                    f"sizes_shape={sizes_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if len(sizes_shape) == 1:
            sizes_len = int(sizes_shape[0])
            if sizes_len == 1:
                # Placeholder length from symbolic shape inference.
                sizes_len = -1
            expected_lengths = [2, 3] if input_rank == 3 else [2, 4]
            if sizes_len > 0 and sizes_len not in expected_lengths:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        f"Resize dynamic sizes input length must be one of {expected_lengths}. "
                        f"sizes_shape={sizes_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        sizes_dtype = str(ctx.get_tensor_dtype(tensor_name)).upper()
        if sizes_dtype not in {"INT32", "INT64"}:
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=(
                    "Resize dynamic sizes input must be INT32/INT64. "
                    f"sizes_dtype={sizes_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 4:
        tensor_name = node.inputs[3].name
        if tensor_name != "":
            sizes_arr = ctx.get_constant_array(tensor_name)
            if sizes_arr is not None:
                if int(np.asarray(sizes_arr).size) > 0:
                    has_const_param = True
            else:
                has_dynamic_sizes_param = True
                _validate_dynamic_resize_sizes_input(tensor_name)
    if len(node.inputs) >= 3:
        tensor_name = node.inputs[2].name
        if tensor_name != "":
            arr = _require_const_input(node, ctx, 2, "Resize scales")
            if int(np.asarray(arr).size) > 0:
                has_const_param = True
    if len(node.inputs) == 2:
        tensor_name = node.inputs[1].name
        if tensor_name != "":
            arr = ctx.get_constant_array(tensor_name)
            if arr is not None:
                if int(np.asarray(arr).size) > 0:
                    has_const_param = True
            else:
                # _NodeWrap drops optional empty inputs, so
                # Resize(x, "", "", sizes) may appear as 2-input form.
                has_dynamic_sizes_param = True
                _validate_dynamic_resize_sizes_input(tensor_name)
    if not has_const_param and not has_dynamic_sizes_param:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="Resize requires non-empty constant scales/sizes or dynamic sizes input.",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_grid_sample(node: Any, ctx: Any) -> None:
    mode = str(node.attrs.get("mode", "bilinear")).lower()
    padding_mode = str(node.attrs.get("padding_mode", "zeros")).lower()
    align_corners = int(node.attrs.get("align_corners", 0))
    if mode not in {"bilinear", "linear", "nearest"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GridSample supports mode=bilinear/linear/nearest only in "
                f"flatbuffer_direct. mode={mode}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if padding_mode not in {"zeros", "border"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GridSample supports padding_mode in {zeros,border} only in flatbuffer_direct. "
                f"padding_mode={padding_mode}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if align_corners not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GridSample supports align_corners in {0,1} only in flatbuffer_direct. "
                f"align_corners={align_corners}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    image_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    grid_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]

    def _merge_with_raw_shape(tensor_name: str, resolved_shape: List[int]) -> List[int]:
        raw = None
        if hasattr(ctx, "shape_map") and isinstance(ctx.shape_map, dict):
            raw = ctx.shape_map.get(str(tensor_name), None)
        if raw is None:
            return [int(v) for v in list(resolved_shape)]
        try:
            raw_shape = [int(v) for v in list(raw)]
        except Exception:
            return [int(v) for v in list(resolved_shape)]
        if len(raw_shape) != len(resolved_shape):
            return [int(v) for v in list(resolved_shape)]
        merged: List[int] = []
        for resolved_dim, raw_dim in zip(resolved_shape, raw_shape):
            if int(raw_dim) > 0:
                merged.append(int(raw_dim))
            elif int(raw_dim) <= 0 and int(resolved_dim) == 1:
                # Preserve unknown-dimension intent from ONNX shape map.
                merged.append(-1)
            else:
                merged.append(int(resolved_dim))
        return merged

    image_shape = _merge_with_raw_shape(node.inputs[0].name, image_shape)
    grid_shape = _merge_with_raw_shape(node.inputs[1].name, grid_shape)
    output_shape = _merge_with_raw_shape(node.outputs[0].name, output_shape)
    rank = int(len(image_shape))
    if rank not in {4, 5} or len(grid_shape) != rank:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "GridSample supports rank-4/5 tensors only in flatbuffer_direct. "
                f"image_shape={image_shape} grid_shape={grid_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if mode == "nearest" and rank != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message="GridSample mode=nearest currently supports rank-4 input only.",
            node_name=node.name,
            node_op=node.op,
        )

    expected_grid_last_dim = 2 if rank == 4 else 3
    if int(grid_shape[-1]) > 0 and int(grid_shape[-1]) != int(expected_grid_last_dim):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                f"GridSample grid last dimension must be {expected_grid_last_dim} "
                + ("([x,y])" if rank == 4 else "([x,y,z])")
                + " in flatbuffer_direct. "
                f"grid_shape={grid_shape}. "
                "When grid is a model input, pass -kat/--keep_shape_absolutely_input_names for it."
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(grid_shape[-1]) <= 0:
        grid_shape[-1] = int(expected_grid_last_dim)

    expected_output_shape = (
        [int(image_shape[0]), int(image_shape[1]), int(grid_shape[1]), int(grid_shape[2])]
        if rank == 4
        else [int(image_shape[0]), int(image_shape[1]), int(grid_shape[1]), int(grid_shape[2]), int(grid_shape[3])]
    )
    resolved_output_shape = [int(v) for v in list(output_shape)]
    if len(resolved_output_shape) != rank:
        resolved_output_shape = [int(v) for v in list(expected_output_shape)]
    else:
        resolved_output_shape = [
            int(expected_output_shape[idx]) if int(dim) <= 0 else int(dim)
            for idx, dim in enumerate(resolved_output_shape)
        ]

    def _dims_match_when_known(*dims: int) -> bool:
        known = [int(v) for v in dims if int(v) > 0]
        return len(set(known)) <= 1

    if rank == 4:
        required_static_dims = [
            int(image_shape[0]),
            int(image_shape[1]),
            int(image_shape[2]),
            int(image_shape[3]),
            int(grid_shape[0]),
            int(grid_shape[3]),
            int(resolved_output_shape[0]),
            int(resolved_output_shape[1]),
        ]
    else:
        required_static_dims = [
            int(image_shape[0]),
            int(image_shape[1]),
            int(image_shape[2]),
            int(image_shape[3]),
            int(image_shape[4]),
            int(grid_shape[0]),
            int(grid_shape[4]),
            int(resolved_output_shape[0]),
            int(resolved_output_shape[1]),
        ]

    if any(int(v) <= 0 for v in required_static_dims):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GridSample requires static positive batch/channel/input-spatial dimensions "
                "and a static grid last dimension in flatbuffer_direct. "
                f"image_shape={image_shape} grid_shape={grid_shape} "
                f"output_shape={output_shape} resolved_output_shape={resolved_output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if rank == 4:
        n, c, h, w = [int(v) for v in image_shape]
        out_n, out_c, out_h, out_w = [int(v) for v in resolved_output_shape]
        grid_n, grid_h, grid_w, _ = [int(v) for v in grid_shape]
        if not (
            _dims_match_when_known(n, out_n, grid_n)
            and _dims_match_when_known(c, out_c)
            and _dims_match_when_known(out_h, grid_h)
            and _dims_match_when_known(out_w, grid_w)
            and h >= 1
            and w >= 1
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GridSample input/grid/output shapes are inconsistent for built-in lowering. "
                    f"image_shape={image_shape} grid_shape={grid_shape} "
                    f"output_shape={output_shape} resolved_output_shape={resolved_output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    else:
        n, c, d, h, w = [int(v) for v in image_shape]
        out_n, out_c, out_d, out_h, out_w = [int(v) for v in resolved_output_shape]
        grid_n, grid_d, grid_h, grid_w, _ = [int(v) for v in grid_shape]
        if not (
            _dims_match_when_known(n, out_n, grid_n)
            and _dims_match_when_known(c, out_c)
            and _dims_match_when_known(out_d, grid_d)
            and _dims_match_when_known(out_h, grid_h)
            and _dims_match_when_known(out_w, grid_w)
            and d >= 1
            and h >= 1
            and w >= 1
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GridSample input/grid/output shapes are inconsistent for built-in lowering. "
                    f"image_shape={image_shape} grid_shape={grid_shape} "
                    f"output_shape={output_shape} resolved_output_shape={resolved_output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    image_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    grid_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if image_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_type",
            message=(
                "GridSample input dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"image_dtype={image_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if grid_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_type",
            message=(
                "GridSample grid dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"grid_dtype={grid_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_type",
            message=(
                "GridSample output dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_roi_align(node: Any, ctx: Any) -> None:
    coordinate_transformation_mode = str(
        node.attrs.get("coordinate_transformation_mode", "half_pixel")
    ).lower()
    if coordinate_transformation_mode not in {"half_pixel", "output_half_pixel"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RoiAlign supports coordinate_transformation_mode in "
                "{half_pixel,output_half_pixel} only in flatbuffer_direct. "
                f"coordinate_transformation_mode={coordinate_transformation_mode}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    mode = str(node.attrs.get("mode", "avg")).lower()
    if mode not in {"avg", "max"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RoiAlign supports mode in {avg,max} only in flatbuffer_direct. "
                f"mode={mode}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    output_height = int(node.attrs.get("output_height", 1))
    output_width = int(node.attrs.get("output_width", 1))
    if int(output_height) <= 0 or int(output_width) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RoiAlign requires positive output_height/output_width in flatbuffer_direct. "
                f"output_height={output_height} output_width={output_width}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    sampling_ratio = int(node.attrs.get("sampling_ratio", 0))
    if int(sampling_ratio) < 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RoiAlign sampling_ratio must be >= 0 in flatbuffer_direct. "
                f"sampling_ratio={sampling_ratio}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    x_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    rois_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    batch_indices_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[2].name)]
    y_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
    if len(x_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"RoiAlign input x must be rank-4. x_shape={x_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(rois_shape) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"RoiAlign input rois must be rank-2. rois_shape={rois_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(batch_indices_shape) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "RoiAlign input batch_indices must be rank-1. "
                f"batch_indices_shape={batch_indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(y_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=f"RoiAlign output must be rank-4. y_shape={y_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    if int(rois_shape[1]) > 0 and int(rois_shape[1]) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RoiAlign rois second dimension must be 4 when statically known. "
                f"rois_shape={rois_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if any(int(v) <= 0 for v in [x_shape[1], x_shape[2], x_shape[3]]):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RoiAlign requires static positive input C/H/W in flatbuffer_direct. "
                f"x_shape={x_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if int(y_shape[1]) > 0 and int(x_shape[1]) > 0 and int(y_shape[1]) != int(x_shape[1]):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "RoiAlign output channel dimension must match input channel dimension. "
                f"x_shape={x_shape} y_shape={y_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(y_shape[2]) > 0 and int(y_shape[2]) != int(output_height):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "RoiAlign output height must match output_height attribute when statically known. "
                f"y_shape={y_shape} output_height={output_height}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(y_shape[3]) > 0 and int(y_shape[3]) != int(output_width):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "RoiAlign output width must match output_width attribute when statically known. "
                f"y_shape={y_shape} output_width={output_width}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    x_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    rois_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    batch_indices_dtype = str(ctx.get_tensor_dtype(node.inputs[2].name)).upper()
    y_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if x_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "RoiAlign input x dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"x_dtype={x_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if rois_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "RoiAlign input rois dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"rois_dtype={rois_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not _is_integer_dtype(batch_indices_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "RoiAlign input batch_indices dtype must be integer in flatbuffer_direct. "
                f"batch_indices_dtype={batch_indices_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if y_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "RoiAlign output dtype must be FLOAT16/FLOAT32 in flatbuffer_direct. "
                f"y_dtype={y_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_prelu(node: Any, ctx: Any) -> None:
    slope = _require_const_input(node, ctx, 1, "PRelu slope")
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    input_rank = len(input_shape)
    slope_size = int(np.asarray(slope).size)
    if slope_size <= 1:
        return
    if input_shape == [1] or input_rank <= 1:
        # Unknown/placeholder shape. Defer broadcast validation to runtime.
        return
    raw_shape = (
        ctx.shape_map.get(str(node.inputs[0].name), None)
        if hasattr(ctx, "shape_map") and isinstance(ctx.shape_map, dict)
        else None
    )
    raw_shape_is_unresolved = (
        raw_shape is None
        or not isinstance(raw_shape, (list, tuple))
        or len(raw_shape) == 0
        or any(
            not isinstance(dim, (int, np.integer)) or int(dim) <= 0
            for dim in raw_shape
        )
    )
    if raw_shape_is_unresolved:
        # Shape reconciliation can recover channels from the producer after
        # lowering. A rank-shaped all-ones placeholder is not evidence that a
        # valid per-channel slope is incompatible.
        return
    if input_rank in [2, 4] and len(input_shape) >= 2:
        channels = int(input_shape[1])
        if slope_size == channels:
            return
    raise NodeValidationError(
        reason_code="unsupported_attribute_value",
        message=(
            "PRelu slope supports scalar or per-channel only in flatbuffer_direct. "
            f"input_shape={input_shape} slope_size={slope_size}"
        ),
        node_name=node.name,
        node_op=node.op,
    )


def _validate_if(node: Any, ctx: Any) -> None:
    _ = ctx
    if not (
        is_supported_if_nms_guard_pattern(node)
        or is_supported_if_axis0_add_branch_pattern(node)
        or is_supported_if_sequenceconstruct_add_branch_pattern(node)
        or is_supported_if_nested_reducemin_add_branch_pattern(node)
        or is_supported_if_generic_branch_mux_pattern(node, ctx)
    ):
        raise NodeValidationError(
            reason_code="unsupported_control_flow_pattern",
            message=(
                "If built-in lowering supports only constrained patterns: "
                "NMS-guard pattern (empty-then + NMS-else), "
                "axis0 Add-branch pattern (single Add in each branch), "
                "or SequenceConstruct Add-branch pattern "
                "(branch-local Constant/Add + terminal SequenceConstruct), "
                "or nested ReduceMin/Add pattern (else-branch ReduceMin/Greater + nested Add/Add If), "
                "or generic branch-mux pattern (no branch inputs, safe branch-op subset, "
                "and matching then/else outputs)."
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_loop(node: Any, ctx: Any) -> None:
    if not (
        is_supported_loop_static_unroll_pattern(node, ctx)
        or is_supported_loop_while_pattern(node, ctx)
    ):
        raise NodeValidationError(
            reason_code="unsupported_control_flow_pattern",
            message=(
                "Loop built-in lowering supports either static-unroll patterns with constant trip_count/cond "
                "or WHILE patterns with loop-carried outputs only (no scan outputs)."
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_string_normalizer(node: Any, ctx: Any) -> None:
    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype != "STRING":
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "StringNormalizer input dtype must be STRING for builtin lowering. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype != "STRING":
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "StringNormalizer output dtype must be STRING for builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    locale = str(node.attrs.get("locale", "en_US")).strip()
    if locale not in {"", "en_US"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "StringNormalizer builtin lowering supports locale '' or 'en_US' only. "
                f"locale={locale}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    # Constant input path is evaluated exactly during lowering.
    if ctx.get_constant_array(node.inputs[0].name) is not None:
        return

    case_change_action = str(node.attrs.get("case_change_action", "NONE")).strip().upper()
    stopwords = node.attrs.get("stopwords", [])
    if stopwords is None:
        stopwords = []
    if isinstance(stopwords, str):
        stopwords = [stopwords]
    stopword_count = len(list(stopwords))
    is_case_sensitive = bool(node.attrs.get("is_case_sensitive", 1))

    if case_change_action not in {"", "NONE"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "StringNormalizer builtin lowering does not support runtime LOWER/UPPER conversion. "
                f"case_change_action={case_change_action}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if stopword_count > 0 and not is_case_sensitive:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "StringNormalizer builtin lowering does not support case-insensitive runtime stopword matching. "
                f"is_case_sensitive={is_case_sensitive}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank not in {1, 2}:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "StringNormalizer builtin lowering supports rank1/rank2 input only. "
                f"input_rank={input_rank}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if stopword_count > 0:
        output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
        output_rank = len(output_shape)
        expected_output_rank = 1 if input_rank == 1 else 2
        if output_rank != expected_output_rank:
            raise NodeValidationError(
                reason_code="unsupported_output_rank",
                message=(
                    "StringNormalizer stopword filtering builtin lowering requires output rank to match "
                    "the filtered tensor rank. "
                    f"input_rank={input_rank} output_rank={output_rank}"
                ),
                node_name=node.name,
                node_op=node.op,
            )




def _make_binary_builder(tflite_op: str) -> Callable[[Any, Any], None]:
    def _builder(node: Any, ctx: Any) -> None:
        build_binary_op(node, ctx, tflite_op)

    return _builder


def _make_unary_builder(tflite_op: str) -> Callable[[Any, Any], None]:
    def _builder(node: Any, ctx: Any) -> None:
        build_unary_op(node, ctx, tflite_op)

    return _builder


def _custom_dispatch_entry_for_node(node_op: str) -> DispatchEntry:
    return DispatchEntry(
        onnx_op=str(node_op),
        tflite_ops=["CUSTOM"],
        builder=build_custom_passthrough_op,
        validation=ValidationSpec(min_inputs=0, max_inputs=None, min_outputs=1, max_outputs=None),
    )


def _normalize_custom_op_allowlist(allowlist: Optional[Any]) -> set:
    if allowlist is None:
        return set()
    if isinstance(allowlist, (str, bytes)):
        items = [str(allowlist)]
    else:
        try:
            items = [str(v) for v in list(allowlist)]
        except Exception:
            items = [str(allowlist)]
    normalized = set()
    for item in items:
        v = str(item).strip()
        if v != "":
            normalized.add(v.upper())
    return normalized


def _resolve_custom_candidate(node: Any, ctx: Any) -> Optional[DispatchResolution]:
    if str(node.op) not in _CUSTOM_OP_CANDIDATES:
        return None
    allow_custom_ops = bool(getattr(ctx, "allow_custom_ops", False))
    if not allow_custom_ops:
        raise NodeValidationError(
            reason_code="custom_op_candidate_disabled",
            message=(
                "This ONNX op is a custom-op candidate, but custom-op lowering is disabled. "
                f"Enable flatbuffer_direct_allow_custom_ops to lower it as CUSTOM. op={node.op}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    allowlist = _normalize_custom_op_allowlist(
        getattr(ctx, "custom_op_allowlist", None)
    )
    if len(allowlist) > 0 and str(node.op).upper() not in allowlist:
        raise NodeValidationError(
            reason_code="custom_op_not_in_allowlist",
            message=(
                "This ONNX op is a custom-op candidate but not included in custom_op_allowlist. "
                f"op={node.op}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    return DispatchResolution(
        entry=_custom_dispatch_entry_for_node(str(node.op)),
        dispatch_mode="custom",
        reason_code="custom_op_lowered",
        message=f"Lowered as CUSTOM op with customCode=ONNX_{str(node.op).upper()}",
    )


def _resolve_generic_custom_fallback(node: Any, ctx: Any) -> Optional[DispatchResolution]:
    if str(node.op) in {
        "ArgMax",
        "ArgMin",
        "BitShift",
        "Cast",
        "Expand",
        "EyeLike",
        "GatherElements",
        "GRU",
        "Hardmax",
        "Mod",
        "NonZero",
        "Range",
        "ReduceL1",
        "ReduceL2",
        "RNN",
        "Trilu",
        "Where",
    }:
        return None
    allow_custom_ops = bool(getattr(ctx, "allow_custom_ops", False))
    if not allow_custom_ops:
        return None
    allowlist = _normalize_custom_op_allowlist(
        getattr(ctx, "custom_op_allowlist", None)
    )
    if len(allowlist) > 0 and str(node.op).upper() not in allowlist:
        return None
    return DispatchResolution(
        entry=_custom_dispatch_entry_for_node(str(node.op)),
        dispatch_mode="custom",
        reason_code="custom_op_lowered_generic",
        message=f"Lowered as CUSTOM op with customCode=ONNX_{str(node.op).upper()}",
    )


def _validate_clip(node: Any, ctx: Any) -> None:
    min_value = node.attrs.get("min", float("-inf"))
    max_value = node.attrs.get("max", float("inf"))
    min_known = True
    max_known = True
    if len(node.inputs) >= 2 and str(node.inputs[1].name) != "":
        min_const = ctx.get_constant_array(str(node.inputs[1].name))
        if min_const is not None:
            min_value = min_const
        else:
            min_known = False
    if len(node.inputs) >= 3 and str(node.inputs[2].name) != "":
        max_const = ctx.get_constant_array(str(node.inputs[2].name))
        if max_const is not None:
            max_value = max_const
        else:
            max_known = False

    def _to_float(v: Any, default: float) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        arr = np.asarray(v)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])

    if min_known and max_known:
        min_f = _to_float(min_value, float("-inf"))
        max_f = _to_float(max_value, float("inf"))
        if np.isfinite(min_f) and np.isfinite(max_f) and min_f > max_f:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "Clip minimum must be <= maximum. "
                    f"min={min_f} max={max_f}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


_DISPATCH_REGISTRY: Dict[str, DispatchEntry] = {
    "Add": DispatchEntry(
        onnx_op="Add",
        tflite_ops=["ADD"],
        builder=_make_binary_builder("ADD"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Sum": DispatchEntry(
        onnx_op="Sum",
        tflite_ops=["ADD"],
        builder=build_sum_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=None, min_outputs=1, max_outputs=1),
    ),
    "Sub": DispatchEntry(
        onnx_op="Sub",
        tflite_ops=["SUB"],
        builder=_make_binary_builder("SUB"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Mul": DispatchEntry(
        onnx_op="Mul",
        tflite_ops=["MUL"],
        builder=_make_binary_builder("MUL"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Div": DispatchEntry(
        onnx_op="Div",
        tflite_ops=["DIV", "MUL"],
        builder=build_div_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Min": DispatchEntry(
        onnx_op="Min",
        tflite_ops=["MINIMUM"],
        builder=build_min_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=None, min_outputs=1, max_outputs=1),
    ),
    "Max": DispatchEntry(
        onnx_op="Max",
        tflite_ops=["MAXIMUM"],
        builder=build_max_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=None, min_outputs=1, max_outputs=1),
    ),
    "Abs": DispatchEntry(
        onnx_op="Abs",
        tflite_ops=["ABS", "NEG", "MAXIMUM"],
        builder=build_abs_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Acos": DispatchEntry(
        onnx_op="Acos",
        tflite_ops=["MUL", "SUB", "SQRT", "ATAN2"],
        builder=build_acos_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Acosh": DispatchEntry(
        onnx_op="Acosh",
        tflite_ops=["SUB", "ADD", "SQRT", "MUL", "LOG"],
        builder=build_acosh_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Asin": DispatchEntry(
        onnx_op="Asin",
        tflite_ops=["MUL", "SUB", "SQRT", "ATAN2"],
        builder=build_asin_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Asinh": DispatchEntry(
        onnx_op="Asinh",
        tflite_ops=["MUL", "ADD", "SQRT", "LOG"],
        builder=build_asinh_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Atan": DispatchEntry(
        onnx_op="Atan",
        tflite_ops=["ATAN2"],
        builder=build_atan_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Atanh": DispatchEntry(
        onnx_op="Atanh",
        tflite_ops=["ADD", "SUB", "DIV", "LOG", "MUL"],
        builder=build_atanh_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Reciprocal": DispatchEntry(
        onnx_op="Reciprocal",
        tflite_ops=["DIV"],
        builder=build_reciprocal_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reciprocal,
    ),
    "Inverse": DispatchEntry(
        onnx_op="Inverse",
        tflite_ops=["SLICE", "MUL", "SUB", "ADD", "NEG", "DIV", "CONCATENATION"],
        builder=build_inverse_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_inverse,
    ),
    "Mod": DispatchEntry(
        onnx_op="Mod",
        tflite_ops=["FLOOR_MOD"],
        builder=build_mod_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_mod,
    ),
    "BitShift": DispatchEntry(
        onnx_op="BitShift",
        tflite_ops=["RIGHT_SHIFT", "MUL"],
        builder=build_bitshift_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_bitshift,
    ),
    "BitwiseAnd": DispatchEntry(
        onnx_op="BitwiseAnd",
        tflite_ops=["LOGICAL_AND"],
        builder=_make_binary_builder("LOGICAL_AND"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_bitwise_bool_binary,
    ),
    "BitwiseOr": DispatchEntry(
        onnx_op="BitwiseOr",
        tflite_ops=["LOGICAL_OR"],
        builder=_make_binary_builder("LOGICAL_OR"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_bitwise_bool_binary,
    ),
    "BitwiseNot": DispatchEntry(
        onnx_op="BitwiseNot",
        tflite_ops=["LOGICAL_NOT", "SUB", "CAST"],
        builder=build_bitwise_not_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_bitwise_not,
    ),
    "BitwiseXor": DispatchEntry(
        onnx_op="BitwiseXor",
        tflite_ops=["BITWISE_XOR"],
        builder=_make_binary_builder("BITWISE_XOR"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_bitwise_xor,
    ),
    "Cast": DispatchEntry(
        onnx_op="Cast",
        tflite_ops=["CAST"],
        builder=build_cast_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_cast,
    ),
    "CastLike": DispatchEntry(
        onnx_op="CastLike",
        tflite_ops=["CAST"],
        builder=build_castlike_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Expand": DispatchEntry(
        onnx_op="Expand",
        tflite_ops=["RESHAPE", "MUL", "CAST"],
        builder=build_expand_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_expand,
    ),
    "Tile": DispatchEntry(
        onnx_op="Tile",
        tflite_ops=["CAST", "TILE"],
        builder=build_tile_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_tile,
    ),
    "QuantizeLinear": DispatchEntry(
        onnx_op="QuantizeLinear",
        tflite_ops=["QUANTIZE"],
        builder=build_quantize_linear_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_quantize_dequantize_linear,
    ),
    "DynamicQuantizeLinear": DispatchEntry(
        onnx_op="DynamicQuantizeLinear",
        tflite_ops=[
            "NEG",
            "REDUCE_MAX",
            "MINIMUM",
            "MAXIMUM",
            "SUB",
            "DIV",
            "ADD",
            "CAST",
        ],
        builder=build_dynamic_quantize_linear_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=3, max_outputs=3),
        extra_validator=_validate_dynamic_quantize_linear,
    ),
    "DequantizeLinear": DispatchEntry(
        onnx_op="DequantizeLinear",
        tflite_ops=["DEQUANTIZE"],
        builder=build_dequantize_linear_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_quantize_dequantize_linear,
    ),
    "QLinearAdd": DispatchEntry(
        onnx_op="QLinearAdd",
        tflite_ops=["ADD"],
        builder=build_qlinear_add_op,
        validation=ValidationSpec(min_inputs=8, max_inputs=8, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_binary,
    ),
    "QLinearConcat": DispatchEntry(
        onnx_op="QLinearConcat",
        tflite_ops=["DEQUANTIZE", "CONCATENATION", "QUANTIZE"],
        builder=build_qlinear_concat_op,
        validation=ValidationSpec(min_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_concat,
    ),
    "QLinearMul": DispatchEntry(
        onnx_op="QLinearMul",
        tflite_ops=["MUL"],
        builder=build_qlinear_mul_op,
        validation=ValidationSpec(min_inputs=8, max_inputs=8, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_binary,
    ),
    "QLinearConv": DispatchEntry(
        onnx_op="QLinearConv",
        tflite_ops=["CONV_2D", "DEPTHWISE_CONV_2D"],
        builder=build_qlinear_conv_op,
        validation=ValidationSpec(
            min_inputs=8,
            max_inputs=9,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_qlinear_conv,
    ),
    "ConvInteger": DispatchEntry(
        onnx_op="ConvInteger",
        tflite_ops=["CAST", "SUB", "PAD", "CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE"],
        builder=build_conv_integer_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=4,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_conv_integer,
    ),
    "QLinearMatMul": DispatchEntry(
        onnx_op="QLinearMatMul",
        tflite_ops=["FULLY_CONNECTED"],
        builder=build_qlinear_matmul_op,
        validation=ValidationSpec(min_inputs=8, max_inputs=8, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_matmul,
    ),
    "QGemm": DispatchEntry(
        onnx_op="QGemm",
        tflite_ops=["FULLY_CONNECTED"],
        builder=build_qgemm_op,
        validation=ValidationSpec(min_inputs=9, max_inputs=9, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qgemm,
    ),
    "QLinearSigmoid": DispatchEntry(
        onnx_op="QLinearSigmoid",
        tflite_ops=["DEQUANTIZE", "LOGISTIC", "QUANTIZE"],
        builder=build_qlinear_sigmoid_op,
        validation=ValidationSpec(min_inputs=5, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_sigmoid,
    ),
    "QLinearLeakyRelu": DispatchEntry(
        onnx_op="QLinearLeakyRelu",
        tflite_ops=["DEQUANTIZE", "PRELU", "QUANTIZE"],
        builder=build_qlinear_leaky_relu_op,
        validation=ValidationSpec(min_inputs=5, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_leaky_relu,
    ),
    "QLinearSoftmax": DispatchEntry(
        onnx_op="QLinearSoftmax",
        tflite_ops=["DEQUANTIZE", "SOFTMAX", "QUANTIZE"],
        builder=build_qlinear_softmax_op,
        validation=ValidationSpec(min_inputs=5, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_softmax,
    ),
    "QLinearGlobalAveragePool": DispatchEntry(
        onnx_op="QLinearGlobalAveragePool",
        tflite_ops=["DEQUANTIZE", "MEAN", "QUANTIZE"],
        builder=build_qlinear_global_average_pool_op,
        validation=ValidationSpec(min_inputs=5, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_qlinear_global_average_pool,
    ),
    "QLinearAveragePool": DispatchEntry(
        onnx_op="QLinearAveragePool",
        tflite_ops=["DEQUANTIZE", "TRANSPOSE", "AVERAGE_POOL_2D", "TRANSPOSE", "QUANTIZE"],
        builder=build_qlinear_average_pool_op,
        validation=ValidationSpec(
            min_inputs=5,
            max_inputs=5,
            min_outputs=1,
            max_outputs=1,
            required_attrs=["kernel_shape"],
        ),
        extra_validator=_validate_qlinear_average_pool,
    ),
    "BatchNormalization": DispatchEntry(
        onnx_op="BatchNormalization",
        tflite_ops=["RESHAPE", "CAST", "ADD", "SQRT", "DIV", "MUL", "SUB"],
        builder=build_batch_normalization_op,
        validation=ValidationSpec(min_inputs=5, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_batch_norm,
    ),
    "InstanceNormalization": DispatchEntry(
        onnx_op="InstanceNormalization",
        tflite_ops=["MEAN", "SUB", "MUL", "ADD", "SQRT", "DIV"],
        builder=build_instance_normalization_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_instance_norm,
    ),
    "MeanVarianceNormalization": DispatchEntry(
        onnx_op="MeanVarianceNormalization",
        tflite_ops=["MEAN", "SUB", "MUL", "ADD", "SQRT", "DIV"],
        builder=build_mean_variance_normalization_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_mean_variance_normalization,
    ),
    "GroupNormalization": DispatchEntry(
        onnx_op="GroupNormalization",
        tflite_ops=["RESHAPE", "MEAN", "SUB", "MUL", "ADD", "SQRT", "DIV", "CAST"],
        builder=build_group_normalization_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_group_normalization,
    ),
    "GroupNorm": DispatchEntry(
        onnx_op="GroupNorm",
        tflite_ops=["RESHAPE", "MEAN", "SUB", "MUL", "ADD", "SQRT", "DIV", "CAST", "LOGISTIC"],
        builder=build_group_normalization_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_group_normalization,
    ),
    "LayerNormalization": DispatchEntry(
        onnx_op="LayerNormalization",
        tflite_ops=["MEAN", "SUB", "MUL", "ADD", "SQRT", "DIV", "CAST"],
        builder=build_layer_normalization_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=3),
        extra_validator=_validate_layer_norm,
    ),
    "ReduceMean": DispatchEntry(
        onnx_op="ReduceMean",
        tflite_ops=["MEAN"],
        builder=lambda node, ctx: build_reduce_op(node, ctx, "MEAN"),
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "ReduceSum": DispatchEntry(
        onnx_op="ReduceSum",
        tflite_ops=["SUM"],
        builder=lambda node, ctx: build_reduce_op(node, ctx, "SUM"),
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "ReduceLogSum": DispatchEntry(
        onnx_op="ReduceLogSum",
        tflite_ops=["SUM", "LOG", "CAST"],
        builder=build_reduce_log_sum_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_reduce,
    ),
    "ReduceLogSumExp": DispatchEntry(
        onnx_op="ReduceLogSumExp",
        tflite_ops=["EXP", "SUM", "LOG", "CAST"],
        builder=build_reduce_log_sum_exp_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_reduce,
    ),
    "ReduceSumSquare": DispatchEntry(
        onnx_op="ReduceSumSquare",
        tflite_ops=["MUL", "SUM", "CAST"],
        builder=build_reduce_sum_square_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_reduce,
    ),
    "CumSum": DispatchEntry(
        onnx_op="CumSum",
        tflite_ops=["CUMSUM"],
        builder=build_cumsum_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_cumsum,
    ),
    "CumProd": DispatchEntry(
        onnx_op="CumProd",
        tflite_ops=[
            "RANGE",
            "LESS",
            "LESS_EQUAL",
            "RESHAPE",
            "TILE",
            "FILL",
            "SELECT_V2",
            "REDUCE_PROD",
            "REVERSE_V2",
        ],
        builder=build_cumprod_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_cumprod,
    ),
    "ReduceMax": DispatchEntry(
        onnx_op="ReduceMax",
        tflite_ops=["REDUCE_MAX"],
        builder=lambda node, ctx: build_reduce_op(node, ctx, "REDUCE_MAX"),
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "ReduceMin": DispatchEntry(
        onnx_op="ReduceMin",
        tflite_ops=["REDUCE_MIN"],
        builder=lambda node, ctx: build_reduce_op(node, ctx, "REDUCE_MIN"),
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "ReduceProd": DispatchEntry(
        onnx_op="ReduceProd",
        tflite_ops=["REDUCE_PROD"],
        builder=lambda node, ctx: build_reduce_op(node, ctx, "REDUCE_PROD"),
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "ReduceL1": DispatchEntry(
        onnx_op="ReduceL1",
        tflite_ops=["ABS", "SUM"],
        builder=build_reduce_l1_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "ReduceL2": DispatchEntry(
        onnx_op="ReduceL2",
        tflite_ops=["MUL", "SUM", "SQRT", "CAST"],
        builder=build_reduce_l2_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
    ),
    "And": DispatchEntry(
        onnx_op="And",
        tflite_ops=["LOGICAL_AND"],
        builder=_make_binary_builder("LOGICAL_AND"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Or": DispatchEntry(
        onnx_op="Or",
        tflite_ops=["LOGICAL_OR"],
        builder=_make_binary_builder("LOGICAL_OR"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Not": DispatchEntry(
        onnx_op="Not",
        tflite_ops=["LOGICAL_NOT"],
        builder=_make_unary_builder("LOGICAL_NOT"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Xor": DispatchEntry(
        onnx_op="Xor",
        tflite_ops=["NOT_EQUAL"],
        builder=_make_binary_builder("NOT_EQUAL"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Equal": DispatchEntry(
        onnx_op="Equal",
        tflite_ops=["EQUAL"],
        builder=_make_binary_builder("EQUAL"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Greater": DispatchEntry(
        onnx_op="Greater",
        tflite_ops=["GREATER"],
        builder=_make_binary_builder("GREATER"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "GreaterOrEqual": DispatchEntry(
        onnx_op="GreaterOrEqual",
        tflite_ops=["GREATER_EQUAL"],
        builder=_make_binary_builder("GREATER_EQUAL"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Less": DispatchEntry(
        onnx_op="Less",
        tflite_ops=["LESS"],
        builder=_make_binary_builder("LESS"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "LessOrEqual": DispatchEntry(
        onnx_op="LessOrEqual",
        tflite_ops=["LESS_EQUAL"],
        builder=_make_binary_builder("LESS_EQUAL"),
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Sigmoid": DispatchEntry(
        onnx_op="Sigmoid",
        tflite_ops=["LOGISTIC"],
        builder=build_logistic_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "HardSigmoid": DispatchEntry(
        onnx_op="HardSigmoid",
        tflite_ops=["MUL", "ADD", "MAXIMUM", "MINIMUM"],
        builder=build_hardsigmoid_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "HardSwish": DispatchEntry(
        onnx_op="HardSwish",
        tflite_ops=["HARD_SWISH"],
        builder=_make_unary_builder("HARD_SWISH"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Relu": DispatchEntry(
        onnx_op="Relu",
        tflite_ops=["RELU"],
        builder=_make_unary_builder("RELU"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "LeakyRelu": DispatchEntry(
        onnx_op="LeakyRelu",
        tflite_ops=["LEAKY_RELU"],
        builder=build_leaky_relu_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Elu": DispatchEntry(
        onnx_op="Elu",
        tflite_ops=["ELU"],
        builder=_make_unary_builder("ELU"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Celu": DispatchEntry(
        onnx_op="Celu",
        tflite_ops=["MAXIMUM", "MINIMUM", "DIV", "EXP", "SUB", "MUL", "ADD"],
        builder=build_celu_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Selu": DispatchEntry(
        onnx_op="Selu",
        tflite_ops=["MAXIMUM", "MINIMUM", "EXP", "SUB", "MUL", "ADD"],
        builder=build_selu_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Gelu": DispatchEntry(
        onnx_op="Gelu",
        tflite_ops=["GELU"],
        builder=_make_unary_builder("GELU"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Tanh": DispatchEntry(
        onnx_op="Tanh",
        tflite_ops=["TANH"],
        builder=_make_unary_builder("TANH"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Exp": DispatchEntry(
        onnx_op="Exp",
        tflite_ops=["EXP"],
        builder=_make_unary_builder("EXP"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Log": DispatchEntry(
        onnx_op="Log",
        tflite_ops=["LOG"],
        builder=_make_unary_builder("LOG"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Erf": DispatchEntry(
        onnx_op="Erf",
        tflite_ops=["ABS", "SIGN", "MUL", "ADD", "DIV", "EXP", "SUB"],
        builder=build_erf_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Cos": DispatchEntry(
        onnx_op="Cos",
        tflite_ops=["COS"],
        builder=_make_unary_builder("COS"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Sin": DispatchEntry(
        onnx_op="Sin",
        tflite_ops=["SIN"],
        builder=_make_unary_builder("SIN"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Tan": DispatchEntry(
        onnx_op="Tan",
        tflite_ops=["SIN", "COS", "DIV"],
        builder=build_tan_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Sinh": DispatchEntry(
        onnx_op="Sinh",
        tflite_ops=["SUB", "EXP", "MUL"],
        builder=build_sinh_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Cosh": DispatchEntry(
        onnx_op="Cosh",
        tflite_ops=["SUB", "EXP", "ADD", "MUL"],
        builder=build_cosh_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Softplus": DispatchEntry(
        onnx_op="Softplus",
        tflite_ops=["EXP", "ADD", "LOG"],
        builder=build_softplus_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Softsign": DispatchEntry(
        onnx_op="Softsign",
        tflite_ops=["ABS", "ADD", "DIV"],
        builder=build_softsign_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Mish": DispatchEntry(
        onnx_op="Mish",
        tflite_ops=["EXP", "ADD", "LOG", "TANH", "MUL"],
        builder=build_mish_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "ThresholdedRelu": DispatchEntry(
        onnx_op="ThresholdedRelu",
        tflite_ops=["GREATER", "CAST", "MUL"],
        builder=build_thresholded_relu_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "Shrink": DispatchEntry(
        onnx_op="Shrink",
        tflite_ops=["ADD", "SUB", "LESS", "GREATER", "SELECT_V2", "CAST"],
        builder=build_shrink_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_unary,
    ),
    "IsInf": DispatchEntry(
        onnx_op="IsInf",
        tflite_ops=["ABS", "EQUAL", "LESS", "GREATER", "LOGICAL_AND"],
        builder=build_is_inf_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_to_bool_unary,
    ),
    "IsNaN": DispatchEntry(
        onnx_op="IsNaN",
        tflite_ops=["NOT_EQUAL"],
        builder=build_is_nan_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_float_to_bool_unary,
    ),
    "Ceil": DispatchEntry(
        onnx_op="Ceil",
        tflite_ops=["CEIL"],
        builder=_make_unary_builder("CEIL"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Floor": DispatchEntry(
        onnx_op="Floor",
        tflite_ops=["FLOOR"],
        builder=_make_unary_builder("FLOOR"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Round": DispatchEntry(
        onnx_op="Round",
        tflite_ops=["ROUND"],
        builder=_make_unary_builder("ROUND"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Sign": DispatchEntry(
        onnx_op="Sign",
        tflite_ops=["SIGN"],
        builder=_make_unary_builder("SIGN"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Sqrt": DispatchEntry(
        onnx_op="Sqrt",
        tflite_ops=["SQRT"],
        builder=_make_unary_builder("SQRT"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Neg": DispatchEntry(
        onnx_op="Neg",
        tflite_ops=["NEG"],
        builder=_make_unary_builder("NEG"),
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Pow": DispatchEntry(
        onnx_op="Pow",
        tflite_ops=["POW"],
        builder=build_pow_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_pow,
    ),
    "Mean": DispatchEntry(
        onnx_op="Mean",
        tflite_ops=["ADD", "DIV", "CAST"],
        builder=build_mean_op,
        validation=ValidationSpec(min_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_mean,
    ),
    "Det": DispatchEntry(
        onnx_op="Det",
        tflite_ops=["GATHER", "MUL", "SUB", "ADD"],
        builder=build_det_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_det,
    ),
    "Pad": DispatchEntry(
        onnx_op="Pad",
        tflite_ops=[
            "PAD",
            "PADV2",
            "MIRROR_PAD",
            "STRIDED_SLICE",
            "TILE",
            "CONCATENATION",
        ],
        builder=build_pad_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=3, min_outputs=1, max_outputs=1),
    ),
    "DFT": DispatchEntry(
        onnx_op="DFT",
        tflite_ops=["RESHAPE", "BATCH_MATMUL", "CONCATENATION", "CAST"],
        builder=build_dft_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_dft,
    ),
    "PRelu": DispatchEntry(
        onnx_op="PRelu",
        tflite_ops=["PRELU"],
        builder=build_prelu_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_prelu,
    ),
    "Clip": DispatchEntry(
        onnx_op="Clip",
        tflite_ops=["RELU", "RELU6", "RELU_N1_TO_1", "MAXIMUM", "MINIMUM"],
        builder=build_clip_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_clip,
    ),
    "LpNormalization": DispatchEntry(
        onnx_op="LpNormalization",
        tflite_ops=["L2_NORMALIZATION"],
        builder=build_l2_normalization_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_l2_norm,
    ),
    "LRN": DispatchEntry(
        onnx_op="LRN",
        tflite_ops=["LOCAL_RESPONSE_NORMALIZATION"],
        builder=build_lrn_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_lrn,
    ),
    "Softmax": DispatchEntry(
        onnx_op="Softmax",
        tflite_ops=["SOFTMAX"],
        builder=build_softmax_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_softmax,
    ),
    "LogSoftmax": DispatchEntry(
        onnx_op="LogSoftmax",
        tflite_ops=["SOFTMAX", "LOG"],
        builder=build_logsoftmax_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_softmax,
    ),
    "NegativeLogLikelihoodLoss": DispatchEntry(
        onnx_op="NegativeLogLikelihoodLoss",
        tflite_ops=[
            "CAST",
            "TRANSPOSE",
            "EQUAL",
            "SELECT_V2",
            "ONE_HOT",
            "MUL",
            "SUM",
            "SUB",
            "GATHER",
            "MEAN",
            "GREATER",
            "DIV",
        ],
        builder=build_negative_log_likelihood_loss_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_negative_log_likelihood_loss,
    ),
    "SoftmaxCrossEntropyLoss": DispatchEntry(
        onnx_op="SoftmaxCrossEntropyLoss",
        tflite_ops=[
            "TRANSPOSE",
            "SOFTMAX",
            "LOG",
            "CAST",
            "EQUAL",
            "SELECT_V2",
            "ONE_HOT",
            "MUL",
            "SUM",
            "SUB",
            "GATHER",
            "MEAN",
            "GREATER",
            "DIV",
        ],
        builder=build_softmax_cross_entropy_loss_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=2),
        extra_validator=_validate_softmax_cross_entropy_loss,
    ),
    "Where": DispatchEntry(
        onnx_op="Where",
        tflite_ops=["CAST", "SELECT", "SELECT_V2"],
        builder=build_where_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_where,
    ),
    "Shape": DispatchEntry(
        onnx_op="Shape",
        tflite_ops=["SHAPE", "SLICE"],
        builder=build_shape_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_shape,
    ),
    "Size": DispatchEntry(
        onnx_op="Size",
        tflite_ops=["SHAPE", "REDUCE_PROD", "CAST"],
        builder=build_size_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "STFT": DispatchEntry(
        onnx_op="STFT",
        tflite_ops=["SLICE", "MUL", "RESHAPE", "BATCH_MATMUL", "CONCATENATION", "CAST"],
        builder=build_stft_op,
        validation=ValidationSpec(min_inputs=4, max_inputs=4, min_outputs=1, max_outputs=1),
        extra_validator=_validate_stft,
    ),
    "Range": DispatchEntry(
        onnx_op="Range",
        tflite_ops=["CAST", "SQUEEZE", "RANGE"],
        builder=build_range_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_range,
    ),
    "ReverseSequence": DispatchEntry(
        onnx_op="ReverseSequence",
        tflite_ops=["CAST", "REVERSE_SEQUENCE"],
        builder=build_reverse_sequence_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reverse_sequence,
    ),
    "Compress": DispatchEntry(
        onnx_op="Compress",
        tflite_ops=["NOT_EQUAL", "WHERE", "RESHAPE", "CAST", "GATHER"],
        builder=build_compress_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_compress,
    ),
    "CenterCropPad": DispatchEntry(
        onnx_op="CenterCropPad",
        tflite_ops=["SLICE", "PAD", "RESHAPE"],
        builder=build_center_crop_pad_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_center_crop_pad,
    ),
    "RandomNormalLike": DispatchEntry(
        onnx_op="RandomNormalLike",
        tflite_ops=["SHAPE", "RANDOM_STANDARD_NORMAL", "MUL", "ADD", "CAST"],
        builder=build_random_normal_like_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_random_normal_like,
    ),
    "RandomNormal": DispatchEntry(
        onnx_op="RandomNormal",
        tflite_ops=["RANDOM_STANDARD_NORMAL", "MUL", "ADD", "CAST"],
        builder=build_random_normal_op,
        validation=ValidationSpec(
            min_inputs=0,
            max_inputs=0,
            min_outputs=1,
            max_outputs=1,
            required_attrs=["shape"],
        ),
        extra_validator=_validate_random_normal,
    ),
    "RandomUniform": DispatchEntry(
        onnx_op="RandomUniform",
        tflite_ops=["RANDOM_UNIFORM", "MUL", "ADD", "CAST"],
        builder=build_random_uniform_op,
        validation=ValidationSpec(
            min_inputs=0,
            max_inputs=0,
            min_outputs=1,
            max_outputs=1,
            required_attrs=["shape"],
        ),
        extra_validator=_validate_random_uniform,
    ),
    "RandomUniformLike": DispatchEntry(
        onnx_op="RandomUniformLike",
        tflite_ops=["SHAPE", "RANDOM_UNIFORM", "MUL", "ADD", "CAST"],
        builder=build_random_uniform_like_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_random_uniform_like,
    ),
    "Bernoulli": DispatchEntry(
        onnx_op="Bernoulli",
        tflite_ops=["SHAPE", "RANDOM_UNIFORM", "LESS", "CAST"],
        builder=build_bernoulli_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_bernoulli,
    ),
    "BlackmanWindow": DispatchEntry(
        onnx_op="BlackmanWindow",
        tflite_ops=["CAST", "SQUEEZE", "RANGE", "MUL", "DIV", "COS", "SUB", "ADD", "MAXIMUM"],
        builder=build_blackman_window_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_window_op,
    ),
    "HammingWindow": DispatchEntry(
        onnx_op="HammingWindow",
        tflite_ops=["CAST", "SQUEEZE", "RANGE", "MUL", "DIV", "COS", "SUB", "MAXIMUM"],
        builder=build_hamming_window_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_window_op,
    ),
    "HannWindow": DispatchEntry(
        onnx_op="HannWindow",
        tflite_ops=["CAST", "SQUEEZE", "RANGE", "MUL", "DIV", "COS", "SUB", "MAXIMUM"],
        builder=build_hann_window_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_window_op,
    ),
    "MelWeightMatrix": DispatchEntry(
        onnx_op="MelWeightMatrix",
        tflite_ops=[],
        builder=build_mel_weight_matrix_op,
        validation=ValidationSpec(min_inputs=5, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_mel_weight_matrix,
    ),
    "EyeLike": DispatchEntry(
        onnx_op="EyeLike",
        tflite_ops=["RESHAPE"],
        builder=build_eyelike_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_eyelike,
    ),
    "Reshape": DispatchEntry(
        onnx_op="Reshape",
        tflite_ops=["RESHAPE"],
        builder=build_reshape_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reshape,
    ),
    "ConstantOfShape": DispatchEntry(
        onnx_op="ConstantOfShape",
        tflite_ops=["CAST", "FILL"],
        builder=build_constant_of_shape_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_constant_of_shape,
    ),
    "Flatten": DispatchEntry(
        onnx_op="Flatten",
        tflite_ops=["RESHAPE"],
        builder=build_flatten_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_flatten,
    ),
    "Dropout": DispatchEntry(
        onnx_op="Dropout",
        tflite_ops=["RESHAPE", "SHAPE", "FILL"],
        builder=build_dropout_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=3, min_outputs=1, max_outputs=2),
    ),
    "OptionalHasElement": DispatchEntry(
        onnx_op="OptionalHasElement",
        tflite_ops=["CONST"],
        builder=build_optional_has_element_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "StringNormalizer": DispatchEntry(
        onnx_op="StringNormalizer",
        tflite_ops=[
            "RESHAPE",
            "EQUAL",
            "LOGICAL_OR",
            "LOGICAL_NOT",
            "WHERE",
            "GATHER",
            "EXPAND_DIMS",
        ],
        builder=build_string_normalizer_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_string_normalizer,
    ),
    "If": DispatchEntry(
        onnx_op="If",
        tflite_ops=[
            "CONCATENATION",
            "REDUCE_MAX",
            "CAST",
            "ADD",
            "MUL",
            "RESHAPE",
            "NON_MAX_SUPPRESSION_V4",
            "NON_MAX_SUPPRESSION_V5",
            "SLICE",
            "GATHER",
            "SHAPE",
            "SUB",
            "SELECT",
            "SELECT_V2",
        ],
        builder=build_if_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=None),
        extra_validator=_validate_if,
    ),
    "Loop": DispatchEntry(
        onnx_op="Loop",
        tflite_ops=["RESHAPE"],
        builder=build_loop_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=None, min_outputs=1, max_outputs=None),
        extra_validator=_validate_loop,
    ),
    "Transpose": DispatchEntry(
        onnx_op="Transpose",
        tflite_ops=["TRANSPOSE"],
        builder=build_transpose_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_transpose,
    ),
    "Squeeze": DispatchEntry(
        onnx_op="Squeeze",
        tflite_ops=["SQUEEZE"],
        builder=build_squeeze_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_squeeze,
    ),
    "Unsqueeze": DispatchEntry(
        onnx_op="Unsqueeze",
        tflite_ops=["RESHAPE"],
        builder=build_unsqueeze_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_unsqueeze,
    ),
    "Trilu": DispatchEntry(
        onnx_op="Trilu",
        tflite_ops=["MUL", "LOGICAL_AND"],
        builder=build_trilu_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_trilu,
    ),
    "Concat": DispatchEntry(
        onnx_op="Concat",
        tflite_ops=["CONCATENATION", "CAST", "RESHAPE"],
        builder=build_concat_op,
        validation=ValidationSpec(min_inputs=2, min_outputs=1, max_outputs=1),
    ),
    "Gather": DispatchEntry(
        onnx_op="Gather",
        tflite_ops=["GATHER"],
        builder=build_gather_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_gather,
    ),
    "GatherND": DispatchEntry(
        onnx_op="GatherND",
        tflite_ops=["CAST", "RESHAPE", "RANGE", "TILE", "CONCATENATION", "GATHER_ND"],
        builder=build_gather_nd_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_gather_nd,
    ),
    "ScatterND": DispatchEntry(
        onnx_op="ScatterND",
        tflite_ops=["CAST", "SHAPE", "FILL", "MUL", "SCATTER_ND", "SUB", "ADD"],
        builder=build_scatter_nd_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_scatter_nd,
    ),
    "Unique": DispatchEntry(
        onnx_op="Unique",
        tflite_ops=[
            "CAST",
            "RESHAPE",
            "GATHER",
            "REDUCE_MIN",
            "REDUCE_MAX",
            "SUB",
            "MUL",
            "ADD",
            "DIV",
            "FLOOR_MOD",
            "UNIQUE",
            "CONCATENATION",
        ],
        builder=build_unique_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=4),
        extra_validator=_validate_unique,
    ),
    "ScatterElements": DispatchEntry(
        onnx_op="ScatterElements",
        tflite_ops=[
            "CAST",
            "LESS",
            "SELECT",
            "SHAPE",
            "GATHER",
            "RANGE",
            "RESHAPE",
            "TILE",
            "CONCATENATION",
            "MUL",
            "ADD",
            "SUB",
            "FILL",
            "SCATTER_ND",
        ],
        builder=build_scatter_elements_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_scatter_elements,
    ),
    "Scatter": DispatchEntry(
        onnx_op="Scatter",
        tflite_ops=[
            "CAST",
            "LESS",
            "SELECT",
            "SHAPE",
            "GATHER",
            "RANGE",
            "RESHAPE",
            "TILE",
            "CONCATENATION",
            "MUL",
            "ADD",
            "SUB",
            "FILL",
            "SCATTER_ND",
        ],
        builder=build_scatter_elements_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_scatter_elements,
    ),
    "TensorScatter": DispatchEntry(
        onnx_op="TensorScatter",
        tflite_ops=[
            "CAST",
            "GATHER",
            "RESHAPE",
            "ADD",
            "FLOOR_MOD",
            "CONCATENATION",
            "FILL",
            "MUL",
            "SUB",
            "SCATTER_ND",
        ],
        builder=build_tensor_scatter_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_tensor_scatter,
    ),
    "GatherElements": DispatchEntry(
        onnx_op="GatherElements",
        tflite_ops=["CAST", "RESHAPE", "CONCATENATION", "GATHER_ND"],
        builder=build_gather_elements_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_gather_elements,
    ),
    "NonMaxSuppression": DispatchEntry(
        onnx_op="NonMaxSuppression",
        tflite_ops=[
            "ARG_MAX",
            "REDUCE_MAX",
            "SQUEEZE",
            "NON_MAX_SUPPRESSION_V4",
            "NON_MAX_SUPPRESSION_V5",
            "SLICE",
            "RANGE",
            "SHAPE",
            "GATHER",
            "SUB",
            "CAST",
            "RESHAPE",
            "CONCATENATION",
        ],
        builder=build_non_max_suppression_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_non_max_suppression,
    ),
    "OneHot": DispatchEntry(
        onnx_op="OneHot",
        tflite_ops=["CAST", "ADD", "FLOOR_MOD", "ONE_HOT"],
        builder=build_one_hot_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_onehot,
    ),
    "ArgMax": DispatchEntry(
        onnx_op="ArgMax",
        tflite_ops=["ARG_MAX", "RESHAPE"],
        builder=build_argmax_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_argmax,
    ),
    "ArgMin": DispatchEntry(
        onnx_op="ArgMin",
        tflite_ops=["ARG_MIN", "RESHAPE"],
        builder=build_argmin_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_argmin,
    ),
    "TopK": DispatchEntry(
        onnx_op="TopK",
        tflite_ops=["CAST", "SQUEEZE", "TRANSPOSE", "NEG", "TOPK_V2"],
        builder=build_topk_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=2),
        extra_validator=_validate_topk,
    ),
    "Hardmax": DispatchEntry(
        onnx_op="Hardmax",
        tflite_ops=["TRANSPOSE", "ARG_MAX", "ONE_HOT"],
        builder=build_hardmax_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_hardmax,
    ),
    "NonZero": DispatchEntry(
        onnx_op="NonZero",
        tflite_ops=["NOT_EQUAL", "WHERE", "TRANSPOSE", "CAST"],
        builder=build_nonzero_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_nonzero,
    ),
    "Slice": DispatchEntry(
        onnx_op="Slice",
        tflite_ops=["SLICE", "STRIDED_SLICE", "REVERSE_V2"],
        builder=build_slice_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=5, min_outputs=1, max_outputs=1),
        extra_validator=_validate_slice,
    ),
    "Split": DispatchEntry(
        onnx_op="Split",
        tflite_ops=["SLICE"],
        builder=build_split_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=None),
        extra_validator=_validate_split,
    ),
    "Identity": DispatchEntry(
        onnx_op="Identity",
        tflite_ops=["RESHAPE"],
        builder=build_identity_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
    ),
    "Resize": DispatchEntry(
        onnx_op="Resize",
        tflite_ops=[
            "RESIZE_NEAREST_NEIGHBOR",
            "RESIZE_BILINEAR",
            "RESHAPE",
            "TRANSPOSE",
            "SHAPE",
            "SLICE",
            "MUL",
            "CAST",
            "FLOOR",
        ],
        builder=build_resize_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=4, min_outputs=1, max_outputs=1),
        extra_validator=_validate_resize,
    ),
    "Upsample": DispatchEntry(
        onnx_op="Upsample",
        tflite_ops=[
            "RESIZE_NEAREST_NEIGHBOR",
            "RESIZE_BILINEAR",
            "SHAPE",
            "SLICE",
            "MUL",
            "CAST",
            "FLOOR",
        ],
        builder=build_resize_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_resize,
    ),
    "GridSample": DispatchEntry(
        onnx_op="GridSample",
        tflite_ops=[
            "PAD",
            "RESHAPE",
            "TRANSPOSE",
            "SQUEEZE",
            "SLICE",
            "ADD",
            "SUB",
            "MUL",
            "FLOOR",
            "ROUND",
            "MAXIMUM",
            "MINIMUM",
            "CAST",
            "NOT_EQUAL",
            "SELECT_V2",
            "GATHER",
        ],
        builder=build_grid_sample_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_grid_sample,
    ),
    "AffineGrid": DispatchEntry(
        onnx_op="AffineGrid",
        tflite_ops=["BATCH_MATMUL", "TRANSPOSE", "RESHAPE"],
        builder=build_affine_grid_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_affine_grid,
    ),
    "RoiAlign": DispatchEntry(
        onnx_op="RoiAlign",
        tflite_ops=[
            "CAST",
            "GATHER",
            "PAD",
            "RESHAPE",
            "ADD",
            "SUB",
            "MUL",
            "DIV",
            "MAXIMUM",
            "MINIMUM",
            "FLOOR",
            "TILE",
            "AVERAGE_POOL_2D",
            "MAX_POOL_2D",
            "TRANSPOSE",
        ],
        builder=build_roi_align_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_roi_align,
    ),
    "RotaryEmbedding": DispatchEntry(
        onnx_op="RotaryEmbedding",
        tflite_ops=["TRANSPOSE", "SLICE", "RESHAPE", "CAST", "MUL", "SUB", "ADD", "CONCATENATION"],
        builder=build_rotary_embedding_op,
        validation=ValidationSpec(
            min_inputs=3,
            max_inputs=4,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [4], 1: [2], 2: [2], 3: [1, 2]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_rotary_embedding,
    ),
    "DeformConv": DispatchEntry(
        onnx_op="DeformConv",
        tflite_ops=[
            "PAD",
            "RESHAPE",
            "TRANSPOSE",
            "SHAPE",
            "RANGE",
            "SQUEEZE",
            "GATHER",
            "FLOOR",
            "MAXIMUM",
            "MINIMUM",
            "CAST",
            "MUL",
            "ADD",
            "SUB",
            "BATCH_MATMUL",
            "GREATER_EQUAL",
            "LESS_EQUAL",
            "LOGICAL_AND",
        ],
        builder=build_deform_conv_op,
        validation=ValidationSpec(
            min_inputs=3,
            max_inputs=5,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [4], 2: [4]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_deform_conv,
    ),
    "SpaceToDepth": DispatchEntry(
        onnx_op="SpaceToDepth",
        tflite_ops=["SPACE_TO_DEPTH"],
        builder=build_space_to_depth_op,
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [4]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_space_to_depth,
    ),
    "DepthToSpace": DispatchEntry(
        onnx_op="DepthToSpace",
        tflite_ops=["DEPTH_TO_SPACE"],
        builder=build_depth_to_space_op,
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [4]},
        ),
        extra_validator=_validate_depth_to_space,
    ),
    "Conv": DispatchEntry(
        onnx_op="Conv",
        tflite_ops=[
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "CONV_3D",
            "SLICE",
            "CONCATENATION",
        ],
        builder=build_conv2d_or_depthwise_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [1, 2, 3, 4, 5]},
            output_rank={0: [1, 2, 3, 4, 5]},
        ),
        extra_validator=_validate_conv,
    ),
    "FusedConv": DispatchEntry(
        onnx_op="FusedConv",
        tflite_ops=[
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "RELU",
            "RELU6",
            "RELU_N1_TO_1",
            "TANH",
            "LOGISTIC",
            "LEAKY_RELU",
            "MUL",
            "ADD",
            "MAXIMUM",
            "MINIMUM",
        ],
        builder=build_fused_conv_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [1, 2, 3, 4]},
            output_rank={0: [1, 2, 3, 4]},
        ),
        extra_validator=_validate_fused_conv,
    ),
    "ConvTranspose": DispatchEntry(
        onnx_op="ConvTranspose",
        tflite_ops=["TRANSPOSE_CONV", "ADD", "CONV_3D_TRANSPOSE"],
        builder=build_conv_transpose_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_conv_transpose,
    ),
    "Col2Im": DispatchEntry(
        onnx_op="Col2Im",
        tflite_ops=["RESHAPE", "TRANSPOSE", "TRANSPOSE_CONV", "SLICE", "CAST"],
        builder=build_col2im_op,
        validation=ValidationSpec(
            min_inputs=3,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [3], 1: [1], 2: [1]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_col2im,
    ),
    "GlobalAveragePool": DispatchEntry(
        onnx_op="GlobalAveragePool",
        tflite_ops=["MEAN"],
        builder=build_global_average_pool_op,
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_global_average_pool,
    ),
    "GlobalMaxPool": DispatchEntry(
        onnx_op="GlobalMaxPool",
        tflite_ops=["REDUCE_MAX"],
        builder=build_global_max_pool_op,
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_global_max_pool,
    ),
    "GlobalLpPool": DispatchEntry(
        onnx_op="GlobalLpPool",
        tflite_ops=["ABS", "POW", "SUM", "RESHAPE", "CAST"],
        builder=build_global_lp_pool_op,
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_global_lp_pool,
    ),
    "AveragePool": DispatchEntry(
        onnx_op="AveragePool",
        tflite_ops=["AVERAGE_POOL_2D"],
        builder=lambda node, ctx: build_pool2d_op(node, ctx, "AVERAGE_POOL_2D"),
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
            required_attrs=["kernel_shape"],
            input_rank={0: [3, 4, 5]},
            # Pool output rank is determined by the input rank. Some exported
            # ONNX graphs leave a stale placeholder rank on the output even
            # though the input and attributes are complete. The lowerer
            # materializes the canonical output shape before serialization.
        ),
        extra_validator=_validate_pool,
    ),
    "LpPool": DispatchEntry(
        onnx_op="LpPool",
        tflite_ops=["ABS", "POW", "AVERAGE_POOL_2D", "MUL", "RESHAPE", "CAST"],
        builder=build_lp_pool_op,
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=1,
            required_attrs=["kernel_shape"],
            input_rank={0: [4]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_lp_pool,
    ),
    "MaxRoiPool": DispatchEntry(
        onnx_op="MaxRoiPool",
        tflite_ops=["TRANSPOSE", "SLICE", "MAX_POOL_2D", "CONCATENATION"],
        builder=build_max_roi_pool_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_max_roi_pool,
    ),
    "MaxPool": DispatchEntry(
        onnx_op="MaxPool",
        tflite_ops=["MAX_POOL_2D"],
        builder=lambda node, ctx: build_pool2d_op(node, ctx, "MAX_POOL_2D"),
        validation=ValidationSpec(
            min_inputs=1,
            max_inputs=1,
            min_outputs=1,
            max_outputs=2,
            required_attrs=["kernel_shape"],
            input_rank={0: [3, 4, 5]},
            output_rank={0: [1, 3, 4, 5], 1: [1, 3, 4, 5]},
        ),
        extra_validator=_validate_pool,
    ),
    "MaxUnpool": DispatchEntry(
        onnx_op="MaxUnpool",
        tflite_ops=["CAST", "RESHAPE", "SCATTER_ND"],
        builder=build_max_unpool_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
            required_attrs=["kernel_shape"],
            input_rank={0: [4], 1: [4], 2: [1]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_max_unpool,
    ),
    "Gemm": DispatchEntry(
        onnx_op="Gemm",
        tflite_ops=["FULLY_CONNECTED", "BATCH_MATMUL", "MUL", "ADD", "CAST"],
        builder=build_fully_connected_from_gemm_or_matmul,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_fc,
    ),
    "MatMul": DispatchEntry(
        onnx_op="MatMul",
        tflite_ops=["BATCH_MATMUL", "RESHAPE", "SQUEEZE", "CAST"],
        builder=build_matmul_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_matmul,
    ),
    "MultiHeadAttention": DispatchEntry(
        onnx_op="MultiHeadAttention",
        tflite_ops=["RESHAPE", "TRANSPOSE", "BATCH_MATMUL", "MUL", "SOFTMAX", "CAST"],
        builder=build_multi_head_attention_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_multi_head_attention,
    ),
    "Attention": DispatchEntry(
        onnx_op="Attention",
        tflite_ops=["RESHAPE", "TRANSPOSE", "BATCH_MATMUL", "MUL", "SOFTMAX", "CAST"],
        builder=build_attention_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=7, min_outputs=1, max_outputs=1),
        extra_validator=_validate_attention,
    ),
    "FusedMatMul": DispatchEntry(
        onnx_op="FusedMatMul",
        tflite_ops=["BATCH_MATMUL", "MUL"],
        builder=build_fused_matmul_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_fused_matmul,
    ),
    "MatMulInteger": DispatchEntry(
        onnx_op="MatMulInteger",
        tflite_ops=["CAST", "SUB", "BATCH_MATMUL"],
        builder=build_matmul_integer_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=4, min_outputs=1, max_outputs=1),
        extra_validator=_validate_matmul_integer,
    ),
    "GRU": DispatchEntry(
        onnx_op="GRU",
        tflite_ops=[
            "TRANSPOSE",
            "SLICE",
            "SQUEEZE",
            "BATCH_MATMUL",
            "ADD",
            "MUL",
            "SUB",
            "LOGISTIC",
            "TANH",
            "RESHAPE",
            "CONCATENATION",
            "EXPAND_DIMS",
        ],
        builder=build_gru_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=None, min_outputs=1, max_outputs=2),
        extra_validator=_validate_gru,
    ),
    "RNN": DispatchEntry(
        onnx_op="RNN",
        tflite_ops=[
            "UNIDIRECTIONAL_SEQUENCE_RNN",
            "REVERSE_V2",
            "CONCATENATION",
            "TRANSPOSE",
            "EXPAND_DIMS",
            "SLICE",
            "SQUEEZE",
            "RESHAPE",
        ],
        builder=build_rnn_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=None, min_outputs=1, max_outputs=2),
        extra_validator=_validate_rnn,
    ),
    "LSTM": DispatchEntry(
        onnx_op="LSTM",
        tflite_ops=[
            "UNIDIRECTIONAL_SEQUENCE_LSTM",
            "BIDIRECTIONAL_SEQUENCE_LSTM",
            "REVERSE_V2",
            "SPLIT",
            "EXPAND_DIMS",
            "RESHAPE",
            "CONCATENATION",
            "BATCH_MATMUL",
            "SLICE",
            "ADD",
            "MUL",
            "LOGISTIC",
            "TANH",
        ],
        builder=build_lstm_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=None, min_outputs=1, max_outputs=3),
        extra_validator=_validate_lstm,
    ),
    "Einsum": DispatchEntry(
        onnx_op="Einsum",
        tflite_ops=[
            "FULLY_CONNECTED",
            "BATCH_MATMUL",
            "CAST",
            "TRANSPOSE",
            "RESHAPE",
            "EXPAND_DIMS",
            "MUL",
        ],
        builder=build_einsum_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_einsum,
    ),
}


def get_dispatch_registry() -> Dict[str, DispatchEntry]:
    return dict(_DISPATCH_REGISTRY)


def get_dispatch_entry(onnx_op: str) -> Optional[DispatchEntry]:
    return _DISPATCH_REGISTRY.get(str(onnx_op))


def get_supported_onnx_ops() -> List[str]:
    return sorted(_DISPATCH_REGISTRY.keys())


def get_custom_op_candidate_ops() -> List[str]:
    return sorted(list(_CUSTOM_OP_CANDIDATES))


def resolve_node_dispatch(node: Any, ctx: Any) -> DispatchResolution:
    entry = get_dispatch_entry(node.op)
    if entry is None:
        custom_resolution = _resolve_custom_candidate(node, ctx)
        if custom_resolution is not None:
            entry = custom_resolution.entry
            _validate_counts(node, entry.validation)
            _validate_attrs(node, entry.validation)
            _validate_rank_constraints(node, ctx, entry.validation)
            return custom_resolution
        generic_custom_resolution = _resolve_generic_custom_fallback(node, ctx)
        if generic_custom_resolution is not None:
            return generic_custom_resolution
        raise NodeValidationError(
            reason_code="unsupported_onnx_op",
            message=f"ONNX op is not supported by flatbuffer_direct: {node.op}",
            node_name=node.name,
            node_op=node.op,
        )
    try:
        _validate_counts(node, entry.validation)
        _validate_attrs(node, entry.validation)
        _validate_rank_constraints(node, ctx, entry.validation)
        if entry.extra_validator is not None:
            entry.extra_validator(node, ctx)
        return DispatchResolution(
            entry=entry,
            dispatch_mode="builtin",
        )
    except NodeValidationError as ve:
        if str(node.op) in _CUSTOM_OP_CANDIDATES:
            custom_resolution = _resolve_custom_candidate(node, ctx)
            if custom_resolution is not None:
                return custom_resolution
        generic_custom_resolution = _resolve_generic_custom_fallback(node, ctx)
        if generic_custom_resolution is not None:
            return generic_custom_resolution
        raise ve


def validate_node_support(node: Any, ctx: Any) -> DispatchEntry:
    return resolve_node_dispatch(node, ctx).entry
