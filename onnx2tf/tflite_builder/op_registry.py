from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.op_builders import (
    build_abs_op,
    build_acos_op,
    build_acosh_op,
    build_argmin_op,
    build_argmax_op,
    build_asin_op,
    build_asinh_op,
    build_atan_op,
    build_atanh_op,
    build_batch_normalization_op,
    build_instance_normalization_op,
    build_binary_op,
    build_bitshift_op,
    build_bitwise_not_op,
    build_cast_op,
    build_celu_op,
    build_clip_op,
    build_col2im_op,
    build_concat_op,
    build_constant_of_shape_op,
    build_conv2d_or_depthwise_op,
    build_conv_transpose_op,
    build_fused_conv_op,
    build_dropout_op,
    build_cosh_op,
    build_custom_passthrough_op,
    build_conv_integer_op,
    build_dequantize_linear_op,
    build_depth_to_space_op,
    build_dynamic_quantize_linear_op,
    build_div_op,
    build_einsum_op,
    build_erf_op,
    build_eyelike_op,
    build_expand_op,
    build_flatten_op,
    build_grid_sample_op,
    build_fused_matmul_op,
    build_fully_connected_from_gemm_or_matmul,
    build_gru_op,
    build_multi_head_attention_op,
    build_matmul_op,
    build_gather_op,
    build_gather_nd_op,
    build_gather_elements_op,
    build_hardmax_op,
    build_roi_align_op,
    build_scatter_elements_op,
    build_scatter_nd_op,
    build_non_max_suppression_op,
    build_hardsigmoid_op,
    build_global_average_pool_op,
    build_global_max_pool_op,
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
    build_lrn_op,
    build_logistic_op,
    build_matmul_integer_op,
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
    build_random_normal_like_op,
    build_range_op,
    build_cumsum_op,
    build_reduce_l1_op,
    build_reduce_l2_op,
    build_reduce_op,
    build_reciprocal_op,
    build_resize_op,
    build_reshape_op,
    build_rnn_op,
    build_selu_op,
    build_shape_op,
    build_sinh_op,
    build_slice_op,
    build_split_op,
    build_space_to_depth_op,
    build_string_normalizer_op,
    build_squeeze_op,
    build_tile_op,
    build_softmax_op,
    build_softplus_op,
    build_softsign_op,
    build_tan_op,
    build_transpose_op,
    build_trilu_op,
    build_unary_op,
    build_unsqueeze_op,
    build_where_op,
    is_supported_if_axis0_add_branch_pattern,
    is_supported_if_nested_reducemin_add_branch_pattern,
    is_supported_if_nms_guard_pattern,
    is_supported_if_sequenceconstruct_add_branch_pattern,
    is_supported_loop_static_unroll_pattern,
    is_supported_loop_while_pattern,
)


class NodeValidationError(ValueError):
    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        node_name: str,
        node_op: str,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.node_name = str(node_name)
        self.node_op = str(node_op)
        self.message = str(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "onnx_op": self.node_op,
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationSpec:
    min_inputs: int = 0
    max_inputs: Optional[int] = None
    min_outputs: int = 1
    max_outputs: Optional[int] = 1
    required_attrs: List[str] = field(default_factory=list)
    input_rank: Dict[int, List[int]] = field(default_factory=dict)
    output_rank: Dict[int, List[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchEntry:
    onnx_op: str
    tflite_ops: List[str]
    builder: Callable[[Any, Any], None]
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    extra_validator: Optional[Callable[[Any, Any], None]] = None


@dataclass(frozen=True)
class DispatchResolution:
    entry: DispatchEntry
    dispatch_mode: str
    reason_code: Optional[str] = None
    message: Optional[str] = None


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


def _validate_counts(node: Any, spec: ValidationSpec) -> None:
    input_count = len(node.inputs)
    output_count = len(node.outputs)
    if input_count < int(spec.min_inputs):
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=f"input_count={input_count} is smaller than min_inputs={spec.min_inputs}",
            node_name=node.name,
            node_op=node.op,
        )
    if spec.max_inputs is not None and input_count > int(spec.max_inputs):
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=f"input_count={input_count} exceeds max_inputs={spec.max_inputs}",
            node_name=node.name,
            node_op=node.op,
        )
    if output_count < int(spec.min_outputs):
        raise NodeValidationError(
            reason_code="invalid_output_count",
            message=f"output_count={output_count} is smaller than min_outputs={spec.min_outputs}",
            node_name=node.name,
            node_op=node.op,
        )
    if spec.max_outputs is not None and output_count > int(spec.max_outputs):
        raise NodeValidationError(
            reason_code="invalid_output_count",
            message=f"output_count={output_count} exceeds max_outputs={spec.max_outputs}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_attrs(node: Any, spec: ValidationSpec) -> None:
    for attr in spec.required_attrs:
        if attr not in node.attrs:
            raise NodeValidationError(
                reason_code="missing_required_attribute",
                message=f"required attribute '{attr}' is missing",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_rank_constraints(node: Any, ctx: Any, spec: ValidationSpec) -> None:
    for input_index, allowed_ranks in spec.input_rank.items():
        if input_index >= len(node.inputs):
            continue
        tensor_name = node.inputs[input_index].name
        rank = len(ctx.get_tensor_shape(tensor_name))
        if rank not in allowed_ranks:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    f"input[{input_index}] rank={rank} is not in supported ranks={allowed_ranks} "
                    f"for tensor={tensor_name}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    for output_index, allowed_ranks in spec.output_rank.items():
        if output_index >= len(node.outputs):
            continue
        tensor_name = node.outputs[output_index].name
        rank = len(ctx.get_tensor_shape(tensor_name))
        if rank not in allowed_ranks:
            raise NodeValidationError(
                reason_code="unsupported_output_rank",
                message=(
                    f"output[{output_index}] rank={rank} is not in supported ranks={allowed_ranks} "
                    f"for tensor={tensor_name}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _require_const_input(node: Any, ctx: Any, input_index: int, input_label: str) -> np.ndarray:
    if input_index >= len(node.inputs):
        raise NodeValidationError(
            reason_code="missing_required_input",
            message=f"{input_label} input index={input_index} is missing",
            node_name=node.name,
            node_op=node.op,
        )
    tensor_name = node.inputs[input_index].name
    const_value = ctx.get_constant_array(tensor_name)
    if const_value is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message=f"{input_label} must be constant. tensor={tensor_name}",
            node_name=node.name,
            node_op=node.op,
        )
    return np.asarray(const_value)


def _get_original_node_inputs(node: Any, ctx: Any) -> List[str]:
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None:
        return [str(v.name) for v in node.inputs]
    for graph_node in onnx_model.graph.node:
        graph_node_name = str(graph_node.name) if str(graph_node.name) != "" else str(graph_node.op_type)
        if graph_node_name == str(node.name) and str(graph_node.op_type) == str(node.op):
            return [str(v) for v in graph_node.input]
    return [str(v.name) for v in node.inputs]


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


def _validate_lstm(node: Any, ctx: Any) -> None:
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "LSTM direction must be one of forward/reverse/bidirectional for builtin lowering. "
                f"direction={direction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    layout = int(node.attrs.get("layout", 0))
    if layout != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LSTM layout must be 0 (time-major). layout={layout}",
            node_name=node.name,
            node_op=node.op,
        )
    input_forget = int(node.attrs.get("input_forget", 0))
    if input_forget != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LSTM input_forget=1 is not supported in builtin lowering. input_forget={input_forget}",
            node_name=node.name,
            node_op=node.op,
        )

    original_inputs = _get_original_node_inputs(node, ctx)

    def _input_name(index: int) -> str:
        if index < 0 or index >= len(original_inputs):
            return ""
        return str(original_inputs[index])

    sequence_lens_name = _input_name(4)
    if sequence_lens_name != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="LSTM sequence_lens is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    peephole_name = _input_name(7)
    if peephole_name != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="LSTM peephole weights input P is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    w_name = _input_name(1)
    r_name = _input_name(2)
    if w_name == "" or r_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="LSTM requires W and R inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    W = ctx.get_constant_array(w_name)
    R = ctx.get_constant_array(r_name)
    if W is None or R is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="LSTM W and R must be constant tensors for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    W = np.asarray(W)
    R = np.asarray(R)
    if W.ndim != 3 or R.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"LSTM W/R must be rank-3. W_shape={list(W.shape)} R_shape={list(R.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM builtin lowering expects weights with num_directions matching direction. "
                f"direction={direction} expected_num_directions={expected_num_directions} "
                f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    hidden_size_attr = int(node.attrs.get("hidden_size", 0))
    hidden_size = int(hidden_size_attr) if hidden_size_attr > 0 else int(W.shape[1] // 4)
    if int(W.shape[1]) != 4 * hidden_size or int(R.shape[1]) != 4 * hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM hidden_size mismatch in W/R. "
                f"hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(R.shape[2]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM projection is not supported in builtin lowering. "
                f"recurrent_output_size={int(R.shape[2])} hidden_size={hidden_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    b_name = _input_name(3)
    if b_name != "":
        B = ctx.get_constant_array(b_name)
        if B is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="LSTM B must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        B = np.asarray(B)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 8 * hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "LSTM B must have shape [num_directions, 8*hidden_size]. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    initial_c_name = _input_name(6)
    if (initial_h_name == "") ^ (initial_c_name == ""):
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="LSTM initial_h and initial_c must be both present or both absent for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    if initial_h_name != "":
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        initial_c_shape = [int(v) for v in ctx.get_tensor_shape(initial_c_name)]
        if len(initial_h_shape) != 3 or len(initial_c_shape) != 3:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "LSTM initial_h and initial_c must be rank-3 with shape "
                    "[num_directions, batch, hidden]. "
                    f"initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if (
            int(initial_h_shape[0]) > 0 and int(initial_h_shape[0]) != expected_num_directions
        ) or (
            int(initial_c_shape[0]) > 0 and int(initial_c_shape[0]) != expected_num_directions
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "LSTM initial_h and initial_c first dimension must match num_directions. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if (
            int(initial_h_shape[2]) > 0 and int(initial_h_shape[2]) != hidden_size
        ) or (
            int(initial_c_shape[2]) > 0 and int(initial_c_shape[2]) != hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "LSTM initial_h and initial_c hidden dimension must match hidden_size. "
                    f"hidden_size={hidden_size} initial_h_shape={initial_h_shape} initial_c_shape={initial_c_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

def _validate_rnn(node: Any, ctx: Any) -> None:
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "RNN direction must be one of forward/reverse/bidirectional for builtin lowering. "
                f"direction={direction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    layout = int(node.attrs.get("layout", 0))
    if layout != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"RNN layout must be 0 (time-major). layout={layout}",
            node_name=node.name,
            node_op=node.op,
        )

    original_inputs = _get_original_node_inputs(node, ctx)

    def _input_name(index: int) -> str:
        if index < 0 or index >= len(original_inputs):
            return ""
        return str(original_inputs[index])

    if _input_name(4) != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="RNN sequence_lens is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    w_name = _input_name(1)
    r_name = _input_name(2)
    if w_name == "" or r_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="RNN requires W and R inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    W = ctx.get_constant_array(w_name)
    R = ctx.get_constant_array(r_name)
    if W is None or R is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="RNN W and R must be constant tensors for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    W = np.asarray(W)
    R = np.asarray(R)
    if W.ndim != 3 or R.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"RNN W/R must be rank-3. W_shape={list(W.shape)} R_shape={list(R.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RNN W/R num_directions mismatch for builtin lowering. "
                f"direction={direction} expected_num_directions={expected_num_directions} "
                f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    hidden_size = int(node.attrs.get("hidden_size", int(W.shape[1])))
    if int(W.shape[1]) != hidden_size or int(R.shape[1]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RNN hidden_size mismatch in W/R for builtin lowering. "
                f"hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(R.shape[2]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "RNN recurrent weight output size must match hidden_size for builtin lowering. "
                f"R_shape={list(R.shape)} hidden_size={hidden_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    b_name = _input_name(3)
    if b_name != "":
        B = ctx.get_constant_array(b_name)
        if B is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="RNN B must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        B = np.asarray(B)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 2 * hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN B must have shape [num_directions, 2*hidden_size]. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    if initial_h_name != "":
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        if len(initial_h_shape) != 3:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN initial_h must be rank-3 [num_directions, batch, hidden] in builtin lowering. "
                    f"initial_h_shape={initial_h_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if int(initial_h_shape[0]) > 0 and int(initial_h_shape[0]) != expected_num_directions:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN initial_h first dimension must match num_directions in builtin lowering. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"initial_h_shape={initial_h_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if int(initial_h_shape[2]) > 0 and int(initial_h_shape[2]) != hidden_size:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "RNN initial_h hidden dimension must match hidden_size in builtin lowering. "
                    f"initial_h_shape={initial_h_shape} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    activations = node.attrs.get("activations", ["Tanh"])
    if not isinstance(activations, (list, tuple)) or len(activations) == 0:
        activations = ["Tanh"]
    for dir_index in range(expected_num_directions):
        activation = str(activations[min(dir_index, len(activations) - 1)]).lower()
        if activation not in {"tanh", "relu", "sigmoid"}:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "RNN activation must be one of tanh/relu/sigmoid for builtin lowering. "
                    f"direction={direction} direction_index={dir_index} activation={activation}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    clip = float(node.attrs.get("clip", 0.0))
    if abs(clip) > 1e-6:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"RNN clip is not supported in builtin lowering. clip={clip}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_gru(node: Any, ctx: Any) -> None:
    direction = str(node.attrs.get("direction", "forward")).lower()
    if direction not in {"forward", "reverse", "bidirectional"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GRU direction must be one of forward/reverse/bidirectional for builtin lowering. "
                f"direction={direction}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    expected_num_directions = 2 if direction == "bidirectional" else 1
    layout = int(node.attrs.get("layout", 0))
    if layout != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GRU layout must be 0 (time-major). layout={layout}",
            node_name=node.name,
            node_op=node.op,
        )
    linear_before_reset = int(node.attrs.get("linear_before_reset", 0))
    if linear_before_reset not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GRU linear_before_reset must be 0 or 1 in flatbuffer_direct builtin lowering. "
                f"linear_before_reset={linear_before_reset}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    clip = float(node.attrs.get("clip", 0.0))
    if abs(clip) > 1e-6:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GRU clip is not supported in builtin lowering. clip={clip}",
            node_name=node.name,
            node_op=node.op,
        )
    activations = node.attrs.get("activations", ["Sigmoid", "Tanh"])
    if not isinstance(activations, (list, tuple)) or len(activations) < 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GRU activations must contain at least 2 entries. activations={activations}",
            node_name=node.name,
            node_op=node.op,
        )
    act0 = str(activations[0]).lower()
    act1 = str(activations[1]).lower()
    if act0 != "sigmoid" or act1 != "tanh":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GRU builtin lowering supports activations=[Sigmoid,Tanh] only. "
                f"activations={activations}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    original_inputs = _get_original_node_inputs(node, ctx)

    def _input_name(index: int) -> str:
        if index < 0 or index >= len(original_inputs):
            return ""
        return str(original_inputs[index])

    if _input_name(4) != "":
        raise NodeValidationError(
            reason_code="unsupported_input",
            message="GRU sequence_lens is not supported in flatbuffer_direct builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )

    w_name = _input_name(1)
    r_name = _input_name(2)
    if w_name == "" or r_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="GRU requires W and R inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    W = ctx.get_constant_array(w_name)
    R = ctx.get_constant_array(r_name)
    if W is None or R is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="GRU W and R must be constant tensors for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    W = np.asarray(W)
    R = np.asarray(R)
    if W.ndim != 3 or R.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"GRU W/R must be rank-3. W_shape={list(W.shape)} R_shape={list(R.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(W.shape[0]) != expected_num_directions or int(R.shape[0]) != expected_num_directions:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU W/R num_directions mismatch for builtin lowering. "
                f"direction={direction} expected_num_directions={expected_num_directions} "
                f"W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    hidden_size = int(node.attrs.get("hidden_size", int(W.shape[1] // 3)))
    if int(W.shape[1]) != 3 * hidden_size or int(R.shape[1]) != 3 * hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU hidden_size mismatch in W/R for builtin lowering. "
                f"hidden_size={hidden_size} W_shape={list(W.shape)} R_shape={list(R.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(R.shape[2]) != hidden_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU recurrent weight output size must match hidden_size for builtin lowering. "
                f"R_shape={list(R.shape)} hidden_size={hidden_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    x_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    if len(x_shape) != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GRU input rank must be 3 [seq,batch,input]. input_shape={x_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(x_shape[0]) <= 0 or int(x_shape[1]) <= 0 or int(x_shape[2]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU builtin lowering requires static positive seq,batch,input dimensions. "
                f"input_shape={x_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    input_size = int(W.shape[2])
    batch_dim = int(x_shape[1])
    if int(x_shape[2]) == input_size:
        batch_dim = int(x_shape[1])
    elif int(x_shape[1]) == input_size:
        batch_dim = int(x_shape[2])
    else:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GRU input size mismatch with W for builtin lowering. "
                "Expected [seq,batch,input] or [seq,input,batch]. "
                f"input_shape={x_shape} W_shape={list(W.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    b_name = _input_name(3)
    if b_name != "":
        B = ctx.get_constant_array(b_name)
        if B is None:
            raise NodeValidationError(
                reason_code="requires_constant_input",
                message="GRU B must be constant when provided.",
                node_name=node.name,
                node_op=node.op,
            )
        B = np.asarray(B)
        if (
            B.ndim != 2
            or int(B.shape[0]) != expected_num_directions
            or int(B.shape[1]) != 6 * hidden_size
        ):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GRU B must have shape [num_directions, 6*hidden_size]. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    if initial_h_name != "":
        initial_h_shape = [int(v) for v in ctx.get_tensor_shape(initial_h_name)]
        if initial_h_shape != [expected_num_directions, int(batch_dim), hidden_size]:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "GRU initial_h shape must be [num_directions, batch, hidden_size] "
                    "for builtin lowering. "
                    f"direction={direction} expected_num_directions={expected_num_directions} "
                    f"initial_h_shape={initial_h_shape} batch={int(batch_dim)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

def _validate_softmax(node: Any, ctx: Any) -> None:
    input_name = node.inputs[0].name
    input_shape = ctx.get_tensor_shape(input_name)
    rank = len(input_shape)
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
    normalized_axes = _normalize_axes_for_rank(axes=axes, rank=rank, node=node)
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

    if len(node.inputs) >= 5 and str(node.inputs[4].name) != "":
        steps_arr = _require_const_input(node, ctx, 4, "slice steps")
        steps = [int(v) for v in np.asarray(steps_arr).reshape(-1).tolist()]
    elif "steps" in node.attrs:
        attr_steps = node.attrs.get("steps")
        if isinstance(attr_steps, (list, tuple, np.ndarray)):
            steps = [int(v) for v in np.asarray(attr_steps).reshape(-1).tolist()]
        elif attr_steps is None:
            steps = [1 for _ in range(len(normalized_axes))]
        else:
            steps = [int(attr_steps)]
    else:
        steps = [1 for _ in range(len(normalized_axes))]

    if len(steps) != len(normalized_axes):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/steps length mismatch. "
                f"axes_len={len(normalized_axes)} steps_len={len(steps)}"
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

    if dynamic_end_name == "" and len(starts_values) != len(ends_values):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/ends length mismatch. "
                f"starts_len={len(starts_values)} ends_len={len(ends_values)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if dynamic_start_name != "" or (dynamic_end_name != "" and len(normalized_axes) == 1):
        is_single_axis_dynamic = (
            len(normalized_axes) == 1
            and len(steps) == 1
            and int(steps[0]) > 0
            and (
                dynamic_start_name != ""
                or (len(starts_values) == 1 and int(starts_values[0]) >= 0)
            )
        )
        if not is_single_axis_dynamic:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Slice dynamic starts/ends lowering supports single-axis slicing "
                    "with positive step only. "
                    f"rank={rank} starts={starts_values} ends={ends_values} "
                    f"axes={normalized_axes} steps={steps}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if dynamic_start_name != "":
            dynamic_start_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_start_name)]
            dynamic_start_len = int(dynamic_start_shape[0]) if len(dynamic_start_shape) == 1 else -1
            if not (len(dynamic_start_shape) == 1 and (dynamic_start_len <= 0 or dynamic_start_len == 1)):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Slice dynamic starts must be rank-1 length-1 (or unknown length) "
                        "for builtin lowering. "
                        f"shape={dynamic_start_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
        if dynamic_end_name != "":
            dynamic_end_shape = [int(v) for v in ctx.get_tensor_shape(dynamic_end_name)]
            dynamic_end_len = int(dynamic_end_shape[0]) if len(dynamic_end_shape) == 1 else -1
            if not (len(dynamic_end_shape) == 1 and (dynamic_end_len <= 0 or dynamic_end_len == 1)):
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Slice dynamic ends must be rank-1 length-1 (or unknown length) "
                        "for builtin lowering. "
                        f"shape={dynamic_end_shape}"
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
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
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
    if int(weights.ndim) == 4:
        if len(input_shape) != 4 or len(output_shape) != 4:
            raise NodeValidationError(
                reason_code="unsupported_tensor_rank",
                message=f"Conv2D input/output rank must be 4. input_shape={input_shape} output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
    elif int(weights.ndim) == 3:
        if len(input_shape) != 3 or len(output_shape) != 3:
            raise NodeValidationError(
                reason_code="unsupported_tensor_rank",
                message=f"Conv1D input/output rank must be 3. input_shape={input_shape} output_shape={output_shape}",
                node_name=node.name,
                node_op=node.op,
            )
    else:
        if len(input_shape) != 5 or len(output_shape) != 5:
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
    if int(weights.ndim) == 5 and group != 1:
        raise NodeValidationError(
            reason_code="unsupported_grouped_convolution",
            message=(
                "Conv3D currently supports group=1 only. "
                f"group={group} input_shape={input_shape} output_shape={output_shape} "
                f"weight_shape={list(weights.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    in_channels = int(input_shape[1])
    out_channels = int(weights.shape[0])
    weight_in_channels_per_group = int(weights.shape[1])
    is_depthwise = (
        group > 1
        and weight_in_channels_per_group == 1
        and (out_channels % group) == 0
    )
    if group != 1 and not is_depthwise:
        if int(weights.ndim) != 4 or len(input_shape) != 4 or len(output_shape) != 4:
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
    if input_shape != [1] and output_shape != [1]:
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
            if input_shape != [1] and len(input_shape) != 4:
                raise NodeValidationError(
                    reason_code="unsupported_tensor_rank",
                    message=(
                        "FusedConv2D input rank must be 4 (or unknown placeholder rank=1). "
                        f"input_shape={input_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if output_shape != [1] and len(output_shape) != 4:
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
            if input_shape != [1] and len(input_shape) != 3:
                raise NodeValidationError(
                    reason_code="unsupported_tensor_rank",
                    message=(
                        "FusedConv1D input rank must be 3 (or unknown placeholder rank=1). "
                        f"input_shape={input_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
            if output_shape != [1] and len(output_shape) != 3:
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
    if len(input_shape) < 3:
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
    if group != 1:
        raise NodeValidationError(
            reason_code="unsupported_grouped_convolution",
            message=f"ConvTranspose currently supports group=1 only. group={group}",
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
        if dilations != [1, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"ConvTranspose dilations must be [1,1]. dilations={dilations}",
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
            storage_order = int(node.attrs.get("storage_order", 0))
            kernel = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
            strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
            dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
            auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
            pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
            if len(pads) < 4:
                pads = [0, 0, 0, 0]
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
            if any(int(v) != 0 for v in pads):
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
    if not (is_standard_matmul or is_vector_rhs_matmul):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "MatMul input ranks must be (a_rank>=2,b_rank>=2) "
                "or vector-rhs form (a_rank>=2,b_rank=1). "
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
    if row_dim not in {2, 3}:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Inverse builtin lowering currently supports only last dims [2,2] or [3,3] "
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

def _extract_axes(
    *,
    node: Any,
    ctx: Any,
    input_index: int = 1,
    attr_name: str = "axes",
    default_if_missing: Optional[List[int]] = None,
) -> List[int]:
    axes: Optional[List[int]] = None
    if len(node.inputs) > input_index and str(node.inputs[input_index].name) != "":
        axes_arr = _require_const_input(node, ctx, input_index, f"{node.op} axes")
        axes = [int(v) for v in np.asarray(axes_arr).reshape(-1).tolist()]
    elif attr_name in node.attrs:
        attr_axes = node.attrs.get(attr_name)
        if isinstance(attr_axes, (list, tuple)):
            axes = [int(v) for v in attr_axes]
        elif attr_axes is None:
            axes = []
        else:
            axes = [int(attr_axes)]
    if axes is None:
        axes = [] if default_if_missing is None else [int(v) for v in default_if_missing]
    return [int(v) for v in axes]


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


def _normalize_axes_for_rank(
    *,
    axes: List[int],
    rank: int,
    node: Any,
) -> List[int]:
    normalized: List[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += rank
        if a < 0 or a >= rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"axis out of range. axis={axis} normalized={a} rank={rank}",
                node_name=node.name,
                node_op=node.op,
            )
        if a not in normalized:
            normalized.append(a)
    return normalized


def _validate_reduce(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=1,
        attr_name="axes",
        default_if_missing=[int(v) for v in range(input_rank)],
    )
    if len(axes) == 0 and int(node.attrs.get("noop_with_empty_axes", 0)) == 1:
        return
    _normalize_axes_for_rank(axes=axes, rank=input_rank, node=node)


def _validate_cumsum(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"CumSum input rank must be >= 1. input_rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) >= 2 and str(node.inputs[1].name) != "":
        axis_arr = _require_const_input(node, ctx, 1, "CumSum axis")
        axis_values = np.asarray(axis_arr).reshape(-1)
        if int(axis_values.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "CumSum axis must be scalar or single-element tensor. "
                    f"axis_shape={list(np.asarray(axis_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        axis_raw = int(axis_values[0])
    else:
        axis_raw = int(node.attrs.get("axis", 0))

    axis = int(axis_raw)
    if axis < 0:
        axis += int(input_rank)
    if axis < 0 or axis >= int(input_rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"CumSum axis out of range. axis={axis_raw} normalized={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    for attr_name in ["exclusive", "reverse"]:
        attr_value = int(node.attrs.get(attr_name, 0))
        if attr_value not in [0, 1]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"CumSum {attr_name} must be 0 or 1. got={attr_value}",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_squeeze(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=1,
        attr_name="axes",
    )
    if len(axes) == 0:
        return
    _normalize_axes_for_rank(axes=axes, rank=input_rank, node=node)


def _validate_unsqueeze(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=1,
        attr_name="axes",
    )
    if len(axes) == 0:
        raise NodeValidationError(
            reason_code="missing_required_attribute",
            message="Unsqueeze requires axes via input tensor or attribute.",
            node_name=node.name,
            node_op=node.op,
        )
    if input_rank == 0:
        output_rank = len(ctx.get_tensor_shape(node.outputs[0].name))
        if output_rank > 0:
            input_rank = int(max(output_rank - len(axes), 0))
    output_rank = int(input_rank + len(axes))
    normalized_axes: List[int] = []
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += output_rank
        if a < 0 or a >= output_rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    f"unsqueeze axis out of range. axis={axis} normalized={a} "
                    f"input_rank={input_rank} output_rank={output_rank}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if a in normalized_axes:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"unsqueeze axes must be unique. axes={axes}",
                node_name=node.name,
                node_op=node.op,
            )
        normalized_axes.append(int(a))


def _validate_gather(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Gather axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(node.attrs.get("batch_dims", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Gather batch_dims must be 0. batch_dims={int(node.attrs.get('batch_dims', 0))}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_gather_nd(node: Any, ctx: Any) -> None:
    params_shape = ctx.get_tensor_shape(node.inputs[0].name)
    indices_shape = ctx.get_tensor_shape(node.inputs[1].name)
    if len(params_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GatherND params rank must be >= 1. params_shape={params_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if len(indices_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"GatherND indices rank must be >= 1. indices_shape={indices_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    batch_dims = int(node.attrs.get("batch_dims", 0))
    if batch_dims != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GatherND batch_dims must be 0. batch_dims={batch_dims}",
            node_name=node.name,
            node_op=node.op,
        )

    indices_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    supported_indices_dtypes = {
        "INT8",
        "UINT8",
        "INT16",
        "UINT16",
        "INT32",
        "UINT32",
        "INT64",
        "UINT64",
    }
    if indices_dtype not in supported_indices_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "GatherND indices dtype must be integer. "
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
                "GatherND requires static positive indices last dimension in flatbuffer_direct. "
                f"indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if k_dim > len(params_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GatherND indices last dimension must be <= params rank. "
                f"indices_last_dim={k_dim} params_rank={len(params_shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_argmax(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    input_rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMax axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    select_last_index = int(node.attrs.get("select_last_index", 0))
    if select_last_index != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMax select_last_index must be 0. got={select_last_index}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_argmin(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    input_rank = len(input_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMin axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    select_last_index = int(node.attrs.get("select_last_index", 0))
    if select_last_index != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ArgMin select_last_index must be 0. got={select_last_index}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_topk(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"TopK input rank must be >= 1. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "TopK currently supports FLOAT16/FLOAT32 input in flatbuffer_direct. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    axis = int(node.attrs.get("axis", -1))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TopK axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    largest = int(node.attrs.get("largest", 1))
    if largest not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TopK largest must be 0 or 1. largest={largest}",
            node_name=node.name,
            node_op=node.op,
        )
    sorted_attr = int(node.attrs.get("sorted", 1))
    if sorted_attr not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"TopK sorted must be 0 or 1 in flatbuffer_direct builtin lowering. sorted={sorted_attr}",
            node_name=node.name,
            node_op=node.op,
        )

    k_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    if len(k_shape) == 0:
        pass
    elif len(k_shape) == 1 and int(k_shape[0]) <= 1:
        pass
    else:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "TopK k input must be scalar-like (shape [] or [1]) in flatbuffer_direct. "
                f"k_shape={k_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    k_dtype = str(ctx.get_tensor_dtype(node.inputs[1].name)).upper()
    if not _is_integer_dtype(k_dtype):
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=f"TopK k input must be integer dtype. k_dtype={k_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.outputs) >= 2:
        indices_dtype = str(ctx.get_tensor_dtype(node.outputs[1].name)).upper()
        if indices_dtype not in {"INT32", "INT64"}:
            raise NodeValidationError(
                reason_code="unsupported_output_dtype",
                message=(
                    "TopK indices output dtype must be INT32 or INT64 in flatbuffer_direct. "
                    f"indices_dtype={indices_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_hardmax(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Hardmax input rank must be >= 1. shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += input_rank
    if axis < 0 or axis >= input_rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Hardmax axis out of range. axis={axis} rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(input_shape[axis]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "Hardmax requires static positive dimension on target axis in flatbuffer_direct. "
                f"axis={axis} input_shape={input_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_nonzero(node: Any, ctx: Any) -> None:
    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    if len(input_shape) < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"NonZero input rank must be >= 1. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
    if len(output_shape) != 2:
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=f"NonZero output rank must be 2. output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_non_max_suppression(node: Any, ctx: Any) -> None:
    boxes_shape = ctx.get_tensor_shape(node.inputs[0].name)
    scores_shape = ctx.get_tensor_shape(node.inputs[1].name)
    output_nms_with_argmax = bool(getattr(ctx, "output_nms_with_argmax", False))
    if len(boxes_shape) != 3 or len(scores_shape) != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "NonMaxSuppression builtin lowering currently supports rank-3 boxes and scores only. "
                f"boxes_shape={boxes_shape} scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(node.attrs.get("center_point_box", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "NonMaxSuppression center_point_box=1 is not supported in flatbuffer_direct builtin lowering."
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(boxes_shape[0]) != 1 or int(scores_shape[0]) != 1:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression builtin lowering currently supports only batch=1. "
                f"boxes_shape={boxes_shape} scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not output_nms_with_argmax and int(scores_shape[1]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression requires static positive class dimension when "
                "--output_nms_with_argmax is disabled for flatbuffer_direct builtin lowering. "
                f"scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(boxes_shape[2]) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=f"NonMaxSuppression boxes last dimension must be 4. boxes_shape={boxes_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if int(boxes_shape[1]) <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression requires static positive num_boxes in flatbuffer_direct builtin lowering. "
                f"boxes_shape={boxes_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(scores_shape[2]) != int(boxes_shape[1]):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "NonMaxSuppression requires scores_shape[2] == boxes_shape[1] in builtin lowering. "
                f"boxes_shape={boxes_shape} scores_shape={scores_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) >= 3:
        max_output_arr = _require_const_input(
            node,
            ctx,
            2,
            "NonMaxSuppression max_output_boxes_per_class",
        )
        max_output_flat = np.asarray(max_output_arr).reshape(-1)
        if int(max_output_flat.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "NonMaxSuppression max_output_boxes_per_class must be scalar or single-element tensor. "
                    f"shape={list(np.asarray(max_output_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 4:
        iou_threshold_arr = _require_const_input(
            node,
            ctx,
            3,
            "NonMaxSuppression iou_threshold",
        )
        iou_threshold_flat = np.asarray(iou_threshold_arr).reshape(-1)
        if int(iou_threshold_flat.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "NonMaxSuppression iou_threshold must be scalar or single-element tensor. "
                    f"shape={list(np.asarray(iou_threshold_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 5:
        score_threshold_arr = _require_const_input(
            node,
            ctx,
            4,
            "NonMaxSuppression score_threshold",
        )
        score_threshold_flat = np.asarray(score_threshold_arr).reshape(-1)
        if int(score_threshold_flat.size) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "NonMaxSuppression score_threshold must be scalar or single-element tensor. "
                    f"shape={list(np.asarray(score_threshold_arr).shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "NonMaxSuppression output dtype must be INT32 or INT64 in flatbuffer_direct builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_gather_elements(node: Any, ctx: Any) -> None:
    data_shape = ctx.get_tensor_shape(node.inputs[0].name)
    indices_shape = ctx.get_tensor_shape(node.inputs[1].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(data_shape) != len(indices_shape):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                "GatherElements requires data and indices with same rank. "
                f"data_shape={data_shape} indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(indices_shape) != len(output_shape):
        raise NodeValidationError(
            reason_code="invalid_output_shape",
            message=(
                "GatherElements requires output rank equal to indices rank. "
                f"indices_shape={indices_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    rank = len(data_shape)
    axis = int(node.attrs.get("axis", 0))
    if axis < 0:
        axis += rank
    if axis < 0 or axis >= rank:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GatherElements axis out of range. axis={axis} rank={rank}",
            node_name=node.name,
            node_op=node.op,
        )


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

    data_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    indices_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[1].name)]
    output_shape = [int(v) for v in ctx.get_tensor_shape(node.outputs[0].name)]
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
    if updates_dtype != data_dtype:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterND updates dtype must match data dtype in flatbuffer_direct. "
                f"data_dtype={data_dtype} updates_dtype={updates_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype != data_dtype:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "ScatterND output dtype must match data dtype in flatbuffer_direct. "
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


def _validate_scatter_elements(node: Any, ctx: Any) -> None:
    reduction = str(node.attrs.get("reduction", "none")).lower()
    if reduction != "none":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "ScatterElements reduction attribute supports 'none' only in flatbuffer_direct. "
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
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "ScatterElements requires indices rank equal to data rank in flatbuffer_direct. "
                f"data_shape={data_shape} indices_shape={indices_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if len(updates_shape) != len(indices_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "ScatterElements requires updates rank equal to indices rank in flatbuffer_direct. "
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
    if updates_dtype != data_dtype:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ScatterElements updates dtype must match data dtype in flatbuffer_direct. "
                f"data_dtype={data_dtype} updates_dtype={updates_dtype}"
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


def _validate_mod(node: Any, _ctx: Any) -> None:
    fmod = int(node.attrs.get("fmod", 0))
    if fmod != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Mod with fmod=1 is not supported by FLOOR_MOD lowering. fmod={fmod}",
            node_name=node.name,
            node_op=node.op,
        )


def _is_integer_dtype(dtype: str) -> bool:
    return str(dtype).upper() in {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    }


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


def _validate_range(node: Any, ctx: Any) -> None:
    for idx, input_obj in enumerate(node.inputs[:3]):
        shape = [int(v) for v in ctx.get_tensor_shape(input_obj.name)]
        if len(shape) != 1 or int(shape[0]) != 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "Range requires scalar-like inputs represented as rank-1 length-1 "
                    "in flatbuffer_direct. "
                    f"input_index={idx} input_shape={shape}"
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
        lhs, rhs_out = equation.split(",", 1)
        rhs, out = rhs_out.split("->", 1)
    except Exception as ex:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Einsum equation format is invalid. equation={equation}",
            node_name=node.name,
            node_op=node.op,
        ) from ex

    # Specialized builtin lowering:
    #   abgd,gf->abdf
    # using TRANSPOSE+RESHAPE+BATCH_MATMUL+RESHAPE.
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


def _validate_space_to_depth(node: Any, ctx: Any) -> None:
    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"SpaceToDepth blocksize must be > 1. got={block_size}",
            node_name=node.name,
            node_op=node.op,
        )
    mode_raw = node.attrs.get("mode", "DCR")
    if isinstance(mode_raw, (bytes, bytearray)):
        mode = mode_raw.decode("utf-8").upper()
    else:
        mode = str(mode_raw).upper()
    if mode != "DCR":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"SpaceToDepth mode must be DCR. got={mode}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_depth_to_space(node: Any, ctx: Any) -> None:
    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"DepthToSpace blocksize must be > 1. got={block_size}",
            node_name=node.name,
            node_op=node.op,
        )
    mode_raw = node.attrs.get("mode", "DCR")
    if isinstance(mode_raw, (bytes, bytearray)):
        mode = mode_raw.decode("utf-8").upper()
    else:
        mode = str(mode_raw).upper()
    if mode not in {"DCR", "CRD"}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"DepthToSpace mode must be DCR or CRD. got={mode}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_batch_norm(node: Any, ctx: Any) -> None:
    for idx, label in enumerate(["scale", "bias", "mean", "var"], start=1):
        _require_const_input(node, ctx, idx, f"BatchNormalization {label}")
    if len(node.inputs) < 5:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message="BatchNormalization expects 5 inputs.",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_instance_norm(node: Any, ctx: Any) -> None:
    if len(node.inputs) < 3:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message="InstanceNormalization expects 3 inputs.",
            node_name=node.name,
            node_op=node.op,
        )

    scale = _require_const_input(node, ctx, 1, "InstanceNormalization scale")
    bias = _require_const_input(node, ctx, 2, "InstanceNormalization bias")

    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank < 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"InstanceNormalization input rank must be >= 3. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "InstanceNormalization input dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "InstanceNormalization output dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    scale_size = int(np.asarray(scale).size)
    bias_size = int(np.asarray(bias).size)
    if scale_size <= 0 or bias_size <= 0:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "InstanceNormalization scale/bias must be non-empty. "
                f"scale_size={scale_size} bias_size={bias_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if scale_size != bias_size:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "InstanceNormalization scale/bias sizes must match. "
                f"scale_size={scale_size} bias_size={bias_size}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    input_tensor = ctx.model_ir.tensors.get(node.inputs[0].name, None)
    if input_tensor is not None and input_tensor.shape_signature is not None and len(input_tensor.shape_signature) >= 2:
        channel_dim = int(input_tensor.shape_signature[1])
        if channel_dim > 0 and scale_size != channel_dim:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "InstanceNormalization scale/bias size must match input channel dimension. "
                    f"channels={channel_dim} scale_size={scale_size} bias_size={bias_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    epsilon = float(node.attrs.get("epsilon", 1e-5))
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"InstanceNormalization epsilon must be finite and >= 0. got={epsilon}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_layer_norm(node: Any, ctx: Any) -> None:
    if len(node.inputs) < 2:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message="LayerNormalization expects at least 2 inputs (X, Scale).",
            node_name=node.name,
            node_op=node.op,
        )

    input_name = node.inputs[0].name
    scale_name = node.inputs[1].name
    bias_name = node.inputs[2].name if len(node.inputs) >= 3 and str(node.inputs[2].name) != "" else ""

    input_shape = [int(v) for v in ctx.get_tensor_shape(input_name)]
    input_rank = len(input_shape)
    if input_rank < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"LayerNormalization input rank must be >= 1. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )

    axis_raw = int(node.attrs.get("axis", -1))
    axis = _normalize_axis_for_rank(axis=axis_raw, rank=input_rank, node=node)
    expected_reduced_shape = [int(v) for v in input_shape]
    for axis_idx in range(axis, input_rank):
        expected_reduced_shape[axis_idx] = 1

    epsilon = float(node.attrs.get("epsilon", 1e-5))
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LayerNormalization epsilon must be finite and >= 0. got={epsilon}",
            node_name=node.name,
            node_op=node.op,
        )

    stash_type = int(node.attrs.get("stash_type", 1))
    if stash_type not in {0, 1}:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LayerNormalization stash_type must be 0 or 1. got={stash_type}",
            node_name=node.name,
            node_op=node.op,
        )

    allowed_float_dtypes = {"FLOAT16", "FLOAT32"}
    input_dtype = str(ctx.get_tensor_dtype(input_name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    scale_dtype = str(ctx.get_tensor_dtype(scale_name)).upper()
    if input_dtype not in allowed_float_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "LayerNormalization input dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in allowed_float_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "LayerNormalization output dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if scale_dtype not in allowed_float_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "LayerNormalization scale dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"scale_dtype={scale_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if bias_name != "":
        bias_dtype = str(ctx.get_tensor_dtype(bias_name)).upper()
        if bias_dtype not in allowed_float_dtypes:
            raise NodeValidationError(
                reason_code="unsupported_input_dtype",
                message=(
                    "LayerNormalization bias dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                    f"bias_dtype={bias_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    for output_idx in [1, 2]:
        if len(node.outputs) <= output_idx:
            continue
        aux_dtype = str(ctx.get_tensor_dtype(node.outputs[output_idx].name)).upper()
        if aux_dtype not in allowed_float_dtypes:
            aux_name = "Mean" if output_idx == 1 else "InvStdDev"
            raise NodeValidationError(
                reason_code="unsupported_output_dtype",
                message=(
                    f"LayerNormalization {aux_name} output dtype must be FLOAT16/FLOAT32 "
                    f"for builtin lowering. output_dtype={aux_dtype}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    def _validate_unidirectional_broadcast(param_name: str, label: str) -> None:
        param_shape = [int(v) for v in ctx.get_tensor_shape(param_name)]
        if param_shape == [1]:
            return
        if len(param_shape) > input_rank:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    f"LayerNormalization {label} rank must be <= input rank for unidirectional broadcast. "
                    f"input_shape={input_shape} {label}_shape={param_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        for rev_idx in range(1, len(param_shape) + 1):
            p_dim = int(param_shape[-rev_idx])
            x_dim = int(input_shape[-rev_idx])
            if p_dim <= 0 or x_dim <= 0:
                continue
            if p_dim != 1 and p_dim != x_dim:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        f"LayerNormalization {label} is not unidirectional-broadcastable to X. "
                        f"input_shape={input_shape} {label}_shape={param_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )

    _validate_unidirectional_broadcast(scale_name, "Scale")
    if bias_name != "":
        _validate_unidirectional_broadcast(bias_name, "Bias")

    for output_idx in [1, 2]:
        if len(node.outputs) <= output_idx:
            continue
        output_name = node.outputs[output_idx].name
        output_shape = [int(v) for v in ctx.get_tensor_shape(output_name)]
        if output_shape == [1]:
            continue
        if len(output_shape) != input_rank:
            raise NodeValidationError(
                reason_code="unsupported_output_rank",
                message=(
                    "LayerNormalization auxiliary output rank must match input rank. "
                    f"output_name={output_name} input_rank={input_rank} output_shape={output_shape}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
        for expected_dim, actual_dim in zip(expected_reduced_shape, output_shape):
            if int(expected_dim) <= 0 or int(actual_dim) <= 0:
                continue
            if int(expected_dim) != int(actual_dim):
                raise NodeValidationError(
                    reason_code="unsupported_output_shape",
                    message=(
                        "LayerNormalization auxiliary output shape mismatch. "
                        f"expected={expected_reduced_shape} actual={output_shape}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )


def _validate_flatten(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Flatten input rank must be >= 1. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_quantize_dequantize_linear(node: Any, ctx: Any) -> None:
    scale = _require_const_input(node, ctx, 1, f"{node.op} scale")
    if len(node.inputs) >= 3:
        _require_const_input(node, ctx, 2, f"{node.op} zero_point")
    if int(np.asarray(scale).size) <= 1:
        return
    axis = int(node.attrs.get("axis", 1))
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank <= 1:
        return
    _ = _normalize_axis_for_rank(
        axis=axis,
        rank=input_rank,
        node=node,
    )


def _validate_dynamic_quantize_linear(node: Any, ctx: Any) -> None:
    x_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if x_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "DynamicQuantizeLinear input dtype must be FLOAT16 or FLOAT32 for builtin lowering. "
                f"input_dtype={x_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    y_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    scale_dtype = str(ctx.get_tensor_dtype(node.outputs[1].name)).upper()
    zero_dtype = str(ctx.get_tensor_dtype(node.outputs[2].name)).upper()
    if y_dtype != "UINT8":
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "DynamicQuantizeLinear output[0] dtype must be UINT8 for builtin lowering. "
                f"output_dtype={y_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if scale_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "DynamicQuantizeLinear output[1] dtype must be FLOAT16 or FLOAT32 for builtin lowering. "
                f"output_dtype={scale_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if zero_dtype != "UINT8":
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "DynamicQuantizeLinear output[2] dtype must be UINT8 for builtin lowering. "
                f"output_dtype={zero_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    for output_index in [1, 2]:
        output_name = node.outputs[output_index].name
        shape = [int(v) for v in list(ctx.get_tensor_shape(output_name))]
        # Accept scalar or scalar-like placeholder [1] only.
        if len(shape) == 0:
            continue
        if len(shape) == 1 and int(shape[0]) == 1:
            continue
        raise NodeValidationError(
            reason_code="unsupported_output_shape",
            message=(
                "DynamicQuantizeLinear scale/zero_point outputs must be scalar for builtin lowering. "
                f"output_index={output_index} shape={shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_binary(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "a_scale"),
        (2, "a_zero_point"),
        (4, "b_scale"),
        (5, "b_zero_point"),
        (6, "c_scale"),
        (7, "c_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"{node.op} {label}")


def _validate_qlinear_concat(node: Any, ctx: Any) -> None:
    if len(node.inputs) < 5 or (len(node.inputs) - 2) % 3 != 0:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=(
                "QLinearConcat expects [y_scale, y_zero_point, (x, x_scale, x_zero_point)+]. "
                f"input_count={len(node.inputs)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    _require_const_input(node, ctx, 0, "QLinearConcat y_scale")
    _require_const_input(node, ctx, 1, "QLinearConcat y_zero_point")

    first_input_shape = ctx.get_tensor_shape(node.inputs[2].name)
    rank = len(first_input_shape)
    axis = int(node.attrs.get("axis", 1))
    _ = _normalize_axis_for_rank(axis=axis, rank=rank, node=node)

    for group_idx in range((len(node.inputs) - 2) // 3):
        base = 2 + group_idx * 3
        x_name = node.inputs[base].name
        x_scale_name = node.inputs[base + 1].name
        x_zero_name = node.inputs[base + 2].name
        _require_const_input(node, ctx, base + 1, f"QLinearConcat input[{group_idx}] scale")
        _require_const_input(node, ctx, base + 2, f"QLinearConcat input[{group_idx}] zero_point")
        shape_i = ctx.get_tensor_shape(x_name)
        if len(shape_i) != rank:
            raise NodeValidationError(
                reason_code="unsupported_input_rank",
                message=(
                    f"QLinearConcat input ranks must match. "
                    f"input={x_name} shape={shape_i} expected_rank={rank}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_qlinear_conv(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(input_shape) not in [1, 4] or len(output_shape) not in [1, 4]:
        raise NodeValidationError(
            reason_code="unsupported_tensor_rank",
            message=(
                "QLinearConv supports only rank-4 tensors. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    weights = _require_const_input(node, ctx, 3, "QLinearConv weights")
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"QLinearConv weight rank must be 4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (4, "w_scale"),
        (5, "w_zero_point"),
        (6, "y_scale"),
        (7, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearConv {label}")
    group = int(node.attrs.get("group", 1))
    if len(input_shape) == 4:
        in_channels = int(input_shape[1])
        weight_in_channels_per_group = int(weights.shape[1])
        weight_out_channels = int(weights.shape[0])
        # Prefer weight/group-based depthwise detection because some quantized
        # models carry incomplete shape metadata during direct lowering.
        is_depthwise = (
            group > 1
            and weight_in_channels_per_group == 1
            and (weight_out_channels % group) == 0
        )
        if group != 1 and not is_depthwise:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "QLinearConv supports only regular or depthwise group conv. "
                    f"group={group} in_channels={in_channels} "
                    f"weight_shape={list(weights.shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )
    if len(node.inputs) >= 9:
        _require_const_input(node, ctx, 8, "QLinearConv bias")


def _validate_conv_integer(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(input_shape) not in [1, 4] or len(output_shape) not in [1, 4]:
        raise NodeValidationError(
            reason_code="unsupported_tensor_rank",
            message=(
                "ConvInteger supports only rank-4 tensors. "
                f"input_shape={input_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    supported_input_dtypes = {"INT8", "UINT8", "INT16", "UINT16", "INT32"}
    x_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    if x_dtype not in supported_input_dtypes:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "ConvInteger input dtype must be an integer tensor type for builtin lowering. "
                f"input_dtype={x_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if output_dtype not in {"INT32", "INT64"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=f"ConvInteger output dtype must be INT32 or INT64. got={output_dtype}",
            node_name=node.name,
            node_op=node.op,
        )

    weights = _require_const_input(node, ctx, 1, "ConvInteger weights")
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"ConvInteger weight rank must be 4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )

    group = int(node.attrs.get("group", 1))
    if len(input_shape) == 4:
        in_channels = int(input_shape[1])
        weight_in_channels_per_group = int(weights.shape[1])
        weight_out_channels = int(weights.shape[0])
        is_depthwise = (
            group > 1
            and weight_in_channels_per_group == 1
            and (weight_out_channels % group) == 0
        )
        if group != 1 and not is_depthwise:
            raise NodeValidationError(
                reason_code="unsupported_grouped_convolution",
                message=(
                    "ConvInteger supports only regular or depthwise group conv. "
                    f"group={group} in_channels={in_channels} weight_shape={list(weights.shape)}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 3:
        x_zero_shape = ctx.get_tensor_shape(node.inputs[2].name)
        if len(x_zero_shape) > 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=f"ConvInteger x_zero_point must be scalar or rank-1. shape={x_zero_shape}",
                node_name=node.name,
                node_op=node.op,
            )

    if len(node.inputs) >= 4:
        w_zero = _require_const_input(node, ctx, 3, "ConvInteger w_zero_point")
        w_zero_shape = list(np.asarray(w_zero).shape)
        if len(w_zero_shape) > 1:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=f"ConvInteger w_zero_point must be scalar or rank-1. shape={w_zero_shape}",
                node_name=node.name,
                node_op=node.op,
            )
        if len(w_zero_shape) == 1 and int(w_zero_shape[0]) > 1 and int(w_zero_shape[0]) != int(weights.shape[0]):
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "ConvInteger w_zero_point length mismatch. "
                    f"shape={w_zero_shape} expected={int(weights.shape[0])}"
                ),
                node_name=node.name,
                node_op=node.op,
            )


def _validate_qlinear_matmul(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank not in [1, 2]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QLinearMatMul input rank must be 2. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    weights = _require_const_input(node, ctx, 3, "QLinearMatMul weights")
    if weights.ndim != 2:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"QLinearMatMul weight rank must be 2. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, label in [
        (1, "a_scale"),
        (2, "a_zero_point"),
        (4, "b_scale"),
        (5, "b_zero_point"),
        (6, "y_scale"),
        (7, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearMatMul {label}")


def _validate_qgemm(node: Any, ctx: Any) -> None:
    input_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    if input_rank not in [1, 2]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QGemm input rank must be 1 or 2. rank={input_rank}",
            node_name=node.name,
            node_op=node.op,
        )
    weights = _require_const_input(node, ctx, 3, "QGemm weights")
    if weights.ndim != 2:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"QGemm weight rank must be 2. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    for idx, label in [
        (1, "a_scale"),
        (2, "a_zero_point"),
        (4, "b_scale"),
        (5, "b_zero_point"),
        (6, "bias"),
        (7, "y_scale"),
        (8, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QGemm {label}")
    trans_a = int(node.attrs.get("transA", 0))
    trans_b = int(node.attrs.get("transB", 0))
    if trans_a != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QGemm transA must be 0. got={trans_a}",
            node_name=node.name,
            node_op=node.op,
        )
    if trans_b not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QGemm transB must be 0 or 1. got={trans_b}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_sigmoid(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearSigmoid {label}")


def _validate_qlinear_leaky_relu(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearLeakyRelu {label}")


def _validate_qlinear_softmax(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearSoftmax {label}")
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += len(input_shape)
    if axis != len(input_shape) - 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearSoftmax axis must be last dimension. axis={axis} shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_global_average_pool(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearGlobalAveragePool {label}")

    channels_last = int(node.attrs.get("channels_last", 0))
    if channels_last not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearGlobalAveragePool channels_last must be 0 or 1. got={channels_last}",
            node_name=node.name,
            node_op=node.op,
        )

    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if input_shape != [1] and len(input_shape) < 3:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QLinearGlobalAveragePool input rank must be >=3. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_qlinear_average_pool(node: Any, ctx: Any) -> None:
    for idx, label in [
        (1, "x_scale"),
        (2, "x_zero_point"),
        (3, "y_scale"),
        (4, "y_zero_point"),
    ]:
        _require_const_input(node, ctx, idx, f"QLinearAveragePool {label}")

    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if input_shape != [1] and len(input_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"QLinearAveragePool supports rank-4 input. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )

    kernel = [int(v) for v in list(node.attrs.get("kernel_shape", []))]
    if len(kernel) != 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool kernel_shape must be 2D. kernel_shape={kernel}",
            node_name=node.name,
            node_op=node.op,
        )
    strides = [int(v) for v in list(node.attrs.get("strides", [1, 1]))]
    if len(strides) != 2:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool strides must be 2D. strides={strides}",
            node_name=node.name,
            node_op=node.op,
        )
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool dilations must be [1,1]. dilations={dilations}",
            node_name=node.name,
            node_op=node.op,
        )
    ceil_mode = int(node.attrs.get("ceil_mode", 0))
    if ceil_mode not in [0, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"QLinearAveragePool ceil_mode must be 0 or 1. got={ceil_mode}",
            node_name=node.name,
            node_op=node.op,
        )
    if ceil_mode == 1:
        auto_pad = str(node.attrs.get("auto_pad", "NOTSET")).upper()
        pads = [int(v) for v in list(node.attrs.get("pads", [0, 0, 0, 0]))]
        if len(pads) < 4:
            pads = [0, 0, 0, 0]
        if auto_pad not in ["NOTSET", "SAME", "SAME_UPPER", "SAME_LOWER"]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "QLinearAveragePool ceil_mode=1 supports auto_pad "
                    "NOTSET/SAME/SAME_UPPER/SAME_LOWER only."
                ),
                node_name=node.name,
                node_op=node.op,
            )
        if auto_pad == "NOTSET" and any(int(v) != 0 for v in pads):
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message="QLinearAveragePool ceil_mode=1 with auto_pad=NOTSET requires pads=[0,0,0,0].",
                node_name=node.name,
                node_op=node.op,
            )
    if int(node.attrs.get("count_include_pad", 0)) != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message="QLinearAveragePool count_include_pad must be 0.",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_resize(node: Any, ctx: Any) -> None:
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if input_shape != [1] and len(input_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"Resize supports rank-4 input. input_shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    if output_shape != [1] and len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_output_rank",
            message=f"Resize supports rank-4 output. output_shape={output_shape}",
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
    ctm = str(node.attrs.get("coordinate_transformation_mode", "half_pixel")).lower()
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
        if nearest_mode not in ["floor", "round_prefer_floor"]:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"Resize nearest_mode must be floor or round_prefer_floor. got={nearest_mode}",
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
            if sizes_len > 0 and sizes_len not in [2, 4]:
                raise NodeValidationError(
                    reason_code="unsupported_input_shape",
                    message=(
                        "Resize dynamic sizes input length must be 2 or 4. "
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
    if mode != "bilinear":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"GridSample supports mode=bilinear only in flatbuffer_direct. mode={mode}",
            node_name=node.name,
            node_op=node.op,
        )
    if padding_mode != "zeros":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                "GridSample supports padding_mode=zeros only in flatbuffer_direct. "
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
    if len(image_shape) != 4 or len(grid_shape) != 4 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=(
                "GridSample supports rank-4 tensors only in flatbuffer_direct. "
                f"image_shape={image_shape} grid_shape={grid_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if any(int(v) <= 0 for v in image_shape + grid_shape + output_shape):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GridSample requires static positive dimensions in flatbuffer_direct. "
                f"image_shape={image_shape} grid_shape={grid_shape} output_shape={output_shape}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if int(grid_shape[3]) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GridSample grid last dimension must be 2 ([x,y]) in flatbuffer_direct. "
                f"grid_shape={grid_shape}. "
                "When grid is a model input, pass -kat/--keep_shape_absolutely_input_names for it."
            ),
            node_name=node.name,
            node_op=node.op,
        )

    n, c, h, w = [int(v) for v in image_shape]
    out_n, out_c, out_h, out_w = [int(v) for v in output_shape]
    grid_n, grid_h, grid_w, _ = [int(v) for v in grid_shape]
    if not (
        n == out_n == grid_n
        and c == out_c
        and out_h == grid_h
        and out_w == grid_w
        and h >= 1
        and w >= 1
    ):
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "GridSample input/grid/output shapes are inconsistent for built-in lowering. "
                f"image_shape={image_shape} grid_shape={grid_shape} output_shape={output_shape}"
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
    ):
        raise NodeValidationError(
            reason_code="unsupported_control_flow_pattern",
            message=(
                "If built-in lowering supports only constrained patterns: "
                "NMS-guard pattern (empty-then + NMS-else), "
                "axis0 Add-branch pattern (single Add in each branch), "
                "or SequenceConstruct Add-branch pattern "
                "(branch-local Constant/Add + terminal SequenceConstruct), "
                "or nested ReduceMin/Add pattern (else-branch ReduceMin/Greater + nested Add/Add If)."
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


def _normalize_axis_for_rank(*, axis: int, rank: int, node: Any) -> int:
    a = int(axis)
    if a < 0:
        a += int(rank)
    if a < 0 or a >= int(rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"axis out of range. axis={axis} normalized={a} rank={rank}",
            node_name=node.name,
            node_op=node.op,
        )
    return a


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
    if len(node.inputs) >= 2:
        min_value = _require_const_input(node, ctx, 1, "clip minimum")
    if len(node.inputs) >= 3:
        max_value = _require_const_input(node, ctx, 2, "clip maximum")

    def _to_float(v: Any, default: float) -> float:
        if isinstance(v, (int, float)):
            return float(v)
        arr = np.asarray(v)
        if arr.size == 0:
            return float(default)
        return float(arr.reshape(-1)[0])

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
        tflite_ops=["MUL", "ADD"],
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
    "CumSum": DispatchEntry(
        onnx_op="CumSum",
        tflite_ops=["CUMSUM"],
        builder=build_cumsum_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_cumsum,
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
    "Pad": DispatchEntry(
        onnx_op="Pad",
        tflite_ops=["PAD", "PADV2", "MIRROR_PAD"],
        builder=build_pad_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=3, min_outputs=1, max_outputs=1),
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
    "Range": DispatchEntry(
        onnx_op="Range",
        tflite_ops=["CAST", "SQUEEZE", "RANGE"],
        builder=build_range_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=3, min_outputs=1, max_outputs=1),
        extra_validator=_validate_range,
    ),
    "RandomNormalLike": DispatchEntry(
        onnx_op="RandomNormalLike",
        tflite_ops=["SHAPE", "RANDOM_STANDARD_NORMAL", "MUL", "ADD", "CAST"],
        builder=build_random_normal_like_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_random_normal_like,
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
        ],
        builder=build_if_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
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
        tflite_ops=["CONCATENATION"],
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
        tflite_ops=["CAST", "GATHER_ND"],
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
        tflite_ops=["RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR"],
        builder=build_resize_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=4, min_outputs=1, max_outputs=1),
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
            "MAXIMUM",
            "MINIMUM",
            "CAST",
            "GATHER",
        ],
        builder=build_grid_sample_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_grid_sample,
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
            output_rank={0: [4]},
        ),
        extra_validator=_validate_depth_to_space,
    ),
    "Conv": DispatchEntry(
        onnx_op="Conv",
        tflite_ops=["CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"],
        builder=build_conv2d_or_depthwise_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [3, 4, 5]},
            output_rank={0: [3, 4, 5]},
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
            input_rank={0: [1, 3, 4]},
            output_rank={0: [1, 3, 4]},
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
            input_rank={0: [4]},
            output_rank={0: [1, 4]},
        ),
        extra_validator=_validate_pool,
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
            input_rank={0: [4]},
            output_rank={0: [1, 4], 1: [1, 4]},
        ),
        extra_validator=_validate_pool,
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
        tflite_ops=["BATCH_MATMUL"],
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
        ],
        builder=build_lstm_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=None, min_outputs=1, max_outputs=3),
        extra_validator=_validate_lstm,
    ),
    "Einsum": DispatchEntry(
        onnx_op="Einsum",
        tflite_ops=["FULLY_CONNECTED", "BATCH_MATMUL", "CAST", "TRANSPOSE", "RESHAPE"],
        builder=build_einsum_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=2,
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
