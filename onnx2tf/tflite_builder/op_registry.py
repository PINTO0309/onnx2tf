from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.op_builders import (
    build_argmax_op,
    build_batch_normalization_op,
    build_binary_op,
    build_cast_op,
    build_clip_op,
    build_concat_op,
    build_constant_of_shape_op,
    build_conv2d_or_depthwise_op,
    build_conv_transpose_op,
    build_custom_passthrough_op,
    build_dequantize_linear_op,
    build_dynamic_quantize_linear_op,
    build_div_op,
    build_expand_op,
    build_flatten_op,
    build_fused_matmul_op,
    build_fully_connected_from_gemm_or_matmul,
    build_matmul_op,
    build_gather_op,
    build_gather_elements_op,
    build_hardsigmoid_op,
    build_logsoftmax_op,
    build_qgemm_op,
    build_identity_op,
    build_lstm_op,
    build_pad_op,
    build_mod_op,
    build_one_hot_op,
    build_l2_normalization_op,
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
    build_reduce_op,
    build_reciprocal_op,
    build_resize_op,
    build_reshape_op,
    build_shape_op,
    build_slice_op,
    build_split_op,
    build_space_to_depth_op,
    build_squeeze_op,
    build_softmax_op,
    build_transpose_op,
    build_unary_op,
    build_unsqueeze_op,
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
    if direction != "bidirectional":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"LSTM direction must be bidirectional for builtin lowering. direction={direction}",
            node_name=node.name,
            node_op=node.op,
        )
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

    for output_index in [1, 2]:
        if output_index < len(node.outputs):
            output_name = str(node.outputs[output_index].name)
            if _is_tensor_consumed_or_graph_output(ctx, output_name):
                raise NodeValidationError(
                    reason_code="unsupported_output_usage",
                    message=(
                        "LSTM output_state/cell_state outputs are not supported by builtin lowering "
                        f"when consumed. output_name={output_name}"
                    ),
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
    if int(W.shape[0]) != 2 or int(R.shape[0]) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM builtin lowering expects bidirectional weights with num_directions=2. "
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
        if B.ndim != 2 or int(B.shape[0]) != 2 or int(B.shape[1]) != 8 * hidden_size:
            raise NodeValidationError(
                reason_code="unsupported_input_shape",
                message=(
                    "LSTM B must have shape [2, 8*hidden_size]. "
                    f"B_shape={list(B.shape)} hidden_size={hidden_size}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    initial_h_name = _input_name(5)
    initial_c_name = _input_name(6)
    if initial_h_name == "" or initial_c_name == "":
        raise NodeValidationError(
            reason_code="missing_required_input",
            message="LSTM builtin lowering requires initial_h and initial_c inputs.",
            node_name=node.name,
            node_op=node.op,
        )
    initial_h = ctx.get_constant_array(initial_h_name)
    initial_c = ctx.get_constant_array(initial_c_name)
    if initial_h is None or initial_c is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="LSTM initial_h and initial_c must be constant tensors for builtin lowering.",
            node_name=node.name,
            node_op=node.op,
        )
    initial_h = np.asarray(initial_h)
    initial_c = np.asarray(initial_c)
    if initial_h.ndim != 3 or initial_c.ndim != 3:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM initial_h and initial_c must be rank-3 with shape [2, batch, hidden]. "
                f"initial_h_shape={list(initial_h.shape)} initial_c_shape={list(initial_c.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if int(initial_h.shape[0]) != 2 or int(initial_c.shape[0]) != 2:
        raise NodeValidationError(
            reason_code="unsupported_input_shape",
            message=(
                "LSTM initial_h and initial_c first dimension must be 2. "
                f"initial_h_shape={list(initial_h.shape)} initial_c_shape={list(initial_c.shape)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if not np.allclose(initial_h, 0.0, rtol=0.0, atol=1e-7) \
        or not np.allclose(initial_c, 0.0, rtol=0.0, atol=1e-7):
        raise NodeValidationError(
            reason_code="unsupported_input_value",
            message=(
                "LSTM builtin lowering currently supports zero-initialized initial_h/initial_c only."
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
    axis = int(node.attrs.get("axis", 1))
    if axis < 0:
        axis += int(rank)
    if axis < 0 or axis >= int(rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Softmax axis is out of range. axis={axis} rank={rank} shape={input_shape}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_reshape(node: Any, ctx: Any) -> None:
    _require_const_input(node, ctx, 1, "reshape shape")


def _validate_slice(node: Any, ctx: Any) -> None:
    starts = _require_const_input(node, ctx, 1, "slice starts")
    ends = _require_const_input(node, ctx, 2, "slice ends")
    starts_values = [int(v) for v in np.asarray(starts).reshape(-1).tolist()]
    ends_values = [int(v) for v in np.asarray(ends).reshape(-1).tolist()]
    if len(starts_values) != len(ends_values):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/ends length mismatch. "
                f"starts_len={len(starts_values)} ends_len={len(ends_values)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    axes = _extract_axes(
        node=node,
        ctx=ctx,
        input_index=3,
        attr_name="axes",
        default_if_missing=[int(v) for v in range(len(starts_values))],
    )
    normalized_axes = _normalize_axes_for_rank(axes=axes, rank=rank, node=node)
    if len(normalized_axes) != len(starts_values):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/axes length mismatch. "
                f"starts_len={len(starts_values)} axes_len={len(normalized_axes)}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    if len(node.inputs) >= 5:
        steps_arr = _require_const_input(node, ctx, 4, "slice steps")
        steps = [int(v) for v in np.asarray(steps_arr).reshape(-1).tolist()]
    elif "steps" in node.attrs:
        attr_steps = node.attrs.get("steps")
        if isinstance(attr_steps, (list, tuple, np.ndarray)):
            steps = [int(v) for v in np.asarray(attr_steps).reshape(-1).tolist()]
        elif attr_steps is None:
            steps = [1 for _ in range(len(starts_values))]
        else:
            steps = [int(attr_steps)]
    else:
        steps = [1 for _ in range(len(starts_values))]

    if len(steps) != len(starts_values):
        raise NodeValidationError(
            reason_code="invalid_input_shape",
            message=(
                f"Slice starts/steps length mismatch. "
                f"starts_len={len(starts_values)} steps_len={len(steps)}"
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
    if any(int(step) < 0 for step in steps):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Slice negative steps are not supported. steps={steps}",
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
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"Conv weight rank must be 4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    output_shape = ctx.get_tensor_shape(node.outputs[0].name)
    if len(input_shape) != 4 or len(output_shape) != 4:
        raise NodeValidationError(
            reason_code="unsupported_tensor_rank",
            message=f"Conv input/output rank must be 4. input_shape={input_shape} output_shape={output_shape}",
            node_name=node.name,
            node_op=node.op,
        )
    group = int(node.attrs.get("group", 1))
    in_channels = int(input_shape[1])
    is_depthwise = group == in_channels and int(weights.shape[1]) == 1 and group > 1
    if group != 1 and not is_depthwise:
        raise NodeValidationError(
            reason_code="unsupported_grouped_convolution",
            message=f"Only regular or depthwise group conv is supported. group={group} in_channels={in_channels}",
            node_name=node.name,
            node_op=node.op,
        )


def _validate_conv_transpose(node: Any, ctx: Any) -> None:
    weights = _require_const_input(node, ctx, 1, "convtranspose weights")
    if weights.ndim != 4:
        raise NodeValidationError(
            reason_code="unsupported_weight_rank",
            message=f"ConvTranspose weight rank must be 4. weight_shape={list(weights.shape)}",
            node_name=node.name,
            node_op=node.op,
        )
    input_shape = ctx.get_tensor_shape(node.inputs[0].name)
    if len(input_shape) not in [1, 4]:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"ConvTranspose input rank must be 4 (or unknown placeholder rank=1). input_shape={input_shape}",
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
    dilations = [int(v) for v in list(node.attrs.get("dilations", [1, 1]))]
    if dilations != [1, 1]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ConvTranspose dilations must be [1,1]. dilations={dilations}",
            node_name=node.name,
            node_op=node.op,
        )
    output_padding = [int(v) for v in list(node.attrs.get("output_padding", [0, 0]))]
    if any(v != 0 for v in output_padding):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"ConvTranspose output_padding must be [0,0]. output_padding={output_padding}",
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
    else:
        if ceil_mode != 0:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message="Pool ceil_mode must be 0.",
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
    if node.op == "Gemm":
        if int(node.attrs.get("transA", 0)) != 0:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message="Gemm transA=1 is not supported.",
                node_name=node.name,
                node_op=node.op,
            )


def _validate_matmul(node: Any, ctx: Any) -> None:
    a_rank = len(ctx.get_tensor_shape(node.inputs[0].name))
    b_rank = len(ctx.get_tensor_shape(node.inputs[1].name))
    if a_rank < 2 or b_rank < 2:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message=f"MatMul input rank must be >= 2. a_rank={a_rank} b_rank={b_rank}",
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
    if len(node.inputs) > input_index:
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
    for axis in axes:
        a = int(axis)
        if a < 0:
            a += input_rank + 1
        if a < 0 or a > input_rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=f"unsqueeze axis out of range. axis={axis} normalized={a} rank={input_rank}",
                node_name=node.name,
                node_op=node.op,
            )


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
    # Expand is lowered via static-shape MUL-broadcast path in op_builders.shape.
    return


def _validate_mod(node: Any, _ctx: Any) -> None:
    fmod = int(node.attrs.get("fmod", 0))
    if fmod != 0:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Mod with fmod=1 is not supported by FLOOR_MOD lowering. fmod={fmod}",
            node_name=node.name,
            node_op=node.op,
        )


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
    _validate_fc(node, ctx)


def _validate_space_to_depth(node: Any, ctx: Any) -> None:
    block_size = int(node.attrs.get("blocksize", 0))
    if block_size <= 1:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"SpaceToDepth blocksize must be > 1. got={block_size}",
            node_name=node.name,
            node_op=node.op,
        )
    mode = str(node.attrs.get("mode", "DCR")).upper()
    if mode != "DCR":
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"SpaceToDepth mode must be DCR. got={mode}",
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
    if mode not in ["nearest", "linear"]:
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=f"Resize mode must be nearest or linear. got={mode}",
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
                    "Resize(linear) supports coordinate_transformation_mode "
                    f"half_pixel/pytorch_half_pixel/asymmetric/align_corners only. got={ctm}"
                ),
                node_name=node.name,
                node_op=node.op,
            )

    has_const_param = False
    has_dynamic_sizes_param = False
    if len(node.inputs) >= 4:
        tensor_name = node.inputs[3].name
        if tensor_name != "":
            sizes_arr = ctx.get_constant_array(tensor_name)
            if sizes_arr is not None:
                if int(np.asarray(sizes_arr).size) > 0:
                    has_const_param = True
            else:
                has_dynamic_sizes_param = True
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
    if len(node.inputs) >= 3:
        tensor_name = node.inputs[2].name
        if tensor_name != "":
            arr = _require_const_input(node, ctx, 2, "Resize scales")
            if int(np.asarray(arr).size) > 0:
                has_const_param = True
    if len(node.inputs) == 2:
        arr = _require_const_input(node, ctx, 1, "Resize scales/sizes")
        if int(np.asarray(arr).size) > 0:
            has_const_param = True
    if not has_const_param and not has_dynamic_sizes_param:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message="Resize requires non-empty constant scales/sizes or dynamic sizes input.",
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
    if str(node.op) in {"ArgMax", "Cast", "Expand", "GatherElements", "Mod"}:
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
    "Reciprocal": DispatchEntry(
        onnx_op="Reciprocal",
        tflite_ops=["DIV"],
        builder=build_reciprocal_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reciprocal,
    ),
    "Mod": DispatchEntry(
        onnx_op="Mod",
        tflite_ops=["FLOOR_MOD"],
        builder=build_mod_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_mod,
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
    "ReduceMax": DispatchEntry(
        onnx_op="ReduceMax",
        tflite_ops=["REDUCE_MAX"],
        builder=lambda node, ctx: build_reduce_op(node, ctx, "REDUCE_MAX"),
        validation=ValidationSpec(min_inputs=1, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_reduce,
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
    "Relu": DispatchEntry(
        onnx_op="Relu",
        tflite_ops=["RELU"],
        builder=_make_unary_builder("RELU"),
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
        tflite_ops=["PAD"],
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
        tflite_ops=["RELU", "RELU6", "MAXIMUM", "MINIMUM"],
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
    "Shape": DispatchEntry(
        onnx_op="Shape",
        tflite_ops=["SHAPE", "SLICE"],
        builder=build_shape_op,
        validation=ValidationSpec(min_inputs=1, max_inputs=1, min_outputs=1, max_outputs=1),
        extra_validator=_validate_shape,
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
    "GatherElements": DispatchEntry(
        onnx_op="GatherElements",
        tflite_ops=["CAST", "RESHAPE", "CONCATENATION", "GATHER_ND"],
        builder=build_gather_elements_op,
        validation=ValidationSpec(min_inputs=2, max_inputs=2, min_outputs=1, max_outputs=1),
        extra_validator=_validate_gather_elements,
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
    "Slice": DispatchEntry(
        onnx_op="Slice",
        tflite_ops=["SLICE", "STRIDED_SLICE"],
        builder=build_slice_op,
        validation=ValidationSpec(min_inputs=3, max_inputs=5, min_outputs=1, max_outputs=1),
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
    "Conv": DispatchEntry(
        onnx_op="Conv",
        tflite_ops=["CONV_2D", "DEPTHWISE_CONV_2D"],
        builder=build_conv2d_or_depthwise_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
            input_rank={0: [4]},
            output_rank={0: [4]},
        ),
        extra_validator=_validate_conv,
    ),
    "ConvTranspose": DispatchEntry(
        onnx_op="ConvTranspose",
        tflite_ops=["TRANSPOSE_CONV", "ADD"],
        builder=build_conv_transpose_op,
        validation=ValidationSpec(
            min_inputs=2,
            max_inputs=3,
            min_outputs=1,
            max_outputs=1,
        ),
        extra_validator=_validate_conv_transpose,
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
            max_outputs=1,
            required_attrs=["kernel_shape"],
            input_rank={0: [4]},
            output_rank={0: [1, 4]},
        ),
        extra_validator=_validate_pool,
    ),
    "Gemm": DispatchEntry(
        onnx_op="Gemm",
        tflite_ops=["FULLY_CONNECTED"],
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
    "LSTM": DispatchEntry(
        onnx_op="LSTM",
        tflite_ops=[
            "BIDIRECTIONAL_SEQUENCE_LSTM",
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
        tflite_ops=["FULLY_CONNECTED"],
        builder=build_fully_connected_from_gemm_or_matmul,
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
