from __future__ import annotations

from typing import Any

import numpy as np

from onnx2tf.tflite_builder.core.op_contracts import (
    NodeValidationError,
    normalize_axis_for_rank as _normalize_axis_for_rank,
    require_const_input as _require_const_input,
)
from onnx2tf.tflite_builder.ir import (
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)


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
        normalized_layout = normalize_logical_layout(getattr(input_tensor, "logical_layout", "UNKNOWN"))
        if is_channel_first_logical_layout(normalized_layout):
            channel_axis = 1
        elif is_channel_last_logical_layout(normalized_layout):
            channel_axis = len(input_tensor.shape_signature) - 1
        else:
            candidate_axes = [
                int(axis)
                for axis in range(1, len(input_tensor.shape_signature))
                if int(input_tensor.shape_signature[axis]) > 0
                and int(input_tensor.shape_signature[axis]) == int(scale_size)
            ]
            if len(candidate_axes) == 1:
                channel_axis = int(candidate_axes[0])
            elif len(candidate_axes) > 1 and (len(input_tensor.shape_signature) - 1) in candidate_axes:
                channel_axis = len(input_tensor.shape_signature) - 1
            else:
                channel_axis = 1
        channel_dim = int(input_tensor.shape_signature[channel_axis])
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


def _validate_mean_variance_normalization(node: Any, ctx: Any) -> None:
    if len(node.inputs) < 1:
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message="MeanVarianceNormalization expects 1 input.",
            node_name=node.name,
            node_op=node.op,
        )

    input_shape = [int(v) for v in ctx.get_tensor_shape(node.inputs[0].name)]
    input_rank = len(input_shape)
    if input_rank < 1:
        raise NodeValidationError(
            reason_code="unsupported_input_rank",
            message="MeanVarianceNormalization input rank must be >= 1.",
            node_name=node.name,
            node_op=node.op,
        )

    input_dtype = str(ctx.get_tensor_dtype(node.inputs[0].name)).upper()
    output_dtype = str(ctx.get_tensor_dtype(node.outputs[0].name)).upper()
    if input_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_input_dtype",
            message=(
                "MeanVarianceNormalization input dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"input_dtype={input_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    if output_dtype not in {"FLOAT16", "FLOAT32"}:
        raise NodeValidationError(
            reason_code="unsupported_output_dtype",
            message=(
                "MeanVarianceNormalization output dtype must be FLOAT16/FLOAT32 for builtin lowering. "
                f"output_dtype={output_dtype}"
            ),
            node_name=node.name,
            node_op=node.op,
        )

    axes_attr = node.attrs.get("axes", None)
    if input_rank < 3:
        return
    if axes_attr is None:
        axes = [0, 2] if input_rank == 3 else [0, 2, 3]
    elif isinstance(axes_attr, np.ndarray):
        axes = [int(v) for v in np.asarray(axes_attr).reshape(-1).tolist()]
    elif isinstance(axes_attr, (list, tuple)):
        axes = [int(v) for v in axes_attr]
    else:
        axes = [int(axes_attr)]
    for axis in axes:
        if axis < -input_rank or axis >= input_rank:
            raise NodeValidationError(
                reason_code="unsupported_attribute_value",
                message=(
                    "MeanVarianceNormalization axes must be within input rank for builtin lowering. "
                    f"rank={input_rank} axis={axis}"
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
