from __future__ import annotations

from typing import Optional, Set

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    is_channel_first_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import _is_all_ones_shape
from onnx2tf.tflite_builder.pytorch_graph_policy import (
    _channel_first_shape_values_for_model_ir,
)


def _binary_runtime_shape_passthrough_operand_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_shape_uncertain_tensors: Set[str],
    lhs_name: str,
    rhs_name: str,
) -> Optional[str]:
    lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
    rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
    if lhs_tensor is None or rhs_tensor is None:
        return None
    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
    rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
    if str(lhs_name) in runtime_shape_uncertain_tensors and _is_all_ones_shape(
        rhs_shape
    ):
        return "lhs"
    if str(rhs_name) in runtime_shape_uncertain_tensors and _is_all_ones_shape(
        lhs_shape
    ):
        return "rhs"
    return None


def _binary_requires_runtime_alignment_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_shape_uncertain_tensors: Set[str],
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> bool:
    lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
    rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if lhs_tensor is None or rhs_tensor is None:
        return False
    if (
        str(lhs_name) in runtime_shape_uncertain_tensors
        or str(rhs_name) in runtime_shape_uncertain_tensors
        or str(output_name) in runtime_shape_uncertain_tensors
    ):
        return True
    lhs_layout = normalize_logical_layout(lhs_tensor.logical_layout)
    rhs_layout = normalize_logical_layout(rhs_tensor.logical_layout)
    lhs_cf_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(lhs_name),
    )
    rhs_cf_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(rhs_name),
    )
    output_cf_shape = _channel_first_shape_values_for_model_ir(
        model_ir=model_ir,
        tensor_name=str(output_name),
    )
    if (
        lhs_cf_shape is not None
        and rhs_cf_shape is not None
        and output_cf_shape is not None
        and len(lhs_cf_shape) == len(rhs_cf_shape) == len(output_cf_shape)
        and len(lhs_cf_shape) in {3, 4, 5}
        and is_channel_first_logical_layout(lhs_layout)
        and is_channel_first_logical_layout(rhs_layout)
    ):
        try:
            broadcast_cf_shape = [
                int(v)
                for v in list(
                    np.broadcast_shapes(tuple(lhs_cf_shape), tuple(rhs_cf_shape))
                )
            ]
            if [int(v) for v in list(broadcast_cf_shape)] == [
                int(v) for v in list(output_cf_shape)
            ]:
                return False
        except Exception:
            pass
    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
    rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
    lhs_signature = (
        [int(v) for v in list(lhs_tensor.shape_signature)]
        if lhs_tensor.shape_signature is not None
        else list(lhs_shape)
    )
    rhs_signature = (
        [int(v) for v in list(rhs_tensor.shape_signature)]
        if rhs_tensor.shape_signature is not None
        else list(rhs_shape)
    )
    if len(lhs_shape) != len(rhs_shape):
        return False
    if lhs_shape == rhs_shape and lhs_signature != rhs_signature:
        return True
    if lhs_shape == rhs_shape:
        return False
    try:
        broadcast_shape = [
            int(v)
            for v in list(
                np.broadcast_shapes(tuple(lhs_shape), tuple(rhs_shape))
            )
        ]
        if output_tensor is None:
            return False
        output_shape = [int(v) for v in list(output_tensor.shape)]
        if broadcast_shape != output_shape:
            return True
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else list(output_shape)
        )
        return (
            lhs_signature != lhs_shape
            or rhs_signature != rhs_shape
            or output_signature != output_shape
        )
    except Exception:
        return True
    return bool(
        (isinstance(lhs_tensor.data, np.ndarray) and len(lhs_shape) > 1)
        or (isinstance(rhs_tensor.data, np.ndarray) and len(rhs_shape) > 1)
    )


def _preferred_binary_alignment_anchor_for_codegen(
    *,
    model_ir: ModelIR,
    lhs_name: str,
    rhs_name: str,
    output_name: str,
) -> Optional[str]:
    lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
    rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if lhs_tensor is None or rhs_tensor is None or output_tensor is None:
        return None
    lhs_signature = (
        [int(v) for v in list(lhs_tensor.shape_signature)]
        if lhs_tensor.shape_signature is not None
        else [int(v) for v in list(lhs_tensor.shape)]
    )
    rhs_signature = (
        [int(v) for v in list(rhs_tensor.shape_signature)]
        if rhs_tensor.shape_signature is not None
        else [int(v) for v in list(rhs_tensor.shape)]
    )
    output_signature = (
        [int(v) for v in list(output_tensor.shape_signature)]
        if output_tensor.shape_signature is not None
        else [int(v) for v in list(output_tensor.shape)]
    )
    if len(lhs_signature) != len(rhs_signature) or len(lhs_signature) != len(
        output_signature
    ):
        return None
    if lhs_signature == output_signature and rhs_signature != output_signature:
        return "lhs"
    if rhs_signature == output_signature and lhs_signature != output_signature:
        return "rhs"
    lhs_dynamic_dims = sum(1 for dim in lhs_signature if int(dim) <= 0)
    rhs_dynamic_dims = sum(1 for dim in rhs_signature if int(dim) <= 0)
    if lhs_dynamic_dims > rhs_dynamic_dims:
        return "lhs"
    if rhs_dynamic_dims > lhs_dynamic_dims:
        return "rhs"
    return None
