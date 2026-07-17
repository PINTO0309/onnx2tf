from __future__ import annotations

from typing import Optional, cast

import numpy as np
import onnx
from onnx import numpy_helper

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.ir import normalize_onnx_shape
from onnx2tf.tflite_builder.tensor_buffer_builder import (
    tflite_dtype_from_numpy,
)


def lower_constant_node(
    *,
    node: onnx.NodeProto,
    ctx: LoweringContext,
) -> None:
    """Lower the tensor-valued ONNX Constant form into one ModelIR tensor."""

    output_name = str(node.output[0])
    value_attr: Optional[onnx.AttributeProto] = None
    for raw_attribute in node.attribute:
        attribute = cast(onnx.AttributeProto, raw_attribute)
        if attribute.name == "value":
            value_attr = attribute
            break
    if value_attr is None:
        raise NotImplementedError(
            "Constant node without value is not supported. "
            f"op={node.name}"
        )

    const_array = np.asarray(numpy_helper.to_array(value_attr.t))
    model_ir = ctx.model_ir
    if output_name in model_ir.tensors:
        tensor = model_ir.tensors[output_name]
        tensor.data = const_array
        tensor.dtype = tflite_dtype_from_numpy(const_array.dtype)
        tensor.shape, tensor.shape_signature = normalize_onnx_shape(
            list(const_array.shape)
        )
        ctx.constants[output_name] = const_array
        return

    added_name = ctx.add_const_tensor(output_name, const_array)
    if added_name != output_name:
        # Preserve the ONNX graph output name if a custom context reports a
        # collision while creating the tensor.
        model_ir.tensors[output_name] = model_ir.tensors.pop(added_name)
        model_ir.tensors[output_name].name = output_name
        ctx.constants[output_name] = ctx.constants.pop(added_name)
