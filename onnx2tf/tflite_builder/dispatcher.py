from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.core.lowering_registry import LoweringRegistry
from onnx2tf.tflite_builder.op_registry import validate_node_support

_LOWERING_REGISTRY = LoweringRegistry(validate_node_support)


def dispatch_node(node: Any, ctx: Any) -> None:
    operator_start = len(ctx.model_ir.operators)
    _LOWERING_REGISTRY.lower(node, ctx)
    node_name = str(getattr(node, "name", "") or getattr(node, "op", ""))
    op_type = str(getattr(node, "op", ""))
    for operator in ctx.model_ir.operators[operator_start:]:
        if operator.onnx_node_name is None:
            operator.onnx_node_name = node_name
        if operator.onnx_op_type is None:
            operator.onnx_op_type = op_type
    for output in getattr(node, "outputs", []):
        tensor_name = str(getattr(output, "name", ""))
        tensor = ctx.model_ir.tensors.get(tensor_name)
        if tensor is not None and tensor.onnx_tensor_name is None:
            tensor.onnx_tensor_name = tensor_name
