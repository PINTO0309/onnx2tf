from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.node import NodeView
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


_SHAPE_SENSITIVE_OPS = frozenset(
    {
        "Attention",
        "Gather",
        "GatherElements",
        "LayerNormalization",
        "MatMul",
        "MultiHeadAttention",
    }
)
_ZERO_RESULT = {"reconciled_static_tensor_shapes": 0}


def _has_unresolved_rank(
    input_name: str,
    *,
    ctx: LoweringContext,
) -> bool:
    if len(ctx.get_tensor_shape(input_name)) > 1:
        return False
    tensor = ctx.model_ir.tensors.get(input_name)
    if tensor is not None and tensor.data is not None:
        return False
    raw_shape = ctx.shape_map.get(input_name)
    return raw_shape is None or len(list(raw_shape)) == 0


def reconcile_shape_sensitive_inputs_on_demand(
    *,
    node: NodeView,
    ctx: LoweringContext,
) -> Dict[str, int]:
    """Reconcile a partial ModelIR only for unresolved shape-sensitive inputs."""

    if str(node.op) not in _SHAPE_SENSITIVE_OPS:
        return dict(_ZERO_RESULT)
    input_names = [
        str(input_value.name)
        for input_value in list(node.inputs)
        if str(input_value.name) != ""
    ]
    if not any(
        _has_unresolved_rank(input_name, ctx=ctx)
        for input_name in input_names
    ):
        return dict(_ZERO_RESULT)
    return reconcile_static_tensor_shapes(ctx.model_ir)
