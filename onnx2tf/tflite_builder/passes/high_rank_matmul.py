from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _compress_static_high_rank_batch_matmul(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Lower static rank>5 BATCH_MATMUL through rank-5 reshape wrappers.

    LiteRT accepts at most three batch dimensions for BATCH_MATMUL. When both
    operands have the same fully-static batch shape, flattening the leading
    batch coordinates preserves the independent matrix products exactly. The
    public/output tensor is restored to its original shape after the builtin.

    Broadcasted or dynamic batch shapes are deliberately left unchanged.
    """

    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        serial = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{serial}"
            serial += 1
        return candidate

    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("BATCH_MATMUL")
    ]
    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue

        lhs_name = str(op.inputs[0])
        rhs_name = str(op.inputs[1])
        output_name = str(op.outputs[0])
        lhs_tensor = model_ir.tensors.get(lhs_name, None)
        rhs_tensor = model_ir.tensors.get(rhs_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        if lhs_tensor is None or rhs_tensor is None or output_tensor is None:
            continue

        lhs_shape = [int(v) for v in lhs_tensor.shape]
        rhs_shape = [int(v) for v in rhs_tensor.shape]
        output_shape = [int(v) for v in output_tensor.shape]
        if (
            len(lhs_shape) <= 5
            or len(lhs_shape) != len(rhs_shape)
            or len(output_shape) != len(lhs_shape)
            or not _is_fully_known_positive_shape(lhs_shape)
            or not _is_fully_known_positive_shape(rhs_shape)
            or not _is_fully_known_positive_shape(output_shape)
        ):
            continue

        lhs_batch = [int(v) for v in lhs_shape[:-2]]
        rhs_batch = [int(v) for v in rhs_shape[:-2]]
        output_batch = [int(v) for v in output_shape[:-2]]
        if lhs_batch != rhs_batch or lhs_batch != output_batch:
            continue

        collapse_count = int(len(lhs_batch) - 2)
        if collapse_count < 2:
            continue
        collapsed_batch = int(
            np.prod(lhs_batch[:collapse_count], dtype=np.int64)
        )
        compressed_batch = [
            int(collapsed_batch),
            int(lhs_batch[-2]),
            int(lhs_batch[-1]),
        ]
        compressed_lhs_shape = compressed_batch + lhs_shape[-2:]
        compressed_rhs_shape = compressed_batch + rhs_shape[-2:]
        compressed_output_shape = compressed_batch + output_shape[-2:]
        if any(
            len(shape) > 5
            for shape in (
                compressed_lhs_shape,
                compressed_rhs_shape,
                compressed_output_shape,
            )
        ):
            continue

        lhs_shape_name = _unique_tensor_name(
            f"{lhs_name}_batch_matmul_rank5_shape"
        )
        rhs_shape_name = _unique_tensor_name(
            f"{rhs_name}_batch_matmul_rank5_shape"
        )
        output_restore_shape_name = _unique_tensor_name(
            f"{output_name}_batch_matmul_restore_shape"
        )
        lhs_rank5_name = _unique_tensor_name(
            f"{lhs_name}_batch_matmul_rank5"
        )
        rhs_rank5_name = _unique_tensor_name(
            f"{rhs_name}_batch_matmul_rank5"
        )
        output_rank5_name = _unique_tensor_name(
            f"{output_name}_batch_matmul_rank5"
        )

        for shape_name, shape_values in (
            (lhs_shape_name, compressed_lhs_shape),
            (rhs_shape_name, compressed_rhs_shape),
            (output_restore_shape_name, output_shape),
        ):
            model_ir.tensors[shape_name] = TensorIR(
                name=shape_name,
                dtype="INT32",
                shape=[int(len(shape_values))],
                shape_signature=[int(len(shape_values))],
                data=np.asarray(shape_values, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )

        model_ir.tensors[lhs_rank5_name] = TensorIR(
            name=lhs_rank5_name,
            dtype=str(lhs_tensor.dtype),
            shape=list(compressed_lhs_shape),
            shape_signature=list(compressed_lhs_shape),
            data=None,
            is_variable=False,
            quantization=_clone_quantization(lhs_tensor.quantization),
            logical_layout=str(lhs_tensor.logical_layout),
            physical_layout=str(lhs_tensor.physical_layout),
            onnx_tensor_name=lhs_tensor.onnx_tensor_name,
        )
        model_ir.tensors[rhs_rank5_name] = TensorIR(
            name=rhs_rank5_name,
            dtype=str(rhs_tensor.dtype),
            shape=list(compressed_rhs_shape),
            shape_signature=list(compressed_rhs_shape),
            data=None,
            is_variable=False,
            quantization=_clone_quantization(rhs_tensor.quantization),
            logical_layout=str(rhs_tensor.logical_layout),
            physical_layout=str(rhs_tensor.physical_layout),
            onnx_tensor_name=rhs_tensor.onnx_tensor_name,
        )
        model_ir.tensors[output_rank5_name] = TensorIR(
            name=output_rank5_name,
            dtype=str(output_tensor.dtype),
            shape=list(compressed_output_shape),
            shape_signature=list(compressed_output_shape),
            data=None,
            is_variable=False,
            quantization=_clone_quantization(output_tensor.quantization),
            logical_layout=str(output_tensor.logical_layout),
            physical_layout=str(output_tensor.physical_layout),
            onnx_tensor_name=output_tensor.onnx_tensor_name,
        )

        lhs_reshape_op = OperatorIR(
            op_type="RESHAPE",
            inputs=[lhs_name, lhs_shape_name],
            outputs=[lhs_rank5_name],
            options={"newShape": list(compressed_lhs_shape)},
        )
        rhs_reshape_op = OperatorIR(
            op_type="RESHAPE",
            inputs=[rhs_name, rhs_shape_name],
            outputs=[rhs_rank5_name],
            options={"newShape": list(compressed_rhs_shape)},
        )
        output_reshape_op = OperatorIR(
            op_type="RESHAPE",
            inputs=[output_rank5_name, output_restore_shape_name],
            outputs=[output_name],
            options={"newShape": list(output_shape)},
        )
        current_index = graph_index.operator_index(op)
        if current_index is None:
            continue
        graph_index.replace_operator_inputs(
            int(current_index),
            [lhs_rank5_name, rhs_rank5_name],
        )
        graph_index.replace_operator_outputs(
            int(current_index),
            [output_rank5_name],
        )
        graph_index.insert_operator(int(current_index), lhs_reshape_op)
        graph_index.insert_operator(int(current_index) + 1, rhs_reshape_op)
        graph_index.insert_operator(int(current_index) + 3, output_reshape_op)
        rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"compressed_static_high_rank_batch_matmul": int(rewritten)}
