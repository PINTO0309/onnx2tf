from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _is_fully_known_positive_shape,
    _prune_unused_tensors,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Materialize a runtime shape for a dynamic rank-one Unsqueeze reshape."""

    def _unique_tensor_name(base: str) -> str:
        if str(base) not in model_ir.tensors:
            return str(base)
        index = 1
        while f"{base}_{index}" in model_ir.tensors:
            index += 1
        return f"{base}_{index}"

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("RESHAPE")
    ]
    rewritten = 0
    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) < 2 or len(op.outputs) != 1:
            continue

        input_name = str(op.inputs[0])
        shape_name = str(op.inputs[1])
        output_name = str(op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        shape_tensor = model_ir.tensors.get(shape_name, None)
        if input_tensor is None or output_tensor is None or shape_tensor is None:
            continue
        if shape_tensor.data is None:
            continue

        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else [int(v) for v in list(input_tensor.shape)]
        )
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None
            else [int(v) for v in list(output_tensor.shape)]
        )
        try:
            shape_values = [
                int(v) for v in np.asarray(shape_tensor.data).reshape(-1).tolist()
            ]
        except Exception:
            continue

        if output_signature != [-1, 1] and output_signature != [1, -1]:
            continue
        if (
            shape_values not in ([-1, 1], [1, -1], [1, 1])
            and list(op.options.get("newShape", [])) not in ([], [1, 1])
        ):
            continue

        if len(input_signature) != 1:
            # Late Squeeze/Unsqueeze folding can remove the rank-1 intermediate
            # while retaining its two-dimensional output contract. In that
            # case SHAPE(input) would expose stale higher-rank metadata. Keep
            # the one runtime-inferred extent directly instead; TFLite RESHAPE
            # accepts one -1 independently of the input rank.
            inferred_shape = [int(v) for v in output_signature]
            shape_tensor.data = np.asarray(inferred_shape, dtype=np.int32)
            shape_tensor.dtype = "INT32"
            shape_tensor.shape = [int(len(inferred_shape))]
            shape_tensor.shape_signature = [int(len(inferred_shape))]
            op.options["newShape"] = []
            rewritten += 1
            continue

        runtime_shape_name = _unique_tensor_name(f"{output_name}_runtime_shape")
        model_ir.tensors[runtime_shape_name] = TensorIR(
            name=runtime_shape_name,
            dtype="INT32",
            shape=[1],
            shape_signature=[1],
        )
        runtime_shape_op = OperatorIR(
            op_type="SHAPE",
            inputs=[input_name],
            outputs=[runtime_shape_name],
            options={"outType": "INT32"},
        )

        one_name = _unique_tensor_name(f"{output_name}_unsqueeze_runtime_one")
        model_ir.tensors[one_name] = TensorIR(
            name=one_name,
            dtype="INT32",
            shape=[1],
            shape_signature=[1],
            data=np.asarray([1], dtype=np.int32),
        )
        merged_shape_name = _unique_tensor_name(
            f"{output_name}_unsqueeze_runtime_shape"
        )
        model_ir.tensors[merged_shape_name] = TensorIR(
            name=merged_shape_name,
            dtype="INT32",
            shape=[2],
            shape_signature=[2],
        )
        concat_inputs = [runtime_shape_name, one_name]
        if output_signature == [1, -1]:
            concat_inputs = [one_name, runtime_shape_name]
        concat_op = OperatorIR(
            op_type="CONCATENATION",
            inputs=concat_inputs,
            outputs=[merged_shape_name],
            options={
                "axis": 0,
                "fusedActivationFunction": "NONE",
            },
        )

        graph_index.replace_operator_inputs(
            int(op_index),
            [input_name, merged_shape_name],
        )
        op.options["newShape"] = []
        graph_index.insert_operator(int(op_index), runtime_shape_op)
        graph_index.insert_operator(int(op_index) + 1, concat_op)
        rewritten += 1

    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "rewritten_dynamic_rank1_unsqueeze_reshape_shape_inputs": int(rewritten),
    }


def restore_placeholder_matmul_flattened_inputs(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Remove fallback MatMul flattening after the original rank is recovered."""

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("RESHAPE")
    ]
    restored = 0
    for reshape_op in candidate_ops:
        reshape_index = graph_index.operator_index(reshape_op)
        if (
            reshape_index is None
            or not bool(
                reshape_op.options.get(
                    "onnxMatMulFlattenedPlaceholder",
                    False,
                )
            )
            or len(reshape_op.inputs) < 1
            or len(reshape_op.outputs) != 1
        ):
            continue
        source_name = str(reshape_op.inputs[0])
        flattened_name = str(reshape_op.outputs[0])
        source_tensor = model_ir.tensors.get(source_name)
        if (
            source_tensor is None
            or len(source_tensor.shape) < 3
            or not _is_fully_known_positive_shape(source_tensor.shape)
        ):
            continue
        user_ops = graph_index.consumers_of(flattened_name)
        if len(user_ops) != 1:
            continue
        matmul_op = user_ops[0]
        matmul_index = graph_index.operator_index(matmul_op)
        if (
            matmul_index is None
            or str(matmul_op.op_type) != "BATCH_MATMUL"
            or len(matmul_op.inputs) != 2
            or str(matmul_op.inputs[0]) != flattened_name
            or bool(matmul_op.options.get("adjX", False))
        ):
            continue
        rhs_tensor = model_ir.tensors.get(str(matmul_op.inputs[1]))
        if (
            rhs_tensor is None
            or len(rhs_tensor.shape) != 2
            or not _is_fully_known_positive_shape(rhs_tensor.shape)
        ):
            continue
        rhs_k = (
            int(rhs_tensor.shape[1])
            if bool(matmul_op.options.get("adjY", False))
            else int(rhs_tensor.shape[0])
        )
        if int(source_tensor.shape[-1]) != rhs_k:
            continue

        graph_index.replace_operator_inputs(
            int(matmul_index),
            [source_name, str(matmul_op.inputs[1])],
        )
        current_reshape_index = graph_index.operator_index(reshape_op)
        if current_reshape_index is None:
            continue
        graph_index.remove_operator(int(current_reshape_index))
        restored += 1

    if restored > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {"restored_placeholder_matmul_flattened_inputs": int(restored)}
