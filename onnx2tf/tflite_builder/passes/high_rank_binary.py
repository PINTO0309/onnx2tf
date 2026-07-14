from __future__ import annotations

from copy import deepcopy
from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_BROADCAST_BINARY_OPS = {
    "ADD",
    "DIV",
    "EQUAL",
    "GREATER",
    "GREATER_EQUAL",
    "LESS",
    "LESS_EQUAL",
    "LOGICAL_AND",
    "LOGICAL_OR",
    "MAXIMUM",
    "MINIMUM",
    "MUL",
    "NOT_EQUAL",
    "POW",
    "SUB",
}


def _static_shape(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None:
        return None
    shape = [int(value) for value in list(tensor.shape)]
    signature = (
        [int(value) for value in list(tensor.shape_signature)]
        if tensor.shape_signature is not None
        else list(shape)
    )
    if (
        not shape
        or len(shape) != len(signature)
        or any(value <= 0 for value in shape)
        or any(value <= 0 for value in signature)
    ):
        return None
    return shape


def _coalescing_groups(
    *,
    input_shapes: List[List[int]],
    output_shape: List[int],
) -> Optional[List[List[int]]]:
    rank = len(output_shape)
    padded_shapes: List[List[int]] = []
    for shape in input_shapes:
        if len(shape) > rank:
            return None
        padded = [1] * (rank - len(shape)) + list(shape)
        if any(
            int(source) not in {1, int(target)}
            for source, target in zip(padded, output_shape)
        ):
            return None
        padded_shapes.append(padded)

    patterns = [
        tuple(
            int(shape[axis]) == int(output_shape[axis])
            for shape in padded_shapes
        )
        for axis in range(rank)
    ]
    groups: List[List[int]] = []
    for axis, pattern in enumerate(patterns):
        if not groups or patterns[groups[-1][-1]] != pattern:
            groups.append([int(axis)])
        else:
            groups[-1].append(int(axis))
    if len(groups) > 4 or len(groups) >= rank:
        return None
    return groups


def _coalesced_shape(shape: List[int], groups: List[List[int]]) -> List[int]:
    return [
        int(np.prod([int(shape[axis]) for axis in group], dtype=np.int64))
        for group in groups
    ]


def coalesce_static_high_rank_binary_operators(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Lower fully-static rank>4 binary broadcasts to equivalent rank<=4 ops."""
    rewritten = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)

    def _unique_name(base: str) -> str:
        candidate = str(base)
        serial = 0
        while candidate in model_ir.tensors:
            serial += 1
            candidate = f"{base}_{serial}"
        return candidate

    def _add_shape_tensor(base: str, shape: List[int]) -> str:
        name = _unique_name(base)
        values = np.asarray(shape, dtype=np.int32)
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="INT32",
            shape=[int(values.size)],
            shape_signature=[int(values.size)],
            data=values,
            is_variable=False,
        )
        return name

    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices_for_types(
            _BROADCAST_BINARY_OPS
        )
    ]
    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue

        output_name = str(op.outputs[0])
        output_tensor = model_ir.tensors.get(output_name)
        output_shape = _static_shape(output_tensor)
        input_shapes = [
            _static_shape(model_ir.tensors.get(str(input_name)))
            for input_name in op.inputs
        ]
        if (
            output_shape is None
            or len(output_shape) <= 4
            or any(shape is None for shape in input_shapes)
        ):
            continue

        padded_input_shapes = [
            [1] * (len(output_shape) - len(shape)) + list(shape)
            for shape in input_shapes
            if shape is not None
        ]
        groups = _coalescing_groups(
            input_shapes=[list(shape) for shape in input_shapes if shape is not None],
            output_shape=output_shape,
        )
        if groups is None:
            continue

        coalesced_inputs: List[str] = []
        replacement_ops: List[OperatorIR] = []
        for input_index, (input_name, padded_shape) in enumerate(
            zip(op.inputs, padded_input_shapes)
        ):
            source_tensor = model_ir.tensors[str(input_name)]
            reshaped_name = _unique_name(
                f"{output_name}_high_rank_input{input_index}"
            )
            reshaped_shape = _coalesced_shape(padded_shape, groups)
            model_ir.tensors[reshaped_name] = TensorIR(
                name=reshaped_name,
                dtype=str(source_tensor.dtype),
                shape=list(reshaped_shape),
                shape_signature=list(reshaped_shape),
                data=None,
                is_variable=False,
                quantization=deepcopy(source_tensor.quantization),
                onnx_tensor_name=source_tensor.onnx_tensor_name,
            )
            replacement_ops.append(
                OperatorIR(
                    op_type="RESHAPE",
                    inputs=[
                        str(input_name),
                        _add_shape_tensor(
                            f"{reshaped_name}_shape",
                            reshaped_shape,
                        ),
                    ],
                    outputs=[reshaped_name],
                    options={"newShape": list(reshaped_shape)},
                    onnx_node_name=op.onnx_node_name,
                    onnx_op_type=op.onnx_op_type,
                )
            )
            coalesced_inputs.append(reshaped_name)

        coalesced_output_shape = _coalesced_shape(output_shape, groups)
        coalesced_output_name = _unique_name(
            f"{output_name}_high_rank_output"
        )
        model_ir.tensors[coalesced_output_name] = TensorIR(
            name=coalesced_output_name,
            dtype=str(output_tensor.dtype),
            shape=list(coalesced_output_shape),
            shape_signature=list(coalesced_output_shape),
            data=None,
            is_variable=False,
            quantization=deepcopy(output_tensor.quantization),
            onnx_tensor_name=output_tensor.onnx_tensor_name,
        )
        replacement_ops.append(
            OperatorIR(
                op_type=str(op.op_type),
                inputs=coalesced_inputs,
                outputs=[coalesced_output_name],
                options=dict(op.options),
                axis_semantics=dict(op.axis_semantics),
                version=int(op.version),
                onnx_node_name=op.onnx_node_name,
                onnx_op_type=op.onnx_op_type,
            )
        )
        replacement_ops.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[
                    coalesced_output_name,
                    _add_shape_tensor(
                        f"{output_name}_high_rank_restore_shape",
                        output_shape,
                    ),
                ],
                outputs=[output_name],
                options={"newShape": list(output_shape)},
                onnx_node_name=op.onnx_node_name,
                onnx_op_type=op.onnx_op_type,
            )
            )
        current_index = graph_index.operator_index(op)
        if current_index is None:
            continue
        graph_index.remove_operator(int(current_index))
        for offset, replacement_op in enumerate(replacement_ops):
            graph_index.insert_operator(
                int(current_index) + int(offset),
                replacement_op,
            )
        rewritten += 1

    if rewritten > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {"coalesced_static_high_rank_binary_operators": int(rewritten)}
