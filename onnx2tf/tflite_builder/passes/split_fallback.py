from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _prune_unused_tensors,
    _read_const_ints_from_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_SPLIT_SUPPORTED_INPUT_DTYPES = {
    "FLOAT32",
    "UINT8",
    "INT8",
    "INT16",
    "INT32",
    "INT64",
}


def replace_unsupported_split_with_slice(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Replace unsupported-dtype SPLIT operators with optional CAST and SLICE."""

    def _unique_tensor_name(base: str) -> str:
        candidate = str(base)
        serial = 1
        while candidate in model_ir.tensors:
            candidate = f"{base}_{serial}"
            serial += 1
        return candidate

    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("SPLIT")
    ]
    rewritten = 0
    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) < 2 or len(op.outputs) <= 0:
            continue

        axis_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        source_name = str(op.inputs[1])
        source_tensor = model_ir.tensors.get(source_name, None)
        if source_tensor is None:
            continue

        source_dtype = str(source_tensor.dtype).upper()
        if source_dtype in _SPLIT_SUPPORTED_INPUT_DTYPES:
            continue

        axis_values = _read_const_ints_from_tensor(axis_tensor)
        if axis_values is None or len(axis_values) == 0:
            continue

        source_shape = (
            [int(v) for v in list(source_tensor.shape)]
            if source_tensor.shape is not None
            else []
        )
        rank = int(len(source_shape))
        if rank <= 0:
            continue

        axis = int(axis_values[0])
        if axis < 0:
            axis += int(rank)
        if axis < 0 or axis >= int(rank):
            continue

        outputs = [str(value) for value in list(op.outputs)]
        num_splits = int(op.options.get("numSplits", len(outputs)))
        if num_splits <= 0 or len(outputs) != int(num_splits):
            continue

        split_sizes: List[int] = []
        output_dtypes: List[str] = []
        can_derive_from_outputs = True
        for output_name in outputs:
            output_tensor = model_ir.tensors.get(output_name, None)
            if output_tensor is None or output_tensor.shape is None:
                can_derive_from_outputs = False
                break
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if len(output_shape) != int(rank):
                can_derive_from_outputs = False
                break
            split_dim = int(output_shape[int(axis)])
            if split_dim <= 0:
                can_derive_from_outputs = False
                break
            split_sizes.append(int(split_dim))
            output_dtypes.append(str(output_tensor.dtype).upper())

        if not can_derive_from_outputs:
            axis_dim = int(source_shape[int(axis)])
            if axis_dim <= 0 or axis_dim % int(num_splits) != 0:
                continue
            each = int(axis_dim // int(num_splits))
            split_sizes = [int(each) for _ in range(int(num_splits))]
            output_dtypes = [
                str(model_ir.tensors.get(output_name, source_tensor).dtype).upper()
                for output_name in outputs
            ]

        replacement_ops: List[OperatorIR] = []
        slice_source_name = str(source_name)
        unique_output_dtypes = sorted(set(output_dtypes))
        if len(unique_output_dtypes) == 1:
            target_dtype = str(unique_output_dtypes[0]).upper()
            if target_dtype != "" and target_dtype != source_dtype:
                cast_output_name = _unique_tensor_name(f"{source_name}_split_cast")
                source_signature = (
                    [int(v) for v in list(source_tensor.shape_signature)]
                    if source_tensor.shape_signature is not None
                    else [int(v) for v in list(source_shape)]
                )
                model_ir.tensors[cast_output_name] = TensorIR(
                    name=cast_output_name,
                    dtype=target_dtype,
                    shape=[int(v) for v in list(source_shape)],
                    shape_signature=[int(v) for v in list(source_signature)],
                    data=None,
                    is_variable=False,
                    quantization=None,
                )
                replacement_ops.append(
                    OperatorIR(
                        op_type="CAST",
                        inputs=[str(source_name)],
                        outputs=[cast_output_name],
                        options={"outDataType": target_dtype},
                    )
                )
                slice_source_name = str(cast_output_name)

        offset = 0
        for output_index, output_name in enumerate(outputs):
            begin = [0 for _ in range(int(rank))]
            begin[int(axis)] = int(offset)
            size = [-1 for _ in range(int(rank))]
            size[int(axis)] = int(split_sizes[int(output_index)])
            offset += int(split_sizes[int(output_index)])

            begin_name = _unique_tensor_name(
                f"{output_name}_split_fallback_begin"
            )
            size_name = _unique_tensor_name(f"{output_name}_split_fallback_size")
            model_ir.tensors[begin_name] = TensorIR(
                name=begin_name,
                dtype="INT32",
                shape=[int(rank)],
                shape_signature=[int(rank)],
                data=np.asarray(begin, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            model_ir.tensors[size_name] = TensorIR(
                name=size_name,
                dtype="INT32",
                shape=[int(rank)],
                shape_signature=[int(rank)],
                data=np.asarray(size, dtype=np.int32),
                is_variable=False,
                quantization=None,
            )
            replacement_ops.append(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[str(slice_source_name), begin_name, size_name],
                    outputs=[output_name],
                    options={},
                )
            )

        graph_index.remove_operator(int(op_index))
        for offset, replacement_op in enumerate(replacement_ops):
            graph_index.insert_operator(int(op_index) + offset, replacement_op)
        rewritten += 1

    if rewritten > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {"replaced_unsupported_split_with_slice": int(rewritten)}
