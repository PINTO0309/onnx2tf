from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    channel_first_logical_layout,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _clone_tensor,
    _is_inconsistent_standard_layout_transpose,
    _is_layout_only_transpose_by_shape,
    _is_standard_channel_layout_permutation,
    _read_transpose_perm,
)


def _remove_redundant_layout_transposes(
    model_ir: ModelIR,
    original_layouts: Dict[str, str],
    preserve_channel_last_tensor_names: set[str],
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> None:
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("TRANSPOSE")
    ]
    changed = False
    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) < 1 or len(op.outputs) != 1:
            continue
        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        if (
            input_name in preserve_channel_last_tensor_names
            or output_name in preserve_channel_last_tensor_names
        ):
            continue
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        reference_tensor = output_tensor if output_tensor is not None else input_tensor
        rank = len(list(reference_tensor.shape)) if reference_tensor is not None else -1
        if rank not in {3, 4, 5}:
            continue
        consumer_ops = [
            consumer
            for consumer in graph_index.consumers_of(output_name)
            if consumer is not op
        ]
        consumer_op_types = {str(consumer.op_type) for consumer in consumer_ops}
        reshape_only_consumers = (
            len(consumer_op_types) > 0 and consumer_op_types == {"RESHAPE"}
        )
        if (
            reshape_only_consumers
            and input_tensor is not None
            and output_tensor is not None
            and [int(value) for value in list(input_tensor.shape)]
            != [int(value) for value in list(output_tensor.shape)]
        ):
            continue
        if consumer_op_types & {"GATHER", "GATHER_ND", "SLICE", "STRIDED_SLICE"}:
            continue
        perm = _read_transpose_perm(model_ir, op)
        input_layout = normalize_logical_layout(
            original_layouts.get(input_name, LOGICAL_LAYOUT_UNKNOWN)
        )
        output_layout = normalize_logical_layout(
            original_layouts.get(output_name, LOGICAL_LAYOUT_UNKNOWN)
        )
        remove_as_identity = bool(
            perm is not None
            and (
                perm == list(range(rank))
                or (
                    _is_layout_only_transpose_by_shape(
                        input_tensor=input_tensor,
                        output_tensor=output_tensor,
                        perm=perm,
                    )
                    and _is_standard_channel_layout_permutation(
                        perm=perm,
                        rank=rank,
                    )
                )
                or (
                    is_channel_last_logical_layout(input_layout)
                    and perm
                    == logical_layout_permutation(
                        source_layout=input_layout,
                        target_layout=channel_first_logical_layout(rank),
                    )
                )
                or (
                    is_channel_last_logical_layout(output_layout)
                    and perm
                    == logical_layout_permutation(
                        source_layout=channel_first_logical_layout(rank),
                        target_layout=output_layout,
                    )
                )
                or (
                    _is_inconsistent_standard_layout_transpose(
                        input_tensor=input_tensor,
                        output_tensor=output_tensor,
                        perm=perm,
                    )
                    and not reshape_only_consumers
                )
            )
        )
        if not remove_as_identity:
            continue
        if output_name in model_ir.outputs:
            source_tensor = input_tensor if input_tensor is not None else output_tensor
            if source_tensor is not None:
                replacement = _clone_tensor(source_tensor)
                replacement.name = output_name
                model_ir.tensors[output_name] = replacement
            graph_index.remove_operator(int(op_index))
            graph_index.insert_operator(
                int(op_index),
                OperatorIR("IDENTITY", [input_name], [output_name], {}),
            )
            changed = True
            continue
        for consumer in consumer_ops:
            consumer_index = graph_index.operator_index(consumer)
            if consumer_index is None:
                continue
            graph_index.replace_operator_inputs(
                int(consumer_index),
                [
                    input_name if str(value) == output_name else str(value)
                    for value in consumer.inputs
                ],
            )
        current_index = graph_index.operator_index(op)
        if current_index is not None:
            graph_index.remove_operator(int(current_index))
        model_ir.tensors.pop(output_name, None)
        changed = True

    if changed and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)


def _restore_same_average_pool_exclude_pad_correction_for_native_runtime(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    def _unique_tensor_name(base_name: str) -> str:
        candidate = str(base_name)
        suffix = 0
        while candidate in model_ir.tensors:
            suffix += 1
            candidate = f"{base_name}_{suffix}"
        return candidate

    def _tensor_nhwc_shape(tensor: Optional[TensorIR]) -> Optional[List[int]]:
        if tensor is None or len(tensor.shape) != 4:
            return None
        shape = [int(v) for v in list(tensor.shape)]
        layout = normalize_logical_layout(tensor.logical_layout)
        if is_channel_first_logical_layout(layout):
            return [int(shape[0]), int(shape[2]), int(shape[3]), int(shape[1])]
        if is_channel_last_logical_layout(layout):
            return [int(shape[0]), int(shape[1]), int(shape[2]), int(shape[3])]
        return None

    restored = 0
    graph_index = graph_index or ModelIRGraphIndex(model_ir)
    candidate_ops = [
        model_ir.operators[int(index)]
        for index in graph_index.operator_indices("AVERAGE_POOL_2D")
    ]
    for op in candidate_ops:
        op_index = graph_index.operator_index(op)
        if op_index is None or len(op.inputs) != 1 or len(op.outputs) != 1:
            continue
        if str(op.options.get("padding", "")).upper() != "SAME":
            continue
        output_name = str(op.outputs[0])
        if output_name in {str(v) for v in list(model_ir.outputs)}:
            pass
        if str(output_name).endswith("_include_pad"):
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        output_tensor = model_ir.tensors.get(output_name, None)
        input_nhwc_shape = _tensor_nhwc_shape(input_tensor)
        output_nhwc_shape = _tensor_nhwc_shape(output_tensor)
        if (
            input_tensor is None
            or output_tensor is None
            or input_nhwc_shape is None
            or output_nhwc_shape is None
        ):
            continue
        _, input_h, input_w, _ = input_nhwc_shape
        _, output_h, output_w, output_c = output_nhwc_shape
        kernel_h = int(op.options.get("filterHeight", 0))
        kernel_w = int(op.options.get("filterWidth", 0))
        stride_h = int(op.options.get("strideH", 0))
        stride_w = int(op.options.get("strideW", 0))
        output_dtype = str(output_tensor.dtype).upper()
        if (
            input_h <= 0
            or input_w <= 0
            or output_h <= 0
            or output_w <= 0
            or output_c <= 0
            or kernel_h <= 0
            or kernel_w <= 0
            or stride_h <= 0
            or stride_w <= 0
            or output_dtype not in {"FLOAT16", "FLOAT32"}
        ):
            continue
        total_pad_h = max((int(output_h) - 1) * int(stride_h) + int(kernel_h) - int(input_h), 0)
        total_pad_w = max((int(output_w) - 1) * int(stride_w) + int(kernel_w) - int(input_w), 0)
        if total_pad_h == 0 and total_pad_w == 0:
            continue
        pad_top = int(total_pad_h) // 2
        pad_left = int(total_pad_w) // 2
        correction_hw = np.ones((int(output_h), int(output_w), 1), dtype=np.float32)
        kernel_area = float(int(kernel_h) * int(kernel_w))
        for out_y in range(int(output_h)):
            start_y = int(out_y) * int(stride_h) - int(pad_top)
            end_y = int(start_y) + int(kernel_h)
            valid_h = max(min(end_y, int(input_h)) - max(start_y, 0), 0)
            for out_x in range(int(output_w)):
                start_x = int(out_x) * int(stride_w) - int(pad_left)
                end_x = int(start_x) + int(kernel_w)
                valid_w = max(min(end_x, int(input_w)) - max(start_x, 0), 0)
                valid_count = int(valid_h) * int(valid_w)
                if valid_count <= 0 or valid_count == int(kernel_h) * int(kernel_w):
                    continue
                correction_hw[out_y, out_x, 0] = float(kernel_area / float(valid_count))
        reciprocal_values = np.broadcast_to(
            correction_hw.reshape(1, int(output_h), int(output_w), 1),
            (1, int(output_h), int(output_w), int(output_c)),
        ).astype(np.float16 if output_dtype == "FLOAT16" else np.float32, copy=False)
        reciprocal_name = _unique_tensor_name(f"{output_name}_div_reciprocal")
        model_ir.tensors[reciprocal_name] = TensorIR(
            name=reciprocal_name,
            dtype=output_dtype,
            shape=[1, int(output_h), int(output_w), int(output_c)],
            shape_signature=[1, int(output_h), int(output_w), int(output_c)],
            data=np.asarray(reciprocal_values),
            logical_layout=normalize_logical_layout("NHWC"),
        )
        include_pad_name = _unique_tensor_name(f"{output_name}_include_pad")
        model_ir.tensors[include_pad_name] = TensorIR(
            name=include_pad_name,
            dtype=str(output_tensor.dtype),
            shape=[int(v) for v in list(output_tensor.shape)],
            shape_signature=(
                [int(v) for v in list(output_tensor.shape_signature)]
                if output_tensor.shape_signature is not None
                else [int(v) for v in list(output_tensor.shape)]
            ),
            quantization=copy.deepcopy(output_tensor.quantization),
            logical_layout=normalize_logical_layout(output_tensor.logical_layout),
        )
        correction_op = OperatorIR(
            op_type="MUL",
            inputs=[include_pad_name, reciprocal_name],
            outputs=[output_name],
            options={"fusedActivationFunction": "NONE"},
        )
        graph_index.replace_operator_outputs(
            int(op_index),
            [include_pad_name],
        )
        graph_index.insert_operator(int(op_index) + 1, correction_op)
        restored += 1
    if restored > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {
        "restored_same_average_pool_exclude_pad_corrections": int(restored),
    }
