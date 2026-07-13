from __future__ import annotations

import copy
from typing import Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)


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
