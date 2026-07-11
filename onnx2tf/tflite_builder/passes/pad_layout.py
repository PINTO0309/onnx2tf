from __future__ import annotations

from typing import Dict

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _append_tensor_lineage_event,
    _clone_quantization,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    candidate = str(base)
    suffix = 0
    while candidate in model_ir.tensors:
        suffix += 1
        candidate = f"{base}_{suffix}"
    return candidate


def repair_channel_last_inputs_for_channel_first_pad(model_ir: ModelIR) -> Dict[str, int]:
    """Insert an NHWC->NCHW adapter when a Pad retains its ONNX NCHW contract.

    Boundary Slice propagation can move the Pad input to NHWC after lowering,
    while the static output shape and paddings remain in ONNX NCHW order.  The
    repair is accepted only when the full input/output/padding shape equation
    proves that exact mismatch.  Names and model-specific patterns are not
    considered.
    """

    repaired = 0
    pad_index = 0
    while pad_index < len(model_ir.operators):
        pad_op = model_ir.operators[pad_index]
        if (
            str(pad_op.op_type) not in {"PAD", "PADV2", "MIRROR_PAD"}
            or len(pad_op.inputs) < 2
            or len(pad_op.outputs) != 1
        ):
            pad_index += 1
            continue

        input_name = str(pad_op.inputs[0])
        pads_name = str(pad_op.inputs[1])
        output_name = str(pad_op.outputs[0])
        input_tensor = model_ir.tensors.get(input_name, None)
        pads_tensor = model_ir.tensors.get(pads_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        if (
            input_tensor is None
            or pads_tensor is None
            or pads_tensor.data is None
            or output_tensor is None
            or str(input_tensor.logical_layout).upper() != "NHWC"
        ):
            pad_index += 1
            continue

        input_shape = [int(v) for v in list(input_tensor.shape)]
        output_shape = [int(v) for v in list(output_tensor.shape)]
        try:
            pad_pairs = np.asarray(pads_tensor.data, dtype=np.int64).reshape(4, 2)
        except (TypeError, ValueError):
            pad_index += 1
            continue
        if (
            len(input_shape) != 4
            or len(output_shape) != 4
            or any(int(v) <= 0 for v in input_shape + output_shape)
            or np.any(pad_pairs < 0)
        ):
            pad_index += 1
            continue

        expected_nchw_input = [
            int(output_shape[axis])
            - int(pad_pairs[axis, 0])
            - int(pad_pairs[axis, 1])
            for axis in range(4)
        ]
        expected_nhwc_input = [
            int(expected_nchw_input[0]),
            int(expected_nchw_input[2]),
            int(expected_nchw_input[3]),
            int(expected_nchw_input[1]),
        ]
        if (
            any(int(v) <= 0 for v in expected_nchw_input)
            or input_shape != expected_nhwc_input
            or input_shape == expected_nchw_input
        ):
            pad_index += 1
            continue

        adapter_name = _unique_tensor_name(model_ir, f"{output_name}_pad_input_nchw")
        perm_name = _unique_tensor_name(model_ir, f"{output_name}_pad_input_nchw_perm")
        input_signature = (
            [int(v) for v in list(input_tensor.shape_signature)]
            if input_tensor.shape_signature is not None
            else list(input_shape)
        )
        adapter_quantization = _clone_quantization(input_tensor.quantization)
        if isinstance(adapter_quantization, QuantParamIR):
            old_axis = int(adapter_quantization.quantized_dimension)
            if 0 <= old_axis < 4:
                adapter_quantization.quantized_dimension = int([0, 3, 1, 2].index(old_axis))

        model_ir.tensors[adapter_name] = TensorIR(
            name=adapter_name,
            dtype=str(input_tensor.dtype),
            shape=list(expected_nchw_input),
            shape_signature=[
                int(input_signature[0]),
                int(input_signature[3]),
                int(input_signature[1]),
                int(input_signature[2]),
            ],
            quantization=adapter_quantization,
            logical_layout="NCHW",
            physical_layout="NCHW",
            onnx_tensor_name=input_tensor.onnx_tensor_name,
        )
        model_ir.tensors[perm_name] = TensorIR(
            name=perm_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
            is_variable=False,
        )
        transpose_op = OperatorIR(
            op_type="TRANSPOSE",
            inputs=[input_name, perm_name],
            outputs=[adapter_name],
        )
        updated_inputs = [str(v) for v in list(pad_op.inputs)]
        updated_inputs[0] = adapter_name
        pad_op.inputs = updated_inputs
        model_ir.operators.insert(pad_index, transpose_op)
        _append_tensor_lineage_event(
            model_ir=model_ir,
            event={
                "kind": "replace_input",
                "operator_type": str(pad_op.op_type),
                "from": input_name,
                "to": adapter_name,
                "reason": "channel_last_input_for_channel_first_pad",
            },
        )
        repaired += 1
        pad_index += 2

    return {"repaired_channel_last_inputs_for_channel_first_pad": int(repaired)}
