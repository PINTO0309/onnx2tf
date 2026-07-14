from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _get_per_tensor_scale_zero_point,
    _prune_unused_tensors,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _tensor_metadata(
    tensor: Optional[TensorIR],
) -> Optional[Tuple[list[int], list[int]]]:
    if tensor is None:
        return None
    try:
        shape = [int(value) for value in tensor.shape]
        signature = (
            [int(value) for value in tensor.shape_signature]
            if tensor.shape_signature is not None
            else list(shape)
        )
    except (TypeError, ValueError):
        return None
    if len(shape) != len(signature):
        return None
    return shape, signature


def _has_supported_quantized_logistic_contract(
    input_tensor: TensorIR,
    output_tensor: TensorIR,
) -> bool:
    input_dtype = str(input_tensor.dtype).upper()
    if input_dtype not in {"INT8", "UINT8"}:
        return False
    if str(output_tensor.dtype).upper() != input_dtype:
        return False

    input_grid = _get_per_tensor_scale_zero_point(input_tensor.quantization)
    output_grid = _get_per_tensor_scale_zero_point(output_tensor.quantization)
    if input_grid is None or output_grid is None:
        return False
    input_scale, input_zero_point = input_grid
    output_scale, output_zero_point = output_grid
    if not np.isfinite(input_scale) or float(input_scale) <= 0.0:
        return False

    zero_point_min, zero_point_max = (-128, 127) if input_dtype == "INT8" else (0, 255)
    if not zero_point_min <= int(input_zero_point) <= zero_point_max:
        return False
    if float(output_scale) != 1.0 / 256.0:
        return False
    expected_output_zero_point = -128 if input_dtype == "INT8" else 0
    return int(output_zero_point) == expected_output_zero_point


def _optimize_dequant_logistic_quantize_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fold a canonical DQ→Logistic→Q chain into quantized Logistic."""

    stats_key = "folded_dequant_logistic_quantize_chains"
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {"DEQUANTIZE", "LOGISTIC", "QUANTIZE"}
        for operator in model_ir.operators:
            required_types.discard(str(operator.op_type))
            if not required_types:
                break
        if required_types:
            _prune_unused_tensors(model_ir, layout_state=layout_state)
            return {stats_key: 0}
        active_index = ModelIRGraphIndex(model_ir)
    elif any(
        not active_index.operator_indices(operator_type)
        for operator_type in ("DEQUANTIZE", "LOGISTIC", "QUANTIZE")
    ):
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        return {stats_key: 0}

    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs
    folded = 0

    while True:
        changed = False
        for dequantize_index in active_index.operator_indices("DEQUANTIZE"):
            dequantize = model_ir.operators[int(dequantize_index)]
            if len(dequantize.inputs) != 1 or len(dequantize.outputs) != 1:
                continue
            quantized_input_name = str(dequantize.inputs[0])
            float_input_name = str(dequantize.outputs[0])
            if float_input_name in public_boundaries:
                continue
            if float_input_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(float_input_name) != int(dequantize_index):
                continue

            float_input_consumers = active_index.consumer_indices(float_input_name)
            if len(float_input_consumers) != 1:
                continue
            logistic_index = int(float_input_consumers[0])
            if int(dequantize_index) >= logistic_index:
                continue
            logistic = model_ir.operators[logistic_index]
            if (
                str(logistic.op_type) != "LOGISTIC"
                or len(logistic.inputs) != 1
                or len(logistic.outputs) != 1
                or str(logistic.inputs[0]) != float_input_name
            ):
                continue

            float_output_name = str(logistic.outputs[0])
            if float_output_name in public_boundaries:
                continue
            if float_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(float_output_name) != logistic_index:
                continue
            float_output_consumers = active_index.consumer_indices(float_output_name)
            if len(float_output_consumers) != 1:
                continue
            quantize_index = int(float_output_consumers[0])
            if logistic_index >= quantize_index:
                continue
            quantize = model_ir.operators[quantize_index]
            if (
                str(quantize.op_type) != "QUANTIZE"
                or len(quantize.inputs) != 1
                or len(quantize.outputs) != 1
                or str(quantize.inputs[0]) != float_output_name
            ):
                continue

            quantized_output_name = str(quantize.outputs[0])
            if quantized_output_name in public_inputs:
                continue
            if quantized_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(quantized_output_name) != int(quantize_index):
                continue

            quantized_input = model_ir.tensors.get(quantized_input_name)
            float_input = model_ir.tensors.get(float_input_name)
            float_output = model_ir.tensors.get(float_output_name)
            quantized_output = model_ir.tensors.get(quantized_output_name)
            if (
                quantized_input is None
                or float_input is None
                or float_output is None
                or quantized_output is None
                or not _has_supported_quantized_logistic_contract(
                    quantized_input,
                    quantized_output,
                )
            ):
                continue
            if not str(float_input.dtype).upper().startswith("FLOAT"):
                continue
            if str(float_output.dtype).upper() != str(float_input.dtype).upper():
                continue

            quantized_input_metadata = _tensor_metadata(quantized_input)
            float_input_metadata = _tensor_metadata(float_input)
            float_output_metadata = _tensor_metadata(float_output)
            quantized_output_metadata = _tensor_metadata(quantized_output)
            if (
                quantized_input_metadata is None
                or float_input_metadata is None
                or float_output_metadata is None
                or quantized_output_metadata is None
            ):
                continue
            if not (
                quantized_input_metadata
                == float_input_metadata
                == float_output_metadata
                == quantized_output_metadata
            ):
                continue
            output_shape, output_signature = float_output_metadata

            # All grid, metadata, topology, and boundary guards are complete.
            # Retain the Logistic object/options and public output identity.
            _set_operator_inputs(
                model_ir=model_ir,
                op=logistic,
                new_inputs=[quantized_input_name],
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=logistic,
                new_outputs=[quantized_output_name],
                graph_index=active_index,
            )
            logistic.version = 2 if str(quantized_output.dtype).upper() == "INT8" else 1
            quantized_output.shape = list(output_shape)
            quantized_output.shape_signature = list(output_signature)
            active_index.remove_operators([int(dequantize_index), int(quantize_index)])

            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    return {stats_key: int(folded)}
