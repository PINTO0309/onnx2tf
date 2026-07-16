from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _get_per_tensor_scale_zero_point,
    _prune_unused_tensors,
    _quantize_tensor_per_tensor,
    _set_operator_inputs,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


@dataclass(frozen=True)
class _WeightPlan:
    source_name: str
    target_name: str
    source_tensor: TensorIR
    data: np.ndarray
    quantization: Any
    update_in_place: bool


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


def _has_valid_int8_grid(tensor: TensorIR) -> bool:
    if str(tensor.dtype).upper() != "INT8":
        return False
    grid = _get_per_tensor_scale_zero_point(tensor.quantization)
    if grid is None:
        return False
    scale, zero_point = grid
    return np.isfinite(scale) and float(scale) > 0.0 and -128 <= int(zero_point) <= 127


def _unique_tensor_name(base: str, existing_names: set[str]) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in existing_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    return candidate


def _plan_weight(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex,
    weight_name: str,
    transpose_conv_index: int,
    public_boundaries: set[str],
) -> Optional[_WeightPlan]:
    if (
        weight_name in graph_index.producers
        or weight_name in graph_index.duplicate_producers
    ):
        return None
    weight = model_ir.tensors.get(weight_name)
    if weight is None or not isinstance(weight.data, np.ndarray):
        return None
    data = np.asarray(weight.data)
    metadata = _tensor_metadata(weight)
    if (
        data.ndim != 4
        or metadata is None
        or len(metadata[0]) != 4
        or metadata[0] != [int(value) for value in data.shape]
        or metadata[1] != [int(value) for value in data.shape]
    ):
        return None

    weight_dtype = str(weight.dtype).upper()
    if weight_dtype == "INT8":
        if data.dtype != np.int8 or not _has_valid_int8_grid(weight):
            return None
        return _WeightPlan(
            source_name=weight_name,
            target_name=weight_name,
            source_tensor=weight,
            data=np.asarray(data),
            quantization=weight.quantization,
            update_in_place=False,
        )
    if weight_dtype not in {"FLOAT16", "FLOAT32", "FLOAT64"}:
        return None
    try:
        float_data = np.asarray(data, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if not np.all(np.isfinite(float_data)):
        return None
    try:
        quantized_data, quantization = _quantize_tensor_per_tensor(
            float_data,
            "INT8",
        )
    except Exception:
        return None
    quantized_data = np.asarray(quantized_data)
    if quantized_data.dtype != np.int8 or quantized_data.shape != data.shape:
        return None

    users = graph_index.consumer_indices(weight_name)
    update_in_place = (
        users == [int(transpose_conv_index)] and weight_name not in public_boundaries
    )
    target_name = weight_name
    if not update_in_place:
        target_name = _unique_tensor_name(
            f"{weight_name}_q",
            {str(name) for name in model_ir.tensors},
        )
    return _WeightPlan(
        source_name=weight_name,
        target_name=target_name,
        source_tensor=weight,
        data=quantized_data,
        quantization=quantization,
        update_in_place=update_in_place,
    )


def _apply_weight_plan(model_ir: ModelIR, plan: _WeightPlan) -> None:
    if plan.target_name == plan.source_name and not plan.update_in_place:
        return
    if plan.update_in_place:
        plan.source_tensor.data = np.asarray(plan.data)
        plan.source_tensor.dtype = "INT8"
        plan.source_tensor.shape = [int(value) for value in plan.data.shape]
        plan.source_tensor.shape_signature = [int(value) for value in plan.data.shape]
        plan.source_tensor.quantization = plan.quantization
        return
    model_ir.tensors[plan.target_name] = TensorIR(
        name=plan.target_name,
        dtype="INT8",
        shape=[int(value) for value in plan.data.shape],
        shape_signature=[int(value) for value in plan.data.shape],
        data=np.asarray(plan.data),
        is_variable=False,
        quantization=plan.quantization,
    )


def _optimize_dequant_transposeconv_quantize_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fold a planned DQ→TransposeConv→Q chain into quantized form."""

    stats_key = "folded_dequant_transposeconv_quantize_chains"
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {"DEQUANTIZE", "TRANSPOSE_CONV", "QUANTIZE"}
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
        for operator_type in ("DEQUANTIZE", "TRANSPOSE_CONV", "QUANTIZE")
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
            transpose_conv_index = int(float_input_consumers[0])
            if int(dequantize_index) >= transpose_conv_index:
                continue
            transpose_conv = model_ir.operators[transpose_conv_index]
            if (
                str(transpose_conv.op_type) != "TRANSPOSE_CONV"
                or len(transpose_conv.inputs) != 3
                or len(transpose_conv.outputs) != 1
                or str(transpose_conv.inputs[2]) != float_input_name
            ):
                continue
            output_shape_name = str(transpose_conv.inputs[0])
            weight_name = str(transpose_conv.inputs[1])
            if len({output_shape_name, weight_name, float_input_name}) != 3:
                continue
            if model_ir.tensors.get(output_shape_name) is None:
                continue

            float_output_name = str(transpose_conv.outputs[0])
            if float_output_name in public_boundaries:
                continue
            if float_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(float_output_name) != transpose_conv_index:
                continue
            float_output_consumers = active_index.consumer_indices(float_output_name)
            if len(float_output_consumers) != 1:
                continue
            quantize_index = int(float_output_consumers[0])
            if transpose_conv_index >= quantize_index:
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
            if active_index.producers.get(quantized_output_name) != quantize_index:
                continue

            quantized_input = model_ir.tensors.get(quantized_input_name)
            float_input = model_ir.tensors.get(float_input_name)
            float_output = model_ir.tensors.get(float_output_name)
            quantized_output = model_ir.tensors.get(quantized_output_name)
            if any(
                tensor is None
                for tensor in (
                    quantized_input,
                    float_input,
                    float_output,
                    quantized_output,
                )
            ):
                continue
            if not _has_valid_int8_grid(quantized_input) or not _has_valid_int8_grid(
                quantized_output
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
                or quantized_input_metadata != float_input_metadata
                or quantized_output_metadata != float_output_metadata
            ):
                continue
            output_shape, output_signature = float_output_metadata

            weight_plan = _plan_weight(
                model_ir,
                graph_index=active_index,
                weight_name=weight_name,
                transpose_conv_index=transpose_conv_index,
                public_boundaries=public_boundaries,
            )
            if weight_plan is None:
                continue
            try:
                output_quantization = _clone_quantization(quantized_output.quantization)
            except Exception:
                continue

            # Weight ownership/data/grid and output metadata are fully planned.
            _apply_weight_plan(model_ir, weight_plan)
            _set_operator_inputs(
                model_ir=model_ir,
                op=transpose_conv,
                new_inputs=[
                    output_shape_name,
                    weight_plan.target_name,
                    quantized_input_name,
                ],
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=transpose_conv,
                new_outputs=[quantized_output_name],
                graph_index=active_index,
            )
            transpose_conv.version = max(int(transpose_conv.version), 3)
            quantized_output.dtype = "INT8"
            quantized_output.quantization = output_quantization
            quantized_output.shape = list(output_shape)
            quantized_output.shape_signature = list(output_signature)
            active_index.remove_operators([int(dequantize_index), int(quantize_index)])

            folded += 1
            changed = True
            break

        if not changed:
            break

    if folded > 0:
        _prune_unused_tensors(model_ir, layout_state=layout_state)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(folded)}
