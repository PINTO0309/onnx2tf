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
    _replace_operator_input_at,
    _set_operator_outputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


@dataclass(frozen=True)
class _ConstantRetargetPlan:
    operator: OperatorIR
    operator_index: int
    input_index: int
    old_name: str
    old_tensor: TensorIR
    new_name: Optional[str]
    quantized_values: np.ndarray
    quantization: Any
    shape: list[int]
    shape_signature: list[int]


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


def _has_exact_supported_grid(
    input_tensor: TensorIR,
    output_tensor: TensorIR,
) -> Optional[Tuple[str, float, int]]:
    dtype = str(input_tensor.dtype).upper()
    if dtype not in {"INT8", "UINT8"}:
        return None
    if str(output_tensor.dtype).upper() != dtype:
        return None
    input_grid = _get_per_tensor_scale_zero_point(input_tensor.quantization)
    output_grid = _get_per_tensor_scale_zero_point(output_tensor.quantization)
    if input_grid is None or output_grid is None:
        return None
    input_scale, input_zero_point = input_grid
    output_scale, output_zero_point = output_grid
    if not np.isfinite(input_scale) or float(input_scale) <= 0.0:
        return None
    if not np.isfinite(output_scale) or float(output_scale) <= 0.0:
        return None
    if float(input_scale) != float(output_scale):
        return None
    if int(input_zero_point) != int(output_zero_point):
        return None
    zero_point_min, zero_point_max = (-128, 127) if dtype == "INT8" else (0, 255)
    if not zero_point_min <= int(input_zero_point) <= zero_point_max:
        return None
    return dtype, float(input_scale), int(input_zero_point)


def _quantize_values_with_qparams(
    *,
    values: np.ndarray,
    dtype: str,
    scale: float,
    zero_point: int,
) -> Optional[np.ndarray]:
    values_f = np.asarray(values, dtype=np.float32)
    if not np.all(np.isfinite(values_f)):
        return None
    quantized = np.round(values_f / float(scale)) + int(zero_point)
    if dtype == "INT8":
        return np.asarray(np.clip(quantized, -128, 127), dtype=np.int8)
    if dtype == "UINT8":
        return np.asarray(np.clip(quantized, 0, 255), dtype=np.uint8)
    return None


def _unique_planned_tensor_name(base: str, reserved_names: set[str]) -> str:
    candidate = str(base)
    suffix = 1
    while candidate in reserved_names:
        candidate = f"{base}_{suffix}"
        suffix += 1
    reserved_names.add(candidate)
    return candidate


def _plan_constant_retarget(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex,
    operator: OperatorIR,
    operator_index: int,
    input_index: int,
    target_dtype: str,
    target_scale: float,
    target_zero_point: int,
    target_quantization: Any,
    public_boundaries: set[str],
    reserved_names: set[str],
) -> Optional[_ConstantRetargetPlan]:
    if input_index < 0 or input_index >= len(operator.inputs):
        return None
    old_name = str(operator.inputs[input_index])
    if old_name in graph_index.producers or old_name in graph_index.duplicate_producers:
        return None
    old_tensor = model_ir.tensors.get(old_name)
    if old_tensor is None or old_tensor.data is None:
        return None
    try:
        old_values = np.asarray(old_tensor.data, dtype=np.float32)
        shape = [int(value) for value in old_tensor.shape]
        shape_signature = (
            [int(value) for value in old_tensor.shape_signature]
            if old_tensor.shape_signature is not None
            else list(shape)
        )
    except (TypeError, ValueError):
        return None
    if int(old_values.size) != 1 or len(shape) != len(shape_signature):
        return None
    quantized_values = _quantize_values_with_qparams(
        values=old_values,
        dtype=target_dtype,
        scale=target_scale,
        zero_point=target_zero_point,
    )
    if quantized_values is None:
        return None
    try:
        quantization = _clone_quantization(target_quantization)
    except Exception:
        return None

    users = graph_index.consumer_indices(old_name)
    update_in_place = (
        users == [int(operator_index)] and old_name not in public_boundaries
    )
    new_name = None
    if not update_in_place:
        new_name = _unique_planned_tensor_name(
            f"{old_name}_q",
            reserved_names,
        )
    return _ConstantRetargetPlan(
        operator=operator,
        operator_index=int(operator_index),
        input_index=int(input_index),
        old_name=old_name,
        old_tensor=old_tensor,
        new_name=new_name,
        quantized_values=np.asarray(quantized_values),
        quantization=quantization,
        shape=shape,
        shape_signature=shape_signature,
    )


def _apply_constant_retarget(
    model_ir: ModelIR,
    *,
    graph_index: ModelIRGraphIndex,
    plan: _ConstantRetargetPlan,
    target_dtype: str,
) -> None:
    if plan.new_name is None:
        plan.old_tensor.data = np.asarray(plan.quantized_values)
        plan.old_tensor.dtype = str(target_dtype)
        plan.old_tensor.quantization = plan.quantization
        return
    model_ir.tensors[plan.new_name] = TensorIR(
        name=plan.new_name,
        dtype=str(target_dtype),
        shape=list(plan.shape),
        shape_signature=list(plan.shape_signature),
        data=np.asarray(plan.quantized_values),
        is_variable=False,
        quantization=plan.quantization,
    )
    _replace_operator_input_at(
        model_ir=model_ir,
        op=plan.operator,
        input_index=plan.input_index,
        new_input_name=plan.new_name,
        graph_index=graph_index,
    )


def _optimize_dequant_hardsigmoid_quantize_chains(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Fold an exact-grid expanded HardSigmoid QDQ chain transactionally."""

    stats_key = "folded_dequant_hardsigmoid_quantize_chains"
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    if active_index is None:
        required_types = {
            "DEQUANTIZE",
            "MUL",
            "ADD",
            "MAXIMUM",
            "MINIMUM",
            "QUANTIZE",
        }
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
        for operator_type in (
            "DEQUANTIZE",
            "MUL",
            "ADD",
            "MAXIMUM",
            "MINIMUM",
            "QUANTIZE",
        )
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
            multiply_index = int(float_input_consumers[0])
            if int(dequantize_index) >= multiply_index:
                continue
            multiply = model_ir.operators[multiply_index]
            if (
                str(multiply.op_type) != "MUL"
                or len(multiply.inputs) != 2
                or len(multiply.outputs) != 1
            ):
                continue
            multiply_inputs = [str(name) for name in multiply.inputs]
            if multiply_inputs[0] == float_input_name:
                multiply_data_input_index = 0
                alpha_input_index = 1
            elif multiply_inputs[1] == float_input_name:
                multiply_data_input_index = 1
                alpha_input_index = 0
            else:
                continue

            multiply_output_name = str(multiply.outputs[0])
            if multiply_output_name in public_boundaries:
                continue
            if multiply_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(multiply_output_name) != multiply_index:
                continue
            multiply_output_consumers = active_index.consumer_indices(
                multiply_output_name
            )
            if len(multiply_output_consumers) != 1:
                continue
            add_index = int(multiply_output_consumers[0])
            if multiply_index >= add_index:
                continue
            add = model_ir.operators[add_index]
            if (
                str(add.op_type) != "ADD"
                or len(add.inputs) != 2
                or len(add.outputs) != 1
            ):
                continue
            add_inputs = [str(name) for name in add.inputs]
            if add_inputs[0] == multiply_output_name:
                beta_input_index = 1
            elif add_inputs[1] == multiply_output_name:
                beta_input_index = 0
            else:
                continue

            add_output_name = str(add.outputs[0])
            if add_output_name in public_boundaries:
                continue
            if add_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(add_output_name) != add_index:
                continue
            add_output_consumers = active_index.consumer_indices(add_output_name)
            if len(add_output_consumers) != 1:
                continue
            maximum_index = int(add_output_consumers[0])
            if add_index >= maximum_index:
                continue
            maximum = model_ir.operators[maximum_index]
            if (
                str(maximum.op_type) != "MAXIMUM"
                or len(maximum.inputs) != 2
                or len(maximum.outputs) != 1
            ):
                continue
            maximum_inputs = [str(name) for name in maximum.inputs]
            if maximum_inputs[0] == add_output_name:
                low_input_index = 1
            elif maximum_inputs[1] == add_output_name:
                low_input_index = 0
            else:
                continue

            maximum_output_name = str(maximum.outputs[0])
            if maximum_output_name in public_boundaries:
                continue
            if maximum_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(maximum_output_name) != maximum_index:
                continue
            maximum_output_consumers = active_index.consumer_indices(
                maximum_output_name
            )
            if len(maximum_output_consumers) != 1:
                continue
            minimum_index = int(maximum_output_consumers[0])
            if maximum_index >= minimum_index:
                continue
            minimum = model_ir.operators[minimum_index]
            if (
                str(minimum.op_type) != "MINIMUM"
                or len(minimum.inputs) != 2
                or len(minimum.outputs) != 1
            ):
                continue
            minimum_inputs = [str(name) for name in minimum.inputs]
            if minimum_inputs[0] == maximum_output_name:
                high_input_index = 1
            elif minimum_inputs[1] == maximum_output_name:
                high_input_index = 0
            else:
                continue

            float_output_name = str(minimum.outputs[0])
            if float_output_name in public_boundaries:
                continue
            if float_output_name in active_index.duplicate_producers:
                continue
            if active_index.producers.get(float_output_name) != minimum_index:
                continue
            float_output_consumers = active_index.consumer_indices(float_output_name)
            if len(float_output_consumers) != 1:
                continue
            quantize_index = int(float_output_consumers[0])
            if minimum_index >= quantize_index:
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

            tensor_names = [
                quantized_input_name,
                float_input_name,
                multiply_output_name,
                add_output_name,
                maximum_output_name,
                float_output_name,
                quantized_output_name,
            ]
            tensors = [model_ir.tensors.get(name) for name in tensor_names]
            if any(tensor is None for tensor in tensors):
                continue
            (
                quantized_input,
                float_input,
                multiply_output,
                add_output,
                maximum_output,
                float_output,
                quantized_output,
            ) = tensors
            grid = _has_exact_supported_grid(
                quantized_input,
                quantized_output,
            )
            if grid is None:
                continue
            target_dtype, target_scale, target_zero_point = grid

            float_tensors = [
                float_input,
                multiply_output,
                add_output,
                maximum_output,
                float_output,
            ]
            float_dtype = str(float_input.dtype).upper()
            if not float_dtype.startswith("FLOAT") or any(
                str(tensor.dtype).upper() != float_dtype for tensor in float_tensors[1:]
            ):
                continue
            metadata = [_tensor_metadata(tensor) for tensor in tensors]
            if any(value is None for value in metadata):
                continue
            if any(value != metadata[0] for value in metadata[1:]):
                continue
            output_shape, output_signature = metadata[-2]

            constant_specs = [
                (multiply, multiply_index, alpha_input_index),
                (add, add_index, beta_input_index),
                (maximum, maximum_index, low_input_index),
                (minimum, minimum_index, high_input_index),
            ]
            reserved_names = {str(name) for name in model_ir.tensors}
            constant_plans: list[_ConstantRetargetPlan] = []
            for operator, operator_index, input_index in constant_specs:
                plan = _plan_constant_retarget(
                    model_ir,
                    graph_index=active_index,
                    operator=operator,
                    operator_index=operator_index,
                    input_index=input_index,
                    target_dtype=target_dtype,
                    target_scale=target_scale,
                    target_zero_point=target_zero_point,
                    target_quantization=quantized_input.quantization,
                    public_boundaries=public_boundaries,
                    reserved_names=reserved_names,
                )
                if plan is None:
                    constant_plans = []
                    break
                reconstructed = (
                    float(np.asarray(plan.quantized_values).reshape(-1)[0])
                    - float(target_zero_point)
                ) * float(target_scale)
                original = float(
                    np.asarray(plan.old_tensor.data, dtype=np.float32).reshape(-1)[0]
                )
                tolerance = max(float(target_scale) * 0.25, 1e-3)
                if abs(reconstructed - original) > tolerance:
                    constant_plans = []
                    break
                constant_plans.append(plan)
            if len(constant_plans) != 4:
                continue

            try:
                intermediate_quantizations = [
                    _clone_quantization(quantized_input.quantization) for _ in range(4)
                ]
            except Exception:
                continue

            # The complete topology, metadata, constant ownership, clone-name,
            # scalar representation, and quantization plan now exists.
            for plan in constant_plans:
                _apply_constant_retarget(
                    model_ir,
                    graph_index=active_index,
                    plan=plan,
                    target_dtype=target_dtype,
                )
            _replace_operator_input_at(
                model_ir=model_ir,
                op=multiply,
                input_index=multiply_data_input_index,
                new_input_name=quantized_input_name,
                graph_index=active_index,
            )
            _set_operator_outputs(
                model_ir=model_ir,
                op=minimum,
                new_outputs=[quantized_output_name],
                graph_index=active_index,
            )

            for tensor, quantization in zip(
                [
                    multiply_output,
                    add_output,
                    maximum_output,
                    quantized_output,
                ],
                intermediate_quantizations,
            ):
                tensor.dtype = str(target_dtype)
                tensor.quantization = quantization
            quantized_output.shape = list(output_shape)
            quantized_output.shape_signature = list(output_signature)
            active_index.remove_operators([int(dequantize_index), int(quantize_index)])

            folded += 1
            changed = True
            break

        if not changed:
            break

    _prune_unused_tensors(model_ir, layout_state=layout_state)
    if folded > 0 and layout_state is not None:
        layout_state.sync_from_model_ir(model_ir)
    return {stats_key: int(folded)}
