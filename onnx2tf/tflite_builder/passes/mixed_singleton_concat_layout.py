from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _permute_shape,
    _prune_unused_tensors,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


_NCHW_TO_NHWC = [0, 2, 3, 1]
_STATS_KEY = "repaired_mixed_singleton_nchw_inputs_for_nhwc_concat"


@dataclass(frozen=True)
class _AdapterPlan:
    adapter_name: str
    shape_name: str
    adapter_tensor: TensorIR
    shape_tensor: TensorIR
    operator: OperatorIR


@dataclass(frozen=True)
class _RepairPlan:
    concat: OperatorIR
    inputs: Tuple[str, ...]
    adapters: Tuple[_AdapterPlan, ...]


def _shape_contract(
    tensor: Optional[TensorIR],
) -> Optional[tuple[list[int], list[int]]]:
    if tensor is None:
        return None
    try:
        shape = [int(value) for value in tensor.shape]
        signature = (
            shape
            if tensor.shape_signature is None
            else [int(value) for value in tensor.shape_signature]
        )
    except (TypeError, ValueError):
        return None
    if len(shape) != 4 or len(signature) != 4:
        return None
    if any(int(value) <= 0 for value in shape) or any(
        int(signature_value) == 0
        or int(signature_value) < -1
        or (int(signature_value) > 0 and int(signature_value) != int(shape_value))
        for shape_value, signature_value in zip(shape, signature)
    ):
        return None
    return shape, signature


def _claim_name(reserved: set[str], base: str) -> str:
    name = str(base)
    suffix = 1
    while name in reserved:
        name = f"{base}_{suffix}"
        suffix += 1
    reserved.add(name)
    return name


def _candidate_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    concat_index: int,
    reserved_names: set[str],
) -> Optional[_RepairPlan]:
    concat = model_ir.operators[int(concat_index)]
    if (
        len(concat.inputs) < 2
        or len(concat.outputs) != 1
        or not isinstance(concat.options, dict)
    ):
        return None
    try:
        if int(concat.options.get("axis", 3)) != 3:
            return None
    except (TypeError, ValueError):
        return None

    output_name = str(concat.outputs[0])
    output_tensor = model_ir.tensors.get(output_name)
    target_contract = _shape_contract(output_tensor)
    model_inputs = {str(name) for name in model_ir.inputs}
    if (
        not output_name
        or target_contract is None
        or graph_index.producers.get(output_name) != int(concat_index)
        or output_name in graph_index.duplicate_producers
        or output_name in model_inputs
    ):
        return None
    target_shape, target_signature = target_contract
    if int(target_shape[3]) != len(concat.inputs) or int(target_signature[3]) != len(
        concat.inputs
    ):
        return None

    expected_input_shape = target_shape[:-1] + [1]
    expected_input_signature = target_signature[:-1] + [1]
    local_names = set(reserved_names)
    new_inputs: list[str] = []
    adapters: list[_AdapterPlan] = []
    adapters_by_source: dict[str, str] = {}
    needs_repair = False

    for raw_input_name in concat.inputs:
        input_name = str(raw_input_name)
        input_tensor = model_ir.tensors.get(input_name)
        input_contract = _shape_contract(input_tensor)
        if (
            not input_name
            or input_tensor is None
            or input_contract is None
            or str(input_tensor.dtype) != str(output_tensor.dtype)
            or input_name in graph_index.duplicate_producers
        ):
            return None
        input_shape, input_signature = input_contract
        producer_index = graph_index.producers.get(input_name)
        if producer_index is not None:
            if input_name in model_inputs or int(producer_index) >= int(concat_index):
                return None
        elif input_name not in model_inputs and input_tensor.data is None:
            return None

        if (
            input_shape == expected_input_shape
            and input_signature == expected_input_signature
        ):
            new_inputs.append(input_name)
            continue
        nhwc_shape = _permute_shape(input_shape, _NCHW_TO_NHWC)
        nhwc_signature = _permute_shape(input_signature, _NCHW_TO_NHWC)
        if (
            int(input_shape[1]) != 1
            or nhwc_shape is None
            or nhwc_signature is None
            or [int(value) for value in nhwc_shape] != expected_input_shape
            or [int(value) for value in nhwc_signature] != expected_input_signature
        ):
            return None

        existing_adapter = adapters_by_source.get(input_name)
        if existing_adapter is not None:
            new_inputs.append(existing_adapter)
            needs_repair = True
            continue

        adapter_name = _claim_name(
            local_names,
            f"{input_name}_nhwc_concat_adapter",
        )
        shape_name = _claim_name(
            local_names,
            f"{adapter_name}_reshape_shape",
        )
        try:
            source_quantization = _clone_quantization(input_tensor.quantization)
            adapter_quantization = _clone_quantization(source_quantization)
        except Exception:
            return None
        concrete_shape = [int(value) for value in nhwc_shape]
        adapter_signature = [int(value) for value in nhwc_signature]
        if adapter_signature.count(-1) > 1:
            return None
        reshape_shape = (
            list(adapter_signature) if -1 in adapter_signature else list(concrete_shape)
        )
        adapter_tensor = TensorIR(
            name=adapter_name,
            dtype=str(input_tensor.dtype),
            shape=list(concrete_shape),
            shape_signature=list(adapter_signature),
            data=None,
            is_variable=False,
            quantization=adapter_quantization,
        )
        shape_tensor = TensorIR(
            name=shape_name,
            dtype="INT32",
            shape=[len(concrete_shape)],
            shape_signature=[len(concrete_shape)],
            data=np.asarray(reshape_shape, dtype=np.int32),
            is_variable=False,
            quantization=None,
        )
        adapter = _AdapterPlan(
            adapter_name=adapter_name,
            shape_name=shape_name,
            adapter_tensor=adapter_tensor,
            shape_tensor=shape_tensor,
            operator=OperatorIR(
                op_type="RESHAPE",
                inputs=[input_name, shape_name],
                outputs=[adapter_name],
                options={"newShape": list(reshape_shape)},
            ),
        )
        adapters.append(adapter)
        adapters_by_source[input_name] = adapter_name
        new_inputs.append(adapter_name)
        needs_repair = True

    if not needs_repair:
        return None
    reserved_names.update(local_names)
    return _RepairPlan(
        concat=concat,
        inputs=tuple(new_inputs),
        adapters=tuple(adapters),
    )


def _apply_plan(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    plan: _RepairPlan,
) -> bool:
    concat_index = graph_index.operator_index(plan.concat)
    if concat_index is None:
        return False
    insert_index = int(concat_index)
    for adapter in plan.adapters:
        model_ir.tensors[adapter.shape_name] = adapter.shape_tensor
        model_ir.tensors[adapter.adapter_name] = adapter.adapter_tensor
        graph_index.insert_operator(insert_index, adapter.operator)
        insert_index += 1
    current_concat_index = graph_index.operator_index(plan.concat)
    if current_concat_index is None:
        return False
    _set_operator_inputs(
        model_ir=model_ir,
        op=plan.concat,
        new_inputs=list(plan.inputs),
        graph_index=graph_index,
    )
    return True


def _repair_mixed_singleton_nchw_inputs_for_nhwc_concat(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    if not any(
        str(operator.op_type) == "CONCATENATION" for operator in model_ir.operators
    ):
        return {_STATS_KEY: 0}
    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    reserved_names = {str(name) for name in model_ir.tensors}
    reserved_names.update(str(name) for name in model_ir.inputs + model_ir.outputs)
    for operator in model_ir.operators:
        reserved_names.update(str(name) for name in operator.inputs + operator.outputs)

    plans: list[_RepairPlan] = []
    for concat_index in active_index.operator_indices("CONCATENATION"):
        plan = _candidate_plan(
            model_ir,
            active_index,
            int(concat_index),
            reserved_names,
        )
        if plan is not None:
            plans.append(plan)

    repaired = sum(bool(_apply_plan(model_ir, active_index, plan)) for plan in plans)
    if repaired:
        _prune_unused_tensors(model_ir)
        if layout_state is not None:
            layout_state.sync_from_model_ir(model_ir)
    return {_STATS_KEY: int(repaired)}
