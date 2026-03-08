from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    normalize_onnx_shape,
)
from onnx2tf.tflite_builder.model_writer import serialize_model, write_model_file
from onnx2tf.tflite_builder.tensor_buffer_builder import tflite_dtype_from_numpy


DEFAULT_TFLITE_SPLIT_MAX_BYTES = 1_073_741_824
DEFAULT_TFLITE_SPLIT_TARGET_BYTES = 1_060_000_000


@dataclass(frozen=True)
class PartitionRange:
    start_op_index: int
    end_op_index: int
    estimated_bytes: int


def _dtype_nbytes(dtype: str) -> int:
    table = {
        "FLOAT32": 4,
        "FLOAT16": 2,
        "FLOAT64": 8,
        "INT8": 1,
        "UINT8": 1,
        "INT16": 2,
        "UINT16": 2,
        "INT32": 4,
        "UINT32": 4,
        "INT64": 8,
        "UINT64": 8,
        "BOOL": 1,
    }
    return int(table.get(str(dtype).upper(), 4))


def estimate_tensor_bytes(tensor: TensorIR) -> int:
    if isinstance(tensor.data, np.ndarray):
        return int(tensor.data.nbytes)
    if not tensor.shape:
        return _dtype_nbytes(tensor.dtype)
    numel = 1
    for dim in tensor.shape:
        numel *= max(1, int(dim))
    return int(numel * _dtype_nbytes(tensor.dtype))


def estimate_ir_constant_buffer_bytes(model_ir: ModelIR) -> int:
    total = 0
    for tensor in model_ir.tensors.values():
        if isinstance(tensor.data, np.ndarray):
            total += int(tensor.data.nbytes)
    return int(total)


def estimate_model_flatbuffer_size_bytes(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
) -> int:
    return int(len(serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)))


def _validate_range(
    *,
    num_ops: int,
    start_op_index: int,
    end_op_index: int,
) -> None:
    if start_op_index < 0 or end_op_index < 0:
        raise ValueError("Partition indices must be non-negative.")
    if start_op_index >= end_op_index:
        raise ValueError(
            "Partition range must satisfy start < end. "
            f"got: start={start_op_index} end={end_op_index}"
        )
    if end_op_index > num_ops:
        raise ValueError(
            f"Partition end exceeds op count. end={end_op_index} num_ops={num_ops}"
        )


def _collect_outputs(operators: Sequence[OperatorIR]) -> List[str]:
    outputs: List[str] = []
    for op in operators:
        for output_name in op.outputs:
            if output_name and output_name not in outputs:
                outputs.append(output_name)
    return outputs


def _collect_inputs(operators: Sequence[OperatorIR]) -> List[str]:
    inputs: List[str] = []
    for op in operators:
        for input_name in op.inputs:
            if input_name and input_name not in inputs:
                inputs.append(input_name)
    return inputs


def _collect_required_operator_indices_for_outputs(
    *,
    operators: Sequence[OperatorIR],
    partition_outputs: Sequence[str],
) -> List[int]:
    producer_index: Dict[str, int] = {}
    for op_idx, op in enumerate(operators):
        for output_name in op.outputs:
            if output_name and output_name not in producer_index:
                producer_index[output_name] = int(op_idx)

    required_ops: Set[int] = set()
    pending_tensors: List[str] = [
        str(tensor_name) for tensor_name in partition_outputs if str(tensor_name) != ""
    ]
    visited_tensors: Set[str] = set()
    while pending_tensors:
        tensor_name = str(pending_tensors.pop())
        if tensor_name == "" or tensor_name in visited_tensors:
            continue
        visited_tensors.add(tensor_name)
        producer_op_idx = producer_index.get(tensor_name, None)
        if producer_op_idx is None:
            continue
        if producer_op_idx in required_ops:
            continue
        required_ops.add(int(producer_op_idx))
        for input_name in operators[producer_op_idx].inputs:
            if input_name:
                pending_tensors.append(str(input_name))
    return sorted(list(required_ops))


def _collect_partition_boundary_inputs(
    *,
    model_ir: ModelIR,
    consumed_tensor_names: Sequence[str],
    produced_tensor_names: Set[str],
) -> List[str]:
    runtime_input_names = {
        str(tensor_name)
        for tensor_name in model_ir.inputs
        if str(tensor_name) != ""
    }
    partition_inputs: List[str] = []
    seen_inputs: Set[str] = set()
    for tensor_name in consumed_tensor_names:
        normalized_name = str(tensor_name)
        if normalized_name == "":
            continue
        if normalized_name in seen_inputs or normalized_name in produced_tensor_names:
            continue
        seen_inputs.add(normalized_name)
        if normalized_name in runtime_input_names:
            partition_inputs.append(normalized_name)
            continue
        tensor = model_ir.tensors.get(normalized_name, None)
        if tensor is not None and tensor.data is not None:
            continue
        partition_inputs.append(normalized_name)
    return partition_inputs


def build_partition_model_ir(
    *,
    model_ir: ModelIR,
    start_op_index: int,
    end_op_index: int,
    partition_id: int,
) -> ModelIR:
    num_ops = len(model_ir.operators)
    _validate_range(
        num_ops=num_ops,
        start_op_index=start_op_index,
        end_op_index=end_op_index,
    )
    range_ops = model_ir.operators[start_op_index:end_op_index]
    produced_in_range = _collect_outputs(range_ops)
    produced_set = set(produced_in_range)
    consumed_in_range = _collect_inputs(range_ops)

    partition_inputs = _collect_partition_boundary_inputs(
        model_ir=model_ir,
        consumed_tensor_names=consumed_in_range,
        produced_tensor_names=produced_set,
    )
    consumed_after: Set[str] = set()
    for op in model_ir.operators[end_op_index:]:
        for input_name in op.inputs:
            if input_name in produced_set:
                consumed_after.add(input_name)

    partition_outputs: List[str] = []
    for name in produced_in_range:
        if name in consumed_after or name in model_ir.outputs:
            partition_outputs.append(name)
    if len(partition_outputs) == 0 and len(range_ops) > 0:
        for name in range_ops[-1].outputs:
            if name:
                partition_outputs.append(name)

    required_op_indices = _collect_required_operator_indices_for_outputs(
        operators=range_ops,
        partition_outputs=partition_outputs,
    )
    if len(required_op_indices) > 0:
        range_ops = [range_ops[op_idx] for op_idx in required_op_indices]
        produced_in_range = _collect_outputs(range_ops)
        produced_set = set(produced_in_range)
        consumed_in_range = _collect_inputs(range_ops)
        partition_inputs = _collect_partition_boundary_inputs(
            model_ir=model_ir,
            consumed_tensor_names=consumed_in_range,
            produced_tensor_names=produced_set,
        )
        partition_outputs = [
            name for name in partition_outputs if name in produced_set
        ]
        if len(partition_outputs) == 0 and len(range_ops) > 0:
            for name in range_ops[-1].outputs:
                if name:
                    partition_outputs.append(name)

    required_tensor_names: Set[str] = set(partition_inputs + partition_outputs)
    for op in range_ops:
        required_tensor_names.update([name for name in op.inputs if name])
        required_tensor_names.update([name for name in op.outputs if name])

    part_model = ModelIR(
        name=f"{model_ir.name}_part_{partition_id:04d}",
        description=f"{model_ir.description} (partition {partition_id})",
    )
    part_model.operators = [
        OperatorIR(
            op_type=op.op_type,
            inputs=list(op.inputs),
            outputs=list(op.outputs),
            options=dict(op.options),
            version=op.version,
        )
        for op in range_ops
    ]
    part_model.inputs = partition_inputs
    part_model.outputs = partition_outputs
    for tensor_name in required_tensor_names:
        if tensor_name not in model_ir.tensors:
            continue
        tensor = model_ir.tensors[tensor_name]
        part_model.tensors[tensor_name] = TensorIR(
            name=tensor.name,
            dtype=tensor.dtype,
            shape=list(tensor.shape),
            shape_signature=list(tensor.shape_signature)
            if tensor.shape_signature is not None
            else None,
            data=tensor.data.copy() if isinstance(tensor.data, np.ndarray) else tensor.data,
            is_variable=bool(tensor.is_variable),
            quantization=tensor.quantization,
        )
    return part_model


def _collect_nested_tensor_names(
    model_ir: ModelIR,
) -> Set[str]:
    nested_tensor_names: Set[str] = set()
    for subgraph in list(getattr(model_ir, "subgraphs", [])):
        for tensor_name in list(getattr(subgraph, "tensors", {}).keys()):
            normalized_name = str(tensor_name)
            if normalized_name != "":
                nested_tensor_names.add(normalized_name)
        nested_tensor_names.update(_collect_nested_tensor_names(subgraph))
    return nested_tensor_names


def crop_model_ir_by_boundary_tensors(
    *,
    model_ir: ModelIR,
    requested_inputs: Optional[Sequence[str]] = None,
    requested_outputs: Optional[Sequence[str]] = None,
) -> ModelIR:
    crop_error_prefix = "flatbuffer_direct ModelIR interrupt crop failed."
    top_level_tensor_names: Set[str] = {
        str(name)
        for name in list(model_ir.tensors.keys())
        if str(name) != ""
    }
    top_level_tensor_names.update(
        str(name)
        for name in list(model_ir.inputs)
        if str(name) != ""
    )
    top_level_tensor_names.update(
        str(name)
        for name in list(model_ir.outputs)
        if str(name) != ""
    )
    nested_tensor_names = _collect_nested_tensor_names(model_ir)

    boundary_inputs = [
        str(name)
        for name in (
            list(model_ir.inputs)
            if requested_inputs is None
            else list(requested_inputs)
        )
        if str(name) != ""
    ]
    boundary_outputs = [
        str(name)
        for name in (
            list(model_ir.outputs)
            if requested_outputs is None
            else list(requested_outputs)
        )
        if str(name) != ""
    ]
    boundary_inputs = list(dict.fromkeys(boundary_inputs))
    boundary_outputs = list(dict.fromkeys(boundary_outputs))

    if len(boundary_inputs) == 0:
        raise ValueError(
            f"{crop_error_prefix} requested input boundary list is empty."
        )
    if len(boundary_outputs) == 0:
        raise ValueError(
            f"{crop_error_prefix} requested output boundary list is empty."
        )

    producer_index: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in list(op.outputs):
            normalized_name = str(output_name)
            if normalized_name != "" and normalized_name not in producer_index:
                producer_index[normalized_name] = int(op_idx)

    for tensor_name in boundary_inputs + boundary_outputs:
        if tensor_name in top_level_tensor_names:
            continue
        if tensor_name in nested_tensor_names:
            raise ValueError(
                f"{crop_error_prefix} nested subgraph tensor names are unsupported. "
                f"name={tensor_name}"
            )
        raise ValueError(
            f"{crop_error_prefix} requested tensor was not found in top-level ModelIR tensors. "
            f"name={tensor_name}"
        )

    for tensor_name in boundary_inputs:
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None:
            continue
        if tensor_name in producer_index:
            continue
        if tensor_name in set(str(name) for name in model_ir.inputs):
            continue
        if tensor.data is not None:
            raise ValueError(
                f"{crop_error_prefix} constant tensors cannot be used as interrupt inputs. "
                f"name={tensor_name}"
            )

    always_available_tensors: Set[str] = set(boundary_inputs)
    for tensor_name, tensor in model_ir.tensors.items():
        normalized_name = str(tensor_name)
        if normalized_name != "" and tensor.data is not None:
            always_available_tensors.add(normalized_name)

    forward_reachable_ops: Set[int] = set()
    forward_available_tensors: Set[str] = set(always_available_tensors)
    for op_idx, op in enumerate(model_ir.operators):
        op_inputs = [str(name) for name in list(op.inputs) if str(name) != ""]
        if all(input_name in forward_available_tensors for input_name in op_inputs):
            forward_reachable_ops.add(int(op_idx))
            for output_name in list(op.outputs):
                normalized_name = str(output_name)
                if normalized_name != "":
                    forward_available_tensors.add(normalized_name)

    backward_required_ops: Set[int] = set()
    pending_tensors: List[str] = list(boundary_outputs)
    visited_tensors: Set[str] = set()
    while pending_tensors:
        tensor_name = str(pending_tensors.pop())
        if tensor_name == "" or tensor_name in visited_tensors:
            continue
        visited_tensors.add(tensor_name)
        if tensor_name in always_available_tensors:
            continue
        producer_op_idx = producer_index.get(tensor_name, None)
        if producer_op_idx is None:
            continue
        if producer_op_idx in backward_required_ops:
            continue
        backward_required_ops.add(int(producer_op_idx))
        for input_name in list(model_ir.operators[int(producer_op_idx)].inputs):
            normalized_name = str(input_name)
            if normalized_name != "":
                pending_tensors.append(normalized_name)

    kept_operator_indices = [
        int(op_idx)
        for op_idx in range(len(model_ir.operators))
        if int(op_idx) in forward_reachable_ops
        and int(op_idx) in backward_required_ops
    ]

    kept_output_tensors: Set[str] = set()
    for op_idx in kept_operator_indices:
        kept_output_tensors.update(
            str(name)
            for name in list(model_ir.operators[int(op_idx)].outputs)
            if str(name) != ""
        )

    missing_runtime_inputs: List[str] = []
    cropped_available_tensors: Set[str] = set(always_available_tensors)
    for op_idx in kept_operator_indices:
        op = model_ir.operators[int(op_idx)]
        for input_name in list(op.inputs):
            normalized_name = str(input_name)
            if normalized_name == "":
                continue
            if normalized_name not in cropped_available_tensors:
                missing_runtime_inputs.append(normalized_name)
        for output_name in list(op.outputs):
            normalized_name = str(output_name)
            if normalized_name != "":
                cropped_available_tensors.add(normalized_name)

    if len(missing_runtime_inputs) > 0:
        raise ValueError(
            f"{crop_error_prefix} requested boundaries require unavailable intermediate tensors. "
            f"missing_inputs={sorted(set(missing_runtime_inputs))}"
        )

    unreachable_outputs = [
        tensor_name
        for tensor_name in boundary_outputs
        if tensor_name not in cropped_available_tensors
    ]
    if len(unreachable_outputs) > 0:
        raise ValueError(
            f"{crop_error_prefix} requested outputs are not reachable from requested inputs. "
            f"outputs={sorted(set(unreachable_outputs))}"
        )

    required_tensor_names: Set[str] = set(boundary_inputs + boundary_outputs)
    for op_idx in kept_operator_indices:
        op = model_ir.operators[int(op_idx)]
        required_tensor_names.update(
            str(name) for name in list(op.inputs) if str(name) != ""
        )
        required_tensor_names.update(
            str(name) for name in list(op.outputs) if str(name) != ""
        )

    cropped_model_ir = ModelIR(
        name=str(model_ir.name),
        description=str(model_ir.description),
        metadata=copy.deepcopy(dict(model_ir.metadata)),
    )
    cropped_model_ir.inputs = list(boundary_inputs)
    cropped_model_ir.outputs = list(boundary_outputs)
    cropped_model_ir.subgraphs = copy.deepcopy(list(model_ir.subgraphs))
    cropped_model_ir.operators = [
        OperatorIR(
            op_type=str(model_ir.operators[int(op_idx)].op_type),
            inputs=list(model_ir.operators[int(op_idx)].inputs),
            outputs=list(model_ir.operators[int(op_idx)].outputs),
            options=copy.deepcopy(dict(model_ir.operators[int(op_idx)].options)),
            version=int(model_ir.operators[int(op_idx)].version),
        )
        for op_idx in kept_operator_indices
    ]
    for tensor_name in required_tensor_names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        cropped_model_ir.tensors[str(tensor_name)] = copy.deepcopy(tensor)
    return cropped_model_ir


class _ModelIRRewriteBuilder:
    def __init__(self, model_ir: ModelIR):
        self.model_ir = copy.deepcopy(model_ir)
        self._serial = 0
        self._aliases: Dict[str, str] = {}

    def remap_name(self, tensor_name: str) -> str:
        name = str(tensor_name)
        seen: Set[str] = set()
        while name in self._aliases and name not in seen:
            seen.add(name)
            name = str(self._aliases[name])
        return name

    def remap_inputs(self, inputs: Sequence[str]) -> List[str]:
        return [
            self.remap_name(str(input_name)) if str(input_name) != "" else ""
            for input_name in list(inputs)
        ]

    def alias_tensor(self, old_name: str, new_name: str) -> None:
        old_key = str(old_name).strip()
        new_key = str(new_name).strip()
        if old_key != "" and new_key != "" and old_key != new_key:
            self._aliases[old_key] = new_key

    def _next_name(self, base_name: str) -> str:
        base = str(base_name).strip() or "tensor"
        candidate = base
        while candidate in self.model_ir.tensors:
            self._serial += 1
            candidate = f"{base}_{self._serial}"
        return candidate

    def add_const_tensor(self, base_name: str, data: np.ndarray) -> str:
        name = self._next_name(base_name)
        data_array = np.asarray(data)
        dtype = tflite_dtype_from_numpy(data_array.dtype)
        shape, signature = normalize_onnx_shape(list(data_array.shape))
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=list(shape),
            shape_signature=list(signature),
            data=data_array,
            is_variable=False,
        )
        return name

    def add_tensor(
        self,
        *,
        base_name: str,
        dtype: str,
        shape: Sequence[int],
        shape_signature: Optional[Sequence[int]] = None,
        data: Optional[np.ndarray] = None,
        is_variable: bool = False,
        quantization: Any = None,
    ) -> str:
        name = self._next_name(base_name)
        norm_shape, norm_signature = normalize_onnx_shape(list(shape))
        if shape_signature is not None:
            signature = [int(v) for v in list(shape_signature)]
        else:
            signature = list(norm_signature)
        self.model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=str(dtype),
            shape=list(norm_shape),
            shape_signature=list(signature),
            data=data,
            is_variable=bool(is_variable),
            quantization=copy.deepcopy(quantization),
        )
        return name

    def clone_tensor(
        self,
        *,
        source_name: str,
        base_name: str,
        shape: Optional[Sequence[int]] = None,
        shape_signature: Optional[Sequence[int]] = None,
        data: Optional[np.ndarray] = None,
        is_variable: Optional[bool] = None,
    ) -> str:
        source_tensor = self.model_ir.tensors.get(str(source_name), None)
        if source_tensor is None:
            raise ValueError(f"ModelIR rewrite failed. tensor not found: {source_name}")
        cloned_data = (
            np.asarray(data)
            if data is not None
            else source_tensor.data.copy()
            if isinstance(source_tensor.data, np.ndarray)
            else source_tensor.data
        )
        return self.add_tensor(
            base_name=base_name,
            dtype=str(source_tensor.dtype),
            shape=list(source_tensor.shape if shape is None else shape),
            shape_signature=(
                list(source_tensor.shape_signature)
                if shape_signature is None and source_tensor.shape_signature is not None
                else list(source_tensor.shape if shape_signature is None else shape_signature)
            ),
            data=cloned_data,
            is_variable=(
                bool(source_tensor.is_variable)
                if is_variable is None
                else bool(is_variable)
            ),
            quantization=source_tensor.quantization,
        )


def _prune_unused_tensors_local(model_ir: ModelIR) -> None:
    used_tensor_names: Set[str] = set()
    used_tensor_names.update(str(name) for name in list(model_ir.inputs) if str(name) != "")
    used_tensor_names.update(str(name) for name in list(model_ir.outputs) if str(name) != "")
    for op in list(model_ir.operators):
        used_tensor_names.update(str(name) for name in list(op.inputs) if str(name) != "")
        used_tensor_names.update(str(name) for name in list(op.outputs) if str(name) != "")
    for tensor_name in list(model_ir.tensors.keys()):
        if str(tensor_name) not in used_tensor_names:
            del model_ir.tensors[tensor_name]


def _shape_signature(tensor: TensorIR) -> List[int]:
    if tensor.shape_signature is not None and len(tensor.shape_signature) == len(tensor.shape):
        return [int(v) for v in list(tensor.shape_signature)]
    return [int(v) for v in list(tensor.shape)]


def _require_static_positive_dims(
    *,
    tensor: TensorIR,
    context: str,
) -> List[int]:
    dims = _shape_signature(tensor)
    if len(dims) == 0 or any(int(v) <= 0 for v in dims):
        raise ValueError(
            f"{context} requires fully static positive shape. "
            f"tensor={tensor.name} shape_signature={dims}"
        )
    return [int(v) for v in dims]


def _copy_operator_with_remapped_inputs(
    op: OperatorIR,
    remapped_inputs: Sequence[str],
) -> OperatorIR:
    return OperatorIR(
        op_type=str(op.op_type),
        inputs=list(remapped_inputs),
        outputs=list(op.outputs),
        options=copy.deepcopy(dict(op.options)),
        version=int(op.version),
    )


def _build_concat_output_shape(
    *,
    input_shapes: Sequence[Sequence[int]],
    axis: int,
) -> List[int]:
    ref_shape = [int(v) for v in list(input_shapes[0])]
    out_shape = list(ref_shape)
    out_shape[int(axis)] = 0
    for shape in input_shapes:
        dims = [int(v) for v in list(shape)]
        out_shape[int(axis)] += int(dims[int(axis)])
    return out_shape


def _activation_op_type(name: str) -> Optional[str]:
    normalized = str(name).strip().upper()
    if normalized in {"TANH"}:
        return "TANH"
    if normalized in {"RELU"}:
        return "RELU"
    if normalized in {"SIGMOID", "LOGISTIC"}:
        return "LOGISTIC"
    if normalized in {"NONE", ""}:
        return None
    raise ValueError(
        "flatbuffer_direct recurrent unroll supports TANH/RELU/LOGISTIC/NONE only. "
        f"got: {name}"
    )


def _emit_activation(
    *,
    builder: _ModelIRRewriteBuilder,
    ops: List[OperatorIR],
    input_name: str,
    output_name: str,
    activation_name: str,
    dtype: str,
    shape: Sequence[int],
) -> str:
    op_type = _activation_op_type(activation_name)
    if op_type is None:
        if output_name != input_name:
            builder.model_ir.tensors[output_name] = copy.deepcopy(
                builder.model_ir.tensors[str(input_name)]
            )
        return str(input_name)
    builder.add_tensor(
        base_name=output_name,
        dtype=str(dtype),
        shape=list(shape),
        shape_signature=list(shape),
    )
    actual_output_name = output_name if output_name in builder.model_ir.tensors else builder._next_name(output_name)
    if actual_output_name not in builder.model_ir.tensors:
        builder.add_tensor(
            base_name=actual_output_name,
            dtype=str(dtype),
            shape=list(shape),
            shape_signature=list(shape),
        )
    ops.append(
        OperatorIR(
            op_type=op_type,
            inputs=[input_name],
            outputs=[actual_output_name],
            options={},
        )
    )
    return actual_output_name


def rewrite_model_ir_disable_group_convolution(
    *,
    model_ir: ModelIR,
) -> Tuple[ModelIR, int]:
    builder = _ModelIRRewriteBuilder(model_ir)
    rewritten_ops: List[OperatorIR] = []
    rewritten_count = 0

    for original_op in list(model_ir.operators):
        remapped_inputs = builder.remap_inputs(original_op.inputs)
        if str(original_op.op_type) != "CONV_2D" or len(remapped_inputs) < 2:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue

        input_tensor = builder.model_ir.tensors.get(str(remapped_inputs[0]), None)
        filter_tensor = builder.model_ir.tensors.get(str(remapped_inputs[1]), None)
        if input_tensor is None or filter_tensor is None:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        input_shape = _shape_signature(input_tensor)
        filter_shape = _shape_signature(filter_tensor)
        if len(input_shape) != 4 or len(filter_shape) != 4:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        in_channels = int(input_shape[3])
        filter_in_channels = int(filter_shape[3])
        out_channels = int(filter_shape[0])
        if in_channels <= 0 or filter_in_channels <= 0 or out_channels <= 0:
            raise ValueError(
                "flatbuffer_direct disable_group_convolution requires static positive CONV_2D channel dimensions."
            )
        if in_channels == filter_in_channels:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        if (in_channels % filter_in_channels) != 0:
            raise ValueError(
                "flatbuffer_direct disable_group_convolution could not infer grouped CONV_2D factor. "
                f"input_channels={in_channels} filter_input_channels={filter_in_channels}"
            )
        group = int(in_channels // filter_in_channels)
        if group <= 1 or (out_channels % group) != 0:
            raise ValueError(
                "flatbuffer_direct disable_group_convolution inferred an invalid grouped CONV_2D layout. "
                f"group={group} out_channels={out_channels}"
            )
        if not isinstance(filter_tensor.data, np.ndarray):
            raise ValueError(
                "flatbuffer_direct disable_group_convolution requires constant CONV_2D filters."
            )
        bias_name = str(remapped_inputs[2]) if len(remapped_inputs) >= 3 else ""
        bias_tensor = builder.model_ir.tensors.get(bias_name, None) if bias_name != "" else None
        if bias_name != "" and (bias_tensor is None or bias_tensor.data is None):
            raise ValueError(
                "flatbuffer_direct disable_group_convolution requires constant CONV_2D bias tensors."
            )
        bias_data = (
            np.asarray(bias_tensor.data)
            if bias_tensor is not None and isinstance(bias_tensor.data, np.ndarray)
            else np.zeros((out_channels,), dtype=np.float32)
        )
        if int(bias_data.size) != out_channels:
            raise ValueError(
                "flatbuffer_direct disable_group_convolution requires bias size to match output channels."
            )

        split_axis_name = builder.add_const_tensor(
            f"{original_op.outputs[0]}_group_axis",
            np.asarray(3, dtype=np.int32),
        )
        split_output_names: List[str] = []
        output_tensor = builder.model_ir.tensors.get(str(original_op.outputs[0]), None)
        output_shape = (
            _shape_signature(output_tensor)
            if output_tensor is not None
            else [input_shape[0], input_shape[1], input_shape[2], out_channels]
        )
        out_channels_per_group = int(out_channels // group)
        for group_idx in range(group):
            split_output_names.append(
                builder.add_tensor(
                    base_name=f"{original_op.outputs[0]}_group_input_{group_idx}",
                    dtype=str(input_tensor.dtype),
                    shape=[
                        int(input_shape[0]),
                        int(input_shape[1]),
                        int(input_shape[2]),
                        int(filter_in_channels),
                    ],
                    shape_signature=[
                        int(input_shape[0]),
                        int(input_shape[1]),
                        int(input_shape[2]),
                        int(filter_in_channels),
                    ],
                )
            )
        rewritten_ops.append(
            OperatorIR(
                op_type="SPLIT",
                inputs=[split_axis_name, remapped_inputs[0]],
                outputs=list(split_output_names),
                options={"numSplits": int(group)},
            )
        )

        group_output_names: List[str] = []
        original_options = copy.deepcopy(dict(original_op.options))
        final_activation = str(original_options.get("fusedActivationFunction", "NONE"))
        for group_idx in range(group):
            out_begin = int(group_idx * out_channels_per_group)
            out_end = int(out_begin + out_channels_per_group)
            group_filter_name = builder.clone_tensor(
                source_name=remapped_inputs[1],
                base_name=f"{original_op.outputs[0]}_group_filter_{group_idx}",
                shape=[out_channels_per_group, filter_shape[1], filter_shape[2], filter_in_channels],
                shape_signature=[out_channels_per_group, filter_shape[1], filter_shape[2], filter_in_channels],
                data=np.asarray(filter_tensor.data)[out_begin:out_end, :, :, :].copy(),
            )
            group_bias_name = builder.add_const_tensor(
                f"{original_op.outputs[0]}_group_bias_{group_idx}",
                np.asarray(bias_data[out_begin:out_end]).reshape(-1),
            )
            group_output_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_group_output_{group_idx}",
                dtype=str(output_tensor.dtype if output_tensor is not None else input_tensor.dtype),
                shape=[
                    int(output_shape[0]),
                    int(output_shape[1]),
                    int(output_shape[2]),
                    int(out_channels_per_group),
                ],
                shape_signature=[
                    int(output_shape[0]),
                    int(output_shape[1]),
                    int(output_shape[2]),
                    int(out_channels_per_group),
                ],
            )
            group_output_names.append(group_output_name)
            group_options = copy.deepcopy(original_options)
            group_options["fusedActivationFunction"] = "NONE"
            rewritten_ops.append(
                OperatorIR(
                    op_type="CONV_2D",
                    inputs=[split_output_names[group_idx], group_filter_name, group_bias_name],
                    outputs=[group_output_name],
                    options=group_options,
                    version=int(original_op.version),
                )
            )
        concat_options = {
            "axis": 3,
            "fusedActivationFunction": final_activation,
        }
        rewritten_ops.append(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=list(group_output_names),
                outputs=list(original_op.outputs),
                options=concat_options,
            )
        )
        rewritten_count += 1

    builder.model_ir.operators = rewritten_ops
    _prune_unused_tensors_local(builder.model_ir)
    return builder.model_ir, int(rewritten_count)


def rewrite_model_ir_unfold_batchmatmul(
    *,
    model_ir: ModelIR,
) -> Tuple[ModelIR, int]:
    builder = _ModelIRRewriteBuilder(model_ir)
    rewritten_ops: List[OperatorIR] = []
    rewritten_count = 0

    for original_op in list(model_ir.operators):
        remapped_inputs = builder.remap_inputs(original_op.inputs)
        if str(original_op.op_type) != "BATCH_MATMUL" or len(remapped_inputs) != 2 or len(original_op.outputs) != 1:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        lhs_tensor = builder.model_ir.tensors.get(str(remapped_inputs[0]), None)
        rhs_tensor = builder.model_ir.tensors.get(str(remapped_inputs[1]), None)
        out_tensor = builder.model_ir.tensors.get(str(original_op.outputs[0]), None)
        if lhs_tensor is None or rhs_tensor is None or out_tensor is None:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        lhs_shape = _require_static_positive_dims(
            tensor=lhs_tensor,
            context="flatbuffer_direct enable_batchmatmul_unfold",
        )
        rhs_shape = _require_static_positive_dims(
            tensor=rhs_tensor,
            context="flatbuffer_direct enable_batchmatmul_unfold",
        )
        out_shape = _require_static_positive_dims(
            tensor=out_tensor,
            context="flatbuffer_direct enable_batchmatmul_unfold",
        )
        if len(lhs_shape) < 3 or len(rhs_shape) < 3 or len(out_shape) < 3:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        lhs_batch_prefix = lhs_shape[:-2]
        rhs_batch_prefix = rhs_shape[:-2]
        out_batch_prefix = out_shape[:-2]
        if lhs_batch_prefix != rhs_batch_prefix or lhs_batch_prefix != out_batch_prefix:
            raise ValueError(
                "flatbuffer_direct enable_batchmatmul_unfold requires matching static batch prefixes. "
                f"lhs_shape={lhs_shape} rhs_shape={rhs_shape} out_shape={out_shape}"
            )
        batch_count = int(np.prod(np.asarray(lhs_batch_prefix, dtype=np.int64)))
        if batch_count <= 1:
            rewritten_ops.append(
                _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
            )
            continue
        m_dim = int(out_shape[-2])
        n_dim = int(out_shape[-1])
        k_dim_lhs = int(lhs_shape[-1])
        k_dim_rhs = int(rhs_shape[-2])
        if k_dim_lhs != k_dim_rhs:
            raise ValueError(
                "flatbuffer_direct enable_batchmatmul_unfold requires compatible contraction dimensions. "
                f"lhs_shape={lhs_shape} rhs_shape={rhs_shape}"
            )

        lhs_flat_shape = [batch_count, int(lhs_shape[-2]), int(lhs_shape[-1])]
        rhs_flat_shape = [batch_count, int(rhs_shape[-2]), int(rhs_shape[-1])]
        out_flat_shape = [batch_count, int(m_dim), int(n_dim)]
        lhs_flat_name = builder.add_tensor(
            base_name=f"{original_op.outputs[0]}_lhs_flat",
            dtype=str(lhs_tensor.dtype),
            shape=list(lhs_flat_shape),
            shape_signature=list(lhs_flat_shape),
        )
        rhs_flat_name = builder.add_tensor(
            base_name=f"{original_op.outputs[0]}_rhs_flat",
            dtype=str(rhs_tensor.dtype),
            shape=list(rhs_flat_shape),
            shape_signature=list(rhs_flat_shape),
        )
        lhs_flat_shape_name = builder.add_const_tensor(
            f"{original_op.outputs[0]}_lhs_flat_shape",
            np.asarray(lhs_flat_shape, dtype=np.int32),
        )
        rhs_flat_shape_name = builder.add_const_tensor(
            f"{original_op.outputs[0]}_rhs_flat_shape",
            np.asarray(rhs_flat_shape, dtype=np.int32),
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[remapped_inputs[0], lhs_flat_shape_name],
                outputs=[lhs_flat_name],
                options={"newShape": list(lhs_flat_shape)},
            )
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[remapped_inputs[1], rhs_flat_shape_name],
                outputs=[rhs_flat_name],
                options={"newShape": list(rhs_flat_shape)},
            )
        )
        expand_axis_name = builder.add_const_tensor(
            f"{original_op.outputs[0]}_expand_axis",
            np.asarray(0, dtype=np.int32),
        )
        batch_outputs_3d: List[str] = []
        for batch_idx in range(batch_count):
            lhs_slice_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_lhs_slice_{batch_idx}",
                dtype=str(lhs_tensor.dtype),
                shape=[1, lhs_shape[-2], lhs_shape[-1]],
                shape_signature=[1, lhs_shape[-2], lhs_shape[-1]],
            )
            rhs_slice_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_rhs_slice_{batch_idx}",
                dtype=str(rhs_tensor.dtype),
                shape=[1, rhs_shape[-2], rhs_shape[-1]],
                shape_signature=[1, rhs_shape[-2], rhs_shape[-1]],
            )
            lhs_begin_name = builder.add_const_tensor(
                f"{original_op.outputs[0]}_lhs_begin_{batch_idx}",
                np.asarray([batch_idx, 0, 0], dtype=np.int32),
            )
            rhs_begin_name = builder.add_const_tensor(
                f"{original_op.outputs[0]}_rhs_begin_{batch_idx}",
                np.asarray([batch_idx, 0, 0], dtype=np.int32),
            )
            lhs_size_name = builder.add_const_tensor(
                f"{original_op.outputs[0]}_lhs_size_{batch_idx}",
                np.asarray([1, lhs_shape[-2], lhs_shape[-1]], dtype=np.int32),
            )
            rhs_size_name = builder.add_const_tensor(
                f"{original_op.outputs[0]}_rhs_size_{batch_idx}",
                np.asarray([1, rhs_shape[-2], rhs_shape[-1]], dtype=np.int32),
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[lhs_flat_name, lhs_begin_name, lhs_size_name],
                    outputs=[lhs_slice_name],
                )
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="SLICE",
                    inputs=[rhs_flat_name, rhs_begin_name, rhs_size_name],
                    outputs=[rhs_slice_name],
                )
            )
            lhs_2d_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_lhs_2d_{batch_idx}",
                dtype=str(lhs_tensor.dtype),
                shape=[lhs_shape[-2], lhs_shape[-1]],
                shape_signature=[lhs_shape[-2], lhs_shape[-1]],
            )
            rhs_2d_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_rhs_2d_{batch_idx}",
                dtype=str(rhs_tensor.dtype),
                shape=[rhs_shape[-2], rhs_shape[-1]],
                shape_signature=[rhs_shape[-2], rhs_shape[-1]],
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[lhs_slice_name],
                    outputs=[lhs_2d_name],
                    options={"squeezeDims": [0]},
                )
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="SQUEEZE",
                    inputs=[rhs_slice_name],
                    outputs=[rhs_2d_name],
                    options={"squeezeDims": [0]},
                )
            )
            batch_output_2d_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_batch_out_{batch_idx}",
                dtype=str(out_tensor.dtype),
                shape=[m_dim, n_dim],
                shape_signature=[m_dim, n_dim],
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[lhs_2d_name, rhs_2d_name],
                    outputs=[batch_output_2d_name],
                    options=copy.deepcopy(dict(original_op.options)),
                    version=int(original_op.version),
                )
            )
            batch_output_3d_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_batch_out_3d_{batch_idx}",
                dtype=str(out_tensor.dtype),
                shape=[1, m_dim, n_dim],
                shape_signature=[1, m_dim, n_dim],
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="EXPAND_DIMS",
                    inputs=[batch_output_2d_name, expand_axis_name],
                    outputs=[batch_output_3d_name],
                )
            )
            batch_outputs_3d.append(batch_output_3d_name)
        out_flat_name = builder.add_tensor(
            base_name=f"{original_op.outputs[0]}_flat_out",
            dtype=str(out_tensor.dtype),
            shape=list(out_flat_shape),
            shape_signature=list(out_flat_shape),
        )
        concat_input_shapes = [
            builder.model_ir.tensors[name].shape for name in batch_outputs_3d
        ]
        if _build_concat_output_shape(input_shapes=concat_input_shapes, axis=0) != out_flat_shape:
            raise ValueError(
                "flatbuffer_direct enable_batchmatmul_unfold produced an inconsistent concatenation shape."
            )
        rewritten_ops.append(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=list(batch_outputs_3d),
                outputs=[out_flat_name],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            )
        )
        out_shape_name = builder.add_const_tensor(
            f"{original_op.outputs[0]}_out_shape",
            np.asarray(out_shape, dtype=np.int32),
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=[out_flat_name, out_shape_name],
                outputs=list(original_op.outputs),
                options={"newShape": list(out_shape)},
            )
        )
        rewritten_count += 1

    builder.model_ir.operators = rewritten_ops
    _prune_unused_tensors_local(builder.model_ir)
    return builder.model_ir, int(rewritten_count)


def _emit_unidirectional_sequence_rnn_unroll(
    *,
    builder: _ModelIRRewriteBuilder,
    op: OperatorIR,
    remapped_inputs: Sequence[str],
    rewritten_ops: List[OperatorIR],
    suffix: str,
    reverse_time: bool = False,
) -> Dict[str, str]:
    if len(remapped_inputs) < 5:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_RNN expects 5 inputs.")
    if not bool(op.options.get("timeMajor", True)):
        raise ValueError(
            "flatbuffer_direct enable_rnn_unroll currently requires timeMajor=True for UNIDIRECTIONAL_SEQUENCE_RNN."
        )
    x_name, w_name, r_name, b_name, h0_name = [str(v) for v in list(remapped_inputs[:5])]
    x_tensor = builder.model_ir.tensors.get(x_name, None)
    w_tensor = builder.model_ir.tensors.get(w_name, None)
    r_tensor = builder.model_ir.tensors.get(r_name, None)
    b_tensor = builder.model_ir.tensors.get(b_name, None)
    if x_tensor is None or w_tensor is None or r_tensor is None or b_tensor is None:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_RNN rewrite could not resolve required tensors.")
    x_shape = _require_static_positive_dims(
        tensor=x_tensor,
        context="flatbuffer_direct enable_rnn_unroll",
    )
    w_shape = _require_static_positive_dims(
        tensor=w_tensor,
        context="flatbuffer_direct enable_rnn_unroll",
    )
    r_shape = _require_static_positive_dims(
        tensor=r_tensor,
        context="flatbuffer_direct enable_rnn_unroll",
    )
    if len(x_shape) != 3 or len(w_shape) != 2 or len(r_shape) != 2:
        raise ValueError(
            "UNIDIRECTIONAL_SEQUENCE_RNN rewrite requires x=[seq,batch,input], w=[hidden,input], r=[hidden,hidden]."
        )
    seq_len, batch_size, input_size = [int(v) for v in x_shape]
    hidden_size = int(w_shape[0])
    if int(w_shape[1]) != input_size or r_shape != [hidden_size, hidden_size]:
        raise ValueError(
            "UNIDIRECTIONAL_SEQUENCE_RNN rewrite requires compatible tensor shapes. "
            f"x_shape={x_shape} w_shape={w_shape} r_shape={r_shape}"
        )
    if b_tensor.data is None:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_RNN rewrite requires constant bias tensor.")
    b_data = np.asarray(b_tensor.data).reshape(-1)
    if int(b_data.size) != hidden_size:
        raise ValueError(
            "UNIDIRECTIONAL_SEQUENCE_RNN rewrite requires bias size to match hidden size."
        )
    h_prev_name = str(h0_name)
    if h_prev_name == "":
        h_prev_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_h0_zero",
            np.zeros((batch_size, hidden_size), dtype=np.float32),
        )
    else:
        h0_tensor = builder.model_ir.tensors.get(h_prev_name, None)
        if h0_tensor is None:
            raise ValueError("UNIDIRECTIONAL_SEQUENCE_RNN rewrite could not resolve initial hidden state tensor.")
        h0_shape = _require_static_positive_dims(
            tensor=h0_tensor,
            context="flatbuffer_direct enable_rnn_unroll",
        )
        if h0_shape != [batch_size, hidden_size]:
            raise ValueError(
                "UNIDIRECTIONAL_SEQUENCE_RNN rewrite requires h0 shape [batch, hidden]. "
                f"h0_shape={h0_shape} expected={[batch_size, hidden_size]}"
            )
        if h0_tensor.data is None and bool(h0_tensor.is_variable):
            h_prev_name = builder.add_const_tensor(
                f"{op.outputs[0]}_{suffix}_h0_materialized",
                np.zeros((batch_size, hidden_size), dtype=np.float32),
            )
    activation_name = str(op.options.get("fusedActivationFunction", "TANH"))
    expand_axis_name = builder.add_const_tensor(
        f"{op.outputs[0]}_{suffix}_expand_axis",
        np.asarray(0, dtype=np.int32),
    )
    seq_outputs: List[str] = []
    for step in range(seq_len):
        time_index = int(seq_len - 1 - step) if reverse_time else int(step)
        x_slice_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_x_slice_{step}",
            dtype=str(x_tensor.dtype),
            shape=[1, batch_size, input_size],
            shape_signature=[1, batch_size, input_size],
        )
        begin_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_begin_{step}",
            np.asarray([time_index, 0, 0], dtype=np.int32),
        )
        size_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_size_{step}",
            np.asarray([1, batch_size, input_size], dtype=np.int32),
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="SLICE",
                inputs=[x_name, begin_name, size_name],
                outputs=[x_slice_name],
            )
        )
        x_t_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_x_t_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, input_size],
            shape_signature=[batch_size, input_size],
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[x_slice_name],
                outputs=[x_t_name],
                options={"squeezeDims": [0]},
            )
        )
        x_proj_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_x_proj_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, hidden_size],
            shape_signature=[batch_size, hidden_size],
        )
        h_proj_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_h_proj_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, hidden_size],
            shape_signature=[batch_size, hidden_size],
        )
        sum_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_sum_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, hidden_size],
            shape_signature=[batch_size, hidden_size],
        )
        pre_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_pre_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, hidden_size],
            shape_signature=[batch_size, hidden_size],
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[x_t_name, w_name],
                outputs=[x_proj_name],
                options={"adjX": False, "adjY": True},
            )
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="BATCH_MATMUL",
                inputs=[h_prev_name, r_name],
                outputs=[h_proj_name],
                options={"adjX": False, "adjY": True},
            )
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="ADD",
                inputs=[x_proj_name, h_proj_name],
                outputs=[sum_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="ADD",
                inputs=[sum_name, b_name],
                outputs=[pre_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        h_new_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_h_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, hidden_size],
            shape_signature=[batch_size, hidden_size],
        )
        rewritten_ops.append(
            OperatorIR(
                op_type=_activation_op_type(activation_name) or "IDENTITY",
                inputs=[pre_name],
                outputs=[h_new_name],
                options={},
            )
        )
        h_step_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_step_{step}",
            dtype=str(x_tensor.dtype),
            shape=[1, batch_size, hidden_size],
            shape_signature=[1, batch_size, hidden_size],
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[h_new_name, expand_axis_name],
                outputs=[h_step_name],
            )
        )
        seq_outputs.append(h_step_name)
        h_prev_name = h_new_name
    concat_output_name = str(op.outputs[0])
    rewritten_ops.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=list(seq_outputs if not reverse_time else list(reversed(seq_outputs))),
            outputs=[concat_output_name],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    alias_updates: Dict[str, str] = {}
    if h0_name != "":
        alias_updates[h0_name] = h_prev_name
    return alias_updates


def _emit_unidirectional_sequence_lstm_unroll(
    *,
    builder: _ModelIRRewriteBuilder,
    op: OperatorIR,
    remapped_inputs: Sequence[str],
    rewritten_ops: List[OperatorIR],
    suffix: str,
    reverse_time: bool = False,
) -> Dict[str, str]:
    if len(remapped_inputs) < 24:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM expects 24 inputs.")
    unsupported_optional_indices = [9, 10, 11, 20, 21, 22, 23]
    unsupported_inputs = [
        str(remapped_inputs[idx]).strip()
        for idx in unsupported_optional_indices
        if idx < len(remapped_inputs) and str(remapped_inputs[idx]).strip() != ""
    ]
    if len(unsupported_inputs) > 0:
        raise ValueError(
            "flatbuffer_direct enable_rnn_unroll does not support peephole/auxiliary UNIDIRECTIONAL_SEQUENCE_LSTM inputs. "
            f"inputs={unsupported_inputs}"
        )
    if not bool(op.options.get("timeMajor", True)):
        raise ValueError(
            "flatbuffer_direct enable_rnn_unroll currently requires timeMajor=True for UNIDIRECTIONAL_SEQUENCE_LSTM."
        )
    cell_clip = float(op.options.get("cellClip", 0.0))
    proj_clip = float(op.options.get("projClip", 0.0))
    if abs(cell_clip) > 1e-12 or abs(proj_clip) > 1e-12:
        raise ValueError(
            "flatbuffer_direct enable_rnn_unroll does not support non-zero cellClip/projClip for UNIDIRECTIONAL_SEQUENCE_LSTM."
        )
    x_name = str(remapped_inputs[0])
    x_tensor = builder.model_ir.tensors.get(x_name, None)
    if x_tensor is None:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM rewrite could not resolve input tensor.")
    x_shape = _require_static_positive_dims(
        tensor=x_tensor,
        context="flatbuffer_direct enable_rnn_unroll",
    )
    if len(x_shape) != 3:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM rewrite requires x=[seq,batch,input].")
    seq_len, batch_size, _ = [int(v) for v in x_shape]

    wi_name, wf_name, wc_name, wo_name = [str(remapped_inputs[idx]) for idx in [1, 2, 3, 4]]
    ri_name, rf_name, rc_name, ro_name = [str(remapped_inputs[idx]) for idx in [5, 6, 7, 8]]
    bi_name, bf_name, bc_name, bo_name = [str(remapped_inputs[idx]) for idx in [12, 13, 14, 15]]
    projection_weights_name = str(remapped_inputs[16]).strip()
    projection_bias_name = str(remapped_inputs[17]).strip()
    h0_name = str(remapped_inputs[18]).strip()
    c0_name = str(remapped_inputs[19]).strip()

    wi_tensor = builder.model_ir.tensors.get(wi_name, None)
    if wi_tensor is None:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM rewrite could not resolve gate weights.")
    wi_shape = _require_static_positive_dims(
        tensor=wi_tensor,
        context="flatbuffer_direct enable_rnn_unroll",
    )
    if len(wi_shape) != 2:
        raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM rewrite requires rank-2 gate weights.")
    hidden_size = int(wi_shape[0])

    h_prev_name = h0_name
    c_prev_name = c0_name
    state_shape = [batch_size, hidden_size]
    if h_prev_name == "":
        h_prev_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_h0_zero",
            np.zeros(tuple(state_shape), dtype=np.float32),
        )
    if c_prev_name == "":
        c_prev_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_c0_zero",
            np.zeros(tuple(state_shape), dtype=np.float32),
        )
    for state_name in [h_prev_name, c_prev_name]:
        state_tensor = builder.model_ir.tensors.get(state_name, None)
        if state_tensor is None:
            raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM rewrite could not resolve initial state tensor.")
        if _require_static_positive_dims(
            tensor=state_tensor,
            context="flatbuffer_direct enable_rnn_unroll",
        ) != state_shape:
            raise ValueError(
                "UNIDIRECTIONAL_SEQUENCE_LSTM rewrite requires state tensors with shape [batch, hidden]."
            )
    if h0_name != "":
        h0_tensor = builder.model_ir.tensors.get(h0_name, None)
        if h0_tensor is not None and h0_tensor.data is None and bool(h0_tensor.is_variable):
            h_prev_name = builder.add_const_tensor(
                f"{op.outputs[0]}_{suffix}_h0_materialized",
                np.zeros(tuple(state_shape), dtype=np.float32),
            )
    if c0_name != "":
        c0_tensor = builder.model_ir.tensors.get(c0_name, None)
        if c0_tensor is not None and c0_tensor.data is None and bool(c0_tensor.is_variable):
            c_prev_name = builder.add_const_tensor(
                f"{op.outputs[0]}_{suffix}_c0_materialized",
                np.zeros(tuple(state_shape), dtype=np.float32),
            )

    expand_axis_name = builder.add_const_tensor(
        f"{op.outputs[0]}_{suffix}_expand_axis",
        np.asarray(0, dtype=np.int32),
    )
    seq_outputs: List[str] = []
    activation_name = str(op.options.get("fusedActivationFunction", "TANH"))
    gate_specs = [
        ("i", wi_name, ri_name, bi_name),
        ("f", wf_name, rf_name, bf_name),
        ("g", wc_name, rc_name, bc_name),
        ("o", wo_name, ro_name, bo_name),
    ]
    for step in range(seq_len):
        time_index = int(seq_len - 1 - step) if reverse_time else int(step)
        x_slice_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_x_slice_{step}",
            dtype=str(x_tensor.dtype),
            shape=[1, batch_size, int(x_shape[2])],
            shape_signature=[1, batch_size, int(x_shape[2])],
        )
        begin_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_begin_{step}",
            np.asarray([time_index, 0, 0], dtype=np.int32),
        )
        size_name = builder.add_const_tensor(
            f"{op.outputs[0]}_{suffix}_size_{step}",
            np.asarray([1, batch_size, int(x_shape[2])], dtype=np.int32),
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="SLICE",
                inputs=[x_name, begin_name, size_name],
                outputs=[x_slice_name],
            )
        )
        x_t_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_x_t_{step}",
            dtype=str(x_tensor.dtype),
            shape=[batch_size, int(x_shape[2])],
            shape_signature=[batch_size, int(x_shape[2])],
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="SQUEEZE",
                inputs=[x_slice_name],
                outputs=[x_t_name],
                options={"squeezeDims": [0]},
            )
        )

        gate_outputs: Dict[str, str] = {}
        for gate_tag, w_name, r_name, b_name in gate_specs:
            x_proj_name = builder.add_tensor(
                base_name=f"{op.outputs[0]}_{suffix}_{gate_tag}_x_proj_{step}",
                dtype=str(x_tensor.dtype),
                shape=state_shape,
                shape_signature=state_shape,
            )
            h_proj_name = builder.add_tensor(
                base_name=f"{op.outputs[0]}_{suffix}_{gate_tag}_h_proj_{step}",
                dtype=str(x_tensor.dtype),
                shape=state_shape,
                shape_signature=state_shape,
            )
            sum_name = builder.add_tensor(
                base_name=f"{op.outputs[0]}_{suffix}_{gate_tag}_sum_{step}",
                dtype=str(x_tensor.dtype),
                shape=state_shape,
                shape_signature=state_shape,
            )
            pre_name = builder.add_tensor(
                base_name=f"{op.outputs[0]}_{suffix}_{gate_tag}_pre_{step}",
                dtype=str(x_tensor.dtype),
                shape=state_shape,
                shape_signature=state_shape,
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[x_t_name, w_name],
                    outputs=[x_proj_name],
                    options={"adjX": False, "adjY": True},
                )
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[h_prev_name, r_name],
                    outputs=[h_proj_name],
                    options={"adjX": False, "adjY": True},
                )
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="ADD",
                    inputs=[x_proj_name, h_proj_name],
                    outputs=[sum_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="ADD",
                    inputs=[sum_name, b_name],
                    outputs=[pre_name],
                    options={"fusedActivationFunction": "NONE"},
                )
            )
            gate_output_name = builder.add_tensor(
                base_name=f"{op.outputs[0]}_{suffix}_{gate_tag}_{step}",
                dtype=str(x_tensor.dtype),
                shape=state_shape,
                shape_signature=state_shape,
            )
            gate_activation = "TANH" if gate_tag == "g" else "LOGISTIC"
            rewritten_ops.append(
                OperatorIR(
                    op_type=gate_activation,
                    inputs=[pre_name],
                    outputs=[gate_output_name],
                    options={},
                )
            )
            gate_outputs[gate_tag] = gate_output_name

        forget_mul_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_forget_mul_{step}",
            dtype=str(x_tensor.dtype),
            shape=state_shape,
            shape_signature=state_shape,
        )
        input_mul_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_input_mul_{step}",
            dtype=str(x_tensor.dtype),
            shape=state_shape,
            shape_signature=state_shape,
        )
        c_new_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_c_{step}",
            dtype=str(x_tensor.dtype),
            shape=state_shape,
            shape_signature=state_shape,
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="MUL",
                inputs=[gate_outputs["f"], c_prev_name],
                outputs=[forget_mul_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="MUL",
                inputs=[gate_outputs["i"], gate_outputs["g"]],
                outputs=[input_mul_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="ADD",
                inputs=[forget_mul_name, input_mul_name],
                outputs=[c_new_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        c_activated_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_c_act_{step}",
            dtype=str(x_tensor.dtype),
            shape=state_shape,
            shape_signature=state_shape,
        )
        rewritten_ops.append(
            OperatorIR(
                op_type=_activation_op_type(activation_name) or "IDENTITY",
                inputs=[c_new_name],
                outputs=[c_activated_name],
                options={},
            )
        )
        h_pre_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_h_pre_{step}",
            dtype=str(x_tensor.dtype),
            shape=state_shape,
            shape_signature=state_shape,
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="MUL",
                inputs=[gate_outputs["o"], c_activated_name],
                outputs=[h_pre_name],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        h_new_name = h_pre_name
        if projection_weights_name != "":
            proj_tensor = builder.model_ir.tensors.get(projection_weights_name, None)
            if proj_tensor is None:
                raise ValueError("UNIDIRECTIONAL_SEQUENCE_LSTM rewrite could not resolve projection weights.")
            proj_shape = _require_static_positive_dims(
                tensor=proj_tensor,
                context="flatbuffer_direct enable_rnn_unroll",
            )
            proj_out_dim = int(proj_shape[0])
            projected_name = builder.add_tensor(
                base_name=f"{op.outputs[0]}_{suffix}_proj_{step}",
                dtype=str(x_tensor.dtype),
                shape=[batch_size, proj_out_dim],
                shape_signature=[batch_size, proj_out_dim],
            )
            rewritten_ops.append(
                OperatorIR(
                    op_type="BATCH_MATMUL",
                    inputs=[h_pre_name, projection_weights_name],
                    outputs=[projected_name],
                    options={"adjX": False, "adjY": True},
                )
            )
            h_new_name = projected_name
            if projection_bias_name != "":
                h_bias_name = builder.add_tensor(
                    base_name=f"{op.outputs[0]}_{suffix}_proj_bias_{step}",
                    dtype=str(x_tensor.dtype),
                    shape=[batch_size, proj_out_dim],
                    shape_signature=[batch_size, proj_out_dim],
                )
                rewritten_ops.append(
                    OperatorIR(
                        op_type="ADD",
                        inputs=[h_new_name, projection_bias_name],
                        outputs=[h_bias_name],
                        options={"fusedActivationFunction": "NONE"},
                    )
                )
                h_new_name = h_bias_name
        h_step_name = builder.add_tensor(
            base_name=f"{op.outputs[0]}_{suffix}_step_{step}",
            dtype=str(x_tensor.dtype),
            shape=[1, batch_size, int(builder.model_ir.tensors[h_new_name].shape[-1])],
            shape_signature=[1, batch_size, int(builder.model_ir.tensors[h_new_name].shape[-1])],
        )
        rewritten_ops.append(
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=[h_new_name, expand_axis_name],
                outputs=[h_step_name],
            )
        )
        seq_outputs.append(h_step_name)
        h_prev_name = h_new_name
        c_prev_name = c_new_name
    rewritten_ops.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=list(seq_outputs if not reverse_time else list(reversed(seq_outputs))),
            outputs=[str(op.outputs[0])],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    alias_updates: Dict[str, str] = {}
    if h0_name != "":
        alias_updates[h0_name] = h_prev_name
    if c0_name != "":
        alias_updates[c0_name] = c_prev_name
    return alias_updates


def rewrite_model_ir_unroll_recurrent_ops(
    *,
    model_ir: ModelIR,
) -> Tuple[ModelIR, int]:
    builder = _ModelIRRewriteBuilder(model_ir)
    rewritten_ops: List[OperatorIR] = []
    rewritten_count = 0

    for original_op in list(model_ir.operators):
        remapped_inputs = builder.remap_inputs(original_op.inputs)
        op_type = str(original_op.op_type)
        if op_type == "UNIDIRECTIONAL_SEQUENCE_RNN":
            alias_updates = _emit_unidirectional_sequence_rnn_unroll(
                builder=builder,
                op=original_op,
                remapped_inputs=remapped_inputs,
                rewritten_ops=rewritten_ops,
                suffix="uni_rnn",
                reverse_time=False,
            )
            for old_name, new_name in alias_updates.items():
                builder.alias_tensor(old_name, new_name)
            rewritten_count += 1
            continue
        if op_type == "UNIDIRECTIONAL_SEQUENCE_LSTM":
            alias_updates = _emit_unidirectional_sequence_lstm_unroll(
                builder=builder,
                op=original_op,
                remapped_inputs=remapped_inputs,
                rewritten_ops=rewritten_ops,
                suffix="uni_lstm",
                reverse_time=False,
            )
            for old_name, new_name in alias_updates.items():
                builder.alias_tensor(old_name, new_name)
            rewritten_count += 1
            continue
        if op_type == "BIDIRECTIONAL_SEQUENCE_LSTM":
            if len(remapped_inputs) < 48:
                raise ValueError("BIDIRECTIONAL_SEQUENCE_LSTM expects 48 inputs.")
            unsupported_optional_indices = [9, 10, 11, 26, 27, 28, 39, 40, 41, 42, 43, 44, 45, 46, 47]
            unsupported_inputs = [
                str(remapped_inputs[idx]).strip()
                for idx in unsupported_optional_indices
                if idx < len(remapped_inputs) and str(remapped_inputs[idx]).strip() != ""
            ]
            if len(unsupported_inputs) > 0:
                raise ValueError(
                    "flatbuffer_direct enable_rnn_unroll does not support peephole/auxiliary BIDIRECTIONAL_SEQUENCE_LSTM inputs. "
                    f"inputs={unsupported_inputs}"
                )
            if not bool(original_op.options.get("timeMajor", True)):
                raise ValueError(
                    "flatbuffer_direct enable_rnn_unroll currently requires timeMajor=True for BIDIRECTIONAL_SEQUENCE_LSTM."
                )
            merge_outputs = bool(original_op.options.get("mergeOutputs", True))
            fw_inputs = [
                str(remapped_inputs[0]),
                str(remapped_inputs[1]), str(remapped_inputs[2]), str(remapped_inputs[3]), str(remapped_inputs[4]),
                str(remapped_inputs[5]), str(remapped_inputs[6]), str(remapped_inputs[7]), str(remapped_inputs[8]),
                "", "", "",
                str(remapped_inputs[12]), str(remapped_inputs[13]), str(remapped_inputs[14]), str(remapped_inputs[15]),
                str(remapped_inputs[16]).strip(), str(remapped_inputs[17]).strip(),
                str(remapped_inputs[35]).strip(), str(remapped_inputs[36]).strip(),
                "", "", "", "",
            ]
            bw_inputs = [
                str(remapped_inputs[0]),
                str(remapped_inputs[18]), str(remapped_inputs[19]), str(remapped_inputs[20]), str(remapped_inputs[21]),
                str(remapped_inputs[22]), str(remapped_inputs[23]), str(remapped_inputs[24]), str(remapped_inputs[25]),
                "", "", "",
                str(remapped_inputs[29]), str(remapped_inputs[30]), str(remapped_inputs[31]), str(remapped_inputs[32]),
                str(remapped_inputs[33]).strip(), str(remapped_inputs[34]).strip(),
                str(remapped_inputs[37]).strip(), str(remapped_inputs[38]).strip(),
                "", "", "", "",
            ]
            fw_output_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_fw",
                dtype=str(builder.model_ir.tensors[str(remapped_inputs[0])].dtype),
                shape=[
                    int(v)
                    for v in _require_static_positive_dims(
                        tensor=builder.model_ir.tensors[str(remapped_inputs[0])],
                        context="flatbuffer_direct enable_rnn_unroll",
                    )[:-1]
                ]
                + [
                    int(
                        _require_static_positive_dims(
                            tensor=builder.model_ir.tensors[str(remapped_inputs[1])],
                            context="flatbuffer_direct enable_rnn_unroll",
                        )[0]
                    )
                ],
                shape_signature=[
                    int(v)
                    for v in _require_static_positive_dims(
                        tensor=builder.model_ir.tensors[str(remapped_inputs[0])],
                        context="flatbuffer_direct enable_rnn_unroll",
                    )[:-1]
                ]
                + [
                    int(
                        _require_static_positive_dims(
                            tensor=builder.model_ir.tensors[str(remapped_inputs[1])],
                            context="flatbuffer_direct enable_rnn_unroll",
                        )[0]
                    )
                ],
            )
            bw_output_name = builder.add_tensor(
                base_name=f"{original_op.outputs[0]}_bw",
                dtype=str(builder.model_ir.tensors[str(remapped_inputs[0])].dtype),
                shape=list(builder.model_ir.tensors[fw_output_name].shape),
                shape_signature=list(_shape_signature(builder.model_ir.tensors[fw_output_name])),
            )
            fw_aliases = _emit_unidirectional_sequence_lstm_unroll(
                builder=builder,
                op=OperatorIR(
                    op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
                    inputs=list(fw_inputs),
                    outputs=[fw_output_name],
                    options=copy.deepcopy(dict(original_op.options)),
                    version=int(original_op.version),
                ),
                remapped_inputs=list(fw_inputs),
                rewritten_ops=rewritten_ops,
                suffix="bilstm_fw",
                reverse_time=False,
            )
            bw_aliases = _emit_unidirectional_sequence_lstm_unroll(
                builder=builder,
                op=OperatorIR(
                    op_type="UNIDIRECTIONAL_SEQUENCE_LSTM",
                    inputs=list(bw_inputs),
                    outputs=[bw_output_name],
                    options=copy.deepcopy(dict(original_op.options)),
                    version=int(original_op.version),
                ),
                remapped_inputs=list(bw_inputs),
                rewritten_ops=rewritten_ops,
                suffix="bilstm_bw",
                reverse_time=True,
            )
            if merge_outputs:
                rewritten_ops.append(
                    OperatorIR(
                        op_type="CONCATENATION",
                        inputs=[fw_output_name, bw_output_name],
                        outputs=list(original_op.outputs),
                        options={"axis": 2, "fusedActivationFunction": "NONE"},
                    )
                )
            else:
                expand_axis_name = builder.add_const_tensor(
                    f"{original_op.outputs[0]}_bilstm_expand_axis",
                    np.asarray(2, dtype=np.int32),
                )
                fw_expanded = builder.add_tensor(
                    base_name=f"{original_op.outputs[0]}_fw_expanded",
                    dtype=str(builder.model_ir.tensors[fw_output_name].dtype),
                    shape=list(builder.model_ir.tensors[fw_output_name].shape[:2]) + [1, int(builder.model_ir.tensors[fw_output_name].shape[2])],
                    shape_signature=list(_shape_signature(builder.model_ir.tensors[fw_output_name])[:2]) + [1, int(_shape_signature(builder.model_ir.tensors[fw_output_name])[2])],
                )
                bw_expanded = builder.add_tensor(
                    base_name=f"{original_op.outputs[0]}_bw_expanded",
                    dtype=str(builder.model_ir.tensors[bw_output_name].dtype),
                    shape=list(builder.model_ir.tensors[bw_output_name].shape[:2]) + [1, int(builder.model_ir.tensors[bw_output_name].shape[2])],
                    shape_signature=list(_shape_signature(builder.model_ir.tensors[bw_output_name])[:2]) + [1, int(_shape_signature(builder.model_ir.tensors[bw_output_name])[2])],
                )
                rewritten_ops.append(
                    OperatorIR(
                        op_type="EXPAND_DIMS",
                        inputs=[fw_output_name, expand_axis_name],
                        outputs=[fw_expanded],
                    )
                )
                rewritten_ops.append(
                    OperatorIR(
                        op_type="EXPAND_DIMS",
                        inputs=[bw_output_name, expand_axis_name],
                        outputs=[bw_expanded],
                    )
                )
                rewritten_ops.append(
                    OperatorIR(
                        op_type="CONCATENATION",
                        inputs=[fw_expanded, bw_expanded],
                        outputs=list(original_op.outputs),
                        options={"axis": 2, "fusedActivationFunction": "NONE"},
                    )
                )
            for alias_map in [fw_aliases, bw_aliases]:
                for old_name, new_name in alias_map.items():
                    builder.alias_tensor(old_name, new_name)
            rewritten_count += 1
            continue
        rewritten_ops.append(
            _copy_operator_with_remapped_inputs(original_op, remapped_inputs)
        )

    builder.model_ir.operators = rewritten_ops
    _prune_unused_tensors_local(builder.model_ir)
    return builder.model_ir, int(rewritten_count)


def find_dependency_safe_split_points(model_ir: ModelIR) -> List[Dict[str, Any]]:
    op_count = len(model_ir.operators)
    if op_count <= 1:
        return []
    producer_index: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for out_name in op.outputs:
            if out_name and out_name not in producer_index:
                producer_index[out_name] = op_idx

    points: List[Dict[str, Any]] = []
    for boundary in range(1, op_count):
        valid = True
        crossing_tensors: Set[str] = set()
        for op_idx, op in enumerate(model_ir.operators):
            for input_name in op.inputs:
                if not input_name:
                    continue
                producer = producer_index.get(input_name)
                if producer is None:
                    continue
                if op_idx < boundary and producer >= boundary:
                    valid = False
                    break
                if op_idx >= boundary and producer < boundary:
                    crossing_tensors.add(input_name)
            if not valid:
                break
        if valid:
            points.append(
                {
                    "index": int(boundary),
                    "crossing_count": int(len(crossing_tensors)),
                    "crossing_tensors": sorted(list(crossing_tensors)),
                }
            )
    return points


def validate_partition_ranges(
    *,
    model_ir: ModelIR,
    partition_ranges: Sequence[Tuple[int, int]],
) -> None:
    op_count = len(model_ir.operators)
    if len(partition_ranges) == 0:
        raise ValueError("partition_ranges must not be empty.")
    expected_start = 0
    producer_index: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for out_name in op.outputs:
            if out_name and out_name not in producer_index:
                producer_index[out_name] = op_idx

    for part_idx, (start_op_index, end_op_index) in enumerate(partition_ranges):
        _validate_range(
            num_ops=op_count,
            start_op_index=start_op_index,
            end_op_index=end_op_index,
        )
        if start_op_index != expected_start:
            raise ValueError(
                "Partition ranges must be contiguous and ordered. "
                f"expected_start={expected_start} actual={start_op_index}"
            )
        expected_start = end_op_index
        for op_idx in range(start_op_index, end_op_index):
            op = model_ir.operators[op_idx]
            for input_name in op.inputs:
                if not input_name:
                    continue
                producer = producer_index.get(input_name)
                if producer is None:
                    continue
                if producer > op_idx:
                    raise ValueError(
                        "Topological dependency violation detected in graph. "
                        f"tensor={input_name} producer={producer} consumer={op_idx}"
                    )
                if producer >= end_op_index:
                    raise ValueError(
                        "Partition dependency violation detected. "
                        f"partition={part_idx} tensor={input_name} "
                        f"producer={producer} end={end_op_index}"
                    )

    if expected_start != op_count:
        raise ValueError(
            "Partition ranges do not cover all operators. "
            f"covered_until={expected_start} op_count={op_count}"
        )


def _build_partition_edges(
    *,
    model_ir: ModelIR,
    partition_ranges: Sequence[PartitionRange],
) -> List[Dict[str, Any]]:
    op_to_part: Dict[int, int] = {}
    for part_idx, partition in enumerate(partition_ranges):
        for op_idx in range(partition.start_op_index, partition.end_op_index):
            op_to_part[op_idx] = part_idx

    producer_to_part: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        part_idx = op_to_part[op_idx]
        for out_name in op.outputs:
            if out_name and out_name not in producer_to_part:
                producer_to_part[out_name] = part_idx

    edge_map: Dict[Tuple[int, int], Set[str]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        dst_part = op_to_part[op_idx]
        for input_name in op.inputs:
            src_part = producer_to_part.get(input_name)
            if src_part is None or src_part >= dst_part:
                continue
            key = (src_part, dst_part)
            if key not in edge_map:
                edge_map[key] = set()
            edge_map[key].add(input_name)

    edges: List[Dict[str, Any]] = []
    for (src_part, dst_part), tensors in sorted(edge_map.items()):
        edges.append(
            {
                "from_partition": int(src_part),
                "to_partition": int(dst_part),
                "tensors": sorted(list(tensors)),
            }
        )
    return edges


def _estimate_partition_size(
    *,
    partition_model_ir: ModelIR,
    schema_tflite: Optional[Dict[str, Any]],
    size_estimator: Optional[Callable[[ModelIR], int]],
) -> int:
    if size_estimator is not None:
        return int(size_estimator(partition_model_ir))
    if schema_tflite is None:
        raise ValueError(
            "schema_tflite is required when size_estimator is not provided."
        )
    return estimate_model_flatbuffer_size_bytes(
        schema_tflite=schema_tflite,
        model_ir=partition_model_ir,
    )


def plan_contiguous_partitions_by_size(
    *,
    model_ir: ModelIR,
    target_max_bytes: int = DEFAULT_TFLITE_SPLIT_TARGET_BYTES,
    hard_max_bytes: int = DEFAULT_TFLITE_SPLIT_MAX_BYTES,
    schema_tflite: Optional[Dict[str, Any]] = None,
    size_estimator: Optional[Callable[[ModelIR], int]] = None,
) -> Dict[str, Any]:
    target_max_bytes = int(target_max_bytes)
    hard_max_bytes = int(hard_max_bytes)
    if target_max_bytes <= 0 or hard_max_bytes <= 0:
        raise ValueError(
            "target_max_bytes and hard_max_bytes must be > 0. "
            f"target={target_max_bytes} hard={hard_max_bytes}"
        )
    if target_max_bytes > hard_max_bytes:
        raise ValueError(
            "target_max_bytes must be <= hard_max_bytes. "
            f"target={target_max_bytes} hard={hard_max_bytes}"
        )

    total_estimated_bytes = _estimate_partition_size(
        partition_model_ir=model_ir,
        schema_tflite=schema_tflite,
        size_estimator=size_estimator,
    )
    candidate_split_points = find_dependency_safe_split_points(model_ir)

    op_count = len(model_ir.operators)
    if op_count == 0:
        return {
            "schema_version": 1,
            "target_max_bytes": int(target_max_bytes),
            "hard_max_bytes": int(hard_max_bytes),
            "total_estimated_bytes": int(total_estimated_bytes),
            "constant_buffer_bytes": int(estimate_ir_constant_buffer_bytes(model_ir)),
            "candidate_split_points": candidate_split_points,
            "partitions": [],
            "edges": [],
            "plan_valid": True,
        }

    partition_ranges: List[PartitionRange] = []
    start_op_index = 0
    partition_id = 1
    while start_op_index < op_count:
        low = start_op_index + 1
        high = op_count
        best_end: Optional[int] = None
        best_size: Optional[int] = None

        while low <= high:
            mid = (low + high) // 2
            part_model = build_partition_model_ir(
                model_ir=model_ir,
                start_op_index=start_op_index,
                end_op_index=mid,
                partition_id=partition_id,
            )
            estimated_size = _estimate_partition_size(
                partition_model_ir=part_model,
                schema_tflite=schema_tflite,
                size_estimator=size_estimator,
            )
            if estimated_size <= target_max_bytes:
                best_end = mid
                best_size = estimated_size
                low = mid + 1
            else:
                high = mid - 1

        if best_end is None:
            end_op_index = start_op_index + 1
            part_model = build_partition_model_ir(
                model_ir=model_ir,
                start_op_index=start_op_index,
                end_op_index=end_op_index,
                partition_id=partition_id,
            )
            estimated_size = _estimate_partition_size(
                partition_model_ir=part_model,
                schema_tflite=schema_tflite,
                size_estimator=size_estimator,
            )
            if estimated_size > hard_max_bytes:
                raise ValueError(
                    "Single partition exceeds hard_max_bytes. "
                    f"partition={partition_id} size={estimated_size} "
                    f"hard_max_bytes={hard_max_bytes}"
                )
        else:
            end_op_index = best_end
            if best_size is None:
                raise ValueError("Internal error: best_size is None while best_end is resolved.")
            estimated_size = int(best_size)

        partition_ranges.append(
            PartitionRange(
                start_op_index=int(start_op_index),
                end_op_index=int(end_op_index),
                estimated_bytes=int(estimated_size),
            )
        )
        start_op_index = end_op_index
        partition_id += 1

    validate_partition_ranges(
        model_ir=model_ir,
        partition_ranges=[
            (part.start_op_index, part.end_op_index)
            for part in partition_ranges
        ],
    )
    for part in partition_ranges:
        if part.estimated_bytes > hard_max_bytes:
            raise ValueError(
                "Estimated partition size exceeds hard_max_bytes. "
                f"range=({part.start_op_index},{part.end_op_index}) "
                f"size={part.estimated_bytes} hard_max_bytes={hard_max_bytes}"
            )

    edges = _build_partition_edges(
        model_ir=model_ir,
        partition_ranges=partition_ranges,
    )
    return {
        "schema_version": 1,
        "target_max_bytes": int(target_max_bytes),
        "hard_max_bytes": int(hard_max_bytes),
        "total_estimated_bytes": int(total_estimated_bytes),
        "constant_buffer_bytes": int(estimate_ir_constant_buffer_bytes(model_ir)),
        "candidate_split_points": candidate_split_points,
        "partitions": [
            {
                "partition_id": idx + 1,
                "start_op_index": int(part.start_op_index),
                "end_op_index": int(part.end_op_index),
                "estimated_bytes": int(part.estimated_bytes),
            }
            for idx, part in enumerate(partition_ranges)
        ],
        "edges": edges,
        "plan_valid": True,
    }


def write_split_plan_report(
    *,
    report: Dict[str, Any],
    output_report_path: str,
) -> str:
    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return output_report_path


def should_split_by_estimate(plan_report: Dict[str, Any]) -> bool:
    total_estimated_bytes = int(plan_report.get("total_estimated_bytes", 0))
    target_max_bytes = int(plan_report.get("target_max_bytes", 0))
    return total_estimated_bytes > target_max_bytes


def write_split_model_files_and_manifest(
    *,
    schema_tflite: Dict[str, Any],
    model_ir: ModelIR,
    plan_report: Dict[str, Any],
    output_folder_path: str,
    output_file_name: str,
    tflite_loader_validator: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    partitions = list(plan_report.get("partitions", []))
    edges = list(plan_report.get("edges", []))
    generated_partition_paths: List[str] = []
    generated_partition_entries: List[Dict[str, Any]] = []

    for part in partitions:
        partition_id = int(part["partition_id"])
        start_op_index = int(part["start_op_index"])
        end_op_index = int(part["end_op_index"])
        part_model_ir = build_partition_model_ir(
            model_ir=model_ir,
            start_op_index=start_op_index,
            end_op_index=end_op_index,
            partition_id=partition_id,
        )
        split_file_name = f"{output_file_name}_{partition_id:04d}.tflite"
        split_file_path = os.path.join(output_folder_path, split_file_name)
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=part_model_ir,
            output_tflite_path=split_file_path,
        )
        if tflite_loader_validator is not None:
            tflite_loader_validator(split_file_path)

        generated_partition_paths.append(split_file_path)
        generated_partition_entries.append(
            {
                "partition_id": partition_id,
                "file": split_file_name,
                "start_op_index": start_op_index,
                "end_op_index": end_op_index,
                "estimated_bytes": int(part.get("estimated_bytes", 0)),
                "inputs": list(part_model_ir.inputs),
                "outputs": list(part_model_ir.outputs),
            }
        )

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "base_model": f"{output_file_name}.tflite",
        "target_max_bytes": int(plan_report.get("target_max_bytes", 0)),
        "hard_max_bytes": int(plan_report.get("hard_max_bytes", 0)),
        "total_estimated_bytes": int(plan_report.get("total_estimated_bytes", 0)),
        "partitions": generated_partition_entries,
        "edges": edges,
    }
    manifest_path = os.path.join(output_folder_path, f"{output_file_name}_split_manifest.json")
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    return {
        "split_manifest_path": manifest_path,
        "split_partition_paths": generated_partition_paths,
        "split_partition_count": len(generated_partition_paths),
    }
