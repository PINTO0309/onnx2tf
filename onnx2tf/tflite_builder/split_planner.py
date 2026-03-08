from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.model_writer import serialize_model, write_model_file


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
