from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _is_fully_known_positive_shape,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


@dataclass(frozen=True)
class UnboundInputLayoutRepairResult:
    repaired: int
    graph_index: Optional[ModelIRGraphIndex]


@dataclass(frozen=True)
class _UnboundInputCandidate:
    consumer: OperatorIR
    input_index: int
    tensor_name: str


def _collect_unbound_input_candidates(
    model_ir: ModelIR,
    *,
    producer_names: set[str],
) -> List[_UnboundInputCandidate]:
    model_inputs = {str(name) for name in model_ir.inputs}
    candidates: List[_UnboundInputCandidate] = []
    for op in model_ir.operators:
        for input_index, raw_input_name in enumerate(op.inputs):
            input_name = str(raw_input_name)
            if not input_name:
                continue
            if input_name in model_inputs or input_name in producer_names:
                continue
            tensor = model_ir.tensors.get(input_name)
            if tensor is not None and tensor.data is not None:
                continue
            candidates.append(
                _UnboundInputCandidate(
                    consumer=op,
                    input_index=int(input_index),
                    tensor_name=input_name,
                )
            )
    return candidates


def find_unbound_nonconstant_operator_inputs(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> List[Dict[str, Any]]:
    """Report nonconstant operator inputs without a producer or public input."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    producer_names = (
        set(active_index.producers)
        if active_index is not None
        else {
            str(output_name)
            for op in model_ir.operators
            for output_name in op.outputs
        }
    )
    operator_indices = {
        id(op): int(op_index)
        for op_index, op in enumerate(model_ir.operators)
    }
    candidates = _collect_unbound_input_candidates(
        model_ir,
        producer_names=producer_names,
    )
    return [
        {
            "op_index": int(operator_indices[id(candidate.consumer)]),
            "op_type": str(candidate.consumer.op_type),
            "input_index": int(candidate.input_index),
            "tensor_name": str(candidate.tensor_name),
        }
        for candidate in candidates
    ]


def _unique_tensor_name(model_ir: ModelIR, base: str) -> str:
    name = str(base)
    serial = 1
    while name in model_ir.tensors:
        name = f"{base}_{serial}"
        serial += 1
    return name


def _expected_nhwc_source_shape(orphan_shape: List[int]) -> List[int]:
    return [
        int(orphan_shape[0]),
        int(orphan_shape[2]),
        int(orphan_shape[3]),
        int(orphan_shape[1]),
    ]


def _select_dequantize_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    tensor_name: str,
    orphan_tensor: TensorIR,
    expected_source_shape: List[int],
    consumer_index: int,
) -> Optional[str]:
    source_name: Optional[str] = None
    source_index: Optional[int] = None
    exact_source_name = f"{tensor_name}_nhwc_bridge"
    exact_source_index = graph_index.producers.get(exact_source_name)
    if exact_source_index is not None:
        source_name = exact_source_name
        source_index = int(exact_source_index)
    else:
        for candidate_name, candidate_index in graph_index.producers.items():
            normalized_name = str(candidate_name)
            normalized_index = int(candidate_index)
            if normalized_index >= int(consumer_index):
                continue
            if not (
                normalized_name.endswith("_nhwc")
                or normalized_name.endswith("_nhwc_bridge")
            ):
                continue
            candidate_op = model_ir.operators[normalized_index]
            candidate_tensor = model_ir.tensors.get(normalized_name)
            if (
                str(candidate_op.op_type) != "ADD"
                or candidate_tensor is None
                or candidate_tensor.data is not None
                or str(candidate_tensor.dtype) != str(orphan_tensor.dtype)
                or [int(value) for value in candidate_tensor.shape]
                != expected_source_shape
            ):
                continue
            if source_index is None or normalized_index > source_index:
                source_name = normalized_name
                source_index = normalized_index
    source_tensor = model_ir.tensors.get(source_name) if source_name else None
    if (
        source_index is None
        or source_name is None
        or source_index >= int(consumer_index)
        or source_tensor is None
        or source_tensor.data is not None
        or str(source_tensor.dtype) != str(orphan_tensor.dtype)
        or [int(value) for value in source_tensor.shape] != expected_source_shape
    ):
        return None
    return source_name


def _select_nearest_shape_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    expected_source_shape: List[int],
    consumer_index: int,
) -> Optional[str]:
    best_source_name: Optional[str] = None
    best_source_index = -1
    for produced_name, source_index in graph_index.producers.items():
        normalized_index = int(source_index)
        if normalized_index >= int(consumer_index):
            continue
        source_tensor = model_ir.tensors.get(str(produced_name))
        if source_tensor is None or source_tensor.data is not None:
            continue
        if not _is_fully_known_positive_shape(source_tensor.shape):
            continue
        if [int(value) for value in source_tensor.shape] != expected_source_shape:
            continue
        if normalized_index > best_source_index:
            best_source_index = normalized_index
            best_source_name = str(produced_name)
    return best_source_name


def _select_mul_alias_source(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    orphan_dtype: str,
    expected_source_shape: List[int],
    consumer_index: int,
) -> Optional[str]:
    best_source_name: Optional[str] = None
    best_source_index = -1
    for produced_name, source_index in graph_index.producers.items():
        normalized_index = int(source_index)
        if normalized_index >= int(consumer_index):
            continue
        source_name = str(produced_name)
        if not source_name.endswith("_input_nhwc"):
            continue
        source_op = model_ir.operators[normalized_index]
        source_tensor = model_ir.tensors.get(source_name)
        if (
            str(source_op.op_type) != "ADD"
            or source_tensor is None
            or source_tensor.data is not None
            or not _is_fully_known_positive_shape(source_tensor.shape)
            or [int(value) for value in source_tensor.shape]
            != expected_source_shape
            or str(source_tensor.dtype) != orphan_dtype
        ):
            continue
        if normalized_index > best_source_index:
            best_source_index = normalized_index
            best_source_name = source_name
    return best_source_name


def _insert_layout_repair_transpose(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
    *,
    consumer_index: int,
    source_name: str,
    tensor_name: str,
) -> None:
    perm_name = _unique_tensor_name(model_ir, f"{tensor_name}_repair_perm")
    model_ir.tensors[perm_name] = TensorIR(
        name=perm_name,
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    graph_index.insert_operator(
        int(consumer_index),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=[str(source_name), perm_name],
            outputs=[str(tensor_name)],
            options={},
        ),
    )


def repair_unbound_nonconstant_inputs_with_layout_transpose(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> UnboundInputLayoutRepairResult:
    """Insert conservative NHWC-to-NCHW bridges for known orphan families."""

    active_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else None
    )
    producer_names = (
        set(active_index.producers)
        if active_index is not None
        else {
            str(output_name)
            for op in model_ir.operators
            for output_name in op.outputs
        }
    )
    candidates = _collect_unbound_input_candidates(
        model_ir,
        producer_names=producer_names,
    )
    if not candidates:
        return UnboundInputLayoutRepairResult(
            repaired=0,
            graph_index=active_index,
        )
    if active_index is None:
        active_index = ModelIRGraphIndex(model_ir)
    repaired = 0
    for candidate in candidates:
        tensor_name = str(candidate.tensor_name)
        if active_index.producer(tensor_name) is not None:
            continue
        consumer_index = active_index.operator_index(candidate.consumer)
        if consumer_index is None:
            continue
        input_index = int(candidate.input_index)
        consumer_op = candidate.consumer
        if (
            input_index < 0
            or input_index >= len(consumer_op.inputs)
            or str(consumer_op.inputs[input_index]) != tensor_name
        ):
            continue
        consumer_type = str(consumer_op.op_type)
        orphan_tensor = model_ir.tensors.get(tensor_name)

        if consumer_type == "DEQUANTIZE" and input_index == 0:
            if (
                orphan_tensor is not None
                and orphan_tensor.data is None
                and _is_fully_known_positive_shape(orphan_tensor.shape)
                and len(orphan_tensor.shape) == 4
            ):
                expected_shape = _expected_nhwc_source_shape(
                    [int(value) for value in orphan_tensor.shape]
                )
                source_name = _select_dequantize_source(
                    model_ir,
                    active_index,
                    tensor_name=tensor_name,
                    orphan_tensor=orphan_tensor,
                    expected_source_shape=expected_shape,
                    consumer_index=int(consumer_index),
                )
                if source_name is not None:
                    _insert_layout_repair_transpose(
                        model_ir,
                        active_index,
                        consumer_index=int(consumer_index),
                        source_name=source_name,
                        tensor_name=tensor_name,
                    )
                    source_tensor = model_ir.tensors.get(source_name)
                    if source_tensor is not None:
                        orphan_tensor.quantization = _clone_quantization(
                            source_tensor.quantization
                        )
                    repaired += 1
                    continue

        target_input_index = 1 if consumer_type == "SPLIT" else 0
        if (
            consumer_type in {"RESHAPE", "SHAPE", "SPLIT"}
            and input_index == target_input_index
        ):
            if orphan_tensor is None or not _is_fully_known_positive_shape(
                orphan_tensor.shape
            ):
                continue
            orphan_shape = [int(value) for value in orphan_tensor.shape]
            if len(orphan_shape) != 4:
                continue
            source_name = _select_nearest_shape_source(
                model_ir,
                active_index,
                expected_source_shape=_expected_nhwc_source_shape(orphan_shape),
                consumer_index=int(consumer_index),
            )
            if source_name is None:
                continue
            _insert_layout_repair_transpose(
                model_ir,
                active_index,
                consumer_index=int(consumer_index),
                source_name=source_name,
                tensor_name=tensor_name,
            )
            repaired += 1
            continue

        if input_index != 0 or consumer_type != "MUL":
            continue
        if not tensor_name.startswith("input."):
            continue
        tensor_consumers = active_index.consumer_indices(tensor_name)
        if not tensor_consumers:
            continue
        if not all(
            0 <= int(index) < len(model_ir.operators)
            and str(model_ir.operators[int(index)].op_type) == "MUL"
            and len(model_ir.operators[int(index)].inputs) > 0
            and str(model_ir.operators[int(index)].inputs[0]) == tensor_name
            for index in tensor_consumers
        ):
            continue
        if orphan_tensor is None or not _is_fully_known_positive_shape(
            orphan_tensor.shape
        ):
            continue
        orphan_shape = [int(value) for value in orphan_tensor.shape]
        if len(orphan_shape) != 4:
            continue
        source_name = _select_mul_alias_source(
            model_ir,
            active_index,
            orphan_dtype=str(orphan_tensor.dtype),
            expected_source_shape=_expected_nhwc_source_shape(orphan_shape),
            consumer_index=int(consumer_index),
        )
        if source_name is None:
            continue
        _insert_layout_repair_transpose(
            model_ir,
            active_index,
            consumer_index=int(consumer_index),
            source_name=source_name,
            tensor_name=tensor_name,
        )
        source_tensor = model_ir.tensors.get(source_name)
        if source_tensor is not None:
            orphan_tensor.quantization = _clone_quantization(
                source_tensor.quantization
            )
            orphan_tensor.shape_signature = (
                [int(value) for value in source_tensor.shape_signature]
                if source_tensor.shape_signature is not None
                else [int(value) for value in orphan_shape]
            )
        repaired += 1

    return UnboundInputLayoutRepairResult(
        repaired=int(repaired),
        graph_index=active_index,
    )
