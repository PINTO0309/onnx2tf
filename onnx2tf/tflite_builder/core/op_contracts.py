from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import onnx


class NodeValidationError(ValueError):
    def __init__(
        self,
        *,
        reason_code: str,
        message: str,
        node_name: str,
        node_op: str,
    ) -> None:
        super().__init__(message)
        self.reason_code = str(reason_code)
        self.node_name = str(node_name)
        self.node_op = str(node_op)
        self.message = str(message)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "node_name": self.node_name,
            "onnx_op": self.node_op,
            "reason_code": self.reason_code,
            "message": self.message,
        }


@dataclass(frozen=True)
class ValidationSpec:
    min_inputs: int = 0
    max_inputs: Optional[int] = None
    min_outputs: int = 1
    max_outputs: Optional[int] = 1
    required_attrs: List[str] = field(default_factory=list)
    input_rank: Dict[int, List[int]] = field(default_factory=dict)
    output_rank: Dict[int, List[int]] = field(default_factory=dict)


@dataclass(frozen=True)
class DispatchEntry:
    onnx_op: str
    tflite_ops: List[str]
    builder: Callable[[Any, Any], None]
    validation: ValidationSpec = field(default_factory=ValidationSpec)
    extra_validator: Optional[Callable[[Any, Any], None]] = None


@dataclass(frozen=True)
class DispatchResolution:
    entry: DispatchEntry
    dispatch_mode: str
    reason_code: Optional[str] = None
    message: Optional[str] = None


def require_const_input(
    node: Any,
    ctx: Any,
    input_index: int,
    input_label: str,
) -> np.ndarray:
    if input_index >= len(node.inputs):
        raise NodeValidationError(
            reason_code="missing_required_input",
            message=f"{input_label} input index={input_index} is missing",
            node_name=node.name,
            node_op=node.op,
        )
    tensor_name = node.inputs[input_index].name
    const_value = ctx.get_constant_array(tensor_name)
    if const_value is None:
        raise NodeValidationError(
            reason_code="requires_constant_input",
            message=f"{input_label} must be constant. tensor={tensor_name}",
            node_name=node.name,
            node_op=node.op,
        )
    return np.asarray(const_value)


def normalize_axis_for_rank(*, axis: int, rank: int, node: Any) -> int:
    normalized = int(axis)
    if normalized < 0:
        normalized += int(rank)
    if normalized < 0 or normalized >= int(rank):
        raise NodeValidationError(
            reason_code="unsupported_attribute_value",
            message=(
                f"axis out of range. axis={axis} normalized={normalized} rank={rank}"
            ),
            node_name=node.name,
            node_op=node.op,
        )
    return normalized


def is_unknown_rank_placeholder_tensor(ctx: Any, tensor_name: str) -> bool:
    shape = [int(value) for value in list(ctx.get_tensor_shape(tensor_name))]
    if len(shape) == 0 or not all(int(value) == 1 for value in shape):
        return False
    tensor = ctx.model_ir.tensors.get(str(tensor_name))
    if tensor is not None:
        signature = (
            [int(value) for value in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else list(shape)
        )
        if len(signature) != len(shape):
            return False
        if any(int(value) < 0 for value in signature):
            return True
        if len(shape) == 1 and int(signature[0]) == 1:
            raw_shape = getattr(ctx, "shape_map", {}).get(str(tensor_name))
            if raw_shape is None:
                return True
            if isinstance(raw_shape, (list, tuple)) and len(raw_shape) == 0:
                return True
        return False
    raw_shape = getattr(ctx, "shape_map", {}).get(str(tensor_name))
    if raw_shape is None or not isinstance(raw_shape, (list, tuple)):
        return True
    if len(raw_shape) == 0:
        return True
    return any(
        not isinstance(dim, (int, np.integer)) or int(dim) <= 0
        for dim in raw_shape
    )


def tensor_shape_with_signature(ctx: Any, tensor_name: str) -> List[int]:
    shape = [int(value) for value in list(ctx.get_tensor_shape(tensor_name))]
    tensor = ctx.model_ir.tensors.get(str(tensor_name))
    if tensor is not None and tensor.shape_signature is not None:
        signature = [int(value) for value in list(tensor.shape_signature)]
    else:
        raw_shape = getattr(ctx, "shape_map", {}).get(str(tensor_name))
        if isinstance(raw_shape, (list, tuple)):
            signature = [
                int(dim)
                if isinstance(dim, (int, np.integer)) and int(dim) >= 0
                else -1
                for dim in raw_shape
            ]
        else:
            signature = list(shape)
    if len(signature) != len(shape):
        return list(shape)
    return [
        int(signature[index])
        if int(signature[index]) < 0
        else int(shape[index])
        for index in range(len(shape))
    ]


def is_integer_dtype(dtype: str) -> bool:
    return str(dtype).upper() in {
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "UINT16",
        "UINT32",
        "UINT64",
    }


def get_original_node_inputs(node: Any, ctx: Any) -> List[str]:
    if bool(getattr(node, "inputs_are_remapped", False)):
        return [str(value.name) for value in node.inputs]
    onnx_model = getattr(ctx, "onnx_model", None)
    if onnx_model is None:
        return [str(value.name) for value in node.inputs]
    stack = [onnx_model.graph]
    while stack:
        graph = stack.pop()
        for graph_node in graph.node:
            graph_node_name = str(graph_node.name) or str(graph_node.op_type)
            if graph_node_name == str(node.name) and str(graph_node.op_type) == str(node.op):
                return [str(value) for value in graph_node.input]
            for attribute in graph_node.attribute:
                if attribute.type == onnx.AttributeProto.GRAPH:
                    stack.append(attribute.g)
                elif attribute.type == onnx.AttributeProto.GRAPHS:
                    stack.extend(attribute.graphs)
    return [str(value.name) for value in node.inputs]


def validate_counts(node: Any, spec: ValidationSpec) -> None:
    input_count = len(node.inputs)
    output_count = len(node.outputs)
    if input_count < int(spec.min_inputs):
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=f"input_count={input_count} is smaller than min_inputs={spec.min_inputs}",
            node_name=node.name,
            node_op=node.op,
        )
    if spec.max_inputs is not None and input_count > int(spec.max_inputs):
        raise NodeValidationError(
            reason_code="invalid_input_count",
            message=f"input_count={input_count} exceeds max_inputs={spec.max_inputs}",
            node_name=node.name,
            node_op=node.op,
        )
    if output_count < int(spec.min_outputs):
        raise NodeValidationError(
            reason_code="invalid_output_count",
            message=f"output_count={output_count} is smaller than min_outputs={spec.min_outputs}",
            node_name=node.name,
            node_op=node.op,
        )
    if spec.max_outputs is not None and output_count > int(spec.max_outputs):
        raise NodeValidationError(
            reason_code="invalid_output_count",
            message=f"output_count={output_count} exceeds max_outputs={spec.max_outputs}",
            node_name=node.name,
            node_op=node.op,
        )


def validate_attrs(node: Any, spec: ValidationSpec) -> None:
    for attr in spec.required_attrs:
        if attr not in node.attrs:
            raise NodeValidationError(
                reason_code="missing_required_attribute",
                message=f"required attribute '{attr}' is missing",
                node_name=node.name,
                node_op=node.op,
            )


def validate_rank_constraints(node: Any, ctx: Any, spec: ValidationSpec) -> None:
    for kind, constraints, values in (
        ("input", spec.input_rank, node.inputs),
        ("output", spec.output_rank, node.outputs),
    ):
        for index, allowed_ranks in constraints.items():
            if index >= len(values):
                continue
            tensor_name = values[index].name
            rank = len(ctx.get_tensor_shape(tensor_name))
            if rank not in allowed_ranks:
                raise NodeValidationError(
                    reason_code=f"unsupported_{kind}_rank",
                    message=(
                        f"{kind}[{index}] rank={rank} is not in supported "
                        f"ranks={allowed_ranks} for tensor={tensor_name}"
                    ),
                    node_name=node.name,
                    node_op=node.op,
                )
