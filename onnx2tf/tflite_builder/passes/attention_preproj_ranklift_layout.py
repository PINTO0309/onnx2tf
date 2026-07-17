from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_static_shapes,
    _clone_quantization,
    _prune_unused_tensors,
    _quant_scale_count,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Rank-lift fully proven attention pre-projection branches atomically."""

    stats_key = "optimized_attention_preproj_reshape_to_batchmatmul_ranklift_chains"
    binary_ops = {"ADD", "SUB", "MUL", "DIV"}
    rewritten = 0

    def _metadata(
        tensor: Optional[TensorIR],
        rank: int,
    ) -> Optional[Tuple[List[int], List[int]]]:
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
        if (
            len(shape) != int(rank)
            or len(signature) != int(rank)
            or any(int(value) <= 0 for value in shape)
            or any(
                int(value) not in {-1, int(shape[index])}
                for index, value in enumerate(signature)
            )
        ):
            return None
        return shape, signature

    def _positive_metadata(
        tensor: Optional[TensorIR],
    ) -> Optional[Tuple[List[int], List[int]]]:
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
        if len(shape) == 0:
            return ([], []) if len(signature) == 0 else None
        if (
            len(signature) != len(shape)
            or any(int(value) <= 0 for value in shape)
            or any(
                int(value) not in {-1, int(shape[index])}
                for index, value in enumerate(signature)
            )
        ):
            return None
        return shape, signature

    def _int32_shape_constant(
        *,
        name: str,
        expected_values: List[int],
        graph_index: ModelIRGraphIndex,
    ) -> Optional[TensorIR]:
        tensor = model_ir.tensors.get(str(name))
        expected = [int(value) for value in expected_values]
        if (
            str(name) in public_inputs
            or str(name) in graph_index.producers
            or str(name) in graph_index.duplicate_producers
            or tensor is None
            or tensor.data is None
            or bool(tensor.is_variable)
            or str(tensor.dtype) != "INT32"
            or tensor.quantization is not None
        ):
            return None
        try:
            shape = [int(value) for value in tensor.shape]
            signature = (
                [int(value) for value in tensor.shape_signature]
                if tensor.shape_signature is not None
                else list(shape)
            )
            array = np.asarray(tensor.data)
            values = [int(value) for value in array.reshape(-1).tolist()]
        except (TypeError, ValueError):
            return None
        if (
            shape != [len(expected)]
            or signature != [len(expected)]
            or array.dtype != np.dtype(np.int32)
            or list(array.shape) != [len(expected)]
            or values != expected
        ):
            return None
        return tensor

    def _quantization_key(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return (
                "ndarray",
                str(value.dtype),
                tuple(int(item) for item in value.shape),
                _quantization_key(value.tolist()),
            )
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, dict):
            return tuple(
                (str(key), _quantization_key(item))
                for key, item in sorted(
                    value.items(),
                    key=lambda pair: str(pair[0]),
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(_quantization_key(item) for item in value)
        if all(
            hasattr(value, attribute)
            for attribute in (
                "scale",
                "zero_point",
                "quantized_dimension",
            )
        ):
            return (
                type(value).__name__,
                tuple(float(item) for item in value.scale),
                tuple(int(item) for item in value.zero_point),
                int(value.quantized_dimension),
                _quantization_key(getattr(value, "min", None)),
                _quantization_key(getattr(value, "max", None)),
            )
        return value

    def _planned_ranklift_quantization(
        quantization: Any,
    ) -> Tuple[bool, Any]:
        try:
            planned = _clone_quantization(quantization)
            scale_count = _quant_scale_count(planned)
        except Exception:
            return False, None
        if planned is None or int(scale_count) <= 1:
            return True, planned
        try:
            if isinstance(planned, dict):
                if "quantized_dimension" not in planned:
                    return False, None
                old_dimension = int(planned["quantized_dimension"])
                if old_dimension < 0 or old_dimension >= 3:
                    return False, None
                planned["quantized_dimension"] = int(old_dimension + 1)
            else:
                old_dimension = int(planned.quantized_dimension)
                if old_dimension < 0 or old_dimension >= 3:
                    return False, None
                planned.quantized_dimension = int(old_dimension + 1)
        except (AttributeError, TypeError, ValueError):
            return False, None
        return True, planned

    def _ranklift_layout(layout: str) -> Optional[str]:
        return {
            "UNKNOWN": "UNKNOWN",
            "NCW": "NCHW",
            "NWC": "NHWC",
        }.get(str(layout))

    def _owned_output(
        *,
        graph_index: ModelIRGraphIndex,
        operator_index: int,
        tensor_name: str,
    ) -> bool:
        return bool(
            str(tensor_name) not in graph_index.duplicate_producers
            and graph_index.producers.get(str(tensor_name)) == int(operator_index)
        )

    def _source_is_available_before(
        *,
        graph_index: ModelIRGraphIndex,
        tensor_name: str,
        operator_index: int,
    ) -> bool:
        name = str(tensor_name)
        if name in graph_index.duplicate_producers:
            return False
        producer_index = graph_index.producers.get(name)
        if producer_index is None:
            return name in model_ir.tensors
        if name in public_inputs:
            return False
        return int(producer_index) < int(operator_index)

    def _false_bmm_flags(op: OperatorIR) -> bool:
        if not isinstance(op.options, dict):
            return False
        for key in ("adjX", "adjY", "adj_x", "adj_y"):
            if key not in op.options:
                continue
            value = op.options[key]
            if isinstance(value, (bool, np.bool_)):
                if bool(value):
                    return False
                continue
            if isinstance(value, (int, np.integer)) and int(value) == 0:
                continue
            return False
        return True

    graph_index = ModelIRGraphIndex(model_ir)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs

    while True:
        changed = False
        for reshape_idx in graph_index.operator_indices("RESHAPE"):
            reshape_op = model_ir.operators[int(reshape_idx)]
            if len(reshape_op.inputs) < 2 or len(reshape_op.outputs) != 1:
                continue
            input_name = str(reshape_op.inputs[0])
            reshape_out_name = str(reshape_op.outputs[0])
            if (
                reshape_out_name in public_boundaries
                or not _source_is_available_before(
                    graph_index=graph_index,
                    tensor_name=input_name,
                    operator_index=int(reshape_idx),
                )
                or not _owned_output(
                    graph_index=graph_index,
                    operator_index=int(reshape_idx),
                    tensor_name=reshape_out_name,
                )
            ):
                continue

            input_metadata = _metadata(
                model_ir.tensors.get(input_name),
                4,
            )
            reshape_metadata = _metadata(
                model_ir.tensors.get(reshape_out_name),
                3,
            )
            if input_metadata is None or reshape_metadata is None:
                continue
            input_shape, input_signature = input_metadata
            reshape_shape, reshape_signature = reshape_metadata
            if (
                int(input_shape[0]) != 1
                or int(input_shape[1]) != 1
                or reshape_shape != [int(input_shape[2]), 1, int(input_shape[3])]
                or reshape_signature
                != [int(input_signature[2]), 1, int(input_signature[3])]
            ):
                continue
            input_tensor = model_ir.tensors[input_name]
            reshape_tensor = model_ir.tensors[reshape_out_name]
            if str(input_tensor.dtype) != str(
                reshape_tensor.dtype
            ) or _quantization_key(input_tensor.quantization) != _quantization_key(
                reshape_tensor.quantization
            ):
                continue

            t_dim = int(input_shape[2])
            c_dim = int(input_shape[3])
            t_signature = int(input_signature[2])
            if (
                _int32_shape_constant(
                    name=str(reshape_op.inputs[1]),
                    expected_values=[t_dim, 1, c_dim],
                    graph_index=graph_index,
                )
                is None
            ):
                continue

            bmm_indices = graph_index.consumer_indices(reshape_out_name)
            if len(bmm_indices) == 0 or len(bmm_indices) != len(set(bmm_indices)):
                continue

            branch_plans: List[Dict[str, Any]] = []
            valid = True
            for bmm_idx in bmm_indices:
                bmm_idx = int(bmm_idx)
                if bmm_idx <= int(reshape_idx):
                    valid = False
                    break
                bmm_op = model_ir.operators[bmm_idx]
                if (
                    str(bmm_op.op_type) != "BATCH_MATMUL"
                    or len(bmm_op.inputs) != 2
                    or len(bmm_op.outputs) != 1
                    or str(bmm_op.inputs[0]) != reshape_out_name
                    or not _false_bmm_flags(bmm_op)
                ):
                    valid = False
                    break
                bmm_out_name = str(bmm_op.outputs[0])
                if bmm_out_name in public_boundaries or not _owned_output(
                    graph_index=graph_index,
                    operator_index=bmm_idx,
                    tensor_name=bmm_out_name,
                ):
                    valid = False
                    break

                rhs_name = str(bmm_op.inputs[1])
                rhs_metadata = _metadata(
                    model_ir.tensors.get(rhs_name),
                    2,
                )
                bmm_metadata = _metadata(
                    model_ir.tensors.get(bmm_out_name),
                    3,
                )
                if rhs_metadata is None or bmm_metadata is None:
                    valid = False
                    break
                rhs_shape, _ = rhs_metadata
                bmm_shape, bmm_signature = bmm_metadata
                if len(rhs_shape) != 2 or int(rhs_shape[0]) != c_dim:
                    valid = False
                    break
                k_dim = int(rhs_shape[1])
                if bmm_shape != [t_dim, 1, k_dim] or bmm_signature != [
                    t_signature,
                    1,
                    k_dim,
                ]:
                    valid = False
                    break

                bmm_users = graph_index.consumer_indices(bmm_out_name)
                if len(bmm_users) != 1:
                    valid = False
                    break
                binary_idx = int(bmm_users[0])
                if binary_idx <= bmm_idx:
                    valid = False
                    break
                binary_op = model_ir.operators[binary_idx]
                binary_inputs = [str(value) for value in binary_op.inputs]
                if (
                    str(binary_op.op_type) not in binary_ops
                    or len(binary_inputs) != 2
                    or len(binary_op.outputs) != 1
                    or binary_inputs.count(bmm_out_name) != 1
                ):
                    valid = False
                    break
                other_input_name = next(
                    name for name in binary_inputs if name != bmm_out_name
                )
                other_metadata = _positive_metadata(
                    model_ir.tensors.get(other_input_name)
                )
                if other_metadata is None:
                    valid = False
                    break
                other_shape, _ = other_metadata
                broadcast_other_shape = other_shape or [1]
                new_branch_shape = [1, 1, t_dim, k_dim]
                if (
                    _broadcast_static_shapes(
                        bmm_shape,
                        broadcast_other_shape,
                    )
                    != bmm_shape
                    or _broadcast_static_shapes(
                        new_branch_shape,
                        broadcast_other_shape,
                    )
                    != new_branch_shape
                ):
                    valid = False
                    break

                binary_out_name = str(binary_op.outputs[0])
                if binary_out_name in public_boundaries or not _owned_output(
                    graph_index=graph_index,
                    operator_index=binary_idx,
                    tensor_name=binary_out_name,
                ):
                    valid = False
                    break
                binary_metadata = _metadata(
                    model_ir.tensors.get(binary_out_name),
                    3,
                )
                if binary_metadata is None:
                    valid = False
                    break
                binary_shape, binary_signature = binary_metadata
                if binary_shape != bmm_shape or binary_signature != bmm_signature:
                    valid = False
                    break

                bmm_tensor = model_ir.tensors[bmm_out_name]
                binary_tensor = model_ir.tensors[binary_out_name]
                other_tensor = model_ir.tensors[other_input_name]
                if any(
                    str(tensor.dtype) != str(input_tensor.dtype)
                    for tensor in (
                        bmm_tensor,
                        binary_tensor,
                        other_tensor,
                    )
                ):
                    valid = False
                    break

                binary_users = graph_index.consumer_indices(binary_out_name)
                if len(binary_users) != 1:
                    valid = False
                    break
                tail_idx = int(binary_users[0])
                if tail_idx <= binary_idx:
                    valid = False
                    break
                tail_op = model_ir.operators[tail_idx]
                if (
                    str(tail_op.op_type) != "RESHAPE"
                    or len(tail_op.inputs) < 2
                    or len(tail_op.outputs) != 1
                    or str(tail_op.inputs[0]) != binary_out_name
                ):
                    valid = False
                    break
                tail_out_name = str(tail_op.outputs[0])
                if (
                    tail_out_name in public_inputs
                    or not _owned_output(
                        graph_index=graph_index,
                        operator_index=tail_idx,
                        tensor_name=tail_out_name,
                    )
                    or any(
                        int(user_index) <= tail_idx
                        for user_index in graph_index.consumer_indices(tail_out_name)
                    )
                ):
                    valid = False
                    break
                tail_metadata = _metadata(
                    model_ir.tensors.get(tail_out_name),
                    4,
                )
                if tail_metadata is None:
                    valid = False
                    break
                tail_shape, tail_signature = tail_metadata
                if (
                    len(tail_shape) != 4
                    or tail_shape[0] != 1
                    or tail_shape[1] != t_dim
                    or any(int(value) <= 0 for value in tail_shape)
                    or int(tail_shape[2]) * int(tail_shape[3]) != k_dim
                    or int(tail_signature[0]) != 1
                    or int(tail_signature[1]) not in {-1, t_signature, t_dim}
                    or int(tail_signature[2]) != int(tail_shape[2])
                    or int(tail_signature[3]) != int(tail_shape[3])
                    or str(model_ir.tensors[tail_out_name].dtype)
                    != str(input_tensor.dtype)
                    or _int32_shape_constant(
                        name=str(tail_op.inputs[1]),
                        expected_values=tail_shape,
                        graph_index=graph_index,
                    )
                    is None
                ):
                    valid = False
                    break

                bmm_quant_valid, bmm_quantization = _planned_ranklift_quantization(
                    bmm_tensor.quantization
                )
                binary_quant_valid, binary_quantization = (
                    _planned_ranklift_quantization(binary_tensor.quantization)
                )
                bmm_logical_layout = _ranklift_layout(bmm_tensor.logical_layout)
                bmm_physical_layout = _ranklift_layout(bmm_tensor.physical_layout)
                binary_logical_layout = _ranklift_layout(binary_tensor.logical_layout)
                binary_physical_layout = _ranklift_layout(binary_tensor.physical_layout)
                if (
                    not bmm_quant_valid
                    or not binary_quant_valid
                    or bmm_logical_layout is None
                    or bmm_physical_layout is None
                    or binary_logical_layout is None
                    or binary_physical_layout is None
                ):
                    valid = False
                    break

                branch_plans.append(
                    {
                        "bmm_idx": bmm_idx,
                        "bmm_op": bmm_op,
                        "bmm_inputs": [input_name, rhs_name],
                        "bmm_tensor": bmm_tensor,
                        "bmm_quantization": bmm_quantization,
                        "bmm_logical_layout": bmm_logical_layout,
                        "bmm_physical_layout": bmm_physical_layout,
                        "binary_tensor": binary_tensor,
                        "binary_quantization": binary_quantization,
                        "binary_logical_layout": binary_logical_layout,
                        "binary_physical_layout": binary_physical_layout,
                        "shape": new_branch_shape,
                        "signature": [1, 1, t_signature, k_dim],
                    }
                )

            if not valid or len(branch_plans) == 0:
                continue

            # The full topology, metadata, constant, option, broadcast,
            # quantization, setter, removal, and pruning plan is complete.
            for plan in branch_plans:
                _set_operator_inputs(
                    model_ir=model_ir,
                    op=plan["bmm_op"],
                    new_inputs=plan["bmm_inputs"],
                    graph_index=graph_index,
                )
                bmm_tensor = plan["bmm_tensor"]
                bmm_tensor.shape = list(plan["shape"])
                bmm_tensor.shape_signature = list(plan["signature"])
                bmm_tensor.quantization = plan["bmm_quantization"]
                bmm_tensor.logical_layout = str(plan["bmm_logical_layout"])
                bmm_tensor.physical_layout = str(plan["bmm_physical_layout"])

                binary_tensor = plan["binary_tensor"]
                binary_tensor.shape = list(plan["shape"])
                binary_tensor.shape_signature = list(plan["signature"])
                binary_tensor.quantization = plan["binary_quantization"]
                binary_tensor.logical_layout = str(plan["binary_logical_layout"])
                binary_tensor.physical_layout = str(plan["binary_physical_layout"])

            graph_index.remove_operator(int(reshape_idx))
            rewritten += 1
            changed = True
            break

        if not changed:
            break

    if rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {stats_key: int(rewritten)}
