from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _clone_quantization,
    _prune_unused_tensors,
    _quant_scale_count,
    _replace_tensor_inputs,
    _set_operator_inputs,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)


def _optimize_attention_gather_transpose_reshape_cleanup_chains(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Simplify proven Gather/Transpose/Reshape attention tails atomically."""

    pattern_a_key = "optimized_attention_gather_transpose_reshape_cleanup_pattern_a"
    pattern_b_key = "optimized_attention_gather_transpose_reshape_cleanup_pattern_b"
    pattern_a_rewritten = 0
    pattern_b_rewritten = 0
    old_pattern_a_perm = [1, 0, 2]
    new_pattern_a_perm = [0, 2, 1, 3]

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

    def _normalized_gather_axis(
        op: OperatorIR,
        rank: int,
    ) -> Optional[int]:
        if not isinstance(op.options, dict):
            return None
        try:
            axis = int(op.options.get("axis", 0))
            for key in ("batchDims", "batch_dims"):
                if key in op.options and int(op.options[key]) != 0:
                    return None
        except (TypeError, ValueError):
            return None
        if axis < 0:
            axis += int(rank)
        if axis < 0 or axis >= int(rank):
            return None
        return int(axis)

    def _int32_constant(
        *,
        name: str,
        expected_values: List[int],
        graph_index: ModelIRGraphIndex,
        scalar: bool = False,
    ) -> Optional[TensorIR]:
        tensor = model_ir.tensors.get(str(name))
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
        expected = [int(value) for value in expected_values]
        if array.dtype != np.dtype(np.int32) or values != expected:
            return None
        if scalar:
            if (
                len(expected) != 1
                or shape not in ([], [1])
                or signature != shape
                or int(array.size) != 1
                or list(array.shape) not in ([], [1])
            ):
                return None
        elif (
            shape != [len(expected)]
            or signature != [len(expected)]
            or list(array.shape) != [len(expected)]
        ):
            return None
        return tensor

    def _quantization_key(value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, QuantParamIR):
            return (
                "QuantParamIR",
                tuple(float(item) for item in value.scale),
                tuple(int(item) for item in value.zero_point),
                int(value.quantized_dimension),
                (
                    None
                    if value.min is None
                    else tuple(float(item) for item in value.min)
                ),
                (
                    None
                    if value.max is None
                    else tuple(float(item) for item in value.max)
                ),
            )
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
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            )
        if isinstance(value, (list, tuple)):
            return tuple(_quantization_key(item) for item in value)
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

    def _same_pattern_b_boundary_metadata(
        source: TensorIR,
        target: TensorIR,
    ) -> bool:
        return bool(
            str(source.dtype) == str(target.dtype)
            and str(source.logical_layout) == str(target.logical_layout)
            and str(source.physical_layout) == str(target.physical_layout)
            and _quantization_key(source.quantization)
            == _quantization_key(target.quantization)
        )

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

    def _tensor_input_sites(
        tensor_name: str,
        graph_index: ModelIRGraphIndex,
    ) -> set[Tuple[int, int]]:
        sites: set[Tuple[int, int]] = set()
        for operator_index in set(graph_index.consumer_indices(str(tensor_name))):
            operator = model_ir.operators[int(operator_index)]
            for input_index, input_name in enumerate(operator.inputs):
                if str(input_name) == str(tensor_name):
                    sites.add((int(operator_index), int(input_index)))
        return sites

    def _unique_tensor_name(base: str, reserved_names: set[str]) -> str:
        candidate = str(base)
        serial = 1
        while candidate in reserved_names:
            candidate = f"{base}_{serial}"
            serial += 1
        reserved_names.add(candidate)
        return candidate

    graph_index = ModelIRGraphIndex(model_ir)
    public_inputs = {str(name) for name in model_ir.inputs}
    public_outputs = {str(name) for name in model_ir.outputs}
    public_boundaries = public_inputs | public_outputs
    reserved_tensor_names = {
        str(name)
        for name in (
            list(model_ir.tensors)
            + list(model_ir.inputs)
            + list(model_ir.outputs)
            + [
                value
                for operator in model_ir.operators
                for value in list(operator.inputs) + list(operator.outputs)
            ]
        )
    }

    while True:
        changed = False

        # Pattern B: remove two scalar axis-zero Gathers and a shape identity.
        for gather0_idx in graph_index.operator_indices("GATHER"):
            gather0_op = model_ir.operators[int(gather0_idx)]
            if len(gather0_op.inputs) < 2 or len(gather0_op.outputs) != 1:
                continue
            input_name = str(gather0_op.inputs[0])
            gather0_out_name = str(gather0_op.outputs[0])
            if (
                gather0_out_name in public_boundaries
                or not _source_is_available_before(
                    graph_index=graph_index,
                    tensor_name=input_name,
                    operator_index=int(gather0_idx),
                )
                or not _owned_output(
                    graph_index=graph_index,
                    operator_index=int(gather0_idx),
                    tensor_name=gather0_out_name,
                )
            ):
                continue

            gather0_users = graph_index.consumer_indices(gather0_out_name)
            if len(gather0_users) != 1:
                continue
            gather1_idx = int(gather0_users[0])
            if gather1_idx <= int(gather0_idx):
                continue
            gather1_op = model_ir.operators[gather1_idx]
            if (
                str(gather1_op.op_type) != "GATHER"
                or len(gather1_op.inputs) < 2
                or len(gather1_op.outputs) != 1
                or str(gather1_op.inputs[0]) != gather0_out_name
            ):
                continue
            gather1_out_name = str(gather1_op.outputs[0])
            if gather1_out_name in public_boundaries or not _owned_output(
                graph_index=graph_index,
                operator_index=gather1_idx,
                tensor_name=gather1_out_name,
            ):
                continue

            gather1_users = graph_index.consumer_indices(gather1_out_name)
            if len(gather1_users) != 1:
                continue
            reshape_idx = int(gather1_users[0])
            if reshape_idx <= gather1_idx:
                continue
            reshape_op = model_ir.operators[reshape_idx]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != gather1_out_name
            ):
                continue
            reshape_out_name = str(reshape_op.outputs[0])
            if (
                reshape_out_name in public_boundaries
                or not _owned_output(
                    graph_index=graph_index,
                    operator_index=reshape_idx,
                    tensor_name=reshape_out_name,
                )
                or any(
                    int(user_index) <= reshape_idx
                    for user_index in graph_index.consumer_indices(reshape_out_name)
                )
            ):
                continue

            input_metadata = _metadata(
                model_ir.tensors.get(input_name),
                4,
            )
            gather0_metadata = _metadata(
                model_ir.tensors.get(gather0_out_name),
                3,
            )
            gather1_metadata = _metadata(
                model_ir.tensors.get(gather1_out_name),
                2,
            )
            reshape_metadata = _metadata(
                model_ir.tensors.get(reshape_out_name),
                4,
            )
            if any(
                metadata is None
                for metadata in (
                    input_metadata,
                    gather0_metadata,
                    gather1_metadata,
                    reshape_metadata,
                )
            ):
                continue
            assert input_metadata is not None
            assert gather0_metadata is not None
            assert gather1_metadata is not None
            assert reshape_metadata is not None
            input_shape, input_signature = input_metadata
            gather0_shape, gather0_signature = gather0_metadata
            gather1_shape, gather1_signature = gather1_metadata
            reshape_shape, reshape_signature = reshape_metadata
            if (
                int(input_shape[0]) != 1
                or int(input_shape[1]) != 1
                or gather0_shape != input_shape[1:]
                or int(gather0_shape[0]) != 1
                or gather1_shape != gather0_shape[1:]
                or reshape_shape != input_shape
                or gather0_signature != input_signature[1:]
                or gather1_signature != gather0_signature[1:]
                or reshape_signature != input_signature
                or _normalized_gather_axis(gather0_op, 4) != 0
                or _normalized_gather_axis(gather1_op, 3) != 0
            ):
                continue
            input_tensor = model_ir.tensors[input_name]
            gather0_tensor = model_ir.tensors[gather0_out_name]
            gather1_tensor = model_ir.tensors[gather1_out_name]
            reshape_tensor = model_ir.tensors[reshape_out_name]
            if any(
                str(tensor.dtype) != str(input_tensor.dtype)
                for tensor in (
                    gather0_tensor,
                    gather1_tensor,
                    reshape_tensor,
                )
            ) or not _same_pattern_b_boundary_metadata(
                input_tensor,
                reshape_tensor,
            ):
                continue

            if (
                _int32_constant(
                    name=str(gather0_op.inputs[1]),
                    expected_values=[0],
                    graph_index=graph_index,
                    scalar=True,
                )
                is None
                or _int32_constant(
                    name=str(gather1_op.inputs[1]),
                    expected_values=[0],
                    graph_index=graph_index,
                    scalar=True,
                )
                is None
                or _int32_constant(
                    name=str(reshape_op.inputs[1]),
                    expected_values=input_shape,
                    graph_index=graph_index,
                )
                is None
            ):
                continue

            remove_indices = {
                int(gather0_idx),
                int(gather1_idx),
                int(reshape_idx),
            }

            # Topology, metadata, constants, replacements, and removals are
            # complete before the first mutation.
            _replace_tensor_inputs(
                model_ir,
                reshape_out_name,
                input_name,
                graph_index=graph_index,
            )
            graph_index.remove_operators(remove_indices)
            pattern_b_rewritten += 1
            changed = True
            break

        if changed:
            continue

        # Pattern A: rank-lift Gather/Transpose into one rank-four Transpose.
        for gather_idx in graph_index.operator_indices("GATHER"):
            gather_op = model_ir.operators[int(gather_idx)]
            if len(gather_op.inputs) < 2 or len(gather_op.outputs) != 1:
                continue
            input_name = str(gather_op.inputs[0])
            gather_out_name = str(gather_op.outputs[0])
            if (
                gather_out_name in public_boundaries
                or not _source_is_available_before(
                    graph_index=graph_index,
                    tensor_name=input_name,
                    operator_index=int(gather_idx),
                )
                or not _owned_output(
                    graph_index=graph_index,
                    operator_index=int(gather_idx),
                    tensor_name=gather_out_name,
                )
            ):
                continue

            gather_users = graph_index.consumer_indices(gather_out_name)
            if len(gather_users) != 1:
                continue
            transpose_idx = int(gather_users[0])
            if transpose_idx <= int(gather_idx):
                continue
            transpose_op = model_ir.operators[transpose_idx]
            if (
                str(transpose_op.op_type) != "TRANSPOSE"
                or len(transpose_op.inputs) < 2
                or len(transpose_op.outputs) != 1
                or str(transpose_op.inputs[0]) != gather_out_name
            ):
                continue
            transpose_out_name = str(transpose_op.outputs[0])
            if transpose_out_name in public_boundaries or not _owned_output(
                graph_index=graph_index,
                operator_index=transpose_idx,
                tensor_name=transpose_out_name,
            ):
                continue

            transpose_users = graph_index.consumer_indices(transpose_out_name)
            if len(transpose_users) != 1:
                continue
            reshape_idx = int(transpose_users[0])
            if reshape_idx <= transpose_idx:
                continue
            reshape_op = model_ir.operators[reshape_idx]
            if (
                str(reshape_op.op_type) != "RESHAPE"
                or len(reshape_op.inputs) < 2
                or len(reshape_op.outputs) != 1
                or str(reshape_op.inputs[0]) != transpose_out_name
            ):
                continue
            reshape_out_name = str(reshape_op.outputs[0])
            if (
                reshape_out_name in public_inputs
                or not _owned_output(
                    graph_index=graph_index,
                    operator_index=reshape_idx,
                    tensor_name=reshape_out_name,
                )
                or any(
                    int(user_index) <= reshape_idx
                    for user_index in graph_index.consumer_indices(reshape_out_name)
                )
            ):
                continue

            input_metadata = _metadata(
                model_ir.tensors.get(input_name),
                4,
            )
            gather_metadata = _metadata(
                model_ir.tensors.get(gather_out_name),
                3,
            )
            transpose_metadata = _metadata(
                model_ir.tensors.get(transpose_out_name),
                3,
            )
            reshape_metadata = _metadata(
                model_ir.tensors.get(reshape_out_name),
                4,
            )
            if any(
                metadata is None
                for metadata in (
                    input_metadata,
                    gather_metadata,
                    transpose_metadata,
                    reshape_metadata,
                )
            ):
                continue
            assert input_metadata is not None
            assert gather_metadata is not None
            assert transpose_metadata is not None
            assert reshape_metadata is not None
            input_shape, input_signature = input_metadata
            gather_shape, gather_signature = gather_metadata
            transpose_shape, transpose_signature = transpose_metadata
            reshape_shape, reshape_signature = reshape_metadata
            expected_transpose_shape = [
                int(input_shape[2]),
                int(input_shape[1]),
                int(input_shape[3]),
            ]
            expected_reshape_shape = [
                1,
                1,
                int(input_shape[2]),
                int(input_shape[1]) * int(input_shape[3]),
            ]
            expected_transpose_signature = [
                int(input_signature[2]),
                int(input_signature[1]),
                int(input_signature[3]),
            ]
            expected_ranklift_signature = [
                int(input_signature[0]),
                int(input_signature[2]),
                int(input_signature[1]),
                int(input_signature[3]),
            ]
            if (
                int(input_shape[0]) != 1
                or gather_shape != input_shape[1:]
                or transpose_shape != expected_transpose_shape
                or reshape_shape != expected_reshape_shape
                or gather_signature != input_signature[1:]
                or transpose_signature != expected_transpose_signature
                or any(
                    int(signature_value) not in {-1, int(shape_value)}
                    for signature_value, shape_value in zip(
                        reshape_signature,
                        expected_reshape_shape,
                    )
                )
                or _normalized_gather_axis(gather_op, 4) != 0
            ):
                continue

            input_tensor = model_ir.tensors[input_name]
            gather_tensor = model_ir.tensors[gather_out_name]
            transpose_tensor = model_ir.tensors[transpose_out_name]
            reshape_tensor = model_ir.tensors[reshape_out_name]
            if any(
                str(tensor.dtype) != str(input_tensor.dtype)
                for tensor in (
                    gather_tensor,
                    transpose_tensor,
                    reshape_tensor,
                )
            ):
                continue

            index_tensor = _int32_constant(
                name=str(gather_op.inputs[1]),
                expected_values=[0],
                graph_index=graph_index,
                scalar=True,
            )
            perm_name = str(transpose_op.inputs[1])
            perm_tensor = _int32_constant(
                name=perm_name,
                expected_values=old_pattern_a_perm,
                graph_index=graph_index,
            )
            shape_tensor = _int32_constant(
                name=str(reshape_op.inputs[1]),
                expected_values=expected_reshape_shape,
                graph_index=graph_index,
            )
            if index_tensor is None or perm_tensor is None or shape_tensor is None:
                continue

            if not isinstance(transpose_op.options, dict):
                continue
            planned_options = dict(transpose_op.options)
            options_valid = True
            for key in ("perm", "onnxPerm"):
                if key not in planned_options:
                    continue
                value = planned_options[key]
                if not isinstance(value, (list, tuple)):
                    options_valid = False
                    break
                try:
                    normalized = [int(item) for item in value]
                except (TypeError, ValueError):
                    options_valid = False
                    break
                if normalized != old_pattern_a_perm:
                    options_valid = False
                    break
                planned_options[key] = list(new_pattern_a_perm)
            if not options_valid:
                continue

            quantization_valid, planned_quantization = _planned_ranklift_quantization(
                transpose_tensor.quantization
            )
            planned_logical_layout = _ranklift_layout(transpose_tensor.logical_layout)
            planned_physical_layout = _ranklift_layout(transpose_tensor.physical_layout)
            if (
                not quantization_valid
                or planned_logical_layout is None
                or planned_physical_layout is None
            ):
                continue

            candidate_reserved_names = set(reserved_tensor_names)
            planned_site = {(int(transpose_idx), 1)}
            shared_outside_plan = any(
                site not in planned_site
                for site in _tensor_input_sites(
                    perm_name,
                    graph_index,
                )
            )
            perm_mode = "update"
            target_perm_name = perm_name
            if shared_outside_plan or perm_name in public_outputs:
                perm_mode = "clone"
                target_perm_name = _unique_tensor_name(
                    f"{perm_name}_qkv_perm4",
                    candidate_reserved_names,
                )
            planned_inputs = [str(value) for value in transpose_op.inputs]
            planned_inputs[0] = input_name
            planned_inputs[1] = target_perm_name
            new_transpose_shape = [
                1,
                int(input_shape[2]),
                int(input_shape[1]),
                int(input_shape[3]),
            ]

            # Topology, metadata, constants, options, quantization, clone,
            # setters, and removal are complete before the first mutation.
            reserved_tensor_names.update(candidate_reserved_names)
            if perm_mode == "clone":
                model_ir.tensors[target_perm_name] = TensorIR(
                    name=target_perm_name,
                    dtype="INT32",
                    shape=[4],
                    shape_signature=[4],
                    data=np.asarray(new_pattern_a_perm, dtype=np.int32),
                    is_variable=False,
                    quantization=None,
                    logical_layout=str(perm_tensor.logical_layout),
                    physical_layout=str(perm_tensor.physical_layout),
                    onnx_tensor_name=perm_tensor.onnx_tensor_name,
                )
            else:
                perm_tensor.data = np.asarray(
                    new_pattern_a_perm,
                    dtype=np.int32,
                )
                perm_tensor.shape = [4]
                perm_tensor.shape_signature = [4]
                perm_tensor.quantization = None

            _set_operator_inputs(
                model_ir=model_ir,
                op=transpose_op,
                new_inputs=planned_inputs,
                graph_index=graph_index,
            )
            transpose_op.options = planned_options
            transpose_tensor.shape = list(new_transpose_shape)
            transpose_tensor.shape_signature = list(expected_ranklift_signature)
            transpose_tensor.quantization = planned_quantization
            transpose_tensor.logical_layout = str(planned_logical_layout)
            transpose_tensor.physical_layout = str(planned_physical_layout)
            graph_index.remove_operator(int(gather_idx))

            pattern_a_rewritten += 1
            changed = True
            break

        if not changed:
            break

    if pattern_a_rewritten > 0 or pattern_b_rewritten > 0:
        _prune_unused_tensors(model_ir)
    return {
        pattern_a_key: int(pattern_a_rewritten),
        pattern_b_key: int(pattern_b_rewritten),
    }
