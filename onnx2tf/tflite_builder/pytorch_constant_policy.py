from __future__ import annotations

from typing import Callable, Dict, List, Optional, Set

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.pytorch_codegen_utils import _constant_int_list
from onnx2tf.tflite_builder.pytorch_codegen_values import (
    _scalar_literal_for_constant_tensor,
    _torch_pad_literal_for_constant_tensor,
)
from onnx2tf.tflite_builder.pytorch_layout_utils import (
    _constant_pad_pairs_for_tensor,
)


def _is_constant_tensor_name_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    return tensor is not None and isinstance(tensor.data, np.ndarray)


def _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> bool:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.size == 0 or not np.issubdtype(arr.dtype, np.integer):
        return False
    return not bool(np.any(arr.reshape(-1) == 0))


def _static_int_tensor_values_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    tensor_name: str,
    _visited: Optional[Set[str]] = None,
) -> Optional[List[int]]:
    visited = set() if _visited is None else _visited
    current_name = str(tensor_name)
    if current_name in visited:
        return None
    visited.add(current_name)

    direct_values = _constant_int_list(model_ir.tensors.get(current_name, None))
    if direct_values is not None:
        return [int(v) for v in list(direct_values)]

    producer_idx = producer_index.get(current_name, None)
    if producer_idx is None:
        return None

    producer = model_ir.operators[int(producer_idx)]
    op_type = str(producer.op_type)

    def _static_shape_values(input_name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(input_name), None)
        if tensor is None:
            return None
        shape_values = (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            and len(list(tensor.shape_signature)) == len(list(tensor.shape))
            else [int(v) for v in list(tensor.shape)]
        )
        if any(int(v) <= 0 for v in shape_values):
            return None
        return shape_values

    def _scalar_or_vector_int_values(
        input_name: str,
    ) -> Optional[List[int]]:
        return _static_int_tensor_values_for_codegen(
            model_ir=model_ir,
            producer_index=producer_index,
            tensor_name=str(input_name),
            _visited=set(visited),
        )

    if op_type == "SHAPE" and len(producer.inputs) >= 1:
        return _static_shape_values(str(producer.inputs[0]))

    if (
        op_type in {"CAST", "EXPAND_DIMS", "IDENTITY", "RESHAPE", "SQUEEZE"}
        and len(producer.inputs) >= 1
    ):
        return _scalar_or_vector_int_values(str(producer.inputs[0]))

    if op_type == "GATHER" and len(producer.inputs) >= 2:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        gather_indices = _scalar_or_vector_int_values(str(producer.inputs[1]))
        axis = int(producer.options.get("axis", 0))
        batch_dims = int(producer.options.get("batchDims", 0))
        if input_values is None or gather_indices is None or batch_dims != 0:
            return None
        if axis < 0:
            axis += 1
        if axis != 0:
            return None
        gathered: List[int] = []
        for raw_index in gather_indices:
            index = int(raw_index)
            if index < 0:
                index += len(input_values)
            if index < 0 or index >= len(input_values):
                return None
            gathered.append(int(input_values[index]))
        return gathered

    if op_type == "GATHER_ND" and len(producer.inputs) >= 2:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        gather_indices = _constant_int_list(
            model_ir.tensors.get(str(producer.inputs[1]), None)
        )
        if input_values is None or gather_indices is None:
            return None
        if len(gather_indices) == 1:
            index = int(gather_indices[0])
            if index < 0:
                index += len(input_values)
            if index < 0 or index >= len(input_values):
                return None
            return [int(input_values[index])]
        return None

    if op_type == "SLICE" and len(producer.inputs) >= 3:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        begin_values = _scalar_or_vector_int_values(str(producer.inputs[1]))
        size_values = _scalar_or_vector_int_values(str(producer.inputs[2]))
        if (
            input_values is None
            or begin_values is None
            or size_values is None
            or len(begin_values) != 1
            or len(size_values) != 1
        ):
            return None
        start = int(begin_values[0])
        if start < 0:
            start += len(input_values)
        size = int(size_values[0])
        stop = None if size < 0 else start + size
        return [int(v) for v in input_values[slice(start, stop)]]

    if op_type == "STRIDED_SLICE" and len(producer.inputs) >= 4:
        input_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        begin_values = _scalar_or_vector_int_values(str(producer.inputs[1]))
        end_values = _scalar_or_vector_int_values(str(producer.inputs[2]))
        stride_values = _scalar_or_vector_int_values(str(producer.inputs[3]))
        if (
            input_values is None
            or begin_values is None
            or end_values is None
            or stride_values is None
            or len(begin_values) != 1
            or len(end_values) != 1
            or len(stride_values) != 1
        ):
            return None
        begin_mask = int(producer.options.get("beginMask", 0))
        end_mask = int(producer.options.get("endMask", 0))
        start = None if (begin_mask & 1) else int(begin_values[0])
        stop = None if (end_mask & 1) else int(end_values[0])
        step = int(stride_values[0])
        if step == 0:
            return None
        return [int(v) for v in input_values[slice(start, stop, step)]]

    if op_type in {"CONCATENATION", "PACK"}:
        output_values: List[int] = []
        for input_name in producer.inputs:
            input_values = _scalar_or_vector_int_values(str(input_name))
            if input_values is None:
                return None
            output_values.extend(int(v) for v in input_values)
        return output_values

    if op_type in {"MAXIMUM", "MINIMUM"} and len(producer.inputs) >= 2:
        lhs_values = _scalar_or_vector_int_values(str(producer.inputs[0]))
        rhs_values = _scalar_or_vector_int_values(str(producer.inputs[1]))
        if lhs_values is None or rhs_values is None:
            return None
        lhs_array = np.asarray(lhs_values, dtype=np.int64)
        rhs_array = np.asarray(rhs_values, dtype=np.int64)
        try:
            output_array = (
                np.maximum(lhs_array, rhs_array)
                if op_type == "MAXIMUM"
                else np.minimum(lhs_array, rhs_array)
            )
        except ValueError:
            return None
        return [int(v) for v in output_array.reshape(-1).tolist()]

    return None


def _reshape_shape_tensor_uses_runtime_dims_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    tensor_name: str,
    _visited: Optional[Set[str]] = None,
) -> bool:
    visited = set() if _visited is None else _visited
    current_name = str(tensor_name)
    if current_name in visited:
        return False
    visited.add(current_name)

    producer_idx = producer_index.get(current_name, None)
    if producer_idx is None:
        return _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
            model_ir=model_ir,
            tensor_name=current_name,
        )

    producer = model_ir.operators[int(producer_idx)]
    op_type = str(producer.op_type)
    if op_type == "SHAPE":
        return True
    if op_type in {"CAST", "EXPAND_DIMS", "IDENTITY", "RESHAPE", "SQUEEZE"}:
        return (
            len(producer.inputs) >= 1
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[0]),
                _visited=set(visited),
            )
        )
    if op_type in {"GATHER", "GATHER_ND", "SLICE", "STRIDED_SLICE"}:
        return (
            len(producer.inputs) >= 1
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[0]),
                _visited=set(visited),
            )
        )
    if op_type == "SPLIT":
        split_data_input_index = 1 if len(producer.inputs) >= 2 else 0
        return (
            len(producer.inputs) > split_data_input_index
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[split_data_input_index]),
                _visited=set(visited),
            )
        )
    if op_type == "UNPACK":
        return (
            len(producer.inputs) >= 1
            and _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(producer.inputs[0]),
                _visited=set(visited),
            )
        )
    if op_type in {"CONCATENATION", "PACK"}:
        saw_runtime_dims = False
        for input_name in producer.inputs:
            if _reshape_shape_tensor_uses_runtime_dims_for_codegen(
                model_ir=model_ir,
                producer_index=producer_index,
                tensor_name=str(input_name),
                _visited=set(visited),
            ):
                saw_runtime_dims = True
                continue
            if _shape_tensor_constant_is_non_zero_int_vector_for_codegen(
                model_ir=model_ir,
                tensor_name=str(input_name),
            ):
                continue
            return False
        return saw_runtime_dims
    return False


def _pad_literal_expr_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    return _torch_pad_literal_for_constant_tensor(
        model_ir.tensors.get(str(tensor_name), None)
    )


def _constant_pad_pairs_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[List[List[int]]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    return _constant_pad_pairs_for_tensor(tensor)


def _scalar_literal_expr_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
) -> Optional[str]:
    return _scalar_literal_for_constant_tensor(
        model_ir.tensors.get(str(tensor_name), None)
    )


def _int_scalar_literal_expr_for_codegen(
    *,
    static_int_tensor_values_fn: Callable[[str], Optional[List[int]]],
    tensor_name: str,
) -> Optional[str]:
    values = static_int_tensor_values_fn(str(tensor_name))
    if values is None or len(values) != 1:
        return None
    return repr(int(values[0]))


def _axis_expr_from_input_for_codegen(
    *,
    runtime_imports: Set[str],
    int_scalar_literal_expr_fn: Callable[[str], Optional[str]],
    tensor_expr_fn: Callable[[str], str],
    tensor_name: str,
    device_expr: str,
) -> str:
    axis_literal = int_scalar_literal_expr_fn(tensor_name)
    if axis_literal is not None:
        return axis_literal
    runtime_imports.add("_coerce_scalar_axis")
    return (
        f"_coerce_scalar_axis({tensor_expr_fn(str(tensor_name))}, "
        f"device={device_expr}.device)"
    )


def _static_mirror_pad_expr_for_codegen(
    *,
    model_ir: ModelIR,
    runtime_imports: Set[str],
    constant_pad_pairs_fn: Callable[[str], Optional[List[List[int]]]],
    tensor_expr_fn: Callable[[str], str],
    input_tensor_name: str,
    pads_tensor_name: str,
) -> Optional[str]:
    pad_pairs = constant_pad_pairs_fn(pads_tensor_name)
    input_tensor = model_ir.tensors.get(str(input_tensor_name), None)
    if pad_pairs is None or input_tensor is None:
        return None
    rank = len(list(input_tensor.shape))
    if rank <= 0:
        return None
    if len(pad_pairs) < rank:
        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs
    elif len(pad_pairs) > rank:
        pad_pairs = pad_pairs[-rank:]
    non_zero_axes = [
        idx
        for idx, (before, after) in enumerate(pad_pairs)
        if int(before) != 0 or int(after) != 0
    ]
    input_expr = tensor_expr_fn(str(input_tensor_name))
    if len(non_zero_axes) == 0:
        return input_expr
    if len(non_zero_axes) > 3:
        return None
    keep_axes = [idx for idx in range(rank) if idx not in non_zero_axes]
    perm = keep_axes + non_zero_axes
    expr = input_expr
    if perm != list(range(rank)):
        runtime_imports.add("_torch_permute")
        expr = f"_torch_permute({expr}, {repr(perm)})"
    torch_pad: List[int] = []
    for axis in reversed(non_zero_axes):
        before, after = pad_pairs[axis]
        torch_pad.extend([int(before), int(after)])
    expr = f"F.pad({expr}, {repr(torch_pad)}, mode='reflect')"
    if perm == list(range(rank)):
        return expr
    inverse_perm = [0] * rank
    for permuted_axis, original_axis in enumerate(perm):
        inverse_perm[int(original_axis)] = int(permuted_axis)
    runtime_imports.add("_torch_permute")
    return f"_torch_permute({expr}, {repr(inverse_perm)})"
