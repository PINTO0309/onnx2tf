from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import keyword
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Optional, Sequence, Set, Tuple, cast

import numpy as np
import onnx

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    channel_first_logical_layout,
    channel_last_logical_layout,
    infer_model_ir_logical_layouts,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_rank,
    logical_layout_permutation,
    normalize_logical_layout,
    rewrite_axis_for_layout,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    SUPPORTED_TORCH_KERNEL_OP_TYPES,
)
from onnx2tf.tflite_builder.split_planner import (
    rewrite_model_ir_unroll_recurrent_ops,
)
from onnx2tf.tflite_builder.tflite_importer import (
    import_model_ir_from_tflite,
)


class ModelIRPyTorchExportError(RuntimeError):
    pass

@dataclass
class _NativeModelFileWriterContext:
    output_folder_path: str
    model_ir: ModelIR
    metadata: Dict[str, Any]
    tensor_storage_name_map: Dict[str, str]
    package_dir: Path
    preserve_channel_last_tensor_names: Set[str]
    tensor_var_names: Dict[str, str]
    producer_index: Dict[str, int]
    consumer_index: Dict[str, List[int]]
    module_init_lines: List[str] = field(default_factory=list)
    load_specs: List[Tuple[str, str]] = field(default_factory=list)
    runtime_imports: Set[str] = field(default_factory=set)
    forward_lines: List[str] = field(default_factory=list)

@dataclass
class _NativeCodegenBindings:
    items: Dict[str, Callable[..., Any]] = field(default_factory=dict)

@dataclass
class _NativeCodegenState:
    context: _NativeModelFileWriterContext
    load_specs_result: Optional[List[Tuple[str, str]]] = None

def _shape_lists_equal(lhs: Optional[Sequence[int]], rhs: Optional[Sequence[int]]) -> bool:
    if lhs is None or rhs is None:
        return False
    return [int(v) for v in list(lhs)] == [int(v) for v in list(rhs)]

def _shape_lists_equal_relaxed(lhs: Optional[Sequence[int]], rhs: Optional[Sequence[int]]) -> bool:
    if lhs is None or rhs is None:
        return False
    lhs_items = [int(v) for v in list(lhs)]
    rhs_items = [int(v) for v in list(rhs)]
    if len(lhs_items) != len(rhs_items):
        return False
    for lhs_dim, rhs_dim in zip(lhs_items, rhs_items):
        if lhs_dim == rhs_dim:
            continue
        if lhs_dim <= 0 or rhs_dim <= 0:
            continue
        return False
    return True

def _shape_can_broadcast_to_target_relaxed(
    shape: Optional[Sequence[int]],
    target_shape: Optional[Sequence[int]],
) -> bool:
    if shape is None or target_shape is None:
        return False
    shape_items = [int(v) for v in list(shape)]
    target_items = [int(v) for v in list(target_shape)]
    if len(shape_items) != len(target_items):
        return False
    for shape_dim, target_dim in zip(shape_items, target_items):
        if shape_dim == 1 or shape_dim == target_dim:
            continue
        if shape_dim <= 0 or target_dim <= 0:
            continue
        return False
    return True

def _broadcast_shapes_relaxed(
    lhs: Optional[Sequence[int]],
    rhs: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if lhs is None or rhs is None:
        return None
    lhs_items = [int(v) for v in list(lhs)]
    rhs_items = [int(v) for v in list(rhs)]
    if len(lhs_items) != len(rhs_items):
        return None
    result: List[int] = []
    for lhs_dim, rhs_dim in zip(lhs_items, rhs_items):
        if lhs_dim == rhs_dim:
            result.append(int(lhs_dim))
            continue
        if lhs_dim == 1:
            result.append(int(rhs_dim))
            continue
        if rhs_dim == 1:
            result.append(int(lhs_dim))
            continue
        if lhs_dim <= 0 and rhs_dim > 0:
            result.append(int(rhs_dim))
            continue
        if rhs_dim <= 0 and lhs_dim > 0:
            result.append(int(lhs_dim))
            continue
        if lhs_dim <= 0 and rhs_dim <= 0:
            result.append(-1)
            continue
        return None
    return result

def _product_expr(items: Sequence[str]) -> str:
    item_list = [str(item) for item in list(items)]
    if len(item_list) == 0:
        return "1"
    expr = item_list[0]
    for item in item_list[1:]:
        expr = f"({expr} * {item})"
    return expr

def _is_all_ones_shape(shape: Sequence[int]) -> bool:
    values = [int(v) for v in list(shape)]
    return len(values) > 0 and all(int(v) == 1 for v in values)

def get_supported_pytorch_kernel_op_types() -> Set[str]:
    return set(SUPPORTED_TORCH_KERNEL_OP_TYPES)

def _perm_cl_to_cf(rank: int) -> Optional[List[int]]:
    if rank == 3:
        return [0, 2, 1]
    if rank == 4:
        return [0, 3, 1, 2]
    if rank == 5:
        return [0, 4, 1, 2, 3]
    return None

def _perm_cf_to_cl(rank: int) -> Optional[List[int]]:
    if rank == 3:
        return [0, 2, 1]
    if rank == 4:
        return [0, 2, 3, 1]
    if rank == 5:
        return [0, 2, 3, 4, 1]
    return None

def _permute_shape(values: Optional[Sequence[int]], perm: Sequence[int]) -> Optional[List[int]]:
    if values is None:
        return None
    items = [int(v) for v in list(values)]
    if len(items) != len(list(perm)):
        return None
    return [int(items[idx]) for idx in perm]

def _clone_tensor(tensor: TensorIR) -> TensorIR:
    return TensorIR(
        name=str(tensor.name),
        dtype=str(tensor.dtype),
        shape=[int(v) for v in list(tensor.shape)],
        shape_signature=(
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else None
        ),
        data=np.asarray(tensor.data).copy() if isinstance(tensor.data, np.ndarray) else tensor.data,
        is_variable=bool(tensor.is_variable),
        quantization=copy.deepcopy(tensor.quantization),
        logical_layout=normalize_logical_layout(tensor.logical_layout),
    )

def _read_transpose_perm(model_ir: ModelIR, op: OperatorIR) -> Optional[List[int]]:
    perm_tensor = model_ir.tensors.get(str(op.inputs[1]), None) if len(op.inputs) >= 2 else None
    if perm_tensor is not None and isinstance(perm_tensor.data, np.ndarray):
        perm = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
        if sorted(perm) == list(range(len(perm))):
            return perm
    perm = [int(v) for v in list(op.options.get("perm", []))]
    if len(perm) > 0 and sorted(perm) == list(range(len(perm))):
        return perm
    return None

def _read_onnx_squeeze_axes(node: Any) -> Optional[List[int]]:
    if node is None or str(getattr(node, "op_type", "")) != "Squeeze":
        return None
    for attribute in list(getattr(node, "attribute", [])):
        if str(getattr(attribute, "name", "")) == "axes":
            values = onnx.helper.get_attribute_value(attribute)
            if isinstance(values, (list, tuple)):
                return [int(v) for v in list(values)]
    return None

def _read_onnx_unsqueeze_axes(node: Any) -> Optional[List[int]]:
    if node is None or str(getattr(node, "op_type", "")) != "Unsqueeze":
        return None
    for attribute in list(getattr(node, "attribute", [])):
        if str(getattr(attribute, "name", "")) == "axes":
            values = onnx.helper.get_attribute_value(attribute)
            if isinstance(values, (list, tuple)):
                return [int(v) for v in list(values)]
    return None

def _compose_axis_permutations(
    first: Optional[Sequence[int]],
    second: Optional[Sequence[int]],
) -> Optional[List[int]]:
    if first is None and second is None:
        return None
    if first is None:
        composed = [int(v) for v in list(second or [])]
    elif second is None:
        composed = [int(v) for v in list(first)]
    else:
        first_values = [int(v) for v in list(first)]
        second_values = [int(v) for v in list(second)]
        if len(first_values) != len(second_values):
            return None
        if sorted(first_values) != list(range(len(first_values))):
            return None
        if sorted(second_values) != list(range(len(second_values))):
            return None
        composed = [int(first_values[int(idx)]) for idx in second_values]
    if composed == list(range(len(composed))):
        return None
    return composed

def _inverse_axis_permutation(perm: Optional[Sequence[int]]) -> Optional[List[int]]:
    if perm is None:
        return None
    values = [int(v) for v in list(perm)]
    if sorted(values) != list(range(len(values))):
        return None
    inverse = [0] * len(values)
    for new_axis, old_axis in enumerate(values):
        inverse[int(old_axis)] = int(new_axis)
    return inverse

def _constant_pad_pairs_for_tensor(tensor: Optional[TensorIR]) -> Optional[List[List[int]]]:
    if tensor is None or tensor.data is None:
        return None
    try:
        pads = np.asarray(tensor.data, dtype=np.int64).reshape(-1, 2)
    except Exception:
        return None
    return [[int(v) for v in list(row)] for row in pads.tolist()]

def _pad_output_matches_pre_permuted_input(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    pads_tensor: Optional[TensorIR],
    input_pre_permute: Optional[Sequence[int]],
) -> bool:
    if (
        input_tensor is None
        or output_tensor is None
        or pads_tensor is None
        or input_pre_permute is None
    ):
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    rank = len(input_shape)
    if rank == 0 or len(output_shape) != rank:
        return False
    inverse_perm = _inverse_axis_permutation(input_pre_permute)
    if inverse_perm is None or len(inverse_perm) != rank:
        return False
    pad_pairs = _constant_pad_pairs_for_tensor(pads_tensor)
    if pad_pairs is None:
        return False
    if len(pad_pairs) < rank:
        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs
    elif len(pad_pairs) > rank:
        pad_pairs = pad_pairs[-rank:]
    permuted_input_shape = _permute_shape(input_shape, inverse_perm)
    if permuted_input_shape is None or len(permuted_input_shape) != rank:
        return False
    padded_shape = [
        int(permuted_input_shape[idx]) + int(pad_pairs[idx][0]) + int(pad_pairs[idx][1])
        for idx in range(rank)
    ]
    return padded_shape == output_shape

def _rewrite_vector_constant_inplace(
    *,
    tensor: TensorIR,
    perm: Sequence[int],
    expected_rank: int,
) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.ndim != 1 or int(arr.size) != int(expected_rank):
        return False
    tensor.data = np.asarray([arr[int(idx)] for idx in perm], dtype=arr.dtype)
    tensor.shape = [int(expected_rank)]
    if tensor.shape_signature is not None and len(tensor.shape_signature) == 1:
        tensor.shape_signature = [int(expected_rank)]
    return True

def _rewrite_matrix_constant_inplace(
    *,
    tensor: TensorIR,
    perm: Sequence[int],
    expected_rank: int,
) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.ndim != 2 or tuple(arr.shape) != (int(expected_rank), 2):
        return False
    tensor.data = np.asarray(arr[list(perm), :], dtype=arr.dtype)
    tensor.shape = [int(expected_rank), 2]
    if tensor.shape_signature is not None and len(tensor.shape_signature) == 2:
        tensor.shape_signature = [int(expected_rank), 2]
    return True

def _rewrite_axis_constant_inplace(
    *,
    tensor: TensorIR,
    source_layout: str,
    target_layout: str,
    rank: int,
) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.ndim == 0:
        axis = int(arr.reshape(-1)[0])
        rewritten = rewrite_axis_for_layout(
            axis=axis,
            source_layout=source_layout,
            target_layout=target_layout,
            rank=rank,
        )
        tensor.data = np.asarray(rewritten, dtype=arr.dtype)
        return True
    if arr.ndim != 1:
        return False
    rewritten_axes = [
        rewrite_axis_for_layout(
            axis=int(v),
            source_layout=source_layout,
            target_layout=target_layout,
            rank=rank,
        )
        for v in arr.reshape(-1).tolist()
    ]
    tensor.data = np.asarray(rewritten_axes, dtype=arr.dtype)
    tensor.shape = [int(len(rewritten_axes))]
    tensor.shape_signature = [int(len(rewritten_axes))]
    return True

def _permute_tensor_to_channel_first_inplace(tensor: TensorIR) -> bool:
    source_layout = normalize_logical_layout(tensor.logical_layout)
    rank = len(list(tensor.shape))
    if not is_channel_last_logical_layout(source_layout):
        return False
    target_layout = channel_first_logical_layout(rank)
    perm = logical_layout_permutation(
        source_layout=source_layout,
        target_layout=target_layout,
    )
    if perm is None:
        return False
    permuted_shape = _permute_shape(tensor.shape, perm)
    if permuted_shape is not None:
        tensor.shape = permuted_shape
    if tensor.shape_signature is not None:
        permuted_signature = _permute_shape(tensor.shape_signature, perm)
        if permuted_signature is not None:
            tensor.shape_signature = permuted_signature
    if isinstance(tensor.data, np.ndarray) and int(np.asarray(tensor.data).ndim) == int(rank):
        tensor.data = np.transpose(np.asarray(tensor.data), axes=perm).copy()
    tensor.logical_layout = target_layout
    return True

def _make_unique_identifier(base_name: str, used_names: Set[str]) -> str:
    candidate = str(base_name)
    suffix = 1
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate

def _collect_model_op_types(model_ir: ModelIR) -> Set[str]:
    ops: Set[str] = set()
    for op in model_ir.operators:
        ops.add(str(op.op_type))
    for subgraph in model_ir.subgraphs:
        ops.update(_collect_model_op_types(subgraph))
    return ops

def _preferred_reshape_target_values(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None:
        return None
    if tensor.shape_signature is not None:
        signature = [int(v) for v in list(tensor.shape_signature)]
        if len(signature) == len(list(tensor.shape)) and any(int(v) <= 0 for v in signature):
            return signature
    return [int(v) for v in list(tensor.shape)]
