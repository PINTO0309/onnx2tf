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
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

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
    logical_layout_permutation,
    normalize_logical_layout,
    rewrite_axis_for_layout,
    validate_model_ir_layout_annotations,
)
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    SUPPORTED_TORCH_KERNEL_OP_TYPES,
)
from onnx2tf.tflite_builder.tflite_importer import (
    import_model_ir_from_tflite,
)


class ModelIRPyTorchExportError(RuntimeError):
    pass


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


def _is_layout_only_transpose_by_shape(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    perm: Optional[Sequence[int]],
) -> bool:
    if input_tensor is None or output_tensor is None or perm is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if len(input_shape) != len(output_shape) or len(input_shape) != len(list(perm)):
        return False
    return _permute_shape(input_shape, perm) == output_shape


def _is_standard_channel_layout_permutation(
    *,
    perm: Optional[Sequence[int]],
    rank: int,
) -> bool:
    if perm is None:
        return False
    perm_values = tuple(int(v) for v in list(perm))
    return perm_values in {
        tuple(_perm_cl_to_cf(rank) or []),
        tuple(_perm_cf_to_cl(rank) or []),
    }


def _is_inconsistent_standard_layout_transpose(
    *,
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
    perm: Optional[Sequence[int]],
) -> bool:
    if input_tensor is None or output_tensor is None or perm is None:
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    rank = len(input_shape)
    if rank not in {3, 4, 5} or len(output_shape) != rank:
        return False
    if not _is_standard_channel_layout_permutation(perm=perm, rank=rank):
        return False
    if input_shape != output_shape:
        return False
    permuted_input_shape = _permute_shape(input_shape, perm)
    if permuted_input_shape is None:
        return False
    # Some layout-bridge transposes survive normalization with stale CF metadata.
    # Executing those transposes would violate the declared tensor shape contract.
    return permuted_input_shape != output_shape


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


def _collect_kernel_weight_tensor_names(model_ir: ModelIR) -> Set[str]:
    names: Set[str] = set()
    for op in model_ir.operators:
        if str(op.op_type) in {
            "CONV_2D",
            "DEPTHWISE_CONV_2D",
            "TRANSPOSE_CONV",
            "CONV_3D",
            "CONV_3D_TRANSPOSE",
        } and len(op.inputs) >= 2:
            names.add(str(op.inputs[1]))
    return names


def _should_emit_channel_last_space_to_depth(
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    block_size: int,
) -> Optional[bool]:
    if len(list(input_shape)) != 4 or len(list(output_shape)) != 4:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    if 0 in {int(block_size)}:
        return None
    n, a, b, c = in_shape
    if a % block_size == 0 and b % block_size == 0:
        if out_shape == [n, a // block_size, b // block_size, c * block_size * block_size]:
            return True
    n, c, h, w = in_shape
    if h % block_size == 0 and w % block_size == 0:
        if out_shape == [n, c * block_size * block_size, h // block_size, w // block_size]:
            return False
    return None


def _should_emit_channel_last_depth_to_space(
    *,
    input_shape: Sequence[int],
    output_shape: Sequence[int],
    block_size: int,
) -> Optional[bool]:
    if len(list(input_shape)) != 4 or len(list(output_shape)) != 4:
        return None
    in_shape = [int(v) for v in list(input_shape)]
    out_shape = [int(v) for v in list(output_shape)]
    if 0 in {int(block_size)}:
        return None
    n, h, w, c = in_shape
    if c % (block_size * block_size) == 0:
        if out_shape == [n, h * block_size, w * block_size, c // (block_size * block_size)]:
            return True
    n, c, h, w = in_shape
    if c % (block_size * block_size) == 0:
        if out_shape == [n, c // (block_size * block_size), h * block_size, w * block_size]:
            return False
    return None


def _primary_data_input_name(op: OperatorIR) -> Optional[str]:
    op_type = str(op.op_type)
    if len(op.inputs) == 0:
        return None
    if op_type == "SPLIT":
        return str(op.inputs[1]) if len(op.inputs) >= 2 else str(op.inputs[0])
    if op_type == "SCATTER_ND":
        return str(op.inputs[1]) if len(op.inputs) >= 2 else None
    if op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"}:
        return str(op.inputs[2]) if len(op.inputs) >= 3 else None
    return str(op.inputs[0])


def _assign_tensor_logical_layout(
    tensor: Optional[TensorIR],
    layout: str,
) -> bool:
    if tensor is None:
        return False
    normalized_target = normalize_logical_layout(layout)
    if normalized_target == LOGICAL_LAYOUT_UNKNOWN:
        return False
    current_layout = normalize_logical_layout(tensor.logical_layout)
    if current_layout == normalized_target:
        return False
    if current_layout != LOGICAL_LAYOUT_UNKNOWN:
        current_rank = len(list(tensor.shape))
        current_is_channel_layout = (
            is_channel_first_logical_layout(current_layout)
            or is_channel_last_logical_layout(current_layout)
        )
        target_is_channel_layout = (
            is_channel_first_logical_layout(normalized_target)
            or is_channel_last_logical_layout(normalized_target)
        )
        if current_is_channel_layout and target_is_channel_layout:
            if current_rank != len(list(tensor.shape)):
                return False
    tensor.logical_layout = normalized_target
    return True


def _shared_tensor_layout(
    tensors: Sequence[Optional[TensorIR]],
) -> str:
    layouts: List[str] = []
    for tensor in tensors:
        if tensor is None:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if layout == LOGICAL_LAYOUT_UNKNOWN:
            return LOGICAL_LAYOUT_UNKNOWN
        if not (
            is_channel_first_logical_layout(layout)
            or is_channel_last_logical_layout(layout)
        ):
            return LOGICAL_LAYOUT_UNKNOWN
        layouts.append(layout)
    if len(layouts) == 0:
        return LOGICAL_LAYOUT_UNKNOWN
    first = layouts[0]
    if any(layout != first for layout in layouts[1:]):
        return LOGICAL_LAYOUT_UNKNOWN
    return first


def _is_degenerate_sequence_like_rank4_or_rank5_tensor(
    tensor: Optional[TensorIR],
) -> bool:
    if tensor is None:
        return False
    shape_signature = (
        [int(v) for v in list(tensor.shape_signature)]
        if tensor.shape_signature is not None and len(list(tensor.shape_signature)) == len(list(tensor.shape))
        else [int(v) for v in list(tensor.shape)]
    )
    rank = len(shape_signature)
    if rank not in {4, 5}:
        return False
    if int(shape_signature[0]) not in {1, -1}:
        return False
    if any(int(dim) not in {1, -1} for dim in shape_signature[1:-1]):
        return False
    return int(shape_signature[-1]) > 0


def _is_channel_last_factorized_reshape(
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
) -> bool:
    if input_tensor is None or output_tensor is None:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    if not is_channel_last_logical_layout(input_layout):
        return False
    input_shape = [int(v) for v in list(input_tensor.shape)]
    output_shape = [int(v) for v in list(output_tensor.shape)]
    if len(input_shape) not in {3, 4, 5} or len(output_shape) not in {4, 5}:
        return False
    if len(output_shape) <= len(input_shape):
        return False
    if any(int(v) <= 0 for v in input_shape + output_shape):
        return False
    spatial_shape = input_shape[1:-1]
    spatial_rank = len(spatial_shape)
    if spatial_rank <= 0:
        return False
    if output_shape[0] != input_shape[0]:
        return False
    if output_shape[1:1 + spatial_rank] != spatial_shape:
        return False
    trailing_shape = output_shape[1 + spatial_rank:]
    if len(trailing_shape) < 2:
        return False
    return int(np.prod(trailing_shape, dtype=np.int64)) == int(input_shape[-1])


def _propagate_pytorch_friendly_layouts(model_ir: ModelIR) -> None:
    unary_passthrough_ops = {
        "ABS",
        "ATAN",
        "CEIL",
        "COS",
        "ELU",
        "EXP",
        "FLOOR",
        "HARD_SWISH",
        "IDENTITY",
        "LEAKY_RELU",
        "LOG",
        "LOGICAL_NOT",
        "LOGISTIC",
        "NEG",
        "RELU",
        "RELU6",
        "ROUND",
        "RSQRT",
        "SIGMOID",
        "SIGN",
        "SIN",
        "SQRT",
        "SQUARE",
        "TAN",
        "TANH",
    }
    binary_passthrough_ops = {
        "ADD",
        "DIV",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "POW",
        "SUB",
    }
    resize_pool_passthrough_ops = {
        "AVERAGE_POOL_2D",
        "MAX_POOL_2D",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
    }
    changed = True
    while changed:
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            output_tensors = [
                model_ir.tensors.get(str(output_name), None)
                for output_name in op.outputs
            ]
            if op_type in unary_passthrough_ops and len(op.inputs) >= 1:
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(op.inputs[0]), None)]
                )
            elif op_type in binary_passthrough_ops and len(op.inputs) >= 2:
                propagated_layout = _shared_tensor_layout(
                    [
                        model_ir.tensors.get(str(op.inputs[0]), None),
                        model_ir.tensors.get(str(op.inputs[1]), None),
                    ]
                )
            elif op_type == "CONCATENATION":
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(input_name), None) for input_name in op.inputs]
                )
            elif op_type in {"PACK", "UNPACK"}:
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(input_name), None) for input_name in op.inputs]
                )
            elif op_type == "SPLIT":
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(op.inputs[-1]), None)]
                )
            elif op_type in resize_pool_passthrough_ops:
                propagated_layout = _shared_tensor_layout(
                    [model_ir.tensors.get(str(op.inputs[0]), None)]
                )
            else:
                continue
            if propagated_layout == LOGICAL_LAYOUT_UNKNOWN:
                continue
                for output_tensor in output_tensors:
                    changed = _assign_tensor_logical_layout(output_tensor, propagated_layout) or changed


def _collect_feature_last_sequence_tensor_names(model_ir: ModelIR) -> Set[str]:
    consumers: Dict[str, List[int]] = {}
    producers: Dict[str, int] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_idx)

    def _trace_feature_last_rhs_seed(tensor_name: str) -> Optional[str]:
        visited: Set[str] = set()
        worklist: List[str] = [str(tensor_name)]
        passthrough_ops = {
            "CAST",
            "EXPAND_DIMS",
            "GATHER",
            "GATHER_ND",
            "IDENTITY",
            "RESHAPE",
            "SLICE",
            "SQUEEZE",
            "STRIDED_SLICE",
            "TRANSPOSE",
        }
        while len(worklist) > 0:
            current_name = str(worklist.pop())
            if current_name in visited:
                continue
            visited.add(current_name)
            current_tensor = model_ir.tensors.get(current_name, None)
            if current_tensor is not None:
                current_rank = len(list(current_tensor.shape))
                current_layout = normalize_logical_layout(current_tensor.logical_layout)
                if current_rank in {3, 4, 5} and is_channel_last_logical_layout(current_layout):
                    return current_name
            producer_idx = producers.get(current_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) not in passthrough_ops or len(producer.inputs) == 0:
                continue
            worklist.append(str(producer.inputs[0]))
        return None

    roots: Set[str] = set()
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type == "BATCH_MATMUL" and len(op.inputs) >= 2:
            rhs_seed = _trace_feature_last_rhs_seed(str(op.inputs[1]))
            if rhs_seed is not None:
                roots.add(rhs_seed)
        if op_type == "TRANSPOSE" and len(op.inputs) >= 1 and len(op.outputs) == 1:
            input_name = str(op.inputs[0])
            output_name = str(op.outputs[0])
            output_tensor = model_ir.tensors.get(output_name, None)
            if output_tensor is None:
                continue
            rank = len(list(output_tensor.shape))
            if rank != 3:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if perm != _perm_cf_to_cl(rank):
                continue
            producer_idx = producers.get(input_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) != "RESHAPE" or len(producer.outputs) != 1:
                continue
            roots.add(output_name)
            continue
        if op_type != "RESHAPE" or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        rank = len(list(output_tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None
        if (
            input_tensor is not None
            and len(list(input_tensor.shape)) >= rank
            and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
            and len(list(output_tensor.shape)) >= 1
            and len(list(input_tensor.shape)) >= 1
            and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
        ):
            roots.add(output_name)
            continue
        if _is_channel_last_factorized_reshape(input_tensor, output_tensor):
            roots.add(output_name)
            continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) in {4, 5}:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape[1:], dtype=np.int64))
                == int(np.prod(output_shape[1:], dtype=np.int64))
                and (
                    is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                    or int(input_shape[-1]) == 1
                )
            ):
                for consumer_idx in consumers.get(output_name, []):
                    consumer = model_ir.operators[int(consumer_idx)]
                    if (
                        str(consumer.op_type) != "BATCH_MATMUL"
                        or len(consumer.inputs) < 2
                        or str(consumer.inputs[1]) != output_name
                    ):
                        continue
                    lhs_tensor = model_ir.tensors.get(str(consumer.inputs[0]), None)
                    if lhs_tensor is None or len(list(lhs_tensor.shape)) < 2:
                        continue
                    lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
                    if int(lhs_shape[-1]) != int(output_shape[-2]):
                        continue
                    roots.add(output_name)
                    break
                if output_name in roots:
                    continue
        raw_shape = op.options.get("onnxRawNewShape", None)
        new_shape = op.options.get("newShape", None)
        if not isinstance(raw_shape, list) or not isinstance(new_shape, list):
            continue
        raw_shape_values = [int(v) for v in list(raw_shape)]
        new_shape_values = [int(v) for v in list(new_shape)]
        if raw_shape_values == new_shape_values:
            continue
        if len(raw_shape_values) != rank or len(new_shape_values) != rank:
            continue
        if consumers.get(output_name):
            if any(
                str(model_ir.operators[int(consumer_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(consumer_idx)]) == _perm_cf_to_cl(rank)
                for consumer_idx in consumers.get(output_name, [])
            ):
                continue
        roots.add(output_name)

    preserve_names: Set[str] = set(roots)
    if len(roots) == 0:
        return preserve_names

    layout_passthrough_ops = {
        "ABS",
        "ADD",
        "ATAN",
        "BATCH_MATMUL",
        "BROADCAST_TO",
        "CAST",
        "CEIL",
        "CONCATENATION",
        "COS",
        "DEPTH_TO_SPACE",
        "DIV",
        "ELU",
        "ERF",
        "EXP",
        "EXPAND_DIMS",
        "GATHER",
        "GATHER_ND",
        "GELU",
        "IDENTITY",
        "LEAKY_RELU",
        "LOG",
        "LOGISTIC",
        "MATMUL",
        "MAXIMUM",
        "MEAN",
        "MINIMUM",
        "MUL",
        "NEG",
        "PACK",
        "POW",
        "RELU",
        "RELU6",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SIGMOID",
        "SIGN",
        "SIN",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQRT",
        "SQUARE",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "SUM",
        "TANH",
        "TILE",
        "TRANSPOSE",
        "UNPACK",
    }
    changed = True
    while changed:
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            if op_type not in layout_passthrough_ops:
                continue
            input_names = [str(v) for v in op.inputs]
            output_names = [str(v) for v in op.outputs]
            if len(output_names) == 0:
                continue
            if not any(name in preserve_names for name in input_names):
                continue
            if op_type == "TRANSPOSE" and len(op.outputs) == 1:
                output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
                perm = _read_transpose_perm(model_ir, op)
                if rank in {3, 4, 5} and (
                    perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank)
                ):
                    continue
            for output_name in output_names:
                if output_name not in preserve_names:
                    preserve_names.add(output_name)
                    changed = True
            if not any(name in preserve_names for name in output_names):
                continue
            if op_type == "TRANSPOSE" and len(op.inputs) >= 1:
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                rank = len(list(input_tensor.shape)) if input_tensor is not None else -1
                perm = _read_transpose_perm(model_ir, op)
                if rank in {3, 4, 5} and (
                    perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank)
                ):
                    continue
            for input_name in input_names:
                if input_name not in preserve_names:
                    preserve_names.add(input_name)
                    changed = True
    return preserve_names


def _apply_feature_last_sequence_layouts(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    if len(preserve_channel_last_tensor_names) == 0:
        return
    producers: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_idx)
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))
    for tensor_name in preserve_channel_last_tensor_names:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        tensor.logical_layout = LOGICAL_LAYOUT_UNKNOWN

    for op in model_ir.operators:
        output_name = str(op.outputs[0]) if len(op.outputs) == 1 else None
        if output_name is None or output_name not in preserve_channel_last_tensor_names:
            continue
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None:
            continue
        rank = len(list(output_tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        op_type = str(op.op_type)
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None
        if op_type == "TRANSPOSE":
            perm = _read_transpose_perm(model_ir, op)
            if perm == _perm_cf_to_cl(rank):
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            elif (
                input_tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                and isinstance(perm, list)
                and len(perm) == rank
                and sorted(int(v) for v in perm) == list(range(rank))
                and int(perm[0]) == 0
                and int(perm[-1]) == rank - 1
            ):
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            continue
        if op_type == "RESHAPE":
            should_mark_channel_last = False
            if (
                input_tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
            ):
                should_mark_channel_last = True
            raw_shape = op.options.get("onnxRawNewShape", None)
            if not should_mark_channel_last and isinstance(raw_shape, list):
                raw_shape_values = [int(v) for v in list(raw_shape)]
                if len(raw_shape_values) == rank:
                    current_shape = [int(v) for v in list(output_tensor.shape)]
                    if raw_shape_values != current_shape and raw_shape_values[-1] == current_shape[-1]:
                        should_mark_channel_last = True
            if not should_mark_channel_last and _is_channel_last_factorized_reshape(input_tensor, output_tensor):
                should_mark_channel_last = True
            if should_mark_channel_last:
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            continue

    safe_passthrough_ops = {
        "ABS",
        "ADD",
        "ATAN",
        "CAST",
        "CONCATENATION",
        "DEPTH_TO_SPACE",
        "DIV",
        "ELU",
        "ERF",
        "EXP",
        "EXPAND_DIMS",
        "GELU",
        "IDENTITY",
        "LOGISTIC",
        "MAXIMUM",
        "MEAN",
        "MINIMUM",
        "MUL",
        "NEG",
        "PACK",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SIGMOID",
        "SIGN",
        "SIN",
        "SLICE",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQRT",
        "SQUARE",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUB",
        "SUM",
        "TANH",
        "TILE",
        "UNPACK",
    }
    changed = True
    while changed:
        changed = False
        for op in model_ir.operators:
            op_type = str(op.op_type)
            if op_type not in safe_passthrough_ops or len(op.outputs) == 0:
                continue
            input_tensors = [model_ir.tensors.get(str(name), None) for name in op.inputs]
            if not any(
                tensor is not None
                and is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout))
                for tensor in input_tensors
            ):
                continue
            for output_name in op.outputs:
                output_tensor = model_ir.tensors.get(str(output_name), None)
                if output_tensor is None:
                    continue
                rank = len(list(output_tensor.shape))
                if rank not in {3, 4, 5}:
                    continue
                target_layout = channel_last_logical_layout(rank)
                if normalize_logical_layout(output_tensor.logical_layout) != target_layout:
                    output_tensor.logical_layout = target_layout
                    changed = True
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE" or len(op.outputs) != 1:
            continue
        output_name = str(op.outputs[0])
        if output_name not in preserve_channel_last_tensor_names:
            continue
        if consumers.get(output_name):
            if any(
                str(model_ir.operators[int(consumer_idx)].op_type) == "TRANSPOSE"
                and _read_transpose_perm(model_ir, model_ir.operators[int(consumer_idx)])
                == _perm_cf_to_cl(len(list(model_ir.tensors[output_name].shape)))
                for consumer_idx in consumers.get(output_name, [])
            ):
                continue
        raw_shape = op.options.get("onnxRawNewShape", None)
        if not isinstance(raw_shape, list):
            continue
        raw_shape_values = [int(v) for v in list(raw_shape)]
        output_tensor = model_ir.tensors.get(output_name, None)
        if output_tensor is None or len(raw_shape_values) != len(list(output_tensor.shape)):
            continue
        output_tensor.shape = list(raw_shape_values)
        output_tensor.shape_signature = list(raw_shape_values)
        op.options["newShape"] = list(raw_shape_values)
        if len(op.inputs) >= 2:
            shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                dtype = np.asarray(shape_tensor.data).dtype
                shape_tensor.data = np.asarray(raw_shape_values, dtype=dtype)
                shape_tensor.shape = [int(len(raw_shape_values))]
                shape_tensor.shape_signature = [int(len(raw_shape_values))]


def _rewrite_layout_sensitive_ops(
    model_ir: ModelIR,
    original_layouts: Dict[str, str],
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    rewritten_constant_tensor_names: Set[str] = set()

    def _rewrite_tensor_once(
        tensor_name: str,
        rewrite_fn: Callable[[], bool],
    ) -> None:
        if str(tensor_name) in rewritten_constant_tensor_names:
            return
        if rewrite_fn():
            rewritten_constant_tensor_names.add(str(tensor_name))

    for op in model_ir.operators:
        op_type = str(op.op_type)
        data_input_name = _primary_data_input_name(op)
        data_tensor = model_ir.tensors.get(str(data_input_name), None) if data_input_name is not None else None
        if data_tensor is None:
            continue
        if any(str(name) in preserve_channel_last_tensor_names for name in list(op.inputs) + list(op.outputs)):
            continue
        original_layout = normalize_logical_layout(original_layouts.get(str(data_input_name), data_tensor.logical_layout))
        rank = len(list(data_tensor.shape))
        if rank not in {3, 4, 5} or not is_channel_last_logical_layout(original_layout):
            continue
        target_layout = channel_first_logical_layout(rank)

        if op_type in {"CONCATENATION", "PACK", "UNPACK", "GATHER", "SOFTMAX", "ARG_MAX", "ARG_MIN"}:
            axis = op.options.get("axis", None)
            if axis is not None:
                op.options["axis"] = rewrite_axis_for_layout(
                    axis=int(axis),
                    source_layout=original_layout,
                    target_layout=target_layout,
                    rank=rank,
                )
            if op_type in {"ARG_MAX", "ARG_MIN"} and len(op.inputs) >= 2:
                axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if axis_tensor is not None:
                    _rewrite_tensor_once(
                        str(op.inputs[1]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
                    )
        elif op_type == "SPLIT":
            axis = op.options.get("axis", None)
            if axis is not None:
                op.options["axis"] = rewrite_axis_for_layout(
                    axis=int(axis),
                    source_layout=original_layout,
                    target_layout=target_layout,
                    rank=rank,
                )
            if len(op.inputs) >= 1:
                axis_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                if axis_tensor is not None:
                    _rewrite_tensor_once(
                        str(op.inputs[0]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
                    )
        elif op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            if len(op.inputs) >= 2:
                axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if axis_tensor is not None:
                    _rewrite_tensor_once(
                        str(op.inputs[1]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
                    )
        elif op_type in {"SLICE", "STRIDED_SLICE"}:
            for input_name in op.inputs[1:4]:
                vector_tensor = model_ir.tensors.get(str(input_name), None)
                if vector_tensor is not None:
                    _rewrite_tensor_once(
                        str(input_name),
                        lambda vector_tensor=vector_tensor: _rewrite_vector_constant_inplace(
                            tensor=vector_tensor,
                            perm=logical_layout_permutation(
                                source_layout=original_layout,
                                target_layout=target_layout,
                            ) or [],
                            expected_rank=rank,
                        ),
                    )
        elif op_type in {"PAD", "PADV2", "MIRROR_PAD"} and len(op.inputs) >= 2:
            pad_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if pad_tensor is not None:
                _rewrite_tensor_once(
                    str(op.inputs[1]),
                    lambda: _rewrite_matrix_constant_inplace(
                        tensor=pad_tensor,
                        perm=logical_layout_permutation(
                            source_layout=original_layout,
                            target_layout=target_layout,
                        ) or [],
                        expected_rank=rank,
                    ),
                )
        elif op_type == "TRANSPOSE":
            layout_perm = logical_layout_permutation(
                source_layout=original_layout,
                target_layout=target_layout,
            ) or []
            if len(layout_perm) != rank:
                continue
            old_axis_to_new_axis = [0] * rank
            for new_axis, old_axis in enumerate(layout_perm):
                old_axis_to_new_axis[int(old_axis)] = int(new_axis)
            if len(op.inputs) >= 2:
                perm_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if perm_tensor is not None and isinstance(perm_tensor.data, np.ndarray):
                    perm_values = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
                    if len(perm_values) == rank:
                        def _rewrite_perm_tensor() -> bool:
                            rewritten_perm = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
                            perm_tensor.data = np.asarray(rewritten_perm, dtype=np.asarray(perm_tensor.data).dtype)
                            perm_tensor.shape = [int(rank)]
                            perm_tensor.shape_signature = [int(rank)]
                            return True
                        _rewrite_tensor_once(str(op.inputs[1]), _rewrite_perm_tensor)
            elif "perm" in op.options:
                perm_values = [int(v) for v in list(op.options.get("perm", []))]
                if len(perm_values) == rank:
                    op.options["perm"] = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
        elif op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} and len(op.inputs) >= 1:
            output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            if output_shape_tensor is not None:
                _rewrite_tensor_once(
                    str(op.inputs[0]),
                    lambda: _rewrite_vector_constant_inplace(
                        tensor=output_shape_tensor,
                        perm=logical_layout_permutation(
                            source_layout=original_layout,
                            target_layout=target_layout,
                        ) or [],
                        expected_rank=rank,
                    ),
                )
        elif op_type == "RESHAPE" and len(op.outputs) == 1:
            out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            if out_tensor is not None:
                preferred_shape = _preferred_reshape_target_values(out_tensor)
                if preferred_shape is None:
                    preferred_shape = [int(v) for v in list(out_tensor.shape)]
                if len(op.inputs) >= 2:
                    shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                    if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                        def _rewrite_shape_tensor() -> bool:
                            shape_tensor.data = np.asarray(preferred_shape, dtype=np.asarray(shape_tensor.data).dtype)
                            shape_tensor.shape = [int(len(preferred_shape))]
                            shape_tensor.shape_signature = [int(len(preferred_shape))]
                            return True
                        _rewrite_tensor_once(str(op.inputs[1]), _rewrite_shape_tensor)
                op.options["newShape"] = list(preferred_shape)


def _rewrite_filter_tensors_for_pytorch(model_ir: ModelIR) -> None:
    rewritten_weights: Set[str] = set()
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type not in {"CONV_2D", "DEPTHWISE_CONV_2D", "TRANSPOSE_CONV", "CONV_3D", "CONV_3D_TRANSPOSE"}:
            continue
        if len(op.inputs) < 2:
            continue
        weight_name = str(op.inputs[1])
        if weight_name in rewritten_weights:
            continue
        tensor = model_ir.tensors.get(weight_name, None)
        if tensor is None or not isinstance(tensor.data, np.ndarray):
            continue
        arr = np.asarray(tensor.data)
        if op_type == "CONV_2D" and arr.ndim == 4:
            tensor.data = np.transpose(arr, (0, 3, 1, 2)).copy()
        elif op_type == "DEPTHWISE_CONV_2D" and arr.ndim == 4:
            permuted = np.transpose(arr, (3, 0, 1, 2)).copy()
            tensor.data = permuted.reshape(int(permuted.shape[0] * permuted.shape[1]), 1, int(permuted.shape[2]), int(permuted.shape[3]))
        elif op_type == "TRANSPOSE_CONV" and arr.ndim == 4:
            tensor.data = np.transpose(arr, (3, 0, 1, 2)).copy()
        elif op_type in {"CONV_3D", "CONV_3D_TRANSPOSE"} and arr.ndim == 5:
            if is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout)):
                tensor.data = np.transpose(arr, (4, 3, 0, 1, 2)).copy()
            else:
                continue
        else:
            continue
        tensor.shape = [int(v) for v in list(tensor.data.shape)]
        if tensor.shape_signature is not None and len(list(tensor.shape_signature)) == int(arr.ndim):
            tensor.shape_signature = [int(v) for v in list(tensor.shape)]
        rewritten_weights.add(weight_name)


def _synchronize_reshape_targets_with_output_tensors(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    for op in model_ir.operators:
        if str(op.op_type) != "RESHAPE" or len(op.outputs) != 1:
            continue
        if str(op.outputs[0]) in preserve_channel_last_tensor_names:
            continue
        out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if out_tensor is None:
            continue
        preferred_shape = _preferred_reshape_target_values(out_tensor)
        if preferred_shape is None or len(preferred_shape) == 0:
            continue
        op.options["newShape"] = list(preferred_shape)
        if len(op.inputs) < 2:
            continue
        shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        if shape_tensor is None or not isinstance(shape_tensor.data, np.ndarray):
            continue
        dtype = np.asarray(shape_tensor.data).dtype
        shape_tensor.data = np.asarray(preferred_shape, dtype=dtype)
        shape_tensor.shape = [int(len(preferred_shape))]
        shape_tensor.shape_signature = [int(len(preferred_shape))]


def _remove_redundant_layout_transposes(
    model_ir: ModelIR,
    original_layouts: Dict[str, str],
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    delete_op_indices: Set[int] = set()
    for op_idx, op in enumerate(model_ir.operators):
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 1 or len(op.outputs) != 1:
            continue
        input_name = str(op.inputs[0])
        output_name = str(op.outputs[0])
        if input_name in preserve_channel_last_tensor_names or output_name in preserve_channel_last_tensor_names:
            continue
        input_tensor = model_ir.tensors.get(input_name, None)
        output_tensor = model_ir.tensors.get(output_name, None)
        reference_tensor = output_tensor if output_tensor is not None else input_tensor
        rank = len(list(reference_tensor.shape)) if reference_tensor is not None else -1
        if rank not in {3, 4, 5}:
            continue
        perm = _read_transpose_perm(model_ir, op)
        input_layout = normalize_logical_layout(original_layouts.get(input_name, LOGICAL_LAYOUT_UNKNOWN))
        output_layout = normalize_logical_layout(original_layouts.get(output_name, LOGICAL_LAYOUT_UNKNOWN))
        remove_as_identity = bool(
            perm is not None
            and (
                perm == list(range(rank))
                or _is_layout_only_transpose_by_shape(
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    perm=perm,
                )
                and _is_standard_channel_layout_permutation(
                    perm=perm,
                    rank=rank,
                )
                or
                (
                    is_channel_last_logical_layout(input_layout)
                    and perm == logical_layout_permutation(
                        source_layout=input_layout,
                        target_layout=channel_first_logical_layout(rank),
                    )
                )
                or (
                    is_channel_last_logical_layout(output_layout)
                    and perm == logical_layout_permutation(
                        source_layout=channel_first_logical_layout(rank),
                        target_layout=output_layout,
                    )
                )
                or _is_inconsistent_standard_layout_transpose(
                    input_tensor=input_tensor,
                    output_tensor=output_tensor,
                    perm=perm,
                )
            )
        )
        if not remove_as_identity:
            continue
        if output_name in model_ir.outputs:
            source_tensor = input_tensor if input_tensor is not None else output_tensor
            if source_tensor is not None:
                replacement = _clone_tensor(source_tensor)
                replacement.name = output_name
                model_ir.tensors[output_name] = replacement
            model_ir.operators[int(op_idx)] = OperatorIR(
                op_type="IDENTITY",
                inputs=[input_name],
                outputs=[output_name],
                options={},
            )
            continue
        for consumer_idx in consumers.get(output_name, []):
            consumer = model_ir.operators[int(consumer_idx)]
            consumer.inputs = [input_name if str(v) == output_name else str(v) for v in consumer.inputs]
        delete_op_indices.add(int(op_idx))
        model_ir.tensors.pop(output_name, None)

    if len(delete_op_indices) > 0:
        model_ir.operators = [
            op for op_idx, op in enumerate(model_ir.operators) if int(op_idx) not in delete_op_indices
        ]


def _rewrite_atan2_ones_like_to_atan(model_ir: ModelIR) -> None:
    for op in model_ir.operators:
        if str(op.op_type) != "ATAN2" or len(op.inputs) != 2 or len(op.outputs) != 1:
            continue
        lhs_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        rhs_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
        out_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if (
            lhs_tensor is None
            or rhs_tensor is None
            or out_tensor is None
            or not isinstance(rhs_tensor.data, np.ndarray)
        ):
            continue
        rhs_values = np.asarray(rhs_tensor.data)
        if rhs_values.size == 0 or not np.allclose(rhs_values, 1.0):
            continue
        lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
        out_shape = [int(v) for v in list(out_tensor.shape)]
        if lhs_shape != out_shape:
            continue
        rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
        if rhs_shape != lhs_shape:
            perm = _perm_cl_to_cf(len(rhs_shape))
            perm_inv = _perm_cf_to_cl(len(rhs_shape))
            if (
                perm is None
                or _permute_shape(rhs_shape, perm) != lhs_shape
            ) and (
                perm_inv is None
                or _permute_shape(rhs_shape, perm_inv) != lhs_shape
            ):
                continue
        op.op_type = "ATAN"
        op.inputs = [str(op.inputs[0])]


def _has_recurrent_sequence_context(model_ir: ModelIR) -> bool:
    recurrent_op_types = {
        "GRU",
        "LSTM",
        "RNN",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if any(str(op.op_type) in recurrent_op_types for op in model_ir.operators):
        return True
    recurrent_name_tokens = ("_gru_", "_lstm_", "_rnn_")
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs) + list(model_ir.tensors.keys()):
        lowered = str(tensor_name).lower()
        if any(token in lowered for token in recurrent_name_tokens):
            return True
    return False


def _repair_orphan_recurrent_step_tensors(model_ir: ModelIR) -> None:
    producer_index: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producer_index[str(output_name)] = int(op_idx)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    for tensor_name in list(model_ir.tensors.keys()):
        tensor_name = str(tensor_name)
        if tensor_name in producer_index or tensor_name in set(str(v) for v in model_ir.inputs):
            continue
        match = re.match(r"^(.+_(?:h|c)_step_)(\d+)$", tensor_name)
        if match is None:
            continue
        shape_tensor_name = f"{match.group(1)}shape_{match.group(2)}"
        replacement_name: Optional[str] = None
        for op in model_ir.operators:
            if str(op.op_type) != "RESHAPE" or len(op.inputs) < 2 or len(op.outputs) != 1:
                continue
            if str(op.inputs[1]) != shape_tensor_name:
                continue
            candidate_name = str(op.outputs[0])
            if candidate_name == tensor_name:
                replacement_name = None
                break
            replacement_name = candidate_name
            break
        if replacement_name is None:
            continue
        for consumer_idx in consumers.get(tensor_name, []):
            consumer = model_ir.operators[int(consumer_idx)]
            consumer.inputs = [
                replacement_name if str(input_name) == tensor_name else str(input_name)
                for input_name in consumer.inputs
            ]
        if tensor_name not in set(str(v) for v in model_ir.outputs):
            model_ir.tensors.pop(tensor_name, None)


def _reject_residual_layout_transposes(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    for op in model_ir.operators:
        if str(op.op_type) != "TRANSPOSE":
            continue
        related_tensor_names = [str(v) for v in list(op.inputs) + list(op.outputs)]
        if any(name in preserve_channel_last_tensor_names for name in related_tensor_names):
            continue
        recurrent_sequence_context = any(
            token in name.lower()
            for name in related_tensor_names
            for token in ("_gru_", "_lstm_", "_rnn_")
        )
        output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
        output_tensor = model_ir.tensors.get(output_name, None)
        rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
        if rank not in {3, 4, 5}:
            continue
        if recurrent_sequence_context and rank == 3:
            continue
        input_name = str(op.inputs[0]) if len(op.inputs) > 0 else ""
        input_tensor = model_ir.tensors.get(input_name, None)
        input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        output_layout = normalize_logical_layout(output_tensor.logical_layout) if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
        if input_layout == LOGICAL_LAYOUT_UNKNOWN and output_layout == LOGICAL_LAYOUT_UNKNOWN:
            continue
        perm = _read_transpose_perm(model_ir, op)
        if _is_layout_only_transpose_by_shape(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
            perm=perm,
        ):
            continue
        if _is_reshape_only_residual_layout_bridge_transpose(
            model_ir=model_ir,
            op=op,
            consumers=consumers,
        ):
            continue
        if perm == _perm_cl_to_cf(rank) or perm == _perm_cf_to_cl(rank):
            raise ModelIRPyTorchExportError(
                "Channel-first normalization failed: residual layout transpose remains. "
                f"op_type={op.op_type} outputs={op.outputs} perm={perm}"
            )


def _is_reshape_only_residual_layout_bridge_transpose(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
    consumers: Optional[Dict[str, List[int]]] = None,
) -> bool:
    if str(op.op_type) != "TRANSPOSE":
        return False
    output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
    output_tensor = model_ir.tensors.get(output_name, None)
    rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
    if rank not in {3, 4, 5}:
        return False
    input_name = str(op.inputs[0]) if len(op.inputs) > 0 else ""
    input_tensor = model_ir.tensors.get(input_name, None)
    if input_tensor is None or output_tensor is None:
        return False
    perm = _read_transpose_perm(model_ir, op)
    if perm != _perm_cl_to_cf(rank) and perm != _perm_cf_to_cl(rank):
        return False
    if [int(v) for v in list(input_tensor.shape)] != [int(v) for v in list(output_tensor.shape)]:
        return False
    if normalize_logical_layout(input_tensor.logical_layout) != normalize_logical_layout(output_tensor.logical_layout):
        return False
    if consumers is None:
        consumers = {}
        for op_idx, candidate in enumerate(model_ir.operators):
            for candidate_input_name in candidate.inputs:
                consumers.setdefault(str(candidate_input_name), []).append(int(op_idx))
    user_indices = [int(v) for v in consumers.get(output_name, [])]
    return len(user_indices) > 0 and all(
        str(model_ir.operators[int(user_idx)].op_type) == "RESHAPE"
        for user_idx in user_indices
    )


def _align_public_boundary_shapes_to_onnx_contract(model_ir: ModelIR) -> None:
    boundary_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(boundary_map, dict):
        return
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    producer_index: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producer_index[str(output_name)] = int(op_idx)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))
    recurrent_sequence_context = _has_recurrent_sequence_context(model_ir)
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        boundary_shape = boundary_map.get(str(tensor_name), None)
        if tensor is None or not isinstance(boundary_shape, list):
            continue
        if len(boundary_shape) != len(list(tensor.shape)):
            continue
        rank = len(list(tensor.shape))
        preserve_runtime_boundary_shape = False
        if recurrent_sequence_context and rank == 3:
            if str(tensor_name) in set(str(v) for v in model_ir.inputs):
                preserve_runtime_boundary_shape = any(
                    str(model_ir.operators[int(op_idx)].op_type) == "TRANSPOSE"
                    for op_idx in consumers.get(str(tensor_name), [])
                )
            else:
                producer_op_idx = producer_index.get(str(tensor_name), None)
                preserve_runtime_boundary_shape = (
                    producer_op_idx is not None
                    and str(model_ir.operators[int(producer_op_idx)].op_type) in {"TRANSPOSE", "RESHAPE"}
                )
        if preserve_runtime_boundary_shape:
            continue
        tensor.shape_signature = [int(v) for v in list(boundary_shape)]
        tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(boundary_shape)]
        if rank in {3, 4, 5}:
            tensor.logical_layout = normalize_logical_layout(
                public_layout_map.get(
                    str(tensor_name),
                    channel_first_logical_layout(rank),
                )
            )


def validate_channel_first_exportability(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    layout_sensitive_ops = {
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "DEPTHWISE_CONV_2D",
        "DEPTH_TO_SPACE",
        "GATHER",
        "GATHER_ND",
        "MAX_POOL_2D",
        "MEAN",
        "MIRROR_PAD",
        "PAD",
        "PADV2",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SCATTER_ND",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "STRIDED_SLICE",
        "TRANSPOSE_CONV",
        "UNPACK",
    }
    problems: List[str] = []
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type not in layout_sensitive_ops:
            continue
        related_tensor_names = [
            str(v) for v in list(op.inputs) + list(op.outputs)
        ]
        related_lowered_names = [name.lower() for name in related_tensor_names]
        recurrent_sequence_context = any(
            token in name
            for name in related_lowered_names
            for token in ("_gru_", "_lstm_", "_rnn_")
        )
        tensor_names: List[str] = []
        primary_name = _primary_data_input_name(op)
        if primary_name is not None:
            tensor_names.append(str(primary_name))
        tensor_names.extend(str(v) for v in list(op.outputs))
        for tensor_name in tensor_names:
            tensor = model_ir.tensors.get(str(tensor_name), None)
            if tensor is None:
                continue
            rank = len(list(tensor.shape))
            if rank not in {3, 4, 5}:
                continue
            if str(tensor_name) in preserve_channel_last_tensor_names:
                continue
            if recurrent_sequence_context and op_type in {"CONCATENATION", "SLICE", "STRIDED_SLICE", "SPLIT"}:
                continue
            layout = normalize_logical_layout(tensor.logical_layout)
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and op_type == "SCATTER_ND"
                and primary_name is not None
                and str(tensor_name) == str(primary_name)
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and op_type == "SCATTER_ND"
                and str(tensor_name) in {str(v) for v in list(op.outputs)}
                and str(tensor_name) not in {str(v) for v in list(model_ir.outputs)}
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and rank in {3, 5}
                and op_type in {"CONCATENATION", "GATHER", "GATHER_ND", "SLICE", "SPLIT", "STRIDED_SLICE"}
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and rank in {4, 5}
                and op_type in {"GATHER", "GATHER_ND", "SLICE", "SPLIT", "STRIDED_SLICE"}
                and _is_degenerate_sequence_like_rank4_or_rank5_tensor(tensor)
            ):
                continue
            if (
                layout == LOGICAL_LAYOUT_UNKNOWN
                and op_type == "SOFTMAX"
                and (
                    _is_attention_like_softmax_op(model_ir, op)
                    or _is_transpose_sandwiched_last_axis_softmax_op(model_ir, op)
                )
            ):
                continue
            if layout == LOGICAL_LAYOUT_UNKNOWN or is_channel_last_logical_layout(layout):
                problems.append(
                    f"op_type={op_type} tensor={tensor_name} logical_layout={layout}"
                )
    if len(problems) > 0:
        raise ModelIRPyTorchExportError(
            "Channel-first normalization failed: semantic layout annotations are incomplete. "
            f"problems={sorted(set(problems))}"
        )


def normalize_model_ir_for_pytorch_channel_first(model_ir: ModelIR) -> ModelIR:
    normalized = copy.deepcopy(model_ir)
    infer_model_ir_logical_layouts(normalized)
    preserve_channel_last_tensor_names = _collect_feature_last_sequence_tensor_names(normalized)
    _apply_feature_last_sequence_layouts(normalized, preserve_channel_last_tensor_names)
    if len(preserve_channel_last_tensor_names) > 0:
        infer_model_ir_logical_layouts(normalized)
    annotation_problems = validate_model_ir_layout_annotations(normalized)
    if len(annotation_problems) > 0:
        raise ModelIRPyTorchExportError(
            "Channel-first normalization failed: invalid semantic layout annotations. "
            f"problems={annotation_problems}"
        )
    original_layouts = {
        str(name): normalize_logical_layout(tensor.logical_layout)
        for name, tensor in normalized.tensors.items()
    }
    _rewrite_layout_sensitive_ops(normalized, original_layouts, preserve_channel_last_tensor_names)
    _propagate_pytorch_friendly_layouts(normalized)
    kernel_weight_tensor_names = _collect_kernel_weight_tensor_names(normalized)
    for tensor_name, tensor in normalized.tensors.items():
        if str(tensor_name) in kernel_weight_tensor_names:
            continue
        if str(tensor_name) in preserve_channel_last_tensor_names:
            continue
        _permute_tensor_to_channel_first_inplace(tensor)
    _synchronize_reshape_targets_with_output_tensors(normalized, preserve_channel_last_tensor_names)
    _rewrite_filter_tensors_for_pytorch(normalized)
    _remove_redundant_layout_transposes(normalized, original_layouts, preserve_channel_last_tensor_names)
    _propagate_pytorch_friendly_layouts(normalized)
    _apply_feature_last_sequence_layouts(normalized, preserve_channel_last_tensor_names)
    _rewrite_atan2_ones_like_to_atan(normalized)
    _repair_orphan_recurrent_step_tensors(normalized)
    _align_public_boundary_shapes_to_onnx_contract(normalized)
    normalized.metadata["assume_channel_last_layout_tensor_names"] = []
    _reject_residual_layout_transposes(normalized, preserve_channel_last_tensor_names)
    validate_channel_first_exportability(normalized, preserve_channel_last_tensor_names)
    return normalized


def _is_attention_like_softmax_op(model_ir: ModelIR, op: OperatorIR) -> bool:
    if str(op.op_type) != "SOFTMAX":
        return False
    reference_tensor: Optional[TensorIR] = None
    if len(op.inputs) > 0:
        reference_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
    if reference_tensor is None and len(op.outputs) > 0:
        reference_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if reference_tensor is None:
        return False
    shape = [int(v) for v in list(reference_tensor.shape)]
    rank = len(shape)
    if rank < 3:
        return False
    axis = op.options.get("axis", None)
    resolved_axis = int(axis) if axis is not None else rank - 1
    if resolved_axis < 0:
        resolved_axis += rank
    if resolved_axis != rank - 1:
        return False
    if int(shape[-1]) <= 1:
        return False
    output_name = str(op.outputs[0]) if len(op.outputs) > 0 else ""
    if output_name != "":
        for consumer in model_ir.operators:
            if output_name not in [str(v) for v in consumer.inputs]:
                continue
            if str(consumer.op_type) == "BATCH_MATMUL":
                return True
    if rank == 3 and int(shape[-2]) == int(shape[-1]):
        return True
    if rank >= 4 and int(shape[-2]) == int(shape[-1]) and 0 < int(shape[-3]) <= 64:
        return True
    return False


def _is_transpose_sandwiched_last_axis_softmax_op(model_ir: ModelIR, op: OperatorIR) -> bool:
    if str(op.op_type) != "SOFTMAX" or len(op.inputs) < 1 or len(op.outputs) != 1:
        return False
    input_name = str(op.inputs[0])
    output_name = str(op.outputs[0])
    input_tensor = model_ir.tensors.get(input_name, None)
    output_tensor = model_ir.tensors.get(output_name, None)
    if input_tensor is None or output_tensor is None:
        return False
    rank = len(list(input_tensor.shape))
    if rank not in {3, 4, 5} or len(list(output_tensor.shape)) != rank:
        return False
    axis = int(op.options.get("axis", rank - 1))
    if axis < 0:
        axis += rank
    if axis != rank - 1:
        return False

    producer_op: Optional[OperatorIR] = None
    for candidate in model_ir.operators:
        if input_name in [str(v) for v in candidate.outputs]:
            producer_op = candidate
            break
    if producer_op is None or str(producer_op.op_type) != "TRANSPOSE" or len(producer_op.inputs) < 1:
        return False
    producer_perm = _read_transpose_perm(model_ir, producer_op)
    if (
        producer_perm is None
        or len(producer_perm) != rank
        or sorted(int(v) for v in producer_perm) != list(range(rank))
        or [int(v) for v in producer_perm] == list(range(rank))
    ):
        return False

    consumer_ops = [
        candidate
        for candidate in model_ir.operators
        if output_name in [str(v) for v in candidate.inputs]
    ]
    if len(consumer_ops) != 1:
        return False
    consumer_op = consumer_ops[0]
    if str(consumer_op.op_type) != "TRANSPOSE" or len(consumer_op.outputs) != 1:
        return False
    consumer_perm = _read_transpose_perm(model_ir, consumer_op)
    if consumer_perm is None or len(consumer_perm) != rank:
        return False
    inverse_perm = [0] * rank
    for new_axis, old_axis in enumerate(producer_perm):
        inverse_perm[int(old_axis)] = int(new_axis)
    if [int(v) for v in consumer_perm] != inverse_perm:
        return False

    source_tensor = model_ir.tensors.get(str(producer_op.inputs[0]), None)
    restored_tensor = model_ir.tensors.get(str(consumer_op.outputs[0]), None)
    if source_tensor is None or restored_tensor is None:
        return False
    source_layout = normalize_logical_layout(source_tensor.logical_layout)
    restored_layout = normalize_logical_layout(restored_tensor.logical_layout)
    if (
        source_layout == LOGICAL_LAYOUT_UNKNOWN
        or restored_layout == LOGICAL_LAYOUT_UNKNOWN
        or source_layout != restored_layout
    ):
        return False
    return True


def _is_layout_agnostic_native_model_ir(model_ir: ModelIR) -> bool:
    channel_sensitive_ops = {
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "TRANSPOSE_CONV",
        "CONV_3D",
        "CONV_3D_TRANSPOSE",
        "MAX_POOL_2D",
        "AVERAGE_POOL_2D",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "NON_MAX_SUPPRESSION_V4",
    }
    op_types = _collect_model_op_types(model_ir)
    return len(op_types & channel_sensitive_ops) == 0


def prepare_model_ir_for_native_pytorch(model_ir: ModelIR) -> ModelIR:
    try:
        return normalize_model_ir_for_pytorch_channel_first(model_ir)
    except ModelIRPyTorchExportError:
        if not _is_layout_agnostic_native_model_ir(model_ir):
            raise
    prepared = copy.deepcopy(model_ir)
    infer_model_ir_logical_layouts(prepared)
    prepared.metadata["assume_channel_last_layout_tensor_names"] = []
    return prepared


def _collect_model_op_types(model_ir: ModelIR) -> Set[str]:
    ops: Set[str] = set()
    for op in model_ir.operators:
        ops.add(str(op.op_type))
    for subgraph in model_ir.subgraphs:
        ops.update(_collect_model_op_types(subgraph))
    return ops


def _ensure_supported_ops(model_ir: ModelIR) -> None:
    unsupported = sorted(
        {
            op_type
            for op_type in _collect_model_op_types(model_ir)
            if op_type not in SUPPORTED_TORCH_KERNEL_OP_TYPES
            and op_type not in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES
            and op_type not in {"MODEL"}
        }
    )
    if len(unsupported) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support some op types in this model. "
            f"unsupported_op_types={unsupported}"
        )


def _ensure_no_custom_ops(model_ir: ModelIR) -> None:
    custom_ops = sorted({str(op.op_type) for op in model_ir.operators if str(op.op_type) == "CUSTOM"})
    if len(custom_ops) > 0:
        raise ModelIRPyTorchExportError(
            "ModelIR->PyTorch exporter does not support CUSTOM ops."
        )


def _serializable_tensor_meta(tensor: TensorIR) -> Dict[str, Any]:
    return {
        "dtype": str(tensor.dtype),
        "shape": [int(v) for v in list(tensor.shape)],
        "shape_signature": (
            [int(v) for v in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else None
        ),
        "is_variable": bool(tensor.is_variable),
        "has_data": bool(isinstance(tensor.data, np.ndarray)),
        "logical_layout": normalize_logical_layout(tensor.logical_layout),
    }


def _serializable_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _serializable_value(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_serializable_value(v) for v in value]
    if isinstance(value, list):
        return [_serializable_value(v) for v in value]
    return value


def _build_metadata_payload(model_ir: ModelIR) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": {
            str(name): _serializable_tensor_meta(tensor)
            for name, tensor in model_ir.tensors.items()
        },
        "operators": [
            {
                "op_type": str(op.op_type),
                "inputs": [str(v) for v in op.inputs],
                "outputs": [str(v) for v in op.outputs],
                "options": _serializable_value(dict(op.options)),
                "axis_semantics": _serializable_value(dict(op.axis_semantics)),
                "version": int(op.version),
            }
            for op in model_ir.operators
        ],
        "public_layouts": _serializable_value(dict(model_ir.metadata.get("onnx_public_layout_map", {}))),
    }


def _is_runtime_wrapper_package_dir(package_dir: Path) -> bool:
    model_path = package_dir / "model.py"
    if not model_path.exists():
        return False
    try:
        model_source = model_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "load_generated_model_package" in model_source


_NUMPY_DTYPE_BY_TENSOR_DTYPE: Dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "INT8": np.dtype(np.int8),
    "INT16": np.dtype(np.int16),
    "INT32": np.dtype(np.int32),
    "INT64": np.dtype(np.int64),
    "UINT8": np.dtype(np.uint8),
    "FLOAT16": np.dtype(np.float16),
    "FLOAT32": np.dtype(np.float32),
    "FLOAT64": np.dtype(np.float64),
}


def _parse_torchscript_shape_hints(
    shape_hints: Optional[List[str]],
) -> Dict[str, List[int]]:
    if shape_hints is None:
        return {}
    parsed: Dict[str, List[int]] = {}
    for hint in shape_hints:
        parts = str(hint).split(":", maxsplit=1)
        if len(parts) != 2:
            continue
        input_name = str(parts[0]).strip()
        shape_str = str(parts[1]).strip()
        if input_name == "" or shape_str == "":
            continue
        try:
            parsed[input_name] = [int(v) for v in shape_str.split(",")]
        except Exception:
            continue
    return parsed


def _lookup_torchscript_shape_hint(
    *,
    input_name: str,
    shape_hints: Dict[str, List[int]],
    normalized_shape_hints: Dict[str, List[int]],
    normalize_name: Callable[[str], str],
) -> Optional[List[int]]:
    direct = shape_hints.get(str(input_name), None)
    if direct is not None:
        return [int(v) for v in list(direct)]
    normalized = normalized_shape_hints.get(normalize_name(str(input_name)), None)
    if normalized is not None:
        return [int(v) for v in list(normalized)]
    return None


def _resolve_torchscript_trace_shape(
    *,
    input_name: str,
    shape_values: Sequence[Any],
    shape_hint: Optional[Sequence[int]],
    export_label: str = "TorchScript export",
) -> Tuple[int, ...]:
    base_shape = [int(v) for v in list(shape_values)]
    if shape_hint is None:
        return _sanitize_torchscript_trace_shape(base_shape)
    hint_values = [int(v) for v in list(shape_hint)]
    if len(hint_values) != len(base_shape):
        raise ModelIRPyTorchExportError(
            f"{export_label} shape_hints rank mismatch. "
            f"input={input_name} expected_rank={len(base_shape)} actual_rank={len(hint_values)}"
        )
    resolved: List[int] = []
    for dim, hint_dim in zip(base_shape, hint_values):
        if int(dim) > 0:
            resolved.append(int(dim))
        elif int(hint_dim) > 0:
            resolved.append(int(hint_dim))
        else:
            raise ModelIRPyTorchExportError(
                f"{export_label} shape_hints must provide positive values for dynamic dimensions. "
                f"input={input_name} shape_hint={hint_values}"
            )
    return tuple(resolved)


def _load_torchscript_test_data_nhwc(
    test_data_nhwc_path: Optional[str],
) -> Optional[np.ndarray]:
    if not test_data_nhwc_path:
        return None
    if not os.path.exists(test_data_nhwc_path):
        raise FileNotFoundError(
            f"test_data_nhwc_path does not exist. path={test_data_nhwc_path}"
        )
    data = np.asarray(np.load(test_data_nhwc_path))
    if data.ndim != 4:
        raise ValueError(
            "test_data_nhwc_path must contain a 4D array [N,H,W,C]. "
            f"actual_shape={tuple(data.shape)}"
        )
    if int(data.shape[-1]) != 3:
        raise ValueError(
            "test_data_nhwc_path must have 3 channels in the last dim. "
            f"actual_shape={tuple(data.shape)}"
        )
    if int(data.shape[0]) <= 0:
        raise ValueError(
            "test_data_nhwc_path must include at least 1 sample. "
            f"actual_shape={tuple(data.shape)}"
        )
    return data


def _build_torchscript_image_input_from_nhwc(
    *,
    data: np.ndarray,
    expected_shape: Tuple[int, ...],
    np_dtype: np.dtype,
) -> np.ndarray:
    import tensorflow as tf

    if len(expected_shape) != 4:
        raise ValueError(
            "test_data_nhwc_path can only be used for rank-4 inputs. "
            f"expected_shape={expected_shape}"
        )

    expected_batch = int(expected_shape[0]) if int(expected_shape[0]) > 0 else int(data.shape[0])
    if data.shape[0] >= expected_batch:
        sample = np.asarray(data[:expected_batch])
    else:
        repeats = int(np.ceil(expected_batch / data.shape[0]))
        sample = np.concatenate([data] * repeats, axis=0)[:expected_batch]

    if int(expected_shape[1]) == 3:
        target_h = int(expected_shape[2])
        target_w = int(expected_shape[3])
        if int(sample.shape[1]) != target_h or int(sample.shape[2]) != target_w:
            sample = np.asarray(tf.image.resize(sample, [target_h, target_w]))
        sample = np.transpose(sample, [0, 3, 1, 2])
    elif int(expected_shape[3]) == 3:
        target_h = int(expected_shape[1])
        target_w = int(expected_shape[2])
        if int(sample.shape[1]) != target_h or int(sample.shape[2]) != target_w:
            sample = np.asarray(tf.image.resize(sample, [target_h, target_w]))
    else:
        raise ValueError(
            "test_data_nhwc_path can only be used for 3-channel image inputs. "
            f"expected_shape={expected_shape}"
        )
    return np.asarray(sample).astype(np_dtype, copy=False)


def _sanitize_torchscript_file_stem(name: str, *, fallback: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(name)).strip("_")
    if sanitized == "":
        sanitized = re.sub(r"[^0-9A-Za-z_]+", "_", str(fallback)).strip("_")
    if sanitized == "":
        sanitized = "model"
    return sanitized


def _sanitize_torchscript_trace_shape(values: Sequence[Any]) -> Tuple[int, ...]:
    sanitized: List[int] = []
    for value in list(values):
        dim = int(value)
        sanitized.append(dim if dim > 0 else 1)
    return tuple(sanitized)


def _can_autoresolve_batch_only_trace_shape(shape_values: Sequence[Any]) -> bool:
    values = [int(v) for v in list(shape_values)]
    if len(values) == 0:
        return False
    if int(values[0]) > 0:
        return False
    return all(int(v) > 0 for v in values[1:])


def _build_pytorch_export_example_inputs(
    *,
    package_dir: str,
    package_metadata: Dict[str, Any],
    custom_input_op_name_np_data_path: Optional[List[Any]],
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    export_label: str = "PyTorch export",
) -> Tuple[Tuple[Any, ...], Dict[str, List[int]], bool]:
    from onnx2tf.tflite_builder.accuracy_evaluator import (
        _generate_seeded_input,
        _extract_sample_from_custom,
        _fill_length_like_input,
        _load_custom_input_data,
        _normalize_tensor_name,
    )
    from onnx2tf.tflite_builder.pytorch_accuracy_evaluator import (
        _convert_inputs_for_package,
        _generate_string_input,
        _is_string_dtype,
    )

    input_names = [str(v) for v in list(package_metadata.get("inputs", []))]
    tensor_meta_map = package_metadata.get("tensors", {})
    if not isinstance(tensor_meta_map, dict):
        tensor_meta_map = {}
    custom_inputs = _load_custom_input_data(custom_input_op_name_np_data_path)
    test_data_nhwc = _load_torchscript_test_data_nhwc(test_data_nhwc_path)
    parsed_shape_hints = _parse_torchscript_shape_hints(shape_hints)
    normalized_shape_hints = {
        _normalize_tensor_name(str(input_name)): [int(v) for v in list(shape_value)]
        for input_name, shape_value in parsed_shape_hints.items()
    }
    normalized_custom_inputs = {
        _normalize_tensor_name(str(input_name)): value
        for input_name, value in custom_inputs.items()
    }

    def _lookup_custom_input(input_name: str) -> Optional[np.ndarray]:
        custom_value = custom_inputs.get(str(input_name), None)
        if custom_value is not None:
            return custom_value
        return normalized_custom_inputs.get(_normalize_tensor_name(str(input_name)), None)

    input_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]] = []
    dynamic_inputs_present = False
    missing_dynamic_hints: List[str] = []
    generated_inputs_np: Dict[str, np.ndarray] = {}
    for input_name in input_names:
        tensor_meta = tensor_meta_map.get(str(input_name), {})
        if not isinstance(tensor_meta, dict):
            raise ModelIRPyTorchExportError(
                f"PyTorch package metadata is missing tensor metadata for input '{input_name}'."
            )
        dtype_name = str(tensor_meta.get("dtype", "FLOAT32")).upper()
        if dtype_name not in _NUMPY_DTYPE_BY_TENSOR_DTYPE:
            raise ModelIRPyTorchExportError(
                f"Unsupported input dtype for {export_label}. input={input_name} dtype={dtype_name}"
            )
        shape_values = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
        if not isinstance(shape_values, list):
            shape_values = tensor_meta.get("shape", [])
        if not isinstance(shape_values, list):
            shape_values = []
        custom_input_value = _lookup_custom_input(str(input_name))
        shape_hint = _lookup_torchscript_shape_hint(
            input_name=str(input_name),
            shape_hints=parsed_shape_hints,
            normalized_shape_hints=normalized_shape_hints,
            normalize_name=_normalize_tensor_name,
        )
        has_dynamic_dim = any(int(v) <= 0 for v in list(shape_values))
        if has_dynamic_dim:
            dynamic_inputs_present = True
        trace_shape_values = _sanitize_torchscript_trace_shape(shape_values)
        dynamic_hint_resolved = False
        if custom_input_value is not None:
            trace_shape_values = tuple(
                int(v) for v in list(np.asarray(custom_input_value).shape)
            )
            dynamic_hint_resolved = True
        elif shape_hint is not None:
            trace_shape_values = _resolve_torchscript_trace_shape(
                input_name=str(input_name),
                shape_values=shape_values,
                shape_hint=shape_hint,
                export_label=export_label,
            )
            dynamic_hint_resolved = True
        elif (
            test_data_nhwc is not None
            and len(list(shape_values)) == 4
            and (
                int(shape_values[1]) in {3, -1, 0}
                or int(shape_values[3]) in {3, -1, 0}
            )
        ):
            trace_shape_values = _resolve_torchscript_trace_shape(
                input_name=str(input_name),
                shape_values=shape_values,
                shape_hint=[
                    int(test_data_nhwc.shape[0]),
                    int(test_data_nhwc.shape[-1]) if int(shape_values[1]) in {3, -1, 0} else int(test_data_nhwc.shape[1]),
                    int(test_data_nhwc.shape[1]) if int(shape_values[1]) in {3, -1, 0} else int(test_data_nhwc.shape[2]),
                    int(test_data_nhwc.shape[2]) if int(shape_values[1]) in {3, -1, 0} else int(test_data_nhwc.shape[-1]),
                ],
                export_label=export_label,
            )
            dynamic_hint_resolved = True
        elif _can_autoresolve_batch_only_trace_shape(shape_values):
            dynamic_hint_resolved = True
        input_specs.append(
            (
                str(input_name),
                _NUMPY_DTYPE_BY_TENSOR_DTYPE[dtype_name],
                trace_shape_values,
            )
        )
        if has_dynamic_dim and not dynamic_hint_resolved:
            missing_dynamic_hints.append(str(input_name))
            continue

        if custom_input_value is not None:
            generated_inputs_np[str(input_name)] = _extract_sample_from_custom(
                data=np.asarray(custom_input_value),
                sample_index=0,
                expected_shape=trace_shape_values,
                np_dtype=_NUMPY_DTYPE_BY_TENSOR_DTYPE[dtype_name],
            )
            continue
        if test_data_nhwc is not None and len(trace_shape_values) == 4:
            try:
                generated_inputs_np[str(input_name)] = _build_torchscript_image_input_from_nhwc(
                    data=test_data_nhwc,
                    expected_shape=trace_shape_values,
                    np_dtype=_NUMPY_DTYPE_BY_TENSOR_DTYPE[dtype_name],
                )
                continue
            except Exception as ex:
                if dynamic_hint_resolved and shape_hint is None and custom_input_value is None:
                    raise ModelIRPyTorchExportError(
                        f"{export_label} could not build an example input from test_data_nhwc_path. "
                        f"input={input_name} expected_shape={list(trace_shape_values)}"
                    ) from ex
    if len(missing_dynamic_hints) > 0:
        raise ModelIRPyTorchExportError(
            f"{export_label} requires concrete trace hints for all dynamic public inputs. "
            "Use --shape_hints as the recommended option, or provide "
            "--test_data_nhwc_path / custom_input_op_name_np_data_path when applicable. "
            f"package_dir={package_dir} missing_inputs={sorted(missing_dynamic_hints)}"
        )
    rng = np.random.default_rng(seed=0)
    example_inputs_np: Dict[str, np.ndarray] = {}
    for input_name, input_dtype, input_shape in input_specs:
        prebuilt = generated_inputs_np.get(str(input_name), None)
        if prebuilt is not None:
            example_inputs_np[str(input_name)] = np.asarray(prebuilt)
            continue
        if _is_string_dtype(np.dtype(input_dtype)):
            example_inputs_np[str(input_name)] = _generate_string_input(
                shape=input_shape,
                rng=rng,
            )
            continue
        if np.issubdtype(np.dtype(input_dtype), np.integer):
            canonical = _normalize_tensor_name(str(input_name))
            if "mask" in canonical.split("_"):
                example_inputs_np[str(input_name)] = np.ones(input_shape, dtype=input_dtype)
                continue
            if any(
                canonical.endswith(suffix)
                for suffix in ("length", "lengths", "len", "lens", "seq_len", "seq_lens")
            ):
                example_inputs_np[str(input_name)] = _fill_length_like_input(
                    input_name=str(input_name),
                    input_shape=input_shape,
                    input_dtype=np.dtype(input_dtype),
                    generated_inputs=example_inputs_np,
                )
                continue
        example_inputs_np[str(input_name)] = _generate_seeded_input(
            shape=input_shape,
            np_dtype=np.dtype(input_dtype),
            rng=rng,
        )
    converted_inputs = _convert_inputs_for_package(
        inputs=example_inputs_np,
        package_metadata=package_metadata,
    )
    example_input_shapes: Dict[str, List[int]] = {}
    ordered_inputs: List[Any] = []
    for input_name in input_names:
        input_value = converted_inputs.get(str(input_name), None)
        if input_value is None:
            raise ModelIRPyTorchExportError(
                f"{export_label} could not resolve an example input. input={input_name}"
            )
        if not hasattr(input_value, "shape"):
            raise ModelIRPyTorchExportError(
                f"{export_label} supports only tensor-like public inputs for native packages. "
                f"input={input_name} type={type(input_value).__name__}"
            )
        example_input_shapes[str(input_name)] = [int(v) for v in list(input_value.shape)]
        ordered_inputs.append(input_value)
    return tuple(ordered_inputs), example_input_shapes, bool(dynamic_inputs_present)


def _build_torchscript_example_inputs(
    *,
    package_dir: str,
    package_metadata: Dict[str, Any],
    custom_input_op_name_np_data_path: Optional[List[Any]],
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
) -> Tuple[Tuple[Any, ...], Dict[str, List[int]], bool]:
    return _build_pytorch_export_example_inputs(
        package_dir=package_dir,
        package_metadata=package_metadata,
        custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
        shape_hints=shape_hints,
        test_data_nhwc_path=test_data_nhwc_path,
        export_label="TorchScript export",
    )


def _load_generated_package_export_metadata(
    *,
    package_dir: str,
    export_label: str,
) -> Tuple[Path, Path, Dict[str, Any]]:
    package_path = Path(package_dir)
    metadata_path = package_path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"PyTorch package metadata is missing. path={metadata_path}"
        )
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    execution_backend = str(metadata.get("execution_backend", "")).strip().lower()
    if execution_backend == "" and _is_runtime_wrapper_package_dir(package_path):
        execution_backend = "runtime_wrapper"
    if execution_backend not in {"", "native"}:
        raise ModelIRPyTorchExportError(
            f"{export_label} is supported only for native PyTorch packages. "
            f"package_dir={package_dir} execution_backend={execution_backend or 'native'}"
        )

    package_init_path = package_path / "__init__.py"
    if not package_init_path.exists():
        raise FileNotFoundError(
            f"Generated PyTorch package is missing __init__.py. path={package_init_path}"
        )
    return package_path, metadata_path, metadata


def _write_generated_package_export_metadata(
    *,
    metadata_path: Path,
    metadata: Dict[str, Any],
    metadata_key: str,
    file_name: Optional[str],
    example_input_shapes: Dict[str, List[int]],
    dynamic_inputs_present: bool,
    error: Optional[str] = None,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "file_name": file_name,
        "example_input_shapes": {
            str(name): [int(v) for v in list(shape)]
            for name, shape in example_input_shapes.items()
        },
        "dynamic_inputs_present": bool(dynamic_inputs_present),
    }
    if extra_fields is not None:
        payload.update(extra_fields)
    if error is not None:
        payload["error"] = str(error)
    metadata[str(metadata_key)] = payload
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def _remove_generated_package_artifact_if_exists(artifact_path: Path) -> None:
    if not artifact_path.exists():
        return
    try:
        artifact_path.unlink()
    except Exception:
        pass


def _metadata_has_dynamic_public_inputs(metadata: Dict[str, Any]) -> bool:
    tensor_meta_map = metadata.get("tensors", {})
    if not isinstance(tensor_meta_map, dict):
        return False
    for input_name in [str(v) for v in list(metadata.get("inputs", []))]:
        tensor_meta = tensor_meta_map.get(str(input_name), {})
        if not isinstance(tensor_meta, dict):
            continue
        shape_values = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
        if not isinstance(shape_values, list):
            shape_values = tensor_meta.get("shape", [])
        if not isinstance(shape_values, list):
            continue
        if any(int(v) <= 0 for v in list(shape_values)):
            return True
    return False


def _run_generated_package_export_child(
    *,
    example_inputs: Tuple[Any, ...],
    child_script: str,
    package_path: Path,
    artifact_path: Path,
    child_payload: Dict[str, Any],
    child_args: Optional[List[str]] = None,
    temp_prefix: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "PyTorch export child execution requires `torch` to be installed."
        ) from ex

    if child_args is None:
        child_args = []
    with tempfile.TemporaryDirectory(prefix=temp_prefix) as temp_dir:
        serialized_inputs_path = os.path.join(temp_dir, "example_inputs.pt")
        payload = dict(child_payload)
        payload["inputs"] = tuple(example_inputs)
        torch.save(payload, serialized_inputs_path)
        child_result = subprocess.run(
            [
                sys.executable,
                "-c",
                child_script,
                str(package_path),
                str(serialized_inputs_path),
                str(artifact_path),
                *[str(v) for v in child_args],
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    if child_result.returncode == 0:
        try:
            return json.loads(child_result.stdout.strip() or "{}"), ""
        except json.JSONDecodeError:
            return {}, ""
    stderr_text = child_result.stderr.strip()
    stdout_text = child_result.stdout.strip()
    return None, (
        f"returncode={child_result.returncode} "
        f"stdout={stdout_text} stderr={stderr_text}"
    )


def export_torchscript_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Optional[str]:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "TorchScript export requires `torch` to be installed."
        ) from ex

    package_path, metadata_path, metadata = _load_generated_package_export_metadata(
        package_dir=package_dir,
        export_label="TorchScript export",
    )
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = _build_pytorch_export_example_inputs(
            package_dir=package_dir,
            package_metadata=metadata,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
            shape_hints=shape_hints,
            test_data_nhwc_path=test_data_nhwc_path,
            export_label="TorchScript export",
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
            extra_fields={
                "trace_mode": None,
            },
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    torchscript_file_name = f"{file_stem}_jit.pt"
    torchscript_path = package_path / torchscript_file_name
    child_script = """
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
torchscript_path = Path(sys.argv[3])
mode = str(sys.argv[4]).strip().lower()

module_name = (
    "_onnx2tf_generated_torchscript_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
with torch.no_grad():
    if mode == "trace":
        artifact = torch.jit.trace(model, example_inputs, check_trace=False)
    elif mode == "script":
        artifact = torch.jit.script(model)
    else:
        raise RuntimeError(f"Unsupported torchscript export mode: {mode}")
    torch.jit.save(artifact, str(torchscript_path))
print(json.dumps({"trace_mode": mode}))
"""
    trace_mode = ""
    last_error_message = ""
    for candidate_mode in ("trace", "script"):
        child_payload, last_error_message = _run_generated_package_export_child(
            example_inputs=example_inputs,
            child_script=child_script,
            package_path=package_path,
            artifact_path=torchscript_path,
            child_payload={},
            child_args=[candidate_mode],
            temp_prefix="onnx2tf_torchscript_",
        )
        if child_payload is not None:
            trace_mode = str(child_payload.get("trace_mode", candidate_mode))
            break
        if last_error_message != "":
            last_error_message = f"mode={candidate_mode} {last_error_message}"
    if trace_mode == "":
        _remove_generated_package_artifact_if_exists(torchscript_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message,
            extra_fields={
                "trace_mode": None,
            },
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "TorchScript export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="torchscript",
        file_name=str(torchscript_file_name),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
        extra_fields={
            "trace_mode": trace_mode,
        },
    )
    return str(torchscript_path)


def export_dynamo_onnx_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Optional[str]:
    package_path, metadata_path, metadata = _load_generated_package_export_metadata(
        package_dir=package_dir,
        export_label="Dynamo ONNX export",
    )
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = _build_pytorch_export_example_inputs(
            package_dir=package_dir,
            package_metadata=metadata,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
            shape_hints=shape_hints,
            test_data_nhwc_path=test_data_nhwc_path,
            export_label="Dynamo ONNX export",
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    dynamo_onnx_file_name = f"{file_stem}_dynamo.onnx"
    dynamo_onnx_path = package_path / dynamo_onnx_file_name
    child_script = """
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
dynamo_onnx_path = Path(sys.argv[3])

module_name = (
    "_onnx2tf_generated_dynamo_onnx_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
input_names = [str(v) for v in list(payload.get("input_names", []))]
output_names = [str(v) for v in list(payload.get("output_names", []))]
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
with torch.no_grad():
    torch.onnx.export(
        model,
        example_inputs,
        str(dynamo_onnx_path),
        dynamo=True,
        input_names=input_names,
        output_names=output_names,
    )
print(json.dumps({"file_name": dynamo_onnx_path.name}))
"""
    child_payload, last_error_message = _run_generated_package_export_child(
        example_inputs=example_inputs,
        child_script=child_script,
        package_path=package_path,
        artifact_path=dynamo_onnx_path,
        child_payload={
            "input_names": [str(v) for v in list(metadata.get("inputs", []))],
            "output_names": [str(v) for v in list(metadata.get("outputs", []))],
        },
        temp_prefix="onnx2tf_dynamo_onnx_",
    )
    if child_payload is None or not dynamo_onnx_path.exists():
        _remove_generated_package_artifact_if_exists(dynamo_onnx_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message or "dynamo=True ONNX export did not produce an artifact.",
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "Dynamo ONNX export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="dynamo_onnx",
        file_name=str(child_payload.get("file_name", dynamo_onnx_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    return str(dynamo_onnx_path)


def export_exported_program_from_generated_package(
    *,
    package_dir: str,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    shape_hints: Optional[List[str]] = None,
    test_data_nhwc_path: Optional[str] = None,
    raise_on_failure: bool = True,
) -> Optional[str]:
    package_path, metadata_path, metadata = _load_generated_package_export_metadata(
        package_dir=package_dir,
        export_label="ExportedProgram export",
    )
    try:
        example_inputs, example_input_shapes, dynamic_inputs_present = _build_pytorch_export_example_inputs(
            package_dir=package_dir,
            package_metadata=metadata,
            custom_input_op_name_np_data_path=custom_input_op_name_np_data_path,
            shape_hints=shape_hints,
            test_data_nhwc_path=test_data_nhwc_path,
            export_label="ExportedProgram export",
        )
    except Exception as ex:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        if raise_on_failure:
            raise
        return None
    file_stem = _sanitize_torchscript_file_stem(
        str(metadata.get("name", "")),
        fallback=package_path.name,
    )
    exported_program_file_name = f"{file_stem}_ep.pt2"
    exported_program_path = package_path / exported_program_file_name
    child_script = """
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch

package_path = Path(sys.argv[1])
package_init_path = package_path / "__init__.py"
inputs_path = Path(sys.argv[2])
exported_program_path = Path(sys.argv[3])

module_name = (
    "_onnx2tf_generated_exported_program_child_"
    + hashlib.sha256(str(package_path.resolve()).encode("utf-8")).hexdigest()
)
if module_name in sys.modules:
    del sys.modules[module_name]
spec = importlib.util.spec_from_file_location(
    module_name,
    str(package_init_path),
    submodule_search_locations=[str(package_path)],
)
if spec is None or spec.loader is None:
    raise ImportError(
        f"Failed to create an import spec for the generated PyTorch package. path={package_init_path}"
    )
module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = module
spec.loader.exec_module(module)
if not hasattr(module, "load_model"):
    raise RuntimeError(
        "Generated native PyTorch package does not expose load_model(). "
        f"package_dir={package_path}"
    )
payload = torch.load(str(inputs_path), map_location="cpu")
example_inputs = tuple(payload["inputs"])
model = module.load_model(device="cpu", eval_mode=True)
if hasattr(model, "cpu"):
    model = model.cpu()
with torch.no_grad():
    exported = torch.export.export(model, example_inputs)
torch.export.save(exported, str(exported_program_path))
print(json.dumps({"file_name": exported_program_path.name}))
"""
    child_payload, last_error_message = _run_generated_package_export_child(
        example_inputs=example_inputs,
        child_script=child_script,
        package_path=package_path,
        artifact_path=exported_program_path,
        child_payload={},
        temp_prefix="onnx2tf_exported_program_",
    )
    if child_payload is None or not exported_program_path.exists():
        _remove_generated_package_artifact_if_exists(exported_program_path)
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes=example_input_shapes,
            dynamic_inputs_present=dynamic_inputs_present,
            error=last_error_message or "torch.export.save did not produce an artifact.",
        )
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "ExportedProgram export failed for the generated native PyTorch package. "
                f"package_dir={package_dir} details={last_error_message}"
            )
        return None
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="exported_program",
        file_name=str(child_payload.get("file_name", exported_program_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    return str(exported_program_path)


def _build_tflite_backed_metadata_payload(
    *,
    model_ir: ModelIR,
    tflite_file_name: str,
) -> Dict[str, Any]:
    inferred = copy.deepcopy(model_ir)
    infer_model_ir_logical_layouts(inferred)
    boundary_shape_map = inferred.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}

    public_tensor_names = {
        str(name) for name in list(inferred.inputs) + list(inferred.outputs)
    }
    tensors: Dict[str, Dict[str, Any]] = {}
    for tensor_name in sorted(public_tensor_names):
        tensor = inferred.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        tensor_meta = _serializable_tensor_meta(tensor)
        boundary_shape = boundary_shape_map.get(str(tensor_name), None)
        if isinstance(boundary_shape, list) and len(boundary_shape) == len(tensor_meta["shape"]):
            tensor_meta["shape"] = [max(1, int(v)) if int(v) >= 0 else 1 for v in boundary_shape]
            tensor_meta["shape_signature"] = [int(v) for v in boundary_shape]
        tensor_meta["has_data"] = False
        tensors[str(tensor_name)] = tensor_meta

    current_public_layouts = {
        str(name): normalize_logical_layout(inferred.tensors[str(name)].logical_layout)
        for name in public_tensor_names
        if str(name) in inferred.tensors
    }
    return {
        "schema_version": 1,
        "execution_backend": "tflite",
        "name": str(inferred.name),
        "description": str(inferred.description),
        "inputs": [str(v) for v in inferred.inputs],
        "outputs": [str(v) for v in inferred.outputs],
        "tensors": tensors,
        "operators": [],
        "public_layouts": _serializable_value(
            dict(inferred.metadata.get("onnx_public_layout_map", {}))
        ),
        "current_public_layouts": _serializable_value(current_public_layouts),
        "boundary_shape_signatures": _serializable_value(boundary_shape_map),
        "tflite_file_name": str(tflite_file_name),
    }


def _build_saved_model_backed_metadata_payload(
    *,
    model_ir: ModelIR,
    saved_model_dir_name: str,
) -> Dict[str, Any]:
    metadata = _build_tflite_backed_metadata_payload(
        model_ir=model_ir,
        tflite_file_name="",
    )
    metadata["execution_backend"] = "saved_model"
    metadata["saved_model_dir_name"] = str(saved_model_dir_name)
    metadata.pop("tflite_file_name", None)
    return metadata


def _extract_string_normalizer_config_from_onnx_graph(
    onnx_graph: Any,
) -> Optional[Dict[str, Any]]:
    if onnx_graph is None:
        return None
    graph = getattr(onnx_graph, "graph", None)
    if graph is None or len(list(graph.node)) != 1:
        return None
    node = list(graph.node)[0]
    if str(node.op_type) != "StringNormalizer":
        return None
    if len(list(graph.input)) == 0 or len(list(graph.output)) == 0:
        return None
    attributes = {
        str(attr.name): onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }
    stopwords = []
    for value in list(attributes.get("stopwords", [])):
        if isinstance(value, bytes):
            stopwords.append(value.decode("utf-8"))
        else:
            stopwords.append(str(value))
    case_change_action = attributes.get("case_change_action", b"")
    locale = attributes.get("locale", b"")
    return {
        "input_name": str(node.input[0]),
        "output_name": str(node.output[0]),
        "case_change_action": (
            case_change_action.decode("utf-8")
            if isinstance(case_change_action, bytes)
            else str(case_change_action)
        ),
        "is_case_sensitive": bool(int(attributes.get("is_case_sensitive", 1))),
        "locale": locale.decode("utf-8") if isinstance(locale, bytes) else str(locale),
        "stopwords": stopwords,
    }


def export_pytorch_package_from_string_normalizer_onnx(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    onnx_graph: Any,
) -> str:
    config = _extract_string_normalizer_config_from_onnx_graph(onnx_graph)
    if config is None:
        raise ModelIRPyTorchExportError(
            "StringNormalizer fallback requires a single-op StringNormalizer ONNX graph."
        )
    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)
    metadata = {
        "schema_version": 1,
        "execution_backend": "string_normalizer",
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": {
            str(name): _serializable_tensor_meta(model_ir.tensors[str(name)])
            for name in list(model_ir.inputs) + list(model_ir.outputs)
            if str(name) in model_ir.tensors
        },
        "operators": [],
        "public_layouts": {},
        "string_normalizer": config,
    }
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)


def _make_tensor_storage_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    storage_name_map: Dict[str, str] = {}
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        if not isinstance(tensor.data, np.ndarray):
            continue
        base_name = re.sub(r"[^0-9A-Za-z_]", "_", str(tensor_name)).strip("_")
        if base_name == "":
            base_name = "tensor"
        if base_name[0].isdigit():
            base_name = f"tensor_{base_name}"
        candidate = base_name
        suffix = 1
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_names.add(candidate)
        storage_name_map[str(tensor_name)] = candidate
    return storage_name_map


_DIRECT_CODEGEN_MODULE_OP_TYPES: Set[str] = {
    "CONV_2D",
    "DEPTHWISE_CONV_2D",
    "TRANSPOSE_CONV",
    "CONV_3D",
    "CONV_3D_TRANSPOSE",
    "FULLY_CONNECTED",
    "PRELU",
}

_DIRECT_CODEGEN_UNARY_EXPRESSIONS: Dict[str, str] = {
    "ABS": "torch.abs({x})",
    "ACOS": "torch.acos({x})",
    "ASIN": "torch.asin({x})",
    "ATAN": "torch.atan({x})",
    "CEIL": "torch.ceil({x})",
    "COS": "torch.cos({x})",
    "ELU": "F.elu({x})",
    "EXP": "torch.exp({x})",
    "FLOOR": "torch.floor({x})",
    "GELU": "F.gelu({x})",
    "HARD_SWISH": "F.hardswish({x})",
    "IDENTITY": "{x}",
    "LEAKY_RELU": "F.leaky_relu({x}, negative_slope={alpha})",
    "LOG": "torch.log({x})",
    "LOGICAL_NOT": "torch.logical_not({x})",
    "LOGISTIC": "torch.sigmoid({x})",
    "NEG": "torch.neg({x})",
    "RELU": "torch.relu({x})",
    "RELU_0_TO_1": "torch.clamp({x}, min=0.0, max=1.0)",
    "RELU_N1_TO_1": "torch.clamp({x}, min=-1.0, max=1.0)",
    "RELU6": "torch.clamp({x}, min=0.0, max=6.0)",
    "ROUND": "torch.round({x})",
    "RSQRT": "torch.rsqrt({x})",
    "SIGMOID": "torch.sigmoid({x})",
    "SIGN": "torch.sign({x})",
    "SIN": "torch.sin({x})",
    "SQRT": "torch.sqrt({x})",
    "SQUARE": "torch.square({x})",
    "TAN": "torch.tan({x})",
    "TANH": "torch.tanh({x})",
}

_DIRECT_CODEGEN_BINARY_FUNCTIONS: Dict[str, str] = {
    "ADD": "torch.add",
    "DIV": "torch.div",
    "EQUAL": "torch.eq",
    "FLOOR_MOD": "torch.remainder",
    "GREATER": "torch.gt",
    "LESS": "torch.lt",
    "LOGICAL_AND": "torch.logical_and",
    "LOGICAL_OR": "torch.logical_or",
    "MAXIMUM": "torch.maximum",
    "MINIMUM": "torch.minimum",
    "MUL": "torch.mul",
    "POW": "torch.pow",
    "SUB": "torch.sub",
}

_DIRECT_CODEGEN_SUPPORTED_OP_TYPES: Set[str] = (
    set(_DIRECT_CODEGEN_MODULE_OP_TYPES)
    | set(_DIRECT_CODEGEN_UNARY_EXPRESSIONS.keys())
    | set(_DIRECT_CODEGEN_BINARY_FUNCTIONS.keys())
    | {
        "ARG_MAX",
        "ARG_MIN",
        "AVERAGE_POOL_2D",
        "BATCH_MATMUL",
        "CAST",
        "CONCATENATION",
        "DEPTH_TO_SPACE",
        "EXPAND_DIMS",
        "FILL",
        "GATHER",
        "GATHER_ND",
        "MAX_POOL_2D",
        "LOCAL_RESPONSE_NORMALIZATION",
        "MEAN",
        "MIRROR_PAD",
        "NON_MAX_SUPPRESSION_V4",
        "PACK",
        "PAD",
        "PADV2",
        "RANGE",
        "REDUCE_ANY",
        "REDUCE_MAX",
        "REDUCE_MIN",
        "REDUCE_PROD",
        "RESHAPE",
        "RESIZE_BILINEAR",
        "RESIZE_NEAREST_NEIGHBOR",
        "SCATTER_ND",
        "SHAPE",
        "SLICE",
        "SOFTMAX",
        "SPACE_TO_DEPTH",
        "SPLIT",
        "SQUEEZE",
        "STRIDED_SLICE",
        "SUM",
        "TOPK_V2",
        "TILE",
        "TRANSPOSE",
        "UNPACK",
        "SELECT",
        "SELECT_V2",
        "WHERE",
    }
)


def _sanitize_python_identifier(name: str, *, prefix: str) -> str:
    identifier = re.sub(r"[^0-9A-Za-z_]", "_", str(name)).strip("_")
    if identifier == "":
        identifier = str(prefix)
    if identifier[0].isdigit():
        identifier = f"{prefix}_{identifier}"
    if keyword.iskeyword(identifier):
        identifier = f"{identifier}_{prefix}"
    return identifier


def _make_unique_identifier(base_name: str, used_names: Set[str]) -> str:
    candidate = str(base_name)
    suffix = 1
    while candidate in used_names:
        candidate = f"{base_name}_{suffix}"
        suffix += 1
    used_names.add(candidate)
    return candidate


def _import_generated_package_from_output(package_path: str) -> Any:
    package_dir = Path(package_path)
    package_name = re.sub(r"[^0-9A-Za-z_]", "_", package_dir.name).strip("_")
    if package_name == "":
        package_name = "generated_pytorch_package"
    module_name = f"_onnx2tf_generated_{package_name}"
    stale_module_names = [
        existing_name
        for existing_name in list(sys.modules.keys())
        if existing_name == module_name or existing_name.startswith(f"{module_name}.")
    ]
    for existing_name in stale_module_names:
        sys.modules.pop(existing_name, None)
    spec = importlib.util.spec_from_file_location(
        module_name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ModelIRPyTorchExportError(
            f"Could not import generated PyTorch package for state_dict export. path={package_path}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _prepare_exported_state_tensor(source: np.ndarray, target: Any) -> Any:
    import torch

    source_tensor = torch.as_tensor(np.asarray(source))
    target_tensor = torch.as_tensor(target)
    candidate = source_tensor.to(dtype=target_tensor.dtype)
    if list(candidate.shape) == list(target_tensor.shape):
        return candidate.detach().cpu().clone()
    if int(candidate.numel()) == int(target_tensor.numel()):
        reshaped = candidate.reshape(target_tensor.shape)
        if list(reshaped.shape) == list(target_tensor.shape):
            return reshaped.detach().cpu().clone()
    perm = _perm_cl_to_cf(candidate.ndim)
    if perm is not None:
        permuted = candidate.permute(*perm).contiguous()
        if list(permuted.shape) == list(target_tensor.shape):
            return permuted.detach().cpu().clone()
    if candidate.ndim <= 5:
        import itertools

        for generic_perm in itertools.permutations(range(candidate.ndim)):
            if list(generic_perm) == list(range(candidate.ndim)):
                continue
            permuted = candidate.permute(*generic_perm).contiguous()
            if list(permuted.shape) == list(target_tensor.shape):
                return permuted.detach().cpu().clone()
    raise ModelIRPyTorchExportError(
        "Native PyTorch state_dict export could not align a tensor to the generated module shape. "
        f"source_shape={list(candidate.shape)} target_shape={list(target_tensor.shape)}"
    )


def _build_native_generated_state_dict(
    *,
    package_path: str,
    model_ir: ModelIR,
    load_specs: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    package_module = _import_generated_package_from_output(package_path)
    model = package_module.Model(load_weights=False)
    exported_state_dict = {
        str(key): value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    expected_keys = {str(key) for key in exported_state_dict.keys()}
    mapped_keys = {str(attr_path) for attr_path, _ in load_specs}
    if expected_keys != mapped_keys:
        raise ModelIRPyTorchExportError(
            "Native PyTorch state_dict export could not reconcile generated state_dict keys. "
            f"expected_keys={sorted(expected_keys)} mapped_keys={sorted(mapped_keys)}"
        )
    for attr_path, tensor_name in load_specs:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None or not isinstance(tensor.data, np.ndarray):
            raise ModelIRPyTorchExportError(
                "Native PyTorch state_dict export requires concrete tensor data for every generated state entry. "
                f"tensor={tensor_name}"
            )
        exported_state_dict[str(attr_path)] = _prepare_exported_state_tensor(
            np.asarray(tensor.data),
            exported_state_dict[str(attr_path)],
        )
    return exported_state_dict


def _build_tensor_var_name_map(model_ir: ModelIR) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}

    def _canonical_tensor_var_source_name(tensor_name: str) -> str:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        base_name = str(tensor_name)
        if tensor is not None:
            layout = normalize_logical_layout(tensor.logical_layout)
            if not is_channel_last_logical_layout(layout):
                base_name = re.sub(
                    r"_(?:nhwc|nwc|ndhwc)$",
                    "",
                    base_name,
                    flags=re.IGNORECASE,
                )
        return base_name

    for tensor_name in list(model_ir.inputs) + [str(out) for op in model_ir.operators for out in op.outputs]:
        if str(tensor_name) in mapping:
            continue
        base = _sanitize_python_identifier(
            _canonical_tensor_var_source_name(str(tensor_name)),
            prefix="t",
        )
        mapping[str(tensor_name)] = _make_unique_identifier(base, used_names)
    return mapping


def _build_buffer_attr_name_map(
    *,
    model_ir: ModelIR,
    tensor_storage_name_map: Dict[str, str],
    excluded_tensor_names: Set[str],
) -> Dict[str, str]:
    used_names: Set[str] = set()
    mapping: Dict[str, str] = {}
    for tensor_name, tensor in sorted(model_ir.tensors.items()):
        if str(tensor_name) in excluded_tensor_names or not isinstance(tensor.data, np.ndarray):
            continue
        storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
        base = _sanitize_python_identifier(f"const_{storage_name}", prefix="const")
        mapping[str(tensor_name)] = _make_unique_identifier(base, used_names)
    return mapping


def _build_model_ir_producer_consumer_index(
    model_ir: ModelIR,
) -> Tuple[Dict[str, int], Dict[str, List[int]]]:
    producers: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_index, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_index)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_index))
    return producers, consumers


def _is_small_inline_constant_tensor(tensor: TensorIR) -> bool:
    if not isinstance(tensor.data, np.ndarray):
        return False
    arr = np.asarray(tensor.data)
    if arr.size > 32:
        return False
    if arr.ndim > 2:
        return False
    return str(tensor.dtype).upper() in {
        "BOOL",
        "INT8",
        "INT16",
        "INT32",
        "INT64",
        "UINT8",
        "FLOAT16",
        "FLOAT32",
        "FLOAT64",
    }


def _python_literal_for_constant_tensor(tensor: TensorIR) -> Optional[str]:
    if not _is_small_inline_constant_tensor(tensor):
        return None
    arr = np.asarray(tensor.data)
    if arr.ndim == 0:
        value = arr.reshape(-1)[0].item()
        if isinstance(value, np.generic):
            value = value.item()
        return repr(value)
    return repr(arr.tolist())


def _torch_pad_literal_for_constant_tensor(tensor: Optional[TensorIR]) -> Optional[str]:
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    pads = np.asarray(tensor.data).astype(np.int64).reshape(-1, 2).tolist()
    torch_pad: List[int] = []
    for before, after in reversed(pads):
        torch_pad.extend([int(before), int(after)])
    while len(torch_pad) >= 2 and int(torch_pad[-2]) == 0 and int(torch_pad[-1]) == 0:
        torch_pad = torch_pad[:-2]
    return repr(torch_pad)


def _scalar_literal_for_constant_tensor(tensor: Optional[TensorIR]) -> Optional[str]:
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    flat = np.asarray(tensor.data).reshape(-1)
    if int(flat.size) != 1:
        return None
    value = flat[0].item()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        value = float(value)
        if math.isnan(value):
            return "float('nan')"
        if math.isinf(value):
            return "float('inf')" if value > 0.0 else "float('-inf')"
        return repr(value)
    return repr(value)


def _constant_int_list(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    arr = np.asarray(tensor.data)
    if arr.size == 0:
        return []
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    return [int(v) for v in arr.reshape(-1).tolist()]


def _preferred_reshape_target_values(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None:
        return None
    if tensor.shape_signature is not None:
        signature = [int(v) for v in list(tensor.shape_signature)]
        if len(signature) == len(list(tensor.shape)) and any(int(v) <= 0 for v in signature):
            return signature
    return [int(v) for v in list(tensor.shape)]


def _torch_dtype_literal(dtype_name: str) -> str:
    mapping = {
        "BOOL": "torch.bool",
        "INT8": "torch.int8",
        "INT16": "torch.int16",
        "INT32": "torch.int32",
        "INT64": "torch.int64",
        "UINT8": "torch.uint8",
        "FLOAT16": "torch.float16",
        "FLOAT32": "torch.float32",
        "FLOAT64": "torch.float64",
    }
    key = str(dtype_name).upper()
    if key not in mapping:
        raise ModelIRPyTorchExportError(
            f"Unsupported dtype for native PyTorch-like model.py codegen: {dtype_name}"
        )
    return str(mapping[key])


def _conv_block_activation_config(op: OperatorIR) -> Tuple[str, Optional[float]]:
    op_type = str(op.op_type)
    if op_type == "LEAKY_RELU":
        return ("leaky_relu", float(op.options.get("alpha", 0.2)))
    if op_type == "RELU":
        return ("relu", None)
    if op_type == "RELU6":
        return ("relu6", None)
    if op_type == "RELU_N1_TO_1":
        return ("relu_n1_to_1", None)
    if op_type == "RELU_0_TO_1":
        return ("relu_0_to_1", None)
    if op_type == "TANH":
        return ("tanh", None)
    if op_type == "LOGISTIC":
        return ("sigmoid", None)
    return ("none", None)


def _conv_block_activation_config_from_fused_name(
    fused_name: str,
    *,
    alpha: Optional[float] = None,
) -> Tuple[str, Optional[float]]:
    key = str(fused_name).upper()
    if key == "LEAKY_RELU":
        return ("leaky_relu", float(0.2 if alpha is None else alpha))
    if key == "RELU":
        return ("relu", None)
    if key == "RELU6":
        return ("relu6", None)
    if key == "RELU_N1_TO_1":
        return ("relu_n1_to_1", None)
    if key == "RELU_0_TO_1":
        return ("relu_0_to_1", None)
    if key == "TANH":
        return ("tanh", None)
    if key == "LOGISTIC":
        return ("sigmoid", None)
    return ("none", None)

def _reshape_special_layout_plan(
    *,
    input_shape: Optional[Sequence[int]],
    output_shape: Optional[Sequence[int]],
    input_layout: Optional[str],
    output_layout: Optional[str],
) -> Optional[Dict[str, Any]]:
    if input_shape is None or output_shape is None:
        return None
    src = [int(v) for v in list(input_shape)]
    dst = [int(v) for v in list(output_shape)]
    in_layout = str(input_layout or "").upper()
    out_layout = str(output_layout or "").upper()
    if (
        in_layout == "NCHW"
        and len(src) == 4
        and len(dst) == 3
        and int(src[0]) == 1
        and int(src[2]) == int(dst[0])
        and int(src[3]) == int(dst[1])
        and int(src[1]) == int(dst[2])
    ):
        return {
            "pre_perm": [0, 2, 3, 1],
            "reshape_shape": list(dst),
            "post_perm": None,
        }
    if (
        len(src) == 3
        and len(dst) == 4
        and int(dst[0]) == 1
        and int(src[0]) == int(dst[2])
        and int(src[1]) == int(dst[3])
        and int(src[2]) == int(dst[1])
        and out_layout == "NCHW"
    ):
        return {
            "pre_perm": None,
            "reshape_shape": [1, int(src[0]), int(src[1]), int(src[2])],
            "post_perm": [0, 3, 1, 2],
        }
    if (
        len(src) == 4
        and len(dst) == 4
        and int(src[0]) == int(dst[0])
        and any(int(v) == 1 for v in src[1:])
    ):
        if [int(src[0]), int(src[3]), int(src[1]), int(src[2])] == dst:
            return {
                "pre_perm": [0, 3, 1, 2],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
        if [int(src[0]), int(src[2]), int(src[3]), int(src[1])] == dst:
            return {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
    if (
        in_layout == "NCHW"
        and out_layout == "NCDHW"
        and len(src) == 4
        and len(dst) == 5
        and int(src[0]) == int(dst[0])
        and int(src[1]) == 1
        and int(dst[2]) == 1
        and int(dst[3]) == 1
        and int(src[2]) == int(dst[4])
        and int(src[3]) == int(dst[1])
    ):
        return {
            "pre_perm": [0, 3, 1, 2],
            "reshape_shape": list(dst),
            "post_perm": None,
        }
    if (
        in_layout == "NCHW"
        and len(src) == 4
        and len(dst) >= 5
        and int(src[0]) == int(dst[0])
        and int(src[2]) == int(dst[1])
        and int(src[3]) == int(dst[2])
    ):
        trailing_product = 1
        for dim in dst[3:]:
            trailing_product *= int(dim)
        if int(src[1]) == int(trailing_product):
            return {
                "pre_perm": [0, 2, 3, 1],
                "reshape_shape": list(dst),
                "post_perm": None,
            }
    return None


def _direct_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    size_values: Sequence[int],
    input_rank: int,
    input_shape: Optional[Sequence[int]] = None,
) -> Optional[str]:
    if len(begin_values) != int(input_rank) or len(size_values) != int(input_rank):
        return None
    parts: List[str] = []
    for axis, (start, length) in enumerate(zip(begin_values, size_values)):
        dim_size: Optional[int] = None
        if input_shape is not None and axis < len(input_shape):
            try:
                dim_size = int(input_shape[axis])
            except Exception:
                dim_size = None
        resolved_start = int(start)
        if int(length) < 0:
            resolved_stop: Optional[int] = None
        else:
            resolved_stop = resolved_start + int(length)
            if dim_size is not None and int(dim_size) > 0:
                resolved_stop = min(int(resolved_stop), int(dim_size))
        if resolved_start == 0 and resolved_stop is None:
            parts.append(":")
        else:
            start_str = "" if resolved_start == 0 else str(resolved_start)
            stop_str = "" if resolved_stop is None else str(int(resolved_stop))
            parts.append(f"{start_str}:{stop_str}")
    return f"{x_expr}[{', '.join(parts)}]"


def _direct_strided_slice_expr(
    *,
    x_expr: str,
    begin_values: Sequence[int],
    end_values: Sequence[int],
    stride_values: Sequence[int],
    begin_mask: int,
    end_mask: int,
    input_rank: int,
) -> Optional[str]:
    if (
        len(begin_values) != int(input_rank)
        or len(end_values) != int(input_rank)
        or len(stride_values) != int(input_rank)
    ):
        return None
    parts: List[str] = []
    for axis, (start, stop, step) in enumerate(zip(begin_values, end_values, stride_values)):
        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)
        resolved_stop = None if ((int(end_mask) >> axis) & 1) else int(stop)
        if resolved_stop is not None and int(resolved_stop) >= 2147483647:
            resolved_stop = None
        resolved_step = int(step)
        if resolved_step == 0:
            return None
        if resolved_start is None and resolved_stop is None and resolved_step == 1:
            parts.append(":")
            continue
        start_str = "" if resolved_start is None else str(int(resolved_start))
        stop_str = "" if resolved_stop is None else str(int(resolved_stop))
        if resolved_step == 1:
            parts.append(f"{start_str}:{stop_str}")
        else:
            parts.append(f"{start_str}:{stop_str}:{resolved_step}")
    return f"{x_expr}[{', '.join(parts)}]"


def _direct_gather_expr(
    *,
    params_expr: str,
    indices_values: Sequence[int],
    indices_shape: Optional[Sequence[int]],
    axis: int,
    batch_dims: int,
    input_rank: int,
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    if len(indices_values) == 0:
        return None
    normalized_indices_shape = (
        [int(v) for v in list(indices_shape)]
        if indices_shape is not None and len(list(indices_shape)) > 0
        else [int(len(indices_values))]
    )
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    if indices_shape is not None and len(list(indices_shape)) == 0:
        literal = int(indices_values[0])
        return (
            f"torch.index_select({params_expr}, {resolved_axis}, "
            f"torch.as_tensor([{literal}], dtype=torch.int64, device={params_expr}.device))"
            f".squeeze({resolved_axis})"
        )
    if len(normalized_indices_shape) > 1:
        literal = repr([int(v) for v in indices_values])
        return (
            f"torch.index_select({params_expr}, {resolved_axis}, "
            f"torch.as_tensor({literal}, dtype=torch.int64, device={params_expr}.device))"
            f".reshape(*{params_expr}.shape[:{resolved_axis}], "
            f"*{repr(tuple(normalized_indices_shape))}, "
            f"*{params_expr}.shape[{resolved_axis + 1}:])"
        )
    parts = [":" for _ in range(int(input_rank))]
    parts[resolved_axis] = repr([int(v) for v in indices_values])
    return f"{params_expr}[{', '.join(parts)}]"


def _direct_dynamic_gather_expr(
    *,
    params_expr: str,
    indices_expr: str,
    axis: int,
    batch_dims: int,
    input_rank: int,
    indices_name: str,
) -> Optional[str]:
    if int(batch_dims) != 0:
        return None
    if str(indices_name).endswith("_crd_to_dcr_indices"):
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis < 0 or resolved_axis >= int(input_rank):
        return None
    flat_indices_expr = f"{indices_expr}.to(dtype=torch.int64).reshape(-1)"
    reshaped_expr = (
        f"torch.index_select({params_expr}, {resolved_axis}, {flat_indices_expr})"
        f".reshape(*{params_expr}.shape[:{resolved_axis}], *{indices_expr}.shape, *{params_expr}.shape[{resolved_axis + 1}:])"
    )
    return reshaped_expr


def _should_elide_crd_to_dcr_gather_for_depth_to_space(
    *,
    model_ir: ModelIR,
    params_name: str,
    indices_name: str,
    output_name: str,
    axis: int,
    batch_dims: int,
) -> bool:
    if int(batch_dims) != 0 or not str(indices_name).endswith("_crd_to_dcr_indices"):
        return False
    input_tensor = model_ir.tensors.get(str(params_name), None)
    output_tensor = model_ir.tensors.get(str(output_name), None)
    if input_tensor is None or output_tensor is None:
        return False
    input_rank = len(list(input_tensor.shape))
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += int(input_rank)
    if resolved_axis != 1:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if not (
        is_channel_first_logical_layout(input_layout)
        or is_channel_first_logical_layout(output_layout)
    ):
        return False
    consumer_op_types = [
        str(consumer.op_type)
        for consumer in model_ir.operators
        if str(output_name) in {str(v) for v in consumer.inputs}
    ]
    return len(consumer_op_types) > 0 and all(
        op_type == "DEPTH_TO_SPACE" for op_type in consumer_op_types
    )


def _direct_codegen_module_attr_name(op_index: int, op_type: str) -> str:
    base = _sanitize_python_identifier(f"op_{op_index}_{str(op_type).lower()}", prefix="op")
    return base


def _ensure_direct_codegen_supported(model_ir: ModelIR) -> None:
    unsupported = sorted(
        {str(op.op_type) for op in model_ir.operators if str(op.op_type) not in _DIRECT_CODEGEN_SUPPORTED_OP_TYPES}
    )
    if len(unsupported) > 0:
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen does not support some op types in this model. "
            f"unsupported_op_types={unsupported}"
        )


def _is_direct_codegen_unsupported_error(ex: BaseException) -> bool:
    return "Native PyTorch-like model.py codegen does not support some op types in this model." in str(ex)


def _write_generated_package_common_files(
    output_folder_path: str,
    *,
    runtime_source: Optional[str] = None,
) -> None:
    package_dir = Path(output_folder_path)
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        "from .model import Model, load_model\n",
        encoding="utf-8",
    )
    if runtime_source is None:
        runtime_source = (
            "from onnx2tf.tflite_builder.pytorch_package_runtime import load_generated_model_package\n"
        )
    (package_dir / "runtime.py").write_text(
        runtime_source,
        encoding="utf-8",
    )


def _write_wrapper_model_file(output_folder_path: str) -> None:
    package_dir = Path(output_folder_path)
    (package_dir / "model.py").write_text(
        "from __future__ import annotations\n\n"
        "from typing import Any, Callable, cast\n\n"
        "from pathlib import Path\n\n"
        "import torch\n\n"
        "from .runtime import load_generated_model_package\n\n"
        "PACKAGE_DIR = Path(__file__).resolve().parent\n\n"
        "class Model(torch.nn.Module):\n"
        "    def __init__(self, device: str | None = None, eval_mode: bool = True):\n"
        "        super().__init__()\n"
        "        self._model: Any = load_generated_model_package(\n"
        "            package_dir=str(PACKAGE_DIR),\n"
        "            device=device,\n"
        "            eval_mode=eval_mode,\n"
        "        )\n\n"
        "    def forward(self, *args: Any, **kwargs: Any) -> Any:\n"
        "        return self._model(*args, **kwargs)\n\n"
        "    def forward_named(self, *args: Any, **kwargs: Any) -> Any:\n"
        "        forward_named = getattr(self._model, 'forward_named', None)\n"
        "        if callable(forward_named):\n"
        "            return cast(Callable[..., Any], forward_named)(*args, **kwargs)\n"
        "        return self.forward(*args, **kwargs)\n\n"
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:\n"
        "    return Model(device=device, eval_mode=eval_mode)\n",
        encoding="utf-8",
    )


_RUNTIME_SUPPORTED_CUSTOM_CODES: Set[str] = {
    "ONNX_SLICE",
}


def _build_native_runtime_source(helper_source: str) -> str:
    runtime_source = (
        "from pathlib import Path\n"
        "import re\n"
        "from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple\n\n"
        "import numpy as np\n"
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        f"{helper_source}"
        "def _resolve_model_attribute(model: torch.nn.Module, attr_path: str) -> Any:\n"
        "    value: Any = model\n"
        "    for part in str(attr_path).split('.'):\n"
        "        value = getattr(value, part)\n"
        "    return value\n\n"
        "def resolve_model_tensor(model: torch.nn.Module, attr_name: str) -> torch.Tensor:\n"
        "    value = _resolve_model_attribute(model, attr_name)\n"
        "    if not isinstance(value, torch.Tensor):\n"
        "        raise RuntimeError(f'Generated model attribute is not a tensor: {attr_name}')\n"
        "    return value\n\n"
        "def load_generated_weights(\n"
        "    *,\n"
        "    model: torch.nn.Module,\n"
        "    package_dir: Path,\n"
        "    device: Optional[str],\n"
        ") -> None:\n"
        "    raw_state_dict = torch.load(package_dir / 'state_dict.pth', map_location=device or 'cpu')\n"
        "    model.load_state_dict(raw_state_dict, strict=True)\n"
        "    if device is not None:\n"
        "        model.to(device)\n"
    )
    runtime_source = runtime_source.replace("Optional[Sequence[int]]", "Optional[List[int]]")
    runtime_source = runtime_source.replace("Sequence[int]", "List[int]")
    runtime_source = runtime_source.replace("Sequence[str]", "List[str]")
    runtime_source = runtime_source.replace("Sequence[torch.Tensor]", "List[torch.Tensor]")
    runtime_source = runtime_source.replace("Sequence[Any]", "List[Any]")
    return runtime_source


def _direct_codegen_module_attr_base(op_type: str) -> str:
    names = {
        "CONV_2D": "conv2d",
        "DEPTHWISE_CONV_2D": "depthwise_conv2d",
        "TRANSPOSE_CONV": "conv_transpose2d",
        "CONV_3D": "conv3d",
        "CONV_3D_TRANSPOSE": "conv_transpose3d",
        "FULLY_CONNECTED": "linear",
        "PRELU": "prelu",
    }
    return str(names.get(str(op_type), str(op_type).lower()))


def _supports_runtime_wrapper_model_ir(model_ir: ModelIR) -> bool:
    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type in SUPPORTED_TORCH_KERNEL_OP_TYPES:
            continue
        if op_type == "CUSTOM":
            custom_code = str(op.options.get("customCode", "")).upper()
            if custom_code in _RUNTIME_SUPPORTED_CUSTOM_CODES:
                continue
        return False
    return True


def _export_runtime_wrapper_package_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
) -> str:
    if not _supports_runtime_wrapper_model_ir(model_ir):
        raise ModelIRPyTorchExportError(
            "PyTorch runtime wrapper export does not support some op types in this model."
        )
    os.makedirs(output_folder_path, exist_ok=True)
    tensor_storage_name_map = _make_tensor_storage_name_map(model_ir)
    metadata = _build_metadata_payload(model_ir)
    metadata["execution_backend"] = "runtime_wrapper"
    metadata["tensor_storage_names"] = dict(tensor_storage_name_map)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    import torch
    state_dict: Dict[str, Any] = {}
    for tensor_name, tensor in model_ir.tensors.items():
        if not isinstance(tensor.data, np.ndarray):
            continue
        storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
        state_dict[storage_name] = torch.as_tensor(np.asarray(tensor.data))
    torch.save(state_dict, os.path.join(output_folder_path, "state_dict.pth"))
    return str(output_folder_path)


def _write_native_model_file(
    output_folder_path: str,
    *,
    model_ir: ModelIR,
    metadata: Dict[str, Any],
    tensor_storage_name_map: Dict[str, str],
) -> List[Tuple[str, str]]:
    package_dir = Path(output_folder_path)
    _ensure_direct_codegen_supported(model_ir)
    tensor_var_names = _build_tensor_var_name_map(model_ir)
    producer_index, consumer_index = _build_model_ir_producer_consumer_index(model_ir)
    module_param_tensor_names: Set[str] = set()
    submodule_state_tensor_names: Set[str] = set()
    module_init_lines: List[str] = []
    load_specs: List[Tuple[str, str]] = []
    op_module_attr_names: Dict[int, str] = {}
    fused_module_specs: Dict[int, Dict[str, Any]] = {}
    affine_layer_norm_specs: Dict[int, Dict[str, Any]] = {}
    nms_method_specs: List[Dict[str, Any]] = []
    module_attr_counts: Dict[str, int] = {}
    inlined_constant_tensor_names: Set[str] = set()
    skipped_op_indices: Set[int] = set()
    conv_module_pad_specs: Dict[int, Optional[List[int]]] = {}

    def _shape_literal(values: Sequence[int]) -> str:
        return repr(tuple(int(v) for v in list(values)))

    def _target_shape_literal(tensor_name: str) -> str:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return "None"
        return repr([int(v) for v in list(tensor.shape)])

    def _resize_target_shape_literal(output_name: str, input_name: str) -> str:
        output_tensor = model_ir.tensors.get(str(output_name), None)
        input_tensor = model_ir.tensors.get(str(input_name), None)
        if output_tensor is None:
            return "None"
        target_shape = [int(v) for v in list(output_tensor.shape)]
        if input_tensor is None:
            return repr(target_shape)
        input_shape = [int(v) for v in list(input_tensor.shape)]
        output_layout = normalize_logical_layout(output_tensor.logical_layout)
        input_layout = normalize_logical_layout(input_tensor.logical_layout)
        if (
            len(input_shape) == 4
            and len(target_shape) == 4
            and is_channel_first_logical_layout(input_layout)
            and is_channel_first_logical_layout(output_layout)
        ):
            if int(target_shape[1]) != int(input_shape[1]) and int(target_shape[-1]) == int(input_shape[1]):
                return repr([int(input_shape[0]), int(input_shape[1]), int(target_shape[1]), int(target_shape[2])])
            if int(target_shape[1]) != int(input_shape[-1]) and int(target_shape[-1]) == int(input_shape[-1]):
                return repr([int(input_shape[0]), int(input_shape[-1]), int(target_shape[1]), int(target_shape[2])])
        return repr(target_shape)

    def _tensor_shape_list(tensor_name: str) -> Optional[List[int]]:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            return None
        return [int(v) for v in list(tensor.shape)]

    def _shape_lists_equal(lhs: Optional[Sequence[int]], rhs: Optional[Sequence[int]]) -> bool:
        if lhs is None or rhs is None:
            return False
        return [int(v) for v in list(lhs)] == [int(v) for v in list(rhs)]

    def _conv2d_output_spatial_shape(
        *,
        input_hw: Sequence[int],
        kernel_hw: Sequence[int],
        stride_hw: Sequence[int],
        dilation_hw: Sequence[int],
        padding_mode: str,
    ) -> Optional[List[int]]:
        input_items = [int(v) for v in list(input_hw)]
        kernel_items = [int(v) for v in list(kernel_hw)]
        stride_items = [max(1, int(v)) for v in list(stride_hw)]
        dilation_items = [max(1, int(v)) for v in list(dilation_hw)]
        if len(input_items) != 2 or len(kernel_items) != 2:
            return None
        effective_kernel = [
            (int(kernel_items[idx]) - 1) * int(dilation_items[idx]) + 1
            for idx in range(2)
        ]
        padding = str(padding_mode or "SAME").upper()
        if padding == "SAME":
            return [
                max(1, int(math.ceil(float(int(input_items[idx])) / float(int(stride_items[idx])))))
                for idx in range(2)
            ]
        if padding != "VALID":
            return None
        output_hw: List[int] = []
        for idx in range(2):
            numerator = int(input_items[idx]) - int(effective_kernel[idx])
            if numerator < 0:
                return None
            output_hw.append(int(numerator // int(stride_items[idx])) + 1)
        return output_hw

    def _conv3d_output_spatial_shape(
        *,
        input_dhw: Sequence[int],
        kernel_dhw: Sequence[int],
        stride_dhw: Sequence[int],
        dilation_dhw: Sequence[int],
        padding_mode: str,
    ) -> Optional[List[int]]:
        input_items = [int(v) for v in list(input_dhw)]
        kernel_items = [int(v) for v in list(kernel_dhw)]
        stride_items = [max(1, int(v)) for v in list(stride_dhw)]
        dilation_items = [max(1, int(v)) for v in list(dilation_dhw)]
        if len(input_items) != 3 or len(kernel_items) != 3:
            return None
        effective_kernel = [
            (int(kernel_items[idx]) - 1) * int(dilation_items[idx]) + 1
            for idx in range(3)
        ]
        padding = str(padding_mode or "SAME").upper()
        if padding == "SAME":
            return [
                max(1, int(math.ceil(float(int(input_items[idx])) / float(int(stride_items[idx])))))
                for idx in range(3)
            ]
        if padding != "VALID":
            return None
        output_dhw: List[int] = []
        for idx in range(3):
            numerator = int(input_items[idx]) - int(effective_kernel[idx])
            if numerator < 0:
                return None
            output_dhw.append(int(numerator // int(stride_items[idx])) + 1)
        return output_dhw

    def _conv3d_transpose_output_spatial_shape(
        *,
        input_dhw: Sequence[int],
        kernel_dhw: Sequence[int],
        stride_dhw: Sequence[int],
        dilation_dhw: Sequence[int],
        padding_mode: str,
    ) -> Optional[List[int]]:
        input_items = [int(v) for v in list(input_dhw)]
        kernel_items = [int(v) for v in list(kernel_dhw)]
        stride_items = [max(1, int(v)) for v in list(stride_dhw)]
        dilation_items = [max(1, int(v)) for v in list(dilation_dhw)]
        if len(input_items) != 3 or len(kernel_items) != 3:
            return None
        effective_kernel = [
            (int(kernel_items[idx]) - 1) * int(dilation_items[idx]) + 1
            for idx in range(3)
        ]
        padding = str(padding_mode or "SAME").upper()
        if padding == "SAME":
            return [int(input_items[idx]) * int(stride_items[idx]) for idx in range(3)]
        if padding != "VALID":
            return None
        return [
            (int(input_items[idx]) - 1) * int(stride_items[idx]) + int(effective_kernel[idx])
            for idx in range(3)
        ]

    def _conv2d_same_pad_arg(
        *,
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        weight_shape: Optional[Sequence[int]],
        options: Optional[Dict[str, Any]],
        input_pre_permute: Optional[Sequence[int]] = None,
    ) -> Optional[List[int]]:
        if str((options or {}).get("padding", "SAME")).upper() != "SAME":
            return None
        if input_shape is None or output_shape is None or weight_shape is None:
            return None
        in_shape = [int(v) for v in list(input_shape)]
        out_shape = [int(v) for v in list(output_shape)]
        kernel_shape = [int(v) for v in list(weight_shape)]
        if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
            return None
        if input_pre_permute is not None:
            perm = [int(v) for v in list(input_pre_permute)]
            if len(perm) != 4:
                return None
            in_shape = [int(in_shape[idx]) for idx in perm]
        stride_hw = [
            max(1, int((options or {}).get("strideH", 1))),
            max(1, int((options or {}).get("strideW", 1))),
        ]
        dilation_hw = [
            max(1, int((options or {}).get("dilationHFactor", 1))),
            max(1, int((options or {}).get("dilationWFactor", 1))),
        ]
        input_hw = [int(in_shape[2]), int(in_shape[3])]
        output_hw = [int(out_shape[2]), int(out_shape[3])]
        if output_hw[0] <= 0:
            output_hw[0] = max(1, int(math.ceil(float(input_hw[0]) / float(stride_hw[0]))))
        if output_hw[1] <= 0:
            output_hw[1] = max(1, int(math.ceil(float(input_hw[1]) / float(stride_hw[1]))))
        effective_kernel = [
            (int(kernel_shape[2 + idx]) - 1) * int(dilation_hw[idx]) + 1
            for idx in range(2)
        ]
        pad_h_total = max((int(output_hw[0]) - 1) * int(stride_hw[0]) + int(effective_kernel[0]) - int(input_hw[0]), 0)
        pad_w_total = max((int(output_hw[1]) - 1) * int(stride_hw[1]) + int(effective_kernel[1]) - int(input_hw[1]), 0)
        if pad_h_total == 0 and pad_w_total == 0:
            return None
        pad_top = int(pad_h_total // 2)
        pad_bottom = int(pad_h_total - pad_top)
        pad_left = int(pad_w_total // 2)
        pad_right = int(pad_w_total - pad_left)
        return [pad_left, pad_right, pad_top, pad_bottom]

    def _infer_conv2d_layout_candidate(
        *,
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        weight_shape: Optional[Sequence[int]],
        options: Optional[Dict[str, Any]],
        depthwise: bool,
    ) -> Optional[Tuple[List[int], int, int]]:
        if input_shape is None or output_shape is None or weight_shape is None:
            return None
        in_shape = [int(v) for v in list(input_shape)]
        out_shape = [int(v) for v in list(output_shape)]
        kernel_shape = [int(v) for v in list(weight_shape)]
        if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
            return None
        if in_shape[0] > 0 and out_shape[0] > 0 and int(in_shape[0]) != int(out_shape[0]):
            return None
        out_channels = int(kernel_shape[0])
        if out_shape[1] > 0 and out_channels > 0 and int(out_shape[1]) != int(out_channels):
            return None
        stride_hw = [
            int((options or {}).get("strideH", 1)),
            int((options or {}).get("strideW", 1)),
        ]
        dilation_hw = [
            int((options or {}).get("dilationHFactor", 1)),
            int((options or {}).get("dilationWFactor", 1)),
        ]
        padding_mode = str((options or {}).get("padding", "SAME"))
        import itertools

        for tail_perm in itertools.permutations((1, 2, 3)):
            perm = [0, *[int(v) for v in tail_perm]]
            permuted_shape = [int(in_shape[idx]) for idx in perm]
            in_channels = int(permuted_shape[1])
            if in_channels <= 0:
                continue
            if depthwise:
                if int(kernel_shape[1]) != 1 or int(out_channels) % int(in_channels) != 0:
                    continue
                groups = int(in_channels)
            else:
                weight_in_channels = int(kernel_shape[1])
                if weight_in_channels <= 0 or int(in_channels) % int(weight_in_channels) != 0:
                    continue
                groups = int(in_channels) // int(weight_in_channels)
                if groups <= 0 or int(out_channels) % int(groups) != 0:
                    continue
            expected_output_hw = _conv2d_output_spatial_shape(
                input_hw=[int(permuted_shape[2]), int(permuted_shape[3])],
                kernel_hw=[int(kernel_shape[2]), int(kernel_shape[3])],
                stride_hw=stride_hw,
                dilation_hw=dilation_hw,
                padding_mode=padding_mode,
            )
            if expected_output_hw is None:
                continue
            if (
                out_shape[2] > 0
                and out_shape[3] > 0
                and expected_output_hw != [int(out_shape[2]), int(out_shape[3])]
            ):
                continue
            return (perm, int(in_channels), int(groups))
        return None

    def _conv2d_input_pre_permute(
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        weight_shape: Optional[Sequence[int]],
        options: Optional[Dict[str, Any]],
        *,
        depthwise: bool = False,
    ) -> Optional[List[int]]:
        if input_shape is None or output_shape is None or weight_shape is None:
            return None
        in_shape = [int(v) for v in list(input_shape)]
        out_shape = [int(v) for v in list(output_shape)]
        kernel_shape = [int(v) for v in list(weight_shape)]
        if len(in_shape) != 4 or len(out_shape) != 4 or len(kernel_shape) != 4:
            return None
        # Depthwise/native NCHW case: keep the input as-is when the channel axis
        # already matches the depthwise filter/output channel count.
        if (
            int(kernel_shape[1]) == 1
            and int(in_shape[1]) == int(kernel_shape[0])
            and int(out_shape[1]) == int(kernel_shape[0])
        ):
            return None
        candidate_channels: List[int] = []
        for candidate in (int(in_shape[1]), int(in_shape[3])):
            if candidate > 0 and candidate not in candidate_channels:
                candidate_channels.append(candidate)
        expected_in_channels = None
        best_groups = None
        for candidate in candidate_channels:
            if int(kernel_shape[1]) <= 0 or int(candidate) % int(kernel_shape[1]) != 0:
                continue
            inferred_groups = int(candidate) // int(kernel_shape[1])
            if inferred_groups <= 0 or int(kernel_shape[0]) % int(inferred_groups) != 0:
                continue
            if best_groups is None or int(inferred_groups) < int(best_groups):
                expected_in_channels = int(candidate)
                best_groups = int(inferred_groups)
        if expected_in_channels is None:
            expected_in_channels = int(kernel_shape[1])
        if int(in_shape[1]) != expected_in_channels and int(in_shape[3]) == expected_in_channels:
            return [0, 3, 1, 2]
        if (
            kernel_shape[2] == 1
            and kernel_shape[3] > 1
            and in_shape[2] > 1
            and in_shape[3] == 1
            and out_shape[2] == 1
            and out_shape[3] > 1
        ):
            return [0, 1, 3, 2]

        inferred_layout = _infer_conv2d_layout_candidate(
            input_shape=in_shape,
            output_shape=out_shape,
            weight_shape=kernel_shape,
            options=options,
            depthwise=depthwise,
        )
        if inferred_layout is None:
            return None
        perm, _, _ = inferred_layout
        if perm == [0, 1, 2, 3]:
            return None
        return perm

    def _reshape_preserves_channel_last_sequence(
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        input_layout: Optional[str],
    ) -> Optional[List[int]]:
        if input_shape is None or output_shape is None:
            return None
        src = [int(v) for v in list(input_shape)]
        dst = [int(v) for v in list(output_shape)]
        layout = str(input_layout or "").upper()
        if layout == "NCHW" and len(src) == 4 and len(dst) == 3:
            flattened_spatial = int(src[2]) * int(src[3])
            sequence_extent_matches = (
                dst[1] == -1
                or (dst[2] > 0 and flattened_spatial * max(1, int(src[1]) // int(dst[2])) == dst[1])
                or flattened_spatial == dst[1]
            )
            if (
                src[0] == dst[0]
                and dst[2] > 0
                and int(src[1]) % int(dst[2]) == 0
                and src[2] > 0
                and src[3] > 0
                and sequence_extent_matches
            ):
                return [0, 2, 3, 1]
        if layout == "NCDHW" and len(src) == 5 and len(dst) == 3:
            spatial = src[2] * src[3] * src[4]
            sequence_extent_matches = (
                dst[1] == -1
                or (dst[2] > 0 and spatial * max(1, int(src[1]) // int(dst[2])) == dst[1])
                or spatial == dst[1]
            )
            if (
                src[0] == dst[0]
                and dst[2] > 0
                and int(src[1]) % int(dst[2]) == 0
                and sequence_extent_matches
            ):
                return [0, 2, 3, 4, 1]
        if layout == "NCW" and len(src) == 3 and len(dst) == 3:
            sequence_extent_matches = (
                dst[1] == -1
                or (dst[2] > 0 and int(src[2]) * max(1, int(src[1]) // int(dst[2])) == dst[1])
                or src[2] == dst[1]
            )
            if (
                src[0] == dst[0]
                and dst[2] > 0
                and int(src[1]) % int(dst[2]) == 0
                and sequence_extent_matches
            ):
                return [0, 2, 1]
        return None

    def _infer_conv2d_ctor_params(
        *,
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        weight_shape: Optional[Sequence[int]],
        options: Optional[Dict[str, Any]],
        depthwise: bool,
    ) -> Tuple[int, int]:
        if input_shape is None or weight_shape is None:
            return (1, 1)
        in_shape = [int(v) for v in list(input_shape)]
        kernel_shape = [int(v) for v in list(weight_shape)]
        if len(in_shape) != 4 or len(kernel_shape) != 4:
            return (1, 1)
        inferred_layout = _infer_conv2d_layout_candidate(
            input_shape=in_shape,
            output_shape=output_shape,
            weight_shape=kernel_shape,
            options=options,
            depthwise=depthwise,
        )
        if inferred_layout is not None:
            _, inferred_in_channels, inferred_groups = inferred_layout
            return (max(1, int(inferred_in_channels)), max(1, int(inferred_groups)))
        candidate_channels: List[int] = []
        for candidate in (int(in_shape[1]), int(in_shape[3])):
            if candidate > 0 and candidate not in candidate_channels:
                candidate_channels.append(candidate)
        out_channels = max(1, int(kernel_shape[0]))
        if depthwise:
            valid_candidates = [
                int(candidate)
                for candidate in candidate_channels
                if int(candidate) > 0 and int(out_channels) % int(candidate) == 0
            ]
            if len(valid_candidates) == 0:
                inferred_in_channels = int(candidate_channels[-1]) if len(candidate_channels) > 0 else 1
            else:
                inferred_in_channels = int(max(valid_candidates))
            return (max(1, inferred_in_channels), max(1, inferred_in_channels))

        weight_in_channels = max(1, int(kernel_shape[1]))
        best_choice: Optional[Tuple[int, int]] = None
        for candidate in candidate_channels:
            if int(candidate) % int(weight_in_channels) != 0:
                continue
            groups = int(candidate) // int(weight_in_channels)
            if groups <= 0 or int(out_channels) % int(groups) != 0:
                continue
            choice = (int(candidate), int(groups))
            if best_choice is None or int(choice[1]) < int(best_choice[1]):
                best_choice = choice
        if best_choice is not None:
            return best_choice
        fallback_in_channels = int(candidate_channels[-1]) if len(candidate_channels) > 0 else int(weight_in_channels)
        return (max(1, fallback_in_channels), 1)

    def _infer_conv3d_ctor_params(
        *,
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        weight_shape: Optional[Sequence[int]],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[int, int, int, List[int]]:
        if input_shape is None or output_shape is None or weight_shape is None:
            return (1, 1, 1, [1, 1, 1])
        in_shape = [int(v) for v in list(input_shape)]
        out_shape = [int(v) for v in list(output_shape)]
        kernel_shape = [int(v) for v in list(weight_shape)]
        if len(in_shape) != 5 or len(out_shape) != 5 or len(kernel_shape) != 5:
            return (1, 1, max(1, int(kernel_shape[0]) if len(kernel_shape) > 0 else 1), [1, 1, 1])
        if in_shape[0] > 0 and out_shape[0] > 0 and int(in_shape[0]) != int(out_shape[0]):
            return (max(1, int(in_shape[1])), 1, max(1, int(out_shape[1])), [int(v) for v in list(kernel_shape[2:5])])
        input_channels = max(1, int(in_shape[1]))
        expected_out_channels = max(1, int(out_shape[1]))
        stride_dhw = [
            int((options or {}).get("strideD", 1)),
            int((options or {}).get("strideH", 1)),
            int((options or {}).get("strideW", 1)),
        ]
        dilation_dhw = [
            int((options or {}).get("dilationDFactor", 1)),
            int((options or {}).get("dilationHFactor", 1)),
            int((options or {}).get("dilationWFactor", 1)),
        ]
        padding_mode = str((options or {}).get("padding", "SAME"))

        import itertools

        best_choice: Optional[Tuple[int, int, int, List[int]]] = None
        for out_axis in range(5):
            out_channels = int(kernel_shape[out_axis])
            if out_channels <= 0 or out_channels != expected_out_channels:
                continue
            for in_axis in range(5):
                if in_axis == out_axis:
                    continue
                weight_in_channels = int(kernel_shape[in_axis])
                if weight_in_channels <= 0 or int(input_channels) % int(weight_in_channels) != 0:
                    continue
                groups = int(input_channels) // int(weight_in_channels)
                if groups <= 0 or int(out_channels) % int(groups) != 0:
                    continue
                kernel_axes = [idx for idx in range(5) if idx not in {out_axis, in_axis}]
                if len(kernel_axes) != 3:
                    continue
                for kernel_order in itertools.permutations(kernel_axes):
                    kernel_dhw = [int(kernel_shape[idx]) for idx in kernel_order]
                    expected_output_dhw = _conv3d_output_spatial_shape(
                        input_dhw=[int(in_shape[2]), int(in_shape[3]), int(in_shape[4])],
                        kernel_dhw=kernel_dhw,
                        stride_dhw=stride_dhw,
                        dilation_dhw=dilation_dhw,
                        padding_mode=padding_mode,
                    )
                    if expected_output_dhw is None:
                        continue
                    if expected_output_dhw != [int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]:
                        continue
                    choice = (int(input_channels), int(groups), int(out_channels), [int(v) for v in kernel_dhw])
                    if best_choice is None or int(choice[1]) < int(best_choice[1]):
                        best_choice = choice
        if best_choice is not None:
            return best_choice
        fallback_kernel = [int(v) for v in list(kernel_shape[2:5])]
        return (max(1, int(input_channels)), 1, max(1, int(expected_out_channels)), fallback_kernel)

    def _infer_conv3d_transpose_ctor_params(
        *,
        input_shape: Optional[Sequence[int]],
        output_shape: Optional[Sequence[int]],
        weight_shape: Optional[Sequence[int]],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[int, int, List[int], int]:
        if input_shape is None or output_shape is None or weight_shape is None:
            return (1, 1, [1, 1, 1], 1)
        in_shape = [int(v) for v in list(input_shape)]
        out_shape = [int(v) for v in list(output_shape)]
        kernel_shape = [int(v) for v in list(weight_shape)]
        if len(in_shape) != 5 or len(out_shape) != 5 or len(kernel_shape) != 5:
            return (1, max(1, int(out_shape[1]) if len(out_shape) > 1 else 1), [1, 1, 1], 1)
        input_channels = max(1, int(in_shape[1]))
        expected_out_channels = max(1, int(out_shape[1]))
        stride_dhw = [
            int((options or {}).get("strideD", 1)),
            int((options or {}).get("strideH", 1)),
            int((options or {}).get("strideW", 1)),
        ]
        dilation_dhw = [
            int((options or {}).get("dilationDFactor", 1)),
            int((options or {}).get("dilationHFactor", 1)),
            int((options or {}).get("dilationWFactor", 1)),
        ]
        padding_mode = str((options or {}).get("padding", "SAME"))

        import itertools

        best_choice: Optional[Tuple[int, int, List[int], int]] = None
        for in_axis in range(5):
            weight_in_channels = int(kernel_shape[in_axis])
            if weight_in_channels <= 0 or weight_in_channels != input_channels:
                continue
            for out_axis in range(5):
                if out_axis == in_axis:
                    continue
                weight_out_per_group = int(kernel_shape[out_axis])
                if weight_out_per_group <= 0 or int(expected_out_channels) % int(weight_out_per_group) != 0:
                    continue
                groups = int(expected_out_channels) // int(weight_out_per_group)
                if groups <= 0 or int(input_channels) % int(groups) != 0:
                    continue
                kernel_axes = [idx for idx in range(5) if idx not in {in_axis, out_axis}]
                if len(kernel_axes) != 3:
                    continue
                for kernel_order in itertools.permutations(kernel_axes):
                    kernel_dhw = [int(kernel_shape[idx]) for idx in kernel_order]
                    expected_output_dhw = _conv3d_transpose_output_spatial_shape(
                        input_dhw=[int(in_shape[2]), int(in_shape[3]), int(in_shape[4])],
                        kernel_dhw=kernel_dhw,
                        stride_dhw=stride_dhw,
                        dilation_dhw=dilation_dhw,
                        padding_mode=padding_mode,
                    )
                    if expected_output_dhw is None:
                        continue
                    if expected_output_dhw != [int(out_shape[2]), int(out_shape[3]), int(out_shape[4])]:
                        continue
                    choice = (
                        int(input_channels),
                        int(expected_out_channels),
                        [int(v) for v in kernel_dhw],
                        int(groups),
                    )
                    if best_choice is None or int(choice[3]) < int(best_choice[3]):
                        best_choice = choice
        if best_choice is not None:
            return best_choice
        fallback_kernel = [int(v) for v in list(kernel_shape[2:5])]
        return (max(1, int(input_channels)), max(1, int(expected_out_channels)), fallback_kernel, 1)

    def _reshape_prefers_feature_last_for_adjx_batch_matmul(
        input_tensor_name: str,
        output_name: str,
    ) -> Optional[Tuple[List[int], List[int]]]:
        input_tensor = model_ir.tensors.get(str(input_tensor_name), None)
        output_tensor = model_ir.tensors.get(str(output_name), None)
        if input_tensor is None or output_tensor is None:
            return None
        input_shape = [int(v) for v in list(input_tensor.shape)]
        output_shape = [int(v) for v in list(output_tensor.shape)]
        if len(input_shape) != 4 or len(output_shape) != 3:
            return None
        perm: Optional[List[int]] = None
        spatial_extent: Optional[int] = None
        if input_shape[2] == 1 and input_shape[3] > 1:
            perm = [0, 3, 1, 2]
            spatial_extent = int(input_shape[3])
        elif input_shape[3] == 1 and input_shape[2] > 1:
            perm = [0, 2, 1, 3]
            spatial_extent = int(input_shape[2])
        if perm is None or spatial_extent is None:
            return None
        if output_shape[0] != input_shape[0] or output_shape[1] != input_shape[1] or output_shape[2] != spatial_extent:
            return None
        passthrough_ops = {
            "ABS",
            "ATAN",
            "CAST",
            "ELU",
            "ERF",
            "EXP",
            "GELU",
            "IDENTITY",
            "LEAKY_RELU",
            "LOG",
            "LOGISTIC",
            "NEG",
            "RELU",
            "RELU6",
            "RELU_0_TO_1",
            "RELU_N1_TO_1",
            "SIGN",
            "SIN",
            "SQRT",
            "SQUARE",
            "TANH",
        }
        pending_outputs: List[str] = [str(output_name)]
        visited_outputs: Set[str] = set()
        while pending_outputs:
            current_name = pending_outputs.pop()
            if current_name in visited_outputs:
                continue
            visited_outputs.add(current_name)
            for consumer_idx in consumer_index.get(current_name, []):
                consumer_op = model_ir.operators[int(consumer_idx)]
                consumer_type = str(consumer_op.op_type)
                if consumer_type == "BATCH_MATMUL":
                    if len(consumer_op.inputs) < 2 or str(consumer_op.inputs[0]) != current_name:
                        continue
                    if not bool(consumer_op.options.get("adjX", False)):
                        continue
                    rhs_tensor = model_ir.tensors.get(str(consumer_op.inputs[1]), None)
                    if rhs_tensor is None or len(list(rhs_tensor.shape)) < 2:
                        continue
                    rhs_contract = int(list(rhs_tensor.shape)[-2])
                    if rhs_contract != spatial_extent:
                        continue
                    return (perm, [int(input_shape[0]), int(spatial_extent), int(input_shape[1])])
                if consumer_type in passthrough_ops and len(consumer_op.outputs) == 1:
                    pending_outputs.append(str(consumer_op.outputs[0]))
        return None

    def _gather_input_pre_permute(
        *,
        params_name: str,
        output_name: str,
        axis: int,
        batch_dims: int,
    ) -> Optional[List[int]]:
        if int(batch_dims) != 0:
            return None
        params_tensor = model_ir.tensors.get(str(params_name), None)
        output_tensor = model_ir.tensors.get(str(output_name), None)
        if params_tensor is None or output_tensor is None:
            return None
        params_shape = [int(v) for v in list(params_tensor.shape)]
        rank = len(params_shape)
        if rank not in {3, 4, 5}:
            return None
        output_signature = (
            [int(v) for v in list(output_tensor.shape_signature)]
            if output_tensor.shape_signature is not None and len(list(output_tensor.shape_signature)) == rank
            else [int(v) for v in list(output_tensor.shape)]
        )
        if len(output_signature) != rank:
            return None
        resolved_axis = int(axis)
        if resolved_axis < 0:
            resolved_axis += rank
        if resolved_axis < 0 or resolved_axis >= rank:
            return None

        def _matches_signature(shape: Sequence[int]) -> bool:
            for dim_idx, expected_dim in enumerate(output_signature):
                if dim_idx == resolved_axis or int(expected_dim) <= 0:
                    continue
                if int(shape[dim_idx]) != int(expected_dim):
                    return False
            return True

        if _matches_signature(params_shape):
            return None
        for perm in (_perm_cf_to_cl(rank), _perm_cl_to_cf(rank)):
            perm_values = [int(v) for v in list(perm)]
            permuted_shape = [int(params_shape[int(idx)]) for idx in perm_values]
            if _matches_signature(permuted_shape):
                return perm_values
        return None

    def _topk_codegen_layout_bridge(
        *,
        input_name: str,
        value_output_name: str,
        index_output_name: Optional[str],
    ) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        input_shape = _tensor_shape_list(str(input_name))
        value_shape = _tensor_shape_list(str(value_output_name))
        index_shape = (
            _tensor_shape_list(str(index_output_name))
            if index_output_name is not None and str(index_output_name) != ""
            else None
        )
        if input_shape is None or value_shape is None:
            return None, None
        rank = len(input_shape)
        if rank not in {3, 4, 5} or len(value_shape) != rank:
            return None, None

        candidate_perms: List[List[int]] = []
        for axis in range(rank):
            perm = [int(v) for v in range(rank) if int(v) != int(axis)] + [int(axis)]
            if perm != list(range(rank)):
                candidate_perms.append(perm)

        import itertools

        for generic_perm in itertools.permutations(range(rank)):
            perm = [int(v) for v in generic_perm]
            if perm == list(range(rank)) or perm in candidate_perms:
                continue
            candidate_perms.append(perm)

        for pre_perm in candidate_perms:
            permuted_input_shape = [int(input_shape[int(idx)]) for idx in pre_perm]
            if permuted_input_shape != value_shape:
                continue
            inverse_perm = [0] * rank
            for new_axis, old_axis in enumerate(pre_perm):
                inverse_perm[int(old_axis)] = int(new_axis)
            if index_shape is None or index_shape == value_shape:
                return pre_perm, None
            if [int(value_shape[int(idx)]) for idx in inverse_perm] == index_shape:
                return pre_perm, inverse_perm
        return None, None

    def _matmul_broadcast_shape(
        lhs_batch: Sequence[int],
        rhs_batch: Sequence[int],
    ) -> Optional[List[int]]:
        lhs_items = [int(v) for v in list(lhs_batch)]
        rhs_items = [int(v) for v in list(rhs_batch)]
        result: List[int] = []
        for lhs_dim, rhs_dim in zip(reversed(lhs_items), reversed(rhs_items)):
            if int(lhs_dim) == int(rhs_dim):
                result.append(int(lhs_dim))
            elif int(lhs_dim) == 1:
                result.append(int(rhs_dim))
            elif int(rhs_dim) == 1:
                result.append(int(lhs_dim))
            else:
                return None
        if len(lhs_items) > len(rhs_items):
            result.extend(reversed(lhs_items[: len(lhs_items) - len(rhs_items)]))
        elif len(rhs_items) > len(lhs_items):
            result.extend(reversed(rhs_items[: len(rhs_items) - len(lhs_items)]))
        return list(reversed(result))

    def _infer_batch_matmul_shape(
        lhs_shape: Optional[Sequence[int]],
        rhs_shape: Optional[Sequence[int]],
        *,
        adj_x: bool,
        adj_y: bool,
    ) -> Optional[List[int]]:
        if lhs_shape is None or rhs_shape is None:
            return None
        lhs_items = [int(v) for v in list(lhs_shape)]
        rhs_items = [int(v) for v in list(rhs_shape)]
        if len(lhs_items) == 0 or len(rhs_items) == 0:
            return None
        if len(lhs_items) == 1:
            lhs_items = [1, int(lhs_items[0])]
        if len(rhs_items) == 1:
            rhs_items = [int(rhs_items[0]), 1]
        if len(lhs_items) < 2 or len(rhs_items) < 2:
            return None
        lhs_m = int(lhs_items[-1 if adj_x else -2])
        lhs_k = int(lhs_items[-2 if adj_x else -1])
        rhs_k = int(rhs_items[-1 if adj_y else -2])
        rhs_n = int(rhs_items[-2 if adj_y else -1])
        if int(lhs_k) != int(rhs_k):
            return None
        batch_shape = _matmul_broadcast_shape(lhs_items[:-2], rhs_items[:-2])
        if batch_shape is None:
            return None
        return list(batch_shape) + [int(lhs_m), int(rhs_n)]

    def _infer_reduction_shape(
        input_shape: Optional[Sequence[int]],
        axes: Optional[Sequence[int]],
        *,
        keepdims: bool,
    ) -> Optional[List[int]]:
        if input_shape is None:
            return None
        dims = [int(v) for v in list(input_shape)]
        if axes is None:
            return [1 for _ in dims] if keepdims else []
        normalized_axes = sorted({int(v) for v in list(axes)})
        if keepdims:
            return [1 if idx in normalized_axes else int(dim) for idx, dim in enumerate(dims)]
        return [int(dim) for idx, dim in enumerate(dims) if idx not in normalized_axes]

    def _infer_gather_nd_shape(
        params_shape: Optional[Sequence[int]],
        indices_tensor_name: str,
    ) -> Optional[List[int]]:
        if params_shape is None:
            return None
        indices_tensor = model_ir.tensors.get(str(indices_tensor_name), None)
        if indices_tensor is None or not isinstance(indices_tensor.data, np.ndarray):
            return None
        indices_shape = [int(v) for v in list(indices_tensor.shape)]
        if len(indices_shape) == 0:
            return None
        index_depth = int(indices_shape[-1])
        params_items = [int(v) for v in list(params_shape)]
        if index_depth > len(params_items):
            return None
        return indices_shape[:-1] + params_items[index_depth:]

    def _emit_maybe_aligned_expr(
        *,
        output_name: str,
        expr: str,
        inferred_shape: Optional[Sequence[int]],
    ) -> str:
        output_shape = _tensor_shape_list(output_name)
        if _shape_lists_equal(inferred_shape, output_shape):
            return expr
        runtime_imports.add("_align_tensor_to_target_shape")
        return f"_align_tensor_to_target_shape({expr}, {_target_shape_literal(output_name)})"

    def _should_skip_align_for_shape_preserving_unary(
        input_name: str,
        output_name: str,
    ) -> bool:
        input_tensor = model_ir.tensors.get(str(input_name), None)
        output_tensor = model_ir.tensors.get(str(output_name), None)
        if input_tensor is None or output_tensor is None:
            return False
        input_layout = normalize_logical_layout(input_tensor.logical_layout)
        output_layout = normalize_logical_layout(output_tensor.logical_layout)
        if input_layout != output_layout:
            return False
        input_shape = _tensor_shape_list(input_name)
        output_shape = _tensor_shape_list(output_name)
        if input_shape is None or output_shape is None:
            return False
        return _shape_lists_equal(input_shape, output_shape)

    def _is_constant_tensor_name(tensor_name: str) -> bool:
        tensor = model_ir.tensors.get(str(tensor_name), None)
        return tensor is not None and isinstance(tensor.data, np.ndarray)

    def _next_unique_attr_name(base_name: str) -> str:
        normalized = re.sub(r"[^0-9a-zA-Z]+", "_", str(base_name)).strip("_").lower()
        if len(normalized) == 0:
            normalized = "generated_module"
        if normalized[0].isdigit():
            normalized = f"n_{normalized}"
        candidate = normalized
        suffix = 1
        existing_names = {
            *module_attr_counts.keys(),
            *(str(spec.get("attr_name")) for spec in affine_layer_norm_specs.values()),
            *(str(value) for value in op_module_attr_names.values()),
        }
        while candidate in existing_names:
            candidate = f"{normalized}_{suffix}"
            suffix += 1
        module_attr_counts[candidate] = 1
        return candidate

    def _canonical_codegen_name(name: str) -> str:
        return re.sub(r"[^0-9a-z]+", "_", str(name).lower()).strip("_")

    def _match_affine_layer_norm(op_index: int, op: OperatorIR) -> Optional[Dict[str, Any]]:
        if str(op.op_type) != "ADD" or len(op.inputs) < 2 or len(op.outputs) != 1:
            return None
        output_name = str(op.outputs[0])
        canonical_output_name = _canonical_codegen_name(output_name)
        if "fakelayernorm" not in canonical_output_name or not canonical_output_name.endswith("add"):
            return None
        beta_input_name = ""
        mul_output_name = ""
        for input_name in op.inputs[:2]:
            input_tensor_name = str(input_name)
            canonical_input_name = _canonical_codegen_name(input_tensor_name)
            if _is_constant_tensor_name(input_tensor_name) and "fakelayernorm_beta" in canonical_input_name:
                beta_input_name = input_tensor_name
            else:
                mul_output_name = input_tensor_name
        if beta_input_name == "" or mul_output_name == "":
            return None
        mul_op_index = producer_index.get(str(mul_output_name), None)
        if mul_op_index is None:
            return None
        mul_op = model_ir.operators[int(mul_op_index)]
        if str(mul_op.op_type) != "MUL" or len(mul_op.inputs) < 2 or len(mul_op.outputs) != 1:
            return None
        if str(mul_op.outputs[0]) != mul_output_name:
            return None
        gamma_input_name = ""
        input_name = ""
        for mul_input_name in mul_op.inputs[:2]:
            candidate_name = str(mul_input_name)
            canonical_candidate_name = _canonical_codegen_name(candidate_name)
            if _is_constant_tensor_name(candidate_name) and "fakelayernorm_gamma" in canonical_candidate_name:
                gamma_input_name = candidate_name
            else:
                input_name = candidate_name
        if gamma_input_name == "" or input_name == "":
            return None
        gamma_tensor = model_ir.tensors.get(str(gamma_input_name), None)
        beta_tensor = model_ir.tensors.get(str(beta_input_name), None)
        if gamma_tensor is None or beta_tensor is None:
            return None
        attr_stem = re.sub(r"(?i)(?:[/_])?FakeLayerNorm(?:[/_])add$", "", output_name)
        attr_stem = re.sub(r"^bert[/_]", "", attr_stem, flags=re.IGNORECASE)
        attr_name = _next_unique_attr_name(f"{attr_stem}_layer_norm")
        return {
            "attr_name": attr_name,
            "input_name": str(input_name),
            "output_name": output_name,
            "gamma_name": str(gamma_input_name),
            "beta_name": str(beta_input_name),
            "gamma_shape": [int(v) for v in list(gamma_tensor.shape)],
            "gamma_dtype": str(gamma_tensor.dtype).upper(),
            "mul_op_index": int(mul_op_index),
        }

    for op_index, op in enumerate(model_ir.operators):
        affine_layer_norm_spec = _match_affine_layer_norm(int(op_index), op)
        if affine_layer_norm_spec is None:
            continue
        attr_name = str(affine_layer_norm_spec["attr_name"])
        module_init_lines.extend(
            [
                f"self.{attr_name} = _AffineLayerNorm(",
                f"    shape={repr(list(affine_layer_norm_spec['gamma_shape']))},",
                f"    dtype={_torch_dtype_literal(str(affine_layer_norm_spec['gamma_dtype']))},",
                ")",
            ]
        )
        load_specs.append((f"{attr_name}.gamma", str(affine_layer_norm_spec["gamma_name"])))
        load_specs.append((f"{attr_name}.beta", str(affine_layer_norm_spec["beta_name"])))
        submodule_state_tensor_names.update(
            {
                str(affine_layer_norm_spec["gamma_name"]),
                str(affine_layer_norm_spec["beta_name"]),
            }
        )
        affine_layer_norm_specs[int(op_index)] = affine_layer_norm_spec
        skipped_op_indices.add(int(affine_layer_norm_spec["mul_op_index"]))

    for op_index, op in enumerate(model_ir.operators):
        op_type = str(op.op_type)
        if op_type not in _DIRECT_CODEGEN_MODULE_OP_TYPES:
            continue
        attr_base = _direct_codegen_module_attr_base(op_type)
        attr_index = int(module_attr_counts.get(attr_base, 0))
        module_attr_counts[attr_base] = attr_index + 1
        attr_name = f"{attr_base}_{attr_index}"
        op_module_attr_names[int(op_index)] = attr_name
        weight_name = str(op.inputs[1]) if len(op.inputs) >= 2 else ""
        bias_name = ""
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"} and len(op.inputs) >= 3 and str(op.inputs[2]) != "":
            bias_name = str(op.inputs[2])
        elif op_type == "FULLY_CONNECTED" and len(op.inputs) >= 3 and str(op.inputs[2]) != "":
            bias_name = str(op.inputs[2])
        elif op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} and len(op.inputs) >= 4 and str(op.inputs[3]) != "":
            bias_name = str(op.inputs[3])
        module_param_tensor_names.add(weight_name)
        if bias_name != "":
            module_param_tensor_names.add(bias_name)
        weight_tensor = model_ir.tensors[weight_name]
        input_tensor_name = str(op.inputs[0]) if op_type not in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} else str(op.inputs[2])
        input_tensor = model_ir.tensors[input_tensor_name]
        options = dict(op.options)
        conv_input_pre_permute: Optional[List[int]] = None
        conv_same_pad_arg: Optional[List[int]] = None
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
            conv_input_pre_permute = _conv2d_input_pre_permute(
                _tensor_shape_list(str(input_tensor_name)),
                _tensor_shape_list(str(op.outputs[0])),
                _tensor_shape_list(str(op.inputs[1])),
                options,
                depthwise=(op_type == "DEPTHWISE_CONV_2D"),
            )
            conv_same_pad_arg = _conv2d_same_pad_arg(
                input_shape=_tensor_shape_list(str(input_tensor_name)),
                output_shape=_tensor_shape_list(str(op.outputs[0])),
                weight_shape=_tensor_shape_list(str(op.inputs[1])),
                options=options,
                input_pre_permute=conv_input_pre_permute,
            )
            conv_module_pad_specs[int(op_index)] = conv_same_pad_arg
        fused_block_input_name = str(input_tensor_name)
        fused_block_output_name = str(op.outputs[0]) if len(op.outputs) == 1 else ""
        fused_block_pad: Optional[str] = None
        fused_block_pad_mode = "constant"
        fused_block_pad_value: Optional[str] = None
        fused_block_activation, fused_block_negative_slope = _conv_block_activation_config_from_fused_name(
            str(op.options.get("fusedActivationFunction", "NONE")),
            alpha=(
                float(op.options.get("alpha", 0.2))
                if "alpha" in op.options
                else None
            ),
        )
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"} and len(op.outputs) == 1:
            input_producer_idx = producer_index.get(str(input_tensor_name), None)
            if input_producer_idx is not None:
                input_producer = model_ir.operators[int(input_producer_idx)]
                input_producer_type = str(input_producer.op_type)
                if (
                    input_producer_type in {"PAD", "PADV2"}
                    and len(consumer_index.get(str(input_tensor_name), [])) == 1
                    and len(input_producer.inputs) >= 2
                ):
                    pad_literal = _torch_pad_literal_for_constant_tensor(
                        model_ir.tensors.get(str(input_producer.inputs[1]), None)
                    )
                    if pad_literal is not None:
                        fused_block_input_name = str(input_producer.inputs[0])
                        fused_block_pad = pad_literal
                        if input_producer_type == "PADV2" and len(input_producer.inputs) >= 3:
                            scalar_literal = _scalar_literal_for_constant_tensor(
                                model_ir.tensors.get(str(input_producer.inputs[2]), None)
                            )
                            if scalar_literal is not None:
                                fused_block_pad_value = scalar_literal
                        skipped_op_indices.add(int(input_producer_idx))
                        inlined_constant_tensor_names.add(str(input_producer.inputs[1]))
                        if input_producer_type == "PADV2" and len(input_producer.inputs) >= 3:
                            inlined_constant_tensor_names.add(str(input_producer.inputs[2]))
            output_consumer_indices = consumer_index.get(str(op.outputs[0]), [])
            if len(output_consumer_indices) == 1:
                activation_op = model_ir.operators[int(output_consumer_indices[0])]
                activation_type = str(activation_op.op_type)
                if activation_type in {
                    "LEAKY_RELU",
                    "LOGISTIC",
                    "RELU",
                    "RELU6",
                    "RELU_N1_TO_1",
                    "RELU_0_TO_1",
                    "TANH",
                } and len(activation_op.inputs) == 1 and len(activation_op.outputs) == 1 and fused_block_activation == "none":
                    fused_block_activation, fused_block_negative_slope = _conv_block_activation_config(activation_op)
                    fused_block_output_name = str(activation_op.outputs[0])
                    skipped_op_indices.add(int(output_consumer_indices[0]))
        use_conv_block = (
            op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
            and (fused_block_pad is not None or fused_block_activation != "none")
        )
        if use_conv_block:
            attr_base = "conv_block"
            attr_index = int(module_attr_counts.get(attr_base, 0))
            module_attr_counts[attr_base] = attr_index + 1
            attr_name = f"{attr_base}_{attr_index}"
            op_module_attr_names[int(op_index)] = attr_name
            fused_module_specs[int(op_index)] = {
                "input_name": str(fused_block_input_name),
                "output_name": str(fused_block_output_name),
                "input_pre_permute": conv_input_pre_permute,
                "pad": fused_block_pad if fused_block_pad is not None else (repr(conv_same_pad_arg) if conv_same_pad_arg is not None else None),
                "pad_mode": fused_block_pad_mode,
                "pad_value": fused_block_pad_value,
                "activation": fused_block_activation,
                "negative_slope": fused_block_negative_slope,
            }
        if op_type == "CONV_2D":
            conv_padding = (
                [0, 0]
                if conv_same_pad_arg is not None
                else [
                    int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                    int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                ]
            )
            conv_in_channels, conv_groups = _infer_conv2d_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
                depthwise=False,
            )
            conv_ctor_lines = [
                "torch.nn.Conv2d(",
                f"    in_channels={conv_in_channels},",
                f"    out_channels={int(weight_tensor.shape[0])},",
                f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                f"    padding={_shape_literal(conv_padding)},",
                f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                f"    groups={conv_groups},",
                f"    bias={str(bias_name != '')},",
                ")",
            ]
            if use_conv_block:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = _Conv2dBlock(",
                        f"    in_channels={conv_in_channels},",
                        f"    out_channels={int(weight_tensor.shape[0])},",
                        f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                        f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                        f"    padding={_shape_literal(conv_padding)},",
                        f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                        f"    groups={conv_groups},",
                        f"    bias={str(bias_name != '')},",
                        f"    pad={fused_block_pad if fused_block_pad is not None else (repr(conv_same_pad_arg) if conv_same_pad_arg is not None else 'None')},",
                        f"    activation={fused_block_activation!r},",
                        f"    negative_slope={repr(fused_block_negative_slope if fused_block_negative_slope is not None else 0.2)},",
                        f"    pad_mode={fused_block_pad_mode!r},",
                        f"    pad_value={fused_block_pad_value if fused_block_pad_value is not None else '0.0'},",
                        ")",
                    ]
                )
            else:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = {conv_ctor_lines[0]}",
                        *conv_ctor_lines[1:],
                    ]
                )
        elif op_type == "DEPTHWISE_CONV_2D":
            conv_padding = (
                [0, 0]
                if conv_same_pad_arg is not None
                else [
                    int(((int(weight_tensor.shape[2]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                    int(((int(weight_tensor.shape[3]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0,
                ]
            )
            depthwise_in_channels, depthwise_groups = _infer_conv2d_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
                depthwise=True,
            )
            conv_ctor_lines = [
                "torch.nn.Conv2d(",
                f"    in_channels={depthwise_in_channels},",
                f"    out_channels={int(weight_tensor.shape[0])},",
                f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                f"    padding={_shape_literal(conv_padding)},",
                f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                f"    groups={depthwise_groups},",
                f"    bias={str(bias_name != '')},",
                ")",
            ]
            if use_conv_block:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = _Conv2dBlock(",
                        f"    in_channels={depthwise_in_channels},",
                        f"    out_channels={int(weight_tensor.shape[0])},",
                        f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                        f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                        f"    padding={_shape_literal(conv_padding)},",
                        f"    dilation={_shape_literal([int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                        f"    groups={depthwise_groups},",
                        f"    bias={str(bias_name != '')},",
                        f"    pad={fused_block_pad if fused_block_pad is not None else (repr(conv_same_pad_arg) if conv_same_pad_arg is not None else 'None')},",
                        f"    activation={fused_block_activation!r},",
                        f"    negative_slope={repr(fused_block_negative_slope if fused_block_negative_slope is not None else 0.2)},",
                        f"    pad_mode={fused_block_pad_mode!r},",
                        f"    pad_value={fused_block_pad_value if fused_block_pad_value is not None else '0.0'},",
                        ")",
                    ]
                )
            else:
                module_init_lines.extend(
                    [
                        f"self.{attr_name} = {conv_ctor_lines[0]}",
                        *conv_ctor_lines[1:],
                    ]
                )
        elif op_type == "TRANSPOSE_CONV":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.ConvTranspose2d(",
                    f"    in_channels={int(weight_tensor.shape[0])},",
                    f"    out_channels={int(weight_tensor.shape[1])},",
                    f"    kernel_size={_shape_literal(weight_tensor.shape[2:])},",
                    f"    stride={_shape_literal([int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(weight_tensor.shape[2]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(weight_tensor.shape[3]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "CONV_3D":
            conv3d_in_channels, conv3d_groups, conv3d_out_channels, conv3d_kernel_size = _infer_conv3d_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
            )
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Conv3d(",
                    f"    in_channels={conv3d_in_channels},",
                    f"    out_channels={conv3d_out_channels},",
                    f"    kernel_size={_shape_literal(conv3d_kernel_size)},",
                    f"    stride={_shape_literal([int(options.get('strideD', 1)), int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(conv3d_kernel_size[0]) - 1) * int(options.get('dilationDFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3d_kernel_size[1]) - 1) * int(options.get('dilationHFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3d_kernel_size[2]) - 1) * int(options.get('dilationWFactor', 1))) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    dilation={_shape_literal([int(options.get('dilationDFactor', 1)), int(options.get('dilationHFactor', 1)), int(options.get('dilationWFactor', 1))])},",
                    f"    groups={conv3d_groups},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "CONV_3D_TRANSPOSE":
            conv3dt_in_channels, conv3dt_out_channels, conv3dt_kernel_size, conv3dt_groups = _infer_conv3d_transpose_ctor_params(
                input_shape=list(input_tensor.shape),
                output_shape=list(model_ir.tensors[str(op.outputs[0])].shape),
                weight_shape=list(weight_tensor.shape),
                options=options,
            )
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.ConvTranspose3d(",
                    f"    in_channels={conv3dt_in_channels},",
                    f"    out_channels={conv3dt_out_channels},",
                    f"    kernel_size={_shape_literal(conv3dt_kernel_size)},",
                    f"    stride={_shape_literal([int(options.get('strideD', 1)), int(options.get('strideH', 1)), int(options.get('strideW', 1))])},",
                    f"    padding={_shape_literal([int(((int(conv3dt_kernel_size[0]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3dt_kernel_size[1]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0, int(((int(conv3dt_kernel_size[2]) - 1)) // 2) if str(options.get('padding', 'SAME')).upper() != 'VALID' else 0])},",
                    f"    groups={conv3dt_groups},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "FULLY_CONNECTED":
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.Linear(",
                    f"    in_features={int(weight_tensor.shape[1])},",
                    f"    out_features={int(weight_tensor.shape[0])},",
                    f"    bias={str(bias_name != '')},",
                    ")",
                ]
            )
        elif op_type == "PRELU":
            num_parameters = 1
            if isinstance(weight_tensor.data, np.ndarray):
                num_parameters = max(1, int(np.asarray(weight_tensor.data).size))
            elif len(list(weight_tensor.shape)) > 0:
                shape_values = [int(v) for v in list(weight_tensor.shape) if int(v) > 0]
                if len(shape_values) > 0:
                    num_parameters = max(1, int(np.prod(shape_values, dtype=np.int64)))
            module_init_lines.extend(
                [
                    f"self.{attr_name} = torch.nn.PReLU(",
                    f"    num_parameters={num_parameters},",
                    "    init=0.0,",
                    ")",
                ]
            )
        load_specs.append((f"{attr_name}.conv.weight" if use_conv_block else f"{attr_name}.weight", str(weight_name)))
        if bias_name != "":
            load_specs.append((f"{attr_name}.conv.bias" if use_conv_block else f"{attr_name}.bias", str(bias_name)))

    for op in model_ir.operators:
        op_type = str(op.op_type)
        if op_type not in {"PAD", "PADV2", "MIRROR_PAD"}:
            continue
        if len(op.inputs) >= 2:
            paddings_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            if paddings_tensor is not None and isinstance(paddings_tensor.data, np.ndarray):
                inlined_constant_tensor_names.add(str(op.inputs[1]))
        if op_type == "PADV2" and len(op.inputs) >= 3:
            value_tensor = model_ir.tensors.get(str(op.inputs[2]), None)
            if value_tensor is not None and isinstance(value_tensor.data, np.ndarray):
                inlined_constant_tensor_names.add(str(op.inputs[2]))
    for tensor_name, tensor in model_ir.tensors.items():
        if str(tensor_name) in module_param_tensor_names:
            continue
        if _is_small_inline_constant_tensor(tensor):
            inlined_constant_tensor_names.add(str(tensor_name))

    buffer_attr_names = _build_buffer_attr_name_map(
        model_ir=model_ir,
        tensor_storage_name_map=tensor_storage_name_map,
        excluded_tensor_names=module_param_tensor_names | submodule_state_tensor_names | inlined_constant_tensor_names,
    )
    buffer_init_lines: List[str] = []
    for tensor_name, attr_name in buffer_attr_names.items():
        tensor = model_ir.tensors[str(tensor_name)]
        dtype_name = str(tensor.dtype).upper()
        shape_values = [int(v) for v in list(tensor.shape)]
        if bool(tensor.is_variable):
            buffer_init_lines.append(
                f"self.register_parameter({attr_name!r}, torch.nn.Parameter(torch.zeros({repr(shape_values)}, dtype={_torch_dtype_literal(dtype_name)}), requires_grad=False))"
            )
        else:
            buffer_init_lines.append(
                f"self.register_buffer({attr_name!r}, torch.zeros({repr(shape_values)}, dtype={_torch_dtype_literal(dtype_name)}), persistent=True)"
            )
        load_specs.append((str(attr_name), str(tensor_name)))

    def _tensor_expr(tensor_name: str) -> str:
        if str(tensor_name) in tensor_var_names:
            return str(tensor_var_names[str(tensor_name)])
        if str(tensor_name) in buffer_attr_names:
            return f"self.{buffer_attr_names[str(tensor_name)]}"
        tensor = model_ir.tensors.get(str(tensor_name), None)
        literal = _python_literal_for_constant_tensor(tensor) if tensor is not None else None
        if tensor is not None and literal is not None:
            return (
                f"torch.as_tensor({literal}, dtype={_torch_dtype_literal(str(tensor.dtype).upper())}, "
                "device=self._device())"
            )
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen could not resolve a tensor expression. "
            f"tensor={tensor_name}"
        )

    def _binary_operand_expr(tensor_name: str, other_tensor_name: str) -> str:
        expr = _tensor_expr(tensor_name)
        tensor = model_ir.tensors.get(str(tensor_name), None)
        other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
        if tensor is None or other_tensor is None:
            return expr
        if not isinstance(tensor.data, np.ndarray):
            return expr
        tensor_shape = [int(v) for v in list(tensor.shape)]
        other_shape = [int(v) for v in list(other_tensor.shape)]
        tensor_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in tensor_shape]
        other_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in other_shape]
        constant_width = max([int(dim) for dim in tensor_shape], default=0)
        if len(tensor_shape) < len(other_shape):
            target_axis: Optional[int] = None
            other_layout = normalize_logical_layout(other_tensor.logical_layout)
            if (
                is_channel_first_logical_layout(other_layout)
                and len(other_shape) >= 2
                and int(other_shape[1]) == constant_width
            ):
                target_axis = 1
            elif (
                is_channel_last_logical_layout(other_layout)
                and len(other_shape) >= 1
                and int(other_shape[-1]) == constant_width
            ):
                target_axis = len(other_shape) - 1
            else:
                matching_axes = [
                    int(axis)
                    for axis, dim in enumerate(other_shape)
                    if int(axis) != 0 and int(dim) == constant_width
                ]
                if len(matching_axes) == 1:
                    target_axis = int(matching_axes[0])
            if target_axis is not None and len(tensor_shape) >= 1:
                batch_dim = int(tensor_shape[0])
                if batch_dim in {1, int(other_shape[0])}:
                    non_batch_non_singleton_axes = [
                        int(axis)
                        for axis, dim in enumerate(tensor_shape[1:], start=1)
                        if int(dim) > 1
                    ]
                    if len(non_batch_non_singleton_axes) == 1:
                        source_axis = int(non_batch_non_singleton_axes[0])
                        if int(tensor_shape[source_axis]) == constant_width:
                            other_non_target_axes = [
                                int(axis)
                                for axis, dim in enumerate(other_shape)
                                if int(axis) not in {0, int(target_axis)} and int(dim) > 1
                            ]
                            if len(other_non_target_axes) > 0:
                                reshape_dims = [1 for _ in other_shape]
                                reshape_dims[0] = int(batch_dim)
                                reshape_dims[int(target_axis)] = int(constant_width)
                                return f"{expr}.reshape({repr(reshape_dims)})"
        if len(tensor_shape) != 1 or len(other_shape) < 2:
            if len(tensor_shape) == len(other_shape) and len(tensor_shape) > 1:
                try:
                    np.broadcast_shapes(tuple(tensor_broadcast_shape), tuple(other_broadcast_shape))
                    return expr
                except Exception:
                    pass
                preferred_perm: Optional[Tuple[int, ...]] = None
                tensor_layout = normalize_logical_layout(tensor.logical_layout)
                if tensor_layout == "NCHW" and len(tensor_shape) == 4:
                    preferred_perm = (0, 2, 3, 1)
                elif tensor_layout == "NCDHW" and len(tensor_shape) == 5:
                    preferred_perm = (0, 2, 3, 4, 1)
                elif tensor_layout == "NCW" and len(tensor_shape) == 3:
                    preferred_perm = (0, 2, 1)
                if preferred_perm is not None:
                    preferred_shape = [int(tensor_shape[int(idx)]) for idx in preferred_perm]
                    preferred_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in preferred_shape]
                    try:
                        np.broadcast_shapes(tuple(preferred_broadcast_shape), tuple(other_broadcast_shape))
                        return f"{expr}.permute(*{repr(tuple(int(v) for v in preferred_perm))}).contiguous()"
                    except Exception:
                        pass
                import itertools

                for generic_perm in itertools.permutations(range(len(tensor_shape))):
                    if list(generic_perm) == list(range(len(tensor_shape))):
                        continue
                    permuted_shape = [int(tensor_shape[int(idx)]) for idx in generic_perm]
                    permuted_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in permuted_shape]
                    try:
                        np.broadcast_shapes(tuple(permuted_broadcast_shape), tuple(other_broadcast_shape))
                        return f"{expr}.permute(*{repr(tuple(int(v) for v in generic_perm))}).contiguous()"
                    except Exception:
                        continue
            return expr
        constant_width = int(tensor_shape[0])
        if constant_width <= 0:
            return expr
        target_axis: Optional[int] = None
        other_layout = normalize_logical_layout(other_tensor.logical_layout)
        if (
            is_channel_first_logical_layout(other_layout)
            and len(other_shape) >= 2
            and int(other_shape[1]) == constant_width
        ):
            target_axis = 1
        elif (
            is_channel_last_logical_layout(other_layout)
            and int(other_shape[-1]) == constant_width
        ):
            target_axis = len(other_shape) - 1
        else:
            matching_axes = [
                int(axis)
                for axis, dim in enumerate(other_shape)
                if int(axis) != 0 and int(dim) == constant_width
            ]
            if len(matching_axes) == 1:
                target_axis = int(matching_axes[0])
        if target_axis is None:
            return expr
        reshape_dims = [1 for _ in other_shape]
        reshape_dims[int(target_axis)] = constant_width
        if reshape_dims == tensor_shape:
            return expr
        return f"{expr}.reshape({repr(reshape_dims)})"

    runtime_shape_uncertain_tensors: Set[str] = set()

    def _binary_requires_runtime_alignment(lhs_name: str, rhs_name: str, output_name: str) -> bool:
        lhs_tensor = model_ir.tensors.get(str(lhs_name), None)
        rhs_tensor = model_ir.tensors.get(str(rhs_name), None)
        output_tensor = model_ir.tensors.get(str(output_name), None)
        if lhs_tensor is None or rhs_tensor is None:
            return False
        if (
            str(lhs_name) in runtime_shape_uncertain_tensors
            or str(rhs_name) in runtime_shape_uncertain_tensors
            or str(output_name) in runtime_shape_uncertain_tensors
        ):
            return True
        lhs_shape = [int(v) for v in list(lhs_tensor.shape)]
        rhs_shape = [int(v) for v in list(rhs_tensor.shape)]
        lhs_signature = (
            [int(v) for v in list(lhs_tensor.shape_signature)]
            if lhs_tensor.shape_signature is not None
            else list(lhs_shape)
        )
        rhs_signature = (
            [int(v) for v in list(rhs_tensor.shape_signature)]
            if rhs_tensor.shape_signature is not None
            else list(rhs_shape)
        )
        if len(lhs_shape) != len(rhs_shape):
            return False
        if lhs_shape == rhs_shape and lhs_signature != rhs_signature:
            return True
        if lhs_shape == rhs_shape:
            return False
        try:
            broadcast_shape = [int(v) for v in list(np.broadcast_shapes(tuple(lhs_shape), tuple(rhs_shape)))]
            if output_tensor is None:
                return False
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if broadcast_shape != output_shape:
                return True
            output_signature = (
                [int(v) for v in list(output_tensor.shape_signature)]
                if output_tensor.shape_signature is not None
                else list(output_shape)
            )
            return (
                lhs_signature != lhs_shape
                or rhs_signature != rhs_shape
                or output_signature != output_shape
            )
        except Exception:
            return True
        return bool(
            (isinstance(lhs_tensor.data, np.ndarray) and len(lhs_shape) > 1)
            or (isinstance(rhs_tensor.data, np.ndarray) and len(rhs_shape) > 1)
        )

    def _pad_literal_expr(tensor_name: str) -> Optional[str]:
        return _torch_pad_literal_for_constant_tensor(model_ir.tensors.get(str(tensor_name), None))

    def _scalar_literal_expr(tensor_name: str) -> Optional[str]:
        return _scalar_literal_for_constant_tensor(model_ir.tensors.get(str(tensor_name), None))

    def _activation_lines(var_name: str, fused: str) -> List[str]:
        key = str(fused).upper()
        if key in {"", "NONE"}:
            return []
        if key == "RELU":
            return [f"{var_name} = torch.relu({var_name})"]
        if key == "RELU6":
            return [f"{var_name} = torch.clamp({var_name}, min=0.0, max=6.0)"]
        if key == "RELU_N1_TO_1":
            return [f"{var_name} = torch.clamp({var_name}, min=-1.0, max=1.0)"]
        if key == "RELU_0_TO_1":
            return [f"{var_name} = torch.clamp({var_name}, min=0.0, max=1.0)"]
        if key == "TANH":
            return [f"{var_name} = torch.tanh({var_name})"]
        return [f"{var_name} = _apply_fused_activation({var_name}, {fused!r})"]

    def _is_sequential_single_input_graph() -> bool:
        if len(model_ir.inputs) != 1 or len(model_ir.outputs) != 1:
            return False
        current_name = str(model_ir.inputs[0])
        for op in model_ir.operators:
            if len(op.outputs) != 1:
                return False
            data_input_index = 2 if str(op.op_type) in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} else 0
            if len(op.inputs) <= data_input_index:
                return False
            if str(op.inputs[data_input_index]) != current_name:
                return False
            for input_index, input_name in enumerate(op.inputs):
                if int(input_index) == int(data_input_index) or str(input_name) == "":
                    continue
                input_tensor = model_ir.tensors.get(str(input_name), None)
                if input_tensor is None or not isinstance(input_tensor.data, np.ndarray):
                    return False
            current_name = str(op.outputs[0])
        return current_name == str(model_ir.outputs[0])

    if _is_sequential_single_input_graph():
        tensor_var_names[str(model_ir.inputs[0])] = "x"
        for op in model_ir.operators:
            tensor_var_names[str(op.outputs[0])] = "x"

    def _is_channel_last_layout(logical_layout: Any) -> bool:
        return str(logical_layout).upper() in {"NWC", "NHWC", "NDHWC"}

    def _can_emit_direct_module_call(op: OperatorIR) -> bool:
        op_type = str(op.op_type)
        if op_type not in {"CONV_2D", "DEPTHWISE_CONV_2D", "CONV_3D"}:
            return False
        if len(op.outputs) != 1:
            return False
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
        if input_tensor is None or output_tensor is None:
            return False
        if _is_channel_last_layout(input_tensor.logical_layout) or _is_channel_last_layout(output_tensor.logical_layout):
            return False
        expected_rank = 5 if op_type == "CONV_3D" else 4
        if len(input_tensor.shape) != expected_rank or len(output_tensor.shape) != expected_rank:
            return False
        if int(input_tensor.shape[1]) <= 0 or int(output_tensor.shape[1]) <= 0:
            return False
        return True

    runtime_imports: Set[str] = {
        "resolve_named_input_value",
        "load_generated_weights",
    }
    forward_lines: List[str] = []
    for op_index, op in enumerate(model_ir.operators):
        if int(op_index) in skipped_op_indices:
            continue
        op_type = str(op.op_type)
        outputs = [str(v) for v in op.outputs]
        output_vars = [tensor_var_names[str(name)] for name in outputs]
        output_target_shape = _target_shape_literal(outputs[0]) if len(outputs) == 1 else "None"
        affine_layer_norm_spec = affine_layer_norm_specs.get(int(op_index), None)
        if affine_layer_norm_spec is not None:
            forward_lines.append(
                f"{output_vars[0]} = self.{affine_layer_norm_spec['attr_name']}({_tensor_expr(str(affine_layer_norm_spec['input_name']))})"
            )
            continue
        if op_type in _DIRECT_CODEGEN_MODULE_OP_TYPES:
            attr_name = op_module_attr_names[int(op_index)]
            fused_module_spec = fused_module_specs.get(int(op_index), None)
            if fused_module_spec is not None:
                output_name = str(fused_module_spec["output_name"])
                output_var = tensor_var_names[output_name]
                output_target_shape = _target_shape_literal(output_name)
                fused_input_expr = _tensor_expr(str(fused_module_spec["input_name"]))
                input_pre_permute = fused_module_spec.get("input_pre_permute", None)
                if isinstance(input_pre_permute, list) and len(input_pre_permute) == 4:
                    fused_input_expr = (
                        f"{fused_input_expr}.permute({', '.join(str(int(v)) for v in input_pre_permute)}).contiguous()"
                    )
                forward_lines.append(
                    f"{output_var} = _align_tensor_to_target_shape(self.{attr_name}({fused_input_expr}), {output_target_shape})"
                )
                runtime_imports.add("_align_tensor_to_target_shape")
                continue
            fused = str(op.options.get("fusedActivationFunction", "NONE"))
            if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}:
                conv_input_expr = _tensor_expr(str(op.inputs[0]))
                input_pre_permute = _conv2d_input_pre_permute(
                    _tensor_shape_list(str(op.inputs[0])),
                    _tensor_shape_list(outputs[0]),
                    _tensor_shape_list(str(op.inputs[1])),
                    op.options,
                    depthwise=(op_type == "DEPTHWISE_CONV_2D"),
                )
                if input_pre_permute is not None:
                    conv_input_expr = (
                        f"{conv_input_expr}.permute({', '.join(str(int(v)) for v in input_pre_permute)}).contiguous()"
                    )
                conv_pad_arg = conv_module_pad_specs.get(int(op_index), None)
                if conv_pad_arg is not None:
                    conv_input_expr = f"F.pad({conv_input_expr}, {repr(conv_pad_arg)}, mode='constant', value=0.0)"
                if _can_emit_direct_module_call(op):
                    forward_lines.append(f"{output_vars[0]} = self.{attr_name}({conv_input_expr})")
                else:
                    runtime_imports.add("_apply_module_conv2d")
                    forward_lines.append(
                        f"{output_vars[0]} = _apply_module_conv2d(self.{attr_name}, {conv_input_expr}, target_shape={output_target_shape}, fused='NONE')"
                    )
                forward_lines.extend(_activation_lines(output_vars[0], fused))
            elif op_type == "TRANSPOSE_CONV":
                runtime_imports.add("_apply_module_transpose_conv2d")
                output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                fallback_shape = (
                    [int(v) for v in np.asarray(output_shape_tensor.data).reshape(-1).tolist()]
                    if output_shape_tensor is not None and isinstance(output_shape_tensor.data, np.ndarray)
                    else [int(v) for v in list(model_ir.tensors[outputs[0]].shape)]
                )
                forward_lines.append(
                    f"{output_vars[0]} = _apply_module_transpose_conv2d(self.{attr_name}, {_tensor_expr(str(op.inputs[2]))}, target_shape={output_target_shape}, fallback_shape={repr(fallback_shape)}, fused='NONE')"
                )
                forward_lines.extend(_activation_lines(output_vars[0], fused))
            elif op_type == "CONV_3D":
                if _can_emit_direct_module_call(op):
                    forward_lines.append(f"{output_vars[0]} = self.{attr_name}({_tensor_expr(str(op.inputs[0]))})")
                else:
                    runtime_imports.add("_apply_module_conv3d")
                    forward_lines.append(
                        f"{output_vars[0]} = _apply_module_conv3d(self.{attr_name}, {_tensor_expr(str(op.inputs[0]))}, target_shape={output_target_shape}, fused='NONE')"
                    )
                forward_lines.extend(_activation_lines(output_vars[0], fused))
            elif op_type == "CONV_3D_TRANSPOSE":
                runtime_imports.add("_apply_module_transpose_conv3d")
                output_shape_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                fallback_shape = (
                    [int(v) for v in np.asarray(output_shape_tensor.data).reshape(-1).tolist()]
                    if output_shape_tensor is not None and isinstance(output_shape_tensor.data, np.ndarray)
                    else [int(v) for v in list(model_ir.tensors[outputs[0]].shape)]
                )
                forward_lines.append(
                    f"{output_vars[0]} = _apply_module_transpose_conv3d(self.{attr_name}, {_tensor_expr(str(op.inputs[2]))}, target_shape={output_target_shape}, fallback_shape={repr(fallback_shape)}, fused='NONE')"
                )
                forward_lines.extend(_activation_lines(output_vars[0], fused))
            elif op_type == "FULLY_CONNECTED":
                forward_lines.append(f"{output_vars[0]} = self.{attr_name}({_tensor_expr(str(op.inputs[0]))})")
                forward_lines.extend(_activation_lines(output_vars[0], fused))
            elif op_type == "PRELU":
                expr = f"self.{attr_name}({_tensor_expr(str(op.inputs[0]))})"
                inferred_shape = _tensor_shape_list(str(op.inputs[0]))
                if _should_skip_align_for_shape_preserving_unary(str(op.inputs[0]), outputs[0]):
                    forward_lines.append(f"{output_vars[0]} = {expr}")
                else:
                    forward_lines.append(
                        f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=expr, inferred_shape=inferred_shape)}"
                    )
            continue
        if op_type in _DIRECT_CODEGEN_BINARY_FUNCTIONS:
            fn_name = _DIRECT_CODEGEN_BINARY_FUNCTIONS[op_type]
            fused = str(op.options.get("fusedActivationFunction", "NONE"))
            lhs_name = str(op.inputs[0])
            rhs_name = str(op.inputs[1])
            if _binary_requires_runtime_alignment(lhs_name, rhs_name, outputs[0]):
                lhs_uncertain = lhs_name in runtime_shape_uncertain_tensors
                rhs_uncertain = rhs_name in runtime_shape_uncertain_tensors
                lhs_var = f"_binary_lhs_{op_index}"
                rhs_var = f"_binary_rhs_{op_index}"
                if lhs_uncertain ^ rhs_uncertain:
                    runtime_imports.add("_align_binary_inputs_to_anchor")
                    if lhs_uncertain:
                        forward_lines.append(
                            f"{lhs_var}, {rhs_var} = _align_binary_inputs_to_anchor({_binary_operand_expr(lhs_name, rhs_name)}, {_binary_operand_expr(rhs_name, lhs_name)}, {output_target_shape})"
                        )
                    else:
                        forward_lines.append(
                            f"{rhs_var}, {lhs_var} = _align_binary_inputs_to_anchor({_binary_operand_expr(rhs_name, lhs_name)}, {_binary_operand_expr(lhs_name, rhs_name)}, {output_target_shape})"
                        )
                else:
                    runtime_imports.add("_align_binary_inputs")
                    forward_lines.append(
                        f"{lhs_var}, {rhs_var} = _align_binary_inputs({_binary_operand_expr(lhs_name, rhs_name)}, {_binary_operand_expr(rhs_name, lhs_name)}, {output_target_shape})"
                    )
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'{fn_name}({lhs_var}, {rhs_var})', inferred_shape=None)}"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'{fn_name}({_binary_operand_expr(lhs_name, rhs_name)}, {_binary_operand_expr(rhs_name, lhs_name)})', inferred_shape=None)}"
                )
            forward_lines.extend(_activation_lines(output_vars[0], fused))
            continue
        if op_type in _DIRECT_CODEGEN_UNARY_EXPRESSIONS:
            template = _DIRECT_CODEGEN_UNARY_EXPRESSIONS[op_type]
            if op_type == "LEAKY_RELU":
                expr = template.format(x=_tensor_expr(str(op.inputs[0])), alpha=float(op.options.get("alpha", 0.2)))
            else:
                expr = template.format(x=_tensor_expr(str(op.inputs[0])))
            inferred_shape = _tensor_shape_list(str(op.inputs[0]))
            if _should_skip_align_for_shape_preserving_unary(str(op.inputs[0]), outputs[0]):
                forward_lines.append(f"{output_vars[0]} = {expr}")
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=expr, inferred_shape=inferred_shape)}"
                )
            continue
        if op_type == "GATHER":
            axis = int(op.options.get("axis", 0))
            batch_dims = int(op.options.get("batchDims", 0))
            params_expr = _tensor_expr(str(op.inputs[0]))
            if _should_elide_crd_to_dcr_gather_for_depth_to_space(
                model_ir=model_ir,
                params_name=str(op.inputs[0]),
                indices_name=str(op.inputs[1]),
                output_name=str(outputs[0]),
                axis=axis,
                batch_dims=batch_dims,
            ):
                inferred_shape = _tensor_shape_list(str(op.inputs[0]))
                if _should_skip_align_for_shape_preserving_unary(str(op.inputs[0]), outputs[0]):
                    forward_lines.append(f"{output_vars[0]} = {params_expr}")
                else:
                    forward_lines.append(
                        f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=params_expr, inferred_shape=inferred_shape)}"
                    )
                continue
            gather_pre_perm = _gather_input_pre_permute(
                params_name=str(op.inputs[0]),
                output_name=str(outputs[0]),
                axis=axis,
                batch_dims=batch_dims,
            )
            gather_input_rank = len(model_ir.tensors[str(op.inputs[0])].shape)
            if gather_pre_perm is not None:
                params_expr = f"{params_expr}.permute({', '.join(str(int(v)) for v in gather_pre_perm)}).contiguous()"
                gather_input_rank = int(len(gather_pre_perm))
            indices_expr = _tensor_expr(str(op.inputs[1]))
            indices_name = str(op.inputs[1])
            direct_indices_values = _constant_int_list(model_ir.tensors.get(indices_name, None))
            direct_gather_expr = None
            if direct_indices_values is not None:
                direct_gather_expr = _direct_gather_expr(
                    params_expr=params_expr,
                    indices_values=direct_indices_values,
                    indices_shape=model_ir.tensors.get(indices_name, None).shape if model_ir.tensors.get(indices_name, None) is not None else None,
                    axis=axis,
                    batch_dims=batch_dims,
                    input_rank=gather_input_rank,
                )
            if direct_gather_expr is None:
                direct_gather_expr = _direct_dynamic_gather_expr(
                    params_expr=params_expr,
                    indices_expr=indices_expr,
                    axis=axis,
                    batch_dims=batch_dims,
                    input_rank=gather_input_rank,
                    indices_name=indices_name,
                )
            if direct_gather_expr is not None:
                forward_lines.append(f"{output_vars[0]} = {direct_gather_expr}")
            else:
                runtime_imports.add("_apply_gather")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_gather({params_expr}, {indices_expr}, axis={axis}, batch_dims={batch_dims}, target_shape={output_target_shape}, indices_name={indices_name!r})"
                )
            continue
        if op_type == "GATHER_ND":
            params_expr = _tensor_expr(str(op.inputs[0]))
            indices_expr = _tensor_expr(str(op.inputs[1]))
            gather_elements_axis_hint = None
            gather_elements_indices_name = None
            if str(op.inputs[1]).endswith("_gather_elements_coords"):
                producer_op_idx = producer_index.get(str(op.inputs[1]), None)
                if producer_op_idx is not None:
                    producer_op = model_ir.operators[int(producer_op_idx)]
                    if str(producer_op.op_type) == "CONCATENATION":
                        gather_elements_axis_hint = next(
                            (
                                idx for idx, name in enumerate(producer_op.inputs)
                                if str(name).endswith("_gather_elements_axis_coord")
                            ),
                            None,
                        )
                candidate_indices_name = str(op.inputs[1]).replace("_gather_elements_coords", "_gather_elements_indices_i32")
                if candidate_indices_name in model_ir.tensors:
                    gather_elements_indices_name = candidate_indices_name
            if gather_elements_axis_hint is not None and gather_elements_indices_name is not None:
                params_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                params_rank = 0 if params_tensor is None else len(list(params_tensor.shape))
                gather_axis = int(gather_elements_axis_hint)
                gather_axis_cl = None
                params_layout = (
                    LOGICAL_LAYOUT_UNKNOWN
                    if params_tensor is None
                    else normalize_logical_layout(params_tensor.logical_layout)
                )
                if params_rank in {3, 4, 5} and is_channel_first_logical_layout(params_layout):
                    perm_cf_to_cl = logical_layout_permutation(
                        source_layout=channel_first_logical_layout(params_rank),
                        target_layout=channel_last_logical_layout(params_rank),
                    )
                    if perm_cf_to_cl is not None and gather_axis in perm_cf_to_cl:
                        gather_axis_cl = int(perm_cf_to_cl.index(gather_axis))
                runtime_imports.add("_apply_gather_elements")
                runtime_shape_uncertain_tensors.add(outputs[0])
                forward_lines.append(
                    f"{output_vars[0]} = _apply_gather_elements({params_expr}, {_tensor_expr(gather_elements_indices_name)}, axis_hint={gather_axis}, axis_hint_cl={repr(gather_axis_cl)}, target_shape={output_target_shape})"
                )
                continue
            gather_nd_indices_var = f"_gather_nd_indices_{op_index}"
            forward_lines.append(f"{gather_nd_indices_var} = {indices_expr}.to(dtype=torch.int64)")
            forward_lines.append(
                f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'{params_expr}[tuple({gather_nd_indices_var}[..., i] for i in range({gather_nd_indices_var}.shape[-1]))]', inferred_shape=_infer_gather_nd_shape(_tensor_shape_list(str(op.inputs[0])), str(op.inputs[1])))}"
            )
            continue
        if op_type == "CAST":
            out_dtype = str(op.options.get("outDataType", "FLOAT32"))
            forward_lines.append(
                f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}.to(dtype={_torch_dtype_literal(out_dtype)})"
            )
            continue
        if op_type == "LOCAL_RESPONSE_NORMALIZATION":
            input_expr = _tensor_expr(str(op.inputs[0]))
            radius = int(op.options.get("radius", 0))
            size = max(1, int(radius) * 2 + 1)
            alpha = float(op.options.get("alpha", 1.0))
            beta = float(op.options.get("beta", 0.5))
            bias = float(op.options.get("bias", 1.0))
            inferred_shape = _tensor_shape_list(str(op.inputs[0]))
            forward_lines.append(
                f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'F.local_response_norm({input_expr}, size={size}, alpha={alpha}, beta={beta}, k={bias})', inferred_shape=inferred_shape)}"
            )
            continue
        if op_type == "RESHAPE":
            runtime_imports.add("_shape_list")
            reshape_input_expr = _tensor_expr(str(op.inputs[0]))
            reshape_input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            reshape_output_tensor = model_ir.tensors.get(str(outputs[0]), None)
            reshape_input_shape = None if reshape_input_tensor is None else [int(v) for v in list(reshape_input_tensor.shape)]
            reshape_output_shape = None if reshape_output_tensor is None else [int(v) for v in list(reshape_output_tensor.shape)]
            reshape_output_preferred_shape = _preferred_reshape_target_values(reshape_output_tensor)
            reshape_input_layout = None if reshape_input_tensor is None else str(reshape_input_tensor.logical_layout)
            reshape_output_layout = None if reshape_output_tensor is None else str(reshape_output_tensor.logical_layout)
            reshape_special_plan = _reshape_special_layout_plan(
                input_shape=reshape_input_shape,
                output_shape=reshape_output_shape,
                input_layout=reshape_input_layout,
                output_layout=reshape_output_layout,
            )
            reshape_pre_perm = _reshape_preserves_channel_last_sequence(
                reshape_input_shape,
                reshape_output_shape,
                reshape_input_layout,
            )
            if reshape_special_plan is not None and reshape_special_plan.get("pre_perm", None) is not None:
                reshape_pre_perm = list(reshape_special_plan["pre_perm"])
            reshape_feature_last_target = _reshape_prefers_feature_last_for_adjx_batch_matmul(
                str(op.inputs[0]),
                str(outputs[0]),
            )
            if reshape_feature_last_target is not None:
                reshape_pre_perm = list(reshape_feature_last_target[0])
            if reshape_pre_perm is not None:
                reshape_input_expr = f"{reshape_input_expr}.permute({', '.join(str(int(v)) for v in reshape_pre_perm)}).contiguous()"
            if reshape_feature_last_target is not None:
                shape_expr = repr([int(v) for v in list(reshape_feature_last_target[1])])
            elif reshape_special_plan is not None and reshape_special_plan.get("reshape_shape", None) is not None:
                shape_expr = repr([int(v) for v in list(reshape_special_plan["reshape_shape"])])
            elif reshape_output_preferred_shape is not None:
                preferred_shape_values = [int(v) for v in list(reshape_output_preferred_shape)]
                if all(int(v) > 0 for v in preferred_shape_values):
                    shape_expr = repr(preferred_shape_values)
                elif (
                    preferred_shape_values.count(-1) <= 1
                    and all(int(v) == -1 or int(v) > 0 for v in preferred_shape_values)
                ):
                    shape_expr = repr(preferred_shape_values)
                else:
                    runtime_imports.add("_resolve_reshape_shape")
                    shape_expr = (
                        f"_resolve_reshape_shape({repr(preferred_shape_values)}, "
                        f"{reshape_input_expr}, allow_zero={bool(op.options.get('allowZero', False))})"
                    )
            elif len(op.inputs) >= 2:
                runtime_imports.add("_resolve_reshape_shape")
                const_shape_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
                if const_shape_values is not None:
                    shape_expr = (
                        f"_resolve_reshape_shape({repr([int(v) for v in list(const_shape_values)])}, "
                        f"{reshape_input_expr}, allow_zero={bool(op.options.get('allowZero', False))})"
                    )
                else:
                    shape_expr = (
                        f"_resolve_reshape_shape({_tensor_expr(str(op.inputs[1]))}, "
                        f"{reshape_input_expr}, allow_zero={bool(op.options.get('allowZero', False))})"
                    )
            else:
                runtime_imports.add("_resolve_reshape_shape")
                raw_new_shape = op.options.get("onnxRawNewShape", op.options.get("newShape", []))
                shape_expr = (
                    f"_resolve_reshape_shape({repr([int(v) for v in list(raw_new_shape)])}, "
                    f"{reshape_input_expr}, allow_zero={bool(op.options.get('allowZero', False))})"
                )
            raw_new_shape = op.options.get("onnxRawNewShape", op.options.get("newShape", None))
            reshape_is_gather_elements_axis_coord = bool(
                str(op.inputs[0]).endswith("_gather_elements_indices_i32")
                and str(outputs[0]).endswith("_gather_elements_axis_coord")
                and reshape_input_shape is not None
                and isinstance(raw_new_shape, list)
                and [int(v) for v in list(raw_new_shape)] == [*reshape_input_shape, 1]
            )
            if reshape_is_gather_elements_axis_coord:
                forward_lines.append(f"{output_vars[0]} = torch.unsqueeze({reshape_input_expr}, dim=-1)")
            else:
                forward_lines.append(
                    f"{output_vars[0]} = torch.reshape({reshape_input_expr}, [int(v) for v in {shape_expr}])"
                )
            if reshape_special_plan is not None and reshape_special_plan.get("post_perm", None) is not None:
                post_perm = [int(v) for v in list(reshape_special_plan["post_perm"])]
                forward_lines.append(
                    f"{output_vars[0]} = {output_vars[0]}.permute({', '.join(str(v) for v in post_perm)}).contiguous()"
                )
            continue
        if op_type == "TRANSPOSE":
            if _is_reshape_only_residual_layout_bridge_transpose(
                model_ir=model_ir,
                op=op,
                consumers=consumer_index,
            ) or _is_inconsistent_standard_layout_transpose(
                input_tensor=model_ir.tensors.get(str(op.inputs[0]), None) if len(op.inputs) >= 1 else None,
                output_tensor=model_ir.tensors.get(outputs[0], None) if len(outputs) == 1 else None,
                perm=_read_transpose_perm(model_ir, op),
            ):
                forward_lines.append(
                    f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}"
                )
                continue
            runtime_imports.add("_shape_list")
            runtime_imports.add("_torch_permute")
            if len(op.inputs) >= 2:
                const_perm_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
                if const_perm_values is not None:
                    perm_expr = repr([int(v) for v in list(const_perm_values)])
                else:
                    perm_expr = f"_shape_list({_tensor_expr(str(op.inputs[1]))})"
            else:
                perm_expr = repr([int(v) for v in list(op.options.get('perm', []))])
            forward_lines.append(
                f"{output_vars[0]} = _torch_permute({_tensor_expr(str(op.inputs[0]))}, {perm_expr})"
            )
            continue
        if op_type == "EXPAND_DIMS":
            axis_expr = (
                f"_coerce_scalar_axis({_tensor_expr(str(op.inputs[1]))}, device={_tensor_expr(str(op.inputs[0]))}.device)"
                if len(op.inputs) >= 2
                else repr(int(op.options.get("axis", 0)))
            )
            if len(op.inputs) >= 2:
                runtime_imports.add("_coerce_scalar_axis")
            forward_lines.append(
                f"{output_vars[0]} = torch.unsqueeze({_tensor_expr(str(op.inputs[0]))}, dim={axis_expr})"
            )
            continue
        if op_type == "SQUEEZE":
            squeeze_dims = [int(v) for v in list(op.options.get("squeezeDims", []))]
            if len(squeeze_dims) == 0:
                forward_lines.append(f"{output_vars[0]} = torch.squeeze({_tensor_expr(str(op.inputs[0]))})")
            else:
                runtime_imports.add("_normalize_dim")
                forward_lines.append(f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}")
                for axis in sorted(squeeze_dims, reverse=True):
                    forward_lines.append(
                        f"{output_vars[0]} = torch.squeeze({output_vars[0]}, dim=_normalize_dim({int(axis)}, {output_vars[0]}.ndim))"
                    )
            continue
        if op_type == "CONCATENATION":
            axis = int(op.options.get("axis", 0))
            gather_elements_axis_coord_input_index = next(
                (idx for idx, name in enumerate(op.inputs) if str(name).endswith("_gather_elements_axis_coord")),
                None,
            )
            gather_elements_coords_concat = bool(
                str(outputs[0]).endswith("_gather_elements_coords")
                and gather_elements_axis_coord_input_index is not None
                and axis == len(op.inputs)
            )
            if gather_elements_coords_concat:
                axis_coord_expr = _tensor_expr(str(op.inputs[gather_elements_axis_coord_input_index]))
                coord_shape_var = f"_gather_elements_coords_shape_{op_index}"
                coord_vars: list[str] = []
                forward_lines.append(
                    f"{coord_shape_var} = [int(v) for v in list({axis_coord_expr}.shape[:-1])]"
                )
                for dim_index in range(len(op.inputs)):
                    if dim_index == gather_elements_axis_coord_input_index:
                        coord_vars.append(axis_coord_expr)
                        continue
                    coord_var = f"_gather_elements_coord_{op_index}_{dim_index}"
                    view_shape_var = f"_gather_elements_coord_view_shape_{op_index}_{dim_index}"
                    forward_lines.append(f"{view_shape_var} = [1] * {len(op.inputs)}")
                    forward_lines.append(
                        f"{view_shape_var}[{dim_index}] = {coord_shape_var}[{dim_index}]"
                    )
                    forward_lines.append(
                        f"{coord_var} = torch.arange({coord_shape_var}[{dim_index}], dtype={axis_coord_expr}.dtype, device={axis_coord_expr}.device).reshape(*{view_shape_var}, 1).expand(*{coord_shape_var}, 1)"
                    )
                    coord_vars.append(coord_var)
                forward_lines.append(
                    f"{output_vars[0]} = torch.cat([{', '.join(coord_vars)}], dim={len(op.inputs)})"
                )
            else:
                inputs_expr = ", ".join(_tensor_expr(str(name)) for name in op.inputs)
                runtime_imports.add("_apply_concat")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_concat([{inputs_expr}], axis={axis}, target_shape={output_target_shape}, fused={str(op.options.get('fusedActivationFunction', 'NONE'))!r})"
                )
            continue
        if op_type == "PACK":
            axis = int(op.options.get("axis", 0))
            inputs_expr = ", ".join(_tensor_expr(str(name)) for name in op.inputs)
            forward_lines.append(
                f"{output_vars[0]} = torch.stack([{inputs_expr}], dim={axis})"
            )
            continue
        if op_type == "UNPACK":
            runtime_imports.add("_normalize_dim")
            axis = int(op.options.get("axis", 0))
            forward_lines.append(
                f"{', '.join(output_vars)} = list(torch.unbind({_tensor_expr(str(op.inputs[0]))}, dim=_normalize_dim({axis}, {_tensor_expr(str(op.inputs[0]))}.ndim)))"
            )
            continue
        if op_type == "SPLIT":
            runtime_imports.add("_normalize_dim")
            data_expr = _tensor_expr(str(op.inputs[-1]))
            if len(op.inputs) >= 2:
                runtime_imports.add("_coerce_scalar_axis")
                axis_expr = (
                    f"_coerce_scalar_axis({_tensor_expr(str(op.inputs[0]))}, device={data_expr}.device)"
                )
            else:
                axis_expr = repr(int(op.options.get("axis", 0)))
            sections = int(op.options.get("numSplits", len(outputs)))
            forward_lines.append(
                f"{', '.join(output_vars)} = list(torch.tensor_split({data_expr}, {sections}, dim=_normalize_dim({axis_expr}, {data_expr}.ndim)))"
            )
            continue
        if op_type == "SLICE":
            direct_slice_expr = _direct_slice_expr(
                x_expr=_tensor_expr(str(op.inputs[0])),
                begin_values=_constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) or [],
                size_values=_constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None)) or [],
                input_rank=len(model_ir.tensors[str(op.inputs[0])].shape),
                input_shape=model_ir.tensors[str(op.inputs[0])].shape,
            )
            if direct_slice_expr is not None:
                forward_lines.append(f"{output_vars[0]} = {direct_slice_expr}")
            else:
                runtime_imports.add("_apply_slice")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_slice({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))}, target_shape={output_target_shape})"
                )
            continue
        if op_type == "STRIDED_SLICE":
            options = dict(op.options)
            direct_strided_slice_expr = _direct_strided_slice_expr(
                x_expr=_tensor_expr(str(op.inputs[0])),
                begin_values=_constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None)) or [],
                end_values=_constant_int_list(model_ir.tensors.get(str(op.inputs[2]), None)) or [],
                stride_values=_constant_int_list(model_ir.tensors.get(str(op.inputs[3]), None)) or [],
                begin_mask=int(options.get("beginMask", 0)),
                end_mask=int(options.get("endMask", 0)),
                input_rank=len(model_ir.tensors[str(op.inputs[0])].shape),
            )
            if direct_strided_slice_expr is not None:
                forward_lines.append(f"{output_vars[0]} = {direct_strided_slice_expr}")
            else:
                runtime_imports.add("_apply_strided_slice")
                forward_lines.append(
                    f"{output_vars[0]} = _apply_strided_slice({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))}, {_tensor_expr(str(op.inputs[3]))}, begin_mask={int(options.get('beginMask', 0))}, end_mask={int(options.get('endMask', 0))}, target_shape={output_target_shape})"
                )
            continue
        if op_type == "SHAPE":
            out_dtype = str(op.options.get("outType", "INT32"))
            forward_lines.append(
                f"{output_vars[0]} = torch.tensor(list({_tensor_expr(str(op.inputs[0]))}.shape), dtype={_torch_dtype_literal(out_dtype)}, device={_tensor_expr(str(op.inputs[0]))}.device)"
            )
            continue
        if op_type == "FILL":
            runtime_imports.add("_shape_list")
            forward_lines.append(
                f"{output_vars[0]} = torch.full([int(v) for v in _shape_list({_tensor_expr(str(op.inputs[0]))})], {_tensor_expr(str(op.inputs[1]))}.reshape(-1)[0].item(), dtype={_tensor_expr(str(op.inputs[1]))}.dtype, device={_tensor_expr(str(op.inputs[1]))}.device)"
            )
            continue
        if op_type == "SCATTER_ND":
            runtime_imports.add("_apply_scatter_nd")
            forward_lines.append(
                f"{output_vars[0]} = _apply_scatter_nd({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))}, {output_target_shape})"
            )
            continue
        if op_type == "RANGE":
            start_expr = _tensor_expr(str(op.inputs[0]))
            limit_expr = _tensor_expr(str(op.inputs[1]))
            delta_expr = _tensor_expr(str(op.inputs[2]))
            forward_lines.append(
                f"{output_vars[0]} = torch.arange(start={start_expr}.reshape(-1)[0].item(), end={limit_expr}.reshape(-1)[0].item(), step={delta_expr}.reshape(-1)[0].item(), device={start_expr}.device, dtype={start_expr}.dtype)"
            )
            continue
        if op_type == "DEPTH_TO_SPACE":
            block_size = int(op.options.get("blockSize", 1))
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            input_shape = _tensor_shape_list(str(op.inputs[0]))
            output_shape = _tensor_shape_list(outputs[0])
            input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            output_layout = normalize_logical_layout(output_tensor.logical_layout) if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            use_channel_last = bool(
                is_channel_last_logical_layout(input_layout)
                or is_channel_last_logical_layout(output_layout)
            )
            inferred_channel_last = None
            if input_shape is not None and output_shape is not None:
                inferred_channel_last = _should_emit_channel_last_depth_to_space(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    block_size=block_size,
                )
            if inferred_channel_last is True:
                use_channel_last = True
            elif inferred_channel_last is False:
                use_channel_last = False
            input_expr = _tensor_expr(str(op.inputs[0]))
            if use_channel_last:
                forward_lines.append(f"_depth_to_space_x_{op_index} = {input_expr}")
                forward_lines.append(
                    f"_depth_to_space_n_{op_index}, _depth_to_space_h_{op_index}, _depth_to_space_w_{op_index}, _depth_to_space_c_{op_index} = _depth_to_space_x_{op_index}.shape"
                )
                forward_lines.append(
                    f"{output_vars[0]} = _depth_to_space_x_{op_index}.reshape(_depth_to_space_n_{op_index}, _depth_to_space_h_{op_index}, _depth_to_space_w_{op_index}, {block_size}, {block_size}, _depth_to_space_c_{op_index} // {block_size * block_size}).permute(0, 1, 3, 2, 4, 5).reshape(_depth_to_space_n_{op_index}, _depth_to_space_h_{op_index} * {block_size}, _depth_to_space_w_{op_index} * {block_size}, _depth_to_space_c_{op_index} // {block_size * block_size})"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'F.pixel_shuffle({input_expr}, {block_size})', inferred_shape=_tensor_shape_list(outputs[0]))}"
                )
            continue
        if op_type == "SPACE_TO_DEPTH":
            block_size = int(op.options.get("blockSize", 1))
            input_expr = _tensor_expr(str(op.inputs[0]))
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
            input_shape = _tensor_shape_list(str(op.inputs[0]))
            output_shape = _tensor_shape_list(outputs[0])
            input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            output_layout = normalize_logical_layout(output_tensor.logical_layout) if output_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
            use_channel_last = bool(
                is_channel_last_logical_layout(input_layout)
                or is_channel_last_logical_layout(output_layout)
            )
            inferred_channel_last = None
            if input_shape is not None and output_shape is not None:
                inferred_channel_last = _should_emit_channel_last_space_to_depth(
                    input_shape=input_shape,
                    output_shape=output_shape,
                    block_size=block_size,
                )
            if inferred_channel_last is True:
                use_channel_last = True
            elif inferred_channel_last is False:
                use_channel_last = False
            forward_lines.append(f"_space_to_depth_x_{op_index} = {input_expr}")
            if use_channel_last:
                forward_lines.append(
                    f"_space_to_depth_n_{op_index}, _space_to_depth_h_{op_index}, _space_to_depth_w_{op_index}, _space_to_depth_c_{op_index} = _space_to_depth_x_{op_index}.shape"
                )
                forward_lines.append(
                    f"{output_vars[0]} = _space_to_depth_x_{op_index}.reshape(_space_to_depth_n_{op_index}, _space_to_depth_h_{op_index} // {block_size}, {block_size}, _space_to_depth_w_{op_index} // {block_size}, {block_size}, _space_to_depth_c_{op_index}).permute(0, 1, 3, 2, 4, 5).reshape(_space_to_depth_n_{op_index}, _space_to_depth_h_{op_index} // {block_size}, _space_to_depth_w_{op_index} // {block_size}, _space_to_depth_c_{op_index} * {block_size * block_size})"
                )
            else:
                forward_lines.append(
                    f"_space_to_depth_n_{op_index}, _space_to_depth_c_{op_index}, _space_to_depth_h_{op_index}, _space_to_depth_w_{op_index} = _space_to_depth_x_{op_index}.shape"
                )
                forward_lines.append(
                    f"{output_vars[0]} = _space_to_depth_x_{op_index}.reshape(_space_to_depth_n_{op_index}, _space_to_depth_c_{op_index}, _space_to_depth_h_{op_index} // {block_size}, {block_size}, _space_to_depth_w_{op_index} // {block_size}, {block_size}).permute(0, 1, 3, 5, 2, 4).reshape(_space_to_depth_n_{op_index}, _space_to_depth_c_{op_index} * {block_size * block_size}, _space_to_depth_h_{op_index} // {block_size}, _space_to_depth_w_{op_index} // {block_size})"
                )
            continue
        if op_type == "SOFTMAX":
            runtime_imports.add("_apply_softmax")
            axis = op.options.get("axis", None)
            if axis is None and len(op.inputs) > 0:
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                if input_tensor is not None:
                    input_layout = normalize_logical_layout(input_tensor.logical_layout)
                    if is_channel_first_logical_layout(input_layout) and len(list(input_tensor.shape)) >= 2:
                        axis = 1
            axis_expr = repr(int(axis)) if axis is not None else "None"
            beta = float(op.options.get("beta", 1.0))
            forward_lines.append(
                f"{output_vars[0]} = _apply_softmax({_tensor_expr(str(op.inputs[0]))}, axis={axis_expr}, beta={beta}, target_shape={output_target_shape})"
            )
            continue
        if op_type in {"ARG_MAX", "ARG_MIN"}:
            runtime_imports.add("_normalize_dim")
            input_expr = _tensor_expr(str(op.inputs[0]))
            if len(op.inputs) >= 2:
                runtime_imports.add("_coerce_scalar_axis")
                axis_expr = f"_coerce_scalar_axis({_tensor_expr(str(op.inputs[1]))}, device={input_expr}.device)"
            else:
                axis_expr = repr(int(op.options.get("axis", 0)))
            input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
            output_tensor = model_ir.tensors.get(outputs[0], None)
            keep_dims = bool(op.options.get("keepDims", False))
            if input_tensor is not None and output_tensor is not None:
                keep_dims = len(list(output_tensor.shape)) == len(list(input_tensor.shape))
            reduce_fn = "torch.argmax" if op_type == "ARG_MAX" else "torch.argmin"
            output_dtype = "INT64" if output_tensor is None else str(output_tensor.dtype)
            forward_lines.append(
                f"{output_vars[0]} = {reduce_fn}({input_expr}, dim=_normalize_dim({axis_expr}, {input_expr}.ndim), keepdim={keep_dims}).to(dtype={_torch_dtype_literal(output_dtype)})"
            )
            continue
        if op_type == "TOPK_V2":
            runtime_imports.add("_normalize_dim")
            input_expr = _tensor_expr(str(op.inputs[0]))
            topk_pre_perm, topk_index_post_perm = _topk_codegen_layout_bridge(
                input_name=str(op.inputs[0]),
                value_output_name=str(outputs[0]),
                index_output_name=str(outputs[1]) if len(outputs) > 1 else None,
            )
            if topk_pre_perm is not None:
                input_expr = f"{input_expr}.permute({', '.join(str(int(v)) for v in topk_pre_perm)}).contiguous()"
            k_expr = f"{_tensor_expr(str(op.inputs[1]))}.reshape(-1)[0].to(dtype=torch.int64)"
            axis_expr = repr(int(op.options.get("axis", -1)))
            largest = bool(op.options.get("largest", True))
            sorted_output = bool(op.options.get("sorted", True))
            forward_lines.append(
                f"{output_vars[0]}, {output_vars[1]} = torch.topk({input_expr}, k=int({k_expr}.item()), dim=_normalize_dim({axis_expr}, {input_expr}.ndim), largest={largest}, sorted={sorted_output})"
            )
            if topk_index_post_perm is not None:
                forward_lines.append(
                    f"{output_vars[1]} = {output_vars[1]}.permute({', '.join(str(int(v)) for v in topk_index_post_perm)}).contiguous()"
                )
            output_value_tensor = model_ir.tensors.get(outputs[0], None)
            output_index_tensor = model_ir.tensors.get(outputs[1], None) if len(outputs) > 1 else None
            if output_value_tensor is not None:
                forward_lines.append(
                    f"{output_vars[0]} = {output_vars[0]}.to(dtype={_torch_dtype_literal(str(output_value_tensor.dtype))})"
                )
            if output_index_tensor is not None:
                forward_lines.append(
                    f"{output_vars[1]} = {output_vars[1]}.to(dtype={_torch_dtype_literal(str(output_index_tensor.dtype))})"
                )
            continue
        if op_type == "AVERAGE_POOL_2D":
            options = dict(op.options)
            runtime_imports.add("_apply_pool2d")
            forward_lines.append(
                f"{output_vars[0]} = _apply_pool2d({_tensor_expr(str(op.inputs[0]))}, "
                f"filter_height={int(options.get('filterHeight', 1))}, "
                f"filter_width={int(options.get('filterWidth', 1))}, "
                f"stride_h={int(options.get('strideH', 1))}, "
                f"stride_w={int(options.get('strideW', 1))}, "
                f"padding={str(options.get('padding', 'VALID')).upper()!r}, "
                f"target_shape={output_target_shape}, "
                "is_max_pool=False)"
            )
            forward_lines.extend(_activation_lines(output_vars[0], str(options.get("fusedActivationFunction", "NONE"))))
            continue
        if op_type == "MAX_POOL_2D":
            options = dict(op.options)
            runtime_imports.add("_apply_pool2d")
            forward_lines.append(
                f"{output_vars[0]} = _apply_pool2d({_tensor_expr(str(op.inputs[0]))}, "
                f"filter_height={int(options.get('filterHeight', 1))}, "
                f"filter_width={int(options.get('filterWidth', 1))}, "
                f"stride_h={int(options.get('strideH', 1))}, "
                f"stride_w={int(options.get('strideW', 1))}, "
                f"padding={str(options.get('padding', 'VALID')).upper()!r}, "
                f"target_shape={output_target_shape}, "
                "is_max_pool=True)"
            )
            forward_lines.extend(_activation_lines(output_vars[0], str(options.get("fusedActivationFunction", "NONE"))))
            continue
        if op_type == "RESIZE_NEAREST_NEIGHBOR":
            size_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            size_literal = _python_literal_for_constant_tensor(size_tensor) if size_tensor is not None else None
            resize_target_shape = _resize_target_shape_literal(outputs[0], str(op.inputs[0]))
            runtime_imports.add("_apply_resize")
            size_expr = (
                f"{size_literal}"
                if size_literal is not None else
                _tensor_expr(str(op.inputs[1]))
            )
            forward_lines.append(
                f"{output_vars[0]} = _apply_resize({_tensor_expr(str(op.inputs[0]))}, {size_expr}, method='nearest', target_shape={resize_target_shape})"
            )
            continue
        if op_type == "RESIZE_BILINEAR":
            size_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
            size_literal = _python_literal_for_constant_tensor(size_tensor) if size_tensor is not None else None
            align_corners = bool(op.options.get("alignCorners", False))
            half_pixel_centers = bool(op.options.get("halfPixelCenters", False))
            resize_target_shape = _resize_target_shape_literal(outputs[0], str(op.inputs[0]))
            runtime_imports.add("_apply_resize")
            size_expr = (
                f"{size_literal}"
                if size_literal is not None else
                _tensor_expr(str(op.inputs[1]))
            )
            forward_lines.append(
                f"{output_vars[0]} = _apply_resize({_tensor_expr(str(op.inputs[0]))}, {size_expr}, method='bilinear', target_shape={resize_target_shape}, align_corners={align_corners}, half_pixel_centers={half_pixel_centers})"
            )
            continue
        if op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            runtime_imports.update({"_normalize_axes"})
            reducer_map = {
                "SUM": "_reduce_sum",
                "MEAN": "_reduce_mean",
                "REDUCE_MAX": "_reduce_max",
                "REDUCE_MIN": "_reduce_min",
                "REDUCE_PROD": "_reduce_prod",
                "REDUCE_ANY": "_reduce_any",
            }
            runtime_imports.add(reducer_map[op_type])
            axis_values = None
            if len(op.inputs) >= 2:
                axis_values = _constant_int_list(model_ir.tensors.get(str(op.inputs[1]), None))
            axis_expr = (
                f"_normalize_axes({repr(axis_values)}, {_tensor_expr(str(op.inputs[0]))}.ndim)"
                if axis_values is not None
                else f"_normalize_axes({_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[0]))}.ndim)"
                if len(op.inputs) >= 2
                else "None"
            )
            keepdims = bool(op.options.get("keepDims", True))
            forward_lines.append(
                f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'{reducer_map[op_type]}({_tensor_expr(str(op.inputs[0]))}, {axis_expr}, {keepdims})', inferred_shape=_infer_reduction_shape(_tensor_shape_list(str(op.inputs[0])), axis_values, keepdims=keepdims))}"
            )
            continue
        if op_type == "PAD":
            if _pad_literal_expr(str(op.inputs[1])) is None:
                runtime_imports.add("_to_torch_pad_arg")
            pad_expr = _pad_literal_expr(str(op.inputs[1])) or f"_to_torch_pad_arg({_tensor_expr(str(op.inputs[1]))})"
            forward_lines.append(
                f"{output_vars[0]} = F.pad({_tensor_expr(str(op.inputs[0]))}, {pad_expr}, mode='constant', value=0.0)"
            )
            continue
        if op_type == "PADV2":
            if _pad_literal_expr(str(op.inputs[1])) is None:
                runtime_imports.add("_to_torch_pad_arg")
            pad_expr = _pad_literal_expr(str(op.inputs[1])) or f"_to_torch_pad_arg({_tensor_expr(str(op.inputs[1]))})"
            value_expr = _scalar_literal_expr(str(op.inputs[2])) or f"float({_tensor_expr(str(op.inputs[2]))}.reshape(-1)[0].item())"
            forward_lines.append(
                f"{output_vars[0]} = F.pad({_tensor_expr(str(op.inputs[0]))}, {pad_expr}, mode='constant', value={value_expr})"
            )
            continue
        if op_type == "MIRROR_PAD":
            runtime_imports.add("_apply_pad_nd")
            forward_lines.append(
                f"{output_vars[0]} = _apply_pad_nd({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, mode='reflect')"
            )
            continue
        if op_type in {"WHERE", "SELECT", "SELECT_V2"}:
            if len(op.inputs) == 1:
                forward_lines.append(
                    f"{output_vars[0]} = torch.nonzero({_tensor_expr(str(op.inputs[0]))}, as_tuple=False)"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = torch.where({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))}, {_tensor_expr(str(op.inputs[2]))})"
                )
            continue
        if op_type == "TILE":
            runtime_imports.add("_shape_list")
            forward_lines.append(
                f"{output_vars[0]} = {_tensor_expr(str(op.inputs[0]))}.repeat(*[int(v) for v in _shape_list({_tensor_expr(str(op.inputs[1]))})])"
            )
            continue
        if op_type == "BATCH_MATMUL":
            x_expr = _tensor_expr(str(op.inputs[0]))
            y_expr = _tensor_expr(str(op.inputs[1]))
            adj_x = bool(op.options.get("adjX", False))
            adj_y = bool(op.options.get("adjY", False))
            x_shape = _tensor_shape_list(str(op.inputs[0]))
            y_shape = _tensor_shape_list(str(op.inputs[1]))
            inferred_shape = _infer_batch_matmul_shape(
                x_shape,
                y_shape,
                adj_x=adj_x,
                adj_y=adj_y,
            )
            forward_lines.append(f"_tmp_x_{op_index} = {x_expr}")
            forward_lines.append(f"_tmp_y_{op_index} = {y_expr}")
            if adj_x:
                forward_lines.append(f"_tmp_x_{op_index} = _tmp_x_{op_index}.transpose(-1, -2)")
            if adj_y:
                forward_lines.append(f"_tmp_y_{op_index} = _tmp_y_{op_index}.transpose(-1, -2)")
            if (
                not adj_x
                and not adj_y
                and x_shape is not None
                and y_shape is not None
                and len(x_shape) >= 3
                and len(y_shape) == 2
            ):
                transposed_x_shape = list(x_shape[:-2]) + [int(x_shape[-1]), int(x_shape[-2])]
                inferred_shape_with_x_transpose = _infer_batch_matmul_shape(
                    transposed_x_shape,
                    y_shape,
                    adj_x=False,
                    adj_y=False,
                )
                input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                input_layout = normalize_logical_layout(input_tensor.logical_layout) if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
                if (
                    inferred_shape is None
                    and inferred_shape_with_x_transpose is not None
                    and is_channel_first_logical_layout(input_layout)
                ):
                    forward_lines.append(f"_tmp_x_{op_index} = _tmp_x_{op_index}.transpose(-1, -2)")
                    inferred_shape = inferred_shape_with_x_transpose
            runtime_imports.add("_align_tensor_to_target_shape")
            if output_target_shape != "None":
                forward_lines.append(
                    f"{output_vars[0]} = _align_tensor_to_target_shape(torch.matmul(_tmp_x_{op_index}, _tmp_y_{op_index}), {output_target_shape})"
                )
            else:
                forward_lines.append(
                    f"{output_vars[0]} = {_emit_maybe_aligned_expr(output_name=outputs[0], expr=f'torch.matmul(_tmp_x_{op_index}, _tmp_y_{op_index})', inferred_shape=inferred_shape)}"
                )
            continue
        if op_type == "NON_MAX_SUPPRESSION_V4":
            runtime_imports.add("_apply_non_max_suppression_v4")
            nms_method_name = f"_run_nms_{len(nms_method_specs)}"
            nms_method_specs.append(
                {
                    "name": nms_method_name,
                    "max_output_expr": _scalar_literal_expr(str(op.inputs[2])) or _tensor_expr(str(op.inputs[2])),
                    "iou_threshold_expr": _scalar_literal_expr(str(op.inputs[3])) or _tensor_expr(str(op.inputs[3])),
                    "score_threshold_expr": _scalar_literal_expr(str(op.inputs[4])) or _tensor_expr(str(op.inputs[4])),
                }
            )
            forward_lines.append(
                f"{', '.join(output_vars)} = self.{nms_method_name}({_tensor_expr(str(op.inputs[0]))}, {_tensor_expr(str(op.inputs[1]))})"
            )
            continue
        raise ModelIRPyTorchExportError(
            "Native PyTorch-like model.py codegen hit an unimplemented op emitter. "
            f"op_type={op_type}"
        )

    helper_source = (
        "def _normalize_tensor_name(name: str) -> str:\n"
        "    normalized = str(name).split(\":\")[0]\n"
        "    if normalized.startswith(\"serving_default_\"):\n"
        "        normalized = normalized[len(\"serving_default_\") :]\n"
        "    return normalized\n\n"
        "_TORCH_DTYPE_BY_TFLITE_DTYPE: Dict[str, torch.dtype] = {\n"
        "    'BOOL': torch.bool,\n"
        "    'INT8': torch.int8,\n"
        "    'INT16': torch.int16,\n"
        "    'INT32': torch.int32,\n"
        "    'INT64': torch.int64,\n"
        "    'UINT8': torch.uint8,\n"
        "    'FLOAT16': torch.float16,\n"
        "    'FLOAT32': torch.float32,\n"
        "    'FLOAT64': torch.float64,\n"
        "}\n\n"
        "def _torch_dtype(dtype_name: str) -> torch.dtype:\n"
        "    key = str(dtype_name).upper()\n"
        "    if key not in _TORCH_DTYPE_BY_TFLITE_DTYPE:\n"
        "        raise RuntimeError(f'Unsupported dtype for PyTorch runtime: {dtype_name}')\n"
        "    return _TORCH_DTYPE_BY_TFLITE_DTYPE[key]\n\n"
        "def _default_tensor_storage_name(tensor_name: str) -> str:\n"
        "    base_name = re.sub(r'[^0-9A-Za-z_]', '_', str(tensor_name)).strip('_')\n"
        "    if base_name == '':\n"
        "        base_name = 'tensor'\n"
        "    if base_name[0].isdigit():\n"
        "        base_name = f'tensor_{base_name}'\n"
        "    return base_name\n\n"
        "def _resolve_named_input_value(kwargs: Dict[str, Any], expected_name: str) -> Any:\n"
        "    if str(expected_name) in kwargs:\n"
        "        return kwargs[str(expected_name)]\n"
        "    normalized_expected_name = _normalize_tensor_name(str(expected_name))\n"
        "    canonical_expected_name = re.sub(r'[^0-9A-Za-z]+', '_', str(expected_name)).strip('_').lower()\n"
        "    for candidate_name, candidate_value in kwargs.items():\n"
        "        normalized_candidate = _normalize_tensor_name(str(candidate_name))\n"
        "        canonical_candidate = re.sub(r'[^0-9A-Za-z]+', '_', str(candidate_name)).strip('_').lower()\n"
        "        if (\n"
        "            normalized_candidate == normalized_expected_name\n"
        "            or canonical_candidate == canonical_expected_name\n"
        "            or normalized_candidate.endswith(normalized_expected_name)\n"
        "            or normalized_expected_name.endswith(normalized_candidate)\n"
        "            or canonical_candidate.endswith(canonical_expected_name)\n"
        "            or canonical_expected_name.endswith(canonical_candidate)\n"
        "        ):\n"
        "            return candidate_value\n"
        "    raise KeyError(str(expected_name))\n\n"
        "def resolve_named_input_value(kwargs: Dict[str, Any], expected_name: str) -> Any:\n"
        "    return _resolve_named_input_value(kwargs, expected_name)\n\n"
        "def _perm_cl_to_cf(rank: int) -> Optional[List[int]]:\n"
        "    if rank == 3:\n"
        "        return [0, 2, 1]\n"
        "    if rank == 4:\n"
        "        return [0, 3, 1, 2]\n"
        "    if rank == 5:\n"
        "        return [0, 4, 1, 2, 3]\n"
        "    return None\n\n"
        "def _perm_cf_to_cl(rank: int) -> Optional[List[int]]:\n"
        "    if rank == 3:\n"
        "        return [0, 2, 1]\n"
        "    if rank == 4:\n"
        "        return [0, 2, 3, 1]\n"
        "    if rank == 5:\n"
        "        return [0, 2, 3, 4, 1]\n"
        "    return None\n\n"
        "def _permute_shape(values: Sequence[int], perm: Sequence[int]) -> List[int]:\n"
        "    items = [int(v) for v in list(values)]\n"
        "    return [int(items[idx]) for idx in perm]\n\n"
        "def _torch_permute(value: torch.Tensor, perm: List[int]) -> torch.Tensor:\n"
        "    if len(perm) == 0:\n"
        "        return value\n"
        "    return torch.permute(value, perm).contiguous()\n\n"
        "def _can_broadcast_shapes(lhs: List[int], rhs: List[int]) -> bool:\n"
        "    lhs_index = len(lhs) - 1\n"
        "    rhs_index = len(rhs) - 1\n"
        "    while lhs_index >= 0 and rhs_index >= 0:\n"
        "        lhs_dim = int(lhs[lhs_index])\n"
        "        rhs_dim = int(rhs[rhs_index])\n"
        "        if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:\n"
        "            return False\n"
        "        lhs_index -= 1\n"
        "        rhs_index -= 1\n"
        "    return True\n\n"
        "def _broadcast_shape(lhs: Sequence[int], rhs: Sequence[int]) -> Optional[List[int]]:\n"
        "    lhs_items = [int(v) for v in list(lhs)]\n"
        "    rhs_items = [int(v) for v in list(rhs)]\n"
        "    result_rev = torch.jit.annotate(List[int], [])\n"
        "    lhs_index = len(lhs_items) - 1\n"
        "    rhs_index = len(rhs_items) - 1\n"
        "    while lhs_index >= 0 or rhs_index >= 0:\n"
        "        lhs_dim = int(lhs_items[lhs_index]) if lhs_index >= 0 else 1\n"
        "        rhs_dim = int(rhs_items[rhs_index]) if rhs_index >= 0 else 1\n"
        "        if lhs_dim != rhs_dim and lhs_dim != 1 and rhs_dim != 1:\n"
        "            return None\n"
        "        result_rev.append(int(lhs_dim if rhs_dim == 1 else rhs_dim))\n"
        "        lhs_index -= 1\n"
        "        rhs_index -= 1\n"
        "    return list(reversed(result_rev))\n\n"
        "def _align_tensor_to_target_shape(value: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    if target_shape is None:\n"
        "        return value\n"
        "    try:\n"
        "        actual_shape = [int(v) for v in list(value.shape)]\n"
        "        target = [int(v) for v in list(target_shape)]\n"
        "    except Exception:\n"
        "        return value\n"
        "    if actual_shape == target:\n"
        "        return value\n"
        "    perm = _perm_cl_to_cf(value.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == target:\n"
        "        return _torch_permute(value, perm)\n"
        "    perm_inv = _perm_cf_to_cl(value.ndim)\n"
        "    if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:\n"
        "        return _torch_permute(value, perm_inv)\n"
        "    return value\n\n"
        "@torch.jit.ignore\n"
        "def _align_scatter_nd_updates_eager(updates: torch.Tensor, expected_shape: Sequence[int]) -> torch.Tensor:\n"
        "    actual_shape = [int(v) for v in list(updates.shape)]\n"
        "    expected = [int(v) for v in list(expected_shape)]\n"
        "    try:\n"
        "        if list(torch.broadcast_shapes(tuple(actual_shape), tuple(expected))) == expected:\n"
        "            return updates\n"
        "    except Exception:\n"
        "        pass\n"
        "    perm = _perm_cf_to_cl(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    perm = _perm_cl_to_cf(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    if updates.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(updates.ndim)):\n"
        "            if list(generic_perm) == list(range(updates.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            if _permute_shape(actual_shape, generic_perm_list) == expected:\n"
        "                return _torch_permute(updates, generic_perm_list)\n"
        "    return updates\n\n"
        "def _align_scatter_nd_updates(updates: torch.Tensor, expected_shape: Sequence[int]) -> torch.Tensor:\n"
        "    actual_shape = [int(v) for v in list(updates.shape)]\n"
        "    expected = [int(v) for v in list(expected_shape)]\n"
        "    if actual_shape == expected:\n"
        "        return updates\n"
        "    broadcast_shape = _broadcast_shape(actual_shape, expected)\n"
        "    if broadcast_shape is not None and [int(v) for v in list(broadcast_shape)] == expected:\n"
        "        return updates\n"
        "    perm = _perm_cf_to_cl(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    perm = _perm_cl_to_cf(updates.ndim)\n"
        "    if perm is not None and _permute_shape(actual_shape, perm) == expected:\n"
        "        return _torch_permute(updates, perm)\n"
        "    if torch.jit.is_scripting():\n"
        "        return updates\n"
        "    return _align_scatter_nd_updates_eager(updates, expected)\n\n"
        "def _matches_target_except_axis(actual_shape: Sequence[int], target_shape: Sequence[int], axis: int) -> bool:\n"
        "    if len(list(actual_shape)) != len(list(target_shape)):\n"
        "        return False\n"
        "    for idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target_shape)):\n"
        "        if int(idx) == int(axis):\n"
        "            continue\n"
        "        if int(actual_dim) != int(target_dim):\n"
        "            return False\n"
        "    return True\n\n"
        "@torch.jit.ignore\n"
        "def _align_binary_inputs_eager(x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    try:\n"
        "        broadcast_shape = list(torch.broadcast_shapes(tuple(int(v) for v in x.shape), tuple(int(v) for v in y.shape)))\n"
        "        if target is None or [int(v) for v in broadcast_shape] == target:\n"
            "            return x, y\n"
        "    except Exception:\n"
            "        pass\n"
        "    perm = _perm_cl_to_cf(x.ndim)\n"
        "    if perm is None:\n"
        "        return x, y\n"
        "    x_shape = [int(v) for v in list(x.shape)]\n"
        "    y_shape = [int(v) for v in list(y.shape)]\n"
        "    if _permute_shape(y_shape, perm) == x_shape:\n"
        "        return x, _torch_permute(y, perm)\n"
        "    if _permute_shape(x_shape, perm) == y_shape:\n"
        "        return _torch_permute(x, perm), y\n"
        "    if target is not None:\n"
        "        if _permute_shape(y_shape, perm) == target:\n"
        "            return x, _torch_permute(y, perm)\n"
        "        if _permute_shape(x_shape, perm) == target:\n"
        "            return _torch_permute(x, perm), y\n"
        "    if x.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(x.ndim)):\n"
        "            if list(generic_perm) == list(range(x.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            permuted_y_shape = _permute_shape(y_shape, generic_perm_list)\n"
        "            if permuted_y_shape is not None:\n"
        "                try:\n"
        "                    torch.broadcast_shapes(tuple(permuted_y_shape), tuple(x_shape))\n"
        "                    return x, _torch_permute(y, generic_perm_list)\n"
        "                except Exception:\n"
        "                    pass\n"
        "                if target is not None:\n"
        "                    try:\n"
        "                        torch.broadcast_shapes(tuple(permuted_y_shape), tuple(target))\n"
        "                        return x, _torch_permute(y, generic_perm_list)\n"
        "                    except Exception:\n"
        "                        pass\n"
        "            permuted_x_shape = _permute_shape(x_shape, generic_perm_list)\n"
        "            if permuted_x_shape is not None:\n"
        "                try:\n"
        "                    torch.broadcast_shapes(tuple(permuted_x_shape), tuple(y_shape))\n"
        "                    return _torch_permute(x, generic_perm_list), y\n"
        "                except Exception:\n"
        "                    pass\n"
        "                if target is not None:\n"
        "                    try:\n"
        "                        torch.broadcast_shapes(tuple(permuted_x_shape), tuple(target))\n"
        "                        return _torch_permute(x, generic_perm_list), y\n"
        "                    except Exception:\n"
        "                        pass\n"
        "    return x, y\n\n"
        "def _align_binary_inputs(x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    if x.ndim != y.ndim:\n"
        "        return x, y\n"
        "    x_shape = [int(v) for v in list(x.shape)]\n"
        "    y_shape = [int(v) for v in list(y.shape)]\n"
        "    if x_shape == y_shape:\n"
        "        return x, y\n"
        "    if _can_broadcast_shapes(x_shape, y_shape):\n"
        "        return x, y\n"
        "    perm = _perm_cl_to_cf(x.ndim)\n"
        "    if perm is None:\n"
        "        return x, y\n"
        "    if _permute_shape(y_shape, perm) == x_shape:\n"
        "        return x, _torch_permute(y, perm)\n"
        "    if _permute_shape(x_shape, perm) == y_shape:\n"
        "        return _torch_permute(x, perm), y\n"
        "    if target is not None:\n"
        "        if _permute_shape(y_shape, perm) == target:\n"
        "            return x, _torch_permute(y, perm)\n"
        "        if _permute_shape(x_shape, perm) == target:\n"
        "            return _torch_permute(x, perm), y\n"
        "    if torch.jit.is_scripting():\n"
        "        return x, y\n"
        "    return _align_binary_inputs_eager(x, y, target_shape)\n\n"
        "@torch.jit.ignore\n"
        "def _align_binary_inputs_to_anchor_eager(anchor: torch.Tensor, other: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    anchor_shape = [int(v) for v in list(anchor.shape)]\n"
        "    other_shape = [int(v) for v in list(other.shape)]\n"
        "    perm = _perm_cl_to_cf(other.ndim)\n"
        "    if perm is not None and _permute_shape(other_shape, perm) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm)\n"
        "    perm_inv = _perm_cf_to_cl(other.ndim)\n"
        "    if perm_inv is not None and _permute_shape(other_shape, perm_inv) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm_inv)\n"
        "    if other.ndim <= 5:\n"
        "        import itertools\n"
        "        for generic_perm in itertools.permutations(range(other.ndim)):\n"
        "            if list(generic_perm) == list(range(other.ndim)):\n"
        "                continue\n"
        "            generic_perm_list = [int(v) for v in list(generic_perm)]\n"
        "            permuted_shape = _permute_shape(other_shape, generic_perm_list)\n"
        "            if permuted_shape == anchor_shape:\n"
        "                return anchor, _torch_permute(other, generic_perm_list)\n"
        "            if permuted_shape is not None:\n"
        "                try:\n"
        "                    broadcast_shape = list(torch.broadcast_shapes(tuple(anchor_shape), tuple(permuted_shape)))\n"
        "                    if broadcast_shape == anchor_shape or (target is not None and broadcast_shape == target):\n"
        "                        return anchor, _torch_permute(other, generic_perm_list)\n"
        "                except Exception:\n"
        "                    pass\n"
        "    return anchor, other\n\n"
        "def _align_binary_inputs_to_anchor(anchor: torch.Tensor, other: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    if anchor.ndim != other.ndim:\n"
            "        return anchor, other\n"
        "    anchor_shape = [int(v) for v in list(anchor.shape)]\n"
        "    other_shape = [int(v) for v in list(other.shape)]\n"
        "    if anchor_shape == other_shape:\n"
            "        return anchor, other\n"
        "    perm = _perm_cl_to_cf(other.ndim)\n"
        "    if perm is not None and _permute_shape(other_shape, perm) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm)\n"
        "    perm_inv = _perm_cf_to_cl(other.ndim)\n"
        "    if perm_inv is not None and _permute_shape(other_shape, perm_inv) == anchor_shape:\n"
        "        return anchor, _torch_permute(other, perm_inv)\n"
        "    if target is not None and _can_broadcast_shapes(anchor_shape, other_shape):\n"
        "        return anchor, other\n"
        "    if torch.jit.is_scripting():\n"
        "        return anchor, other\n"
        "    return _align_binary_inputs_to_anchor_eager(anchor, other, target_shape)\n\n"
        "def _normalize_dim(dim: int, rank: int) -> int:\n"
        "    resolved = int(dim)\n"
        "    if resolved < 0:\n"
        "        resolved += int(rank)\n"
        "    return resolved\n\n"
        "def _coerce_scalar_axis(value: Any, *, device: torch.device) -> int:\n"
        "    if isinstance(value, torch.Tensor):\n"
        "        flat = value.to(dtype=torch.int64, device=device).reshape(-1)\n"
        "        if int(flat.numel()) == 0:\n"
        "            return 0\n"
        "        return int(flat[0].item())\n"
        "    return int(value)\n\n"
        "def _shape_list(value: Any) -> List[int]:\n"
        "    if isinstance(value, torch.Tensor):\n"
        "        flat = value.to(dtype=torch.int64).reshape(-1)\n"
        "        result = torch.jit.annotate(List[int], [])\n"
        "        for idx in range(int(flat.numel())):\n"
        "            result.append(int(flat[idx].item()))\n"
        "        return result\n"
        "    if torch.jit.isinstance(value, List[int]):\n"
        "        result = torch.jit.annotate(List[int], [])\n"
        "        for item in value:\n"
        "            result.append(int(item))\n"
        "        return result\n"
        "    raise RuntimeError('Unsupported shape spec type for _shape_list')\n\n"
        "def _resolve_reshape_shape(shape_spec: Any, input_tensor: torch.Tensor, *, allow_zero: bool) -> List[int]:\n"
        "    raw_shape = _shape_list(shape_spec)\n"
        "    input_shape = [int(v) for v in list(input_tensor.shape)]\n"
        "    resolved: List[int] = []\n"
        "    infer_index: Optional[int] = None\n"
        "    known_product = 1\n"
        "    total_elements = int(input_tensor.numel())\n"
        "    for dim_index, raw_dim in enumerate(raw_shape):\n"
        "        dim_value = int(raw_dim)\n"
        "        if dim_value == 0 and not allow_zero and dim_index < len(input_shape):\n"
        "            dim_value = int(input_shape[dim_index])\n"
        "        if dim_value == -1:\n"
        "            if infer_index is not None:\n"
        "                raise RuntimeError(f'Multiple -1 values are not allowed in reshape spec: {raw_shape}')\n"
        "            infer_index = len(resolved)\n"
        "            resolved.append(-1)\n"
        "            continue\n"
        "        resolved.append(dim_value)\n"
        "        known_product *= int(dim_value)\n"
        "    if infer_index is not None:\n"
        "        if known_product == 0 or total_elements % known_product != 0:\n"
        "            raise RuntimeError(\n"
        "                'Failed to infer reshape dimension. '\n"
        "                f'shape_spec={raw_shape} input_shape={input_shape} total_elements={total_elements}'\n"
        "            )\n"
        "        resolved[infer_index] = int(total_elements // known_product)\n"
        "    return resolved\n\n"
        "def _to_torch_pad_arg(paddings: torch.Tensor) -> List[int]:\n"
        "    pads_tensor = paddings.to(dtype=torch.int64).reshape(-1, 2)\n"
        "    torch_pad: List[int] = []\n"
        "    for row_index in range(int(pads_tensor.shape[0]) - 1, -1, -1):\n"
        "        torch_pad.extend([\n"
        "            int(pads_tensor[row_index, 0].item()),\n"
        "            int(pads_tensor[row_index, 1].item()),\n"
        "        ])\n"
        "    while len(torch_pad) >= 2 and int(torch_pad[-2]) == 0 and int(torch_pad[-1]) == 0:\n"
        "        torch_pad = torch_pad[:-2]\n"
        "    return torch_pad\n\n"
        "def _apply_pad_nd(x: torch.Tensor, paddings: torch.Tensor, *, mode: str, value: float = 0.0) -> torch.Tensor:\n"
        "    pads_tensor = paddings.to(dtype=torch.int64).reshape(-1, 2)\n"
        "    pad_pairs = torch.jit.annotate(List[List[int]], [])\n"
        "    for row_index in range(int(pads_tensor.shape[0])):\n"
        "        pad_pairs.append([\n"
        "            int(pads_tensor[row_index, 0].item()),\n"
        "            int(pads_tensor[row_index, 1].item()),\n"
        "        ])\n"
        "    rank = int(x.ndim)\n"
        "    if len(pad_pairs) < rank:\n"
        "        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs\n"
        "    elif len(pad_pairs) > rank:\n"
        "        pad_pairs = pad_pairs[-rank:]\n"
        "    non_zero_axes = [idx for idx, (before, after) in enumerate(pad_pairs) if int(before) != 0 or int(after) != 0]\n"
        "    if len(non_zero_axes) == 0:\n"
        "        return x\n"
        "    if mode != 'constant' and len(non_zero_axes) > 3:\n"
        "        raise RuntimeError(f'Non-constant pad supports at most 3 padded dims. mode={mode} padded_dims={len(non_zero_axes)}')\n"
        "    keep_axes = [idx for idx in range(rank) if idx not in non_zero_axes]\n"
        "    perm = keep_axes + non_zero_axes\n"
        "    permuted = _torch_permute(x, perm) if perm != list(range(rank)) else x\n"
        "    torch_pad: List[int] = []\n"
        "    for axis in reversed(non_zero_axes):\n"
        "        before, after = pad_pairs[axis]\n"
        "        torch_pad.extend([int(before), int(after)])\n"
        "    if mode == 'constant':\n"
        "        padded = F.pad(permuted, torch_pad, mode=mode, value=float(value))\n"
        "    else:\n"
        "        padded = F.pad(permuted, torch_pad, mode=mode)\n"
        "    if perm == list(range(rank)):\n"
        "        return padded\n"
        "    inverse_perm = [0] * rank\n"
        "    for permuted_axis, original_axis in enumerate(perm):\n"
        "        inverse_perm[int(original_axis)] = int(permuted_axis)\n"
        "    return _torch_permute(padded, inverse_perm)\n\n"
        "def _infer_spatial_shape_for_transposed_conv2d(*, raw_output: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int]) -> Tuple[int, int]:\n"
        "    output_channels = int(raw_output.shape[1])\n"
        "    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]\n"
        "    if len(source) == 4:\n"
        "        if int(source[1]) == output_channels:\n"
        "            return int(source[2]), int(source[3])\n"
        "        if int(source[-1]) == output_channels:\n"
        "            return int(source[1]), int(source[2])\n"
        "    return int(source[-2]), int(source[-1])\n\n"
        "def _infer_spatial_shape_for_transposed_conv3d(*, raw_output: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int]) -> Tuple[int, int, int]:\n"
        "    output_channels = int(raw_output.shape[1])\n"
        "    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]\n"
        "    if len(source) == 5:\n"
        "        if int(source[1]) == output_channels:\n"
        "            return int(source[2]), int(source[3]), int(source[4])\n"
        "        if int(source[-1]) == output_channels:\n"
        "            return int(source[1]), int(source[2]), int(source[3])\n"
        "    return int(source[-3]), int(source[-2]), int(source[-1])\n\n"
        "def _apply_fused_activation(x: torch.Tensor, fused: str) -> torch.Tensor:\n"
        "    key = str(fused).upper()\n"
        "    if key == '' or key == 'NONE':\n"
        "        return x\n"
        "    if key == 'RELU':\n"
        "        return torch.relu(x)\n"
        "    if key == 'RELU6':\n"
        "        return torch.clamp(x, min=0.0, max=6.0)\n"
        "    if key == 'RELU_N1_TO_1':\n"
        "        return torch.clamp(x, min=-1.0, max=1.0)\n"
        "    if key == 'RELU_0_TO_1':\n"
        "        return torch.clamp(x, min=0.0, max=1.0)\n"
        "    if key == 'TANH':\n"
        "        return torch.tanh(x)\n"
        "    return x\n\n"
        "def _lookup_state_tensor(raw_state_dict: Dict[str, Any], tensor_name: str, storage_names: Dict[str, str]) -> torch.Tensor:\n"
        "    original_key = str(tensor_name)\n"
        "    storage_key = storage_names.get(original_key, _default_tensor_storage_name(original_key))\n"
        "    if original_key in raw_state_dict:\n"
        "        return torch.as_tensor(raw_state_dict[original_key])\n"
        "    if storage_key in raw_state_dict:\n"
        "        return torch.as_tensor(raw_state_dict[storage_key])\n"
        "    raise KeyError(original_key)\n\n"
        "def _copy_tensor_data(target: torch.Tensor, source: torch.Tensor) -> None:\n"
        "    target.data.copy_(source.to(device=target.device, dtype=target.dtype))\n\n"
        "def _validate_state_dict_keys(raw_state_dict: Dict[str, Any], storage_names: Dict[str, str], expected_tensor_names: Sequence[str]) -> None:\n"
        "    recognized_keys: Set[str] = set()\n"
        "    missing: List[str] = []\n"
        "    for tensor_name in expected_tensor_names:\n"
        "        storage_key = storage_names.get(str(tensor_name), _default_tensor_storage_name(str(tensor_name)))\n"
        "        if str(tensor_name) in raw_state_dict:\n"
        "            recognized_keys.add(str(tensor_name))\n"
        "            continue\n"
        "        if storage_key in raw_state_dict:\n"
        "            recognized_keys.add(storage_key)\n"
        "            continue\n"
        "        missing.append(str(tensor_name))\n"
        "    unexpected = sorted(str(key) for key in raw_state_dict.keys() if str(key) not in recognized_keys)\n"
        "    if len(missing) > 0 or len(unexpected) > 0:\n"
        "        raise RuntimeError(f'state_dict mismatch. missing={missing} unexpected={unexpected}')\n\n"
        "def _apply_concat(values: Sequence[torch.Tensor], axis: int, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    if any(int(value.ndim) == 0 for value in values):\n"
        "        values = [value.reshape(1) if int(value.ndim) == 0 else value for value in values]\n"
        "    rank = int(values[0].ndim)\n"
        "    resolved_axis = _normalize_dim(int(axis), rank)\n"
        "    try:\n"
        "        target = [int(v) for v in list(target_shape)] if target_shape is not None else None\n"
        "    except Exception:\n"
        "        target = None\n"
        "    if target is not None and len(target) == rank:\n"
        "        aligned_values: List[torch.Tensor] = []\n"
        "        for value in values:\n"
        "            try:\n"
        "                actual = [int(v) for v in list(value.shape)]\n"
        "            except Exception:\n"
        "                aligned_values.append(value)\n"
        "                continue\n"
        "            chosen = value\n"
        "            if actual != target:\n"
        "                if _matches_target_except_axis(actual, target, resolved_axis):\n"
        "                    aligned_values.append(chosen)\n"
        "                    continue\n"
        "                perm = _perm_cl_to_cf(value.ndim)\n"
        "                if perm is not None:\n"
        "                    permuted_shape = _permute_shape(actual, perm)\n"
        "                    if _matches_target_except_axis(permuted_shape, target, resolved_axis):\n"
        "                        chosen = _torch_permute(value, perm)\n"
        "            aligned_values.append(chosen)\n"
        "        values = aligned_values\n"
        "    y = torch.cat(list(values), dim=resolved_axis)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_conv2d(module: torch.nn.Conv2d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    expected_in_channels = int(module.in_channels)\n"
        "    if x.ndim == 4 and int(x.shape[1]) != expected_in_channels and int(x.shape[-1]) == expected_in_channels:\n"
        "        x = x.permute(0, 3, 1, 2).contiguous()\n"
        "    y = module(x)\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_transpose_conv2d(module: torch.nn.ConvTranspose2d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 4 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):\n"
        "        x = x.permute(0, 3, 1, 2).contiguous()\n"
        "    raw = module(x)\n"
        "    target_h, target_w = _infer_spatial_shape_for_transposed_conv2d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)\n"
        "    y = raw[..., :target_h, :target_w]\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_conv3d(module: torch.nn.Conv3d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[1]) and int(x.shape[-1]) == int(weight.shape[1]):\n"
        "        x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    y = module(x)\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_module_transpose_conv3d(module: torch.nn.ConvTranspose3d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], fused: str) -> torch.Tensor:\n"
        "    weight = module.weight\n"
        "    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):\n"
        "        x = x.permute(0, 4, 1, 2, 3).contiguous()\n"
        "    raw = module(x)\n"
        "    target_d, target_h, target_w = _infer_spatial_shape_for_transposed_conv3d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)\n"
        "    y = raw[..., :target_d, :target_h, :target_w]\n"
        "    y = _align_tensor_to_target_shape(y, target_shape)\n"
        "    return _apply_fused_activation(y, fused)\n\n"
        "def _apply_softmax(x: torch.Tensor, axis: Optional[int], beta: float, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    resolved_axis = _normalize_dim(int(axis), x.ndim) if axis is not None else -1\n"
        "    if beta != 1.0:\n"
        "        x = x * beta\n"
        "    y = torch.softmax(x, dim=resolved_axis)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_gather(params: torch.Tensor, indices: torch.Tensor, axis: int, batch_dims: int, target_shape: Optional[Sequence[int]], indices_name: str) -> torch.Tensor:\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    resolved_axis = _normalize_dim(int(axis), params.ndim)\n"
        "    if int(batch_dims) == 0 and int(resolved_axis) == 1 and str(indices_name).endswith('_crd_to_dcr_indices'):\n"
        "        return _align_tensor_to_target_shape(params, target_shape)\n"
        "    resolved_batch_dims = int(batch_dims)\n"
        "    if resolved_batch_dims < 0:\n"
        "        resolved_batch_dims += indices_i64.ndim\n"
        "    if resolved_batch_dims > 0:\n"
        "        leading_shape = [int(v) for v in list(indices_i64.shape[:resolved_batch_dims])]\n"
        "        flat_batch = int(np.prod(leading_shape, dtype=np.int64))\n"
        "        params_flat = params.reshape(flat_batch, *params.shape[resolved_batch_dims:])\n"
        "        indices_flat = indices_i64.reshape(flat_batch, *indices_i64.shape[resolved_batch_dims:])\n"
        "        gathered_batches: List[torch.Tensor] = []\n"
        "        adjusted_axis = int(resolved_axis - resolved_batch_dims + 1)\n"
        "        for batch_index in range(flat_batch):\n"
        "            batch_params = params_flat[batch_index]\n"
        "            batch_indices = indices_flat[batch_index]\n"
        "            flat_indices = batch_indices.reshape(-1)\n"
        "            batch_gathered = torch.index_select(batch_params, adjusted_axis - 1, flat_indices)\n"
        "            batch_gathered = batch_gathered.reshape(*batch_params.shape[: adjusted_axis - 1], *batch_indices.shape, *batch_params.shape[adjusted_axis:])\n"
        "            gathered_batches.append(batch_gathered)\n"
        "        y = torch.stack(gathered_batches, dim=0).reshape(*leading_shape, *gathered_batches[0].shape)\n"
        "        return _align_tensor_to_target_shape(y, target_shape)\n"
        "    if indices_i64.ndim == 0:\n"
        "        y = torch.index_select(params, resolved_axis, indices_i64.reshape(1)).squeeze(resolved_axis)\n"
        "        return _align_tensor_to_target_shape(y, target_shape)\n"
        "    flat_indices = indices_i64.reshape(-1)\n"
        "    gathered = torch.index_select(params, resolved_axis, flat_indices)\n"
        "    y = gathered.reshape(*params.shape[:resolved_axis], *indices_i64.shape, *params.shape[resolved_axis + 1:])\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_gather_elements(params: torch.Tensor, indices: torch.Tensor, axis_hint: int, axis_hint_cl: Optional[int], target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    index_max = -1 if indices_i64.numel() == 0 else int(indices_i64.max().item())\n"
        "    gather_axis = int(axis_hint)\n"
        "    if index_max >= 0 and index_max >= int(params.shape[gather_axis]):\n"
        "        if axis_hint_cl is not None and index_max < int(params.shape[int(axis_hint_cl)]):\n"
        "            gather_axis = int(axis_hint_cl)\n"
        "        else:\n"
        "            exact_axes = [axis for axis in range(params.ndim) if int(params.shape[axis]) == index_max + 1]\n"
        "            if len(exact_axes) > 0:\n"
        "                gather_axis = int(exact_axes[-1])\n"
        "            else:\n"
        "                valid_axes = [axis for axis in range(params.ndim) if index_max < int(params.shape[axis])]\n"
        "                if len(valid_axes) > 0:\n"
        "                    gather_axis = int(valid_axes[-1])\n"
        "    y = torch.gather(params, dim=gather_axis, index=indices_i64)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_gather_nd(params: torch.Tensor, indices: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    index_tuple = tuple(indices_i64[..., i] for i in range(indices_i64.shape[-1]))\n"
        "    y = params[index_tuple]\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    output_shape = _shape_list(shape)\n"
        "    indices_i64 = indices.to(dtype=torch.int64)\n"
        "    index_depth = int(indices_i64.shape[-1])\n"
        "    prefix_shape = [int(v) for v in list(indices_i64.shape[:-1])]\n"
        "    slice_shape = [int(v) for v in output_shape[index_depth:]]\n"
        "    expected_updates_shape = prefix_shape + slice_shape\n"
        "    aligned_updates = _align_scatter_nd_updates(updates, expected_updates_shape)\n"
        "    flat_indices = indices_i64.reshape(-1, index_depth)\n"
        "    if int(flat_indices.shape[0]) == 0:\n"
        "        y = torch.zeros(output_shape, dtype=updates.dtype, device=updates.device)\n"
        "        return _align_tensor_to_target_shape(y, target_shape)\n"
        "    row_shape = [int(v) for v in output_shape[:index_depth]]\n"
        "    strides: List[int] = [1] * len(row_shape)\n"
        "    row_count = 1\n"
        "    running = 1\n"
        "    for axis in range(len(row_shape) - 1, -1, -1):\n"
        "        strides[axis] = int(running)\n"
        "        running *= int(row_shape[axis])\n"
        "    row_count = int(running)\n"
        "    slice_size = 1\n"
        "    for dim in slice_shape:\n"
        "        slice_size *= int(dim)\n"
        "    stride_tensor = torch.as_tensor(strides, dtype=torch.int64, device=indices_i64.device)\n"
        "    flat_rows = torch.sum(flat_indices * stride_tensor, dim=-1)\n"
        "    if int(slice_size) == 1:\n"
        "        y_flat = torch.zeros((row_count,), dtype=aligned_updates.dtype, device=aligned_updates.device)\n"
        "        y_flat.index_copy_(0, flat_rows, aligned_updates.reshape(-1))\n"
        "        y = y_flat.reshape(output_shape)\n"
        "    else:\n"
        "        y_rows = torch.zeros((row_count, slice_size), dtype=aligned_updates.dtype, device=aligned_updates.device)\n"
        "        y_rows.index_copy_(0, flat_rows, aligned_updates.reshape(-1, slice_size))\n"
        "        y = y_rows.reshape(output_shape)\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_slice(x: torch.Tensor, begin: torch.Tensor, size: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    begin_values = _shape_list(begin)\n"
        "    size_values = _shape_list(size)\n"
        "    slices: List[slice] = []\n"
        "    for axis, start in enumerate(begin_values):\n"
        "        dim_size = int(x.shape[axis])\n"
        "        length = int(size_values[axis])\n"
        "        stop = None if length < 0 else min(int(start) + length, dim_size)\n"
        "        slices.append(slice(int(start), stop))\n"
        "    y = x[tuple(slices)]\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _apply_strided_slice(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, strides: torch.Tensor, begin_mask: int, end_mask: int, target_shape: Optional[Sequence[int]]) -> torch.Tensor:\n"
        "    begin_values = _shape_list(begin)\n"
        "    end_values = _shape_list(end)\n"
        "    stride_values = _shape_list(strides)\n"
        "    slices: List[slice] = []\n"
        "    for axis, (start, stop, step) in enumerate(zip(begin_values, end_values, stride_values)):\n"
        "        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)\n"
        "        resolved_stop = None if ((int(end_mask) >> axis) & 1) else int(stop)\n"
        "        slices.append(slice(resolved_start, resolved_stop, int(step)))\n"
        "    y = x[tuple(slices)]\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _resolve_same_padding(kernel_size: int, stride: int) -> Tuple[int, int]:\n"
        "    total = max(int(kernel_size) - int(stride), 0)\n"
        "    before = total // 2\n"
        "    after = total - before\n"
        "    return before, after\n\n"
        "def _apply_pool2d(x: torch.Tensor, filter_height: int, filter_width: int, stride_h: int, stride_w: int, padding: str, target_shape: Optional[Sequence[int]], is_max_pool: bool) -> torch.Tensor:\n"
        "    resize_as_channel_last = False\n"
        "    if x.ndim == 4 and target_shape is not None and len(list(target_shape)) == 4:\n"
        "        actual_shape = [int(v) for v in list(x.shape)]\n"
        "        target = [int(v) for v in list(target_shape)]\n"
        "        if actual_shape[-1] == target[-1] and actual_shape[1] != target[1]:\n"
        "            resize_as_channel_last = True\n"
        "    pool_input = x.permute(0, 3, 1, 2).contiguous() if resize_as_channel_last and x.ndim == 4 else x\n"
        "    if str(padding).upper() == 'SAME':\n"
        "        pad_w = _resolve_same_padding(filter_width, stride_w)\n"
        "        pad_h = _resolve_same_padding(filter_height, stride_h)\n"
        "        pool_input = F.pad(pool_input, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]], mode='constant', value=float('-inf') if is_max_pool else 0.0)\n"
        "        padding_value = 0\n"
        "    else:\n"
        "        padding_value = 0\n"
        "    if is_max_pool:\n"
        "        y = F.max_pool2d(pool_input, kernel_size=(filter_height, filter_width), stride=(stride_h, stride_w), padding=padding_value)\n"
        "    else:\n"
        "        y = F.avg_pool2d(pool_input, kernel_size=(filter_height, filter_width), stride=(stride_h, stride_w), padding=padding_value)\n"
        "    if resize_as_channel_last and y.ndim == 4:\n"
        "        y = y.permute(0, 2, 3, 1).contiguous()\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _resize_bilinear_exact(x: torch.Tensor, size: Sequence[int], *, align_corners: bool, half_pixel_centers: bool) -> torch.Tensor:\n"
        "    if x.ndim != 4:\n"
        "        return F.interpolate(x, size=[int(size[0]), int(size[1])], mode='bilinear', align_corners=align_corners)\n"
        "    out_h = int(size[0])\n"
        "    out_w = int(size[1])\n"
        "    in_h = int(x.shape[-2])\n"
        "    in_w = int(x.shape[-1])\n"
        "    if out_h <= 0 or out_w <= 0:\n"
        "        raise RuntimeError('Resize target dimensions must be positive.')\n"
        "    if align_corners:\n"
        "        ys = torch.zeros([out_h], dtype=torch.float32, device=x.device) if out_h == 1 else torch.arange(out_h, dtype=torch.float32, device=x.device) * float(max(in_h - 1, 0)) / float(max(out_h - 1, 1))\n"
        "        xs = torch.zeros([out_w], dtype=torch.float32, device=x.device) if out_w == 1 else torch.arange(out_w, dtype=torch.float32, device=x.device) * float(max(in_w - 1, 0)) / float(max(out_w - 1, 1))\n"
        "    elif half_pixel_centers:\n"
        "        ys = (torch.arange(out_h, dtype=torch.float32, device=x.device) + 0.5) * float(in_h) / float(out_h) - 0.5\n"
        "        xs = (torch.arange(out_w, dtype=torch.float32, device=x.device) + 0.5) * float(in_w) / float(out_w) - 0.5\n"
        "    else:\n"
        "        ys = torch.arange(out_h, dtype=torch.float32, device=x.device) * float(in_h) / float(out_h)\n"
        "        xs = torch.arange(out_w, dtype=torch.float32, device=x.device) * float(in_w) / float(out_w)\n"
        "    y0 = torch.floor(ys).to(dtype=torch.int64)\n"
        "    x0 = torch.floor(xs).to(dtype=torch.int64)\n"
        "    y1 = y0 + 1\n"
        "    x1 = x0 + 1\n"
        "    y0c = y0.clamp(0, max(in_h - 1, 0))\n"
        "    x0c = x0.clamp(0, max(in_w - 1, 0))\n"
        "    y1c = y1.clamp(0, max(in_h - 1, 0))\n"
        "    x1c = x1.clamp(0, max(in_w - 1, 0))\n"
        "    ly = (ys - y0.to(dtype=torch.float32)).view(1, 1, out_h, 1)\n"
        "    lx = (xs - x0.to(dtype=torch.float32)).view(1, 1, 1, out_w)\n"
        "    hy = 1.0 - ly\n"
        "    hx = 1.0 - lx\n"
        "    top_left = x[:, :, y0c[:, None], x0c[None, :]]\n"
        "    top_right = x[:, :, y0c[:, None], x1c[None, :]]\n"
        "    bottom_left = x[:, :, y1c[:, None], x0c[None, :]]\n"
        "    bottom_right = x[:, :, y1c[:, None], x1c[None, :]]\n"
        "    return top_left * hy * hx + top_right * hy * lx + bottom_left * ly * hx + bottom_right * ly * lx\n\n"
        "def _apply_resize(x: torch.Tensor, size: Any, method: str, target_shape: Optional[Sequence[int]], align_corners: bool = False, half_pixel_centers: bool = False) -> torch.Tensor:\n"
        "    resize_size = _shape_list(size)\n"
        "    resize_as_channel_last = False\n"
        "    if x.ndim == 4 and target_shape is not None and len(list(target_shape)) == 4:\n"
        "        actual_shape = [int(v) for v in list(x.shape)]\n"
        "        target = [int(v) for v in list(target_shape)]\n"
        "        if actual_shape[-1] == target[-1] and actual_shape[1] != target[1]:\n"
        "            resize_as_channel_last = True\n"
        "    resize_input = x.permute(0, 3, 1, 2).contiguous() if resize_as_channel_last and x.ndim == 4 else x\n"
        "    if str(method).lower() == 'nearest':\n"
        "        y = F.interpolate(resize_input, size=resize_size, mode='nearest')\n"
        "    else:\n"
        "        y = _resize_bilinear_exact(resize_input, resize_size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)\n"
        "    if resize_as_channel_last and y.ndim == 4:\n"
        "        y = y.permute(0, 2, 3, 1).contiguous()\n"
        "    return _align_tensor_to_target_shape(y, target_shape)\n\n"
        "def _box_iou(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:\n"
        "    x1 = torch.maximum(boxes[:, 0], box[0])\n"
        "    y1 = torch.maximum(boxes[:, 1], box[1])\n"
        "    x2 = torch.minimum(boxes[:, 2], box[2])\n"
        "    y2 = torch.minimum(boxes[:, 3], box[3])\n"
        "    inter_w = torch.clamp(x2 - x1, min=0.0)\n"
        "    inter_h = torch.clamp(y2 - y1, min=0.0)\n"
        "    inter = inter_w * inter_h\n"
        "    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)\n"
        "    box_area = torch.clamp(box[2] - box[0], min=0.0) * torch.clamp(box[3] - box[1], min=0.0)\n"
        "    union = boxes_area + box_area - inter\n"
        "    safe_union = torch.where(union > 0, union, torch.ones_like(union))\n"
        "    iou = inter / safe_union\n"
        "    return torch.where(union > 0, iou, torch.zeros_like(iou))\n\n"
        "def _apply_non_max_suppression_v4(boxes: torch.Tensor, scores: torch.Tensor, max_output_size: torch.Tensor, iou_threshold: torch.Tensor, score_threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:\n"
        "    flat_boxes = boxes.to(dtype=torch.float32).reshape(-1, 4)\n"
        "    flat_scores = scores.to(dtype=torch.float32).reshape(-1)\n"
        "    max_outputs = max(0, int(max_output_size.reshape(-1)[0].to(dtype=torch.int64).item()))\n"
        "    selected = torch.zeros([max_outputs], dtype=torch.int64, device=flat_boxes.device)\n"
        "    valid_count = torch.zeros([], dtype=torch.int32, device=flat_boxes.device)\n"
        "    if max_outputs == 0:\n"
        "        return selected.to(dtype=torch.int32), valid_count\n"
        "    iou_thresh = float(iou_threshold.reshape(-1)[0].item())\n"
        "    score_thresh = float(score_threshold.reshape(-1)[0].item())\n"
        "    candidate_scores = torch.where(\n"
        "        flat_scores > score_thresh,\n"
        "        flat_scores,\n"
        "        torch.full_like(flat_scores, float('-inf')),\n"
        "    )\n"
        "    all_indices = torch.arange(flat_scores.shape[0], dtype=torch.int64, device=flat_boxes.device)\n"
        "    neg_inf = torch.full_like(candidate_scores, float('-inf'))\n"
        "    for out_index in range(max_outputs):\n"
        "        current_score, current_index = torch.max(candidate_scores, dim=0)\n"
        "        current_index = current_index.to(dtype=torch.int64)\n"
        "        is_valid = torch.isfinite(current_score)\n"
        "        selected[out_index : out_index + 1] = torch.where(is_valid, current_index, torch.zeros_like(current_index)).reshape(1)\n"
        "        valid_count = valid_count + is_valid.to(dtype=torch.int32)\n"
        "        current_box = torch.index_select(flat_boxes, 0, current_index.reshape(1)).reshape(4)\n"
        "        suppress = _box_iou(flat_boxes, current_box) > iou_thresh\n"
        "        suppress = torch.logical_or(suppress, all_indices == current_index)\n"
        "        suppress = torch.logical_and(suppress, is_valid)\n"
        "        candidate_scores = torch.where(suppress, neg_inf, candidate_scores)\n"
        "    return selected.to(dtype=torch.int32), valid_count\n\n"
        "def _normalize_axes(value: Any, rank: int) -> Optional[Tuple[int, ...]]:\n"
        "    if value is None:\n"
        "        return None\n"
        "    axes = _shape_list(value)\n"
        "    return tuple(sorted({_normalize_dim(int(v), rank) for v in axes}))\n\n"
        "def _reduce_sum(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.sum(x) if not keepdims else torch.sum(x).reshape([1] * x.ndim)\n"
        "    return torch.sum(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_mean(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.mean(x) if not keepdims else torch.mean(x).reshape([1] * x.ndim)\n"
        "    return torch.mean(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_max(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.amax(x, keepdim=keepdims)\n"
        "    return torch.amax(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_min(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        return torch.amin(x, keepdim=keepdims)\n"
        "    return torch.amin(x, dim=axis, keepdim=keepdims)\n\n"
        "def _reduce_prod(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        y = torch.prod(x)\n"
        "        return y if not keepdims else y.reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    for dim in sorted(axis, reverse=True):\n"
        "        result = torch.prod(result, dim=dim, keepdim=keepdims)\n"
        "    return result\n\n"
        "def _reduce_any(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:\n"
        "    if axis is None:\n"
        "        y = torch.any(x)\n"
        "        return y if not keepdims else y.reshape([1] * x.ndim)\n"
        "    result = x\n"
        "    for dim in sorted(axis, reverse=True):\n"
        "        result = torch.any(result, dim=dim, keepdim=keepdims)\n"
        "    return result\n\n"
    )

    runtime_source = _build_native_runtime_source(helper_source)
    _write_generated_package_common_files(
        output_folder_path,
        runtime_source=runtime_source,
    )
    buffer_init_block = "\n".join(f"        {line}" for line in buffer_init_lines)
    buffer_annotation_block = ""
    init_constants_method = ""
    if len(buffer_init_lines) > 0:
        init_constants_method = (
            "    def _init_constants(self) -> None:\n"
            f"{buffer_init_block}\n\n"
        )
    init_constants_call = "        self._init_constants()\n" if len(buffer_init_lines) > 0 else ""
    forward_kwargs_lines: List[str] = []
    forward_args_lines: List[str] = []
    forward_signature_args: List[str] = []
    for input_index, input_name in enumerate(model_ir.inputs):
        input_var = tensor_var_names[str(input_name)]
        forward_signature_args.append(f"{input_var}: torch.Tensor")
        forward_kwargs_lines.append(
            f"            {input_var} = resolve_named_input_value(kwargs, {str(input_name)!r})"
        )
        forward_args_lines.append(
            f"            {input_var} = args[{input_index}]"
        )

    def _extract_statement_assignments(statement: ast.stmt) -> List[str]:
        names: List[str] = []

        def _walk_target(target: ast.expr) -> None:
            if isinstance(target, ast.Name):
                names.append(str(target.id))
                return
            if isinstance(target, (ast.Tuple, ast.List)):
                for item in target.elts:
                    _walk_target(item)

        if isinstance(statement, ast.Assign):
            for target in statement.targets:
                _walk_target(target)
        elif isinstance(statement, ast.AnnAssign):
            _walk_target(statement.target)
        return names

    def _extract_statement_loads(statement: ast.stmt) -> List[str]:
        names: List[str] = []
        seen: Set[str] = set()
        for node in ast.walk(statement):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) and str(node.id) not in seen:
                seen.add(str(node.id))
                names.append(str(node.id))
        return names

    def _build_forward_stage_methods(
        lines: Sequence[str],
    ) -> Tuple[str, List[str], List[Dict[str, Any]]]:
        if len(lines) < 80:
            return "", [f"        {line}" for line in lines], []

        parsed_statements: List[ast.stmt] = []
        top_level_assigned_names: List[List[str]] = []
        raw_used_names: List[List[str]] = []
        for line in lines:
            statement = ast.parse(line).body[0]
            parsed_statements.append(statement)
            top_level_assigned_names.append(_extract_statement_assignments(statement))
            raw_used_names.append(_extract_statement_loads(statement))

        local_name_candidates: Set[str] = {
            str(tensor_var_names[str(name)]) for name in model_ir.inputs
        }
        local_name_candidates.update(str(tensor_var_names[str(name)]) for name in model_ir.outputs)
        for assigned_names in top_level_assigned_names:
            local_name_candidates.update(str(name) for name in assigned_names)

        assigned_names_by_line: List[List[str]] = []
        used_names_by_line: List[List[str]] = []
        for assigned_names, used_names in zip(top_level_assigned_names, raw_used_names):
            assigned_filtered = [str(name) for name in assigned_names if str(name) in local_name_candidates]
            used_filtered = [str(name) for name in used_names if str(name) in local_name_candidates]
            assigned_names_by_line.append(assigned_filtered)
            used_names_by_line.append(used_filtered)

        final_output_names = {
            str(tensor_var_names[str(name)]) for name in model_ir.outputs
        }
        stage_min_lines = 18
        stage_target_lines = 28
        stage_max_lines = 36
        stage_methods: List[str] = []
        forward_stage_calls: List[str] = []
        stage_specs: List[Dict[str, Any]] = []
        stage_index = 0
        start_index = 0
        total_lines = len(lines)

        def _chunk_io(start: int, end: int) -> Tuple[List[str], List[str]]:
            defined: Set[str] = set()
            assigned_order: List[str] = []
            inputs: List[str] = []
            seen_inputs: Set[str] = set()
            for line_index in range(start, end + 1):
                for name in used_names_by_line[line_index]:
                    if name not in defined and name not in seen_inputs:
                        seen_inputs.add(name)
                        inputs.append(name)
                for name in assigned_names_by_line[line_index]:
                    if name not in defined:
                        defined.add(name)
                        assigned_order.append(name)
            later_needed: Set[str] = set(final_output_names)
            for line_index in range(end + 1, total_lines):
                later_needed.update(used_names_by_line[line_index])
            outputs = [name for name in assigned_order if name in later_needed]
            return inputs, outputs

        def _append_stage(start: int, end: int) -> None:
            nonlocal stage_index
            stage_inputs, stage_outputs = _chunk_io(start, end)
            if len(stage_outputs) == 0:
                forward_stage_calls.extend(f"        {line}" for line in lines[start:end + 1])
                return
            method_name = f"_forward_stage_{stage_index}"
            arg_list = ", ".join(f"{name}: torch.Tensor" for name in stage_inputs)
            if len(arg_list) > 0:
                signature = f"    def {method_name}(self, {arg_list})"
            else:
                signature = f"    def {method_name}(self)"
            if len(stage_outputs) == 1:
                signature += " -> torch.Tensor:\n"
            else:
                signature += " -> tuple[" + ", ".join("torch.Tensor" for _ in stage_outputs) + "]:\n"
            stage_body = "\n".join(f"        {line}" for line in lines[start:end + 1])
            if len(stage_outputs) == 1:
                stage_return = f"        return {stage_outputs[0]}\n"
            else:
                stage_return = f"        return ({', '.join(stage_outputs)})\n"
            stage_methods.append(f"{signature}{stage_body}\n{stage_return}")

            call_args = ", ".join(stage_inputs)
            call_expr = f"self.{method_name}({call_args})" if len(call_args) > 0 else f"self.{method_name}()"
            if len(stage_outputs) == 1:
                forward_stage_calls.append(f"        {stage_outputs[0]} = {call_expr}")
            else:
                forward_stage_calls.append(f"        {', '.join(stage_outputs)} = {call_expr}")
            stage_specs.append(
                {
                    "stage_index": int(stage_index),
                    "method_name": str(method_name),
                    "inputs": list(stage_inputs),
                    "outputs": list(stage_outputs),
                }
            )
            stage_index += 1

        while start_index < total_lines:
            remaining = total_lines - start_index
            if remaining < 80 or remaining <= stage_max_lines:
                _append_stage(start_index, total_lines - 1)
                break

            candidate_min_end = start_index + stage_min_lines - 1
            candidate_max_end = min(start_index + stage_max_lines - 1, total_lines - stage_min_lines - 1)
            if candidate_min_end > candidate_max_end:
                _append_stage(start_index, total_lines - 1)
                break

            best_candidate: Optional[Tuple[int, List[str], List[str], Tuple[int, int, int]]] = None
            for end_index in range(candidate_min_end, candidate_max_end + 1):
                inputs, outputs = _chunk_io(start_index, end_index)
                if len(outputs) == 0:
                    continue
                score = (
                    len(inputs) + len(outputs),
                    abs((end_index - start_index + 1) - stage_target_lines),
                    len(outputs),
                )
                if best_candidate is None or score < best_candidate[3]:
                    best_candidate = (end_index, inputs, outputs, score)

            if best_candidate is None:
                _append_stage(start_index, total_lines - 1)
                break

            end_index, stage_inputs, stage_outputs, _ = best_candidate
            _append_stage(start_index, end_index)
            start_index = end_index + 1

        stage_methods_source = "\n".join(stage_methods)
        if len(stage_methods_source) > 0:
            stage_methods_source += "\n"
        return stage_methods_source, forward_stage_calls, stage_specs

    def _build_named_encoder_methods(
        stage_specs: Sequence[Dict[str, Any]],
        *,
        final_output_names: Set[str],
    ) -> Tuple[str, List[str], List[str]]:
        if len(stage_specs) == 0:
            return "", [], []

        def _call_line_from_spec(spec: Dict[str, Any]) -> str:
            outputs = [str(name) for name in list(spec.get("outputs", []))]
            call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
            call_expr = f"self.{spec['method_name']}({call_args})" if len(call_args) > 0 else f"self.{spec['method_name']}()"
            if len(outputs) == 1:
                return f"        {outputs[0]} = {call_expr}"
            return f"        {', '.join(outputs)} = {call_expr}"

        layer_pattern = re.compile(r"bert_encoder_layer_(\d+)", flags=re.IGNORECASE)
        default_forward_lines = [_call_line_from_spec(spec) for spec in stage_specs]
        if not any(
            any(
                layer_pattern.search(str(name)) is not None
                and "attention_self_mul_1" not in str(name)
                for name in list(spec.get("outputs", []))
            )
            for spec in stage_specs
        ):
            return "", [], default_forward_lines

        def _stage_layer_index(spec: Dict[str, Any]) -> Optional[int]:
            matches: List[int] = []
            for name in list(spec.get("outputs", [])):
                if "attention_self_mul_1" in str(name):
                    continue
                match = layer_pattern.search(str(name))
                if match is not None:
                    matches.append(int(match.group(1)))
            if len(matches) == 0:
                return None
            return min(matches)

        grouped_ranges: List[Tuple[int, int, int]] = []
        start_spec_index: Optional[int] = None
        active_layer_index: Optional[int] = None
        for spec_index, spec in enumerate(stage_specs):
            layer_index = _stage_layer_index(spec)
            if layer_index is None:
                if start_spec_index is not None and active_layer_index is not None:
                    grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(spec_index - 1)))
                    start_spec_index = None
                    active_layer_index = None
                continue
            if start_spec_index is None:
                start_spec_index = int(spec_index)
                active_layer_index = int(layer_index)
                continue
            if active_layer_index is None:
                active_layer_index = int(layer_index)
                continue
            if int(layer_index) != int(active_layer_index):
                grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(spec_index - 1)))
                start_spec_index = int(spec_index)
                active_layer_index = int(layer_index)
        if start_spec_index is not None and active_layer_index is not None:
            grouped_ranges.append((int(active_layer_index), int(start_spec_index), int(len(stage_specs) - 1)))

        if len(grouped_ranges) == 0:
            return "", [], default_forward_lines

        class_chunks: List[str] = []
        init_lines: List[str] = []
        forward_lines_local: List[str] = []
        previous_end = 0

        def _group_io(start_idx: int, end_idx: int) -> Tuple[List[str], List[str]]:
            defined: Set[str] = set()
            assigned_order: List[str] = []
            seen_inputs: Set[str] = set()
            method_inputs: List[str] = []
            for spec_index in range(start_idx, end_idx + 1):
                spec = stage_specs[spec_index]
                for name in list(spec.get("inputs", [])):
                    normalized = str(name)
                    if normalized not in defined and normalized not in seen_inputs:
                        seen_inputs.add(normalized)
                        method_inputs.append(normalized)
                for name in list(spec.get("outputs", [])):
                    normalized = str(name)
                    if normalized not in defined:
                        defined.add(normalized)
                        assigned_order.append(normalized)
            later_needed = set(final_output_names)
            for spec_index in range(end_idx + 1, len(stage_specs)):
                later_needed.update(str(name) for name in list(stage_specs[spec_index].get("inputs", [])))
            method_outputs = [name for name in assigned_order if name in later_needed]
            return method_inputs, method_outputs

        def _emit_submodule_class_from_stage_range(
            *,
            class_name: str,
            start_idx: int,
            end_idx: int,
        ) -> Optional[Tuple[str, List[str], List[str], List[str]]]:
            method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
            if len(method_outputs) == 0:
                return None
            class_stage_specs = list(stage_specs[start_idx:end_idx + 1])
            init_signature_lines = [
                f"class {class_name}(torch.nn.Module):",
                "    def __init__(",
                "        self,",
                "        *,",
            ]
            init_body_lines = ["        super().__init__()"]
            for spec in class_stage_specs:
                stage_name = str(spec["method_name"])
                init_signature_lines.append(f"        {stage_name}: Callable[..., Any],")
                init_body_lines.append(f"        self.{stage_name} = {stage_name}")
            init_signature_lines.append("    ) -> None:")
            arg_list = ", ".join(f"{name}: torch.Tensor" for name in method_inputs)
            signature = "    def forward(self"
            if len(arg_list) > 0:
                signature += f", {arg_list}"
            signature += ")"
            if len(method_outputs) == 1:
                signature += " -> torch.Tensor:\n"
            else:
                signature += " -> tuple[" + ", ".join("torch.Tensor" for _ in method_outputs) + "]:\n"
            method_body_lines: List[str] = []
            init_call_lines: List[str] = []
            for spec in class_stage_specs:
                outputs = [str(name) for name in list(spec.get("outputs", []))]
                call_args = ", ".join(str(name) for name in list(spec.get("inputs", [])))
                call_expr = f"self.{spec['method_name']}({call_args})" if len(call_args) > 0 else f"self.{spec['method_name']}()"
                if len(outputs) == 1:
                    method_body_lines.append(f"        {outputs[0]} = {call_expr}")
                else:
                    method_body_lines.append(f"        {', '.join(outputs)} = {call_expr}")
                init_call_lines.append(f"            {spec['method_name']}=self.{spec['method_name']},")
            if len(method_outputs) == 1:
                return_line = f"        return {method_outputs[0]}\n"
            else:
                return_line = f"        return ({', '.join(method_outputs)})\n"
            class_source = (
                "\n".join(init_signature_lines)
                + "\n"
                + "\n".join(init_body_lines)
                + "\n\n"
                + f"{signature}"
                + "\n".join(method_body_lines)
                + "\n"
                + return_line
            )
            return class_source, method_inputs, method_outputs, init_call_lines

        for layer_index, start_idx, end_idx in grouped_ranges:
            while previous_end < start_idx:
                spec = stage_specs[previous_end]
                forward_lines_local.append(_call_line_from_spec(spec))
                previous_end += 1

            method_inputs, method_outputs = _group_io(int(start_idx), int(end_idx))
            if len(method_outputs) == 0:
                for spec_index in range(start_idx, end_idx + 1):
                    spec = stage_specs[spec_index]
                    forward_lines_local.append(_call_line_from_spec(spec))
                previous_end = int(end_idx + 1)
                continue

            method_name = f"_forward_encoder_layer_{int(layer_index)}"
            layer_prefix = f"bert_encoder_layer_{int(layer_index)}_"
            split_idx: Optional[int] = None
            for spec_index in range(start_idx, end_idx + 1):
                output_names = [str(name) for name in list(stage_specs[spec_index].get("outputs", []))]
                if any(
                    output_name.startswith(layer_prefix + marker)
                    for output_name in output_names
                    for marker in ("ffn_", "output_", "output_bottleneck_")
                ):
                    split_idx = int(spec_index)
                    break

            composite_init_lines: Optional[List[str]] = None
            layer_attr_name = f"encoder_layer_{int(layer_index)}"
            if split_idx is not None and int(split_idx) > int(start_idx):
                attention_class_name = f"_GeneratedEncoderLayer{int(layer_index)}Attention"
                ffn_class_name = f"_GeneratedEncoderLayer{int(layer_index)}FFN"
                layer_class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
                attention_emitted = _emit_submodule_class_from_stage_range(
                    class_name=attention_class_name,
                    start_idx=int(start_idx),
                    end_idx=int(split_idx - 1),
                )
                ffn_emitted = _emit_submodule_class_from_stage_range(
                    class_name=ffn_class_name,
                    start_idx=int(split_idx),
                    end_idx=int(end_idx),
                )
                if attention_emitted is not None and ffn_emitted is not None:
                    attention_source, attention_inputs, attention_outputs, attention_init_call_lines = attention_emitted
                    ffn_source, ffn_inputs, ffn_outputs, ffn_init_call_lines = ffn_emitted
                    class_chunks.append(attention_source)
                    class_chunks.append(ffn_source)
                    class_chunks.append(
                        "class {layer_class_name}(torch.nn.Module):\n"
                        "    def __init__(self, *, attention: torch.nn.Module, ffn: torch.nn.Module) -> None:\n"
                        "        super().__init__()\n"
                        "        self.attention = attention\n"
                        "        self.ffn = ffn\n\n"
                        "    def forward({signature_args}){signature_return}"
                        "{body}"
                        "{return_line}".format(
                            layer_class_name=layer_class_name,
                            signature_args=(
                                "self"
                                + (", " + ", ".join(f"{name}: torch.Tensor" for name in method_inputs) if len(method_inputs) > 0 else "")
                            ),
                            signature_return=(
                                " -> torch.Tensor:\n"
                                if len(method_outputs) == 1 else
                                " -> tuple[" + ", ".join("torch.Tensor" for _ in method_outputs) + "]:\n"
                            ),
                            body="".join(
                                [
                                    (
                                        f"        {attention_outputs[0]} = self.attention({', '.join(attention_inputs)})\n"
                                        if len(attention_outputs) == 1 else
                                        f"        {', '.join(attention_outputs)} = self.attention({', '.join(attention_inputs)})\n"
                                    ),
                                    (
                                        f"        {ffn_outputs[0]} = self.ffn({', '.join(ffn_inputs)})\n"
                                        if len(ffn_outputs) == 1 else
                                        f"        {', '.join(ffn_outputs)} = self.ffn({', '.join(ffn_inputs)})\n"
                                    ),
                                ]
                            ),
                            return_line=(
                                f"        return {method_outputs[0]}\n"
                                if len(method_outputs) == 1 else
                                f"        return ({', '.join(method_outputs)})\n"
                            ),
                        )
                    )
                    composite_init_lines = [
                        f"self.{layer_attr_name} = {layer_class_name}(",
                        f"    attention={attention_class_name}(",
                        *attention_init_call_lines,
                        "    ),",
                        f"    ffn={ffn_class_name}(",
                        *ffn_init_call_lines,
                        "    ),",
                        ")",
                    ]

            if composite_init_lines is None:
                layer_class_name = f"_GeneratedEncoderLayer{int(layer_index)}"
                emitted = _emit_submodule_class_from_stage_range(
                    class_name=layer_class_name,
                    start_idx=int(start_idx),
                    end_idx=int(end_idx),
                )
                if emitted is None:
                    for spec_index in range(start_idx, end_idx + 1):
                        spec = stage_specs[spec_index]
                        forward_lines_local.append(_call_line_from_spec(spec))
                    previous_end = int(end_idx + 1)
                    continue
                layer_source, _, _, layer_init_call_lines = emitted
                class_chunks.append(layer_source)
                composite_init_lines = [
                    f"self.{layer_attr_name} = {layer_class_name}(",
                    *layer_init_call_lines,
                    ")",
                ]
            if composite_init_lines is not None:
                init_lines.extend(composite_init_lines)

            call_args = ", ".join(method_inputs)
            call_expr = f"self.{layer_attr_name}({call_args})" if len(call_args) > 0 else f"self.{layer_attr_name}()"
            if len(method_outputs) == 1:
                forward_lines_local.append(f"        {method_outputs[0]} = {call_expr}")
            else:
                forward_lines_local.append(f"        {', '.join(method_outputs)} = {call_expr}")
            previous_end = int(end_idx + 1)

        while previous_end < len(stage_specs):
            spec = stage_specs[previous_end]
            forward_lines_local.append(_call_line_from_spec(spec))
            previous_end += 1

        named_class_source = "\n".join(class_chunks)
        if len(named_class_source) > 0:
            named_class_source += "\n"
        return named_class_source, init_lines, forward_lines_local

    stage_methods_source, forward_stage_calls, stage_specs = _build_forward_stage_methods(forward_lines)
    named_encoder_class_source = ""
    named_encoder_init_lines: List[str] = []
    if len(stage_specs) > 0:
        named_encoder_class_source, named_encoder_init_lines, forward_stage_calls = _build_named_encoder_methods(
            stage_specs,
            final_output_names={str(tensor_var_names[str(name)]) for name in model_ir.outputs},
        )
    if len(named_encoder_init_lines) > 0:
        module_init_lines.extend(named_encoder_init_lines)
    module_init_block = "\n".join(f"        {line}" for line in module_init_lines)
    forward_block = "\n".join(forward_stage_calls)
    forward_kwargs_block = "\n".join(forward_kwargs_lines) if len(forward_kwargs_lines) > 0 else "            pass"
    forward_args_block = "\n".join(forward_args_lines) if len(forward_args_lines) > 0 else "            pass"
    forward_signature = ", ".join(["self"] + forward_signature_args)
    forward_named_call_args = ", ".join(
        str(tensor_var_names[str(input_name)])
        for input_name in model_ir.inputs
    )
    outputs_expr = ", ".join(_tensor_expr(str(name)) for name in model_ir.outputs)
    has_conv_blocks = len(fused_module_specs) > 0
    runtime_import_order = [
        "_align_binary_inputs",
        "_align_binary_inputs_to_anchor",
        "_align_tensor_to_target_shape",
        "_apply_concat",
        "_apply_fused_activation",
        "_apply_gather",
        "_apply_gather_elements",
        "_apply_gather_nd",
        "_apply_scatter_nd",
        "_apply_module_conv2d",
        "_apply_module_conv3d",
        "_apply_module_transpose_conv2d",
        "_apply_module_transpose_conv3d",
        "_apply_non_max_suppression_v4",
        "_apply_pool2d",
        "_apply_pad_nd",
        "_apply_resize",
        "_apply_slice",
        "_apply_softmax",
        "_apply_strided_slice",
        "_coerce_scalar_axis",
        "_normalize_axes",
        "_normalize_dim",
        "_reduce_any",
        "_reduce_max",
        "_reduce_mean",
        "_reduce_min",
        "_reduce_prod",
        "_reduce_sum",
        "_resolve_reshape_shape",
        "resolve_named_input_value",
        "_shape_list",
        "_to_torch_pad_arg",
        "_torch_dtype",
        "_torch_permute",
        "load_generated_weights",
    ]
    runtime_import_block = "".join(
        f"    {name},\n" for name in runtime_import_order if name in runtime_imports
    )
    conv_block_source = (
        "class _Conv2dBlock(torch.nn.Module):\n"
        "    def __init__(\n"
        "        self,\n"
        "        *,\n"
        "        in_channels: int,\n"
        "        out_channels: int,\n"
        "        kernel_size: tuple[int, int],\n"
        "        stride: tuple[int, int],\n"
        "        padding: tuple[int, int],\n"
        "        dilation: tuple[int, int],\n"
        "        groups: int,\n"
        "        bias: bool,\n"
        "        pad: Optional[list[int]] = None,\n"
        "        activation: str = 'none',\n"
        "        negative_slope: float = 0.2,\n"
        "        pad_mode: str = 'constant',\n"
        "        pad_value: float = 0.0,\n"
        "    ) -> None:\n"
        "        super().__init__()\n"
        "        self.conv = torch.nn.Conv2d(\n"
        "            in_channels=in_channels,\n"
        "            out_channels=out_channels,\n"
        "            kernel_size=kernel_size,\n"
        "            stride=stride,\n"
        "            padding=padding,\n"
        "            dilation=dilation,\n"
        "            groups=groups,\n"
        "            bias=bias,\n"
        "        )\n"
        "        self.pad = pad\n"
        "        self.activation = str(activation)\n"
        "        self.negative_slope = float(negative_slope)\n"
        "        self.pad_mode = str(pad_mode)\n"
        "        self.pad_value = float(pad_value)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        if self.pad is not None:\n"
        "            x = F.pad(x, self.pad, mode=self.pad_mode, value=self.pad_value)\n"
        "        x = self.conv(x)\n"
        "        if self.activation == 'leaky_relu':\n"
        "            return F.leaky_relu(x, negative_slope=self.negative_slope)\n"
        "        if self.activation == 'relu':\n"
        "            return torch.relu(x)\n"
        "        if self.activation == 'relu6':\n"
        "            return torch.clamp(x, min=0.0, max=6.0)\n"
        "        if self.activation == 'relu_n1_to_1':\n"
        "            return torch.clamp(x, min=-1.0, max=1.0)\n"
        "        if self.activation == 'relu_0_to_1':\n"
        "            return torch.clamp(x, min=0.0, max=1.0)\n"
        "        if self.activation == 'tanh':\n"
        "            return torch.tanh(x)\n"
        "        if self.activation == 'sigmoid':\n"
        "            return torch.sigmoid(x)\n"
        "        return x\n\n"
    ) if has_conv_blocks else ""
    has_affine_layer_norms = len(affine_layer_norm_specs) > 0
    affine_layer_norm_source = (
        "class _AffineLayerNorm(torch.nn.Module):\n"
        "    gamma: torch.Tensor\n"
        "    beta: torch.Tensor\n\n"
        "    def __init__(self, *, shape: list[int], dtype: torch.dtype) -> None:\n"
        "        super().__init__()\n"
        "        self.register_buffer('gamma', torch.zeros(shape, dtype=dtype), persistent=True)\n"
        "        self.register_buffer('beta', torch.zeros(shape, dtype=dtype), persistent=True)\n\n"
        "    def forward(self, x: torch.Tensor) -> torch.Tensor:\n"
        "        return torch.add(torch.mul(x, self.gamma), self.beta)\n\n"
    ) if has_affine_layer_norms else ""
    nms_method_source = ""
    if len(nms_method_specs) > 0:
        method_chunks: List[str] = []
        for spec in nms_method_specs:
            method_chunks.append(
                "    def {name}(self, boxes: torch.Tensor, scores: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:\n"
                "        return _apply_non_max_suppression_v4(\n"
                "            boxes,\n"
                "            scores,\n"
                "            torch.as_tensor({max_output_expr}, dtype=torch.int32, device=self._device()),\n"
                "            torch.as_tensor({iou_threshold_expr}, dtype=torch.float32, device=self._device()),\n"
                "            torch.as_tensor({score_threshold_expr}, dtype=torch.float32, device=self._device()),\n"
                "        )\n\n".format(
                    name=str(spec["name"]),
                    max_output_expr=str(spec["max_output_expr"]),
                    iou_threshold_expr=str(spec["iou_threshold_expr"]),
                    score_threshold_expr=str(spec["score_threshold_expr"]),
                )
            )
        nms_method_source = "".join(method_chunks)

    model_source = (
        "from pathlib import Path\n"
        "from typing import Any, Callable, Dict, Optional\n\n"
        "import torch\n"
        "import torch.nn.functional as F\n\n"
        "from .runtime import (\n"
        f"{runtime_import_block}"
        ")\n\n"
        "PACKAGE_DIR = Path(__file__).resolve().parent\n"
        f"INPUT_NAMES = {repr([str(v) for v in model_ir.inputs])}\n"
        f"OUTPUT_NAMES = {repr([str(v) for v in model_ir.outputs])}\n"
        f"{conv_block_source}"
        f"{affine_layer_norm_source}"
        f"{named_encoder_class_source}"
        "class Model(torch.nn.Module):\n"
        f"{buffer_annotation_block}"
        "    def __init__(self, *, device: str | None = None, eval_mode: bool = True, load_weights: bool = True):\n"
        "        super().__init__()\n"
        "        self.input_names = list(INPUT_NAMES)\n"
        "        self.output_names = list(OUTPUT_NAMES)\n"
        f"{module_init_block}\n"
        f"{init_constants_call}"
        "        if load_weights:\n"
        "            load_generated_weights(\n"
        "                model=self,\n"
        "                package_dir=PACKAGE_DIR,\n"
        "                device=device,\n"
        "            )\n"
        "        elif device is not None:\n"
        "            self.to(device)\n"
        "        if eval_mode:\n"
        "            self.eval()\n\n"
        f"{init_constants_method}"
        "    def _device(self) -> torch.device:\n"
        "        for parameter in self.parameters():\n"
        "            return parameter.device\n"
        "        for buffer in self.buffers():\n"
        "            return buffer.device\n"
        "        return torch.device('cpu')\n\n"
        "    def _max_pool2d_same(self, x: torch.Tensor, *, kernel_size: tuple[int, int], stride: tuple[int, int]) -> torch.Tensor:\n"
        "        use_channel_last = bool(\n"
        "            x.ndim == 4\n"
        "            and int(x.shape[1]) > int(x.shape[-1])\n"
        "            and int(x.shape[2]) > int(x.shape[-1])\n"
        "        )\n"
        "        pool_input = x.permute(0, 3, 1, 2).contiguous() if use_channel_last else x\n"
        "        in_h = int(pool_input.shape[-2])\n"
        "        in_w = int(pool_input.shape[-1])\n"
        "        out_h = (in_h + int(stride[0]) - 1) // int(stride[0])\n"
        "        out_w = (in_w + int(stride[1]) - 1) // int(stride[1])\n"
        "        pad_h_total = max((out_h - 1) * int(stride[0]) + int(kernel_size[0]) - in_h, 0)\n"
        "        pad_w_total = max((out_w - 1) * int(stride[1]) + int(kernel_size[1]) - in_w, 0)\n"
        "        pad_top = pad_h_total // 2\n"
        "        pad_bottom = pad_h_total - pad_top\n"
        "        pad_left = pad_w_total // 2\n"
        "        pad_right = pad_w_total - pad_left\n"
        "        pool_input = F.pad(pool_input, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=float('-inf'))\n"
        "        y = F.max_pool2d(pool_input, kernel_size=kernel_size, stride=stride)\n"
        "        return y.permute(0, 2, 3, 1).contiguous() if use_channel_last else y\n\n"
        "    def _avg_pool2d_same(self, x: torch.Tensor, *, kernel_size: tuple[int, int], stride: tuple[int, int]) -> torch.Tensor:\n"
        "        use_channel_last = bool(\n"
        "            x.ndim == 4\n"
        "            and int(x.shape[1]) > int(x.shape[-1])\n"
        "            and int(x.shape[2]) > int(x.shape[-1])\n"
        "        )\n"
        "        pool_input = x.permute(0, 3, 1, 2).contiguous() if use_channel_last else x\n"
        "        in_h = int(pool_input.shape[-2])\n"
        "        in_w = int(pool_input.shape[-1])\n"
        "        out_h = (in_h + int(stride[0]) - 1) // int(stride[0])\n"
        "        out_w = (in_w + int(stride[1]) - 1) // int(stride[1])\n"
        "        pad_h_total = max((out_h - 1) * int(stride[0]) + int(kernel_size[0]) - in_h, 0)\n"
        "        pad_w_total = max((out_w - 1) * int(stride[1]) + int(kernel_size[1]) - in_w, 0)\n"
        "        pad_top = pad_h_total // 2\n"
        "        pad_bottom = pad_h_total - pad_top\n"
        "        pad_left = pad_w_total // 2\n"
        "        pad_right = pad_w_total - pad_left\n"
        "        pool_input = F.pad(pool_input, [pad_left, pad_right, pad_top, pad_bottom], mode='constant', value=0.0)\n"
        "        y = F.avg_pool2d(pool_input, kernel_size=kernel_size, stride=stride)\n"
        "        return y.permute(0, 2, 3, 1).contiguous() if use_channel_last else y\n\n"
        f"{nms_method_source}"
        f"{stage_methods_source}"
        f"    def forward({forward_signature}) -> Any:\n"
        f"{forward_block}\n"
    )
    if len(model_ir.outputs) == 1:
        model_source += (
            f"        return {outputs_expr}\n\n"
            "    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:\n"
            "        if len(args) > 0 and len(kwargs) > 0:\n"
            "            raise RuntimeError('Use either positional inputs or keyword inputs, not both.')\n"
            "        if len(kwargs) > 0:\n"
            f"{forward_kwargs_block}\n"
            "        else:\n"
            f"            if len(args) != {len(model_ir.inputs)}:\n"
            "                raise RuntimeError(f'Input arity mismatch. expected={len(self.input_names)} actual={len(args)}')\n"
            f"{forward_args_block}\n"
            f"        return {{{str(model_ir.outputs[0])!r}: self.forward({forward_named_call_args})}}\n\n"
        )
    else:
        model_source += (
            f"        return ({outputs_expr})\n\n"
            "    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:\n"
            "        if len(args) > 0 and len(kwargs) > 0:\n"
            "            raise RuntimeError('Use either positional inputs or keyword inputs, not both.')\n"
            "        if len(kwargs) > 0:\n"
            f"{forward_kwargs_block}\n"
            "        else:\n"
            f"            if len(args) != {len(model_ir.inputs)}:\n"
            "                raise RuntimeError(f'Input arity mismatch. expected={len(self.input_names)} actual={len(args)}')\n"
            f"{forward_args_block}\n"
            f"        result = self.forward({forward_named_call_args})\n"
            f"        return {{name: value for name, value in zip({repr([str(v) for v in model_ir.outputs])}, result)}}\n\n"
        )
    model_source += (
        "def load_model(device: str | None = None, eval_mode: bool = True) -> Model:\n"
        "    return Model(device=device, eval_mode=eval_mode)\n"
    )
    (package_dir / "model.py").write_text(model_source, encoding="utf-8")
    return [(str(attr_path), str(tensor_name)) for attr_path, tensor_name in load_specs]


def export_pytorch_package_from_tflite_artifact(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    tflite_file_path: str,
) -> str:
    if not os.path.exists(tflite_file_path):
        raise ModelIRPyTorchExportError(
            f"TFLite-backed PyTorch package export requires an existing float32 TFLite file. path={tflite_file_path}"
        )

    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)

    package_tflite_name = "model_float32.tflite"
    shutil.copyfile(
        str(tflite_file_path),
        os.path.join(output_folder_path, package_tflite_name),
    )
    metadata = _build_tflite_backed_metadata_payload(
        model_ir=model_ir,
        tflite_file_name=package_tflite_name,
    )
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)


def export_pytorch_package_from_saved_model_artifact(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    saved_model_path: str,
) -> str:
    if not os.path.exists(saved_model_path):
        raise ModelIRPyTorchExportError(
            f"SavedModel-backed PyTorch package export requires an existing SavedModel directory. path={saved_model_path}"
        )

    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)

    package_saved_model_dir = os.path.join(output_folder_path, "saved_model")
    if os.path.exists(package_saved_model_dir):
        shutil.rmtree(package_saved_model_dir)
    shutil.copytree(str(saved_model_path), package_saved_model_dir)
    metadata = _build_saved_model_backed_metadata_payload(
        model_ir=model_ir,
        saved_model_dir_name="saved_model",
    )
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)


def _should_prefer_tflite_backed_package(model_ir: ModelIR) -> bool:
    op_types = [str(op.op_type) for op in model_ir.operators]
    recurrent_or_control_ops = {
        "WHILE",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if any(op_type in recurrent_or_control_ops for op_type in op_types):
        return False
    has_length_like_input = False
    for input_name in model_ir.inputs:
        canonical = re.sub(r"[^0-9a-z]+", "_", str(input_name).lower()).strip("_")
        if canonical.endswith(("length", "lengths", "len", "lens", "seq_len", "seq_lens")):
            has_length_like_input = True
            break
    if has_length_like_input:
        return False
    if any(op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} for op_type in op_types):
        return True
    for op in model_ir.operators:
        if str(op.op_type) != "SOFTMAX" or len(op.inputs) == 0:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        if input_tensor is None:
            continue
        if is_channel_first_logical_layout(normalize_logical_layout(input_tensor.logical_layout)):
            return True
    conv_like_count = sum(
        1
        for op_type in op_types
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
    )
    strided_slice_count = sum(1 for op_type in op_types if op_type == "STRIDED_SLICE")
    concat_count = sum(1 for op_type in op_types if op_type == "CONCATENATION")
    resize_count = sum(
        1
        for op_type in op_types
        if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}
    )
    split_count = sum(1 for op_type in op_types if op_type == "SPLIT")
    softmax_count = sum(1 for op_type in op_types if op_type == "SOFTMAX")
    nhwc_named_tensor_count = sum(
        1
        for tensor_name in model_ir.tensors.keys()
        if str(tensor_name).lower().endswith(("_nhwc", "_nwc", "_ndhwc"))
    )
    has_rank3_channel_first_output = any(
        len(list(model_ir.tensors[str(output_name)].shape)) == 3
        and normalize_logical_layout(model_ir.tensors[str(output_name)].logical_layout) == "NCW"
        for output_name in model_ir.outputs
        if str(output_name) in model_ir.tensors
    )
    if (
        has_rank3_channel_first_output
        and conv_like_count >= 20
        and strided_slice_count >= 4
        and concat_count >= 4
    ):
        return True
    if (
        conv_like_count >= 40
        and nhwc_named_tensor_count >= 40
        and (resize_count >= 4 or softmax_count >= 1 or split_count >= 1)
    ):
        return True
    if (
        conv_like_count >= 60
        and nhwc_named_tensor_count >= 80
        and resize_count >= 2
    ):
        return True
    if (
        conv_like_count >= 15
        and nhwc_named_tensor_count >= 30
        and resize_count >= 3
    ):
        return True
    return False


def _should_prefer_saved_model_backed_package(model_ir: ModelIR) -> bool:
    return _should_prefer_tflite_backed_package(model_ir)


def _merge_reference_public_boundary_metadata(
    *,
    imported_model_ir: ModelIR,
    reference_model_ir: Optional[ModelIR],
    reference_onnx_graph: Optional[Any] = None,
) -> None:
    if reference_model_ir is None:
        return
    imported_model_ir.inputs = [str(v) for v in list(reference_model_ir.inputs)]
    imported_model_ir.outputs = [str(v) for v in list(reference_model_ir.outputs)]
    boundary_shape_map = reference_model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = reference_model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    recurrent_public_boundary_context = any(
        token in str(op.op_type)
        for op in reference_model_ir.operators
        for token in ("GRU", "LSTM", "RNN")
    )
    if not recurrent_public_boundary_context and reference_onnx_graph is not None:
        graph = getattr(reference_onnx_graph, "graph", None)
        if graph is not None:
            recurrent_public_boundary_context = any(
                str(node.op_type) in {"GRU", "LSTM", "RNN"}
                for node in list(graph.node)
            )
    for tensor_name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs):
        ref_tensor = reference_model_ir.tensors.get(str(tensor_name), None)
        imported_tensor = imported_model_ir.tensors.get(str(tensor_name), None)
        if ref_tensor is None or imported_tensor is None:
            continue
        imported_tensor.shape_signature = [int(v) for v in list(ref_tensor.shape_signature or ref_tensor.shape)]
        inferred_public_layout = normalize_logical_layout(ref_tensor.logical_layout)
        if recurrent_public_boundary_context and len(list(ref_tensor.shape)) == 3:
            inferred_public_layout = "NWC"
        imported_tensor.logical_layout = inferred_public_layout
    imported_model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        str(name): [int(v) for v in list(boundary_shape_map.get(str(name), reference_model_ir.tensors[str(name)].shape))]
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }
    imported_model_ir.metadata["onnx_public_layout_map"] = {
        str(name): (
            "NWC"
            if recurrent_public_boundary_context and len(list(reference_model_ir.tensors[str(name)].shape)) == 3
            else normalize_logical_layout(
                public_layout_map.get(
                    str(name),
                    reference_model_ir.tensors[str(name)].logical_layout,
                )
            )
        )
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }


def _try_export_native_package_from_tflite_import(
    *,
    output_folder_path: str,
    fallback_tflite_path: str,
    reference_model_ir: Optional[ModelIR] = None,
    reference_onnx_graph: Optional[Any] = None,
) -> Optional[str]:
    if not os.path.exists(str(fallback_tflite_path)):
        return None
    imported_model_ir = import_model_ir_from_tflite(
        tflite_file_path=str(fallback_tflite_path),
    )
    _merge_reference_public_boundary_metadata(
        imported_model_ir=imported_model_ir,
        reference_model_ir=reference_model_ir,
        reference_onnx_graph=reference_onnx_graph,
    )
    return export_pytorch_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=output_folder_path,
        fallback_tflite_path=None,
        fallback_onnx_graph=None,
        fallback_saved_model_path=None,
        fallback_tflite_has_custom_ops=False,
    )


def _try_export_runtime_wrapper_package_from_tflite_import(
    *,
    output_folder_path: str,
    fallback_tflite_path: str,
) -> Optional[str]:
    if not os.path.exists(str(fallback_tflite_path)):
        return None
    imported_model_ir = import_model_ir_from_tflite(
        tflite_file_path=str(fallback_tflite_path),
    )
    if not _supports_runtime_wrapper_model_ir(imported_model_ir):
        return None
    return _export_runtime_wrapper_package_from_model_ir(
        model_ir=imported_model_ir,
        output_folder_path=output_folder_path,
    )


def export_pytorch_package_from_model_ir(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    fallback_tflite_path: Optional[str] = None,
    fallback_onnx_graph: Optional[Any] = None,
    fallback_saved_model_path: Optional[str] = None,
    fallback_saved_model_factory: Optional[Callable[[], Optional[str]]] = None,
    fallback_tflite_has_custom_ops: bool = False,
) -> str:
    try:
        import torch
    except Exception as ex:
        raise ModelIRPyTorchExportError(
            "PyTorch export requires `torch` to be installed."
        ) from ex

    resolved_fallback_saved_model_path = (
        str(fallback_saved_model_path)
        if fallback_saved_model_path is not None and str(fallback_saved_model_path).strip() != ""
        else None
    )

    def _get_fallback_saved_model_path() -> Optional[str]:
        nonlocal resolved_fallback_saved_model_path
        if resolved_fallback_saved_model_path is not None:
            return resolved_fallback_saved_model_path
        if fallback_saved_model_factory is None:
            return None
        try:
            generated_path = fallback_saved_model_factory()
        except Exception:
            return None
        if generated_path is None or str(generated_path).strip() == "":
            return None
        resolved_fallback_saved_model_path = str(generated_path)
        return resolved_fallback_saved_model_path

    try:
        normalized: Optional[ModelIR] = None
        native_prep_error: Optional[Exception] = None

        try:
            normalized = prepare_model_ir_for_native_pytorch(model_ir)
            _ensure_no_custom_ops(normalized)
            _ensure_supported_ops(normalized)
        except Exception as ex:
            normalized = None
            native_prep_error = ex
        if (
            normalized is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and not bool(fallback_tflite_has_custom_ops)
        ):
            try:
                imported_native_package_path = _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
                if imported_native_package_path is not None:
                    return imported_native_package_path
            except Exception:
                pass
        fallback_saved_model_path_for_export = _get_fallback_saved_model_path() if normalized is None else resolved_fallback_saved_model_path
        if (
            normalized is None
            and fallback_saved_model_path_for_export is not None
            and _should_prefer_saved_model_backed_package(model_ir)
        ):
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path_for_export),
            )
        if (
            normalized is None
            and fallback_saved_model_path is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and not bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            return export_pytorch_package_from_tflite_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                tflite_file_path=str(fallback_tflite_path),
            )
        if (
            normalized is None
            and fallback_tflite_path is not None
            and str(fallback_tflite_path).strip() != ""
            and bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            runtime_wrapper_path = _try_export_runtime_wrapper_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
            )
            if runtime_wrapper_path is not None:
                return runtime_wrapper_path

        if normalized is None:
            if native_prep_error is not None:
                raise native_prep_error
            raise ModelIRPyTorchExportError(
                "Native PyTorch export preparation failed for an unknown reason."
            )
        tensor_storage_name_map = _make_tensor_storage_name_map(normalized)

        os.makedirs(output_folder_path, exist_ok=True)
        metadata = _build_metadata_payload(normalized)
        metadata["execution_backend"] = "native"
        metadata["tensor_storage_names"] = dict(tensor_storage_name_map)
        native_load_specs: Optional[List[Tuple[str, str]]] = None
        try:
            native_load_specs = _write_native_model_file(
                output_folder_path,
                model_ir=normalized,
                metadata=metadata,
                tensor_storage_name_map=tensor_storage_name_map,
            )
        except ModelIRPyTorchExportError as ex:
            if not _is_direct_codegen_unsupported_error(ex):
                raise
            # Keep torch-kernel-backed packages native when runtime kernels
            # support the graph, even if direct Python codegen does not yet.
            _write_generated_package_common_files(output_folder_path)
            _write_wrapper_model_file(output_folder_path)
            metadata["execution_backend"] = "runtime_wrapper"
        metadata_path = os.path.join(output_folder_path, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        if native_load_specs is not None:
            state_dict = _build_native_generated_state_dict(
                package_path=output_folder_path,
                model_ir=normalized,
                load_specs=native_load_specs,
            )
        else:
            state_dict = {}
            for tensor_name, tensor in normalized.tensors.items():
                if not isinstance(tensor.data, np.ndarray):
                    continue
                dtype_name = str(tensor.dtype).upper()
                if dtype_name not in {"BOOL", "INT8", "INT16", "INT32", "INT64", "UINT8", "FLOAT16", "FLOAT32", "FLOAT64"}:
                    raise ModelIRPyTorchExportError(
                        f"Unsupported tensor dtype for PyTorch export: tensor={tensor_name} dtype={tensor.dtype}"
                    )
                storage_name = tensor_storage_name_map.get(str(tensor_name), str(tensor_name))
                state_dict[storage_name] = torch.as_tensor(np.asarray(tensor.data))
        torch.save(state_dict, os.path.join(output_folder_path, "state_dict.pth"))
        return str(output_folder_path)
    except Exception:
        string_config = _extract_string_normalizer_config_from_onnx_graph(
            fallback_onnx_graph,
        )
        if string_config is not None:
            return export_pytorch_package_from_string_normalizer_onnx(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                onnx_graph=fallback_onnx_graph,
            )
        fallback_saved_model_path_for_export = _get_fallback_saved_model_path()
        if fallback_saved_model_path_for_export is not None:
            return export_pytorch_package_from_saved_model_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                saved_model_path=str(fallback_saved_model_path_for_export),
            )
        if fallback_tflite_path is None or str(fallback_tflite_path).strip() == "":
            raise
        try:
            runtime_wrapper_path = _try_export_runtime_wrapper_package_from_tflite_import(
                output_folder_path=output_folder_path,
                fallback_tflite_path=str(fallback_tflite_path),
            )
            if runtime_wrapper_path is not None:
                return runtime_wrapper_path
        except Exception:
            pass
        if not bool(fallback_tflite_has_custom_ops):
            try:
                imported_native_package_path = _try_export_native_package_from_tflite_import(
                    output_folder_path=output_folder_path,
                    fallback_tflite_path=str(fallback_tflite_path),
                    reference_model_ir=model_ir,
                    reference_onnx_graph=fallback_onnx_graph,
                )
                if imported_native_package_path is not None:
                    return imported_native_package_path
            except Exception:
                pass
        if (
            not bool(fallback_tflite_has_custom_ops)
            and _should_prefer_tflite_backed_package(model_ir)
        ):
            return export_pytorch_package_from_tflite_artifact(
                model_ir=model_ir,
                output_folder_path=output_folder_path,
                tflite_file_path=str(fallback_tflite_path),
            )
        return export_pytorch_package_from_tflite_artifact(
            model_ir=model_ir,
            output_folder_path=output_folder_path,
            tflite_file_path=str(fallback_tflite_path),
        )
