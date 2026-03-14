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


from .common import ModelIRPyTorchExportError, _clone_tensor, _collect_model_op_types, _make_unique_identifier, _perm_cf_to_cl, _perm_cl_to_cf, _permute_shape, _permute_tensor_to_channel_first_inplace, _preferred_reshape_target_values, _read_transpose_perm, _rewrite_axis_constant_inplace, _rewrite_matrix_constant_inplace, _rewrite_vector_constant_inplace

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
    if permuted_input_shape != output_shape:
        return True
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if input_layout == LOGICAL_LAYOUT_UNKNOWN or output_layout == LOGICAL_LAYOUT_UNKNOWN:
        return False
    if input_layout != output_layout:
        return False
    return False

def _is_inconsistent_same_layout_transpose(
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
    if input_shape != output_shape:
        return False
    input_layout = normalize_logical_layout(input_tensor.logical_layout)
    output_layout = normalize_logical_layout(output_tensor.logical_layout)
    if input_layout == LOGICAL_LAYOUT_UNKNOWN or output_layout == LOGICAL_LAYOUT_UNKNOWN:
        return False
    if input_layout != output_layout:
        return False
    perm_values = [int(v) for v in list(perm)]
    if perm_values == list(range(rank)):
        return False
    permuted_input_shape = _permute_shape(input_shape, perm_values)
    if permuted_input_shape is None:
        return False
    # The metadata contract says the tensor stayed in the same known layout and
    # same shape. If the recorded permutation would produce a different shape,
    # the transpose is stale and must be elided.
    return permuted_input_shape != output_shape

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
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
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

def _infer_concat_peer_layout(
    op: OperatorIR,
    input_tensors: Sequence[Optional[TensorIR]],
) -> str:
    axis = op.options.get("axis", None)
    if axis is None:
        return LOGICAL_LAYOUT_UNKNOWN
    known_layout: Optional[str] = None
    known_rank: Optional[int] = None
    reference_shape: Optional[List[int]] = None
    for tensor in input_tensors:
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if layout == LOGICAL_LAYOUT_UNKNOWN:
            continue
        if not (
            is_channel_first_logical_layout(layout)
            or is_channel_last_logical_layout(layout)
        ):
            return LOGICAL_LAYOUT_UNKNOWN
        current_shape = [int(v) for v in list(tensor.shape)]
        if known_layout is None:
            known_layout = layout
            known_rank = rank
            reference_shape = current_shape
            continue
        if layout != known_layout or rank != known_rank:
            return LOGICAL_LAYOUT_UNKNOWN
        if reference_shape is not None:
            for dim_idx, (candidate_dim, expected_dim) in enumerate(zip(current_shape, reference_shape)):
                if int(dim_idx) == int(axis):
                    continue
                if int(candidate_dim) > 0 and int(expected_dim) > 0 and int(candidate_dim) != int(expected_dim):
                    return LOGICAL_LAYOUT_UNKNOWN
    if known_layout is None or known_rank is None:
        return LOGICAL_LAYOUT_UNKNOWN
    expected_axis = 1 if is_channel_first_logical_layout(known_layout) else int(known_rank) - 1
    if int(axis) != int(expected_axis):
        return LOGICAL_LAYOUT_UNKNOWN
    return str(known_layout)

def _can_emit_direct_torch_reshape_shape(
    shape_values: Sequence[int],
    *,
    allow_zero: bool,
) -> bool:
    values = [int(v) for v in list(shape_values)]
    if values.count(-1) > 1:
        return False
    for dim_value in values:
        if dim_value == -1:
            continue
        if dim_value == 0:
            if allow_zero:
                continue
            return False
        if dim_value < 0:
            return False
    return True

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

def _is_channel_last_factorized_rank3_sequence_reshape(
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
    if len(input_shape) not in {4, 5} or len(output_shape) != 3:
        return False
    if any(int(v) <= 0 for v in input_shape + output_shape):
        return False
    if int(output_shape[0]) != int(input_shape[0]):
        return False
    input_channels = int(input_shape[-1])
    output_features = int(output_shape[-1])
    if output_features <= 0 or input_channels <= 0 or input_channels % output_features != 0:
        return False
    spatial_extent = int(np.prod(input_shape[1:-1], dtype=np.int64))
    factor = int(input_channels // output_features)
    expected_sequence_extent = int(spatial_extent * factor)
    return int(output_shape[1]) == expected_sequence_extent

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
                concat_input_tensors = [
                    model_ir.tensors.get(str(input_name), None) for input_name in op.inputs
                ]
                propagated_layout = _shared_tensor_layout(concat_input_tensors)
                if propagated_layout == LOGICAL_LAYOUT_UNKNOWN:
                    propagated_layout = _infer_concat_peer_layout(op, concat_input_tensors)
                    if propagated_layout != LOGICAL_LAYOUT_UNKNOWN:
                        for input_tensor in concat_input_tensors:
                            changed = _assign_tensor_logical_layout(input_tensor, propagated_layout) or changed
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

    def _is_time_major_recurrent_bridge(output_name: str) -> bool:
        for consumer_idx in consumers.get(str(output_name), []):
            consumer = model_ir.operators[int(consumer_idx)]
            if str(consumer.op_type) != "TRANSPOSE" or len(consumer.outputs) != 1:
                continue
            perm = _read_transpose_perm(model_ir, consumer)
            if perm != [1, 0, 2]:
                continue
            transpose_output_name = str(consumer.outputs[0])
            for next_idx in consumers.get(transpose_output_name, []):
                next_op_type = str(model_ir.operators[int(next_idx)].op_type)
                if next_op_type in {
                    "BIDIRECTIONAL_SEQUENCE_LSTM",
                    "UNIDIRECTIONAL_SEQUENCE_LSTM",
                    "UNIDIRECTIONAL_SEQUENCE_RNN",
                }:
                    return True
        return False

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

    def _trace_feature_last_passthrough_inputs(tensor_name: str) -> Set[str]:
        traced_names: Set[str] = set()
        visited: Set[str] = set()
        worklist: List[str] = [str(tensor_name)]
        passthrough_ops = {
            "AVERAGE_POOL_2D",
            "CAST",
            "EXPAND_DIMS",
            "IDENTITY",
            "LEAKY_RELU",
            "LOGISTIC",
            "MAX_POOL_2D",
            "PAD",
            "PADV2",
            "RELU",
            "RELU6",
            "RESHAPE",
            "SQUEEZE",
            "STRIDED_SLICE",
            "TRANSPOSE",
        }
        while len(worklist) > 0:
            current_name = str(worklist.pop())
            if current_name in visited:
                continue
            visited.add(current_name)
            traced_names.add(current_name)
            producer_idx = producers.get(current_name, None)
            if producer_idx is None:
                continue
            producer = model_ir.operators[int(producer_idx)]
            if str(producer.op_type) not in passthrough_ops or len(producer.inputs) == 0:
                continue
            upstream_name = str(producer.inputs[0])
            traced_names.add(upstream_name)
            worklist.append(upstream_name)
        return traced_names

    roots: Set[str] = set()
    if _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(model_ir):
        for tensor_name, tensor in model_ir.tensors.items():
            rank = len(list(tensor.shape))
            layout = normalize_logical_layout(tensor.logical_layout)
            if rank in {4, 5} and is_channel_last_logical_layout(layout):
                roots.add(str(tensor_name))
    for tensor_name, tensor in model_ir.tensors.items():
        normalized_name = str(tensor_name)
        rank = len(list(tensor.shape))
        layout = normalize_logical_layout(tensor.logical_layout)
        lowered_name = normalized_name.lower()
        if (
            rank in {3, 4, 5}
            and is_channel_last_logical_layout(layout)
            and any(token in lowered_name for token in ("_nwc", "_nhwc", "_ndhwc"))
        ):
            roots.add(normalized_name)
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
            input_tensor = model_ir.tensors.get(input_name, None)
            if output_tensor is None:
                continue
            rank = len(list(output_tensor.shape))
            if rank != 3:
                continue
            perm = _read_transpose_perm(model_ir, op)
            if (
                perm == [1, 0, 2]
                and input_tensor is not None
                and (
                    is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
                    or is_channel_last_logical_layout(normalize_logical_layout(output_tensor.logical_layout))
                )
            ):
                roots.add(input_name)
                roots.add(output_name)
                continue
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
            output_name in set(str(v) for v in model_ir.outputs)
            and input_tensor is not None
            and len(list(input_tensor.shape)) >= rank
            and len(list(output_tensor.shape)) >= 1
            and len(list(input_tensor.shape)) >= 1
            and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
        ):
            roots.add(output_name)
            continue
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
        if _is_channel_last_factorized_rank3_sequence_reshape(input_tensor, output_tensor):
            roots.add(output_name)
            continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) == 3:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
                and is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout))
            ):
                for consumer_idx in consumers.get(output_name, []):
                    consumer = model_ir.operators[int(consumer_idx)]
                    if (
                        str(consumer.op_type) != "BATCH_MATMUL"
                        or len(consumer.inputs) < 2
                        or str(consumer.inputs[0]) != output_name
                        or not bool(consumer.options.get("adjX", False))
                    ):
                        continue
                    rhs_tensor = model_ir.tensors.get(str(consumer.inputs[1]), None)
                    if rhs_tensor is None or len(list(rhs_tensor.shape)) < 2:
                        continue
                    rhs_contract = int(list(rhs_tensor.shape)[-2])
                    if rhs_contract != int(input_shape[-1]):
                        continue
                    roots.add(output_name)
                    break
                if output_name in roots:
                    continue
        if input_tensor is not None and rank == 3 and len(list(input_tensor.shape)) in {4, 5}:
            input_shape = [int(v) for v in list(input_tensor.shape)]
            output_shape = [int(v) for v in list(output_tensor.shape)]
            if (
                int(np.prod(input_shape, dtype=np.int64)) == int(np.prod(output_shape, dtype=np.int64))
                and _is_time_major_recurrent_bridge(output_name)
            ):
                roots.add(output_name)
                input_name = str(op.inputs[0])
                roots.add(input_name)
                producer: Optional[OperatorIR] = None
                producer_output_name = ""
                producer_rank = -1
                producer_idx = producers.get(input_name, None)
                if producer_idx is not None:
                    producer = model_ir.operators[int(producer_idx)]
                    producer_output_name = str(producer.outputs[0]) if len(producer.outputs) == 1 else ""
                    producer_output_tensor = (
                        model_ir.tensors.get(producer_output_name, None)
                        if producer_output_name != ""
                        else None
                    )
                    producer_rank = (
                        len(list(producer_output_tensor.shape))
                        if producer_output_tensor is not None
                        else -1
                    )
                if (
                    producer is not None
                    and str(producer.op_type) == "TRANSPOSE"
                    and producer_rank in {4, 5}
                ):
                    if len(producer.inputs) >= 1:
                        producer_input_name = str(producer.inputs[0])
                        roots.update(_trace_feature_last_passthrough_inputs(producer_input_name))
                    roots.add(producer_output_name)
                continue
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
        "AVERAGE_POOL_2D",
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
        "MAX_POOL_2D",
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
            has_preserved_input = any(name in preserve_names for name in input_names)
            has_preserved_output = any(name in preserve_names for name in output_names)
            if not has_preserved_input and not has_preserved_output:
                continue
            if has_preserved_input:
                if op_type != "TRANSPOSE" or len(op.outputs) != 1:
                    for output_name in output_names:
                        if output_name not in preserve_names:
                            preserve_names.add(output_name)
                            changed = True
                else:
                    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
                    rank = len(list(output_tensor.shape)) if output_tensor is not None else -1
                    perm = _read_transpose_perm(model_ir, op)
                    if not (
                        rank in {3, 4, 5}
                        and (
                            perm == _perm_cl_to_cf(rank)
                            or perm == _perm_cf_to_cl(rank)
                        )
                    ):
                        for output_name in output_names:
                            if output_name not in preserve_names:
                                preserve_names.add(output_name)
                                changed = True
            if has_preserved_output:
                if (
                    op_type == "RESHAPE"
                    and len(op.inputs) >= 1
                    and len(op.outputs) == 1
                    and _is_channel_last_factorized_rank3_sequence_reshape(
                        model_ir.tensors.get(str(op.inputs[0]), None),
                        model_ir.tensors.get(str(op.outputs[0]), None),
                    )
                ):
                    continue
                if op_type != "TRANSPOSE" or len(op.inputs) < 1:
                    for input_name in input_names:
                        if input_name not in preserve_names:
                            preserve_names.add(input_name)
                            changed = True
                else:
                    input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
                    rank = len(list(input_tensor.shape)) if input_tensor is not None else -1
                    perm = _read_transpose_perm(model_ir, op)
                    if not (
                        rank in {3, 4, 5}
                        and (
                            perm == _perm_cl_to_cf(rank)
                            or perm == _perm_cf_to_cl(rank)
                        )
                    ):
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
        if is_channel_last_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        ):
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
            if output_name in preserve_channel_last_tensor_names and rank == 3 and perm == [1, 0, 2]:
                output_tensor.logical_layout = channel_last_logical_layout(rank)
                continue
            if perm == _perm_cf_to_cl(rank):
                input_layout = normalize_logical_layout(
                    input_tensor.logical_layout if input_tensor is not None else LOGICAL_LAYOUT_UNKNOWN
                )
                if (
                    rank == 3
                    and output_name in set(str(v) for v in model_ir.outputs)
                    and output_name not in preserve_channel_last_tensor_names
                    and is_channel_last_logical_layout(input_layout)
                ):
                    output_tensor.logical_layout = channel_first_logical_layout(rank)
                else:
                    output_tensor.logical_layout = channel_last_logical_layout(rank)
            elif rank in {4, 5}:
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
            if output_name in preserve_channel_last_tensor_names and rank == 3:
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and output_name in set(str(v) for v in model_ir.outputs)
                and input_tensor is not None
                and len(list(input_tensor.shape)) >= rank
                and len(list(output_tensor.shape)) >= 1
                and len(list(input_tensor.shape)) >= 1
                and int(list(output_tensor.shape)[-1]) == int(list(input_tensor.shape)[-1])
            ):
                should_mark_channel_last = True
            if (
                not should_mark_channel_last
                and input_tensor is not None
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
            if (
                not should_mark_channel_last
                and _is_channel_last_factorized_rank3_sequence_reshape(input_tensor, output_tensor)
            ):
                should_mark_channel_last = True
            if should_mark_channel_last:
                output_tensor.logical_layout = channel_last_logical_layout(rank)
            continue

    safe_passthrough_ops = {
        "ABS",
        "ADD",
        "ATAN",
        "AVERAGE_POOL_2D",
        "BATCH_MATMUL",
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
        "MAX_POOL_2D",
        "MEAN",
        "MINIMUM",
        "MUL",
        "NEG",
        "PACK",
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
                    resolved_axis_tensor = axis_tensor
                    _rewrite_tensor_once(
                        str(op.inputs[1]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=resolved_axis_tensor,
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
                    resolved_axis_tensor = axis_tensor
                    _rewrite_tensor_once(
                        str(op.inputs[0]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=resolved_axis_tensor,
                            source_layout=original_layout,
                            target_layout=target_layout,
                            rank=rank,
                        ),
                    )
        elif op_type in {"SUM", "MEAN", "REDUCE_MAX", "REDUCE_MIN", "REDUCE_PROD", "REDUCE_ANY"}:
            if len(op.inputs) >= 2:
                axis_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                if axis_tensor is not None:
                    resolved_axis_tensor = axis_tensor
                    _rewrite_tensor_once(
                        str(op.inputs[1]),
                        lambda: _rewrite_axis_constant_inplace(
                            tensor=resolved_axis_tensor,
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
                    lambda pad_tensor=pad_tensor: _rewrite_matrix_constant_inplace(
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
                    resolved_perm_tensor = perm_tensor
                    perm_tensor_dtype = np.asarray(resolved_perm_tensor.data).dtype
                    perm_values = [int(v) for v in np.asarray(resolved_perm_tensor.data).reshape(-1).tolist()]
                    if len(perm_values) == rank:
                        def _rewrite_perm_tensor() -> bool:
                            rewritten_perm = [int(old_axis_to_new_axis[int(axis)]) for axis in perm_values]
                            resolved_perm_tensor.data = np.asarray(rewritten_perm, dtype=perm_tensor_dtype)
                            resolved_perm_tensor.shape = [int(rank)]
                            resolved_perm_tensor.shape_signature = [int(rank)]
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
                    lambda output_shape_tensor=output_shape_tensor: _rewrite_vector_constant_inplace(
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
                resolved_preferred_shape = [int(v) for v in list(preferred_shape)]
                if len(op.inputs) >= 2:
                    shape_tensor = model_ir.tensors.get(str(op.inputs[1]), None)
                    if shape_tensor is not None and isinstance(shape_tensor.data, np.ndarray):
                        resolved_shape_tensor = shape_tensor
                        shape_tensor_dtype = np.asarray(resolved_shape_tensor.data).dtype
                        def _rewrite_shape_tensor() -> bool:
                            resolved_shape_tensor.data = np.asarray(resolved_preferred_shape, dtype=shape_tensor_dtype)
                            resolved_shape_tensor.shape = [int(len(resolved_preferred_shape))]
                            resolved_shape_tensor.shape_signature = [int(len(resolved_preferred_shape))]
                            return True
                        _rewrite_tensor_once(str(op.inputs[1]), _rewrite_shape_tensor)
                op.options["newShape"] = list(resolved_preferred_shape)

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
        consumer_op_types = {
            str(model_ir.operators[int(consumer_idx)].op_type)
            for consumer_idx in consumers.get(output_name, [])
            if int(consumer_idx) != int(op_idx)
        }
        reshape_only_consumers = len(consumer_op_types) > 0 and consumer_op_types == {"RESHAPE"}
        if (
            reshape_only_consumers
            and input_tensor is not None
            and output_tensor is not None
            and [int(v) for v in list(input_tensor.shape)] != [int(v) for v in list(output_tensor.shape)]
        ):
            continue
        if len(consumer_op_types & {"GATHER", "GATHER_ND", "SLICE", "STRIDED_SLICE"}) > 0:
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
                ) and not reshape_only_consumers
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
                "IDENTITY",
                [input_name],
                [output_name],
                {},
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
    public_layout_bridge_tensor_names = model_ir.metadata.get("public_layout_bridge_tensor_names", [])
    if not isinstance(public_layout_bridge_tensor_names, list):
        public_layout_bridge_tensor_names = []
    public_layout_bridge_tensor_name_set = {str(name) for name in public_layout_bridge_tensor_names}
    for op_idx, op in enumerate(model_ir.operators):
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    for op in model_ir.operators:
        if str(op.op_type) != "TRANSPOSE":
            continue
        related_tensor_names = [str(v) for v in list(op.inputs) + list(op.outputs)]
        if any(name in public_layout_bridge_tensor_name_set for name in related_tensor_names):
            continue
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
        transpose_consumer_indices = [int(v) for v in consumers.get(output_name, [])]
        if (
            len(transpose_consumer_indices) > 0
            and all(
                str(model_ir.operators[int(consumer_idx)].op_type) == "RESHAPE"
                for consumer_idx in transpose_consumer_indices
            )
        ):
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
        boundary_map = {}
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
        if tensor is None:
            continue
        rank = len(list(tensor.shape))
        desired_layout = normalize_logical_layout(
            public_layout_map.get(
                str(tensor_name),
                channel_last_logical_layout(rank) if recurrent_sequence_context and rank == 3 else channel_first_logical_layout(rank),
            )
        ) if rank in {3, 4, 5} else LOGICAL_LAYOUT_UNKNOWN
        current_layout = normalize_logical_layout(tensor.logical_layout)
        if isinstance(boundary_shape, list) and len(boundary_shape) == rank:
            tensor.shape_signature = [int(v) for v in list(boundary_shape)]
            tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(boundary_shape)]
        elif (
            rank in {3, 4, 5}
            and desired_layout != LOGICAL_LAYOUT_UNKNOWN
            and current_layout != LOGICAL_LAYOUT_UNKNOWN
            and desired_layout != current_layout
        ):
            perm_to_public = logical_layout_permutation(
                source_layout=current_layout,
                target_layout=desired_layout,
            )
            current_shape_signature = list(tensor.shape_signature or tensor.shape)
            permuted_shape = (
                None
                if perm_to_public is None
                else _permute_shape(current_shape_signature, perm_to_public)
            )
            if permuted_shape is not None:
                tensor.shape_signature = [int(v) for v in list(permuted_shape)]
                tensor.shape = [int(v) if int(v) > 0 else 1 for v in list(permuted_shape)]
        if rank in {3, 4, 5}:
            tensor.logical_layout = desired_layout

def _ensure_public_boundary_layout_bridges(
    *,
    model_ir: ModelIR,
    desired_public_shape_map: Dict[str, List[int]],
    desired_public_layout_map: Dict[str, str],
) -> None:
    used_tensor_names: Set[str] = set(model_ir.tensors.keys())
    bridge_tensor_names = model_ir.metadata.get("public_layout_bridge_tensor_names", [])
    if not isinstance(bridge_tensor_names, list):
        bridge_tensor_names = []
    model_ir.metadata["public_layout_bridge_tensor_names"] = bridge_tensor_names

    def _insert_public_boundary_layout_bridge(
        *,
        tensor_name: str,
        current_tensor: TensorIR,
        desired_shape: Sequence[int],
        desired_layout: str,
        is_input: bool,
    ) -> None:
        current_shape = [int(v) for v in list(current_tensor.shape_signature or current_tensor.shape)]
        target_shape = [int(v) for v in list(desired_shape)]
        current_layout = normalize_logical_layout(current_tensor.logical_layout)
        normalized_target_layout = normalize_logical_layout(desired_layout)
        if (
            len(current_shape) not in {3, 4, 5}
            or len(current_shape) != len(target_shape)
            or current_layout == LOGICAL_LAYOUT_UNKNOWN
            or normalized_target_layout == LOGICAL_LAYOUT_UNKNOWN
            or current_layout == normalized_target_layout
        ):
            return
        perm = logical_layout_permutation(
            source_layout=normalized_target_layout if is_input else current_layout,
            target_layout=current_layout if is_input else normalized_target_layout,
        )
        expected_shape = current_shape if is_input else target_shape
        seed_shape = target_shape if is_input else current_shape
        if perm is None or _permute_shape(seed_shape, perm) != expected_shape:
            return
        bridge_tensor_name = _make_unique_identifier(
            f"{tensor_name}_public_layout_bridge",
            used_tensor_names,
        )
        bridge_tensor = _clone_tensor(current_tensor)
        bridge_tensor.name = str(bridge_tensor_name)
        model_ir.tensors[str(bridge_tensor_name)] = bridge_tensor
        if str(bridge_tensor_name) not in bridge_tensor_names:
            bridge_tensor_names.append(str(bridge_tensor_name))
        perm_name = _make_unique_identifier(
            f"{bridge_tensor_name}_perm",
            used_tensor_names,
        )
        perm_arr = np.asarray([int(v) for v in list(perm)], dtype=np.int32)
        model_ir.tensors[str(perm_name)] = TensorIR(
            name=str(perm_name),
            dtype="INT32",
            shape=[int(perm_arr.size)],
            shape_signature=[int(perm_arr.size)],
            data=perm_arr,
        )
        if is_input:
            for op in model_ir.operators:
                op.inputs = [
                    str(bridge_tensor_name) if str(name) == str(tensor_name) else str(name)
                    for name in list(op.inputs)
                ]
            model_ir.operators.insert(
                0,
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=[str(tensor_name), str(perm_name)],
                    outputs=[str(bridge_tensor_name)],
                    options={"perm": [int(v) for v in list(perm)]},
                ),
            )
            return
        for op in model_ir.operators:
            op.outputs = [
                str(bridge_tensor_name) if str(name) == str(tensor_name) else str(name)
                for name in list(op.outputs)
            ]
            op.inputs = [
                str(bridge_tensor_name) if str(name) == str(tensor_name) else str(name)
                for name in list(op.inputs)
            ]
        model_ir.operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[str(bridge_tensor_name), str(perm_name)],
                outputs=[str(tensor_name)],
                options={"perm": [int(v) for v in list(perm)]},
            )
        )

    for tensor_name in list(model_ir.inputs):
        current_tensor = model_ir.tensors.get(str(tensor_name), None)
        desired_shape = desired_public_shape_map.get(str(tensor_name), None)
        desired_layout = desired_public_layout_map.get(str(tensor_name), LOGICAL_LAYOUT_UNKNOWN)
        if current_tensor is None or desired_shape is None:
            continue
        _insert_public_boundary_layout_bridge(
            tensor_name=str(tensor_name),
            current_tensor=current_tensor,
            desired_shape=desired_shape,
            desired_layout=desired_layout,
            is_input=True,
        )

    for tensor_name in list(model_ir.outputs):
        current_tensor = model_ir.tensors.get(str(tensor_name), None)
        desired_shape = desired_public_shape_map.get(str(tensor_name), None)
        desired_layout = desired_public_layout_map.get(str(tensor_name), LOGICAL_LAYOUT_UNKNOWN)
        if current_tensor is None or desired_shape is None:
            continue
        _insert_public_boundary_layout_bridge(
            tensor_name=str(tensor_name),
            current_tensor=current_tensor,
            desired_shape=desired_shape,
            desired_layout=desired_layout,
            is_input=False,
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
    public_layout_bridge_tensor_names = set(
        str(name)
        for name in list(model_ir.metadata.get("public_layout_bridge_tensor_names", []))
    )
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
            if str(tensor_name) in public_layout_bridge_tensor_names:
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
            if (
                rank == 4
                and is_channel_last_logical_layout(layout)
                and _is_pytorch_channel_first_safe_rank4_island_op(model_ir, op)
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

def _is_rank4_channel_last_dynamic_tensor(tensor: Optional[TensorIR]) -> bool:
    if tensor is None or isinstance(tensor.data, np.ndarray):
        return False
    return (
        len(list(tensor.shape)) == 4
        and is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout))
    )

def _is_pytorch_channel_first_safe_rank4_island_op(
    model_ir: ModelIR,
    op: OperatorIR,
) -> bool:
    op_type = str(op.op_type)
    passthrough_op_types = {
        "ADD",
        "AVERAGE_POOL_2D",
        "CONCATENATION",
        "CONV_2D",
        "DEPTHWISE_CONV_2D",
        "DIV",
        "LEAKY_RELU",
        "LOGISTIC",
        "MAX_POOL_2D",
        "MAXIMUM",
        "MINIMUM",
        "MUL",
        "RELU",
        "RELU6",
        "SUB",
        "TANH",
    }
    if op_type in passthrough_op_types:
        relevant_dynamic_tensors = [
            tensor
            for tensor_name in list(op.inputs) + list(op.outputs)
            for tensor in [model_ir.tensors.get(str(tensor_name), None)]
            if _is_rank4_channel_last_dynamic_tensor(tensor)
        ]
        return len(relevant_dynamic_tensors) > 0
    return False

def _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
    model_ir: ModelIR,
) -> bool:
    public_boundary_names = [str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)]
    if len(public_boundary_names) == 0:
        return False
    for tensor_name in public_boundary_names:
        tensor = model_ir.tensors.get(tensor_name, None)
        if tensor is None:
            return False
        rank = len(list(tensor.shape))
        if rank not in {4, 5}:
            return False
        if not is_channel_last_logical_layout(normalize_logical_layout(tensor.logical_layout)):
            return False
    if any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators):
        return False
    for op in model_ir.operators:
        if not _is_pytorch_channel_first_safe_rank4_island_op(model_ir, op):
            return False
    return True

def _shrink_preserved_channel_last_regions_for_pytorch(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> Set[str]:
    if len(preserve_channel_last_tensor_names) == 0:
        return set()
    if _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(model_ir):
        return {str(name) for name in preserve_channel_last_tensor_names}
    producers: Dict[str, int] = {}
    consumers: Dict[str, List[int]] = {}
    for op_idx, op in enumerate(model_ir.operators):
        for output_name in op.outputs:
            producers[str(output_name)] = int(op_idx)
        for input_name in op.inputs:
            consumers.setdefault(str(input_name), []).append(int(op_idx))

    public_boundary_names = {str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)}
    shrunken_preserve_names: Set[str] = {
        str(name) for name in preserve_channel_last_tensor_names
    }
    for tensor_name in sorted(str(name) for name in preserve_channel_last_tensor_names):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if not _is_rank4_channel_last_dynamic_tensor(tensor):
            continue
        if str(tensor_name) in public_boundary_names:
            continue
        producer_idx = producers.get(str(tensor_name), None)
        if producer_idx is None:
            continue
        producer_op = model_ir.operators[int(producer_idx)]
        if not _is_pytorch_channel_first_safe_rank4_island_op(model_ir, producer_op):
            continue
        consumer_indices = consumers.get(str(tensor_name), [])
        if len(consumer_indices) == 0:
            continue
        if any(
            str(model_ir.operators[int(consumer_idx)].op_type) == "DEPTHWISE_CONV_2D"
            for consumer_idx in consumer_indices
        ):
            continue
        if any(
            not _is_pytorch_channel_first_safe_rank4_island_op(
                model_ir,
                model_ir.operators[int(consumer_idx)],
            )
            for consumer_idx in consumer_indices
        ):
            continue
        shrunken_preserve_names.discard(str(tensor_name))
    return shrunken_preserve_names

def _restore_non_preserved_channel_first_layouts(
    model_ir: ModelIR,
    preserve_channel_last_tensor_names: Set[str],
) -> None:
    public_layout_bridge_tensor_names = {
        str(name)
        for name in list(model_ir.metadata.get("public_layout_bridge_tensor_names", []))
    }
    for tensor_name, tensor in model_ir.tensors.items():
        if str(tensor_name) in preserve_channel_last_tensor_names:
            continue
        if str(tensor_name) in public_layout_bridge_tensor_names:
            continue
        rank = len(list(tensor.shape))
        if rank not in {3, 4, 5}:
            continue
        layout = normalize_logical_layout(tensor.logical_layout)
        if is_channel_last_logical_layout(layout):
            tensor.logical_layout = channel_first_logical_layout(rank)

def normalize_model_ir_for_pytorch_channel_first(model_ir: ModelIR) -> ModelIR:
    normalized = copy.deepcopy(model_ir)
    original_public_boundary_shapes: Dict[str, List[int]] = {}
    original_public_boundary_layouts: Dict[str, str] = {}
    for tensor_name in list(model_ir.inputs) + list(model_ir.outputs):
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        original_public_boundary_shapes[str(tensor_name)] = [
            int(v) for v in list(tensor.shape_signature or tensor.shape)
        ]
        original_public_boundary_layouts[str(tensor_name)] = normalize_logical_layout(
            tensor.logical_layout
        )
    infer_model_ir_logical_layouts(normalized)
    preserve_channel_last_tensor_names = _collect_feature_last_sequence_tensor_names(normalized)
    _apply_feature_last_sequence_layouts(normalized, preserve_channel_last_tensor_names)
    if len(preserve_channel_last_tensor_names) > 0:
        infer_model_ir_logical_layouts(normalized)
    preserve_channel_last_tensor_names = _shrink_preserved_channel_last_regions_for_pytorch(
        normalized,
        preserve_channel_last_tensor_names,
    )
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
    _restore_non_preserved_channel_first_layouts(normalized, preserve_channel_last_tensor_names)
    _rewrite_atan2_ones_like_to_atan(normalized)
    _repair_orphan_recurrent_step_tensors(normalized)
    public_layout_map = normalized.metadata.get("onnx_public_layout_map", None)
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
        normalized.metadata["onnx_public_layout_map"] = public_layout_map
    boundary_shape_map = normalized.metadata.get("onnx_boundary_shape_signature_map", None)
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
        normalized.metadata["onnx_boundary_shape_signature_map"] = boundary_shape_map
    preserve_public_channel_last_boundaries = _is_pytorch_preserved_channel_last_rank4_or_rank5_model_island(
        normalized
    )
    if isinstance(public_layout_map, dict):
        for tensor_name in list(normalized.inputs) + list(normalized.outputs):
            normalized_tensor_name = str(tensor_name)
            original_layout = original_public_boundary_layouts.get(
                normalized_tensor_name,
                LOGICAL_LAYOUT_UNKNOWN,
            )
            if (
                original_layout in {"NWC", "NHWC", "NDHWC"}
                and (
                    preserve_public_channel_last_boundaries
                    or normalized_tensor_name in preserve_channel_last_tensor_names
                )
                and normalized_tensor_name in original_public_boundary_shapes
            ):
                public_layout_map[normalized_tensor_name] = original_layout
                boundary_shape_map[normalized_tensor_name] = list(
                    original_public_boundary_shapes[normalized_tensor_name]
                )
        for output_name in list(normalized.outputs):
            normalized_output_name = str(output_name)
            if normalized_output_name not in preserve_channel_last_tensor_names:
                continue
            output_tensor = normalized.tensors.get(normalized_output_name, None)
            if output_tensor is None:
                continue
            output_rank = len(list(output_tensor.shape))
            if output_rank in {3, 4, 5}:
                public_layout_map[normalized_output_name] = channel_last_logical_layout(output_rank)
        for op in normalized.operators:
            if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 1 or len(op.outputs) != 1:
                continue
            output_name = str(op.outputs[0])
            if output_name not in set(str(v) for v in normalized.outputs):
                continue
            if output_name in preserve_channel_last_tensor_names:
                continue
            output_tensor = normalized.tensors.get(output_name, None)
            input_tensor = normalized.tensors.get(str(op.inputs[0]), None)
            if output_tensor is None or input_tensor is None:
                continue
            output_rank = len(list(output_tensor.shape))
            if output_rank != 3:
                continue
            if _read_transpose_perm(normalized, op) != _perm_cf_to_cl(output_rank):
                continue
            if is_channel_last_logical_layout(normalize_logical_layout(input_tensor.logical_layout)):
                public_layout_map[output_name] = channel_first_logical_layout(output_rank)
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
