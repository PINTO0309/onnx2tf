from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np
import onnx

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    TensorIR,
    channel_first_logical_layout,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    logical_layout_permutation,
    normalize_logical_layout,
    rewrite_axis_for_layout,
)


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


def _tensor_name_suggests_channel_last_layout_for_codegen(
    tensor_name: str,
) -> bool:
    return str(tensor_name).lower().endswith(("_nhwc", "_nwc", "_ndhwc"))


def _preferred_reshape_target_values(
    tensor: Optional[TensorIR],
) -> Optional[List[int]]:
    if tensor is None:
        return None
    preferred = [int(value) for value in list(tensor.shape)]
    if tensor.shape_signature is not None:
        signature = [
            int(value) for value in list(tensor.shape_signature)
        ]
        if (
            len(signature) == len(list(tensor.shape))
            and any(int(value) <= 0 for value in signature)
        ):
            preferred = signature
    rank = len(list(preferred))
    perm_to_cf = _perm_cl_to_cf(rank)
    if (
        perm_to_cf is not None
        and is_channel_first_logical_layout(
            normalize_logical_layout(tensor.logical_layout)
        )
        and _tensor_name_suggests_channel_last_layout_for_codegen(
            str(tensor.name)
        )
    ):
        permuted = _permute_shape(preferred, perm_to_cf)
        if permuted is not None:
            return [int(value) for value in list(permuted)]
    return preferred


def _preferred_reshape_target_values_for_op(
    *,
    model_ir: ModelIR,
    op: OperatorIR,
) -> Optional[List[int]]:
    if (
        str(op.op_type) != "RESHAPE"
        or len(op.inputs) == 0
        or len(op.outputs) == 0
    ):
        return None
    output_tensor = model_ir.tensors.get(str(op.outputs[0]), None)
    if output_tensor is None:
        return None
    preferred = _preferred_reshape_target_values(output_tensor)
    if preferred is None:
        preferred = [int(value) for value in list(output_tensor.shape)]
    return preferred


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
        physical_layout=normalize_logical_layout(tensor.physical_layout),
        onnx_tensor_name=tensor.onnx_tensor_name,
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


def _normalize_constant_pad_pairs(
    values: np.ndarray,
) -> Optional[List[List[int]]]:
    arr = np.asarray(values, dtype=np.int64)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return [[int(v) for v in list(row)] for row in arr.tolist()]
    if arr.ndim != 1 or int(arr.size) == 0 or int(arr.size) % 2 != 0:
        return None
    flat_values = [int(v) for v in arr.reshape(-1).tolist()]
    rank = int(len(flat_values) // 2)
    begins = flat_values[:rank]
    ends = flat_values[rank:]
    return [[int(begins[idx]), int(ends[idx])] for idx in range(rank)]


def _constant_pad_pairs_for_tensor(tensor: Optional[TensorIR]) -> Optional[List[List[int]]]:
    if tensor is None or tensor.data is None:
        return None
    try:
        pads = np.asarray(tensor.data, dtype=np.int64)
    except Exception:
        return None
    return _normalize_constant_pad_pairs(pads)


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
    tensor.physical_layout = target_layout
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
    return _is_channel_last_factorized_rank3_sequence_reshape_by_shape(
        input_tensor=input_tensor,
        output_tensor=output_tensor,
    )


def _is_channel_last_factorized_rank3_sequence_reshape_by_shape(
    input_tensor: Optional[TensorIR],
    output_tensor: Optional[TensorIR],
) -> bool:
    if input_tensor is None or output_tensor is None:
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


def _has_channel_last_factorized_rank3_sequence_consumer(
    *,
    model_ir: ModelIR,
    consumers: Dict[str, List[int]],
    tensor_name: str,
) -> bool:
    input_tensor = model_ir.tensors.get(str(tensor_name), None)
    if input_tensor is None:
        return False
    for consumer_idx in consumers.get(str(tensor_name), []):
        consumer = model_ir.operators[int(consumer_idx)]
        if str(consumer.op_type) != "RESHAPE" or len(consumer.outputs) != 1:
            continue
        output_tensor = model_ir.tensors.get(str(consumer.outputs[0]), None)
        if _is_channel_last_factorized_rank3_sequence_reshape_by_shape(
            input_tensor=input_tensor,
            output_tensor=output_tensor,
        ):
            return True
    return False
