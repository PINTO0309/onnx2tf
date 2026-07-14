from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    is_channel_first_logical_layout,
    is_channel_last_logical_layout,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_codegen_utils import (
    _shape_lists_equal_relaxed,
)


def _binary_trailing_axis_constant_buffer_alias_shape_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    inlined_constant_tensor_names: Set[str],
    tensor_name: str,
    other_tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if tensor is None or other_tensor is None:
        return None
    if not isinstance(tensor.data, np.ndarray):
        return None
    if str(tensor_name) in model_ir.inputs or str(tensor_name) in producer_index:
        return None
    if str(tensor_name) in inlined_constant_tensor_names:
        return None
    tensor_shape = [int(v) for v in list(tensor.shape)]
    other_shape = [int(v) for v in list(other_tensor.shape)]
    if len(tensor_shape) != 1 or len(other_shape) < 2:
        return None
    constant_width = int(tensor_shape[0])
    if constant_width <= 0:
        return None
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
    if target_axis is None or int(target_axis) != len(other_shape) - 1:
        return None
    return [1 for _ in range(len(other_shape) - 1)] + [int(constant_width)]


def _channel_first_rank4_constant_buffer_alias_shape_for_codegen(
    *,
    model_ir: ModelIR,
    producer_index: Dict[str, int],
    inlined_constant_tensor_names: Set[str],
    tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    if tensor is None or not isinstance(tensor.data, np.ndarray):
        return None
    if str(tensor_name) in model_ir.inputs or str(tensor_name) in producer_index:
        return None
    if str(tensor_name) in inlined_constant_tensor_names:
        return None
    if bool(tensor.is_variable):
        return None
    tensor_shape = [int(v) for v in list(tensor.shape)]
    if len(tensor_shape) != 4:
        return None
    if (
        tensor_shape[0] != 1
        or tensor_shape[1] != 1
        or tensor_shape[2] != 1
        or tensor_shape[3] <= 0
    ):
        return None
    return [1, int(tensor_shape[3]), 1, 1]


def _constant_permute_for_broadcast_for_codegen(
    *,
    model_ir: ModelIR,
    tensor_name: str,
    other_tensor_name: str,
) -> Optional[List[int]]:
    tensor = model_ir.tensors.get(str(tensor_name), None)
    other_tensor = model_ir.tensors.get(str(other_tensor_name), None)
    if (
        tensor is None
        or other_tensor is None
        or not isinstance(tensor.data, np.ndarray)
    ):
        return None
    tensor_shape = [int(v) for v in list(tensor.shape)]
    other_shape = [int(v) for v in list(other_tensor.shape)]
    if len(tensor_shape) != len(other_shape) or len(tensor_shape) <= 1:
        return None
    if _shape_lists_equal_relaxed(tensor_shape, other_shape):
        return None
    tensor_broadcast_shape = [
        int(v) if int(v) > 0 else 1 for v in tensor_shape
    ]
    other_broadcast_shape = [int(v) if int(v) > 0 else 1 for v in other_shape]
    non_singleton_axes = [
        idx for idx, dim in enumerate(tensor_shape) if int(dim) > 1
    ]
    try:
        np.broadcast_shapes(
            tuple(tensor_broadcast_shape), tuple(other_broadcast_shape)
        )
        # Keep singleton-expanded constants on their original axis when they
        # already broadcast correctly. Permuting them to exactly match the peer
        # tensor can collapse the intended broadcast result, e.g. [1,384,1]
        # with [1,1,384] should stay as-is and broadcast to [1,384,384].
        if len(non_singleton_axes) == 1:
            return None
        if (
            len(non_singleton_axes) == 1
            and int(non_singleton_axes[0]) == len(tensor_shape) - 1
            and int(tensor_shape[-1]) == int(other_shape[-1])
        ):
            return None
    except Exception:
        pass

    def _try_exact_match(perm: Sequence[int]) -> Optional[List[int]]:
        permuted_shape = [int(tensor_shape[int(idx)]) for idx in list(perm)]
        if [int(v) for v in list(permuted_shape)] == [
            int(v) for v in list(other_shape)
        ]:
            return [int(v) for v in list(perm)]
        return None

    preferred_perm: Optional[Tuple[int, ...]] = None
    tensor_layout = normalize_logical_layout(tensor.logical_layout)
    if tensor_layout == "NCHW" and len(tensor_shape) == 4:
        preferred_perm = (0, 2, 3, 1)
    elif tensor_layout == "NCDHW" and len(tensor_shape) == 5:
        preferred_perm = (0, 2, 3, 4, 1)
    elif tensor_layout == "NCW" and len(tensor_shape) == 3:
        preferred_perm = (0, 2, 1)
    if preferred_perm is not None:
        exact_perm = _try_exact_match(preferred_perm)
        if exact_perm is not None:
            return exact_perm
        preferred_shape = [
            int(tensor_shape[int(idx)]) for idx in preferred_perm
        ]
        preferred_broadcast_shape = [
            int(v) if int(v) > 0 else 1 for v in preferred_shape
        ]
        try:
            np.broadcast_shapes(
                tuple(preferred_broadcast_shape), tuple(other_broadcast_shape)
            )
            return [int(v) for v in list(preferred_perm)]
        except Exception:
            pass

    import itertools

    for generic_perm in itertools.permutations(range(len(tensor_shape))):
        if list(generic_perm) == list(range(len(tensor_shape))):
            continue
        exact_perm = _try_exact_match(generic_perm)
        if exact_perm is not None:
            return exact_perm

    best_broadcast_perm: Optional[Tuple[int, List[int]]] = None
    for generic_perm in itertools.permutations(range(len(tensor_shape))):
        if list(generic_perm) == list(range(len(tensor_shape))):
            continue
        permuted_shape = [int(tensor_shape[int(idx)]) for idx in generic_perm]
        permuted_broadcast_shape = [
            int(v) if int(v) > 0 else 1 for v in permuted_shape
        ]
        try:
            np.broadcast_shapes(
                tuple(permuted_broadcast_shape), tuple(other_broadcast_shape)
            )
            score = sum(
                1
                for permuted_dim, other_dim in zip(permuted_shape, other_shape)
                if int(permuted_dim) == int(other_dim)
            )
            candidate = (int(score), [int(v) for v in list(generic_perm)])
            if (
                best_broadcast_perm is None
                or candidate[0] > best_broadcast_perm[0]
            ):
                best_broadcast_perm = candidate
        except Exception:
            continue
    return None if best_broadcast_perm is None else list(best_broadcast_perm[1])
