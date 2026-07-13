from __future__ import annotations

import ast
from typing import List, Optional, Sequence, Set

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR


def _constant_int_list(tensor: Optional[TensorIR]) -> Optional[List[int]]:
    if tensor is None or tensor.data is None:
        return None
    arr = np.asarray(tensor.data)
    if arr.size == 0:
        return []
    if not np.issubdtype(arr.dtype, np.integer):
        return None
    return [int(value) for value in arr.reshape(-1).tolist()]


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


def _shape_literal(values: Sequence[int]) -> str:
    return repr(tuple(int(v) for v in list(values)))


def _remap_axis_values_through_permutation(
    values: Sequence[int],
    perm: Sequence[int],
) -> List[int]:
    remapped = [0] * len(list(perm))
    for output_axis, input_axis in enumerate(list(perm)):
        remapped[int(input_axis)] = int(values[output_axis])
    return [int(v) for v in list(remapped)]


def _remap_mask_bits_through_permutation(
    mask: int,
    perm: Sequence[int],
) -> int:
    remapped_mask = 0
    for output_axis, input_axis in enumerate(list(perm)):
        if int(mask) & (1 << int(output_axis)):
            remapped_mask |= 1 << int(input_axis)
    return int(remapped_mask)


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


def _add_synthetic_tensor_to_model_ir(
    *,
    model_ir: ModelIR,
    base_name: str,
    data: np.ndarray,
    dtype: str,
    synthetic_tensor_serial_ref: List[int],
) -> str:
    candidate = str(base_name)
    while candidate in model_ir.tensors:
        synthetic_tensor_serial_ref[0] += 1
        candidate = f"{base_name}_{synthetic_tensor_serial_ref[0]}"
    array = np.asarray(data)
    model_ir.tensors[candidate] = TensorIR(
        name=candidate,
        dtype=str(dtype),
        shape=[int(v) for v in list(array.shape)],
        shape_signature=[int(v) for v in list(array.shape)],
        data=array,
    )
    return candidate
