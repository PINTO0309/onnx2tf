from __future__ import annotations

import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F


_TORCH_DTYPE_BY_TFLITE_DTYPE: Dict[str, torch.dtype] = {
    "BOOL": torch.bool,
    "INT8": torch.int8,
    "INT16": torch.int16,
    "INT32": torch.int32,
    "INT64": torch.int64,
    "UINT8": torch.uint8,
    "FLOAT16": torch.float16,
    "FLOAT32": torch.float32,
    "FLOAT64": torch.float64,
}


def _torch_dtype(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).upper()
    if key not in _TORCH_DTYPE_BY_TFLITE_DTYPE:
        raise RuntimeError(f"Unsupported dtype for PyTorch runtime: {dtype_name}")
    return _TORCH_DTYPE_BY_TFLITE_DTYPE[key]


def _as_shape_signature(tensor_meta: Dict[str, Any]) -> List[Optional[int]]:
    signature = tensor_meta.get("shape_signature", tensor_meta.get("shape", []))
    normalized: List[Optional[int]] = []
    for dim in list(signature):
        value = int(dim)
        normalized.append(None if value < 0 else value)
    return normalized


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


def _permute_shape(values: Sequence[int], perm: Sequence[int]) -> List[int]:
    items = [int(v) for v in list(values)]
    return [int(items[idx]) for idx in perm]


def _target_output_shape(executor: "_GraphExecutor", op: Dict[str, Any]) -> Optional[List[int]]:
    outputs = list(op.get("outputs", []))
    if len(outputs) != 1:
        return None
    output_meta = executor._metadata["tensors"].get(str(outputs[0]), {})
    if "shape" not in output_meta:
        return None
    return [int(v) for v in list(output_meta.get("shape", []))]


def _align_tensor_to_target_shape(
    value: torch.Tensor,
    target_shape: Optional[Sequence[int]],
) -> torch.Tensor:
    if target_shape is None:
        return value
    actual_shape = [int(v) for v in list(value.shape)]
    target = [int(v) for v in list(target_shape)]
    if actual_shape == target:
        return value
    perm = _perm_cl_to_cf(value.ndim)
    if perm is None:
        perm_inv = _perm_cf_to_cl(value.ndim)
        if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:
            return value.permute(*perm_inv).contiguous()
        return value
    if _permute_shape(actual_shape, perm) == target:
        return value.permute(*perm).contiguous()
    perm_inv = _perm_cf_to_cl(value.ndim)
    if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:
        return value.permute(*perm_inv).contiguous()
    return value


def _align_binary_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    target_shape: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != y.ndim:
        return x, y
    if [int(v) for v in list(x.shape)] == [int(v) for v in list(y.shape)]:
        return x, y
    try:
        torch.broadcast_shapes(tuple(int(v) for v in x.shape), tuple(int(v) for v in y.shape))
        return x, y
    except Exception:
        pass
    perm = _perm_cl_to_cf(x.ndim)
    if perm is None:
        return x, y
    x_shape = [int(v) for v in list(x.shape)]
    y_shape = [int(v) for v in list(y.shape)]
    target = [int(v) for v in list(target_shape)] if target_shape is not None else None
    if _permute_shape(y_shape, perm) == x_shape:
        return x, y.permute(*perm).contiguous()
    if _permute_shape(x_shape, perm) == y_shape:
        return x.permute(*perm).contiguous(), y
    if target is not None:
        if _permute_shape(y_shape, perm) == target:
            return x, y.permute(*perm).contiguous()
        if _permute_shape(x_shape, perm) == target:
            return x.permute(*perm).contiguous(), y
    return x, y


def _tensor_layout_variants(value: torch.Tensor) -> List[torch.Tensor]:
    variants: List[torch.Tensor] = [value]
    if value.ndim >= 2:
        variants.append(value.transpose(-1, -2).contiguous())
    perm_cf_to_cl = _perm_cf_to_cl(value.ndim)
    if perm_cf_to_cl is not None:
        variants.append(value.permute(*perm_cf_to_cl).contiguous())
    perm_cl_to_cf = _perm_cl_to_cf(value.ndim)
    if perm_cl_to_cf is not None:
        variants.append(value.permute(*perm_cl_to_cf).contiguous())
    unique: List[torch.Tensor] = []
    seen_shapes: Set[Tuple[int, ...]] = set()
    for candidate in variants:
        shape_key = tuple(int(v) for v in list(candidate.shape))
        if shape_key in seen_shapes:
            continue
        seen_shapes.add(shape_key)
        unique.append(candidate)
    return unique


def _matches_target_except_axis(
    actual_shape: Sequence[int],
    target_shape: Sequence[int],
    axis: int,
) -> bool:
    if len(list(actual_shape)) != len(list(target_shape)):
        return False
    for idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target_shape)):
        if int(idx) == int(axis):
            continue
        if int(actual_dim) != int(target_dim):
            return False
    return True


def _infer_split_axis_and_sizes(
    executor: "_GraphExecutor",
    op: Dict[str, Any],
    x: torch.Tensor,
) -> Tuple[Optional[int], Optional[List[int]]]:
    output_shapes: List[List[int]] = []
    for output_name in list(op.get("outputs", [])):
        output_meta = executor._metadata["tensors"].get(str(output_name), {})
        shape = output_meta.get("shape", None)
        if not isinstance(shape, list) or len(shape) != x.ndim:
            return None, None
        output_shapes.append([int(v) for v in list(shape)])
    if len(output_shapes) == 0:
        return None, None
    input_shape = [int(v) for v in list(x.shape)]
    candidate_shape_sets: List[List[List[int]]] = [output_shapes]
    perm = _perm_cl_to_cf(x.ndim)
    if perm is not None:
        candidate_shape_sets.append([_permute_shape(shape, perm) for shape in output_shapes])
    for candidate_shapes in candidate_shape_sets:
        candidate_axes: List[int] = []
        for axis in range(x.ndim):
            valid = True
            size_sum = 0
            for output_shape in candidate_shapes:
                for dim_idx, dim_value in enumerate(output_shape):
                    if dim_idx == axis:
                        continue
                    if int(dim_value) != int(input_shape[dim_idx]):
                        valid = False
                        break
                if not valid:
                    break
                size_sum += int(output_shape[axis])
            if valid and size_sum == int(input_shape[axis]):
                candidate_axes.append(int(axis))
        if len(candidate_axes) == 1:
            axis = int(candidate_axes[0])
            sizes = [int(shape[axis]) for shape in candidate_shapes]
            return axis, sizes
    return None, None


def _infer_reduce_axes_from_metadata(
    executor: "_GraphExecutor",
    op: Dict[str, Any],
    x: torch.Tensor,
    keepdims: bool,
) -> Optional[Tuple[int, ...]]:
    target_shape = _target_output_shape(executor, op)
    if target_shape is None or len(target_shape) != x.ndim:
        return None
    input_shape = [int(v) for v in list(x.shape)]
    candidate_shapes: List[List[int]] = [list(target_shape)]
    perm = _perm_cl_to_cf(x.ndim)
    if perm is not None:
        candidate_shapes.append(_permute_shape(target_shape, perm))
    for candidate in candidate_shapes:
        axes: List[int] = []
        valid = True
        for idx, (input_dim, output_dim) in enumerate(zip(input_shape, candidate)):
            if keepdims:
                if int(output_dim) == int(input_dim):
                    continue
                if int(output_dim) == 1:
                    axes.append(int(idx))
                    continue
                valid = False
                break
            else:
                # Non-keepdims reduction inference is intentionally conservative.
                valid = False
                break
        if valid:
            return tuple(axes)
    return None


def _infer_concat_axis_from_metadata(
    values: Sequence[torch.Tensor],
    target_shape: Optional[Sequence[int]],
) -> Optional[int]:
    if target_shape is None or len(values) == 0:
        return None
    rank = int(values[0].ndim)
    if len(list(target_shape)) != rank:
        return None
    input_shapes = [[int(v) for v in list(value.shape)] for value in values]
    target = [int(v) for v in list(target_shape)]
    candidate_axes: List[int] = []
    for axis in range(rank):
        valid = True
        axis_sum = 0
        for shape in input_shapes:
            if len(shape) != rank:
                valid = False
                break
            for dim_idx, dim_value in enumerate(shape):
                if dim_idx == axis:
                    continue
                if int(dim_value) != int(target[dim_idx]):
                    valid = False
                    break
            if not valid:
                break
            axis_sum += int(shape[axis])
        if valid and axis_sum == int(target[axis]):
            candidate_axes.append(int(axis))
    if len(candidate_axes) == 1:
        return int(candidate_axes[0])
    return None


def _vector_permutation_variants(values: Sequence[int]) -> List[List[int]]:
    variants: List[List[int]] = [[int(v) for v in list(values)]]
    rank = len(list(values))
    perm_cf_to_cl = _perm_cf_to_cl(rank)
    if perm_cf_to_cl is not None:
        variants.append([int(values[idx]) for idx in perm_cf_to_cl])
    perm_cl_to_cf = _perm_cl_to_cf(rank)
    if perm_cl_to_cf is not None:
        variants.append([int(values[idx]) for idx in perm_cl_to_cf])
    unique: List[List[int]] = []
    seen: Set[Tuple[int, ...]] = set()
    for variant in variants:
        key = tuple(int(v) for v in variant)
        if key in seen:
            continue
        seen.add(key)
        unique.append(variant)
    return unique


def _apply_fused_activation(x: torch.Tensor, fused: str) -> torch.Tensor:
    key = str(fused).upper()
    if key in {"", "NONE"}:
        return x
    if key == "RELU":
        return torch.relu(x)
    if key == "RELU6":
        return torch.clamp(x, min=0.0, max=6.0)
    if key == "RELU_N1_TO_1":
        return torch.clamp(x, min=-1.0, max=1.0)
    if key == "RELU_0_TO_1":
        return torch.clamp(x, min=0.0, max=1.0)
    if key == "TANH":
        return torch.tanh(x)
    return x


def _apply_named_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    key = str(activation).upper()
    if key in {"", "NONE"}:
        return x
    if key == "TANH":
        return torch.tanh(x)
    if key == "RELU":
        return torch.relu(x)
    if key == "RELU6":
        return torch.clamp(x, min=0.0, max=6.0)
    if key == "SIGMOID":
        return torch.sigmoid(x)
    return x


def _resolve_padding_2d(
    *,
    padding: str,
    weight: torch.Tensor,
    dilation: Tuple[int, int],
) -> Tuple[int, int]:
    if str(padding).upper() == "VALID":
        return (0, 0)
    kernel_h = int(weight.shape[-2])
    kernel_w = int(weight.shape[-1])
    return (
        int(((kernel_h - 1) * dilation[0]) // 2),
        int(((kernel_w - 1) * dilation[1]) // 2),
    )


def _resolve_padding_3d(
    *,
    padding: str,
    weight: torch.Tensor,
    dilation: Tuple[int, int, int],
) -> Tuple[int, int, int]:
    if str(padding).upper() == "VALID":
        return (0, 0, 0)
    kernel_d = int(weight.shape[-3])
    kernel_h = int(weight.shape[-2])
    kernel_w = int(weight.shape[-1])
    return (
        int(((kernel_d - 1) * dilation[0]) // 2),
        int(((kernel_h - 1) * dilation[1]) // 2),
        int(((kernel_w - 1) * dilation[2]) // 2),
    )


def _to_torch_pad_arg(paddings: torch.Tensor) -> List[int]:
    pads = paddings.to(dtype=torch.int64).reshape(-1, 2).tolist()
    torch_pad: List[int] = []
    for before, after in reversed(pads):
        torch_pad.extend([int(before), int(after)])
    return torch_pad


def _coerce_scalar_axis(value: Any, *, device: torch.device) -> int:
    if isinstance(value, torch.Tensor):
        flat = value.to(dtype=torch.int64, device=device).reshape(-1)
        if int(flat.numel()) == 0:
            return 0
        return int(flat[0].item())
    return int(value)


def _normalize_dim(dim: int, rank: int) -> int:
    resolved = int(dim)
    if resolved < 0:
        resolved += int(rank)
    return resolved


def _default_tensor_storage_name(tensor_name: str) -> str:
    base_name = re.sub(r"[^0-9A-Za-z_]", "_", str(tensor_name)).strip("_")
    if base_name == "":
        base_name = "tensor"
    if base_name[0].isdigit():
        base_name = f"tensor_{base_name}"
    return base_name


def _tensor_storage_name_map_from_metadata(metadata: Dict[str, Any]) -> Dict[str, str]:
    explicit_map = metadata.get("tensor_storage_names", {})
    if isinstance(explicit_map, dict) and len(explicit_map) > 0:
        return {str(name): str(storage_name) for name, storage_name in explicit_map.items()}
    used_names: Dict[str, int] = {}
    resolved: Dict[str, str] = {}
    for tensor_name, tensor_meta in sorted(metadata.get("tensors", {}).items()):
        if not bool(tensor_meta.get("has_data", False)):
            continue
        base_name = _default_tensor_storage_name(str(tensor_name))
        candidate = base_name
        suffix = 1
        while candidate in used_names:
            candidate = f"{base_name}_{suffix}"
            suffix += 1
        used_names[candidate] = 1
        resolved[str(tensor_name)] = candidate
    return resolved


class _GraphExecutor:
    def __init__(
        self,
        *,
        model: "_GeneratedModel",
        metadata: Dict[str, Any],
    ) -> None:
        self._model = model
        self._metadata = metadata
        self._tensor_meta = metadata["tensors"]
        self._operators = list(metadata["operators"])
        self._kernels = _register_supported_kernels()

    def _resolve_tensor(
        self,
        tensor_name: str,
        env: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        if tensor_name in env:
            return env[tensor_name]
        storage_name = self._model.tensor_storage_names.get(
            tensor_name,
            _default_tensor_storage_name(tensor_name),
        )
        if hasattr(self._model, storage_name):
            tensor = getattr(self._model, storage_name)
            if isinstance(tensor, torch.Tensor):
                return tensor
        raise RuntimeError(f"Tensor not found in PyTorch runtime: {tensor_name}")

    def _assign_outputs(
        self,
        op: Dict[str, Any],
        values: Sequence[torch.Tensor],
        env: Dict[str, torch.Tensor],
    ) -> None:
        outputs = list(op.get("outputs", []))
        if len(outputs) != len(values):
            raise RuntimeError(
                f"Output arity mismatch for op={op.get('op_type')} "
                f"expected={len(outputs)} actual={len(values)}"
            )
        for output_name, value in zip(outputs, values):
            env[str(output_name)] = value

    def run(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        env: Dict[str, torch.Tensor] = dict(inputs)
        for op in self._operators:
            op_type = str(op["op_type"])
            if op_type not in self._kernels:
                raise RuntimeError(f"Unsupported op in generated PyTorch runtime: {op_type}")
            self._kernels[op_type](self, op, env)
        return env


def _kernel_unary(
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        y = fn(x)
        y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
        executor._assign_outputs(op, [y], env)
    return _impl


def _kernel_binary(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        y = executor._resolve_tensor(str(op["inputs"][1]), env)
        target_shape = _target_output_shape(executor, op)
        x, y = _align_binary_inputs(x, y, target_shape)
        z = fn(x, y)
        z = _align_tensor_to_target_shape(z, target_shape)
        executor._assign_outputs(op, [z], env)
    return _impl


def _kernel_identity(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    executor._assign_outputs(op, [x], env)


def _kernel_logical_not(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    executor._assign_outputs(op, [torch.logical_not(x)], env)


def _kernel_cast(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    out_dtype = _torch_dtype(str(op.get("options", {}).get("outDataType", "FLOAT32")))
    executor._assign_outputs(op, [x.to(dtype=out_dtype)], env)


def _kernel_reshape(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    if len(op["inputs"]) >= 2:
        new_shape = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    else:
        new_shape = list(op.get("options", {}).get("newShape", []))
    executor._assign_outputs(op, [torch.reshape(x, [int(v) for v in new_shape])], env)


def _kernel_transpose(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    if len(op["inputs"]) >= 2:
        perm = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    else:
        perm = list(op.get("options", {}).get("perm", []))
    executor._assign_outputs(op, [x.permute(*[int(v) for v in perm])], env)


def _kernel_concat(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    values = [executor._resolve_tensor(str(name), env) for name in op["inputs"]]
    rank = int(values[0].ndim)
    axis = _normalize_dim(int(options.get("axis", 0)), rank)
    target_shape = _target_output_shape(executor, op)
    if target_shape is not None and len(target_shape) == rank:
        aligned_values: List[torch.Tensor] = []
        target = [int(v) for v in list(target_shape)]
        for value in values:
            actual = [int(v) for v in list(value.shape)]
            chosen = value
            if actual != target:
                perm = _perm_cl_to_cf(value.ndim)
                if perm is not None:
                    permuted_shape = _permute_shape(actual, perm)
                    if _matches_target_except_axis(permuted_shape, target, axis):
                        chosen = value.permute(*perm).contiguous()
            aligned_values.append(chosen)
        values = aligned_values
    y = torch.cat(values, dim=axis)
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_squeeze(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    axes = [int(v) for v in list(op.get("options", {}).get("squeezeDims", []))]
    if len(axes) == 0:
        executor._assign_outputs(op, [torch.squeeze(x)], env)
        return
    y = x
    for axis in sorted([_normalize_dim(v, y.ndim) for v in axes], reverse=True):
        y = torch.squeeze(y, dim=axis)
    executor._assign_outputs(op, [y], env)


def _kernel_expand_dims(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    if len(op["inputs"]) >= 2:
        axis = _coerce_scalar_axis(executor._resolve_tensor(str(op["inputs"][1]), env), device=x.device)
    else:
        axis = int(op.get("options", {}).get("axis", 0))
    executor._assign_outputs(op, [torch.unsqueeze(x, dim=axis)], env)


def _kernel_split(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    x: torch.Tensor
    axis = 0
    if len(op["inputs"]) >= 2:
        first = executor._resolve_tensor(str(op["inputs"][0]), env)
        second = executor._resolve_tensor(str(op["inputs"][1]), env)
        if first.ndim <= 1 and first.dtype in {torch.int8, torch.int16, torch.int32, torch.int64} and int(first.numel()) == 1:
            axis = int(first.reshape(-1)[0].item())
            x = second
        elif second.ndim <= 1 and second.dtype in {torch.int8, torch.int16, torch.int32, torch.int64} and int(second.numel()) == 1:
            axis = int(second.reshape(-1)[0].item())
            x = first
        else:
            axis = int(first.reshape(-1)[0].item())
            x = second
    else:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        axis = int(options.get("axis", 0))
    axis = _normalize_dim(axis, x.ndim)
    sections = int(options.get("numSplits", len(op["outputs"])))
    outputs = list(torch.tensor_split(x, sections, dim=axis))
    executor._assign_outputs(op, outputs, env)


def _kernel_pack(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    axis = int(op.get("options", {}).get("axis", 0))
    values = [executor._resolve_tensor(str(name), env) for name in op["inputs"]]
    executor._assign_outputs(op, [torch.stack(values, dim=axis)], env)


def _kernel_unpack(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    axis = _normalize_dim(int(op.get("options", {}).get("axis", 0)), x.ndim)
    values = list(torch.unbind(x, dim=axis))
    executor._assign_outputs(op, values, env)


def _kernel_slice(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    begin = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    size = executor._resolve_tensor(str(op["inputs"][2]), env).to(dtype=torch.int64).reshape(-1).tolist()
    slices = []
    for axis, start in enumerate(begin):
        dim_size = int(x.shape[axis])
        length = int(size[axis])
        if length < 0:
            stop = None
        else:
            stop = min(int(start) + length, dim_size)
        slices.append(slice(int(start), stop))
    y = x[tuple(slices)]
    executor._assign_outputs(op, [y], env)


def _kernel_strided_slice(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    begin = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    end = executor._resolve_tensor(str(op["inputs"][2]), env).to(dtype=torch.int64).reshape(-1).tolist()
    strides = executor._resolve_tensor(str(op["inputs"][3]), env).to(dtype=torch.int64).reshape(-1).tolist()
    begin_mask = int(op.get("options", {}).get("beginMask", 0))
    end_mask = int(op.get("options", {}).get("endMask", 0))
    slices = []
    for axis, (start, stop, step) in enumerate(zip(begin, end, strides)):
        resolved_start = None if ((begin_mask >> axis) & 1) else int(start)
        resolved_stop = None if ((end_mask >> axis) & 1) else int(stop)
        slices.append(slice(resolved_start, resolved_stop, int(step)))
    y = x[tuple(slices)]
    executor._assign_outputs(op, [y], env)


def _kernel_shape(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    out_dtype = _torch_dtype(str(op.get("options", {}).get("outType", "INT32")))
    executor._assign_outputs(op, [torch.tensor(list(x.shape), dtype=out_dtype, device=x.device)], env)


def _kernel_fill(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    dims = executor._resolve_tensor(str(op["inputs"][0]), env).to(dtype=torch.int64).reshape(-1).tolist()
    value = executor._resolve_tensor(str(op["inputs"][1]), env).reshape(-1)[0]
    executor._assign_outputs(op, [torch.full([int(v) for v in dims], value.item(), dtype=value.dtype, device=value.device)], env)


def _kernel_range(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    start = executor._resolve_tensor(str(op["inputs"][0]), env).reshape(-1)[0]
    limit = executor._resolve_tensor(str(op["inputs"][1]), env).reshape(-1)[0]
    delta = executor._resolve_tensor(str(op["inputs"][2]), env).reshape(-1)[0]
    executor._assign_outputs(op, [torch.arange(start=start.item(), end=limit.item(), step=delta.item(), device=start.device, dtype=start.dtype)], env)


def _kernel_softmax(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    options = dict(op.get("options", {}))
    beta = float(options.get("beta", 1.0))
    axis = int(options.get("axis", -1))
    axis = _normalize_dim(axis, x.ndim)
    if beta != 1.0:
        x = x * beta
    executor._assign_outputs(op, [torch.softmax(x, dim=axis)], env)


def _kernel_reduce(
    fn: Callable[[torch.Tensor, Optional[Tuple[int, ...]], bool], torch.Tensor],
) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        keepdims = bool(op.get("options", {}).get("keepDims", True))
        axis: Optional[Tuple[int, ...]] = None
        if len(op["inputs"]) >= 2:
            raw_axis = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
            axis = tuple(sorted({_normalize_dim(int(v), x.ndim) for v in raw_axis}))
        y = fn(x, axis, keepdims)
        y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
        executor._assign_outputs(op, [y], env)
    return _impl


def _reduce_sum(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.sum(x) if not keepdims else torch.sum(x).reshape([1] * x.ndim)
    return torch.sum(x, dim=axis, keepdim=keepdims)


def _reduce_mean(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.mean(x) if not keepdims else torch.mean(x).reshape([1] * x.ndim)
    return torch.mean(x, dim=axis, keepdim=keepdims)


def _reduce_max(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.amax(x, keepdim=keepdims)
    return torch.amax(x, dim=axis, keepdim=keepdims)


def _reduce_min(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.amin(x, keepdim=keepdims)
    return torch.amin(x, dim=axis, keepdim=keepdims)


def _reduce_prod(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        y = torch.prod(x)
        return y if not keepdims else y.reshape([1] * x.ndim)
    result = x
    for dim in sorted(axis, reverse=True):
        result = torch.prod(result, dim=dim, keepdim=keepdims)
    return result


def _reduce_any(x: torch.Tensor, axis: Optional[Tuple[int, ...]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        y = torch.any(x)
        return y if not keepdims else y.reshape([1] * x.ndim)
    result = x
    for dim in sorted(axis, reverse=True):
        result = torch.any(result, dim=dim, keepdim=keepdims)
    return result


def _kernel_pad(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    paddings = executor._resolve_tensor(str(op["inputs"][1]), env)
    pad = _to_torch_pad_arg(paddings)
    if str(op["op_type"]) == "PADV2":
        constant_values = executor._resolve_tensor(str(op["inputs"][2]), env).reshape(-1)[0].item()
        y = F.pad(x, pad, mode="constant", value=float(constant_values))
    else:
        y = F.pad(x, pad, mode="constant", value=0.0)
    executor._assign_outputs(op, [y], env)


def _kernel_mirror_pad(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    paddings = executor._resolve_tensor(str(op["inputs"][1]), env)
    pad = _to_torch_pad_arg(paddings)
    y = F.pad(x, pad, mode="reflect")
    executor._assign_outputs(op, [y], env)


def _kernel_where(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    if len(op["inputs"]) == 1:
        cond = executor._resolve_tensor(str(op["inputs"][0]), env)
        executor._assign_outputs(op, [torch.nonzero(cond, as_tuple=False)], env)
        return
    cond = executor._resolve_tensor(str(op["inputs"][0]), env)
    x = executor._resolve_tensor(str(op["inputs"][1]), env)
    y = executor._resolve_tensor(str(op["inputs"][2]), env)
    executor._assign_outputs(op, [torch.where(cond, x, y)], env)


def _kernel_gather(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    params = executor._resolve_tensor(str(op["inputs"][0]), env)
    indices = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64)
    axis = _normalize_dim(int(op.get("options", {}).get("axis", 0)), params.ndim)
    if indices.ndim == 0:
        y = torch.index_select(params, axis, indices.reshape(1)).squeeze(axis)
    else:
        expanded = indices
        while expanded.ndim < params.ndim:
            expanded = expanded.unsqueeze(-1)
        expanded = expanded.expand(*indices.shape, *params.shape[axis + 1 :])
        source = params
        while source.ndim < expanded.ndim:
            source = source.unsqueeze(0)
        y = torch.take_along_dim(source, expanded, dim=axis)
    executor._assign_outputs(op, [y], env)


def _kernel_gather_nd(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    params = executor._resolve_tensor(str(op["inputs"][0]), env)
    indices = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64)
    index_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
    y = params[index_tuple]
    executor._assign_outputs(op, [y], env)


def _kernel_scatter_nd(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    indices = executor._resolve_tensor(str(op["inputs"][0]), env).to(dtype=torch.int64)
    updates = executor._resolve_tensor(str(op["inputs"][1]), env)
    shape = executor._resolve_tensor(str(op["inputs"][2]), env).to(dtype=torch.int64).reshape(-1).tolist()
    y = torch.zeros([int(v) for v in shape], dtype=updates.dtype, device=updates.device)
    index_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
    y[index_tuple] = updates
    executor._assign_outputs(op, [y], env)


def _kernel_tile(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    multiples = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    executor._assign_outputs(op, [x.repeat(*[int(v) for v in multiples])], env)


def _kernel_broadcast_to(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    shape = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    executor._assign_outputs(op, [torch.broadcast_to(x, [int(v) for v in shape])], env)


def _kernel_arg(is_max: bool) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        axis = _normalize_dim(int(op.get("options", {}).get("axis", 0)), x.ndim)
        keepdims = bool(op.get("options", {}).get("keepDims", True))
        fn = torch.argmax if is_max else torch.argmin
        y = fn(x, dim=axis, keepdim=keepdims)
        output_meta = executor._metadata["tensors"].get(str(op["outputs"][0]), {})
        executor._assign_outputs(op, [y.to(dtype=_torch_dtype(str(output_meta.get("dtype", "INT64"))))], env)
    return _impl


def _kernel_topk(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    k = executor._resolve_tensor(str(op["inputs"][1]), env).reshape(-1)[0].to(dtype=torch.int64)
    axis = _normalize_dim(int(op.get("options", {}).get("axis", -1)), x.ndim)
    largest = bool(op.get("options", {}).get("largest", True))
    sorted_output = bool(op.get("options", {}).get("sorted", True))
    values, indices = torch.topk(x, k=int(k.item()), dim=axis, largest=largest, sorted=sorted_output)
    executor._assign_outputs(op, [values, indices], env)


def _kernel_leaky_relu(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    alpha = float(op.get("options", {}).get("alpha", 0.2))
    executor._assign_outputs(op, [F.leaky_relu(x, negative_slope=alpha)], env)


def _kernel_prelu(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    alpha = executor._resolve_tensor(str(op["inputs"][1]), env)
    executor._assign_outputs(op, [F.prelu(x, alpha.reshape(-1))], env)


def _kernel_l2_norm(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    axis = _normalize_dim(int(op.get("options", {}).get("axis", -1)), x.ndim)
    epsilon = float(op.get("options", {}).get("epsilon", 1e-6))
    denom = torch.sqrt(torch.sum(x * x, dim=axis, keepdim=True) + epsilon)
    executor._assign_outputs(op, [x / denom], env)


def _kernel_cumsum(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    axis = _coerce_scalar_axis(executor._resolve_tensor(str(op["inputs"][1]), env), device=x.device)
    executor._assign_outputs(op, [torch.cumsum(x, dim=_normalize_dim(axis, x.ndim))], env)


def _kernel_one_hot(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    indices = executor._resolve_tensor(str(op["inputs"][0]), env).to(dtype=torch.int64)
    depth = executor._resolve_tensor(str(op["inputs"][1]), env).reshape(-1)[0].to(dtype=torch.int64)
    on_off = executor._resolve_tensor(str(op["inputs"][2]), env).reshape(-1)
    axis = int(op.get("options", {}).get("axis", -1))
    y = F.one_hot(indices, num_classes=int(depth.item())).to(dtype=on_off.dtype)
    y = y * (on_off[1] - on_off[0]) + on_off[0]
    if axis != -1 and axis != y.ndim - 1:
        dims = list(range(y.ndim))
        last = dims.pop(-1)
        dims.insert(_normalize_dim(axis, len(dims) + 1), last)
        y = y.permute(*dims)
    executor._assign_outputs(op, [y], env)


def _kernel_reverse_v2(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    axes = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
    dims = [_normalize_dim(int(v), x.ndim) for v in axes]
    executor._assign_outputs(op, [torch.flip(x, dims=dims)], env)


def _kernel_depth_to_space(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    block = int(op.get("options", {}).get("blockSize", 1))
    executor._assign_outputs(op, [F.pixel_shuffle(x, block)], env)


def _kernel_space_to_depth(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    block = int(op.get("options", {}).get("blockSize", 1))
    n, c, h, w = x.shape
    y = x.reshape(n, c, h // block, block, w // block, block)
    y = y.permute(0, 1, 3, 5, 2, 4).reshape(n, c * block * block, h // block, w // block)
    executor._assign_outputs(op, [y], env)


def _kernel_batch_matmul(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    y = executor._resolve_tensor(str(op["inputs"][1]), env)
    if bool(options.get("adjX", False)):
        x = x.transpose(-1, -2)
    if bool(options.get("adjY", False)):
        y = y.transpose(-1, -2)
    target_shape = _target_output_shape(executor, op)
    best_z: Optional[torch.Tensor] = None
    best_score: Optional[Tuple[int, int]] = None
    for x_variant_idx, x_variant in enumerate(_tensor_layout_variants(x)):
        for y_variant_idx, y_variant in enumerate(_tensor_layout_variants(y)):
            try:
                candidate = torch.matmul(x_variant, y_variant)
            except Exception:
                continue
            score = 2
            if target_shape is not None and len(target_shape) == candidate.ndim:
                actual_shape = [int(v) for v in list(candidate.shape)]
                if actual_shape == target_shape:
                    score = 0
                else:
                    aligned_shape = [
                        int(v)
                        for v in list(_align_tensor_to_target_shape(candidate, target_shape).shape)
                    ]
                    if aligned_shape == [int(v) for v in list(target_shape)]:
                        score = 1
            candidate_score = (int(score), int(x_variant_idx + y_variant_idx))
            if best_score is None or candidate_score < best_score:
                best_score = candidate_score
                best_z = candidate
    z = torch.matmul(x, y) if best_z is None else best_z
    z = _align_tensor_to_target_shape(z, target_shape)
    executor._assign_outputs(op, [_apply_fused_activation(z, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_conv2d(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    w = executor._resolve_tensor(str(op["inputs"][1]), env)
    b = executor._resolve_tensor(str(op["inputs"][2]), env) if len(op["inputs"]) >= 3 and str(op["inputs"][2]) != "" else None
    if x.ndim == 4 and int(x.shape[1]) != int(w.shape[1]) and int(x.shape[-1]) == int(w.shape[1]):
        x = x.permute(0, 3, 1, 2).contiguous()
    stride = (int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    dilation = (int(options.get("dilationHFactor", 1)), int(options.get("dilationWFactor", 1)))
    groups = max(1, int(x.shape[1]) // max(1, int(w.shape[1])))
    padding = _resolve_padding_2d(padding=str(options.get("padding", "SAME")), weight=w, dilation=dilation)
    y = F.conv2d(x, w, bias=b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_depthwise_conv2d(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    w = executor._resolve_tensor(str(op["inputs"][1]), env)
    b = executor._resolve_tensor(str(op["inputs"][2]), env) if len(op["inputs"]) >= 3 and str(op["inputs"][2]) != "" else None
    if x.ndim == 4 and int(x.shape[1]) != int(w.shape[1]) and int(x.shape[-1]) == int(w.shape[1]):
        x = x.permute(0, 3, 1, 2).contiguous()
    in_channels = int(x.shape[1])
    stride = (int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    dilation = (int(options.get("dilationHFactor", 1)), int(options.get("dilationWFactor", 1)))
    padding = _resolve_padding_2d(padding=str(options.get("padding", "SAME")), weight=w, dilation=dilation)
    y = F.conv2d(x, w, bias=b, stride=stride, padding=padding, dilation=dilation, groups=in_channels)
    y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_transpose_conv(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    output_shape = executor._resolve_tensor(str(op["inputs"][0]), env).to(dtype=torch.int64).reshape(-1).tolist()
    w = executor._resolve_tensor(str(op["inputs"][1]), env)
    x = executor._resolve_tensor(str(op["inputs"][2]), env)
    b = executor._resolve_tensor(str(op["inputs"][3]), env) if len(op["inputs"]) >= 4 and str(op["inputs"][3]) != "" else None
    stride = (int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    padding = _resolve_padding_2d(padding=str(options.get("padding", "SAME")), weight=w, dilation=(1, 1))
    raw = F.conv_transpose2d(x, w, bias=b, stride=stride, padding=padding)
    target = [int(v) for v in output_shape]
    y = raw[..., : target[-2], : target[-1]]
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_conv3d(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    w = executor._resolve_tensor(str(op["inputs"][1]), env)
    b = executor._resolve_tensor(str(op["inputs"][2]), env) if len(op["inputs"]) >= 3 and str(op["inputs"][2]) != "" else None
    if x.ndim == 5 and int(x.shape[1]) != int(w.shape[1]) and int(x.shape[-1]) == int(w.shape[1]):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    stride = (int(options.get("strideD", 1)), int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    dilation = (int(options.get("dilationDFactor", 1)), int(options.get("dilationHFactor", 1)), int(options.get("dilationWFactor", 1)))
    groups = max(1, int(x.shape[1]) // max(1, int(w.shape[1])))
    padding = _resolve_padding_3d(padding=str(options.get("padding", "SAME")), weight=w, dilation=dilation)
    y = F.conv3d(x, w, bias=b, stride=stride, padding=padding, dilation=dilation, groups=groups)
    y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_conv3d_transpose(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    options = dict(op.get("options", {}))
    output_shape = executor._resolve_tensor(str(op["inputs"][0]), env).to(dtype=torch.int64).reshape(-1).tolist()
    w = executor._resolve_tensor(str(op["inputs"][1]), env)
    x = executor._resolve_tensor(str(op["inputs"][2]), env)
    b = executor._resolve_tensor(str(op["inputs"][3]), env) if len(op["inputs"]) >= 4 and str(op["inputs"][3]) != "" else None
    stride = (int(options.get("strideD", 1)), int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    padding = _resolve_padding_3d(padding=str(options.get("padding", "SAME")), weight=w, dilation=(1, 1, 1))
    raw = F.conv_transpose3d(x, w, bias=b, stride=stride, padding=padding)
    target = [int(v) for v in output_shape]
    y = raw[..., : target[-3], : target[-2], : target[-1]]
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_pool2d(is_max_pool: bool) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        options = dict(op.get("options", {}))
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        kernel_size = (int(options.get("filterHeight", 1)), int(options.get("filterWidth", 1)))
        stride = (int(options.get("strideH", 1)), int(options.get("strideW", 1)))
        padding_mode = str(options.get("padding", "SAME"))
        padding = (
            int((kernel_size[0] - 1) // 2) if padding_mode.upper() == "SAME" else 0,
            int((kernel_size[1] - 1) // 2) if padding_mode.upper() == "SAME" else 0,
        )
        if is_max_pool:
            y = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            y = F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
        executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)
    return _impl


def _kernel_resize(method: str) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        if len(op["inputs"]) >= 2:
            size = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
        else:
            size = [
                int(op.get("options", {}).get("newHeight", int(x.shape[-2]))),
                int(op.get("options", {}).get("newWidth", int(x.shape[-1]))),
            ]
        if method == "nearest":
            y = F.interpolate(x, size=[int(size[0]), int(size[1])], mode="nearest")
        else:
            y = F.interpolate(
                x,
                size=[int(size[0]), int(size[1])],
                mode="bilinear",
                align_corners=bool(op.get("options", {}).get("alignCorners", False)),
            )
        executor._assign_outputs(op, [y], env)
    return _impl


def _register_supported_kernels() -> Dict[str, Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]]:
    kernels: Dict[str, Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]] = {
        "IDENTITY": _kernel_identity,
        "LOGICAL_NOT": _kernel_logical_not,
        "CAST": _kernel_cast,
        "RESHAPE": _kernel_reshape,
        "TRANSPOSE": _kernel_transpose,
        "CONCATENATION": _kernel_concat,
        "SQUEEZE": _kernel_squeeze,
        "EXPAND_DIMS": _kernel_expand_dims,
        "SPLIT": _kernel_split,
        "PACK": _kernel_pack,
        "UNPACK": _kernel_unpack,
        "SLICE": _kernel_slice,
        "STRIDED_SLICE": _kernel_strided_slice,
        "SHAPE": _kernel_shape,
        "FILL": _kernel_fill,
        "RANGE": _kernel_range,
        "SOFTMAX": _kernel_softmax,
        "SUM": _kernel_reduce(_reduce_sum),
        "MEAN": _kernel_reduce(_reduce_mean),
        "REDUCE_MAX": _kernel_reduce(_reduce_max),
        "REDUCE_MIN": _kernel_reduce(_reduce_min),
        "REDUCE_PROD": _kernel_reduce(_reduce_prod),
        "REDUCE_ANY": _kernel_reduce(_reduce_any),
        "PAD": _kernel_pad,
        "PADV2": _kernel_pad,
        "MIRROR_PAD": _kernel_mirror_pad,
        "WHERE": _kernel_where,
        "SELECT": _kernel_where,
        "SELECT_V2": _kernel_where,
        "GATHER": _kernel_gather,
        "GATHER_ND": _kernel_gather_nd,
        "SCATTER_ND": _kernel_scatter_nd,
        "TILE": _kernel_tile,
        "BROADCAST_TO": _kernel_broadcast_to,
        "ARG_MAX": _kernel_arg(is_max=True),
        "ARG_MIN": _kernel_arg(is_max=False),
        "TOPK_V2": _kernel_topk,
        "LEAKY_RELU": _kernel_leaky_relu,
        "PRELU": _kernel_prelu,
        "L2_NORMALIZATION": _kernel_l2_norm,
        "CUMSUM": _kernel_cumsum,
        "ONE_HOT": _kernel_one_hot,
        "REVERSE_V2": _kernel_reverse_v2,
        "DEPTH_TO_SPACE": _kernel_depth_to_space,
        "SPACE_TO_DEPTH": _kernel_space_to_depth,
        "BATCH_MATMUL": _kernel_batch_matmul,
        "CONV_2D": _kernel_conv2d,
        "DEPTHWISE_CONV_2D": _kernel_depthwise_conv2d,
        "TRANSPOSE_CONV": _kernel_transpose_conv,
        "CONV_3D": _kernel_conv3d,
        "CONV_3D_TRANSPOSE": _kernel_conv3d_transpose,
        "AVERAGE_POOL_2D": _kernel_pool2d(is_max_pool=False),
        "MAX_POOL_2D": _kernel_pool2d(is_max_pool=True),
        "RESIZE_BILINEAR": _kernel_resize(method="bilinear"),
        "RESIZE_NEAREST_NEIGHBOR": _kernel_resize(method="nearest"),
        "ABS": _kernel_unary(torch.abs),
        "CEIL": _kernel_unary(torch.ceil),
        "COS": _kernel_unary(torch.cos),
        "ELU": _kernel_unary(F.elu),
        "EXP": _kernel_unary(torch.exp),
        "FLOOR": _kernel_unary(torch.floor),
        "LOG": _kernel_unary(torch.log),
        "NEG": _kernel_unary(torch.neg),
        "RELU": _kernel_unary(torch.relu),
        "RELU6": _kernel_unary(lambda x: torch.clamp(x, min=0.0, max=6.0)),
        "LOGISTIC": _kernel_unary(torch.sigmoid),
        "SIGN": _kernel_unary(torch.sign),
        "SIN": _kernel_unary(torch.sin),
        "SQRT": _kernel_unary(torch.sqrt),
        "TANH": _kernel_unary(torch.tanh),
        "ADD": _kernel_binary(torch.add),
        "ATAN2": _kernel_binary(torch.atan2),
        "SUB": _kernel_binary(torch.sub),
        "MUL": _kernel_binary(torch.mul),
        "DIV": _kernel_binary(torch.div),
        "FLOOR_MOD": _kernel_binary(torch.remainder),
        "MAXIMUM": _kernel_binary(torch.maximum),
        "MINIMUM": _kernel_binary(torch.minimum),
        "POW": _kernel_binary(torch.pow),
        "RIGHT_SHIFT": _kernel_binary(torch.bitwise_right_shift),
        "EQUAL": _kernel_binary(torch.eq),
        "NOT_EQUAL": _kernel_binary(torch.ne),
        "GREATER": _kernel_binary(torch.gt),
        "GREATER_EQUAL": _kernel_binary(torch.ge),
        "LESS": _kernel_binary(torch.lt),
        "LESS_EQUAL": _kernel_binary(torch.le),
        "LOGICAL_AND": _kernel_binary(torch.logical_and),
        "LOGICAL_OR": _kernel_binary(torch.logical_or),
        "RELU_N1_TO_1": _kernel_unary(lambda x: torch.clamp(x, min=-1.0, max=1.0)),
        "RELU_0_TO_1": _kernel_unary(lambda x: torch.clamp(x, min=0.0, max=1.0)),
    }
    return kernels


SUPPORTED_TORCH_KERNEL_OP_TYPES = set(_register_supported_kernels().keys())


class _GeneratedModel(torch.nn.Module):
    def __init__(self, *, metadata: Dict[str, Any]) -> None:
        super().__init__()
        self._metadata = metadata
        self.input_names = list(metadata.get("inputs", []))
        self.output_names = list(metadata.get("outputs", []))
        self.tensor_storage_names = _tensor_storage_name_map_from_metadata(metadata)
        self._executor = _GraphExecutor(model=self, metadata=metadata)
        for tensor_name, tensor_meta in sorted(metadata.get("tensors", {}).items()):
            if not bool(tensor_meta.get("has_data", False)):
                continue
            dtype = _torch_dtype(str(tensor_meta["dtype"]))
            shape = [int(v) for v in list(tensor_meta.get("shape", []))]
            if len(shape) == 0:
                shape = []
            value = torch.zeros(shape, dtype=dtype)
            storage_name = self.tensor_storage_names.get(
                str(tensor_name),
                _default_tensor_storage_name(str(tensor_name)),
            )
            if bool(tensor_meta.get("is_variable", False)):
                self.register_parameter(storage_name, torch.nn.Parameter(value, requires_grad=False))
            else:
                self.register_buffer(storage_name, value, persistent=True)

    def forward(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Any:
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError("Use either positional inputs or keyword inputs, not both.")
        if len(kwargs) > 0:
            inputs = {str(name): kwargs[str(name)] for name in self.input_names}
        else:
            if len(args) != len(self.input_names):
                raise RuntimeError(
                    f"Input arity mismatch. expected={len(self.input_names)} actual={len(args)}"
                )
            inputs = {str(name): value for name, value in zip(self.input_names, args)}
        env = self._executor.run(inputs)
        outputs = [env[str(name)] for name in self.output_names]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = self.forward(*args, **kwargs)
        if len(self.output_names) == 1:
            return {str(self.output_names[0]): result}
        return {str(name): value for name, value in zip(self.output_names, result)}


def load_generated_model_package(
    *,
    package_dir: str,
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> _GeneratedModel:
    metadata_path = os.path.join(package_dir, "metadata.json")
    state_dict_path = os.path.join(package_dir, "state_dict.pth")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    raw_state_dict = torch.load(state_dict_path, map_location=device or "cpu")
    storage_name_map = _tensor_storage_name_map_from_metadata(metadata)
    for tensor_name, tensor_meta in metadata.get("tensors", {}).items():
        if not bool(tensor_meta.get("has_data", False)):
            continue
        original_key = str(tensor_name)
        storage_key = storage_name_map.get(original_key, original_key)
        tensor_value = None
        if original_key in raw_state_dict:
            tensor_value = raw_state_dict[original_key]
        elif storage_key in raw_state_dict:
            tensor_value = raw_state_dict[storage_key]
        if not isinstance(tensor_value, torch.Tensor):
            continue
        actual_shape = [int(v) for v in list(tensor_value.shape)]
        tensor_meta["shape"] = list(actual_shape)
        tensor_meta["shape_signature"] = list(actual_shape)
    model = _GeneratedModel(metadata=metadata)
    state_dict = {}
    for key, value in raw_state_dict.items():
        lookup_key = str(key)
        target_key = storage_name_map.get(lookup_key, lookup_key)
        state_dict[str(target_key)] = value
    model.load_state_dict(state_dict, strict=True)
    if device is not None:
        model = model.to(device)
    if eval_mode:
        model.eval()
    return model
