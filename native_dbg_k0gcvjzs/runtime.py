from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

def _normalize_tensor_name(name: str) -> str:
    normalized = str(name).split(":")[0]
    if normalized.startswith("serving_default_"):
        normalized = normalized[len("serving_default_") :]
    return normalized

_TORCH_DTYPE_BY_TFLITE_DTYPE: Dict[str, torch.dtype] = {
    'BOOL': torch.bool,
    'INT8': torch.int8,
    'INT16': torch.int16,
    'INT32': torch.int32,
    'INT64': torch.int64,
    'UINT8': torch.uint8,
    'FLOAT16': torch.float16,
    'FLOAT32': torch.float32,
    'FLOAT64': torch.float64,
}

def _torch_dtype(dtype_name: str) -> torch.dtype:
    key = str(dtype_name).upper()
    if key not in _TORCH_DTYPE_BY_TFLITE_DTYPE:
        raise RuntimeError(f'Unsupported dtype for PyTorch runtime: {dtype_name}')
    return _TORCH_DTYPE_BY_TFLITE_DTYPE[key]

def _default_tensor_storage_name(tensor_name: str) -> str:
    base_name = re.sub(r'[^0-9A-Za-z_]', '_', str(tensor_name)).strip('_')
    if base_name == '':
        base_name = 'tensor'
    if base_name[0].isdigit():
        base_name = f'tensor_{base_name}'
    return base_name

def _resolve_named_input_value(kwargs: Dict[str, Any], expected_name: str) -> Any:
    if str(expected_name) in kwargs:
        return kwargs[str(expected_name)]
    normalized_expected_name = _normalize_tensor_name(str(expected_name))
    canonical_expected_name = re.sub(r'[^0-9A-Za-z]+', '_', str(expected_name)).strip('_').lower()
    for candidate_name, candidate_value in kwargs.items():
        normalized_candidate = _normalize_tensor_name(str(candidate_name))
        canonical_candidate = re.sub(r'[^0-9A-Za-z]+', '_', str(candidate_name)).strip('_').lower()
        if (
            normalized_candidate == normalized_expected_name
            or canonical_candidate == canonical_expected_name
            or normalized_candidate.endswith(normalized_expected_name)
            or normalized_expected_name.endswith(normalized_candidate)
            or canonical_candidate.endswith(canonical_expected_name)
            or canonical_expected_name.endswith(canonical_candidate)
        ):
            return candidate_value
    raise KeyError(str(expected_name))

def resolve_named_input_value(kwargs: Dict[str, Any], expected_name: str) -> Any:
    return _resolve_named_input_value(kwargs, expected_name)

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

def _align_tensor_to_target_shape(value: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:
    if target_shape is None:
        return value
    actual_shape = [int(v) for v in list(value.shape)]
    target = [int(v) for v in list(target_shape)]
    if actual_shape == target:
        return value
    perm = _perm_cl_to_cf(value.ndim)
    if perm is not None and _permute_shape(actual_shape, perm) == target:
        return value.permute(*perm).contiguous()
    perm_inv = _perm_cf_to_cl(value.ndim)
    if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:
        return value.permute(*perm_inv).contiguous()
    return value

def _align_scatter_nd_updates(updates: torch.Tensor, expected_shape: Sequence[int]) -> torch.Tensor:
    actual_shape = [int(v) for v in list(updates.shape)]
    expected = [int(v) for v in list(expected_shape)]
    if actual_shape == expected:
        return updates
    try:
        if list(torch.broadcast_shapes(tuple(actual_shape), tuple(expected))) == expected:
            return updates
    except Exception:
        pass
    perm = _perm_cf_to_cl(updates.ndim)
    if perm is not None and _permute_shape(actual_shape, perm) == expected:
        return updates.permute(*perm).contiguous()
    perm = _perm_cl_to_cf(updates.ndim)
    if perm is not None and _permute_shape(actual_shape, perm) == expected:
        return updates.permute(*perm).contiguous()
    if updates.ndim <= 5:
        import itertools
        for generic_perm in itertools.permutations(range(updates.ndim)):
            if list(generic_perm) == list(range(updates.ndim)):
                continue
            if _permute_shape(actual_shape, generic_perm) == expected:
                return updates.permute(*generic_perm).contiguous()
    return updates

def _matches_target_except_axis(actual_shape: Sequence[int], target_shape: Sequence[int], axis: int) -> bool:
    if len(list(actual_shape)) != len(list(target_shape)):
        return False
    for idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target_shape)):
        if int(idx) == int(axis):
            continue
        if int(actual_dim) != int(target_dim):
            return False
    return True

def _align_binary_inputs(x: torch.Tensor, y: torch.Tensor, target_shape: Optional[Sequence[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    target = [int(v) for v in list(target_shape)] if target_shape is not None else None
    if x.ndim != y.ndim:
        return x, y
    if [int(v) for v in list(x.shape)] == [int(v) for v in list(y.shape)]:
        return x, y
    try:
        broadcast_shape = list(torch.broadcast_shapes(tuple(int(v) for v in x.shape), tuple(int(v) for v in y.shape)))
        if target is None or [int(v) for v in broadcast_shape] == target:
            return x, y
    except Exception:
        pass
    perm = _perm_cl_to_cf(x.ndim)
    if perm is None:
        return x, y
    x_shape = [int(v) for v in list(x.shape)]
    y_shape = [int(v) for v in list(y.shape)]
    if _permute_shape(y_shape, perm) == x_shape:
        return x, y.permute(*perm).contiguous()
    if _permute_shape(x_shape, perm) == y_shape:
        return x.permute(*perm).contiguous(), y
    if target is not None:
        if _permute_shape(y_shape, perm) == target:
            return x, y.permute(*perm).contiguous()
        if _permute_shape(x_shape, perm) == target:
            return x.permute(*perm).contiguous(), y
    if x.ndim <= 5:
        import itertools
        for generic_perm in itertools.permutations(range(x.ndim)):
            if list(generic_perm) == list(range(x.ndim)):
                continue
            permuted_y_shape = _permute_shape(y_shape, generic_perm)
            if permuted_y_shape is not None:
                try:
                    torch.broadcast_shapes(tuple(permuted_y_shape), tuple(x_shape))
                    return x, y.permute(*generic_perm).contiguous()
                except Exception:
                    pass
                if target is not None:
                    try:
                        torch.broadcast_shapes(tuple(permuted_y_shape), tuple(target))
                        return x, y.permute(*generic_perm).contiguous()
                    except Exception:
                        pass
            permuted_x_shape = _permute_shape(x_shape, generic_perm)
            if permuted_x_shape is not None:
                try:
                    torch.broadcast_shapes(tuple(permuted_x_shape), tuple(y_shape))
                    return x.permute(*generic_perm).contiguous(), y
                except Exception:
                    pass
                if target is not None:
                    try:
                        torch.broadcast_shapes(tuple(permuted_x_shape), tuple(target))
                        return x.permute(*generic_perm).contiguous(), y
                    except Exception:
                        pass
    return x, y

def _normalize_dim(dim: int, rank: int) -> int:
    resolved = int(dim)
    if resolved < 0:
        resolved += int(rank)
    return resolved

def _coerce_scalar_axis(value: Any, *, device: torch.device) -> int:
    if isinstance(value, torch.Tensor):
        flat = value.to(dtype=torch.int64, device=device).reshape(-1)
        if int(flat.numel()) == 0:
            return 0
        return int(flat[0].item())
    return int(value)

def _shape_list(value: Any) -> List[int]:
    if isinstance(value, torch.Tensor):
        return [int(v) for v in value.to(dtype=torch.int64).reshape(-1).tolist()]
    if isinstance(value, np.ndarray):
        return [int(v) for v in value.reshape(-1).tolist()]
    return [int(v) for v in list(value)]

def _resolve_reshape_shape(shape_spec: Any, input_tensor: torch.Tensor, *, allow_zero: bool) -> List[int]:
    raw_shape = _shape_list(shape_spec)
    input_shape = [int(v) for v in list(input_tensor.shape)]
    resolved: List[int] = []
    infer_index: Optional[int] = None
    known_product = 1
    total_elements = int(input_tensor.numel())
    for dim_index, raw_dim in enumerate(raw_shape):
        dim_value = int(raw_dim)
        if dim_value == 0 and not allow_zero and dim_index < len(input_shape):
            dim_value = int(input_shape[dim_index])
        if dim_value == -1:
            if infer_index is not None:
                raise RuntimeError(f'Multiple -1 values are not allowed in reshape spec: {raw_shape}')
            infer_index = len(resolved)
            resolved.append(-1)
            continue
        resolved.append(dim_value)
        known_product *= int(dim_value)
    if infer_index is not None:
        if known_product == 0 or total_elements % known_product != 0:
            raise RuntimeError(
                'Failed to infer reshape dimension. '
                f'shape_spec={raw_shape} input_shape={input_shape} total_elements={total_elements}'
            )
        resolved[infer_index] = int(total_elements // known_product)
    return resolved

def _to_torch_pad_arg(paddings: torch.Tensor) -> List[int]:
    pads = paddings.to(dtype=torch.int64).reshape(-1, 2).tolist()
    torch_pad: List[int] = []
    for before, after in reversed(pads):
        torch_pad.extend([int(before), int(after)])
    while len(torch_pad) >= 2 and int(torch_pad[-2]) == 0 and int(torch_pad[-1]) == 0:
        torch_pad = torch_pad[:-2]
    return torch_pad

def _apply_pad_nd(x: torch.Tensor, paddings: torch.Tensor, *, mode: str, value: float = 0.0) -> torch.Tensor:
    pad_pairs = paddings.to(dtype=torch.int64).reshape(-1, 2).tolist()
    rank = int(x.ndim)
    if len(pad_pairs) < rank:
        pad_pairs = ([[0, 0]] * (rank - len(pad_pairs))) + pad_pairs
    elif len(pad_pairs) > rank:
        pad_pairs = pad_pairs[-rank:]
    non_zero_axes = [idx for idx, (before, after) in enumerate(pad_pairs) if int(before) != 0 or int(after) != 0]
    if len(non_zero_axes) == 0:
        return x
    if mode != 'constant' and len(non_zero_axes) > 3:
        raise RuntimeError(f'Non-constant pad supports at most 3 padded dims. mode={mode} padded_dims={len(non_zero_axes)}')
    keep_axes = [idx for idx in range(rank) if idx not in non_zero_axes]
    perm = keep_axes + non_zero_axes
    permuted = x.permute(*perm).contiguous() if perm != list(range(rank)) else x
    torch_pad: List[int] = []
    for axis in reversed(non_zero_axes):
        before, after = pad_pairs[axis]
        torch_pad.extend([int(before), int(after)])
    if mode == 'constant':
        padded = F.pad(permuted, torch_pad, mode=mode, value=float(value))
    else:
        padded = F.pad(permuted, torch_pad, mode=mode)
    if perm == list(range(rank)):
        return padded
    inverse_perm = [0] * rank
    for permuted_axis, original_axis in enumerate(perm):
        inverse_perm[int(original_axis)] = int(permuted_axis)
    return padded.permute(*inverse_perm).contiguous()

def _infer_spatial_shape_for_transposed_conv2d(*, raw_output: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int]) -> Tuple[int, int]:
    output_channels = int(raw_output.shape[1])
    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]
    if len(source) == 4:
        if int(source[1]) == output_channels:
            return int(source[2]), int(source[3])
        if int(source[-1]) == output_channels:
            return int(source[1]), int(source[2])
    return int(source[-2]), int(source[-1])

def _infer_spatial_shape_for_transposed_conv3d(*, raw_output: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int]) -> Tuple[int, int, int]:
    output_channels = int(raw_output.shape[1])
    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]
    if len(source) == 5:
        if int(source[1]) == output_channels:
            return int(source[2]), int(source[3]), int(source[4])
        if int(source[-1]) == output_channels:
            return int(source[1]), int(source[2]), int(source[3])
    return int(source[-3]), int(source[-2]), int(source[-1])

def _apply_fused_activation(x: torch.Tensor, fused: str) -> torch.Tensor:
    key = str(fused).upper()
    if key in {'', 'NONE'}:
        return x
    if key == 'RELU':
        return torch.relu(x)
    if key == 'RELU6':
        return torch.clamp(x, min=0.0, max=6.0)
    if key == 'RELU_N1_TO_1':
        return torch.clamp(x, min=-1.0, max=1.0)
    if key == 'RELU_0_TO_1':
        return torch.clamp(x, min=0.0, max=1.0)
    if key == 'TANH':
        return torch.tanh(x)
    return x

def _lookup_state_tensor(raw_state_dict: Dict[str, Any], tensor_name: str, storage_names: Dict[str, str]) -> torch.Tensor:
    original_key = str(tensor_name)
    storage_key = storage_names.get(original_key, _default_tensor_storage_name(original_key))
    if original_key in raw_state_dict:
        return torch.as_tensor(raw_state_dict[original_key])
    if storage_key in raw_state_dict:
        return torch.as_tensor(raw_state_dict[storage_key])
    raise KeyError(original_key)

def _copy_tensor_data(target: torch.Tensor, source: torch.Tensor) -> None:
    target.data.copy_(source.to(device=target.device, dtype=target.dtype))

def _validate_state_dict_keys(raw_state_dict: Dict[str, Any], storage_names: Dict[str, str], expected_tensor_names: Sequence[str]) -> None:
    recognized_keys: Set[str] = set()
    missing: List[str] = []
    for tensor_name in expected_tensor_names:
        storage_key = storage_names.get(str(tensor_name), _default_tensor_storage_name(str(tensor_name)))
        if str(tensor_name) in raw_state_dict:
            recognized_keys.add(str(tensor_name))
            continue
        if storage_key in raw_state_dict:
            recognized_keys.add(storage_key)
            continue
        missing.append(str(tensor_name))
    unexpected = sorted(str(key) for key in raw_state_dict.keys() if str(key) not in recognized_keys)
    if len(missing) > 0 or len(unexpected) > 0:
        raise RuntimeError(f'state_dict mismatch. missing={missing} unexpected={unexpected}')

def _apply_concat(values: Sequence[torch.Tensor], axis: int, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:
    if any(int(value.ndim) == 0 for value in values):
        values = [value.reshape(1) if int(value.ndim) == 0 else value for value in values]
    rank = int(values[0].ndim)
    resolved_axis = _normalize_dim(int(axis), rank)
    target = [int(v) for v in list(target_shape)] if target_shape is not None else None
    if target is not None and len(target) == rank:
        aligned_values: List[torch.Tensor] = []
        for value in values:
            actual = [int(v) for v in list(value.shape)]
            chosen = value
            if actual != target:
                perm = _perm_cl_to_cf(value.ndim)
                if perm is not None:
                    permuted_shape = _permute_shape(actual, perm)
                    if _matches_target_except_axis(permuted_shape, target, resolved_axis):
                        chosen = value.permute(*perm).contiguous()
            aligned_values.append(chosen)
        values = aligned_values
    y = torch.cat(list(values), dim=resolved_axis)
    return _apply_fused_activation(y, fused)

def _apply_module_conv2d(module: torch.nn.Conv2d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:
    expected_in_channels = int(module.in_channels)
    if x.ndim == 4 and int(x.shape[1]) != expected_in_channels and int(x.shape[-1]) == expected_in_channels:
        x = x.permute(0, 3, 1, 2).contiguous()
    y = module(x)
    y = _align_tensor_to_target_shape(y, target_shape)
    return _apply_fused_activation(y, fused)

def _apply_module_transpose_conv2d(module: torch.nn.ConvTranspose2d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], fused: str) -> torch.Tensor:
    weight = module.weight
    if x.ndim == 4 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):
        x = x.permute(0, 3, 1, 2).contiguous()
    raw = module(x)
    target_h, target_w = _infer_spatial_shape_for_transposed_conv2d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)
    y = raw[..., :target_h, :target_w]
    y = _align_tensor_to_target_shape(y, target_shape)
    return _apply_fused_activation(y, fused)

def _apply_module_conv3d(module: torch.nn.Conv3d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fused: str) -> torch.Tensor:
    weight = module.weight
    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[1]) and int(x.shape[-1]) == int(weight.shape[1]):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    y = module(x)
    y = _align_tensor_to_target_shape(y, target_shape)
    return _apply_fused_activation(y, fused)

def _apply_module_transpose_conv3d(module: torch.nn.ConvTranspose3d, x: torch.Tensor, target_shape: Optional[Sequence[int]], fallback_shape: Sequence[int], fused: str) -> torch.Tensor:
    weight = module.weight
    if x.ndim == 5 and int(x.shape[1]) != int(weight.shape[0]) and int(x.shape[-1]) == int(weight.shape[0]):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    raw = module(x)
    target_d, target_h, target_w = _infer_spatial_shape_for_transposed_conv3d(raw_output=raw, target_shape=target_shape, fallback_shape=fallback_shape)
    y = raw[..., :target_d, :target_h, :target_w]
    y = _align_tensor_to_target_shape(y, target_shape)
    return _apply_fused_activation(y, fused)

def _apply_softmax(x: torch.Tensor, axis: Optional[int], beta: float, target_shape: Optional[Sequence[int]]) -> torch.Tensor:
    resolved_axis = _normalize_dim(int(axis), x.ndim) if axis is not None else -1
    if beta != 1.0:
        x = x * beta
    y = torch.softmax(x, dim=resolved_axis)
    return _align_tensor_to_target_shape(y, target_shape)

def _apply_gather(params: torch.Tensor, indices: torch.Tensor, axis: int, batch_dims: int, target_shape: Optional[Sequence[int]], indices_name: str) -> torch.Tensor:
    indices_i64 = indices.to(dtype=torch.int64)
    resolved_axis = _normalize_dim(int(axis), params.ndim)
    if int(batch_dims) == 0 and int(resolved_axis) == 1 and str(indices_name).endswith('_crd_to_dcr_indices'):
        return _align_tensor_to_target_shape(params, target_shape)
    resolved_batch_dims = int(batch_dims)
    if resolved_batch_dims < 0:
        resolved_batch_dims += indices_i64.ndim
    if resolved_batch_dims > 0:
        leading_shape = [int(v) for v in list(indices_i64.shape[:resolved_batch_dims])]
        flat_batch = int(np.prod(leading_shape, dtype=np.int64))
        params_flat = params.reshape(flat_batch, *params.shape[resolved_batch_dims:])
        indices_flat = indices_i64.reshape(flat_batch, *indices_i64.shape[resolved_batch_dims:])
        gathered_batches: List[torch.Tensor] = []
        adjusted_axis = int(resolved_axis - resolved_batch_dims + 1)
        for batch_index in range(flat_batch):
            batch_params = params_flat[batch_index]
            batch_indices = indices_flat[batch_index]
            flat_indices = batch_indices.reshape(-1)
            batch_gathered = torch.index_select(batch_params, adjusted_axis - 1, flat_indices)
            batch_gathered = batch_gathered.reshape(*batch_params.shape[: adjusted_axis - 1], *batch_indices.shape, *batch_params.shape[adjusted_axis:])
            gathered_batches.append(batch_gathered)
        y = torch.stack(gathered_batches, dim=0).reshape(*leading_shape, *gathered_batches[0].shape)
        return _align_tensor_to_target_shape(y, target_shape)
    if indices_i64.ndim == 0:
        y = torch.index_select(params, resolved_axis, indices_i64.reshape(1)).squeeze(resolved_axis)
        return _align_tensor_to_target_shape(y, target_shape)
    flat_indices = indices_i64.reshape(-1)
    gathered = torch.index_select(params, resolved_axis, flat_indices)
    y = gathered.reshape(*params.shape[:resolved_axis], *indices_i64.shape, *params.shape[resolved_axis + 1:])
    return _align_tensor_to_target_shape(y, target_shape)

def _apply_gather_nd(params: torch.Tensor, indices: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:
    indices_i64 = indices.to(dtype=torch.int64)
    index_tuple = tuple(indices_i64[..., i] for i in range(indices_i64.shape[-1]))
    y = params[index_tuple]
    return _align_tensor_to_target_shape(y, target_shape)

def _apply_scatter_nd(indices: torch.Tensor, updates: torch.Tensor, shape: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:
    output_shape = [int(v) for v in shape.to(dtype=torch.int64).reshape(-1).tolist()]
    y = torch.zeros(output_shape, dtype=updates.dtype, device=updates.device)
    indices_i64 = indices.to(dtype=torch.int64)
    index_tuple = tuple(indices_i64[..., i] for i in range(indices_i64.shape[-1]))
    selected = y[index_tuple]
    y[index_tuple] = _align_scatter_nd_updates(updates, selected.shape)
    return _align_tensor_to_target_shape(y, target_shape)

def _apply_slice(x: torch.Tensor, begin: torch.Tensor, size: torch.Tensor, target_shape: Optional[Sequence[int]]) -> torch.Tensor:
    begin_values = begin.to(dtype=torch.int64).reshape(-1).tolist()
    size_values = size.to(dtype=torch.int64).reshape(-1).tolist()
    slices: List[slice] = []
    for axis, start in enumerate(begin_values):
        dim_size = int(x.shape[axis])
        length = int(size_values[axis])
        stop = None if length < 0 else min(int(start) + length, dim_size)
        slices.append(slice(int(start), stop))
    y = x[tuple(slices)]
    return _align_tensor_to_target_shape(y, target_shape)

def _apply_strided_slice(x: torch.Tensor, begin: torch.Tensor, end: torch.Tensor, strides: torch.Tensor, begin_mask: int, end_mask: int, target_shape: Optional[Sequence[int]]) -> torch.Tensor:
    begin_values = begin.to(dtype=torch.int64).reshape(-1).tolist()
    end_values = end.to(dtype=torch.int64).reshape(-1).tolist()
    stride_values = strides.to(dtype=torch.int64).reshape(-1).tolist()
    slices: List[slice] = []
    for axis, (start, stop, step) in enumerate(zip(begin_values, end_values, stride_values)):
        resolved_start = None if ((int(begin_mask) >> axis) & 1) else int(start)
        resolved_stop = None if ((int(end_mask) >> axis) & 1) else int(stop)
        slices.append(slice(resolved_start, resolved_stop, int(step)))
    y = x[tuple(slices)]
    return _align_tensor_to_target_shape(y, target_shape)

def _resolve_same_padding(kernel_size: int, stride: int) -> Tuple[int, int]:
    total = max(int(kernel_size) - int(stride), 0)
    before = total // 2
    after = total - before
    return before, after

def _apply_pool2d(x: torch.Tensor, filter_height: int, filter_width: int, stride_h: int, stride_w: int, padding: str, target_shape: Optional[Sequence[int]], is_max_pool: bool) -> torch.Tensor:
    resize_as_channel_last = False
    if x.ndim == 4 and target_shape is not None and len(list(target_shape)) == 4:
        actual_shape = [int(v) for v in list(x.shape)]
        target = [int(v) for v in list(target_shape)]
        if actual_shape[-1] == target[-1] and actual_shape[1] != target[1]:
            resize_as_channel_last = True
    pool_input = x.permute(0, 3, 1, 2).contiguous() if resize_as_channel_last and x.ndim == 4 else x
    if str(padding).upper() == 'SAME':
        pad_w = _resolve_same_padding(filter_width, stride_w)
        pad_h = _resolve_same_padding(filter_height, stride_h)
        pool_input = F.pad(pool_input, [pad_w[0], pad_w[1], pad_h[0], pad_h[1]], mode='constant', value=float('-inf') if is_max_pool else 0.0)
        padding_value = 0
    else:
        padding_value = 0
    if is_max_pool:
        y = F.max_pool2d(pool_input, kernel_size=(filter_height, filter_width), stride=(stride_h, stride_w), padding=padding_value)
    else:
        y = F.avg_pool2d(pool_input, kernel_size=(filter_height, filter_width), stride=(stride_h, stride_w), padding=padding_value)
    if resize_as_channel_last and y.ndim == 4:
        y = y.permute(0, 2, 3, 1).contiguous()
    return _align_tensor_to_target_shape(y, target_shape)

def _resize_bilinear_exact(x: torch.Tensor, size: Sequence[int], *, align_corners: bool, half_pixel_centers: bool) -> torch.Tensor:
    if x.ndim != 4:
        return F.interpolate(x, size=[int(size[0]), int(size[1])], mode='bilinear', align_corners=align_corners)
    out_h = int(size[0])
    out_w = int(size[1])
    in_h = int(x.shape[-2])
    in_w = int(x.shape[-1])
    if out_h <= 0 or out_w <= 0:
        raise RuntimeError('Resize target dimensions must be positive.')
    if align_corners:
        ys = torch.zeros([out_h], dtype=torch.float32, device=x.device) if out_h == 1 else torch.arange(out_h, dtype=torch.float32, device=x.device) * float(max(in_h - 1, 0)) / float(max(out_h - 1, 1))
        xs = torch.zeros([out_w], dtype=torch.float32, device=x.device) if out_w == 1 else torch.arange(out_w, dtype=torch.float32, device=x.device) * float(max(in_w - 1, 0)) / float(max(out_w - 1, 1))
    elif half_pixel_centers:
        ys = (torch.arange(out_h, dtype=torch.float32, device=x.device) + 0.5) * float(in_h) / float(out_h) - 0.5
        xs = (torch.arange(out_w, dtype=torch.float32, device=x.device) + 0.5) * float(in_w) / float(out_w) - 0.5
    else:
        ys = torch.arange(out_h, dtype=torch.float32, device=x.device) * float(in_h) / float(out_h)
        xs = torch.arange(out_w, dtype=torch.float32, device=x.device) * float(in_w) / float(out_w)
    y0 = torch.floor(ys).to(dtype=torch.int64)
    x0 = torch.floor(xs).to(dtype=torch.int64)
    y1 = y0 + 1
    x1 = x0 + 1
    y0c = y0.clamp(0, max(in_h - 1, 0))
    x0c = x0.clamp(0, max(in_w - 1, 0))
    y1c = y1.clamp(0, max(in_h - 1, 0))
    x1c = x1.clamp(0, max(in_w - 1, 0))
    ly = (ys - y0.to(dtype=torch.float32)).view(1, 1, out_h, 1)
    lx = (xs - x0.to(dtype=torch.float32)).view(1, 1, 1, out_w)
    hy = 1.0 - ly
    hx = 1.0 - lx
    top_left = x[:, :, y0c[:, None], x0c[None, :]]
    top_right = x[:, :, y0c[:, None], x1c[None, :]]
    bottom_left = x[:, :, y1c[:, None], x0c[None, :]]
    bottom_right = x[:, :, y1c[:, None], x1c[None, :]]
    return top_left * hy * hx + top_right * hy * lx + bottom_left * ly * hx + bottom_right * ly * lx

def _apply_resize(x: torch.Tensor, size: torch.Tensor, method: str, target_shape: Optional[Sequence[int]], align_corners: bool = False, half_pixel_centers: bool = False) -> torch.Tensor:
    resize_size = [int(v) for v in size.to(dtype=torch.int64).reshape(-1).tolist()]
    resize_as_channel_last = False
    if x.ndim == 4 and target_shape is not None and len(list(target_shape)) == 4:
        actual_shape = [int(v) for v in list(x.shape)]
        target = [int(v) for v in list(target_shape)]
        if actual_shape[-1] == target[-1] and actual_shape[1] != target[1]:
            resize_as_channel_last = True
    resize_input = x.permute(0, 3, 1, 2).contiguous() if resize_as_channel_last and x.ndim == 4 else x
    if str(method).lower() == 'nearest':
        y = F.interpolate(resize_input, size=resize_size, mode='nearest')
    else:
        y = _resize_bilinear_exact(resize_input, resize_size, align_corners=align_corners, half_pixel_centers=half_pixel_centers)
    if resize_as_channel_last and y.ndim == 4:
        y = y.permute(0, 2, 3, 1).contiguous()
    return _align_tensor_to_target_shape(y, target_shape)

def _box_iou(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes[:, 0], box[0])
    y1 = torch.maximum(boxes[:, 1], box[1])
    x2 = torch.minimum(boxes[:, 2], box[2])
    y2 = torch.minimum(boxes[:, 3], box[3])
    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    inter = inter_w * inter_h
    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(boxes[:, 3] - boxes[:, 1], min=0.0)
    box_area = torch.clamp(box[2] - box[0], min=0.0) * torch.clamp(box[3] - box[1], min=0.0)
    union = boxes_area + box_area - inter
    safe_union = torch.where(union > 0, union, torch.ones_like(union))
    iou = inter / safe_union
    return torch.where(union > 0, iou, torch.zeros_like(iou))

def _apply_non_max_suppression_v4(boxes: torch.Tensor, scores: torch.Tensor, max_output_size: torch.Tensor, iou_threshold: torch.Tensor, score_threshold: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_boxes = boxes.to(dtype=torch.float32).reshape(-1, 4)
    flat_scores = scores.to(dtype=torch.float32).reshape(-1)
    max_outputs = max(0, int(max_output_size.reshape(-1)[0].to(dtype=torch.int64).item()))
    iou_thresh = float(iou_threshold.reshape(-1)[0].item())
    score_thresh = float(score_threshold.reshape(-1)[0].item())
    if max_outputs == 0 or int(flat_boxes.shape[0]) == 0 or int(flat_scores.shape[0]) == 0:
        return torch.zeros([max_outputs], dtype=torch.int32, device=flat_boxes.device), torch.zeros([], dtype=torch.int32, device=flat_boxes.device)
    candidate_indices = torch.nonzero(flat_scores > score_thresh, as_tuple=False).reshape(-1)
    if int(candidate_indices.numel()) == 0:
        return torch.zeros([max_outputs], dtype=torch.int32, device=flat_boxes.device), torch.zeros([], dtype=torch.int32, device=flat_boxes.device)
    order = candidate_indices[torch.argsort(flat_scores[candidate_indices], descending=True)]
    selected: List[int] = []
    while int(order.numel()) > 0 and len(selected) < max_outputs:
        current = int(order[0].item())
        selected.append(current)
        if int(order.numel()) == 1:
            break
        remaining = order[1:]
        ious = _box_iou(flat_boxes[remaining], flat_boxes[current])
        order = remaining[ious <= iou_thresh]
    selected_tensor = torch.as_tensor(selected, dtype=torch.int32, device=flat_boxes.device)
    valid_count = torch.as_tensor(int(selected_tensor.numel()), dtype=torch.int32, device=flat_boxes.device)
    if int(selected_tensor.numel()) < max_outputs:
        selected_tensor = torch.cat([selected_tensor, torch.zeros([max_outputs - int(selected_tensor.numel())], dtype=torch.int32, device=flat_boxes.device)], dim=0)
    return selected_tensor, valid_count

def _normalize_axes(value: Any, rank: int) -> Optional[Tuple[int, ...]]:
    if value is None:
        return None
    axes = _shape_list(value)
    return tuple(sorted({_normalize_dim(int(v), rank) for v in axes}))

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

def _resolve_model_attribute(model: torch.nn.Module, attr_path: str) -> Any:
    value: Any = model
    for part in str(attr_path).split('.'):
        value = getattr(value, part)
    return value

def resolve_model_tensor(model: torch.nn.Module, attr_name: str) -> torch.Tensor:
    value = _resolve_model_attribute(model, attr_name)
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f'Generated model attribute is not a tensor: {attr_name}')
    return value

def load_generated_weights(
    *,
    model: torch.nn.Module,
    package_dir: Path,
    device: Optional[str],
) -> None:
    raw_state_dict = torch.load(package_dir / 'state_dict.pth', map_location=device or 'cpu')
    model.load_state_dict(raw_state_dict, strict=True)
    if device is not None:
        model.to(device)
