from __future__ import annotations

import copy
import json
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _adapt_input_layout_for_tflite_input,
    _build_tflite_detail_map,
    _create_tflite_interpreter,
    _dequantize_tflite_output,
    _normalize_tensor_name,
    _quantize_for_tflite_input,
    _resize_tflite_inputs_if_needed,
)


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


def _module_device(module: Any) -> torch.device:
    if torch.jit.is_scripting():
        return torch.device("cpu")
    for parameter in module.parameters():
        return parameter.device
    for buffer in module.buffers():
        return buffer.device
    return torch.device("cpu")


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


def _tensor_name_layout_hint(name: str) -> Optional[str]:
    normalized = str(name).lower()
    if normalized.endswith(("_nhwc", "_nwc", "_ndhwc")):
        return "channel_last"
    if normalized.endswith(("_nchw", "_ncw", "_ncdhw")):
        return "channel_first"
    return None


def _should_resize_as_channel_last(
    executor: "_GraphExecutor",
    op: Dict[str, Any],
    x: torch.Tensor,
    target_shape: Optional[Sequence[int]],
) -> bool:
    for tensor_name in list(op.get("outputs", [])) + list(op.get("inputs", [])):
        hint = _tensor_name_layout_hint(str(tensor_name))
        if hint == "channel_last":
            return True
        if hint == "channel_first":
            return False
    if x.ndim == 4 and target_shape is not None and len(list(target_shape)) == 4:
        actual_shape = [int(v) for v in list(x.shape)]
        target = [int(v) for v in list(target_shape)]
        if actual_shape[-1] == target[-1]:
            return True
        if actual_shape[1] == target[1]:
            return False
    return False


def _align_tensor_to_target_shape(
    value: torch.Tensor,
    target_shape: Optional[Sequence[int]],
) -> torch.Tensor:
    if target_shape is None:
        return value
    actual_shape: List[int] = []
    for dim in list(value.shape):
        if not isinstance(dim, int):
            return value
        actual_shape.append(int(dim))
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
    if len(actual_shape) == len(target):
        can_narrow = True
        has_mismatch = False
        for dim_idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target)):
            if int(target_dim) <= 0 or int(actual_dim) < int(target_dim):
                can_narrow = False
                break
            if int(actual_dim) != int(target_dim):
                has_mismatch = True
                if int(dim_idx) == 0:
                    can_narrow = False
                    break
        if can_narrow and has_mismatch:
            narrowed = value
            for dim_idx, (actual_dim, target_dim) in enumerate(zip(actual_shape, target)):
                if int(actual_dim) > int(target_dim):
                    narrowed = torch.narrow(narrowed, int(dim_idx), 0, int(target_dim))
            return narrowed
    return value


def _infer_spatial_shape_for_transposed_conv2d(
    *,
    raw_output: torch.Tensor,
    target_shape: Optional[Sequence[int]],
    fallback_shape: Sequence[int],
) -> Tuple[int, int]:
    output_channels = int(raw_output.shape[1])
    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]
    if len(source) == 4:
        if int(source[1]) == output_channels:
            return int(source[2]), int(source[3])
        if int(source[-1]) == output_channels:
            return int(source[1]), int(source[2])
    return int(source[-2]), int(source[-1])


def _infer_spatial_shape_for_transposed_conv3d(
    *,
    raw_output: torch.Tensor,
    target_shape: Optional[Sequence[int]],
    fallback_shape: Sequence[int],
) -> Tuple[int, int, int]:
    output_channels = int(raw_output.shape[1])
    source = [int(v) for v in list(target_shape)] if target_shape is not None else [int(v) for v in list(fallback_shape)]
    if len(source) == 5:
        if int(source[1]) == output_channels:
            return int(source[2]), int(source[3]), int(source[4])
        if int(source[-1]) == output_channels:
            return int(source[1]), int(source[2]), int(source[3])
    return int(source[-3]), int(source[-2]), int(source[-1])


def _align_numpy_to_target_shape(
    value: np.ndarray,
    target_shape: Optional[Sequence[int]],
) -> np.ndarray:
    if target_shape is None:
        return np.asarray(value)
    array = np.asarray(value)
    actual_shape = [int(v) for v in list(array.shape)]
    target = [int(v) for v in list(target_shape)]
    if actual_shape == target:
        return array
    perm = _perm_cl_to_cf(array.ndim)
    if perm is not None and _permute_shape(actual_shape, perm) == target:
        return np.transpose(array, perm)
    perm_inv = _perm_cf_to_cl(array.ndim)
    if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:
        return np.transpose(array, perm_inv)
    return array


def _align_numpy_to_signature_shape(
    value: np.ndarray,
    target_shape: Optional[Sequence[Optional[int]]],
) -> np.ndarray:
    if target_shape is None:
        return np.asarray(value)
    array = np.asarray(value)
    if array.ndim != len(list(target_shape)):
        return array
    normalized_target = [
        int(array.shape[idx]) if dim is None or int(dim) < 0 else int(dim)
        for idx, dim in enumerate(list(target_shape))
    ]
    return _align_numpy_to_target_shape(array, normalized_target)


def _numpy_dtype_is_string(dtype: np.dtype) -> bool:
    return dtype.kind in {"U", "S", "O"}


def _canonical_tensor_name(name: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()


def _coerce_input_to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes)):
        return np.asarray([value], dtype=object)
    return np.asarray(value)


def _resolve_named_input_value(
    kwargs: Dict[str, Any],
    expected_name: str,
) -> Any:
    if str(expected_name) in kwargs:
        return kwargs[str(expected_name)]
    normalized_expected_name = _normalize_tensor_name(str(expected_name))
    canonical_expected_name = _canonical_tensor_name(str(expected_name))
    for candidate_name, candidate_value in kwargs.items():
        normalized_candidate = _normalize_tensor_name(str(candidate_name))
        canonical_candidate = _canonical_tensor_name(str(candidate_name))
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


def _coerce_output_value(value: np.ndarray, *, device: Optional[str]) -> Any:
    array = np.asarray(value)
    if _numpy_dtype_is_string(array.dtype):
        return array
    tensor = torch.as_tensor(array)
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def _default_value_for_tflite_detail(detail: Dict[str, Any]) -> np.ndarray:
    raw_shape = detail.get("shape_signature", detail.get("shape", []))
    shape = [max(1, int(v)) if int(v) >= 0 else 1 for v in list(np.asarray(raw_shape).reshape(-1).tolist())]
    target_dtype = np.dtype(detail["dtype"])
    if _numpy_dtype_is_string(target_dtype):
        fill_value: Any = b"" if target_dtype.kind == "S" else ""
        return np.full(shape, fill_value, dtype=target_dtype if target_dtype.kind in {"S", "U"} else object)
    if np.issubdtype(target_dtype, np.bool_):
        return np.zeros(shape, dtype=np.bool_)
    if np.issubdtype(target_dtype, np.integer):
        return np.zeros(shape, dtype=target_dtype)
    return np.zeros(shape, dtype=target_dtype)


def _recover_missing_tflite_tensor_data(
    *,
    interpreter: Any,
    error: RuntimeError,
) -> bool:
    match = re.search(r"Input tensor\s+(\d+)\s+lacks data", str(error))
    if match is None:
        return False
    missing_index = int(match.group(1))
    for detail in interpreter.get_tensor_details():
        if int(detail["index"]) != missing_index:
            continue
        interpreter.set_tensor(
            missing_index,
            _default_value_for_tflite_detail(detail),
        )
        return True
    return False


def _invoke_tflite_with_recovery(interpreter: Any, *, max_attempts: int = 16) -> None:
    for _ in range(max(1, int(max_attempts))):
        try:
            interpreter.invoke()
            return
        except RuntimeError as ex:
            if not _recover_missing_tflite_tensor_data(
                interpreter=interpreter,
                error=ex,
            ):
                raise
    raise RuntimeError(
        "TFLite invocation did not converge after missing-data recovery attempts. "
        f"max_attempts={int(max_attempts)}"
    )


def _align_binary_inputs(
    x: torch.Tensor,
    y: torch.Tensor,
    target_shape: Optional[Sequence[int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = [int(v) for v in list(target_shape)] if target_shape is not None else None
    if x.ndim != y.ndim:
        return x, y
    try:
        torch.broadcast_shapes(tuple(x.shape), tuple(y.shape))
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
        if int(target_dim) <= 1:
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
        options = dict(op.get("options", {}))
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        y = executor._resolve_tensor(str(op["inputs"][1]), env)
        target_shape = _target_output_shape(executor, op)
        x, y = _align_binary_inputs(x, y, target_shape)
        z = fn(x, y)
        z = _align_tensor_to_target_shape(z, target_shape)
        executor._assign_outputs(
            op,
            [_apply_fused_activation(z, str(options.get("fusedActivationFunction", "NONE")))],
            env,
        )
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
    new_shape = None
    if len(op["inputs"]) >= 2:
        try:
            new_shape = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
        except RuntimeError:
            new_shape = None
    if new_shape is None:
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
    if any(int(value.ndim) == 0 for value in values):
        values = [value.reshape(1) if int(value.ndim) == 0 else value for value in values]
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


def _kernel_custom(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    custom_code = str(op.get("options", {}).get("customCode", "")).upper()
    if custom_code == "ONNX_SLICE":
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        starts = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
        ends = executor._resolve_tensor(str(op["inputs"][2]), env).to(dtype=torch.int64).reshape(-1).tolist()
        if len(op["inputs"]) >= 4:
            axes = executor._resolve_tensor(str(op["inputs"][3]), env).to(dtype=torch.int64).reshape(-1).tolist()
        else:
            axes = list(range(len(starts)))
        if len(op["inputs"]) >= 5:
            steps = executor._resolve_tensor(str(op["inputs"][4]), env).to(dtype=torch.int64).reshape(-1).tolist()
        else:
            steps = [1 for _ in range(len(starts))]
        slices = [slice(None, None, None) for _ in range(x.ndim)]
        for start, end, axis, step in zip(starts, ends, axes, steps):
            axis_index = int(axis)
            if axis_index < 0:
                axis_index += int(x.ndim)
            slices[axis_index] = slice(int(start), None if int(end) >= int(np.iinfo(np.int64).max // 2) else int(end), int(step))
        executor._assign_outputs(op, [x[tuple(slices)]], env)
        return
    raise RuntimeError(f"Unsupported CUSTOM op in generated PyTorch runtime: customCode={custom_code}")


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
    if "axis" in options and options.get("axis", None) is not None:
        axis = int(options.get("axis", -1))
    else:
        input_meta = executor._metadata.get("tensors", {}).get(str(op["inputs"][0]), {})
        logical_layout = str(input_meta.get("logical_layout", "UNKNOWN")).upper()
        axis = 1 if logical_layout in {"NC", "NCHW", "NCDHW"} and x.ndim >= 2 else -1
    axis = _normalize_dim(axis, x.ndim)
    if beta != 1.0:
        x = x * beta
    y = torch.softmax(x, dim=axis)
    y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
    executor._assign_outputs(op, [y], env)


def _normalize_axes_list(value: Sequence[int], rank: int) -> List[int]:
    normalized: List[int] = []
    for raw_axis in list(value):
        axis = _normalize_dim(int(raw_axis), rank)
        insert_at = 0
        while insert_at < len(normalized) and int(normalized[insert_at]) < axis:
            insert_at += 1
        if insert_at < len(normalized) and int(normalized[insert_at]) == axis:
            continue
        normalized.insert(insert_at, axis)
    return normalized


def _kernel_reduce(
    fn: Callable[[torch.Tensor, Optional[List[int]], bool], torch.Tensor],
) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        keepdims = bool(op.get("options", {}).get("keepDims", True))
        axis: Optional[List[int]] = None
        if len(op["inputs"]) >= 2:
            raw_axis = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
            axis = _normalize_axes_list(raw_axis, x.ndim)
        y = fn(x, axis, keepdims)
        y = _align_tensor_to_target_shape(y, _target_output_shape(executor, op))
        executor._assign_outputs(op, [y], env)
    return _impl


def _reduce_sum(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.sum(x) if not keepdims else torch.sum(x).reshape([1] * x.ndim)
    result = x
    if keepdims:
        for dim in axis:
            result = torch.sum(result, dim=int(dim), keepdim=True)
        return result
    reverse_index = len(axis) - 1
    while reverse_index >= 0:
        result = torch.sum(result, dim=int(axis[reverse_index]), keepdim=False)
        reverse_index -= 1
    return result


def _reduce_mean(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.mean(x) if not keepdims else torch.mean(x).reshape([1] * x.ndim)
    result = x
    if keepdims:
        for dim in axis:
            result = torch.mean(result, dim=int(dim), keepdim=True)
        return result
    reverse_index = len(axis) - 1
    while reverse_index >= 0:
        result = torch.mean(result, dim=int(axis[reverse_index]), keepdim=False)
        reverse_index -= 1
    return result


def _reduce_max(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.amax(x, keepdim=keepdims)
    result = x
    if keepdims:
        for dim in axis:
            result = torch.amax(result, dim=int(dim), keepdim=True)
        return result
    reverse_index = len(axis) - 1
    while reverse_index >= 0:
        result = torch.amax(result, dim=int(axis[reverse_index]), keepdim=False)
        reverse_index -= 1
    return result


def _reduce_min(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        return torch.amin(x, keepdim=keepdims)
    result = x
    if keepdims:
        for dim in axis:
            result = torch.amin(result, dim=int(dim), keepdim=True)
        return result
    reverse_index = len(axis) - 1
    while reverse_index >= 0:
        result = torch.amin(result, dim=int(axis[reverse_index]), keepdim=False)
        reverse_index -= 1
    return result


def _reduce_prod(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        y = torch.prod(x)
        return y if not keepdims else y.reshape([1] * x.ndim)
    result = x
    if keepdims:
        for dim in axis:
            result = torch.prod(result, dim=int(dim), keepdim=True)
        return result
    reverse_index = len(axis) - 1
    while reverse_index >= 0:
        result = torch.prod(result, dim=int(axis[reverse_index]), keepdim=False)
        reverse_index -= 1
    return result


def _reduce_any(x: torch.Tensor, axis: Optional[List[int]], keepdims: bool) -> torch.Tensor:
    if axis is None:
        y = torch.any(x)
        return y if not keepdims else y.reshape([1] * x.ndim)
    result = x
    if keepdims:
        for dim in axis:
            result = torch.any(result, dim=int(dim), keepdim=True)
        return result
    reverse_index = len(axis) - 1
    while reverse_index >= 0:
        result = torch.any(result, dim=int(axis[reverse_index]), keepdim=False)
        reverse_index -= 1
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
    batch_dims = int(op.get("options", {}).get("batchDims", 0))
    if (
        int(batch_dims) == 0
        and int(axis) == 1
        and str(op["inputs"][1]).endswith("_crd_to_dcr_indices")
    ):
        executor._assign_outputs(op, [params], env)
        return
    if batch_dims < 0:
        batch_dims += indices.ndim
    if batch_dims > 0:
        leading_shape = [int(v) for v in list(indices.shape[:batch_dims])]
        flat_batch = int(np.prod(leading_shape, dtype=np.int64))
        params_flat = params.reshape(flat_batch, *params.shape[batch_dims:])
        indices_flat = indices.reshape(flat_batch, *indices.shape[batch_dims:])
        gathered_batches: List[torch.Tensor] = []
        adjusted_axis = int(axis - batch_dims + 1)
        for batch_index in range(flat_batch):
            batch_params = params_flat[batch_index]
            batch_indices = indices_flat[batch_index]
            flat_indices = batch_indices.reshape(-1)
            batch_gathered = torch.index_select(batch_params, adjusted_axis - 1, flat_indices)
            batch_gathered = batch_gathered.reshape(
                *batch_params.shape[: adjusted_axis - 1],
                *batch_indices.shape,
                *batch_params.shape[adjusted_axis:],
            )
            gathered_batches.append(batch_gathered)
        y = torch.stack(gathered_batches, dim=0).reshape(
            *leading_shape,
            *gathered_batches[0].shape,
        )
        executor._assign_outputs(op, [y], env)
        return
    if indices.ndim == 0:
        y = torch.index_select(params, axis, indices.reshape(1)).squeeze(axis)
    else:
        flat_indices = indices.reshape(-1)
        gathered = torch.index_select(params, axis, flat_indices)
        y = gathered.reshape(
            *params.shape[:axis],
            *indices.shape,
            *params.shape[axis + 1 :],
        )
    executor._assign_outputs(op, [y], env)


def _kernel_gather_nd(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    params = executor._resolve_tensor(str(op["inputs"][0]), env)
    indices = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64)
    index_tuple = tuple(indices[..., i] for i in range(indices.shape[-1]))
    y = params[index_tuple]
    executor._assign_outputs(op, [y], env)


def _box_iou(boxes: torch.Tensor, box: torch.Tensor) -> torch.Tensor:
    x1 = torch.maximum(boxes[:, 0], box[0])
    y1 = torch.maximum(boxes[:, 1], box[1])
    x2 = torch.minimum(boxes[:, 2], box[2])
    y2 = torch.minimum(boxes[:, 3], box[3])
    inter_w = torch.clamp(x2 - x1, min=0.0)
    inter_h = torch.clamp(y2 - y1, min=0.0)
    inter = inter_w * inter_h
    boxes_area = torch.clamp(boxes[:, 2] - boxes[:, 0], min=0.0) * torch.clamp(
        boxes[:, 3] - boxes[:, 1], min=0.0
    )
    box_area = torch.clamp(box[2] - box[0], min=0.0) * torch.clamp(box[3] - box[1], min=0.0)
    union = boxes_area + box_area - inter
    safe_union = torch.where(union > 0, union, torch.ones_like(union))
    iou = inter / safe_union
    return torch.where(union > 0, iou, torch.zeros_like(iou))


def _run_non_max_suppression_v4(
    *,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    max_output_size: torch.Tensor,
    iou_threshold: torch.Tensor,
    score_threshold: torch.Tensor,
    pad_to_max_output_size: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    flat_boxes = boxes.to(dtype=torch.float32).reshape(-1, 4)
    flat_scores = scores.to(dtype=torch.float32).reshape(-1)
    max_outputs = max(0, int(max_output_size.reshape(-1)[0].to(dtype=torch.int64).item()))
    selected_tensor = torch.zeros([max_outputs], dtype=torch.int64, device=flat_boxes.device)
    valid_count = torch.zeros([], dtype=torch.int32, device=flat_boxes.device)
    if max_outputs == 0:
        return selected_tensor.to(dtype=torch.int32), valid_count
    iou_thresh = float(iou_threshold.reshape(-1)[0].item())
    score_thresh = float(score_threshold.reshape(-1)[0].item())
    candidate_scores = torch.where(
        flat_scores > score_thresh,
        flat_scores,
        torch.full_like(flat_scores, float("-inf")),
    )
    all_indices = torch.arange(flat_scores.shape[0], dtype=torch.int64, device=flat_boxes.device)
    neg_inf = torch.full_like(candidate_scores, float("-inf"))
    for output_index in range(max_outputs):
        current_score, current_index = torch.max(candidate_scores, dim=0)
        current_index = current_index.to(dtype=torch.int64)
        is_valid = torch.isfinite(current_score)
        selected_tensor[output_index : output_index + 1] = torch.where(
            is_valid,
            current_index,
            torch.zeros_like(current_index),
        ).reshape(1)
        valid_count = valid_count + is_valid.to(dtype=torch.int32)
        current_box = torch.index_select(flat_boxes, 0, current_index.reshape(1)).reshape(4)
        suppress = _box_iou(flat_boxes, current_box) > iou_thresh
        suppress = torch.logical_or(suppress, all_indices == current_index)
        suppress = torch.logical_and(suppress, is_valid)
        candidate_scores = torch.where(suppress, neg_inf, candidate_scores)
    if not pad_to_max_output_size:
        selected_tensor = selected_tensor[: int(valid_count.item())]
    return selected_tensor.to(dtype=torch.int32), valid_count


def _kernel_non_max_suppression_v4(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    boxes = executor._resolve_tensor(str(op["inputs"][0]), env)
    scores = executor._resolve_tensor(str(op["inputs"][1]), env)
    max_output_size = executor._resolve_tensor(str(op["inputs"][2]), env)
    iou_threshold = executor._resolve_tensor(str(op["inputs"][3]), env)
    score_threshold = executor._resolve_tensor(str(op["inputs"][4]), env)
    selected_indices, valid_count = _run_non_max_suppression_v4(
        boxes=boxes,
        scores=scores,
        max_output_size=max_output_size,
        iou_threshold=iou_threshold,
        score_threshold=score_threshold,
    )
    executor._assign_outputs(op, [selected_indices, valid_count], env)


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


def _apply_cumsum(
    x: torch.Tensor,
    *,
    axis: int,
    exclusive: bool,
    reverse: bool,
) -> torch.Tensor:
    dim = _normalize_dim(int(axis), x.ndim)
    y = torch.flip(x, dims=[dim]) if reverse else x
    y = torch.cumsum(y, dim=dim)
    if exclusive:
        axis_size = int(y.shape[dim])
        if axis_size > 0:
            zeros = torch.zeros_like(torch.narrow(y, dim, 0, 1))
            prefix = torch.narrow(y, dim, 0, max(axis_size - 1, 0))
            y = torch.cat([zeros, prefix], dim=dim)
    if reverse:
        y = torch.flip(y, dims=[dim])
    return y


def _kernel_cumsum(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    axis = _coerce_scalar_axis(executor._resolve_tensor(str(op["inputs"][1]), env), device=x.device)
    options = dict(op.get("options", {}))
    executor._assign_outputs(
        op,
        [
            _apply_cumsum(
                x,
                axis=axis,
                exclusive=bool(options.get("exclusive", False)),
                reverse=bool(options.get("reverse", False)),
            )
        ],
        env,
    )


def _kernel_gelu(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
    x = executor._resolve_tensor(str(op["inputs"][0]), env)
    executor._assign_outputs(op, [F.gelu(x)], env)


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
    if x.ndim <= 2 and y.ndim <= 2:
        try:
            z = torch.matmul(x, y)
        except Exception:
            z = torch.matmul(x, y.transpose(-1, -2))
        executor._assign_outputs(op, [_align_tensor_to_target_shape(z, target_shape)], env)
        return
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
    if (
        x.ndim == 4
        and w.ndim == 4
        and int(w.shape[-1]) > 0
        and int(x.shape[-1]) == int(w.shape[-1])
    ):
        x = x.permute(0, 3, 1, 2).contiguous()
        w = w.permute(0, 3, 1, 2).contiguous()
    elif (
        x.ndim == 4
        and w.ndim == 4
        and int(w.shape[-1]) > 0
        and int(x.shape[1]) == int(w.shape[-1])
        and int(w.shape[1]) != int(x.shape[1])
    ):
        w = w.permute(0, 3, 1, 2).contiguous()
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
    depth_multiplier = max(1, int(options.get("depthMultiplier", 1)))
    if (
        x.ndim == 4
        and w.ndim == 4
        and int(w.shape[0]) == 1
        and int(x.shape[-1]) > 0
        and int(w.shape[-1]) % int(x.shape[-1]) == 0
    ):
        x = x.permute(0, 3, 1, 2).contiguous()
        w = w.permute(3, 0, 1, 2).contiguous()
    elif (
        x.ndim == 4
        and w.ndim == 4
        and int(w.shape[0]) == 1
        and int(x.shape[1]) > 0
        and int(w.shape[-1]) % int(x.shape[1]) == 0
    ):
        w = w.permute(3, 0, 1, 2).contiguous()
    expected_in_channels = max(1, int(w.shape[0]) // depth_multiplier)
    if x.ndim == 4 and int(x.shape[1]) != expected_in_channels and int(x.shape[-1]) == expected_in_channels:
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
    if x.ndim == 4 and int(x.shape[1]) != int(w.shape[0]) and int(x.shape[-1]) == int(w.shape[0]):
        x = x.permute(0, 3, 1, 2).contiguous()
    stride = (int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    padding = _resolve_padding_2d(padding=str(options.get("padding", "SAME")), weight=w, dilation=(1, 1))
    raw = F.conv_transpose2d(x, w, bias=b, stride=stride, padding=padding)
    target_shape = _target_output_shape(executor, op)
    target_h, target_w = _infer_spatial_shape_for_transposed_conv2d(
        raw_output=raw,
        target_shape=target_shape,
        fallback_shape=output_shape,
    )
    y = raw[..., : target_h, : target_w]
    y = _align_tensor_to_target_shape(y, target_shape)
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
    if x.ndim == 5 and int(x.shape[1]) != int(w.shape[0]) and int(x.shape[-1]) == int(w.shape[0]):
        x = x.permute(0, 4, 1, 2, 3).contiguous()
    stride = (int(options.get("strideD", 1)), int(options.get("strideH", 1)), int(options.get("strideW", 1)))
    padding = _resolve_padding_3d(padding=str(options.get("padding", "SAME")), weight=w, dilation=(1, 1, 1))
    raw = F.conv_transpose3d(x, w, bias=b, stride=stride, padding=padding)
    target_shape = _target_output_shape(executor, op)
    target_d, target_h, target_w = _infer_spatial_shape_for_transposed_conv3d(
        raw_output=raw,
        target_shape=target_shape,
        fallback_shape=output_shape,
    )
    y = raw[..., : target_d, : target_h, : target_w]
    y = _align_tensor_to_target_shape(y, target_shape)
    executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)


def _kernel_pool2d(is_max_pool: bool) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        options = dict(op.get("options", {}))
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        target_shape = _target_output_shape(executor, op)
        kernel_size = (int(options.get("filterHeight", 1)), int(options.get("filterWidth", 1)))
        stride = (int(options.get("strideH", 1)), int(options.get("strideW", 1)))
        padding_mode = str(options.get("padding", "SAME"))
        padding = (
            int((kernel_size[0] - 1) // 2) if padding_mode.upper() == "SAME" else 0,
            int((kernel_size[1] - 1) // 2) if padding_mode.upper() == "SAME" else 0,
        )
        channel_last = False
        pool_input = x
        if x.ndim == 4 and _should_resize_as_channel_last(executor, op, x, target_shape):
            channel_last = True
            pool_input = x.permute(0, 3, 1, 2).contiguous()
        if is_max_pool:
            y = F.max_pool2d(pool_input, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            y = F.avg_pool2d(pool_input, kernel_size=kernel_size, stride=stride, padding=padding)
        if channel_last:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = _align_tensor_to_target_shape(y, target_shape)
        executor._assign_outputs(op, [_apply_fused_activation(y, str(options.get("fusedActivationFunction", "NONE")))], env)
    return _impl


def _kernel_resize(method: str) -> Callable[[_GraphExecutor, Dict[str, Any], Dict[str, torch.Tensor]], None]:
    def _resize_bilinear_exact(
        x: torch.Tensor,
        size: Sequence[int],
        *,
        align_corners: bool,
        half_pixel_centers: bool,
    ) -> torch.Tensor:
        if x.ndim != 4:
            return F.interpolate(x, size=[int(size[0]), int(size[1])], mode="bilinear", align_corners=align_corners)
        out_h = int(size[0])
        out_w = int(size[1])
        in_h = int(x.shape[-2])
        in_w = int(x.shape[-1])
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

    def _impl(executor: _GraphExecutor, op: Dict[str, Any], env: Dict[str, torch.Tensor]) -> None:
        x = executor._resolve_tensor(str(op["inputs"][0]), env)
        target_shape = _target_output_shape(executor, op)
        if len(op["inputs"]) >= 2:
            size = executor._resolve_tensor(str(op["inputs"][1]), env).to(dtype=torch.int64).reshape(-1).tolist()
        else:
            size = [
                int(op.get("options", {}).get("newHeight", int(x.shape[-2]))),
                int(op.get("options", {}).get("newWidth", int(x.shape[-1]))),
            ]
        resize_as_channel_last = _should_resize_as_channel_last(
            executor,
            op,
            x,
            target_shape,
        )
        resize_input = x
        resize_size = [int(size[0]), int(size[1])]
        if resize_as_channel_last and x.ndim == 4:
            resize_input = x.permute(0, 3, 1, 2).contiguous()
            if target_shape is not None and len(list(target_shape)) == 4:
                resize_size = [int(target_shape[1]), int(target_shape[2])]
        if method == "nearest":
            y = F.interpolate(resize_input, size=resize_size, mode="nearest")
        else:
            y = _resize_bilinear_exact(
                resize_input,
                resize_size,
                align_corners=bool(op.get("options", {}).get("alignCorners", False)),
                half_pixel_centers=bool(op.get("options", {}).get("halfPixelCenters", False)),
            )
        if resize_as_channel_last and x.ndim == 4:
            y = y.permute(0, 2, 3, 1).contiguous()
        y = _align_tensor_to_target_shape(y, target_shape)
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
        "NON_MAX_SUPPRESSION_V4": _kernel_non_max_suppression_v4,
        "SCATTER_ND": _kernel_scatter_nd,
        "TILE": _kernel_tile,
        "BROADCAST_TO": _kernel_broadcast_to,
        "ARG_MAX": _kernel_arg(is_max=True),
        "ARG_MIN": _kernel_arg(is_max=False),
        "TOPK_V2": _kernel_topk,
        "CUSTOM": _kernel_custom,
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
        "ATAN": _kernel_unary(torch.atan),
        "CEIL": _kernel_unary(torch.ceil),
        "COS": _kernel_unary(torch.cos),
        "ELU": _kernel_unary(F.elu),
        "EXP": _kernel_unary(torch.exp),
        "FLOOR": _kernel_unary(torch.floor),
        "GELU": _kernel_gelu,
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


SUPPORTED_TORCH_KERNEL_OP_TYPES = {
    op_type for op_type in _register_supported_kernels().keys() if str(op_type) != "CUSTOM"
}


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
            inputs = {
                str(name): _resolve_named_input_value(kwargs, str(name))
                for name in self.input_names
            }
        else:
            if len(args) != len(self.input_names):
                raise RuntimeError(
                    f"Input arity mismatch. expected={len(self.input_names)} actual={len(args)}"
                )
            inputs = {str(name): value for name, value in zip(self.input_names, args)}
        env = self._executor.run(inputs)
        outputs = [self._executor._resolve_tensor(str(name), env) for name in self.output_names]
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward_named(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> Dict[str, torch.Tensor]:
        result = self.forward(*args, **kwargs)
        if len(self.output_names) == 1:
            return {str(self.output_names[0]): result}
        return {str(name): value for name, value in zip(self.output_names, result)}


GeneratedModelBase = _GeneratedModel


def prepare_generated_model_metadata(
    *,
    metadata: Dict[str, Any],
    raw_state_dict: Dict[str, Any],
) -> Dict[str, Any]:
    prepared = copy.deepcopy(metadata)
    storage_name_map = _tensor_storage_name_map_from_metadata(prepared)
    for tensor_name, tensor_meta in prepared.get("tensors", {}).items():
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
    return prepared


def load_generated_model_weights(
    *,
    model: torch.nn.Module,
    metadata: Dict[str, Any],
    raw_state_dict: Dict[str, Any],
    device: Optional[str] = None,
) -> torch.nn.Module:
    storage_name_map = _tensor_storage_name_map_from_metadata(metadata)
    normalized_state_dict: Dict[str, torch.Tensor] = {}
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
        if isinstance(tensor_value, torch.Tensor):
            normalized_state_dict[str(storage_key)] = tensor_value
    model.load_state_dict(normalized_state_dict, strict=False)
    if device is not None:
        model = model.to(device)
    return model


class _TFLiteBackedGeneratedModel(torch.nn.Module):
    def __init__(
        self,
        *,
        metadata: Dict[str, Any],
        package_dir: str,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        self._metadata = metadata
        self._package_dir = str(package_dir)
        self._device = device
        self.input_names = list(metadata.get("inputs", []))
        self.output_names = list(metadata.get("outputs", []))
        tflite_file_name = str(metadata.get("tflite_file_name", "model_float32.tflite"))
        self._tflite_path = os.path.join(self._package_dir, tflite_file_name)
        self._interpreter = _create_tflite_interpreter(self._tflite_path)
        self._interpreter.allocate_tensors()

    def _input_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError("Use either positional inputs or keyword inputs, not both.")
        if len(kwargs) > 0:
            return {
                str(name): _resolve_named_input_value(kwargs, str(name))
                for name in self.input_names
            }
        if len(args) != len(self.input_names):
            raise RuntimeError(
                f"Input arity mismatch. expected={len(self.input_names)} actual={len(args)}"
            )
        return {str(name): value for name, value in zip(self.input_names, args)}

    def _target_shape(self, tensor_name: str) -> Optional[List[int]]:
        boundary_map = self._metadata.get("boundary_shape_signatures", {})
        if isinstance(boundary_map, dict):
            boundary_shape = boundary_map.get(str(tensor_name), None)
            if isinstance(boundary_shape, list):
                return [max(1, int(v)) if int(v) >= 0 else 1 for v in boundary_shape]
        tensor_meta = self._metadata.get("tensors", {}).get(str(tensor_name), {})
        shape_sig = tensor_meta.get("shape_signature", tensor_meta.get("shape", None))
        if isinstance(shape_sig, list):
            return [max(1, int(v)) if int(v) >= 0 else 1 for v in shape_sig]
        return None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._input_dict(*args, **kwargs)
        input_details = self._interpreter.get_input_details()
        input_map = _build_tflite_detail_map(
            onnx_names=self.input_names,
            tflite_details=input_details,
        )

        adapted_inputs: Dict[str, np.ndarray] = {}
        for input_name in self.input_names:
            detail = input_map[str(input_name)]
            array = _coerce_input_to_numpy(inputs[str(input_name)])
            adapted = _adapt_input_layout_for_tflite_input(array, detail)
            target_dtype = np.dtype(detail["dtype"])
            if _numpy_dtype_is_string(np.asarray(adapted).dtype) or _numpy_dtype_is_string(target_dtype):
                if target_dtype.kind == "S":
                    adapted_inputs[str(input_name)] = np.asarray(adapted).astype(np.bytes_)
                elif target_dtype.kind == "U":
                    adapted_inputs[str(input_name)] = np.asarray(adapted).astype(str)
                else:
                    adapted_inputs[str(input_name)] = np.asarray(adapted, dtype=object)
            else:
                adapted_inputs[str(input_name)] = _quantize_for_tflite_input(adapted, detail)

        signature_list_fn = getattr(self._interpreter, "get_signature_list", None)
        signature_runner_fn = getattr(self._interpreter, "get_signature_runner", None)
        if callable(signature_list_fn) and callable(signature_runner_fn):
            signature_list = signature_list_fn()
            if isinstance(signature_list, dict) and len(signature_list) > 0:
                signature_key = next(iter(signature_list.keys()))
                signature_meta = signature_list[signature_key]
                signature_inputs = [str(v) for v in list(signature_meta.get("inputs", []))]
                signature_outputs = [str(v) for v in list(signature_meta.get("outputs", []))]
                runner_inputs: Dict[str, np.ndarray] = {}
                for input_name in self.input_names:
                    assigned_name = None
                    normalized_input_name = _normalize_tensor_name(str(input_name))
                    canonical_input_name = _canonical_tensor_name(str(input_name))
                    for candidate in signature_inputs:
                        normalized_candidate = _normalize_tensor_name(str(candidate))
                        canonical_candidate = _canonical_tensor_name(str(candidate))
                        if (
                            str(candidate) == str(input_name)
                            or normalized_candidate == normalized_input_name
                            or canonical_candidate == canonical_input_name
                            or str(candidate).endswith(f"_{normalized_input_name}")
                        ):
                            assigned_name = str(candidate)
                            break
                    if assigned_name is None and len(signature_inputs) == 1:
                        assigned_name = str(signature_inputs[0])
                    if assigned_name is not None:
                        runner_inputs[str(assigned_name)] = adapted_inputs[str(input_name)]
                if len(runner_inputs) > 0:
                    raw_outputs = signature_runner_fn(signature_key=signature_key)(**runner_inputs)
                    output_lookup = {
                        str(name): value
                        for name, value in raw_outputs.items()
                    }
                    for name, value in list(output_lookup.items()):
                        output_lookup.setdefault(_normalize_tensor_name(str(name)), value)
                    outputs: List[Any] = []
                    for output_name in self.output_names:
                        value = output_lookup.get(str(output_name))
                        if value is None:
                            value = output_lookup.get(_normalize_tensor_name(str(output_name)))
                        if value is None and len(signature_outputs) == 1:
                            value = raw_outputs[str(signature_outputs[0])]
                        if value is None:
                            raise RuntimeError(f"Signature output was not found. output_name={output_name}")
                        output_array = np.asarray(value)
                        output_array = _align_numpy_to_target_shape(
                            output_array,
                            self._target_shape(str(output_name)),
                        )
                        outputs.append(_coerce_output_value(output_array, device=self._device))
                    if len(outputs) == 1:
                        return outputs[0]
                    return tuple(outputs)

        resized = _resize_tflite_inputs_if_needed(
            interpreter=self._interpreter,
            onnx_input_names=self.input_names,
            tflite_input_map=input_map,
            adapted_inputs=adapted_inputs,
        )
        if resized:
            self._interpreter.allocate_tensors()
            input_details = self._interpreter.get_input_details()
            input_map = _build_tflite_detail_map(
                onnx_names=self.input_names,
                tflite_details=input_details,
            )

        output_details = self._interpreter.get_output_details()
        output_map = _build_tflite_detail_map(
            onnx_names=self.output_names,
            tflite_details=output_details,
        )

        assigned_indices: Set[int] = set()
        for input_name in self.input_names:
            detail = input_map[str(input_name)]
            detail_index = int(detail["index"])
            self._interpreter.set_tensor(detail_index, adapted_inputs[str(input_name)])
            assigned_indices.add(detail_index)
        for detail in input_details:
            detail_index = int(detail["index"])
            if detail_index in assigned_indices:
                continue
            self._interpreter.set_tensor(
                detail_index,
                _default_value_for_tflite_detail(detail),
            )
        _invoke_tflite_with_recovery(self._interpreter)

        outputs: List[Any] = []
        for output_name in self.output_names:
            detail = output_map[str(output_name)]
            raw = self._interpreter.get_tensor(int(detail["index"]))
            output_array = np.asarray(raw)
            target_meta = self._metadata.get("tensors", {}).get(str(output_name), {})
            target_dtype = str(target_meta.get("dtype", "")).upper()
            if (
                target_dtype.startswith("FLOAT")
                and np.issubdtype(output_array.dtype, np.integer)
            ):
                output_array = _dequantize_tflite_output(output_array, detail)
            output_array = _align_numpy_to_target_shape(
                output_array,
                self._target_shape(str(output_name)),
            )
            outputs.append(_coerce_output_value(output_array, device=self._device))

        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward_named(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.forward(*args, **kwargs)
        if len(self.output_names) == 1:
            return {str(self.output_names[0]): result}
        return {str(name): value for name, value in zip(self.output_names, result)}


class _StringNormalizerGeneratedModel(torch.nn.Module):
    def __init__(self, *, metadata: Dict[str, Any]) -> None:
        super().__init__()
        self._metadata = metadata
        self.input_names = list(metadata.get("inputs", []))
        self.output_names = list(metadata.get("outputs", []))
        self._config = dict(metadata.get("string_normalizer", {}))

    def _input_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError("Use either positional inputs or keyword inputs, not both.")
        if len(kwargs) > 0:
            return {
                str(name): _resolve_named_input_value(kwargs, str(name))
                for name in self.input_names
            }
        if len(args) != len(self.input_names):
            raise RuntimeError(
                f"Input arity mismatch. expected={len(self.input_names)} actual={len(args)}"
            )
        return {str(name): value for name, value in zip(self.input_names, args)}

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._input_dict(*args, **kwargs)
        input_name = str(self.input_names[0])
        array = np.asarray(_coerce_input_to_numpy(inputs[input_name]), dtype=object)
        flat = array.reshape(-1)
        case_action = str(self._config.get("case_change_action", "")).upper()
        is_case_sensitive = bool(self._config.get("is_case_sensitive", True))
        stopwords = [str(v) for v in list(self._config.get("stopwords", []))]
        stopword_set = (
            set(stopwords)
            if is_case_sensitive
            else {word.lower() for word in stopwords}
        )

        normalized: List[str] = []
        for raw in flat.tolist():
            token = raw.decode("utf-8") if isinstance(raw, bytes) else str(raw)
            if case_action == "LOWER":
                transformed = token.lower()
            elif case_action == "UPPER":
                transformed = token.upper()
            else:
                transformed = token
            compare_token = transformed if is_case_sensitive else transformed.lower()
            if compare_token in stopword_set:
                normalized.append("")
            else:
                normalized.append(transformed)
        output = np.asarray(normalized, dtype=object).reshape(array.shape)
        if output.ndim == 0:
            return output.reshape(()).item()
        return output

    def forward_named(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        return {str(self.output_names[0]): self.forward(*args, **kwargs)}


class _SavedModelBackedGeneratedModel(torch.nn.Module):
    def __init__(
        self,
        *,
        metadata: Dict[str, Any],
        package_dir: str,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()
        import tensorflow as tf

        self._tf = tf
        self._metadata = metadata
        self._device = device
        self.input_names = list(metadata.get("inputs", []))
        self.output_names = list(metadata.get("outputs", []))
        saved_model_dir_name = str(metadata.get("saved_model_dir_name", "saved_model"))
        saved_model_path = os.path.join(str(package_dir), saved_model_dir_name)
        module = tf.saved_model.load(saved_model_path)
        signatures = getattr(module, "signatures", {})
        signature_callable = None
        if hasattr(signatures, "get"):
            signature_callable = signatures.get("serving_default", None)
        elif isinstance(signatures, dict) and "serving_default" in signatures:
            signature_callable = signatures["serving_default"]
        if callable(signature_callable):
            self._callable = signature_callable
        elif callable(module):
            self._callable = module
        else:
            raise RuntimeError(
                "SavedModel-backed PyTorch package could not resolve a callable endpoint. "
                f"path={saved_model_path}"
            )
        self._signature_input_name_map: Dict[str, str] = {
            str(name): str(name) for name in self.input_names
        }
        self._input_signature_shapes: Dict[str, List[Optional[int]]] = {}
        structured_input_signature = getattr(self._callable, "structured_input_signature", None)
        if (
            isinstance(structured_input_signature, tuple)
            and len(structured_input_signature) == 2
            and isinstance(structured_input_signature[1], dict)
        ):
            signature_inputs = {
                str(input_name): tensor_spec
                for input_name, tensor_spec in structured_input_signature[1].items()
            }
            for input_name, tensor_spec in signature_inputs.items():
                shape = getattr(tensor_spec, "shape", None)
                if shape is None:
                    continue
                self._input_signature_shapes[str(input_name)] = [
                    None if dim is None else int(dim)
                    for dim in list(shape)
                ]
            signature_names = list(signature_inputs.keys())
            for input_name in self.input_names:
                assigned_name = None
                normalized_input_name = _normalize_tensor_name(str(input_name))
                canonical_input_name = _canonical_tensor_name(str(input_name))
                for candidate in signature_names:
                    normalized_candidate = _normalize_tensor_name(str(candidate))
                    canonical_candidate = _canonical_tensor_name(str(candidate))
                    if (
                        str(candidate) == str(input_name)
                        or normalized_candidate == normalized_input_name
                        or canonical_candidate == canonical_input_name
                        or str(candidate).endswith(f"_{normalized_input_name}")
                    ):
                        assigned_name = str(candidate)
                        break
                if assigned_name is not None:
                    self._signature_input_name_map[str(input_name)] = assigned_name

    def _input_dict(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        if len(args) > 0 and len(kwargs) > 0:
            raise RuntimeError("Use either positional inputs or keyword inputs, not both.")
        if len(kwargs) > 0:
            return {
                str(name): _resolve_named_input_value(kwargs, str(name))
                for name in self.input_names
            }
        if len(args) != len(self.input_names):
            raise RuntimeError(
                f"Input arity mismatch. expected={len(self.input_names)} actual={len(args)}"
            )
        return {str(name): value for name, value in zip(self.input_names, args)}

    def _target_shape(self, tensor_name: str) -> Optional[List[int]]:
        boundary_map = self._metadata.get("boundary_shape_signatures", {})
        if isinstance(boundary_map, dict):
            boundary_shape = boundary_map.get(str(tensor_name), None)
            if isinstance(boundary_shape, list):
                return [max(1, int(v)) if int(v) >= 0 else 1 for v in boundary_shape]
        tensor_meta = self._metadata.get("tensors", {}).get(str(tensor_name), {})
        shape_sig = tensor_meta.get("shape_signature", tensor_meta.get("shape", None))
        if isinstance(shape_sig, list):
            return [max(1, int(v)) if int(v) >= 0 else 1 for v in shape_sig]
        return None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        inputs = self._input_dict(*args, **kwargs)
        tf_inputs: Dict[str, Any] = {}
        for input_name in self.input_names:
            signature_input_name = self._signature_input_name_map.get(
                str(input_name),
                str(input_name),
            )
            value = _coerce_input_to_numpy(inputs[str(input_name)])
            array = _align_numpy_to_signature_shape(
                np.asarray(value),
                self._input_signature_shapes.get(str(signature_input_name)),
            )
            if _numpy_dtype_is_string(array.dtype):
                tf_inputs[str(signature_input_name)] = self._tf.convert_to_tensor(
                    np.asarray(array, dtype=str),
                    dtype=self._tf.string,
                )
            else:
                tf_inputs[str(signature_input_name)] = self._tf.convert_to_tensor(array)

        outputs_raw = self._callable(**tf_inputs)
        if isinstance(outputs_raw, dict):
            outputs_map = {
                str(name): value
                for name, value in outputs_raw.items()
            }
        else:
            outputs_map = {
                str(self.output_names[0]): outputs_raw
            }

        normalized_output_map: Dict[str, Any] = {}
        for name, value in outputs_map.items():
            key = str(name)
            normalized_output_map[key] = value
            normalized_output_map.setdefault(_normalize_tensor_name(key), value)
        outputs: List[Any] = []
        for output_name in self.output_names:
            value = normalized_output_map.get(str(output_name))
            if value is None:
                value = normalized_output_map.get(_normalize_tensor_name(str(output_name)))
            if value is None:
                raise RuntimeError(f"SavedModel output was not found. output_name={output_name}")
            array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
            array = _align_numpy_to_target_shape(array, self._target_shape(str(output_name)))
            outputs.append(_coerce_output_value(array, device=self._device))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def forward_named(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = self.forward(*args, **kwargs)
        if len(self.output_names) == 1:
            return {str(self.output_names[0]): result}
        return {str(name): value for name, value in zip(self.output_names, result)}


def load_generated_model_package(
    *,
    package_dir: str,
    device: Optional[str] = None,
    eval_mode: bool = True,
) -> torch.nn.Module:
    metadata_path = os.path.join(package_dir, "metadata.json")
    state_dict_path = os.path.join(package_dir, "state_dict.pth")
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    if str(metadata.get("execution_backend", "")).lower() == "string_normalizer":
        model = _StringNormalizerGeneratedModel(metadata=metadata)
        if eval_mode:
            model.eval()
        return model
    if str(metadata.get("execution_backend", "")).lower() == "saved_model":
        model = _SavedModelBackedGeneratedModel(
            metadata=metadata,
            package_dir=package_dir,
            device=device,
        )
        if eval_mode:
            model.eval()
        return model
    if str(metadata.get("execution_backend", "")).lower() == "tflite":
        model = _TFLiteBackedGeneratedModel(
            metadata=metadata,
            package_dir=package_dir,
            device=device,
        )
        if eval_mode:
            model.eval()
        return model
    raw_state_dict = torch.load(state_dict_path, map_location=device or "cpu")
    metadata = prepare_generated_model_metadata(metadata=metadata, raw_state_dict=raw_state_dict)
    model = _GeneratedModel(metadata=metadata)
    model = load_generated_model_weights(
        model=model,
        metadata=metadata,
        raw_state_dict=raw_state_dict,
        device=device,
    )
    if eval_mode:
        model.eval()
    return model
