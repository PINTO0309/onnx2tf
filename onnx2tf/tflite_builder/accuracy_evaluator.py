from __future__ import annotations

import itertools
import json
import os
import queue as queue_module
import re
import hashlib
import time
import traceback
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
from onnx2tf.utils.tempdir_cleanup import (
    cleanup_managed_tempdir,
    make_managed_tempdir,
)

if TYPE_CHECKING:
    from ai_edge_litert.interpreter import Interpreter as LiteRTInterpreter


def _sanitize_tensor_name_for_path(name: str) -> str:
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(name))
    safe = re.sub(r"_+", "_", safe).strip("._")
    if safe == "":
        return "tensor"
    return safe


def _make_safe_memmap_filename(
    *,
    file_prefix: str,
    index: int,
    tensor_name: str,
    suffix: str = ".npy",
    max_stem_len: int = 220,
) -> str:
    safe_name = _sanitize_tensor_name_for_path(tensor_name)
    stem = f"{file_prefix}_{int(index)}_{safe_name}"
    if len(stem) <= int(max_stem_len):
        return f"{stem}{suffix}"
    digest = hashlib.sha1(str(tensor_name).encode("utf-8")).hexdigest()[:16]
    reserved = len(f"{file_prefix}_{int(index)}__{digest}")
    available = max(1, int(max_stem_len) - int(reserved))
    shortened = safe_name[:available].rstrip("._")
    if shortened == "":
        shortened = "tensor"
    return f"{file_prefix}_{int(index)}_{shortened}_{digest}{suffix}"


def _write_named_arrays_to_memmap_files(
    *,
    arrays: Dict[str, np.ndarray],
    memmap_dir: str,
    file_prefix: str,
) -> Dict[str, Dict[str, Any]]:
    os.makedirs(memmap_dir, exist_ok=True)
    manifest: Dict[str, Dict[str, Any]] = {}
    for idx, (name, value) in enumerate(arrays.items()):
        array = np.asarray(value)
        output_path = os.path.join(
            memmap_dir,
            _make_safe_memmap_filename(
                file_prefix=str(file_prefix),
                index=int(idx),
                tensor_name=str(name),
            ),
        )
        mm = np.lib.format.open_memmap(
            output_path,
            mode="w+",
            dtype=array.dtype,
            shape=array.shape,
        )
        mm[...] = array
        mm.flush()
        manifest[str(name)] = {
            "path": output_path,
            "shape": [int(v) for v in array.shape],
            "dtype": str(array.dtype),
        }
    return manifest


def _load_named_arrays_from_memmap_manifest(
    *,
    manifest: Dict[str, Dict[str, Any]],
) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    for name, meta in manifest.items():
        path = str(meta.get("path", ""))
        if path == "":
            raise RuntimeError(
                "Missing memmap path in worker output manifest. "
                f"name={name}"
            )
        arrays[str(name)] = np.load(path, mmap_mode="r")
    return arrays


def _cleanup_memmap_manifest(
    manifest: Optional[Dict[str, Dict[str, Any]]],
) -> None:
    if not isinstance(manifest, dict):
        return
    for meta in manifest.values():
        if not isinstance(meta, dict):
            continue
        path = str(meta.get("path", ""))
        if path != "" and os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


def _load_onnx_inputs_from_payload(
    payload: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    manifest_raw = payload.get("onnx_inputs_memmap_manifest", None)
    if isinstance(manifest_raw, dict):
        manifest = {
            str(k): v
            for k, v in manifest_raw.items()
        }
        return _load_named_arrays_from_memmap_manifest(
            manifest=manifest,
        )
    raw_inputs = payload.get("onnx_inputs", {})
    return {
        str(name): np.asarray(value)
        for name, value in raw_inputs.items()
    }


def _estimate_onnx_graph_output_bytes(
    *,
    onnx_graph: onnx.ModelProto,
    onnx_output_names: Sequence[str],
) -> int:
    if onnx_graph is None or onnx_graph.graph is None:
        return 0
    value_info_map = {str(v.name): v for v in onnx_graph.graph.value_info}
    for graph_output in onnx_graph.graph.output:
        value_info_map[str(graph_output.name)] = graph_output
    type_size_map = {
        np.dtype("float16"): 2,
        np.dtype("float32"): 4,
        np.dtype("float64"): 8,
        np.dtype("uint8"): 1,
        np.dtype("uint16"): 2,
        np.dtype("uint32"): 4,
        np.dtype("uint64"): 8,
        np.dtype("int8"): 1,
        np.dtype("int16"): 2,
        np.dtype("int32"): 4,
        np.dtype("int64"): 8,
        np.dtype("bool_"): 1,
    }
    total_bytes = 0
    for output_name in onnx_output_names:
        value_info = value_info_map.get(str(output_name), None)
        if value_info is None:
            continue
        tensor_type = value_info.type.tensor_type
        try:
            np_dtype = np.dtype(
                onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
            )
        except Exception:
            np_dtype = np.dtype("float32")
        numel = 1
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and int(dim.dim_value) > 0:
                numel *= int(dim.dim_value)
            else:
                numel *= 1
        total_bytes += int(numel) * int(type_size_map.get(np_dtype, 4))
    return int(total_bytes)


def _resolve_eval_memmap_config(
    *,
    onnx_graph: onnx.ModelProto,
    onnx_output_names: Sequence[str],
    onnxruntime_output_memmap: bool,
    onnxruntime_output_memmap_dir: Optional[str],
) -> Tuple[bool, Optional[str]]:
    if not bool(onnxruntime_output_memmap):
        return False, None

    estimated_output_bytes = _estimate_onnx_graph_output_bytes(
        onnx_graph=onnx_graph,
        onnx_output_names=onnx_output_names,
    )

    memmap_dir = onnxruntime_output_memmap_dir
    cleanup_dir = False
    if memmap_dir is None or str(memmap_dir).strip() == "":
        memmap_dir = make_managed_tempdir(
            prefix="onnx2tf_eval_mm_",
            stale_prefixes=["onnx2tf_eval_mm_", "onnx2tf_eval_in_mm_"],
        )
        cleanup_dir = True
    os.makedirs(memmap_dir, exist_ok=True)

    try:
        if estimated_output_bytes > 0:
            try:
                import psutil
                # ONNX and TFLite worker outputs are persisted independently.
                required_bytes = int(estimated_output_bytes) * 2
                disk_free = int(psutil.disk_usage(memmap_dir).free)
                if required_bytes > disk_free:
                    if cleanup_dir:
                        cleanup_managed_tempdir(str(memmap_dir))
                    raise RuntimeError(
                        "Not enough disk space for evaluation memmap outputs. "
                        f"required={required_bytes} free={disk_free} dir={memmap_dir}"
                    )
            except ImportError:
                pass
    except Exception:
        if cleanup_dir:
            cleanup_managed_tempdir(str(memmap_dir))
        raise
    return True, memmap_dir


def _resolve_eval_input_memmap_dir(
    *,
    onnxruntime_output_memmap: bool,
    onnxruntime_output_memmap_dir: Optional[str],
    fallback_dir: Optional[str] = None,
) -> Optional[str]:
    if not bool(onnxruntime_output_memmap):
        return None
    if fallback_dir is not None and str(fallback_dir).strip() != "":
        os.makedirs(fallback_dir, exist_ok=True)
        return str(fallback_dir)
    if (
        onnxruntime_output_memmap_dir is not None
        and str(onnxruntime_output_memmap_dir).strip() != ""
    ):
        os.makedirs(onnxruntime_output_memmap_dir, exist_ok=True)
        return str(onnxruntime_output_memmap_dir)

    memmap_dir = make_managed_tempdir(
        prefix="onnx2tf_eval_in_mm_",
        stale_prefixes=["onnx2tf_eval_mm_", "onnx2tf_eval_in_mm_"],
    )
    return memmap_dir


def _normalize_tensor_name(name: str) -> str:
    normalized = str(name).split(":")[0]
    # SignatureDef-based TF/TFLite exports often prefix external I/O names
    # with `serving_default_` (e.g. serving_default_input:0).
    if normalized.startswith("serving_default_"):
        normalized = normalized[len("serving_default_") :]
    return normalized


def _create_tflite_interpreter(model_path: str) -> "LiteRTInterpreter":
    from ai_edge_litert.interpreter import (
        Interpreter,
        OpResolverType,
    )

    try:
        # Disable default delegates (e.g. XNNPACK) during evaluation to reduce
        # runtime crashes on some dynamic-shape models.
        return Interpreter(
            model_path=model_path,
            experimental_preserve_all_tensors=True,
            experimental_op_resolver_type=OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES,
        )
    except TypeError:
        return Interpreter(model_path=model_path)


def _load_custom_input_data(
    custom_input_op_name_np_data_path: Optional[List[Any]],
) -> Dict[str, np.ndarray]:
    custom_inputs: Dict[str, np.ndarray] = {}
    if not custom_input_op_name_np_data_path:
        return custom_inputs
    for param in custom_input_op_name_np_data_path:
        if not isinstance(param, (list, tuple)) or len(param) < 2:
            continue
        input_name = str(param[0])
        numpy_file_path = str(param[1])
        if not os.path.exists(numpy_file_path):
            raise FileNotFoundError(
                "Evaluation input file does not exist. "
                f"input_name={input_name} path={numpy_file_path}"
            )
        custom_inputs[input_name] = np.asarray(np.load(numpy_file_path))
    return custom_inputs


def _collect_onnx_input_specs(
    onnx_graph: onnx.ModelProto,
) -> List[Tuple[str, np.dtype, Tuple[int, ...]]]:
    def _dynamic_dim_default() -> int:
        raw = str(os.environ.get("ONNX2TF_EVAL_DYNAMIC_DIM_DEFAULT", "16")).strip()
        try:
            value = int(raw)
        except Exception:
            value = 16
        return max(1, int(value))

    def _resolve_dynamic_dim(
        *,
        rank: int,
        axis: int,
        known_dims: List[int],
    ) -> int:
        if int(axis) == 0:
            return 1
        default_dim = _dynamic_dim_default()

        if int(rank) == 4:
            c_first = int(known_dims[1]) if int(known_dims[1]) > 0 else -1
            c_last = int(known_dims[3]) if int(known_dims[3]) > 0 else -1

            # NCHW-like input: [N, C, H, W]
            if c_first in (1, 3, 4) and int(axis) in (2, 3):
                partner_axis = 3 if int(axis) == 2 else 2
                partner = int(known_dims[partner_axis])
                if partner > 0:
                    return int(partner)
                return max(16, int(default_dim))

            # NHWC-like input: [N, H, W, C]
            if c_last in (1, 3, 4) and int(axis) in (1, 2):
                partner_axis = 2 if int(axis) == 1 else 1
                partner = int(known_dims[partner_axis])
                if partner > 0:
                    return int(partner)
                return max(16, int(default_dim))

        return int(default_dim)

    initializer_names = {initializer.name for initializer in onnx_graph.graph.initializer}
    input_specs: List[Tuple[str, np.dtype, Tuple[int, ...]]] = []
    for graph_input in onnx_graph.graph.input:
        if graph_input.name in initializer_names:
            continue
        tensor_type = graph_input.type.tensor_type
        np_dtype = np.dtype(onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type))
        raw_shape: List[int] = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value") and int(dim.dim_value) > 0:
                raw_shape.append(int(dim.dim_value))
            else:
                raw_shape.append(0)

        shape: List[int] = []
        rank = int(len(raw_shape))
        for axis, dim_value in enumerate(raw_shape):
            if int(dim_value) > 0:
                shape.append(int(dim_value))
            else:
                shape.append(
                    _resolve_dynamic_dim(
                        rank=int(rank),
                        axis=int(axis),
                        known_dims=[int(v) for v in list(raw_shape)],
                    )
                )
        input_specs.append((graph_input.name, np_dtype, tuple(shape)))
    return input_specs


def _generate_seeded_input(
    *,
    shape: Tuple[int, ...],
    np_dtype: np.dtype,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.issubdtype(np_dtype, np.bool_):
        return (rng.random(shape) > 0.5).astype(np_dtype)
    if np.issubdtype(np_dtype, np.floating):
        distribution = _resolve_float_seeded_distribution(shape=shape)
        if distribution == "uniform_0_1":
            return rng.uniform(0.0, 1.0, size=shape).astype(np_dtype)
        if distribution == "uniform_-1_1":
            return rng.uniform(-1.0, 1.0, size=shape).astype(np_dtype)
        return rng.standard_normal(shape).astype(np_dtype)
    if np.issubdtype(np_dtype, np.signedinteger):
        mode = str(
            os.environ.get("ONNX2TF_EVAL_INTEGER_RANDOM_MODE", "zeros")
        ).strip().lower()
        if mode in {"zeros", "zero"}:
            return np.zeros(shape, dtype=np_dtype)
        if mode in {"uniform_small", "small"}:
            return rng.integers(low=-8, high=9, size=shape, dtype=np_dtype)
        raise ValueError(
            "Invalid ONNX2TF_EVAL_INTEGER_RANDOM_MODE. "
            "Expected one of: zeros, uniform_small. "
            f"got: {mode}"
        )
    if np.issubdtype(np_dtype, np.unsignedinteger):
        mode = str(
            os.environ.get("ONNX2TF_EVAL_INTEGER_RANDOM_MODE", "zeros")
        ).strip().lower()
        if mode in {"zeros", "zero"}:
            return np.zeros(shape, dtype=np_dtype)
        if mode in {"uniform_small", "small"}:
            return rng.integers(low=0, high=17, size=shape, dtype=np_dtype)
        raise ValueError(
            "Invalid ONNX2TF_EVAL_INTEGER_RANDOM_MODE. "
            "Expected one of: zeros, uniform_small. "
            f"got: {mode}"
        )
    return rng.standard_normal(shape).astype(np.float32).astype(np_dtype)


def _is_likely_image_tensor_shape(shape: Tuple[int, ...]) -> bool:
    if len(shape) != 4:
        return False
    c_first = int(shape[1]) if int(shape[1]) > 0 else -1
    c_last = int(shape[3]) if int(shape[3]) > 0 else -1
    return (c_first in (1, 3, 4)) or (c_last in (1, 3, 4))


def _is_likely_nchw_image_tensor_shape(shape: Tuple[int, ...]) -> bool:
    if len(shape) != 4:
        return False
    c_first = int(shape[1]) if int(shape[1]) > 0 else -1
    c_last = int(shape[3]) if int(shape[3]) > 0 else -1
    return (c_first in (1, 3, 4)) and (c_last not in (1, 3, 4))


def _resolve_float_seeded_distribution(*, shape: Tuple[int, ...]) -> str:
    env_value = str(
        os.environ.get("ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION", "auto")
    ).strip().lower()
    if env_value in {"", "auto"}:
        # For channel-last image tensors, use non-negative inputs by default.
        # For channel-first image tensors, use normal distribution to avoid
        # persistent positive-bias evaluation artifacts on some models.
        if _is_likely_nchw_image_tensor_shape(shape):
            return "normal"
        if _is_likely_image_tensor_shape(shape):
            return "uniform_0_1"
        return "normal"
    if env_value in {"normal", "gaussian", "standard_normal"}:
        return "normal"
    if env_value in {"uniform_0_1", "uniform01", "0_1"}:
        return "uniform_0_1"
    if env_value in {"uniform_-1_1", "uniform_m1_p1", "uniform11", "-1_1"}:
        return "uniform_-1_1"
    raise ValueError(
        "Invalid ONNX2TF_EVAL_FLOAT_RANDOM_DISTRIBUTION. "
        "Expected one of: auto, normal, uniform_0_1, uniform_-1_1. "
        f"got: {env_value}"
    )


def _extract_sample_from_custom(
    *,
    data: np.ndarray,
    sample_index: int,
    expected_shape: Tuple[int, ...],
    np_dtype: np.dtype,
) -> np.ndarray:
    sample = np.asarray(data)
    if (
        sample.ndim == len(expected_shape)
        and len(expected_shape) > 0
        and expected_shape[0] == 1
        and sample.shape[0] >= 1
    ):
        start = int(sample_index % sample.shape[0])
        sample = sample[start : start + 1]
    if tuple(sample.shape) != tuple(expected_shape):
        expected_numel = int(np.prod(expected_shape, dtype=np.int64)) if len(expected_shape) > 0 else 1
        if sample.size == expected_numel:
            sample = sample.reshape(expected_shape)
        else:
            raise ValueError(
                "Custom evaluation input shape mismatch. "
                f"expected={expected_shape} actual={tuple(sample.shape)}"
            )
    return sample.astype(np_dtype, copy=False)


def _is_scatter_nd_index_oob_error(ex: BaseException) -> bool:
    ex_str = str(ex)
    return (
        "scatter_nd index out of bounds" in ex_str
        or "SCATTER_ND" in ex_str
    )


def _build_eval_inputs_for_sample(
    *,
    input_specs: Sequence[Tuple[str, np.dtype, Tuple[int, ...]]],
    custom_inputs: Dict[str, np.ndarray],
    sample_index: int,
    rng: np.random.Generator,
    force_zero_generated_inputs: bool = False,
) -> Dict[str, np.ndarray]:
    onnx_inputs: Dict[str, np.ndarray] = {}
    for input_name, input_dtype, input_shape in input_specs:
        custom_data = custom_inputs.get(input_name)
        if custom_data is None:
            custom_data = custom_inputs.get(_normalize_tensor_name(input_name))
        if custom_data is not None:
            sample = _extract_sample_from_custom(
                data=custom_data,
                sample_index=sample_index,
                expected_shape=input_shape,
                np_dtype=input_dtype,
            )
        elif bool(force_zero_generated_inputs):
            sample = np.zeros(input_shape, dtype=input_dtype)
        else:
            sample = _generate_seeded_input(
                shape=input_shape,
                np_dtype=input_dtype,
                rng=rng,
            )
        onnx_inputs[input_name] = sample
    return onnx_inputs


def _build_tflite_detail_map(
    *,
    onnx_names: Sequence[str],
    tflite_details: Sequence[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    detail_by_exact = {str(detail["name"]): detail for detail in tflite_details}
    onnx_base_names = [
        _normalize_tensor_name(onnx_name)
        for onnx_name in onnx_names
    ]
    unique_onnx_bases = [
        name for name in dict.fromkeys(onnx_base_names) if name != ""
    ]
    detail_by_base: Dict[str, Dict[str, Any]] = {}
    for detail in tflite_details:
        base_name = _normalize_tensor_name(str(detail["name"]))
        if base_name not in detail_by_base:
            detail_by_base[base_name] = detail
        # Generic SignatureDef name absorption:
        # e.g. "custom_signature_input_name" -> "input_name"
        # Match ONNX tensor-name suffixes and pick the longest unique match.
        suffix_matches: List[str] = [
            onnx_base
            for onnx_base in unique_onnx_bases
            if base_name.endswith(f"_{onnx_base}")
        ]
        if len(suffix_matches) == 0:
            continue
        max_match_len = max(len(match) for match in suffix_matches)
        longest_matches = [
            match for match in suffix_matches
            if len(match) == max_match_len
        ]
        if len(longest_matches) != 1:
            continue
        alias_name = longest_matches[0]
        if alias_name not in detail_by_base:
            detail_by_base[alias_name] = detail

    mapped: Dict[str, Dict[str, Any]] = {}
    for idx, onnx_name in enumerate(onnx_names):
        detail = detail_by_exact.get(onnx_name)
        if detail is None:
            detail = detail_by_base.get(_normalize_tensor_name(onnx_name))
        if detail is None and idx < len(tflite_details):
            detail = tflite_details[idx]
        if detail is None:
            raise ValueError(
                "Failed to map ONNX tensor name to TFLite tensor detail. "
                f"name={onnx_name}"
            )
        mapped[onnx_name] = detail
    return mapped


def _read_quantization_params(detail: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, int]:
    qparams = detail.get("quantization_parameters", {}) or {}
    scales = np.asarray(qparams.get("scales", []), dtype=np.float32).reshape(-1)
    zero_points = np.asarray(qparams.get("zero_points", []), dtype=np.int64).reshape(-1)
    quantized_dimension = int(qparams.get("quantized_dimension", 0))
    if scales.size == 0:
        legacy = detail.get("quantization", None)
        if isinstance(legacy, (tuple, list)) and len(legacy) == 2:
            legacy_scale = float(legacy[0])
            legacy_zero = int(legacy[1])
            if legacy_scale > 0.0:
                scales = np.asarray([legacy_scale], dtype=np.float32)
                zero_points = np.asarray([legacy_zero], dtype=np.int64)
    return scales, zero_points, quantized_dimension


def _quantize_for_tflite_input(data: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    target_dtype = np.dtype(detail["dtype"])
    value = np.asarray(data)
    if target_dtype == value.dtype:
        return value
    if np.issubdtype(target_dtype, np.floating):
        return value.astype(target_dtype)
    if np.issubdtype(target_dtype, np.bool_):
        return (value.astype(np.float32) > 0.0).astype(np.bool_)
    if np.issubdtype(target_dtype, np.integer):
        scales, zero_points, quantized_dimension = _read_quantization_params(detail)
        value_f32 = value.astype(np.float32)
        if scales.size <= 1:
            scale = float(scales[0]) if scales.size == 1 and scales[0] > 0.0 else 1.0
            zero = int(zero_points[0]) if zero_points.size > 0 else 0
            q = np.round(value_f32 / scale + zero)
        else:
            axis = quantized_dimension if quantized_dimension >= 0 else value_f32.ndim + quantized_dimension
            if axis < 0 or axis >= value_f32.ndim or value_f32.shape[axis] != scales.size:
                scale = float(scales[0]) if scales[0] > 0.0 else 1.0
                zero = int(zero_points[0]) if zero_points.size > 0 else 0
                q = np.round(value_f32 / scale + zero)
            else:
                shape = [1 for _ in range(value_f32.ndim)]
                shape[axis] = int(scales.size)
                scales_view = scales.reshape(shape)
                if zero_points.size == scales.size:
                    zero_points_view = zero_points.reshape(shape)
                else:
                    zero_points_view = np.zeros(shape, dtype=np.int64)
                safe_scales = np.where(scales_view == 0.0, 1.0, scales_view)
                q = np.round(value_f32 / safe_scales + zero_points_view)
        dtype_info = np.iinfo(target_dtype)
        return np.clip(q, dtype_info.min, dtype_info.max).astype(target_dtype)
    return value.astype(target_dtype)


def _dim_matches(expected: int, actual: int) -> bool:
    if int(expected) <= 0:
        return True
    return int(expected) == int(actual)


def _adapt_input_layout_for_tflite_input(data: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    value = np.asarray(data)
    target_shape = detail.get("shape_signature", None)
    if target_shape is None:
        target_shape = detail.get("shape", None)
    if target_shape is None:
        return value
    target_shape = [int(v) for v in np.asarray(target_shape).reshape(-1).tolist()]

    # ONNX scalar ([] rank) is commonly represented as [1] in TFLite runtime tensors.
    if value.ndim == 0 and len(target_shape) == 1 and _dim_matches(target_shape[0], 1):
        return value.reshape((1,))
    # Conversely, some runtime tensors can stay scalar while ONNX sample is [1].
    if value.ndim == 1 and int(value.size) == 1 and len(target_shape) == 0:
        return np.asarray(value).reshape(())

    if len(target_shape) != value.ndim:
        return value

    if all(_dim_matches(target_shape[idx], int(value.shape[idx])) for idx in range(value.ndim)):
        return value

    # ONNX test data is usually NCW/NCHW/NCDHW while flatbuffer_direct input
    # tensor layout may be converted to NWC/NHWC/NDHWC.
    transpose_candidates: List[Tuple[int, ...]] = []
    if value.ndim == 3:
        transpose_candidates = [(0, 2, 1)]
    elif value.ndim == 4:
        transpose_candidates = [(0, 2, 3, 1)]
    elif value.ndim == 5:
        transpose_candidates = [(0, 2, 3, 4, 1)]

    for perm in transpose_candidates:
        candidate_shape = [int(value.shape[int(axis)]) for axis in perm]
        if all(_dim_matches(target_shape[i], candidate_shape[i]) for i in range(value.ndim)):
            return np.transpose(value, perm)
    return value


def _shape_list_from_detail_value(value: Any) -> List[int]:
    if value is None:
        return []
    try:
        return [int(v) for v in np.asarray(value).reshape(-1).tolist()]
    except Exception:
        return []


def _is_target_shape_compatible_with_detail_signature(
    *,
    target_shape: Sequence[int],
    detail: Dict[str, Any],
) -> bool:
    target = [int(v) for v in list(target_shape)]
    signature = _shape_list_from_detail_value(detail.get("shape_signature", None))
    if len(signature) == 0:
        signature = _shape_list_from_detail_value(detail.get("shape", None))
    if len(signature) != len(target):
        return False
    for expected, actual in zip(signature, target):
        if int(expected) <= 0:
            continue
        if int(expected) != int(actual):
            return False
    return True


def _resize_tflite_inputs_if_needed(
    *,
    interpreter: Any,
    onnx_input_names: Sequence[str],
    tflite_input_map: Dict[str, Dict[str, Any]],
    adapted_inputs: Dict[str, np.ndarray],
) -> bool:
    resized = False
    for input_name in onnx_input_names:
        detail = tflite_input_map[str(input_name)]
        current_shape = _shape_list_from_detail_value(detail.get("shape", None))
        target_shape = [int(v) for v in list(np.asarray(adapted_inputs[str(input_name)]).shape)]
        if current_shape == target_shape:
            continue
        if not _is_target_shape_compatible_with_detail_signature(
            target_shape=target_shape,
            detail=detail,
        ):
            raise ValueError(
                "TFLite input shape is incompatible with shape_signature. "
                f"name={detail.get('name')} current_shape={current_shape} "
                f"target_shape={target_shape} "
                f"shape_signature={_shape_list_from_detail_value(detail.get('shape_signature', None))}"
            )
        try:
            interpreter.resize_tensor_input(
                int(detail["index"]),
                [int(v) for v in list(target_shape)],
                strict=False,
            )
        except TypeError:
            interpreter.resize_tensor_input(
                int(detail["index"]),
                [int(v) for v in list(target_shape)],
            )
        resized = True
    return bool(resized)


def _dequantize_tflite_output(data: np.ndarray, detail: Dict[str, Any]) -> np.ndarray:
    value = np.asarray(data)
    if np.issubdtype(value.dtype, np.floating):
        return value.astype(np.float32)
    if np.issubdtype(value.dtype, np.bool_):
        return value.astype(np.float32)
    if np.issubdtype(value.dtype, np.integer):
        scales, zero_points, quantized_dimension = _read_quantization_params(detail)
        value_f32 = value.astype(np.float32)
        if scales.size <= 1:
            if scales.size == 0:
                return value_f32
            scale = float(scales[0])
            zero = int(zero_points[0]) if zero_points.size > 0 else 0
            return (value_f32 - zero) * scale
        axis = quantized_dimension if quantized_dimension >= 0 else value_f32.ndim + quantized_dimension
        if axis < 0 or axis >= value_f32.ndim or value_f32.shape[axis] != scales.size:
            scale = float(scales[0])
            zero = int(zero_points[0]) if zero_points.size > 0 else 0
            return (value_f32 - zero) * scale
        shape = [1 for _ in range(value_f32.ndim)]
        shape[axis] = int(scales.size)
        scales_view = scales.reshape(shape).astype(np.float32)
        if zero_points.size == scales.size:
            zero_points_view = zero_points.reshape(shape).astype(np.float32)
        else:
            zero_points_view = np.zeros(shape, dtype=np.float32)
        return (value_f32 - zero_points_view) * scales_view
    return value.astype(np.float32)


def _max_abs_error(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(
            "max-abs error requires equal shapes. "
            f"a={tuple(a.shape)} b={tuple(b.shape)}"
        )
    if a.size == 0:
        return 0.0
    a64, b64 = _sanitize_metric_vectors(a, b)
    return float(np.max(np.abs(a64 - b64)))


def _sanitize_metric_vectors(
    ref: np.ndarray,
    pred: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return metric-safe vectors:
    - Matching NaN / +Inf / -Inf pairs are treated as equal (set to 0).
    - Remaining non-finite values are mapped to large finite sentinels so
      aggregate metrics stay finite and deterministic.
    """
    ref_flat = np.asarray(ref, dtype=np.float64).reshape(-1)
    pred_flat = np.asarray(pred, dtype=np.float64).reshape(-1)
    if ref_flat.shape != pred_flat.shape:
        raise ValueError(
            "Evaluation tensor shape mismatch. "
            f"ref={tuple(ref.shape)} pred={tuple(pred.shape)}"
        )
    if ref_flat.size == 0:
        return ref_flat, pred_flat

    both_nan = np.isnan(ref_flat) & np.isnan(pred_flat)
    both_pos_inf = np.isposinf(ref_flat) & np.isposinf(pred_flat)
    both_neg_inf = np.isneginf(ref_flat) & np.isneginf(pred_flat)
    equal_nonfinite = both_nan | both_pos_inf | both_neg_inf

    needs_copy = bool(np.any(equal_nonfinite))
    has_nonfinite = bool(np.any(~np.isfinite(ref_flat) | ~np.isfinite(pred_flat)))
    if not needs_copy and not has_nonfinite:
        return ref_flat, pred_flat

    ref_clean = np.array(ref_flat, dtype=np.float64, copy=True)
    pred_clean = np.array(pred_flat, dtype=np.float64, copy=True)

    if needs_copy:
        ref_clean[equal_nonfinite] = 0.0
        pred_clean[equal_nonfinite] = 0.0

    remaining_nonfinite = ~np.isfinite(ref_clean) | ~np.isfinite(pred_clean)
    if np.any(remaining_nonfinite):
        large = 1.0e30
        ref_clean = np.nan_to_num(ref_clean, nan=0.0, posinf=large, neginf=-large)
        pred_clean = np.nan_to_num(pred_clean, nan=0.0, posinf=large, neginf=-large)

    return ref_clean, pred_clean


def _align_output_layout_for_compare(
    *,
    onnx_output: np.ndarray,
    tflite_output: np.ndarray,
    rtol: float,
    atol: float,
) -> Tuple[np.ndarray, str, Optional[List[int]]]:
    """
    Align TFLite output layout to ONNX output layout for elementwise comparison.

    Strategy:
    1. If shapes already match, use as-is.
    2. If ranks match, brute-force all axis permutations whose output shape matches ONNX.
       - Prefer the first permutation that satisfies allclose.
       - Otherwise choose the permutation with smallest max-abs error.
    3. If no permutation fits but total elements match, fallback to reshape.
    4. Otherwise raise shape mismatch.
    """
    onnx_arr = np.asarray(onnx_output)
    tflite_arr = np.asarray(tflite_output)

    if onnx_arr.shape == tflite_arr.shape:
        return tflite_arr, "identity", None

    best_candidate: Optional[np.ndarray] = None
    best_perm: Optional[List[int]] = None
    best_err: float = float("inf")
    if onnx_arr.ndim == tflite_arr.ndim and onnx_arr.ndim > 0:
        rank = int(tflite_arr.ndim)
        for perm_tuple in itertools.permutations(range(rank)):
            if tuple(np.asarray(tflite_arr.transpose(perm_tuple)).shape) != tuple(onnx_arr.shape):
                continue
            candidate = np.asarray(tflite_arr.transpose(perm_tuple))
            if np.allclose(
                onnx_arr,
                candidate,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                return candidate, "transpose", [int(v) for v in perm_tuple]
            err = _max_abs_error(onnx_arr, candidate)
            if err < best_err:
                best_err = float(err)
                best_candidate = candidate
                best_perm = [int(v) for v in perm_tuple]
        if best_candidate is not None:
            return best_candidate, "transpose", best_perm

    if onnx_arr.size == tflite_arr.size:
        return np.asarray(tflite_arr).reshape(onnx_arr.shape), "reshape", None

    raise ValueError(
        "Evaluation output shape mismatch after layout alignment. "
        f"onnx_shape={tuple(onnx_arr.shape)} tflite_shape={tuple(tflite_arr.shape)}"
    )


class _MetricAccumulator:
    def __init__(self) -> None:
        self.numel = 0
        self.max_abs = 0.0
        self.max_ref_abs = 0.0
        self.sum_abs = 0.0
        self.sum_sq = 0.0
        self.sum_dot = 0.0
        self.sum_ref_norm = 0.0
        self.sum_pred_norm = 0.0

    def update(self, ref: np.ndarray, pred: np.ndarray) -> None:
        ref_flat, pred_flat = _sanitize_metric_vectors(ref, pred)
        if ref_flat.size == 0:
            return
        diff = ref_flat - pred_flat
        abs_diff = np.abs(diff)
        self.max_abs = max(self.max_abs, float(np.max(abs_diff)))
        self.max_ref_abs = max(self.max_ref_abs, float(np.max(np.abs(ref_flat))))
        self.sum_abs += float(np.sum(abs_diff))
        self.sum_sq += float(np.sum(diff * diff))
        self.sum_dot += float(np.dot(ref_flat, pred_flat))
        self.sum_ref_norm += float(np.dot(ref_flat, ref_flat))
        self.sum_pred_norm += float(np.dot(pred_flat, pred_flat))
        self.numel += int(ref_flat.size)

    def to_dict(self) -> Dict[str, float]:
        if self.numel == 0:
            return {
                "max_abs": 0.0,
                "ref_max_abs": 0.0,
                "ref_rms": 0.0,
                "mean_abs": 0.0,
                "rmse": 0.0,
                "cosine_similarity": 1.0,
            }
        mean_abs = self.sum_abs / float(self.numel)
        rmse = float(np.sqrt(self.sum_sq / float(self.numel)))
        ref_rms = float(np.sqrt(self.sum_ref_norm / float(self.numel)))
        if self.sum_ref_norm == 0.0 and self.sum_pred_norm == 0.0:
            cosine = 1.0
        elif self.sum_ref_norm == 0.0 or self.sum_pred_norm == 0.0:
            cosine = 0.0
        else:
            cosine = float(
                self.sum_dot / np.sqrt(self.sum_ref_norm * self.sum_pred_norm)
            )
            cosine = float(np.clip(cosine, -1.0, 1.0))
        return {
            "max_abs": float(self.max_abs),
            "ref_max_abs": float(self.max_ref_abs),
            "ref_rms": float(ref_rms),
            "mean_abs": float(mean_abs),
            "rmse": float(rmse),
            "cosine_similarity": float(cosine),
        }


_FLOAT_METRIC_THRESHOLDS = {
    "max_abs": 1.0e-4,
    "mean_abs": 1.0e-5,
    "rmse": 1.0e-5,
    "cosine_similarity": 0.9999,
}


_QUANT_METRIC_THRESHOLDS = {
    "max_abs": 5.0e-2,
    "mean_abs": 1.0e-2,
    "rmse": 2.0e-2,
    "cosine_similarity": 0.98,
}


def _is_integer_or_bool_dtype(np_dtype: np.dtype) -> bool:
    return np.issubdtype(np_dtype, np.integer) or np.issubdtype(np_dtype, np.bool_)


def _resolve_compare_mode(compare_mode: str, *, has_quantized_outputs: bool) -> str:
    compare_mode = str(compare_mode).strip().lower()
    if compare_mode not in ["auto", "dequant", "raw"]:
        raise ValueError(
            "compare_mode must be one of ['auto', 'dequant', 'raw']. "
            f"got: {compare_mode}"
        )
    if compare_mode == "auto":
        return "dequant" if has_quantized_outputs else "raw"
    return compare_mode


def _resolve_metric_thresholds(
    *,
    metric_thresholds: Optional[Dict[str, float]],
    use_quant_defaults: bool,
) -> Dict[str, float]:
    base = dict(_QUANT_METRIC_THRESHOLDS if use_quant_defaults else _FLOAT_METRIC_THRESHOLDS)
    if metric_thresholds is None:
        return base
    for key in base.keys():
        if key in metric_thresholds and metric_thresholds[key] is not None:
            base[key] = float(metric_thresholds[key])
    return base


def _judge_metrics(
    *,
    metrics: Dict[str, float],
    thresholds: Dict[str, float],
    rtol: float = 0.0,
) -> Dict[str, Any]:
    ref_max_abs = max(float(metrics.get("ref_max_abs", 0.0)), 0.0)
    ref_rms = max(float(metrics.get("ref_rms", 0.0)), 0.0)
    max_abs_limit = max(float(thresholds["max_abs"]), float(rtol) * ref_max_abs)
    rmse_limit = max(float(thresholds["rmse"]), float(rtol) * ref_rms)
    checks = {
        "max_abs": float(metrics["max_abs"]) <= max_abs_limit,
        "mean_abs": float(metrics["mean_abs"]) <= float(thresholds["mean_abs"]),
        "rmse": float(metrics["rmse"]) <= rmse_limit,
        "cosine_similarity": float(metrics["cosine_similarity"]) >= float(thresholds["cosine_similarity"]),
    }
    return {
        "pass": bool(all(checks.values())),
        "checks": checks,
        "effective_thresholds": {
            "max_abs": float(max_abs_limit),
            "mean_abs": float(thresholds["mean_abs"]),
            "rmse": float(rmse_limit),
            "cosine_similarity": float(thresholds["cosine_similarity"]),
        },
    }


def evaluate_onnx_tflite_outputs(
    *,
    onnx_graph: onnx.ModelProto,
    tflite_path: str,
    output_report_path: str,
    num_samples: int = 10,
    seed: int = 0,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
    compare_mode: str = "auto",
    fail_on_threshold: bool = False,
    metric_thresholds: Optional[Dict[str, float]] = None,
    onnxruntime_output_memmap: bool = True,
    onnxruntime_output_memmap_dir: Optional[str] = None,
) -> Dict[str, Any]:
    import onnxruntime as ort

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0. got: {num_samples}")
    rtol = float(rtol)
    atol = float(atol)
    if rtol < 0.0:
        raise ValueError(f"rtol must be >= 0.0. got: {rtol}")
    if atol < 0.0:
        raise ValueError(f"atol must be >= 0.0. got: {atol}")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite file for evaluation does not exist. path={tflite_path}")

    rng = np.random.default_rng(seed=int(seed))
    custom_inputs = _load_custom_input_data(custom_input_op_name_np_data_path)
    input_specs = _collect_onnx_input_specs(onnx_graph)
    onnx_input_names = [name for name, _, _ in input_specs]
    onnx_output_names = [output.name for output in onnx_graph.graph.output]

    try:
        onnx_session = ort.InferenceSession(
            onnx_graph.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
    except Exception as ex:
        err = str(ex)
        if "Unsupported model IR version" not in err:
            raise
        fallback_graph = onnx.ModelProto()
        fallback_graph.CopyFrom(onnx_graph)
        fallback_graph.ir_version = min(int(fallback_graph.ir_version), 10)
        onnx_session = ort.InferenceSession(
            fallback_graph.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
    interpreter = _create_tflite_interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    tflite_input_details = interpreter.get_input_details()
    tflite_output_details = interpreter.get_output_details()
    tflite_input_map = _build_tflite_detail_map(
        onnx_names=onnx_input_names,
        tflite_details=tflite_input_details,
    )
    tflite_output_map = _build_tflite_detail_map(
        onnx_names=onnx_output_names,
        tflite_details=tflite_output_details,
    )
    has_quantized_outputs = any(
        _is_integer_or_bool_dtype(np.dtype(detail["dtype"]))
        for detail in tflite_output_details
    )
    resolved_compare_mode = _resolve_compare_mode(
        compare_mode,
        has_quantized_outputs=has_quantized_outputs,
    )
    resolved_thresholds = _resolve_metric_thresholds(
        metric_thresholds=metric_thresholds,
        use_quant_defaults=has_quantized_outputs,
    )

    total_metrics = _MetricAccumulator()
    per_output_metrics: Dict[str, _MetricAccumulator] = {
        output_name: _MetricAccumulator() for output_name in onnx_output_names
    }
    tflite_output_name_map: Dict[str, str] = {
        output_name: str(tflite_output_map[output_name]["name"])
        for output_name in onnx_output_names
    }
    allclose_total = 0
    allclose_matched = 0
    per_output_allclose: Dict[str, Dict[str, int]] = {
        output_name: {"matched": 0, "total": 0}
        for output_name in onnx_output_names
    }
    layout_alignment_summary = {
        "identity": 0,
        "transpose": 0,
        "reshape": 0,
    }
    per_output_layout_alignment: Dict[str, Dict[str, Any]] = {
        output_name: {
            "identity": 0,
            "transpose": 0,
            "reshape": 0,
            "permutation_counts": {},
        }
        for output_name in onnx_output_names
    }

    for sample_index in range(int(num_samples)):
        force_zero_generated_inputs = False
        while True:
            onnx_inputs = _build_eval_inputs_for_sample(
                input_specs=input_specs,
                custom_inputs=custom_inputs,
                sample_index=int(sample_index),
                rng=rng,
                force_zero_generated_inputs=bool(force_zero_generated_inputs),
            )

            onnx_outputs = onnx_session.run(onnx_output_names, onnx_inputs)
            onnx_outputs_by_name = {
                output_name: output_value
                for output_name, output_value in zip(onnx_output_names, onnx_outputs)
            }

            adapted_inputs: Dict[str, np.ndarray] = {
                input_name: _adapt_input_layout_for_tflite_input(
                    onnx_inputs[input_name],
                    tflite_input_map[input_name],
                )
                for input_name in onnx_input_names
            }
            if _resize_tflite_inputs_if_needed(
                interpreter=interpreter,
                onnx_input_names=onnx_input_names,
                tflite_input_map=tflite_input_map,
                adapted_inputs=adapted_inputs,
            ):
                interpreter.allocate_tensors()
                tflite_input_details = interpreter.get_input_details()
                tflite_output_details = interpreter.get_output_details()
                tflite_input_map = _build_tflite_detail_map(
                    onnx_names=onnx_input_names,
                    tflite_details=tflite_input_details,
                )
                tflite_output_map = _build_tflite_detail_map(
                    onnx_names=onnx_output_names,
                    tflite_details=tflite_output_details,
                )
                adapted_inputs = {
                    input_name: _adapt_input_layout_for_tflite_input(
                        onnx_inputs[input_name],
                        tflite_input_map[input_name],
                    )
                    for input_name in onnx_input_names
                }

            for input_name in onnx_input_names:
                detail = tflite_input_map[input_name]
                interpreter.set_tensor(
                    detail["index"],
                    _quantize_for_tflite_input(adapted_inputs[input_name], detail),
                )

            try:
                interpreter.invoke()
            except Exception as ex:
                if (
                    custom_input_op_name_np_data_path is None
                    and not bool(force_zero_generated_inputs)
                    and _is_scatter_nd_index_oob_error(ex)
                ):
                    force_zero_generated_inputs = True
                    continue
                raise

            for output_name in onnx_output_names:
                detail = tflite_output_map[output_name]
                tflite_output = interpreter.get_tensor(detail["index"])
                if resolved_compare_mode == "dequant":
                    tflite_output = _dequantize_tflite_output(tflite_output, detail)
                else:
                    tflite_output = np.asarray(tflite_output)

                onnx_output = np.asarray(onnx_outputs_by_name[output_name])
                try:
                    tflite_output, align_mode, align_perm = _align_output_layout_for_compare(
                        onnx_output=onnx_output,
                        tflite_output=tflite_output,
                        rtol=rtol,
                        atol=atol,
                    )
                except ValueError as ex:
                    raise ValueError(
                        "Evaluation output shape mismatch. "
                        f"name={output_name} onnx_shape={tuple(onnx_output.shape)} "
                        f"tflite_shape={tuple(np.asarray(tflite_output).shape)} "
                        f"reason={ex}"
                    ) from ex
                layout_alignment_summary[align_mode] = int(layout_alignment_summary[align_mode]) + 1
                per_output_layout_alignment[output_name][align_mode] = (
                    int(per_output_layout_alignment[output_name][align_mode]) + 1
                )
                if align_perm is not None:
                    perm_key = ",".join(str(int(v)) for v in align_perm)
                    per_output_layout_alignment[output_name]["permutation_counts"][perm_key] = (
                        int(
                            per_output_layout_alignment[output_name]["permutation_counts"].get(
                                perm_key, 0
                            )
                        )
                        + 1
                    )

                total_metrics.update(onnx_output, tflite_output)
                per_output_metrics[output_name].update(onnx_output, tflite_output)
                is_allclose = bool(
                    np.allclose(
                        onnx_output,
                        tflite_output,
                        rtol=rtol,
                        atol=atol,
                        equal_nan=True,
                    )
                )
                allclose_total += 1
                allclose_matched += int(is_allclose)
                per_output_allclose[output_name]["total"] += 1
                per_output_allclose[output_name]["matched"] += int(is_allclose)
            break

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    overall_metrics = total_metrics.to_dict()
    metric_judgement = _judge_metrics(
        metrics=overall_metrics,
        thresholds=resolved_thresholds,
        rtol=rtol,
    )
    allclose_pass = bool(allclose_total == allclose_matched)
    evaluation_pass = bool(metric_judgement["pass"] and allclose_pass)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "rtol": float(rtol),
        "atol": float(atol),
        "compare_mode": resolved_compare_mode,
        "has_quantized_outputs": bool(has_quantized_outputs),
        "inputs_source": (
            "custom_input_op_name_np_data_path"
            if len(custom_inputs) > 0
            else "seeded_random"
        ),
        "onnx_input_names": onnx_input_names,
        "onnx_output_names": onnx_output_names,
        "tflite_path": tflite_path,
        "metric_thresholds": {
            k: float(v) for k, v in resolved_thresholds.items()
        },
        "overall_metrics": overall_metrics,
        "metric_threshold_judgement": metric_judgement,
        "allclose_summary": {
            "matched": int(allclose_matched),
            "total": int(allclose_total),
            "pass": allclose_pass,
        },
        "layout_alignment_summary": {
            "identity": int(layout_alignment_summary["identity"]),
            "transpose": int(layout_alignment_summary["transpose"]),
            "reshape": int(layout_alignment_summary["reshape"]),
        },
        "evaluation_pass": evaluation_pass,
        "per_output_metrics": {
            output_name: {
                "tflite_output_name": tflite_output_name_map[output_name],
                "allclose": {
                    "matched": int(per_output_allclose[output_name]["matched"]),
                    "total": int(per_output_allclose[output_name]["total"]),
                    "pass": (
                        per_output_allclose[output_name]["matched"]
                        == per_output_allclose[output_name]["total"]
                    ),
                },
                "layout_alignment": {
                    "identity": int(per_output_layout_alignment[output_name]["identity"]),
                    "transpose": int(per_output_layout_alignment[output_name]["transpose"]),
                    "reshape": int(per_output_layout_alignment[output_name]["reshape"]),
                    "permutation_counts": {
                        k: int(v)
                        for k, v in per_output_layout_alignment[output_name][
                            "permutation_counts"
                        ].items()
                    },
                },
                **per_output_metrics[output_name].to_dict(),
            }
            for output_name in onnx_output_names
        },
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if bool(fail_on_threshold) and not evaluation_pass:
        raise RuntimeError(
            "ONNX/TFLite evaluation failed thresholds. "
            f"report={output_report_path} "
            f"metrics={overall_metrics} "
            f"allclose={report['allclose_summary']}"
        )
    return report


def _onnx_inference_worker(
    payload: Dict[str, Any],
    result_queue: Any,
) -> None:
    try:
        import onnxruntime as ort

        onnx_graph = onnx.load_from_string(payload["onnx_graph_serialized"])
        onnx_output_names = [str(v) for v in payload["onnx_output_names"]]
        onnx_inputs = _load_onnx_inputs_from_payload(payload)
        try:
            onnx_session = ort.InferenceSession(
                onnx_graph.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
        except Exception as ex:
            err = str(ex)
            if "Unsupported model IR version" not in err:
                raise
            fallback_graph = onnx.ModelProto()
            fallback_graph.CopyFrom(onnx_graph)
            fallback_graph.ir_version = min(int(fallback_graph.ir_version), 10)
            onnx_session = ort.InferenceSession(
                fallback_graph.SerializeToString(),
                providers=["CPUExecutionProvider"],
            )
        onnx_outputs = onnx_session.run(onnx_output_names, onnx_inputs)
        use_memmap_outputs = bool(payload.get("use_memmap_outputs", False))
        if use_memmap_outputs:
            memmap_dir = str(payload.get("memmap_dir", ""))
            file_prefix = str(payload.get("memmap_file_prefix", "onnx_out"))
            named_outputs = {
                name: np.asarray(value)
                for name, value in zip(onnx_output_names, onnx_outputs)
            }
            manifest = _write_named_arrays_to_memmap_files(
                arrays=named_outputs,
                memmap_dir=memmap_dir,
                file_prefix=file_prefix,
            )
            result_queue.put(
                {
                    "ok": True,
                    "onnx_outputs_memmap_manifest": manifest,
                }
            )
            return
        result_queue.put(
            {
                "ok": True,
                "onnx_outputs": {
                    name: np.asarray(value)
                    for name, value in zip(onnx_output_names, onnx_outputs)
                },
            }
        )
    except BaseException as ex:
        result_queue.put(
            {
                "ok": False,
                "error": str(ex),
                "traceback": traceback.format_exc(),
            }
        )


def _tflite_inference_worker(
    payload: Dict[str, Any],
    result_queue: Any,
) -> None:
    try:
        onnx_input_names = [str(v) for v in payload["onnx_input_names"]]
        onnx_output_names = [str(v) for v in payload["onnx_output_names"]]
        onnx_inputs = _load_onnx_inputs_from_payload(payload)
        compare_mode = str(payload["compare_mode"])

        interpreter = _create_tflite_interpreter(
            model_path=str(payload["tflite_path"]),
        )
        interpreter.allocate_tensors()
        tflite_input_details = interpreter.get_input_details()
        tflite_output_details = interpreter.get_output_details()
        tflite_input_map = _build_tflite_detail_map(
            onnx_names=onnx_input_names,
            tflite_details=tflite_input_details,
        )
        tflite_output_map = _build_tflite_detail_map(
            onnx_names=onnx_output_names,
            tflite_details=tflite_output_details,
        )
        adapted_inputs: Dict[str, np.ndarray] = {
            input_name: _adapt_input_layout_for_tflite_input(
                onnx_inputs[input_name],
                tflite_input_map[input_name],
            )
            for input_name in onnx_input_names
        }
        if _resize_tflite_inputs_if_needed(
            interpreter=interpreter,
            onnx_input_names=onnx_input_names,
            tflite_input_map=tflite_input_map,
            adapted_inputs=adapted_inputs,
        ):
            interpreter.allocate_tensors()
            tflite_input_details = interpreter.get_input_details()
            tflite_output_details = interpreter.get_output_details()
            tflite_input_map = _build_tflite_detail_map(
                onnx_names=onnx_input_names,
                tflite_details=tflite_input_details,
            )
            tflite_output_map = _build_tflite_detail_map(
                onnx_names=onnx_output_names,
                tflite_details=tflite_output_details,
            )
            adapted_inputs = {
                input_name: _adapt_input_layout_for_tflite_input(
                    onnx_inputs[input_name],
                    tflite_input_map[input_name],
                )
                for input_name in onnx_input_names
            }

        for input_name in onnx_input_names:
            detail = tflite_input_map[input_name]
            interpreter.set_tensor(
                detail["index"],
                _quantize_for_tflite_input(adapted_inputs[input_name], detail),
            )
        interpreter.invoke()

        outputs: Dict[str, np.ndarray] = {}
        output_name_map: Dict[str, str] = {}
        for output_name in onnx_output_names:
            detail = tflite_output_map[output_name]
            out = interpreter.get_tensor(detail["index"])
            if compare_mode == "dequant":
                out = _dequantize_tflite_output(out, detail)
            else:
                out = np.asarray(out)
            outputs[output_name] = np.asarray(out)
            output_name_map[output_name] = str(detail["name"])

        has_quantized_outputs = any(
            _is_integer_or_bool_dtype(np.dtype(detail["dtype"]))
            for detail in tflite_output_details
        )
        use_memmap_outputs = bool(payload.get("use_memmap_outputs", False))
        if use_memmap_outputs:
            memmap_dir = str(payload.get("memmap_dir", ""))
            file_prefix = str(payload.get("memmap_file_prefix", "tflite_out"))
            manifest = _write_named_arrays_to_memmap_files(
                arrays=outputs,
                memmap_dir=memmap_dir,
                file_prefix=file_prefix,
            )
            result_queue.put(
                {
                    "ok": True,
                    "tflite_outputs_memmap_manifest": manifest,
                    "tflite_output_name_map": output_name_map,
                    "has_quantized_outputs": bool(has_quantized_outputs),
                }
            )
            return
        result_queue.put(
            {
                "ok": True,
                "tflite_outputs": outputs,
                "tflite_output_name_map": output_name_map,
                "has_quantized_outputs": bool(has_quantized_outputs),
            }
        )
    except BaseException as ex:
        result_queue.put(
            {
                "ok": False,
                "error": str(ex),
                "traceback": traceback.format_exc(),
            }
        )


def _run_worker_in_subprocess(
    *,
    worker: Any,
    payload: Dict[str, Any],
    timeout_sec: int,
) -> Dict[str, Any]:
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    process = ctx.Process(
        target=worker,
        args=(payload, result_queue),
        daemon=True,
    )
    process.start()
    result = None
    timed_out = False
    start_time = time.monotonic()
    timeout_sec_float = float(timeout_sec)
    try:
        # Read worker result first. join-before-get can block indefinitely when
        # worker exits while queue feeder is back-pressured by large payloads.
        #
        # Also poll worker liveness in short intervals so abnormal worker exits
        # (e.g. SIGABRT/SIGSEGV) are surfaced immediately instead of waiting for
        # the full queue timeout.
        poll_interval_sec = 1.0
        while True:
            elapsed = max(0.0, float(time.monotonic() - start_time))
            remaining = max(0.0, timeout_sec_float - elapsed)
            if remaining <= 0.0:
                timed_out = True
                break
            try:
                result = result_queue.get(timeout=min(poll_interval_sec, remaining))
                break
            except queue_module.Empty:
                if not process.is_alive():
                    break
    finally:
        # Ensure subprocess is fully reaped.
        elapsed = max(0.0, float(time.monotonic() - start_time))
        remaining = max(0.0, timeout_sec_float - elapsed)
        process.join(timeout=remaining if remaining > 0.0 else 0.0)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)
        result_queue.close()
        result_queue.join_thread()

    if timed_out:
        raise RuntimeError(
            f"Worker timed out. worker={getattr(worker, '__name__', str(worker))} "
            f"timeout_sec={int(timeout_sec)}"
        )

    if isinstance(result, dict) and not result.get("ok", False):
        raise RuntimeError(
            f'Worker failed. worker={getattr(worker, "__name__", str(worker))} '
            f'error={result.get("error", "")}\n{result.get("traceback", "")}'
        )

    exit_code = int(process.exitcode) if process.exitcode is not None else -9999
    if exit_code != 0:
        raise RuntimeError(
            f"Worker exited abnormally. worker={getattr(worker, '__name__', str(worker))} "
            f"exit_code={exit_code}"
        )

    if not isinstance(result, dict) or not result.get("ok", False):
        raise RuntimeError(
            f"Worker returned no result. worker={getattr(worker, '__name__', str(worker))}"
        )
    return result


def evaluate_onnx_tflite_outputs_isolated(
    *,
    onnx_graph: onnx.ModelProto,
    tflite_path: str,
    output_report_path: str,
    num_samples: int = 10,
    seed: int = 0,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
    compare_mode: str = "auto",
    fail_on_threshold: bool = False,
    metric_thresholds: Optional[Dict[str, float]] = None,
    timeout_sec: int = 600,
    onnxruntime_output_memmap: bool = True,
    onnxruntime_output_memmap_dir: Optional[str] = None,
) -> Dict[str, Any]:
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0. got: {num_samples}")
    rtol = float(rtol)
    atol = float(atol)
    if rtol < 0.0:
        raise ValueError(f"rtol must be >= 0.0. got: {rtol}")
    if atol < 0.0:
        raise ValueError(f"atol must be >= 0.0. got: {atol}")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite file for evaluation does not exist. path={tflite_path}")

    rng = np.random.default_rng(seed=int(seed))
    custom_inputs = _load_custom_input_data(custom_input_op_name_np_data_path)
    input_specs = _collect_onnx_input_specs(onnx_graph)
    onnx_input_names = [name for name, _, _ in input_specs]
    onnx_output_names = [output.name for output in onnx_graph.graph.output]
    onnx_graph_serialized = onnx_graph.SerializeToString()
    use_worker_output_memmap, worker_output_memmap_dir = _resolve_eval_memmap_config(
        onnx_graph=onnx_graph,
        onnx_output_names=onnx_output_names,
        onnxruntime_output_memmap=bool(onnxruntime_output_memmap),
        onnxruntime_output_memmap_dir=onnxruntime_output_memmap_dir,
    )
    worker_input_memmap_dir = _resolve_eval_input_memmap_dir(
        onnxruntime_output_memmap=bool(onnxruntime_output_memmap),
        onnxruntime_output_memmap_dir=onnxruntime_output_memmap_dir,
        fallback_dir=worker_output_memmap_dir,
    )
    use_worker_input_memmap = bool(worker_input_memmap_dir is not None)

    runtime_compare_mode = str(compare_mode).lower()
    if runtime_compare_mode == "auto":
        # Use dequant mode for subprocess inference; float outputs are unchanged by dequant.
        runtime_compare_mode = "dequant"

    total_metrics = _MetricAccumulator()
    per_output_metrics: Dict[str, _MetricAccumulator] = {
        output_name: _MetricAccumulator() for output_name in onnx_output_names
    }
    per_output_allclose: Dict[str, Dict[str, int]] = {
        output_name: {"matched": 0, "total": 0}
        for output_name in onnx_output_names
    }
    layout_alignment_summary = {
        "identity": 0,
        "transpose": 0,
        "reshape": 0,
    }
    per_output_layout_alignment: Dict[str, Dict[str, Any]] = {
        output_name: {
            "identity": 0,
            "transpose": 0,
            "reshape": 0,
            "permutation_counts": {},
        }
        for output_name in onnx_output_names
    }

    has_quantized_outputs = False
    resolved_compare_mode = str(compare_mode).lower()
    resolved_thresholds: Dict[str, float] = {}
    tflite_output_name_map: Dict[str, str] = {}

    allclose_total = 0
    allclose_matched = 0
    per_worker_timeout = max(60, int(timeout_sec) // max(int(num_samples), 1))

    for sample_index in range(int(num_samples)):
        force_zero_generated_inputs = False
        while True:
            onnx_inputs = _build_eval_inputs_for_sample(
                input_specs=input_specs,
                custom_inputs=custom_inputs,
                sample_index=int(sample_index),
                rng=rng,
                force_zero_generated_inputs=bool(force_zero_generated_inputs),
            )

            onnx_input_manifest: Optional[Dict[str, Dict[str, Any]]] = None
            onnx_manifest: Optional[Dict[str, Dict[str, Any]]] = None
            tflite_manifest: Optional[Dict[str, Dict[str, Any]]] = None
            try:
                worker_input_payload: Dict[str, Any]
                if bool(use_worker_input_memmap):
                    if worker_input_memmap_dir is None:
                        raise RuntimeError("Evaluation input memmap directory is unavailable.")
                    onnx_input_manifest = _write_named_arrays_to_memmap_files(
                        arrays=onnx_inputs,
                        memmap_dir=worker_input_memmap_dir,
                        file_prefix=f"input_sample{sample_index}",
                    )
                    worker_input_payload = {
                        "onnx_inputs_memmap_manifest": onnx_input_manifest,
                    }
                else:
                    worker_input_payload = {
                        "onnx_inputs": onnx_inputs,
                    }

                onnx_result = _run_worker_in_subprocess(
                    worker=_onnx_inference_worker,
                    payload={
                        "onnx_graph_serialized": onnx_graph_serialized,
                        "onnx_output_names": onnx_output_names,
                        **worker_input_payload,
                        "use_memmap_outputs": bool(use_worker_output_memmap),
                        "memmap_dir": worker_output_memmap_dir,
                        "memmap_file_prefix": f"onnx_sample{sample_index}",
                    },
                    timeout_sec=per_worker_timeout,
                )
                tflite_result = _run_worker_in_subprocess(
                    worker=_tflite_inference_worker,
                    payload={
                        "tflite_path": str(tflite_path),
                        "onnx_input_names": onnx_input_names,
                        "onnx_output_names": onnx_output_names,
                        **worker_input_payload,
                        "compare_mode": runtime_compare_mode,
                        "use_memmap_outputs": bool(use_worker_output_memmap),
                        "memmap_dir": worker_output_memmap_dir,
                        "memmap_file_prefix": f"tflite_sample{sample_index}",
                    },
                    timeout_sec=per_worker_timeout,
                )

                if bool(use_worker_output_memmap):
                    onnx_manifest = {
                        str(k): v
                        for k, v in onnx_result["onnx_outputs_memmap_manifest"].items()
                    }
                    tflite_manifest = {
                        str(k): v
                        for k, v in tflite_result["tflite_outputs_memmap_manifest"].items()
                    }
                    onnx_outputs_by_name = _load_named_arrays_from_memmap_manifest(
                        manifest=onnx_manifest,
                    )
                    tflite_outputs_by_name = _load_named_arrays_from_memmap_manifest(
                        manifest=tflite_manifest,
                    )
                else:
                    onnx_outputs_by_name = {
                        str(k): np.asarray(v)
                        for k, v in onnx_result["onnx_outputs"].items()
                    }
                    tflite_outputs_by_name = {
                        str(k): np.asarray(v)
                        for k, v in tflite_result["tflite_outputs"].items()
                    }

                if sample_index == 0:
                    has_quantized_outputs = bool(tflite_result.get("has_quantized_outputs", False))
                    resolved_compare_mode = _resolve_compare_mode(
                        compare_mode,
                        has_quantized_outputs=has_quantized_outputs,
                    )
                    resolved_thresholds = _resolve_metric_thresholds(
                        metric_thresholds=metric_thresholds,
                        use_quant_defaults=has_quantized_outputs,
                    )
                    tflite_output_name_map = {
                        str(k): str(v)
                        for k, v in tflite_result.get("tflite_output_name_map", {}).items()
                    }

                for output_name in onnx_output_names:
                    onnx_output = np.asarray(onnx_outputs_by_name[output_name])
                    tflite_output = np.asarray(tflite_outputs_by_name[output_name])
                    try:
                        tflite_output, align_mode, align_perm = _align_output_layout_for_compare(
                            onnx_output=onnx_output,
                            tflite_output=tflite_output,
                            rtol=rtol,
                            atol=atol,
                        )
                    except ValueError as ex:
                        raise ValueError(
                            "Evaluation output shape mismatch. "
                            f"name={output_name} onnx_shape={tuple(onnx_output.shape)} "
                            f"tflite_shape={tuple(np.asarray(tflite_output).shape)} "
                            f"reason={ex}"
                        ) from ex

                    layout_alignment_summary[align_mode] = int(layout_alignment_summary[align_mode]) + 1
                    per_output_layout_alignment[output_name][align_mode] = (
                        int(per_output_layout_alignment[output_name][align_mode]) + 1
                    )
                    if align_perm is not None:
                        perm_key = ",".join(str(int(v)) for v in align_perm)
                        per_output_layout_alignment[output_name]["permutation_counts"][perm_key] = (
                            int(
                                per_output_layout_alignment[output_name]["permutation_counts"].get(
                                    perm_key, 0
                                )
                            )
                                    + 1
                            )

                    total_metrics.update(onnx_output, tflite_output)
                    per_output_metrics[output_name].update(onnx_output, tflite_output)
                    is_allclose = bool(
                        np.allclose(
                            onnx_output,
                            tflite_output,
                            rtol=rtol,
                            atol=atol,
                            equal_nan=True,
                        )
                    )
                    allclose_total += 1
                    allclose_matched += int(is_allclose)
                    per_output_allclose[output_name]["total"] += 1
                    per_output_allclose[output_name]["matched"] += int(is_allclose)
                break
            except Exception as ex:
                if (
                    custom_input_op_name_np_data_path is None
                    and not bool(force_zero_generated_inputs)
                    and _is_scatter_nd_index_oob_error(ex)
                ):
                    force_zero_generated_inputs = True
                    continue
                raise
            finally:
                if bool(use_worker_output_memmap):
                    _cleanup_memmap_manifest(onnx_manifest)
                    _cleanup_memmap_manifest(tflite_manifest)
                if bool(use_worker_input_memmap):
                    _cleanup_memmap_manifest(onnx_input_manifest)

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    overall_metrics = total_metrics.to_dict()
    metric_judgement = _judge_metrics(
        metrics=overall_metrics,
        thresholds=resolved_thresholds,
        rtol=rtol,
    )
    allclose_pass = bool(allclose_total == allclose_matched)
    evaluation_pass = bool(metric_judgement["pass"] and allclose_pass)
    report: Dict[str, Any] = {
        "schema_version": 1,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "rtol": float(rtol),
        "atol": float(atol),
        "compare_mode": resolved_compare_mode,
        "has_quantized_outputs": bool(has_quantized_outputs),
        "inputs_source": (
            "custom_input_op_name_np_data_path"
            if len(custom_inputs) > 0
            else "seeded_random"
        ),
        "onnx_input_names": onnx_input_names,
        "onnx_output_names": onnx_output_names,
        "tflite_path": tflite_path,
        "metric_thresholds": {
            k: float(v) for k, v in resolved_thresholds.items()
        },
        "overall_metrics": overall_metrics,
        "metric_threshold_judgement": metric_judgement,
        "allclose_summary": {
            "matched": int(allclose_matched),
            "total": int(allclose_total),
            "pass": allclose_pass,
        },
        "layout_alignment_summary": {
            "identity": int(layout_alignment_summary["identity"]),
            "transpose": int(layout_alignment_summary["transpose"]),
            "reshape": int(layout_alignment_summary["reshape"]),
        },
        "evaluation_pass": evaluation_pass,
        "per_output_metrics": {
            output_name: {
                "tflite_output_name": tflite_output_name_map.get(output_name, output_name),
                "allclose": {
                    "matched": int(per_output_allclose[output_name]["matched"]),
                    "total": int(per_output_allclose[output_name]["total"]),
                    "pass": (
                        per_output_allclose[output_name]["matched"]
                        == per_output_allclose[output_name]["total"]
                    ),
                },
                "layout_alignment": {
                    "identity": int(per_output_layout_alignment[output_name]["identity"]),
                    "transpose": int(per_output_layout_alignment[output_name]["transpose"]),
                    "reshape": int(per_output_layout_alignment[output_name]["reshape"]),
                    "permutation_counts": {
                        k: int(v)
                        for k, v in per_output_layout_alignment[output_name][
                            "permutation_counts"
                        ].items()
                    },
                },
                **per_output_metrics[output_name].to_dict(),
            }
            for output_name in onnx_output_names
        },
    }
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    if bool(fail_on_threshold) and not evaluation_pass:
        raise RuntimeError(
            "ONNX/TFLite evaluation failed thresholds. "
            f"report={output_report_path} "
            f"metrics={overall_metrics} "
            f"allclose={report['allclose_summary']}"
        )
    return report
