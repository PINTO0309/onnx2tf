from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    TensorIR,
    logical_layout_rank,
    normalize_logical_layout,
)
from onnx2tf.tflite_builder.pytorch_export_errors import ModelIRPyTorchExportError


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
        parts = str(hint).rsplit(":", maxsplit=1)
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


def _resize_nhwc_image_batch(
    data: np.ndarray,
    *,
    target_h: int,
    target_w: int,
) -> np.ndarray:
    import torch

    tensor = torch.from_numpy(np.asarray(data)).permute(0, 3, 1, 2)
    resized = torch.nn.functional.interpolate(
        tensor.to(dtype=torch.float32),
        size=(int(target_h), int(target_w)),
        mode="bilinear",
        align_corners=False,
    )
    return resized.permute(0, 2, 3, 1).cpu().numpy()


def _build_torchscript_image_input_from_nhwc(
    *,
    data: np.ndarray,
    expected_shape: Tuple[int, ...],
    np_dtype: np.dtype,
) -> np.ndarray:
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
            sample = _resize_nhwc_image_batch(sample, target_h=target_h, target_w=target_w)
        sample = np.transpose(sample, [0, 3, 1, 2])
    elif int(expected_shape[3]) == 3:
        target_h = int(expected_shape[1])
        target_w = int(expected_shape[2])
        if int(sample.shape[1]) != target_h or int(sample.shape[2]) != target_w:
            sample = _resize_nhwc_image_batch(sample, target_h=target_h, target_w=target_w)
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



def _is_runtime_wrapper_package_dir(package_dir: Path) -> bool:
    model_path = package_dir / "model.py"
    if not model_path.exists():
        return False
    try:
        model_source = model_path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "load_generated_model_package" in model_source


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


def _generated_package_non_native_skip_reason(package_path: Path) -> Optional[str]:
    metadata_path = package_path / "metadata.json"
    metadata: Dict[str, Any] = {}
    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    execution_backend = str(metadata.get("execution_backend", "")).strip().lower()
    if execution_backend == "" and _is_runtime_wrapper_package_dir(package_path):
        execution_backend = "runtime_wrapper"
    if execution_backend not in {"", "native"}:
        return (
            "artifact export is skipped for generated packages with non-native execution "
            f"backend. execution_backend={execution_backend or 'native'}"
        )
    return None


def _generated_package_torch_export_skip_reason(package_path: Path) -> Optional[str]:
    non_native_skip_reason = _generated_package_non_native_skip_reason(package_path)
    if non_native_skip_reason is not None:
        return non_native_skip_reason
    model_path = package_path / "model.py"
    if not model_path.exists():
        return None
    try:
        model_source = model_path.read_text(encoding="utf-8")
    except Exception:
        return None
    if re.search(
        r"def _run_nms_\d+\(self, boxes: torch\.Tensor, scores: torch\.Tensor, max_output_size: torch\.Tensor",
        model_source,
    ):
        return (
            "torch.export-based artifacts are skipped for generated packages "
            "with data-dependent NON_MAX_SUPPRESSION_V4 parameters."
        )
    if (
        "_apply_non_max_suppression_v4(" in model_source
        and (
            "torch.as_tensor(min(2147483647, (_shape_list(" in model_source
            or "torch.as_tensor(min(2147483647, (_tensor_shape_list(" in model_source
            or (
                "torch.as_tensor(min(2147483647, " in model_source
                and ".shape[" in model_source
            )
        )
    ):
        return (
            "torch.export-based artifacts are skipped for generated packages "
            "with data-dependent NON_MAX_SUPPRESSION_V4 parameters."
        )

    if re.search(
        r"selected_indices_nms_valid_indices_c\d+\s*=\s*torch\.arange\(\s*start=0,\s*"
        r"end=selected_indices_nms_valid_count_scalar_c\d+\.reshape\(-1\)\[0\]\.item\(\)",
        model_source,
    ):
        return (
            "torch.export-based artifacts are skipped for generated packages "
            "with data-dependent NON_MAX_SUPPRESSION output-shape post-processing."
        )
    return None


def _run_generated_package_export_child(
    *,
    example_inputs: Tuple[Any, ...],
    child_script: str,
    package_path: Path,
    artifact_path: Path,
    child_payload: Dict[str, Any],
    child_args: Optional[List[str]] = None,
    temp_prefix: str,
    timeout_sec: int = 0,
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
        child_run_kwargs: Dict[str, Any] = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "text": True,
            "check": False,
        }
        if int(timeout_sec) > 0:
            child_run_kwargs["timeout"] = float(timeout_sec)
        try:
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
                **child_run_kwargs,
            )
        except subprocess.TimeoutExpired:
            return None, (
                "timed out after "
                f"{int(timeout_sec)}s while exporting generated PyTorch package child artifact."
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
    boundary_shape_map = model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    public_tensor_names = {
        str(name) for name in list(model_ir.inputs) + list(model_ir.outputs)
    }
    current_public_layouts: Dict[str, str] = {}
    tensors: Dict[str, Dict[str, Any]] = {}
    for name, tensor in model_ir.tensors.items():
        tensor_name = str(name)
        tensor_meta = _serializable_tensor_meta(tensor)
        if tensor_name in public_tensor_names:
            current_public_layouts[tensor_name] = str(tensor_meta["logical_layout"])
            boundary_shape = boundary_shape_map.get(tensor_name, None)
            if isinstance(boundary_shape, list) and len(boundary_shape) == len(tensor_meta["shape"]):
                tensor_meta["shape"] = [
                    max(1, int(v)) if int(v) >= 0 else 1
                    for v in list(boundary_shape)
                ]
                tensor_meta["shape_signature"] = [int(v) for v in list(boundary_shape)]
            public_layout = normalize_logical_layout(public_layout_map.get(tensor_name, None))
            if logical_layout_rank(public_layout) == len(tensor_meta["shape"]):
                tensor_meta["logical_layout"] = public_layout
        tensors[tensor_name] = tensor_meta
    return {
        "schema_version": 1,
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": tensors,
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
        "current_public_layouts": _serializable_value(current_public_layouts),
        "boundary_shape_signatures": _serializable_value(boundary_shape_map),
    }
