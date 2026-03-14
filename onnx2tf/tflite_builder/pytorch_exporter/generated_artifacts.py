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


from .common import ModelIRPyTorchExportError

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

def _clear_onnx_graph_and_node_metadata_in_place(graph: onnx.GraphProto) -> None:
    del graph.metadata_props[:]
    for node in graph.node:
        del node.metadata_props[:]
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.GRAPH:
                _clear_onnx_graph_and_node_metadata_in_place(attr.g)
            elif attr.type == onnx.AttributeProto.GRAPHS:
                for subgraph in attr.graphs:
                    _clear_onnx_graph_and_node_metadata_in_place(subgraph)

def _onnx_node_maps(
    graph: onnx.GraphProto,
) -> Tuple[Dict[str, onnx.NodeProto], Dict[str, List[onnx.NodeProto]]]:
    producer_map: Dict[str, onnx.NodeProto] = {}
    consumer_map: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for output_name in node.output:
            producer_map[str(output_name)] = node
        for input_name in node.input:
            consumer_map.setdefault(str(input_name), []).append(node)
    return producer_map, consumer_map

def _onnx_node_attr(node: onnx.NodeProto, name: str) -> Optional[Any]:
    for attr in node.attribute:
        if attr.name == name:
            return onnx.helper.get_attribute_value(attr)
    return None

def _onnx_replace_all_node_inputs(
    graph: onnx.GraphProto,
    *,
    old_name: str,
    new_name: str,
) -> None:
    if old_name == new_name:
        return
    for node in graph.node:
        for input_index, input_name in enumerate(node.input):
            if str(input_name) == str(old_name):
                node.input[input_index] = str(new_name)
    for output in graph.output:
        if str(output.name) == str(old_name):
            output.name = str(new_name)

def _onnx_remove_nodes_by_name(
    graph: onnx.GraphProto,
    node_names: Sequence[str],
) -> None:
    remove_name_set = {str(name) for name in list(node_names)}
    if not remove_name_set:
        return
    kept_nodes = [node for node in graph.node if str(node.name) not in remove_name_set]
    del graph.node[:]
    graph.node.extend(kept_nodes)

def _onnx_get_initializer_index(graph: onnx.GraphProto, name: str) -> Optional[int]:
    for initializer_index, initializer in enumerate(graph.initializer):
        if str(initializer.name) == str(name):
            return int(initializer_index)
    return None

def _onnx_set_initializer_array(
    graph: onnx.GraphProto,
    *,
    name: str,
    array: np.ndarray,
) -> None:
    tensor = onnx.numpy_helper.from_array(np.asarray(array), name=str(name))
    initializer_index = _onnx_get_initializer_index(graph, str(name))
    if initializer_index is None:
        graph.initializer.append(tensor)
    else:
        graph.initializer[initializer_index].CopyFrom(tensor)

def _onnx_make_unique_initializer_name(graph: onnx.GraphProto, base_name: str) -> str:
    existing_names = {str(initializer.name) for initializer in graph.initializer}
    existing_names.update(str(node.output[0]) for node in graph.node if len(node.output) >= 1)
    candidate = str(base_name)
    suffix_index = 0
    while candidate in existing_names:
        suffix_index += 1
        candidate = f"{base_name}_{suffix_index}"
    return candidate

def _onnx_get_initializer_array(
    graph: onnx.GraphProto,
    name: str,
) -> Optional[np.ndarray]:
    initializer_index = _onnx_get_initializer_index(graph, str(name))
    if initializer_index is None:
        return None
    return onnx.numpy_helper.to_array(graph.initializer[initializer_index])

def _onnx_convert_pads_nhwc_to_nchw(pads: Sequence[int] | np.ndarray) -> Optional[np.ndarray]:
    pad_values = np.asarray(list(pads), dtype=np.int64).reshape(-1)
    if pad_values.size != 8:
        return None
    begin = pad_values[:4]
    end = pad_values[4:]
    reorder = [0, 3, 1, 2]
    return np.concatenate([begin[reorder], end[reorder]], axis=0).astype(np.int64)

def _onnx_fold_relu_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        transpose_consumers = consumer_map.get(str(transpose_node.output[0]), [])
        if len(transpose_consumers) != 1:
            continue
        relu_node = transpose_consumers[0]
        if str(relu_node.op_type) != "Relu":
            continue
        relu_consumers = consumer_map.get(str(relu_node.output[0]), [])
        if len(relu_consumers) != 1:
            continue
        inverse_node = relu_consumers[0]
        if str(inverse_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        relu_node.input[0] = str(transpose_node.input[0])
        relu_node.output[0] = str(inverse_node.output[0])
        remove_node_names.extend([str(transpose_node.name), str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)

def _onnx_fold_reducesum_sigmoid_layout_bridges_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for transpose_node in list(graph.node):
        if str(transpose_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(transpose_node, "perm") or []) != [0, 2, 3, 1]:
            continue
        transpose_consumers = consumer_map.get(str(transpose_node.output[0]), [])
        if len(transpose_consumers) != 1:
            continue
        reduce_node = transpose_consumers[0]
        if str(reduce_node.op_type) != "ReduceSum":
            continue
        axes_name = str(reduce_node.input[1]) if len(reduce_node.input) >= 2 else ""
        axes_array = _onnx_get_initializer_array(graph, axes_name)
        if axes_array is None or [int(v) for v in axes_array.reshape(-1)] != [3]:
            continue
        if int(_onnx_node_attr(reduce_node, "keepdims") or 0) != 1:
            continue
        reduce_consumers = consumer_map.get(str(reduce_node.output[0]), [])
        if len(reduce_consumers) != 1:
            continue
        sigmoid_node = reduce_consumers[0]
        if str(sigmoid_node.op_type) != "Sigmoid":
            continue
        sigmoid_consumers = consumer_map.get(str(sigmoid_node.output[0]), [])
        if len(sigmoid_consumers) != 1:
            continue
        inverse_node = sigmoid_consumers[0]
        if str(inverse_node.op_type) != "Transpose":
            continue
        if list(_onnx_node_attr(inverse_node, "perm") or []) != [0, 3, 1, 2]:
            continue
        new_axes_name = _onnx_make_unique_initializer_name(graph, f"{axes_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_axes_name, array=np.asarray([1], dtype=np.int64))
        reduce_node.input[0] = str(transpose_node.input[0])
        reduce_node.input[1] = str(new_axes_name)
        sigmoid_node.output[0] = str(inverse_node.output[0])
        remove_node_names.extend([str(transpose_node.name), str(inverse_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)

def _onnx_fold_inverse_transpose_pairs_in_place(graph: onnx.GraphProto) -> None:
    _, consumer_map = _onnx_node_maps(graph)
    remove_node_names: List[str] = []
    for first_node in list(graph.node):
        if str(first_node.op_type) != "Transpose":
            continue
        first_perm = [int(v) for v in list(_onnx_node_attr(first_node, "perm") or [])]
        if not first_perm:
            continue
        first_consumers = consumer_map.get(str(first_node.output[0]), [])
        if len(first_consumers) != 1:
            continue
        second_node = first_consumers[0]
        if str(second_node.op_type) != "Transpose":
            continue
        second_perm = [int(v) for v in list(_onnx_node_attr(second_node, "perm") or [])]
        inverse_perm = [0] * len(first_perm)
        for perm_index, perm_value in enumerate(first_perm):
            inverse_perm[int(perm_value)] = int(perm_index)
        if second_perm != inverse_perm:
            continue
        _onnx_replace_all_node_inputs(
            graph,
            old_name=str(second_node.output[0]),
            new_name=str(first_node.input[0]),
        )
        remove_node_names.extend([str(first_node.name), str(second_node.name)])
    _onnx_remove_nodes_by_name(graph, remove_node_names)

def _onnx_optimize_pidnet_spp_transpose_bridges_in_place(graph: onnx.GraphProto) -> None:
    node_by_name = {str(node.name): node for node in graph.node}
    required_node_names = {
        "node_permute_27",
        "node_permute_28",
        "node_pad_38",
        "node_pad_39",
        "node_pad_40",
        "node_permute_33",
        "node_permute_36",
        "node_avg_pool2d_1",
        "node_avg_pool2d_2",
        "node_mean",
        "node_mul_6",
        "node_mul_7",
        "node_mul_11",
        "node_permute_41",
        "node_add_28",
    }
    if any(node_name not in node_by_name for node_name in required_node_names):
        return

    node_permute_27 = node_by_name["node_permute_27"]
    node_permute_28 = node_by_name["node_permute_28"]
    node_pad_38 = node_by_name["node_pad_38"]
    node_pad_39 = node_by_name["node_pad_39"]
    node_pad_40 = node_by_name["node_pad_40"]
    node_permute_33 = node_by_name["node_permute_33"]
    node_permute_36 = node_by_name["node_permute_36"]
    node_avg_pool2d_1 = node_by_name["node_avg_pool2d_1"]
    node_avg_pool2d_2 = node_by_name["node_avg_pool2d_2"]
    node_mean = node_by_name["node_mean"]
    node_mul_6 = node_by_name["node_mul_6"]
    node_mul_7 = node_by_name["node_mul_7"]
    node_mul_11 = node_by_name["node_mul_11"]
    node_permute_41 = node_by_name["node_permute_41"]
    node_add_28 = node_by_name["node_add_28"]

    if list(_onnx_node_attr(node_permute_27, "perm") or []) != [0, 2, 3, 1]:
        return
    if list(_onnx_node_attr(node_permute_28, "perm") or []) != [0, 3, 1, 2]:
        return
    if list(_onnx_node_attr(node_permute_33, "perm") or []) != [0, 3, 1, 2]:
        return
    if list(_onnx_node_attr(node_permute_36, "perm") or []) != [0, 3, 1, 2]:
        return
    if list(_onnx_node_attr(node_permute_41, "perm") or []) != [0, 3, 1, 2]:
        return

    base_input_name = str(node_permute_27.input[0])
    node_mul_6.input[0] = base_input_name
    node_pad_38.input[0] = base_input_name
    node_mul_7.input[0] = base_input_name
    node_pad_39.input[0] = base_input_name
    node_pad_40.input[0] = base_input_name
    node_mean.input[0] = base_input_name
    node_avg_pool2d_1.input[0] = str(node_pad_39.output[0])
    node_avg_pool2d_2.input[0] = str(node_pad_40.output[0])
    node_add_28.input[0] = str(node_mul_11.output[0])

    mean_axes_name = str(node_mean.input[1]) if len(node_mean.input) >= 2 else ""
    if mean_axes_name:
        new_mean_axes_name = _onnx_make_unique_initializer_name(graph, f"{mean_axes_name}_nchw")
        _onnx_set_initializer_array(graph, name=new_mean_axes_name, array=np.asarray([2, 3], dtype=np.int64))
        node_mean.input[1] = str(new_mean_axes_name)

    for pad_node in [node_pad_39, node_pad_40]:
        pad_name = str(pad_node.input[1]) if len(pad_node.input) >= 2 else ""
        pad_values = _onnx_get_initializer_array(graph, pad_name)
        nchw_pad_values = (
            _onnx_convert_pads_nhwc_to_nchw(pad_values)
            if pad_values is not None
            else None
        )
        if pad_name and nchw_pad_values is not None:
            new_pad_name = _onnx_make_unique_initializer_name(graph, f"{pad_name}_nchw")
            _onnx_set_initializer_array(graph, name=new_pad_name, array=nchw_pad_values)
            pad_node.input[1] = str(new_pad_name)

    mul_const_name = str(node_mul_11.input[1]) if len(node_mul_11.input) >= 2 else ""
    mul_const_array = _onnx_get_initializer_array(graph, mul_const_name)
    if mul_const_name and mul_const_array is not None and len(mul_const_array.shape) == 4:
        _onnx_set_initializer_array(
            graph,
            name=mul_const_name,
            array=np.transpose(mul_const_array, (0, 3, 1, 2)),
        )

    _onnx_remove_nodes_by_name(
        graph,
        [
            str(node_permute_27.name),
            str(node_permute_28.name),
            str(node_permute_33.name),
            str(node_permute_36.name),
            str(node_permute_41.name),
        ],
    )

def _optimize_dynamo_exported_onnx_in_place(model: onnx.ModelProto) -> None:
    _onnx_fold_relu_layout_bridges_in_place(model.graph)
    _onnx_fold_reducesum_sigmoid_layout_bridges_in_place(model.graph)
    _onnx_fold_inverse_transpose_pairs_in_place(model.graph)
    _onnx_optimize_pidnet_spp_transpose_bridges_in_place(model.graph)

def _onnx_model_uses_external_data(model: onnx.ModelProto) -> bool:
    return any(
        initializer.data_location == onnx.TensorProto.EXTERNAL
        for initializer in model.graph.initializer
    )

def _inspect_onnx_uses_external_data(onnx_path: Path) -> bool:
    model = onnx.load(str(onnx_path), load_external_data=False)
    return _onnx_model_uses_external_data(model)

def _sanitize_dynamo_exported_onnx_metadata(onnx_path: Path) -> None:
    external_data_sidecar_path = onnx_path.with_name(f"{onnx_path.name}.data")
    original_uses_external_data = _inspect_onnx_uses_external_data(onnx_path)
    model = onnx.load(str(onnx_path))
    del model.metadata_props[:]
    _clear_onnx_graph_and_node_metadata_in_place(model.graph)
    _optimize_dynamo_exported_onnx_in_place(model)
    onnx.checker.check_model(model)
    if original_uses_external_data:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_sidecar_path.name,
            size_threshold=0,
        )
    else:
        onnx.save_model(
            model,
            str(onnx_path),
            save_as_external_data=False,
        )
    if not _inspect_onnx_uses_external_data(onnx_path) and external_data_sidecar_path.exists():
        external_data_sidecar_path.unlink()

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

    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="TorchScript export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
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
        return None
    skip_reason = _generated_package_non_native_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="torchscript",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "trace_mode": None,
                "skipped_reason": skip_reason,
            },
        )
        return None
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
import importlib
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
runtime_module = importlib.import_module(f"{module_name}.runtime")
setattr(runtime_module, "_ONNX2TF_DISABLE_SYMBOLIC_SHAPE_TENSORS", True)
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
    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="Dynamo ONNX export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        return None
    skip_reason = _generated_package_torch_export_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="dynamo_onnx",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "skipped_reason": skip_reason,
            },
        )
        return None
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
import importlib
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
runtime_module = importlib.import_module(f"{module_name}.runtime")
setattr(runtime_module, "_ONNX2TF_DISABLE_SYMBOLIC_SHAPE_TENSORS", True)
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
setattr(model, "_onnx2tf_torch_export_mode", True)
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
    _sanitize_dynamo_exported_onnx_metadata(dynamo_onnx_path)
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
    try:
        package_path, metadata_path, metadata = _load_generated_package_export_metadata(
            package_dir=package_dir,
            export_label="ExportedProgram export",
        )
    except Exception as ex:
        if raise_on_failure:
            raise
        package_path = Path(package_dir)
        metadata_path = package_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        else:
            metadata = {}
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            error=str(ex),
        )
        return None
    skip_reason = _generated_package_torch_export_skip_reason(package_path)
    if skip_reason is not None:
        _write_generated_package_export_metadata(
            metadata_path=metadata_path,
            metadata=metadata,
            metadata_key="exported_program",
            file_name=None,
            example_input_shapes={},
            dynamic_inputs_present=_metadata_has_dynamic_public_inputs(metadata),
            extra_fields={
                "skipped_reason": skip_reason,
            },
        )
        return None
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
setattr(model, "_onnx2tf_torch_export_mode", True)

def _prune_alias_nodes(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    graph_signature = getattr(exported_program, "graph_signature", None)
    user_output_names = set()
    if graph_signature is not None:
        try:
            user_output_names = {str(name) for name in list(graph_signature.user_outputs)}
        except Exception:
            user_output_names = set()
    changed = False
    for node in list(graph.nodes):
        if (
            node.op == "call_function"
            and str(node.target) == "aten.alias.default"
            and len(node.args) >= 1
            and isinstance(node.args[0], torch.fx.Node)
            and str(node.name) not in user_output_names
        ):
            node.replace_all_uses_with(node.args[0])
            graph.erase_node(node)
            changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _fold_inverse_permute_round_trips(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    changed = False

    def _normalize_perm(arg):
        if not isinstance(arg, (list, tuple)):
            return None
        return [int(v) for v in arg]

    def _inverse_perm(perm):
        inverse = [0] * len(perm)
        for idx, value in enumerate(perm):
            inverse[int(value)] = int(idx)
        return inverse

    def _match_permute_chain_source(node, perm):
        if not isinstance(node, torch.fx.Node):
            return None
        if (
            node.op == "call_function"
            and str(node.target) == "aten.permute.default"
            and len(node.args) >= 2
            and _normalize_perm(node.args[1]) == perm
            and isinstance(node.args[0], torch.fx.Node)
        ):
            return node.args[0]
        if (
            node.op == "call_function"
            and str(node.target) == "aten.contiguous.default"
            and len(node.args) >= 1
            and isinstance(node.args[0], torch.fx.Node)
        ):
            input_node = node.args[0]
            if (
                input_node.op == "call_function"
                and str(input_node.target) == "aten.permute.default"
                and len(input_node.args) >= 2
                and _normalize_perm(input_node.args[1]) == perm
                and isinstance(input_node.args[0], torch.fx.Node)
            ):
                return input_node.args[0]
        return None

    def _match_binary_input_source(node, perm):
        return _match_permute_chain_source(node, perm)

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        perm = _normalize_perm(node.args[1])
        if perm is None:
            continue
        root_replacement = source
        root_users = list(node.users)
        if len(root_users) == 1 and (
            root_users[0].op == "call_function"
            and str(root_users[0].target) == "aten.contiguous.default"
        ):
            branch_input = root_users[0]
        else:
            branch_input = node
        branch_users = list(branch_input.users)
        if len(branch_users) == 1:
            inverse_node = branch_users[0]
            if (
                inverse_node.op == "call_function"
                and str(inverse_node.target) == "aten.permute.default"
                and len(inverse_node.args) >= 2
            ):
                inverse_perm = _normalize_perm(inverse_node.args[1])
                if inverse_perm is not None and inverse_perm == _inverse_perm(perm):
                    inverse_users = list(inverse_node.users)
                    if (
                        len(inverse_users) == 1
                        and inverse_users[0].op == "call_function"
                        and str(inverse_users[0].target) == "aten.contiguous.default"
                    ):
                        inverse_users[0].replace_all_uses_with(root_replacement)
                    else:
                        inverse_node.replace_all_uses_with(root_replacement)
                    changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        inverse_perm = _normalize_perm(node.args[1])
        if inverse_perm is None:
            continue
        source = _match_permute_chain_source(node.args[0], _inverse_perm(inverse_perm))
        if source is None:
            continue
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(source)
        else:
            node.replace_all_uses_with(source)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        pad_node = node.args[0]
        if not isinstance(pad_node, torch.fx.Node):
            continue
        if (
            pad_node.op != "call_function"
            or str(pad_node.target) != "aten.pad.default"
            or len(pad_node.args) < 4
        ):
            continue
        source = _match_permute_chain_source(pad_node.args[0], [0, 2, 3, 1])
        if source is None:
            continue
        pad_values = list(pad_node.args[1])
        if len(pad_values) != 6:
            continue
        if [int(v) for v in pad_values[:2]] != [0, 0]:
            continue
        cf_pad = [int(pad_values[2]), int(pad_values[3]), int(pad_values[4]), int(pad_values[5])]
        with graph.inserting_before(pad_node):
            folded_pad = graph.call_function(
                pad_node.target,
                args=(source, cf_pad, *tuple(pad_node.args[2:])),
                kwargs=dict(pad_node.kwargs),
            )
        folded_pad.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(folded_pad)
        else:
            node.replace_all_uses_with(folded_pad)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 2, 3, 1]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        node_users = list(node.users)
        if len(node_users) != 1:
            continue
        contiguous_node = node_users[0]
        if (
            contiguous_node.op != "call_function"
            or str(contiguous_node.target) != "aten.contiguous.default"
        ):
            continue
        contiguous_users = list(contiguous_node.users)
        if len(contiguous_users) != 1:
            continue
        sum_node = contiguous_users[0]
        if (
            sum_node.op != "call_function"
            or str(sum_node.target) != "aten.sum.dim_IntList"
            or len(sum_node.args) < 3
            or list(sum_node.args[1]) != [3]
            or bool(sum_node.args[2]) is not True
        ):
            continue
        sum_users = list(sum_node.users)
        if len(sum_users) != 1:
            continue
        sigmoid_node = sum_users[0]
        if (
            sigmoid_node.op != "call_function"
            or str(sigmoid_node.target) != "aten.sigmoid.default"
        ):
            continue
        sigmoid_users = list(sigmoid_node.users)
        if len(sigmoid_users) != 1:
            continue
        inverse_node = sigmoid_users[0]
        if (
            inverse_node.op != "call_function"
            or str(inverse_node.target) != "aten.permute.default"
            or len(inverse_node.args) < 2
            or _normalize_perm(inverse_node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        with graph.inserting_before(node):
            folded_sum = graph.call_function(
                sum_node.target,
                args=(source, [1], True),
                kwargs=dict(sum_node.kwargs),
            )
            folded_sigmoid = graph.call_function(
                sigmoid_node.target,
                args=(folded_sum,),
                kwargs=dict(sigmoid_node.kwargs),
            )
        folded_sum.meta = dict(getattr(sum_node, "meta", {}))
        folded_sigmoid.meta = dict(getattr(inverse_node, "meta", {}))
        inverse_users = list(inverse_node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(folded_sigmoid)
        else:
            inverse_node.replace_all_uses_with(folded_sigmoid)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        source_shape = None
        source_meta_val = getattr(source, "meta", {}).get("val", None)
        if isinstance(source_meta_val, torch.Tensor):
            source_shape = [int(v) for v in list(source_meta_val.shape)]
        elif source.op == "get_attr" and isinstance(source.target, str):
            source_tensor = getattr(graph_module, source.target, None)
            if isinstance(source_tensor, torch.Tensor):
                source_shape = [int(v) for v in list(source_tensor.shape)]
        if source_shape is None or len(source_shape) != 4:
            continue
        non_singleton_axes = [idx for idx, dim in enumerate(source_shape) if int(dim) > 1]
        if len(non_singleton_axes) != 1 or int(non_singleton_axes[0]) != 3:
            continue
        reshaped_shape = [int(source_shape[0]), int(source_shape[3]), int(source_shape[1]), int(source_shape[2])]
        with graph.inserting_before(node):
            folded_reshape = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(source, reshaped_shape),
                kwargs={},
            )
        folded_reshape.meta = dict(getattr(node, "meta", {}))
        node_users = list(node.users)
        if (
            len(node_users) == 1
            and node_users[0].op == "call_function"
            and str(node_users[0].target) == "aten.contiguous.default"
        ):
            node_users[0].replace_all_uses_with(folded_reshape)
        else:
            node.replace_all_uses_with(folded_reshape)
        changed = True

    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or _normalize_perm(node.args[1]) != [0, 3, 1, 2]
        ):
            continue
        mul_node = node.args[0]
        if (
            not isinstance(mul_node, torch.fx.Node)
            or mul_node.op != "call_function"
            or str(mul_node.target) != "aten.mul.Tensor"
            or len(mul_node.args) != 2
        ):
            continue
        mean_node = None
        const_node = None
        for arg in mul_node.args:
            if (
                isinstance(arg, torch.fx.Node)
                and arg.op == "call_function"
                and str(arg.target) == "aten.mean.dim"
                and len(arg.args) >= 3
                and list(arg.args[1]) == [1, 2]
                and bool(arg.args[2]) is True
            ):
                mean_node = arg
            elif isinstance(arg, torch.fx.Node):
                const_node = arg
        if mean_node is None or const_node is None:
            continue
        mean_input = mean_node.args[0]
        if not (
            isinstance(mean_input, torch.fx.Node)
            and mean_input.op == "call_function"
            and str(mean_input.target) == "aten.contiguous.default"
            and len(mean_input.args) >= 1
            and isinstance(mean_input.args[0], torch.fx.Node)
            and mean_input.args[0].op == "call_function"
            and str(mean_input.args[0].target) == "aten.permute.default"
            and len(mean_input.args[0].args) >= 2
            and _normalize_perm(mean_input.args[0].args[1]) == [0, 2, 3, 1]
            and isinstance(mean_input.args[0].args[0], torch.fx.Node)
        ):
            continue
        source = mean_input.args[0].args[0]
        const_shape = None
        const_meta_val = getattr(const_node, "meta", {}).get("val", None)
        if isinstance(const_meta_val, torch.Tensor):
            const_shape = [int(v) for v in list(const_meta_val.shape)]
        if const_node.op == "get_attr" and isinstance(const_node.target, str):
            const_tensor = getattr(graph_module, const_node.target, None)
            if isinstance(const_tensor, torch.Tensor):
                const_shape = [int(v) for v in list(const_tensor.shape)]
        if const_shape is None or len(const_shape) != 4 or const_shape[:3] != [1, 1, 1]:
            continue
        reshaped_const_shape = [1, int(const_shape[3]), 1, 1]
        with graph.inserting_before(mean_node):
            folded_mean = graph.call_function(
                mean_node.target,
                args=(source, [2, 3], True),
                kwargs=dict(mean_node.kwargs),
            )
            folded_const = graph.call_function(
                torch.ops.aten.reshape.default,
                args=(const_node, reshaped_const_shape),
                kwargs={},
            )
            folded_mul = graph.call_function(
                mul_node.target,
                args=(
                    folded_mean if mul_node.args[0] is mean_node else folded_const,
                    folded_const if mul_node.args[0] is mean_node else folded_mean,
                ),
                kwargs=dict(mul_node.kwargs),
            )
        folded_mean.meta = dict(getattr(mean_node, "meta", {}))
        folded_const.meta = dict(getattr(const_node, "meta", {}))
        folded_mul.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(folded_mul)
        else:
            node.replace_all_uses_with(folded_mul)
        changed = True

    binary_targets = {
        "aten.add.Tensor",
        "aten.div.Tensor",
        "aten.mul.Tensor",
        "aten.sub.Tensor",
    }
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
        ):
            continue
        inverse_perm = _normalize_perm(node.args[1])
        if inverse_perm is None:
            continue
        binary_node = node.args[0]
        if not isinstance(binary_node, torch.fx.Node):
            continue
        if (
            binary_node.op != "call_function"
            or str(binary_node.target) not in binary_targets
            or len(binary_node.args) != 2
        ):
            continue
        input_perm = _inverse_perm(inverse_perm)
        lhs_source = _match_binary_input_source(binary_node.args[0], input_perm)
        rhs_source = _match_binary_input_source(binary_node.args[1], input_perm)
        if lhs_source is None or rhs_source is None:
            continue
        with graph.inserting_before(binary_node):
            folded_binary = graph.call_function(
                binary_node.target,
                args=(lhs_source, rhs_source),
                kwargs=dict(binary_node.kwargs),
            )
        folded_binary.meta = dict(getattr(node, "meta", {}))
        inverse_users = list(node.users)
        if (
            len(inverse_users) == 1
            and inverse_users[0].op == "call_function"
            and str(inverse_users[0].target) == "aten.contiguous.default"
        ):
            inverse_users[0].replace_all_uses_with(folded_binary)
        else:
            node.replace_all_uses_with(folded_binary)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

def _fold_layout_preserving_permute_chains(exported_program):
    graph_module = getattr(exported_program, "graph_module", None)
    if graph_module is None:
        return exported_program
    graph = graph_module.graph
    changed = False
    unary_targets = {
        "aten.relu.default",
    }
    for node in list(graph.nodes):
        if (
            node.op != "call_function"
            or str(node.target) != "aten.permute.default"
            or len(node.args) < 2
            or list(node.args[1]) != [0, 2, 3, 1]
        ):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node):
            continue
        permute_users = list(node.users)
        if len(permute_users) != 1:
            continue
        contiguous_node = permute_users[0]
        if (
            contiguous_node.op != "call_function"
            or str(contiguous_node.target) != "aten.contiguous.default"
        ):
            continue
        contiguous_users = list(contiguous_node.users)
        if len(contiguous_users) != 1:
            continue
        unary_node = contiguous_users[0]
        if (
            unary_node.op != "call_function"
            or str(unary_node.target) not in unary_targets
        ):
            continue
        inverse_permute_nodes = list(unary_node.users)
        if len(inverse_permute_nodes) == 0:
            continue
        if any(
            inverse_node.op != "call_function"
            or str(inverse_node.target) != "aten.permute.default"
            or len(inverse_node.args) < 2
            or list(inverse_node.args[1]) != [0, 3, 1, 2]
            for inverse_node in inverse_permute_nodes
        ):
            continue
        with graph.inserting_before(node):
            folded_unary = graph.call_function(
                unary_node.target,
                args=(source, *tuple(unary_node.args[1:])),
                kwargs=dict(unary_node.kwargs),
            )
        folded_unary.meta = dict(getattr(source, "meta", {}))
        for inverse_node in inverse_permute_nodes:
            inverse_users = list(inverse_node.users)
            if (
                len(inverse_users) == 1
                and inverse_users[0].op == "call_function"
                and str(inverse_users[0].target) == "aten.contiguous.default"
            ):
                inverse_users[0].replace_all_uses_with(folded_unary)
            else:
                inverse_node.replace_all_uses_with(folded_unary)
        changed = True
    if changed:
        graph.eliminate_dead_code()
        graph.lint()
        graph_module.recompile()
    return exported_program

with torch.no_grad():
    exported = torch.export.export(model, example_inputs)
exported = _prune_alias_nodes(exported)
exported = _fold_inverse_permute_round_trips(exported)
exported = _fold_layout_preserving_permute_chains(exported)
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
    try:
        _strip_stack_traces_from_exported_program_archive(exported_program_path)
    except Exception as ex:
        if raise_on_failure:
            raise ModelIRPyTorchExportError(
                "ExportedProgram archive cleanup failed for the generated native PyTorch package. "
                f"package_dir={package_dir} artifact={exported_program_path} details={ex}"
            ) from ex
        last_error_message = str(ex)
    _write_generated_package_export_metadata(
        metadata_path=metadata_path,
        metadata=metadata,
        metadata_key="exported_program",
        file_name=str(child_payload.get("file_name", exported_program_file_name)),
        example_input_shapes=example_input_shapes,
        dynamic_inputs_present=dynamic_inputs_present,
    )
    return str(exported_program_path)

def _strip_stack_traces_from_exported_program_archive(exported_program_path: Path) -> None:
    archive_path = Path(exported_program_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"ExportedProgram archive not found. path={archive_path}")
    with tempfile.NamedTemporaryFile(
        prefix="onnx2tf_exported_program_strip_",
        suffix=".pt2",
        delete=False,
        dir=str(archive_path.parent),
    ) as tmp_file:
        temp_archive_path = Path(tmp_file.name)
    try:
        removed_count = 0
        with zipfile.ZipFile(str(archive_path), "r") as source_archive, zipfile.ZipFile(
            str(temp_archive_path),
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as stripped_archive:
            for info in source_archive.infolist():
                payload = source_archive.read(info.filename)
                if info.filename.endswith("models/model.json"):
                    model_json = json.loads(payload)

                    def _strip_stack_trace_fields(value: Any) -> None:
                        nonlocal removed_count
                        if isinstance(value, dict):
                            if "stack_trace" in value:
                                del value["stack_trace"]
                                removed_count += 1
                            for child in value.values():
                                _strip_stack_trace_fields(child)
                            return
                        if isinstance(value, list):
                            for child in value:
                                _strip_stack_trace_fields(child)

                    _strip_stack_trace_fields(model_json)
                    payload = json.dumps(model_json, separators=(",", ":")).encode("utf-8")
                stripped_archive.writestr(info, payload)
        if removed_count == 0:
            temp_archive_path.unlink(missing_ok=True)
            return
        temp_archive_path.replace(archive_path)
    except Exception:
        temp_archive_path.unlink(missing_ok=True)
        raise

