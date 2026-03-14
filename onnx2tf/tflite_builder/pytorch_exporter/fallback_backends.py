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


from .common import ModelIRPyTorchExportError, _perm_cf_to_cl, _perm_cl_to_cf, _read_onnx_squeeze_axes, _read_onnx_unsqueeze_axes
from .layout_normalization import _ensure_public_boundary_layout_bridges
from .native_codegen import _serializable_tensor_meta, _serializable_value, _write_generated_package_common_files, _write_wrapper_model_file

def _build_tflite_backed_metadata_payload(
    *,
    model_ir: ModelIR,
    tflite_file_name: str,
) -> Dict[str, Any]:
    inferred = copy.deepcopy(model_ir)
    infer_model_ir_logical_layouts(inferred)
    boundary_shape_map = inferred.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}

    public_tensor_names = {
        str(name) for name in list(inferred.inputs) + list(inferred.outputs)
    }
    tensors: Dict[str, Dict[str, Any]] = {}
    for tensor_name in sorted(public_tensor_names):
        tensor = inferred.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        tensor_meta = _serializable_tensor_meta(tensor)
        boundary_shape = boundary_shape_map.get(str(tensor_name), None)
        if isinstance(boundary_shape, list) and len(boundary_shape) == len(tensor_meta["shape"]):
            tensor_meta["shape"] = [max(1, int(v)) if int(v) >= 0 else 1 for v in boundary_shape]
            tensor_meta["shape_signature"] = [int(v) for v in boundary_shape]
        tensor_meta["has_data"] = False
        tensors[str(tensor_name)] = tensor_meta

    current_public_layouts = {
        str(name): normalize_logical_layout(inferred.tensors[str(name)].logical_layout)
        for name in public_tensor_names
        if str(name) in inferred.tensors
    }
    return {
        "schema_version": 1,
        "execution_backend": "tflite",
        "name": str(inferred.name),
        "description": str(inferred.description),
        "inputs": [str(v) for v in inferred.inputs],
        "outputs": [str(v) for v in inferred.outputs],
        "tensors": tensors,
        "operators": [],
        "public_layouts": _serializable_value(
            dict(inferred.metadata.get("onnx_public_layout_map", {}))
        ),
        "current_public_layouts": _serializable_value(current_public_layouts),
        "boundary_shape_signatures": _serializable_value(boundary_shape_map),
        "tflite_file_name": str(tflite_file_name),
    }

def _build_saved_model_backed_metadata_payload(
    *,
    model_ir: ModelIR,
    saved_model_dir_name: str,
) -> Dict[str, Any]:
    metadata = _build_tflite_backed_metadata_payload(
        model_ir=model_ir,
        tflite_file_name="",
    )
    metadata["execution_backend"] = "saved_model"
    metadata["saved_model_dir_name"] = str(saved_model_dir_name)
    metadata.pop("tflite_file_name", None)
    return metadata

def _extract_string_normalizer_config_from_onnx_graph(
    onnx_graph: Any,
) -> Optional[Dict[str, Any]]:
    if onnx_graph is None:
        return None
    graph = getattr(onnx_graph, "graph", None)
    if graph is None or len(list(graph.node)) != 1:
        return None
    node = list(graph.node)[0]
    if str(node.op_type) != "StringNormalizer":
        return None
    if len(list(graph.input)) == 0 or len(list(graph.output)) == 0:
        return None
    attributes = {
        str(attr.name): onnx.helper.get_attribute_value(attr)
        for attr in node.attribute
    }
    stopwords = []
    for value in list(attributes.get("stopwords", [])):
        if isinstance(value, bytes):
            stopwords.append(value.decode("utf-8"))
        else:
            stopwords.append(str(value))
    case_change_action = attributes.get("case_change_action", b"")
    locale = attributes.get("locale", b"")
    return {
        "input_name": str(node.input[0]),
        "output_name": str(node.output[0]),
        "case_change_action": (
            case_change_action.decode("utf-8")
            if isinstance(case_change_action, bytes)
            else str(case_change_action)
        ),
        "is_case_sensitive": bool(int(attributes.get("is_case_sensitive", 1))),
        "locale": locale.decode("utf-8") if isinstance(locale, bytes) else str(locale),
        "stopwords": stopwords,
    }

def export_pytorch_package_from_string_normalizer_onnx(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    onnx_graph: Any,
) -> str:
    config = _extract_string_normalizer_config_from_onnx_graph(onnx_graph)
    if config is None:
        raise ModelIRPyTorchExportError(
            "StringNormalizer fallback requires a single-op StringNormalizer ONNX graph."
        )
    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)
    metadata = {
        "schema_version": 1,
        "execution_backend": "string_normalizer",
        "name": str(model_ir.name),
        "description": str(model_ir.description),
        "inputs": [str(v) for v in model_ir.inputs],
        "outputs": [str(v) for v in model_ir.outputs],
        "tensors": {
            str(name): _serializable_tensor_meta(model_ir.tensors[str(name)])
            for name in list(model_ir.inputs) + list(model_ir.outputs)
            if str(name) in model_ir.tensors
        },
        "operators": [],
        "public_layouts": {},
        "string_normalizer": config,
    }
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)

def export_pytorch_package_from_tflite_artifact(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    tflite_file_path: str,
) -> str:
    if not os.path.exists(tflite_file_path):
        raise ModelIRPyTorchExportError(
            f"TFLite-backed PyTorch package export requires an existing float32 TFLite file. path={tflite_file_path}"
        )

    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)

    package_tflite_name = "model_float32.tflite"
    shutil.copyfile(
        str(tflite_file_path),
        os.path.join(output_folder_path, package_tflite_name),
    )
    metadata = _build_tflite_backed_metadata_payload(
        model_ir=model_ir,
        tflite_file_name=package_tflite_name,
    )
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)

def export_pytorch_package_from_saved_model_artifact(
    *,
    model_ir: ModelIR,
    output_folder_path: str,
    saved_model_path: str,
) -> str:
    if not os.path.exists(saved_model_path):
        raise ModelIRPyTorchExportError(
            f"SavedModel-backed PyTorch package export requires an existing SavedModel directory. path={saved_model_path}"
        )

    os.makedirs(output_folder_path, exist_ok=True)
    _write_generated_package_common_files(output_folder_path)
    _write_wrapper_model_file(output_folder_path)

    package_saved_model_dir = os.path.join(output_folder_path, "saved_model")
    if os.path.exists(package_saved_model_dir):
        shutil.rmtree(package_saved_model_dir)
    shutil.copytree(str(saved_model_path), package_saved_model_dir)
    metadata = _build_saved_model_backed_metadata_payload(
        model_ir=model_ir,
        saved_model_dir_name="saved_model",
    )
    metadata_path = os.path.join(output_folder_path, "metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return str(output_folder_path)

def _should_prefer_tflite_backed_package(model_ir: ModelIR) -> bool:
    op_types = [str(op.op_type) for op in model_ir.operators]
    recurrent_or_control_ops = {
        "WHILE",
        "UNIDIRECTIONAL_SEQUENCE_RNN",
        "UNIDIRECTIONAL_SEQUENCE_LSTM",
        "BIDIRECTIONAL_SEQUENCE_LSTM",
    }
    if any(op_type in recurrent_or_control_ops for op_type in op_types):
        return False
    has_length_like_input = False
    for input_name in model_ir.inputs:
        canonical = re.sub(r"[^0-9a-z]+", "_", str(input_name).lower()).strip("_")
        if canonical.endswith(("length", "lengths", "len", "lens", "seq_len", "seq_lens")):
            has_length_like_input = True
            break
    if has_length_like_input:
        return False
    if any(op_type in {"TRANSPOSE_CONV", "CONV_3D_TRANSPOSE"} for op_type in op_types):
        return True
    for op in model_ir.operators:
        if str(op.op_type) != "SOFTMAX" or len(op.inputs) == 0:
            continue
        input_tensor = model_ir.tensors.get(str(op.inputs[0]), None)
        if input_tensor is None:
            continue
        if is_channel_first_logical_layout(normalize_logical_layout(input_tensor.logical_layout)):
            return True
    conv_like_count = sum(
        1
        for op_type in op_types
        if op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
    )
    strided_slice_count = sum(1 for op_type in op_types if op_type == "STRIDED_SLICE")
    concat_count = sum(1 for op_type in op_types if op_type == "CONCATENATION")
    resize_count = sum(
        1
        for op_type in op_types
        if op_type in {"RESIZE_BILINEAR", "RESIZE_NEAREST_NEIGHBOR"}
    )
    split_count = sum(1 for op_type in op_types if op_type == "SPLIT")
    softmax_count = sum(1 for op_type in op_types if op_type == "SOFTMAX")
    nhwc_named_tensor_count = sum(
        1
        for tensor_name in model_ir.tensors.keys()
        if str(tensor_name).lower().endswith(("_nhwc", "_nwc", "_ndhwc"))
    )
    has_rank3_channel_first_output = any(
        len(list(model_ir.tensors[str(output_name)].shape)) == 3
        and normalize_logical_layout(model_ir.tensors[str(output_name)].logical_layout) == "NCW"
        for output_name in model_ir.outputs
        if str(output_name) in model_ir.tensors
    )
    if (
        has_rank3_channel_first_output
        and conv_like_count >= 20
        and strided_slice_count >= 4
        and concat_count >= 4
    ):
        return True
    if (
        conv_like_count >= 40
        and nhwc_named_tensor_count >= 40
        and (resize_count >= 4 or softmax_count >= 1 or split_count >= 1)
    ):
        return True
    if (
        conv_like_count >= 60
        and nhwc_named_tensor_count >= 80
        and resize_count >= 2
    ):
        return True
    if (
        conv_like_count >= 15
        and nhwc_named_tensor_count >= 30
        and resize_count >= 3
    ):
        return True
    return False

def _should_prefer_saved_model_backed_package(model_ir: ModelIR) -> bool:
    return _should_prefer_tflite_backed_package(model_ir)

def _read_onnx_transpose_perm(node: Any) -> Optional[List[int]]:
    if str(getattr(node, "op_type", "")) != "Transpose":
        return None
    for attr in list(getattr(node, "attribute", [])):
        if str(getattr(attr, "name", "")) != "perm":
            continue
        try:
            values = onnx.helper.get_attribute_value(attr)
        except Exception:
            return None
        try:
            return [int(v) for v in list(values)]
        except Exception:
            return None
        return None

def _is_onnx_boundary_layout_passthrough_node(
    *,
    node: Any,
    source_tensor_name: str,
) -> bool:
    passthrough_op_types = {
        "Abs",
        "Add",
        "Cast",
        "Clip",
        "Div",
        "Identity",
        "LeakyRelu",
        "Mul",
        "Relu",
        "Sigmoid",
        "Softmax",
        "Sub",
        "Tanh",
    }
    if str(getattr(node, "op_type", "")) not in passthrough_op_types:
        return False
    inputs = [str(v) for v in list(getattr(node, "input", []))]
    outputs = [str(v) for v in list(getattr(node, "output", []))]
    return len(outputs) == 1 and str(source_tensor_name) in set(inputs)

def _infer_public_layouts_from_onnx_graph(reference_onnx_graph: Any) -> Dict[str, str]:
    graph = getattr(reference_onnx_graph, "graph", None)
    if graph is None:
        return {}
    consumers: Dict[str, List[Any]] = {}
    producer_by_output: Dict[str, Any] = {}
    for node in list(graph.node):
        for output_name in list(getattr(node, "output", [])):
            producer_by_output[str(output_name)] = node
        for input_name in list(getattr(node, "input", [])):
            consumers.setdefault(str(input_name), []).append(node)

    def _walk_input_boundary(tensor_name: str, rank: int) -> Optional[str]:
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            user_nodes = consumers.get(current_tensor_name, [])
            if len(user_nodes) != 1:
                return None
            node = user_nodes[0]
            perm = _read_onnx_transpose_perm(node)
            if perm == _perm_cl_to_cf(rank):
                return channel_last_logical_layout(rank)
            if perm == _perm_cf_to_cl(rank):
                return channel_first_logical_layout(rank)
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=current_tensor_name,
            ):
                return None
            current_tensor_name = str(list(getattr(node, "output", []))[0])
        return None

    def _walk_output_boundary(tensor_name: str, rank: int) -> Optional[str]:
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            node = producer_by_output.get(current_tensor_name, None)
            if node is None:
                return None
            perm = _read_onnx_transpose_perm(node)
            if perm == _perm_cf_to_cl(rank):
                return channel_last_logical_layout(rank)
            if perm == _perm_cl_to_cf(rank):
                return channel_first_logical_layout(rank)
            inputs = [str(v) for v in list(getattr(node, "input", []))]
            if len(inputs) != 1:
                return None
            previous_tensor_name = inputs[0]
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=previous_tensor_name,
            ):
                return None
            current_tensor_name = previous_tensor_name
        return None

    public_layout_map: Dict[str, str] = {}
    for value_info in list(graph.input):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        rank = len(dims)
        if rank not in {3, 4, 5}:
            continue
        inferred_layout = _walk_input_boundary(tensor_name, rank)
        if inferred_layout is not None:
            public_layout_map[tensor_name] = inferred_layout
    for value_info in list(graph.output):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        rank = len(dims)
        if rank not in {3, 4, 5}:
            continue
        inferred_layout = _walk_output_boundary(tensor_name, rank)
        if inferred_layout is not None:
            public_layout_map[tensor_name] = inferred_layout
    return public_layout_map

def _infer_batchless_rank3_image_boundaries_from_onnx_graph(
    reference_onnx_graph: Any,
) -> Set[str]:
    graph = getattr(reference_onnx_graph, "graph", None)
    if graph is None:
        return set()
    consumers: Dict[str, List[Any]] = {}
    producer_by_output: Dict[str, Any] = {}
    for node in list(graph.node):
        for output_name in list(getattr(node, "output", [])):
            producer_by_output[str(output_name)] = node
        for input_name in list(getattr(node, "input", [])):
            consumers.setdefault(str(input_name), []).append(node)

    def _input_is_batchless_channel_first_image(tensor_name: str, rank: int) -> bool:
        if rank != 3:
            return False
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            user_nodes = consumers.get(current_tensor_name, [])
            if len(user_nodes) != 1:
                return False
            node = user_nodes[0]
            axes = _read_onnx_unsqueeze_axes(node)
            if axes == [0]:
                return True
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=current_tensor_name,
            ):
                return False
            current_tensor_name = str(list(getattr(node, "output", []))[0])
        return False

    def _output_is_batchless_channel_first_image(tensor_name: str, rank: int) -> bool:
        if rank != 3:
            return False
        current_tensor_name = str(tensor_name)
        for _ in range(4):
            node = producer_by_output.get(current_tensor_name, None)
            if node is None:
                return False
            axes = _read_onnx_squeeze_axes(node)
            if axes == [0]:
                return True
            inputs = [str(v) for v in list(getattr(node, "input", []))]
            if len(inputs) != 1:
                return False
            previous_tensor_name = inputs[0]
            if not _is_onnx_boundary_layout_passthrough_node(
                node=node,
                source_tensor_name=previous_tensor_name,
            ):
                return False
            current_tensor_name = previous_tensor_name
        return False

    boundary_names: Set[str] = set()
    for value_info in list(graph.input):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        if _input_is_batchless_channel_first_image(tensor_name, len(dims)):
            boundary_names.add(tensor_name)
    for value_info in list(graph.output):
        tensor_name = str(getattr(value_info, "name", ""))
        if tensor_name == "":
            continue
        tensor_type = getattr(value_info, "type", None)
        tensor_shape = getattr(getattr(tensor_type, "tensor_type", None), "shape", None)
        dims = list(getattr(tensor_shape, "dim", [])) if tensor_shape is not None else []
        if _output_is_batchless_channel_first_image(tensor_name, len(dims)):
            boundary_names.add(tensor_name)
    return boundary_names

def _merge_reference_public_boundary_metadata(
    *,
    imported_model_ir: ModelIR,
    reference_model_ir: Optional[ModelIR],
    reference_onnx_graph: Optional[Any] = None,
) -> None:
    if reference_model_ir is None:
        return
    boundary_shape_map = reference_model_ir.metadata.get("onnx_boundary_shape_signature_map", {})
    if not isinstance(boundary_shape_map, dict):
        boundary_shape_map = {}
    public_layout_map = reference_model_ir.metadata.get("onnx_public_layout_map", {})
    if not isinstance(public_layout_map, dict):
        public_layout_map = {}
    onnx_graph_public_layout_map = (
        _infer_public_layouts_from_onnx_graph(reference_onnx_graph)
        if reference_onnx_graph is not None
        else {}
    )
    batchless_rank3_boundary_names = (
        _infer_batchless_rank3_image_boundaries_from_onnx_graph(reference_onnx_graph)
        if reference_onnx_graph is not None
        else set()
    )
    recurrent_public_boundary_context = any(
        token in str(op.op_type)
        for op in reference_model_ir.operators
        for token in ("GRU", "LSTM", "RNN")
    )
    if not recurrent_public_boundary_context and reference_onnx_graph is not None:
        graph = getattr(reference_onnx_graph, "graph", None)
        if graph is not None:
            recurrent_public_boundary_context = any(
                str(node.op_type) in {"GRU", "LSTM", "RNN"}
                for node in list(graph.node)
            )
    imported_model_ir.inputs = [str(v) for v in list(reference_model_ir.inputs)]
    imported_model_ir.outputs = [str(v) for v in list(reference_model_ir.outputs)]

    desired_public_layout_map: Dict[str, str] = {}
    desired_public_shape_map: Dict[str, List[int]] = {}
    for tensor_name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs):
        ref_tensor = reference_model_ir.tensors.get(str(tensor_name), None)
        if ref_tensor is None:
            continue
        desired_public_shape_map[str(tensor_name)] = [
            int(v) for v in list(ref_tensor.shape_signature or ref_tensor.shape)
        ]
        desired_layout = normalize_logical_layout(
            onnx_graph_public_layout_map.get(
                str(tensor_name),
                public_layout_map.get(str(tensor_name), ref_tensor.logical_layout),
            )
        )
        if recurrent_public_boundary_context and len(list(ref_tensor.shape)) == 3:
            desired_layout = "NWC"
        desired_public_layout_map[str(tensor_name)] = desired_layout

    _ensure_public_boundary_layout_bridges(
        model_ir=imported_model_ir,
        desired_public_shape_map=desired_public_shape_map,
        desired_public_layout_map=desired_public_layout_map,
    )

    for tensor_name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs):
        ref_tensor = reference_model_ir.tensors.get(str(tensor_name), None)
        imported_tensor = imported_model_ir.tensors.get(str(tensor_name), None)
        if ref_tensor is None or imported_tensor is None:
            continue
        imported_tensor.shape_signature = [int(v) for v in list(ref_tensor.shape_signature or ref_tensor.shape)]
        imported_tensor.logical_layout = desired_public_layout_map.get(
            str(tensor_name),
            normalize_logical_layout(ref_tensor.logical_layout),
        )
    imported_model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        str(name): [int(v) for v in list(boundary_shape_map.get(str(name), reference_model_ir.tensors[str(name)].shape))]
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }
    imported_model_ir.metadata["onnx_public_layout_map"] = {
        str(name): (
            "NWC"
            if recurrent_public_boundary_context and len(list(reference_model_ir.tensors[str(name)].shape)) == 3
            else normalize_logical_layout(
                onnx_graph_public_layout_map.get(
                    str(name),
                    public_layout_map.get(
                        str(name),
                        reference_model_ir.tensors[str(name)].logical_layout,
                    ),
                )
            )
        )
        for name in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
        if str(name) in reference_model_ir.tensors
    }
    imported_model_ir.metadata["batchless_rank3_public_boundary_names"] = sorted(
        str(name)
        for name in list(batchless_rank3_boundary_names)
        if str(name) in list(imported_model_ir.inputs) + list(imported_model_ir.outputs)
    )

