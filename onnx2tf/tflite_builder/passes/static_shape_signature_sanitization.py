from __future__ import annotations

from typing import Dict, List

import numpy as np

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _build_tensor_producer_map,
    _is_fully_known_positive_shape,
    _read_const_ints_from_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR


def sanitize_static_shape_signature_consistency(
    model_ir: ModelIR,
) -> Dict[str, int]:
    """Keep static runtime shapes and serialized signatures consistent."""

    fixed = 0
    preserved_dynamic_boundary = 0
    preserved_dynamic_leading_axis = 0
    preserved_dynamic_lineage = 0
    dynamic_boundary_names = set()
    for key in (
        "onnx_dynamic_input_tensor_names",
        "onnx_dynamic_output_tensor_names",
    ):
        names = model_ir.metadata.get(key, [])
        if isinstance(names, list):
            dynamic_boundary_names.update(
                str(value) for value in names if str(value) != ""
            )
    dynamic_boundary_signature_map = model_ir.metadata.get(
        "dynamic_boundary_shape_signature_map", {}
    )
    if not isinstance(dynamic_boundary_signature_map, dict):
        dynamic_boundary_signature_map = {}
    graph_output_names = set(str(value) for value in list(model_ir.outputs))
    producer_map = _build_tensor_producer_map(model_ir)
    dynamic_lineage_root_names = set(dynamic_boundary_names)
    for tensor_name, producer_idx in producer_map.items():
        if int(producer_idx) < 0 or int(producer_idx) >= len(
            model_ir.operators
        ):
            continue
        producer_op = model_ir.operators[int(producer_idx)]
        producer_op_type = str(producer_op.op_type)
        tensor = model_ir.tensors.get(str(tensor_name), None)
        if tensor is None:
            continue
        signature = (
            [int(value) for value in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else [int(value) for value in list(tensor.shape)]
        )
        if (
            producer_op_type == "WHERE"
            and len(signature) >= 1
            and int(signature[0]) < 0
        ):
            dynamic_lineage_root_names.add(str(tensor_name))
            continue
        if producer_op_type == "RANGE" and any(
            int(value) < 0 for value in signature
        ):
            dynamic_lineage_root_names.add(str(tensor_name))
            continue
        if producer_op_type == "RESHAPE":
            reshape_target: List[int] = []
            try:
                reshape_target = [
                    int(value)
                    for value in np.asarray(
                        producer_op.options.get("newShape", [])
                    )
                    .reshape(-1)
                    .tolist()
                ]
            except Exception:
                reshape_target = []
            if len(reshape_target) == 0 and len(producer_op.inputs) >= 2:
                shape_tensor = model_ir.tensors.get(
                    str(producer_op.inputs[1]), None
                )
                shape_values = _read_const_ints_from_tensor(shape_tensor)
                if shape_values is not None:
                    reshape_target = [
                        int(value) for value in list(shape_values)
                    ]
            if any(int(value) < 0 for value in reshape_target):
                dynamic_lineage_root_names.add(str(tensor_name))
                continue
        if producer_op_type == "TOPK_V2" and any(
            int(value) < 0 for value in signature
        ):
            dynamic_lineage_root_names.add(str(tensor_name))
    dynamic_lineage_cache: Dict[str, bool] = {}
    dynamic_lineage_visiting: set[str] = set()

    def _has_dynamic_boundary_lineage(tensor_name: str) -> bool:
        key = str(tensor_name)
        cached = dynamic_lineage_cache.get(key, None)
        if cached is not None:
            return bool(cached)
        if key in dynamic_lineage_root_names:
            dynamic_lineage_cache[key] = True
            return True
        if key in dynamic_lineage_visiting:
            dynamic_lineage_cache[key] = False
            return False
        tensor = model_ir.tensors.get(key, None)
        if tensor is None:
            dynamic_lineage_cache[key] = False
            return False
        if tensor.data is not None:
            dynamic_lineage_cache[key] = False
            return False
        producer_idx = producer_map.get(key, None)
        if producer_idx is None:
            dynamic_lineage_cache[key] = False
            return False
        dynamic_lineage_visiting.add(key)
        producer = model_ir.operators[int(producer_idx)]
        has_dynamic_ancestor = False
        for input_name in producer.inputs:
            parent_name = str(input_name)
            if parent_name == "":
                continue
            if _has_dynamic_boundary_lineage(parent_name):
                has_dynamic_ancestor = True
                break
        dynamic_lineage_visiting.discard(key)
        dynamic_lineage_cache[key] = bool(has_dynamic_ancestor)
        return bool(has_dynamic_ancestor)

    for tensor in model_ir.tensors.values():
        if tensor.shape is None:
            continue
        shape = [int(value) for value in list(tensor.shape)]
        if len(shape) == 0:
            if tensor.shape_signature is None:
                tensor.shape_signature = []
                fixed += 1
            continue
        if not _is_fully_known_positive_shape(shape):
            continue

        boundary_signature = dynamic_boundary_signature_map.get(
            str(tensor.name), None
        )
        if isinstance(boundary_signature, list) and len(
            boundary_signature
        ) == len(shape):
            normalized_boundary_signature = [
                int(-1 if int(value) < 0 else shape[index])
                for index, value in enumerate(boundary_signature)
            ]
            if any(
                int(value) < 0 for value in normalized_boundary_signature
            ):
                if tensor.shape_signature != normalized_boundary_signature:
                    tensor.shape_signature = list(
                        normalized_boundary_signature
                    )
                    fixed += 1
                preserved_dynamic_boundary += 1
                continue

        signature = (
            [int(value) for value in list(tensor.shape_signature)]
            if tensor.shape_signature is not None
            else None
        )
        if signature is None:
            tensor.shape_signature = [int(value) for value in list(shape)]
            fixed += 1
            continue
        if len(signature) != len(shape):
            tensor.shape_signature = [int(value) for value in list(shape)]
            fixed += 1
            continue
        if any(int(value) < 0 for value in signature):
            negative_axes = [
                int(index)
                for index, value in enumerate(signature)
                if int(value) < 0
            ]
            trailing_axes_static = all(
                int(signature[index]) > 0
                for index in range(1, len(signature))
            )
            producer_idx = producer_map.get(str(tensor.name), None)
            producer_op_type = (
                str(model_ir.operators[int(producer_idx)].op_type)
                if producer_idx is not None
                and int(producer_idx) < len(model_ir.operators)
                else ""
            )
            if str(tensor.name) in dynamic_lineage_root_names:
                preserved_dynamic_boundary += 1
                continue
            if (
                producer_op_type == "WHERE"
                and negative_axes == [0]
                and trailing_axes_static
            ):
                preserved_dynamic_leading_axis += 1
                continue
            has_dynamic_lineage = _has_dynamic_boundary_lineage(
                str(tensor.name)
            )
            if negative_axes == [0] and trailing_axes_static:
                if str(tensor.name) in graph_output_names:
                    preserved_dynamic_leading_axis += 1
                    continue
                if has_dynamic_lineage:
                    preserved_dynamic_leading_axis += 1
                    continue
            if has_dynamic_lineage:
                if negative_axes == [0] and trailing_axes_static:
                    preserved_dynamic_leading_axis += 1
                else:
                    preserved_dynamic_lineage += 1
                continue
            tensor.shape_signature = [int(value) for value in list(shape)]
            fixed += 1
            continue
        if signature != shape:
            tensor.shape_signature = [int(value) for value in list(shape)]
            fixed += 1

    return {
        "sanitized_static_shape_signature_consistency": int(fixed),
        "preserved_dynamic_boundary_shape_signature": int(
            preserved_dynamic_boundary
        ),
        "preserved_dynamic_leading_axis_shape_signature": int(
            preserved_dynamic_leading_axis
        ),
        "preserved_dynamic_lineage_shape_signature": int(
            preserved_dynamic_lineage
        ),
    }
