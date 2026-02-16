#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnx

from onnx2tf import gs
from onnx2tf.tflite_builder.accuracy_evaluator import (
    _MetricAccumulator,
    _adapt_input_layout_for_tflite_input,
    _align_output_layout_for_compare,
    _build_tflite_detail_map,
    _collect_onnx_input_specs,
    _create_tflite_interpreter,
    _dequantize_tflite_output,
    _is_integer_or_bool_dtype,
    _normalize_tensor_name,
    _quantize_for_tflite_input,
)
from onnx2tf.utils.common_functions import dummy_onnx_inference


def _default_output_paths(onnx_path: str, output_dir: str) -> Tuple[str, str]:
    model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
    json_path = os.path.join(output_dir, f"{model_stem}_op_error_report.json")
    csv_path = os.path.join(output_dir, f"{model_stem}_op_error_report.csv")
    return json_path, csv_path


def _get_onnx_initializer_map(
    onnx_graph: onnx.ModelProto,
) -> Dict[str, np.ndarray]:
    initializer_map: Dict[str, np.ndarray] = {}
    for initializer in onnx_graph.graph.initializer:
        initializer_map[str(initializer.name)] = np.asarray(
            onnx.numpy_helper.to_array(initializer)
        )
    return initializer_map


def _read_onnx_attr_int(node: onnx.NodeProto, attr_name: str) -> Optional[int]:
    for attr in node.attribute:
        if str(attr.name) == str(attr_name):
            try:
                return int(onnx.helper.get_attribute_value(attr))
            except Exception:
                return None
    return None


def _read_initializer_array(
    initializer_map: Dict[str, np.ndarray],
    tensor_name: str,
) -> Optional[np.ndarray]:
    if str(tensor_name) == "":
        return None
    array = initializer_map.get(str(tensor_name), None)
    if array is None:
        return None
    return np.asarray(array)


def _normalize_scale_zero_for_onnx(
    *,
    scale_array: Optional[np.ndarray],
    zero_point_array: Optional[np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if scale_array is None:
        return None
    scales = np.asarray(scale_array, dtype=np.float32).reshape(-1)
    if scales.size == 0:
        return None
    if zero_point_array is None:
        zeros = np.zeros_like(scales, dtype=np.float32)
    else:
        zeros = np.asarray(zero_point_array, dtype=np.float32).reshape(-1)
        if zeros.size == 0:
            zeros = np.zeros_like(scales, dtype=np.float32)
    if zeros.size == 1 and scales.size > 1:
        zeros = np.full((int(scales.size),), float(zeros[0]), dtype=np.float32)
    if scales.size == 1 and zeros.size > 1:
        scales = np.full((int(zeros.size),), float(scales[0]), dtype=np.float32)
    if scales.size != zeros.size:
        return None
    return scales, zeros


def _build_onnx_quant_param_map(
    onnx_graph: onnx.ModelProto,
) -> Dict[str, Dict[str, Any]]:
    initializer_map = _get_onnx_initializer_map(onnx_graph)
    quant_map: Dict[str, Dict[str, Any]] = {}

    def _upsert_tensor_quant_param(
        *,
        tensor_name: str,
        scale_name: str,
        zero_name: str = "",
        axis: Optional[int] = None,
    ) -> None:
        if str(tensor_name) == "":
            return
        scale_array = _read_initializer_array(initializer_map, str(scale_name))
        zero_array = _read_initializer_array(initializer_map, str(zero_name))
        normalized = _normalize_scale_zero_for_onnx(
            scale_array=scale_array,
            zero_point_array=zero_array,
        )
        if normalized is None:
            return
        scales, zeros = normalized
        next_entry = {
            "scale": np.asarray(scales, dtype=np.float32),
            "zero_point": np.asarray(zeros, dtype=np.float32),
            "axis": axis,
        }
        prev = quant_map.get(str(tensor_name), None)
        if prev is not None:
            prev_scales = np.asarray(prev.get("scale", []), dtype=np.float32).reshape(-1)
            prev_zeros = np.asarray(prev.get("zero_point", []), dtype=np.float32).reshape(-1)
            prev_axis = prev.get("axis", None)
            if (
                prev_scales.shape == next_entry["scale"].shape
                and np.array_equal(prev_scales, next_entry["scale"])
                and prev_zeros.shape == next_entry["zero_point"].shape
                and np.array_equal(prev_zeros, next_entry["zero_point"])
                and prev_axis == next_entry["axis"]
            ):
                return
        quant_map[str(tensor_name)] = next_entry

    for node in onnx_graph.graph.node:
        op_type = str(node.op_type)
        outputs = [str(v) for v in node.output if str(v) != ""]

        if op_type in {"QuantizeLinear", "DequantizeLinear"}:
            if len(node.input) < 2:
                continue
            scale_name = str(node.input[1])
            zero_name = str(node.input[2]) if len(node.input) >= 3 else ""
            axis = _read_onnx_attr_int(node, "axis")
            if axis is None and op_type == "DequantizeLinear":
                axis = 1
            if op_type == "DequantizeLinear" and len(node.input) >= 1:
                _upsert_tensor_quant_param(
                    tensor_name=str(node.input[0]),
                    scale_name=scale_name,
                    zero_name=zero_name,
                    axis=axis,
                )
            for output_name in outputs:
                _upsert_tensor_quant_param(
                    tensor_name=output_name,
                    scale_name=scale_name,
                    zero_name=zero_name,
                    axis=axis,
                )
        elif op_type == "QLinearConcat":
            # QLinearConcat: y_scale/y_zero_point are the first two inputs.
            if len(node.input) < 2:
                continue
            y_scale_name = str(node.input[0])
            y_zero_name = str(node.input[1])
            for output_name in outputs:
                _upsert_tensor_quant_param(
                    tensor_name=output_name,
                    scale_name=y_scale_name,
                    zero_name=y_zero_name,
                )
            # Follow ONNX spec input triplets: x_i, x_i_scale, x_i_zero_point
            triplet_start = 2
            while triplet_start + 2 < len(node.input):
                _upsert_tensor_quant_param(
                    tensor_name=str(node.input[triplet_start]),
                    scale_name=str(node.input[triplet_start + 1]),
                    zero_name=str(node.input[triplet_start + 2]),
                )
                triplet_start += 3
        elif op_type in {"QLinearConv", "QLinearMatMul", "QLinearAdd", "QLinearMul"}:
            # x, x_scale, x_zp, w/b, w/b_scale, w/b_zp, y_scale, y_zp, (optional bias for conv)
            if len(node.input) < 8:
                continue
            for output_name in outputs:
                _upsert_tensor_quant_param(
                    tensor_name=output_name,
                    scale_name=str(node.input[6]),
                    zero_name=str(node.input[7]),
                )
            _upsert_tensor_quant_param(
                tensor_name=str(node.input[0]),
                scale_name=str(node.input[1]),
                zero_name=str(node.input[2]),
            )
            _upsert_tensor_quant_param(
                tensor_name=str(node.input[3]),
                scale_name=str(node.input[4]),
                zero_name=str(node.input[5]),
            )
        elif op_type in {"QLinearLeakyRelu", "QLinearSigmoid", "QLinearSoftmax", "QLinearAveragePool", "QLinearGlobalAveragePool"}:
            # x, x_scale, x_zp, y_scale, y_zp
            if len(node.input) < 5:
                continue
            for output_name in outputs:
                _upsert_tensor_quant_param(
                    tensor_name=output_name,
                    scale_name=str(node.input[3]),
                    zero_name=str(node.input[4]),
                )
            _upsert_tensor_quant_param(
                tensor_name=str(node.input[0]),
                scale_name=str(node.input[1]),
                zero_name=str(node.input[2]),
            )
        elif op_type.startswith("QLinear"):
            # Fallback for other QLinear ops.
            if len(node.input) < 2:
                continue
            for output_name in outputs:
                _upsert_tensor_quant_param(
                    tensor_name=output_name,
                    scale_name=str(node.input[-2]),
                    zero_name=str(node.input[-1]),
                )
            if len(node.input) >= 3:
                _upsert_tensor_quant_param(
                    tensor_name=str(node.input[0]),
                    scale_name=str(node.input[1]),
                    zero_name=str(node.input[2]),
                )
        else:
            continue

    passthrough_ops = {
        "Transpose",
        "Reshape",
        "Identity",
        "Squeeze",
        "Unsqueeze",
        "Flatten",
        "Slice",
        "Pad",
    }
    changed = True
    while changed:
        changed = False
        for node in onnx_graph.graph.node:
            op_type = str(node.op_type)
            if op_type not in passthrough_ops:
                continue
            if len(node.input) == 0:
                continue
            src_name = str(node.input[0])
            src_q = quant_map.get(src_name, None)
            if src_q is None:
                continue

            scales = np.asarray(src_q.get("scale", []), dtype=np.float32).reshape(-1)
            zeros = np.asarray(src_q.get("zero_point", []), dtype=np.float32).reshape(-1)
            axis = src_q.get("axis", None)

            if scales.size > 1 and op_type != "Transpose":
                # Conservative: per-channel axis remapping is non-trivial for non-transpose ops.
                continue
            if scales.size > 1 and op_type == "Transpose":
                if len(node.input) < 2:
                    continue
                perm = _read_initializer_array(initializer_map, str(node.input[1]))
                if perm is None:
                    continue
                perm_list = [int(v) for v in np.asarray(perm).reshape(-1).tolist()]
                if axis is None:
                    continue
                old_axis = int(axis)
                if old_axis < 0:
                    old_axis += int(len(perm_list))
                if old_axis < 0 or old_axis >= int(len(perm_list)):
                    continue
                new_axis = None
                for out_dim, in_dim in enumerate(perm_list):
                    if int(in_dim) == int(old_axis):
                        new_axis = int(out_dim)
                        break
                if new_axis is None:
                    continue
                axis = int(new_axis)

            for output_name in node.output:
                out_name = str(output_name)
                if out_name == "":
                    continue
                prev = quant_map.get(out_name, None)
                next_entry = {
                    "scale": np.asarray(scales, dtype=np.float32),
                    "zero_point": np.asarray(zeros, dtype=np.float32),
                    "axis": axis,
                }
                if prev is not None:
                    prev_scales = np.asarray(prev.get("scale", []), dtype=np.float32).reshape(-1)
                    prev_zeros = np.asarray(prev.get("zero_point", []), dtype=np.float32).reshape(-1)
                    prev_axis = prev.get("axis", None)
                    if (
                        prev_scales.shape == next_entry["scale"].shape
                        and np.array_equal(prev_scales, next_entry["scale"])
                        and prev_zeros.shape == next_entry["zero_point"].shape
                        and np.array_equal(prev_zeros, next_entry["zero_point"])
                        and prev_axis == next_entry["axis"]
                    ):
                        continue
                quant_map[out_name] = next_entry
                changed = True

    return quant_map


def _dequantize_onnx_output(
    onnx_tensor: np.ndarray,
    quant_param: Optional[Dict[str, Any]],
) -> np.ndarray:
    tensor = np.asarray(onnx_tensor)
    if not _is_integer_or_bool_dtype(np.dtype(tensor.dtype)):
        return tensor
    if quant_param is None:
        return tensor.astype(np.float32)

    scales = np.asarray(quant_param.get("scale", []), dtype=np.float32).reshape(-1)
    zeros = np.asarray(quant_param.get("zero_point", []), dtype=np.float32).reshape(-1)
    if scales.size == 0:
        return tensor.astype(np.float32)
    if zeros.size == 0:
        zeros = np.zeros_like(scales, dtype=np.float32)
    if scales.size == 1 and zeros.size > 1:
        scales = np.full((int(zeros.size),), float(scales[0]), dtype=np.float32)
    if zeros.size == 1 and scales.size > 1:
        zeros = np.full((int(scales.size),), float(zeros[0]), dtype=np.float32)
    if scales.size != zeros.size:
        return tensor.astype(np.float32)

    tensor_f = tensor.astype(np.float32)
    if scales.size == 1:
        return (tensor_f - float(zeros[0])) * float(scales[0])

    axis = quant_param.get("axis", None)
    if axis is None:
        axis = 1
    axis = int(axis)
    if axis < 0:
        axis += int(tensor_f.ndim)
    if axis < 0 or axis >= int(tensor_f.ndim):
        return tensor.astype(np.float32)
    if int(tensor_f.shape[axis]) != int(scales.size):
        return tensor.astype(np.float32)

    reshape = [1 for _ in range(int(tensor_f.ndim))]
    reshape[int(axis)] = int(scales.size)
    scale_b = scales.reshape(reshape)
    zero_b = zeros.reshape(reshape)
    return (tensor_f - zero_b) * scale_b


def _build_onnx_output_meta(
    onnx_graph: onnx.ModelProto,
) -> Dict[str, Dict[str, Any]]:
    output_meta: Dict[str, Dict[str, Any]] = {}
    for identifier, node in enumerate(onnx_graph.graph.node, start=1):
        for output_name in node.output:
            if not output_name:
                continue
            output_meta[output_name] = {
                "identifier": int(identifier),
                "onnx_op_name": str(node.name),
                "onnx_op_type": str(node.op_type),
            }
    return output_meta


def _filter_dummy_onnx_inferable_outputs(
    *,
    onnx_graph: onnx.ModelProto,
    candidate_names: List[str],
) -> List[str]:
    # dummy_onnx_inference only exposes outputs with known dtype in GS graph.
    gs_graph = gs.import_onnx(onnx_graph)
    dtype_by_name: Dict[str, Any] = {}
    for node in gs_graph.nodes:
        for node_output in node.outputs:
            dtype_by_name[node_output.name] = node_output.dtype
    return [
        name
        for name in candidate_names
        if name in dtype_by_name and dtype_by_name[name] is not None
    ]


def _build_tflite_base_detail_map(
    details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    base_map: Dict[str, Dict[str, Any]] = {}
    for detail in details:
        detail_name = str(detail.get("name", ""))
        if not detail_name:
            continue
        base_name = _normalize_tensor_name(detail_name)
        if base_name not in base_map:
            base_map[base_name] = detail
    return base_map


def _build_tflite_exact_detail_map(
    details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    return {str(detail.get("name", "")): detail for detail in details if str(detail.get("name", "")) != ""}


def _default_tensor_correspondence_report_path(onnx_path: str, output_dir: str) -> str:
    model_stem = os.path.splitext(os.path.basename(onnx_path))[0]
    return os.path.join(output_dir, f"{model_stem}_tensor_correspondence_report.json")


def _load_correspondence_map(
    report_path: str,
) -> Dict[str, str]:
    if not report_path or not os.path.exists(report_path):
        return {}
    with open(report_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    records = payload.get("records", [])
    if not isinstance(records, list):
        return {}
    mapped: Dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        status = str(record.get("status", ""))
        if status not in {"direct", "rewritten", "inferred"}:
            continue
        if not bool(record.get("exists_in_final_model", False)):
            continue
        onnx_output_name = str(record.get("onnx_output_name", ""))
        resolved_name = str(record.get("resolved_tflite_tensor_name", ""))
        if onnx_output_name == "" or resolved_name == "":
            continue
        mapped[onnx_output_name] = resolved_name
    return mapped


def _get_onnx_eval_outputs(
    *,
    onnx_graph: onnx.ModelProto,
    target_output_names: List[str],
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    onnx_input_datas_for_validation: Dict[str, np.ndarray] = {}
    outputs = dummy_onnx_inference(
        onnx_graph=onnx_graph,
        output_names=target_output_names,
        input_datas_for_validation=onnx_input_datas_for_validation,
    )
    onnx_outputs = {
        output_name: np.asarray(output_value)
        for output_name, output_value in zip(target_output_names, outputs)
    }
    return onnx_outputs, onnx_input_datas_for_validation


def _invoke_tflite_with_onnx_inputs(
    *,
    onnx_graph: onnx.ModelProto,
    interpreter: Any,
    onnx_input_datas_for_validation: Dict[str, np.ndarray],
) -> None:
    input_specs = _collect_onnx_input_specs(onnx_graph)
    onnx_input_names = [name for name, _, _ in input_specs]
    tflite_input_details = interpreter.get_input_details()
    tflite_input_map = _build_tflite_detail_map(
        onnx_names=onnx_input_names,
        tflite_details=tflite_input_details,
    )

    for onnx_input_name in onnx_input_names:
        input_data = onnx_input_datas_for_validation.get(onnx_input_name)
        if input_data is None:
            input_data = onnx_input_datas_for_validation.get(
                _normalize_tensor_name(onnx_input_name)
            )
        if input_data is None:
            raise ValueError(
                "Failed to find ONNX evaluation input tensor. "
                f"input_name={onnx_input_name}"
            )
        detail = tflite_input_map[onnx_input_name]
        adapted = _adapt_input_layout_for_tflite_input(np.asarray(input_data), detail)
        quantized = _quantize_for_tflite_input(adapted, detail)
        interpreter.set_tensor(detail["index"], quantized)
    interpreter.invoke()


def _compare_tensor_pair(
    *,
    onnx_tensor_name: str,
    onnx_tensor: np.ndarray,
    onnx_quant_param_map: Dict[str, Dict[str, Any]],
    tflite_tensor_raw: np.ndarray,
    tflite_detail: Dict[str, Any],
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    tflite_for_compare = np.asarray(tflite_tensor_raw)
    if _is_integer_or_bool_dtype(np.dtype(tflite_for_compare.dtype)):
        tflite_for_compare = _dequantize_tflite_output(
            tflite_for_compare,
            tflite_detail,
        )
    onnx_for_compare = _dequantize_onnx_output(
        np.asarray(onnx_tensor),
        onnx_quant_param_map.get(str(onnx_tensor_name), None),
    )

    aligned_tflite, align_mode, align_perm = _align_output_layout_for_compare(
        onnx_output=onnx_for_compare,
        tflite_output=tflite_for_compare,
        rtol=rtol,
        atol=atol,
    )
    metrics = _MetricAccumulator()
    metrics.update(onnx_for_compare, aligned_tflite)
    metric_values = metrics.to_dict()
    allclose = bool(
        np.allclose(
            onnx_for_compare,
            aligned_tflite,
            rtol=rtol,
            atol=atol,
            equal_nan=True,
        )
    )
    return {
        "status": "compared",
        "allclose": allclose,
        "alignment_mode": align_mode,
        "alignment_perm": [] if align_perm is None else [int(v) for v in align_perm],
        "max_abs": float(metric_values["max_abs"]),
        "mean_abs": float(metric_values["mean_abs"]),
        "rmse": float(metric_values["rmse"]),
        "cosine_similarity": float(metric_values["cosine_similarity"]),
    }


def _write_csv(output_csv_path: str, rows: List[Dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "identifier",
        "onnx_op_type",
        "onnx_op_name",
        "tensor_name",
        "tflite_tensor_name",
        "mapped_tflite_tensor_name",
        "mapping_source",
        "status",
        "reason",
        "onnx_shape",
        "tflite_shape_raw",
        "onnx_dtype",
        "tflite_dtype_raw",
        "allclose",
        "alignment_mode",
        "alignment_perm",
        "max_abs",
        "mean_abs",
        "rmse",
        "cosine_similarity",
    ]
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "rank": row.get("rank", ""),
                    "identifier": row.get("identifier", ""),
                    "onnx_op_type": row.get("onnx_op_type", ""),
                    "onnx_op_name": row.get("onnx_op_name", ""),
                    "tensor_name": row.get("tensor_name", ""),
                    "tflite_tensor_name": row.get("tflite_tensor_name", ""),
                    "mapped_tflite_tensor_name": row.get("mapped_tflite_tensor_name", ""),
                    "mapping_source": row.get("mapping_source", ""),
                    "status": row.get("status", ""),
                    "reason": row.get("reason", ""),
                    "onnx_shape": row.get("onnx_shape", ""),
                    "tflite_shape_raw": row.get("tflite_shape_raw", ""),
                    "onnx_dtype": row.get("onnx_dtype", ""),
                    "tflite_dtype_raw": row.get("tflite_dtype_raw", ""),
                    "allclose": row.get("allclose", ""),
                    "alignment_mode": row.get("alignment_mode", ""),
                    "alignment_perm": row.get("alignment_perm", ""),
                    "max_abs": row.get("max_abs", ""),
                    "mean_abs": row.get("mean_abs", ""),
                    "rmse": row.get("rmse", ""),
                    "cosine_similarity": row.get("cosine_similarity", ""),
                }
            )


def generate_op_error_report(
    *,
    onnx_path: str,
    tflite_path: str,
    output_dir: str = "saved_model",
    output_json: Optional[str] = None,
    output_csv: Optional[str] = None,
    tensor_correspondence_report: Optional[str] = None,
    top: int = 30,
    rtol: float = 0.0,
    atol: float = 1.0e-4,
    verbose: bool = True,
) -> Dict[str, Any]:
    if not os.path.exists(onnx_path):
        raise FileNotFoundError(f"ONNX file not found. path={onnx_path}")
    if not os.path.exists(tflite_path):
        raise FileNotFoundError(f"TFLite file not found. path={tflite_path}")
    if int(top) <= 0:
        raise ValueError(f"top must be > 0. got={top}")
    if float(rtol) < 0.0:
        raise ValueError(f"rtol must be >= 0. got={rtol}")
    if float(atol) < 0.0:
        raise ValueError(f"atol must be >= 0. got={atol}")

    default_json, default_csv = _default_output_paths(onnx_path, output_dir)
    output_json_path = output_json if output_json else default_json
    output_csv_path = output_csv if output_csv else default_csv
    default_correspondence_path = _default_tensor_correspondence_report_path(
        onnx_path,
        output_dir,
    )
    correspondence_path = (
        tensor_correspondence_report
        if tensor_correspondence_report is not None
        else default_correspondence_path
    )
    correspondence_map = _load_correspondence_map(correspondence_path)

    onnx_graph = onnx.load(onnx_path)
    onnx_quant_param_map = _build_onnx_quant_param_map(onnx_graph)
    onnx_output_meta = _build_onnx_output_meta(onnx_graph)
    onnx_tensor_names = list(onnx_output_meta.keys())

    interpreter = _create_tflite_interpreter(tflite_path)
    interpreter.allocate_tensors()
    tflite_tensor_details = interpreter.get_tensor_details()
    tflite_base_detail_map = _build_tflite_base_detail_map(tflite_tensor_details)
    tflite_exact_detail_map = _build_tflite_exact_detail_map(tflite_tensor_details)

    inferable_output_names = _filter_dummy_onnx_inferable_outputs(
        onnx_graph=onnx_graph,
        candidate_names=onnx_tensor_names,
    )
    inferable_output_name_set = set(inferable_output_names)

    target_output_names: List[str] = []
    for onnx_tensor_name in inferable_output_names:
        if onnx_tensor_name in correspondence_map:
            mapped_name = correspondence_map[onnx_tensor_name]
            if (
                mapped_name in tflite_exact_detail_map
                or _normalize_tensor_name(mapped_name) in tflite_base_detail_map
            ):
                target_output_names.append(onnx_tensor_name)
                continue
        base_name = _normalize_tensor_name(onnx_tensor_name)
        if base_name in tflite_base_detail_map:
            target_output_names.append(onnx_tensor_name)

    if len(target_output_names) == 0:
        raise RuntimeError(
            "No common intermediate tensor names between ONNX and TFLite were found."
        )

    onnx_outputs, onnx_input_datas_for_validation = _get_onnx_eval_outputs(
        onnx_graph=onnx_graph,
        target_output_names=target_output_names,
    )
    _invoke_tflite_with_onnx_inputs(
        onnx_graph=onnx_graph,
        interpreter=interpreter,
        onnx_input_datas_for_validation=onnx_input_datas_for_validation,
    )

    records: List[Dict[str, Any]] = []
    for tensor_name in onnx_tensor_names:
        meta = onnx_output_meta[tensor_name]
        mapped_tflite_name = correspondence_map.get(tensor_name, "")
        detail = None
        mapping_source = "name_match"
        if mapped_tflite_name != "":
            detail = tflite_exact_detail_map.get(mapped_tflite_name, None)
            if detail is None:
                detail = tflite_base_detail_map.get(
                    _normalize_tensor_name(mapped_tflite_name),
                    None,
                )
            if detail is not None:
                mapping_source = "tensor_correspondence_report"
        if detail is None:
            base_name = _normalize_tensor_name(tensor_name)
            detail = tflite_base_detail_map.get(base_name)
        record: Dict[str, Any] = {
            "identifier": int(meta["identifier"]),
            "onnx_op_name": str(meta["onnx_op_name"]),
            "onnx_op_type": str(meta["onnx_op_type"]),
            "tensor_name": tensor_name,
            "tflite_tensor_name": str(detail["name"]) if detail is not None else "",
            "mapped_tflite_tensor_name": mapped_tflite_name,
            "mapping_source": mapping_source if detail is not None else "",
            "status": "skipped",
            "reason": "",
            "onnx_shape": (
                list(np.asarray(onnx_outputs[tensor_name]).shape)
                if tensor_name in onnx_outputs
                else None
            ),
            "tflite_shape_raw": None,
            "onnx_dtype": (
                str(np.asarray(onnx_outputs[tensor_name]).dtype)
                if tensor_name in onnx_outputs
                else ""
            ),
            "tflite_dtype_raw": "",
            "allclose": None,
            "alignment_mode": "",
            "alignment_perm": [],
            "max_abs": None,
            "mean_abs": None,
            "rmse": None,
            "cosine_similarity": None,
        }

        if tensor_name not in inferable_output_name_set:
            record["reason"] = "onnx_output_not_inferable"
            records.append(record)
            continue

        if detail is None:
            record["reason"] = "no_matching_tflite_tensor"
            records.append(record)
            continue

        try:
            tflite_raw = np.asarray(interpreter.get_tensor(detail["index"]))
        except Exception as ex:
            record["reason"] = f"tflite_get_tensor_failed: {ex}"
            records.append(record)
            continue

        record["tflite_shape_raw"] = list(tflite_raw.shape)
        record["tflite_dtype_raw"] = str(tflite_raw.dtype)

        try:
            compare_result = _compare_tensor_pair(
                onnx_tensor_name=tensor_name,
                onnx_tensor=np.asarray(onnx_outputs[tensor_name]),
                onnx_quant_param_map=onnx_quant_param_map,
                tflite_tensor_raw=tflite_raw,
                tflite_detail=detail,
                rtol=float(rtol),
                atol=float(atol),
            )
        except Exception as ex:
            record["reason"] = f"compare_failed: {ex}"
            records.append(record)
            continue

        record.update(compare_result)
        records.append(record)

    compared = [r for r in records if r["status"] == "compared"]
    skipped = [r for r in records if r["status"] != "compared"]

    compared_sorted = sorted(
        compared,
        key=lambda x: float(x["max_abs"]) if x["max_abs"] is not None else -1.0,
        reverse=True,
    )
    all_rows: List[Dict[str, Any]] = []
    for rank, row in enumerate(compared_sorted, start=1):
        row_copy = dict(row)
        row_copy["rank"] = rank
        all_rows.append(row_copy)
    for row in skipped:
        row_copy = dict(row)
        row_copy["rank"] = ""
        all_rows.append(row_copy)

    top_n = int(min(top, len(compared_sorted)))
    top_errors = compared_sorted[:top_n]

    summary = {
        "total_targets": len(onnx_tensor_names),
        "inferable_targets": len(inferable_output_names),
        "matchable_inferable_targets": len(target_output_names),
        "compared_count": len(compared),
        "skipped_count": len(skipped),
        "allclose_pass_count": int(sum(1 for r in compared if bool(r["allclose"]))),
    }
    report = {
        "schema_version": 1,
        "onnx_path": onnx_path,
        "tflite_path": tflite_path,
        "rtol": float(rtol),
        "atol": float(atol),
        "summary": summary,
        "top_errors": top_errors,
        "records": all_rows,
    }

    os.makedirs(os.path.dirname(output_json_path) or ".", exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    _write_csv(output_csv_path, all_rows)

    if verbose:
        print(
            f"[op-error] compared={summary['compared_count']}/{summary['total_targets']} "
            f"skipped={summary['skipped_count']} "
            f"allclose_pass={summary['allclose_pass_count']}/{summary['compared_count']}"
        )
        print(f"[op-error] json={output_json_path}")
        print(f"[op-error] csv={output_csv_path}")
        print("[op-error] top max_abs:")
        for idx, item in enumerate(top_errors, start=1):
            print(
                f"  {idx:2d}. id={item['identifier']:<3d} "
                f"{item['onnx_op_type']:<20} "
                f"{item['onnx_op_name']:<30} "
                f"max_abs={item['max_abs']:.8g} "
                f"tensor={item['tensor_name']}"
            )

    return {
        "report": report,
        "summary": summary,
        "output_json_path": output_json_path,
        "output_csv_path": output_csv_path,
        "top_errors": top_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize tensor-level errors between ONNX and flatbuffer_direct TFLite "
            "by matching intermediate tensor names."
        )
    )
    parser.add_argument("--onnx", required=True, help="Path to ONNX model.")
    parser.add_argument("--tflite", required=True, help="Path to TFLite model.")
    parser.add_argument(
        "--output_dir",
        default="saved_model",
        help="Directory to place generated report files.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional explicit output JSON path.",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Optional explicit output CSV path.",
    )
    parser.add_argument(
        "--tensor_correspondence_report",
        default=None,
        help=(
            "Optional tensor correspondence report generated by flatbuffer_direct. "
            "If omitted, {output_dir}/{model}_tensor_correspondence_report.json is auto-detected."
        ),
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="Number of worst tensors to print and include in top_errors.",
    )
    parser.add_argument("--rtol", type=float, default=0.0, help="allclose rtol.")
    parser.add_argument("--atol", type=float, default=1.0e-4, help="allclose atol.")
    args = parser.parse_args()

    generate_op_error_report(
        onnx_path=str(args.onnx),
        tflite_path=str(args.tflite),
        output_dir=str(args.output_dir),
        output_json=args.output_json,
        output_csv=args.output_csv,
        tensor_correspondence_report=args.tensor_correspondence_report,
        top=int(args.top),
        rtol=float(args.rtol),
        atol=float(args.atol),
        verbose=True,
    )


if __name__ == "__main__":
    main()
