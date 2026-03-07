from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _align_output_layout_for_compare,
    _build_tflite_detail_map,
    _create_tflite_interpreter,
    _quantize_for_tflite_input,
)


def _read_split_manifest(split_manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(
            f"Split manifest does not exist. path={split_manifest_path}"
        )
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_name(name: str) -> str:
    normalized = str(name).split(":")[0]
    if normalized.startswith("serving_default_"):
        normalized = normalized[len("serving_default_") :]
    return normalized


def _runtime_shape_from_detail(detail: Dict[str, Any]) -> Tuple[int, ...]:
    shape_signature = detail.get("shape_signature", None)
    if shape_signature is None:
        shape_signature = detail.get("shape", None)
    values = [int(v) for v in np.asarray(shape_signature).reshape(-1).tolist()]
    runtime_shape: List[int] = []
    for axis, dim in enumerate(values):
        runtime_shape.append(1 if int(dim) <= 0 else int(dim))
    if len(runtime_shape) == 0:
        runtime_shape = [1]
    return tuple(runtime_shape)


def _fallback_shape_from_tensor_spec(tensor_spec: tf.TensorSpec) -> Tuple[int, ...]:
    shape: List[int] = []
    for axis, dim in enumerate(tensor_spec.shape.as_list()):
        if dim is None or int(dim) <= 0:
            shape.append(1 if axis == 0 else 16)
        else:
            shape.append(int(dim))
    if len(shape) == 0:
        shape = [1]
    return tuple(shape)


def _cast_array_for_tensor_spec(
    *,
    value: np.ndarray,
    tensor_spec: tf.TensorSpec,
) -> np.ndarray:
    casted = np.asarray(value)
    expected_shape = _fallback_shape_from_tensor_spec(tensor_spec)
    if casted.size == int(np.prod(expected_shape, dtype=np.int64)):
        casted = casted.reshape(expected_shape)
    elif tuple(casted.shape) != expected_shape:
        raise ValueError(
            "SavedModel input shape mismatch. "
            f"input_name={tensor_spec.name} expected={expected_shape} actual={tuple(casted.shape)}"
        )

    target_dtype = np.dtype(tensor_spec.dtype.as_numpy_dtype)
    if casted.dtype != target_dtype:
        if np.issubdtype(target_dtype, np.bool_):
            casted = (casted.astype(np.float32) > 0.0).astype(np.bool_)
        else:
            casted = casted.astype(target_dtype)
    return casted


def _resolve_saved_model_input_name(
    *,
    requested_name: str,
    signature_inputs: Dict[str, tf.TensorSpec],
) -> str:
    normalized_requested = _normalize_name(requested_name)
    for candidate in signature_inputs.keys():
        if str(candidate) == str(requested_name):
            return str(candidate)
    for candidate in signature_inputs.keys():
        if _normalize_name(str(candidate)) == normalized_requested:
            return str(candidate)
    raise ValueError(
        "Failed to map split SavedModel input. "
        f"requested_name={requested_name} available={list(signature_inputs.keys())}"
    )


def _resolve_saved_model_outputs(
    *,
    requested_outputs: List[str],
    raw_outputs: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    output_arrays: Dict[str, np.ndarray] = {}
    normalized_outputs = {
        _normalize_name(str(name)): np.asarray(value)
        for name, value in raw_outputs.items()
    }
    ordered_names = [str(name) for name in raw_outputs.keys()]
    for output_index, requested_name in enumerate(requested_outputs):
        if requested_name in raw_outputs:
            output_arrays[requested_name] = np.asarray(raw_outputs[requested_name])
            continue
        normalized_name = _normalize_name(requested_name)
        if normalized_name in normalized_outputs:
            output_arrays[requested_name] = np.asarray(normalized_outputs[normalized_name])
            continue
        if output_index < len(ordered_names):
            output_arrays[requested_name] = np.asarray(raw_outputs[ordered_names[output_index]])
            continue
        raise ValueError(
            "Failed to map split SavedModel output. "
            f"requested_name={requested_name} available={ordered_names}"
        )
    return output_arrays


def _run_split_saved_models(
    *,
    split_manifest: Dict[str, Any],
    base_folder_path: str,
    sample_inputs: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    tensor_store: Dict[str, np.ndarray] = dict(sample_inputs)
    last_outputs: Dict[str, np.ndarray] = {}
    for partition in list(split_manifest.get("partitions", [])):
        saved_model_dir = str(partition.get("saved_model_dir", "")).strip()
        if saved_model_dir == "":
            raise ValueError(
                "Split manifest partition is missing saved_model_dir. "
                f"partition_id={partition.get('partition_id')}"
            )
        saved_model_path = os.path.join(base_folder_path, saved_model_dir)
        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(
                f"Split SavedModel directory does not exist. path={saved_model_path}"
            )

        module = tf.saved_model.load(saved_model_path)
        signature_fn = module.signatures.get("serving_default", None)
        if signature_fn is None:
            raise RuntimeError(
                "serving_default signature is missing in split SavedModel. "
                f"path={saved_model_path}"
            )
        signature_inputs = signature_fn.structured_input_signature[1]
        if not isinstance(signature_inputs, dict):
            raise RuntimeError(
                "Split SavedModel serving_default signature inputs are invalid. "
                f"path={saved_model_path}"
            )

        serving_inputs: Dict[str, tf.Tensor] = {}
        for input_name in list(partition.get("inputs", [])):
            input_value = tensor_store.get(str(input_name), None)
            if input_value is None:
                input_value = tensor_store.get(_normalize_name(str(input_name)), None)
            if input_value is None:
                raise ValueError(
                    "Split SavedModel input is missing from tensor store. "
                    f"partition_id={partition.get('partition_id')} input_name={input_name}"
                )
            signature_input_name = _resolve_saved_model_input_name(
                requested_name=str(input_name),
                signature_inputs=signature_inputs,
            )
            tensor_spec = signature_inputs[signature_input_name]
            casted_input = _cast_array_for_tensor_spec(
                value=np.asarray(input_value),
                tensor_spec=tensor_spec,
            )
            serving_inputs[signature_input_name] = tf.convert_to_tensor(
                casted_input,
                dtype=tensor_spec.dtype,
            )

        with tf.device("/CPU:0"):
            raw_outputs = signature_fn(**serving_inputs)
        if not isinstance(raw_outputs, dict):
            raise RuntimeError(
                "Split SavedModel serving_default did not return a dict. "
                f"path={saved_model_path}"
            )

        current_outputs = _resolve_saved_model_outputs(
            requested_outputs=[str(v) for v in list(partition.get("outputs", []))],
            raw_outputs=raw_outputs,
        )
        tensor_store.update(current_outputs)
        for name, value in current_outputs.items():
            tensor_store[_normalize_name(name)] = np.asarray(value)
        last_outputs = current_outputs
    return last_outputs


def evaluate_split_saved_model_outputs(
    *,
    split_manifest_path: str,
    reference_tflite_path: str,
    output_report_path: str,
    source_label: str,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
) -> Dict[str, Any]:
    split_manifest = _read_split_manifest(split_manifest_path)
    partitions = list(split_manifest.get("partitions", []))
    if len(partitions) == 0:
        raise ValueError("Split manifest contains no partition entries.")

    if not os.path.exists(reference_tflite_path):
        raise FileNotFoundError(
            f"Reference tflite does not exist. path={reference_tflite_path}"
        )

    report: Dict[str, Any] = {
        "schema_version": 1,
        "mode": "split_saved_model",
        "source_label": str(source_label),
        "split_manifest_path": str(split_manifest_path),
        "reference_tflite_path": str(reference_tflite_path),
        "inference": {
            "status": "not_run",
            "reason": "",
            "partitions_run": 0,
        },
        "comparison": {
            "status": "not_run",
            "reason": "",
            "pass": None,
            "matched": 0,
            "total": 0,
            "max_abs": None,
            "unmatched_outputs": [],
        },
        "overall_pass": False,
    }

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)

    failure_reason = ""
    try:
        interpreter = _create_tflite_interpreter(model_path=reference_tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        rng = np.random.default_rng(0)
        sample_inputs_by_name: Dict[str, np.ndarray] = {}
        tflite_inputs_by_index: Dict[int, np.ndarray] = {}
        for detail in input_details:
            runtime_shape = _runtime_shape_from_detail(detail)
            random_input = rng.uniform(-1.0, 1.0, size=runtime_shape).astype(np.float32)
            sample_inputs_by_name[str(detail.get("name", ""))] = np.asarray(random_input)
            sample_inputs_by_name[_normalize_name(str(detail.get("name", "")))] = np.asarray(random_input)
            tflite_inputs_by_index[int(detail["index"])] = _quantize_for_tflite_input(
                np.asarray(random_input),
                detail,
            )

        for detail in input_details:
            interpreter.set_tensor(
                int(detail["index"]),
                tflite_inputs_by_index[int(detail["index"])],
            )
        interpreter.invoke()

        final_partition_outputs = [str(v) for v in list(partitions[-1].get("outputs", []))]
        tflite_output_map = _build_tflite_detail_map(
            onnx_names=final_partition_outputs,
            tflite_details=output_details,
        )
        reference_outputs = {
            output_name: np.asarray(
                interpreter.get_tensor(int(tflite_output_map[output_name]["index"]))
            )
            for output_name in final_partition_outputs
        }

        split_outputs = _run_split_saved_models(
            split_manifest=split_manifest,
            base_folder_path=os.path.dirname(split_manifest_path),
            sample_inputs=sample_inputs_by_name,
        )
        report["inference"]["status"] = "passed"
        report["inference"]["reason"] = ""
        report["inference"]["partitions_run"] = int(len(partitions))

        comparison_total = 0
        comparison_matched = 0
        max_abs = 0.0
        unmatched_outputs: List[str] = []
        for output_name in final_partition_outputs:
            if output_name not in split_outputs:
                unmatched_outputs.append(str(output_name))
                continue
            ref = np.asarray(reference_outputs[output_name])
            pred = np.asarray(split_outputs[output_name])
            aligned_pred = np.asarray(pred)
            try:
                aligned_pred, _, _ = _align_output_layout_for_compare(
                    onnx_output=ref,
                    tflite_output=pred,
                    rtol=float(rtol),
                    atol=float(atol),
                )
            except Exception:
                if ref.size == pred.size and ref.shape != pred.shape:
                    aligned_pred = np.asarray(pred).reshape(ref.shape)
            comparison_total += 1
            if bool(np.allclose(ref, aligned_pred, rtol=float(rtol), atol=float(atol), equal_nan=True)):
                comparison_matched += 1
            else:
                unmatched_outputs.append(str(output_name))
            if ref.size > 0:
                with np.errstate(invalid="ignore"):
                    diff = np.abs(
                        np.asarray(ref, dtype=np.float64) - np.asarray(aligned_pred, dtype=np.float64)
                    )
                    if np.any(np.isfinite(diff)):
                        local_max_abs = float(np.nanmax(diff))
                        if local_max_abs > max_abs:
                            max_abs = local_max_abs

        comparison_pass = (
            int(comparison_total) == int(comparison_matched)
            and len(unmatched_outputs) == 0
        )
        report["comparison"]["status"] = "passed" if comparison_pass else "failed"
        report["comparison"]["reason"] = "" if comparison_pass else "output_mismatch"
        report["comparison"]["pass"] = bool(comparison_pass)
        report["comparison"]["matched"] = int(comparison_matched)
        report["comparison"]["total"] = int(comparison_total)
        report["comparison"]["max_abs"] = float(max_abs)
        report["comparison"]["unmatched_outputs"] = sorted(set(unmatched_outputs))
        report["overall_pass"] = bool(comparison_pass)

        if not comparison_pass:
            failure_reason = (
                "Split SavedModel/TFLite output comparison failed. "
                f"source={source_label} matched={comparison_matched}/{comparison_total} "
                f"max_abs={max_abs:.6g} unmatched_outputs={sorted(set(unmatched_outputs))}"
            )
            raise RuntimeError(failure_reason)
    except Exception as ex:
        if report["inference"]["status"] != "passed":
            report["inference"]["status"] = "failed"
            report["inference"]["reason"] = str(ex)
            report["comparison"]["status"] = "skipped"
            report["comparison"]["reason"] = "inference_failed"
            report["comparison"]["pass"] = False
        report["overall_pass"] = False
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        if failure_reason == "":
            failure_reason = (
                "Split SavedModel inference check failed. "
                f"source={source_label} reason={ex}"
            )
        raise RuntimeError(failure_reason) from ex

    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
