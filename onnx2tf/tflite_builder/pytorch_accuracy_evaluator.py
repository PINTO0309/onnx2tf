from __future__ import annotations

import hashlib
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import onnx
import torch

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _align_output_layout_for_compare,
    _build_eval_inputs_for_sample,
    _collect_onnx_input_specs,
    _judge_metrics,
    _load_custom_input_data,
    _MetricAccumulator,
    _normalize_tensor_name,
    _resolve_metric_thresholds,
)

_PYTORCH_FLOAT_METRIC_THRESHOLDS = {
    "max_abs": 5.0e-2,
    "mean_abs": 5.0e-3,
    "rmse": 6.0e-3,
    "cosine_similarity": 0.9990,
}


def _import_generated_package(package_path: str) -> Any:
    package_root = Path(package_path)
    package_init = package_root / "__init__.py"
    if not package_init.exists():
        raise FileNotFoundError(f"Generated package is missing __init__.py. path={package_init}")
    module_name = (
        "_onnx2tf_generated_"
        f"{hashlib.sha256(str(package_root.resolve()).encode('utf-8')).hexdigest()}"
    )
    if module_name in sys.modules:
        del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        module_name,
        str(package_init),
        submodule_search_locations=[str(package_root)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for generated package. path={package_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _is_string_dtype(np_dtype: np.dtype) -> bool:
    return np_dtype.kind in {"U", "S", "O"}


def _generate_string_input(
    *,
    shape: Tuple[int, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    if len(shape) == 0:
        return np.asarray(f"token_{int(rng.integers(0, 1000000))}", dtype=object)
    total = int(np.prod(shape, dtype=np.int64))
    values = np.asarray(
        [f"token_{int(rng.integers(0, 1000000))}" for _ in range(total)],
        dtype=object,
    )
    return values.reshape(shape)


def _build_pytorch_eval_inputs_for_sample(
    *,
    input_specs: Sequence[Tuple[str, np.dtype, Tuple[int, ...]]],
    custom_inputs: Dict[str, np.ndarray],
    sample_index: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    inputs: Dict[str, np.ndarray] = {}
    for input_name, input_dtype, input_shape in input_specs:
        if _is_string_dtype(np.dtype(input_dtype)):
            inputs[str(input_name)] = _generate_string_input(
                shape=input_shape,
                rng=rng,
            )
            continue
        inputs[str(input_name)] = _build_eval_inputs_for_sample(
            input_specs=[(input_name, input_dtype, input_shape)],
            custom_inputs=custom_inputs,
            sample_index=sample_index,
            rng=rng,
        )[str(input_name)]
    return inputs


def _load_package_metadata(package_dir: str) -> Dict[str, Any]:
    metadata_path = Path(package_dir) / "metadata.json"
    if not metadata_path.exists():
        return {}
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _permute_shape(values: Sequence[int], perm: Sequence[int]) -> List[int]:
    return [int(values[idx]) for idx in perm]


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


def _adapt_numpy_input_for_package(
    *,
    value: np.ndarray,
    target_shape: Optional[Sequence[int]],
) -> np.ndarray:
    array = np.asarray(value)
    if target_shape is None:
        return array
    actual_shape = [int(v) for v in list(array.shape)]
    target = [int(v) for v in list(target_shape)]
    if actual_shape == target:
        return array
    if len(actual_shape) == len(target):
        perm = _perm_cl_to_cf(len(actual_shape))
        if perm is not None and _permute_shape(actual_shape, perm) == target:
            return np.transpose(array, axes=perm)
        perm_inv = _perm_cf_to_cl(len(actual_shape))
        if perm_inv is not None and _permute_shape(actual_shape, perm_inv) == target:
            return np.transpose(array, axes=perm_inv)
    return array


def _convert_inputs_for_package(
    inputs: Dict[str, np.ndarray],
    *,
    package_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    converted: Dict[str, Any] = {}
    tensor_meta_map = {}
    if isinstance(package_metadata, dict):
        tensor_meta_map = package_metadata.get("tensors", {}) or {}
    for input_name, value in inputs.items():
        array = np.asarray(value)
        input_meta = tensor_meta_map.get(str(input_name), {}) if isinstance(tensor_meta_map, dict) else {}
        target_shape = input_meta.get("shape_signature", input_meta.get("shape", None))
        if isinstance(target_shape, list):
            array = _adapt_numpy_input_for_package(
                value=array,
                target_shape=target_shape,
            )
        if _is_string_dtype(array.dtype):
            converted[str(input_name)] = array
        else:
            converted[str(input_name)] = torch.as_tensor(array)
    return converted


def _normalize_package_output(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (str, bytes)):
        return np.asarray(value, dtype=object)
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return np.asarray(value)


def _infer_probability_axis(
    *,
    output_name: str,
    output_value: np.ndarray,
    package_metadata: Optional[Dict[str, Any]],
) -> Optional[int]:
    if output_value.ndim < 3:
        return None
    tensor_meta_map = (package_metadata or {}).get("tensors", {}) if isinstance(package_metadata, dict) else {}
    output_meta = tensor_meta_map.get(str(output_name), {}) if isinstance(tensor_meta_map, dict) else {}
    logical_layout = str(output_meta.get("logical_layout", "UNKNOWN")).upper()
    if logical_layout in {"NC", "NCW", "NCHW", "NCDHW"} and output_value.ndim >= 2:
        return 1
    if logical_layout in {"NWC", "NHWC", "NDHWC"} and output_value.ndim >= 2:
        return output_value.ndim - 1
    if output_value.ndim >= 2 and int(output_value.shape[1]) <= 8:
        return 1
    if int(output_value.shape[-1]) <= 8:
        return output_value.ndim - 1
    return None


def _evaluate_probability_map_equivalence(
    *,
    ref: np.ndarray,
    pred: np.ndarray,
    axis: Optional[int],
) -> Optional[Dict[str, Any]]:
    if axis is None or ref.ndim != pred.ndim or ref.ndim < 3:
        return None
    resolved_axis = int(axis)
    if resolved_axis < 0:
        resolved_axis += ref.ndim
    if resolved_axis < 0 or resolved_axis >= ref.ndim:
        return None
    class_dim = int(ref.shape[resolved_axis])
    if class_dim < 2 or class_dim > 8 or int(pred.shape[resolved_axis]) != class_dim:
        return None
    ref_sum = np.sum(ref, axis=resolved_axis)
    pred_sum = np.sum(pred, axis=resolved_axis)
    if not (
        np.allclose(ref_sum, 1.0, atol=1.0e-3, rtol=1.0e-3)
        and np.allclose(pred_sum, 1.0, atol=1.0e-3, rtol=1.0e-3)
    ):
        return None
    ref_argmax = np.argmax(ref, axis=resolved_axis)
    pred_argmax = np.argmax(pred, axis=resolved_axis)
    match_ratio = float(np.mean(ref_argmax == pred_argmax))
    return {
        "class_axis": int(resolved_axis),
        "class_count": int(class_dim),
        "argmax_match_ratio": match_ratio,
        "pass": bool(match_ratio == 1.0),
    }


def _canonical_tensor_name(name: str) -> str:
    pieces: List[str] = []
    prev_was_sep = False
    for ch in str(name):
        if ch.isalnum():
            pieces.append(ch.lower())
            prev_was_sep = False
            continue
        if not prev_was_sep:
            pieces.append("_")
            prev_was_sep = True
    return "".join(pieces).strip("_")


def _resolve_named_output_value(
    outputs: Dict[str, Any],
    output_name: str,
) -> Any:
    if str(output_name) in outputs:
        return outputs[str(output_name)]
    normalized_output_name = _normalize_tensor_name(str(output_name))
    canonical_output_name = _canonical_tensor_name(str(output_name))
    for candidate_name, candidate_value in outputs.items():
        normalized_candidate_name = _normalize_tensor_name(str(candidate_name))
        canonical_candidate_name = _canonical_tensor_name(str(candidate_name))
        if (
            normalized_candidate_name == normalized_output_name
            or canonical_candidate_name == canonical_output_name
            or normalized_candidate_name.endswith(normalized_output_name)
            or canonical_candidate_name.endswith(canonical_output_name)
        ):
            return candidate_value
    raise KeyError(str(output_name))


def _can_use_identity_string_reference(onnx_graph: onnx.ModelProto) -> bool:
    nodes = list(onnx_graph.graph.node)
    graph_inputs = [
        graph_input.name
        for graph_input in onnx_graph.graph.input
        if graph_input.name not in {init.name for init in onnx_graph.graph.initializer}
    ]
    graph_outputs = [graph_output.name for graph_output in onnx_graph.graph.output]
    return (
        len(nodes) == 1
        and str(nodes[0].op_type) == "StringNormalizer"
        and len(graph_inputs) == 1
        and len(graph_outputs) == 1
    )


def evaluate_pytorch_package_outputs(
    *,
    onnx_graph: onnx.ModelProto,
    package_dir: str,
    output_report_path: str,
    num_samples: int = 1,
    seed: int = 0,
    rtol: float = 1.0e-4,
    atol: float = 1.0e-4,
    metric_thresholds: Optional[Dict[str, float]] = None,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    import onnxruntime as ort

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0. got: {num_samples}")
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"PyTorch package does not exist. path={package_dir}")

    rng = np.random.default_rng(seed=int(seed))
    input_specs = _collect_onnx_input_specs(onnx_graph)
    custom_inputs = _load_custom_input_data(custom_input_op_name_np_data_path)
    onnx_output_names = [str(output.name) for output in onnx_graph.graph.output]
    ort_session = None
    ort_fallback_identity = False
    ort_reference_error: Optional[Dict[str, str]] = None
    try:
        ort_session = ort.InferenceSession(
            onnx_graph.SerializeToString(),
            providers=["CPUExecutionProvider"],
        )
    except Exception as ex:
        if _can_use_identity_string_reference(onnx_graph):
            ort_fallback_identity = True
        else:
            ort_reference_error = {
                "stage": "session_create",
                "error_type": type(ex).__name__,
                "error_message": str(ex),
            }
    pkg = _import_generated_package(package_dir)
    package_metadata = _load_package_metadata(package_dir)
    model = pkg.load_model(eval_mode=True)

    if ort_reference_error is not None:
        report = {
            "schema_version": 1,
            "backend": "pytorch_package",
            "package_dir": str(package_dir),
            "num_samples": int(num_samples),
            "rtol": float(rtol),
            "atol": float(atol),
            "inputs_source": (
                "custom_input_op_name_np_data_path"
                if len(custom_inputs) > 0
                else "seeded_random"
            ),
            "evaluation_pass": None,
            "evaluation_skipped": True,
            "skip_reason": "onnxruntime_reference_error",
            "onnxruntime_reference_error": ort_reference_error,
            "overall_metrics": {},
            "metric_judgement": None,
            "string_exact_match_failures": [],
            "numeric_allclose_failures": [],
            "per_output": {},
        }
        os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
        with open(output_report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return report

    overall_metrics = _MetricAccumulator()
    per_output_metrics: Dict[str, _MetricAccumulator] = {
        output_name: _MetricAccumulator() for output_name in onnx_output_names
    }
    per_output_report: Dict[str, Dict[str, Any]] = {
        output_name: {"kind": "numeric"} for output_name in onnx_output_names
    }
    exact_match_failures: List[str] = []
    allclose_failures: List[str] = []
    task_equivalent_passes: List[str] = []

    for sample_index in range(int(num_samples)):
        onnx_inputs = _build_pytorch_eval_inputs_for_sample(
            input_specs=input_specs,
            custom_inputs=custom_inputs,
            sample_index=sample_index,
            rng=rng,
        )
        package_outputs = model.forward_named(
            **_convert_inputs_for_package(
                onnx_inputs,
                package_metadata=package_metadata,
            )
        )
        try:
            if ort_session is not None:
                onnx_outputs = ort_session.run(None, onnx_inputs)
                onnx_outputs_by_name = {
                    str(name): np.asarray(value)
                    for name, value in zip(onnx_output_names, onnx_outputs)
                }
            elif ort_fallback_identity:
                input_name = str(input_specs[0][0])
                onnx_outputs_by_name = {
                    str(onnx_output_names[0]): np.asarray(onnx_inputs[input_name], dtype=object)
                }
            else:
                raise RuntimeError("No ONNX reference execution path is available.")
        except Exception as ex:
            report = {
                "schema_version": 1,
                "backend": "pytorch_package",
                "package_dir": str(package_dir),
                "num_samples": int(num_samples),
                "rtol": float(rtol),
                "atol": float(atol),
                "inputs_source": (
                    "custom_input_op_name_np_data_path"
                    if len(custom_inputs) > 0
                    else "seeded_random"
                ),
                "evaluation_pass": None,
                "evaluation_skipped": True,
                "skip_reason": "onnxruntime_reference_error",
                "onnxruntime_reference_error": {
                    "stage": "inference",
                    "sample_index": int(sample_index),
                    "error_type": type(ex).__name__,
                    "error_message": str(ex),
                },
                "overall_metrics": {},
                "metric_judgement": None,
                "string_exact_match_failures": [],
                "numeric_allclose_failures": [],
                "per_output": {},
            }
            os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
            with open(output_report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)
            return report

        for output_name in onnx_output_names:
            ref = np.asarray(onnx_outputs_by_name[str(output_name)])
            pred = _normalize_package_output(
                _resolve_named_output_value(
                    outputs=package_outputs,
                    output_name=str(output_name),
                )
            )
            if _is_string_dtype(ref.dtype) or _is_string_dtype(pred.dtype):
                ref_obj = np.asarray(ref, dtype=object)
                pred_obj = np.asarray(pred, dtype=object)
                matched = bool(np.array_equal(ref_obj, pred_obj))
                per_output_report[str(output_name)] = {
                    "kind": "string",
                    "exact_match": matched,
                }
                if not matched:
                    exact_match_failures.append(str(output_name))
                continue

            aligned_pred, align_mode, align_perm = _align_output_layout_for_compare(
                onnx_output=ref,
                tflite_output=pred,
                rtol=float(rtol),
                atol=float(atol),
            )
            per_output_metrics[str(output_name)].update(ref, aligned_pred)
            overall_metrics.update(ref, aligned_pred)
            allclose = bool(
                np.allclose(
                    np.asarray(ref),
                    np.asarray(aligned_pred),
                    rtol=float(rtol),
                    atol=float(atol),
                    equal_nan=True,
                )
            )
            if not allclose:
                allclose_failures.append(str(output_name))
            probability_equivalence = _evaluate_probability_map_equivalence(
                ref=np.asarray(ref),
                pred=np.asarray(aligned_pred),
                axis=_infer_probability_axis(
                    output_name=str(output_name),
                    output_value=np.asarray(ref),
                    package_metadata=package_metadata,
                ),
            )
            if probability_equivalence is not None and bool(probability_equivalence.get("pass", False)):
                task_equivalent_passes.append(str(output_name))
            per_output_report[str(output_name)] = {
                "kind": "numeric",
                "allclose": allclose,
                "align_mode": str(align_mode),
                "align_perm": [int(v) for v in align_perm] if align_perm is not None else None,
                "probability_map_equivalence": probability_equivalence,
            }

    thresholds = _resolve_metric_thresholds(
        metric_thresholds=(
            dict(_PYTORCH_FLOAT_METRIC_THRESHOLDS)
            if metric_thresholds is None
            else metric_thresholds
        ),
        use_quant_defaults=False,
    )
    overall_metrics_dict = overall_metrics.to_dict()
    metric_judgement = _judge_metrics(
        metrics=overall_metrics_dict,
        thresholds=thresholds,
        rtol=float(rtol),
    )
    for output_name, metrics in per_output_metrics.items():
        if per_output_report[str(output_name)].get("kind") != "numeric":
            continue
        output_metrics = metrics.to_dict()
        per_output_report[str(output_name)]["metrics"] = output_metrics
        per_output_report[str(output_name)]["metric_judgement"] = _judge_metrics(
            metrics=output_metrics,
            thresholds=thresholds,
            rtol=float(rtol),
        )
    numeric_outputs_pass = True
    for output_name in onnx_output_names:
        output_report = per_output_report[str(output_name)]
        if output_report.get("kind") != "numeric":
            continue
        if bool(output_report.get("metric_judgement", {}).get("pass", False)):
            continue
        if bool((output_report.get("probability_map_equivalence") or {}).get("pass", False)):
            continue
        numeric_outputs_pass = False
        break

    report = {
        "schema_version": 1,
        "backend": "pytorch_package",
        "package_dir": str(package_dir),
        "num_samples": int(num_samples),
        "rtol": float(rtol),
        "atol": float(atol),
        "inputs_source": (
            "custom_input_op_name_np_data_path"
            if len(custom_inputs) > 0
            else "seeded_random"
        ),
        "evaluation_pass": bool(
            (metric_judgement["pass"] or numeric_outputs_pass)
            and len(exact_match_failures) == 0
        ),
        "evaluation_skipped": False,
        "skip_reason": None,
        "onnxruntime_reference_error": None,
        "overall_metrics": overall_metrics_dict,
        "metric_judgement": metric_judgement,
        "task_equivalent_numeric_outputs": sorted(set(task_equivalent_passes)),
        "string_exact_match_failures": sorted(set(exact_match_failures)),
        "numeric_allclose_failures": sorted(set(allclose_failures)),
        "per_output": per_output_report,
    }
    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def smoke_test_pytorch_package_inference(
    *,
    onnx_graph: onnx.ModelProto,
    package_dir: str,
    output_report_path: str,
    num_samples: int = 1,
    seed: int = 0,
) -> Dict[str, Any]:
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0. got: {num_samples}")
    if not os.path.exists(package_dir):
        raise FileNotFoundError(f"PyTorch package does not exist. path={package_dir}")

    rng = np.random.default_rng(seed=int(seed))
    input_specs = _collect_onnx_input_specs(onnx_graph)
    custom_inputs: Dict[str, np.ndarray] = {}
    pkg = _import_generated_package(package_dir)
    package_metadata = _load_package_metadata(package_dir)
    model = pkg.load_model(eval_mode=True)

    sample_reports: List[Dict[str, Any]] = []
    inference_pass = True
    error_payload: Optional[Dict[str, Any]] = None
    for sample_index in range(int(num_samples)):
        try:
            model_inputs = _build_pytorch_eval_inputs_for_sample(
                input_specs=input_specs,
                custom_inputs=custom_inputs,
                sample_index=sample_index,
                rng=rng,
            )
            outputs = model.forward_named(
                **_convert_inputs_for_package(
                    model_inputs,
                    package_metadata=package_metadata,
                )
            )
            sample_reports.append(
                {
                    "sample_index": int(sample_index),
                    "outputs": {
                        str(name): {
                            "dtype": str(np.asarray(_normalize_package_output(value)).dtype),
                            "shape": [
                                int(v) for v in list(np.asarray(_normalize_package_output(value)).shape)
                            ],
                        }
                        for name, value in outputs.items()
                    },
                }
            )
        except Exception as ex:
            inference_pass = False
            error_payload = {
                "sample_index": int(sample_index),
                "error_type": type(ex).__name__,
                "error_message": str(ex),
            }
            break

    report = {
        "schema_version": 1,
        "backend": "pytorch_package_smoke",
        "package_dir": str(package_dir),
        "num_samples": int(num_samples),
        "inference_pass": bool(inference_pass),
        "error": error_payload,
        "samples": sample_reports,
    }
    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report
