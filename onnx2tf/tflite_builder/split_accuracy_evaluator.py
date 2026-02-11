from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import onnx

from onnx2tf.tflite_builder.accuracy_evaluator import (
    _MetricAccumulator,
    _build_tflite_detail_map,
    _collect_onnx_input_specs,
    _dequantize_tflite_output,
    _extract_sample_from_custom,
    _FLOAT_METRIC_THRESHOLDS,
    _generate_seeded_input,
    _judge_metrics,
    _load_custom_input_data,
    _normalize_tensor_name,
    _quantize_for_tflite_input,
    _QUANT_METRIC_THRESHOLDS,
    _resolve_compare_mode,
    _resolve_metric_thresholds,
)


def _read_split_manifest(split_manifest_path: str) -> Dict[str, Any]:
    if not os.path.exists(split_manifest_path):
        raise FileNotFoundError(
            f"Split manifest does not exist. path={split_manifest_path}"
        )
    with open(split_manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_tflite_model(
    *,
    interpreter: Any,
    model_input_names: List[str],
    provided_inputs: Dict[str, np.ndarray],
    model_output_names: List[str],
    compare_mode: str,
) -> Dict[str, np.ndarray]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_map = _build_tflite_detail_map(
        onnx_names=model_input_names,
        tflite_details=input_details,
    )
    output_map = _build_tflite_detail_map(
        onnx_names=model_output_names,
        tflite_details=output_details,
    )

    for input_name in model_input_names:
        input_detail = input_map[input_name]
        if input_name not in provided_inputs:
            raise ValueError(
                f"Missing model input tensor for tflite run. input_name={input_name}"
            )
        interpreter.set_tensor(
            input_detail["index"],
            _quantize_for_tflite_input(provided_inputs[input_name], input_detail),
        )
    interpreter.invoke()

    outputs: Dict[str, np.ndarray] = {}
    for output_name in model_output_names:
        output_detail = output_map[output_name]
        out = interpreter.get_tensor(output_detail["index"])
        if compare_mode == "dequant":
            out = _dequantize_tflite_output(out, output_detail)
        else:
            out = np.asarray(out)
        outputs[output_name] = out
    return outputs


def _run_split_partitions(
    *,
    split_manifest: Dict[str, Any],
    base_folder_path: str,
    sample_inputs: Dict[str, np.ndarray],
    compare_mode: str,
) -> Dict[str, np.ndarray]:
    from ai_edge_litert.interpreter import Interpreter

    partitions = list(split_manifest.get("partitions", []))
    if len(partitions) == 0:
        raise ValueError("Split manifest contains no partitions.")

    tensor_store: Dict[str, np.ndarray] = dict(sample_inputs)
    last_outputs: Dict[str, np.ndarray] = {}
    for partition in partitions:
        part_file = str(partition["file"])
        part_path = os.path.join(base_folder_path, part_file)
        if not os.path.exists(part_path):
            raise FileNotFoundError(
                f"Split partition tflite does not exist. path={part_path}"
            )
        interpreter = Interpreter(model_path=part_path)
        interpreter.allocate_tensors()
        part_inputs = list(partition.get("inputs", []))
        part_outputs = list(partition.get("outputs", []))

        provided_inputs: Dict[str, np.ndarray] = {}
        for input_name in part_inputs:
            if input_name in tensor_store:
                provided_inputs[input_name] = tensor_store[input_name]
            elif _normalize_tensor_name(input_name) in tensor_store:
                provided_inputs[input_name] = tensor_store[_normalize_tensor_name(input_name)]
            else:
                raise ValueError(
                    "Split partition input is missing from tensor store. "
                    f"partition={partition.get('partition_id')} input_name={input_name}"
                )
        current_outputs = _run_tflite_model(
            interpreter=interpreter,
            model_input_names=part_inputs,
            provided_inputs=provided_inputs,
            model_output_names=part_outputs,
            compare_mode=compare_mode,
        )
        tensor_store.update(current_outputs)
        last_outputs = current_outputs
    return last_outputs


def _load_test_data_nhwc(test_data_nhwc_path: Optional[str]) -> Optional[np.ndarray]:
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
    if data.shape[-1] != 3:
        raise ValueError(
            "test_data_nhwc_path must have 3 channels in the last dim. "
            f"actual_shape={tuple(data.shape)}"
        )
    if data.shape[0] <= 0:
        raise ValueError(
            "test_data_nhwc_path must include at least 1 sample. "
            f"actual_shape={tuple(data.shape)}"
        )
    return data


def _extract_sample_from_test_data_nhwc(
    *,
    data: np.ndarray,
    sample_index: int,
    expected_shape: tuple[int, ...],
    np_dtype: np.dtype,
) -> np.ndarray:
    import tensorflow as tf

    if len(expected_shape) != 4:
        raise ValueError(
            "test_data_nhwc_path can only be used for rank-4 inputs. "
            f"expected_shape={expected_shape}"
        )

    expected_batch = int(expected_shape[0]) if int(expected_shape[0]) > 0 else 1
    if data.shape[0] >= expected_batch:
        start = int(sample_index % data.shape[0])
        if start + expected_batch <= data.shape[0]:
            sample = data[start : start + expected_batch]
        else:
            indices = [(start + i) % data.shape[0] for i in range(expected_batch)]
            sample = data[indices]
    else:
        repeats = int(np.ceil(expected_batch / data.shape[0]))
        sample = np.concatenate([data] * repeats, axis=0)[:expected_batch]

    # ONNX input is NCHW
    if expected_shape[1] == 3:
        target_h = int(expected_shape[2]) if int(expected_shape[2]) > 0 else sample.shape[1]
        target_w = int(expected_shape[3]) if int(expected_shape[3]) > 0 else sample.shape[2]
        if sample.shape[1] != target_h or sample.shape[2] != target_w:
            sample = tf.image.resize(sample, [target_h, target_w]).numpy()
        sample = np.transpose(sample, [0, 3, 1, 2])
    # ONNX input is NHWC
    elif expected_shape[3] == 3:
        target_h = int(expected_shape[1]) if int(expected_shape[1]) > 0 else sample.shape[1]
        target_w = int(expected_shape[2]) if int(expected_shape[2]) > 0 else sample.shape[2]
        if sample.shape[1] != target_h or sample.shape[2] != target_w:
            sample = tf.image.resize(sample, [target_h, target_w]).numpy()
    else:
        raise ValueError(
            "test_data_nhwc_path can only be used for 3-channel image inputs. "
            f"expected_shape={expected_shape}"
        )

    return np.asarray(sample).astype(np_dtype, copy=False)


def evaluate_split_manifest_outputs(
    *,
    onnx_graph: onnx.ModelProto,
    split_manifest_path: str,
    reference_mode: str = "unsplit_tflite",
    reference_tflite_path: Optional[str] = None,
    output_report_path: str,
    num_samples: int = 10,
    seed: int = 0,
    custom_input_op_name_np_data_path: Optional[List[Any]] = None,
    test_data_nhwc_path: Optional[str] = None,
    rtol: float = 0.0,
    atol: float = 1e-4,
    compare_mode: str = "auto",
    fail_on_threshold: bool = False,
    metric_thresholds: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    import onnxruntime as ort
    from ai_edge_litert.interpreter import Interpreter

    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be > 0. got: {num_samples}")
    rtol = float(rtol)
    atol = float(atol)
    if rtol < 0.0 or atol < 0.0:
        raise ValueError(f"rtol/atol must be >= 0.0. got rtol={rtol} atol={atol}")
    reference_mode = str(reference_mode).lower()
    if reference_mode not in ["unsplit_tflite", "onnx"]:
        raise ValueError(
            f"reference_mode must be one of ['unsplit_tflite', 'onnx']. got: {reference_mode}"
        )

    split_manifest = _read_split_manifest(split_manifest_path)
    split_manifest_folder = os.path.dirname(split_manifest_path)
    partitions = list(split_manifest.get("partitions", []))
    if len(partitions) == 0:
        raise ValueError("Split manifest contains no partition entries.")

    rng = np.random.default_rng(seed=int(seed))
    custom_inputs = _load_custom_input_data(custom_input_op_name_np_data_path)
    test_data_nhwc = _load_test_data_nhwc(test_data_nhwc_path)
    input_specs = _collect_onnx_input_specs(onnx_graph)
    onnx_input_names = [name for name, _, _ in input_specs]
    onnx_output_names = [output.name for output in onnx_graph.graph.output]

    final_partition_outputs = list(partitions[-1].get("outputs", []))
    if len(final_partition_outputs) == 0:
        raise ValueError("Final split partition has no declared outputs.")

    has_quantized_outputs = False
    if reference_mode == "unsplit_tflite":
        if reference_tflite_path is None:
            raise ValueError("reference_tflite_path is required for reference_mode='unsplit_tflite'.")
        if not os.path.exists(reference_tflite_path):
            raise FileNotFoundError(
                f"Reference tflite does not exist. path={reference_tflite_path}"
            )
        ref_interpreter = Interpreter(model_path=reference_tflite_path)
        ref_interpreter.allocate_tensors()
        has_quantized_outputs = any(
            np.issubdtype(np.dtype(detail["dtype"]), np.integer)
            or np.issubdtype(np.dtype(detail["dtype"]), np.bool_)
            for detail in ref_interpreter.get_output_details()
        )
    else:
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
        output_name: _MetricAccumulator() for output_name in final_partition_outputs
    }
    allclose_total = 0
    allclose_matched = 0
    per_output_allclose: Dict[str, Dict[str, int]] = {
        output_name: {"matched": 0, "total": 0}
        for output_name in final_partition_outputs
    }
    used_custom_inputs = False
    used_test_data_nhwc = False

    for sample_index in range(int(num_samples)):
        sample_inputs: Dict[str, np.ndarray] = {}
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
                used_custom_inputs = True
            elif (
                test_data_nhwc is not None
                and len(input_shape) == 4
                and (int(input_shape[1]) == 3 or int(input_shape[3]) == 3)
            ):
                sample = _extract_sample_from_test_data_nhwc(
                    data=test_data_nhwc,
                    sample_index=sample_index,
                    expected_shape=input_shape,
                    np_dtype=input_dtype,
                )
                used_test_data_nhwc = True
            else:
                sample = _generate_seeded_input(
                    shape=input_shape,
                    np_dtype=input_dtype,
                    rng=rng,
                )
            sample_inputs[input_name] = sample

        split_outputs = _run_split_partitions(
            split_manifest=split_manifest,
            base_folder_path=split_manifest_folder,
            sample_inputs=sample_inputs,
            compare_mode=resolved_compare_mode,
        )

        if reference_mode == "unsplit_tflite":
            ref_outputs = _run_tflite_model(
                interpreter=ref_interpreter,
                model_input_names=onnx_input_names,
                provided_inputs=sample_inputs,
                model_output_names=final_partition_outputs,
                compare_mode=resolved_compare_mode,
            )
        else:
            onnx_outputs = onnx_session.run(onnx_output_names, sample_inputs)
            onnx_outputs_map = {
                name: value for name, value in zip(onnx_output_names, onnx_outputs)
            }
            ref_outputs = {}
            for output_name in final_partition_outputs:
                if output_name in onnx_outputs_map:
                    ref_outputs[output_name] = np.asarray(onnx_outputs_map[output_name])
                elif _normalize_tensor_name(output_name) in onnx_outputs_map:
                    ref_outputs[output_name] = np.asarray(
                        onnx_outputs_map[_normalize_tensor_name(output_name)]
                    )
                else:
                    raise ValueError(
                        "Failed to map split output to ONNX output. "
                        f"split_output={output_name}"
                    )

        for output_name in final_partition_outputs:
            ref = np.asarray(ref_outputs[output_name])
            pred = np.asarray(split_outputs[output_name])
            if ref.shape != pred.shape:
                if ref.size == pred.size:
                    pred = pred.reshape(ref.shape)
                else:
                    raise ValueError(
                        f"Split/reference output shape mismatch. output={output_name} "
                        f"ref={tuple(ref.shape)} pred={tuple(pred.shape)}"
                    )
            total_metrics.update(ref, pred)
            per_output_metrics[output_name].update(ref, pred)
            is_allclose = bool(
                np.allclose(ref, pred, rtol=rtol, atol=atol, equal_nan=True)
            )
            allclose_total += 1
            allclose_matched += int(is_allclose)
            per_output_allclose[output_name]["total"] += 1
            per_output_allclose[output_name]["matched"] += int(is_allclose)

    overall_metrics = total_metrics.to_dict()
    metric_judgement = _judge_metrics(
        metrics=overall_metrics,
        thresholds=resolved_thresholds,
    )
    allclose_pass = bool(allclose_total == allclose_matched)
    evaluation_pass = bool(metric_judgement["pass"] and allclose_pass)
    if used_custom_inputs and used_test_data_nhwc:
        inputs_source = "mixed_custom_and_test_data_nhwc"
    elif used_test_data_nhwc:
        inputs_source = "test_data_nhwc_path"
    elif used_custom_inputs:
        inputs_source = "custom_input_op_name_np_data_path"
    else:
        inputs_source = "seeded_random"

    report: Dict[str, Any] = {
        "schema_version": 1,
        "reference_mode": reference_mode,
        "split_manifest_path": split_manifest_path,
        "reference_tflite_path": reference_tflite_path,
        "inputs_source": inputs_source,
        "seed": int(seed),
        "num_samples": int(num_samples),
        "rtol": float(rtol),
        "atol": float(atol),
        "compare_mode": resolved_compare_mode,
        "metric_thresholds": {
            k: float(v) for k, v in resolved_thresholds.items()
        },
        "overall_metrics": overall_metrics,
        "metric_threshold_judgement": metric_judgement,
        "allclose_summary": {
            "matched": int(allclose_matched),
            "total": int(allclose_total),
            "pass": bool(allclose_pass),
        },
        "evaluation_pass": bool(evaluation_pass),
        "per_output_metrics": {
            output_name: {
                "allclose": {
                    "matched": int(per_output_allclose[output_name]["matched"]),
                    "total": int(per_output_allclose[output_name]["total"]),
                    "pass": (
                        per_output_allclose[output_name]["matched"]
                        == per_output_allclose[output_name]["total"]
                    ),
                },
                **per_output_metrics[output_name].to_dict(),
            }
            for output_name in final_partition_outputs
        },
    }

    os.makedirs(os.path.dirname(output_report_path) or ".", exist_ok=True)
    with open(output_report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    if bool(fail_on_threshold) and not evaluation_pass:
        raise RuntimeError(
            "Split-model evaluation failed thresholds. "
            f"report={output_report_path} "
            f"metrics={overall_metrics} "
            f"allclose={report['allclose_summary']}"
        )
    return report
