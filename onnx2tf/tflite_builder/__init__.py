from __future__ import annotations

import os
from typing import Any, Dict

from onnx2tf.tflite_builder.ir import clone_model_ir_with_float16
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    build_op_coverage_report,
    lower_onnx_to_ir,
    write_op_coverage_report,
)
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.quantization import (
    build_dynamic_range_quantized_model_ir,
    build_full_integer_quantized_model_ir,
    build_full_integer_quantized_with_int16_act_model_ir,
    build_integer_quantized_model_ir,
    build_integer_quantized_with_int16_act_model_ir,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module
from onnx2tf.tflite_builder.split_planner import (
    DEFAULT_TFLITE_SPLIT_MAX_BYTES,
    DEFAULT_TFLITE_SPLIT_TARGET_BYTES,
    plan_contiguous_partitions_by_size,
    should_split_by_estimate,
    write_split_model_files_and_manifest,
    write_split_plan_report,
)
from onnx2tf.utils.common_functions import weights_export


def _reject_unsupported_quantization(**kwargs: Any) -> None:
    return


def _resolve_quantization_controls(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    calibration_method = kwargs.get(
        "flatbuffer_direct_calibration_method",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD", "max"),
    )
    calibration_percentile = kwargs.get(
        "flatbuffer_direct_calibration_percentile",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE", "99.99"),
    )
    quant_min_numel = kwargs.get(
        "flatbuffer_direct_quant_min_numel",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_NUMEL", "1"),
    )
    quant_min_abs_max = kwargs.get(
        "flatbuffer_direct_quant_min_abs_max",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX", "0.0"),
    )
    quant_scale_floor = kwargs.get(
        "flatbuffer_direct_quant_scale_floor",
        os.environ.get("ONNX2TF_FLATBUFFER_DIRECT_QUANT_SCALE_FLOOR", "1e-8"),
    )
    return {
        "calibration_method": str(calibration_method),
        "calibration_percentile": float(calibration_percentile),
        "min_numel": int(quant_min_numel),
        "min_abs_max": float(quant_min_abs_max),
        "scale_floor": float(quant_scale_floor),
    }


def export_tflite_model_flatbuffer_direct(**kwargs: Any) -> Dict[str, Any]:
    _reject_unsupported_quantization(**kwargs)

    output_folder_path = kwargs.get("output_folder_path", "saved_model")
    output_file_name = kwargs.get("output_file_name", "model")
    onnx_graph = kwargs.get("onnx_graph", None)
    output_weights = bool(kwargs.get("output_weights", False))
    quant_type = kwargs.get("quant_type", "per-channel")
    input_quant_dtype = kwargs.get("input_quant_dtype", "int8")
    output_quant_dtype = kwargs.get("output_quant_dtype", "int8")
    output_dynamic_range_quantized_tflite = bool(
        kwargs.get("output_dynamic_range_quantized_tflite", False)
    )
    output_integer_quantized_tflite = bool(
        kwargs.get("output_integer_quantized_tflite", False)
    )
    auto_split_tflite_by_size = bool(
        kwargs.get("auto_split_tflite_by_size", False)
    )
    report_op_coverage = bool(kwargs.get("report_op_coverage", False))
    tflite_split_max_bytes = int(
        kwargs.get(
            "tflite_split_max_bytes",
            os.environ.get(
                "ONNX2TF_FLATBUFFER_DIRECT_SPLIT_MAX_BYTES",
                str(DEFAULT_TFLITE_SPLIT_MAX_BYTES),
            ),
        )
    )
    tflite_split_target_bytes = int(
        kwargs.get(
            "tflite_split_target_bytes",
            os.environ.get(
                "ONNX2TF_FLATBUFFER_DIRECT_SPLIT_TARGET_BYTES",
                str(DEFAULT_TFLITE_SPLIT_TARGET_BYTES),
            ),
        )
    )
    quant_controls = _resolve_quantization_controls(kwargs)

    if onnx_graph is None:
        raise ValueError(
            "onnx_graph is required for tflite_backend='flatbuffer_direct'."
        )

    os.makedirs(output_folder_path, exist_ok=True)
    schema_tflite = load_schema_module(output_folder_path)

    op_coverage_report_path = None
    if report_op_coverage:
        op_coverage_report_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_op_coverage_report.json",
        )

    def _write_coverage_report(conversion_error: str | None) -> None:
        if not report_op_coverage or op_coverage_report_path is None:
            return
        report = build_op_coverage_report(
            onnx_graph=onnx_graph,
            output_file_name=output_file_name,
            conversion_error=conversion_error,
        )
        write_op_coverage_report(
            report=report,
            output_report_path=op_coverage_report_path,
        )

    try:
        model_ir = lower_onnx_to_ir(
            onnx_graph=onnx_graph,
            output_file_name=output_file_name,
        )
    except Exception as ex:
        try:
            _write_coverage_report(str(ex))
        except Exception:
            pass
        raise

    _write_coverage_report(None)

    split_plan_report_path = None
    split_required_by_estimate = False
    split_plan_total_estimated_bytes = None
    split_manifest_path = None
    split_partition_paths = None
    split_partition_count = 0
    if auto_split_tflite_by_size:
        split_plan_report = plan_contiguous_partitions_by_size(
            model_ir=model_ir,
            target_max_bytes=tflite_split_target_bytes,
            hard_max_bytes=tflite_split_max_bytes,
            schema_tflite=schema_tflite,
        )
        split_required_by_estimate = bool(should_split_by_estimate(split_plan_report))
        split_plan_total_estimated_bytes = int(
            split_plan_report.get("total_estimated_bytes", 0)
        )
        split_plan_report_path = write_split_plan_report(
            report=split_plan_report,
            output_report_path=os.path.join(
                output_folder_path,
                f"{output_file_name}_split_plan.json",
            ),
        )
        if split_required_by_estimate:
            from ai_edge_litert.interpreter import Interpreter

            def _validate_split_tflite_loadable(tflite_path: str) -> None:
                interpreter = Interpreter(model_path=tflite_path)
                interpreter.allocate_tensors()

            split_outputs = write_split_model_files_and_manifest(
                schema_tflite=schema_tflite,
                model_ir=model_ir,
                plan_report=split_plan_report,
                output_folder_path=output_folder_path,
                output_file_name=output_file_name,
                tflite_loader_validator=_validate_split_tflite_loadable,
            )
            split_manifest_path = split_outputs["split_manifest_path"]
            split_partition_paths = split_outputs["split_partition_paths"]
            split_partition_count = int(split_outputs["split_partition_count"])

    float32_path = os.path.join(output_folder_path, f"{output_file_name}_float32.tflite")
    write_model_file(
        schema_tflite=schema_tflite,
        model_ir=model_ir,
        output_tflite_path=float32_path,
    )

    model_ir_fp16 = clone_model_ir_with_float16(model_ir)
    float16_path = os.path.join(output_folder_path, f"{output_file_name}_float16.tflite")
    write_model_file(
        schema_tflite=schema_tflite,
        model_ir=model_ir_fp16,
        output_tflite_path=float16_path,
    )

    dynamic_range_path = None
    if output_dynamic_range_quantized_tflite:
        dynamic_model_ir = build_dynamic_range_quantized_model_ir(
            model_ir,
            quant_type=str(quant_type),
            calibration_method=quant_controls["calibration_method"],
            calibration_percentile=quant_controls["calibration_percentile"],
            min_numel=quant_controls["min_numel"],
            min_abs_max=quant_controls["min_abs_max"],
            scale_floor=quant_controls["scale_floor"],
        )
        dynamic_range_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_dynamic_range_quant.tflite",
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=dynamic_model_ir,
            output_tflite_path=dynamic_range_path,
        )

    integer_quant_path = None
    full_integer_quant_path = None
    if output_integer_quantized_tflite:
        integer_model_ir = build_integer_quantized_model_ir(
            model_ir,
            quant_type=str(quant_type),
            calibration_method=quant_controls["calibration_method"],
            calibration_percentile=quant_controls["calibration_percentile"],
            min_numel=quant_controls["min_numel"],
            min_abs_max=quant_controls["min_abs_max"],
            scale_floor=quant_controls["scale_floor"],
        )
        integer_quant_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_integer_quant.tflite",
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=integer_model_ir,
            output_tflite_path=integer_quant_path,
        )

        full_integer_model_ir = build_full_integer_quantized_model_ir(
            model_ir,
            quant_type=str(quant_type),
            input_quant_dtype=str(input_quant_dtype),
            output_quant_dtype=str(output_quant_dtype),
            calibration_method=quant_controls["calibration_method"],
            calibration_percentile=quant_controls["calibration_percentile"],
            min_numel=quant_controls["min_numel"],
            min_abs_max=quant_controls["min_abs_max"],
            scale_floor=quant_controls["scale_floor"],
        )
        full_integer_quant_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_full_integer_quant.tflite",
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=full_integer_model_ir,
            output_tflite_path=full_integer_quant_path,
        )

        integer_quant_with_int16_act_model_ir = build_integer_quantized_with_int16_act_model_ir(
            model_ir,
            quant_type=str(quant_type),
            calibration_method=quant_controls["calibration_method"],
            calibration_percentile=quant_controls["calibration_percentile"],
            min_numel=quant_controls["min_numel"],
            min_abs_max=quant_controls["min_abs_max"],
            scale_floor=quant_controls["scale_floor"],
        )
        integer_quant_with_int16_act_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_integer_quant_with_int16_act.tflite",
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=integer_quant_with_int16_act_model_ir,
            output_tflite_path=integer_quant_with_int16_act_path,
        )

        full_integer_quant_with_int16_act_model_ir = build_full_integer_quantized_with_int16_act_model_ir(
            model_ir,
            quant_type=str(quant_type),
            calibration_method=quant_controls["calibration_method"],
            calibration_percentile=quant_controls["calibration_percentile"],
            min_numel=quant_controls["min_numel"],
            min_abs_max=quant_controls["min_abs_max"],
            scale_floor=quant_controls["scale_floor"],
        )
        full_integer_quant_with_int16_act_path = os.path.join(
            output_folder_path,
            f"{output_file_name}_full_integer_quant_with_int16_act.tflite",
        )
        write_model_file(
            schema_tflite=schema_tflite,
            model_ir=full_integer_quant_with_int16_act_model_ir,
            output_tflite_path=full_integer_quant_with_int16_act_path,
        )
    else:
        integer_quant_with_int16_act_path = None
        full_integer_quant_with_int16_act_path = None

    if output_weights:
        weights_export(
            extract_target_tflite_file_path=float32_path,
            output_weights_file_path=os.path.join(
                output_folder_path,
                f"{output_file_name}_float32_weights.h5",
            ),
        )
        weights_export(
            extract_target_tflite_file_path=float16_path,
            output_weights_file_path=os.path.join(
                output_folder_path,
                f"{output_file_name}_float16_weights.h5",
            ),
        )
        if dynamic_range_path is not None:
            weights_export(
                extract_target_tflite_file_path=dynamic_range_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_dynamic_range_quant_weights.h5",
                ),
            )
        if integer_quant_path is not None:
            weights_export(
                extract_target_tflite_file_path=integer_quant_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_integer_quant_weights.h5",
                ),
            )
        if full_integer_quant_path is not None:
            weights_export(
                extract_target_tflite_file_path=full_integer_quant_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_full_integer_quant_weights.h5",
                ),
            )
        if integer_quant_with_int16_act_path is not None:
            weights_export(
                extract_target_tflite_file_path=integer_quant_with_int16_act_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_integer_quant_with_int16_act_weights.h5",
                ),
            )
        if full_integer_quant_with_int16_act_path is not None:
            weights_export(
                extract_target_tflite_file_path=full_integer_quant_with_int16_act_path,
                output_weights_file_path=os.path.join(
                    output_folder_path,
                    f"{output_file_name}_full_integer_quant_with_int16_act_weights.h5",
                ),
            )

    outputs: Dict[str, Any] = {
        "float32_tflite_path": float32_path,
        "float16_tflite_path": float16_path,
    }
    if dynamic_range_path is not None:
        outputs["dynamic_range_quant_tflite_path"] = dynamic_range_path
    if integer_quant_path is not None:
        outputs["integer_quant_tflite_path"] = integer_quant_path
    if full_integer_quant_path is not None:
        outputs["full_integer_quant_tflite_path"] = full_integer_quant_path
    if integer_quant_with_int16_act_path is not None:
        outputs["integer_quant_with_int16_act_tflite_path"] = integer_quant_with_int16_act_path
    if full_integer_quant_with_int16_act_path is not None:
        outputs["full_integer_quant_with_int16_act_tflite_path"] = full_integer_quant_with_int16_act_path
    if split_plan_report_path is not None:
        outputs["split_plan_report_path"] = split_plan_report_path
        outputs["split_required_by_estimate"] = bool(split_required_by_estimate)
        outputs["split_plan_total_estimated_bytes"] = int(split_plan_total_estimated_bytes)
    if split_manifest_path is not None:
        outputs["split_manifest_path"] = split_manifest_path
    if split_partition_paths is not None:
        outputs["split_partition_paths"] = split_partition_paths
        outputs["split_partition_count"] = int(split_partition_count)
    if op_coverage_report_path is not None:
        outputs["op_coverage_report_path"] = op_coverage_report_path
    return outputs
