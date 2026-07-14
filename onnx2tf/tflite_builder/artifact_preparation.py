from __future__ import annotations

import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Optional

from onnx2tf.tflite_builder.ir import ModelIR, clone_model_ir_with_float32


_QUANTIZATION_CONTROL_SPECS = (
    (
        "calibration_method",
        "flatbuffer_direct_calibration_method",
        "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD",
        "max",
        str,
    ),
    (
        "calibration_percentile",
        "flatbuffer_direct_calibration_percentile",
        "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE",
        "99.99",
        float,
    ),
    (
        "min_numel",
        "flatbuffer_direct_quant_min_numel",
        "ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_NUMEL",
        "1",
        int,
    ),
    (
        "min_abs_max",
        "flatbuffer_direct_quant_min_abs_max",
        "ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX",
        "0.0",
        float,
    ),
    (
        "scale_floor",
        "flatbuffer_direct_quant_scale_floor",
        "ONNX2TF_FLATBUFFER_DIRECT_QUANT_SCALE_FLOOR",
        "1e-8",
        float,
    ),
)


def _option_or_environment(
    options: Mapping[str, Any],
    *,
    option_name: str,
    environment_name: str,
    default: Any,
) -> Any:
    return options.get(
        option_name,
        os.environ.get(environment_name, default),
    )


@dataclass(frozen=True)
class ArtifactExecutionControls:
    split_max_bytes: Optional[int]
    split_target_bytes: Optional[int]
    quantization: Optional[Mapping[str, Any]]


def resolve_requested_artifact_controls(
    options: Mapping[str, Any],
    *,
    split_plan_requested: bool,
    quantization_requested: bool,
    default_split_max_bytes: int,
    default_split_target_bytes: int,
) -> ArtifactExecutionControls:
    split_max_bytes: Optional[int] = None
    split_target_bytes: Optional[int] = None
    if split_plan_requested:
        split_max_bytes = int(
            _option_or_environment(
                options,
                option_name="tflite_split_max_bytes",
                environment_name="ONNX2TF_FLATBUFFER_DIRECT_SPLIT_MAX_BYTES",
                default=str(default_split_max_bytes),
            )
        )
        split_target_bytes = int(
            _option_or_environment(
                options,
                option_name="tflite_split_target_bytes",
                environment_name="ONNX2TF_FLATBUFFER_DIRECT_SPLIT_TARGET_BYTES",
                default=str(default_split_target_bytes),
            )
        )

    quantization: Optional[Mapping[str, Any]] = None
    if quantization_requested:
        quantization = MappingProxyType(
            {
                key: cast(
                    _option_or_environment(
                        options,
                        option_name=option_name,
                        environment_name=environment_name,
                        default=default,
                    )
                )
                for key, option_name, environment_name, default, cast in (
                    _QUANTIZATION_CONTROL_SPECS
                )
            }
        )

    return ArtifactExecutionControls(
        split_max_bytes=split_max_bytes,
        split_target_bytes=split_target_bytes,
        quantization=quantization,
    )


def isolate_float32_model_ir_for_tflite_write(
    model_ir: ModelIR,
    *,
    split_manifest_path: Optional[str],
    output_saved_model_from_model_ir: bool,
    output_pytorch_from_model_ir: bool,
) -> ModelIR:
    preserve_for_later_exporters = split_manifest_path is None and (
        bool(output_saved_model_from_model_ir) or bool(output_pytorch_from_model_ir)
    )
    if preserve_for_later_exporters:
        return clone_model_ir_with_float32(model_ir)
    return model_ir
