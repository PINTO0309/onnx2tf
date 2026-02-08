from __future__ import annotations

import os
from typing import Any, Dict

from onnx2tf.tflite_builder.ir import clone_model_ir_with_float16
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.schema_loader import load_schema_module
from onnx2tf.utils.common_functions import weights_export


def _reject_unsupported_quantization(**kwargs: Any) -> None:
    if kwargs.get("output_dynamic_range_quantized_tflite", False):
        raise NotImplementedError(
            "flatbuffer_direct does not support dynamic-range quantized tflite yet. "
            "Use tflite_backend='tf_converter' for quantized export."
        )
    if kwargs.get("output_integer_quantized_tflite", False):
        raise NotImplementedError(
            "flatbuffer_direct does not support integer-quantized tflite yet. "
            "Use tflite_backend='tf_converter' for quantized export."
        )


def export_tflite_model_flatbuffer_direct(**kwargs: Any) -> Dict[str, str]:
    _reject_unsupported_quantization(**kwargs)

    output_folder_path = kwargs.get("output_folder_path", "saved_model")
    output_file_name = kwargs.get("output_file_name", "model")
    onnx_graph = kwargs.get("onnx_graph", None)
    output_weights = bool(kwargs.get("output_weights", False))

    if onnx_graph is None:
        raise ValueError(
            "onnx_graph is required for tflite_backend='flatbuffer_direct'."
        )

    os.makedirs(output_folder_path, exist_ok=True)
    schema_tflite = load_schema_module(output_folder_path)

    model_ir = lower_onnx_to_ir(
        onnx_graph=onnx_graph,
        output_file_name=output_file_name,
    )

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

    return {
        "float32_tflite_path": float32_path,
        "float16_tflite_path": float16_path,
    }
