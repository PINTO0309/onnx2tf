from __future__ import annotations

import os
from typing import Any, Dict

from onnx2tf.tflite_builder.ir import clone_model_ir_with_float16
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.quantization import (
    build_dynamic_range_quantized_model_ir,
    build_full_integer_quantized_model_ir,
    build_integer_quantized_model_ir,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module
from onnx2tf.utils.common_functions import weights_export


def _reject_unsupported_quantization(**kwargs: Any) -> None:
    return


def export_tflite_model_flatbuffer_direct(**kwargs: Any) -> Dict[str, str]:
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

    dynamic_range_path = None
    if output_dynamic_range_quantized_tflite:
        dynamic_model_ir = build_dynamic_range_quantized_model_ir(
            model_ir,
            quant_type=str(quant_type),
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

    outputs: Dict[str, str] = {
        "float32_tflite_path": float32_path,
        "float16_tflite_path": float16_path,
    }
    if dynamic_range_path is not None:
        outputs["dynamic_range_quant_tflite_path"] = dynamic_range_path
    if integer_quant_path is not None:
        outputs["integer_quant_tflite_path"] = integer_quant_path
    if full_integer_quant_path is not None:
        outputs["full_integer_quant_tflite_path"] = full_integer_quant_path
    return outputs
