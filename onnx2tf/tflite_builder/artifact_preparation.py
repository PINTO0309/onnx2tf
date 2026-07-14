from __future__ import annotations

from typing import Optional

from onnx2tf.tflite_builder.ir import ModelIR, clone_model_ir_with_float32


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
