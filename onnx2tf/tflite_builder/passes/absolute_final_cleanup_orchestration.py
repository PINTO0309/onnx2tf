from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.absolute_final_affine_instancenorm_orchestration import (
    run_absolute_final_affine_instancenorm_cleanup,
)
from onnx2tf.tflite_builder.passes.absolute_final_normalization_attention_orchestration import (
    run_absolute_final_normalization_attention_rank1_cleanup,
)
from onnx2tf.tflite_builder.passes.static_shape_signature_sanitization import (
    run_boundary_shape_signature_cleanup,
)


AbsoluteFinalCleanupContext = ModelIRPassContext


def run_absolute_final_cleanup(
    context: AbsoluteFinalCleanupContext,
) -> Tuple[
    Tuple[Dict[str, int], Dict[str, int]],
    Tuple[Dict[str, int], Dict[str, int]],
    Tuple[Tuple[Dict[str, int], ...], Dict[str, int]],
]:
    return (
        run_boundary_shape_signature_cleanup(context.model_ir),
        run_absolute_final_affine_instancenorm_cleanup(context),
        run_absolute_final_normalization_attention_rank1_cleanup(context),
    )
