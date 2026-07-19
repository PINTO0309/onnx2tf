from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_post_add_layout import (
    optimize_transpose_mul_posttranspose_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.recurrent_alias_repair_orchestration import (
    repair_orphan_recurrent_step_tensors_summary,
)
from onnx2tf.tflite_builder.passes.unbound_input_repair_orchestration import (
    repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
)
from onnx2tf.tflite_builder.passes.very_late_gather_constant_normalization_orchestration import (
    run_very_late_gather_constant_normalization_summary,
)


LateInputAffineNormalizationContext = ModelIRPassContext


def run_late_input_affine_normalization_cleanup(
    context: LateInputAffineNormalizationContext,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Run final input repair before affine and normalization cleanup."""

    return (
        repair_orphan_recurrent_step_tensors_summary(context.model_ir),
        repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
            context.model_ir
        ),
        optimize_transpose_mul_posttranspose_add_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        run_very_late_gather_constant_normalization_summary(context),
    )
