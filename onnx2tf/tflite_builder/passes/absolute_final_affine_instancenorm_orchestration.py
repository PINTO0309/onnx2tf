from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_post_add_layout import (
    optimize_transpose_mul_posttranspose_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.instance_norm_post_bias_layout import (
    optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains,
)


AbsoluteFinalAffineInstanceNormContext = ModelIRPassContext


def run_absolute_final_affine_instancenorm_cleanup(
    context: AbsoluteFinalAffineInstanceNormContext,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    return (
        optimize_transpose_mul_posttranspose_add_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
