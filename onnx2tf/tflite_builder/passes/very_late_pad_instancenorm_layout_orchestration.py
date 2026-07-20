from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.instance_norm_dual_stats_layout import (
    optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.instance_norm_post_bias_layout import (
    optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.instance_norm_residual_mul_concat_layout import (
    optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pad_layout import run_pad_layout_cleanup


VERY_LATE_PAD_INSTANCENORM_LAYOUT_PASS_IDS = (
    "run_pad_layout_cleanup",
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
)


VeryLatePadInstanceNormLayoutContext = ModelIRPassContext


def run_very_late_pad_instancenorm_layout_cleanup(
    context: VeryLatePadInstanceNormLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run very-late Pad then InstanceNorm layout repairs in fixed order."""

    return (
        run_pad_layout_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
