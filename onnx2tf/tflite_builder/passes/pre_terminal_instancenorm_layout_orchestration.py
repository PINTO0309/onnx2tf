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


PRE_TERMINAL_INSTANCENORM_LAYOUT_PASS_IDS = (
    "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains",
    "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains",
    "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains",
)


PreTerminalInstanceNormLayoutContext = ModelIRPassContext


def run_pre_terminal_instancenorm_layout_cleanup(
    context: PreTerminalInstanceNormLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run pre-terminal InstanceNorm layout repairs in fixed order."""

    return (
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
