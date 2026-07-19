from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.hardswish_se_layout import (
    run_hardswish_se_layout_summary,
)
from onnx2tf.tflite_builder.passes.late_hard_activation_layout_orchestration import (
    run_late_hard_activation_layout_summary,
)
from onnx2tf.tflite_builder.passes.split_conv_concat_bridge_layout import (
    optimize_split_conv_concat_transpose_bridge_to_single_post_nchw,
)


TerminalActivationBridgeContext = ModelIRPassContext


def run_terminal_activation_bridge_cleanup(
    context: TerminalActivationBridgeContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Run the terminal Split, HardSwish-SE, and hard-activation tail."""

    return (
        optimize_split_conv_concat_transpose_bridge_to_single_post_nchw(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        run_hardswish_se_layout_summary(context.model_ir),
        run_late_hard_activation_layout_summary(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
