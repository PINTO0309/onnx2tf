from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.concat_input_adapter_layout import (
    optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
)
from onnx2tf.tflite_builder.passes.concat_unary_conv_layout import (
    run_concat_unary_conv_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.shape_extract_layout import (
    optimize_transpose_shape_extract_nhwc_to_nchw_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    optimize_transpose_relu_split_all_outputs_to_nhwc_chains,
    optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_mixed_concat_layout import (
    optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains,
)


TERMINAL_CONCAT_BRIDGE_LAYOUT_PASS_IDS = (
    "_optimize_transpose_relu_split_all_outputs_to_nhwc_chains",
    "_optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains",
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "run_concat_unary_conv_layout_cleanup",
    "_optimize_transpose_shape_extract_nhwc_to_nchw_chains",
)


TerminalConcatBridgeLayoutContext = ModelIRPassContext


def run_terminal_concat_bridge_layout_cleanup(
    context: TerminalConcatBridgeLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run terminal Split/Concat bridge repairs in their fixed order."""

    return (
        optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        run_concat_unary_conv_layout_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        optimize_transpose_shape_extract_nhwc_to_nchw_chains(
            context.model_ir,
        ),
    )
