from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.expand_squeeze_reshape import (
    replace_expand_dims_and_squeeze_with_reshape,
)
from onnx2tf.tflite_builder.passes.late_layout_mean_spp_gather_constant_cast_orchestration import (
    run_late_layout_mean_spp_gather_constant_cast_summary,
)
from onnx2tf.tflite_builder.passes.pre_concat_nhwc_layout import (
    optimize_transpose_pre_concat_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.shape_extract_layout import (
    optimize_transpose_shape_extract_nhwc_to_nchw_chains,
)


TerminalLayoutShapeContext = ModelIRPassContext


def run_terminal_layout_shape_cleanup(
    context: TerminalLayoutShapeContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Run the terminal layout/shape tail before static reconciliation."""

    return (
        optimize_transpose_pre_concat_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        optimize_transpose_shape_extract_nhwc_to_nchw_chains(
            context.model_ir
        ),
        run_late_layout_mean_spp_gather_constant_cast_summary(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
        replace_expand_dims_and_squeeze_with_reshape(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
