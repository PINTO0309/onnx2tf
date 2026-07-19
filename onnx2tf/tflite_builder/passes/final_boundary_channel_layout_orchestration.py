from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.boundary_input_chains import (
    run_boundary_input_normalization_cleanup,
)
from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    _optimize_internal_transpose_channel_slice_nhwc_propagation_chains,
    _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains,
)


FINAL_BOUNDARY_CHANNEL_LAYOUT_PASS_IDS = (
    "run_boundary_input_normalization_cleanup",
    "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains",
    "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains",
)


FinalBoundaryChannelLayoutContext = ModelIRPassContext


def run_final_boundary_channel_layout_cleanup(
    context: FinalBoundaryChannelLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run final boundary/channel repairs with their fixed argument policy."""

    return (
        run_boundary_input_normalization_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        _optimize_internal_transpose_channel_slice_nhwc_propagation_chains(
            context.model_ir,
        ),
        _optimize_transpose_channel_slice_muladd_nhwc_bridge_chains(
            context.model_ir,
        ),
    )
