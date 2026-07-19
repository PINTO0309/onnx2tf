from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.late_conv1d_decoder_layout_orchestration import (
    run_late_conv1d_decoder_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    run_singleton_consecutive_reshape,
)
from onnx2tf.tflite_builder.passes.very_late_layout_broadcast_orchestration import (
    run_very_late_layout_broadcast_cleanup,
)
from onnx2tf.tflite_builder.passes.very_late_pad_instancenorm_layout_orchestration import (
    run_very_late_pad_instancenorm_layout_cleanup,
)


VeryLateLayoutTailContext = ModelIRPassContext
LayoutCleanupResults = Tuple[Dict[str, int], ...]
BroadcastCleanupResults = Tuple[Optional[Dict[str, int]], Dict[str, int]]


def run_very_late_layout_tail_cleanup(
    context: VeryLateLayoutTailContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[
    LayoutCleanupResults,
    LayoutCleanupResults,
    Tuple[Dict[str, int], Dict[str, int], Dict[str, int]],
    BroadcastCleanupResults,
]:
    """Run the complete very-late layout tail in fixed order."""

    return (
        run_late_conv1d_decoder_layout_cleanup(context),
        run_very_late_pad_instancenorm_layout_cleanup(context),
        run_singleton_consecutive_reshape(context),
        run_very_late_layout_broadcast_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
