from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    run_channel_shuffle_gather,
)
from onnx2tf.tflite_builder.passes.late_attention_layout_orchestration import (
    run_late_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_reshape_layout_orchestration import (
    run_late_reshape_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_window_layout_orchestration import (
    run_late_window_layout_cleanup,
)


LateReshapeShuffleAttentionWindowContext = ModelIRPassContext
LayoutCleanupResults = Tuple[Dict[str, int], ...]


def run_late_reshape_shuffle_attention_window_cleanup(
    context: LateReshapeShuffleAttentionWindowContext,
) -> Tuple[
    LayoutCleanupResults,
    LayoutCleanupResults,
    LayoutCleanupResults,
    LayoutCleanupResults,
]:
    """Run the complete late reshape-to-window layout stage in fixed order."""

    return (
        run_late_reshape_layout_cleanup(context),
        run_channel_shuffle_gather(
            context,
            include_two_way_shuffle=False,
            include_nhwc_shuffle=False,
        ),
        run_late_attention_layout_cleanup(context),
        run_late_window_layout_cleanup(context),
    )
