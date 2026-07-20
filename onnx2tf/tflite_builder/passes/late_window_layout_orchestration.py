from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.window_partition_layout import (
    _optimize_window_partition_reshape_transpose_to_space_to_depth_chains,
    _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains,
)


LATE_WINDOW_LAYOUT_PASS_IDS = (
    "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains",
    "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains",
)


LateWindowLayoutContext = ModelIRPassContext


def run_late_window_layout_cleanup(
    context: LateWindowLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run adjacent late window-layout repairs in their fixed order."""

    return (
        _optimize_window_partition_reshape_transpose_to_space_to_depth_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
