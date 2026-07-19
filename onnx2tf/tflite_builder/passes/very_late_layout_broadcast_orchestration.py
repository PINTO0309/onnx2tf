from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    repair_rank4_channelwise_broadcast_constants_to_runtime_layout,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)


VERY_LATE_LAYOUT_BROADCAST_PASS_IDS = (
    "run_layout_transpose_cleanup",
    "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout",
)


VeryLateLayoutBroadcastContext = ModelIRPassContext


def run_very_late_layout_broadcast_cleanup(
    context: VeryLateLayoutBroadcastContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[Optional[Dict[str, int]], Dict[str, int]]:
    """Run optional layout-Transpose then broadcast-constant cleanup."""

    layout_transpose_result: Optional[Dict[str, int]] = None
    if include_layout_transpose:
        layout_transpose_result = run_layout_transpose_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        )
    broadcast_result = (
        repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
            context.model_ir,
        )
    )
    return layout_transpose_result, broadcast_result
