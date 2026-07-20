from __future__ import annotations

from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    optimize_swish_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.very_late_layout_tail_orchestration import (
    run_very_late_layout_tail_cleanup,
)


LateSwishLayoutTailContext = ModelIRPassContext
LayoutTailResults = Tuple[Any, ...]


def run_late_swish_layout_tail_cleanup(
    context: LateSwishLayoutTailContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[Dict[str, int], LayoutTailResults]:
    """Run late swish and very-late layout-tail cleanup in fixed order."""

    return (
        optimize_swish_transpose_passthrough_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        run_very_late_layout_tail_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
