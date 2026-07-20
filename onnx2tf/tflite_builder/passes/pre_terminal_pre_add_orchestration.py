from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.pre_add_layout import (
    optimize_transpose_pre_add_nhwc_chains,
)


PRE_TERMINAL_PRE_ADD_PASS_IDS = (
    "_optimize_transpose_pre_add_nhwc_chains",
)


PreTerminalPreAddContext = ModelIRPassContext


def run_pre_terminal_pre_add_cleanup(
    context: PreTerminalPreAddContext,
) -> Dict[str, int]:
    """Run pre-terminal pre-add cleanup and retain prune-only evidence."""

    initial_tensor_count = len(context.model_ir.tensors)
    return {
        **optimize_transpose_pre_add_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        "pruned_unused_tensors": max(
            0,
            int(initial_tensor_count - len(context.model_ir.tensors)),
        ),
    }
