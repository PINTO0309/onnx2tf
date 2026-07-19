from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.terminal_activation_bridge_orchestration import (
    run_terminal_activation_bridge_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_qkv_shape_attention_orchestration import (
    run_terminal_qkv_shape_attention_cleanup,
)


TerminalQKVActivationBridgeContext = ModelIRPassContext
CleanupResults = Tuple[Any, ...]


def run_terminal_qkv_activation_bridge_cleanup(
    context: TerminalQKVActivationBridgeContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run terminal QKV and activation-bridge cleanup in fixed order."""

    return (
        run_terminal_qkv_shape_attention_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
        run_terminal_activation_bridge_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
