from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.terminal_layout_shape_orchestration import (
    run_terminal_layout_shape_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_qkv_activation_bridge_orchestration import (
    run_terminal_qkv_activation_bridge_cleanup,
)


TerminalQKVActivationLayoutShapeContext = ModelIRPassContext
CleanupResults = Tuple[Any, ...]


def run_terminal_qkv_activation_layout_shape_cleanup(
    context: TerminalQKVActivationLayoutShapeContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run terminal QKV/activation and layout/shape cleanup in fixed order."""

    return (
        run_terminal_qkv_activation_bridge_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
        run_terminal_layout_shape_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
