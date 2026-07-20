from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.mean_attention_orchestration import (
    run_mean_attention,
)
from onnx2tf.tflite_builder.passes.terminal_boundary_layout_orchestration import (
    run_terminal_boundary_layout,
)


RecoveryResults = Tuple[Dict[str, int], ...]


def run_terminal_boundary_mean_attention_cleanup(
    context: ModelIRPassContext,
    *,
    include_mean_attention: bool,
) -> Tuple[RecoveryResults, Optional[RecoveryResults]]:
    """Run terminal boundary recovery and its optional mean recovery."""

    boundary_results = run_terminal_boundary_layout(context)
    mean_results: Optional[RecoveryResults] = None
    if include_mean_attention:
        mean_results = run_mean_attention(
            context,
            include_conv_attention=False,
        )
    return boundary_results, mean_results
