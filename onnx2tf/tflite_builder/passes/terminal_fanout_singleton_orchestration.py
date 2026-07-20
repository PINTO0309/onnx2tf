from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_maxpool_reshape_orchestration import (
    run_terminal_singleton_maxpool_reshape,
)


SingletonResults = Tuple[Dict[str, int], ...]


def run_terminal_fanout_singleton_cleanup(
    context: ModelIRPassContext,
    *,
    include_elementwise_fanout: bool,
) -> Tuple[Optional[Dict[str, int]], SingletonResults]:
    """Run optional fan-out cleanup before terminal singleton cleanup."""

    fanout_results: Optional[Dict[str, int]] = None
    if include_elementwise_fanout:
        fanout_results = (
            optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
                context.model_ir
            )
        )
    singleton_results = run_terminal_singleton_maxpool_reshape(context)
    return fanout_results, singleton_results
