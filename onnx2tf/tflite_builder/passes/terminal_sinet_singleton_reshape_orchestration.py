from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    run_sinet_preadd_resize_recovery,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_orchestration import (
    run_singleton_reshape,
)


CleanupResults = Tuple[Dict[str, int], ...]


def run_terminal_sinet_singleton_reshape_cleanup(
    context: ModelIRPassContext,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run terminal SiNet and singleton-Reshape recovery in fixed order."""

    return (
        run_sinet_preadd_resize_recovery(context),
        run_singleton_reshape(
            context,
            include_duplicate_fanout=True,
            include_spatial_concat_post_transpose=False,
        ),
    )
