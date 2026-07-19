from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LayoutRecoveryContext,
    run_layout_reshape_attention_recovery_prefix,
)
from onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration import (
    run_qlinear_mean_concat_recovery,
)


CleanupResults = Tuple[Any, ...]


def run_layout_pass_set_1_qlinear_attention_recovery(
    context: LayoutRecoveryContext,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run QLinear recovery before the final layout/attention prefix."""

    return (
        run_qlinear_mean_concat_recovery(context.pass_context),
        run_layout_reshape_attention_recovery_prefix(context),
    )
