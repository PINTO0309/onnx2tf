from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LayoutRecoveryContext,
    run_layout_recovery_prefix,
)
from onnx2tf.tflite_builder.passes.qlinear_recovery_orchestration import (
    run_qlinear_mean_concat_recovery,
)


CleanupResults = Tuple[Any, ...]


def run_layout_pass_set_2_qlinear_layout_recovery(
    context: LayoutRecoveryContext,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run QLinear recovery before the second layout-recovery prefix."""

    return (
        run_qlinear_mean_concat_recovery(context.pass_context),
        run_layout_recovery_prefix(context),
    )
