from __future__ import annotations

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.late_binary_layout_recovery import (
    run_late_binary_layout_recovery,
)


OPTIONAL_LATE_BINARY_LAYOUT_RECOVERY_PASS_IDS = (
    "run_late_binary_layout_recovery",
)


OptionalLateBinaryLayoutRecoveryContext = ModelIRPassContext


def run_optional_late_binary_layout_recovery_cleanup(
    context: OptionalLateBinaryLayoutRecoveryContext,
    *,
    enabled: bool,
    include_layout_transpose: bool,
) -> bool:
    """Run optional late-binary layout recovery and report any mutation."""

    if not enabled:
        return False
    stats = run_late_binary_layout_recovery(
        context.model_ir,
        include_layout_transpose=include_layout_transpose,
        layout_state=context.layout_state,
        diagnostics=context.diagnostics,
    )
    return any(int(value) > 0 for value in stats.values())
