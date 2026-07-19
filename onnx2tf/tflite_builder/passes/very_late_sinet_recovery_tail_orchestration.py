from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
    run_sinet_terminal_layout_recovery,
)


RecoveryResults = Tuple[Any, ...]


def run_very_late_sinet_recovery_tail_cleanup(
    context: SINetTerminalLayoutRecoveryContext,
) -> Tuple[RecoveryResults, Any]:
    """Run the absolute-end SiNet recovery tail in fixed order."""

    return (
        run_sinet_terminal_layout_recovery(context),
        context.preadd_resize_recovery(),
    )
