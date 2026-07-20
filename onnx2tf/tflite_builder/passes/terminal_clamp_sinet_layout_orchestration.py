from __future__ import annotations

from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
    run_sinet_terminal_layout_recovery,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_unary_relu_orchestration import (
    run_terminal_clamp_unary_relu,
)


CleanupResults = Tuple[Any, ...]


def run_terminal_clamp_sinet_layout_cleanup(
    context: SINetTerminalLayoutRecoveryContext,
) -> Tuple[Tuple[Dict[str, int], ...], CleanupResults]:
    """Run terminal Clamp and SiNet layout recovery in fixed order."""

    return (
        run_terminal_clamp_unary_relu(context.pass_context),
        run_sinet_terminal_layout_recovery(context),
    )
