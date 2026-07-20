from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.passes.indexed_final_shape_activation_convergence import (
    run_indexed_shape_convergence_cleanup,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.terminal_sinet_singleton_reshape_orchestration import (
    CleanupResults,
    run_terminal_sinet_singleton_reshape_cleanup,
)


TerminalResults = Tuple[CleanupResults, CleanupResults]


def run_terminal_sinet_singleton_reshape_convergence_cleanup(
    context: SINetTerminalLayoutRecoveryContext,
) -> Tuple[TerminalResults, Dict[str, int]]:
    """Run terminal SiNet/Reshape and indexed convergence in order."""

    terminal_results = run_terminal_sinet_singleton_reshape_cleanup(
        context.pass_context
    )
    convergence_results = run_indexed_shape_convergence_cleanup(
        context.pass_context.model_ir,
        layout_state=context.pass_context.layout_state,
    )
    return terminal_results, convergence_results
