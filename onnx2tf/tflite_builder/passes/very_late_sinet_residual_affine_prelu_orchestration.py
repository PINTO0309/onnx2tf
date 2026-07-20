from __future__ import annotations

from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.passes.residual_affine_prelu_layout import (
    optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.very_late_sinet_recovery_tail_orchestration import (
    RecoveryResults,
    run_very_late_sinet_recovery_tail_cleanup,
)


SINetResults = Tuple[RecoveryResults, Any]


def run_very_late_sinet_residual_affine_prelu_cleanup(
    context: SINetTerminalLayoutRecoveryContext,
) -> Tuple[SINetResults, Dict[str, int]]:
    """Run very-late SiNet and residual-affine/PReLU cleanup in order."""

    sinet_results = run_very_late_sinet_recovery_tail_cleanup(context)
    prelu_results = optimize_transpose_pre_add_mul_add_prelu_nhwc_chains(
        context.pass_context.model_ir
    )
    return sinet_results, prelu_results
