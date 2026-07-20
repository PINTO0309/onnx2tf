from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.passes.hardswish_se_layout import (
    optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_clamp_sinet_orchestration import (
    ClampSINetResults,
    SingletonResults,
    run_terminal_singleton_clamp_sinet_cleanup,
)


TerminalResults = Tuple[Optional[SingletonResults], ClampSINetResults]


def run_terminal_singleton_clamp_sinet_hardswish_cleanup(
    context: SINetTerminalLayoutRecoveryContext,
    *,
    include_terminal_singleton: bool,
) -> Tuple[TerminalResults, Dict[str, int]]:
    """Run terminal singleton/Clamp-SiNet and HardSwish cleanup in order."""

    terminal_results = run_terminal_singleton_clamp_sinet_cleanup(
        context,
        include_terminal_singleton=include_terminal_singleton,
    )
    hardswish_results = (
        optimize_transpose_hardswish_se_conv_hardsigmoid_mul_prepost_nhwc_chains(
            context.pass_context.model_ir
        )
    )
    return terminal_results, hardswish_results
