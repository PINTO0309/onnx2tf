from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from onnx2tf.tflite_builder.passes.sinet_terminal_layout_recovery_orchestration import (
    SINetTerminalLayoutRecoveryContext,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_orchestration import (
    run_singleton_reshape,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_sinet_layout_orchestration import (
    run_terminal_clamp_sinet_layout_cleanup,
)


SingletonResults = Tuple[Dict[str, int], ...]
ClampSINetResults = Tuple[Tuple[Dict[str, int], ...], Tuple[Any, ...]]


def run_terminal_singleton_clamp_sinet_cleanup(
    context: SINetTerminalLayoutRecoveryContext,
    *,
    include_terminal_singleton: bool,
) -> Tuple[Optional[SingletonResults], ClampSINetResults]:
    """Run optional terminal singleton cleanup before Clamp/SiNet cleanup."""

    singleton_results: Optional[SingletonResults] = None
    if include_terminal_singleton:
        singleton_results = run_singleton_reshape(
            context.pass_context,
            include_layout_transpose=True,
            include_multi_branch_gate=True,
        )
    clamp_sinet_results = run_terminal_clamp_sinet_layout_cleanup(context)
    return singleton_results, clamp_sinet_results
