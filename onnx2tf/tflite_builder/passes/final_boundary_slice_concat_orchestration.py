from __future__ import annotations

from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.passes.final_boundary_channel_layout_orchestration import (
    run_final_boundary_channel_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.final_slice_pre_concat_layout_orchestration import (
    run_final_slice_pre_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_concat_bridge_layout_orchestration import (
    run_terminal_concat_bridge_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
    run_terminal_slice_concat_recovery,
)


FinalBoundarySliceConcatContext = TerminalSliceConcatRecoveryContext
LayoutCleanupResults = Tuple[Dict[str, int], ...]


def run_final_boundary_slice_concat_cleanup(
    context: FinalBoundarySliceConcatContext,
) -> Tuple[
    LayoutCleanupResults,
    Tuple[Any, ...],
    LayoutCleanupResults,
    LayoutCleanupResults,
]:
    """Run final boundary and Slice/Concat cleanup in fixed order."""

    return (
        run_final_boundary_channel_layout_cleanup(context.pass_context),
        run_terminal_slice_concat_recovery(context),
        run_final_slice_pre_concat_layout_cleanup(context.pass_context),
        run_terminal_concat_bridge_layout_cleanup(context.pass_context),
    )
