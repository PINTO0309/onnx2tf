from __future__ import annotations

from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
    run_terminal_slice_concat_recovery,
)


TerminalSliceConcatRecoveryResults = Tuple[Any, ...]


def run_terminal_slice_concat_boundary_stridedslice_cleanup(
    context: TerminalSliceConcatRecoveryContext,
) -> Tuple[TerminalSliceConcatRecoveryResults, Dict[str, int]]:
    """Run terminal Slice/Concat recovery and boundary cleanup in order."""

    recovery_results = run_terminal_slice_concat_recovery(context)
    boundary_results = (
        _optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks(
            context.pass_context.model_ir,
            layout_state=context.pass_context.layout_state,
        )
    )
    return recovery_results, boundary_results
