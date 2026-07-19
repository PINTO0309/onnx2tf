from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.final_boundary_slice_concat_orchestration import (
    run_final_boundary_slice_concat_cleanup,
)
from onnx2tf.tflite_builder.passes.indexed_final_shape_activation_convergence import (
    run_indexed_final_shape_activation_convergence,
)
from onnx2tf.tflite_builder.passes.late_reshape_shuffle_attention_window_orchestration import (
    run_late_reshape_shuffle_attention_window_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_slice_concat_recovery_orchestration import (
    TerminalSliceConcatRecoveryContext,
)


@dataclass(frozen=True)
class LateFinalShapeBoundaryContext:
    pass_context: ModelIRPassContext
    terminal_slice_concat_context: TerminalSliceConcatRecoveryContext


LateLayoutResults = Tuple[Tuple[Dict[str, int], ...], ...]
FinalBoundaryResults = Tuple[Any, ...]


def run_late_final_shape_boundary_cleanup(
    context: LateFinalShapeBoundaryContext,
) -> Tuple[LateLayoutResults, Dict[str, int], FinalBoundaryResults]:
    """Run late layout, shape convergence, and final boundary cleanup."""

    return (
        run_late_reshape_shuffle_attention_window_cleanup(
            context.pass_context,
        ),
        run_indexed_final_shape_activation_convergence(
            context.pass_context.model_ir,
            layout_state=context.pass_context.layout_state,
        ),
        run_final_boundary_slice_concat_cleanup(
            context.terminal_slice_concat_context,
        ),
    )
