from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.passes.late_final_shape_boundary_orchestration import (
    FinalBoundaryResults,
    LateFinalShapeBoundaryContext,
    LateLayoutResults,
    run_late_final_shape_boundary_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_fanout_singleton_orchestration import (
    SingletonResults,
    run_terminal_fanout_singleton_cleanup,
)


LateFinalShapeResults = Tuple[
    LateLayoutResults,
    Dict[str, int],
    FinalBoundaryResults,
]
TerminalFanoutResults = Tuple[Optional[Dict[str, int]], SingletonResults]


def run_late_final_shape_terminal_fanout_cleanup(
    context: LateFinalShapeBoundaryContext,
    *,
    include_elementwise_fanout: bool,
) -> Tuple[LateFinalShapeResults, TerminalFanoutResults]:
    """Run final-shape cleanup before terminal fan-out convergence."""

    return (
        run_late_final_shape_boundary_cleanup(context),
        run_terminal_fanout_singleton_cleanup(
            context.pass_context,
            include_elementwise_fanout=include_elementwise_fanout,
        ),
    )
