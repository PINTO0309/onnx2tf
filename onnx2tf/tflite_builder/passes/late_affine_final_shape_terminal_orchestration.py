from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.passes.late_affine_optional_fanout_orchestration import (
    LateAffineResults,
    run_late_affine_optional_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.late_final_shape_boundary_orchestration import (
    LateFinalShapeBoundaryContext,
)
from onnx2tf.tflite_builder.passes.late_final_shape_terminal_fanout_orchestration import (
    LateFinalShapeResults,
    TerminalFanoutResults,
    run_late_final_shape_terminal_fanout_cleanup,
)


LateAffineOptionalResults = Tuple[
    LateAffineResults,
    Optional[Dict[str, int]],
]
LateFinalShapeTerminalResults = Tuple[
    LateFinalShapeResults,
    TerminalFanoutResults,
]


def run_late_affine_final_shape_terminal_cleanup(
    context: LateFinalShapeBoundaryContext,
    *,
    include_elementwise_fanout: bool,
) -> Tuple[LateAffineOptionalResults, LateFinalShapeTerminalResults]:
    """Run affine cleanup through final-shape terminal convergence."""

    return (
        run_late_affine_optional_fanout_cleanup(
            context.pass_context,
            include_elementwise_fanout=include_elementwise_fanout,
        ),
        run_late_final_shape_terminal_fanout_cleanup(
            context,
            include_elementwise_fanout=include_elementwise_fanout,
        ),
    )
