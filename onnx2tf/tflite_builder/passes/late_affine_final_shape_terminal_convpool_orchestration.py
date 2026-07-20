from __future__ import annotations

from typing import Dict, Optional, Tuple

from onnx2tf.tflite_builder.passes.convpool_output_passthrough_compat import (
    optimize_convpool_output_transpose_nhwc_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.late_affine_final_shape_terminal_orchestration import (
    LateAffineOptionalResults,
    LateFinalShapeTerminalResults,
    run_late_affine_final_shape_terminal_cleanup,
)
from onnx2tf.tflite_builder.passes.late_final_shape_boundary_orchestration import (
    LateFinalShapeBoundaryContext,
)


LateTerminalResults = Tuple[
    LateAffineOptionalResults,
    LateFinalShapeTerminalResults,
]


def run_late_affine_final_shape_terminal_convpool_cleanup(
    context: LateFinalShapeBoundaryContext,
    *,
    optimize_layout_transpose_chains: bool,
) -> Tuple[LateTerminalResults, Optional[Dict[str, int]]]:
    """Run late terminal cleanup with optional Conv/Pool passthrough."""

    late_results = run_late_affine_final_shape_terminal_cleanup(
        context,
        include_elementwise_fanout=optimize_layout_transpose_chains,
    )
    convpool_results: Optional[Dict[str, int]] = None
    if optimize_layout_transpose_chains:
        convpool_results = (
            optimize_convpool_output_transpose_nhwc_passthrough_chains(
                context.pass_context.model_ir
            )
        )
    return late_results, convpool_results
