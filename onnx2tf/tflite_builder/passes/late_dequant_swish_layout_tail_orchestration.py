from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.late_dequant_hardsigmoid_unary_orchestration import (
    run_late_dequant_hardsigmoid_unary_cleanup,
)
from onnx2tf.tflite_builder.passes.late_swish_layout_tail_orchestration import (
    run_late_swish_layout_tail_cleanup,
)


CleanupResults = Tuple[Any, ...]


def run_late_dequant_swish_layout_tail_cleanup(
    context: ModelIRPassContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[CleanupResults, CleanupResults]:
    """Run late dequant and swish/layout-tail cleanup in fixed order."""

    return (
        run_late_dequant_hardsigmoid_unary_cleanup(context),
        run_late_swish_layout_tail_cleanup(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
    )
