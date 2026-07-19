from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.precision_cleanup_orchestration import (
    run_precision_cleanup_sequence,
)
from onnx2tf.tflite_builder.passes.unbound_input_repair_orchestration import (
    repair_unbound_nonconstant_operator_inputs_with_layout_transpose,
)


FallbackPrecisionUnboundContext = ModelIRPassContext
PrecisionCleanupResults = Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]


def run_fallback_precision_unbound_cleanup(
    context: FallbackPrecisionUnboundContext,
) -> Tuple[PrecisionCleanupResults, Dict[str, int]]:
    """Run fallback precision and unbound-input cleanup in fixed order."""

    return (
        run_precision_cleanup_sequence(context),
        repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
            context.model_ir,
        ),
    )
