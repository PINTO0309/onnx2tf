from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.late_dequant_unary_fanout_orchestration import (
    run_late_dequant_unary_fanout,
)
from onnx2tf.tflite_builder.passes.quantized_activation import (
    optimize_transpose_dequant_hardsigmoid_quantize_bridges,
)


LateDequantHardsigmoidUnaryContext = ModelIRPassContext
NestedCleanupResults = Tuple[Dict[str, int], ...]


def run_late_dequant_hardsigmoid_unary_cleanup(
    context: LateDequantHardsigmoidUnaryContext,
) -> Tuple[Dict[str, int], NestedCleanupResults]:
    """Run late hard-sigmoid bridge and unary/fan-out cleanup in fixed order."""

    return (
        optimize_transpose_dequant_hardsigmoid_quantize_bridges(
            context.model_ir
        ),
        run_late_dequant_unary_fanout(context),
    )
