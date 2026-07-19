from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.binary_layout_convergence import (
    run_indexed_binary_layout_convergence,
)
from onnx2tf.tflite_builder.passes.high_rank_binary import (
    coalesce_static_high_rank_binary_operators,
)
from onnx2tf.tflite_builder.passes.static_shape_signature_sanitization import (
    realign_dynamic_boundary_shape_signature_map,
)


TerminalStabilizationContext = ModelIRPassContext


def run_terminal_stabilization_cleanup(
    context: TerminalStabilizationContext,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    return (
        run_indexed_binary_layout_convergence(context.model_ir),
        coalesce_static_high_rank_binary_operators(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        realign_dynamic_boundary_shape_signature_map(context.model_ir),
    )
