from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.axis3_const_concat_layout import (
    run_axis3_const_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dequant_concat_quantize_layout import (
    run_dequant_concat_quantize_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.layernorm_layout import (
    run_layernorm_statistics_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)


LATE_CONCAT_LAYOUT_PASS_IDS = (
    "run_axis3_const_concat_layout_cleanup",
    "run_dequant_concat_quantize_layout_cleanup",
    "run_layernorm_statistics_layout_cleanup",
    "run_layout_transpose_cleanup",
)


LateConcatLayoutContext = ModelIRPassContext


def run_late_concat_layout_cleanup(
    context: LateConcatLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run the adjacent late Concat/layout groups with one pass state."""

    state_scope = ModelIRPassStateScope(
        context.model_ir,
        layout_state=context.layout_state,
    )
    return (
        run_axis3_const_concat_layout_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
            state_scope=state_scope,
        ),
        run_dequant_concat_quantize_layout_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
            state_scope=state_scope,
        ),
        run_layernorm_statistics_layout_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
            state_scope=state_scope,
        ),
        run_layout_transpose_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
            state_scope=state_scope,
        ),
    )
