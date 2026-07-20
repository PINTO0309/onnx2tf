from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.se_layout import run_se_fc_layout_cleanup


NoLayoutFinalCleanupContext = ModelIRPassContext


def run_no_layout_final_cleanup(
    context: NoLayoutFinalCleanupContext,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    return (
        run_se_fc_layout_cleanup(
            context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        optimize_transpose_mul_add_const_prepost_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
