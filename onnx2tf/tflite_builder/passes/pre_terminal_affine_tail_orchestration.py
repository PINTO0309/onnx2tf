from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_post_add_layout import (
    optimize_transpose_mul_posttranspose_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.stridedslice_pad_concat_bridge_layout import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains,
)


PRE_TERMINAL_AFFINE_TAIL_PASS_IDS = (
    "_optimize_transpose_mul_posttranspose_add_nhwc_chains",
    "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
)


PreTerminalAffineTailContext = ModelIRPassContext


def run_pre_terminal_affine_tail_cleanup(
    context: PreTerminalAffineTailContext,
) -> Tuple[Dict[str, int], ...]:
    """Run the pre-terminal affine tail repairs in fixed order."""

    return (
        optimize_transpose_mul_posttranspose_add_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            context.model_ir,
        ),
    )
