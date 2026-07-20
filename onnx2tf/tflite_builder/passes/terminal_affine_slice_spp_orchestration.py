from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.late_spp_concat_unary_conv_orchestration import (
    run_late_spp_concat_unary_conv_summary,
)
from onnx2tf.tflite_builder.passes.stridedslice_pad_concat_bridge_layout import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.terminal_affine_concat_split_recovery_orchestration import (
    run_terminal_affine_concat_split_recovery_summary,
)


TerminalAffineSliceSPPContext = ModelIRPassContext


def run_terminal_affine_slice_spp_cleanup(
    context: TerminalAffineSliceSPPContext,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    """Run the terminal affine, Slice bridge, and SPP tail in fixed order."""

    return (
        run_terminal_affine_concat_split_recovery_summary(context),
        _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains(
            context.model_ir
        ),
        run_late_spp_concat_unary_conv_summary(context),
    )
