from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.conv1d_instance_norm_layout import (
    _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.conv1d_tencoder_layout import (
    _optimize_tencoder_add_expand_transpose_conv_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import (
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains,
    _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains,
    _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.decoder_deconv_layout import (
    _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input,
)
from onnx2tf.tflite_builder.passes.terminal_squeeze_mean_layout import (
    _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains,
)


LATE_CONV1D_DECODER_LAYOUT_PASS_IDS = (
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains",
    "_optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains",
    "_optimize_tencoder_add_expand_transpose_conv_nhwc_chains",
    "_optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains",
    "_optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input",
    "_optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains",
)


LateConv1DDecoderLayoutContext = ModelIRPassContext


def run_late_conv1d_decoder_layout_cleanup(
    context: LateConv1DDecoderLayoutContext,
) -> Tuple[Dict[str, int], ...]:
    """Run late Conv1D and decoder layout repairs in their fixed order."""

    return (
        _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
            context.model_ir,
            layout_state=context.layout_state,
        ),
        _optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains(
            context.model_ir,
            layout_state=context.layout_state,
        ),
    )
