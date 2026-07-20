from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    run_qkv_attention_summary,
)
from onnx2tf.tflite_builder.passes.shape_extract_layout import (
    optimize_transpose_shape_extract_nhwc_to_nchw_chains,
)


TerminalQKVShapeAttentionContext = ModelIRPassContext


def run_terminal_qkv_shape_attention_cleanup(
    context: TerminalQKVShapeAttentionContext,
    *,
    include_layout_transpose: bool,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """Run terminal QKV shape extraction and attention cleanup in order."""

    return (
        optimize_transpose_shape_extract_nhwc_to_nchw_chains(
            context.model_ir
        ),
        run_qkv_attention_summary(
            context,
            include_layout_transpose=include_layout_transpose,
            include_prefix=False,
        ),
    )
