from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    run_qkv_attention,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    optimize_transpose_relu_split_all_outputs_to_nhwc_chains,
)


QKVResults = Tuple[Dict[str, int], ...]


def run_post_sinet_qkv_relu_split_all_cleanup(
    context: ModelIRPassContext,
) -> Tuple[QKVResults, Dict[str, int]]:
    """Run post-SiNet QKV and ReLU/Split-all cleanup in order."""

    qkv_results = run_qkv_attention(context)
    relu_results = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        context.model_ir,
        layout_state=context.layout_state,
    )
    return qkv_results, relu_results
