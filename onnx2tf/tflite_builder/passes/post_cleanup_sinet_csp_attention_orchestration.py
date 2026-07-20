from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.attention_layout import (
    _optimize_transpose_csp_attention_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_preadd_resize_recovery_orchestration import (
    run_sinet_preadd_resize_recovery,
)


SINetResults = Tuple[Dict[str, int], ...]


def run_post_cleanup_sinet_csp_attention_cleanup(
    context: ModelIRPassContext,
) -> Tuple[SINetResults, Dict[str, int]]:
    """Run post-cleanup SiNet recovery and CSP-attention cleanup in order."""

    sinet_results = run_sinet_preadd_resize_recovery(context)
    csp_results = _optimize_transpose_csp_attention_nhwc_chains(
        context.model_ir,
        layout_state=context.layout_state,
    )
    return sinet_results, csp_results
