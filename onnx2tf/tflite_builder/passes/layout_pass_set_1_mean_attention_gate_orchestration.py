from __future__ import annotations

from typing import Any, Dict, Tuple

from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
    run_attention_gate_qdq_recovery,
)
from onnx2tf.tflite_builder.passes.mean_attention_orchestration import (
    run_mean_attention,
)


MeanAttentionResults = Tuple[Dict[str, int], ...]
AttentionGateQDQResults = Tuple[Any, ...]


def run_layout_pass_set_1_mean_attention_gate_cleanup(
    context: AttentionRecoveryContext,
) -> Tuple[MeanAttentionResults, AttentionGateQDQResults]:
    """Run layout-pass-set-1 mean/attention and gate/QDQ cleanup."""

    return (
        run_mean_attention(
            context.pass_context,
            include_layernorm=True,
        ),
        run_attention_gate_qdq_recovery(context),
    )
