from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
    run_attention_gate_qdq_recovery,
    run_preadd_mean_attention_recovery,
)


RecoveryResults = Tuple[Any, ...]


def run_layout_pass_set_2_preadd_attention_gate_recovery(
    context: AttentionRecoveryContext,
) -> Tuple[RecoveryResults, RecoveryResults]:
    """Run pass-set-2 pre-add recovery before attention-gate recovery."""

    return (
        run_preadd_mean_attention_recovery(context),
        run_attention_gate_qdq_recovery(context),
    )
