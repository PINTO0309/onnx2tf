from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
    run_preadd_mean_attention_recovery,
)
from onnx2tf.tflite_builder.passes.channel_shuffle_gather_orchestration import (
    run_channel_shuffle_gather,
)


RecoveryResults = Tuple[Any, ...]


def run_layout_pass_set_2_channel_preadd_recovery(
    context: AttentionRecoveryContext,
) -> Tuple[RecoveryResults, RecoveryResults]:
    """Run full channel recovery before the later pre-add recovery."""

    return (
        run_channel_shuffle_gather(
            context.pass_context,
            include_post_gather_cleanup=True,
        ),
        run_preadd_mean_attention_recovery(context),
    )
