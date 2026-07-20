from __future__ import annotations

from typing import Any, Tuple

from onnx2tf.tflite_builder.passes.attention_recovery_orchestration import (
    AttentionRecoveryContext,
)
from onnx2tf.tflite_builder.passes.layout_pass_set_2_preadd_attention_gate_orchestration import (
    run_layout_pass_set_2_preadd_attention_gate_recovery,
)
from onnx2tf.tflite_builder.passes.layout_pass_set_2_qlinear_layout_recovery_orchestration import (
    run_layout_pass_set_2_qlinear_layout_recovery,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LayoutRecoveryContext,
)


RecoveryResults = Tuple[Any, ...]
ChildResults = Tuple[RecoveryResults, RecoveryResults]


def run_layout_pass_set_2_qlinear_preadd_cleanup(
    layout_context: LayoutRecoveryContext,
    attention_context: AttentionRecoveryContext,
) -> Tuple[ChildResults, ChildResults]:
    """Run the leading layout-pass-set-2 recovery children in order."""

    qlinear_results = run_layout_pass_set_2_qlinear_layout_recovery(
        layout_context
    )
    preadd_results = run_layout_pass_set_2_preadd_attention_gate_recovery(
        attention_context
    )
    return qlinear_results, preadd_results
