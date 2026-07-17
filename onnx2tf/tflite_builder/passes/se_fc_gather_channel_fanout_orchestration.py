from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_gather_channel_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.se_layout import run_se_fc_layout_cleanup


SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS = (
    "run_se_fc_layout_cleanup",
    "run_transpose_gather_channel_fanout_cleanup",
)


@dataclass(frozen=True)
class SEFCGatherChannelFanoutContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def build_se_fc_gather_channel_fanout_invocations(
    context: SEFCGatherChannelFanoutContext,
) -> Tuple[RecoveryInvocation, ...]:
    state_scope = ModelIRPassStateScope(
        context.model_ir,
        layout_state=context.layout_state,
    )
    shared_keyword_args = (
        ("layout_state", context.layout_state),
        ("diagnostics", context.diagnostics),
        ("state_scope", state_scope),
    )
    return (
        RecoveryInvocation(
            pass_id=SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS[0],
            callback=run_se_fc_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS[1],
            callback=run_transpose_gather_channel_fanout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )


def run_se_fc_gather_channel_fanout(
    context: SEFCGatherChannelFanoutContext,
) -> None:
    run_recovery_invocations(
        build_se_fc_gather_channel_fanout_invocations(context),
        expected_pass_ids=SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS,
        phase_name="SE-FC/gather-channel-fanout",
    )
