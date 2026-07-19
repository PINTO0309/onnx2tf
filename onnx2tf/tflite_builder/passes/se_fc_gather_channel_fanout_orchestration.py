from __future__ import annotations

from typing import Dict, Tuple, cast

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_gather_channel_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.se_layout import run_se_fc_layout_cleanup
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains,
)


SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS = (
    "run_se_fc_layout_cleanup",
    "run_transpose_gather_channel_fanout_cleanup",
)


SEFCGatherChannelFanoutContext = ModelIRPassContext


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
) -> Tuple[Dict[str, int], Dict[str, int]]:
    return cast(
        Tuple[Dict[str, int], Dict[str, int]],
        run_recovery_invocations(
            build_se_fc_gather_channel_fanout_invocations(context),
            expected_pass_ids=SE_FC_GATHER_CHANNEL_FANOUT_PASS_IDS,
            phase_name="SE-FC/gather-channel-fanout",
        ),
    )


def run_sinet_se_fc_gather_summary(
    context: SEFCGatherChannelFanoutContext,
) -> Dict[str, int]:
    initial_tensor_count = len(context.model_ir.tensors)
    sinet_stats = optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains(
        context.model_ir,
        layout_state=context.layout_state,
    )
    se_fc_stats, gather_stats = run_se_fc_gather_channel_fanout(context)
    return {
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains": int(
            sinet_stats.get(
                "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains",
                0,
            )
        ),
        "optimized_transpose_se_fc_mul_prepost_nhwc_chains": int(
            se_fc_stats.get(
                "optimized_transpose_se_fc_mul_prepost_nhwc_chains",
                0,
            )
        ),
        "optimized_transpose_gather_transpose_nhwc_channel_chains": int(
            gather_stats.get(
                "optimized_transpose_gather_transpose_nhwc_channel_chains",
                0,
            )
        ),
        "pruned_unused_tensors": max(
            0,
            initial_tensor_count - len(context.model_ir.tensors),
        ),
    }
