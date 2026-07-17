from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    run_channel_slice_merge_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import run_pad_mul_layout_cleanup
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


CHANNEL_SLICE_PAD_MUL_PASS_IDS = (
    "run_channel_slice_merge_layout_cleanup",
    "run_pad_mul_layout_cleanup",
)


ChannelSlicePadMulContext = ModelIRPassContext


def build_channel_slice_pad_mul_invocations(
    context: ChannelSlicePadMulContext,
) -> Tuple[RecoveryInvocation, ...]:
    state_scope = ModelIRPassStateScope(
        context.model_ir,
        layout_state=context.layout_state,
    )
    keyword_args = (
        ("layout_state", context.layout_state),
        ("diagnostics", context.diagnostics),
        ("state_scope", state_scope),
    )
    return (
        RecoveryInvocation(
            pass_id=CHANNEL_SLICE_PAD_MUL_PASS_IDS[0],
            callback=run_channel_slice_merge_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        RecoveryInvocation(
            pass_id=CHANNEL_SLICE_PAD_MUL_PASS_IDS[1],
            callback=run_pad_mul_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_channel_slice_pad_mul(
    context: ChannelSlicePadMulContext,
) -> None:
    run_recovery_invocations(
        build_channel_slice_pad_mul_invocations(context),
        expected_pass_ids=CHANNEL_SLICE_PAD_MUL_PASS_IDS,
        phase_name="channel-slice/pad-mul",
    )
