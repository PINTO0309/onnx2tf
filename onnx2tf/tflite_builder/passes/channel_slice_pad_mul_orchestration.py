from __future__ import annotations

from typing import Dict, Tuple

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
_MUTATION_KEYS_BY_RESULT = (
    (
        "optimized_transpose_channel_slice_dual_add_bridges_strict",
        "optimized_transpose_slice_muladd_conv_mergeadd_strict",
        "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict",
    ),
    ("optimized_transpose_pad_mul_posttranspose_add_nhwc_chains",),
)


def summarize_channel_slice_pad_mul_mutations(
    pass_results: Tuple[Dict[str, int], ...],
) -> Dict[str, int]:
    """Normalize the ordered pair into its four declared mutation counters."""

    expected_count = len(CHANNEL_SLICE_PAD_MUL_PASS_IDS)
    if len(pass_results) != expected_count:
        raise ValueError(
            "channel-slice/pad-Mul mutation summary expected "
            f"{expected_count} pass results, got {len(pass_results)}"
        )
    return {
        key: int(result.get(key, 0))
        for result, keys in zip(pass_results, _MUTATION_KEYS_BY_RESULT)
        for key in keys
    }


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
) -> Tuple[Dict[str, int], ...]:
    return run_recovery_invocations(
        build_channel_slice_pad_mul_invocations(context),
        expected_pass_ids=CHANNEL_SLICE_PAD_MUL_PASS_IDS,
        phase_name="channel-slice/pad-mul",
    )
