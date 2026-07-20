from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.concat_unary_conv_layout import (
    run_concat_unary_conv_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.spp_layout import run_spp_layout_cleanup


LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS = (
    "run_spp_layout_cleanup",
    "run_concat_unary_conv_layout_cleanup",
)


LateSPPConcatUnaryConvContext = ModelIRPassContext
_MUTATION_KEYS = (
    "optimized_transpose_resize_add_concat_affine_conv_spp_nhwc_chains",
    "optimized_transpose_concat_unary_fanout_conv_nhwc_chains",
)


def summarize_late_spp_concat_unary_conv_mutations(
    pass_results: Tuple[Dict[str, int], ...],
) -> Dict[str, int]:
    """Normalize the ordered pair into its two declared mutation counters."""

    expected_count = len(LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS)
    if len(pass_results) != expected_count:
        raise ValueError(
            "late SPP mutation summary expected "
            f"{expected_count} pass results, got {len(pass_results)}"
        )
    return {
        key: int(result.get(key, 0))
        for key, result in zip(_MUTATION_KEYS, pass_results)
    }


def build_late_spp_concat_unary_conv_invocations(
    context: LateSPPConcatUnaryConvContext,
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
            pass_id=LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS[0],
            callback=run_spp_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        RecoveryInvocation(
            pass_id=LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS[1],
            callback=run_concat_unary_conv_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_late_spp_concat_unary_conv(
    context: LateSPPConcatUnaryConvContext,
) -> Tuple[Dict[str, int], ...]:
    return run_recovery_invocations(
        build_late_spp_concat_unary_conv_invocations(context),
        expected_pass_ids=LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS,
        phase_name="late SPP/concat-unary-conv",
    )


def run_late_spp_concat_unary_conv_summary(
    context: LateSPPConcatUnaryConvContext,
) -> Dict[str, int]:
    """Run late SPP/Concat/Unary cleanup and return its normalized summary."""

    return summarize_late_spp_concat_unary_conv_mutations(
        run_late_spp_concat_unary_conv(context)
    )
