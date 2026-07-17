from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.attention_layout import (
    run_mixed_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import (
    run_normalization_pad_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS = (
    "run_normalization_pad_layout_cleanup",
    "run_mixed_attention_layout_cleanup",
)


AbsoluteFinalNormalizationAttentionContext = ModelIRPassContext


def build_absolute_final_normalization_attention_invocations(
    context: AbsoluteFinalNormalizationAttentionContext,
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
            pass_id=ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[0],
            callback=run_normalization_pad_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=(
                ("include_instance", False),
                ("include_flatten", True),
                *shared_keyword_args,
            ),
        ),
        # Some late boundary/layout repairs can still recreate the DEA/SiNet
        # mixed NHWC/NCHW SA branch around REDUCE_MAX->CONCAT->MIRROR_PAD.
        RecoveryInvocation(
            pass_id=ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS[1],
            callback=run_mixed_attention_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )


def run_absolute_final_normalization_attention(
    context: AbsoluteFinalNormalizationAttentionContext,
) -> None:
    run_recovery_invocations(
        build_absolute_final_normalization_attention_invocations(context),
        expected_pass_ids=ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_PASS_IDS,
        phase_name="absolute-final normalization/attention",
    )
