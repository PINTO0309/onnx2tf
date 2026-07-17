from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
    build_constant_fold_cast_invocations,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_gather_axis_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import (
    run_normalization_pad_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS = (
    "run_transpose_gather_axis_cleanup",
    *CONSTANT_FOLD_CAST_PASS_IDS,
    "run_normalization_pad_layout_cleanup",
)


VeryLateGatherConstantNormalizationContext = ModelIRPassContext


def build_very_late_gather_constant_normalization_invocations(
    context: VeryLateGatherConstantNormalizationContext,
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
    constant_fold_cast_invocations = build_constant_fold_cast_invocations(
        context,
        state_scope=state_scope,
    )
    return (
        RecoveryInvocation(
            pass_id=VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[0],
            callback=run_transpose_gather_axis_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        *constant_fold_cast_invocations,
        RecoveryInvocation(
            pass_id=VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[-1],
            callback=run_normalization_pad_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=(
                ("include_instance", False),
                ("include_flatten", True),
                *shared_keyword_args,
            ),
        ),
    )


def run_very_late_gather_constant_normalization(
    context: VeryLateGatherConstantNormalizationContext,
) -> None:
    run_recovery_invocations(
        build_very_late_gather_constant_normalization_invocations(context),
        expected_pass_ids=VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS,
        phase_name="very-late gather/constant/normalization",
    )
