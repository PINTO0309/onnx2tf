from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.cast_cleanup import run_redundant_cast_cleanup
from onnx2tf.tflite_builder.passes.constant_fold import (
    run_constant_input_fold_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


CONSTANT_FOLD_CAST_PASS_IDS = (
    "run_constant_input_fold_cleanup",
    "run_redundant_cast_cleanup",
)


ConstantFoldCastContext = ModelIRPassContext


def build_constant_fold_cast_invocations(
    context: ConstantFoldCastContext,
    *,
    state_scope: ModelIRPassStateScope | None = None,
) -> Tuple[RecoveryInvocation, ...]:
    if state_scope is None:
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
            pass_id=CONSTANT_FOLD_CAST_PASS_IDS[0],
            callback=run_constant_input_fold_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=CONSTANT_FOLD_CAST_PASS_IDS[1],
            callback=run_redundant_cast_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )


def run_constant_fold_cast(
    context: ConstantFoldCastContext,
    *,
    state_scope: ModelIRPassStateScope | None = None,
) -> None:
    run_recovery_invocations(
        build_constant_fold_cast_invocations(
            context,
            state_scope=state_scope,
        ),
        expected_pass_ids=CONSTANT_FOLD_CAST_PASS_IDS,
        phase_name="constant-fold/redundant-cast",
    )
