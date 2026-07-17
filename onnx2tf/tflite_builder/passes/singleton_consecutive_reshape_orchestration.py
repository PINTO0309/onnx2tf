from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_consecutive_reshape_cleanup,
    run_duplicate_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_singleton_channel_transpose_cleanup,
)


SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS = (
    "run_singleton_channel_transpose_cleanup",
    "run_duplicate_fanout_cleanup",
    "run_consecutive_reshape_cleanup",
)


SingletonConsecutiveReshapeContext = ModelIRPassContext


def build_singleton_consecutive_reshape_invocations(
    context: SingletonConsecutiveReshapeContext,
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
            pass_id=SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS[0],
            callback=run_singleton_channel_transpose_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS[1],
            callback=run_duplicate_fanout_cleanup,
            args=(context.model_ir,),
            keyword_args=(
                ("include_transpose", False),
                *shared_keyword_args,
            ),
        ),
        RecoveryInvocation(
            pass_id=SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS[2],
            callback=run_consecutive_reshape_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )


def run_singleton_consecutive_reshape(
    context: SingletonConsecutiveReshapeContext,
) -> None:
    run_recovery_invocations(
        build_singleton_consecutive_reshape_invocations(context),
        expected_pass_ids=SINGLETON_CONSECUTIVE_RESHAPE_PASS_IDS,
        phase_name="singleton-channel/consecutive-reshape",
    )
