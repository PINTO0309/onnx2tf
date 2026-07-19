from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
    run_transpose_unary_binary_fanout_bridge_cleanup,
    run_transpose_unary_fanout_bridge_cleanup,
    run_transpose_unary_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


TRANSPOSE_UNARY_FANOUT_PASS_IDS = (
    "run_layout_transpose_cleanup",
    "run_transpose_unary_passthrough_cleanup",
    "run_transpose_unary_fanout_bridge_cleanup",
    "run_transpose_unary_binary_fanout_bridge_cleanup",
)


TransposeUnaryFanoutContext = ModelIRPassContext


def active_transpose_unary_fanout_pass_ids(
    *,
    include_layout_transpose: bool = False,
    include_unary_passthrough: bool = True,
) -> Tuple[str, ...]:
    pass_ids: list[str] = []
    if include_layout_transpose:
        pass_ids.append(TRANSPOSE_UNARY_FANOUT_PASS_IDS[0])
    if include_unary_passthrough:
        pass_ids.append(TRANSPOSE_UNARY_FANOUT_PASS_IDS[1])
    pass_ids.extend(TRANSPOSE_UNARY_FANOUT_PASS_IDS[2:])
    return tuple(pass_ids)


def build_transpose_unary_fanout_invocations(
    context: TransposeUnaryFanoutContext,
    *,
    include_layout_transpose: bool = False,
    include_unary_passthrough: bool = True,
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
    invocations: list[RecoveryInvocation] = []
    if include_layout_transpose:
        invocations.append(
            RecoveryInvocation(
                pass_id=TRANSPOSE_UNARY_FANOUT_PASS_IDS[0],
                callback=run_layout_transpose_cleanup,
                args=(context.model_ir,),
                keyword_args=keyword_args,
            )
        )
    if include_unary_passthrough:
        invocations.append(
            RecoveryInvocation(
                pass_id=TRANSPOSE_UNARY_FANOUT_PASS_IDS[1],
                callback=run_transpose_unary_passthrough_cleanup,
                args=(context.model_ir,),
                keyword_args=keyword_args,
            )
        )
    invocations.extend(
        (
            RecoveryInvocation(
                pass_id=TRANSPOSE_UNARY_FANOUT_PASS_IDS[2],
                callback=run_transpose_unary_fanout_bridge_cleanup,
                args=(context.model_ir,),
                keyword_args=keyword_args,
            ),
            RecoveryInvocation(
                pass_id=TRANSPOSE_UNARY_FANOUT_PASS_IDS[3],
                callback=run_transpose_unary_binary_fanout_bridge_cleanup,
                args=(context.model_ir,),
                keyword_args=keyword_args,
            ),
        )
    )
    return tuple(invocations)


def run_transpose_unary_fanout(
    context: TransposeUnaryFanoutContext,
    *,
    include_layout_transpose: bool = False,
    include_unary_passthrough: bool = True,
) -> Tuple[Dict[str, int], ...]:
    expected_pass_ids = active_transpose_unary_fanout_pass_ids(
        include_layout_transpose=include_layout_transpose,
        include_unary_passthrough=include_unary_passthrough,
    )
    return run_recovery_invocations(
        build_transpose_unary_fanout_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_unary_passthrough=include_unary_passthrough,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="transpose/unary fanout",
    )
