from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.sinet_shuffle_residual_layout import (
    optimize_sinet_shuffle_residual_transpose_chains,
)
from onnx2tf.tflite_builder.passes.terminal_affine_prelu_layout import (
    optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains,
)


SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS = (
    "_optimize_sinet_shuffle_residual_transpose_chains",
    "_run_sinet_preadd_resize_recovery_sequence",
    "_optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains",
)


@dataclass(frozen=True)
class SINetTerminalLayoutRecoveryContext:
    pass_context: ModelIRPassContext
    preadd_resize_recovery: Callable[[], Any]


def build_sinet_terminal_layout_recovery_invocations(
    context: SINetTerminalLayoutRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        RecoveryInvocation(
            pass_id=SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS[0],
            callback=optimize_sinet_shuffle_residual_transpose_chains,
            args=(context.pass_context.model_ir,),
            keyword_args=(("layout_state", context.pass_context.layout_state),),
        ),
        RecoveryInvocation(
            pass_id=SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS[1],
            callback=context.preadd_resize_recovery,
        ),
        RecoveryInvocation(
            pass_id=SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS[2],
            callback=(
                optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains
            ),
            args=(context.pass_context.model_ir,),
        ),
    )


def run_sinet_terminal_layout_recovery(
    context: SINetTerminalLayoutRecoveryContext,
) -> None:
    run_recovery_invocations(
        build_sinet_terminal_layout_recovery_invocations(context),
        expected_pass_ids=SINET_TERMINAL_LAYOUT_RECOVERY_PASS_IDS,
        phase_name="SINet terminal-layout recovery",
    )
