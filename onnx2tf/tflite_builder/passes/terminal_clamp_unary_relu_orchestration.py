from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_clamp_cleanup,
    run_maximum_zero_relu_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_unary_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


TERMINAL_CLAMP_UNARY_RELU_PASS_IDS = (
    "run_clamp_cleanup",
    "run_transpose_unary_passthrough_cleanup",
    "run_maximum_zero_relu_cleanup",
)


TerminalClampUnaryReLUContext = ModelIRPassContext


def build_terminal_clamp_unary_relu_invocations(
    context: TerminalClampUnaryReLUContext,
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
            pass_id=TERMINAL_CLAMP_UNARY_RELU_PASS_IDS[0],
            callback=run_clamp_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        # MAXIMUM/MINIMUM -> RELU_0_TO_1 canonicalization can expose fresh
        # NHWC<->NCHW unary wrappers (e.g. HardSigmoid clamp). Re-run the
        # strict transpose-unary passthrough fold once in terminal stage.
        RecoveryInvocation(
            pass_id=TERMINAL_CLAMP_UNARY_RELU_PASS_IDS[1],
            callback=run_transpose_unary_passthrough_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        RecoveryInvocation(
            pass_id=TERMINAL_CLAMP_UNARY_RELU_PASS_IDS[2],
            callback=run_maximum_zero_relu_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_terminal_clamp_unary_relu(
    context: TerminalClampUnaryReLUContext,
) -> Tuple[Dict[str, int], ...]:
    return run_recovery_invocations(
        build_terminal_clamp_unary_relu_invocations(context),
        expected_pass_ids=TERMINAL_CLAMP_UNARY_RELU_PASS_IDS,
        phase_name="terminal clamp/unary/ReLU",
    )
