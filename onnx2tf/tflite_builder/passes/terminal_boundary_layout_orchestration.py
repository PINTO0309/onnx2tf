from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.boundary_input_layout import (
    run_boundary_input_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dual_mul_concat_layout import (
    run_dual_mul_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
    run_transpose_gather_channel_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import run_pad_layout_cleanup
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


TERMINAL_BOUNDARY_LAYOUT_PASS_IDS = (
    "run_dual_mul_concat_layout_cleanup",
    "run_boundary_input_layout_cleanup",
    "run_pad_layout_cleanup",
    "run_layout_transpose_cleanup",
    "run_transpose_gather_channel_fanout_cleanup",
)


TerminalBoundaryLayoutContext = ModelIRPassContext


def build_terminal_boundary_layout_invocations(
    context: TerminalBoundaryLayoutContext,
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
            pass_id=TERMINAL_BOUNDARY_LAYOUT_PASS_IDS[0],
            callback=run_dual_mul_concat_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=TERMINAL_BOUNDARY_LAYOUT_PASS_IDS[1],
            callback=run_boundary_input_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        # Boundary input recovery can recreate input-head PAD wrappers.
        RecoveryInvocation(
            pass_id=TERMINAL_BOUNDARY_LAYOUT_PASS_IDS[2],
            callback=run_pad_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        # Remove transpose pairs introduced by the recovered adapters.
        RecoveryInvocation(
            pass_id=TERMINAL_BOUNDARY_LAYOUT_PASS_IDS[3],
            callback=run_layout_transpose_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=TERMINAL_BOUNDARY_LAYOUT_PASS_IDS[4],
            callback=run_transpose_gather_channel_fanout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )


def run_terminal_boundary_layout(
    context: TerminalBoundaryLayoutContext,
) -> None:
    run_recovery_invocations(
        build_terminal_boundary_layout_invocations(context),
        expected_pass_ids=TERMINAL_BOUNDARY_LAYOUT_PASS_IDS,
        phase_name="terminal-boundary layout",
    )
