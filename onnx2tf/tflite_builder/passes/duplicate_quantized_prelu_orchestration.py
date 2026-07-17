from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_duplicate_fanout_cleanup,
)
from onnx2tf.tflite_builder.passes.quantized_prelu import (
    run_quantized_prelu_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


DUPLICATE_QUANTIZED_PRELU_PASS_IDS = (
    "run_duplicate_fanout_cleanup",
    "run_quantized_prelu_cleanup",
)


DuplicateQuantizedPReLUContext = ModelIRPassContext


def build_duplicate_quantized_prelu_invocations(
    context: DuplicateQuantizedPReLUContext,
    *,
    include_transpose: bool,
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
            pass_id=DUPLICATE_QUANTIZED_PRELU_PASS_IDS[0],
            callback=run_duplicate_fanout_cleanup,
            args=(context.model_ir,),
            keyword_args=(
                ("include_transpose", include_transpose),
                *shared_keyword_args,
            ),
        ),
        RecoveryInvocation(
            pass_id=DUPLICATE_QUANTIZED_PRELU_PASS_IDS[1],
            callback=run_quantized_prelu_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )


def run_duplicate_quantized_prelu(
    context: DuplicateQuantizedPReLUContext,
    *,
    include_transpose: bool,
) -> None:
    run_recovery_invocations(
        build_duplicate_quantized_prelu_invocations(
            context,
            include_transpose=include_transpose,
        ),
        expected_pass_ids=DUPLICATE_QUANTIZED_PRELU_PASS_IDS,
        phase_name="duplicate-fanout/quantized-PReLU",
    )
