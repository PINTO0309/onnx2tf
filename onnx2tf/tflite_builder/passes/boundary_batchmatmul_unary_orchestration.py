from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.boundary_input_chains import (
    run_boundary_input_batchmatmul_cleanup,
)
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    run_input_unary_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS = (
    "run_boundary_input_batchmatmul_cleanup",
    "run_input_unary_passthrough_cleanup",
)


@dataclass(frozen=True)
class BoundaryBatchMatMulUnaryContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def build_boundary_batchmatmul_unary_invocations(
    context: BoundaryBatchMatMulUnaryContext,
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
            pass_id=BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS[0],
            callback=run_boundary_input_batchmatmul_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        RecoveryInvocation(
            pass_id=BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS[1],
            callback=run_input_unary_passthrough_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_boundary_batchmatmul_unary(
    context: BoundaryBatchMatMulUnaryContext,
) -> None:
    run_recovery_invocations(
        build_boundary_batchmatmul_unary_invocations(context),
        expected_pass_ids=BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS,
        phase_name="boundary batchmatmul/input-unary",
    )
