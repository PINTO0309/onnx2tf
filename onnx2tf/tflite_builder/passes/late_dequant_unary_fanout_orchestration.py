from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.dequant_concat_quantize_layout import (
    run_dequant_concat_quantize_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_unary_fanout_bridge_cleanup,
    run_transpose_unary_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


LATE_DEQUANT_UNARY_FANOUT_PASS_IDS = (
    "run_dequant_concat_quantize_layout_cleanup",
    "run_transpose_unary_passthrough_cleanup",
    "run_transpose_unary_fanout_bridge_cleanup",
)


@dataclass(frozen=True)
class LateDequantUnaryFanoutContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def build_late_dequant_unary_fanout_invocations(
    context: LateDequantUnaryFanoutContext,
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
            pass_id=LATE_DEQUANT_UNARY_FANOUT_PASS_IDS[0],
            callback=run_dequant_concat_quantize_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        # Some late specialized rewrites can still leave trivial
        # NHWC->NCHW->NHWC wrappers around unary activations.
        # Run one final strict unary transpose fold before serialization guards.
        RecoveryInvocation(
            pass_id=LATE_DEQUANT_UNARY_FANOUT_PASS_IDS[1],
            callback=run_transpose_unary_passthrough_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        RecoveryInvocation(
            pass_id=LATE_DEQUANT_UNARY_FANOUT_PASS_IDS[2],
            callback=run_transpose_unary_fanout_bridge_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_late_dequant_unary_fanout(
    context: LateDequantUnaryFanoutContext,
) -> None:
    run_recovery_invocations(
        build_late_dequant_unary_fanout_invocations(context),
        expected_pass_ids=LATE_DEQUANT_UNARY_FANOUT_PASS_IDS,
        phase_name="late dequant/unary/fanout",
    )
