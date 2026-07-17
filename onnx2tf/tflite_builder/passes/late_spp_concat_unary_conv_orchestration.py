from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.concat_unary_conv_layout import (
    run_concat_unary_conv_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.spp_layout import run_spp_layout_cleanup


LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS = (
    "run_spp_layout_cleanup",
    "run_concat_unary_conv_layout_cleanup",
)


@dataclass(frozen=True)
class LateSPPConcatUnaryConvContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def build_late_spp_concat_unary_conv_invocations(
    context: LateSPPConcatUnaryConvContext,
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
            pass_id=LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS[0],
            callback=run_spp_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
        RecoveryInvocation(
            pass_id=LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS[1],
            callback=run_concat_unary_conv_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        ),
    )


def run_late_spp_concat_unary_conv(
    context: LateSPPConcatUnaryConvContext,
) -> None:
    run_recovery_invocations(
        build_late_spp_concat_unary_conv_invocations(context),
        expected_pass_ids=LATE_SPP_CONCAT_UNARY_CONV_PASS_IDS,
        phase_name="late SPP/concat-unary-conv",
    )
