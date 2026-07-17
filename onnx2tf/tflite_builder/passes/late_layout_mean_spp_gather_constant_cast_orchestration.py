from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
    ConstantFoldCastContext,
    build_constant_fold_cast_invocations,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
    run_transpose_gather_axis_cleanup,
)
from onnx2tf.tflite_builder.passes.mean_layout import (
    run_mean_mul_add_conv_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.spp_layout import run_spp_layout_cleanup


LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS = (
    "run_mean_mul_add_conv_layout_cleanup",
    "run_spp_layout_cleanup",
    "run_transpose_gather_axis_cleanup",
    *CONSTANT_FOLD_CAST_PASS_IDS,
)
LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS = (
    "run_layout_transpose_cleanup",
    *LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS,
)


@dataclass(frozen=True)
class LateLayoutMeanSPPGatherConstantCastContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def build_late_layout_mean_spp_gather_constant_cast_invocations(
    context: LateLayoutMeanSPPGatherConstantCastContext,
    *,
    include_layout_transpose: bool,
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
    layout_invocations = (
        (
            RecoveryInvocation(
                pass_id=LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS[0],
                callback=run_layout_transpose_cleanup,
                args=(context.model_ir,),
                keyword_args=shared_keyword_args,
            ),
        )
        if include_layout_transpose
        else ()
    )
    constant_fold_cast_invocations = build_constant_fold_cast_invocations(
        ConstantFoldCastContext(
            model_ir=context.model_ir,
            layout_state=context.layout_state,
            diagnostics=context.diagnostics,
        ),
        state_scope=state_scope,
    )
    return (
        *layout_invocations,
        RecoveryInvocation(
            pass_id=LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS[0],
            callback=run_mean_mul_add_conv_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS[1],
            callback=run_spp_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS[2],
            callback=run_transpose_gather_axis_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        *constant_fold_cast_invocations,
    )


def run_late_layout_mean_spp_gather_constant_cast(
    context: LateLayoutMeanSPPGatherConstantCastContext,
    *,
    include_layout_transpose: bool,
) -> None:
    expected_pass_ids = (
        LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_PASS_IDS
        if include_layout_transpose
        else LATE_LAYOUT_MEAN_SPP_GATHER_CONSTANT_CAST_REQUIRED_PASS_IDS
    )
    run_recovery_invocations(
        build_late_layout_mean_spp_gather_constant_cast_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="late layout/mean/SPP/gather/constant-fold/cast",
    )
