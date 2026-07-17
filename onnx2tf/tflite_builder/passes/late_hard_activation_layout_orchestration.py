from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    run_hard_activation_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS = (
    "run_hard_activation_passthrough_cleanup",
    "run_layout_transpose_cleanup",
)


@dataclass(frozen=True)
class LateHardActivationLayoutContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]


def active_late_hard_activation_layout_pass_ids(
    *,
    include_layout_transpose: bool,
) -> Tuple[str, ...]:
    if include_layout_transpose:
        return LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS
    return LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[:1]


def build_late_hard_activation_layout_invocations(
    context: LateHardActivationLayoutContext,
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
    invocations = [
        RecoveryInvocation(
            pass_id=LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[0],
            callback=run_hard_activation_passthrough_cleanup,
            args=(context.model_ir,),
            keyword_args=(
                ("include_hardswish", False),
                ("include_hardsigmoid", True),
                ("include_hardsigmoid_mul", True),
                ("reverse_hardsigmoid_order", True),
                *shared_keyword_args,
            ),
        )
    ]
    if include_layout_transpose:
        invocations.append(
            RecoveryInvocation(
                pass_id=LATE_HARD_ACTIVATION_LAYOUT_PASS_IDS[1],
                callback=run_layout_transpose_cleanup,
                args=(context.model_ir,),
                keyword_args=shared_keyword_args,
            )
        )
    return tuple(invocations)


def run_late_hard_activation_layout(
    context: LateHardActivationLayoutContext,
    *,
    include_layout_transpose: bool,
) -> None:
    expected_pass_ids = active_late_hard_activation_layout_pass_ids(
        include_layout_transpose=include_layout_transpose,
    )
    run_recovery_invocations(
        build_late_hard_activation_layout_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="late hard-activation/layout",
    )
