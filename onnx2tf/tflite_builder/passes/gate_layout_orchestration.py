from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.add_concat_suffix_layout import (
    run_add_concat_suffix_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.attention_layout import (
    run_mixed_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.cost_volume_scatter_layout import (
    run_cost_volume_scatter_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dual_mul_concat_layout import (
    run_dual_mul_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dual_postconv_gate_layout import (
    run_dual_postconv_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.elementwise_gate_layout import (
    run_elementwise_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.ndhwc_gate_layout import (
    run_ndhwc_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import run_pad_layout_cleanup
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


GATE_LAYOUT_REQUIRED_PASS_IDS = (
    "run_elementwise_gate_layout_cleanup",
    "run_pad_layout_cleanup",
    "run_dual_postconv_gate_layout_cleanup",
    "run_ndhwc_gate_layout_cleanup",
    "run_cost_volume_scatter_layout_cleanup",
    "run_add_concat_suffix_layout_cleanup",
    "run_dual_mul_concat_layout_cleanup",
)
GATE_LAYOUT_PASS_IDS = (
    "run_mixed_attention_layout_cleanup",
    *GATE_LAYOUT_REQUIRED_PASS_IDS,
)
LATE_NDHWC_COST_VOLUME_PASS_IDS = (
    "run_ndhwc_gate_layout_cleanup",
    "run_cost_volume_scatter_layout_cleanup",
)


GateLayoutContext = ModelIRPassContext


def run_late_ndhwc_cost_volume_layout_cleanup(
    context: GateLayoutContext,
) -> Dict[str, int]:
    """Run the adjacent late NDHWC and cost-volume groups with one state."""

    state_scope = ModelIRPassStateScope(
        context.model_ir,
        layout_state=context.layout_state,
    )
    ndhwc_details = run_ndhwc_gate_layout_cleanup(
        context.model_ir,
        layout_state=context.layout_state,
        diagnostics=context.diagnostics,
        state_scope=state_scope,
    )
    cost_volume_details = run_cost_volume_scatter_layout_cleanup(
        context.model_ir,
        layout_state=context.layout_state,
        diagnostics=context.diagnostics,
        state_scope=state_scope,
    )
    return {
        str(key): int(value)
        for details in (ndhwc_details, cost_volume_details)
        for key, value in details.items()
    }


def build_gate_layout_invocations(
    context: GateLayoutContext,
    *,
    include_mixed_attention: bool = True,
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
    required_invocations = (
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[0],
            callback=run_elementwise_gate_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[1],
            callback=run_pad_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[2],
            callback=run_dual_postconv_gate_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[3],
            callback=run_ndhwc_gate_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[4],
            callback=run_cost_volume_scatter_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[5],
            callback=run_add_concat_suffix_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_REQUIRED_PASS_IDS[6],
            callback=run_dual_mul_concat_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
    )
    if not include_mixed_attention:
        return required_invocations
    return (
        RecoveryInvocation(
            pass_id=GATE_LAYOUT_PASS_IDS[0],
            callback=run_mixed_attention_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        *required_invocations,
    )


def run_gate_layout(
    context: GateLayoutContext,
    *,
    include_mixed_attention: bool = True,
) -> Tuple[Dict[str, int], ...]:
    expected_pass_ids = (
        GATE_LAYOUT_PASS_IDS
        if include_mixed_attention
        else GATE_LAYOUT_REQUIRED_PASS_IDS
    )
    return run_recovery_invocations(
        build_gate_layout_invocations(
            context,
            include_mixed_attention=include_mixed_attention,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="gate-layout",
    )
