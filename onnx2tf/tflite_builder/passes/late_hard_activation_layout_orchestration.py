from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
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


LateHardActivationLayoutContext = ModelIRPassContext
_LAYOUT_MUTATION_KEYS = (
    "removed_identity_transpose",
    "removed_inverse_transpose_pairs",
    "removed_inverse_transpose_fanout_branches",
    "composed_consecutive_transpose_pairs",
)


def summarize_late_hard_activation_layout_mutations(
    pass_results: Tuple[Dict[str, int], ...],
    *,
    include_layout_transpose: bool,
    pruned_unused_tensors: int,
) -> Dict[str, int]:
    """Normalize raw cluster results into mutation-only counters."""

    expected_count = len(
        active_late_hard_activation_layout_pass_ids(
            include_layout_transpose=include_layout_transpose,
        )
    )
    if len(pass_results) != expected_count:
        raise ValueError(
            "late hard-activation mutation summary expected "
            f"{expected_count} pass results, got {len(pass_results)}"
        )

    summary = {
        str(key): int(value)
        for key, value in pass_results[0].items()
        if str(key) != "iterations"
    }
    summary.update({key: 0 for key in _LAYOUT_MUTATION_KEYS})
    if include_layout_transpose:
        layout_result = pass_results[1]
        summary.update(
            {
                key: int(layout_result.get(key, 0))
                for key in _LAYOUT_MUTATION_KEYS
            }
        )
    summary["pruned_unused_tensors"] = max(
        0,
        int(pruned_unused_tensors),
    )
    return summary


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
) -> Tuple[Dict[str, int], ...]:
    expected_pass_ids = active_late_hard_activation_layout_pass_ids(
        include_layout_transpose=include_layout_transpose,
    )
    return run_recovery_invocations(
        build_late_hard_activation_layout_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="late hard-activation/layout",
    )


def run_late_hard_activation_layout_summary(
    context: LateHardActivationLayoutContext,
    *,
    include_layout_transpose: bool,
) -> Dict[str, int]:
    """Run late hard-activation cleanup and return its prune-aware summary."""

    initial_tensor_count = len(context.model_ir.tensors)
    pass_results = run_late_hard_activation_layout(
        context,
        include_layout_transpose=include_layout_transpose,
    )
    return summarize_late_hard_activation_layout_mutations(
        pass_results,
        include_layout_transpose=include_layout_transpose,
        pruned_unused_tensors=max(
            0,
            int(initial_tensor_count - len(context.model_ir.tensors)),
        ),
    )
