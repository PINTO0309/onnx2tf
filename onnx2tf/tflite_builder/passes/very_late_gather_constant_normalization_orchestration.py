from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    CONSTANT_FOLD_CAST_PASS_IDS,
    build_constant_fold_cast_invocations,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_gather_axis_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import (
    run_normalization_pad_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS = (
    "run_transpose_gather_axis_cleanup",
    *CONSTANT_FOLD_CAST_PASS_IDS,
    "run_normalization_pad_layout_cleanup",
)


VeryLateGatherConstantNormalizationContext = ModelIRPassContext
_MUTATION_KEYS_BY_RESULT = (
    ("optimized_transpose_gather_transpose_axis_remap_nhwc_chains",),
    (
        "optimized_constant_input_pad_chains",
        "optimized_constant_input_pool_chains",
        "optimized_constant_input_cast_chains",
    ),
    (
        "optimized_redundant_int32_to_int64_passthrough_cast_chains",
        "optimized_redundant_int64_to_int32_cast_chains",
    ),
    (
        "optimized_transpose_instancenorm_pad_prepost_nhwc_chains",
        "optimized_transpose_flatten_globalnorm_pad_prepost_nhwc_chains",
    ),
)


def summarize_very_late_gather_constant_normalization_mutations(
    pass_results: Tuple[Dict[str, int], ...],
    *,
    pruned_unused_tensors: int,
) -> Dict[str, int]:
    """Normalize the four ordered child results into mutation-only counters."""

    if len(pass_results) != len(_MUTATION_KEYS_BY_RESULT):
        raise ValueError(
            "very-late normalization mutation summary expected "
            f"{len(_MUTATION_KEYS_BY_RESULT)} pass results, "
            f"got {len(pass_results)}"
        )
    summary = {
        key: int(result.get(key, 0))
        for result, keys in zip(pass_results, _MUTATION_KEYS_BY_RESULT)
        for key in keys
    }
    summary["pruned_unused_tensors"] = max(0, int(pruned_unused_tensors))
    return summary


def build_very_late_gather_constant_normalization_invocations(
    context: VeryLateGatherConstantNormalizationContext,
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
    constant_fold_cast_invocations = build_constant_fold_cast_invocations(
        context,
        state_scope=state_scope,
    )
    return (
        RecoveryInvocation(
            pass_id=VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[0],
            callback=run_transpose_gather_axis_cleanup,
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        ),
        *constant_fold_cast_invocations,
        RecoveryInvocation(
            pass_id=VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS[-1],
            callback=run_normalization_pad_layout_cleanup,
            args=(context.model_ir,),
            keyword_args=(
                ("include_instance", False),
                ("include_flatten", True),
                *shared_keyword_args,
            ),
        ),
    )


def run_very_late_gather_constant_normalization(
    context: VeryLateGatherConstantNormalizationContext,
) -> Tuple[Dict[str, int], ...]:
    return run_recovery_invocations(
        build_very_late_gather_constant_normalization_invocations(context),
        expected_pass_ids=VERY_LATE_GATHER_CONSTANT_NORMALIZATION_PASS_IDS,
        phase_name="very-late gather/constant/normalization",
    )


def run_very_late_gather_constant_normalization_summary(
    context: VeryLateGatherConstantNormalizationContext,
) -> Dict[str, int]:
    """Run very-late normalization and return its prune-aware summary."""

    initial_tensor_count = len(context.model_ir.tensors)
    pass_results = run_very_late_gather_constant_normalization(context)
    return summarize_very_late_gather_constant_normalization_mutations(
        pass_results,
        pruned_unused_tensors=max(
            0,
            int(initial_tensor_count - len(context.model_ir.tensors)),
        ),
    )
