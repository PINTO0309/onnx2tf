from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.attention_layout import (
    run_qkv_attention_bridge_cleanup,
    run_qkv_attention_prefix_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


QKV_ATTENTION_PASS_IDS = (
    "run_layout_transpose_cleanup",
    "run_qkv_attention_prefix_cleanup",
    "run_qkv_attention_bridge_cleanup",
)


QKVAttentionContext = ModelIRPassContext
_LAYOUT_MUTATION_KEYS = (
    "removed_identity_transpose",
    "removed_inverse_transpose_pairs",
    "removed_inverse_transpose_fanout_branches",
    "composed_consecutive_transpose_pairs",
)
_PREFIX_MUTATION_KEYS = (
    "optimized_attention_qkv_gather_reshape_transpose_hoist_chains",
    "optimized_attention_qkv_slice_replace_gather_reshape_chains",
    "optimized_attention_qkv_slice_to_split_chains",
    "optimized_attention_split_post_reshape_collapse_chains",
)
_BRIDGE_MUTATION_KEYS = (
    "optimized_attention_qkv_shared_pretranspose_slice_nchw_chains",
    "optimized_attention_qkv_weighted_sum_bridge_to_nhwc_chains",
)


def summarize_qkv_attention_mutations(
    pass_results: Tuple[Dict[str, int], ...],
    *,
    include_layout_transpose: bool,
    include_prefix: bool,
    pruned_unused_tensors: int,
) -> Dict[str, int]:
    """Normalize QKV results into one stable mutation-only schema."""

    expected_count = len(
        active_qkv_attention_pass_ids(
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        )
    )
    if len(pass_results) != expected_count:
        raise ValueError(
            "QKV attention mutation summary expected "
            f"{expected_count} pass results, got {len(pass_results)}"
        )

    summary = {
        key: 0
        for key in (
            *_LAYOUT_MUTATION_KEYS,
            *_PREFIX_MUTATION_KEYS,
            *_BRIDGE_MUTATION_KEYS,
        )
    }
    result_index = 0
    if include_layout_transpose:
        layout_result = pass_results[result_index]
        summary.update(
            {
                key: int(layout_result.get(key, 0))
                for key in _LAYOUT_MUTATION_KEYS
            }
        )
        result_index += 1
    if include_prefix:
        prefix_result = pass_results[result_index]
        summary.update(
            {
                key: int(prefix_result.get(key, 0))
                for key in _PREFIX_MUTATION_KEYS
            }
        )
        result_index += 1
    bridge_result = pass_results[result_index]
    summary.update(
        {
            key: int(bridge_result.get(key, 0))
            for key in _BRIDGE_MUTATION_KEYS
        }
    )
    summary["pruned_unused_tensors"] = max(0, int(pruned_unused_tensors))
    return summary


def active_qkv_attention_pass_ids(
    *,
    include_layout_transpose: bool = False,
    include_prefix: bool = True,
) -> Tuple[str, ...]:
    pass_ids: list[str] = []
    if include_layout_transpose:
        pass_ids.append(QKV_ATTENTION_PASS_IDS[0])
    if include_prefix:
        pass_ids.append(QKV_ATTENTION_PASS_IDS[1])
    pass_ids.append(QKV_ATTENTION_PASS_IDS[2])
    return tuple(pass_ids)


def build_qkv_attention_invocations(
    context: QKVAttentionContext,
    *,
    include_layout_transpose: bool = False,
    include_prefix: bool = True,
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
    invocations: list[RecoveryInvocation] = []
    if include_layout_transpose:
        invocations.append(
            RecoveryInvocation(
                pass_id=QKV_ATTENTION_PASS_IDS[0],
                callback=run_layout_transpose_cleanup,
                args=(context.model_ir,),
                keyword_args=keyword_args,
            )
        )
    if include_prefix:
        invocations.append(
            RecoveryInvocation(
                pass_id=QKV_ATTENTION_PASS_IDS[1],
                callback=run_qkv_attention_prefix_cleanup,
                args=(context.model_ir,),
                keyword_args=keyword_args,
            )
        )
    invocations.append(
        RecoveryInvocation(
            pass_id=QKV_ATTENTION_PASS_IDS[2],
            callback=run_qkv_attention_bridge_cleanup,
            args=(context.model_ir,),
            keyword_args=keyword_args,
        )
    )
    return tuple(invocations)


def run_qkv_attention(
    context: QKVAttentionContext,
    *,
    include_layout_transpose: bool = False,
    include_prefix: bool = True,
) -> Tuple[Dict[str, int], ...]:
    expected_pass_ids = active_qkv_attention_pass_ids(
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )
    return run_recovery_invocations(
        build_qkv_attention_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="QKV attention",
    )
