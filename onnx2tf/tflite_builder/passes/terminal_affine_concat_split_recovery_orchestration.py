from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_chain_fold import (
    optimize_fold_mul_add_mul_affine_chains,
)
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_add_mean_reshape_layout import (
    _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_bridge_layout import (
    _optimize_concat_mul_add_transpose_nhwc_bridge_chains,
)
from onnx2tf.tflite_builder.passes.concat_mul_add_transpose_add_bridge_layout import (
    _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains,
)
from onnx2tf.tflite_builder.passes.concat_tree_mul_add_bridge_layout import (
    _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains,
)
from onnx2tf.tflite_builder.passes.probable_nhwc_axis_sanitizer import (
    sanitize_probable_nhwc_axis_sensitive_ops,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.singleton_gate_layout import (
    optimize_singleton_gate_conv_concat_nhwc_bridge_blocks,
)
from onnx2tf.tflite_builder.passes.split_channelwise_layout import (
    optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw,
    optimize_transpose_split_channelwise_tail_to_single_post_nchw,
    optimize_transpose_unary_split_concat_single_post_nchw,
)


TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS = (
    "_optimize_fold_mul_add_mul_affine_chains",
    "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
    "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
    "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
    "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
    "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
    "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
    "_optimize_transpose_unary_split_concat_single_post_nchw",
    "_optimize_transpose_split_channelwise_tail_to_single_post_nchw",
    "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw",
    "_sanitize_probable_nhwc_axis_sensitive_ops",
)


TerminalAffineConcatSplitRecoveryContext = ModelIRPassContext
_MUTATION_KEYS_BY_RESULT = (
    ("optimized_fold_mul_add_mul_affine_chains",),
    ("optimized_transpose_mul_add_const_prepost_nhwc_chains",),
    ("optimized_concat_mul_add_transpose_nhwc_bridge_chains",),
    ("optimized_concat_mul_add_transpose_add_nhwc_bridge_chains",),
    ("optimized_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",),
    ("optimized_concat_tree_mul_add_transpose_nhwc_bridge_chains",),
    ("optimized_singleton_gate_conv_concat_nhwc_bridge_blocks",),
    ("optimized_transpose_unary_split_concat_single_post_nchw",),
    ("optimized_transpose_split_channelwise_tail_to_single_post_nchw",),
    ("optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw",),
    (
        "sanitized_probable_nhwc_axis_sensitive_ops",
        "inserted_probable_nhwc_terminal_transposes",
    ),
)


def summarize_terminal_affine_concat_split_mutations(
    pass_results: Tuple[Dict[str, int], ...],
    *,
    pruned_unused_tensors: int,
) -> Dict[str, int]:
    """Normalize raw recovery results into declared mutation counters."""

    expected_count = len(TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS)
    if len(pass_results) != expected_count:
        raise ValueError(
            "terminal affine mutation summary expected "
            f"{expected_count} pass results, got {len(pass_results)}"
        )
    summary: Dict[str, int] = {}
    for keys, result in zip(_MUTATION_KEYS_BY_RESULT, pass_results):
        summary.update({key: int(result.get(key, 0)) for key in keys})
    summary["pruned_unused_tensors"] = max(0, int(pruned_unused_tensors))
    return summary


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: TerminalAffineConcatSplitRecoveryContext,
    *,
    include_layout: bool = False,
) -> RecoveryInvocation:
    keyword_args = ()
    if include_layout:
        keyword_args = (("layout_state", context.layout_state),)
    return RecoveryInvocation(
        pass_id=pass_id,
        callback=callback,
        args=(context.model_ir,),
        keyword_args=keyword_args,
    )


def build_terminal_affine_concat_split_recovery_invocations(
    context: TerminalAffineConcatSplitRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[0],
            optimize_fold_mul_add_mul_affine_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[1],
            optimize_transpose_mul_add_const_prepost_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[2],
            _optimize_concat_mul_add_transpose_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[3],
            _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[4],
            _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[5],
            _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[6],
            optimize_singleton_gate_conv_concat_nhwc_bridge_blocks,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[7],
            optimize_transpose_unary_split_concat_single_post_nchw,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[8],
            optimize_transpose_split_channelwise_tail_to_single_post_nchw,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[9],
            optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS[10],
            sanitize_probable_nhwc_axis_sensitive_ops,
            context,
        ),
    )


def run_terminal_affine_concat_split_recovery(
    context: TerminalAffineConcatSplitRecoveryContext,
) -> Tuple[Dict[str, int], ...]:
    return run_recovery_invocations(
        build_terminal_affine_concat_split_recovery_invocations(context),
        expected_pass_ids=TERMINAL_AFFINE_CONCAT_SPLIT_RECOVERY_PASS_IDS,
        phase_name="terminal affine/concat/split recovery",
    )
