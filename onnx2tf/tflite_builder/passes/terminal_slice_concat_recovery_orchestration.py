from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_post_add_layout import (
    optimize_transpose_mul_posttranspose_add_nhwc_chains,
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
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)
from onnx2tf.tflite_builder.passes.pre_add_layout import (
    optimize_transpose_pre_add_nhwc_chains,
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
from onnx2tf.tflite_builder.passes.stridedslice_pad_concat_bridge_layout import (
    _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains,
)


TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS = (
    "_run_channel_slice_pad_mul_layout_pass_cluster",
    "_optimize_transpose_mul_posttranspose_add_nhwc_chains",
    "_optimize_concat_mul_add_transpose_nhwc_bridge_chains",
    "_optimize_concat_mul_add_transpose_add_nhwc_bridge_chains",
    "_optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains",
    "_optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains",
    "_optimize_singleton_gate_conv_concat_nhwc_bridge_blocks",
    "_optimize_transpose_unary_split_concat_single_post_nchw",
    "_optimize_transpose_split_channelwise_tail_to_single_post_nchw",
    "_optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw",
    "_sanitize_probable_nhwc_axis_sensitive_ops",
    "_optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains",
    "_optimize_transpose_pre_add_nhwc_chains",
    "run_layout_transpose_cleanup",
)


@dataclass(frozen=True)
class TerminalSliceConcatRecoveryContext:
    pass_context: ModelIRPassContext
    channel_slice_pad_mul_cluster: Callable[[], Any]


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: TerminalSliceConcatRecoveryContext,
    *,
    include_layout: bool = False,
    include_diagnostics: bool = False,
) -> RecoveryInvocation:
    keyword_args = []
    if include_layout:
        keyword_args.append(("layout_state", context.pass_context.layout_state))
    if include_diagnostics:
        keyword_args.append(("diagnostics", context.pass_context.diagnostics))
    return RecoveryInvocation(
        pass_id=pass_id,
        callback=callback,
        args=(context.pass_context.model_ir,),
        keyword_args=tuple(keyword_args),
    )


def build_terminal_slice_concat_recovery_invocations(
    context: TerminalSliceConcatRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        RecoveryInvocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[0],
            context.channel_slice_pad_mul_cluster,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[1],
            optimize_transpose_mul_posttranspose_add_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[2],
            _optimize_concat_mul_add_transpose_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[3],
            _optimize_concat_mul_add_transpose_add_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[4],
            _optimize_concat_mul_add_add_mean_reshape_tail_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[5],
            _optimize_concat_tree_mul_add_transpose_nhwc_bridge_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[6],
            optimize_singleton_gate_conv_concat_nhwc_bridge_blocks,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[7],
            optimize_transpose_unary_split_concat_single_post_nchw,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[8],
            optimize_transpose_split_channelwise_tail_to_single_post_nchw,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[9],
            optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[10],
            sanitize_probable_nhwc_axis_sensitive_ops,
            context,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[11],
            _optimize_transpose_stridedslice_pad_concat_mul_add_posttranspose_nhwc_chains,
            context,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[12],
            optimize_transpose_pre_add_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS[13],
            run_layout_transpose_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
    )


def run_terminal_slice_concat_recovery(
    context: TerminalSliceConcatRecoveryContext,
) -> None:
    run_recovery_invocations(
        build_terminal_slice_concat_recovery_invocations(context),
        expected_pass_ids=TERMINAL_SLICE_CONCAT_RECOVERY_PASS_IDS,
        phase_name="terminal slice/concat recovery",
    )
