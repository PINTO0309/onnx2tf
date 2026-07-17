from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.activation_passthrough_layout import (
    optimize_gelu_tanh_transpose_passthrough_chains,
    optimize_swish_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.attention_gather_cleanup_layout import (
    _optimize_attention_gather_transpose_reshape_cleanup_chains,
)
from onnx2tf.tflite_builder.passes.attention_preproj_ranklift_layout import (
    _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains,
)
from onnx2tf.tflite_builder.passes.attention_qkv_reshape_compat_layout import (
    optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat,
)
from onnx2tf.tflite_builder.passes.center_size_offset_layout import (
    optimize_center_size_offset_terminal_transpose_chains,
)
from onnx2tf.tflite_builder.passes.concat_input_adapter_layout import (
    optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
)
from onnx2tf.tflite_builder.passes.elementwise_concat_layout import (
    optimize_transpose_elementwise_concat_conv_nhwc_groups,
)
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)
from onnx2tf.tflite_builder.passes.elementwise_roundtrip_nchw_nhwc_layout import (
    _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.expanddims_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.flatten_hw_reshape_compat_layout import (
    optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.gather_reshape_cleanup import (
    _optimize_gather_axis0_singleton_to_reshape_input_chains,
)
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_squeeze_reshape_identity_cleanup,
)
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    run_hard_activation_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.leakyrelu_passthrough_layout import (
    optimize_leakyrelu_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.ndhwc_concat_layout import (
    run_ndhwc_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.pre_add_layout import (
    optimize_transpose_pre_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_add_mulconst_reshape_suffix_compat_layout import (
    optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.pre_unary_reshape_suffix_compat_layout import (
    optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.pre_unary_squeeze_suffix_compat_layout import (
    optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat,
)
from onnx2tf.tflite_builder.passes.prelu_passthrough_layout import (
    optimize_prelu_transpose_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.reshape_transpose_collapse_layout import (
    _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains,
)
from onnx2tf.tflite_builder.passes.slice_logistic_concat_reshape_tail_layout import (
    optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_mixed_concat_layout import (
    optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.spp_layout import run_spp_layout_cleanup
from onnx2tf.tflite_builder.passes.stridedslice_concat_layout import (
    optimize_transpose_stridedslice_pre_concat_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.transpose_qdq_bridge_layout import (
    optimize_transpose_quant_dequant_bridges,
)
from onnx2tf.tflite_builder.passes.window_partition_layout import (
    _optimize_window_partition_reshape_transpose_to_space_to_depth_chains,
    _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains,
)


LAYOUT_RECOVERY_PASS_IDS = (
    "_optimize_transpose_quant_dequant_bridges",
    "_run_boundary_batchmatmul_unary_layout_pass_cluster",
    "_optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains",
    "_optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains",
    "run_hard_activation_passthrough_cleanup",
    "_optimize_swish_transpose_passthrough_chains",
    "_optimize_gelu_tanh_transpose_passthrough_chains",
    "_optimize_center_size_offset_terminal_transpose_chains",
    "_optimize_leakyrelu_transpose_passthrough_chains",
    "_optimize_prelu_transpose_passthrough_chains",
    "_optimize_transpose_elementwise_concat_conv_nhwc_groups",
    "run_spp_layout_cleanup",
    "_optimize_transpose_pre_concat_nhwc_chains",
    "run_ndhwc_concat_layout_cleanup",
    "_optimize_transpose_stridedslice_pre_concat_nhwc_chains",
    "_optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains",
    "_optimize_transpose_input_chains_pre_concat_to_single_post_adapter",
    "_optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains",
    "_run_channel_shuffle_gather_layout_pass_cluster",
)

ATTENTION_RECOVERY_PASS_IDS = (
    "_run_layout_recovery_prefix_pass_sequence",
    "_optimize_transpose_pre_add_nhwc_chains",
    "_optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains",
    "_optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains",
    "_optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains",
    "_optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains",
    "_optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains",
    "_optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains",
    "_optimize_attention_gather_transpose_reshape_cleanup_chains",
    "_optimize_gather_axis0_singleton_to_reshape_input_chains",
    "_optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains",
    "_optimize_window_partition_reshape_transpose_to_space_to_depth_chains",
    "_optimize_window_reverse_reshape_transpose_to_depth_to_space_chains",
    "_optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains",
    "run_squeeze_reshape_identity_cleanup",
)


@dataclass(frozen=True)
class LayoutRecoveryContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]
    boundary_batchmatmul_unary_cluster: Callable[[], Any]
    pre_concat_cleanup: Callable[..., Any]
    channel_shuffle_gather_cluster: Callable[[], Any]


@dataclass(frozen=True)
class RecoveryInvocation:
    pass_id: str
    callback: Callable[..., Any]
    args: Tuple[Any, ...] = ()
    keyword_args: Tuple[Tuple[str, Any], ...] = ()

    def run(self) -> Any:
        return self.callback(*self.args, **dict(self.keyword_args))


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: LayoutRecoveryContext,
    *,
    include_layout: bool = False,
    include_diagnostics: bool = False,
    keyword_args: Tuple[Tuple[str, Any], ...] = (),
) -> RecoveryInvocation:
    keywords = list(keyword_args)
    if include_layout:
        keywords.append(("layout_state", context.layout_state))
    if include_diagnostics:
        keywords.append(("diagnostics", context.diagnostics))
    return RecoveryInvocation(
        pass_id=pass_id,
        callback=callback,
        args=(context.model_ir,),
        keyword_args=tuple(keywords),
    )


def build_layout_recovery_invocations(
    context: LayoutRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[0],
            optimize_transpose_quant_dequant_bridges,
            context,
        ),
        RecoveryInvocation(
            LAYOUT_RECOVERY_PASS_IDS[1],
            context.boundary_batchmatmul_unary_cluster,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[2],
            _optimize_transpose_elementwise_roundtrip_nchw_nhwc_chains,
            context,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[3],
            optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
            context,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[4],
            run_hard_activation_passthrough_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[5],
            optimize_swish_transpose_passthrough_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[6],
            optimize_gelu_tanh_transpose_passthrough_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[7],
            optimize_center_size_offset_terminal_transpose_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[8],
            optimize_leakyrelu_transpose_passthrough_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[9],
            optimize_prelu_transpose_passthrough_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[10],
            optimize_transpose_elementwise_concat_conv_nhwc_groups,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[11],
            run_spp_layout_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[12],
            context.pre_concat_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[13],
            run_ndhwc_concat_layout_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[14],
            optimize_transpose_stridedslice_pre_concat_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[15],
            optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[16],
            optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_RECOVERY_PASS_IDS[17],
            optimize_transpose_slice_logistic_concat_reshape_tail_nhwc_chains,
            context,
            include_layout=True,
        ),
        RecoveryInvocation(
            LAYOUT_RECOVERY_PASS_IDS[18],
            context.channel_shuffle_gather_cluster,
        ),
    )


def run_layout_recovery_prefix(context: LayoutRecoveryContext) -> None:
    invocations = build_layout_recovery_invocations(context)
    if (
        tuple(invocation.pass_id for invocation in invocations)
        != LAYOUT_RECOVERY_PASS_IDS
    ):
        raise RuntimeError("layout recovery pass IDs diverged from their order")
    for invocation in invocations:
        invocation.run()


def build_attention_recovery_invocations(
    context: LayoutRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        RecoveryInvocation(
            ATTENTION_RECOVERY_PASS_IDS[0],
            run_layout_recovery_prefix,
            args=(context,),
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[1],
            optimize_transpose_pre_add_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[2],
            optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains_compat,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[3],
            optimize_transpose_pre_unary_reshape_transpose_suffix_nhwc_chains_compat,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[4],
            optimize_transpose_reshape_transpose_to_expanddims_nhwc_chains_compat,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[5],
            optimize_transpose_reshape_transpose_to_flatten_hw_nhwc_chains_compat,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[6],
            _optimize_reshape_transpose_reshape_transpose_to_nhwc_reshape_chains,
            context,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[7],
            optimize_attention_qkv_reshape_transpose_reshape_to_reshape_transpose_chains_compat,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[8],
            _optimize_attention_gather_transpose_reshape_cleanup_chains,
            context,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[9],
            _optimize_gather_axis0_singleton_to_reshape_input_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[10],
            _optimize_attention_preproj_reshape_to_batchmatmul_ranklift_chains,
            context,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[11],
            _optimize_window_partition_reshape_transpose_to_space_to_depth_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[12],
            _optimize_window_reverse_reshape_transpose_to_depth_to_space_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[13],
            optimize_transpose_pre_unary_squeeze_transpose_suffix_nhwc_chains_compat,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_RECOVERY_PASS_IDS[14],
            run_squeeze_reshape_identity_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
            keyword_args=(("include_unary_passthrough", True),),
        ),
    )


def run_layout_reshape_attention_recovery_prefix(
    context: LayoutRecoveryContext,
) -> None:
    invocations = build_attention_recovery_invocations(context)
    if (
        tuple(invocation.pass_id for invocation in invocations)
        != ATTENTION_RECOVERY_PASS_IDS
    ):
        raise RuntimeError("attention recovery pass IDs diverged from their order")
    for invocation in invocations:
        invocation.run()
