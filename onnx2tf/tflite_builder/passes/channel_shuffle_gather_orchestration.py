from __future__ import annotations

from typing import Dict, Tuple, cast

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    run_nchw_channel_shuffle_cleanup,
    run_nhwc_channel_shuffle_cleanup,
    run_two_way_channel_shuffle_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
    run_transpose_gather_axis_cleanup,
    run_transpose_unary_binary_fanout_bridge_cleanup,
    run_transpose_unary_fanout_bridge_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS = (
    "run_two_way_channel_shuffle_cleanup",
    "run_nhwc_channel_shuffle_cleanup",
)
CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS = (
    "run_nchw_channel_shuffle_cleanup",
    "run_transpose_gather_axis_cleanup",
)
CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS = (
    "run_layout_transpose_cleanup",
    "run_transpose_unary_fanout_bridge_cleanup",
    "run_transpose_unary_binary_fanout_bridge_cleanup",
)
CHANNEL_SHUFFLE_GATHER_DEFAULT_PASS_IDS = (
    *CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS,
    *CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS,
)
CHANNEL_SHUFFLE_GATHER_PASS_IDS = (
    *CHANNEL_SHUFFLE_GATHER_DEFAULT_PASS_IDS,
    *CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS,
)


ChannelShuffleGatherContext = ModelIRPassContext


def _selected_pass_ids(
    *,
    include_two_way_shuffle: bool,
    include_nhwc_shuffle: bool,
    include_post_gather_cleanup: bool,
) -> Tuple[str, ...]:
    return (
        *(
            (CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS[0],)
            if include_two_way_shuffle
            else ()
        ),
        *(
            (CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS[1],)
            if include_nhwc_shuffle
            else ()
        ),
        *CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS,
        *(CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS if include_post_gather_cleanup else ()),
    )


def build_channel_shuffle_gather_invocations(
    context: ChannelShuffleGatherContext,
    *,
    include_two_way_shuffle: bool = True,
    include_nhwc_shuffle: bool = True,
    include_post_gather_cleanup: bool = False,
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
    callback_by_pass_id = {
        CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS[0]: (
            run_two_way_channel_shuffle_cleanup
        ),
        CHANNEL_SHUFFLE_GATHER_LEADING_PASS_IDS[1]: (run_nhwc_channel_shuffle_cleanup),
        CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS[0]: run_nchw_channel_shuffle_cleanup,
        CHANNEL_SHUFFLE_GATHER_BASE_PASS_IDS[1]: run_transpose_gather_axis_cleanup,
        CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS[0]: run_layout_transpose_cleanup,
        CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS[1]: (
            run_transpose_unary_fanout_bridge_cleanup
        ),
        CHANNEL_SHUFFLE_GATHER_POST_PASS_IDS[2]: (
            run_transpose_unary_binary_fanout_bridge_cleanup
        ),
    }
    selected_pass_ids = _selected_pass_ids(
        include_two_way_shuffle=include_two_way_shuffle,
        include_nhwc_shuffle=include_nhwc_shuffle,
        include_post_gather_cleanup=include_post_gather_cleanup,
    )
    return tuple(
        RecoveryInvocation(
            pass_id=pass_id,
            callback=callback_by_pass_id[pass_id],
            args=(context.model_ir,),
            keyword_args=shared_keyword_args,
        )
        for pass_id in selected_pass_ids
    )


def run_channel_shuffle_gather(
    context: ChannelShuffleGatherContext,
    *,
    include_two_way_shuffle: bool = True,
    include_nhwc_shuffle: bool = True,
    include_post_gather_cleanup: bool = False,
) -> Tuple[Dict[str, int], ...]:
    expected_pass_ids = _selected_pass_ids(
        include_two_way_shuffle=include_two_way_shuffle,
        include_nhwc_shuffle=include_nhwc_shuffle,
        include_post_gather_cleanup=include_post_gather_cleanup,
    )
    return cast(
        Tuple[Dict[str, int], ...],
        run_recovery_invocations(
            build_channel_shuffle_gather_invocations(
                context,
                include_two_way_shuffle=include_two_way_shuffle,
                include_nhwc_shuffle=include_nhwc_shuffle,
                include_post_gather_cleanup=include_post_gather_cleanup,
            ),
            expected_pass_ids=expected_pass_ids,
            phase_name="channel-shuffle/gather",
        ),
    )
