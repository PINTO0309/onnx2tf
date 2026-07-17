from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_consecutive_reshape_cleanup,
    run_duplicate_fanout_cleanup,
    run_squeeze_reshape_identity_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)
from onnx2tf.tflite_builder.passes.multi_branch_gate_layout import (
    run_multi_branch_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.singleton_maxpool_layout import (
    run_singleton_maxpool_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_flatten_concat_reshape_cleanup,
    run_singleton_channel_transpose_cleanup,
    run_singleton_reshape_layout_cleanup,
    run_singleton_spatial_reshape_cleanup,
)


SINGLETON_RESHAPE_LAYOUT_PASS_IDS = ("run_layout_transpose_cleanup",)
SINGLETON_RESHAPE_PREFIX_PASS_IDS = ("run_singleton_channel_transpose_cleanup",)
SINGLETON_RESHAPE_DUPLICATE_PASS_IDS = ("run_duplicate_fanout_cleanup",)
SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS = (
    "run_singleton_reshape_layout_cleanup",
    "run_singleton_maxpool_layout_cleanup",
    "run_flatten_concat_reshape_cleanup",
    "run_consecutive_reshape_cleanup",
    "run_squeeze_reshape_identity_cleanup",
    "run_singleton_spatial_reshape_cleanup",
)
SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS = ("run_multi_branch_gate_layout_cleanup",)
SINGLETON_RESHAPE_BASE_PASS_IDS = (
    *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
    *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
)
SINGLETON_RESHAPE_LAYOUT_MULTI_PASS_IDS = (
    *SINGLETON_RESHAPE_LAYOUT_PASS_IDS,
    *SINGLETON_RESHAPE_BASE_PASS_IDS,
    *SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS,
)
SINGLETON_RESHAPE_DUPLICATE_BASE_PASS_IDS = (
    *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
    *SINGLETON_RESHAPE_DUPLICATE_PASS_IDS,
    *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
)
SINGLETON_RESHAPE_PASS_IDS = (
    *SINGLETON_RESHAPE_LAYOUT_PASS_IDS,
    *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
    *SINGLETON_RESHAPE_DUPLICATE_PASS_IDS,
    *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
    *SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS,
)


SingletonReshapeContext = ModelIRPassContext


def _selected_pass_ids(
    *,
    include_layout_transpose: bool,
    include_duplicate_fanout: bool,
    include_multi_branch_gate: bool,
) -> Tuple[str, ...]:
    return (
        *(SINGLETON_RESHAPE_LAYOUT_PASS_IDS if include_layout_transpose else ()),
        *SINGLETON_RESHAPE_PREFIX_PASS_IDS,
        *(SINGLETON_RESHAPE_DUPLICATE_PASS_IDS if include_duplicate_fanout else ()),
        *SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS,
        *(SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS if include_multi_branch_gate else ()),
    )


def build_singleton_reshape_invocations(
    context: SingletonReshapeContext,
    *,
    include_layout_transpose: bool = False,
    include_duplicate_fanout: bool = False,
    include_multi_branch_gate: bool = False,
    include_spatial_concat_post_transpose: bool = True,
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
        SINGLETON_RESHAPE_LAYOUT_PASS_IDS[0]: run_layout_transpose_cleanup,
        SINGLETON_RESHAPE_PREFIX_PASS_IDS[0]: (run_singleton_channel_transpose_cleanup),
        SINGLETON_RESHAPE_DUPLICATE_PASS_IDS[0]: run_duplicate_fanout_cleanup,
        SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[0]: (run_singleton_reshape_layout_cleanup),
        SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[1]: (run_singleton_maxpool_layout_cleanup),
        SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[2]: (run_flatten_concat_reshape_cleanup),
        SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[3]: run_consecutive_reshape_cleanup,
        SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[4]: (run_squeeze_reshape_identity_cleanup),
        SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[5]: (
            run_singleton_spatial_reshape_cleanup
        ),
        SINGLETON_RESHAPE_MULTI_BRANCH_PASS_IDS[0]: (
            run_multi_branch_gate_layout_cleanup
        ),
    }
    selected_pass_ids = _selected_pass_ids(
        include_layout_transpose=include_layout_transpose,
        include_duplicate_fanout=include_duplicate_fanout,
        include_multi_branch_gate=include_multi_branch_gate,
    )
    invocations = []
    for pass_id in selected_pass_ids:
        keyword_args = shared_keyword_args
        if pass_id == SINGLETON_RESHAPE_DUPLICATE_PASS_IDS[0]:
            keyword_args = (("include_transpose", False), *shared_keyword_args)
        elif pass_id == SINGLETON_RESHAPE_BASE_TAIL_PASS_IDS[5]:
            keyword_args = (
                (
                    "include_concat_post_transpose",
                    include_spatial_concat_post_transpose,
                ),
                *shared_keyword_args,
            )
        invocations.append(
            RecoveryInvocation(
                pass_id=pass_id,
                callback=callback_by_pass_id[pass_id],
                args=(context.model_ir,),
                keyword_args=keyword_args,
            )
        )
    return tuple(invocations)


def run_singleton_reshape(
    context: SingletonReshapeContext,
    *,
    include_layout_transpose: bool = False,
    include_duplicate_fanout: bool = False,
    include_multi_branch_gate: bool = False,
    include_spatial_concat_post_transpose: bool = True,
) -> None:
    expected_pass_ids = _selected_pass_ids(
        include_layout_transpose=include_layout_transpose,
        include_duplicate_fanout=include_duplicate_fanout,
        include_multi_branch_gate=include_multi_branch_gate,
    )
    run_recovery_invocations(
        build_singleton_reshape_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_duplicate_fanout=include_duplicate_fanout,
            include_multi_branch_gate=include_multi_branch_gate,
            include_spatial_concat_post_transpose=(
                include_spatial_concat_post_transpose
            ),
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="singleton-reshape",
    )
