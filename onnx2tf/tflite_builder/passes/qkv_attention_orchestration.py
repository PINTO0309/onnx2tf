from __future__ import annotations

from typing import Tuple

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
) -> None:
    expected_pass_ids = active_qkv_attention_pass_ids(
        include_layout_transpose=include_layout_transpose,
        include_prefix=include_prefix,
    )
    run_recovery_invocations(
        build_qkv_attention_invocations(
            context,
            include_layout_transpose=include_layout_transpose,
            include_prefix=include_prefix,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="QKV attention",
    )
