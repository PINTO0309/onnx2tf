from __future__ import annotations

from typing import Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.passes.attention_layout import (
    run_conv_attention_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.layernorm_layout import (
    run_layernorm_statistics_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.mean_layout import (
    run_mean_mul_add_conv_layout_cleanup,
    run_transpose_mean_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.se_layout import (
    run_se_conv_layout_cleanup,
    run_se_fc_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_mean_layout import (
    run_terminal_mean_layout_cleanup,
)


MEAN_ATTENTION_PREFIX_PASS_IDS = (
    "run_transpose_mean_passthrough_cleanup",
    "run_mean_mul_add_conv_layout_cleanup",
)
MEAN_ATTENTION_LAYERNORM_PASS_IDS = ("run_layernorm_statistics_layout_cleanup",)
MEAN_ATTENTION_BASE_TAIL_PASS_IDS = (
    "run_terminal_mean_layout_cleanup",
    "run_se_conv_layout_cleanup",
    "run_se_fc_layout_cleanup",
)
MEAN_ATTENTION_CONV_PASS_IDS = ("run_conv_attention_layout_cleanup",)
MEAN_ATTENTION_BASE_PASS_IDS = (
    *MEAN_ATTENTION_PREFIX_PASS_IDS,
    *MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
)
MEAN_ATTENTION_DEFAULT_PASS_IDS = (
    *MEAN_ATTENTION_BASE_PASS_IDS,
    *MEAN_ATTENTION_CONV_PASS_IDS,
)
MEAN_ATTENTION_PASS_IDS = (
    *MEAN_ATTENTION_PREFIX_PASS_IDS,
    *MEAN_ATTENTION_LAYERNORM_PASS_IDS,
    *MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
    *MEAN_ATTENTION_CONV_PASS_IDS,
)


MeanAttentionContext = ModelIRPassContext


def _selected_pass_ids(
    *,
    include_layernorm: bool,
    include_conv_attention: bool,
) -> Tuple[str, ...]:
    return (
        *MEAN_ATTENTION_PREFIX_PASS_IDS,
        *(MEAN_ATTENTION_LAYERNORM_PASS_IDS if include_layernorm else ()),
        *MEAN_ATTENTION_BASE_TAIL_PASS_IDS,
        *(MEAN_ATTENTION_CONV_PASS_IDS if include_conv_attention else ()),
    )


def build_mean_attention_invocations(
    context: MeanAttentionContext,
    *,
    include_layernorm: bool = False,
    include_conv_attention: bool = True,
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
        MEAN_ATTENTION_PREFIX_PASS_IDS[0]: run_transpose_mean_passthrough_cleanup,
        MEAN_ATTENTION_PREFIX_PASS_IDS[1]: run_mean_mul_add_conv_layout_cleanup,
        MEAN_ATTENTION_LAYERNORM_PASS_IDS[0]: (run_layernorm_statistics_layout_cleanup),
        MEAN_ATTENTION_BASE_TAIL_PASS_IDS[0]: run_terminal_mean_layout_cleanup,
        MEAN_ATTENTION_BASE_TAIL_PASS_IDS[1]: run_se_conv_layout_cleanup,
        MEAN_ATTENTION_BASE_TAIL_PASS_IDS[2]: run_se_fc_layout_cleanup,
        MEAN_ATTENTION_CONV_PASS_IDS[0]: run_conv_attention_layout_cleanup,
    }
    selected_pass_ids = _selected_pass_ids(
        include_layernorm=include_layernorm,
        include_conv_attention=include_conv_attention,
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


def run_mean_attention(
    context: MeanAttentionContext,
    *,
    include_layernorm: bool = False,
    include_conv_attention: bool = True,
) -> None:
    expected_pass_ids = _selected_pass_ids(
        include_layernorm=include_layernorm,
        include_conv_attention=include_conv_attention,
    )
    run_recovery_invocations(
        build_mean_attention_invocations(
            context,
            include_layernorm=include_layernorm,
            include_conv_attention=include_conv_attention,
        ),
        expected_pass_ids=expected_pass_ids,
        phase_name="mean/attention",
    )
