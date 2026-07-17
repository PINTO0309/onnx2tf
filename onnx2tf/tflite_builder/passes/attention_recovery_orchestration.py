from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.conv_output_passthrough_layout import (
    optimize_transposeconv_output_channel1_terminal_transpose_chains,
    optimize_transposeconv_output_nhwc_passthrough_chains,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_trailing_output_transpose_cleanup,
)
from onnx2tf.tflite_builder.passes.mean_affine_prepost_layout import (
    optimize_transpose_mean_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_add_layout import (
    optimize_transpose_pre_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_unary_affine_fanout_layout import (
    optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.quantized_activation import (
    optimize_transpose_dequant_hardsigmoid_quantize_bridges,
    optimize_transpose_dequant_mul_add_prelu_quantize_bridges,
    optimize_transpose_dequant_relu_quantize_bridges,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.residual_affine_fanout_layout import (
    optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.residual_affine_prelu_layout import (
    optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_mix_attention_layout import (
    optimize_sinet_mix_attention_double_logistic_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.sinet_sa_pa_mirrorpad_layout import (
    optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains,
)


PREADD_MEAN_ATTENTION_PASS_IDS = (
    "_optimize_transpose_pre_add_nhwc_chains",
    "_optimize_transpose_pre_add_mul_add_prelu_nhwc_chains",
    "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains",
    "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
    "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
    "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains",
    "_run_mean_attention_layout_pass_cluster",
)

ATTENTION_GATE_QDQ_PASS_IDS = (
    "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains",
    "_optimize_sinet_mix_attention_double_logistic_nhwc_chains",
    "_run_gate_layout_pass_cluster",
    "_optimize_transposeconv_output_nhwc_passthrough_chains",
    "_optimize_transposeconv_output_channel1_terminal_transpose_chains",
    "_run_transpose_unary_fanout_layout_pass_cluster",
    "_optimize_transpose_dequant_relu_quantize_bridges",
    "_optimize_transpose_dequant_hardsigmoid_quantize_bridges",
    "run_trailing_output_transpose_cleanup",
    "_optimize_transpose_dequant_mul_add_prelu_quantize_bridges",
)


@dataclass(frozen=True)
class AttentionRecoveryContext:
    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]
    mean_attention_cluster: Callable[[], Any]
    gate_layout_cluster: Callable[[], Any]
    transpose_unary_fanout_cluster: Callable[[], Any]


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: AttentionRecoveryContext,
    *,
    include_layout: bool = False,
    include_diagnostics: bool = False,
) -> RecoveryInvocation:
    keyword_args = []
    if include_layout:
        keyword_args.append(("layout_state", context.layout_state))
    if include_diagnostics:
        keyword_args.append(("diagnostics", context.diagnostics))
    return RecoveryInvocation(
        pass_id=pass_id,
        callback=callback,
        args=(context.model_ir,),
        keyword_args=tuple(keyword_args),
    )


def build_preadd_mean_attention_invocations(
    context: AttentionRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[0],
            optimize_transpose_pre_add_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[1],
            optimize_transpose_pre_add_mul_add_prelu_nhwc_chains,
            context,
        ),
        _model_invocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[2],
            optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains,
            context,
        ),
        _model_invocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[3],
            optimize_transpose_mul_add_const_prepost_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[4],
            optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
            context,
        ),
        _model_invocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[5],
            optimize_transpose_mean_mul_add_const_prepost_nhwc_chains,
            context,
        ),
        RecoveryInvocation(
            PREADD_MEAN_ATTENTION_PASS_IDS[6],
            context.mean_attention_cluster,
        ),
    )


def run_preadd_mean_attention_recovery(
    context: AttentionRecoveryContext,
) -> None:
    run_recovery_invocations(
        build_preadd_mean_attention_invocations(context),
        expected_pass_ids=PREADD_MEAN_ATTENTION_PASS_IDS,
        phase_name="preadd/mean/attention recovery",
    )


def build_attention_gate_qdq_invocations(
    context: AttentionRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[0],
            optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[1],
            optimize_sinet_mix_attention_double_logistic_nhwc_chains,
            context,
            include_layout=True,
        ),
        RecoveryInvocation(
            ATTENTION_GATE_QDQ_PASS_IDS[2],
            context.gate_layout_cluster,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[3],
            optimize_transposeconv_output_nhwc_passthrough_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[4],
            optimize_transposeconv_output_channel1_terminal_transpose_chains,
            context,
            include_layout=True,
        ),
        RecoveryInvocation(
            ATTENTION_GATE_QDQ_PASS_IDS[5],
            context.transpose_unary_fanout_cluster,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[6],
            optimize_transpose_dequant_relu_quantize_bridges,
            context,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[7],
            optimize_transpose_dequant_hardsigmoid_quantize_bridges,
            context,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[8],
            run_trailing_output_transpose_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
        _model_invocation(
            ATTENTION_GATE_QDQ_PASS_IDS[9],
            optimize_transpose_dequant_mul_add_prelu_quantize_bridges,
            context,
        ),
    )


def run_attention_gate_qdq_recovery(
    context: AttentionRecoveryContext,
) -> None:
    run_recovery_invocations(
        build_attention_gate_qdq_invocations(context),
        expected_pass_ids=ATTENTION_GATE_QDQ_PASS_IDS,
        phase_name="attention/gate/QDQ recovery",
    )
