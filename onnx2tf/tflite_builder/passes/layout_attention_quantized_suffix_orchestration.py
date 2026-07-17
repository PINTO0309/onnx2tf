from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.affine_prepost_layout import (
    optimize_transpose_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.mean_affine_prepost_layout import (
    optimize_transpose_mean_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_unary_affine_fanout_layout import (
    optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.quantized_hardsigmoid import (
    _optimize_dequant_hardsigmoid_quantize_chains,
)
from onnx2tf.tflite_builder.passes.quantized_logistic import (
    _optimize_dequant_logistic_quantize_chains,
)
from onnx2tf.tflite_builder.passes.quantized_pool import (
    _optimize_dequant_maxpool_quantize_chains,
)
from onnx2tf.tflite_builder.passes.quantized_reshape import (
    run_quantized_reshape_cleanup,
)
from onnx2tf.tflite_builder.passes.quantized_softmax import (
    _optimize_dequant_softmax_quantize_chains,
)
from onnx2tf.tflite_builder.passes.quantized_transpose_conv import (
    _optimize_dequant_transposeconv_quantize_chains,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.softmax_transpose_canonicalization import (
    _canonicalize_softmax_transpose_chains,
)


LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS = (
    "_optimize_transpose_mul_add_const_prepost_nhwc_chains",
    "_optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains",
    "_optimize_transpose_mean_mul_add_const_prepost_nhwc_chains",
    "_run_mean_attention_layout_pass_cluster",
    "_run_attention_gate_qdq_recovery_sequence",
    "_run_duplicate_quantized_prelu_pass_cluster",
    "_optimize_dequant_transposeconv_quantize_chains",
    "run_quantized_reshape_cleanup",
    "_optimize_dequant_hardsigmoid_quantize_chains",
    "_optimize_dequant_maxpool_quantize_chains",
    "_optimize_dequant_softmax_quantize_chains",
    "_optimize_dequant_logistic_quantize_chains",
    "_canonicalize_softmax_transpose_chains",
)


@dataclass(frozen=True)
class LayoutAttentionQuantizedSuffixContext:
    pass_context: ModelIRPassContext
    mean_attention_cluster: Callable[[], Any]
    attention_gate_qdq_recovery: Callable[[], Any]
    duplicate_quantized_prelu_cluster: Callable[..., Any]


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: LayoutAttentionQuantizedSuffixContext,
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


def build_layout_attention_quantized_suffix_invocations(
    context: LayoutAttentionQuantizedSuffixContext,
    *,
    include_duplicate_transpose: bool,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[0],
            optimize_transpose_mul_add_const_prepost_nhwc_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[1],
            optimize_transpose_pre_unary_mul_add_transpose_fanout_nhwc_chains,
            context,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[2],
            optimize_transpose_mean_mul_add_const_prepost_nhwc_chains,
            context,
        ),
        RecoveryInvocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[3],
            context.mean_attention_cluster,
        ),
        RecoveryInvocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[4],
            context.attention_gate_qdq_recovery,
        ),
        RecoveryInvocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[5],
            context.duplicate_quantized_prelu_cluster,
            keyword_args=(("include_transpose", include_duplicate_transpose),),
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[6],
            _optimize_dequant_transposeconv_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[7],
            run_quantized_reshape_cleanup,
            context,
            include_layout=True,
            include_diagnostics=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[8],
            _optimize_dequant_hardsigmoid_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[9],
            _optimize_dequant_maxpool_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[10],
            _optimize_dequant_softmax_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[11],
            _optimize_dequant_logistic_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[12],
            _canonicalize_softmax_transpose_chains,
            context,
        ),
    )


def run_layout_attention_quantized_suffix(
    context: LayoutAttentionQuantizedSuffixContext,
    *,
    include_duplicate_transpose: bool,
) -> None:
    run_recovery_invocations(
        build_layout_attention_quantized_suffix_invocations(
            context,
            include_duplicate_transpose=include_duplicate_transpose,
        ),
        expected_pass_ids=LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
        phase_name="layout/attention/quantized recovery suffix",
    )
