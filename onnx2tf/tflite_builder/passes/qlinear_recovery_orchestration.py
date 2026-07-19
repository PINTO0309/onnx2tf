from __future__ import annotations

from typing import Any, Callable, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.mean_hardsigmoid_muladd_layout import (
    optimize_transpose_mean_hardsigmoid_muladd_chains,
)
from onnx2tf.tflite_builder.passes.mean_maxpool_concat_layout import (
    _optimize_transpose_mean_maxpool_concat_conv_chains,
)
from onnx2tf.tflite_builder.passes.qlinear_concat_conv_compat import (
    optimize_nhwc_propagation_qlinear_concat_conv,
)
from onnx2tf.tflite_builder.passes.qlinear_silu_prefix_layout import (
    _optimize_nhwc_prefix_qlinear_silu_chains,
)
from onnx2tf.tflite_builder.passes.quantization_cleanup import (
    _optimize_concat_pre_quantize_dequantize,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)


QLINEAR_MEAN_CONCAT_PASS_IDS = (
    "_optimize_transpose_mean_hardsigmoid_muladd_chains",
    "_optimize_nhwc_prefix_qlinear_silu_chains",
    "_optimize_nhwc_propagation_qlinear_concat_conv",
    "_optimize_concat_pre_quantize_dequantize",
    "_optimize_transpose_mean_maxpool_concat_conv_chains",
)


QLinearRecoveryContext = ModelIRPassContext


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: QLinearRecoveryContext,
) -> RecoveryInvocation:
    return RecoveryInvocation(
        pass_id=pass_id,
        callback=callback,
        args=(context.model_ir,),
    )


def build_qlinear_mean_concat_invocations(
    context: QLinearRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            QLINEAR_MEAN_CONCAT_PASS_IDS[0],
            optimize_transpose_mean_hardsigmoid_muladd_chains,
            context,
        ),
        _model_invocation(
            QLINEAR_MEAN_CONCAT_PASS_IDS[1],
            _optimize_nhwc_prefix_qlinear_silu_chains,
            context,
        ),
        _model_invocation(
            QLINEAR_MEAN_CONCAT_PASS_IDS[2],
            optimize_nhwc_propagation_qlinear_concat_conv,
            context,
        ),
        _model_invocation(
            QLINEAR_MEAN_CONCAT_PASS_IDS[3],
            _optimize_concat_pre_quantize_dequantize,
            context,
        ),
        _model_invocation(
            QLINEAR_MEAN_CONCAT_PASS_IDS[4],
            _optimize_transpose_mean_maxpool_concat_conv_chains,
            context,
        ),
    )


def run_qlinear_mean_concat_recovery(
    context: QLinearRecoveryContext,
) -> Tuple[Any, ...]:
    return run_recovery_invocations(
        build_qlinear_mean_concat_invocations(context),
        expected_pass_ids=QLINEAR_MEAN_CONCAT_PASS_IDS,
        phase_name="qlinear mean/concat recovery",
    )
