from __future__ import annotations

from typing import Any, Callable, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.binary_bridge_layout import (
    run_safe_binary_bridge_recovery,
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
from onnx2tf.tflite_builder.passes.quantized_softmax import (
    _optimize_dequant_softmax_quantize_chains,
)
from onnx2tf.tflite_builder.passes.recovery_orchestration import (
    RecoveryInvocation,
    run_recovery_invocations,
)
from onnx2tf.tflite_builder.passes.softmax_transpose_canonicalization import (
    _canonicalize_softmax_transpose_chains,
)


SAFE_BINARY_RECOVERY_PASS_IDS = ("_run_safe_binary_bridge_recovery_pass",)

QUANTIZED_ACTIVATION_BINARY_PASS_IDS = (
    "_optimize_dequant_hardsigmoid_quantize_chains",
    "_optimize_dequant_maxpool_quantize_chains",
    "_optimize_dequant_softmax_quantize_chains",
    "_optimize_dequant_logistic_quantize_chains",
    "_canonicalize_softmax_transpose_chains",
    "_run_safe_binary_bridge_recovery_sequence",
)


QuantizedRecoveryContext = ModelIRPassContext


def _model_invocation(
    pass_id: str,
    callback: Callable[..., Any],
    context: QuantizedRecoveryContext,
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


def build_safe_binary_recovery_invocations(
    context: QuantizedRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            SAFE_BINARY_RECOVERY_PASS_IDS[0],
            run_safe_binary_bridge_recovery,
            context,
            include_layout=True,
        ),
    )


def run_safe_binary_recovery(
    context: QuantizedRecoveryContext,
) -> Tuple[Any, ...]:
    return run_recovery_invocations(
        build_safe_binary_recovery_invocations(context),
        expected_pass_ids=SAFE_BINARY_RECOVERY_PASS_IDS,
        phase_name="safe binary recovery",
    )


def build_quantized_activation_binary_invocations(
    context: QuantizedRecoveryContext,
) -> Tuple[RecoveryInvocation, ...]:
    return (
        _model_invocation(
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS[0],
            _optimize_dequant_hardsigmoid_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS[1],
            _optimize_dequant_maxpool_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS[2],
            _optimize_dequant_softmax_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS[3],
            _optimize_dequant_logistic_quantize_chains,
            context,
            include_layout=True,
        ),
        _model_invocation(
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS[4],
            _canonicalize_softmax_transpose_chains,
            context,
        ),
        RecoveryInvocation(
            QUANTIZED_ACTIVATION_BINARY_PASS_IDS[5],
            run_safe_binary_recovery,
            args=(context,),
        ),
    )


def run_quantized_activation_binary_recovery(
    context: QuantizedRecoveryContext,
) -> Tuple[Any, ...]:
    return run_recovery_invocations(
        build_quantized_activation_binary_invocations(context),
        expected_pass_ids=QUANTIZED_ACTIVATION_BINARY_PASS_IDS,
        phase_name="quantized activation/binary recovery",
    )
