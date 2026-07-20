from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_consecutive_mul_constants_cleanup,
)
from onnx2tf.tflite_builder.passes.precision import (
    _restore_precision_sensitive_reciprocal_divisions,
    _rewrite_constant_divisors_to_multiplicative_reciprocals,
)


PrecisionCleanupContext = ModelIRPassContext


def run_precision_cleanup_sequence(
    context: PrecisionCleanupContext,
) -> Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]:
    layout_keyword_args: Dict[str, LayoutState] = (
        {}
        if context.layout_state is None
        else {"layout_state": context.layout_state}
    )
    return (
        _rewrite_constant_divisors_to_multiplicative_reciprocals(
            context.model_ir,
            **layout_keyword_args,
        ),
        run_consecutive_mul_constants_cleanup(
            context.model_ir,
            **layout_keyword_args,
            diagnostics=context.diagnostics,
        ),
        _restore_precision_sensitive_reciprocal_divisions(
            context.model_ir,
            **layout_keyword_args,
        ),
    )
