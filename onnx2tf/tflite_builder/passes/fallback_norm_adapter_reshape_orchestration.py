from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    run_indexed_binary_layout_adapter_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_consecutive_reshape_orchestration import (
    run_singleton_consecutive_reshape,
)


AdapterResults = Tuple[Dict[str, int], Dict[str, int]]
ReshapeResults = Tuple[Dict[str, int], Dict[str, int], Dict[str, int]]


def run_fallback_norm_adapter_reshape_cleanup(
    context: ModelIRPassContext,
) -> Tuple[AdapterResults, ReshapeResults]:
    """Run indexed binary adapters before fallback singleton reshapes."""

    adapter_results = run_indexed_binary_layout_adapter_cleanup(
        context.model_ir
    )
    reshape_results = run_singleton_consecutive_reshape(context)
    return adapter_results, reshape_results
