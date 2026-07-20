from __future__ import annotations

from typing import Callable, Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.final_input_dynamic_orchestration import (
    run_final_input_dynamic_cleanup,
)
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


NestedMutationResults = Tuple[Dict[str, int], ...]
FinalInputDynamicResults = Tuple[
    NestedMutationResults,
    NestedMutationResults,
]
StaticShapeReconciler = Callable[..., Dict[str, int]]


def run_final_input_dynamic_shape_cleanup(
    context: ModelIRPassContext,
    *,
    shape_reconciler: StaticShapeReconciler = reconcile_static_tensor_shapes,
) -> Tuple[FinalInputDynamicResults, Dict[str, int]]:
    """Run final input/dynamic cleanup and static-shape repair in order."""

    dynamic_results = run_final_input_dynamic_cleanup(context)
    shape_results = shape_reconciler(
        context.model_ir,
        include_mutation_count=True,
    )
    return dynamic_results, shape_results
