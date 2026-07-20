from __future__ import annotations

from typing import Dict, Tuple

from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.passes.late_input_affine_normalization_orchestration import (
    run_late_input_affine_normalization_cleanup,
)
from onnx2tf.tflite_builder.passes.very_late_dynamic_adapter_orchestration import (
    run_very_late_dynamic_adapter_cleanup,
)


FinalInputDynamicContext = ModelIRPassContext
NestedMutationResults = Tuple[Dict[str, int], ...]


def run_final_input_dynamic_cleanup(
    context: FinalInputDynamicContext,
) -> Tuple[NestedMutationResults, NestedMutationResults]:
    """Run final input repair and dynamic-adapter cleanup in fixed order."""

    return (
        run_late_input_affine_normalization_cleanup(context),
        run_very_late_dynamic_adapter_cleanup(context),
    )
