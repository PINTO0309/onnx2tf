from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)
from onnx2tf.tflite_builder.passes.unbound_input_layout import (
    repair_unbound_nonconstant_inputs_with_layout_transpose,
)


def repair_unbound_nonconstant_operator_inputs_with_layout_transpose(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
) -> Dict[str, int]:
    """Repair eligible unbound inputs and reconcile shapes after mutation."""

    result = repair_unbound_nonconstant_inputs_with_layout_transpose(
        model_ir,
        graph_index=graph_index,
    )
    if result.repaired > 0:
        reconcile_static_tensor_shapes(
            model_ir,
            graph_index=result.graph_index,
        )
    return {
        "repaired_unbound_nonconstant_inputs_with_layout_transpose": int(
            result.repaired
        )
    }
