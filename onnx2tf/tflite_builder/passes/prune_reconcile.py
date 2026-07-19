from __future__ import annotations

from typing import Dict, Optional

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.graph_cleanup import prune_dead_operators
from onnx2tf.tflite_builder.passes.static_shape_reconciliation import (
    reconcile_static_tensor_shapes,
)


def run_indexed_prune_reconcile_cleanup(
    model_ir: ModelIR,
    *,
    graph_index: Optional[ModelIRGraphIndex] = None,
    layout_state: Optional[LayoutState] = None,
) -> Dict[str, int]:
    """Prune dead operators and reconcile shapes with one graph index."""

    graph_index = (
        graph_index
        if graph_index is not None and graph_index.model_ir is model_ir
        else ModelIRGraphIndex(model_ir)
    )
    prune_stats = prune_dead_operators(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )
    reconcile_stats = reconcile_static_tensor_shapes(
        model_ir,
        graph_index=graph_index,
    )
    return {
        "removed_dead_operators": int(
            prune_stats.get("removed_dead_operators", 0)
        ),
        "reconciled_static_tensor_shapes": int(
            reconcile_stats.get("reconciled_static_tensor_shapes", 0)
        ),
    }
