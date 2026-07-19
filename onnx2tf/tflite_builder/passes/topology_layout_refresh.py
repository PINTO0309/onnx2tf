from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    infer_model_ir_logical_layouts,
)


def run_topology_layout_refresh(model_ir: ModelIR) -> Dict[str, int]:
    """Sort operators, refresh logical layouts, and retain small sort stats."""

    sort_stats = _topologically_sort_operators(model_ir)
    infer_model_ir_logical_layouts(model_ir)
    return {
        "reordered_operators": int(sort_stats.get("reordered_operators", 0)),
        "cycle_detected": int(sort_stats.get("cycle_detected", 0)),
    }
