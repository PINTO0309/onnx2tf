from __future__ import annotations

from typing import Dict

from onnx2tf.tflite_builder.core.model_ir_utils import (
    _topologically_sort_operators,
)
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    validate_model_ir_layout_annotations,
)


_VALIDATION_METADATA_KEY = "logical_layout_validation_errors"


def run_topology_layout_validation(model_ir: ModelIR) -> Dict[str, int]:
    """Sort operators, validate logical layouts, and update diagnostics."""

    sort_stats = _topologically_sort_operators(model_ir)
    layout_problems = validate_model_ir_layout_annotations(model_ir)
    if layout_problems:
        model_ir.metadata[_VALIDATION_METADATA_KEY] = list(layout_problems)
    else:
        model_ir.metadata.pop(_VALIDATION_METADATA_KEY, None)
    return {
        "reordered_operators": int(sort_stats.get("reordered_operators", 0)),
        "cycle_detected": int(sort_stats.get("cycle_detected", 0)),
        "layout_validation_errors": int(len(layout_problems)),
    }
