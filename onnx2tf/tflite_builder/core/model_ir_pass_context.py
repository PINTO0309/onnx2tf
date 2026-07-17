from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR


@dataclass(frozen=True)
class ModelIRPassContext:
    """Identity-bound state shared by ModelIR orchestration phases."""

    model_ir: ModelIR
    layout_state: LayoutState | None
    diagnostics: List[Dict[str, Any]]
