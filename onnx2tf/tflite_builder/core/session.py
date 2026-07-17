from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from onnx2tf.tflite_builder.core.graph import GraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR


@dataclass
class ConversionSession:
    """State shared by lowering handlers during one conversion only."""

    onnx_model: Any
    model_ir: ModelIR
    shape_map: Dict[str, List[Any]]
    dtype_map: Dict[str, str]
    constants: Dict[str, np.ndarray]
    graph_index: GraphIndex = field(init=False)
    layout_state: LayoutState = field(init=False)
    model_ir_pass_context: ModelIRPassContext = field(init=False)
    diagnostics: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.graph_index = GraphIndex(self.onnx_model)
        self.layout_state = LayoutState.from_model_ir(self.model_ir)
        self.model_ir_pass_context = ModelIRPassContext(
            model_ir=self.model_ir,
            layout_state=self.layout_state,
            diagnostics=self.diagnostics,
        )

    @property
    def tensor_consumer_count(self) -> Dict[str, int]:
        return {
            name: len(consumers)
            for name, consumers in self.graph_index.consumers.items()
        }

    def refresh_indexes(self) -> None:
        self.graph_index.refresh()
        self.layout_state.sync_from_model_ir(self.model_ir)

    def record_diagnostic(self, *, stage: str, code: str, message: str) -> None:
        self.diagnostics.append(
            {"stage": str(stage), "code": str(code), "message": str(message)}
        )
