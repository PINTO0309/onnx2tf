from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, List, Mapping

import numpy as np

from onnx2tf.tflite_builder.core.graph import GraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.pass_diagnostics import ModelIRPassDiagnostics
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
    diagnostics: List[Dict[str, Any]] = field(
        default_factory=ModelIRPassDiagnostics
    )
    _phase_results: Dict[str, Dict[str, int]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

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

    def record_phase_result(
        self,
        phase_id: str,
        counters: Mapping[str, Any],
    ) -> None:
        """Retain one bounded, integer-only internal phase result."""

        normalized_phase_id = str(phase_id).strip()
        if not normalized_phase_id:
            raise ValueError("phase_id must not be empty")
        if len(counters) > 32:
            raise ValueError("phase result must contain at most 32 counters")
        if (
            normalized_phase_id not in self._phase_results
            and len(self._phase_results) >= 128
        ):
            raise ValueError("phase result store must contain at most 128 phases")

        normalized_counters: Dict[str, int] = {}
        for name, value in counters.items():
            if not isinstance(value, Integral):
                raise TypeError("phase result counters must be integers")
            normalized_counters[str(name)] = int(value)
        self._phase_results[normalized_phase_id] = normalized_counters

    def phase_results_snapshot(self) -> Dict[str, Dict[str, int]]:
        """Return an isolated snapshot of bounded internal phase results."""

        return {
            str(phase_id): dict(counters)
            for phase_id, counters in self._phase_results.items()
        }
