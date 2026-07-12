from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional

from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    normalize_logical_layout,
)


@dataclass
class LayoutState:
    """Single session-owned source of logical and physical tensor layouts."""

    logical: Dict[str, str] = field(default_factory=dict)
    physical: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_model_ir(cls, model_ir: ModelIR) -> "LayoutState":
        state = cls()
        for name, tensor in model_ir.tensors.items():
            state.logical[str(name)] = normalize_logical_layout(tensor.logical_layout)
            state.physical[str(name)] = normalize_logical_layout(tensor.physical_layout)
        return state

    def logical_of(self, tensor_name: str) -> str:
        return self.logical.get(str(tensor_name), LOGICAL_LAYOUT_UNKNOWN)

    def physical_of(self, tensor_name: str) -> str:
        return self.physical.get(str(tensor_name), LOGICAL_LAYOUT_UNKNOWN)

    def set(
        self,
        tensor_name: str,
        *,
        logical: Optional[str] = None,
        physical: Optional[str] = None,
    ) -> None:
        name = str(tensor_name)
        if logical is not None:
            self.logical[name] = normalize_logical_layout(logical)
        if physical is not None:
            self.physical[name] = normalize_logical_layout(physical)

    def remove(self, tensor_names: Iterable[str]) -> None:
        for name in tensor_names:
            self.logical.pop(str(name), None)
            self.physical.pop(str(name), None)

    def sync_to_model_ir(self, model_ir: ModelIR) -> None:
        for name, tensor in model_ir.tensors.items():
            layout = self.logical.get(str(name))
            if layout is not None:
                tensor.logical_layout = normalize_logical_layout(layout)
            physical = self.physical.get(str(name))
            if physical is not None:
                tensor.physical_layout = normalize_logical_layout(physical)
