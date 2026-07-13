from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

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
        state.sync_from_model_ir(model_ir)
        return state

    def sync_from_model_ir(self, model_ir: ModelIR) -> None:
        self.logical.clear()
        self.physical.clear()
        for name, tensor in model_ir.tensors.items():
            self.logical[str(name)] = normalize_logical_layout(tensor.logical_layout)
            self.physical[str(name)] = normalize_logical_layout(tensor.physical_layout)

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

    def rename(self, old_name: str, new_name: str) -> None:
        old = str(old_name)
        new = str(new_name)
        if old == new:
            return
        logical = self.logical.pop(old, LOGICAL_LAYOUT_UNKNOWN)
        physical = self.physical.pop(old, LOGICAL_LAYOUT_UNKNOWN)
        self.logical[new] = normalize_logical_layout(logical)
        self.physical[new] = normalize_logical_layout(physical)

    def validate_against_model_ir(self, model_ir: ModelIR) -> List[str]:
        problems: List[str] = []
        tensor_names = set(str(name) for name in model_ir.tensors)
        state_names = set(self.logical) | set(self.physical)
        for name in sorted(tensor_names - state_names):
            problems.append(f"layout_state_missing_tensor:{name}")
        for name in sorted(state_names - tensor_names):
            problems.append(f"layout_state_stale_tensor:{name}")
        for name in sorted(tensor_names & state_names):
            tensor = model_ir.tensors[name]
            expected_logical = normalize_logical_layout(tensor.logical_layout)
            expected_physical = normalize_logical_layout(tensor.physical_layout)
            if self.logical_of(name) != expected_logical:
                problems.append(f"layout_state_logical_mismatch:{name}")
            if self.physical_of(name) != expected_physical:
                problems.append(f"layout_state_physical_mismatch:{name}")
        return problems

    def sync_to_model_ir(self, model_ir: ModelIR) -> None:
        for name, tensor in model_ir.tensors.items():
            layout = self.logical.get(str(name))
            if layout is not None:
                tensor.logical_layout = normalize_logical_layout(layout)
            physical = self.physical.get(str(name))
            if physical is not None:
                tensor.physical_layout = normalize_logical_layout(physical)
