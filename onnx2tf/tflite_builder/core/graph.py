from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR


def _names(values: Iterable[Any]) -> Iterable[str]:
    for value in values:
        name = str(value).strip()
        if name:
            yield name


@dataclass
class GraphIndex:
    """Producer/consumer index for an ONNX graph.

    The index is built once per conversion session.  Rewriters that mutate the
    ONNX graph may call :meth:`refresh`; incremental mutation methods can be
    introduced without changing consumers of this contract.
    """

    onnx_model: Any
    producers: Dict[str, Any] = field(default_factory=dict, init=False)
    consumers: Dict[str, List[Any]] = field(default_factory=dict, init=False)
    duplicate_producers: Dict[str, List[Any]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.producers.clear()
        self.consumers.clear()
        self.duplicate_producers.clear()
        graph = getattr(self.onnx_model, "graph", None)
        for node in list(getattr(graph, "node", [])):
            for name in _names(getattr(node, "output", [])):
                previous = self.producers.get(name)
                if previous is not None:
                    self.duplicate_producers.setdefault(name, [previous]).append(node)
                self.producers[name] = node
            for name in _names(getattr(node, "input", [])):
                self.consumers.setdefault(name, []).append(node)

    def producer(self, tensor_name: str) -> Optional[Any]:
        return self.producers.get(str(tensor_name))

    def consumers_of(self, tensor_name: str) -> List[Any]:
        return list(self.consumers.get(str(tensor_name), []))

    def consumer_count(self, tensor_name: str) -> int:
        return len(self.consumers.get(str(tensor_name), []))


@dataclass
class ModelIRGraphIndex:
    """Producer/consumer index for ModelIR validation and post passes."""

    model_ir: ModelIR
    producers: Dict[str, int] = field(default_factory=dict, init=False)
    consumers: Dict[str, List[int]] = field(default_factory=dict, init=False)
    duplicate_producers: Dict[str, List[int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.producers.clear()
        self.consumers.clear()
        self.duplicate_producers.clear()
        for index, op in enumerate(self.model_ir.operators):
            for name in _names(op.outputs):
                previous = self.producers.get(name)
                if previous is not None:
                    self.duplicate_producers.setdefault(name, [previous]).append(index)
                self.producers[name] = index
            for name in _names(op.inputs):
                self.consumers.setdefault(name, []).append(index)

    def producer(self, tensor_name: str) -> Optional[OperatorIR]:
        index = self.producers.get(str(tensor_name))
        return self.model_ir.operators[index] if index is not None else None

    def consumers_of(self, tensor_name: str) -> List[OperatorIR]:
        return [
            self.model_ir.operators[index]
            for index in self.consumers.get(str(tensor_name), [])
        ]
