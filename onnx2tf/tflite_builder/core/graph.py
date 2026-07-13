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

    The index is built once per conversion session. Rewriters notify the index
    through :meth:`update_node`, :meth:`register_node`, and
    :meth:`unregister_node`, avoiding a complete graph rescan after a local
    mutation. :meth:`refresh` remains the compatibility fallback for external
    mutations that bypass this contract.
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

    def _producer_candidates(self, tensor_name: str) -> List[Any]:
        name = str(tensor_name)
        if name in self.duplicate_producers:
            return list(self.duplicate_producers[name])
        producer = self.producers.get(name)
        return [] if producer is None else [producer]

    def _graph_order(self) -> Dict[int, int]:
        graph = getattr(self.onnx_model, "graph", None)
        return {
            id(node): int(index)
            for index, node in enumerate(list(getattr(graph, "node", [])))
        }

    def _sort_nodes(self, nodes: Iterable[Any]) -> List[Any]:
        order = self._graph_order()
        return sorted(
            list(nodes),
            key=lambda node: (order.get(id(node), len(order)), id(node)),
        )

    def _set_producer_candidates(
        self,
        tensor_name: str,
        candidates: Iterable[Any],
    ) -> None:
        name = str(tensor_name)
        ordered = self._sort_nodes(candidates)
        if len(ordered) == 0:
            self.producers.pop(name, None)
            self.duplicate_producers.pop(name, None)
            return
        self.producers[name] = ordered[-1]
        if len(ordered) > 1:
            self.duplicate_producers[name] = ordered
        else:
            self.duplicate_producers.pop(name, None)

    def _detach_node(
        self,
        node: Any,
        *,
        inputs: Iterable[Any],
        outputs: Iterable[Any],
    ) -> None:
        for name in set(_names(inputs)):
            remaining = [
                consumer
                for consumer in self.consumers.get(name, [])
                if consumer is not node
            ]
            if remaining:
                self.consumers[name] = remaining
            else:
                self.consumers.pop(name, None)
        for name in set(_names(outputs)):
            self._set_producer_candidates(
                name,
                (
                    producer
                    for producer in self._producer_candidates(name)
                    if producer is not node
                ),
            )

    def _attach_node(self, node: Any) -> None:
        for name in _names(getattr(node, "input", [])):
            consumers = list(self.consumers.get(name, []))
            consumers.append(node)
            self.consumers[name] = self._sort_nodes(consumers)
        for name in _names(getattr(node, "output", [])):
            candidates = self._producer_candidates(name)
            candidates.append(node)
            self._set_producer_candidates(name, candidates)

    def update_node(
        self,
        node: Any,
        *,
        previous_inputs: Iterable[Any],
        previous_outputs: Iterable[Any],
    ) -> None:
        """Update references after mutating one existing ONNX node."""

        self._detach_node(
            node,
            inputs=previous_inputs,
            outputs=previous_outputs,
        )
        self._attach_node(node)

    def register_node(self, node: Any) -> None:
        """Register a node already inserted into ``onnx_model.graph.node``."""

        self._attach_node(node)

    def unregister_node(
        self,
        node: Any,
        *,
        previous_inputs: Optional[Iterable[Any]] = None,
        previous_outputs: Optional[Iterable[Any]] = None,
    ) -> None:
        """Remove references for a node being removed from the ONNX graph."""

        self._detach_node(
            node,
            inputs=(
                getattr(node, "input", [])
                if previous_inputs is None
                else previous_inputs
            ),
            outputs=(
                getattr(node, "output", [])
                if previous_outputs is None
                else previous_outputs
            ),
        )


@dataclass
class ModelIRGraphIndex:
    """Producer/consumer index for ModelIR validation and post passes."""

    model_ir: ModelIR
    producers: Dict[str, int] = field(default_factory=dict, init=False)
    consumers: Dict[str, List[int]] = field(default_factory=dict, init=False)
    duplicate_producers: Dict[str, List[int]] = field(default_factory=dict, init=False)
    _operator_indices_by_id: Dict[int, int] = field(default_factory=dict, init=False)
    _operator_indices_by_type: Dict[str, List[int]] = field(
        default_factory=dict,
        init=False,
    )

    def __post_init__(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self.producers.clear()
        self.consumers.clear()
        self.duplicate_producers.clear()
        self._operator_indices_by_id.clear()
        self._operator_indices_by_type.clear()
        for index, op in enumerate(self.model_ir.operators):
            self._operator_indices_by_id[id(op)] = int(index)
            self._operator_indices_by_type.setdefault(
                str(op.op_type),
                [],
            ).append(int(index))
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

    def consumer_indices(self, tensor_name: str) -> List[int]:
        return list(self.consumers.get(str(tensor_name), []))

    def operator_index(self, op: OperatorIR) -> Optional[int]:
        index = self._operator_indices_by_id.get(id(op))
        if index is None or index < 0 or index >= len(self.model_ir.operators):
            return None
        if self.model_ir.operators[index] is not op:
            return None
        return int(index)

    def operator_indices(self, op_type: str) -> List[int]:
        """Return graph-order indices for one current ModelIR operator type."""

        return list(self._operator_indices_by_type.get(str(op_type), []))

    def operator_indices_for_types(
        self,
        op_types: Iterable[str],
    ) -> List[int]:
        """Return graph-order indices for the requested operator-type union."""

        return sorted(
            {
                int(index)
                for op_type in {str(value) for value in op_types}
                for index in self._operator_indices_by_type.get(op_type, [])
            }
        )

    def _producer_indices(self, tensor_name: str) -> List[int]:
        name = str(tensor_name)
        if name in self.duplicate_producers:
            return list(self.duplicate_producers[name])
        producer = self.producers.get(name)
        return [] if producer is None else [int(producer)]

    def _set_producer_indices(
        self,
        tensor_name: str,
        indices: Iterable[int],
    ) -> None:
        name = str(tensor_name)
        ordered = sorted(int(index) for index in indices)
        if len(ordered) == 0:
            self.producers.pop(name, None)
            self.duplicate_producers.pop(name, None)
            return
        self.producers[name] = ordered[-1]
        if len(ordered) > 1:
            self.duplicate_producers[name] = ordered
        else:
            self.duplicate_producers.pop(name, None)

    def _detach_operator(
        self,
        operator_index: int,
        *,
        inputs: Iterable[Any],
        outputs: Iterable[Any],
    ) -> None:
        index = int(operator_index)
        for name in set(_names(inputs)):
            remaining = [
                consumer_index
                for consumer_index in self.consumers.get(name, [])
                if int(consumer_index) != index
            ]
            if remaining:
                self.consumers[name] = remaining
            else:
                self.consumers.pop(name, None)
        for name in set(_names(outputs)):
            self._set_producer_indices(
                name,
                (
                    producer_index
                    for producer_index in self._producer_indices(name)
                    if int(producer_index) != index
                ),
            )

    def _attach_operator(self, operator_index: int, op: OperatorIR) -> None:
        index = int(operator_index)
        for name in _names(op.inputs):
            consumers = list(self.consumers.get(name, []))
            consumers.append(index)
            self.consumers[name] = sorted(consumers)
        for name in _names(op.outputs):
            producers = self._producer_indices(name)
            producers.append(index)
            self._set_producer_indices(name, producers)

    def replace_operator_inputs(
        self,
        operator_index: int,
        new_inputs: Iterable[Any],
    ) -> None:
        """Mutate one operator's inputs and update only affected consumers."""

        index = int(operator_index)
        op = self.model_ir.operators[index]
        old_inputs = list(op.inputs)
        self._detach_operator(
            index,
            inputs=old_inputs,
            outputs=[],
        )
        op.inputs = [str(name) for name in new_inputs]
        for name in _names(op.inputs):
            consumers = list(self.consumers.get(name, []))
            consumers.append(index)
            self.consumers[name] = sorted(consumers)

    def replace_operator_outputs(
        self,
        operator_index: int,
        new_outputs: Iterable[Any],
    ) -> None:
        """Mutate one operator's outputs and update only affected producers."""

        index = int(operator_index)
        op = self.model_ir.operators[index]
        old_outputs = list(op.outputs)
        self._detach_operator(
            index,
            inputs=[],
            outputs=old_outputs,
        )
        op.outputs = [str(name) for name in new_outputs]
        for name in _names(op.outputs):
            producers = self._producer_indices(name)
            producers.append(index)
            self._set_producer_indices(name, producers)

    def replace_operator_type(
        self,
        operator_index: int,
        new_op_type: str,
    ) -> None:
        """Mutate one operator type while maintaining type-index dispatch."""

        index = int(operator_index)
        op = self.model_ir.operators[index]
        old_op_type = str(op.op_type)
        normalized_new_op_type = str(new_op_type)
        if old_op_type == normalized_new_op_type:
            return
        remaining = [
            value
            for value in self._operator_indices_by_type.get(old_op_type, [])
            if int(value) != index
        ]
        if remaining:
            self._operator_indices_by_type[old_op_type] = remaining
        else:
            self._operator_indices_by_type.pop(old_op_type, None)
        op.op_type = normalized_new_op_type
        new_indices = list(
            self._operator_indices_by_type.get(normalized_new_op_type, [])
        )
        new_indices.append(index)
        self._operator_indices_by_type[normalized_new_op_type] = sorted(
            new_indices
        )

    def insert_operator(self, operator_index: int, op: OperatorIR) -> None:
        """Insert an operator while shifting existing index references once."""

        index = int(operator_index)
        if index < 0 or index > len(self.model_ir.operators):
            raise IndexError(f"operator index out of range: {index}")
        self.producers = {
            name: value + 1 if int(value) >= index else int(value)
            for name, value in self.producers.items()
        }
        self.duplicate_producers = {
            name: [value + 1 if int(value) >= index else int(value) for value in values]
            for name, values in self.duplicate_producers.items()
        }
        self.consumers = {
            name: [value + 1 if int(value) >= index else int(value) for value in values]
            for name, values in self.consumers.items()
        }
        self.model_ir.operators.insert(index, op)
        self._operator_indices_by_id = {
            operator_id: value + 1 if int(value) >= index else int(value)
            for operator_id, value in self._operator_indices_by_id.items()
        }
        self._operator_indices_by_id[id(op)] = index
        self._operator_indices_by_type = {
            op_type: [
                value + 1 if int(value) >= index else int(value)
                for value in values
            ]
            for op_type, values in self._operator_indices_by_type.items()
        }
        self._operator_indices_by_type.setdefault(str(op.op_type), []).append(
            index
        )
        self._operator_indices_by_type[str(op.op_type)].sort()
        self._attach_operator(index, op)

    def append_operator(self, op: OperatorIR) -> None:
        self.insert_operator(len(self.model_ir.operators), op)

    def remove_operator(self, operator_index: int) -> OperatorIR:
        """Remove an operator while shifting existing index references once."""

        index = int(operator_index)
        if index < 0 or index >= len(self.model_ir.operators):
            raise IndexError(f"operator index out of range: {index}")
        op = self.model_ir.operators[index]
        self._detach_operator(
            index,
            inputs=op.inputs,
            outputs=op.outputs,
        )
        del self.model_ir.operators[index]
        self._operator_indices_by_id.pop(id(op), None)
        self.producers = {
            name: value - 1 if int(value) > index else int(value)
            for name, value in self.producers.items()
        }
        self.duplicate_producers = {
            name: [value - 1 if int(value) > index else int(value) for value in values]
            for name, values in self.duplicate_producers.items()
        }
        self.consumers = {
            name: [value - 1 if int(value) > index else int(value) for value in values]
            for name, values in self.consumers.items()
        }
        self._operator_indices_by_id = {
            operator_id: value - 1 if int(value) > index else int(value)
            for operator_id, value in self._operator_indices_by_id.items()
        }
        self._operator_indices_by_type = {
            op_type: [
                value - 1 if int(value) > index else int(value)
                for value in values
                if int(value) != index
            ]
            for op_type, values in self._operator_indices_by_type.items()
        }
        return op
