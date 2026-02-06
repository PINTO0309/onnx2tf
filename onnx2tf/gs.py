from __future__ import annotations

from collections import deque
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _ensure_numpy_dtype(dtype: Any) -> Optional[np.dtype]:
    if dtype is None:
        return None
    if isinstance(dtype, np.dtype):
        return dtype
    if isinstance(dtype, int):
        return np.dtype(helper.tensor_dtype_to_np_dtype(dtype))
    if hasattr(dtype, "as_numpy_dtype"):
        return np.dtype(dtype.as_numpy_dtype)
    try:
        return np.dtype(dtype)
    except Exception:
        return None


def _onnx_dtype_to_numpy(elem_type: int) -> Optional[np.dtype]:
    try:
        return np.dtype(helper.tensor_dtype_to_np_dtype(elem_type))
    except Exception:
        return None


def _numpy_dtype_to_onnx(dtype: Any) -> int:
    if isinstance(dtype, int):
        return dtype
    np_dtype = _ensure_numpy_dtype(dtype)
    if np_dtype is None:
        return onnx.TensorProto.FLOAT
    try:
        return helper.np_dtype_to_tensor_dtype(np_dtype)
    except Exception:
        pass
    if np_dtype == np.dtype("bool"):
        return onnx.TensorProto.BOOL
    if np_dtype == np.dtype("float16"):
        return onnx.TensorProto.FLOAT16
    if np_dtype == np.dtype("float32"):
        return onnx.TensorProto.FLOAT
    if np_dtype == np.dtype("float64"):
        return onnx.TensorProto.DOUBLE
    if np_dtype == np.dtype("int8"):
        return onnx.TensorProto.INT8
    if np_dtype == np.dtype("int16"):
        return onnx.TensorProto.INT16
    if np_dtype == np.dtype("int32"):
        return onnx.TensorProto.INT32
    if np_dtype == np.dtype("int64"):
        return onnx.TensorProto.INT64
    if np_dtype == np.dtype("uint8"):
        return onnx.TensorProto.UINT8
    if np_dtype == np.dtype("uint16"):
        return onnx.TensorProto.UINT16
    if np_dtype == np.dtype("uint32"):
        return onnx.TensorProto.UINT32
    if np_dtype == np.dtype("uint64"):
        return onnx.TensorProto.UINT64
    return onnx.TensorProto.FLOAT


def _parse_dim(dim: onnx.TensorShapeProto.Dimension) -> Optional[int | str]:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    if dim.HasField("dim_param"):
        return dim.dim_param
    return None


def _parse_tensor_shape(value_info: onnx.ValueInfoProto) -> Optional[List[int | str | None]]:
    if not value_info.type.HasField("tensor_type"):
        return None
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape"):
        return None
    return [_parse_dim(dim) for dim in tensor_type.shape.dim]


@dataclass(eq=False)
class Variable:
    name: str = ""
    dtype: Any = None
    shape: Optional[List[int | str | None]] = None
    inputs: List[Any] = field(default_factory=list)
    outputs: List[Any] = field(default_factory=list)

    def is_empty(self) -> bool:
        return self.name == ""

    def i(self, producer_idx: int = 0) -> Any:
        if producer_idx >= len(self.inputs):
            raise IndexError("producer index out of range")
        return self.inputs[producer_idx]

    def o(self, consumer_idx: int = 0) -> Any:
        if consumer_idx >= len(self.outputs):
            raise IndexError("consumer index out of range")
        return self.outputs[consumer_idx]


@dataclass(eq=False)
class Constant(Variable):
    values: np.ndarray = field(default_factory=lambda: np.asarray(0, dtype=np.float32))

    def __init__(
        self,
        name: str = "",
        values: Any = None,
        dtype: Any = None,
        shape: Optional[List[int | str | None]] = None,
    ):
        np_values = np.asarray(values) if values is not None else np.asarray(0, dtype=np.float32)
        resolved_dtype = _ensure_numpy_dtype(dtype) if dtype is not None else np_values.dtype
        if resolved_dtype is not None and np_values.dtype != resolved_dtype:
            np_values = np_values.astype(resolved_dtype)
        resolved_shape = shape if shape is not None else list(np_values.shape)
        super().__init__(
            name=name,
            dtype=np_values.dtype,
            shape=resolved_shape,
            inputs=[],
            outputs=[],
        )
        self.values = np_values

    def is_empty(self) -> bool:
        return False


@dataclass(eq=False)
class Node:
    op: str
    name: str = ""
    inputs: List[Variable | Constant] = field(default_factory=list)
    outputs: List[Variable] = field(default_factory=list)
    attrs: Dict[str, Any] = field(default_factory=dict)
    id: Optional[int] = None

    def __post_init__(self):
        for inp in self.inputs:
            if self not in inp.outputs:
                inp.outputs.append(self)
        for out in self.outputs:
            if self not in out.inputs:
                out.inputs.append(self)

    def i(self, tensor_idx: int = 0, producer_idx: int = 0) -> Any:
        if tensor_idx >= len(self.inputs):
            raise IndexError("input tensor index out of range")
        tensor = self.inputs[tensor_idx]
        producers = getattr(tensor, "inputs", [])
        if producer_idx >= len(producers):
            raise IndexError("producer index out of range")
        return producers[producer_idx]

    def o(self, consumer_idx: int = 0, tensor_idx: int = 0) -> Any:
        if tensor_idx >= len(self.outputs):
            raise IndexError("output tensor index out of range")
        tensor = self.outputs[tensor_idx]
        consumers = getattr(tensor, "outputs", [])
        if consumer_idx >= len(consumers):
            raise IndexError("consumer index out of range")
        return consumers[consumer_idx]


class _NodeIDContext(AbstractContextManager):
    def __init__(self, graph: "Graph"):
        self.graph = graph

    def __enter__(self) -> "Graph":
        for idx, node in enumerate(self.graph.nodes):
            node.id = idx
        return self.graph

    def __exit__(self, exc_type, exc, exc_tb) -> bool:
        return False


@dataclass
class Graph:
    nodes: List[Node] = field(default_factory=list)
    inputs: List[Variable | Constant] = field(default_factory=list)
    outputs: List[Variable | Constant] = field(default_factory=list)
    opset: int = 13
    name: str = ""

    def node_ids(self) -> _NodeIDContext:
        return _NodeIDContext(self)

    def _iter_tensors(self) -> Iterable[Variable | Constant]:
        seen = set()

        def _yield_tensor(tensor: Variable | Constant):
            key = id(tensor)
            if key in seen:
                return
            seen.add(key)
            yield tensor

        for tensor in self.inputs:
            yield from _yield_tensor(tensor)
        for tensor in self.outputs:
            yield from _yield_tensor(tensor)
        for node in self.nodes:
            for tensor in node.inputs:
                yield from _yield_tensor(tensor)
            for tensor in node.outputs:
                yield from _yield_tensor(tensor)

    def _rebuild_edges(self) -> None:
        for tensor in self._iter_tensors():
            tensor.inputs = []
            tensor.outputs = []
        for node in self.nodes:
            for inp in node.inputs:
                if node not in inp.outputs:
                    inp.outputs.append(node)
            for out in node.outputs:
                if node not in out.inputs:
                    out.inputs.append(node)
        # Keep GraphSurgeon-like behavior where Constant node outputs can expose
        # the constant value as the first producer via `tensor.i()`.
        for node in self.nodes:
            if node.op == "Constant" and isinstance(node.attrs.get("value"), Constant):
                const_value = node.attrs["value"]
                for out in node.outputs:
                    if const_value not in out.inputs:
                        out.inputs.insert(0, const_value)

    def cleanup(self) -> "Graph":
        self._rebuild_edges()
        node_set = set(self.nodes)
        required_nodes = set()
        stack = []
        for out in self.outputs:
            for producer in getattr(out, "inputs", []):
                if isinstance(producer, Node) and producer in node_set:
                    stack.append(producer)

        while stack:
            node = stack.pop()
            if node in required_nodes:
                continue
            required_nodes.add(node)
            for inp in node.inputs:
                for producer in getattr(inp, "inputs", []):
                    if isinstance(producer, Node) and producer in node_set and producer not in required_nodes:
                        stack.append(producer)

        self.nodes = [node for node in self.nodes if node in required_nodes]
        self._rebuild_edges()
        return self

    def toposort(self) -> "Graph":
        self._rebuild_edges()
        node_to_idx = {node: idx for idx, node in enumerate(self.nodes)}
        indegree = {node: 0 for node in self.nodes}
        dependents = {node: [] for node in self.nodes}

        for node in self.nodes:
            deps = set()
            for inp in node.inputs:
                for producer in getattr(inp, "inputs", []):
                    if isinstance(producer, Node) and producer in node_to_idx and producer is not node:
                        deps.add(producer)
            indegree[node] = len(deps)
            for dep in deps:
                dependents[dep].append(node)

        ready = deque(sorted([node for node, deg in indegree.items() if deg == 0], key=lambda n: node_to_idx[n]))
        sorted_nodes: List[Node] = []
        while ready:
            node = ready.popleft()
            sorted_nodes.append(node)
            for nxt in dependents[node]:
                indegree[nxt] -= 1
                if indegree[nxt] == 0:
                    ready.append(nxt)

        if len(sorted_nodes) != len(self.nodes):
            in_sorted = {id(node) for node in sorted_nodes}
            sorted_nodes.extend([node for node in self.nodes if id(node) not in in_sorted])

        self.nodes = sorted_nodes
        self._rebuild_edges()
        return self


def _sanitize_string_attr(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, list):
        return [_sanitize_string_attr(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_sanitize_string_attr(v) for v in value)
    return value


def _make_constant_from_tensor(name: str, tensor_proto: onnx.TensorProto) -> Constant:
    values = numpy_helper.to_array(tensor_proto)
    return Constant(name=name, values=values)


def _import_graph_proto(graph_proto: onnx.GraphProto, opset: int) -> Graph:
    value_info_by_name: Dict[str, Dict[str, Any]] = {}
    for value_info in list(graph_proto.input) + list(graph_proto.value_info) + list(graph_proto.output):
        if not value_info.name:
            continue
        dtype = None
        if value_info.type.HasField("tensor_type"):
            dtype = _onnx_dtype_to_numpy(value_info.type.tensor_type.elem_type)
        value_info_by_name[value_info.name] = {
            "dtype": dtype,
            "shape": _parse_tensor_shape(value_info),
        }

    initializer_by_name: Dict[str, Constant] = {}
    for initializer in graph_proto.initializer:
        initializer_by_name[initializer.name] = _make_constant_from_tensor(initializer.name, initializer)

    tensors_by_name: Dict[str, Variable | Constant] = dict(initializer_by_name)

    def _get_or_create_tensor(name: str) -> Variable | Constant:
        if name in tensors_by_name:
            return tensors_by_name[name]
        if name == "":
            tensor = Variable(name="", dtype=None, shape=None)
            tensors_by_name[name] = tensor
            return tensor
        info = value_info_by_name.get(name, {})
        tensor = Variable(
            name=name,
            dtype=info.get("dtype"),
            shape=info.get("shape"),
        )
        tensors_by_name[name] = tensor
        return tensor

    imported_nodes: List[Node] = []
    for node_proto in graph_proto.node:
        node_inputs: List[Variable | Constant] = [_get_or_create_tensor(name) for name in node_proto.input]
        node_outputs: List[Variable] = []
        for out_name in node_proto.output:
            tensor = _get_or_create_tensor(out_name)
            if isinstance(tensor, Constant):
                info = value_info_by_name.get(out_name, {})
                tensor = Variable(name=out_name, dtype=info.get("dtype"), shape=info.get("shape"))
                tensors_by_name[out_name] = tensor
            node_outputs.append(tensor)

        attrs: Dict[str, Any] = {}
        for attr in node_proto.attribute:
            attr_val = onnx.helper.get_attribute_value(attr)
            attr_val = _sanitize_string_attr(attr_val)
            if isinstance(attr_val, onnx.GraphProto):
                attrs[attr.name] = _import_graph_proto(attr_val, opset)
            elif isinstance(attr_val, list):
                converted = []
                for item in attr_val:
                    if isinstance(item, onnx.GraphProto):
                        converted.append(_import_graph_proto(item, opset))
                    else:
                        converted.append(_sanitize_string_attr(item))
                attrs[attr.name] = converted
            elif isinstance(attr_val, onnx.TensorProto) and node_proto.op_type == "Constant" and attr.name == "value":
                const_name = node_outputs[0].name if node_outputs else (node_proto.name or attr.name)
                attrs[attr.name] = _make_constant_from_tensor(const_name, attr_val)
            else:
                attrs[attr.name] = attr_val

        imported_nodes.append(
            Node(
                op=node_proto.op_type,
                name=node_proto.name,
                inputs=node_inputs,
                outputs=node_outputs,
                attrs=attrs,
            )
        )

    graph_inputs: List[Variable | Constant] = []
    initializer_names = set(initializer_by_name.keys())
    for graph_input in graph_proto.input:
        if graph_input.name in initializer_names:
            continue
        graph_inputs.append(_get_or_create_tensor(graph_input.name))

    graph_outputs: List[Variable | Constant] = []
    for graph_output in graph_proto.output:
        graph_outputs.append(_get_or_create_tensor(graph_output.name))

    graph = Graph(
        nodes=imported_nodes,
        inputs=graph_inputs,
        outputs=graph_outputs,
        opset=opset,
        name=graph_proto.name,
    )
    graph._rebuild_edges()
    return graph


def import_onnx(model: onnx.ModelProto) -> Graph:
    if isinstance(model, onnx.GraphProto):
        model = helper.make_model(model)
    if not isinstance(model, onnx.ModelProto):
        raise TypeError("import_onnx expects an onnx.ModelProto or onnx.GraphProto")

    opset = 13
    for opset_import in model.opset_import:
        if opset_import.domain in ("", "ai.onnx"):
            opset = int(opset_import.version)
            break

    graph = _import_graph_proto(model.graph, opset)
    graph.name = model.graph.name
    return graph


def _normalize_shape(shape: Any) -> Optional[List[Any]]:
    if shape is None:
        return None
    if isinstance(shape, tuple):
        shape = list(shape)
    if not isinstance(shape, list):
        return None
    normalized = []
    for dim in shape:
        if isinstance(dim, (int, np.integer)):
            normalized.append(int(dim))
        elif isinstance(dim, str):
            normalized.append(dim)
        else:
            normalized.append(None)
    return normalized


def _tensor_to_initializer_proto(const: Constant, fallback_name: str) -> onnx.TensorProto:
    name = const.name if const.name else fallback_name
    return numpy_helper.from_array(np.asarray(const.values), name=name)


def _export_graph_proto(graph: Graph) -> onnx.GraphProto:
    graph._rebuild_edges()

    initializers: Dict[str, onnx.TensorProto] = {}
    constant_name_by_id: Dict[int, str] = {}
    const_counter = 0

    def _register_constant(const: Constant) -> str:
        nonlocal const_counter
        name = const.name
        if not name:
            name = f"_const_{const_counter}"
            const_counter += 1
        const.name = name
        if name not in initializers:
            initializers[name] = _tensor_to_initializer_proto(const, name)
        constant_name_by_id[id(const)] = name
        return name

    def _convert_attr_value(attr_name: str, value: Any) -> Any:
        if isinstance(value, Graph):
            return _export_graph_proto(value)
        if isinstance(value, Constant):
            return _tensor_to_initializer_proto(value, value.name or attr_name)
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                return value.item()
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, tuple):
            return [_convert_attr_value(attr_name, item) for item in value]
        if isinstance(value, list):
            return [_convert_attr_value(attr_name, item) for item in value]
        if isinstance(value, bytes):
            return value.decode("utf-8", errors="replace")
        return value

    serialized_nodes = []
    for node in graph.nodes:
        input_names = []
        for inp in node.inputs:
            if isinstance(inp, Constant):
                input_names.append(_register_constant(inp))
            else:
                input_names.append(inp.name)
        output_names = [out.name for out in node.outputs]

        attrs: Dict[str, Any] = {}
        for key, value in node.attrs.items():
            converted = _convert_attr_value(key, value)
            if converted is not None:
                attrs[key] = converted

        serialized_nodes.append(
            helper.make_node(
                node.op,
                input_names,
                output_names,
                name=node.name,
                **attrs,
            )
        )

    initializer_names = set(initializers.keys())

    serialized_inputs = []
    for graph_input in graph.inputs:
        if isinstance(graph_input, Constant):
            _register_constant(graph_input)
            initializer_names.add(graph_input.name)
            continue
        if graph_input.name in initializer_names or graph_input.name == "":
            continue
        serialized_inputs.append(
            helper.make_tensor_value_info(
                graph_input.name,
                _numpy_dtype_to_onnx(graph_input.dtype),
                _normalize_shape(graph_input.shape),
            )
        )

    serialized_outputs = []
    for graph_output in graph.outputs:
        if isinstance(graph_output, Constant):
            _register_constant(graph_output)
            dtype = graph_output.values.dtype
            shape = list(graph_output.values.shape)
        else:
            dtype = graph_output.dtype
            shape = graph_output.shape
        if graph_output.name == "":
            continue
        serialized_outputs.append(
            helper.make_tensor_value_info(
                graph_output.name,
                _numpy_dtype_to_onnx(dtype),
                _normalize_shape(shape),
            )
        )

    return helper.make_graph(
        nodes=serialized_nodes,
        name=graph.name or "graph",
        inputs=serialized_inputs,
        outputs=serialized_outputs,
        initializer=list(initializers.values()),
    )


def export_onnx(graph: Graph, do_type_check: bool = False, **kwargs: Any) -> onnx.ModelProto:
    del do_type_check
    if not isinstance(graph, Graph):
        raise TypeError("export_onnx expects a Graph")

    opset = int(getattr(graph, "opset", 13) or 13)
    graph_proto = _export_graph_proto(graph)
    model = helper.make_model(
        graph_proto,
        opset_imports=[helper.make_opsetid("", opset)],
    )

    for key, value in kwargs.items():
        if hasattr(model, key):
            setattr(model, key, value)
    return model


__all__ = [
    "Graph",
    "Node",
    "Variable",
    "Constant",
    "import_onnx",
    "export_onnx",
]
