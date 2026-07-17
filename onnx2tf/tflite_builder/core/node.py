from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import onnx
from onnx import numpy_helper


class _ValueView:
    """Typed tensor reference exposed through :class:`NodeView`."""

    __slots__ = ("name", "onnx_name", "shape", "dtype")

    def __init__(
        self,
        *,
        name: str,
        onnx_name: str,
        shape: Any,
        dtype: Optional[str],
    ) -> None:
        self.name = name
        self.onnx_name = onnx_name
        self.shape = shape
        self.dtype = dtype


class NodeView:
    """Minimal stable view of an ONNX NodeProto used by op-family builders."""

    def __init__(
        self,
        node: onnx.NodeProto,
        input_name_remap: Optional[Dict[str, str]] = None,
        shape_map: Optional[Dict[str, Any]] = None,
        dtype_map: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = node.name if node.name else node.op_type
        self.op = node.op_type
        self.attrs = {}
        for attribute in node.attribute:
            if attribute.type == onnx.AttributeProto.INT:
                self.attrs[attribute.name] = int(attribute.i)
            elif attribute.type == onnx.AttributeProto.FLOAT:
                self.attrs[attribute.name] = float(attribute.f)
            elif attribute.type == onnx.AttributeProto.INTS:
                self.attrs[attribute.name] = [int(value) for value in attribute.ints]
            elif attribute.type == onnx.AttributeProto.FLOATS:
                self.attrs[attribute.name] = [float(value) for value in attribute.floats]
            elif attribute.type == onnx.AttributeProto.STRING:
                self.attrs[attribute.name] = attribute.s.decode("utf-8")
            elif attribute.type == onnx.AttributeProto.STRINGS:
                self.attrs[attribute.name] = [value.decode("utf-8") for value in attribute.strings]
            elif attribute.type == onnx.AttributeProto.TENSOR:
                self.attrs[attribute.name] = np.asarray(
                    numpy_helper.to_array(attribute.t)
                )
            elif attribute.type == onnx.AttributeProto.GRAPH:
                self.attrs[attribute.name] = attribute.g
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                self.attrs[attribute.name] = list(attribute.graphs)
        remap = input_name_remap if isinstance(input_name_remap, dict) else {}
        shapes = shape_map if isinstance(shape_map, dict) else {}
        dtypes = dtype_map if isinstance(dtype_map, dict) else {}
        self.inputs = [
            _ValueView(
                name=remap.get(name, name) if name else "",
                onnx_name=name if name else "",
                shape=shapes.get(name),
                dtype=dtypes.get(name),
            )
            for name in node.input
        ]
        self.outputs = [
            _ValueView(
                name=name,
                onnx_name=name,
                shape=shapes.get(name),
                dtype=dtypes.get(name),
            )
            for name in node.output
            if name
        ]
