from __future__ import annotations

from typing import Dict, Optional

import onnx


class NodeView:
    """Minimal stable view of an ONNX NodeProto used by op-family builders."""

    def __init__(
        self,
        node: onnx.NodeProto,
        input_name_remap: Optional[Dict[str, str]] = None,
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
            elif attribute.type == onnx.AttributeProto.GRAPH:
                self.attrs[attribute.name] = attribute.g
            elif attribute.type == onnx.AttributeProto.GRAPHS:
                self.attrs[attribute.name] = list(attribute.graphs)
        remap = input_name_remap if isinstance(input_name_remap, dict) else {}
        self.inputs = [
            type("In", (), {"name": remap.get(name, name) if name else ""})
            for name in node.input
        ]
        self.outputs = [
            type("Out", (), {"name": name}) for name in node.output if name
        ]
