from __future__ import annotations

from typing import Any

from onnx2tf.tflite_builder.op_registry import validate_node_support


def dispatch_node(node: Any, ctx: Any) -> None:
    entry = validate_node_support(node, ctx)
    entry.builder(node, ctx)
