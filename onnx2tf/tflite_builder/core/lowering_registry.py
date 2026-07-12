from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


ResolveCallback = Callable[[Any, Any], Any]


@dataclass(frozen=True)
class LoweringResolution:
    onnx_op: str
    entry: Any


class LoweringRegistry:
    """Small dispatch boundary joining validation and lowering.

    During migration the resolver is backed by the legacy declarative table.
    Op-family modules can move behind this boundary independently without
    changing the dispatcher or public conversion API.
    """

    def __init__(self, resolver: ResolveCallback) -> None:
        if not callable(resolver):
            raise TypeError("lowering resolver must be callable")
        self._resolver = resolver

    def resolve(self, node: Any, context: Any) -> LoweringResolution:
        entry = self._resolver(node, context)
        return LoweringResolution(onnx_op=str(node.op), entry=entry)

    def lower(self, node: Any, context: Any) -> None:
        resolution = self.resolve(node, context)
        resolution.entry.builder(node, context)
