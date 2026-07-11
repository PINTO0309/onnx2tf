from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.op_contracts import NodeValidationError
from onnx2tf.tflite_builder.op_registry import _validate_prelu


class _PReluValidationContext:
    def __init__(self, *, raw_shape: list[int]) -> None:
        self.shape_map = {"x": list(raw_shape)}

    def get_constant_array(self, tensor_name: str):
        if tensor_name == "slope":
            return np.ones((128,), dtype=np.float32)
        return None

    def get_tensor_shape(self, tensor_name: str) -> list[int]:
        assert tensor_name == "x"
        return [1, 1, 1, 1]


def _prelu_node():
    return SimpleNamespace(
        name="prelu",
        op="PRelu",
        inputs=[SimpleNamespace(name="x"), SimpleNamespace(name="slope")],
        outputs=[SimpleNamespace(name="y")],
    )


def test_prelu_validation_defers_stale_all_ones_placeholder() -> None:
    _validate_prelu(
        _prelu_node(),
        _PReluValidationContext(raw_shape=[]),
    )


def test_prelu_validation_rejects_resolved_channel_mismatch() -> None:
    with pytest.raises(NodeValidationError, match="per-channel"):
        _validate_prelu(
            _prelu_node(),
            _PReluValidationContext(raw_shape=[1, 1, 1, 1]),
        )
