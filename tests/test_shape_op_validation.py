from __future__ import annotations

from types import SimpleNamespace

import pytest

from onnx2tf.tflite_builder.core.op_contracts import NodeValidationError
from onnx2tf.tflite_builder.op_families.shape import _validate_range


class _ShapeContext:
    def __init__(self, shapes):
        self._shapes = shapes

    def get_tensor_shape(self, name):
        return list(self._shapes[name])


def _range_node():
    return SimpleNamespace(
        name="range",
        op="Range",
        inputs=[
            SimpleNamespace(name="start"),
            SimpleNamespace(name="limit"),
            SimpleNamespace(name="delta"),
        ],
    )


def test_range_validation_accepts_scalar_and_single_element_inputs() -> None:
    _validate_range(
        _range_node(),
        _ShapeContext({"start": [], "limit": [1], "delta": []}),
    )


def test_range_validation_rejects_non_scalar_input() -> None:
    with pytest.raises(NodeValidationError) as exc_info:
        _validate_range(
            _range_node(),
            _ShapeContext({"start": [], "limit": [2], "delta": []}),
        )

    assert exc_info.value.reason_code == "unsupported_input_shape"
    assert "input_index=1 input_shape=[2]" in str(exc_info.value)
