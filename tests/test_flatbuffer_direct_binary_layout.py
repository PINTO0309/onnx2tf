from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_rank4_channelwise_broadcast_constants_to_runtime_layout,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        logical_layout=layout,
    )


def test_repair_stale_nhwc_constant_back_to_broadcastable_spatial_shape() -> None:
    model_ir = ModelIR("stale_nhwc_spatial_constant")
    model_ir.tensors = {
        "coordinate": _tensor(
            "coordinate",
            [1, 64, 1, 1],
            data=np.arange(64, dtype=np.float32).reshape(1, 64, 1, 1),
            layout="NHWC",
        ),
        "x": _tensor("x", [1, 32, 64, 3], layout="NHWC"),
        "y": _tensor("y", [1, 32, 64, 3], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR(op_type="SUB", inputs=["coordinate", "x"], outputs=["y"])
    ]

    stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
        model_ir
    )

    assert stats == {"repaired_rank4_channelwise_broadcast_constants": 1}
    assert model_ir.tensors["coordinate"].shape == [1, 1, 64, 1]
    np.testing.assert_array_equal(
        model_ir.tensors["coordinate"].data.reshape(-1),
        np.arange(64, dtype=np.float32),
    )


def test_broadcast_constant_repair_keeps_already_compatible_spatial_shape() -> None:
    coordinate = np.arange(64, dtype=np.float32).reshape(1, 1, 64, 1)
    model_ir = ModelIR("compatible_spatial_constant")
    model_ir.tensors = {
        "coordinate": _tensor(
            "coordinate", [1, 1, 64, 1], data=coordinate, layout="NHWC"
        ),
        "x": _tensor("x", [1, 32, 64, 3], layout="NHWC"),
        "y": _tensor("y", [1, 32, 64, 3], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR(op_type="SUB", inputs=["coordinate", "x"], outputs=["y"])
    ]

    stats = _repair_rank4_channelwise_broadcast_constants_to_runtime_layout(
        model_ir
    )

    assert stats == {"repaired_rank4_channelwise_broadcast_constants": 0}
    np.testing.assert_array_equal(model_ir.tensors["coordinate"].data, coordinate)
