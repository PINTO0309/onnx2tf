from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_nchw_channel_shuffle_concat_gathers,
)


def _tensor(
    model_ir: ModelIR,
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype="INT32" if data is not None else "FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def test_repair_restores_channel_axis_for_nchw_shuffle_gather() -> None:
    model_ir = ModelIR("stale_shuffle_concat_axis")
    model_ir.inputs = ["left", "right"]
    model_ir.outputs = ["shuffled"]
    _tensor(model_ir, "left", [1, 232, 7, 7])
    _tensor(model_ir, "right", [1, 232, 7, 7])
    _tensor(model_ir, "concat", [1, 232, 7, 14])
    _tensor(
        model_ir,
        "indices",
        [464],
        data=np.arange(464, dtype=np.int32),
    )
    _tensor(model_ir, "shuffled", [1, 464, 7, 14])
    model_ir.operators = [
        OperatorIR(
            "CONCATENATION",
            ["left", "right"],
            ["concat"],
            {"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "GATHER",
            ["concat", "indices"],
            ["shuffled"],
            {"axis": 1, "batchDims": 0},
        ),
    ]

    stats = _repair_nchw_channel_shuffle_concat_gathers(model_ir)

    assert stats == {"repaired_nchw_channel_shuffle_concat_gathers": 1}
    assert model_ir.operators[0].options["axis"] == 1
    assert model_ir.tensors["concat"].shape == [1, 464, 7, 7]
    assert model_ir.tensors["shuffled"].shape == [1, 464, 7, 7]
