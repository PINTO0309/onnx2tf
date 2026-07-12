from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.op_builders.scatter_utils import (
    add_zero_safe_runtime_scatter_shape,
)


def test_adds_runtime_shape_clamp_for_zero_safe_scatter() -> None:
    model_ir = ModelIR("zero_safe_scatter")
    model_ir.tensors["data"] = TensorIR(
        name="data",
        dtype="FLOAT32",
        shape=[1, 256, 7, 7],
        shape_signature=[-1, 256, 7, 7],
    )
    context = LoweringContext(
        model_ir=model_ir,
        shape_map={"data": [-1, 256, 7, 7]},
        dtype_map={"data": "FLOAT32"},
        constants={},
    )

    shape_name = add_zero_safe_runtime_scatter_shape(
        ctx=context,
        data_name="data",
        name_prefix="scatter_shape",
        rank=4,
    )

    assert shape_name == "scatter_shape"
    assert [op.op_type for op in model_ir.operators] == ["SHAPE", "MAXIMUM"]
    maximum = model_ir.operators[-1]
    np.testing.assert_array_equal(
        model_ir.tensors[maximum.inputs[1]].data,
        np.ones((4,), dtype=np.int32),
    )
    assert maximum.outputs == [shape_name]
