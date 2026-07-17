from __future__ import annotations

import numpy as np
from onnx import helper, numpy_helper

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.op_families.constant import lower_constant_node


def _constant_node(value: np.ndarray):
    return helper.make_node(
        "Constant",
        [],
        ["constant_output"],
        name="constant_node",
        value=numpy_helper.from_array(np.asarray(value)),
    )


def _context(model_ir: ModelIR) -> LoweringContext:
    return LoweringContext(
        model_ir=model_ir,
        shape_map={},
        dtype_map={},
        constants={},
    )


def test_constant_owner_replaces_existing_placeholder_in_place() -> None:
    placeholder = TensorIR(
        name="constant_output",
        dtype="INT32",
        shape=[9],
        shape_signature=[9],
        onnx_tensor_name="constant_output",
    )
    model_ir = ModelIR(
        name="constant_placeholder",
        tensors={"constant_output": placeholder},
    )
    ctx = _context(model_ir)
    value = np.asarray([[1.0, 2.0]], dtype=np.float32)

    lower_constant_node(node=_constant_node(value), ctx=ctx)

    tensor = model_ir.tensors["constant_output"]
    assert tensor is placeholder
    assert tensor.dtype == "FLOAT32"
    assert tensor.shape == [1, 2]
    assert tensor.shape_signature == [1, 2]
    assert tensor.onnx_tensor_name == "constant_output"
    np.testing.assert_array_equal(tensor.data, value)
    np.testing.assert_array_equal(ctx.constants["constant_output"], value)


def test_constant_owner_preserves_output_name_after_reported_collision(
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="constant_collision")
    ctx = _context(model_ir)
    value = np.asarray([3, 4], dtype=np.int64)

    def add_colliding_tensor(_base_name: str, data: np.ndarray) -> str:
        added_value = np.asarray(data)
        model_ir.tensors["constant_output_1"] = TensorIR(
            name="constant_output_1",
            dtype="INT64",
            shape=[2],
            shape_signature=[2],
            data=added_value,
        )
        ctx.constants["constant_output_1"] = added_value
        return "constant_output_1"

    monkeypatch.setattr(ctx, "add_const_tensor", add_colliding_tensor)

    lower_constant_node(node=_constant_node(value), ctx=ctx)

    assert set(model_ir.tensors) == {"constant_output"}
    assert model_ir.tensors["constant_output"].name == "constant_output"
    assert set(ctx.constants) == {"constant_output"}
    np.testing.assert_array_equal(ctx.constants["constant_output"], value)
