from __future__ import annotations

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.op_builders.control_utils import (
    tensor_shape_is_statically_proven,
)


def _context(*, shape: list[int], signature: list[int], hint: list[int]):
    model_ir = ModelIR("control_shape")
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=shape,
        shape_signature=signature,
    )
    return LoweringContext(
        model_ir=model_ir,
        shape_map={"value": hint},
        dtype_map={"value": "FLOAT32"},
        constants={},
    )


def test_accepts_shape_when_tensor_and_onnx_hint_are_static() -> None:
    context = _context(
        shape=[1, 8],
        signature=[1, 8],
        hint=[1, 8],
    )
    assert tensor_shape_is_statically_proven(ctx=context, tensor_name="value")


def test_rejects_placeholder_shape_when_onnx_hint_is_dynamic() -> None:
    context = _context(
        shape=[1],
        signature=[1],
        hint=[-1, -1],
    )
    assert not tensor_shape_is_statically_proven(ctx=context, tensor_name="value")


def test_rejects_dynamic_tensor_signature_despite_static_hint() -> None:
    context = _context(
        shape=[1, 8],
        signature=[-1, 8],
        hint=[1, 8],
    )
    assert not tensor_shape_is_statically_proven(ctx=context, tensor_name="value")
