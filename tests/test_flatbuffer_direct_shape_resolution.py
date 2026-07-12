from __future__ import annotations

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.shape_resolution import (
    shape_hint_only_adds_singleton_or_dynamic_axes,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def test_detects_hint_that_only_adds_singleton_or_dynamic_axes() -> None:
    assert shape_hint_only_adds_singleton_or_dynamic_axes(
        resolved_shape=[1, 12543],
        shape_hint=[-1, 1, 12543],
    )


def test_does_not_treat_same_rank_layout_permutation_as_inserted_axes() -> None:
    assert (
        shape_hint_only_adds_singleton_or_dynamic_axes(
            resolved_shape=[1, 224, 224, 3],
            shape_hint=[1, 3, 224, 224],
        )
        is False
    )


def test_does_not_override_dynamic_or_different_non_singleton_shapes() -> None:
    assert (
        shape_hint_only_adds_singleton_or_dynamic_axes(
            resolved_shape=[1, 12543],
            shape_hint=[1, -1],
        )
        is False
    )
    assert (
        shape_hint_only_adds_singleton_or_dynamic_axes(
            resolved_shape=[1, 12543],
            shape_hint=[1, 1, 9408],
        )
        is False
    )


def test_context_preserves_static_producer_rank_over_singleton_hint() -> None:
    model_ir = ModelIR("reshape_rank")
    model_ir.tensors["reshaped"] = TensorIR(
        name="reshaped",
        dtype="FLOAT32",
        shape=[1, 12543],
        shape_signature=[1, 12543],
    )
    context = LoweringContext(
        model_ir=model_ir,
        shape_map={"reshaped": [1, 1, 12543]},
        dtype_map={"reshaped": "FLOAT32"},
        constants={},
    )
    context.add_operator(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["source", "shape"],
            outputs=["reshaped"],
            options={
                "newShape": [1, 12543],
                "onnxRawNewShape": [1, -1],
            },
        )
    )

    context.ensure_tensor("reshaped")

    assert model_ir.tensors["reshaped"].shape == [1, 12543]
    assert model_ir.tensors["reshaped"].shape_signature == [1, 12543]


def test_context_retains_legacy_rank_expansion_without_static_producer() -> None:
    model_ir = ModelIR("rank_hint")
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=[1, 12543],
        shape_signature=[1, 12543],
    )
    context = LoweringContext(
        model_ir=model_ir,
        shape_map={"value": [1, 1, 12543]},
        dtype_map={"value": "FLOAT32"},
        constants={},
    )

    context.ensure_tensor("value")

    assert model_ir.tensors["value"].shape == [1, 1, 12543]
    assert model_ir.tensors["value"].shape_signature == [-1, 1, 12543]


def test_context_does_not_preserve_dynamic_producer_rank() -> None:
    model_ir = ModelIR("dynamic_producer")
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=[1, 12543],
        shape_signature=[-1, 12543],
    )
    context = LoweringContext(
        model_ir=model_ir,
        shape_map={"value": [-1, 1, 12543]},
        dtype_map={"value": "FLOAT32"},
        constants={},
    )
    context.add_operator(
        OperatorIR(
            op_type="IDENTITY",
            inputs=["source"],
            outputs=["value"],
        )
    )

    context.ensure_tensor("value")

    assert model_ir.tensors["value"].shape == [1, 1, 12543]
    assert model_ir.tensors["value"].shape_signature == [-1, -1, 12543]
