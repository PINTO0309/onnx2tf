from __future__ import annotations

from onnx2tf.tflite_builder.core.lowering_context import LoweringContext
from onnx2tf.tflite_builder.core.shape_resolution import (
    preserve_rewritten_output_dynamic_axes,
    shape_hint_only_adds_singleton_or_dynamic_axes,
    static_shape_vector_length,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import _set_operator_outputs


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


def test_reads_static_shape_vector_length() -> None:
    tensor = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
    )
    assert static_shape_vector_length(tensor) == 2


def test_rejects_dynamic_shape_vector_length() -> None:
    tensor = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    assert static_shape_vector_length(tensor) is None


def test_rewritten_output_preserves_dynamic_source_axes() -> None:
    source = TensorIR(
        name="source",
        dtype="FLOAT32",
        shape=[1, 28, 28, 256],
        shape_signature=[-1, -1, -1, 256],
    )
    target = TensorIR(
        name="target",
        dtype="FLOAT32",
        shape=[1, 28, 28, 256],
        shape_signature=[1, 28, 28, 256],
    )

    assert preserve_rewritten_output_dynamic_axes(
        source_tensor=source,
        target_tensor=target,
    )
    assert target.shape_signature == [-1, -1, -1, 256]


def test_rewritten_output_rejects_rank_mismatch() -> None:
    source = TensorIR(
        name="source",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[-1, 4],
    )
    target = TensorIR(
        name="target",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
    )

    assert not preserve_rewritten_output_dynamic_axes(
        source_tensor=source,
        target_tensor=target,
    )
    assert target.shape_signature == [4]


def test_set_operator_outputs_preserves_dynamic_source_axes() -> None:
    model_ir = ModelIR("rewritten_output")
    model_ir.tensors["source"] = TensorIR(
        name="source",
        dtype="FLOAT32",
        shape=[1, 28, 28, 256],
        shape_signature=[-1, -1, -1, 256],
    )
    model_ir.tensors["target"] = TensorIR(
        name="target",
        dtype="FLOAT32",
        shape=[1, 28, 28, 256],
        shape_signature=[1, 28, 28, 256],
    )
    operator = OperatorIR(
        op_type="ADD",
        inputs=["lhs", "rhs"],
        outputs=["source"],
    )

    _set_operator_outputs(
        model_ir=model_ir,
        op=operator,
        new_outputs=["target"],
    )

    assert operator.outputs == ["target"]
    assert model_ir.tensors["target"].shape_signature == [-1, -1, -1, 256]
