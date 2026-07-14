from __future__ import annotations

from onnx2tf.tflite_builder.ir import ModelIR, TensorIR
from onnx2tf.tflite_builder.pytorch_binary_policy import (
    _binary_requires_runtime_alignment_for_codegen,
    _binary_runtime_shape_passthrough_operand_for_codegen,
    _preferred_binary_alignment_anchor_for_codegen,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=shape,
        shape_signature=signature,
    )


def test_binary_runtime_shape_passthrough_uses_all_ones_peer() -> None:
    model_ir = ModelIR(
        name="binary_passthrough",
        tensors={
            "lhs": _tensor("lhs", [2, 3]),
            "rhs": _tensor("rhs", [1, 1]),
        },
    )

    assert _binary_runtime_shape_passthrough_operand_for_codegen(
        model_ir=model_ir,
        runtime_shape_uncertain_tensors={"lhs"},
        lhs_name="lhs",
        rhs_name="rhs",
    ) == "lhs"


def test_binary_alignment_uses_dynamic_signature_anchor() -> None:
    model_ir = ModelIR(
        name="binary_alignment",
        tensors={
            "lhs": _tensor("lhs", [2, 3], signature=[-1, 3]),
            "rhs": _tensor("rhs", [2, 3], signature=[2, 3]),
            "output": _tensor("output", [2, 3], signature=[-1, 3]),
        },
    )

    assert _binary_requires_runtime_alignment_for_codegen(
        model_ir=model_ir,
        runtime_shape_uncertain_tensors=set(),
        lhs_name="lhs",
        rhs_name="rhs",
        output_name="output",
    )
    assert _preferred_binary_alignment_anchor_for_codegen(
        model_ir=model_ir,
        lhs_name="lhs",
        rhs_name="rhs",
        output_name="output",
    ) == "lhs"
