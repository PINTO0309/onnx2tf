from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.high_rank_binary import (
    coalesce_static_high_rank_binary_operators,
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
        shape=list(shape),
        shape_signature=list(signature) if signature is not None else list(shape),
    )


def _rank6_div_ir(*, dynamic_signature: bool) -> ModelIR:
    lhs_shape = [2, 11097, 8, 4, 4, 2]
    rhs_shape = [1, 1, 1, 4, 1, 2]
    signature = [2, -1, 8, 4, 4, 2] if dynamic_signature else lhs_shape
    model_ir = ModelIR("rank6_div")
    model_ir.inputs = ["lhs", "rhs"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "lhs": _tensor("lhs", lhs_shape, signature=signature),
        "rhs": _tensor("rhs", rhs_shape),
        "output": _tensor("output", lhs_shape, signature=signature),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="DIV",
            inputs=["lhs", "rhs"],
            outputs=["output"],
            options={"fusedActivationFunction": "NONE"},
            onnx_node_name="Div_0",
            onnx_op_type="Div",
        )
    ]
    return model_ir


def test_coalesces_static_rank6_binary_broadcast_to_rank4() -> None:
    model_ir = _rank6_div_ir(dynamic_signature=False)

    stats = coalesce_static_high_rank_binary_operators(model_ir)

    assert stats == {"coalesced_static_high_rank_binary_operators": 1}
    assert [op.op_type for op in model_ir.operators] == [
        "RESHAPE",
        "RESHAPE",
        "DIV",
        "RESHAPE",
    ]
    div_op = model_ir.operators[2]
    assert model_ir.tensors[div_op.inputs[0]].shape == [177552, 4, 4, 2]
    assert model_ir.tensors[div_op.inputs[1]].shape == [1, 4, 1, 2]
    assert model_ir.tensors[div_op.outputs[0]].shape == [177552, 4, 4, 2]
    assert model_ir.operators[-1].outputs == ["output"]
    assert np.prod(model_ir.tensors[div_op.outputs[0]].shape) == np.prod(
        model_ir.tensors["output"].shape
    )


def test_keeps_dynamic_rank6_binary_uncoalesced() -> None:
    model_ir = _rank6_div_ir(dynamic_signature=True)

    stats = coalesce_static_high_rank_binary_operators(model_ir)

    assert stats == {"coalesced_static_high_rank_binary_operators": 0}
    assert [op.op_type for op in model_ir.operators] == ["DIV"]
