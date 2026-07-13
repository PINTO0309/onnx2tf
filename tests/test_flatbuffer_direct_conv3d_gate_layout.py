from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _model(
    *,
    semantic_rank: int = 4,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("conv3d_leaky_unsqueeze_gate")
    semantic_shape = [1, 3, 4, 5] if semantic_rank == 4 else [1, 1, 3, 4, 5]
    semantic_perm = (
        [0, 3, 1, 2] if semantic_rank == 4 else [0, 4, 1, 2, 3]
    )
    reshape_values = [1, 5, 1, 3, 4]
    if boundary == "reshape_rank":
        reshape_values = [1, 5, 3, 4]
    conv_pre_perm = (
        [0, 1, 2, 3, 4]
        if boundary == "permutation"
        else [0, 4, 1, 2, 3]
    )
    model_ir.inputs = ["semantic", "conv_ndhwc"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "semantic": _tensor("semantic", semantic_shape),
        "conv_ndhwc": _tensor("conv_ndhwc", [1, 2, 3, 4, 5]),
        "semantic_perm": _tensor(
            "semantic_perm",
            [len(semantic_perm)],
            dtype="INT32",
            data=np.asarray(semantic_perm, dtype=np.int32),
        ),
        "conv_pre_perm": _tensor(
            "conv_pre_perm",
            [5],
            dtype="INT32",
            data=np.asarray(conv_pre_perm, dtype=np.int32),
        ),
        "post_perm": _tensor(
            "post_perm",
            [5],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 4, 1], dtype=np.int32),
        ),
        "unsqueeze_shape": _tensor(
            "unsqueeze_shape",
            [len(reshape_values)],
            dtype="INT32",
            data=np.asarray(reshape_values, dtype=np.int32),
        ),
        "weight": _tensor(
            "weight",
            [2, 1, 1, 1, 5],
            data=np.ones([2, 1, 1, 1, 5], dtype=np.float32),
        ),
        "bias": _tensor(
            "bias",
            [2],
            data=np.zeros([2], dtype=np.float32),
        ),
    }
    shapes = {
        "semantic_ncdhw": [1, 5, 3, 4] if semantic_rank == 4 else [1, 5, 1, 3, 4],
        "gate_ncdhw": [1, 5, 1, 3, 4],
        "conv_ncdhw": [1, 5, 2, 3, 4],
        "act_ncdhw": [1, 5, 2, 3, 4],
        "mul_ncdhw": [1, 5, 2, 3, 4],
        "mul_ndhwc": [1, 2, 3, 4, 5],
        "y": [1, 2, 3, 4, 2],
    }
    for name, shape in shapes.items():
        model_ir.tensors[name] = _tensor(name, shape)

    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            ["semantic", "semantic_perm"],
            ["semantic_ncdhw"],
        ),
        OperatorIR(
            "RESHAPE",
            ["semantic_ncdhw", "unsqueeze_shape"],
            ["gate_ncdhw"],
            options={"newShape": list(reshape_values)},
        ),
        OperatorIR(
            "TRANSPOSE",
            ["conv_ndhwc", "conv_pre_perm"],
            ["conv_ncdhw"],
        ),
        OperatorIR(
            "LEAKY_RELU",
            ["conv_ncdhw"],
            ["act_ncdhw"],
            options={"alpha": 0.1},
        ),
        OperatorIR("MUL", ["act_ncdhw", "gate_ncdhw"], ["mul_ncdhw"]),
        OperatorIR("TRANSPOSE", ["mul_ncdhw", "post_perm"], ["mul_ndhwc"]),
        OperatorIR("CONV_3D", ["mul_ndhwc", "weight", "bias"], ["y"]),
    ]
    side_sources = {
        "conv_adapter_fanout": "conv_ncdhw",
        "leaky_fanout": "act_ncdhw",
        "reshape_fanout": "gate_ncdhw",
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    if boundary == "public_intermediate":
        model_ir.outputs.append("mul_ncdhw")
    return model_ir


@pytest.mark.parametrize("semantic_rank", [4, 5])
def test_conv3d_gate_layout_characterization(semantic_rank: int) -> None:
    model_ir = _model(semantic_rank=semantic_rank)

    stats = _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains"
    ] == 1
    assert all(operator.op_type != "TRANSPOSE" for operator in model_ir.operators)
    reshape_op = next(
        operator for operator in model_ir.operators if operator.op_type == "RESHAPE"
    )
    assert reshape_op.inputs == ["semantic", "unsqueeze_shape"]
    np.testing.assert_array_equal(
        model_ir.tensors["unsqueeze_shape"].data,
        np.asarray([1, 1, 3, 4, 5], dtype=np.int32),
    )
    assert next(
        operator for operator in model_ir.operators if operator.op_type == "LEAKY_RELU"
    ).inputs == ["conv_ndhwc"]
    mul_op = next(
        operator for operator in model_ir.operators if operator.op_type == "MUL"
    )
    assert mul_op.outputs == ["mul_ndhwc"]
    assert next(
        operator for operator in model_ir.operators if operator.op_type == "CONV_3D"
    ).inputs[0] == "mul_ndhwc"


@pytest.mark.parametrize(
    "boundary",
    [
        "conv_adapter_fanout",
        "leaky_fanout",
        "reshape_fanout",
        "public_intermediate",
        "permutation",
        "reshape_rank",
    ],
)
def test_conv3d_gate_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original_operators = [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ]

    stats = _optimize_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_conv3d_leaky_mul_unsqueeze_ndhwc_chains"
    ] == 0
    assert [
        (operator.op_type, list(operator.inputs), list(operator.outputs))
        for operator in model_ir.operators
    ] == original_operators
    assert [operator.op_type for operator in model_ir.operators].count(
        "TRANSPOSE"
    ) == 3
