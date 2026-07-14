from __future__ import annotations

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.split_planner import (
    rewrite_model_ir_disable_group_convolution,
    rewrite_model_ir_unfold_batchmatmul,
    rewrite_model_ir_unroll_recurrent_ops,
)


@pytest.mark.parametrize(
    "rewrite",
    [
        rewrite_model_ir_disable_group_convolution,
        rewrite_model_ir_unfold_batchmatmul,
        rewrite_model_ir_unroll_recurrent_ops,
    ],
)
def test_split_rewrite_builder_emits_into_fresh_operator_stream(rewrite) -> None:
    model_ir = ModelIR("append_only_rewrite")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        name: TensorIR(name, "FLOAT32", [1, 3], [1, 3])
        for name in ("x", "y", "z")
    }
    original_op = OperatorIR(
        "ADD",
        ["x", "y"],
        ["z"],
        options={"fusedActivationFunction": "NONE"},
        axis_semantics={"axis": "feature"},
        version=2,
        onnx_node_name="add_node",
        onnx_op_type="Add",
    )
    model_ir.operators = [original_op]

    rewritten, rewritten_count = rewrite(model_ir=model_ir)

    assert rewritten_count == 0
    assert len(rewritten.operators) == 1
    assert rewritten.operators[0] is not original_op
    assert rewritten.operators[0] == original_op
    assert rewritten.tensors["x"] is not model_ir.tensors["x"]
    assert model_ir.operators == [original_op]


def test_append_only_builder_rewrites_group_convolution_once() -> None:
    model_ir = ModelIR("grouped_conv")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 4, 4, 4], [1, 4, 4, 4]),
        "weight": TensorIR(
            "weight",
            "FLOAT32",
            [4, 1, 1, 2],
            [4, 1, 1, 2],
            data=np.ones((4, 1, 1, 2), dtype=np.float32),
        ),
        "bias": TensorIR(
            "bias",
            "FLOAT32",
            [4],
            [4],
            data=np.zeros((4,), dtype=np.float32),
        ),
        "y": TensorIR("y", "FLOAT32", [1, 4, 4, 4], [1, 4, 4, 4]),
    }
    model_ir.operators = [
        OperatorIR(
            "CONV_2D",
            ["x", "weight", "bias"],
            ["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "RELU",
            },
        )
    ]

    rewritten, rewritten_count = rewrite_model_ir_disable_group_convolution(
        model_ir=model_ir,
    )

    assert rewritten_count == 1
    assert [op.op_type for op in rewritten.operators] == [
        "SPLIT",
        "CONV_2D",
        "CONV_2D",
        "CONCATENATION",
    ]
    assert len(model_ir.operators) == 1


def test_append_only_builder_unfolds_batchmatmul_once() -> None:
    model_ir = ModelIR("batched_matmul")
    model_ir.inputs = ["lhs", "rhs"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "lhs": TensorIR("lhs", "FLOAT32", [2, 3, 4], [2, 3, 4]),
        "rhs": TensorIR("rhs", "FLOAT32", [2, 4, 5], [2, 4, 5]),
        "y": TensorIR("y", "FLOAT32", [2, 3, 5], [2, 3, 5]),
    }
    model_ir.operators = [
        OperatorIR(
            "BATCH_MATMUL",
            ["lhs", "rhs"],
            ["y"],
            options={"adjX": False, "adjY": False},
        )
    ]

    rewritten, rewritten_count = rewrite_model_ir_unfold_batchmatmul(
        model_ir=model_ir,
    )

    assert rewritten_count == 1
    op_types = [op.op_type for op in rewritten.operators]
    assert op_types[:2] == ["RESHAPE", "RESHAPE"]
    assert op_types.count("BATCH_MATMUL") == 2
    assert rewritten.operators[-2].op_type == "CONCATENATION"
    assert rewritten.operators[-1].op_type == "RESHAPE"
    assert len(model_ir.operators) == 1
