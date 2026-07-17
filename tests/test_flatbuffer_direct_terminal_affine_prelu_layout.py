from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains,
)
from onnx2tf.tflite_builder.passes.terminal_affine_prelu_layout import (
    optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains,
)


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (operator.op_type, tuple(operator.inputs), tuple(operator.outputs), repr(operator.options))
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
                tensor.logical_layout,
                tensor.physical_layout,
                (
                    None
                    if tensor.data is None
                    else (
                        str(np.asarray(tensor.data).dtype),
                        tuple(np.asarray(tensor.data).shape),
                        tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                    )
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


def test_flatbuffer_direct_transpose_mul_add_const_prelu_prepost_terminal_optimized() -> None:
    model_ir = ModelIR("transpose_mul_add_const_prelu_prepost_terminal_opt_test")
    model_ir.inputs = ["x_nhwc", "legacy_nchw"]
    model_ir.outputs = ["z", "legacy_cat"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["legacy_nchw"] = TensorIR(
        name="legacy_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.ones((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["add_const"] = TensorIR(
        name="add_const",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.zeros((1, 16, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["prelu_alpha"] = TensorIR(
        name="prelu_alpha",
        dtype="FLOAT32",
        shape=[1, 16, 1, 1],
        shape_signature=[1, 16, 1, 1],
        data=np.full((1, 16, 1, 1), 0.25, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["prelu_out"] = TensorIR(
        name="prelu_out",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["legacy_cat"] = TensorIR(
        name="legacy_cat",
        dtype="FLOAT32",
        shape=[1, 32, 8, 8],
        shape_signature=[1, 32, 8, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_const"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="PRELU", inputs=["add_out", "prelu_alpha"], outputs=["prelu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["prelu_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["legacy_nchw", "prelu_out"],
            outputs=["legacy_cat"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    owner_ir = deepcopy(model_ir)
    owner_stats = optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains(
        owner_ir
    )
    stats = _optimize_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains(
        model_ir
    )
    assert owner_stats == stats
    assert _fingerprint(owner_ir) == _fingerprint(model_ir)
    assert stats["optimized_transpose_mul_add_const_prelu_prepost_nhwc_terminal_chains"] == 1
    assert list(model_ir.tensors["mul_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["add_const"].shape) == [1, 1, 1, 16]
    assert list(model_ir.tensors["prelu_alpha"].shape) == [1, 1, 1, 16]
    assert any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["prelu_out"]
        for op in model_ir.operators
    )
