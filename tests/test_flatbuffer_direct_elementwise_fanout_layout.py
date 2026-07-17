from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)
from onnx2tf.tflite_builder.passes.elementwise_fanout_layout import (
    optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains,
)


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
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
                repr(tensor.quantization),
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


def test_flatbuffer_direct_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chain_optimized() -> None:
    model_ir = ModelIR("transpose_elementwise_roundtrip_nhwc_nchw_fanout_chain_opt_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z0", "z1"]

    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
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
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["mul_scale"] = TensorIR(
        name="mul_scale",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.ones((1, 8, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["branch0_mul"] = TensorIR(
        name="branch0_mul",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["branch0_erf"] = TensorIR(
        name="branch0_erf",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["add_bias"] = TensorIR(
        name="add_bias",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.zeros((1, 8, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["branch0_out_nchw"] = TensorIR(
        name="branch0_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["branch1_sign"] = TensorIR(
        name="branch1_sign",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["branch1_scale"] = TensorIR(
        name="branch1_scale",
        dtype="FLOAT32",
        shape=[1, 8, 1, 1],
        shape_signature=[1, 8, 1, 1],
        data=np.full((1, 8, 1, 1), 0.5, dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["branch1_out_nchw"] = TensorIR(
        name="branch1_out_nchw",
        dtype="FLOAT32",
        shape=[1, 8, 6, 6],
        shape_signature=[1, 8, 6, 6],
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["z0"] = TensorIR(
        name="z0",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.tensors["z1"] = TensorIR(
        name="z1",
        dtype="FLOAT32",
        shape=[1, 6, 6, 8],
        shape_signature=[1, 6, 6, 8],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "mul_scale"], outputs=["branch0_mul"]),
        OperatorIR(op_type="ERF", inputs=["branch0_mul"], outputs=["branch0_erf"]),
        OperatorIR(op_type="ADD", inputs=["branch0_erf", "add_bias"], outputs=["branch0_out_nchw"]),
        OperatorIR(op_type="SIGN", inputs=["x_nchw"], outputs=["branch1_sign"]),
        OperatorIR(op_type="MUL", inputs=["branch1_sign", "branch1_scale"], outputs=["branch1_out_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["branch0_out_nchw", "post_perm"], outputs=["y0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["branch1_out_nchw", "post_perm"], outputs=["y1"]),
        OperatorIR(op_type="RELU", inputs=["y0"], outputs=["z0"]),
        OperatorIR(op_type="RELU", inputs=["y1"], outputs=["z1"]),
    ]

    owner_model_ir = deepcopy(model_ir)
    wrapper_model_ir = deepcopy(model_ir)
    owner_stats = optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
        owner_model_ir
    )
    wrapper_stats = (
        _optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains(
            wrapper_model_ir
        )
    )
    assert owner_stats == wrapper_stats == {
        "optimized_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains": 1
    }
    assert _fingerprint(owner_model_ir) == _fingerprint(wrapper_model_ir)
    model_ir = owner_model_ir

    assert not any(str(op.op_type) == "TRANSPOSE" for op in model_ir.operators)
    assert list(model_ir.tensors["mul_scale"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["add_bias"].shape) == [1, 1, 1, 8]
    assert list(model_ir.tensors["branch1_scale"].shape) == [1, 1, 1, 8]

    mul0_op = next(op for op in model_ir.operators if list(op.outputs) == ["branch0_mul"])
    sign_op = next(op for op in model_ir.operators if list(op.outputs) == ["branch1_sign"])
    assert [str(v) for v in list(mul0_op.inputs)][0] == "x_nhwc"
    assert [str(v) for v in list(sign_op.inputs)] == ["x_nhwc"]

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    mul1_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["y1"])
    assert list(add_op.outputs) == ["y0"]
    assert list(mul1_op.outputs) == ["y1"]
