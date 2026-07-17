from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_mean_mul_add_const_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.mean_affine_prepost_layout import (
    optimize_transpose_mean_mul_add_const_prepost_nhwc_chains,
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


def test_transpose_mean_affine_rewrite_maps_reduction_axes_to_nhwc() -> None:
    model_ir = ModelIR("transpose_mean_affine_axes")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    tensor_specs = {
        "x_nhwc": [1, 8, 8, 4],
        "x_nchw": [1, 4, 8, 8],
        "mean_nchw": [1, 4, 1, 1],
        "mul_nchw": [1, 4, 1, 1],
        "add_nchw": [1, 4, 1, 1],
        "y_nhwc": [1, 1, 1, 4],
        "z": [1, 1, 1, 4],
    }
    for name, shape in tensor_specs.items():
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=list(shape),
            shape_signature=list(shape),
        )
    for name, values in (
        ("pre_perm", [0, 3, 1, 2]),
        ("post_perm", [0, 2, 3, 1]),
        ("axes", [2, 3]),
    ):
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="INT32",
            shape=[len(values)],
            shape_signature=[len(values)],
            data=np.asarray(values, dtype=np.int32),
        )
    for name, value in (("scale", 2.0), ("bias", 1.0)):
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
            data=np.asarray([value], dtype=np.float32),
        )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "pre_perm"], ["x_nchw"]),
        OperatorIR("MEAN", ["x_nchw", "axes"], ["mean_nchw"], {"keepDims": True}),
        OperatorIR("MUL", ["mean_nchw", "scale"], ["mul_nchw"]),
        OperatorIR("ADD", ["mul_nchw", "bias"], ["add_nchw"]),
        OperatorIR("TRANSPOSE", ["add_nchw", "post_perm"], ["y_nhwc"]),
        OperatorIR("RELU", ["y_nhwc"], ["z"]),
    ]

    owner_ir = deepcopy(model_ir)
    owner_stats = optimize_transpose_mean_mul_add_const_prepost_nhwc_chains(
        owner_ir
    )
    stats = _optimize_transpose_mean_mul_add_const_prepost_nhwc_chains(
        model_ir
    )

    assert owner_stats == stats
    assert _fingerprint(owner_ir) == _fingerprint(model_ir)
    assert stats["optimized_transpose_mean_mul_add_const_prepost_nhwc_chains"] == 1
    mean_op = next(op for op in model_ir.operators if str(op.op_type) == "MEAN")
    assert list(mean_op.inputs) == ["x_nhwc", "axes"]
    assert np.asarray(model_ir.tensors["axes"].data).tolist() == [1, 2]
