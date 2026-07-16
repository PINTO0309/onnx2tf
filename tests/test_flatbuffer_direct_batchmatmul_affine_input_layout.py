from __future__ import annotations

from copy import deepcopy

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_batchmatmul_affine_transpose_input_chains,
)
from onnx2tf.tflite_builder.passes.batchmatmul_affine_input_layout import (
    optimize_batchmatmul_affine_transpose_input_chains,
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


def test_flatbuffer_direct_batchmatmul_affine_transpose_input_chains() -> None:
    model_ir = ModelIR("batchmatmul_affine_transpose_inputs_test")
    model_ir.inputs = ["lhs_nhwc", "rhs_nhwc"]
    model_ir.outputs = ["bmm_out"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("lhs_nhwc", [1, 8, 8, 96])
    _add_tensor("lhs_nchw", [1, 96, 8, 8])
    _add_tensor("lhs_mul_out", [1, 96, 8, 8])
    _add_tensor("lhs_add_out", [1, 96, 8, 8])
    _add_tensor("lhs_reshape", [1, 96, 64])
    _add_tensor("lhs_mat", [1, 64, 96])

    _add_tensor("rhs_nhwc", [1, 16, 16, 96])
    _add_tensor("rhs_nchw", [1, 96, 16, 16])
    _add_tensor("rhs_mul_out", [1, 96, 16, 16])
    _add_tensor("rhs_add_out", [1, 96, 16, 16])
    _add_tensor("rhs_reshape", [1, 96, 256])

    _add_tensor("bmm_out", [1, 64, 256])

    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_swap_last2_rank3", [3], "INT32", np.asarray([0, 2, 1], dtype=np.int32))
    _add_tensor("lhs_mul_const", [1, 96, 1, 1], data=np.ones((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("lhs_add_const", [1, 96, 1, 1], data=np.zeros((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("rhs_mul_const", [1, 96, 1, 1], data=np.ones((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("rhs_add_const", [1, 96, 1, 1], data=np.zeros((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("lhs_shape", [3], "INT32", np.asarray([1, 96, 64], dtype=np.int32))
    _add_tensor("rhs_shape", [3], "INT32", np.asarray([1, 96, 256], dtype=np.int32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["lhs_nhwc", "perm_nhwc_to_nchw"], outputs=["lhs_nchw"]),
        OperatorIR(op_type="MUL", inputs=["lhs_nchw", "lhs_mul_const"], outputs=["lhs_mul_out"]),
        OperatorIR(op_type="ADD", inputs=["lhs_mul_out", "lhs_add_const"], outputs=["lhs_add_out"]),
        OperatorIR(op_type="RESHAPE", inputs=["lhs_add_out", "lhs_shape"], outputs=["lhs_reshape"], options={"newShape": [1, 96, 64]}),
        OperatorIR(op_type="TRANSPOSE", inputs=["lhs_reshape", "perm_swap_last2_rank3"], outputs=["lhs_mat"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["rhs_nhwc", "perm_nhwc_to_nchw"], outputs=["rhs_nchw"]),
        OperatorIR(op_type="MUL", inputs=["rhs_nchw", "rhs_mul_const"], outputs=["rhs_mul_out"]),
        OperatorIR(op_type="ADD", inputs=["rhs_mul_out", "rhs_add_const"], outputs=["rhs_add_out"]),
        OperatorIR(op_type="RESHAPE", inputs=["rhs_add_out", "rhs_shape"], outputs=["rhs_reshape"], options={"newShape": [1, 96, 256]}),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_mat", "rhs_reshape"],
            outputs=["bmm_out"],
            options={"adjX": False, "adjY": False},
        ),
    ]

    owner_ir = deepcopy(model_ir)
    owner_stats = optimize_batchmatmul_affine_transpose_input_chains(owner_ir)
    stats = _optimize_batchmatmul_affine_transpose_input_chains(model_ir)
    assert owner_stats == stats
    assert _fingerprint(owner_ir) == _fingerprint(model_ir)
    assert stats["optimized_batchmatmul_affine_transpose_input_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    bmm_op = next(op for op in model_ir.operators if str(op.op_type) == "BATCH_MATMUL")
    assert list(bmm_op.inputs) == ["lhs_reshape", "rhs_reshape"]
    assert bool(dict(bmm_op.options).get("adjY", False))

    lhs_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["lhs_mul_out"])
    rhs_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["rhs_mul_out"])
    assert list(lhs_mul_op.inputs)[0] == "lhs_nhwc"
    assert list(rhs_mul_op.inputs)[0] == "rhs_nhwc"

    lhs_shape_vals = np.asarray(model_ir.tensors["lhs_shape"].data, dtype=np.int32).reshape(-1).tolist()
    rhs_shape_vals = np.asarray(model_ir.tensors["rhs_shape"].data, dtype=np.int32).reshape(-1).tolist()
    assert lhs_shape_vals == [1, 64, 96]
    assert rhs_shape_vals == [1, 256, 96]
