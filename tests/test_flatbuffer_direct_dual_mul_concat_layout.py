from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_dual_mul_concat_prepost_nhwc_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        data=None if data is None else np.asarray(data),
        is_variable=False,
    )


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("dual_mul_concat_nhwc")
    model_ir.inputs = ["x_nhwc", "aux_nchw"]
    model_ir.outputs = ["z", "aux_out"]
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 4, 5, 3]),
        "aux_nchw": _tensor("aux_nchw", [1, 3, 4, 5]),
        "x_nchw": _tensor("x_nchw", [1, 3, 4, 5]),
        "m0": _tensor("m0", [1, 3, 4, 5]),
        "m1": _tensor("m1", [1, 3, 4, 5]),
        "cat_nchw": _tensor("cat_nchw", [1, 6, 4, 5]),
        "y_nhwc": _tensor("y_nhwc", [1, 4, 5, 6]),
        "z": _tensor("z", [1, 4, 5, 6]),
        "aux_out": _tensor("aux_out", [1, 3, 4, 5]),
        "shared_const": _tensor(
            "shared_const",
            [1, 3, 1, 1],
            data=np.asarray([[[[1.0]], [[2.0]], [[3.0]]]], dtype=np.float32),
        ),
        "const1": _tensor(
            "const1",
            [1, 3, 1, 1],
            data=(
                None
                if boundary == "missing_constant"
                else np.asarray(
                    [[[[0.5]], [[0.25]], [[0.125]]]],
                    dtype=np.float32,
                )
            ),
        ),
        "pre_perm": TensorIR(
            name="pre_perm",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(
                [0, 1, 2, 3]
                if boundary == "pre_permutation"
                else [0, 3, 1, 2],
                dtype=np.int32,
            ),
            is_variable=False,
        ),
        "post_perm": TensorIR(
            name="post_perm",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(
                [0, 1, 2, 3]
                if boundary == "post_permutation"
                else [0, 2, 3, 1],
                dtype=np.int32,
            ),
            is_variable=False,
        ),
    }
    if boundary == "different_data_branch":
        model_ir.inputs.append("x1_nchw")
        model_ir.tensors["x1_nchw"] = _tensor("x1_nchw", [1, 3, 4, 5])
    second_data = "x1_nchw" if boundary == "different_data_branch" else "x_nchw"
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "pre_perm"], ["x_nchw"]),
        OperatorIR("MUL", ["x_nchw", "shared_const"], ["m0"]),
        OperatorIR("MUL", [second_data, "const1"], ["m1"]),
        OperatorIR(
            "CONCATENATION",
            ["m0", "m1"],
            ["cat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
        OperatorIR("TRANSPOSE", ["cat_nchw", "post_perm"], ["y_nhwc"]),
        OperatorIR("RELU", ["y_nhwc"], ["z"]),
        OperatorIR("MUL", ["aux_nchw", "shared_const"], ["aux_out"]),
    ]
    side_sources = {
        "pre_adapter_fanout": "x_nchw",
        "mul_output_fanout": "m0",
        "concat_fanout": "cat_nchw",
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
        model_ir.outputs.append("cat_nchw")
    if boundary == "public_post_output":
        model_ir.outputs.append("y_nhwc")
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in actual.operators
    ] == [
        (op.op_type, op.inputs, op.outputs, op.options)
        for op in expected.operators
    ]
    assert actual.tensors.keys() == expected.tensors.keys()
    for name, tensor in actual.tensors.items():
        expected_tensor = expected.tensors[name]
        assert tensor.dtype == expected_tensor.dtype
        assert tensor.shape == expected_tensor.shape
        assert tensor.shape_signature == expected_tensor.shape_signature
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def test_dual_mul_concat_layout_characterization() -> None:
    model_ir = _model()

    stats = _optimize_transpose_dual_mul_concat_prepost_nhwc_chains(model_ir)

    assert stats["optimized_transpose_dual_mul_concat_prepost_nhwc_chains"] == 1
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    mul0 = next(op for op in model_ir.operators if op.outputs == ["m0"])
    mul1 = next(op for op in model_ir.operators if op.outputs == ["m1"])
    assert mul0.inputs[0] == "x_nhwc"
    assert mul1.inputs[0] == "x_nhwc"
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.options["axis"] == 3
    assert concat_op.outputs == ["y_nhwc"]
    aux_mul = next(op for op in model_ir.operators if op.outputs == ["aux_out"])
    assert aux_mul.inputs == ["aux_nchw", "shared_const"]
    assert model_ir.tensors["shared_const"].shape == [1, 3, 1, 1]
    mul0_side = mul0.inputs[1]
    assert mul0_side != "shared_const"
    assert model_ir.tensors[mul0_side].shape == [1, 1, 1, 3]
    assert model_ir.tensors["const1"].shape == [1, 1, 1, 3]


@pytest.mark.parametrize(
    "boundary",
    [
        "pre_adapter_fanout",
        "mul_output_fanout",
        "concat_fanout",
        "public_intermediate",
        "public_post_output",
        "pre_permutation",
        "post_permutation",
        "concat_axis",
        "missing_constant",
        "different_data_branch",
    ],
)
def test_dual_mul_concat_layout_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_dual_mul_concat_prepost_nhwc_chains(model_ir)

    assert stats["optimized_transpose_dual_mul_concat_prepost_nhwc_chains"] == 0
    _assert_model_equal(model_ir, original)
