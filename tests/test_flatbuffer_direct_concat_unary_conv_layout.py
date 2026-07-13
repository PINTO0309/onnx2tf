from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_concat_unary_fanout_conv_nhwc_chains,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32" if data is None else str(data.dtype).upper(),
        shape=list(shape),
        shape_signature=list(shape),
        data=None if data is None else np.asarray(data),
        is_variable=False,
    )


def _model(
    *,
    unary_ops: tuple[str, ...] = ("RELU",),
    post_count: int = 1,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("concat_unary_conv_nhwc")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = [f"y{index}" for index in range(post_count)]
    pre_perm = [0, 1, 2, 3] if boundary == "pre_permutation" else [0, 3, 1, 2]
    post_perm = [0, 1, 2, 3] if boundary == "post_permutation" else [0, 2, 3, 1]
    model_ir.tensors = {
        "x0_nhwc": _tensor("x0_nhwc", [1, 2, 3, 2]),
        "x1_nhwc": _tensor("x1_nhwc", [1, 2, 3, 2]),
        "x0_nchw": _tensor("x0_nchw", [1, 2, 2, 3]),
        "x1_nchw": _tensor("x1_nchw", [1, 2, 2, 3]),
        "cat_nchw": _tensor("cat_nchw", [1, 4, 2, 3]),
        "pre_perm": TensorIR(
            name="pre_perm",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(pre_perm, dtype=np.int32),
            is_variable=False,
        ),
        "post_perm": TensorIR(
            name="post_perm",
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray(post_perm, dtype=np.int32),
            is_variable=False,
        ),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre_perm"], ["x1_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["x0_nchw", "x1_nchw"],
            ["cat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
    ]
    tail_name = "cat_nchw"
    for index, op_type in enumerate(unary_ops):
        output_name = f"unary{index}_nchw"
        model_ir.tensors[output_name] = _tensor(output_name, [1, 4, 2, 3])
        model_ir.operators.append(
            OperatorIR(
                "ABS" if boundary == "unsupported_unary" and index == 0 else op_type,
                [tail_name],
                [output_name],
            )
        )
        tail_name = output_name

    for index in range(post_count):
        post_name = f"post{index}_nhwc"
        weight_name = f"weight{index}"
        bias_name = f"bias{index}"
        output_name = f"y{index}"
        model_ir.tensors[post_name] = _tensor(post_name, [1, 2, 3, 4])
        model_ir.tensors[weight_name] = _tensor(
            weight_name,
            [3, 1, 1, 4],
            data=np.ones([3, 1, 1, 4], dtype=np.float32),
        )
        model_ir.tensors[bias_name] = _tensor(
            bias_name,
            [3],
            data=np.zeros([3], dtype=np.float32),
        )
        model_ir.tensors[output_name] = _tensor(output_name, [1, 2, 3, 3])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [tail_name, "post_perm"],
                    [post_name],
                ),
                OperatorIR(
                    (
                        "ABS"
                        if boundary == "post_nonconv" and index == 0
                        else "CONV_2D"
                        if index == 0
                        else "DEPTHWISE_CONV_2D"
                    ),
                    [post_name, weight_name, bias_name],
                    [output_name],
                ),
            ]
        )

    side_sources = {
        "pre_adapter_fanout": "x0_nchw",
        "concat_fanout": "cat_nchw",
        "unary_fanout": tail_name,
    }
    if boundary in side_sources:
        source = side_sources[boundary]
        model_ir.tensors["side"] = _tensor(
            "side",
            list(model_ir.tensors[source].shape),
        )
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", [source], ["side"]))
    public_sources = {
        "public_pre": "x0_nchw",
        "public_concat": "cat_nchw",
        "public_unary": tail_name,
        "public_post": "post0_nhwc",
    }
    if boundary in public_sources:
        model_ir.outputs.append(public_sources[boundary])
    if boundary == "non_transpose_input":
        model_ir.operators[0].op_type = "IDENTITY"
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


def _assert_nhwc_rewrite(model_ir: ModelIR, tail_name: str) -> None:
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    concat = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["x0_nhwc", "x1_nhwc"]
    assert concat.options["axis"] == 3
    assert model_ir.tensors["cat_nchw"].shape == [1, 2, 3, 4]
    if tail_name != "cat_nchw":
        assert model_ir.tensors[tail_name].shape == [1, 2, 3, 4]
    conv_ops = [
        op
        for op in model_ir.operators
        if op.op_type in {"CONV_2D", "DEPTHWISE_CONV_2D"}
    ]
    assert conv_ops
    assert all(op.inputs[0] == tail_name for op in conv_ops)


def test_concat_unary_conv_layout_without_unary_characterization() -> None:
    model_ir = _model(unary_ops=())

    stats = _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(model_ir)

    assert stats["optimized_transpose_concat_unary_fanout_conv_nhwc_chains"] == 1
    _assert_nhwc_rewrite(model_ir, "cat_nchw")


def test_concat_unary_conv_layout_with_post_fanout_characterization() -> None:
    model_ir = _model(unary_ops=("RELU", "TANH"), post_count=2)

    stats = _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(model_ir)

    assert stats["optimized_transpose_concat_unary_fanout_conv_nhwc_chains"] == 1
    _assert_nhwc_rewrite(model_ir, "unary1_nchw")


@pytest.mark.parametrize(
    "boundary",
    [
        "pre_adapter_fanout",
        "concat_fanout",
        "unary_fanout",
        "public_pre",
        "public_concat",
        "public_unary",
        "public_post",
        "pre_permutation",
        "post_permutation",
        "concat_axis",
        "non_transpose_input",
        "unsupported_unary",
        "post_nonconv",
    ],
)
def test_concat_unary_conv_layout_rejects_unsafe_boundary(
    boundary: str,
) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_concat_unary_fanout_conv_nhwc_chains(model_ir)

    assert stats["optimized_transpose_concat_unary_fanout_conv_nhwc_chains"] == 0
    _assert_model_equal(model_ir, original)
