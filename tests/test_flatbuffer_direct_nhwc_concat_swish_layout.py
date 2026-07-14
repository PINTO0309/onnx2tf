from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_concat_nhwc_chains,
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
    )


def _swish_model(
    *,
    logistic_first: bool = False,
    all_swish: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_swish_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y"]
    source_shape = (
        [1, 5, 3]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 3]
    )
    swish_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_swish_output_rank":
        swish_shape = [1, 3, 5]
    logistic_shape = (
        [1, 3, 5]
        if boundary == "invalid_logistic_output_rank"
        else [1, 3, 5, 7]
    )
    model_ir.tensors = {
        "x0_nhwc": _tensor("x0_nhwc", [1, 5, 7, 2]),
        "x1_nhwc": _tensor("x1_nhwc", source_shape),
        "pre_perm": TensorIR(
            "pre_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "post_perm": TensorIR(
            "post_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "x0_nchw": _tensor("x0_nchw", [1, 2, 5, 7]),
        "x0_relu": _tensor("x0_relu", [1, 2, 5, 7]),
        "x1_nchw": _tensor("x1_nchw", [1, 3, 5, 7]),
        "x1_logistic": _tensor("x1_logistic", logistic_shape),
        "x1_swish": _tensor("x1_swish", swish_shape),
        "concat_nchw": _tensor("concat_nchw", [1, 5, 5, 7]),
        "concat_nhwc": _tensor("concat_nhwc", [1, 5, 7, 5]),
        "y": _tensor("y", [1, 5, 7, 5]),
    }
    model_ir.tensors["x1_logistic"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=1,
    )
    model_ir.tensors["x1_swish"].quantization = {
        "scale": [0.5] * 3,
        "zero_point": [0] * 3,
        "quantized_dimension": 1,
    }
    mul_inputs = (
        ["x1_logistic", "x1_nchw"]
        if logistic_first
        else ["x1_nchw", "x1_logistic"]
    )
    if boundary == "wrong_mul_data_input":
        model_ir.tensors["wrong_data"] = _tensor(
            "wrong_data",
            [1, 3, 5, 7],
        )
        mul_inputs = ["wrong_data", "x1_logistic"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre_perm"], ["x0_nchw"]),
        OperatorIR("RELU", ["x0_nchw"], ["x0_relu"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre_perm"], ["x1_nchw"]),
        OperatorIR(
            "TANH" if boundary == "unsupported_logistic" else "LOGISTIC",
            ["x1_nchw"],
            ["x1_logistic"],
        ),
        OperatorIR(
            "ADD" if boundary == "unsupported_mul" else "MUL",
            list(mul_inputs),
            ["x1_swish"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "CONCATENATION",
            ["x0_relu", "x1_swish"],
            ["concat_nchw"],
            options={"axis": 1},
        ),
        OperatorIR(
            "TRANSPOSE",
            ["concat_nchw", "post_perm"],
            ["concat_nhwc"],
        ),
        OperatorIR("RELU", ["concat_nhwc"], ["y"]),
    ]

    if all_swish:
        model_ir.tensors["x0_logistic"] = _tensor(
            "x0_logistic",
            [1, 2, 5, 7],
        )
        model_ir.tensors["x0_swish"] = _tensor(
            "x0_swish",
            [1, 2, 5, 7],
        )
        relu_index = next(
            index
            for index, op in enumerate(model_ir.operators)
            if op.outputs == ["x0_relu"]
        )
        model_ir.operators[relu_index : relu_index + 1] = [
            OperatorIR("LOGISTIC", ["x0_nchw"], ["x0_logistic"]),
            OperatorIR(
                "MUL",
                ["x0_nchw", "x0_logistic"],
                ["x0_swish"],
            ),
        ]
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["x0_swish", "x1_swish"]

    if boundary == "swish_output_fanout":
        model_ir.tensors["swish_side"] = _tensor(
            "swish_side",
            list(swish_shape),
        )
        model_ir.outputs.append("swish_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_swish"], ["swish_side"])
        )
    if boundary == "public_swish_output":
        model_ir.outputs.append("x1_swish")
    if boundary == "logistic_output_fanout":
        model_ir.tensors["logistic_side"] = _tensor(
            "logistic_side",
            list(logistic_shape),
        )
        model_ir.outputs.append("logistic_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_logistic"], ["logistic_side"])
        )
    if boundary == "public_logistic_output":
        model_ir.outputs.append("x1_logistic")
    if boundary == "swish_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 3, 5, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_swish_adapter":
        model_ir.outputs.append("x1_nchw")
    if boundary == "raw_residual_input":
        model_ir.inputs.append("residual_nchw")
        model_ir.tensors["residual_nchw"] = _tensor(
            "residual_nchw",
            [1, 2, 5, 7],
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["residual_nchw", "x1_swish"]
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert actual.metadata == expected.metadata
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
        assert tensor.quantization == expected_tensor.quantization
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


@pytest.mark.parametrize("logistic_first", [False, True])
def test_nhwc_swish_with_unary_input_is_indexed(logistic_first: bool) -> None:
    model_ir = _swish_model(logistic_first=logistic_first)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    logistic_op = next(
        op for op in model_ir.operators if op.op_type == "LOGISTIC"
    )
    mul_op = next(op for op in model_ir.operators if op.op_type == "MUL")
    assert logistic_op.inputs == ["x1_nhwc"]
    assert set(mul_op.inputs) == {"x1_nhwc", "x1_logistic"}
    assert model_ir.tensors["x1_logistic"].shape == [1, 5, 7, 3]
    assert model_ir.tensors["x1_swish"].shape == [1, 5, 7, 3]
    logistic_quantization = model_ir.tensors["x1_logistic"].quantization
    assert isinstance(logistic_quantization, QuantParamIR)
    assert logistic_quantization.quantized_dimension == 3
    swish_quantization = model_ir.tensors["x1_swish"].quantization
    assert isinstance(swish_quantization, dict)
    assert swish_quantization["quantized_dimension"] == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_relu", "x1_swish"]
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_swish"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_all_swish_inputs_use_indexed_family() -> None:
    model_ir = _swish_model(all_swish=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    logistic_ops = [
        op for op in model_ir.operators if op.op_type == "LOGISTIC"
    ]
    assert [op.inputs for op in logistic_ops] == [["x0_nhwc"], ["x1_nhwc"]]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_swish", "x1_swish"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_logistic",
        "unsupported_mul",
        "wrong_mul_data_input",
        "swish_output_fanout",
        "public_swish_output",
        "logistic_output_fanout",
        "public_logistic_output",
        "swish_adapter_fanout",
        "public_swish_adapter",
        "invalid_source_rank",
        "invalid_logistic_output_rank",
        "invalid_swish_output_rank",
        "spatial_shape_mismatch",
        "raw_residual_input",
    ],
)
def test_nhwc_swish_rejects_unsafe_or_partial_match(boundary: str) -> None:
    model_ir = _swish_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)
