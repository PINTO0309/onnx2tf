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


def _int_tensor(name: str, values: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32",
        shape=[len(values)],
        shape_signature=[len(values)],
        data=np.asarray(values, dtype=np.int32),
    )


def _add_model(
    *,
    all_add: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_add_pre_concat")
    model_ir.inputs = ["x_nhwc", "a_nhwc", "b_nhwc"]
    model_ir.outputs = ["y"]
    a_source_shape = (
        [1, 5, 3]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 3]
    )
    a_adapter_shape = (
        [1, 3, 5]
        if boundary == "invalid_adapter_rank"
        else [1, 3, 5, 7]
    )
    add_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_add_output_rank":
        add_shape = [1, 3, 5]
    output_channels = 3 if all_add else 5
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 5, 7, 2]),
        "a_nhwc": _tensor("a_nhwc", a_source_shape),
        "b_nhwc": _tensor("b_nhwc", [1, 5, 7, 3]),
        "pre_perm": _int_tensor("pre_perm", [0, 3, 1, 2]),
        "post_perm": _int_tensor(
            "post_perm",
            [0, 3, 1, 2]
            if boundary == "wrong_post_permutation"
            else [0, 2, 3, 1],
        ),
        "x_nchw": _tensor("x_nchw", [1, 2, 5, 7]),
        "a_nchw": _tensor("a_nchw", a_adapter_shape),
        "b_nchw": _tensor("b_nchw", [1, 3, 5, 7]),
        "sum_nchw": _tensor("sum_nchw", add_shape),
        "concat_nchw": _tensor(
            "concat_nchw",
            [1, output_channels, 5, 7],
        ),
        "concat_nhwc": _tensor(
            "concat_nhwc",
            [1, 5, 7, output_channels],
        ),
        "y": _tensor("y", [1, 5, 7, output_channels]),
    }
    if boundary == "wrong_add_pre_permutation":
        model_ir.tensors["bad_pre_perm"] = _int_tensor(
            "bad_pre_perm",
            [0, 2, 3, 1],
        )
    model_ir.tensors["sum_nchw"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=1,
    )
    add_inputs = ["a_nchw", "b_nchw"]
    concat_inputs = ["sum_nchw"] if all_add else ["x_nchw", "sum_nchw"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "pre_perm"], ["x_nchw"]),
        OperatorIR(
            "TRANSPOSE",
            [
                "a_nhwc",
                "bad_pre_perm" if boundary == "wrong_add_pre_permutation" else "pre_perm",
            ],
            ["a_nchw"],
        ),
        OperatorIR("TRANSPOSE", ["b_nhwc", "pre_perm"], ["b_nchw"]),
        OperatorIR(
            "SUB" if boundary == "unsupported_add" else "ADD",
            add_inputs,
            ["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            "CONCATENATION",
            concat_inputs,
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
    if all_add:
        model_ir.inputs = ["a_nhwc", "b_nhwc"]
        model_ir.operators = [
            op for op in model_ir.operators if op.outputs != ["x_nchw"]
        ]
        del model_ir.tensors["x_nhwc"]
        del model_ir.tensors["x_nchw"]

    if boundary == "add_output_fanout":
        model_ir.tensors["sum_side"] = _tensor(
            "sum_side",
            list(add_shape),
        )
        model_ir.outputs.append("sum_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["sum_nchw"], ["sum_side"])
        )
    if boundary == "add_output_post_transpose":
        model_ir.tensors["sum_nhwc"] = _tensor(
            "sum_nhwc",
            [1, 5, 7, 3],
        )
        model_ir.tensors["sum_side"] = _tensor(
            "sum_side",
            [1, 5, 7, 3],
        )
        model_ir.outputs.append("sum_side")
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["sum_nchw", "post_perm"],
                    ["sum_nhwc"],
                ),
                OperatorIR("IDENTITY", ["sum_nhwc"], ["sum_side"]),
            ]
        )
    if boundary == "public_add_post":
        model_ir.tensors["sum_nhwc"] = _tensor(
            "sum_nhwc",
            [1, 5, 7, 3],
        )
        model_ir.outputs.append("sum_nhwc")
        model_ir.operators.append(
            OperatorIR(
                "TRANSPOSE",
                ["sum_nchw", "post_perm"],
                ["sum_nhwc"],
            )
        )
    if boundary == "public_add_output":
        model_ir.outputs.append("sum_nchw")
    if boundary in {"add_adapter_fanout", "add_second_adapter_fanout"}:
        adapter_name = (
            "b_nchw"
            if boundary == "add_second_adapter_fanout"
            else "a_nchw"
        )
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            list(model_ir.tensors[adapter_name].shape),
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [adapter_name], ["adapter_side"])
        )
    if boundary == "public_add_adapter":
        model_ir.outputs.append("a_nchw")
    if boundary == "adapter_shared_with_concat":
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["x_nchw", "a_nchw", "sum_nchw"]
        model_ir.tensors["concat_nchw"].shape = [1, 8, 5, 7]
        model_ir.tensors["concat_nchw"].shape_signature = [1, 8, 5, 7]
        model_ir.tensors["concat_nhwc"].shape = [1, 5, 7, 8]
        model_ir.tensors["concat_nhwc"].shape_signature = [1, 5, 7, 8]
        model_ir.tensors["y"].shape = [1, 5, 7, 8]
        model_ir.tensors["y"].shape_signature = [1, 5, 7, 8]
    if boundary == "unary_operand":
        model_ir.tensors["a_relu"] = _tensor(
            "a_relu",
            [1, 3, 5, 7],
        )
        add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
        add_index = model_ir.operators.index(add_op)
        model_ir.operators.insert(
            add_index,
            OperatorIR("RELU", ["a_nchw"], ["a_relu"]),
        )
        add_op.inputs[0] = "a_relu"
    if boundary == "public_concat":
        model_ir.outputs.append("concat_nchw")
    if boundary == "public_post":
        model_ir.outputs.append("concat_nhwc")
    if boundary == "raw_residual_input":
        model_ir.inputs.append("residual_nchw")
        model_ir.tensors["residual_nchw"] = _tensor(
            "residual_nchw",
            [1, 2, 5, 7],
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["residual_nchw", "sum_nchw"]
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


def _assert_add_rewritten(
    model_ir: ModelIR,
    *,
    expected_inputs: list[str] | None = None,
) -> None:
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == (expected_inputs or ["a_nhwc", "b_nhwc"])
    assert add_op.options == {"fusedActivationFunction": "NONE"}
    assert model_ir.tensors["sum_nchw"].shape == [1, 5, 7, 3]
    quantization = model_ir.tensors["sum_nchw"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs[-1] == "sum_nchw"
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize("all_add", [False, True])
def test_nhwc_direct_only_add_is_indexed(all_add: bool) -> None:
    model_ir = _add_model(all_add=all_add)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_add",
        "add_output_fanout",
        "public_add_output",
        "public_add_post",
        "invalid_source_rank",
        "invalid_adapter_rank",
        "invalid_add_output_rank",
        "wrong_add_pre_permutation",
        "spatial_shape_mismatch",
        "raw_residual_input",
        "wrong_post_permutation",
        "public_concat",
        "public_post",
    ],
)
def test_nhwc_add_rejects_unsafe_or_partial_match(boundary: str) -> None:
    model_ir = _add_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


@pytest.mark.parametrize(
    "boundary",
    [
        "add_adapter_fanout",
        "add_second_adapter_fanout",
        "public_add_adapter",
    ],
)
def test_nhwc_add_retains_shared_or_public_source_adapter(
    boundary: str,
) -> None:
    model_ir = _add_model(boundary=boundary)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    remaining_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    retained_adapter = (
        "b_nchw"
        if boundary == "add_second_adapter_fanout"
        else "a_nchw"
    )
    assert [op.outputs for op in remaining_transposes] == [[retained_adapter]]
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_adapter_shared_with_root_concat_is_indexed() -> None:
    model_ir = _add_model(boundary="adapter_shared_with_concat")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x_nhwc", "a_nhwc", "sum_nchw"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_unary_operand_is_indexed() -> None:
    model_ir = _add_model(boundary="unary_operand")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(
        model_ir,
        expected_inputs=["a_relu", "b_nhwc"],
    )
    add_op = next(op for op in model_ir.operators if op.op_type == "ADD")
    assert add_op.inputs == ["a_relu", "b_nhwc"]
    unary_op = next(op for op in model_ir.operators if op.op_type == "RELU")
    assert unary_op.inputs == ["a_nhwc"]
    assert model_ir.tensors["a_relu"].shape == [1, 5, 7, 3]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"


def test_nhwc_add_output_post_adapter_is_indexed() -> None:
    model_ir = _add_model(boundary="add_output_post_transpose")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_add_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    side_identity = next(
        op for op in model_ir.operators if op.outputs == ["sum_side"]
    )
    assert side_identity.inputs == ["sum_nchw"]
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_add"
    )
    assert event["status"] == "changed"
