from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_concat_nhwc_chains,
)


def _tensor(name: str, shape: list[int], *, dtype: str = "FLOAT32") -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
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


def _slice_model(
    *,
    all_slice: bool = False,
    share_begin: bool = False,
    public_begin: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_slice_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y"]
    x1_adapter_shape = (
        [1, 4, 5]
        if boundary == "invalid_adapter_rank"
        else [1, 4, 5, 7]
    )
    x1_slice_shape = (
        [1, 2, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 2, 5, 7]
    )
    if boundary == "invalid_slice_output_rank":
        x1_slice_shape = [1, 2, 5]
    begin_values = [0, 1, 0, 0]
    size_values = [1, 2, 5, 7]
    if boundary == "invalid_begin_length":
        begin_values = [0, 1, 0]
    if boundary == "invalid_size_length":
        size_values = [1, 2, 5]
    if boundary == "zero_channel_size":
        size_values = [1, 0, 5, 7]
    if boundary == "nonzero_spatial_begin":
        begin_values = [0, 1, 2, 0]

    model_ir.tensors = {
        "x0_nhwc": _tensor("x0_nhwc", [1, 5, 7, 2]),
        "x1_nhwc": _tensor("x1_nhwc", [1, 5, 7, 4]),
        "pre_perm": _int_tensor("pre_perm", [0, 3, 1, 2]),
        "post_perm": _int_tensor(
            "post_perm",
            [0, 3, 1, 2]
            if boundary == "wrong_post_permutation"
            else [0, 2, 3, 1],
        ),
        "x0_nchw": _tensor("x0_nchw", [1, 2, 5, 7]),
        "x1_nchw": _tensor("x1_nchw", x1_adapter_shape),
        "begin": _int_tensor("begin", begin_values),
        "size": _int_tensor("size", size_values),
        "x1_slice": _tensor("x1_slice", x1_slice_shape),
        "concat_nchw": _tensor("concat_nchw", [1, 4, 5, 7]),
        "concat_nhwc": _tensor("concat_nhwc", [1, 5, 7, 4]),
        "y": _tensor("y", [1, 5, 7, 4]),
    }
    model_ir.tensors["begin"].logical_layout = "NCHW"
    model_ir.tensors["begin"].physical_layout = "NCHW"
    model_ir.tensors["begin"].onnx_tensor_name = "onnx_begin"
    model_ir.tensors["x1_slice"].quantization = QuantParamIR(
        scale=[0.25] * 2,
        zero_point=[0] * 2,
        quantized_dimension=1,
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre_perm"], ["x1_nchw"]),
        OperatorIR(
            "GATHER" if boundary == "unsupported_slice" else "SLICE",
            ["x1_nchw", "begin", "size"],
            ["x1_slice"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["x0_nchw", "x1_slice"],
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

    if all_slice:
        model_ir.tensors.update(
            {
                "x0_begin": _int_tensor("x0_begin", [0, 0, 0, 0]),
                "x0_size": _int_tensor("x0_size", [1, 2, 5, 7]),
                "x0_slice": _tensor("x0_slice", [1, 2, 5, 7]),
            }
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_index = model_ir.operators.index(concat_op)
        model_ir.operators.insert(
            concat_index,
            OperatorIR(
                "SLICE",
                ["x0_nchw", "x0_begin", "x0_size"],
                ["x0_slice"],
            ),
        )
        concat_op.inputs = ["x0_slice", "x1_slice"]

    if share_begin:
        model_ir.tensors["begin_side"] = _tensor(
            "begin_side",
            [4],
            dtype="INT32",
        )
        model_ir.outputs.append("begin_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["begin"], ["begin_side"])
        )
    if public_begin:
        model_ir.outputs.append("begin")
    if boundary == "slice_output_fanout":
        model_ir.tensors["slice_side"] = _tensor(
            "slice_side",
            list(x1_slice_shape),
        )
        model_ir.outputs.append("slice_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_slice"], ["slice_side"])
        )
    if boundary == "slice_output_post_transpose":
        model_ir.tensors["slice_nhwc"] = _tensor(
            "slice_nhwc",
            [1, 5, 7, 2],
        )
        model_ir.tensors["slice_side"] = _tensor(
            "slice_side",
            [1, 5, 7, 2],
        )
        model_ir.outputs.append("slice_side")
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["x1_slice", "post_perm"],
                    ["slice_nhwc"],
                ),
                OperatorIR("IDENTITY", ["slice_nhwc"], ["slice_side"]),
            ]
        )
    if boundary == "public_slice_output":
        model_ir.outputs.append("x1_slice")
    if boundary == "slice_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 4, 5, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_slice_adapter":
        model_ir.outputs.append("x1_nchw")
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
        concat_op.inputs = ["residual_nchw", "x1_slice"]
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
        assert tensor.logical_layout == expected_tensor.logical_layout
        assert tensor.physical_layout == expected_tensor.physical_layout
        assert tensor.onnx_tensor_name == expected_tensor.onnx_tensor_name
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


def _assert_slice_rewritten(model_ir: ModelIR) -> None:
    slice_op = next(
        op
        for op in model_ir.operators
        if op.op_type == "SLICE" and op.outputs == ["x1_slice"]
    )
    assert slice_op.inputs[0] == "x1_nhwc"
    np.testing.assert_array_equal(
        model_ir.tensors[slice_op.inputs[1]].data,
        np.asarray([0, 0, 0, 1], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model_ir.tensors[slice_op.inputs[2]].data,
        np.asarray([1, 5, 7, 2], dtype=np.int32),
    )
    assert model_ir.tensors["x1_slice"].shape == [1, 5, 7, 2]
    quantization = model_ir.tensors["x1_slice"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs[-1] == "x1_slice"
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3


def test_nhwc_direct_slice_with_direct_input_is_indexed() -> None:
    model_ir = _slice_model()
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_slice_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_slice"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_all_direct_slice_inputs_use_indexed_family() -> None:
    model_ir = _slice_model(all_slice=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_slice_rewritten(model_ir)
    slice_ops = [op for op in model_ir.operators if op.op_type == "SLICE"]
    assert {
        op.outputs[0]: op.inputs[0]
        for op in slice_ops
    } == {
        "x0_slice": "x0_nhwc",
        "x1_slice": "x1_nhwc",
    }
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("public_begin", [False, True])
def test_nhwc_slice_copy_on_write_preserves_shared_or_public_begin(
    public_begin: bool,
) -> None:
    model_ir = _slice_model(
        share_begin=not public_begin,
        public_begin=public_begin,
    )

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_slice_rewritten(model_ir)
    np.testing.assert_array_equal(
        model_ir.tensors["begin"].data,
        np.asarray([0, 1, 0, 0], dtype=np.int32),
    )
    slice_op = next(op for op in model_ir.operators if op.op_type == "SLICE")
    assert slice_op.inputs[1] != "begin"
    cloned_begin = model_ir.tensors[slice_op.inputs[1]]
    assert cloned_begin.dtype == "INT32"
    assert cloned_begin.logical_layout == "NCHW"
    assert cloned_begin.physical_layout == "NCHW"
    assert cloned_begin.onnx_tensor_name == "onnx_begin"


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_slice",
        "slice_output_fanout",
        "public_slice_output",
        "public_slice_adapter",
        "invalid_adapter_rank",
        "invalid_slice_output_rank",
        "invalid_begin_length",
        "invalid_size_length",
        "zero_channel_size",
        "nonzero_spatial_begin",
        "spatial_shape_mismatch",
        "raw_residual_input",
        "wrong_post_permutation",
        "public_concat",
        "public_post",
    ],
)
def test_nhwc_slice_rejects_unsafe_or_partial_match(boundary: str) -> None:
    model_ir = _slice_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


@pytest.mark.parametrize(
    "boundary",
    ["slice_adapter_fanout", "slice_output_post_transpose"],
)
def test_nhwc_slice_broader_legacy_cases_remain_available(boundary: str) -> None:
    model_ir = _slice_model(boundary=boundary)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    assert not any(
        event["code"] == "layout.nhwc_pre_concat_slice"
        and event["status"] == "changed"
        for event in diagnostics
    )
