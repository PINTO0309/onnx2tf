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


def _split_model(
    *,
    all_split: bool = False,
    share_axis: bool = False,
    public_axis: bool = False,
    negative_axis: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_split_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y"]
    adapter_shape = (
        [1, 4, 5]
        if boundary == "invalid_adapter_rank"
        else [1, 4, 5, 7]
    )
    split1_shape = (
        [1, 2, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 2, 5, 7]
    )
    if boundary == "invalid_split_output_rank":
        split1_shape = [1, 2, 5]
    axis_values = [-3 if negative_axis else 1]
    if boundary == "invalid_axis_length":
        axis_values = [1, 1]
    if boundary == "wrong_axis":
        axis_values = [2]

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
        "x1_nchw": _tensor("x1_nchw", adapter_shape),
        "axis": _int_tensor("axis", axis_values),
        "x1_split0": _tensor("x1_split0", [1, 2, 5, 7]),
        "x1_split1": _tensor("x1_split1", split1_shape),
        "concat_nchw": _tensor(
            "concat_nchw",
            [1, 4 if all_split else 6, 5, 7],
        ),
        "concat_nhwc": _tensor(
            "concat_nhwc",
            [1, 5, 7, 4 if all_split else 6],
        ),
        "y": _tensor("y", [1, 5, 7, 4 if all_split else 6]),
    }
    model_ir.tensors["axis"].logical_layout = "NCHW"
    model_ir.tensors["axis"].physical_layout = "NCHW"
    model_ir.tensors["axis"].onnx_tensor_name = "onnx_axis"
    for output_name in ("x1_split0", "x1_split1"):
        model_ir.tensors[output_name].quantization = QuantParamIR(
            scale=[0.25] * 2,
            zero_point=[0] * 2,
            quantized_dimension=1,
        )

    split_outputs = (
        ["x1_split0"]
        if boundary == "single_output"
        else ["x1_split0", "x1_split1"]
    )
    if boundary == "missing_axis_data":
        model_ir.tensors["axis"].data = None
    concat_inputs = ["x1_split0", "x1_split1"]
    if not all_split:
        concat_inputs.insert(0, "x0_nchw")
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre_perm"], ["x1_nchw"]),
        OperatorIR(
            "UNPACK" if boundary == "unsupported_split" else "SPLIT",
            ["axis", "x1_nchw"],
            split_outputs,
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
    if all_split:
        model_ir.inputs = ["x1_nhwc"]
        model_ir.operators = [
            op for op in model_ir.operators if op.outputs != ["x0_nchw"]
        ]
        del model_ir.tensors["x0_nhwc"]
        del model_ir.tensors["x0_nchw"]

    if share_axis:
        model_ir.tensors["axis_side"] = _tensor(
            "axis_side",
            [1],
            dtype="INT32",
        )
        model_ir.outputs.append("axis_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["axis"], ["axis_side"])
        )
    if public_axis:
        model_ir.outputs.append("axis")
    if boundary == "split_output_fanout":
        model_ir.tensors["split_side"] = _tensor(
            "split_side",
            list(split1_shape),
        )
        model_ir.outputs.append("split_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_split1"], ["split_side"])
        )
    if boundary == "split_output_post_transpose":
        model_ir.tensors["split_nhwc"] = _tensor(
            "split_nhwc",
            [1, 5, 7, 2],
        )
        model_ir.tensors["split_side"] = _tensor(
            "split_side",
            [1, 5, 7, 2],
        )
        model_ir.outputs.append("split_side")
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["x1_split1", "post_perm"],
                    ["split_nhwc"],
                ),
                OperatorIR("IDENTITY", ["split_nhwc"], ["split_side"]),
            ]
        )
    if boundary == "public_split_output":
        model_ir.outputs.append("x1_split1")
    if boundary == "split_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 4, 5, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_split_adapter":
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
        concat_op.inputs = ["residual_nchw", "x1_split0", "x1_split1"]
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


def _assert_split_rewritten(model_ir: ModelIR) -> None:
    split_op = next(op for op in model_ir.operators if op.op_type == "SPLIT")
    assert split_op.inputs[1] == "x1_nhwc"
    np.testing.assert_array_equal(
        model_ir.tensors[split_op.inputs[0]].data,
        np.asarray([3], dtype=np.int32),
    )
    for output_name in ("x1_split0", "x1_split1"):
        assert model_ir.tensors[output_name].shape == [1, 5, 7, 2]
        quantization = model_ir.tensors[output_name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs[-2:] == ["x1_split0", "x1_split1"]
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize("negative_axis", [False, True])
def test_nhwc_direct_split_with_direct_input_is_indexed(
    negative_axis: bool,
) -> None:
    model_ir = _split_model(negative_axis=negative_axis)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_split_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_split"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_all_outputs_from_one_split_use_indexed_family() -> None:
    model_ir = _split_model(all_split=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_split_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("public_axis", [False, True])
def test_nhwc_split_copy_on_write_preserves_shared_or_public_axis(
    public_axis: bool,
) -> None:
    model_ir = _split_model(
        share_axis=not public_axis,
        public_axis=public_axis,
    )

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_split_rewritten(model_ir)
    np.testing.assert_array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([1], dtype=np.int32),
    )
    split_op = next(op for op in model_ir.operators if op.op_type == "SPLIT")
    assert split_op.inputs[0] != "axis"
    cloned_axis = model_ir.tensors[split_op.inputs[0]]
    assert cloned_axis.dtype == "INT32"
    assert cloned_axis.logical_layout == "NCHW"
    assert cloned_axis.physical_layout == "NCHW"
    assert cloned_axis.onnx_tensor_name == "onnx_axis"


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_split",
        "single_output",
        "split_output_fanout",
        "public_split_output",
        "public_split_adapter",
        "invalid_adapter_rank",
        "invalid_split_output_rank",
        "invalid_axis_length",
        "missing_axis_data",
        "wrong_axis",
        "spatial_shape_mismatch",
        "raw_residual_input",
        "wrong_post_permutation",
        "public_concat",
        "public_post",
    ],
)
def test_nhwc_split_rejects_unsafe_or_partial_match(boundary: str) -> None:
    model_ir = _split_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


@pytest.mark.parametrize(
    "boundary",
    ["split_adapter_fanout", "split_output_post_transpose"],
)
def test_nhwc_split_broader_legacy_cases_remain_available(boundary: str) -> None:
    model_ir = _split_model(boundary=boundary)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    assert not any(
        event["code"] == "layout.nhwc_pre_concat_split"
        and event["status"] == "changed"
        for event in diagnostics
    )
