from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
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


def _model(*, boundary: str | None = None) -> ModelIR:
    model_ir = ModelIR("nhwc_direct_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors = {
        "x0_nhwc": _tensor("x0_nhwc", [1, 5, 7, 2]),
        "x1_nhwc": _tensor(
            "x1_nhwc",
            [1, 9 if boundary == "stale_spatial_metadata" else 5, 7, 3],
        ),
        "pre0_perm": TensorIR(
            "pre0_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray(
                [0, 1, 2, 3]
                if boundary == "invalid_pre_permutation"
                else [0, 3, 1, 2],
                dtype=np.int32,
            ),
        ),
        "pre1_perm": TensorIR(
            "pre1_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "post0_perm": TensorIR(
            "post0_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray(
                [0, 1, 2, 3]
                if boundary == "invalid_post_permutation"
                else [0, 2, 3, 1],
                dtype=np.int32,
            ),
        ),
        "post1_perm": TensorIR(
            "post1_perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "x0_nchw": _tensor("x0_nchw", [1, 2, 5, 7]),
        "x1_nchw": _tensor("x1_nchw", [1, 3, 5, 7]),
        "concat_nchw": _tensor("concat_nchw", [1, 5, 5, 7]),
        "post0_nhwc": _tensor("post0_nhwc", [1, 5, 7, 5]),
        "post1_nhwc": _tensor("post1_nhwc", [1, 5, 7, 5]),
        "y0": _tensor("y0", [1, 5, 7, 5]),
        "y1": _tensor("y1", [1, 5, 7, 5]),
    }
    if boundary == "invalid_source_rank":
        model_ir.tensors["x0_nhwc"].shape = [1, 5, 2]
        model_ir.tensors["x0_nhwc"].shape_signature = [1, 5, 2]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre0_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre1_perm"], ["x1_nchw"]),
        OperatorIR(
            "CONCATENATION",
            ["x0_nchw", "x1_nchw"],
            ["concat_nchw"],
            options={"axis": 2 if boundary == "concat_axis" else 1},
        ),
        OperatorIR(
            "TRANSPOSE",
            ["concat_nchw", "post0_perm"],
            ["post0_nhwc"],
        ),
        OperatorIR("RELU", ["post0_nhwc"], ["y0"]),
        OperatorIR(
            "TRANSPOSE",
            ["concat_nchw", "post1_perm"],
            ["post1_nhwc"],
        ),
        OperatorIR("RELU", ["post1_nhwc"], ["y1"]),
    ]
    if boundary == "concat_nontranspose_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 5, 5, 7])
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["concat_nchw"], ["side"])
        )
    if boundary == "public_concat":
        model_ir.outputs.append("concat_nchw")
    if boundary == "public_post":
        model_ir.outputs.append("post0_nhwc")
    return model_ir


def _unary_model(
    *,
    unary_op: str = "RELU",
    boundary: str | None = None,
) -> ModelIR:
    model_ir = _model()
    if boundary == "spatial_shape_mismatch":
        model_ir.tensors["x1_nhwc"].shape = [1, 9, 7, 3]
        model_ir.tensors["x1_nhwc"].shape_signature = [1, 9, 7, 3]
        model_ir.tensors["x1_nchw"].shape = [1, 3, 9, 7]
        model_ir.tensors["x1_nchw"].shape_signature = [1, 3, 9, 7]
    unary_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_unary_rank":
        unary_shape = [1, 3, 5]
    model_ir.tensors["x1_unary"] = _tensor("x1_unary", unary_shape)
    model_ir.operators.insert(
        2,
        OperatorIR(unary_op, ["x1_nchw"], ["x1_unary"]),
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat_op.inputs = ["x0_nchw", "x1_unary"]

    if boundary == "unary_output_fanout":
        model_ir.tensors["unary_side"] = _tensor(
            "unary_side",
            list(unary_shape),
        )
        model_ir.outputs.append("unary_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_unary"], ["unary_side"])
        )
    if boundary == "public_unary_output":
        model_ir.outputs.append("x1_unary")
    if boundary == "unary_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            list(model_ir.tensors["x1_nchw"].shape),
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_unary_adapter":
        model_ir.outputs.append("x1_nchw")
    return model_ir


def _assert_model_equal(actual: ModelIR, expected: ModelIR) -> None:
    assert actual.inputs == expected.inputs
    assert actual.outputs == expected.outputs
    assert actual.description == expected.description
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


@pytest.mark.parametrize("boundary", [None, "stale_spatial_metadata"])
def test_nhwc_direct_pre_concat_multi_post_is_transactionally_optimized(
    boundary: str | None,
) -> None:
    model_ir = _model(boundary=boundary)
    model_ir.tensors["concat_nchw"].quantization = QuantParamIR(
        scale=[0.25] * 5,
        zero_point=[0] * 5,
        quantized_dimension=1,
    )
    model_ir.tensors["side0"] = _tensor("side0", [1, 2, 5, 7])
    model_ir.outputs.extend(["side0", "x1_nchw"])
    model_ir.operators.append(
        OperatorIR("IDENTITY", ["x0_nchw"], ["side0"])
    )
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    assert [op.op_type for op in model_ir.operators].count("TRANSPOSE") == 2
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_nhwc"]
    assert concat_op.outputs == ["post0_nhwc"]
    assert concat_op.options["axis"] == 3
    assert model_ir.tensors["post0_nhwc"].shape == [1, 5, 7, 5]
    quantization = model_ir.tensors["post0_nhwc"].quantization
    assert isinstance(quantization, QuantParamIR)
    assert quantization.quantized_dimension == 3
    assert [
        op.inputs for op in model_ir.operators if op.op_type == "RELU"
    ] == [["post0_nhwc"], ["post0_nhwc"]]
    assert diagnostics[0]["code"] == "layout.nhwc_pre_concat_direct"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"] == {
        "preflight_operators_visited": 3,
        "state_built": True,
        "snapshot_count": 1,
        "fingerprint_count": 0,
    }


@pytest.mark.parametrize(
    "boundary",
    [
        "invalid_pre_permutation",
        "invalid_post_permutation",
        "concat_axis",
        "invalid_source_rank",
        "concat_nontranspose_fanout",
        "public_concat",
        "public_post",
    ],
)
def test_nhwc_direct_pre_concat_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


@pytest.mark.parametrize(
    "unary_op",
    ["RELU", "RELU6", "LOGISTIC", "TANH", "GELU"],
)
def test_nhwc_one_unary_plus_direct_pre_concat_is_indexed(
    unary_op: str,
) -> None:
    model_ir = _unary_model(unary_op=unary_op)
    model_ir.tensors["x1_unary"].quantization = {
        "scale": [0.5] * 3,
        "zero_point": [0] * 3,
        "quantized_dimension": 1,
    }
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    unary = next(op for op in model_ir.operators if op.op_type == unary_op)
    assert unary.inputs == ["x1_nhwc"]
    assert model_ir.tensors["x1_unary"].shape == [1, 5, 7, 3]
    quantization = model_ir.tensors["x1_unary"].quantization
    assert isinstance(quantization, dict)
    assert quantization["quantized_dimension"] == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_unary"]
    assert concat_op.outputs == ["post0_nhwc"]
    assert concat_op.options["axis"] == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    unary_event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_unary"
    )
    assert unary_event["status"] == "changed"
    assert unary_event["metrics"]["snapshot_count"] == 1
    assert unary_event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize(
    ("boundary", "unary_op"),
    [
        ("unsupported_unary", "ABS"),
        ("unary_output_fanout", "RELU"),
        ("public_unary_output", "RELU"),
        ("unary_adapter_fanout", "RELU"),
        ("public_unary_adapter", "RELU"),
        ("spatial_shape_mismatch", "RELU"),
        ("invalid_unary_rank", "RELU"),
    ],
)
def test_nhwc_one_unary_plus_direct_rejects_unsafe_boundary(
    boundary: str,
    unary_op: str,
) -> None:
    model_ir = _unary_model(unary_op=unary_op, boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)
