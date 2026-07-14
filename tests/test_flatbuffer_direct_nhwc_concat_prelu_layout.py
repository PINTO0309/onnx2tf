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


def _tensor(name: str, shape: list[int], dtype: str = "FLOAT32") -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
    )


def _prelu_model(
    *,
    alpha_rank: int = 4,
    boundary: str | None = None,
    shared_alpha: bool = False,
    all_prelu: bool = False,
) -> ModelIR:
    model_ir = ModelIR("nhwc_prelu_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y"]
    alpha_shapes = {
        1: [3],
        3: [3, 1, 1],
        4: [1, 3, 1, 1],
    }
    alpha_shape = alpha_shapes[alpha_rank]
    if boundary == "unbroadcastable_alpha":
        alpha_shape = [2, 2]
    prelu_output_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_prelu_output_rank":
        prelu_output_shape = [1, 3, 5]
    source_shape = (
        [1, 5, 3]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 3]
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
        "x1_nchw": _tensor("x1_nchw", [1, 3, 5, 7]),
        "x1_prelu": _tensor("x1_prelu", prelu_output_shape),
        "alpha": TensorIR(
            "alpha",
            "FLOAT32",
            list(alpha_shape),
            list(alpha_shape),
            data=(
                None
                if boundary == "missing_alpha"
                else np.full(alpha_shape, 0.25, dtype=np.float32)
            ),
            quantization=QuantParamIR(
                scale=[0.01] * 3,
                zero_point=[0] * 3,
                quantized_dimension=0 if alpha_rank in {1, 3} else 1,
            ),
            onnx_tensor_name="onnx_alpha",
        ),
        "concat_nchw": _tensor("concat_nchw", [1, 5, 5, 7]),
        "concat_nhwc": _tensor("concat_nhwc", [1, 5, 7, 5]),
        "y": _tensor("y", [1, 5, 7, 5]),
    }
    model_ir.tensors["x1_prelu"].quantization = {
        "scale": [0.5] * 3,
        "zero_point": [0] * 3,
        "quantized_dimension": 1,
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre_perm"], ["x1_nchw"]),
        OperatorIR(
            "RELU" if boundary == "unsupported_prelu" else "PRELU",
            ["x1_nchw", "alpha"],
            ["x1_prelu"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["x0_nchw", "x1_prelu"],
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

    if all_prelu:
        alpha0 = np.full((1, 2, 1, 1), 0.125, dtype=np.float32)
        model_ir.tensors["alpha0"] = TensorIR(
            "alpha0",
            "FLOAT32",
            list(alpha0.shape),
            list(alpha0.shape),
            data=alpha0,
        )
        model_ir.tensors["x0_prelu"] = _tensor(
            "x0_prelu",
            [1, 2, 5, 7],
        )
        model_ir.operators.insert(
            1,
            OperatorIR("PRELU", ["x0_nchw", "alpha0"], ["x0_prelu"]),
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["x0_prelu", "x1_prelu"]

    if boundary == "prelu_output_fanout":
        model_ir.tensors["prelu_side"] = _tensor(
            "prelu_side",
            list(prelu_output_shape),
        )
        model_ir.outputs.append("prelu_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_prelu"], ["prelu_side"])
        )
    if boundary == "public_prelu_output":
        model_ir.outputs.append("x1_prelu")
    if boundary == "prelu_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 3, 5, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_prelu_adapter":
        model_ir.outputs.append("x1_nchw")
    if shared_alpha:
        model_ir.inputs.append("outside_nchw")
        model_ir.outputs.append("outside_prelu")
        model_ir.tensors["outside_nchw"] = _tensor(
            "outside_nchw",
            [1, 3, 5, 7],
        )
        model_ir.tensors["outside_prelu"] = _tensor(
            "outside_prelu",
            [1, 3, 5, 7],
        )
        model_ir.operators.append(
            OperatorIR(
                "PRELU",
                ["outside_nchw", "alpha"],
                ["outside_prelu"],
            )
        )
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
        assert tensor.onnx_tensor_name == expected_tensor.onnx_tensor_name
        if tensor.data is None or expected_tensor.data is None:
            assert tensor.data is expected_tensor.data
        else:
            np.testing.assert_array_equal(tensor.data, expected_tensor.data)


@pytest.mark.parametrize(
    ("alpha_rank", "expected_shape", "expected_quantized_dimension"),
    [
        (4, [1, 1, 1, 3], 3),
        (3, [1, 1, 3], 2),
        (1, [3], 0),
    ],
)
def test_nhwc_prelu_plus_direct_selects_broadcastable_alpha(
    alpha_rank: int,
    expected_shape: list[int],
    expected_quantized_dimension: int,
) -> None:
    model_ir = _prelu_model(alpha_rank=alpha_rank)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    prelu_op = next(op for op in model_ir.operators if op.op_type == "PRELU")
    assert prelu_op.inputs == ["x1_nhwc", "alpha"]
    assert model_ir.tensors["alpha"].shape == expected_shape
    alpha_quantization = model_ir.tensors["alpha"].quantization
    assert isinstance(alpha_quantization, QuantParamIR)
    assert (
        alpha_quantization.quantized_dimension
        == expected_quantized_dimension
    )
    assert model_ir.tensors["x1_prelu"].shape == [1, 5, 7, 3]
    output_quantization = model_ir.tensors["x1_prelu"].quantization
    assert isinstance(output_quantization, dict)
    assert output_quantization["quantized_dimension"] == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_prelu"]
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_prelu"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_prelu_clones_shared_alpha() -> None:
    model_ir = _prelu_model(shared_alpha=True)
    original_alpha = np.array(model_ir.tensors["alpha"].data, copy=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    prelu_ops = [op for op in model_ir.operators if op.op_type == "PRELU"]
    selected_prelu = next(op for op in prelu_ops if op.outputs == ["x1_prelu"])
    outside_prelu = next(
        op for op in prelu_ops if op.outputs == ["outside_prelu"]
    )
    assert outside_prelu.inputs[1] == "alpha"
    np.testing.assert_array_equal(
        model_ir.tensors["alpha"].data,
        original_alpha,
    )
    selected_alpha_name = selected_prelu.inputs[1]
    assert selected_alpha_name != "alpha"
    selected_alpha = model_ir.tensors[selected_alpha_name]
    assert selected_alpha.shape == [1, 1, 1, 3]
    assert selected_alpha.onnx_tensor_name == "onnx_alpha"
    selected_quantization = selected_alpha.quantization
    assert isinstance(selected_quantization, QuantParamIR)
    assert selected_quantization.quantized_dimension == 3


def test_nhwc_all_prelu_inputs_use_indexed_family() -> None:
    model_ir = _prelu_model(all_prelu=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    prelu_ops = [op for op in model_ir.operators if op.op_type == "PRELU"]
    assert [op.inputs[0] for op in prelu_ops] == ["x0_nhwc", "x1_nhwc"]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_prelu", "x1_prelu"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_prelu",
        "prelu_output_fanout",
        "public_prelu_output",
        "prelu_adapter_fanout",
        "public_prelu_adapter",
        "missing_alpha",
        "unbroadcastable_alpha",
        "invalid_prelu_output_rank",
        "invalid_source_rank",
        "spatial_shape_mismatch",
    ],
)
def test_nhwc_prelu_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _prelu_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)
