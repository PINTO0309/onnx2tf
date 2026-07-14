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


def _softmax_model(
    *,
    boundary: str | None = None,
    all_softmax: bool = False,
) -> ModelIR:
    model_ir = ModelIR("nhwc_softmax_pre_concat")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["y"]
    source_shape = (
        [1, 5, 3]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 3]
    )
    softmax_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 3, 5, 7]
    )
    if boundary == "invalid_softmax_output_rank":
        softmax_shape = [1, 3, 5]
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
        "x1_softmax": _tensor("x1_softmax", softmax_shape),
        "concat_nchw": _tensor("concat_nchw", [1, 5, 5, 7]),
        "concat_nhwc": _tensor("concat_nhwc", [1, 5, 7, 5]),
        "y": _tensor("y", [1, 5, 7, 5]),
    }
    model_ir.tensors["x1_nhwc"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=3,
    )
    model_ir.tensors["x1_nchw"].quantization = QuantParamIR(
        scale=[0.25] * 3,
        zero_point=[0] * 3,
        quantized_dimension=1,
    )
    model_ir.tensors["x1_softmax"].quantization = {
        "scale": [0.5] * 3,
        "zero_point": [0] * 3,
        "quantized_dimension": 1,
    }
    softmax_inputs = ["x1_nchw"]
    if boundary == "invalid_softmax_arity":
        model_ir.tensors["extra"] = _tensor("extra", [1])
        softmax_inputs.append("extra")
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x0_nhwc", "pre_perm"], ["x0_nchw"]),
        OperatorIR("TRANSPOSE", ["x1_nhwc", "pre_perm"], ["x1_nchw"]),
        OperatorIR(
            "IDENTITY" if boundary == "unsupported_softmax" else "SOFTMAX",
            softmax_inputs,
            ["x1_softmax"],
            options={"beta": 0.75},
        ),
        OperatorIR(
            "CONCATENATION",
            ["x0_nchw", "x1_softmax"],
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

    if all_softmax:
        model_ir.tensors["x0_softmax"] = _tensor(
            "x0_softmax",
            [1, 2, 5, 7],
        )
        model_ir.operators.insert(
            1,
            OperatorIR(
                "SOFTMAX",
                ["x0_nchw"],
                ["x0_softmax"],
                options={"beta": 1.0},
            ),
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs = ["x0_softmax", "x1_softmax"]

    if boundary == "softmax_output_fanout":
        model_ir.tensors["softmax_side"] = _tensor(
            "softmax_side",
            list(softmax_shape),
        )
        model_ir.outputs.append("softmax_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_softmax"], ["softmax_side"])
        )
    if boundary == "public_softmax_output":
        model_ir.outputs.append("x1_softmax")
    if boundary == "softmax_adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 3, 5, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["x1_nchw"], ["adapter_side"])
        )
    if boundary == "public_softmax_adapter":
        model_ir.outputs.append("x1_nchw")
    if boundary == "quantized_post":
        model_ir.tensors["concat_quantized"] = TensorIR(
            "concat_quantized",
            "INT8",
            [1, 5, 5, 7],
            [1, 5, 5, 7],
            quantization=QuantParamIR(
                scale=[0.125],
                zero_point=[0],
                quantized_dimension=0,
            ),
        )
        concat_index = next(
            index
            for index, op in enumerate(model_ir.operators)
            if op.op_type == "CONCATENATION"
        )
        model_ir.operators.insert(
            concat_index + 1,
            OperatorIR(
                "QUANTIZE",
                ["concat_nchw"],
                ["concat_quantized"],
            ),
        )
        post_op = next(
            op
            for op in model_ir.operators
            if op.op_type == "TRANSPOSE" and op.outputs == ["concat_nhwc"]
        )
        post_op.inputs[0] = "concat_quantized"
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


def _softmax(value: np.ndarray, *, beta: float) -> np.ndarray:
    shifted = value * beta - np.max(value * beta, axis=-1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=-1, keepdims=True)


def test_nhcw_local_adapters_preserve_nchw_last_axis_softmax() -> None:
    rng = np.random.default_rng(0)
    x_nhwc = rng.normal(size=(1, 5, 7, 3)).astype(np.float32)

    original_nchw = np.transpose(x_nhwc, axes=[0, 3, 1, 2])
    original_nhwc = np.transpose(
        _softmax(original_nchw, beta=0.75),
        axes=[0, 2, 3, 1],
    )
    local_nhcw = np.transpose(x_nhwc, axes=[0, 1, 3, 2])
    rewritten_nhwc = np.transpose(
        _softmax(local_nhcw, beta=0.75),
        axes=[0, 1, 3, 2],
    )

    np.testing.assert_allclose(rewritten_nhwc, original_nhwc, rtol=0, atol=0)


def test_nhwc_softmax_plus_direct_inserts_indexed_local_adapters() -> None:
    model_ir = _softmax_model()
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    softmax_op = next(op for op in model_ir.operators if op.op_type == "SOFTMAX")
    assert softmax_op.options == {"beta": 0.75}
    assert softmax_op.inputs == ["x1_nchw_axis_last"]
    assert softmax_op.outputs == ["x1_softmax_axis_last"]
    assert model_ir.tensors["x1_nchw_axis_last"].shape == [1, 5, 3, 7]
    assert model_ir.tensors["x1_softmax_axis_last"].shape == [1, 5, 3, 7]
    input_quantization = model_ir.tensors[
        "x1_nchw_axis_last"
    ].quantization
    assert isinstance(input_quantization, QuantParamIR)
    assert input_quantization.quantized_dimension == 2
    axis_output_quantization = model_ir.tensors[
        "x1_softmax_axis_last"
    ].quantization
    assert isinstance(axis_output_quantization, dict)
    assert axis_output_quantization["quantized_dimension"] == 2
    final_quantization = model_ir.tensors["x1_softmax"].quantization
    assert isinstance(final_quantization, dict)
    assert final_quantization["quantized_dimension"] == 3
    local_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert len(local_transposes) == 2
    for transpose in local_transposes:
        perm_tensor = model_ir.tensors[transpose.inputs[1]]
        np.testing.assert_array_equal(
            perm_tensor.data,
            np.asarray([0, 1, 3, 2], dtype=np.int32),
        )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_softmax"]
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_softmax"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_softmax",
        "invalid_softmax_arity",
        "softmax_output_fanout",
        "public_softmax_output",
        "softmax_adapter_fanout",
        "public_softmax_adapter",
        "invalid_source_rank",
        "invalid_softmax_output_rank",
        "spatial_shape_mismatch",
    ],
)
def test_nhwc_softmax_rejects_unsafe_boundary(boundary: str) -> None:
    model_ir = _softmax_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_softmax_quantized_post_uses_indexed_family() -> None:
    model_ir = _softmax_model(boundary="quantized_post")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    quantize_op = next(
        op for op in model_ir.operators if op.op_type == "QUANTIZE"
    )
    assert concat_op.inputs == ["x0_nhwc", "x1_softmax"]
    assert concat_op.options["axis"] == 3
    assert quantize_op.inputs == ["concat_nchw"]
    assert quantize_op.outputs == ["concat_nhwc"]
    assert model_ir.tensors["concat_nhwc"].dtype == "INT8"
    assert model_ir.tensors["concat_nhwc"].shape == [1, 5, 7, 5]
    local_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert len(local_transposes) == 2
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_quantized_softmax"
    )
    assert event["status"] == "changed"


def test_nhwc_softmax_rejects_multiple_softmax_inputs() -> None:
    model_ir = _softmax_model(all_softmax=True)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)
