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


def _quantized_model(
    *,
    multiple_posts: bool = False,
    shared_adapter: bool = False,
    public_adapter: bool = False,
    unary_input: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_quantized_direct_pre_concat")
    model_ir.inputs = ["a_nhwc", "b_nhwc"]
    model_ir.outputs = ["y"]
    a_source_shape = (
        [1, 5, 2]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 2]
    )
    concat_shape = (
        [1, 5, 5]
        if boundary == "invalid_concat_rank"
        else [1, 5, 5, 7]
    )
    quantized_shape = (
        [1, 5, 5]
        if boundary == "invalid_quantized_rank"
        else [1, 5, 5, 7]
    )
    post_shape = (
        [1, 5, 7]
        if boundary == "invalid_post_rank"
        else [1, 5, 7, 5]
    )
    model_ir.tensors = {
        "a_nhwc": _tensor("a_nhwc", a_source_shape),
        "b_nhwc": _tensor("b_nhwc", [1, 5, 7, 3]),
        "pre_perm": _int_tensor("pre_perm", [0, 3, 1, 2]),
        "post_perm": _int_tensor(
            "post_perm",
            [0, 3, 1, 2]
            if boundary == "wrong_post_permutation"
            else [0, 2, 3, 1],
        ),
        "a_nchw": _tensor("a_nchw", [1, 2, 5, 7]),
        "b_nchw": _tensor("b_nchw", [1, 3, 5, 7]),
        "concat_nchw": _tensor("concat_nchw", concat_shape),
        "quant_nchw": _tensor(
            "quant_nchw",
            quantized_shape,
            dtype="INT8",
        ),
        "quant_nhwc": _tensor("quant_nhwc", post_shape, dtype="INT8"),
        "y": _tensor("y", [1, 5, 7, 5], dtype="INT8"),
    }
    model_ir.tensors["concat_nchw"].quantization = QuantParamIR(
        scale=[0.1] * 5,
        zero_point=[0] * 5,
        quantized_dimension=1,
    )
    model_ir.tensors["quant_nchw"].quantization = QuantParamIR(
        scale=[0.2] * 5,
        zero_point=[1] * 5,
        quantized_dimension=1,
    )
    model_ir.tensors["quant_nhwc"].quantization = QuantParamIR(
        scale=[9.0],
        zero_point=[9],
        quantized_dimension=0,
    )
    if boundary == "wrong_pre_permutation":
        model_ir.tensors["bad_pre_perm"] = _int_tensor(
            "bad_pre_perm",
            [0, 2, 3, 1],
        )
    if boundary == "wrong_quantize_arity":
        model_ir.tensors["quant_extra"] = _tensor("quant_extra", [1])

    concat_inputs = ["a_nchw", "b_nchw"]
    quantize_inputs = ["concat_nchw"]
    if boundary == "wrong_quantize_arity":
        quantize_inputs.append("quant_extra")
    model_ir.operators = [
        OperatorIR(
            "TRANSPOSE",
            [
                "a_nhwc",
                "bad_pre_perm" if boundary == "wrong_pre_permutation" else "pre_perm",
            ],
            ["a_nchw"],
        ),
        OperatorIR("TRANSPOSE", ["b_nhwc", "pre_perm"], ["b_nchw"]),
        OperatorIR(
            "CONCATENATION",
            concat_inputs,
            ["concat_nchw"],
            options={"axis": 2 if boundary == "wrong_concat_axis" else 1},
        ),
        OperatorIR(
            "CAST" if boundary == "unsupported_quantize" else "QUANTIZE",
            quantize_inputs,
            ["quant_nchw"],
        ),
        OperatorIR(
            "TRANSPOSE",
            ["quant_nchw", "post_perm"],
            ["quant_nhwc"],
        ),
        OperatorIR("IDENTITY", ["quant_nhwc"], ["y"]),
    ]

    if unary_input:
        model_ir.tensors["a_relu"] = _tensor("a_relu", [1, 2, 5, 7])
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_index = model_ir.operators.index(concat_op)
        model_ir.operators.insert(
            concat_index,
            OperatorIR("RELU", ["a_nchw"], ["a_relu"]),
        )
        concat_op.inputs[0] = "a_relu"
    if multiple_posts:
        model_ir.tensors["quant_nhwc_2"] = _tensor(
            "quant_nhwc_2",
            [1, 5, 7, 5],
            dtype="INT8",
        )
        model_ir.tensors["y2"] = _tensor(
            "y2",
            [1, 5, 7, 5],
            dtype="INT8",
        )
        model_ir.outputs.append("y2")
        identity_index = next(
            index
            for index, op in enumerate(model_ir.operators)
            if op.outputs == ["y"]
        )
        model_ir.operators[identity_index:identity_index] = [
            OperatorIR(
                "TRANSPOSE",
                ["quant_nchw", "post_perm"],
                ["quant_nhwc_2"],
            ),
            OperatorIR("IDENTITY", ["quant_nhwc_2"], ["y2"]),
        ]
    if shared_adapter:
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            [1, 2, 5, 7],
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["a_nchw"], ["adapter_side"])
        )
    if public_adapter:
        model_ir.outputs.append("a_nchw")
    if boundary == "quantized_output_fanout":
        model_ir.tensors["quant_side"] = _tensor(
            "quant_side",
            list(quantized_shape),
            dtype="INT8",
        )
        model_ir.outputs.append("quant_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["quant_nchw"], ["quant_side"])
        )
    if boundary == "public_quantized_output":
        model_ir.outputs.append("quant_nchw")
    if boundary == "concat_fanout":
        model_ir.tensors["concat_side"] = _tensor(
            "concat_side",
            list(concat_shape),
        )
        model_ir.outputs.append("concat_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["concat_nchw"], ["concat_side"])
        )
    if boundary == "public_concat":
        model_ir.outputs.append("concat_nchw")
    if boundary == "public_post":
        model_ir.outputs.append("quant_nhwc")
    if boundary == "raw_concat_input":
        model_ir.inputs.append("raw_nchw")
        model_ir.tensors["raw_nchw"] = _tensor(
            "raw_nchw",
            [1, 2, 5, 7],
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_op.inputs[0] = "raw_nchw"
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


def _assert_quantized_rewritten(model_ir: ModelIR) -> None:
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    quantize_op = next(op for op in model_ir.operators if op.op_type == "QUANTIZE")
    assert concat_op.inputs == ["a_nhwc", "b_nhwc"]
    assert concat_op.options["axis"] == 3
    assert model_ir.tensors["concat_nchw"].shape == [1, 5, 7, 5]
    concat_quantization = model_ir.tensors["concat_nchw"].quantization
    assert isinstance(concat_quantization, QuantParamIR)
    assert concat_quantization.quantized_dimension == 3
    assert quantize_op.inputs == ["concat_nchw"]
    assert quantize_op.outputs == ["quant_nhwc"]
    canonical_tensor = model_ir.tensors["quant_nhwc"]
    assert canonical_tensor.dtype == "INT8"
    assert canonical_tensor.shape == [1, 5, 7, 5]
    canonical_quantization = canonical_tensor.quantization
    assert isinstance(canonical_quantization, QuantParamIR)
    assert canonical_quantization.scale == [0.2] * 5
    assert canonical_quantization.zero_point == [1] * 5
    assert canonical_quantization.quantized_dimension == 3


def test_nhwc_quantized_all_direct_family_is_indexed() -> None:
    model_ir = _quantized_model()
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_quantized_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_quantized_direct"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_quantized_multiple_posts_share_canonical_output() -> None:
    model_ir = _quantized_model(multiple_posts=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_quantized_rewritten(model_ir)
    identity_inputs = [
        op.inputs
        for op in model_ir.operators
        if op.op_type == "IDENTITY"
    ]
    assert identity_inputs == [["quant_nhwc"], ["quant_nhwc"]]
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize("public_adapter", [False, True])
def test_nhwc_quantized_shared_or_public_adapter_is_retained(
    public_adapter: bool,
) -> None:
    model_ir = _quantized_model(
        shared_adapter=not public_adapter,
        public_adapter=public_adapter,
    )

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_quantized_rewritten(model_ir)
    remaining_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert [op.outputs for op in remaining_transposes] == [["a_nchw"]]


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_quantize",
        "wrong_quantize_arity",
        "quantized_output_fanout",
        "public_quantized_output",
        "concat_fanout",
        "public_concat",
        "public_post",
        "invalid_source_rank",
        "invalid_concat_rank",
        "invalid_quantized_rank",
        "invalid_post_rank",
        "wrong_concat_axis",
        "raw_concat_input",
        "wrong_pre_permutation",
        "wrong_post_permutation",
    ],
)
def test_nhwc_quantized_direct_rejects_unsafe_or_partial_match(
    boundary: str,
) -> None:
    model_ir = _quantized_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_quantized_unary_input_remains_in_legacy() -> None:
    model_ir = _quantized_model(unary_input=True)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    assert not any(
        event["code"] == "layout.nhwc_pre_concat_quantized_direct"
        and event["status"] == "changed"
        for event in diagnostics
    )
