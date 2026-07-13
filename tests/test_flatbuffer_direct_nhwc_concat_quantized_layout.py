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
    unary_op_type: str = "RELU",
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
        unary_shape = (
            [1, 2, 9, 7]
            if boundary == "unary_spatial_mismatch"
            else [1, 2, 5, 7]
        )
        if boundary == "invalid_unary_output_rank":
            unary_shape = [1, 2, 5]
        model_ir.tensors["a_relu"] = _tensor("a_relu", unary_shape)
        model_ir.tensors["a_relu"].quantization = QuantParamIR(
            scale=[0.3] * 2,
            zero_point=[0] * 2,
            quantized_dimension=1,
        )
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_index = model_ir.operators.index(concat_op)
        model_ir.operators.insert(
            concat_index,
            OperatorIR(
                "ABS" if boundary == "unsupported_unary" else unary_op_type,
                ["a_nchw"],
                ["a_relu"],
            ),
        )
        concat_op.inputs[0] = "a_relu"
        if boundary == "unary_output_fanout":
            model_ir.tensors["unary_side"] = _tensor(
                "unary_side",
                list(unary_shape),
            )
            model_ir.outputs.append("unary_side")
            model_ir.operators.append(
                OperatorIR("IDENTITY", ["a_relu"], ["unary_side"])
            )
        if boundary == "public_unary_output":
            model_ir.outputs.append("a_relu")
        if boundary == "unary_adapter_fanout":
            model_ir.tensors["unary_adapter_side"] = _tensor(
                "unary_adapter_side",
                [1, 2, 5, 7],
            )
            model_ir.outputs.append("unary_adapter_side")
            model_ir.operators.append(
                OperatorIR(
                    "IDENTITY",
                    ["a_nchw"],
                    ["unary_adapter_side"],
                )
            )
        if boundary == "public_unary_adapter":
            model_ir.outputs.append("a_nchw")
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


def _quantized_pad_model(
    *,
    boundary: str | None = None,
    shared_adapter: bool = False,
    shared_pads: bool = False,
    public_pads: bool = False,
) -> ModelIR:
    model_ir = _quantized_model()
    model_ir.tensors["a_nhwc"].shape = [1, 4, 7, 2]
    model_ir.tensors["a_nhwc"].shape_signature = [1, 4, 7, 2]
    model_ir.tensors["a_nchw"].shape = [1, 2, 4, 7]
    model_ir.tensors["a_nchw"].shape_signature = [1, 2, 4, 7]
    pads_data = np.asarray(
        [[0, 0], [0, 0], [0, 1], [0, 0]],
        dtype=np.int32,
    )
    if boundary == "invalid_pads_shape":
        pads_data = pads_data[:3]
    model_ir.tensors["pads_nchw"] = TensorIR(
        name="pads_nchw",
        dtype="INT32",
        shape=list(pads_data.shape),
        shape_signature=list(pads_data.shape),
        data=None if boundary == "missing_pads_data" else pads_data,
        is_variable=True,
        quantization={"scale": [1.0], "zero_point": [0]},
        logical_layout="NCHW",
        physical_layout="NCHW",
        onnx_tensor_name="onnx_pads",
    )
    model_ir.tensors["pad_value"] = TensorIR(
        name="pad_value",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.25], dtype=np.float32),
    )
    pad_shape = (
        [1, 2, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else [1, 2, 5, 7]
    )
    if boundary == "invalid_pad_output_rank":
        pad_shape = [1, 2, 5]
    model_ir.tensors["a_pad"] = _tensor("a_pad", pad_shape)
    model_ir.tensors["a_pad"].quantization = QuantParamIR(
        scale=[0.3] * 2,
        zero_point=[0] * 2,
        quantized_dimension=1,
    )
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    concat_index = model_ir.operators.index(concat_op)
    model_ir.operators.insert(
        concat_index,
        OperatorIR(
            "MIRROR_PAD" if boundary == "unsupported_pad" else "PAD",
            ["a_nchw", "pads_nchw", "pad_value"],
            ["a_pad"],
        ),
    )
    concat_op.inputs[0] = "a_pad"

    if boundary == "pad_output_fanout":
        model_ir.tensors["pad_side"] = _tensor(
            "pad_side",
            list(pad_shape),
        )
        model_ir.outputs.append("pad_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["a_pad"], ["pad_side"])
        )
    if boundary == "public_pad_output":
        model_ir.outputs.append("a_pad")
    if boundary == "public_pad_adapter":
        model_ir.outputs.append("a_nchw")
    if shared_adapter:
        model_ir.tensors["pad_adapter_side"] = _tensor(
            "pad_adapter_side",
            [1, 2, 4, 7],
        )
        model_ir.outputs.append("pad_adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["a_nchw"], ["pad_adapter_side"])
        )
    if shared_pads:
        model_ir.tensors["outside_pad"] = _tensor(
            "outside_pad",
            [1, 3, 6, 7],
        )
        model_ir.outputs.append("outside_pad")
        model_ir.operators.append(
            OperatorIR(
                "PAD",
                ["b_nchw", "pads_nchw", "pad_value"],
                ["outside_pad"],
            )
        )
    if public_pads:
        model_ir.outputs.append("pads_nchw")
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


def _assert_quantized_rewritten(
    model_ir: ModelIR,
    *,
    expected_concat_inputs: list[str] | None = None,
) -> None:
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    quantize_op = next(op for op in model_ir.operators if op.op_type == "QUANTIZE")
    assert concat_op.inputs == (
        expected_concat_inputs or ["a_nhwc", "b_nhwc"]
    )
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


@pytest.mark.parametrize(
    "unary_op_type",
    ["RELU", "RELU6", "LOGISTIC", "TANH", "GELU"],
)
def test_nhwc_quantized_unary_input_is_indexed(unary_op_type: str) -> None:
    model_ir = _quantized_model(
        unary_input=True,
        unary_op_type=unary_op_type,
    )
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_quantized_rewritten(
        model_ir,
        expected_concat_inputs=["a_relu", "b_nhwc"],
    )
    unary_op = next(op for op in model_ir.operators if op.outputs == ["a_relu"])
    assert unary_op.op_type == unary_op_type
    assert unary_op.inputs == ["a_nhwc"]
    unary_tensor = model_ir.tensors["a_relu"]
    assert unary_tensor.shape == [1, 5, 7, 2]
    unary_quantization = unary_tensor.quantization
    assert isinstance(unary_quantization, QuantParamIR)
    assert unary_quantization.quantized_dimension == 3
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_quantized_unary"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_unary",
        "unary_output_fanout",
        "public_unary_output",
        "unary_adapter_fanout",
        "public_unary_adapter",
        "invalid_unary_output_rank",
        "unary_spatial_mismatch",
    ],
)
def test_nhwc_quantized_unary_rejects_unsafe_or_partial_match(
    boundary: str,
) -> None:
    model_ir = _quantized_model(unary_input=True, boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_quantized_pad_plus_direct_is_indexed() -> None:
    model_ir = _quantized_pad_model(shared_adapter=True)
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_quantized_rewritten(
        model_ir,
        expected_concat_inputs=["a_pad", "b_nhwc"],
    )
    pad_op = next(op for op in model_ir.operators if op.op_type == "PAD")
    assert pad_op.inputs == ["a_nhwc", "pads_nchw", "pad_value"]
    np.testing.assert_array_equal(
        model_ir.tensors["pads_nchw"].data,
        np.asarray(
            [[0, 0], [0, 1], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )
    pad_tensor = model_ir.tensors["a_pad"]
    assert pad_tensor.shape == [1, 5, 7, 2]
    assert isinstance(pad_tensor.quantization, QuantParamIR)
    assert pad_tensor.quantization.quantized_dimension == 3
    remaining_transposes = [
        op for op in model_ir.operators if op.op_type == "TRANSPOSE"
    ]
    assert [op.outputs for op in remaining_transposes] == [["a_nchw"]]
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_quantized_pad"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


@pytest.mark.parametrize("sharing", ["consumer", "public"])
def test_nhwc_quantized_pad_clones_shared_or_public_pads(
    sharing: str,
) -> None:
    model_ir = _quantized_pad_model(
        shared_pads=sharing == "consumer",
        public_pads=sharing == "public",
    )
    original_pads = np.array(model_ir.tensors["pads_nchw"].data, copy=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    rewritten_pad = next(
        op for op in model_ir.operators if op.outputs == ["a_pad"]
    )
    rewritten_pads_name = rewritten_pad.inputs[1]
    assert rewritten_pads_name != "pads_nchw"
    np.testing.assert_array_equal(
        model_ir.tensors["pads_nchw"].data,
        original_pads,
    )
    rewritten_pads = model_ir.tensors[rewritten_pads_name]
    np.testing.assert_array_equal(
        rewritten_pads.data,
        np.asarray(
            [[0, 0], [0, 1], [0, 0], [0, 0]],
            dtype=np.int32,
        ),
    )
    assert rewritten_pads.is_variable
    assert rewritten_pads.quantization == {
        "scale": [1.0],
        "zero_point": [0],
    }
    assert rewritten_pads.logical_layout == "NCHW"
    assert rewritten_pads.physical_layout == "NCHW"
    assert rewritten_pads.onnx_tensor_name == "onnx_pads"
    if sharing == "consumer":
        outside_pad = next(
            op for op in model_ir.operators if op.outputs == ["outside_pad"]
        )
        assert outside_pad.inputs[1] == "pads_nchw"


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_pad",
        "pad_output_fanout",
        "public_pad_output",
        "public_pad_adapter",
        "missing_pads_data",
        "invalid_pads_shape",
        "invalid_pad_output_rank",
        "spatial_shape_mismatch",
    ],
)
def test_nhwc_quantized_pad_rejects_unsafe_or_partial_match(
    boundary: str,
) -> None:
    model_ir = _quantized_pad_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)
