from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_concat_nhwc_chains,
    _optimize_transpose_pre_concat_nhwc_chains_legacy,
)
from onnx2tf.tflite_builder.passes.nhwc_concat_legacy_layout import (
    optimize_transpose_pre_concat_nhwc_chains_legacy,
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


def _leaky_model(
    *,
    alpha_first: bool = False,
    unary_companion: bool = False,
    all_leaky: bool = False,
    boundary: str | None = None,
) -> ModelIR:
    model_ir = ModelIR("nhwc_pseudo_leaky_pre_concat")
    model_ir.inputs = ["x_nhwc", "leaky_nhwc"]
    model_ir.outputs = ["y"]
    source_shape = (
        [1, 5, 3]
        if boundary == "invalid_source_rank"
        else [1, 5, 7, 3]
    )
    internal_shape = [1, 3, 5, 7]
    neg_shape = (
        [1, 3, 5]
        if boundary == "invalid_internal_rank"
        else list(internal_shape)
    )
    leaky_shape = (
        [1, 3, 9, 7]
        if boundary == "spatial_shape_mismatch"
        else list(internal_shape)
    )
    alpha_data = (
        np.asarray([0.1, 0.2], dtype=np.float32)
        if boundary == "non_singleton_alpha"
        else np.asarray([0.1], dtype=np.float32)
    )
    output_channels = 3 if all_leaky else 5
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 5, 7, 2]),
        "leaky_nhwc": _tensor("leaky_nhwc", source_shape),
        "pre_perm": _int_tensor("pre_perm", [0, 3, 1, 2]),
        "post_perm": _int_tensor(
            "post_perm",
            [0, 3, 1, 2]
            if boundary == "wrong_post_permutation"
            else [0, 2, 3, 1],
        ),
        "x_nchw": _tensor("x_nchw", [1, 2, 5, 7]),
        "leaky_nchw": _tensor("leaky_nchw", list(internal_shape)),
        "neg_out": _tensor("neg_out", neg_shape),
        "neg_relu": _tensor("neg_relu", list(internal_shape)),
        "pos_relu": _tensor("pos_relu", list(internal_shape)),
        "alpha": TensorIR(
            name="alpha",
            dtype="FLOAT32",
            shape=list(alpha_data.shape),
            shape_signature=list(alpha_data.shape),
            data=alpha_data,
        ),
        "neg_scaled": _tensor("neg_scaled", list(internal_shape)),
        "leaky_out": _tensor("leaky_out", leaky_shape),
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
    for tensor_name in (
        "neg_out",
        "neg_relu",
        "pos_relu",
        "neg_scaled",
        "leaky_out",
    ):
        model_ir.tensors[tensor_name].quantization = QuantParamIR(
            scale=[0.25] * 3,
            zero_point=[0] * 3,
            quantized_dimension=1,
        )
    if boundary == "wrong_pre_permutation":
        model_ir.tensors["bad_pre_perm"] = _int_tensor(
            "bad_pre_perm",
            [0, 2, 3, 1],
        )

    mul_inputs = (
        ["alpha", "neg_relu"]
        if alpha_first
        else ["neg_relu", "alpha"]
    )
    sub_inputs = (
        ["neg_scaled", "pos_relu"]
        if boundary == "reversed_sub_inputs"
        else ["pos_relu", "neg_scaled"]
    )
    companion_name = "x_nchw"
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "pre_perm"], ["x_nchw"]),
        OperatorIR(
            "TRANSPOSE",
            [
                "leaky_nhwc",
                "bad_pre_perm" if boundary == "wrong_pre_permutation" else "pre_perm",
            ],
            ["leaky_nchw"],
        ),
        OperatorIR(
            "ABS" if boundary == "unsupported_neg" else "NEG",
            ["leaky_nchw"],
            ["neg_out"],
        ),
        OperatorIR(
            "TANH" if boundary == "unsupported_negative_relu" else "RELU",
            ["neg_out"],
            ["neg_relu"],
        ),
        OperatorIR(
            "TANH" if boundary == "unsupported_positive_relu" else "RELU",
            ["leaky_nchw"],
            ["pos_relu"],
        ),
        OperatorIR("MUL", mul_inputs, ["neg_scaled"]),
        OperatorIR(
            "ADD" if boundary == "unsupported_sub" else "SUB",
            sub_inputs,
            ["leaky_out"],
        ),
        OperatorIR(
            "CONCATENATION",
            ["leaky_out"] if all_leaky else [companion_name, "leaky_out"],
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
    if all_leaky:
        model_ir.inputs = ["leaky_nhwc"]
        model_ir.operators = [
            op for op in model_ir.operators if op.outputs != ["x_nchw"]
        ]
        del model_ir.tensors["x_nhwc"]
        del model_ir.tensors["x_nchw"]
    elif unary_companion:
        model_ir.tensors["x_relu"] = _tensor("x_relu", [1, 2, 5, 7])
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_index = model_ir.operators.index(concat_op)
        model_ir.operators.insert(
            concat_index,
            OperatorIR("RELU", ["x_nchw"], ["x_relu"]),
        )
        concat_op.inputs[0] = "x_relu"

    if boundary == "pad_companion":
        model_ir.tensors["pads"] = TensorIR(
            name="pads",
            dtype="INT32",
            shape=[4, 2],
            shape_signature=[4, 2],
            data=np.zeros((4, 2), dtype=np.int32),
        )
        model_ir.tensors["x_pad"] = _tensor("x_pad", [1, 2, 5, 7])
        concat_op = next(
            op for op in model_ir.operators if op.op_type == "CONCATENATION"
        )
        concat_index = model_ir.operators.index(concat_op)
        model_ir.operators.insert(
            concat_index,
            OperatorIR("PAD", ["x_nchw", "pads"], ["x_pad"]),
        )
        concat_op.inputs[0] = "x_pad"
    if boundary == "leaky_output_fanout":
        model_ir.tensors["leaky_side"] = _tensor(
            "leaky_side",
            list(leaky_shape),
        )
        model_ir.outputs.append("leaky_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["leaky_out"], ["leaky_side"])
        )
    if boundary == "public_leaky_output":
        model_ir.outputs.append("leaky_out")
    if boundary == "internal_fanout":
        model_ir.tensors["internal_side"] = _tensor(
            "internal_side",
            list(internal_shape),
        )
        model_ir.outputs.append("internal_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["neg_out"], ["internal_side"])
        )
    if boundary == "public_internal":
        model_ir.outputs.append("neg_scaled")
    if boundary == "adapter_fanout":
        model_ir.tensors["adapter_side"] = _tensor(
            "adapter_side",
            list(internal_shape),
        )
        model_ir.outputs.append("adapter_side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["leaky_nchw"], ["adapter_side"])
        )
    if boundary == "public_adapter":
        model_ir.outputs.append("leaky_nchw")
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
        concat_op.inputs = ["residual_nchw", "leaky_out"]
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


def _assert_leaky_rewritten(model_ir: ModelIR) -> None:
    neg_op = next(op for op in model_ir.operators if op.op_type == "NEG")
    pos_relu_op = next(
        op
        for op in model_ir.operators
        if op.op_type == "RELU" and op.outputs == ["pos_relu"]
    )
    assert neg_op.inputs == ["leaky_nhwc"]
    assert pos_relu_op.inputs == ["leaky_nhwc"]
    for tensor_name in (
        "neg_out",
        "neg_relu",
        "pos_relu",
        "neg_scaled",
        "leaky_out",
    ):
        assert model_ir.tensors[tensor_name].shape == [1, 5, 7, 3]
        quantization = model_ir.tensors[tensor_name].quantization
        assert isinstance(quantization, QuantParamIR)
        assert quantization.quantized_dimension == 3
    concat_op = next(
        op for op in model_ir.operators if op.op_type == "CONCATENATION"
    )
    assert concat_op.inputs[-1] == "leaky_out"
    assert concat_op.outputs == ["concat_nhwc"]
    assert concat_op.options["axis"] == 3


@pytest.mark.parametrize("alpha_first", [False, True])
@pytest.mark.parametrize("unary_companion", [False, True])
def test_nhwc_pseudo_leaky_is_indexed(
    alpha_first: bool,
    unary_companion: bool,
) -> None:
    model_ir = _leaky_model(
        alpha_first=alpha_first,
        unary_companion=unary_companion,
    )
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_leaky_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    event = next(
        event
        for event in diagnostics
        if event["code"] == "layout.nhwc_pre_concat_leaky"
    )
    assert event["status"] == "changed"
    assert event["metrics"]["snapshot_count"] == 1
    assert event["metrics"]["fingerprint_count"] == 0


def test_nhwc_all_pseudo_leaky_inputs_use_indexed_family() -> None:
    model_ir = _leaky_model(all_leaky=True)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    _assert_leaky_rewritten(model_ir)
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)


@pytest.mark.parametrize(
    "boundary",
    [
        "unsupported_sub",
        "reversed_sub_inputs",
        "non_singleton_alpha",
        "unsupported_neg",
        "unsupported_negative_relu",
        "unsupported_positive_relu",
        "leaky_output_fanout",
        "public_leaky_output",
        "internal_fanout",
        "public_internal",
        "adapter_fanout",
        "public_adapter",
        "invalid_source_rank",
        "invalid_internal_rank",
        "spatial_shape_mismatch",
        "raw_residual_input",
        "wrong_pre_permutation",
        "wrong_post_permutation",
        "public_concat",
        "public_post",
    ],
)
def test_nhwc_pseudo_leaky_rejects_unsafe_or_partial_match(
    boundary: str,
) -> None:
    model_ir = _leaky_model(boundary=boundary)
    original = deepcopy(model_ir)

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 0}
    _assert_model_equal(model_ir, original)


def test_nhwc_pseudo_leaky_pad_companion_remains_in_legacy() -> None:
    model_ir = _leaky_model(boundary="pad_companion")
    diagnostics: list[dict] = []

    stats = _optimize_transpose_pre_concat_nhwc_chains(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {"optimized_transpose_pre_concat_nhwc_chains": 1}
    assert not any(
        event["code"] == "layout.nhwc_pre_concat_leaky"
        and event["status"] == "changed"
        for event in diagnostics
    )


def test_nhwc_legacy_owner_matches_lowerer_compatibility_wrapper() -> None:
    direct_model_ir = _leaky_model(boundary="pad_companion")
    wrapper_model_ir = deepcopy(direct_model_ir)

    direct_stats = optimize_transpose_pre_concat_nhwc_chains_legacy(
        direct_model_ir
    )
    wrapper_stats = _optimize_transpose_pre_concat_nhwc_chains_legacy(
        wrapper_model_ir
    )

    assert direct_stats == {
        "optimized_transpose_pre_concat_nhwc_chains": 1,
    }
    assert wrapper_stats == direct_stats
    _assert_model_equal(wrapper_model_ir, direct_model_ir)
    assert optimize_transpose_pre_concat_nhwc_chains_legacy(
        direct_model_ir
    ) == {
        "optimized_transpose_pre_concat_nhwc_chains": 0,
    }
