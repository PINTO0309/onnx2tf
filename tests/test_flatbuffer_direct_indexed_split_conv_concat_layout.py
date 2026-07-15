from __future__ import annotations

import copy
import pickle
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import (
    optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains,
)


_N = 1
_H = 3
_W = 5
_SOURCE_CHANNELS = 8
_SPLIT_CHANNELS = 4
_CONV_CHANNELS = 3
_OUTPUT_CHANNELS = _SPLIT_CHANNELS + _CONV_CHANNELS


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    dtype: str = "FLOAT32",
    is_variable: bool = False,
    quantization: QuantParamIR | None = None,
    layout: str = LOGICAL_LAYOUT_UNKNOWN,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=is_variable,
        quantization=quantization,
        logical_layout=layout,
        physical_layout=layout,
    )


def _make_model(
    *,
    dynamic: bool = False,
    integer_dtype: str = "INT32",
    negative_axis: bool = False,
    branch_output_index: int = 1,
    branch_first_in_concat: bool = True,
    extra_post_consumer: bool = False,
    shared_axis: bool = False,
) -> ModelIR:
    numpy_integer_dtype = np.int64 if integer_dtype == "INT64" else np.int32
    source_nhwc = [_N, _H, _W, _SOURCE_CHANNELS]
    source_nchw = [_N, _SOURCE_CHANNELS, _H, _W]
    split_nhwc = [_N, _H, _W, _SPLIT_CHANNELS]
    split_nchw = [_N, _SPLIT_CHANNELS, _H, _W]
    conv_nhwc = [_N, _H, _W, _CONV_CHANNELS]
    conv_nchw = [_N, _CONV_CHANNELS, _H, _W]
    output_nhwc = [_N, _H, _W, _OUTPUT_CHANNELS]
    output_nchw = [_N, _OUTPUT_CHANNELS, _H, _W]
    source_nhwc_signature = (
        [_N, -1, -1, _SOURCE_CHANNELS]
        if dynamic
        else list(source_nhwc)
    )
    source_nchw_signature = (
        [_N, _SOURCE_CHANNELS, -1, -1]
        if dynamic
        else list(source_nchw)
    )
    split_nhwc_signature = (
        [_N, -1, -1, _SPLIT_CHANNELS]
        if dynamic
        else list(split_nhwc)
    )
    split_nchw_signature = (
        [_N, _SPLIT_CHANNELS, -1, -1]
        if dynamic
        else list(split_nchw)
    )
    conv_nhwc_signature = (
        [_N, -1, -1, _CONV_CHANNELS]
        if dynamic
        else list(conv_nhwc)
    )
    conv_nchw_signature = (
        [_N, _CONV_CHANNELS, -1, -1]
        if dynamic
        else list(conv_nchw)
    )
    output_nhwc_signature = (
        [_N, -1, -1, _OUTPUT_CHANNELS]
        if dynamic
        else list(output_nhwc)
    )
    output_nchw_signature = (
        [_N, _OUTPUT_CHANNELS, -1, -1]
        if dynamic
        else list(output_nchw)
    )
    filter_data = (
        np.arange(_CONV_CHANNELS * _SPLIT_CHANNELS, dtype=np.float32)
        .reshape(_CONV_CHANNELS, 1, 1, _SPLIT_CHANNELS)
        / 17.0
    )
    bias_data = np.asarray([0.25, -0.125, 0.5], dtype=np.float32)

    model_ir = ModelIR("indexed_split_conv_concat")
    model_ir.inputs = ["src_nhwc"]
    model_ir.outputs = ["sink"]
    model_ir.tensors = {
        "src_nhwc": _tensor(
            "src_nhwc",
            source_nhwc,
            signature=source_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=numpy_integer_dtype),
            dtype=integer_dtype,
        ),
        "to_nhwc": _tensor(
            "to_nhwc",
            [4],
            data=np.asarray([0, 2, 3, 1], dtype=numpy_integer_dtype),
            dtype=integer_dtype,
        ),
        "axis": _tensor(
            "axis",
            [1],
            data=np.asarray(
                [-3 if negative_axis else 1],
                dtype=numpy_integer_dtype,
            ),
            dtype=integer_dtype,
        ),
        "pre_nchw": _tensor(
            "pre_nchw",
            source_nchw,
            signature=source_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "relu0_nchw": _tensor(
            "relu0_nchw",
            source_nchw,
            signature=source_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "keep_nchw": _tensor(
            "keep_nchw",
            split_nchw,
            signature=split_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "branch_nchw": _tensor(
            "branch_nchw",
            split_nchw,
            signature=split_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "branch_nhwc": _tensor(
            "branch_nhwc",
            split_nhwc,
            signature=split_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "filter": _tensor(
            "filter",
            list(filter_data.shape),
            data=filter_data,
        ),
        "bias": _tensor("bias", [_CONV_CHANNELS], data=bias_data),
        "conv_nhwc": _tensor(
            "conv_nhwc",
            conv_nhwc,
            signature=conv_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "conv_nchw": _tensor(
            "conv_nchw",
            conv_nchw,
            signature=conv_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "relu1_nchw": _tensor(
            "relu1_nchw",
            conv_nchw,
            signature=conv_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "concat_nchw": _tensor(
            "concat_nchw",
            output_nchw,
            signature=output_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "post_nhwc": _tensor(
            "post_nhwc",
            output_nhwc,
            signature=output_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        "sink": _tensor(
            "sink",
            output_nhwc,
            signature=output_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
    }
    split_outputs = ["keep_nchw", "branch_nchw"]
    if int(branch_output_index) == 0:
        split_outputs.reverse()
    concat_inputs = ["relu1_nchw", "keep_nchw"]
    if not branch_first_in_concat:
        concat_inputs.reverse()
    model_ir.operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["src_nhwc", "to_nchw"],
            outputs=["pre_nchw"],
        ),
        OperatorIR(op_type="RELU", inputs=["pre_nchw"], outputs=["relu0_nchw"]),
        OperatorIR(
            op_type="SPLIT",
            inputs=["axis", "relu0_nchw"],
            outputs=split_outputs,
            options={"numSplits": 2},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["branch_nchw", "to_nhwc"],
            outputs=["branch_nhwc"],
        ),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["branch_nhwc", "filter", "bias"],
            outputs=["conv_nhwc"],
            options={
                "padding": "VALID",
                "strideH": 1,
                "strideW": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["conv_nhwc", "to_nchw"],
            outputs=["conv_nchw"],
        ),
        OperatorIR(op_type="RELU", inputs=["conv_nchw"], outputs=["relu1_nchw"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=concat_inputs,
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["concat_nchw", "to_nhwc"],
            outputs=["post_nhwc"],
        ),
        OperatorIR(op_type="RELU", inputs=["post_nhwc"], outputs=["sink"]),
    ]
    if extra_post_consumer:
        model_ir.outputs.append("extra")
        model_ir.tensors["extra"] = _tensor(
            "extra",
            output_nhwc,
            signature=output_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        model_ir.operators.append(
            OperatorIR(op_type="RELU6", inputs=["post_nhwc"], outputs=["extra"])
        )
    if shared_axis:
        model_ir.inputs.append("legacy_value")
        model_ir.outputs.append("legacy_keep")
        model_ir.tensors["legacy_value"] = _tensor("legacy_value", [1])
        model_ir.tensors["legacy_keep"] = _tensor("legacy_keep", [1])
        model_ir.operators.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=["legacy_value", "axis"],
                outputs=["legacy_keep"],
            )
        )
    return model_ir


def _snapshot(model_ir: ModelIR) -> bytes:
    return pickle.dumps(model_ir, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("branch_output_index", [0, 1])
@pytest.mark.parametrize("branch_first_in_concat", [False, True])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("integer_dtype", "negative_axis"),
    [("INT32", False), ("INT64", True)],
    ids=["int32_positive", "int64_negative"],
)
def test_indexed_split_conv_concat_rewrites_exact_two_branch_island(
    branch_output_index: int,
    branch_first_in_concat: bool,
    dynamic: bool,
    integer_dtype: str,
    negative_axis: bool,
) -> None:
    model_ir = _make_model(
        dynamic=dynamic,
        integer_dtype=integer_dtype,
        negative_axis=negative_axis,
        branch_output_index=branch_output_index,
        branch_first_in_concat=branch_first_in_concat,
    )
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains": 1
    }
    assert "TRANSPOSE" not in {
        str(operator.op_type) for operator in model_ir.operators
    }
    split = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "SPLIT"
    )
    conv = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONV_2D"
    )
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    assert list(model_ir.operators[0].inputs) == ["src_nhwc"]
    assert list(split.inputs) == ["axis", "relu0_nchw"]
    assert list(conv.inputs) == ["branch_nchw", "filter", "bias"]
    post_relu = next(
        operator
        for operator in model_ir.operators
        if list(operator.outputs) == ["relu1_nchw"]
    )
    assert list(post_relu.inputs) == ["conv_nhwc"]
    assert int(concat.options["axis"]) == 3
    expected_concat_inputs = ["relu1_nchw", "keep_nchw"]
    if not branch_first_in_concat:
        expected_concat_inputs.reverse()
    assert list(concat.inputs) == expected_concat_inputs
    assert list(model_ir.operators[-1].inputs) == ["concat_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([3], dtype=np.int64 if integer_dtype == "INT64" else np.int32),
    )
    expected_spatial_signature = [-1, -1] if dynamic else [_H, _W]
    assert model_ir.tensors["relu0_nchw"].shape == [
        _N,
        _H,
        _W,
        _SOURCE_CHANNELS,
    ]
    assert model_ir.tensors["branch_nchw"].shape == [
        _N,
        _H,
        _W,
        _SPLIT_CHANNELS,
    ]
    assert model_ir.tensors["keep_nchw"].shape_signature == [
        _N,
        *expected_spatial_signature,
        _SPLIT_CHANNELS,
    ]
    assert model_ir.tensors["relu1_nchw"].shape == [
        _N,
        _H,
        _W,
        _CONV_CHANNELS,
    ]
    assert model_ir.tensors["concat_nchw"].shape == [
        _N,
        _H,
        _W,
        _OUTPUT_CHANNELS,
    ]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_split_conv_concat_preserves_numerical_semantics() -> None:
    model_ir = _make_model(
        branch_output_index=0,
        branch_first_in_concat=False,
    )
    rng = np.random.default_rng(71)
    source = rng.normal(size=(_N, _H, _W, _SOURCE_CHANNELS)).astype(np.float32)
    relu_nchw = np.maximum(np.transpose(source, (0, 3, 1, 2)), 0.0)
    split_values = np.split(relu_nchw, 2, axis=1)
    branch_nhwc = np.transpose(split_values[0], (0, 2, 3, 1))
    keep_nchw = split_values[1]
    filter_data = np.asarray(model_ir.tensors["filter"].data)
    weights = filter_data[:, 0, 0, :]
    bias = np.asarray(model_ir.tensors["bias"].data)
    conv_nhwc = np.einsum("nhwi,oi->nhwo", branch_nhwc, weights) + bias
    relu1_nchw = np.maximum(np.transpose(conv_nhwc, (0, 3, 1, 2)), 0.0)
    expected = np.transpose(
        np.concatenate([keep_nchw, relu1_nchw], axis=1),
        (0, 2, 3, 1),
    )

    stats = optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
    ] == 1
    new_split_values = np.split(np.maximum(source, 0.0), 2, axis=3)
    new_conv = np.einsum("nhwi,oi->nhwo", new_split_values[0], weights) + bias
    actual = np.concatenate(
        [new_split_values[1], np.maximum(new_conv, 0.0)],
        axis=3,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_indexed_split_conv_concat_rewires_every_post_consumer() -> None:
    model_ir = _make_model(extra_post_consumer=True)

    stats = optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir
    )

    assert stats[
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
    ] == 1
    output_consumers = [
        operator
        for operator in model_ir.operators
        if list(operator.outputs) in (["sink"], ["extra"])
    ]
    assert [list(operator.inputs) for operator in output_consumers] == [
        ["concat_nchw"],
        ["concat_nchw"],
    ]


def test_indexed_split_conv_concat_clones_shared_int64_axis() -> None:
    model_ir = _make_model(integer_dtype="INT64", shared_axis=True)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats[
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains"
    ] == 1
    np.testing.assert_array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([1], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        model_ir.tensors["axis_nhwc"].data,
        np.asarray([3], dtype=np.int64),
    )
    split = next(
        operator for operator in model_ir.operators if str(operator.op_type) == "SPLIT"
    )
    reshape = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "RESHAPE"
    )
    assert list(split.inputs) == ["axis_nhwc", "relu0_nchw"]
    assert list(reshape.inputs) == ["legacy_value", "axis"]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_split_conv_concat_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    concat = model_ir.operators[7]
    index = ModelIRGraphIndex(model_ir)

    assert optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir,
        graph_index=index,
        candidate=concat,
        max_rewrites=0,
    ) == {
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains": 0
    }
    assert optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir,
        graph_index=index,
        candidate=concat,
    ) == {
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains": 1
    }
    assert optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir,
        graph_index=index,
    ) == {
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains": 0
    }


UnsafeMutation = Callable[[ModelIR], None]


def _wrong_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nchw"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _wrong_prebranch_permutation(model_ir: ModelIR) -> None:
    model_ir.operators[3].inputs[1] = "to_nchw"


def _wrong_postbranch_permutation(model_ir: ModelIR) -> None:
    model_ir.operators[5].inputs[1] = "to_nhwc"


def _wrong_postconcat_permutation(model_ir: ModelIR) -> None:
    model_ir.operators[8].inputs[1] = "to_nchw"


def _unresolved_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("src_nhwc")


def _pre_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_fanout"] = _tensor(
        "pre_fanout",
        [_N, _SOURCE_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["pre_nchw"], outputs=["pre_fanout"])
    )


def _prerelu_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["prerelu_fanout"] = _tensor(
        "prerelu_fanout",
        [_N, _SOURCE_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["relu0_nchw"],
            outputs=["prerelu_fanout"],
        )
    )


def _split_branch_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["split_fanout"] = _tensor(
        "split_fanout",
        [_N, _SPLIT_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["branch_nchw"],
            outputs=["split_fanout"],
        )
    )


def _keep_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["keep_fanout"] = _tensor(
        "keep_fanout",
        [_N, _SPLIT_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["keep_nchw"], outputs=["keep_fanout"])
    )


def _third_split_output(model_ir: ModelIR) -> None:
    model_ir.tensors["third"] = _tensor(
        "third",
        [_N, _SPLIT_CHANNELS, _H, _W],
    )
    model_ir.operators[2].outputs.append("third")


def _unequal_split(model_ir: ModelIR) -> None:
    model_ir.tensors["keep_nchw"].shape[1] -= 1


def _wrong_num_splits(model_ir: ModelIR) -> None:
    model_ir.operators[2].options["numSplits"] = 3


def _wrong_split_axis(model_ir: ModelIR) -> None:
    model_ir.tensors["axis"].data = np.asarray([2], dtype=np.int32)


def _variable_split_axis(model_ir: ModelIR) -> None:
    model_ir.tensors["axis"].is_variable = True


def _conv_input_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["conv_input_fanout"] = _tensor(
        "conv_input_fanout",
        [_N, _H, _W, _SPLIT_CHANNELS],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["branch_nhwc"],
            outputs=["conv_input_fanout"],
        )
    )


def _conv_output_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["conv_output_fanout"] = _tensor(
        "conv_output_fanout",
        [_N, _H, _W, _CONV_CHANNELS],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["conv_nhwc"],
            outputs=["conv_output_fanout"],
        )
    )


def _postbranch_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["postbranch_fanout"] = _tensor(
        "postbranch_fanout",
        [_N, _CONV_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["conv_nchw"],
            outputs=["postbranch_fanout"],
        )
    )


def _postrelu_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["postrelu_fanout"] = _tensor(
        "postrelu_fanout",
        [_N, _CONV_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["relu1_nchw"],
            outputs=["postrelu_fanout"],
        )
    )


def _concat_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["concat_fanout"] = _tensor(
        "concat_fanout",
        [_N, _OUTPUT_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["concat_nchw"],
            outputs=["concat_fanout"],
        )
    )


def _public_intermediate(model_ir: ModelIR) -> None:
    model_ir.outputs.append("concat_nchw")


def _public_post_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("post_nhwc")


def _missing_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["conv_nchw"]


def _dtype_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_nhwc"].dtype = "FLOAT16"


def _quantization_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_nhwc"].quantization = QuantParamIR(
        scale=[0.25],
        zero_point=[0],
        quantized_dimension=0,
    )


def _per_axis_activation(model_ir: ModelIR) -> None:
    model_ir.tensors["keep_nchw"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _contradictory_layout(model_ir: ModelIR) -> None:
    model_ir.tensors["branch_nhwc"].physical_layout = LOGICAL_LAYOUT_NCHW


def _consumer_before_post(model_ir: ModelIR) -> None:
    consumer = model_ir.operators.pop(9)
    model_ir.operators.insert(8, consumer)


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["src_nhwc"], outputs=["pre_nchw"]),
    )


def _unresolved_filter(model_ir: ModelIR) -> None:
    model_ir.tensors["filter"].data = None


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_pre_permutation,
        _wrong_prebranch_permutation,
        _wrong_postbranch_permutation,
        _wrong_postconcat_permutation,
        _unresolved_source,
        _pre_fanout,
        _prerelu_fanout,
        _split_branch_fanout,
        _keep_fanout,
        _third_split_output,
        _unequal_split,
        _wrong_num_splits,
        _wrong_split_axis,
        _variable_split_axis,
        _conv_input_fanout,
        _conv_output_fanout,
        _postbranch_fanout,
        _postrelu_fanout,
        _concat_fanout,
        _public_intermediate,
        _public_post_output,
        _missing_tensor,
        _dtype_mismatch,
        _quantization_mismatch,
        _per_axis_activation,
        _contradictory_layout,
        _consumer_before_post,
        _duplicate_producer,
        _unresolved_filter,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_split_conv_concat_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_relu_split_conv_relu_concat_posttranspose_to_nhwc_chains": 0
    }
    assert _snapshot(model_ir) == before
    assert layout_state.validate_against_model_ir(model_ir) == []
