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
    optimize_transpose_relu_split_all_outputs_to_nhwc_chains,
)


_N = 1
_H = 3
_W = 5
_BRANCH_CHANNELS = 2


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
    branches: int = 2,
    dynamic: bool = False,
    integer_dtype: str = "INT32",
    negative_axis: bool = False,
    combined_consumer: bool = False,
    extra_post_consumer: bool = False,
    shared_axis: bool = False,
) -> ModelIR:
    channels = int(branches) * _BRANCH_CHANNELS
    nhwc = [_N, _H, _W, channels]
    nchw = [_N, channels, _H, _W]
    branch_nhwc = [_N, _H, _W, _BRANCH_CHANNELS]
    branch_nchw = [_N, _BRANCH_CHANNELS, _H, _W]
    nhwc_signature = [_N, -1, -1, channels] if dynamic else list(nhwc)
    nchw_signature = [_N, channels, -1, -1] if dynamic else list(nchw)
    branch_nhwc_signature = (
        [_N, -1, -1, _BRANCH_CHANNELS]
        if dynamic
        else list(branch_nhwc)
    )
    branch_nchw_signature = (
        [_N, _BRANCH_CHANNELS, -1, -1]
        if dynamic
        else list(branch_nchw)
    )
    numpy_integer_dtype = np.int64 if integer_dtype == "INT64" else np.int32

    model_ir = ModelIR("indexed_split_all_outputs")
    model_ir.inputs = ["src_nhwc"]
    model_ir.tensors = {
        "src_nhwc": _tensor(
            "src_nhwc",
            nhwc,
            signature=nhwc_signature,
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
            nchw,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        "relu_nchw": _tensor(
            "relu_nchw",
            nchw,
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
    }
    split_outputs = [f"split_{index}_nchw" for index in range(branches)]
    operators = [
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["src_nhwc", "to_nchw"],
            outputs=["pre_nchw"],
        ),
        OperatorIR(op_type="RELU", inputs=["pre_nchw"], outputs=["relu_nchw"]),
        OperatorIR(
            op_type="SPLIT",
            inputs=["axis", "relu_nchw"],
            outputs=split_outputs,
            options={"numSplits": int(branches)},
        ),
    ]
    post_outputs = []
    for index, split_output in enumerate(split_outputs):
        post_output = f"post_{index}_nhwc"
        post_outputs.append(post_output)
        model_ir.tensors[split_output] = _tensor(
            split_output,
            branch_nchw,
            signature=branch_nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        )
        model_ir.tensors[post_output] = _tensor(
            post_output,
            branch_nhwc,
            signature=branch_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=[split_output, "to_nhwc"],
                outputs=[post_output],
            )
        )

    if combined_consumer:
        model_ir.outputs = ["joined"]
        model_ir.tensors["joined"] = _tensor(
            "joined",
            nhwc,
            signature=nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=list(post_outputs),
                outputs=["joined"],
                options={"axis": 3},
            )
        )
    else:
        model_ir.outputs = [f"out_{index}" for index in range(branches)]
        for index, post_output in enumerate(post_outputs):
            output_name = f"out_{index}"
            model_ir.tensors[output_name] = _tensor(
                output_name,
                branch_nhwc,
                signature=branch_nhwc_signature,
                layout=LOGICAL_LAYOUT_NHWC,
            )
            operators.append(
                OperatorIR(
                    op_type="RELU",
                    inputs=[post_output],
                    outputs=[output_name],
                )
            )
    if extra_post_consumer:
        model_ir.outputs.append("extra")
        model_ir.tensors["extra"] = _tensor(
            "extra",
            branch_nhwc,
            signature=branch_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(
            OperatorIR(
                op_type="RELU6",
                inputs=[post_outputs[0]],
                outputs=["extra"],
            )
        )
    if shared_axis:
        model_ir.inputs.append("legacy_value")
        model_ir.outputs.append("legacy_keep")
        model_ir.tensors["legacy_value"] = _tensor("legacy_value", [1])
        model_ir.tensors["legacy_keep"] = _tensor("legacy_keep", [1])
        operators.append(
            OperatorIR(
                op_type="RESHAPE",
                inputs=["legacy_value", "axis"],
                outputs=["legacy_keep"],
            )
        )
    model_ir.operators = operators
    return model_ir


def _snapshot(model_ir: ModelIR) -> bytes:
    return pickle.dumps(model_ir, protocol=pickle.HIGHEST_PROTOCOL)


@pytest.mark.parametrize("branches", [2, 3])
@pytest.mark.parametrize("dynamic", [False, True], ids=["static", "dynamic"])
@pytest.mark.parametrize(
    ("integer_dtype", "negative_axis"),
    [("INT32", False), ("INT64", True)],
    ids=["int32_positive", "int64_negative"],
)
def test_indexed_split_all_outputs_rewrites_exact_closed_island(
    branches: int,
    dynamic: bool,
    integer_dtype: str,
    negative_axis: bool,
) -> None:
    model_ir = _make_model(
        branches=branches,
        dynamic=dynamic,
        integer_dtype=integer_dtype,
        negative_axis=negative_axis,
    )
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_relu_split_all_outputs_to_nhwc_chains": 1
    }
    assert "TRANSPOSE" not in {
        str(operator.op_type) for operator in model_ir.operators
    }
    relu = model_ir.operators[0]
    split = model_ir.operators[1]
    assert list(relu.inputs) == ["src_nhwc"]
    assert list(split.inputs) == ["axis", "relu_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors["axis"].data,
        np.asarray([3], dtype=np.int64 if integer_dtype == "INT64" else np.int32),
    )
    channels = branches * _BRANCH_CHANNELS
    expected_relu_signature = (
        [_N, -1, -1, channels]
        if dynamic
        else [_N, _H, _W, channels]
    )
    assert model_ir.tensors["relu_nchw"].shape == [_N, _H, _W, channels]
    assert model_ir.tensors["relu_nchw"].shape_signature == expected_relu_signature
    for branch in range(branches):
        name = f"split_{branch}_nchw"
        expected_signature = (
            [_N, -1, -1, _BRANCH_CHANNELS]
            if dynamic
            else [_N, _H, _W, _BRANCH_CHANNELS]
        )
        assert model_ir.tensors[name].shape == [
            _N,
            _H,
            _W,
            _BRANCH_CHANNELS,
        ]
        assert model_ir.tensors[name].shape_signature == expected_signature
        consumer = next(
            operator
            for operator in model_ir.operators
            if list(operator.outputs) == [f"out_{branch}"]
        )
        assert list(consumer.inputs) == [name]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_split_all_outputs_preserves_numerical_semantics() -> None:
    model_ir = _make_model(branches=3, combined_consumer=True)
    rng = np.random.default_rng(53)
    source = rng.normal(size=(_N, _H, _W, 6)).astype(np.float32)
    old_relu = np.maximum(np.transpose(source, (0, 3, 1, 2)), 0.0)
    expected = np.concatenate(
        [
            np.transpose(value, (0, 2, 3, 1))
            for value in np.split(old_relu, 3, axis=1)
        ],
        axis=3,
    )

    stats = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(model_ir)

    assert stats[
        "optimized_transpose_relu_split_all_outputs_to_nhwc_chains"
    ] == 1
    split_values = np.split(np.maximum(source, 0.0), 3, axis=3)
    actual = np.concatenate(split_values, axis=3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    concat = next(
        operator
        for operator in model_ir.operators
        if str(operator.op_type) == "CONCATENATION"
    )
    assert list(concat.inputs) == [
        "split_0_nchw",
        "split_1_nchw",
        "split_2_nchw",
    ]


def test_indexed_split_all_outputs_rewires_every_post_consumer() -> None:
    model_ir = _make_model(extra_post_consumer=True)

    stats = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(model_ir)

    assert stats[
        "optimized_transpose_relu_split_all_outputs_to_nhwc_chains"
    ] == 1
    consumers = [
        operator
        for operator in model_ir.operators
        if str(operator.op_type) in {"RELU", "RELU6"}
        and list(operator.outputs) != ["relu_nchw"]
    ]
    assert [list(operator.inputs) for operator in consumers].count(
        ["split_0_nchw"]
    ) == 2


def test_indexed_split_all_outputs_clones_shared_axis_with_layout_state() -> None:
    model_ir = _make_model(integer_dtype="INT64", shared_axis=True)
    index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        model_ir,
        graph_index=index,
        layout_state=layout_state,
    )

    assert stats[
        "optimized_transpose_relu_split_all_outputs_to_nhwc_chains"
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
    assert list(split.inputs) == ["axis_nhwc", "relu_nchw"]
    assert list(reshape.inputs) == ["legacy_value", "axis"]
    assert validate_model_ir_invariants(model_ir, graph_index=index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_indexed_split_all_outputs_candidate_limit_and_idempotence() -> None:
    model_ir = _make_model()
    split = model_ir.operators[2]
    index = ModelIRGraphIndex(model_ir)

    assert optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        model_ir,
        graph_index=index,
        candidate=split,
        max_rewrites=0,
    ) == {"optimized_transpose_relu_split_all_outputs_to_nhwc_chains": 0}
    assert optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        model_ir,
        graph_index=index,
        candidate=split,
    ) == {"optimized_transpose_relu_split_all_outputs_to_nhwc_chains": 1}
    assert optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        model_ir,
        graph_index=index,
    ) == {"optimized_transpose_relu_split_all_outputs_to_nhwc_chains": 0}


UnsafeMutation = Callable[[ModelIR], None]


def _wrong_pre_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nchw"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _wrong_post_permutation(model_ir: ModelIR) -> None:
    model_ir.tensors["to_nhwc"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)


def _unresolved_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("src_nhwc")


def _late_source(model_ir: ModelIR) -> None:
    model_ir.inputs.remove("src_nhwc")
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["late"], outputs=["src_nhwc"])
    )
    model_ir.tensors["late"] = _tensor("late", list(model_ir.tensors["src_nhwc"].shape))
    model_ir.inputs.append("late")


def _pre_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["pre_fanout"] = _tensor("pre_fanout", [1, 4, _H, _W])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["pre_nchw"], outputs=["pre_fanout"])
    )


def _relu_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["relu_fanout"] = _tensor("relu_fanout", [1, 4, _H, _W])
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["relu_nchw"], outputs=["relu_fanout"])
    )


def _split_output_fanout(model_ir: ModelIR) -> None:
    model_ir.tensors["split_fanout"] = _tensor(
        "split_fanout",
        [1, _BRANCH_CHANNELS, _H, _W],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RELU",
            inputs=["split_0_nchw"],
            outputs=["split_fanout"],
        )
    )


def _uneven_split_shape(model_ir: ModelIR) -> None:
    model_ir.tensors["split_0_nchw"].shape[1] += 1


def _wrong_split_count(model_ir: ModelIR) -> None:
    model_ir.operators[2].options["numSplits"] = 3


def _public_split_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("split_0_nchw")


def _public_post_output(model_ir: ModelIR) -> None:
    model_ir.outputs.append("post_0_nhwc")


def _missing_post_tensor(model_ir: ModelIR) -> None:
    del model_ir.tensors["post_0_nhwc"]


def _variable_intermediate(model_ir: ModelIR) -> None:
    model_ir.tensors["split_0_nchw"].is_variable = True


def _constant_intermediate(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0_nhwc"].data = np.zeros(
        model_ir.tensors["post_0_nhwc"].shape,
        dtype=np.float32,
    )


def _dtype_mismatch(model_ir: ModelIR) -> None:
    model_ir.tensors["post_0_nhwc"].dtype = "FLOAT16"


def _per_axis_quantization(model_ir: ModelIR) -> None:
    model_ir.tensors["split_0_nchw"].quantization = QuantParamIR(
        scale=[0.25, 0.5],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _contradictory_source_layout(model_ir: ModelIR) -> None:
    model_ir.tensors["src_nhwc"].physical_layout = LOGICAL_LAYOUT_NCHW


def _duplicate_producer(model_ir: ModelIR) -> None:
    model_ir.operators.insert(
        1,
        OperatorIR(op_type="RELU", inputs=["src_nhwc"], outputs=["pre_nchw"]),
    )


def _consumer_before_post(model_ir: ModelIR) -> None:
    consumer = model_ir.operators.pop(-1)
    model_ir.operators.insert(3, consumer)


@pytest.mark.parametrize(
    "mutation",
    [
        _wrong_pre_permutation,
        _wrong_post_permutation,
        _unresolved_source,
        _late_source,
        _pre_fanout,
        _relu_fanout,
        _split_output_fanout,
        _uneven_split_shape,
        _wrong_split_count,
        _public_split_output,
        _public_post_output,
        _missing_post_tensor,
        _variable_intermediate,
        _constant_intermediate,
        _dtype_mismatch,
        _per_axis_quantization,
        _contradictory_source_layout,
        _duplicate_producer,
        _consumer_before_post,
    ],
    ids=lambda mutation: mutation.__name__.removeprefix("_"),
)
def test_indexed_split_all_outputs_rejects_unsafe_candidate_transactionally(
    mutation: UnsafeMutation,
) -> None:
    model_ir = _make_model()
    mutation(model_ir)
    before = _snapshot(copy.deepcopy(model_ir))
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_relu_split_all_outputs_to_nhwc_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_relu_split_all_outputs_to_nhwc_chains": 0
    }
    assert _snapshot(model_ir) == before
    assert layout_state.validate_against_model_ir(model_ir) == []
