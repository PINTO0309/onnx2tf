from __future__ import annotations

import copy
from typing import Dict

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.split_channelwise_layout import (
    optimize_transpose_split_channelwise_tail_to_single_post_nchw,
)


_NHWC_SHAPE = [1, 3, 4, 4]
_NCHW_SHAPE = [1, 4, 3, 4]
_CROPPED_NHWC_SHAPE = [1, 2, 3, 4]
_CROPPED_NCHW_SHAPE = [1, 4, 2, 3]


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    logical_layout: str = "UNKNOWN",
    physical_layout: str = "UNKNOWN",
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        logical_layout=logical_layout,
        physical_layout=physical_layout,
        quantization=quantization,
    )


def _make_direct_tail(
    *,
    axis_dtype: str = "INT32",
    shared_constants: bool = False,
    dynamic: bool = False,
    unrelated_source_consumer: bool = False,
) -> ModelIR:
    numpy_int_dtype = np.int64 if axis_dtype == "INT64" else np.int32
    model_ir = ModelIR("indexed_direct_split_tail")
    model_ir.inputs = ["x_nhwc", "gate_nhwc"]
    model_ir.outputs = ["y_nchw"]
    model_ir.tensors["x_nhwc"] = _tensor(
        "x_nhwc",
        _NHWC_SHAPE,
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
    )
    model_ir.tensors["x_nchw"] = _tensor(
        "x_nchw",
        _NCHW_SHAPE,
        logical_layout=LOGICAL_LAYOUT_NCHW,
        physical_layout=LOGICAL_LAYOUT_NCHW,
    )
    model_ir.tensors["gate_nhwc"] = _tensor(
        "gate_nhwc",
        [1, 2, 3, 1],
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
    )
    model_ir.tensors["perm"] = _tensor(
        "perm",
        [4],
        dtype="INT64",
        data=np.asarray([0, 3, 1, 2], dtype=np.int64),
    )
    model_ir.tensors["split_axis"] = _tensor(
        "split_axis",
        [1],
        dtype=axis_dtype,
        data=np.asarray([1], dtype=numpy_int_dtype),
    )
    model_ir.tensors["slice_begin"] = _tensor(
        "slice_begin",
        [4],
        dtype=axis_dtype,
        data=np.asarray([0, 0, 1, 1], dtype=numpy_int_dtype),
    )
    model_ir.tensors["slice_size"] = _tensor(
        "slice_size",
        [4],
        dtype=axis_dtype,
        data=np.asarray([1, 1, 2, 3], dtype=numpy_int_dtype),
    )
    for index in range(4):
        model_ir.tensors[f"s{index}"] = _tensor(
            f"s{index}",
            [1, 1, 3, 4],
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        model_ir.tensors[f"crop{index}"] = _tensor(
            f"crop{index}",
            [1, 1, 2, 3],
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
    for name in ("cat_out", "binary_out", "y_nchw"):
        model_ir.tensors[name] = _tensor(
            name,
            _CROPPED_NCHW_SHAPE,
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )

    operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "perm"], ["x_nchw"]),
        OperatorIR(
            "SPLIT",
            ["split_axis", "x_nchw"],
            ["s0", "s1", "s2", "s3"],
            options={"numSplits": 4},
        ),
    ]
    operators.extend(
        OperatorIR(
            "SLICE",
            [f"s{index}", "slice_begin", "slice_size"],
            [f"crop{index}"],
        )
        for index in range(4)
    )
    operators.extend(
        [
            OperatorIR(
                "CONCATENATION",
                ["crop0", "crop1", "crop2", "crop3"],
                ["cat_out"],
                options={"axis": 1},
            ),
            OperatorIR("MUL", ["gate_nhwc", "cat_out"], ["binary_out"]),
            OperatorIR("RELU_0_TO_1", ["binary_out"], ["y_nchw"]),
        ]
    )
    if unrelated_source_consumer:
        model_ir.tensors["source_side"] = _tensor(
            "source_side",
            _NHWC_SHAPE,
            logical_layout=LOGICAL_LAYOUT_NHWC,
            physical_layout=LOGICAL_LAYOUT_NHWC,
        )
        operators.append(OperatorIR("RELU", ["x_nhwc"], ["source_side"]))
    if shared_constants:
        model_ir.inputs.append("other_nchw")
        model_ir.tensors["other_nchw"] = _tensor(
            "other_nchw",
            _NCHW_SHAPE,
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        for index in range(4):
            model_ir.tensors[f"other_s{index}"] = _tensor(
                f"other_s{index}",
                [1, 1, 3, 4],
            )
        model_ir.tensors["other_crop"] = _tensor(
            "other_crop",
            [1, 1, 2, 3],
        )
        operators.extend(
            [
                OperatorIR(
                    "SPLIT",
                    ["split_axis", "other_nchw"],
                    ["other_s0", "other_s1", "other_s2", "other_s3"],
                    options={"numSplits": 4},
                ),
                OperatorIR(
                    "SLICE",
                    ["other_s0", "slice_begin", "slice_size"],
                    ["other_crop"],
                ),
            ]
        )
    model_ir.operators = operators

    if dynamic:
        model_ir.tensors["x_nhwc"].shape_signature = [-1, -1, 4, 4]
        model_ir.tensors["x_nchw"].shape_signature = [-1, 4, -1, 4]
        model_ir.tensors["gate_nhwc"].shape_signature = [-1, 2, 3, 1]
        for index in range(4):
            model_ir.tensors[f"s{index}"].shape_signature = [-1, 1, -1, 4]
            model_ir.tensors[f"crop{index}"].shape_signature = [-1, 1, 2, 3]
        for name in ("cat_out", "binary_out", "y_nchw"):
            model_ir.tensors[name].shape_signature = [-1, 4, 2, 3]
    return model_ir


def _evaluate(
    model_ir: ModelIR,
    feeds: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    values = {name: np.asarray(value) for name, value in feeds.items()}
    for name, tensor in model_ir.tensors.items():
        if tensor.data is not None:
            values[name] = np.asarray(tensor.data)
    for operator in model_ir.operators:
        inputs = [values[str(name)] for name in operator.inputs]
        if operator.op_type == "TRANSPOSE":
            results = [
                np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
            ]
        elif operator.op_type == "SPLIT":
            results = list(
                np.split(
                    inputs[1],
                    int(operator.options["numSplits"]),
                    axis=int(np.asarray(inputs[0]).reshape(-1)[0]),
                )
            )
        elif operator.op_type == "SLICE":
            begin = np.asarray(inputs[1]).reshape(-1).tolist()
            size = np.asarray(inputs[2]).reshape(-1).tolist()
            slices = tuple(
                slice(
                    int(offset),
                    None if int(extent) == -1 else int(offset) + int(extent),
                )
                for offset, extent in zip(begin, size)
            )
            results = [inputs[0][slices]]
        elif operator.op_type == "CONCATENATION":
            results = [np.concatenate(inputs, axis=int(operator.options["axis"]))]
        elif operator.op_type == "MUL":
            results = [inputs[0] * inputs[1]]
        elif operator.op_type == "RELU":
            results = [np.maximum(inputs[0], 0.0)]
        elif operator.op_type == "RELU_0_TO_1":
            results = [np.clip(inputs[0], 0.0, 1.0)]
        else:
            raise AssertionError(operator.op_type)
        for name, value in zip(operator.outputs, results):
            values[str(name)] = value
    return {name: values[name] for name in model_ir.outputs}


def _feeds(model_ir: ModelIR) -> Dict[str, np.ndarray]:
    random = np.random.default_rng(83)
    feeds = {
        "x_nhwc": random.uniform(0.25, 1.5, _NHWC_SHAPE).astype(np.float32),
        "gate_nhwc": random.uniform(0.5, 1.25, [1, 2, 3, 1]).astype(np.float32),
    }
    if "other_nchw" in model_ir.inputs:
        feeds["other_nchw"] = random.uniform(
            0.25,
            1.5,
            _NCHW_SHAPE,
        ).astype(np.float32)
    return feeds


def _fingerprint(model_ir: ModelIR):
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                repr(operator.options),
            )
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or ()),
                tensor.logical_layout,
                tensor.physical_layout,
                repr(tensor.quantization),
                None
                if tensor.data is None
                else (
                    str(np.asarray(tensor.data).dtype),
                    tuple(np.asarray(tensor.data).shape),
                    tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                ),
            )
            for name, tensor in model_ir.tensors.items()
        ),
    )


def _assert_index_current(
    graph_index: ModelIRGraphIndex,
    model_ir: ModelIR,
) -> None:
    rebuilt = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == rebuilt.producers
    assert graph_index.consumers == rebuilt.consumers
    assert graph_index.duplicate_producers == rebuilt.duplicate_producers


@pytest.mark.parametrize("axis_dtype", ["INT32", "INT64"])
@pytest.mark.parametrize("dynamic", [False, True])
def test_indexed_direct_split_tail_preserves_slice_closure_semantics(
    axis_dtype: str,
    dynamic: bool,
) -> None:
    model_ir = _make_direct_tail(axis_dtype=axis_dtype, dynamic=dynamic)
    feeds = _feeds(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    expected = np.transpose(
        np.clip(
            feeds["x_nhwc"][:, 1:3, 1:4, :] * feeds["gate_nhwc"],
            0.0,
            1.0,
        ),
        (0, 3, 1, 2),
    )
    actual = _evaluate(model_ir, feeds)["y_nchw"]
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert stats == {
        "optimized_transpose_split_channelwise_tail_to_single_post_nchw": 1
    }
    assert np.asarray(model_ir.tensors["split_axis"].data).reshape(-1).tolist() == [3]
    assert np.asarray(model_ir.tensors["slice_begin"].data).reshape(-1).tolist() == [
        0,
        1,
        1,
        0,
    ]
    assert np.asarray(model_ir.tensors["slice_size"].data).reshape(-1).tolist() == [
        1,
        2,
        3,
        1,
    ]
    assert all(
        int(operator.options["axis"]) == 3
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    )
    assert model_ir.tensors["y_nchw"].shape == _CROPPED_NCHW_SHAPE
    assert model_ir.tensors["y_nchw"].physical_layout == LOGICAL_LAYOUT_NCHW
    assert [operator.op_type for operator in model_ir.operators].count("TRANSPOSE") == 1
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_direct_split_tail_clones_shared_axis_and_slice_constants() -> None:
    model_ir = _make_direct_tail(axis_dtype="INT64", shared_constants=True)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats["optimized_transpose_split_channelwise_tail_to_single_post_nchw"] == 1
    root_split = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SPLIT" and operator.inputs[1] == "x_nhwc"
    )
    unrelated_split = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SPLIT" and operator.inputs[1] == "other_nchw"
    )
    assert root_split.inputs[0] != "split_axis"
    assert unrelated_split.inputs[0] == "split_axis"
    assert np.asarray(model_ir.tensors["split_axis"].data).tolist() == [1]
    assert np.asarray(model_ir.tensors[root_split.inputs[0]].data).dtype == np.int64
    assert np.asarray(model_ir.tensors[root_split.inputs[0]].data).tolist() == [3]
    main_slices = [
        operator
        for operator in model_ir.operators
        if operator.op_type == "SLICE" and operator.inputs[0].startswith("s")
    ]
    unrelated_slice = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SLICE" and operator.inputs[0] == "other_s0"
    )
    assert len({operator.inputs[1] for operator in main_slices}) == 1
    assert len({operator.inputs[2] for operator in main_slices}) == 1
    assert main_slices[0].inputs[1] != "slice_begin"
    assert main_slices[0].inputs[2] != "slice_size"
    assert unrelated_slice.inputs[1:] == ["slice_begin", "slice_size"]
    assert np.asarray(model_ir.tensors["slice_begin"].data).tolist() == [0, 0, 1, 1]
    assert np.asarray(model_ir.tensors["slice_size"].data).tolist() == [1, 1, 2, 3]
    assert np.asarray(model_ir.tensors[main_slices[0].inputs[1]].data).dtype == np.int64
    assert np.asarray(model_ir.tensors[main_slices[0].inputs[1]].data).tolist() == [
        0,
        1,
        1,
        0,
    ]
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_direct_split_tail_does_not_absorb_unrelated_source_consumer() -> None:
    model_ir = _make_direct_tail(unrelated_source_consumer=True)
    side_operator = model_ir.operators[-1]
    side_tensor_before = copy.deepcopy(model_ir.tensors["source_side"])

    stats = optimize_transpose_split_channelwise_tail_to_single_post_nchw(model_ir)

    assert stats["optimized_transpose_split_channelwise_tail_to_single_post_nchw"] == 1
    assert side_operator in model_ir.operators
    assert side_operator.inputs == ["x_nhwc"]
    assert model_ir.tensors["source_side"] == side_tensor_before


def test_indexed_direct_split_tail_candidate_limit_and_idempotence() -> None:
    model_ir = _make_direct_tail()
    original = _fingerprint(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    pre = model_ir.operators[0]

    assert optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=0,
    ) == {"optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=model_ir.operators[1],
    ) == {"optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=pre,
    ) == {"optimized_transpose_split_channelwise_tail_to_single_post_nchw": 1}
    after = _fingerprint(model_ir)
    assert optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {"optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0}
    assert _fingerprint(model_ir) == after


def _mutate_unsafe(model_ir: ModelIR, case: str) -> None:
    if case == "wrong_permutation":
        model_ir.tensors["perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int64)
    elif case == "wrong_axis":
        model_ir.tensors["split_axis"].data = np.asarray([2], dtype=np.int32)
    elif case == "per_axis_quantization":
        model_ir.tensors["crop0"].quantization = QuantParamIR(
            scale=[0.25, 0.5, 0.75],
            zero_point=[0, 0, 0],
            quantized_dimension=3,
        )
    elif case == "unsupported_consumer":
        model_ir.tensors["shape"] = _tensor(
            "shape",
            [4],
            dtype="INT32",
            data=np.asarray([1, 1, 3, 4], dtype=np.int32),
        )
        model_ir.tensors["unsupported"] = _tensor(
            "unsupported",
            [1, 1, 3, 4],
        )
        model_ir.operators.append(
            OperatorIR("RESHAPE", ["s0", "shape"], ["unsupported"])
        )
    elif case == "multiple_outputs":
        model_ir.outputs.append("s0")
    elif case == "public_intermediate":
        model_ir.outputs = ["crop0"]
    elif case == "slice_dtype_mismatch":
        model_ir.tensors["slice_begin"].dtype = "INT64"
    elif case == "slice_out_of_bounds":
        model_ir.tensors["slice_size"].data = np.asarray(
            [1, 1, 5, 3],
            dtype=np.int32,
        )
    elif case == "shared_pre_output":
        model_ir.tensors["legacy"] = _tensor("legacy", _NCHW_SHAPE)
        model_ir.operators.append(OperatorIR("RELU", ["x_nchw"], ["legacy"]))
    elif case == "consumer_before_producer":
        model_ir.operators[1], model_ir.operators[2] = (
            model_ir.operators[2],
            model_ir.operators[1],
        )
    elif case == "duplicate_producer":
        model_ir.operators.append(OperatorIR("RELU", ["x_nhwc"], ["crop0"]))
    elif case == "missing_slice_constant":
        del model_ir.tensors["slice_begin"]
    elif case == "variable_slice_constant":
        model_ir.tensors["slice_size"].is_variable = True
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_permutation",
        "wrong_axis",
        "per_axis_quantization",
        "unsupported_consumer",
        "multiple_outputs",
        "public_intermediate",
        "slice_dtype_mismatch",
        "slice_out_of_bounds",
        "shared_pre_output",
        "consumer_before_producer",
        "duplicate_producer",
        "missing_slice_constant",
        "variable_slice_constant",
    ],
)
def test_indexed_direct_split_tail_rejects_unsafe_candidate_transactionally(
    case: str,
) -> None:
    model_ir = _make_direct_tail()
    _mutate_unsafe(model_ir, case)
    before = copy.deepcopy(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_split_channelwise_tail_to_single_post_nchw": 0
    }
    assert _fingerprint(model_ir) == _fingerprint(before)
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)
