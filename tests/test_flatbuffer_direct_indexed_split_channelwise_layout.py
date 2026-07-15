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
    optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw,
)


_NHWC_SHAPE = [1, 2, 3, 4]
_NCHW_SHAPE = [1, 4, 2, 3]


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


def _make_tail(
    *,
    first_type: str = "MUL",
    second_type: str = "ADD",
    first_chain_lhs: bool = False,
    second_chain_lhs: bool = False,
    axis_dtype: str = "INT32",
    shared_axis: bool = False,
    downstream_split: bool = False,
    dynamic: bool = False,
) -> ModelIR:
    model_ir = ModelIR("indexed_binary_split_tail")
    model_ir.inputs = ["x_nhwc", "gate_nhwc", "rgb_nhwc"]
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
    model_ir.tensors["perm"] = _tensor(
        "perm",
        [4],
        dtype="INT64",
        data=np.asarray([0, 3, 1, 2], dtype=np.int64),
    )
    model_ir.tensors["gate_nhwc"] = _tensor(
        "gate_nhwc",
        [1, 2, 3, 1],
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
    )
    model_ir.tensors["rgb_nhwc"] = _tensor(
        "rgb_nhwc",
        _NHWC_SHAPE,
        logical_layout=LOGICAL_LAYOUT_NHWC,
        physical_layout=LOGICAL_LAYOUT_NHWC,
    )
    for name in ("first_out", "second_out", "cat_out", "y_nchw"):
        model_ir.tensors[name] = _tensor(
            name,
            _NCHW_SHAPE,
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
    numpy_axis_dtype = np.int64 if axis_dtype == "INT64" else np.int32
    model_ir.tensors["split_axis"] = _tensor(
        "split_axis",
        [1],
        dtype=axis_dtype,
        data=np.asarray([1], dtype=numpy_axis_dtype),
    )
    for index in range(4):
        model_ir.tensors[f"s{index}"] = _tensor(
            f"s{index}",
            [1, 1, 2, 3],
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )

    first_inputs = (
        ["x_nchw", "gate_nhwc"] if first_chain_lhs else ["gate_nhwc", "x_nchw"]
    )
    second_inputs = (
        ["first_out", "rgb_nhwc"] if second_chain_lhs else ["rgb_nhwc", "first_out"]
    )
    operators = [
        OperatorIR("TRANSPOSE", ["x_nhwc", "perm"], ["x_nchw"]),
        OperatorIR(first_type, first_inputs, ["first_out"]),
        OperatorIR(second_type, second_inputs, ["second_out"]),
        OperatorIR(
            "SPLIT",
            ["split_axis", "second_out"],
            ["s0", "s1", "s2", "s3"],
            options={"numSplits": 4},
        ),
        OperatorIR(
            "CONCATENATION",
            ["s0", "s1", "s2", "s3"],
            ["cat_out"],
            options={"axis": 1},
        ),
    ]
    terminal_input = "cat_out"
    if downstream_split:
        model_ir.tensors["down_axis"] = _tensor(
            "down_axis",
            [1],
            dtype=axis_dtype,
            data=np.asarray([1], dtype=numpy_axis_dtype),
        )
        for name in ("d0", "d1"):
            model_ir.tensors[name] = _tensor(
                name,
                [1, 2, 2, 3],
                logical_layout=LOGICAL_LAYOUT_NCHW,
                physical_layout=LOGICAL_LAYOUT_NCHW,
            )
        model_ir.tensors["down_cat"] = _tensor(
            "down_cat",
            _NCHW_SHAPE,
            logical_layout=LOGICAL_LAYOUT_NCHW,
            physical_layout=LOGICAL_LAYOUT_NCHW,
        )
        operators.extend(
            [
                OperatorIR(
                    "SPLIT",
                    ["down_axis", "cat_out"],
                    ["d0", "d1"],
                    options={"numSplits": 2},
                ),
                OperatorIR(
                    "CONCATENATION",
                    ["d0", "d1"],
                    ["down_cat"],
                    options={"axis": -3},
                ),
            ]
        )
        terminal_input = "down_cat"
    operators.append(OperatorIR("RELU_0_TO_1", [terminal_input], ["y_nchw"]))
    if shared_axis:
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
                [1, 1, 2, 3],
            )
        operators.append(
            OperatorIR(
                "SPLIT",
                ["split_axis", "other_nchw"],
                ["other_s0", "other_s1", "other_s2", "other_s3"],
                options={"numSplits": 4},
            )
        )
    model_ir.operators = operators

    if dynamic:
        dynamic_nhwc = [-1, -1, 3, 4]
        dynamic_nchw = [-1, 4, -1, 3]
        model_ir.tensors["x_nhwc"].shape_signature = dynamic_nhwc
        model_ir.tensors["gate_nhwc"].shape_signature = [-1, -1, 3, 1]
        model_ir.tensors["rgb_nhwc"].shape_signature = dynamic_nhwc
        for name in ("x_nchw", "first_out", "second_out", "cat_out", "y_nchw"):
            model_ir.tensors[name].shape_signature = dynamic_nchw
        for index in range(4):
            model_ir.tensors[f"s{index}"].shape_signature = [-1, 1, -1, 3]
        if downstream_split:
            model_ir.tensors["d0"].shape_signature = [-1, 2, -1, 3]
            model_ir.tensors["d1"].shape_signature = [-1, 2, -1, 3]
            model_ir.tensors["down_cat"].shape_signature = dynamic_nchw
    return model_ir


def _binary(
    op_type: str,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    if op_type == "ADD":
        return left + right
    if op_type == "SUB":
        return left - right
    if op_type == "MUL":
        return left * right
    if op_type == "DIV":
        return left / right
    if op_type == "MAXIMUM":
        return np.maximum(left, right)
    if op_type == "MINIMUM":
        return np.minimum(left, right)
    raise AssertionError(op_type)


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
        op_type = str(operator.op_type)
        if op_type == "TRANSPOSE":
            results = [
                np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
            ]
        elif op_type in {"ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM"}:
            results = [_binary(op_type, inputs[0], inputs[1])]
        elif op_type == "SPLIT":
            axis = int(np.asarray(inputs[0]).reshape(-1)[0])
            results = list(
                np.split(inputs[1], int(operator.options["numSplits"]), axis=axis)
            )
        elif op_type == "CONCATENATION":
            results = [np.concatenate(inputs, axis=int(operator.options["axis"]))]
        elif op_type == "RELU_0_TO_1":
            results = [np.clip(inputs[0], 0.0, 1.0)]
        else:
            raise AssertionError(op_type)
        for name, value in zip(operator.outputs, results):
            values[str(name)] = value
    return {name: values[name] for name in model_ir.outputs}


def _feeds(model_ir: ModelIR) -> Dict[str, np.ndarray]:
    random = np.random.default_rng(71)
    feeds = {
        "x_nhwc": random.uniform(0.75, 1.75, _NHWC_SHAPE).astype(np.float32),
        "gate_nhwc": random.uniform(0.75, 1.25, [1, 2, 3, 1]).astype(np.float32),
        "rgb_nhwc": random.uniform(0.75, 1.75, _NHWC_SHAPE).astype(np.float32),
    }
    if "other_nchw" in model_ir.inputs:
        feeds["other_nchw"] = random.uniform(
            0.75,
            1.75,
            _NCHW_SHAPE,
        ).astype(np.float32)
    return feeds


def _expected(
    feeds: Dict[str, np.ndarray],
    *,
    first_type: str,
    second_type: str,
    first_chain_lhs: bool,
    second_chain_lhs: bool,
) -> np.ndarray:
    first = _binary(
        first_type,
        feeds["x_nhwc"] if first_chain_lhs else feeds["gate_nhwc"],
        feeds["gate_nhwc"] if first_chain_lhs else feeds["x_nhwc"],
    )
    second = _binary(
        second_type,
        first if second_chain_lhs else feeds["rgb_nhwc"],
        feeds["rgb_nhwc"] if second_chain_lhs else first,
    )
    return np.transpose(np.clip(second, 0.0, 1.0), (0, 3, 1, 2))


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


@pytest.mark.parametrize(
    "first_type,second_type",
    [
        ("ADD", "ADD"),
        ("SUB", "SUB"),
        ("MUL", "MUL"),
        ("DIV", "DIV"),
        ("MAXIMUM", "MAXIMUM"),
        ("MINIMUM", "MINIMUM"),
    ],
)
@pytest.mark.parametrize("chain_lhs", [False, True])
def test_indexed_binary_split_tail_preserves_all_binary_semantics(
    first_type: str,
    second_type: str,
    chain_lhs: bool,
) -> None:
    model_ir = _make_tail(
        first_type=first_type,
        second_type=second_type,
        first_chain_lhs=chain_lhs,
        second_chain_lhs=chain_lhs,
    )
    feeds = _feeds(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 1
    }
    actual = _evaluate(model_ir, feeds)["y_nchw"]
    expected = _expected(
        feeds,
        first_type=first_type,
        second_type=second_type,
        first_chain_lhs=chain_lhs,
        second_chain_lhs=chain_lhs,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
    assert validate_model_ir_invariants(model_ir, graph_index=graph_index) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_binary_split_tail_preserves_public_nchw_contract() -> None:
    model_ir = _make_tail()
    original_public_shape = list(model_ir.tensors["y_nchw"].shape)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert next(
        operator for operator in model_ir.operators if operator.op_type == "MUL"
    ).inputs == [
        "gate_nhwc",
        "x_nhwc",
    ]
    assert [operator.op_type for operator in model_ir.operators].count("TRANSPOSE") == 1
    terminal = next(
        operator for operator in model_ir.operators if operator.op_type == "TRANSPOSE"
    )
    assert terminal.outputs == ["y_nchw"]
    assert np.asarray(model_ir.tensors[terminal.inputs[1]].data).tolist() == [
        0,
        3,
        1,
        2,
    ]
    assert model_ir.tensors["y_nchw"].shape == original_public_shape
    assert model_ir.tensors["y_nchw"].logical_layout == LOGICAL_LAYOUT_NCHW
    assert model_ir.tensors["y_nchw"].physical_layout == LOGICAL_LAYOUT_NCHW
    assert model_ir.tensors[terminal.inputs[0]].shape == _NHWC_SHAPE
    assert model_ir.tensors[terminal.inputs[0]].logical_layout == LOGICAL_LAYOUT_NHWC
    assert "x_nchw" not in model_ir.tensors
    assert (
        stats["optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw"]
        == 1
    )


def test_indexed_binary_split_tail_uses_bounded_downstream_split_worklist() -> None:
    model_ir = _make_tail(
        downstream_split=True,
        dynamic=True,
        axis_dtype="INT64",
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert (
        stats["optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw"]
        == 1
    )
    split_ops = [
        operator for operator in model_ir.operators if operator.op_type == "SPLIT"
    ]
    assert len(split_ops) == 2
    for operator in split_ops:
        axis_tensor = model_ir.tensors[operator.inputs[0]]
        assert axis_tensor.dtype == "INT64"
        assert np.asarray(axis_tensor.data).dtype == np.int64
        assert np.asarray(axis_tensor.data).reshape(-1).tolist() == [3]
    assert [
        int(operator.options["axis"])
        for operator in model_ir.operators
        if operator.op_type == "CONCATENATION"
    ] == [3, 3]
    assert model_ir.tensors["first_out"].shape_signature == [-1, -1, 3, 4]
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_binary_split_tail_clones_shared_axis_with_original_dtype() -> None:
    model_ir = _make_tail(axis_dtype="INT64", shared_axis=True)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert (
        stats["optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw"]
        == 1
    )
    root_split = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SPLIT" and operator.inputs[1] == "second_out"
    )
    unrelated_split = next(
        operator
        for operator in model_ir.operators
        if operator.op_type == "SPLIT" and operator.inputs[1] == "other_nchw"
    )
    assert root_split.inputs[0] != "split_axis"
    assert unrelated_split.inputs[0] == "split_axis"
    assert np.asarray(model_ir.tensors["split_axis"].data).tolist() == [1]
    clone = model_ir.tensors[root_split.inputs[0]]
    assert clone.dtype == "INT64"
    assert np.asarray(clone.data).dtype == np.int64
    assert np.asarray(clone.data).tolist() == [3]
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_binary_split_tail_candidate_limit_and_idempotence() -> None:
    model_ir = _make_tail()
    original = _fingerprint(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    pre = model_ir.operators[0]

    assert optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=0,
    ) == {"optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=model_ir.operators[1],
    ) == {"optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0}
    assert _fingerprint(model_ir) == original
    assert optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
        candidate=pre,
    ) == {"optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 1}
    after = _fingerprint(model_ir)
    assert optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {"optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0}
    assert _fingerprint(model_ir) == after


def _mutate_unsafe(model_ir: ModelIR, case: str) -> None:
    if case == "wrong_permutation":
        model_ir.tensors["perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int64)
    elif case == "wrong_root_axis":
        model_ir.tensors["split_axis"].data = np.asarray([2], dtype=np.int32)
        model_ir.tensors["split_axis"].dtype = "INT32"
    elif case == "fused_binary":
        model_ir.operators[1].options["fusedActivationFunction"] = "RELU"
    elif case == "per_axis_quantization":
        model_ir.tensors["first_out"].quantization = QuantParamIR(
            scale=[0.25, 0.5, 0.75, 1.0],
            zero_point=[0, 0, 0, 0],
            quantized_dimension=1,
        )
    elif case == "external_nchw_layout":
        model_ir.tensors["gate_nhwc"].physical_layout = LOGICAL_LAYOUT_NCHW
    elif case == "external_shape_mismatch":
        model_ir.tensors["gate_nhwc"].shape = [1, 4, 2, 3]
        model_ir.tensors["gate_nhwc"].shape_signature = [1, 4, 2, 3]
    elif case == "unsupported_closure_consumer":
        model_ir.tensors["reshape_shape"] = _tensor(
            "reshape_shape",
            [4],
            dtype="INT32",
            data=np.asarray([1, 1, 2, 3], dtype=np.int32),
        )
        model_ir.tensors["unsupported_out"] = _tensor(
            "unsupported_out",
            [1, 1, 2, 3],
        )
        model_ir.operators.append(
            OperatorIR(
                "RESHAPE",
                ["s0", "reshape_shape"],
                ["unsupported_out"],
            )
        )
    elif case == "multiple_public_outputs":
        model_ir.outputs.append("s0")
    elif case == "public_intermediate":
        model_ir.outputs = ["first_out"]
    elif case == "duplicate_producer":
        model_ir.operators.append(OperatorIR("RELU", ["x_nhwc"], ["first_out"]))
    elif case == "late_external_producer":
        model_ir.inputs.remove("rgb_nhwc")
        model_ir.operators.append(OperatorIR("RELU", ["x_nhwc"], ["rgb_nhwc"]))
    elif case == "missing_external_tensor":
        del model_ir.tensors["rgb_nhwc"]
    elif case == "wrong_split_metadata":
        model_ir.tensors["s0"].shape = [1, 2, 2, 3]
        model_ir.tensors["s0"].shape_signature = [1, 2, 2, 3]
    elif case == "shared_pre_output":
        model_ir.tensors["legacy_out"] = _tensor(
            "legacy_out",
            _NCHW_SHAPE,
        )
        model_ir.operators.append(OperatorIR("RELU", ["x_nchw"], ["legacy_out"]))
    elif case == "variable_permutation":
        model_ir.tensors["perm"].is_variable = True
    elif case == "consumer_before_producer":
        model_ir.operators[3], model_ir.operators[4] = (
            model_ir.operators[4],
            model_ir.operators[3],
        )
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_permutation",
        "wrong_root_axis",
        "fused_binary",
        "per_axis_quantization",
        "external_nchw_layout",
        "external_shape_mismatch",
        "unsupported_closure_consumer",
        "multiple_public_outputs",
        "public_intermediate",
        "duplicate_producer",
        "late_external_producer",
        "missing_external_tensor",
        "wrong_split_metadata",
        "shared_pre_output",
        "variable_permutation",
        "consumer_before_producer",
    ],
)
def test_indexed_binary_split_tail_rejects_unsafe_candidate_transactionally(
    case: str,
) -> None:
    model_ir = _make_tail()
    _mutate_unsafe(model_ir, case)
    before = copy.deepcopy(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw": 0
    }
    assert _fingerprint(model_ir) == _fingerprint(before)
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(graph_index, model_ir)


def test_indexed_binary_split_tail_uses_deterministic_private_name() -> None:
    model_ir = _make_tail()
    model_ir.tensors["y_nchw_nhwc"] = _tensor(
        "y_nchw_nhwc",
        [1],
        data=np.asarray([0.0], dtype=np.float32),
    )

    stats = optimize_transpose_binary_split_channelwise_tail_to_single_post_nchw(
        model_ir
    )

    assert (
        stats["optimized_transpose_binary_split_channelwise_tail_to_single_post_nchw"]
        == 1
    )
    terminal = next(
        operator for operator in model_ir.operators if operator.op_type == "TRANSPOSE"
    )
    assert terminal.inputs[0] == "y_nchw_nhwc_1"
