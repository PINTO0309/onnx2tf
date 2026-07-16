from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    LOGICAL_LAYOUT_UNKNOWN,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.pre_add_mulconst_reshape_suffix_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


_STATS_KEY = "optimized_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains"


def _tensor(name: str, shape: list[int], layout: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        logical_layout=layout,
        physical_layout=layout,
        onnx_tensor_name=name,
    )


def _int_constant(name: str, values: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32",
        shape=[len(values)],
        shape_signature=[len(values)],
        data=np.asarray(values, dtype=np.int32),
    )


def _model(
    *,
    mul_const: bool = True,
    legacy: bool = True,
) -> ModelIR:
    model = ModelIR("indexed_pre_add_mulconst_reshape_suffix")
    model.inputs = ["a", "b"]
    model.outputs = ["tail"]
    model.tensors = {
        "a": _tensor("a", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC),
        "b": _tensor("b", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC),
        "a_nchw": _tensor("a_nchw", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "b_nchw": _tensor("b_nchw", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "b_scaled": _tensor("b_scaled", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "sum": _tensor("sum", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "sum_ncw": _tensor("sum_ncw", [1, 3, 20], LOGICAL_LAYOUT_UNKNOWN),
        "sum_nwc": _tensor("sum_nwc", [1, 20, 3], LOGICAL_LAYOUT_UNKNOWN),
        "tail": _tensor("tail", [1, 20, 3], LOGICAL_LAYOUT_UNKNOWN),
        "to_nchw": _int_constant("to_nchw", [0, 3, 1, 2]),
        "to_nwc": _int_constant("to_nwc", [0, 2, 1]),
        "reshape_shape": _int_constant("reshape_shape", [1, 3, 20]),
        "scale": TensorIR(
            name="scale",
            dtype="FLOAT32",
            shape=[1, 3, 1, 1],
            shape_signature=[1, 3, 1, 1],
            data=np.asarray([[[[0.25]], [[0.5]], [[0.75]]]], dtype=np.float32),
        ),
    }
    model.operators = [
        OperatorIR("TRANSPOSE", ["a", "to_nchw"], ["a_nchw"]),
        OperatorIR("TRANSPOSE", ["b", "to_nchw"], ["b_nchw"]),
    ]
    add_rhs = "b_nchw"
    if mul_const:
        model.operators.append(OperatorIR("MUL", ["b_nchw", "scale"], ["b_scaled"]))
        add_rhs = "b_scaled"
    model.operators.extend(
        [
            OperatorIR("ADD", ["a_nchw", add_rhs], ["sum"]),
            OperatorIR(
                "RESHAPE",
                ["sum", "reshape_shape"],
                ["sum_ncw"],
                options={
                    "newShape": [1, 3, 20],
                    "onnxRawNewShape": [1, 3, 20],
                },
            ),
            OperatorIR("TRANSPOSE", ["sum_ncw", "to_nwc"], ["sum_nwc"]),
            OperatorIR("RELU", ["sum_nwc"], ["tail"]),
        ]
    )
    if legacy:
        model.outputs.append("legacy")
        model.tensors["legacy"] = _tensor("legacy", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW)
        model.operators.append(OperatorIR("ABS", ["sum"], ["legacy"]))
    return model


def _snapshot(model: ModelIR):
    return (
        tuple(model.inputs),
        tuple(model.outputs),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                _freeze(tensor.quantization),
                tensor.logical_layout,
                tensor.physical_layout,
            )
            for name, tensor in sorted(model.tensors.items())
        ),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
            )
            for operator in model.operators
        ),
    )


def _assert_index_matches(model: ModelIR, index: ModelIRGraphIndex) -> None:
    rebuilt = ModelIRGraphIndex(model)
    assert index.producers == rebuilt.producers
    assert index.consumers == rebuilt.consumers
    assert index.duplicate_producers == rebuilt.duplicate_producers
    assert index._operator_indices_by_type == rebuilt._operator_indices_by_type


@pytest.mark.parametrize("mul_const", [False, True])
def test_direct_and_mulconst_families_are_indexed_once(mul_const: bool) -> None:
    model = _model(mul_const=mul_const)
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 1}

    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    reshape = next(
        operator for operator in model.operators if operator.op_type == "RESHAPE"
    )
    assert add.inputs == ["a", "b_scaled" if mul_const else "b"]
    assert add.outputs == ["sum_nhwc"]
    assert reshape.inputs == ["sum_nhwc", "reshape_shape"]
    assert reshape.outputs == ["sum_nwc"]
    assert reshape.options == {
        "newShape": [1, 20, 3],
        "onnxRawNewShape": [1, 20, 3],
    }
    np.testing.assert_array_equal(
        model.tensors["reshape_shape"].data,
        np.asarray([1, 20, 3], dtype=np.int32),
    )
    assert any(
        operator.op_type == "TRANSPOSE"
        and operator.inputs[0] == "sum_nhwc"
        and operator.outputs == ["sum"]
        for operator in model.operators
    )
    if mul_const:
        mul = next(
            operator for operator in model.operators if operator.op_type == "MUL"
        )
        assert mul.inputs == ["b", "scale"]
        assert model.tensors["scale"].shape == [1, 1, 1, 3]
        np.testing.assert_array_equal(
            model.tensors["scale"].data,
            np.asarray([[[[0.25, 0.5, 0.75]]]], dtype=np.float32),
        )
    assert layout_state.validate_against_model_ir(model) == []
    _assert_index_matches(model, graph_index)
    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_closed_family_removes_all_three_transposes() -> None:
    model = _model(legacy=False)

    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    assert not any(operator.op_type == "TRANSPOSE" for operator in model.operators)
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    assert add.outputs == ["sum"]
    assert model.tensors["sum"].shape == [1, 4, 5, 3]
    assert next(
        operator for operator in model.operators if operator.outputs == ["tail"]
    ).inputs == ["sum_nwc"]


def test_shared_scale_and_reshape_shape_use_copy_on_write() -> None:
    model = _model(legacy=False)
    model.tensors["scale_copy"] = TensorIR(
        name="scale_copy",
        dtype="FLOAT32",
        shape=[1, 3, 1, 1],
        shape_signature=[1, 3, 1, 1],
    )
    model.tensors["shape_copy"] = _int_constant("shape_copy", [1, 3, 20])
    model.outputs.extend(["scale_copy", "shape_copy"])
    model.operators.extend(
        [
            OperatorIR("IDENTITY", ["scale"], ["scale_copy"]),
            OperatorIR("IDENTITY", ["reshape_shape"], ["shape_copy"]),
        ]
    )

    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        model
    ) == {_STATS_KEY: 1}

    mul = next(operator for operator in model.operators if operator.op_type == "MUL")
    reshape = next(
        operator for operator in model.operators if operator.op_type == "RESHAPE"
    )
    assert mul.inputs == ["b", "scale_nhwc"]
    assert reshape.inputs == ["sum", "reshape_shape_nhwc"]
    np.testing.assert_array_equal(
        model.tensors["scale"].data,
        np.asarray([[[[0.25]], [[0.5]], [[0.75]]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        model.tensors["scale_nhwc"].data,
        np.asarray([[[[0.25, 0.5, 0.75]]]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        model.tensors["reshape_shape"].data,
        np.asarray([1, 3, 20], dtype=np.int32),
    )
    np.testing.assert_array_equal(
        model.tensors["reshape_shape_nhwc"].data,
        np.asarray([1, 20, 3], dtype=np.int32),
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.outputs.append("sum_nwc"),
        lambda model: model.tensors["to_nwc"].data.__setitem__(
            slice(None), np.asarray([0, 1, 2], dtype=np.int32)
        ),
        lambda model: model.tensors["reshape_shape"].data.__setitem__(
            slice(None), np.asarray([1, 4, 15], dtype=np.int32)
        ),
        lambda model: setattr(
            model.tensors["sum_nwc"],
            "quantization",
            QuantParamIR(
                scale=[0.25, 0.5, 0.75],
                zero_point=[0, 0, 0],
                quantized_dimension=2,
            ),
        ),
        lambda model: (
            model.tensors.__setitem__(
                "scaled_side",
                _tensor("scaled_side", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
            ),
            model.outputs.append("scaled_side"),
            model.operators.append(OperatorIR("ABS", ["b_scaled"], ["scaled_side"])),
        ),
    ],
)
def test_guard_rejections_are_atomic(mutate) -> None:
    model = _model()
    mutate(model)
    before = _snapshot(model)

    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        model,
        layout_state=LayoutState.from_model_ir(model),
    ) == {_STATS_KEY: 0}
    assert _snapshot(model) == before


def test_stale_plan_is_revalidated_before_mutation() -> None:
    model = _model()
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    plan = _resolve_candidate(
        model,
        graph_index,
        add,
        layout_state=layout_state,
    )
    assert plan is not None
    model.tensors["scale"].data[0, 0, 0, 0] = np.float32(0.375)
    before = _snapshot(model)

    assert not _apply_plan(
        model,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _snapshot(model) == before
    _assert_index_matches(model, graph_index)


def test_candidate_and_max_rewrites_bound_dispatch() -> None:
    first = _model(legacy=False)
    second = copy.deepcopy(first)
    for name, tensor in list(second.tensors.items()):
        tensor.name = f"b_{name}"
    second.tensors = {f"b_{name}": tensor for name, tensor in second.tensors.items()}
    second.inputs = [f"b_{name}" for name in second.inputs]
    second.outputs = [f"b_{name}" for name in second.outputs]
    for operator in second.operators:
        operator.inputs = [f"b_{name}" for name in operator.inputs]
        operator.outputs = [f"b_{name}" for name in operator.outputs]
    combined = ModelIR("two_pre_add_mulconst_reshape_suffix_groups")
    combined.inputs = list(first.inputs) + list(second.inputs)
    combined.outputs = list(first.outputs) + list(second.outputs)
    combined.tensors = {**first.tensors, **second.tensors}
    combined.operators = list(first.operators) + list(second.operators)
    graph_index = ModelIRGraphIndex(combined)
    second_add = next(
        operator
        for operator in combined.operators
        if operator.op_type == "ADD" and operator.outputs == ["b_sum"]
    )

    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        combined,
        graph_index=graph_index,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        combined,
        graph_index=graph_index,
        candidate=second_add,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    assert optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        combined,
        graph_index=graph_index,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    _assert_index_matches(combined, graph_index)


def test_compatibility_wrapper_keeps_one_prune_boundary() -> None:
    model = _model(legacy=False)

    assert _optimize_transpose_pre_add_mulconst_reshape_transpose_suffix_nhwc_chains(
        model,
        layout_state=LayoutState.from_model_ir(model),
    ) == {_STATS_KEY: 1}

    prune_events = [
        event
        for event in model.metadata.get("tensor_lineage_events", [])
        if event.get("kind") == "prune_unused_tensors"
    ]
    assert len(prune_events) == 1
    assert {
        "a_nchw",
        "b_nchw",
        "sum_ncw",
        "to_nchw",
        "to_nwc",
    }.issubset(set(prune_events[0]["removed_names"]))
