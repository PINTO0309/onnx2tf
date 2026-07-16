from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    LOGICAL_LAYOUT_NCHW,
    LOGICAL_LAYOUT_NHWC,
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_pre_add_nhwc_chains,
)
from onnx2tf.tflite_builder.passes import pre_add_layout
from onnx2tf.tflite_builder.passes.pre_add_direct_unary_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_pre_add_direct_unary_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


_STATS_KEY = "optimized_transpose_pre_add_direct_unary_nhwc_chains"
_UNARY_TYPES = (
    "RELU",
    "RELU6",
    "LOGISTIC",
    "TANH",
    "GELU",
    "HARD_SWISH",
    "LEAKY_RELU",
)


def _tensor(name: str, shape: list[int], layout: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=[-1, *shape[1:]],
        logical_layout=layout,
        physical_layout=layout,
        onnx_tensor_name=name,
    )


def _constant(name: str, values: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="INT32",
        shape=[len(values)],
        shape_signature=[len(values)],
        data=np.asarray(values, dtype=np.int32),
    )


def _model(*, unary_type: str = "RELU", legacy: bool = True) -> ModelIR:
    model = ModelIR("indexed_pre_add_direct_unary")
    model.inputs = ["a", "b"]
    model.outputs = ["tail"]
    model.tensors = {
        "a": _tensor("a", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC),
        "b": _tensor("b", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC),
        "a_nchw": _tensor("a_nchw", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "b_nchw": _tensor("b_nchw", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "b_unary": _tensor("b_unary", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "sum": _tensor("sum", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW),
        "sum_nhwc": _tensor(
            "sum_nhwc", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC
        ),
        "tail": _tensor("tail", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC),
        "to_nchw": _constant("to_nchw", [0, 3, 1, 2]),
        "to_nhwc": _constant("to_nhwc", [0, 2, 3, 1]),
    }
    model.operators = [
        OperatorIR("TRANSPOSE", ["a", "to_nchw"], ["a_nchw"]),
        OperatorIR("TRANSPOSE", ["b", "to_nchw"], ["b_nchw"]),
        OperatorIR(unary_type, ["b_nchw"], ["b_unary"]),
        OperatorIR(
            "ADD",
            ["a_nchw", "b_unary"],
            ["sum"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR("TRANSPOSE", ["sum", "to_nhwc"], ["sum_nhwc"]),
        OperatorIR("RELU", ["sum_nhwc"], ["tail"]),
    ]
    if legacy:
        model.outputs.extend(["legacy_sum", "legacy_a"])
        model.tensors["legacy_sum"] = _tensor(
            "legacy_sum", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW
        )
        model.tensors["legacy_a"] = _tensor(
            "legacy_a", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW
        )
        model.operators.extend(
            [
                OperatorIR("ABS", ["sum"], ["legacy_sum"]),
                OperatorIR("ABS", ["a_nchw"], ["legacy_a"]),
            ]
        )
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


@pytest.mark.parametrize("unary_type", _UNARY_TYPES)
def test_strict_direct_unary_family_is_indexed_once(unary_type: str) -> None:
    model = _model(unary_type=unary_type)
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    stats = optimize_transpose_pre_add_direct_unary_nhwc_chains(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS_KEY: 1}
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    unary = next(
        operator for operator in model.operators if operator.op_type == unary_type
    )
    assert add.inputs == ["a", "b_unary"]
    assert add.outputs == ["sum_nhwc"]
    assert add.options["__transpose_pre_add_nhwc_optimized__"] is True
    assert unary.inputs == ["b"]
    assert model.tensors["b_unary"].shape == [1, 4, 5, 3]
    assert not any(operator.outputs == ["b_nchw"] for operator in model.operators)
    assert any(
        operator.op_type == "TRANSPOSE"
        and operator.inputs[0] == "sum_nhwc"
        and operator.outputs == ["sum"]
        for operator in model.operators
    )
    np.testing.assert_array_equal(
        model.tensors["to_nhwc"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert layout_state.validate_against_model_ir(model) == []
    _assert_index_matches(model, graph_index)
    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_closed_family_removes_all_three_adapters() -> None:
    model = _model(legacy=False)

    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(model) == {
        _STATS_KEY: 1
    }

    assert not any(operator.op_type == "TRANSPOSE" for operator in model.operators)
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    assert add.inputs == ["a", "b_unary"]
    assert add.outputs == ["sum_nhwc"]
    assert next(
        operator for operator in model.operators if operator.outputs == ["tail"]
    ).inputs == ["sum_nhwc"]


def test_multiple_post_aliases_share_one_canonical_nhwc_output() -> None:
    model = _model(legacy=False)
    post = next(
        operator
        for operator in model.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["sum_nhwc"]
    )
    post_index = model.operators.index(post)
    model.tensors["sum_nhwc_alias"] = _tensor(
        "sum_nhwc_alias", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC
    )
    model.tensors["tail_alias"] = _tensor(
        "tail_alias", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC
    )
    model.outputs.append("tail_alias")
    model.operators.insert(
        post_index + 1,
        OperatorIR(
            "TRANSPOSE",
            ["sum", "to_nhwc"],
            ["sum_nhwc_alias"],
        ),
    )
    model.operators.append(OperatorIR("ABS", ["sum_nhwc_alias"], ["tail_alias"]))

    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(model) == {
        _STATS_KEY: 1
    }

    assert not any(operator.op_type == "TRANSPOSE" for operator in model.operators)
    assert next(
        operator for operator in model.operators if operator.outputs == ["tail_alias"]
    ).inputs == ["sum_nhwc"]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.outputs.append("b_nchw"),
        lambda model: model.tensors["to_nchw"].data.__setitem__(
            slice(None), np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        lambda model: setattr(model.tensors["a_nchw"], "shape", [1, 4, 4, 5]),
        lambda model: setattr(
            model.tensors["b_unary"],
            "quantization",
            QuantParamIR(scale=[0.5], zero_point=[0], quantized_dimension=1),
        ),
        lambda model: (
            model.tensors.__setitem__(
                "b_side", _tensor("b_side", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW)
            ),
            model.outputs.append("b_side"),
            model.operators.insert(3, OperatorIR("ABS", ["b_nchw"], ["b_side"])),
        ),
        lambda model: (
            model.tensors.__setitem__(
                "shared_post", _tensor(
                    "shared_post", [1, 4, 5, 3], LOGICAL_LAYOUT_NHWC
                )
            ),
            model.outputs.append("shared_post"),
            model.operators.append(
                OperatorIR("TRANSPOSE", ["legacy_a", "to_nhwc"], ["shared_post"])
            ),
        ),
    ],
)
def test_guard_rejections_are_atomic(mutate) -> None:
    model = _model()
    mutate(model)
    before = _snapshot(model)

    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(
        model,
        layout_state=LayoutState.from_model_ir(model),
    ) == {_STATS_KEY: 0}
    assert _snapshot(model) == before


def test_optional_output_unary_is_owned_transactionally() -> None:
    model = _model(legacy=False)
    post = next(
        operator
        for operator in model.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["sum_nhwc"]
    )
    model.tensors["sum_relu"] = _tensor(
        "sum_relu", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW
    )
    output_unary = OperatorIR("RELU", ["sum"], ["sum_relu"])
    model.operators.insert(model.operators.index(post), output_unary)
    post.inputs[0] = "sum_relu"

    layout_state = LayoutState.from_model_ir(model)
    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(
        model,
        layout_state=layout_state,
    ) == {
        _STATS_KEY: 1
    }
    assert output_unary.inputs == ["sum"]
    assert output_unary.outputs == ["sum_nhwc"]
    assert model.tensors["sum"].shape == [1, 4, 5, 3]
    assert model.tensors["sum"].physical_layout == LOGICAL_LAYOUT_NHWC
    assert not any(operator is post for operator in model.operators)
    assert layout_state.validate_against_model_ir(model) == []
    assert _optimize_transpose_pre_add_nhwc_chains(
        model,
        layout_state=layout_state,
    ) == {"optimized_transpose_pre_add_nhwc_chains": 0}


def test_optional_output_unary_retains_one_owned_legacy_adapter() -> None:
    model = _model(legacy=True)
    post = next(
        operator
        for operator in model.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["sum_nhwc"]
    )
    legacy_sum = next(
        operator for operator in model.operators if operator.outputs == ["legacy_sum"]
    )
    model.tensors["sum_relu"] = _tensor(
        "sum_relu", [1, 3, 4, 5], LOGICAL_LAYOUT_NCHW
    )
    output_unary = OperatorIR("RELU", ["sum"], ["sum_relu"])
    model.operators.insert(model.operators.index(post), output_unary)
    post.inputs[0] = "sum_relu"
    legacy_sum.inputs[0] = "sum_relu"

    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(model) == {
        _STATS_KEY: 1
    }

    assert output_unary.inputs == ["sum"]
    assert output_unary.outputs == ["sum_nhwc"]
    assert model.tensors["sum"].shape == [1, 4, 5, 3]
    assert post.inputs == ["sum_nhwc", "to_nhwc"]
    assert post.outputs == ["sum_relu"]
    assert legacy_sum.inputs == ["sum_relu"]
    np.testing.assert_array_equal(
        model.tensors["to_nhwc"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )


def test_compatibility_wrapper_keeps_pruning_at_legacy_boundary() -> None:
    model = _model(legacy=False)

    assert _optimize_transpose_pre_add_nhwc_chains(model) == {
        "optimized_transpose_pre_add_nhwc_chains": 1
    }

    prune_events = [
        event
        for event in model.metadata.get("tensor_lineage_events", [])
        if event.get("kind") == "prune_unused_tensors"
    ]
    assert len(prune_events) == 1
    assert {
        "a_nchw",
        "b_nchw",
        "sum",
        "to_nchw",
        "to_nhwc",
    }.issubset(set(prune_events[0]["removed_names"]))


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
    model.tensors["a"].shape = [1, 4, 6, 3]
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
        new_name = f"b_{name}"
        tensor.name = new_name
    second.tensors = {f"b_{name}": tensor for name, tensor in second.tensors.items()}
    second.inputs = [f"b_{name}" for name in second.inputs]
    second.outputs = [f"b_{name}" for name in second.outputs]
    for operator in second.operators:
        operator.inputs = [f"b_{name}" for name in operator.inputs]
        operator.outputs = [f"b_{name}" for name in operator.outputs]
    combined = ModelIR("two_indexed_pre_add_groups")
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

    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(
        combined,
        graph_index=graph_index,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(
        combined,
        graph_index=graph_index,
        candidate=second_add,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    assert optimize_transpose_pre_add_direct_unary_nhwc_chains(
        combined,
        graph_index=graph_index,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    _assert_index_matches(combined, graph_index)


def test_composite_owner_matches_private_wrapper_on_compatibility_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pre_add_layout,
        "_optimize_transpose_pre_add_direct_unary_nhwc_chains_pass",
        lambda *args, **kwargs: {_STATS_KEY: 0},
    )
    direct = _model(legacy=True)
    wrapped = copy.deepcopy(direct)

    assert pre_add_layout.optimize_transpose_pre_add_nhwc_chains(direct) == {
        "optimized_transpose_pre_add_nhwc_chains": 1
    }
    assert _optimize_transpose_pre_add_nhwc_chains(wrapped) == {
        "optimized_transpose_pre_add_nhwc_chains": 1
    }
    assert _snapshot(direct) == _snapshot(wrapped)
