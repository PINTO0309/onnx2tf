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
    _optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
    _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze
from onnx2tf.tflite_builder.passes.split_mixed_concat_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains,
)


_STATS_KEY = (
    "optimized_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains"
)


def _name(prefix: str, value: str) -> str:
    return f"{prefix}{value}"


def _tensor(
    name: str,
    shape: list[int],
    *,
    signature: list[int] | None = None,
    layout: str = LOGICAL_LAYOUT_UNKNOWN,
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        quantization=quantization,
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


def _model(
    *,
    prefix: str = "",
    dynamic: bool = False,
    shared_axis: bool = False,
    negative_concat_axis: bool = False,
    without_direct: bool = False,
) -> ModelIR:
    def n(value: str) -> str:
        return _name(prefix, value)

    batch_signature = -1 if dynamic else 1
    nchw_signature = [batch_signature, 6, 3, 5]
    split0_signature = [batch_signature, 4, 3, 5]
    split1_signature = [batch_signature, 2, 3, 5]
    split1_nhwc_signature = [batch_signature, 3, 5, 2]

    model = ModelIR(n("split_mixed_concat"))
    model.inputs = [n("split_in")]
    model.outputs = [n("y"), n("gate")]
    model.tensors = {
        n("axis"): _constant(n("axis"), [1]),
        n("to_nhwc"): _constant(n("to_nhwc"), [0, 2, 3, 1]),
        n("split_in"): _tensor(
            n("split_in"),
            [1, 6, 3, 5],
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        n("s0"): _tensor(
            n("s0"),
            [1, 4, 3, 5],
            signature=split0_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        n("s1"): _tensor(
            n("s1"),
            [1, 2, 3, 5],
            signature=split1_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        n("s1_nhwc"): _tensor(
            n("s1_nhwc"),
            [1, 3, 5, 2],
            signature=split1_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        n("gate"): _tensor(
            n("gate"),
            [1, 3, 5, 2],
            signature=split1_nhwc_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        ),
        n("cat"): _tensor(
            n("cat"),
            [1, 6, 3, 5],
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
        n("reshape_shape"): _constant(n("reshape_shape"), [1, 6, 3, 5]),
        n("y"): _tensor(
            n("y"),
            [1, 6, 3, 5],
            signature=nchw_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        ),
    }
    split = OperatorIR(
        "SPLIT",
        [n("axis"), n("split_in")],
        [n("s0"), n("s1")],
    )
    post_split = OperatorIR(
        "TRANSPOSE",
        [n("s1"), n("to_nhwc")],
        [n("s1_nhwc")],
    )
    gate = OperatorIR("RELU", [n("s1_nhwc")], [n("gate")])
    operators = [split, post_split, gate]
    concat_inputs = [n("s0"), n("s1")]
    if not without_direct:
        model.inputs.append(n("direct"))
        model.tensors.update(
            {
                n("to_nchw"): _constant(n("to_nchw"), [0, 3, 1, 2]),
                n("direct"): _tensor(
                    n("direct"),
                    [1, 3, 5, 2],
                    signature=split1_nhwc_signature,
                    layout=LOGICAL_LAYOUT_NHWC,
                ),
                n("direct_nchw"): _tensor(
                    n("direct_nchw"),
                    [1, 2, 3, 5],
                    signature=split1_signature,
                    layout=LOGICAL_LAYOUT_NCHW,
                ),
            }
        )
        direct = OperatorIR(
            "TRANSPOSE",
            [n("direct"), n("to_nchw")],
            [n("direct_nchw")],
        )
        operators.append(direct)
        concat_inputs = [n("s0"), n("direct_nchw")]
    concat = OperatorIR(
        "CONCATENATION",
        concat_inputs,
        [n("cat")],
        options={"axis": -3 if negative_concat_axis else 1},
    )
    operators.extend(
        [
            concat,
            OperatorIR(
                "RESHAPE",
                [n("cat"), n("reshape_shape")],
                [n("y")],
            ),
        ]
    )
    if shared_axis:
        model.inputs.append(n("side"))
        model.outputs.append(n("side_y"))
        model.tensors[n("side")] = _tensor(n("side"), [1])
        model.tensors[n("side_y")] = _tensor(n("side_y"), [1])
        operators.append(
            OperatorIR(
                "RESHAPE",
                [n("side"), n("axis")],
                [n("side_y")],
            )
        )
    model.operators = operators
    return model


def _snapshot(model: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model.inputs),
        tuple(model.outputs),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
                _freeze(operator.axis_semantics),
                operator.version,
                operator.onnx_node_name,
                operator.onnx_op_type,
            )
            for operator in model.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                tensor.is_variable,
                _freeze(tensor.quantization),
                tensor.logical_layout,
                tensor.physical_layout,
                tensor.onnx_tensor_name,
            )
            for name, tensor in sorted(model.tensors.items())
        ),
        _freeze(model.metadata),
    )


def _assert_index_matches(model: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert {
        id(operator): graph_index.operator_index(operator)
        for operator in model.operators
    } == {id(operator): fresh.operator_index(operator) for operator in model.operators}


def test_indexed_split_mixed_concat_rewrite_preserves_terminal_nchw_contract() -> None:
    model = _model(dynamic=True, negative_concat_axis=True)
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    stats = (
        optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            model,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_STATS_KEY: 1}
    split = next(
        operator for operator in model.operators if operator.op_type == "SPLIT"
    )
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert np.asarray(model.tensors[split.inputs[0]].data).tolist() == [3]
    assert split.inputs[1] == "split_in_nhwc"
    assert concat.inputs == ["s0", "direct"]
    assert concat.outputs == ["cat_nhwc"]
    assert concat.options["axis"] == 3
    concat_index = model.operators.index(concat)
    compatibility = model.operators[concat_index + 1]
    assert compatibility.op_type == "TRANSPOSE"
    assert compatibility.inputs == ["cat_nhwc", "to_nchw"]
    assert compatibility.outputs == ["cat"]
    gate = next(operator for operator in model.operators if operator.op_type == "RELU")
    assert gate.inputs == ["s1"]
    assert [operator.op_type for operator in model.operators].count("TRANSPOSE") == 2
    assert model.tensors["s0"].shape == [1, 3, 5, 4]
    assert model.tensors["s0"].shape_signature == [-1, 3, 5, 4]
    assert model.tensors["s0"].physical_layout == LOGICAL_LAYOUT_NHWC
    assert model.tensors["cat"].physical_layout == LOGICAL_LAYOUT_NCHW
    assert model.tensors["cat_nhwc"].physical_layout == LOGICAL_LAYOUT_NHWC
    assert layout_state.validate_against_model_ir(model) == []
    _assert_index_matches(model, graph_index)

    assert optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}
    assert _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        model
    ) == {"optimized_transpose_input_chains_pre_concat_to_single_post_adapter": 0}


def test_compatibility_dispatcher_delegates_to_indexed_owner() -> None:
    model = _model()
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    stats = (
        _optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            model,
            graph_index=graph_index,
            layout_state=layout_state,
            max_rewrites=1,
        )
    )

    assert stats == {_STATS_KEY: 1}
    _assert_index_matches(model, graph_index)


def test_shared_split_axis_is_cloned_without_changing_other_consumer() -> None:
    model = _model(shared_axis=True, dynamic=True)
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    stats = (
        optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            model,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_STATS_KEY: 1}
    split = next(
        operator for operator in model.operators if operator.op_type == "SPLIT"
    )
    side = next(
        operator
        for operator in model.operators
        if operator.op_type == "RESHAPE" and operator.outputs == ["side_y"]
    )
    assert split.inputs[0] == "axis_nhwc"
    assert side.inputs[1] == "axis"
    assert np.asarray(model.tensors["axis"].data).tolist() == [1]
    assert np.asarray(model.tensors["axis_nhwc"].data).tolist() == [3]
    assert layout_state.validate_against_model_ir(model) == []
    _assert_index_matches(model, graph_index)


def test_all_split_concat_creates_only_missing_terminal_permutation() -> None:
    model = _model(without_direct=True)

    stats = (
        optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            model
        )
    )

    assert stats == {_STATS_KEY: 1}
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["s0", "s1"]
    assert concat.options["axis"] == 3
    compatibility = model.operators[model.operators.index(concat) + 1]
    assert compatibility.inputs[1] == "mixed_pre_concat_nhwc_to_nchw_perm"
    np.testing.assert_array_equal(
        model.tensors[compatibility.inputs[1]].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert not any(
        name.startswith("mixed_pre_concat_nchw_to_nhwc_perm") for name in model.tensors
    )


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.outputs.append("cat"),
        lambda model: model.outputs.append("s1_nhwc"),
        lambda model: model.tensors["axis"].data.__setitem__(0, 2),
        lambda model: model.tensors["to_nchw"].data.__setitem__(
            slice(None), np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        lambda model: (
            model.tensors.__setitem__(
                "direct_side", _tensor("direct_side", [1, 2, 3, 5])
            ),
            model.outputs.append("direct_side"),
            model.operators.insert(
                -2,
                OperatorIR("RELU", ["direct_nchw"], ["direct_side"]),
            ),
        ),
        lambda model: setattr(
            model.tensors["s0"],
            "quantization",
            QuantParamIR(
                scale=[0.1, 0.2],
                zero_point=[0, 0],
                quantized_dimension=1,
            ),
        ),
        lambda model: setattr(
            model.tensors["direct"],
            "physical_layout",
            LOGICAL_LAYOUT_NCHW,
        ),
    ],
)
def test_rejected_candidate_is_an_atomic_noop(mutate) -> None:
    model = _model()
    mutate(model)
    before = _snapshot(model)

    stats = (
        optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
            model,
            layout_state=LayoutState.from_model_ir(model),
        )
    )

    assert stats == {_STATS_KEY: 0}
    assert _snapshot(model) == before


def test_produced_axis_and_duplicate_direct_producer_are_rejected() -> None:
    produced_axis_model = _model()
    produced_axis_model.tensors["axis_source"] = _constant("axis_source", [1])
    produced_axis_model.operators.insert(
        0,
        OperatorIR("CAST", ["axis_source"], ["axis"]),
    )
    produced_before = _snapshot(produced_axis_model)
    assert optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
        produced_axis_model
    ) == {_STATS_KEY: 0}
    assert _snapshot(produced_axis_model) == produced_before

    duplicate_model = _model()
    duplicate_model.operators.insert(
        -2,
        OperatorIR(
            "TRANSPOSE",
            ["direct", "to_nchw"],
            ["direct_nchw"],
        ),
    )
    duplicate_before = _snapshot(duplicate_model)
    assert optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
        duplicate_model
    ) == {_STATS_KEY: 0}
    assert _snapshot(duplicate_model) == duplicate_before


def test_stale_plan_is_revalidated_before_any_owner_mutation() -> None:
    model = _model()
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    plan = _resolve_candidate(
        model,
        graph_index,
        concat,
        layout_state=layout_state,
    )
    assert plan is not None
    model.tensors["direct"].shape = [1, 4, 5, 2]
    before_apply = _snapshot(model)

    assert not _apply_plan(
        model,
        graph_index,
        plan,
        layout_state=layout_state,
    )
    assert _snapshot(model) == before_apply
    _assert_index_matches(model, graph_index)


def test_candidate_and_rewrite_limit_bound_the_indexed_dispatch() -> None:
    first = _model(prefix="a_")
    second = _model(prefix="b_")
    combined = ModelIR("two_split_mixed_concat_groups")
    combined.inputs = list(first.inputs) + list(second.inputs)
    combined.outputs = list(first.outputs) + list(second.outputs)
    combined.tensors = {**first.tensors, **second.tensors}
    combined.operators = list(first.operators) + list(second.operators)
    graph_index = ModelIRGraphIndex(combined)
    second_concat = next(
        operator
        for operator in combined.operators
        if operator.op_type == "CONCATENATION" and operator.outputs == ["b_cat"]
    )

    assert optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
        combined,
        graph_index=graph_index,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
        combined,
        graph_index=graph_index,
        candidate=second_concat,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    assert any(
        operator.op_type == "CONCATENATION" and operator.outputs == ["a_cat"]
        for operator in combined.operators
    )
    assert any(
        operator.op_type == "CONCATENATION" and operator.outputs == ["b_cat_nhwc"]
        for operator in combined.operators
    )
    assert optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
        combined,
        graph_index=graph_index,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    _assert_index_matches(combined, graph_index)


def test_copy_isolation_and_repeated_production_sweeps_are_deterministic() -> None:
    first = _model()
    second = copy.deepcopy(first)

    first_counts = []
    second_counts = []
    for _ in range(3):
        first_counts.append(
            optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
                first
            )[_STATS_KEY]
        )
        second_counts.append(
            optimize_transpose_split_mixed_pre_concat_to_single_post_adapter_nhwc_chains(
                second
            )[_STATS_KEY]
        )

    assert first_counts == [1, 0, 0]
    assert second_counts == [1, 0, 0]
    assert _snapshot(first) == _snapshot(second)
