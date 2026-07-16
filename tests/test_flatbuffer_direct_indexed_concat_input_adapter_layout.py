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
    _apply_safe_transpose_reduction_lite,
    _optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
)
from onnx2tf.tflite_builder.passes.concat_input_adapter_layout import (
    _apply_plan,
    _resolve_candidate,
    optimize_transpose_input_chains_pre_concat_to_single_post_adapter,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


_STATS_KEY = "optimized_transpose_input_chains_pre_concat_to_single_post_adapter"
_UNARY_TYPES = (
    "LOGISTIC",
    "RELU",
    "RELU6",
    "RELU_0_TO_1",
    "ELU",
    "LEAKY_RELU",
    "TANH",
    "GELU",
    "HARD_SWISH",
    "ABS",
    "EXP",
    "NEG",
    "SQRT",
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
    branch_kinds: tuple[str, ...] = ("direct", "direct"),
    unary_type: str = "RELU",
    dynamic: bool = False,
    negative_axis: bool = False,
    public_concat: bool = False,
    add_existing_post: bool = False,
) -> ModelIR:
    def n(value: str) -> str:
        return _name(prefix, value)

    batch_signature = -1 if dynamic else 1
    model = ModelIR(n("concat_input_adapter"))
    model.outputs = [n("y")]
    model.tensors[n("perm")] = _constant(n("perm"), [0, 3, 1, 2])
    concat_inputs = []
    operators = []
    channel_counts = []

    for index, kind in enumerate(branch_kinds):
        channel_count = 1 if kind == "reshape_unary" else index + 2
        channel_counts.append(channel_count)
        source_name = n(f"x{index}")
        adapter_name = n(f"x{index}_nchw")
        output_name = n(f"b{index}")
        source_shape = [1, 4, 5, channel_count]
        source_signature = [batch_signature, 4, 5, channel_count]
        adapter_shape = [1, channel_count, 4, 5]
        adapter_signature = [batch_signature, channel_count, 4, 5]
        model.inputs.append(source_name)
        model.tensors[source_name] = _tensor(
            source_name,
            source_shape,
            signature=source_signature,
            layout=LOGICAL_LAYOUT_NHWC,
        )
        model.tensors[adapter_name] = _tensor(
            adapter_name,
            adapter_shape,
            signature=adapter_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        )

        if kind == "direct":
            operators.append(
                OperatorIR(
                    "TRANSPOSE",
                    [source_name, n("perm")],
                    [adapter_name],
                )
            )
            concat_inputs.append(adapter_name)
            continue

        if kind == "reshape_unary":
            shape_name = n(f"shape{index}")
            model.tensors[shape_name] = _constant(shape_name, adapter_shape)
            operators.append(
                OperatorIR(
                    "RESHAPE",
                    [source_name, shape_name],
                    [adapter_name],
                    options={"newShape": list(adapter_shape)},
                )
            )
        elif kind == "transpose_unary":
            operators.append(
                OperatorIR(
                    "TRANSPOSE",
                    [source_name, n("perm")],
                    [adapter_name],
                )
            )
        else:
            raise ValueError(kind)
        model.tensors[output_name] = _tensor(
            output_name,
            adapter_shape,
            signature=adapter_signature,
            layout=LOGICAL_LAYOUT_NCHW,
        )
        operators.append(
            OperatorIR(
                unary_type,
                [adapter_name],
                [output_name],
            )
        )
        concat_inputs.append(output_name)

    total_channels = sum(channel_counts)
    concat_name = n("cat")
    concat_signature = [batch_signature, total_channels, 4, 5]
    model.tensors[concat_name] = _tensor(
        concat_name,
        [1, total_channels, 4, 5],
        signature=concat_signature,
        layout=LOGICAL_LAYOUT_NCHW,
    )
    concat = OperatorIR(
        "CONCATENATION",
        concat_inputs,
        [concat_name],
        options={"axis": -3 if negative_axis else 1},
    )
    operators.append(concat)
    reshape_shape = n("output_shape")
    model.tensors[reshape_shape] = _constant(
        reshape_shape,
        [1, total_channels, 20],
    )
    model.tensors[n("y")] = _tensor(
        n("y"),
        [1, total_channels, 20],
        signature=[batch_signature, total_channels, 20],
    )
    operators.append(
        OperatorIR(
            "RESHAPE",
            [concat_name, reshape_shape],
            [n("y")],
        )
    )
    if add_existing_post:
        model.tensors[n("to_nhwc")] = _constant(
            n("to_nhwc"),
            [0, 2, 3, 1],
        )
        model.tensors[n("post")] = _tensor(
            n("post"),
            [1, 4, 5, total_channels],
            signature=[batch_signature, 4, 5, total_channels],
            layout=LOGICAL_LAYOUT_NHWC,
        )
        model.outputs.append(n("post"))
        operators.append(
            OperatorIR(
                "TRANSPOSE",
                [concat_name, n("to_nhwc")],
                [n("post")],
            )
        )
    if public_concat:
        model.outputs.append(concat_name)
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


def _assert_rewritten(model: ModelIR, *, prefix: str = "") -> None:
    def n(value: str) -> str:
        return _name(prefix, value)

    concat = next(
        operator
        for operator in model.operators
        if operator.op_type == "CONCATENATION" and operator.outputs == [n("cat_nhwc")]
    )
    assert concat.options["axis"] == 3
    assert concat.outputs == [n("cat_nhwc")]
    concat_index = model.operators.index(concat)
    compatibility = model.operators[concat_index + 1]
    assert compatibility.op_type == "TRANSPOSE"
    assert compatibility.inputs == [n("cat_nhwc"), n("perm")]
    assert compatibility.outputs == [n("cat")]
    assert model.tensors[n("cat")].physical_layout == LOGICAL_LAYOUT_NCHW
    assert model.tensors[n("cat_nhwc")].physical_layout == LOGICAL_LAYOUT_NHWC


def test_indexed_direct_inputs_preserve_terminal_nchw_boundary() -> None:
    model = _model(dynamic=True, negative_axis=True, add_existing_post=True)
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS_KEY: 1}
    _assert_rewritten(model)
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["x0", "x1"]
    assert [operator.op_type for operator in model.operators].count("TRANSPOSE") == 2
    assert model.tensors["cat_nhwc"].shape_signature == [-1, 4, 5, 5]
    existing_post = next(
        operator for operator in model.operators if operator.outputs == ["post"]
    )
    assert existing_post.inputs[0] == "cat"
    assert layout_state.validate_against_model_ir(model) == []
    _assert_index_matches(model, graph_index)
    assert optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
    ) == {_STATS_KEY: 0}


def test_compatibility_dispatcher_uses_index_and_layout_state() -> None:
    model = _model()
    graph_index = ModelIRGraphIndex(model)
    layout_state = LayoutState.from_model_ir(model)

    stats = _optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        model,
        graph_index=graph_index,
        layout_state=layout_state,
        max_rewrites=1,
    )

    assert stats == {_STATS_KEY: 1}
    _assert_index_matches(model, graph_index)
    assert layout_state.validate_against_model_ir(model) == []


@pytest.mark.parametrize("unary_type", _UNARY_TYPES)
def test_all_historical_unary_types_are_lifted_once(unary_type: str) -> None:
    model = _model(
        branch_kinds=("transpose_unary", "direct"),
        unary_type=unary_type,
        dynamic=True,
    )

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model)

    assert stats == {_STATS_KEY: 1}
    unary = next(
        operator for operator in model.operators if operator.op_type == unary_type
    )
    assert unary.inputs == ["x0"]
    assert model.tensors["b0"].shape == [1, 4, 5, 2]
    assert model.tensors["b0"].shape_signature == [-1, 4, 5, 2]
    assert model.tensors["b0"].physical_layout == LOGICAL_LAYOUT_NHWC
    _assert_rewritten(model)


def test_singleton_reshape_unary_is_removed_without_changing_unary_output_name() -> (
    None
):
    model = _model(branch_kinds=("reshape_unary", "direct"), unary_type="LOGISTIC")

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model)

    assert stats == {_STATS_KEY: 1}
    assert not any(
        operator.op_type == "RESHAPE" and operator.outputs == ["x0_nchw"]
        for operator in model.operators
    )
    logistic = next(
        operator for operator in model.operators if operator.op_type == "LOGISTIC"
    )
    assert logistic.inputs == ["x0"]
    assert model.tensors["b0"].shape == [1, 4, 5, 1]
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["b0", "x1"]
    _assert_rewritten(model)


def test_all_reshape_branches_create_one_terminal_permutation() -> None:
    model = _model(
        branch_kinds=("reshape_unary", "reshape_unary"),
        unary_type="RELU",
    )
    del model.tensors["perm"]

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model)

    assert stats == {_STATS_KEY: 1}
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    compatibility = model.operators[model.operators.index(concat) + 1]
    assert compatibility.inputs[1] == "transpose_input_chains_nhwc_to_nchw_perm"
    np.testing.assert_array_equal(
        model.tensors[compatibility.inputs[1]].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )


def test_public_concat_and_existing_consumers_keep_original_nchw_name() -> None:
    model = _model(public_concat=True, add_existing_post=True)

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model)

    assert stats == {_STATS_KEY: 1}
    assert "cat" in model.outputs
    producer = ModelIRGraphIndex(model).producer("cat")
    assert producer is not None and producer.op_type == "TRANSPOSE"
    assert all(
        operator.inputs[0] == "cat"
        for operator in model.operators
        if operator.op_type in {"RESHAPE", "TRANSPOSE"}
        and operator.outputs in (["y"], ["post"])
    )


def test_repeated_direct_concat_slot_is_rewritten_once() -> None:
    model = _model(branch_kinds=("direct",))
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    concat.inputs = ["x0_nchw", "x0_nchw"]
    model.tensors["cat"].shape = [1, 4, 4, 5]
    model.tensors["cat"].shape_signature = [1, 4, 4, 5]
    model.tensors["y"].shape = [1, 4, 20]
    model.tensors["y"].shape_signature = [1, 4, 20]
    model.tensors["output_shape"].data = np.asarray([1, 4, 20], dtype=np.int32)

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model)

    assert stats == {_STATS_KEY: 1}
    assert concat.inputs == ["x0", "x0"]
    assert [operator.op_type for operator in model.operators].count("TRANSPOSE") == 1


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: model.tensors["perm"].data.__setitem__(
            slice(None), np.asarray([0, 2, 3, 1], dtype=np.int32)
        ),
        lambda model: (
            model.tensors.__setitem__(
                "fanout",
                _tensor("fanout", [1, 2, 4, 5], layout=LOGICAL_LAYOUT_NCHW),
            ),
            model.outputs.append("fanout"),
            model.operators.insert(
                -2,
                OperatorIR("RELU", ["x0_nchw"], ["fanout"]),
            ),
        ),
        lambda model: model.outputs.append("x0_nchw"),
        lambda model: setattr(
            model.tensors["x0"],
            "physical_layout",
            LOGICAL_LAYOUT_NCHW,
        ),
        lambda model: setattr(model.tensors["x0_nchw"], "shape", [1, 3, 4, 5]),
        lambda model: setattr(
            model.tensors["x0"],
            "quantization",
            QuantParamIR(
                scale=[0.1, 0.2],
                zero_point=[0, 0],
                quantized_dimension=3,
            ),
        ),
        lambda model: (
            model.tensors.__setitem__("early", _tensor("early", [1, 5, 4, 5])),
            model.operators.insert(
                0,
                OperatorIR("RELU", ["cat"], ["early"]),
            ),
        ),
    ],
)
def test_direct_candidate_rejections_are_atomic(mutate) -> None:
    model = _model()
    mutate(model)
    before = _snapshot(model)

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        model,
        layout_state=LayoutState.from_model_ir(model),
    )

    assert stats == {_STATS_KEY: 0}
    assert _snapshot(model) == before


def test_duplicate_adapter_producer_is_rejected_atomically() -> None:
    model = _model()
    model.operators.insert(
        1,
        OperatorIR("TRANSPOSE", ["x0", "perm"], ["x0_nchw"]),
    )
    before = _snapshot(model)

    assert optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model) == {
        _STATS_KEY: 0
    }
    assert _snapshot(model) == before


@pytest.mark.parametrize(
    "mutate",
    [
        lambda model: setattr(
            next(
                operator for operator in model.operators if operator.op_type == "RELU"
            ),
            "op_type",
            "CUSTOM_UNARY",
        ),
        lambda model: model.outputs.append("b0"),
        lambda model: model.tensors["shape0"].data.__setitem__(1, 2),
        lambda model: (
            model.tensors.__setitem__(
                "unary_side", _tensor("unary_side", [1, 1, 4, 5])
            ),
            model.outputs.append("unary_side"),
            model.operators.insert(
                -2,
                OperatorIR("ABS", ["b0"], ["unary_side"]),
            ),
        ),
    ],
)
def test_unary_and_reshape_candidate_rejections_are_atomic(mutate) -> None:
    model = _model(
        branch_kinds=("reshape_unary", "direct"),
        unary_type="RELU",
    )
    mutate(model)
    before = _snapshot(model)

    assert optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model) == {
        _STATS_KEY: 0
    }
    assert _snapshot(model) == before


def test_stale_plan_is_revalidated_before_mutation() -> None:
    model = _model(branch_kinds=("transpose_unary", "direct"))
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
    model.tensors["x0"].shape = [1, 3, 5, 2]
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
    first = _model(prefix="a_")
    second = _model(prefix="b_")
    combined = ModelIR("two_concat_input_adapter_groups")
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

    assert optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        combined,
        graph_index=graph_index,
        max_rewrites=0,
    ) == {_STATS_KEY: 0}
    assert optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        combined,
        graph_index=graph_index,
        candidate=second_concat,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    _assert_rewritten(combined, prefix="b_")
    assert optimize_transpose_input_chains_pre_concat_to_single_post_adapter(
        combined,
        graph_index=graph_index,
        max_rewrites=1,
    ) == {_STATS_KEY: 1}
    _assert_rewritten(combined, prefix="a_")
    _assert_index_matches(combined, graph_index)


def test_zero_match_still_prunes_unused_tensors() -> None:
    model = _model()
    model.tensors["unused"] = _constant("unused", [7])
    model.tensors["perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)

    stats = optimize_transpose_input_chains_pre_concat_to_single_post_adapter(model)

    assert stats == {_STATS_KEY: 0}
    assert "unused" not in model.tensors


def test_safe_transpose_reduction_bundle_keeps_compatibility_entry_point() -> None:
    model = _model()

    stats = _apply_safe_transpose_reduction_lite(model)

    assert stats["safe_transpose_reduction_lite_applied"] > 0
    assert stats["safe_transpose_reduction_lite_reduced"] > 0
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.options["axis"] == 3
    assert concat.inputs == ["x0", "x1"]


def test_copy_isolation_and_repeated_sweeps_are_deterministic() -> None:
    first = _model(branch_kinds=("transpose_unary", "direct"))
    second = copy.deepcopy(first)

    first_counts = [
        optimize_transpose_input_chains_pre_concat_to_single_post_adapter(first)[
            _STATS_KEY
        ]
        for _ in range(3)
    ]
    second_counts = [
        optimize_transpose_input_chains_pre_concat_to_single_post_adapter(second)[
            _STATS_KEY
        ]
        for _ in range(3)
    ]

    assert first_counts == [1, 0, 0]
    assert second_counts == [1, 0, 0]
    assert _snapshot(first) == _snapshot(second)
