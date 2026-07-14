from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_normalization_layout as instance_norm_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.instance_normalization_layout import (
    _repair_decomposed_instance_normalization_layouts,
)


def _tensor(
    name: str,
    *,
    dtype: str = "FLOAT32",
    shape: list[int],
    data: np.ndarray | None = None,
    logical_layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        logical_layout=logical_layout,
    )


def _add_instance_norm_branch(
    model_ir: ModelIR,
    *,
    prefix: str,
    x_shape: list[int],
    layout: str,
    post_transpose: bool = False,
    separate_axes: bool = False,
) -> dict[str, str | list[int]]:
    rank = len(x_shape)
    channel_axis = 1 if layout in {"NCHW", "NCDHW"} else rank - 1
    channel_size = int(x_shape[channel_axis])
    desired_axes = [axis for axis in range(1, rank) if axis != channel_axis]
    wrong_axes = [axis + 1 for axis in desired_axes]
    names = {
        "x": f"{prefix}_x",
        "axes1": f"{prefix}_axes1",
        "axes2": (f"{prefix}_axes2" if separate_axes else f"{prefix}_axes1"),
        "mean": f"{prefix}_mean",
        "centered": f"{prefix}_centered",
        "squared": f"{prefix}_squared",
        "variance": f"{prefix}_variance",
        "epsilon": f"{prefix}_epsilon",
        "variance_epsilon": f"{prefix}_variance_epsilon",
        "sqrt": f"{prefix}_sqrt",
        "one": f"{prefix}_one",
        "inverse_std": f"{prefix}_inverse_std",
        "normalized": f"{prefix}_normalized",
        "scale": f"{prefix}_scale",
        "scaled": f"{prefix}_scaled",
        "bias": f"{prefix}_bias",
        "output": f"{prefix}_output",
        "perm": f"{prefix}_perm",
        "post_scaled": f"{prefix}_post_scaled",
    }
    model_ir.inputs.append(str(names["x"]))
    model_ir.outputs.append(str(names["output"]))
    model_ir.tensors[str(names["x"])] = _tensor(
        str(names["x"]),
        shape=x_shape,
        logical_layout=layout,
    )
    model_ir.tensors[str(names["axes1"])] = _tensor(
        str(names["axes1"]),
        dtype="INT32",
        shape=[len(wrong_axes)],
        data=np.asarray(wrong_axes, dtype=np.int32),
    )
    if separate_axes:
        model_ir.tensors[str(names["axes2"])] = _tensor(
            str(names["axes2"]),
            dtype="INT64",
            shape=[len(wrong_axes)],
            data=np.asarray(wrong_axes, dtype=np.int64),
        )
    wrong_shape = [1 for _ in range(rank)]
    for key in (
        "mean",
        "centered",
        "squared",
        "variance",
        "variance_epsilon",
        "sqrt",
        "inverse_std",
        "normalized",
        "scaled",
    ):
        model_ir.tensors[str(names[key])] = _tensor(
            str(names[key]),
            shape=wrong_shape,
        )
    model_ir.tensors[str(names["epsilon"])] = _tensor(
        str(names["epsilon"]),
        shape=[],
        data=np.asarray(1e-5, dtype=np.float32),
    )
    model_ir.tensors[str(names["one"])] = _tensor(
        str(names["one"]),
        shape=[],
        data=np.asarray(1.0, dtype=np.float32),
    )
    model_ir.tensors[str(names["scale"])] = _tensor(
        str(names["scale"]),
        shape=[channel_size],
        data=np.linspace(0.5, 1.5, channel_size, dtype=np.float32),
    )
    model_ir.tensors[str(names["bias"])] = _tensor(
        str(names["bias"]),
        shape=[channel_size],
        data=np.linspace(-0.25, 0.25, channel_size, dtype=np.float32),
    )

    model_ir.operators.extend(
        [
            OperatorIR(
                "MEAN",
                [str(names["x"]), str(names["axes1"])],
                [str(names["mean"])],
                options={"keepDims": True},
                onnx_node_name=f"{prefix}_instance_norm",
                onnx_op_type="InstanceNormalization",
            ),
            OperatorIR(
                "SUB",
                [str(names["x"]), str(names["mean"])],
                [str(names["centered"])],
            ),
            OperatorIR(
                "MUL",
                [str(names["centered"]), str(names["centered"])],
                [str(names["squared"])],
            ),
            OperatorIR(
                "MEAN",
                [str(names["squared"]), str(names["axes2"])],
                [str(names["variance"])],
                options={"keepDims": True},
            ),
            OperatorIR(
                "ADD",
                [str(names["variance"]), str(names["epsilon"])],
                [str(names["variance_epsilon"])],
            ),
            OperatorIR(
                "SQRT",
                [str(names["variance_epsilon"])],
                [str(names["sqrt"])],
            ),
            OperatorIR(
                "DIV",
                [str(names["one"]), str(names["sqrt"])],
                [str(names["inverse_std"])],
            ),
            OperatorIR(
                "MUL",
                [str(names["centered"]), str(names["inverse_std"])],
                [str(names["normalized"])],
            ),
            OperatorIR(
                "MUL",
                [str(names["normalized"]), str(names["scale"])],
                [str(names["scaled"])],
            ),
        ]
    )
    bias_data_name = str(names["scaled"])
    bias_shape = list(x_shape)
    if post_transpose:
        if rank == 4 and layout == "NHWC":
            permutation = [0, 3, 1, 2]
        elif rank == 4 and layout == "NCHW":
            permutation = [0, 2, 3, 1]
        else:
            permutation = list(range(rank))
        model_ir.tensors[str(names["perm"])] = _tensor(
            str(names["perm"]),
            dtype="INT32",
            shape=[rank],
            data=np.asarray(permutation, dtype=np.int32),
        )
        post_shape = [x_shape[index] for index in permutation]
        model_ir.tensors[str(names["post_scaled"])] = _tensor(
            str(names["post_scaled"]),
            shape=post_shape,
        )
        model_ir.operators.append(
            OperatorIR(
                "TRANSPOSE",
                [str(names["scaled"]), str(names["perm"])],
                [str(names["post_scaled"])],
            )
        )
        bias_data_name = str(names["post_scaled"])
        bias_shape = post_shape
    model_ir.tensors[str(names["output"])] = _tensor(
        str(names["output"]),
        shape=bias_shape,
    )
    model_ir.operators.append(
        OperatorIR(
            "ADD",
            [bias_data_name, str(names["bias"])],
            [str(names["output"])],
        )
    )
    names["desired_axes"] = desired_axes
    names["desired_reduced_shape"] = [
        1 if axis in desired_axes else int(value) for axis, value in enumerate(x_shape)
    ]
    names["desired_scale_shape"] = [
        channel_size if axis == channel_axis else 1 for axis in range(rank)
    ]
    if post_transpose:
        inverse = [0 for _ in range(rank)]
        for new_axis, old_axis in enumerate(permutation):
            inverse[old_axis] = new_axis
        post_channel_axis = inverse[channel_axis]
        names["desired_bias_shape"] = [
            channel_size if axis == post_channel_axis else 1 for axis in range(rank)
        ]
    else:
        names["desired_bias_shape"] = list(names["desired_scale_shape"])
    return names


def _model_with_branches() -> tuple[ModelIR, list[dict[str, str | list[int]]]]:
    model_ir = ModelIR("indexed_instance_normalization_layout")
    branches = [
        _add_instance_norm_branch(
            model_ir,
            prefix="nhwc",
            x_shape=[1, 4, 5, 3],
            layout="NHWC",
        ),
        _add_instance_norm_branch(
            model_ir,
            prefix="nchw",
            x_shape=[1, 3, 4, 5],
            layout="NCHW",
            separate_axes=True,
        ),
        _add_instance_norm_branch(
            model_ir,
            prefix="post",
            x_shape=[1, 4, 5, 3],
            layout="NHWC",
            post_transpose=True,
        ),
    ]
    return model_ir, branches


def _assert_index_current(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_instance_normalization_layout_repairs_multiple_layouts_with_one_index(
    monkeypatch,
) -> None:
    model_ir, branches = _model_with_branches()
    original_operators = copy.deepcopy(model_ir.operators)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _repair_decomposed_instance_normalization_layouts(model_ir)

    assert stats == {"repaired_decomposed_instance_normalization_layouts": 3}
    assert refresh_count == 1
    assert model_ir.operators == original_operators
    for names in branches:
        desired_axes = names["desired_axes"]
        desired_reduced = names["desired_reduced_shape"]
        x_shape = model_ir.tensors[str(names["x"])].shape
        for axes_key in {str(names["axes1"]), str(names["axes2"])}:
            assert np.asarray(model_ir.tensors[axes_key].data).tolist() == desired_axes
        for key in (
            "mean",
            "variance",
            "variance_epsilon",
            "sqrt",
            "inverse_std",
        ):
            assert model_ir.tensors[str(names[key])].shape == desired_reduced
        for key in ("centered", "squared", "normalized", "scaled"):
            assert model_ir.tensors[str(names[key])].shape == x_shape
        assert (
            model_ir.tensors[str(names["scale"])].shape == names["desired_scale_shape"]
        )
        assert model_ir.tensors[str(names["bias"])].shape == names["desired_bias_shape"]


def test_instance_normalization_layout_keeps_supplied_index_and_layout_current() -> (
    None
):
    model_ir = ModelIR("maintained_instance_normalization_layout")
    _add_instance_norm_branch(
        model_ir,
        prefix="rank5",
        x_shape=[1, 3, 2, 4, 5],
        layout="NCDHW",
    )
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _repair_decomposed_instance_normalization_layouts(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"repaired_decomposed_instance_normalization_layouts": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_instance_normalization_layout_already_correct_is_complete_noop() -> None:
    model_ir = ModelIR("already_correct_instance_normalization_layout")
    names = _add_instance_norm_branch(
        model_ir,
        prefix="rank3",
        x_shape=[1, 5, 3],
        layout="NWC",
    )
    assert _repair_decomposed_instance_normalization_layouts(model_ir) == {
        "repaired_decomposed_instance_normalization_layouts": 1
    }
    before = repr(model_ir)

    stats = _repair_decomposed_instance_normalization_layouts(model_ir)

    assert stats == {"repaired_decomposed_instance_normalization_layouts": 0}
    assert repr(model_ir) == before
    assert model_ir.tensors[str(names["scale"])].shape == [1, 1, 3]


def test_instance_normalization_layout_plan_failure_is_transactional(
    monkeypatch,
) -> None:
    model_ir = ModelIR("transactional_instance_normalization_layout")
    _add_instance_norm_branch(
        model_ir,
        prefix="branch",
        x_shape=[1, 4, 5, 3],
        layout="NHWC",
    )
    before = repr(model_ir)
    original_plan = instance_norm_module._constant_plan
    calls = 0

    def fail_second_plan(tensor, broadcast_shape):
        nonlocal calls
        calls += 1
        if calls == 2:
            return None
        return original_plan(tensor, broadcast_shape)

    monkeypatch.setattr(
        instance_norm_module,
        "_constant_plan",
        fail_second_plan,
    )

    stats = _repair_decomposed_instance_normalization_layouts(model_ir)

    assert stats == {"repaired_decomposed_instance_normalization_layouts": 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "case",
    [
        "unknown_layout",
        "rank_two",
        "dynamic_channel",
        "wrong_marker",
        "mean1_keepdims_false",
        "reverse_sub_inputs",
        "mean_public",
        "mean_fanout",
        "duplicate_mean_producer",
        "centered_extra_fanout",
        "square_wrong_type",
        "duplicate_squared_producer",
        "mean2_wrong_type",
        "add_epsilon_wrong_type",
        "epsilon_nonfinite",
        "sqrt_wrong_type",
        "div_reversed_inputs",
        "one_not_one",
        "normalized_mul_missing_inverse",
        "normalized_fanout",
        "scale_wrong_size",
        "scale_shared",
        "scale_public",
        "bias_wrong_size",
        "bias_shared",
        "bias_public",
        "axes_float_dtype",
        "axes_shared",
        "axes_public",
        "invalid_intermediate_signature",
        "malformed_post_permutation",
        "post_output_public",
    ],
)
def test_instance_normalization_layout_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = ModelIR("rejected_instance_normalization_layout")
    names = _add_instance_norm_branch(
        model_ir,
        prefix="branch",
        x_shape=[1, 4, 5, 3],
        layout="NHWC",
        post_transpose=case in {"malformed_post_permutation", "post_output_public"},
    )
    mean1, sub, square, mean2, add_epsilon, sqrt, div, normalized_mul = (
        model_ir.operators[:8]
    )

    if case == "unknown_layout":
        model_ir.tensors[str(names["x"])].logical_layout = "UNKNOWN"
    elif case == "rank_two":
        model_ir.tensors[str(names["x"])].shape = [1, 3]
    elif case == "dynamic_channel":
        model_ir.tensors[str(names["x"])].shape[-1] = -1
    elif case == "wrong_marker":
        mean1.onnx_op_type = "Mean"
    elif case == "mean1_keepdims_false":
        mean1.options["keepDims"] = False
    elif case == "reverse_sub_inputs":
        sub.inputs = list(reversed(sub.inputs))
    elif case == "mean_public":
        model_ir.outputs.append(str(names["mean"]))
    elif case == "mean_fanout":
        model_ir.tensors["side"] = _tensor("side", shape=[1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["mean"])], ["side"])
        )
    elif case == "duplicate_mean_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["x"])], [str(names["mean"])])
        )
    elif case == "centered_extra_fanout":
        model_ir.tensors["side"] = _tensor("side", shape=[1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["centered"])], ["side"])
        )
    elif case == "square_wrong_type":
        square.op_type = "ADD"
    elif case == "duplicate_squared_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [str(names["centered"])],
                [str(names["squared"])],
            )
        )
    elif case == "mean2_wrong_type":
        mean2.op_type = "SUM"
    elif case == "add_epsilon_wrong_type":
        add_epsilon.op_type = "MUL"
    elif case == "epsilon_nonfinite":
        model_ir.tensors[str(names["epsilon"])].data = np.asarray(np.inf)
    elif case == "sqrt_wrong_type":
        sqrt.op_type = "RSQRT"
    elif case == "div_reversed_inputs":
        div.inputs = list(reversed(div.inputs))
    elif case == "one_not_one":
        model_ir.tensors[str(names["one"])].data = np.asarray(2.0)
    elif case == "normalized_mul_missing_inverse":
        normalized_mul.inputs[1] = str(names["one"])
    elif case == "normalized_fanout":
        model_ir.tensors["side"] = _tensor("side", shape=[1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["normalized"])], ["side"])
        )
    elif case == "scale_wrong_size":
        model_ir.tensors[str(names["scale"])].data = np.ones(2, dtype=np.float32)
    elif case == "scale_shared":
        model_ir.tensors["side"] = _tensor("side", shape=[3])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["scale"])], ["side"])
        )
    elif case == "scale_public":
        model_ir.outputs.append(str(names["scale"]))
    elif case == "bias_wrong_size":
        model_ir.tensors[str(names["bias"])].data = np.ones(2, dtype=np.float32)
    elif case == "bias_shared":
        model_ir.tensors["side"] = _tensor("side", shape=[3])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["bias"])], ["side"])
        )
    elif case == "bias_public":
        model_ir.inputs.append(str(names["bias"]))
    elif case == "axes_float_dtype":
        model_ir.tensors[str(names["axes1"])].data = np.asarray(
            [2, 3],
            dtype=np.float32,
        )
    elif case == "axes_shared":
        model_ir.tensors["side"] = _tensor("side", shape=[2])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [str(names["axes1"])], ["side"])
        )
    elif case == "axes_public":
        model_ir.outputs.append(str(names["axes1"]))
    elif case == "invalid_intermediate_signature":
        model_ir.tensors[str(names["sqrt"])].shape_signature = [
            1,
            None,
            1,
            1,
        ]
    elif case == "malformed_post_permutation":
        model_ir.tensors[str(names["perm"])].data = np.asarray(
            [0, 3, 3, 1],
            dtype=np.int32,
        )
    elif case == "post_output_public":
        model_ir.outputs.append(str(names["post_scaled"]))

    before = repr(model_ir)
    stats = _repair_decomposed_instance_normalization_layouts(model_ir)

    assert stats == {"repaired_decomposed_instance_normalization_layouts": 0}
    assert repr(model_ir) == before


def test_instance_normalization_layout_skips_index_without_marked_mean(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_instance_normalization_layout")
    model_ir.tensors["x"] = _tensor("x", shape=[1, 2])
    model_ir.tensors["y"] = _tensor("y", shape=[1, 2])
    model_ir.operators = [OperatorIR("MEAN", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(
        instance_norm_module,
        "ModelIRGraphIndex",
        unexpected_index,
    )

    stats = _repair_decomposed_instance_normalization_layouts(model_ir)

    assert stats == {"repaired_decomposed_instance_normalization_layouts": 0}
