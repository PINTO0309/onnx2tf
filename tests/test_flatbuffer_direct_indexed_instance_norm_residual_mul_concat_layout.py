from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_norm_residual_mul_concat_layout as tail_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.instance_norm_residual_mul_concat_layout import (
    optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains,
)
from tests.test_flatbuffer_direct_indexed_instance_norm_post_bias_layout import (
    _assert_index_current,
    _build_model,
    _coefficient,
    _operator,
    _tensor,
)


_STATS = "optimized_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"


def _build_tail_model(
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    produced_source: bool = False,
    produced_residual: bool = False,
    separate_axes: bool = False,
    axes: tuple[int, int] = (2, 3),
    commuted_sub: bool = False,
    commuted_affine: bool = False,
    scale_mode: str = "nchw",
    bias_mode: str = "nchw",
    tail_modes: tuple[str, str] = ("nchw", "nchw"),
    reverse_concat: bool = False,
) -> tuple[ModelIR, dict[str, str]]:
    model_ir, names = _build_model(
        prefix=prefix,
        dtype=dtype,
        produced_source=produced_source,
        separate_axes=separate_axes,
        axes=axes,
        commuted_sub=commuted_sub,
        commuted_affine=commuted_affine,
        scale_mode=scale_mode,
        bias_mode=bias_mode,
    )
    names.update(
        {
            key: f"{prefix}{key}"
            for key in (
                "residual_upstream",
                "residual",
                "add_output",
                "tail0",
                "tail1",
                "mul0",
                "mul1",
                "concat_output",
                "tail_post_output",
            )
        }
    )
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    source_shape = [1, 2, 4, 3]
    nchw_shape = [1, 3, 2, 4]
    concat_nchw_shape = [1, 6, 2, 4]
    concat_nhwc_shape = [1, 2, 4, 6]

    post = _operator(model_ir, names["post_output"])
    add_bias = _operator(model_ir, names["inst_output"])
    downstream = _operator(model_ir, names["output"])
    model_ir.operators.remove(post)
    model_ir.operators.remove(downstream)
    add_bias.inputs[add_bias.inputs.index(names["post_output"])] = names["scaled"]
    model_ir.tensors[names["inst_output"]].shape = list(nchw_shape)
    model_ir.tensors[names["inst_output"]].shape_signature = list(nchw_shape)
    post.inputs = [names["inst_output"], names["post_perm"]]
    post.outputs = [names["post_output"]]
    add_bias_index = model_ir.operators.index(add_bias)
    model_ir.operators.insert(add_bias_index + 1, post)

    model_ir.tensors[names["residual"]] = _tensor(
        names["residual"], source_shape, dtype=dtype
    )
    if produced_residual:
        model_ir.tensors[names["residual_upstream"]] = _tensor(
            names["residual_upstream"], source_shape, dtype=dtype
        )
        model_ir.inputs.append(names["residual_upstream"])
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [names["residual_upstream"]],
                [names["residual"]],
            )
        )
    else:
        model_ir.inputs.append(names["residual"])

    for name, shape in {
        names["add_output"]: source_shape,
        names["mul0"]: nchw_shape,
        names["mul1"]: nchw_shape,
        names["concat_output"]: concat_nchw_shape,
        names["tail_post_output"]: concat_nhwc_shape,
        names["output"]: concat_nhwc_shape,
    }.items():
        model_ir.tensors[name] = _tensor(name, shape, dtype=dtype)
    tail_values = (
        [1.0, -0.5, 0.25],
        [0.125, 0.5, -1.5],
    )
    for key, values, mode in zip(("tail0", "tail1"), tail_values, tail_modes):
        data = _coefficient(values, dtype=np_dtype, mode=mode)
        model_ir.tensors[names[key]] = _tensor(
            names[key],
            list(data.shape),
            dtype=dtype,
            data=data,
        )

    def _binary(left: str, right: str) -> list[str]:
        values = [left, right]
        if commuted_affine:
            values.reverse()
        return values

    mul_outputs = [names["mul0"], names["mul1"]]
    if reverse_concat:
        mul_outputs.reverse()
    model_ir.operators.extend(
        [
            OperatorIR(
                "ADD",
                _binary(names["post_output"], names["residual"]),
                [names["add_output"]],
            ),
            OperatorIR(
                "MUL",
                _binary(names["add_output"], names["tail0"]),
                [names["mul0"]],
            ),
            OperatorIR(
                "MUL",
                _binary(names["add_output"], names["tail1"]),
                [names["mul1"]],
            ),
            OperatorIR(
                "CONCATENATION",
                mul_outputs,
                [names["concat_output"]],
                {"axis": 1, "fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["concat_output"], names["post_perm"]],
                [names["tail_post_output"]],
            ),
            OperatorIR(
                "RELU",
                [names["tail_post_output"]],
                [names["output"]],
            ),
        ]
    )
    model_ir.outputs = [names["output"]]
    return model_ir, names


def _logical_channel_coefficient(
    model_ir: ModelIR,
    name: str,
    channel_count: int,
) -> np.ndarray:
    data = np.asarray(model_ir.tensors[name].data)
    if data.size == 1:
        return data.reshape(1, 1, 1, 1)
    assert data.size == channel_count
    return data.reshape(1, 1, 1, channel_count)


def _reference_output(
    model_ir: ModelIR,
    names: dict[str, str],
    feeds: dict[str, np.ndarray],
) -> np.ndarray:
    source_feed = names["upstream"] if names["upstream"] in feeds else names["source"]
    residual_feed = (
        names["residual_upstream"]
        if names["residual_upstream"] in feeds
        else names["residual"]
    )
    source = np.asarray(feeds[source_feed])
    residual = np.asarray(feeds[residual_feed])
    x_nchw = np.transpose(source, (0, 3, 1, 2))
    axes = tuple(
        int(value) % 4
        for value in np.asarray(model_ir.tensors[names["axes1"]].data).reshape(-1)
    )
    mean = np.mean(x_nchw, axis=axes, keepdims=True)
    sub = _operator(model_ir, names["centered"])
    centered = x_nchw - mean if sub.inputs[0] == names["x"] else mean - x_nchw
    squared = centered * centered
    axes2 = tuple(
        int(value) % 4
        for value in np.asarray(model_ir.tensors[names["axes2"]].data).reshape(-1)
    )
    variance = np.mean(squared, axis=axes2, keepdims=True)
    epsilon = np.asarray(model_ir.tensors[names["epsilon"]].data)
    inverse_std = np.asarray(model_ir.tensors[names["one"]].data) / np.sqrt(
        variance + epsilon
    )
    normalized = centered * inverse_std
    scale = _logical_channel_coefficient(model_ir, names["scale"], 3)
    bias = _logical_channel_coefficient(model_ir, names["bias"], 3)
    scale_nchw = np.transpose(scale, (0, 3, 1, 2))
    bias_nchw = np.transpose(bias, (0, 3, 1, 2))
    inst_nhwc = np.transpose(normalized * scale_nchw + bias_nchw, (0, 2, 3, 1))
    add_output = inst_nhwc + residual
    tail_values = {
        names["mul0"]: add_output
        * _logical_channel_coefficient(model_ir, names["tail0"], 3),
        names["mul1"]: add_output
        * _logical_channel_coefficient(model_ir, names["tail1"], 3),
    }
    concat = _operator(model_ir, names["concat_output"])
    return np.maximum(
        np.concatenate([tail_values[str(name)] for name in concat.inputs], axis=3),
        0,
    )


def _evaluate_rewritten(
    model_ir: ModelIR,
    feeds: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({name: np.asarray(value) for name, value in feeds.items()})
    for operator in model_ir.operators:
        inputs = [values[str(name)] for name in operator.inputs]
        if operator.op_type == "IDENTITY":
            output = inputs[0]
        elif operator.op_type == "TRANSPOSE":
            output = np.transpose(
                inputs[0],
                tuple(int(value) for value in inputs[1].reshape(-1)),
            )
        elif operator.op_type == "MEAN":
            output = np.mean(
                inputs[0],
                axis=tuple(int(value) for value in inputs[1].reshape(-1)),
                keepdims=bool(operator.options["keepDims"]),
            )
        elif operator.op_type == "SUB":
            output = inputs[0] - inputs[1]
        elif operator.op_type == "MUL":
            output = inputs[0] * inputs[1]
        elif operator.op_type == "ADD":
            output = inputs[0] + inputs[1]
        elif operator.op_type == "SQRT":
            output = np.sqrt(inputs[0])
        elif operator.op_type == "DIV":
            output = inputs[0] / inputs[1]
        elif operator.op_type == "CONCATENATION":
            output = np.concatenate(inputs, axis=int(operator.options["axis"]))
        elif operator.op_type == "RELU":
            output = np.maximum(inputs[0], 0)
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("produced_residual", (False, True))
@pytest.mark.parametrize(
    "separate_axes,axes,commuted_sub,commuted_affine,scale_mode,bias_mode,tail_modes,reverse_concat",
    (
        (False, (2, 3), False, False, "nchw", "nchw", ("nchw", "nchw"), False),
        (False, (-2, -1), True, True, "scalar", "nchw", ("scalar", "nchw"), True),
        (True, (3, 2), False, True, "nchw", "scalar", ("nchw", "scalar"), False),
        (True, (-1, -2), True, False, "nchw", "nchw", ("nhwc", "nchw"), True),
    ),
)
def test_residual_mul_concat_instance_norm_is_indexed_and_equivalent(
    dtype: str,
    produced_source: bool,
    produced_residual: bool,
    separate_axes: bool,
    axes: tuple[int, int],
    commuted_sub: bool,
    commuted_affine: bool,
    scale_mode: str,
    bias_mode: str,
    tail_modes: tuple[str, str],
    reverse_concat: bool,
) -> None:
    model_ir, names = _build_tail_model(
        dtype=dtype,
        produced_source=produced_source,
        produced_residual=produced_residual,
        separate_axes=separate_axes,
        axes=axes,
        commuted_sub=commuted_sub,
        commuted_affine=commuted_affine,
        scale_mode=scale_mode,
        bias_mode=bias_mode,
        tail_modes=tail_modes,
        reverse_concat=reverse_concat,
    )
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    rng = np.random.default_rng(97)
    main_feed = names["upstream"] if produced_source else names["source"]
    residual_feed = (
        names["residual_upstream"] if produced_residual else names["residual"]
    )
    feeds = {
        main_feed: rng.normal(size=[1, 2, 4, 3]).astype(np_dtype),
        residual_feed: rng.normal(size=[1, 2, 4, 3]).astype(np_dtype),
    }
    expected = _reference_output(copy.deepcopy(model_ir), names, feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    assert _operator(model_ir, names["mean1"]).inputs[0] == names["source"]
    assert names["source"] in _operator(model_ir, names["centered"]).inputs
    assert names["inst_output"] in _operator(model_ir, names["add_output"]).inputs
    concat = _operator(model_ir, names["tail_post_output"])
    assert concat.op_type == "CONCATENATION"
    assert concat.options == {"axis": 3, "fusedActivationFunction": "NONE"}
    assert [str(value) for value in concat.inputs] == (
        [names["mul1"], names["mul0"]]
        if reverse_concat
        else [names["mul0"], names["mul1"]]
    )
    assert list(model_ir.tensors[names["inst_output"]].shape) == [1, 2, 4, 3]
    assert list(model_ir.tensors[names["mul0"]].shape) == [1, 2, 4, 3]
    for name in (names["scale"], names["bias"], names["tail0"], names["tail1"]):
        shape = list(model_ir.tensors[name].shape)
        assert shape in ([1], [1, 1, 1, 3])
    actual = _evaluate_rewritten(model_ir, feeds)[names["output"]]
    tolerance = 4e-3 if dtype == "FLOAT16" else 1e-6
    np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert (
        names["tail_post_output"]
        in model_ir.metadata["assume_channel_last_layout_tensor_names"]
    )


def test_residual_mul_concat_accepts_scalar_backing_for_shape_one_constants() -> None:
    model_ir, names = _build_tail_model()
    for name in (names["epsilon"], names["one"]):
        tensor = model_ir.tensors[name]
        assert tensor.shape == [1]
        tensor.data = np.asarray(tensor.data).reshape(())

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir,
        graph_index=ModelIRGraphIndex(model_ir),
        layout_state=LayoutState.from_model_ir(model_ir),
    )

    assert stats == {_STATS: 1}
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)


def _set_dynamic_signature(
    model_ir: ModelIR,
    names: dict[str, str],
    dynamic_axis: int,
) -> None:
    source_signature = [1, 2, 4, 3]
    source_signature[dynamic_axis] = -1
    nchw_signature = [source_signature[index] for index in (0, 3, 1, 2)]
    reduced_signature = [nchw_signature[0], nchw_signature[1], 1, 1]
    for key in ("source", "residual", "post_output", "add_output"):
        model_ir.tensors[names[key]].shape_signature = list(source_signature)
    for key in ("upstream", "residual_upstream"):
        if names[key] in model_ir.tensors:
            model_ir.tensors[names[key]].shape_signature = list(source_signature)
    for key in ("x", "centered", "squared", "normalized", "scaled", "inst_output"):
        model_ir.tensors[names[key]].shape_signature = list(nchw_signature)
    for key in ("mean1", "mean2", "variance_epsilon", "std", "inverse_std"):
        model_ir.tensors[names[key]].shape_signature = list(reduced_signature)
    for key in ("mul0", "mul1"):
        model_ir.tensors[names[key]].shape_signature = list(nchw_signature)
    concat_signature = list(nchw_signature)
    concat_signature[1] = -1 if nchw_signature[1] == -1 else 6
    model_ir.tensors[names["concat_output"]].shape_signature = concat_signature
    output_signature = [concat_signature[index] for index in (0, 2, 3, 1)]
    for key in ("tail_post_output", "output"):
        model_ir.tensors[names[key]].shape_signature = list(output_signature)


@pytest.mark.parametrize("dynamic_axis", (1, 2, 3))
def test_residual_mul_concat_preserves_dynamic_signatures(dynamic_axis: int) -> None:
    model_ir, names = _build_tail_model(
        produced_source=True,
        produced_residual=True,
    )
    _set_dynamic_signature(model_ir, names, dynamic_axis)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {_STATS: 1}
    source_signature = list(model_ir.tensors[names["source"]].shape_signature)
    assert (
        list(model_ir.tensors[names["inst_output"]].shape_signature) == source_signature
    )
    assert list(model_ir.tensors[names["mul0"]].shape_signature) == source_signature
    expected_output_signature = list(source_signature)
    expected_output_signature[3] = -1 if source_signature[3] == -1 else 6
    assert list(model_ir.tensors[names["tail_post_output"]].shape_signature) == (
        expected_output_signature
    )
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)


@pytest.mark.parametrize("repeated_slots", (False, True))
def test_residual_mul_concat_preserves_downstream_fanout(
    repeated_slots: bool,
) -> None:
    model_ir, names = _build_tail_model()
    downstream = _operator(model_ir, names["output"])
    if repeated_slots:
        downstream.op_type = "ADD"
        downstream.inputs = [names["tail_post_output"], names["tail_post_output"]]
    side_name = "tail_side"
    model_ir.tensors[side_name] = _tensor(side_name, [1, 2, 4, 6])
    model_ir.operators.append(
        OperatorIR("ABS", [names["tail_post_output"]], [side_name])
    )
    model_ir.outputs.append(side_name)

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    concat = _operator(model_ir, names["tail_post_output"])
    assert concat.op_type == "CONCATENATION"
    assert all(name == names["tail_post_output"] for name in downstream.inputs)
    assert _operator(model_ir, side_name).inputs == [names["tail_post_output"]]
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_mul_concat_preserves_public_tail_output() -> None:
    model_ir, names = _build_tail_model()
    downstream = _operator(model_ir, names["output"])
    model_ir.operators.remove(downstream)
    model_ir.outputs = [names["tail_post_output"]]

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert _operator(model_ir, names["tail_post_output"]).op_type == "CONCATENATION"
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_mul_concat_clones_shared_changed_tail_constant() -> None:
    model_ir, names = _build_tail_model()
    side_name = "tail_constant_side"
    model_ir.tensors[side_name] = _tensor(side_name, [1, 3, 1, 1])
    model_ir.operators.append(OperatorIR("IDENTITY", [names["tail0"]], [side_name]))
    model_ir.outputs.append(side_name)

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    tail_mul = _operator(model_ir, names["mul0"])
    clone_name = next(name for name in tail_mul.inputs if name != names["add_output"])
    assert clone_name != names["tail0"]
    assert list(model_ir.tensors[clone_name].shape) == [1, 1, 1, 3]
    assert list(model_ir.tensors[names["tail0"]].shape) == [1, 3, 1, 1]
    assert _operator(model_ir, side_name).inputs == [names["tail0"]]
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_mul_concat_updates_one_coefficient_shared_by_all_uses() -> None:
    model_ir, names = _build_tail_model()
    shared_name = names["scale"]
    for output_name in (names["inst_output"], names["mul0"], names["mul1"]):
        operator = _operator(model_ir, output_name)
        data_input = (
            names["scaled"]
            if output_name == names["inst_output"]
            else names["add_output"]
        )
        operator.inputs = [
            data_input if name == data_input else shared_name
            for name in operator.inputs
        ]
    for name in (names["bias"], names["tail0"], names["tail1"]):
        del model_ir.tensors[name]

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert list(model_ir.tensors[shared_name].shape) == [1, 1, 1, 3]
    assert all(
        shared_name in _operator(model_ir, output_name).inputs
        for output_name in (
            names["scaled"],
            names["inst_output"],
            names["mul0"],
            names["mul1"],
        )
    )
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_mul_concat_rewrites_multiple_chains_with_total_cap() -> None:
    first, first_names = _build_tail_model(prefix="a_")
    second, second_names = _build_tail_model(prefix="b_")
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    graph_index = ModelIRGraphIndex(first)

    first_stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        first,
        graph_index=graph_index,
        max_rewrites=1,
    )
    second_stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        first,
        graph_index=graph_index,
        max_rewrites=1,
    )

    assert first_stats == {_STATS: 1}
    assert second_stats == {_STATS: 1}
    assert _operator(first, first_names["mean1"]).inputs[0] == first_names["source"]
    assert _operator(first, second_names["mean1"]).inputs[0] == second_names["source"]
    _assert_index_current(first, graph_index)


@pytest.mark.parametrize(
    "case",
    (
        "pre_perm",
        "pre_perm_quantized",
        "source_public_output",
        "source_unbound",
        "pre_public",
        "pre_shape",
        "duplicate_pre",
        "mean_axes",
        "negative_epsilon",
        "nonunit_numerator",
        "core_quantized",
        "bias_shape",
        "bias_dtype",
        "bias_public",
        "bias_produced",
        "inst_public",
        "inst_fanout",
        "inst_shape",
        "post_perm",
        "post_perm_quantized",
        "post_public",
        "post_fanout",
        "post_shape",
        "tail_add_op",
        "residual_shape",
        "residual_dtype",
        "residual_unbound",
        "add_public",
        "add_quantized",
        "add_missing_consumer",
        "add_repeated_slot",
        "tail_mul_op",
        "tail_mul_shape",
        "tail_mul_dtype",
        "tail_mul_public",
        "tail_mul_fanout",
        "tail0_shape",
        "tail0_dtype",
        "tail0_quantized",
        "tail0_public",
        "tail0_produced",
        "tail1_nonfinite",
        "concat_inputs",
        "concat_extra_input",
        "concat_axis",
        "concat_axis_type",
        "concat_shape",
        "concat_public",
        "tail_post_op",
        "tail_post_perm",
        "tail_post_shape",
        "tail_post_dtype",
        "tail_post_public_input",
        "duplicate_tail_output",
        "backward_tail_post",
        "backward_downstream",
        "backward_tail_mul",
    ),
)
def test_residual_mul_concat_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_tail_model()
    tail_add = _operator(model_ir, names["add_output"])
    tail_mul0 = _operator(model_ir, names["mul0"])
    tail_mul1 = _operator(model_ir, names["mul1"])
    concat = _operator(model_ir, names["concat_output"])
    tail_post = _operator(model_ir, names["tail_post_output"])
    downstream = _operator(model_ir, names["output"])
    if case == "pre_perm":
        model_ir.tensors[names["pre_perm"]].data[:] = [0, 2, 3, 1]
    elif case == "pre_perm_quantized":
        model_ir.tensors[names["pre_perm"]].quantization = {"scale": [1.0]}
    elif case == "source_public_output":
        model_ir.outputs.append(names["source"])
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "pre_public":
        model_ir.outputs.append(names["x"])
    elif case == "pre_shape":
        model_ir.tensors[names["x"]].shape[2] = 3
    elif case == "duplicate_pre":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["x"]])
        )
    elif case == "mean_axes":
        model_ir.tensors[names["axes1"]].data[:] = [1, 2]
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    elif case == "nonunit_numerator":
        model_ir.tensors[names["one"]].data[0] = 2.0
    elif case == "core_quantized":
        model_ir.tensors[names["centered"]].quantization = {"scale": [0.25]}
    elif case == "bias_shape":
        model_ir.tensors[names["bias"]].shape = [3]
        model_ir.tensors[names["bias"]].shape_signature = [3]
        model_ir.tensors[names["bias"]].data = np.asarray(
            [0.1, 0.2, 0.3], dtype=np.float32
        )
    elif case == "bias_dtype":
        model_ir.tensors[names["bias"]].dtype = "FLOAT16"
        model_ir.tensors[names["bias"]].data = np.asarray(
            model_ir.tensors[names["bias"]].data, dtype=np.float16
        )
    elif case == "bias_public":
        model_ir.inputs.append(names["bias"])
    elif case == "bias_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["bias"]])
        )
    elif case == "inst_public":
        model_ir.outputs.append(names["inst_output"])
    elif case == "inst_fanout":
        model_ir.tensors["inst_side"] = _tensor("inst_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inst_output"]], ["inst_side"])
        )
    elif case == "inst_shape":
        model_ir.tensors[names["inst_output"]].shape[2] = 3
    elif case == "post_perm":
        model_ir.tensors[names["post_perm"]].data[:] = [0, 3, 1, 2]
    elif case == "post_perm_quantized":
        model_ir.tensors[names["post_perm"]].quantization = {"scale": [1.0]}
    elif case == "post_public":
        model_ir.outputs.append(names["post_output"])
    elif case == "post_fanout":
        model_ir.tensors["post_side"] = _tensor("post_side", [1, 2, 4, 3])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["post_output"]], ["post_side"])
        )
    elif case == "post_shape":
        model_ir.tensors[names["post_output"]].shape[1] = 3
    elif case == "tail_add_op":
        tail_add.op_type = "SUB"
    elif case == "residual_shape":
        model_ir.tensors[names["residual"]].shape[1] = 3
    elif case == "residual_dtype":
        model_ir.tensors[names["residual"]].dtype = "FLOAT16"
    elif case == "residual_unbound":
        model_ir.inputs.remove(names["residual"])
    elif case == "add_public":
        model_ir.outputs.append(names["add_output"])
    elif case == "add_quantized":
        model_ir.tensors[names["add_output"]].quantization = {"scale": [0.25]}
    elif case == "add_missing_consumer":
        model_ir.operators.remove(tail_mul1)
    elif case == "add_repeated_slot":
        tail_mul0.inputs = [names["add_output"], names["add_output"]]
    elif case == "tail_mul_op":
        tail_mul0.op_type = "ADD"
    elif case == "tail_mul_shape":
        model_ir.tensors[names["mul0"]].shape[2] = 3
    elif case == "tail_mul_dtype":
        model_ir.tensors[names["mul0"]].dtype = "FLOAT16"
    elif case == "tail_mul_public":
        model_ir.outputs.append(names["mul0"])
    elif case == "tail_mul_fanout":
        model_ir.tensors["mul_side"] = _tensor("mul_side", [1, 3, 2, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["mul0"]], ["mul_side"]))
    elif case == "tail0_shape":
        model_ir.tensors[names["tail0"]].shape = [3]
        model_ir.tensors[names["tail0"]].shape_signature = [3]
        model_ir.tensors[names["tail0"]].data = np.asarray(
            [1.0, 2.0, 3.0], dtype=np.float32
        )
    elif case == "tail0_dtype":
        model_ir.tensors[names["tail0"]].dtype = "FLOAT16"
        model_ir.tensors[names["tail0"]].data = np.asarray(
            model_ir.tensors[names["tail0"]].data, dtype=np.float16
        )
    elif case == "tail0_quantized":
        model_ir.tensors[names["tail0"]].quantization = {"scale": [0.25]}
    elif case == "tail0_public":
        model_ir.inputs.append(names["tail0"])
    elif case == "tail0_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["tail0"]])
        )
    elif case == "tail1_nonfinite":
        model_ir.tensors[names["tail1"]].data.reshape(-1)[0] = np.nan
    elif case == "concat_inputs":
        concat.inputs = [names["mul0"], names["mul0"]]
    elif case == "concat_extra_input":
        concat.inputs.append(names["mul0"])
    elif case == "concat_axis":
        concat.options["axis"] = 2
    elif case == "concat_axis_type":
        concat.options["axis"] = "channel"
    elif case == "concat_shape":
        model_ir.tensors[names["concat_output"]].shape[1] = 5
    elif case == "concat_public":
        model_ir.outputs.append(names["concat_output"])
    elif case == "tail_post_op":
        tail_post.op_type = "RESHAPE"
    elif case == "tail_post_perm":
        model_ir.tensors[names["post_perm"]].data[:] = [0, 3, 1, 2]
    elif case == "tail_post_shape":
        model_ir.tensors[names["tail_post_output"]].shape[3] = 5
    elif case == "tail_post_dtype":
        model_ir.tensors[names["tail_post_output"]].dtype = "FLOAT16"
    elif case == "tail_post_public_input":
        model_ir.inputs.append(names["tail_post_output"])
    elif case == "duplicate_tail_output":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["tail_post_output"]])
        )
    elif case == "backward_tail_post":
        model_ir.operators.remove(tail_post)
        model_ir.operators.insert(model_ir.operators.index(concat), tail_post)
    elif case == "backward_downstream":
        model_ir.operators.remove(downstream)
        model_ir.operators.insert(model_ir.operators.index(tail_post), downstream)
    elif case == "backward_tail_mul":
        model_ir.operators.remove(tail_mul0)
        model_ir.operators.insert(model_ir.operators.index(tail_add), tail_mul0)
    model_ir.tensors["unused"] = _tensor("unused", [1])
    before = repr(model_ir)

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_residual_mul_concat_clone_collision_is_transactional(monkeypatch) -> None:
    model_ir, names = _build_tail_model()
    model_ir.tensors["scale_side"] = _tensor("scale_side", [1, 3, 1, 1])
    model_ir.operators.append(OperatorIR("IDENTITY", [names["scale"]], ["scale_side"]))
    model_ir.outputs.append("scale_side")
    original_resolve = tail_module._resolve_candidate
    injected = False

    def _resolve_with_collision(*args, **kwargs):
        nonlocal injected
        plan = original_resolve(*args, **kwargs)
        if plan is None or injected:
            return plan
        clone_name = next(
            update.clone_name
            for update in plan.constant_updates
            if update.clone_name is not None
        )
        assert clone_name is not None
        model_ir.tensors[clone_name] = _tensor(clone_name, [1])
        injected = True
        return plan

    monkeypatch.setattr(tail_module, "_resolve_candidate", _resolve_with_collision)
    before_ops = repr(model_ir.operators)
    before_scale = np.asarray(model_ir.tensors[names["scale"]].data).copy()

    stats = optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir.operators) == before_ops
    np.testing.assert_array_equal(model_ir.tensors[names["scale"]].data, before_scale)


def test_residual_mul_concat_preflight_avoids_index_construction(
    monkeypatch,
) -> None:
    model_ir = ModelIR("insufficient_residual_mul_concat_topology")

    def _unexpected_index(*args, **kwargs):
        raise AssertionError("index must not be constructed")

    monkeypatch.setattr(tail_module, "ModelIRGraphIndex", _unexpected_index)

    assert optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains(
        model_ir
    ) == {_STATS: 0}
