from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_norm_prepost_layout as direct_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_instancenorm_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.instance_norm_prepost_layout import (
    _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains,
)


_STATS = (
    "optimized_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains"
)
_COMPAT_STATS = "optimized_transpose_instancenorm_prepost_nhwc_chains"


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data=None,
    signature: list[int] | None = None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _build_model(
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    produced_source: bool = False,
    separate_axes: bool = False,
    negative_axes: bool = False,
    commuted_affine: bool = False,
) -> tuple[ModelIR, dict[str, str]]:
    model_ir = ModelIR(f"{prefix}instance_norm_direct")
    names = {
        key: f"{prefix}{key}"
        for key in (
            "upstream",
            "source",
            "pre_perm",
            "pre",
            "squeezed",
            "reshape_shape",
            "x",
            "axes1",
            "axes2",
            "mean1",
            "centered",
            "squared",
            "mean2",
            "epsilon",
            "variance_epsilon",
            "std",
            "one",
            "inverse_std",
            "normalized",
            "scale",
            "scaled",
            "bias",
            "inst_output",
            "post_perm",
            "post_output",
            "output",
        )
    }
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    n, height, width, channels = 1, 2, 4, 3
    source_shape = [n, height, width, channels]
    nchw_shape = [n, channels, height, width]
    reduced_shape = [n, channels, 1, 1]
    for name, shape in {
        names["source"]: source_shape,
        names["pre"]: nchw_shape,
        names["squeezed"]: [channels, height, width],
        names["x"]: nchw_shape,
        names["mean1"]: reduced_shape,
        names["centered"]: nchw_shape,
        names["squared"]: nchw_shape,
        names["mean2"]: reduced_shape,
        names["variance_epsilon"]: reduced_shape,
        names["std"]: reduced_shape,
        names["inverse_std"]: reduced_shape,
        names["normalized"]: nchw_shape,
        names["scaled"]: nchw_shape,
        names["inst_output"]: nchw_shape,
        names["post_output"]: source_shape,
        names["output"]: source_shape,
    }.items():
        model_ir.tensors[name] = _tensor(name, shape, dtype=dtype)
    model_ir.tensors[names["pre_perm"]] = _tensor(
        names["pre_perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors[names["post_perm"]] = _tensor(
        names["post_perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors[names["reshape_shape"]] = _tensor(
        names["reshape_shape"],
        [4],
        dtype="INT64",
        data=np.asarray(nchw_shape, dtype=np.int64),
    )
    axes = [-2, -1] if negative_axes else [2, 3]
    model_ir.tensors[names["axes1"]] = _tensor(
        names["axes1"],
        [2],
        dtype="INT64",
        data=np.asarray(axes, dtype=np.int64),
    )
    if separate_axes:
        model_ir.tensors[names["axes2"]] = _tensor(
            names["axes2"],
            [2],
            dtype="INT32",
            data=np.asarray(list(reversed(axes)), dtype=np.int32),
        )
    else:
        names["axes2"] = names["axes1"]
    model_ir.tensors[names["epsilon"]] = _tensor(
        names["epsilon"],
        [1],
        dtype=dtype,
        data=np.asarray([0.01], dtype=np_dtype),
    )
    model_ir.tensors[names["one"]] = _tensor(
        names["one"],
        [1],
        dtype=dtype,
        data=np.asarray([1.0], dtype=np_dtype),
    )
    model_ir.tensors[names["scale"]] = _tensor(
        names["scale"],
        [1, channels, 1, 1],
        dtype=dtype,
        data=np.asarray([0.75, 1.25, -0.5], dtype=np_dtype).reshape(
            1, channels, 1, 1
        ),
    )
    model_ir.tensors[names["bias"]] = _tensor(
        names["bias"],
        [1, channels, 1, 1],
        dtype=dtype,
        data=np.asarray([0.1, -0.2, 0.3], dtype=np_dtype).reshape(
            1, channels, 1, 1
        ),
    )
    if produced_source:
        model_ir.inputs = [names["upstream"]]
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            source_shape,
            dtype=dtype,
        )
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [names["upstream"]],
                [names["source"]],
            )
        )
    else:
        model_ir.inputs = [names["source"]]

    def _binary_inputs(data_name: str, constant_name: str) -> list[str]:
        values = [data_name, constant_name]
        if commuted_affine:
            values.reverse()
        return values

    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["source"], names["pre_perm"]],
                [names["pre"]],
            ),
            OperatorIR(
                "SQUEEZE",
                [names["pre"]],
                [names["squeezed"]],
                {"squeezeDims": [0], "marker": "squeeze"},
            ),
            OperatorIR(
                "RESHAPE",
                [names["squeezed"], names["reshape_shape"]],
                [names["x"]],
                {"newShape": nchw_shape, "marker": "reshape"},
            ),
            OperatorIR(
                "MEAN",
                [names["x"], names["axes1"]],
                [names["mean1"]],
                {"keepDims": True, "marker": "mean1"},
                axis_semantics={"marker": "preserved"},
                version=2,
                onnx_node_name="instance_norm_mean1",
                onnx_op_type="InstanceNormalization",
            ),
            OperatorIR(
                "SUB",
                [names["x"], names["mean1"]],
                [names["centered"]],
            ),
            OperatorIR(
                "MUL",
                [names["centered"], names["centered"]],
                [names["squared"]],
            ),
            OperatorIR(
                "MEAN",
                [names["squared"], names["axes2"]],
                [names["mean2"]],
                {"keepDims": True, "marker": "mean2"},
            ),
            OperatorIR(
                "ADD",
                _binary_inputs(names["mean2"], names["epsilon"]),
                [names["variance_epsilon"]],
            ),
            OperatorIR("SQRT", [names["variance_epsilon"]], [names["std"]]),
            OperatorIR(
                "DIV",
                [names["one"], names["std"]],
                [names["inverse_std"]],
            ),
            OperatorIR(
                "MUL",
                _binary_inputs(names["centered"], names["inverse_std"]),
                [names["normalized"]],
            ),
            OperatorIR(
                "MUL",
                _binary_inputs(names["normalized"], names["scale"]),
                [names["scaled"]],
            ),
            OperatorIR(
                "ADD",
                _binary_inputs(names["scaled"], names["bias"]),
                [names["inst_output"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["inst_output"], names["post_perm"]],
                [names["post_output"]],
            ),
            OperatorIR("RELU", [names["post_output"]], [names["output"]]),
        ]
    )
    model_ir.outputs = [names["output"]]
    model_ir.metadata["assume_channel_last_layout_tensor_names"] = [
        "existing_hint"
    ]
    return model_ir, names


def _evaluate(
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
        inputs = [values[name] for name in operator.inputs]
        if operator.op_type == "IDENTITY":
            output = inputs[0]
        elif operator.op_type == "TRANSPOSE":
            output = np.transpose(
                inputs[0],
                tuple(int(value) for value in inputs[1].reshape(-1)),
            )
        elif operator.op_type == "SQUEEZE":
            axes = tuple(
                int(value)
                for value in np.asarray(
                    operator.options["squeezeDims"]
                ).reshape(-1)
            )
            output = np.squeeze(inputs[0], axis=axes)
        elif operator.op_type == "RESHAPE":
            output = np.reshape(
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
        elif operator.op_type == "RELU":
            output = np.maximum(inputs[0], 0)
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


def _operator(model_ir: ModelIR, output_name: str) -> OperatorIR:
    return next(
        operator
        for operator in model_ir.operators
        if output_name in operator.outputs
    )


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


def _replace_tail(model_ir: ModelIR, names: dict[str, str], mode: str) -> None:
    post = _operator(model_ir, names["post_output"])
    post_index = model_ir.operators.index(post)
    if mode == "side_squeeze":
        model_ir.tensors["side_output"] = _tensor(
            "side_output",
            [3, 2, 4],
        )
        model_ir.operators.append(
            OperatorIR(
                "SQUEEZE",
                [names["inst_output"]],
                ["side_output"],
                {"squeezeDims": [0]},
            )
        )
        model_ir.outputs.append("side_output")
        return

    model_ir.operators.remove(post)
    model_ir.tensors["tail_squeezed"] = _tensor(
        "tail_squeezed",
        [3, 2, 4],
    )
    tail_ops = [
        OperatorIR(
            "SQUEEZE",
            [names["inst_output"]],
            ["tail_squeezed"],
            {"squeezeDims": [0]},
        )
    ]
    tail_data_name = "tail_squeezed"
    if mode == "squeeze_unary_reshape":
        model_ir.tensors["tail_unary"] = _tensor("tail_unary", [3, 2, 4])
        tail_ops.append(
            OperatorIR("RELU", ["tail_squeezed"], ["tail_unary"])
        )
        tail_data_name = "tail_unary"
    elif mode == "squeeze_add_reshape":
        model_ir.inputs.append("residual_source")
        model_ir.tensors["residual_source"] = _tensor(
            "residual_source",
            [2, 4, 3],
        )
        model_ir.tensors["residual_perm"] = _tensor(
            "residual_perm",
            [3],
            dtype="INT32",
            data=np.asarray([2, 0, 1], dtype=np.int32),
        )
        model_ir.tensors["residual_chw"] = _tensor(
            "residual_chw",
            [3, 2, 4],
        )
        model_ir.tensors["tail_add"] = _tensor("tail_add", [3, 2, 4])
        tail_ops.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    ["residual_source", "residual_perm"],
                    ["residual_chw"],
                ),
                OperatorIR(
                    "ADD",
                    ["tail_squeezed", "residual_chw"],
                    ["tail_add"],
                ),
            ]
        )
        tail_data_name = "tail_add"
    else:
        raise AssertionError(f"unsupported legacy tail mode: {mode}")
    model_ir.tensors["tail_shape"] = _tensor(
        "tail_shape",
        [4],
        dtype="INT64",
        data=np.asarray([1, 3, 2, 4], dtype=np.int64),
    )
    model_ir.tensors["tail_reshape"] = _tensor(
        "tail_reshape",
        [1, 3, 2, 4],
    )
    tail_ops.append(
        OperatorIR(
            "RESHAPE",
            [tail_data_name, "tail_shape"],
            ["tail_reshape"],
            {"newShape": [1, 3, 2, 4]},
        )
    )
    post.inputs[0] = "tail_reshape"
    tail_ops.append(post)
    for offset, operator in enumerate(tail_ops):
        model_ir.operators.insert(post_index + offset, operator)


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("separate_axes", (False, True))
@pytest.mark.parametrize("negative_axes", (False, True))
@pytest.mark.parametrize("commuted_affine", (False, True))
def test_direct_instance_norm_rewrite_is_indexed_and_numerically_equivalent(
    dtype: str,
    produced_source: bool,
    separate_axes: bool,
    negative_axes: bool,
    commuted_affine: bool,
) -> None:
    model_ir, names = _build_model(
        dtype=dtype,
        produced_source=produced_source,
        separate_axes=separate_axes,
        negative_axes=negative_axes,
        commuted_affine=commuted_affine,
    )
    feed_name = names["upstream"] if produced_source else names["source"]
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    feeds = {
        feed_name: np.random.default_rng(59)
        .normal(size=model_ir.tensors[feed_name].shape)
        .astype(np_dtype)
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    original_mean = copy.deepcopy(_operator(model_ir, names["mean1"]))
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    actual = _evaluate(model_ir, feeds)
    tolerance = 1e-2 if dtype == "FLOAT16" else 1e-6
    np.testing.assert_allclose(
        actual[names["output"]],
        expected[names["output"]],
        rtol=tolerance,
        atol=tolerance,
    )
    assert names["pre"] not in model_ir.tensors
    assert names["inst_output"] not in model_ir.tensors
    assert _operator(model_ir, names["squeezed"]).inputs == [names["source"]]
    reshape = _operator(model_ir, names["x"])
    assert reshape.options == {
        "newShape": [1, 2, 4, 3],
        "marker": "reshape",
    }
    assert np.asarray(model_ir.tensors[reshape.inputs[1]].data).tolist() == [
        1,
        2,
        4,
        3,
    ]
    mean1 = _operator(model_ir, names["mean1"])
    mean2 = _operator(model_ir, names["mean2"])
    assert mean1.inputs[0] == names["source"]
    assert _operator(model_ir, names["centered"]).inputs[0] == names["source"]
    assert np.asarray(model_ir.tensors[mean1.inputs[1]].data).tolist() == [1, 2]
    assert np.asarray(model_ir.tensors[mean2.inputs[1]].data).tolist() == [1, 2]
    assert mean1.options == original_mean.options
    assert mean1.axis_semantics == original_mean.axis_semantics
    assert mean1.version == original_mean.version
    assert mean1.onnx_node_name == original_mean.onnx_node_name
    assert mean1.onnx_op_type == original_mean.onnx_op_type
    assert model_ir.tensors[names["squeezed"]].shape == [2, 4, 3]
    assert model_ir.tensors[names["centered"]].shape == [1, 2, 4, 3]
    assert model_ir.tensors[names["mean1"]].shape == [1, 1, 1, 3]
    assert model_ir.tensors[names["scale"]].shape == [1, 1, 1, 3]
    assert model_ir.tensors[names["bias"]].shape == [1, 1, 1, 3]
    assert _operator(model_ir, names["post_output"]).op_type == "ADD"
    assert "existing_hint" in model_ir.metadata[
        "assume_channel_last_layout_tensor_names"
    ]
    assert names["post_output"] in model_ir.metadata[
        "assume_channel_last_layout_tensor_names"
    ]


def test_direct_instance_norm_clones_all_shared_changed_constants() -> None:
    model_ir, names = _build_model()
    changed_names = (
        names["reshape_shape"],
        names["axes1"],
        names["scale"],
        names["bias"],
    )
    originals = {
        name: np.asarray(model_ir.tensors[name].data).copy()
        for name in changed_names
    }
    for serial, name in enumerate(changed_names):
        tensor = model_ir.tensors[name]
        output_name = f"preserved_{serial}"
        model_ir.tensors[output_name] = _tensor(
            output_name,
            list(tensor.shape),
            dtype=str(tensor.dtype),
        )
        model_ir.operators.append(OperatorIR("IDENTITY", [name], [output_name]))
        model_ir.outputs.append(output_name)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 1}
    for name, data in originals.items():
        np.testing.assert_array_equal(model_ir.tensors[name].data, data)
    reshape = _operator(model_ir, names["x"])
    mean1 = _operator(model_ir, names["mean1"])
    mean2 = _operator(model_ir, names["mean2"])
    scale = _operator(model_ir, names["scaled"])
    bias = _operator(model_ir, names["post_output"])
    assert reshape.inputs[1] != names["reshape_shape"]
    assert mean1.inputs[1] == mean2.inputs[1]
    assert mean1.inputs[1] != names["axes1"]
    assert names["scale"] not in scale.inputs
    assert names["bias"] not in bias.inputs
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("dynamic_axis", ("height", "width", "channel"))
def test_direct_instance_norm_preserves_dynamic_signatures(
    dynamic_axis: str,
) -> None:
    model_ir, names = _build_model()
    source_axis = {"height": 1, "width": 2, "channel": 3}[dynamic_axis]
    nchw_axis = {"height": 2, "width": 3, "channel": 1}[dynamic_axis]
    chw_axis = {"height": 1, "width": 2, "channel": 0}[dynamic_axis]
    model_ir.tensors[names["source"]].shape_signature[source_axis] = -1
    model_ir.tensors[names["post_output"]].shape_signature[source_axis] = -1
    model_ir.tensors[names["output"]].shape_signature[source_axis] = -1
    model_ir.tensors[names["pre"]].shape_signature[nchw_axis] = -1
    model_ir.tensors[names["squeezed"]].shape_signature[chw_axis] = -1
    for name in (
        names["x"],
        names["centered"],
        names["squared"],
        names["normalized"],
        names["scaled"],
        names["inst_output"],
    ):
        model_ir.tensors[name].shape_signature[nchw_axis] = -1
    if dynamic_axis == "channel":
        for name in (
            names["mean1"],
            names["mean2"],
            names["variance_epsilon"],
            names["std"],
            names["inverse_std"],
        ):
            model_ir.tensors[name].shape_signature[1] = -1

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 1}
    expected_squeeze_axis = {"height": 0, "width": 1, "channel": 2}[
        dynamic_axis
    ]
    assert (
        model_ir.tensors[names["squeezed"]].shape_signature[
            expected_squeeze_axis
        ]
        == -1
    )
    assert (
        model_ir.tensors[names["centered"]].shape_signature[source_axis]
        == -1
    )
    if dynamic_axis == "channel":
        assert model_ir.tensors[names["mean1"]].shape_signature == [1, 1, 1, -1]


def test_direct_instance_norm_rewrites_multiple_chains() -> None:
    first, _ = _build_model(prefix="a_")
    second, _ = _build_model(prefix="b_", produced_source=True)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            first
        )
    )

    assert stats == {_STATS: 2}
    assert validate_model_ir_invariants(first) == []


@pytest.mark.parametrize(
    "case",
    (
        "floating_pre_perm",
        "produced_pre_perm",
        "source_public_output",
        "source_unbound",
        "backward_source",
        "pre_public",
        "pre_fanout",
        "pre_shape",
        "pre_signature",
        "squeeze_axis",
        "squeeze_dynamic_batch",
        "squeeze_shape",
        "reshape_shape_value",
        "floating_reshape_shape",
        "produced_reshape_shape",
        "x_public",
        "x_fanout",
        "mean1_axes",
        "floating_mean1_axes",
        "produced_mean1_axes",
        "mean1_keepdims",
        "mean1_shape",
        "mean1_fanout",
        "reversed_sub",
        "centered_fanout",
        "mean2_axes",
        "mean2_keepdims",
        "negative_epsilon",
        "nonfinite_epsilon",
        "produced_epsilon",
        "reversed_div",
        "nonunit_numerator",
        "inverse_std_fanout",
        "scale_shape",
        "scale_dtype",
        "bias_shape",
        "shared_scale_bias",
        "inst_output_fanout",
        "post_public_output",
        "post_shape",
        "post_dtype",
        "per_tensor_quantization",
        "duplicate_post_output",
        "backward_post_consumer",
    ),
)
def test_direct_instance_norm_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model(produced_source=case == "backward_source")
    pre = _operator(model_ir, names["pre"])
    squeeze = _operator(model_ir, names["squeezed"])
    mean1 = _operator(model_ir, names["mean1"])
    sub = _operator(model_ir, names["centered"])
    mean2 = _operator(model_ir, names["mean2"])
    div = _operator(model_ir, names["inverse_std"])
    bias = _operator(model_ir, names["inst_output"])
    post = _operator(model_ir, names["post_output"])
    if case == "floating_pre_perm":
        model_ir.tensors[names["pre_perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["pre_perm"]].data = np.asarray(
            [0, 3, 1, 2], dtype=np.float32
        )
    elif case == "produced_pre_perm":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["reshape_shape"]], [names["pre_perm"]])
        )
    elif case == "source_public_output":
        model_ir.outputs.append(names["source"])
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "backward_source":
        producer = model_ir.operators.pop(0)
        model_ir.operators.insert(model_ir.operators.index(pre) + 1, producer)
    elif case == "pre_public":
        model_ir.outputs.append(names["pre"])
    elif case == "pre_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["pre"]], ["side"]))
        model_ir.outputs.append("side")
    elif case == "pre_shape":
        model_ir.tensors[names["pre"]].shape[2] = 3
    elif case == "pre_signature":
        model_ir.tensors[names["pre"]].shape_signature[2] = -1
    elif case == "squeeze_axis":
        squeeze.options["squeezeDims"] = [1]
    elif case == "squeeze_dynamic_batch":
        model_ir.tensors[names["pre"]].shape_signature[0] = -1
        model_ir.tensors[names["squeezed"]].shape_signature[0] = -1
    elif case == "squeeze_shape":
        model_ir.tensors[names["squeezed"]].shape[0] = 4
    elif case == "reshape_shape_value":
        model_ir.tensors[names["reshape_shape"]].data[2] = 5
    elif case == "floating_reshape_shape":
        model_ir.tensors[names["reshape_shape"]].dtype = "FLOAT32"
        model_ir.tensors[names["reshape_shape"]].data = np.asarray(
            [1, 3, 2, 4], dtype=np.float32
        )
    elif case == "produced_reshape_shape":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                [names["pre_perm"]],
                [names["reshape_shape"]],
            )
        )
    elif case == "x_public":
        model_ir.outputs.append(names["x"])
    elif case == "x_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["x"]], ["side"]))
        model_ir.outputs.append("side")
    elif case == "mean1_axes":
        model_ir.tensors[names["axes1"]].data[:] = [1, 2]
    elif case == "floating_mean1_axes":
        model_ir.tensors[names["axes1"]].dtype = "FLOAT32"
        model_ir.tensors[names["axes1"]].data = np.asarray(
            [2, 3], dtype=np.float32
        )
    elif case == "produced_mean1_axes":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pre_perm"]], [names["axes1"]])
        )
    elif case == "mean1_keepdims":
        mean1.options["keepDims"] = False
    elif case == "mean1_shape":
        model_ir.tensors[names["mean1"]].shape[2] = 2
    elif case == "mean1_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["mean1"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "reversed_sub":
        sub.inputs.reverse()
    elif case == "centered_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["centered"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "mean2_axes":
        model_ir.tensors[names["axes1"]].data[:] = [1, 3]
    elif case == "mean2_keepdims":
        mean2.options["keepDims"] = False
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    elif case == "nonfinite_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = np.nan
    elif case == "produced_epsilon":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["one"]], [names["epsilon"]])
        )
    elif case == "reversed_div":
        div.inputs.reverse()
    elif case == "nonunit_numerator":
        model_ir.tensors[names["one"]].data[0] = 2.0
    elif case == "inverse_std_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inverse_std"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "scale_shape":
        model_ir.tensors[names["scale"]].shape = [1, 1, 1, 3]
        model_ir.tensors[names["scale"]].shape_signature = [1, 1, 1, 3]
        model_ir.tensors[names["scale"]].data = np.transpose(
            model_ir.tensors[names["scale"]].data,
            (0, 2, 3, 1),
        )
    elif case == "scale_dtype":
        model_ir.tensors[names["scale"]].dtype = "FLOAT16"
        model_ir.tensors[names["scale"]].data = np.asarray(
            model_ir.tensors[names["scale"]].data,
            dtype=np.float16,
        )
    elif case == "bias_shape":
        model_ir.tensors[names["bias"]].shape[2] = 2
    elif case == "shared_scale_bias":
        for input_index, input_name in enumerate(bias.inputs):
            if input_name == names["bias"]:
                bias.inputs[input_index] = names["scale"]
    elif case == "inst_output_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inst_output"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "post_public_output":
        model_ir.outputs.append(names["post_output"])
    elif case == "post_shape":
        model_ir.tensors[names["post_output"]].shape[1] = 3
    elif case == "post_dtype":
        model_ir.tensors[names["post_output"]].dtype = "FLOAT16"
    elif case == "per_tensor_quantization":
        model_ir.tensors[names["centered"]].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "duplicate_post_output":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["post_output"]])
        )
    elif case == "backward_post_consumer":
        output = _operator(model_ir, names["output"])
        model_ir.operators.remove(output)
        model_ir.operators.insert(model_ir.operators.index(post), output)
    before = repr(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_direct_instance_norm_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir, names = _build_model()
    model_ir.tensors[names["reshape_shape"]].quantization = {
        "fault": Unclonable()
    }
    model_ir.tensors["preserved_shape"] = _tensor(
        "preserved_shape",
        [4],
        dtype="INT64",
    )
    model_ir.operators.append(
        OperatorIR(
            "IDENTITY",
            [names["reshape_shape"]],
            ["preserved_shape"],
        )
    )
    model_ir.outputs.append("preserved_shape")
    before = repr(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_direct_instance_norm_apply_preflight_is_transactional(monkeypatch) -> None:
    model_ir, names = _build_model()
    model_ir.tensors["preserved_shape"] = _tensor(
        "preserved_shape",
        [4],
        dtype="INT64",
    )
    model_ir.operators.append(
        OperatorIR(
            "IDENTITY",
            [names["reshape_shape"]],
            ["preserved_shape"],
        )
    )
    model_ir.outputs.append("preserved_shape")
    original_resolve = direct_module._resolve_candidate
    injected = False

    def resolve_with_collision(*args, **kwargs):
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
        model_ir.tensors[clone_name] = _tensor(
            clone_name,
            [1],
            dtype="INT64",
            data=np.asarray([9], dtype=np.int64),
        )
        injected = True
        return plan

    monkeypatch.setattr(direct_module, "_resolve_candidate", resolve_with_collision)
    before_operators = repr(model_ir.operators)
    before_source = copy.deepcopy(model_ir.tensors[names["source"]])

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir.operators) == before_operators
    assert repr(model_ir.tensors[names["source"]]) == repr(before_source)


def test_compatibility_wrapper_reports_direct_count() -> None:
    model_ir, names = _build_model()

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir)

    assert stats == {_COMPAT_STATS: 1}
    assert _operator(model_ir, names["post_output"]).op_type == "ADD"
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    ("reversed_sub", "wrong_axes", "negative_epsilon", "quantized_tensor"),
)
def test_compatibility_wrapper_does_not_fallback_for_unsafe_direct_tail(
    case: str,
) -> None:
    model_ir, names = _build_model()
    if case == "reversed_sub":
        _operator(model_ir, names["centered"]).inputs.reverse()
    elif case == "wrong_axes":
        model_ir.tensors[names["axes1"]].data[:] = [1, 3]
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    elif case == "quantized_tensor":
        model_ir.tensors[names["centered"]].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    before = repr(model_ir)

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir)

    assert stats == {_COMPAT_STATS: 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "mode",
    ("side_squeeze", "squeeze_unary_reshape", "squeeze_add_reshape"),
)
def test_legacy_instance_norm_tail_modes_remain_numerically_equivalent(
    mode: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, mode)
    rng = np.random.default_rng(61)
    feeds = {
        name: rng.normal(size=model_ir.tensors[name].shape).astype(np.float32)
        for name in model_ir.inputs
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    before = repr(model_ir)

    direct_stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
    )

    assert direct_stats == {_STATS: 0}
    assert repr(model_ir) == before

    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    compatibility_stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert compatibility_stats == {_COMPAT_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    actual = _evaluate(model_ir, feeds)
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-6)


def test_compatibility_wrapper_preserves_mixed_tail_order_and_total_limit() -> None:
    model_ir, first_names = _build_model(prefix="m00_")
    _replace_tail(model_ir, first_names, "side_squeeze")
    names_by_serial = [first_names]
    for serial in range(1, 33):
        candidate, names = _build_model(prefix=f"m{serial:02d}_")
        model_ir.tensors.update(candidate.tensors)
        model_ir.operators.extend(candidate.operators)
        model_ir.inputs.extend(candidate.inputs)
        model_ir.outputs.extend(candidate.outputs)
        names_by_serial.append(names)

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir)

    assert stats == {_COMPAT_STATS: 32}
    assert _operator(model_ir, first_names["post_output"]).op_type == "ADD"
    for names in names_by_serial[1:32]:
        assert _operator(model_ir, names["post_output"]).op_type == "ADD"
    final_names = names_by_serial[32]
    assert _operator(model_ir, final_names["post_output"]).op_type == "TRANSPOSE"
    assert _operator(model_ir, final_names["pre"]).op_type == "TRANSPOSE"
    assert validate_model_ir_invariants(model_ir) == []


def test_direct_instance_norm_preflight_does_not_allocate_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("no_instance_norm_direct")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(direct_module, "ModelIRGraphIndex", unexpected_index)

    assert (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
        == {_STATS: 0}
    )
