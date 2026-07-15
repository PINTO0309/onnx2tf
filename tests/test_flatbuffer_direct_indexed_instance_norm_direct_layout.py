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
    _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains,
    _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains,
    _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains,
)


_STATS = (
    "optimized_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains"
)
_COMPAT_STATS = "optimized_transpose_instancenorm_prepost_nhwc_chains"
_SIDE_STATS = (
    "optimized_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains"
)
_UNARY_RESHAPE_STATS = (
    "optimized_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains"
)
_RESIDUAL_RESHAPE_STATS = (
    "optimized_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains"
)


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
    prefix = names["source"][: -len("source")]

    def _name(value: str) -> str:
        return f"{prefix}{value}"

    data_dtype = str(model_ir.tensors[names["inst_output"]].dtype)
    post = _operator(model_ir, names["post_output"])
    post_index = model_ir.operators.index(post)
    if mode == "side_squeeze":
        model_ir.tensors[_name("side_output")] = _tensor(
            _name("side_output"),
            [3, 2, 4],
            dtype=data_dtype,
        )
        model_ir.operators.append(
            OperatorIR(
                "SQUEEZE",
                [names["inst_output"]],
                [_name("side_output")],
                {"squeezeDims": [0]},
            )
        )
        model_ir.outputs.append(_name("side_output"))
        return

    model_ir.operators.remove(post)
    model_ir.tensors[_name("tail_squeezed")] = _tensor(
        _name("tail_squeezed"),
        [3, 2, 4],
        dtype=data_dtype,
    )
    tail_ops = [
        OperatorIR(
            "SQUEEZE",
            [names["inst_output"]],
            [_name("tail_squeezed")],
            {"squeezeDims": [0]},
        )
    ]
    tail_data_name = _name("tail_squeezed")
    if mode == "squeeze_unary_reshape":
        model_ir.tensors[_name("tail_unary")] = _tensor(
            _name("tail_unary"),
            [3, 2, 4],
            dtype=data_dtype,
        )
        tail_ops.append(
            OperatorIR(
                "RELU",
                [_name("tail_squeezed")],
                [_name("tail_unary")],
            )
        )
        tail_data_name = _name("tail_unary")
    elif mode == "squeeze_add_reshape":
        model_ir.inputs.append(_name("residual_source"))
        model_ir.tensors[_name("residual_source")] = _tensor(
            _name("residual_source"),
            [2, 4, 3],
            dtype=data_dtype,
        )
        model_ir.tensors[_name("residual_perm")] = _tensor(
            _name("residual_perm"),
            [3],
            dtype="INT32",
            data=np.asarray([2, 0, 1], dtype=np.int32),
        )
        model_ir.tensors[_name("residual_chw")] = _tensor(
            _name("residual_chw"),
            [3, 2, 4],
            dtype=data_dtype,
        )
        model_ir.tensors[_name("tail_add")] = _tensor(
            _name("tail_add"),
            [3, 2, 4],
            dtype=data_dtype,
        )
        tail_ops.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [_name("residual_source"), _name("residual_perm")],
                    [_name("residual_chw")],
                ),
                OperatorIR(
                    "ADD",
                    [_name("tail_squeezed"), _name("residual_chw")],
                    [_name("tail_add")],
                ),
            ]
        )
        tail_data_name = _name("tail_add")
    else:
        raise AssertionError(f"unsupported legacy tail mode: {mode}")
    model_ir.tensors[_name("tail_shape")] = _tensor(
        _name("tail_shape"),
        [4],
        dtype="INT64",
        data=np.asarray([1, 3, 2, 4], dtype=np.int64),
    )
    model_ir.tensors[_name("tail_reshape")] = _tensor(
        _name("tail_reshape"),
        [1, 3, 2, 4],
        dtype=data_dtype,
    )
    tail_ops.append(
        OperatorIR(
            "RESHAPE",
            [tail_data_name, _name("tail_shape")],
            [_name("tail_reshape")],
            {"newShape": [1, 3, 2, 4]},
        )
    )
    post.inputs[0] = _name("tail_reshape")
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


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("separate_axes", (False, True))
def test_unary_reshape_instance_norm_tail_is_indexed_and_equivalent(
    dtype: str,
    produced_source: bool,
    separate_axes: bool,
) -> None:
    model_ir, names = _build_model(
        dtype=dtype,
        produced_source=produced_source,
        separate_axes=separate_axes,
    )
    _replace_tail(model_ir, names, "squeeze_unary_reshape")
    feed_name = names["upstream"] if produced_source else names["source"]
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    feeds = {
        feed_name: np.random.default_rng(71)
        .normal(size=model_ir.tensors[feed_name].shape)
        .astype(np_dtype)
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    before = repr(model_ir)

    assert (
        _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains(
            model_ir
        )
        == {_STATS: 0}
    )
    assert (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir
        )
        == {_SIDE_STATS: 0}
    )
    assert repr(model_ir) == before
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_UNARY_RESHAPE_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    actual = _evaluate(model_ir, feeds)
    assert set(actual) == set(expected)
    tolerance = 1e-2 if dtype == "FLOAT16" else 1e-6
    for name in expected:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=tolerance,
            atol=tolerance,
        )
    tail_reshape = _operator(model_ir, names["post_output"])
    assert tail_reshape.op_type == "RESHAPE"
    assert tail_reshape.options["newShape"] == [1, 2, 4, 3]
    assert np.asarray(model_ir.tensors[tail_reshape.inputs[1]].data).tolist() == [
        1,
        2,
        4,
        3,
    ]
    assert model_ir.tensors[names["inst_output"]].shape == [1, 2, 4, 3]
    assert model_ir.tensors["tail_squeezed"].shape == [2, 4, 3]
    assert model_ir.tensors["tail_unary"].shape == [2, 4, 3]
    assert names["post_output"] in model_ir.metadata[
        "assume_channel_last_layout_tensor_names"
    ]


@pytest.mark.parametrize(
    "unary_op",
    (
        "RELU",
        "RELU6",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "ABS",
        "NEG",
        "SQRT",
        "EXP",
        "CAST",
        "FLOOR",
        "CEIL",
        "ROUND",
    ),
)
def test_unary_reshape_instance_norm_preserves_supported_unary_family(
    unary_op: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_unary_reshape")
    unary = _operator(model_ir, "tail_unary")
    unary.op_type = unary_op
    if unary_op == "CAST":
        unary.options = {"to": "FLOAT16"}
        for name in (
            "tail_unary",
            "tail_reshape",
            names["post_output"],
            names["output"],
        ):
            model_ir.tensors[name].dtype = "FLOAT16"

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_UNARY_RESHAPE_STATS: 1}
    assert _operator(model_ir, "tail_unary").op_type == unary_op
    assert _operator(model_ir, names["post_output"]).op_type == "RESHAPE"
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("dynamic_axis", ("height", "width", "channel"))
def test_unary_reshape_instance_norm_preserves_dynamic_signatures(
    dynamic_axis: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_unary_reshape")
    source_axis = {"height": 1, "width": 2, "channel": 3}[dynamic_axis]
    nchw_axis = {"height": 2, "width": 3, "channel": 1}[dynamic_axis]
    chw_axis = {"height": 1, "width": 2, "channel": 0}[dynamic_axis]
    hwc_axis = {"height": 0, "width": 1, "channel": 2}[dynamic_axis]
    for name in (names["source"], names["post_output"], names["output"]):
        model_ir.tensors[name].shape_signature[source_axis] = -1
    model_ir.tensors[names["pre"]].shape_signature[nchw_axis] = -1
    model_ir.tensors[names["squeezed"]].shape_signature[chw_axis] = -1
    for name in (
        names["x"],
        names["centered"],
        names["squared"],
        names["normalized"],
        names["scaled"],
        names["inst_output"],
        "tail_reshape",
    ):
        model_ir.tensors[name].shape_signature[nchw_axis] = -1
    for name in ("tail_squeezed", "tail_unary"):
        model_ir.tensors[name].shape_signature[chw_axis] = -1
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
        _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_UNARY_RESHAPE_STATS: 1}
    assert model_ir.tensors[names["inst_output"]].shape_signature[source_axis] == -1
    assert model_ir.tensors["tail_squeezed"].shape_signature[hwc_axis] == -1
    assert model_ir.tensors["tail_unary"].shape_signature[hwc_axis] == -1
    assert model_ir.tensors[names["post_output"]].shape_signature[source_axis] == -1


def test_unary_reshape_instance_norm_clones_shared_tail_shape() -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_unary_reshape")
    original = np.asarray(model_ir.tensors["tail_shape"].data).copy()
    model_ir.tensors["preserved_tail_shape"] = _tensor(
        "preserved_tail_shape",
        [4],
        dtype="INT64",
    )
    model_ir.operators.append(
        OperatorIR("IDENTITY", ["tail_shape"], ["preserved_tail_shape"])
    )
    model_ir.outputs.append("preserved_tail_shape")

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_UNARY_RESHAPE_STATS: 1}
    np.testing.assert_array_equal(model_ir.tensors["tail_shape"].data, original)
    tail_reshape = _operator(model_ir, names["post_output"])
    assert tail_reshape.inputs[1] != "tail_shape"
    assert np.asarray(model_ir.tensors[tail_reshape.inputs[1]].data).tolist() == [
        1,
        2,
        4,
        3,
    ]
    assert validate_model_ir_invariants(model_ir) == []


def test_unary_reshape_instance_norm_rewrites_multiple_chains() -> None:
    first, first_names = _build_model(prefix="a_")
    _replace_tail(first, first_names, "squeeze_unary_reshape")
    second, second_names = _build_model(prefix="b_")
    _replace_tail(second, second_names, "squeeze_unary_reshape")
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
            first
        )
    )

    assert stats == {_UNARY_RESHAPE_STATS: 2}
    assert _operator(first, first_names["post_output"]).op_type == "RESHAPE"
    assert _operator(first, second_names["post_output"]).op_type == "RESHAPE"
    assert validate_model_ir_invariants(first) == []


@pytest.mark.parametrize(
    "case",
    (
        "tail_axis",
        "unsupported_unary",
        "unary_fanout",
        "tail_shape_value",
        "tail_shape_floating",
        "tail_shape_produced",
        "tail_shape_public",
        "tail_shape_quantized",
        "tail_dtype",
        "tail_quantized",
        "tail_public",
        "tail_duplicate_producer",
        "tail_backward_consumer",
    ),
)
def test_unary_reshape_rejects_unsafe_tail_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_unary_reshape")
    tail_squeeze = _operator(model_ir, "tail_squeezed")
    tail_unary = _operator(model_ir, "tail_unary")
    if case == "tail_axis":
        tail_squeeze.options["squeezeDims"] = [1]
    elif case == "unsupported_unary":
        tail_unary.op_type = "SIN"
    elif case == "unary_fanout":
        model_ir.tensors["unary_side"] = _tensor("unary_side", [3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["tail_unary"], ["unary_side"])
        )
        model_ir.outputs.append("unary_side")
    elif case == "tail_shape_value":
        model_ir.tensors["tail_shape"].data[:] = [1, 3, 4, 2]
    elif case == "tail_shape_floating":
        model_ir.tensors["tail_shape"].dtype = "FLOAT32"
        model_ir.tensors["tail_shape"].data = np.asarray(
            [1, 3, 2, 4], dtype=np.float32
        )
    elif case == "tail_shape_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["post_perm"]], ["tail_shape"])
        )
    elif case == "tail_shape_public":
        model_ir.outputs.append("tail_shape")
    elif case == "tail_shape_quantized":
        model_ir.tensors["tail_shape"].quantization = {
            "scale": [1.0],
            "zero_point": [0],
        }
    elif case == "tail_dtype":
        model_ir.tensors["tail_unary"].dtype = "FLOAT16"
        model_ir.tensors["tail_reshape"].dtype = "FLOAT16"
        model_ir.tensors[names["post_output"]].dtype = "FLOAT16"
    elif case == "tail_quantized":
        model_ir.tensors["tail_unary"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "tail_public":
        model_ir.outputs.append("tail_unary")
    elif case == "tail_duplicate_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], ["tail_unary"])
        )
    elif case == "tail_backward_consumer":
        model_ir.tensors["tail_side"] = _tensor("tail_side", [3, 2, 4])
        model_ir.operators.insert(
            model_ir.operators.index(tail_unary),
            OperatorIR("IDENTITY", ["tail_unary"], ["tail_side"]),
        )
        model_ir.outputs.append("tail_side")
    before = repr(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_UNARY_RESHAPE_STATS: 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "case",
    ("tail_shape", "tail_quantization", "negative_epsilon"),
)
def test_compatibility_wrapper_does_not_fallback_for_unsafe_unary_reshape(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_unary_reshape")
    if case == "tail_shape":
        model_ir.tensors["tail_shape"].data[:] = [1, 3, 4, 2]
    elif case == "tail_quantization":
        model_ir.tensors["tail_unary"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    before = repr(model_ir)

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir)

    assert stats == {_COMPAT_STATS: 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize("side_before_post", (False, True))
@pytest.mark.parametrize("side_is_public_output", (False, True))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
def test_side_squeeze_instance_norm_tail_is_indexed_and_equivalent(
    side_before_post: bool,
    side_is_public_output: bool,
    produced_source: bool,
    dtype: str,
) -> None:
    model_ir, names = _build_model(
        dtype=dtype,
        produced_source=produced_source,
    )
    _replace_tail(model_ir, names, "side_squeeze")
    side_squeeze = _operator(model_ir, "side_output")
    post = _operator(model_ir, names["post_output"])
    if side_before_post:
        model_ir.operators.remove(side_squeeze)
        model_ir.operators.insert(model_ir.operators.index(post), side_squeeze)
    if not side_is_public_output:
        model_ir.outputs.remove("side_output")
        model_ir.tensors["side_final"] = _tensor(
            "side_final",
            [3, 2, 4],
            dtype=dtype,
        )
        model_ir.operators.append(
            OperatorIR("RELU", ["side_output"], ["side_final"])
        )
        model_ir.outputs.append("side_final")
    feed_name = names["upstream"] if produced_source else names["source"]
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    feeds = {
        feed_name: np.random.default_rng(67)
        .normal(size=model_ir.tensors[feed_name].shape)
        .astype(np_dtype)
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

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_SIDE_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    actual = _evaluate(model_ir, feeds)
    assert set(actual) == set(expected)
    tolerance = 1e-2 if dtype == "FLOAT16" else 1e-6
    for name in expected:
        np.testing.assert_allclose(
            actual[name],
            expected[name],
            rtol=tolerance,
            atol=tolerance,
        )
    side_squeeze = _operator(model_ir, "side_output")
    adapter = _operator(model_ir, names["inst_output"])
    assert side_squeeze.inputs == [names["inst_output"]]
    assert adapter.op_type == "TRANSPOSE"
    assert adapter.inputs[0] == names["post_output"]
    assert np.asarray(model_ir.tensors[adapter.inputs[1]].data).tolist() == [
        0,
        3,
        1,
        2,
    ]
    assert model_ir.operators.index(adapter) < model_ir.operators.index(side_squeeze)
    assert model_ir.tensors[names["inst_output"]].shape == [1, 3, 2, 4]
    assert names["inst_output"] not in model_ir.metadata[
        "assume_channel_last_layout_tensor_names"
    ]


@pytest.mark.parametrize("dynamic_axis", ("height", "width", "channel"))
def test_side_squeeze_instance_norm_preserves_dynamic_signatures(
    dynamic_axis: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "side_squeeze")
    source_axis = {"height": 1, "width": 2, "channel": 3}[dynamic_axis]
    nchw_axis = {"height": 2, "width": 3, "channel": 1}[dynamic_axis]
    chw_axis = {"height": 1, "width": 2, "channel": 0}[dynamic_axis]
    for name in (names["source"], names["post_output"], names["output"]):
        model_ir.tensors[name].shape_signature[source_axis] = -1
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
    model_ir.tensors["side_output"].shape_signature[chw_axis] = -1
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
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_SIDE_STATS: 1}
    assert model_ir.tensors[names["inst_output"]].shape_signature[nchw_axis] == -1
    assert model_ir.tensors["side_output"].shape_signature[chw_axis] == -1
    assert model_ir.tensors[names["centered"]].shape_signature[source_axis] == -1


def test_side_squeeze_instance_norm_rewrites_multiple_chains() -> None:
    first, first_names = _build_model(prefix="a_")
    _replace_tail(first, first_names, "side_squeeze")
    second, second_names = _build_model(prefix="b_")
    _replace_tail(second, second_names, "side_squeeze")
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            first
        )
    )

    assert stats == {_SIDE_STATS: 2}
    assert validate_model_ir_invariants(first) == []
    assert np.asarray(
        first.tensors[direct_module._SIDE_ADAPTER_PERM_NAME].data
    ).tolist() == [0, 3, 1, 2]
    assert _operator(first, first_names["inst_output"]).op_type == "TRANSPOSE"
    assert _operator(first, second_names["inst_output"]).op_type == "TRANSPOSE"


@pytest.mark.parametrize(
    "case",
    ("wrong_value", "floating", "produced", "public", "quantized"),
)
def test_side_squeeze_rejects_invalid_existing_adapter_permutation(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "side_squeeze")
    perm_name = direct_module._SIDE_ADAPTER_PERM_NAME
    model_ir.tensors[perm_name] = _tensor(
        perm_name,
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    if case == "wrong_value":
        model_ir.tensors[perm_name].data[1] = 2
    elif case == "floating":
        model_ir.tensors[perm_name].dtype = "FLOAT32"
        model_ir.tensors[perm_name].data = np.asarray(
            [0, 3, 1, 2], dtype=np.float32
        )
    elif case == "produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pre_perm"]], [perm_name])
        )
    elif case == "public":
        model_ir.outputs.append(perm_name)
    elif case == "quantized":
        model_ir.tensors[perm_name].quantization = {
            "scale": [1.0],
            "zero_point": [0],
        }
    before = repr(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_SIDE_STATS: 0}
    assert repr(model_ir) == before


def test_side_squeeze_reuses_valid_existing_adapter_permutation() -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "side_squeeze")
    perm_name = direct_module._SIDE_ADAPTER_PERM_NAME
    existing = _tensor(
        perm_name,
        [4],
        dtype="INT64",
        data=np.asarray([0, 3, 1, 2], dtype=np.int64),
    )
    model_ir.tensors[perm_name] = existing

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_SIDE_STATS: 1}
    assert model_ir.tensors[perm_name] is existing
    assert _operator(model_ir, names["inst_output"]).inputs[1] == perm_name


@pytest.mark.parametrize(
    "case",
    (
        "side_axis",
        "side_shape",
        "side_dtype",
        "side_public_input",
        "side_duplicate_producer",
        "side_backward_consumer",
        "third_inst_consumer",
        "side_quantization",
        "side_before_bias",
    ),
)
def test_side_squeeze_rejects_unsafe_tail_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "side_squeeze")
    side = _operator(model_ir, "side_output")
    bias = _operator(model_ir, names["inst_output"])
    if case == "side_axis":
        side.options["squeezeDims"] = [1]
    elif case == "side_shape":
        model_ir.tensors["side_output"].shape[1] = 3
    elif case == "side_dtype":
        model_ir.tensors["side_output"].dtype = "FLOAT16"
    elif case == "side_public_input":
        model_ir.inputs.append("side_output")
    elif case == "side_duplicate_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], ["side_output"])
        )
    elif case == "side_backward_consumer":
        model_ir.tensors["side_final"] = _tensor("side_final", [3, 2, 4])
        model_ir.operators.insert(
            model_ir.operators.index(side),
            OperatorIR("IDENTITY", ["side_output"], ["side_final"]),
        )
        model_ir.outputs.append("side_final")
    elif case == "third_inst_consumer":
        model_ir.tensors["inst_side"] = _tensor("inst_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inst_output"]], ["inst_side"])
        )
        model_ir.outputs.append("inst_side")
    elif case == "side_quantization":
        model_ir.tensors["side_output"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "side_before_bias":
        model_ir.operators.remove(side)
        model_ir.operators.insert(model_ir.operators.index(bias), side)
    before = repr(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_SIDE_STATS: 0}
    assert repr(model_ir) == before


def test_side_squeeze_adapter_preflight_collision_is_transactional(
    monkeypatch,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "side_squeeze")
    original_resolve = direct_module._resolve_candidate
    injected = False

    def resolve_with_collision(*args, **kwargs):
        nonlocal injected
        plan = original_resolve(*args, **kwargs)
        if plan is None or injected:
            return plan
        assert plan.side_adapter_permutation is not None
        perm_name = direct_module._SIDE_ADAPTER_PERM_NAME
        model_ir.tensors[perm_name] = _tensor(
            perm_name,
            [4],
            dtype="INT32",
            data=np.asarray([9, 9, 9, 9], dtype=np.int32),
        )
        injected = True
        return plan

    monkeypatch.setattr(direct_module, "_resolve_candidate", resolve_with_collision)
    before_operators = repr(model_ir.operators)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_SIDE_STATS: 0}
    assert repr(model_ir.operators) == before_operators


@pytest.mark.parametrize(
    "case",
    ("invalid_adapter", "side_quantization", "negative_epsilon"),
)
def test_compatibility_wrapper_does_not_fallback_for_unsafe_side_squeeze(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "side_squeeze")
    if case == "invalid_adapter":
        perm_name = direct_module._SIDE_ADAPTER_PERM_NAME
        model_ir.tensors[perm_name] = _tensor(
            perm_name,
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        )
    elif case == "side_quantization":
        model_ir.tensors["side_output"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    before = repr(model_ir)

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir)

    assert stats == {_COMPAT_STATS: 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    ("first_mode", "first_output_op"),
    (
        ("side_squeeze", "ADD"),
        ("squeeze_unary_reshape", "RESHAPE"),
        ("squeeze_add_reshape", "RESHAPE"),
    ),
)
def test_compatibility_wrapper_preserves_mixed_tail_order_and_total_limit(
    first_mode: str,
    first_output_op: str,
) -> None:
    model_ir, first_names = _build_model(prefix="m00_")
    _replace_tail(model_ir, first_names, first_mode)
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
    assert _operator(model_ir, first_names["post_output"]).op_type == first_output_op
    for names in names_by_serial[1:32]:
        assert _operator(model_ir, names["post_output"]).op_type == "ADD"
    final_names = names_by_serial[32]
    assert _operator(model_ir, final_names["post_output"]).op_type == "TRANSPOSE"
    assert _operator(model_ir, final_names["pre"]).op_type == "TRANSPOSE"
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    ("owner", "stats_key"),
    (
        (
            _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains,
            _STATS,
        ),
        (
            _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains,
            _SIDE_STATS,
        ),
        (
            _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains,
            _UNARY_RESHAPE_STATS,
        ),
        (
            _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains,
            _RESIDUAL_RESHAPE_STATS,
        ),
    ),
    ids=("direct", "side_squeeze", "unary_reshape", "residual_reshape"),
)
def test_instance_norm_prepost_preflight_does_not_allocate_index(
    monkeypatch,
    owner,
    stats_key: str,
) -> None:
    model_ir = ModelIR("no_instance_norm_direct")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(direct_module, "ModelIRGraphIndex", unexpected_index)

    assert owner(model_ir) == {stats_key: 0}
