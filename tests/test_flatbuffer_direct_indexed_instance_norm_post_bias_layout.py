from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_norm_post_bias_layout as post_bias_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.instance_norm_post_bias_layout import (
    optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains,
)


_STATS = "optimized_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"


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


def _coefficient(
    values: list[float],
    *,
    dtype,
    mode: str,
) -> np.ndarray:
    data = np.asarray(values, dtype=dtype)
    if mode == "scalar":
        return data[:1]
    if mode == "nhwc":
        return data.reshape(1, 1, 1, -1)
    assert mode == "nchw"
    return data.reshape(1, -1, 1, 1)


def _build_model(
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    produced_source: bool = False,
    separate_axes: bool = False,
    axes: tuple[int, int] = (2, 3),
    commuted_sub: bool = False,
    commuted_affine: bool = False,
    scale_mode: str = "nchw",
    bias_mode: str = "nhwc",
) -> tuple[ModelIR, dict[str, str]]:
    names = {
        key: f"{prefix}{key}"
        for key in (
            "upstream",
            "source",
            "pre_perm",
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
            "post_perm",
            "post_output",
            "bias",
            "inst_output",
            "output",
        )
    }
    model_ir = ModelIR(f"{prefix}instance_norm_post_bias")
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    source_shape = [1, 2, 4, 3]
    nchw_shape = [1, 3, 2, 4]
    reduced_shape = [1, 3, 1, 1]
    for name, shape in {
        names["source"]: source_shape,
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
        names["post_output"]: source_shape,
        names["inst_output"]: source_shape,
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
        dtype="INT64",
        data=np.asarray([0, 2, 3, 1], dtype=np.int64),
    )
    model_ir.tensors[names["axes1"]] = _tensor(
        names["axes1"],
        [2],
        dtype="INT64",
        data=np.asarray(axes, dtype=np.int64),
    )
    if separate_axes:
        names["axes2"] = f"{prefix}axes2"
        model_ir.tensors[names["axes2"]] = _tensor(
            names["axes2"],
            [2],
            dtype="INT32",
            data=np.asarray(tuple(reversed(axes)), dtype=np.int32),
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
    scale = _coefficient(
        [0.75, 1.25, -0.5],
        dtype=np_dtype,
        mode=scale_mode,
    )
    bias = _coefficient(
        [0.1, -0.2, 0.3],
        dtype=np_dtype,
        mode=bias_mode,
    )
    model_ir.tensors[names["scale"]] = _tensor(
        names["scale"],
        list(scale.shape),
        dtype=dtype,
        data=scale,
    )
    model_ir.tensors[names["bias"]] = _tensor(
        names["bias"],
        list(bias.shape),
        dtype=dtype,
        data=bias,
    )
    if produced_source:
        model_ir.inputs = [names["upstream"]]
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            source_shape,
            dtype=dtype,
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["source"]])
        )
    else:
        model_ir.inputs = [names["source"]]

    def _binary(left: str, right: str, commute: bool) -> list[str]:
        values = [left, right]
        if commute:
            values.reverse()
        return values

    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [names["source"], names["pre_perm"]],
                [names["x"]],
            ),
            OperatorIR(
                "MEAN",
                [names["x"], names["axes1"]],
                [names["mean1"]],
                {"keepDims": True},
            ),
            OperatorIR(
                "SUB",
                _binary(names["x"], names["mean1"], commuted_sub),
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
                {"keepDims": True},
            ),
            OperatorIR(
                "ADD",
                _binary(
                    names["mean2"],
                    names["epsilon"],
                    commuted_affine,
                ),
                [names["variance_epsilon"]],
            ),
            OperatorIR(
                "SQRT",
                [names["variance_epsilon"]],
                [names["std"]],
            ),
            OperatorIR(
                "DIV",
                [names["one"], names["std"]],
                [names["inverse_std"]],
            ),
            OperatorIR(
                "MUL",
                _binary(
                    names["centered"],
                    names["inverse_std"],
                    commuted_affine,
                ),
                [names["normalized"]],
            ),
            OperatorIR(
                "MUL",
                _binary(
                    names["normalized"],
                    names["scale"],
                    commuted_affine,
                ),
                [names["scaled"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["scaled"], names["post_perm"]],
                [names["post_output"]],
            ),
            OperatorIR(
                "ADD",
                _binary(
                    names["post_output"],
                    names["bias"],
                    commuted_affine,
                ),
                [names["inst_output"]],
            ),
            OperatorIR("RELU", [names["inst_output"]], [names["output"]]),
        ]
    )
    model_ir.outputs = [names["output"]]
    model_ir.metadata["assume_channel_last_layout_tensor_names"] = ["existing_hint"]
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
        elif operator.op_type == "RELU":
            output = np.maximum(inputs[0], 0)
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


def _operator(model_ir: ModelIR, output_name: str) -> OperatorIR:
    return next(op for op in model_ir.operators if output_name in op.outputs)


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


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize(
    "separate_axes,axes,commuted_sub,commuted_affine,scale_mode,bias_mode",
    (
        (False, (2, 3), False, False, "nchw", "nhwc"),
        (False, (-2, -1), True, True, "nchw", "nhwc"),
        (True, (3, 2), False, True, "nchw", "scalar"),
        (True, (-1, -2), True, False, "scalar", "nhwc"),
        (False, (2, 3), False, True, "nchw", "scalar"),
        (True, (-2, -1), False, False, "scalar", "nhwc"),
    ),
)
def test_post_bias_instance_norm_is_indexed_and_numerically_equivalent(
    dtype: str,
    produced_source: bool,
    separate_axes: bool,
    axes: tuple[int, int],
    commuted_sub: bool,
    commuted_affine: bool,
    scale_mode: str,
    bias_mode: str,
) -> None:
    model_ir, names = _build_model(
        dtype=dtype,
        produced_source=produced_source,
        separate_axes=separate_axes,
        axes=axes,
        commuted_sub=commuted_sub,
        commuted_affine=commuted_affine,
        scale_mode=scale_mode,
        bias_mode=bias_mode,
    )
    feed_name = names["upstream"] if produced_source else names["source"]
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    feeds = {
        feed_name: np.random.default_rng(79)
        .normal(size=model_ir.tensors[feed_name].shape)
        .astype(np_dtype)
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert all(op.op_type != "TRANSPOSE" for op in model_ir.operators)
    assert _operator(model_ir, names["mean1"]).inputs[0] == names["source"]
    assert names["source"] in _operator(model_ir, names["centered"]).inputs
    assert names["scaled"] in _operator(model_ir, names["inst_output"]).inputs
    assert names["post_output"] not in model_ir.tensors
    assert names["x"] not in model_ir.tensors
    np.testing.assert_array_equal(
        model_ir.tensors[_operator(model_ir, names["mean1"]).inputs[1]].data,
        np.asarray([1, 2]),
    )
    if scale_mode == "nchw":
        assert list(model_ir.tensors[names["scale"]].shape) == [1, 1, 1, 3]
    if bias_mode == "nchw":
        assert list(model_ir.tensors[names["bias"]].shape) == [1, 1, 1, 3]
    actual = _evaluate(model_ir, feeds)
    tolerance = 3e-3 if dtype == "FLOAT16" else 1e-6
    np.testing.assert_allclose(
        actual[names["output"]],
        expected[names["output"]],
        rtol=tolerance,
        atol=tolerance,
    )
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


@pytest.mark.parametrize("dynamic_axis", (1, 2, 3))
def test_post_bias_instance_norm_preserves_dynamic_signatures(
    dynamic_axis: int,
) -> None:
    model_ir, names = _build_model()
    model_ir.tensors[names["source"]].shape_signature[dynamic_axis] = -1
    nchw_axis = (0, 3, 1, 2).index(dynamic_axis)
    for name in (
        "x",
        "centered",
        "squared",
        "normalized",
        "scaled",
    ):
        model_ir.tensors[names[name]].shape_signature[nchw_axis] = -1
    if dynamic_axis == 3:
        for name in (
            "mean1",
            "mean2",
            "variance_epsilon",
            "std",
            "inverse_std",
        ):
            model_ir.tensors[names[name]].shape_signature[1] = -1
    for name in ("post_output", "inst_output", "output"):
        model_ir.tensors[names[name]].shape_signature[dynamic_axis] = -1

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    assert model_ir.tensors[names["scaled"]].shape_signature[dynamic_axis] == -1
    assert validate_model_ir_invariants(model_ir) == []


def test_post_bias_instance_norm_preserves_public_bias_add_output() -> None:
    model_ir, names = _build_model()
    relu = _operator(model_ir, names["output"])
    model_ir.operators.remove(relu)
    model_ir.outputs = [names["inst_output"]]
    feed = np.random.default_rng(83).normal(size=[1, 2, 4, 3]).astype(np.float32)
    expected = _evaluate(copy.deepcopy(model_ir), {names["source"]: feed})

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    actual = _evaluate(model_ir, {names["source"]: feed})
    np.testing.assert_allclose(
        actual[names["inst_output"]],
        expected[names["inst_output"]],
        rtol=1e-6,
        atol=1e-6,
    )
    assert validate_model_ir_invariants(model_ir) == []


def test_post_bias_instance_norm_clones_shared_changed_constants() -> None:
    model_ir, names = _build_model(separate_axes=True)
    original_scale = np.asarray(model_ir.tensors[names["scale"]].data).copy()
    model_ir.tensors["axes_side"] = _tensor("axes_side", [2], dtype="INT64")
    model_ir.tensors["scale_side"] = _tensor("scale_side", [1, 3, 1, 1])
    model_ir.operators.extend(
        [
            OperatorIR("IDENTITY", [names["axes1"]], ["axes_side"]),
            OperatorIR("IDENTITY", [names["scale"]], ["scale_side"]),
        ]
    )
    model_ir.outputs.extend(["axes_side", "scale_side"])

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    np.testing.assert_array_equal(model_ir.tensors[names["scale"]].data, original_scale)
    mean1 = _operator(model_ir, names["mean1"])
    scale = _operator(model_ir, names["scaled"])
    assert mean1.inputs[1] != names["axes1"]
    assert names["scale"] not in scale.inputs
    np.testing.assert_array_equal(
        model_ir.tensors[mean1.inputs[1]].data,
        np.asarray([1, 2], dtype=np.int64),
    )
    assert list(model_ir.tensors[scale.inputs[1]].shape) == [1, 1, 1, 3]
    assert validate_model_ir_invariants(model_ir) == []


def test_post_bias_instance_norm_supports_one_shared_scale_and_bias() -> None:
    model_ir, names = _build_model()
    bias = _operator(model_ir, names["inst_output"])
    bias.inputs[bias.inputs.index(names["bias"])] = names["scale"]
    del model_ir.tensors[names["bias"]]

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    assert list(model_ir.tensors[names["scale"]].shape) == [1, 1, 1, 3]
    assert names["scale"] in _operator(model_ir, names["scaled"]).inputs
    assert names["scale"] in _operator(model_ir, names["inst_output"]).inputs


def test_post_bias_instance_norm_preserves_legacy_coefficient_layout_inputs() -> None:
    model_ir, names = _build_model(scale_mode="nhwc", bias_mode="nchw")

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    assert list(model_ir.tensors[names["scale"]].shape) == [1, 1, 1, 3]
    assert list(model_ir.tensors[names["bias"]].shape) == [1, 1, 1, 3]
    assert validate_model_ir_invariants(model_ir) == []


def test_post_bias_instance_norm_rewrites_multiple_chains_with_total_cap() -> None:
    first, first_names = _build_model(prefix="a_")
    second, second_names = _build_model(prefix="b_")
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    graph_index = ModelIRGraphIndex(first)

    first_stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
        first,
        graph_index=graph_index,
        max_rewrites=1,
    )
    second_stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
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
        "pre_perm_quantized",
        "source_public_output",
        "source_unbound",
        "pre_public",
        "pre_shape",
        "mean_axes",
        "mean_axes_quantized",
        "mean_keepdims",
        "mean_fanout",
        "sub_input",
        "centered_fanout",
        "negative_epsilon",
        "nonfinite_epsilon",
        "produced_epsilon",
        "nonunit_numerator",
        "reversed_div",
        "inverse_std_fanout",
        "scale_shape",
        "scale_dtype",
        "scale_quantized",
        "nonfinite_scale",
        "produced_scale",
        "bias_shape",
        "bias_dtype",
        "bias_public",
        "post_perm",
        "post_perm_quantized",
        "post_public",
        "post_fanout",
        "post_shape",
        "tensor_quantized",
        "duplicate_pre_output",
        "duplicate_output",
        "backward_post",
        "backward_output_consumer",
    ),
)
def test_post_bias_instance_norm_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model()
    mean1 = _operator(model_ir, names["mean1"])
    div = _operator(model_ir, names["inverse_std"])
    post = _operator(model_ir, names["post_output"])
    if case == "pre_perm_quantized":
        model_ir.tensors[names["pre_perm"]].quantization = {"scale": [1.0]}
    elif case == "source_public_output":
        model_ir.outputs.append(names["source"])
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "pre_public":
        model_ir.outputs.append(names["x"])
    elif case == "pre_shape":
        model_ir.tensors[names["x"]].shape[2] = 3
    elif case == "mean_axes":
        model_ir.tensors[names["axes1"]].data[:] = [1, 2]
    elif case == "mean_axes_quantized":
        model_ir.tensors[names["axes1"]].quantization = {"scale": [1.0]}
    elif case == "mean_keepdims":
        mean1.options["keepDims"] = False
    elif case == "mean_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 1, 1])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["mean1"]], ["side"]))
        model_ir.outputs.append("side")
    elif case == "sub_input":
        _operator(model_ir, names["centered"]).inputs[0] = names["scale"]
    elif case == "centered_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["centered"]], ["side"]))
        model_ir.outputs.append("side")
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    elif case == "nonfinite_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = np.nan
    elif case == "produced_epsilon":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["one"]], [names["epsilon"]])
        )
    elif case == "nonunit_numerator":
        model_ir.tensors[names["one"]].data[0] = 2.0
    elif case == "reversed_div":
        div.inputs.reverse()
    elif case == "inverse_std_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inverse_std"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "scale_shape":
        model_ir.tensors[names["scale"]].shape = [3]
        model_ir.tensors[names["scale"]].shape_signature = [3]
        model_ir.tensors[names["scale"]].data = np.asarray(
            [0.75, 1.25, -0.5], dtype=np.float32
        )
    elif case == "scale_dtype":
        model_ir.tensors[names["scale"]].dtype = "FLOAT16"
        model_ir.tensors[names["scale"]].data = np.asarray(
            model_ir.tensors[names["scale"]].data,
            dtype=np.float16,
        )
    elif case == "scale_quantized":
        model_ir.tensors[names["scale"]].quantization = {"scale": [0.25]}
    elif case == "nonfinite_scale":
        model_ir.tensors[names["scale"]].data.reshape(-1)[0] = np.nan
    elif case == "produced_scale":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["scale"]])
        )
    elif case == "bias_shape":
        model_ir.tensors[names["bias"]].shape = [3]
        model_ir.tensors[names["bias"]].shape_signature = [3]
        model_ir.tensors[names["bias"]].data = np.asarray(
            [0.1, -0.2, 0.3], dtype=np.float32
        )
    elif case == "bias_dtype":
        model_ir.tensors[names["bias"]].dtype = "FLOAT16"
        model_ir.tensors[names["bias"]].data = np.asarray(
            model_ir.tensors[names["bias"]].data,
            dtype=np.float16,
        )
    elif case == "bias_public":
        model_ir.inputs.append(names["bias"])
    elif case == "post_perm":
        model_ir.tensors[names["post_perm"]].data[:] = [0, 3, 1, 2]
    elif case == "post_perm_quantized":
        model_ir.tensors[names["post_perm"]].quantization = {"scale": [1.0]}
    elif case == "post_public":
        model_ir.outputs.append(names["post_output"])
    elif case == "post_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 2, 4, 3])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["post_output"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "post_shape":
        model_ir.tensors[names["post_output"]].shape[1] = 3
    elif case == "tensor_quantized":
        model_ir.tensors[names["centered"]].quantization = {"scale": [0.25]}
    elif case == "duplicate_pre_output":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["x"]])
        )
    elif case == "duplicate_output":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["inst_output"]])
        )
    elif case == "backward_post":
        model_ir.operators.remove(post)
        model_ir.operators.insert(model_ir.operators.index(mean1), post)
    elif case == "backward_output_consumer":
        relu = _operator(model_ir, names["output"])
        model_ir.operators.remove(relu)
        model_ir.operators.insert(model_ir.operators.index(post), relu)
    model_ir.tensors["unused"] = _tensor("unused", [1])
    before = repr(model_ir)

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(model_ir)

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_post_bias_instance_norm_clone_collision_is_transactional(
    monkeypatch,
) -> None:
    model_ir, names = _build_model()
    side = _tensor("scale_side", [1, 3, 1, 1])
    model_ir.tensors[side.name] = side
    model_ir.operators.append(OperatorIR("IDENTITY", [names["scale"]], [side.name]))
    model_ir.outputs.append(side.name)
    original_resolve = post_bias_module._resolve_candidate
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

    monkeypatch.setattr(post_bias_module, "_resolve_candidate", _resolve_with_collision)
    before_ops = repr(model_ir.operators)
    before_scale = np.asarray(model_ir.tensors[names["scale"]].data).copy()

    stats = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(model_ir)

    assert stats == {_STATS: 0}
    assert repr(model_ir.operators) == before_ops
    np.testing.assert_array_equal(model_ir.tensors[names["scale"]].data, before_scale)


def test_post_bias_preflight_avoids_index_construction(monkeypatch) -> None:
    model_ir = ModelIR("insufficient_post_bias_topology")

    def _unexpected_index(*args, **kwargs):
        raise AssertionError("index must not be constructed")

    monkeypatch.setattr(post_bias_module, "ModelIRGraphIndex", _unexpected_index)

    assert optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
        model_ir
    ) == {_STATS: 0}


def test_post_bias_counter_is_complete_mutation_evidence(monkeypatch) -> None:
    model_ir, _ = _build_model()
    prune_calls: list[tuple[ModelIR, LayoutState | None]] = []

    def record_prune(
        active_model_ir: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> None:
        prune_calls.append((active_model_ir, layout_state))

    monkeypatch.setattr(post_bias_module, "_prune_unused_tensors", record_prune)

    first = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
        model_ir
    )
    second = optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains(
        model_ir
    )

    assert first == {_STATS: 1}
    assert second == {_STATS: 0}
    assert prune_calls == [(model_ir, None)]
