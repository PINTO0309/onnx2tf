from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_norm_dual_stats_layout as dual_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.instance_norm_dual_stats_layout import (
    optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains,
)
from tests.test_flatbuffer_direct_indexed_instance_norm_post_bias_layout import (
    _assert_index_current,
    _coefficient,
    _operator,
    _tensor,
)


_STATS = (
    "optimized_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
)


def _build_dual_stats_model(
    *,
    prefix: str = "",
    dtype: str = "FLOAT32",
    produced_source: bool = False,
    tail_mode: str = "direct",
    separate_axes: bool = False,
    spatial_axes: tuple[int, int] = (2, 3),
    global_axes: tuple[int, int, int] = (1, 2, 3),
    commuted_sub: bool = False,
    commuted_affine: bool = False,
    scale_modes: tuple[str, str] = ("nchw", "nchw"),
    gamma_mode: str = "nchw",
    beta_mode: str = "nchw",
) -> tuple[ModelIR, dict[str, str]]:
    assert tail_mode in {"direct", "residual", "produced_residual"}
    names = {
        key: f"{prefix}{key}"
        for key in (
            "upstream",
            "source",
            "pre_perm",
            "x",
            "s_axes1",
            "s_axes2",
            "g_axes1",
            "g_axes2",
            "s_mean1",
            "s_centered",
            "s_squared",
            "s_mean2",
            "s_factor",
            "s_factored",
            "s_epsilon",
            "s_variance_epsilon",
            "s_std",
            "s_divided",
            "s_scale",
            "s_scaled",
            "g_mean1",
            "g_centered",
            "g_squared",
            "g_mean2",
            "g_factor",
            "g_factored",
            "g_epsilon",
            "g_variance_epsilon",
            "g_std",
            "g_divided",
            "g_scale",
            "g_scaled",
            "blend",
            "gamma_vector",
            "gamma_shape",
            "gamma",
            "blended_scaled",
            "beta_vector",
            "beta_shape",
            "beta",
            "inst_output",
            "residual_upstream",
            "residual_source",
            "residual_nchw",
            "tail_output",
            "post_perm",
            "post_output",
            "output",
        )
    }
    model_ir = ModelIR(f"{prefix}dual_stats_instance_norm")
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    source_shape = [1, 2, 4, 3]
    nchw_shape = [1, 3, 2, 4]
    spatial_shape = [1, 3, 1, 1]
    global_shape = [1, 1, 1, 1]

    model_ir.tensors[names["source"]] = _tensor(
        names["source"], source_shape, dtype=dtype
    )
    model_ir.tensors[names["x"]] = _tensor(names["x"], nchw_shape, dtype=dtype)
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
    axes_specs = (
        ("s_axes1", spatial_axes, np.int64),
        ("g_axes1", global_axes, np.int32),
    )
    for key, axes, np_axes_dtype in axes_specs:
        model_ir.tensors[names[key]] = _tensor(
            names[key],
            [len(axes)],
            dtype="INT64" if np_axes_dtype is np.int64 else "INT32",
            data=np.asarray(axes, dtype=np_axes_dtype),
    )
    if separate_axes:
        for key, axes, np_axes_dtype in (
            ("s_axes2", spatial_axes, np.int32),
            ("g_axes2", global_axes, np.int64),
        ):
            model_ir.tensors[names[key]] = _tensor(
                names[key],
                [len(axes)],
                dtype="INT32" if np_axes_dtype is np.int32 else "INT64",
                data=np.asarray(tuple(reversed(axes)), dtype=np_axes_dtype),
            )
    else:
        names["s_axes2"] = names["s_axes1"]
        names["g_axes2"] = names["g_axes1"]

    if produced_source:
        model_ir.inputs = [names["upstream"]]
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"], source_shape, dtype=dtype
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["source"]])
        )
    else:
        model_ir.inputs = [names["source"]]
    model_ir.operators.append(
        OperatorIR(
            "TRANSPOSE",
            [names["source"], names["pre_perm"]],
            [names["x"]],
        )
    )

    def _binary(left: str, right: str) -> list[str]:
        values = [left, right]
        if commuted_affine:
            values.reverse()
        return values

    def _add_path(label: str, reduced_shape: list[int], scale_mode: str) -> None:
        for suffix, shape in (
            ("mean1", reduced_shape),
            ("centered", nchw_shape),
            ("squared", nchw_shape),
            ("mean2", reduced_shape),
            ("factored", reduced_shape),
            ("variance_epsilon", reduced_shape),
            ("std", reduced_shape),
            ("divided", nchw_shape),
            ("scaled", nchw_shape),
        ):
            key = f"{label}_{suffix}"
            model_ir.tensors[names[key]] = _tensor(
                names[key], shape, dtype=dtype
            )
        for suffix, value in (("factor", 0.75), ("epsilon", 0.01)):
            key = f"{label}_{suffix}"
            model_ir.tensors[names[key]] = _tensor(
                names[key],
                [1],
                dtype=dtype,
                data=np.asarray([value], dtype=np_dtype),
            )
        scale = _coefficient(
            [0.5, 1.25, -0.75] if label == "s" else [1.5, -0.25, 0.8],
            dtype=np_dtype,
            mode=scale_mode,
        )
        model_ir.tensors[names[f"{label}_scale"]] = _tensor(
            names[f"{label}_scale"],
            list(scale.shape),
            dtype=dtype,
            data=scale,
        )
        axes1 = names[f"{label}_axes1"]
        axes2 = names[f"{label}_axes2"]
        sub_inputs = [names["x"], names[f"{label}_mean1"]]
        if commuted_sub:
            sub_inputs.reverse()
        model_ir.operators.extend(
            [
                OperatorIR(
                    "MEAN",
                    [names["x"], axes1],
                    [names[f"{label}_mean1"]],
                    {"keepDims": True},
                ),
                OperatorIR(
                    "SUB",
                    sub_inputs,
                    [names[f"{label}_centered"]],
                ),
                OperatorIR(
                    "MUL",
                    [names[f"{label}_centered"], names[f"{label}_centered"]],
                    [names[f"{label}_squared"]],
                ),
                OperatorIR(
                    "MEAN",
                    [names[f"{label}_squared"], axes2],
                    [names[f"{label}_mean2"]],
                    {"keepDims": True},
                ),
                OperatorIR(
                    "MUL",
                    _binary(names[f"{label}_mean2"], names[f"{label}_factor"]),
                    [names[f"{label}_factored"]],
                ),
                OperatorIR(
                    "ADD",
                    _binary(
                        names[f"{label}_factored"],
                        names[f"{label}_epsilon"],
                    ),
                    [names[f"{label}_variance_epsilon"]],
                ),
                OperatorIR(
                    "SQRT",
                    [names[f"{label}_variance_epsilon"]],
                    [names[f"{label}_std"]],
                ),
                OperatorIR(
                    "DIV",
                    [names[f"{label}_centered"], names[f"{label}_std"]],
                    [names[f"{label}_divided"]],
                ),
                OperatorIR(
                    "MUL",
                    _binary(names[f"{label}_divided"], names[f"{label}_scale"]),
                    [names[f"{label}_scaled"]],
                ),
            ]
        )

    _add_path("s", spatial_shape, scale_modes[0])
    _add_path("g", global_shape, scale_modes[1])
    for key in ("blend", "blended_scaled", "inst_output"):
        model_ir.tensors[names[key]] = _tensor(
            names[key], nchw_shape, dtype=dtype
        )

    def _add_affine_coefficient(key: str, mode: str, values: list[float]) -> None:
        if mode in {"reshape1", "reshape2"}:
            vector_shape = [3] if mode == "reshape1" else [1, 3]
            vector = np.asarray(values, dtype=np_dtype).reshape(vector_shape)
            model_ir.tensors[names[f"{key}_vector"]] = _tensor(
                names[f"{key}_vector"],
                vector_shape,
                dtype=dtype,
                data=vector,
            )
            model_ir.tensors[names[f"{key}_shape"]] = _tensor(
                names[f"{key}_shape"],
                [4],
                dtype="INT32",
                data=np.asarray([1, 3, 1, 1], dtype=np.int32),
            )
            model_ir.tensors[names[key]] = _tensor(
                names[key], [1, 3, 1, 1], dtype=dtype
            )
            model_ir.operators.append(
                OperatorIR(
                    "RESHAPE",
                    [names[f"{key}_vector"], names[f"{key}_shape"]],
                    [names[key]],
                )
            )
        else:
            data = _coefficient(values, dtype=np_dtype, mode=mode)
            model_ir.tensors[names[key]] = _tensor(
                names[key], list(data.shape), dtype=dtype, data=data
            )

    _add_affine_coefficient("gamma", gamma_mode, [1.1, 0.9, -0.6])
    _add_affine_coefficient("beta", beta_mode, [0.1, -0.2, 0.05])
    model_ir.operators.extend(
        [
            OperatorIR(
                "ADD",
                _binary(names["s_scaled"], names["g_scaled"]),
                [names["blend"]],
            ),
            OperatorIR(
                "MUL",
                _binary(names["blend"], names["gamma"]),
                [names["blended_scaled"]],
            ),
            OperatorIR(
                "ADD",
                _binary(names["blended_scaled"], names["beta"]),
                [names["inst_output"]],
            ),
        ]
    )

    post_input = names["inst_output"]
    if tail_mode != "direct":
        for key, shape in (
            ("residual_source", source_shape),
            ("residual_nchw", nchw_shape),
            ("tail_output", nchw_shape),
        ):
            model_ir.tensors[names[key]] = _tensor(names[key], shape, dtype=dtype)
        if tail_mode == "produced_residual":
            model_ir.tensors[names["residual_upstream"]] = _tensor(
                names["residual_upstream"], source_shape, dtype=dtype
            )
            model_ir.inputs.append(names["residual_upstream"])
            model_ir.operators.append(
                OperatorIR(
                    "IDENTITY",
                    [names["residual_upstream"]],
                    [names["residual_source"]],
                )
            )
        else:
            model_ir.inputs.append(names["residual_source"])
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [names["residual_source"], names["pre_perm"]],
                    [names["residual_nchw"]],
                ),
                OperatorIR(
                    "ADD",
                    _binary(names["inst_output"], names["residual_nchw"]),
                    [names["tail_output"]],
                ),
            ]
        )
        post_input = names["tail_output"]
    for key in ("post_output", "output"):
        model_ir.tensors[names[key]] = _tensor(
            names[key], source_shape, dtype=dtype
        )
    model_ir.operators.extend(
        [
            OperatorIR(
                "TRANSPOSE",
                [post_input, names["post_perm"]],
                [names["post_output"]],
            ),
            OperatorIR("RELU", [names["post_output"]], [names["output"]]),
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
        elif operator.op_type == "RESHAPE":
            output = np.reshape(
                inputs[0], tuple(int(value) for value in inputs[1].reshape(-1))
            )
        elif operator.op_type == "RELU":
            output = np.maximum(inputs[0], 0)
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("tail_mode", ("direct", "residual", "produced_residual"))
@pytest.mark.parametrize(
    "separate_axes,spatial_axes,global_axes,commuted_sub,commuted_affine,scale_modes,gamma_mode,beta_mode",
    (
        (False, (2, 3), (1, 2, 3), False, False, ("nchw", "nchw"), "nchw", "nchw"),
        (False, (-2, -1), (-3, -2, -1), True, True, ("scalar", "nchw"), "reshape1", "nchw"),
        (True, (3, 2), (2, 3, 1), False, True, ("nchw", "scalar"), "nchw", "reshape2"),
        (True, (-1, -2), (-2, -1, -3), True, False, ("nchw", "nchw"), "reshape1", "reshape2"),
    ),
)
def test_dual_stats_instance_norm_is_indexed_and_numerically_equivalent(
    dtype: str,
    produced_source: bool,
    tail_mode: str,
    separate_axes: bool,
    spatial_axes: tuple[int, int],
    global_axes: tuple[int, int, int],
    commuted_sub: bool,
    commuted_affine: bool,
    scale_modes: tuple[str, str],
    gamma_mode: str,
    beta_mode: str,
) -> None:
    model_ir, names = _build_dual_stats_model(
        dtype=dtype,
        produced_source=produced_source,
        tail_mode=tail_mode,
        separate_axes=separate_axes,
        spatial_axes=spatial_axes,
        global_axes=global_axes,
        commuted_sub=commuted_sub,
        commuted_affine=commuted_affine,
        scale_modes=scale_modes,
        gamma_mode=gamma_mode,
        beta_mode=beta_mode,
    )
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    rng = np.random.default_rng(101)
    main_feed = names["upstream"] if produced_source else names["source"]
    feeds = {main_feed: rng.normal(size=[1, 2, 4, 3]).astype(np_dtype)}
    if tail_mode != "direct":
        residual_feed = (
            names["residual_upstream"]
            if tail_mode == "produced_residual"
            else names["residual_source"]
        )
        feeds[residual_feed] = rng.normal(size=[1, 2, 4, 3]).astype(np_dtype)
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert not any(op.op_type == "TRANSPOSE" for op in model_ir.operators)
    assert _operator(model_ir, names["s_mean1"]).inputs[0] == names["source"]
    assert _operator(model_ir, names["g_mean1"]).inputs[0] == names["source"]
    output_owner = _operator(model_ir, names["post_output"])
    if tail_mode == "direct":
        assert output_owner.op_type == "ADD"
        assert names["blended_scaled"] in output_owner.inputs
    else:
        assert output_owner.op_type == "ADD"
        assert names["inst_output"] in output_owner.inputs
        assert names["residual_source"] in output_owner.inputs
    for key in ("s_axes1", "s_axes2"):
        np.testing.assert_array_equal(
            np.asarray(model_ir.tensors[names[key]].data).reshape(-1),
            np.asarray([1, 2]),
        )
    for key in ("s_scale", "g_scale"):
        shape = list(model_ir.tensors[names[key]].shape)
        assert shape in ([1], [1, 1, 1, 3])
    if gamma_mode.startswith("reshape"):
        assert names["gamma_vector"] in _operator(
            model_ir, names["blended_scaled"]
        ).inputs
        assert names["gamma"] not in model_ir.tensors
    if beta_mode.startswith("reshape"):
        beta_owner = (
            output_owner
            if tail_mode == "direct"
            else _operator(model_ir, names["inst_output"])
        )
        assert names["beta_vector"] in beta_owner.inputs
        assert names["beta"] not in model_ir.tensors
    actual = _evaluate(model_ir, feeds)
    tolerance = 6e-3 if dtype == "FLOAT16" else 2e-6
    np.testing.assert_allclose(
        actual[names["output"]],
        expected[names["output"]],
        rtol=tolerance,
        atol=tolerance,
    )
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert names["post_output"] in model_ir.metadata[
        "assume_channel_last_layout_tensor_names"
    ]


def test_dual_stats_accepts_already_nhwc_direct_coefficients() -> None:
    model_ir, names = _build_dual_stats_model(
        scale_modes=("nhwc", "nhwc"),
        gamma_mode="nhwc",
        beta_mode="nhwc",
    )

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    for key in ("s_scale", "g_scale", "gamma", "beta"):
        assert list(model_ir.tensors[names[key]].shape) == [1, 1, 1, 3]
    assert validate_model_ir_invariants(model_ir) == []


def _set_dynamic_signature(
    model_ir: ModelIR,
    names: dict[str, str],
    dynamic_axis: int,
) -> None:
    source_signature = [1, 2, 4, 3]
    source_signature[dynamic_axis] = -1
    nchw_signature = [
        source_signature[index] for index in (0, 3, 1, 2)
    ]
    spatial_signature = [nchw_signature[0], nchw_signature[1], 1, 1]
    global_signature = [nchw_signature[0], 1, 1, 1]
    for key in ("source", "post_output", "output", "residual_source"):
        if names[key] in model_ir.tensors:
            model_ir.tensors[names[key]].shape_signature = list(source_signature)
    for key in ("upstream", "residual_upstream"):
        if names[key] in model_ir.tensors:
            model_ir.tensors[names[key]].shape_signature = list(source_signature)
    full_keys = (
        "x",
        "s_centered",
        "s_squared",
        "s_divided",
        "s_scaled",
        "g_centered",
        "g_squared",
        "g_divided",
        "g_scaled",
        "blend",
        "blended_scaled",
        "inst_output",
        "residual_nchw",
        "tail_output",
    )
    for key in full_keys:
        if names[key] in model_ir.tensors:
            model_ir.tensors[names[key]].shape_signature = list(nchw_signature)
    for label, signature in (("s", spatial_signature), ("g", global_signature)):
        for suffix in (
            "mean1",
            "mean2",
            "factored",
            "variance_epsilon",
            "std",
        ):
            model_ir.tensors[names[f"{label}_{suffix}"]].shape_signature = list(
                signature
            )


@pytest.mark.parametrize("tail_mode", ("direct", "residual"))
@pytest.mark.parametrize("dynamic_axis", (1, 2, 3))
def test_dual_stats_preserves_dynamic_signatures(
    tail_mode: str,
    dynamic_axis: int,
) -> None:
    model_ir, names = _build_dual_stats_model(
        produced_source=True,
        tail_mode=tail_mode,
        gamma_mode="reshape1",
        beta_mode="reshape2",
    )
    _set_dynamic_signature(model_ir, names, dynamic_axis)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir,
        graph_index=graph_index,
    )

    assert stats == {_STATS: 1}
    source_signature = list(model_ir.tensors[names["source"]].shape_signature)
    assert list(model_ir.tensors[names["s_centered"]].shape_signature) == (
        source_signature
    )
    assert list(model_ir.tensors[names["g_centered"]].shape_signature) == (
        source_signature
    )
    expected_spatial = [source_signature[0], 1, 1, source_signature[3]]
    assert list(model_ir.tensors[names["s_mean1"]].shape_signature) == (
        expected_spatial
    )
    assert list(model_ir.tensors[names["g_mean1"]].shape_signature) == [
        source_signature[0],
        1,
        1,
        1,
    ]
    assert list(model_ir.tensors[names["post_output"]].shape_signature) == (
        source_signature
    )
    assert validate_model_ir_invariants(model_ir) == []
    _assert_index_current(model_ir, graph_index)


@pytest.mark.parametrize("repeated_slots", (False, True))
def test_dual_stats_preserves_downstream_fanout_without_resize(
    repeated_slots: bool,
) -> None:
    model_ir, names = _build_dual_stats_model()
    downstream = _operator(model_ir, names["output"])
    if repeated_slots:
        downstream.op_type = "ADD"
        downstream.inputs = [names["post_output"], names["post_output"]]
    side_name = "post_side"
    model_ir.tensors[side_name] = _tensor(side_name, [1, 2, 4, 3])
    model_ir.operators.append(OperatorIR("ABS", [names["post_output"]], [side_name]))
    model_ir.outputs.append(side_name)

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert all(name == names["post_output"] for name in downstream.inputs)
    assert _operator(model_ir, side_name).inputs == [names["post_output"]]
    assert validate_model_ir_invariants(model_ir) == []


def test_dual_stats_clones_shared_spatial_axes_transactionally() -> None:
    model_ir, names = _build_dual_stats_model()
    side_name = "axes_side"
    model_ir.tensors[side_name] = _tensor(side_name, [2])
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["s_axes1"]], [side_name])
    )
    model_ir.outputs.append(side_name)

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    mean1 = _operator(model_ir, names["s_mean1"])
    mean2 = _operator(model_ir, names["s_mean2"])
    clone_name = str(mean1.inputs[1])
    assert clone_name == str(mean2.inputs[1])
    assert clone_name != names["s_axes1"]
    np.testing.assert_array_equal(model_ir.tensors[clone_name].data, [1, 2])
    np.testing.assert_array_equal(model_ir.tensors[names["s_axes1"]].data, [2, 3])
    assert _operator(model_ir, side_name).inputs == [names["s_axes1"]]


def test_dual_stats_updates_one_coefficient_shared_by_all_direct_uses() -> None:
    model_ir, names = _build_dual_stats_model()
    shared_name = names["s_scale"]
    for output_name, data_name in (
        (names["g_scaled"], names["g_divided"]),
        (names["blended_scaled"], names["blend"]),
        (names["inst_output"], names["blended_scaled"]),
    ):
        operator = _operator(model_ir, output_name)
        operator.inputs = [
            data_name if str(name) == data_name else shared_name
            for name in operator.inputs
        ]
    for key in ("g_scale", "gamma", "beta"):
        del model_ir.tensors[names[key]]

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert list(model_ir.tensors[shared_name].shape) == [1, 1, 1, 3]
    assert all(
        shared_name in _operator(model_ir, output_name).inputs
        for output_name in (
            names["s_scaled"],
            names["g_scaled"],
            names["blended_scaled"],
            names["post_output"],
        )
    )
    assert validate_model_ir_invariants(model_ir) == []


def test_dual_stats_rewrites_multiple_chains_with_total_cap() -> None:
    first, first_names = _build_dual_stats_model(prefix="a_")
    second, second_names = _build_dual_stats_model(
        prefix="b_", tail_mode="residual"
    )
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    graph_index = ModelIRGraphIndex(first)

    first_stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        first,
        graph_index=graph_index,
        max_rewrites=1,
    )
    second_stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        first,
        graph_index=graph_index,
        max_rewrites=1,
    )

    assert first_stats == {_STATS: 1}
    assert second_stats == {_STATS: 1}
    assert _operator(first, first_names["s_mean1"]).inputs[0] == first_names["source"]
    assert _operator(first, second_names["s_mean1"]).inputs[0] == second_names["source"]
    _assert_index_current(first, graph_index)


@pytest.mark.parametrize(
    "case",
    (
        "pre_perm",
        "pre_perm_quantized",
        "pre_perm_produced",
        "source_public_output",
        "source_unbound",
        "source_dtype",
        "source_quantized",
        "pre_public",
        "pre_shape",
        "duplicate_pre",
        "x_extra_consumer",
        "x_repeated_consumer",
        "spatial_axes",
        "spatial_axes_quantized",
        "spatial_axes_public",
        "spatial_axes_produced",
        "global_axes",
        "global_axes_quantized",
        "global_axes_public",
        "global_axes_produced",
        "mean_keepdims",
        "mean_shape",
        "mean_dtype",
        "mean_quantized",
        "mean_public",
        "mean_fanout",
        "sub_op",
        "sub_inputs",
        "centered_shape",
        "centered_quantized",
        "centered_public",
        "centered_fanout",
        "square_op",
        "square_inputs",
        "squared_shape",
        "squared_fanout",
        "mean2_keepdims",
        "mean2_axes",
        "mean2_shape",
        "mean2_fanout",
        "factor_op",
        "factor_negative",
        "factor_nonfinite",
        "factor_shape",
        "factor_dtype",
        "factor_quantized",
        "factor_public",
        "factor_produced",
        "factored_shape",
        "factored_fanout",
        "epsilon_negative",
        "epsilon_nonfinite",
        "epsilon_dtype",
        "epsilon_quantized",
        "epsilon_public",
        "epsilon_produced",
        "sqrt_op",
        "sqrt_input",
        "sqrt_shape",
        "sqrt_fanout",
        "div_reversed",
        "div_shape",
        "div_public",
        "div_fanout",
        "scale_op",
        "scale_shape",
        "scale_dtype",
        "scale_quantized",
        "scale_public",
        "scale_produced",
        "scale_nonfinite",
        "scaled_shape",
        "scaled_public",
        "scaled_fanout",
        "global_scale_shape",
        "blend_op",
        "blend_inputs",
        "blend_shape",
        "blend_public",
        "blend_fanout",
        "blend_mul_op",
        "gamma_shape",
        "gamma_dtype",
        "gamma_quantized",
        "gamma_public",
        "gamma_produced",
        "gamma_nonfinite",
        "blend_mul_shape",
        "blend_mul_public",
        "blend_mul_fanout",
        "beta_shape",
        "beta_dtype",
        "beta_quantized",
        "beta_public",
        "beta_produced",
        "beta_nonfinite",
        "inst_shape",
        "inst_dtype",
        "inst_quantized",
        "inst_public",
        "inst_fanout",
        "reshape_op",
        "reshape_shape_value",
        "reshape_shape_dtype",
        "reshape_shape_quantized",
        "reshape_shape_public",
        "reshape_shape_produced",
        "reshape_output_shape",
        "reshape_output_dtype",
        "reshape_output_quantized",
        "reshape_output_public",
        "reshape_output_fanout",
        "reshape_source_shape",
        "reshape_source_dtype",
        "reshape_source_quantized",
        "reshape_source_unbound",
        "reshape_source_duplicate",
        "post_op",
        "post_perm",
        "post_perm_quantized",
        "post_input",
        "post_shape",
        "post_dtype",
        "post_quantized",
        "post_public",
        "post_duplicate",
        "post_backward_consumer",
        "backward_blend",
        "residual_pre_op",
        "residual_perm",
        "residual_shape",
        "residual_dtype",
        "residual_quantized",
        "residual_public",
        "residual_fanout",
        "residual_source_public_output",
        "residual_source_unbound",
        "tail_add_op",
        "tail_shape",
        "tail_dtype",
        "tail_quantized",
        "tail_public",
        "tail_fanout",
    ),
)
def test_dual_stats_rejects_unsafe_contracts_transactionally(case: str) -> None:
    reshape_case = case.startswith("reshape_")
    residual_case = case.startswith("residual_") or case.startswith("tail_")
    model_ir, names = _build_dual_stats_model(
        tail_mode="residual" if residual_case else "direct",
        gamma_mode="reshape1" if reshape_case else "nchw",
        beta_mode="reshape2" if reshape_case else "nchw",
    )
    mean1 = _operator(model_ir, names["s_mean1"])
    sub = _operator(model_ir, names["s_centered"])
    square = _operator(model_ir, names["s_squared"])
    mean2 = _operator(model_ir, names["s_mean2"])
    factor = _operator(model_ir, names["s_factored"])
    sqrt = _operator(model_ir, names["s_std"])
    div = _operator(model_ir, names["s_divided"])
    scale = _operator(model_ir, names["s_scaled"])
    global_scale = _operator(model_ir, names["g_scaled"])
    blend = _operator(model_ir, names["blend"])
    blend_mul = _operator(model_ir, names["blended_scaled"])
    post = _operator(model_ir, names["post_output"])
    downstream = _operator(model_ir, names["output"])
    if case == "pre_perm":
        model_ir.tensors[names["pre_perm"]].data[:] = [0, 2, 3, 1]
    elif case == "pre_perm_quantized":
        model_ir.tensors[names["pre_perm"]].quantization = {"scale": [1.0]}
    elif case == "pre_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_axes1"]], [names["pre_perm"]])
        )
    elif case == "source_public_output":
        model_ir.outputs.append(names["source"])
    elif case == "source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "source_dtype":
        model_ir.tensors[names["source"]].dtype = "FLOAT16"
    elif case == "source_quantized":
        model_ir.tensors[names["source"]].quantization = {"scale": [0.25]}
    elif case == "pre_public":
        model_ir.outputs.append(names["x"])
    elif case == "pre_shape":
        model_ir.tensors[names["x"]].shape[2] = 3
    elif case == "duplicate_pre":
        model_ir.operators.append(OperatorIR("IDENTITY", [names["source"]], [names["x"]]))
    elif case == "x_extra_consumer":
        model_ir.tensors["x_side"] = _tensor("x_side", [1, 3, 2, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["x"]], ["x_side"]))
    elif case == "x_repeated_consumer":
        sub.inputs = [names["x"], names["x"]]
    elif case == "spatial_axes":
        model_ir.tensors[names["s_axes1"]].data[:] = [1, 2]
    elif case == "spatial_axes_quantized":
        model_ir.tensors[names["s_axes1"]].quantization = {"scale": [1.0]}
    elif case == "spatial_axes_public":
        model_ir.outputs.append(names["s_axes1"])
    elif case == "spatial_axes_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["g_axes1"]], [names["s_axes1"]])
        )
    elif case == "global_axes":
        model_ir.tensors[names["g_axes1"]].data[:] = [0, 2, 3]
    elif case == "global_axes_quantized":
        model_ir.tensors[names["g_axes1"]].quantization = {"scale": [1.0]}
    elif case == "global_axes_public":
        model_ir.outputs.append(names["g_axes1"])
    elif case == "global_axes_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_axes1"]], [names["g_axes1"]])
        )
    elif case == "mean_keepdims":
        mean1.options["keepDims"] = False
    elif case == "mean_shape":
        model_ir.tensors[names["s_mean1"]].shape[1] = 2
    elif case == "mean_dtype":
        model_ir.tensors[names["s_mean1"]].dtype = "FLOAT16"
    elif case == "mean_quantized":
        model_ir.tensors[names["s_mean1"]].quantization = {"scale": [0.25]}
    elif case == "mean_public":
        model_ir.outputs.append(names["s_mean1"])
    elif case == "mean_fanout":
        model_ir.tensors["mean_side"] = _tensor("mean_side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_mean1"]], ["mean_side"])
        )
    elif case == "sub_op":
        sub.op_type = "ADD"
    elif case == "sub_inputs":
        sub.inputs[1] = names["g_mean1"]
    elif case == "centered_shape":
        model_ir.tensors[names["s_centered"]].shape[2] = 3
    elif case == "centered_quantized":
        model_ir.tensors[names["s_centered"]].quantization = {"scale": [0.25]}
    elif case == "centered_public":
        model_ir.outputs.append(names["s_centered"])
    elif case == "centered_fanout":
        model_ir.tensors["centered_side"] = _tensor("centered_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_centered"]], ["centered_side"])
        )
    elif case == "square_op":
        square.op_type = "ADD"
    elif case == "square_inputs":
        square.inputs[1] = names["g_centered"]
    elif case == "squared_shape":
        model_ir.tensors[names["s_squared"]].shape[2] = 3
    elif case == "squared_fanout":
        model_ir.tensors["square_side"] = _tensor("square_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_squared"]], ["square_side"])
        )
    elif case == "mean2_keepdims":
        mean2.options["keepDims"] = False
    elif case == "mean2_axes":
        mean2.inputs[1] = names["g_axes1"]
    elif case == "mean2_shape":
        model_ir.tensors[names["s_mean2"]].shape[1] = 2
    elif case == "mean2_fanout":
        model_ir.tensors["mean2_side"] = _tensor("mean2_side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_mean2"]], ["mean2_side"])
        )
    elif case == "factor_op":
        factor.op_type = "ADD"
    elif case == "factor_negative":
        model_ir.tensors[names["s_factor"]].data[0] = -1.0
    elif case == "factor_nonfinite":
        model_ir.tensors[names["s_factor"]].data[0] = np.nan
    elif case == "factor_shape":
        model_ir.tensors[names["s_factor"]].shape = [2]
        model_ir.tensors[names["s_factor"]].shape_signature = [2]
        model_ir.tensors[names["s_factor"]].data = np.asarray([1.0, 1.0], dtype=np.float32)
    elif case == "factor_dtype":
        model_ir.tensors[names["s_factor"]].dtype = "FLOAT16"
        model_ir.tensors[names["s_factor"]].data = np.asarray([0.75], dtype=np.float16)
    elif case == "factor_quantized":
        model_ir.tensors[names["s_factor"]].quantization = {"scale": [0.25]}
    elif case == "factor_public":
        model_ir.inputs.append(names["s_factor"])
    elif case == "factor_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_epsilon"]], [names["s_factor"]])
        )
    elif case == "factored_shape":
        model_ir.tensors[names["s_factored"]].shape[1] = 2
    elif case == "factored_fanout":
        model_ir.tensors["factor_side"] = _tensor("factor_side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_factored"]], ["factor_side"])
        )
    elif case == "epsilon_negative":
        model_ir.tensors[names["s_epsilon"]].data[0] = -0.01
    elif case == "epsilon_nonfinite":
        model_ir.tensors[names["s_epsilon"]].data[0] = np.inf
    elif case == "epsilon_dtype":
        model_ir.tensors[names["s_epsilon"]].dtype = "FLOAT16"
        model_ir.tensors[names["s_epsilon"]].data = np.asarray([0.01], dtype=np.float16)
    elif case == "epsilon_quantized":
        model_ir.tensors[names["s_epsilon"]].quantization = {"scale": [0.25]}
    elif case == "epsilon_public":
        model_ir.outputs.append(names["s_epsilon"])
    elif case == "epsilon_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["g_epsilon"]], [names["s_epsilon"]])
        )
    elif case == "sqrt_op":
        sqrt.op_type = "ABS"
    elif case == "sqrt_input":
        sqrt.inputs[0] = names["g_variance_epsilon"]
    elif case == "sqrt_shape":
        model_ir.tensors[names["s_std"]].shape[1] = 2
    elif case == "sqrt_fanout":
        model_ir.tensors["sqrt_side"] = _tensor("sqrt_side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_std"]], ["sqrt_side"])
        )
    elif case == "div_reversed":
        div.inputs.reverse()
    elif case == "div_shape":
        model_ir.tensors[names["s_divided"]].shape[2] = 3
    elif case == "div_public":
        model_ir.outputs.append(names["s_divided"])
    elif case == "div_fanout":
        model_ir.tensors["div_side"] = _tensor("div_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_divided"]], ["div_side"])
        )
    elif case == "scale_op":
        scale.op_type = "ADD"
    elif case == "scale_shape":
        model_ir.tensors[names["s_scale"]].shape = [3]
        model_ir.tensors[names["s_scale"]].shape_signature = [3]
        model_ir.tensors[names["s_scale"]].data = np.ones(3, dtype=np.float32)
    elif case == "scale_dtype":
        model_ir.tensors[names["s_scale"]].dtype = "FLOAT16"
        model_ir.tensors[names["s_scale"]].data = np.asarray(
            model_ir.tensors[names["s_scale"]].data, dtype=np.float16
        )
    elif case == "scale_quantized":
        model_ir.tensors[names["s_scale"]].quantization = {"scale": [0.25]}
    elif case == "scale_public":
        model_ir.inputs.append(names["s_scale"])
    elif case == "scale_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["s_scale"]])
        )
    elif case == "scale_nonfinite":
        model_ir.tensors[names["s_scale"]].data.reshape(-1)[0] = np.nan
    elif case == "scaled_shape":
        model_ir.tensors[names["s_scaled"]].shape[2] = 3
    elif case == "scaled_public":
        model_ir.outputs.append(names["s_scaled"])
    elif case == "scaled_fanout":
        model_ir.tensors["scaled_side"] = _tensor("scaled_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["s_scaled"]], ["scaled_side"])
        )
    elif case == "global_scale_shape":
        model_ir.tensors[names["g_scale"]].shape = [3]
        model_ir.tensors[names["g_scale"]].shape_signature = [3]
        model_ir.tensors[names["g_scale"]].data = np.ones(3, dtype=np.float32)
    elif case == "blend_op":
        blend.op_type = "MUL"
    elif case == "blend_inputs":
        blend.inputs = [names["s_scaled"], names["s_scaled"]]
    elif case == "blend_shape":
        model_ir.tensors[names["blend"]].shape[2] = 3
    elif case == "blend_public":
        model_ir.outputs.append(names["blend"])
    elif case == "blend_fanout":
        model_ir.tensors["blend_side"] = _tensor("blend_side", [1, 3, 2, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["blend"]], ["blend_side"]))
    elif case == "blend_mul_op":
        blend_mul.op_type = "ADD"
    elif case in {"gamma_shape", "beta_shape"}:
        key = "gamma" if case.startswith("gamma") else "beta"
        model_ir.tensors[names[key]].shape = [3]
        model_ir.tensors[names[key]].shape_signature = [3]
        model_ir.tensors[names[key]].data = np.ones(3, dtype=np.float32)
    elif case in {"gamma_dtype", "beta_dtype"}:
        key = "gamma" if case.startswith("gamma") else "beta"
        model_ir.tensors[names[key]].dtype = "FLOAT16"
        model_ir.tensors[names[key]].data = np.asarray(
            model_ir.tensors[names[key]].data, dtype=np.float16
        )
    elif case in {"gamma_quantized", "beta_quantized"}:
        key = "gamma" if case.startswith("gamma") else "beta"
        model_ir.tensors[names[key]].quantization = {"scale": [0.25]}
    elif case in {"gamma_public", "beta_public"}:
        key = "gamma" if case.startswith("gamma") else "beta"
        model_ir.inputs.append(names[key])
    elif case in {"gamma_produced", "beta_produced"}:
        key = "gamma" if case.startswith("gamma") else "beta"
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names[key]])
        )
    elif case in {"gamma_nonfinite", "beta_nonfinite"}:
        key = "gamma" if case.startswith("gamma") else "beta"
        model_ir.tensors[names[key]].data.reshape(-1)[0] = np.nan
    elif case == "blend_mul_shape":
        model_ir.tensors[names["blended_scaled"]].shape[2] = 3
    elif case == "blend_mul_public":
        model_ir.outputs.append(names["blended_scaled"])
    elif case == "blend_mul_fanout":
        model_ir.tensors["mul_side"] = _tensor("mul_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["blended_scaled"]], ["mul_side"])
        )
    elif case == "inst_shape":
        model_ir.tensors[names["inst_output"]].shape[2] = 3
    elif case == "inst_dtype":
        model_ir.tensors[names["inst_output"]].dtype = "FLOAT16"
    elif case == "inst_quantized":
        model_ir.tensors[names["inst_output"]].quantization = {"scale": [0.25]}
    elif case == "inst_public":
        model_ir.outputs.append(names["inst_output"])
    elif case == "inst_fanout":
        model_ir.tensors["inst_side"] = _tensor("inst_side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inst_output"]], ["inst_side"])
        )
    elif case == "reshape_op":
        _operator(model_ir, names["gamma"]).op_type = "EXPAND_DIMS"
    elif case == "reshape_shape_value":
        model_ir.tensors[names["gamma_shape"]].data[:] = [1, 1, 3, 1]
    elif case == "reshape_shape_dtype":
        model_ir.tensors[names["gamma_shape"]].dtype = "FLOAT32"
        model_ir.tensors[names["gamma_shape"]].data = np.asarray(
            [1, 3, 1, 1], dtype=np.float32
        )
    elif case == "reshape_shape_quantized":
        model_ir.tensors[names["gamma_shape"]].quantization = {"scale": [1.0]}
    elif case == "reshape_shape_public":
        model_ir.outputs.append(names["gamma_shape"])
    elif case == "reshape_shape_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["pre_perm"]], [names["gamma_shape"]])
        )
    elif case == "reshape_output_shape":
        model_ir.tensors[names["gamma"]].shape[1] = 2
    elif case == "reshape_output_dtype":
        model_ir.tensors[names["gamma"]].dtype = "FLOAT16"
    elif case == "reshape_output_quantized":
        model_ir.tensors[names["gamma"]].quantization = {"scale": [0.25]}
    elif case == "reshape_output_public":
        model_ir.outputs.append(names["gamma"])
    elif case == "reshape_output_fanout":
        model_ir.tensors["gamma_side"] = _tensor("gamma_side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["gamma"]], ["gamma_side"])
        )
    elif case == "reshape_source_shape":
        model_ir.tensors[names["gamma_vector"]].shape = [1, 1, 3]
        model_ir.tensors[names["gamma_vector"]].shape_signature = [1, 1, 3]
        model_ir.tensors[names["gamma_vector"]].data = np.ones((1, 1, 3), dtype=np.float32)
    elif case == "reshape_source_dtype":
        model_ir.tensors[names["gamma_vector"]].dtype = "FLOAT16"
        model_ir.tensors[names["gamma_vector"]].data = np.asarray(
            model_ir.tensors[names["gamma_vector"]].data, dtype=np.float16
        )
    elif case == "reshape_source_quantized":
        model_ir.tensors[names["gamma_vector"]].quantization = {"scale": [0.25]}
    elif case == "reshape_source_unbound":
        model_ir.tensors[names["gamma_vector"]].data = None
    elif case == "reshape_source_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["gamma_vector"]])
        )
    elif case == "post_op":
        post.op_type = "RESHAPE"
    elif case == "post_perm":
        model_ir.tensors[names["post_perm"]].data[:] = [0, 3, 1, 2]
    elif case == "post_perm_quantized":
        model_ir.tensors[names["post_perm"]].quantization = {"scale": [1.0]}
    elif case == "post_input":
        post.inputs[0] = names["blend"]
    elif case == "post_shape":
        model_ir.tensors[names["post_output"]].shape[3] = 2
    elif case == "post_dtype":
        model_ir.tensors[names["post_output"]].dtype = "FLOAT16"
    elif case == "post_quantized":
        model_ir.tensors[names["post_output"]].quantization = {"scale": [0.25]}
    elif case == "post_public":
        model_ir.outputs.append(names["post_output"])
    elif case == "post_duplicate":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["post_output"]])
        )
    elif case == "post_backward_consumer":
        model_ir.operators.remove(downstream)
        model_ir.operators.insert(model_ir.operators.index(post), downstream)
    elif case == "backward_blend":
        model_ir.operators.remove(blend)
        model_ir.operators.insert(model_ir.operators.index(global_scale), blend)
    elif case.startswith("residual_"):
        residual_pre = _operator(model_ir, names["residual_nchw"])
        if case == "residual_pre_op":
            residual_pre.op_type = "IDENTITY"
        elif case == "residual_perm":
            residual_pre.inputs[1] = names["post_perm"]
        elif case == "residual_shape":
            model_ir.tensors[names["residual_nchw"]].shape[2] = 3
        elif case == "residual_dtype":
            model_ir.tensors[names["residual_source"]].dtype = "FLOAT16"
        elif case == "residual_quantized":
            model_ir.tensors[names["residual_nchw"]].quantization = {"scale": [0.25]}
        elif case == "residual_public":
            model_ir.outputs.append(names["residual_nchw"])
        elif case == "residual_fanout":
            model_ir.tensors["res_side"] = _tensor("res_side", [1, 3, 2, 4])
            model_ir.operators.append(
                OperatorIR("IDENTITY", [names["residual_nchw"]], ["res_side"])
            )
        elif case == "residual_source_public_output":
            model_ir.outputs.append(names["residual_source"])
        elif case == "residual_source_unbound":
            model_ir.inputs.remove(names["residual_source"])
    elif case.startswith("tail_"):
        tail_add = _operator(model_ir, names["tail_output"])
        if case == "tail_add_op":
            tail_add.op_type = "SUB"
        elif case == "tail_shape":
            model_ir.tensors[names["tail_output"]].shape[2] = 3
        elif case == "tail_dtype":
            model_ir.tensors[names["tail_output"]].dtype = "FLOAT16"
        elif case == "tail_quantized":
            model_ir.tensors[names["tail_output"]].quantization = {"scale": [0.25]}
        elif case == "tail_public":
            model_ir.outputs.append(names["tail_output"])
        elif case == "tail_fanout":
            model_ir.tensors["tail_side"] = _tensor("tail_side", [1, 3, 2, 4])
            model_ir.operators.append(
                OperatorIR("IDENTITY", [names["tail_output"]], ["tail_side"])
            )
    model_ir.tensors["unused"] = _tensor("unused", [1])
    before = repr(model_ir)

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_dual_stats_clone_collision_is_transactional(monkeypatch) -> None:
    model_ir, names = _build_dual_stats_model()
    model_ir.tensors["scale_side"] = _tensor("scale_side", [1, 3, 1, 1])
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["s_scale"]], ["scale_side"])
    )
    model_ir.outputs.append("scale_side")
    original_resolve = dual_module._resolve_candidate
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

    monkeypatch.setattr(dual_module, "_resolve_candidate", _resolve_with_collision)
    before_ops = repr(model_ir.operators)
    before_axes = np.asarray(model_ir.tensors[names["s_axes1"]].data).copy()

    stats = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir.operators) == before_ops
    np.testing.assert_array_equal(model_ir.tensors[names["s_axes1"]].data, before_axes)


def test_dual_stats_preflight_avoids_index_construction(monkeypatch) -> None:
    model_ir = ModelIR("insufficient_dual_stats_topology")

    def _unexpected_index(*args, **kwargs):
        raise AssertionError("index must not be constructed")

    monkeypatch.setattr(dual_module, "ModelIRGraphIndex", _unexpected_index)

    assert optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    ) == {_STATS: 0}


def test_dual_stats_counter_is_complete_mutation_evidence(monkeypatch) -> None:
    model_ir, _ = _build_dual_stats_model()
    prune_calls: list[tuple[ModelIR, LayoutState | None]] = []

    def record_prune(
        active_model_ir: ModelIR,
        *,
        layout_state: LayoutState | None = None,
    ) -> None:
        prune_calls.append((active_model_ir, layout_state))

    monkeypatch.setattr(dual_module, "_prune_unused_tensors", record_prune)

    first = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )
    second = optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains(
        model_ir
    )

    assert first == {_STATS: 1}
    assert second == {_STATS: 0}
    assert prune_calls == [(model_ir, None)]
