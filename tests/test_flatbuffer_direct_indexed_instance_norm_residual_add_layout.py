from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_norm_residual_add_layout as residual_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.instance_norm_residual_add_layout import (
    optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains,
)
from tests.test_flatbuffer_direct_indexed_instance_norm_post_bias_layout import (
    _assert_index_current,
    _build_model,
    _evaluate,
    _operator,
    _tensor,
)


_STATS = (
    "optimized_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains"
)


def _build_residual_model(
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
    for key in (
        "residual_upstream",
        "residual_source",
        "residual_perm",
        "residual_nchw",
        "add_output",
    ):
        names[key] = f"{prefix}{key}"
    dtype_name = str(model_ir.tensors[names["source"]].dtype)
    source_shape = [1, 2, 4, 3]
    nchw_shape = [1, 3, 2, 4]
    post = _operator(model_ir, names["post_output"])
    model_ir.operators.remove(post)
    del model_ir.tensors[names["post_output"]]
    del model_ir.tensors[names["post_perm"]]
    add_bias = _operator(model_ir, names["inst_output"])
    add_bias.inputs[add_bias.inputs.index(names["post_output"])] = names["scaled"]
    model_ir.tensors[names["inst_output"]].shape = list(nchw_shape)
    model_ir.tensors[names["inst_output"]].shape_signature = list(nchw_shape)
    model_ir.tensors[names["output"]].shape = list(nchw_shape)
    model_ir.tensors[names["output"]].shape_signature = list(nchw_shape)

    model_ir.tensors[names["residual_perm"]] = _tensor(
        names["residual_perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors[names["residual_source"]] = _tensor(
        names["residual_source"],
        source_shape,
        dtype=dtype_name,
    )
    model_ir.tensors[names["residual_nchw"]] = _tensor(
        names["residual_nchw"],
        nchw_shape,
        dtype=dtype_name,
    )
    model_ir.tensors[names["add_output"]] = _tensor(
        names["add_output"],
        nchw_shape,
        dtype=dtype_name,
    )
    main_pre = _operator(model_ir, names["x"])
    insertion_index = model_ir.operators.index(main_pre) + 1
    if produced_residual:
        model_ir.tensors[names["residual_upstream"]] = _tensor(
            names["residual_upstream"],
            source_shape,
            dtype=dtype_name,
        )
        model_ir.inputs.append(names["residual_upstream"])
        model_ir.operators.insert(
            insertion_index,
            OperatorIR(
                "IDENTITY",
                [names["residual_upstream"]],
                [names["residual_source"]],
            ),
        )
        insertion_index += 1
    else:
        model_ir.inputs.append(names["residual_source"])
    model_ir.operators.insert(
        insertion_index,
        OperatorIR(
            "TRANSPOSE",
            [names["residual_source"], names["residual_perm"]],
            [names["residual_nchw"]],
        ),
    )
    tail_add_inputs = [names["inst_output"], names["residual_nchw"]]
    if commuted_affine:
        tail_add_inputs.reverse()
    tail_add = OperatorIR(
        "ADD",
        tail_add_inputs,
        [names["add_output"]],
    )
    bias_index = model_ir.operators.index(add_bias)
    model_ir.operators.insert(bias_index + 1, tail_add)
    relu = _operator(model_ir, names["output"])
    relu.inputs = [names["add_output"]]
    return model_ir, names


@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
@pytest.mark.parametrize("produced_residual", (False, True))
@pytest.mark.parametrize(
    "separate_axes,axes,commuted_sub,commuted_affine,scale_mode,bias_mode",
    (
        (False, (2, 3), False, False, "nchw", "nchw"),
        (False, (-2, -1), True, True, "nchw", "scalar"),
        (True, (3, 2), False, True, "scalar", "nchw"),
        (True, (-1, -2), True, False, "nchw", "nchw"),
    ),
)
def test_residual_add_instance_norm_is_indexed_and_numerically_equivalent(
    dtype: str,
    produced_source: bool,
    produced_residual: bool,
    separate_axes: bool,
    axes: tuple[int, int],
    commuted_sub: bool,
    commuted_affine: bool,
    scale_mode: str,
    bias_mode: str,
) -> None:
    model_ir, names = _build_residual_model(
        dtype=dtype,
        produced_source=produced_source,
        produced_residual=produced_residual,
        separate_axes=separate_axes,
        axes=axes,
        commuted_sub=commuted_sub,
        commuted_affine=commuted_affine,
        scale_mode=scale_mode,
        bias_mode=bias_mode,
    )
    main_feed = names["upstream"] if produced_source else names["source"]
    residual_feed = (
        names["residual_upstream"]
        if produced_residual
        else names["residual_source"]
    )
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    rng = np.random.default_rng(89)
    feeds = {
        main_feed: rng.normal(size=[1, 2, 4, 3]).astype(np_dtype),
        residual_feed: rng.normal(size=[1, 2, 4, 3]).astype(np_dtype),
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_STATS: 1}
    transposes = [op for op in model_ir.operators if op.op_type == "TRANSPOSE"]
    assert len(transposes) == 1
    adapter = transposes[0]
    assert adapter.outputs == [names["add_output"]]
    assert adapter.inputs[0].startswith(f"{names['add_output']}_nhwc")
    assert _operator(model_ir, names["mean1"]).inputs[0] == names["source"]
    assert names["source"] in _operator(model_ir, names["centered"]).inputs
    tail_add = _operator(model_ir, adapter.inputs[0])
    assert names["residual_source"] in tail_add.inputs
    assert names["inst_output"] in tail_add.inputs
    assert list(model_ir.tensors[names["inst_output"]].shape) == [1, 2, 4, 3]
    assert list(model_ir.tensors[adapter.inputs[0]].shape) == [1, 2, 4, 3]
    assert list(model_ir.tensors[names["add_output"]].shape) == [1, 3, 2, 4]
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
def test_residual_add_instance_norm_preserves_dynamic_signatures(
    dynamic_axis: int,
) -> None:
    model_ir, names = _build_residual_model()
    for name in (names["source"], names["residual_source"]):
        model_ir.tensors[name].shape_signature[dynamic_axis] = -1
    nchw_axis = (0, 3, 1, 2).index(dynamic_axis)
    for key in (
        "x",
        "centered",
        "squared",
        "normalized",
        "scaled",
        "inst_output",
        "residual_nchw",
        "add_output",
        "output",
    ):
        model_ir.tensors[names[key]].shape_signature[nchw_axis] = -1
    if dynamic_axis == 3:
        for key in (
            "mean1",
            "mean2",
            "variance_epsilon",
            "std",
            "inverse_std",
        ):
            model_ir.tensors[names[key]].shape_signature[1] = -1

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 1}
    adapter = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    assert model_ir.tensors[adapter.inputs[0]].shape_signature[dynamic_axis] == -1
    assert model_ir.tensors[names["add_output"]].shape_signature[nchw_axis] == -1
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("repeated_slots", (False, True))
def test_residual_add_instance_norm_preserves_downstream_fanout(
    repeated_slots: bool,
) -> None:
    model_ir, names = _build_residual_model()
    side_name = f"{names['add_output']}_side"
    model_ir.tensors[side_name] = _tensor(side_name, [1, 3, 2, 4])
    inputs = [names["add_output"], names["add_output"]] if repeated_slots else [names["add_output"]]
    op_type = "ADD" if repeated_slots else "IDENTITY"
    model_ir.operators.append(OperatorIR(op_type, inputs, [side_name]))
    model_ir.outputs.append(side_name)
    feeds = {
        names["source"]: np.random.default_rng(97)
        .normal(size=[1, 2, 4, 3])
        .astype(np.float32),
        names["residual_source"]: np.random.default_rng(101)
        .normal(size=[1, 2, 4, 3])
        .astype(np.float32),
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 1}
    assert sum(op.op_type == "TRANSPOSE" for op in model_ir.operators) == 1
    actual = _evaluate(model_ir, feeds)
    for output_name in model_ir.outputs:
        np.testing.assert_allclose(
            actual[output_name],
            expected[output_name],
            rtol=1e-6,
            atol=1e-6,
        )
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_add_instance_norm_clones_shared_changed_constants() -> None:
    model_ir, names = _build_residual_model(separate_axes=True)
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

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 1}
    np.testing.assert_array_equal(model_ir.tensors[names["scale"]].data, original_scale)
    mean1 = _operator(model_ir, names["mean1"])
    scale = _operator(model_ir, names["scaled"])
    assert mean1.inputs[1] != names["axes1"]
    assert names["scale"] not in scale.inputs
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_add_instance_norm_reuses_valid_adapter_permutation() -> None:
    model_ir, names = _build_residual_model()
    perm_name = residual_module._ADAPTER_PERM_NAME
    model_ir.tensors[perm_name] = _tensor(
        perm_name,
        [4],
        dtype="INT64",
        data=np.asarray([0, 3, 1, 2], dtype=np.int64),
    )

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 1}
    adapter = next(op for op in model_ir.operators if op.op_type == "TRANSPOSE")
    assert adapter.inputs[1] == perm_name
    assert model_ir.tensors[perm_name].dtype == "INT64"


def test_residual_add_instance_norm_rewrites_multiple_chains_with_total_cap() -> None:
    first, first_names = _build_residual_model(prefix="a_")
    second, second_names = _build_residual_model(prefix="b_")
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    graph_index = ModelIRGraphIndex(first)

    first_stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            first,
            graph_index=graph_index,
            max_rewrites=1,
        )
    )
    second_stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            first,
            graph_index=graph_index,
            max_rewrites=1,
        )
    )

    assert first_stats == {_STATS: 1}
    assert second_stats == {_STATS: 1}
    assert _operator(first, first_names["mean1"]).inputs[0] == first_names["source"]
    assert _operator(first, second_names["mean1"]).inputs[0] == second_names["source"]
    _assert_index_current(first, graph_index)


@pytest.mark.parametrize(
    "case",
    (
        "main_perm",
        "main_perm_quantized",
        "main_perm_produced",
        "main_source_public_output",
        "main_source_unbound",
        "main_pre_public",
        "main_pre_shape",
        "main_pre_duplicate",
        "mean_axes",
        "negative_epsilon",
        "nonunit_numerator",
        "core_quantized",
        "bias_shape",
        "bias_dtype",
        "bias_public",
        "bias_produced",
        "inst_output_public",
        "inst_output_fanout",
        "inst_output_shape",
        "tail_op",
        "tail_output_public",
        "tail_output_missing_tensor",
        "tail_output_quantized",
        "tail_no_consumers",
        "tail_backward_consumer",
        "residual_producer",
        "residual_perm",
        "residual_perm_quantized",
        "residual_pre_public",
        "residual_pre_fanout",
        "residual_source_public_output",
        "residual_source_unbound",
        "residual_source_shape",
        "residual_dtype",
        "adapter_value",
        "adapter_floating",
        "adapter_produced",
        "adapter_public",
        "adapter_quantized",
    ),
)
def test_residual_add_instance_norm_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_residual_model()
    residual_pre = _operator(model_ir, names["residual_nchw"])
    tail_add = _operator(model_ir, names["add_output"])
    relu = _operator(model_ir, names["output"])
    if case == "main_perm":
        model_ir.tensors[names["pre_perm"]].data[:] = [0, 2, 3, 1]
    elif case == "main_perm_quantized":
        model_ir.tensors[names["pre_perm"]].quantization = {"scale": [1.0]}
    elif case == "main_perm_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["axes1"]], [names["pre_perm"]])
        )
    elif case == "main_source_public_output":
        model_ir.outputs.append(names["source"])
    elif case == "main_source_unbound":
        model_ir.inputs.remove(names["source"])
    elif case == "main_pre_public":
        model_ir.outputs.append(names["x"])
    elif case == "main_pre_shape":
        model_ir.tensors[names["x"]].shape[2] = 3
    elif case == "main_pre_duplicate":
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
    elif case == "bias_produced":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["bias"]])
        )
    elif case == "inst_output_public":
        model_ir.outputs.append(names["inst_output"])
    elif case == "inst_output_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["inst_output"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "inst_output_shape":
        model_ir.tensors[names["inst_output"]].shape[2] = 3
    elif case == "tail_op":
        tail_add.op_type = "MUL"
    elif case == "tail_output_public":
        model_ir.outputs.append(names["add_output"])
    elif case == "tail_output_missing_tensor":
        del model_ir.tensors[names["add_output"]]
    elif case == "tail_output_quantized":
        model_ir.tensors[names["add_output"]].quantization = {"scale": [0.25]}
    elif case == "tail_no_consumers":
        model_ir.operators.remove(relu)
    elif case == "tail_backward_consumer":
        model_ir.operators.remove(relu)
        model_ir.operators.insert(model_ir.operators.index(tail_add), relu)
    elif case == "residual_producer":
        residual_pre.op_type = "IDENTITY"
    elif case == "residual_perm":
        model_ir.tensors[names["residual_perm"]].data[:] = [0, 2, 3, 1]
    elif case == "residual_perm_quantized":
        model_ir.tensors[names["residual_perm"]].quantization = {"scale": [1.0]}
    elif case == "residual_pre_public":
        model_ir.outputs.append(names["residual_nchw"])
    elif case == "residual_pre_fanout":
        model_ir.tensors["side"] = _tensor("side", [1, 3, 2, 4])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["residual_nchw"]], ["side"])
        )
        model_ir.outputs.append("side")
    elif case == "residual_source_public_output":
        model_ir.outputs.append(names["residual_source"])
    elif case == "residual_source_unbound":
        model_ir.inputs.remove(names["residual_source"])
    elif case == "residual_source_shape":
        model_ir.tensors[names["residual_source"]].shape[1] = 3
    elif case == "residual_dtype":
        model_ir.tensors[names["residual_nchw"]].dtype = "FLOAT16"
    elif case.startswith("adapter_"):
        perm_name = residual_module._ADAPTER_PERM_NAME
        model_ir.tensors[perm_name] = _tensor(
            perm_name,
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        )
        if case == "adapter_value":
            model_ir.tensors[perm_name].data[:] = [0, 2, 3, 1]
        elif case == "adapter_floating":
            model_ir.tensors[perm_name].dtype = "FLOAT32"
            model_ir.tensors[perm_name].data = np.asarray(
                [0, 3, 1, 2], dtype=np.float32
            )
        elif case == "adapter_produced":
            model_ir.operators.append(
                OperatorIR("IDENTITY", [names["axes1"]], [perm_name])
            )
        elif case == "adapter_public":
            model_ir.outputs.append(perm_name)
        elif case == "adapter_quantized":
            model_ir.tensors[perm_name].quantization = {"scale": [1.0]}
    model_ir.tensors["unused"] = _tensor("unused", [1])
    before = repr(model_ir)

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize("collision", ("constant_clone", "adapter", "output"))
def test_residual_add_allocation_collisions_are_transactional(
    monkeypatch,
    collision: str,
) -> None:
    model_ir, names = _build_residual_model()
    if collision == "constant_clone":
        model_ir.tensors["scale_side"] = _tensor("scale_side", [1, 3, 1, 1])
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["scale"]], ["scale_side"])
        )
        model_ir.outputs.append("scale_side")
    original_resolve = residual_module._resolve_candidate
    injected = False

    def _resolve_with_collision(*args, **kwargs):
        nonlocal injected
        plan = original_resolve(*args, **kwargs)
        if plan is None or injected:
            return plan
        if collision == "constant_clone":
            name = next(
                update.clone_name
                for update in plan.constant_updates
                if update.clone_name is not None
            )
            assert name is not None
        elif collision == "adapter":
            assert plan.adapter_permutation is not None
            name = residual_module._ADAPTER_PERM_NAME
        else:
            name = plan.new_add_output.name
        model_ir.tensors[name] = _tensor(name, [1])
        injected = True
        return plan

    monkeypatch.setattr(residual_module, "_resolve_candidate", _resolve_with_collision)
    before_ops = repr(model_ir.operators)
    before_scale = np.asarray(model_ir.tensors[names["scale"]].data).copy()

    stats = (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir.operators) == before_ops
    np.testing.assert_array_equal(model_ir.tensors[names["scale"]].data, before_scale)


def test_residual_add_preflight_avoids_index_construction(monkeypatch) -> None:
    model_ir = ModelIR("insufficient_residual_add_topology")

    def _unexpected_index(*args, **kwargs):
        raise AssertionError("index must not be constructed")

    monkeypatch.setattr(residual_module, "ModelIRGraphIndex", _unexpected_index)

    assert (
        optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains(
            model_ir
        )
        == {_STATS: 0}
    )
