from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.instance_norm_prepost_layout as direct_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_instancenorm_prepost_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.instance_norm_prepost_layout import (
    _optimize_transpose_squeeze_reshape_instancenorm_direct_post_nhwc_chains,
    _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains,
    _optimize_transpose_squeeze_reshape_instancenorm_side_squeeze_nhwc_chains,
    _optimize_transpose_squeeze_reshape_instancenorm_unary_reshape_nhwc_chains,
)
from tests.test_flatbuffer_direct_indexed_instance_norm_direct_layout import (
    _assert_index_current,
    _build_model,
    _evaluate,
    _operator,
    _replace_tail,
    _tensor,
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


def _replace_residual_source_mode(
    model_ir: ModelIR,
    names: dict[str, str],
    mode: str,
) -> None:
    if mode == "rank3_transpose":
        return
    prefix = names["source"][: -len("source")]

    def _name(value: str) -> str:
        return f"{prefix}{value}"

    dtype = str(model_ir.tensors[names["inst_output"]].dtype)
    source_name = _name("residual_source")
    pre_name = _name("residual_chw")
    squeeze_name = _name("residual_squeezed")
    unary_name = _name("residual_unary")
    bridge = _operator(model_ir, pre_name)
    add = _operator(model_ir, _name("tail_add"))
    model_ir.tensors[source_name].shape = [1, 2, 4, 3]
    model_ir.tensors[source_name].shape_signature = [1, 2, 4, 3]
    model_ir.tensors[_name("residual_perm")].shape = [4]
    model_ir.tensors[_name("residual_perm")].shape_signature = [4]
    model_ir.tensors[_name("residual_perm")].data = np.asarray(
        [0, 3, 1, 2], dtype=np.int32
    )
    model_ir.tensors[pre_name].shape = [1, 3, 2, 4]
    model_ir.tensors[pre_name].shape_signature = [1, 3, 2, 4]
    model_ir.tensors[squeeze_name] = _tensor(
        squeeze_name,
        [3, 2, 4],
        dtype=dtype,
    )
    squeeze = OperatorIR(
        "SQUEEZE",
        [pre_name],
        [squeeze_name],
        {"squeezeDims": [0]},
    )
    insert_index = model_ir.operators.index(bridge) + 1
    model_ir.operators.insert(insert_index, squeeze)
    replacement_name = squeeze_name
    if mode == "pre_squeeze_unary":
        model_ir.tensors[unary_name] = _tensor(
            unary_name,
            [3, 2, 4],
            dtype=dtype,
        )
        model_ir.operators.insert(
            insert_index + 1,
            OperatorIR("RELU", [squeeze_name], [unary_name]),
        )
        replacement_name = unary_name
    elif mode != "pre_squeeze":
        raise AssertionError(f"unsupported residual source mode: {mode}")
    add.inputs = [
        replacement_name if str(value) == pre_name else str(value)
        for value in add.inputs
    ]


def _add_tail_fanout(
    model_ir: ModelIR,
    names: dict[str, str],
    *,
    before_reshape: bool = False,
    repeated_input: bool = False,
) -> OperatorIR:
    prefix = names["source"][: -len("source")]
    add_name = f"{prefix}tail_add"
    side_name = f"{prefix}tail_side"
    dtype = str(model_ir.tensors[add_name].dtype)
    model_ir.tensors[side_name] = _tensor(
        side_name,
        [3, 2, 4],
        dtype=dtype,
    )
    side = OperatorIR(
        "ADD" if repeated_input else "RELU",
        [add_name, add_name] if repeated_input else [add_name],
        [side_name],
    )
    reshape = _operator(model_ir, f"{prefix}tail_reshape")
    if before_reshape:
        model_ir.operators.insert(model_ir.operators.index(reshape), side)
    else:
        model_ir.operators.append(side)
    model_ir.outputs.append(side_name)
    return side



@pytest.mark.parametrize(
    "residual_mode",
    ("rank3_transpose", "pre_squeeze", "pre_squeeze_unary"),
)
@pytest.mark.parametrize("dtype", ("FLOAT16", "FLOAT32"))
@pytest.mark.parametrize("produced_source", (False, True))
def test_residual_reshape_instance_norm_tail_is_indexed_and_equivalent(
    residual_mode: str,
    dtype: str,
    produced_source: bool,
) -> None:
    model_ir, names = _build_model(
        dtype=dtype,
        produced_source=produced_source,
    )
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    _replace_residual_source_mode(model_ir, names, residual_mode)
    rng = np.random.default_rng(61)
    np_dtype = np.float16 if dtype == "FLOAT16" else np.float32
    feeds = {
        name: rng.normal(size=model_ir.tensors[name].shape).astype(np_dtype)
        for name in model_ir.inputs
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    before = repr(model_ir)

    for owner, stats_key in (
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
    ):
        assert owner(model_ir) == {stats_key: 0}
        assert repr(model_ir) == before

    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)
    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir,
            graph_index=graph_index,
            layout_state=layout_state,
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 1}
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
    assert _operator(model_ir, names["post_output"]).op_type == "RESHAPE"
    tail_add = _operator(model_ir, "tail_add")
    if residual_mode == "rank3_transpose":
        assert "residual_source" in tail_add.inputs
    else:
        residual_squeeze = _operator(model_ir, "residual_squeezed")
        assert residual_squeeze.inputs == ["residual_source"]
        assert model_ir.tensors["residual_squeezed"].shape == [2, 4, 3]
        if residual_mode == "pre_squeeze_unary":
            assert "residual_unary" in tail_add.inputs


def test_compatibility_wrapper_dispatches_residual_reshape_owner() -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_COMPAT_STATS: 1}
    assert _operator(model_ir, names["post_output"]).op_type == "RESHAPE"
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []


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
def test_residual_reshape_preserves_supported_residual_unary_family(
    unary_op: str,
) -> None:
    dtype = "FLOAT16" if unary_op == "CAST" else "FLOAT32"
    model_ir, names = _build_model(dtype=dtype)
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    _replace_residual_source_mode(model_ir, names, "pre_squeeze_unary")
    unary = _operator(model_ir, "residual_unary")
    unary.op_type = unary_op
    if unary_op == "CAST":
        for name in ("residual_source", "residual_chw", "residual_squeezed"):
            model_ir.tensors[name].dtype = "FLOAT32"

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 1}
    assert _operator(model_ir, "residual_unary").op_type == unary_op
    assert _operator(model_ir, "residual_squeezed").inputs == [
        "residual_source"
    ]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "residual_mode",
    ("rank3_transpose", "pre_squeeze", "pre_squeeze_unary"),
)
@pytest.mark.parametrize("dynamic_axis", ("height", "width", "channel"))
def test_residual_reshape_preserves_dynamic_signatures(
    residual_mode: str,
    dynamic_axis: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    _replace_residual_source_mode(model_ir, names, residual_mode)
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
    for name in ("tail_squeezed", "tail_add"):
        model_ir.tensors[name].shape_signature[chw_axis] = -1
    if residual_mode == "rank3_transpose":
        model_ir.tensors["residual_source"].shape_signature[hwc_axis] = -1
        model_ir.tensors["residual_chw"].shape_signature[chw_axis] = -1
    else:
        model_ir.tensors["residual_source"].shape_signature[source_axis] = -1
        model_ir.tensors["residual_chw"].shape_signature[nchw_axis] = -1
        model_ir.tensors["residual_squeezed"].shape_signature[chw_axis] = -1
        if residual_mode == "pre_squeeze_unary":
            model_ir.tensors["residual_unary"].shape_signature[chw_axis] = -1
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
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 1}
    assert model_ir.tensors["tail_squeezed"].shape_signature[hwc_axis] == -1
    assert model_ir.tensors["tail_add"].shape_signature[hwc_axis] == -1
    assert model_ir.tensors[names["post_output"]].shape_signature[source_axis] == -1
    if residual_mode != "rank3_transpose":
        assert (
            model_ir.tensors["residual_squeezed"].shape_signature[hwc_axis]
            == -1
        )


@pytest.mark.parametrize("side_before_reshape", (False, True))
@pytest.mark.parametrize("repeated_input", (False, True))
def test_residual_reshape_preserves_tail_add_fanout_with_one_adapter(
    side_before_reshape: bool,
    repeated_input: bool,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    side = _add_tail_fanout(
        model_ir,
        names,
        before_reshape=side_before_reshape,
        repeated_input=repeated_input,
    )
    side_name = str(side.outputs[0])
    rng = np.random.default_rng(73)
    feeds = {
        name: rng.normal(size=model_ir.tensors[name].shape).astype(np.float32)
        for name in model_ir.inputs
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 1}
    actual = _evaluate(model_ir, feeds)
    assert set(actual) == set(expected)
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-6)
    side = _operator(model_ir, side_name)
    assert len(set(side.inputs)) == 1
    adapter_name = str(side.inputs[0])
    adapter = _operator(model_ir, adapter_name)
    assert adapter.op_type == "TRANSPOSE"
    assert adapter.inputs[0] == "tail_add"
    assert np.asarray(model_ir.tensors[adapter.inputs[1]].data).tolist() == [
        2,
        0,
        1,
    ]
    assert model_ir.operators.index(adapter) < model_ir.operators.index(side)
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 1
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_reshape_rewrites_multiple_chains() -> None:
    first, first_names = _build_model(prefix="a_")
    _replace_tail(first, first_names, "squeeze_add_reshape")
    second, second_names = _build_model(prefix="b_")
    _replace_tail(second, second_names, "squeeze_add_reshape")
    _replace_residual_source_mode(second, second_names, "pre_squeeze_unary")
    first.tensors.update(second.tensors)
    first.operators.extend(second.operators)
    first.inputs.extend(second.inputs)
    first.outputs.extend(second.outputs)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            first
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 2}
    assert _operator(first, first_names["post_output"]).op_type == "RESHAPE"
    assert _operator(first, second_names["post_output"]).op_type == "RESHAPE"
    assert validate_model_ir_invariants(first) == []


def test_residual_reshape_accepts_commuted_tail_add() -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    tail_add = _operator(model_ir, "tail_add")
    tail_add.inputs.reverse()

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 1}
    assert tail_add.inputs == ["residual_source", "tail_squeezed"]
    assert validate_model_ir_invariants(model_ir) == []


def test_residual_reshape_reuses_valid_existing_fanout_permutation() -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    side = _add_tail_fanout(model_ir, names)
    perm_name = direct_module._RESIDUAL_ADAPTER_PERM_NAME
    existing = _tensor(
        perm_name,
        [3],
        dtype="INT64",
        data=np.asarray([2, 0, 1], dtype=np.int64),
    )
    model_ir.tensors[perm_name] = existing

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 1}
    assert model_ir.tensors[perm_name] is existing
    adapter = _operator(model_ir, str(side.inputs[0]))
    assert adapter.inputs[1] == perm_name


@pytest.mark.parametrize(
    "case",
    (
        "residual_perm",
        "residual_fanout",
        "residual_dtype",
        "residual_quantization",
        "residual_source_public",
        "residual_duplicate_producer",
        "pre_squeeze_axis",
        "pre_output_fanout",
        "unsupported_residual_unary",
        "tail_add_public",
        "tail_add_quantization",
        "tail_shape",
        "post_public",
        "fanout_backward",
        "fanout_adapter",
    ),
)
def test_residual_reshape_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    if case in {
        "pre_squeeze_axis",
        "pre_output_fanout",
        "unsupported_residual_unary",
    }:
        _replace_residual_source_mode(
            model_ir,
            names,
            "pre_squeeze_unary",
        )
    if case == "residual_perm":
        model_ir.tensors["residual_perm"].data[:] = [0, 2, 1]
    elif case == "residual_fanout":
        model_ir.tensors["residual_side"] = _tensor(
            "residual_side",
            [3, 2, 4],
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["residual_chw"], ["residual_side"])
        )
        model_ir.outputs.append("residual_side")
    elif case == "residual_dtype":
        model_ir.tensors["residual_chw"].dtype = "FLOAT16"
    elif case == "residual_quantization":
        model_ir.tensors["residual_chw"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "residual_source_public":
        model_ir.outputs.append("residual_source")
    elif case == "residual_duplicate_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], ["residual_chw"])
        )
    elif case == "pre_squeeze_axis":
        _operator(model_ir, "residual_squeezed").options["squeezeDims"] = [1]
    elif case == "pre_output_fanout":
        model_ir.tensors["residual_pre_side"] = _tensor(
            "residual_pre_side",
            [1, 3, 2, 4],
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["residual_chw"], ["residual_pre_side"])
        )
        model_ir.outputs.append("residual_pre_side")
    elif case == "unsupported_residual_unary":
        _operator(model_ir, "residual_unary").op_type = "SIN"
    elif case == "tail_add_public":
        model_ir.outputs.append("tail_add")
    elif case == "tail_add_quantization":
        model_ir.tensors["tail_add"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "tail_shape":
        model_ir.tensors["tail_shape"].data[:] = [1, 3, 4, 2]
    elif case == "post_public":
        model_ir.outputs.append(names["post_output"])
    elif case == "fanout_backward":
        side = _add_tail_fanout(model_ir, names)
        model_ir.operators.remove(side)
        tail_add = _operator(model_ir, "tail_add")
        model_ir.operators.insert(model_ir.operators.index(tail_add), side)
    elif case == "fanout_adapter":
        _add_tail_fanout(model_ir, names)
        perm_name = direct_module._RESIDUAL_ADAPTER_PERM_NAME
        model_ir.tensors[perm_name] = _tensor(
            perm_name,
            [3],
            dtype="INT32",
            data=np.asarray([0, 1, 2], dtype=np.int32),
        )
    before = repr(model_ir)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 0}
    assert repr(model_ir) == before


@pytest.mark.parametrize(
    "case",
    ("residual_perm", "tail_quantization", "negative_epsilon", "fanout_adapter"),
)
def test_compatibility_wrapper_does_not_fallback_for_unsafe_residual_reshape(
    case: str,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    if case == "residual_perm":
        model_ir.tensors["residual_perm"].data[:] = [0, 2, 1]
    elif case == "tail_quantization":
        model_ir.tensors["tail_add"].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
    elif case == "negative_epsilon":
        model_ir.tensors[names["epsilon"]].data[0] = -0.01
    elif case == "fanout_adapter":
        _add_tail_fanout(model_ir, names)
        perm_name = direct_module._RESIDUAL_ADAPTER_PERM_NAME
        model_ir.tensors[perm_name] = _tensor(
            perm_name,
            [3],
            dtype="INT32",
            data=np.asarray([0, 1, 2], dtype=np.int32),
        )
    before = repr(model_ir)

    stats = _optimize_transpose_instancenorm_prepost_nhwc_chains(model_ir)

    assert stats == {_COMPAT_STATS: 0}
    assert repr(model_ir) == before


def test_residual_reshape_fanout_adapter_preflight_collision_is_transactional(
    monkeypatch,
) -> None:
    model_ir, names = _build_model()
    _replace_tail(model_ir, names, "squeeze_add_reshape")
    _add_tail_fanout(model_ir, names)
    original_resolve = direct_module._resolve_candidate
    injected = False

    def resolve_with_collision(*args, **kwargs):
        nonlocal injected
        plan = original_resolve(*args, **kwargs)
        if plan is None or plan.fanout_adapter is None or injected:
            return plan
        adapter = plan.fanout_adapter.tensor
        model_ir.tensors[adapter.name] = _tensor(
            adapter.name,
            list(adapter.shape),
            dtype=str(adapter.dtype),
        )
        injected = True
        return plan

    monkeypatch.setattr(direct_module, "_resolve_candidate", resolve_with_collision)
    before_operators = repr(model_ir.operators)

    stats = (
        _optimize_transpose_squeeze_reshape_instancenorm_residual_reshape_nhwc_chains(
            model_ir
        )
    )

    assert stats == {_RESIDUAL_RESHAPE_STATS: 0}
    assert repr(model_ir.operators) == before_operators
