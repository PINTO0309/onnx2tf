from __future__ import annotations

import copy

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze
from onnx2tf.tflite_builder.passes.stridedslice_concat_layout import (
    optimize_transpose_stridedslice_pre_concat_nhwc_chains,
)


def _name(prefix: str, value: str) -> str:
    return f"{prefix}{value}"


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    signature: list[int] | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
    )


def _constant(
    name: str,
    values: list[int],
    *,
    dtype: str,
) -> TensorIR:
    numpy_dtype = np.int64 if dtype == "INT64" else np.int32
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=[4],
        shape_signature=[4],
        data=np.asarray(values, dtype=numpy_dtype),
    )


def _model(
    *,
    prefix: str = "",
    parameter_dtype: str = "INT32",
    dynamic: bool = False,
    negative_axis: bool = False,
    multiple_posts: bool = False,
    legacy: bool = False,
    public_concat: bool = False,
    public_post: bool = False,
    shared_post_permutation: bool = False,
) -> ModelIR:
    def n(value: str) -> str:
        return _name(prefix, value)

    source_signature = [-1, 4, 5, 6] if dynamic else [1, 4, 5, 6]
    nchw_signature = [-1, 6, 4, 5] if dynamic else [1, 6, 4, 5]
    slice_signature = [-1, 3, 4, 5] if dynamic else [1, 3, 4, 5]

    model = ModelIR(n("stridedslice_concat"))
    model.inputs = [n("x")]
    model.outputs = [n("y")]
    model.tensors = {
        n("x"): _tensor(
            n("x"),
            [1, 4, 5, 6],
            signature=source_signature,
        ),
        n("pre_perm"): _constant(
            n("pre_perm"),
            [0, 3, 1, 2],
            dtype=parameter_dtype,
        ),
        n("post_perm"): _constant(
            n("post_perm"),
            [0, 2, 3, 1],
            dtype=parameter_dtype,
        ),
        n("x_nchw"): _tensor(
            n("x_nchw"),
            [1, 6, 4, 5],
            signature=nchw_signature,
        ),
        n("s0_begin"): _constant(
            n("s0_begin"),
            [0, 0, 0, 0],
            dtype=parameter_dtype,
        ),
        n("s0_end"): _constant(
            n("s0_end"),
            [1, 3, 4, 5],
            dtype=parameter_dtype,
        ),
        n("s0_stride"): _constant(
            n("s0_stride"),
            [1, 1, 1, 1],
            dtype=parameter_dtype,
        ),
        n("s1_begin"): _constant(
            n("s1_begin"),
            [0, 3, 0, 0],
            dtype=parameter_dtype,
        ),
        n("s1_end"): _constant(
            n("s1_end"),
            [1, 6, 4, 5],
            dtype=parameter_dtype,
        ),
        n("s1_stride"): _constant(
            n("s1_stride"),
            [1, 1, 1, 1],
            dtype=parameter_dtype,
        ),
        n("s0"): _tensor(
            n("s0"),
            [1, 3, 4, 5],
            signature=slice_signature,
        ),
        n("s1"): _tensor(
            n("s1"),
            [1, 3, 4, 5],
            signature=slice_signature,
        ),
        n("cat_nchw"): _tensor(
            n("cat_nchw"),
            [1, 6, 4, 5],
            signature=nchw_signature,
        ),
        n("cat_nhwc"): _tensor(
            n("cat_nhwc"),
            [1, 4, 5, 6],
            signature=source_signature,
        ),
        n("y"): _tensor(
            n("y"),
            [1, 4, 5, 6],
            signature=source_signature,
        ),
    }
    slice_options = {
        "beginMask": 0,
        "endMask": 0,
        "ellipsisMask": 0,
        "newAxisMask": 0,
        "shrinkAxisMask": 0,
        "offset": False,
    }
    pre = OperatorIR(
        "TRANSPOSE",
        [n("x"), n("pre_perm")],
        [n("x_nchw")],
    )
    slice0 = OperatorIR(
        "STRIDED_SLICE",
        [n("x_nchw"), n("s0_begin"), n("s0_end"), n("s0_stride")],
        [n("s0")],
        options=dict(slice_options),
    )
    slice1 = OperatorIR(
        "STRIDED_SLICE",
        [n("x_nchw"), n("s1_begin"), n("s1_end"), n("s1_stride")],
        [n("s1")],
        options=dict(slice_options),
    )
    concat = OperatorIR(
        "CONCATENATION",
        [n("s1"), n("s0")],
        [n("cat_nchw")],
        options={"axis": -3 if negative_axis else 1},
    )
    post = OperatorIR(
        "TRANSPOSE",
        [n("cat_nchw"), n("post_perm")],
        [n("cat_nhwc")],
    )
    operators = [pre, slice0, slice1, concat, post]

    if multiple_posts:
        model.tensors[n("cat_alias")] = _tensor(
            n("cat_alias"),
            [1, 4, 5, 6],
            signature=source_signature,
        )
        operators.append(
            OperatorIR(
                "TRANSPOSE",
                [n("cat_nchw"), n("post_perm")],
                [n("cat_alias")],
            )
        )
        operators.append(
            OperatorIR(
                "ADD",
                [n("cat_alias"), n("cat_alias")],
                [n("y")],
            )
        )
    else:
        operators.append(OperatorIR("RELU", [n("cat_nhwc")], [n("y")]))

    if legacy:
        model.tensors[n("legacy_y")] = _tensor(
            n("legacy_y"),
            [1, 6, 4, 5],
            signature=nchw_signature,
        )
        operators.append(
            OperatorIR("RELU", [n("cat_nchw")], [n("legacy_y")])
        )
        model.outputs.append(n("legacy_y"))
    if public_concat:
        model.outputs.append(n("cat_nchw"))
    if public_post:
        model.tensors[n("public_post")] = _tensor(
            n("public_post"),
            [1, 4, 5, 6],
            signature=source_signature,
        )
        operators.append(
            OperatorIR(
                "TRANSPOSE",
                [n("cat_nchw"), n("post_perm")],
                [n("public_post")],
            )
        )
        model.outputs.append(n("public_post"))
    if shared_post_permutation:
        model.inputs.append(n("other"))
        model.outputs.append(n("other_out"))
        model.tensors[n("other")] = _tensor(n("other"), [1, 6, 4, 5])
        model.tensors[n("other_out")] = _tensor(n("other_out"), [1, 4, 5, 6])
        operators.append(
            OperatorIR(
                "TRANSPOSE",
                [n("other"), n("post_perm")],
                [n("other_out")],
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
    )


def _stats_key() -> str:
    return "optimized_transpose_stridedslice_pre_concat_nhwc_chains"


def _assert_rewritten(model: ModelIR, *, prefix: str = "") -> None:
    def n(value: str) -> str:
        return _name(prefix, value)

    slices = [
        operator
        for operator in model.operators
        if operator.op_type == "STRIDED_SLICE"
    ]
    assert len(slices) == 2
    assert {operator.inputs[0] for operator in slices} == {n("x")}
    np.testing.assert_array_equal(
        model.tensors[n("s1_begin")].data,
        np.asarray([0, 0, 0, 3], dtype=model.tensors[n("s1_begin")].data.dtype),
    )
    np.testing.assert_array_equal(
        model.tensors[n("s0_end")].data,
        np.asarray([1, 4, 5, 3], dtype=model.tensors[n("s0_end")].data.dtype),
    )
    assert model.tensors[n("s0")].shape == [1, 4, 5, 3]
    assert model.tensors[n("s0")].physical_layout == "NHWC"
    concat = next(
        operator
        for operator in model.operators
        if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == [n("s1"), n("s0")]
    assert concat.outputs == [n("cat_nhwc")]
    assert concat.options["axis"] == 3
    assert model.tensors[n("cat_nhwc")].physical_layout == "NHWC"


@pytest.mark.parametrize(
    ("parameter_dtype", "dynamic", "negative_axis"),
    [
        ("INT32", False, False),
        ("INT64", False, False),
        ("INT32", True, False),
        ("INT32", False, True),
    ],
)
def test_basic_typed_dynamic_and_negative_axis_rewrite(
    parameter_dtype: str,
    dynamic: bool,
    negative_axis: bool,
) -> None:
    model = _model(
        parameter_dtype=parameter_dtype,
        dynamic=dynamic,
        negative_axis=negative_axis,
    )
    index = ModelIRGraphIndex(model)
    layout = LayoutState.from_model_ir(model)

    stats = optimize_transpose_stridedslice_pre_concat_nhwc_chains(
        model,
        graph_index=index,
        layout_state=layout,
    )

    assert stats == {_stats_key(): 1}
    _assert_rewritten(model)
    assert all(
        not (
            operator.op_type == "TRANSPOSE"
            and operator.outputs == ["cat_nhwc"]
        )
        for operator in model.operators
    )
    assert layout.validate_against_model_ir(model) == []
    fresh = ModelIRGraphIndex(model)
    assert index.producers == fresh.producers
    assert index.consumers == fresh.consumers
    assert index.duplicate_producers == fresh.duplicate_producers


def test_multiple_private_posts_rewrite_every_repeated_alias_slot() -> None:
    model = _model(multiple_posts=True)

    stats = optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    assert stats == {_stats_key(): 1}
    _assert_rewritten(model)
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    assert add.inputs == ["cat_nhwc", "cat_nhwc"]
    assert "cat_alias" not in model.tensors


@pytest.mark.parametrize("multiple_posts", [False, True])
def test_lineage_event_order_matches_the_legacy_active_path(
    multiple_posts: bool,
) -> None:
    model = _model(multiple_posts=multiple_posts)

    optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    events = model.metadata["tensor_lineage_events"]
    assert [event["kind"] for event in events[:3]] == [
        "replace_input",
        "replace_input",
        "rename_tensor",
    ]
    assert [event.get("source") for event in events[:3]] == [
        "replace_operator_input_at",
        "replace_operator_input_at",
        "set_operator_outputs",
    ]
    alias_events = [
        event
        for event in events
        if event.get("src_name") == "cat_alias"
    ]
    assert len(alias_events) == (1 if multiple_posts else 0)
    if alias_events:
        assert "source" not in alias_events[0]


@pytest.mark.parametrize(
    ("legacy", "public_concat", "public_post"),
    [
        (True, False, False),
        (False, True, False),
        (False, False, True),
    ],
)
def test_local_inverse_adapter_preserves_every_nchw_boundary(
    legacy: bool,
    public_concat: bool,
    public_post: bool,
) -> None:
    model = _model(
        legacy=legacy,
        public_concat=public_concat,
        public_post=public_post,
    )

    stats = optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    assert stats == {_stats_key(): 1}
    _assert_rewritten(model)
    adapter = next(
        operator
        for operator in model.operators
        if operator.op_type == "TRANSPOSE"
        and operator.outputs == ["cat_nchw"]
    )
    assert adapter.inputs == ["cat_nhwc", "pre_perm"]
    np.testing.assert_array_equal(
        model.tensors["pre_perm"].data,
        np.asarray([0, 3, 1, 2], dtype=model.tensors["pre_perm"].data.dtype),
    )
    assert model.tensors["cat_nchw"].physical_layout == "NCHW"


def test_shared_post_permutation_is_never_mutated() -> None:
    model = _model(legacy=True, shared_post_permutation=True)
    before = np.asarray(model.tensors["post_perm"].data).copy()

    stats = optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    assert stats == {_stats_key(): 1}
    np.testing.assert_array_equal(model.tensors["post_perm"].data, before)
    other = next(
        operator
        for operator in model.operators
        if operator.outputs == ["other_out"]
    )
    assert other.inputs == ["other", "post_perm"]


def test_numerical_equivalence() -> None:
    model = _model()
    rng = np.random.default_rng(1337)
    source = rng.normal(size=(1, 4, 5, 6)).astype(np.float32)
    old_nchw = np.transpose(source, (0, 3, 1, 2))
    old_s0 = old_nchw[:, 0:3, 0:4, 0:5]
    old_s1 = old_nchw[:, 3:6, 0:4, 0:5]
    expected = np.transpose(np.concatenate([old_s1, old_s0], axis=1), (0, 2, 3, 1))

    optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    actual_s0 = source[:, 0:4, 0:5, 0:3]
    actual_s1 = source[:, 0:4, 0:5, 3:6]
    actual = np.concatenate([actual_s1, actual_s0], axis=3)
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_candidate_limit_and_idempotence() -> None:
    first = _model(prefix="a_")
    second = _model(prefix="b_")
    model = ModelIR("two_stridedslice_concat_groups")
    model.inputs = [*first.inputs, *second.inputs]
    model.outputs = [*first.outputs, *second.outputs]
    model.tensors = {**first.tensors, **second.tensors}
    model.operators = [*first.operators, *second.operators]

    assert optimize_transpose_stridedslice_pre_concat_nhwc_chains(
        model,
        max_rewrites=1,
    ) == {_stats_key(): 1}
    assert sum(operator.op_type == "TRANSPOSE" for operator in model.operators) == 2
    assert optimize_transpose_stridedslice_pre_concat_nhwc_chains(
        model,
        max_rewrites=1,
    ) == {_stats_key(): 1}
    assert optimize_transpose_stridedslice_pre_concat_nhwc_chains(model) == {
        _stats_key(): 0
    }


def test_candidate_only_rewrites_the_selected_seed() -> None:
    model = _model()
    wrong = next(
        operator
        for operator in model.operators
        if operator.op_type == "TRANSPOSE" and operator.outputs == ["cat_nhwc"]
    )
    before = _snapshot(copy.deepcopy(model))

    assert optimize_transpose_stridedslice_pre_concat_nhwc_chains(
        model,
        candidate=wrong,
    ) == {_stats_key(): 0}
    assert _snapshot(model) == before


def test_zero_match_still_preserves_legacy_pruning_contract() -> None:
    model = _model()
    model.tensors["unused"] = _tensor("unused", [1])
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    concat.options["axis"] = 0

    stats = optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    assert stats == {_stats_key(): 0}
    assert "unused" not in model.tensors


def _mutate(model: ModelIR, case: str) -> None:
    slice0 = model.operators[1]
    concat = model.operators[3]
    post = model.operators[4]
    if case == "wrong_pre_perm":
        model.tensors["pre_perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)
    elif case == "public_pre_output":
        model.outputs.append("x_nchw")
    elif case == "pre_fanout":
        model.tensors["pre_side"] = _tensor("pre_side", [1, 6, 4, 5])
        model.outputs.append("pre_side")
        model.operators.append(OperatorIR("IDENTITY", ["x_nchw"], ["pre_side"]))
    elif case == "unsupported_slice":
        slice0.op_type = "SLICE"
    elif case == "masked_slice":
        slice0.options["beginMask"] = 1
    elif case == "offset_slice":
        slice0.options["offset"] = True
    elif case == "zero_stride":
        model.tensors["s0_stride"].data[1] = 0
    elif case == "wrong_constant_dtype":
        model.tensors["s0_begin"].dtype = "FLOAT32"
    elif case == "wrong_constant_numpy_dtype":
        model.tensors["s0_begin"].data = model.tensors["s0_begin"].data.astype(np.float32)
    elif case == "variable_constant":
        model.tensors["s0_begin"].is_variable = True
    elif case == "public_constant":
        model.outputs.append("s0_begin")
    elif case == "shared_constant":
        model.tensors["constant_side"] = _tensor("constant_side", [4], dtype="INT32")
        model.outputs.append("constant_side")
        model.operators.append(
            OperatorIR("IDENTITY", ["s0_begin"], ["constant_side"])
        )
    elif case == "conflicting_constant_roles":
        slice0.inputs[2] = "s0_begin"
    elif case == "public_slice_output":
        model.outputs.append("s0")
    elif case == "slice_fanout":
        model.tensors["slice_side"] = _tensor("slice_side", [1, 3, 4, 5])
        model.outputs.append("slice_side")
        model.operators.append(OperatorIR("IDENTITY", ["s0"], ["slice_side"]))
    elif case == "wrong_slice_shape":
        model.tensors["s0"].shape[2] = 7
    elif case == "wrong_slice_signature":
        model.tensors["s0"].shape_signature[2] = 7
    elif case == "wrong_slice_dtype":
        model.tensors["s0"].dtype = "FLOAT16"
    elif case == "per_axis_slice_quantization":
        model.tensors["s0"].quantization = QuantParamIR(
            scale=[0.1, 0.2, 0.3], zero_point=[0, 0, 0], quantized_dimension=1
        )
    elif case == "wrong_slice_layout":
        model.tensors["s0"].physical_layout = "NHWC"
    elif case == "split_concat_consumers":
        model.tensors["other_cat"] = _tensor("other_cat", [1, 3, 4, 5])
        model.outputs.append("other_cat")
        model.operators.append(OperatorIR("IDENTITY", ["s1"], ["other_cat"]))
    elif case == "duplicate_concat_input":
        concat.inputs = ["s0", "s0"]
    elif case == "wrong_concat_axis":
        concat.options["axis"] = 2
    elif case == "wrong_concat_shape":
        model.tensors["cat_nchw"].shape[1] = 7
    elif case == "wrong_concat_signature":
        model.tensors["cat_nchw"].shape_signature[1] = 7
    elif case == "wrong_concat_dtype":
        model.tensors["cat_nchw"].dtype = "FLOAT16"
    elif case == "per_axis_concat_quantization":
        model.tensors["cat_nchw"].quantization = QuantParamIR(
            scale=[0.1] * 6, zero_point=[0] * 6, quantized_dimension=1
        )
    elif case == "wrong_post_perm":
        model.tensors["post_perm"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)
    elif case == "wrong_post_shape":
        model.tensors["cat_nhwc"].shape[3] = 7
    elif case == "wrong_post_signature":
        model.tensors["cat_nhwc"].shape_signature[3] = 7
    elif case == "wrong_post_dtype":
        model.tensors["cat_nhwc"].dtype = "FLOAT16"
    elif case == "per_axis_post_quantization":
        model.tensors["cat_nhwc"].quantization = QuantParamIR(
            scale=[0.1] * 6, zero_point=[0] * 6, quantized_dimension=3
        )
    elif case == "post_before_concat":
        model.operators.remove(post)
        model.operators.insert(3, post)
    elif case == "duplicate_producer":
        model.operators.append(OperatorIR("IDENTITY", ["s1"], ["s0"]))
    elif case == "wrong_source_shape":
        model.tensors["x"].shape[1] = 8
    elif case == "wrong_source_signature":
        model.tensors["x"].shape_signature[1] = 8
    elif case == "wrong_source_dtype":
        model.tensors["x"].dtype = "FLOAT16"
    elif case == "source_quantization_mismatch":
        model.tensors["x"].quantization = {"scale": [0.1], "zero_point": [0]}
    elif case == "wrong_source_layout":
        model.tensors["x"].physical_layout = "NCHW"
    elif case == "wrong_pre_output_layout":
        model.tensors["x_nchw"].physical_layout = "NHWC"
    elif case == "no_private_post":
        model.outputs.append("cat_nhwc")
    elif case == "legacy_before_adapter":
        model.tensors["legacy"] = _tensor("legacy", [1, 6, 4, 5])
        model.outputs.append("legacy")
        legacy = OperatorIR("RELU", ["cat_nchw"], ["legacy"])
        model.operators.insert(4, legacy)
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "wrong_pre_perm",
        "public_pre_output",
        "pre_fanout",
        "unsupported_slice",
        "masked_slice",
        "offset_slice",
        "zero_stride",
        "wrong_constant_dtype",
        "wrong_constant_numpy_dtype",
        "variable_constant",
        "public_constant",
        "shared_constant",
        "conflicting_constant_roles",
        "public_slice_output",
        "slice_fanout",
        "wrong_slice_shape",
        "wrong_slice_signature",
        "wrong_slice_dtype",
        "per_axis_slice_quantization",
        "wrong_slice_layout",
        "split_concat_consumers",
        "duplicate_concat_input",
        "wrong_concat_axis",
        "wrong_concat_shape",
        "wrong_concat_signature",
        "wrong_concat_dtype",
        "per_axis_concat_quantization",
        "wrong_post_perm",
        "wrong_post_shape",
        "wrong_post_signature",
        "wrong_post_dtype",
        "per_axis_post_quantization",
        "post_before_concat",
        "duplicate_producer",
        "wrong_source_shape",
        "wrong_source_signature",
        "wrong_source_dtype",
        "source_quantization_mismatch",
        "wrong_source_layout",
        "wrong_pre_output_layout",
        "no_private_post",
        "legacy_before_adapter",
    ],
)
def test_unsafe_candidate_is_transactional_noop(case: str) -> None:
    model = _model()
    _mutate(model, case)
    optimize_transpose_stridedslice_pre_concat_nhwc_chains(
        model,
        max_rewrites=0,
    )
    before = _snapshot(copy.deepcopy(model))

    stats = optimize_transpose_stridedslice_pre_concat_nhwc_chains(model)

    assert stats == {_stats_key(): 0}
    assert _snapshot(model) == before


def test_external_index_is_not_refreshed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    index = ModelIRGraphIndex(model)
    original_refresh = ModelIRGraphIndex.refresh
    calls = 0

    def counted_refresh(active: ModelIRGraphIndex) -> None:
        nonlocal calls
        calls += 1
        original_refresh(active)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    assert optimize_transpose_stridedslice_pre_concat_nhwc_chains(
        model,
        graph_index=index,
    ) == {_stats_key(): 1}
    assert calls == 0
