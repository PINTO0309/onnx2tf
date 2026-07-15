from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import pytest

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, QuantParamIR, TensorIR
from onnx2tf.tflite_builder.passes.elementwise_concat_layout import (
    optimize_transpose_elementwise_concat_conv_nhwc_groups,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
    )


def _make_model(
    *,
    unary_type: str | None = "RELU",
    binary_type: str | None = None,
    constant_first: bool = False,
    permutation_dtype: np.dtype = np.dtype(np.int32),
    dynamic: bool = False,
    negative_axis: bool = False,
    connected_concat: bool = False,
    legacy: bool = False,
    pre_fanout: bool = False,
    multiple_posts: bool = False,
    direct_input: bool = False,
    existing_post_boundary: bool = False,
) -> ModelIR:
    source_shape = [1, 2, 3, 4]
    source_signature = [1, -1, -1, 4] if dynamic else list(source_shape)
    nchw_shape = [1, 4, 2, 3]
    nchw_signature = [1, 4, -1, -1] if dynamic else list(nchw_shape)
    concat_shape = [1, 8, 2, 3]
    concat_signature = [1, 8, -1, -1] if dynamic else list(concat_shape)
    post_shape = [1, 2, 3, 8]
    post_signature = [1, -1, -1, 8] if dynamic else list(post_shape)
    pre_perm = np.asarray([0, 3, 1, 2], dtype=permutation_dtype)
    post_perm = np.asarray([0, 2, 3, 1], dtype=permutation_dtype)

    model = ModelIR("elementwise_concat")
    model.inputs = ["a", "b"]
    model.outputs = ["y"]
    model.tensors = {
        "a": _tensor("a", source_shape, signature=source_signature),
        "b": _tensor("b", source_shape, signature=source_signature),
        "pre_perm": _tensor(
            "pre_perm",
            [4],
            dtype=str(permutation_dtype).upper(),
            data=pre_perm,
        ),
        "post_perm": _tensor(
            "post_perm",
            [4],
            dtype=str(permutation_dtype).upper(),
            data=post_perm,
        ),
        "a_t": _tensor("a_t", nchw_shape, signature=nchw_signature),
        "b_t": _tensor("b_t", nchw_shape, signature=nchw_signature),
        "branch": _tensor("branch", nchw_shape, signature=nchw_signature),
        "concat_t": _tensor(
            "concat_t",
            concat_shape,
            signature=concat_signature,
        ),
        "concat": _tensor("concat", post_shape, signature=post_signature),
        "y": _tensor("y", post_shape, signature=post_signature),
    }
    pre_a = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["a", "pre_perm"],
        outputs=["a_t"],
    )
    pre_b = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["b", "pre_perm"],
        outputs=["b_t"],
    )
    operators = [pre_a, pre_b]
    branch_name = "a_t"
    if binary_type is not None:
        constant = np.arange(4, dtype=np.float32).reshape(1, 4, 1, 1) / 10
        model.tensors["constant"] = _tensor(
            "constant",
            [1, 4, 1, 1],
            data=constant,
        )
        inputs = ["constant", "a_t"] if constant_first else ["a_t", "constant"]
        operators.append(
            OperatorIR(
                op_type=binary_type,
                inputs=inputs,
                outputs=["branch"],
                options={"fusedActivationFunction": "NONE"},
            )
        )
        branch_name = "branch"
    elif unary_type is not None:
        operators.append(
            OperatorIR(
                op_type=unary_type,
                inputs=["a_t"],
                outputs=["branch"],
                options=(
                    {"alpha": 0.125}
                    if unary_type in {"LEAKY_RELU", "ELU"}
                    else {}
                ),
            )
        )
        branch_name = "branch"

    second_name = "b_t"
    if direct_input:
        operators.remove(pre_b)
        second_name = "b"
    elif existing_post_boundary:
        operators.remove(pre_b)
        model.tensors["external_t"] = _tensor(
            "external_t",
            nchw_shape,
            signature=nchw_signature,
        )
        model.tensors["external"] = _tensor(
            "external",
            source_shape,
            signature=source_signature,
        )
        operators.extend(
            (
                OperatorIR(
                    op_type="CUSTOM_SOURCE",
                    inputs=["b"],
                    outputs=["external_t"],
                ),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=["external_t", "post_perm"],
                    outputs=["external"],
                ),
            )
        )
        second_name = "external_t"

    concat = OperatorIR(
        op_type="CONCATENATION",
        inputs=[branch_name, second_name],
        outputs=["concat_t"],
        options={
            "axis": -3 if negative_axis else 1,
            "fusedActivationFunction": "NONE",
        },
    )
    post = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["concat_t", "post_perm"],
        outputs=["concat"],
    )
    operators.extend((concat, post))
    if multiple_posts:
        model.tensors["concat_alias"] = _tensor(
            "concat_alias",
            post_shape,
            signature=post_signature,
        )
        model.tensors["y"] = _tensor("y", post_shape, signature=post_signature)
        operators.append(
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["concat_t", "post_perm"],
                outputs=["concat_alias"],
            )
        )
        operators.append(
            OperatorIR(
                op_type="ADD",
                inputs=["concat", "concat_alias"],
                outputs=["y"],
            )
        )
    else:
        operators.append(OperatorIR(op_type="RELU", inputs=["concat"], outputs=["y"]))

    if connected_concat:
        model.tensors["concat2_t"] = _tensor(
            "concat2_t",
            concat_shape,
            signature=concat_signature,
        )
        model.tensors["concat2"] = _tensor(
            "concat2",
            post_shape,
            signature=post_signature,
        )
        model.tensors["y2"] = _tensor("y2", post_shape, signature=post_signature)
        operators.extend(
            (
                OperatorIR(
                    op_type="CONCATENATION",
                    inputs=[branch_name, second_name],
                    outputs=["concat2_t"],
                    options={"axis": 1, "fusedActivationFunction": "NONE"},
                ),
                OperatorIR(
                    op_type="TRANSPOSE",
                    inputs=["concat2_t", "post_perm"],
                    outputs=["concat2"],
                ),
                OperatorIR(op_type="RELU", inputs=["concat2"], outputs=["y2"]),
            )
        )
        model.outputs.append("y2")
    if legacy:
        model.tensors["legacy"] = _tensor(
            "legacy",
            nchw_shape,
            signature=nchw_signature,
        )
        operators.append(
            OperatorIR(op_type="RELU", inputs=[branch_name], outputs=["legacy"])
        )
        model.outputs.append("legacy")
    if pre_fanout:
        model.tensors["pre_fanout"] = _tensor(
            "pre_fanout",
            nchw_shape,
            signature=nchw_signature,
        )
        operators.append(
            OperatorIR(op_type="RELU", inputs=["a_t"], outputs=["pre_fanout"])
        )
        model.outputs.append("pre_fanout")
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


@pytest.mark.parametrize(
    "unary_type",
    [
        "RELU",
        "RELU6",
        "LEAKY_RELU",
        "LOGISTIC",
        "TANH",
        "NEG",
        "EXP",
        "ABS",
        "SQRT",
        "GELU",
        "ELU",
    ],
)
def test_all_unary_types_are_layout_passthrough(unary_type: str) -> None:
    model = _make_model(unary_type=unary_type)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    unary = next(operator for operator in model.operators if operator.op_type == unary_type)
    assert unary.inputs == ["a"]
    assert model.tensors["branch"].shape == [1, 2, 3, 4]
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.options["axis"] == 3
    assert concat.outputs == ["concat"]


@pytest.mark.parametrize(
    ("binary_type", "constant_first"),
    [
        (binary_type, constant_first)
        for binary_type in ("ADD", "SUB", "MUL", "DIV", "MAXIMUM", "MINIMUM")
        for constant_first in (False, True)
    ],
)
def test_all_binary_types_preserve_operand_order_and_constant_layout(
    binary_type: str,
    constant_first: bool,
) -> None:
    model = _make_model(
        unary_type=None,
        binary_type=binary_type,
        constant_first=constant_first,
    )

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    binary = next(operator for operator in model.operators if operator.op_type == binary_type)
    expected = ["constant", "a"] if constant_first else ["a", "constant"]
    assert binary.inputs == expected
    assert model.tensors["constant"].shape == [1, 1, 1, 4]
    assert np.asarray(model.tensors["constant"].data).shape == (1, 1, 1, 4)


@pytest.mark.parametrize("permutation_dtype", [np.dtype(np.int32), np.dtype(np.int64)])
def test_dynamic_typed_negative_axis_updates_index_and_layout(
    permutation_dtype: np.dtype,
) -> None:
    model = _make_model(
        dynamic=True,
        negative_axis=True,
        permutation_dtype=permutation_dtype,
    )
    index = ModelIRGraphIndex(model)
    layout = LayoutState.from_model_ir(model)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(
        model,
        graph_index=index,
        layout_state=layout,
    )

    assert stats[_stats_key()] == 1
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.options["axis"] == 3
    assert model.tensors["branch"].shape_signature == [1, -1, -1, 4]
    assert model.tensors["concat"].shape_signature == [1, -1, -1, 8]
    fresh = ModelIRGraphIndex(model)
    assert index.producers == fresh.producers
    assert index.consumers == fresh.consumers
    assert index.duplicate_producers == fresh.duplicate_producers
    assert layout.validate_against_model_ir(model) == []


def test_connected_concats_are_one_group() -> None:
    model = _make_model(connected_concat=True)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    concats = [
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    ]
    assert len(concats) == 2
    assert all(operator.options["axis"] == 3 for operator in concats)
    assert all(operator.op_type != "TRANSPOSE" for operator in model.operators)


def test_legacy_consumer_receives_one_local_adapter() -> None:
    model = _make_model(legacy=True)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    legacy = next(operator for operator in model.operators if operator.outputs == ["legacy"])
    adapter_name = legacy.inputs[0]
    adapter = next(
        operator for operator in model.operators if operator.outputs == [adapter_name]
    )
    assert adapter.op_type == "TRANSPOSE"
    assert adapter.inputs == ["branch", "__nhwc_to_nchw_perm_rank4__"]
    assert model.tensors[adapter_name].shape == [1, 4, 2, 3]


def test_pre_fanout_retains_only_required_input_adapter() -> None:
    model = _make_model(pre_fanout=True)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    pre = next(operator for operator in model.operators if operator.outputs == ["a_t"])
    assert pre.op_type == "TRANSPOSE"
    branch = next(operator for operator in model.operators if operator.outputs == ["branch"])
    fanout = next(operator for operator in model.operators if operator.outputs == ["pre_fanout"])
    assert branch.inputs == ["a"]
    assert fanout.inputs == ["a_t"]


def test_existing_inverse_boundary_is_reused() -> None:
    model = _make_model(existing_post_boundary=True)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["branch", "external"]
    assert any(operator.outputs == ["external"] for operator in model.operators)


def test_direct_graph_input_boundary_is_supported() -> None:
    model = _make_model(direct_input=True)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )
    assert concat.inputs == ["branch", "b"]


def test_multiple_post_aliases_rewrite_repeated_downstream_slots() -> None:
    model = _make_model(multiple_posts=True)

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 1
    assert all(operator.op_type != "TRANSPOSE" for operator in model.operators)
    add = next(operator for operator in model.operators if operator.op_type == "ADD")
    assert add.inputs == ["concat", "concat"]


def test_numerical_equivalence_for_binary_concat_group() -> None:
    model = _make_model(unary_type=None, binary_type="MUL")
    rng = np.random.default_rng(0)
    a = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    b = rng.normal(size=(1, 2, 3, 4)).astype(np.float32)
    constant_nchw = np.asarray(model.tensors["constant"].data)
    expected = np.transpose(
        np.concatenate(
            (
                np.transpose(a, (0, 3, 1, 2)) * constant_nchw,
                np.transpose(b, (0, 3, 1, 2)),
            ),
            axis=1,
        ),
        (0, 2, 3, 1),
    )

    optimize_transpose_elementwise_concat_conv_nhwc_groups(model)
    constant_nhwc = np.asarray(model.tensors["constant"].data)
    actual = np.concatenate((a * constant_nhwc, b), axis=3)

    assert np.array_equal(actual, expected)


def test_candidate_limit_and_idempotence() -> None:
    model = _make_model()
    concat = next(
        operator for operator in model.operators if operator.op_type == "CONCATENATION"
    )

    assert optimize_transpose_elementwise_concat_conv_nhwc_groups(
        model,
        candidate=concat,
        max_rewrites=0,
    )[_stats_key()] == 0
    assert optimize_transpose_elementwise_concat_conv_nhwc_groups(
        model,
        candidate=concat,
        max_rewrites=1,
    )[_stats_key()] == 1
    first = _snapshot(model)
    assert optimize_transpose_elementwise_concat_conv_nhwc_groups(model)[
        _stats_key()
    ] == 0
    assert _snapshot(model) == first


def _stats_key() -> str:
    return "optimized_transpose_elementwise_concat_conv_nhwc_groups"


def _wrong_pre_permutation(model: ModelIR) -> None:
    model.tensors["pre_perm"].data = np.asarray([0, 2, 3, 1], dtype=np.int32)


def _wrong_post_permutation(model: ModelIR) -> None:
    model.tensors["post_perm"].data = np.asarray([0, 3, 1, 2], dtype=np.int32)


def _wrong_concat_axis(model: ModelIR) -> None:
    next(op for op in model.operators if op.op_type == "CONCATENATION").options["axis"] = 2


def _public_concat_output(model: ModelIR) -> None:
    model.outputs.append("concat_t")


def _public_post_output(model: ModelIR) -> None:
    model.outputs.append("concat")


def _nontranspose_concat_consumer(model: ModelIR) -> None:
    model.tensors["concat_t_use"] = _tensor("concat_t_use", [1, 8, 2, 3])
    model.operators.append(
        OperatorIR(op_type="RELU", inputs=["concat_t"], outputs=["concat_t_use"])
    )


def _public_closure_output(model: ModelIR) -> None:
    model.outputs.append("branch")


def _unsafe_legacy_consumer(model: ModelIR) -> None:
    model.tensors["unsafe"] = _tensor("unsafe", [1, 4, 2, 3])
    model.operators.append(
        OperatorIR(op_type="CONV_2D", inputs=["branch"], outputs=["unsafe"])
    )


def _constant_external_consumer(model: ModelIR) -> None:
    model.tensors["constant_use"] = _tensor("constant_use", [1, 4, 1, 1])
    model.operators.append(
        OperatorIR(op_type="RELU", inputs=["constant"], outputs=["constant_use"])
    )


def _wrong_constant_dtype(model: ModelIR) -> None:
    model.tensors["constant"].dtype = "FLOAT16"


def _wrong_constant_numpy_dtype(model: ModelIR) -> None:
    model.tensors["constant"].data = np.asarray(
        model.tensors["constant"].data,
        dtype=np.float64,
    )


def _wrong_constant_shape(model: ModelIR) -> None:
    model.tensors["constant"].shape = [1, 1, 4, 1]


def _per_axis_constant(model: ModelIR) -> None:
    model.tensors["constant"].quantization = QuantParamIR(
        scale=[0.1, 0.2],
        zero_point=[0, 0],
        quantized_dimension=1,
    )


def _wrong_closure_shape(model: ModelIR) -> None:
    model.tensors["branch"].shape[-1] += 1


def _wrong_closure_signature(model: ModelIR) -> None:
    model.tensors["branch"].shape_signature = [1, 4, -1, 2]


def _wrong_concat_shape(model: ModelIR) -> None:
    model.tensors["concat_t"].shape[1] += 1


def _wrong_post_shape(model: ModelIR) -> None:
    model.tensors["concat"].shape[-1] += 1


def _duplicate_closure_producer(model: ModelIR) -> None:
    model.operators.insert(
        3,
        OperatorIR(op_type="RELU", inputs=["a_t"], outputs=["branch"]),
    )


def _out_of_order_post(model: ModelIR) -> None:
    post_index = next(
        index
        for index, operator in enumerate(model.operators)
        if operator.outputs == ["concat"]
    )
    model.operators.insert(0, model.operators.pop(post_index))


def _known_source_layout_mismatch(model: ModelIR) -> None:
    model.tensors["a"].physical_layout = "NCHW"


def _variable_closure_output(model: ModelIR) -> None:
    model.tensors["branch"].is_variable = True


def _rank3_closure_output(model: ModelIR) -> None:
    model.tensors["branch"].shape = [1, 4, 6]
    model.tensors["branch"].shape_signature = [1, 4, 6]


def _unbroadcastable_constant(model: ModelIR) -> None:
    data = np.ones((1, 5, 1, 1), dtype=np.float32)
    model.tensors["constant"].data = data
    model.tensors["constant"].shape = [1, 5, 1, 1]
    model.tensors["constant"].shape_signature = [1, 5, 1, 1]


@pytest.mark.parametrize(
    "mutate",
    [
        _wrong_pre_permutation,
        _wrong_post_permutation,
        _wrong_concat_axis,
        _public_concat_output,
        _public_post_output,
        _nontranspose_concat_consumer,
        _public_closure_output,
        _unsafe_legacy_consumer,
        _constant_external_consumer,
        _wrong_constant_dtype,
        _wrong_constant_numpy_dtype,
        _wrong_constant_shape,
        _per_axis_constant,
        _wrong_closure_shape,
        _wrong_closure_signature,
        _wrong_concat_shape,
        _wrong_post_shape,
        _duplicate_closure_producer,
        _out_of_order_post,
        _known_source_layout_mismatch,
        _variable_closure_output,
        _rank3_closure_output,
        _unbroadcastable_constant,
    ],
)
def test_unsafe_candidates_are_transactional_noops(
    mutate: Callable[[ModelIR], None],
) -> None:
    model = _make_model(unary_type=None, binary_type="MUL")
    mutate(model)
    before = _snapshot(copy.deepcopy(model))

    stats = optimize_transpose_elementwise_concat_conv_nhwc_groups(model)

    assert stats[_stats_key()] == 0
    assert _snapshot(model) == before
