from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.conv1d_tencoder_layout as tencoder_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import _prune_unused_tensors
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR
from onnx2tf.tflite_builder.passes.conv1d_tencoder_layout import (
    _optimize_tencoder_add_expand_transpose_conv_nhwc_chains,
)
from tests.test_flatbuffer_direct_indexed_conv1d_instance_norm_layout import (
    _add_branch,
    _tensor,
)


_STATS = "optimized_tencoder_add_expand_transpose_conv_nhwc_chains"


def _operator(model_ir: ModelIR, output_name: str) -> OperatorIR:
    return next(operator for operator in model_ir.operators if output_name in operator.outputs)


def _build_model(*, legacy_lhs: bool = False, side_user: bool = False) -> ModelIR:
    model_ir = ModelIR("indexed_tencoder")
    rhs = _add_branch(model_ir, "rhs")
    discarded_outputs = {
        rhs["unary"],
        rhs["expanded"],
        rhs["post"],
        rhs["terminal"],
    }
    model_ir.operators = [
        operator
        for operator in model_ir.operators
        if not discarded_outputs.intersection(operator.outputs)
    ]
    model_ir.outputs.clear()

    n, w, c = 1, 5, 3
    if legacy_lhs:
        model_ir.inputs.insert(0, "lhs_nwc")
        model_ir.tensors["lhs_nwc"] = _tensor("lhs_nwc", [n, w, c])
        model_ir.tensors["lhs_perm"] = _tensor(
            "lhs_perm",
            [3],
            dtype="INT32",
            data=np.asarray([0, 2, 1], dtype=np.int32),
        )
        model_ir.tensors["lhs_ncw"] = _tensor("lhs_ncw", [n, c, w])
        lhs_operators = [
            OperatorIR("TRANSPOSE", ["lhs_nwc", "lhs_perm"], ["lhs_ncw"])
        ]
        lhs_output = "lhs_ncw"
    else:
        model_ir.inputs.insert(0, "lhs4")
        model_ir.tensors["lhs4"] = _tensor("lhs4", [n, 1, w, c])
        model_ir.tensors["lhs_perm"] = _tensor(
            "lhs_perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        )
        model_ir.tensors["lhs4_nchw"] = _tensor(
            "lhs4_nchw",
            [n, c, 1, w],
        )
        model_ir.tensors["lhs_ncw"] = _tensor("lhs_ncw", [n, c, w])
        lhs_operators = [
            OperatorIR("TRANSPOSE", ["lhs4", "lhs_perm"], ["lhs4_nchw"]),
            OperatorIR(
                "SQUEEZE",
                ["lhs4_nchw"],
                ["lhs_ncw"],
                {"squeezeDims": [2]},
            ),
        ]
        lhs_output = "lhs_ncw"
    model_ir.operators[0:0] = lhs_operators

    constants = {
        "slice0_begin": np.asarray([0, 0, 0], dtype=np.int32),
        "slice0_size": np.asarray([n, c, w], dtype=np.int32),
        "slice1_begin": np.asarray([0, c, 0], dtype=np.int32),
        "slice1_size": np.asarray([n, c, w], dtype=np.int32),
        "expand_axis": np.asarray([2], dtype=np.int32),
        "post_perm": np.asarray([0, 2, 3, 1], dtype=np.int32),
    }
    for name, data in constants.items():
        model_ir.tensors[name] = _tensor(
            name,
            list(data.shape),
            dtype="INT32",
            data=data,
        )
    for name, shape in {
        "slice0": [n, c, w],
        "slice1": [n, c, w],
        "sigmoid": [n, c, w],
        "gate": [n, c, w],
        "rhs_gate": [n, c, w],
        "add_out": [n, c, w],
        "add4": [n, c, 1, w],
        "conv_input": [n, 1, w, c],
        "conv_out": [n, 1, w, 2],
    }.items():
        model_ir.tensors[name] = _tensor(name, shape)
    model_ir.tensors["scale"] = _tensor(
        "scale",
        [c, 1],
        data=np.asarray([[0.5], [1.0], [1.5]], dtype=np.float32),
    )
    model_ir.tensors["conv_filter"] = _tensor(
        "conv_filter",
        [2, 1, 1, c],
        data=np.asarray(
            [
                [[[0.25, 0.5, 0.75]]],
                [[[1.0, -0.5, 0.125]]],
            ],
            dtype=np.float32,
        ),
    )
    model_ir.tensors["conv_bias"] = _tensor(
        "conv_bias",
        [2],
        data=np.asarray([0.1, -0.2], dtype=np.float32),
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                "SLICE",
                [rhs["reshape2"], "slice0_begin", "slice0_size"],
                ["slice0"],
            ),
            OperatorIR(
                "SLICE",
                [rhs["reshape2"], "slice1_begin", "slice1_size"],
                ["slice1"],
            ),
            OperatorIR("LOGISTIC", ["slice1"], ["sigmoid"]),
            OperatorIR("MUL", ["slice0", "sigmoid"], ["gate"]),
            OperatorIR("MUL", ["scale", "gate"], ["rhs_gate"]),
            OperatorIR("ADD", [lhs_output, "rhs_gate"], ["add_out"]),
            OperatorIR("EXPAND_DIMS", ["add_out", "expand_axis"], ["add4"]),
            OperatorIR("TRANSPOSE", ["add4", "post_perm"], ["conv_input"]),
            OperatorIR(
                "CONV_2D",
                ["conv_input", "conv_filter", "conv_bias"],
                ["conv_out"],
            ),
        ]
    )
    model_ir.outputs.append("conv_out")
    if side_user:
        model_ir.tensors["side_bias"] = _tensor(
            "side_bias",
            [1],
            data=np.asarray([0.25], dtype=np.float32),
        )
        model_ir.tensors["side_out"] = _tensor("side_out", [n, c, w])
        model_ir.operators.append(
            OperatorIR("ADD", ["add_out", "side_bias"], ["side_out"])
        )
        model_ir.outputs.append("side_out")
    _prune_unused_tensors(model_ir)
    return model_ir


def _evaluate(model_ir: ModelIR, feeds: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    values = {
        name: np.asarray(tensor.data)
        for name, tensor in model_ir.tensors.items()
        if tensor.data is not None
    }
    values.update({name: np.asarray(value) for name, value in feeds.items()})
    for operator in model_ir.operators:
        inputs = [values[name] for name in operator.inputs]
        op_type = str(operator.op_type)
        if op_type == "TRANSPOSE":
            output = np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
        elif op_type == "SQUEEZE":
            output = np.squeeze(
                inputs[0],
                axis=tuple(int(value) for value in operator.options["squeezeDims"]),
            )
        elif op_type == "RESHAPE":
            output = np.reshape(inputs[0], tuple(int(value) for value in inputs[1]))
        elif op_type == "MEAN":
            output = np.mean(
                inputs[0],
                axis=tuple(int(value) for value in inputs[1]),
                keepdims=bool(operator.options.get("keepDims", False)),
            )
        elif op_type == "SUB":
            output = inputs[0] - inputs[1]
        elif op_type == "MUL":
            output = inputs[0] * inputs[1]
        elif op_type == "ADD":
            output = inputs[0] + inputs[1]
        elif op_type == "SQRT":
            output = np.sqrt(inputs[0])
        elif op_type == "DIV":
            output = inputs[0] / inputs[1]
        elif op_type == "SLICE":
            begin = [int(value) for value in inputs[1]]
            size = [int(value) for value in inputs[2]]
            output = inputs[0][
                tuple(
                    slice(start, start + length)
                    for start, length in zip(begin, size)
                )
            ]
        elif op_type == "LOGISTIC":
            output = 1.0 / (1.0 + np.exp(-inputs[0]))
        elif op_type == "EXPAND_DIMS":
            output = np.expand_dims(inputs[0], int(inputs[1].reshape(-1)[0]))
        elif op_type == "CONV_2D":
            output = np.einsum("nhwi,oi->nhwo", inputs[0], inputs[1][:, 0, 0, :])
            output = output + inputs[2].reshape(1, 1, 1, -1)
        elif op_type == "IDENTITY":
            output = inputs[0]
        else:
            raise AssertionError(f"unsupported evaluator op: {op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def _set_dynamic_signatures(
    model_ir: ModelIR,
    *,
    dynamic_axis: str,
    legacy_lhs: bool,
) -> tuple[int, int]:
    n_signature = -1 if dynamic_axis == "batch" else 1
    w_signature = -1 if dynamic_axis == "width" else 5
    if legacy_lhs:
        model_ir.tensors["lhs_nwc"].shape_signature = [
            n_signature,
            w_signature,
            3,
        ]
        model_ir.tensors["lhs_ncw"].shape_signature = [
            n_signature,
            3,
            w_signature,
        ]
    else:
        model_ir.tensors["lhs4"].shape_signature = [
            n_signature,
            1,
            w_signature,
            3,
        ]
        model_ir.tensors["lhs4_nchw"].shape_signature = [
            n_signature,
            3,
            1,
            w_signature,
        ]
        model_ir.tensors["lhs_ncw"].shape_signature = [
            n_signature,
            3,
            w_signature,
        ]
    model_ir.tensors["rhs_source"].shape_signature = [
        n_signature,
        1,
        w_signature,
        6,
    ]
    model_ir.tensors["rhs_pre_output"].shape_signature = [
        n_signature,
        6,
        1,
        w_signature,
    ]
    model_ir.tensors["rhs_squeezed"].shape_signature = [
        n_signature,
        6,
        w_signature,
    ]
    flat_signature = [
        n_signature,
        1,
        -1 if dynamic_axis == "width" else 30,
    ]
    for name in (
        "rhs_flat",
        "rhs_centered",
        "rhs_square",
        "rhs_norm",
        "rhs_scaled",
        "rhs_biased",
    ):
        model_ir.tensors[name].shape_signature = list(flat_signature)
    for name in (
        "rhs_mean1",
        "rhs_mean2",
        "rhs_variance",
        "rhs_standard_deviation",
        "rhs_inverse",
    ):
        model_ir.tensors[name].shape_signature = [n_signature, 1, 1]
    model_ir.tensors["rhs_reshape2"].shape_signature = [
        n_signature,
        6,
        w_signature,
    ]
    for name in (
        "slice0",
        "slice1",
        "sigmoid",
        "gate",
        "rhs_gate",
        "add_out",
    ):
        model_ir.tensors[name].shape_signature = [
            n_signature,
            3,
            w_signature,
        ]
    model_ir.tensors["add4"].shape_signature = [
        n_signature,
        3,
        1,
        w_signature,
    ]
    model_ir.tensors["conv_input"].shape_signature = [
        n_signature,
        1,
        w_signature,
        3,
    ]
    model_ir.tensors["conv_out"].shape_signature = [
        n_signature,
        1,
        w_signature,
        2,
    ]
    return n_signature, w_signature


@pytest.mark.parametrize("legacy_lhs", (False, True))
@pytest.mark.parametrize("side_user", (False, True))
def test_tencoder_rewrite_is_indexed_topological_and_numerically_equivalent(
    legacy_lhs: bool,
    side_user: bool,
) -> None:
    model_ir = _build_model(legacy_lhs=legacy_lhs, side_user=side_user)
    rng = np.random.default_rng(23)
    feeds = {
        "rhs_source": rng.normal(size=(1, 1, 5, 6)).astype(np.float32),
        ("lhs_nwc" if legacy_lhs else "lhs4"): rng.normal(
            size=(1, 5, 3) if legacy_lhs else (1, 1, 5, 3)
        ).astype(np.float32),
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    actual = _evaluate(model_ir, feeds)
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=1e-6, atol=1e-6)
    producer_positions = {
        output: index
        for index, operator in enumerate(model_ir.operators)
        for output in operator.outputs
    }
    assert all(
        input_name not in producer_positions
        or producer_positions[input_name] < operator_index
        for operator_index, operator in enumerate(model_ir.operators)
        for input_name in operator.inputs
    )


@pytest.mark.parametrize("legacy_lhs", (False, True))
@pytest.mark.parametrize("dynamic_axis", ("batch", "width"))
def test_tencoder_preserves_one_dynamic_nonchannel_dimension(
    legacy_lhs: bool,
    dynamic_axis: str,
) -> None:
    model_ir = _build_model(legacy_lhs=legacy_lhs)
    n_signature, w_signature = _set_dynamic_signatures(
        model_ir,
        dynamic_axis=dynamic_axis,
        legacy_lhs=legacy_lhs,
    )

    stats = _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    assert model_ir.tensors["rhs_reshape2"].shape_signature == [
        n_signature,
        w_signature,
        6,
    ]
    assert model_ir.tensors["add_out"].shape_signature == [
        n_signature,
        w_signature,
        3,
    ]
    assert model_ir.tensors["add4"].shape_signature == [
        n_signature,
        1,
        w_signature,
        3,
    ]
    assert np.asarray(
        model_ir.tensors[_operator(model_ir, "rhs_reshape2").inputs[1]].data
    ).reshape(-1).tolist() == [n_signature, w_signature, 6]
    assert np.asarray(
        model_ir.tensors[_operator(model_ir, "slice0").inputs[2]].data
    ).reshape(-1).tolist() == [n_signature, w_signature, 3]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    (
        "div_reversed",
        "lhs_fanout",
        "produced_reshape_shape",
        "scale_dtype",
        "nonconv_post_user",
        "duplicate_add_producer",
        "public_gate",
        "dynamic_gate_signature",
    ),
)
def test_tencoder_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir = _build_model()
    if case == "div_reversed":
        _operator(model_ir, "rhs_inverse").inputs.reverse()
    elif case == "lhs_fanout":
        model_ir.tensors["lhs_extra"] = _tensor("lhs_extra", [1, 3, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", ["lhs_ncw"], ["lhs_extra"]))
        model_ir.outputs.append("lhs_extra")
    elif case == "produced_reshape_shape":
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["rhs_axes"], ["rhs_reshape2_shape"])
        )
    elif case == "scale_dtype":
        model_ir.tensors["scale"].dtype = "FLOAT64"
        model_ir.tensors["scale"].data = np.asarray(
            model_ir.tensors["scale"].data,
            dtype=np.float64,
        )
    elif case == "nonconv_post_user":
        _operator(model_ir, "conv_out").op_type = "IDENTITY"
    elif case == "duplicate_add_producer":
        model_ir.operators.append(OperatorIR("IDENTITY", ["lhs_ncw"], ["add_out"]))
    elif case == "public_gate":
        model_ir.outputs.append("gate")
    elif case == "dynamic_gate_signature":
        model_ir.tensors["gate"].shape_signature[1] = -1
    _prune_unused_tensors(model_ir)
    before = repr(model_ir)

    stats = _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(model_ir)

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_tencoder_clones_shared_rewritten_constants() -> None:
    model_ir = _build_model()
    _operator(model_ir, "slice1").inputs[2] = "slice0_size"
    _prune_unused_tensors(model_ir)
    shared_names = (
        "rhs_reshape2_shape",
        "slice0_size",
        "slice1_begin",
        "scale",
        "expand_axis",
    )
    original_data = {
        name: np.asarray(model_ir.tensors[name].data).copy() for name in shared_names
    }
    for serial, name in enumerate(shared_names):
        output_name = f"preserved_{serial}"
        model_ir.tensors[output_name] = _tensor(
            output_name,
            list(model_ir.tensors[name].shape),
            dtype=str(model_ir.tensors[name].dtype),
        )
        model_ir.operators.append(OperatorIR("IDENTITY", [name], [output_name]))
        model_ir.outputs.append(output_name)

    stats = _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    for name, data in original_data.items():
        np.testing.assert_array_equal(model_ir.tensors[name].data, data)
    assert _operator(model_ir, "rhs_reshape2").inputs[1] != "rhs_reshape2_shape"
    slice0_size = _operator(model_ir, "slice0").inputs[2]
    slice1_size = _operator(model_ir, "slice1").inputs[2]
    assert slice0_size != "slice0_size"
    assert slice1_size != "slice0_size"
    assert slice0_size != slice1_size
    assert _operator(model_ir, "slice1").inputs[1] != "slice1_begin"
    assert "scale" not in _operator(model_ir, "rhs_gate").inputs
    assert _operator(model_ir, "add4").inputs[1] != "expand_axis"
    assert validate_model_ir_invariants(model_ir) == []


def test_tencoder_preflight_prunes_without_allocating_index(monkeypatch) -> None:
    model_ir = ModelIR("no_tencoder")
    model_ir.tensors["unused"] = _tensor("unused", [1])
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(tencoder_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_tencoder_add_expand_transpose_conv_nhwc_chains(model_ir) == {
        _STATS: 0
    }
    assert "unused" not in model_ir.tensors
