from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.decoder_deconv_layout as decoder_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.decoder_deconv_layout import (
    _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input,
)


_STATS = "optimized_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input"


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data=None,
    quantization=None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _build_model(
    *,
    adj_x: bool = False,
    adj_y: bool = False,
    commuted_add: bool = False,
    produced_operands: bool = False,
) -> tuple[ModelIR, dict[str, str]]:
    model_ir = ModelIR("indexed_decoder_deconv")
    names = {
        "lhs_upstream": "lhs_upstream",
        "rhs_upstream": "rhs_upstream",
        "lhs": "lhs",
        "rhs": "rhs",
        "matmul": "matmul",
        "bias": "bias",
        "add": "add",
        "axis": "axis",
        "expanded": "expanded",
        "perm": "perm",
        "transposed": "transposed",
        "output_shape": "output_shape",
        "filter": "filter",
        "output": "output",
    }
    n, c, depth, length = 2, 3, 4, 5
    lhs_shape = [n, depth, c] if adj_x else [n, c, depth]
    rhs_shape = [n, length, depth] if adj_y else [n, depth, length]
    rng = np.random.default_rng(41)
    for name, shape in {
        names["lhs"]: lhs_shape,
        names["rhs"]: rhs_shape,
        names["matmul"]: [n, c, length],
        names["add"]: [n, c, length],
        names["expanded"]: [n, c, 1, length],
        names["transposed"]: [n, 1, length, c],
        names["output"]: [n, 1, length, c],
    }.items():
        model_ir.tensors[name] = _tensor(name, shape)
    model_ir.tensors[names["bias"]] = _tensor(
        names["bias"],
        [length],
        data=rng.normal(size=(length,)).astype(np.float32),
    )
    model_ir.tensors[names["axis"]] = _tensor(
        names["axis"],
        [1],
        dtype="INT32",
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors[names["perm"]] = _tensor(
        names["perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors[names["output_shape"]] = _tensor(
        names["output_shape"],
        [4],
        dtype="INT32",
        data=np.asarray([n, 1, length, c], dtype=np.int32),
    )
    model_ir.tensors[names["filter"]] = _tensor(
        names["filter"],
        [c, 1, 1, c],
        data=np.ones((c, 1, 1, c), dtype=np.float32),
    )
    if produced_operands:
        model_ir.inputs = [names["lhs_upstream"], names["rhs_upstream"]]
        model_ir.tensors[names["lhs_upstream"]] = _tensor(
            names["lhs_upstream"], lhs_shape
        )
        model_ir.tensors[names["rhs_upstream"]] = _tensor(
            names["rhs_upstream"], rhs_shape
        )
        model_ir.operators.extend(
            [
                OperatorIR("IDENTITY", [names["lhs_upstream"]], [names["lhs"]]),
                OperatorIR("IDENTITY", [names["rhs_upstream"]], [names["rhs"]]),
            ]
        )
    else:
        model_ir.inputs = [names["lhs"], names["rhs"]]
    add_inputs = [names["matmul"], names["bias"]]
    if commuted_add:
        add_inputs.reverse()
    model_ir.operators.extend(
        [
            OperatorIR(
                "BATCH_MATMUL",
                [names["lhs"], names["rhs"]],
                [names["matmul"]],
                {"adjX": bool(adj_x), "adjY": bool(adj_y), "marker": "keep"},
                axis_semantics={"marker": "preserved"},
                version=3,
                onnx_node_name="decoder_matmul",
                onnx_op_type="MatMul",
            ),
            OperatorIR("ADD", add_inputs, [names["add"]], {"marker": "keep"}),
            OperatorIR(
                "EXPAND_DIMS",
                [names["add"], names["axis"]],
                [names["expanded"]],
            ),
            OperatorIR(
                "TRANSPOSE",
                [names["expanded"], names["perm"]],
                [names["transposed"]],
            ),
            OperatorIR(
                "TRANSPOSE_CONV",
                [names["output_shape"], names["filter"], names["transposed"]],
                [names["output"]],
                {"marker": "keep"},
            ),
        ]
    )
    model_ir.outputs = [names["output"]]
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
        elif operator.op_type == "BATCH_MATMUL":
            lhs = np.swapaxes(inputs[0], -1, -2) if operator.options["adjX"] else inputs[0]
            rhs = np.swapaxes(inputs[1], -1, -2) if operator.options["adjY"] else inputs[1]
            output = np.matmul(lhs, rhs)
        elif operator.op_type == "ADD":
            output = inputs[0] + inputs[1]
        elif operator.op_type == "EXPAND_DIMS":
            output = np.expand_dims(inputs[0], int(inputs[1].reshape(-1)[0]))
        elif operator.op_type == "TRANSPOSE":
            output = np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
        elif operator.op_type == "TRANSPOSE_CONV":
            output = inputs[2]
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


def _operator(model_ir: ModelIR, output_name: str) -> OperatorIR:
    return next(operator for operator in model_ir.operators if output_name in operator.outputs)


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


@pytest.mark.parametrize("adj_x", (False, True))
@pytest.mark.parametrize("adj_y", (False, True))
@pytest.mark.parametrize("commuted_add", (False, True))
@pytest.mark.parametrize("produced_operands", (False, True))
def test_decoder_deconv_rewrite_is_indexed_and_numerically_equivalent(
    adj_x: bool,
    adj_y: bool,
    commuted_add: bool,
    produced_operands: bool,
) -> None:
    model_ir, names = _build_model(
        adj_x=adj_x,
        adj_y=adj_y,
        commuted_add=commuted_add,
        produced_operands=produced_operands,
    )
    rng = np.random.default_rng(43)
    lhs_feed = names["lhs_upstream"] if produced_operands else names["lhs"]
    rhs_feed = names["rhs_upstream"] if produced_operands else names["rhs"]
    feeds = {
        lhs_feed: rng.normal(size=model_ir.tensors[lhs_feed].shape).astype(np.float32),
        rhs_feed: rng.normal(size=model_ir.tensors[rhs_feed].shape).astype(np.float32),
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)
    original_matmul = copy.deepcopy(_operator(model_ir, names["matmul"]))
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    actual = _evaluate(model_ir, feeds)
    np.testing.assert_allclose(actual[names["output"]], expected[names["output"]])
    matmul = _operator(model_ir, names["matmul"])
    assert matmul.inputs == [names["rhs"], names["lhs"]]
    assert matmul.options["adjX"] is (not bool(adj_y))
    assert matmul.options["adjY"] is (not bool(adj_x))
    assert matmul.options["marker"] == original_matmul.options["marker"]
    assert matmul.axis_semantics == original_matmul.axis_semantics
    assert matmul.version == original_matmul.version
    assert matmul.onnx_node_name == original_matmul.onnx_node_name
    assert np.asarray(model_ir.tensors[_operator(model_ir, names["add"]).inputs[1 if not commuted_add else 0]].data).shape == (1, 5, 1)
    assert _operator(model_ir, names["expanded"]).inputs[1] == names["axis"]
    assert np.asarray(model_ir.tensors[names["axis"]].data).tolist() == [1]
    assert _operator(model_ir, names["output"]).inputs[2] == names["expanded"]


def test_decoder_deconv_clones_shared_bias_and_axis() -> None:
    model_ir, names = _build_model()
    originals = {
        names["bias"]: np.asarray(model_ir.tensors[names["bias"]].data).copy(),
        names["axis"]: np.asarray(model_ir.tensors[names["axis"]].data).copy(),
    }
    for serial, name in enumerate((names["bias"], names["axis"])):
        output_name = f"preserved_{serial}"
        tensor = model_ir.tensors[name]
        model_ir.tensors[output_name] = _tensor(
            output_name,
            list(tensor.shape),
            dtype=str(tensor.dtype),
        )
        model_ir.operators.append(OperatorIR("IDENTITY", [name], [output_name]))
        model_ir.outputs.append(output_name)

    stats = _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir
    )

    assert stats == {_STATS: 1}
    for name, data in originals.items():
        np.testing.assert_array_equal(model_ir.tensors[name].data, data)
    add = _operator(model_ir, names["add"])
    expand = _operator(model_ir, names["expanded"])
    assert names["bias"] not in add.inputs
    assert expand.inputs[1] != names["axis"]
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    ("rank2_rhs", "rank2_bias", "rank3_bias", "negative_axis"),
)
def test_decoder_deconv_preserves_optional_input_variants(case: str) -> None:
    model_ir, names = _build_model()
    if case == "rank2_rhs":
        model_ir.tensors[names["rhs"]].shape = [4, 5]
        model_ir.tensors[names["rhs"]].shape_signature = [4, 5]
    elif case == "rank2_bias":
        model_ir.tensors[names["bias"]].shape = [1, 5]
        model_ir.tensors[names["bias"]].shape_signature = [1, 5]
        model_ir.tensors[names["bias"]].data = np.asarray(
            model_ir.tensors[names["bias"]].data
        ).reshape(1, 5)
    elif case == "rank3_bias":
        model_ir.tensors[names["bias"]].shape = [1, 1, 5]
        model_ir.tensors[names["bias"]].shape_signature = [1, 1, 5]
        model_ir.tensors[names["bias"]].data = np.asarray(
            model_ir.tensors[names["bias"]].data
        ).reshape(1, 1, 5)
    elif case == "negative_axis":
        model_ir.tensors[names["axis"]].data[0] = -2
    rng = np.random.default_rng(47)
    feeds = {
        name: rng.normal(size=model_ir.tensors[name].shape).astype(np.float32)
        for name in model_ir.inputs
    }
    expected = _evaluate(copy.deepcopy(model_ir), feeds)

    stats = _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir
    )

    assert stats == {_STATS: 1}
    actual = _evaluate(model_ir, feeds)
    np.testing.assert_allclose(actual[names["output"]], expected[names["output"]])
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    (
        "floating_perm",
        "produced_perm",
        "transpose_public",
        "transpose_fanout",
        "wrong_deconv_input",
        "expand_axis",
        "floating_axis",
        "produced_axis",
        "expand_public",
        "expand_fanout",
        "add_public",
        "add_fanout",
        "matmul_public",
        "matmul_fanout",
        "two_matmul_inputs",
        "produced_bias",
        "bias_shape",
        "bias_dtype",
        "contracted_dimension",
        "matmul_shape",
        "expand_shape",
        "transpose_shape",
        "duplicate_transpose",
        "per_axis_quantization",
        "transpose_quantization_grid",
        "backward_deconvolution",
    ),
)
def test_decoder_deconv_rejects_unsafe_contracts_transactionally(case: str) -> None:
    model_ir, names = _build_model()
    transpose = _operator(model_ir, names["transposed"])
    deconvolution = _operator(model_ir, names["output"])
    if case == "floating_perm":
        model_ir.tensors[names["perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["perm"]].data = np.asarray(
            [0, 2, 3, 1], dtype=np.float32
        )
    elif case == "produced_perm":
        model_ir.operators.append(OperatorIR("IDENTITY", [names["axis"]], [names["perm"]]))
    elif case == "transpose_public":
        model_ir.outputs.append(names["transposed"])
    elif case == "transpose_fanout":
        model_ir.tensors["extra"] = _tensor("extra", [2, 1, 5, 3])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["transposed"]], ["extra"]))
        model_ir.outputs.append("extra")
    elif case == "wrong_deconv_input":
        deconvolution.inputs[2] = names["expanded"]
    elif case == "expand_axis":
        model_ir.tensors[names["axis"]].data[0] = 1
    elif case == "floating_axis":
        model_ir.tensors[names["axis"]].dtype = "FLOAT32"
        model_ir.tensors[names["axis"]].data = np.asarray([2], dtype=np.float32)
    elif case == "produced_axis":
        model_ir.operators.append(OperatorIR("IDENTITY", [names["perm"]], [names["axis"]]))
    elif case == "expand_public":
        model_ir.outputs.append(names["expanded"])
    elif case == "expand_fanout":
        model_ir.tensors["extra"] = _tensor("extra", [2, 3, 1, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["expanded"]], ["extra"]))
        model_ir.outputs.append("extra")
    elif case == "add_public":
        model_ir.outputs.append(names["add"])
    elif case == "add_fanout":
        model_ir.tensors["extra"] = _tensor("extra", [2, 3, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["add"]], ["extra"]))
        model_ir.outputs.append("extra")
    elif case == "matmul_public":
        model_ir.outputs.append(names["matmul"])
    elif case == "matmul_fanout":
        model_ir.tensors["extra"] = _tensor("extra", [2, 3, 5])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["matmul"]], ["extra"]))
        model_ir.outputs.append("extra")
    elif case == "two_matmul_inputs":
        _operator(model_ir, names["add"]).inputs[1] = names["matmul"]
    elif case == "produced_bias":
        model_ir.operators.append(OperatorIR("IDENTITY", [names["lhs"]], [names["bias"]]))
    elif case == "bias_shape":
        model_ir.tensors[names["bias"]].shape = [5, 1]
        model_ir.tensors[names["bias"]].shape_signature = [5, 1]
        model_ir.tensors[names["bias"]].data = np.asarray(
            model_ir.tensors[names["bias"]].data
        ).reshape(5, 1)
    elif case == "bias_dtype":
        model_ir.tensors[names["bias"]].dtype = "FLOAT64"
        model_ir.tensors[names["bias"]].data = np.asarray(
            model_ir.tensors[names["bias"]].data, dtype=np.float64
        )
    elif case == "contracted_dimension":
        model_ir.tensors[names["rhs"]].shape[1] = 6
        model_ir.tensors[names["rhs"]].shape_signature[1] = 6
    elif case == "matmul_shape":
        model_ir.tensors[names["matmul"]].shape[1] = 4
        model_ir.tensors[names["matmul"]].shape_signature[1] = 4
    elif case == "expand_shape":
        model_ir.tensors[names["expanded"]].shape[2] = 2
        model_ir.tensors[names["expanded"]].shape_signature[2] = 2
    elif case == "transpose_shape":
        model_ir.tensors[names["transposed"]].shape[2] = 4
        model_ir.tensors[names["transposed"]].shape_signature[2] = 4
    elif case == "duplicate_transpose":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["expanded"]], [names["transposed"]])
        )
    elif case == "per_axis_quantization":
        model_ir.tensors[names["matmul"]].quantization = {
            "scale": [0.25, 0.5, 0.75],
            "zero_point": [0, 0, 0],
            "quantized_dimension": 1,
        }
    elif case == "transpose_quantization_grid":
        model_ir.tensors[names["expanded"]].quantization = {
            "scale": [0.25],
            "zero_point": [0],
        }
        model_ir.tensors[names["transposed"]].quantization = {
            "scale": [0.5],
            "zero_point": [0],
        }
    elif case == "backward_deconvolution":
        model_ir.operators.remove(deconvolution)
        model_ir.operators.insert(model_ir.operators.index(transpose), deconvolution)
    before = repr(model_ir)

    stats = _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_decoder_deconv_clone_failure_is_transactional() -> None:
    class Unclonable:
        def __deepcopy__(self, memo):
            raise RuntimeError("cannot clone")

    model_ir, names = _build_model()
    model_ir.tensors[names["bias"]].quantization = {"fault": Unclonable()}
    model_ir.tensors["preserved"] = _tensor("preserved", [5])
    model_ir.operators.append(
        OperatorIR("IDENTITY", [names["bias"]], ["preserved"])
    )
    model_ir.outputs.append("preserved")
    before = repr(model_ir)

    stats = _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir
    )

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_decoder_deconv_preserves_dynamic_batch_signature() -> None:
    model_ir, names = _build_model()
    for name in (
        names["lhs"],
        names["rhs"],
        names["matmul"],
        names["add"],
        names["expanded"],
        names["transposed"],
        names["output"],
    ):
        model_ir.tensors[name].shape_signature[0] = -1

    stats = _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir
    )

    assert stats == {_STATS: 1}
    assert model_ir.tensors[names["matmul"]].shape_signature == [-1, 5, 3]
    assert model_ir.tensors[names["add"]].shape_signature == [-1, 5, 3]
    assert model_ir.tensors[names["expanded"]].shape_signature == [-1, 1, 5, 3]


def test_decoder_deconv_preflight_does_not_allocate_index(monkeypatch) -> None:
    model_ir = ModelIR("no_decoder_deconv")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(decoder_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input(
        model_ir
    ) == {_STATS: 0}
