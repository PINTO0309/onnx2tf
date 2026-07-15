from __future__ import annotations

import copy

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout as batch_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.validation import validate_model_ir_invariants
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.conv1d_batchmatmul_layout import (
    _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.conv1d_unary_layout import _UNARY_OPS


_STATS = "optimized_transpose_squeeze_unary_batchmatmul_nhwc_chains"


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
    squeeze_axis: int = 2,
    adj_x: bool = False,
    unary_types: tuple[str, ...] = ("ABS", "RELU"),
    produced_source: bool = False,
) -> tuple[ModelIR, dict[str, str]]:
    model_ir = ModelIR("indexed_conv1d_batchmatmul")
    names = {
        "upstream": "upstream",
        "source": "source",
        "perm": "perm",
        "pre": "pre",
        "squeezed": "squeezed",
        "rhs": "rhs",
        "output": "output",
    }
    source_shapes = {
        1: [2, 3, 4, 1],
        2: [2, 1, 4, 3],
        3: [2, 4, 1, 3],
    }
    source_shape = source_shapes[int(squeeze_axis)]
    pre_shape = [
        source_shape[0],
        source_shape[3],
        source_shape[1],
        source_shape[2],
    ]
    old_rank3_shape = [
        value for index, value in enumerate(pre_shape) if index != int(squeeze_axis)
    ]
    if old_rank3_shape != [2, 3, 4]:
        raise AssertionError(old_rank3_shape)
    rhs_shape = [2, 3 if adj_x else 4, 5]
    output_shape = [2, 4 if adj_x else 3, 5]
    for name, shape in {
        names["source"]: source_shape,
        names["pre"]: pre_shape,
        names["squeezed"]: old_rank3_shape,
        names["rhs"]: rhs_shape,
        names["output"]: output_shape,
    }.items():
        model_ir.tensors[name] = _tensor(name, shape)
    model_ir.tensors[names["perm"]] = _tensor(
        names["perm"],
        [4],
        dtype="INT32",
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    rng = np.random.default_rng(19)
    model_ir.tensors[names["rhs"]].data = rng.normal(size=rhs_shape).astype(
        np.float32
    )
    model_ir.tensors[names["rhs"]].is_variable = False
    if produced_source:
        model_ir.inputs = [names["upstream"]]
        model_ir.tensors[names["upstream"]] = _tensor(
            names["upstream"],
            source_shape,
        )
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["upstream"]], [names["source"]])
        )
    else:
        model_ir.inputs = [names["source"]]
    model_ir.operators.extend(
        [
            OperatorIR("TRANSPOSE", [names["source"], names["perm"]], [names["pre"]]),
            OperatorIR(
                "SQUEEZE",
                [names["pre"]],
                [names["squeezed"]],
                {"squeezeDims": [int(squeeze_axis)]},
            ),
        ]
    )
    current_name = names["squeezed"]
    for serial, unary_type in enumerate(unary_types):
        output_name = f"unary_{serial}"
        output_dtype = "FLOAT16" if str(unary_type) == "CAST" else str(
            model_ir.tensors[current_name].dtype
        )
        model_ir.tensors[output_name] = _tensor(
            output_name,
            old_rank3_shape,
            dtype=output_dtype,
        )
        options = (
            {"inDataType": "FLOAT32", "outDataType": "FLOAT16"}
            if str(unary_type) == "CAST"
            else {"marker": serial}
        )
        model_ir.operators.append(
            OperatorIR(
                str(unary_type),
                [current_name],
                [output_name],
                options,
                axis_semantics={"marker": "preserved"},
                version=3,
                onnx_node_name=f"unary_{serial}",
                onnx_op_type=str(unary_type).title(),
            )
        )
        current_name = output_name
    if unary_types and unary_types[-1] == "CAST":
        model_ir.tensors[names["rhs"]].dtype = "FLOAT16"
        model_ir.tensors[names["rhs"]].data = np.asarray(
            model_ir.tensors[names["rhs"]].data,
            dtype=np.float16,
        )
        model_ir.tensors[names["output"]].dtype = "FLOAT16"
    model_ir.operators.append(
        OperatorIR(
            "BATCH_MATMUL",
            [current_name, names["rhs"]],
            [names["output"]],
            {"adjX": bool(adj_x), "adjY": False, "asymmetricQuantizeInputs": False},
        )
    )
    model_ir.outputs = [names["output"]]
    names["tail"] = current_name
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
            output = np.transpose(inputs[0], tuple(int(value) for value in inputs[1]))
        elif operator.op_type == "SQUEEZE":
            output = np.squeeze(inputs[0], axis=tuple(operator.options["squeezeDims"]))
        elif operator.op_type == "ABS":
            output = np.abs(inputs[0])
        elif operator.op_type == "RELU":
            output = np.maximum(inputs[0], 0)
        elif operator.op_type == "BATCH_MATMUL":
            lhs = np.swapaxes(inputs[0], -1, -2) if operator.options["adjX"] else inputs[0]
            rhs = np.swapaxes(inputs[1], -1, -2) if operator.options["adjY"] else inputs[1]
            output = np.matmul(lhs, rhs)
        else:
            raise AssertionError(f"unsupported evaluator op: {operator.op_type}")
        values[str(operator.outputs[0])] = np.asarray(output)
    return {name: values[name] for name in model_ir.outputs}


def _assert_index_current(model_ir: ModelIR, graph_index: ModelIRGraphIndex) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


@pytest.mark.parametrize("squeeze_axis", (1, 2, 3))
@pytest.mark.parametrize("adj_x", (False, True))
@pytest.mark.parametrize("produced_source", (False, True))
def test_conv1d_batchmatmul_rewrite_is_indexed_and_numerically_equivalent(
    squeeze_axis: int,
    adj_x: bool,
    produced_source: bool,
) -> None:
    model_ir, names = _build_model(
        squeeze_axis=squeeze_axis,
        adj_x=adj_x,
        produced_source=produced_source,
    )
    source_feed_name = names["upstream"] if produced_source else names["source"]
    source_shape = model_ir.tensors[source_feed_name].shape
    feed = {
        source_feed_name: np.random.default_rng(31).normal(size=source_shape).astype(
            np.float32
        )
    }
    expected = _evaluate(copy.deepcopy(model_ir), feed)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {_STATS: 1}
    assert validate_model_ir_invariants(model_ir) == []
    assert layout_state.validate_against_model_ir(model_ir) == []
    _assert_index_current(model_ir, graph_index)
    actual = _evaluate(model_ir, feed)
    np.testing.assert_allclose(actual[names["output"]], expected[names["output"]])
    squeeze = next(operator for operator in model_ir.operators if operator.op_type == "SQUEEZE")
    assert squeeze.inputs == [names["source"]]
    assert squeeze.options["squeezeDims"] == [{1: 3, 2: 1, 3: 2}[squeeze_axis]]
    matmul = next(
        operator for operator in model_ir.operators if operator.op_type == "BATCH_MATMUL"
    )
    assert matmul.options["adjX"] is (
        bool(adj_x) if squeeze_axis == 1 else not bool(adj_x)
    )


@pytest.mark.parametrize("case", ("no_unary", "rank2_rhs", "adj_y"))
def test_conv1d_batchmatmul_preserves_optional_tail_variants(case: str) -> None:
    model_ir, names = _build_model(
        unary_types=() if case == "no_unary" else ("ABS", "RELU")
    )
    matmul = next(
        operator for operator in model_ir.operators if operator.op_type == "BATCH_MATMUL"
    )
    rhs = model_ir.tensors[names["rhs"]]
    if case == "rank2_rhs":
        rhs.data = np.asarray(rhs.data)[0]
        rhs.shape = [4, 5]
        rhs.shape_signature = [4, 5]
    elif case == "adj_y":
        rhs.data = np.swapaxes(np.asarray(rhs.data), -1, -2)
        rhs.shape = [2, 5, 4]
        rhs.shape_signature = [2, 5, 4]
        matmul.options["adjY"] = True
    feed = {
        names["source"]: np.random.default_rng(37).normal(
            size=model_ir.tensors[names["source"]].shape
        ).astype(np.float32)
    }
    expected = _evaluate(copy.deepcopy(model_ir), feed)

    stats = _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    actual = _evaluate(model_ir, feed)
    np.testing.assert_allclose(actual[names["output"]], expected[names["output"]])
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize("unary_type", sorted(_UNARY_OPS))
def test_conv1d_batchmatmul_preserves_supported_unary(unary_type: str) -> None:
    model_ir, _ = _build_model(unary_types=(unary_type,))
    original = copy.deepcopy(
        next(operator for operator in model_ir.operators if operator.op_type == unary_type)
    )

    stats = _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    unary = next(
        operator for operator in model_ir.operators if operator.onnx_node_name == "unary_0"
    )
    assert unary.op_type == original.op_type
    assert unary.options == original.options
    assert unary.axis_semantics == original.axis_semantics
    assert unary.version == original.version
    assert unary.onnx_op_type == original.onnx_op_type
    assert validate_model_ir_invariants(model_ir) == []


@pytest.mark.parametrize(
    "case",
    (
        "floating_perm",
        "produced_perm",
        "pre_public",
        "pre_fanout",
        "missing_axis",
        "squeeze_public",
        "unary_fanout",
        "unary_public",
        "rhs_position",
        "rhs_depth",
        "output_shape",
        "duplicate_output",
        "mixed_dtype",
        "per_axis_quantization",
        "backward_output_consumer",
    ),
)
def test_conv1d_batchmatmul_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir, names = _build_model()
    squeeze = next(
        operator for operator in model_ir.operators if operator.outputs == [names["squeezed"]]
    )
    matmul = next(
        operator for operator in model_ir.operators if operator.op_type == "BATCH_MATMUL"
    )
    if case == "floating_perm":
        model_ir.tensors[names["perm"]].dtype = "FLOAT32"
        model_ir.tensors[names["perm"]].data = np.asarray(
            [0, 3, 1, 2], dtype=np.float32
        )
    elif case == "produced_perm":
        model_ir.operators.append(OperatorIR("IDENTITY", [names["rhs"]], [names["perm"]]))
    elif case == "pre_public":
        model_ir.outputs.append(names["pre"])
    elif case == "pre_fanout":
        model_ir.tensors["extra"] = _tensor("extra", [2, 3, 1, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["pre"]], ["extra"]))
        model_ir.outputs.append("extra")
    elif case == "missing_axis":
        squeeze.options.clear()
    elif case == "squeeze_public":
        model_ir.outputs.append(names["squeezed"])
    elif case == "unary_fanout":
        model_ir.tensors["extra"] = _tensor("extra", [2, 3, 4])
        model_ir.operators.append(OperatorIR("IDENTITY", [names["tail"]], ["extra"]))
        model_ir.outputs.append("extra")
    elif case == "unary_public":
        model_ir.outputs.append(names["tail"])
    elif case == "rhs_position":
        matmul.inputs.reverse()
    elif case == "rhs_depth":
        model_ir.tensors[names["rhs"]].shape[1] = 5
        model_ir.tensors[names["rhs"]].shape_signature[1] = 5
    elif case == "output_shape":
        model_ir.tensors[names["output"]].shape[1] = 4
        model_ir.tensors[names["output"]].shape_signature[1] = 4
    elif case == "duplicate_output":
        model_ir.operators.append(
            OperatorIR("IDENTITY", [names["source"]], [names["output"]])
        )
    elif case == "mixed_dtype":
        model_ir.tensors[names["tail"]].dtype = "FLOAT16"
    elif case == "per_axis_quantization":
        model_ir.tensors[names["tail"]].quantization = {
            "scale": [0.25, 0.5, 0.75],
            "zero_point": [0, 0, 0],
            "quantized_dimension": 1,
        }
    elif case == "backward_output_consumer":
        model_ir.tensors["extra"] = _tensor("extra", [2, 3, 5])
        model_ir.outputs.append("extra")
        model_ir.operators.insert(
            model_ir.operators.index(matmul),
            OperatorIR("IDENTITY", [names["output"]], ["extra"]),
        )
    before = repr(model_ir)

    stats = _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(model_ir)

    assert stats == {_STATS: 0}
    assert repr(model_ir) == before


def test_conv1d_batchmatmul_preserves_dynamic_batch_signature() -> None:
    model_ir, names = _build_model(squeeze_axis=2)
    for name in (names["source"],):
        model_ir.tensors[name].shape_signature[0] = -1
    for name in (names["pre"], names["squeezed"], "unary_0", "unary_1", names["rhs"], names["output"]):
        model_ir.tensors[name].shape_signature[0] = -1

    stats = _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(model_ir)

    assert stats == {_STATS: 1}
    assert model_ir.tensors[names["squeezed"]].shape_signature == [-1, 4, 3]
    assert model_ir.tensors[names["tail"]].shape_signature == [-1, 4, 3]


def test_conv1d_batchmatmul_preflight_does_not_allocate_index(monkeypatch) -> None:
    model_ir = ModelIR("no_conv1d_batchmatmul")
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(batch_module, "ModelIRGraphIndex", unexpected_index)

    assert _optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains(model_ir) == {
        _STATS: 0
    }
