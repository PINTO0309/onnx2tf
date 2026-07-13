from __future__ import annotations

from typing import Optional

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pytorch_control_flow import (
    _rewrite_counter_bounded_while_ops_for_native_export,
    _rewrite_static_while_ops_for_native_export,
)


def _add_tensor(
    model_ir: ModelIR,
    name: str,
    dtype: str,
    *,
    shape: Optional[list[int]] = None,
    data: Optional[np.ndarray] = None,
) -> None:
    resolved_shape = list(shape or [1])
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype=dtype,
        shape=resolved_shape,
        shape_signature=list(resolved_shape),
        data=data,
    )


def _canonical_cond_subgraph(name: str, *, state_count: int) -> ModelIR:
    subgraph = ModelIR(name=name)
    subgraph.inputs = ["iter_in", "trip_in", "cond_in"] + [
        f"state_{index}_in" for index in range(state_count)
    ]
    subgraph.outputs = ["cond_eval"]
    _add_tensor(subgraph, "iter_in", "INT64")
    _add_tensor(subgraph, "trip_in", "INT64")
    _add_tensor(subgraph, "cond_in", "BOOL")
    for index in range(state_count):
        _add_tensor(subgraph, f"state_{index}_in", "FLOAT32", shape=[3])
    _add_tensor(subgraph, "iter_lt_trip", "BOOL")
    _add_tensor(subgraph, "cond_eval", "BOOL")
    subgraph.operators = [
        OperatorIR("LESS", ["iter_in", "trip_in"], ["iter_lt_trip"]),
        OperatorIR(
            "LOGICAL_AND",
            ["cond_in", "iter_lt_trip"],
            ["cond_eval"],
        ),
    ]
    return subgraph


def _static_while_model_ir() -> ModelIR:
    model_ir = ModelIR(name="static_while")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    _add_tensor(model_ir, "x", "FLOAT32", shape=[3])
    _add_tensor(model_ir, "x_alias", "FLOAT32", shape=[3])
    _add_tensor(
        model_ir,
        "iter_init",
        "INT64",
        data=np.asarray([0], dtype=np.int64),
    )
    _add_tensor(
        model_ir,
        "trip_count",
        "INT64",
        data=np.asarray([2], dtype=np.int64),
    )
    _add_tensor(
        model_ir,
        "cond_init",
        "BOOL",
        data=np.asarray([True], dtype=np.bool_),
    )
    for name, dtype, shape in [
        ("iter_out", "INT64", [1]),
        ("trip_out", "INT64", [1]),
        ("cond_out", "BOOL", [1]),
        ("y", "FLOAT32", [3]),
    ]:
        _add_tensor(model_ir, name, dtype, shape=shape)

    body = ModelIR(name="static_body")
    body.inputs = ["iter_in", "trip_in", "cond_in", "state_in"]
    body.outputs = ["iter_body", "trip_body", "cond_body", "state_body"]
    for name, dtype, shape in [
        ("iter_in", "INT64", [1]),
        ("trip_in", "INT64", [1]),
        ("cond_in", "BOOL", [1]),
        ("state_in", "FLOAT32", [3]),
        ("iter_body", "INT64", [1]),
        ("trip_body", "INT64", [1]),
        ("cond_body", "BOOL", [1]),
        ("state_body", "FLOAT32", [3]),
    ]:
        _add_tensor(body, name, dtype, shape=shape)
    _add_tensor(body, "one", "INT64", data=np.asarray([1], dtype=np.int64))
    _add_tensor(
        body,
        "scalar_shape",
        "INT32",
        data=np.asarray([1], dtype=np.int32),
    )
    body.operators = [
        OperatorIR("ADD", ["iter_in", "one"], ["iter_body"]),
        OperatorIR(
            "RESHAPE",
            ["trip_in", "scalar_shape"],
            ["trip_body"],
            {"newShape": [1], "allowZero": False},
        ),
        OperatorIR(
            "RESHAPE",
            ["cond_in", "scalar_shape"],
            ["cond_body"],
            {"newShape": [1], "allowZero": False},
        ),
        OperatorIR(
            "LOGISTIC",
            ["state_in"],
            ["state_body"],
            axis_semantics={"axis": "logical"},
            version=3,
            onnx_node_name="sigmoid_body",
            onnx_op_type="Sigmoid",
        ),
    ]
    model_ir.subgraphs = [
        _canonical_cond_subgraph("static_cond", state_count=1),
        body,
    ]
    model_ir.operators = [
        OperatorIR(
            "IDENTITY",
            ["x"],
            ["x_alias"],
            onnx_node_name="keep_me",
            onnx_op_type="Identity",
        ),
        OperatorIR(
            "WHILE",
            ["iter_init", "trip_count", "cond_init", "x_alias"],
            ["iter_out", "trip_out", "cond_out", "y"],
            {"condSubgraphIndex": 1, "bodySubgraphIndex": 2},
        ),
    ]
    return model_ir


def _counter_bounded_while_model_ir() -> ModelIR:
    model_ir = ModelIR(name="counter_while")
    model_ir.inputs = ["x", "counter"]
    model_ir.outputs = ["y"]
    for name, dtype, shape in [
        ("x", "FLOAT32", [3]),
        ("counter", "INT64", [1]),
        ("counter_i32", "INT32", [1]),
        ("cond_init", "BOOL", [1]),
        ("iter_out", "INT64", [1]),
        ("trip_out", "INT64", [1]),
        ("cond_out", "BOOL", [1]),
        ("y", "FLOAT32", [3]),
        ("counter_out", "INT64", [1]),
    ]:
        _add_tensor(model_ir, name, dtype, shape=shape)
    _add_tensor(
        model_ir,
        "iter_init",
        "INT64",
        data=np.asarray([0], dtype=np.int64),
    )
    _add_tensor(
        model_ir,
        "trip_count",
        "INT64",
        data=np.asarray([100], dtype=np.int64),
    )
    _add_tensor(
        model_ir,
        "threshold_i32",
        "INT32",
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR("CAST", ["counter"], ["counter_i32"]),
        OperatorIR("LESS", ["counter_i32", "threshold_i32"], ["cond_init"]),
    ]

    body = ModelIR(name="counter_body")
    body.inputs = ["iter_in", "trip_in", "cond_in", "state_in", "counter_in"]
    body.outputs = [
        "iter_body",
        "trip_body",
        "cond_body",
        "state_body",
        "counter_body",
    ]
    for name, dtype, shape in [
        ("iter_in", "INT64", [1]),
        ("trip_in", "INT64", [1]),
        ("cond_in", "BOOL", [1]),
        ("state_in", "FLOAT32", [3]),
        ("counter_in", "INT64", [1]),
        ("iter_body", "INT64", [1]),
        ("trip_body", "INT64", [1]),
        ("cond_body", "BOOL", [1]),
        ("state_body", "FLOAT32", [3]),
        ("counter_raw", "INT64", [1]),
        ("counter_body", "INT64", [1]),
        ("counter_i32", "INT32", [1]),
        ("threshold_i32", "INT32", [1]),
    ]:
        _add_tensor(body, name, dtype, shape=shape)
    for name, dtype, value in [
        ("one", "INT64", np.asarray([1], dtype=np.int64)),
        ("threshold_i64", "INT64", np.asarray([2], dtype=np.int64)),
        ("scalar_shape", "INT32", np.asarray([1], dtype=np.int32)),
    ]:
        _add_tensor(body, name, dtype, data=value)
    body.operators = [
        OperatorIR("ADD", ["iter_in", "one"], ["iter_body"]),
        OperatorIR("RESHAPE", ["trip_in", "scalar_shape"], ["trip_body"]),
        OperatorIR(
            "LOGISTIC",
            ["state_in"],
            ["state_body"],
            version=2,
            onnx_node_name="bounded_sigmoid",
            onnx_op_type="Sigmoid",
        ),
        OperatorIR("ADD", ["counter_in", "one"], ["counter_raw"]),
        OperatorIR("CAST", ["counter_raw"], ["counter_i32"]),
        OperatorIR("CAST", ["threshold_i64"], ["threshold_i32"]),
        OperatorIR("LESS", ["counter_i32", "threshold_i32"], ["cond_body"]),
        OperatorIR(
            "RESHAPE",
            ["counter_raw", "scalar_shape"],
            ["counter_body"],
        ),
    ]
    model_ir.subgraphs = [
        _canonical_cond_subgraph("counter_cond", state_count=2),
        body,
    ]
    model_ir.operators.append(
        OperatorIR(
            "WHILE",
            ["iter_init", "trip_count", "cond_init", "x", "counter"],
            ["iter_out", "trip_out", "cond_out", "y", "counter_out"],
            {"condSubgraphIndex": 1, "bodySubgraphIndex": 2},
        )
    )
    return model_ir


def test_control_flow_rewriters_return_borrowed_graph_for_no_op() -> None:
    model_ir = ModelIR(name="no_control_flow")
    model_ir.operators.append(OperatorIR("RELU", ["x"], ["y"]))

    assert _rewrite_static_while_ops_for_native_export(model_ir) is model_ir
    assert _rewrite_counter_bounded_while_ops_for_native_export(model_ir) is model_ir


def test_static_while_rewrite_streams_independent_provenance_complete_ops() -> None:
    source = _static_while_model_ir()

    rewritten = _rewrite_static_while_ops_for_native_export(source)

    assert rewritten is not source
    assert [op.op_type for op in source.operators] == ["IDENTITY", "WHILE"]
    assert all(op.op_type != "WHILE" for op in rewritten.operators)
    assert [op.op_type for op in rewritten.operators].count("LOGISTIC") == 2
    assert rewritten.operators[0] is not source.operators[0]
    assert rewritten.operators[0].onnx_node_name == "keep_me"
    logistic_ops = [op for op in rewritten.operators if op.op_type == "LOGISTIC"]
    assert all(op.version == 3 for op in logistic_ops)
    assert all(op.axis_semantics == {"axis": "logical"} for op in logistic_ops)
    assert all(op.onnx_node_name == "sigmoid_body" for op in logistic_ops)
    assert logistic_ops[-1].outputs == ["y"]


def test_static_while_rewrite_rejects_duplicate_body_producer() -> None:
    source = _static_while_model_ir()
    source.subgraphs[1].operators.append(
        OperatorIR("IDENTITY", ["iter_in"], ["iter_body"])
    )

    rewritten = _rewrite_static_while_ops_for_native_export(source)

    assert rewritten is source
    assert [op.op_type for op in source.operators] == ["IDENTITY", "WHILE"]


def test_counter_bounded_while_rewrite_streams_masked_outputs() -> None:
    source = _counter_bounded_while_model_ir()

    rewritten = _rewrite_counter_bounded_while_ops_for_native_export(source)

    assert rewritten is not source
    assert source.operators[-1].op_type == "WHILE"
    assert all(op.op_type != "WHILE" for op in rewritten.operators)
    assert [op.op_type for op in rewritten.operators].count("LOGISTIC") == 2
    assert any(op.op_type == "SELECT" for op in rewritten.operators)
    assert any(op.op_type == "LOGICAL_AND" for op in rewritten.operators)
    logistic_ops = [op for op in rewritten.operators if op.op_type == "LOGISTIC"]
    assert all(op.version == 2 for op in logistic_ops)
    assert all(op.onnx_node_name == "bounded_sigmoid" for op in logistic_ops)
    assert "y" in {name for op in rewritten.operators for name in op.outputs}
