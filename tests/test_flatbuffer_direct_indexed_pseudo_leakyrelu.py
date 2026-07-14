from __future__ import annotations

import numpy as np
import pytest
import onnx2tf.tflite_builder.passes.graph_cleanup as cleanup_module

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    _optimize_fuse_pseudo_leakyrelu_chains,
)


def _tensor(
    name: str,
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=[] if data is not None else [1, 3],
        shape_signature=[] if data is not None else [1, 3],
        data=data,
        is_variable=data is None,
    )


def _pseudo_leakyrelu_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_pseudo_leakyrelu")
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        source = f"{prefix}_source"
        negative = f"{prefix}_negative"
        negative_relu = f"{prefix}_negative_relu"
        alpha = f"{prefix}_alpha"
        scaled = f"{prefix}_scaled"
        positive_relu = f"{prefix}_positive_relu"
        output = f"{prefix}_output"
        model_ir.inputs.append(source)
        model_ir.outputs.append(output)
        for tensor_name in (
            source,
            negative,
            negative_relu,
            scaled,
            positive_relu,
            output,
        ):
            model_ir.tensors[tensor_name] = _tensor(tensor_name)
        model_ir.tensors[alpha] = _tensor(
            alpha,
            data=np.asarray(0.125 * (branch_index + 1), dtype=np.float32),
        )
        multiply_inputs = (
            [alpha, negative_relu] if branch_index % 2 == 0 else [negative_relu, alpha]
        )
        model_ir.operators.extend(
            [
                OperatorIR("NEG", [source], [negative]),
                OperatorIR("RELU", [negative], [negative_relu]),
                OperatorIR("MUL", multiply_inputs, [scaled]),
                OperatorIR("RELU", [source], [positive_relu]),
                OperatorIR(
                    "SUB",
                    [positive_relu, scaled],
                    [output],
                    options={"legacy": True},
                    axis_semantics={"axis": "legacy"},
                    version=2,
                    onnx_node_name="expanded_leakyrelu",
                    onnx_op_type="Sub",
                ),
            ]
        )
    return model_ir


def _fresh_index_matches(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_pseudo_leakyrelu_fuses_multiple_constant_orders_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _pseudo_leakyrelu_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_fuse_pseudo_leakyrelu_chains(model_ir)

    assert stats == {"fused_pseudo_leakyrelu_chains": 2}
    assert refresh_count == 1
    assert [str(operator.op_type) for operator in model_ir.operators] == [
        "LEAKY_RELU",
        "LEAKY_RELU",
    ]
    assert [operator.options for operator in model_ir.operators] == [
        {"alpha": 0.125},
        {"alpha": 0.25},
    ]
    assert [operator.inputs for operator in model_ir.operators] == [
        ["branch0_source"],
        ["branch1_source"],
    ]
    assert [operator.outputs for operator in model_ir.operators] == [
        ["branch0_output"],
        ["branch1_output"],
    ]
    assert all(operator.axis_semantics == {} for operator in model_ir.operators)
    assert all(operator.version == 1 for operator in model_ir.operators)
    assert all(operator.onnx_node_name is None for operator in model_ir.operators)
    assert all(operator.onnx_op_type is None for operator in model_ir.operators)
    assert set(model_ir.tensors) == {
        "branch0_source",
        "branch0_output",
        "branch1_source",
        "branch1_output",
    }


def test_pseudo_leakyrelu_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _pseudo_leakyrelu_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_fuse_pseudo_leakyrelu_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"fused_pseudo_leakyrelu_chains": 1}
    _fresh_index_matches(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert set(layout_state.logical) == {"branch0_source", "branch0_output"}


@pytest.mark.parametrize(
    "case",
    [
        "reversed_sub",
        "nonconstant_alpha",
        "public_negative",
        "public_negative_relu",
        "public_scaled",
        "public_positive_relu",
        "negative_fanout",
        "negative_relu_fanout",
        "scaled_fanout",
        "positive_relu_fanout",
        "source_mismatch",
        "integer_source",
        "integer_output",
    ],
)
def test_pseudo_leakyrelu_preserves_grammar_boundary_and_fanout_guards(
    case: str,
) -> None:
    model_ir = _pseudo_leakyrelu_model(branches=1)
    names = {
        "public_negative": "branch0_negative",
        "public_negative_relu": "branch0_negative_relu",
        "public_scaled": "branch0_scaled",
        "public_positive_relu": "branch0_positive_relu",
    }
    fanout_names = {
        "negative_fanout": "branch0_negative",
        "negative_relu_fanout": "branch0_negative_relu",
        "scaled_fanout": "branch0_scaled",
        "positive_relu_fanout": "branch0_positive_relu",
    }
    if case == "reversed_sub":
        model_ir.operators[4].inputs = ["branch0_scaled", "branch0_positive_relu"]
    elif case == "nonconstant_alpha":
        model_ir.tensors["branch0_alpha"].data = None
    elif case in names:
        model_ir.outputs.append(names[case])
    elif case in fanout_names:
        model_ir.tensors["side"] = _tensor("side")
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", [fanout_names[case]], ["side"])
        )
    elif case == "source_mismatch":
        model_ir.tensors["other"] = _tensor("other")
        model_ir.inputs.append("other")
        model_ir.operators[0].inputs = ["other"]
    elif case == "integer_source":
        model_ir.tensors["branch0_source"].dtype = "INT8"
    elif case == "integer_output":
        model_ir.tensors["branch0_output"].dtype = "INT8"

    before_operators = [
        (
            str(operator.op_type),
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
            dict(operator.options),
        )
        for operator in model_ir.operators
    ]
    before_tensors = set(model_ir.tensors)

    stats = _optimize_fuse_pseudo_leakyrelu_chains(model_ir)

    assert stats == {"fused_pseudo_leakyrelu_chains": 0}
    assert [
        (
            str(operator.op_type),
            [str(name) for name in operator.inputs],
            [str(name) for name in operator.outputs],
            dict(operator.options),
        )
        for operator in model_ir.operators
    ] == before_operators
    assert set(model_ir.tensors) == before_tensors


def test_pseudo_leakyrelu_skips_index_without_complete_operator_family(
    monkeypatch,
) -> None:
    model_ir = ModelIR("pseudo_leakyrelu_without_sub")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "y": _tensor("y"),
        "unused": _tensor("unused"),
    }
    model_ir.operators = [OperatorIR("RELU", ["x"], ["y"])]

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(cleanup_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_fuse_pseudo_leakyrelu_chains(model_ir)

    assert stats == {"fused_pseudo_leakyrelu_chains": 0}
    assert set(model_ir.tensors) == {"x", "y"}
