from __future__ import annotations

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.terminal_softmax_layout as softmax_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.terminal_softmax_layout import (
    _SOFTMAX_NHWC_PROPAGATED_MARKER,
    _optimize_terminal_softmax_transpose_after_nhwc_propagation,
)


def _tensor(
    name: str,
    *,
    dtype: str = "FLOAT32",
    shape: list[int],
    signature: list[int] | None = None,
    data: np.ndarray | None = None,
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape if signature is None else signature),
        data=data,
        is_variable=data is None,
        quantization=quantization,
    )


def _terminal_softmax_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_terminal_softmax_layout")
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        source = f"{prefix}_source"
        softmax_output = f"{prefix}_softmax_output"
        permutation = f"{prefix}_permutation"
        output = f"{prefix}_output"
        model_ir.inputs.append(source)
        model_ir.outputs.append(output)
        model_ir.tensors.update(
            {
                source: _tensor(
                    source,
                    shape=[1, 2, 3, 4],
                    signature=[1, 2, -1, 4],
                ),
                softmax_output: _tensor(
                    softmax_output,
                    shape=[1, 2, 3, 4],
                    signature=[1, 2, -1, 4],
                ),
                permutation: _tensor(
                    permutation,
                    dtype="INT32",
                    shape=[4],
                    data=np.asarray([0, 3, 1, 2], dtype=np.int32),
                ),
                output: _tensor(
                    output,
                    dtype="FLOAT16",
                    shape=[1, 4, 2, 3],
                    signature=[1, 4, 2, -1],
                ),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "SOFTMAX",
                    [source],
                    [softmax_output],
                    options={
                        "axis": 3,
                        "beta": 1.0,
                        _SOFTMAX_NHWC_PROPAGATED_MARKER: True,
                    },
                    axis_semantics={"axis": "physical"},
                    version=2,
                    onnx_node_name=f"{prefix}_softmax",
                    onnx_op_type="Softmax",
                ),
                OperatorIR(
                    "TRANSPOSE",
                    [softmax_output, permutation],
                    [output],
                ),
            ]
        )
    return model_ir


def _assert_index_current(
    model_ir: ModelIR,
    graph_index: ModelIRGraphIndex,
) -> None:
    fresh = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == fresh.producers
    assert graph_index.consumers == fresh.consumers
    assert graph_index.duplicate_producers == fresh.duplicate_producers
    assert graph_index._operator_indices_by_id == fresh._operator_indices_by_id
    assert graph_index._operator_indices_by_type == fresh._operator_indices_by_type


def test_terminal_softmax_layout_rewrites_multiple_outputs_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _terminal_softmax_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_terminal_softmax_transpose_after_nhwc_propagation(model_ir)

    assert stats == {
        "removed_terminal_softmax_transpose_after_nhwc_propagation": 2,
    }
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "SOFTMAX",
        "SOFTMAX",
    ]
    for branch_index, operator in enumerate(model_ir.operators):
        prefix = f"branch{branch_index}"
        assert operator.inputs == [f"{prefix}_source"]
        assert operator.outputs == [f"{prefix}_output"]
        assert operator.options == {"axis": 3, "beta": 1.0}
        assert operator.axis_semantics == {"axis": "physical"}
        assert operator.version == 2
        assert operator.onnx_node_name == f"{prefix}_softmax"
        assert operator.onnx_op_type == "Softmax"
        output = model_ir.tensors[f"{prefix}_output"]
        assert output.dtype == "FLOAT32"
        assert output.shape == [1, 2, 3, 4]
        assert output.shape_signature == [1, 2, -1, 4]
    assert set(model_ir.tensors) == {
        "branch0_source",
        "branch0_output",
        "branch1_source",
        "branch1_output",
    }
    lineage = model_ir.metadata["tensor_lineage_events"]
    assert [event["kind"] for event in lineage] == [
        "rename_tensor",
        "rename_tensor",
        "prune_unused_tensors",
    ]


def test_terminal_softmax_layout_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _terminal_softmax_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_terminal_softmax_transpose_after_nhwc_propagation(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "removed_terminal_softmax_transpose_after_nhwc_propagation": 1,
    }
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert set(layout_state.logical) == {"branch0_source", "branch0_output"}


def test_terminal_softmax_layout_clones_source_quantization() -> None:
    model_ir = _terminal_softmax_model(branches=1)
    source_quantization = QuantParamIR(
        scale=[0.125],
        zero_point=[-3],
        quantized_dimension=0,
    )
    source = model_ir.tensors["branch0_softmax_output"]
    destination = model_ir.tensors["branch0_output"]
    source.dtype = "INT8"
    source.quantization = source_quantization
    destination.dtype = "UINT8"
    destination.quantization = QuantParamIR([0.5], [7])

    stats = _optimize_terminal_softmax_transpose_after_nhwc_propagation(model_ir)

    assert stats == {
        "removed_terminal_softmax_transpose_after_nhwc_propagation": 1,
    }
    assert destination.dtype == "INT8"
    assert destination.quantization == source_quantization
    assert destination.quantization is not source_quantization


@pytest.mark.parametrize(
    "case",
    [
        "public_softmax_output",
        "public_softmax_input",
        "terminal_output_is_input",
        "terminal_output_consumer",
        "softmax_fanout",
        "duplicate_terminal_producer",
        "duplicate_softmax_producer",
        "reverse_operator_order",
        "missing_marker",
        "false_marker",
        "nondict_options",
        "wrong_permutation",
        "missing_permutation",
        "dynamic_permutation",
        "wrong_terminal_type",
        "extra_terminal_output",
        "wrong_softmax_type",
        "extra_softmax_input",
        "extra_softmax_output",
        "missing_source_tensor",
        "missing_destination_tensor",
        "source_rank_three",
        "source_signature_rank_three",
        "invalid_source_signature",
        "not_graph_output",
    ],
)
def test_terminal_softmax_layout_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _terminal_softmax_model(branches=1)
    softmax = model_ir.operators[0]
    transpose = model_ir.operators[1]
    if case == "public_softmax_output":
        model_ir.outputs.insert(0, "branch0_softmax_output")
    elif case == "public_softmax_input":
        model_ir.inputs.append("branch0_softmax_output")
    elif case == "terminal_output_is_input":
        model_ir.inputs.append("branch0_output")
    elif case == "terminal_output_consumer":
        model_ir.tensors["side"] = _tensor("side", shape=[1, 4, 2, 3])
        model_ir.outputs.append("side")
        model_ir.operators.append(OperatorIR("IDENTITY", ["branch0_output"], ["side"]))
    elif case == "softmax_fanout":
        model_ir.tensors["side"] = _tensor("side", shape=[1, 2, 3, 4])
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["branch0_softmax_output"], ["side"])
        )
    elif case == "duplicate_terminal_producer":
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["branch0_source"], ["branch0_output"])
        )
    elif case == "duplicate_softmax_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_source"],
                ["branch0_softmax_output"],
            )
        )
    elif case == "reverse_operator_order":
        model_ir.operators = [transpose, softmax]
    elif case == "missing_marker":
        softmax.options.pop(_SOFTMAX_NHWC_PROPAGATED_MARKER)
    elif case == "false_marker":
        softmax.options[_SOFTMAX_NHWC_PROPAGATED_MARKER] = False
    elif case == "nondict_options":
        softmax.options = None
    elif case == "wrong_permutation":
        model_ir.tensors["branch0_permutation"].data = np.asarray(
            [0, 2, 3, 1],
            dtype=np.int32,
        )
    elif case == "missing_permutation":
        del model_ir.tensors["branch0_permutation"]
    elif case == "dynamic_permutation":
        model_ir.tensors["branch0_permutation"].data = None
    elif case == "wrong_terminal_type":
        transpose.op_type = "RESHAPE"
    elif case == "extra_terminal_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            shape=[1, 4, 2, 3],
        )
        transpose.outputs.append("extra")
    elif case == "wrong_softmax_type":
        softmax.op_type = "LOGISTIC"
    elif case == "extra_softmax_input":
        softmax.inputs.append("branch0_source")
    elif case == "extra_softmax_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            shape=[1, 2, 3, 4],
        )
        softmax.outputs.append("extra")
    elif case == "missing_source_tensor":
        del model_ir.tensors["branch0_softmax_output"]
    elif case == "missing_destination_tensor":
        del model_ir.tensors["branch0_output"]
    elif case == "source_rank_three":
        model_ir.tensors["branch0_softmax_output"].shape = [1, 2, 3]
    elif case == "source_signature_rank_three":
        model_ir.tensors["branch0_softmax_output"].shape_signature = [1, 2, -1]
    elif case == "invalid_source_signature":
        model_ir.tensors["branch0_softmax_output"].shape_signature = [
            1,
            2,
            None,
            4,
        ]
    elif case == "not_graph_output":
        model_ir.tensors["other"] = _tensor("other", shape=[1])
        model_ir.inputs.append("other")
        model_ir.outputs = ["other"]

    before = repr(model_ir)
    stats = _optimize_terminal_softmax_transpose_after_nhwc_propagation(model_ir)

    assert stats == {
        "removed_terminal_softmax_transpose_after_nhwc_propagation": 0,
    }
    assert repr(model_ir) == before


def test_terminal_softmax_layout_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("terminal_softmax_layout_without_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", shape=[1, 2]),
        "y": _tensor("y", shape=[1, 2]),
        "unused": _tensor("unused", shape=[1]),
    }
    model_ir.operators = [OperatorIR("IDENTITY", ["x"], ["y"])]
    layout_state = LayoutState.from_model_ir(model_ir)

    def unexpected_index(*args, **kwargs):
        raise AssertionError("unexpected graph index allocation")

    monkeypatch.setattr(softmax_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_terminal_softmax_transpose_after_nhwc_propagation(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "removed_terminal_softmax_transpose_after_nhwc_propagation": 0,
    }
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
