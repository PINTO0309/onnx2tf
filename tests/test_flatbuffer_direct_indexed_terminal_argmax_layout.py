from __future__ import annotations

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.terminal_argmax_layout as argmax_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.terminal_argmax_layout import (
    _optimize_transpose_pre_argmax_nhwc_terminal_chains,
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


def _terminal_argmax_model(
    *,
    branches: int = 2,
    shared_axis: bool = True,
    negative_axis: bool = False,
) -> ModelIR:
    model_ir = ModelIR("indexed_terminal_argmax_layout")
    axis_value = -3 if negative_axis else 1
    axis_quantization = QuantParamIR([0.25], [0])
    if shared_axis:
        model_ir.tensors["axis"] = _tensor(
            "axis",
            dtype="INT64",
            shape=[1],
            data=np.asarray([axis_value], dtype=np.int64),
            quantization=axis_quantization,
        )
    model_ir.tensors["permutation"] = _tensor(
        "permutation",
        dtype="INT32",
        shape=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )

    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        source = f"{prefix}_source"
        transposed = f"{prefix}_transposed"
        output = f"{prefix}_output"
        axis = "axis" if shared_axis else f"{prefix}_axis"
        if not shared_axis:
            model_ir.tensors[axis] = _tensor(
                axis,
                dtype="INT64",
                shape=[1],
                data=np.asarray([axis_value], dtype=np.int64),
                quantization=QuantParamIR([0.25], [0]),
            )
        model_ir.inputs.append(source)
        model_ir.outputs.append(output)
        model_ir.tensors.update(
            {
                source: _tensor(
                    source,
                    shape=[1, 2, 3, 4],
                    signature=[1, 2, -1, 4],
                ),
                transposed: _tensor(
                    transposed,
                    shape=[1, 4, 2, 3],
                    signature=[1, 4, 2, -1],
                ),
                output: _tensor(
                    output,
                    dtype="INT64",
                    shape=[1, 2, 3],
                    signature=[1, 2, -1],
                ),
            }
        )
        model_ir.operators.extend(
            [
                OperatorIR(
                    "TRANSPOSE",
                    [source, "permutation"],
                    [transposed],
                ),
                OperatorIR(
                    "ARG_MAX",
                    [transposed, axis],
                    [output],
                    options={"outputType": "INT64"},
                    axis_semantics={"axis": "physical"},
                    version=2,
                    onnx_node_name=f"{prefix}_argmax",
                    onnx_op_type="ArgMax",
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


def test_terminal_argmax_layout_rewrites_shared_axis_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _terminal_argmax_model()
    original_quantization = model_ir.tensors["axis"].quantization
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_transpose_pre_argmax_nhwc_terminal_chains(model_ir)

    assert stats == {"optimized_transpose_pre_argmax_nhwc_terminal_chains": 2}
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "ARG_MAX",
        "ARG_MAX",
    ]
    assert model_ir.operators[0].inputs == ["branch0_source", "axis_nhwc"]
    assert model_ir.operators[1].inputs == ["branch1_source", "axis"]
    assert all(
        operator.options == {"outputType": "INT64"} for operator in model_ir.operators
    )
    assert all(
        operator.axis_semantics == {"axis": "physical"}
        for operator in model_ir.operators
    )
    assert all(operator.version == 2 for operator in model_ir.operators)
    np.testing.assert_array_equal(model_ir.tensors["axis"].data, [3])
    np.testing.assert_array_equal(model_ir.tensors["axis_nhwc"].data, [3])
    assert model_ir.tensors["axis_nhwc"].dtype == "INT64"
    assert model_ir.tensors["axis_nhwc"].quantization == original_quantization
    assert model_ir.tensors["axis_nhwc"].quantization is not original_quantization
    assert set(model_ir.tensors) == {
        "axis",
        "axis_nhwc",
        "branch0_source",
        "branch0_output",
        "branch1_source",
        "branch1_output",
    }


def test_terminal_argmax_layout_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _terminal_argmax_model(branches=1, shared_axis=False)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_transpose_pre_argmax_nhwc_terminal_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"optimized_transpose_pre_argmax_nhwc_terminal_chains": 1}
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert set(layout_state.logical) == {
        "branch0_axis",
        "branch0_source",
        "branch0_output",
    }


def test_terminal_argmax_layout_normalizes_negative_channel_axis() -> None:
    model_ir = _terminal_argmax_model(
        branches=1,
        shared_axis=False,
        negative_axis=True,
    )

    stats = _optimize_transpose_pre_argmax_nhwc_terminal_chains(model_ir)

    assert stats == {"optimized_transpose_pre_argmax_nhwc_terminal_chains": 1}
    np.testing.assert_array_equal(model_ir.tensors["branch0_axis"].data, [3])


@pytest.mark.parametrize("boundary", ["input", "output"])
def test_terminal_argmax_layout_clones_public_axis_constant(
    boundary: str,
) -> None:
    model_ir = _terminal_argmax_model(branches=1)
    getattr(model_ir, f"{boundary}s").append("axis")

    stats = _optimize_transpose_pre_argmax_nhwc_terminal_chains(model_ir)

    assert stats == {"optimized_transpose_pre_argmax_nhwc_terminal_chains": 1}
    assert model_ir.operators[0].inputs == ["branch0_source", "axis_nhwc"]
    np.testing.assert_array_equal(model_ir.tensors["axis"].data, [1])
    np.testing.assert_array_equal(model_ir.tensors["axis_nhwc"].data, [3])


@pytest.mark.parametrize(
    "case",
    [
        "source_public_output",
        "transposed_public_output",
        "transposed_public_input",
        "transposed_fanout",
        "duplicate_transposed_producer",
        "reverse_operator_order",
        "wrong_permutation",
        "missing_permutation",
        "dynamic_permutation",
        "extra_transpose_input",
        "extra_transpose_output",
        "wrong_argmax_type",
        "extra_argmax_input",
        "extra_argmax_output",
        "argmax_data_position",
        "wrong_axis",
        "out_of_range_axis",
        "multiple_axes",
        "float_axis",
        "unsigned_axis",
        "dynamic_axis",
        "missing_axis",
        "missing_source_tensor",
        "missing_transposed_tensor",
        "missing_output_tensor",
        "source_rank_three",
        "transposed_shape_mismatch",
        "transposed_signature_mismatch",
        "output_shape_mismatch",
        "output_signature_mismatch",
        "dtype_mismatch",
        "invalid_source_signature",
    ],
)
def test_terminal_argmax_layout_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _terminal_argmax_model(branches=1)
    transpose = model_ir.operators[0]
    argmax = model_ir.operators[1]
    axis = model_ir.tensors["axis"]

    if case == "source_public_output":
        model_ir.outputs.append("branch0_source")
    elif case == "transposed_public_output":
        model_ir.outputs.append("branch0_transposed")
    elif case == "transposed_public_input":
        model_ir.inputs.append("branch0_transposed")
    elif case == "transposed_fanout":
        model_ir.tensors["side"] = _tensor("side", shape=[1, 4, 2, 3])
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["branch0_transposed"], ["side"])
        )
    elif case == "duplicate_transposed_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_source"],
                ["branch0_transposed"],
            )
        )
    elif case == "reverse_operator_order":
        model_ir.operators = [argmax, transpose]
    elif case == "wrong_permutation":
        model_ir.tensors["permutation"].data = np.asarray(
            [0, 2, 3, 1],
            dtype=np.int32,
        )
    elif case == "missing_permutation":
        del model_ir.tensors["permutation"]
    elif case == "dynamic_permutation":
        model_ir.tensors["permutation"].data = None
    elif case == "extra_transpose_input":
        transpose.inputs.append("axis")
    elif case == "extra_transpose_output":
        model_ir.tensors["extra"] = _tensor(
            "extra",
            shape=[1, 4, 2, 3],
        )
        transpose.outputs.append("extra")
    elif case == "wrong_argmax_type":
        argmax.op_type = "ARG_MIN"
    elif case == "extra_argmax_input":
        argmax.inputs.append("axis")
    elif case == "extra_argmax_output":
        model_ir.tensors["extra"] = _tensor("extra", shape=[1, 2, 3])
        argmax.outputs.append("extra")
    elif case == "argmax_data_position":
        argmax.inputs = ["axis", "branch0_transposed"]
    elif case == "wrong_axis":
        axis.data = np.asarray([0], dtype=np.int64)
    elif case == "out_of_range_axis":
        axis.data = np.asarray([-5], dtype=np.int64)
    elif case == "multiple_axes":
        axis.data = np.asarray([1, 1], dtype=np.int64)
        axis.shape = axis.shape_signature = [2]
    elif case == "float_axis":
        axis.dtype = "FLOAT32"
        axis.data = np.asarray([1.0], dtype=np.float32)
    elif case == "unsigned_axis":
        axis.data = np.asarray([1], dtype=np.uint64)
    elif case == "dynamic_axis":
        axis.data = None
    elif case == "missing_axis":
        del model_ir.tensors["axis"]
    elif case == "missing_source_tensor":
        del model_ir.tensors["branch0_source"]
    elif case == "missing_transposed_tensor":
        del model_ir.tensors["branch0_transposed"]
    elif case == "missing_output_tensor":
        del model_ir.tensors["branch0_output"]
    elif case == "source_rank_three":
        model_ir.tensors["branch0_source"].shape = [1, 2, 3]
        model_ir.tensors["branch0_source"].shape_signature = [1, 2, -1]
    elif case == "transposed_shape_mismatch":
        model_ir.tensors["branch0_transposed"].shape = [1, 3, 2, 4]
    elif case == "transposed_signature_mismatch":
        model_ir.tensors["branch0_transposed"].shape_signature = [1, 3, 2, -1]
    elif case == "output_shape_mismatch":
        model_ir.tensors["branch0_output"].shape = [1, 3, 2]
    elif case == "output_signature_mismatch":
        model_ir.tensors["branch0_output"].shape_signature = [1, 3, -1]
    elif case == "dtype_mismatch":
        model_ir.tensors["branch0_transposed"].dtype = "FLOAT16"
    elif case == "invalid_source_signature":
        model_ir.tensors["branch0_source"].shape_signature = [1, 2, None, 4]

    before = repr(model_ir)
    stats = _optimize_transpose_pre_argmax_nhwc_terminal_chains(model_ir)

    assert stats == {"optimized_transpose_pre_argmax_nhwc_terminal_chains": 0}
    assert repr(model_ir) == before


def test_terminal_argmax_layout_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("terminal_argmax_layout_without_transpose")
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

    monkeypatch.setattr(argmax_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_transpose_pre_argmax_nhwc_terminal_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {"optimized_transpose_pre_argmax_nhwc_terminal_chains": 0}
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
