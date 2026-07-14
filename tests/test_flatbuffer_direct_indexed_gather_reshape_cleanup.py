from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np
import pytest

import onnx2tf.tflite_builder.passes.gather_reshape_cleanup as cleanup_module
from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.passes.gather_reshape_cleanup import (
    _optimize_gather_axis0_singleton_to_reshape_input_chains,
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


def _gather_reshape_model(*, branches: int = 2) -> ModelIR:
    model_ir = ModelIR("indexed_gather_reshape_cleanup")
    for branch_index in range(int(branches)):
        prefix = f"branch{branch_index}"
        source = f"{prefix}_source"
        indices = f"{prefix}_indices"
        gathered = f"{prefix}_gathered"
        shape = f"{prefix}_shape"
        output = f"{prefix}_output"
        model_ir.inputs.append(source)
        model_ir.outputs.append(output)
        model_ir.tensors.update(
            {
                source: _tensor(
                    source,
                    shape=[1, 2, 3],
                    signature=[1, 2, -1],
                ),
                indices: _tensor(
                    indices,
                    dtype="INT64",
                    shape=[1],
                    data=np.asarray([0], dtype=np.int64),
                ),
                gathered: _tensor(
                    gathered,
                    shape=[2, 3],
                    signature=[2, -1],
                ),
                shape: _tensor(
                    shape,
                    dtype="INT32",
                    shape=[1],
                    data=np.asarray([6], dtype=np.int32),
                ),
                output: _tensor(
                    output,
                    shape=[6],
                    signature=[-1],
                ),
            }
        )
        axis = 0 if branch_index % 2 == 0 else -3
        model_ir.operators.extend(
            [
                OperatorIR(
                    "GATHER",
                    [source, indices],
                    [gathered],
                    options={"axis": axis, "batchDims": 0},
                ),
                OperatorIR(
                    "RESHAPE",
                    [gathered, shape],
                    [output],
                    options={"newShape": [6]},
                ),
            ]
        )
    return model_ir


def _normalized(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return (
            "ndarray",
            str(value.dtype),
            tuple(int(dim) for dim in value.shape),
            tuple(value.reshape(-1).tolist()),
        )
    if isinstance(value, np.generic):
        return value.item()
    if is_dataclass(value):
        return (
            value.__class__.__name__,
            tuple(
                (field.name, _normalized(getattr(value, field.name)))
                for field in fields(value)
            ),
        )
    if isinstance(value, dict):
        return tuple(
            (str(key), _normalized(item))
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        )
    if isinstance(value, (list, tuple)):
        return tuple(_normalized(item) for item in value)
    return value


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


def test_gather_reshape_cleanup_rewrites_multiple_matches_with_one_index(
    monkeypatch,
) -> None:
    model_ir = _gather_reshape_model()
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_gather_axis0_singleton_to_reshape_input_chains(model_ir)

    assert stats == {
        "optimized_gather_axis0_singleton_to_reshape_input_chains": 2,
    }
    assert refresh_count == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RESHAPE",
        "RESHAPE",
    ]
    assert [operator.inputs for operator in model_ir.operators] == [
        ["branch0_source", "branch0_shape"],
        ["branch1_source", "branch1_shape"],
    ]
    assert set(model_ir.tensors) == {
        "branch0_source",
        "branch0_shape",
        "branch0_output",
        "branch1_source",
        "branch1_shape",
        "branch1_output",
    }


def test_gather_reshape_cleanup_reaches_nested_fixed_point_with_one_index(
    monkeypatch,
) -> None:
    model_ir = ModelIR("indexed_nested_gather_reshape_cleanup")
    model_ir.inputs = ["source"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "source": _tensor(
            "source",
            shape=[1, 1, 2, 3],
            signature=[1, 1, 2, -1],
        ),
        "indices0": _tensor(
            "indices0",
            dtype="INT64",
            shape=[1],
            data=np.asarray([0], dtype=np.int64),
        ),
        "gathered0": _tensor(
            "gathered0",
            shape=[1, 2, 3],
            signature=[1, 2, -1],
        ),
        "indices1": _tensor(
            "indices1",
            dtype="INT64",
            shape=[1],
            data=np.asarray([0], dtype=np.int64),
        ),
        "gathered1": _tensor(
            "gathered1",
            shape=[2, 3],
            signature=[2, -1],
        ),
        "shape": _tensor(
            "shape",
            dtype="INT32",
            shape=[1],
            data=np.asarray([6], dtype=np.int32),
        ),
        "output": _tensor("output", shape=[6], signature=[-1]),
    }
    model_ir.operators = [
        OperatorIR(
            "GATHER",
            ["source", "indices0"],
            ["gathered0"],
            options={"axis": 0},
        ),
        OperatorIR(
            "GATHER",
            ["gathered0", "indices1"],
            ["gathered1"],
            options={"axis": 0},
        ),
        OperatorIR(
            "RESHAPE",
            ["gathered1", "shape"],
            ["output"],
        ),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    stats = _optimize_gather_axis0_singleton_to_reshape_input_chains(model_ir)

    assert stats == {
        "optimized_gather_axis0_singleton_to_reshape_input_chains": 2,
    }
    assert refresh_count == 1
    assert len(model_ir.operators) == 1
    assert model_ir.operators[0].op_type == "RESHAPE"
    assert model_ir.operators[0].inputs == ["source", "shape"]
    assert set(model_ir.tensors) == {"source", "shape", "output"}


def test_gather_reshape_cleanup_keeps_supplied_index_and_layout_current() -> None:
    model_ir = _gather_reshape_model(branches=1)
    graph_index = ModelIRGraphIndex(model_ir)
    layout_state = LayoutState.from_model_ir(model_ir)

    stats = _optimize_gather_axis0_singleton_to_reshape_input_chains(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_gather_axis0_singleton_to_reshape_input_chains": 1,
    }
    _assert_index_current(model_ir, graph_index)
    assert layout_state.validate_against_model_ir(model_ir) == []
    assert set(layout_state.logical) == {
        "branch0_source",
        "branch0_shape",
        "branch0_output",
    }


def test_gather_reshape_cleanup_accepts_matching_quantized_buffers() -> None:
    model_ir = _gather_reshape_model(branches=1)
    source = model_ir.tensors["branch0_source"]
    gathered = model_ir.tensors["branch0_gathered"]
    output = model_ir.tensors["branch0_output"]
    source.dtype = gathered.dtype = output.dtype = "INT8"
    source.quantization = QuantParamIR(
        scale=[0.125],
        zero_point=[-3],
        quantized_dimension=0,
    )
    gathered.quantization = QuantParamIR(
        scale=[0.125],
        zero_point=[-3],
        quantized_dimension=0,
    )

    stats = _optimize_gather_axis0_singleton_to_reshape_input_chains(model_ir)

    assert stats == {
        "optimized_gather_axis0_singleton_to_reshape_input_chains": 1,
    }
    assert model_ir.operators[0].inputs[0] == "branch0_source"


@pytest.mark.parametrize(
    "case",
    [
        "public_output",
        "public_input",
        "fanout",
        "duplicate_producer",
        "reversed_operator_order",
        "wrong_consumer",
        "reshape_data_input_position",
        "extra_gather_input",
        "extra_gather_output",
        "axis_nonzero",
        "axis_out_of_range",
        "batch_dims_nonzero",
        "snake_batch_dims_nonzero",
        "conflicting_batch_dims",
        "leading_dimension_not_one",
        "dynamic_leading_signature",
        "unknown_input_shape",
        "wrong_output_rank",
        "wrong_output_shape",
        "wrong_output_signature",
        "invalid_signature_value",
        "missing_input_tensor",
        "missing_output_tensor",
        "missing_indices_tensor",
        "dynamic_indices",
        "empty_indices",
        "multiple_zero_indices",
        "nonzero_index",
        "float_index",
        "unsigned_index",
        "dtype_mismatch",
        "quantization_mismatch",
    ],
)
def test_gather_reshape_cleanup_rejects_unsafe_contracts_transactionally(
    case: str,
) -> None:
    model_ir = _gather_reshape_model(branches=1)
    gather = model_ir.operators[0]
    reshape = model_ir.operators[1]
    source = model_ir.tensors["branch0_source"]
    indices = model_ir.tensors["branch0_indices"]
    gathered = model_ir.tensors["branch0_gathered"]

    if case == "public_output":
        model_ir.outputs.append("branch0_gathered")
    elif case == "public_input":
        model_ir.inputs.append("branch0_gathered")
    elif case == "fanout":
        model_ir.tensors["side"] = _tensor("side", shape=[2, 3])
        model_ir.outputs.append("side")
        model_ir.operators.append(
            OperatorIR("IDENTITY", ["branch0_gathered"], ["side"])
        )
    elif case == "duplicate_producer":
        model_ir.operators.append(
            OperatorIR(
                "IDENTITY",
                ["branch0_source"],
                ["branch0_gathered"],
            )
        )
    elif case == "reversed_operator_order":
        model_ir.operators = [reshape, gather]
    elif case == "wrong_consumer":
        reshape.op_type = "IDENTITY"
    elif case == "reshape_data_input_position":
        reshape.inputs = ["branch0_shape", "branch0_gathered"]
    elif case == "extra_gather_input":
        gather.inputs.append("branch0_shape")
    elif case == "extra_gather_output":
        model_ir.tensors["extra"] = _tensor("extra", shape=[2, 3])
        gather.outputs.append("extra")
    elif case == "axis_nonzero":
        gather.options["axis"] = 1
    elif case == "axis_out_of_range":
        gather.options["axis"] = -4
    elif case == "batch_dims_nonzero":
        gather.options["batchDims"] = 1
    elif case == "snake_batch_dims_nonzero":
        gather.options.pop("batchDims")
        gather.options["batch_dims"] = 1
    elif case == "conflicting_batch_dims":
        gather.options["batchDims"] = 0
        gather.options["batch_dims"] = 1
    elif case == "leading_dimension_not_one":
        source.shape[0] = source.shape_signature[0] = 2
    elif case == "dynamic_leading_signature":
        source.shape_signature[0] = -1
    elif case == "unknown_input_shape":
        source.shape[1] = -1
    elif case == "wrong_output_rank":
        gathered.shape = [1, 2, 3]
        gathered.shape_signature = [1, 2, -1]
    elif case == "wrong_output_shape":
        gathered.shape = [3, 2]
        gathered.shape_signature = [3, -1]
    elif case == "wrong_output_signature":
        gathered.shape_signature = [3, -1]
    elif case == "invalid_signature_value":
        gathered.shape_signature = [2, None]
    elif case == "missing_input_tensor":
        del model_ir.tensors["branch0_source"]
    elif case == "missing_output_tensor":
        del model_ir.tensors["branch0_gathered"]
    elif case == "missing_indices_tensor":
        del model_ir.tensors["branch0_indices"]
    elif case == "dynamic_indices":
        indices.data = None
    elif case == "empty_indices":
        indices.data = np.asarray([], dtype=np.int64)
        indices.shape = indices.shape_signature = [0]
    elif case == "multiple_zero_indices":
        indices.data = np.asarray([0, 0], dtype=np.int64)
        indices.shape = indices.shape_signature = [2]
    elif case == "nonzero_index":
        indices.data = np.asarray([1], dtype=np.int64)
    elif case == "float_index":
        indices.dtype = "FLOAT32"
        indices.data = np.asarray([0.0], dtype=np.float32)
    elif case == "unsigned_index":
        indices.dtype = "INT64"
        indices.data = np.asarray([0], dtype=np.uint64)
    elif case == "dtype_mismatch":
        gathered.dtype = "INT8"
    elif case == "quantization_mismatch":
        source.dtype = gathered.dtype = "INT8"
        source.quantization = QuantParamIR([0.25], [0])
        gathered.quantization = QuantParamIR([0.5], [0])

    before = _normalized(model_ir)
    stats = _optimize_gather_axis0_singleton_to_reshape_input_chains(model_ir)

    assert stats == {
        "optimized_gather_axis0_singleton_to_reshape_input_chains": 0,
    }
    assert _normalized(model_ir) == before


def test_gather_reshape_cleanup_skips_index_without_complete_family_and_prunes(
    monkeypatch,
) -> None:
    model_ir = ModelIR("gather_reshape_cleanup_without_gather")
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

    monkeypatch.setattr(cleanup_module, "ModelIRGraphIndex", unexpected_index)

    stats = _optimize_gather_axis0_singleton_to_reshape_input_chains(
        model_ir,
        layout_state=layout_state,
    )

    assert stats == {
        "optimized_gather_axis0_singleton_to_reshape_input_chains": 0,
    }
    assert set(model_ir.tensors) == {"x", "y"}
    assert layout_state.validate_against_model_ir(model_ir) == []
