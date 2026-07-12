from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_utils import (
    _broadcast_shape_signatures,
    _broadcast_static_shapes,
    _build_tensor_consumer_map,
    _is_fully_known_positive_shape,
    _invert_perm,
    _is_singleton_constant_tensor,
    _normalize_squeeze_axes_for_rank,
    _prune_unused_tensors,
    _read_transpose_perm,
    _read_singleton_constant_float,
    _read_const_ints_from_tensor,
    _replace_tensor_inputs,
    _rename_tensor_globally,
    _replace_operator_input_at,
    _set_operator_inputs,
    _set_operator_outputs,
    _write_const_ints_to_tensor,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR


def _tensor(name: str) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
    )


def test_prune_unused_tensors_records_deterministic_lineage() -> None:
    model_ir = ModelIR("prune_unused_tensor_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "y": _tensor("y"),
        "unused": _tensor("unused"),
    }

    layout_state = LayoutState.from_model_ir(model_ir)
    _prune_unused_tensors(model_ir, layout_state=layout_state)

    assert set(model_ir.tensors) == {"x", "y"}
    assert "unused" not in layout_state.logical
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "prune_unused_tensors",
            "removed_names": ["unused"],
            "event_index": 0,
        }
    ]


def test_fully_known_positive_shape_rejects_dynamic_or_empty_shapes() -> None:
    assert _is_fully_known_positive_shape([1, 2, 3]) is True
    assert _is_fully_known_positive_shape([1, -1, 3]) is False
    assert _is_fully_known_positive_shape([1, 0, 3]) is False
    assert _is_fully_known_positive_shape([]) is False
    assert _is_fully_known_positive_shape(None) is False


def test_invert_perm_rejects_invalid_permutations() -> None:
    assert _invert_perm([0, 3, 1, 2]) == [0, 2, 3, 1]
    assert _invert_perm([0, 0, 1]) is None
    assert _invert_perm([0, 1, 3]) is None
    assert _normalize_squeeze_axes_for_rank([0, -1, 0], 4) == [0, 3]
    assert _normalize_squeeze_axes_for_rank([4], 4) is None


def test_singleton_constant_requires_one_materialized_value() -> None:
    model_ir = ModelIR("singleton_constant_test")
    model_ir.tensors = {
        "scalar": TensorIR(
            name="scalar",
            dtype="FLOAT32",
            shape=[],
            shape_signature=[],
            data=np.asarray(1.0, dtype=np.float32),
            is_variable=False,
        ),
        "vector": TensorIR(
            name="vector",
            dtype="FLOAT32",
            shape=[2],
            shape_signature=[2],
            data=np.asarray([1.0, 2.0], dtype=np.float32),
            is_variable=False,
        ),
        "runtime": _tensor("runtime"),
    }

    assert _is_singleton_constant_tensor(model_ir, "scalar") is True
    assert _read_singleton_constant_float(model_ir, "scalar") == 1.0
    assert _is_singleton_constant_tensor(model_ir, "vector") is False
    assert _read_singleton_constant_float(model_ir, "vector") is None
    assert _is_singleton_constant_tensor(model_ir, "runtime") is False
    assert _is_singleton_constant_tensor(model_ir, "missing") is False


def test_graph_helpers_read_transpose_and_record_input_replacement() -> None:
    model_ir = ModelIR("graph_helper_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x"),
        "x_internal": _tensor("x_internal"),
        "perm": TensorIR(
            name="perm",
            dtype="INT32",
            shape=[3],
            shape_signature=[3],
            data=np.asarray([0, 2, 1], dtype=np.int32),
            is_variable=False,
        ),
        "y": _tensor("y"),
    }
    transpose = OperatorIR(
        op_type="TRANSPOSE",
        inputs=["x", "perm"],
        outputs=["x_internal"],
    )
    consumer = OperatorIR(
        op_type="ABS",
        inputs=["x_internal"],
        outputs=["y"],
    )
    model_ir.operators = [transpose, consumer]

    assert _read_transpose_perm(model_ir, transpose) == [0, 2, 1]
    assert _build_tensor_consumer_map(model_ir) == {
        "x": [0],
        "perm": [0],
        "x_internal": [1],
    }

    _replace_tensor_inputs(model_ir, "x_internal", "x")

    assert consumer.inputs == ["x"]
    assert model_ir.metadata["tensor_lineage_events"] == [
        {
            "kind": "replace_input",
            "src_name": "x_internal",
            "dst_name": "x",
            "event_index": 0,
        }
    ]


def test_graph_mutation_helpers_update_optional_incremental_index() -> None:
    model_ir = ModelIR("indexed_graph_mutation_test")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors = {
        name: _tensor(name)
        for name in ["x", "y", "z", "w"]
    }
    op = OperatorIR(op_type="ADD", inputs=["x", "y"], outputs=["z"])
    model_ir.operators = [op]
    graph_index = ModelIRGraphIndex(model_ir)

    _replace_operator_input_at(
        model_ir=model_ir,
        op=op,
        input_index=1,
        new_input_name="x",
        graph_index=graph_index,
    )
    assert graph_index.consumer_indices("x") == [0, 0]
    assert graph_index.consumer_indices("y") == []

    _set_operator_outputs(
        model_ir=model_ir,
        op=op,
        new_outputs=["w"],
        graph_index=graph_index,
    )
    assert graph_index.producer("z") is None
    assert graph_index.producer("w") is op

    _set_operator_inputs(
        model_ir=model_ir,
        op=op,
        new_inputs=["y", "x"],
        graph_index=graph_index,
    )
    assert graph_index.consumer_indices("x") == [0]
    assert graph_index.consumer_indices("y") == [0]
    _replace_tensor_inputs(
        model_ir,
        "y",
        "x",
        graph_index=graph_index,
    )
    assert graph_index.consumer_indices("x") == [0, 0]
    assert graph_index.consumer_indices("y") == []
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers


def test_global_tensor_rename_updates_optional_layout_state() -> None:
    model_ir = ModelIR("layout_aware_rename")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR(
            name="x",
            dtype="FLOAT32",
            shape=[1, 3, 2, 2],
            shape_signature=[1, 3, 2, 2],
            logical_layout="NCHW",
            physical_layout="NCHW",
        ),
        "y": _tensor("y"),
    }
    model_ir.operators = [
        OperatorIR(op_type="IDENTITY", inputs=["x"], outputs=["y"]),
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    graph_index = ModelIRGraphIndex(model_ir)

    _rename_tensor_globally(
        model_ir,
        "x",
        "renamed",
        layout_state=layout_state,
        graph_index=graph_index,
    )

    assert model_ir.inputs == ["renamed"]
    assert model_ir.operators[0].inputs == ["renamed"]
    assert layout_state.logical_of("renamed") == "NCHW"
    assert "x" not in layout_state.logical
    assert layout_state.validate_against_model_ir(model_ir) == []
    refreshed = ModelIRGraphIndex(model_ir)
    assert graph_index.producers == refreshed.producers
    assert graph_index.consumers == refreshed.consumers


def test_static_shape_and_constant_vector_helpers_are_deterministic() -> None:
    tensor = TensorIR(
        name="axes",
        dtype="INT64",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 3], dtype=np.int64),
        is_variable=False,
    )

    assert _broadcast_static_shapes([2, 1, 4], [1, 3, 4]) == [2, 3, 4]
    assert _broadcast_static_shapes([2, 3], [4, 3]) is None
    assert _broadcast_shape_signatures([-1, 1, 4], [1, 3, 4]) == [-1, 3, 4]
    assert _broadcast_shape_signatures([-1, 2], [3, 2]) == [3, 2]
    assert _broadcast_shape_signatures([2, 3], [4, 3]) is None
    assert _broadcast_shape_signatures(None, [1]) is None
    assert _read_const_ints_from_tensor(tensor) == [1, 3]
    assert _write_const_ints_to_tensor(tensor, [0, 2, 3]) is True
    assert _read_const_ints_from_tensor(tensor) == [0, 2, 3]
    assert tensor.data is not None and tensor.data.dtype == np.int64
    assert tensor.shape == [3]
    assert tensor.shape_signature == [3]
    assert _write_const_ints_to_tensor(tensor, [0, 2, 3]) is False
