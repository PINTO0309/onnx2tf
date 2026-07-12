from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from onnx2tf.tflite_builder.core import (
    ConversionRequest,
    ConversionResult,
    ConversionSession,
    GraphIndex,
    LayoutState,
    ModelIRGraphIndex,
    OrderedPassManager,
    PassPhase,
    PassSpec,
    validate_model_ir_invariants,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.dispatcher import dispatch_node


def _add_onnx_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "y"], ["z"], name="add")],
        "g",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])


def _add_model_ir() -> ModelIR:
    return ModelIR(
        name="add",
        tensors={
            name: TensorIR(name=name, dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
            for name in ["x", "y", "z"]
        },
        operators=[OperatorIR(op_type="ADD", inputs=["x", "y"], outputs=["z"])],
        inputs=["x", "y"],
        outputs=["z"],
    )


def test_conversion_request_normalizes_artifact_dependencies() -> None:
    request = ConversionRequest.from_kwargs(
        {
            "onnx_graph": _add_onnx_model(),
            "output_torchscript_from_model_ir": True,
        }
    )
    assert request.output_folder_path == "saved_model"
    assert request.output_file_name == "model"
    assert request.artifacts.torchscript is True
    assert request.artifacts.pytorch is True
    with pytest.raises(TypeError):
        request.options["new"] = True  # type: ignore[index]


def test_conversion_result_legacy_adapter_does_not_add_keys() -> None:
    legacy = {"float32_tflite_path": "a.tflite", "custom_op_count": 0}
    result = ConversionResult.from_legacy_dict(legacy, diagnostics={"phase": "done"})
    assert result.to_legacy_dict() == legacy


def test_conversion_session_builds_one_graph_index() -> None:
    model = _add_onnx_model()
    session = ConversionSession(
        onnx_model=model,
        model_ir=ModelIR(name="m"),
        shape_map={"x": [1, 3]},
        dtype_map={"x": "FLOAT32"},
        constants={"c": np.asarray([1.0], dtype=np.float32)},
    )
    assert session.graph_index.producer("z").name == "add"
    assert session.tensor_consumer_count == {"x": 1, "y": 1}

    session.model_ir.tensors["new"] = TensorIR(
        name="new",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        logical_layout="NHWC",
        physical_layout="NHWC",
    )
    session.refresh_indexes()
    assert session.layout_state.logical_of("new") == "NHWC"


def test_layout_state_sync_rename_remove_and_validation() -> None:
    model_ir = _add_model_ir()
    model_ir.tensors["x"].logical_layout = "NCHW"
    model_ir.tensors["x"].physical_layout = "NCHW"
    state = LayoutState.from_model_ir(model_ir)
    assert state.validate_against_model_ir(model_ir) == []

    state.set("x", logical="NHWC")
    assert state.validate_against_model_ir(model_ir) == [
        "layout_state_logical_mismatch:x"
    ]
    state.sync_from_model_ir(model_ir)
    state.rename("x", "renamed")
    assert state.logical_of("renamed") == "NCHW"
    assert state.logical_of("x") == "UNKNOWN"
    state.remove(["renamed"])
    assert "renamed" not in state.logical


def test_onnx_graph_index_updates_only_mutated_node_references() -> None:
    model = _add_onnx_model()
    node = model.graph.node[0]
    index = GraphIndex(model)
    previous_inputs = list(node.input)
    previous_outputs = list(node.output)
    node.input[1] = "x"
    node.output[0] = "w"

    index.update_node(
        node,
        previous_inputs=previous_inputs,
        previous_outputs=previous_outputs,
    )

    assert index.consumer_count("x") == 2
    assert index.consumer_count("y") == 0
    assert index.producer("z") is None
    assert index.producer("w") is node

    duplicate = helper.make_node("Identity", ["x"], ["w"], name="duplicate")
    model.graph.node.append(duplicate)
    duplicate = model.graph.node[-1]
    index.register_node(duplicate)
    assert index.producer("w") is duplicate
    assert index.duplicate_producers["w"] == [node, duplicate]

    index.unregister_node(duplicate)
    del model.graph.node[-1]
    assert index.producer("w") is node
    assert "w" not in index.duplicate_producers
    refreshed = GraphIndex(model)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers
    assert index.duplicate_producers == refreshed.duplicate_producers


def test_model_ir_index_and_invariants_detect_duplicate_producer() -> None:
    model_ir = _add_model_ir()
    model_ir.operators.append(OperatorIR(op_type="MUL", inputs=["x", "y"], outputs=["z"]))
    index = ModelIRGraphIndex(model_ir)
    assert index.duplicate_producers == {"z": [0, 1]}
    assert any(
        problem.startswith("duplicate_producer:z")
        for problem in validate_model_ir_invariants(model_ir)
    )


def test_model_ir_index_incremental_input_output_mutation_matches_refresh() -> None:
    model_ir = _add_model_ir()
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[1, 3],
    )
    index = ModelIRGraphIndex(model_ir)

    index.replace_operator_inputs(0, ["x", "x"])
    index.replace_operator_outputs(0, ["w"])

    assert index.consumer_indices("x") == [0, 0]
    assert index.consumer_indices("y") == []
    assert index.producer("z") is None
    assert index.producer("w") is model_ir.operators[0]
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers
    assert index.duplicate_producers == refreshed.duplicate_producers


def test_model_ir_index_incremental_insert_remove_shifts_references() -> None:
    model_ir = _add_model_ir()
    index = ModelIRGraphIndex(model_ir)
    identity = OperatorIR(op_type="IDENTITY", inputs=["x"], outputs=["z"])

    index.insert_operator(0, identity)

    assert index.duplicate_producers == {"z": [0, 1]}
    assert index.consumer_indices("x") == [0, 1]
    assert index.consumer_indices("y") == [1]
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers
    assert index.duplicate_producers == refreshed.duplicate_producers

    removed = index.remove_operator(0)

    assert removed is identity
    assert index.producers == {"z": 0}
    assert index.consumer_indices("x") == [0]
    assert index.consumer_indices("y") == [0]
    assert index.duplicate_producers == {}
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers


def test_model_ir_invariants_allow_empty_optional_operator_slots() -> None:
    model_ir = _add_model_ir()
    model_ir.operators[0].inputs.extend(["", "  "])
    model_ir.operators[0].outputs.append("")

    assert validate_model_ir_invariants(model_ir) == []
    index = ModelIRGraphIndex(model_ir)
    assert "" not in index.consumers
    assert "" not in index.producers


def test_ordered_pass_manager_orders_and_stops_at_fixed_point() -> None:
    state = {"value": 0, "events": []}
    manager = OrderedPassManager[dict](
        fingerprint=lambda value: str(value["value"]).encode(),
    )

    def canonicalize(value: dict) -> dict:
        value["events"].append("canonicalize")
        if value["value"] < 2:
            value["value"] += 1
            return {"changed": True}
        return {"changed": False}

    manager.register(
        PassSpec(
            pass_id="canonicalize",
            phase=PassPhase.CANONICALIZE,
            callback=canonicalize,
            max_iterations=4,
        )
    )
    manager.register(
        PassSpec(
            pass_id="normalize",
            phase=PassPhase.NORMALIZE,
            callback=lambda value: value["events"].append("normalize") or {"changed": False},
        )
    )
    results = manager.run(state)
    assert state["events"] == ["normalize", "canonicalize", "canonicalize", "canonicalize"]
    assert [result.pass_id for result in results] == ["normalize", "canonicalize"]
    assert results[1].iterations == 3


def test_transactional_pass_rolls_back_on_invariant_failure() -> None:
    state = {"values": [1]}
    manager = OrderedPassManager[dict](
        validator=lambda value: ["empty"] if not value["values"] else [],
        clone=copy.deepcopy,
        restore=lambda value, snapshot: value.update(snapshot),
    )
    manager.register(
        PassSpec(
            pass_id="bad",
            phase=PassPhase.FUSION,
            callback=lambda value: value["values"].clear() or {"changed": True},
            transactional=True,
        )
    )
    with pytest.raises(RuntimeError, match="pass invariant violation"):
        manager.run(state)
    assert state == {"values": [1]}


def test_dispatcher_records_onnx_provenance() -> None:
    class _Entry:
        @staticmethod
        def builder(node, context):
            context.model_ir.operators.append(
                OperatorIR(op_type="ADD", inputs=["x", "y"], outputs=["z"])
            )

    class _Registry:
        @staticmethod
        def lower(node, context):
            _Entry.builder(node, context)

    import onnx2tf.tflite_builder.dispatcher as dispatcher

    original = dispatcher._LOWERING_REGISTRY
    dispatcher._LOWERING_REGISTRY = _Registry()
    try:
        model_ir = _add_model_ir()
        model_ir.operators.clear()
        context = type("Context", (), {"model_ir": model_ir})()
        node = type(
            "Node",
            (),
            {
                "name": "onnx_add",
                "op": "Add",
                "outputs": [type("Output", (), {"name": "z"})()],
            },
        )()
        dispatch_node(node, context)
    finally:
        dispatcher._LOWERING_REGISTRY = original
    assert model_ir.operators[0].onnx_node_name == "onnx_add"
    assert model_ir.operators[0].onnx_op_type == "Add"
    assert model_ir.tensors["z"].onnx_tensor_name == "z"
