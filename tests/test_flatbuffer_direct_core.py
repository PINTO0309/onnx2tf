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
    ModelIRPassState,
    OrderedPassManager,
    PassInvariantError,
    PassPhase,
    PassSpec,
    run_model_ir_pass_group,
    summarize_model_ir_pass_diagnostics,
    validate_model_ir_invariants,
)
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.dispatcher import dispatch_node
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


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


def test_lowerer_private_sink_collects_internal_pass_diagnostics() -> None:
    diagnostics: list[dict] = []

    model_ir = lower_onnx_to_ir(
        _add_onnx_model(),
        "diagnostic_sink",
        _internal_pass_diagnostics=diagnostics,
    )

    assert model_ir.name == "diagnostic_sink"
    assert diagnostics
    assert all(event["stage"] == "model_ir_pass" for event in diagnostics)
    summary = summarize_model_ir_pass_diagnostics(diagnostics)
    assert summary["event_count"] == len(diagnostics)


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


def test_pass_precondition_skips_snapshot_callback_and_validation() -> None:
    state = {"value": 1}
    calls = {"clone": 0, "callback": 0, "validator": 0}

    def clone(value: dict) -> dict:
        calls["clone"] += 1
        return copy.deepcopy(value)

    def callback(value: dict) -> dict:
        calls["callback"] += 1
        value["value"] += 1
        return {"changed": True}

    def validator(value: dict) -> list[str]:
        calls["validator"] += 1
        return []

    manager = OrderedPassManager[dict](
        validator=validator,
        clone=clone,
        restore=lambda value, snapshot: value.update(snapshot),
    )
    manager.register(
        PassSpec(
            pass_id="guarded",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=callback,
            precondition=lambda value: value["value"] > 10,
            transactional=True,
        )
    )

    results = manager.run(state)

    assert state == {"value": 1}
    assert calls == {"clone": 0, "callback": 0, "validator": 0}
    assert results[0].iterations == 0
    assert results[0].changed is False
    assert results[0].details == {"skipped_by_precondition": True}


def test_single_iteration_pass_skips_fingerprint_work() -> None:
    state = {"value": 0}
    fingerprint_calls = 0

    def fingerprint(value: dict) -> bytes:
        nonlocal fingerprint_calls
        fingerprint_calls += 1
        return str(value["value"]).encode()

    manager = OrderedPassManager[dict](fingerprint=fingerprint)
    manager.register(
        PassSpec(
            pass_id="single_iteration",
            phase=PassPhase.CANONICALIZE,
            callback=lambda value: value.update(value=1) or {"changed": True},
        )
    )

    results = manager.run(state)

    assert state == {"value": 1}
    assert results[0].changed is True
    assert fingerprint_calls == 0


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


def test_model_ir_pass_state_restores_graph_index_and_layout_state() -> None:
    model_ir = _add_model_ir()
    state = ModelIRPassState(model_ir)
    manager = state.create_ordered_manager()

    def invalidate(pass_state: ModelIRPassState) -> dict:
        del pass_state.model_ir.tensors["z"]
        return {"changed": True}

    manager.register(
        PassSpec(
            pass_id="invalid_model_ir_mutation",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=invalidate,
            transactional=True,
        )
    )

    with pytest.raises(RuntimeError, match="missing_output_tensor"):
        manager.run(state)

    assert "z" in model_ir.tensors
    assert state.graph_index.producer("z") is model_ir.operators[0]
    assert state.layout_state is not None
    assert state.layout_state.validate_against_model_ir(model_ir) == []


def test_model_ir_pass_state_fingerprint_is_deterministic_and_caches_constants() -> None:
    model_ir = _add_model_ir()
    constant = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    model_ir.tensors["constant"] = TensorIR(
        name="constant",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=constant,
    )
    state = ModelIRPassState(model_ir)

    initial = state.fingerprint()
    repeated = state.fingerprint()

    assert initial == repeated
    assert len(state._constant_digest_cache) == 1
    assert constant.flags.writeable is False
    with pytest.raises(ValueError):
        constant[0] = 9.0

    model_ir.operators[0].options["fusedActivationFunction"] = "RELU"
    assert state.fingerprint() != initial
    model_ir.operators[0].options.clear()
    model_ir.tensors["constant"].data = np.asarray(
        [1.0, 2.0, 4.0],
        dtype=np.float32,
    )
    assert state.fingerprint() != initial
    assert len(state._constant_digest_cache) == 2


def test_model_ir_pass_group_stops_and_records_two_state_cycle() -> None:
    model_ir = _add_model_ir()
    diagnostics: list[dict] = []

    def toggle_operator(state: ModelIRPassState) -> dict:
        operator = state.model_ir.operators[0]
        operator.op_type = "MUL" if operator.op_type == "ADD" else "ADD"
        return {"changed": True}

    _, results = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="canonicalize.toggle",
                phase=PassPhase.CANONICALIZE,
                callback=toggle_operator,
                max_iterations=8,
            )
        ],
        diagnostics=diagnostics,
    )

    assert model_ir.operators[0].op_type == "ADD"
    assert results[0].iterations == 2
    assert results[0].changed is True
    assert results[0].stopped_by_cycle is True
    assert diagnostics[0] == {
        "stage": "model_ir_pass",
        "code": "canonicalize.toggle",
        "message": "model ir pass cycle_stopped",
        "phase": "canonicalize",
        "status": "cycle_stopped",
        "iterations": 2,
        "changed": True,
        "stopped_by_cycle": True,
        "skipped_by_precondition": False,
        "sequence": 1,
        "invocation": 1,
        "metrics": {
            "preflight_operators_visited": 0,
            "state_built": True,
            "snapshot_count": 0,
            "fingerprint_count": 5,
        },
    }


def test_model_ir_pass_group_runs_specs_and_normalizes_details() -> None:
    model_ir = _add_model_ir()
    diagnostics: list[dict] = []
    specs = [
        PassSpec(
            pass_id="cleanup.executed",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda state: {"changed": False, "rewritten": 2},
        ),
        PassSpec(
            pass_id="cleanup.skipped",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda state: {"changed": True, "unexpected": 1},
            precondition=lambda state: False,
            transactional=True,
        ),
    ]

    details, results = run_model_ir_pass_group(
        model_ir,
        specs=specs,
        default_details={"removed": 0},
        diagnostics=diagnostics,
    )

    assert details == {"removed": 0, "rewritten": 2}
    assert [result.pass_id for result in results] == [
        "cleanup.executed",
        "cleanup.skipped",
    ]
    assert results[1].details == {"skipped_by_precondition": True}
    assert diagnostics == [
        {
            "stage": "model_ir_pass",
            "code": "cleanup.executed",
            "message": "model ir pass unchanged",
            "phase": "post_lowering_cleanup",
            "status": "unchanged",
            "iterations": 1,
            "changed": False,
            "stopped_by_cycle": False,
            "skipped_by_precondition": False,
            "sequence": 1,
            "invocation": 1,
            "metrics": {
                "preflight_operators_visited": 0,
                "state_built": True,
                "snapshot_count": 0,
                "fingerprint_count": 0,
            },
        },
        {
            "stage": "model_ir_pass",
            "code": "cleanup.skipped",
            "message": "model ir pass skipped",
            "phase": "post_lowering_cleanup",
            "status": "skipped",
            "iterations": 0,
            "changed": False,
            "stopped_by_cycle": False,
            "skipped_by_precondition": True,
            "sequence": 2,
            "invocation": 1,
            "metrics": {
                "preflight_operators_visited": 0,
                "state_built": True,
                "snapshot_count": 0,
                "fingerprint_count": 0,
            },
        },
    ]


def test_model_ir_pass_group_records_typed_invariant_failure() -> None:
    model_ir = _add_model_ir()
    diagnostics: list[dict] = []

    def invalidate(state: ModelIRPassState) -> dict:
        del state.model_ir.tensors["z"]
        return {"changed": True}

    with pytest.raises(PassInvariantError) as caught:
        run_model_ir_pass_group(
            model_ir,
            specs=[
                PassSpec(
                    pass_id="cleanup.invalid",
                    phase=PassPhase.POST_LOWERING_CLEANUP,
                    callback=invalidate,
                    transactional=True,
                )
            ],
            diagnostics=diagnostics,
        )

    assert caught.value.pass_id == "cleanup.invalid"
    assert caught.value.phase == "post_lowering_cleanup"
    assert caught.value.iterations == 1
    assert caught.value.problems == (
        "missing_output_tensor:0:z",
        "missing_graph_output_tensor:z",
        "layout_state_stale_tensor:z",
    )
    assert "z" in model_ir.tensors
    assert diagnostics == [
        {
            "stage": "model_ir_pass",
            "code": "cleanup.invalid",
            "message": "invariant validation failed; transaction rolled back",
            "phase": "post_lowering_cleanup",
            "status": "failed",
            "iterations": 1,
            "changed": False,
            "stopped_by_cycle": False,
            "skipped_by_precondition": False,
            "problems": list(caught.value.problems),
            "sequence": 1,
            "invocation": 1,
            "metrics": {
                "preflight_operators_visited": 0,
                "state_built": True,
                "snapshot_count": 1,
                "fingerprint_count": 0,
            },
        }
    ]


def test_model_ir_pass_diagnostics_number_repeated_invocations() -> None:
    model_ir = _add_model_ir()
    diagnostics = [
        {"stage": "lowering", "code": "existing", "message": "preserved"}
    ]
    spec = PassSpec(
        pass_id="cleanup.repeated",
        phase=PassPhase.POST_LOWERING_CLEANUP,
        callback=lambda state: {"changed": False},
    )

    run_model_ir_pass_group(model_ir, specs=[spec], diagnostics=diagnostics)
    run_model_ir_pass_group(model_ir, specs=[spec], diagnostics=diagnostics)

    assert diagnostics[0] == {
        "stage": "lowering",
        "code": "existing",
        "message": "preserved",
    }
    assert diagnostics[1]["sequence"] == 1
    assert diagnostics[1]["invocation"] == 1
    assert diagnostics[2]["sequence"] == 2
    assert diagnostics[2]["invocation"] == 2

    summary = summarize_model_ir_pass_diagnostics(diagnostics)
    assert summary == {
        "schema_version": 1,
        "event_count": 2,
        "status_counts": {"unchanged": 2},
        "totals": {
            "preflight_operators_visited": 0,
            "state_backed_event_count": 2,
            "snapshot_count": 0,
            "fingerprint_count": 0,
        },
        "by_pass": {
            "cleanup.repeated": {
                "event_count": 2,
                "changed_count": 0,
                "skipped_count": 0,
                "preflight_operators_visited": 0,
                "state_backed_event_count": 2,
                "snapshot_count": 0,
                "fingerprint_count": 0,
            }
        },
    }


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
