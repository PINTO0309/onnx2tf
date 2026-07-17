from __future__ import annotations

import copy

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
import onnx2tf.tflite_builder.core.validation as validation_module
import onnx2tf.tflite_builder.lower_from_onnx2tf as lowering_module

from onnx2tf.tflite_builder.core import (
    ConversionRequest,
    ConversionResult,
    ConversionSession,
    GraphIndex,
    LayoutState,
    LoweringContext,
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
from onnx2tf.tflite_builder.core.pass_diagnostics import ModelIRPassDiagnostics
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.dispatcher import dispatch_node
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.op_builders.shared import make_transpose


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


def _rank3_resize_onnx_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="roi")
    scales = numpy_helper.from_array(
        np.asarray([1.0, 1.0, 2.0], dtype=np.float32),
        name="scales",
    )
    resize = helper.make_node(
        "Resize",
        ["x", "roi", "scales"],
        ["y"],
        name="rank3_resize",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph(
        [resize],
        "rank3_resize_graph",
        [x],
        [y],
        initializer=[roi, scales],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 13)],
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


def test_validation_pipeline_reuses_current_caller_index(monkeypatch) -> None:
    model_ir = _add_model_ir()
    graph_index = ModelIRGraphIndex(model_ir)

    class _UnexpectedGraphIndex:
        def __init__(self, _model_ir):
            raise AssertionError("validation rebuilt a caller-owned graph index")

    monkeypatch.setattr(
        validation_module,
        "ModelIRGraphIndex",
        _UnexpectedGraphIndex,
    )

    validation_module.run_model_ir_validation_pipeline(
        model_ir,
        graph_index=graph_index,
    )


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


def test_lowerer_context_reuses_session_consumer_counts(monkeypatch) -> None:
    captured_counts: dict[str, int] = {}
    original_context = lowering_module.LoweringContext
    model = _add_onnx_model()
    model.graph.node.append(
        helper.make_node("Identity", ["x"], ["side"], name="side")
    )
    model.graph.output.append(
        helper.make_tensor_value_info("side", TensorProto.FLOAT, [1, 3])
    )

    class _TrackingLoweringContext(original_context):
        def __init__(self, *args, **kwargs):
            captured_counts.update(kwargs["tensor_consumer_count"])
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(
        lowering_module,
        "LoweringContext",
        _TrackingLoweringContext,
    )

    lower_onnx_to_ir(model, "session_consumer_counts")

    assert captured_counts == {"x": 2, "y": 1}


def test_lowerer_keeps_session_layout_current_before_first_post_pass(
    monkeypatch,
) -> None:
    observed_problems: list[list[str]] = []
    observed_layouts: dict[str, str] = {}
    original_cleanup = lowering_module.run_layout_transpose_cleanup

    def _tracking_cleanup(
        model_ir,
        *,
        layout_state=None,
        diagnostics=None,
        state_scope=None,
    ):
        assert layout_state is not None
        if not observed_problems:
            observed_problems.append(layout_state.validate_against_model_ir(model_ir))
            observed_layouts.update(
                {
                    name: layout_state.logical_of(name)
                    for name in (
                        "rank3_resize_input_nwc",
                        "rank3_resize_input_nhwc",
                        "rank3_resize_output_nhwc",
                        "rank3_resize_output_nwc",
                    )
                }
            )
        return original_cleanup(
            model_ir,
            layout_state=layout_state,
            diagnostics=diagnostics,
            state_scope=state_scope,
        )

    monkeypatch.setattr(
        lowering_module,
        "run_layout_transpose_cleanup",
        _tracking_cleanup,
    )

    lower_onnx_to_ir(_rank3_resize_onnx_model(), "session_layout_handoff")

    assert observed_problems == [[]]
    assert observed_layouts == {
        "rank3_resize_input_nwc": "NWC",
        "rank3_resize_input_nhwc": "NHWC",
        "rank3_resize_output_nhwc": "NHWC",
        "rank3_resize_output_nwc": "NWC",
    }


def _inverse_transpose_lowering_context() -> LoweringContext:
    model_ir = ModelIR(
        name="inverse_transpose_context",
        tensors={
            "source": TensorIR(
                "source",
                "FLOAT32",
                [1, 3, 2, 2],
                [1, 3, 2, 2],
            ),
            "bridge": TensorIR(
                "bridge",
                "FLOAT32",
                [1, 2, 2, 3],
                [1, 2, 2, 3],
            ),
            "side": TensorIR(
                "side",
                "FLOAT32",
                [1, 2, 2, 3],
                [1, 2, 2, 3],
            ),
            "restored": TensorIR(
                "restored",
                "FLOAT32",
                [1, 3, 2, 2],
                [1, 3, 2, 2],
            ),
        },
    )
    context = LoweringContext(
        model_ir=model_ir,
        shape_map={},
        dtype_map={},
        constants={},
        tensor_consumer_count={},
    )
    perm_name = context.add_const_tensor(
        "to_nhwc_perm",
        np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    context.add_operator(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["source", perm_name],
            outputs=["bridge"],
        )
    )
    return context


def test_inverse_transpose_elision_removes_lowering_indexes() -> None:
    context = _inverse_transpose_lowering_context()

    result = make_transpose(
        context,
        "bridge",
        "restored",
        [0, 3, 1, 2],
        allow_elide_inverse_chain=True,
    )

    assert result == "source"
    assert context.model_ir.operators == []
    assert "bridge" not in context.ir_tensor_producers
    assert context.ir_tensor_consumer_count == {}


def test_inverse_transpose_elision_preserves_synthetic_fanout() -> None:
    context = _inverse_transpose_lowering_context()
    context.add_operator(
        OperatorIR(
            op_type="IDENTITY",
            inputs=["bridge"],
            outputs=["side"],
        )
    )

    result = make_transpose(
        context,
        "bridge",
        "restored",
        [0, 3, 1, 2],
        allow_elide_inverse_chain=True,
    )

    assert result == "source"
    assert [str(op.op_type) for op in context.model_ir.operators] == [
        "TRANSPOSE",
        "IDENTITY",
    ]
    assert context.ir_tensor_producers["bridge"] is context.model_ir.operators[0]
    assert context.ir_tensor_consumer_count["bridge"] == 1


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

    assert index.operator_indices("ADD") == [0]
    assert index.operator_indices("MUL") == []
    assert index.operator_indices_for_types(["MUL", "ADD", "ADD"]) == [0]
    assert index.operator_indices_for_normalized_types(["mul", "add"]) == [0]
    assert index.consumer_indices("x") == [0, 0]
    assert index.consumer_indices("y") == []
    assert index.producer("z") is None
    assert index.producer("w") is model_ir.operators[0]
    index.replace_operator_type(0, "DIV")
    assert index.operator_indices("ADD") == []
    assert index.operator_indices("DIV") == [0]
    assert index.operator_indices_for_types({"ADD", "DIV"}) == [0]
    assert index.operator_indices_for_normalized_types({"add", "div"}) == [0]
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers
    assert index.duplicate_producers == refreshed.duplicate_producers


def test_model_ir_index_incremental_insert_remove_shifts_references() -> None:
    model_ir = _add_model_ir()
    index = ModelIRGraphIndex(model_ir)
    identity = OperatorIR(op_type="IDENTITY", inputs=["x"], outputs=["z"])

    index.insert_operator(0, identity)

    assert index.operator_indices("IDENTITY") == [0]
    assert index.operator_indices("ADD") == [1]
    assert index.operator_indices_for_types({"ADD", "IDENTITY"}) == [0, 1]
    assert index.duplicate_producers == {"z": [0, 1]}
    assert index.consumer_indices("x") == [0, 1]
    assert index.consumer_indices("y") == [1]
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers
    assert index.duplicate_producers == refreshed.duplicate_producers

    removed = index.remove_operator(0)

    assert removed is identity
    assert index.operator_indices("IDENTITY") == []
    assert index.operator_indices("ADD") == [0]
    assert index.operator_indices_for_types({"ADD", "IDENTITY"}) == [0]
    assert index.producers == {"z": 0}
    assert index.consumer_indices("x") == [0]
    assert index.consumer_indices("y") == [0]
    assert index.duplicate_producers == {}
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers


def test_model_ir_index_batch_remove_compacts_references_once() -> None:
    model_ir = ModelIR(name="batch_remove")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["out"]
    model_ir.operators = [
        OperatorIR("ADD", ["x", "y"], ["a"]),
        OperatorIR("MUL", ["a", "y"], ["b"]),
        OperatorIR("IDENTITY", ["x"], ["dead0"]),
        OperatorIR("SUB", ["b", "y"], ["out"]),
        OperatorIR("IDENTITY", ["y"], ["dead1"]),
    ]
    dead0 = model_ir.operators[2]
    dead1 = model_ir.operators[4]
    index = ModelIRGraphIndex(model_ir)

    removed = index.remove_operators([4, 2, 4])

    assert removed == [dead0, dead1]
    assert [op.op_type for op in model_ir.operators] == ["ADD", "MUL", "SUB"]
    assert index.operator_indices("IDENTITY") == []
    assert index.operator_indices_for_types({"ADD", "MUL", "SUB"}) == [0, 1, 2]
    refreshed = ModelIRGraphIndex(model_ir)
    assert index.producers == refreshed.producers
    assert index.consumers == refreshed.consumers
    assert index.duplicate_producers == refreshed.duplicate_producers
    assert index._operator_indices_by_id == refreshed._operator_indices_by_id
    assert index._operator_indices_by_type == refreshed._operator_indices_by_type


def test_model_ir_pass_state_prepared_data_is_session_local_and_rollback_safe() -> None:
    first = ModelIRPassState(_add_model_ir())
    second = ModelIRPassState(_add_model_ir())

    first.set_prepared_pass_data("candidate", {"index": 0})
    assert second.take_prepared_pass_data("candidate") is None
    assert first.take_prepared_pass_data("candidate") == {"index": 0}
    assert first.take_prepared_pass_data("candidate") is None

    snapshot = first.snapshot()
    first.set_prepared_pass_data("candidate", {"index": 1})
    first.restore(snapshot)
    assert first.take_prepared_pass_data("candidate") is None


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


def test_transactional_pass_allows_unrelated_preexisting_invariant_problem() -> None:
    state = {"problems": ["preexisting"], "value": 0}
    manager = OrderedPassManager[dict](
        validator=lambda value: list(value["problems"]),
        clone=copy.deepcopy,
        restore=lambda value, snapshot: value.update(snapshot),
    )
    manager.register(
        PassSpec(
            pass_id="unrelated",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda value: value.update(value=1) or {"changed": True},
            transactional=True,
        )
    )

    results = manager.run(state)

    assert state == {"problems": ["preexisting"], "value": 1}
    assert results[0].changed is True


def test_transactional_pass_rolls_back_only_new_invariant_problems() -> None:
    state = {"problems": ["preexisting"], "value": 0}
    manager = OrderedPassManager[dict](
        validator=lambda value: list(value["problems"]),
        clone=copy.deepcopy,
        restore=lambda value, snapshot: value.clear() or value.update(snapshot),
    )

    def introduce_problem(value: dict) -> dict:
        value["problems"].append("introduced")
        value["value"] = 1
        return {"changed": True}

    manager.register(
        PassSpec(
            pass_id="introduce",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=introduce_problem,
            transactional=True,
        )
    )

    with pytest.raises(PassInvariantError) as caught:
        manager.run(state)

    assert caught.value.problems == ("introduced",)
    assert state == {"problems": ["preexisting"], "value": 0}


def test_transactional_pass_detects_problem_reintroduced_after_repair() -> None:
    state = {"problems": ["rank_mismatch"]}
    manager = OrderedPassManager[dict](
        validator=lambda value: list(value["problems"]),
        clone=copy.deepcopy,
        restore=lambda value, snapshot: value.clear() or value.update(snapshot),
    )
    manager.register(
        PassSpec(
            pass_id="repair",
            phase=PassPhase.CANONICALIZE,
            callback=lambda value: value["problems"].clear() or {"changed": True},
            transactional=True,
        )
    )
    manager.register(
        PassSpec(
            pass_id="reintroduce",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda value: value["problems"].append("rank_mismatch")
            or {"changed": True},
            transactional=True,
        )
    )

    with pytest.raises(PassInvariantError) as caught:
        manager.run(state)

    assert caught.value.pass_id == "reintroduce"
    assert caught.value.problems == ("rank_mismatch",)
    assert state == {"problems": []}


def test_nontransactional_validation_rejects_preexisting_problem() -> None:
    state = {"problems": ["preexisting"]}
    manager = OrderedPassManager[dict](
        validator=lambda value: list(value["problems"]),
    )
    manager.register(
        PassSpec(
            pass_id="final_validation",
            phase=PassPhase.VALIDATE,
            callback=lambda value: {"changed": False},
        )
    )

    with pytest.raises(PassInvariantError) as caught:
        manager.run(state)

    assert caught.value.problems == ("preexisting",)


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
        "group_sequence": 1,
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
            "group_sequence": 1,
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
            "group_sequence": 1,
            "metrics": {
                "preflight_operators_visited": 0,
                "state_built": True,
                "snapshot_count": 0,
                "fingerprint_count": 0,
            },
        },
    ]
    summary = summarize_model_ir_pass_diagnostics(diagnostics)
    assert summary["totals"] == {
        "preflight_operators_visited": 0,
        "state_build_count": 1,
        "snapshot_count": 0,
        "fingerprint_count": 0,
    }
    assert summary["groups"] == {
        "1": {
            "pass_ids": ["cleanup.executed", "cleanup.skipped"],
            "preflight_operators_visited": 0,
            "state_built": True,
        }
    }


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
            "group_sequence": 1,
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
    assert diagnostics[1]["group_sequence"] == 1
    assert diagnostics[2]["group_sequence"] == 2

    summary = summarize_model_ir_pass_diagnostics(diagnostics)
    assert summary == {
        "schema_version": 2,
        "event_count": 2,
        "status_counts": {"unchanged": 2},
        "totals": {
            "preflight_operators_visited": 0,
            "state_build_count": 2,
            "snapshot_count": 0,
            "fingerprint_count": 0,
        },
        "by_pass": {
            "cleanup.repeated": {
                "event_count": 2,
                "changed_count": 0,
                "skipped_count": 0,
                "snapshot_count": 0,
                "fingerprint_count": 0,
            }
        },
        "groups": {
            "1": {
                "pass_ids": ["cleanup.repeated"],
                "preflight_operators_visited": 0,
                "state_built": True,
            },
            "2": {
                "pass_ids": ["cleanup.repeated"],
                "preflight_operators_visited": 0,
                "state_built": True,
            },
        },
    }


def test_model_ir_pass_diagnostics_preserve_existing_numbering_contract() -> None:
    model_ir = _add_model_ir()
    diagnostics = [
        {"stage": "lowering", "code": "unrelated"},
        {
            "stage": "model_ir_pass",
            "code": "cleanup.first",
            "group_sequence": 4,
        },
        {
            "stage": "model_ir_pass",
            "code": "cleanup.other",
            "group_sequence": 7,
        },
    ]
    specs = [
        PassSpec(
            pass_id="cleanup.first",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda state: {"changed": False},
        ),
        PassSpec(
            pass_id="cleanup.second",
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda state: {"changed": False},
        ),
    ]

    run_model_ir_pass_group(model_ir, specs=specs, diagnostics=diagnostics)

    first, second = diagnostics[-2:]
    assert (first["sequence"], first["invocation"], first["group_sequence"]) == (
        3,
        2,
        8,
    )
    assert (
        second["sequence"],
        second["invocation"],
        second["group_sequence"],
    ) == (4, 1, 8)


def test_model_ir_pass_diagnostic_numbering_scans_existing_history_once() -> None:
    class CountingDiagnostics(list):
        def __init__(self, values) -> None:
            super().__init__(values)
            self.iteration_count = 0

        def __iter__(self):
            self.iteration_count += 1
            return super().__iter__()

    model_ir = _add_model_ir()
    diagnostics = CountingDiagnostics(
        [
            {
                "stage": "model_ir_pass",
                "code": "cleanup.first",
                "group_sequence": 5,
            }
        ]
    )
    specs = [
        PassSpec(
            pass_id=pass_id,
            phase=PassPhase.POST_LOWERING_CLEANUP,
            callback=lambda state: {"changed": False},
        )
        for pass_id in ["cleanup.first", "cleanup.second", "cleanup.third"]
    ]

    run_model_ir_pass_group(model_ir, specs=specs, diagnostics=diagnostics)

    assert diagnostics.iteration_count == 1


def test_conversion_session_reuses_pass_diagnostic_ledger_across_groups(
    monkeypatch,
) -> None:
    session = ConversionSession(
        onnx_model=_add_onnx_model(),
        model_ir=_add_model_ir(),
        shape_map={},
        dtype_map={},
        constants={},
    )
    diagnostics = session.diagnostics
    ledger_type = type(diagnostics)
    original_rebuild = getattr(ledger_type, "_rebuild_model_ir_numbering")
    rebuild_count = 0

    def counted_rebuild(ledger) -> None:
        nonlocal rebuild_count
        rebuild_count += 1
        original_rebuild(ledger)

    monkeypatch.setattr(
        ledger_type,
        "_rebuild_model_ir_numbering",
        counted_rebuild,
    )
    diagnostics[:] = [
        {
            "stage": "model_ir_pass",
            "code": "cleanup.repeated",
            "group_sequence": 4,
        }
    ]
    spec = PassSpec(
        pass_id="cleanup.repeated",
        phase=PassPhase.POST_LOWERING_CLEANUP,
        callback=lambda state: {"changed": False},
    )

    run_model_ir_pass_group(session.model_ir, specs=[spec], diagnostics=diagnostics)
    run_model_ir_pass_group(session.model_ir, specs=[spec], diagnostics=diagnostics)

    assert rebuild_count == 1
    assert [event["sequence"] for event in diagnostics[1:]] == [2, 3]
    assert [event["invocation"] for event in diagnostics[1:]] == [2, 3]
    assert [event["group_sequence"] for event in diagnostics[1:]] == [5, 6]


def test_conversion_session_append_only_diagnostics_need_no_ledger_rebuild(
    monkeypatch,
) -> None:
    session = ConversionSession(
        onnx_model=_add_onnx_model(),
        model_ir=_add_model_ir(),
        shape_map={},
        dtype_map={},
        constants={},
    )
    rebuild_count = 0
    original_rebuild = ModelIRPassDiagnostics._rebuild_model_ir_numbering

    def counted_rebuild(ledger) -> None:
        nonlocal rebuild_count
        rebuild_count += 1
        original_rebuild(ledger)

    monkeypatch.setattr(
        ModelIRPassDiagnostics,
        "_rebuild_model_ir_numbering",
        counted_rebuild,
    )
    session.record_diagnostic(stage="lowering", code="note", message="kept")
    spec = PassSpec(
        pass_id="cleanup.repeated",
        phase=PassPhase.POST_LOWERING_CLEANUP,
        callback=lambda state: {"changed": False},
    )

    run_model_ir_pass_group(
        session.model_ir,
        specs=[spec],
        diagnostics=session.diagnostics,
    )
    run_model_ir_pass_group(
        session.model_ir,
        specs=[spec],
        diagnostics=session.diagnostics,
    )

    assert rebuild_count == 0
    assert [event["sequence"] for event in session.diagnostics[1:]] == [1, 2]
    assert [event["invocation"] for event in session.diagnostics[1:]] == [1, 2]
    assert [event["group_sequence"] for event in session.diagnostics[1:]] == [1, 2]


def test_pass_diagnostic_ledger_tracks_append_only_list_mutations() -> None:
    diagnostics = ModelIRPassDiagnostics()
    diagnostics.append({"stage": "lowering", "code": "ignored"})
    diagnostics.extend(
        [
            {
                "stage": "model_ir_pass",
                "code": "cleanup.first",
                "group_sequence": -2,
            },
            {
                "stage": "model_ir_pass",
                "code": "cleanup.first",
                "group_sequence": 3,
            },
        ]
    )
    diagnostics.insert(
        0,
        {
            "stage": "model_ir_pass",
            "code": "cleanup.second",
            "group_sequence": 2,
        },
    )

    assert diagnostics.model_ir_numbering_snapshot() == (
        3,
        3,
        {"cleanup.first": 2, "cleanup.second": 1},
    )


def test_pass_diagnostic_ledger_rebuilds_after_destructive_list_mutations() -> None:
    diagnostics = ModelIRPassDiagnostics(
        [
            {
                "stage": "model_ir_pass",
                "code": "cleanup.first",
                "group_sequence": 8,
            },
            {
                "stage": "model_ir_pass",
                "code": "cleanup.second",
                "group_sequence": 5,
            },
        ]
    )
    diagnostics.pop(0)
    diagnostics[0] = {
        "stage": "model_ir_pass",
        "code": "cleanup.replaced",
        "group_sequence": -4,
    }

    assert diagnostics.model_ir_numbering_snapshot() == (
        1,
        -4,
        {"cleanup.replaced": 1},
    )
    diagnostics.clear()
    assert diagnostics.model_ir_numbering_snapshot() == (0, None, {})


def test_final_sinet_counters_gate_shape_reconciliation(monkeypatch) -> None:
    pass_counters = {
        "_optimize_sinet_late_residual_pre_add_mul_add_prelu_chains": (
            "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains"
        ),
        "_optimize_sinet_deep_skip_pre_add_concat_prelu_fanout_chains": (
            "optimized_sinet_deep_skip_pre_add_concat_prelu_fanout_chains"
        ),
        "_optimize_sinet_deep_skip_dual_resize_affine_transpose_chains": (
            "optimized_sinet_deep_skip_dual_resize_affine_transpose_chains"
        ),
        "_optimize_sinet_shared_post_prelu_transpose_fanout_chains": (
            "optimized_sinet_shared_post_prelu_transpose_fanout_chains"
        ),
        "_optimize_sinet_deep_skip_concat_resize_affine_tail_chains": (
            "optimized_sinet_deep_skip_concat_resize_affine_tail_chains"
        ),
        "_optimize_sinet_concat_resize_affine_transpose_chains": (
            "optimized_sinet_concat_resize_affine_transpose_chains"
        ),
    }
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0

    def counted_reconcile(*args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_changed_pass(changed_pass: str | None) -> int:
        nonlocal reconcile_count
        for pass_name, counter_name in pass_counters.items():
            changed = pass_name == changed_pass

            def pass_result(*args, _counter=counter_name, _changed=changed, **kwargs):
                return {_counter: int(_changed)}

            monkeypatch.setattr(lowering_module, pass_name, pass_result)
        reconcile_count = 0
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"sinet_reconcile_{changed_pass or 'none'}",
        )
        return reconcile_count

    unchanged_count = run_with_changed_pass(None)
    for pass_name in pass_counters:
        assert run_with_changed_pass(pass_name) == unchanged_count + 1


def test_final_mixed_singleton_concat_counter_gates_shape_reconciliation(
    monkeypatch,
) -> None:
    counter_name = "repaired_mixed_singleton_nchw_inputs_for_nhwc_concat"
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0

    def counted_reconcile(*args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_changed_counter(changed: bool) -> int:
        nonlocal reconcile_count

        def pass_result(*args, **kwargs):
            return {counter_name: int(changed)}

        monkeypatch.setattr(
            lowering_module,
            "_repair_mixed_singleton_nchw_inputs_for_nhwc_concat",
            pass_result,
        )
        reconcile_count = 0
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"mixed_singleton_concat_reconcile_{int(changed)}",
        )
        return reconcile_count

    unchanged_count = run_with_changed_counter(False)
    assert run_with_changed_counter(True) == unchanged_count + 1


def test_final_consecutive_reshape_counters_gate_shape_reconciliation(
    monkeypatch,
) -> None:
    counter_names = {
        "removed_noop_reshape_chains",
        "rewritten_consecutive_reshape_passthrough_chains",
        "rewritten_fanout_bypass_reshape_passthrough_chains",
    }
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0

    def counted_reconcile(*args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        return original_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_changed_counter(changed_counter: str | None) -> int:
        nonlocal reconcile_count

        def pass_result(*args, **kwargs):
            return {
                counter_name: int(counter_name == changed_counter)
                for counter_name in counter_names
            }

        monkeypatch.setattr(
            lowering_module,
            "run_consecutive_reshape_cleanup",
            pass_result,
        )
        reconcile_count = 0
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"reshape_reconcile_{changed_counter or 'none'}",
        )
        return reconcile_count

    unchanged_count = run_with_changed_counter(None)
    for counter_name in counter_names:
        assert run_with_changed_counter(counter_name) == unchanged_count + 1


def test_final_prelu_reconciles_after_rewrite_or_prune(monkeypatch) -> None:
    counter_name = "rewritten_prelu_transpose_passthrough_chains"
    probe_name = "unused_final_prelu_probe"
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0
    prelu_invocations = 0

    def counted_reconcile(model_ir, *args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        result = original_reconcile(model_ir, *args, **kwargs)
        model_ir.tensors[probe_name] = TensorIR(
            name=probe_name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
        return result

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_outcome(outcome: str) -> int:
        nonlocal reconcile_count, prelu_invocations
        reconcile_count = 0
        prelu_invocations = 0

        def pass_result(model_ir, *args, **kwargs):
            nonlocal prelu_invocations
            prelu_invocations += 1
            is_final_invocation = prelu_invocations == 2
            if is_final_invocation and outcome == "prune":
                assert model_ir.tensors.pop(probe_name, None) is not None
            return {
                counter_name: int(
                    is_final_invocation and outcome == "rewrite"
                )
            }

        monkeypatch.setattr(
            lowering_module,
            "_optimize_prelu_transpose_passthrough_chains",
            pass_result,
        )
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"prelu_reconcile_{outcome}",
        )
        assert prelu_invocations == 2
        return reconcile_count

    unchanged_count = run_with_outcome("unchanged")
    assert run_with_outcome("rewrite") == unchanged_count + 1
    assert run_with_outcome("prune") == unchanged_count + 1


def test_final_se_fc_gather_reconciles_after_rewrite_or_prune(monkeypatch) -> None:
    sinet_counter = (
        "optimized_sinet_shuffle_residual_mul_posttranspose_tail_chains"
    )
    se_fc_counter = "optimized_transpose_se_fc_mul_prepost_nhwc_chains"
    gather_counter = (
        "optimized_transpose_gather_transpose_nhwc_channel_chains"
    )
    probe_name = "unused_final_se_fc_gather_probe"
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0

    def counted_reconcile(model_ir, *args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        result = original_reconcile(model_ir, *args, **kwargs)
        model_ir.tensors[probe_name] = TensorIR(
            name=probe_name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
        return result

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_outcome(outcome: str) -> int:
        nonlocal reconcile_count
        reconcile_count = 0

        def sinet_result(model_ir, *args, **kwargs):
            if outcome == "prune":
                assert model_ir.tensors.pop(probe_name, None) is not None
            return {sinet_counter: int(outcome == "sinet")}

        def cluster_result(*args, **kwargs):
            return (
                {se_fc_counter: int(outcome == "se_fc")},
                {gather_counter: int(outcome == "gather")},
            )

        monkeypatch.setattr(
            lowering_module,
            "_optimize_sinet_shuffle_residual_mul_posttranspose_tail_chains",
            sinet_result,
        )
        monkeypatch.setattr(
            lowering_module,
            "run_se_fc_gather_channel_fanout",
            cluster_result,
        )
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"se_fc_gather_reconcile_{outcome}",
        )
        return reconcile_count

    unchanged_count = run_with_outcome("unchanged")
    for outcome in ("sinet", "se_fc", "gather", "prune"):
        assert run_with_outcome(outcome) == unchanged_count + 1


def test_late_binary_repair_reconciles_after_change_or_prune(monkeypatch) -> None:
    signature_counter = "sanitized_static_shape_signature_consistency"
    exact_counter = "inserted_rank4_binary_layout_fix_transpose"
    singleton_counter = (
        "repaired_rank4_binary_singleton_broadcast_layout_mismatch"
    )
    probe_name = "unused_late_binary_repair_probe"
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0
    signature_invocations = 0
    exact_invocations = 0
    singleton_invocations = 0

    def counted_reconcile(model_ir, *args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        result = original_reconcile(model_ir, *args, **kwargs)
        model_ir.tensors[probe_name] = TensorIR(
            name=probe_name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
        return result

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_outcome(outcome: str) -> int:
        nonlocal reconcile_count
        nonlocal signature_invocations, exact_invocations, singleton_invocations
        reconcile_count = 0
        signature_invocations = 0
        exact_invocations = 0
        singleton_invocations = 0

        def signature_result(*args, **kwargs):
            nonlocal signature_invocations
            signature_invocations += 1
            return {
                signature_counter: int(
                    signature_invocations == 1 and outcome == "signature"
                )
            }

        def exact_result(model_ir, *args, **kwargs):
            nonlocal exact_invocations
            exact_invocations += 1
            is_late_boundary = exact_invocations == 2
            if is_late_boundary and outcome == "prune":
                assert model_ir.tensors.pop(probe_name, None) is not None
            return {
                exact_counter: int(
                    is_late_boundary and outcome == "exact"
                )
            }

        def singleton_result(*args, **kwargs):
            nonlocal singleton_invocations
            singleton_invocations += 1
            return {
                singleton_counter: int(
                    singleton_invocations == 2 and outcome == "singleton"
                )
            }

        monkeypatch.setattr(
            lowering_module,
            "_sanitize_static_shape_signature_consistency",
            signature_result,
        )
        monkeypatch.setattr(
            lowering_module,
            "_repair_rank4_binary_layout_mismatch_with_transpose_adapter",
            exact_result,
        )
        monkeypatch.setattr(
            lowering_module,
            "_repair_rank4_binary_singleton_broadcast_layout_mismatch",
            singleton_result,
        )
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"late_binary_reconcile_{outcome}",
        )
        assert signature_invocations == 2
        assert exact_invocations >= 2
        assert singleton_invocations >= 2
        return reconcile_count

    unchanged_count = run_with_outcome("unchanged")
    for outcome in ("signature", "exact", "singleton", "prune"):
        assert run_with_outcome(outcome) == unchanged_count + 1


def test_stats_have_positive_count_accepts_only_positive_mutations() -> None:
    assert lowering_module._stats_have_positive_count() is False
    assert lowering_module._stats_have_positive_count(
        {"first": 0},
        {"second": -1},
    ) is False
    assert lowering_module._stats_have_positive_count(
        {"first": 0},
        {"second": 2},
        {"third": 0},
    ) is True


def test_shared_late_reconciliation_uses_all_results_and_pruning(
    monkeypatch,
) -> None:
    probe_name = "unused_shared_late_reconcile_probe"
    original_reconcile = lowering_module._reconcile_static_tensor_shapes
    reconcile_count = 0

    def counted_reconcile(model_ir, *args, **kwargs):
        nonlocal reconcile_count
        reconcile_count += 1
        result = original_reconcile(model_ir, *args, **kwargs)
        model_ir.tensors[probe_name] = TensorIR(
            name=probe_name,
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
        return result

    monkeypatch.setattr(
        lowering_module,
        "_reconcile_static_tensor_shapes",
        counted_reconcile,
    )

    def run_with_outcome(outcome: str) -> int:
        nonlocal reconcile_count
        reconcile_count = 0
        invocations = {
            "boundary": 0,
            "hardswish": 0,
            "squeeze": 0,
            "wrongway": 0,
            "exact": 0,
            "singleton": 0,
            "cluster": 0,
        }

        def direct_result(name: str, counter: str, *, target: int = 1):
            def result(model_ir, *args, **kwargs):
                invocations[name] += 1
                is_target = invocations[name] == target
                if name == "exact" and is_target and outcome == "prune":
                    assert model_ir.tensors.pop(probe_name, None) is not None
                return {counter: int(is_target and outcome == name)}

            return result

        def cluster_result(*args, **kwargs):
            invocations["cluster"] += 1
            is_target = invocations["cluster"] == 2
            return (
                {"singleton_channel": int(is_target and outcome == "cluster_1")},
                {"duplicate_fanout": int(is_target and outcome == "cluster_2")},
                {"consecutive_reshape": int(is_target and outcome == "cluster_3")},
            )

        monkeypatch.setattr(
            lowering_module,
            "_realign_dynamic_boundary_shape_signature_map",
            direct_result("boundary", "boundary_signature"),
        )
        monkeypatch.setattr(
            lowering_module,
            "_sanitize_hardswish_tensor_shapes",
            direct_result("hardswish", "hardswish", target=2),
        )
        monkeypatch.setattr(
            lowering_module,
            "_sanitize_squeeze_axes_with_static_input_shapes",
            direct_result("squeeze", "squeeze"),
        )
        monkeypatch.setattr(
            lowering_module,
            "_sanitize_wrong_way_nchw_to_nhwc_transpose_before_conv",
            direct_result("wrongway", "wrongway"),
        )
        monkeypatch.setattr(
            lowering_module,
            "_repair_rank4_binary_layout_mismatch_with_transpose_adapter",
            direct_result("exact", "exact"),
        )
        monkeypatch.setattr(
            lowering_module,
            "_repair_rank4_binary_singleton_broadcast_layout_mismatch",
            direct_result("singleton", "singleton"),
        )
        monkeypatch.setattr(
            lowering_module,
            "run_singleton_consecutive_reshape",
            cluster_result,
        )
        lower_onnx_to_ir(
            _add_onnx_model(),
            output_file_name=f"shared_late_reconcile_{outcome}",
        )
        assert invocations["boundary"] >= 1
        assert invocations["cluster"] >= 2
        return reconcile_count

    unchanged_count = run_with_outcome("unchanged")
    for outcome in (
        "boundary",
        "hardswish",
        "squeeze",
        "wrongway",
        "exact",
        "singleton",
        "cluster_1",
        "cluster_2",
        "cluster_3",
        "prune",
    ):
        assert run_with_outcome(outcome) == unchanged_count + 1


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
