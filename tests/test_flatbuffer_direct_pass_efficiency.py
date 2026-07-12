from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    run_model_ir_pass_group,
)
from onnx2tf.tflite_builder.core.passes import PassPhase, PassSpec
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.attention_layout import (
    run_conv_attention_layout_cleanup,
    run_mixed_attention_layout_cleanup,
    run_qkv_attention_bridge_cleanup,
    run_qkv_attention_prefix_cleanup,
)
from onnx2tf.tflite_builder.passes.boundary_input_layout import (
    run_boundary_input_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.cast_cleanup import run_redundant_cast_cleanup
from onnx2tf.tflite_builder.passes.constant_fold import (
    run_constant_input_fold_cleanup,
)
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_clamp_cleanup,
    run_consecutive_mul_constants_cleanup,
    run_duplicate_fanout_cleanup,
    run_maximum_zero_relu_cleanup,
    run_squeeze_reshape_identity_cleanup,
)
from onnx2tf.tflite_builder.passes.quantization_cleanup import (
    run_terminal_quantize_dequantize_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import (
    run_normalization_pad_layout_cleanup,
    run_pad_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    run_input_unary_passthrough_cleanup,
)


def _identity_chain(operator_count: int) -> ModelIR:
    model_ir = ModelIR("identity_chain")
    model_ir.inputs = ["t0"]
    model_ir.outputs = [f"t{operator_count}"]
    model_ir.tensors = {
        f"t{index}": TensorIR(
            name=f"t{index}",
            dtype="FLOAT32",
            shape=[1],
            shape_signature=[1],
        )
        for index in range(operator_count + 1)
    }
    model_ir.operators = [
        OperatorIR("IDENTITY", [f"t{index}"], [f"t{index + 1}"])
        for index in range(operator_count)
    ]
    return model_ir


def test_model_only_preflight_visits_graph_once_and_skips_state(monkeypatch) -> None:
    model_ir = _identity_chain(256)
    visited = 0
    refresh_count = 0
    callback_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def preflight(candidate_model: ModelIR) -> ModelIRPreflightResult:
        nonlocal visited
        for _ in candidate_model.operators:
            visited += 1
        return ModelIRPreflightResult(False, visited)

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    def callback(state: ModelIRPassState) -> dict:
        nonlocal callback_count
        callback_count += 1
        return {"changed": False}

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)

    _, results = run_model_ir_pass_group(
        model_ir,
        specs=[
            PassSpec(
                pass_id="cleanup.preflight_probe",
                phase=PassPhase.POST_LOWERING_CLEANUP,
                callback=callback,
                transactional=True,
            )
        ],
        preflight=preflight,
    )

    assert visited == 256
    assert refresh_count == 0
    assert callback_count == 0
    assert results[0].iterations == 0
    assert results[0].details == {"skipped_by_precondition": True}


def test_all_production_runner_preflights_avoid_heavy_no_candidate_work(
    monkeypatch,
) -> None:
    model_ir = _identity_chain(256)
    diagnostics: list[dict] = []
    calls = {"refresh": 0, "snapshot": 0, "fingerprint": 0}
    original_refresh = ModelIRGraphIndex.refresh
    original_snapshot = ModelIRPassState.snapshot
    original_fingerprint = ModelIRPassState.fingerprint

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        calls["refresh"] += 1
        original_refresh(index)

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        calls["snapshot"] += 1
        return original_snapshot(state)

    def counted_fingerprint(state: ModelIRPassState) -> bytes:
        calls["fingerprint"] += 1
        return original_fingerprint(state)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)
    monkeypatch.setattr(ModelIRPassState, "fingerprint", counted_fingerprint)

    run_duplicate_fanout_cleanup(model_ir, diagnostics=diagnostics)
    run_clamp_cleanup(model_ir, diagnostics=diagnostics)
    run_maximum_zero_relu_cleanup(model_ir, diagnostics=diagnostics)
    run_consecutive_mul_constants_cleanup(model_ir, diagnostics=diagnostics)
    run_squeeze_reshape_identity_cleanup(model_ir, diagnostics=diagnostics)
    run_conv_attention_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_mixed_attention_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_qkv_attention_bridge_cleanup(model_ir, diagnostics=diagnostics)
    run_qkv_attention_prefix_cleanup(model_ir, diagnostics=diagnostics)
    run_pad_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_normalization_pad_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_input_unary_passthrough_cleanup(model_ir, diagnostics=diagnostics)
    run_boundary_input_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_constant_input_fold_cleanup(model_ir, diagnostics=diagnostics)
    run_redundant_cast_cleanup(model_ir, diagnostics=diagnostics)
    run_terminal_quantize_dequantize_cleanup(model_ir, diagnostics=diagnostics)

    assert calls == {"refresh": 0, "snapshot": 0, "fingerprint": 0}
    assert len(diagnostics) == 29
    assert all(event["status"] == "skipped" for event in diagnostics)
    assert all(
        event["metrics"]
        == {
            "preflight_operators_visited": 256,
            "state_built": False,
            "snapshot_count": 0,
            "fingerprint_count": 0,
        }
        for event in diagnostics
    )


def test_one_candidate_builds_one_index_and_one_snapshot_without_fingerprint(
    monkeypatch,
) -> None:
    model_ir = ModelIR("maximum_zero_candidate")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1], [1]),
        "zero": TensorIR(
            "zero",
            "FLOAT32",
            [],
            [],
            data=np.asarray(0.0, dtype=np.float32),
        ),
        "y": TensorIR("y", "FLOAT32", [1], [1]),
    }
    model_ir.operators = [OperatorIR("MAXIMUM", ["x", "zero"], ["y"])]
    calls = {"refresh": 0, "snapshot": 0, "fingerprint": 0}
    original_refresh = ModelIRGraphIndex.refresh
    original_snapshot = ModelIRPassState.snapshot
    original_fingerprint = ModelIRPassState.fingerprint

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        calls["refresh"] += 1
        original_refresh(index)

    def counted_snapshot(state: ModelIRPassState) -> ModelIR:
        calls["snapshot"] += 1
        return original_snapshot(state)

    def counted_fingerprint(state: ModelIRPassState) -> bytes:
        calls["fingerprint"] += 1
        return original_fingerprint(state)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)
    monkeypatch.setattr(ModelIRPassState, "fingerprint", counted_fingerprint)

    diagnostics: list[dict] = []
    stats = run_maximum_zero_relu_cleanup(model_ir, diagnostics=diagnostics)

    assert stats == {"rewritten_maximum_with_zero_input2_to_relu": 1}
    assert calls == {"refresh": 1, "snapshot": 1, "fingerprint": 0}
    assert diagnostics[0]["metrics"] == {
        "preflight_operators_visited": 1,
        "state_built": True,
        "snapshot_count": 1,
        "fingerprint_count": 0,
    }
