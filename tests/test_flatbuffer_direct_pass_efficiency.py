from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import (
    ModelIRPreflightResult,
    ModelIRPassState,
    ModelIRPassStateScope,
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
from onnx2tf.tflite_builder.passes.boundary_input_chains import (
    run_boundary_input_normalization_cleanup,
)
from onnx2tf.tflite_builder.passes.channel_slice_layout import (
    run_channel_slice_merge_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.channel_shuffle import (
    run_nchw_channel_shuffle_cleanup,
    run_nhwc_channel_shuffle_cleanup,
    run_stale_nchw_channel_shuffle_repair,
    run_two_way_channel_shuffle_cleanup,
)
from onnx2tf.tflite_builder.passes.cast_cleanup import run_redundant_cast_cleanup
from onnx2tf.tflite_builder.passes.constant_fold import (
    run_constant_input_fold_cleanup,
)
from onnx2tf.tflite_builder.passes.graph_cleanup import (
    run_clamp_cleanup,
    run_consecutive_reshape_cleanup,
    run_consecutive_mul_constants_cleanup,
    run_duplicate_fanout_cleanup,
    run_maximum_zero_relu_cleanup,
    run_squeeze_reshape_identity_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_clamp_unary_relu_orchestration import (
    TerminalClampUnaryReLUContext,
    run_terminal_clamp_unary_relu,
)
from onnx2tf.tflite_builder.passes.terminal_singleton_maxpool_reshape_orchestration import (
    TerminalSingletonMaxPoolReshapeContext,
    run_terminal_singleton_maxpool_reshape,
)
from onnx2tf.tflite_builder.passes.late_dequant_unary_fanout_orchestration import (
    LateDequantUnaryFanoutContext,
    run_late_dequant_unary_fanout,
)
from onnx2tf.tflite_builder.passes.transpose_unary_fanout_orchestration import (
    TransposeUnaryFanoutContext,
    run_transpose_unary_fanout,
)
from onnx2tf.tflite_builder.passes.late_spp_concat_unary_conv_orchestration import (
    LateSPPConcatUnaryConvContext,
    run_late_spp_concat_unary_conv,
)
from onnx2tf.tflite_builder.passes.boundary_batchmatmul_unary_orchestration import (
    BoundaryBatchMatMulUnaryContext,
    run_boundary_batchmatmul_unary,
)
from onnx2tf.tflite_builder.passes.channel_slice_pad_mul_orchestration import (
    ChannelSlicePadMulContext,
    run_channel_slice_pad_mul,
)
from onnx2tf.tflite_builder.passes.late_hard_activation_layout_orchestration import (
    LateHardActivationLayoutContext,
    run_late_hard_activation_layout,
)
from onnx2tf.tflite_builder.passes.absolute_final_normalization_attention_orchestration import (
    AbsoluteFinalNormalizationAttentionContext,
    run_absolute_final_normalization_attention,
)
from onnx2tf.tflite_builder.passes.qkv_attention_orchestration import (
    QKVAttentionContext,
    run_qkv_attention,
)
from onnx2tf.tflite_builder.passes.duplicate_quantized_prelu_orchestration import (
    DuplicateQuantizedPReLUContext,
    run_duplicate_quantized_prelu,
)
from onnx2tf.tflite_builder.passes.constant_fold_cast_orchestration import (
    ConstantFoldCastContext,
    run_constant_fold_cast,
)
from onnx2tf.tflite_builder.passes.very_late_gather_constant_normalization_orchestration import (
    VeryLateGatherConstantNormalizationContext,
    run_very_late_gather_constant_normalization,
)
from onnx2tf.tflite_builder.passes.se_fc_gather_channel_fanout_orchestration import (
    SEFCGatherChannelFanoutContext,
    run_se_fc_gather_channel_fanout,
)
from onnx2tf.tflite_builder.passes.terminal_boundary_layout_orchestration import (
    TerminalBoundaryLayoutContext,
    run_terminal_boundary_layout,
)
from onnx2tf.tflite_builder.passes.quantization_cleanup import (
    run_terminal_quantize_dequantize_cleanup,
)
from onnx2tf.tflite_builder.passes.quantized_prelu import (
    run_quantized_prelu_cleanup,
)
from onnx2tf.tflite_builder.passes.quantized_reshape import (
    run_quantized_reshape_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_maxpool_layout import (
    run_singleton_maxpool_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.singleton_reshape_layout import (
    run_flatten_concat_reshape_cleanup,
    run_singleton_channel_transpose_cleanup,
    run_singleton_reshape_layout_cleanup,
    run_singleton_spatial_reshape_cleanup,
)
from onnx2tf.tflite_builder.passes.pad_layout import (
    run_normalization_pad_layout_cleanup,
    run_pad_layout_cleanup,
    run_pad_mul_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.input_passthrough_layout import (
    run_hard_activation_passthrough_cleanup,
    run_input_unary_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
    run_trailing_output_transpose_cleanup,
    run_transpose_gather_axis_cleanup,
    run_transpose_gather_channel_fanout_cleanup,
    run_transpose_unary_binary_fanout_bridge_cleanup,
    run_transpose_unary_fanout_bridge_cleanup,
    run_transpose_unary_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.layernorm_layout import (
    run_layernorm_statistics_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.mean_layout import (
    run_mean_mul_add_conv_layout_cleanup,
    run_transpose_mean_passthrough_cleanup,
)
from onnx2tf.tflite_builder.passes.terminal_mean_layout import (
    run_terminal_mean_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.se_layout import (
    run_se_conv_layout_cleanup,
    run_se_fc_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.elementwise_gate_layout import (
    run_elementwise_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.multi_branch_gate_layout import (
    run_multi_branch_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dual_postconv_gate_layout import (
    run_dual_postconv_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.ndhwc_gate_layout import (
    run_ndhwc_gate_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.cost_volume_scatter_layout import (
    run_cost_volume_scatter_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.add_concat_suffix_layout import (
    run_add_concat_suffix_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dual_mul_concat_layout import (
    run_dual_mul_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.axis3_const_concat_layout import (
    run_axis3_const_concat_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.dequant_concat_quantize_layout import (
    run_dequant_concat_quantize_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.concat_unary_conv_layout import (
    run_concat_unary_conv_layout_cleanup,
)
from onnx2tf.tflite_builder.passes.spp_layout import run_spp_layout_cleanup


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
    run_pad_mul_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_input_unary_passthrough_cleanup(model_ir, diagnostics=diagnostics)
    run_hard_activation_passthrough_cleanup(model_ir, diagnostics=diagnostics)
    run_boundary_input_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_boundary_input_normalization_cleanup(model_ir, diagnostics=diagnostics)
    run_channel_slice_merge_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_nchw_channel_shuffle_cleanup(model_ir, diagnostics=diagnostics)
    run_nhwc_channel_shuffle_cleanup(model_ir, diagnostics=diagnostics)
    run_stale_nchw_channel_shuffle_repair(model_ir, diagnostics=diagnostics)
    run_two_way_channel_shuffle_cleanup(model_ir, diagnostics=diagnostics)
    run_constant_input_fold_cleanup(model_ir, diagnostics=diagnostics)
    run_redundant_cast_cleanup(model_ir, diagnostics=diagnostics)
    run_terminal_quantize_dequantize_cleanup(model_ir, diagnostics=diagnostics)
    run_quantized_prelu_cleanup(model_ir, diagnostics=diagnostics)
    run_quantized_reshape_cleanup(model_ir, diagnostics=diagnostics)
    run_singleton_maxpool_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_singleton_reshape_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_consecutive_reshape_cleanup(model_ir, diagnostics=diagnostics)
    run_flatten_concat_reshape_cleanup(model_ir, diagnostics=diagnostics)
    run_singleton_spatial_reshape_cleanup(model_ir, diagnostics=diagnostics)
    run_singleton_channel_transpose_cleanup(model_ir, diagnostics=diagnostics)
    run_layout_transpose_cleanup(model_ir, diagnostics=diagnostics)
    run_trailing_output_transpose_cleanup(model_ir, diagnostics=diagnostics)
    run_transpose_gather_axis_cleanup(model_ir, diagnostics=diagnostics)
    run_transpose_gather_channel_fanout_cleanup(model_ir, diagnostics=diagnostics)
    run_transpose_unary_binary_fanout_bridge_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )
    run_transpose_unary_fanout_bridge_cleanup(model_ir, diagnostics=diagnostics)
    run_transpose_unary_passthrough_cleanup(model_ir, diagnostics=diagnostics)
    run_transpose_mean_passthrough_cleanup(model_ir, diagnostics=diagnostics)
    run_mean_mul_add_conv_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_layernorm_statistics_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_terminal_mean_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_se_conv_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_se_fc_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_elementwise_gate_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_multi_branch_gate_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_dual_postconv_gate_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_ndhwc_gate_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_cost_volume_scatter_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_add_concat_suffix_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_dual_mul_concat_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_axis3_const_concat_layout_cleanup(model_ir, diagnostics=diagnostics)
    run_dequant_concat_quantize_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )
    run_concat_unary_conv_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )
    run_spp_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert calls == {"refresh": 0, "snapshot": 0, "fingerprint": 0}
    assert len(diagnostics) == 85
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


def test_adjacent_gate_runners_reuse_one_lazy_pass_state(monkeypatch) -> None:
    operator_types = [
        "TRANSPOSE",
        "MEAN",
        "REDUCE_MAX",
        "CONCATENATION",
        "MIRROR_PAD",
        "TRANSPOSE",
        "CONV_2D",
        "MUL",
        "ADD",
        "LOGISTIC",
        "SUB",
        "RESHAPE",
        "LEAKY_RELU",
        "SCATTER_ND",
        "CONV_3D",
    ]
    model_ir = ModelIR(
        "gate_scope_preflight_only",
        operators=[OperatorIR(op_type, [], []) for op_type in operator_types],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    for runner in [
        run_mixed_attention_layout_cleanup,
        run_elementwise_gate_layout_cleanup,
        run_pad_layout_cleanup,
        run_dual_postconv_gate_layout_cleanup,
        run_ndhwc_gate_layout_cleanup,
        run_cost_volume_scatter_layout_cleanup,
        run_add_concat_suffix_layout_cleanup,
        run_dual_mul_concat_layout_cleanup,
    ]:
        runner(
            model_ir,
            diagnostics=diagnostics,
            state_scope=state_scope,
        )

    assert refresh_count == 1
    assert len(diagnostics) == 15
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_qkv_attention_pair_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "qkv_attention_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "GATHER",
                "GATHER",
                "RESHAPE",
                "RESHAPE",
                "TRANSPOSE",
                "TRANSPOSE",
                "SLICE",
                "SLICE",
                "SLICE",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_qkv_attention(
        QKVAttentionContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 6
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        True,
        True,
        True,
        False,
        False,
    ]


def test_late_layout_qkv_bridge_pair_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "late_layout_qkv_bridge_scope_preflight_only",
        operators=[
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("SLICE", [], []),
            OperatorIR("SLICE", [], []),
            OperatorIR("SLICE", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    run_layout_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_qkv_attention_bridge_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_duplicate_quantized_prelu_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "duplicate_quantized_prelu_scope_preflight_only",
        operators=[
            OperatorIR("RESHAPE", [], []),
            OperatorIR("RESHAPE", [], []),
            OperatorIR("DEQUANTIZE", [], []),
            OperatorIR("PRELU", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_duplicate_quantized_prelu(
        DuplicateQuantizedPReLUContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        ),
        include_transpose=False,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 5
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_constant_fold_cast_pair_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "constant_fold_cast_scope_preflight_only",
        operators=[
            OperatorIR(
                "CAST",
                [],
                [],
                options={
                    "inDataType": "INT32",
                    "outDataType": "INT64",
                },
            ),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    run_constant_fold_cast(
        ConstantFoldCastContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        ),
        state_scope=state_scope,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 5
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        True,
        True,
        False,
        False,
    ]


def test_very_late_gather_constant_normalization_cluster_reuses_one_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "very_late_gather_constant_normalization_scope_preflight_only",
        operators=[
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("GATHER", [], []),
            OperatorIR(
                "CAST",
                [],
                [],
                options={
                    "inDataType": "INT32",
                    "outDataType": "INT64",
                },
            ),
            OperatorIR("PAD", [], []),
            OperatorIR("MEAN", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_very_late_gather_constant_normalization(
        VeryLateGatherConstantNormalizationContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 7
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_se_fc_gather_channel_fanout_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "se_fc_gather_channel_fanout_scope_preflight_only",
        operators=[
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("MUL", [], []),
            OperatorIR("FULLY_CONNECTED", [], []),
            OperatorIR("GATHER", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_se_fc_gather_channel_fanout(
        SEFCGatherChannelFanoutContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 2
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        False,
    ]


def test_terminal_boundary_layout_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "terminal_boundary_layout_scope_preflight_only",
        operators=[
            OperatorIR(
                "TRANSPOSE",
                ["x", "perm"],
                ["x_onnx_ncx_internal"],
            ),
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("MUL", [], []),
            OperatorIR("CONCATENATION", [], []),
            OperatorIR("PAD", [], []),
            OperatorIR("GATHER", [], []),
        ],
    )
    model_ir.inputs = ["x"]
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_terminal_boundary_layout(
        TerminalBoundaryLayoutContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 7
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_late_ndhwc_cost_volume_pair_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "late_ndhwc_cost_volume_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "RESHAPE",
                "LEAKY_RELU",
                "MUL",
                "SCATTER_ND",
                "CONV_3D",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    run_ndhwc_gate_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_cost_volume_scatter_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        True,
        False,
    ]


def test_late_concat_layout_cluster_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "late_concat_layout_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "CONCATENATION",
                "DEQUANTIZE",
                "QUANTIZE",
                "MEAN",
                "SUB",
                "MUL",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    for runner in [
        run_axis3_const_concat_layout_cleanup,
        run_dequant_concat_quantize_layout_cleanup,
        run_layernorm_statistics_layout_cleanup,
        run_layout_transpose_cleanup,
    ]:
        runner(
            model_ir,
            diagnostics=diagnostics,
            state_scope=state_scope,
        )

    assert refresh_count == 1
    assert len(diagnostics) == 5
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        False,
        False,
        False,
        False,
    ]


def test_late_dequant_unary_fanout_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "late_dequant_unary_fanout_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "DEQUANTIZE",
                "CONCATENATION",
                "QUANTIZE",
                "RELU",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_late_dequant_unary_fanout(
        LateDequantUnaryFanoutContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_terminal_singleton_maxpool_reshape_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "terminal_singleton_maxpool_reshape_scope_preflight_only",
        operators=[
            OperatorIR("RESHAPE", [], []),
            OperatorIR("MAX_POOL_2D", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_terminal_singleton_maxpool_reshape(
        TerminalSingletonMaxPoolReshapeContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        True,
        False,
    ]


def test_terminal_clamp_unary_relu_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "terminal_clamp_unary_relu_scope_preflight_only",
        operators=[
            OperatorIR("MAXIMUM", [], []),
            OperatorIR("MINIMUM", [], []),
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("RELU", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_terminal_clamp_unary_relu(
        TerminalClampUnaryReLUContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_late_mean_spp_gather_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "late_mean_spp_gather_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "MEAN",
                "MUL",
                "RESHAPE",
                "ADD",
                "CONV_2D",
                "RESIZE_BILINEAR",
                "CONCATENATION",
                "GATHER",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    for runner in [
        run_mean_mul_add_conv_layout_cleanup,
        run_spp_layout_cleanup,
        run_transpose_gather_axis_cleanup,
    ]:
        runner(
            model_ir,
            diagnostics=diagnostics,
            state_scope=state_scope,
        )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_late_layout_mean_spp_gather_constant_cast_cluster_reuses_one_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "late_layout_mean_spp_gather_constant_cast_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "MEAN",
                "MUL",
                "RESHAPE",
                "ADD",
                "CONV_2D",
                "RESIZE_BILINEAR",
                "CONCATENATION",
                "GATHER",
            ]
        ]
        + [
            OperatorIR(
                "CAST",
                [],
                [],
                options={
                    "inDataType": "INT32",
                    "outDataType": "INT64",
                },
            )
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    for runner in [
        run_layout_transpose_cleanup,
        run_mean_mul_add_conv_layout_cleanup,
        run_spp_layout_cleanup,
        run_transpose_gather_axis_cleanup,
        run_constant_input_fold_cleanup,
        run_redundant_cast_cleanup,
    ]:
        runner(
            model_ir,
            diagnostics=diagnostics,
            state_scope=state_scope,
        )

    assert refresh_count == 1
    assert len(diagnostics) == 9
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_late_spp_concat_unary_conv_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "late_spp_concat_unary_conv_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "RESIZE_BILINEAR",
                "ADD",
                "CONCATENATION",
                "MUL",
                "CONV_2D",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_late_spp_concat_unary_conv(
        LateSPPConcatUnaryConvContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 2
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[1]["metrics"]["state_built"] is False


def test_late_hard_activation_layout_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "late_hard_activation_layout_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in ["TRANSPOSE", "MUL", "ADD"]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_late_hard_activation_layout(
        LateHardActivationLayoutContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        ),
        include_layout_transpose=True,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        True,
        False,
    ]


def test_absolute_final_normalization_attention_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "absolute_final_normalization_attention_scope_preflight_only",
        operators=[
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("MEAN", [], []),
            OperatorIR("REDUCE_MAX", [], []),
            OperatorIR("CONCATENATION", [], []),
            OperatorIR("MIRROR_PAD", [], []),
            OperatorIR("CONV_2D", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_absolute_final_normalization_attention(
        AbsoluteFinalNormalizationAttentionContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 2
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[1]["metrics"]["state_built"] is False


def test_shuffle_gather_cluster_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "shuffle_gather_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "CONCATENATION",
                "RESHAPE",
                "GATHER",
                "RELU",
                "ADD",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    for runner in [
        run_two_way_channel_shuffle_cleanup,
        run_nhwc_channel_shuffle_cleanup,
        run_nchw_channel_shuffle_cleanup,
        run_transpose_gather_axis_cleanup,
        run_layout_transpose_cleanup,
        run_transpose_unary_fanout_bridge_cleanup,
        run_transpose_unary_binary_fanout_bridge_cleanup,
    ]:
        runner(
            model_ir,
            diagnostics=diagnostics,
            state_scope=state_scope,
        )

    assert refresh_count == 1
    assert len(diagnostics) == 7
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_late_nchw_shuffle_gather_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "late_nchw_shuffle_gather_scope_preflight_only",
        operators=[
            OperatorIR("RESHAPE", [], []),
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("GATHER", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    run_nchw_channel_shuffle_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_transpose_gather_axis_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 2
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[1]["metrics"]["state_built"] is False


def test_unary_fanout_cluster_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "unary_fanout_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in ["TRANSPOSE", "RELU", "ADD"]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_transpose_unary_fanout(
        TransposeUnaryFanoutContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_post_qdq_layout_unary_fanout_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "post_qdq_layout_unary_fanout_scope_preflight_only",
        operators=[
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("RELU", [], []),
            OperatorIR("ADD", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_transpose_unary_fanout(
        TransposeUnaryFanoutContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        ),
        include_layout_transpose=True,
        include_unary_passthrough=False,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_boundary_batchmatmul_unary_pair_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR("boundary_batchmatmul_unary_scope_preflight_only")
    model_ir.inputs = ["x"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1], [1]),
        "perm": TensorIR("perm", "INT32", [1], [1]),
        "transposed": TensorIR("transposed", "FLOAT32", [1], [1]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["transposed"]),
        OperatorIR("BATCH_MATMUL", [], []),
        OperatorIR("ADD", [], []),
    ]
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_boundary_batchmatmul_unary(
        BoundaryBatchMatMulUnaryContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 4
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        False,
        False,
        False,
    ]


def test_channel_slice_pad_mul_pair_reuses_one_pass_state(monkeypatch) -> None:
    model_ir = ModelIR(
        "channel_slice_pad_mul_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "MUL",
                "ADD",
                "SLICE",
                "SLICE",
                "PAD",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    run_channel_slice_pad_mul(
        ChannelSlicePadMulContext(
            model_ir=model_ir,
            layout_state=None,
            diagnostics=diagnostics,
        )
    )

    assert refresh_count == 1
    assert len(diagnostics) == 4
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        True,
        True,
        False,
    ]


def test_singleton_reshape_layout_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "singleton_reshape_layout_scope_preflight_only",
        operators=[
            OperatorIR(op_type, [], [])
            for op_type in [
                "TRANSPOSE",
                "RESHAPE",
                "RESHAPE",
                "MAX_POOL_2D",
                "CONCATENATION",
                "SQUEEZE",
                "RELU",
                "MEAN",
                "LOGISTIC",
                "MUL",
                "ADD",
            ]
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    run_layout_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_singleton_channel_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_duplicate_fanout_cleanup(
        model_ir,
        include_transpose=False,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_singleton_reshape_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_singleton_maxpool_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_flatten_concat_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_consecutive_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_squeeze_reshape_identity_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_singleton_spatial_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_multi_branch_gate_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 13
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert all(
        event["metrics"]["state_built"] is False
        for event in diagnostics[1:]
    )


def test_singleton_consecutive_reshape_cluster_reuses_one_pass_state(
    monkeypatch,
) -> None:
    model_ir = ModelIR(
        "singleton_consecutive_reshape_scope_preflight_only",
        operators=[
            OperatorIR("TRANSPOSE", [], []),
            OperatorIR("RESHAPE", [], []),
            OperatorIR("RESHAPE", [], []),
        ],
    )
    diagnostics: list[dict] = []
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    state_scope = ModelIRPassStateScope(model_ir)

    run_singleton_channel_transpose_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_duplicate_fanout_cleanup(
        model_ir,
        include_transpose=False,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )
    run_consecutive_reshape_cleanup(
        model_ir,
        diagnostics=diagnostics,
        state_scope=state_scope,
    )

    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert [event["metrics"]["state_built"] for event in diagnostics] == [
        True,
        False,
        False,
    ]


def test_channel_slice_merge_guard_rejects_incomplete_prefix_before_snapshot(
    monkeypatch,
) -> None:
    model_ir = ModelIR("incomplete_channel_slice_merge")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": TensorIR("x", "FLOAT32", [1, 2, 2, 4], [1, 2, 2, 4]),
        "perm": TensorIR(
            "perm",
            "INT32",
            [4],
            [4],
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "t": TensorIR("t", "FLOAT32", [1, 4, 2, 2], [1, 4, 2, 2]),
        "a_begin": TensorIR(
            "a_begin", "INT32", [4], [4], data=np.asarray([0, 0, 0, 0])
        ),
        "a_size": TensorIR(
            "a_size", "INT32", [4], [4], data=np.asarray([1, 2, 2, 2])
        ),
        "b_begin": TensorIR(
            "b_begin", "INT32", [4], [4], data=np.asarray([0, 2, 0, 0])
        ),
        "b_size": TensorIR(
            "b_size", "INT32", [4], [4], data=np.asarray([1, 2, 2, 2])
        ),
        "a": TensorIR("a", "FLOAT32", [1, 2, 2, 2], [1, 2, 2, 2]),
        "b": TensorIR("b", "FLOAT32", [1, 2, 2, 2], [1, 2, 2, 2]),
        "scale": TensorIR(
            "scale", "FLOAT32", [1], [1], data=np.asarray([1.0], dtype=np.float32)
        ),
        "m": TensorIR("m", "FLOAT32", [1, 2, 2, 2], [1, 2, 2, 2]),
        "bias": TensorIR(
            "bias", "FLOAT32", [1], [1], data=np.asarray([0.0], dtype=np.float32)
        ),
        "y": TensorIR("y", "FLOAT32", [1, 2, 2, 2], [1, 2, 2, 2]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["t"]),
        OperatorIR("SLICE", ["t", "a_begin", "a_size"], ["a"]),
        OperatorIR("SLICE", ["t", "b_begin", "b_size"], ["b"]),
        OperatorIR("MUL", ["a", "scale"], ["m"]),
        OperatorIR("ADD", ["m", "bias"], ["y"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []

    stats = run_channel_slice_merge_layout_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats == {
        "optimized_transpose_channel_slice_dual_add_bridges_strict": 0,
        "optimized_transpose_slice_muladd_conv_mergeadd_strict": 0,
        "optimized_transpose_slice_muladd_mergeadd_posttranspose_strict": 0,
    }
    assert refresh_count == 1
    assert len(diagnostics) == 3
    assert all(event["status"] == "skipped" for event in diagnostics)
    assert all(event["metrics"]["state_built"] is True for event in diagnostics)
    assert all(event["metrics"]["snapshot_count"] == 0 for event in diagnostics)


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
