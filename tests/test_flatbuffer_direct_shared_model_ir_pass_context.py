from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, fields, is_dataclass
from importlib import import_module
from pathlib import Path

import onnx
import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.session import ConversionSession
from onnx2tf.tflite_builder.ir import ModelIR


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
PASSES_ROOT = REPO_ROOT / "onnx2tf" / "tflite_builder" / "passes"
SHARED_CONTEXT_TYPES = (
    (
        "absolute_final_normalization_attention_orchestration",
        "AbsoluteFinalNormalizationAttentionContext",
    ),
    (
        "boundary_batchmatmul_unary_orchestration",
        "BoundaryBatchMatMulUnaryContext",
    ),
    ("channel_shuffle_gather_orchestration", "ChannelShuffleGatherContext"),
    ("channel_slice_pad_mul_orchestration", "ChannelSlicePadMulContext"),
    ("constant_fold_cast_orchestration", "ConstantFoldCastContext"),
    (
        "duplicate_quantized_prelu_orchestration",
        "DuplicateQuantizedPReLUContext",
    ),
    ("gate_layout_orchestration", "GateLayoutContext"),
    ("late_dequant_unary_fanout_orchestration", "LateDequantUnaryFanoutContext"),
    (
        "late_hard_activation_layout_orchestration",
        "LateHardActivationLayoutContext",
    ),
    (
        "late_layout_mean_spp_gather_constant_cast_orchestration",
        "LateLayoutMeanSPPGatherConstantCastContext",
    ),
    ("late_spp_concat_unary_conv_orchestration", "LateSPPConcatUnaryConvContext"),
    ("mean_attention_orchestration", "MeanAttentionContext"),
    ("qkv_attention_orchestration", "QKVAttentionContext"),
    ("qlinear_recovery_orchestration", "QLinearRecoveryContext"),
    ("quantized_recovery_orchestration", "QuantizedRecoveryContext"),
    (
        "se_fc_gather_channel_fanout_orchestration",
        "SEFCGatherChannelFanoutContext",
    ),
    (
        "sinet_preadd_resize_recovery_orchestration",
        "SINetPreaddResizeRecoveryContext",
    ),
    (
        "singleton_consecutive_reshape_orchestration",
        "SingletonConsecutiveReshapeContext",
    ),
    ("singleton_reshape_orchestration", "SingletonReshapeContext"),
    (
        "terminal_affine_concat_split_recovery_orchestration",
        "TerminalAffineConcatSplitRecoveryContext",
    ),
    ("terminal_boundary_layout_orchestration", "TerminalBoundaryLayoutContext"),
    (
        "terminal_clamp_unary_relu_orchestration",
        "TerminalClampUnaryReLUContext",
    ),
    (
        "terminal_singleton_maxpool_reshape_orchestration",
        "TerminalSingletonMaxPoolReshapeContext",
    ),
    (
        "transpose_unary_fanout_orchestration",
        "TransposeUnaryFanoutContext",
    ),
    (
        "very_late_gather_constant_normalization_orchestration",
        "VeryLateGatherConstantNormalizationContext",
    ),
)
MAIN_SHARED_CONTEXT_NAMES = (
    "boundary_batchmatmul_unary_context",
    "channel_shuffle_gather_context",
    "channel_slice_pad_mul_context",
    "duplicate_quantized_prelu_context",
    "gate_layout_context",
    "late_dequant_unary_fanout_context",
    "late_hard_activation_layout_context",
    "late_layout_mean_spp_gather_constant_cast_context",
    "late_spp_concat_unary_conv_context",
    "mean_attention_context",
    "qkv_attention_context",
    "qlinear_recovery_context",
    "quantized_recovery_context",
    "sinet_preadd_resize_recovery_context",
    "singleton_reshape_context",
    "terminal_affine_concat_split_recovery_context",
    "terminal_boundary_layout_context",
    "terminal_clamp_unary_relu_context",
    "terminal_singleton_maxpool_reshape_context",
    "transpose_unary_fanout_context",
    "very_late_gather_constant_normalization_context",
)


def _expression_path(expression: ast.expr) -> object:
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        return f"{_expression_path(expression.value)}.{expression.attr}"
    if isinstance(expression, ast.Constant):
        return expression.value
    return type(expression).__name__


@pytest.mark.parametrize(("module_name", "context_name"), SHARED_CONTEXT_TYPES)
def test_orchestration_contexts_share_one_frozen_identity_contract(
    module_name: str,
    context_name: str,
) -> None:
    module = import_module(f"onnx2tf.tflite_builder.passes.{module_name}")
    context_type = getattr(module, context_name)
    model_ir = ModelIR(f"shared_context_{module_name}")
    layout_state = LayoutState.from_model_ir(model_ir)
    diagnostics: list[dict] = []

    assert context_type is ModelIRPassContext
    assert is_dataclass(context_type)
    assert tuple(field.name for field in fields(context_type)) == (
        "model_ir",
        "layout_state",
        "diagnostics",
    )

    context = context_type(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    assert context.model_ir is model_ir
    assert context.layout_state is layout_state
    assert context.diagnostics is diagnostics
    with pytest.raises(FrozenInstanceError):
        context.model_ir = ModelIR("replacement")


def test_conversion_session_owns_the_main_model_ir_pass_context() -> None:
    onnx_model = onnx.helper.make_model(
        onnx.helper.make_graph([], "shared_context_graph", [], [])
    )
    model_ir = ModelIR("shared_context_session")
    session = ConversionSession(
        onnx_model=onnx_model,
        model_ir=model_ir,
        shape_map={},
        dtype_map={},
        constants={},
    )

    assert session.model_ir_pass_context.model_ir is model_ir
    assert session.model_ir_pass_context.layout_state is session.layout_state
    assert session.model_ir_pass_context.diagnostics is session.diagnostics
    session.diagnostics.append({"code": "identity_probe"})
    assert session.model_ir_pass_context.diagnostics == [{"code": "identity_probe"}]


def test_main_and_target_context_wiring_preserves_identity_boundaries() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    assignments = {
        target.id: _expression_path(statement.value)
        for statement in lowerer.body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance((target := statement.targets[0]), ast.Name)
    }
    assert assignments["shared_model_ir_pass_context"] == (
        "session.model_ir_pass_context"
    )
    assert {name: assignments[name] for name in MAIN_SHARED_CONTEXT_NAMES} == {
        name: "shared_model_ir_pass_context" for name in MAIN_SHARED_CONTEXT_NAMES
    }

    target_constructions_by_helper = {
        helper.name: calls
        for helper in lowerer.body
        if isinstance(helper, ast.FunctionDef)
        and (
            calls := [
                node
                for node in ast.walk(helper)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "ModelIRPassContext"
            ]
        )
    }
    assert set(target_constructions_by_helper) == {
        "_run_se_fc_gather_channel_fanout_pass_cluster",
        "_run_sinet_se_fc_gather_summary",
        "_run_precision_cleanup_sequence",
        "_run_singleton_consecutive_reshape_pass_cluster",
    }
    assert all(
        len(calls) == 1
        for calls in target_constructions_by_helper.values()
    )
    for calls in target_constructions_by_helper.values():
        call = calls[0]
        assert call.args == []
        assert [keyword.arg for keyword in call.keywords] == [
            "model_ir",
            "layout_state",
            "diagnostics",
        ]
        contract = {
            str(keyword.arg): _expression_path(keyword.value)
            for keyword in call.keywords
        }
        assert contract == {
            "model_ir": "target_model_ir",
            "layout_state": "target_layout_state",
            "diagnostics": "session.diagnostics",
        }


def test_composed_constant_fold_cast_builders_reuse_the_parent_context() -> None:
    parent_modules = (
        "late_layout_mean_spp_gather_constant_cast_orchestration",
        "very_late_gather_constant_normalization_orchestration",
    )
    for module_name in parent_modules:
        path = PASSES_ROOT / f"{module_name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_constant_fold_cast_invocations"
        ]
        assert len(calls) == 1
        assert len(calls[0].args) == 1
        assert isinstance(calls[0].args[0], ast.Name)
        assert calls[0].args[0].id == "context"


def test_shared_context_modules_do_not_import_the_lowerer() -> None:
    for module_name, _ in SHARED_CONTEXT_TYPES:
        path = PASSES_ROOT / f"{module_name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imported_modules = {
            str(node.module)
            for node in tree.body
            if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert "onnx2tf.tflite_builder.lower_from_onnx2tf" not in imported_modules
