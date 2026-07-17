from __future__ import annotations

import ast
from collections import Counter
from dataclasses import FrozenInstanceError, fields, is_dataclass
from importlib import import_module
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
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
    (
        "se_fc_gather_channel_fanout_orchestration",
        "SEFCGatherChannelFanoutContext",
    ),
    (
        "singleton_consecutive_reshape_orchestration",
        "SingletonConsecutiveReshapeContext",
    ),
    ("singleton_reshape_orchestration", "SingletonReshapeContext"),
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


def test_shared_context_construction_sites_have_no_hidden_policy_state() -> None:
    context_names = {context_name for _, context_name in SHARED_CONTEXT_TYPES}
    construction_sites: list[tuple[str, ast.Call]] = []
    source_paths = (LOWERER_PATH, *sorted(PASSES_ROOT.glob("*orchestration.py")))
    for path in source_paths:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in context_names
            ):
                continue
            construction_sites.append((path.name, node))

    assert Counter(
        call.func.id
        for _, call in construction_sites
        if isinstance(call.func, ast.Name)
    ) == Counter(
        {
            **{context_name: 1 for _, context_name in SHARED_CONTEXT_TYPES},
            "ConstantFoldCastContext": 2,
        }
    )
    assert len(construction_sites) == 22
    for _, call in construction_sites:
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
        source = contract["model_ir"]
        if source == "model_ir":
            assert contract == {
                "model_ir": "model_ir",
                "layout_state": "session.layout_state",
                "diagnostics": "session.diagnostics",
            }
        elif source == "target_model_ir":
            assert contract == {
                "model_ir": "target_model_ir",
                "layout_state": "target_layout_state",
                "diagnostics": "session.diagnostics",
            }
        else:
            assert contract == {
                "model_ir": "context.model_ir",
                "layout_state": "context.layout_state",
                "diagnostics": "context.diagnostics",
            }


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
