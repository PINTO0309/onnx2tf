from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassStateScope
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes.boundary_batchmatmul_unary_orchestration import (
    BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS,
    build_boundary_batchmatmul_unary_invocations,
    run_boundary_batchmatmul_unary,
)
from onnx2tf.tflite_builder.passes.duplicate_quantized_prelu_orchestration import (
    DUPLICATE_QUANTIZED_PRELU_PASS_IDS,
    build_duplicate_quantized_prelu_invocations,
    run_duplicate_quantized_prelu,
)
from onnx2tf.tflite_builder.passes.layout_attention_quantized_suffix_orchestration import (
    LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS,
)
from onnx2tf.tflite_builder.passes.layout_recovery_orchestration import (
    LAYOUT_RECOVERY_PASS_IDS,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
DUPLICATE_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "duplicate_quantized_prelu_orchestration.py"
)
BOUNDARY_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "boundary_batchmatmul_unary_orchestration.py"
)
DUPLICATE_RUNNER = "run_duplicate_quantized_prelu"
DUPLICATE_HELPER = "_run_duplicate_quantized_prelu_pass_cluster"
BOUNDARY_RUNNER = "run_boundary_batchmatmul_unary"
BOUNDARY_HELPER = "_run_boundary_batchmatmul_unary_layout_pass_cluster"
DUPLICATE_SCHEMAS = {
    False: (
        {"removed_duplicate_reshape_fanout": 0},
        {
            "removed_transpose_dequant_prelu_quantize_bridges": 0,
            "removed_transpose_dequant_prelu_transpose_bridges": 0,
            "folded_dequant_prelu_quantize_chains": 0,
            "folded_dequant_prelu_depthwise_quantize_chains": 0,
        },
    ),
    True: (
        {
            "removed_duplicate_reshape_fanout": 0,
            "removed_duplicate_transpose_fanout": 0,
        },
        {
            "removed_transpose_dequant_prelu_quantize_bridges": 0,
            "removed_transpose_dequant_prelu_transpose_bridges": 0,
            "folded_dequant_prelu_quantize_chains": 0,
            "folded_dequant_prelu_depthwise_quantize_chains": 0,
        },
    ),
}
BOUNDARY_SCHEMA = (
    {"rewritten_boundary_input_transpose_batchmatmul_chains": 0},
    {
        "rewritten_leading_input_transpose_passthrough_chains": 0,
        "rewritten_asin_transpose_passthrough_chains": 0,
        "rewritten_erf_transpose_passthrough_chains": 0,
    },
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer_functions() -> dict[str, ast.FunctionDef]:
    lowerer = _functions(LOWERER_PATH)["lower_onnx_to_ir"]
    return {
        node.name: node
        for node in lowerer.body
        if isinstance(node, ast.FunctionDef)
    }


def _context(name: str) -> ModelIRPassContext:
    model_ir = ModelIR(name)
    return ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )


def test_void_orchestration_child_schemas_and_scopes_are_explicit() -> None:
    assert DUPLICATE_QUANTIZED_PRELU_PASS_IDS == (
        "run_duplicate_fanout_cleanup",
        "run_quantized_prelu_cleanup",
    )
    for include_transpose, expected in DUPLICATE_SCHEMAS.items():
        context = _context(f"void_duplicate_{include_transpose}")
        invocations = build_duplicate_quantized_prelu_invocations(
            context,
            include_transpose=include_transpose,
        )
        assert tuple(invocation.run() for invocation in invocations) == expected
        scopes = [dict(invocation.keyword_args)["state_scope"] for invocation in invocations]
        assert isinstance(scopes[0], ModelIRPassStateScope)
        assert all(scope is scopes[0] for scope in scopes)

    assert BOUNDARY_BATCHMATMUL_UNARY_PASS_IDS == (
        "run_boundary_input_batchmatmul_cleanup",
        "run_input_unary_passthrough_cleanup",
    )
    boundary_context = _context("void_boundary")
    boundary_invocations = build_boundary_batchmatmul_unary_invocations(
        boundary_context
    )
    assert tuple(
        invocation.run() for invocation in boundary_invocations
    ) == BOUNDARY_SCHEMA
    boundary_scopes = [
        dict(invocation.keyword_args)["state_scope"]
        for invocation in boundary_invocations
    ]
    assert isinstance(boundary_scopes[0], ModelIRPassStateScope)
    assert all(scope is boundary_scopes[0] for scope in boundary_scopes)


def test_void_runner_helpers_and_parent_routes_are_explicit() -> None:
    lowerer_functions = _lowerer_functions()
    contracts = (
        (DUPLICATE_PATH, DUPLICATE_RUNNER, DUPLICATE_HELPER),
        (BOUNDARY_PATH, BOUNDARY_RUNNER, BOUNDARY_HELPER),
    )
    for path, runner_name, helper_name in contracts:
        runner = _functions(path)[runner_name]
        helper = lowerer_functions[helper_name]
        assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
        assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
        assert len(runner.body) == 1
        assert isinstance(runner.body[0], ast.Return)
        assert len(helper.body) == 1
        assert isinstance(helper.body[0], ast.Return)

    assert LAYOUT_ATTENTION_QUANTIZED_SUFFIX_PASS_IDS[5] == DUPLICATE_HELPER
    assert LAYOUT_RECOVERY_PASS_IDS[1] == BOUNDARY_HELPER


def test_void_orchestrations_propagate_ordered_results() -> None:
    for include_transpose, expected in DUPLICATE_SCHEMAS.items():
        assert run_duplicate_quantized_prelu(
            _context(f"propagate_duplicate_{include_transpose}"),
            include_transpose=include_transpose,
        ) == expected
    assert run_boundary_batchmatmul_unary(
        _context("propagate_boundary")
    ) == BOUNDARY_SCHEMA

    lowerer_functions = _lowerer_functions()
    for path, runner_name, helper_name in (
        (DUPLICATE_PATH, DUPLICATE_RUNNER, DUPLICATE_HELPER),
        (BOUNDARY_PATH, BOUNDARY_RUNNER, BOUNDARY_HELPER),
    ):
        runner = _functions(path)[runner_name]
        helper = lowerer_functions[helper_name]
        assert ast.unparse(runner.returns) == "Tuple[Dict[str, int], ...]"
        assert ast.unparse(helper.returns) == "Tuple[Dict[str, int], ...]"
        assert isinstance(runner.body[0], ast.Return)
        assert isinstance(helper.body[0], ast.Return)
