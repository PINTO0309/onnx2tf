from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    absolute_final_cleanup_orchestration as owner_module,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_cleanup_orchestration.py"
)
OWNER = "run_absolute_final_cleanup"
RESULT_NAME = "_absolute_final_cleanup_results"
RAW_CONTRACTS = (
    (
        "_absolute_final_boundary_signature_results",
        "run_boundary_shape_signature_cleanup",
        ("model_ir",),
    ),
    (
        "_absolute_final_affine_instancenorm_results",
        "run_absolute_final_affine_instancenorm_cleanup",
        ("shared_model_ir_pass_context",),
    ),
    (
        "_absolute_final_normalization_attention_rank1_results",
        "run_absolute_final_normalization_attention_rank1_cleanup",
        ("shared_model_ir_pass_context",),
    ),
)
OWNER_CONTRACTS = (
    (
        "run_boundary_shape_signature_cleanup",
        ("context.model_ir",),
    ),
    (
        "run_absolute_final_affine_instancenorm_cleanup",
        ("context",),
    ),
    (
        "run_absolute_final_normalization_attention_rank1_cleanup",
        ("context",),
    ),
)


def _function(path: Path, name: str) -> ast.FunctionDef:
    return next(
        node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def _lowerer() -> ast.FunctionDef:
    return _function(LOWERER_PATH, "lower_onnx_to_ir")


def _assignment_name(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _assert_call(
    expression: ast.expr,
    *,
    name: str,
    arguments: tuple[str, ...],
) -> None:
    assert isinstance(expression, ast.Call)
    assert isinstance(expression.func, ast.Name)
    assert expression.func.id == name
    assert tuple(
        ast.unparse(argument) for argument in expression.args
    ) == arguments
    assert expression.keywords == []


def _assert_refresh_successor(body: list[ast.stmt], index: int) -> None:
    statement = body[index]
    assert isinstance(statement, ast.Expr)
    assert ast.unparse(statement) == (
        "session.record_phase_result("
        "'topology_layout.primary.absolute_final', "
        "run_topology_layout_refresh(model_ir))"
    )


def test_absolute_final_cleanup_context_owner_preserves_raw_contracts() -> None:
    owner = _function(OWNER_PATH, OWNER)
    assert [argument.arg for argument in owner.args.args] == ["context"]
    assert len(owner.body) == 1
    terminal = owner.body[0]
    assert isinstance(terminal, ast.Return)
    assert isinstance(terminal.value, ast.Tuple)
    assert len(terminal.value.elts) == len(OWNER_CONTRACTS)
    for expression, (name, arguments) in zip(
        terminal.value.elts,
        OWNER_CONTRACTS,
        strict=True,
    ):
        _assert_call(expression, name=name, arguments=arguments)

    body = _lowerer().body
    matches = [
        (index, statement)
        for index, statement in enumerate(body)
        if _assignment_name(statement) == RESULT_NAME
    ]
    assert len(matches) == 1
    index, statement = matches[0]
    assert isinstance(statement, ast.Assign)
    _assert_call(
        statement.value,
        name=OWNER,
        arguments=("shared_model_ir_pass_context",),
    )
    _assert_refresh_successor(body, index + 1)

    assert not any(
        _assignment_name(statement) in {
            result_name for result_name, *_ in RAW_CONTRACTS
        }
        for statement in body
    )


def test_absolute_final_cleanup_lowerer_retains_one_composite_result() -> None:
    body = _lowerer().body
    matches = [
        (index, statement)
        for index, statement in enumerate(body)
        if _assignment_name(statement) == RESULT_NAME
    ]
    assert len(matches) == 1
    index, statement = matches[0]
    assert isinstance(statement, ast.Assign)
    _assert_call(
        statement.value,
        name=OWNER,
        arguments=("shared_model_ir_pass_context",),
    )
    _assert_refresh_successor(body, index + 1)


def test_absolute_final_cleanup_runtime_preserves_identity_order_and_nesting(
    monkeypatch,
) -> None:
    model_ir = ModelIR("absolute_final_cleanup")
    context = owner_module.AbsoluteFinalCleanupContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    expected = (
        (
            {"realigned_dynamic_boundary_shape_signature_map": 1},
            {"sanitized_static_shape_signature_consistency": 2},
        ),
        (
            {"optimized_affine_post_add": 3},
            {"optimized_instancenorm_post_bias": 4},
        ),
        (
            (
                {"optimized_normalization_pad": 5},
                {"optimized_mixed_attention": 6},
            ),
            {"rewritten_dynamic_rank1_reshape": 7},
        ),
    )
    calls: list[str] = []

    def boundary_signature(received_model_ir):
        assert received_model_ir is context.model_ir
        calls.append("boundary_signature")
        return expected[0]

    def affine_instancenorm(received_context):
        assert received_context is context
        calls.append("affine_instancenorm")
        return expected[1]

    def normalization_attention_rank1(received_context):
        assert received_context is context
        calls.append("normalization_attention_rank1")
        return expected[2]

    monkeypatch.setattr(
        owner_module,
        "run_boundary_shape_signature_cleanup",
        boundary_signature,
    )
    monkeypatch.setattr(
        owner_module,
        "run_absolute_final_affine_instancenorm_cleanup",
        affine_instancenorm,
    )
    monkeypatch.setattr(
        owner_module,
        "run_absolute_final_normalization_attention_rank1_cleanup",
        normalization_attention_rank1,
    )

    result = owner_module.run_absolute_final_cleanup(context)

    assert calls == [
        "boundary_signature",
        "affine_instancenorm",
        "normalization_attention_rank1",
    ]
    assert result == expected
    assert all(actual is wanted for actual, wanted in zip(result, expected))
