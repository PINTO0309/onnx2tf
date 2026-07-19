from __future__ import annotations

import ast
from pathlib import Path

import pytest


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


def test_absolute_final_cleanup_raw_lowerer_contract_is_fixed() -> None:
    body = _lowerer().body
    indices = []
    for result_name, owner_name, arguments in RAW_CONTRACTS:
        matches = [
            (index, statement)
            for index, statement in enumerate(body)
            if _assignment_name(statement) == result_name
        ]
        assert len(matches) == 1
        index, statement = matches[0]
        assert isinstance(statement, ast.Assign)
        _assert_call(
            statement.value,
            name=owner_name,
            arguments=arguments,
        )
        indices.append(index)

    assert indices == list(range(indices[0], indices[0] + len(indices)))
    _assert_refresh_successor(body, indices[-1] + 1)


@pytest.mark.xfail(
    strict=True,
    reason="absolute-final cleanup still has three lowerer-owned results",
)
def test_absolute_final_cleanup_has_one_context_owner() -> None:
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
