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
    / "terminal_stabilization_orchestration.py"
)
OWNER = "run_terminal_stabilization_cleanup"
RESULT_NAME = "_final_terminal_stabilization_results"
RAW_CONTRACTS = (
    (
        "_final_binary_layout_convergence_stats",
        "_run_indexed_binary_layout_convergence",
        ("model_ir",),
        {},
    ),
    (
        "_final_high_rank_binary_stats",
        "coalesce_static_high_rank_binary_operators",
        ("model_ir",),
        {"layout_state": "session.layout_state"},
    ),
    (
        "_final_dynamic_boundary_signature_stats",
        "_realign_dynamic_boundary_shape_signature_map",
        ("model_ir",),
        {},
    ),
)
OWNER_CONTRACTS = (
    (
        "run_indexed_binary_layout_convergence",
        ("context.model_ir",),
        {},
    ),
    (
        "coalesce_static_high_rank_binary_operators",
        ("context.model_ir",),
        {"layout_state": "context.layout_state"},
    ),
    (
        "realign_dynamic_boundary_shape_signature_map",
        ("context.model_ir",),
        {},
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
    call: ast.expr,
    *,
    name: str,
    arguments: tuple[str, ...],
    keywords: dict[str, str],
) -> None:
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == name
    assert tuple(ast.unparse(argument) for argument in call.args) == arguments
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == keywords


def _assert_terminal_successor(body: list[ast.stmt], index: int) -> None:
    record = body[index]
    assert isinstance(record, ast.Expr)
    assert ast.unparse(record) == (
        "session.record_phase_result("
        "'layout_validation.primary.terminal', "
        "run_topology_layout_validation(model_ir))"
    )
    terminal = body[index + 1]
    assert isinstance(terminal, ast.Return)
    assert ast.unparse(terminal.value) == "_finalize_model_ir(model_ir)"


def test_terminal_stabilization_raw_lowerer_contract_is_fixed() -> None:
    body = _lowerer().body
    indices = []
    for result_name, owner_name, arguments, keywords in RAW_CONTRACTS:
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
            keywords=keywords,
        )
        indices.append(index)

    assert indices == list(range(indices[0], indices[0] + len(indices)))
    _assert_terminal_successor(body, indices[-1] + 1)


@pytest.mark.xfail(
    strict=True,
    reason="terminal stabilization still has three lowerer-owned results",
)
def test_terminal_stabilization_has_one_context_owner() -> None:
    owner = _function(OWNER_PATH, OWNER)
    assert [argument.arg for argument in owner.args.args] == ["context"]
    assert len(owner.body) == 1
    terminal = owner.body[0]
    assert isinstance(terminal, ast.Return)
    assert isinstance(terminal.value, ast.Tuple)
    assert len(terminal.value.elts) == len(OWNER_CONTRACTS)
    for expression, (name, arguments, keywords) in zip(
        terminal.value.elts,
        OWNER_CONTRACTS,
        strict=True,
    ):
        _assert_call(
            expression,
            name=name,
            arguments=arguments,
            keywords=keywords,
        )

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
        keywords={},
    )
    _assert_terminal_successor(body, index + 1)
