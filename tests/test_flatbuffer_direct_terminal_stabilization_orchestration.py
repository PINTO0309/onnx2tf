from __future__ import annotations

import ast
from pathlib import Path

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    terminal_stabilization_orchestration as owner_module,
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


def test_terminal_stabilization_context_owner_preserves_raw_contracts() -> None:
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

    assert not any(
        _assignment_name(statement) in {
            result_name for result_name, *_ in RAW_CONTRACTS
        }
        for statement in body
    )


def test_terminal_stabilization_lowerer_retains_one_composite_result() -> None:
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


def test_terminal_stabilization_runtime_preserves_identity_order_and_tuple(
    monkeypatch,
) -> None:
    model_ir = ModelIR("terminal_stabilization")
    layout_state = LayoutState.from_model_ir(model_ir)
    context = owner_module.TerminalStabilizationContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=[],
    )
    expected = (
        {"repaired_rank4_channelwise_broadcast_constants": 1},
        {"coalesced_static_high_rank_binary_operators": 2},
        {"realigned_dynamic_boundary_shape_signature_map": 3},
    )
    calls: list[str] = []

    def convergence(received_model_ir):
        assert received_model_ir is model_ir
        calls.append("convergence")
        return expected[0]

    def coalesce(received_model_ir, *, layout_state):
        assert received_model_ir is model_ir
        assert layout_state is context.layout_state
        calls.append("coalesce")
        return expected[1]

    def realign(received_model_ir):
        assert received_model_ir is model_ir
        calls.append("realign")
        return expected[2]

    monkeypatch.setattr(
        owner_module,
        "run_indexed_binary_layout_convergence",
        convergence,
    )
    monkeypatch.setattr(
        owner_module,
        "coalesce_static_high_rank_binary_operators",
        coalesce,
    )
    monkeypatch.setattr(
        owner_module,
        "realign_dynamic_boundary_shape_signature_map",
        realign,
    )

    result = owner_module.run_terminal_stabilization_cleanup(context)

    assert calls == ["convergence", "coalesce", "realign"]
    assert result == expected
    assert all(actual is wanted for actual, wanted in zip(result, expected))
