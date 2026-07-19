from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import precision_cleanup_orchestration


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = (
    REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
)
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "precision_cleanup_orchestration.py"
)
REWRITE_OWNER = "_rewrite_constant_divisors_to_multiplicative_reciprocals"
CONSECUTIVE_OWNER = "run_consecutive_mul_constants_cleanup"
RESTORE_OWNER = "_restore_precision_sensitive_reciprocal_divisions"
SEQUENCE_OWNER = "run_precision_cleanup_sequence"
SEQUENCE_HELPER = "_run_precision_cleanup_sequence"
SITE_CONTRACTS = (
    (
        (
            "_fallback_precision_div_rewrite_stats",
            "_fallback_precision_consecutive_mul_stats",
            "_fallback_precision_div_restore_stats",
        ),
        "_fallback_precision_cleanup_results",
        "fallback_ir",
        "None",
        "topology.fallback.post_placeholder",
        "_fallback_unbound_repair_stats",
    ),
    (
        (
            "_final_precision_div_rewrite_stats",
            "_final_precision_consecutive_mul_stats",
            "_final_precision_div_restore_stats",
        ),
        "_final_precision_cleanup_results",
        "model_ir",
        "session.layout_state",
        None,
        None,
    ),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _containing_body(root: ast.AST, target: ast.stmt) -> list[ast.stmt]:
    for node in ast.walk(root):
        for _, value in ast.iter_fields(node):
            if isinstance(value, list) and target in value:
                return value
    raise AssertionError("statement is not contained by an AST body")


def test_precision_cleanup_duplicate_raw_sequences_are_fixed() -> None:
    lowerer = _lowerer()
    old_targets = {
        target
        for site_targets, _, _, _, _, _ in SITE_CONTRACTS
        for target in site_targets
    }
    for (
        _,
        sequence_target,
        model_name,
        layout_expression,
        predecessor_phase,
        successor_target,
    ) in SITE_CONTRACTS:
        sequence = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == sequence_target
        )
        body = _containing_body(lowerer, sequence)
        index = body.index(sequence)
        assert isinstance(sequence, ast.Assign)
        assert ast.unparse(sequence.value) == (
            f"{SEQUENCE_HELPER}({model_name}, {layout_expression})"
        )

        if predecessor_phase is not None:
            assert ast.unparse(body[index - 1]) == (
                "session.record_phase_result("
                f"'{predecessor_phase}', _topologically_sort_operators("
                f"{model_name}))"
            )
        if successor_target is not None:
            assert _single_target(body[index + 1]) == successor_target
        else:
            assert ast.unparse(body[index + 1]) == (
                "_set_post_progress_desc('topological sort')"
            )

    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for node in ast.walk(lowerer)
    )
    raw_calls = [
        node.func.id
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {REWRITE_OWNER, CONSECUTIVE_OWNER, RESTORE_OWNER}
    ]
    assert raw_calls.count(REWRITE_OWNER) == 0
    assert raw_calls.count(CONSECUTIVE_OWNER) == 1
    assert raw_calls.count(RESTORE_OWNER) == 0


def test_precision_cleanup_uses_one_shared_ordered_sequence_owner() -> None:
    owner = _functions(OWNER_PATH)[SEQUENCE_OWNER]
    raw_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        in {REWRITE_OWNER, CONSECUTIVE_OWNER, RESTORE_OWNER}
    ]
    assert raw_calls == [REWRITE_OWNER, CONSECUTIVE_OWNER, RESTORE_OWNER]

    lowerer = _lowerer()
    helper = next(
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.FunctionDef) and node.name == SEQUENCE_HELPER
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == SEQUENCE_OWNER
        for node in ast.walk(helper)
    )
    old_targets = {
        target
        for site_targets, _, _, _, _, _ in SITE_CONTRACTS
        for target in site_targets
    }
    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for node in ast.walk(lowerer)
    )

    for (
        _,
        sequence_target,
        model_name,
        layout_expression,
        predecessor_phase,
        successor_target,
    ) in SITE_CONTRACTS:
        sequence = next(
            statement
            for statement in ast.walk(lowerer)
            if _single_target(statement) == sequence_target
        )
        body = _containing_body(lowerer, sequence)
        index = body.index(sequence)
        assert isinstance(sequence, ast.Assign)
        assert ast.unparse(sequence.value) == (
            f"{SEQUENCE_HELPER}({model_name}, {layout_expression})"
        )
        if predecessor_phase is not None:
            assert ast.unparse(body[index - 1]) == (
                "session.record_phase_result("
                f"'{predecessor_phase}', _topologically_sort_operators("
                f"{model_name}))"
            )
        if successor_target is not None:
            assert _single_target(body[index + 1]) == successor_target
        else:
            assert ast.unparse(body[index + 1]) == (
                "_set_post_progress_desc('topological sort')"
            )


@pytest.mark.parametrize("use_layout_state", (False, True))
def test_precision_cleanup_sequence_preserves_order_context_and_raw_schemas(
    monkeypatch: pytest.MonkeyPatch,
    use_layout_state: bool,
) -> None:
    model_ir = ModelIR("precision_cleanup_sequence")
    layout_state = (
        LayoutState.from_model_ir(model_ir) if use_layout_state else None
    )
    diagnostics: list[dict[str, object]] = []
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=layout_state,
        diagnostics=diagnostics,
    )
    results = (
        {"rewritten_constant_div_to_mul": 2},
        {"optimized_fold_consecutive_mul_constants_chains": 3},
        {"restored_precision_sensitive_reciprocal_divisions": 4},
    )
    events: list[tuple[str, ModelIR, dict[str, object]]] = []

    def _recorder(name: str, result: dict[str, int]):
        def _run(candidate: ModelIR, **kwargs: object) -> dict[str, int]:
            events.append((name, candidate, dict(kwargs)))
            return dict(result)

        return _run

    for name, result in zip(
        (REWRITE_OWNER, CONSECUTIVE_OWNER, RESTORE_OWNER),
        results,
        strict=True,
    ):
        monkeypatch.setattr(
            precision_cleanup_orchestration,
            name,
            _recorder(name, result),
        )

    assert precision_cleanup_orchestration.run_precision_cleanup_sequence(
        context
    ) == results
    layout_keywords: dict[str, object] = (
        {} if layout_state is None else {"layout_state": layout_state}
    )
    assert events == [
        (REWRITE_OWNER, model_ir, layout_keywords),
        (
            CONSECUTIVE_OWNER,
            model_ir,
            {**layout_keywords, "diagnostics": diagnostics},
        ),
        (RESTORE_OWNER, model_ir, layout_keywords),
    ]
