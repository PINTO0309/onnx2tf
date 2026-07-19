from __future__ import annotations

import ast
from pathlib import Path

import pytest

from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.core.model_ir_pass_context import ModelIRPassContext
from onnx2tf.tflite_builder.ir import ModelIR
from onnx2tf.tflite_builder.passes import (
    late_conv1d_decoder_layout_orchestration,
)
from onnx2tf.tflite_builder.passes.late_conv1d_decoder_layout_orchestration import (
    LATE_CONV1D_DECODER_LAYOUT_PASS_IDS,
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
    / "late_conv1d_decoder_layout_orchestration.py"
)
OWNER = "run_late_conv1d_decoder_layout_cleanup"
COMPOSITE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_layout_tail_orchestration.py"
)
COMPOSITE_OWNER = "run_very_late_layout_tail_cleanup"
LOWERER_OWNER = "run_late_dequant_swish_layout_tail_cleanup"
RESULT_TARGET = "_late_dequant_swish_layout_tail_results"
PREDECESSOR_GUARD = "optimize_layout_transpose_chains"
SUCCESSOR_PHASE_ID = "shape_reconciliation.primary.very_late_broadcast"
OLD_RESULT_TARGETS = (
    "_late_conv1d_squeeze_unary_stats",
    "_late_conv1d_rank4_unary_stats",
    "_late_conv1d_unary_fanout_stats",
    "_late_conv1d_instancenorm_unary_stats",
    "_late_conv1d_tencoder_stats",
    "_late_conv1d_batchmatmul_stats",
    "_late_decoder_deconv_stats",
    "_late_terminal_squeeze_mean_stats",
)
PASS_IDS = (
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_unary_transpose_reshape_expanddims_transpose_nhwc_chains",
    "_optimize_transpose_squeeze_unary_expanddims_transpose_nhwc_fanout_bypass_chains",
    "_optimize_transpose_squeeze_instancenorm_unary_expanddims_transpose_nhwc_chains",
    "_optimize_tencoder_add_expand_transpose_conv_nhwc_chains",
    "_optimize_transpose_squeeze_unary_batchmatmul_nhwc_chains",
    "_optimize_decoder_batchmatmul_add_expand_transpose_to_nhwc_deconv_input",
    "_optimize_transpose_squeeze_mean_squeeze_terminal_nhwc_chains",
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _phase_id(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def test_late_conv1d_decoder_cluster_uses_composite_result_outside_store() -> None:
    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "record_phase_result"
        and any(
            isinstance(child, ast.Name) and child.id == OWNER
            for child in ast.walk(node)
        )
        for node in ast.walk(lowerer)
    )


def test_late_conv1d_decoder_cluster_uses_one_composite_owner() -> None:
    assert OWNER_PATH.exists()
    owner = _functions(OWNER_PATH)[OWNER]
    owner_calls = [
        node.func.id
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in PASS_IDS
    ]
    assert owner_calls == list(PASS_IDS)

    composite_owner = _functions(COMPOSITE_OWNER_PATH)[COMPOSITE_OWNER]
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == OWNER
        for node in ast.walk(composite_owner)
    ) == 1

    lowerer = _lowerer()
    assignment = next(
        statement
        for statement in lowerer.body
        if _single_target(statement) == RESULT_TARGET
    )
    index = lowerer.body.index(assignment)
    assert _call_name(assignment) == LOWERER_OWNER
    predecessor = lowerer.body[index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == PREDECESSOR_GUARD
    assert _phase_id(lowerer.body[index + 1]) == SUCCESSOR_PHASE_ID
    assert not any(
        isinstance(node, ast.Name) and node.id in OLD_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )


def test_late_conv1d_decoder_owner_preserves_context_and_result_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_ir = ModelIR("late_conv1d_decoder_layout_owner")
    context = ModelIRPassContext(
        model_ir=model_ir,
        layout_state=LayoutState.from_model_ir(model_ir),
        diagnostics=[],
    )
    observed: list[tuple[str, object, object]] = []
    expected_results = tuple(
        {f"result_{index}": index}
        for index in range(1, len(PASS_IDS) + 1)
    )

    def _callback(name: str, result: dict[str, int]):
        def _run(
            candidate: ModelIR,
            *,
            layout_state: object,
        ) -> dict[str, int]:
            observed.append((name, candidate, layout_state))
            return dict(result)

        return _run

    for pass_id, result in zip(PASS_IDS, expected_results, strict=True):
        monkeypatch.setattr(
            late_conv1d_decoder_layout_orchestration,
            pass_id,
            _callback(pass_id, result),
        )

    assert (
        late_conv1d_decoder_layout_orchestration.run_late_conv1d_decoder_layout_cleanup(
            context
        )
        == expected_results
    )
    assert [entry[0] for entry in observed] == list(PASS_IDS)
    assert all(entry[1] is context.model_ir for entry in observed)
    assert all(entry[2] is context.layout_state for entry in observed)
    assert LATE_CONV1D_DECODER_LAYOUT_PASS_IDS == PASS_IDS
