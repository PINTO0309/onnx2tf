from __future__ import annotations

import ast
import copy
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _prune_dead_operators,
    _reconcile_static_tensor_shapes,
)
from onnx2tf.tflite_builder.passes.prune_reconcile import (
    run_indexed_prune_reconcile_cleanup,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER = "run_indexed_prune_reconcile_cleanup"
RESULT_TARGETS = (
    "_core_cleanup_prune_reconcile_stats",
    "_layout_pass_set_2_prune_reconcile_stats",
    "_very_late_prune_reconcile_stats",
)
PREDECESSOR_TARGETS = (
    "_core_cleanup_squeeze_reshape_identity_stats",
    "_layout_pass_set_2_squeeze_reshape_identity_stats",
    "_very_late_residual_affine_fanout_stats",
)
SUCCESSORS = (
    "_advance_post_progress",
    "_advance_post_progress",
    "cleanup.post_cleanup.csp_attention",
)
POST_CLEANUP_OWNER_EXPRESSION = (
    "run_post_cleanup_sinet_csp_attention_cleanup("
    "shared_model_ir_pass_context)[1]"
)
RESULT_SCHEMA = {
    "removed_dead_operators": 0,
    "reconciled_static_tensor_shapes": 0,
}
VERY_LATE_RESULT_TARGETS = (
    "_very_late_residual_affine_prelu_stats",
    "_very_late_residual_affine_fanout_stats",
    "_very_late_prune_reconcile_stats",
)
VERY_LATE_PHASE_IDS = (
    "cleanup.very_late.residual_affine_prelu",
    "cleanup.very_late.residual_affine_fanout",
    "cleanup.very_late.prune_reconcile",
)
VERY_LATE_OWNER_EXPRESSIONS = (
    (
        "run_very_late_sinet_residual_affine_prelu_cleanup("
        "sinet_terminal_layout_recovery_context)[1]"
    ),
    (
        "_optimize_transpose_pre_add_mul_add_transpose_fanout_nhwc_chains("
        "model_ir)"
    ),
    (
        "run_indexed_prune_reconcile_cleanup("
        "model_ir, layout_state=session.layout_state)"
    ),
)


def _functions(path: Path) -> dict[str, ast.FunctionDef]:
    return {
        node.name: node
        for node in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(node, ast.FunctionDef)
    }


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if not isinstance(statement, (ast.Assign, ast.Expr)):
        return None
    call = statement.value if isinstance(statement.value, ast.Call) else None
    if (
        call is not None
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
    ):
        return call.args[1]
    return call


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _single_target(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return None
    target = statement.targets[0]
    return target.id if isinstance(target, ast.Name) else None


def _phase_id(statement: ast.stmt) -> str | None:
    if not isinstance(statement, ast.Expr) or not isinstance(statement.value, ast.Call):
        return None
    call = statement.value
    if (
        not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
    ):
        return None
    return ast.literal_eval(call.args[0])


def _pipeline_blocks(statements: list[ast.stmt]) -> list[list[ast.stmt]]:
    blocks = [statements]
    for statement in statements:
        if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for attribute in ("body", "orelse", "finalbody"):
            nested = getattr(statement, attribute, None)
            if isinstance(nested, list) and nested:
                blocks.extend(_pipeline_blocks(nested))
        if isinstance(statement, (ast.Try, ast.TryStar)):
            for handler in statement.handlers:
                blocks.extend(_pipeline_blocks(handler.body))
    return blocks


def _lowerer() -> ast.FunctionDef:
    return _functions(LOWERER_PATH)["lower_onnx_to_ir"]


def _raw_pairs(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index in range(len(block) - 1)
            if isinstance(block[index], ast.Expr)
            and isinstance(block[index + 1], ast.Expr)
            and _call_name(block[index]) == "_prune_dead_operators"
            and _call_name(block[index + 1]) == "_reconcile_static_tensor_shapes"
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _owner_locations(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index, statement in enumerate(block)
            if _call_name(statement) == OWNER
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _tensor(
    name: str,
    shape: list[int],
    *,
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32" if data is None else str(data.dtype).upper(),
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=data is None,
    )


def _make_cleanup_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_prune_reconcile")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 4]),
        "dead_source": _tensor(
            "dead_source",
            [1, 4],
            data=np.ones([1, 4], dtype=np.float32),
        ),
        "shape": _tensor(
            "shape",
            [2],
            data=np.asarray([2, 2], dtype=np.int32),
        ),
        "dead": _tensor("dead", [1, 4]),
        "output": _tensor("output", [1, 4]),
    }
    model_ir.operators = [
        OperatorIR("IDENTITY", ["dead_source"], ["dead"]),
        OperatorIR(
            "RESHAPE",
            ["input", "shape"],
            ["output"],
            options={"newShape": [2, 2]},
        ),
    ]
    return model_ir


def _snapshot(model_ir: ModelIR) -> dict[str, object]:
    return {
        "inputs": list(model_ir.inputs),
        "outputs": list(model_ir.outputs),
        "operators": [
            (str(op.op_type), list(op.inputs), list(op.outputs), dict(op.options))
            for op in model_ir.operators
        ],
        "tensors": {
            name: (
                list(tensor.shape),
                list(tensor.shape_signature or []),
                None
                if tensor.data is None
                else np.asarray(tensor.data).tolist(),
            )
            for name, tensor in model_ir.tensors.items()
        },
    }


def test_indexed_prune_reconcile_phase_boundaries_are_explicit() -> None:
    lowerer = _lowerer()
    locations = _owner_locations(lowerer)
    assert len(locations) == 3
    for occurrence, (
        (block, index),
        target,
        predecessor_target,
        successor,
    ) in enumerate(
        zip(
            locations,
            RESULT_TARGETS,
            PREDECESSOR_TARGETS,
            SUCCESSORS,
        )
    ):
        invocation = block[index]
        if occurrence == 0:
            assert ast.unparse(invocation) == (
                "session.record_phase_result("
                "'cleanup.core.prune_reconcile', "
                "run_indexed_prune_reconcile_cleanup(model_ir, "
                "layout_state=session.layout_state))"
            )
            assert ast.unparse(block[index - 1]) == (
                "session.record_phase_result("
                "'cleanup.core.squeeze_reshape_identity', "
                "run_squeeze_reshape_identity_cleanup(model_ir, "
                "include_unary_passthrough=True, "
                "layout_state=session.layout_state, "
                "diagnostics=session.diagnostics))"
            )
        elif occurrence == 1:
            assert ast.unparse(invocation) == (
                "session.record_phase_result("
                "'cleanup.layout_pass_set_2.prune_reconcile', "
                "run_indexed_prune_reconcile_cleanup(model_ir, "
                "layout_state=session.layout_state))"
            )
            assert ast.unparse(block[index - 1]) == (
                "session.record_phase_result("
                "'cleanup.layout_pass_set_2.squeeze_reshape_identity', "
                "run_squeeze_reshape_identity_cleanup(model_ir, "
                "include_unary_passthrough=True, "
                "layout_state=session.layout_state, "
                "diagnostics=session.diagnostics))"
            )
        else:
            assert _phase_id(invocation) == "cleanup.very_late.prune_reconcile"
            assert _phase_id(block[index - 1]) == (
                "cleanup.very_late.residual_affine_fanout"
            )
        call = _statement_call(invocation)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            "model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
        if successor == "_advance_post_progress":
            assert _call_name(block[index + 1]) == successor
        else:
            assert _phase_id(block[index + 1]) == successor
            assert ast.unparse(block[index + 1].value.args[1]) == (
                POST_CLEANUP_OWNER_EXPRESSION
            )


def test_raw_prune_reconcile_result_schemas_are_explicit() -> None:
    assert _prune_dead_operators(ModelIR("prune_schema")) == {
        "removed_dead_operators": 0,
    }
    assert _reconcile_static_tensor_shapes(ModelIR("reconcile_schema")) == {
        "reconciled_static_tensor_shapes": 0,
    }


def test_indexed_prune_reconcile_owner_reuses_one_index_and_retains_results(
    monkeypatch,
) -> None:
    expected_ir = _make_cleanup_model_ir()
    actual_ir = copy.deepcopy(expected_ir)
    expected_layout = LayoutState.from_model_ir(expected_ir)
    actual_layout = LayoutState.from_model_ir(actual_ir)
    prune_stats = _prune_dead_operators(
        expected_ir,
        layout_state=expected_layout,
    )
    reconcile_stats = _reconcile_static_tensor_shapes(expected_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    result = run_indexed_prune_reconcile_cleanup(
        actual_ir,
        layout_state=actual_layout,
    )

    assert result == {
        "removed_dead_operators": prune_stats["removed_dead_operators"],
        "reconciled_static_tensor_shapes": reconcile_stats[
            "reconciled_static_tensor_shapes"
        ],
    }
    assert result == {
        "removed_dead_operators": 1,
        "reconciled_static_tensor_shapes": 1,
    }
    assert refresh_count == 1
    assert _snapshot(actual_ir) == _snapshot(expected_ir)

    lowerer = _lowerer()
    invocations = [
        block[index]
        for block, index in _owner_locations(lowerer)
    ]
    assert tuple(_single_target(statement) for statement in invocations) == (
        None,
        None,
        None,
    )
    for invocation in invocations:
        call = _statement_call(invocation)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in call.keywords
        } == {"layout_state": "session.layout_state"}
    assert _raw_pairs(lowerer) == []
    assert not any(
        isinstance(node, ast.Name)
        and node.id in RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )


def test_very_late_residual_cleanup_uses_phase_result_store() -> None:
    lowerer = _lowerer()
    records = [
        statement
        for statement in lowerer.body
        if _phase_id(statement) in VERY_LATE_PHASE_IDS
    ]
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == (
        VERY_LATE_PHASE_IDS
    )
    assert tuple(ast.unparse(statement.value.args[1]) for statement in records) == (
        VERY_LATE_OWNER_EXPRESSIONS
    )
    assert indices == list(range(indices[0], indices[0] + 3))
    assert _phase_id(lowerer.body[indices[0] - 1]) == (
        "shape_topology.terminal.indexed_convergence"
    )
    assert _phase_id(lowerer.body[indices[-1] + 1]) == (
        "cleanup.post_cleanup.csp_attention"
    )
    assert ast.unparse(lowerer.body[indices[-1] + 1].value.args[1]) == (
        POST_CLEANUP_OWNER_EXPRESSION
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in VERY_LATE_RESULT_TARGETS
        for node in ast.walk(lowerer)
    )
