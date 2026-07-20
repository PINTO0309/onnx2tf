from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
VERY_LATE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_pad_instancenorm_layout_orchestration.py"
)
VERY_LATE_OWNER = "run_very_late_pad_instancenorm_layout_cleanup"
VERY_LATE_RESULT = "_very_late_pad_instancenorm_layout_results"
VERY_LATE_LAYOUT_BROADCAST_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_layout_broadcast_orchestration.py"
)
VERY_LATE_LAYOUT_BROADCAST_OWNER = (
    "run_very_late_layout_broadcast_cleanup"
)
VERY_LATE_LAYOUT_BROADCAST_RESULT = "_very_late_layout_broadcast_results"
VERY_LATE_LAYOUT_TAIL_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_layout_tail_orchestration.py"
)
VERY_LATE_LAYOUT_TAIL_OWNER = "run_very_late_layout_tail_cleanup"
VERY_LATE_LAYOUT_TAIL_RESULT = "_very_late_layout_tail_results"
LOWERER_LAYOUT_TAIL_OWNER = "run_late_dequant_swish_layout_tail_cleanup"
LOWERER_LAYOUT_TAIL_RESULT = "_late_dequant_swish_layout_tail_results"
SHARED_LATE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "shared_late_reconciliation_orchestration.py"
)
SHARED_LATE_OWNER = "run_shared_late_reconciliation_cleanup"
SHARED_LATE_RESULT = "_shared_late_requires_reconciliation"
LATE_BINARY_REPAIR_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_binary_repair_orchestration.py"
)
LATE_BINARY_REPAIR_OWNER = "run_late_binary_repair_cleanup"
LATE_BINARY_REPAIR_RESULT = (
    "_late_binary_repair_requires_reconciliation"
)
OPTIONAL_LATE_BINARY_LAYOUT_RECOVERY_RESULT = (
    "_late_binary_layout_recovery_requires_reconciliation"
)
PRE_TERMINAL_INSTANCENORM_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "pre_terminal_instancenorm_layout_orchestration.py"
)
PRE_TERMINAL_INSTANCENORM_OWNER = (
    "run_pre_terminal_instancenorm_layout_cleanup"
)
ABSOLUTE_FINAL_AFFINE_INSTANCENORM_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_affine_instancenorm_orchestration.py"
)
ABSOLUTE_FINAL_AFFINE_INSTANCENORM_OWNER = (
    "run_absolute_final_affine_instancenorm_cleanup"
)
ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_normalization_attention_orchestration.py"
)
ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_OWNER = (
    "run_absolute_final_normalization_attention_rank1_cleanup"
)
ABSOLUTE_FINAL_CLEANUP_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "absolute_final_cleanup_orchestration.py"
)
ABSOLUTE_FINAL_CLEANUP_OWNER = "run_absolute_final_cleanup"
ABSOLUTE_FINAL_CLEANUP_RESULT = "_absolute_final_cleanup_results"
BINARY_LAYOUT_CONVERGENCE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "binary_layout_convergence.py"
)
BINARY_LAYOUT_CONVERGENCE_OWNER = "run_indexed_binary_layout_convergence"
TERMINAL_STABILIZATION_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_stabilization_orchestration.py"
)
TERMINAL_STABILIZATION_OWNER = "run_terminal_stabilization_cleanup"
TERMINAL_STABILIZATION_RESULT = "_final_terminal_stabilization_results"
VERY_LATE_DYNAMIC_ADAPTER_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "very_late_dynamic_adapter_orchestration.py"
)
VERY_LATE_DYNAMIC_ADAPTER_OWNER = (
    "run_very_late_dynamic_adapter_cleanup"
)
LATE_LAYOUT_COMPOSITE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_reshape_shuffle_attention_window_orchestration.py"
)
LATE_LAYOUT_COMPOSITE_OWNER = (
    "run_late_reshape_shuffle_attention_window_cleanup"
)
LATE_LAYOUT_COMPOSITE_RESULT = (
    "_late_reshape_shuffle_attention_window_results"
)
FINAL_BOUNDARY_COMPOSITE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "final_boundary_slice_concat_orchestration.py"
)
FINAL_BOUNDARY_COMPOSITE_OWNER = "run_final_boundary_slice_concat_cleanup"
FINAL_BOUNDARY_COMPOSITE_RESULT = "_final_boundary_slice_concat_results"
LATE_FINAL_SHAPE_BOUNDARY_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_final_shape_boundary_orchestration.py"
)
LATE_FINAL_SHAPE_BOUNDARY_OWNER = "run_late_final_shape_boundary_cleanup"
LATE_FINAL_SHAPE_TERMINAL_OWNER = (
    "run_late_affine_final_shape_terminal_convpool_cleanup"
)
LATE_FINAL_SHAPE_BOUNDARY_RESULT = (
    "_late_affine_final_shape_terminal_convpool_results"
)
LATE_AFFINE_CONCAT_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_affine_concat_orchestration.py"
)
LATE_AFFINE_CONCAT_OWNER = "run_late_affine_concat_cleanup"
LATE_AFFINE_OPTIONAL_FANOUT_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_affine_optional_fanout_orchestration.py"
)
LATE_AFFINE_OPTIONAL_FANOUT_OWNER = (
    "run_late_affine_optional_fanout_cleanup"
)
LATE_AFFINE_CONCAT_RESULT = "_late_affine_final_shape_terminal_convpool_results"
TERMINAL_FANOUT_SINGLETON_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "terminal_fanout_singleton_orchestration.py"
)
TERMINAL_FANOUT_SINGLETON_OWNER = "run_terminal_fanout_singleton_cleanup"
TERMINAL_FANOUT_SINGLETON_RESULT = (
    "_late_affine_final_shape_terminal_convpool_results"
)
NO_LAYOUT_FALLBACK_AFFINE_RESULT = (
    "_no_layout_fallback_affine_prepost_stats"
)


def _lowerer_body() -> list[ast.stmt]:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    return lowerer.body


def _late_layout_composite_owner_calls(function_name: str) -> list[ast.Call]:
    tree = ast.parse(
        LATE_LAYOUT_COMPOSITE_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == LATE_LAYOUT_COMPOSITE_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _late_layout_composite_assignment(body: list[ast.stmt]) -> ast.Assign:
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LATE_FINAL_SHAPE_BOUNDARY_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == LATE_FINAL_SHAPE_TERMINAL_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in assignment.value.keywords
    } == {
        "optimize_layout_transpose_chains": "optimize_layout_transpose_chains"
    }
    return assignment


def _final_boundary_composite_owner_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        FINAL_BOUNDARY_COMPOSITE_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == FINAL_BOUNDARY_COMPOSITE_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name
    ]


def _final_boundary_composite_assignment(body: list[ast.stmt]) -> ast.Assign:
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LATE_FINAL_SHAPE_BOUNDARY_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == LATE_FINAL_SHAPE_TERMINAL_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in assignment.value.keywords
    } == {
        "optimize_layout_transpose_chains": "optimize_layout_transpose_chains"
    }
    return assignment


def _late_affine_concat_owner_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        LATE_AFFINE_CONCAT_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == LATE_AFFINE_CONCAT_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name.removeprefix("_")
    ]


def _late_affine_concat_assignment(body: list[ast.stmt]) -> ast.Assign:
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LATE_AFFINE_CONCAT_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == LATE_FINAL_SHAPE_TERMINAL_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in assignment.value.keywords
    } == {"optimize_layout_transpose_chains": "optimize_layout_transpose_chains"}
    return assignment


def _very_late_owner_call_count(function_name: str) -> int:
    tree = ast.parse(VERY_LATE_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == VERY_LATE_OWNER
    )
    owner_function_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_function_name
        for node in ast.walk(owner)
    )


def _very_late_assignment(body: list[ast.stmt]) -> ast.Assign:
    tree = ast.parse(
        VERY_LATE_LAYOUT_TAIL_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == VERY_LATE_LAYOUT_TAIL_OWNER
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == VERY_LATE_OWNER
        for node in ast.walk(owner)
    ) == 1
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_LAYOUT_TAIL_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == LOWERER_LAYOUT_TAIL_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in assignment.value.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}
    return assignment


def _very_late_layout_broadcast_owner_call_count(
    function_name: str,
) -> int:
    tree = ast.parse(
        VERY_LATE_LAYOUT_BROADCAST_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == VERY_LATE_LAYOUT_BROADCAST_OWNER
    )
    owner_function_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_function_name
        for node in ast.walk(owner)
    )


def _very_late_layout_broadcast_assignment(
    body: list[ast.stmt],
) -> ast.Assign:
    tree = ast.parse(
        VERY_LATE_LAYOUT_TAIL_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == VERY_LATE_LAYOUT_TAIL_OWNER
    )
    assert sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == VERY_LATE_LAYOUT_BROADCAST_OWNER
        for node in ast.walk(owner)
    ) == 1
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LOWERER_LAYOUT_TAIL_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == LOWERER_LAYOUT_TAIL_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in assignment.value.keywords
    } == {"include_layout_transpose": "optimize_layout_transpose_chains"}
    return assignment


def _shared_late_owner_call_count(function_name: str) -> int:
    tree = ast.parse(SHARED_LATE_OWNER_PATH.read_text(encoding="utf-8"))
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == SHARED_LATE_OWNER
    )
    owner_function_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_function_name
        for node in ast.walk(owner)
    )


def _shared_late_assignment(body: list[ast.stmt]) -> ast.Assign:
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == SHARED_LATE_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == SHARED_LATE_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert assignment.value.keywords == []
    return assignment


def _late_binary_repair_owner_call_count(function_name: str) -> int:
    tree = ast.parse(
        LATE_BINARY_REPAIR_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == LATE_BINARY_REPAIR_OWNER
    )
    owner_function_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_function_name
        for node in ast.walk(owner)
    )


def _late_binary_repair_assignment(
    body: list[ast.stmt],
) -> ast.Assign:
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == LATE_BINARY_REPAIR_RESULT
    )
    assert isinstance(assignment.value, ast.Call)
    assert isinstance(assignment.value.func, ast.Name)
    assert assignment.value.func.id == LATE_BINARY_REPAIR_OWNER
    assert [ast.unparse(argument) for argument in assignment.value.args] == [
        "shared_model_ir_pass_context"
    ]
    assert assignment.value.keywords == []
    return assignment


def _pre_terminal_instancenorm_owner_call_count(
    function_name: str,
) -> int:
    tree = ast.parse(
        PRE_TERMINAL_INSTANCENORM_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == PRE_TERMINAL_INSTANCENORM_OWNER
    )
    owner_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
        for node in ast.walk(owner)
    )


def _absolute_final_affine_instancenorm_owner_call_count(
    function_name: str,
) -> int:
    tree = ast.parse(
        ABSOLUTE_FINAL_AFFINE_INSTANCENORM_OWNER_PATH.read_text(
            encoding="utf-8"
        )
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_AFFINE_INSTANCENORM_OWNER
    )
    owner_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
        for node in ast.walk(owner)
    )


def _absolute_final_normalization_attention_owner_call_count(
    function_name: str,
) -> int:
    tree = ast.parse(
        ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_OWNER_PATH.read_text(
            encoding="utf-8"
        )
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_NORMALIZATION_ATTENTION_OWNER
    )
    owner_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
        for node in ast.walk(owner)
    )


def _absolute_final_cleanup_owner_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        ABSOLUTE_FINAL_CLEANUP_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == ABSOLUTE_FINAL_CLEANUP_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name.removeprefix("_")
    ]


def _binary_layout_convergence_owner_call_count(function_name: str) -> int:
    tree = ast.parse(
        BINARY_LAYOUT_CONVERGENCE_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == BINARY_LAYOUT_CONVERGENCE_OWNER
    )
    owner_name = function_name.removeprefix("_")
    return sum(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == owner_name
        for node in ast.walk(owner)
    )


def _terminal_stabilization_owner_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        TERMINAL_STABILIZATION_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_STABILIZATION_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name.removeprefix("_")
    ]


def _very_late_dynamic_adapter_owner_calls(
    function_name: str,
) -> list[ast.Call]:
    tree = ast.parse(
        VERY_LATE_DYNAMIC_ADAPTER_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == VERY_LATE_DYNAMIC_ADAPTER_OWNER
    )
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == function_name.removeprefix("_")
    ]


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(
        statement.value,
        ast.Call,
    ):
        return statement.value
    return None


def _call_name(call: ast.Call | None) -> str | None:
    if call is None:
        return None
    if isinstance(call.func, ast.Name):
        return call.func.id
    if (
        isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "session"
        and call.func.attr == "record_phase_result"
        and len(call.args) == 2
        and isinstance(call.args[1], ast.Call)
        and isinstance(call.args[1].func, ast.Name)
    ):
        return call.args[1].func.id
    return None


def _phase_result_owner(statement: ast.stmt) -> ast.Call | None:
    call = _statement_call(statement)
    if (
        call is None
        or not isinstance(call.func, ast.Attribute)
        or not isinstance(call.func.value, ast.Name)
        or call.func.value.id != "session"
        or call.func.attr != "record_phase_result"
        or len(call.args) != 2
        or not isinstance(call.args[1], ast.Call)
    ):
        return None
    return call.args[1]


def _owner_call(statement: ast.stmt) -> ast.Call | None:
    return _phase_result_owner(statement) or _statement_call(statement)


def _assert_phase_result_record(
    statement: ast.stmt,
    *,
    phase_id: str,
    owner_expression: str,
) -> None:
    assert isinstance(statement, ast.Expr)
    assert ast.unparse(statement) == (
        f"session.record_phase_result('{phase_id}', {owner_expression})"
    )


def _call_index(
    body: list[ast.stmt],
    function_name: str,
    *,
    start: int = 0,
) -> int:
    return next(
        index
        for index, statement in enumerate(body[start:], start=start)
        if _call_name(_statement_call(statement)) == function_name
    )


def test_primary_path_validates_terminal_layout_and_clears_stale_errors() -> None:
    body = _lowerer_body()
    stabilization_index = _call_index(body, TERMINAL_STABILIZATION_OWNER)
    validation_index = next(
        index
        for index in range(stabilization_index + 1, len(body))
        if _call_name(_phase_result_owner(body[index]))
        == "run_topology_layout_validation"
    )

    assert stabilization_index + 1 == validation_index
    validation = body[validation_index]
    _assert_phase_result_record(
        validation,
        phase_id="layout_validation.primary.terminal",
        owner_expression="run_topology_layout_validation(model_ir)",
    )

    terminal = body[validation_index + 1]
    assert isinstance(terminal, ast.Return)
    assert ast.unparse(terminal.value) == "_finalize_model_ir(model_ir)"


def test_primary_path_retains_terminal_mutation_results() -> None:
    body = _lowerer_body()
    stabilization_index = _call_index(body, TERMINAL_STABILIZATION_OWNER)
    statement = body[stabilization_index]
    assert isinstance(statement, ast.Assign)
    assert len(statement.targets) == 1
    assert isinstance(statement.targets[0], ast.Name)
    assert statement.targets[0].id == TERMINAL_STABILIZATION_RESULT
    call = statement.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == TERMINAL_STABILIZATION_OWNER
    assert [ast.unparse(argument) for argument in call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert call.keywords == []

    expected_owner_calls = (
        (
            "run_indexed_binary_layout_convergence",
            ["context.model_ir"],
            {},
        ),
        (
            "coalesce_static_high_rank_binary_operators",
            ["context.model_ir"],
            {"layout_state": "context.layout_state"},
        ),
        (
            "realign_dynamic_boundary_shape_signature_map",
            ["context.model_ir"],
            {},
        ),
    )
    for function_name, arguments, keywords in expected_owner_calls:
        owner_calls = _terminal_stabilization_owner_calls(function_name)
        assert len(owner_calls) == 1
        owner_call = owner_calls[0]
        assert [ast.unparse(argument) for argument in owner_call.args] == arguments
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in owner_call.keywords
        } == keywords


def test_primary_path_stages_final_high_rank_bmm_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_high_rank_bmm_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(final_high_rank_bmm_stats.get("
        "'compressed_static_high_rank_batch_matmul', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_topology.primary.final_high_rank_batch_matmul",
        owner_expression=(
            "run_static_shape_topology_reconciliation(model_ir)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_pad_layout_stats"


def test_primary_path_stages_final_pad_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_pad_layout_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(final_pad_layout_stats.get("
        "'repaired_channel_last_inputs_for_channel_first_pad', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_topology.primary.final_pad_layout",
        owner_expression=(
            "run_static_shape_topology_reconciliation(model_ir)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_conv_input_stats"


def test_primary_path_stages_complete_final_conv_input_evidence() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_conv_input_stats"
    )

    stats = body[stats_index]
    assert isinstance(stats, ast.Assign)
    assert ast.unparse(stats.value) == (
        "run_stale_conv_input_adapter_repair_summary(model_ir)"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(final_conv_input_stats.get("
        "'repaired_stale_nchw_to_nhwc_conv_input_transposes', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_topology.primary.final_conv_input",
        owner_expression=(
            "run_static_shape_topology_reconciliation(model_ir)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_concat_layout_stats"


def test_primary_path_stages_final_mixed_concat_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_concat_layout_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(final_concat_layout_stats.get("
        "'repaired_mixed_nhwc_inputs_for_nchw_concat', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_topology.primary.final_mixed_concat",
        owner_expression=(
            "run_static_shape_topology_reconciliation(model_ir)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_concat_axis_stats"


def test_primary_path_stages_complete_final_concat_axis_binary_evidence() -> None:
    body = _lowerer_body()
    axis_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_concat_axis_stats"
    )

    axis_guard = body[axis_index + 1]
    assert isinstance(axis_guard, ast.If)
    assert ast.unparse(axis_guard.test) == (
        "int(final_concat_axis_stats.get("
        "'repaired_nchw_concat_transpose_conv_axes', 0)) > 0"
    )
    assert len(axis_guard.body) == 1
    axis_reconciliation = axis_guard.body[0]
    _assert_phase_result_record(
        axis_reconciliation,
        phase_id="shape_topology.primary.final_concat_axis",
        owner_expression=(
            "run_static_shape_topology_reconciliation(model_ir)"
        ),
    )

    binary_index = axis_index + 2
    binary_stats = body[binary_index]
    assert isinstance(binary_stats, ast.Assign)
    assert isinstance(binary_stats.targets[0], ast.Name)
    assert binary_stats.targets[0].id == "final_binary_layout_stats"
    assert ast.unparse(binary_stats.value) == (
        "run_stale_binary_adapter_repair_summary(model_ir)"
    )

    binary_guard = body[binary_index + 1]
    assert isinstance(binary_guard, ast.If)
    assert ast.unparse(binary_guard.test) == (
        "int(final_binary_layout_stats.get("
        "'repaired_stale_nchw_to_nhwc_channelwise_binary_transposes', 0)) > 0"
    )
    assert len(binary_guard.body) == 1
    binary_reconciliation = binary_guard.body[0]
    _assert_phase_result_record(
        binary_reconciliation,
        phase_id="shape_topology.primary.final_binary_layout",
        owner_expression=(
            "run_static_shape_topology_reconciliation(model_ir)"
        ),
    )

    following = body[binary_index + 2]
    assert isinstance(following, ast.Expr)
    assert ast.unparse(following.value) == "_advance_post_progress()"


def test_primary_path_stages_final_prelu_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_prelu_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "_stats_have_positive_count(final_prelu_stats)"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_prelu",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_consecutive_reshape_stats"


def test_primary_path_stages_complete_final_placeholder_reconciliations() -> None:
    body = _lowerer_body()
    restore_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_placeholder_matmul_stats"
    )

    result_name = "_final_placeholder_matmul_static_shape_stats"
    default_stats = body[restore_index + 1]
    assert isinstance(default_stats, ast.Assign)
    assert isinstance(default_stats.targets[0], ast.Name)
    assert default_stats.targets[0].id == result_name
    assert isinstance(default_stats.value, ast.Dict)
    assert {
        key.value: value.value
        for key, value in zip(
            default_stats.value.keys,
            default_stats.value.values,
        )
        if isinstance(key, ast.Constant) and isinstance(value, ast.Constant)
    } == {
        "reconciled_static_tensor_shapes": 0,
        "reconciled_static_shape_mutations": 0,
    }

    outer_guard = body[restore_index + 2]
    assert isinstance(outer_guard, ast.If)
    assert len(outer_guard.body) == 5

    first_reconciliation = outer_guard.body[0]
    assert isinstance(first_reconciliation, ast.Assign)
    assert isinstance(first_reconciliation.targets[0], ast.Name)
    assert first_reconciliation.targets[0].id == result_name
    assert isinstance(first_reconciliation.value, ast.Call)
    assert isinstance(first_reconciliation.value.func, ast.Name)
    assert first_reconciliation.value.func.id == "_reconcile_static_tensor_shapes"
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in first_reconciliation.value.keywords
    } == {"include_mutation_count": "True"}

    legacy_projection = outer_guard.body[1]
    assert isinstance(legacy_projection, ast.Assign)
    assert isinstance(legacy_projection.targets[0], ast.Name)
    assert legacy_projection.targets[0].id == "final_placeholder_reconcile_stats"
    assert isinstance(legacy_projection.value, ast.Dict)
    assert ast.unparse(legacy_projection.value) == (
        "{'reconciled_static_tensor_shapes': "
        "int(_final_placeholder_matmul_static_shape_stats.get("
        "'reconciled_static_tensor_shapes', 0))}"
    )

    binary_adapter = outer_guard.body[2]
    assert isinstance(binary_adapter, ast.Assign)
    assert len(binary_adapter.targets) == 1
    assert isinstance(binary_adapter.targets[0], ast.Name)
    assert binary_adapter.targets[0].id == "final_placeholder_binary_stats"
    assert isinstance(binary_adapter.value, ast.Call)
    assert isinstance(binary_adapter.value.func, ast.Name)
    assert (
        binary_adapter.value.func.id
        == "run_indexed_binary_layout_adapter_summary"
    )

    binary_guard = outer_guard.body[3]
    assert isinstance(binary_guard, ast.If)
    assert ast.unparse(binary_guard.test) == (
        "_stats_have_positive_count(final_placeholder_reconcile_stats, "
        "final_placeholder_binary_stats)"
    )
    assert len(binary_guard.body) == 1
    second_reconciliation = binary_guard.body[0]
    _assert_phase_result_record(
        second_reconciliation,
        phase_id="shape_reconciliation.primary.final_placeholder_binary",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    topology_checkpoint = outer_guard.body[4]
    _assert_phase_result_record(
        topology_checkpoint,
        phase_id="topology.primary.final_placeholder",
        owner_expression="_topologically_sort_operators(model_ir)",
    )

    following = body[restore_index + 3]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_se_fc_gather_stats"


def test_primary_path_stages_final_mixed_singleton_concat_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_mixed_singleton_concat_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_mixed_singleton_nchw_inputs_for_nhwc_concat"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_mixed_singleton_concat",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_placeholder_matmul_stats"


def test_primary_path_stages_final_broadcast_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_broadcast_repair_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_rank4_channelwise_broadcast_constants"
    )
    assert len(guard.body) == 2

    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_broadcast",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )
    topology_layout_refresh = guard.body[1]
    _assert_phase_result_record(
        topology_layout_refresh,
        phase_id="topology_layout.primary.final_broadcast",
        owner_expression="run_topology_layout_refresh(model_ir)",
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_mixed_singleton_concat_stats"


def test_primary_path_stages_final_instancenorm_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_instancenorm_repair_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_decomposed_instance_normalization_layouts"
    )
    assert len(guard.body) == 2

    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_instancenorm",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )
    topology_layout_refresh = guard.body[1]
    _assert_phase_result_record(
        topology_layout_refresh,
        phase_id="topology_layout.primary.final_instancenorm",
        owner_expression="run_topology_layout_refresh(model_ir)",
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_broadcast_repair_stats"


def test_primary_path_stages_final_convinteger_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_convinteger_layout_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert len(get_calls) == 1
    assert isinstance(get_calls[0].args[0], ast.Constant)
    assert get_calls[0].args[0].value == (
        "repaired_channel_last_convinteger_input_transposes"
    )
    assert "propagated_channel_last_layout_hints" not in ast.unparse(guard.test)
    assert len(guard.body) == 2

    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_convinteger",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )
    topology_layout_refresh = guard.body[1]
    _assert_phase_result_record(
        topology_layout_refresh,
        phase_id="topology_layout.primary.final_convinteger",
        owner_expression="run_topology_layout_refresh(model_ir)",
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_instancenorm_repair_stats"


def test_primary_path_stages_absolute_final_dynamic_rank1_result() -> None:
    body = _lowerer_body()
    owner_name = "_rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs"
    direct_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == owner_name
    ]
    assert len(direct_indices) == 0
    very_late_calls = _very_late_dynamic_adapter_owner_calls(owner_name)
    assert len(very_late_calls) == 1
    assert ast.unparse(very_late_calls[0]) == (
        "rewrite_dynamic_rank1_unsqueeze_reshape_shape_inputs("
        "context.model_ir, layout_state=context.layout_state)"
    )
    assert (
        _absolute_final_normalization_attention_owner_call_count(owner_name)
        == 1
    )

    assert len(
        _absolute_final_cleanup_owner_calls(
            "run_absolute_final_normalization_attention_rank1_cleanup"
        )
    ) == 1
    final_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == ABSOLUTE_FINAL_CLEANUP_RESULT
    )
    final = body[final_index]
    assert isinstance(final, ast.Assign)
    assert isinstance(final.targets[0], ast.Name)
    assert final.targets[0].id == ABSOLUTE_FINAL_CLEANUP_RESULT
    assert _call_name(_statement_call(final)) == ABSOLUTE_FINAL_CLEANUP_OWNER

    topology_layout_refresh = body[final_index + 1]
    _assert_phase_result_record(
        topology_layout_refresh,
        phase_id="topology_layout.primary.absolute_final",
        owner_expression="run_topology_layout_refresh(model_ir)",
    )
    following = body[final_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_convinteger_layout_stats"

    all_calls = sorted(
        (
            node
            for node in ast.walk(next(
                function
                for function in ast.parse(
                    LOWERER_PATH.read_text(encoding="utf-8")
                ).body
                if isinstance(function, ast.FunctionDef)
                and function.name == "lower_onnx_to_ir"
            ))
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == owner_name
        ),
        key=lambda call: call.lineno,
    )
    assert len(all_calls) == 1
    assert (
        len(all_calls)
        + len(very_late_calls)
        + _absolute_final_normalization_attention_owner_call_count(owner_name)
        == 3
    )


def test_primary_path_retains_absolute_final_boundary_signature_results() -> None:
    body = _lowerer_body()
    realign_name = "_realign_dynamic_boundary_shape_signature_map"
    sanitize_name = "_sanitize_static_shape_signature_consistency"
    summary_name = "run_boundary_shape_signature_cleanup"
    summary_calls = _absolute_final_cleanup_owner_calls(summary_name)
    assert len(summary_calls) == 1
    assert ast.unparse(summary_calls[0]) == (
        "run_boundary_shape_signature_cleanup(context.model_ir)"
    )
    summary_index = _call_index(body, ABSOLUTE_FINAL_CLEANUP_OWNER)
    summary = body[summary_index]
    assert isinstance(summary, ast.Assign)
    assert len(summary.targets) == 1
    assert isinstance(summary.targets[0], ast.Name)
    assert (
        summary.targets[0].id == ABSOLUTE_FINAL_CLEANUP_RESULT
    )
    assert ast.unparse(summary.value) == (
        "run_absolute_final_cleanup(shared_model_ir_pass_context)"
    )

    realign_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == realign_name
    ]
    sanitize_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == sanitize_name
    ]
    assert len(realign_indices) == 0
    assert len(_terminal_stabilization_owner_calls(realign_name)) == 1
    assert _shared_late_owner_call_count(realign_name) == 1
    assert len(sanitize_indices) == 0
    assert _late_binary_repair_owner_call_count(sanitize_name) == 1

    following = body[summary_index + 1]
    _assert_phase_result_record(
        following,
        phase_id="topology_layout.primary.absolute_final",
        owner_expression="run_topology_layout_refresh(model_ir)",
    )


def test_primary_path_retains_guarded_no_layout_final_cleanup_results() -> None:
    body = _lowerer_body()
    guard_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.If)
        and ast.unparse(statement.test)
        == "apply_safe_transpose_reduction_lite_on_no_layout_opt"
        and any(
            _call_name(_statement_call(child))
            == "run_no_layout_final_cleanup"
            for child in statement.body
        )
    )
    guard = body[guard_index]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 2
    summary = guard.body[0]
    assert isinstance(summary, ast.Assign)
    assert isinstance(summary.targets[0], ast.Name)
    assert summary.targets[0].id == "_no_layout_final_cleanup_results"
    assert ast.unparse(summary.value) == (
        "run_no_layout_final_cleanup(shared_model_ir_pass_context)"
    )

    topology_checkpoint = guard.body[1]
    _assert_phase_result_record(
        topology_checkpoint,
        phase_id="topology.primary.no_layout_post_reduction",
        owner_expression="_topologically_sort_operators(model_ir)",
    )
    previous = body[guard_index - 1]
    _assert_phase_result_record(
        previous,
        phase_id="topology.primary.post_lowering",
        owner_expression="_topologically_sort_operators(model_ir)",
    )
    following = body[guard_index + 1]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert (
        following.targets[0].id == ABSOLUTE_FINAL_CLEANUP_RESULT
    )


def test_primary_path_retains_final_precision_cleanup_results() -> None:
    body = _lowerer_body()
    sequence_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "_final_precision_cleanup_results"
    )
    sequence = body[sequence_index]
    assert isinstance(sequence, ast.Assign)
    assert ast.unparse(sequence.value) == (
        "_run_precision_cleanup_sequence(model_ir, session.layout_state)"
    )
    assert _call_name(_statement_call(body[sequence_index + 1])) == (
        "_set_post_progress_desc"
    )


def test_primary_path_retains_core_cleanup_consecutive_mul_result() -> None:
    body = _lowerer_body()
    owner_name = "run_consecutive_mul_constants_cleanup"
    direct_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == owner_name
    ]
    assert len(direct_indices) == 1

    core_index = direct_indices[0]
    core = body[core_index]
    _assert_phase_result_record(
        core,
        phase_id="cleanup.core.consecutive_mul",
        owner_expression=(
            "run_consecutive_mul_constants_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )

    assert _call_name(_statement_call(body[core_index - 2])) == (
        "_optimize_fuse_pseudo_leakyrelu_chains"
    )
    assert _call_name(_statement_call(body[core_index - 1])) == (
        "_optimize_yolo_decode_mul_square_anchor_chains"
    )
    assert _call_name(_statement_call(body[core_index + 1])) == (
        "_sanitize_terminal_transpose_before_dequantize"
    )



def test_primary_path_retains_core_cleanup_fusion_results() -> None:
    body = _lowerer_body()
    pseudo_name = "_optimize_fuse_pseudo_leakyrelu_chains"
    yolo_name = "_optimize_yolo_decode_mul_square_anchor_chains"
    consecutive_name = "run_consecutive_mul_constants_cleanup"
    pseudo_index = _call_index(body, pseudo_name)
    yolo_index = _call_index(body, yolo_name, start=pseudo_index + 1)
    consecutive_index = _call_index(
        body,
        consecutive_name,
        start=yolo_index + 1,
    )
    assert yolo_index == pseudo_index + 1
    assert consecutive_index == yolo_index + 1

    expected = (
        (pseudo_index, "cleanup.core.pseudo_leakyrelu", pseudo_name),
        (yolo_index, "cleanup.core.yolo_decode", yolo_name),
    )
    for index, phase_id, function_name in expected:
        statement = body[index]
        _assert_phase_result_record(
            statement,
            phase_id=phase_id,
            owner_expression=f"{function_name}(model_ir)",
        )

    progress = body[pseudo_index - 1]
    assert ast.unparse(progress) == (
        "_set_post_progress_desc('core cleanup passes')"
    )
    consecutive = body[consecutive_index]
    _assert_phase_result_record(
        consecutive,
        phase_id="cleanup.core.consecutive_mul",
        owner_expression=(
            "run_consecutive_mul_constants_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )


def test_primary_path_retains_terminal_quantization_cleanup_results() -> None:
    body = _lowerer_body()
    sanitizer_name = "_sanitize_terminal_transpose_before_dequantize"
    qdq_name = "run_terminal_quantize_dequantize_cleanup"
    sanitizer_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == sanitizer_name
    ]
    qdq_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == qdq_name
    ]
    assert len(sanitizer_indices) == 2
    assert len(qdq_indices) == 2
    assert qdq_indices == [index + 1 for index in sanitizer_indices]

    expected_phase_ids = (
        ("cleanup.core.terminal_dequant", "cleanup.core.terminal_qdq"),
        ("cleanup.terminal.dequant", "cleanup.terminal.qdq"),
    )
    for pair_index, (sanitizer_index, qdq_index) in enumerate(
        zip(sanitizer_indices, qdq_indices)
    ):
        sanitizer = body[sanitizer_index]
        _assert_phase_result_record(
            sanitizer,
            phase_id=expected_phase_ids[pair_index][0],
            owner_expression=(
                "_sanitize_terminal_transpose_before_dequantize(model_ir)"
            ),
        )
        sanitizer_call = _phase_result_owner(sanitizer)
        assert isinstance(sanitizer_call, ast.Call)
        assert isinstance(sanitizer_call.func, ast.Name)
        assert sanitizer_call.func.id == sanitizer_name
        assert [
            ast.unparse(argument) for argument in sanitizer_call.args
        ] == ["model_ir"]
        assert sanitizer_call.keywords == []

        qdq = body[qdq_index]
        _assert_phase_result_record(
            qdq,
            phase_id=expected_phase_ids[pair_index][1],
            owner_expression=(
                "run_terminal_quantize_dequantize_cleanup(model_ir, "
                "layout_state=session.layout_state, "
                "diagnostics=session.diagnostics)"
            ),
        )
        qdq_call = _phase_result_owner(qdq)
        assert isinstance(qdq_call, ast.Call)
        assert isinstance(qdq_call.func, ast.Name)
        assert qdq_call.func.id == qdq_name
        assert [ast.unparse(argument) for argument in qdq_call.args] == [
            "model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in qdq_call.keywords
        } == {
            "layout_state": "session.layout_state",
            "diagnostics": "session.diagnostics",
        }
        assert _call_name(_statement_call(body[qdq_index + 1])) == (
            "_optimize_fold_conv_mul_add_affine_chains"
        )

    first_previous = body[sanitizer_indices[0] - 1]
    _assert_phase_result_record(
        first_previous,
        phase_id="cleanup.core.consecutive_mul",
        owner_expression=(
            "run_consecutive_mul_constants_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )
    assert ast.unparse(body[sanitizer_indices[1] - 1]) == (
        "_set_post_progress_desc('terminal cleanup passes')"
    )


def test_primary_path_retains_quantization_successor_conv_results() -> None:
    body = _lowerer_body()
    affine_name = "_optimize_fold_conv_mul_add_affine_chains"
    activation_name = "_optimize_fuse_conv_activation_chains"
    affine_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == affine_name
    ]
    activation_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == activation_name
    ]
    late_affine_calls = _late_affine_concat_owner_calls(affine_name)
    assert len(affine_indices) == 2
    assert len(affine_indices) + len(late_affine_calls) == 3
    assert len(activation_indices) == 2
    assert activation_indices == [index + 1 for index in affine_indices]

    expected_phase_ids = (
        (
            "cleanup.core.conv_affine",
            "cleanup.core.conv_activation",
            "cleanup.core.terminal_qdq",
        ),
        (
            "cleanup.terminal.conv_affine",
            "cleanup.terminal.conv_activation",
            "cleanup.terminal.qdq",
        ),
    )
    for pair_index, (affine_index, activation_index) in enumerate(
        zip(affine_indices, activation_indices)
    ):
        affine = body[affine_index]
        _assert_phase_result_record(
            affine,
            phase_id=expected_phase_ids[pair_index][0],
            owner_expression=(
                "_optimize_fold_conv_mul_add_affine_chains(model_ir, "
                "enable_conv_add_only_fold=True, "
                "layout_state=session.layout_state)"
            ),
        )
        affine_call = _phase_result_owner(affine)
        assert isinstance(affine_call, ast.Call)
        assert isinstance(affine_call.func, ast.Name)
        assert affine_call.func.id == affine_name
        assert [ast.unparse(argument) for argument in affine_call.args] == [
            "model_ir"
        ]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in affine_call.keywords
        } == {
            "enable_conv_add_only_fold": "True",
            "layout_state": "session.layout_state",
        }

        activation = body[activation_index]
        _assert_phase_result_record(
            activation,
            phase_id=expected_phase_ids[pair_index][1],
            owner_expression=(
                "_optimize_fuse_conv_activation_chains(model_ir, "
                "layout_state=session.layout_state)"
            ),
        )
        activation_call = _phase_result_owner(activation)
        assert isinstance(activation_call, ast.Call)
        assert isinstance(activation_call.func, ast.Name)
        assert activation_call.func.id == activation_name
        assert [
            ast.unparse(argument) for argument in activation_call.args
        ] == ["model_ir"]
        assert {
            keyword.arg: ast.unparse(keyword.value)
            for keyword in activation_call.keywords
        } == {"layout_state": "session.layout_state"}

        predecessor = body[affine_index - 1]
        _assert_phase_result_record(
            predecessor,
            phase_id=expected_phase_ids[pair_index][2],
            owner_expression=(
                "run_terminal_quantize_dequantize_cleanup(model_ir, "
                "layout_state=session.layout_state, "
                "diagnostics=session.diagnostics)"
            ),
        )

    _assert_phase_result_record(
        body[activation_indices[0] + 1],
        phase_id="shape_resolution.core.dynamic_reshape",
        owner_expression="_resolve_dynamic_reshape_shapes(model_ir)",
    )
    assert _call_name(_statement_call(body[activation_indices[1] + 1])) == (
        "_optimize_transpose_pre_argmax_nhwc_terminal_chains"
    )
    late_affine = late_affine_calls[0]
    assert [ast.unparse(argument) for argument in late_affine.args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late_affine.keywords
    } == {
        "enable_conv_add_only_fold": "True",
        "layout_state": "context.layout_state",
    }


def test_primary_path_retains_late_cost_volume_conv_affine_result() -> None:
    body = _lowerer_body()
    affine_name = "_optimize_fold_conv_mul_add_affine_chains"
    late_calls = _late_affine_concat_owner_calls(affine_name)
    assert len(late_calls) == 1
    late_call = late_calls[0]
    assert [ast.unparse(argument) for argument in late_call.args] == [
        "context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in late_call.keywords
    } == {
        "enable_conv_add_only_fold": "True",
        "layout_state": "context.layout_state",
    }

    composite = _late_affine_concat_assignment(body)
    late_index = body.index(composite)
    _assert_phase_result_record(
        body[late_index - 1],
        phase_id="cleanup.late.ndhwc_cost_volume",
        owner_expression=(
            "run_late_ndhwc_cost_volume_layout_cleanup("
            "shared_model_ir_pass_context)"
        ),
    )

    following = body[late_index + 1]
    assert isinstance(following, ast.If)
    assert isinstance(following.body[1], ast.Assign)
    assert isinstance(following.body[1].targets[0], ast.Name)
    assert following.body[1].targets[0].id == NO_LAYOUT_FALLBACK_AFFINE_RESULT
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_late_cost_volume_conv_affine_stats"
        for statement in body
        for node in ast.walk(statement)
    )


def test_primary_path_retains_late_concat_composite_results() -> None:
    body = _lowerer_body()
    result = _late_affine_concat_assignment(body)
    result_index = body.index(result)

    previous = body[result_index - 1]
    _assert_phase_result_record(
        previous,
        phase_id="cleanup.late.ndhwc_cost_volume",
        owner_expression=(
            "run_late_ndhwc_cost_volume_layout_cleanup("
            "shared_model_ir_pass_context)"
        ),
    )
    concat_calls = _late_affine_concat_owner_calls(
        "run_late_concat_layout_cleanup"
    )
    assert len(concat_calls) == 1
    assert [ast.unparse(argument) for argument in concat_calls[0].args] == [
        "context"
    ]
    assert concat_calls[0].keywords == []
    assert not any(
        isinstance(node, ast.Name)
        and node.id
        in {
            "_late_cost_volume_conv_affine_stats",
            "_late_concat_layout_results",
            "late_concat_layout_state_scope",
            "_late_concat_axis3_const_layout_stats",
            "_late_concat_dequant_quantize_layout_stats",
            "_late_concat_layernorm_layout_stats",
            "_late_concat_transpose_layout_stats",
        }
        for statement in body
        for node in ast.walk(statement)
    )

    following = body[result_index + 1]
    assert isinstance(following, ast.If)
    assert isinstance(following.body[1], ast.Assign)
    assert isinstance(following.body[1].targets[0], ast.Name)
    assert following.body[1].targets[0].id == NO_LAYOUT_FALLBACK_AFFINE_RESULT

    layout_cleanup_statements = [
        statement
        for root in body
        for statement in ast.walk(root)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and _call_name(_statement_call(statement)) == "run_layout_transpose_cleanup"
    ]
    assert (
        len(layout_cleanup_statements)
        + _very_late_layout_broadcast_owner_call_count(
            "run_layout_transpose_cleanup"
        )
        == 2
    )
    assert sum(
        isinstance(statement, ast.Assign)
        for statement in layout_cleanup_statements
    ) == 0
    phase_records = [
        statement
        for statement in layout_cleanup_statements
        if _phase_result_owner(statement) is not None
    ]
    assert len(phase_records) == 1
    _assert_phase_result_record(
        phase_records[0],
        phase_id="cleanup.layout_pass_set_1.layout_transpose",
        owner_expression=(
            "run_layout_transpose_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )


def test_primary_path_retains_very_late_layout_transpose_cleanup_result() -> None:
    body = _lowerer_body()
    callback_name = "run_layout_transpose_cleanup"
    statement = _very_late_layout_broadcast_assignment(body)
    very_late_index = body.index(statement)
    assert _very_late_layout_broadcast_owner_call_count(callback_name) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_very_late_layout_transpose_cleanup_stats"
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )

    predecessor = body[very_late_index - 1]
    assert isinstance(predecessor, ast.If)
    assert ast.unparse(predecessor.test) == (
        "not optimize_layout_transpose_chains and "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )

    successor = body[very_late_index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="shape_reconciliation.primary.very_late_broadcast",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    cleanup_statements = [
        statement
        for root in body
        for statement in ast.walk(root)
        if isinstance(statement, (ast.Assign, ast.Expr))
        and _call_name(_statement_call(statement)) == callback_name
    ]
    assert (
        len(cleanup_statements)
        + _very_late_layout_broadcast_owner_call_count(callback_name)
        == 2
    )
    _assert_phase_result_record(
        cleanup_statements[0],
        phase_id="cleanup.layout_pass_set_1.layout_transpose",
        owner_expression=(
            "run_layout_transpose_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )
    assert not any(isinstance(statement, ast.Assign) for statement in cleanup_statements)


def test_primary_path_retains_very_late_broadcast_constant_repair_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_repair_rank4_channelwise_broadcast_constants_to_runtime_layout"
    )
    direct_indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(direct_indices) == 1
    final_index = direct_indices[0]
    assert _very_late_layout_broadcast_owner_call_count(callback_name) == 1
    statement = _very_late_layout_broadcast_assignment(body)
    very_late_index = body.index(statement)
    assert very_late_index < final_index
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_very_late_broadcast_repair_stats"
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )

    successor = body[very_late_index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="shape_reconciliation.primary.very_late_broadcast",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    final = body[final_index]
    assert isinstance(final, ast.Assign)
    assert isinstance(final.targets[0], ast.Name)
    assert final.targets[0].id == "final_broadcast_repair_stats"

    all_calls = [
        node
        for root in body
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert (
        len(all_calls)
        + _very_late_layout_broadcast_owner_call_count(callback_name)
        == 3
    )
    assert sum(
        ast.unparse(call_node).startswith(f"{callback_name}(fallback_ir")
        for call_node in all_calls
    ) == 1

    module_tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    module_calls = [
        node
        for node in ast.walk(module_tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert (
        len(module_calls)
        + _very_late_layout_broadcast_owner_call_count(callback_name)
        + _binary_layout_convergence_owner_call_count(callback_name)
        == 4
    )
    assert sum(
        any(
            keyword.arg == "graph_index"
            and ast.unparse(keyword.value) == "graph_index"
            for keyword in call_node.keywords
        )
        for call_node in module_calls
    ) == 0
    assert _binary_layout_convergence_owner_call_count(callback_name) == 1


def test_primary_path_retains_very_late_broadcast_shape_result() -> None:
    body = _lowerer_body()
    broadcast_index = body.index(_very_late_layout_broadcast_assignment(body))

    statement = body[broadcast_index + 1]
    _assert_phase_result_record(
        statement,
        phase_id="shape_reconciliation.primary.very_late_broadcast",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    predecessor = body[broadcast_index]
    assert isinstance(predecessor, ast.Assign)
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == LOWERER_LAYOUT_TAIL_RESULT

    successor = body[broadcast_index + 2]
    assert isinstance(successor, ast.Assign)
    assert isinstance(successor.targets[0], ast.Name)
    assert successor.targets[0].id == SHARED_LATE_RESULT
    assert ast.unparse(successor.value) == (
        "run_shared_late_reconciliation_cleanup("
        "shared_model_ir_pass_context)"
    )


def test_primary_path_retains_shared_late_shape_result() -> None:
    body = _lowerer_body()
    decision = _shared_late_assignment(body)
    decision_index = body.index(decision)
    guard_index = decision_index + 1
    guard = body[guard_index]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 1
    assert ast.unparse(guard.test) == SHARED_LATE_RESULT

    statement = guard.body[0]
    _assert_phase_result_record(
        statement,
        phase_id="shape_reconciliation.primary.shared_late",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[guard_index + 1]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == LATE_BINARY_REPAIR_RESULT
    assert ast.unparse(following.value) == (
        "run_late_binary_repair_cleanup(shared_model_ir_pass_context)"
    )


def test_primary_path_retains_late_binary_repair_shape_result() -> None:
    body = _lowerer_body()
    decision = _late_binary_repair_assignment(body)
    decision_index = body.index(decision)
    guard_index = decision_index + 1
    guard = body[guard_index]
    assert isinstance(guard, ast.If)
    assert len(guard.body) == 1
    assert ast.unparse(guard.test) == LATE_BINARY_REPAIR_RESULT

    statement = guard.body[0]
    _assert_phase_result_record(
        statement,
        phase_id="shape_reconciliation.primary.late_binary_repair",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[guard_index + 1]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert (
        following.targets[0].id
        == OPTIONAL_LATE_BINARY_LAYOUT_RECOVERY_RESULT
    )
    assert "optimize_layout_transpose_chains" in ast.unparse(following.value)
    assert (
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
        in ast.unparse(following.value)
    )


def test_primary_path_retains_guarded_elementwise_fanout_results() -> None:
    body = _lowerer_body()
    assignment = next(
        statement
        for statement in body
        if isinstance(statement, ast.Assign)
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == TERMINAL_FANOUT_SINGLETON_RESULT
    )
    call = assignment.value
    assert isinstance(call, ast.Call)
    assert isinstance(call.func, ast.Name)
    assert call.func.id == LATE_FINAL_SHAPE_TERMINAL_OWNER
    assert [ast.unparse(argument) for argument in call.args] == [
        "late_final_shape_boundary_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value) for keyword in call.keywords
    } == {"optimize_layout_transpose_chains": "optimize_layout_transpose_chains"}

    terminal_owner = next(
        node
        for node in ast.parse(
            TERMINAL_FANOUT_SINGLETON_OWNER_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
        and node.name == TERMINAL_FANOUT_SINGLETON_OWNER
    )
    terminal_calls = [
        node
        for node in ast.walk(terminal_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
    ]
    assert len(terminal_calls) == 1
    assert [ast.unparse(argument) for argument in terminal_calls[0].args] == [
        "context.model_ir"
    ]
    terminal_guard = next(
        statement
        for statement in terminal_owner.body
        if isinstance(statement, ast.If)
    )
    assert ast.unparse(terminal_guard.test) == "include_elementwise_fanout"
    assert terminal_calls[0] in list(ast.walk(terminal_guard))

    optional_owner = next(
        node
        for node in ast.parse(
            LATE_AFFINE_OPTIONAL_FANOUT_OWNER_PATH.read_text(encoding="utf-8")
        ).body
        if isinstance(node, ast.FunctionDef)
        and node.name == LATE_AFFINE_OPTIONAL_FANOUT_OWNER
    )
    optional_calls = [
        node
        for node in ast.walk(optional_owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id
        == "optimize_transpose_elementwise_roundtrip_nhwc_nchw_fanout_chains"
    ]
    assert len(optional_calls) == 1
    assert [ast.unparse(argument) for argument in optional_calls[0].args] == [
        "context.model_ir"
    ]


def test_primary_path_retains_late_reshape_layout_composite() -> None:
    body = _lowerer_body()
    statement = _late_layout_composite_assignment(body)
    index = body.index(statement)
    calls = _late_layout_composite_owner_calls(
        "run_late_reshape_layout_cleanup"
    )
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []

    preceding_assignment = body[index - 1]
    _assert_phase_result_record(
        preceding_assignment,
        phase_id="cleanup.late.ndhwc_cost_volume",
        owner_expression=(
            "run_late_ndhwc_cost_volume_layout_cleanup("
            "shared_model_ir_pass_context)"
        ),
    )
    successor = body[index + 1]
    assert isinstance(successor, ast.If)
    assert isinstance(successor.body[1], ast.Assign)
    assert isinstance(successor.body[1].targets[0], ast.Name)
    assert successor.body[1].targets[0].id == NO_LAYOUT_FALLBACK_AFFINE_RESULT


def test_primary_path_removes_late_reshape_layout_result_locals() -> None:
    body = _lowerer_body()
    old_targets = (
        "_late_expanddims_reshape_layout_stats",
        "_late_flatten_hw_reshape_layout_stats",
        "_late_nhwc_reshape_collapse_stats",
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for statement in body
        for node in ast.walk(statement)
    )
    calls = _late_layout_composite_owner_calls("run_channel_shuffle_gather")
    assert len(calls) == 1
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {
        "include_two_way_shuffle": "False",
        "include_nhwc_shuffle": "False",
    }


def test_primary_path_retains_late_attention_layout_composite() -> None:
    body = _lowerer_body()
    _late_layout_composite_assignment(body)
    calls = _late_layout_composite_owner_calls(
        "run_late_attention_layout_cleanup"
    )
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []


def test_primary_path_removes_late_attention_layout_result_locals() -> None:
    body = _lowerer_body()
    old_targets = (
        "_late_attention_qkv_reshape_stats",
        "_late_attention_gather_cleanup_stats",
        "_late_gather_axis0_reshape_stats",
        "_late_attention_preproj_ranklift_stats",
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for statement in body
        for node in ast.walk(statement)
    )


def test_primary_path_retains_late_window_layout_composite() -> None:
    body = _lowerer_body()
    statement = _late_layout_composite_assignment(body)
    index = body.index(statement)
    calls = _late_layout_composite_owner_calls(
        "run_late_window_layout_cleanup"
    )
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == ["context"]
    assert calls[0].keywords == []

    successor = body[index + 1]
    assert isinstance(successor, ast.If)
    assert isinstance(successor.body[1], ast.Assign)
    assert isinstance(successor.body[1].targets[0], ast.Name)
    assert successor.body[1].targets[0].id == NO_LAYOUT_FALLBACK_AFFINE_RESULT


def test_primary_path_removes_late_window_layout_result_locals() -> None:
    body = _lowerer_body()
    old_targets = (
        "_late_window_partition_stats",
        "_late_window_reverse_stats",
    )
    assert not any(
        isinstance(node, ast.Name) and node.id in old_targets
        for statement in body
        for node in ast.walk(statement)
    )


def test_primary_path_retains_late_final_shape_activation_convergence_result() -> None:
    body = _lowerer_body()
    _late_layout_composite_assignment(body)
    tree = ast.parse(
        LATE_FINAL_SHAPE_BOUNDARY_OWNER_PATH.read_text(encoding="utf-8")
    )
    owner = next(
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == LATE_FINAL_SHAPE_BOUNDARY_OWNER
    )
    calls = [
        node for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "run_indexed_final_shape_activation_convergence"
    ]
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context.pass_context.model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in calls[0].keywords
    } == {"layout_state": "context.pass_context.layout_state"}


def test_primary_path_retains_final_boundary_channel_layout_composite() -> None:
    body = _lowerer_body()
    statement = _final_boundary_composite_assignment(body)
    index = body.index(statement)
    calls = _final_boundary_composite_owner_calls(
        "run_final_boundary_channel_layout_cleanup"
    )
    assert len(calls) == 1
    assert [ast.unparse(argument) for argument in calls[0].args] == [
        "context.pass_context"
    ]
    assert calls[0].keywords == []
    predecessor_assignment = body[index - 1]
    _assert_phase_result_record(
        predecessor_assignment,
        phase_id="cleanup.late.ndhwc_cost_volume",
        owner_expression=(
            "run_late_ndhwc_cost_volume_layout_cleanup("
            "shared_model_ir_pass_context)"
        ),
    )
    successor = body[index + 1]
    assert isinstance(successor, ast.If)
    assert isinstance(successor.body[1], ast.Assign)
    assert isinstance(successor.body[1].targets[0], ast.Name)
    assert successor.body[1].targets[0].id == NO_LAYOUT_FALLBACK_AFFINE_RESULT


def test_primary_path_retains_terminal_boundary_input_normalization_result() -> None:
    body = _lowerer_body()
    callback_name = "run_boundary_input_normalization_cleanup"
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.boundary_input_normalization",
        owner_expression=(
            "run_boundary_input_normalization_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )

    predecessor_call = _owner_call(body[terminal_index - 1])
    assert _call_name(predecessor_call) == (
        "_optimize_terminal_softmax_transpose_after_nhwc_propagation"
    )
    assert predecessor_call is not None
    assert [ast.unparse(argument) for argument in predecessor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in predecessor_call.keywords
    } == {"layout_state": "session.layout_state"}

    successor_call = _owner_call(body[terminal_index + 1])
    assert _call_name(successor_call) == (
        "_optimize_boundary_input_transpose_channel_slice_blocks"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {"layout_state": "session.layout_state"}


def test_primary_path_retains_terminal_boundary_input_channel_slice_result() -> None:
    body = _lowerer_body()
    callback_name = "_optimize_boundary_input_transpose_channel_slice_blocks"
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.boundary_input_channel_slice",
        owner_expression=(
            "_optimize_boundary_input_transpose_channel_slice_blocks(model_ir, "
            "layout_state=session.layout_state)"
        ),
    )

    predecessor = body[index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.boundary_input_normalization",
        owner_expression=(
            "run_boundary_input_normalization_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )

    successor_call = _owner_call(body[index + 1])
    assert _call_name(successor_call) == (
        "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {"layout_state": "session.layout_state"}


def test_primary_path_retains_first_terminal_internal_channel_slice_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    first_index = indices[0]

    statement = body[first_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.internal_channel_slice",
        owner_expression=(
            "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor = body[first_index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.boundary_input_channel_slice",
        owner_expression=(
            "_optimize_boundary_input_transpose_channel_slice_blocks(model_ir, "
            "layout_state=session.layout_state)"
        ),
    )

    successor_call = _owner_call(body[first_index + 1])
    assert _call_name(successor_call) == (
        "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {"layout_state": "session.layout_state"}

def test_primary_path_removes_final_internal_channel_slice_result() -> None:
    body = _lowerer_body()
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_final_internal_channel_slice_stats"
        for statement in body
        for node in ast.walk(statement)
    )


def test_primary_path_removes_final_channel_slice_muladd_bridge_result() -> None:
    body = _lowerer_body()
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_final_channel_slice_muladd_bridge_stats"
        for statement in body
        for node in ast.walk(statement)
    )


def test_primary_path_retains_terminal_channel_slice_muladd_bridge_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.channel_slice_muladd_bridge",
        owner_expression=(
            "_optimize_transpose_channel_slice_muladd_nhwc_bridge_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor = body[terminal_index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.internal_channel_slice",
        owner_expression=(
            "_optimize_internal_transpose_channel_slice_nhwc_propagation_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )
    successor_call = _statement_call(body[terminal_index + 1])
    assert _call_name(successor_call) == (
        "_run_terminal_slice_concat_layout_recovery_sequence"
    )
    assert successor_call is not None
    assert successor_call.args == []
    assert successor_call.keywords == []



def test_primary_path_retains_terminal_boundary_stridedslice_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.boundary_stridedslice_qdq_concat",
        owner_expression=(
            "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor_call = _statement_call(body[index - 1])
    assert _call_name(predecessor_call) == (
        "_run_terminal_slice_concat_layout_recovery_sequence"
    )
    assert predecessor_call is not None
    assert predecessor_call.args == []
    assert predecessor_call.keywords == []

    successor_call = _owner_call(body[index + 1])
    assert _call_name(successor_call) == (
        "_optimize_transpose_swish_residual_concat_closure_nhwc_chains"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert successor_call.keywords == []


def test_primary_path_retains_terminal_swish_residual_closure_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_swish_residual_concat_closure_nhwc_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.swish_residual_concat_closure",
        owner_expression=(
            "_optimize_transpose_swish_residual_concat_closure_nhwc_chains("
            "model_ir)"
        ),
    )

    predecessor = body[index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.boundary_stridedslice_qdq_concat",
        owner_expression=(
            "_optimize_boundary_input_transpose_stridedslice_qdq_concat_blocks("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    successor_call = _owner_call(body[index + 1])
    assert _call_name(successor_call) == (
        "_optimize_transpose_dequant_logistic_mul_quantize_bridges"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert successor_call.keywords == []


def test_primary_path_retains_terminal_dequant_logistic_bridge_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_dequant_logistic_mul_quantize_bridges"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.dequant_logistic_mul_quantize_bridge",
        owner_expression=(
            "_optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir)"
        ),
    )

    predecessor = body[index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.swish_residual_concat_closure",
        owner_expression=(
            "_optimize_transpose_swish_residual_concat_closure_nhwc_chains("
            "model_ir)"
        ),
    )

    successor_call = _owner_call(body[index + 1])
    assert _call_name(successor_call) == (
        "_optimize_transpose_swish_qdq_nhwc_islands"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert successor_call.keywords == []


def test_primary_path_retains_terminal_swish_qdq_island_result() -> None:
    body = _lowerer_body()
    callback_name = "_optimize_transpose_swish_qdq_nhwc_islands"
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.swish_qdq_island",
        owner_expression="_optimize_transpose_swish_qdq_nhwc_islands(model_ir)",
    )

    predecessor = body[index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.dequant_logistic_mul_quantize_bridge",
        owner_expression=(
            "_optimize_transpose_dequant_logistic_mul_quantize_bridges(model_ir)"
        ),
    )

    successor_call = _owner_call(body[index + 1])
    assert _call_name(successor_call) == (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {"layout_state": "session.layout_state"}


def test_primary_path_retains_terminal_instancenorm_post_bias_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]
    assert _very_late_owner_call_count(callback_name) == 1
    assert _pre_terminal_instancenorm_owner_call_count(callback_name) == 1
    assert (
        _absolute_final_affine_instancenorm_owner_call_count(callback_name)
        == 1
    )

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.instancenorm_post_bias",
        owner_expression=(
            "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor = body[terminal_index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.swish_qdq_island",
        owner_expression="_optimize_transpose_swish_qdq_nhwc_islands(model_ir)",
    )

    successor_call = _owner_call(body[terminal_index + 1])
    assert _call_name(successor_call) == "run_normalization_pad_layout_cleanup"
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    absolute_final_calls = _absolute_final_cleanup_owner_calls(
        "run_absolute_final_affine_instancenorm_cleanup"
    )
    assert len(absolute_final_calls) == 1
    assert ast.unparse(absolute_final_calls[0]) == (
        "run_absolute_final_affine_instancenorm_cleanup(context)"
    )


def test_primary_path_retains_terminal_normalization_pad_result() -> None:
    body = _lowerer_body()
    callback_name = "run_normalization_pad_layout_cleanup"
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]

    all_calls = [
        node
        for root in body
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert len(all_calls) == 2
    loop_results = [
        node
        for root in body
        for node in ast.walk(root)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == "normalization_pad_stats"
        and _call_name(_statement_call(node)) == callback_name
    ]
    assert len(loop_results) == 1

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.normalization_pad",
        owner_expression=(
            "run_normalization_pad_layout_cleanup(model_ir, "
            "layout_state=session.layout_state, diagnostics=session.diagnostics)"
        ),
    )

    predecessor = body[terminal_index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.instancenorm_post_bias",
        owner_expression=(
            "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    successor = body[terminal_index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="cleanup.terminal.instancenorm_residual_add",
        owner_expression=(
            "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_"
            "nhwc_chains(model_ir, layout_state=session.layout_state)"
        ),
    )


def test_primary_path_retains_very_late_instancenorm_post_bias_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_posttranspose_bias_add_nhwc_chains"
    )
    statement = _very_late_assignment(body)
    index = body.index(statement)
    assert _very_late_owner_call_count(callback_name) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_very_late_instancenorm_post_bias_stats"
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )
    assert isinstance(body[index - 1], ast.If)
    assert ast.unparse(body[index - 1].test) == (
        "not optimize_layout_transpose_chains and "
        "apply_safe_transpose_reduction_lite_on_no_layout_opt"
    )
    _assert_phase_result_record(
        body[index + 1],
        phase_id="shape_reconciliation.primary.very_late_broadcast",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )


def test_primary_path_retains_very_late_instancenorm_residual_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
    )
    _very_late_assignment(body)
    assert _very_late_owner_call_count(callback_name) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_very_late_instancenorm_residual_mul_concat_stats"
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )
    all_calls = [
        node
        for statement in body
        for node in ast.walk(statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert (
        len(all_calls)
        + _very_late_owner_call_count(callback_name)
        + _pre_terminal_instancenorm_owner_call_count(callback_name)
        == 4
    )
    nested_call = next(
        call
        for call in all_calls
        if any(keyword.arg == "graph_index" for keyword in call.keywords)
    )
    assert [ast.unparse(argument) for argument in nested_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in nested_call.keywords
    } == {
        "graph_index": "residual_graph_index",
        "layout_state": "session.layout_state",
    }


def test_primary_path_retains_terminal_instancenorm_residual_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]
    assert _very_late_owner_call_count(callback_name) == 1
    assert _pre_terminal_instancenorm_owner_call_count(callback_name) == 1

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.instancenorm_residual_mul_concat",
        owner_expression=(
            "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor_call = _owner_call(body[terminal_index - 1])
    assert _call_name(predecessor_call) == (
        "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains"
    )
    assert predecessor_call is not None
    assert [ast.unparse(argument) for argument in predecessor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in predecessor_call.keywords
    } == {"layout_state": "session.layout_state"}

    successor_call = _owner_call(body[terminal_index + 1])
    assert _call_name(successor_call) == (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {"layout_state": "session.layout_state"}

    all_calls = [
        node
        for root in body
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert (
        len(all_calls)
        + _very_late_owner_call_count(callback_name)
        + _pre_terminal_instancenorm_owner_call_count(callback_name)
        == 4
    )
    assert sum(
        any(keyword.arg == "graph_index" for keyword in call_node.keywords)
        for call_node in all_calls
    ) == 1


def test_primary_path_retains_terminal_instancenorm_residual_add_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_nhwc_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.instancenorm_residual_add",
        owner_expression=(
            "_optimize_transpose_instancenorm_residual_add_to_single_post_adapter_"
            "nhwc_chains(model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor_call = _owner_call(body[terminal_index - 1])
    assert _call_name(predecessor_call) == "run_normalization_pad_layout_cleanup"
    assert predecessor_call is not None
    assert [ast.unparse(argument) for argument in predecessor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in predecessor_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    successor = body[terminal_index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="cleanup.terminal.instancenorm_residual_mul_concat",
        owner_expression=(
            "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    all_calls = [
        node
        for root in body
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert len(all_calls) == 2
    assert sum(
        any(keyword.arg == "graph_index" for keyword in call_node.keywords)
        for call_node in all_calls
    ) == 1
    indexed_call = next(
        call_node
        for call_node in all_calls
        if any(keyword.arg == "graph_index" for keyword in call_node.keywords)
    )
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in indexed_call.keywords
    } == {
        "graph_index": "residual_graph_index",
        "layout_state": "session.layout_state",
    }


def test_primary_path_retains_very_late_instancenorm_dualstats_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    _very_late_assignment(body)
    assert _very_late_owner_call_count(callback_name) == 1
    assert not any(
        isinstance(node, ast.Name)
        and node.id == "_very_late_instancenorm_dualstats_stats"
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
    )
    all_calls = [
        node
        for body_statement in body
        for node in ast.walk(body_statement)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert (
        len(all_calls)
        + _very_late_owner_call_count(callback_name)
        + _pre_terminal_instancenorm_owner_call_count(callback_name)
        == 4
    )
    nested_call = next(
        call
        for call in all_calls
        if any(keyword.arg == "graph_index" for keyword in call.keywords)
    )
    assert [ast.unparse(argument) for argument in nested_call.args] == ["model_ir"]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in nested_call.keywords
    } == {
        "graph_index": "residual_graph_index",
        "layout_state": "session.layout_state",
    }


def test_primary_path_retains_terminal_instancenorm_dualstats_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_transpose_instancenorm_dualstats_residual_add_resize_nhwc_chains"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    terminal_index = indices[0]
    assert _very_late_owner_call_count(callback_name) == 1
    assert _pre_terminal_instancenorm_owner_call_count(callback_name) == 1

    statement = body[terminal_index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.instancenorm_dualstats",
        owner_expression=(
            "_optimize_transpose_instancenorm_dualstats_residual_add_resize_"
            "nhwc_chains(model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor = body[terminal_index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.instancenorm_residual_mul_concat",
        owner_expression=(
            "_optimize_transpose_instancenorm_residual_mul_concat_conv_nhwc_chains("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    successor_call = _statement_call(body[terminal_index + 1])
    assert _call_name(successor_call) == (
        "run_terminal_boundary_mean_attention_cleanup"
    )
    assert successor_call is not None
    assert [ast.unparse(argument) for argument in successor_call.args] == [
        "shared_model_ir_pass_context"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in successor_call.keywords
    } == {"include_mean_attention": "optimize_layout_transpose_chains"}

    all_calls = [
        node
        for root in body
        for node in ast.walk(root)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert (
        len(all_calls)
        + _very_late_owner_call_count(callback_name)
        + _pre_terminal_instancenorm_owner_call_count(callback_name)
        == 4
    )
    assert sum(
        any(keyword.arg == "graph_index" for keyword in call_node.keywords)
        for call_node in all_calls
    ) == 1


def test_primary_path_retains_terminal_softmax_transpose_result() -> None:
    body = _lowerer_body()
    callback_name = (
        "_optimize_terminal_softmax_transpose_after_nhwc_propagation"
    )
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.softmax_transpose",
        owner_expression=(
            "_optimize_terminal_softmax_transpose_after_nhwc_propagation("
            "model_ir, layout_state=session.layout_state)"
        ),
    )

    predecessor_call = _owner_call(body[index - 1])
    assert _call_name(predecessor_call) == (
        "run_transpose_gather_channel_fanout_cleanup"
    )
    assert predecessor_call is not None
    assert [ast.unparse(argument) for argument in predecessor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in predecessor_call.keywords
    } == {
        "layout_state": "session.layout_state",
        "diagnostics": "session.diagnostics",
    }

    successor = body[index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="cleanup.terminal.boundary_input_normalization",
        owner_expression=(
            "run_boundary_input_normalization_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )


def test_primary_path_retains_terminal_gather_channel_fanout_result() -> None:
    body = _lowerer_body()
    callback_name = "run_transpose_gather_channel_fanout_cleanup"
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.transpose_gather_channel_fanout",
        owner_expression=(
            "run_transpose_gather_channel_fanout_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )

    predecessor_call = _owner_call(body[index - 1])
    assert _call_name(predecessor_call) == (
        "_optimize_transpose_pre_argmax_nhwc_terminal_chains"
    )
    assert predecessor_call is not None
    assert [ast.unparse(argument) for argument in predecessor_call.args] == [
        "model_ir"
    ]
    assert {
        keyword.arg: ast.unparse(keyword.value)
        for keyword in predecessor_call.keywords
    } == {"layout_state": "session.layout_state"}

    successor = body[index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="cleanup.terminal.softmax_transpose",
        owner_expression=(
            "_optimize_terminal_softmax_transpose_after_nhwc_propagation("
            "model_ir, layout_state=session.layout_state)"
        ),
    )


def test_primary_path_retains_terminal_pre_argmax_result() -> None:
    body = _lowerer_body()
    callback_name = "_optimize_transpose_pre_argmax_nhwc_terminal_chains"
    indices = [
        index
        for index, statement in enumerate(body)
        if _call_name(_statement_call(statement)) == callback_name
    ]
    assert len(indices) == 1
    index = indices[0]

    statement = body[index]
    _assert_phase_result_record(
        statement,
        phase_id="cleanup.terminal.pre_argmax",
        owner_expression=(
            "_optimize_transpose_pre_argmax_nhwc_terminal_chains(model_ir, "
            "layout_state=session.layout_state)"
        ),
    )

    predecessor = body[index - 1]
    _assert_phase_result_record(
        predecessor,
        phase_id="cleanup.terminal.conv_activation",
        owner_expression=(
            "_optimize_fuse_conv_activation_chains(model_ir, "
            "layout_state=session.layout_state)"
        ),
    )

    successor = body[index + 1]
    _assert_phase_result_record(
        successor,
        phase_id="cleanup.terminal.transpose_gather_channel_fanout",
        owner_expression=(
            "run_transpose_gather_channel_fanout_cleanup(model_ir, "
            "layout_state=session.layout_state, "
            "diagnostics=session.diagnostics)"
        ),
    )


def test_primary_path_stages_final_consecutive_reshape_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_consecutive_reshape_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    get_calls = [
        node
        for node in ast.walk(guard.test)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "get"
    ]
    assert {
        str(call.args[0].value)
        for call in get_calls
        if len(call.args) >= 1 and isinstance(call.args[0], ast.Constant)
    } == {
        "removed_noop_reshape_chains",
        "rewritten_consecutive_reshape_passthrough_chains",
        "rewritten_fanout_bypass_reshape_passthrough_chains",
    }
    assert len(get_calls) == 3
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_consecutive_reshape",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_sinet_late_residual_stats"


def test_primary_path_stages_final_sinet_concat_resize_reconciliation() -> None:
    body = _lowerer_body()
    stats_index = next(
        index
        for index, statement in enumerate(body)
        if isinstance(statement, ast.Assign)
        and len(statement.targets) == 1
        and isinstance(statement.targets[0], ast.Name)
        and statement.targets[0].id == "final_sinet_concat_resize_stats"
    )

    guard = body[stats_index + 1]
    assert isinstance(guard, ast.If)
    assert ast.unparse(guard.test) == (
        "int(final_sinet_concat_resize_stats.get("
        "'optimized_sinet_concat_resize_affine_transpose_chains', 0)) > 0"
    )
    assert len(guard.body) == 1
    reconciliation = guard.body[0]
    _assert_phase_result_record(
        reconciliation,
        phase_id="shape_reconciliation.primary.final_sinet_concat_resize",
        owner_expression=(
            "_reconcile_static_tensor_shapes(model_ir, "
            "include_mutation_count=True)"
        ),
    )

    following = body[stats_index + 2]
    assert isinstance(following, ast.Assign)
    assert isinstance(following.targets[0], ast.Name)
    assert following.targets[0].id == "final_high_rank_bmm_stats"


def test_primary_path_stages_remaining_final_sinet_reconciliations() -> None:
    body = _lowerer_body()
    owners = (
        (
            "final_sinet_late_residual_stats",
            "shape_reconciliation.primary.final_sinet_late_residual",
            "optimized_sinet_late_residual_pre_add_mul_add_prelu_chains",
            "final_sinet_preadd_fanout_stats",
        ),
        (
            "final_sinet_preadd_fanout_stats",
            "shape_reconciliation.primary.final_sinet_preadd_fanout",
            "optimized_sinet_deep_skip_pre_add_concat_prelu_fanout_chains",
            "final_sinet_dual_resize_stats",
        ),
        (
            "final_sinet_dual_resize_stats",
            "shape_reconciliation.primary.final_sinet_dual_resize",
            "optimized_sinet_deep_skip_dual_resize_affine_transpose_chains",
            "final_sinet_shared_post_stats",
        ),
        (
            "final_sinet_shared_post_stats",
            "shape_reconciliation.primary.final_sinet_shared_post",
            "optimized_sinet_shared_post_prelu_transpose_fanout_chains",
            "final_sinet_deep_skip_stats",
        ),
        (
            "final_sinet_deep_skip_stats",
            "shape_reconciliation.primary.final_sinet_deep_skip",
            "optimized_sinet_deep_skip_concat_resize_affine_tail_chains",
            "final_sinet_concat_resize_stats",
        ),
    )

    for stats_name, phase_id, stats_key, following_name in owners:
        stats_index = next(
            index
            for index, statement in enumerate(body)
            if isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
            and statement.targets[0].id == stats_name
        )

        guard = body[stats_index + 1]
        assert isinstance(guard, ast.If)
        get_calls = [
            node
            for node in ast.walk(guard.test)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
        ]
        assert len(get_calls) == 1
        assert isinstance(get_calls[0].args[0], ast.Constant)
        assert get_calls[0].args[0].value == stats_key
        assert len(guard.body) == 1
        reconciliation = guard.body[0]
        _assert_phase_result_record(
            reconciliation,
            phase_id=phase_id,
            owner_expression=(
                "_reconcile_static_tensor_shapes(model_ir, "
                "include_mutation_count=True)"
            ),
        )

        following = body[stats_index + 2]
        assert isinstance(following, ast.Assign)
        assert isinstance(following.targets[0], ast.Name)
        assert following.targets[0].id == following_name
