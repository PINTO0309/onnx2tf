from __future__ import annotations

import ast
import copy
from pathlib import Path

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    QuantParamIR,
    TensorIR,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _repair_rank4_binary_layout_mismatch_with_transpose_adapter,
    _repair_rank4_binary_singleton_broadcast_layout_mismatch,
)
from onnx2tf.tflite_builder.passes.binary_layout_adapter import (
    repair_rank4_binary_layout_mismatch_with_transpose_adapter,
    repair_rank4_binary_singleton_broadcast_layout_mismatch,
    run_indexed_binary_layout_adapter_cleanup,
)
from onnx2tf.tflite_builder.passes.split_all_outputs_layout import _freeze


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "binary_layout_adapter.py"
)
SHARED_LATE_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "shared_late_reconciliation_orchestration.py"
)
LATE_BINARY_REPAIR_OWNER_PATH = (
    REPO_ROOT
    / "onnx2tf"
    / "tflite_builder"
    / "passes"
    / "late_binary_repair_orchestration.py"
)
EXACT_OWNER = "repair_rank4_binary_layout_mismatch_with_transpose_adapter"
SINGLETON_OWNER = "repair_rank4_binary_singleton_broadcast_layout_mismatch"
EXACT_WRAPPER = f"_{EXACT_OWNER}"
SINGLETON_WRAPPER = f"_{SINGLETON_OWNER}"
RUNNER = "run_indexed_binary_layout_adapter_cleanup"
SUMMARY_RUNNER = "run_indexed_binary_layout_adapter_summary"
EXPECTED_PAIR_TARGETS = (
    ("shared_binary_adapter_stats", "shared_singleton_adapter_stats"),
    ("late_binary_adapter_stats", "late_singleton_adapter_stats"),
    ("_fallback_binary_adapter_stats", "_fallback_singleton_adapter_stats"),
)
EXPECTED_MODEL_ARGUMENTS = (
    "model_ir",
    "model_ir",
    "fallback_ir",
)
RESULT_SCHEMA = (
    {"inserted_rank4_binary_layout_fix_transpose": 0},
    {"repaired_rank4_binary_singleton_broadcast_layout_mismatch": 0},
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
    return statement.value if isinstance(statement.value, ast.Call) else None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _assignment_targets(statement: ast.stmt) -> tuple[str, ...]:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        return ()
    target = statement.targets[0]
    if isinstance(target, ast.Name):
        return (target.id,)
    if isinstance(target, (ast.Tuple, ast.List)):
        return tuple(
            element.id
            for element in target.elts
            if isinstance(element, ast.Name)
        )
    return ()


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


def _raw_pair_locations(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index in range(len(block) - 1)
            if _call_name(block[index]) == EXACT_WRAPPER
            and _call_name(block[index + 1]) == SINGLETON_WRAPPER
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _runner_locations(
    lowerer: ast.FunctionDef,
) -> list[tuple[list[ast.stmt], int]]:
    return sorted(
        [
            (block, index)
            for block in _pipeline_blocks(lowerer.body)
            for index, statement in enumerate(block)
            if _call_name(statement) == RUNNER
        ],
        key=lambda item: item[0][item[1]].lineno,
    )


def _shared_late_runner_calls() -> list[ast.Call]:
    owner = _functions(SHARED_LATE_OWNER_PATH)[
        "run_shared_late_reconciliation_cleanup"
    ]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RUNNER
    ]


def _late_binary_repair_runner_calls() -> list[ast.Call]:
    owner = _functions(LATE_BINARY_REPAIR_OWNER_PATH)[
        "run_late_binary_repair_cleanup"
    ]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RUNNER
    ]


def _summary_runner_calls() -> list[ast.Call]:
    owner = _functions(OWNER_PATH)[SUMMARY_RUNNER]
    return [
        node
        for node in ast.walk(owner)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == RUNNER
    ]


def _tensor(
    name: str,
    shape: tuple[int, ...],
    *,
    quantization: QuantParamIR | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        quantization=quantization,
    )


def _make_combined_adapter_model_ir() -> ModelIR:
    model_ir = ModelIR("indexed_binary_adapter_cleanup")
    model_ir.inputs = ["exact_lhs", "exact_rhs", "singleton_lhs", "singleton_rhs"]
    model_ir.outputs = ["exact_out", "singleton_out"]
    model_ir.tensors = {
        "exact_lhs": _tensor("exact_lhs", (1, 4, 5, 3)),
        "exact_rhs": _tensor(
            "exact_rhs",
            (1, 3, 4, 5),
            quantization=QuantParamIR(scale=[0.25], zero_point=[3]),
        ),
        "exact_out": _tensor("exact_out", (1, 4, 5, 3)),
        "singleton_lhs": _tensor("singleton_lhs", (1, 1, 4, 5)),
        "singleton_rhs": _tensor(
            "singleton_rhs",
            (1, 4, 5, 3),
            quantization=QuantParamIR(scale=[0.125], zero_point=[1]),
        ),
        "singleton_out": _tensor("singleton_out", (1, 4, 5, 3)),
    }
    model_ir.operators = [
        OperatorIR(
            "ADD",
            ["exact_lhs", "exact_rhs"],
            ["exact_out"],
        ),
        OperatorIR(
            "MUL",
            ["singleton_lhs", "singleton_rhs"],
            ["singleton_out"],
        ),
    ]
    return model_ir


def _snapshot(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.shape_signature or tensor.shape),
                _freeze(tensor.data),
                _freeze(tensor.quantization),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        tuple(
            (
                operator.op_type,
                tuple(operator.inputs),
                tuple(operator.outputs),
                _freeze(operator.options),
            )
            for operator in model_ir.operators
        ),
    )


def test_binary_adapter_owner_schemas_and_indexed_contracts_are_explicit() -> None:
    assert repair_rank4_binary_layout_mismatch_with_transpose_adapter(
        ModelIR("exact_schema")
    ) == RESULT_SCHEMA[0]
    assert repair_rank4_binary_singleton_broadcast_layout_mismatch(
        ModelIR("singleton_schema")
    ) == RESULT_SCHEMA[1]

    owner_functions = _functions(OWNER_PATH)
    for owner_name in (EXACT_OWNER, SINGLETON_OWNER):
        owner = owner_functions[owner_name]
        assert [argument.arg for argument in owner.args.args] == ["model_ir"]
        assert [argument.arg for argument in owner.args.kwonlyargs] == [
            "graph_index",
            "layout_state",
        ]
        assert [ast.unparse(default) for default in owner.args.kw_defaults] == [
            "None",
            "None",
        ]

    lowerer = _lowerer()
    assert _raw_pair_locations(lowerer) == []
    locations = _runner_locations(lowerer)
    shared_late_calls = _shared_late_runner_calls()
    late_binary_calls = _late_binary_repair_runner_calls()
    summary_calls = _summary_runner_calls()
    assert (
        len(locations)
        + len(shared_late_calls)
        + len(late_binary_calls)
        + len(summary_calls)
        == 4
    )
    assert tuple(
        _assignment_targets(block[index]) for block, index in locations
    ) == EXPECTED_PAIR_TARGETS[2:]
    for (block, index), model_argument in zip(
        locations,
        EXPECTED_MODEL_ARGUMENTS[2:],
    ):
        call = _statement_call(block[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            model_argument
        ]
        assert call.keywords == []
    assert len(shared_late_calls) == 1
    assert [
        ast.unparse(argument) for argument in shared_late_calls[0].args
    ] == ["context.model_ir"]
    assert shared_late_calls[0].keywords == []
    assert len(late_binary_calls) == 1
    assert [
        ast.unparse(argument) for argument in late_binary_calls[0].args
    ] == ["context.model_ir"]
    assert late_binary_calls[0].keywords == []
    assert len(summary_calls) == 1
    assert [ast.unparse(argument) for argument in summary_calls[0].args] == [
        "model_ir"
    ]


def test_binary_adapter_compatibility_wrappers_preserve_current_contract() -> None:
    exact_model_ir = _make_combined_adapter_model_ir()
    singleton_model_ir = _make_combined_adapter_model_ir()
    assert _repair_rank4_binary_layout_mismatch_with_transpose_adapter(
        exact_model_ir
    ) == {"inserted_rank4_binary_layout_fix_transpose": 1}
    assert _repair_rank4_binary_singleton_broadcast_layout_mismatch(
        singleton_model_ir
    ) == {
        "repaired_rank4_binary_singleton_broadcast_layout_mismatch": 1,
    }


def test_indexed_binary_adapter_runner_reuses_one_index_and_retains_results(
    monkeypatch,
) -> None:
    owner_functions = _functions(OWNER_PATH)
    for owner_name in (EXACT_OWNER, SINGLETON_OWNER):
        owner = owner_functions[owner_name]
        assert [argument.arg for argument in owner.args.kwonlyargs] == [
            "graph_index",
            "layout_state",
        ]
        assert [ast.unparse(default) for default in owner.args.kw_defaults] == [
            "None",
            "None",
        ]
        owner_source = ast.get_source_segment(
            OWNER_PATH.read_text(encoding="utf-8"),
            owner,
        )
        assert owner_source is not None
        assert "model_ir.operators.insert(" not in owner_source
        assert "graph_index.insert_operator(" in owner_source
        assert "graph_index=graph_index" in owner_source

    expected_ir = _make_combined_adapter_model_ir()
    actual_ir = copy.deepcopy(expected_ir)
    expected_results = (
        repair_rank4_binary_layout_mismatch_with_transpose_adapter(expected_ir),
        repair_rank4_binary_singleton_broadcast_layout_mismatch(expected_ir),
    )
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    actual_results = run_indexed_binary_layout_adapter_cleanup(actual_ir)

    assert expected_results == (
        {"inserted_rank4_binary_layout_fix_transpose": 1},
        {"repaired_rank4_binary_singleton_broadcast_layout_mismatch": 1},
    )
    assert actual_results == expected_results
    assert refresh_count == 1
    assert _snapshot(actual_ir) == _snapshot(expected_ir)

    lowerer = _lowerer()
    locations = _runner_locations(lowerer)
    shared_late_calls = _shared_late_runner_calls()
    late_binary_calls = _late_binary_repair_runner_calls()
    summary_calls = _summary_runner_calls()
    assert (
        len(locations)
        + len(shared_late_calls)
        + len(late_binary_calls)
        + len(summary_calls)
        == 4
    )
    assert tuple(
        _assignment_targets(block[index])
        for block, index in locations
    ) == EXPECTED_PAIR_TARGETS[2:]
    for (block, index), model_argument in zip(
        locations,
        EXPECTED_MODEL_ARGUMENTS[2:],
    ):
        call = _statement_call(block[index])
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == [
            model_argument
        ]
        assert call.keywords == []
    assert len(shared_late_calls) == 1
    assert [
        ast.unparse(argument) for argument in shared_late_calls[0].args
    ] == ["context.model_ir"]
    assert shared_late_calls[0].keywords == []
    assert len(late_binary_calls) == 1
    assert [
        ast.unparse(argument) for argument in late_binary_calls[0].args
    ] == ["context.model_ir"]
    assert late_binary_calls[0].keywords == []
    assert len(summary_calls) == 1
    assert [ast.unparse(argument) for argument in summary_calls[0].args] == [
        "model_ir"
    ]
    assert _raw_pair_locations(lowerer) == []
    for target in EXPECTED_PAIR_TARGETS[2]:
        assert not any(
            isinstance(node, ast.Name)
            and node.id == target
            and isinstance(node.ctx, ast.Load)
            for node in ast.walk(lowerer)
        )
