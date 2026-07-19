from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_batchmatmul_affine_transpose_input_chains,
)
from onnx2tf.tflite_builder.passes.batchmatmul_affine_input_layout import (
    optimize_batchmatmul_affine_transpose_input_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"
POST_SINET_RESULT_TARGETS = (
    "_post_sinet_batchmatmul_affine_input_stats",
    "_post_sinet_batchmatmul_reshape_se_stats",
    "_post_sinet_batchmatmul_adj_flags_stats",
)
POST_SINET_PHASE_IDS = (
    "cleanup.post_sinet.batchmatmul_affine_input",
    "cleanup.post_sinet.batchmatmul_reshape_se",
    "cleanup.post_sinet.batchmatmul_adj_flags",
)
POST_SINET_OWNER_EXPRESSIONS = (
    "_optimize_batchmatmul_affine_transpose_input_chains(model_ir)",
    "_optimize_batchmatmul_reshape_se_nhwc_chains(model_ir)",
    "_optimize_batchmatmul_transpose_input_to_adj_flags(model_ir)",
)


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(
        statement.value,
        ast.Call,
    ):
        call = statement.value
        if (
            isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "session"
            and call.func.attr == "record_phase_result"
            and len(call.args) == 2
            and isinstance(call.args[1], ast.Call)
        ):
            return call.args[1]
        return call
    return None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


def _assert_phase_result_record(statement: ast.stmt, phase_id: str) -> None:
    assert isinstance(statement, ast.Expr)
    record = statement.value
    assert isinstance(record, ast.Call)
    assert isinstance(record.func, ast.Attribute)
    assert isinstance(record.func.value, ast.Name)
    assert record.func.value.id == "session"
    assert record.func.attr == "record_phase_result"
    assert len(record.args) == 2
    assert ast.literal_eval(record.args[0]) == phase_id


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


def _fingerprint(model_ir: ModelIR) -> tuple[object, ...]:
    return (
        tuple(model_ir.inputs),
        tuple(model_ir.outputs),
        tuple(
            (operator.op_type, tuple(operator.inputs), tuple(operator.outputs), repr(operator.options))
            for operator in model_ir.operators
        ),
        tuple(
            (
                name,
                tensor.dtype,
                tuple(tensor.shape),
                (
                    None
                    if tensor.shape_signature is None
                    else tuple(tensor.shape_signature)
                ),
                tensor.logical_layout,
                tensor.physical_layout,
                (
                    None
                    if tensor.data is None
                    else (
                        str(np.asarray(tensor.data).dtype),
                        tuple(np.asarray(tensor.data).shape),
                        tuple(np.asarray(tensor.data).reshape(-1).tolist()),
                    )
                ),
            )
            for name, tensor in sorted(model_ir.tensors.items())
        ),
        repr(model_ir.metadata),
    )


def test_flatbuffer_direct_batchmatmul_affine_transpose_input_chains() -> None:
    model_ir = ModelIR("batchmatmul_affine_transpose_inputs_test")
    model_ir.inputs = ["lhs_nhwc", "rhs_nhwc"]
    model_ir.outputs = ["bmm_out"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("lhs_nhwc", [1, 8, 8, 96])
    _add_tensor("lhs_nchw", [1, 96, 8, 8])
    _add_tensor("lhs_mul_out", [1, 96, 8, 8])
    _add_tensor("lhs_add_out", [1, 96, 8, 8])
    _add_tensor("lhs_reshape", [1, 96, 64])
    _add_tensor("lhs_mat", [1, 64, 96])

    _add_tensor("rhs_nhwc", [1, 16, 16, 96])
    _add_tensor("rhs_nchw", [1, 96, 16, 16])
    _add_tensor("rhs_mul_out", [1, 96, 16, 16])
    _add_tensor("rhs_add_out", [1, 96, 16, 16])
    _add_tensor("rhs_reshape", [1, 96, 256])

    _add_tensor("bmm_out", [1, 64, 256])

    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_swap_last2_rank3", [3], "INT32", np.asarray([0, 2, 1], dtype=np.int32))
    _add_tensor("lhs_mul_const", [1, 96, 1, 1], data=np.ones((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("lhs_add_const", [1, 96, 1, 1], data=np.zeros((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("rhs_mul_const", [1, 96, 1, 1], data=np.ones((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("rhs_add_const", [1, 96, 1, 1], data=np.zeros((1, 96, 1, 1), dtype=np.float32))
    _add_tensor("lhs_shape", [3], "INT32", np.asarray([1, 96, 64], dtype=np.int32))
    _add_tensor("rhs_shape", [3], "INT32", np.asarray([1, 96, 256], dtype=np.int32))

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["lhs_nhwc", "perm_nhwc_to_nchw"], outputs=["lhs_nchw"]),
        OperatorIR(op_type="MUL", inputs=["lhs_nchw", "lhs_mul_const"], outputs=["lhs_mul_out"]),
        OperatorIR(op_type="ADD", inputs=["lhs_mul_out", "lhs_add_const"], outputs=["lhs_add_out"]),
        OperatorIR(op_type="RESHAPE", inputs=["lhs_add_out", "lhs_shape"], outputs=["lhs_reshape"], options={"newShape": [1, 96, 64]}),
        OperatorIR(op_type="TRANSPOSE", inputs=["lhs_reshape", "perm_swap_last2_rank3"], outputs=["lhs_mat"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["rhs_nhwc", "perm_nhwc_to_nchw"], outputs=["rhs_nchw"]),
        OperatorIR(op_type="MUL", inputs=["rhs_nchw", "rhs_mul_const"], outputs=["rhs_mul_out"]),
        OperatorIR(op_type="ADD", inputs=["rhs_mul_out", "rhs_add_const"], outputs=["rhs_add_out"]),
        OperatorIR(op_type="RESHAPE", inputs=["rhs_add_out", "rhs_shape"], outputs=["rhs_reshape"], options={"newShape": [1, 96, 256]}),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_mat", "rhs_reshape"],
            outputs=["bmm_out"],
            options={"adjX": False, "adjY": False},
        ),
    ]

    owner_ir = deepcopy(model_ir)
    owner_stats = optimize_batchmatmul_affine_transpose_input_chains(owner_ir)
    stats = _optimize_batchmatmul_affine_transpose_input_chains(model_ir)
    assert owner_stats == stats
    assert _fingerprint(owner_ir) == _fingerprint(model_ir)
    assert stats["optimized_batchmatmul_affine_transpose_input_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    bmm_op = next(op for op in model_ir.operators if str(op.op_type) == "BATCH_MATMUL")
    assert list(bmm_op.inputs) == ["lhs_reshape", "rhs_reshape"]
    assert bool(dict(bmm_op.options).get("adjY", False))

    lhs_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["lhs_mul_out"])
    rhs_mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL" and list(op.outputs) == ["rhs_mul_out"])
    assert list(lhs_mul_op.inputs)[0] == "lhs_nhwc"
    assert list(rhs_mul_op.inputs)[0] == "rhs_nhwc"

    lhs_shape_vals = np.asarray(model_ir.tensors["lhs_shape"].data, dtype=np.int32).reshape(-1).tolist()
    rhs_shape_vals = np.asarray(model_ir.tensors["rhs_shape"].data, dtype=np.int32).reshape(-1).tolist()
    assert lhs_shape_vals == [1, 64, 96]
    assert rhs_shape_vals == [1, 256, 96]


def test_batchmatmul_affine_input_results_are_recorded_at_both_boundaries() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    callback_name = "_optimize_batchmatmul_affine_transpose_input_chains"
    all_calls = [
        node
        for node in ast.walk(lowerer)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == callback_name
    ]
    assert len(all_calls) == 2

    terminal_guard = next(
        statement
        for statement in lowerer.body
        if isinstance(statement, ast.If)
        and isinstance(statement.test, ast.Name)
        and statement.test.id == "optimize_layout_transpose_chains"
        and any(
            isinstance(node, ast.Name)
            and node.id == "_terminal_mean_attention_results"
            for node in ast.walk(statement)
        )
    )
    terminal_index = next(
        index
        for index, statement in enumerate(terminal_guard.body)
        if _call_name(statement) == callback_name
    )
    terminal = terminal_guard.body[terminal_index]
    _assert_phase_result_record(
        terminal,
        "cleanup.terminal.batchmatmul_affine_input",
    )
    assert terminal_index > 0
    predecessor = terminal_guard.body[terminal_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == "_terminal_mean_attention_results"
    assert _call_name(terminal_guard.body[terminal_index + 1]) == (
        "_optimize_batchmatmul_reshape_se_nhwc_chains"
    )

    post_sinet_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == callback_name
    )
    post_sinet = lowerer.body[post_sinet_index]
    _assert_phase_result_record(
        post_sinet,
        "cleanup.post_sinet.batchmatmul_affine_input",
    )
    post_sinet_predecessor = lowerer.body[post_sinet_index - 1]
    _assert_phase_result_record(
        post_sinet_predecessor,
        "cleanup.post_cleanup.sa_pa_mirrorpad",
    )
    assert _call_name(post_sinet_predecessor) == (
        "_optimize_transpose_sa_pa_mirrorpad_nhwc_propagation_chains"
    )
    assert _call_name(lowerer.body[post_sinet_index + 1]) == (
        "_optimize_batchmatmul_reshape_se_nhwc_chains"
    )

    for statement in (terminal, post_sinet):
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert call.keywords == []


def test_post_sinet_batchmatmul_results_use_phase_result_store() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    records = [
        statement
        for statement in lowerer.body
        if _phase_id(statement) in POST_SINET_PHASE_IDS
    ]
    indices = [lowerer.body.index(statement) for statement in records]

    assert tuple(_phase_id(statement) for statement in records) == POST_SINET_PHASE_IDS
    assert tuple(ast.unparse(statement.value.args[1]) for statement in records) == (
        POST_SINET_OWNER_EXPRESSIONS
    )
    assert indices == list(range(indices[0], indices[0] + 3))
    _assert_phase_result_record(
        lowerer.body[indices[0] - 1],
        "cleanup.post_cleanup.sa_pa_mirrorpad",
    )
    successor = lowerer.body[indices[-1] + 1]
    assert isinstance(successor, ast.Assign)
    assert len(successor.targets) == 1
    assert isinstance(successor.targets[0], ast.Name)
    assert successor.targets[0].id == "_post_sinet_qkv_attention_results"
    assert not any(
        isinstance(node, ast.Name)
        and node.id in POST_SINET_RESULT_TARGETS
        and isinstance(node.ctx, ast.Load)
        for node in ast.walk(lowerer)
    )
