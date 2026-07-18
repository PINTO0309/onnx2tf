from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_batchmatmul_reshape_se_nhwc_chains,
)
from onnx2tf.tflite_builder.passes.batchmatmul_se_layout import (
    optimize_batchmatmul_reshape_se_nhwc_chains,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWERER_PATH = REPO_ROOT / "onnx2tf" / "tflite_builder" / "lower_from_onnx2tf.py"


def _statement_call(statement: ast.stmt) -> ast.Call | None:
    if isinstance(statement, (ast.Assign, ast.Expr)) and isinstance(
        statement.value,
        ast.Call,
    ):
        return statement.value
    return None


def _call_name(statement: ast.stmt) -> str | None:
    call = _statement_call(statement)
    if call is None or not isinstance(call.func, ast.Name):
        return None
    return call.func.id


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


def test_flatbuffer_direct_batchmatmul_reshape_se_nhwc_chains() -> None:
    model_ir = ModelIR("batchmatmul_reshape_se_nhwc_chain_test")
    model_ir.inputs = ["lhs_mat", "rhs_mat"]
    model_ir.outputs = ["z"]

    def _add_tensor(name: str, shape: list[int], dtype: str = "FLOAT32", data: np.ndarray | None = None) -> None:
        model_ir.tensors[name] = TensorIR(
            name=name,
            dtype=dtype,
            shape=[int(v) for v in shape],
            shape_signature=[int(v) for v in shape],
            data=data,
            is_variable=False if data is not None else True,
        )

    _add_tensor("lhs_mat", [1, 64, 96])
    _add_tensor("rhs_mat", [1, 256, 96])
    _add_tensor("bmm_out", [1, 64, 256])
    _add_tensor("x_nchw", [1, 64, 16, 16])
    _add_tensor("mean_nchw", [1, 64, 1, 1])
    _add_tensor("mean_nhwc", [1, 1, 1, 64])
    _add_tensor("conv1_out", [1, 1, 1, 64])
    _add_tensor("conv2_out_nhwc", [1, 1, 1, 64])
    _add_tensor("gate_nchw", [1, 64, 1, 1])
    _add_tensor("gate_sig", [1, 64, 1, 1])
    _add_tensor("y_nchw", [1, 64, 16, 16])
    _add_tensor("y_nhwc", [1, 16, 16, 64])
    _add_tensor("z", [1, 16, 16, 64])

    _add_tensor("shape_x", [4], "INT32", np.asarray([1, 64, 16, 16], dtype=np.int32))
    _add_tensor("mean_axes", [2], "INT32", np.asarray([2, 3], dtype=np.int32))
    _add_tensor("perm_nhwc_to_nchw", [4], "INT32", np.asarray([0, 3, 1, 2], dtype=np.int32))
    _add_tensor("perm_nchw_to_nhwc", [4], "INT32", np.asarray([0, 2, 3, 1], dtype=np.int32))
    _add_tensor("conv_w", [64, 1, 1, 64], data=np.ones((64, 1, 1, 64), dtype=np.float32))
    _add_tensor("conv_b", [64], data=np.zeros((64,), dtype=np.float32))

    model_ir.operators = [
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["lhs_mat", "rhs_mat"],
            outputs=["bmm_out"],
            options={"adjX": False, "adjY": True},
        ),
        OperatorIR(op_type="RESHAPE", inputs=["bmm_out", "shape_x"], outputs=["x_nchw"], options={"newShape": [1, 64, 16, 16]}),
        OperatorIR(op_type="MEAN", inputs=["x_nchw", "mean_axes"], outputs=["mean_nchw"], options={"keepDims": True, "axes": [2, 3]}),
        OperatorIR(op_type="TRANSPOSE", inputs=["mean_nchw", "perm_nchw_to_nhwc"], outputs=["mean_nhwc"]),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["mean_nhwc", "conv_w", "conv_b"],
            outputs=["conv1_out"],
            options={"padding": "SAME", "strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1},
        ),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["conv1_out", "conv_w", "conv_b"],
            outputs=["conv2_out_nhwc"],
            options={"padding": "SAME", "strideH": 1, "strideW": 1, "dilationHFactor": 1, "dilationWFactor": 1},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["conv2_out_nhwc", "perm_nhwc_to_nchw"], outputs=["gate_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["gate_nchw"], outputs=["gate_sig"]),
        OperatorIR(op_type="MUL", inputs=["x_nchw", "gate_sig"], outputs=["y_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["y_nchw", "perm_nchw_to_nhwc"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    owner_ir = deepcopy(model_ir)
    owner_stats = optimize_batchmatmul_reshape_se_nhwc_chains(owner_ir)
    stats = _optimize_batchmatmul_reshape_se_nhwc_chains(model_ir)
    assert owner_stats == stats
    assert _fingerprint(owner_ir) == _fingerprint(model_ir)
    assert stats["optimized_batchmatmul_reshape_se_nhwc_chains"] == 1
    assert all(str(op.op_type) != "TRANSPOSE" for op in model_ir.operators)

    bmm_op = next(op for op in model_ir.operators if str(op.op_type) == "BATCH_MATMUL")
    assert list(bmm_op.inputs) == ["rhs_mat", "lhs_mat"]
    assert bool(dict(bmm_op.options).get("adjY", False))

    shape_vals = np.asarray(model_ir.tensors["shape_x"].data, dtype=np.int32).reshape(-1).tolist()
    assert shape_vals == [1, 16, 16, 64]
    mean_axes_vals = np.asarray(model_ir.tensors["mean_axes"].data, dtype=np.int32).reshape(-1).tolist()
    assert mean_axes_vals == [1, 2]

    gate_op = next(op for op in model_ir.operators if str(op.op_type) == "LOGISTIC")
    assert list(gate_op.inputs) == ["conv2_out_nhwc"]

    mul_op = next(op for op in model_ir.operators if str(op.op_type) == "MUL")
    assert list(mul_op.outputs) == ["y_nhwc"]
    relu_op = next(op for op in model_ir.operators if str(op.op_type) == "RELU")
    assert list(relu_op.inputs) == ["y_nhwc"]


def test_batchmatmul_reshape_se_results_are_retained_at_both_boundaries() -> None:
    tree = ast.parse(LOWERER_PATH.read_text(encoding="utf-8"))
    lowerer = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "lower_onnx_to_ir"
    )
    callback_name = "_optimize_batchmatmul_reshape_se_nhwc_chains"
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
            and node.id == "_terminal_batchmatmul_affine_input_stats"
            for node in ast.walk(statement)
        )
    )
    terminal_index = next(
        index
        for index, statement in enumerate(terminal_guard.body)
        if _call_name(statement) == callback_name
    )
    terminal = terminal_guard.body[terminal_index]
    assert isinstance(terminal, ast.Assign)
    assert len(terminal.targets) == 1
    assert isinstance(terminal.targets[0], ast.Name)
    assert terminal.targets[0].id == "_terminal_batchmatmul_reshape_se_stats"
    predecessor = terminal_guard.body[terminal_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == (
        "_terminal_batchmatmul_affine_input_stats"
    )
    assert _call_name(terminal_guard.body[terminal_index + 1]) == (
        "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )

    post_sinet_index = next(
        index
        for index, statement in enumerate(lowerer.body)
        if _call_name(statement) == callback_name
    )
    post_sinet = lowerer.body[post_sinet_index]
    assert isinstance(post_sinet, ast.Assign)
    assert len(post_sinet.targets) == 1
    assert isinstance(post_sinet.targets[0], ast.Name)
    assert post_sinet.targets[0].id == "_post_sinet_batchmatmul_reshape_se_stats"
    predecessor = lowerer.body[post_sinet_index - 1]
    assert isinstance(predecessor, ast.Assign)
    assert isinstance(predecessor.targets[0], ast.Name)
    assert predecessor.targets[0].id == (
        "_post_sinet_batchmatmul_affine_input_stats"
    )
    assert _call_name(lowerer.body[post_sinet_index + 1]) == (
        "_optimize_batchmatmul_transpose_input_to_adj_flags"
    )

    for statement in (terminal, post_sinet):
        call = _statement_call(statement)
        assert call is not None
        assert [ast.unparse(argument) for argument in call.args] == ["model_ir"]
        assert call.keywords == []
