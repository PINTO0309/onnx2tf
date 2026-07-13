from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_layout_transpose_cleanup,
)


def _tensor(name: str, shape: list[int]) -> TensorIR:
    return TensorIR(
        name=name,
        dtype="FLOAT32",
        shape=list(shape),
        shape_signature=list(shape),
        is_variable=False,
    )


def _perm(model_ir: ModelIR, name: str, values: list[int]) -> None:
    model_ir.tensors[name] = TensorIR(
        name=name,
        dtype="INT32",
        shape=[len(values)],
        shape_signature=[len(values)],
        data=np.asarray(values, dtype=np.int32),
        is_variable=False,
    )


def test_layout_transpose_runner_removes_identity_with_one_index(monkeypatch) -> None:
    model_ir = ModelIR("identity_transpose")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3]),
        "middle": _tensor("middle", [1, 2, 3]),
        "output": _tensor("output", [1, 2, 3]),
    }
    _perm(model_ir, "identity_perm", [0, 1, 2])
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "identity_perm"], ["middle"]),
        OperatorIR("RELU", ["middle"], ["output"]),
    ]
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    diagnostics: list[dict] = []
    stats = run_layout_transpose_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["removed_identity_transpose"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].inputs == ["input"]
    assert refresh_count == 1
    assert diagnostics[0]["code"] == "layout.transpose_chain_cleanup"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_layout_transpose_runner_removes_strict_inverse_pair() -> None:
    model_ir = ModelIR("inverse_transpose_pair")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3, 4]),
        "middle0": _tensor("middle0", [1, 3, 4, 2]),
        "middle1": _tensor("middle1", [1, 2, 3, 4]),
        "output": _tensor("output", [1, 2, 3, 4]),
    }
    _perm(model_ir, "perm", [0, 2, 3, 1])
    _perm(model_ir, "inverse_perm", [0, 3, 1, 2])
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "perm"], ["middle0"]),
        OperatorIR("TRANSPOSE", ["middle0", "inverse_perm"], ["middle1"]),
        OperatorIR("RELU", ["middle1"], ["output"]),
    ]

    stats = run_layout_transpose_cleanup(model_ir)

    assert stats["removed_inverse_transpose_pairs"] == 1
    assert [op.op_type for op in model_ir.operators] == ["RELU"]
    assert model_ir.operators[0].inputs == ["input"]


def test_layout_transpose_runner_composes_noninverse_pair() -> None:
    model_ir = ModelIR("composed_transpose_pair")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    for name in ("input", "middle0", "middle1", "output"):
        model_ir.tensors[name] = _tensor(name, [2, 3, 4])
    _perm(model_ir, "perm0", [1, 0, 2])
    _perm(model_ir, "perm1", [0, 2, 1])
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "perm0"], ["middle0"]),
        OperatorIR("TRANSPOSE", ["middle0", "perm1"], ["middle1"]),
        OperatorIR("RELU", ["middle1"], ["output"]),
    ]

    stats = run_layout_transpose_cleanup(model_ir)

    assert stats["composed_consecutive_transpose_pairs"] == 1
    assert [op.op_type for op in model_ir.operators] == ["TRANSPOSE", "RELU"]
    assert model_ir.operators[0].inputs[0] == "input"
    composed_perm = model_ir.tensors[model_ir.operators[0].inputs[1]].data
    np.testing.assert_array_equal(composed_perm, np.asarray([1, 2, 0], dtype=np.int32))


def test_layout_transpose_runner_skips_isolated_nonidentity() -> None:
    model_ir = ModelIR("isolated_transpose")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3]),
        "output": _tensor("output", [1, 3, 2]),
    }
    _perm(model_ir, "perm", [0, 2, 1])
    model_ir.operators = [OperatorIR("TRANSPOSE", ["input", "perm"], ["output"])]
    diagnostics: list[dict] = []

    stats = run_layout_transpose_cleanup(model_ir, diagnostics=diagnostics)

    assert stats["removed_identity_transpose"] == 0
    assert model_ir.operators[0].op_type == "TRANSPOSE"
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
