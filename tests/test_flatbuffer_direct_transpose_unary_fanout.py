from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_unary_fanout_inverse_post_bridges,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_unary_fanout_bridge_cleanup,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _base_model() -> ModelIR:
    model_ir = ModelIR("transpose_unary_fanout")
    model_ir.inputs = ["input"]
    model_ir.tensors = {
        "input": _tensor("input", [1, 2, 3, 4]),
        "to_nchw": _tensor(
            "to_nchw",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "nchw": _tensor("nchw", [1, 4, 2, 3]),
        "relu_nchw": _tensor("relu_nchw", [1, 4, 2, 3]),
        "to_nhwc_0": _tensor(
            "to_nhwc_0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "branch_0": _tensor("branch_0", [1, 2, 3, 4]),
        "output_0": _tensor("output_0", [1, 2, 3, 4]),
    }
    return model_ir


def test_transpose_unary_fanout_characterization_merges_inverse_posts() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0", "output_1"]
    model_ir.tensors.update(
        {
            "to_nhwc_1": _tensor(
                "to_nhwc_1",
                [4],
                dtype="INT32",
                data=np.asarray([0, 2, 3, 1], dtype=np.int32),
            ),
            "branch_1": _tensor("branch_1", [1, 2, 3, 4]),
            "output_1": _tensor("output_1", [1, 2, 3, 4]),
        }
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_0"], ["branch_0"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_1"], ["branch_1"]),
        OperatorIR("IDENTITY", ["branch_0"], ["output_0"]),
        OperatorIR("IDENTITY", ["branch_1"], ["output_1"]),
    ]

    stats = _optimize_transpose_unary_fanout_inverse_post_bridges(model_ir)

    assert stats["rewritten_transpose_unary_fanout_inverse_post_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "IDENTITY",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["branch_0"]
    assert model_ir.operators[1].inputs == ["branch_0"]
    assert model_ir.operators[2].inputs == ["branch_0"]


def test_transpose_unary_fanout_characterization_keeps_legacy_adapter() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0", "legacy_output"]
    model_ir.tensors["legacy_output"] = _tensor(
        "legacy_output",
        [1, 4, 2, 3],
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_0"], ["branch_0"]),
        OperatorIR("IDENTITY", ["branch_0"], ["output_0"]),
        OperatorIR("IDENTITY", ["relu_nchw"], ["legacy_output"]),
    ]

    stats = _optimize_transpose_unary_fanout_inverse_post_bridges(model_ir)

    assert stats["rewritten_transpose_unary_fanout_inverse_post_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "TRANSPOSE",
        "IDENTITY",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["branch_0"]
    adapter = model_ir.operators[1]
    assert adapter.inputs == ["branch_0", "to_nhwc_0"]
    assert adapter.outputs == ["relu_nchw"]
    np.testing.assert_array_equal(
        model_ir.tensors["to_nhwc_0"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert model_ir.operators[3].inputs == ["relu_nchw"]


def test_transpose_unary_fanout_runner_rewrites_with_one_index(monkeypatch) -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_0"], ["branch_0"]),
        OperatorIR("IDENTITY", ["branch_0"], ["output_0"]),
    ]
    refresh_count = 0
    snapshot_count = 0
    original_refresh = ModelIRGraphIndex.refresh
    original_snapshot = ModelIRPassState.snapshot

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    def counted_snapshot(pass_state: ModelIRPassState) -> ModelIR:
        nonlocal snapshot_count
        snapshot_count += 1
        return original_snapshot(pass_state)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    monkeypatch.setattr(ModelIRPassState, "snapshot", counted_snapshot)
    diagnostics: list[dict] = []

    stats = run_transpose_unary_fanout_bridge_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_transpose_unary_fanout_inverse_post_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["input"]
    assert model_ir.operators[0].outputs == ["branch_0"]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert diagnostics[0]["code"] == "layout.transpose_unary_fanout_bridge"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_transpose_unary_fanout_runner_rejects_public_post_output() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["branch_0"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["input", "to_nchw"], ["nchw"]),
        OperatorIR("RELU", ["nchw"], ["relu_nchw"]),
        OperatorIR("TRANSPOSE", ["relu_nchw", "to_nhwc_0"], ["branch_0"]),
    ]
    diagnostics: list[dict] = []

    stats = run_transpose_unary_fanout_bridge_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_transpose_unary_fanout_inverse_post_bridges"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "RELU",
        "TRANSPOSE",
    ]
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
