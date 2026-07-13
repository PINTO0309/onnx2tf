from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.model_ir_pass_state import ModelIRPassState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_transpose_unary_binary_full_post_fanout_bridges,
)
from onnx2tf.tflite_builder.passes.layout_transpose import (
    run_transpose_unary_binary_fanout_bridge_cleanup,
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
    model_ir = ModelIR("transpose_unary_binary_fanout")
    model_ir.inputs = ["x", "y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 3, 4]),
        "y": _tensor("y", [1, 2, 3, 4]),
        "to_nchw_x": _tensor(
            "to_nchw_x",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "to_nchw_y": _tensor(
            "to_nchw_y",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "x_t": _tensor("x_t", [1, 4, 2, 3]),
        "y_t": _tensor("y_t", [1, 4, 2, 3]),
        "u_t": _tensor("u_t", [1, 4, 2, 3]),
        "z_t": _tensor("z_t", [1, 4, 2, 3]),
        "to_nhwc_0": _tensor(
            "to_nhwc_0",
            [4],
            dtype="INT32",
            data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "z_0": _tensor("z_0", [1, 2, 3, 4]),
        "output_0": _tensor("output_0", [1, 2, 3, 4]),
    }
    return model_ir


def test_unary_binary_fanout_characterization_preserves_rhs_sub_order() -> None:
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
            "z_1": _tensor("z_1", [1, 2, 3, 4]),
            "output_1": _tensor("output_1", [1, 2, 3, 4]),
        }
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "to_nchw_x"], ["x_t"]),
        OperatorIR("RELU", ["x_t"], ["u_t"]),
        OperatorIR("TRANSPOSE", ["y", "to_nchw_y"], ["y_t"]),
        OperatorIR("SUB", ["y_t", "u_t"], ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "to_nhwc_0"], ["z_0"]),
        OperatorIR("TRANSPOSE", ["z_t", "to_nhwc_1"], ["z_1"]),
        OperatorIR("IDENTITY", ["z_0"], ["output_0"]),
        OperatorIR("IDENTITY", ["z_1"], ["output_1"]),
    ]

    stats = _optimize_transpose_unary_binary_full_post_fanout_bridges(model_ir)

    assert stats["rewritten_transpose_unary_binary_full_post_fanout_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "SUB",
        "IDENTITY",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[1].inputs == ["y", "u_t"]
    assert model_ir.operators[1].outputs == ["z_0"]
    assert model_ir.operators[2].inputs == ["z_0"]
    assert model_ir.operators[3].inputs == ["z_0"]
    assert model_ir.tensors["u_t"].shape == [1, 2, 3, 4]


def test_unary_binary_fanout_characterization_keeps_legacy_adapter() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0", "legacy_output"]
    model_ir.tensors["legacy_output"] = _tensor(
        "legacy_output",
        [1, 4, 2, 3],
    )
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "to_nchw_x"], ["x_t"]),
        OperatorIR("RELU", ["x_t"], ["u_t"]),
        OperatorIR("TRANSPOSE", ["y", "to_nchw_y"], ["y_t"]),
        OperatorIR("ADD", ["u_t", "y_t"], ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "to_nhwc_0"], ["z_0"]),
        OperatorIR("IDENTITY", ["z_0"], ["output_0"]),
        OperatorIR("IDENTITY", ["z_t"], ["legacy_output"]),
    ]

    stats = _optimize_transpose_unary_binary_full_post_fanout_bridges(model_ir)

    assert stats["rewritten_transpose_unary_binary_full_post_fanout_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "ADD",
        "TRANSPOSE",
        "IDENTITY",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[1].inputs == ["u_t", "y"]
    assert model_ir.operators[1].outputs == ["z_0"]
    adapter = model_ir.operators[2]
    assert adapter.inputs == ["z_0", "to_nhwc_0"]
    assert adapter.outputs == ["z_t"]
    np.testing.assert_array_equal(
        model_ir.tensors["to_nhwc_0"].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert model_ir.operators[4].inputs == ["z_t"]


def test_unary_binary_fanout_runner_rewrites_with_one_index(monkeypatch) -> None:
    model_ir = _base_model()
    model_ir.outputs = ["output_0"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "to_nchw_x"], ["x_t"]),
        OperatorIR("RELU", ["x_t"], ["u_t"]),
        OperatorIR("TRANSPOSE", ["y", "to_nchw_y"], ["y_t"]),
        OperatorIR("ADD", ["u_t", "y_t"], ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "to_nhwc_0"], ["z_0"]),
        OperatorIR("IDENTITY", ["z_0"], ["output_0"]),
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

    stats = run_transpose_unary_binary_fanout_bridge_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_transpose_unary_binary_full_post_fanout_bridges"] == 1
    assert [operator.op_type for operator in model_ir.operators] == [
        "RELU",
        "ADD",
        "IDENTITY",
    ]
    assert model_ir.operators[0].inputs == ["x"]
    assert model_ir.operators[1].inputs == ["u_t", "y"]
    assert model_ir.operators[1].outputs == ["z_0"]
    assert refresh_count == 1
    assert snapshot_count == 1
    assert diagnostics[0]["code"] == "layout.transpose_unary_binary_fanout_bridge"
    assert diagnostics[0]["status"] == "changed"
    assert diagnostics[0]["metrics"]["snapshot_count"] == 1


def test_unary_binary_fanout_runner_rejects_public_post_output() -> None:
    model_ir = _base_model()
    model_ir.outputs = ["z_0"]
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "to_nchw_x"], ["x_t"]),
        OperatorIR("RELU", ["x_t"], ["u_t"]),
        OperatorIR("TRANSPOSE", ["y", "to_nchw_y"], ["y_t"]),
        OperatorIR("ADD", ["u_t", "y_t"], ["z_t"]),
        OperatorIR("TRANSPOSE", ["z_t", "to_nhwc_0"], ["z_0"]),
    ]
    diagnostics: list[dict] = []

    stats = run_transpose_unary_binary_fanout_bridge_cleanup(
        model_ir,
        diagnostics=diagnostics,
    )

    assert stats["rewritten_transpose_unary_binary_full_post_fanout_bridges"] == 0
    assert [operator.op_type for operator in model_ir.operators] == [
        "TRANSPOSE",
        "RELU",
        "TRANSPOSE",
        "ADD",
        "TRANSPOSE",
    ]
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
