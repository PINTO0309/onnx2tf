from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pad_layout import (
    repair_channel_last_inputs_for_channel_first_pad,
    run_pad_mul_layout_cleanup,
)


def _tensor(
    name: str,
    shape: list[int],
    *,
    dtype: str = "FLOAT32",
    data: np.ndarray | None = None,
    layout: str = "UNKNOWN",
) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        logical_layout=layout,
        physical_layout=layout,
    )


def test_repair_channel_last_input_for_channel_first_padv2(monkeypatch) -> None:
    model_ir = ModelIR("channel_last_input_for_channel_first_pad")
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 120, 160, 3], layout="NHWC"),
        "pads": _tensor(
            "pads",
            [4, 2],
            dtype="INT32",
            data=np.asarray([[0, 0], [0, 0], [0, 8], [0, 96]], dtype=np.int32),
        ),
        "value": _tensor(
            "value",
            [1],
            data=np.asarray([1.0], dtype=np.float32),
        ),
        "y_nchw": _tensor("y_nchw", [1, 3, 128, 256], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR(
            op_type="PADV2",
            inputs=["x_nhwc", "pads", "value"],
            outputs=["y_nchw"],
        )
    ]
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = repair_channel_last_inputs_for_channel_first_pad(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats == {"repaired_channel_last_inputs_for_channel_first_pad": 1}
    assert [str(op.op_type) for op in model_ir.operators] == ["TRANSPOSE", "PADV2"]
    transpose_op, pad_op = model_ir.operators
    adapter_name = str(transpose_op.outputs[0])
    assert str(pad_op.inputs[0]) == adapter_name
    assert model_ir.tensors[adapter_name].shape == [1, 3, 120, 160]
    np.testing.assert_array_equal(
        model_ir.tensors[str(transpose_op.inputs[1])].data,
        np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    assert refresh_count == 1
    assert graph_index.operator_indices("TRANSPOSE") == [0]
    assert graph_index.operator_indices("PADV2") == [1]
    assert graph_index.producer(adapter_name) is transpose_op
    assert graph_index.consumer_indices(adapter_name) == [1]
    assert layout_state.validate_against_model_ir(model_ir) == []


def test_pad_layout_repair_is_noop_for_native_nchw_input() -> None:
    model_ir = ModelIR("native_nchw_pad")
    model_ir.tensors = {
        "x_nchw": _tensor("x_nchw", [1, 3, 120, 160], layout="NCHW"),
        "pads": _tensor(
            "pads",
            [4, 2],
            dtype="INT32",
            data=np.asarray([[0, 0], [0, 0], [0, 8], [0, 96]], dtype=np.int32),
        ),
        "y_nchw": _tensor("y_nchw", [1, 3, 128, 256], layout="NCHW"),
    }
    model_ir.operators = [
        OperatorIR(op_type="PAD", inputs=["x_nchw", "pads"], outputs=["y_nchw"])
    ]

    stats = repair_channel_last_inputs_for_channel_first_pad(model_ir)

    assert stats == {"repaired_channel_last_inputs_for_channel_first_pad": 0}
    assert [str(op.op_type) for op in model_ir.operators] == ["PAD"]


def test_pad_layout_repair_rejects_unproven_shape_mismatch() -> None:
    model_ir = ModelIR("unproven_pad_layout")
    model_ir.tensors = {
        "x_nhwc": _tensor("x_nhwc", [1, 120, 159, 3], layout="NHWC"),
        "pads": _tensor(
            "pads",
            [4, 2],
            dtype="INT32",
            data=np.asarray([[0, 0], [0, 0], [0, 8], [0, 96]], dtype=np.int32),
        ),
        "y_nchw": _tensor("y_nchw", [1, 3, 128, 256], layout="NHWC"),
    }
    model_ir.operators = [
        OperatorIR(op_type="PAD", inputs=["x_nhwc", "pads"], outputs=["y_nchw"])
    ]

    stats = repair_channel_last_inputs_for_channel_first_pad(model_ir)

    assert stats == {"repaired_channel_last_inputs_for_channel_first_pad": 0}
    assert [str(op.op_type) for op in model_ir.operators] == ["PAD"]


def test_pad_mul_runner_rejects_chain_without_post_transpose_before_snapshot() -> None:
    model_ir = ModelIR("pad_mul_without_post_transpose")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors = {
        "x": _tensor("x", [1, 2, 2, 3]),
        "perm": _tensor(
            "perm",
            [4],
            dtype="INT32",
            data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        ),
        "t": _tensor("t", [1, 3, 2, 2]),
        "pads": _tensor(
            "pads",
            [4, 2],
            dtype="INT32",
            data=np.zeros((4, 2), dtype=np.int32),
        ),
        "p": _tensor("p", [1, 3, 2, 2]),
        "scale": _tensor(
            "scale",
            [1],
            data=np.asarray([1.0], dtype=np.float32),
        ),
        "m": _tensor("m", [1, 3, 2, 2]),
        "bias": _tensor(
            "bias",
            [1],
            data=np.asarray([0.0], dtype=np.float32),
        ),
        "y": _tensor("y", [1, 3, 2, 2]),
    }
    model_ir.operators = [
        OperatorIR("TRANSPOSE", ["x", "perm"], ["t"]),
        OperatorIR("PAD", ["t", "pads"], ["p"]),
        OperatorIR("MUL", ["p", "scale"], ["m"]),
        OperatorIR("ADD", ["m", "bias"], ["y"]),
    ]
    diagnostics: list[dict] = []

    stats = run_pad_mul_layout_cleanup(model_ir, diagnostics=diagnostics)

    assert stats == {
        "optimized_transpose_pad_mul_posttranspose_add_nhwc_chains": 0,
    }
    assert [str(op.op_type) for op in model_ir.operators] == [
        "TRANSPOSE",
        "PAD",
        "MUL",
        "ADD",
    ]
    assert len(diagnostics) == 1
    assert diagnostics[0]["status"] == "skipped"
    assert diagnostics[0]["metrics"]["state_built"] is True
    assert diagnostics[0]["metrics"]["snapshot_count"] == 0
