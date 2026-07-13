from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.core.graph import ModelIRGraphIndex
from onnx2tf.tflite_builder.core.layout import LayoutState
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.pytorch_compat import (
    _restore_same_average_pool_exclude_pad_correction_for_native_runtime,
)


def test_same_average_pool_correction_updates_shared_graph_state(monkeypatch) -> None:
    model_ir = ModelIR(name="avg_pool_same_native_fix")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 3, 3],
        shape_signature=[1, 1, 3, 3],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 3, 3],
        shape_signature=[1, 1, 3, 3],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="AVERAGE_POOL_2D",
            inputs=["x"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "filterHeight": 3,
                "filterWidth": 3,
                "fusedActivationFunction": "NONE",
            },
        )
    )
    layout_state = LayoutState.from_model_ir(model_ir)
    refresh_count = 0
    original_refresh = ModelIRGraphIndex.refresh

    def counted_refresh(graph_index: ModelIRGraphIndex) -> None:
        nonlocal refresh_count
        refresh_count += 1
        original_refresh(graph_index)

    monkeypatch.setattr(ModelIRGraphIndex, "refresh", counted_refresh)
    graph_index = ModelIRGraphIndex(model_ir)

    stats = _restore_same_average_pool_exclude_pad_correction_for_native_runtime(
        model_ir,
        graph_index=graph_index,
        layout_state=layout_state,
    )

    assert stats["restored_same_average_pool_exclude_pad_corrections"] == 1
    assert refresh_count == 1
    assert [str(op.op_type) for op in model_ir.operators] == [
        "AVERAGE_POOL_2D",
        "MUL",
    ]
    assert str(model_ir.operators[0].outputs[0]).endswith("_include_pad")
    assert list(model_ir.operators[1].outputs) == ["y"]
    reciprocal_name = str(model_ir.operators[1].inputs[1])
    reciprocal_tensor = model_ir.tensors[reciprocal_name]
    assert reciprocal_tensor.logical_layout == "NHWC"
    assert reciprocal_tensor.shape == [1, 3, 3, 1]
    expected = np.asarray(
        [
            [[2.25], [1.5], [2.25]],
            [[1.5], [1.0], [1.5]],
            [[2.25], [1.5], [2.25]],
        ],
        dtype=np.float32,
    ).reshape(1, 3, 3, 1)
    np.testing.assert_allclose(
        np.asarray(reciprocal_tensor.data),
        expected,
        rtol=0.0,
        atol=0.0,
    )
    assert graph_index.operator_indices("AVERAGE_POOL_2D") == [0]
    assert graph_index.operator_indices("MUL") == [1]
    assert layout_state.validate_against_model_ir(model_ir) == []
