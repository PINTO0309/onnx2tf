from __future__ import annotations

import numpy as np

from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.passes.quantized_layout import (
    repair_channel_last_convinteger_input_transposes,
)


def _tensor(name: str, dtype: str, shape: list[int], data=None) -> TensorIR:
    return TensorIR(
        name=name,
        dtype=dtype,
        shape=list(shape),
        shape_signature=list(shape),
        data=data,
        is_variable=False,
    )


def _make_convinteger_bridge_ir(*, mark_channel_last: bool) -> ModelIR:
    model_ir = ModelIR("convinteger_channel_last_bridge")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["assume_channel_last_layout_tensor_names"] = (
        ["x"] if mark_channel_last else []
    )
    model_ir.tensors = {
        "x": _tensor("x", "UINT8", [1, 6, 6, 8]),
        "x_f32_nchw": _tensor("x_f32_nchw", "FLOAT32", [1, 8, 6, 6]),
        "zero": _tensor("zero", "FLOAT32", [1]),
        "centered_nchw": _tensor("centered_nchw", "FLOAT32", [1, 8, 6, 6]),
        "perm": _tensor(
            "perm",
            "INT32",
            [4],
            np.asarray([0, 2, 3, 1], dtype=np.int32),
        ),
        "input_nhwc": _tensor("input_nhwc", "FLOAT32", [1, 6, 6, 8]),
        "filter": _tensor(
            "filter",
            "FLOAT32",
            [16, 1, 1, 8],
            np.zeros((16, 1, 1, 8), dtype=np.float32),
        ),
        "bias": _tensor(
            "bias",
            "FLOAT32",
            [16],
            np.zeros((16,), dtype=np.float32),
        ),
        "y": _tensor("y", "FLOAT32", [1, 6, 6, 16]),
    }
    provenance = {
        "onnx_node_name": "ConvInteger_0",
        "onnx_op_type": "ConvInteger",
    }
    model_ir.operators = [
        OperatorIR(
            op_type="CAST",
            inputs=["x"],
            outputs=["x_f32_nchw"],
            **provenance,
        ),
        OperatorIR(
            op_type="SUB",
            inputs=["x_f32_nchw", "zero"],
            outputs=["centered_nchw"],
            **provenance,
        ),
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["centered_nchw", "perm"],
            outputs=["input_nhwc"],
            **provenance,
        ),
        OperatorIR(
            op_type="CONV_2D",
            inputs=["input_nhwc", "filter", "bias"],
            outputs=["y"],
            **provenance,
        ),
    ]
    return model_ir


def test_repairs_stale_convinteger_transpose_after_channel_last_promotion() -> None:
    model_ir = _make_convinteger_bridge_ir(mark_channel_last=True)

    stats = repair_channel_last_convinteger_input_transposes(model_ir)

    assert stats == {
        "propagated_channel_last_layout_hints": 2,
        "repaired_channel_last_convinteger_input_transposes": 1,
    }
    assert [op.op_type for op in model_ir.operators] == ["CAST", "SUB", "CONV_2D"]
    assert model_ir.operators[-1].inputs[0] == "centered_nchw"
    assert model_ir.tensors["x_f32_nchw"].shape == [1, 6, 6, 8]
    assert model_ir.tensors["centered_nchw"].shape == [1, 6, 6, 8]
    assert model_ir.tensors["centered_nchw"].physical_layout == "NHWC"
    assert "input_nhwc" not in model_ir.tensors
    assert "perm" not in model_ir.tensors


def test_keeps_convinteger_transpose_without_channel_last_provenance() -> None:
    model_ir = _make_convinteger_bridge_ir(mark_channel_last=False)

    stats = repair_channel_last_convinteger_input_transposes(model_ir)

    assert stats == {
        "propagated_channel_last_layout_hints": 0,
        "repaired_channel_last_convinteger_input_transposes": 0,
    }
    assert [op.op_type for op in model_ir.operators] == [
        "CAST",
        "SUB",
        "TRANSPOSE",
        "CONV_2D",
    ]


def test_propagates_channel_last_provenance_through_dynamic_quantize_decomposition() -> None:
    model_ir = _make_convinteger_bridge_ir(mark_channel_last=False)
    model_ir.metadata["assume_channel_last_layout_tensor_names"] = ["source_nhwc"]
    model_ir.inputs = ["source_nhwc"]
    model_ir.tensors["source_nhwc"] = _tensor(
        "source_nhwc",
        "FLOAT32",
        [1, 6, 6, 8],
    )
    model_ir.tensors["scale"] = _tensor("scale", "FLOAT32", [])
    model_ir.operators.insert(
        0,
        OperatorIR(
            op_type="DIV",
            inputs=["source_nhwc", "scale"],
            outputs=["x"],
            onnx_node_name="DynamicQuantizeLinear_0",
            onnx_op_type="DynamicQuantizeLinear",
        ),
    )

    stats = repair_channel_last_convinteger_input_transposes(model_ir)

    assert stats == {
        "propagated_channel_last_layout_hints": 3,
        "repaired_channel_last_convinteger_input_transposes": 1,
    }
