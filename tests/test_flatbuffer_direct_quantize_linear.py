from __future__ import annotations

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from tests.flatbuffer_direct_fingerprint import model_ir_fingerprint


def _make_quantize_dequantize_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    initializers = [
        numpy_helper.from_array(np.asarray(0.125, dtype=np.float32), "scale"),
        numpy_helper.from_array(np.asarray(-3, dtype=np.int8), "zero"),
    ]
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "scale", "zero"],
            ["q"],
            name="QuantizeNode",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["q", "scale", "zero"],
            ["y"],
            name="DequantizeNode",
        ),
    ]
    return helper.make_model(
        helper.make_graph(
            nodes,
            "QuantizeDequantizeGraph",
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )


def test_quantize_linear_family_preserves_pre_extraction_model_ir_fingerprint() -> None:
    assert model_ir_fingerprint(
        _make_quantize_dequantize_model(),
        "QuantizeDequantize",
    ) == "333343018c7bb32db3138cefdf4007353140b044472017ae6c3b4cce762e8f91"
