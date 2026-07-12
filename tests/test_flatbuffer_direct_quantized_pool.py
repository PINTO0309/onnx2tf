from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def test_qlinear_conv_eager_shape_keeps_explicit_stride2_maxpool_origin() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 4, 4])
    scale = numpy_helper.from_array(np.asarray(0.1, dtype=np.float32), "scale")
    zero = numpy_helper.from_array(np.asarray(0, dtype=np.int8), "zero")
    weight = numpy_helper.from_array(
        np.ones((1, 1, 3, 3), dtype=np.int8),
        "weight",
    )
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zero"], ["x_q"]),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "scale",
                "zero",
                "weight",
                "scale",
                "zero",
                "scale",
                "zero",
            ],
            ["conv_q"],
            name="QConv",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_q", "scale", "zero"],
            ["conv"],
        ),
        helper.make_node(
            "MaxPool",
            ["conv"],
            ["y"],
            name="Pool",
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            strides=[2, 2],
        ),
    ]
    model = helper.make_model(
        helper.make_graph(nodes, "quantized_pool", [x], [y], [scale, zero, weight]),
        opset_imports=[helper.make_operatorsetid("", 11)],
    )

    model_ir = lower_onnx_to_ir(model, "quantized_pool")

    pool = next(op for op in model_ir.operators if op.op_type == "MAX_POOL_2D")
    assert pool.options["padding"] == "VALID"
    pool_input = pool.inputs[0]
    pad = next(op for op in model_ir.operators if pool_input in op.outputs)
    assert pad.op_type == "PADV2"
