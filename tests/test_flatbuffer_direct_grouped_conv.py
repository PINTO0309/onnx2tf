from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def test_grouped_conv_materializes_unresolved_output_rank() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    weights = numpy_helper.from_array(
        np.ones((8, 2, 1, 1), dtype=np.float32),
        name="weights",
    )
    node = helper.make_node(
        "Conv",
        ["x", "weights"],
        ["y"],
        name="grouped_conv",
        group=2,
        kernel_shape=[1, 1],
    )
    model = helper.make_model(
        helper.make_graph([node], "grouped_conv", [x], [y], [weights]),
        opset_imports=[helper.make_operatorsetid("", 13)],
    )

    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="grouped_conv_unresolved_output_rank",
    )

    assert len(model_ir.tensors["y"].shape) == 4
    assert model_ir.tensors["y"].shape == [1, 5, 5, 8]
    assert all(not op.op_type.startswith("ONNX_") for op in model_ir.operators)


def test_average_pool_materializes_stale_output_rank() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])
    # Reproduce a stale inferred output placeholder seen in older ShuffleNet
    # exports. ONNX pooling semantics still determine a rank-4 result.
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    node = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        name="average_pool",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    )
    model = helper.make_model(
        helper.make_graph([node], "average_pool", [x], [y]),
        opset_imports=[helper.make_operatorsetid("", 9)],
    )

    model_ir = lower_onnx_to_ir(
        onnx_graph=model,
        output_file_name="average_pool_stale_output_rank",
    )

    assert model_ir.tensors["y"].shape == [1, 4, 4, 4]
    assert all(not op.op_type.startswith("ONNX_") for op in model_ir.operators)
