from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.core.onnx_analysis import _extract_tensor_info
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def test_softmax_axis_restores_missing_leading_rank_dimension() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6, 196, 196])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [6, 196, 196])
    node = helper.make_node("Softmax", ["x"], ["y"], axis=3)
    model = helper.make_model(
        helper.make_graph([node], "softmax_missing_leading_rank", [x], [y]),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    shape_map, _ = _extract_tensor_info(model)

    assert shape_map["x"] == [-1, 6, 196, 196]
    assert shape_map["y"] == [-1, 6, 196, 196]


def test_axis_rank_inference_keeps_already_complete_shape() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 6, 196, 196])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 196, 196])
    node = helper.make_node("Softmax", ["x"], ["y"], axis=3)
    model = helper.make_model(
        helper.make_graph([node], "softmax_complete_rank", [x], [y]),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    shape_map, _ = _extract_tensor_info(model)

    assert shape_map["x"] == [1, 6, 196, 196]
    assert shape_map["y"] == [1, 6, 196, 196]


def test_softmax_lowering_materializes_inferred_leading_dimension() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [6, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [6, 4, 4])
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Identity", ["x"], ["hidden"]),
                helper.make_node("Softmax", ["hidden"], ["y"], axis=3),
            ],
            "softmax_lowering_missing_leading_rank",
            [x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    model_ir = lower_onnx_to_ir(
        model,
        "softmax_lowering_missing_leading_rank",
        optimize_layout_transpose_chains=False,
    )

    assert model_ir.tensors["x"].shape == [1, 6, 4, 4]
    assert model_ir.tensors["y"].shape == [1, 6, 4, 4]
    assert [str(op.op_type) for op in model_ir.operators] == ["SOFTMAX"]


def test_flatten_axis_restores_missing_input_rank() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [384])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 384])
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Flatten", ["x"], ["y"], axis=2)],
            "flatten_missing_leading_rank",
            [x],
            [y],
        ),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    shape_map, _ = _extract_tensor_info(model)

    assert shape_map["x"] == [-1, -1, 384]
    assert shape_map["y"] == [1, 384]


def test_flatten_rank_uses_downstream_batch_norm_channel_contract() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, None)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, None)
    initializers = [
        numpy_helper.from_array(np.ones(384, dtype=np.float32), name="scale"),
        numpy_helper.from_array(np.zeros(384, dtype=np.float32), name="bias"),
        numpy_helper.from_array(np.zeros(384, dtype=np.float32), name="mean"),
        numpy_helper.from_array(np.ones(384, dtype=np.float32), name="variance"),
    ]
    model = helper.make_model(
        helper.make_graph(
            [
                helper.make_node("Flatten", ["x"], ["flat"], axis=2),
                helper.make_node(
                    "BatchNormalization",
                    ["flat", "scale", "bias", "mean", "variance"],
                    ["y"],
                ),
            ],
            "flatten_batch_norm_rank_contract",
            [x],
            [y],
            initializer=initializers,
        ),
        opset_imports=[helper.make_operatorsetid("", 17)],
    )

    shape_map, _ = _extract_tensor_info(model)

    assert shape_map["x"] == [-1, -1, -1]
    assert shape_map["flat"] == [-1, -1]


def test_split_and_qlinear_unary_restore_shortened_rank_metadata() -> None:
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [1, 4, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [8, 8])
    scale = numpy_helper.from_array(
        np.asarray(0.1, dtype=np.float32),
        name="scale",
    )
    zero = numpy_helper.from_array(
        np.asarray(0, dtype=np.uint8),
        name="zero",
    )
    graph = helper.make_graph(
        [
            helper.make_node("Split", ["x"], ["lhs", "rhs"], axis=1),
            helper.make_node(
                "QLinearLeakyRelu",
                ["rhs", "scale", "zero", "scale", "zero"],
                ["rhs_act"],
                domain="com.microsoft",
            ),
            helper.make_node(
                "QLinearConcat",
                [
                    "scale",
                    "zero",
                    "lhs",
                    "scale",
                    "zero",
                    "rhs_act",
                    "scale",
                    "zero",
                ],
                ["y"],
                axis=1,
                domain="com.microsoft",
            ),
        ],
        "shortened_quantized_rank_contract",
        [x],
        [y],
        initializer=[scale, zero],
        value_info=[
            helper.make_tensor_value_info("lhs", TensorProto.UINT8, [8, 8]),
            helper.make_tensor_value_info("rhs", TensorProto.UINT8, [8, 8]),
            helper.make_tensor_value_info("rhs_act", TensorProto.UINT8, [8, 8]),
        ],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 17),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )

    shape_map, _ = _extract_tensor_info(model)

    assert shape_map["lhs"] == [-1, -1, 8, 8]
    assert shape_map["rhs"] == [-1, -1, 8, 8]
    assert shape_map["rhs_act"] == [-1, -1, 8, 8]
    assert shape_map["y"] == [-1, -1, 8, 8]
