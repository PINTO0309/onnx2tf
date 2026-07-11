from __future__ import annotations

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from onnx2tf.tflite_builder.core import validate_model_ir_invariants
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir


def _make_rank3_layer_normalization_model():
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4])
    scale = numpy_helper.from_array(
        np.ones([4], dtype=np.float32),
        name="scale",
    )
    bias = numpy_helper.from_array(
        np.zeros([4], dtype=np.float32),
        name="bias",
    )
    layer_norm = helper.make_node(
        "LayerNormalization",
        ["x", "scale", "bias"],
        ["y"],
        name="LayerNorm",
        axis=-1,
    )
    graph = helper.make_graph(
        [layer_norm],
        "rank3_layer_norm",
        [x],
        [y],
        initializer=[scale, bias],
    )
    return helper.make_model(
        graph,
        opset_imports=[helper.make_operatorsetid("", 17)],
    )


def test_instance_norm_repair_does_not_rewrite_layer_norm_affine_axis() -> None:
    model_ir = lower_onnx_to_ir(
        _make_rank3_layer_normalization_model(),
        output_file_name="layer_norm_repair_provenance",
    )

    assert validate_model_ir_invariants(model_ir) == []
    for tensor_name in ["scale", "bias"]:
        tensor = model_ir.tensors[tensor_name]
        assert tensor.shape == [4]
        assert tensor.shape_signature == [4]
        assert np.asarray(tensor.data).shape == (4,)
