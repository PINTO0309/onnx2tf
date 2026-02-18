import glob
import json
import math
import os
import shutil
import tempfile
from contextlib import contextmanager

import numpy as np
import onnx
import onnx2tf
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnx2tf.tflite_builder.ir import ModelIR, OperatorIR, TensorIR
from onnx2tf.tflite_builder.lower_from_onnx2tf import (
    _optimize_center_size_offset_terminal_transpose_chains,
    _optimize_duplicate_transpose_fanout,
    _optimize_transpose_pre_add_nhwc_chains,
    _optimize_transpose_pre_concat_nhwc_chains,
    _optimize_transpose_mul_add_const_prepost_nhwc_chains,
    _reconcile_static_tensor_shapes,
    _resolve_dynamic_reshape_shapes,
    lower_onnx_to_ir,
)
from onnx2tf.tflite_builder.model_writer import serialize_model
from onnx2tf.tflite_builder.preprocess import (
    register_default_preprocess_rules,
    run_preprocess_pipeline,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module

Interpreter = pytest.importorskip("ai_edge_litert.interpreter").Interpreter
pytest.importorskip("onnxruntime")


def _save_model(tmpdir: str, name: str, model: onnx.ModelProto) -> str:
    model_path = os.path.join(tmpdir, f"{name}.onnx")
    onnx.save(model, model_path)
    return model_path


def _convert(
    model_path: str,
    output_dir: str,
    backend: str,
    output_dynamic_range_quantized_tflite: bool = False,
    output_integer_quantized_tflite: bool = False,
    quant_type: str = "per-channel",
    input_quant_dtype: str = "int8",
    output_quant_dtype: str = "int8",
    eval_with_onnx: bool = False,
    eval_num_samples: int = 10,
    eval_rtol: float = 0.0,
    eval_atol: float = 1e-4,
    eval_fail_on_threshold: bool = False,
    eval_target_tflite: str = "float32",
    eval_compare_mode: str = "auto",
    eval_split_models: bool = False,
    eval_split_reference: str = "unsplit_tflite",
    eval_split_fail_on_threshold: bool = False,
    auto_split_tflite_by_size: bool = False,
    report_op_coverage: bool = False,
    flatbuffer_direct_fallback_to_tf_converter: bool = False,
    flatbuffer_direct_allow_custom_ops: bool = False,
    flatbuffer_direct_custom_op_allowlist: list[str] | None = None,
    auto_split_max_size: str | int | None = None,
    tflite_split_max_bytes: int = 1073741824,
    tflite_split_target_bytes: int = 1060000000,
) -> str:
    onnx2tf.convert(
        input_onnx_file_path=model_path,
        output_folder_path=output_dir,
        disable_strict_mode=True,
        verbosity="error",
        tflite_backend=backend,
        output_dynamic_range_quantized_tflite=output_dynamic_range_quantized_tflite,
        output_integer_quantized_tflite=output_integer_quantized_tflite,
        quant_type=quant_type,
        input_quant_dtype=input_quant_dtype,
        output_quant_dtype=output_quant_dtype,
        eval_with_onnx=eval_with_onnx,
        eval_num_samples=eval_num_samples,
        eval_rtol=eval_rtol,
        eval_atol=eval_atol,
        eval_fail_on_threshold=eval_fail_on_threshold,
        eval_target_tflite=eval_target_tflite,
        eval_compare_mode=eval_compare_mode,
        eval_split_models=eval_split_models,
        eval_split_reference=eval_split_reference,
        eval_split_fail_on_threshold=eval_split_fail_on_threshold,
        auto_split_tflite_by_size=auto_split_tflite_by_size,
        report_op_coverage=report_op_coverage,
        flatbuffer_direct_fallback_to_tf_converter=flatbuffer_direct_fallback_to_tf_converter,
        flatbuffer_direct_allow_custom_ops=flatbuffer_direct_allow_custom_ops,
        flatbuffer_direct_custom_op_allowlist=flatbuffer_direct_custom_op_allowlist,
        auto_split_max_size=auto_split_max_size,
        tflite_split_max_bytes=tflite_split_max_bytes,
        tflite_split_target_bytes=tflite_split_target_bytes,
    )
    model_name = os.path.splitext(os.path.basename(model_path))[0]
    return os.path.join(output_dir, f"{model_name}_float32.tflite")


def _run_add_inference(tflite_path: str) -> np.ndarray:
    interpreter = Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    x = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    y = np.array([[4.0, 5.0, 6.0]], dtype=np.float32)
    by_name = {detail["name"]: detail for detail in input_details}
    if "x" in by_name and "y" in by_name:
        interpreter.set_tensor(by_name["x"]["index"], x)
        interpreter.set_tensor(by_name["y"]["index"], y)
    else:
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.set_tensor(input_details[1]["index"], y)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"])


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_erf_tile_scatternd_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    updates = helper.make_tensor_value_info("updates", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 6])

    repeats = numpy_helper.from_array(np.asarray([1, 2], dtype=np.int64), name="repeats")
    indices = numpy_helper.from_array(
        np.asarray([[0, 1], [1, 4]], dtype=np.int64),
        name="indices",
    )
    nodes = [
        helper.make_node("Erf", ["x"], ["x_erf"], name="ErfNode"),
        helper.make_node("Tile", ["x_erf", "repeats"], ["x_tiled"], name="TileNode"),
        helper.make_node(
            "ScatterND",
            ["x_tiled", "indices", "updates"],
            ["y"],
            name="ScatterNDNode",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "erf_tile_scatternd_graph",
        [x, updates],
        [y],
        initializer=[repeats, indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 16)])


def _make_add_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    c = numpy_helper.from_array(
        np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
        name="c",
    )
    node = helper.make_node("Add", ["x", "c"], ["y"], name="AddConstNode")
    graph = helper.make_graph([node], "add_const_graph", [x], [y], initializer=[c])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvNode",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv1d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 8])
    w = numpy_helper.from_array(np.ones((2, 1, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="Conv1DNode",
        pads=[1, 1],
        strides=[1],
        dilations=[1],
    )
    graph = helper.make_graph([node], "conv1d_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_convtranspose1d_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 8])
    w = numpy_helper.from_array(np.ones((2, 1, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "ConvTranspose",
        ["x", "W", "B"],
        ["y"],
        name="ConvTranspose1DNode",
        pads=[1, 1],
        strides=[1],
        dilations=[1],
    )
    graph = helper.make_graph([node], "convtranspose1d_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, 2, 2])
    w = numpy_helper.from_array(np.ones((1, 1, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((1,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvNodeDynamicBatch",
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "conv_dynamic_batch_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_conv_stride2_symmetric_pad_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4])
    w = numpy_helper.from_array(np.ones((4, 3, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((4,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="ConvStride2SymPadNode",
        pads=[1, 1, 1, 1],
        strides=[2, 2],
        dilations=[1, 1],
    )
    graph = helper.make_graph(
        [node],
        "conv_stride2_symmetric_pad_graph",
        [x],
        [y],
        initializer=[w, b],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_grouped_conv_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 6, 5, 5])
    w = numpy_helper.from_array(np.ones((6, 2, 3, 3), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((6,), dtype=np.float32), name="B")
    node = helper.make_node(
        "Conv",
        ["x", "W", "B"],
        ["y"],
        name="GroupedConvNode",
        group=2,
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )
    graph = helper.make_graph([node], "grouped_conv_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    node = helper.make_node(
        "AveragePool",
        ["x"],
        ["y"],
        name="PoolNode",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([node], "pool_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_maxpool_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, 2, 2])
    node = helper.make_node(
        "MaxPool",
        ["x"],
        ["y"],
        name="MaxPoolNodeDynamicBatch",
        kernel_shape=[2, 2],
        strides=[2, 2],
        pads=[0, 0, 0, 0],
    )
    graph = helper.make_graph([node], "maxpool_dynamic_batch_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gemm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    w = numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    node = helper.make_node("Gemm", ["x", "W", "B"], ["y"], name="GemmNode", transB=1)
    graph = helper.make_graph([node], "gemm_graph", [x], [y], initializer=[w, b])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_add_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node("Add", ["x", "y"], ["a0"], name="AddChain0")
    n1 = helper.make_node("Add", ["a0", "y"], ["a1"], name="AddChain1")
    n2 = helper.make_node("Add", ["a1", "y"], ["z"], name="AddChain2")
    graph = helper.make_graph([n0, n1, n2], "add_chain_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_add_relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node("Add", ["x", "y"], ["a0"], name="AddReluAdd")
    n1 = helper.make_node("Relu", ["a0"], ["z"], name="AddReluRelu")
    graph = helper.make_graph([n0, n1], "add_relu_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_binary_activation_model(binary_op: str, activation_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    n0 = helper.make_node(binary_op, ["x", "y"], ["a0"], name=f"{binary_op}ActivationBinary")
    if activation_op == "Relu6":
        n1 = helper.make_node(
            "Clip",
            ["a0"],
            ["z"],
            name=f"{binary_op}ActivationRelu6",
            min=0.0,
            max=6.0,
        )
    else:
        n1 = helper.make_node(activation_op, ["a0"], ["z"], name=f"{binary_op}Activation{activation_op}")
    graph = helper.make_graph(
        [n0, n1],
        f"{binary_op.lower()}_{activation_op.lower()}_graph",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_forward_lstm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [3, 1, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 1, 1, 2])
    W = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.20, -0.10],
                    [0.05, 0.15],
                    [0.30, -0.25],
                    [0.10, 0.12],
                    [-0.18, 0.08],
                    [0.11, -0.09],
                    [0.04, 0.07],
                    [-0.06, 0.03],
                ]
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    R = numpy_helper.from_array(
        np.asarray(
            [
                [
                    [0.05, -0.02],
                    [0.03, 0.01],
                    [0.02, -0.04],
                    [0.01, 0.03],
                    [-0.02, 0.02],
                    [0.04, -0.01],
                    [0.03, 0.02],
                    [-0.01, 0.01],
                ]
            ],
            dtype=np.float32,
        ),
        name="R",
    )
    B = numpy_helper.from_array(
        np.asarray(
            [[0.01, -0.02, 0.03, 0.01, -0.01, 0.00, 0.02, -0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
            dtype=np.float32,
        ),
        name="B",
    )
    initial_h = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), name="initial_h")
    initial_c = numpy_helper.from_array(np.zeros((1, 1, 2), dtype=np.float32), name="initial_c")
    lstm = helper.make_node(
        "LSTM",
        ["x", "W", "R", "B", "", "initial_h", "initial_c"],
        ["y"],
        name="ForwardLSTMNode",
        hidden_size=2,
        direction="forward",
    )
    graph = helper.make_graph(
        [lstm],
        "forward_lstm_graph",
        [x],
        [y],
        initializer=[W, R, B, initial_h, initial_c],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_wrapped_add_relu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TWAR_X", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TWAR_Y", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_nchw", "y_nchw"], ["sum_nchw"], name="TWAR_Add"),
        helper.make_node("Relu", ["sum_nchw"], ["sum_relu_nchw"], name="TWAR_Relu0"),
        helper.make_node(
            "Transpose",
            ["sum_relu_nchw"],
            ["sum_nhwc"],
            name="TWAR_Out",
            perm=[0, 2, 3, 1],
        ),
        helper.make_node("Relu", ["sum_nhwc"], ["z"], name="TWAR_Relu1"),
    ]
    graph = helper.make_graph(nodes, "transpose_wrapped_add_relu_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_mul_const_add_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    scale = numpy_helper.from_array(
        np.asarray([[[[1.0]], [[0.5]], [[1.5]], [[2.0]]]], dtype=np.float32),
        name="tmcat_scale_nchw",
    )
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TMCAT_X", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TMCAT_Y", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["y_nchw", "tmcat_scale_nchw"], ["y_scaled_nchw"], name="TMCAT_Mul"),
        helper.make_node("Add", ["x_nchw", "y_scaled_nchw"], ["sum_nchw"], name="TMCAT_Add"),
        helper.make_node("Transpose", ["sum_nchw"], ["sum_nhwc"], name="TMCAT_Out", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum_nhwc"], ["z"], name="TMCAT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_mul_const_add_transpose_graph",
        [x, y],
        [z],
        initializer=[scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_non_max_suppression_model() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 1, 5])
    selected_indices = helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [20, 3])

    max_output = numpy_helper.from_array(np.asarray([20], dtype=np.int64), name="max_output")
    iou_threshold = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="iou_threshold")
    score_threshold = numpy_helper.from_array(np.asarray([0.4], dtype=np.float32), name="score_threshold")

    node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        name="NMSNode",
        center_point_box=0,
    )
    graph = helper.make_graph(
        [node],
        "nms_graph",
        [boxes, scores],
        [selected_indices],
        initializer=[max_output, iou_threshold, score_threshold],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_non_max_suppression_multiclass_model() -> onnx.ModelProto:
    boxes = helper.make_tensor_value_info("boxes", TensorProto.FLOAT, [1, 5, 4])
    scores = helper.make_tensor_value_info("scores", TensorProto.FLOAT, [1, 3, 5])
    selected_indices = helper.make_tensor_value_info("selected_indices", TensorProto.INT64, [20, 3])

    max_output = numpy_helper.from_array(np.asarray([20], dtype=np.int64), name="max_output")
    iou_threshold = numpy_helper.from_array(np.asarray([0.5], dtype=np.float32), name="iou_threshold")
    score_threshold = numpy_helper.from_array(np.asarray([0.4], dtype=np.float32), name="score_threshold")

    node = helper.make_node(
        "NonMaxSuppression",
        ["boxes", "scores", "max_output", "iou_threshold", "score_threshold"],
        ["selected_indices"],
        name="NMSNodeMultiClass",
        center_point_box=0,
    )
    graph = helper.make_graph(
        [node],
        "nms_multiclass_graph",
        [boxes, scores],
        [selected_indices],
        initializer=[max_output, iou_threshold, score_threshold],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unary_model(op_type: str, *, name: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node(op_type, ["x"], ["y"], name=f"{name}Node")
    graph = helper.make_graph([node], f"{name}_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_clip_relu6_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Clip", ["x"], ["y"], name="ClipNode", min=0.0, max=6.0)
    graph = helper.make_graph([node], "clip_relu6_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_elu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Elu", ["x"], ["y"], name="EluNode", alpha=1.0)
    graph = helper.make_graph([node], "elu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_hardswish_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("HardSwish", ["x"], ["y"], name="HardSwishNode")
    graph = helper.make_graph([node], "hardswish_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 14)])


def _make_leakyrelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("LeakyRelu", ["x"], ["y"], name="LeakyReluNode", alpha=0.2)
    graph = helper.make_graph([node], "leakyrelu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_prelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    slope = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="slope")
    node = helper.make_node("PRelu", ["x", "slope"], ["y"], name="PReluNode")
    graph = helper.make_graph([node], "prelu_graph", [x], [y], initializer=[slope])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gelu_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Gelu", ["x"], ["y"], name="GeluNode")
    graph = helper.make_graph([node], "gelu_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 20)])


def _make_pow_square_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    exponent = numpy_helper.from_array(np.array([2.0], dtype=np.float32), name="pow_exp")
    node = helper.make_node("Pow", ["x", "pow_exp"], ["y"], name="PowNode")
    graph = helper.make_graph([node], "pow_graph", [x], [y], initializer=[exponent])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_pow_general_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    exponent = numpy_helper.from_array(np.array([1.25], dtype=np.float32), name="pow_general_exp")
    node = helper.make_node("Pow", ["x", "pow_general_exp"], ["y"], name="PowGeneralNode")
    graph = helper.make_graph([node], "pow_general_graph", [x], [y], initializer=[exponent])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reciprocal_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Reciprocal", ["x"], ["y"], name="ReciprocalNode")
    graph = helper.make_graph([node], "reciprocal_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_onehot_model() -> onnx.ModelProto:
    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [2, 3])
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3, 4])
    depth = numpy_helper.from_array(np.array([4], dtype=np.int64), name="onehot_depth")
    values = numpy_helper.from_array(np.array([0.0, 1.0], dtype=np.float32), name="onehot_values")
    node = helper.make_node(
        "OneHot",
        ["indices", "onehot_depth", "onehot_values"],
        ["y"],
        name="OneHotNode",
        axis=-1,
    )
    graph = helper.make_graph(
        [node],
        "onehot_graph",
        [indices],
        [output],
        initializer=[depth, values],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_matmul_integer_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.UINT8, [2, 3])
    a_zero = helper.make_tensor_value_info("a_zero", TensorProto.UINT8, [])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [2, 4])
    w = numpy_helper.from_array(
        np.array(
            [
                [1, -2, 3, -4],
                [5, -6, 7, -8],
                [9, -10, 11, -12],
            ],
            dtype=np.int8,
        ),
        name="mmi_w",
    )
    b_zero = numpy_helper.from_array(np.zeros((4,), dtype=np.int8), name="mmi_b_zero")
    node = helper.make_node(
        "MatMulInteger",
        ["x", "mmi_w", "a_zero", "mmi_b_zero"],
        ["y"],
        name="MatMulIntegerNode",
    )
    graph = helper.make_graph(
        [node],
        "matmul_integer_graph",
        [x, a_zero],
        [y],
        initializer=[w, b_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_dynamic_quantize_linear_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.UINT8, [2, 3])
    y_scale = helper.make_tensor_value_info("y_scale", TensorProto.FLOAT, [])
    y_zero = helper.make_tensor_value_info("y_zero", TensorProto.UINT8, [])
    node = helper.make_node(
        "DynamicQuantizeLinear",
        ["x"],
        ["y", "y_scale", "y_zero"],
        name="DynamicQuantizeLinearNode",
    )
    graph = helper.make_graph(
        [node],
        "dynamic_quantize_linear_graph",
        [x],
        [y, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_shape_slice_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3, 4, 5])
    y = helper.make_tensor_value_info("y", TensorProto.INT64, [2])
    node = helper.make_node(
        "Shape",
        ["x"],
        ["y"],
        name="ShapeSliceNode",
        start=1,
        end=3,
    )
    graph = helper.make_graph([node], "shape_slice_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 15)])


def _make_legacy_slice_attr_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 4])
    nodes = [
        helper.make_node(
            "Slice",
            ["x"],
            ["y0"],
            name="LegacySlice0",
            starts=[0],
            ends=[2],
            axes=[2],
        ),
        helper.make_node(
            "Slice",
            ["x"],
            ["y1"],
            name="LegacySlice1",
            starts=[2],
            ends=[9223372036854775807],
            axes=[2],
        ),
        helper.make_node("Concat", ["y0", "y1"], ["y"], name="LegacySliceConcat", axis=2),
    ]
    graph = helper.make_graph(nodes, "legacy_slice_attr_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 9)])


def _make_constant_of_shape_model() -> onnx.ModelProto:
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 3])
    shape = numpy_helper.from_array(np.array([2, 3], dtype=np.int64), name="cos_shape")
    value = numpy_helper.from_array(np.array([0.25], dtype=np.float32), name="cos_value")
    node = helper.make_node(
        "ConstantOfShape",
        ["cos_shape"],
        ["y"],
        name="ConstantOfShapeNode",
        value=value,
    )
    graph = helper.make_graph(
        [node],
        "constant_of_shape_graph",
        [],
        [y],
        initializer=[shape],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_fused_matmul_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 4])
    w = numpy_helper.from_array(
        np.array(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            dtype=np.float32,
        ),
        name="fmm_w",
    )
    node = helper.make_node(
        "FusedMatMul",
        ["x", "fmm_w"],
        ["y"],
        name="FusedMatMulNode",
        alpha=0.125,
        transA=0,
        transB=1,
    )
    graph = helper.make_graph([node], "fused_matmul_graph", [x], [y], initializer=[w])
    return helper.make_model(
        graph,
        opset_imports=[
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.microsoft", 1),
        ],
    )


def _make_space_to_depth_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2, 2])
    node = helper.make_node(
        "SpaceToDepth",
        ["x"],
        ["y"],
        name="SpaceToDepthNode",
        blocksize=2,
        mode="DCR",
    )
    graph = helper.make_graph([node], "space_to_depth_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_resize_nearest_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="roi_empty")
    scales = numpy_helper.from_array(np.asarray([1.0, 1.0, 2.0, 2.0], dtype=np.float32), name="resize_scales")
    node = helper.make_node(
        "Resize",
        ["x", "roi_empty", "resize_scales"],
        ["y"],
        name="ResizeNearestNode",
        mode="nearest",
        coordinate_transformation_mode="asymmetric",
        nearest_mode="floor",
    )
    graph = helper.make_graph([node], "resize_nearest_graph", [x], [y], initializer=[roi, scales])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_resize_dynamic_sizes_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 3, "H", "W"])
    ref = helper.make_tensor_value_info("ref", TensorProto.FLOAT, ["N", 3, "RH", "RW"])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 3, "RH", "RW"])
    roi = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="resize_dyn_roi_empty")
    scales = numpy_helper.from_array(np.asarray([], dtype=np.float32), name="resize_dyn_scales_empty")
    nodes = [
        helper.make_node("Shape", ["ref"], ["resize_dyn_sizes"], name="ResizeDynShapeRef"),
        helper.make_node(
            "Resize",
            ["x", "resize_dyn_roi_empty", "resize_dyn_scales_empty", "resize_dyn_sizes"],
            ["y"],
            name="ResizeDynSizesNode",
            mode="linear",
            coordinate_transformation_mode="pytorch_half_pixel",
            nearest_mode="floor",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "resize_dynamic_sizes_graph",
        [x, ref],
        [y],
        initializer=[roi, scales],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_sigmoid_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qsig_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qsig_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([1.0 / 256.0], dtype=np.float32), name="qsig_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([-128], dtype=np.int8), name="qsig_y_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qsig_x_scale", "qsig_x_zero"], ["x_q"], name="QSigQ0"),
        helper.make_node(
            "QLinearSigmoid",
            ["x_q", "qsig_x_scale", "qsig_x_zero", "qsig_y_scale", "qsig_y_zero"],
            ["y_q"],
            name="QSigNode",
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qsig_y_scale", "qsig_y_zero"], ["y"], name="QSigDQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_sigmoid_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_global_average_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1, 1])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qgap_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="qgap_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_y_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qgap_x_scale", "qgap_x_zero"], ["x_q"], name="QGapQ0"),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q", "qgap_x_scale", "qgap_x_zero", "qgap_y_scale", "qgap_y_zero"],
            ["y_q"],
            name="QGapNode",
            channels_last=0,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qgap_y_scale", "qgap_y_zero"], ["y"], name="QGapDQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_global_average_pool_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_global_average_pool_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 2, 1, 1])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qgap_dyn_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_dyn_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="qgap_dyn_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_dyn_y_zero")
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qgap_dyn_x_scale", "qgap_dyn_x_zero"],
            ["x_q"],
            name="QGapDynQ0",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q", "qgap_dyn_x_scale", "qgap_dyn_x_zero", "qgap_dyn_y_scale", "qgap_dyn_y_zero"],
            ["y_q"],
            name="QGapDynNode",
            channels_last=0,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "qgap_dyn_y_scale", "qgap_dyn_y_zero"],
            ["y"],
            name="QGapDynDQ0",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_global_average_pool_dynamic_batch_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_qlinear_global_average_pool_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 2])
    y = helper.make_tensor_value_info("y_q", TensorProto.INT8, [1, 2, 1, 1])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qgap_t_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_t_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.125], dtype=np.float32), name="qgap_t_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qgap_t_y_zero")
    t_perm = numpy_helper.from_array(np.asarray([0, 3, 1, 2], dtype=np.int64), name="qgap_t_perm")
    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qgap_t_x_scale", "qgap_t_x_zero"],
            ["x_q"],
            name="QGapTQ0",
        ),
        helper.make_node(
            "Transpose",
            ["x_q", "qgap_t_perm"],
            ["x_q_nchw"],
            name="QGapTTranspose0",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q_nchw", "qgap_t_x_scale", "qgap_t_x_zero", "qgap_t_y_scale", "qgap_t_y_zero"],
            ["y_q"],
            name="QGapTNode",
            channels_last=0,
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_qlinear_global_average_pool_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero, t_perm],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_concat_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    x_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qcat_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qcat_x_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qcat_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qcat_y_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qcat_x_scale", "qcat_x_zero"], ["x_q"], name="QCatQ0"),
        helper.make_node(
            "QLinearConcat",
            [
                "qcat_y_scale",
                "qcat_y_zero",
                "x_q",
                "qcat_x_scale",
                "qcat_x_zero",
                "x_q",
                "qcat_x_scale",
                "qcat_x_zero",
            ],
            ["y_q"],
            name="QCatNode",
            axis=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qcat_y_scale", "qcat_y_zero"], ["y"], name="QCatDQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_concat_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_concat_conv_layout_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 1, 1])

    in_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="qlcc_in_scale")
    in_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_in_zero")
    gap_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="qlcc_gap_scale")
    gap_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_gap_zero")
    cat_scale = numpy_helper.from_array(np.asarray([0.15], dtype=np.float32), name="qlcc_cat_scale")
    cat_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_cat_zero")
    conv_w = numpy_helper.from_array(
        np.asarray(
            [
                [[[1]], [[1]], [[1]], [[1]]],
                [[[1]], [[0]], [[-1]], [[1]]],
                [[[0]], [[1]], [[1]], [[0]]],
            ],
            dtype=np.int8,
        ),
        name="qlcc_conv_w",
    )
    conv_w_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="qlcc_conv_w_scale")
    conv_w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_conv_w_zero")
    conv_y_scale = numpy_helper.from_array(np.asarray([0.12], dtype=np.float32), name="qlcc_conv_y_scale")
    conv_y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qlcc_conv_y_zero")

    nodes = [
        helper.make_node(
            "QuantizeLinear",
            ["x", "qlcc_in_scale", "qlcc_in_zero"],
            ["x_q"],
            name="QLCatConv_Q0",
        ),
        helper.make_node(
            "QLinearGlobalAveragePool",
            ["x_q", "qlcc_in_scale", "qlcc_in_zero", "qlcc_gap_scale", "qlcc_gap_zero"],
            ["gap_q"],
            name="QLCatConv_GAP",
            channels_last=0,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["x_q", "qlcc_in_scale", "qlcc_in_zero"],
            ["x_f"],
            name="QLCatConv_DQ0",
        ),
        helper.make_node(
            "MaxPool",
            ["x_f"],
            ["pool_f"],
            name="QLCatConv_MaxPool",
            kernel_shape=[4, 4],
            strides=[4, 4],
            pads=[0, 0, 0, 0],
        ),
        helper.make_node(
            "QuantizeLinear",
            ["pool_f", "qlcc_in_scale", "qlcc_in_zero"],
            ["pool_q"],
            name="QLCatConv_QPool",
        ),
        helper.make_node(
            "QLinearConcat",
            [
                "qlcc_cat_scale",
                "qlcc_cat_zero",
                "gap_q",
                "qlcc_gap_scale",
                "qlcc_gap_zero",
                "pool_q",
                "qlcc_in_scale",
                "qlcc_in_zero",
            ],
            ["cat_q"],
            name="QLCatConv_Concat",
            axis=1,
        ),
        helper.make_node(
            "QLinearConv",
            [
                "cat_q",
                "qlcc_cat_scale",
                "qlcc_cat_zero",
                "qlcc_conv_w",
                "qlcc_conv_w_scale",
                "qlcc_conv_w_zero",
                "qlcc_conv_y_scale",
                "qlcc_conv_y_zero",
            ],
            ["y_q"],
            name="QLCatConv_Conv",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "qlcc_conv_y_scale", "qlcc_conv_y_zero"],
            ["y"],
            name="QLCatConv_DQ1",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_concat_conv_layout_chain_graph",
        [x],
        [y],
        initializer=[
            in_scale,
            in_zero,
            gap_scale,
            gap_zero,
            cat_scale,
            cat_zero,
            conv_w,
            conv_w_scale,
            conv_w_zero,
            conv_y_scale,
            conv_y_zero,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="x_zero")
    mul_const = numpy_helper.from_array(np.asarray([1], dtype=np.int8), name="mul_const")
    mul_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mul_scale")
    mul_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mul_zero")
    mul_out_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mul_out_scale")
    conv_w = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="conv_w")
    conv_w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="conv_w_scale")
    conv_w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="conv_w_zero")
    conv_out_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="conv_out_scale")
    conv_out_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="conv_out_zero")
    bn_scale = numpy_helper.from_array(np.asarray([1.1], dtype=np.float32), name="bn_scale")
    bn_bias = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="bn_bias")
    bn_mean = numpy_helper.from_array(np.asarray([0.0], dtype=np.float32), name="bn_mean")
    bn_var = numpy_helper.from_array(np.asarray([1.0], dtype=np.float32), name="bn_var")
    prelu_slope = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="prelu_slope")
    q2_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="q2_scale")
    q2_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="q2_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale", "x_zero"], ["x_q"], name="Q0"),
        helper.make_node(
            "QLinearMul",
            ["x_q", "x_scale", "x_zero", "mul_const", "mul_scale", "mul_zero", "mul_out_scale", "mul_zero"],
            ["x_mul_q"],
            name="QMul0",
        ),
        helper.make_node(
            "QLinearConv",
            [
                "x_mul_q",
                "mul_out_scale",
                "mul_zero",
                "conv_w",
                "conv_w_scale",
                "conv_w_zero",
                "conv_out_scale",
                "conv_out_zero",
            ],
            ["conv_q"],
            name="QConv0",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "DequantizeLinear",
            ["conv_q", "conv_out_scale", "conv_out_zero"],
            ["conv_f"],
            name="DQ0",
        ),
        helper.make_node(
            "BatchNormalization",
            ["conv_f", "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            ["bn_out"],
            name="BN0",
            epsilon=1e-5,
        ),
        helper.make_node("PRelu", ["bn_out", "prelu_slope"], ["prelu_out"], name="PRelu0"),
        helper.make_node("QuantizeLinear", ["prelu_out", "q2_scale", "q2_zero"], ["q2"], name="Q1"),
        helper.make_node("DequantizeLinear", ["q2", "q2_scale", "q2_zero"], ["y"], name="DQ1"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_chain_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zero,
            mul_const,
            mul_scale,
            mul_zero,
            mul_out_scale,
            conv_w,
            conv_w_scale,
            conv_w_zero,
            conv_out_scale,
            conv_out_zero,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            prelu_slope,
            q2_scale,
            q2_zero,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_pair_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="pair_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_x_zero")
    w1 = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="pair_w1")
    w1_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="pair_w1_scale")
    w1_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_w1_zero")
    y1_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="pair_y1_scale")
    y1_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_y1_zero")
    w2 = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="pair_w2")
    w2_scale = numpy_helper.from_array(np.asarray([0.3], dtype=np.float32), name="pair_w2_scale")
    w2_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_w2_zero")
    y2_scale = numpy_helper.from_array(np.asarray([0.04], dtype=np.float32), name="pair_y2_scale")
    y2_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="pair_y2_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "pair_x_scale", "pair_x_zero"], ["x_q"], name="PairQ0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "pair_x_scale",
                "pair_x_zero",
                "pair_w1",
                "pair_w1_scale",
                "pair_w1_zero",
                "pair_y1_scale",
                "pair_y1_zero",
            ],
            ["q1"],
            name="PairQConv1",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node(
            "QLinearConv",
            [
                "q1",
                "pair_y1_scale",
                "pair_y1_zero",
                "pair_w2",
                "pair_w2_scale",
                "pair_w2_zero",
                "pair_y2_scale",
                "pair_y2_zero",
            ],
            ["q2"],
            name="PairQConv2",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["q2", "pair_y2_scale", "pair_y2_zero"], ["y"], name="PairDQ"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_pair_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zero,
            w1,
            w1_scale,
            w1_zero,
            y1_scale,
            y1_zero,
            w2,
            w2_scale,
            w2_zero,
            y2_scale,
            y2_zero,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_multichannel_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mc_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mc_x_zero")
    # ONNX QLinearConv weights: OIHW
    w = numpy_helper.from_array(
        np.asarray(
            [
                [[[1]], [[2]], [[3]]],
                [[[1]], [[0]], [[-1]]],
                [[[2]], [[1]], [[0]]],
                [[[0]], [[1]], [[2]]],
            ],
            dtype=np.int8,
        ),
        name="mc_w",
    )
    w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="mc_w_scale")
    w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mc_w_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="mc_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mc_y_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "mc_x_scale", "mc_x_zero"], ["x_q"], name="QMC0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "mc_x_scale",
                "mc_x_zero",
                "mc_w",
                "mc_w_scale",
                "mc_w_zero",
                "mc_y_scale",
                "mc_y_zero",
            ],
            ["y_q"],
            name="QConvMC",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "mc_y_scale", "mc_y_zero"], ["y"], name="DQMC0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_multichannel_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, ["N", 1, 2, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="dyn_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="dyn_x_zero")
    w = numpy_helper.from_array(np.asarray([[[[1]]]], dtype=np.int8), name="dyn_w")
    w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="dyn_w_scale")
    w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="dyn_w_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="dyn_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="dyn_y_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "dyn_x_scale", "dyn_x_zero"], ["x_q"], name="QDyn0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "dyn_x_scale",
                "dyn_x_zero",
                "dyn_w",
                "dyn_w_scale",
                "dyn_w_zero",
                "dyn_y_scale",
                "dyn_y_zero",
            ],
            ["y_q"],
            name="QConvDyn",
            kernel_shape=[1, 1],
            pads=[0, 0, 0, 0],
            strides=[1, 1],
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "dyn_y_scale", "dyn_y_zero"], ["y"], name="DQDyn0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_dynamic_batch_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_conv_unknown_rank_model() -> onnx.ModelProto:
    model = _make_qlinear_conv_multichannel_model()
    for value_info in list(model.graph.input) + list(model.graph.value_info) + list(model.graph.output):
        if (
            value_info.type.HasField("tensor_type")
            and value_info.type.tensor_type.HasField("shape")
        ):
            value_info.type.tensor_type.ClearField("shape")
    return model


def _make_qlinear_conv_valid_explicit_pad_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 5, 5])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="qv_x_scale")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qv_x_zero")
    w = numpy_helper.from_array(np.ones((1, 1, 2, 2), dtype=np.int8), name="qv_w")
    w_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="qv_w_scale")
    w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qv_w_zero")
    y_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="qv_y_scale")
    y_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="qv_y_zero")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "qv_x_scale", "qv_x_zero"], ["x_q"], name="QV_Q0"),
        helper.make_node(
            "QLinearConv",
            [
                "x_q",
                "qv_x_scale",
                "qv_x_zero",
                "qv_w",
                "qv_w_scale",
                "qv_w_zero",
                "qv_y_scale",
                "qv_y_zero",
            ],
            ["y_q"],
            name="QV_QConv",
            kernel_shape=[2, 2],
            pads=[1, 1, 1, 1],
            strides=[1, 1],
            auto_pad="VALID",
            group=1,
        ),
        helper.make_node("DequantizeLinear", ["y_q", "qv_y_scale", "qv_y_zero"], ["y"], name="QV_DQ0"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_conv_valid_explicit_pad_graph",
        [x],
        [y],
        initializer=[x_scale, x_zero, w, w_scale, w_zero, y_scale, y_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_terminal_quantize_dequantize_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 2, 2])
    q_scale = numpy_helper.from_array(np.asarray([0.05], dtype=np.float32), name="tail_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tail_q_zero")
    nodes = [
        helper.make_node("Relu", ["x"], ["relu_out"], name="TailRelu"),
        helper.make_node(
            "QuantizeLinear",
            ["relu_out", "tail_q_scale", "tail_q_zero"],
            ["y_q"],
            name="TailQ",
        ),
        helper.make_node(
            "DequantizeLinear",
            ["y_q", "tail_q_scale", "tail_q_zero"],
            ["y"],
            name="TailDQ",
        ),
    ]
    graph = helper.make_graph(
        nodes,
        "terminal_quantize_dequantize_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_terminal_transpose_dequantize_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 2])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="ttd_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="ttd_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TTD_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "ttd_q_scale", "ttd_q_zero"], ["y"], name="TTD_DQ"),
    ]
    graph = helper.make_graph(
        nodes,
        "terminal_transpose_dequantize_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqt_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqt_q_scale", "tqt_q_zero"], ["x_q"], name="TQT_Q"),
        helper.make_node("Transpose", ["x_q"], ["y"], name="TQT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_transpose_dynamic_batch_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["N", 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, ["N", 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqtd_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqtd_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQTD_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqtd_q_scale", "tqtd_q_zero"], ["x_q"], name="TQTD_Q"),
        helper.make_node("Transpose", ["x_q"], ["y"], name="TQTD_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_transpose_dynamic_batch_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_transpose_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqtf_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqtf_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQTF_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqtf_q_scale", "tqtf_q_zero"], ["x_q"], name="TQTF_Q"),
        helper.make_node("Transpose", ["x_q"], ["x_q0"], name="TQTF_PostT0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x_q"], ["x_q1"], name="TQTF_PostT1", perm=[0, 3, 1, 2]),
        helper.make_node("DequantizeLinear", ["x_q0", "tqtf_q_scale", "tqtf_q_zero"], ["x_f0"], name="TQTF_DQ0"),
        helper.make_node("DequantizeLinear", ["x_q1", "tqtf_q_scale", "tqtf_q_zero"], ["x_f1"], name="TQTF_DQ1"),
        helper.make_node("Add", ["x_f0", "x_f1"], ["y"], name="TQTF_Add"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_transpose_fanout_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_dequantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tdt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tdt_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TDT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "tdt_q_scale", "tdt_q_zero"], ["x_f"], name="TDT_DQ"),
        helper.make_node("Transpose", ["x_f"], ["y"], name="TDT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_dequantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_fanout_chain_model() -> onnx.ModelProto:
    a = helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 2, 3, 4])
    b = helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 2, 3, 4])
    c = helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["a"], ["a_t"], name="TBFC_PreA", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["b"], ["b_t"], name="TBFC_PreB", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["a_t", "b_t"], ["ab_t"], name="TBFC_Add0"),
        helper.make_node("Transpose", ["ab_t"], ["ab"], name="TBFC_Post0", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["c"], ["c_t"], name="TBFC_PreC", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["ab_t", "c_t"], ["abc_t"], name="TBFC_Add1"),
        helper.make_node("Transpose", ["abc_t"], ["abc"], name="TBFC_Post1", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["abc"], ["y"], name="TBFC_ReluY"),
        helper.make_node("Relu", ["ab"], ["z"], name="TBFC_ReluZ"),
    ]
    graph = helper.make_graph(nodes, "transpose_binary_fanout_chain_graph", [a, b, c], [y, z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_dequantize_transpose_with_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT8, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 3, 4])
    y2 = helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 3, 4, 2])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tdtf_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tdtf_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TDTF_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "tdtf_q_scale", "tdtf_q_zero"], ["x_f1"], name="TDTF_DQ1"),
        helper.make_node("Transpose", ["x_f1"], ["y1"], name="TDTF_PostT", perm=[0, 3, 1, 2]),
        helper.make_node("DequantizeLinear", ["x_t", "tdtf_q_scale", "tdtf_q_zero"], ["y2"], name="TDTF_DQ2"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_dequantize_transpose_with_fanout_graph",
        [x],
        [y1, y2],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_quantize_dequantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tqdt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tqdt_q_zero")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TQDT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("QuantizeLinear", ["x_t", "tqdt_q_scale", "tqdt_q_zero"], ["x_q"], name="TQDT_Q"),
        helper.make_node("DequantizeLinear", ["x_q", "tqdt_q_scale", "tqdt_q_zero"], ["x_f"], name="TQDT_DQ"),
        helper.make_node("Transpose", ["x_f"], ["y"], name="TQDT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_quantize_dequantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_dequantize_relu6_quantize_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT8, [1, 2, 3, 4])
    q_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="tdrqt_q_scale")
    q_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="tdrqt_q_zero")
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "tdrqt_q_scale", "tdrqt_q_zero"], ["x_q_src"], name="TDRQT_Q0"),
        helper.make_node("Transpose", ["x_q_src"], ["x_t"], name="TDRQT_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("DequantizeLinear", ["x_t", "tdrqt_q_scale", "tdrqt_q_zero"], ["x_f"], name="TDRQT_DQ"),
        helper.make_node("Clip", ["x_f"], ["x_r6"], name="TDRQT_Relu6", min=0.0, max=6.0),
        helper.make_node("QuantizeLinear", ["x_r6", "tdrqt_q_scale", "tdrqt_q_zero"], ["x_q"], name="TDRQT_Q1"),
        helper.make_node("Transpose", ["x_q"], ["y"], name="TDRQT_PostT", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_dequantize_relu6_quantize_transpose_graph",
        [x],
        [y],
        initializer=[q_scale, q_zero],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_transpose_model(binary_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name=f"TBT_PreT0_{binary_op}", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["y"], ["y_t"], name=f"TBT_PreT1_{binary_op}", perm=[0, 2, 3, 1]),
        helper.make_node(binary_op, ["x_t", "y_t"], ["z_t"], name=f"TBT_{binary_op}"),
        helper.make_node("Transpose", ["z_t"], ["z"], name=f"TBT_PostT_{binary_op}", perm=[0, 3, 1, 2]),
    ]
    graph = helper.make_graph(
        nodes,
        f"transpose_{binary_op.lower()}_transpose_graph",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_transpose_softmax_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2, 5])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t0"], name="TTS_Pre0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x_t0"], ["x_t1"], name="TTS_Pre1", perm=[0, 3, 2, 1]),
        helper.make_node("Softmax", ["x_t1"], ["y"], name="TTS_Softmax", axis=-1),
    ]
    graph = helper.make_graph(nodes, "transpose_transpose_softmax_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_terminal_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 3, 2])
    nodes = [
        helper.make_node("Softmax", ["x"], ["s"], name="STT_Softmax", axis=-1),
        helper.make_node("Transpose", ["s"], ["y"], name="STT_OutTranspose", perm=[0, 3, 2, 1]),
    ]
    graph = helper.make_graph(nodes, "softmax_terminal_transpose_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_terminal_double_pretranspose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t0"], name="STDPT_Pre0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x_t0"], ["x_t1"], name="STDPT_Pre1", perm=[0, 3, 2, 1]),
        helper.make_node("Softmax", ["x_t1"], ["s"], name="STDPT_Softmax", axis=-1),
        helper.make_node("Transpose", ["s"], ["y"], name="STDPT_OutTranspose", perm=[0, 3, 2, 1]),
    ]
    graph = helper.make_graph(nodes, "softmax_terminal_double_pretranspose_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_swish_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TSWT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["x_t"], ["s_t"], name="TSWT_Sigmoid"),
        helper.make_node("Mul", ["x_t", "s_t"], ["y_t"], name="TSWT_Mul"),
        helper.make_node("Transpose", ["y_t"], ["y"], name="TSWT_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TSWT_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_swish_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_gelu_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TGT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Gelu", ["x_t"], ["g_t"], name="TGT_Gelu"),
        helper.make_node("Transpose", ["g_t"], ["z"], name="TGT_PostT", perm=[0, 2, 3, 1]),
    ]
    graph = helper.make_graph(nodes, "transpose_gelu_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 20)])


def _make_transpose_gelu_tanh_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    c0 = numpy_helper.from_array(np.asarray(0.044715, dtype=np.float32), name="tgtt_c0")
    c1 = numpy_helper.from_array(np.asarray(0.7978845834732056, dtype=np.float32), name="tgtt_c1")
    c2 = numpy_helper.from_array(np.asarray(1.0, dtype=np.float32), name="tgtt_c2")
    c3 = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="tgtt_c3")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TGTT_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["x_t", "x_t"], ["x2_t"], name="TGTT_MulSq"),
        helper.make_node("Mul", ["x2_t", "x_t"], ["x3_t"], name="TGTT_MulCube"),
        helper.make_node("Mul", ["x3_t", "tgtt_c0"], ["m0_t"], name="TGTT_MulC0"),
        helper.make_node("Add", ["x_t", "m0_t"], ["a0_t"], name="TGTT_Add0"),
        helper.make_node("Mul", ["a0_t", "tgtt_c1"], ["m1_t"], name="TGTT_MulC1"),
        helper.make_node("Tanh", ["m1_t"], ["t_t"], name="TGTT_Tanh"),
        helper.make_node("Add", ["t_t", "tgtt_c2"], ["a1_t"], name="TGTT_Add1"),
        helper.make_node("Mul", ["x_t", "a1_t"], ["m2_t"], name="TGTT_MulX"),
        helper.make_node("Mul", ["m2_t", "tgtt_c3"], ["y_t"], name="TGTT_MulHalf"),
        helper.make_node("Transpose", ["y_t"], ["y"], name="TGTT_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TGTT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_gelu_tanh_transpose_graph",
        [x],
        [z],
        initializer=[c0, c1, c2, c3],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_add_reshape_transpose_with_legacy_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 6, 4])
    legacy = helper.make_tensor_value_info("legacy", TensorProto.FLOAT, [1, 4, 2, 3])
    shape = numpy_helper.from_array(np.asarray([1, 4, 6], dtype=np.int64), name="tars_shape")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TARS_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TARS_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_nchw", "y_nchw"], ["sum_nchw"], name="TARS_Add"),
        helper.make_node("Reshape", ["sum_nchw", "tars_shape"], ["sum_ncw"], name="TARS_Reshape"),
        helper.make_node("Transpose", ["sum_ncw"], ["z"], name="TARS_Post", perm=[0, 2, 1]),
        helper.make_node("Add", ["sum_nchw", "x_nchw"], ["legacy"], name="TARS_LegacyAdd"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_add_reshape_transpose_with_legacy_graph",
        [x, y],
        [z, legacy],
        initializer=[shape],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_mulconst_add_reshape_transpose_with_legacy_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 6, 4])
    legacy = helper.make_tensor_value_info("legacy", TensorProto.FLOAT, [1, 4, 2, 3])
    shape = numpy_helper.from_array(np.asarray([1, 4, 6], dtype=np.int64), name="tmacrs_shape")
    scale = numpy_helper.from_array(
        np.asarray(np.full((1, 4, 1, 1), 0.5, dtype=np.float32)),
        name="tmacrs_scale",
    )
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TMACRS_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TMACRS_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["y_nchw", "tmacrs_scale"], ["y_scaled_nchw"], name="TMACRS_Mul"),
        helper.make_node("Add", ["x_nchw", "y_scaled_nchw"], ["sum_nchw"], name="TMACRS_Add"),
        helper.make_node("Reshape", ["sum_nchw", "tmacrs_shape"], ["sum_ncw"], name="TMACRS_Reshape"),
        helper.make_node("Transpose", ["sum_ncw"], ["z_mid"], name="TMACRS_Post", perm=[0, 2, 1]),
        helper.make_node("Identity", ["z_mid"], ["z"], name="TMACRS_Out"),
        helper.make_node("Add", ["sum_nchw", "x_nchw"], ["legacy"], name="TMACRS_LegacyAdd"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_mulconst_add_reshape_transpose_with_legacy_graph",
        [x, y],
        [z, legacy],
        initializer=[shape, scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_nested_add_with_shared_skip_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 4])
    scale = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="tnas_scale")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TNAS_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_nchw"], name="TNAS_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Mul", ["y_nchw", "tnas_scale"], ["y_scaled_nchw"], name="TNAS_Mul"),
        helper.make_node("Add", ["x_nchw", "y_scaled_nchw"], ["sum0_nchw"], name="TNAS_Add0"),
        helper.make_node("Add", ["sum0_nchw", "x_nchw"], ["sum1_nchw"], name="TNAS_Add1"),
        helper.make_node("Transpose", ["sum1_nchw"], ["sum1_nhwc"], name="TNAS_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum1_nhwc"], ["z"], name="TNAS_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_nested_add_with_shared_skip_graph",
        [x, y],
        [z],
        initializer=[scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_swish_add_concat_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2, 3, 8])
    nodes = [
        helper.make_node("Transpose", ["x"], ["a_t"], name="TSACT_PreA", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["a_t"], ["a_sig"], name="TSACT_ASig"),
        helper.make_node("Mul", ["a_t", "a_sig"], ["a_swish"], name="TSACT_AMul"),
        helper.make_node("Transpose", ["x"], ["b_t"], name="TSACT_PreB", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["b_t"], ["b_sig"], name="TSACT_BSig"),
        helper.make_node("Mul", ["b_t", "b_sig"], ["b_swish"], name="TSACT_BMul"),
        helper.make_node("Transpose", ["x"], ["c_t"], name="TSACT_PreC", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["c_t"], ["c_sig"], name="TSACT_CSig"),
        helper.make_node("Mul", ["c_t", "c_sig"], ["c_swish"], name="TSACT_CMul"),
        helper.make_node("Add", ["b_swish", "c_swish"], ["sum_nchw"], name="TSACT_Add"),
        helper.make_node("Concat", ["sum_nchw", "a_swish"], ["cat_nchw"], name="TSACT_Concat", axis=1),
        helper.make_node("Transpose", ["cat_nchw"], ["y"], name="TSACT_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["y"], ["z"], name="TSACT_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_swish_add_concat_transpose_graph", [x], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_swish_add_transpose_cascade_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Sigmoid", ["x"], ["a_sig"], name="TSATC_ASig"),
        helper.make_node("Mul", ["x", "a_sig"], ["a_nhwc"], name="TSATC_AMul"),
        helper.make_node("Transpose", ["a_nhwc"], ["b_nchw"], name="TSATC_BT", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["x"], ["c_nchw"], name="TSATC_CT", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["c_nchw"], ["c_sig"], name="TSATC_CSig"),
        helper.make_node("Mul", ["c_nchw", "c_sig"], ["c_swish"], name="TSATC_CMul"),
        helper.make_node("Add", ["c_swish", "b_nchw"], ["sum1_nchw"], name="TSATC_Add1"),
        helper.make_node("Transpose", ["sum1_nchw"], ["sum1_nhwc"], name="TSATC_Post1", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["x"], ["d_nchw"], name="TSATC_DT", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["d_nchw"], ["d_sig"], name="TSATC_DSig"),
        helper.make_node("Mul", ["d_nchw", "d_sig"], ["d_swish"], name="TSATC_DMul"),
        helper.make_node("Add", ["d_swish", "sum1_nchw"], ["sum2_nchw"], name="TSATC_Add2"),
        helper.make_node("Transpose", ["sum2_nchw"], ["sum2_nhwc"], name="TSATC_Post2", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["sum2_nhwc"], ["y"], name="TSATC_Relu"),
    ]
    graph = helper.make_graph(nodes, "transpose_swish_add_transpose_cascade_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_slice_concat_transpose_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 8, 8, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 12])
    starts_00 = numpy_helper.from_array(np.asarray([0, 0], dtype=np.int64), name="tsct_starts_00")
    starts_01 = numpy_helper.from_array(np.asarray([0, 1], dtype=np.int64), name="tsct_starts_01")
    starts_10 = numpy_helper.from_array(np.asarray([1, 0], dtype=np.int64), name="tsct_starts_10")
    starts_11 = numpy_helper.from_array(np.asarray([1, 1], dtype=np.int64), name="tsct_starts_11")
    ends = numpy_helper.from_array(np.asarray([8, 8], dtype=np.int64), name="tsct_ends")
    axes = numpy_helper.from_array(np.asarray([2, 3], dtype=np.int64), name="tsct_axes")
    steps = numpy_helper.from_array(np.asarray([2, 2], dtype=np.int64), name="tsct_steps")
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_nchw"], name="TSCT_Pre", perm=[0, 3, 1, 2]),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_00", "tsct_ends", "tsct_axes", "tsct_steps"], ["s00"], name="TSCT_Slice00"),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_01", "tsct_ends", "tsct_axes", "tsct_steps"], ["s01"], name="TSCT_Slice01"),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_10", "tsct_ends", "tsct_axes", "tsct_steps"], ["s10"], name="TSCT_Slice10"),
        helper.make_node("Slice", ["x_nchw", "tsct_starts_11", "tsct_ends", "tsct_axes", "tsct_steps"], ["s11"], name="TSCT_Slice11"),
        helper.make_node("Concat", ["s00", "s10", "s01", "s11"], ["cat_nchw"], name="TSCT_Concat", axis=1),
        helper.make_node("Transpose", ["cat_nchw"], ["cat_nhwc"], name="TSCT_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["cat_nhwc"], ["y"], name="TSCT_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_slice_concat_transpose_graph",
        [x],
        [y],
        initializer=[starts_00, starts_01, starts_10, starts_11, ends, axes, steps],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_logistic_concat_reshape_model() -> onnx.ModelProto:
    bbox = helper.make_tensor_value_info("bbox", TensorProto.FLOAT, [1, 8, 8, 4])
    obj = helper.make_tensor_value_info("obj", TensorProto.FLOAT, [1, 8, 8, 1])
    cls = helper.make_tensor_value_info("cls", TensorProto.FLOAT, [1, 8, 8, 80])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 85, 64])
    reshape_shape = numpy_helper.from_array(
        np.asarray([1, 85, 64], dtype=np.int64),
        name="tlcr_shape",
    )
    nodes = [
        helper.make_node("Transpose", ["bbox"], ["bbox_nchw"], name="TLCR_PreBbox", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["obj"], ["obj_nchw"], name="TLCR_PreObj", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["cls"], ["cls_nchw"], name="TLCR_PreCls", perm=[0, 3, 1, 2]),
        helper.make_node("Sigmoid", ["obj_nchw"], ["obj_sig"], name="TLCR_ObjSigmoid"),
        helper.make_node("Sigmoid", ["cls_nchw"], ["cls_sig"], name="TLCR_ClsSigmoid"),
        helper.make_node("Concat", ["bbox_nchw", "obj_sig", "cls_sig"], ["cat_nchw"], name="TLCR_Concat", axis=1),
        helper.make_node("Reshape", ["cat_nchw", "tlcr_shape"], ["y"], name="TLCR_Reshape"),
    ]
    graph = helper.make_graph(
        nodes,
        "transpose_logistic_concat_reshape_graph",
        [bbox, obj, cls],
        [y],
        initializer=[reshape_shape],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_transpose_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRTF_PreT", perm=[0, 2, 3, 1]),
        helper.make_node("Clip", ["x_t"], ["r_t"], name="TRTF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["r_t"], ["r0"], name="TRTF_PostT0", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["r_t"], ["r1"], name="TRTF_PostT1", perm=[0, 3, 1, 2]),
        helper.make_node("Relu", ["r0"], ["y0"], name="TRTF_Relu0"),
        helper.make_node("Neg", ["r1"], ["y1"], name="TRTF_Neg1"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_transpose_fanout_graph", [x], [y0, y1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_transpose_mixed_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4, 2, 3])
    y0 = helper.make_tensor_value_info("y0", TensorProto.FLOAT, [1, 2, 3, 4])
    y1 = helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRTMF_PreT", perm=[0, 3, 1, 2]),
        helper.make_node("Clip", ["x_t"], ["r_t"], name="TRTMF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["r_t"], ["r0"], name="TRTMF_PostT", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["r0"], ["y0"], name="TRTMF_Relu0"),
        helper.make_node("Add", ["r_t", "s"], ["y1"], name="TRTMF_LegacyAdd"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_transpose_mixed_fanout_graph", [x, s], [y0, y1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_binary_transpose_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    o0 = helper.make_tensor_value_info("o0", TensorProto.FLOAT, [1, 2, 3, 4])
    o1 = helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1, 2, 3, 4])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRBTF_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Clip", ["x_t"], ["x_r6_t"], name="TRBTF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["y"], ["y_t"], name="TRBTF_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_r6_t", "y_t"], ["z_t"], name="TRBTF_Add"),
        helper.make_node("Transpose", ["z_t"], ["z0"], name="TRBTF_Post0", perm=[0, 2, 3, 1]),
        helper.make_node("Transpose", ["z_t"], ["z1"], name="TRBTF_Post1", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["z0"], ["o0"], name="TRBTF_Relu0"),
        helper.make_node("Neg", ["z1"], ["o1"], name="TRBTF_Neg1"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_binary_transpose_fanout_graph", [x, y], [o0, o1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_relu6_binary_mixed_fanout_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4, 2, 3])
    o0 = helper.make_tensor_value_info("o0", TensorProto.FLOAT, [1, 2, 3, 4])
    o1 = helper.make_tensor_value_info("o1", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name="TRBMF_PreX", perm=[0, 3, 1, 2]),
        helper.make_node("Clip", ["x_t"], ["x_r6_t"], name="TRBMF_Relu6", min=0.0, max=6.0),
        helper.make_node("Transpose", ["y"], ["y_t"], name="TRBMF_PreY", perm=[0, 3, 1, 2]),
        helper.make_node("Add", ["x_r6_t", "y_t"], ["z_t"], name="TRBMF_Add"),
        helper.make_node("Transpose", ["z_t"], ["z"], name="TRBMF_Post", perm=[0, 2, 3, 1]),
        helper.make_node("Relu", ["z"], ["o0"], name="TRBMF_Relu0"),
        helper.make_node("Add", ["z_t", "s"], ["o1"], name="TRBMF_LegacyAdd"),
    ]
    graph = helper.make_graph(nodes, "transpose_relu6_binary_mixed_fanout_graph", [x, y, s], [o0, o1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_transpose_fanout_model(binary_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    s = helper.make_tensor_value_info("s", TensorProto.FLOAT, [1, 4, 2, 3])
    out_nhwc = helper.make_tensor_value_info("out_nhwc", TensorProto.FLOAT, [1, 2, 3, 4])
    out_nchw = helper.make_tensor_value_info("out_nchw", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name=f"TBTF_PreT0_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_t"], name=f"TBTF_PreT1_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node(binary_op, ["x_t", "y_t"], ["z_t"], name=f"TBTF_{binary_op}"),
        helper.make_node("Transpose", ["z_t"], ["z"], name=f"TBTF_PostT_{binary_op}", perm=[0, 2, 3, 1]),
        helper.make_node("Add", ["z_t", "s"], ["out_nchw"], name="TBTF_FanoutAdd"),
        helper.make_node("Relu", ["z"], ["out_nhwc"], name="TBTF_Relu"),
    ]
    graph = helper.make_graph(
        nodes,
        f"transpose_{binary_op.lower()}_transpose_fanout_graph",
        [x, y, s],
        [out_nhwc, out_nchw],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_no_post_transpose_model(binary_op: str) -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 4, 2, 3])
    nodes = [
        helper.make_node("Transpose", ["x"], ["x_t"], name=f"TBNP_PreT0_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node("Transpose", ["y"], ["y_t"], name=f"TBNP_PreT1_{binary_op}", perm=[0, 3, 1, 2]),
        helper.make_node(binary_op, ["x_t", "y_t"], ["z_t"], name=f"TBNP_{binary_op}"),
        helper.make_node("Relu", ["z_t"], ["z"], name=f"TBNP_Relu_{binary_op}"),
    ]
    graph = helper.make_graph(
        nodes,
        f"transpose_{binary_op.lower()}_no_post_transpose_graph",
        [x, y],
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_binary_single_side_transpose_model(
    binary_op: str,
    *,
    transpose_on_lhs: bool,
) -> onnx.ModelProto:
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3, 4, 2])

    if transpose_on_lhs:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4, 2])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 3, 4])
        nodes = [
            helper.make_node("Transpose", ["x"], ["x_t"], name=f"TSB_PreT0_{binary_op}", perm=[0, 3, 1, 2]),
            helper.make_node(binary_op, ["x_t", "y"], ["z_t"], name=f"TSB_{binary_op}"),
            helper.make_node("Transpose", ["z_t"], ["z"], name=f"TSB_PostT_{binary_op}", perm=[0, 2, 3, 1]),
        ]
        inputs = [x, y]
    else:
        x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3, 4])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4, 2])
        nodes = [
            helper.make_node("Transpose", ["y"], ["y_t"], name=f"TSB_PreT1_{binary_op}", perm=[0, 3, 1, 2]),
            helper.make_node(binary_op, ["x", "y_t"], ["z_t"], name=f"TSB_{binary_op}"),
            helper.make_node("Transpose", ["z_t"], ["z"], name=f"TSB_PostT_{binary_op}", perm=[0, 2, 3, 1]),
        ]
        inputs = [x, y]

    graph = helper.make_graph(
        nodes,
        f"transpose_single_side_{binary_op.lower()}_transpose_graph",
        inputs,
        [z],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_qlinear_fc_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])

    x_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="x_scale_fc")
    x_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="x_zero_fc")
    mul_const = numpy_helper.from_array(np.asarray([2], dtype=np.int8), name="mul_const_fc")
    mul_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mul_scale_fc")
    mul_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mul_zero_fc")
    mul_out_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="mul_out_scale_fc")
    bn_scale = numpy_helper.from_array(np.asarray([1.0, 0.9, 1.1, 1.2], dtype=np.float32), name="bn_scale_fc")
    bn_bias = numpy_helper.from_array(np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32), name="bn_bias_fc")
    bn_mean = numpy_helper.from_array(np.asarray([0.0, 0.0, 0.0, 0.0], dtype=np.float32), name="bn_mean_fc")
    bn_var = numpy_helper.from_array(np.asarray([1.0, 1.0, 1.0, 1.0], dtype=np.float32), name="bn_var_fc")
    prelu_slope = numpy_helper.from_array(np.asarray([0.1, 0.1, 0.1, 0.1], dtype=np.float32), name="prelu_slope_fc")
    q_mid_scale = numpy_helper.from_array(np.asarray([0.2], dtype=np.float32), name="q_mid_scale_fc")
    q_mid_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="q_mid_zero_fc")
    mm_w = numpy_helper.from_array(
        np.asarray(
            [
                [1, 0],
                [0, 1],
                [1, 0],
                [0, 1],
            ],
            dtype=np.int8,
        ),
        name="mm_w_fc",
    )
    mm_w_scale = numpy_helper.from_array(np.asarray([0.1], dtype=np.float32), name="mm_w_scale_fc")
    mm_w_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mm_w_zero_fc")
    mm_out_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="mm_out_scale_fc")
    mm_out_zero = numpy_helper.from_array(np.asarray([0], dtype=np.int8), name="mm_out_zero_fc")
    add_b = numpy_helper.from_array(np.asarray([[1, -1]], dtype=np.int8), name="add_b_fc")
    add_b_scale = numpy_helper.from_array(np.asarray([0.25], dtype=np.float32), name="add_b_scale_fc")

    nodes = [
        helper.make_node("QuantizeLinear", ["x", "x_scale_fc", "x_zero_fc"], ["x_q"], name="Q0FC"),
        helper.make_node(
            "QLinearMul",
            [
                "x_q",
                "x_scale_fc",
                "x_zero_fc",
                "mul_const_fc",
                "mul_scale_fc",
                "mul_zero_fc",
                "mul_out_scale_fc",
                "mul_zero_fc",
            ],
            ["x_mul_q"],
            name="QMulFC",
        ),
        helper.make_node("DequantizeLinear", ["x_mul_q", "mul_out_scale_fc", "mul_zero_fc"], ["x_dq"], name="DQ0FC"),
        helper.make_node(
            "BatchNormalization",
            ["x_dq", "bn_scale_fc", "bn_bias_fc", "bn_mean_fc", "bn_var_fc"],
            ["x_bn"],
            name="BNFC",
            epsilon=1e-5,
        ),
        helper.make_node("PRelu", ["x_bn", "prelu_slope_fc"], ["x_prelu"], name="PReluFC"),
        helper.make_node("QuantizeLinear", ["x_prelu", "q_mid_scale_fc", "q_mid_zero_fc"], ["x_mid_q"], name="Q1FC"),
        helper.make_node(
            "QLinearMatMul",
            [
                "x_mid_q",
                "q_mid_scale_fc",
                "q_mid_zero_fc",
                "mm_w_fc",
                "mm_w_scale_fc",
                "mm_w_zero_fc",
                "mm_out_scale_fc",
                "mm_out_zero_fc",
            ],
            ["mm_q"],
            name="QMMFC",
        ),
        helper.make_node(
            "QLinearAdd",
            [
                "mm_q",
                "mm_out_scale_fc",
                "mm_out_zero_fc",
                "add_b_fc",
                "add_b_scale_fc",
                "mm_out_zero_fc",
                "mm_out_scale_fc",
                "mm_out_zero_fc",
            ],
            ["y_q"],
            name="QAddFC",
        ),
        helper.make_node("DequantizeLinear", ["y_q", "mm_out_scale_fc", "mm_out_zero_fc"], ["y"], name="DQ1FC"),
    ]
    graph = helper.make_graph(
        nodes,
        "qlinear_fc_chain_graph",
        [x],
        [y],
        initializer=[
            x_scale,
            x_zero,
            mul_const,
            mul_scale,
            mul_zero,
            mul_out_scale,
            bn_scale,
            bn_bias,
            bn_mean,
            bn_var,
            prelu_slope,
            q_mid_scale,
            q_mid_zero,
            mm_w,
            mm_w_scale,
            mm_w_zero,
            mm_out_scale,
            mm_out_zero,
            add_b,
            add_b_scale,
        ],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_space_to_depth_chain_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 4, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 8, 2, 2])
    shape1 = numpy_helper.from_array(np.array([1, 2, 2, 2, 2, 2], dtype=np.int64), name="shape1")
    shape2 = numpy_helper.from_array(np.array([1, 8, 2, 2], dtype=np.int64), name="shape2")
    n0 = helper.make_node("Reshape", ["x", "shape1"], ["r1"], name="ReshapeNode0")
    n1 = helper.make_node("Transpose", ["r1"], ["t1"], name="TransposeNode", perm=[0, 1, 3, 5, 2, 4])
    n2 = helper.make_node("Reshape", ["t1", "shape2"], ["y"], name="ReshapeNode1")
    graph = helper.make_graph([n0, n1, n2], "space_to_depth_chain_graph", [x], [y], initializer=[shape1, shape2])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_transpose_attr_only_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 2])
    node = helper.make_node("Transpose", ["x"], ["y"], name="TransposeAttrOnlyNode", perm=[0, 2, 1])
    graph = helper.make_graph([node], "transpose_attr_only_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reshape_minus_one_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    shape = numpy_helper.from_array(np.array([-1, 2], dtype=np.int64), name="reshape_shape")
    node = helper.make_node("Reshape", ["x", "reshape_shape"], ["y"], name="ReshapeMinusOneNode")
    graph = helper.make_graph([node], "reshape_minus_one_graph", [x], [y], initializer=[shape])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_softmax_axis_non_last_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxAxisNonLastNode", axis=1)
    graph = helper.make_graph([node], "softmax_axis_non_last_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_logsoftmax_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 4])
    node = helper.make_node("LogSoftmax", ["x"], ["y"], name="LogSoftmaxNode", axis=2)
    graph = helper.make_graph([node], "logsoftmax_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_axes_concat_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    c0 = numpy_helper.from_array(np.array([1], dtype=np.int64), name="c0")
    c1 = numpy_helper.from_array(np.array([], dtype=np.int64), name="c1")
    n0 = helper.make_node("Concat", ["c0", "c1"], ["axes"], name="AxesConcatNode", axis=0)
    n1 = helper.make_node("ReduceSum", ["x", "axes"], ["y"], name="ReduceConstFoldNode", keepdims=1)
    graph = helper.make_graph([n0, n1], "reduce_axes_concat_const_graph", [x], [y], initializer=[c0, c1])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_mean_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 1])
    axes = numpy_helper.from_array(np.array([2], dtype=np.int64), name="axes")
    node = helper.make_node(
        "ReduceMean",
        ["x", "axes"],
        ["y"],
        name="ReduceMeanNode",
        keepdims=1,
    )
    graph = helper.make_graph([node], "reduce_mean_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_reduce_sum_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node(
        "ReduceSum",
        ["x", "axes"],
        ["y"],
        name="ReduceSumNode",
        keepdims=1,
    )
    graph = helper.make_graph([node], "reduce_sum_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_squeeze_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node("Squeeze", ["x", "axes"], ["y"], name="SqueezeNode")
    graph = helper.make_graph([node], "squeeze_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_unsqueeze_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1, 3])
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    node = helper.make_node("Unsqueeze", ["x", "axes"], ["y"], name="UnsqueezeNode")
    graph = helper.make_graph([node], "unsqueeze_graph", [x], [y], initializer=[axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2, 2])
    indices = numpy_helper.from_array(np.array([3, 1], dtype=np.int64), name="indices")
    node = helper.make_node("Gather", ["x", "indices"], ["y"], name="GatherNode", axis=1)
    graph = helper.make_graph([node], "gather_graph", [x], [y], initializer=[indices])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_singleton_indices_rank_reduced_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    indices = numpy_helper.from_array(np.array([0], dtype=np.int64), name="indices")
    node = helper.make_node(
        "Gather",
        ["x", "indices"],
        ["y"],
        name="GatherSingletonIndicesNode",
        axis=2,
    )
    graph = helper.make_graph(
        [node],
        "gather_singleton_indices_rank_reduced_graph",
        [x],
        [y],
        initializer=[indices],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_l2_norm_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4])
    node = helper.make_node(
        "LpNormalization",
        ["x"],
        ["y"],
        name="LpNormNode",
        axis=-1,
        p=2,
    )
    graph = helper.make_graph([node], "l2norm_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_lrn_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 5, 5])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 5, 5])
    node = helper.make_node(
        "LRN",
        ["x"],
        ["y"],
        name="LRNNode",
        size=3,
        alpha=1e-4,
        beta=0.75,
        bias=1.0,
    )
    graph = helper.make_graph([node], "lrn_graph", [x], [y])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gather_int32_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.INT32, [2, 4])
    y = helper.make_tensor_value_info("y", TensorProto.INT32, [2, 2])
    indices = numpy_helper.from_array(np.array([3, 1], dtype=np.int64), name="indices")
    node = helper.make_node("Gather", ["x", "indices"], ["y"], name="GatherIntNode", axis=1)
    graph = helper.make_graph([node], "gather_int_graph", [x], [y], initializer=[indices])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_gemm_reduce_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 1])
    w = numpy_helper.from_array(np.ones((3, 4), dtype=np.float32), name="W")
    b = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="B")
    axes = numpy_helper.from_array(np.array([1], dtype=np.int64), name="axes")
    gemm = helper.make_node("Gemm", ["x", "W", "B"], ["h"], name="GemmNode", transB=1)
    relu = helper.make_node("Relu", ["h"], ["r"], name="ReluNode")
    reduce = helper.make_node("ReduceMean", ["r", "axes"], ["y"], name="ReduceNode", keepdims=1)
    graph = helper.make_graph([gemm, relu, reduce], "gemm_reduce_graph", [x], [y], initializer=[w, b, axes])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_shared_input_conv_and_mul_branches_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 8, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3, 8, 8])
    w0 = numpy_helper.from_array(
        np.ones((3, 3, 1, 1), dtype=np.float32),
        name="sib_w0",
    )
    b0 = numpy_helper.from_array(np.zeros((3,), dtype=np.float32), name="sib_b0")
    scale = numpy_helper.from_array(np.asarray(0.5, dtype=np.float32), name="sib_scale")
    nodes = [
        helper.make_node("Conv", ["x", "sib_w0", "sib_b0"], ["y0"], name="SIB_Conv0"),
        helper.make_node("Mul", ["x", "sib_scale"], ["y1"], name="SIB_Mul1"),
        helper.make_node("Add", ["y0", "y1"], ["y"], name="SIB_Add"),
    ]
    graph = helper.make_graph(
        nodes,
        "shared_input_conv_and_mul_branches_graph",
        [x],
        [y],
        initializer=[w0, b0, scale],
    )
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_custom_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 2])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [2, 3])
    node = helper.make_node(
        "Einsum",
        ["x", "y"],
        ["z"],
        name="EinsumCustomNode",
        equation="ij,jk->kj",
    )
    graph = helper.make_graph([node], "einsum_custom_graph", [x, y], [z])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _make_einsum_fc_const_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 2])
    w = numpy_helper.from_array(
        np.array(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ],
            dtype=np.float32,
        ),
        name="W",
    )
    node = helper.make_node(
        "Einsum",
        ["x", "W"],
        ["z"],
        name="EinsumConstNode",
        equation="ij,jk->ik",
    )
    graph = helper.make_graph([node], "einsum_fc_const_graph", [x], [z], initializer=[w])
    return helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])


def _requires_flatbuffer_tools() -> bool:
    return shutil.which("flatc") is not None and shutil.which("curl") is not None


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {}
    for key, value in updates.items():
        previous[key] = os.environ.get(key, None)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _collect_int8_quant_scale_lengths(tflite_path: str) -> list[int]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    subgraph = model.subgraphs[0]
    int8_type = getattr(schema["TensorType"], "INT8")
    lengths: list[int] = []
    for tensor in subgraph.tensors:
        if tensor.type != int8_type:
            continue
        if tensor.quantization is None or tensor.quantization.scale is None:
            continue
        lengths.append(len(list(tensor.quantization.scale)))
    return lengths


def _collect_custom_codes(tflite_path: str) -> list[str]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    builtin_custom = getattr(schema["BuiltinOperator"], "CUSTOM")
    custom_codes: list[str] = []
    for op_code in model.operatorCodes:
        if int(op_code.builtinCode) == int(builtin_custom):
            value = op_code.customCode
            if isinstance(value, bytes):
                custom_codes.append(value.decode("utf-8"))
            else:
                custom_codes.append(str(value))
    return custom_codes


def _collect_builtin_op_names(tflite_path: str) -> list[str]:
    output_dir = os.path.dirname(tflite_path)
    schema = load_schema_module(output_dir)
    with open(tflite_path, "rb") as f:
        model_bytes = f.read()
    model = schema["ModelT"].InitFromObj(schema["Model"].GetRootAs(model_bytes, 0))
    subgraph = model.subgraphs[0]
    op_enum = schema["BuiltinOperator"]
    op_names: list[str] = []
    for op in subgraph.operators:
        op_code = model.operatorCodes[int(op.opcodeIndex)]
        builtin_code = int(op_code.builtinCode)
        resolved = str(builtin_code)
        for attr in dir(op_enum):
            if not attr.isupper():
                continue
            if int(getattr(op_enum, attr)) == builtin_code:
                resolved = attr
                break
        op_names.append(resolved)
    return op_names


def test_tflite_backend_matrix_add() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add", model)

        tf_out = os.path.join(tmpdir, "tf_converter")
        tf_tflite = _convert(model_path, tf_out, "tf_converter")
        tf_pred = _run_add_inference(tf_tflite)

        if not _requires_flatbuffer_tools():
            pytest.skip("flatbuffer_direct requires flatc and curl")

        fb_out = os.path.join(tmpdir, "flatbuffer_direct")
        fb_tflite = _convert(model_path, fb_out, "flatbuffer_direct")
        fb_pred = _run_add_inference(fb_tflite)

        np.testing.assert_allclose(tf_pred, np.array([[5.0, 7.0, 9.0]], dtype=np.float32), rtol=0.0, atol=1e-6)
        np.testing.assert_allclose(fb_pred, tf_pred, rtol=0.0, atol=1e-6)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_tflite_backend_matrix_hardswish_rewrite_on_off(monkeypatch: pytest.MonkeyPatch) -> None:
    import onnx2tf.tflite_builder as tflite_builder_backend
    from onnx2tf.tflite_builder.preprocess import clear_preprocess_rules

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_hardswish_model()
        model_path = _save_model(tmpdir, "hardswish_rewrite_matrix", model)

        # tf_converter backend should pass without direct preprocess dependency.
        tf_out = os.path.join(tmpdir, "tf_converter")
        tf_tflite = _convert(model_path, tf_out, "tf_converter")
        assert os.path.isfile(tf_tflite)

        # flatbuffer_direct with default preprocess should pass (rewrite enabled).
        fb_out = os.path.join(tmpdir, "flatbuffer_direct")
        fb_tflite = _convert(model_path, fb_out, "flatbuffer_direct")
        assert os.path.isfile(fb_tflite)

        # HardSwish is now lowered as builtin even when rewrite preprocess is disabled.
        clear_preprocess_rules()
        monkeypatch.setattr(
            tflite_builder_backend,
            "register_default_preprocess_rules",
            lambda: None,
        )
        fb_no_rewrite_out = os.path.join(tmpdir, "flatbuffer_direct_no_rewrite")
        _convert(
            model_path,
            fb_no_rewrite_out,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(
            fb_no_rewrite_out,
            "hardswish_rewrite_matrix_op_coverage_report.json",
        )
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["unsupported_nodes"] == 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
@pytest.mark.parametrize(
    "name, model_factory",
    [
        ("conv", _make_conv_model),
        ("conv1d", _make_conv1d_model),
        ("convtranspose1d", _make_convtranspose1d_model),
        ("pool", _make_pool_model),
        ("gemm", _make_gemm_model),
        ("reduce_mean", _make_reduce_mean_model),
        ("reduce_sum", _make_reduce_sum_model),
        ("squeeze", _make_squeeze_model),
        ("unsqueeze", _make_unsqueeze_model),
        ("gather", _make_gather_model),
        ("l2_norm", _make_l2_norm_model),
        ("lrn", _make_lrn_model),
        ("relu", lambda: _make_unary_model("Relu", name="relu")),
        ("tanh", lambda: _make_unary_model("Tanh", name="tanh")),
        ("exp", lambda: _make_unary_model("Exp", name="exp")),
        ("sqrt", lambda: _make_unary_model("Sqrt", name="sqrt")),
        ("neg", lambda: _make_unary_model("Neg", name="neg")),
        ("hardswish", _make_hardswish_model),
        ("leakyrelu", _make_leakyrelu_model),
        ("prelu", _make_prelu_model),
        ("gelu", _make_gelu_model),
        ("pow_square", _make_pow_square_model),
        ("dynamic_quantize_linear", _make_dynamic_quantize_linear_model),
        ("shape_slice", _make_shape_slice_model),
        ("fused_matmul", _make_fused_matmul_model),
        ("space_to_depth", _make_space_to_depth_model),
        ("resize_nearest", _make_resize_nearest_model),
        ("qlinear_sigmoid", _make_qlinear_sigmoid_model),
        ("qlinear_global_average_pool", _make_qlinear_global_average_pool_model),
        ("qlinear_concat", _make_qlinear_concat_model),
        ("space_to_depth_chain", _make_space_to_depth_chain_model),
        ("qlinear_conv_chain", _make_qlinear_conv_chain_model),
        ("qlinear_fc_chain", _make_qlinear_fc_chain_model),
        ("transpose_attr_only", _make_transpose_attr_only_model),
        ("softmax_axis_non_last", _make_softmax_axis_non_last_model),
        ("logsoftmax", _make_logsoftmax_model),
        ("reduce_axes_concat_const", _make_reduce_axes_concat_const_model),
        ("einsum_fc_const", _make_einsum_fc_const_model),
    ],
)
def test_flatbuffer_direct_operator_smoke(name: str, model_factory) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = model_factory()
        model_path = _save_model(tmpdir, name, model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        if name == "qlinear_fc_chain":
            custom_codes = _collect_custom_codes(tflite_path)
            assert "ONNX_QLINEARMATMUL" in custom_codes
            return
        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.ones(input_details[0]["shape"], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])


def test_flatbuffer_direct_transpose_chain_optimization() -> None:
    model = _make_qlinear_conv_pair_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1


def test_flatbuffer_direct_qlinear_conv_filter_layout_is_ohwi() -> None:
    model = _make_qlinear_conv_multichannel_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_multichannel_layout_test",
        optimize_layout_transpose_chains=False,
    )

    w_tensor = model_ir.tensors.get("QConvMC_conv_filter_q")
    assert w_tensor is not None
    assert list(w_tensor.shape) == [4, 1, 1, 3]


def test_flatbuffer_direct_input_layout_defaults_to_nhwc_when_enabled() -> None:
    model = _make_qlinear_conv_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="input_layout_default_nhwc_test",
        transpose_inputs_to_nhwc=True,
    )
    assert model_ir.inputs == ["x"]
    assert list(model_ir.tensors["x"].shape) == [1, 2, 2, 1]


def test_flatbuffer_direct_input_layout_respects_keep_shape_absolute() -> None:
    model = _make_qlinear_conv_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="input_layout_keep_absolute_test",
        transpose_inputs_to_nhwc=True,
        keep_shape_absolutely_input_names=["x"],
    )
    assert model_ir.inputs == ["x"]
    assert list(model_ir.tensors["x"].shape) == [1, 1, 2, 2]


def test_flatbuffer_direct_transpose_input_remap_preserves_boundary_transpose_for_multibranch_input() -> None:
    model = _make_shared_input_conv_and_mul_branches_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="shared_input_transpose_remap_test",
        transpose_inputs_to_nhwc=True,
        optimize_layout_transpose_chains=False,
    )

    internal_input_name = "x_onnx_ncx_internal"
    producer_count = sum(
        1 for op in model_ir.operators if internal_input_name in [str(v) for v in op.outputs]
    )
    consumer_count = sum(
        1 for op in model_ir.operators if internal_input_name in [str(v) for v in op.inputs]
    )
    if consumer_count > 0:
        assert producer_count == 1

    model_inputs = set(str(v) for v in model_ir.inputs)
    produced_tensors = set(
        str(output_name)
        for op in model_ir.operators
        for output_name in list(op.outputs)
    )
    dangling_runtime_inputs = []
    for op in model_ir.operators:
        for input_name in list(op.inputs):
            normalized = str(input_name)
            if normalized in model_inputs or normalized in produced_tensors:
                continue
            tensor = model_ir.tensors.get(normalized, None)
            if tensor is not None and tensor.data is not None:
                continue
            dangling_runtime_inputs.append(normalized)
    assert internal_input_name not in dangling_runtime_inputs


def test_flatbuffer_direct_grouped_conv_default_avoids_split_op() -> None:
    model = _make_grouped_conv_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="grouped_conv_default_no_split_test",
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "SPLIT" not in op_types
    assert "STRIDED_SLICE" not in op_types
    assert op_types.count("CONV_2D") == 1
    assert "CONCATENATION" not in op_types


def test_flatbuffer_direct_grouped_conv_dgc_uses_split_op() -> None:
    model = _make_grouped_conv_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="grouped_conv_dgc_split_test",
        disable_group_convolution=True,
        optimize_layout_transpose_chains=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SPLIT") == 1
    assert op_types.count("CONV_2D") == 2
    assert op_types.count("CONCATENATION") == 1


def test_flatbuffer_direct_reshape_minus_one_resolved_to_static_shape() -> None:
    model = _make_reshape_minus_one_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="reshape_minus_one_resolved_test",
    )
    reshape_ops = [op for op in model_ir.operators if str(op.op_type) == "RESHAPE"]
    assert len(reshape_ops) == 1
    reshape_op = reshape_ops[0]
    assert list(reshape_op.options.get("newShape", [])) == [3, 2]
    shape_tensor_name = reshape_op.inputs[1]
    shape_tensor = model_ir.tensors[shape_tensor_name]
    assert shape_tensor is not None
    assert shape_tensor.data is not None
    assert np.asarray(shape_tensor.data).reshape(-1).tolist() == [3, 2]
    output_tensor = model_ir.tensors["y"]
    assert list(output_tensor.shape) == [3, 2]
    assert list(output_tensor.shape_signature) == [3, 2]


def test_flatbuffer_direct_resolve_dynamic_reshape_shapes_pass() -> None:
    model_ir = ModelIR(name="reshape_fixup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT8",
        shape=[1, 40, 40, 1],
        shape_signature=[1, 40, 40, 1],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, -1, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="INT8",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "reshape_shape"],
            outputs=["y"],
            options={"newShape": [1, -1, 1]},
        )
    )

    stats = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.operators[0].options["newShape"]) == [1, 1600, 1]
    assert list(model_ir.tensors["y"].shape) == [1, 1600, 1]
    assert list(model_ir.tensors["y"].shape_signature) == [1, 1600, 1]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [1, 1600, 1]


def test_flatbuffer_direct_logsoftmax_lowering() -> None:
    model = _make_logsoftmax_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="logsoftmax_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("LOG") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_lrn_lowering() -> None:
    model = _make_lrn_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="lrn_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("LOCAL_RESPONSE_NORMALIZATION") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_pow_lowering() -> None:
    model = _make_pow_general_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="pow_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("POW") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_reciprocal_lowering() -> None:
    model = _make_reciprocal_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="reciprocal_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("DIV") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_onehot_lowering() -> None:
    model = _make_onehot_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="onehot_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("ONE_HOT") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_matmul_integer_lowering() -> None:
    model = _make_matmul_integer_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="matmul_integer_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("BATCH_MATMUL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_matmul_integer_batch_matmul_runtime_dtypes() -> None:
    model = _make_matmul_integer_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="matmul_integer_runtime_dtype_test",
        allow_custom_ops=False,
    )
    allowed = {"FLOAT32", "INT8", "INT16"}
    for op in model_ir.operators:
        if str(op.op_type) != "BATCH_MATMUL":
            continue
        in_dtypes = [str(model_ir.tensors[name].dtype).upper() for name in op.inputs]
        assert all(dtype in allowed for dtype in in_dtypes), in_dtypes


def test_flatbuffer_direct_dynamic_quantize_linear_lowering() -> None:
    model = _make_dynamic_quantize_linear_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dynamic_quantize_linear_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("REDUCE_MAX") == 2
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_shape_lowering() -> None:
    model = _make_shape_slice_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="shape_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SHAPE") == 1
    assert op_types.count("SLICE") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_legacy_slice_attr_lowering() -> None:
    model = _make_legacy_slice_attr_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="legacy_slice_attr_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SLICE") == 2
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_resize_dynamic_sizes_lowering() -> None:
    model = _make_resize_dynamic_sizes_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="resize_dynamic_sizes_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("RESIZE_BILINEAR") == 1
    assert op_types.count("CUSTOM") == 0

    resize_op = next(op for op in model_ir.operators if str(op.op_type) == "RESIZE_BILINEAR")
    size_input_name = str(resize_op.inputs[1])
    size_tensor = model_ir.tensors[size_input_name]
    assert str(size_tensor.dtype).upper() == "INT32"
    assert size_tensor.data is None


def test_flatbuffer_direct_constant_of_shape_lowering() -> None:
    model = _make_constant_of_shape_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="constant_of_shape_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("FILL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_fused_matmul_lowering() -> None:
    model = _make_fused_matmul_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fused_matmul_lowering_test",
        allow_custom_ops=False,
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("BATCH_MATMUL") == 1
    assert op_types.count("MUL") == 1
    assert op_types.count("CUSTOM") == 0


def test_flatbuffer_direct_resolve_dynamic_reshape_zero_copy_dims_pass() -> None:
    model_ir = ModelIR(name="reshape_zero_copy_fixup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 2, 256],
        shape_signature=[1, 1, 2, 256],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 0, -1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "reshape_shape"],
            outputs=["y"],
            options={
                "newShape": [0, 0, -1],
                "onnxRawNewShape": [0, 0, -1],
                "allowZero": False,
            },
        )
    )

    stats = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.operators[0].options["newShape"]) == [1, 1, 512]
    assert list(model_ir.tensors["y"].shape) == [1, 1, 512]
    assert list(model_ir.tensors["y"].shape_signature) == [1, 1, 512]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [1, 1, 512]


def test_flatbuffer_direct_reconcile_bilstm_chain_and_resolve_reshape_zero_copy_dims() -> None:
    model_ir = ModelIR(name="reshape_zero_copy_bilstm_chain_fixup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[25, 1, 512],
        shape_signature=[25, 1, 512],
    )
    model_ir.tensors["merged"] = TensorIR(
        name="merged",
        dtype="FLOAT32",
        shape=[1, 1, 512],
        shape_signature=[1, 1, 512],
    )
    model_ir.tensors["split_axis"] = TensorIR(
        name="split_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["fw"] = TensorIR(
        name="fw",
        dtype="FLOAT32",
        shape=[1, 1, 256],
        shape_signature=[1, 1, 256],
    )
    model_ir.tensors["bw"] = TensorIR(
        name="bw",
        dtype="FLOAT32",
        shape=[1, 1, 256],
        shape_signature=[1, 1, 256],
    )
    model_ir.tensors["expand_axis"] = TensorIR(
        name="expand_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["fw_expanded"] = TensorIR(
        name="fw_expanded",
        dtype="FLOAT32",
        shape=[1, 1, 1, 256],
        shape_signature=[1, 1, 1, 256],
    )
    model_ir.tensors["bw_expanded"] = TensorIR(
        name="bw_expanded",
        dtype="FLOAT32",
        shape=[1, 1, 1, 256],
        shape_signature=[1, 1, 1, 256],
    )
    model_ir.tensors["lstm_y"] = TensorIR(
        name="lstm_y",
        dtype="FLOAT32",
        shape=[1, 2, 1, 256],
        shape_signature=[1, 2, 1, 256],
    )
    model_ir.tensors["transpose_perm"] = TensorIR(
        name="transpose_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 1, 3], dtype=np.int32),
    )
    model_ir.tensors["transpose_out"] = TensorIR(
        name="transpose_out",
        dtype="FLOAT32",
        shape=[1, 1, 2, 256],
        shape_signature=[1, 1, 2, 256],
    )
    model_ir.tensors["reshape_shape"] = TensorIR(
        name="reshape_shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 0, -1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 1, 512],
        shape_signature=[1, 1, 512],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="BIDIRECTIONAL_SEQUENCE_LSTM",
                inputs=["x"],
                outputs=["merged"],
                options={"timeMajor": True},
            ),
            OperatorIR(
                op_type="SPLIT",
                inputs=["split_axis", "merged"],
                outputs=["fw", "bw"],
                options={"numSplits": 2},
            ),
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=["fw", "expand_axis"],
                outputs=["fw_expanded"],
            ),
            OperatorIR(
                op_type="EXPAND_DIMS",
                inputs=["bw", "expand_axis"],
                outputs=["bw_expanded"],
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["fw_expanded", "bw_expanded"],
                outputs=["lstm_y"],
                options={"axis": 1},
            ),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["lstm_y", "transpose_perm"],
                outputs=["transpose_out"],
            ),
            OperatorIR(
                op_type="RESHAPE",
                inputs=["transpose_out", "reshape_shape"],
                outputs=["y"],
                options={
                    "newShape": [0, 0, -1],
                    "onnxRawNewShape": [0, 0, -1],
                    "allowZero": False,
                },
            ),
        ]
    )

    stats_reconcile = _reconcile_static_tensor_shapes(model_ir)
    stats_resolve = _resolve_dynamic_reshape_shapes(model_ir)
    assert stats_reconcile["reconciled_static_tensor_shapes"] > 0
    assert stats_resolve["resolved_dynamic_reshape_shapes"] == 1
    assert list(model_ir.tensors["transpose_out"].shape) == [25, 1, 2, 256]
    assert list(model_ir.tensors["y"].shape) == [25, 1, 512]
    assert np.asarray(model_ir.tensors["reshape_shape"].data).reshape(-1).tolist() == [25, 1, 512]


def test_flatbuffer_direct_reconcile_cast_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_cast_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="CAST",
            inputs=["x"],
            outputs=["y"],
            options={"inDataType": "INT32", "outDataType": "FLOAT32"},
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 2]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 2]


def test_flatbuffer_direct_reconcile_neg_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_neg_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[-1, 2],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2],
        shape_signature=[1, 2],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="NEG",
            inputs=["x"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 2]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 2]


def test_flatbuffer_direct_reconcile_floor_mod_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_floor_mod_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="INT32",
        shape=[1],
        shape_signature=[-1],
    )
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="FLOOR_MOD",
            inputs=["x", "c"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1]
    assert list(model_ir.tensors["y"].shape_signature) == [-1]


def test_flatbuffer_direct_reconcile_maximum_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_maximum_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[-1, 3, 1],
    )
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="FLOAT32",
        shape=[1, 1, 4],
        shape_signature=[1, 1, 4],
        data=np.asarray([[[0.0, 0.0, 0.0, 0.0]]], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="MAXIMUM",
            inputs=["x", "c"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 3, 4]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 3, 4]


def test_flatbuffer_direct_reconcile_minimum_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_minimum_shape_signature_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 1],
        shape_signature=[-1, 3, 1],
    )
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="FLOAT32",
        shape=[1, 1, 4],
        shape_signature=[1, 1, 4],
        data=np.asarray([[[1.0, 1.0, 1.0, 1.0]]], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4],
        shape_signature=[1, 3, 4],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="MINIMUM",
            inputs=["x", "c"],
            outputs=["y"],
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 3, 4]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 3, 4]


def test_flatbuffer_direct_reconcile_batch_matmul_propagates_output_shape_signature() -> None:
    model_ir = ModelIR(name="reconcile_batch_matmul_shape_signature_test")
    model_ir.inputs = ["a", "b"]
    model_ir.outputs = ["y"]
    model_ir.tensors["a"] = TensorIR(
        name="a",
        dtype="FLOAT32",
        shape=[1, 2, 4],
        shape_signature=[-1, 2, 4],
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[1, 4, 3],
        shape_signature=[-1, 4, 3],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 3],
        shape_signature=[1, 2, 3],
    )
    model_ir.operators = [
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["a", "b"],
            outputs=["y"],
            options={"adjX": False, "adjY": False},
        )
    ]

    stats = _reconcile_static_tensor_shapes(model_ir)
    assert stats["reconciled_static_tensor_shapes"] >= 1
    assert list(model_ir.tensors["y"].shape) == [1, 2, 3]
    assert list(model_ir.tensors["y"].shape_signature) == [-1, 2, 3]


def test_flatbuffer_direct_transpose_mul_add_const_rewrite_keeps_nhwc_consts_stable() -> None:
    model_ir = ModelIR(name="transpose_mul_add_const_rewrite_idempotent_const_shape_test")
    model_ir.inputs = ["x_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 1, 1, 40],
        shape_signature=[1, 1, 1, 40],
        data=np.ones((1, 1, 1, 40), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["add_const"] = TensorIR(
        name="add_const",
        dtype="FLOAT32",
        shape=[1, 1, 1, 40],
        shape_signature=[1, 1, 1, 40],
        data=np.ones((1, 1, 1, 40), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "add_const"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["add_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_mul_add_const_prepost_nhwc_chains"] == 1
    assert list(model_ir.tensors["mul_const"].shape) == [1, 1, 1, 40]
    assert list(model_ir.tensors["add_const"].shape) == [1, 1, 1, 40]

    stats_second = _optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    assert stats_second["optimized_transpose_mul_add_const_prepost_nhwc_chains"] == 0
    assert list(model_ir.tensors["mul_const"].shape) == [1, 1, 1, 40]
    assert list(model_ir.tensors["add_const"].shape) == [1, 1, 1, 40]


def test_flatbuffer_direct_transpose_mul_add_const_rewrite_does_not_mutate_on_partial_match() -> None:
    model_ir = ModelIR(name="transpose_mul_add_const_partial_match_no_mutation_test")
    model_ir.inputs = ["x_nhwc", "residual_nchw"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["residual_nchw"] = TensorIR(
        name="residual_nchw",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["mul_const"] = TensorIR(
        name="mul_const",
        dtype="FLOAT32",
        shape=[1, 40, 1, 1],
        shape_signature=[1, 40, 1, 1],
        data=np.ones((1, 40, 1, 1), dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["mul_out"] = TensorIR(
        name="mul_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["add_out"] = TensorIR(
        name="add_out",
        dtype="FLOAT32",
        shape=[1, 40, 56, 56],
        shape_signature=[1, 40, 56, 56],
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 56, 56, 40],
        shape_signature=[1, 56, 56, 40],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "mul_const"],
            outputs=["mul_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="ADD",
            inputs=["mul_out", "residual_nchw"],
            outputs=["add_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["add_out", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["y_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_mul_add_const_prepost_nhwc_chains(model_ir)
    assert stats["optimized_transpose_mul_add_const_prepost_nhwc_chains"] == 0
    assert list(model_ir.tensors["mul_const"].shape) == [1, 40, 1, 1]


def test_flatbuffer_direct_transpose_pre_concat_nhwc_partial_match_does_not_mutate_swish_input() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_partial_match_no_mutation_test")
    model_ir.inputs = ["x_nhwc", "residual_nchw"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 384],
        shape_signature=[1, 6, 6, 384],
    )
    model_ir.tensors["residual_nchw"] = TensorIR(
        name="residual_nchw",
        dtype="FLOAT32",
        shape=[1, 192, 6, 6],
        shape_signature=[1, 192, 6, 6],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_nchw"] = TensorIR(
        name="x_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["sigmoid_out"] = TensorIR(
        name="sigmoid_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["swish_out"] = TensorIR(
        name="swish_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 576, 6, 6],
        shape_signature=[1, 576, 6, 6],
    )
    model_ir.tensors["concat_nhwc"] = TensorIR(
        name="concat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc", "pre_perm"], outputs=["x_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x_nchw"], outputs=["sigmoid_out"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x_nchw", "sigmoid_out"],
            outputs=["swish_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["swish_out", "residual_nchw"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["concat_nchw", "post_perm"], outputs=["concat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["concat_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 0
    assert list(model_ir.tensors["swish_out"].shape) == [1, 384, 6, 6]
    assert list(model_ir.tensors["swish_out"].shape_signature) == [1, 384, 6, 6]
    assert [str(op.op_type) for op in model_ir.operators] == [
        "TRANSPOSE",
        "LOGISTIC",
        "MUL",
        "CONCATENATION",
        "TRANSPOSE",
        "RELU",
    ]


def test_flatbuffer_direct_transpose_pre_concat_nhwc_relu_and_swish_inputs_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_relu_swish_opt_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 192],
        shape_signature=[1, 6, 6, 192],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 384],
        shape_signature=[1, 6, 6, 384],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x0_nchw"] = TensorIR(
        name="x0_nchw",
        dtype="FLOAT32",
        shape=[1, 192, 6, 6],
        shape_signature=[1, 192, 6, 6],
    )
    model_ir.tensors["relu_out"] = TensorIR(
        name="relu_out",
        dtype="FLOAT32",
        shape=[1, 192, 6, 6],
        shape_signature=[1, 192, 6, 6],
    )
    model_ir.tensors["x1_nchw"] = TensorIR(
        name="x1_nchw",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["sigmoid_out"] = TensorIR(
        name="sigmoid_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["swish_out"] = TensorIR(
        name="swish_out",
        dtype="FLOAT32",
        shape=[1, 384, 6, 6],
        shape_signature=[1, 384, 6, 6],
    )
    model_ir.tensors["concat_nchw"] = TensorIR(
        name="concat_nchw",
        dtype="FLOAT32",
        shape=[1, 576, 6, 6],
        shape_signature=[1, 576, 6, 6],
    )
    model_ir.tensors["concat_nhwc"] = TensorIR(
        name="concat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 576],
        shape_signature=[1, 6, 6, 576],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x0_nhwc", "pre_perm"], outputs=["x0_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x0_nchw"], outputs=["relu_out"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x1_nhwc", "pre_perm"], outputs=["x1_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["x1_nchw"], outputs=["sigmoid_out"]),
        OperatorIR(
            op_type="MUL",
            inputs=["x1_nchw", "sigmoid_out"],
            outputs=["swish_out"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["swish_out", "relu_out"],
            outputs=["concat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["concat_nchw", "post_perm"], outputs=["concat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["concat_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 1
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(model_ir.tensors["relu_out"].shape) == [1, 6, 6, 192]
    assert list(model_ir.tensors["swish_out"].shape) == [1, 6, 6, 384]


def test_flatbuffer_direct_transpose_pre_add_nhwc_shared_concat_and_unary_input() -> None:
    model_ir = ModelIR(name="transpose_pre_add_shared_concat_and_unary_test")
    model_ir.inputs = ["x0_nhwc", "x1_nhwc"]
    model_ir.outputs = ["legacy_cat"]
    model_ir.tensors["x0_nhwc"] = TensorIR(
        name="x0_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 64],
        shape_signature=[1, 6, 6, 64],
    )
    model_ir.tensors["x1_nhwc"] = TensorIR(
        name="x1_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 64],
        shape_signature=[1, 6, 6, 64],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x0_nchw"] = TensorIR(
        name="x0_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["x1_nchw"] = TensorIR(
        name="x1_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["x1_relu"] = TensorIR(
        name="x1_relu",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 64, 6, 6],
        shape_signature=[1, 64, 6, 6],
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 64],
        shape_signature=[1, 6, 6, 64],
    )
    model_ir.tensors["legacy_cat"] = TensorIR(
        name="legacy_cat",
        dtype="FLOAT32",
        shape=[1, 128, 6, 6],
        shape_signature=[1, 128, 6, 6],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x0_nhwc", "pre_perm"], outputs=["x0_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x1_nhwc", "pre_perm"], outputs=["x1_nchw"]),
        OperatorIR(op_type="RELU", inputs=["x1_nchw"], outputs=["x1_relu"]),
        OperatorIR(
            op_type="ADD",
            inputs=["x0_nchw", "x1_relu"],
            outputs=["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["sum_nchw", "post_perm"], outputs=["y_nhwc"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x0_nchw", "sum_nchw"],
            outputs=["legacy_cat"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
    ]

    stats = _optimize_transpose_pre_add_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_add_nhwc_chains"] == 1

    add_op = next(op for op in model_ir.operators if str(op.op_type) == "ADD")
    assert list(add_op.inputs) == ["x0_nhwc", "x1_relu"]
    assert list(model_ir.tensors["x1_relu"].shape) == [1, 6, 6, 64]
    assert any(
        str(op.op_type) == "TRANSPOSE" and list(op.outputs) == ["sum_nchw"]
        for op in model_ir.operators
    )


def test_flatbuffer_direct_transpose_pre_concat_nhwc_shared_direct_and_nested_add_optimized() -> None:
    model_ir = ModelIR(name="transpose_pre_concat_nhwc_shared_direct_nested_add_test")
    model_ir.inputs = ["a_nhwc", "b_nhwc", "c_nhwc", "d_nhwc"]
    model_ir.outputs = ["z"]
    model_ir.tensors["a_nhwc"] = TensorIR(
        name="a_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["b_nhwc"] = TensorIR(
        name="b_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["c_nhwc"] = TensorIR(
        name="c_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["d_nhwc"] = TensorIR(
        name="d_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 32],
        shape_signature=[1, 6, 6, 32],
    )
    model_ir.tensors["pre_perm"] = TensorIR(
        name="pre_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["post_perm"] = TensorIR(
        name="post_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["a_nchw"] = TensorIR(
        name="a_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["a_relu"] = TensorIR(
        name="a_relu",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["b_nchw"] = TensorIR(
        name="b_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["c_nchw"] = TensorIR(
        name="c_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["d_nchw"] = TensorIR(
        name="d_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["d_relu"] = TensorIR(
        name="d_relu",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["sum_nchw"] = TensorIR(
        name="sum_nchw",
        dtype="FLOAT32",
        shape=[1, 32, 6, 6],
        shape_signature=[1, 32, 6, 6],
    )
    model_ir.tensors["cat_nchw"] = TensorIR(
        name="cat_nchw",
        dtype="FLOAT32",
        shape=[1, 128, 6, 6],
        shape_signature=[1, 128, 6, 6],
    )
    model_ir.tensors["cat_nhwc"] = TensorIR(
        name="cat_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 6, 128],
        shape_signature=[1, 6, 6, 128],
    )
    model_ir.tensors["z"] = TensorIR(
        name="z",
        dtype="FLOAT32",
        shape=[1, 6, 6, 128],
        shape_signature=[1, 6, 6, 128],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["a_nhwc", "pre_perm"], outputs=["a_nchw"]),
        OperatorIR(op_type="RELU", inputs=["a_nchw"], outputs=["a_relu"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["b_nhwc", "pre_perm"], outputs=["b_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["c_nhwc", "pre_perm"], outputs=["c_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["d_nhwc", "pre_perm"], outputs=["d_nchw"]),
        OperatorIR(op_type="RELU", inputs=["d_nchw"], outputs=["d_relu"]),
        OperatorIR(
            op_type="ADD",
            inputs=["c_nchw", "d_relu"],
            outputs=["sum_nchw"],
            options={"fusedActivationFunction": "NONE"},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["a_relu", "b_nchw", "c_nchw", "sum_nchw"],
            outputs=["cat_nchw"],
            options={"axis": 1, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="TRANSPOSE", inputs=["cat_nchw", "post_perm"], outputs=["cat_nhwc"]),
        OperatorIR(op_type="RELU", inputs=["cat_nhwc"], outputs=["z"]),
    ]

    stats = _optimize_transpose_pre_concat_nhwc_chains(model_ir)
    assert stats["optimized_transpose_pre_concat_nhwc_chains"] == 1

    concat_op = next(op for op in model_ir.operators if str(op.op_type) == "CONCATENATION")
    assert int(concat_op.options.get("axis", -1)) == 3
    assert list(concat_op.inputs) == ["a_relu", "b_nhwc", "c_nhwc", "sum_nchw"]
    assert list(model_ir.tensors["sum_nchw"].shape) == [1, 6, 6, 32]

    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    assert len(transpose_ops) == 1
    remaining_transpose = transpose_ops[0]
    assert list(remaining_transpose.outputs) == ["c_nchw"]
    all_inputs = [str(v) for op in model_ir.operators for v in list(op.inputs)]
    assert "c_nchw" not in all_inputs


def test_flatbuffer_direct_duplicate_transpose_fanout_deduplicates_shared_layout_adapter() -> None:
    model_ir = ModelIR(name="duplicate_transpose_fanout_dedup_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y0", "y1"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 16, 8, 8],
        shape_signature=[1, 16, 8, 8],
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["x_t0"] = TensorIR(
        name="x_t0",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["x_t1"] = TensorIR(
        name="x_t1",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["y0"] = TensorIR(
        name="y0",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.tensors["y1"] = TensorIR(
        name="y1",
        dtype="FLOAT32",
        shape=[1, 8, 8, 16],
        shape_signature=[1, 8, 8, 16],
    )
    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["x_t0"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["x_t1"]),
        OperatorIR(op_type="RELU", inputs=["x_t0"], outputs=["y0"]),
        OperatorIR(op_type="RELU", inputs=["x_t1"], outputs=["y1"]),
    ]

    stats = _optimize_duplicate_transpose_fanout(model_ir)
    assert stats["removed_duplicate_transpose_fanout"] == 1
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 1
    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]
    assert len(relu_ops) == 2
    assert list(relu_ops[0].inputs) == ["x_t0"]
    assert list(relu_ops[1].inputs) == ["x_t0"]


def test_flatbuffer_direct_center_size_offset_terminal_transpose_optimization() -> None:
    model_ir = ModelIR(name="center_size_offset_terminal_transpose_opt_test")
    model_ir.inputs = ["center_nhwc", "size_nhwc", "offset_nhwc"]
    model_ir.outputs = ["center_flat", "gather_size", "gather_offset"]
    model_ir.tensors["center_nhwc"] = TensorIR(
        name="center_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 1],
        shape_signature=[1, 7, 7, 1],
    )
    model_ir.tensors["size_nhwc"] = TensorIR(
        name="size_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 2],
        shape_signature=[1, 7, 7, 2],
    )
    model_ir.tensors["offset_nhwc"] = TensorIR(
        name="offset_nhwc",
        dtype="FLOAT32",
        shape=[1, 7, 7, 2],
        shape_signature=[1, 7, 7, 2],
    )
    model_ir.tensors["perm_nhwc_to_nchw"] = TensorIR(
        name="perm_nhwc_to_nchw",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["center_nchw"] = TensorIR(
        name="center_nchw",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["size_nchw"] = TensorIR(
        name="size_nchw",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["offset_nchw"] = TensorIR(
        name="offset_nchw",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["clip_min"] = TensorIR(
        name="clip_min",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["clip_max"] = TensorIR(
        name="clip_max",
        dtype="FLOAT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1.0], dtype=np.float32),
        is_variable=False,
    )
    model_ir.tensors["center_sig"] = TensorIR(
        name="center_sig",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["center_clip_min_out"] = TensorIR(
        name="center_clip_min_out",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["center_clip_out"] = TensorIR(
        name="center_clip_out",
        dtype="FLOAT32",
        shape=[1, 1, 7, 7],
        shape_signature=[1, 1, 7, 7],
    )
    model_ir.tensors["size_sig"] = TensorIR(
        name="size_sig",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["size_clip_min_out"] = TensorIR(
        name="size_clip_min_out",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["size_clip_out"] = TensorIR(
        name="size_clip_out",
        dtype="FLOAT32",
        shape=[1, 2, 7, 7],
        shape_signature=[1, 2, 7, 7],
    )
    model_ir.tensors["center_flat_shape"] = TensorIR(
        name="center_flat_shape",
        dtype="INT64",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([1, 49], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["size_reshape_shape"] = TensorIR(
        name="size_reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 2, 49], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["offset_reshape_shape"] = TensorIR(
        name="offset_reshape_shape",
        dtype="INT64",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 2, 49], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["center_flat"] = TensorIR(
        name="center_flat",
        dtype="FLOAT32",
        shape=[1, 49],
        shape_signature=[1, 49],
    )
    model_ir.tensors["size_reshaped"] = TensorIR(
        name="size_reshaped",
        dtype="FLOAT32",
        shape=[1, 2, 49],
        shape_signature=[1, 2, 49],
    )
    model_ir.tensors["offset_reshaped"] = TensorIR(
        name="offset_reshaped",
        dtype="FLOAT32",
        shape=[1, 2, 49],
        shape_signature=[1, 2, 49],
    )
    model_ir.tensors["indices_i32"] = TensorIR(
        name="indices_i32",
        dtype="INT32",
        shape=[1, 2, 1],
        shape_signature=[1, 2, 1],
        data=np.asarray([[[5], [5]]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["axis_coord_shape"] = TensorIR(
        name="axis_coord_shape",
        dtype="INT64",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 2, 1, 1], dtype=np.int64),
        is_variable=False,
    )
    model_ir.tensors["axis_coord"] = TensorIR(
        name="axis_coord",
        dtype="INT32",
        shape=[1, 2, 1, 1],
        shape_signature=[1, 2, 1, 1],
    )
    model_ir.tensors["coord0"] = TensorIR(
        name="coord0",
        dtype="INT32",
        shape=[1, 2, 1, 1],
        shape_signature=[1, 2, 1, 1],
        data=np.asarray([[[[0]], [[0]]]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["coord1"] = TensorIR(
        name="coord1",
        dtype="INT32",
        shape=[1, 2, 1, 1],
        shape_signature=[1, 2, 1, 1],
        data=np.asarray([[[[0]], [[1]]]], dtype=np.int32),
        is_variable=False,
    )
    model_ir.tensors["coords_size"] = TensorIR(
        name="coords_size",
        dtype="INT32",
        shape=[1, 2, 1, 3],
        shape_signature=[1, 2, 1, 3],
    )
    model_ir.tensors["coords_offset"] = TensorIR(
        name="coords_offset",
        dtype="INT32",
        shape=[1, 2, 1, 3],
        shape_signature=[1, 2, 1, 3],
    )
    model_ir.tensors["gather_size"] = TensorIR(
        name="gather_size",
        dtype="FLOAT32",
        shape=[1, 2, 1],
        shape_signature=[1, 2, 1],
    )
    model_ir.tensors["gather_offset"] = TensorIR(
        name="gather_offset",
        dtype="FLOAT32",
        shape=[1, 2, 1],
        shape_signature=[1, 2, 1],
    )

    model_ir.operators = [
        OperatorIR(op_type="TRANSPOSE", inputs=["center_nhwc", "perm_nhwc_to_nchw"], outputs=["center_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["size_nhwc", "perm_nhwc_to_nchw"], outputs=["size_nchw"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["offset_nhwc", "perm_nhwc_to_nchw"], outputs=["offset_nchw"]),
        OperatorIR(op_type="LOGISTIC", inputs=["center_nchw"], outputs=["center_sig"]),
        OperatorIR(op_type="LOGISTIC", inputs=["size_nchw"], outputs=["size_sig"]),
        OperatorIR(op_type="MAXIMUM", inputs=["center_sig", "clip_min"], outputs=["center_clip_min_out"]),
        OperatorIR(op_type="MINIMUM", inputs=["center_clip_min_out", "clip_max"], outputs=["center_clip_out"]),
        OperatorIR(op_type="MAXIMUM", inputs=["size_sig", "clip_min"], outputs=["size_clip_min_out"]),
        OperatorIR(op_type="MINIMUM", inputs=["size_clip_min_out", "clip_max"], outputs=["size_clip_out"]),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["center_clip_out", "center_flat_shape"],
            outputs=["center_flat"],
            options={"newShape": [1, 49]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["size_clip_out", "size_reshape_shape"],
            outputs=["size_reshaped"],
            options={"newShape": [1, 2, 49], "onnxRawNewShape": [1, 2, 49]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["offset_nchw", "offset_reshape_shape"],
            outputs=["offset_reshaped"],
            options={"newShape": [1, 2, 49], "onnxRawNewShape": [1, 2, 49]},
        ),
        OperatorIR(
            op_type="RESHAPE",
            inputs=["indices_i32", "axis_coord_shape"],
            outputs=["axis_coord"],
            options={"newShape": [1, 2, 1, 1]},
        ),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["coord0", "coord1", "axis_coord"],
            outputs=["coords_size"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="GATHER_ND", inputs=["size_reshaped", "coords_size"], outputs=["gather_size"]),
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["coord0", "coord1", "axis_coord"],
            outputs=["coords_offset"],
            options={"axis": 3, "fusedActivationFunction": "NONE"},
        ),
        OperatorIR(op_type="GATHER_ND", inputs=["offset_reshaped", "coords_offset"], outputs=["gather_offset"]),
    ]

    stats = _optimize_center_size_offset_terminal_transpose_chains(model_ir)
    assert stats["optimized_center_size_offset_terminal_transpose_chains"] == 1
    assert [str(op.op_type) for op in model_ir.operators].count("TRANSPOSE") == 0

    size_shape = model_ir.tensors["size_reshape_shape"]
    offset_shape = model_ir.tensors["offset_reshape_shape"]
    assert np.asarray(size_shape.data).tolist() == [1, 49, 2]
    assert np.asarray(offset_shape.data).tolist() == [1, 49, 2]

    concat_size = next(op for op in model_ir.operators if list(op.outputs) == ["coords_size"])
    concat_offset = next(op for op in model_ir.operators if list(op.outputs) == ["coords_offset"])
    assert list(concat_size.inputs) == ["coord0", "axis_coord", "coord1"]
    assert list(concat_offset.inputs) == ["coord0", "axis_coord", "coord1"]

    center_log = next(op for op in model_ir.operators if list(op.outputs) == ["center_sig"])
    size_log = next(op for op in model_ir.operators if list(op.outputs) == ["size_sig"])
    offset_reshape = next(op for op in model_ir.operators if list(op.outputs) == ["offset_reshaped"])
    assert list(center_log.inputs) == ["center_nhwc"]
    assert list(size_log.inputs) == ["size_nhwc"]
    assert list(offset_reshape.inputs)[0] == "offset_nhwc"


def test_flatbuffer_direct_qlinear_global_average_pool_lowering() -> None:
    model = _make_qlinear_global_average_pool_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_global_average_pool_lowering_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("AVERAGE_POOL_2D") == 1
    assert op_types.count("MEAN") == 0
    assert op_types.count("DEQUANTIZE") >= 1

    y = model_ir.tensors.get("y")
    assert y is not None
    assert list(y.shape) == [1, 1, 1, 2]
    assert list(y.shape_signature) == [1, 1, 1, 2]


def test_flatbuffer_direct_qlinear_concat_lowering() -> None:
    model = _make_qlinear_concat_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_concat_lowering_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("DEQUANTIZE") >= 1
    assert op_types.count("QUANTIZE") >= 1

    y = model_ir.tensors.get("y")
    assert y is not None
    assert list(y.shape) == [1, 2, 2, 2]
    assert list(y.shape_signature) == [1, 2, 2, 2]


def test_flatbuffer_direct_elides_inverse_transpose_chain_at_generation() -> None:
    model = _make_qlinear_conv_pair_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_generation_elide_test",
        optimize_layout_transpose_chains=False,
    )
    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    transpose_outputs = {str(op.outputs[0]) for op in transpose_ops if len(op.outputs) == 1}

    # PairQConv1 output(NCHW) -> PairQConv2 input(NHWC) bridge is suppressed at generation time.
    assert "PairQConv2_input_nhwc" not in transpose_outputs
    assert len(transpose_ops) == 1


def test_flatbuffer_direct_no_dead_operator_outputs_after_prune() -> None:
    model = _make_qlinear_conv_pair_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="dead_operator_prune_test",
        optimize_layout_transpose_chains=False,
    )

    consumed_tensors = set()
    produced_tensors = []
    for op in model_ir.operators:
        for input_name in op.inputs:
            consumed_tensors.add(str(input_name))
        for output_name in op.outputs:
            produced_tensors.append(str(output_name))

    graph_outputs = set(str(v) for v in model_ir.outputs)
    dead_outputs = [
        name for name in produced_tensors
        if name not in consumed_tensors and name not in graph_outputs
    ]
    assert dead_outputs == []


def test_flatbuffer_direct_serialize_model_prunes_dead_ops() -> None:
    model_ir = ModelIR(name="serialize_prune_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["tmp"] = TensorIR(name="tmp", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["dead_out"] = TensorIR(name="dead_out", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["c"] = TensorIR(
        name="c",
        dtype="FLOAT32",
        shape=[1, 3],
        data=np.array([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[2],
        data=np.array([0, 1], dtype=np.int32),
    )
    model_ir.operators = [
        OperatorIR(op_type="ADD", inputs=["x", "c"], outputs=["tmp"]),
        OperatorIR(op_type="RELU", inputs=["tmp"], outputs=["y"]),
        OperatorIR(op_type="TRANSPOSE", inputs=["tmp", "perm"], outputs=["dead_out"]),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 2
        tensor_names = {
            subgraph.Tensors(i).Name().decode()
            for i in range(subgraph.TensorsLength())
        }
        assert "dead_out" not in tensor_names
        assert "perm" not in tensor_names


def test_flatbuffer_direct_serialize_model_supports_one_hot() -> None:
    model_ir = ModelIR(name="serialize_one_hot_test")
    model_ir.inputs = ["indices"]
    model_ir.outputs = ["y"]
    model_ir.tensors["indices"] = TensorIR(name="indices", dtype="INT32", shape=[2])
    model_ir.tensors["depth"] = TensorIR(
        name="depth",
        dtype="INT32",
        shape=[1],
        data=np.asarray(3, dtype=np.int32),
    )
    model_ir.tensors["on"] = TensorIR(
        name="on",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray(1.0, dtype=np.float32),
    )
    model_ir.tensors["off"] = TensorIR(
        name="off",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray(0.0, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="ONE_HOT",
            inputs=["indices", "depth", "on", "off"],
            outputs=["y"],
            options={"axis": -1},
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1


def test_flatbuffer_direct_serialize_model_supports_pow() -> None:
    model_ir = ModelIR(name="serialize_pow_test")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3])
    model_ir.tensors["exp"] = TensorIR(
        name="exp",
        dtype="FLOAT32",
        shape=[1],
        data=np.asarray(1.25, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="POW",
            inputs=["x", "exp"],
            outputs=["y"],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1


def test_flatbuffer_direct_serialize_model_supports_fill() -> None:
    model_ir = ModelIR(name="serialize_fill_test")
    model_ir.outputs = ["y"]
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[2],
        data=np.asarray([2, 3], dtype=np.int32),
    )
    model_ir.tensors["value"] = TensorIR(
        name="value",
        dtype="FLOAT32",
        shape=[],
        data=np.asarray(0.25, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 3])
    model_ir.operators = [
        OperatorIR(
            op_type="FILL",
            inputs=["shape", "value"],
            outputs=["y"],
        ),
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        schema_tflite = load_schema_module(tmpdir)
        model_bytes = serialize_model(schema_tflite=schema_tflite, model_ir=model_ir)
        model_obj = schema_tflite["Model"].GetRootAsModel(model_bytes, 0)
        subgraph = model_obj.Subgraphs(0)
        assert subgraph.OperatorsLength() == 1


def test_flatbuffer_direct_terminal_quantize_dequantize_optimization() -> None:
    model = _make_terminal_quantize_dequantize_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="terminal_qdq_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert "QUANTIZE" not in op_types
    assert "DEQUANTIZE" not in op_types
    assert op_types.count("RELU") == 1


def test_flatbuffer_direct_terminal_transpose_before_dequantize_sanitization() -> None:
    model = _make_terminal_transpose_dequantize_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="terminal_transpose_before_dequantize_sanitize_test",
    )

    dequantize_ops = [op for op in model_ir.operators if str(op.op_type) == "DEQUANTIZE"]
    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    assert len(dequantize_ops) == 1
    assert len(transpose_ops) == 0

    dq_op = dequantize_ops[0]
    assert dq_op.outputs[0] == "y"


def test_flatbuffer_direct_transpose_quantize_transpose_optimization() -> None:
    model = _make_transpose_quantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1


def test_flatbuffer_direct_transpose_quantize_transpose_fanout_optimization() -> None:
    model = _make_transpose_quantize_transpose_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1
    assert op_types.count("DEQUANTIZE") == 2
    assert op_types.count("ADD") == 1


def test_flatbuffer_direct_transpose_quantize_transpose_preserves_dynamic_batch_signature() -> None:
    model = _make_transpose_quantize_transpose_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_transpose_dynamic_batch_opt_test",
    )
    quant_ops = [op for op in model_ir.operators if str(op.op_type) == "QUANTIZE"]
    assert len(quant_ops) == 1
    out_name = quant_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor is not None
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_transpose_binary_fanout_chain_optimization() -> None:
    model = _make_transpose_binary_fanout_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_binary_fanout_chain_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("ADD") == 2
    assert op_types.count("RELU") == 1
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert any(str(op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU" for op in add_ops)


def test_flatbuffer_direct_gather_singleton_const_indices_scalarized_for_rank_reduced_output() -> None:
    model = _make_gather_singleton_indices_rank_reduced_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="gather_singleton_indices_scalarized_test",
    )
    gather_ops = [op for op in model_ir.operators if str(op.op_type) == "GATHER"]
    assert len(gather_ops) == 1
    gather_op = gather_ops[0]
    gather_indices_name = str(gather_op.inputs[1])
    gather_indices_tensor = model_ir.tensors[gather_indices_name]
    assert gather_indices_name != "indices"
    assert list(gather_indices_tensor.shape) == []
    assert list(gather_indices_tensor.shape_signature) == []

    gather_output_tensor = model_ir.tensors[str(gather_op.outputs[0])]
    assert list(gather_output_tensor.shape_signature) == [1, 3]


def test_flatbuffer_direct_nms_scalar_const_inputs_do_not_emit_squeeze() -> None:
    model = _make_non_max_suppression_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="nms_scalar_const_inputs_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("NON_MAX_SUPPRESSION_V4") == 1
    # boxes/scores squeeze remain, but scalar const inputs are now fed directly.
    assert op_types.count("SQUEEZE") == 2

    nms_op = next(op for op in model_ir.operators if str(op.op_type) == "NON_MAX_SUPPRESSION_V4")
    for scalar_input_name in list(nms_op.inputs)[2:5]:
        scalar_tensor = model_ir.tensors.get(str(scalar_input_name), None)
        assert scalar_tensor is not None
        assert scalar_tensor.data is not None
        assert list(scalar_tensor.shape) == []


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_multiclass_nms_auto_argmax_retry_avoids_custom_lowering() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_non_max_suppression_multiclass_model()
        model_path = _save_model(tmpdir, "nms_multiclass_auto_argmax", model)
        out_dir = os.path.join(tmpdir, "flatbuffer_direct")
        tflite_path = _convert(
            model_path=model_path,
            output_dir=out_dir,
            backend="flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "nms_multiclass_auto_argmax_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["graph_custom_ops"] == []
        nms_reports = [
            node
            for node in report["graph_node_reports"]
            if str(node.get("onnx_op", "")) == "NonMaxSuppression"
        ]
        assert len(nms_reports) == 1
        assert nms_reports[0]["dispatch_mode"] == "builtin"


def test_flatbuffer_direct_conv2d_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_conv_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="conv_dynamic_batch_signature_test",
    )
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1
    out_name = conv_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_conv_stride2_notset_symmetric_pad_uses_explicit_pad() -> None:
    model = _make_conv_stride2_symmetric_pad_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="conv_stride2_notset_symmetric_pad_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("PAD") >= 1
    assert op_types.count("CONV_2D") == 1
    conv_op = next(op for op in model_ir.operators if str(op.op_type) == "CONV_2D")
    assert str(conv_op.options.get("padding", "")) == "VALID"


def test_flatbuffer_direct_qlinear_conv_valid_nonzero_pad_uses_explicit_pad() -> None:
    model = _make_qlinear_conv_valid_explicit_pad_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_valid_nonzero_pad_explicit_pad_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("PAD") >= 1
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1
    assert str(conv_ops[0].options.get("padding", "")) == "VALID"


def test_flatbuffer_direct_maxpool2d_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_maxpool_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="maxpool_dynamic_batch_signature_test",
    )
    maxpool_ops = [op for op in model_ir.operators if str(op.op_type) == "MAX_POOL_2D"]
    assert len(maxpool_ops) == 1
    out_name = maxpool_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_qlinear_conv2d_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_qlinear_conv_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_dynamic_batch_signature_test",
    )
    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1
    out_name = conv_ops[0].outputs[0]
    out_tensor = model_ir.tensors[out_name]
    assert out_tensor.quantization is not None
    assert out_tensor.shape_signature is not None
    assert int(out_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_qlinear_conv_unknown_rank_placeholder_is_promoted_to_rank4() -> None:
    model = _make_qlinear_conv_unknown_rank_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_conv_unknown_rank_signature_test",
    )

    conv_ops = [op for op in model_ir.operators if str(op.op_type) == "CONV_2D"]
    assert len(conv_ops) == 1

    conv_out = model_ir.tensors[conv_ops[0].outputs[0]]
    assert len(list(conv_out.shape)) == 4
    assert int(conv_out.shape[3]) == 4
    assert conv_out.shape_signature is not None
    assert len(list(conv_out.shape_signature)) == 4


def test_flatbuffer_direct_qlinear_global_average_pool_intermediate_preserves_dynamic_batch_signature() -> None:
    model = _make_qlinear_global_average_pool_dynamic_batch_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_global_average_pool_dynamic_batch_signature_test",
    )

    avg_pool_ops = [op for op in model_ir.operators if str(op.op_type) == "AVERAGE_POOL_2D"]
    assert len(avg_pool_ops) == 1
    avg_tensor = model_ir.tensors[avg_pool_ops[0].outputs[0]]
    assert avg_tensor.shape_signature is not None
    assert int(avg_tensor.shape_signature[0]) == -1

    y_tensor = model_ir.tensors["y"]
    assert y_tensor.shape_signature is not None
    assert int(y_tensor.shape_signature[0]) == -1


def test_flatbuffer_direct_transpose_qlinear_global_average_pool_bridge_optimization() -> None:
    model = _make_transpose_qlinear_global_average_pool_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_qlinear_global_average_pool_bridge_opt_test",
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("AVERAGE_POOL_2D") == 1
    assert op_types.count("MEAN") == 0
    assert op_types.count("DEQUANTIZE") == 0
    assert op_types.count("QUANTIZE") == 1

    assert all(
        not (str(op.op_type) == "TRANSPOSE" and len(op.outputs) == 1 and op.outputs[0] == "x_q_nchw")
        for op in model_ir.operators
    )

    avg_pool_ops = [op for op in model_ir.operators if str(op.op_type) == "AVERAGE_POOL_2D"]
    assert len(avg_pool_ops) == 1
    assert avg_pool_ops[0].inputs == ["x_q"]

    producer = None
    for op in model_ir.operators:
        if "y_q" in set(op.outputs):
            producer = op
            break
    assert producer is not None
    assert str(producer.op_type) == "TRANSPOSE"


def test_flatbuffer_direct_qlinear_concat_conv_layout_propagation() -> None:
    model = _make_qlinear_concat_conv_layout_chain_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="qlinear_concat_conv_layout_propagation_test",
    )

    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    concat_axis = int(concat_ops[0].options.get("axis", -1))
    assert concat_axis == 3

    cat_q = model_ir.tensors.get("cat_q")
    assert cat_q is not None
    assert list(cat_q.shape) == [1, 1, 1, 4]

    assert all(
        not (
            str(op.op_type) == "TRANSPOSE"
            and len(op.outputs) == 1
            and op.outputs[0] == "QLCatConv_Conv_input_nhwc"
        )
        for op in model_ir.operators
    )

    # If a QUANTIZE input is already in dequantized flow, avoid redundant Q->DQ before CONCAT.
    assert all(
        not (
            str(op.op_type) == "DEQUANTIZE"
            and len(op.inputs) == 1
            and op.inputs[0] == "pool_q"
        )
        for op in model_ir.operators
    )
    concat_inputs = list(concat_ops[0].inputs)
    assert "QLCatConv_MaxPool_output_nhwc" in concat_inputs


def test_flatbuffer_direct_fuse_add_relu_activation_chain() -> None:
    model = _make_add_relu_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fuse_add_relu_activation_chain_test",
    )
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]

    assert len(add_ops) == 1
    assert len(relu_ops) == 0
    assert str(add_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"
    assert list(add_ops[0].outputs) == ["z"]


def test_flatbuffer_direct_fuse_add_relu_after_transpose_pre_add_rewrite() -> None:
    model = _make_transpose_wrapped_add_relu_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="fuse_add_relu_after_transpose_pre_add_rewrite_test",
    )

    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    relu_ops = [op for op in model_ir.operators if str(op.op_type) == "RELU"]
    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]

    assert len(add_ops) == 1
    assert len(relu_ops) == 1
    assert len(transpose_ops) == 0
    assert str(add_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"
    assert "__skip_add_activation_fuse__" not in add_ops[0].options
    assert list(add_ops[0].outputs) == ["sum_nhwc"]


@pytest.mark.parametrize(
    ("binary_onnx_op", "binary_tflite_op"),
    [
        ("Add", "ADD"),
        ("Sub", "SUB"),
        ("Mul", "MUL"),
        ("Div", "DIV"),
    ],
)
@pytest.mark.parametrize(
    ("activation_onnx_op", "activation_tflite_op", "fused_activation"),
    [
        ("Relu", "RELU", "RELU"),
        ("Relu6", "RELU6", "RELU6"),
        ("Tanh", "TANH", "TANH"),
    ],
)
def test_flatbuffer_direct_fuse_binary_activation_chain(
    binary_onnx_op: str,
    binary_tflite_op: str,
    activation_onnx_op: str,
    activation_tflite_op: str,
    fused_activation: str,
) -> None:
    model = _make_binary_activation_model(binary_onnx_op, activation_onnx_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"fuse_{binary_onnx_op.lower()}_{activation_onnx_op.lower()}_chain_test",
    )

    binary_ops = [op for op in model_ir.operators if str(op.op_type) == binary_tflite_op]
    activation_ops = [op for op in model_ir.operators if str(op.op_type) == activation_tflite_op]

    assert len(binary_ops) == 1
    assert len(activation_ops) == 0
    assert str(binary_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == fused_activation
    assert list(binary_ops[0].outputs) == ["z"]


@pytest.mark.parametrize(
    ("binary_onnx_op", "binary_tflite_op"),
    [
        ("Add", "ADD"),
        ("Sub", "SUB"),
        ("Mul", "MUL"),
        ("Div", "DIV"),
    ],
)
def test_flatbuffer_direct_no_fuse_binary_sigmoid_chain(
    binary_onnx_op: str,
    binary_tflite_op: str,
) -> None:
    model = _make_binary_activation_model(binary_onnx_op, "Sigmoid")
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"no_fuse_{binary_onnx_op.lower()}_sigmoid_chain_test",
    )

    binary_ops = [op for op in model_ir.operators if str(op.op_type) == binary_tflite_op]
    logistic_ops = [op for op in model_ir.operators if str(op.op_type) == "LOGISTIC"]

    assert len(binary_ops) == 1
    assert len(logistic_ops) == 1
    assert str(binary_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "NONE"
    assert list(binary_ops[0].outputs) == ["a0"]
    assert list(logistic_ops[0].outputs) == ["z"]


def test_flatbuffer_direct_lstm_forward_builtin_lowering() -> None:
    model = _make_forward_lstm_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="lstm_forward_builtin_lowering_test",
        allow_custom_ops=False,
    )

    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("UNIDIRECTIONAL_SEQUENCE_LSTM") == 1
    assert op_types.count("BIDIRECTIONAL_SEQUENCE_LSTM") == 0
    assert "SPLIT" not in op_types
    assert op_types.count("CUSTOM") == 0
    assert "CONCATENATION" not in op_types

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [3, 1, 1, 2]
    assert list(y_tensor.shape_signature) == [3, 1, 1, 2]


def test_flatbuffer_direct_transpose_mul_const_add_transpose_optimization() -> None:
    model = _make_transpose_mul_const_add_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mul_const_add_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("MUL") == 1
    assert op_types.count("ADD") == 1
    assert op_types.count("RELU") == 0
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert len(add_ops) == 1
    assert str(add_ops[0].options.get("fusedActivationFunction", "NONE")).upper() == "RELU"
    mul_ops = [op for op in model_ir.operators if str(op.op_type) == "MUL"]
    assert len(mul_ops) == 1
    scale_tensor = model_ir.tensors.get("tmcat_scale_nchw")
    assert scale_tensor is not None
    assert list(scale_tensor.shape) == [1, 1, 1, 4]


def test_flatbuffer_direct_transpose_dequantize_transpose_optimization() -> None:
    model = _make_transpose_dequantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_dequantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("DEQUANTIZE") == 1


def test_flatbuffer_direct_transpose_dequantize_transpose_optimization_fanout_safe() -> None:
    model = _make_transpose_dequantize_transpose_with_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_dequantize_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("DEQUANTIZE") == 2
    assert op_types.count("TRANSPOSE") == 0


def test_flatbuffer_direct_transpose_quantize_dequantize_transpose_optimization() -> None:
    model = _make_transpose_quantize_dequantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_quantize_dequantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("QUANTIZE") == 1
    assert op_types.count("DEQUANTIZE") == 1


def test_flatbuffer_direct_transpose_dequantize_relu6_quantize_transpose_optimization() -> None:
    model = _make_transpose_dequantize_relu6_quantize_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_dequantize_relu6_quantize_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("DEQUANTIZE") == 1
    assert op_types.count("RELU6") == 1
    assert op_types.count("QUANTIZE") == 2


def test_flatbuffer_direct_transpose_relu6_transpose_fanout_optimization() -> None:
    model = _make_transpose_relu6_transpose_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("RELU6") == 1
    assert op_types.count("RELU") == 1
    assert op_types.count("NEG") == 1


def test_flatbuffer_direct_transpose_relu6_transpose_mixed_fanout_optimization() -> None:
    model = _make_transpose_relu6_transpose_mixed_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_transpose_mixed_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("RELU6") == 1
    assert op_types.count("RELU") == 1
    assert op_types.count("ADD") == 1


def test_flatbuffer_direct_transpose_relu6_binary_transpose_fanout_optimization() -> None:
    model = _make_transpose_relu6_binary_transpose_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_binary_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("RELU6") == 1
    assert op_types.count("ADD") == 1
    assert op_types.count("RELU") == 1
    assert op_types.count("NEG") == 1


def test_flatbuffer_direct_transpose_relu6_binary_mixed_fanout_optimization() -> None:
    model = _make_transpose_relu6_binary_mixed_fanout_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_relu6_binary_mixed_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("RELU6") == 1
    assert op_types.count("ADD") == 2
    assert op_types.count("RELU") == 1


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
def test_flatbuffer_direct_transpose_binary_transpose_optimization(binary_op: str) -> None:
    model = _make_transpose_binary_transpose_model(binary_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_{binary_op.lower()}_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    expected_binary = binary_op.upper()
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count(expected_binary) == 1


def test_flatbuffer_direct_transpose_transpose_softmax_compose_optimization() -> None:
    model = _make_transpose_transpose_softmax_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_transpose_softmax_compose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 1


def test_flatbuffer_direct_softmax_terminal_transpose_preserved() -> None:
    model = _make_softmax_terminal_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="softmax_terminal_transpose_preserve_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 1
    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [1, 4, 3, 2]


def test_flatbuffer_direct_softmax_terminal_double_pretranspose_canonicalization() -> None:
    model = _make_softmax_terminal_double_pretranspose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="softmax_terminal_double_pretranspose_canon_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("SOFTMAX") == 1
    assert op_types.count("TRANSPOSE") == 0

    softmax_op = next(op for op in model_ir.operators if str(op.op_type) == "SOFTMAX")
    assert list(softmax_op.inputs) == ["x"]
    assert list(softmax_op.outputs) == ["y"]

    y_tensor = model_ir.tensors.get("y")
    assert y_tensor is not None
    assert list(y_tensor.shape) == [1, 2, 3, 4]


def test_flatbuffer_direct_transpose_swish_transpose_optimization() -> None:
    model = _make_transpose_swish_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_swish_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("LOGISTIC") == 1
    assert op_types.count("MUL") == 1
    assert op_types.count("RELU") == 0


def test_flatbuffer_direct_transpose_gelu_transpose_optimization() -> None:
    model = _make_transpose_gelu_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_gelu_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("GELU") == 1
    gelu_op = next(op for op in model_ir.operators if str(op.op_type) == "GELU")
    assert [str(v) for v in list(gelu_op.inputs)] == ["x"]
    assert [str(v) for v in list(gelu_op.outputs)] == ["z"]


def test_flatbuffer_direct_transpose_gelu_tanh_transpose_optimization() -> None:
    model = _make_transpose_gelu_tanh_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_gelu_tanh_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("TANH") == 0
    assert op_types.count("MUL") == 6
    assert op_types.count("ADD") == 2
    assert op_types.count("RELU") == 0


def test_flatbuffer_direct_transpose_add_reshape_transpose_suffix_optimization() -> None:
    model = _make_transpose_add_reshape_transpose_with_legacy_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir_raw = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_add_reshape_suffix_raw_test",
        optimize_layout_transpose_chains=False,
    )
    model_ir_opt = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_add_reshape_suffix_opt_test",
    )

    raw_transpose_count = sum(1 for op in model_ir_raw.operators if str(op.op_type) == "TRANSPOSE")
    opt_transpose_count = sum(1 for op in model_ir_opt.operators if str(op.op_type) == "TRANSPOSE")
    assert opt_transpose_count < raw_transpose_count

    for op in model_ir_opt.operators:
        if str(op.op_type) != "TRANSPOSE" or len(op.inputs) < 2:
            continue
        perm_tensor = model_ir_opt.tensors.get(str(op.inputs[1]))
        if perm_tensor is None or perm_tensor.data is None:
            continue
        perm = [int(v) for v in np.asarray(perm_tensor.data).reshape(-1).tolist()]
        assert perm != [0, 2, 1]

    producer_of_z = None
    for op in model_ir_opt.operators:
        if "z" in [str(v) for v in op.outputs]:
            producer_of_z = op
            break
    assert producer_of_z is not None
    assert str(producer_of_z.op_type) == "RESHAPE"


def test_flatbuffer_direct_transpose_mulconst_add_reshape_transpose_suffix_optimization() -> None:
    model = _make_transpose_mulconst_add_reshape_transpose_with_legacy_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir_raw = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mulconst_add_reshape_suffix_raw_test",
        optimize_layout_transpose_chains=False,
    )
    model_ir_opt = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_mulconst_add_reshape_suffix_opt_test",
    )

    raw_transpose_count = sum(1 for op in model_ir_raw.operators if str(op.op_type) == "TRANSPOSE")
    opt_transpose_count = sum(1 for op in model_ir_opt.operators if str(op.op_type) == "TRANSPOSE")
    assert opt_transpose_count < raw_transpose_count

    output_names_by_op = {
        str(output_name)
        for op in model_ir_opt.operators
        if str(op.op_type) == "TRANSPOSE"
        for output_name in list(op.outputs)
    }
    assert "y_nchw" not in output_names_by_op

    producer_of_z_mid = None
    for op in model_ir_opt.operators:
        if "z_mid" in [str(v) for v in op.outputs]:
            producer_of_z_mid = op
            break
    assert producer_of_z_mid is not None
    assert str(producer_of_z_mid.op_type) == "RESHAPE"


def test_flatbuffer_direct_transpose_nested_add_with_shared_skip_optimization() -> None:
    model = _make_transpose_nested_add_with_shared_skip_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir_opt = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_nested_add_with_shared_skip_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir_opt.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("MUL") == 1
    add_ops = [op for op in model_ir_opt.operators if str(op.op_type) == "ADD"]
    assert len(add_ops) == 2
    producer_of_z = next(
        (op for op in model_ir_opt.operators if "z" in [str(v) for v in op.outputs]),
        None,
    )
    assert producer_of_z is not None
    assert str(producer_of_z.op_type) == "ADD"


def test_flatbuffer_direct_transpose_swish_add_concat_transpose_optimization() -> None:
    model = _make_transpose_swish_add_concat_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_swish_add_concat_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("LOGISTIC") == 3
    assert op_types.count("MUL") == 3
    assert op_types.count("ADD") == 1
    assert op_types.count("CONCATENATION") == 1
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    assert int(concat_ops[0].options.get("axis", -1)) == 3


def test_flatbuffer_direct_transpose_swish_add_transpose_cascade_optimization() -> None:
    model = _make_transpose_swish_add_transpose_cascade_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_swish_add_transpose_cascade_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("ADD") == 2
    assert op_types.count("LOGISTIC") == 3
    assert op_types.count("MUL") == 3
    assert op_types.count("RELU") == 0
    add_ops = [op for op in model_ir.operators if str(op.op_type) == "ADD"]
    assert any(str(op.options.get("fusedActivationFunction", "NONE")).upper() == "RELU" for op in add_ops)


def test_flatbuffer_direct_transpose_slice_concat_transpose_optimization() -> None:
    model = _make_transpose_slice_concat_transpose_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_slice_concat_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 0
    assert op_types.count("STRIDED_SLICE") == 4
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("RELU") == 1
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    assert int(concat_ops[0].options.get("axis", -1)) == 3


def test_flatbuffer_direct_transpose_logistic_concat_reshape_optimization() -> None:
    model = _make_transpose_logistic_concat_reshape_model()
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name="transpose_logistic_concat_reshape_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count("LOGISTIC") == 2
    assert op_types.count("CONCATENATION") == 1
    assert op_types.count("RESHAPE") == 1
    concat_ops = [op for op in model_ir.operators if str(op.op_type) == "CONCATENATION"]
    assert len(concat_ops) == 1
    assert int(concat_ops[0].options.get("axis", -1)) == 3


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
def test_flatbuffer_direct_transpose_binary_transpose_fanout_optimization(binary_op: str) -> None:
    model = _make_transpose_binary_transpose_fanout_model(binary_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_{binary_op.lower()}_transpose_fanout_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    expected_binary = binary_op.upper()
    assert op_types.count("TRANSPOSE") == 1
    expected_binary_count = 2 if expected_binary == "ADD" else 1
    assert op_types.count(expected_binary) == expected_binary_count
    expected_add_count = 2 if expected_binary == "ADD" else 1
    assert op_types.count("ADD") == expected_add_count
    assert op_types.count("RELU") == 1


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
def test_flatbuffer_direct_transpose_binary_no_post_transpose_optimization(binary_op: str) -> None:
    model = _make_transpose_binary_no_post_transpose_model(binary_op)
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_{binary_op.lower()}_no_post_transpose_opt_test",
    )
    op_types = [str(op.op_type) for op in model_ir.operators]
    expected_binary = binary_op.upper()
    assert op_types.count("TRANSPOSE") == 1
    assert op_types.count(expected_binary) == 1
    assert op_types.count("RELU") == 1


@pytest.mark.parametrize("binary_op", ["Add", "Sub", "Mul", "Div"])
@pytest.mark.parametrize("transpose_on_lhs", [True, False])
def test_flatbuffer_direct_transpose_binary_single_side_transpose_optimization(
    binary_op: str,
    transpose_on_lhs: bool,
) -> None:
    model = _make_transpose_binary_single_side_transpose_model(
        binary_op,
        transpose_on_lhs=transpose_on_lhs,
    )
    register_default_preprocess_rules()
    preprocessed_model, _ = run_preprocess_pipeline(onnx_graph=model)
    model_ir = lower_onnx_to_ir(
        onnx_graph=preprocessed_model,
        output_file_name=f"transpose_single_side_{binary_op.lower()}_transpose_opt_test",
    )

    expected_binary = binary_op.upper()
    op_types = [str(op.op_type) for op in model_ir.operators]
    assert op_types.count(expected_binary) == 1
    assert op_types.count("TRANSPOSE") == 1

    binary_ops = [op for op in model_ir.operators if str(op.op_type) == expected_binary]
    assert len(binary_ops) == 1
    binary_ir = binary_ops[0]
    assert binary_ir.outputs == ["z"]

    transpose_ops = [op for op in model_ir.operators if str(op.op_type) == "TRANSPOSE"]
    assert len(transpose_ops) == 1
    assert transpose_ops[0].outputs[0] in set(binary_ir.inputs)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_clip_relu6_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_clip_relu6_model()
        model_path = _save_model(tmpdir, "clip_relu6", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[-1.0, 3.0, 10.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            y,
            np.array([[0.0, 3.0, 6.0]], dtype=np.float32),
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_prelu_emits_builtin_prelu() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_prelu_model()
        model_path = _save_model(tmpdir, "prelu_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")
        op_names = _collect_builtin_op_names(tflite_path)
        assert "PRELU" in op_names


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_candidate_disabled_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_disabled", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "einsum_custom_disabled_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert "Einsum" in report["graph_custom_ops"]
        assert report["graph_summary"]["custom_lowered_nodes"] == 1
        assert any(
            node["onnx_op"] == "Einsum" and node["dispatch_mode"] == "custom"
            for node in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_enabled_generates_custom_code() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_enabled", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
        )
        assert os.path.isfile(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_EINSUM" in custom_codes


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_not_in_allowlist_fails() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_allowlist_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(NotImplementedError, match="custom_op_not_in_allowlist"):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                flatbuffer_direct_allow_custom_ops=True,
                flatbuffer_direct_custom_op_allowlist=["TopK"],
            )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_custom_op_coverage_report() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_custom_model()
        model_path = _save_model(tmpdir, "einsum_custom_report", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
        )
        report_path = os.path.join(out_dir, "einsum_custom_report_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["custom_op_policy"]["allow_custom_ops"] is True
        assert "Einsum" in report["graph_custom_ops"]
        assert report["graph_summary"]["custom_lowered_nodes"] == 1
        assert (
            report["custom_op_policy"]["candidate_count_excluding_builtin_supported"]
            == report["custom_op_policy"]["candidate_count"]
            - len(report["custom_op_policy"]["candidate_ops_now_builtin_supported"])
        )
        assert "Einsum" in report["custom_op_policy"]["candidate_ops_now_builtin_supported"]
        assert "Einsum" in report["custom_op_policy"]["allowlist_builtin_supported_ops"]
        assert "Einsum" in report["custom_op_policy"]["allowlist_custom_candidate_ops"]
        assert report["custom_op_policy"]["allowlist_unknown_ops"] == []
        node_reports = report["graph_node_reports"]
        assert any(
            r["onnx_op"] == "Einsum" and r["dispatch_mode"] == "custom"
            for r in node_reports
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_einsum_builtin_preferred_over_custom() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_fc_const_model()
        model_path = _save_model(tmpdir, "einsum_builtin_preferred", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_EINSUM" not in custom_codes

        report_path = os.path.join(out_dir, "einsum_builtin_preferred_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert "Einsum" not in report["graph_custom_ops"]
        assert any(
            r["onnx_op"] == "Einsum" and r["dispatch_mode"] == "builtin"
            for r in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_einsum_nonconst_rhs_builtin() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_einsum_model()
        model_path = _save_model(tmpdir, "einsum_nonconst_builtin", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_allow_custom_ops=True,
            flatbuffer_direct_custom_op_allowlist=["Einsum"],
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)
        custom_codes = _collect_custom_codes(tflite_path)
        assert "ONNX_EINSUM" not in custom_codes

        report_path = os.path.join(out_dir, "einsum_nonconst_builtin_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert "Einsum" not in report["graph_custom_ops"]
        assert any(
            r["onnx_op"] == "Einsum" and r["dispatch_mode"] == "builtin"
            for r in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_fallback_to_tf_converter_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_elu_model()
        model_path = _save_model(tmpdir, "elu_fallback", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            flatbuffer_direct_fallback_to_tf_converter=True,
            report_op_coverage=True,
        )

        assert os.path.isfile(os.path.join(out_dir, "elu_fallback_float32.tflite"))
        assert os.path.isfile(os.path.join(out_dir, "elu_fallback_float16.tflite"))
        report_path = os.path.join(out_dir, "elu_fallback_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["unsupported_nodes"] == 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integration_quant_eval_coverage_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_integ", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
            output_integer_quantized_tflite=True,
            eval_with_onnx=True,
            eval_num_samples=2,
            eval_target_tflite="full_integer_quant",
            eval_compare_mode="dequant",
            report_op_coverage=True,
        )

        expected_files = [
            "gemm_integ_float32.tflite",
            "gemm_integ_float16.tflite",
            "gemm_integ_dynamic_range_quant.tflite",
            "gemm_integ_integer_quant.tflite",
            "gemm_integ_full_integer_quant.tflite",
            "gemm_integ_accuracy_report.json",
            "gemm_integ_op_coverage_report.json",
        ]
        for name in expected_files:
            assert os.path.isfile(os.path.join(out_dir, name))

        with open(os.path.join(out_dir, "gemm_integ_accuracy_report.json"), "r", encoding="utf-8") as f:
            acc = json.load(f)
        assert acc["num_samples"] == 2

        with open(os.path.join(out_dir, "gemm_integ_op_coverage_report.json"), "r", encoding="utf-8") as f:
            cov = json.load(f)
        assert "schema_policy_counts" in cov
        assert cov["schema_unresolved_ops"] == []


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integration_split_eval_coverage_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_integ", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
            eval_split_models=True,
            eval_num_samples=2,
            report_op_coverage=True,
        )

        expected_files = [
            "add_chain_integ_split_plan.json",
            "add_chain_integ_split_manifest.json",
            "add_chain_integ_split_accuracy_report.json",
            "add_chain_integ_op_coverage_report.json",
        ]
        for name in expected_files:
            assert os.path.isfile(os.path.join(out_dir, name))

        with open(os.path.join(out_dir, "add_chain_integ_split_accuracy_report.json"), "r", encoding="utf-8") as f:
            split_report = json.load(f)
        assert split_report["reference_mode"] == "unsplit_tflite"


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_gather_int32_dtype_boundary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gather_int32_model()
        model_path = _save_model(tmpdir, "gather_int32", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(model_path, out_dir, "flatbuffer_direct")

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array(
            [
                [10, 20, 30, 40],
                [1, 2, 3, 4],
            ],
            dtype=np.int32,
        )
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_array_equal(
            y,
            np.array(
                [
                    [40, 20],
                    [4, 2],
                ],
                dtype=np.int32,
            ),
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_quantized_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_dq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
        )
        tflite_path = os.path.join(out_dir, "gemm_dq_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        assert y.shape == (1, 3)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_quantized_add_const_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_const_model()
        model_path = _save_model(tmpdir, "add_const_dq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
        )
        tflite_path = os.path.join(out_dir, "add_const_dq_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        x = np.array([[1.0, 1.0, 1.0]], dtype=np.float32)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])
        np.testing.assert_allclose(
            y,
            np.array([[2.0, 3.0, 4.0]], dtype=np.float32),
            rtol=0.0,
            atol=5e-2,
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
@pytest.mark.parametrize(
    "quant_type, expected_multi_scale",
    [
        ("per-channel", True),
        ("per-tensor", False),
    ],
)
def test_flatbuffer_direct_dynamic_range_quantized_fc_quant_type(
    quant_type: str,
    expected_multi_scale: bool,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, f"gemm_{quant_type}", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_dynamic_range_quantized_tflite=True,
            quant_type=quant_type,
        )
        tflite_path = os.path.join(
            out_dir,
            f"gemm_{quant_type}_dynamic_range_quant.tflite",
        )
        assert os.path.isfile(tflite_path)

        scale_lengths = _collect_int8_quant_scale_lengths(tflite_path)
        assert len(scale_lengths) > 0
        if expected_multi_scale:
            assert any(length > 1 for length in scale_lengths)
        else:
            assert all(length == 1 for length in scale_lengths)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integer_quantized_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_iq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
            input_quant_dtype="int8",
            output_quant_dtype="int8",
        )

        integer_tflite = os.path.join(out_dir, "gemm_iq_integer_quant.tflite")
        full_integer_tflite = os.path.join(out_dir, "gemm_iq_full_integer_quant.tflite")
        integer_i16_tflite = os.path.join(out_dir, "gemm_iq_integer_quant_with_int16_act.tflite")
        full_integer_i16_tflite = os.path.join(out_dir, "gemm_iq_full_integer_quant_with_int16_act.tflite")
        assert os.path.isfile(integer_tflite)
        assert os.path.isfile(full_integer_tflite)
        assert os.path.isfile(integer_i16_tflite)
        assert os.path.isfile(full_integer_i16_tflite)

        # integer_quant: float input/output path
        interpreter = Interpreter(model_path=integer_tflite)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(in_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_details[0]["index"])
        assert y.shape == (1, 3)

        # full_integer_quant: quantized io path
        interpreter2 = Interpreter(model_path=full_integer_tflite)
        interpreter2.allocate_tensors()
        in2 = interpreter2.get_input_details()
        out2 = interpreter2.get_output_details()
        assert in2[0]["dtype"] == np.int8
        assert out2[0]["dtype"] == np.int8
        xq = np.zeros(in2[0]["shape"], dtype=np.int8)
        interpreter2.set_tensor(in2[0]["index"], xq)
        interpreter2.invoke()
        yq = interpreter2.get_tensor(out2[0]["index"])
        assert yq.dtype == np.int8

        # integer_quant_with_int16_act: float input/output path
        interpreter3 = Interpreter(model_path=integer_i16_tflite)
        interpreter3.allocate_tensors()
        in3 = interpreter3.get_input_details()
        out3 = interpreter3.get_output_details()
        x3 = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter3.set_tensor(in3[0]["index"], x3)
        interpreter3.invoke()
        y3 = interpreter3.get_tensor(out3[0]["index"])
        assert y3.shape == (1, 3)

        # full_integer_quant_with_int16_act: int16 input/output path
        interpreter4 = Interpreter(model_path=full_integer_i16_tflite)
        interpreter4.allocate_tensors()
        in4 = interpreter4.get_input_details()
        out4 = interpreter4.get_output_details()
        assert in4[0]["dtype"] == np.int16
        assert out4[0]["dtype"] == np.int16
        x4 = np.zeros(in4[0]["shape"], dtype=np.int16)
        interpreter4.set_tensor(in4[0]["index"], x4)
        interpreter4.invoke()
        y4 = interpreter4.get_tensor(out4[0]["index"])
        assert y4.dtype == np.int16


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_integer_quantized_reduce_compatibility() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_reduce_model()
        model_path = _save_model(tmpdir, "gemm_reduce_iq", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            quant_type="per-channel",
        )
        tflite_path = os.path.join(out_dir, "gemm_reduce_iq_integer_quant.tflite")
        assert os.path.isfile(tflite_path)

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()
        x = np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        interpreter.set_tensor(in_details[0]["index"], x)
        interpreter.invoke()
        y = interpreter.get_tensor(out_details[0]["index"])
        assert y.shape == (1, 1)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_percentile_calibration_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_pct", model)
        out_dir = os.path.join(tmpdir, "out")
        with _temporary_env(
            {
                "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_METHOD": "percentile",
                "ONNX2TF_FLATBUFFER_DIRECT_CALIBRATION_PERCENTILE": "99.0",
            }
        ):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                output_dynamic_range_quantized_tflite=True,
            )
        tflite_path = os.path.join(out_dir, "gemm_pct_dynamic_range_quant.tflite")
        assert os.path.isfile(tflite_path)


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_dynamic_range_threshold_control() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_th", model)
        out_dir = os.path.join(tmpdir, "out")
        with _temporary_env(
            {
                "ONNX2TF_FLATBUFFER_DIRECT_QUANT_MIN_ABS_MAX": "1000.0",
            }
        ):
            with pytest.raises(NotImplementedError):
                _convert(
                    model_path,
                    out_dir,
                    "flatbuffer_direct",
                    output_dynamic_range_quantized_tflite=True,
                )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add_eval", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            eval_with_onnx=True,
            eval_num_samples=3,
        )

        report_path = os.path.join(out_dir, "add_eval_accuracy_report.json")
        assert os.path.isfile(report_path)

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["schema_version"] == 1
        assert report["num_samples"] == 3
        assert report["seed"] == 0
        assert report["inputs_source"] == "seeded_random"
        assert report["compare_mode"] == "raw"
        assert report["evaluation_pass"] is True
        assert report["allclose_summary"]["pass"] is True
        assert "overall_metrics" in report
        assert "per_output_metrics" in report
        assert "z" in report["per_output_metrics"]
        assert report["overall_metrics"]["max_abs"] <= 1e-6


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_quant_dequant_mode() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_eval_q", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            output_integer_quantized_tflite=True,
            eval_with_onnx=True,
            eval_num_samples=3,
            eval_target_tflite="full_integer_quant",
            eval_compare_mode="dequant",
        )

        report_path = os.path.join(out_dir, "gemm_eval_q_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["compare_mode"] == "dequant"
        assert report["has_quantized_outputs"] is True
        assert "metric_threshold_judgement" in report


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_accuracy_report_fail_on_threshold() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_eval_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        with pytest.raises(RuntimeError):
            _convert(
                model_path,
                out_dir,
                "flatbuffer_direct",
                output_integer_quantized_tflite=True,
                eval_with_onnx=True,
                eval_num_samples=2,
                eval_target_tflite="full_integer_quant",
                eval_compare_mode="raw",
                eval_fail_on_threshold=True,
            )
        report_path = os.path.join(out_dir, "gemm_eval_fail_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["evaluation_pass"] is False


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_plan_report_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_split_plan", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=9_000_000,
        )
        report_path = os.path.join(out_dir, "gemm_split_plan_split_plan.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["schema_version"] == 1
        assert report["plan_valid"] is True
        assert report["total_estimated_bytes"] > 0
        assert len(report["partitions"]) >= 1


def test_parse_auto_split_size_to_bytes_units() -> None:
    from onnx2tf.onnx2tf import _parse_size_to_bytes

    assert _parse_size_to_bytes("1KB", param_name="x") == 1_024
    assert _parse_size_to_bytes("2mb", param_name="x") == 2 * 1_048_576
    assert _parse_size_to_bytes("1.5GB", param_name="x") == int(1.5 * 1_073_741_824)
    assert _parse_size_to_bytes("256", default_unit="MB", param_name="x") == 256 * 1_048_576


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_plan_uses_auto_split_max_size_when_specified() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_gemm_model()
        model_path = _save_model(tmpdir, "gemm_split_size_opt", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            auto_split_max_size="256KB",
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=9_000_000,
        )
        report_path = os.path.join(out_dir, "gemm_split_size_opt_split_plan.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["target_max_bytes"] == 256 * 1024
        assert report["hard_max_bytes"] >= report["target_max_bytes"]


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_manifest_and_partition_outputs() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_split", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
        )

        manifest_path = os.path.join(out_dir, "add_chain_split_split_manifest.json")
        assert os.path.isfile(manifest_path)
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        assert manifest["schema_version"] == 1
        assert manifest["base_model"] == "add_chain_split.tflite"
        assert manifest["target_max_bytes"] == 1
        assert len(manifest["partitions"]) >= 1
        assert "edges" in manifest

        part_files = sorted(
            glob.glob(os.path.join(out_dir, "add_chain_split_[0-9][0-9][0-9][0-9].tflite"))
        )
        assert len(part_files) >= 1
        for part_file in part_files:
            interpreter = Interpreter(model_path=part_file)
            interpreter.allocate_tensors()


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_split_accuracy_report_with_unsplit_reference() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_eval_split", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
            eval_split_models=True,
            eval_split_reference="unsplit_tflite",
            eval_num_samples=3,
        )
        report_path = os.path.join(out_dir, "add_chain_eval_split_split_accuracy_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["reference_mode"] == "unsplit_tflite"
        assert report["evaluation_pass"] is True
        assert report["allclose_summary"]["pass"] is True


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_split_accuracy_report_fail_on_threshold() -> None:
    from onnx2tf.tflite_builder.split_accuracy_evaluator import evaluate_split_manifest_outputs

    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_chain_model()
        model_path = _save_model(tmpdir, "add_chain_eval_split_fail", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            auto_split_tflite_by_size=True,
            tflite_split_max_bytes=10_000_000,
            tflite_split_target_bytes=1,
        )
        split_manifest_path = os.path.join(out_dir, "add_chain_eval_split_fail_split_manifest.json")
        reference_tflite_path = os.path.join(out_dir, "add_chain_eval_split_fail_float32.tflite")
        onnx_graph = onnx.load(model_path)
        report_path = os.path.join(out_dir, "add_chain_eval_split_fail_split_accuracy_report.json")
        with pytest.raises(RuntimeError):
            evaluate_split_manifest_outputs(
                onnx_graph=onnx_graph,
                split_manifest_path=split_manifest_path,
                reference_mode="unsplit_tflite",
                reference_tflite_path=reference_tflite_path,
                output_report_path=report_path,
                num_samples=2,
                fail_on_threshold=True,
                metric_thresholds={
                    "max_abs": 0.0,
                    "mean_abs": 0.0,
                    "rmse": 0.0,
                    "cosine_similarity": 1.1,
                },
            )
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["evaluation_pass"] is False


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_op_coverage_report_generation() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_add_model()
        model_path = _save_model(tmpdir, "add_cov", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "add_cov_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["schema_version"] == 1
        assert report["conversion_error"] is None
        assert "Add" in report["graph_ops"]
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert report["graph_summary"]["coverage_ratio"] == 1.0
        assert "preprocess_report" in report
        assert report["preprocess_report"]["schema_version"] == 1
        assert report["preprocess_report"]["summary"]["executed_rule_count"] >= 0


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_op_coverage_report_on_unsupported_op() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_elu_model()
        model_path = _save_model(tmpdir, "elu_cov", model)
        out_dir = os.path.join(tmpdir, "out")
        _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        report_path = os.path.join(out_dir, "elu_cov_op_coverage_report.json")
        assert os.path.isfile(report_path)
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["conversion_error"] is None
        assert report["graph_summary"]["unsupported_nodes"] == 0
        assert any(
            node["onnx_op"] == "Elu" and node["dispatch_mode"] == "builtin"
            for node in report["graph_node_reports"]
        )


@pytest.mark.skipif(not _requires_flatbuffer_tools(), reason="flatbuffer_direct requires flatc and curl")
def test_flatbuffer_direct_erf_tile_scatternd_builtin_smoke() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        model = _make_erf_tile_scatternd_model()
        model_path = _save_model(tmpdir, "erf_tile_scatternd", model)
        out_dir = os.path.join(tmpdir, "out")
        tflite_path = _convert(
            model_path,
            out_dir,
            "flatbuffer_direct",
            report_op_coverage=True,
        )
        assert os.path.isfile(tflite_path)

        report_path = os.path.join(out_dir, "erf_tile_scatternd_op_coverage_report.json")
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        assert report["graph_custom_ops"] == []
        assert report["graph_summary"]["custom_lowered_nodes"] == 0
        assert report["graph_summary"]["unsupported_nodes"] == 0

        interpreter = Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        by_name = {d["name"]: d for d in input_details}

        x = np.asarray(
            [
                [-0.5, 0.0, 0.5],
                [1.0, -1.0, 0.25],
            ],
            dtype=np.float32,
        )
        updates = np.asarray([10.0, -20.0], dtype=np.float32)
        interpreter.set_tensor(by_name["x"]["index"], x)
        interpreter.set_tensor(by_name["updates"]["index"], updates)
        interpreter.invoke()
        y = interpreter.get_tensor(output_details[0]["index"])

        expected = np.tile(np.vectorize(math.erf)(x).astype(np.float32), (1, 2))
        expected[0, 1] = updates[0]
        expected[1, 4] = updates[1]
        np.testing.assert_allclose(y, expected, rtol=1e-4, atol=1e-4)
