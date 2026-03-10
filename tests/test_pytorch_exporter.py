import builtins
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np
import onnx
import pytest
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper

import onnx2tf
from onnx2tf.tflite_builder import export_tflite_model_flatbuffer_direct
from onnx2tf.tflite_builder.ir import (
    ModelIR,
    OperatorIR,
    TensorIR,
    infer_model_ir_logical_layouts,
)
from onnx2tf.tflite_builder.lower_from_onnx2tf import lower_onnx_to_ir
from onnx2tf.tflite_builder.model_writer import write_model_file
from onnx2tf.tflite_builder.pytorch_package_runtime import (
    _infer_spatial_shape_for_transposed_conv2d,
    load_generated_model_package,
)
from onnx2tf.tflite_builder.pytorch_accuracy_evaluator import (
    evaluate_pytorch_package_outputs,
    evaluate_tflite_pytorch_package_outputs,
    smoke_test_pytorch_package_inference,
)
from onnx2tf.tflite_builder.pytorch_exporter import (
    ModelIRPyTorchExportError,
    _export_runtime_wrapper_package_from_model_ir,
    _reject_residual_layout_transposes,
    _remove_redundant_layout_transposes,
    _should_prefer_tflite_backed_package,
    export_pytorch_package_from_model_ir,
    normalize_model_ir_for_pytorch_channel_first,
)
from onnx2tf.tflite_builder.schema_loader import load_schema_module


def _import_generated_package(package_path: str):
    parent = str(Path(package_path).parent)
    package_name = Path(package_path).name
    sys.path.insert(0, parent)
    try:
        if package_name in sys.modules:
            del sys.modules[package_name]
        return importlib.import_module(package_name)
    finally:
        if sys.path[0] == parent:
            sys.path.pop(0)


def _make_add_model() -> onnx.ModelProto:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 3])
    z = helper.make_tensor_value_info("z", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Add", ["x", "y"], ["z"], name="AddNode")
    graph = helper.make_graph([node], "add_graph", [x, y], [z])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    return model


def _make_add_model_ir() -> ModelIR:
    model_ir = ModelIR(name="add_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.operators.append(
        OperatorIR(op_type="ADD", inputs=["x", "y"], outputs=["z"], options={})
    )
    return model_ir


def _make_add_relu_model_ir() -> ModelIR:
    model_ir = ModelIR(name="add_relu_model_ir")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["sum"] = TensorIR(name="sum", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.operators.append(
        OperatorIR(op_type="ADD", inputs=["x", "y"], outputs=["sum"], options={})
    )
    model_ir.operators.append(
        OperatorIR(op_type="RELU", inputs=["sum"], outputs=["z"], options={})
    )
    return model_ir


def _make_abs_model_ir() -> ModelIR:
    model_ir = ModelIR(name="abs_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.operators.append(
        OperatorIR(op_type="ABS", inputs=["x"], outputs=["y"], options={})
    )
    return model_ir


def _make_relu_0_to_1_model_ir() -> ModelIR:
    model_ir = ModelIR(name="relu_0_to_1_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.operators.append(
        OperatorIR(op_type="RELU_0_TO_1", inputs=["x"], outputs=["y"], options={})
    )
    return model_ir


def _make_gather_model_ir() -> ModelIR:
    model_ir = ModelIR(name="gather_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4], shape_signature=[1, 4])
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([0, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 2], shape_signature=[1, 2])
    model_ir.operators.append(
        OperatorIR(
            op_type="GATHER",
            inputs=["x", "indices"],
            outputs=["y"],
            options={"axis": 1},
        )
    )
    return model_ir


def _make_concat_scalar_model_ir() -> ModelIR:
    model_ir = ModelIR(name="concat_scalar_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1.0, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2], shape_signature=[2])
    model_ir.operators.append(
        OperatorIR(
            op_type="CONCATENATION",
            inputs=["x", "one"],
            outputs=["y"],
            options={"axis": 0, "fusedActivationFunction": "NONE"},
        )
    )
    return model_ir


def _make_gather_concat_scalar_model_ir() -> ModelIR:
    model_ir = ModelIR(name="gather_concat_scalar_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[2], shape_signature=[2])
    model_ir.tensors["indices"] = TensorIR(
        name="indices",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["gathered"] = TensorIR(name="gathered", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["one"] = TensorIR(
        name="one",
        dtype="FLOAT32",
        shape=[],
        shape_signature=[],
        data=np.asarray(1.0, dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2], shape_signature=[2])
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="GATHER",
                inputs=["x", "indices"],
                outputs=["gathered"],
                options={"axis": 0},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["gathered", "one"],
                outputs=["y"],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            ),
        ]
    )
    return model_ir


def _make_sigmoid_mul_nchw_model_ir() -> ModelIR:
    model_ir = ModelIR(name="sigmoid_mul_nchw_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["sigmoid_out_nhwc"] = TensorIR(
        name="sigmoid_out_nhwc",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SIGMOID",
                inputs=["x"],
                outputs=["sigmoid_out_nhwc"],
                options={},
            ),
            OperatorIR(
                op_type="MUL",
                inputs=["x", "sigmoid_out_nhwc"],
                outputs=["y_nhwc"],
                options={},
            ),
        ]
    )
    return model_ir


def _make_resize_concat_nchw_model_ir(*, resize_op_type: str) -> ModelIR:
    model_ir = ModelIR(name=f"resize_concat_{resize_op_type.lower()}_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y_nhwc"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["resize_size"] = TensorIR(
        name="resize_size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([8, 8], dtype=np.int32),
    )
    model_ir.tensors["up_nhwc"] = TensorIR(
        name="up_nhwc",
        dtype="FLOAT32",
        shape=[1, 3, 8, 8],
        shape_signature=[1, 3, 8, 8],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 6, 8, 8],
        shape_signature=[1, 6, 8, 8],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type=resize_op_type,
                inputs=["x", "resize_size"],
                outputs=["up_nhwc"],
                options={"alignCorners": False},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["up_nhwc", "up_nhwc"],
                outputs=["y_nhwc"],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            ),
        ]
    )
    return model_ir


def _make_unsupported_add_model_ir() -> ModelIR:
    model_ir = _make_add_model_ir()
    model_ir.name = "unsupported_add_model_ir"
    model_ir.operators = [
        OperatorIR(op_type="WHILE", inputs=["x", "y"], outputs=["z"], options={})
    ]
    return model_ir


def _make_custom_slice_model_ir() -> ModelIR:
    model_ir = ModelIR(name="custom_slice_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[8], shape_signature=[8])
    model_ir.tensors["starts"] = TensorIR(
        name="starts",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([2], dtype=np.int32),
    )
    model_ir.tensors["ends"] = TensorIR(
        name="ends",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([6], dtype=np.int32),
    )
    model_ir.tensors["axes"] = TensorIR(
        name="axes",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([0], dtype=np.int32),
    )
    model_ir.tensors["steps"] = TensorIR(
        name="steps",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[4], shape_signature=[4])
    model_ir.operators.append(
        OperatorIR(
            op_type="CUSTOM",
            inputs=["x", "starts", "ends", "axes", "steps"],
            outputs=["y"],
            options={"customCode": "ONNX_SLICE"},
        )
    )
    return model_ir


def _make_runtime_wrapper_reshape_fallback_model_ir() -> ModelIR:
    model_ir = ModelIR(name="runtime_wrapper_reshape_fallback_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[0],
        shape_signature=[0],
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[], shape_signature=[])
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={"newShape": []},
        )
    )
    return model_ir


def _make_runtime_wrapper_resize_nhwc_model_ir() -> ModelIR:
    model_ir = ModelIR(name="runtime_wrapper_resize_nhwc_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 2],
        shape_signature=[1, 2, 3, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["resize_size"] = TensorIR(
        name="resize_size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        data=np.asarray([4, 6], dtype=np.int32),
    )
    model_ir.tensors["x_nhwc"] = TensorIR(
        name="x_nhwc",
        dtype="FLOAT32",
        shape=[1, 2, 3, 2],
        shape_signature=[1, 2, 3, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y_nhwc"] = TensorIR(
        name="y_nhwc",
        dtype="FLOAT32",
        shape=[1, 4, 6, 2],
        shape_signature=[1, 4, 6, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 6, 2],
        shape_signature=[1, 4, 6, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="ADD", inputs=["x", "zero"], outputs=["x_nhwc"], options={"fusedActivationFunction": "NONE"}),
            OperatorIR(
                op_type="RESIZE_BILINEAR",
                inputs=["x_nhwc", "resize_size"],
                outputs=["y_nhwc"],
                options={"alignCorners": False, "halfPixelCenters": True},
            ),
            OperatorIR(op_type="ADD", inputs=["y_nhwc", "zero_out"], outputs=["y"], options={"fusedActivationFunction": "NONE"}),
        ]
    )
    model_ir.tensors["zero"] = TensorIR(
        name="zero",
        dtype="FLOAT32",
        shape=[1, 2, 3, 2],
        shape_signature=[1, 2, 3, 2],
        data=np.zeros((1, 2, 3, 2), dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["zero_out"] = TensorIR(
        name="zero_out",
        dtype="FLOAT32",
        shape=[1, 4, 6, 2],
        shape_signature=[1, 4, 6, 2],
        data=np.zeros((1, 4, 6, 2), dtype=np.float32),
        logical_layout="UNKNOWN",
    )
    return model_ir


def _make_split_axis_tensor_codegen_model_ir() -> ModelIR:
    model_ir = ModelIR(name="split_axis_tensor_codegen_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["left", "right"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 3],
        shape_signature=[1, 4, 3],
        logical_layout="NCW",
    )
    model_ir.tensors["split_axis"] = TensorIR(
        name="split_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([1], dtype=np.int32),
    )
    model_ir.tensors["left"] = TensorIR(
        name="left",
        dtype="FLOAT32",
        shape=[1, 2, 3],
        shape_signature=[1, 2, 3],
        logical_layout="NCW",
    )
    model_ir.tensors["right"] = TensorIR(
        name="right",
        dtype="FLOAT32",
        shape=[1, 2, 3],
        shape_signature=[1, 2, 3],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="SPLIT",
            inputs=["split_axis", "x"],
            outputs=["left", "right"],
            options={"numSplits": 2},
        )
    )
    return model_ir


def _make_conv2d_model_ir() -> ModelIR:
    rng = np.random.default_rng(0)
    model_ir = ModelIR(name="conv2d_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3, 4, 4], shape_signature=[1, 3, 4, 4])
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[2, 3, 3, 3],
        shape_signature=[2, 3, 3, 3],
        data=rng.standard_normal((2, 3, 3, 3)).astype(np.float32),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[2],
        shape_signature=[2],
        data=rng.standard_normal((2,)).astype(np.float32),
    )
    model_ir.tensors["x"].logical_layout = "NCHW"
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 2, 4, 4], shape_signature=[1, 2, 4, 4], logical_layout="NCHW")
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    )
    return model_ir


def _make_tflite_layout_conv2d_model_ir() -> ModelIR:
    rng = np.random.default_rng(11)
    model_ir = ModelIR(name="tflite_layout_conv2d_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 4, 3],
        shape_signature=[1, 4, 4, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[2, 3, 3, 3],
        shape_signature=[2, 3, 3, 3],
        data=rng.standard_normal((2, 3, 3, 3)).astype(np.float32),
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[2],
        shape_signature=[2],
        data=rng.standard_normal((2,)).astype(np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 4, 2],
        shape_signature=[1, 4, 4, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    )
    return model_ir


def _make_conv2d_relu_model_ir() -> ModelIR:
    model_ir = _make_conv2d_model_ir()
    model_ir.name = "conv2d_relu_model_ir"
    model_ir.tensors["conv"] = TensorIR(
        name="conv",
        dtype="FLOAT32",
        shape=[1, 2, 4, 4],
        shape_signature=[1, 2, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 2, 4, 4],
        shape_signature=[1, 2, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators = [
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["conv"],
            options={
                "padding": "SAME",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "fusedActivationFunction": "NONE",
            },
        ),
        OperatorIR(op_type="RELU", inputs=["conv"], outputs=["y"], options={}),
    ]
    return model_ir


def _make_fused_conv_relu_style_model_ir() -> ModelIR:
    rng = np.random.default_rng(7)
    model_ir = ModelIR(name="fused_conv_relu_style_model_ir")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors["input"] = TensorIR(
        name="input",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["pads"] = TensorIR(
        name="pads",
        dtype="INT32",
        shape=[4, 2],
        shape_signature=[4, 2],
        data=np.asarray([[0, 0], [0, 0], [1, 1], [1, 1]], dtype=np.int32),
    )
    model_ir.tensors["padded"] = TensorIR(
        name="padded",
        dtype="FLOAT32",
        shape=[1, 3, 6, 6],
        shape_signature=[1, 3, 6, 6],
        logical_layout="NCHW",
    )
    model_ir.tensors["w0"] = TensorIR(
        name="w0",
        dtype="FLOAT32",
        shape=[3, 3, 3, 3],
        shape_signature=[3, 3, 3, 3],
        data=rng.standard_normal((3, 3, 3, 3)).astype(np.float32),
    )
    model_ir.tensors["b0"] = TensorIR(
        name="b0",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=rng.standard_normal((3,)).astype(np.float32),
    )
    model_ir.tensors["hidden"] = TensorIR(
        name="hidden",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["w1"] = TensorIR(
        name="w1",
        dtype="FLOAT32",
        shape=[3, 3, 3, 3],
        shape_signature=[3, 3, 3, 3],
        data=rng.standard_normal((3, 3, 3, 3)).astype(np.float32),
    )
    model_ir.tensors["b1"] = TensorIR(
        name="b1",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=rng.standard_normal((3,)).astype(np.float32),
    )
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="PAD", inputs=["input", "pads"], outputs=["padded"], options={}),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["padded", "w0", "b0"],
                outputs=["hidden"],
                options={
                    "padding": "VALID",
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "fusedActivationFunction": "RELU",
                },
            ),
            OperatorIR(
                op_type="CONV_2D",
                inputs=["hidden", "w1", "b1"],
                outputs=["output"],
                options={
                    "padding": "SAME",
                    "strideH": 1,
                    "strideW": 1,
                    "dilationHFactor": 1,
                    "dilationWFactor": 1,
                    "fusedActivationFunction": "RELU",
                },
            ),
        ]
    )
    return model_ir


def _make_depthwise_conv2d_model_ir() -> ModelIR:
    rng = np.random.default_rng(1)
    model_ir = ModelIR(name="depthwise_conv2d_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 1, 1],
        shape_signature=[1, 4, 1, 1],
        logical_layout="NCHW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[4, 1, 1, 1],
        shape_signature=[4, 1, 1, 1],
        data=rng.standard_normal((4, 1, 1, 1)).astype(np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=rng.standard_normal((4,)).astype(np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 1, 1],
        shape_signature=[1, 4, 1, 1],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="DEPTHWISE_CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "padding": "VALID",
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "depthMultiplier": 1,
                "fusedActivationFunction": "NONE",
            },
        )
    )
    return model_ir


def _make_boundary_wrapped_model_ir() -> ModelIR:
    model_ir = ModelIR(name="boundary_wrapped")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 3, 4, 4],
        "y": [1, 3, 4, 4],
    }
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4, 4, 3], shape_signature=[1, 4, 4, 3], logical_layout="NHWC")
    model_ir.tensors["x_internal"] = TensorIR(name="x_internal", dtype="FLOAT32", shape=[1, 3, 4, 4], shape_signature=[1, 3, 4, 4], logical_layout="NCHW")
    model_ir.tensors["x_perm"] = TensorIR(
        name="x_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        data=np.ones((1, 3, 4, 4), dtype=np.float32),
        logical_layout="NCHW",
    )
    model_ir.tensors["sum_internal"] = TensorIR(name="sum_internal", dtype="FLOAT32", shape=[1, 3, 4, 4], shape_signature=[1, 3, 4, 4], logical_layout="NCHW")
    model_ir.tensors["y_perm"] = TensorIR(
        name="y_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 4, 3], shape_signature=[1, 4, 4, 3], logical_layout="NHWC")
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["x", "x_perm"], outputs=["x_internal"], options={}),
            OperatorIR(op_type="ADD", inputs=["x_internal", "bias"], outputs=["sum_internal"], options={}),
            OperatorIR(op_type="TRANSPOSE", inputs=["sum_internal", "y_perm"], outputs=["y"], options={}),
        ]
    )
    return model_ir


def _make_recurrent_transpose_model_ir() -> ModelIR:
    model_ir = ModelIR(name="recurrent_transpose_model_ir")
    model_ir.inputs = ["GRU_1_gru_initial_h"]
    model_ir.outputs = ["GRU_1_gru_initial_h_nbh"]
    model_ir.tensors["GRU_1_gru_initial_h"] = TensorIR(
        name="GRU_1_gru_initial_h",
        dtype="FLOAT32",
        shape=[1, 5, 1],
        shape_signature=[1, 5, 1],
        logical_layout="NCW",
    )
    model_ir.tensors["GRU_1_gru_initial_h_perm"] = TensorIR(
        name="GRU_1_gru_initial_h_perm",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0, 2, 1], dtype=np.int32),
    )
    model_ir.tensors["GRU_1_gru_initial_h_nbh"] = TensorIR(
        name="GRU_1_gru_initial_h_nbh",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
        logical_layout="NWC",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="TRANSPOSE",
            inputs=["GRU_1_gru_initial_h", "GRU_1_gru_initial_h_perm"],
            outputs=["GRU_1_gru_initial_h_nbh"],
            options={},
        )
    )
    return model_ir


def _make_unknown_layout_transpose_model_ir() -> ModelIR:
    model_ir = ModelIR(name="unknown_layout_transpose_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 8, 4],
        shape_signature=[1, 1, 8, 4],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 2, 3, 1], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8, 4, 1],
        shape_signature=[1, 8, 4, 1],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="TRANSPOSE", inputs=["x", "perm"], outputs=["y"], options={})
    )
    return model_ir


def _make_shape_only_layout_transpose_model_ir() -> ModelIR:
    model_ir = ModelIR(name="shape_only_layout_transpose_model_ir")
    model_ir.inputs = ["x_nhwc_stale"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x_nhwc_stale"] = TensorIR(
        name="x_nhwc_stale",
        dtype="FLOAT32",
        shape=[1, 3, 5, 96],
        shape_signature=[1, 3, 5, 96],
        logical_layout="NCHW",
    )
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 96, 3, 5],
        shape_signature=[1, 96, 3, 5],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="TRANSPOSE", inputs=["x_nhwc_stale", "perm"], outputs=["y"], options={})
    )
    return model_ir


def _make_recurrent_public_boundary_model_ir() -> ModelIR:
    model_ir = ModelIR(name="recurrent_public_boundary_model_ir")
    model_ir.inputs = ["data_input", "init_hidden"]
    model_ir.outputs = ["preds", "hidden_out"]
    model_ir.tensors["data_input"] = TensorIR(
        name="data_input",
        dtype="FLOAT32",
        shape=[1, 10, 3],
        shape_signature=[1, 10, 3],
    )
    model_ir.tensors["init_hidden"] = TensorIR(
        name="init_hidden",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
    )
    model_ir.tensors["preds"] = TensorIR(
        name="preds",
        dtype="FLOAT32",
        shape=[1, 10, 5],
        shape_signature=[1, 10, 5],
    )
    model_ir.tensors["hidden_out"] = TensorIR(
        name="hidden_out",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="GRU",
            inputs=["data_input", "", "", "", "", "", "init_hidden"],
            outputs=["preds", "hidden_out"],
            options={},
        )
    )
    return model_ir


def _make_large_detection_head_model_ir() -> ModelIR:
    model_ir = ModelIR(name="large_detection_head")
    model_ir.inputs = ["images"]
    model_ir.outputs = ["output"]
    model_ir.tensors["images"] = TensorIR(
        name="images",
        dtype="FLOAT32",
        shape=[1, 3, 32, 32],
        shape_signature=[1, 3, 32, 32],
        logical_layout="NCHW",
    )
    for idx in range(4):
        begin_name = f"slice_begin_{idx}"
        end_name = f"slice_end_{idx}"
        stride_name = f"slice_stride_{idx}"
        out_name = f"slice_out_{idx}"
        model_ir.tensors[begin_name] = TensorIR(
            name=begin_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([0, 0, 0, 0], dtype=np.int32),
        )
        model_ir.tensors[end_name] = TensorIR(
            name=end_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([1, 3, 32, 32], dtype=np.int32),
        )
        model_ir.tensors[stride_name] = TensorIR(
            name=stride_name,
            dtype="INT32",
            shape=[4],
            shape_signature=[4],
            data=np.asarray([1, 1, 1, 1], dtype=np.int32),
        )
        model_ir.tensors[out_name] = TensorIR(
            name=out_name,
            dtype="FLOAT32",
            shape=[1, 3, 32, 32],
            shape_signature=[1, 3, 32, 32],
            logical_layout="NCHW",
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="STRIDED_SLICE",
                inputs=["images", begin_name, end_name, stride_name],
                outputs=[out_name],
                options={},
            )
        )
    prev_name = "slice_out_0"
    for idx in range(20):
        weight_name = f"conv_weight_{idx}"
        bias_name = f"conv_bias_{idx}"
        out_name = f"conv_out_{idx}"
        model_ir.tensors[weight_name] = TensorIR(
            name=weight_name,
            dtype="FLOAT32",
            shape=[3, 3, 1, 1],
            shape_signature=[3, 3, 1, 1],
            data=np.zeros((3, 3, 1, 1), dtype=np.float32),
            logical_layout="OIHW",
        )
        model_ir.tensors[bias_name] = TensorIR(
            name=bias_name,
            dtype="FLOAT32",
            shape=[3],
            shape_signature=[3],
            data=np.zeros((3,), dtype=np.float32),
        )
        model_ir.tensors[out_name] = TensorIR(
            name=out_name,
            dtype="FLOAT32",
            shape=[1, 3, 32, 32],
            shape_signature=[1, 3, 32, 32],
            logical_layout="NCHW",
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="CONV_2D",
                inputs=[prev_name, weight_name, bias_name],
                outputs=[out_name],
                options={"padding": "SAME", "strideH": 1, "strideW": 1},
            )
        )
        prev_name = out_name
    concat_inputs = []
    for idx in range(4):
        concat_name = f"concat_in_{idx}"
        model_ir.tensors[concat_name] = TensorIR(
            name=concat_name,
            dtype="FLOAT32",
            shape=[1, 2100, 85],
            shape_signature=[1, 2100, 85],
            logical_layout="NCW",
        )
        concat_inputs.append(concat_name)
    for idx in range(4):
        out_name = "output" if idx == 3 else f"concat_out_{idx}"
        current_inputs = concat_inputs[: idx + 1]
        model_ir.tensors.setdefault(
            out_name,
            TensorIR(
                name=out_name,
                dtype="FLOAT32",
                shape=[1, 2100 * (idx + 1), 85],
                shape_signature=[1, 2100 * (idx + 1), 85],
                logical_layout="NCW",
            ),
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="CONCATENATION",
                inputs=current_inputs,
                outputs=[out_name],
                options={"axis": 1, "fusedActivationFunction": "NONE"},
            )
        )
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[1, 8400, 85],
        shape_signature=[1, 8400, 85],
        logical_layout="NCW",
    )
    return model_ir


def _make_large_nhwc_heavy_model_ir() -> ModelIR:
    model_ir = ModelIR(name="large_nhwc_heavy")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["output"]
    model_ir.tensors["input"] = TensorIR(
        name="input",
        dtype="FLOAT32",
        shape=[1, 64, 64, 3],
        shape_signature=[1, 64, 64, 3],
        logical_layout="UNKNOWN",
    )
    prev_name = "input"
    for idx in range(40):
        weight_name = f"conv_weight_{idx}"
        bias_name = f"conv_bias_{idx}"
        out_name = f"feature_{idx}_nhwc"
        model_ir.tensors[weight_name] = TensorIR(
            name=weight_name,
            dtype="FLOAT32",
            shape=[8, 3, 3, 3 if idx == 0 else 8],
            shape_signature=[8, 3, 3, 3 if idx == 0 else 8],
            data=np.zeros((8, 3, 3, 3 if idx == 0 else 8), dtype=np.float32),
        )
        model_ir.tensors[bias_name] = TensorIR(
            name=bias_name,
            dtype="FLOAT32",
            shape=[8],
            shape_signature=[8],
            data=np.zeros((8,), dtype=np.float32),
        )
        model_ir.tensors[out_name] = TensorIR(
            name=out_name,
            dtype="FLOAT32",
            shape=[1, 64, 64, 8],
            shape_signature=[1, 64, 64, 8],
            logical_layout="UNKNOWN",
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="CONV_2D",
                inputs=[prev_name, weight_name, bias_name],
                outputs=[out_name],
                options={"padding": "SAME", "strideH": 1, "strideW": 1},
            )
        )
        prev_name = out_name
    for idx in range(4):
        size_name = f"resize_size_{idx}"
        resized_name = f"resized_{idx}_nhwc"
        model_ir.tensors[size_name] = TensorIR(
            name=size_name,
            dtype="INT32",
            shape=[2],
            shape_signature=[2],
            data=np.asarray([64, 64], dtype=np.int32),
        )
        model_ir.tensors[resized_name] = TensorIR(
            name=resized_name,
            dtype="FLOAT32",
            shape=[1, 64, 64, 8],
            shape_signature=[1, 64, 64, 8],
            logical_layout="UNKNOWN",
        )
        model_ir.operators.append(
            OperatorIR(
                op_type="RESIZE_BILINEAR",
                inputs=[prev_name, size_name],
                outputs=[resized_name],
                options={"alignCorners": False, "halfPixelCenters": True},
            )
        )
        prev_name = resized_name
    model_ir.tensors["output"] = TensorIR(
        name="output",
        dtype="FLOAT32",
        shape=[1, 64, 64, 8],
        shape_signature=[1, 64, 64, 8],
        logical_layout="UNKNOWN",
    )
    model_ir.operators.append(
        OperatorIR(op_type="SOFTMAX", inputs=[prev_name], outputs=["output"], options={})
    )
    return model_ir


def _make_lowered_recurrent_boundary_model_ir() -> ModelIR:
    model_ir = ModelIR(name="lowered_recurrent_boundary_model_ir")
    model_ir.inputs = ["data_input"]
    model_ir.outputs = ["preds"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "data_input": [1, 10, 3],
        "preds": [1, 10, 5],
    }
    model_ir.tensors["data_input"] = TensorIR(
        name="data_input",
        dtype="FLOAT32",
        shape=[1, 3, 10],
        shape_signature=[1, 3, 10],
        logical_layout="NCW",
    )
    model_ir.tensors["6_perm"] = TensorIR(
        name="6_perm",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([2, 0, 1], dtype=np.int32),
    )
    model_ir.tensors["GRU_1_gru_x"] = TensorIR(
        name="GRU_1_gru_x",
        dtype="FLOAT32",
        shape=[10, 1, 3],
        shape_signature=[10, 1, 3],
    )
    model_ir.tensors["preds"] = TensorIR(
        name="preds",
        dtype="FLOAT32",
        shape=[1, 5, 10],
        shape_signature=[1, 5, 10],
        logical_layout="NCW",
    )
    model_ir.operators.extend(
        [
            OperatorIR(op_type="TRANSPOSE", inputs=["data_input", "6_perm"], outputs=["GRU_1_gru_x"], options={}),
            OperatorIR(op_type="IDENTITY", inputs=["GRU_1_gru_x"], outputs=["preds"], options={}),
        ]
    )
    return model_ir


def _make_recurrent_orphan_step_alias_model_ir() -> ModelIR:
    model_ir = ModelIR(name="recurrent_orphan_step_alias_model_ir")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["hidden_out", "seq"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 5], shape_signature=[1, 5])
    model_ir.tensors["GRU_1_gru_fw_h_step_8"] = TensorIR(
        name="GRU_1_gru_fw_h_step_8",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
    )
    model_ir.tensors["GRU_1_gru_fw_h_step_9"] = TensorIR(
        name="GRU_1_gru_fw_h_step_9",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
    )
    model_ir.tensors["GRU_1_gru_fw_h_step_shape_9"] = TensorIR(
        name="GRU_1_gru_fw_h_step_shape_9",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 1, 5], dtype=np.int32),
    )
    model_ir.tensors["hidden_out"] = TensorIR(
        name="hidden_out",
        dtype="FLOAT32",
        shape=[1, 1, 5],
        shape_signature=[1, 1, 5],
    )
    model_ir.tensors["seq"] = TensorIR(
        name="seq",
        dtype="FLOAT32",
        shape=[2, 1, 5],
        shape_signature=[2, 1, 5],
    )
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="RESHAPE",
                inputs=["x", "GRU_1_gru_fw_h_step_shape_9"],
                outputs=["hidden_out"],
                options={"newShape": [1, 1, 5]},
            ),
            OperatorIR(
                op_type="CONCATENATION",
                inputs=["GRU_1_gru_fw_h_step_8", "GRU_1_gru_fw_h_step_9"],
                outputs=["seq"],
                options={"axis": 0, "fusedActivationFunction": "NONE"},
            ),
        ]
    )
    return model_ir


def _make_split_with_local_transpose_model_ir() -> ModelIR:
    model_ir = ModelIR(name="split_local_transpose")
    model_ir.inputs = ["input"]
    model_ir.outputs = ["out0", "out1"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "input": [1, 3, 4, 4],
        "out0": [1, 1, 4, 4],
        "out1": [1, 3, 4, 4],
    }
    model_ir.tensors["input"] = TensorIR(name="input", dtype="FLOAT32", shape=[1, 4, 4, 3], shape_signature=[1, 4, 4, 3], logical_layout="NHWC")
    model_ir.tensors["split_axis"] = TensorIR(
        name="split_axis",
        dtype="INT32",
        shape=[1],
        shape_signature=[1],
        data=np.asarray([3], dtype=np.int32),
    )
    model_ir.tensors["out0"] = TensorIR(name="out0", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1], logical_layout="NHWC")
    model_ir.tensors["out0_bias"] = TensorIR(
        name="out0_bias",
        dtype="FLOAT32",
        shape=[1, 4, 4, 1],
        shape_signature=[1, 4, 4, 1],
        data=np.ones((1, 4, 4, 1), dtype=np.float32),
        logical_layout="NHWC",
    )
    model_ir.tensors["out0_sum"] = TensorIR(name="out0_sum", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1], logical_layout="NHWC")
    model_ir.tensors["input_perm"] = TensorIR(
        name="input_perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["input_onnx_ncx_internal_local"] = TensorIR(
        name="input_onnx_ncx_internal_local",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["out1_mul_const"] = TensorIR(
        name="out1_mul_const",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        data=np.full((1, 3, 4, 4), 2.0, dtype=np.float32),
        logical_layout="NCHW",
    )
    model_ir.tensors["out1"] = TensorIR(name="out1", dtype="FLOAT32", shape=[1, 3, 4, 4], shape_signature=[1, 3, 4, 4], logical_layout="NCHW")
    model_ir.operators.extend(
        [
            OperatorIR(
                op_type="SPLIT",
                inputs=["split_axis", "input"],
                outputs=["out0"],
                options={"numSplits": 1},
            ),
            OperatorIR(
                op_type="ADD",
                inputs=["out0", "out0_bias"],
                outputs=["out0_sum"],
                options={"fusedActivationFunction": "NONE"},
            ),
            OperatorIR(
                op_type="TRANSPOSE",
                inputs=["input", "input_perm"],
                outputs=["input_onnx_ncx_internal_local"],
                options={},
            ),
            OperatorIR(
                op_type="MUL",
                inputs=["input_onnx_ncx_internal_local", "out1_mul_const"],
                outputs=["out1"],
                options={"fusedActivationFunction": "NONE"},
            ),
        ]
    )
    model_ir.outputs = ["out0_sum", "out1"]
    model_ir.metadata["onnx_boundary_shape_signature_map"]["out0_sum"] = [1, 1, 4, 4]
    return model_ir


def _write_model_ir_as_tflite(tmpdir: str, name: str, model_ir: ModelIR) -> str:
    schema_tflite = load_schema_module(tmpdir)
    tflite_path = os.path.join(tmpdir, f"{name}.tflite")
    write_model_file(
        schema_tflite=schema_tflite,
        model_ir=model_ir,
        output_tflite_path=tflite_path,
    )
    return tflite_path


def test_normalize_model_ir_restores_channel_first_boundaries() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_boundary_wrapped_model_ir())
    assert normalized.tensors["x"].shape == [1, 3, 4, 4]
    assert normalized.tensors["y"].shape == [1, 3, 4, 4]
    assert all(str(op.op_type) != "TRANSPOSE" for op in normalized.operators)


def test_normalize_model_ir_rewrites_split_and_local_boundary_transpose() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_split_with_local_transpose_model_ir())
    assert normalized.tensors["input"].shape == [1, 3, 4, 4]
    assert normalized.tensors["out0_sum"].shape == [1, 1, 4, 4]
    assert normalized.tensors["out1"].shape == [1, 3, 4, 4]
    split_op = next(op for op in normalized.operators if str(op.op_type) == "SPLIT")
    assert split_op.inputs[1] == "input"
    split_axis_tensor = normalized.tensors[str(split_op.inputs[0])]
    assert np.asarray(split_axis_tensor.data).reshape(-1).tolist() == [1]
    assert all(str(op.op_type) != "TRANSPOSE" for op in normalized.operators)


def test_normalize_model_ir_rewrites_rank3_recurrent_boundary_transpose_perm() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_lowered_recurrent_boundary_model_ir())
    assert normalized.tensors["data_input"].shape == [1, 10, 3]
    perm_tensor = normalized.tensors["6_perm"]
    assert np.asarray(perm_tensor.data).reshape(-1).tolist() == [1, 0, 2]


def test_normalize_model_ir_repairs_orphan_recurrent_step_aliases() -> None:
    normalized = normalize_model_ir_for_pytorch_channel_first(_make_recurrent_orphan_step_alias_model_ir())
    concat_op = next(op for op in normalized.operators if str(op.op_type) == "CONCATENATION")
    assert concat_op.inputs == ["GRU_1_gru_fw_h_step_8", "hidden_out"]
    assert "GRU_1_gru_fw_h_step_9" not in normalized.tensors


def test_normalize_model_ir_does_not_repermute_reshape_outputs() -> None:
    model_ir = ModelIR(name="reshape_no_repermute")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.metadata["onnx_boundary_shape_signature_map"] = {
        "x": [1, 1, 4, 4],
        "y": [1, 4, 4],
    }
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 4, 4, 1], shape_signature=[1, 4, 4, 1])
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 4, 4], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 4], shape_signature=[1, 4, 4])
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={"newShape": [1, 4, 4]})
    )
    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
    assert normalized.tensors["y"].shape == [1, 4, 4]
    assert np.asarray(normalized.tensors["shape"].data).reshape(-1).tolist() == [1, 4, 4]


def test_normalize_model_ir_synchronizes_reshape_targets_after_layout_permutation() -> None:
    model_ir = ModelIR(name="reshape_layout_sync")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 4, 3],
        shape_signature=[1, 2, 4, 3],
        logical_layout="NHWC",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, 8, 3], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 8, 3],
        shape_signature=[1, 8, 3],
        logical_layout="NWC",
    )
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={"newShape": [1, 8, 3]})
    )

    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)

    assert normalized.tensors["y"].shape == [1, 3, 8]
    assert np.asarray(normalized.tensors["shape"].data).reshape(-1).tolist() == [1, 3, 8]
    reshape_op = next(op for op in normalized.operators if str(op.op_type) == "RESHAPE")
    assert reshape_op.options["newShape"] == [1, 3, 8]


def test_export_pytorch_package_roundtrip_add_relu(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_relu_model_ir(),
        output_folder_path=str(tmp_path / "add_relu_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    assert hasattr(pkg, "Model")
    model = pkg.load_model()
    assert isinstance(model, pkg.Model)
    assert not hasattr(model, "_model")
    x = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float32)
    y = torch.tensor([[4.0, 5.0, -10.0]], dtype=torch.float32)
    out = model(x, y)
    assert torch.allclose(out, torch.relu(x + y))
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["outputs"] == ["z"]


def test_export_pytorch_package_model_class_is_directly_importable(tmp_path) -> None:
    model_ir = _make_conv2d_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_importable_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.arange(1, 1 + (1 * 3 * 4 * 4), dtype=torch.float32).reshape(1, 3, 4, 4)
    out = model(x)
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    runtime_source = (Path(package_path) / "runtime.py").read_text(encoding="utf-8")
    saved_state_dict = torch.load(Path(package_path) / "state_dict.pth", map_location="cpu")
    assert "class Model(torch.nn.Module):" in model_source
    assert "load_generated_model_package" not in model_source
    assert "_GraphExecutor" not in model_source
    assert "from .runtime import (" in model_source
    assert "load_generated_weights(" in model_source
    assert "self.conv2d_0 = torch.nn.Conv2d(" in model_source
    assert "LOAD_SPECS" not in model_source
    assert "_copy_tensor_data" not in model_source
    assert "_validate_state_dict_keys" not in model_source
    assert "def _copy_tensor_data" in runtime_source
    assert "def load_generated_weights(" in runtime_source
    assert set(saved_state_dict.keys()) == set(model.state_dict().keys())
    assert len(model.state_dict()) > 0
    w = torch.as_tensor(model_ir.tensors["w"].data).permute(0, 3, 1, 2)
    b = torch.as_tensor(model_ir.tensors["b"].data)
    ref = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_runtime_wrapper_package_supports_custom_onnx_slice(tmp_path) -> None:
    package_path = _export_runtime_wrapper_package_from_model_ir(
        model_ir=_make_custom_slice_model_ir(),
        output_folder_path=str(tmp_path / "custom_slice_runtime_wrapper"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(8, dtype=torch.float32)
    out = model(x)
    assert torch.equal(out, x[2:6])


def test_export_runtime_wrapper_package_supports_tflite_layout_conv2d(tmp_path) -> None:
    model_ir = _make_tflite_layout_conv2d_model_ir()
    package_path = _export_runtime_wrapper_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "tflite_layout_conv_runtime_wrapper"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 4 * 4 * 3), dtype=torch.float32).reshape(1, 4, 4, 3)
    out = model(x)
    w = torch.as_tensor(model_ir.tensors["w"].data).permute(0, 3, 1, 2)
    b = torch.as_tensor(model_ir.tensors["b"].data)
    ref = torch.nn.functional.conv2d(x.permute(0, 3, 1, 2), w, b, stride=1, padding=1).permute(0, 2, 3, 1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_runtime_wrapper_package_reshape_falls_back_to_newshape_option(tmp_path) -> None:
    package_path = _export_runtime_wrapper_package_from_model_ir(
        model_ir=_make_runtime_wrapper_reshape_fallback_model_ir(),
        output_folder_path=str(tmp_path / "reshape_runtime_wrapper"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([3.0], dtype=torch.float32)
    out = model(x)
    assert out.shape == torch.Size([])
    assert torch.equal(out.reshape(1), x)


def test_export_runtime_wrapper_package_supports_nhwc_resize(tmp_path) -> None:
    package_path = _export_runtime_wrapper_package_from_model_ir(
        model_ir=_make_runtime_wrapper_resize_nhwc_model_ir(),
        output_folder_path=str(tmp_path / "resize_nhwc_runtime_wrapper"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1 * 2 * 3 * 2, dtype=torch.float32).reshape(1, 2, 3, 2)
    out = model(x)
    expected = F.interpolate(
        x.permute(0, 3, 1, 2),
        size=(4, 6),
        mode="bilinear",
        align_corners=False,
    ).permute(0, 2, 3, 1).contiguous()
    assert out.shape == torch.Size([1, 4, 6, 2])
    assert torch.allclose(out, expected)


def test_wrapper_model_source_uses_lint_safe_forward_named_dispatch(tmp_path) -> None:
    package_path = _export_runtime_wrapper_package_from_model_ir(
        model_ir=_make_custom_slice_model_ir(),
        output_folder_path=str(tmp_path / "wrapper_lint_safe"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self._model: Any = load_generated_model_package(" in model_source
    assert "forward_named = getattr(self._model, 'forward_named', None)" in model_source
    assert "if callable(forward_named):" in model_source
    assert "cast(Callable[..., Any], forward_named)(*args, **kwargs)" in model_source


def test_export_pytorch_package_model_source_emits_direct_abs(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_abs_model_ir(),
        output_folder_path=str(tmp_path / "abs_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.tensor([[-1.0, 2.0, -3.0]], dtype=torch.float32)
    out = model(x)
    assert torch.allclose(out, torch.abs(x))
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.abs(x)" in model_source
    assert "_GraphExecutor" not in model_source


def test_export_pytorch_package_model_source_emits_direct_relu_0_to_1(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_relu_0_to_1_model_ir(),
        output_folder_path=str(tmp_path / "relu_0_to_1_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.tensor([[-1.0, 0.5, 2.0]], dtype=torch.float32)
    out = model(x)
    assert torch.allclose(out, torch.clamp(x, min=0.0, max=1.0))
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "torch.clamp(x, min=0.0, max=1.0)" in model_source


def test_export_pytorch_package_shape_preserving_unary_skips_stale_align(tmp_path) -> None:
    model_ir = ModelIR(name="shape_preserving_unary_skips_stale_align")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[180, 320, 3],
        shape_signature=[180, 320, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[180, 3, 320],
        shape_signature=[180, 3, 320],
        logical_layout="NCW",
    )
    model_ir.operators = [
        OperatorIR(op_type="RELU_0_TO_1", inputs=["x"], outputs=["y"], options={}),
    ]
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "shape_preserving_unary"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_align_tensor_to_target_shape(torch.clamp" not in model_source
    assert "torch.clamp(" in model_source


def test_export_pytorch_package_model_source_mixes_module_and_functional_ops(tmp_path) -> None:
    model_ir = _make_conv2d_relu_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv_relu_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.arange(1, 1 + (1 * 3 * 4 * 4), dtype=torch.float32).reshape(1, 3, 4, 4)
    out_direct = model(x)
    out_loaded = pkg.load_model()(x)
    assert torch.allclose(out_direct, out_loaded, atol=1e-5, rtol=1e-5)
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "self.conv_block_0 = _Conv2dBlock(" in model_source
    assert "torch.relu(x)" in model_source


def test_export_pytorch_package_model_source_is_pytorch_like_for_fused_conv_relu(tmp_path) -> None:
    model_ir = _make_fused_conv_relu_style_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "fused_conv_relu_like_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    reloaded_model = pkg.Model(load_weights=False)
    reloaded_model.load_state_dict(torch.load(Path(package_path) / "state_dict.pth", map_location="cpu"))
    x = torch.arange(1, 1 + (1 * 3 * 4 * 4), dtype=torch.float32).reshape(1, 3, 4, 4)
    out = model(x)
    out_reloaded = reloaded_model(x)
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    runtime_source = (Path(package_path) / "runtime.py").read_text(encoding="utf-8")
    saved_state_dict = torch.load(Path(package_path) / "state_dict.pth", map_location="cpu")

    assert "_copy_tensor_data" not in model_source
    assert "_validate_state_dict_keys" not in model_source
    assert "_torch_dtype(" not in model_source
    assert "cast(torch.Tensor, self." not in model_source
    assert "load_generated_weights(" in model_source
    assert "class _Conv2dBlock(torch.nn.Module):" in model_source
    assert "self.conv_block_0 = _Conv2dBlock(" in model_source
    assert "self.conv_block_1 = _Conv2dBlock(" in model_source
    assert "_apply_module_conv2d" not in model_source
    assert "_forward_stage_" not in model_source
    assert "x = self.conv_block_0(x)" in model_source
    assert "x = self.conv_block_1(x)" in model_source
    assert "def _copy_tensor_data" in runtime_source
    assert "def load_generated_weights(" in runtime_source
    assert "def _apply_binary(" not in runtime_source
    assert list(saved_state_dict.keys()) == list(model.state_dict().keys())
    assert torch.allclose(out, out_reloaded, atol=1e-5, rtol=1e-5)
    assert getattr(model.conv_block_0, "activation") == "relu"
    assert getattr(model.conv_block_1, "activation") == "relu"

    ref = model.conv_block_0(x)
    ref = model.conv_block_1(ref)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_reshape_flattens_channel_first_spatial_to_feature_last_sequence_even_with_negative_one(tmp_path) -> None:
    rng = np.random.default_rng(11)
    model_ir = ModelIR(name="reshape_channel_first_to_feature_last_negative_one")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 2, 3, 4],
        shape_signature=[1, 2, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, -1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, -1, 2],
        shape_signature=[1, -1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.operators = [
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x", "shape"],
            outputs=["y"],
            options={},
        ),
    ]

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_negative_one"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.from_numpy(rng.standard_normal((1, 2, 3, 4), dtype=np.float32))
    out = model(x)
    ref = x.permute(0, 2, 3, 1).contiguous().reshape(1, -1, 2)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".permute(0, 2, 3, 1).contiguous()" in model_source


def test_export_pytorch_package_reshape_flattens_packed_channel_detection_head_to_feature_last_sequence(tmp_path) -> None:
    rng = np.random.default_rng(13)
    model_ir = ModelIR(name="reshape_packed_channel_detection_head")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 6, 3, 4],
        shape_signature=[1, 6, 3, 4],
        logical_layout="NCHW",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1, -1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, -1, 2],
        shape_signature=[1, -1, 2],
        logical_layout="UNKNOWN",
    )
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={}),
    ]

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_packed_detection_head"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.from_numpy(rng.standard_normal((1, 6, 3, 4), dtype=np.float32))
    out = model(x)
    ref = x.permute(0, 2, 3, 1).contiguous().reshape(1, -1, 2)
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".permute(0, 2, 3, 1).contiguous()" in model_source


def test_export_pytorch_package_reshape_batchless_hwc_back_to_nchw(tmp_path) -> None:
    rng = np.random.default_rng(17)
    model_ir = ModelIR(name="reshape_batchless_hwc_back_to_nchw")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[180, 320, 3],
        shape_signature=[180, 320, 3],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["shape"] = TensorIR(
        name="shape",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([1, 180, 320, 3], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 180, 320],
        shape_signature=[1, 3, 180, 320],
        logical_layout="NCHW",
    )
    model_ir.operators = [
        OperatorIR(op_type="RESHAPE", inputs=["x", "shape"], outputs=["y"], options={}),
    ]

    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_hwc_to_nchw"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.Model()
    x = torch.from_numpy(rng.standard_normal((180, 320, 3), dtype=np.float32))
    out = model(x)
    ref = x.reshape(1, 180, 320, 3).permute(0, 3, 1, 2).contiguous()
    assert torch.allclose(out, ref, atol=1e-6, rtol=1e-6)
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert ".permute(0, 3, 1, 2).contiguous()" in model_source


def test_normalize_model_ir_allows_attention_like_softmax_consumed_by_batch_matmul() -> None:
    model_ir = ModelIR(name="attention_like_softmax_batch_matmul")
    model_ir.inputs = ["x", "v"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 10, 3600],
        shape_signature=[1, 4, 10, 3600],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["scores"] = TensorIR(
        name="scores",
        dtype="FLOAT32",
        shape=[1, 4, 10, 3600],
        shape_signature=[1, 4, 10, 3600],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["v"] = TensorIR(
        name="v",
        dtype="FLOAT32",
        shape=[1, 4, 3600, 32],
        shape_signature=[1, 4, 3600, 32],
        logical_layout="UNKNOWN",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 10, 32],
        shape_signature=[1, 4, 10, 32],
        logical_layout="UNKNOWN",
    )
    model_ir.operators = [
        OperatorIR(op_type="SOFTMAX", inputs=["x"], outputs=["scores"], options={"beta": 1.0}),
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["scores", "v"],
            outputs=["y"],
            options={"adjX": False, "adjY": False, "fusedActivationFunction": "NONE"},
        ),
    ]

    normalized = normalize_model_ir_for_pytorch_channel_first(model_ir)
    assert normalized.tensors["scores"].shape == [1, 4, 10, 3600]


def test_export_pytorch_package_state_dict_is_load_state_dict_compatible(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_conv2d_relu_model_ir(),
        output_folder_path=str(tmp_path / "conv_relu_state_dict_compatible"),
    )
    pkg = _import_generated_package(package_path)
    direct_model = pkg.Model()
    loaded_model = pkg.Model(load_weights=False)
    loaded_model.load_state_dict(torch.load(Path(package_path) / "state_dict.pth", map_location="cpu"))
    x = torch.arange(1, 1 + (1 * 3 * 4 * 4), dtype=torch.float32).reshape(1, 3, 4, 4)
    assert list(torch.load(Path(package_path) / "state_dict.pth", map_location="cpu").keys()) == list(direct_model.state_dict().keys())
    assert torch.allclose(direct_model(x), loaded_model(x), atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_does_not_materialize_saved_model_bridge_when_native_succeeds(tmp_path) -> None:
    calls = {"count": 0}

    def _unused_saved_model_factory() -> str | None:
        calls["count"] += 1
        bridge_dir = tmp_path / "unused_saved_model_bridge"
        bridge_dir.mkdir(parents=True, exist_ok=True)
        return str(bridge_dir)

    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_conv2d_relu_model_ir(),
        output_folder_path=str(tmp_path / "native_without_bridge"),
        fallback_saved_model_factory=_unused_saved_model_factory,
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert calls["count"] == 0
    assert not (tmp_path / "unused_saved_model_bridge").exists()
    assert "load_generated_model_package" not in model_source


def test_export_pytorch_package_generates_native_yolov7_package_when_model_is_available(tmp_path) -> None:
    model_path = Path("yolov7_tiny_head_0.768_post_480x640.onnx")
    if not model_path.exists():
        pytest.skip("yolov7_tiny_head_0.768_post_480x640.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = lower_onnx_to_ir(
        model_proto,
        output_file_name="yolov7_native_codegen_test",
        show_progress=False,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "yolov7_native_pytorch"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert "execution_backend" not in metadata
    assert "load_generated_model_package" not in model_source
    assert "TENSOR_STORAGE_NAMES" not in model_source
    assert "resolve_model_tensor" not in model_source
    assert "_torch_dtype(" not in model_source
    assert "cast(torch.Tensor, self." not in model_source
    assert "def _init_constants(self) -> None:" in model_source
    assert model_source.count("_apply_") <= 24
    assert "_forward_stage_0" in model_source


def test_export_pytorch_package_generates_native_yolox_package_when_model_is_available(tmp_path) -> None:
    model_path = Path("yolox_s.onnx")
    if not model_path.exists():
        pytest.skip("yolox_s.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = lower_onnx_to_ir(
        model_proto,
        output_file_name="yolox_native_codegen_test",
        show_progress=False,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "yolox_native_pytorch"),
        fallback_saved_model_path=str(tmp_path / "saved_model_fallback"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert "execution_backend" not in metadata
    assert "load_generated_model_package" not in model_source
    assert "class _Conv2dBlock(torch.nn.Module):" in model_source
    assert "_apply_gather(" not in model_source
    assert "_apply_gather_nd(" not in model_source
    assert "_apply_slice(" not in model_source
    assert "_apply_strided_slice(" not in model_source
    assert model_source.count("register_buffer(") <= 12
    assert "class _Conv2dBlock(torch.nn.Module):" in model_source
    assert "self.conv_block_0 = _Conv2dBlock(" in model_source
    assert "_forward_stage_0" in model_source

    pkg = _import_generated_package(package_path)
    loaded_model = pkg.Model(load_weights=True, eval_mode=True)
    reloaded_model = pkg.Model(load_weights=False, eval_mode=True)
    reloaded_model.load_state_dict(torch.load(package_dir / "state_dict.pth", map_location="cpu"), strict=True)
    x = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        loaded_outputs = loaded_model(x)
        reloaded_outputs = reloaded_model(x)
    assert isinstance(loaded_outputs, torch.Tensor)
    assert isinstance(reloaded_outputs, torch.Tensor)
    assert list(loaded_outputs.shape) == list(reloaded_outputs.shape)
    assert torch.allclose(loaded_outputs, reloaded_outputs)


def test_export_pytorch_package_generates_native_mobilebert_package_when_model_is_available(tmp_path) -> None:
    model_path = Path("lite_model_mobilebert_1_metadata_1.onnx")
    if not model_path.exists():
        pytest.skip("lite_model_mobilebert_1_metadata_1.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = lower_onnx_to_ir(
        model_proto,
        output_file_name="mobilebert_native_codegen_test",
        show_progress=False,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "mobilebert_native_pytorch"),
        fallback_saved_model_path=str(tmp_path / "saved_model_fallback"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert "execution_backend" not in metadata
    assert "load_generated_model_package" not in model_source
    assert "class _AffineLayerNorm(torch.nn.Module):" in model_source
    assert "class _GeneratedEncoderLayer0Attention(torch.nn.Module):" in model_source
    assert "class _GeneratedEncoderLayer0FFN(torch.nn.Module):" in model_source
    assert "class _GeneratedEncoderLayer0(torch.nn.Module):" in model_source
    assert "self.encoder_layer_0 = _GeneratedEncoderLayer0(" in model_source
    assert "_layer_norm(" in model_source
    assert "FakeLayerNorm_gamma_read: torch.Tensor" not in model_source
    assert "self.linear_0 = torch.nn.Linear(" in model_source
    assert "torch.where(" in model_source
    assert "torch.lt(" in model_source

    pkg = _import_generated_package(package_path)
    loaded_model = pkg.Model(load_weights=True, eval_mode=True)
    reloaded_model = pkg.Model(load_weights=False, eval_mode=True)
    reloaded_model.load_state_dict(torch.load(package_dir / "state_dict.pth", map_location="cpu"), strict=True)
    input_ids = torch.zeros((1, 384), dtype=torch.int32)
    input_mask = torch.ones((1, 384), dtype=torch.int32)
    segment_ids = torch.zeros((1, 384), dtype=torch.int32)
    with torch.no_grad():
        loaded_outputs = loaded_model(input_ids, input_mask, segment_ids)
        reloaded_outputs = reloaded_model(input_ids, input_mask, segment_ids)
    assert isinstance(loaded_outputs, tuple)
    assert isinstance(reloaded_outputs, tuple)
    state_dict_keys = set(torch.load(package_dir / "state_dict.pth", map_location="cpu").keys())
    assert any(key.endswith("_layer_norm.gamma") for key in state_dict_keys)
    assert any(key.endswith("_layer_norm.beta") for key in state_dict_keys)
    assert len(loaded_outputs) == len(reloaded_outputs) == 2
    for loaded_output, reloaded_output in zip(loaded_outputs, reloaded_outputs):
        assert list(loaded_output.shape) == list(reloaded_output.shape)
        assert torch.allclose(loaded_output, reloaded_output, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_imported_tflite_with_gelu_stays_native(
    tmp_path,
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="gelu_fallback")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4],
        shape_signature=[1, 4],
    )
    model_ir.operators.append(
        OperatorIR(op_type="GELU", inputs=["x"], outputs=["y"], options={})
    )
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "gelu_fallback", model_ir)

    def _raise(*_args, **_kwargs):
        raise ModelIRPyTorchExportError("force tflite import fallback")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.pytorch_exporter.normalize_model_ir_for_pytorch_channel_first",
        _raise,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "gelu_native_pytorch"),
        fallback_tflite_path=tflite_path,
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert "execution_backend" not in metadata
    assert "load_generated_model_package" not in model_source
    assert "F.gelu(" in model_source


def test_export_pytorch_package_generates_native_swinir_package_when_model_is_available(tmp_path) -> None:
    model_path = Path("swinir-m_64x64_12.onnx")
    if not model_path.exists():
        pytest.skip("swinir-m_64x64_12.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = lower_onnx_to_ir(
        model_proto,
        output_file_name="swinir_native_codegen_test",
        show_progress=False,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "swinir_native_pytorch"),
        fallback_tflite_path=str(tmp_path / "unused_fallback.tflite"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert "execution_backend" not in metadata
    assert "load_generated_model_package" not in model_source
    assert "_forward_stage_0" in model_source
    assert "_depth_to_space_x_" in model_source
    assert "_space_to_depth_x_" in model_source
    assert "F.pixel_shuffle(" not in model_source


def test_export_pytorch_package_generates_native_pidnet_package_when_model_is_available(tmp_path) -> None:
    model_path = Path("pidnet_S_cityscapes_192x320.onnx")
    if not model_path.exists():
        pytest.skip("pidnet_S_cityscapes_192x320.onnx is not available")
    model_proto = onnx.load(model_path)
    model_ir = lower_onnx_to_ir(
        model_proto,
        output_file_name="pidnet_native_codegen_test",
        show_progress=False,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "pidnet_native_pytorch"),
        fallback_tflite_path=str(tmp_path / "unused_fallback.tflite"),
    )
    package_dir = Path(package_path)
    metadata = json.loads((package_dir / "metadata.json").read_text())
    model_source = (package_dir / "model.py").read_text()
    assert "execution_backend" not in metadata
    assert "load_generated_model_package" not in model_source
    assert "torch.argmax(" in model_source
    assert "F.avg_pool2d(" in model_source or "self._avg_pool2d_same(" in model_source


def test_export_pytorch_package_conv2d_nchw(tmp_path) -> None:
    model_ir = _make_conv2d_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv2d_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (1 * 3 * 4 * 4), dtype=torch.float32).reshape(1, 3, 4, 4)
    out = model(x)
    w = torch.as_tensor(model_ir.tensors["w"].data).permute(0, 3, 1, 2)
    b = torch.as_tensor(model_ir.tensors["b"].data)
    ref = torch.nn.functional.conv2d(x, w, b, stride=1, padding=1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_broadcasts_1d_constant_along_channel_axis(tmp_path) -> None:
    model_ir = ModelIR(name="channel_affine")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 5],
        shape_signature=[1, 3, 5],
        logical_layout="NCW",
    )
    model_ir.tensors["gamma"] = TensorIR(
        name="gamma",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[3],
        shape_signature=[3],
        data=np.asarray([0.5, 1.5, 2.5], dtype=np.float32),
    )
    model_ir.tensors["scaled"] = TensorIR(
        name="scaled",
        dtype="FLOAT32",
        shape=[1, 3, 5],
        shape_signature=[1, 3, 5],
        logical_layout="NCW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 5],
        shape_signature=[1, 3, 5],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="MUL", inputs=["x", "gamma"], outputs=["scaled"], options={})
    )
    model_ir.operators.append(
        OperatorIR(op_type="ADD", inputs=["scaled", "bias"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "channel_affine_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".reshape([1, 3, 1])" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(15, dtype=torch.float32).reshape(1, 3, 5)
    out = model(x)
    ref = x * torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32).reshape(1, 3, 1)
    ref = ref + torch.tensor([0.5, 1.5, 2.5], dtype=torch.float32).reshape(1, 3, 1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_batch_matmul_transposes_channel_first_sequence_input(tmp_path) -> None:
    model_ir = ModelIR(name="channel_first_batch_matmul")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 5],
        shape_signature=[1, 3, 5],
        logical_layout="NCW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[3, 4],
        shape_signature=[3, 4],
        data=np.arange(12, dtype=np.float32).reshape(3, 4),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 5],
        shape_signature=[1, 4, 5],
        logical_layout="NCW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="BATCH_MATMUL", inputs=["x", "w"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "batch_matmul_cf_pytorch"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".transpose(-1, -2)" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(15, dtype=torch.float32).reshape(1, 3, 5)
    out = model(x)
    ref = torch.matmul(x.transpose(-1, -2), torch.arange(12, dtype=torch.float32).reshape(3, 4)).transpose(-1, -2)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_reshape_flattens_channel_first_spatial_to_feature_last_sequence(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_channel_first_spatial_to_sequence")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 3],
        shape_signature=[1, 4, 3],
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESHAPE",
            inputs=["x"],
            outputs=["y"],
            options={"newShape": [1, 4, 3]},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_channel_first_spatial_to_sequence"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(0, 2, 3, 1).contiguous()" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    out = model(x)
    ref = x.permute(0, 2, 3, 1).reshape(1, 4, 3)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_prefers_standard_channel_last_permute_for_rank4_constant_binary_inputs(tmp_path) -> None:
    model_ir = ModelIR(name="rank4_binary_constant_channel_last_permute")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[5, 2, 2, 4],
        shape_signature=[5, 2, 2, 4],
    )
    model_ir.tensors["bias"] = TensorIR(
        name="bias",
        dtype="FLOAT32",
        shape=[1, 4, 2, 2],
        shape_signature=[1, 4, 2, 2],
        logical_layout="NCHW",
        data=np.arange(16, dtype=np.float32).reshape(1, 4, 2, 2),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[5, 2, 2, 4],
        shape_signature=[5, 2, 2, 4],
    )
    model_ir.operators.append(
        OperatorIR(op_type="ADD", inputs=["x", "bias"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "rank4_binary_constant_channel_last_permute"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(*(0, 2, 3, 1)).contiguous()" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(5 * 2 * 2 * 4, dtype=torch.float32).reshape(5, 2, 2, 4)
    out = model(x)
    ref = x + torch.arange(16, dtype=torch.float32).reshape(1, 4, 2, 2).permute(0, 2, 3, 1)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_swaps_spatial_axes_for_conv1d_bridge_input(tmp_path) -> None:
    model_ir = ModelIR(name="conv1d_bridge_conv2d")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 1, 64, 1],
        shape_signature=[1, 1, 64, 1],
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[4, 1, 3, 1],
        shape_signature=[4, 1, 3, 1],
        data=np.arange(12, dtype=np.float32).reshape(4, 1, 3, 1),
    )
    model_ir.tensors["b"] = TensorIR(
        name="b",
        dtype="FLOAT32",
        shape=[4],
        shape_signature=[4],
        data=np.zeros((4,), dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 1, 64],
        shape_signature=[1, 4, 1, 64],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="CONV_2D",
            inputs=["x", "w", "b"],
            outputs=["y"],
            options={
                "strideH": 1,
                "strideW": 1,
                "dilationHFactor": 1,
                "dilationWFactor": 1,
                "padding": "SAME",
                "fusedActivationFunction": "NONE",
            },
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "conv1d_bridge_conv2d"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(0, 1, 3, 2).contiguous()" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(64, dtype=torch.float32).reshape(1, 1, 64, 1)
    out = model(x)
    ref = model.conv2d_0(x.permute(0, 1, 3, 2).contiguous())
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_export_pytorch_package_reshape_prefers_feature_last_before_adjx_batch_matmul(tmp_path) -> None:
    model_ir = ModelIR(name="reshape_feature_last_before_adjx_batch_matmul")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 4, 1, 6],
        shape_signature=[1, 4, 1, 6],
        logical_layout="NCHW",
    )
    model_ir.tensors["seq"] = TensorIR(
        name="seq",
        dtype="FLOAT32",
        shape=[1, 4, 6],
        shape_signature=[1, 4, 6],
        logical_layout="NCW",
    )
    model_ir.tensors["w"] = TensorIR(
        name="w",
        dtype="FLOAT32",
        shape=[6, 2],
        shape_signature=[6, 2],
        data=np.arange(12, dtype=np.float32).reshape(6, 2),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 4, 2],
        shape_signature=[1, 4, 2],
    )
    model_ir.operators.append(
        OperatorIR(op_type="RESHAPE", inputs=["x"], outputs=["seq"], options={"newShape": [1, 4, 6]})
    )
    model_ir.operators.append(
        OperatorIR(op_type="BATCH_MATMUL", inputs=["seq", "w"], outputs=["y"], options={"adjX": True, "adjY": False})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "reshape_feature_last_before_adjx_batch_matmul"),
    )
    model_source = (Path(package_path) / "model.py").read_text()
    assert ".permute(0, 3, 1, 2).contiguous()" in model_source
    assert "[1, 6, 4]" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(24, dtype=torch.float32).reshape(1, 4, 1, 6)
    out = model(x)
    seq = x.permute(0, 3, 1, 2).reshape(1, 6, 4)
    ref = torch.matmul(seq.transpose(-1, -2), torch.arange(12, dtype=torch.float32).reshape(6, 2))
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-5)


def test_convert_flatbuffer_direct_outputs_pytorch_package(tmp_path) -> None:
    model_path = tmp_path / "add.onnx"
    onnx.save(_make_add_model(), str(model_path))
    output_dir = tmp_path / "out"
    onnx2tf.convert(
        input_onnx_file_path=str(model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "add_pytorch"
    assert (package_path / "state_dict.pth").exists()
    pkg = _import_generated_package(str(package_path))
    model = pkg.load_model()
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    assert torch.allclose(model(x, y), x + y)


def test_convert_input_tflite_outputs_pytorch_package(tmp_path) -> None:
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "add_input", _make_add_model_ir())
    output_dir = tmp_path / "out_tflite"
    onnx2tf.convert(
        input_tflite_file_path=str(tflite_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "add_input_pytorch"
    assert (package_path / "metadata.json").exists()
    pkg = _import_generated_package(str(package_path))
    model = pkg.load_model()
    x = torch.tensor([[0.5, 1.5, 2.5]], dtype=torch.float32)
    y = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    assert torch.allclose(model(x, y), x + y)


def test_evaluate_tflite_pytorch_package_outputs(tmp_path) -> None:
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "add_input_eval", _make_add_model_ir())
    output_dir = tmp_path / "out_tflite_eval"
    onnx2tf.convert(
        input_tflite_file_path=str(tflite_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        disable_model_save=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_path = output_dir / "add_input_eval_pytorch"
    report = evaluate_tflite_pytorch_package_outputs(
        tflite_path=str(tflite_path),
        package_dir=str(package_path),
        output_report_path=str(output_dir / "add_input_eval_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert report["reference_backend"] == "tflite"
    assert report["evaluation_pass"] is True
    assert report["overall_metrics"]["max_abs"] == 0.0


def test_export_tflite_model_flatbuffer_direct_split_pytorch_package(tmp_path) -> None:
    output_dir = tmp_path / "split_out"
    outputs = export_tflite_model_flatbuffer_direct(
        onnx_graph=_make_add_model(),
        output_folder_path=str(output_dir),
        output_file_name="split_add",
        force_split_manifest=True,
        output_pytorch_from_model_ir=True,
        pytorch_output_folder_path=str(output_dir / "split_add_pytorch"),
    )
    assert "split_manifest_path" in outputs
    assert "split_pytorch_package_dirs" in outputs
    manifest = json.loads(Path(outputs["split_manifest_path"]).read_text(encoding="utf-8"))
    assert len(manifest["partitions"]) >= 1
    assert "pytorch_package_dir" in manifest["partitions"][0]
    first_package = output_dir / manifest["partitions"][0]["pytorch_package_dir"]
    assert (first_package / "state_dict.pth").exists()


def test_export_pytorch_package_rejects_custom_ops(tmp_path) -> None:
    model_ir = ModelIR(name="custom_model")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1], shape_signature=[1])
    model_ir.operators.append(
        OperatorIR(op_type="CUSTOM", inputs=["x"], outputs=["y"], options={})
    )
    with pytest.raises(ModelIRPyTorchExportError, match="CUSTOM"):
        export_pytorch_package_from_model_ir(
            model_ir=model_ir,
            output_folder_path=str(tmp_path / "custom_pytorch"),
        )


def test_export_pytorch_package_rejects_residual_layout_transpose(tmp_path) -> None:
    model_ir = ModelIR(name="residual_layout")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3, 4, 4], shape_signature=[1, 3, 4, 4], logical_layout="NCHW")
    model_ir.tensors["x_mid"] = TensorIR(name="x_mid", dtype="FLOAT32", shape=[1, 3, 4, 4], shape_signature=[1, 3, 4, 4], logical_layout="NCHW")
    model_ir.tensors["perm"] = TensorIR(
        name="perm",
        dtype="INT32",
        shape=[4],
        shape_signature=[4],
        data=np.asarray([0, 3, 1, 2], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 4, 3, 4], shape_signature=[1, 4, 3, 4], logical_layout="UNKNOWN")
    model_ir.operators.extend(
        [
            OperatorIR(op_type="IDENTITY", inputs=["x"], outputs=["x_mid"], options={}),
            OperatorIR(op_type="TRANSPOSE", inputs=["x_mid", "perm"], outputs=["y"], options={}),
        ]
    )
    with pytest.raises(ModelIRPyTorchExportError, match="residual layout transpose"):
        export_pytorch_package_from_model_ir(
            model_ir=model_ir,
            output_folder_path=str(tmp_path / "residual_pytorch"),
        )


def test_export_pytorch_package_roundtrip_batch_matmul(tmp_path) -> None:
    model_ir = ModelIR(name="batch_matmul")
    model_ir.inputs = ["x", "y"]
    model_ir.outputs = ["z"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[2, 3, 4], shape_signature=[2, 3, 4])
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[2, 4, 5], shape_signature=[2, 4, 5])
    model_ir.tensors["z"] = TensorIR(name="z", dtype="FLOAT32", shape=[2, 3, 5], shape_signature=[2, 3, 5])
    model_ir.operators.append(
        OperatorIR(
            op_type="BATCH_MATMUL",
            inputs=["x", "y"],
            outputs=["z"],
            options={"adjX": False, "adjY": False, "fusedActivationFunction": "NONE"},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "batch_matmul_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(1, 1 + (2 * 3 * 4), dtype=torch.float32).reshape(2, 3, 4)
    y = torch.arange(1, 1 + (2 * 4 * 5), dtype=torch.float32).reshape(2, 4, 5)
    out = model(x, y)
    assert torch.allclose(out, torch.matmul(x, y))


def test_export_pytorch_package_roundtrip_unsafe_tensor_names(tmp_path) -> None:
    model_ir = ModelIR(name="unsafe_tensor_names")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["bias.0/part"] = TensorIR(
        name="bias.0/part",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[1, 3],
        data=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32),
    )
    model_ir.tensors["y"] = TensorIR(name="y", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.operators.append(
        OperatorIR(op_type="ADD", inputs=["x", "bias.0/part"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "unsafe_names_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32)
    out = model(x)
    assert torch.allclose(out, x + torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32))


def test_export_pytorch_package_supports_constant_tensor_outputs(tmp_path) -> None:
    model_ir = ModelIR(name="constant_output")
    model_ir.outputs = ["const"]
    model_ir.tensors["const"] = TensorIR(
        name="const",
        dtype="FLOAT32",
        shape=[2, 2],
        shape_signature=[2, 2],
        data=np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "constant_output_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    out = model()
    assert torch.allclose(out, torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32))


def test_export_pytorch_package_reports_missing_torch(tmp_path, monkeypatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError("No module named 'torch'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _fake_import)
    with pytest.raises(ModelIRPyTorchExportError, match="requires `torch`"):
        export_pytorch_package_from_model_ir(
            model_ir=_make_add_model_ir(),
            output_folder_path=str(tmp_path / "missing_torch"),
        )


def test_export_pytorch_package_prefers_reimported_native_wrapper_before_tflite_backend(
    tmp_path,
) -> None:
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "fallback_add", _make_add_model_ir())
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_unsupported_add_model_ir(),
        output_folder_path=str(tmp_path / "fallback_add_pytorch"),
        fallback_tflite_path=tflite_path,
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert "execution_backend" not in metadata
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    y = torch.tensor([[4.0, 5.0, 6.0]], dtype=torch.float32)
    out = model(x, y)
    assert torch.allclose(out, x + y)


def test_export_pytorch_package_codegen_supports_gather_without_wrapper(
    tmp_path,
) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_gather_model_ir(),
        output_folder_path=str(tmp_path / "gather_pytorch"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert "execution_backend" not in metadata
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "load_generated_model_package" not in model_py
    assert "_apply_gather" not in model_py
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[10.0, 20.0, 30.0, 40.0]], dtype=torch.float32)
    out = model(x)
    expected = x[:, [0, 2]]
    assert torch.allclose(out, expected)


def test_codegen_sigmoid_mul_chain_stays_channel_first_without_align(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_sigmoid_mul_nchw_model_ir(),
        output_folder_path=str(tmp_path / "sigmoid_mul_pytorch"),
    )
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_align_tensor_to_target_shape(" not in model_py
    assert "sigmoid_out_nhwc =" not in model_py
    assert "y_nhwc =" not in model_py
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(1, 3, 4, 4)
    out = model(x)
    assert torch.allclose(out, x * torch.sigmoid(x), atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("resize_op_type", ["RESIZE_NEAREST_NEIGHBOR", "RESIZE_BILINEAR"])
def test_codegen_resize_concat_chain_avoids_stale_nhwc_and_align(
    tmp_path,
    resize_op_type: str,
) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_resize_concat_nchw_model_ir(resize_op_type=resize_op_type),
        output_folder_path=str(tmp_path / f"{resize_op_type.lower()}_concat_pytorch"),
    )
    model_py = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "F.interpolate(" in model_py
    assert "_align_tensor_to_target_shape(" not in model_py
    assert "up_nhwc =" not in model_py
    assert "y_nhwc =" not in model_py
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.randn(1, 3, 4, 4)
    out = model(x)
    mode = "nearest" if resize_op_type == "RESIZE_NEAREST_NEIGHBOR" else "bilinear"
    kwargs = {} if mode == "nearest" else {"align_corners": False}
    expected = torch.cat(
        [F.interpolate(x, size=(8, 8), mode=mode, **kwargs)] * 2,
        dim=1,
    )
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_direct_codegen_concat_accepts_scalar_constant_input(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_concat_scalar_model_ir(),
        output_folder_path=str(tmp_path / "concat_scalar_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    out = model(torch.tensor([5.0], dtype=torch.float32))
    assert torch.allclose(out, torch.tensor([5.0, 1.0], dtype=torch.float32))


def test_direct_codegen_split_uses_axis_tensor_input(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_split_axis_tensor_codegen_model_ir(),
        output_folder_path=str(tmp_path / "split_axis_tensor_codegen_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    left, right = model(x)
    assert torch.equal(left, x[:, :2, :])
    assert torch.equal(right, x[:, 2:, :])


def test_native_runtime_wrapper_concat_accepts_scalar_constant_input(tmp_path) -> None:
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_gather_concat_scalar_model_ir(),
        output_folder_path=str(tmp_path / "gather_concat_scalar_pytorch"),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert "execution_backend" not in metadata
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    out = model(torch.tensor([5.0, 9.0], dtype=torch.float32))
    assert torch.allclose(out, torch.tensor([5.0, 1.0], dtype=torch.float32))


def test_direct_codegen_depthwise_conv_keeps_nchw_input_layout(tmp_path) -> None:
    model_ir = _make_depthwise_conv2d_model_ir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "depthwise_conv2d_pytorch"),
    )
    pkg = _import_generated_package(package_path)
    model = pkg.load_model()
    x = torch.tensor([[[[1.0]], [[2.0]], [[3.0]], [[4.0]]]], dtype=torch.float32)
    out = model(x)
    weight = torch.as_tensor(model_ir.tensors["w"].data)
    bias = torch.as_tensor(model_ir.tensors["b"].data)
    expected = F.conv2d(x, weight, bias=bias, stride=1, padding=0, groups=4)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected)


def test_reject_residual_layout_transposes_allows_recurrent_rank3_state_transpose() -> None:
    _reject_residual_layout_transposes(
        _make_recurrent_transpose_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_reject_residual_layout_transposes_allows_unknown_layout_transpose() -> None:
    _reject_residual_layout_transposes(
        _make_unknown_layout_transpose_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_reject_residual_layout_transposes_allows_shape_only_layout_transpose() -> None:
    _reject_residual_layout_transposes(
        _make_shape_only_layout_transpose_model_ir(),
        preserve_channel_last_tensor_names=set(),
    )


def test_remove_redundant_layout_transposes_removes_shape_only_layout_transpose() -> None:
    model_ir = _make_shape_only_layout_transpose_model_ir()
    _remove_redundant_layout_transposes(
        model_ir,
        original_layouts={"x_nhwc_stale": "NCHW", "y": "NCHW"},
        preserve_channel_last_tensor_names=set(),
    )
    assert model_ir.operators[0].op_type == "IDENTITY"


def test_infer_model_ir_logical_layouts_uses_nwc_for_recurrent_public_rank3_boundaries() -> None:
    model_ir = _make_recurrent_public_boundary_model_ir()
    infer_model_ir_logical_layouts(model_ir)
    assert model_ir.metadata["onnx_public_layout_map"]["data_input"] == "NWC"
    assert model_ir.metadata["onnx_public_layout_map"]["preds"] == "NWC"


def test_export_pytorch_package_prefers_tflite_backend_for_large_channel_first_softmax_without_saved_model(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="softmax_fallback")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="SOFTMAX", inputs=["x"], outputs=["y"], options={})
    )
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "softmax_fallback", model_ir)
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "softmax_fallback_pytorch"),
        fallback_tflite_path=tflite_path,
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("execution_backend") == "tflite"


def test_should_prefer_tflite_backend_for_large_detection_head_runtime_fallback() -> None:
    model_ir = _make_large_detection_head_model_ir()
    assert _should_prefer_tflite_backed_package(model_ir) is True


def test_should_prefer_tflite_backend_for_large_nhwc_heavy_graph() -> None:
    model_ir = _make_large_nhwc_heavy_model_ir()
    assert _should_prefer_tflite_backed_package(model_ir) is True


def test_export_pytorch_package_does_not_force_tflite_backend_when_custom_ops_exist(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="softmax_fallback_custom")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="SOFTMAX", inputs=["x"], outputs=["y"], options={})
    )
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "softmax_fallback_custom", model_ir)
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "softmax_fallback_custom_pytorch"),
        fallback_tflite_path=tflite_path,
        fallback_tflite_has_custom_ops=True,
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata.get("execution_backend") is None


def test_export_pytorch_package_prefers_saved_model_fallback_over_tflite_shortcut(
    tmp_path,
    monkeypatch,
) -> None:
    model_ir = ModelIR(name="softmax_fallback_saved_model")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="SOFTMAX", inputs=["x"], outputs=["y"], options={})
    )
    tflite_path = _write_model_ir_as_tflite(str(tmp_path), "softmax_fallback_saved_model", model_ir)
    saved_model_dir = tmp_path / "saved_model"
    saved_model_dir.mkdir()

    def _raise(*_args, **_kwargs):
        raise ModelIRPyTorchExportError("force saved_model fallback")

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.pytorch_exporter.normalize_model_ir_for_pytorch_channel_first",
        _raise,
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "softmax_fallback_saved_model_pytorch"),
        fallback_tflite_path=tflite_path,
        fallback_saved_model_path=str(saved_model_dir),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["execution_backend"] == "saved_model"


def test_export_pytorch_package_prefers_native_runtime_over_saved_model_for_large_supported_models(
    tmp_path,
) -> None:
    model_ir = ModelIR(name="saved_model_preferred")
    model_ir.inputs = ["x0"]
    model_ir.tensors["x0"] = TensorIR(
        name="x0",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    prev_name = "x0"
    for idx in range(64):
        next_name = f"x{idx + 1}"
        model_ir.tensors[next_name] = TensorIR(
            name=next_name,
            dtype="FLOAT32",
            shape=[1, 3, 2, 2],
            shape_signature=[1, 3, 2, 2],
            logical_layout="NCHW",
        )
        model_ir.operators.append(
            OperatorIR(op_type="IDENTITY", inputs=[prev_name], outputs=[next_name], options={})
        )
        prev_name = next_name
    model_ir.outputs = [prev_name]

    saved_model_dir = tmp_path / "saved_model"
    saved_model_dir.mkdir()
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "saved_model_preferred_pytorch"),
        fallback_saved_model_path=str(saved_model_dir),
    )
    metadata = json.loads((Path(package_path) / "metadata.json").read_text(encoding="utf-8"))
    assert "execution_backend" not in metadata


def test_evaluate_pytorch_package_outputs_numeric(tmp_path) -> None:
    model_path = tmp_path / "add.onnx"
    onnx.save(_make_add_model(), str(model_path))
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_model_ir(),
        output_folder_path=str(tmp_path / "eval_add_pytorch"),
    )
    report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(tmp_path / "add_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert report["evaluation_pass"] is True


def test_evaluate_pytorch_package_outputs_accepts_task_equivalent_probability_maps(tmp_path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2, 2, 2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2, 2, 2])
    node = helper.make_node("Softmax", ["x"], ["y"], name="SoftmaxNode", axis=1)
    graph = helper.make_graph([node], "softmax_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10
    package_dir = tmp_path / "probability_map_pkg"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text(
        "from .model import load_model\n",
        encoding="utf-8",
    )
    (package_dir / "model.py").write_text(
        "from __future__ import annotations\n\n"
        "import torch\n\n"
        "class _Model:\n"
        "    def forward_named(self, *, x):\n"
        "        return {'y': torch.softmax(x * 0.9, dim=1)}\n\n"
        "def load_model(*, eval_mode=True):\n"
        "    return _Model()\n",
        encoding="utf-8",
    )
    (package_dir / "metadata.json").write_text(
        json.dumps(
            {
                "inputs": ["x"],
                "outputs": ["y"],
                "tensors": {
                    "x": {"shape": [1, 2, 2, 2], "shape_signature": [1, 2, 2, 2], "logical_layout": "NCHW"},
                    "y": {"shape": [1, 2, 2, 2], "shape_signature": [1, 2, 2, 2], "logical_layout": "NCHW"},
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    custom_input_path = tmp_path / "probability_map_input.npy"
    np.save(
        custom_input_path,
        np.asarray([[[[0.1, 0.2], [0.3, 0.4]], [[1.0, 1.1], [1.2, 1.3]]]], dtype=np.float32),
    )
    report = evaluate_pytorch_package_outputs(
        onnx_graph=model,
        package_dir=str(package_dir),
        output_report_path=str(tmp_path / "probability_map_report.json"),
        num_samples=1,
        custom_input_op_name_np_data_path=[
            ["x", str(custom_input_path)]
        ],
    )
    assert report["evaluation_pass"] is True
    assert report["task_equivalent_numeric_outputs"] == ["y"]
    assert report["per_output"]["y"]["probability_map_equivalence"]["pass"] is True


def test_evaluate_pytorch_package_outputs_resolves_sanitized_output_aliases(tmp_path) -> None:
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3])
    y = helper.make_tensor_value_info("Identity:0", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["x"], ["Identity:0"], name="IdentityNode")
    graph = helper.make_graph([node], "identity_graph", [x], [y])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 13)])
    model.ir_version = 10

    model_ir = ModelIR(name="identity_alias")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["Identity__0"]
    model_ir.tensors["x"] = TensorIR(name="x", dtype="FLOAT32", shape=[1, 3], shape_signature=[1, 3])
    model_ir.tensors["Identity__0"] = TensorIR(
        name="Identity__0",
        dtype="FLOAT32",
        shape=[1, 3],
        shape_signature=[1, 3],
    )
    model_ir.operators.append(
        OperatorIR(op_type="IDENTITY", inputs=["x"], outputs=["Identity__0"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "eval_identity_alias_pytorch"),
    )
    report = evaluate_pytorch_package_outputs(
        onnx_graph=model,
        package_dir=str(package_path),
        output_report_path=str(tmp_path / "identity_alias_report.json"),
        num_samples=1,
    )
    assert report["evaluation_pass"] is True


def test_smoke_test_pytorch_package_imports_dotted_package_dir(tmp_path) -> None:
    model_path = tmp_path / "add.onnx"
    onnx.save(_make_add_model(), str(model_path))
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_model_ir(),
        output_folder_path=str(tmp_path / "eval.add_pytorch"),
    )
    report = smoke_test_pytorch_package_inference(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(tmp_path / "add_pytorch_smoke_report.json"),
        num_samples=1,
    )
    assert report["inference_pass"] is True


def test_saved_model_backend_remaps_sanitized_input_names_and_aligns_shapes(
    tmp_path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "saved_model_pkg"
    (package_dir / "saved_model").mkdir(parents=True)
    metadata = {
        "execution_backend": "saved_model",
        "saved_model_dir_name": "saved_model",
        "inputs": ["bert/embeddings/embedding_lookup_1"],
        "outputs": ["out"],
        "boundary_shape_signatures": {
            "bert/embeddings/embedding_lookup_1": [1, 384, 512],
            "out": [1, 512, 384],
        },
        "tensors": {
            "out": {
                "shape": [1, 512, 384],
                "shape_signature": [1, 512, 384],
            },
        },
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    captured = {}

    class _FakeTensorSpec:
        def __init__(self, shape):
            self.shape = tuple(shape)

    class _FakeTensor:
        def __init__(self, array):
            self._array = np.asarray(array)

        def numpy(self):
            return self._array

    class _FakeSignatureMap(dict):
        pass

    class _FakeCallable:
        structured_input_signature = (
            (),
            {
                "bert_embeddings_embedding_lookup_1": _FakeTensorSpec((1, 512, 384)),
            },
        )

        def __call__(self, **kwargs):
            captured.update(kwargs)
            return {
                "out": _FakeTensor(kwargs["bert_embeddings_embedding_lookup_1"]),
            }

    class _FakeSavedModelModule:
        def __init__(self):
            self.signatures = _FakeSignatureMap(
                serving_default=_FakeCallable(),
            )

    class _FakeTensorFlowModule:
        class saved_model:
            @staticmethod
            def load(_path):
                return _FakeSavedModelModule()

        @staticmethod
        def convert_to_tensor(value, dtype=None):
            return np.asarray(value, dtype=dtype)

        string = str

    monkeypatch.setitem(sys.modules, "tensorflow", _FakeTensorFlowModule())
    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.randn(1, 384, 512, dtype=torch.float32)
    outputs = cast(Any, model).forward_named(**{"bert/embeddings/embedding_lookup_1": x})
    assert "bert_embeddings_embedding_lookup_1" in captured
    assert np.asarray(captured["bert_embeddings_embedding_lookup_1"]).shape == (1, 512, 384)
    assert tuple(outputs["out"].shape) == (1, 512, 384)


def test_generated_backend_accepts_original_input_alias_for_sanitized_metadata_name(
    tmp_path,
) -> None:
    package_dir = tmp_path / "generated_pkg"
    package_dir.mkdir(parents=True)
    metadata = {
        "inputs": ["onnx____Reshape_0"],
        "outputs": ["y"],
        "tensors": {
            "onnx____Reshape_0": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
            "y": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
        },
        "operators": [
            {
                "op_type": "IDENTITY",
                "inputs": ["onnx____Reshape_0"],
                "outputs": ["y"],
                "options": {},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save({}, package_dir / "state_dict.pth")

    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(**{"onnx::Reshape_0": x})
    assert torch.allclose(outputs["y"], x)


def test_generated_backend_accepts_input_alias_with_extra_runtime_prefix(
    tmp_path,
) -> None:
    package_dir = tmp_path / "generated_pkg_prefixed"
    package_dir.mkdir(parents=True)
    metadata = {
        "inputs": ["wa/xvector/block1/tdnnd1/nonlinear2/relu/Relu_output_0"],
        "outputs": ["y"],
        "tensors": {
            "wa/xvector/block1/tdnnd1/nonlinear2/relu/Relu_output_0": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
            "y": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
        },
        "operators": [
            {
                "op_type": "IDENTITY",
                "inputs": ["wa/xvector/block1/tdnnd1/nonlinear2/relu/Relu_output_0"],
                "outputs": ["y"],
                "options": {},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save({}, package_dir / "state_dict.pth")

    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(**{"/xvector/block1/tdnnd1/nonlinear2/relu/Relu_output_0": x})
    assert torch.allclose(outputs["y"], x)


def test_infer_spatial_shape_for_transposed_conv2d_uses_nhwc_target_shape() -> None:
    raw = torch.zeros((1, 64, 1, 68), dtype=torch.float32)
    target_h, target_w = _infer_spatial_shape_for_transposed_conv2d(
        raw_output=raw,
        target_shape=[1, 1, 68, 64],
        fallback_shape=[1, 64, 1, 68],
    )
    assert (target_h, target_w) == (1, 68)


def test_tflite_backend_recovers_missing_internal_tensor_data(
    tmp_path,
    monkeypatch,
) -> None:
    package_dir = tmp_path / "tflite_missing_data_pkg"
    package_dir.mkdir(parents=True)
    metadata = {
        "execution_backend": "tflite",
        "tflite_file_name": "model_float32.tflite",
        "inputs": ["input"],
        "outputs": ["output"],
        "boundary_shape_signatures": {
            "input": [1, 3],
            "output": [1, 3],
        },
        "tensors": {
            "input": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
            "output": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
        },
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (package_dir / "model_float32.tflite").write_bytes(b"dummy")

    class _FakeInterpreter:
        def __init__(self):
            self._tensors = {}

        def allocate_tensors(self):
            return None

        def get_input_details(self):
            return [
                {
                    "name": "input",
                    "index": 0,
                    "shape": np.asarray([1, 3], dtype=np.int32),
                    "shape_signature": np.asarray([1, 3], dtype=np.int32),
                    "dtype": np.float32,
                },
            ]

        def get_output_details(self):
            return [
                {
                    "name": "output",
                    "index": 1,
                    "shape": np.asarray([1, 3], dtype=np.int32),
                    "shape_signature": np.asarray([1, 3], dtype=np.int32),
                    "dtype": np.float32,
                    "quantization": (0.0, 0),
                    "quantization_parameters": {},
                },
            ]

        def get_tensor_details(self):
            return [
                *self.get_input_details(),
                *self.get_output_details(),
                {
                    "name": "onnx____Slice_2112",
                    "index": 749,
                    "shape": np.asarray([1], dtype=np.int32),
                    "shape_signature": np.asarray([1], dtype=np.int32),
                    "dtype": np.int64,
                    "quantization": (0.0, 0),
                    "quantization_parameters": {},
                },
            ]

        def get_signature_list(self):
            return {}

        def set_tensor(self, index, value):
            self._tensors[int(index)] = np.asarray(value)

        def invoke(self):
            if 749 not in self._tensors:
                raise RuntimeError("Input tensor 749 lacks data")
            self._tensors[1] = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)

        def get_tensor(self, index):
            return np.asarray(self._tensors[int(index)])

    monkeypatch.setattr(
        "onnx2tf.tflite_builder.pytorch_package_runtime._create_tflite_interpreter",
        lambda _path: _FakeInterpreter(),
    )

    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    outputs = cast(Any, model).forward_named(input=x)
    assert torch.allclose(outputs["output"], x)


def test_generated_softmax_aligns_output_to_target_shape(tmp_path) -> None:
    package_dir = tmp_path / "softmax_align_pkg"
    package_dir.mkdir(parents=True)
    metadata = {
        "inputs": ["x"],
        "outputs": ["y"],
        "tensors": {
            "x": {
                "dtype": "FLOAT32",
                "shape": [1, 2, 2, 3],
                "shape_signature": [1, 2, 2, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
            "y": {
                "dtype": "FLOAT32",
                "shape": [1, 3, 2, 2],
                "shape_signature": [1, 3, 2, 2],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
        },
        "operators": [
            {
                "op_type": "SOFTMAX",
                "inputs": ["x"],
                "outputs": ["y"],
                "options": {},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save({}, package_dir / "state_dict.pth")
    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    out = model(x)
    ref = torch.softmax(x, dim=1)
    assert torch.allclose(out, ref)


def test_native_softmax_codegen_uses_channel_axis_for_channel_first_input(tmp_path) -> None:
    model_ir = ModelIR(name="softmax_native_codegen")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(op_type="SOFTMAX", inputs=["x"], outputs=["y"], options={})
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "softmax_native_codegen_pkg"),
    )
    package_dir = Path(package_path)
    model_source = (package_dir / "model.py").read_text(encoding="utf-8")
    assert "_apply_softmax(x, axis=1, beta=1.0" in model_source
    pkg = _import_generated_package(package_path)
    model = pkg.Model(load_weights=True, eval_mode=True)
    x = torch.arange(12, dtype=torch.float32).reshape(1, 3, 2, 2)
    out = model(x)
    ref = torch.softmax(x, dim=1)
    assert torch.allclose(out, ref)


def test_native_resize_bilinear_half_pixel_codegen_uses_runtime_helper(tmp_path) -> None:
    model_ir = ModelIR(name="resize_bilinear_half_pixel_codegen")
    model_ir.inputs = ["x"]
    model_ir.outputs = ["y"]
    model_ir.tensors["x"] = TensorIR(
        name="x",
        dtype="FLOAT32",
        shape=[1, 3, 2, 2],
        shape_signature=[1, 3, 2, 2],
        logical_layout="NCHW",
    )
    model_ir.tensors["size"] = TensorIR(
        name="size",
        dtype="INT32",
        shape=[2],
        shape_signature=[2],
        logical_layout="UNKNOWN",
        data=np.asarray([4, 4], dtype=np.int32),
    )
    model_ir.tensors["y"] = TensorIR(
        name="y",
        dtype="FLOAT32",
        shape=[1, 3, 4, 4],
        shape_signature=[1, 3, 4, 4],
        logical_layout="NCHW",
    )
    model_ir.operators.append(
        OperatorIR(
            op_type="RESIZE_BILINEAR",
            inputs=["x", "size"],
            outputs=["y"],
            options={"alignCorners": False, "halfPixelCenters": True},
        )
    )
    package_path = export_pytorch_package_from_model_ir(
        model_ir=model_ir,
        output_folder_path=str(tmp_path / "resize_bilinear_half_pixel_codegen_pkg"),
    )
    model_source = (Path(package_path) / "model.py").read_text(encoding="utf-8")
    assert "_apply_resize(x, torch.as_tensor([4, 4], dtype=torch.int32, device=x.device), method='bilinear'" in model_source


def test_generated_gather_handles_axis_one_before_depth_to_space(tmp_path) -> None:
    package_dir = tmp_path / "gather_depth_to_space_pkg"
    package_dir.mkdir(parents=True)
    metadata = {
        "inputs": ["input"],
        "outputs": ["output"],
        "tensors": {
            "input": {
                "dtype": "FLOAT32",
                "shape": [2, 12, 8, 8],
                "shape_signature": [2, 12, 8, 8],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
            "indices": {
                "dtype": "INT32",
                "shape": [12],
                "shape_signature": [12],
                "is_variable": False,
                "has_data": True,
                "logical_layout": "UNKNOWN",
            },
            "reordered": {
                "dtype": "FLOAT32",
                "shape": [2, 12, 8, 8],
                "shape_signature": [2, 12, 8, 8],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
            "output": {
                "dtype": "FLOAT32",
                "shape": [2, 3, 16, 16],
                "shape_signature": [2, 3, 16, 16],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
        },
        "operators": [
            {
                "op_type": "GATHER",
                "inputs": ["input", "indices"],
                "outputs": ["reordered"],
                "options": {"axis": 1, "batchDims": 0},
            },
            {
                "op_type": "DEPTH_TO_SPACE",
                "inputs": ["reordered"],
                "outputs": ["output"],
                "options": {"blockSize": 2},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "indices": torch.tensor(
                [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
                dtype=torch.int32,
            ),
        },
        package_dir / "state_dict.pth",
    )
    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.arange(2 * 12 * 8 * 8, dtype=torch.float32).reshape(2, 12, 8, 8)
    out = cast(Any, model).forward_named(input=x)["output"]
    expected = F.pixel_shuffle(
        torch.index_select(
            x,
            1,
            torch.tensor([0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11], dtype=torch.int64),
        ),
        2,
    )
    assert torch.allclose(out, expected)


def test_generated_gather_elides_crd_to_dcr_reorder_after_channel_first_normalization(
    tmp_path,
) -> None:
    package_dir = tmp_path / "gather_crd_depth_to_space_pkg"
    package_dir.mkdir(parents=True)
    metadata = {
        "inputs": ["input"],
        "outputs": ["output"],
        "tensors": {
            "input": {
                "dtype": "FLOAT32",
                "shape": [2, 12, 8, 8],
                "shape_signature": [2, 12, 8, 8],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
            "wa/DepthToSpace_crd_to_dcr_indices": {
                "dtype": "INT32",
                "shape": [12],
                "shape_signature": [12],
                "is_variable": False,
                "has_data": True,
                "logical_layout": "UNKNOWN",
            },
            "reordered": {
                "dtype": "FLOAT32",
                "shape": [2, 12, 8, 8],
                "shape_signature": [2, 12, 8, 8],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
            "output": {
                "dtype": "FLOAT32",
                "shape": [2, 3, 16, 16],
                "shape_signature": [2, 3, 16, 16],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "NCHW",
            },
        },
        "operators": [
            {
                "op_type": "GATHER",
                "inputs": ["input", "wa/DepthToSpace_crd_to_dcr_indices"],
                "outputs": ["reordered"],
                "options": {"axis": 1, "batchDims": 0},
            },
            {
                "op_type": "DEPTH_TO_SPACE",
                "inputs": ["reordered"],
                "outputs": ["output"],
                "options": {"blockSize": 2},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save(
        {
            "wa_DepthToSpace_crd_to_dcr_indices": torch.tensor(
                [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11],
                dtype=torch.int32,
            ),
        },
        package_dir / "state_dict.pth",
    )
    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.arange(2 * 12 * 8 * 8, dtype=torch.float32).reshape(2, 12, 8, 8)
    out = cast(Any, model).forward_named(input=x)["output"]
    expected = F.pixel_shuffle(x, 2)
    assert torch.allclose(out, expected)


def test_generated_batch_matmul_handles_rank_two_dense_case(tmp_path) -> None:
    package_dir = tmp_path / "batch_matmul_dense_pkg"
    package_dir.mkdir(parents=True)
    weight = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.5, 2.0, 0.5],
        ],
        dtype=torch.float32,
    )
    metadata = {
        "inputs": ["x"],
        "outputs": ["y"],
        "tensors": {
            "x": {
                "dtype": "FLOAT32",
                "shape": [1, 2],
                "shape_signature": [1, 2],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
            "w": {
                "dtype": "FLOAT32",
                "shape": [2, 3],
                "shape_signature": [2, 3],
                "is_variable": False,
                "has_data": True,
                "logical_layout": "UNKNOWN",
            },
            "logits": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
            "y": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
        },
        "operators": [
            {
                "op_type": "BATCH_MATMUL",
                "inputs": ["x", "w"],
                "outputs": ["logits"],
                "options": {"adjX": False, "adjY": False},
            },
            {
                "op_type": "SOFTMAX",
                "inputs": ["logits"],
                "outputs": ["y"],
                "options": {"beta": 1.0},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save({"w": weight}, package_dir / "state_dict.pth")
    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.tensor([[2.0, -1.0]], dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = torch.softmax(torch.matmul(x, weight), dim=-1)
    assert torch.allclose(out, expected)


def test_generated_binary_kernel_applies_fused_relu(tmp_path) -> None:
    package_dir = tmp_path / "fused_binary_pkg"
    package_dir.mkdir(parents=True)
    metadata = {
        "inputs": ["x"],
        "outputs": ["y"],
        "tensors": {
            "x": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
            "bias": {
                "dtype": "FLOAT32",
                "shape": [3],
                "shape_signature": [3],
                "is_variable": False,
                "has_data": True,
                "logical_layout": "UNKNOWN",
            },
            "y": {
                "dtype": "FLOAT32",
                "shape": [1, 3],
                "shape_signature": [1, 3],
                "is_variable": False,
                "has_data": False,
                "logical_layout": "UNKNOWN",
            },
        },
        "operators": [
            {
                "op_type": "ADD",
                "inputs": ["x", "bias"],
                "outputs": ["y"],
                "options": {"fusedActivationFunction": "RELU"},
            },
        ],
    }
    (package_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    torch.save({"bias": torch.tensor([-2.0, 1.0, -1.0], dtype=torch.float32)}, package_dir / "state_dict.pth")
    model = load_generated_model_package(
        package_dir=str(package_dir),
        eval_mode=True,
    )
    x = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float32)
    out = cast(Any, model).forward_named(x=x)["y"]
    expected = torch.relu(x + torch.tensor([-2.0, 1.0, -1.0], dtype=torch.float32))
    assert torch.allclose(out, expected)


def test_evaluate_pytorch_package_outputs_skips_onnxruntime_inference_error(
    tmp_path,
    monkeypatch,
) -> None:
    model_path = tmp_path / "add.onnx"
    onnx.save(_make_add_model(), str(model_path))
    package_path = export_pytorch_package_from_model_ir(
        model_ir=_make_add_model_ir(),
        output_folder_path=str(tmp_path / "eval_add_pytorch_skip"),
    )

    class _FakeOrtSession:
        def __init__(self, *_args, **_kwargs):
            pass

        def run(self, *_args, **_kwargs):
            raise RuntimeError("ort run failed")

    class _FakeOrtModule:
        InferenceSession = _FakeOrtSession

    monkeypatch.setitem(sys.modules, "onnxruntime", _FakeOrtModule())
    report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(model_path)),
        package_dir=str(package_path),
        output_report_path=str(tmp_path / "add_pytorch_accuracy_report_skip.json"),
        num_samples=1,
    )
    assert report["evaluation_pass"] is None
    assert report["evaluation_skipped"] is True
    assert report["skip_reason"] == "onnxruntime_reference_error"
    assert report["onnxruntime_reference_error"]["stage"] == "inference"


def test_evaluate_pytorch_package_outputs_string_model(tmp_path) -> None:
    string_model_path = Path("/home/b920405/git/onnx2tf/string_normalizer_11.onnx")
    output_dir = tmp_path / "string_out"
    onnx2tf.convert(
        input_onnx_file_path=str(string_model_path),
        output_folder_path=str(output_dir),
        tflite_backend="flatbuffer_direct",
        flatbuffer_direct_output_pytorch=True,
        output_signaturedefs=False,
        non_verbose=True,
        verbosity="error",
    )
    package_dir = output_dir / "string_normalizer_11_pytorch"
    report = evaluate_pytorch_package_outputs(
        onnx_graph=onnx.load(str(string_model_path)),
        package_dir=str(package_dir),
        output_report_path=str(tmp_path / "string_pytorch_accuracy_report.json"),
        num_samples=1,
    )
    assert report["evaluation_pass"] is True
